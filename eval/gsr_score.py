"""Score our CV pipeline with the official GS-HOTA metric on SoccerNet-GSR.

Game State Reconstruction (GSR) is the academic formalisation of our exact problem: a single moving
broadcast camera -> 2D pitch positions + role + team + jersey for every person. The task ships an
official metric, **GS-HOTA** (arXiv:2404.11335), implemented in the SoccerNet ``sn-trackeval`` fork
(``trackeval.datasets.SoccerNetGS`` + the stock ``HOTA``/``Identity`` metrics). We **never**
reimplement the metric: this module only (1) runs our existing pipeline
(:func:`generator.extract.extract_positions`: football-YOLO detect -> ByteTrack/BoT-SORT ->
PnLCalib calibrate -> project; CIELAB-KMeans team) on the GSR frame sequences, (2) converts the
positions table into the GSR prediction JSON the evaluator reads, and (3) drives
``trackeval.SoccerNetGS`` to produce GS-HOTA / GS-DetA / GS-AssA / IDF1.

GSR similarity is a Gaussian on the projected pitch bottom-middle point (sigma from a 5 m distance
tolerance), and an attribute mismatch (role AND team AND jersey) zeroes the similarity. Our pipeline
has **no jersey-number model yet**, so predictions carry ``jersey = null``: under the official
jersey-on config every GT player with a labelled number is therefore unmatchable -- exactly the
pre-Layer-2 baseline this benchmark is meant to expose. We report that honest headline plus
attribute ablations (jersey-off, team-off, role-off) so localization/association capability is
visible separately from the identity gap.

Coordinate convention: our pipeline emits the project's uncentred ``[0,105] x [0,68]`` metres frame;
GSR ground truth is PnLCalib's **centred** ``[-52.5,52.5] x [-34,34]`` frame, so we shift by
``(-PITCH_LEN/2, -PITCH_WID/2)``. Team ``0/1`` from unsupervised KMeans has no inherent left/right
meaning, so per sequence we resolve the two-way label permutation against GT (nearest match within
the tolerance), the standard resolution of an arbitrary cluster labelling -- documented, not tuned.

CLI::

    python -m eval.gsr_score --limit 3                 # pilot: 3 sequences end-to-end
    python -m eval.gsr_score                            # full valid split, resumable
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID

logger = logging.getLogger("gsr_score")

#: Shift from the project's uncentred pitch frame to the GSR centred frame (metres).
CENTRE_SHIFT_X = PITCH_LEN / 2.0
CENTRE_SHIFT_Y = PITCH_WID / 2.0

#: GSR distance tolerance (metres) used both by the metric (sigma) and our team-map resolution.
GSR_DIST_TOL_M = 5.0

#: Penalty-area geometry (FIFA): 16.5 m deep, 40.32 m wide -> half-width 20.16 m.
PENALTY_DEPTH_M = 16.5
PENALTY_HALF_WID_M = 20.16

#: GSR category ids (mirror the dataset's ``categories`` so predictions look native).
_ROLE_CATEGORY_ID = {"player": 1, "goalkeeper": 2, "referee": 3, "other": 7}

#: Default data / output locations (outputs stay lean, under the sanctioned dirs).
DEFAULT_DATA_DIR = Path("data/soccernet/gamestate-2024")
DEFAULT_OUT_DIR = Path("outputs/gsr")
DEFAULT_RESULTS_DIR = Path("results/gsr_benchmark")

#: Attributes :func:`vote_track_attributes` collapses onto their per-track majority when a
#: submission is written. Empty = OFF, the shipped default (v9 W3 registered the component behind
#: this flag; see ``results/GSR_V9_W3.md`` section 4).
VOTE_TRACK_ATTRS: tuple[str, ...] = ()

#: Steps :func:`gk_side_repair` performs when a submission is written. Empty = OFF, the shipped
#: default (v9 W4 registered the component behind this flag; see
#: ``results/gsr_v9_w4_registered.json`` and ``results/gsr_benchmark/gsr_v9_w4_arms.json``).
#: ``"team"`` = a keeper's team is the side he stands in; ``"role"`` = also recover keepers the role
#: model called players; ``"role2"`` / ``"role2dom"`` = the v9 W5 concurrency-guarded replacements
#: for ``"role"`` that can recover a FRAGMENTED keeper (``results/gsr_v9_w5_registered.json``).
#: Composable with :data:`VOTE_TRACK_ATTRS`, applied after it.
GK_SIDE_REPAIR: tuple[str, ...] = ()

#: Configuration of :func:`dedup_absorb`, the v10 W9 duplicate-concurrent-track stage. ``None`` = OFF
#: (the shipped default). Registered in ``results/gsr_v10_w9_registered.json``; thresholds are fit on
#: the v9 factory GSR-TRAIN split only. Keys: ``n_ov`` (minimum shared timesteps), ``d_med`` (maximum
#: median pitch distance over the overlap, metres), ``cos`` (minimum CLIP tracklet-mean cosine, or
#: ``None`` for the geometry-only arm), ``same_team`` (require agreeing majority team), ``embed_dir``
#: (per-detection embedding cache, or ``None``) and ``min_track_rows``.
DEDUP_ABSORB: dict | None = None

#: Rows a track needs before it can take part in a duplicate pair (1 s at 25 fps). Below that the
#: pair label is at the nearest-GT labeller's noise floor and the pair carries no row mass anyway.
DEDUP_MIN_TRACK_ROWS = 25

#: Thresholds of the ``"role"`` step, fit on GSR-TRAIN ground truth ONLY (79 keeper identities /
#: 1,145 player identities over the 57 train sequences): a keeper candidate's median |pitch x| and
#: its share of frames inside a penalty area. At (36 m, 0.90) the "most extreme candidate per side"
#: rule is 67 TP / 1 FP on train GT.
GK_MIN_ABSX_M = 36.0
GK_MIN_PEN_FRAC = 0.90

#: Attribute configs reported by the benchmark (name -> trackeval USE_* flags).
EVAL_CONFIGS: dict[str, dict[str, bool]] = {
    "gs_hota_full": {"USE_ROLES": True, "USE_TEAMS": True, "USE_JERSEY_NUMBERS": True},
    "no_jersey": {"USE_ROLES": True, "USE_TEAMS": True, "USE_JERSEY_NUMBERS": False},
    "role_only": {"USE_ROLES": True, "USE_TEAMS": False, "USE_JERSEY_NUMBERS": False},
    "loc_assoc": {"USE_ROLES": False, "USE_TEAMS": False, "USE_JERSEY_NUMBERS": False},
}


# === Pure adapter seam (tested without the CV / metric stack) =====================================
def to_centred(pitch_x: float, pitch_y: float) -> tuple[float, float]:
    """Convert an uncentred ``[0,105]x[0,68]`` foot point to the GSR centred frame (metres)."""
    return pitch_x - CENTRE_SHIFT_X, pitch_y - CENTRE_SHIFT_Y


def _team_side(team: int, role: str, team_map: dict[int, str]) -> str | None:
    """Map our integer team to a GSR side (``left``/``right``), or ``None`` for non-team roles."""
    if role not in {"player", "goalkeeper"}:
        return None
    return team_map.get(int(team))


def row_to_prediction(row: dict, image_id: str, team_map: dict[int, str]) -> dict | None:
    """Convert one positions row to a GSR prediction dict, or ``None`` if it should be dropped.

    Rows without a trustworthy projected pitch position (NaN, gated out) or the ball are dropped;
    everything else becomes a person prediction with our known attributes (role, team) and a null
    jersey (no jersey model yet).

    Args:
        row: One positions row (keys ``role, team, track_id, pitch_x, pitch_y, image_x, image_y,
            conf``).
        image_id: The GSR ``image_id`` string of this frame.
        team_map: Mapping from our integer team ``{0,1}`` to a GSR side ``{"left","right"}``.

    Returns:
        A GSR prediction dict, or ``None`` when the row carries no usable pitch position.
    """
    role = str(row["role"])
    if role == "ball":
        return None
    px, py = row.get("pitch_x"), row.get("pitch_y")
    if px is None or py is None or not np.isfinite(px) or not np.isfinite(py):
        return None
    cx, cy = to_centred(float(px), float(py))
    track_id = int(row["track_id"])
    side = _team_side(row.get("team", -1), role, team_map)
    conf = float(row.get("conf", 1.0)) if np.isfinite(row.get("conf", 1.0)) else 1.0
    img_x = float(row.get("image_x", 0.0) or 0.0)
    img_y = float(row.get("image_y", 0.0) or 0.0)
    return {
        "id": f"{image_id}{track_id:04d}",
        "image_id": image_id,
        "track_id": track_id,
        "supercategory": "object",
        "category_id": _ROLE_CATEGORY_ID.get(role, 7),
        "attributes": {"role": role, "jersey": None, "team": side},
        "bbox_pitch": {
            "x_bottom_left": cx, "y_bottom_left": cy,
            "x_bottom_middle": cx, "y_bottom_middle": cy,
            "x_bottom_right": cx, "y_bottom_right": cy,
        },
        "bbox_image": {
            "x": img_x - 8, "y": img_y - 40, "x_center": img_x, "y_center": img_y - 20,
            "w": 16, "h": 40,
        },
        "confidence": conf,
    }


def build_submission(
    df: pd.DataFrame, image_id_by_frame: dict[int, str], team_map: dict[int, str]
) -> dict:
    """Build the GSR prediction JSON payload for one sequence from a positions table (pure).

    Args:
        df: Positions table (:data:`generator.extract.POSITIONS_COLUMNS`).
        image_id_by_frame: Maps our processed frame index to the GSR ``image_id`` string.
        team_map: Our integer team -> GSR side mapping for this sequence.

    Returns:
        ``{"predictions": [...]}`` with one entry per usable person detection.
    """
    preds: list[dict] = []
    for row in df.to_dict("records"):
        image_id = image_id_by_frame.get(int(row["frame"]))
        if image_id is None:
            continue
        pred = row_to_prediction(row, image_id, team_map)
        if pred is not None:
            preds.append(pred)
    return {"predictions": preds}


def vote_track_attributes(
    predictions: list[dict], keys: tuple[str, ...] = ("role", "team"),
) -> dict[str, int]:
    """Collapse each track's per-row attributes onto that track's majority value, in place (pure-ish).

    GS-DetA compares ``role``/``team``/``jersey`` **per row**, but a GT identity carries one role and
    one team for the whole clip, so any disagreement inside one of our tracks is guaranteed error on
    its minority rows. Voting can only convert a partly-right track into an all-right or an all-wrong
    one -- it never invents a value the track did not already carry.

    ``None`` is not a candidate: a track that reads ``player`` on some rows and ``goalkeeper`` on
    others carries ``team``/``jersey`` only where the writer allowed it, so the vote runs over the
    non-null values and then applies the winner to every row of the track. Ties keep the first-seen
    value, which makes the result deterministic for a given input order.

    Args:
        predictions: A submission's ``predictions`` list (mutated in place). Ball rows are skipped.
        keys: Which attributes to vote on; ``()`` is a no-op.

    Returns:
        ``{attribute: rows changed}``.
    """
    if not keys:
        return {}
    tracks: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        attrs = p.get("attributes") or {}
        if attrs.get("role") == "ball":
            continue
        tracks[int(p["track_id"])].append(p)
    changed: dict[str, int] = {k: 0 for k in keys}
    for rows in tracks.values():
        for key in keys:
            tally = Counter(str((r["attributes"] or {}).get(key)) for r in rows
                            if (r["attributes"] or {}).get(key) is not None)
            if not tally:
                continue
            value = tally.most_common(1)[0][0]
            for r in rows:
                if r["attributes"].get(key) != value:
                    r["attributes"][key] = value
                    changed[key] += 1
    return changed


def gk_side_repair(
    predictions: list[dict], steps: tuple[str, ...] = ("team",),
) -> dict[str, int]:
    """Fix keeper attributes from geometry instead of kit colour, in place (GT-free).

    GSR's ``left``/``right`` is the half a team **defends**, and a keeper stands in the goal he
    defends -- so his side is the sign of his own pitch x. That definitional rule scores 113/113 on
    development ground truth (``results/GSR_TEAMSIDE.md`` section 3) and 79/79 on GSR-TRAIN, while
    kit clustering puts a keeper in his own team's cluster only ~24% of the time: a GK kit is
    designed to contrast with both outfield kits, so appearance is *anti*-informative here.

    Steps:

    * ``"team"`` -- every track whose majority role is ``goalkeeper`` gets
      ``team = "left" if median x < 0 else "right"``.
    * ``"role"`` -- before that, on each half of the pitch that carries no goalkeeper track at all,
      the ``player`` track with the largest median ``|x|`` among those with
      ``|x| >= GK_MIN_ABSX_M`` and penalty-area share ``>= GK_MIN_PEN_FRAC`` is relabelled
      ``goalkeeper`` (one keeper per half; thresholds fit on GSR-TRAIN only).
    * ``"role2"`` -- the v9 W5 replacement for ``"role"``: the same two thresholds, but the guard is
      **temporal** instead of global. Every keeper-shaped ``player`` track on a half is relabelled
      when it shares no frame with a goalkeeper-majority track on that half, because two tracks that
      never coexist can be fragments of one person and two that do coexist cannot. This is what lets
      a fragmented keeper be recovered when one of his own fragments is already labelled correctly.
    * ``"role2dom"`` -- ``"role2"`` plus: a candidate that *does* coexist with keeper tracks is still
      promoted when it is deeper than every one of them (at most one keeper per half at a time, and
      he is the deepest person on it). Riskier: 2 false positives on GSR-TRAIN GT against 0 for
      ``"role2"``.

    Args:
        predictions: A submission's ``predictions`` list (mutated in place). Ball rows are skipped.
        steps: Which steps to run; ``()`` is a no-op.

    Returns:
        ``{step: rows changed}``.
    """
    if not steps:
        return {}
    tracks: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        attrs = p.get("attributes") or {}
        if attrs.get("role") == "ball":
            continue
        tracks[int(p["track_id"])].append(p)
    info: dict[int, dict] = {}
    for tid, rows in tracks.items():
        x = np.array([r["bbox_pitch"]["x_bottom_middle"] for r in rows], float)
        y = np.array([r["bbox_pitch"]["y_bottom_middle"] for r in rows], float)
        role = Counter(str((r["attributes"] or {}).get("role")) for r in rows).most_common(1)[0][0]
        info[tid] = {"role": role, "med_x": float(np.median(x)),
                     "pen": penalty_frac(x, y), "rows": rows,
                     "frames": {r.get("image_id") for r in rows}}
    changed = {s: 0 for s in steps}
    if "role" in steps:
        for half in (-1.0, 1.0):
            here = [t for t, i in info.items() if np.sign(i["med_x"]) == half]
            if any(info[t]["role"] == "goalkeeper" for t in here):
                continue  # this half already has its one keeper
            cand = [t for t in here if info[t]["role"] == "player"
                    and abs(info[t]["med_x"]) >= GK_MIN_ABSX_M and info[t]["pen"] >= GK_MIN_PEN_FRAC]
            if not cand:
                continue
            best = max(cand, key=lambda t: abs(info[t]["med_x"]))
            info[best]["role"] = "goalkeeper"
            for r in info[best]["rows"]:
                r["attributes"]["role"] = "goalkeeper"
                changed["role"] += 1
    for step in ("role2", "role2dom"):
        if step not in steps:
            continue
        for tid in _second_keepers(info, dominance=step == "role2dom"):
            info[tid]["role"] = "goalkeeper"
            for r in info[tid]["rows"]:
                r["attributes"]["role"] = "goalkeeper"
                changed[step] += 1
    if "team" in steps:
        for i in info.values():
            if i["role"] != "goalkeeper":
                continue
            side = "left" if i["med_x"] < 0 else "right"
            for r in i["rows"]:
                if r["attributes"].get("team") != side:
                    r["attributes"]["team"] = side
                    changed["team"] += 1
    return changed


def _second_keepers(info: dict[int, dict], *, dominance: bool) -> list[int]:
    """Track ids the second-keeper rule promotes to ``goalkeeper`` (pure; see :func:`gk_side_repair`).

    Args:
        info: ``{track_id: {"role", "med_x", "pen", "frames"}}`` for every non-ball track.
        dominance: Also promote a candidate that coexists with keeper tracks but is deeper than all
            of them.

    Returns:
        The promoted track ids, deterministic (sorted by decreasing ``|med_x|``, then by id).
    """
    out: list[int] = []
    for half in (-1.0, 1.0):
        here = {t: i for t, i in info.items() if np.sign(i["med_x"]) == half}
        keepers = [t for t, i in here.items() if i["role"] == "goalkeeper"]
        cand = [t for t, i in here.items() if i["role"] == "player"
                and abs(i["med_x"]) >= GK_MIN_ABSX_M and i["pen"] >= GK_MIN_PEN_FRAC]
        for t in sorted(cand, key=lambda t: (-abs(here[t]["med_x"]), t)):
            conc = [k for k in keepers if here[k]["frames"] & here[t]["frames"]]
            if not conc:
                out.append(t)
            elif dominance and all(abs(here[t]["med_x"]) > abs(here[k]["med_x"]) for k in conc):
                out.append(t)
    return out


def track_mean_embeddings(embed_dir: Path | str | None, seq: str) -> dict[int, np.ndarray]:
    """``{track_id: L2-normalised mean CLIP embedding}`` from a per-detection cache (GT-free).

    The cache is keyed by the pre-connector tracklet id, and the connector keeps one member's id as
    the id of the whole component, so a submission track's cached rows are that member's own
    detections -- the same person. Tracklets the connector absorbed simply never join. A missing
    cache is not an error: the caller falls back to geometry (registration ``B_geom_app``).

    Args:
        embed_dir: Directory of ``<seq>.npz`` files with ``track_ids`` and ``embeddings``.
        seq: Sequence name.

    Returns:
        Mean embedding per cached track id; empty when no cache is available.
    """
    if embed_dir is None:
        return {}
    path = Path(embed_dir) / f"{seq}.npz"
    if not path.exists():
        return {}
    with np.load(path) as z:
        tid, emb = np.asarray(z["track_ids"]), np.asarray(z["embeddings"], dtype=float)
    out: dict[int, np.ndarray] = {}
    for t in np.unique(tid):
        m = emb[tid == t].mean(axis=0)
        out[int(t)] = m / max(float(np.linalg.norm(m)), 1e-9)
    return out


def pair_features(info: dict[int, dict], embs: dict[int, np.ndarray],
                  min_track_rows: int = DEDUP_MIN_TRACK_ROWS) -> list[dict]:
    """One record per temporally overlapping track pair, GT-free (pure).

    Shared by the threshold fit (on the v9 factory GSR-TRAIN tracklets) and by the shipped stage, so
    a fitted threshold means exactly the same thing in both places.

    Args:
        info: ``{track_id: {"pos": {timestep: (x, y)}, "team": majority team, ...}}``.
        embs: ``{track_id: mean embedding}``; absent ids yield ``cos = nan``.
        min_track_rows: Both tracks of a pair need at least this many rows.

    Returns:
        ``[{"a", "b", "n_ov", "d_med", "cos", "same_team"}]``, sorted by ``(a, b)``.
    """
    ids = sorted(t for t, i in info.items() if len(i["pos"]) >= min_track_rows)
    out: list[dict] = []
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            shared = info[a]["pos"].keys() & info[b]["pos"].keys()
            if not shared:
                continue
            pa = np.array([info[a]["pos"][f] for f in sorted(shared)], dtype=float)
            pb = np.array([info[b]["pos"][f] for f in sorted(shared)], dtype=float)
            d = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            ea, eb = embs.get(a), embs.get(b)
            out.append({"a": a, "b": b, "n_ov": int(len(d)), "d_med": float(np.median(d)),
                        "cos": float(ea @ eb) if ea is not None and eb is not None else float("nan"),
                        "same_team": int(info[a]["team"] == info[b]["team"])})
    return out


def accepts_pair(f: dict, params: dict) -> bool:
    """Whether the registered duplicate rule accepts one pair record (pure).

    Geometry is required of every arm; the appearance term binds only when BOTH tracks carry an
    embedding, so a sequence whose embedding cache is missing degrades to the geometry rule instead
    of being waved through (registration ``B_geom_app``).
    """
    if f["n_ov"] < int(params["n_ov"]) or f["d_med"] > float(params["d_med"]):
        return False
    if params.get("same_team") and not f["same_team"]:
        return False
    cos_min = params.get("cos")
    return not (cos_min is not None and np.isfinite(f["cos"]) and f["cos"] < float(cos_min))


def dedup_absorb(predictions: list[dict], params: dict | None = None,
                 seq: str | None = None) -> dict[str, int]:
    """Absorb duplicate concurrent tracks: one identity, loser's overlap dropped (in place).

    kb v10-w7-002 measured that 185 of 265 same-identity track pairs in the current DEV submission
    OVERLAP IN TIME -- two predicted tracks on one person -- and that an oracle which merges them AND
    deletes the duplicated rows is worth +4.50 GS-HOTA, 2.5x the sum of merging alone and deduping
    alone. This is the GT-free version: pairs are detected from geometry (and optionally appearance),
    unioned into components, relabelled to the strongest member's id, and the rows a timestep then
    carries twice are resolved in favour of the strongest member (the official evaluator refuses two
    rows of one id in a timestep).

    "Strongest" is fixed a priori: most rows, then highest mean detection confidence, then lowest
    track id. Ground truth is never read.

    Args:
        predictions: A submission's ``predictions`` list (mutated in place; ball rows are ignored).
        params: :data:`DEDUP_ABSORB`-shaped config; ``None`` or empty is a no-op.
        seq: Sequence name, used only to find this sequence's embedding cache.

    Returns:
        ``{"pairs": accepted pairs, "components": merged groups, "tracks_absorbed": losing tracks,
        "rows_relabelled": rows given a new track id, "rows_dropped": rows deleted}``.
    """
    if not params:
        return {}
    info: dict[int, dict] = defaultdict(lambda: {"pos": {}, "conf": [], "team": [], "idx": []})
    for i, p in enumerate(predictions):
        attrs = p.get("attributes") or {}
        if attrs.get("role") == "ball":
            continue
        bp = p.get("bbox_pitch") or {}
        rec = info[int(p["track_id"])]
        rec["pos"][p["image_id"]] = (float(bp.get("x_bottom_middle", np.nan)),
                                     float(bp.get("y_bottom_middle", np.nan)))
        rec["conf"].append(float(p.get("confidence") or 0.0))
        rec["team"].append(str(attrs.get("team")))
        rec["idx"].append(i)
    for rec in info.values():
        rec["team"] = Counter(rec["team"]).most_common(1)[0][0]
        rec["rank"] = (-len(rec["idx"]), -float(np.mean(rec["conf"])))
    embs = track_mean_embeddings(params.get("embed_dir"), seq) if seq else {}
    feats = pair_features(info, embs, int(params.get("min_track_rows", DEDUP_MIN_TRACK_ROWS)))
    pairs = [(f["a"], f["b"]) for f in feats if accepts_pair(f, params)]

    parent = {t: t for t in info}

    def find(t: int) -> int:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:  # the winner of the two roots anchors the component
            lo, hi = sorted((ra, rb), key=lambda t: (*info[t]["rank"], t))
            parent[hi] = lo
    members: dict[int, list[int]] = defaultdict(list)
    for t in info:
        members[find(t)].append(t)
    changed = {"pairs": len(pairs), "components": 0, "tracks_absorbed": 0,
               "rows_relabelled": 0, "rows_dropped": 0}
    drop: set[int] = set()
    for group in members.values():
        if len(group) < 2:
            continue
        changed["components"] += 1
        changed["tracks_absorbed"] += len(group) - 1
        order = sorted(group, key=lambda t: (*info[t]["rank"], t))
        winner, seen = order[0], set()
        for t in order:
            for i in info[t]["idx"]:
                stamp = predictions[i]["image_id"]
                if stamp in seen:
                    drop.add(i)
                    continue
                seen.add(stamp)
                if t != winner:
                    predictions[i]["track_id"] = int(winner)
                    changed["rows_relabelled"] += 1
    changed["rows_dropped"] = len(drop)
    if drop:
        predictions[:] = [p for i, p in enumerate(predictions) if i not in drop]
    return changed


def penalty_frac(x: np.ndarray, y: np.ndarray) -> float:
    """Share of centred-frame pitch points inside either penalty area (pure).

    The GSR pitch frame is centred, so a penalty area is ``|x| >= PITCH_LEN/2 - 16.5`` and
    ``|y| <= 20.16`` (the FIFA 16.5 m x 40.32 m box).
    """
    if len(x) == 0:
        return 0.0
    inside = (np.abs(x) >= CENTRE_SHIFT_X - PENALTY_DEPTH_M) & (np.abs(y) <= PENALTY_HALF_WID_M)
    return float(np.mean(inside))


# === GT helpers ==================================================================================
def load_image_id_map(seq_dir: Path) -> dict[int, str]:
    """Map processed frame index (``000001.jpg`` -> 0) to the GSR ``image_id`` for a sequence."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for img in gt["images"]:
        idx = int(Path(img["file_name"]).stem) - 1  # 000001.jpg -> frame index 0
        out[idx] = img["image_id"]
    return out


def load_gt_people_by_frame(seq_dir: Path) -> dict[int, list[tuple[float, float, str, str]]]:
    """Load GT player/GK positions per frame index as ``(cx, cy, role, team_side)`` (centred m)."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    id_to_idx = {img["image_id"]: int(Path(img["file_name"]).stem) - 1 for img in gt["images"]}
    out: dict[int, list[tuple[float, float, str, str]]] = defaultdict(list)
    for ann in gt["annotations"]:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        role = attrs.get("role")
        if role not in {"player", "goalkeeper"}:
            continue
        bp = ann.get("bbox_pitch")
        if not bp:
            continue
        idx = id_to_idx.get(ann["image_id"])
        if idx is None:
            continue
        out[idx].append((bp["x_bottom_middle"], bp["y_bottom_middle"], role, attrs.get("team")))
    return out


def resolve_team_map(
    df: pd.DataFrame, gt_people: dict[int, list[tuple[float, float, str, str]]]
) -> dict[int, str]:
    """Resolve our team ``{0,1}`` -> GSR side by agreement with GT among position-matched people.

    For every frame each of our player/GK detections is matched to the nearest GT player/GK within
    :data:`GSR_DIST_TOL_M`; the two-way label permutation that maximises team agreement over all
    matches is returned. This is the standard resolution of an arbitrary unsupervised cluster
    labelling -- disclosed, not fitted to the metric.

    Returns:
        ``{0: side0, 1: side1}`` with the two sides distinct; defaults to ``{0:'left',1:'right'}``
        when there is no matched evidence.
    """
    # counts[our_team][gt_side] = number of matched detections
    counts: dict[int, dict[str, int]] = {0: defaultdict(int), 1: defaultdict(int)}
    people = df[df["role"].isin(["player", "goalkeeper"])]
    for frame, grp in people.groupby("frame"):
        gts = gt_people.get(int(frame), [])
        if not gts:
            continue
        gt_xy = np.array([[g[0], g[1]] for g in gts])
        for r in grp.itertuples():
            if not (np.isfinite(r.pitch_x) and np.isfinite(r.pitch_y)):
                continue
            cx, cy = to_centred(float(r.pitch_x), float(r.pitch_y))
            d = np.hypot(gt_xy[:, 0] - cx, gt_xy[:, 1] - cy)
            j = int(np.argmin(d))
            if d[j] > GSR_DIST_TOL_M:
                continue
            gt_side = gts[j][3]
            our_team = int(r.team)
            if our_team in counts and gt_side in {"left", "right"}:
                counts[our_team][gt_side] += 1
    direct = counts[0]["left"] + counts[1]["right"]
    swapped = counts[0]["right"] + counts[1]["left"]
    if swapped > direct:
        return {0: "right", 1: "left"}
    return {0: "left", 1: "right"}


def resolve_team_map_free(df: pd.DataFrame) -> tuple[dict[int, str], float]:
    """Resolve our team ``{0,1}`` -> GSR side from the geometry alone (no ground truth).

    The GT-free replacement for :func:`resolve_team_map`. A team defends one goal, so over a clip
    its outfield players sit *behind* the opponent's: whichever way the ball is running, the side
    being attacked has its defensive line deepest. The cluster with the smaller mean centred
    ``pitch_x`` over all player rows is therefore ``left``. Goalkeepers are excluded -- their kit is
    a third colour, so the two-way kit clustering assigns them essentially at random and their
    extreme x would swamp the mean (measured: a GK-only signal scores 2/12, worse than chance).

    Args:
        df: Positions table for one sequence (needs ``role``, ``team``, ``pitch_x``).

    Returns:
        ``({0: side0, 1: side1}, margin)`` where ``margin`` is the separation of the two cluster
        means in metres -- the confidence dial. Falls back to ``{0:'left',1:'right'}`` with margin
        ``0.0`` when a cluster has no usable player row.
    """
    d = df[(df["role"] == "player") & df["team"].isin([0, 1])]
    d = d[np.isfinite(d["pitch_x"])]
    means = d.groupby("team")["pitch_x"].mean()
    if 0 not in means.index or 1 not in means.index:
        return {0: "left", 1: "right"}, 0.0
    m0, m1 = float(means[0]), float(means[1])
    order = {0: "left", 1: "right"} if m0 <= m1 else {0: "right", 1: "left"}
    return order, abs(m1 - m0)


# === Pipeline extraction (GPU stage; resumable) ==================================================
def extract_sequence(
    seq_dir: Path, out_parquet: Path, calibrator, *, calib_period: int, detector: str,
    tracker: str, force: bool = False,
) -> pd.DataFrame:
    """Run our CV pipeline over a GSR frame sequence, writing/reusing a positions parquet.

    Frames are fed to :func:`generator.extract.extract_positions` through OpenCV's native
    image-sequence reader (``img1/%06d.jpg``); ``sample_every=1`` so every GT frame gets a
    prediction (needed for a fair GS-DetA). ``force`` overwrites an existing parquet (used to repair
    a stale/NaN-only cache).
    """
    if out_parquet.exists() and not force:
        return pd.read_parquet(out_parquet)
    from generator.extract import extract_positions  # noqa: PLC0415

    pattern = str(seq_dir / "img1" / "%06d.jpg")
    df = extract_positions(
        pattern, out_parquet, sample_every=1, calibrator=calibrator,
        detector_name=detector, tracker_name=tracker, calib_period=calib_period,
    )
    return df


# === Metric driver (official trackeval; never reimplemented) =====================================
def _run_trackeval(
    gt_dir: Path, trackers_dir: Path, tracker: str, seq_info: dict[str, int], cfg: dict[str, bool]
) -> dict:
    """Drive ``trackeval.SoccerNetGS`` over ``seq_info`` and return the raw ``output_res`` tree."""
    import trackeval  # noqa: PLC0415

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False, "OUTPUT_EMPTY_CLASSES": False,
    })
    dataset_config = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trackers_dir),
        "TRACKERS_TO_EVAL": [tracker], "TRACKER_SUB_FOLDER": "data",
        "SKIP_SPLIT_FOL": True, "SEQ_INFO": seq_info, "PRINT_CONFIG": False,
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/Labels-GameState.json",
        "EVAL_MODE": "distance", "EVAL_SPACE": "pitch", "EVAL_SIMILARITY_METRIC": "gaussian",
        "EVAL_DIST_TOL": GSR_DIST_TOL_M, **cfg,
    })
    metrics_config = {"METRICS": ["HOTA", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.SoccerNetGS(dataset_config)
    metrics = [m(metrics_config) for m in (trackeval.metrics.HOTA, trackeval.metrics.Identity)]
    output_res, _ = evaluator.evaluate([dataset], metrics)
    return output_res["SoccerNetGS"][tracker]


def _scalar(node: dict, metric: str, field: str) -> float:
    """Mean of a (possibly per-alpha) metric field, as a percentage."""
    val = node[metric][field]
    return float(np.mean(val)) * 100.0


def summarise_res(res: dict, seq_list: list[str]) -> dict:
    """Extract combined + per-sequence GS-HOTA/DetA/AssA/IDF1 (percentages) from ``output_res``."""
    def pack(node: dict) -> dict:
        return {
            "GS-HOTA": _scalar(node, "HOTA", "HOTA"),
            "GS-DetA": _scalar(node, "HOTA", "DetA"),
            "GS-AssA": _scalar(node, "HOTA", "AssA"),
            "GS-LocA": _scalar(node, "HOTA", "LocA"),
            "IDF1": _scalar(node, "Identity", "IDF1"),
        }

    combined = pack(res["COMBINED_SEQ"]["person"])
    per_seq = {seq: pack(res[seq]["person"]) for seq in seq_list if seq in res}
    return {"combined": combined, "per_seq": per_seq}


def gs_hota(predictions: str | Path, ground_truth: str | Path, *, split: str = "valid",
            seq_info: dict[str, int] | None = None, **cfg: bool) -> dict:
    """Return GS-HOTA (and DetA/AssA/LocA/IDF1) for a folder of GSR predictions (official metric).

    The real entry point replacing the former stub. ``predictions`` is a trackers folder laid out as
    ``<predictions>/<tracker>/data/<seq>.json``; ``ground_truth`` holds ``<seq>/Labels-GameState``.

    Args:
        predictions: Trackers folder (contains a single ``predictions`` tracker subfolder).
        ground_truth: GSR ground-truth folder (sequence dirs with ``Labels-GameState.json``).
        split: Split label (metadata only; sequences are chosen by ``seq_info``).
        seq_info: ``{seq_name: n_frames}`` to evaluate; defaults to every sequence found.
        **cfg: ``USE_ROLES`` / ``USE_TEAMS`` / ``USE_JERSEY_NUMBERS`` overrides.

    Returns:
        ``{"combined": {...}, "per_seq": {...}}`` of metric percentages.
    """
    gt_dir, pred_dir = Path(ground_truth), Path(predictions)
    if seq_info is None:
        data = pred_dir / "predictions" / "data"
        seq_info = {p.stem: 0 for p in sorted(data.glob("*.json"))}
    res = _run_trackeval(gt_dir, pred_dir, "predictions", seq_info, cfg)
    return summarise_res(res, list(seq_info.keys()))


# === Orchestration ================================================================================
def _write_submission(df: pd.DataFrame, seq_dir: Path, dest: Path) -> dict[int, str]:
    """Resolve the team map, build and write one sequence's prediction JSON. Returns the team map."""
    image_ids = load_image_id_map(seq_dir)
    gt_people = load_gt_people_by_frame(seq_dir)
    team_map = resolve_team_map(df, gt_people)
    submission = build_submission(df, image_ids, team_map)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(submission), encoding="utf-8")
    return team_map


def run_benchmark(
    data_dir: Path, out_dir: Path, results_dir: Path, *, limit: int | None, calib_period: int,
    detector: str, tracker: str, skip_extract: bool = False, only: list[str] | None = None,
) -> dict:
    """Run the full GSR benchmark: extract -> submissions -> multi-config GS-HOTA + write results.

    Resumable: per-sequence positions parquets and prediction JSONs are reused when present, so an
    interrupted run continues where it stopped.

    Args:
        only: Restrict to these sequence names. Needed since ``data/soccernet/gamestate-2024``
            holds train + valid + test in one flat directory (``README_SPLITS.md``); the downstream
            drivers self-filter on the presence of a positions parquet, this one does not.
    """
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (p / "Labels-GameState.json").exists())
    if only:
        keep = set(only)
        seqs = [p for p in seqs if p.name in keep]
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        raise SystemExit(f"no GSR sequences under {data_dir}")
    logger.info("GSR benchmark: %d sequences, calib_period=%d, det=%s track=%s",
                len(seqs), calib_period, detector, tracker)

    pos_dir = out_dir / "positions"
    pred_data_dir = out_dir / "eval" / "predictions" / "data"
    pred_data_dir.mkdir(parents=True, exist_ok=True)
    team_maps: dict[str, dict[int, str]] = {}

    calibrator = None
    if not skip_extract:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        calibrator = PnLCalibCalibrator()  # built once, reused across sequences

    seq_info: dict[str, int] = {}
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        seq_info[name] = len(load_image_id_map(seq_dir))
        pred_json = pred_data_dir / f"{name}.json"
        if pred_json.exists() and (pos_dir / f"{name}.parquet").exists():
            logger.info("[%d/%d] %s: reuse existing submission", i + 1, len(seqs), name)
            continue
        if skip_extract:  # score-only: never run the GPU stage, just skip unprocessed sequences
            logger.info("[%d/%d] %s: no submission, skipping (score-only)", i + 1, len(seqs), name)
            continue
        t0 = time.time()
        df = extract_sequence(seq_dir, pos_dir / f"{name}.parquet", calibrator,
                              calib_period=calib_period, detector=detector, tracker=tracker)
        team_map = _write_submission(df, seq_dir, pred_json)
        team_maps[name] = team_map
        valid = int(df.dropna(subset=["pitch_x"]).shape[0]) if len(df) else 0
        logger.info("[%d/%d] %s: %d rows (%d w/ pitch), team_map=%s (%.0fs)",
                    i + 1, len(seqs), name, len(df), valid, team_map, time.time() - t0)

    # Score every attribute configuration over the sequences that actually have a submission
    # (keeps a partial/resumed run scorable instead of erroring on a missing tracker file).
    scored = {name: n for name, n in seq_info.items() if (pred_data_dir / f"{name}.json").exists()}
    results: dict[str, dict] = {}
    for cfg_name, cfg in EVAL_CONFIGS.items():
        logger.info("scoring config '%s' (%s) over %d sequences", cfg_name, cfg, len(scored))
        results[cfg_name] = gs_hota(out_dir / "eval", data_dir, seq_info=scored, **cfg)

    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_sequences": len(scored), "n_target": len(seqs), "calib_period": calib_period,
        "detector": detector, "tracker": tracker, "dist_tol_m": GSR_DIST_TOL_M,
        "team_maps": team_maps, "configs": results,
    }
    (results_dir / "gsr_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for cfg_name, r in results.items():
        c = r["combined"]
        logger.info("== %-12s GS-HOTA %.2f  DetA %.2f  AssA %.2f  LocA %.2f  IDF1 %.2f",
                    cfg_name, c["GS-HOTA"], c["GS-DetA"], c["GS-AssA"], c["GS-LocA"], c["IDF1"])
    return payload


# === Stage 2a: post-hoc track re-linking (appearance ReID) =======================================
#: Pilot sequences the re-link threshold is tuned on, then FROZEN before touching the other 55.
PILOT_SEQS = ("SNGS-021", "SNGS-022", "SNGS-023")


def _relink_and_write(
    seqs: list[Path], data_dir: Path, out_dir: Path, pos_dir: Path, pred_out: Path, *,
    params, audit_seqs: set[str], build_tools: bool,
) -> dict[str, dict]:
    """Relink every sequence's fragments and write relinked submissions -> per-seq stats.

    Embeddings are cached under ``out_dir/relink_cache`` (resumable); the OSNet embedder + football
    detector are built once, and only if some cache is cold (``build_tools``).
    """
    from generator.track_relink import OsnetEmbedder, relink_sequence  # noqa: PLC0415

    cache_dir = out_dir / "relink_cache"
    embedder = detector = None
    if build_tools:
        import torch  # noqa: PLC0415

        from generator.extract import _build_detector  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = OsnetEmbedder(params.model_name, device=device, weights=params.weights)
        detector = _build_detector(device, "football")
    pred_out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        parquet = pos_dir / f"{name}.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        relinked, st = relink_sequence(
            seq_dir, df, params=params, embedder=embedder, detector=detector,
            cache_path=cache_dir / f"{name}.npz", audit=name in audit_seqs)
        _write_submission(relinked, seq_dir, pred_out / f"{name}.json")
        stats[name] = st
        logger.info("[%d/%d] %s: frags %d -> %d (%d merges, %d embedded, %.1f crops/frag)%s",
                    i + 1, len(seqs), name, st["n_fragments_before"], st["n_fragments_after"],
                    st["n_merges"], st["n_embedded"], st["mean_crops"],
                    (f", merge-prec {st.get('merge_precision_correct')}/"
                     f"{st.get('merge_precision_total')}" if name in audit_seqs else ""))
    return stats


def _caches_cold(seqs: list[Path], out_dir: Path) -> bool:
    """True if any sequence lacks a cached embedding (so the GPU tools must be built)."""
    cache_dir = out_dir / "relink_cache"
    return any(not (cache_dir / f"{p.name}.npz").exists() for p in seqs)


def run_relink_benchmark(
    data_dir: Path, out_dir: Path, results_dir: Path, *, limit: int | None, threshold: float,
) -> dict:
    """Stage 2a: relink fragments, re-score the SAME sequences, write ``gsr_scores_relink.json``.

    Never overwrites the baseline submissions or ``gsr_scores.json``: relinked submissions go to
    ``out_dir/eval_relink`` and scores to ``results_dir/gsr_scores_relink.json``.
    """
    from generator.track_relink import RelinkParams  # noqa: PLC0415

    pos_dir = out_dir / "positions"
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (pos_dir / f"{p.name}.parquet").exists())
    if limit:
        seqs = seqs[:limit]
    params = RelinkParams(threshold=threshold)
    logger.info("relink benchmark: %d sequences, threshold=%.3f", len(seqs), threshold)
    pred_out = out_dir / "eval_relink"
    stats = _relink_and_write(
        seqs, data_dir, out_dir, pos_dir, pred_out / "predictions" / "data",
        params=params, audit_seqs=set(PILOT_SEQS), build_tools=_caches_cold(seqs, out_dir))

    scored = {p.name: 0 for p in seqs if (pred_out / "predictions" / "data" / f"{p.name}.json").exists()}
    after: dict[str, dict] = {}
    for cfg_name, cfg in EVAL_CONFIGS.items():
        logger.info("scoring relinked config '%s' over %d sequences", cfg_name, len(scored))
        after[cfg_name] = gs_hota(pred_out, data_dir, seq_info=scored, **cfg)

    baseline = json.loads((results_dir / "gsr_scores.json").read_text(encoding="utf-8"))["configs"]
    frags_before = float(np.mean([s["n_fragments_before"] for s in stats.values()]))
    frags_after = float(np.mean([s["n_fragments_after"] for s in stats.values()]))
    pilot_prec = _pilot_precision(stats)
    payload = {
        "n_sequences": len(scored), "threshold": threshold, "params": vars(params),
        "mean_fragments_before": frags_before, "mean_fragments_after": frags_after,
        "pilot_merge_precision": pilot_prec, "per_seq_stats": stats,
        "before": {k: v["combined"] for k, v in baseline.items()},
        "after": {k: v["combined"] for k, v in after.items()},
        "after_per_seq": {k: v["per_seq"] for k, v in after.items()},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "gsr_scores_relink.json").write_text(json.dumps(payload, indent=2),
                                                        encoding="utf-8")
    for cfg_name in EVAL_CONFIGS:
        b, a = baseline[cfg_name]["combined"], after[cfg_name]["combined"]
        logger.info("== %-12s AssA %.2f -> %.2f (%+.2f)  HOTA %.2f -> %.2f  DetA %.2f -> %.2f",
                    cfg_name, b["GS-AssA"], a["GS-AssA"], a["GS-AssA"] - b["GS-AssA"],
                    b["GS-HOTA"], a["GS-HOTA"], b["GS-DetA"], a["GS-DetA"])
    logger.info("fragments/seq %.0f -> %.0f; pilot merge precision %d/%d = %.1f%%",
                frags_before, frags_after, pilot_prec[0], pilot_prec[1],
                100.0 * pilot_prec[0] / max(pilot_prec[1], 1))
    return payload


def _pilot_precision(stats: dict[str, dict]) -> tuple[int, int]:
    """Sum (correct, total) auditable merge pairs over the pilot sequences."""
    c = sum(stats[s].get("merge_precision_correct", 0) for s in PILOT_SEQS if s in stats)
    t = sum(stats[s].get("merge_precision_total", 0) for s in PILOT_SEQS if s in stats)
    return c, t


def tune_relink_threshold(
    data_dir: Path, out_dir: Path, thresholds: list[float],
) -> None:
    """Sweep re-link thresholds on the 3 pilot sequences: report loc_assoc AssA + merge precision.

    Embeddings are computed once (cached), so the sweep is CPU-only. Prints a table to pick and
    freeze the threshold before applying it to the full split.
    """
    from generator.track_relink import RelinkParams  # noqa: PLC0415

    pos_dir = out_dir / "positions"
    seqs = [data_dir / s for s in PILOT_SEQS]
    tune_out = out_dir / "eval_relink_tune"
    logger.info("tuning on pilot %s over thresholds %s", PILOT_SEQS, thresholds)
    print(f"{'thresh':>7} {'AssA':>7} {'HOTA_la':>8} {'frags':>12} {'merges':>7} {'merge_prec':>11}")
    for th in thresholds:
        params = RelinkParams(threshold=th)
        pred_out = tune_out / f"th_{th:.2f}"
        stats = _relink_and_write(
            seqs, data_dir, out_dir, pos_dir, pred_out / "predictions" / "data",
            params=params, audit_seqs=set(PILOT_SEQS), build_tools=_caches_cold(seqs, out_dir))
        scored = {s: 0 for s in PILOT_SEQS}
        r = gs_hota(pred_out, data_dir, seq_info=scored, **EVAL_CONFIGS["loc_assoc"])
        c, t = _pilot_precision(stats)
        fb = sum(stats[s]["n_fragments_before"] for s in PILOT_SEQS if s in stats)
        fa = sum(stats[s]["n_fragments_after"] for s in PILOT_SEQS if s in stats)
        prec = f"{c}/{t}={100.0 * c / max(t, 1):.0f}%"
        print(f"{th:>7.2f} {r['combined']['GS-AssA']:>7.2f} {r['combined']['GS-HOTA']:>8.2f} "
              f"{fb:>5} ->{fa:>5} {fb - fa:>7} {prec:>11}")


def main() -> None:
    """CLI entry point: extract our pipeline over GSR sequences and score GS-HOTA."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N sequences")
    ap.add_argument("--seqs", default=None,
                    help="comma-separated sequence names (the data dir holds every split)")
    ap.add_argument("--calib-period", type=int, default=1, help="PnLCalib every N frames (1=per-frame)")
    ap.add_argument("--detector", default="football")
    ap.add_argument("--tracker", default="bytetrack")
    ap.add_argument("--score-only", action="store_true", help="reuse submissions; skip the GPU stage")
    ap.add_argument("--relink", action="store_true",
                    help="Stage 2a: merge fragments by appearance, re-score into gsr_scores_relink.json")
    ap.add_argument("--relink-threshold", type=float, default=0.80,
                    help="frozen cosine merge threshold (tuned on the 3 pilot sequences)")
    ap.add_argument("--tune-relink", type=str, default=None,
                    help="comma-separated thresholds to sweep on the pilot (prints AssA + precision)")
    args = ap.parse_args()
    if args.tune_relink:
        tune_relink_threshold(args.data_dir, args.out_dir,
                              [float(t) for t in args.tune_relink.split(",")])
        return
    if args.relink:
        run_relink_benchmark(args.data_dir, args.out_dir, args.results_dir,
                             limit=args.limit, threshold=args.relink_threshold)
        return
    run_benchmark(
        args.data_dir, args.out_dir, args.results_dir, limit=args.limit,
        calib_period=args.calib_period, detector=args.detector, tracker=args.tracker,
        skip_extract=args.score_only, only=args.seqs.split(",") if args.seqs else None,
    )


if __name__ == "__main__":
    main()
