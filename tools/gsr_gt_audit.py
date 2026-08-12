"""Audit the SoccerNet-GSR ground-truth ``team`` attribute on the sequences we lose to side flips.

The WACV 2026 Broadcast2Pitch paper states that test samples SNGS-126, SNGS-131 and SNGS-197 "have
incorrect team annotations in the SoccerNet-GSR test set". Our own audited flip set
(``results/gsr_benchmark/gsr_v6_legitimacy_audit.json``) is SNGS-126, 130, 131, 197 -- three of the
four names match. This module tests the claim locally, with two instruments that need no human eye:

1. **Geometry vs the label** (``oracle``). The GT ``left``/``right`` attribute is re-coded as clusters
   0/1 and every side rule of :mod:`tools.gsr_teamside` is run on the ground-truth positions. The
   definitional rule ``gk_self`` -- "a keeper's own team defends the goal he stands in" -- is 113/113
   on the dev split, so a clip where ``gk_self`` points the wrong way is a clip whose GT label
   contradicts the GT pitch coordinates. ``meanx`` (the shipped rule) is reported beside it: a clip
   where ``meanx`` inverts but ``gk_self`` does not is a *framing* artefact, not a GT error.
2. **Kit coherence vs the label** (``kit``). Player crops are cut from the frames, summarised by
   :func:`generator.teams.jersey_color` (median CIELAB of non-grass torso pixels) and clustered with
   KMeans(2). If the two GT ``team`` groups are two coherent kits, the clustering agrees with the GT
   split; a low agreement rate on a clip whose kits are separable means individual players carry the
   wrong ``team`` value.

Contact sheets (one PNG per sequence, crops labelled with GT team / jersey / track) are written so a
human can confirm the automated verdict.

CLI::

    python -m tools.gsr_gt_audit --run              # oracle + kit + contact sheets, default pool
    python -m tools.gsr_gt_audit --run --seqs SNGS-126 SNGS-116
    python -m tools.gsr_gt_audit --demo             # runnable self-check, no data needed
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import CENTRE_SHIFT_X, DEFAULT_DATA_DIR
from tools.gsr_teamside import gt_positions, side_scores

logger = logging.getLogger("gsr_gt_audit")

#: The four sequences our v6 package flips (gsr_v6_legitimacy_audit.json).
FLIP_SET = ("SNGS-126", "SNGS-130", "SNGS-131", "SNGS-197")
#: Controls: a healthy test clip, the v4/v5-era flip that v6 fixed, and the GT-oracle-only inversion.
CONTROLS = ("SNGS-116", "SNGS-129", "SNGS-195")
#: Dev clips already known to be GT-oracle inversions (results/GSR_TEAMSIDE.md section 3).
DEV_REF = ("SNGS-038", "SNGS-092", "SNGS-111")
#: Where the audit lands.
OUT_DIR = Path("results/gsr_benchmark/gt_audit")
#: Crop sampling for the kit check: every Nth frame, boxes at least this tall.
KIT_FRAME_STEP = 15
MIN_BOX_H = 40
MIN_BOX_W = 16
#: Contact-sheet geometry.
CELL_W, CELL_H, LABEL_H, PER_TEAM = 84, 168, 34, 10


# === Ground-truth reading ========================================================================
def gt_boxes(seq_dir: Path) -> pd.DataFrame:
    """Player/keeper image boxes with their GT team, jersey, track and pitch-x.

    Args:
        seq_dir: A sequence directory holding ``Labels-GameState.json`` and ``img1/``.

    Returns:
        One row per annotated person box, with ``frame`` (1-based image stem), ``file``, ``track``,
        ``role``, ``team`` (``left``/``right``/``None``), ``jersey``, ``x``/``y``/``w``/``h`` and the
        centred ``pitch_x``.
    """
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    files = {i["image_id"]: i["file_name"] for i in gt["images"]}
    rows = []
    for a in gt["annotations"]:
        at = a.get("attributes") or {}
        if at.get("role") not in ("player", "goalkeeper"):
            continue
        b, bp = a.get("bbox_image"), a.get("bbox_pitch")
        if not b or a["image_id"] not in files:
            continue
        f = files[a["image_id"]]
        rows.append({
            "frame": int(Path(f).stem), "file": f, "track": int(a["track_id"]),
            "role": at["role"], "team": at.get("team"), "jersey": at.get("jersey"),
            "x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"]),
            "pitch_x": float(bp["x_bottom_middle"]) if bp else float("nan"),
        })
    return pd.DataFrame(rows)


# === Instrument 1: geometry vs the label =========================================================
def oracle(seq_dir: Path) -> dict:
    """Every side rule on the GT positions of one sequence, whole clip and in thirds.

    Sign convention of :func:`tools.gsr_teamside.side_scores` with GT re-coded (``left`` -> cluster
    0): **negative is correct**. A positive ``gk_self`` means the keeper labelled ``left`` stands in
    the right-hand goal -- the label contradicts its own pitch coordinates.

    Args:
        seq_dir: Sequence directory.

    Returns:
        Whole-clip rule scores, the same scores per time third, and the GT keeper positions.
    """
    df = gt_positions(seq_dir)
    whole = side_scores(df)
    f0, f1 = int(df["frame"].min()), int(df["frame"].max())
    thirds = []
    for w in range(3):
        lo = f0 + (f1 - f0 + 1) * w / 3
        hi = f0 + (f1 - f0 + 1) * (w + 1) / 3
        sub = df[(df["frame"] >= lo) & (df["frame"] < hi)]
        s = side_scores(sub)
        thirds.append({"third": w, **{k: s.get(k) for k in
                                      ("meanx", "gk_self", "n_rows", "n_gk_frames")}})
    gk = df[(df["role"] == "goalkeeper") & df["team"].isin([0, 1])]
    keepers = [{"gt_team": ["left", "right"][int(t)], "n_frames": int(len(g)),
                "mean_pitch_x_centred": round(float(g["pitch_x"].mean() - CENTRE_SHIFT_X), 2)}
               for t, g in gk.groupby("team")]
    pl = df[(df["role"] == "player") & df["team"].isin([0, 1])]
    return {
        "rules": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                      else round(float(v), 4)) for k, v in whole.items()},
        "thirds": thirds, "keepers": keepers,
        "n_player_rows": {"left": int((pl["team"] == 0).sum()), "right": int((pl["team"] == 1).sum())},
    }


# === Instrument 2: kit coherence vs the label ====================================================
def kit_check(seq_dir: Path, *, frame_step: int = KIT_FRAME_STEP) -> dict:
    """Cluster sampled player crops by CIELAB kit colour and compare with the GT team split.

    Args:
        seq_dir: Sequence directory.
        frame_step: Sample every Nth frame.

    Returns:
        Crop-level and track-level agreement between KMeans(2) on kit colour and the GT ``team``
        attribute (best of the two cluster -> team permutations), the cluster centroids, and the
        tracks whose modal kit cluster disagrees with their labelled team-mates.
    """
    import cv2  # noqa: PLC0415
    from sklearn.cluster import KMeans  # noqa: PLC0415

    from generator.teams import jersey_color  # noqa: PLC0415

    df = gt_boxes(seq_dir)
    df = df[(df["role"] == "player") & df["team"].isin(["left", "right"])
            & (df["h"] >= MIN_BOX_H) & (df["w"] >= MIN_BOX_W)
            & (df["frame"] % frame_step == 1)]
    if df.empty:
        return {"n_crops": 0}
    cols, keep = [], []
    for f, g in df.groupby("file"):
        img = cv2.imread(str(seq_dir / "img1" / f))
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in g.itertuples():
            crop = img[max(r.y, 0):min(r.y + r.h, h), max(r.x, 0):min(r.x + r.w, w)]
            if crop.size == 0:
                continue
            cols.append(jersey_color(crop))
            keep.append(r)
    if len(cols) < 8:
        return {"n_crops": len(cols)}
    x = np.stack(cols)
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x)
    cl, gtside = lab.labels_, np.array([r.team for r in keep])
    acc = max(float(((cl == c) == (gtside == "left")).mean()) for c in (0, 1))
    per_track: dict[int, Counter] = defaultdict(Counter)
    track_team = {}
    for r, c in zip(keep, cl, strict=True):
        per_track[r.track][int(c)] += 1
        track_team[r.track] = r.team
    modal = {t: cnt.most_common(1)[0][0] for t, cnt in per_track.items()}
    tacc = max(float(np.mean([(modal[t] == c) == (track_team[t] == "left") for t in modal]))
               for c in (0, 1))
    # which permutation won at track level, so disagreeing tracks can be named
    best_c = max((0, 1), key=lambda c: np.mean([(modal[t] == c) == (track_team[t] == "left")
                                                for t in modal]))
    odd = [{"track": int(t), "gt_team": track_team[t],
            "kit_cluster_purity": round(per_track[t].most_common(1)[0][1] / sum(per_track[t].values()), 3),
            "n_crops": int(sum(per_track[t].values()))}
           for t in sorted(modal) if (modal[t] == best_c) != (track_team[t] == "left")]
    d = np.linalg.norm(lab.cluster_centers_[0] - lab.cluster_centers_[1])
    within = float(np.mean(np.linalg.norm(x - lab.cluster_centers_[cl], axis=1)))
    return {
        "n_crops": int(len(cl)), "n_tracks": len(modal),
        "crop_agreement": round(acc, 4), "track_agreement": round(tacc, 4),
        "centroids_Lab": [[round(float(v), 1) for v in c] for c in lab.cluster_centers_],
        "centroid_sep": round(float(d), 2), "mean_within_dist": round(within, 2),
        "separability": round(float(d / max(within, 1e-6)), 2),
        "disagreeing_tracks": odd,
    }


# === Instrument 3: cross-clip consistency within one game half ===================================
def kit_signature(seq_dir: Path, *, frame_step: int = 75) -> dict:
    """Mean kit colour of each GT side, plus the game and half the clip belongs to.

    Teams swap ends at half time and not within a half, so two clips of the same ``game_id`` and the
    same half must map the same kit to the same side. This is a consistency test the per-clip
    instruments cannot see: a lone clip that maps the kits the other way round contradicts its own
    siblings even though it is internally coherent.

    Args:
        seq_dir: Sequence directory.
        frame_step: Sample every Nth frame (a signature needs far fewer crops than a clustering).

    Returns:
        ``game``, ``half``, ``clock``, mean CIELAB per GT side and ``warm_side`` -- the side wearing
        the redder kit (higher ``a*``), which is the cross-clip handle.
    """
    import cv2  # noqa: PLC0415

    from generator.teams import jersey_color  # noqa: PLC0415

    info = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))["info"]
    df = gt_boxes(seq_dir)
    df = df[(df["role"] == "player") & df["team"].isin(["left", "right"])
            & (df["h"] >= MIN_BOX_H) & (df["w"] >= MIN_BOX_W) & (df["frame"] % frame_step == 1)]
    cols: dict[str, list] = {"left": [], "right": []}
    for f, g in df.groupby("file"):
        img = cv2.imread(str(seq_dir / "img1" / f))
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in g.itertuples():
            crop = img[max(r.y, 0):min(r.y + r.h, h), max(r.x, 0):min(r.x + r.w, w)]
            if crop.size:
                cols[r.team].append(jersey_color(crop))
    sig = {s: [round(float(v), 1) for v in np.median(np.stack(c), axis=0)] if c else None
           for s, c in cols.items()}
    return {"game": info["game_id"], "half": info["game_time_start"].split(" - ")[0],
            "clock": info["game_time_start"], "n_left": len(cols["left"]),
            "n_right": len(cols["right"]), "lab_left": sig["left"], "lab_right": sig["right"]}


def _kit_of_side(sig: dict[str, dict], names: list[str]) -> dict[str, dict]:
    """Name each clip's GT ``left`` side by which of the game's two kits it wears.

    The two kits of one game are recovered by KMeans(2) over every clip's per-side median CIELAB, so
    the handle is the kit itself, not a colour heuristic. An earlier "the redder side" rule
    mis-called SNGS-160, whose two kits differ in lightness but not in ``a*``; the prototype
    assignment gets it right and carries a margin so ties are visible.

    Args:
        sig: Signature per sequence, from :func:`kit_signature`.
        names: The sequences of one game.

    Returns:
        Per sequence: ``kit_of_left`` (prototype index), the assignment ``margin``, and
        ``degenerate`` when both sides land on the same prototype.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    x = np.array([sig[s]["lab_" + side] for s in names for side in ("left", "right")])
    cent = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x).cluster_centers_
    out = {}
    for i, s in enumerate(names):
        dl, dr = (np.linalg.norm(x[2 * i + j] - cent, axis=1) for j in (0, 1))
        out[s] = {"kit_of_left": int(np.argmin(dl)), "kit_of_right": int(np.argmin(dr)),
                  "margin": round(float(min(abs(dl[1] - dl[0]), abs(dr[1] - dr[0]))), 1),
                  "degenerate": bool(np.argmin(dl) == np.argmin(dr))}
    out["_kit_sep"] = {"kit_sep": round(float(np.linalg.norm(cent[0] - cent[1])), 1)}
    return out


def game_consistency(seqs: list[str], data_dir: Path) -> dict:
    """Group clips by (game, half) and flag any clip whose kit -> side map contradicts its siblings.

    Teams change ends once, at half time. So within one (game, half) every clip must give the same
    kit the same side. A lone dissenter contradicts a fact about football, not a modelling choice.

    Args:
        seqs: Sequence names.
        data_dir: GSR data root.

    Returns:
        Per-clip signatures with their kit assignment, the majority mapping of each (game, half)
        group, and the minority clips -- each of which is a candidate ground-truth error.
    """
    sig = {s: kit_signature(data_dir / s) for s in seqs}
    sig = {s: v for s, v in sig.items() if v["lab_left"] and v["lab_right"]}
    by_game: dict[str, list[str]] = defaultdict(list)
    for s, v in sig.items():
        by_game[v["game"]].append(s)
    for game, names in by_game.items():
        kits = _kit_of_side(sig, sorted(names))
        for s in names:
            sig[s].update(kits[s], kit_sep=kits["_kit_sep"]["kit_sep"])
            logger.info("signature %s game %s half %s kit_of_left=%d margin=%.1f%s", s, game,
                        sig[s]["half"], sig[s]["kit_of_left"], sig[s]["margin"],
                        "  DEGENERATE" if sig[s]["degenerate"] else "")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s, v in sig.items():
        groups[(v["game"], v["half"])].append(s)
    out, odd = {}, []
    for key, names in sorted(groups.items()):
        c = Counter(sig[n]["kit_of_left"] for n in names)
        maj = c.most_common(1)[0][0]
        minority = [n for n in names if sig[n]["kit_of_left"] != maj]
        out[f"game{key[0]}_half{key[1]}"] = {
            "n": len(names), "majority_kit_of_left": maj, "counts": {str(k): v for k, v in c.items()},
            "kit_sep": sig[names[0]]["kit_sep"], "minority": minority,
            "n_degenerate": sum(sig[n]["degenerate"] for n in names)}
        odd += minority
    return {"per_seq": sig, "groups": out, "minority_clips": sorted(odd)}


# === Contact sheets ==============================================================================
def contact_sheet(seq_dir: Path, out_png: Path, *, per_team: int = PER_TEAM) -> dict:
    """Write one PNG per sequence: sampled crops in two rows (GT ``left`` / GT ``right``).

    Crops are the largest box of each track in the first and second half of the clip, so the sheet
    spreads over time and covers every labelled tracklet.

    Args:
        seq_dir: Sequence directory.
        out_png: Destination PNG path.
        per_team: Target crops per team row.

    Returns:
        The chosen crops as records (team, track, jersey, frame, box).
    """
    import cv2  # noqa: PLC0415

    df = gt_boxes(seq_dir)
    df = df[df["team"].isin(["left", "right"]) & (df["h"] >= MIN_BOX_H)]
    mid = (df["frame"].min() + df["frame"].max()) / 2
    picks: list[pd.Series] = []
    for team in ("left", "right"):
        t = df[df["team"] == team].assign(area=lambda d: d["w"] * d["h"],
                                          half=lambda d: (d["frame"] > mid).astype(int))
        cand = t.sort_values("area", ascending=False).groupby(["track", "half"], as_index=False).head(1)
        cand = cand.sort_values("area", ascending=False).head(per_team).sort_values("frame")
        picks.append(cand.assign(row=0 if team == "left" else 1))
    sel = pd.concat(picks, ignore_index=True)

    ncol = max(int(sel.groupby("row").size().max()), 1)
    sheet = np.full(((CELL_H + LABEL_H) * 2 + 30, CELL_W * ncol, 3), 32, np.uint8)
    imgs: dict[str, np.ndarray] = {}
    for row in (0, 1):
        for j, r in enumerate(sel[sel["row"] == row].itertuples()):
            if r.file not in imgs:
                imgs[r.file] = cv2.imread(str(seq_dir / "img1" / r.file))
            img = imgs[r.file]
            if img is None:
                continue
            h, w = img.shape[:2]
            crop = img[max(r.y, 0):min(r.y + r.h, h), max(r.x, 0):min(r.x + r.w, w)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (CELL_W, CELL_H))
            y0 = 30 + row * (CELL_H + LABEL_H)
            sheet[y0:y0 + CELL_H, j * CELL_W:(j + 1) * CELL_W] = crop
            tag = f"{r.team[0].upper()} #{r.jersey or '-'}"
            cv2.putText(sheet, tag, (j * CELL_W + 2, y0 + CELL_H + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(sheet, f"t{r.track} f{r.frame}{'GK' if r.role == 'goalkeeper' else ''}",
                        (j * CELL_W + 2, y0 + CELL_H + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(sheet, f"{seq_dir.name}  row1 = GT team 'left'   row2 = GT team 'right'",
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    logger.info("wrote %s (%d crops)", out_png, len(sel))
    return sel[["team", "track", "jersey", "frame", "role", "x", "y", "w", "h"]].to_dict("records")


# === Verdict =====================================================================================
def verdict(orc: dict, kit: dict, *, minority: bool = False) -> str:
    """Turn the three instruments into one verdict.

    Ordering matters. The cross-clip test (:func:`game_consistency`) is decisive and outranks the
    per-clip ones: a clip that maps the kits to sides the other way round from its own siblings in
    the same game half contradicts a fact of the world (teams change ends only at half time), and it
    does so even when the clip is internally coherent. ``gk_self`` is blind to that failure, because
    it compares GT keeper labels with GT keeper positions and never looks at the outfield labels.

    Args:
        orc: :func:`oracle` output for a sequence.
        kit: :func:`kit_check` output for the same sequence.
        minority: The clip's kit -> side map disagrees with its own game-half siblings.

    Returns:
        ``GT SIDE-SWAPPED`` (cross-clip contradiction), ``GT INVERTED`` (the keeper's own label
        contradicts his position), ``GT MIXED`` (kit clustering contradicts the label on separable
        kits), ``GT CONSISTENT (depth inverts)`` (the label survives but the shipped depth rule does
        not), or ``GT CONSISTENT``.
    """
    gk, mx = orc["rules"].get("gk_self"), orc["rules"].get("meanx")
    if minority:
        return "GT SIDE-SWAPPED"
    if gk is not None and gk > 0:
        return "GT INVERTED"
    ok_kit = kit.get("track_agreement")
    if ok_kit is not None and kit.get("separability", 0) >= 1.0 and ok_kit < 0.9:
        return "GT MIXED"
    if mx is not None and mx > 0:
        return "GT CONSISTENT (depth inverts)"
    return "GT CONSISTENT"


def run(seqs: list[str], data_dir: Path, *, sheets: bool = True,
        minority: frozenset[str] = frozenset()) -> dict:
    """Run every instrument (and optionally the contact sheets) over a pool of sequences.

    Args:
        seqs: Sequence names.
        data_dir: GSR data root.
        sheets: Write one contact sheet per sequence.
        minority: Clips flagged by :func:`game_consistency` as contradicting their siblings.
    """
    out = {}
    for s in seqs:
        seq_dir = data_dir / s
        if not seq_dir.exists():
            logger.warning("missing %s", seq_dir)
            continue
        orc = oracle(seq_dir)
        kit = kit_check(seq_dir)
        rec = {"oracle": orc, "kit": kit, "cross_clip_minority": s in minority,
               "verdict": verdict(orc, kit, minority=s in minority)}
        if sheets:
            rec["sheet_crops"] = contact_sheet(seq_dir, OUT_DIR / f"{s}_contact.png")
        out[s] = rec
        logger.info("%s: %s  meanx=%s gk_self=%s kit_track_agree=%s", s, rec["verdict"],
                    orc["rules"].get("meanx"), orc["rules"].get("gk_self"),
                    kit.get("track_agreement"))
    return out


# === Self-check ==================================================================================
def _demo() -> None:
    """Self-check of the verdict logic and the GT re-coding sign convention (asserts; runnable)."""
    frames = np.repeat(np.arange(30), 4)
    df = pd.DataFrame({"frame": frames, "role": ["player"] * 120, "team": [0, 0, 1, 1] * 30,
                       "pitch_x": [20.0, 30.0, 70.0, 80.0] * 30, "pitch_y": [34.0] * 120})
    assert side_scores(df)["meanx"] < 0                       # cluster 0 deep left -> correct
    gk = pd.DataFrame({"frame": [0, 0], "role": ["goalkeeper"] * 2, "team": [0, 1],
                       "pitch_x": [3.0, 102.0], "pitch_y": [34.0] * 2})
    good = side_scores(pd.concat([df, gk]))
    assert good["gk_self"] < 0, good["gk_self"]
    bad = side_scores(pd.concat([df, gk.assign(team=[1, 0])]))
    assert bad["gk_self"] > 0, bad["gk_self"]                 # label contradicts the goal he stands in

    o_ok = {"rules": {"gk_self": good["gk_self"], "meanx": good["meanx"]}}
    o_inv = {"rules": {"gk_self": bad["gk_self"], "meanx": bad["meanx"]}}
    strong = {"track_agreement": 1.0, "separability": 3.0}
    weak = {"track_agreement": 0.6, "separability": 3.0}
    assert verdict(o_ok, strong) == "GT CONSISTENT"
    assert verdict(o_inv, strong) == "GT INVERTED"
    assert verdict(o_ok, weak) == "GT MIXED"
    assert verdict({"rules": {"gk_self": -9.0, "meanx": 2.0}}, strong).startswith("GT CONSISTENT (")
    # the cross-clip contradiction outranks a clip that is internally perfect
    assert verdict(o_ok, strong, minority=True) == "GT SIDE-SWAPPED"

    # the kit -> side handle: two kits, one dark one light, C has them the other way round
    dark, light = [30.0, 5.0, 8.0], [80.0, 1.0, -4.0]
    sig = {"A": {"lab_left": dark, "lab_right": light}, "B": {"lab_left": dark, "lab_right": light},
           "C": {"lab_left": light, "lab_right": dark}, "D": {"lab_left": dark, "lab_right": light}}
    kits = _kit_of_side(sig, ["A", "B", "C", "D"])
    assert kits["A"]["kit_of_left"] == kits["B"]["kit_of_left"] == kits["D"]["kit_of_left"]
    assert kits["C"]["kit_of_left"] != kits["A"]["kit_of_left"], kits
    assert not any(kits[s]["degenerate"] for s in "ABCD")
    # a kit pair that differs only in lightness must still resolve (the a*-only rule failed here)
    pale = {"X": {"lab_left": [42.0, 2.0, -5.0], "lab_right": [77.0, 2.0, -7.0]},
            "Y": {"lab_left": [40.0, 2.0, -4.0], "lab_right": [79.0, 2.0, -6.0]}}
    kx = _kit_of_side(pale, ["X", "Y"])
    assert kx["X"]["kit_of_left"] == kx["Y"]["kit_of_left"], kx
    print("gsr_gt_audit demo OK: gk_self flips sign with the keeper label, the cross-clip minority "
          "outranks it, verdict routes swapped/inverted/mixed/consistent as specified")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--seqs", nargs="*", default=list(FLIP_SET + CONTROLS + DEV_REF))
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--games", action="store_true", help="cross-clip kit -> side consistency")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "gt_audit.json")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.games:
        payload = game_consistency(args.seqs, args.data_dir)
        p = OUT_DIR / "gt_game_consistency.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("MINORITY CLIPS: %s", payload["minority_clips"])
        logger.info("wrote %s", p)
        return
    if not args.run:
        ap.error("choose --run or --demo")
    gc_path = OUT_DIR / "gt_game_consistency.json"
    minority = frozenset(json.loads(gc_path.read_text(encoding="utf-8"))["minority_clips"]
                         if gc_path.exists() else ())
    logger.info("cross-clip minority set: %s", sorted(minority) or "(run --games first)")
    payload = {"pool": args.seqs, "flip_set": list(FLIP_SET), "cross_clip_minority": sorted(minority),
               "per_seq": run(args.seqs, args.data_dir, sheets=not args.no_sheets,
                              minority=minority)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
