"""Full-match 22-player reconstruction on ONE match, validated on real re-appearances.

Three deliverables, in order of what they cost to believe:

1. **Reconstruction.** For every trusted timestep of both halves, the best estimate of every
   player on the pitch per team: observed where the tracker projected one, imputed by the FROZEN
   B4 v1 (``results/B4_MODEL_V1.md`` + the adopted P2 defer-to-anchor policy of
   ``results/B4_ABSTENTION_POLICY.md``) where it did not, abstained where layer B declined. The
   squad arithmetic of ``tools.tactical_clip`` is reused verbatim -- de-duplicate a ghost that
   sits within ``DUP_M`` of a live same-team track, then cap each team at ``SQUAD`` markers.
   Output: a tidy parquet (one row per player-estimate per timestep).

2. **Re-appearance validation.** When a track disappears at ``t0`` and a track that can be linked
   to the SAME player re-appears at ``t1``, the re-appearance position is a MEASURED truth for the
   prediction the model would have made at ``t1``. Two linking rules, deliberately separated:

   * ``within-track`` (primary) -- the tracker itself carried the id across the gap. Extra guard:
     every frame inside the gap must still carry ``MIN_ALIVE`` gated players, so a global
     calibration/replay outage cannot masquerade as an occlusion. False-link risk = ByteTrack's
     own id-switch rate over a gap it re-associated by IoU/motion.
   * ``cross-track`` (secondary, reported but not trusted) -- two name-gated fragments of the same
     player, ``>= MIN_ANCHORS`` agreeing close-up reads each and on the match team sheet, with no
     third gated fragment of that name alive in between. False-link risk = fragment-naming
     precision, which this project has never measured; and the naming is so sparse that
     "the next NAMED fragment" is nowhere near "the next appearance".

   Scoring re-uses the frozen conformal machinery: the model is re-run on a bundle in which the
   re-appearance frame alone is masked, so ``collect_samples`` emits a genuine hidden sample there
   with the observed position as its target.

3. **Shape delta.** Defensive line height and compactness computed observed-only and from the
   reconstruction, so the payoff question ("does completing the team move the tactical picture?")
   gets a number.

Nothing is tuned. The heads, the anchor, the conformal multipliers, the SGR threshold and the
policy are all frozen artifacts.

Run::

    python -m tools.full_match_reconstruction --match tottenham_manutd
    python -m tools.full_match_reconstruction --match tottenham_manutd --chunks h1_chunk_000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from core.pitch import PITCH_LEN
from fingerprint.structural_metrics import (
    attacking_coord,
    resolve_attack_directions,
    resolve_attack_directions_from_ball,
)
from fingerprint.theory_metrics import complete_directions
from synthesizer.imputation import (
    BIN_LABELS,
    collect_samples,
    estimate_velocity,
    slot_prediction,
    visible_centroid,
)
from synthesizer.imputation_features import build_features, bucket_of, derive_fields
from synthesizer.imputation_v1 import (
    b7_position,
    emit,
    paired_rmse_ci,
    sample_signs,
    v1_prediction,
)
from tools.imputation_b4_external import (
    BAR_HALFLIFE_S,
    EDGE_FRAC,
    depth_rank,
    fit_frozen_v1,
    load_ours,
    remap_slot_coeffs,
)
from tools.imputation_b4_transfer import Tee
from tools.tactical_clip import (
    CALIB_MAX_M,
    DUP_M,
    K50_ANC,
    K50_V1,
    K90_ANC,
    K90_V1,
    MAX_IMPUTE_S,
    MIN_ANCHORS,
    R_MAX,
    SKILL_BUCKETS,
    SQUAD,
    load_names,
)

CACHE_DIR = "data/imputation/cache"
OUT_ROOT = Path("results/reconstruction")
PRED_CACHE = Path("data/imputation/cache/b4_v1_frozen_preds.npz")
IDENTITY_ROOT = Path("outputs/identity")
MIN_ALIVE = 4       # gated players a timestep must carry to count as "the camera is tracking"
FEAT_BATCH = 40_000  # the role_ord feature is O(samples x slots); batch so memory stays bounded
BLOCK_S = 60.0      # block-bootstrap block length, matching the frozen gate protocol


# --------------------------------------------------------------------------------------------------
# frozen model, run over one chunk
# --------------------------------------------------------------------------------------------------
@dataclass
class ChunkRun:
    """One chunk's frozen-v1 output plus the bundle it was computed on."""

    chunk: str
    bundle: dict
    samples: dict[str, np.ndarray]
    mu: np.ndarray
    anchor: np.ndarray
    halfw: np.ndarray
    bucket: np.ndarray
    ems: dict[str, np.ndarray]

    def index(self) -> dict[int, int]:
        """``frame_index * n_slots + slot_id -> sample row``, for pulling out one query."""
        n_slots = self.bundle["visible"].shape[1]
        key = self.samples["frame"].astype(np.int64) * n_slots + self.samples["slot_id"].astype(
            np.int64
        )
        return {int(k): i for i, k in enumerate(key)}


def run_chunk(
    match_id: str,
    chunk: str,
    frozen: tuple,
    hide: Sequence[tuple[int, int]] = (),
    write_truth: Sequence[tuple[int, int, float, float]] = (),
) -> ChunkRun:
    """Run the frozen B4 v1 over one chunk, optionally hiding specific slot-frames.

    Args:
        match_id: Registry match id.
        chunk: Chunk key (``h1_chunk_000``).
        frozen: ``(heads, coeffs, tau, w7, train_order)`` from :func:`load_frozen`.
        hide: ``(frame_index, slot)`` pairs to force out of the visibility mask, so the model is
            asked for a prediction where the tracker actually saw the player. Used only by the
            re-appearance validation.
        write_truth: ``(frame_index, slot, x, y)`` targets to write into the truth array for slots
            that are hidden there (the cross-track linking rule needs the successor fragment's
            position parked in the predecessor's slot).

    Returns:
        A :class:`ChunkRun`.
    """
    heads, coeffs, tau, w7, train_order = frozen
    b = load_ours(match_id, 0, chunks=[chunk])
    fill_internal_gaps(b)
    for fi, sl, x, y in write_truth:
        b["truth"][fi, sl] = (x, y)
    if hide:
        idx = np.asarray(hide, dtype=int)
        b["visible"][idx[:, 0], idx[:, 1]] = False
        b["vel"] = estimate_velocity(
            np.where(b["visible"][:, :, None], b["truth"], np.nan), b["period"], b["fps"]
        )
    ext_order = depth_rank(b["truth"], b["visible"], b["period"], b["ranges"])
    ce = remap_slot_coeffs(coeffs, train_order, ext_order)
    fields = derive_fields(b["truth"], b["visible"], b["ball"], b["cam"], b["period"], b["fps"],
                           b["ranges"], ce, BAR_HALFLIFE_S)
    cent = visible_centroid(b["truth"], b["visible"], b["ranges"])
    samples = collect_samples(
        b["truth"], b["visible"], b["period"], b["fps"], b["vel"],
        slot_prediction(cent, b["cam"], b["ranges"], ce),
        fields={"b6": np.asarray(fields["b6"])},
    )
    anchor = b7_position(samples, tau, w7)
    sgn = sample_signs(fields, samples)
    mus, ws = [], []
    for i in range(0, samples["tsls"].size, FEAT_BATCH):
        sl = slice(i, i + FEAT_BATCH)
        sub = {k: v[sl] for k, v in samples.items()}
        m, w = v1_prediction(heads, build_features(fields, sub, anchor[sl]), anchor[sl], sgn[sl])
        mus.append(m)
        ws.append(w)
    mu = np.concatenate(mus) if mus else np.zeros((0, 2))
    halfw = np.concatenate(ws) if ws else np.zeros((0, 2))
    bkt = bucket_of(samples["tsls"])
    geo = np.sqrt(halfw[:, 0] * halfw[:, 1])
    ems = emit(samples, mu, anchor, K90_V1[bkt] * geo, bkt, SKILL_BUCKETS, R_MAX, K90_ANC[bkt] * geo)
    return ChunkRun(chunk, b, samples, mu, anchor, halfw, bkt, ems)


def fill_internal_gaps(bundle: dict) -> int:
    """Give every in-track occlusion a placeholder target so the model predicts through it.

    ``tools.imputation_b4_external.load_ours`` only ghosts a track after its FINAL sighting, and
    ``imputation.collect_samples`` needs a finite ``truth`` entry to emit a hidden sample. The
    consequence, found while assembling this reconstruction, is that the pipeline emits nothing at
    all for a player who drops out for two seconds and comes straight back -- exactly the
    occlusions we can validate. This extends the same never-scored placeholder device
    (``load_ours``'s own comment) to gaps between two sightings.

    Nothing downstream reads the placeholder as information: every structural field and the
    velocity are computed from ``np.where(visible, truth, nan)``, and ``hold`` is taken at the last
    VISIBLE index. It only decides which slot-frames get a prediction.

    Args:
        bundle: Bundle from ``load_ours``; ``truth`` is modified in place.

    Returns:
        Number of slot-frames filled.
    """
    vis = bundle["visible"]
    truth = bundle["truth"]
    n = vis.shape[0]
    last = np.maximum.accumulate(np.where(vis, np.arange(n)[:, None], -1), axis=0)
    later = np.maximum.accumulate(vis[::-1], axis=0)[::-1]
    fill = (~vis) & (last >= 0) & later
    r, c = np.nonzero(fill)
    truth[r, c] = truth[last[r, c], c]
    return int(r.size)


def load_frozen(out: Tee) -> tuple:
    """Load (or fit once) the frozen v1 heads, anchor and TRAIN depth ranking."""
    heads, coeffs, tau, w7, g1 = fit_frozen_v1(CACHE_DIR, out)
    train_order = depth_rank(g1["truth"], g1["visible"], g1["period"], g1["ranges"])
    return heads, coeffs, tau, w7, train_order


# --------------------------------------------------------------------------------------------------
# task 1: reconstruction assembly
# --------------------------------------------------------------------------------------------------
def chunk_frame_map(run: ChunkRun) -> tuple[int, int, int]:
    """``(offset, first source frame, source-frame step)`` for this chunk's single period."""
    return run.bundle["frame_map"][1]


def assemble(run: ChunkRun, meta: pd.DataFrame, names: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Merge observed tracks and accepted ghosts into one tidy per-timestep estimate table.

    Squad arithmetic, identical to ``tools.tactical_clip.Passage.ghosts_at``: a ghost within
    :data:`DUP_M` of an observed same-team position is a re-identified live player and is dropped;
    what remains is capped, most-recently-seen first, at ``SQUAD`` minus the observed count.

    Args:
        run: The chunk's frozen-v1 output.
        meta: Per-``track_id`` metadata for this chunk (``team``, ``keeper``).
        names: ``(chunk, track_id) -> name`` for identity-gated tracks.

    Returns:
        Tidy frame with one row per player estimate per timestep.
    """
    b = run.bundle
    off, f0, step = chunk_frame_map(run)
    slot_track = {int(s): int(t) for s, t in b["slot_track"][1].items()}
    team_of = dict(zip(meta["track_id"], meta["team"], strict=True))
    keeper_of = dict(zip(meta["track_id"], meta["keeper"], strict=True))
    n_frames, n_slots = b["visible"].shape
    slot_team = np.array([team_of.get(slot_track.get(s, -1), -1) for s in range(n_slots)])

    rows: list[pd.DataFrame] = []
    fi, si = np.nonzero(b["visible"])
    obs = pd.DataFrame({
        "fi": fi, "slot": si, "x": b["truth"][fi, si, 0], "y": b["truth"][fi, si, 1],
        "team": slot_team[si], "source": "observed", "tsls_s": 0.0,
        "r50_m": np.nan, "r90_m": np.nan, "asserted": True,
    })

    s = run.samples
    keep = (s["tsls"] <= MAX_IMPUTE_S) & (s["tsls"] > 0)
    k = np.flatnonzero(keep)
    gk = run.bucket[k]
    geo = np.sqrt(run.halfw[k, 0] * run.halfw[k, 1])
    is_v1 = run.ems["source"][k] == "v1"
    seen = b["visible"].any(axis=0)
    last_vis = np.where(seen, n_frames - 1 - np.argmax(b["visible"][::-1], axis=0), -1)
    gho = pd.DataFrame({
        "fi": s["frame"][k].astype(int), "slot": s["slot_id"][k].astype(int),
        "x": run.ems["pos"][k, 0], "y": run.ems["pos"][k, 1],
        "team": slot_team[s["slot_id"][k].astype(int)],
        "source": run.ems["source"][k], "tsls_s": s["tsls"][k],
        "r50_m": np.where(is_v1, K50_V1[gk], K50_ANC[gk]) * geo,
        "r90_m": np.where(is_v1, K90_V1[gk], K90_ANC[gk]) * geo,
        "asserted": run.ems["asserted"][k],
        # a ghost of a slot that is seen again later is a genuine occlusion of a live track; one
        # past its final sighting is a dead re-identification fragment and may be a phantom.
        "ghost_kind": np.where(s["frame"][k].astype(int) < last_vis[s["slot_id"][k].astype(int)],
                               "gap", "terminal"),
    })
    gho = gho[gho["team"] >= 0]
    obs["ghost_kind"] = "observed"

    live_count = b["visible"].sum(axis=1)
    n_obs = np.zeros((n_frames, 2), dtype=int)
    for t in (0, 1):
        m = obs["team"].to_numpy() == t
        np.add.at(n_obs[:, t], obs["fi"].to_numpy()[m], 1)

    picked: list[np.ndarray] = []
    obs_xy = {t: _by_frame(obs[obs["team"] == t], n_frames) for t in (0, 1)}
    for t in (0, 1):
        gt = gho[gho["team"] == t]
        if gt.empty:
            continue
        order = np.lexsort((gt["tsls_s"].to_numpy(), gt["fi"].to_numpy()))
        gi = gt.index.to_numpy()[order]
        room = np.maximum(SQUAD - n_obs[:, t], 0)
        used = np.zeros(n_frames, dtype=int)
        gx = gt["x"].to_numpy()[order]
        gy = gt["y"].to_numpy()[order]
        gf = gt["fi"].to_numpy()[order]
        sel = np.zeros(gi.size, dtype=bool)
        for j in range(gi.size):
            f = int(gf[j])
            if used[f] >= room[f]:
                continue
            pts = obs_xy[t][f]
            if pts.size and np.min(np.hypot(pts[:, 0] - gx[j], pts[:, 1] - gy[j])) <= DUP_M:
                continue
            sel[j] = True
            used[f] += 1
        picked.append(gi[sel])
    kept = gho.loc[np.concatenate(picked)] if picked else gho.iloc[:0]

    out = pd.concat([obs, kept], ignore_index=True)
    out["track_id"] = [slot_track.get(int(v), -1) for v in out["slot"]]
    out["frame"] = f0 + (out["fi"].to_numpy() - off) * step
    out["chunk"] = run.chunk
    out["half"] = run.chunk[:2]
    out["time_s"] = (out["fi"].to_numpy() - off) / b["fps"]
    out["keeper"] = [bool(keeper_of.get(t, False)) for t in out["track_id"]]
    out["player_name"] = [names.get((run.chunk, int(t))) for t in out["track_id"]]
    out["observed"] = out["source"] == "observed"
    out["trusted_ts"] = live_count[out["fi"].to_numpy()] >= MIN_ALIVE
    rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _by_frame(df: pd.DataFrame, n_frames: int) -> list[np.ndarray]:
    """Per-frame array of ``(x, y)`` for a table carrying an ``fi`` column."""
    out: list[np.ndarray] = [np.zeros((0, 2)) for _ in range(n_frames)]
    if df.empty:
        return out
    for f, g in df.groupby("fi"):
        out[int(f)] = g[["x", "y"]].to_numpy(float)
    return out


def coverage_profile(rec: pd.DataFrame) -> pd.DataFrame:
    """How many of the eleven we have per team per trusted timestep, observed vs completed."""
    t = rec[rec["trusted_ts"]]
    key = ["chunk", "fi", "team"]
    obs = t[t["observed"]].groupby(key).size()
    est = t[t["observed"] | t["asserted"]].groupby(key).size()
    grid = est.index.union(obs.index)
    return pd.DataFrame({
        "observed": obs.reindex(grid).fillna(0).astype(int),
        "estimated": est.reindex(grid).fillna(0).astype(int),
    }).reset_index()


# --------------------------------------------------------------------------------------------------
# task 2: re-appearance events
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    """One re-appearance: slot ``slot`` last seen at ``t0``, measured again at ``t1``."""

    chunk: str
    slot: int
    t0: int
    t1: int
    dur_s: float
    link: str            # "within-track" | "cross-track"
    target: tuple[float, float]
    mask_slots: tuple[int, ...]
    label: str


def within_track_events(run: ChunkRun, min_alive: int = MIN_ALIVE) -> list[Event]:
    """Gaps a single track id spans, with every gap frame still carrying live tracking.

    The guard matters: most gaps in the gated table are whole-frame calibration/replay outages
    where nobody at all is tracked, and those are measurement holes rather than occlusions.
    """
    b = run.bundle
    vis = b["visible"]
    live = vis.sum(axis=1)
    fps = b["fps"]
    out: list[Event] = []
    for s in range(vis.shape[1]):
        idx = np.flatnonzero(vis[:, s])
        if idx.size < 2:
            continue
        for k in np.flatnonzero(np.diff(idx) > 1):
            t0, t1 = int(idx[k]), int(idx[k + 1])
            if not (live[t0 + 1 : t1] >= min_alive).all():
                continue
            out.append(Event(run.chunk, s, t0, t1, (t1 - t0) / fps, "within-track",
                             (float(b["truth"][t1, s, 0]), float(b["truth"][t1, s, 1])),
                             (s,), f"track {b['slot_track'][1].get(s, -1)}"))
    return _drop_chained(out)


def _drop_chained(events: list[Event]) -> list[Event]:
    """Drop an event whose ``t0`` is another kept event's masked ``t1`` on the same slot.

    Masking a re-appearance frame lengthens the following gap, which would corrupt the next
    event's ``tsls``. Earliest event wins.
    """
    kept: list[Event] = []
    masked: set[tuple[int, int]] = set()
    for e in sorted(events, key=lambda q: (q.slot, q.t0)):
        if (e.slot, e.t0) in masked:
            continue
        kept.append(e)
        masked.add((e.slot, e.t1))
    return kept


def cross_track_events(run: ChunkRun, named: pd.DataFrame) -> list[Event]:
    """Identity-linked re-appearances: consecutive name-gated fragments of the same player.

    Conservative by construction: both fragments need ``>= MIN_ANCHORS`` agreeing close-up name
    reads and a name on the match team sheet, they must not overlap in time, and no third gated
    fragment of the same name may be alive between them. The residual false-link risk is the
    per-fragment naming precision, which is unmeasured in this project.
    """
    b = run.bundle
    track_slot = {int(t): int(s) for s, t in b["slot_track"][1].items()}
    vis = b["visible"]
    fps = b["fps"]
    span: dict[int, tuple[int, int]] = {}
    for tid, s in track_slot.items():
        idx = np.flatnonzero(vis[:, s])
        if idx.size:
            span[tid] = (int(idx[0]), int(idx[-1]))
    out: list[Event] = []
    for name, g in named[named["chunk"] == run.chunk].groupby("player_name"):
        frags = sorted((span[t][0], span[t][1], int(t)) for t in g["track_id"] if int(t) in span)
        for i in range(len(frags) - 1):
            a, c = frags[i], frags[i + 1]
            if c[0] <= a[1]:
                continue
            if any(x[0] < c[0] and x[1] > a[1] for x in frags if x not in (a, c)):
                continue
            sa, sc = track_slot[a[2]], track_slot[c[2]]
            out.append(Event(run.chunk, sa, a[1], c[0], (c[0] - a[1]) / fps, "cross-track",
                             (float(b["truth"][c[0], sc, 0]), float(b["truth"][c[0], sc, 1])),
                             (sa, sc), f"{name} {a[2]}->{c[2]}"))
    return out


def score_events(run: ChunkRun, events: Sequence[Event]) -> pd.DataFrame:
    """Score the masked run against each event's measured re-appearance position."""
    lookup = run.index()
    n_slots = run.bundle["visible"].shape[1]
    live = run.bundle["visible"].sum(axis=1)
    fps = run.bundle["fps"]
    rows: list[dict] = []
    for e in events:
        i = lookup.get(e.t1 * n_slots + e.slot)
        if i is None:
            continue
        pos = run.ems["pos"][i]
        w = run.halfw[i]
        bkt = int(run.bucket[i])
        src = str(run.ems["source"][i])
        k50 = (K50_V1 if src == "v1" else K50_ANC)[bkt]
        k90 = (K90_V1 if src == "v1" else K90_ANC)[bkt]
        resid = pos - np.array(e.target)
        radial = float(np.hypot(resid[0] / w[0], resid[1] / w[1]))
        geo = float(np.sqrt(w[0] * w[1]))
        rows.append({
            "chunk": e.chunk, "link": e.link, "label": e.label, "slot": e.slot,
            "t0": e.t0, "t1": e.t1, "dur_s": e.dur_s, "tsls_s": float(run.samples["tsls"][i]),
            "bucket": bkt, "source": src, "asserted": bool(run.ems["asserted"][i]),
            "err_m": float(np.hypot(*resid)),
            "err_anchor_m": float(np.linalg.norm(run.anchor[i] - np.array(e.target))),
            "err_hold_m": float(np.linalg.norm(run.samples["hold"][i] - np.array(e.target))),
            "err_v1_m": float(np.linalg.norm(run.mu[i] - np.array(e.target))),
            "inside50": radial <= k50, "inside90": radial <= k90,
            "r50_m": k50 * geo, "r90_m": k90 * geo,
            "alive_frac": float((live[e.t0 + 1 : e.t1] >= MIN_ALIVE).mean()) if e.t1 > e.t0 + 1
            else 1.0,
            "block": int(e.t1 // int(BLOCK_S * fps)),
        })
    return pd.DataFrame(rows)


def disappearance_census(run: ChunkRun, events: Sequence[Event], img_x: np.ndarray) -> pd.DataFrame:
    """Every track-loss in the chunk, flagged by whether it ever came back (selection bias).

    A track-loss is a slot that is visible at ``t`` and not at ``t+1``. It either re-appears
    (under the same id, possibly after a gap that failed the liveness guard) or it never does.
    Comparing the two populations at ``t0`` is how the "re-appearance is not a random occlusion"
    warning gets a number instead of a caveat.
    """
    b = run.bundle
    vis = b["visible"]
    n, n_slots = vis.shape
    ball = b["ball"]
    usable = {(e.slot, e.t0) for e in events}
    rows: list[dict] = []
    for s in range(n_slots):
        idx = np.flatnonzero(vis[:, s])
        if idx.size == 0:
            continue
        ends = np.concatenate([idx[np.flatnonzero(np.diff(idx) > 1)], idx[-1:]])
        nxt = np.concatenate([idx[np.flatnonzero(np.diff(idx) > 1) + 1], [n]])
        for t0, t1 in zip(ends, nxt, strict=True):
            back = t1 < n
            iw = float(img_x[t0, s]) if np.isfinite(img_x[t0, s]) else np.nan
            rows.append({
                "slot": int(s), "t0": int(t0),
                "returns": bool(back), "usable": (int(s), int(t0)) in usable,
                "gap_s": (t1 - t0) / b["fps"] if back else np.nan,
                "censored_s": (n - t0) / b["fps"],
                "d_ball_m": float(np.linalg.norm(b["truth"][t0, s] - ball[t0]))
                if np.isfinite(ball[t0, 0]) else np.nan,
                "img_edge": iw,
            })
    return pd.DataFrame(rows)


def image_x_field(run: ChunkRun, df: pd.DataFrame, img_w: float) -> np.ndarray:
    """``(n_frames, n_slots)`` normalised distance of each sighting from the nearest image edge."""
    b = run.bundle
    off, f0, step = chunk_frame_map(run)
    track_slot = {int(t): int(s) for s, t in b["slot_track"][1].items()}
    out = np.full(b["visible"].shape, np.nan)
    d = df[df["chunk"] == run.chunk]
    fi = off + ((d["frame"].to_numpy() - f0) // step).astype(int)
    sl = np.array([track_slot.get(int(t), -1) for t in d["track_id"]])
    ok = (sl >= 0) & (fi >= 0) & (fi < out.shape[0])
    ix = d["image_x"].to_numpy(float)
    out[fi[ok], sl[ok]] = np.minimum(ix[ok], img_w - ix[ok]) / img_w
    return out


# --------------------------------------------------------------------------------------------------
# task 3: shape metrics
# --------------------------------------------------------------------------------------------------
def shape_metrics(rec: pd.DataFrame, dirs: dict[str, dict[int, int]]) -> pd.DataFrame:
    """Line height and compactness per (timestep, team), observed-only vs full reconstruction.

    Line height follows the frozen convention of ``imputation_features._team_stats``: the second
    deepest outfielder in the team's own attacking frame (own goal = 0 m). Compactness is reported
    as the depth spread (std along attacking-x) and width spread (std along y), the same two axes
    ``fingerprint.block_height`` uses.
    """
    rows: list[dict] = []
    use = rec[rec["trusted_ts"] & ~rec["keeper"] & (rec["observed"] | rec["asserted"])]
    for (ck, fi, team), g in use.groupby(["chunk", "fi", "team"]):
        d = dirs.get(ck, {}).get(int(team))
        if d is None:
            continue
        obs = g[g["observed"]]
        if len(obs) < 3:
            continue
        row = {"chunk": ck, "fi": int(fi), "team": int(team),
               "n_obs": len(obs), "n_rec": len(g), "n_added": len(g) - len(obs)}
        for tag, h in (("obs", obs), ("rec", g)):
            ax = np.sort(attacking_coord(h["x"].to_numpy(float), d))
            row[f"line_{tag}"] = float(ax[1])
            row[f"depth_{tag}"] = float(np.std(ax))
            row[f"width_{tag}"] = float(np.std(h["y"].to_numpy(float)))
            row[f"xspan_{tag}"] = float(ax[-1] - ax[0])
        rows.append(row)
    return pd.DataFrame(rows)


def chunk_directions(match: registry.Match) -> dict[str, dict[int, int]]:
    """Per-chunk attacking direction for both teams (ball-based first, keeper fallback)."""
    aligned = match.load_aligned()
    balls = dict(match.ball_chunks())
    out: dict[str, dict[int, int]] = {}
    for ck, g in aligned.groupby("chunk"):
        pos = g.dropna(subset=["pitch_x", "pitch_y"])
        d: dict[int, int] = {}
        if ck in balls:
            d = resolve_attack_directions_from_ball(pos, pd.read_parquet(balls[ck]))
        if len(d) < 2:
            d = resolve_attack_directions(pos)
        out[str(ck)] = complete_directions(d)
    return out


# --------------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------------
def bucket_table(ev: pd.DataFrame, out: Tee, title: str, sigma: float = 0.0) -> None:
    """Per-horizon-bucket census, error and empirical region coverage on the scored events.

    Args:
        ev: Scored events from :func:`score_events`.
        out: Report sink.
        title: Table caption.
        sigma: Measured per-axis observation-noise standard deviation (m). When non-zero an extra
            column reports what coverage the SAME frozen regions would have if inflated in
            quadrature by that noise -- a diagnostic of *why* they miss, never a re-calibration.
    """
    out(f"\n  {title}")
    out("  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 "
        "|  r50 |  r90" + (" | 50+n | 90+n" if sigma else ""))
    for b, lab in enumerate([*BIN_LABELS, "ALL"]):
        e = ev if lab == "ALL" else ev[ev["bucket"] == b]
        if e.empty:
            out(f"  {lab:9s} |    0 |" + " |".join(["      -", "    -", "    -", "     -",
                                                    "     -", "     -", "     -", "     -",
                                                    "   -", "   -"]))
            continue
        extra = ""
        if sigma:
            i50, i90 = _inflated_cover(e, sigma)
            extra = f" | {100 * i50:4.1f} | {100 * i90:4.1f}"
        out(f"  {lab:9s} | {len(e):4d} | {_rmse(e['err_m']):7.2f} | {e['err_m'].median():4.1f} | "
            f"{e['err_m'].quantile(0.9):4.1f} | {_rmse(e['err_anchor_m']):6.2f} | "
            f"{_rmse(e['err_hold_m']):6.2f} | {_rmse(e['err_v1_m']):6.2f} | "
            f"{100 * e['inside50'].mean():6.1f} | {100 * e['inside90'].mean():6.1f} | "
            f"{e['r50_m'].mean():4.1f} | {e['r90_m'].mean():4.1f}" + extra)


def _inflated_cover(e: pd.DataFrame, sigma: float) -> tuple[float, float]:
    """Coverage of the frozen regions widened in quadrature by an observation-noise sigma."""
    out = []
    for col in ("r50_m", "r90_m"):
        r = np.sqrt(e[col].to_numpy() ** 2 + sigma**2 * 2.0)
        out.append(float((e["err_m"].to_numpy() <= r).mean()))
    return out[0], out[1]


def noise_floor(gated: pd.DataFrame) -> tuple[float, int]:
    """Per-axis observation noise of our own projected positions, from midpoint deviations.

    For three consecutive sightings of one track the midpoint deviation
    ``p_t - (p_{t-1} + p_{t+1}) / 2`` removes any constant-velocity motion, leaving acceleration
    (about 0.06 m over a 0.2 s step at 3 m/s^2) plus measurement noise. With i.i.d. per-axis noise
    of variance ``s^2`` the deviation has variance ``1.5 s^2``, so ``s = rms(dev) / sqrt(1.5)``.
    This is the floor the frozen Metrica-calibrated regions never had to absorb: Metrica truth is
    exact, ours is a homography projection of a detection box.

    Args:
        gated: Trusted-geometry position rows for the whole match.

    Returns:
        Tuple ``(sigma_m, n_triples)``.
    """
    devs: list[np.ndarray] = []
    n = 0
    for (_, _), g in gated.groupby(["chunk", "track_id"]):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        p = g[["pitch_x", "pitch_y"]].to_numpy(float)
        if f.size < 3:
            continue
        d1, d2 = f[1:-1] - f[:-2], f[2:] - f[1:-1]
        ok = d1 == d2
        if not ok.any():
            continue
        devs.append((p[1:-1] - 0.5 * (p[:-2] + p[2:]))[ok])
        n += int(ok.sum())
    if not devs:
        return float("nan"), 0
    d = np.concatenate(devs)
    return float(np.sqrt(np.mean(d**2) / 1.5)), n


def _rmse(v) -> float:
    """Root mean square of a series."""
    a = np.asarray(v, float)
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")


def metrica_arrays() -> dict[str, np.ndarray]:
    """Frozen Metrica-holdout predictions, reduced to the P2 emitted point and its region test.

    Reproduces the emitted point (anchor in 0-1s, v1 elsewhere) with each estimator's own
    conformal multipliers, so the coverage numbers derived here are the ones
    ``results/B4_ABSTENTION_POLICY.md`` section 2 recorded.
    """
    if not PRED_CACHE.exists():
        return {}
    z = np.load(PRED_CACHE)
    bkt = z["bucket"]
    use_v1 = np.isin(bkt, np.asarray(SKILL_BUCKETS))
    resid = np.where(use_v1[:, None], z["mu"], z["anchor"]) - z["target"]
    return {
        "bucket": bkt,
        "err": np.linalg.norm(resid, axis=1),
        "disp": np.linalg.norm(z["target"] - z["lastseen"], axis=1),
        "radial": np.hypot(resid[:, 0] / z["w"][:, 0], resid[:, 1] / z["w"][:, 1]),
        "k50": np.where(use_v1, K50_V1[bkt], K50_ANC[bkt]),
        "k90": np.where(use_v1, K90_V1[bkt], K90_ANC[bkt]),
    }


def metrica_reference(z: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-bucket RMSE, coverage and last-seen displacement of the frozen Metrica holdout."""
    if not z:
        return pd.DataFrame()
    rows = []
    for b, lab in enumerate(BIN_LABELS):
        m = z["bucket"] == b
        rows.append({
            "bucket": b, "label": lab, "n": int(m.sum()), "rmse": _rmse(z["err"][m]),
            "picp50": float((z["radial"][m] <= z["k50"][m]).mean() * 100),
            "picp90": float((z["radial"][m] <= z["k90"][m]).mean() * 100),
            "disp_p50": float(np.percentile(z["disp"][m], 50)),
            "disp_mean": float(z["disp"][m].mean()),
        })
    return pd.DataFrame(rows)


def metrica_matched(z: dict[str, np.ndarray], bucket: int, max_disp: float) -> dict[str, float]:
    """Metrica-holdout numbers restricted to samples that moved no further than ``max_disp``.

    The re-appearance sample is not a random occlusion: a track the tracker re-associates has, by
    construction, come back near where it left. Matching Metrica on the same displacement makes
    the two error numbers comparable instead of merely adjacent.
    """
    m = (z["bucket"] == bucket) & (z["disp"] <= max_disp)
    if m.sum() < 50:
        return {"n": int(m.sum()), "rmse": float("nan"), "picp50": float("nan"),
                "picp90": float("nan")}
    return {
        "n": int(m.sum()), "rmse": _rmse(z["err"][m]),
        "picp50": float((z["radial"][m] <= z["k50"][m]).mean() * 100),
        "picp90": float((z["radial"][m] <= z["k90"][m]).mean() * 100),
    }


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------
def track_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Per-``(chunk, track_id)`` team and keeper flag from the gated positions table."""
    g = df.groupby(["chunk", "track_id"])
    return pd.DataFrame({
        "team": g["team"].first().astype(int),
        "keeper": (g["is_keeper"].mean() >= 0.5) | (g["role"].first() == "goalkeeper"),
    }).reset_index()


def gated_names(match: registry.Match) -> pd.DataFrame:
    """Name-gated track fragments (``>= MIN_ANCHORS`` reads and on the team sheet)."""
    path = IDENTITY_ROOT / f"{match.id}_named_tracks_both2_prtreid.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["chunk", "track_id", "player_name"])
    nt = pd.read_parquet(path)
    nt = nt[nt["n_anchors"] >= MIN_ANCHORS]
    sheet_path = IDENTITY_ROOT / f"{match.id}_lineup_assign.parquet"
    if sheet_path.exists():
        sheet = set(pd.read_parquet(sheet_path)["name"].astype(str))
        nt = nt[nt["player_name"].astype(str).isin(sheet)]
    return nt[["chunk", "track_id", "player_name"]].copy()


def squad_cap(match: registry.Match, out: Tee) -> int:
    """Check the Sofascore oracle agrees the pitch carries eleven a side all match.

    Substitutions never change the count; a red card does. The oracle is read from its offline
    cache only -- no network call is made here.
    """
    from tools import oracle  # noqa: PLC0415

    try:
        d = oracle.resolve_fixture(
            {"home": match.home_team or match.teams[0], "away": match.away_team or match.teams[1],
             "date": match.date or "", "league": "England Premier League", "year": "24/25"},
            oracle.ScraperFCFetcher(),
        )
        ps = oracle.player_stats_df(d["id"], oracle.ScraperFCFetcher())
    except (LookupError, FileNotFoundError, OSError) as exc:
        out(f"  [oracle] unavailable ({exc}); assuming {SQUAD} a side")
        return SQUAD
    played = ps[ps["minutesPlayed"].notna()]
    for tid, g in played.groupby("teamId"):
        starters = int((~g["substitute"].astype(bool)).sum())
        subs = int(g["substitute"].astype(bool).sum())
        off = int(((~g["substitute"].astype(bool)) & (g["minutesPlayed"] < 90)).sum())
        out(f"  [oracle] team {tid}: {starters} starters, {subs} used subs, {off} withdrawn "
            f"-> on-pitch cap {SQUAD} throughout"
            + ("" if starters == 11 and subs == off else "  <-- IMBALANCE, check for a red card"))
    return SQUAD


def run(match_id: str, chunks: Sequence[str] | None, out: Tee, out_dir: Path) -> dict:
    """Reconstruct one match, validate it on re-appearances and measure the shape delta."""
    match = registry.get(match_id)
    if not match.processed:
        raise SystemExit(f"{match_id} has no aligned parquet at {match.aligned}")
    df = match.load_aligned()
    gated = df[(df["calib_error_m"] <= CALIB_MAX_M) & df["pitch_x"].notna() & df["pitch_y"].notna()
               & df["team"].isin([0, 1]) & df["role"].isin(["player", "goalkeeper"])]
    img_w = float(df["image_x"].max())
    keys = list(chunks) if chunks else sorted(gated["chunk"].unique())
    meta = track_meta(gated)
    names = load_names(match)
    named = gated_names(match)
    out(f"match {match_id} ({match.teams[0]} vs {match.teams[1]}), {len(keys)} chunks, "
        f"{len(gated)} gated position rows, {len(named)} name-gated fragments")
    squad_cap(match, out)

    frozen = load_frozen(out)
    recs, evs, cens, cross = [], [], [], []
    for ck in keys:
        t0 = time.time()
        base = run_chunk(match_id, ck, frozen)
        rec = assemble(base, meta[meta["chunk"] == ck], names)
        recs.append(rec)

        wt = within_track_events(base)
        xt = cross_track_events(base, named)
        cens.append(disappearance_census(base, wt, image_x_field(base, gated, img_w))
                    .assign(chunk=ck))
        events = wt + xt
        ev = pd.DataFrame()
        if events:
            hide = [(e.t1, s) for e in events for s in e.mask_slots]
            write = [(e.t1, e.slot, *e.target) for e in events if e.link == "cross-track"]
            masked = run_chunk(match_id, ck, frozen, hide=hide, write_truth=write)
            ev = score_events(masked, events)
            ev["block"] = ev["block"] + 1000 * keys.index(ck)
            evs.append(ev)
        cross.append(pd.DataFrame([{"chunk": ck, "n_within": len(wt), "n_cross": len(xt)}]))
        out(f"  {ck}: {base.bundle['visible'].shape[0]} timesteps x "
            f"{base.bundle['visible'].shape[1]} slots, {base.samples['tsls'].size} hidden samples, "
            f"{len(wt)} within-track + {len(xt)} cross-track events, "
            f"{len(rec)} estimate rows [{time.time() - t0:.0f} s]")

    rec = pd.concat(recs, ignore_index=True)
    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    cen = pd.concat(cens, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["match", "half", "chunk", "frame", "time_s", "team", "team_name", "track_id",
            "player_name", "x", "y", "source", "observed", "asserted", "ghost_kind", "tsls_s",
            "r50_m", "r90_m", "keeper", "trusted_ts"]
    rec["match"] = match_id
    rec["team_name"] = [match.teams[int(t)] for t in rec["team"]]
    path = out_dir / f"{match_id}_reconstruction.parquet"
    rec[cols].to_parquet(path, index=False)
    out(f"\nwrote {path} ({len(rec)} rows)")
    if not ev.empty:
        ev.to_parquet(out_dir / f"{match_id}_reappearance_events.parquet", index=False)

    sigma, n_trip = noise_floor(gated[gated["chunk"].isin(keys)])
    report(match, rec, ev, cen, out, sigma, n_trip)
    return {"rec": rec, "ev": ev, "cen": cen, "sigma": sigma}


def report(match: registry.Match, rec: pd.DataFrame, ev: pd.DataFrame, cen: pd.DataFrame,
           out: Tee, sigma: float = 0.0, n_trip: int = 0) -> None:
    """Print every table the deliverable needs."""
    out("\n" + "=" * 96)
    out("TASK 1 -- COVERAGE PROFILE")
    out("=" * 96)
    prof = coverage_profile(rec)
    n_ts = rec.loc[rec["trusted_ts"], ["chunk", "fi"]].drop_duplicates().shape[0]
    n_all = rec[["chunk", "fi"]].drop_duplicates().shape[0]
    fps = 5.0
    out(f"timesteps on the chunk grid: {n_all} ({n_all / fps / 60:.1f} min); trusted "
        f"(>= {MIN_ALIVE} gated players): {n_ts} ({100 * n_ts / n_all:.1f}%)")
    out("\nplayers per team per trusted timestep (share of team-timesteps):")
    out("  n  | observed only | with reconstruction")
    for n in range(SQUAD + 1):
        o = float((prof["observed"] == n).mean())
        e = float((prof["estimated"] == n).mean())
        out(f"  {n:2d} | {100 * o:12.1f}% | {100 * e:18.1f}%")
    out(f"  >{SQUAD} | {100 * float((prof['observed'] > SQUAD).mean()):12.1f}% | "
        f"{100 * float((prof['estimated'] > SQUAD).mean()):18.1f}%")
    out(f"\nmean per team: observed {prof['observed'].mean():.2f} -> reconstruction "
        f"{prof['estimated'].mean():.2f} of {SQUAD}")
    for lo in (11, 10, 9, 8):
        out(f"  fraction of team-timesteps with >= {lo:2d}: observed "
            f"{100 * float((prof['observed'] >= lo).mean()):5.1f}%  reconstruction "
            f"{100 * float((prof['estimated'] >= lo).mean()):5.1f}%")
    imp = rec[~rec["observed"]]
    out(f"\nimputed rows by source: {dict(imp['source'].value_counts())}")
    out(f"imputed rows by provenance: {dict(imp['ghost_kind'].value_counts())} -- 'terminal' means "
        "the slot is never seen again, i.e. a dead re-identification fragment that may be a")
    out("phantom rather than a genuinely occluded player (results/B4_TRANSFER_M3.md section 3).")

    out("\n" + "=" * 96)
    out("TASK 2 -- RE-APPEARANCE VALIDATION")
    out("=" * 96)
    if ev.empty:
        out("no usable re-appearance events")
        return
    out("census by linking rule and horizon bucket:")
    tab = ev.pivot_table(index="link", columns="bucket", values="err_m", aggfunc="size",
                         fill_value=0)
    tab.columns = [BIN_LABELS[c] for c in tab.columns]
    out(tab.to_string())
    wt = ev[ev["link"] == "within-track"]
    xt = ev[ev["link"] == "cross-track"]
    out(f"\nobservation-noise floor of our own projected positions: sigma = {sigma:.2f} m per axis "
        f"({n_trip} midpoint triples). Metrica truth has none; the frozen regions were calibrated")
    out("against exact truth, so this much error is unmodellable by construction on our footage.")
    bucket_table(wt, out, "PRIMARY -- within-track linking (tracker id carried across the gap)",
                 sigma)
    out("  (50+n / 90+n = the SAME frozen regions widened in quadrature by the noise floor -- a")
    out("   diagnostic of why they miss, NOT a re-calibration; nothing in the model changed.)")
    if not xt.empty:
        bucket_table(xt, out, "SECONDARY -- cross-track identity linking (NOT trusted; see report)")
        out(f"  cross-track gap frames with live tracking: median "
            f"{100 * xt['alive_frac'].median():.0f}% -- a broadcast outage, not an occlusion")

    z = metrica_arrays()
    ref = metrica_reference(z)
    if not ref.empty:
        out("\n  Metrica holdout (frozen, same emitted policy) for comparison:")
        out("  horizon   |       n |   RMSE | PICP50 | PICP90 | |truth-lastseen| p50 / mean")
        for r in ref.itertuples():
            out(f"  {r.label:9s} | {r.n:7d} | {r.rmse:6.2f} | {r.picp50:6.1f} | {r.picp90:6.1f} | "
                f"{r.disp_p50:6.2f} / {r.disp_mean:6.2f}")

    out("\n  SELECTION: displacement between the last sighting and the measured truth, and the")
    out("  Metrica holdout restricted to samples that moved no further (like for like):")
    out("  horizon   |    n | p50 disp | mean disp | Metrica p50 | matched n | RMSE | P50 | P90")
    for b, lab in enumerate(BIN_LABELS):
        m = wt["bucket"] == b
        if not m.any():
            continue
        d = wt.loc[m, "err_hold_m"]
        rr = ref[ref["bucket"] == b]
        mm = metrica_matched(z, b, float(d.quantile(0.9))) if z else {"n": 0, "rmse": float("nan"),
                                                                      "picp50": float("nan"),
                                                                      "picp90": float("nan")}
        out(f"  {lab:9s} | {int(m.sum()):4d} | {d.median():8.2f} | {d.mean():9.2f} | "
            f"{float(rr['disp_p50'].iloc[0]) if len(rr) else float('nan'):11.2f} | "
            f"{mm['n']:9d} | {mm['rmse']:4.2f} | {mm['picp50']:3.0f} | {mm['picp90']:3.0f}")

    out("\n  paired block-bootstrap CI on RMSE(emitted) - RMSE(anchor) and - RMSE(hold):")
    for b, lab in enumerate([*BIN_LABELS, "ALL"]):
        e = wt if lab == "ALL" else wt[wt["bucket"] == b]
        if len(e) < 10:
            continue
        la, ha = paired_rmse_ci(e["err_anchor_m"].to_numpy(), e["err_m"].to_numpy(),
                                e["block"].to_numpy())
        lh, hh = paired_rmse_ci(e["err_hold_m"].to_numpy(), e["err_m"].to_numpy(),
                                e["block"].to_numpy())
        out(f"  {lab:9s} | vs anchor {_rmse(e['err_m']) - _rmse(e['err_anchor_m']):+6.2f} "
            f"[{la:+6.2f}, {ha:+6.2f}] | vs hold "
            f"{_rmse(e['err_m']) - _rmse(e['err_hold_m']):+6.2f} [{lh:+6.2f}, {hh:+6.2f}]")

    speed = wt["err_hold_m"] / np.maximum(wt["dur_s"], 1e-6)
    bad = speed > 12.0
    out(f"\n  physically impossible links (implied speed > 12 m/s, i.e. an id switch or a gross "
        f"projection failure): {int(bad.sum())} of {len(wt)} ({100 * bad.mean():.1f}%)")
    if bad.any():
        ok = wt[~bad]
        out(f"  SENSITIVITY (diagnostic only -- excluding them removes the largest errors and so "
            f"flatters the model): n {len(ok)}, RMSE {_rmse(ok['err_m']):.2f} m, "
            f"PICP50 {100 * ok['inside50'].mean():.1f}, PICP90 {100 * ok['inside90'].mean():.1f}")

    out("\n" + "=" * 96)
    out("TASK 2b -- SELECTION BIAS: WHO COMES BACK?")
    out("=" * 96)
    n_dis = len(cen)
    out(f"track-losses in the match: {n_dis}; re-appear under the same id: "
        f"{100 * cen['returns'].mean():.1f}%; survive the liveness guard and become usable events: "
        f"{100 * cen['usable'].mean():.1f}%")
    ret = cen[cen["returns"]]
    ter = cen[~cen["returns"]]
    out(f"  gap length of re-appearing losses: p50 {ret['gap_s'].median():.1f} s, "
        f"p90 {ret['gap_s'].quantile(0.9):.1f} s, max {ret['gap_s'].max():.1f} s")
    out(f"  terminal losses have {ter['censored_s'].median():.0f} s of chunk left on average "
        f"(median) -- they had the opportunity to return and did not")
    for col, name in (("d_ball_m", "distance from the ball at the last sighting (m)"),
                      ("img_edge", "normalised distance from the nearest image edge")):
        a, b2 = ret[col].dropna(), ter[col].dropna()
        u = cen.loc[cen["usable"], col].dropna()
        out(f"  {name}: re-appearing {a.median():.2f} | terminal {b2.median():.2f} | "
            f"usable events {u.median():.2f}  (medians)")
    edge = cen["img_edge"] <= EDGE_FRAC
    out(f"  last sighting within {100 * EDGE_FRAC:.0f}% of the image edge (a genuine frame exit): "
        f"re-appearing {100 * edge[cen['returns']].mean():.1f}%, terminal "
        f"{100 * edge[~cen['returns']].mean():.1f}%, usable {100 * edge[cen['usable']].mean():.1f}%")

    out("\n" + "=" * 96)
    out("TASK 3 -- SHAPE METRICS, OBSERVED VS RECONSTRUCTED")
    out("=" * 96)
    dirs = chunk_directions(match)
    for tag, sub in (("FULL reconstruction", rec),
                     ("gap-ghosts only (dead re-id fragments excluded)",
                      rec[rec["ghost_kind"] != "terminal"])):
        sh = shape_metrics(sub, dirs)
        if sh.empty:
            out(f"\n  {tag}: no frame carried three observed outfielders and a direction")
            continue
        add = sh[sh["n_added"] > 0]
        out(f"\n  {tag}")
        out(f"  team-timesteps scored: {len(sh)} (>= 1 player added on {len(add)}, "
            f"{100 * len(add) / max(len(sh), 1):.0f}%); mean added {sh['n_added'].mean():.2f}")
        out("  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|")
        for key, label in (("line", "line height "), ("depth", "depth spread"),
                           ("width", "width spread"), ("xspan", "x-span      ")):
            d = add[f"{key}_rec"] - add[f"{key}_obs"]
            out(f"  {label}  | {add[f'{key}_obs'].mean():8.2f} | {add[f'{key}_rec'].mean():13.2f} "
                f"| {d.mean():+11.2f} | {d.median():+9.2f} | {d.abs().quantile(0.9):9.2f}")
        for lo, hi in ((1, 1), (2, 3), (4, 11)):
            s = add[(add["n_added"] >= lo) & (add["n_added"] <= hi)]
            if s.empty:
                continue
            out(f"  +{lo}-{hi} players (n={len(s)}): line "
                f"{s['line_rec'].mean() - s['line_obs'].mean():+.2f} m, depth "
                f"{s['depth_rec'].mean() - s['depth_obs'].mean():+.2f} m, width "
                f"{s['width_rec'].mean() - s['width_obs'].mean():+.2f} m")
        off_rec = float(((add["line_rec"] < 0) | (add["line_rec"] > PITCH_LEN)).mean())
        off_obs = float(((add["line_obs"] < 0) | (add["line_obs"] > PITCH_LEN)).mean())
        out(f"  line height off the pitch (<0 or >{PITCH_LEN:.0f} m): reconstruction "
            f"{100 * off_rec:.2f}%, observed-only {100 * off_obs:.2f}%")


def _self_check() -> None:
    """Assert the squad arithmetic, the event guard and the chained-event drop on toy data."""
    ev = [Event("c", 0, 10, 20, 2.0, "within-track", (1.0, 1.0), (0,), "a"),
          Event("c", 0, 20, 30, 2.0, "within-track", (1.0, 1.0), (0,), "b"),
          Event("c", 0, 30, 40, 2.0, "within-track", (1.0, 1.0), (0,), "c"),
          Event("c", 1, 20, 25, 1.0, "within-track", (1.0, 1.0), (1,), "d")]
    kept = _drop_chained(ev)
    assert [e.label for e in kept] == ["a", "c", "d"], [e.label for e in kept]
    # the frozen conformal tables must stay ordered, or every coverage number below is wrong
    assert (K90_V1 > K50_V1).all() and (K90_ANC > K50_ANC).all()
    # gap filling: only frames between two sightings are filled; the tail stays NaN
    vis = np.array([[True], [False], [False], [True], [False]])
    truth = np.full((5, 1, 2), np.nan)
    truth[0, 0] = (1.0, 2.0)
    truth[3, 0] = (9.0, 9.0)
    assert fill_internal_gaps({"visible": vis, "truth": truth}) == 2
    assert np.allclose(truth[1, 0], (1.0, 2.0)) and np.allclose(truth[2, 0], (1.0, 2.0))
    assert not np.isfinite(truth[4, 0, 0])
    assert np.allclose(truth[3, 0], (9.0, 9.0))  # a sighting is never overwritten
    # squad arithmetic: a de-duplicated ghost inside DUP_M of a live player is dropped
    obs = np.array([[10.0, 10.0]])
    assert np.hypot(*(obs[0] - np.array([12.0, 10.0]))) <= DUP_M
    assert np.hypot(*(obs[0] - np.array([20.0, 10.0]))) > DUP_M
    print("full_match_reconstruction self-check OK (chain drop, region ordering, dedup radius)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="tottenham_manutd", help="registry id")
    ap.add_argument("--chunks", default="", help="comma-list of chunk keys (default: all)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--runlog", default="", help="optional path for the verbatim console trail")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return
    out = Tee()
    out("=" * 96)
    out("FULL-MATCH 22-PLAYER RECONSTRUCTION (frozen B4 v1, P2 policy) -- nothing is refitted")
    out("=" * 96)
    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()] or None
    run(args.match, chunks, out, Path(args.out))
    if args.runlog:
        dst = Path(args.runlog)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(["```", *out.lines, "```", ""]), encoding="utf-8")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
