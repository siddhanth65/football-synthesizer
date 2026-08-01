"""GTA-Link tracklet repair: appearance **Splitter** + **Connector** (training-free, CPU).

Port of Sun et al., *GTA: Global Tracklet Association for Multi-Object Tracking in Sports*
(ACCV 2024 W, arXiv:2411.08216) onto this repo's existing tracking output. Two passes over the
persisted positions table, no re-tracking and no training:

1. **Splitter** (:func:`split_track`, :func:`split_tracklets`) -- a single ``track_id`` that jumped
   between two players carries two appearance modes. DBSCAN (cosine) over the tracklet's
   *per-detection* embeddings exposes that bimodality; the tracklet is cut into temporally contiguous
   sub-tracklets at the cluster changes. Noise points inherit their nearest-in-time labelled
   neighbour, and runs shorter than ``min_run`` are absorbed, so only clear bimodality splits.
2. **Connector** (:func:`connect`) -- true agglomerative merging over the (split) tracklets. Distance
   is cosine distance between tracklet-**mean** embeddings, and the mean is recomputed after every
   merge (this is what separates it from the shipped single-linkage
   :func:`generator.track_relink.greedy_merge`, which keeps the original pair similarities). Merges
   are gated by hard constraints: no temporal overlap beyond ``overlap_slack`` frames, same team,
   same role, pitch-plane displacement reachable at ``<= max_speed_mps``, and distance ``< tau``.

The Splitter needs per-detection embeddings, which the shipped ``relink_cache_prtreid`` does **not**
hold (it stores one median vector per fragment). :func:`load_or_build_det_embeddings` builds and
caches them -- the only GPU stage in this module; everything else is CPU-pure and self-checked by
:func:`_demo`.

Validation is external: :mod:`eval.gsr_gta` re-scores the repaired submissions with the official
SoccerNet GS-HOTA evaluator. This module never touches the metric.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from generator.track_relink import (
    GSR_FPS,
    MAX_PLAYER_SPEED_MPS,
    POS_SLACK_M,
    Fragment,
    _frame_path,
)

logger = logging.getLogger("gta_link")

#: Version stamp written into caches and result payloads (bump when the algorithm changes).
GTA_VERSION = "gta-link-1.0"


@dataclass(frozen=True)
class GtaParams:
    """GTA-Link hyper-parameters.

    Attributes:
        eps: DBSCAN neighbourhood radius in **cosine distance** for the Splitter. Conservative
            (small) values split only on clear bimodality.
        min_samples: DBSCAN ``min_samples`` (a cluster must be at least this dense).
        min_run: Minimum number of sampled detections a sub-tracklet must keep; shorter runs are
            absorbed into their neighbour instead of becoming a fragment.
        tau: Connector merge ceiling in **cosine distance** (merge iff ``1 - cos < tau``).
        overlap_slack: Frames of temporal overlap tolerated between two merged tracklets. GTA allows
            1; we default to **0** because the SoccerNet evaluator rejects a submission outright if
            one track id appears twice in a single timestep ("Tracker predicts the same ID more than
            once in a single timestep"), which a 1-frame overlap produces.
        fps: Sequence frame rate (for the motion budget).
        max_speed_mps: Speed ceiling a player may use to close a gap between tracklets.
        pos_slack_m: Positional slack absorbing calibration/projection jitter at short gaps.
        frame_stride: Sample every Nth frame when building per-detection embeddings (GPU cost).
        match_tol_px: Max foot-point distance (px) to accept a re-detected box as the crop source.
    """

    eps: float = 0.30
    min_samples: int = 5
    min_run: int = 5
    tau: float = 0.04
    overlap_slack: int = 0
    fps: float = GSR_FPS
    max_speed_mps: float = MAX_PLAYER_SPEED_MPS
    pos_slack_m: float = POS_SLACK_M
    frame_stride: int = 2
    match_tol_px: float = 6.0


# === Splitter (pure) =============================================================================
def split_track(
    frames: np.ndarray, embs: np.ndarray, *, eps: float, min_samples: int, min_run: int,
) -> np.ndarray:
    """Label one tracklet's per-detection embeddings with sub-tracklet indices (pure).

    Runs DBSCAN in cosine distance over ``embs``; if two or more clusters survive, the samples are
    walked in time order and cut at every cluster change. Noise points (``-1``) first inherit the
    label of their nearest labelled neighbour **in time**, then runs shorter than ``min_run`` are
    absorbed into the preceding run (the first run absorbs backwards), so an isolated blip cannot
    manufacture a fragment.

    Args:
        frames: ``(N,)`` frame indices, ascending.
        embs: ``(N, D)`` L2-normalised per-detection embeddings aligned with ``frames``.
        eps: DBSCAN radius in cosine distance.
        min_samples: DBSCAN density requirement.
        min_run: Minimum samples per surviving sub-tracklet.

    Returns:
        ``(N,)`` int array of contiguous sub-tracklet indices starting at 0. All zeros means the
        tracklet is judged single-identity and must not be split.
    """
    n = len(frames)
    out = np.zeros(n, int)
    if n < 2 * max(min_samples, min_run):
        return out
    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embs)
    if len({int(v) for v in labels if v >= 0}) < 2:
        return out

    # Noise inherits its nearest labelled neighbour in time (forward then backward fill).
    lab = labels.astype(int).copy()
    known = np.flatnonzero(lab >= 0)
    if len(known) == 0:
        return out
    nearest = known[np.abs(frames[:, None] - frames[None, known]).argmin(axis=1)]
    lab = np.where(lab >= 0, lab, lab[nearest])

    runs: list[list[int]] = []  # each run = [label, count]
    for value in lab:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([int(value), 1])
    # Absorb short runs until every surviving run is long enough (or only one remains).
    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda i: runs[i][1])
        if runs[k][1] >= min_run:
            break
        j = k - 1 if k > 0 else 1
        runs[j][1] += runs[k][1]
        runs.pop(k)
        merged: list[list[int]] = []  # re-coalesce neighbours that now share a label
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][1] += r[1]
            else:
                merged.append(r)
        runs = merged
    if len(runs) < 2:
        return out
    pos = 0
    for idx, (_label, count) in enumerate(runs):
        out[pos:pos + count] = idx
        pos += count
    return out


def split_tracklets(
    df: pd.DataFrame, det: dict[int, tuple[np.ndarray, np.ndarray]], params: GtaParams,
) -> tuple[dict[int, list[tuple[int, int, int]]], dict]:
    """Split every contaminated tracklet -> ``{old_track_id: [(lo, hi, new_id), ...]}`` (pure).

    Segment ``(lo, hi, new_id)`` claims frames ``lo <= frame <= hi``; the first segment keeps the
    original track id so unsplit tracklets are untouched and the map is stable. Cut points sit at the
    midpoint between the last sample of one sub-tracklet and the first sample of the next, and the
    outer bounds are widened past the track's own span so every row is claimed.

    Args:
        df: Positions table (must carry ``frame``, ``track_id``, ``role``).
        det: ``{track_id: (frames, embeddings)}`` per-detection embeddings (frames ascending).
        params: GTA hyper-parameters.

    Returns:
        ``(splits, stats)`` -- only genuinely split tracks appear in ``splits``.
    """
    people = df[df["role"] != "ball"]
    next_id = int(people["track_id"].max()) + 1 if len(people) else 0
    splits: dict[int, list[tuple[int, int, int]]] = {}
    n_sub = 0
    for tid in sorted(det):
        frames, embs = det[tid]
        labels = split_track(frames, embs, eps=params.eps, min_samples=params.min_samples,
                             min_run=params.min_run)
        k = int(labels.max()) + 1
        if k < 2:
            continue
        bounds: list[tuple[int, int, int]] = []
        for s in range(k):
            idx = np.flatnonzero(labels == s)
            lo = -(1 << 30) if s == 0 else (
                int(frames[np.flatnonzero(labels == s - 1)[-1]]) + int(frames[idx[0]])) // 2 + 1
            hi = (1 << 30) if s == k - 1 else (
                int(frames[idx[-1]]) + int(frames[np.flatnonzero(labels == s + 1)[0]])) // 2
            bounds.append((lo, hi, int(tid) if s == 0 else next_id))
            if s > 0:
                next_id += 1
        splits[int(tid)] = bounds
        n_sub += k - 1
    return splits, {
        "n_tracks_embedded": len(det), "n_tracks_split": len(splits), "n_new_subtracks": n_sub,
    }


def apply_splits(
    df: pd.DataFrame, splits: dict[int, list[tuple[int, int, int]]],
) -> tuple[pd.DataFrame, dict[tuple[int, int], int]]:
    """Rewrite ``track_id`` per row according to ``splits`` -> ``(new_df, {(old_tid, frame): sub})``.

    The returned lookup is what the submission rewriter needs, because a split is a *per-row*
    relabelling: the same original track id maps to different sub-tracklets on different frames.
    """
    out = df.copy()
    if not splits:
        return out, {}
    tid = out["track_id"].to_numpy().astype(int)
    frame = out["frame"].to_numpy().astype(int)
    new = tid.copy()
    lookup: dict[tuple[int, int], int] = {}
    for old, bounds in splits.items():
        sel = np.flatnonzero(tid == old)
        for lo, hi, sub in bounds:
            hit = sel[(frame[sel] >= lo) & (frame[sel] <= hi)]
            new[hit] = sub
            for f in frame[hit]:
                lookup[(old, int(f))] = sub
    out["track_id"] = new
    return out, lookup


def split_detection_embeddings(
    det: dict[int, tuple[np.ndarray, np.ndarray]],
    splits: dict[int, list[tuple[int, int, int]]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Re-key per-detection embeddings onto the post-split track ids (pure)."""
    if not splits:
        return det
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tid, (frames, embs) in det.items():
        bounds = splits.get(tid)
        if bounds is None:
            out[tid] = (frames, embs)
            continue
        for lo, hi, sub in bounds:
            m = (frames >= lo) & (frames <= hi)
            if m.any():
                out[sub] = (frames[m], embs[m])
    return out


def mean_embeddings(
    det: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Tracklet-mean L2-normalised embedding and sample count per track id (pure)."""
    embs: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for tid, (_frames, e) in det.items():
        if len(e) == 0:
            continue
        m = e.mean(axis=0)
        embs[tid] = m / max(float(np.linalg.norm(m)), 1e-8)
        counts[tid] = int(len(e))
    return embs, counts


# === Connector (pure) ============================================================================
def _chain_ok(segs: list[tuple[int, int, tuple[float, float], tuple[float, float]]],
              params: GtaParams) -> bool:
    """Segments sorted by start: overlap within slack and every seam feasible at <= max_speed."""
    for p, q in zip(segs, segs[1:]):
        if p[1] - q[0] + 1 > params.overlap_slack:
            return False
        gap_s = max(q[0] - p[1], 0) / params.fps
        dist = float(np.hypot(p[3][0] - q[2][0], p[3][1] - q[2][1]))
        if dist > params.max_speed_mps * gap_s + params.pos_slack_m:
            return False
    return True


def connect(
    fragments: list[Fragment], embs: dict[int, np.ndarray], counts: dict[int, int],
    params: GtaParams,
) -> dict[int, int]:
    """Agglomerative tracklet connection under hard constraints -> ``{old_id: new_id}`` (pure).

    Repeatedly merges the globally closest admissible pair of components and **recomputes the merged
    component's mean embedding** (sample-count weighted, renormalised) before the next round, per the
    GTA connector. A pair is admissible iff both components share role and team, their union of
    segments forms a valid temporal chain (:func:`_chain_ok`), and their cosine distance is below
    ``params.tau``. Stops when no pair qualifies.

    Args:
        fragments: Tracklets to connect (post-split).
        embs: ``track_id -> L2-normalised mean embedding``. Tracklets absent here never merge.
        counts: ``track_id -> number of embedded detections`` (weights the running mean).
        params: GTA hyper-parameters.

    Returns:
        ``{old_track_id: new_track_id}`` for every fragment; each component takes its smallest id.
    """
    comps: dict[int, dict] = {}
    for f in fragments:
        e = embs.get(f.track_id)
        if e is None:
            continue
        comps[f.track_id] = {
            "segs": [(f.start_frame, f.end_frame, f.start_xy, f.end_xy)],
            "role": f.role, "team": f.team, "emb": e,
            "w": float(counts.get(f.track_id, 1)), "members": [f.track_id],
        }

    def pair(a: int, b: int) -> tuple[float, list] | None:
        """Cosine distance + merged segment chain for an admissible pair, else ``None``."""
        ca, cb = comps[a], comps[b]
        if ca["role"] != cb["role"] or ca["team"] != cb["team"]:
            return None
        d = 1.0 - float(np.dot(ca["emb"], cb["emb"]))
        if d >= params.tau:
            return None
        merged = sorted(ca["segs"] + cb["segs"])
        return (d, merged) if _chain_ok(merged, params) else None

    # ponytail: O(n^2) candidate table refreshed only for the merged component; n <= ~200
    # tracklets/sequence so this is milliseconds. Swap for a heap if sequences get much longer.
    cand: dict[tuple[int, int], tuple[float, list]] = {}
    keys = sorted(comps)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            got = pair(a, b)
            if got is not None:
                cand[(a, b)] = got
    while cand:
        (a, b), (_d, merged) = min(cand.items(), key=lambda kv: (kv[1][0], kv[0]))
        ca, cb = comps[a], comps[b]
        wa, wb = ca["w"], cb["w"]
        emb = ca["emb"] * wa + cb["emb"] * wb
        ca.update({"segs": merged, "emb": emb / max(float(np.linalg.norm(emb)), 1e-8),
                   "w": wa + wb, "members": ca["members"] + cb["members"]})
        del comps[b]
        cand = {k: v for k, v in cand.items() if a not in k and b not in k}
        for other in comps:
            if other == a:
                continue
            got = pair(*sorted((a, other)))
            if got is not None:
                cand[tuple(sorted((a, other)))] = got  # type: ignore[index]

    remap = {f.track_id: f.track_id for f in fragments}
    for c in comps.values():
        rep = min(c["members"])
        for tid in c["members"]:
            remap[tid] = rep
    return remap


# === Per-detection embeddings (GPU stage; cached) =================================================
def detection_embeddings(
    seq_dir: Path, df: pd.DataFrame, embedder, detector, *, params: GtaParams,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Embed every sampled detection of every tracklet by re-detecting the frames (GPU/IO).

    Frames are visited with stride ``params.frame_stride``; on each visited frame every person row is
    matched to the nearest re-detected box (within ``params.match_tol_px``) and that box is cropped.
    All crops of a frame are embedded in one batch.

    Args:
        seq_dir: Sequence directory holding ``img1/``.
        df: The sequence's positions table.
        embedder: An embedder exposing ``embed(crops) -> (N, D)`` L2-normalised features
            (:class:`tools.prtreid_probe.PrtreidEmbedder` or
            :class:`generator.track_relink.OsnetEmbedder`).
        detector: A ``generator.extract`` detector exposing ``detect(rgb) -> (xyxy, ...)``.
        params: GTA hyper-parameters (stride, match tolerance).

    Returns:
        ``{track_id: (frames, embeddings)}`` with ``frames`` ascending.
    """
    import cv2  # noqa: PLC0415

    people = df[(df["role"] != "ball") & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    per_tid: dict[int, list[tuple[int, np.ndarray]]] = {}
    frames = sorted(people["frame"].astype(int).unique())[::max(params.frame_stride, 1)]
    by_frame = {int(f): g for f, g in people.groupby("frame")}
    for frame_idx in frames:
        grp = by_frame.get(int(frame_idx))
        if grp is None:
            continue
        bgr = cv2.imread(str(_frame_path(seq_dir, int(frame_idx))))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xyxy = detector.detect(rgb)[0]
        if len(xyxy) == 0:
            continue
        foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]])
        crops: list[np.ndarray] = []
        owners: list[int] = []
        for r in grp.itertuples():
            d = np.hypot(foot[:, 0] - float(r.image_x), foot[:, 1] - float(r.image_y))
            k = int(np.argmin(d))
            if d[k] > params.match_tol_px:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy[k])
            crop = rgb[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
            if crop.size:
                crops.append(crop)
                owners.append(int(r.track_id))
        if not crops:
            continue
        feats = embedder.embed(crops)
        for tid, f in zip(owners, feats):
            per_tid.setdefault(tid, []).append((int(frame_idx), f))
    return {tid: (np.array([f for f, _ in v], int), np.stack([e for _, e in v]))
            for tid, v in per_tid.items() if v}


def load_or_build_det_embeddings(
    seq_dir: Path, df: pd.DataFrame, cache_path: Path, *, params: GtaParams, embedder=None,
    detector=None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-detection embeddings from ``cache_path`` (flat ``.npz``) or build and cache them.

    Raises:
        RuntimeError: If the cache is cold and no ``embedder`` was supplied (CPU-only callers).
    """
    if cache_path.exists():
        z = np.load(cache_path)
        tids, frames, embs = z["track_ids"].astype(int), z["frames"].astype(int), z["embeddings"]
        order = np.lexsort((frames, tids))
        tids, frames, embs = tids[order], frames[order], embs[order]
        cuts = np.flatnonzero(np.diff(tids)) + 1
        return {int(t[0]): (f, e) for t, f, e in
                zip(np.split(tids, cuts), np.split(frames, cuts), np.split(embs, cuts))}
    if embedder is None:
        raise RuntimeError(f"no per-detection embedding cache at {cache_path} and no embedder")
    det = detection_embeddings(seq_dir, df, embedder, detector, params=params)
    if det:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tids = np.concatenate([np.full(len(f), t, int) for t, (f, _) in sorted(det.items())])
        np.savez_compressed(
            cache_path, version=np.array([GTA_VERSION]), track_ids=tids,
            frames=np.concatenate([f for _t, (f, _e) in sorted(det.items())]),
            embeddings=np.concatenate([e for _t, (_f, e) in sorted(det.items())]),
        )
    return det


# === Purity diagnostics (GT-audited; used by the eval driver) ====================================
def tracklet_purity(gt_ids_by_row: dict[tuple[int, int], int]) -> dict:
    """Row-weighted tracklet purity against per-row GT ids -> pooled and per-tracklet stats.

    Args:
        gt_ids_by_row: ``{(track_id, frame): gt_track_id}`` for auditable rows. Keys carry the
            track ids of whichever partition is being audited (pre- or post-split).

    Returns:
        ``{"n_auditable", "n_rows", "n_dominant", "purity", "n_contaminated"}``; a tracklet counts as
        contaminated when a second GT id holds >= 20% of its audited rows and >= 10 rows.
    """
    votes: dict[int, Counter] = {}
    for (tid, _frame), gid in gt_ids_by_row.items():
        votes.setdefault(int(tid), Counter())[int(gid)] += 1
    n_rows = n_dom = n_contam = 0
    for c in votes.values():
        tot = sum(c.values())
        if tot < 5:
            continue
        ranked = c.most_common(2)
        n_rows += tot
        n_dom += ranked[0][1]
        if len(ranked) > 1 and ranked[1][1] >= 10 and ranked[1][1] / tot >= 0.20:
            n_contam += 1
    return {"n_auditable": len(votes), "n_rows": n_rows, "n_dominant": n_dom,
            "purity": n_dom / max(n_rows, 1), "n_contaminated": n_contam}


def _demo() -> None:
    """Self-check of the Splitter and Connector on synthetic embeddings (asserts; runnable)."""
    rng = np.random.default_rng(0)
    d = 16

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    a_dir, b_dir = np.zeros(d), np.zeros(d)
    a_dir[0], b_dir[1] = 1.0, 1.0
    frames = np.arange(60)
    embs = unit(np.where((frames < 30)[:, None], a_dir, b_dir) + 0.02 * rng.normal(size=(60, d)))
    params = GtaParams(eps=0.30, min_samples=5, min_run=5, tau=0.04)

    # --- Splitter: a clean identity switch at frame 30 is found; a pure tracklet is not split.
    labels = split_track(frames, embs, eps=params.eps, min_samples=params.min_samples,
                         min_run=params.min_run)
    assert labels.max() == 1, labels
    assert labels[:30].max() == 0 and labels[30:].min() == 1, labels
    pure = unit(a_dir + 0.02 * rng.normal(size=(60, d)))
    assert split_track(frames, pure, eps=params.eps, min_samples=params.min_samples,
                       min_run=params.min_run).max() == 0
    # A 3-sample blip is below min_run -> absorbed, no split.
    blip = embs.copy()
    blip[:] = unit(a_dir + 0.02 * rng.normal(size=(60, d)))
    blip[20:23] = unit(b_dir + 0.02 * rng.normal(size=(3, d)))
    assert split_track(frames, blip, eps=params.eps, min_samples=params.min_samples,
                       min_run=params.min_run).max() == 0

    df = pd.DataFrame({
        "frame": frames, "track_id": 7, "role": "player", "team": 0,
        "pitch_x": 10.0, "pitch_y": 10.0, "image_x": 0.0, "image_y": 0.0,
    })
    splits, stats = split_tracklets(df, {7: (frames, embs)}, params)
    assert stats["n_tracks_split"] == 1 and stats["n_new_subtracks"] == 1
    sdf, lookup = apply_splits(df, splits)
    assert set(sdf["track_id"]) == {7, 8}
    assert lookup[(7, 0)] == 7 and lookup[(7, 59)] == 8
    sub = split_detection_embeddings({7: (frames, embs)}, splits)
    assert len(sub[7][0]) == 30 and len(sub[8][0]) == 30

    # --- Connector: two disjoint reachable tracklets of one player merge; the impostor does not.
    f1 = Fragment(1, 0, 10, (10.0, 10.0), (12.0, 10.0), "player", 0, 11)
    f2 = Fragment(2, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 0, 11)
    f3 = Fragment(3, 30, 40, (14.5, 10.0), (16.0, 10.0), "player", 0, 11)  # same kit, other player
    e = {1: unit(a_dir + 0.01 * rng.normal(size=d)), 2: unit(a_dir + 0.01 * rng.normal(size=d)),
         3: unit(b_dir + 0.01 * rng.normal(size=d))}
    remap = connect([f1, f2, f3], e, {1: 10, 2: 10, 3: 10}, params)
    assert remap[2] == remap[1] == 1 and remap[3] == 3, remap
    # Motion gate: same appearance but 40 m away in 5 frames -> no merge.
    far = Fragment(2, 15, 25, (60.0, 10.0), (61.0, 10.0), "player", 0, 11)
    assert connect([f1, far], e, {1: 10, 2: 10}, params)[2] == 2
    # Overlap gate: at the default slack of 0 even a single shared frame blocks the merge, because
    # the GSR evaluator rejects any submission repeating a track id within one timestep.
    touch = Fragment(2, 10, 20, (12.0, 10.0), (13.0, 10.0), "player", 0, 11)
    assert connect([f1, touch], e, {1: 10, 2: 10}, params)[2] == 2
    assert connect([f1, touch], e, {1: 10, 2: 10},
                   GtaParams(**{**vars(params), "overlap_slack": 1}))[2] == 1
    olap = Fragment(2, 6, 20, (12.0, 10.0), (13.0, 10.0), "player", 0, 11)
    assert connect([f1, olap], e, {1: 10, 2: 10}, params)[2] == 2
    # Team gate: identical appearance, feasible motion, different team -> no merge.
    other = Fragment(2, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 1, 11)
    assert connect([f1, other], e, {1: 10, 2: 10}, params)[2] == 2
    print("gta_link demo OK: splitter cuts one switch (and no blip), connector honours all gates")


if __name__ == "__main__":
    _demo()
