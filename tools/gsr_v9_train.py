"""Campaign v9 W2: train + gate the metric-space TWiX associator, and cost the GTA splitter.

Four stages, all driven off the W1 dataset (``outputs/gsr/v9_assoc/v1/``, GSR **train** split only):

* ``--stage baseline`` -- run the incumbent GTA connector (tau 0.450, jersey gate, no splitter) on
  the 12 registered validation sequences and write its merge count, merge precision, fragments per
  GT identity, post-merge purity and its *implied pair ranking* (average precision of
  ``-clip_cos_dist`` with its hard gates applied). These are the a-priori bars for G1-G3.
* ``--stage windows`` -- materialise the pair-window tensors (cached ``.npz``).
* ``--stage train`` -- train :class:`generator.assoc_twix.TwixMetric` on 40 development sequences,
  early-stop on a 5-sequence inner-val slice, evaluate on the 12 registered validation sequences,
  over several seeds; ``--no-side`` is the registered kinematics-only ablation arm.
* ``--stage split`` -- measure :func:`generator.gta_link.split_tracklets` (and a per-row GT oracle
  splitter) against the W1 labels: split recall on contaminated tracklets, split precision, purity
  gain and the fragment cost.

Model selection never touches DEV-20/TEST-38/test-49/challenge; the validation slice is the one
registered in ``results/GSR_V9_W1.md`` S3.7 and stored in the dataset manifest.

CLI::

    python -m tools.gsr_v9_train --stage baseline
    python -m tools.gsr_v9_train --stage windows
    python -m tools.gsr_v9_train --stage train --seeds 0,1,2
    python -m tools.gsr_v9_train --stage split
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generator.assoc_twix import (
    ASSOC_VERSION,
    OBS_DIM,
    SIDE_DIM,
    V_SCALE,
    X_SCALE,
    Y_SCALE,
    TwixConfig,
    TwixMetric,
)
from generator.gta_link import GtaParams, _chain_ok
from tools.gsr_v9_factory import FactoryParams, summarise_tracklets

logger = logging.getLogger("gsr_v9_train")

#: Observations kept per side of a pair, and the frame stride they are sampled at (K=8 at stride 5
#: covers 1.4 s of context each side at 25 fps).
N_OBS = 8
OBS_STRIDE = 5
#: Inter-pair window: candidate pairs whose future tracklet starts in the same 125-frame (5 s) bin
#: are scored in each other's context (TWiX's ``inter_pair`` stage).
WIN_FRAMES = 125
#: Hard cap on pairs per window (only a handful of windows come close; larger ones are chunked).
MAX_PAIRS = 384
#: Every 9th development sequence is the inner-val slice used for early stopping, so the registered
#: 12-sequence validation slice is only ever *measured*, never selected on.
INNER_STRIDE = 9
#: The incumbent connector's shipped configuration (tools/gsr_v7_solver.py TAU, jersey gate on).
INCUMBENT_TAU = 0.450
#: Default of the on-record "no evidence" appearance distance (negative-pair median, W1 S4.3).
CLIP_MISSING = 0.6


@dataclass(frozen=True)
class TrainConfig:
    """Registered training knobs (fixed before any training run).

    Attributes:
        epochs: Passes over the training windows.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_groups: Windows per optimisation step.
        clip_grad: Gradient-norm clip.
        warmup_frac: Fraction of steps spent linearly warming the learning rate up.
    """

    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_groups: int = 8
    clip_grad: float = 1.0
    warmup_frac: float = 0.05


# === Window tensors ==============================================================================
def trajectories(rows: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-tracklet ``(frames, xy, velocity)`` over its finite-pitch rows (pure).

    Args:
        rows: One sequence's tracklet table (``frame, track_id, pitch_x, pitch_y``).

    Returns:
        ``{track_id: (frames (n,), xy (n, 2) in metres, v (n, 2) in m/s)}``; tracklets with fewer
        than two finite rows are omitted (they cannot carry a velocity).
    """
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fin = rows[np.isfinite(rows["pitch_x"]) & np.isfinite(rows["pitch_y"])]
    for tid, g in fin.groupby("track_id"):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy(np.float64)
        xy = np.column_stack([g["pitch_x"].to_numpy(), g["pitch_y"].to_numpy()])
        if len(f) < 2:
            continue
        t = f / FactoryParams.fps
        v = np.column_stack([np.gradient(xy[:, 0], t), np.gradient(xy[:, 1], t)])
        out[int(tid)] = (f, xy, v)
    return out


def side_features(pairs: pd.DataFrame) -> np.ndarray:
    """Pair-level side features -> ``(N, SIDE_DIM)`` (pure).

    Layout (registered in ``results/GSR_V9_W2.md`` S1): ``[team_same, team_unknown, role_same,
    jersey_match, jersey_conflict, log1p(min jersey mass)/3, clip cosine distance, clip missing,
    gap_s/10]``. Team/role/jersey are *features* here, never filters (kb v9-w1-004).

    Args:
        pairs: One or more sequences' candidate-pair rows.

    Returns:
        ``float32`` feature matrix.
    """
    ta, tb = pairs["team_a"].to_numpy(), pairs["team_b"].to_numpy()
    mass = np.minimum(pairs["num_mass_a"].to_numpy(), pairs["num_mass_b"].to_numpy())
    clip = pairs["clip_cos_dist"].to_numpy(np.float64)
    miss = ~np.isfinite(clip)
    agree = pairs["num_agree"].to_numpy()
    return np.column_stack([
        ((ta == tb) & (ta >= 0) & (tb >= 0)).astype(np.float32),
        ((ta < 0) | (tb < 0)).astype(np.float32),
        pairs["role_same"].to_numpy().astype(np.float32),
        (agree == 1).astype(np.float32),
        (agree == 0).astype(np.float32),
        (np.log1p(np.where(agree < 0, 0.0, mass)) / 3.0).astype(np.float32),
        np.where(miss, CLIP_MISSING, clip).astype(np.float32),
        miss.astype(np.float32),
        np.clip(pairs["gap_s"].to_numpy() / 10.0, 0.0, 4.0).astype(np.float32),
    ]).astype(np.float32)


def _obs_index(n: int, *, tail: bool) -> np.ndarray:
    """Chronological indices of the sampled observations at one end of a tracklet (pure)."""
    if tail:
        idx = np.arange(n - 1, -1, -OBS_STRIDE)[:N_OBS][::-1]
    else:
        idx = np.arange(0, n, OBS_STRIDE)[:N_OBS]
    return idx


def sequence_windows(rows: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Window tensors for one sequence's candidate pairs (pure).

    Each pair becomes ``2 * N_OBS`` observation tokens -- the tail of the past tracklet and the head
    of the future one -- in constant-normalised metric units, with signed time offsets referenced to
    the future tracklet's first observation (TWiX's convention).

    Args:
        rows: The sequence's tracklet table.
        pairs: The sequence's candidate-pair table.

    Returns:
        Arrays ``obs (N, 2*N_OBS, OBS_DIM) f32``, ``dt (N, 2*N_OBS) f32``,
        ``mask (N, 2*N_OBS) bool``, ``side (N, SIDE_DIM) f32``, ``label (N,) i8``,
        ``win (N,) i32`` (frame bin), ``tid_a``/``tid_b`` ``(N,) i32``.
    """
    traj = trajectories(rows)
    n, t = len(pairs), 2 * N_OBS
    obs = np.zeros((n, t, OBS_DIM), np.float32)
    dt = np.zeros((n, t), np.float32)
    mask = np.zeros((n, t), bool)
    for i, (ta, tb) in enumerate(zip(pairs["tid_a"].to_numpy(), pairs["tid_b"].to_numpy())):
        a, b = traj.get(int(ta)), traj.get(int(tb))
        if a is None or b is None:
            continue
        t0 = b[0][0]
        for half, (tr, is_tail) in enumerate(((a, True), (b, False))):
            idx = _obs_index(len(tr[0]), tail=is_tail)
            lo = half * N_OBS
            k = len(idx)
            obs[i, lo:lo + k, 0] = (tr[1][idx, 0] - X_SCALE) / X_SCALE
            obs[i, lo:lo + k, 1] = (tr[1][idx, 1] - Y_SCALE) / Y_SCALE
            obs[i, lo:lo + k, 2] = np.clip(tr[2][idx, 0] / V_SCALE, -2.0, 2.0)
            obs[i, lo:lo + k, 3] = np.clip(tr[2][idx, 1] / V_SCALE, -2.0, 2.0)
            obs[i, lo:lo + k, 4] = -1.0 if is_tail else 1.0
            dt[i, lo:lo + k] = (tr[0][idx] - t0) / FactoryParams.fps
            mask[i, lo:lo + k] = True
    return {
        "obs": obs, "dt": dt, "mask": mask, "side": side_features(pairs),
        "label": pairs["label"].to_numpy().astype(np.int8),
        "win": _win_from_traj(pairs, traj),
        "tid_a": pairs["tid_a"].to_numpy().astype(np.int32),
        "tid_b": pairs["tid_b"].to_numpy().astype(np.int32),
    }


def _win_from_traj(pairs: pd.DataFrame, traj: dict) -> np.ndarray:
    """Window bin of each pair, from the future tracklet's first observed frame (pure)."""
    first = {tid: int(v[0][0]) for tid, v in traj.items()}
    return np.array([first.get(int(b), 0) // WIN_FRAMES for b in pairs["tid_b"]], np.int32)


def build_windows(dest: Path, seqs: list[str]) -> dict[str, np.ndarray]:
    """Window tensors for a list of sequences, with a global group id per (sequence, window).

    Args:
        dest: Dataset directory (``outputs/gsr/v9_assoc/v1``).
        seqs: Sequence names.

    Returns:
        The concatenated arrays plus ``seq (N,) i16`` and ``group (N,) i32``.
    """
    parts: list[dict[str, np.ndarray]] = []
    for si, name in enumerate(seqs):
        rows = pd.read_parquet(dest / "tracklets" / f"{name}.parquet")
        pairs = pd.read_parquet(dest / "pairs" / f"{name}.parquet")
        if pairs.empty:
            continue
        w = sequence_windows(rows, pairs)
        w["seq"] = np.full(len(pairs), si, np.int16)
        parts.append(w)
        logger.info("windows %s: %d pairs", name, len(pairs))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    key = out["seq"].astype(np.int64) * 10_000 + out["win"].astype(np.int64)
    _, out["group"] = np.unique(key, return_inverse=True)
    out["group"] = out["group"].astype(np.int32)
    return out


# === Batching ====================================================================================
def group_slices(group: np.ndarray) -> list[np.ndarray]:
    """Row indices of each window group, chunked at :data:`MAX_PAIRS` (pure)."""
    order = np.argsort(group, kind="stable")
    cuts = np.flatnonzero(np.diff(group[order])) + 1
    out: list[np.ndarray] = []
    for chunk in np.split(order, cuts):
        for i in range(0, len(chunk), MAX_PAIRS):
            out.append(chunk[i:i + MAX_PAIRS])
    return out


def collate(arr: dict[str, np.ndarray], groups: list[np.ndarray],
            device: torch.device) -> tuple[torch.Tensor, ...]:
    """Pad a list of window groups into one batch -> ``(obs, dt, mask, side, pair_mask, label)``."""
    g, p, t = len(groups), max(len(x) for x in groups), 2 * N_OBS
    obs = np.zeros((g, p, t, OBS_DIM), np.float32)
    dt = np.zeros((g, p, t), np.float32)
    mask = np.zeros((g, p, t), bool)
    side = np.zeros((g, p, SIDE_DIM), np.float32)
    pmask = np.zeros((g, p), bool)
    label = np.full((g, p), -1, np.int8)
    for i, idx in enumerate(groups):
        k = len(idx)
        obs[i, :k], dt[i, :k], mask[i, :k] = arr["obs"][idx], arr["dt"][idx], arr["mask"][idx]
        side[i, :k], label[i, :k], pmask[i, :k] = arr["side"][idx], arr["label"][idx], True
    tt = lambda x, d: torch.as_tensor(x, dtype=d, device=device)  # noqa: E731
    return (tt(obs, torch.float32), tt(dt, torch.float32), tt(mask, torch.bool),
            tt(side, torch.float32), tt(pmask, torch.bool), tt(label, torch.int8))


# === Incumbent baseline ==========================================================================
def load_tracklets(dest: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(rows, pairs, tracklet summary)`` for one sequence, reusing the W1 factory summariser."""
    rows = pd.read_parquet(dest / "tracklets" / f"{name}.parquet")
    pairs = pd.read_parquet(dest / "pairs" / f"{name}.parquet")
    # The per-tracklet jersey evidence is recoverable from the pair table (both sides carry it), so
    # the percrop artifacts (cluster-side only) are not needed to reproduce the incumbent's gate.
    jersey: dict[int, dict] = {}
    for side in ("a", "b"):
        for tid, num, mass in zip(pairs[f"tid_{side}"], pairs[f"num_{side}"],
                                  pairs[f"num_mass_{side}"]):
            if int(num) >= 0:
                jersey[int(tid)] = {"number": int(num), "mass": float(mass), "share": 1.0,
                                    "n_reads": 1}
    tr = summarise_tracklets(rows, jersey, FactoryParams())
    return rows, pairs, tr


def det_embeddings(dest: Path, name: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-detection CLIP embeddings of one sequence from the mirrored cache."""
    z = np.load(dest / "detembed" / f"{name}.npz")
    tids, frames, embs = z["track_ids"].astype(int), z["frames"].astype(int), z["embeddings"]
    order = np.lexsort((frames, tids))
    tids, frames, embs = tids[order], frames[order], embs[order]
    cuts = np.flatnonzero(np.diff(tids)) + 1
    return {int(t[0]): (f, e) for t, f, e in
            zip(np.split(tids, cuts), np.split(frames, cuts), np.split(embs, cuts))}


def partition_stats(rows: pd.DataFrame, tr: pd.DataFrame, remap: dict[int, int]) -> dict:
    """Merge quality of a partition of the tracklets -> counts, precision, fragmentation, purity.

    Args:
        rows: The sequence's per-row table (carries ``gt_id`` per row).
        tr: Tracklet summary (carries the dominant ``gt_id`` per tracklet).
        remap: ``{track_id: component_id}``.

    Returns:
        ``{n_tracklets, n_components, n_merges, correct_pairs, total_pairs, frag_num, frag_den,
        pur_num, pur_den}`` -- pooled numerators/denominators so sequences can be summed.
    """
    from generator.gta_link import tracklet_purity  # noqa: PLC0415
    from generator.track_relink import merge_precision  # noqa: PLC0415

    keep = set(tr["track_id"].astype(int))
    remap = {k: v for k, v in remap.items() if k in keep}
    gt = {int(t): int(g) for t, g in zip(tr["track_id"], tr["gt_id"]) if int(g) >= 0}
    correct, total = merge_precision(remap, gt)
    comp_of_gt: dict[int, set[int]] = {}
    for tid, g in gt.items():
        comp_of_gt.setdefault(g, set()).add(remap.get(tid, tid))
    aud = rows[(rows["gt_id"] >= 0) & rows["track_id"].isin(keep)]
    by_row = {(remap.get(int(t), int(t)), int(f)): int(g)
              for t, f, g in zip(aud["track_id"], aud["frame"], aud["gt_id"])}
    pur = tracklet_purity(by_row)
    return {"n_tracklets": len(keep), "n_components": len(set(remap.values())),
            "n_merges": len(keep) - len(set(remap.values())),
            "correct_pairs": correct, "total_pairs": total,
            "frag_num": sum(len(v) for v in comp_of_gt.values()), "frag_den": len(comp_of_gt),
            "pur_num": pur["n_dominant"], "pur_den": pur["n_rows"]}


def incumbent_remap(rows: pd.DataFrame, tr: pd.DataFrame,
                    det: dict[int, tuple[np.ndarray, np.ndarray]], tau: float) -> dict[int, int]:
    """The shipped GTA connector (no splitter) on one sequence -> ``{track_id: component}``."""
    from generator.gta_link import connect, mean_embeddings  # noqa: PLC0415
    from generator.track_relink import summarize_fragments  # noqa: PLC0415

    keep = set(tr["track_id"].astype(int))
    frags = [f for f in summarize_fragments(rows) if f.track_id in keep]
    embs, counts = mean_embeddings({k: v for k, v in det.items() if k in keep})
    numbers = {int(t): int(n) for t, n in zip(tr["track_id"], tr["jersey_number"]) if int(n) >= 0}
    params = GtaParams(tau=tau, fps=FactoryParams.fps)
    remap = connect(frags, embs, counts, params, numbers=numbers)
    return {int(k): int(v) for k, v in remap.items()}


def incumbent_scores(pairs: pd.DataFrame) -> np.ndarray:
    """The connector's *implied* ranking of candidate pairs (pure).

    Its only continuous signal is appearance, so the ranking is ``-clip_cos_dist``; pairs its hard
    gates forbid (team, role, jersey conflict, missing embedding) are pushed below every admissible
    pair, which is exactly what the gates do at inference.
    """
    clip = pairs["clip_cos_dist"].to_numpy(np.float64)
    ta, tb = pairs["team_a"].to_numpy(), pairs["team_b"].to_numpy()
    blocked = ((ta != tb) | (pairs["role_same"].to_numpy() == 0)
               | (pairs["num_agree"].to_numpy() == 0) | ~np.isfinite(clip))
    return np.where(blocked, -1e3, -np.nan_to_num(clip, nan=1.0))


# === Score-ranked merging (the model's inference path) ============================================
def greedy_merges(tr: pd.DataFrame, pairs: pd.DataFrame, scores: np.ndarray,
                  n_merges: int) -> dict[int, int]:
    """Accept the top-scoring candidate pairs until ``n_merges`` merges are made (pure).

    Single-linkage over pair scores, with the only structural constraint a valid track must satisfy:
    the merged component's segments stay temporally disjoint and physically reachable
    (:func:`generator.gta_link._chain_ok`, the same test the incumbent applies). No team, role or
    jersey gate -- those are inside the model.

    Args:
        tr: Tracklet summary (frames + endpoints).
        pairs: Candidate pairs of the same sequence.
        scores: One score per pair row (higher = merge).
        n_merges: How many merges to make (matched to the incumbent's count).

    Returns:
        ``({track_id: component_id}, accepted)`` where ``accepted`` holds the row indices of the
        pairs that were actually merged (the diagnostic surface: which merges the model bought).
    """
    params = GtaParams(fps=FactoryParams.fps)
    seg = {int(r.track_id): [(int(r.start_frame), int(r.end_frame),
                             (float(r.start_x), float(r.start_y)),
                             (float(r.end_x), float(r.end_y)))] for r in tr.itertuples()}
    parent = {t: t for t in seg}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if len(accepted) >= n_merges:
            break
        a, b = int(pairs["tid_a"].iloc[i]), int(pairs["tid_b"].iloc[i])
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        merged = sorted(seg[ra] + seg[rb])
        if not _chain_ok(merged, params):
            continue
        parent[rb] = ra
        seg[ra] = merged
        accepted.append(int(i))
    return {t: find(t) for t in parent}, accepted


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision over the labelled pairs only (``label`` in ``{0, 1}``)."""
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    m = labels >= 0
    return float(average_precision_score(labels[m], scores[m]))


# === Stages ======================================================================================
def stage_baseline(dest: Path, out: Path, tau: float = INCUMBENT_TAU) -> dict:
    """Measure the incumbent connector on the registered validation slice (a-priori gate bars)."""
    man = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    val = man["split"]["val_slice"]
    per, labels, scores, merges = [], [], [], {}
    for name in val:
        rows, pairs, tr = load_tracklets(dest, name)
        remap = incumbent_remap(rows, tr, det_embeddings(dest, name), tau)
        st = partition_stats(rows, tr, remap)
        st["seq"] = name
        per.append(st)
        merges[name] = st["n_merges"]
        s = incumbent_scores(pairs)
        labels.append(pairs["label"].to_numpy())
        scores.append(s)
        logger.info("%s: %d tracklets, %d merges, prec %.4f", name, st["n_tracklets"],
                    st["n_merges"], st["correct_pairs"] / max(st["total_pairs"], 1))
    res = {"tau": tau, "val_slice": val, "per_seq": per, "merges_per_seq": merges,
           "pooled": pool(per),
           "pair_ap": average_precision(np.concatenate(labels), np.concatenate(scores))}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    p = res["pooled"]
    print(f"INCUMBENT tau {tau}: merges {p['n_merges']}  merge-precision {p['precision']:.4f}  "
          f"frag/GT {p['frag_per_gt']:.3f}  purity {p['purity']:.4f}  "
          f"pair-AP {res['pair_ap']:.4f}")
    return res


def pool(per: list[dict]) -> dict:
    """Sum per-sequence numerators into pooled merge precision / fragmentation / purity."""
    s = {k: sum(int(d[k]) for d in per) for k in
         ("n_tracklets", "n_components", "n_merges", "correct_pairs", "total_pairs",
          "frag_num", "frag_den", "pur_num", "pur_den")}
    return {**s, "precision": s["correct_pairs"] / max(s["total_pairs"], 1),
            "frag_per_gt": s["frag_num"] / max(s["frag_den"], 1),
            "purity": s["pur_num"] / max(s["pur_den"], 1)}


def evaluate(model: TwixMetric, dest: Path, seqs: list[str], merges: dict[str, int],
             device: torch.device) -> dict:
    """Score every candidate pair of ``seqs``, then merge at the incumbent's matched count."""
    model.eval()
    per, labels, scores = [], [], []
    for name in seqs:
        rows, pairs, tr = load_tracklets(dest, name)
        w = sequence_windows(rows, pairs)
        w["group"] = w["win"].astype(np.int32)
        s = predict(model, w, device)
        labels.append(pairs["label"].to_numpy())
        scores.append(s)
        remap, _acc = greedy_merges(tr, pairs, s, merges[name])
        st = partition_stats(rows, tr, remap)
        st["seq"] = name
        per.append(st)
    return {"per_seq": per, "pooled": pool(per),
            "pair_ap": average_precision(np.concatenate(labels), np.concatenate(scores))}


def predict(model: TwixMetric, arr: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    """Merge logits for every row of ``arr``, evaluated window by window."""
    out = np.zeros(len(arr["label"]), np.float32)
    with torch.no_grad():
        for idx in group_slices(arr["group"]):
            obs, dt, mask, side, pmask, _lab = collate(arr, [idx], device)
            out[idx] = model(obs, dt, mask, side, pmask)[0, :len(idx)].float().cpu().numpy()
    return out


def train_one(arr: dict[str, np.ndarray], inner: dict[str, np.ndarray], cfg: TrainConfig,
              tcfg: TwixConfig, seed: int, device: torch.device) -> tuple[TwixMetric, dict]:
    """Train one seed, early-stopping on the inner-val slice's pair AP.

    Args:
        arr: Training windows.
        inner: Inner-val windows (never the registered validation slice).
        cfg: Registered training knobs.
        tcfg: Architecture config (``use_side=False`` is the ablation arm).
        seed: Torch/NumPy seed.
        device: Compute device.

    Returns:
        ``(best_model, history)``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwixMetric(tcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    groups = group_slices(arr["group"])
    steps = cfg.epochs * math.ceil(len(groups) / cfg.batch_groups)
    warm = max(int(cfg.warmup_frac * steps), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1.0) * 0.5 * (
        1 + math.cos(math.pi * min(s / max(steps, 1), 1.0))))
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, best_ap, hist = None, -1.0, []
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(groups))
        tot, nb = 0.0, 0
        for i in range(0, len(order), cfg.batch_groups):
            batch = [groups[j] for j in order[i:i + cfg.batch_groups]]
            obs, dt, mask, side, pmask, lab = collate(arr, batch, device)
            logit = model(obs, dt, mask, side, pmask)
            sup = pmask & (lab >= 0)
            if not sup.any():
                continue
            loss = (lossf(logit, (lab > 0).float()) * sup).sum() / sup.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        ap = average_precision(inner["label"].astype(int), predict(model, inner, device))
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "inner_ap": ap})
        logger.info("seed %d epoch %d: loss %.4f inner-AP %.4f", seed, ep, tot / max(nb, 1), ap)
        if ap > best_ap:
            best_ap, best = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, {"best_inner_ap": best_ap, "history": hist}


def stage_train(dest: Path, out: Path, seeds: list[int], cfg: TrainConfig, tcfg: TwixConfig,
                baseline: Path, device: torch.device, ckpt_dir: Path | None = None) -> dict:
    """Train every seed and report the registered gate quantities on the validation slice."""
    man = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    val = man["split"]["val_slice"]
    dev = man["split"]["train_slice"]
    inner_names = dev[::INNER_STRIDE]
    train_names = [s for s in dev if s not in set(inner_names)]
    base = json.loads(baseline.read_text(encoding="utf-8"))
    logger.info("train %d seqs, inner-val %d, val %d", len(train_names), len(inner_names), len(val))
    tr_arr = build_windows(dest, train_names)
    in_arr = build_windows(dest, inner_names)
    runs = []
    for seed in seeds:
        t0 = time.time()
        model, hist = train_one(tr_arr, in_arr, cfg, tcfg, seed, device)
        ev = evaluate(model, dest, val, base["merges_per_seq"], device)
        runs.append({"seed": seed, "minutes": (time.time() - t0) / 60.0,
                     "best_inner_ap": hist["best_inner_ap"], "val_ap": ev["pair_ap"],
                     "val": ev["pooled"], "history": hist["history"]})
        p = ev["pooled"]
        print(f"seed {seed}: val pair-AP {ev['pair_ap']:.4f}  merges {p['n_merges']}  "
              f"precision {p['precision']:.4f}  frag/GT {p['frag_per_gt']:.3f}  "
              f"purity {p['purity']:.4f}  ({(time.time() - t0) / 60:.1f} min)")
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"version": ASSOC_VERSION, "cfg": asdict(tcfg), "seed": seed,
                        "state_dict": model.state_dict()},
                       ckpt_dir / f"twixm_{'side' if tcfg.use_side else 'kin'}_s{seed}.pt")
    res = {"version": ASSOC_VERSION, "train_cfg": asdict(cfg), "model_cfg": asdict(tcfg),
           "train_seqs": train_names, "inner_val": inner_names, "val": val,
           "incumbent": base["pooled"], "incumbent_pair_ap": base["pair_ap"], "runs": runs}
    res["summary"] = {"val_ap": seed_stats([r["val_ap"] for r in runs]),
                      **{k: seed_stats([r["val"][k] for r in runs])
                         for k in ("precision", "frag_per_gt", "purity")}}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    s = res["summary"]
    print(f"POOLED over {len(seeds)} seeds: pair-AP {s['val_ap']['mean']:.4f} "
          f"+-{s['val_ap']['spread']:.4f} (incumbent {base['pair_ap']:.4f}); "
          f"precision {s['precision']['mean']:.4f} +-{s['precision']['spread']:.4f} "
          f"(incumbent {base['pooled']['precision']:.4f}); "
          f"frag/GT {s['frag_per_gt']['mean']:.3f} (incumbent {base['pooled']['frag_per_gt']:.3f})")
    return res


def seed_stats(values: list[float]) -> dict:
    """Mean, standard deviation and max-min spread of a seed sweep (the noise floor)."""
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "spread": float(a.max() - a.min()), "values": [float(v) for v in a]}


# === Splitter costing ============================================================================
def split_quality(rows: pd.DataFrame, tr: pd.DataFrame,
                  splits: dict[int, list[tuple[int, int, int]]]) -> dict:
    """Did the splitter cut the contaminated tracklets, and what did the cuts cost? (pure).

    Args:
        rows: Per-row table with ``gt_id``.
        tr: Tracklet summary (carries ``purity`` over audited rows).
        splits: :func:`generator.gta_link.split_tracklets` output.

    Returns:
        Pooled counters: contaminated/pure tracklet counts, how many of each were split,
        row-weighted
        purity before/after, and the number of sub-tracklets created.
    """
    from generator.gta_link import apply_splits, tracklet_purity  # noqa: PLC0415

    keep = set(tr["track_id"].astype(int))
    pur = {int(t): float(p) for t, p in zip(tr["track_id"], tr["purity"])}
    aud = rows[(rows["gt_id"] >= 0) & rows["track_id"].isin(keep)]
    before = tracklet_purity({(int(t), int(f)): int(g)
                              for t, f, g in zip(aud["track_id"], aud["frame"], aud["gt_id"])})
    sdf, _lookup = apply_splits(rows[rows["track_id"].isin(keep)], splits)
    saud = sdf[sdf["gt_id"] >= 0]
    after = tracklet_purity({(int(t), int(f)): int(g)
                             for t, f, g in zip(saud["track_id"], saud["frame"], saud["gt_id"])})
    impure = {t for t, p in pur.items() if np.isfinite(p) and p < 0.8}  # noqa: PLR2004
    clean = {t for t, p in pur.items() if np.isfinite(p) and p >= 0.99}  # noqa: PLR2004
    split_ids = {int(t) for t in splits if int(t) in keep}
    return {"n_tracklets": len(keep), "n_impure": len(impure), "n_clean": len(clean),
            "n_split": len(split_ids), "n_impure_split": len(impure & split_ids),
            "n_clean_split": len(clean & split_ids),
            "n_subtracks": sum(len(b) - 1 for t, b in splits.items() if int(t) in keep),
            "pur_num_before": before["n_dominant"], "pur_den_before": before["n_rows"],
            "pur_num_after": after["n_dominant"], "pur_den_after": after["n_rows"],
            "contam_before": before["n_contaminated"], "contam_after": after["n_contaminated"]}


def oracle_splits(rows: pd.DataFrame, tr: pd.DataFrame,
                  min_run: int = 5) -> dict[int, list[tuple[int, int, int]]]:
    """Per-row GT oracle splitter: cut wherever the audited GT identity changes (pure).

    Runs shorter than ``min_run`` audited rows are absorbed, so the oracle is not allowed to shatter
    a tracklet on single-frame label noise -- the same courtesy the DBSCAN splitter gets.
    """
    keep = set(tr["track_id"].astype(int))
    aud = rows[(rows["gt_id"] >= 0) & rows["track_id"].isin(keep)]
    aud = aud.sort_values(["track_id", "frame"])
    next_id = int(rows["track_id"].max()) + 1
    out: dict[int, list[tuple[int, int, int]]] = {}
    for tid, g in aud.groupby("track_id"):
        f = g["frame"].to_numpy(int)
        gid = g["gt_id"].to_numpy(int)
        runs: list[list] = []
        for i, v in enumerate(gid):
            if runs and runs[-1][0] == v:
                runs[-1][2] = i
                runs[-1][3] += 1
            else:
                runs.append([int(v), i, i, 1])
        while len(runs) > 1:
            k = min(range(len(runs)), key=lambda i: runs[i][3])
            if runs[k][3] >= min_run:
                break
            j = k - 1 if k > 0 else 1
            runs[j][1], runs[j][2] = min(runs[j][1], runs[k][1]), max(runs[j][2], runs[k][2])
            runs[j][3] += runs[k][3]
            runs.pop(k)
            merged: list[list] = []
            for r in runs:
                if merged and merged[-1][0] == r[0]:
                    merged[-1][2], merged[-1][3] = r[2], merged[-1][3] + r[3]
                else:
                    merged.append(r)
            runs = merged
        if len(runs) < 2:
            continue
        bounds = []
        for s, r in enumerate(runs):
            lo = -(1 << 30) if s == 0 else (int(f[runs[s - 1][2]]) + int(f[r[1]])) // 2 + 1
            hi = (1 << 30) if s == len(runs) - 1 else (int(f[r[2]]) + int(f[runs[s + 1][1]])) // 2
            bounds.append((lo, hi, int(tid) if s == 0 else next_id))
            if s > 0:
                next_id += 1
        out[int(tid)] = bounds
    return out


def stage_split(dest: Path, out: Path, eps_list: list[float]) -> dict:
    """Cost the incumbent splitter (and the GT oracle splitter) on the whole W1 label set."""
    from generator.gta_link import split_tracklets  # noqa: PLC0415

    man = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(man["split"]["val_slice"] + man["split"]["train_slice"])
    arms: dict[str, list[dict]] = {f"dbscan_eps{e}": [] for e in eps_list}
    arms["oracle"] = []
    for name in names:
        if not (dest / "detembed" / f"{name}.npz").exists():
            logger.warning("%s: no per-detection embeddings, skipped", name)
            continue
        rows, _pairs, tr = load_tracklets(dest, name)
        det = det_embeddings(dest, name)
        keep = set(tr["track_id"].astype(int))
        det = {k: v for k, v in det.items() if k in keep}
        for e in eps_list:
            sp, _st = split_tracklets(rows, det, GtaParams(eps=e))
            arms[f"dbscan_eps{e}"].append({**split_quality(rows, tr, sp), "seq": name})
        arms["oracle"].append({**split_quality(rows, tr, oracle_splits(rows, tr)), "seq": name})
        logger.info("split %s done", name)
    res = {"n_sequences": len(arms["oracle"]),
           "arms": {k: {"per_seq": v, "pooled": pool_split(v)} for k, v in arms.items()}}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res["arms"].items():
        p = v["pooled"]
        print(f"{k}: split {p['n_split']}/{p['n_tracklets']} tracklets "
              f"(impure hit {p['impure_recall']:.3f}, of splits {p['split_precision']:.3f} "
              f"were impure), purity {p['purity_before']:.4f} -> {p['purity_after']:.4f}, "
              f"+{p['n_subtracks']} fragments, contaminated {p['contam_before']} -> "
              f"{p['contam_after']}")
    return res


def pool_split(per: list[dict]) -> dict:
    """Pool per-sequence splitter counters into recall/precision/purity."""
    s = {k: sum(int(d[k]) for d in per) for k in
         ("n_tracklets", "n_impure", "n_clean", "n_split", "n_impure_split", "n_clean_split",
          "n_subtracks", "pur_num_before", "pur_den_before", "pur_num_after", "pur_den_after",
          "contam_before", "contam_after")}
    return {**s, "impure_recall": s["n_impure_split"] / max(s["n_impure"], 1),
            "split_precision": s["n_impure_split"] / max(s["n_split"], 1),
            "clean_split_rate": s["n_clean_split"] / max(s["n_clean"], 1),
            "purity_before": s["pur_num_before"] / max(s["pur_den_before"], 1),
            "purity_after": s["pur_num_after"] / max(s["pur_den_after"], 1)}


# === Self-check ==================================================================================
def _demo() -> None:
    """Deterministic self-check of the window builder, the merger and the oracle splitter."""
    rows = pd.DataFrame({
        "frame": list(range(0, 20)) + list(range(30, 50)),
        "track_id": [1] * 20 + [2] * 20,
        "role": "player", "team": 0, "conf": 0.9, "calib_error_m": 0.1,
        "pitch_x": np.concatenate([np.linspace(10, 14, 20), np.linspace(16, 20, 20)]),
        "pitch_y": 34.0, "image_x": 0.0, "image_y": 0.0,
        "gt_id": [5] * 20 + [5] * 20,
    })
    pairs = pd.DataFrame([{
        "tid_a": 1, "tid_b": 2, "gap_frames": 11, "gap_s": 0.44, "label": 1,
        "team_a": 0, "team_b": 0, "team_same": 1, "role_same": 1, "num_agree": 1,
        "num_mass_a": 2.0, "num_mass_b": 2.0, "clip_cos_dist": 0.1,
    }])
    w = sequence_windows(rows, pairs)
    assert w["obs"].shape == (1, 2 * N_OBS, OBS_DIM), w["obs"].shape
    assert w["mask"].sum() == 2 * min(N_OBS, math.ceil(20 / OBS_STRIDE)), w["mask"].sum()
    # Past tokens carry the -1 side flag and negative time offsets; future tokens the opposite.
    assert (w["obs"][0, :4, 4] == -1.0).all() and (w["obs"][0, N_OBS:N_OBS + 4, 4] == 1.0).all()
    # Time is referenced to the future tracklet's first observation; the past side is negative.
    past = np.flatnonzero(w["mask"][0, :N_OBS])
    assert w["dt"][0, N_OBS] == 0.0 and (w["dt"][0, past] < 0).all(), w["dt"][0]
    # Constant normalisation, not per-window: x = 14 m must land at (14 - 52.5) / 52.5.
    assert abs(w["obs"][0, past[-1], 0] - (14.0 - X_SCALE) / X_SCALE) < 1e-6
    assert side_features(pairs).shape == (1, SIDE_DIM)

    tr = summarise_tracklets(rows, {}, FactoryParams())
    remap, acc = greedy_merges(tr, pairs, np.array([1.0]), 1)
    assert acc == [0], acc
    assert len(set(remap.values())) == 1, remap
    assert greedy_merges(tr, pairs, np.array([1.0]), 0)[0] == {1: 1, 2: 2}
    st = partition_stats(rows, tr, remap)
    assert st["n_merges"] == 1 and st["correct_pairs"] == 1 and st["total_pairs"] == 1, st

    # Oracle splitter: one tracklet that is two identities gets cut exactly once.
    mixed = rows.copy()
    mixed["track_id"] = 1
    mixed["gt_id"] = [5] * 20 + [6] * 20
    tr2 = summarise_tracklets(mixed, {}, FactoryParams())
    sp = oracle_splits(mixed, tr2)
    assert list(sp) == [1] and len(sp[1]) == 2, sp
    q = split_quality(mixed, tr2, sp)
    assert q["n_impure"] == 1 and q["n_impure_split"] == 1, q
    assert q["pur_num_after"] == q["pur_den_after"], q
    print("gsr_v9_train self-check OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("outputs/gsr/v9_assoc/v1"))
    ap.add_argument("--results", type=Path, default=Path("outputs/gsr/v9_assoc"))
    ap.add_argument("--stage", choices=("baseline", "windows", "train", "split", "demo"),
                    default="demo")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--no-side", action="store_true", help="registered ablation: kinematics only")
    ap.add_argument("--no-inter-pair", action="store_true", help="drop TWiX's cross-pair stage")
    ap.add_argument("--eps", default="0.20,0.30,0.40")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if args.stage == "demo":
        _demo()
    elif args.stage == "baseline":
        stage_baseline(args.dest, args.results / "baseline_incumbent.json")
    elif args.stage == "windows":
        w = build_windows(args.dest, sorted(
            json.loads((args.dest / "manifest.json").read_text(encoding="utf-8"))
            ["split"]["val_slice"]))
        print(f"windows {w['obs'].shape}, groups {len(np.unique(w['group']))}, "
              f"max group {np.bincount(w['group']).max()}")
    elif args.stage == "train":
        tcfg = TwixConfig(use_side=not args.no_side, inter_pair=not args.no_inter_pair)
        tag = args.tag or ("kin" if args.no_side else "side")
        stage_train(args.dest, args.results / f"train_{tag}.json",
                    [int(s) for s in args.seeds.split(",")],
                    TrainConfig(epochs=args.epochs), tcfg,
                    args.results / "baseline_incumbent.json", torch.device(args.device),
                    ckpt_dir=args.results / "ckpts")
    else:
        stage_split(args.dest, args.results / "splitter_cost.json",
                    [float(e) for e in args.eps.split(",")])


if __name__ == "__main__":
    main()
