"""Clean-room ExpansionIoU association: re-link the cached GSR detections without a Kalman filter.

Campaign v6 session S5. The association math is implemented **from the paper only** -- Huang et al.,
*Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking in Sports*
(arXiv:2306.13074). The reference implementation carries no license file, so nothing was read from
it; the three ideas taken from the paper are:

1. **ExpansionIoU** -- scale both boxes about their centres by ``1 + e`` before computing IoU, so a
   pair that a constant-velocity prior would need a motion model to link is linked by geometry alone.
2. **Iterative scale-up** -- associate in rounds, raising ``e`` for whatever is still unmatched, so
   an easy match is never spent on a loose gate.
3. **Deep features** -- a per-track appearance state (EMA over the matched detections' embeddings)
   fused into the assignment cost. Both cached embedding spaces are supported through the existing
   ``GSR_EMBEDDER`` mechanism (:data:`eval.gsr_gta.EMBEDDER`).

There is **no Kalman filter**: a track's matching geometry is its last *observed* box.

Two deviations from the paper, both forced by what our cache holds, both measured rather than
assumed:

* **The cached rows carry no bounding box.** ``generator.extract``'s positions schema persists only
  the foot point ``(image_x, image_y)`` and ``conf``. ``--build-boxes`` therefore re-runs the same
  football detector at stride 1 and attaches each positions row to its own detection box (greedy
  nearest foot point, the ``match_tol_px`` rule :func:`generator.gta_link.detection_embeddings`
  already uses); rows with no match fall back to
  :func:`generator.team_anchor.estimate_player_box`. The fallback share is reported per sequence.
* **ByteTrack's low-score second association is omitted.** Our cached rows are already ByteTrack's
  *output*; every one of them must receive a track id, because dropping a row would move GS-DetA and
  destroy the like-for-like against the control. A confidence split cannot do its job here, so it is
  not implemented.

The output contract is what matters downstream, not the tracker internals: this module rewrites
``track_id`` on the cached artifacts the frozen v4/v5 chain consumes -- the positions parquet, the
per-detection embedding npz and the per-crop OCR parquet -- into ``*_eiou`` variant directories, and
then runs :func:`tools.gsr_v4.solve_v4_arm` (connector + solver unchanged) over them.

CLI::

    python -m tools.gsr_eiou --demo                          # deterministic self-check
    python -m tools.gsr_eiou --build-boxes --split dev       # GPU: per-detection box cache
    python -m tools.gsr_eiou --dev --embedder clip           # DEV-20 sweep, one embedder arm
    python -m tools.gsr_eiou --t38 --embedder clip           # the ONE TEST-38 component check
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_eiou")

#: Version stamp for the association (bump when the math changes).
EIOU_VERSION = "eiou-1.0"
#: Per-detection box cache (built by ``--build-boxes``; the only GPU stage in this module).
BOX_SUBDIR = "detbox_cache"
#: Variant suffix every EIoU artifact directory carries, so nothing on-record can mix.
VARIANT = "_eiou"
#: The frozen v4/v5 downstream arm: 0.80 aggregation floor, jersey-compatible merge gate, no refit.
FLOOR = "0.80"
#: Connector tau per embedder -- the on-record frozen value for each arm (GSR_V4.md / GSR_V5.md).
CONTROL_TAU = {"prtreid": 0.080, "clip": 0.450}


# === Parameters ==================================================================================
@dataclass(frozen=True)
class EiouParams:
    """Clean-room Deep-EIoU association hyper-parameters.

    Attributes:
        e: Expansion scale of the **first** association round. Both boxes are scaled about their
            centres by ``1 + e``, so ``e = 0`` is plain IoU.
        e_step: Added to ``e`` for each subsequent scale-up round.
        rounds: Number of association rounds per frame (``1`` disables the scale-up).
        w_app: Weight of the appearance term in the fused cost, in ``[0, 1]``. ``0`` is geometry
            only.
        app_max: Appearance gate and normaliser, in **cosine distance**. A pair further apart than
            this never matches, and the appearance cost is ``dist / app_max`` clipped to ``[0, 1]``
            -- which is what makes ``w_app`` transferable between embedding spaces whose absolute
            scales differ by ~4.8x (results/CLUSTER_SESSION4.md section 3).
        ema: Appearance-state momentum: ``feat <- ema * feat + (1 - ema) * observed``, renormalised.
        max_lost: Frames a track survives unmatched before it can no longer be revived.
        match_tol_px: Foot-point tolerance when attaching a detector box to a positions row.
    """

    e: float = 0.7
    e_step: float = 0.35
    rounds: int = 3
    w_app: float = 0.5
    app_max: float = 0.30
    ema: float = 0.9
    max_lost: int = 30
    match_tol_px: float = 6.0


# === ExpansionIoU (pure) =========================================================================
def expand_boxes(boxes: np.ndarray, e: float) -> np.ndarray:
    """Scale each ``(x1, y1, x2, y2)`` box about its centre by ``1 + e`` (pure).

    Args:
        boxes: ``(N, 4)`` boxes in ``xyxy``.
        e: Expansion scale; ``0`` returns the boxes unchanged.

    Returns:
        ``(N, 4)`` expanded boxes.
    """
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
    hw = (b[:, 2] - b[:, 0]) * (1.0 + e) / 2.0
    hh = (b[:, 3] - b[:, 1]) * (1.0 + e) / 2.0
    return np.column_stack([cx - hw, cy - hh, cx + hw, cy + hh])


def eiou_matrix(a: np.ndarray, b: np.ndarray, e: float) -> np.ndarray:
    """ExpansionIoU between two box sets -> ``(len(a), len(b))`` (pure).

    Both sides are expanded by :func:`expand_boxes` before a standard IoU, so a pair separated by
    more than a box width can still score above zero without any motion model.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ea, eb = expand_boxes(a, e), expand_boxes(b, e)
    x1 = np.maximum(ea[:, None, 0], eb[None, :, 0])
    y1 = np.maximum(ea[:, None, 1], eb[None, :, 1])
    x2 = np.minimum(ea[:, None, 2], eb[None, :, 2])
    y2 = np.minimum(ea[:, None, 3], eb[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (ea[:, 2] - ea[:, 0]) * (ea[:, 3] - ea[:, 1])
    area_b = (eb[:, 2] - eb[:, 0]) * (eb[:, 3] - eb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _match(cost: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian assignment over ``cost``, keeping only pairs ``valid`` allows (pure)."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    if cost.size == 0 or not valid.any():
        return []
    big = float(cost[valid].max()) + 1.0 if valid.any() else 1.0
    rows, cols = linear_sum_assignment(np.where(valid, cost, big * 10.0))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]


def associate(
    frames: np.ndarray, boxes: np.ndarray, embs: np.ndarray | None, has_emb: np.ndarray | None,
    params: EiouParams,
) -> np.ndarray:
    """Re-link per-frame detections into tracks -> one new track id per input row (pure).

    Online, frame by frame, with no motion model: a track's matching geometry is its last observed
    box. Each frame runs ``params.rounds`` association rounds at expansion scales
    ``e, e + e_step, ...``; whatever is still unmatched starts a new track. Every input row receives
    an id -- a detection is never dropped, so the row set (and therefore GS-DetA) is identical to
    the control's.

    Args:
        frames: ``(N,)`` frame index per detection, **ascending**.
        boxes: ``(N, 4)`` ``xyxy`` box per detection.
        embs: ``(N, D)`` L2-normalised appearance vectors, or ``None`` for geometry-only.
        has_emb: ``(N,)`` bool -- which rows carry a real embedding (the cache is strided).
        params: Association hyper-parameters.

    Returns:
        ``(N,)`` int array of new track ids, numbered from 1 in order of creation.
    """
    n = len(frames)
    out = np.zeros(n, dtype=int)
    if n == 0:
        return out
    use_app = embs is not None and has_emb is not None and params.w_app > 0.0
    # track state: box, last matched frame, appearance vector (or None)
    tb: list[np.ndarray] = []
    tf: list[int] = []
    tv: list[np.ndarray | None] = []
    order = np.argsort(frames, kind="stable")
    cuts = np.flatnonzero(np.diff(frames[order])) + 1
    for chunk in np.split(order, cuts):
        f = int(frames[chunk[0]])
        det_box = boxes[chunk]
        pending = list(range(len(chunk)))
        live = [t for t in range(len(tb)) if f - tf[t] <= params.max_lost]
        for r in range(max(params.rounds, 1)):
            if not pending or not live:
                break
            e = params.e + r * params.e_step
            tbox = np.stack([tb[t] for t in live])
            sim = eiou_matrix(tbox, det_box[pending], e)
            valid = sim > 0.0
            cost = 1.0 - sim
            if use_app:
                tvec = np.stack([tv[t] if tv[t] is not None else np.zeros(embs.shape[1])
                                 for t in live])
                known = np.array([tv[t] is not None for t in live])[:, None]
                dvec = embs[chunk][pending]
                dknown = has_emb[chunk][pending][None, :]
                dist = 1.0 - tvec @ dvec.T
                pairable = known & dknown
                valid &= ~(pairable & (dist > params.app_max))
                app = np.clip(dist / max(params.app_max, 1e-9), 0.0, 1.0)
                cost = np.where(pairable,
                                params.w_app * app + (1.0 - params.w_app) * cost, cost)
            hits = _match(cost, valid)
            taken_t, taken_d = set(), set()
            for i, j in hits:
                t, d = live[i], pending[j]
                out[chunk[d]] = t + 1
                tb[t] = det_box[d]
                tf[t] = f
                if use_app and has_emb[chunk[d]]:
                    v = embs[chunk[d]] if tv[t] is None else (
                        params.ema * tv[t] + (1.0 - params.ema) * embs[chunk[d]])
                    tv[t] = v / max(float(np.linalg.norm(v)), 1e-8)
                taken_t.add(t)
                taken_d.add(d)
            live = [t for t in live if t not in taken_t]
            pending = [d for d in pending if d not in taken_d]
        for d in pending:  # unmatched detection -> a new track
            tb.append(det_box[d])
            tf.append(f)
            tv.append(embs[chunk[d]] if use_app and has_emb[chunk[d]] else None)
            out[chunk[d]] = len(tb)
    return out


# === Box cache (the one GPU stage) ===============================================================
def attach_boxes(df: pd.DataFrame, seq_dir: Path, detector, *,
                 match_tol_px: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Re-detect every frame and attach a box to every non-ball positions row (GPU/IO).

    Each frame's rows are matched greedily (nearest first) to that frame's detections by foot-point
    distance, one box per row; a row with no detection inside ``match_tol_px`` falls back to
    :func:`generator.team_anchor.estimate_player_box`.

    Args:
        df: The sequence's positions table (non-ball rows are the ones boxed).
        seq_dir: Sequence directory holding ``img1/``.
        detector: A ``generator.extract`` detector exposing ``detect(rgb) -> (xyxy, ...)``.
        match_tol_px: Foot-point tolerance for accepting a detection as this row's box.

    Returns:
        ``(boxes, matched)`` aligned to ``df`` row order: ``(N, 4)`` float and ``(N,)`` bool.
    """
    import cv2  # noqa: PLC0415

    from generator.team_anchor import estimate_player_box  # noqa: PLC0415
    from generator.track_relink import _frame_path  # noqa: PLC0415

    boxes = np.zeros((len(df), 4), dtype=np.float32)
    matched = np.zeros(len(df), dtype=bool)
    pos = {idx: k for k, idx in enumerate(df.index)}
    fh = fw = 0
    for frame_idx, grp in df.groupby("frame"):
        bgr = cv2.imread(str(_frame_path(seq_dir, int(frame_idx))))
        if bgr is None:
            logger.warning("%s: frame %d image missing", seq_dir.name, int(frame_idx))
            continue
        fh, fw = bgr.shape[:2]
        xyxy = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[0]
        rows = list(grp.itertuples())
        foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]]) if len(xyxy) else None
        pairs: list[tuple[float, int, int]] = []
        if foot is not None:
            for i, r in enumerate(rows):
                d = np.hypot(foot[:, 0] - float(r.image_x), foot[:, 1] - float(r.image_y))
                for j in np.flatnonzero(d <= match_tol_px):
                    pairs.append((float(d[j]), i, int(j)))
        used_r, used_d = set(), set()
        for _d, i, j in sorted(pairs):
            if i in used_r or j in used_d:
                continue
            k = pos[rows[i].Index]
            boxes[k] = xyxy[j]
            matched[k] = True
            used_r.add(i)
            used_d.add(j)
        for i, r in enumerate(rows):
            if i in used_r:
                continue
            k = pos[r.Index]
            boxes[k] = estimate_player_box(float(r.image_x), float(r.image_y), fh, fw)
    return boxes, matched


def build_box_cache(data_dir: Path, out_dir: Path, seqs: list[str], *,
                    positions_subdir: str, params: EiouParams) -> dict[str, dict]:
    """GPU stage: cache one detector box per positions row per sequence (resumable by disk state)."""
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from generator.extract import _build_detector  # noqa: PLC0415

    dest_dir = out_dir / BOX_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    cold = [s for s in seqs if not (dest_dir / f"{s}.npz").exists()]
    logger.info("box cache: %d sequences, %d cold", len(seqs), len(cold))
    if not cold:
        return {}
    detector = _build_detector("cuda" if torch.cuda.is_available() else "cpu", "football")
    stats: dict[str, dict] = {}
    for i, name in enumerate(cold):
        t0 = time.time()
        df = pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet")
        people = df[df["role"] != "ball"]
        boxes, matched = attach_boxes(people, data_dir / name, detector,
                                      match_tol_px=params.match_tol_px)
        np.savez_compressed(
            dest_dir / f"{name}.npz", version=np.array([EIOU_VERSION]),
            frames=people["frame"].to_numpy(dtype=np.int32),
            track_ids=people["track_id"].to_numpy(dtype=np.int32),
            boxes=boxes, matched=matched)
        stats[name] = {"rows": int(len(people)), "matched": int(matched.sum()),
                       "match_rate": float(matched.mean()) if len(people) else 0.0,
                       "seconds": round(time.time() - t0, 1)}
        logger.info("[%d/%d] %s: %d rows, %.3f detector-box match rate (%.0fs)", i + 1, len(cold),
                    name, stats[name]["rows"], stats[name]["match_rate"], stats[name]["seconds"])
    return stats


def load_boxes(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Read a box cache back as ``{(track_id, frame): (x1, y1, x2, y2)}``."""
    z = np.load(path)
    return {(int(t), int(f)): b for t, f, b in zip(z["track_ids"], z["frames"], z["boxes"])}


# === Re-linking one sequence + rewriting the cached artifacts ====================================
def relink_sequence(df: pd.DataFrame, boxes: dict, det: dict, params: EiouParams) -> dict:
    """Re-associate one sequence -> ``{(old_track_id, frame): new_track_id}`` for its non-ball rows.

    Args:
        df: The sequence's positions table.
        boxes: ``{(track_id, frame): box}`` from :func:`load_boxes`.
        det: ``{track_id: (frames, embeddings)}`` from
            :func:`generator.gta_link.load_or_build_det_embeddings` (strided, so most rows carry no
            embedding).
        params: Association hyper-parameters.

    Returns:
        The per-row relabelling map. Ball rows are not in it (the caller keeps them on their own
        id space).
    """
    people = df[df["role"] != "ball"].sort_values(["frame", "track_id"])
    keys = list(zip(people["track_id"].astype(int), people["frame"].astype(int)))
    box = np.stack([boxes.get(k, np.zeros(4, np.float32)) for k in keys]).astype(float)
    dim = next((e.shape[1] for _f, e in det.values() if len(e)), 0)
    embs = np.zeros((len(keys), dim), np.float32) if dim else None
    has = np.zeros(len(keys), bool)
    if dim:
        lut = {(int(t), int(f)): e for t, (fr, ev) in det.items() for f, e in zip(fr, ev)}
        for i, k in enumerate(keys):
            v = lut.get(k)
            if v is not None:
                embs[i] = v
                has[i] = True
    new = associate(people["frame"].to_numpy(dtype=int), box, embs, has, params)
    return dict(zip(keys, (int(v) for v in new)))


def write_variant(out_dir: Path, name: str, remap: dict, *, positions_subdir: str,
                  cache_subdir: str, percrop_subdir: str) -> dict:
    """Rewrite every ``track_id``-keyed artifact of one sequence into its ``_eiou`` variant.

    The three artifacts the frozen chain reads downstream are the positions parquet (submission +
    connector + solver geometry), the per-detection embedding npz (connector + solver gallery) and
    the per-crop OCR parquet (jersey evidence). All three are keyed by ``(track_id, frame)``, so the
    relabelling is exact and no GPU work is repeated.

    ``koshkina_jersey`` is deliberately **not** remapped: it only ever fills ``attributes.jersey`` on
    ``role == "player"`` rows (:func:`eval.gsr_jersey.patch_submission`,
    :func:`eval.gsr_prtreid_relink.propagation_fill`), and every such row is overwritten by the
    solver in :func:`tools.gsr_deleak.write_arm`, so it cannot reach the scored submission.
    """
    ball_base = 900_000

    def new_id(tid: int, frame: int) -> int:
        return remap.get((int(tid), int(frame)), ball_base + int(tid))

    df = pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet")
    df["track_id"] = [new_id(t, f) for t, f in zip(df["track_id"], df["frame"])]
    dest = out_dir / (positions_subdir + VARIANT)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / f"{name}.parquet", index=False)

    src = out_dir / cache_subdir / f"{name}.npz"
    z = np.load(src)
    dest = out_dir / (cache_subdir + VARIANT)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / f"{name}.npz", version=z["version"], embeddings=z["embeddings"],
        frames=z["frames"],
        track_ids=np.array([new_id(t, f) for t, f in zip(z["track_ids"], z["frames"])], np.int64))

    pc = pd.read_parquet(out_dir / percrop_subdir / f"{name}.parquet")
    pc["track_id"] = [new_id(t, f) for t, f in zip(pc["track_id"], pc["frame"])]
    dest = out_dir / (percrop_subdir + VARIANT)
    dest.mkdir(parents=True, exist_ok=True)
    pc.to_parquet(dest / f"{name}.parquet", index=False)
    return {"n_rows": int(len(df)), "n_tracks_before": int(len({k[0] for k in remap})),
            "n_tracks_after": int(len(set(remap.values())))}


def load_remap(path: Path) -> dict[tuple[int, int], int]:
    """Read a precomputed association as ``{(old_track_id, frame): new_track_id}``.

    The schema an external associator must write (campaign v7 V4 step 2: CAMELTrack): three aligned
    int arrays ``old_tid`` / ``frame`` / ``new_tid``, one entry per non-ball positions row. Every row
    must be present -- a missing row would silently fall back to the ball id space in
    :func:`write_variant` and move GS-DetA.
    """
    z = np.load(path)
    return {(int(t), int(f)): int(n) for t, f, n in zip(z["old_tid"], z["frame"], z["new_tid"])}


def build_arm_artifacts(data_dir: Path, out_dir: Path, seqs: list[str], params: EiouParams, *,
                        positions_subdir: str, cache_subdir: str,
                        percrop_subdir: str = "koshkina_percrop",
                        remap_dir: Path | None = None) -> dict[str, dict]:
    """Re-link and rewrite the variant artifacts for every sequence (CPU).

    ``remap_dir`` substitutes a precomputed ``{seq}.npz`` association (:func:`load_remap`) for
    :func:`relink_sequence`, so an external tracker can be measured through this exact chain with
    nothing else changed. ``params`` is then unused for the association itself.
    """
    from generator.gta_link import GtaParams, load_or_build_det_embeddings  # noqa: PLC0415

    gp = GtaParams()
    stats: dict[str, dict] = {}
    for i, name in enumerate(seqs):
        box_path = out_dir / BOX_SUBDIR / f"{name}.npz"
        if not box_path.exists():
            logger.warning("%s: no box cache, skipped", name)
            continue
        df = pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet")
        if remap_dir is not None:
            remap = load_remap(remap_dir / f"{name}.npz")
            n_people = int((df["role"] != "ball").sum())
            if len(remap) != n_people:
                raise SystemExit(f"{name}: remap has {len(remap)} rows, positions has {n_people}")
        else:
            det = load_or_build_det_embeddings(
                data_dir / name, df, out_dir / cache_subdir / f"{name}.npz", params=gp)
            remap = relink_sequence(df, load_boxes(box_path), det, params)
        stats[name] = write_variant(out_dir, name, remap, positions_subdir=positions_subdir,
                                    cache_subdir=cache_subdir, percrop_subdir=percrop_subdir)
        logger.info("[%d/%d] %s: %d tracks -> %d", i + 1, len(seqs), name,
                    stats[name]["n_tracks_before"], stats[name]["n_tracks_after"])
    return stats


def clear_arm_caches(out_dir: Path, key: str, variant: str = VARIANT) -> None:
    """Delete the per-arm caches an EIoU sweep point must not inherit from the previous point.

    Every sweep point produces a different track partition, so the bundle pickles, the connector arm
    and the densified votes of the previous point are all stale. They are wiped rather than keyed,
    which keeps the sweep to one point's worth of disk.

    Args:
        out_dir: Pipeline output root.
        key: The arm's :func:`tools.gsr_v4.config_key`.
        variant: The evidence variant whose vote cache is stale -- ``VARIANT`` for the shipped
            reader, ``<percrop_variant> + VARIANT`` when the arm swaps the reader weights.
    """
    for sub in (f"identity_bundles_v4_{key}", f"eval_v4_gta_{key}",
                f"koshkina_percrop_votes_f{FLOOR.replace('.', '')}{variant}"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)


# === Arms ========================================================================================
def set_embedder(name: str) -> None:
    """Point every downstream consumer at one embedding-cache space.

    :data:`eval.gsr_gta.EMBEDDER` / ``CACHE_SUBDIR`` are module constants, but every consumer
    (:func:`tools.gsr_calibfill.gta_arm`, :func:`eval.gsr_identity.build_bundle`,
    :func:`tools.gsr_v4.config_key`) imports them *inside* the function, so rebinding the module
    attribute switches the whole chain -- cache directory and arm key together, which is exactly the
    invariant the constant exists to protect.
    """
    import eval.gsr_gta as gta  # noqa: PLC0415

    gta.EMBEDDER = name
    gta.CACHE_SUBDIR = f"detembed_cache_{name}"


def run_point(data_dir: Path, out_dir: Path, seqs: list[str], params: EiouParams, *,
              embedder: str, tau: float, tag: str, positions_subdir: str,
              percrop_variant: str = "", remap_dir: Path | None = None) -> dict:
    """One EIoU sweep point: re-link, rewrite, then the frozen v4/v5 arm on the variant artifacts.

    ``percrop_variant`` selects which per-crop OCR evidence rides the partition (``""`` = the
    shipped reader's ``koshkina_percrop``, ``"_v6"`` = a retrained reader's). The association, the
    positions and the embeddings are untouched by it, so an arm at a non-empty ``percrop_variant``
    differs from the control in the reader weights ALONE.

    ``remap_dir`` swaps the association for a precomputed one (see :func:`build_arm_artifacts`);
    everything downstream -- connector, solver, votes, scorer -- is rebuilt identically.
    """
    from tools.gsr_v4 import config_key, solve_v4_arm  # noqa: PLC0415

    rl = build_arm_artifacts(data_dir, out_dir, seqs, params, positions_subdir=positions_subdir,
                             cache_subdir=f"detembed_cache_{embedder}",
                             percrop_subdir="koshkina_percrop" + percrop_variant,
                             remap_dir=remap_dir)
    set_embedder(embedder + VARIANT)
    variant = percrop_variant + VARIANT
    clear_arm_caches(out_dir, config_key(FLOOR, tau, True, variant, False), variant)
    payload = solve_v4_arm(data_dir, out_dir, seqs, tag=tag, floor=FLOOR, tau=tau, gate=True,
                           variant=variant, refit=False,
                           positions_subdir=positions_subdir + VARIANT)
    payload["eiou"] = {
        "version": EIOU_VERSION, "params": asdict(params), "embedder": embedder,
        "percrop_variant": percrop_variant, "remap_dir": str(remap_dir) if remap_dir else None,
        "n_tracks_before": int(sum(v["n_tracks_before"] for v in rl.values())),
        "n_tracks_after": int(sum(v["n_tracks_after"] for v in rl.values())),
        "per_seq": rl,
    }
    return payload


def run_control(data_dir: Path, out_dir: Path, seqs: list[str], *, tau: float, tag: str,
                positions_subdir: str) -> dict:
    """The same-session control: the identical chain on the shipped ByteTrack track ids."""
    from tools.gsr_v4 import solve_v4_arm  # noqa: PLC0415

    return solve_v4_arm(data_dir, out_dir, seqs, tag=tag, floor=FLOOR, tau=tau, gate=True,
                        variant="", refit=False, positions_subdir=positions_subdir)


def _grid(spec: str | None) -> list[EiouParams]:
    """Parse the sweep grid ``e:rounds[:w_app[:app_max]]`` comma-separated, or the default."""
    if not spec:
        return []
    out = []
    for point in spec.split(","):
        f = point.split(":")
        out.append(EiouParams(e=float(f[0]), rounds=int(f[1]),
                              w_app=float(f[2]) if len(f) > 2 else EiouParams.w_app,  # noqa: PLR2004
                              app_max=float(f[3]) if len(f) > 3 else EiouParams.app_max))  # noqa: PLR2004
    return out


def _report(rows: list[dict]) -> None:
    """ASCII sweep table (cp1252-safe console)."""
    print(f"{'arm':<34}{'GS-HOTA':>9}{'DetA':>8}{'AssA':>8}{'LocA':>8}{'IDF1':>8}"
          f"{'tracks':>8}{'mean':>7}{'med':>7}{'help':>5}{'hurt':>5}{'p':>9}")
    for r in rows:
        h, p = r["gs_hota"], r.get("paired") or {}
        print(f"{r['name']:<34}{h['GS-HOTA']:>9.4f}{h['GS-DetA']:>8.4f}{h['GS-AssA']:>8.4f}"
              f"{h['GS-LocA']:>8.4f}{h['IDF1']:>8.4f}{r.get('tracks', 0):>8}"
              f"{p.get('mean', 0.0):>7.2f}{p.get('median', 0.0):>7.2f}"
              f"{p.get('helped', 0):>5}{p.get('hurt', 0):>5}{p.get('wilcoxon_p', 1.0):>9.3g}")


def sweep(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str],
          grid: list[EiouParams], *, embedder: str, tau: float, split: str,
          positions_subdir: str, tag: str = "",
          percrop_variants: tuple[str, ...] = ("",)) -> dict:
    """Score the control and every (grid point x percrop variant) on ``seqs``, paired per sequence.

    ``percrop_variants`` crosses each association point with one or more per-crop OCR evidence
    variants (see :func:`run_point`), so a reader swap is measured on an identical partition inside
    one run.
    """
    from eval.gsr_identity import paired_stats  # noqa: PLC0415

    set_embedder(embedder)
    ctrl = run_control(data_dir, out_dir, seqs, tau=tau, tag=f"eiou{split}_ctrl_{embedder}",
                       positions_subdir=positions_subdir)
    rows = [{"name": f"control ({embedder}, tau {tau:g})", "gs_hota": ctrl["gs_hota"],
             "tracks": ctrl["v4"]["frags_after"]}]
    arms = {"control": ctrl}
    points = [(p, v) for p in grid for v in percrop_variants]
    for i, (p, pv) in enumerate(points):
        arm_tag = (f"eiou{split}_{embedder}_e{p.e:g}r{p.rounds}w{p.w_app:g}a{p.app_max:g}{pv}")
        res = run_point(data_dir, out_dir, seqs, p, embedder=embedder, tau=tau, tag=arm_tag,
                        positions_subdir=positions_subdir, percrop_variant=pv)
        res["paired"] = paired_stats(ctrl["gs_hota_per_seq"], res["gs_hota_per_seq"], seqs)
        arms[arm_tag] = res
        rows.append({"name": f"e {p.e:g} r{p.rounds} w_app {p.w_app:g} ocr{pv or '(shipped)'}",
                     "gs_hota": res["gs_hota"], "tracks": res["v4"]["frags_after"],
                     "paired": res["paired"]})
        logger.info("point %d/%d done", i + 1, len(points))
    _report(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"gsr_eiou_{split}_{embedder}{tag}.json"
    dest.write_text(json.dumps({"version": EIOU_VERSION, "split": split, "seqs": seqs,
                                "embedder": embedder, "tau": tau, "arms": arms},
                               indent=1, default=str), encoding="utf-8")
    print(f"wrote {dest}")
    return arms


def partition_purity(data_dir: Path, out_dir: Path, seqs: list[str],
                     positions_subdir: str) -> dict:
    """GT-audit one track partition: row-weighted purity and fragments per GT identity.

    This is the mechanism claim, measured directly on the tracker's own output *before* the
    connector: an EIoU tracklet may be shorter than a ByteTrack one and still be worth more, because
    a contaminated tracklet poisons the single jersey the solver assigns to every one of its rows.
    Ground truth is read **only to score**.
    """
    from collections import defaultdict  # noqa: PLC0415

    from eval.gsr_gta import gt_rows  # noqa: PLC0415
    from generator.gta_link import tracklet_purity  # noqa: PLC0415
    from generator.track_relink import load_gt_ids_by_frame  # noqa: PLC0415

    dom = row = contam = auditable = 0
    per_gt: list[float] = []
    for name in seqs:
        df = pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet")
        rows = gt_rows(df, load_gt_ids_by_frame(data_dir / name))
        p = tracklet_purity(rows)
        dom += p["n_dominant"]
        row += p["n_rows"]
        contam += p["n_contaminated"]
        auditable += p["n_auditable"]
        seen: dict[int, set] = defaultdict(set)
        for (tid, _f), gid in rows.items():
            seen[gid].add(tid)
        per_gt.append(float(np.mean([len(v) for v in seen.values()])) if seen else 0.0)
    return {"purity": dom / max(row, 1), "n_rows": row, "n_tracklets": auditable,
            "n_contaminated": contam, "frag_per_gt": float(np.mean(per_gt)) if per_gt else 0.0}


def purity_report(data_dir: Path, out_dir: Path, seqs: list[str], grid: list[EiouParams], *,
                  embedder: str, positions_subdir: str) -> list[dict]:
    """Purity of the ByteTrack partition and of each EIoU grid point (CPU, no arm, no scoring)."""
    rows = [{"arm": "ByteTrack (control)",
             **partition_purity(data_dir, out_dir, seqs, positions_subdir)}]
    for p in grid:
        build_arm_artifacts(data_dir, out_dir, seqs, p, positions_subdir=positions_subdir,
                            cache_subdir=f"detembed_cache_{embedder}")
        rows.append({"arm": f"EIoU e {p.e:g} r {p.rounds} w_app {p.w_app:g} app_max {p.app_max:g}",
                     **partition_purity(data_dir, out_dir, seqs, positions_subdir + VARIANT)})
    print(f"{'partition':<44}{'purity':>9}{'tracklets':>11}{'contam':>8}{'frag/GT':>9}")
    for r in rows:
        print(f"{r['arm']:<44}{r['purity']:>9.4f}{r['n_tracklets']:>11}{r['n_contaminated']:>8}"
              f"{r['frag_per_gt']:>9.2f}")
    return rows


# === Self-check ==================================================================================
def _demo() -> None:
    """Deterministic self-check: the paper's motivating case -- plain IoU fails, EIoU 0.7 recovers.

    Three boxes over two frames: one player moves further than his own width between frames (plain
    IoU = 0, so IoU association breaks the track), and a distractor sits far away and must not be
    picked up at any expansion the sweep visits.
    """
    a0 = np.array([[0.0, 0.0, 20.0, 40.0]])                       # frame 0: the player
    a1 = np.array([[26.0, 0.0, 46.0, 40.0],                       # frame 1: same player, moved 26 px
                   [200.0, 0.0, 220.0, 40.0]])                    # frame 1: the distractor
    assert eiou_matrix(a0, a1, 0.0).max() == 0.0, "plain IoU must fail across this gap"
    m = eiou_matrix(a0, a1, 0.7)
    assert m[0, 0] > 0.0 and m[0, 1] == 0.0, m
    assert np.allclose(eiou_matrix(a0, a0, 0.0), 1.0) and np.allclose(eiou_matrix(a0, a0, 3.0), 1.0)
    # Expansion is about the centre and monotone in e.
    ex = expand_boxes(a0, 1.0)
    assert np.allclose(ex, [[-10.0, -20.0, 30.0, 60.0]]), ex
    assert eiou_matrix(a0, a1, 1.4)[0, 0] > m[0, 0]

    frames = np.array([0, 1, 1])
    boxes = np.concatenate([a0, a1])
    plain = associate(frames, boxes, None, None, EiouParams(e=0.0, rounds=1, w_app=0.0))
    assert len(set(plain.tolist())) == 3, plain          # IoU: the track breaks -> 3 ids
    got = associate(frames, boxes, None, None, EiouParams(e=0.7, rounds=1, w_app=0.0))
    assert got[0] == got[1] and got[2] != got[0], got    # EIoU: 2 ids, distractor stays separate
    # Iterative scale-up reaches the same match from a tighter start, in a later round.
    step = associate(frames, boxes, None, None,
                     EiouParams(e=0.0, e_step=0.7, rounds=2, w_app=0.0))
    assert step[0] == step[1] and step[2] != step[0], step

    # Deep features veto a geometrically-plausible but visually-wrong match.
    d = 8
    p, q = np.zeros(d), np.zeros(d)
    p[0], q[1] = 1.0, 1.0
    e2 = np.stack([p, q, q])
    has = np.ones(3, bool)
    veto = associate(frames, boxes, e2, has, EiouParams(e=0.7, rounds=1, w_app=0.5, app_max=0.30))
    assert len(set(veto.tolist())) == 3, veto
    same = associate(frames, boxes, np.stack([p, p, q]), has,
                     EiouParams(e=0.7, rounds=1, w_app=0.5, app_max=0.30))
    assert same[0] == same[1] and same[2] != same[0], same
    # A row with no cached embedding is matched on geometry alone rather than refused.
    partial = associate(frames, boxes, e2, np.array([True, False, True]),
                        EiouParams(e=0.7, rounds=1, w_app=0.5, app_max=0.30))
    assert partial[0] == partial[1], partial
    print("gsr_eiou self-check OK: IoU 0.000 -> EIoU(e=0.7) %.3f on the paper's motivating case"
          % m[0, 0])


# === CLI =========================================================================================
def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--positions", default="positions_gate", help="cached positions source")
    ap.add_argument("--embedder", choices=("prtreid", "clip"), default="clip")
    ap.add_argument("--tau", type=float, default=None, help="connector tau (default: the frozen one)")
    ap.add_argument("--split", choices=("dev", "t38"), default="dev")
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names")
    ap.add_argument("--grid", default=None, help="sweep points 'e:rounds[:w_app[:app_max]]', comma-sep")
    ap.add_argument("--build-boxes", action="store_true", help="GPU: per-detection box cache")
    ap.add_argument("--run", action="store_true", help="control + grid on --split")
    ap.add_argument("--purity", action="store_true", help="GT purity audit of the partitions only")
    ap.add_argument("--tag", default="", help="suffix for the results filename (extra sweep batches)")
    ap.add_argument("--percrop-variants", default="",
                    help="comma-separated per-crop OCR variants crossed with the grid "
                         "(e.g. ',_v6' = shipped reader then the v6 reader on the same partition)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return

    params = EiouParams()
    from eval.gsr_identity import split_sequences  # noqa: PLC0415

    set_embedder(args.embedder)

    dev, t38 = split_sequences(args.data_dir, args.out_dir)
    seqs = args.seqs.split(",") if args.seqs else (dev if args.split == "dev" else t38)
    logger.info("%s: %d sequences", args.split, len(seqs))
    if args.build_boxes:
        st = build_box_cache(args.data_dir, args.out_dir, seqs, positions_subdir=args.positions,
                             params=params)
        if st:
            rows = sum(v["rows"] for v in st.values())
            hit = sum(v["matched"] for v in st.values())
            print(f"box cache: {len(st)} sequences, {rows} rows, detector-box match rate "
                  f"{hit / max(rows, 1):.4f}")
        return
    if args.purity:
        rows = purity_report(args.data_dir, args.out_dir, seqs, _grid(args.grid),
                             embedder=args.embedder, positions_subdir=args.positions)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dest = args.results_dir / f"gsr_eiou_purity_{args.split}_{args.embedder}{args.tag}.json"
        dest.write_text(json.dumps({"seqs": seqs, "rows": rows}, indent=1), encoding="utf-8")
        print(f"wrote {dest}")
        return
    if args.run:
        tau = args.tau if args.tau is not None else CONTROL_TAU[args.embedder]
        sweep(args.data_dir, args.out_dir, args.results_dir, seqs, _grid(args.grid),
              embedder=args.embedder, tau=tau, split=args.split, positions_subdir=args.positions,
              tag=args.tag, percrop_variants=tuple(args.percrop_variants.split(",")))
        return
    ap.error("choose --demo / --build-boxes / --run")


if __name__ == "__main__":
    main()
