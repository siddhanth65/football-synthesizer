"""Post-hoc track re-linking: merge ByteTrack fragments of the same player by appearance + motion.

ByteTrack fragments heavily on broadcast football (hundreds of ids for 22 players): every id break
costs association score (GS-AssA) and blocks identity propagation. This module merges fragments that
provably belong to the same player, *after* the CV pipeline has run, from the persisted positions
table -- no re-tracking, no change to geometry.

Two clean seams:

- **Pure merge logic** (:func:`summarize_fragments`, :func:`frags_mergeable`, :func:`greedy_merge`,
  :func:`pair_similarities`): fragments are merge candidates iff temporally disjoint, same team, same
  role, and gap-consistent motion (end position -> start position feasible at ``<= max_speed`` m/s);
  candidates are then merged greedily by appearance cosine similarity above a frozen threshold. These
  are CPU-pure and unit-tested with synthetic fragments.
- **Appearance embedding** (:class:`OsnetEmbedder`, :func:`fragment_embeddings`): a small pretrained
  torchreid OSNet (ImageNet ``osnet_x0_25``) embeds up to ``max_crops`` crops per fragment; the
  fragment embedding is their L2-normalised median. Crops are recovered by re-detecting the sequence
  frames and matching the persisted foot points -- the positions parquet stores only the projected
  foot point, not the source box (only ``pitch_x/pitch_y`` are smoothed; ``image_x/image_y`` are the
  raw box bottom-middle, so the same detector reproduces the box exactly).

The GS-AssA lift is measured externally by re-scoring the relinked submissions with the official
evaluator in :mod:`eval.gsr_score`; this module never touches the metric.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("track_relink")

#: SoccerNet-GSR broadcast frame rate (frames per second).
GSR_FPS = 25.0
#: Physical ceiling for a player closing the gap between two fragments (metres/second).
MAX_PLAYER_SPEED_MPS = 9.0
#: Slack added to the motion budget to absorb calibration/projection jitter at short gaps (metres).
POS_SLACK_M = 2.0
#: OSNet input size (h, w) and ImageNet normalisation (torchreid convention).
_OSNET_HW = (256, 128)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class Fragment:
    """One ByteTrack fragment (a single ``track_id`` within a sequence), summarised for merging.

    Attributes:
        track_id: The fragment's original track id.
        start_frame: First frame index the fragment appears in.
        end_frame: Last frame index the fragment appears in.
        start_xy: Pitch position (metres) at the earliest finite-pitch frame.
        end_xy: Pitch position (metres) at the latest finite-pitch frame.
        role: Majority role (``player``/``goalkeeper``/``referee``).
        team: Majority integer team (``-1`` for officials).
        n: Number of rows in the fragment.
    """

    track_id: int
    start_frame: int
    end_frame: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    role: str
    team: int
    n: int


@dataclass(frozen=True)
class RelinkParams:
    """Frozen re-linking hyper-parameters (threshold tuned on the 3 pilot sequences, then frozen).

    ``threshold=0.80`` was the pilot-combined GS-AssA / GS-HOTA maximiser on SNGS-021/022/023 (swept
    0.50-0.90); it is frozen here before touching the other 55 sequences.
    """

    threshold: float = 0.80
    fps: float = GSR_FPS
    max_speed_mps: float = MAX_PLAYER_SPEED_MPS
    pos_slack_m: float = POS_SLACK_M
    max_crops: int = 10
    match_tol_px: float = 6.0
    model_name: str = "osnet_x0_25"
    weights: str | None = None


# === Pure merge logic (unit-tested; no CV / metric stack) ========================================
def summarize_fragments(df: pd.DataFrame) -> list[Fragment]:
    """Summarise each non-ball ``track_id`` into a :class:`Fragment` (pure).

    Fragments with no finite pitch position anywhere are dropped (they cannot be motion-checked and
    carry no usable prediction). ``start_xy``/``end_xy`` use the earliest/latest *finite-pitch* rows
    while ``start_frame``/``end_frame`` use the true frame span (for temporal disjointness).

    Args:
        df: A positions table (:data:`generator.extract.POSITIONS_COLUMNS`).

    Returns:
        One :class:`Fragment` per mergeable track id, ordered by ``start_frame``.
    """
    frags: list[Fragment] = []
    people = df[df["role"] != "ball"]
    for tid, grp in people.groupby("track_id"):
        grp = grp.sort_values("frame")
        fin = grp[np.isfinite(grp["pitch_x"]) & np.isfinite(grp["pitch_y"])]
        if fin.empty:
            continue
        first, last = fin.iloc[0], fin.iloc[-1]
        frags.append(
            Fragment(
                track_id=int(tid),
                start_frame=int(grp["frame"].iloc[0]),
                end_frame=int(grp["frame"].iloc[-1]),
                start_xy=(float(first["pitch_x"]), float(first["pitch_y"])),
                end_xy=(float(last["pitch_x"]), float(last["pitch_y"])),
                role=str(Counter(grp["role"]).most_common(1)[0][0]),
                team=int(Counter(grp["team"]).most_common(1)[0][0]),
                n=int(len(grp)),
            )
        )
    return sorted(frags, key=lambda f: f.start_frame)


def _order(a: Fragment, b: Fragment) -> tuple[Fragment, Fragment] | tuple[None, None]:
    """Return ``(early, late)`` if temporally disjoint, else ``(None, None)`` (they overlap)."""
    if a.end_frame < b.start_frame:
        return a, b
    if b.end_frame < a.start_frame:
        return b, a
    return None, None


def frags_mergeable(a: Fragment, b: Fragment, *, fps: float = GSR_FPS,
                    max_speed_mps: float = MAX_PLAYER_SPEED_MPS,
                    pos_slack_m: float = POS_SLACK_M) -> bool:
    """Whether two fragments *could* be the same player (constraints only, no appearance) (pure).

    True iff they are temporally disjoint, share role and team, and the earlier fragment's end
    position can reach the later fragment's start position within the gap at ``<= max_speed_mps``
    (plus ``pos_slack_m`` for calibration jitter).
    """
    early, late = _order(a, b)
    if early is None or late is None:
        return False
    if early.role != late.role or early.team != late.team:
        return False
    gap_s = (late.start_frame - early.end_frame) / fps
    dist = float(np.hypot(early.end_xy[0] - late.start_xy[0], early.end_xy[1] - late.start_xy[1]))
    return dist <= max_speed_mps * gap_s + pos_slack_m


def pair_similarities(
    fragments: list[Fragment], embeddings: dict[int, np.ndarray], *, fps: float = GSR_FPS,
    max_speed_mps: float = MAX_PLAYER_SPEED_MPS, pos_slack_m: float = POS_SLACK_M,
) -> dict[tuple[int, int], float]:
    """Cosine similarity for every constraint-valid fragment pair that has embeddings (pure).

    Args:
        fragments: The sequence's fragments.
        embeddings: ``track_id -> L2-normalised embedding``.
        fps: Frame rate for the motion budget.
        max_speed_mps: Speed ceiling for the motion budget.
        pos_slack_m: Positional slack for the motion budget.

    Returns:
        ``{(i, j): cosine}`` keyed by fragment-list indices ``i < j``, only for mergeable pairs.
    """
    sims: dict[tuple[int, int], float] = {}
    for i in range(len(fragments)):
        ei = embeddings.get(fragments[i].track_id)
        if ei is None:
            continue
        for j in range(i + 1, len(fragments)):
            ej = embeddings.get(fragments[j].track_id)
            if ej is None:
                continue
            if not frags_mergeable(fragments[i], fragments[j], fps=fps,
                                   max_speed_mps=max_speed_mps, pos_slack_m=pos_slack_m):
                continue
            sims[(i, j)] = float(np.dot(ei, ej))
    return sims


def greedy_merge(
    fragments: list[Fragment], sims: dict[tuple[int, int], float], threshold: float, *,
    fps: float = GSR_FPS, max_speed_mps: float = MAX_PLAYER_SPEED_MPS,
    pos_slack_m: float = POS_SLACK_M,
) -> dict[int, int]:
    """Greedy agglomerative merge of fragments -> ``old_track_id -> new_track_id`` map (pure).

    Candidate pairs (``sims`` above ``threshold``) are processed by descending similarity. Two
    components merge only if one entirely precedes the other and the seam between them is motion-
    feasible, so every component stays a valid temporal chain (this re-checks the seam because merging
    grows a component's span). Each component is relabelled to its smallest original track id.

    Args:
        fragments: The sequence's fragments (index space of ``sims``).
        sims: ``{(i, j): cosine}`` from :func:`pair_similarities`.
        threshold: Minimum cosine similarity to consider a merge.
        fps: Frame rate for the seam motion check.
        max_speed_mps: Speed ceiling for the seam motion check.
        pos_slack_m: Positional slack for the seam motion check.

    Returns:
        ``{old_track_id: new_track_id}`` for every fragment (identity map for unmerged fragments).
    """
    parent = list(range(len(fragments)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def chain_ok(segs: list[tuple]) -> bool:
        """Segments sorted by start: no overlap and every seam feasible at <= max_speed."""
        for p, q in zip(segs, segs[1:]):
            if q[0] <= p[1]:  # q.start <= p.end -> intervals overlap
                return False
            gap_s = (q[0] - p[1]) / fps
            dist = float(np.hypot(p[3][0] - q[2][0], p[3][1] - q[2][1]))  # p.end_xy -> q.start_xy
            if dist > max_speed_mps * gap_s + pos_slack_m:
                return False
        return True

    # Component = (sorted disjoint segments, role, team); segment = (start, end, start_xy, end_xy).
    comp: dict[int, tuple[list[tuple], str, int]] = {
        i: ([(f.start_frame, f.end_frame, f.start_xy, f.end_xy)], f.role, f.team)
        for i, f in enumerate(fragments)
    }
    for _sim, i, j in sorted(((s, i, j) for (i, j), s in sims.items() if s >= threshold),
                             reverse=True):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        (si, role_i, team_i), (sj, role_j, team_j) = comp[ri], comp[rj]
        if role_i != role_j or team_i != team_j:
            continue
        merged = sorted(si + sj)
        if not chain_ok(merged):
            continue
        parent[rj] = ri
        comp[ri] = (merged, role_i, team_i)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(fragments)):
        groups[find(i)].append(fragments[i].track_id)
    remap: dict[int, int] = {}
    for members in groups.values():
        rep = min(members)
        for tid in members:
            remap[tid] = rep
    return remap


# === Appearance embedding (torchreid OSNet; GPU) =================================================
#: Where downloaded torchreid re-ID model-zoo weights (Market-1501/MSMT17) are cached.
_REID_ZOO_DIR = Path.home() / ".cache" / "torchreid" / "reid_zoo"


def resolve_reid_weights(weights: str) -> Path:
    """Resolve a re-ID ``weights`` spec to a local checkpoint path, downloading if needed (IO).

    Two forms are accepted:

    * an existing filesystem path to a ``.pth``/``.pth.tar``/``.pt`` checkpoint (used verbatim,
      e.g. a fine-tuned or football-domain checkpoint dropped on disk); or
    * a torchreid model-zoo key such as ``"osnet_x1_0_market1501"`` -- looked up in the torchreid
      ``reid_model_factory`` URL table and fetched with ``gdown`` into :data:`_REID_ZOO_DIR`.

    Args:
        weights: A checkpoint path or a model-zoo key (without the ``.pt`` suffix).

    Returns:
        Path to a local checkpoint file.

    Raises:
        KeyError: If ``weights`` is neither an existing file nor a known model-zoo key.
    """
    p = Path(weights)
    if p.exists():
        return p
    import torchreid.reid_model_factory as _rf  # noqa: PLC0415

    urls: dict[str, str] = _rf.__dict__["__trained_urls"]
    key = weights if weights.endswith(".pt") else f"{weights}.pt"
    if key not in urls:
        raise KeyError(f"unknown re-ID weights '{weights}': not a file and not in {sorted(urls)}")
    dst = _REID_ZOO_DIR / key
    if not (dst.exists() and dst.stat().st_size > 1_000_000):
        import gdown  # noqa: PLC0415

        _REID_ZOO_DIR.mkdir(parents=True, exist_ok=True)
        gdown.download(urls[key], str(dst), quiet=True)
    return dst


class OsnetEmbedder:
    """torchreid OSNet appearance embedder (fp16-friendly on 4 GB).

    With ``weights=None`` this is the original ImageNet-classification embedder (kit-dominated;
    Stage-2a baseline). Passing a torchreid re-ID model-zoo key (e.g. ``"osnet_x1_0_market1501"``,
    ``"osnet_ain_x1_0_msmt17"``) or a checkpoint path loads re-ID-objective weights instead --
    ``model_name`` must then be the matching backbone (``osnet_x1_0`` / ``osnet_ain_x1_0``).
    """

    def __init__(self, model_name: str = "osnet_x0_25", device: str | None = None,
                 weights: str | None = None) -> None:
        import torch  # noqa: PLC0415
        import torchreid  # noqa: PLC0415

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights
        model = torchreid.models.build_model(model_name, num_classes=1000, pretrained=weights is None)
        if weights is not None:
            from torchreid.reid.utils import load_pretrained_weights  # noqa: PLC0415

            load_pretrained_weights(model, str(resolve_reid_weights(weights)))
        self.model = model.to(self.device).eval()

    def embed(self, crops: list[np.ndarray], batch_size: int = 128) -> np.ndarray:
        """Embed a list of RGB uint8 crops -> ``(N, D)`` L2-normalised float32 features.

        Processes ``crops`` in mini-batches of at most ``batch_size`` so callers can pass
        arbitrarily large crop lists without risking a CUDA OOM (4 GB GPU budget).
        """
        torch = self._torch
        if not crops:
            return np.zeros((0, 512), np.float32)
        import cv2  # noqa: PLC0415

        out = []
        for start in range(0, len(crops), batch_size):
            chunk = crops[start:start + batch_size]
            batch = np.empty((len(chunk), _OSNET_HW[0], _OSNET_HW[1], 3), np.float32)
            for i, c in enumerate(chunk):
                r = cv2.resize(c, (_OSNET_HW[1], _OSNET_HW[0]), interpolation=cv2.INTER_LINEAR)
                batch[i] = (r.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode():
                feats = self.model(t).float().cpu().numpy()
            del t
            if self.device == "cuda":
                torch.cuda.empty_cache()
            out.append(feats)
        feats = np.concatenate(out, axis=0)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norms, 1e-8, None)


def _frame_path(seq_dir: Path, frame_idx: int) -> Path:
    """Map a positions ``frame`` index to its source image (frame N -> ``00000(N+1).jpg``)."""
    return seq_dir / "img1" / f"{frame_idx + 1:06d}.jpg"


def _sample_frames(frames: list[int], k: int) -> list[int]:
    """Pick up to ``k`` frame indices evenly spread across a fragment's frame list."""
    if len(frames) <= k:
        return frames
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [frames[i] for i in dict.fromkeys(idx)]


def fragment_embeddings(
    seq_dir: Path, df: pd.DataFrame, embedder: OsnetEmbedder, detector, *,
    max_crops: int = 10, match_tol_px: float = 6.0,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Compute one median appearance embedding per fragment by re-detecting and cropping (GPU/IO).

    For each fragment up to ``max_crops`` sample frames are chosen; each sample's persisted foot point
    is matched to the nearest re-detected box (within ``match_tol_px``) and that box is cropped. The
    fragment embedding is the L2-normalised median of its crop embeddings.

    Args:
        seq_dir: The GSR sequence directory (holds ``img1/``).
        df: The sequence's positions table.
        embedder: An :class:`OsnetEmbedder`.
        detector: A ``generator.extract`` detector exposing ``detect(frame_rgb) -> (xyxy, ...)``.
        max_crops: Max crops sampled per fragment.
        match_tol_px: Max foot-point distance (px) to accept a re-detected box as the crop source.

    Returns:
        ``(embeddings, crop_counts)`` -- ``track_id -> median embedding`` and ``track_id -> n crops``.
    """
    import cv2  # noqa: PLC0415

    people = df[(df["role"] != "ball") & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    # Choose sample rows per fragment, then invert to per-frame work lists.
    need: dict[int, list[tuple[int, float, float]]] = defaultdict(list)  # frame -> [(tid, ix, iy)]
    for tid, grp in people.groupby("track_id"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].astype(int).tolist()
        keep = set(_sample_frames(frames, max_crops))
        for r in grp.itertuples():
            if int(r.frame) in keep:
                need[int(r.frame)].append((int(tid), float(r.image_x), float(r.image_y)))

    crops_by_tid: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_idx in sorted(need):
        img_path = _frame_path(seq_dir, frame_idx)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xyxy = detector.detect(rgb)[0]
        if len(xyxy) == 0:
            continue
        foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]])
        for tid, ix, iy in need[frame_idx]:
            d = np.hypot(foot[:, 0] - ix, foot[:, 1] - iy)
            k = int(np.argmin(d))
            if d[k] > match_tol_px:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy[k])
            crop = rgb[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
            if crop.size:
                crops_by_tid[tid].append(crop)

    embeddings: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for tid, crops in crops_by_tid.items():
        feats = embedder.embed(crops)
        if len(feats) == 0:
            continue
        med = np.median(feats, axis=0)
        embeddings[tid] = med / max(float(np.linalg.norm(med)), 1e-8)
        counts[tid] = len(crops)
    return embeddings, counts


# === GT-based merge precision (pilot audit; GT track ids are available) ===========================
def load_gt_ids_by_frame(seq_dir: Path) -> dict[int, list[tuple[float, float, int]]]:
    """Load GT player/GK ``(x_centred, y_centred, gt_track_id)`` per frame index for merge audit."""
    from eval.gsr_score import CENTRE_SHIFT_X, CENTRE_SHIFT_Y  # noqa: PLC0415, F401

    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    id_to_idx = {im["image_id"]: int(Path(im["file_name"]).stem) - 1 for im in gt["images"]}
    out: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for ann in gt["annotations"]:
        attrs = ann.get("attributes") or {}
        if attrs.get("role") not in {"player", "goalkeeper"}:
            continue
        bp = ann.get("bbox_pitch")
        idx = id_to_idx.get(ann["image_id"])
        if not bp or idx is None or ann.get("track_id") is None:
            continue
        out[idx].append((bp["x_bottom_middle"], bp["y_bottom_middle"], int(ann["track_id"])))
    return out


def fragment_gt_ids(df: pd.DataFrame, gt_by_frame: dict, *, dist_tol_m: float = 5.0) -> dict[int, int]:
    """Dominant GT track id per fragment (nearest GT within ``dist_tol_m``), for merge precision."""
    from eval.gsr_score import CENTRE_SHIFT_X, CENTRE_SHIFT_Y  # noqa: PLC0415

    votes: dict[int, Counter] = defaultdict(Counter)
    people = df[df["role"].isin(["player", "goalkeeper"])]
    for frame, grp in people.groupby("frame"):
        gts = gt_by_frame.get(int(frame), [])
        if not gts:
            continue
        gxy = np.array([[g[0], g[1]] for g in gts])
        for r in grp.itertuples():
            if not (np.isfinite(r.pitch_x) and np.isfinite(r.pitch_y)):
                continue
            cx, cy = float(r.pitch_x) - CENTRE_SHIFT_X, float(r.pitch_y) - CENTRE_SHIFT_Y
            d = np.hypot(gxy[:, 0] - cx, gxy[:, 1] - cy)
            k = int(np.argmin(d))
            if d[k] <= dist_tol_m:
                votes[int(r.track_id)][gts[k][2]] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items() if c}


def merge_precision(remap: dict[int, int], gt_ids: dict[int, int]) -> tuple[int, int]:
    """Pair-level merge precision: ``(correct_pairs, total_pairs)`` among fragments sharing a group.

    A merged pair is correct iff both fragments have the same dominant GT id. Pairs where either
    fragment has no GT id are excluded (unauditable), so this is precision over auditable merges.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for tid, rep in remap.items():
        groups[rep].append(tid)
    correct = total = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ga, gb = gt_ids.get(members[a]), gt_ids.get(members[b])
                if ga is None or gb is None:
                    continue
                total += 1
                correct += int(ga == gb)
    return correct, total


# === Top-level per-sequence relink (resumable via an embedding cache) =============================
def relink_sequence(
    seq_dir: Path, df: pd.DataFrame, *, params: RelinkParams, embedder: OsnetEmbedder | None,
    detector, cache_path: Path | None = None, audit: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Relink one sequence's fragments -> ``(relinked_df, stats)``. Embeddings cached to ``cache_path``.

    Args:
        seq_dir: The GSR sequence directory.
        df: The sequence's positions table.
        params: Frozen re-link hyper-parameters.
        embedder: An :class:`OsnetEmbedder`, or ``None`` to require a warm cache.
        detector: A detector for crop re-extraction (unused if the cache is warm).
        cache_path: ``.npz`` embedding cache; reused when present (resumable).
        audit: When True also compute GT-based merge precision (pilot only).

    Returns:
        ``(relinked_df, stats)`` where ``stats`` has fragment counts, merges, and (if audited)
        merge-precision numbers.
    """
    fragments = summarize_fragments(df)
    embeddings, counts = _load_or_build_embeddings(
        seq_dir, df, params=params, embedder=embedder, detector=detector, cache_path=cache_path)
    sims = pair_similarities(fragments, embeddings, fps=params.fps,
                             max_speed_mps=params.max_speed_mps, pos_slack_m=params.pos_slack_m)
    remap = greedy_merge(fragments, sims, params.threshold, fps=params.fps,
                         max_speed_mps=params.max_speed_mps, pos_slack_m=params.pos_slack_m)
    n_before = len(fragments)
    n_after = len(set(remap.values())) if remap else n_before
    relinked = df.copy()
    relinked["track_id"] = relinked["track_id"].map(lambda t: remap.get(int(t), int(t)))
    stats = {
        "n_fragments_before": n_before, "n_fragments_after": n_after,
        "n_merges": n_before - n_after, "n_embedded": len(embeddings),
        "mean_crops": (float(np.mean(list(counts.values()))) if counts else 0.0),
    }
    if audit:
        gt_ids = fragment_gt_ids(df, load_gt_ids_by_frame(seq_dir))
        correct, total = merge_precision(remap, gt_ids)
        stats["merge_precision_correct"] = correct
        stats["merge_precision_total"] = total
    return relinked, stats


def _load_or_build_embeddings(
    seq_dir: Path, df: pd.DataFrame, *, params: RelinkParams, embedder: OsnetEmbedder | None,
    detector, cache_path: Path | None,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Load fragment embeddings from ``cache_path`` or build them (and cache) via re-detection."""
    if cache_path is not None and cache_path.exists():
        z = np.load(cache_path)
        tids = z["track_ids"].astype(int)
        embs = z["embeddings"]
        cnts = z["crop_counts"].astype(int)
        return ({int(t): embs[i] for i, t in enumerate(tids)},
                {int(t): int(cnts[i]) for i, t in enumerate(tids)})
    if embedder is None:
        raise RuntimeError(f"no embedding cache at {cache_path} and no embedder provided")
    embeddings, counts = fragment_embeddings(
        seq_dir, df, embedder, detector, max_crops=params.max_crops,
        match_tol_px=params.match_tol_px)
    if cache_path is not None and embeddings:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tids = np.array(sorted(embeddings), int)
        np.savez(
            cache_path, track_ids=tids,
            embeddings=np.stack([embeddings[int(t)] for t in tids]),
            crop_counts=np.array([counts.get(int(t), 0) for t in tids], int),
        )
    return embeddings, counts


def _demo() -> None:
    """Self-check of the pure merge seam on synthetic fragments (asserts; runnable)."""
    # Two temporally-disjoint fragments of a slow-moving player, nearly identical appearance -> merge.
    a = Fragment(1, 0, 10, (10.0, 10.0), (12.0, 10.0), "player", 0, 11)
    b = Fragment(2, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 0, 11)  # 5-frame gap, 0.5 m away
    c = Fragment(3, 5, 12, (50.0, 30.0), (52.0, 30.0), "player", 1, 8)  # overlaps a, other team
    e = {1: np.array([1.0, 0.0], np.float32), 2: np.array([0.99, 0.14], np.float32),
         3: np.array([0.0, 1.0], np.float32)}
    e = {k: v / np.linalg.norm(v) for k, v in e.items()}
    frags = [a, b, c]
    assert frags_mergeable(a, b) and not frags_mergeable(a, c)  # motion+team gate
    sims = pair_similarities(frags, e)
    assert (0, 1) in sims and (0, 2) not in sims  # only a-b is a constraint-valid pair
    remap = greedy_merge(frags, sims, threshold=0.9)
    assert remap[2] == remap[1] == 1 and remap[3] == 3  # a,b merge to id 1; c untouched
    # Same appearance but too far to reach in the gap -> motion gate blocks the merge.
    far = Fragment(2, 15, 25, (40.0, 10.0), (41.0, 10.0), "player", 0, 11)  # 28 m in 5 frames
    assert not frags_mergeable(a, far)
    print("track_relink demo OK: a+b merged, c and far-fragment correctly blocked")


if __name__ == "__main__":
    _demo()
