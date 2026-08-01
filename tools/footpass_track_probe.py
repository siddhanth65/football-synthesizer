"""Why does the chain track only ~half the players FOOTPASS says are visible? (CPU only.)

`results/FOOTPASS_E2E_PLAN.md` §6.1 measured, on two 10 s windows, that the full extraction chain
emits 0.44-0.66 tracked players per annotated-visible player while the **detector alone** finds
1.23-1.29x. That points at the association step, but two windows and a count ratio are not enough to
rank a fix. This probe replaces both weaknesses:

* **A real recall, not a count ratio.** FOOTPASS ships a broadcast ROI box per visible player, so
  recall is measured by greedy IoU matching against those boxes -- "was the player our pipeline
  would have to name actually covered by a track", which is exactly what the carrier gate needs.
* **Matched conditions.** The detector runs ONCE per frame at stride 1 and every box is cached;
  every stride and every ByteTrack setting is then replayed on the *identical* detections. Nothing
  in the comparison can move except the association.

Windows are picked at fixed quantiles of each half's annotated span (no shot-type cherry-picking --
whatever wide / close-up / replay footage lives there is what gets measured), and the first
:data:`WARMUP` frames of each window are discarded so a cold tracker does not penalise the large
strides. Every condition is scored on the SAME stride-5 evaluation grid, so a lower stride is
credited only for the association it buys, not for having more rows.

CLI::

    python -m tools.footpass_track_probe --stage cache    # ~0.35 s/frame CPU, resumable per window
    python -m tools.footpass_track_probe --stage sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from tools.footpass_prep import DIGEST_DIR, GAMES, MATCHES_ROOT, chunk_offsets, match_id

logger = logging.getLogger("footpass_track_probe")

#: Detection cache (gitignored scratch under outputs/).
CACHE_DIR = Path("outputs/footpass/track_probe")
#: Result table.
OUT_JSON = Path("results/footpass/track_probe.json")
#: Extracted VAL HDF5 (``tools/footpass_digest.py`` conventions); only this probe needs the ROI boxes.
VAL_H5 = Path("data/footpass/work/val_tactical_data.h5")
#: Window starts as a fraction of each half's annotated span.
QUANTILES = (0.15, 0.45, 0.75)
#: Frames captured per window, of which the first are tracker warm-up.
WINDOW = 175
WARMUP = 25
#: Evaluation grid: the stride the shipped pipeline runs at.
EVAL_STRIDE = 5
#: Lowest detector confidence cached, so a confidence sweep needs no second detector pass.
CACHE_CONF = 0.05
#: A GT box counts as covered at this IoU.
IOU_HIT = 0.3
#: Referee role id (excluded -- FOOTPASS annotates players and keepers only).
ROLE_REFEREE = 2


def windows(game: str) -> list[tuple[str, str, int]]:
    """``(half, chunk_key, local_start_frame)`` per window: 3 per half at fixed quantiles."""
    offs = chunk_offsets(game)
    out: list[tuple[str, str, int]] = []
    for half in ("H1", "H2"):
        d = np.load(DIGEST_DIR / f"val_{game}_{half}.npz")
        f0, f1 = int(d["window"][:, 0].min()), int(d["window"][:, 1].max())
        for q in QUANTILES:
            g = int(f0 + q * (f1 - f0))
            chunk = max((k for k, o in offs.items() if o <= g), key=lambda k: offs[k])
            out.append((half, chunk, g - offs[chunk]))
    return out


def gt_boxes(game: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{global_frame: ((n, 4) xyxy, (n,) player_id)}`` of every visible player, from the VAL HDF5.

    The player id is what makes fragmentation measurable: how many distinct track ids our tracker
    hands the same annotated player, and whether one track id covers two of them.

    Only frames inside a cached window are kept, so the 555 MB file is read once and the returned
    dict stays small.
    """
    import h5py  # noqa: PLC0415

    wanted: set[int] = set()
    offs = chunk_offsets(game)
    for _half, chunk, start in windows(game):
        wanted |= set(range(offs[chunk] + start, offs[chunk] + start + WINDOW))
    acc: dict[int, list] = {}
    with h5py.File(VAL_H5, "r") as f:
        for half in ("H1", "H2"):
            a = f[f"{game}_{half}"][:]
            keep = np.isin(a[:, 0].astype(np.int64), list(wanted)) & ~np.isnan(a[:, 9])
            for r in a[keep]:
                x, y, w, h = r[9], r[10], r[11], r[12]
                acc.setdefault(int(r[0]), []).append([x, y, x + w, y + h, r[1]])
    return {k: (np.asarray(v, float)[:, :4], np.asarray(v, float)[:, 4].astype(int))
            for k, v in acc.items()}


def cache_window(game: str, chunk: str, start: int, *, overwrite: bool = False) -> Path:
    """Run the CPU detector over one window at stride 1 and persist every box (resumable)."""
    import cv2  # noqa: PLC0415

    path = CACHE_DIR / f"{game}_{chunk}_{start:06d}.npz"
    if path.exists() and not overwrite:
        return path
    import generator.extract as gx  # noqa: PLC0415

    gx.DETECT_CONF = CACHE_CONF   # cache below the shipped 0.20 so confidence is a replay knob
    gx.BALL_CONF = CACHE_CONF
    det = gx._build_detector("cpu", "football")
    num = chunk.split("chunk_")[1]
    cap = cv2.VideoCapture(str(MATCHES_ROOT / match_id(game) / chunk[:2] / f"chunk_{num}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    boxes, confs, roles, owner = [], [], [], []
    for i in range(WINDOW):
        ok, bgr = cap.read()
        if not ok:
            break
        b, c, r, _ball = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes.append(b)
        confs.append(c)
        roles.append(r)
        owner.append(np.full(len(b), i, int))
    cap.release()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, game=game, chunk=chunk, start=start, n_frames=len(boxes),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4)),
        confs=np.concatenate(confs) if confs else np.zeros(0),
        roles=np.concatenate(roles) if roles else np.zeros(0, int),
        owner=np.concatenate(owner) if owner else np.zeros(0, int))
    logger.info("%s %s@%d: %d frames, %d boxes -> %s", game, chunk, start, len(boxes),
                sum(len(b) for b in boxes), path.name)
    return path


def load_window(path: Path) -> dict:
    """Cached window -> ``{n_frames, per_frame: [(boxes, confs, roles)], game, chunk, start}``."""
    z = np.load(path)
    n = int(z["n_frames"])
    owner = z["owner"].astype(int)
    per = [(z["boxes"][owner == i], z["confs"][owner == i], z["roles"][owner == i].astype(int))
           for i in range(n)]
    return {"n_frames": n, "per_frame": per, "game": str(z["game"]), "chunk": str(z["chunk"]),
            "start": int(z["start"])}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(len(a), len(b))`` IoU between two xyxy box sets (pure)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, thr: float = IOU_HIT) -> dict[int, int]:
    """Greedy IoU matching, one prediction per GT box -> ``{gt index: pred index}`` (pure)."""
    m = iou_matrix(gt, pred).copy()
    out: dict[int, int] = {}
    while m.size:
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        if m[i, j] < thr:
            break
        out[int(i)] = int(j)
        m[i, :] = -1.0
        m[:, j] = -1.0
    return out


def covered(gt: np.ndarray, pred: np.ndarray, thr: float = IOU_HIT) -> int:
    """How many GT boxes a prediction set covers, one prediction per GT (greedy by IoU; pure)."""
    return len(match(gt, pred, thr))


class _Replay:
    """A detector stand-in that hands ByteTrack the cached boxes of one frame."""

    def __init__(self, det: tuple, min_conf: float):
        boxes, confs, roles = det
        keep = (confs >= min_conf) & (roles != ROLE_REFEREE)
        self._out = (boxes[keep], confs[keep], roles[keep], None)

    def detect(self, _frame_rgb):
        """Return the cached ``(xyxy, conf, role_ids, ball_xy)``."""
        return self._out


def run_condition(win: dict, gt: dict, offset: int, cfg: dict) -> dict:
    """Score one window under one condition.

    Args:
        win: :func:`load_window` output.
        gt: ``{global_frame: (boxes, player_ids)}``.
        offset: Global frame of the window's first cached frame.
        cfg: ``stride``, ``min_conf`` and any ``supervision.ByteTrack`` kwargs.

    Returns:
        ``n_gt`` / ``n_det`` / ``n_trk`` box counts plus the fragmentation evidence:
        ``by_player`` (``{gt player: set of track ids}``) and ``by_track``
        (``{track id: set of gt players}``). Both are accumulated on the common stride-5
        evaluation grid only, so they are comparable across strides.
    """
    import supervision as sv  # noqa: PLC0415

    bt = sv.ByteTrack(
        track_activation_threshold=cfg.get("activation", 0.25),
        lost_track_buffer=cfg.get("lost_buffer", 60),
        minimum_matching_threshold=cfg.get("matching", 0.8),
        frame_rate=30, minimum_consecutive_frames=cfg.get("min_hits", 3))
    stride, min_conf = cfg["stride"], cfg.get("min_conf", 0.20)
    n_gt = n_det = n_trk = 0
    by_player: dict[int, set] = {}
    by_track: dict[int, set] = {}
    for i in range(0, win["n_frames"], stride):
        boxes, confs, roles, _ball = _Replay(win["per_frame"][i], min_conf).detect(None)
        tracked, tids = np.zeros((0, 4)), np.zeros(0, int)
        if len(boxes):
            d = sv.Detections(xyxy=boxes, confidence=confs, class_id=roles)
            d = bt.update_with_detections(d)
            if len(d):
                tracked = d.xyxy
                tids = (np.asarray(d.tracker_id, int) if d.tracker_id is not None
                        else np.full(len(d), -1))
        if i < WARMUP or i % EVAL_STRIDE:      # warm-up, and score on the common stride-5 grid
            continue
        g = gt.get(offset + i)
        if g is None or not len(g[0]):
            continue
        gbox, gpid = g
        n_gt += len(gbox)
        n_det += covered(gbox, boxes)
        for gi, pj in match(gbox, tracked).items():
            n_trk += 1
            by_player.setdefault(int(gpid[gi]), set()).add(int(tids[pj]))
            by_track.setdefault(int(tids[pj]), set()).add(int(gpid[gi]))
    return {"n_gt": n_gt, "n_det": n_det, "n_trk": n_trk,
            "by_player": by_player, "by_track": by_track}


#: The sweep. Baseline first; every arm differs from it in the fields it names.
CONDITIONS: dict[str, dict] = {
    "stride 5 (shipped)": {"stride": 5},
    "stride 2": {"stride": 2},
    "stride 1": {"stride": 1},
    "stride 5, min_hits=2": {"stride": 5, "min_hits": 2},
    "stride 5, min_hits=1": {"stride": 5, "min_hits": 1},
    "stride 5, matching=0.9": {"stride": 5, "matching": 0.9},
    "stride 5, lost_buffer=150": {"stride": 5, "lost_buffer": 150},
    "stride 5, activation=0.10": {"stride": 5, "activation": 0.10},
    "stride 5, det_conf=0.10": {"stride": 5, "min_conf": 0.10},
    "stride 5, min_hits=1 + matching=0.9": {"stride": 5, "min_hits": 1, "matching": 0.9},
    "stride 5, min_hits=1 + matching=0.9 + det_conf=0.10":
        {"stride": 5, "min_hits": 1, "matching": 0.9, "min_conf": 0.10},
    "stride 2, min_hits=1 + matching=0.9": {"stride": 2, "min_hits": 1, "matching": 0.9},
    "stride 1, min_hits=1 + matching=0.9": {"stride": 1, "min_hits": 1, "matching": 0.9},
}


#: BoT-SORT arms. ``cfg`` is a patch applied to ``generator/botsort_tuned.yaml``; ``None`` = as
#: shipped. The permissive arm exists because the tuned thresholds (``new_track_thresh`` 0.60,
#: ``track_high_thresh`` 0.50) were fitted on held-camera 25 fps footage, and on FOOTPASS's wide
#: cameras many true players never clear them -- reporting only the shipped config could produce a
#: negative that is about the thresholds rather than about GMC.
BOTSORT_ARMS: dict[str, dict | None] = {
    "botsort tuned (as shipped)": None,
    "botsort permissive": {"new_track_thresh": 0.25, "track_high_thresh": 0.25,
                           "track_low_thresh": 0.05},
}


def run_botsort(game: str, chunk: str, start: int, gt: dict, offset: int, *, stride: int,
                cfg_path: Path | None) -> dict:
    """Score one window through the real Ultralytics BoT-SORT path (detector + GMC + ReID).

    Cannot be replayed from the cached detections: Ultralytics couples tracking to the detector
    call, so the detector runs again -- but only on the *sampled* frames, which is far cheaper than
    the stride-1 cache. A fresh detector is built per window so tracker state never leaks across
    windows (the ByteTrack sweep resets per window for the same reason).

    Args:
        game: FOOTPASS game id.
        chunk: chunk key.
        start: local start frame of the window.
        gt: ``{global_frame: (boxes, player_ids)}``.
        offset: global frame of the window's first frame.
        stride: frames between tracker updates (5 = the shipped cadence).
        cfg_path: BoT-SORT yaml to use; ``None`` = the repo's tuned config.

    Returns:
        The same shape :func:`run_condition` returns.
    """
    import cv2  # noqa: PLC0415

    import generator.extract as gx  # noqa: PLC0415
    from generator.tracking import build_tracker  # noqa: PLC0415

    if cfg_path is not None:
        gx._BOTSORT_CFG = str(cfg_path)
    det = gx._build_detector("cpu", "football")
    tracker = build_tracker("botsort")
    num = chunk.split("chunk_")[1]
    cap = cv2.VideoCapture(str(MATCHES_ROOT / match_id(game) / chunk[:2] / f"chunk_{num}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    n_gt = n_det = n_trk = 0
    by_player: dict[int, set] = {}
    by_track: dict[int, set] = {}
    for i in range(WINDOW):
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride:
            continue
        boxes, _c, roles, tids, _ball = tracker.update(det, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        keep = (np.asarray(roles, int) != ROLE_REFEREE) if len(boxes) else np.zeros(0, bool)
        boxes, tids = (boxes[keep], np.asarray(tids, int)[keep]) if len(boxes) else (boxes, tids)
        live = tids >= 0 if len(boxes) else np.zeros(0, bool)   # BoT-SORT emits unconfirmed as -1
        if i < WARMUP or i % EVAL_STRIDE:
            continue
        g = gt.get(offset + i)
        if g is None or not len(g[0]):
            continue
        gbox, gpid = g
        n_gt += len(gbox)
        n_det += covered(gbox, boxes)
        for gi, pj in match(gbox, boxes[live] if len(boxes) else boxes).items():
            n_trk += 1
            tid = int(tids[live][pj])
            by_player.setdefault(int(gpid[gi]), set()).add(tid)
            by_track.setdefault(tid, set()).add(int(gpid[gi]))
    cap.release()
    return {"n_gt": n_gt, "n_det": n_det, "n_trk": n_trk,
            "by_player": by_player, "by_track": by_track}


def botsort_sweep(games: tuple[str, ...], strides: tuple[int, ...],
                  arms: tuple[str, ...] | None = None) -> dict:
    """Every BoT-SORT arm x stride over the same windows the ByteTrack sweep used.

    Note on the ``det_recall`` it reports: Ultralytics' ``track()`` **replaces** the result's boxes
    with the tracked ones, so an untracked detection is not visible to this path at all and
    ``n_det == n_trk`` by construction. The comparable detector recall is the 0.9759 the stride-1
    cache measured with the identical model and thresholds.
    """
    import time  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    base = yaml.safe_load(Path("generator/botsort_tuned.yaml").read_text(encoding="utf-8"))
    per_game_gt = {g: gt_boxes(g) for g in games}
    offs = {g: chunk_offsets(g) for g in games}
    res: dict[str, dict] = {}
    for arm, patch in BOTSORT_ARMS.items():
        if arms and arm not in arms:
            continue
        cfg_path = None
        if patch:
            cfg_path = CACHE_DIR / f"botsort_{abs(hash(str(patch))) % 10**8}.yaml"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(yaml.safe_dump({**base, **patch}), encoding="utf-8")
        for stride in strides:
            name = f"{arm}, stride {stride}"
            tot = {"n_gt": 0, "n_det": 0, "n_trk": 0}
            frags: list[int] = []
            mixed = tracks = 0
            by_game: dict[str, dict] = {}
            t0 = time.time()
            for g in games:
                acc = {"n_gt": 0, "n_det": 0, "n_trk": 0}
                for _half, chunk, start in windows(g):
                    r = run_botsort(g, chunk, start, per_game_gt[g],
                                    offs[g][chunk] + start, stride=stride, cfg_path=cfg_path)
                    for k in ("n_gt", "n_det", "n_trk"):
                        acc[k] += r[k]
                    frags += [len(v) for v in r["by_player"].values()]
                    tracks += len(r["by_track"])
                    mixed += sum(1 for v in r["by_track"].values() if len(v) > 1)
                acc["det_recall"] = acc["n_det"] / max(acc["n_gt"], 1)
                acc["trk_recall"] = acc["n_trk"] / max(acc["n_gt"], 1)
                by_game[g] = acc
                for k in ("n_gt", "n_det", "n_trk"):
                    tot[k] += acc[k]
            lo, hi = wilson(tot["n_trk"], tot["n_gt"])
            res[name] = {**tot, "det_recall": tot["n_det"] / max(tot["n_gt"], 1),
                         "trk_recall": tot["n_trk"] / max(tot["n_gt"], 1),
                         "wilson95": [lo, hi], "per_game": by_game, "config": {**(patch or {}),
                                                                               "stride": stride},
                         "fragments_per_player_window": float(np.mean(frags)) if frags
                         else float("nan"),
                         "n_player_windows": len(frags), "n_tracks": tracks,
                         "contaminated_track_frac": mixed / max(tracks, 1),
                         "wall_s": round(time.time() - t0, 1)}
            logger.info("%-40s trk %.4f [%.4f-%.4f]  det %.4f  frag/player %.2f  contam %.3f "
                        "(%.0fs)", name, res[name]["trk_recall"], lo, hi, res[name]["det_recall"],
                        res[name]["fragments_per_player_window"],
                        res[name]["contaminated_track_frac"], res[name]["wall_s"])
    return res


def gpu_price(game: str, chunk: str, *, stride: int = 2, n_source: int = 1500,
              warmup: int = 60) -> dict:
    """**GPU**: per-sampled-frame cost of detector+tracker, ByteTrack vs BoT-SORT, same frames.

    Appendix A.5 could only price BoT-SORT's sparseOptFlow overhead on CPU (+0.07-0.14 s/frame,
    a 2x-wide band). This measures it where it matters. Only the tracked stage is timed --
    calibration, team classification and I/O are identical between the two arms and would only
    dilute the difference.

    Args:
        game: FOOTPASS game id.
        chunk: chunk key (e.g. ``h1_chunk_000``).
        stride: frames between tracker updates.
        n_source: source frames consumed per arm.
        warmup: sampled frames excluded from the timing (weight load, cuDNN autotune).

    Returns:
        Per-arm ``s_per_sampled_frame`` plus the projected 3-game overhead in hours.
    """
    import time  # noqa: PLC0415

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import generator.extract as gx  # noqa: PLC0415
    from generator.tracking import build_tracker  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num = chunk.split("chunk_")[1]
    video = MATCHES_ROOT / match_id(game) / chunk[:2] / f"chunk_{num}.mp4"
    out: dict = {"device": device, "video": str(video), "stride": stride, "n_source": n_source}
    for arm in ("bytetrack", "botsort"):
        det = gx._build_detector(device, "football")
        tracker = build_tracker(arm)
        cap = cv2.VideoCapture(str(video))
        n = 0
        t0 = None
        for i in range(n_source):
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride:
                continue
            tracker.update(det, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            n += 1
            if n == warmup:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
        if device == "cuda":
            torch.cuda.synchronize()
        cap.release()
        timed = max(n - warmup, 1)
        out[arm] = {"sampled_frames_timed": timed,
                    "s_per_sampled_frame": (time.time() - t0) / timed if t0 else float("nan")}
        del det, tracker
        if device == "cuda":
            torch.cuda.empty_cache()
    delta = out["botsort"]["s_per_sampled_frame"] - out["bytetrack"]["s_per_sampled_frame"]
    # 18,587 s of VAL video at 25 fps, sampled every ``stride`` frames.
    sampled_3games = 18_587 * 25 / stride
    out["delta_s_per_sampled_frame"] = delta
    out["projected_3game_overhead_h"] = delta * sampled_3games / 3600
    out["sampled_frames_3games"] = sampled_3games
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for ``k`` of ``n`` (pure)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def sweep(games: tuple[str, ...]) -> dict:
    """Score every condition on every cached window; returns per-condition and per-game totals."""
    per_game_gt = {g: gt_boxes(g) for g in games}
    offs = {g: chunk_offsets(g) for g in games}
    cached = {g: sorted(CACHE_DIR.glob(f"{g}_*.npz")) for g in games}
    res: dict[str, dict] = {}
    for name, cfg in CONDITIONS.items():
        tot = {"n_gt": 0, "n_det": 0, "n_trk": 0}
        by_game: dict[str, dict] = {}
        frags: list[int] = []          # distinct track ids per (player, window)
        mixed = tracks = 0             # track ids covering >= 2 annotated players
        for g in games:
            acc = {"n_gt": 0, "n_det": 0, "n_trk": 0}
            for p in cached[g]:
                win = load_window(p)
                offset = offs[g][win["chunk"]] + win["start"]
                r = run_condition(win, per_game_gt[g], offset, cfg)
                for k in ("n_gt", "n_det", "n_trk"):
                    acc[k] += r[k]
                frags += [len(v) for v in r["by_player"].values()]
                tracks += len(r["by_track"])
                mixed += sum(1 for v in r["by_track"].values() if len(v) > 1)
            acc["det_recall"] = acc["n_det"] / max(acc["n_gt"], 1)
            acc["trk_recall"] = acc["n_trk"] / max(acc["n_gt"], 1)
            by_game[g] = acc
            for k in ("n_gt", "n_det", "n_trk"):
                tot[k] += acc[k]
        lo, hi = wilson(tot["n_trk"], tot["n_gt"])
        res[name] = {**tot, "det_recall": tot["n_det"] / max(tot["n_gt"], 1),
                     "trk_recall": tot["n_trk"] / max(tot["n_gt"], 1),
                     "trk_over_det": tot["n_trk"] / max(tot["n_det"], 1),
                     "wilson95": [lo, hi], "config": cfg, "per_game": by_game,
                     "fragments_per_player_window": float(np.mean(frags)) if frags else float("nan"),
                     "n_player_windows": len(frags), "n_tracks": tracks,
                     "contaminated_track_frac": mixed / max(tracks, 1)}
        logger.info("%-52s trk %.4f [%.4f-%.4f]  det %.4f  frag/player %.2f  contam %.3f", name,
                    res[name]["trk_recall"], lo, hi, res[name]["det_recall"],
                    res[name]["fragments_per_player_window"],
                    res[name]["contaminated_track_frac"])
    return res


def _selftest() -> None:
    """Pure-seam checks: IoU, greedy cover, Wilson."""
    a = np.array([[0.0, 0, 10, 10]])
    assert abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-9
    assert abs(iou_matrix(a, np.array([[5.0, 0, 15, 10]]))[0, 0] - 1 / 3) < 1e-9
    gt = np.array([[0.0, 0, 10, 10], [100.0, 0, 110, 10]])
    assert covered(gt, np.array([[0.0, 0, 10, 10]])) == 1
    assert covered(gt, gt) == 2
    assert covered(gt, np.zeros((0, 4))) == 0
    # one prediction cannot cover two GT boxes
    assert covered(np.array([[0.0, 0, 10, 10], [1.0, 1, 11, 11]]), np.array([[0.0, 0, 10, 10]])) == 1
    assert match(gt, gt[::-1]) == {0: 1, 1: 0}, match(gt, gt[::-1])
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25
    print("footpass_track_probe selftest ok")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["cache", "sweep", "botsort", "gpuprice", "selftest"])
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--strides", default="5", help="botsort: comma-separated tracker strides")
    ap.add_argument("--arms", default=None, help="botsort: comma-separated arm names")
    ap.add_argument("--chunk", default="h1_chunk_000", help="gpuprice: chunk key")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    games = tuple(args.games.split(","))
    if args.stage == "selftest":
        _selftest()
        return
    if args.stage == "cache":
        for g in games:
            for _half, chunk, start in windows(g):
                cache_window(g, chunk, start, overwrite=args.overwrite)
        return
    if args.stage == "gpuprice":
        res = gpu_price(games[0], args.chunk)
        out = OUT_JSON.with_name("gpu_price.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return
    if args.stage == "botsort":
        res = botsort_sweep(games, tuple(int(s) for s in args.strides.split(",")),
                            tuple(args.arms.split(",")) if args.arms else None)
        out = OUT_JSON.with_name("track_probe_botsort.json")
    else:
        res = sweep(games)
        out = OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n{'condition':52s} {'trk recall':>11s} {'95% CI':>17s} {'det':>7s} {'frag/plyr':>10s} "
          f"{'contam':>8s}")
    for name, r in res.items():
        print(f"{name:52s} {r['trk_recall']:11.4f} "
              f"[{r['wilson95'][0]:.4f}-{r['wilson95'][1]:.4f}] {r['det_recall']:7.4f} "
              f"{r['fragments_per_player_window']:10.2f} {r['contaminated_track_frac']:8.3f}")
    print(f"\nGT player-boxes scored: {next(iter(res.values()))['n_gt']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
