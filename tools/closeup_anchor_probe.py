"""B2 Stage-2b opening probe: close-up jersey anchors on brighton_manutd broadcast footage.

PROBE ONLY -- writes nothing into the pipeline. Question: end-to-end, how many *high-confidence
(number, track-attachable) anchors* does a full match yield from close-up shots, before any fusion
is designed?

Pipeline (all reused modules):
  1. Select close-up frames with the shipped shot classifier (:mod:`generator.live_play`:
     ``shot_type == close_up`` = 1..4 football-YOLO detections). The football detector is blind on
     close-ups, so these frames carry few/no tactical boxes -- exactly the frames we re-detect.
  2. Re-detect people with **COCO yolov8s** (the July feasibility probe established football-YOLO
     caps close-up boxes ~160 px while COCO finds foreground figures up to ~1050 px).
  3. Crop each sizeable person and read the jersey number with the trained
     :class:`generator.jersey_id.JerseyRecognizer` (torso checkpoint,
     ``outputs/jersey/jersey_torso_r224_acc417.pt``). The recognizer was trained on broadcast-wide
     tracklet crops; close-up crops are much higher-res (downscaling to its 224x112 input puts more
     pixels on the number) -- measured here, not assumed.
  4. High-confidence anchor = per-crop number prediction with peak-class confidence
     ``>= --threshold``. The threshold is FROZEN by a manual 20-crop spot-check on ONE chunk
     (``--mode spotcheck``) and then applied unchanged to the whole match (``--mode full``).

Propagation feasibility (no fusion): for each anchor frame, is there a ``live_wide`` tactical frame
within +/- ``NEAR_WINDOW_S`` seconds across the cut whose team roster the anchor's kit could attach
to? That is the "could this read propagate onto a track" count.

Run (GPU, one job)::

    python -m tools.closeup_anchor_probe --mode spotcheck --chunk h1_chunk_000
    # ... inspect results/closeup_anchor_probe/spotcheck/, freeze a threshold ...
    python -m tools.closeup_anchor_probe --mode full --threshold 0.90
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import registry
from generator import live_play
from generator.jersey_id import ILLEGIBLE, aggregate_votes, decide, roster_mask
from generator.jersey_id import JerseyRecognizer
from generator.team_anchor import PLAYER_ROLES, estimate_player_box
from generator.teams import jersey_color

OUT_DIR = Path("results/closeup_anchor_probe")
CKPT = Path("outputs/jersey/jersey_torso_r224_acc417.pt")
VIDEO_ROOT = Path("matches")
ORACLE = Path("outputs/oracle/sofascore/player_stats_12436888.parquet")

# --- Step-4 lever constants (FROZEN before the yield run) -----------------------------------------
# LEVER 1 (roster-constrained decoding): mask the 100-way softmax to {both squads' shirtNumbers} u
# {illegible} before confidence/voting (:func:`generator.jersey_id.roster_mask`). Same 0.70 anchor
# threshold and same kit+OCR gates -- masking is a pure yield lever inside the step-3 gate.
# LEVER 2 (N-consecutive-agreement): within a close-up shot, accept a (IoU-track, number) pair that
# reads identically on >= N consecutive sampled frames at conf >= AGREE_FLOOR (below the 0.70 bar),
# provided the kit gate passes on every admitted crop and OCR agrees with the number on >= 1 of them.
AGREE_FLOOR = 0.50   # per-crop confidence floor for an agreement-admitted borderline read
IOU_LINK = 0.30      # min IoU to link two boxes across adjacent sampled frames into one track
# ponytail: greedy IoU chaining is the naive within-shot tracker (no motion model); the kit gate +
# per-number run + OCR-agreement guard precision, so a mis-link degrades yield, not precision.

# --- PRE-COMMITTED probe constants (frozen before any yield measurement) --------------------------
COCO_CONF = 0.20          # person-detection confidence floor (matches the July feasibility probe)
MIN_BOX_H = 100           # px: crop-eligibility floor. A back number ~= 0.35*0.40*box_h; box_h<100
#                           gives a <14 px number even in a close-up -> hopeless, not a candidate.
NEAR_WINDOW_S = 2.0       # +/- seconds across the cut for a tactical (live_wide) frame to attach to
SHOT_GAP_FRAMES = 15      # consecutive close-up sampled frames within this gap = one close-up shot
TEAM0_KIT = "red"         # registry: teams[0]=Man Utd (dark/anchored red home); teams[1]=Brighton

# --- Stage-2b step-3 gate constants (FROZEN on the 20-crop spot-check set, before the full run) ----
# LEVER A (kit-color gate): a candidate crop's median-torso CIELAB (generator.teams.jersey_color)
# must sit within KIT_DIST_MAX of one of the two match-kit centroids (recomputed per match from
# wide-play crops). On the spot-check set the 5 confirmed-correct Man Utd back crops sit at
# dmin <= 18.7 while the referee (only clear non-player) sits at 40.5 -> 22.0 keeps players, drops
# crowd/ref/coach. Single-negative caveat noted in the report.
KIT_DIST_MAX = 22.0
# LEVER B (OCR digit-evidence + agreement gate): easyocr on the upscaled torso band. Anchor valid
# iff OCR returns a digit token (conf >= OCR_MIN_CONF) whose value AGREES with the classifier number.
# Frozen on the spot-check set: band [0.15,0.55]v x [0.15,0.85]h, 3x cubic upscale reads the "8" and
# "20" backs at conf 1.0, returns nothing on the referee / illegible Brighton / 2-body crops, and
# reads 20 on the classifier's 20->24 misread (disagreement -> correctly rejected).
OCR_BAND = (0.15, 0.55, 0.15, 0.85)  # (top, bot, left, right) fractions of the crop
OCR_UPSCALE = 3
OCR_TEXT_THRESH = 0.5
OCR_MIN_CONF = 0.5


def _video_for(match_id: str, chunk_key: str) -> Path:
    """``h1_chunk_000`` -> ``matches/<id>/h1/chunk_000.mp4``."""
    half, num = chunk_key.split("_chunk_")
    return VIDEO_ROOT / match_id / half / f"chunk_{num}.mp4"


def classify_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """Shot-type per frame of one chunk via the shipped live-play classifier.

    Args:
        df_chunk: aligned rows for a single chunk (one row per detection per sampled frame).

    Returns:
        The :func:`live_play.classify_dense` table (index = frame) with a ``shot_type`` column.
    """
    return live_play.classify_dense(df_chunk)


def team_sets_by_frame(df_chunk: pd.DataFrame) -> dict[int, set[int]]:
    """``{frame: {team ids present}}`` for outfield/keeper detections (propagation team match)."""
    players = df_chunk[df_chunk["role"].isin(live_play.PLAYER_ROLES)]
    return {int(f): set(int(t) for t in g) for f, g in players.groupby("frame")["team"]}


def player_points_by_frame(df_chunk: pd.DataFrame) -> dict[int, np.ndarray]:
    """``{frame: [[image_x, image_y], ...]}`` for football-YOLO player/keeper detections.

    Football-YOLO fires only on pitch players (and keepers); its detection points are the weak
    label used to separate real players from non-players when harvesting reject negatives.

    Args:
        df_chunk: aligned rows for a single chunk.

    Returns:
        Per-frame array of player/keeper image points (empty array for frames with none).
    """
    players = df_chunk[df_chunk["role"].isin(live_play.PLAYER_ROLES)]
    return {int(f): g[["image_x", "image_y"]].to_numpy(dtype=np.float32)
            for f, g in players.groupby("frame")}


def _zero_detection_frames(df_chunk: pd.DataFrame) -> list[int]:
    """Sampled grid frames with no football-YOLO detection (SHOT_GRAPHIC: crowd/graphics/tunnel).

    The aligned parquet stores only frames that carry >=1 detection, so zero-detection frames are
    the gaps in the chunk's sampled grid. The grid step is the modal spacing between stored frames.

    Args:
        df_chunk: aligned rows for a single chunk.

    Returns:
        Sorted frame indices in the grid span that are absent from the parquet.
    """
    fr = np.array(sorted(df_chunk["frame"].unique()), dtype=int)
    if fr.size < 2:
        return []
    step = int(np.median(np.diff(fr)))
    if step < 1:
        return []
    grid = set(range(int(fr.min()), int(fr.max()) + 1, step))
    return sorted(grid - set(int(x) for x in fr))


def _has_player_inside(box: np.ndarray, pts: np.ndarray, margin: float = 12.0) -> bool:
    """True if any football-YOLO player point lies inside ``box`` (``x1,y1,x2,y2``) with a margin."""
    if pts.size == 0:
        return False
    x1, y1, x2, y2 = box
    inside = ((pts[:, 0] >= x1 - margin) & (pts[:, 0] <= x2 + margin)
              & (pts[:, 1] >= y1 - margin) & (pts[:, 1] <= y2 + margin))
    return bool(inside.any())


def kit_team_guess(crop_bgr: np.ndarray) -> int | None:
    """Cheap red-vs-not kit guess on a person crop (team 0 = Man Utd red, team 1 = Brighton).

    Uses the torso band (upper 15-55% of the crop) hue: a strong red share -> team 0, else team 1.
    Deliberately naive -- this is only the *plausibility* leg of propagation, not a committed label.

    Args:
        crop_bgr: BGR person crop.

    Returns:
        ``0`` (red kit), ``1`` (other), or ``None`` if the crop is empty.

    # ponytail: hue red-share heuristic, replace with the pipeline's kit-anchor model if this leg
    # ever becomes load-bearing (today it is only a sanity column).
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return None
    band = crop_bgr[int(0.15 * h):int(0.55 * h)]
    if band.size == 0:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (sat > 80) & (val > 60)
    red = strong & ((hue < 10) | (hue > 170))
    red_share = float(red.sum()) / max(float(strong.sum()), 1.0)
    return 0 if red_share > 0.30 else 1


def kit_centroids(match_id: str, df: pd.DataFrame, *, max_crops: int = 1200,
                  frames_per_chunk: int = 20) -> np.ndarray:
    """The two match-kit CIELAB centroids, recomputed from wide-play player crops (cached).

    Samples ``live_wide`` player detections across chunks, estimates a torso box per foot point
    (:func:`generator.team_anchor.estimate_player_box`), summarises each with
    :func:`generator.teams.jersey_color`, and KMeans(2)s the colours into the two kit centroids.
    Cached to ``kit_centroids.json`` in :data:`OUT_DIR` (delete it to recompute).

    Args:
        match_id: registry match id.
        df: the match's aligned parquet.
        max_crops: stop once this many wide crops are collected.
        frames_per_chunk: wide frames sampled per chunk.

    Returns:
        ``(2, 3)`` float32 array of ``[L*, a*, b*]`` kit centroids.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    cache = OUT_DIR / "kit_centroids.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        if blob.get("match") == match_id:
            return np.array(blob["centroids"], dtype=np.float32)

    cols: list[np.ndarray] = []
    for chunk_key, dfc in df.groupby("chunk", sort=True):
        cls = classify_chunk(dfc)
        wide = set(int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_LIVE_WIDE])
        pl = dfc[dfc["role"].isin(PLAYER_ROLES) & dfc["frame"].isin(wide)]
        frames = sorted(int(f) for f in pl["frame"].unique())
        if not frames:
            continue
        pick = [frames[i] for i in
                np.linspace(0, len(frames) - 1, min(frames_per_chunk, len(frames))).astype(int)]
        video = _video_for(match_id, chunk_key)
        if not video.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        for fr in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, bgr = cap.read()
            if not ok:
                continue
            for r in pl[pl["frame"] == fr].itertuples(index=False):
                x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
                crop = bgr[y1:y2, x1:x2]
                if crop.size:
                    cols.append(jersey_color(crop))
        cap.release()
        if len(cols) >= max_crops:
            break
    if len(cols) < 2:
        raise ValueError(f"too few wide crops to fit kit centroids ({len(cols)})")
    cent = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.stack(cols)).cluster_centers_
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"match": match_id, "n_crops": len(cols),
                                 "centroids": cent.round(3).tolist()}, indent=2), encoding="utf-8")
    return cent.astype(np.float32)


def _kit_dist_ok(lab: np.ndarray, centroids: np.ndarray, dmax: float = KIT_DIST_MAX) -> bool:
    """LEVER A: True iff ``lab`` is within ``dmax`` CIELAB of the nearest kit centroid (pure)."""
    return bool(np.linalg.norm(centroids - lab, axis=1).min() <= dmax)


def _ocr_reader():  # noqa: ANN202
    """Lazy CPU easyocr reader (close-up digits are 100-200 px; GPU is busy with YOLO/recognizer)."""
    import easyocr  # noqa: PLC0415

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _ocr_tokens(crop_bgr: np.ndarray, reader) -> list[tuple[str, float]]:  # noqa: ANN001
    """OCR digit tokens ``(text, conf)`` from the upscaled torso band of a crop (impure: reader)."""
    h, w = crop_bgr.shape[:2]
    top, bot, left, right = OCR_BAND
    band = crop_bgr[int(top * h):int(bot * h), int(left * w):int(right * w)]
    if band.size == 0:
        return []
    band = cv2.resize(band, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    out = reader.readtext(band, allowlist="0123456789", text_threshold=OCR_TEXT_THRESH)
    return [(t, float(c)) for _, t, c in out]


def _digit_agreement(tokens: list[tuple[str, float]], pred: int,
                     min_conf: float = OCR_MIN_CONF) -> bool:
    """LEVER B: True iff any OCR digit token (conf >= ``min_conf``) equals ``pred`` (pure).

    This is the precision play: the classifier already committed to ``pred``; requiring an
    independent OCR read of the *same* number in the torso band both proves a digit region exists
    (kills front/side no-number crops) and cross-checks the value (kills 20->24-style misreads).
    """
    return any(c >= min_conf and t.isdigit() and int(t) == pred for t, c in tokens)


def detect_and_read(
    match_id: str,
    chunk_key: str,
    close_frames: list[int],
    yolo,  # noqa: ANN001 - ultralytics YOLO
    recog: JerseyRecognizer,
    tmp_dir: Path,
    keep_probs: bool = False,
    mask: np.ndarray | None = None,
) -> list[dict]:
    """Detect people on the chunk's close-up frames and read jersey numbers.

    Args:
        match_id: registry match id.
        chunk_key: e.g. ``h1_chunk_000``.
        close_frames: sorted sampled frame indices classified ``close_up`` in this chunk.
        yolo: loaded COCO ``yolov8s`` model on the GPU.
        recog: loaded :class:`JerseyRecognizer` (torso checkpoint).
        tmp_dir: scratch dir for per-crop JPGs (recycled per chunk).
        keep_probs: also stash each crop's softmax row (per-shot consensus pooling).
        mask: optional :func:`roster_mask`; when given, each record also carries the box and the
            roster-masked read (``pred_m``, ``conf_m``) for the step-4 lever arms.

    Returns:
        One record per eligible crop: ``frame, box_h, pred, conf, team_guess, crop_path`` (plus
        ``box, pred_m, conf_m`` when ``mask`` is set).
    """
    video = _video_for(match_id, chunk_key)
    if not video.exists():
        print(f"WARN missing video {video}")
        return []
    cap = cv2.VideoCapture(str(video))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    paths: list[Path] = []
    n_persons = 0
    for fi in close_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = yolo(rgb, verbose=False, conf=COCO_CONF, classes=[0])[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        n_persons += len(xyxy)
        for box in xyxy:
            x1, y1, x2, y2 = (int(v) for v in box)
            box_h = float(y2 - y1)
            if box_h < MIN_BOX_H:
                continue
            crop = frame_bgr[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            k = len(paths)
            p = tmp_dir / f"crop_{k:06d}.jpg"
            cv2.imwrite(str(p), crop)
            paths.append(p)
            rec = {
                "frame": fi, "box_h": round(box_h, 1),
                "team_guess": kit_team_guess(crop), "crop_path": str(p),
            }
            if mask is not None:
                rec["box"] = (max(x1, 0), max(y1, 0), x2, y2)
            meta.append(rec)
    cap.release()
    if not paths:
        return [{"_n_persons": n_persons}] if n_persons else []

    probs = recog.crop_probs(paths)  # [n, 100] softmax rows, order preserved
    assert len(probs) == len(meta), f"crop/prob misalign {len(probs)} vs {len(meta)}"
    for m, row in zip(meta, probs, strict=True):
        num, conf = decide(row, min_conf=0.0)  # raw peak; anchor gate applied later
        m["pred"] = int(num)
        m["conf"] = round(float(conf), 4)
        m["illegible"] = bool(int(row.argmax()) == ILLEGIBLE)
        if mask is not None:
            num_m, conf_m = decide(row, min_conf=0.0, mask=mask)  # roster-masked read
            m["pred_m"] = int(num_m)
            m["conf_m"] = round(float(conf_m), 4)
        if keep_probs:
            m["_prob"] = row  # kept for per-shot consensus pooling (not written to CSV)
    meta.append({"_n_persons": n_persons})  # funnel bookkeeping
    return meta


def _load_yolo():  # noqa: ANN202
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = YOLO("yolov8s.pt")
    y.to(device)
    return y


def harvest_negatives(match_id: str, out_dir: Path, per_chunk_cap: int, seed: int) -> None:
    """Harvest non-player reject negatives from close-up frames (weak-labeled, no manual labels).

    Weak-label rule (documented), two non-player sources:
      * **zero-detection (SHOT_GRAPHIC) frames** -- grid frames where football-YOLO found nothing
        (:func:`_zero_detection_frames`): crowd, graphics, tunnel, extreme face close-ups. No pitch
        player is present, so back-number leakage is near zero (the purest source).
      * **close-up frames** -- a COCO person box (``box_h >= MIN_BOX_H``) containing **no**
        football-YOLO player/keeper point (:func:`_has_player_inside`): crowd/ref behind play and,
        usefully, front/side pitch players with no visible number (the class-20/29/11 hallucinations).

    These crops are trained as the illegible/reject class so the recognizer stops hallucinating
    attractor numbers on out-of-distribution close-up crops.

    Ceiling: football-YOLO under-detects on close-ups, so the close-up source leaks real
    back-number players; a downstream recognizer post-filter (keep illegible/attractor reads, drop
    confident non-attractor numbers) plus a visual purity spot-check handle this -- not assumed clean.

    Args:
        match_id: registry match id.
        out_dir: destination directory for negative crop JPGs (created).
        per_chunk_cap: max negatives kept per chunk (spreads the set across the match).
        seed: RNG seed for the per-chunk frame shuffle (deterministic harvest).
    """
    import random  # noqa: PLC0415

    match = registry.get(match_id)
    df = match.load_aligned()
    yolo = _load_yolo()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    kept = 0
    for chunk_key, dfc in df.groupby("chunk", sort=True):
        cls = classify_chunk(dfc)
        close_frames = [int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_CLOSE_UP]]
        # Zero-detection grid frames (football-YOLO found nothing) are SHOT_GRAPHIC: crowd,
        # graphics, tunnel, extreme face close-ups -- the purest non-player source (no pitch player
        # to mislabel). These are absent from the aligned parquet, so reconstruct the sampled grid.
        graphic_frames = _zero_detection_frames(dfc)
        harvest_frames = close_frames + graphic_frames
        rng.shuffle(harvest_frames)  # sample across the chunk, not just its first frames
        pts_by_frame = player_points_by_frame(dfc)
        video = _video_for(match_id, chunk_key)
        if not video.exists():
            print(f"WARN missing video {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        chunk_kept = 0
        for fi in harvest_frames:
            if chunk_kept >= per_chunk_cap:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = yolo(rgb, verbose=False, conf=COCO_CONF, classes=[0])[0]
            xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            pts = pts_by_frame.get(fi, np.empty((0, 2), dtype=np.float32))
            for box in xyxy:
                if chunk_kept >= per_chunk_cap:
                    break
                x1, y1, x2, y2 = (int(v) for v in box)
                if float(y2 - y1) < MIN_BOX_H or _has_player_inside(box, pts):
                    continue
                crop = frame_bgr[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(out_dir / f"{chunk_key}_f{fi}_{chunk_kept:04d}.jpg"), crop)
                chunk_kept += 1
                kept += 1
        cap.release()
        print(f"{chunk_key}: kept {chunk_kept} negatives (running total {kept})")
    print(f"harvested {kept} non-player negatives -> {out_dir}")


def run_spotcheck(match_id: str, chunk_key: str, n: int, seed: int, ckpt: Path = CKPT) -> None:
    """Dump ~``n`` legible-candidate crops from ONE chunk spanning the confidence range to freeze
    the anchor threshold by manual verification."""
    match = registry.get(match_id)
    df = match.load_aligned()
    dfc = df[df["chunk"] == chunk_key]
    cls = classify_chunk(dfc)
    close_frames = sorted(int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_CLOSE_UP])
    print(f"{chunk_key}: {len(close_frames)} close-up frames of {len(cls)} sampled")

    yolo = _load_yolo()
    recog = JerseyRecognizer.from_checkpoint(ckpt)
    tmp = OUT_DIR / "_tmp_spot"
    recs = [r for r in detect_and_read(match_id, chunk_key, close_frames, yolo, recog, tmp)
            if "pred" in r]
    cands = [r for r in recs if not r["illegible"]]
    cands.sort(key=lambda r: -r["conf"])
    print(f"eligible crops (box_h>={MIN_BOX_H}): {len(recs)}; legible reads: {len(cands)}")

    sc_dir = OUT_DIR / "spotcheck"
    if sc_dir.exists():
        shutil.rmtree(sc_dir)
    sc_dir.mkdir(parents=True)
    # Spread the sample across the confidence spectrum so the reliable floor is visible.
    idx = np.linspace(0, len(cands) - 1, min(n, len(cands))).round().astype(int)
    picks = [cands[i] for i in dict.fromkeys(idx.tolist())]
    table = []
    for rank, r in enumerate(picks):
        dst = sc_dir / f"sc_{rank:02d}_pred{r['pred']:02d}_conf{r['conf']:.3f}.jpg"
        shutil.copy(r["crop_path"], dst)
        table.append({"rank": rank, "file": dst.name, "pred": r["pred"],
                      "conf": r["conf"], "box_h": r["box_h"], "team_guess": r["team_guess"]})
    (sc_dir / "verdicts.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {len(picks)} spot-check crops -> {sc_dir}")
    print("Inspect the crops, fill the 'actual' column, then freeze --threshold.")


def run_full(match_id: str, threshold: float, ckpt: Path = CKPT, *, consensus: bool = False,
             kit_gate: bool = False, ocr_gate: bool = False) -> None:
    """Full-match yield + propagation-feasibility measurement at the frozen ``threshold``.

    With ``kit_gate`` and/or ``ocr_gate`` set, every conf-gated anchor is additionally scored by
    LEVER A (:func:`_kit_dist_ok`) and LEVER B (:func:`_digit_agreement`) in a **single** detect+read
    pass, and the funnel reports all four arms (baseline / +A / +B / +A+B) at once -- so the GPU
    detector and recognizer run only once, not four times.

    Args:
        match_id: registry match id.
        threshold: anchor confidence floor (per-crop peak, and pooled floor for consensus).
        ckpt: recognizer checkpoint to load (e.g. a negatives-retrained model).
        consensus: also aggregate reads per close-up shot (:func:`_consensus_anchors`).
        kit_gate: evaluate LEVER A (kit-colour) on every anchor.
        ocr_gate: evaluate LEVER B (OCR digit-evidence + agreement) on every anchor.
    """
    match = registry.get(match_id)
    df = match.load_aligned()
    fps = {ck: match.chunk_fps(ck) for ck in df["chunk"].unique()}

    centroids = kit_centroids(match_id, df) if kit_gate else None
    if kit_gate:
        print(f"kit centroids: {np.round(centroids, 1).tolist()}")
    reader = _ocr_reader() if ocr_gate else None
    step3_dir = OUT_DIR / "spotcheck_step3"
    if (kit_gate and ocr_gate) and step3_dir.exists():
        shutil.rmtree(step3_dir)
    survivors: list[dict] = []  # +A+B survivor crops (persistent copies) for the step-3 spot-check

    yolo = _load_yolo()
    recog = JerseyRecognizer.from_checkpoint(ckpt)
    # keep the baseline (torso-ckpt) anchors/ intact as before/after evidence; a different
    # checkpoint writes its per-frame anchors to a separate dir.
    anchors_dir = OUT_DIR / ("anchors" if Path(ckpt) == CKPT else "anchors_retrained")
    anchors_dir.mkdir(parents=True, exist_ok=True)
    cons_dir = OUT_DIR / "consensus_anchors"
    if consensus and cons_dir.exists():
        shutil.rmtree(cons_dir)
    if consensus:
        cons_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cons_reads: list[dict] = []
    per_chunk: dict[str, dict] = {}
    for chunk_key, dfc in df.groupby("chunk", sort=True):
        cls = classify_chunk(dfc)
        close_frames = sorted(int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_CLOSE_UP])
        wide_frames = np.array(sorted(int(f) for f in
                                      cls.index[cls["shot_type"] == live_play.SHOT_LIVE_WIDE]))
        teams_by_frame = team_sets_by_frame(dfc)
        tmp = OUT_DIR / "_tmp_full"
        if tmp.exists():
            shutil.rmtree(tmp)
        recs = detect_and_read(match_id, chunk_key, close_frames, yolo, recog, tmp,
                               keep_probs=consensus)
        n_persons = sum(r.get("_n_persons", 0) for r in recs)
        crops = [r for r in recs if "pred" in r]
        legible = [r for r in crops if not r["illegible"]]
        anchors = [r for r in legible if r["conf"] >= threshold]

        # LEVER A/B gates: score each conf-gated anchor once; the four arms are subsets.
        for a in anchors:
            crop = cv2.imread(a["crop_path"]) if (kit_gate or ocr_gate) else None
            a["kit_ok"] = (not kit_gate) or (
                crop is not None and _kit_dist_ok(jersey_color(crop), centroids))
            a["ocr_ok"] = (not ocr_gate) or (
                crop is not None and _digit_agreement(_ocr_tokens(crop, reader), a["pred"]))
        arm_kit = [a for a in anchors if a["kit_ok"]]
        arm_ocr = [a for a in anchors if a["ocr_ok"]]
        arm_both = [a for a in anchors if a["kit_ok"] and a["ocr_ok"]]
        final = arm_both  # strictest active arm (== baseline when no gate is set)

        # Propagation (final arm): is there a live_wide frame within +/- NEAR_WINDOW_S seconds?
        win = int(round(NEAR_WINDOW_S * fps[chunk_key]))
        attachable = 0
        team_matchable = 0
        for a in final:
            f = a["frame"]
            near = wide_frames[np.abs(wide_frames - f) <= win] if wide_frames.size else np.array([])
            a["near_wide_gap_frames"] = (int(np.abs(near - f).min()) if near.size else None)
            if near.size:
                attachable += 1
                # both teams are present on essentially any wide frame; team-match = the anchor's
                # kit team appears in at least one near wide frame.
                tg = a["team_guess"]
                if tg is not None and any(tg in teams_by_frame.get(int(nf), set()) for nf in near):
                    team_matchable += 1

        # Persist +A+B survivors (both gates active) for the step-3 visual verification.
        if kit_gate and ocr_gate:
            (step3_dir / "_survivors").mkdir(parents=True, exist_ok=True)
            for a in arm_both:
                src = Path(a["crop_path"])
                if src.exists():
                    dst = (step3_dir / "_survivors"
                           / f"{chunk_key}_f{a['frame']}_n{a['pred']:02d}_c{a['conf']:.3f}.jpg")
                    shutil.copy(src, dst)
                    survivors.append({"chunk": chunk_key, "frame": a["frame"], "pred": a["pred"],
                                      "conf": a["conf"], "team_guess": a["team_guess"],
                                      "file": dst.name, "path": str(dst)})

        # Collapse into distinct close-up shots (contiguous close-up sampled-frame runs).
        shots = _shots(close_frames)
        n_shots = len(shots)
        anchor_frames = {a["frame"] for a in anchors}
        n_shots_anchor = sum(any(lo <= af <= hi for af in anchor_frames) for lo, hi in shots)
        final_frames = {a["frame"] for a in final}
        n_shots_final = sum(any(lo <= af <= hi for af in final_frames) for lo, hi in shots)

        cons_anchors = _consensus_anchors(crops, shots, threshold) if consensus else []
        for ca in cons_anchors:  # persist a representative crop per shot-consensus anchor
            src = Path(ca.pop("rep_crop"))
            if src.exists():
                shutil.copy(src, cons_dir / f"{chunk_key}_s{ca['shot']:03d}_f{ca['frame']}"
                            f"_n{ca['pred']:02d}_c{ca['conf']:.3f}.jpg")
            ca["chunk"] = chunk_key
            cons_reads.append(ca)

        for a in anchors:  # persist anchor crops for audit
            src = Path(a["crop_path"])
            if src.exists():
                shutil.copy(src, anchors_dir / f"{chunk_key}_f{a['frame']}_n{a['pred']:02d}"
                            f"_c{a['conf']:.3f}.jpg")
        for r in crops:
            r["chunk"] = chunk_key
            r.pop("crop_path", None)
            r.pop("_prob", None)
            rows.append(r)
        per_chunk[chunk_key] = {
            "close_up_frames": len(close_frames),
            "persons_detected": int(n_persons),
            "eligible_crops": len(crops),
            "legible_reads": len(legible),
            "anchors": len(anchors),
            "anchors_kit": len(arm_kit),
            "anchors_ocr": len(arm_ocr),
            "anchors_kit_ocr": len(arm_both),
            "anchors_attachable_2s": attachable,
            "anchors_team_matchable": team_matchable,
            "closeup_shots": n_shots,
            "shots_with_anchor": n_shots_anchor,
            "shots_with_final_anchor": n_shots_final,
            "consensus_shot_anchors": len(cons_anchors),
        }
        c = per_chunk[chunk_key]
        print(f"{chunk_key}: close {c['close_up_frames']:5d} | persons {c['persons_detected']:5d} "
              f"| eligible {c['eligible_crops']:4d} | anchors {c['anchors']:3d} "
              f"| +A {c['anchors_kit']:3d} | +B {c['anchors_ocr']:3d} | +A+B {c['anchors_kit_ocr']:3d}"
              f" | attach {attachable:3d} | shots {n_shots_final}/{n_shots}")

    if kit_gate and ocr_gate:
        _finalize_step3(step3_dir, survivors)
    _write_full_report(match_id, threshold, ckpt, per_chunk, rows, cons_reads)


def _shots(close_frames: list[int]) -> list[tuple[int, int]]:
    """Segment sorted close-up sampled frames into contiguous shot ``(lo, hi)`` runs."""
    if not close_frames:
        return []
    shots: list[tuple[int, int]] = []
    lo = prev = close_frames[0]
    for f in close_frames[1:]:
        if f - prev > SHOT_GAP_FRAMES:
            shots.append((lo, prev))
            lo = f
        prev = f
    shots.append((lo, prev))
    return shots


def _consensus_anchors(
    crops: list[dict], shots: list[tuple[int, int]], min_conf: float
) -> list[dict]:
    """Per-close-up-shot consensus reads: pool every crop in a shot, one vote per shot.

    Each shot is treated as a pseudo-tracklet: :func:`aggregate_votes` over its crops' softmax rows
    yields one ``(number, pooled_conf)``. A shot dominated by non-players (rejected as illegible by
    the negatives-retrained head) or by scattered hallucinations pools to ``-1``; a shot with a real
    back number that reads consistently surfaces it. One representative crop (highest mass on the
    winning number) is kept per anchor for the audit spot-check.

    Args:
        crops: per-crop records carrying ``frame``, ``_prob`` (softmax row) and ``crop_path``.
        shots: contiguous close-up shot spans from :func:`_shots`.
        min_conf: floor on the pooled winning-number probability.

    Returns:
        One record per shot that yields a number: ``shot, frame, pred, conf, n_crops, rep_crop``.
    """
    out: list[dict] = []
    for si, (lo, hi) in enumerate(shots):
        members = [c for c in crops if lo <= c["frame"] <= hi and "_prob" in c]
        if not members:
            continue
        probs = np.stack([c["_prob"] for c in members])
        num, conf = aggregate_votes(probs, min_conf=min_conf)
        if num == -1:
            continue
        rep = max(members, key=lambda c: float(c["_prob"][num]))
        out.append({"shot": si, "frame": rep["frame"], "pred": int(num),
                    "conf": round(float(conf), 4), "n_crops": len(members),
                    "rep_crop": rep["crop_path"]})
    return out


def _finalize_step3(step3_dir: Path, survivors: list[dict], *, keep_all_max: int = 60,
                    sample_n: int = 40) -> None:
    """Copy the +A+B survivors into the step-3 spot-check with a verdicts template.

    Keeps every survivor if there are ``<= keep_all_max``; otherwise takes a ``sample_n`` stratified
    sample spread across the predicted numbers (sort by pred, even stride) so no single attractor
    class dominates the audit.

    Args:
        step3_dir: ``spotcheck_step3`` output directory.
        survivors: +A+B survivor records (each with a persistent ``path`` and ``file``).
        keep_all_max: verify every survivor at or below this count.
        sample_n: sample size when there are more survivors than ``keep_all_max``.
    """
    step3_dir.mkdir(parents=True, exist_ok=True)
    picks = survivors
    if len(survivors) > keep_all_max:
        ordered = sorted(survivors, key=lambda s: (s["pred"], s["conf"]))
        idx = np.linspace(0, len(ordered) - 1, sample_n).round().astype(int)
        picks = [ordered[i] for i in dict.fromkeys(idx.tolist())]
    table = []
    for rank, s in enumerate(sorted(picks, key=lambda s: (s["chunk"], s["frame"]))):
        dst = step3_dir / f"s3_{rank:02d}_n{s['pred']:02d}_c{s['conf']:.3f}.jpg"
        if Path(s["path"]).exists():
            shutil.copy(s["path"], dst)
        table.append({"rank": rank, "file": dst.name, "chunk": s["chunk"], "frame": s["frame"],
                      "pred": s["pred"], "conf": s["conf"], "team_guess": s["team_guess"],
                      "verdict": ""})
    (step3_dir / "verdicts.json").write_text(json.dumps(
        {"n_survivors": len(survivors), "n_verified": len(table), "crops": table},
        indent=2), encoding="utf-8")
    print(f"step-3 spot-check: {len(survivors)} +A+B survivors, wrote {len(table)} crops "
          f"-> {step3_dir}")


def _write_full_report(
    match_id: str, threshold: float, ckpt: Path, per_chunk: dict, rows: list[dict],
    cons_reads: list[dict],
) -> None:
    """Aggregate the funnel across chunks and write the JSON yield artifact."""
    keys = ["close_up_frames", "persons_detected", "eligible_crops", "legible_reads",
            "anchors", "anchors_kit", "anchors_ocr", "anchors_kit_ocr",
            "anchors_attachable_2s", "anchors_team_matchable",
            "closeup_shots", "shots_with_anchor", "shots_with_final_anchor",
            "consensus_shot_anchors"]
    total = {k: int(sum(c.get(k, 0) for c in per_chunk.values())) for k in keys}
    numbers = pd.Series(
        [r["pred"] for r in rows if r["conf"] >= threshold and not r["illegible"]]
    )
    cons_numbers = pd.Series([r["pred"] for r in cons_reads])
    out = {
        "match": match_id,
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "min_box_h_px": MIN_BOX_H,
        "coco_conf": COCO_CONF,
        "near_window_s": NEAR_WINDOW_S,
        "per_chunk": per_chunk,
        "total": total,
        "anchor_number_histogram": {int(k): int(v) for k, v in numbers.value_counts().items()},
        "consensus_number_histogram":
            {int(k): int(v) for k, v in cons_numbers.value_counts().items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "closeup_anchor_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "closeup_reads.csv", index=False)
    if cons_reads:
        pd.DataFrame(cons_reads).to_csv(OUT_DIR / "consensus_reads.csv", index=False)
    print("\nTOTAL FUNNEL:", json.dumps(total, indent=2))
    print(f"artifacts -> {OUT_DIR}")


def _roster_valid_numbers(oracle_path: Path = ORACLE) -> list[int]:
    """Both squads' back-of-shirt numbers (the roster-masking valid set) from the oracle parquet."""
    df = pd.read_parquet(oracle_path)
    return sorted(int(x) for x in pd.to_numeric(df["shirtNumber"], errors="coerce").dropna().unique())


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes (pure; 0.0 when disjoint or degenerate)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _link_tracklets(crops: list[dict], *, iou_min: float = IOU_LINK,
                    step_max: int = SHOT_GAP_FRAMES) -> list[list[dict]]:
    """Greedy IoU-chain the per-frame crops of one close-up shot into within-shot tracks (pure).

    Each crop carries ``frame`` and ``box``; boxes on adjacent sampled frames (gap ``<= step_max``)
    are matched highest-IoU-first (``>= iou_min``). Consecutive entries of a returned track are the
    same person on consecutive sampled frames -- the unit LEVER 2 requires for N-frame agreement.

    Args:
        crops: crop records for a single shot (each with ``frame`` and ``box``).
        iou_min: minimum IoU to link a box to a track's tail.
        step_max: maximum sampled-frame gap across which to link (a longer gap starts a new track).

    Returns:
        One list of crops per track, each ordered by ascending frame.
    """
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for c in crops:
        by_frame[c["frame"]].append(c)
    chains: list[dict] = []  # {"crops": [...], "tail_box": box, "tail_frame": f}
    for f in sorted(by_frame):
        cur = by_frame[f]
        prev = [(ci, ch) for ci, ch in enumerate(chains) if 0 < f - ch["tail_frame"] <= step_max]
        pairs = sorted(
            ((_iou(ch["tail_box"], c["box"]), ci, xi)
             for ci, ch in prev for xi, c in enumerate(cur)
             if _iou(ch["tail_box"], c["box"]) >= iou_min),
            reverse=True, key=lambda t: t[0])
        used_ch: set[int] = set()
        used_cur: set[int] = set()
        for _, ci, xi in pairs:
            if ci in used_ch or xi in used_cur:
                continue
            ch, c = chains[ci], cur[xi]
            ch["crops"].append(c)
            ch["tail_box"], ch["tail_frame"] = c["box"], f
            used_ch.add(ci)
            used_cur.add(xi)
        for xi, c in enumerate(cur):
            if xi not in used_cur:
                chains.append({"crops": [c], "tail_box": c["box"], "tail_frame": f})
    return [ch["crops"] for ch in chains]


def _agreement_admit(crops: list[dict], *, n: int, floor: float = AGREE_FLOOR,
                     use_masked: bool = False, iou_min: float = IOU_LINK) -> list[dict]:
    """LEVER 2: crops admitted by an N-consecutive same-number agreement run within a shot (pure).

    Links the shot's crops into within-shot tracks (:func:`_link_tracklets`), then on each track
    finds maximal runs of consecutive frames whose read (``pred``/``pred_m``) is the same number at
    ``conf >= floor``. A run of length ``>= n`` is admitted iff the kit gate passes on every crop in
    it and OCR agrees with that number on at least one crop -- so all N crops become anchors.

    Args:
        crops: crop records for one shot (each with ``frame, box, pred, conf[, pred_m, conf_m],
            kit_ok, toks``).
        n: minimum run length (consecutive agreeing frames).
        floor: per-crop confidence floor for run membership.
        use_masked: read ``pred_m``/``conf_m`` (roster-masked) instead of ``pred``/``conf``.
        iou_min: linking IoU threshold.

    Returns:
        The admitted crop records (a subset of ``crops``).
    """
    pk = "pred_m" if use_masked else "pred"
    ck = "conf_m" if use_masked else "conf"
    admitted: list[dict] = []
    for tr in _link_tracklets(crops, iou_min=iou_min):
        i, length = 0, len(tr)
        while i < length:
            num = tr[i][pk]
            if num < 1 or tr[i][ck] < floor:
                i += 1
                continue
            j = i
            while j + 1 < length and tr[j + 1][pk] == num and tr[j + 1][ck] >= floor:
                j += 1
            run = tr[i:j + 1]
            if (len(run) >= n and all(c["kit_ok"] for c in run)
                    and any(_digit_agreement(c["toks"], num) for c in run)):
                admitted.extend(run)
            i = j + 1
    return admitted


def _montage(paths: list[str], out: Path, *, cols: int = 10, cw: int = 96, ch: int = 160) -> None:
    """Tile crops into a labelled grid PNG for one-image visual verification (best-effort)."""
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    canvas = np.full((rows * ch, cols * cw, 3), 40, np.uint8)
    for k, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        r, c = divmod(k, cols)
        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cv2.resize(img, (cw, ch))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)


def _survivor_name(chunk: str, frame: int, number: int, conf: float) -> str:
    """The step-3 survivor filename schema (`anchor_wire.parse_survivor_name` reads it back)."""
    return f"{chunk}_f{frame}_n{number:02d}_c{conf:.3f}.jpg"


LEVER_ARMS = ("roster", "agree2", "agree3", "both2", "both3")
"""Step-4 lever arms layered on the step-3 baseline (+A+B kit+OCR gate)."""


def run_levers(match_id: str, threshold: float = 0.70, floor: float = AGREE_FLOOR,
               ckpt: Path = CKPT) -> None:
    """Step-4: one detect+read pass measuring the roster + N-agreement levers as 4 arms.

    Every close-up crop is read twice (raw and roster-masked); each candidate (conf ``>= floor`` in
    either read) is scored once by the kit gate and, if it passes, once by OCR. From those cached
    per-crop verdicts the arms are assembled without re-running the GPU detector/recognizer:
    ``baseline`` (raw, conf ``>= threshold``, kit+OCR), ``roster`` (masked, same gate), and the
    ``agree{2,3}`` / ``both{2,3}`` arms that additionally admit N-consecutive-agreement runs
    (:func:`_agreement_admit`) on the raw / masked reads respectively. New anchors (not in the
    baseline set) are persisted per arm with survivor filenames for the precision spot-check and for
    a best-arm re-wire.

    Args:
        match_id: registry match id.
        threshold: baseline/roster per-crop confidence bar (unchanged 0.70).
        floor: agreement-run per-crop confidence floor (0.50).
        ckpt: recognizer checkpoint (torso baseline).
    """
    match = registry.get(match_id)
    df = match.load_aligned()
    fps = {ck: match.chunk_fps(ck) for ck in df["chunk"].unique()}
    valid = _roster_valid_numbers()
    mask = roster_mask(valid)
    print(f"roster valid numbers ({len(valid)}): {valid}")

    centroids = kit_centroids(match_id, df)
    print(f"kit centroids: {np.round(centroids, 1).tolist()}")
    reader = _ocr_reader()

    all_arms = ("baseline", *LEVER_ARMS)
    lev_dir = OUT_DIR / "levers"
    chunks_dir = lev_dir / "_chunks"
    # Resume iff at least one per-chunk checkpoint exists; a fresh run wipes any partial (older,
    # non-resumable) output so survivor/new dirs are never double-counted across runs.
    resume = chunks_dir.exists() and any(chunks_dir.glob("*.json"))
    if not resume and lev_dir.exists():
        shutil.rmtree(lev_dir)
    (lev_dir / "new").mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    surv_dirs = {a: lev_dir / f"{a}_survivors" for a in all_arms}
    for d in surv_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    totals = {a: {"anchors": 0, "shots": 0, "feasible": 0} for a in all_arms}
    hist = {a: defaultdict(int) for a in all_arms}
    new_recs: list[dict] = []  # new-vs-baseline crops (for the spot-check)

    def _merge(blob: dict) -> None:
        """Fold one chunk's checkpoint (processed or cached) into the running totals."""
        for a in all_arms:
            s = blob["stats"][a]
            totals[a]["anchors"] += s["anchors"]
            totals[a]["shots"] += s["shots"]
            totals[a]["feasible"] += s["feasible"]
            for k, v in s["hist"].items():
                hist[a][int(k)] += int(v)
        for r in blob["new"]:
            new_recs.append({**r, "path": str(lev_dir / "new" / r["file"])})

    yolo = recog = None  # lazy GPU load -- skip entirely if every chunk is cached
    for chunk_key, dfc in df.groupby("chunk", sort=True):
        cj = chunks_dir / f"{chunk_key}.json"
        if cj.exists():
            _merge(json.loads(cj.read_text(encoding="utf-8")))
            print(f"{chunk_key}: cached, skipped "
                  f"(baseline total {totals['baseline']['anchors']})", flush=True)
            continue
        if yolo is None:
            yolo = _load_yolo()
            recog = JerseyRecognizer.from_checkpoint(ckpt)
        cls = classify_chunk(dfc)
        close_frames = sorted(int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_CLOSE_UP])
        wide = np.array(sorted(int(f) for f in
                               cls.index[cls["shot_type"] == live_play.SHOT_LIVE_WIDE]))
        win = int(round(NEAR_WINDOW_S * fps[chunk_key]))
        tmp = OUT_DIR / "_tmp_lev"
        if tmp.exists():
            shutil.rmtree(tmp)
        recs = detect_and_read(match_id, chunk_key, close_frames, yolo, recog, tmp, mask=mask)
        crops = [r for r in recs if "pred" in r]

        # Gate scoring: one imread + kit test per candidate; OCR only if kit passes (cached tokens).
        for c in crops:
            c["kit_ok"], c["toks"] = False, []
            cand = (c["pred"] >= 1 and c["conf"] >= floor) or \
                   (c["pred_m"] >= 1 and c["conf_m"] >= floor)
            if not cand:
                continue
            bgr = cv2.imread(c["crop_path"])
            if bgr is None:
                continue
            c["kit_ok"] = _kit_dist_ok(jersey_color(bgr), centroids)
            if c["kit_ok"]:
                c["toks"] = _ocr_tokens(bgr, reader)

        # Per-crop arm membership. baseline/roster = conf>=threshold + kit + OCR-agrees-with-number.
        for c in crops:
            c["in_baseline"] = (c["pred"] >= 1 and c["conf"] >= threshold and c["kit_ok"]
                                and _digit_agreement(c["toks"], c["pred"]))
            c["in_roster"] = (c["pred_m"] >= 1 and c["conf_m"] >= threshold and c["kit_ok"]
                              and _digit_agreement(c["toks"], c["pred_m"]))
        shots = _shots(close_frames)
        adm: dict[str, set[int]] = {a: set() for a in ("agree2", "agree3", "both2", "both3")}
        for lo, hi in shots:
            members = [c for c in crops if lo <= c["frame"] <= hi]
            for a, n, m_ in (("agree2", 2, False), ("agree3", 3, False),
                             ("both2", 2, True), ("both3", 3, True)):
                for c in _agreement_admit(members, n=n, floor=floor, use_masked=m_):
                    adm[a].add(id(c))
        for c in crops:
            c["in_agree2"] = c["in_baseline"] or id(c) in adm["agree2"]
            c["in_agree3"] = c["in_baseline"] or id(c) in adm["agree3"]
            c["in_both2"] = c["in_roster"] or id(c) in adm["both2"]
            c["in_both3"] = c["in_roster"] or id(c) in adm["both3"]

        # Number reported per arm: masked arms use pred_m, raw arms use pred.
        def _num(c: dict, arm: str) -> int:
            return c["pred_m"] if arm in ("roster", "both2", "both3") else c["pred"]

        chunk_stats: dict[str, dict] = {}
        for arm in all_arms:
            anchors = [c for c in crops if c[f"in_{arm}"]]
            frames = {c["frame"] for c in anchors}
            n_shots = sum(any(lo <= f <= hi for f in frames) for lo, hi in shots)
            feasible = 0
            h: dict[int, int] = defaultdict(int)
            for c in anchors:
                near = wide[np.abs(wide - c["frame"]) <= win] if wide.size else np.array([])
                if near.size:
                    feasible += 1
                h[_num(c, arm)] += 1
                src = Path(c["crop_path"])
                if src.exists():
                    shutil.copy(src, surv_dirs[arm]
                                / _survivor_name(chunk_key, c["frame"], _num(c, arm), c["conf"]))
            chunk_stats[arm] = {"anchors": len(anchors), "shots": n_shots, "feasible": feasible,
                                "hist": {int(k): int(v) for k, v in h.items()}}

        # Persist new-vs-baseline crops (any lever arm) for the precision spot-check.
        chunk_new: list[dict] = []
        for c in crops:
            new_arms = [a for a in LEVER_ARMS if c[f"in_{a}"] and not c["in_baseline"]]
            if not new_arms:
                continue
            src = Path(c["crop_path"])
            if not src.exists():
                continue
            masked_arm = any(a in ("roster", "both2", "both3") for a in new_arms)
            num = c["pred_m"] if masked_arm else c["pred"]
            conf = c["conf_m"] if masked_arm else c["conf"]
            dst = lev_dir / "new" / _survivor_name(chunk_key, c["frame"], num, conf)
            shutil.copy(src, dst)
            chunk_new.append({"chunk": chunk_key, "frame": c["frame"], "arms": new_arms,
                              "pred": c["pred"], "conf": c["conf"], "pred_m": c["pred_m"],
                              "conf_m": c["conf_m"], "file": dst.name})
        blob = {"stats": chunk_stats, "new": chunk_new}
        cj.write_text(json.dumps(blob), encoding="utf-8")  # checkpoint (resume-safe)
        _merge(blob)
        print(f"{chunk_key}: crops {len(crops):4d} | " + " | ".join(
            f"{a} {totals[a]['anchors']:4d}" for a in all_arms), flush=True)

    _finalize_levers(match_id, threshold, floor, valid, totals, hist, new_recs, lev_dir)


def _finalize_levers(match_id: str, threshold: float, floor: float, valid: list[int],
                     totals: dict, hist: dict, new_recs: list[dict], lev_dir: Path,
                     keep_all_max: int = 60, sample_n: int = 40) -> None:
    """Write the 4-arm table + per-arm new-anchor spot-check picks and montages."""
    all_arms = ("baseline", *LEVER_ARMS)
    # Per-arm new anchors + stratified spot-check picks.
    spot: dict[str, list[dict]] = {}
    for arm in LEVER_ARMS:
        news = [r for r in new_recs if arm in r["arms"]]
        if len(news) > keep_all_max:
            ordered = sorted(news, key=lambda r: (r["pred_m"] if "both" in arm or arm == "roster"
                                                  else r["pred"]))
            idx = np.linspace(0, len(ordered) - 1, sample_n).round().astype(int)
            picks = [ordered[i] for i in dict.fromkeys(idx.tolist())]
        else:
            picks = news
        spot[arm] = picks
        _montage([r["path"] for r in picks], lev_dir / f"montage_new_{arm}.png")

    out = {
        "match": match_id, "threshold": threshold, "agree_floor": floor,
        "roster_valid_numbers": valid,
        "arms": {a: {**totals[a],
                     "number_histogram": {int(k): int(v) for k, v in sorted(hist[a].items())}}
                 for a in all_arms},
        "new_vs_baseline": {a: len([r for r in new_recs if a in r["arms"]]) for a in LEVER_ARMS},
        "spotcheck_picks": {a: [{"file": r["file"], "pred": r["pred"], "conf": r["conf"],
                                 "pred_m": r["pred_m"], "conf_m": r["conf_m"], "verdict": ""}
                                for r in spot[a]] for a in LEVER_ARMS},
    }
    (lev_dir / "levers_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== STEP-4 LEVER ARMS ===")
    print(f"{'arm':<10} {'anchors':>8} {'shots':>6} {'feasible':>9} {'new':>5}")
    for a in all_arms:
        nv = 0 if a == "baseline" else out["new_vs_baseline"][a]
        t = totals[a]
        print(f"{a:<10} {t['anchors']:>8} {t['shots']:>6} {t['feasible']:>9} {nv:>5}")
    print(f"artifacts -> {lev_dir}")


def main() -> None:
    """CLI entry point (GPU; one job at a time)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="brighton_manutd")
    ap.add_argument("--mode", choices=["spotcheck", "full", "harvest_neg", "levers"], required=True)
    ap.add_argument("--chunk", default="h1_chunk_000", help="spotcheck: which chunk to sample")
    ap.add_argument("--n", type=int, default=24, help="spotcheck: crops to dump")
    ap.add_argument("--threshold", type=float, default=0.90, help="full: frozen anchor confidence")
    ap.add_argument("--ckpt", default=str(CKPT), help="full/spotcheck: recognizer checkpoint")
    ap.add_argument("--consensus", action="store_true",
                    help="full: also aggregate reads per close-up shot (per-shot consensus)")
    ap.add_argument("--kit-gate", action="store_true",
                    help="full: LEVER A -- require anchor torso colour near a match-kit centroid")
    ap.add_argument("--ocr-gate", action="store_true",
                    help="full: LEVER B -- require OCR digit in torso band agreeing with the number")
    ap.add_argument("--neg-out", default="data/jersey_negatives",
                    help="harvest_neg: output dir for non-player negative crops")
    ap.add_argument("--neg-cap", type=int, default=400,
                    help="harvest_neg: max negatives kept per chunk")
    ap.add_argument("--floor", type=float, default=AGREE_FLOOR,
                    help="levers: per-crop confidence floor for an agreement-run member")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "spotcheck":
        run_spotcheck(args.match, args.chunk, args.n, args.seed, Path(args.ckpt))
    elif args.mode == "harvest_neg":
        harvest_negatives(args.match, Path(args.neg_out), args.neg_cap, args.seed)
    elif args.mode == "levers":
        run_levers(args.match, args.threshold, args.floor, Path(args.ckpt))
    else:
        run_full(args.match, args.threshold, Path(args.ckpt), consensus=args.consensus,
                 kit_gate=args.kit_gate, ocr_gate=args.ocr_gate)


if __name__ == "__main__":
    main()
