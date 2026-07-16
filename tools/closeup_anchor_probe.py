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
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import registry
from generator import live_play
from generator.jersey_id import ILLEGIBLE, aggregate_votes, decide
from generator.jersey_id import JerseyRecognizer
from generator.team_anchor import PLAYER_ROLES, estimate_player_box
from generator.teams import jersey_color

OUT_DIR = Path("results/closeup_anchor_probe")
CKPT = Path("outputs/jersey/jersey_torso_r224_acc417.pt")
VIDEO_ROOT = Path("matches")

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
) -> list[dict]:
    """Detect people on the chunk's close-up frames and read jersey numbers.

    Args:
        match_id: registry match id.
        chunk_key: e.g. ``h1_chunk_000``.
        close_frames: sorted sampled frame indices classified ``close_up`` in this chunk.
        yolo: loaded COCO ``yolov8s`` model on the GPU.
        recog: loaded :class:`JerseyRecognizer` (torso checkpoint).
        tmp_dir: scratch dir for per-crop JPGs (recycled per chunk).

    Returns:
        One record per eligible crop: ``frame, box_h, pred, conf, team_guess, crop_path``.
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
            meta.append({
                "frame": fi, "box_h": round(box_h, 1),
                "team_guess": kit_team_guess(crop), "crop_path": str(p),
            })
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


def main() -> None:
    """CLI entry point (GPU; one job at a time)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="brighton_manutd")
    ap.add_argument("--mode", choices=["spotcheck", "full", "harvest_neg"], required=True)
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "spotcheck":
        run_spotcheck(args.match, args.chunk, args.n, args.seed, Path(args.ckpt))
    elif args.mode == "harvest_neg":
        harvest_negatives(args.match, Path(args.neg_out), args.neg_cap, args.seed)
    else:
        run_full(args.match, args.threshold, Path(args.ckpt), consensus=args.consensus,
                 kit_gate=args.kit_gate, ocr_gate=args.ocr_gate)


if __name__ == "__main__":
    main()
