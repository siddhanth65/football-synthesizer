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
from generator.jersey_id import ILLEGIBLE, decide
from generator.jersey_id import JerseyRecognizer

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


def detect_and_read(
    match_id: str,
    chunk_key: str,
    close_frames: list[int],
    yolo,  # noqa: ANN001 - ultralytics YOLO
    recog: JerseyRecognizer,
    tmp_dir: Path,
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
    meta.append({"_n_persons": n_persons})  # funnel bookkeeping
    return meta


def _load_yolo():  # noqa: ANN202
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = YOLO("yolov8s.pt")
    y.to(device)
    return y


def run_spotcheck(match_id: str, chunk_key: str, n: int, seed: int) -> None:
    """Dump ~``n`` legible-candidate crops from ONE chunk spanning the confidence range to freeze
    the anchor threshold by manual verification."""
    match = registry.get(match_id)
    df = match.load_aligned()
    dfc = df[df["chunk"] == chunk_key]
    cls = classify_chunk(dfc)
    close_frames = sorted(int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_CLOSE_UP])
    print(f"{chunk_key}: {len(close_frames)} close-up frames of {len(cls)} sampled")

    yolo = _load_yolo()
    recog = JerseyRecognizer.from_checkpoint(CKPT)
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


def run_full(match_id: str, threshold: float) -> None:
    """Full-match yield + propagation-feasibility measurement at the frozen ``threshold``."""
    match = registry.get(match_id)
    df = match.load_aligned()
    fps = {ck: match.chunk_fps(ck) for ck in df["chunk"].unique()}

    yolo = _load_yolo()
    recog = JerseyRecognizer.from_checkpoint(CKPT)
    anchors_dir = OUT_DIR / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
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
        recs = detect_and_read(match_id, chunk_key, close_frames, yolo, recog, tmp)
        n_persons = sum(r.get("_n_persons", 0) for r in recs)
        crops = [r for r in recs if "pred" in r]
        legible = [r for r in crops if not r["illegible"]]
        anchors = [r for r in legible if r["conf"] >= threshold]

        # Propagation: is there a live_wide frame within +/- NEAR_WINDOW_S seconds?
        win = int(round(NEAR_WINDOW_S * fps[chunk_key]))
        attachable = 0
        team_matchable = 0
        for a in anchors:
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

        # Collapse into distinct close-up shots (contiguous close-up sampled-frame runs).
        n_shots, n_shots_anchor = _shot_yield(close_frames, anchors)

        for a in anchors:  # persist anchor crops for audit
            src = Path(a["crop_path"])
            if src.exists():
                shutil.copy(src, anchors_dir / f"{chunk_key}_f{a['frame']}_n{a['pred']:02d}"
                            f"_c{a['conf']:.3f}.jpg")
        for r in crops:
            r["chunk"] = chunk_key
            r.pop("crop_path", None)
            rows.append(r)
        per_chunk[chunk_key] = {
            "close_up_frames": len(close_frames),
            "persons_detected": int(n_persons),
            "eligible_crops": len(crops),
            "legible_reads": len(legible),
            "anchors": len(anchors),
            "anchors_attachable_2s": attachable,
            "anchors_team_matchable": team_matchable,
            "closeup_shots": n_shots,
            "shots_with_anchor": n_shots_anchor,
        }
        c = per_chunk[chunk_key]
        print(f"{chunk_key}: close {c['close_up_frames']:5d} | persons {c['persons_detected']:5d} "
              f"| eligible {c['eligible_crops']:4d} | legible {c['legible_reads']:4d} "
              f"| anchors {c['anchors']:3d} | attach {attachable:3d} | shots "
              f"{n_shots_anchor}/{n_shots}")

    _write_full_report(match_id, threshold, per_chunk, rows)


def _shot_yield(close_frames: list[int], anchors: list[dict]) -> tuple[int, int]:
    """Distinct close-up shots and how many carry >=1 anchor (contiguous run segmentation)."""
    if not close_frames:
        return 0, 0
    shots: list[tuple[int, int]] = []
    lo = prev = close_frames[0]
    for f in close_frames[1:]:
        if f - prev > SHOT_GAP_FRAMES:
            shots.append((lo, prev))
            lo = f
        prev = f
    shots.append((lo, prev))
    anchor_frames = {a["frame"] for a in anchors}
    with_anchor = sum(any(lo <= af <= hi for af in anchor_frames) for lo, hi in shots)
    return len(shots), with_anchor


def _write_full_report(match_id: str, threshold: float, per_chunk: dict, rows: list[dict]) -> None:
    """Aggregate the funnel across chunks and write the JSON yield artifact."""
    keys = ["close_up_frames", "persons_detected", "eligible_crops", "legible_reads",
            "anchors", "anchors_attachable_2s", "anchors_team_matchable",
            "closeup_shots", "shots_with_anchor"]
    total = {k: int(sum(c[k] for c in per_chunk.values())) for k in keys}
    numbers = pd.Series(
        [r["pred"] for r in rows if r["conf"] >= threshold and not r["illegible"]]
    )
    out = {
        "match": match_id,
        "checkpoint": str(CKPT),
        "threshold": threshold,
        "min_box_h_px": MIN_BOX_H,
        "coco_conf": COCO_CONF,
        "near_window_s": NEAR_WINDOW_S,
        "per_chunk": per_chunk,
        "total": total,
        "anchor_number_histogram": {int(k): int(v) for k, v in numbers.value_counts().items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "closeup_anchor_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "closeup_reads.csv", index=False)
    print("\nTOTAL FUNNEL:", json.dumps(total, indent=2))
    print(f"artifacts -> {OUT_DIR}")


def main() -> None:
    """CLI entry point (GPU; one job at a time)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="brighton_manutd")
    ap.add_argument("--mode", choices=["spotcheck", "full"], required=True)
    ap.add_argument("--chunk", default="h1_chunk_000", help="spotcheck: which chunk to sample")
    ap.add_argument("--n", type=int, default=24, help="spotcheck: crops to dump")
    ap.add_argument("--threshold", type=float, default=0.90, help="full: frozen anchor confidence")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "spotcheck":
        run_spotcheck(args.match, args.chunk, args.n, args.seed)
    else:
        run_full(args.match, args.threshold)


if __name__ == "__main__":
    main()
