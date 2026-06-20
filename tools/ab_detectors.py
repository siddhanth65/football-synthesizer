"""A/B a detector swap on identical frames: YOLOv8s-COCO vs RF-DETR-COCO.

Runs both detectors over the same sampled frames of a clip and reports the metrics that matter for
the freeze-frame generator: players found per frame, ball-detection rate, mean confidence, and speed.
Adopt RF-DETR only if these numbers actually move (the point the user raised: newer != better until
measured).

Run: ``python tools/ab_detectors.py --video "<clip>.mp4" --start 13025 --n 30 --step 2``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.extract import _build_detector  # noqa: E402


def _sample_frames(video: str, start: int, n: int, step: int):
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(video)
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * step)
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _eval(detector, frames) -> dict:
    counts, balls, confs = [], 0, []
    t0 = time.time()
    for f in frames:
        boxes, c, ball = detector.detect(f)
        counts.append(len(boxes))
        balls += int(ball is not None)
        if len(c):
            confs.append(float(np.mean(c)))
    dt = time.time() - t0
    return {
        "players_per_frame": float(np.mean(counts)) if counts else 0.0,
        "players_max": int(np.max(counts)) if counts else 0,
        "ball_rate": balls / len(frames) if frames else 0.0,
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "sec_per_frame": dt / len(frames) if frames else 0.0,
    }


def main() -> None:
    import torch  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=13025)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = _sample_frames(args.video, args.start, args.n, args.step)
    print(f"A/B on {len(frames)} frames (start={args.start} step={args.step}) device={device}\n")
    rows = {}
    for name in ("yolo", "rfdetr"):
        det = _build_detector(device, name)
        rows[name] = _eval(det, frames)
    hdr = f"{'detector':10} {'players/frame':>14} {'max':>5} {'ball_rate':>10} {'mean_conf':>10} {'s/frame':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:10} {r['players_per_frame']:14.1f} {r['players_max']:5d} "
              f"{r['ball_rate']:10.2f} {r['mean_conf']:10.2f} {r['sec_per_frame']:9.3f}")


if __name__ == "__main__":
    main()
