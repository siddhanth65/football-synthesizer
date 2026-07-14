"""Interactive ball annotator — label the ball in a spread of live-play frames to fine-tune a detector.

The pretrained TrackNetV2/WASB models don't transfer to this 1024x576 broadcast (see STATUS), so we
fine-tune on a few hundred frames annotated from *this* footage. This tool picks diverse live-play frames
(accepted-tactical, many players, away from graphics), shows each, and you click the ball:

  * **left-click the ball centre**  -> records (frame, x, y, visible=1)
  * **press Enter (no click)**      -> records visible=0 (ball off-screen / not findable)
  * close the window                -> stops (progress is saved; rerun to resume)

Writes/resumes an annotations CSV. Aim for ~300 frames (~30-45 min). Then run ``tools/finetune_ball.py``.

Run:
    python tools/annotate_ball.py --video "<...>/chunk_000.mp4" \
        --positions outputs/chunk000_dense.parquet --out outputs/ball_annotations/chunk_000.csv --n 300
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def pick_live_frames(positions: pd.DataFrame, n: int, *, min_players: int = 12) -> list[int]:
    """Spread of frames with many on-pitch players (live tactical play, not replays/graphics)."""
    pc = positions[positions["role"].isin(["player", "goalkeeper"])].dropna(
        subset=["pitch_x"]).groupby("frame").size()
    good = np.sort(pc[pc >= min_players].index.to_numpy())
    if len(good) == 0:
        good = np.sort(pc.index.to_numpy())
    idx = np.linspace(0, len(good) - 1, min(n, len(good))).astype(int)
    return [int(good[i]) for i in idx]


def _load_done(out: Path) -> set[int]:
    if not out.exists():
        return set()
    return set(pd.read_csv(out)["frame"].astype(int))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--positions", required=True, help="dense parquet for the same video (frame picker)")
    ap.add_argument("--out", required=True, help="annotations CSV (resumable)")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend (local GUI)
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = pick_live_frames(pd.read_parquet(args.positions), args.n)
    done = _load_done(out)
    todo = [f for f in frames if f not in done]
    print(f"{len(frames)} target frames, {len(done)} already done, {len(todo)} to go.")
    if not out.exists():
        with out.open("w", newline="") as fh:
            csv.writer(fh).writerow(["frame", "x", "y", "visible"])

    cap = cv2.VideoCapture(args.video)
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, fr in enumerate(todo):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, bgr = cap.read()
        if not ok:
            continue
        ax.clear()
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"[{i+1}/{len(todo)}] frame {fr} — click the BALL, or press Enter if not visible")
        ax.axis("off")
        plt.tight_layout()
        plt.draw()
        pts = plt.ginput(1, timeout=0)  # one left-click, or [] on Enter
        if not plt.fignum_exists(fig.number):
            print("window closed — stopping (progress saved).")
            break
        x, y, vis = (pts[0][0], pts[0][1], 1) if pts else (-1, -1, 0)
        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([fr, round(float(x), 1), round(float(y), 1), vis])
    cap.release()
    plt.close(fig)
    n_done = len(_load_done(out))
    print(f"saved {n_done} annotations -> {out}")


if __name__ == "__main__":
    main()
