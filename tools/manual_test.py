"""Manual visual test kit: turn a processed chunk into an eyeball-able gallery.

Renders, into one folder:
  * gallery_*.png  -- side-by-side (broadcast frame | top-down freeze frame) for a spread of accepted
    tactical frames, with the reconstructed pitch lines drawn on the video (yellow). This is the main
    "is it right?" check: yellow lines on the real markings = calibration ok; dots on players = good
    detection; red/blue = the two teams; grey x = officials; black ring = goalkeeper; white star = ball.
  * coverage.png   -- a timeline of the whole chunk: which sampled moments are usable tactical frames
    (green) vs detected-but-sparse (orange) vs no players found / replay-closeup (grey).
  * tracks.png     -- every player's path during the longest continuous tactical shot (each line = one
    persistent track id; check they don't jump across the pitch).

Run: ``python tools/manual_test.py --positions outputs/chunk000_dense.parquet \
        --tactical outputs/chunk000_tactical.parquet \
        --video "<...>/chunk_000.mp4" --out outputs/manual_test``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.visualize import render_overlays, render_tracks  # noqa: E402

FPS = 50.0
STRIDE = 5  # must match the extract --sample-every used to make the parquet


def _pick_accepted(tactical: pd.DataFrame, n: int) -> list[int]:
    acc = np.sort(tactical[tactical["accepted"] == True]["frame"].to_numpy())  # noqa: E712
    if len(acc) == 0:
        return []
    idx = np.linspace(0, len(acc) - 1, min(n, len(acc))).astype(int)
    return [int(acc[i]) for i in idx]


def _longest_accepted_run(tactical: pd.DataFrame, stride: int = STRIDE) -> list[int]:
    acc = sorted(int(f) for f in tactical[tactical["accepted"] == True]["frame"])  # noqa: E712
    best: list[int] = []
    cur: list[int] = []
    for f in acc:
        cur = cur + [f] if (cur and f - cur[-1] == stride) else [f]
        if len(cur) > len(best):
            best = cur[:]
    return best


def _coverage_plot(tactical: pd.DataFrame, out: Path, fps: float, stride: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = tactical["frame"].to_numpy()
    t_min = fr / fps / 60.0
    status = np.where(tactical["accepted"].to_numpy() == True, 2, 1)  # noqa: E712  2=accepted,1=sparse
    colors = np.where(status == 2, "#2ca02c", "#ff7f0e")
    fig, ax = plt.subplots(figsize=(14, 2.2))
    ax.scatter(t_min, np.zeros_like(t_min), c=colors, s=8, marker="|")
    acc = int((status == 2).sum())
    ax.set_title(f"chunk coverage: {acc} accepted tactical frames (green) / {len(fr)} with detections "
                 f"(orange=sparse) over {t_min.max():.1f} min")
    ax.set_xlabel("match time (min)")
    ax.set_yticks([])
    fig.tight_layout()
    p = out / "coverage.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--tactical", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/manual_test")
    ap.add_argument("--n", type=int, default=6, help="gallery frames (spread across accepted frames)")
    args = ap.parse_args()

    dense = pd.read_parquet(args.positions)
    tactical = pd.read_parquet(args.tactical)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

    calib = PnLCalibCalibrator()
    picks = _pick_accepted(tactical, args.n)
    overlays = render_overlays(dense, args.video, out, calibrator=calib, frames=picks)
    # rename the generic overlay_*.png to gallery_<min>m_* so the time is obvious in the filename
    for p in overlays:
        fr = int(p.stem.split("_")[-1])
        p.rename(out / f"gallery_{fr/FPS/60:04.1f}min_f{fr}.png")

    run = _longest_accepted_run(tactical)
    if run:
        render_tracks(dense[dense["frame"].isin(run)], out)  # tracks.png over one continuous shot
    cov = _coverage_plot(tactical, out, FPS, STRIDE)

    (out / "README.txt").write_text(
        "MANUAL TEST GALLERY -- what to check\n"
        "====================================\n"
        "gallery_*min_*.png  (one per sampled moment across the match):\n"
        "  LEFT (broadcast frame):\n"
        "   - YELLOW lines should sit on the real painted pitch lines  -> calibration is correct\n"
        "   - coloured dots should sit on players' feet                -> detection is correct\n"
        "   - RED vs BLUE                                              -> the two teams\n"
        "   - grey X = referee/official, BLACK ring = goalkeeper, YELLOW ring = ball-carrier\n"
        "   - white star = ball; small number = persistent track id\n"
        "  RIGHT (top-down freeze frame): the same players placed on a 105x68 pitch.\n"
        "   - left-right and goal ends should match the video; a near-goal keeper near that goal.\n\n"
        "coverage.png : green = usable tactical frames, orange = too few players. Shows how much of\n"
        "  the broadcast is live tactical play vs replays/close-ups (correctly skipped).\n\n"
        "tracks.png   : each line is one player's path during the longest continuous shot. Lines should\n"
        "  be smooth; a line teleporting across the pitch = an id switch.\n"
    )
    print(f"wrote {len(overlays)} gallery frames + coverage + tracks to {out}/ (see README.txt)")
    print("open:", out / "README.txt")


if __name__ == "__main__":
    main()
