"""Top-down tactical-projection sheet for eyeball cross-verification.

Plots calibrated player positions (bird's-eye) for a few well-populated frames.
If the projected shapes look like real football formations (two banks, keepers
deep, sane spread), the calibration + tracking are sound.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

UTD, CITY, REF = "#DA291C", "#6CABDD", "#888"


def pitch(ax):
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, ec="#2a7", lw=1.5))
    ax.plot([52.5, 52.5], [0, 68], color="#2a7", lw=1)
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, ec="#2a7", lw=1))
    for x0 in (0, 105 - 16.5):
        ax.add_patch(plt.Rectangle((x0, 13.84), 16.5, 40.3, fill=False, ec="#2a7", lw=1))
    ax.set_xlim(-3, 108); ax.set_ylim(-3, 71); ax.set_aspect("equal"); ax.axis("off")


def main() -> None:
    df = pd.read_parquet("outputs/match_anchored_dense.parquet")
    c = df[df.chunk == "chunk_006"]
    # frames with a full-ish, clean cast (closest to 22 outfield+keepers)
    counts = c.groupby("frame").size()
    frames = (counts - 22).abs().sort_values().index[:4]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Top-down cross-verification — chunk_006 (calibrated bird's-eye)\n"
                 "red=Man Utd  blue=Man City  grey=ref/unknown  |  do the shapes look like real formations?",
                 fontsize=13, fontweight="bold")
    for ax, fr in zip(axes.ravel(), sorted(frames)):
        pitch(ax)
        f = c[c.frame == fr]
        for _, r in f.iterrows():
            col = {0: UTD, 1: CITY}.get(int(r.team), REF)
            mk = "s" if r.is_keeper else ("D" if r.is_actor else "o")
            ax.scatter(r.pitch_x, r.pitch_y, c=col, marker=mk, s=90,
                       edgecolors="k", linewidths=0.5, zorder=3)
        err = f.calib_error_m.median()
        ax.set_title(f"frame {fr}  |  {len(f)} players  |  calib err {err:.2f} m", fontsize=10)

    out = Path("results/verify_topdown.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
