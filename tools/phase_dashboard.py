"""Visualise the ball-driven phase split (FIFA EFI pp.4/27-28 style) for one chunk."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UTD, CITY = "#DA291C", "#6CABDD"
IN_ORDER = ["build_up", "progression", "final_third"]
OUT_ORDER = ["low_block", "mid_block", "high_press"]
NAMES = {0: "Man Utd", 1: "Man City"}


def main() -> None:
    z = pd.read_parquet("outputs/phases_chunk006.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Ball-driven phase split — chunk_006  |  defensive-line height (m) per FIFA phase",
                 fontsize=13, fontweight="bold")

    for ax, (in_poss, order, title) in zip(
        axes, [(True, IN_ORDER, "In possession (by ball location)"),
               (False, OUT_ORDER, "Out of possession (by block height)")]):
        sub = z[z["in_possession"] == in_poss]
        x = np.arange(len(order))
        w = 0.38
        for k, team in enumerate((0, 1)):
            t = sub[sub["team"] == team].set_index("phase")
            heights = [t.loc[p, "def_line_height"] if p in t.index else np.nan for p in order]
            frames = [int(t.loc[p, "frames"]) if p in t.index else 0 for p in order]
            bars = ax.bar(x + (k - 0.5) * w, heights, w, color=(UTD if team == 0 else CITY),
                          label=NAMES[team])
            for b, fr in zip(bars, frames):
                if not np.isnan(b.get_height()):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={fr}",
                            ha="center", fontsize=7, color="#444")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([p.replace("_", " ") for p in order])
        ax.set_ylabel("defensive-line height (m, 0=own goal)"); ax.set_ylim(0, 90)
        ax.legend(); ax.grid(axis="y", alpha=0.2)

    out = Path("results/phase_dashboard.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
