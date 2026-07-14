"""C6 per-phase graphic: France line height ours vs FIFA, split by framing reliability."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# phase: (ours, fifa, reliable?)  -- defensive phases are fully framed; in-possession follow-ball biased
DATA = [("Low Block", 22.7, 19, True), ("Mid Block", 40.4, 38, True),
        ("Build-up", 24.9, 44, False), ("Final Third", 73.7, 59, False)]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(DATA)); w = 0.38
    ours = [d[1] for d in DATA]; fifa = [d[2] for d in DATA]
    cols = ["#2ca02c" if d[3] else "#ff9900" for d in DATA]
    ax.bar(x - w / 2, fifa, w, color="#1f77b4", label="FIFA published")
    ax.bar(x + w / 2, ours, w, color=cols, label="ours (CV)")
    for i, d in enumerate(DATA):
        ax.text(i, max(d[1], d[2]) + 1.5, f"Δ{abs(d[1]-d[2]):.0f}m", ha="center", fontweight="bold",
                color=cols[i])
    ax.axvspan(-0.5, 1.5, color="#2ca02c", alpha=0.06)
    ax.axvspan(1.5, 3.5, color="#ff9900", alpha=0.06)
    ax.text(0.5, 68, "DEFENSIVE phases\nteam fully framed → MATCH (Δ2-4m)", ha="center",
            fontsize=9, color="#1a7a1a", fontweight="bold")
    ax.text(2.5, 68, "IN-POSSESSION phases\ncamera follows ball → biased", ha="center",
            fontsize=9, color="#cc7000", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DATA])
    ax.set_ylabel("defensive-line height (m, 0=own goal)"); ax.set_ylim(0, 80)
    ax.set_title("C6 — France line height: our CV (vs Iraq) vs FIFA PMSR (vs Senegal)\n"
                 "Cross-opponent check: defensive blocks (opponent-stable) match to 2-4 m; "
                 "in-possession biased by ball-following camera",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.2)
    out = Path("results/fra_sen_phase_c6.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
