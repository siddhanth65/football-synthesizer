"""Final France-Senegal C6 graphic: line-height stability + our France vs FIFA published line/width."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FR = "#0055A4"   # France navy
SEN = "#00853F"  # Senegal green
# per-chunk def_line (from analyze_fra_sen.py); chunk_000 of h1 is kickoff (boundary noise)
H1_FR = [39, 59, 62, 55, 57, 43]; H1_SEN = [np.nan, 27, 24, 34, 28, 42]
H2_FR = [53, 50, 53, 47, 52];     H2_SEN = [32, 35, 34, 37, 34]
FIFA_FR_LINE = {"Build-up": 44, "Final Third": 59, "Mid Block": 38, "Low Block": 19}


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("France-Senegal — line-height stability + C6 vs FIFA (complete labels, kit-anchored)",
                 fontsize=13, fontweight="bold")

    # 1. per-chunk line height across both halves
    fr = H1_FR + H2_FR; sen = H1_SEN + H2_SEN
    x = np.arange(len(fr))
    ax1.plot(x, fr, "o-", color=FR, label="France", lw=2)
    ax1.plot(x, sen, "s-", color=SEN, label="Senegal", lw=2)
    ax1.axvline(5.5, ls="--", color="#888"); ax1.text(2.5, 70, "Half 1", ha="center")
    ax1.text(8, 70, "Half 2 (CV 5%)", ha="center", fontweight="bold")
    ax1.set_title("Defensive-line height per chunk", fontsize=11, fontweight="bold")
    ax1.set_xlabel("chunk (across both halves)"); ax1.set_ylabel("line height (m, 0=own goal)")
    ax1.set_ylim(0, 75); ax1.legend(); ax1.grid(alpha=0.2)
    ax1.annotate("kickoff\nboundary", (0, 39), (0.3, 20), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#a00"), color="#a00")

    # 2. France line vs FIFA published phase lines
    phases = list(FIFA_FR_LINE)
    ax2.bar(np.arange(len(phases)), list(FIFA_FR_LINE.values()), 0.5, color="#1f77b4",
            label="FIFA France (per phase)")
    ax2.axhline(53.0, color=FR, lw=2.5, label="ours France (overall 53 m)")
    ax2.fill_between([-0.5, 3.5], 47, 53, color=FR, alpha=0.15)
    ax2.set_title("France line: ours (overall) vs FIFA (per phase)", fontsize=11, fontweight="bold")
    ax2.set_xticks(np.arange(len(phases))); ax2.set_xticklabels(phases)
    ax2.set_ylabel("line height (m)"); ax2.set_ylim(0, 70); ax2.legend(loc="upper right")
    ax2.text(1.5, 10, "in FIFA range (19-59); reads high vs build-up/mid\n"
             "because broadcast shows ~6/11 (deep defenders off-frame)",
             ha="center", fontsize=8, color="#555")

    out = Path("results/fra_sen_c6_final.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
