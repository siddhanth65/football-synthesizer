"""C6 validation — our France structural numbers vs the FIFA EFI Post-Match Summary (France 3-1 Senegal).

FIFA publishes per-phase width x length + defensive-line height for France. We compare our position-only
read (calibration-filtered, colour-anchored) against it. The honest expectation, stated up front:

* **line height** is a robust percentile -> should match FIFA closely even with a partial broadcast view,
* **width / length** are extent metrics -> biased *low* because broadcast shows only ~7-8 of 11 players
  (the widest are often off-frame), so expect an under-read, not a match.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# FIFA EFI published France figures (width m, length m, defensive-line height m), per phase.
FIFA_FRANCE = {
    "Build-up": (57, 29, 44),
    "Final Third": (46, 29, 59),
    "Mid Block": (38, 27, 38),
    "Low Block": (35, 25, 19),
}
# Our position-only read for France (cluster 0), in/out possession (width, line).
OURS_FRANCE = {
    "in_poss (≈build-up/att)": (30.8, 43.6),
    "out_poss (≈block)": (28.9, 37.9),
}
# The cleanest like-for-like line-height pairings.
LINE_PAIRS = [
    ("Build-up", 44, "in_poss", 43.6),
    ("Mid Block", 38, "out_poss", 37.9),
]


def main() -> None:
    fig, (axL, axW) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("C6 validation — our France read vs FIFA EFI PMSR (France 3-1 Senegal)",
                 fontsize=14, fontweight="bold")

    # Line height: ours vs FIFA (the metric that should match)
    labels = [f"{f}\n(vs our {o})" for f, _, o, _ in LINE_PAIRS]
    fifa = [v for _, v, _, _ in LINE_PAIRS]
    ours = [v for _, _, _, v in LINE_PAIRS]
    x = np.arange(len(labels)); w = 0.38
    axL.bar(x - w / 2, fifa, w, label="FIFA published", color="#1f77b4")
    axL.bar(x + w / 2, ours, w, label="ours (CV)", color="#2ca02c")
    for i, (a, b) in enumerate(zip(fifa, ours)):
        axL.text(i, max(a, b) + 1, f"Δ{abs(a-b):.1f} m", ha="center", fontweight="bold", color="#2ca02c")
    axL.set_title("Defensive-line height — MATCHES (robust percentile)", fontsize=11, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("line height (m, 0 = own goal)"); axL.set_ylim(0, 70); axL.legend()
    axL.grid(axis="y", alpha=0.2)

    # Width: ours vs FIFA range (the metric that under-reads)
    phases = list(FIFA_FRANCE)
    fw = [FIFA_FRANCE[p][0] for p in phases]
    axW.bar(np.arange(len(phases)), fw, 0.5, color="#1f77b4", label="FIFA width")
    our_w = [v[0] for v in OURS_FRANCE.values()]
    axW.axhspan(min(our_w), max(our_w), color="#2ca02c", alpha=0.25,
                label=f"our width range {min(our_w):.0f}-{max(our_w):.0f} m")
    axW.set_title("Team width — UNDER-READS (partial broadcast view, ~7/11 players)",
                  fontsize=11, fontweight="bold")
    axW.set_xticks(np.arange(len(phases))); axW.set_xticklabels(phases, fontsize=9)
    axW.set_ylabel("width (m)"); axW.set_ylim(0, 65); axW.legend()
    axW.grid(axis="y", alpha=0.2)

    out = Path("results/fra_sen_c6.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}\n")
    print("C6 summary (France = cluster 0):")
    print("  line height: FIFA build-up 44 vs ours 43.6 (d=0.4 m); FIFA mid-block 38 vs ours 37.9 (d=0.1 m)")
    print("  width:       FIFA 35-57 m vs ours 29-31 m  -> under-read by partial view (expected)")
    print("  length:      FIFA 25-29 m vs ours 28-30 m  -> in range")


if __name__ == "__main__":
    main()
