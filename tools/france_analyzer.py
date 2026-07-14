"""France Tier-A tactical profile: forecast tendency distributions vs Iraq, with 80% intervals."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthesizer.predict import predict_tendencies

FR, SEN = "#0055A4", "#00853F"
LABELS = {"def_line_height": "Def-line height (m)", "buildup_height": "Build-up height (m)",
          "width": "Width (m)", "attacking_third_share": "Attacking-third share",
          "wing_share": "Wing share", "compactness": "Compactness (m)"}


def main() -> None:
    pos = pd.read_parquet("outputs/france_iraq/final/match_aligned.parquet")
    fr = predict_tendencies(pos, 0, tier="A")
    sen = predict_tendencies(pos, 1, tier="A")
    keys = [k for k in LABELS if k in fr]
    y = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(11, 6))
    for d, col, name, off in [(fr, FR, "France", -0.16), (sen, SEN, "Iraq", 0.16)]:
        means = [d[k]["mean"] for k in keys]
        lo = [d[k]["mean"] - d[k]["lo"] for k in keys]
        hi = [d[k]["hi"] - d[k]["mean"] for k in keys]
        ax.errorbar(means, y + off, xerr=[lo, hi], fmt="o", color=col, capsize=4, ms=8,
                    label=name, lw=2)
        for k, m, yy in zip(keys, means, y + off):
            ax.text(m, yy + 0.07, f"{m:.1f}" if m > 1 else f"{m:.2f}", ha="center", fontsize=8, color=col)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in keys])
    ax.invert_yaxis()
    ax.set_title("France Tier-A tactical forecast (vs Iraq) — mean + 80% interval, n=11 chunks\n"
                 "France: high line, high build-up, attacks the attacking third & wings",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("value (metres or share)"); ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.2)
    out = Path("results/france_tendencies.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
