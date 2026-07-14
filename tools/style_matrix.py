"""Cross-match team style-distance matrix (Tactical DNA) -> heatmap + clustering callout.

Wasserstein distance between teams' attack-normalised spatial marginals, across BOTH ingested matches.
The first cross-match comparison (the C5 / opponent-matchup seed): which teams occupy space alike.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from core.registry import matches
from fingerprint.structural_metrics import attacking_coord, resolve_attack_directions


def marginals(path, names) -> dict:
    pos = pd.read_parquet(path)
    groups = pos.groupby("chunk") if "chunk" in pos.columns else [(None, pos)]
    acc = {0: ([], []), 1: ([], [])}
    for _, g in groups:
        dirs = resolve_attack_directions(g)
        pl = g[g["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
        for t, gg in pl.groupby("team"):
            t = int(t)
            d = dirs.get(t)
            if t not in (0, 1) or d is None:
                continue
            acc[t][0].extend(attacking_coord(gg["pitch_x"].to_numpy(), d))
            acc[t][1].extend(gg["pitch_y"].to_numpy())
    return {names[0]: acc[0], names[1]: acc[1]}


def main() -> None:
    teams: dict = {}
    for reg in matches(processed_only=True):
        teams.update(marginals(reg.aligned, list(reg.teams)))
    labels = list(teams)
    n = len(labels)
    mat = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            mat[i, j] = (wasserstein_distance(teams[a][0], teams[b][0])
                         + wasserstein_distance(teams[a][1], teams[b][1]))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mat, cmap="RdYlGn_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontweight="bold")
    ax.set_title("Tactical-DNA style distance (m) — across both matches\n"
                 "France≈Man City (dominant) · Iraq≈Man Utd (reactive)", fontsize=12, fontweight="bold")
    fig.colorbar(im, label="spatial style distance (lower = more alike)")
    out = Path("results/style_matrix.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
