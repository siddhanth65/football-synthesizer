"""P2 diagnosis: prove the partial-view bias mechanism before building an imputer.

Two questions, answered from the aligned tracking alone:
1. **How many players do we actually see?** Per-team visible-count distribution per frame — this sets
   what an imputer must reconstruct, and whether high-visibility frames exist to learn/validate from.
2. **Does the defensive line read higher when fewer defenders are visible?** If ``def_line_height`` rises
   as the visible defender count falls, the ~+11 m inflation is a *visibility artifact* (deep defenders
   off-screen), which imputation can fix — not a broken metric. We bin frames by visible-defender count
   and report the mean line height per bin + the rank correlation.

Run: ``python -m tools.impute_diagnose`` (all processed matches).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.registry import matches
from fingerprint.structural_metrics import DEF_LINE_QUANTILE, attacking_coord, resolve_attack_directions
from fingerprint.theory_metrics import complete_directions

PLAYERS = ("player", "goalkeeper")


def visible_counts(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per (chunk, frame, team) count of visible outfield/keeper players."""
    pl = aligned[aligned["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
    return pl.groupby(["chunk", "frame", "team"]).size().reset_index(name="n_vis")


def line_vs_visibility(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per frame: the *defending* team's visible-defender count and the line height it produces.

    For each frame we take each team as the defending side, compute its visible count and its
    ``def_line_height`` (deep quantile of attacking-x, the same estimator the metric engine uses), and
    return one row per (frame, team) so we can bin line height by how many players were seen.
    """
    rows = []
    groups = aligned.groupby("chunk") if "chunk" in aligned.columns else [(None, aligned)]
    for ck, g in groups:
        dirs = complete_directions(resolve_attack_directions(g))
        if len(dirs) < 2:
            continue
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs or len(fg) < 2:
                continue
            # line height in the team's OWN defensive frame: deep = low attacking-x
            ax = attacking_coord(fg["pitch_x"].to_numpy(), dirs[team])
            line = float(np.percentile(ax, DEF_LINE_QUANTILE))
            rows.append({"chunk": ck, "frame": int(fr), "team": team, "n_vis": len(fg),
                         "def_line_height": line})
    return pd.DataFrame(rows)


def _spearman(x, y) -> float:
    if len(x) < 3 or len(set(x)) < 2:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    print("=== P2 diagnosis: visibility + line-height bias ===\n")
    all_lv = []
    for m in matches(processed_only=True):
        aligned = m.load_aligned()
        vc = visible_counts(aligned)
        print(f"[{m.id}] visible players per team-frame:")
        for t in (0, 1):
            s = vc[vc["team"] == t]["n_vis"]
            if s.empty:
                continue
            name = m.teams[t]
            print(f"    {name:<10} mean={s.mean():4.1f}  median={s.median():2.0f}  p90={s.quantile(.9):2.0f}"
                  f"  max={s.max():2d}  frames>=9={int((s>=9).sum()):4d}  frames>=10={int((s>=10).sum()):4d}"
                  f"  n={len(s)}")
        lv = line_vs_visibility(aligned)
        lv["match"] = m.id
        all_lv.append(lv)
    lv = pd.concat(all_lv, ignore_index=True)
    print("\n=== defensive-line height binned by visible-defender count (all matches) ===")
    lv["bin"] = pd.cut(lv["n_vis"], [0, 4, 6, 8, 20], labels=["<=4", "5-6", "7-8", ">=9"])
    tbl = lv.groupby("bin", observed=True)["def_line_height"].agg(["mean", "count"])
    print(tbl.round(1).to_string())
    rho = _spearman(lv["n_vis"].to_numpy(), lv["def_line_height"].to_numpy())
    print(f"\n  Spearman(n_visible, def_line_height) = {rho:+.3f}")
    print("  Negative => fewer visible defenders -> higher line (the partial-view inflation). "
          "That is the bias imputation targets.")


if __name__ == "__main__":
    main()
