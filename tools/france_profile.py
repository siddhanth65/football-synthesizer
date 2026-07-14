"""France multi-match profile: Tier-A tendencies per match + the opponent-conditioning signal (C5 seed).

With 3 France matches we can (a) tighten France's marginal tendency distribution (Tier A over matches,
not just chunks) and (b) show the *opponent term*: how France's shape shifts with the opponent's
defensive depth. This is the evidence the Tier-B (opponent-conditioned) synthesizer would learn from.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.registry import matches
from synthesizer.predict import predict_tendencies


def main() -> None:
    rows = []
    for reg in matches(processed_only=True):
        if "France" not in reg.teams:
            continue
        m = reg.id
        fifa = reg.load_pmsr()
        if fifa is None:
            continue
        cv = predict_tendencies(reg.load_aligned(), 0, tier="A")
        roster = json.loads(Path("data/france_roster.json").read_text())["matches"][m]
        oi = 1 if roster["france_is_home"] else 0
        rows.append({
            "match": m.replace("france_", ""), "opponent": roster["opponent"],
            "france_line": cv["def_line_height"]["mean"],
            "france_att3rd": cv["attacking_third_share"]["mean"],
            "france_wing": cv["wing_share"]["mean"],
            "opp_low_block": fifa["phases"].get("low_block", [None, None])[oi],
            "opp_mid_block": fifa["phases"].get("mid_block", [None, None])[oi],
        })
    df = pd.DataFrame(rows)
    Path("outputs/france_profile.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet("outputs/france_profile.parquet", index=False)
    print("=== France 3-match profile (CV France shape vs FIFA opponent block) ===\n")
    print(df.round(2).to_string(index=False))

    # Tier-A over matches (tighter than per-chunk): mean +/- spread of France's line across matches
    print(f"\nFrance marginal line height across matches: "
          f"{df['france_line'].mean():.0f} +/- {df['france_line'].std():.0f} m")

    # opponent-conditioning plot: France attacking-third share vs opponent low-block %
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["opp_low_block"], df["france_att3rd"] * 100, s=160, c="#0055A4", zorder=3)
    for r in df.itertuples(index=False):
        ax.annotate(f"vs {r.opponent}", (r.opp_low_block, r.france_att3rd * 100),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    if len(df) >= 2:
        z = np.polyfit(df["opp_low_block"], df["france_att3rd"] * 100, 1)
        xs = np.linspace(df["opp_low_block"].min() - 2, df["opp_low_block"].max() + 2, 10)
        ax.plot(xs, np.polyval(z, xs), "--", color="#888", label="trend")
    ax.set_xlabel("opponent time in LOW BLOCK (%, FIFA)")
    ax.set_ylabel("France attacking-third share (%, CV)")
    ax.set_title("Opponent-conditioning signal (C5 seed)\n"
                 "France commits further forward against deeper-sitting opponents", fontweight="bold")
    ax.grid(alpha=0.2); ax.legend()
    out = Path("results/france_opponent_signal.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out} + outputs/france_profile.parquet")


if __name__ == "__main__":
    main()
