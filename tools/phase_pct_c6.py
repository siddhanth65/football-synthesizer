"""Clean same-match C6: our CV phase-time distribution for France vs each match's own FIFA phase %.

For each France match we classify every possession frame into a phase (ball-driven) and measure the
**share of time** France spends in each, then compare to the FIFA PMSR's published phase % for the SAME
match (apples-to-apples, unlike the earlier cross-opponent line-height check). Our classifier has 6
coarse buckets, so we normalise within in-possession (build-up/progression/final-third) and within
out-of-possession (high/mid/low) and compare those distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.registry import get, matches
from fingerprint.phase_metrics import ball_phase_by_frame
from fingerprint.structural_metrics import resolve_attack_directions_from_ball
from generator.ball import assign_possession

MATCHES = [m.id for m in matches() if "France" in m.teams]
IN_PH = ["build_up", "progression", "final_third"]
OUT_PH = ["high_press", "mid_block", "low_block"]


def cv_phase_pct(match: str) -> dict[str, float] | None:
    """France's CV phase-time shares (normalised within in-poss and within out-of-poss)."""
    reg = get(match)
    if not reg.processed:
        return None
    aligned = reg.load_aligned()
    counts = {p: 0 for p in IN_PH + OUT_PH}
    for ck, f in reg.ball_chunks():
        ball = pd.read_parquet(f)
        pos = aligned[(aligned["chunk"] == ck) & (aligned["calib_error_m"] <= 1.0)].dropna(subset=["pitch_x"])
        if pos.empty or ball.empty:
            continue
        dirs = resolve_attack_directions_from_ball(pos, ball)
        if len(dirs) < 2:
            continue
        poss = assign_possession(ball, pos, smooth=True)
        ph = ball_phase_by_frame(ball, poss, pos)
        fr = ph[ph["team"] == 0]  # France = team 0 (kit-anchored)
        for p in fr["phase"]:
            if p in counts:
                counts[p] += 1
    in_tot = sum(counts[p] for p in IN_PH) or 1
    out_tot = sum(counts[p] for p in OUT_PH) or 1
    out = {p: 100 * counts[p] / in_tot for p in IN_PH}
    out.update({p: 100 * counts[p] / out_tot for p in OUT_PH})
    return out


def fifa_phase_pct(match: str, france_idx: int) -> dict[str, float]:
    """FIFA France phase % mapped to our 6 buckets, normalised the same way."""
    ph = json.loads(Path(f"outputs/pmsr/{match}.json").read_text())["phases"]

    def g(k):
        v = ph.get(k)
        return v[france_idx] if isinstance(v, list) else 0
    raw = {
        "build_up": g("build_up_unopposed") + g("build_up_opposed"),
        "progression": g("progression"), "final_third": g("final_third"),
        "high_press": g("high_press") + g("mid_press") + g("high_block"),
        "mid_block": g("mid_block"), "low_block": g("low_block") + g("low_press"),
    }
    in_tot = sum(raw[p] for p in IN_PH) or 1
    out_tot = sum(raw[p] for p in OUT_PH) or 1
    out = {p: 100 * raw[p] / in_tot for p in IN_PH}
    out.update({p: 100 * raw[p] / out_tot for p in OUT_PH})
    return out


def main() -> None:
    roster = json.loads(Path("data/france_roster.json").read_text())["matches"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    summary = []
    for ax, m in zip(axes, MATCHES):
        cv = cv_phase_pct(m)
        fidx = 0 if roster[m]["france_is_home"] else 1
        fifa = fifa_phase_pct(m, fidx)
        if cv is None:
            ax.set_title(f"{m}: no CV ball data"); continue
        phases = IN_PH + OUT_PH
        x = np.arange(len(phases)); w = 0.38
        ax.bar(x - w / 2, [fifa[p] for p in phases], w, label="FIFA", color="#1f77b4")
        ax.bar(x + w / 2, [cv[p] for p in phases], w, label="ours (CV)", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=8)
        ax.set_title(f"France vs {roster[m]['opponent']}", fontweight="bold")
        ax.set_ylabel("% (norm. within in/out)"); ax.legend(fontsize=8)
        mae = np.mean([abs(cv[p] - fifa[p]) for p in phases])
        summary.append((m, mae))
        ax.text(0.5, 0.95, f"mean abs err {mae:.0f}pp", transform=ax.transAxes, ha="center",
                fontsize=9, color="#a00")
    fig.suptitle("Clean same-match C6 — France phase-time distribution: ours (CV) vs FIFA PMSR",
                 fontsize=13, fontweight="bold")
    out = Path("results/phase_pct_c6.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    print("\nphase-distribution mean abs error (pp) per match:")
    for m, mae in summary:
        print(f"  {m}: {mae:.1f}")


if __name__ == "__main__":
    main()
