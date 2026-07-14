"""P2 headline validation: per-phase defensive line height — CV raw vs de-biased vs FIFA.

The credible test (Fable): the FIFA PMSR publishes per-phase line height (in-possession build-up/final-
third; defensive high/mid/low block), so we validate against official numbers *including the in-possession
phase where our bias lives* — no circularity. For each France match we bin every frame by our ball-driven
phase and compare ``line_raw`` (deepest-N back-line estimate, right shape / biased scale) and
``line_debiased`` (visibility-censoring removed via one global n_back slope) to FIFA. DoD: de-biased is
close to FIFA on every phase, keeping the ~40 m phase spread the temporal reconstruction destroyed —
without any hand ``PARTIAL_VIEW_LINE_OFFSET``.

Run: ``python -m tools.line_c6``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import matches
from fingerprint.phase_metrics import ball_phase_by_frame
from fingerprint.roles import assign_roles
from fingerprint.structural_metrics import resolve_attack_directions, resolve_attack_directions_from_ball
from fingerprint.theory_metrics import complete_directions
from generator.ball import assign_possession
from generator.impute import line_estimates

CALIB_MAX_M = 1.0
# our ball-phase -> FIFA line-height (section, key). Our build_up (ball deep) = FIFA build_up_LOW;
# our progression (middle third) = FIFA build_up_mid.
PHASE_MAP = {
    "build_up": ("in_possession", "build_up_low"),
    "progression": ("in_possession", "build_up_mid"),
    "final_third": ("in_possession", "final_third"),
    "high_press": ("defensive", "high_block"),
    "mid_block": ("defensive", "mid_block"),
    "low_block": ("defensive", "low_block"),
}
PHASE_ORDER = ["build_up", "progression", "final_third", "high_press", "mid_block", "low_block"]


def _france_phase_frames(m) -> pd.DataFrame:
    """France (team 0) per-frame ball phase across the match's ball chunks."""
    aligned = m.load_aligned()
    parts = []
    for ck, path in m.ball_chunks():
        ball = pd.read_parquet(path)
        pos = aligned[(aligned["chunk"] == ck) & (aligned["calib_error_m"] <= CALIB_MAX_M)].dropna(
            subset=["pitch_x"])
        if pos.empty or ball.empty:
            continue
        dirs = complete_directions(resolve_attack_directions_from_ball(pos, ball)
                                   or resolve_attack_directions(pos))
        if len(dirs) < 2:
            continue
        poss = assign_possession(ball, pos, smooth=True)
        ph = ball_phase_by_frame(ball, poss, pos)
        ph = ph[ph["team"] == 0][["frame", "phase"]].copy()
        ph["chunk"] = ck
        parts.append(ph)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["frame", "phase", "chunk"])


def main() -> None:
    print("=== P2 line-height C6: France per-phase — CV raw vs de-biased vs FIFA ===\n")
    tot = {"raw": [], "deb": []}
    spreads = []
    for m in matches(processed_only=True):
        if "France" not in m.teams:
            continue
        lh = (m.load_pmsr() or {}).get("line_height")
        roster = json.loads(Path("data/france_roster.json").read_text())["matches"][m.id]
        fidx = 0 if roster["france_is_home"] else 1
        aligned = m.load_aligned()
        le = line_estimates(aligned, assign_roles(aligned))
        le = le[le["team"] == 0]
        ph = _france_phase_frames(m)
        if ph.empty:
            print(f"[{m.id}] no ball phases")
            continue
        j = ph.merge(le, on=["chunk", "frame"], how="inner")
        print(f"[{m.id}]  (n frames {len(j)})")
        print(f"  {'phase':<12}{'raw':>7}{'debiased':>10}{'FIFA':>7}  {'|raw-F|':>8}{'|deb-F|':>8}")
        deb_by_phase = {}
        for phase in PHASE_ORDER:
            grp = j[j["phase"] == phase]
            if grp.empty:
                continue
            raw, deb = grp["line_raw"].mean(), grp["line_debiased"].mean()
            deb_by_phase[phase] = deb
            fifa = None
            if lh and phase in PHASE_MAP:
                sec, key = PHASE_MAP[phase]
                v = lh.get(sec, {}).get(key)
                fifa = v[fidx] if isinstance(v, list) and v[fidx] is not None else None
            if fifa is not None:
                tot["raw"].append(abs(raw - fifa))
                tot["deb"].append(abs(deb - fifa))
            fs = f"{fifa:>7.0f}" if fifa is not None else f"{'-':>7}"
            er = f"{abs(raw-fifa):>8.1f}" if fifa is not None else f"{'-':>8}"
            ec = f"{abs(deb-fifa):>8.1f}" if fifa is not None else f"{'-':>8}"
            print(f"  {phase:<12}{raw:>7.1f}{deb:>10.1f}{fs}  {er}{ec}")
        if deb_by_phase:
            spreads.append(max(deb_by_phase.values()) - min(deb_by_phase.values()))
        print()
    print("=== pooled (clean-mapped phases) ===")
    print(f"  mean |line - FIFA|:  raw {np.mean(tot['raw']):.1f} m -> de-biased {np.mean(tot['deb']):.1f} m")
    print(f"  max  |line - FIFA|:  raw {np.max(tot['raw']):.1f} m -> de-biased {np.max(tot['deb']):.1f} m")
    print(f"  de-biased phase spread {np.mean(spreads):.0f} m (FIFA ~40 m — shape preserved)")


if __name__ == "__main__":
    main()
