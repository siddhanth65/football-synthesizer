"""Per-phase line-height C6: France's build-up/progression/final-third + block heights vs FIFA's PMSR.

The real FIFA validation, finally with a ball: ball location buckets the in-possession phase and the
defending team's line buckets the block, so France's per-phase defensive-line height can be compared
directly to FIFA's published figures (build-up 44 / final-third 59 / mid-block 38 / low-block 19 m).
Uses v4 ball tracks + the kit-anchored positions; direction is ball-anchored per chunk.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fingerprint.phase_metrics import ball_phase_fingerprint
from fingerprint.structural_metrics import resolve_attack_directions_from_ball
from generator.ball import assign_possession

# FIFA France published lines (m) per phase.
FIFA = {"build_up": 44, "progression": None, "final_third": 59, "high_press": None,
        "mid_block": 38, "low_block": 19}


def main() -> None:
    aligned = pd.read_parquet("outputs/france_iraq/final/match_aligned.parquet")
    parts = []
    for f in sorted(glob.glob("outputs/france_iraq/final/ball/ball_*_chunk*.parquet")):
        m = re.search(r"ball_(h\d)_chunk(\d+)", f)
        ck = f"{m.group(1)}_chunk_{m.group(2)}"
        ball = pd.read_parquet(f)
        pos = aligned[aligned["chunk"] == ck]
        if pos.empty or ball.empty:
            continue
        pos = pos[(pos["calib_error_m"] <= 1.0) & pos["pitch_x"].notna()]
        dirs = resolve_attack_directions_from_ball(pos, ball)
        if len(dirs) < 2:
            continue
        poss = assign_possession(ball, pos, smooth=True)
        z = ball_phase_fingerprint(ball, poss, pos)
        if not z.empty:
            z["chunk"] = ck
            parts.append(z)
    if not parts:
        print("no phase data yet — ball tracks missing/too sparse")
        return
    allz = pd.concat(parts, ignore_index=True)
    # France = team 0; aggregate per phase (frame-weighted)
    fr = allz[allz["team"] == 0]
    agg = (fr.groupby("phase").apply(
        lambda g: pd.Series({"def_line": np.average(g["def_line_height"], weights=g["frames"]),
                             "frames": g["frames"].sum()}), include_groups=False)
        .reset_index())
    print("=== France per-phase defensive-line height: ours vs FIFA ===\n")
    print(f"  {'phase':<14}{'ours(m)':>9}{'FIFA(m)':>9}{'frames':>8}")
    for r in agg.itertuples(index=False):
        fifa = FIFA.get(r.phase)
        fs = f"{fifa:>9}" if fifa else f"{'-':>9}"
        print(f"  {r.phase:<14}{r.def_line:>9.1f}{fs}{int(r.frames):>8}")
    Path("outputs/france_iraq/final").mkdir(parents=True, exist_ok=True)
    agg.to_parquet("outputs/france_iraq/final/france_phase_c6.parquet", index=False)
    print("\nwrote outputs/france_iraq/final/france_phase_c6.parquet")


if __name__ == "__main__":
    main()
