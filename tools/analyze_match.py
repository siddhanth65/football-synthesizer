"""Full structural analysis of France-Iraq from the re-extracted (complete-label) halves.

Loads both halves' per-chunk positions, makes team identity globally consistent + kit-anchored
(France = navy = 0, Iraq = white = 1) via :func:`generator.team_anchor.align_teams_by_color`, then
computes the per-team structural fingerprint and the honest C6 comparison against FIFA's published France
figures. Direction is keeper-resolved per chunk (reliable now that team labels are complete).

    python tools/analyze_fra_sen.py
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fingerprint.structural_metrics import (
    compute_metrics_table,
    resolve_attack_directions,
    team_fingerprint,
)
from generator.team_anchor import align_teams_by_color

HALVES = [("h1", "outputs/france_iraq/h1/match", "matches/france_iraq/h1"),
          ("h2", "outputs/france_iraq/h2/match", "matches/france_iraq/h2")]
FIFA_FRANCE = {"Build-up": (57, 29, 44), "Final Third": (46, 29, 59),
               "Mid Block": (38, 27, 38), "Low Block": (35, 25, 19)}


def load_half(match_dir: str, chunks_dir: str) -> pd.DataFrame:
    """Load a half's per-chunk dense parquets, tag chunk, and kit-anchor teams (France=0 navy)."""
    dfs = []
    for f in sorted(glob.glob(f"{match_dir}/chunk_*_dense.parquet")):
        ck = "chunk_" + re.search(r"chunk_(\d+)", f).group(1)
        d = pd.read_parquet(f)
        d["chunk"] = ck
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    pos = pd.concat(dfs, ignore_index=True)
    video_map = {ck: f"{chunks_dir}/{ck}.mp4" for ck in pos["chunk"].unique()}
    return align_teams_by_color(pos, video_map, dark_is_team0=True)


def half_metrics(pos: pd.DataFrame) -> pd.DataFrame:
    """Per-(chunk, frame, team) structural metrics over calibration-filtered frames (per-chunk direction)."""
    parts = []
    for ck, g in pos.groupby("chunk"):
        g = g[(g["calib_error_m"] <= 1.0) & g["pitch_x"].notna()]
        if g.empty:
            continue
        mt = compute_metrics_table(g, attack_dirs=resolve_attack_directions(g))
        if not mt.empty:
            mt["chunk"] = ck
            parts.append(mt)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    Path("outputs/france_iraq/final").mkdir(parents=True, exist_ok=True)
    allm = []
    for tag, mdir, cdir in HALVES:
        if not glob.glob(f"{mdir}/chunk_*_dense.parquet"):
            print(f"[{tag}] no dense parquets yet — skipped")
            continue
        print(f"[{tag}] aligning teams + computing metrics ...")
        pos = load_half(mdir, cdir)
        pos.to_parquet(f"outputs/france_iraq/final/{tag}_aligned.parquet", index=False)
        m = half_metrics(pos)
        m["half"] = tag
        allm.append(m)
        # per-chunk line-height stability check
        for t, name in [(0, "France"), (1, "Iraq")]:
            v = m[m["team"] == t].groupby("chunk")["def_line_height"].mean()
            if len(v):
                print(f"  {name}: def_line per chunk {v.round(0).tolist()}  "
                      f"(mean {v.mean():.1f}, CV {v.std()/v.mean():.0%})")
    if not allm:
        print("no data yet — run batch_match on both halves first")
        return
    allm = pd.concat(allm, ignore_index=True)
    fp = team_fingerprint(allm)
    fp["name"] = fp["team"].map({0: "France", 1: "Iraq"})
    pd.set_option("display.width", 240, "display.max_columns", 40)
    cols = ["name", "frames", "avg_players", "width", "length", "compactness", "surface_area",
            "buildup_height", "def_line_height", "attacking_third_share"]
    print("\n=== France-Iraq structural fingerprint (kit-anchored, complete labels) ===\n")
    print(fp[cols].round(2).to_string(index=False))
    print("\n=== C6: our France vs FIFA published (per-phase line / width) ===")
    fr = fp[fp["team"] == 0].iloc[0]
    print(f"  FIFA France line by phase: {[v[2] for v in FIFA_FRANCE.values()]} m "
          f"(build-up/final-third/mid/low)")
    print(f"  ours France def_line (overall): {fr['def_line_height']:.1f} m  "
          f"buildup_height: {fr['buildup_height']:.1f} m")
    print(f"  FIFA France width range: {min(v[0] for v in FIFA_FRANCE.values())}-"
          f"{max(v[0] for v in FIFA_FRANCE.values())} m   ours: {fr['width']:.1f} m")
    fp.to_parquet("outputs/france_iraq/final/fingerprint.parquet", index=False)
    print("\nwrote outputs/france_iraq/final/{h1,h2}_aligned.parquet + fingerprint.parquet")


if __name__ == "__main__":
    main()
