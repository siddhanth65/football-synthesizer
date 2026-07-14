"""Assemble synthesizer inputs: a team's tendency samples (per-chunk) from match positions.

Tier A needs the *distribution* of a team's tendencies, so we summarise each chunk into one sample of
each tendency metric (calibration-filtered, keeper-resolved direction). Stacking chunks (and, later,
matches) gives the empirical distribution the predictor turns into calibrated forecasts. Pure pandas.
"""
from __future__ import annotations

import pandas as pd

from fingerprint.structural_metrics import compute_metrics_table, resolve_attack_directions

# The tendencies we forecast (FIFA-EFI-aligned, all model-free / position-only).
TENDENCIES = ("def_line_height", "buildup_height", "width", "length", "compactness",
              "attacking_third_share", "wing_share")
MIN_FRAMES = 20  # a chunk needs this many tactical frames to give a stable sample


def _wing_share(metrics: pd.DataFrame) -> pd.Series:
    """Share of players in the two wing lanes (L+R) per row -- a width-of-attack proxy."""
    return metrics.get("lane_left_wing", 0.0) + metrics.get("lane_right_wing", 0.0)


def team_tendency_samples(positions: pd.DataFrame, team: int, *, calib_max: float = 1.0,
                          min_frames: int = MIN_FRAMES) -> pd.DataFrame:
    """One row per chunk of a team's mean tendencies (the samples for a Tier-A distribution).

    Args:
        positions: match positions (needs ``chunk``; colour-anchored teams).
        team: which team (0/1) to summarise.
        calib_max: drop frames with calibration error above this (metres).
        min_frames: skip chunks with fewer tactical frames for that team.

    Returns:
        ``chunk`` + one column per :data:`TENDENCIES` (chunk-mean), one row per qualifying chunk.
    """
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [("_", positions)]
    rows = []
    for ck, g in groups:
        g = g[(g["calib_error_m"] <= calib_max)] if "calib_error_m" in g.columns else g
        g = g.dropna(subset=["pitch_x", "pitch_y"])
        mt = compute_metrics_table(g, attack_dirs=resolve_attack_directions(g))
        s = mt[mt["team"] == team]
        if len(s) < min_frames:
            continue
        s = s.assign(wing_share=_wing_share(s))
        rows.append({"chunk": ck, **{t: float(s[t].mean()) for t in TENDENCIES if t in s.columns}})
    return pd.DataFrame(rows)
