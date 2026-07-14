"""Tests for velocity synchrony + style distance (no ball)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.style_metrics import style_distance, velocity_synchrony


def test_velocity_synchrony_aligned_vs_scattered():
    aligned = pd.DataFrame({"frame": [0] * 4, "team": [0] * 4, "vx": [2.0, 2.0, 2.0, 2.0],
                            "vy": [0.0, 0.0, 0.0, 0.0]})
    assert velocity_synchrony(aligned).set_index("team").loc[0, "velocity_synchrony"] > 0.99
    scattered = pd.DataFrame({"frame": [0] * 4, "team": [0] * 4, "vx": [2.0, -2.0, 0.0, 0.0],
                              "vy": [0.0, 0.0, 2.0, -2.0]})
    assert velocity_synchrony(scattered).set_index("team").loc[0, "velocity_synchrony"] < 0.1


def _team_rows(team, gk_x, xs):
    rows = [{"frame": 0, "team": team, "role": "goalkeeper", "pitch_x": gk_x, "pitch_y": 34.0,
             "is_keeper": True}]
    return rows + [{"frame": 0, "team": team, "role": "player", "pitch_x": x, "pitch_y": 34.0,
                    "is_keeper": False} for x in xs]


def test_style_distance_runs_and_nonnegative():
    df = pd.DataFrame(_team_rows(0, 5.0, [30, 40, 50]) + _team_rows(1, 100.0, [55, 65, 75]))
    d = style_distance(df)
    assert np.isfinite(d) and d >= 0
