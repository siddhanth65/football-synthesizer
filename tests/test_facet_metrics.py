"""Tests for the new goalkeeping + passing-connectivity facet metrics."""

from __future__ import annotations

import pandas as pd

from fingerprint.facet_metrics import goalkeeper_metrics, passing_connectivity


def test_goalkeeper_sweeper_height_is_distance_from_own_goal():
    rows = [
        {"frame": 0, "role": "goalkeeper", "team": 0, "pitch_x": 10.0, "pitch_y": 34.0, "is_keeper": True},
        {"frame": 1, "role": "goalkeeper", "team": 0, "pitch_x": 12.0, "pitch_y": 30.0, "is_keeper": True},
        {"frame": 0, "role": "goalkeeper", "team": 1, "pitch_x": 95.0, "pitch_y": 34.0, "is_keeper": True},
    ]
    gk = goalkeeper_metrics(pd.DataFrame(rows), {0: 1, 1: -1}).set_index("team")
    assert abs(gk.loc[0, "gk_sweeper_height"] - 11.0) < 1e-9   # defends x=0, median of [10,12]
    assert abs(gk.loc[1, "gk_sweeper_height"] - 10.0) < 1e-9   # defends x=105 -> 105-95


def test_passing_connectivity_nearest_mate_distance():
    rows = [{"frame": 0, "team": 0, "role": "player", "pitch_x": x, "pitch_y": y}
            for x, y in [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]]  # 10 m square
    p = passing_connectivity(pd.DataFrame(rows), min_team=4).set_index("team")
    assert abs(p.loc[0, "pass_nearest_mate_m"] - 10.0) < 1e-9
