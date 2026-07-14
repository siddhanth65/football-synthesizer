"""Tests for the dense-positions -> per-frame tactical-frame table (pure)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.tactical_frames import (
    TACTICAL_COLUMNS,
    build_tactical_frame,
    build_tactical_table,
)


def _frame(fr, n_per_team=6, with_ball=True, with_ref=False):
    rows = []
    for i in range(n_per_team):
        rows.append({"frame": fr, "track_id": i, "role": "player", "team": 0,
                     "pitch_x": 10 + i * 3, "pitch_y": 20 + i * 2, "is_actor": i == 0,
                     "calib_error_m": 0.3})
    for i in range(n_per_team):
        rows.append({"frame": fr, "track_id": 100 + i, "role": "player", "team": 1,
                     "pitch_x": 60 + i * 3, "pitch_y": 30 + i * 2, "is_actor": False,
                     "calib_error_m": 0.3})
    if with_ref:
        rows.append({"frame": fr, "track_id": 200, "role": "referee", "team": -1,
                     "pitch_x": 52.0, "pitch_y": 34.0, "is_actor": False, "calib_error_m": 0.3})
    if with_ball:
        rows.append({"frame": fr, "track_id": -1, "role": "ball", "team": -1,
                     "pitch_x": 11.0, "pitch_y": 20.0, "is_actor": False, "calib_error_m": 0.3})
    return pd.DataFrame(rows)


def test_accepted_frame_summary():
    rec = build_tactical_frame(_frame(0), fps=50.0)
    assert rec["accepted"] is True
    assert rec["n_onpitch"] == 12 and rec["n_team0"] == 6 and rec["n_team1"] == 6
    assert rec["has_ball"] is True and abs(rec["ball_x"] - 11.0) < 1e-9
    assert rec["actor_team"] == 0 and rec["actor_track_id"] == 0
    assert abs(rec["team0_xspan"] - 15.0) < 1e-9  # x = 10..25
    assert abs(rec["time_s"] - 0.0) < 1e-9


def test_referee_excluded_from_counts():
    rec = build_tactical_frame(_frame(0, with_ref=True))
    assert rec["n_onpitch"] == 12  # the referee row is not counted as a player


def test_sparse_frame_is_rejected_without_shape():
    rec = build_tactical_frame(_frame(0, n_per_team=4))  # 8 players < MIN_PLAYERS
    assert rec["accepted"] is False and rec["n_onpitch"] == 8
    assert np.isnan(rec["team0_cx"])  # shapes left NaN for a rejected frame


def test_table_is_one_sorted_row_per_frame():
    df = pd.concat([_frame(5), _frame(1), _frame(3)], ignore_index=True)
    tab = build_tactical_table(df)
    assert list(tab["frame"]) == [1, 3, 5]
    assert tuple(tab.columns) == TACTICAL_COLUMNS
    assert bool(tab["accepted"].all())
