"""Tests for the heuristic set-piece / restart detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.set_pieces import (
    _classify,
    ball_speed,
    detect_set_pieces,
    set_piece_summary,
)


def test_classify_by_location():
    assert _classify(1.0, 1.0) == "corner"        # pitch corner
    assert _classify(50.0, 0.5) == "throw_in"     # touchline, mid-pitch
    assert _classify(2.0, 34.0) == "goal_kick"    # six-yard box, central
    assert _classify(60.0, 34.0) == "free_kick"   # open play location


def test_ball_speed_is_metres_per_second():
    ball = pd.DataFrame([{"frame": 0, "x": 0.0, "y": 0.0}, {"frame": 50, "x": 10.0, "y": 0.0}])
    bs = ball_speed(ball, fps=50.0)
    assert np.isnan(bs["speed_ms"].iloc[0])
    assert abs(bs["speed_ms"].iloc[1] - 10.0) < 1e-6   # 10 m in 1 s


def test_detect_restart_after_out_of_play_gap():
    # open play, then the ball goes out (big frame gap), then re-enters settled in the corner.
    rows = [{"frame": 0, "x": 50.0, "y": 34.0}, {"frame": 5, "x": 40.0, "y": 30.0}]
    rows += [{"frame": 200, "x": 1.5, "y": 1.5}, {"frame": 205, "x": 8.0, "y": 6.0}]  # re-entry @ corner
    ev = detect_set_pieces(pd.DataFrame(rows), fps=10.0, min_gap_frames=15)
    assert len(ev) == 1
    assert ev.iloc[0]["type"] == "corner"
    assert ev.iloc[0]["gap_frames"] >= 15


def test_no_restart_without_a_gap():
    rows = [{"frame": 5 * k, "x": 1.5, "y": 1.5} for k in range(6)]   # continuous, no out-of-play gap
    ev = detect_set_pieces(pd.DataFrame(rows), fps=10.0, min_gap_frames=15)
    assert ev.empty


def test_summary_splits_reliable_from_candidate():
    ev = pd.DataFrame({"type": ["corner", "corner", "throw_in", "free_kick"]})
    s = set_piece_summary(ev)
    assert s["corner"] == 2 and s["throw_in"] == 1 and s["total"] == 4
    assert s["reliable"] == 3 and s["candidate"] == 1   # 3 boundary-anchored, 1 free-kick guess
    assert set_piece_summary(pd.DataFrame(columns=["type"])) == {"total": 0, "reliable": 0, "candidate": 0}
