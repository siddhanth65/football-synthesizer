"""Tests for transition / press-trigger metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.transitions import detect_turnovers, transition_metrics


def test_detect_turnovers_on_possession_switch():
    poss = pd.DataFrame({"frame": [0, 5, 10, 15], "team": [0, 0, 1, 1],
                         "carrier": [1, 1, 2, 2], "dist_m": [0.5] * 4})
    ball = pd.DataFrame({"frame": [0, 5, 10, 15], "x": [50, 52, 54, 56], "y": [34, 34, 34, 34]})
    tv = detect_turnovers(poss, ball)
    assert len(tv) == 1
    assert tv.iloc[0]["frame"] == 10
    assert tv.iloc[0]["lost_team"] == 0 and tv.iloc[0]["won_team"] == 1


def test_counterpress_detected_when_losing_team_closes_down():
    # team 0 loses at frame 10 (ball at x=54); a team-0 player is right on the ball next frame.
    poss = pd.DataFrame({"frame": [0, 5, 10, 15], "team": [0, 0, 1, 1],
                         "carrier": [1, 1, 2, 2], "dist_m": [0.5] * 4})
    ball = pd.DataFrame({"frame": [0, 5, 10, 15, 20], "x": [50, 52, 54, 55, 56], "y": [34] * 5})
    players = pd.DataFrame([
        {"frame": 15, "track_id": 1, "team": 0, "pitch_x": 55.5, "pitch_y": 34.0},  # team 0 chases
        {"frame": 15, "track_id": 2, "team": 1, "pitch_x": 55.0, "pitch_y": 34.0},
        {"frame": 20, "track_id": 1, "team": 0, "pitch_x": 56.0, "pitch_y": 34.0},
    ])
    z = transition_metrics(poss, players, ball, {0: 1, 1: -1}, window_frames=50).set_index("team")
    assert z.loc[0, "lost"] == 1
    assert z.loc[0, "counterpress_rate"] == 1.0          # team 0 pressed after losing
    assert z.loc[0, "mean_recovery_frames"] <= 5         # within a few frames


def test_high_regain_counts_attacking_third_wins():
    # team 1 attacks -x (dir -1); winning the ball at x=10 is deep in its attacking third.
    poss = pd.DataFrame({"frame": [0, 5], "team": [0, 1], "carrier": [1, 2], "dist_m": [0.5, 0.5]})
    ball = pd.DataFrame({"frame": [0, 5], "x": [12.0, 10.0], "y": [34.0, 34.0]})
    z = transition_metrics(poss, pd.DataFrame(columns=["frame", "track_id", "team", "pitch_x", "pitch_y"]),
                           ball, {0: 1, 1: -1}, window_frames=50).set_index("team")
    assert z.loc[1, "won"] == 1
    assert z.loc[1, "high_regains"] == 1                  # won deep in its attacking third
