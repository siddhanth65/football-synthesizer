"""Tests for the deterministic team-style fingerprint vector (z_T)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.team_style import (
    align_teams_by_defended_goal,
    physical_profile,
    team_style_vector,
)


def test_physical_profile_zone_fractions_and_top_speed():
    # team 0: three samples at 1 m/s (3.6 km/h -> zone 1) and one at 6 m/s (21.6 km/h -> zone 4)
    tracks = pd.DataFrame({"team": [0, 0, 0, 0], "speed": [1.0, 1.0, 1.0, 6.0]})
    p = physical_profile(tracks).set_index("team").loc[0]
    assert abs(p["time_walk_z1"] - 0.75) < 1e-9 and abs(p["time_hispeed_z4"] - 0.25) < 1e-9
    assert abs(p["sprint_share"] - 0.25) < 1e-9      # one sample >= 20 km/h
    assert 20.0 < p["top_speed_kmh"] <= 21.6         # ~6 m/s (99th pct interpolates just under the max)


def test_align_teams_by_defended_goal_is_consistent_across_chunks():
    rows = []
    # chunk A: team 0 keeper at left (x=5), team 1 keeper at right (x=100)
    # chunk B: labels SWAPPED -> team 0 keeper at right, team 1 keeper at left
    for ck, left_team, right_team in [("A", 0, 1), ("B", 1, 0)]:
        rows.append({"chunk": ck, "frame": 0, "track_id": 1, "role": "goalkeeper",
                     "team": left_team, "pitch_x": 5.0, "pitch_y": 34.0, "is_keeper": True})
        rows.append({"chunk": ck, "frame": 0, "track_id": 2, "role": "goalkeeper",
                     "team": right_team, "pitch_x": 100.0, "pitch_y": 34.0, "is_keeper": True})
    aligned = align_teams_by_defended_goal(pd.DataFrame(rows))
    # every left-goal keeper (x=5) must now carry the same aligned team id (0)
    left = aligned[aligned["pitch_x"] == 5.0]["team"].unique()
    right = aligned[aligned["pitch_x"] == 100.0]["team"].unique()
    assert list(left) == [0] and list(right) == [1]


def test_team_style_vector_has_shape_physical_and_indices():
    rows = []
    for fr in range(4):
        for t in (0, 1):
            for i in range(6):  # 6 outfield players per team, moving a little each frame
                rows.append({"frame": fr * 5, "track_id": t * 100 + i, "role": "player", "team": t,
                             "pitch_x": 20 + t * 40 + i + fr, "pitch_y": 10 + i * 9,
                             "is_keeper": False, "is_actor": False})
            rows.append({"frame": fr * 5, "track_id": t * 100 + 9, "role": "goalkeeper", "team": t,
                         "pitch_x": 4.0 if t == 0 else 101.0, "pitch_y": 34.0,
                         "is_keeper": True, "is_actor": False})
    z = team_style_vector(pd.DataFrame(rows), fps=50.0)
    assert set(z["team"]) == {0, 1} and len(z) == 2
    for col in ("width", "length", "buildup_height", "top_speed_kmh", "wing_share", "lr_bias"):
        assert col in z.columns
