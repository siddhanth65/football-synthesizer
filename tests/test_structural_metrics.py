"""Tests for the deterministic team-shape metric engine (pure)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.structural_metrics import (
    attacking_coord,
    compute_metrics_table,
    direction_metrics,
    lane_fractions,
    resolve_attack_directions,
    team_fingerprint,
    team_shape,
)


def test_lane_fractions_sum_to_one_and_bucket_evenly():
    py = np.array([6.8, 20.4, 34.0, 47.6, 61.2])  # one player per width-lane
    f = lane_fractions(py)
    assert abs(f.sum() - 1.0) < 1e-9
    assert np.allclose(f, 0.2)


def test_team_shape_rectangle():
    px = np.array([10.0, 30.0, 10.0, 30.0])  # 20 m (x) x 40 m (y) rectangle
    py = np.array([14.0, 14.0, 54.0, 54.0])
    s = team_shape(px, py)
    assert abs(s["width"] - 40.0) < 1e-9 and abs(s["length"] - 20.0) < 1e-9
    assert abs(s["surface_area"] - 800.0) < 1e-6  # 20 * 40
    assert abs(s["centroid_x"] - 20.0) < 1e-9 and abs(s["centroid_y"] - 34.0) < 1e-9


def test_metrics_table_excludes_referees_and_sparse_teams():
    rows = []
    for i in range(5):  # team 0: enough players
        rows.append({"frame": 0, "track_id": i, "role": "player", "team": 0,
                     "pitch_x": 10 + i, "pitch_y": 20 + i * 5})
    for i in range(3):  # team 1: too few -> dropped
        rows.append({"frame": 0, "track_id": 10 + i, "role": "player", "team": 1,
                     "pitch_x": 60 + i, "pitch_y": 30 + i})
    rows.append({"frame": 0, "track_id": 99, "role": "referee", "team": -1,
                 "pitch_x": 52, "pitch_y": 34})
    t = compute_metrics_table(pd.DataFrame(rows), min_team=4)
    assert list(t["team"]) == [0]  # team 1 sparse, referee excluded


def test_team_fingerprint_one_row_per_team():
    rows = []
    for fr in range(3):
        for t in (0, 1):
            for i in range(6):
                rows.append({"frame": fr, "track_id": t * 100 + i, "role": "player", "team": t,
                             "pitch_x": 10 + t * 40 + i, "pitch_y": 10 + i * 8})
    fp = team_fingerprint(compute_metrics_table(pd.DataFrame(rows)))
    assert set(fp["team"]) == {0, 1} and bool((fp["frames"] == 3).all())
    assert "width" in fp.columns and "lane_centre" in fp.columns


def test_attacking_coord_flips_for_negative_direction():
    px = np.array([0.0, 52.5, 105.0])
    assert np.allclose(attacking_coord(px, 1), px)  # already toward x=105
    assert np.allclose(attacking_coord(px, -1), [105.0, 52.5, 0.0])  # mirrored


def test_resolve_attack_directions_from_keeper_goal():
    rows = []
    for fr in range(3):  # team 0 keeper at left goal -> attacks +1; team 1 at right goal -> -1
        rows.append({"frame": fr, "track_id": 0, "role": "goalkeeper", "team": 0,
                     "pitch_x": 5.0, "pitch_y": 34.0, "is_keeper": True})
        rows.append({"frame": fr, "track_id": 1, "role": "goalkeeper", "team": 1,
                     "pitch_x": 100.0, "pitch_y": 34.0, "is_keeper": True})
    dirs = resolve_attack_directions(pd.DataFrame(rows))
    assert dirs == {0: 1, 1: -1}


def test_resolve_directions_forces_opposite_when_keepers_same_side():
    # Follow-play broadcast: both teams' keepers land at the right goal, but team 1 is better-sampled
    # (more keeper rows). Team 1 must anchor (defends right -> attacks -1), team 0 forced opposite.
    rows = []
    for fr in range(10):
        rows.append({"frame": fr, "track_id": 1, "role": "goalkeeper", "team": 1,
                     "pitch_x": 99.0, "pitch_y": 34.0, "is_keeper": True})
    for fr in range(2):  # only 2 misdetected "keeper" rows for team 0, same (right) side
        rows.append({"frame": fr, "track_id": 0, "role": "goalkeeper", "team": 0,
                     "pitch_x": 95.0, "pitch_y": 34.0, "is_keeper": True})
    dirs = resolve_attack_directions(pd.DataFrame(rows))
    assert dirs[1] == -1 and dirs[0] == 1     # opposite, anchored on the better-sampled keeper (team 1)
    assert len(set(dirs.values())) == 2


def test_resolve_directions_from_ball_uses_possession_location():
    from fingerprint.structural_metrics import resolve_attack_directions_from_ball  # noqa: PLC0415

    # team 0's possessions sit toward x=105 (attacks +1); team 1's toward x=0 (attacks -1).
    players, ball = [], []
    for fr in range(20):
        t0x, t1x = (75.0, 30.0)
        players.append({"frame": fr, "track_id": 0, "team": 0, "role": "player",
                        "pitch_x": t0x, "pitch_y": 34.0})
        players.append({"frame": fr, "track_id": 1, "team": 1, "role": "player",
                        "pitch_x": t1x, "pitch_y": 34.0})
        # ball alternates between the two carriers
        ball.append({"frame": fr, "x": t0x if fr % 2 == 0 else t1x, "y": 34.2})
    dirs = resolve_attack_directions_from_ball(pd.DataFrame(players), pd.DataFrame(ball))
    assert dirs == {0: 1, 1: -1}              # team 0 (further right) attacks +1


def test_direction_metrics_are_team_comparable_and_exclude_keeper():
    px = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    keep = np.zeros(5, bool)
    m_pos = direction_metrics(px, keep, 1)             # attacking toward x=105
    m_neg = direction_metrics(105.0 - px, keep, -1)    # mirror image, attacking toward x=0
    assert abs(m_pos["buildup_height"] - m_neg["buildup_height"]) < 1e-9
    assert abs(m_pos["buildup_height"] - 40.0) < 1e-9  # mean attacking-x
    # keeper (deepest) is excluded from the line height
    with_keeper = direction_metrics(np.array([2.0, 40.0, 60.0]), np.array([True, False, False]), 1)
    assert with_keeper["def_line_height"] >= 40.0
    assert np.isnan(direction_metrics(px, keep, None)["buildup_height"])  # unknown dir -> NaN


def test_metrics_table_carries_direction_columns():
    rows = [{"frame": 0, "track_id": i, "role": "player", "team": 0,
             "pitch_x": 40 + i * 4, "pitch_y": 20 + i * 5, "is_keeper": False} for i in range(5)]
    rows.append({"frame": 0, "track_id": 9, "role": "goalkeeper", "team": 0,
                 "pitch_x": 4.0, "pitch_y": 34.0, "is_keeper": True})
    t = compute_metrics_table(pd.DataFrame(rows), min_team=4)
    assert {"attack_dir", "buildup_height", "def_line_height"} <= set(t.columns)
    assert int(t.loc[0, "attack_dir"]) == 1  # keeper at x=4 -> defends left -> attacks +1
