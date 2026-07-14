"""Tests for the geometric xT surface + controlled-threat / ball-xT metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.xt import (
    ball_xt_timeline,
    controlled_threat_metrics,
    geometric_xt,
    xt_grids,
)


def test_geometric_xt_rises_toward_goal_and_centre():
    # closer to the attacking goal (x=105) and more central (y=34) -> higher value
    far = geometric_xt(np.array([10.0]), np.array([34.0]))[0]
    near = geometric_xt(np.array([100.0]), np.array([34.0]))[0]
    wide = geometric_xt(np.array([100.0]), np.array([5.0]))[0]
    assert near > far
    assert near > wide
    assert 0 < far < near <= 1


def test_xt_grids_orient_opposite_for_the_two_teams():
    gx, gy = np.linspace(0, 105, 21), np.linspace(0, 68, 14)
    xt0, xt1 = xt_grids(gx, gy, dir0=1)        # team 0 attacks +x
    # team 0's most valuable column is near x=105; team 1's near x=0
    assert xt0[:, -1].mean() > xt0[:, 0].mean()
    assert xt1[:, 0].mean() > xt1[:, -1].mean()


def _swarm(team, xs, n_frames=4):
    rows = []
    for f in range(n_frames):
        for i, x in enumerate(xs):
            rows.append({"frame": f * 5, "track_id": team * 100 + i, "team": team,
                         "role": "player", "pitch_x": x, "pitch_y": 34.0})
    return rows


def test_controlled_threat_favours_the_team_dominating_the_attacking_third():
    # team 0 (attacks +x via keeper at x=2) parks players in the attacking third; team 1 sits deep.
    rows = [{"frame": f * 5, "track_id": 999, "team": 0, "role": "goalkeeper",
             "pitch_x": 2.0, "pitch_y": 34.0} for f in range(4)]
    rows += _swarm(0, [80, 85, 90, 95, 88])
    rows += _swarm(1, [70, 72, 74, 60, 65])    # team 1 also forward but team 0 is higher/closer to goal
    z = controlled_threat_metrics(pd.DataFrame(rows), sample=10).set_index("team")
    assert z.loc[0, "threat_share"] > z.loc[1, "threat_share"]


def test_ball_xt_timeline_credits_forward_progression():
    # team 0 attacks +x; ball moves from x=40 -> 90 while team 0 holds -> positive xT created.
    ball = pd.DataFrame([{"frame": f, "x": x, "y": 34.0} for f, x in [(0, 40), (5, 60), (10, 90)]])
    poss = pd.DataFrame([{"frame": f, "carrier": 1, "team": 0} for f in (0, 5, 10)])
    bt = ball_xt_timeline(ball, poss, {0: 1, 1: -1}).set_index("team")
    assert bt.loc[0, "xt_created"] > 0
    assert bt.loc[0, "n_moves"] >= 1
