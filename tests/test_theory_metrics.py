"""Tests for the P1 theory metrics — synthetic scenarios with a known tactical answer."""

from __future__ import annotations

import pandas as pd

from fingerprint.theory_metrics import (
    complete_directions,
    counterpress_curve,
    lane_occupation,
    line_breaks,
    local_overload,
    pressing_intensity,
    verticality,
)


def _players(rows):
    return pd.DataFrame(rows, columns=["frame", "track_id", "role", "team", "pitch_x", "pitch_y",
                                       "is_keeper"])


def _ball(frames_xy):
    return pd.DataFrame([{"frame": f, "x": x, "y": y} for f, x, y in frames_xy])


def _poss(frames_team):
    return pd.DataFrame([{"frame": f, "carrier": 1, "team": t, "dist_m": 0.5} for f, t in frames_team])


def test_complete_directions_fills_opposite():
    assert complete_directions({0: 1}) == {0: 1, 1: -1}
    assert complete_directions({1: -1}) == {1: -1, 0: 1}
    assert complete_directions({0: 1, 1: -1}) == {0: 1, 1: -1}   # already complete, unchanged


def test_local_overload_counts_numerical_superiority():
    # team 0 has the ball at (50,34) with 3 team-mates near it and only 1 defender nearby
    rows = []
    for f in range(5):
        rows += [(f, 1, "player", 0, 50.0, 34.0, False),
                 (f, 2, "player", 0, 52.0, 35.0, False),
                 (f, 3, "player", 0, 48.0, 33.0, False),
                 (f, 9, "player", 1, 51.0, 34.0, False),
                 (f, 8, "player", 1, 90.0, 34.0, False)]  # far defender, out of radius
    ov = local_overload(_ball([(f, 50.0, 34.0) for f in range(5)]),
                         _poss([(f, 0) for f in range(5)]), _players(rows))
    r = ov[ov["team"] == 0].iloc[0]
    assert r["mean_overload"] == 2.0        # 3 attackers - 1 defender within 12 m
    assert r["overload_share"] == 1.0


def test_pressing_intensity_high_when_defender_on_the_carrier():
    # carrier (team 0) at (50,34); a team-1 defender sits right on top of it across frames
    rows = []
    for f in range(6):
        rows += [(f, 1, "player", 0, 50.0, 34.0, False),
                 (f, 9, "player", 1, 51.0, 34.0, False)]
    close = pressing_intensity(_ball([(f, 50.0, 34.0) for f in range(6)]),
                               _poss([(f, 0) for f in range(6)]), _players(rows))
    # a far defender instead
    rows2 = []
    for f in range(6):
        rows2 += [(f, 1, "player", 0, 50.0, 34.0, False),
                  (f, 9, "player", 1, 95.0, 5.0, False)]
    far = pressing_intensity(_ball([(f, 50.0, 34.0) for f in range(6)]),
                             _poss([(f, 0) for f in range(6)]), _players(rows2))
    assert close[close["team"] == 1].iloc[0]["pressing_intensity"] > 0.7
    assert far[far["team"] == 1].iloc[0]["pressing_intensity"] < 0.2


def test_line_break_fires_on_front_to_behind_crossing():
    # opponent (team 1) defensive line around ax=60 (they attack -1, so their deep line is high x);
    # team 0 attacks +1. Ball goes from x=40 (front) to x=80 (clearly behind the line).
    dirs = {0: 1, 1: -1}
    line_players = [(f, 20 + i, "player", 1, 60.0, 20.0 + 8 * i, False) for f in range(40)
                    for i in range(5)]
    ball = _ball([(f, 40.0 if f < 20 else 80.0, 34.0) for f in range(40)])
    poss = _poss([(f, 0) for f in range(40)])
    lb = line_breaks(ball, poss, _players(line_players), dirs)
    assert lb[lb["team"] == 0].iloc[0]["line_breaks"] == 1


def test_verticality_direct_vs_sideways():
    dirs = {0: 1, 1: -1}
    straight = verticality(_ball([(f, 20.0 + f * 3, 34.0) for f in range(15)]),
                           _poss([(f, 0) for f in range(15)]), dirs)
    sideways = verticality(_ball([(f, 40.0, 5.0 + f * 3) for f in range(15)]),
                           _poss([(f, 0) for f in range(15)]), dirs)
    assert straight[straight["team"] == 0].iloc[0]["verticality"] > 0.9    # pure goalward
    assert sideways[sideways["team"] == 0].iloc[0]["verticality"] < 0.1    # pure lateral


def test_lane_occupation_detects_halfspace_side():
    # team 0 attacks +1; put all its players in the two half-space lanes (y bands ~13-27 and 41-55)
    rows = []
    for f in range(3):
        rows += [(f, 1, "player", 0, 60.0, 20.0, False),
                 (f, 2, "player", 0, 70.0, 48.0, False),
                 (f, 3, "player", 0, 55.0, 21.0, False)]
    lo = lane_occupation(_players(rows), {0: 1, 1: -1})
    r = lo[lo["team"] == 0].iloc[0]
    assert r["halfspace_share"] > 0.9
    assert r["wing_share"] == 0.0


def test_counterpress_curve_rewards_fast_regain():
    # team 0 loses at frame 10, wins back at frame 60 (=1 s at 50 fps): regained by 3 s, not by... it is
    poss = _poss([(0, 0), (10, 1), (60, 0)])
    ball = _ball([(0, 50.0, 34), (10, 50.0, 34), (60, 50.0, 34)])
    players = _players([(f, 1, "player", 0, 50.0, 34.0, False) for f in (0, 10, 60)])
    cp = counterpress_curve(ball, poss, players, fps=50.0)
    r = cp[cp["team"] == 0].iloc[0]
    assert r["losses"] == 1
    assert r["regain_3s"] == 1.0   # regained 1 s after the loss -> within every window >=1 s


def test_metrics_return_empty_on_empty_input():
    empty_ball = pd.DataFrame(columns=["frame", "x", "y"])
    empty_poss = pd.DataFrame(columns=["frame", "carrier", "team", "dist_m"])
    empty_pl = _players([])
    assert pressing_intensity(empty_ball, empty_poss, empty_pl).empty
    assert line_breaks(empty_ball, empty_poss, empty_pl, {0: 1, 1: -1}).empty
    assert local_overload(empty_ball, empty_poss, empty_pl).empty
