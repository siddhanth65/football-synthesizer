"""B-3 stage 2 unit tests: frame mapping, nearest-carrier rule, contact window, abstention.

The 25 fps BAS clock <-> stride-5 parquet grid mapping is the load-bearing seam (get it wrong and
every attribution lands on the wrong player); the nearest-carrier + on-the-ball radius drive team
attribution and abstention. All pure-function tests, no data files needed.
"""
from __future__ import annotations

import numpy as np

from tools.event_ledger import (
    STEP,
    TEAM_MAX_M,
    _contact_frame,
    _nearest_row,
    bas_to_grid_frame,
    nearest_carrier,
)


def test_bas_to_grid_frame_maps_25fps_to_stride5_grid() -> None:
    """A 25 fps BAS frame index snaps to the nearest stride-5 grid frame in the SAME chunk."""
    assert bas_to_grid_frame(240) == 240        # already on the grid
    assert bas_to_grid_frame(236) == 235        # rounds down to nearest 5
    assert bas_to_grid_frame(234) == 235        # rounds up to nearest 5
    assert bas_to_grid_frame(82) == 80
    assert bas_to_grid_frame(0) == 0
    # Property: result is always a multiple of STEP and within STEP/2 of the input.
    for fi in range(0, 15000, 37):
        g = bas_to_grid_frame(fi)
        assert g % STEP == 0
        assert abs(g - fi) <= STEP / 2 + 1e-9


def test_nearest_row_respects_gap() -> None:
    """Nearest ascending frame within the gap, else None (the ball/player time lookup)."""
    frames = np.array([100, 105, 110, 200], dtype=int)
    assert frames[_nearest_row(frames, 106, 25)] == 105
    assert frames[_nearest_row(frames, 108, 25)] == 110
    assert _nearest_row(frames, 150, 25) is None        # 150 is >25 from any frame
    assert frames[_nearest_row(frames, 150, 60)] == 110  # widen the gap -> 110 wins
    assert _nearest_row(np.empty(0, dtype=int), 100, 25) is None


def test_nearest_carrier_picks_closest_player_and_team() -> None:
    """The ball-carrier is the nearest player; its team + distance come back."""
    pos = np.array([[50.0, 30.0], [60.0, 30.0]])
    teams = np.array([0, 1])
    tracks = np.array([11, 22])
    team, track, dist = nearest_carrier(pos, teams, tracks, (51.0, 30.0))
    assert team == 0 and track == 11
    assert abs(dist - 1.0) < 1e-6
    team, track, _ = nearest_carrier(pos, teams, tracks, (59.0, 30.0))
    assert team == 1 and track == 22
    assert nearest_carrier(np.empty((0, 2)), np.empty(0), np.empty(0), (0.0, 0.0)) is None


def _toy_chunk() -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Two players (teams 0/1) present at frames 95 and 100; used by the window tests."""
    by = {
        95: (np.array([[51.0, 30.0], [70.0, 30.0]]), np.array([0, 1]), np.array([11, 22])),
        100: (np.array([[58.0, 30.0], [70.0, 30.0]]), np.array([0, 1]), np.array([11, 22])),
    }
    pf = np.array([95, 100], dtype=int)
    bf = np.array([95, 100], dtype=int)
    bxy = np.array([[51.0, 30.0], [55.0, 30.0]])   # ball on player 11 at f95, mid-flight at f100
    return pf, by, bf, bxy


def test_contact_frame_takes_window_minimum() -> None:
    """The contact is the window frame where a player is closest to the ball (the kick moment)."""
    pf, by, bf, bxy = _toy_chunk()
    # Event grid frame 100; the window reaches back to f95 where player 11 is ON the ball.
    ct = _contact_frame(pf, by, bf, bxy, 100)
    assert ct is not None
    assert ct["frame"] == 95          # -5 offset beats the at-peak (mid-flight) frame
    assert ct["team"] == 0
    assert ct["dist"] < 0.01


def test_contact_frame_abstains_without_ball() -> None:
    """No ball anywhere in the window -> None -> attribution abstains."""
    pf, by, _bf, _bxy = _toy_chunk()
    assert _contact_frame(pf, by, np.empty(0, dtype=int), np.empty((0, 2)), 100) is None


def test_loose_ball_exceeds_carrier_radius() -> None:
    """A ball far from every tracked player yields a distance past the on-the-ball radius."""
    by = {100: (np.array([[10.0, 10.0], [20.0, 20.0]]), np.array([0, 1]), np.array([11, 22]))}
    pf = np.array([100], dtype=int)
    bf = np.array([100], dtype=int)
    bxy = np.array([[90.0, 60.0]])    # nowhere near either player
    ct = _contact_frame(pf, by, bf, bxy, 100)
    assert ct is not None
    assert ct["dist"] > TEAM_MAX_M    # -> attribute_chunk leaves team NA (abstain)
