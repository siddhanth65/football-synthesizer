"""Tests for the coarse possession phase-split (A2)."""

from __future__ import annotations

import pandas as pd

from fingerprint.phase_metrics import (
    ball_phase_by_frame,
    phase_split_fingerprint,
    possession_by_frame,
)


def test_possession_carries_the_carrier_forward():
    rows = [
        {"chunk": "A", "frame": 0, "team": 0, "is_actor": True, "role": "player", "pitch_x": 50.0, "pitch_y": 40.0},
        {"chunk": "A", "frame": 5, "team": 0, "is_actor": False, "role": "player", "pitch_x": 50.0, "pitch_y": 40.0},
        {"chunk": "A", "frame": 10, "team": 1, "is_actor": True, "role": "player", "pitch_x": 50.0, "pitch_y": 40.0},
        {"chunk": "A", "frame": 15, "team": 1, "is_actor": False, "role": "player", "pitch_x": 50.0, "pitch_y": 40.0},
    ]
    p = possession_by_frame(pd.DataFrame(rows)).set_index("frame")["poss_team"]
    assert p[0] == 0 and p[5] == 0          # carried from the frame-0 carrier
    assert p[10] == 1 and p[15] == 1        # switches at frame 10


def test_phase_split_labels_both_phases_for_a_team():
    rows = []
    for fr, carrier in [(0, 0), (5, 1)]:    # team 0 has the ball at f0, team 1 at f5
        for t in (0, 1):
            for i in range(5):
                rows.append({"chunk": "A", "frame": fr, "team": t, "role": "player",
                             "pitch_x": 20 + t * 40 + i, "pitch_y": 20 + i * 6,
                             "is_actor": (i == 0 and t == carrier), "is_keeper": False})
    z = phase_split_fingerprint(pd.DataFrame(rows))
    phases = set(z[z["team"] == 0]["phase"])
    assert phases == {"in_poss", "out_poss"}   # team 0 attacks at f0, defends at f5


def _keeper_anchored(team, attack_dir, n_out=6):
    """Outfield players + a keeper so resolve_attack_directions gives ``team`` the wanted direction."""
    gk_x = 2.0 if attack_dir == 1 else 103.0   # keeper sits by the goal the team defends
    return [{"frame": 0, "team": team, "role": "goalkeeper", "is_keeper": True,
             "track_id": 100 + team, "pitch_x": gk_x, "pitch_y": 34.0}]


def test_ball_phase_buckets_by_ball_location_and_block_height():
    # team 0 attacks +x (keeper at x=2), team 1 attacks -x (keeper at x=103).
    players = []
    for t, adir in [(0, 1), (1, -1)]:
        players += _keeper_anchored(t, adir)
        for i in range(6):                       # outfield spread for a defensive block at x~30
            players += [{"frame": 0, "team": t, "role": "player", "is_keeper": False,
                         "track_id": t * 10 + i, "pitch_x": 30.0 + i, "pitch_y": 20.0 + i * 5}]
    pl = pd.DataFrame(players)
    # team 0 holds the ball deep in its own half (attacking-x = 20 -> build_up).
    ball = pd.DataFrame([{"frame": 0, "x": 20.0, "y": 34.0}])
    poss = pd.DataFrame([{"frame": 0, "carrier": 0, "team": 0, "dist_m": 0.5}])
    ph = ball_phase_by_frame(ball, poss, pl).set_index("team")["phase"]
    assert ph[0] == "build_up"                   # ball at own-third for the holder
    assert ph[1] in {"high_press", "mid_block", "low_block"}  # defender gets a block label
