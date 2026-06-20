"""Tests for the pure positions post-processing (smoothing + actor/keeper derivation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.postprocess import (
    ACTOR_MAX_DIST_M,
    clamp_to_pitch,
    derive_actor,
    derive_keeper,
    postprocess,
    reject_implausible_frames,
    smooth_tracks,
)


def _frame_rows(frame, xs, ys, team=0):
    return [{"frame": frame, "track_id": i, "role": "player", "team": team,
             "pitch_x": float(x), "pitch_y": float(y)} for i, (x, y) in enumerate(zip(xs, ys))]


def test_reject_keeps_well_spread_frame_and_drops_corner_frame():
    import numpy as np

    rng = np.random.default_rng(0)
    # Good frame 0: 12 players spread across ~40m of pitch.
    good = _frame_rows(0, rng.uniform(10, 55, 12), rng.uniform(10, 60, 12))
    # Bad frame 1: 4 players crammed into a 6m corner (the seg2 failure mode).
    bad = _frame_rows(1, rng.uniform(1, 7, 4), rng.uniform(0, 6, 4))
    out = reject_implausible_frames(pd.DataFrame(good + bad))
    f0 = out[out["frame"] == 0]
    f1 = out[out["frame"] == 1]
    assert f0["pitch_x"].notna().all()  # good frame kept
    assert f1["pitch_x"].isna().all()  # corner frame NaN'd out


def test_clamp_keeps_inside_and_drops_far_off_pitch():
    assert clamp_to_pitch(50.0, 30.0) == (50.0, 30.0)
    # Just outside the line but within the noise tolerance -> clamped to the edge (a real player).
    assert clamp_to_pitch(-1.0, 68.5) == (0.0, 68.0)
    # An official a few metres outside the touchline -> dropped, NOT clamped onto the line as a player.
    cx, cy = clamp_to_pitch(50.0, 71.0)
    assert np.isnan(cx) and np.isnan(cy)
    # Wildly off -> dropped to NaN.
    assert all(np.isnan(v) for v in clamp_to_pitch(300.0, 30.0))
    assert all(np.isnan(v) for v in clamp_to_pitch(float("nan"), 10.0))


def test_smooth_tracks_medians_per_track_and_ignores_ball():
    rows = [{"frame": f, "track_id": 1, "role": "player", "team": 0,
             "pitch_x": x, "pitch_y": 30.0} for f, x in enumerate([10.0, 50.0, 11.0, 12.0, 13.0])]
    rows.append({"frame": 0, "track_id": -1, "role": "ball", "team": -1,
                 "pitch_x": 99.0, "pitch_y": 1.0})
    out = smooth_tracks(pd.DataFrame(rows), window=3)
    # The spike (50.0 at frame 1) is pulled back toward its neighbours by the rolling median.
    spike = out[(out["role"] == "player") & (out["frame"] == 1)]["pitch_x"].iloc[0]
    assert spike < 30.0
    # Ball row untouched.
    assert out[out["role"] == "ball"]["pitch_x"].iloc[0] == 99.0


def test_derive_actor_tags_nearest_player_within_threshold():
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "player", "team": 0, "pitch_x": 50.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 2, "role": "player", "team": 0, "pitch_x": 51.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": -1, "role": "ball", "team": -1, "pitch_x": 50.2, "pitch_y": 34.0},
    ])
    out = derive_actor(df)
    actors = out[out["is_actor"]]
    assert len(actors) == 1 and int(actors["track_id"].iloc[0]) == 1


def test_derive_actor_no_tag_when_ball_far():
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "player", "team": 0, "pitch_x": 10.0, "pitch_y": 10.0},
        {"frame": 0, "track_id": -1, "role": "ball", "team": -1,
         "pitch_x": 10.0 + 2 * ACTOR_MAX_DIST_M, "pitch_y": 10.0},
    ])
    assert derive_actor(df)["is_actor"].sum() == 0


def test_derive_keeper_one_per_team_at_extremes():
    # Team 0 sits left (attacks ->): keeper = deepest (min x). Team 1 sits right: keeper = max x.
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "player", "team": 0, "pitch_x": 5.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 2, "role": "player", "team": 0, "pitch_x": 40.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 3, "role": "player", "team": 1, "pitch_x": 65.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 4, "role": "player", "team": 1, "pitch_x": 100.0, "pitch_y": 34.0},
    ])
    out = derive_keeper(df)
    keepers = set(out[out["is_keeper"]]["track_id"])
    assert keepers == {1, 4}


def test_derive_keeper_not_crowned_at_midfield():
    """The 'keeper at halfway' fix: with no player near a goal, no keeper is invented this frame."""
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "player", "team": 0, "pitch_x": 40.0, "pitch_y": 30.0},
        {"frame": 0, "track_id": 2, "role": "player", "team": 0, "pitch_x": 55.0, "pitch_y": 40.0},
        {"frame": 0, "track_id": 3, "role": "player", "team": 1, "pitch_x": 60.0, "pitch_y": 30.0},
        {"frame": 0, "track_id": 4, "role": "player", "team": 1, "pitch_x": 75.0, "pitch_y": 40.0},
    ])
    assert derive_keeper(df)["is_keeper"].sum() == 0


def test_derive_keeper_prefers_detected_goalkeeper_role_over_position():
    """A detected ``goalkeeper`` is the keeper even if an outfielder is nearer the goal line."""
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "goalkeeper", "team": 0, "pitch_x": 5.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 2, "role": "player", "team": 0, "pitch_x": 2.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 3, "role": "player", "team": 1, "pitch_x": 98.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 4, "role": "goalkeeper", "team": 1, "pitch_x": 101.0, "pitch_y": 34.0},
    ])
    keepers = set(derive_keeper(df)[lambda d: d["is_keeper"]]["track_id"])
    assert keepers == {1, 4}  # the GK-role tracks, not the nearer-goal outfielder (track 2)


def test_referee_excluded_from_keeper_and_actor():
    """Officials (role=referee, team=-1) are never crowned keeper/actor, even when nearest goal/ball."""
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "player", "team": 0, "pitch_x": 5.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 9, "role": "referee", "team": -1, "pitch_x": 2.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": 3, "role": "player", "team": 1, "pitch_x": 100.0, "pitch_y": 34.0},
        {"frame": 0, "track_id": -1, "role": "ball", "team": -1, "pitch_x": 2.1, "pitch_y": 34.0},
    ])
    k = derive_keeper(df)
    a = derive_actor(df)
    assert not bool(k[k["track_id"] == 9]["is_keeper"].iloc[0])  # ref not keeper despite nearest goal
    assert not bool(a[a["track_id"] == 9]["is_actor"].iloc[0])  # ref not actor despite nearest ball
    assert int(a[a["is_actor"]]["track_id"].iloc[0]) == 1  # the real player is the actor


def test_postprocess_chain_adds_both_flags_and_sorts():
    import numpy as np

    # A plausible frame (>=8 players, well spread) so it survives the plausibility gate; team 0 left,
    # team 1 right; the ball sits next to one team-0 player (the actor).
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        rows.append({"frame": 0, "track_id": i, "role": "player", "team": 0,
                     "pitch_x": float(rng.uniform(5, 45)), "pitch_y": float(rng.uniform(10, 58))})
    for i in range(6):
        rows.append({"frame": 0, "track_id": 10 + i, "role": "player", "team": 1,
                     "pitch_x": float(rng.uniform(60, 100)), "pitch_y": float(rng.uniform(10, 58))})
    actor_xy = (rows[0]["pitch_x"], rows[0]["pitch_y"])
    rows.append({"frame": 0, "track_id": -1, "role": "ball", "team": -1,
                 "pitch_x": actor_xy[0] + 0.3, "pitch_y": actor_xy[1]})
    out = postprocess(pd.DataFrame(rows))
    assert {"is_actor", "is_keeper"}.issubset(out.columns)
    assert out["pitch_x"].notna().any()  # plausible frame survived the gate
    assert out["is_actor"].sum() == 1
    assert out["is_keeper"].sum() == 2  # one per team
