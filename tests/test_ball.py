"""Tests for the deterministic ball trajectory linker + possession (no detector / video)."""

from __future__ import annotations

import pandas as pd

from generator.ball import _debounce_team, assign_possession, link_ball


def _line_detections(step=5, fps=50.0, n=11, vx=2.0):
    rows = []
    for k in range(n):
        fr = k * step
        rows.append({"frame": fr, "x": (fr / fps) * vx, "y": 34.0})
    return pd.DataFrame(rows)


def test_link_ball_rejects_outlier_and_interpolates_gap():
    det = _line_detections(n=11)              # frames 0..50 at 2 m/s
    det = det[det["frame"] != 25]             # drop frame 25 -> a one-sample gap
    det = pd.concat([det, pd.DataFrame([{"frame": 20, "x": 100.0, "y": 10.0}])])  # teleport outlier
    track = link_ball(det, fps=50.0).set_index("frame")
    assert abs(track.loc[20, "x"] - 0.8) < 1e-6        # picked the true point, not the outlier
    assert bool(track.loc[20, "observed"])
    assert 25 in track.index and not bool(track.loc[25, "observed"])  # gap interpolated
    assert abs(track.loc[25, "x"] - 1.0) < 1e-6        # midway between 0.8 (f20) and 1.2 (f30)


def test_link_ball_reseeds_after_long_gap():
    # Segment A (frames 0..20, ~2 m/s) then no candidates for > max_gap samples, then segment B far
    # away (frames 65..75). The stale constant-velocity prediction from A points nowhere near B, so
    # without re-seeding the greedy pass rejects every B frame and the track dies at 20. Re-seeding on
    # the first post-gap candidate recovers B.
    seg_a = _line_detections(n=5)                          # frames 0,5,10,15,20 at 2 m/s
    seg_b = pd.DataFrame([{"frame": f, "x": 80.0, "y": 20.0} for f in (65, 70, 75)])
    track = link_ball(pd.concat([seg_a, seg_b]), fps=50.0)
    frames = set(track["frame"])
    assert {0, 20}.issubset(frames)                        # segment A kept
    assert {65, 70, 75}.issubset(frames)                   # segment B recovered by re-seed
    obs_b = track[track["frame"].isin([65, 70, 75])]["observed"]
    assert bool(obs_b.all())                               # and re-picked as real observations
    # the 45-frame hole (> max_gap*step) is not interpolated across
    assert not (track["frame"].between(21, 64)).any()


def test_link_ball_clamps_super_physical_velocity():
    # A fast, gate-legal acceleration drives the raw velocity estimate above BALL_MAX_SPEED_MS. If the
    # velocity is left unclamped, the constant-velocity prediction overshoots after a short gap and
    # rejects a valid later point; clamping the estimate to the physical ceiling keeps the prediction
    # in reach so the point survives. Gaps here stay under max_gap so re-seeding is not involved.
    det = pd.DataFrame([
        {"frame": 0, "x": 0.0, "y": 34.0},
        {"frame": 5, "x": 3.99, "y": 34.0},               # ~39.9 m/s
        {"frame": 10, "x": 11.9, "y": 34.0},              # raw ~79 m/s (clamped to 40)
        {"frame": 15, "x": 23.7, "y": 34.0},              # raw ~118 m/s (clamped to 40)
        {"frame": 45, "x": 40.0, "y": 34.0},              # reachable at 40 m/s, not at 118 m/s
    ])
    frames = set(link_ball(det, fps=50.0)["frame"])
    assert 45 in frames                                   # clamped prediction keeps the point in reach


def test_assign_possession_nearest_within_radius():
    ball = pd.DataFrame([{"frame": 0, "x": 50.0, "y": 34.0}, {"frame": 5, "x": 10.0, "y": 10.0}])
    players = pd.DataFrame([
        {"frame": 0, "track_id": 1, "team": 0, "pitch_x": 50.4, "pitch_y": 34.0},
        {"frame": 0, "track_id": 2, "team": 1, "pitch_x": 60.0, "pitch_y": 34.0},
        {"frame": 5, "track_id": 1, "team": 0, "pitch_x": 50.0, "pitch_y": 34.0},  # far from ball@(10,10)
    ])
    poss = assign_possession(ball, players, radius_m=2.0, debounce=1)
    assert list(poss["frame"]) == [0]                  # only frame 0 has a player within 2 m
    assert int(poss.loc[0, "carrier"]) == 1 and int(poss.loc[0, "team"]) == 0


def test_debounce_drops_brief_possession_flips():
    poss = pd.DataFrame({"frame": range(7), "team": [0, 0, 0, 1, 0, 0, 0],
                         "carrier": [1] * 7, "dist_m": [0.5] * 7})
    out = _debounce_team(poss, 3)
    assert list(out["team"]) == [0, 0, 0, 0, 0, 0]     # the lone team-1 sample removed


def test_debounce_seconds_converts_via_native_stride():
    # Ball sampled every 5 native frames; a lone team-1 flip at one sample. A 0.3 s hysteresis at 25 fps
    # = round(0.3*25/5) = 2 samples -> drops the 1-sample flip; the same seconds window is fps-consistent.
    ball = pd.DataFrame([{"frame": 5 * i, "x": 50.0, "y": 34.0} for i in range(7)])
    rows = []
    for i in range(7):
        fr = 5 * i
        tx, ty = (50.1, 34.0) if i == 3 else (52.0, 34.0)   # team 1 only sneaks ahead at sample 3
        rows += [{"frame": fr, "track_id": 10, "team": 0, "pitch_x": 50.3, "pitch_y": 34.0},
                 {"frame": fr, "track_id": 20, "team": 1, "pitch_x": tx, "pitch_y": ty}]
    players = pd.DataFrame(rows)
    raw = assign_possession(ball, players, radius_m=2.0, debounce=1)
    assert int(raw.loc[raw["frame"] == 15, "team"].iloc[0]) == 1        # unsmoothed: closer opponent wins
    deb = assign_possession(ball, players, radius_m=2.0, debounce_s=0.3, fps=25.0)
    assert set(deb["team"]) == {0}                                      # 2-sample hysteresis removes it


def test_debounce_seconds_requires_fps():
    ball = pd.DataFrame([{"frame": 0, "x": 50.0, "y": 34.0}])
    players = pd.DataFrame([{"frame": 0, "track_id": 1, "team": 0, "pitch_x": 50.0, "pitch_y": 34.0}])
    try:
        assign_possession(ball, players, debounce_s=0.3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when debounce_s given without fps")


def _two_team_frame(fr, ball_xy, t0_xy, t1_xy):
    """One frame: a team-0 and a team-1 player, used to build viterbi/smoothing inputs."""
    return [
        {"frame": fr, "track_id": 10, "team": 0, "pitch_x": t0_xy[0], "pitch_y": t0_xy[1]},
        {"frame": fr, "track_id": 20, "team": 1, "pitch_x": t1_xy[0], "pitch_y": t1_xy[1]},
    ]


def test_viterbi_smoothing_resists_single_frame_flip():
    # Team 0 clearly holds throughout, but at frame 2 a team-1 player edges marginally closer.
    ball = pd.DataFrame([{"frame": f, "x": 50.0, "y": 34.0} for f in range(5)])
    rows = []
    for f in range(5):
        t0 = (50.3, 34.0)               # team 0 ~0.3 m away every frame
        t1 = (50.1, 34.0) if f == 2 else (51.5, 34.0)  # team 1 only sneaks ahead at f2
        rows += _two_team_frame(f, (50.0, 34.0), t0, t1)
    players = pd.DataFrame(rows)
    smooth = assign_possession(ball, players, radius_m=2.0, smooth=True, switch_penalty_m=1.5)
    assert set(smooth["team"]) == {0}   # the lone f2 flip is penalised away -> all team 0
    raw = assign_possession(ball, players, radius_m=2.0, debounce=1)
    assert int(raw.loc[raw["frame"] == 2, "team"].iloc[0]) == 1   # unsmoothed picks the closer opponent


def test_viterbi_allows_a_genuine_sustained_switch():
    # Team 0 holds frames 0-1, team 1 holds 2-4 (sustained) -> the switch should survive smoothing.
    ball = pd.DataFrame([{"frame": f, "x": 50.0, "y": 34.0} for f in range(5)])
    rows = []
    for f in range(5):
        if f < 2:
            rows += _two_team_frame(f, (50.0, 34.0), (50.2, 34.0), (52.0, 34.0))
        else:
            rows += _two_team_frame(f, (50.0, 34.0), (52.0, 34.0), (50.2, 34.0))
    out = assign_possession(pd.DataFrame(ball), pd.DataFrame(rows), radius_m=2.0, smooth=True)
    seq = out.sort_values("frame")["team"].tolist()
    assert seq == [0, 0, 1, 1, 1]
