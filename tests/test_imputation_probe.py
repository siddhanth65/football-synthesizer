"""Seam tests for the off-screen imputation probe: span splitting, window margins, baselines.

CPU-only, synthetic tracks with known motion -- the point is to prove the masking/windowing math
and the closed-form baselines are correct on inputs where the right answer is analytic, so a
regression in the seams cannot silently poison the baseline table.
"""
from __future__ import annotations

import numpy as np

from tools import imputation_probe as ip


def _span(t, x, y, chunk="c0", tid=1, team=0):
    t = np.asarray(t, float)
    return ip.Span(chunk, tid, team, t - t[0], (t * 25).astype(int), np.asarray(x, float),
                   np.asarray(y, float))


def test_enumerate_windows_respects_margins_and_duration():
    # 0..10 s at 0.2 s spacing (25 fps / step 5); a 2 s gap must keep >=2 before, >=1 after.
    t = np.arange(0, 10.001, 0.2)
    span = _span(t, t, t)  # positions irrelevant here
    wins = ip.enumerate_windows(span, 2.0)
    assert wins, "expected at least one valid 2 s window"
    for i0, i1 in wins:
        assert i0 >= ip.MIN_BEFORE                      # enough context before
        assert (span.t.size - 1 - i1) >= ip.MIN_AFTER   # enough context after
        assert span.t[i1] - span.t[i0] <= 2.0 + 1e-9    # gap not wider than requested


def test_enumerate_windows_empty_when_span_too_short_for_gap():
    t = np.arange(0, 8.001, 0.2)   # 8 s span cannot host an 8 s gap plus margins
    span = _span(t, t, t)
    assert ip.enumerate_windows(span, 8.0) == []


def test_iter_spans_splits_on_large_hole():
    import pandas as pd

    # two runs of a track separated by a 2 s hole (>> CONTINUITY_TOL_S) -> two spans, each >=8 s.
    fps = 25.0
    f1 = np.arange(0, 9 * fps, ip.SAMPLE_STEP)
    f2 = np.arange(11 * fps, 20 * fps, ip.SAMPLE_STEP)  # 2 s gap between runs
    frame = np.r_[f1, f2].astype(int)
    df = pd.DataFrame({
        "chunk": "c0", "track_id": 7, "team": 0, "role": "player", "is_keeper": False,
        "frame": frame, "pitch_x": frame * 0.1, "pitch_y": 0.0,
    })

    class _M:
        def chunk_fps(self, _c):
            return fps

    spans = [s for s, _kp in ip.iter_spans(df, _M())]
    assert len(spans) == 2
    assert all(s.t[-1] - s.t[0] >= ip.MIN_SPAN_S for s in spans)


def test_linear_interp_exact_on_straight_line():
    # a player moving in a straight line: linear interpolation must be ~0 error.
    t = np.arange(0, 12.001, 0.2)
    x = 1.5 * t + 3.0
    y = -0.7 * t + 10.0
    span = _span(t, x, y)
    i0, i1 = ip.enumerate_windows(span, 2.0)[0]
    err = ip.predict_window(span, False, i0, i1, {})
    assert np.max(err["linear_interp"]) < 1e-9


def test_const_vel_beats_hold_on_moving_player_short_gap():
    # constant-velocity motion, short gap: const_vel error < hold_last error (damping still small).
    t = np.arange(0, 12.001, 0.2)
    x = 2.0 * t
    y = 0.0 * t
    span = _span(t, x, y)
    i0, i1 = ip.enumerate_windows(span, 1.0)[0]
    err = ip.predict_window(span, False, i0, i1, {})
    assert err["const_vel"].mean() < err["hold_last"].mean()


def test_centroid_rel_recovers_pure_team_translation():
    # whole team (target + 2 mates) translates rigidly: centroid-relative hold must be near-exact,
    # since the offset from centroid is constant.
    import pandas as pd

    fps = 25.0
    frame = np.arange(0, 12 * fps, ip.SAMPLE_STEP).astype(int)
    t = frame / fps
    drift = 1.0 * t          # common rigid translation in x
    rows = []
    # target track id 1 at offset (+4, 0) from the moving origin
    tx, ty = drift + 4.0, np.zeros_like(t)
    for f, xx, yy in zip(frame, tx, ty, strict=True):
        rows.append({"chunk": "c0", "track_id": 1, "team": 0, "role": "player",
                     "is_keeper": False, "frame": int(f), "pitch_x": xx, "pitch_y": yy})
    # two mates at fixed offsets, same drift
    for tid, (ox, oy) in [(2, (-2.0, 3.0)), (3, (-2.0, -3.0))]:
        for f, d in zip(frame, drift, strict=True):
            rows.append({"chunk": "c0", "track_id": tid, "team": 0, "role": "player",
                         "is_keeper": False, "frame": int(f), "pitch_x": d + ox, "pitch_y": oy})
    df = pd.DataFrame(rows)
    cents = ip.build_centroids(df)
    span = _span(t, tx, ty)
    span.frame[:] = frame
    i0, i1 = ip.enumerate_windows(span, 2.0)[0]
    err = ip.predict_window(span, False, i0, i1, cents)
    assert "centroid_rel_hold" in err
    assert np.max(err["centroid_rel_hold"]) < 1e-9


def test_cell_summary_rmse_and_reentry():
    c = ip.Cell()
    c.add(np.array([3.0, 4.0]))     # last sample = 4.0 -> re-entry
    c.add(np.array([0.0]))
    s = c.summary()
    assert s["n_win"] == 2
    assert s["n_samp"] == 3
    assert abs(s["rmse"] - np.sqrt((9 + 16 + 0) / 3)) < 1e-9
    assert s["reentry"] == 2.0        # median of [4.0, 0.0]
