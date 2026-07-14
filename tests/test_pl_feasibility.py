"""Tests for the Phase A feasibility probe's pure metric seams (no GPU / video / model)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.pl_feasibility import (
    calibration_yield,
    players_per_frame,
    post_link_coverage,
    track_stats,
)


def _dense(frames, players_per, *, pitch=True, calib_err=1.0):
    """Build a synthetic dense table: ``players_per`` tracks on each frame in ``frames``."""
    rows = []
    for f in frames:
        for t in range(players_per):
            rows.append({
                "frame": f, "track_id": t, "role": "player",
                "pitch_x": float(t) if pitch else np.nan,
                "pitch_y": 1.0 if pitch else np.nan,
                "image_x": t * 10.0, "image_y": 5.0, "calib_error_m": calib_err,
            })
    return pd.DataFrame(rows)


def test_players_per_frame_averages_over_present_frames():
    df = _dense([0, 5], 6)
    assert players_per_frame(df) == 6.0
    assert players_per_frame(df.iloc[:0]) == 0.0


def test_track_stats_counts_tracks_and_fragmentation():
    df = _dense([0, 5, 10], 6)  # 6 tracks, each spanning 3 frames
    s = track_stats(df, sample_every=5, fps=25.0)
    assert s["n_tracks"] == 6
    assert s["mean_track_frames"] == 3.0
    # frag index = n_tracks / mean_ppf = 6 / 6 = 1.0 (perfectly stable)
    assert abs(s["frag_index"] - 1.0) < 1e-9
    assert abs(s["mean_track_s"] - 3 * 5 / 25.0) < 1e-9


def test_calibration_yield_gate_and_ge6():
    ok = _dense([0, 5, 10], 6, calib_err=1.0)          # 3 frames, gate-pass, >=6 corr
    bad = _dense([15], 3, pitch=False, calib_err=5.0)  # 1 frame, gate-fail, <6 corr
    df = pd.concat([ok, bad], ignore_index=True)
    y = calibration_yield(df)
    assert y["n_frames"] == 4
    assert abs(y["pct_calibrated"] - 0.75) < 1e-9   # 3/4 frames <= 2 m
    assert abs(y["pct_ge6_corr"] - 0.75) < 1e-9     # 3/4 frames have >= 6 pitched players


def test_calibration_yield_below_min_corr_is_not_counted():
    df = _dense([0, 5], 5, calib_err=1.0)  # only 5 pitched players/frame < MIN_CORR (6)
    y = calibration_yield(df)
    assert y["pct_calibrated"] == 1.0   # both frames pass the metre gate
    assert y["pct_ge6_corr"] == 0.0     # but neither reaches 6 correspondences


def test_post_link_coverage_ratio():
    assert post_link_coverage(200, 400) == 0.5
    assert post_link_coverage(0, 400) == 0.0
    assert post_link_coverage(50, 0) == 0.0
