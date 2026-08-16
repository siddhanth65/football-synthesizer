"""Targeted checks for the temporal camera model (v10-W4): decomposition and RTS smoothing."""

from __future__ import annotations

import numpy as np

from generator.camera_track import (
    Pose,
    compose,
    decompose,
    grid_compose,
    grid_series,
    grid_shift_m,
    rts_smooth,
    smooth_homographies,
)

TRUTH = Pose(pan=0.35, tilt=1.85, roll=0.02, log_f=float(np.log(3800.0)),
             log_aspect=float(np.log(0.95)), cx=2.0, cy=-55.0, cz=15.0)


def test_decompose_compose_round_trip() -> None:
    """A ground homography survives the camera parameterisation exactly, aspect ratio included."""
    h = compose(TRUTH)
    assert h is not None
    got = decompose(h)
    assert got is not None
    assert np.allclose(TRUTH.to_array(), got.to_array(), atol=1e-6)
    assert grid_shift_m(h, compose(got)) < 1e-9
    fx, fy = got.focals()
    assert abs(fy / fx - 0.95) < 1e-9, (fx, fy)


def test_mapping_parameterisation_is_lossless() -> None:
    """The grid encoding re-fits the same homography it was read from."""
    h = compose(TRUTH)
    z, valid = grid_series({7: h}, 10)
    assert valid[7] and not valid[0]
    assert grid_shift_m(h, grid_compose(z[7])) < 1e-6


def test_rts_smooth_beats_the_measurements() -> None:
    """A noisy ramp is smoothed toward its truth and predicted where it is not measured."""
    rng = np.random.default_rng(0)
    truth = 0.5 + 0.01 * np.arange(120)
    z = truth + rng.normal(0, 0.05, 120)
    valid = np.ones(120, bool)
    valid[50:60] = False
    out = rts_smooth(z, valid, meas_sd=0.05, accel_sd=1e-4)
    assert np.abs(out - truth).mean() < 0.4 * np.abs(z - truth).mean()
    assert np.abs(out[50:60] - truth[50:60]).max() < 0.05


def test_smooth_homographies_predicts_a_dead_run_and_clamps() -> None:
    """Dead frames get a prediction; a solve the smoother disagrees with by > clamp_m is kept."""
    homs = {}
    for i in range(120):
        if 60 <= i < 70:
            continue  # a dead run the smoother must fill
        homs[i] = compose(Pose(0.35 + 0.0008 * i, 1.85, 0.02, TRUTH.log_f, TRUTH.log_aspect,
                               2.0, -55.0, 15.0))
    out, stats = smooth_homographies(homs, 120)
    assert stats["predicted"] == 10 and stats["clamped"] == 0, stats
    assert set(out) == set(range(120))
    assert grid_shift_m(out[65], compose(Pose(0.35 + 0.0008 * 65, 1.85, 0.02, TRUTH.log_f,
                                              TRUTH.log_aspect, 2.0, -55.0, 15.0))) < 0.05
    # dead_only leaves every solved frame alone
    only, _ = smooth_homographies(homs, 120, dead_only=True)
    assert set(only) == set(range(60, 70)), sorted(only)[:5]
    # an absurd clamp keeps the raw solve everywhere it exists
    tight, st_tight = smooth_homographies(homs, 120, clamp_m=0.0)
    assert st_tight["clamped"] == len(homs)
    assert all(np.allclose(tight[f], homs[f]) for f in homs)
