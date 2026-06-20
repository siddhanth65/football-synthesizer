"""Tests for the temporal-calibration policy + reuse/no-resurrect semantics (Pass B).

The pure ``decide_recalibration`` policy and the wrapper's bookkeeping are tested with a fake base
calibrator and the frame-motion signals stubbed out, so no video/model is needed.
"""

from __future__ import annotations

import numpy as np

from generator.calibrate import CalibrationResult
from generator.temporal_calib import TemporalCalibrator, decide_recalibration


def _ok():
    return CalibrationResult(np.eye(3), 0.27, 20, ok=True)


def _bad():
    return CalibrationResult(None, float("inf"), 0, ok=False)


class _FakeBase:
    def __init__(self, seq):
        self.seq = list(seq)
        self.calls = 0

    def calibrate_frame(self, frame_bgr):
        r = self.seq[min(self.calls, len(self.seq) - 1)]
        self.calls += 1
        return r


def _static(base, **kw):
    """A TemporalCalibrator with motion signals stubbed: no cut, no drift, identity gray."""
    tc = TemporalCalibrator(base, **kw)
    tc._gray = lambda f: f
    tc._is_cut = lambda g: False
    tc._drift = lambda g: 0.0
    return tc


_F = np.zeros((4, 4, 3), np.uint8)


def test_decide_recalibration_policy():
    assert decide_recalibration(1, 25, 0.0, 6.0, True, True)       # cut
    assert decide_recalibration(1, 25, 0.0, 6.0, False, False)     # nothing to reuse
    assert decide_recalibration(25, 25, 0.0, 6.0, False, True)     # period elapsed
    assert decide_recalibration(2, 25, 9.0, 6.0, False, True)      # camera drift
    assert not decide_recalibration(2, 25, 0.5, 6.0, False, True)  # static -> reuse


def test_reuses_pose_within_period():
    base = _FakeBase([_ok()])
    tc = _static(base, period=10)
    results = [tc.calibrate_frame(_F) for _ in range(5)]
    assert base.calls == 1  # only the first frame ran a full calibration
    assert all(r.ok for r in results)  # the rest reused the accepted pose
    assert tc.n_full == 1 and tc.n_reuse == 4


def test_recalibrates_every_period_frames():
    base = _FakeBase([_ok()])
    tc = _static(base, period=2)
    for _ in range(6):
        tc.calibrate_frame(_F)
    assert base.calls == 3  # frames 0, 2, 4 -> full; 1, 3, 5 -> reused
    assert tc.n_reuse == 3


def test_never_reuses_a_rejected_calibration():
    base = _FakeBase([_bad(), _bad(), _ok()])
    tc = _static(base, period=10)
    assert not tc.calibrate_frame(_F).ok and base.calls == 1
    assert not tc.calibrate_frame(_F).ok and base.calls == 2  # did NOT reuse the reject
    assert tc.calibrate_frame(_F).ok and base.calls == 3
    assert tc.n_reuse == 0


def test_cut_forces_recalibration_despite_unspent_period():
    base = _FakeBase([_ok()])
    tc = _static(base, period=100)
    tc.calibrate_frame(_F)  # full
    tc._is_cut = lambda g: True  # next frame is a shot cut
    tc.calibrate_frame(_F)
    assert base.calls == 2  # recalibrated even though the period had not elapsed
