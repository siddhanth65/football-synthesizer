"""Tests for the ball-detector scoring (pure; no model/video)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.ball_eval import score_detections


def _gt(frames_visible, frames_invisible=()):
    rows = [{"frame": f, "x": 100.0, "y": 50.0, "visible": 1} for f in frames_visible]
    rows += [{"frame": f, "x": 0.0, "y": 0.0, "visible": 0} for f in frames_invisible]
    return pd.DataFrame(rows)


def test_perfect_detection_scores_one():
    gt = _gt([1, 2, 3])
    det = {1: (100.0, 50.0), 2: (101.0, 50.0), 3: (100.0, 51.0)}  # all within tol
    r = score_detections(gt, det, tol_px=12.0)
    assert r["tp"] == 3 and r["fp"] == 0 and r["fn"] == 0
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["med_err_px"] <= 1.0


def test_far_detection_is_fp_and_fn():
    gt = _gt([1])
    det = {1: (300.0, 300.0)}            # fired, but nowhere near the ball
    r = score_detections(gt, det, tol_px=12.0)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_detection_on_invisible_frame_is_false_positive():
    gt = _gt([], frames_invisible=[7])
    r = score_detections(gt, {7: (10.0, 10.0)}, tol_px=12.0)
    assert r["fp"] == 1 and r["tn"] == 0
    r2 = score_detections(gt, {}, tol_px=12.0)
    assert r2["tn"] == 1 and r2["fp"] == 0


def test_missed_visible_ball_is_false_negative():
    gt = _gt([1, 2])
    r = score_detections(gt, {1: (100.0, 50.0)}, tol_px=12.0)   # frame 2 missed
    assert r["tp"] == 1 and r["fn"] == 1
    assert r["recall"] == 0.5


def test_nan_when_no_detections_at_all():
    r = score_detections(_gt([1, 2]), {}, tol_px=12.0)
    assert np.isnan(r["precision"])     # no detections -> precision undefined
    assert r["recall"] == 0.0
