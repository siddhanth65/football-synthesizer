"""Tests for the pure box-matching core of tools/validate_detection (no models/video needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validate_detection import (  # noqa: E402
    consensus_boxes,
    greedy_match,
    iou_matrix,
    score,
)


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]])
    m = iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert abs(m[0, 0] - 1.0) < 1e-9  # identical
    assert 0.0 < m[0, 1] < 1.0  # partial overlap
    assert m[0, 2] == 0.0  # disjoint


def test_greedy_match_and_score_counts_fp_fn():
    pred = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])  # last is a false positive
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    pairs, fp, fn = greedy_match(pred, gt)
    assert len(pairs) == 2 and fp == 1 and fn == 0
    s = score(pred, gt)
    assert s["tp"] == 2 and s["fp"] == 1 and s["fn"] == 0
    assert abs(s["recall"] - 1.0) < 1e-9 and abs(s["precision"] - 2 / 3) < 1e-9


def test_score_missed_gt_is_a_false_negative():
    pred = np.array([[0, 0, 10, 10]])
    gt = np.array([[0, 0, 10, 10], [40, 40, 50, 50]])  # second GT is missed
    s = score(pred, gt)
    assert s["tp"] == 1 and s["fn"] == 1 and abs(s["recall"] - 0.5) < 1e-9


def test_consensus_keeps_agreed_box_drops_lone_box():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[1, 1, 11, 11]])  # strong overlap with a -> 2 detectors agree
    c = np.array([[100, 100, 110, 110]])  # lone detection
    cons = consensus_boxes([a, b, c], min_agree=2)
    assert len(cons) == 1
    assert iou_matrix(cons, np.array([[100, 100, 110, 110]]))[0, 0] < 0.5  # the lone box excluded
