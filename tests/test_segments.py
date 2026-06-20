"""Tests for the wide-shot scoring (pure; the CV scan driver is exercised via the scorer)."""

from __future__ import annotations

import numpy as np

from generator.segments import MIN_PLAYERS_WIDE, best_wide_segment, wide_shot_score


def _boxes(centers_x, *, w=20):
    """Build (N,4) boxes from x-centres (y fixed); width w per box."""
    return np.array([[cx - w / 2, 100, cx + w / 2, 200] for cx in centers_x])


def test_too_few_players_scores_zero():
    boxes = _boxes(np.linspace(100, 900, MIN_PLAYERS_WIDE - 1))
    assert wide_shot_score(boxes, 1024) == 0.0


def test_wide_spread_scores_higher_than_clustered():
    n = 16
    wide = _boxes(np.linspace(50, 1000, n))      # spread across the frame
    tight = _boxes(np.linspace(480, 540, n))     # clustered in the centre
    sw = wide_shot_score(wide, 1024)
    st = wide_shot_score(tight, 1024)
    assert sw > st
    assert sw > 1.0 and st < sw / 3  # clearly separated


def test_zero_width_is_safe():
    assert wide_shot_score(_boxes(np.linspace(0, 100, 14)), 0) == 0.0


def test_best_wide_segment_picks_top_score():
    scan = [(100, 2.0, 12), (13050, 5.8, 21), (9600, 4.1, 19)]
    # scan as returned by scan_wide_segments is sorted desc, but best_ should pick the max anyway:
    scan_sorted = sorted(scan, key=lambda r: r[1], reverse=True)
    assert best_wide_segment(scan_sorted) == 13050
    assert best_wide_segment([]) is None
