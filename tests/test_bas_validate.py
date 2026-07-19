"""Unit tests for the pure seams of :mod:`tools.bas_validate` (dedup, segment build/membership)."""
from __future__ import annotations

import numpy as np

from tools.bas_validate import build_live_segments, dedup_peaks, in_any_segment


def test_dedup_keeps_highest_conf_within_gap() -> None:
    # Three peaks inside one 25-frame window: only the highest-conf survives; the far one stays.
    frames = [100, 110, 118, 400]
    confs = [0.5, 0.9, 0.6, 0.8]
    assert dedup_peaks(frames, confs, gap_frames=25) == [110, 400]


def test_dedup_zero_gap_is_identity() -> None:
    frames = [10, 12, 14]
    assert dedup_peaks(frames, [0.1, 0.2, 0.3], gap_frames=0) == [10, 12, 14]


def test_dedup_chain_suppression_is_from_kept_peaks_only() -> None:
    # 0 and 20 are >gap apart so both kept; 10 is within gap of the higher-conf 20 -> dropped.
    assert dedup_peaks([0, 10, 20], [0.9, 0.5, 0.8], gap_frames=15) == [0, 20]


def test_build_live_segments_merges_small_gap_and_pads() -> None:
    grid = np.array([0, 5, 10, 15, 20, 25, 30])
    is_live = np.array([True, True, False, False, True, True, False])
    # Two runs (0-5) and (20-25); gap 15 frames. Merge if gap<=15, pad 2 -> one segment (-2, 27).
    segs = build_live_segments(grid, is_live, gap_merge_frames=15, pad_frames=2)
    assert segs == [(-2, 27)]


def test_build_live_segments_keeps_runs_apart_when_gap_too_large() -> None:
    grid = np.array([0, 5, 10, 15, 20, 25, 30])
    is_live = np.array([True, True, False, False, True, True, False])
    segs = build_live_segments(grid, is_live, gap_merge_frames=5, pad_frames=0)
    assert segs == [(0, 5), (20, 25)]


def test_build_live_segments_empty_when_no_live() -> None:
    grid = np.array([0, 5, 10])
    assert build_live_segments(grid, np.array([False, False, False]), 10, 1) == []


def test_in_any_segment_boundaries_inclusive() -> None:
    segs = [(10, 20), (40, 50)]
    assert in_any_segment(10, segs) and in_any_segment(50, segs)
    assert not in_any_segment(25, segs)
    assert not in_any_segment(9, segs)
