"""CPU tests for the pure seams of tools.closeup_anchor_probe (no GPU/video).

The shot-yield helper ``_shot_yield`` was refactored into :func:`tools.closeup_anchor_probe._shots`
(segmentation) plus an inline "shots with an anchor" tally; these tests track the current API.
"""
from __future__ import annotations

from tools.closeup_anchor_probe import SHOT_GAP_FRAMES, _shots


def _shots_with_anchor(frames: list[int], anchors: list[dict]) -> tuple[int, int]:
    """(#shots, #shots containing an anchor frame) -- the tally run_full/run_levers compute inline."""
    shots = _shots(frames)
    af = {a["frame"] for a in anchors}
    return len(shots), sum(any(lo <= f <= hi for f in af) for lo, hi in shots)


def test_shot_yield_segments_contiguous_runs() -> None:
    """Frames within SHOT_GAP_FRAMES form one shot; a bigger gap splits."""
    # two runs: {100,105,110} and {200,205}; gap 90 > SHOT_GAP_FRAMES splits them.
    assert SHOT_GAP_FRAMES < 90
    frames = [100, 105, 110, 200, 205]
    anchors = [{"frame": 105}]  # lands in the first shot only
    assert _shots_with_anchor(frames, anchors) == (2, 1)


def test_shot_yield_empty() -> None:
    assert _shots_with_anchor([], []) == (0, 0)


def test_shot_yield_all_anchored() -> None:
    frames = [10, 12, 14]  # one shot (gaps <= SHOT_GAP_FRAMES)
    assert _shots_with_anchor(frames, [{"frame": 14}]) == (1, 1)
