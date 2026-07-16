"""CPU tests for the pure seams of tools.closeup_anchor_probe (no GPU/video)."""
from __future__ import annotations

from tools.closeup_anchor_probe import SHOT_GAP_FRAMES, _shot_yield


def test_shot_yield_segments_contiguous_runs() -> None:
    """Frames within SHOT_GAP_FRAMES form one shot; a bigger gap splits."""
    # two runs: {100,105,110} and {200,205}; gap 90 > SHOT_GAP_FRAMES splits them.
    assert SHOT_GAP_FRAMES < 90
    frames = [100, 105, 110, 200, 205]
    anchors = [{"frame": 105}]  # lands in the first shot only
    n_shots, with_anchor = _shot_yield(frames, anchors)
    assert n_shots == 2
    assert with_anchor == 1


def test_shot_yield_empty() -> None:
    assert _shot_yield([], []) == (0, 0)


def test_shot_yield_all_anchored() -> None:
    frames = [10, 12, 14]  # one shot (gaps <= SHOT_GAP_FRAMES)
    n_shots, with_anchor = _shot_yield(frames, [{"frame": 14}])
    assert (n_shots, with_anchor) == (1, 1)
