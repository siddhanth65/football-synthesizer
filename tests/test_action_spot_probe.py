"""Pure-seam tests for the action-spotting probe (no torch / no GPU)."""
from __future__ import annotations

import numpy as np

from tools.action_spot_probe import (
    SOCCERNET_V2_CLASSES,
    class_index,
    find_peaks,
)


def test_class_index_goal_is_six() -> None:
    """Background is 0; foreground index = sorted position + 1 (load_classes convention)."""
    assert class_index("Goal") == 6
    assert class_index("Corner") == 3
    assert class_index("Ball out of play") == 1
    assert len(SOCCERNET_V2_CLASSES) == 17


def test_find_peaks_separation_and_threshold() -> None:
    """Two well-separated bumps -> two peaks; a close second bump is NMS-suppressed."""
    sig = np.zeros(200, dtype=np.float32)
    sig[50], sig[52], sig[150] = 0.9, 0.8, 0.7
    peaks = find_peaks(sig, thresh=0.5, min_sep=10)
    assert [p[0] for p in peaks] == [50, 150]
    assert abs(peaks[0][1] - 0.9) < 1e-6  # keeps the stronger of the suppressed pair
    assert find_peaks(sig, thresh=0.95, min_sep=10) == []


def test_find_peaks_empty_signal() -> None:
    """A silent timeline yields no peaks."""
    assert find_peaks(np.zeros(100, dtype=np.float32), thresh=0.3, min_sep=10) == []
