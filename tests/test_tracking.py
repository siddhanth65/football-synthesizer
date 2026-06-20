"""Tests for the tracking-backend seam (dispatch + BoT-SORT delegation).

The pure seams here don't run a real tracker; the end-to-end BoT-SORT/GMC behaviour is exercised by
regenerating segments. ByteTrackBackend construction is covered (it just builds supervision.ByteTrack).
"""

from __future__ import annotations

import numpy as np
import pytest

from generator.tracking import BotSortBackend, ByteTrackBackend, build_tracker


class _FakeYoloDetector:
    """Stands in for a YOLO detector: exposes track() returning a fixed 5-tuple."""

    def __init__(self, ret):
        self._ret = ret

    def track(self, frame_rgb):
        return self._ret


def test_build_tracker_dispatch():
    assert isinstance(build_tracker("botsort"), BotSortBackend)
    assert isinstance(build_tracker("bytetrack"), ByteTrackBackend)
    with pytest.raises(ValueError, match="unknown tracker"):
        build_tracker("nope")


def test_botsort_backend_delegates_to_detector_track():
    ret = (np.zeros((2, 4)), np.array([0.9, 0.8]), np.array([0, 1]), np.array([5, 6]),
           np.array([1.0, 2.0]))
    out = BotSortBackend().update(_FakeYoloDetector(ret), frame_rgb=None)
    assert out is ret  # straight pass-through of the detector's tracked result


def test_botsort_backend_requires_a_track_method():
    class _DetectOnly:  # a detector with detect() but no track() (e.g. RF-DETR)
        def detect(self, frame_rgb):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None

    with pytest.raises(ValueError, match="botsort"):
        BotSortBackend().update(_DetectOnly(), frame_rgb=None)
