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


# --- v9 W7: the keep-untracked pass-through -------------------------------------------------------
class _FixedDetector:
    """Detector stub returning one fixed per-frame detection set (no ball)."""

    def __init__(self, boxes, confs, roles):
        self._ret = (boxes, confs, roles, None)

    def detect(self, frame_rgb):
        return self._ret


def test_untracked_mask_is_exact_box_identity():
    from generator.tracking import untracked_mask

    boxes = np.array([[0.0, 0, 10, 20], [50.0, 0, 60, 20], [100.0, 0, 110, 20]])
    kept = boxes[[0, 2]]
    assert untracked_mask(boxes, kept).tolist() == [False, True, False]
    assert untracked_mask(boxes, np.zeros((0, 4))).tolist() == [True] * 3
    assert untracked_mask(np.zeros((0, 4)), kept).tolist() == []


def test_keep_untracked_emits_every_detection_with_unique_ids():
    """With the flag ON no detection is lost, and the extra ids never collide with ByteTrack's."""
    import generator.tracking as tk

    boxes = np.array([[0.0, 0, 10, 20], [50.0, 0, 60, 20]])
    confs = np.array([0.9, 0.9])
    roles = np.array([0, 0])
    det = _FixedDetector(boxes, confs, roles)

    off = tk.ByteTrackBackend(min_hits=3)
    first_off = off.update(det, frame_rgb=None)[0]
    assert len(first_off) == 0  # 3-frame confirmation: nothing on frame 1

    tk.KEEP_UNTRACKED = True
    try:
        on = tk.ByteTrackBackend(min_hits=3)
        seen_ids = set()
        for _ in range(4):
            xyxy, conf, cls, tids, _ball = on.update(det, frame_rgb=None)
            assert len(xyxy) == len(boxes), "a detection was dropped with the flag ON"
            assert len(conf) == len(cls) == len(tids) == len(boxes)
            assert len(set(tids.tolist())) == len(boxes), "duplicate ids inside one frame"
            seen_ids.update(tids.tolist())
    finally:
        tk.KEEP_UNTRACKED = False
    assert any(i >= tk.UNTRACKED_ID_BASE for i in seen_ids)
