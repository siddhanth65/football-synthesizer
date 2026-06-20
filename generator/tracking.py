"""Tracking backends behind one interface, so the generator can swap trackers without touching its
per-frame loop.

* ``bytetrack`` (default -- won the benchmark on this footage): ``supervision.ByteTrack`` on
  image-space boxes, with a 3-frame confirmation that suppresses spurious tracks. On this broadcast it
  gave ~22 IDs (≈ the true player count) with long tracks.
* ``botsort``: Ultralytics BoT-SORT via the YOLO detector's ``track()``, adding **global motion
  compensation** (``gmc_method: sparseOptFlow``) + appearance ReID (see ``botsort_tuned.yaml``).
  Designed for heavy camera pan/zoom -- but on this clip, even tuned, it fragmented more than
  ByteTrack (GMC has little to compensate on the held tactical camera). Kept available for high-motion
  footage; re-benchmark per source before switching.

Both expose ``update(detector, frame_rgb) -> (xyxy, conf, role_ids, track_ids, ball_xy)`` so the caller
only sees tracked boxes + their canonical role ids + the ball, never the backend's internals.
"""

from __future__ import annotations

import numpy as np

_EMPTY = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int))


def build_tracker(name: str = "botsort", *, min_hits: int = 3, lost_buffer: int = 60):
    """Construct a tracking backend. ``name`` in {``botsort``, ``bytetrack``}."""
    if name == "bytetrack":
        return ByteTrackBackend(min_hits=min_hits, lost_buffer=lost_buffer)
    if name == "botsort":
        return BotSortBackend()
    raise ValueError(f"unknown tracker {name!r} (expected 'botsort' or 'bytetrack')")


class ByteTrackBackend:
    """``supervision.ByteTrack`` on the detector's per-frame boxes (the image-space baseline)."""

    def __init__(self, *, min_hits: int = 3, lost_buffer: int = 60):
        import supervision as sv  # noqa: PLC0415

        self._sv = sv
        self._bt = sv.ByteTrack(minimum_consecutive_frames=min_hits, lost_track_buffer=lost_buffer)

    def update(self, detector, frame_rgb):
        boxes, confs, role_ids, ball_xy = detector.detect(frame_rgb)
        if len(boxes) == 0:
            return (*_EMPTY, ball_xy)
        det = self._sv.Detections(xyxy=boxes, confidence=confs, class_id=role_ids)
        det = self._bt.update_with_detections(det)
        if len(det) == 0:
            return (*_EMPTY, ball_xy)
        tids = det.tracker_id if det.tracker_id is not None else np.full(len(det), -1)
        cls = det.class_id if det.class_id is not None else np.zeros(len(det), int)
        return det.xyxy, det.confidence, cls.astype(int), np.asarray(tids, int), ball_xy


class BotSortBackend:
    """Ultralytics BoT-SORT (GMC + optional ReID) via the YOLO detector's ``track()``."""

    def update(self, detector, frame_rgb):
        if not hasattr(detector, "track"):
            raise ValueError(
                "the 'botsort' tracker needs a YOLO-based detector exposing track() "
                "(use --detector football or yolo, or switch to --tracker bytetrack)"
            )
        return detector.track(frame_rgb)
