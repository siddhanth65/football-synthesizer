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

#: Process-wide flag, default OFF (the shipped behaviour). When True, :class:`ByteTrackBackend` also
#: emits the detections ByteTrack did not return, each on a fresh track id.
#:
#: ``supervision.ByteTrack.update_with_detections`` re-matches its *Kalman* track boxes to the input
#: boxes at IoU >= 0.5 and returns only the input rows that match, so a detection is discarded both
#: when its track is unconfirmed and when the motion model drifts (camera pan). On GSR DEV-20 that
#: costs 8.0% of the detector's confident boxes (250,705 -> 230,753) and is 50% of the whole GS-DetA
#: detection-miss bucket. Nothing downstream needs the filtering: ``tools.gsr_eiou.associate``
#: re-derives every tracklet from raw boxes and never drops a row.
KEEP_UNTRACKED: bool = False

#: First id handed to a written-through detection; far above any ByteTrack external id, so the two
#: id spaces cannot collide inside one sequence.
UNTRACKED_ID_BASE = 1_000_000

#: Process-wide default for ByteTrack's ``minimum_consecutive_frames`` (shipped: 3). A knob because
#: :func:`generator.extract.extract_positions` builds the tracker itself and takes no tracker args.
MIN_HITS: int = 3


def untracked_mask(boxes: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Which input boxes the tracker did not return (pure).

    ``supervision.ByteTrack`` returns a *subset of its input rows verbatim* (it selects on the
    Detections it was handed), so identity is exact box equality -- no tolerance needed, and no
    second IoU pass that could disagree with the tracker's own bookkeeping.

    Args:
        boxes: ``(N, 4)`` boxes handed to the tracker.
        kept: ``(M, 4)`` boxes it returned.

    Returns:
        ``(N,)`` bool mask, True where the row was dropped.
    """
    if len(boxes) == 0:
        return np.zeros(0, bool)
    if len(kept) == 0:
        return np.ones(len(boxes), bool)
    survivors = {tuple(b) for b in np.asarray(kept, float)}
    return np.array([tuple(b) not in survivors for b in np.asarray(boxes, float)])


def build_tracker(name: str = "botsort", *, min_hits: int | None = None, lost_buffer: int = 60):
    """Construct a tracking backend. ``name`` in {``botsort``, ``bytetrack``}.

    ``min_hits=None`` takes the process-wide :data:`MIN_HITS` (3, the shipped value).
    """
    if name == "bytetrack":
        return ByteTrackBackend(min_hits=MIN_HITS if min_hits is None else min_hits,
                                lost_buffer=lost_buffer)
    if name == "botsort":
        return BotSortBackend()
    raise ValueError(f"unknown tracker {name!r} (expected 'botsort' or 'bytetrack')")


class ByteTrackBackend:
    """``supervision.ByteTrack`` on the detector's per-frame boxes (the image-space baseline)."""

    def __init__(self, *, min_hits: int | None = None, lost_buffer: int = 60):
        import supervision as sv  # noqa: PLC0415

        min_hits = MIN_HITS if min_hits is None else min_hits

        self._sv = sv
        self._bt = sv.ByteTrack(minimum_consecutive_frames=min_hits, lost_track_buffer=lost_buffer)
        self._next_id = UNTRACKED_ID_BASE

    def update(self, detector, frame_rgb):
        boxes, confs, role_ids, ball_xy = detector.detect(frame_rgb)
        if len(boxes) == 0:
            return (*_EMPTY, ball_xy)
        det = self._sv.Detections(xyxy=boxes, confidence=confs, class_id=role_ids)
        out = self._bt.update_with_detections(det)
        tids = (np.asarray(out.tracker_id, int) if out.tracker_id is not None
                else np.full(len(out), -1))
        cls = (np.asarray(out.class_id, int) if out.class_id is not None
               else np.zeros(len(out), int))
        xyxy, conf = out.xyxy, out.confidence
        if KEEP_UNTRACKED:
            drop = untracked_mask(boxes, xyxy)
            if drop.any():
                n = int(drop.sum())
                fresh = np.arange(self._next_id, self._next_id + n)
                self._next_id += n
                xyxy = np.vstack([xyxy, boxes[drop]])
                conf = np.concatenate([conf, confs[drop]])
                cls = np.concatenate([cls, np.asarray(role_ids, int)[drop]])
                tids = np.concatenate([tids, fresh])
        if len(xyxy) == 0:
            return (*_EMPTY, ball_xy)
        return xyxy, conf, cls.astype(int), tids.astype(int), ball_xy


class BotSortBackend:
    """Ultralytics BoT-SORT (GMC + optional ReID) via the YOLO detector's ``track()``."""

    def update(self, detector, frame_rgb):
        if not hasattr(detector, "track"):
            raise ValueError(
                "the 'botsort' tracker needs a YOLO-based detector exposing track() "
                "(use --detector football or yolo, or switch to --tracker bytetrack)"
            )
        return detector.track(frame_rgb)
