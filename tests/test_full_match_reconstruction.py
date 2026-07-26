"""Unit tests for the full-match reconstruction seams that a bug would silently corrupt."""

from __future__ import annotations

import numpy as np

from tools.full_match_reconstruction import (
    ChunkRun,
    Event,
    _drop_chained,
    fill_internal_gaps,
    within_track_events,
)


def _bundle(vis: np.ndarray, fps: float = 5.0) -> dict:
    """Minimal bundle: ``visible`` drives everything, ``truth`` records where a sighting was."""
    n, s = vis.shape
    truth = np.full((n, s, 2), np.nan)
    idx = np.nonzero(vis)
    truth[idx[0], idx[1], 0] = idx[0].astype(float)
    truth[idx[0], idx[1], 1] = 0.0
    return {"visible": vis.copy(), "truth": truth, "fps": fps,
            "slot_track": {1: dict(enumerate(range(s)))}}


def _run(vis: np.ndarray) -> ChunkRun:
    """A :class:`ChunkRun` carrying only the fields the event finders read."""
    empty = np.zeros((0, 2))
    return ChunkRun("h1_chunk_000", _bundle(vis), {}, empty, empty, empty,
                    np.zeros(0, dtype=int), {})


def test_fill_only_touches_frames_between_two_sightings() -> None:
    """Internal gaps get a placeholder; the pre-first and post-last tails stay NaN."""
    vis = np.array([[False], [True], [False], [False], [True], [False]])
    b = _bundle(vis)
    assert fill_internal_gaps(b) == 2
    assert np.allclose(b["truth"][2:4, 0, 0], 1.0)          # held from the frame-1 sighting
    assert not np.isfinite(b["truth"][0, 0, 0])             # before the first sighting
    assert not np.isfinite(b["truth"][5, 0, 0])             # after the last one
    assert b["truth"][4, 0, 0] == 4.0                       # a sighting is never overwritten
    assert not b["visible"][2, 0]                           # visibility is untouched


def test_liveness_guard_rejects_a_whole_frame_outage() -> None:
    """A gap whose frames carry no other tracking is a calibration hole, not an occlusion."""
    n = 12
    vis = np.zeros((n, 6), dtype=bool)
    vis[:, 1:] = True                       # five companions keep the frame "alive"
    vis[:, 0] = True
    vis[4:7, 0] = False                     # slot 0 occluded while the camera keeps tracking
    good = within_track_events(_run(vis), min_alive=4)
    assert [(e.t0, e.t1) for e in good if e.slot == 0] == [(3, 7)]
    assert good[0].dur_s == 4 / 5.0

    vis[4:7, :] = False                     # now nobody is tracked during the gap
    assert [e for e in within_track_events(_run(vis), min_alive=4) if e.slot == 0] == []


def test_chained_events_are_dropped() -> None:
    """Masking a re-appearance lengthens the next gap, so the successor event must be dropped."""
    ev = [Event("c", 0, 10, 20, 2.0, "within-track", (0.0, 0.0), (0,), "a"),
          Event("c", 0, 20, 30, 2.0, "within-track", (0.0, 0.0), (0,), "b"),
          Event("c", 0, 30, 40, 2.0, "within-track", (0.0, 0.0), (0,), "c"),
          Event("c", 1, 20, 25, 1.0, "within-track", (0.0, 0.0), (1,), "d")]
    assert [e.label for e in _drop_chained(ev)] == ["a", "c", "d"]
