"""Unit tests for the temporal homography carry-over lever (:mod:`generator.ball_carry`).

Realistic-usage invariant mirrored here: a ball detection is only ever produced on a *dense* frame,
so a carry target always appears in the dense table (with 1+ tracked bodies) even when it has too few
correspondences to fit its own homography. Tests therefore add "thin" frames (few players) as carry
targets rather than frames absent from the dense table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from generator.ball_carry import (
    _nearest_sources,
    camera_segment_ids,
    fit_frame_homographies,
    project_ball_carry,
)

cv2 = pytest.importorskip("cv2")


def _players(frame: int, track_ids: list[int]) -> pd.DataFrame:
    """Players at ``frame`` whose image coords equal pitch coords (fitted homography ~ identity)."""
    rng = np.random.default_rng(frame)
    pts = rng.uniform(5, 60, size=(len(track_ids), 2))
    return pd.DataFrame({
        "frame": frame, "track_id": track_ids, "role": "player",
        "image_x": pts[:, 0], "image_y": pts[:, 1], "pitch_x": pts[:, 0], "pitch_y": pts[:, 1],
        "team": 0, "conf": 1.0, "calib_error_m": 0.1,
    })


def _good(frame: int, n: int = 8) -> pd.DataFrame:
    """A frame with enough correspondences to fit its own homography."""
    return _players(frame, list(range(n)))


def test_fit_frame_homographies_needs_min_pts() -> None:
    """A frame with < min_pts valid correspondences yields no homography."""
    homs = fit_frame_homographies(pd.concat([_good(10, 8), _players(20, [0, 1, 2, 3])]), min_pts=6)
    assert 10 in homs and 20 not in homs


def test_fit_identity_projects_points_unchanged() -> None:
    """An identity-correspondence frame maps image (x, y) to the same pitch (x, y)."""
    homs = fit_frame_homographies(_good(10), min_pts=6)
    p = cv2.perspectiveTransform(np.array([[[30.0, 20.0]]], np.float32), homs[10])[0, 0]
    assert p == pytest.approx([30.0, 20.0], abs=1e-3)


def test_camera_segment_ids_splits_on_cut() -> None:
    """A track-id set with no overlap between two frame blocks is two camera segments."""
    seg = camera_segment_ids(
        pd.concat([_players(0, [1, 2]), _players(5, [1, 2]),
                   _players(10, [90, 91]), _players(15, [90, 91])]), jaccard_cut=0.3)
    assert seg[0] == seg[5] == 0
    assert seg[10] == seg[15] == 1


def test_camera_segment_ids_no_cut_when_overlap_high() -> None:
    """Stable track ids across frames stay in one camera segment."""
    seg = camera_segment_ids(pd.concat([_players(0, [1, 2, 3]), _players(5, [1, 2, 3])]),
                             jaccard_cut=0.3)
    assert seg[0] == seg[5] == 0


def test_nearest_sources_respects_window_and_segment() -> None:
    """Nearest source is the closest good frame in-window and in the same camera segment."""
    good = np.array([0, 40, 200])
    seg = {0: 0, 40: 0, 100: 0, 200: 1}
    left, right = _nearest_sources(100, good, seg, window=50)
    assert left is None and right is None  # 40 is 60 frames away (> window); 200 is another segment
    left2, right2 = _nearest_sources(100, good, seg, window=150)
    assert left2 == 40 and right2 is None  # 40 now in-window; 200 excluded by segment


def test_carry_off_matches_own_only() -> None:
    """With carry=False only own-frame homographies project (baseline behaviour)."""
    dense = pd.concat([_good(10), _players(12, [0, 1, 2])])  # frame 12 too thin for own H
    ball = {10: (30.0, 20.0), 12: (25.0, 25.0)}
    df, stats = project_ball_carry(ball, dense, carry=False)
    assert stats.own == 1 and stats.no_source == 1
    assert set(df["frame"]) == {10}


def test_carry_projects_uncalibrated_frame() -> None:
    """A ball frame lacking its own homography is projected via a carried neighbour homography."""
    # frame 15 is thin but shares track ids with the good frames -> same camera segment, no own H.
    dense = pd.concat([_good(10), _players(15, [0, 1, 2]), _good(20)])
    df, stats = project_ball_carry({15: (30.0, 20.0)}, dense, carry=True, interpolate=True, window=50)
    assert len(df) == 1 and stats.carried + stats.interpolated == 1
    assert df.iloc[0][["x", "y"]].to_numpy() == pytest.approx([30.0, 20.0], abs=1e-2)


def test_carry_blocked_across_cut() -> None:
    """Carry-over never crosses a camera cut (disjoint track ids -> different segment)."""
    # good frame 10 (ids 0-7); target frame 15 shares no ids -> a cut sits between them.
    dense = pd.concat([_good(10), _players(15, [500, 501, 502])])
    df, stats = project_ball_carry({15: (30.0, 20.0)}, dense, carry=True, window=100)
    assert df.empty and stats.no_source == 1


def test_offpitch_carry_is_rejected() -> None:
    """A carried projection landing off the pitch is dropped, not planted on the touchline."""
    dense = pd.concat([_good(10), _players(15, [0, 1, 2]), _good(20)])
    df, stats = project_ball_carry({15: (500.0, 500.0)}, dense, carry=True, window=50)
    assert df.empty and stats.offpitch_dropped == 1
