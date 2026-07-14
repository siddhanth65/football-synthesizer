"""Tests for :mod:`generator.pose_carry` (player pose carry-over)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from generator.pose_carry import (
    carry_player_poses,
    geometry_yield,
    good_geometry_frames,
    project_players,
)
from generator.postprocess import MIN_ONPITCH_PLAYERS

cv2 = pytest.importorskip("cv2")

# A simple, invertible image->pitch homography (scale + translate; no projective term).
H_TRUE = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 1.0]])


def _pitch_to_image(px: float, py: float) -> tuple[float, float]:
    return px / 0.05, py / 0.05


def _frame_rows(frame: int, pitch: list[tuple[float, float]], *, valid: bool,
                calib: float = 0.2) -> list[dict]:
    """Rows for one frame; ``valid=False`` NaNs the pitch coords (a geometry-less frame)."""
    rows = []
    for i, (px, py) in enumerate(pitch):
        ix, iy = _pitch_to_image(px, py)
        rows.append({
            "frame": frame, "track_id": i, "role": "player", "team": i % 2,
            "pitch_x": px if valid else np.nan, "pitch_y": py if valid else np.nan,
            "image_x": ix, "image_y": iy, "conf": 0.9, "calib_error_m": calib,
            "is_actor": False, "is_keeper": False,
        })
    return rows


def _spread(n: int = 10) -> list[tuple[float, float]]:
    """``n`` players spread far enough to clear the pitch-span gate."""
    return [(5.0 + 9.0 * i, 10.0 + 4.0 * (i % 5)) for i in range(n)]


def _dense(good_frames, bad_frames, *, n_players: int = 10, bad_players: int | None = None):
    rows = []
    for f in good_frames:
        rows += _frame_rows(f, _spread(n_players), valid=True)
    for f in bad_frames:
        rows += _frame_rows(f, _spread(bad_players or n_players), valid=False, calib=9.9)
    return pd.DataFrame(rows)


def test_good_geometry_frames_and_yield():
    """Only frames with >= MIN_ONPITCH_PLAYERS valid pitch coords count as good."""
    df = _dense([0, 5], [10])
    assert good_geometry_frames(df).tolist() == [0, 5]
    assert geometry_yield(df) == pytest.approx(2 / 3)


def test_project_players_gates_offpitch():
    """Points projecting beyond the pitch tolerance become NaN, never clamped onto the line."""
    img = np.array([[100.0, 200.0], [1e5, 1e5]])
    out = project_players(H_TRUE, img)
    assert out[0] == pytest.approx([5.0, 10.0])
    assert np.isnan(out[1]).all()


def test_carry_recovers_a_geometry_less_frame():
    """A frame with no geometry borrows a camera-continuous neighbour's pose and is recovered."""
    df = _dense([0, 10], [5])
    out, st = carry_player_poses(df, window=50)
    assert st.good_before == 2
    assert st.carried + st.interpolated == 1
    assert st.good_after == 3
    rec = out[(out["frame"] == 5) & (out["role"] == "player")]
    assert rec["pitch_x"].notna().all()
    # the borrowed pose is the true one here, so the recovered coords are the true coords
    np.testing.assert_allclose(sorted(rec["pitch_x"]), sorted(p[0] for p in _spread(10)), atol=1e-3)
    assert set(rec["pose_source"]) == {"interp"}  # bracketed on both sides


def test_carry_stamps_source_calib_error():
    """A carried frame inherits its SOURCE pose's reprojection error (the honest provenance).

    Without this the downstream ``calib_error_m <= 1 m`` fact-store gate would drop every carried
    frame -- the exact reason the ball carry-over contributed zero possession samples.
    """
    df = _dense([0, 10], [5])
    out, _ = carry_player_poses(df, window=50)
    assert float(out[out["frame"] == 5]["calib_error_m"].iloc[0]) == pytest.approx(0.2)
    assert float(out[out["frame"] == 0]["calib_error_m"].iloc[0]) == pytest.approx(0.2)


def test_frame_plausibility_gate_rejects_thin_frames():
    """A close-up (too few players to reach MIN_ONPITCH_PLAYERS) is NOT recovered at any pose."""
    df = _dense([0, 10], [5], n_players=10, bad_players=MIN_ONPITCH_PLAYERS - 3)
    out, st = carry_player_poses(df, window=50)
    assert st.carried + st.interpolated == 0
    assert st.frame_gate_dropped == 1
    assert out[out["frame"] == 5]["pitch_x"].isna().all()


def test_carry_never_crosses_a_camera_cut():
    """A track-id reset (camera cut) between source and target blocks the carry."""
    good = _dense([0], [])
    bad = _dense([], [5])
    far = _dense([10], [])
    bad["track_id"] += 500   # a cut: no track ids survive -> Jaccard 0
    far["track_id"] += 500
    df = pd.concat([good, bad, far], ignore_index=True)
    _, st = carry_player_poses(df, window=50)
    # frame 5 sits in a new segment; its only same-segment source is frame 10
    assert st.no_source + st.carried + st.interpolated == st.targets


def test_window_blocks_a_distant_source():
    """No good frame inside +-window -> no carry (targets are reported as no_source)."""
    df = _dense([0, 400], [200])
    _, st = carry_player_poses(df, window=50)
    assert st.carried + st.interpolated == 0
    assert st.no_source == 1


def test_good_frames_are_left_untouched():
    """Carry-over never rewrites a frame that already had usable geometry."""
    df = _dense([0, 10], [5])
    out, _ = carry_player_poses(df, window=50)
    for f in (0, 10):
        a = df[df["frame"] == f].sort_values("track_id")
        b = out[out["frame"] == f].sort_values("track_id")
        np.testing.assert_allclose(a["pitch_x"].to_numpy(), b["pitch_x"].to_numpy())
        assert set(b["pose_source"]) == {"own"}


def test_empty_source_pool_is_a_noop():
    """With no good frame anywhere, the table comes back unchanged."""
    df = _dense([], [0, 5, 10])
    out, st = carry_player_poses(df, window=50)
    assert st.good_before == 0
    assert st.good_after == 0
    assert out["pitch_x"].isna().all()
