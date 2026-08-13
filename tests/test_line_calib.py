"""Line-evidence calibration: the model, the line solve, the support test and the dead-frame stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from generator.line_calib import (
    _project_to_image,
    _residuals_m,
    _segment_lines,
    line_mask,
    line_support,
    pitch_model_lines,
    recover_dead_frames,
    refit_homography,
    ridge_field,
    sample_model_points,
    solve_point_on_line,
)

cv2 = pytest.importorskip("cv2")

H_TRUE = np.linalg.inv(np.array([[7.0, 1.2, 60.0], [0.6, 4.0, 40.0], [0.0006, 0.004, 1.0]]))
SHAPE = (540, 960)


def _synthetic_frame() -> np.ndarray:
    """A turf-coloured frame with the pitch model drawn through ``H_TRUE``."""
    segs = pitch_model_lines()
    pts, _ = sample_model_points(segs)
    img_pts = _project_to_image(H_TRUE, pts)
    frame = np.zeros((*SHAPE, 3), np.uint8)
    frame[:] = (40, 120, 40)
    for x, y in img_pts:
        if 0 <= x < SHAPE[1] and 0 <= y < SHAPE[0]:
            cv2.circle(frame, (int(round(x)), int(round(y))), 2, (245, 245, 245), -1)
    return frame


def test_pitch_model_is_inside_the_pitch_and_lines_are_metric():
    segs = pitch_model_lines()
    assert segs.shape == (41, 2, 2)
    flat = segs.reshape(-1, 2)
    assert flat[:, 0].min() >= 0.0 and flat[:, 0].max() <= 105.0
    assert flat[:, 1].min() >= 0.0 and flat[:, 1].max() <= 68.0
    lines = _segment_lines(segs)
    assert np.allclose(np.linalg.norm(lines[:, :2], axis=1), 1.0)
    # a point 3 m inside the touchline y=0 is 3 m from it
    touchline = lines[0]
    assert abs(abs(float(touchline @ np.array([50.0, 3.0, 1.0]))) - 3.0) < 1e-9


def test_line_solve_recovers_the_homography_exactly():
    segs = pitch_model_lines()
    pts, seg_id = sample_model_points(segs)
    img = _project_to_image(H_TRUE, pts)
    inside = ((img[:, 0] >= 0) & (img[:, 0] < SHAPE[1])
              & (img[:, 1] >= 0) & (img[:, 1] < SHAPE[0]))
    lines = _segment_lines(segs)[seg_id[inside]]
    h = solve_point_on_line(img[inside], lines, SHAPE)
    assert h is not None
    assert np.abs(_residuals_m(h, img[inside], lines)).max() < 1e-6
    assert solve_point_on_line(img[:3], lines[:3], SHAPE) is None  # under-determined


def test_support_prefers_the_right_homography_and_icp_converges():
    dist, nearest = ridge_field(line_mask(_synthetic_frame()))
    sup, n_vis = line_support(H_TRUE, dist)
    assert n_vis > 100 and sup > 0.9
    shifted = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ H_TRUE  # 3 m off
    assert line_support(shifted, dist)[0] < sup
    seed = H_TRUE @ np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -18.0], [0.0, 0.0, 1.0]])
    fit = refit_homography(seed, dist, nearest)
    assert fit.homography is not None and fit.residual_m < 0.25
    assert fit.support > 0.9 and fit.n_segments >= 4


def _table() -> pd.DataFrame:
    """Frame 0 live, frame 1 dead, four players each."""
    rows = []
    for fr in (0, 1):
        for i in range(4):
            rows.append({"frame": fr, "track_id": i, "role": "player", "team": i % 2,
                         "pitch_x": 10.0 + 5 * i if fr == 0 else np.nan,
                         "pitch_y": 20.0 + 3 * i if fr == 0 else np.nan,
                         "image_x": 100.0 + 50 * i, "image_y": 200.0 + 30 * i,
                         "conf": 1.0, "calib_error_m": 0.1, "is_actor": False, "is_keeper": False})
    return pd.DataFrame(rows)


def test_stage_is_dead_frame_only_and_every_gate_bites():
    from generator.calibrate import CalibCandidate

    h = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    df = _table()
    ok = [CalibCandidate(homography=h, error_m=0.2, n_points=9, mode="full", use_ransac=0.0,
                         rep_err_px=1.0)]
    blank = np.zeros((400, 800, 3), np.uint8)  # no grass, no lines -> support 0

    out, rec = recover_dead_frames(df, {1: blank}, candidates={1: ok}, min_support=0.0, refit=False)
    assert rec == 4
    assert np.allclose(out.loc[out.frame == 1, "pitch_x"], [10.0, 15.0, 20.0, 25.0])
    assert np.allclose(out.loc[out.frame == 0, "pitch_x"], df.loc[df.frame == 0, "pitch_x"])

    assert recover_dead_frames(df, {1: blank}, candidates={1: ok}, min_support=0.5,
                               refit=False)[1] == 0
    far = [CalibCandidate(homography=h, error_m=9.9, n_points=9, mode="full", use_ransac=0.0,
                          rep_err_px=1.0)]
    assert recover_dead_frames(df, {1: blank}, candidates={1: far}, min_support=0.0,
                               refit=False)[1] == 0
    assert recover_dead_frames(df, {1: None}, candidates={1: ok}, min_support=0.0,
                               refit=False)[1] == 0
    assert recover_dead_frames(df, {}, candidates=None, min_support=0.0, refit=True)[1] == 0
