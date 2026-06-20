"""Tests for the calibration math core + the per-frame confidence gate.

All pure (numpy DLT path forced via ``use_cv2=False``) so they run anywhere and don't depend on the
RANSAC behaviour of whichever OpenCV is installed.
"""

from __future__ import annotations

import numpy as np

import pytest

from generator.calibrate import (
    MAX_REPROJ_ERROR_M,
    PNLCALIB_PATH_ENV,
    PnLCalibCalibrator,
    apply_homography,
    calibrate_correspondences,
    estimate_homography,
    ground_homography_from_cam_params,
    pnlcalib_root,
    reprojection_error_m,
)


def _lookat_camera(position, target=(52.5, 34.0, 0.0), focal=1200.0, pp=(512.0, 288.0)):
    """A synthetic look-at camera as PnLCalib-style ``cam_params`` (+ its 3x4 P)."""
    p = np.asarray(position, float)
    f = np.asarray(target, float) - p
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    rot = np.array([r, -u, f])  # image x=right, y=down, z=forward
    cam_params = {
        "x_focal_length": focal, "y_focal_length": focal,
        "principal_point": list(pp), "position_meters": p.tolist(),
        "rotation_matrix": rot.tolist(),
    }
    q = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]])
    it = np.eye(4)[:-1]
    it[:, -1] = -p
    return cam_params, q @ (rot @ it)


def test_ground_homography_maps_image_to_uncentred_pitch():
    """Lock the convention: PnLCalib's cam params live in a frame CENTRED on the centre spot, and the
    derived homography must output the project-wide UNCENTRED ``[0,105] x [0,68]`` coords.

    Build a synthetic look-at camera in the centred frame (origin = centre spot), project centred
    ground points to the image, and assert the homography recovers ``centred + (52.5, 34)``. If the
    de-centring shift is dropped (the real bug that put keepers at "halfway"), this fails.
    """
    cam_params, p_mat = _lookat_camera(position=(0.0, 96.0, 35.0), target=(0.0, 0.0, 0.0))
    centred = np.array([[-42.5, -14.0], [37.5, 16.0], [-12.5, 26.0], [17.5, -22.0], [0.0, 0.0]])
    img = []
    for x, y in centred:
        v = p_mat @ np.array([x, y, 0.0, 1.0])
        img.append(v[:2] / v[2])
    img = np.array(img)
    h = ground_homography_from_cam_params(cam_params)
    assert h is not None
    uncentred = centred + np.array([52.5, 34.0])  # centre spot -> (52.5, 34), a corner -> (0/105, 0/68)
    np.testing.assert_allclose(apply_homography(h, img), uncentred, atol=1e-6)


def test_ground_homography_none_when_degenerate():
    cam_params, _ = _lookat_camera(position=(52.5, 130.0, 35.0))
    cam_params["x_focal_length"] = 0.0  # collapses the projection
    cam_params["y_focal_length"] = 0.0
    assert ground_homography_from_cam_params(cam_params) is None

# A plausible image->pitch homography (perspective from a broadcast camera to top-down metres).
_H_TRUE = np.array(
    [
        [0.05, 0.002, -3.0],
        [0.001, 0.06, -2.0],
        [0.0005, 0.0008, 1.0],
    ]
)

# Spread image points across a frame; their pitch coords come from the true homography.
_IMG = np.array([[100.0, 80.0], [1180.0, 90.0], [1170.0, 700.0], [120.0, 690.0],
                 [640.0, 360.0], [300.0, 500.0]])
_PITCH = apply_homography(_H_TRUE, _IMG)


def test_estimate_recovers_known_homography():
    h = estimate_homography(_IMG, _PITCH, use_cv2=False)
    assert h is not None
    # Homographies are scale-free; compare their action on fresh points.
    probe = np.array([[500.0, 400.0], [900.0, 200.0]])
    np.testing.assert_allclose(apply_homography(h, probe), apply_homography(_H_TRUE, probe), atol=1e-6)


def test_reprojection_error_near_zero_for_exact_fit():
    h = estimate_homography(_IMG, _PITCH, use_cv2=False)
    assert reprojection_error_m(_IMG, _PITCH, h) < 1e-6


def test_gate_accepts_clean_correspondences():
    res = calibrate_correspondences(_IMG, _PITCH, use_cv2=False)
    assert res.ok and res.homography is not None
    assert res.error_m < MAX_REPROJ_ERROR_M and res.n_points == len(_IMG)


def test_gate_rejects_noisy_correspondences():
    rng = np.random.default_rng(1)
    # Add several metres of pitch noise -> the best-fit homography can't reconcile it -> reject.
    noisy_pitch = _PITCH + rng.normal(0, 8.0, _PITCH.shape)
    res = calibrate_correspondences(_IMG, noisy_pitch, use_cv2=False, max_error_m=MAX_REPROJ_ERROR_M)
    assert not res.ok
    assert res.error_m > MAX_REPROJ_ERROR_M


def test_too_few_points_is_not_ok():
    res = calibrate_correspondences(_IMG[:3], _PITCH[:3], use_cv2=False)
    assert not res.ok and res.homography is None and np.isinf(res.error_m)


# --- PnLCalib seam (config is pure; the heavy model run is opt-in) -------------------------------
def test_pnlcalib_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv(PNLCALIB_PATH_ENV, str(tmp_path))
    assert pnlcalib_root() == tmp_path.resolve()


def test_pnlcalib_root_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(PNLCALIB_PATH_ENV, str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="PnLCalib"):
        pnlcalib_root()


def test_pnlcalib_calibrator_constructs_lazily(monkeypatch, tmp_path):
    """Constructing the calibrator must not load weights (so import/CI stays cheap)."""
    monkeypatch.setenv(PNLCALIB_PATH_ENV, str(tmp_path / "nope"))
    c = PnLCalibCalibrator(device="cpu", max_error_m=1.5)
    assert c._loaded is False and c.max_error_m == 1.5
