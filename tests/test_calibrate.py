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
    CalibCandidate,
    PnLCalibCalibrator,
    apply_homography,
    calibrate_correspondences,
    estimate_homography,
    ground_homography_from_cam_params,
    pnlcalib_root,
    reprojection_error_m,
    select_calibration,
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


# --- the on-pitch plausibility half of the gate --------------------------------------------------
_GOOD_H = np.array([[0.06, 0.004, -8.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])
_OFF_H = _GOOD_H + np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # x >> 105 m
_FOOT = np.column_stack([np.linspace(300, 1500, 12), 400 + 20 * (np.arange(12) % 5)])


def test_onpitch_plausible_separates_a_gate_passing_off_pitch_homography():
    from generator.postprocess import onpitch_plausible

    assert onpitch_plausible(_GOOD_H, _FOOT)
    assert not onpitch_plausible(_OFF_H, _FOOT)
    assert not onpitch_plausible(_GOOD_H, _FOOT[:2])  # too few players to judge
    # A zoomed frame showing 4 players over ~12 m is trusted (the wide-shot rule voided it).
    assert onpitch_plausible(_GOOD_H, _FOOT[:4])
    assert not onpitch_plausible(_GOOD_H, _FOOT[:4], min_onpitch=8, min_span_m=25.0)


def test_select_calibration_is_unchanged_without_foot_points():
    """The regression guard: no foot points -> exactly PnLCalib's own pick, error gate only."""
    cands = [CalibCandidate(_OFF_H, 0.36, 12, "full", 0.0, 1.0),
             CalibCandidate(_GOOD_H, 1.10, 12, "main", 5.0, 3.0)]
    res = select_calibration(cands)
    assert res.ok and np.allclose(res.homography, _OFF_H) and res.error_m == 0.36


def test_select_calibration_falls_through_to_a_plausible_candidate():
    cands = [CalibCandidate(_OFF_H, 0.36, 12, "full", 0.0, 1.0),
             CalibCandidate(_GOOD_H, 1.10, 12, "main", 5.0, 3.0)]
    res = select_calibration(cands, foot_points=_FOOT)
    assert res.ok and np.allclose(res.homography, _GOOD_H)


def test_select_calibration_rejects_when_nothing_is_plausible():
    """No plausible hypothesis -> ok=False, so the caller falls back (temporal fill), not garbage."""
    res = select_calibration([CalibCandidate(_OFF_H, 0.36, 12, "full", 0.0, 1.0)],
                             foot_points=_FOOT)
    assert not res.ok
    assert not select_calibration([], foot_points=_FOOT).ok
    # Plausible but over the error gate is still a reject.
    assert not select_calibration([CalibCandidate(_GOOD_H, 9.9, 12, "full", 0.0, 1.0)],
                                  foot_points=_FOOT).ok


# --- v10-W2: sub-cell heatmap decoding -----------------------------------------------------------
def test_refine_peaks_recovers_a_known_subcell_gaussian():
    """A Gaussian centred off-cell must be decoded to its true centre, not the arg-max cell."""
    torch = pytest.importorskip("torch")
    from generator.calibrate import HEATMAP_SIGMA, HEATMAP_SCALE, refine_peaks

    h = w = 40
    cy, cx = 20.3, 15.7  # true sub-cell centre
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    hm = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * HEATMAP_SIGMA**2))
    hm_t = torch.tensor(hm, dtype=torch.float32).view(1, 1, h, w)
    peak = np.unravel_index(int(hm.argmax()), hm.shape)
    stock = torch.tensor([[[[float(peak[1] * HEATMAP_SCALE), float(peak[0] * HEATMAP_SCALE), 1.0]]]])
    out = refine_peaks(stock, hm_t).numpy()[0, 0, 0]
    assert abs(stock.numpy()[0, 0, 0][0] / HEATMAP_SCALE - cx) >= 0.25, "argmax must be quantised"
    assert abs(out[0] / HEATMAP_SCALE - cx) < 0.005, out
    assert abs(out[1] / HEATMAP_SCALE - cy) < 0.005, out
    assert out[2] == 1.0, "the score column is untouched"


def test_refine_peaks_keeps_the_argmax_on_a_degenerate_neighbourhood():
    torch = pytest.importorskip("torch")
    from generator.calibrate import refine_peaks

    coords = torch.tensor([[[[4.0, 6.0, 0.0]]]])
    assert torch.equal(refine_peaks(coords, torch.zeros(1, 1, 8, 8)), coords), "flat -> no shift"
    assert torch.equal(refine_peaks(coords, -torch.ones(1, 1, 8, 8)), coords), "negative -> no shift"


def test_calibrator_decode_flags_default_to_stock():
    c = PnLCalibCalibrator(device="cpu")
    assert c.subpix_decode is False and c.derived_kp_scale == 1


def test_shift_half_cell_moves_to_the_cell_centre():
    """PnLCalib floors labels into cells, so the decoded corner needs half a cell back."""
    torch = pytest.importorskip("torch")
    from generator.calibrate import HEATMAP_SCALE, shift_half_cell

    coords = torch.tensor([[[[40.0, 24.0, 0.75]]]])
    got = shift_half_cell(coords).numpy()[0, 0, 0]
    assert got.tolist() == [40.0 + HEATMAP_SCALE / 2, 24.0 + HEATMAP_SCALE / 2, 0.75], got
