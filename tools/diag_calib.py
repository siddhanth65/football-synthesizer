"""Diagnose calibration by drawing each method's pitch model back onto the real frame.

If a method's reconstructed pitch lines sit on the actual painted lines in the video, its calibration
is correct. Compares: (A) our current homography (correspondences -> cv2.findHomography), (B)
PnLCalib's own ground-plane homography (its RANSAC + line refinement), (C) PnLCalib's full camera
model (heuristic_voting -> 3x4 projection). Also reports keypoint count + spread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.calibrate import PnLCalibCalibrator, estimate_homography  # noqa: E402

VIDEO = r"C:\Users\siddh_ygv5bws\OneDrive\Desktop\cv-football\chunks\chunk_000.mp4"

# Pitch model lines in metres on a 105x68 pitch (uncentered), z=0 ground plane.
_PITCH_LINES = [
    [(0, 0), (105, 0)], [(0, 68), (105, 68)], [(0, 0), (0, 68)], [(105, 0), (105, 68)],
    [(52.5, 0), (52.5, 68)],
    [(0, 13.84), (16.5, 13.84)], [(16.5, 13.84), (16.5, 54.16)], [(16.5, 54.16), (0, 54.16)],
    [(105, 13.84), (88.5, 13.84)], [(88.5, 13.84), (88.5, 54.16)], [(88.5, 54.16), (105, 54.16)],
]
_CIRCLE = [(52.5 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 40)]


def _draw_lines_via_pitch_to_image(frame, project_fn, color):
    """Project each pitch line (metres) to image via project_fn and draw it."""
    for (a, b) in _PITCH_LINES:
        ia, ib = project_fn(np.array(a, float)), project_fn(np.array(b, float))
        if ia is None or ib is None:
            continue
        cv2.line(frame, tuple(ia.astype(int)), tuple(ib.astype(int)), color, 2)
    pts = [project_fn(np.array(p, float)) for p in _CIRCLE]
    pts = [p for p in pts if p is not None]
    if len(pts) > 2:
        cv2.polylines(frame, [np.array(pts, np.int32)], True, color, 2)


def _homography_projector(h_img2pitch):
    """Return pitch(m)->image projector using the inverse of an image->pitch homography."""
    hinv = np.linalg.inv(h_img2pitch)

    def proj(xy):
        v = hinv @ np.array([xy[0], xy[1], 1.0])
        if abs(v[2]) < 1e-9:
            return None
        out = v[:2] / v[2]
        return out if np.all(np.isfinite(out)) and np.all(np.abs(out) < 1e5) else None

    return proj


def _camera_projector(P):
    """Return pitch(m)->image projector for PnLCalib's 3x4 P (expects CENTERED world coords)."""

    def proj(xy):
        world = np.array([xy[0] - 105 / 2, xy[1] - 68 / 2, 0.0, 1.0])
        v = P @ world
        if abs(v[2]) < 1e-9:
            return None
        out = v[:2] / v[2]
        return out if np.all(np.isfinite(out)) and np.all(np.abs(out) < 1e5) else None

    return proj


def main() -> None:
    FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 13028
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read frame"

    c = PnLCalibCalibrator()
    c._load()
    kp_dict, lines_dict, w0, h0 = c._detect(frame)
    n_kp = len([k for k, v in kp_dict.items() if isinstance(v, dict)])
    print(f"frame {FRAME}: image {w0}x{h0}, detected keypoints={n_kp}, lines={len(lines_dict)}")

    from utils.utils_calib import FramebyFrameCalib  # noqa: PLC0415

    cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
    cam.update(kp_dict, lines_dict)
    cam.get_per_plane_correspondences(mode="ground_plane", use_ransac=5.0)
    obj, img = cam.get_correspondences("ground_plane")
    if len(img):
        print(f"correspondences={len(img)} | image x-span={np.ptp(img[:,0]):.0f}px "
              f"y-span={np.ptp(img[:,1]):.0f}px | world x-span={np.ptp(obj[:,0]):.0f}m "
              f"y-span={np.ptp(obj[:,1]):.0f}m")

    out = Path("outputs/diag")
    out.mkdir(parents=True, exist_ok=True)

    # (A) our homography
    if len(img) >= 4:
        h = estimate_homography(img[:, :2], obj[:, :2])
        if h is not None:
            fa = frame.copy()
            _draw_lines_via_pitch_to_image(fa, _homography_projector(h), (0, 0, 255))
            cv2.imwrite(str(out / f"A_ours_{FRAME}.png"), fa)
            print("wrote A_ours (red lines = our homography's pitch)")

    # (B) PnLCalib ground homography
    res = cam.heuristic_voting_ground(refine_lines=True)
    if res is not None:
        h_inv_img2pitch = res["homography"]  # image->pitch (inverse=True in their call)
        fb = frame.copy()
        _draw_lines_via_pitch_to_image(fb, _homography_projector(h_inv_img2pitch), (0, 255, 0))
        cv2.imwrite(str(out / f"B_pnl_ground_{FRAME}.png"), fb)
        print(f"wrote B_pnl_ground (green) | rep_err={res['rep_err']:.1f}px")

    # (C) PnLCalib full camera model
    final = cam.heuristic_voting(refine_lines=True)
    if final is not None:
        cam_params = final["cam_params"]
        Q = np.array([[cam_params["x_focal_length"], 0, cam_params["principal_point"][0]],
                      [0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
                      [0, 0, 1]])
        It = np.eye(4)[:-1]
        It[:, -1] = -np.array(cam_params["position_meters"])
        P = Q @ (np.array(cam_params["rotation_matrix"]) @ It)
        fc = frame.copy()
        _draw_lines_via_pitch_to_image(fc, _camera_projector(P), (255, 0, 0))
        cv2.imwrite(str(out / f"C_pnl_camera_{FRAME}.png"), fc)
        print(f"wrote C_pnl_camera (blue) | mode={final['mode']} rep_err={final['rep_err']:.1f}")


if __name__ == "__main__":
    main()
