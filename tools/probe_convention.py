"""Prove the centered-vs-uncentered world-frame bug using PnLCalib's own detected keypoints.

PnLCalib's camera params are in a CENTERED pitch frame ([-52.5,52.5] x [-34,34]); our downstream
assumes UNCENTERED [0,105] x [0,68]. For each detected ground keypoint we know its true world coord,
so we can measure which convention each homography actually produces -- no manual annotation needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.calibrate import PnLCalibCalibrator, apply_homography  # noqa: E402

VIDEO = r"C:\Users\siddh_ygv5bws\OneDrive\Desktop\cv-football\chunks\chunk_000.mp4"
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 13028
SHIFT = np.array([[1.0, 0, 52.5], [0, 1.0, 34.0], [0, 0, 1.0]])  # centered -> uncentered


def _ground_homography(cam_params: dict, *, decentre: bool) -> np.ndarray:
    """Image->pitch ground homography from cam params. ``decentre`` shifts centered->uncentered.

    Self-contained (does not call the production function) so this probe stays a valid proof
    regardless of whether the production code has been fixed yet.
    """
    q = np.array([[cam_params["x_focal_length"], 0, cam_params["principal_point"][0]],
                  [0, cam_params["y_focal_length"], cam_params["principal_point"][1]], [0, 0, 1]])
    it = np.eye(4)[:-1]
    it[:, -1] = -np.asarray(cam_params["position_meters"])
    p = q @ (np.asarray(cam_params["rotation_matrix"]) @ it)
    h_img_to_centered = np.linalg.inv(p[:, [0, 1, 3]])  # P expects CENTERED world -> image
    h = SHIFT @ h_img_to_centered if decentre else h_img_to_centered
    return h / h[2, 2]


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, frame = cap.read()
    cap.release()
    assert ok

    c = PnLCalibCalibrator()
    c._load()
    from utils.utils_calib import FramebyFrameCalib  # noqa: PLC0415

    kp_dict, lines_dict, w0, h0 = c._detect(frame)
    cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
    cam.update(kp_dict, lines_dict)
    final = cam.heuristic_voting(refine_lines=True)
    assert final is not None, "calibration failed"

    h_buggy = _ground_homography(final["cam_params"], decentre=False)  # centered (the bug)
    h_fixed = _ground_homography(final["cam_params"], decentre=True)  # de-centered (the fix)

    # PnLCalib's detected ground keypoints: image <-> world (world is CENTERED, from get_correspondences)
    cam.get_per_plane_correspondences(mode="ground_plane", use_ransac=5.0)
    obj_centered, img = cam.get_correspondences("ground_plane")  # obj in centered frame
    obj_centered, img = obj_centered[:, :2], img[:, :2]
    obj_uncentered = obj_centered + np.array([52.5, 34.0])  # the TRUE uncentered coords

    proj_buggy = apply_homography(h_buggy, img)
    proj_fixed = apply_homography(h_fixed, img)

    err_buggy_vs_uncentered = np.linalg.norm(proj_buggy - obj_uncentered, axis=1).mean()
    err_fixed_vs_uncentered = np.linalg.norm(proj_fixed - obj_uncentered, axis=1).mean()

    print(f"frame {FRAME}: {len(img)} ground keypoints\n")
    print("Does the homography output UNCENTERED [0,105]x[0,68] coords (what downstream needs)?")
    print(f"  current  production homography : mean err vs true uncentered = "
          f"{err_buggy_vs_uncentered:6.2f} m   <- off by ~(52.5, 34)")
    print(f"  fixed (de-centered) homography : mean err vs true uncentered = "
          f"{err_fixed_vs_uncentered:6.2f} m\n")

    # Concrete landmarks: show a near-goal and a far point.
    order = np.argsort(obj_uncentered[:, 0])
    for label, idx in [("left-most keypoint ", order[0]), ("right-most keypoint", order[-1])]:
        print(f"{label} @ image {img[idx].round(0)} | true uncentered = {obj_uncentered[idx].round(1)}")
        print(f"    current homography -> {proj_buggy[idx].round(1)}   (looks 'near halfway')")
        print(f"    fixed   homography -> {proj_fixed[idx].round(1)}")
    print()

    # Visual: draw the pitch model through both homographies.
    from tools.diag_calib import _CIRCLE, _PITCH_LINES, _homography_projector  # noqa: PLC0415

    out = Path("outputs/diag")
    out.mkdir(parents=True, exist_ok=True)
    for tag, h, color in [("BUGGY", h_buggy, (0, 0, 255)), ("FIXED", h_fixed, (0, 255, 0))]:
        f = frame.copy()
        proj = _homography_projector(h)
        for a, b in _PITCH_LINES:
            ia, ib = proj(np.array(a, float)), proj(np.array(b, float))
            if ia is not None and ib is not None:
                cv2.line(f, tuple(ia.astype(int)), tuple(ib.astype(int)), color, 2)
        pts = [proj(np.array(p, float)) for p in _CIRCLE]
        pts = [p for p in pts if p is not None]
        if len(pts) > 2:
            cv2.polylines(f, [np.array(pts, np.int32)], True, color, 2)
        cv2.imwrite(str(out / f"convention_{tag}_{FRAME}.png"), f)
    print(f"wrote outputs/diag/convention_BUGGY_{FRAME}.png (red) and convention_FIXED_{FRAME}.png (green)")


if __name__ == "__main__":
    main()
