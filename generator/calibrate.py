"""Pitch calibration + a per-frame confidence gate ("no wrong frames").

Two layers:

1. **The math core (pure, tested, runs anywhere).** Given image<->pitch point correspondences this
   estimates an image->pitch homography (normalised DLT, numpy only -- or ``cv2.findHomography`` with
   RANSAC when OpenCV is present) and scores it by **mean reprojection error in metres**. Frames whose
   error exceeds :data:`MAX_REPROJ_ERROR_M` are **rejected**, so the generator never emits a *wrong*
   freeze-frame -- the measurable form of the "no wrong/unclear frames" requirement.

2. **The detector seam (lazy).** :class:`PnLCalibCalibrator` plugs PnLCalib (points+lines on a 3D
   pitch model; SOTA on SoccerNet-Calibration, arXiv:2404.08401) in as the correspondence source,
   replacing ``cv-football``'s Roboflow-keypoint homography (which needs the paid ``inference`` pkg).
   PnLCalib's weights are an external download; the class imports it lazily and documents setup, while
   everything downstream depends only on the pure core above.

The pitch convention here is the 105x68 source pitch (the contract rescales to 120x80).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_REPROJ_ERROR_M = 2.0  # mean reprojection error (m) above which a frame is rejected by the gate
MIN_CORRESPONDENCES = 4  # a homography needs >= 4 point pairs

#: Environment variable pointing at the cloned PnLCalib repo (else the sibling default below).
PNLCALIB_PATH_ENV = "FOOTBALL_PNLCALIB_PATH"
_DEFAULT_PNLCALIB_ROOT = Path.home() / "PnLCalib"
# PnLCalib's detector defaults (from its inference.py).
_KP_THRESHOLD = 0.3434
_LINE_THRESHOLD = 0.7867


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of calibrating one frame.

    Args:
        homography: ``(3, 3)`` image->pitch transform (metres), or ``None`` if estimation failed.
        error_m: Mean reprojection error in metres (``inf`` if no transform).
        n_points: Number of correspondences used.
        ok: True iff a transform was found AND ``error_m <= max_error_m`` (passes the gate).
    """

    homography: np.ndarray | None
    error_m: float
    n_points: int
    ok: bool


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a ``(3, 3)`` homography to ``(N, 2)`` points; returns ``(N, 2)`` (divides by w)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = homog @ np.asarray(h, dtype=float).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return proj[:, :2] / w


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalisation: returns ``(T, pts_n)`` with ``pts_n = (T @ [x,y,1]^T)``."""
    mean = pts.mean(axis=0)
    centred = pts - mean
    mean_dist = float(np.sqrt((centred**2).sum(axis=1)).mean())
    scale = np.sqrt(2) / mean_dist if mean_dist > 1e-12 else 1.0
    t = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])
    homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return t, (homog @ t.T)[:, :2]


def estimate_homography(
    image_pts: np.ndarray, pitch_pts: np.ndarray, *, use_cv2: bool = True
) -> np.ndarray | None:
    """Estimate an image->pitch homography from correspondences.

    Uses ``cv2.findHomography`` (RANSAC) when OpenCV is importable and ``use_cv2`` is set, else a pure
    normalised-DLT solver. Returns the ``(3, 3)`` matrix, or ``None`` if there are too few points or
    the system is degenerate.
    """
    image_pts = np.asarray(image_pts, dtype=float).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=float).reshape(-1, 2)
    if len(image_pts) < MIN_CORRESPONDENCES or len(image_pts) != len(pitch_pts):
        return None

    if use_cv2:
        try:
            import cv2  # noqa: PLC0415 - optional; pure DLT is the fallback

            h, _ = cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC, 5.0)
            return None if h is None else np.asarray(h, dtype=float)
        except Exception:  # noqa: BLE001 - fall through to the pure solver
            pass

    t_src, src = _normalise(image_pts)
    t_dst, dst = _normalise(pitch_pts)
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        rows.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) < 1e-12:
        return None
    return h / h[2, 2]


def reprojection_error_m(image_pts: np.ndarray, pitch_pts: np.ndarray, h: np.ndarray) -> float:
    """Mean reprojection distance (metres) between reprojected image points and known pitch points."""
    proj = apply_homography(h, image_pts)
    pitch = np.asarray(pitch_pts, dtype=float).reshape(-1, 2)
    return float(np.linalg.norm(proj - pitch, axis=1).mean())


def calibrate_correspondences(
    image_pts: np.ndarray,
    pitch_pts: np.ndarray,
    *,
    max_error_m: float = MAX_REPROJ_ERROR_M,
    use_cv2: bool = True,
) -> CalibrationResult:
    """Estimate a homography from correspondences and apply the confidence gate.

    This is the pure, testable heart of "no wrong frames": it always returns a
    :class:`CalibrationResult` whose ``ok`` flag tells the generator whether to trust the frame.
    """
    n = int(min(len(np.asarray(image_pts).reshape(-1, 2)), len(np.asarray(pitch_pts).reshape(-1, 2))))
    h = estimate_homography(image_pts, pitch_pts, use_cv2=use_cv2)
    if h is None:
        return CalibrationResult(homography=None, error_m=float("inf"), n_points=n, ok=False)
    err = reprojection_error_m(image_pts, pitch_pts, h)
    return CalibrationResult(homography=h, error_m=err, n_points=n, ok=err <= max_error_m)


# source pitch axes (uncentred [0,105]x[0,68]); the contract rescales to 120x80
from core.pitch import PITCH_LEN as PITCH_LENGTH_M, PITCH_WID as PITCH_WIDTH_M


def ground_homography_from_cam_params(cam_params: dict) -> np.ndarray | None:
    """Build an image -> pitch(105x68, **uncentred**) homography from PnLCalib camera parameters.

    PnLCalib returns a full 3D camera calibration (focal lengths, principal point, rotation,
    position) -- far more robust on broadcast frames than a planar homography fit to the (often
    band-clustered) ground keypoints. Its 3x4 projection ``P = Q @ (R @ [I | -position])`` maps world
    points ``[X, Y, Z, 1]`` to the image, but in PnLCalib's **centred** pitch frame
    (``[-52.5, 52.5] x [-34, 34]``, origin at the centre spot -- its ``keypoint_world_coords_2D`` are
    re-centred by ``[x - 52.5, y - 34]``). Restricting to the ground plane ``Z = 0`` makes columns
    ``[0, 1, 3]`` a centred-world->image homography; we invert it (image->centred metres) and shift
    the origin to the corner, yielding the project-wide **uncentred** ``[0,105] x [0,68]`` convention
    that the off-pitch gate, post-processing and contract all assume. (Skipping that shift offsets
    every player by ~(52.5, 34) m -- the centre spot lands at a corner, players cram into one half,
    and a keeper at the far goal plots at "halfway": the symptom this fixes.)

    Returns ``None`` if the projection is degenerate (non-invertible).
    """
    q = np.array(
        [
            [cam_params["x_focal_length"], 0, cam_params["principal_point"][0]],
            [0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
            [0, 0, 1],
        ]
    )
    it = np.eye(4)[:-1]
    it[:, -1] = -np.asarray(cam_params["position_meters"])
    p = q @ (np.asarray(cam_params["rotation_matrix"]) @ it)  # 3x4 centred-world -> image
    h_world_to_img = p[:, [0, 1, 3]]  # ground plane Z=0: centred [-52.5,52.5]x[-34,34] -> image
    try:
        h_img_to_centred = np.linalg.inv(h_world_to_img)  # image -> centred pitch metres
    except np.linalg.LinAlgError:
        return None
    # Shift the centre-spot origin to the corner -> uncentred [0,105] x [0,68] (project convention).
    to_uncentred = np.array(
        [[1.0, 0.0, PITCH_LENGTH_M / 2], [0.0, 1.0, PITCH_WIDTH_M / 2], [0.0, 0.0, 1.0]]
    )
    h_img_to_world = to_uncentred @ h_img_to_centred
    if abs(h_img_to_world[2, 2]) < 1e-12:
        return None
    return h_img_to_world / h_img_to_world[2, 2]


def pnlcalib_root() -> Path:
    """Resolve the cloned PnLCalib repo (``$FOOTBALL_PNLCALIB_PATH`` or ``~/PnLCalib``)."""
    root = Path(os.environ.get(PNLCALIB_PATH_ENV, _DEFAULT_PNLCALIB_ROOT)).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"PnLCalib not found at {root}. Clone github.com/mguti97/PnLCalib and download its "
            f"SV_kp/SV_lines weights, or set ${PNLCALIB_PATH_ENV}."
        )
    return root


class PnLCalibCalibrator:
    """PnLCalib-backed per-frame calibrator: SOTA pitch keypoint/line detection -> homography + gate.

    Uses PnLCalib (arXiv:2404.08401) purely as the *detector*: two HRNet models produce keypoint and
    line heatmaps, ``FramebyFrameCalib`` matches them to the 3D pitch model and yields image<->pitch
    correspondences, and those are handed to the project's own tested :func:`calibrate_correspondences`
    so the homography (image->pitch, metres on a 105x68 pitch) and the **reprojection-error confidence
    gate** stay consistent with the rest of the generator.

    Heavy and lazy: the HRNet weights (~265 MB each) and torch/torchvision are loaded on first use.
    Clone https://github.com/mguti97/PnLCalib (set ``$FOOTBALL_PNLCALIB_PATH``) and place the
    ``SV_kp`` / ``SV_lines`` release weights under ``<repo>/weights`` (or pass explicit paths).
    """

    def __init__(
        self,
        *,
        weights_kp: str | Path | None = None,
        weights_line: str | Path | None = None,
        device: str | None = None,
        max_error_m: float = MAX_REPROJ_ERROR_M,
        kp_threshold: float = _KP_THRESHOLD,
        line_threshold: float = _LINE_THRESHOLD,
    ):
        #: ``None`` -> auto-detect CUDA at load time (falls back to CPU).
        self.device = device
        self.max_error_m = max_error_m
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self._weights_kp = weights_kp
        self._weights_line = weights_line
        self._loaded = False

    def _load(self) -> None:
        """Inject the PnLCalib repo onto ``sys.path`` and build both HRNet models (once)."""
        if self._loaded:
            return
        import yaml  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchvision.transforms as T  # noqa: PLC0415

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        root = pnlcalib_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from model.cls_hrnet import get_cls_net  # noqa: PLC0415
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l  # noqa: PLC0415

        wk = Path(self._weights_kp or root / "weights" / "SV_kp")
        wl = Path(self._weights_line or root / "weights" / "SV_lines")
        for w in (wk, wl):
            if not w.exists():
                raise FileNotFoundError(f"PnLCalib weights missing: {w}")
        cfg = yaml.safe_load(open(root / "config" / "hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(root / "config" / "hrnetv2_w48_l.yaml"))
        self._model = get_cls_net(cfg)
        self._model.load_state_dict(torch.load(wk, map_location=self.device))
        self._model.to(self.device).eval()
        self._model_l = get_cls_net_l(cfg_l)
        self._model_l.load_state_dict(torch.load(wl, map_location=self.device))
        self._model_l.to(self.device).eval()
        self._resize = T.Resize((540, 960))
        self._loaded = True

    def _detect(self, frame_bgr: np.ndarray):
        """Run the HRNet models on one BGR frame -> ``(kp_dict, lines_dict, width, height)``."""
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchvision.transforms.functional as fT  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from utils.utils_heatmap import (  # noqa: PLC0415
            coords_to_dict,
            complete_keypoints,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = fT.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
        _, _, h0, w0 = t.size()
        t = t if t.size()[-1] == 960 else self._resize(t)
        t = t.to(self.device)
        _, _, h, w = t.size()
        with torch.no_grad():
            heatmaps = self._model(t)
            heatmaps_l = self._model_l(t)
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
        return kp_dict, lines_dict, w0, h0

    def calibrate_frame(self, frame_bgr: np.ndarray) -> CalibrationResult:
        """Detect pitch points/lines on a BGR frame and calibrate via PnLCalib's full camera model.

        Uses PnLCalib's ``heuristic_voting`` (points + lines + PnL refinement, the method's real
        strength) to get a 3D camera calibration, then derives the image->pitch ground homography
        from it -- robust on broadcast frames where the visible keypoints cluster in a band and a
        plain planar homography (the old approach) extrapolates players to nonsense positions. The
        gate uses the **metre** reprojection error of the detected keypoints through that homography.

        Returns ``ok=False`` when PnLCalib cannot calibrate the frame or the error exceeds the gate.
        """
        self._load()
        from utils.utils_calib import FramebyFrameCalib  # noqa: PLC0415

        kp_dict, lines_dict, w0, h0 = self._detect(frame_bgr)
        cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
        cam.update(kp_dict, lines_dict)

        final = cam.heuristic_voting(refine_lines=True)
        if final is None:
            return CalibrationResult(homography=None, error_m=float("inf"), n_points=0, ok=False)
        h = ground_homography_from_cam_params(final["cam_params"])
        if h is None:
            return CalibrationResult(homography=None, error_m=float("inf"), n_points=0, ok=False)

        # Metre reprojection error of the ground keypoints through the derived homography. NB:
        # ``get_correspondences`` returns PnLCalib's **centred** world coords, while ``h`` outputs the
        # **uncentred** convention -- shift the obj points to match, else the error reads as the whole
        # ~62 m centre offset and every frame is (wrongly) rejected.
        cam.get_per_plane_correspondences(mode="ground_plane", use_ransac=5.0)
        n_pts, err = 0, float("inf")
        if cam.obj_pts:
            obj_pts, img_pts = cam.get_correspondences("ground_plane")
            n_pts = len(obj_pts)
            if n_pts >= MIN_CORRESPONDENCES:
                obj_uncentred = obj_pts[:, :2] + np.array([PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2])
                err = reprojection_error_m(img_pts[:, :2], obj_uncentred, h)
        return CalibrationResult(homography=h, error_m=err, n_points=n_pts, ok=err <= self.max_error_m)
