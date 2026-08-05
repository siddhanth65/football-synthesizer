"""Video -> positions table (detection + tracking + calibration + team/role), the CV core.

Reproduce-and-adapt of ``cv-football/extract_positions.py`` and the ``sn-gamestate``/TrackLab approach,
restructured so the project's own, tested modules do the trustworthy work:

- **Calibration + confidence gate** -> :mod:`generator.calibrate` (PnLCalib seam + reprojection-error
  gate, replacing the Roboflow-keypoint homography).
- **Smoothing + actor/keeper derivation** -> :mod:`generator.postprocess` (pure, tested).
- **Schema** -> the positions table :data:`POSITIONS_COLUMNS` consumed by :mod:`generator.to_frames`.

Heavy CV (OpenCV / Ultralytics / supervision / the ``sports`` helpers) is imported **lazily** inside
:func:`extract_positions`, so this module imports -- and its pure seams are tested -- without the
``[cv]`` extra. Detection uses a Roboflow football model when ``inference`` + a key are present, else a
COCO ``yolov8s`` fallback (person + sports-ball).

Usage (run under an interpreter with the ``[cv]`` stack)::

    python -m generator.extract --source clip.mp4 --out positions/clip.parquet --sample-every 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from generator.calibrate import MAX_REPROJ_ERROR_M, CalibrationResult, apply_homography
from generator.postprocess import PLAYER_ROLES, SRC_LEN, SRC_WID, clamp_to_pitch, postprocess
from generator.teams import JerseyColorTeamClassifier

logger = logging.getLogger(__name__)

#: Columns of the positions parquet (the seam to :mod:`generator.to_frames`).
POSITIONS_COLUMNS = (
    "frame", "track_id", "role", "team", "pitch_x", "pitch_y",
    "image_x", "image_y", "conf", "calib_error_m", "is_actor", "is_keeper",
)

DETECT_CONF = 0.20  # player detection confidence
BALL_CONF = 0.10  # the ball is small/fast -> a lower threshold recovers more ball frames

# Canonical per-detection role ids returned by every detector (ball is returned separately). COCO/
# RF-DETR have no role classes so they emit ROLE_PLAYER for everyone; a football-trained detector maps
# its goalkeeper/referee classes onto these so the role flows through tracking untouched.
ROLE_PLAYER, ROLE_GOALKEEPER, ROLE_REFEREE = 0, 1, 2
ROLE_NAME = {ROLE_PLAYER: "player", ROLE_GOALKEEPER: "goalkeeper", ROLE_REFEREE: "referee"}
ROLE_ID = {"player": ROLE_PLAYER, "goalkeeper": ROLE_GOALKEEPER, "referee": ROLE_REFEREE}
# BoT-SORT config tuned for broadcast football (GMC + ReID + strict track creation); see the yaml.
_BOTSORT_CFG = str(Path(__file__).with_name("botsort_tuned.yaml"))


def _norm_role_name(class_name: str) -> str | None:
    """Map a detector's class name to a canonical role (or ``None`` for non-football classes)."""
    n = class_name.lower()
    if "ball" in n:
        return "ball"
    if "goal" in n or n == "gk":
        return "goalkeeper"
    if "ref" in n:
        return "referee"
    if "player" in n or n == "person":
        return "player"
    return None


def _split_tracked_yolo(r, role_of):
    """Map an Ultralytics tracked result to ``(xyxy, conf, role_ids, track_ids, ball_xy)``.

    ``role_of(class_id) -> role str`` ("player"/"goalkeeper"/"referee"/"ball"/None). The ball is the
    top-confidence ball box (no track id); people are the player/GK/referee boxes above DETECT_CONF
    carrying BoT-SORT's track ids.
    """
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int), None
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.int().cpu().tolist()
    ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else np.full(len(cls), -1)
    roles = np.array([role_of(c) for c in cls], object)

    ball_xy = None
    ball_sel = (roles == "ball") & (conf >= BALL_CONF)
    if ball_sel.any():
        b = xyxy[ball_sel][int(np.argmax(conf[ball_sel]))]
        ball_xy = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    person = np.isin(roles, ["player", "goalkeeper", "referee"]) & (conf >= DETECT_CONF)
    role_ids = np.array([ROLE_ID[x] for x in roles[person]], int)
    return xyxy[person], conf[person], role_ids, np.asarray(ids)[person].astype(int), ball_xy


# === Pure seams (tested without the CV stack) ====================================================
def transform_and_gate(
    image_pts: np.ndarray,
    calib: CalibrationResult | None,
    *,
    src_len: float = SRC_LEN,
    src_wid: float = SRC_WID,
) -> np.ndarray:
    """Project image foot-points to pitch metres, honouring the confidence gate.

    If ``calib`` is missing, has no homography, or **failed the gate** (``not calib.ok``), every point
    is returned as ``(nan, nan)`` -- a rejected frame yields no (wrong) pitch coordinates. Otherwise
    points are projected and clamped to the pitch (wildly off-pitch points -> NaN).

    Args:
        image_pts: ``(N, 2)`` image-space foot points.
        calib: The frame's :class:`~generator.calibrate.CalibrationResult`, or ``None``.

    Returns:
        ``(N, 2)`` pitch coordinates (metres) with NaNs for untrustworthy/out-of-bounds points.
    """
    image_pts = np.asarray(image_pts, dtype=float).reshape(-1, 2)
    if calib is None or calib.homography is None or not calib.ok:
        return np.full((len(image_pts), 2), np.nan)
    projected = apply_homography(calib.homography, image_pts)
    out = np.empty_like(projected)
    for i, (px, py) in enumerate(projected):
        out[i] = clamp_to_pitch(px, py, src_len=src_len, src_wid=src_wid)
    return out


def build_player_rows(
    frame_idx: int,
    track_ids: np.ndarray,
    teams: np.ndarray,
    foot_xy: np.ndarray,
    pitch_xy: np.ndarray,
    confs: np.ndarray,
    calib_error_m: float,
    roles: np.ndarray | None = None,
) -> list[dict]:
    """Assemble per-player positions rows for one frame (pure; NaNs preserved).

    ``roles`` (optional) is a per-detection role string (``player`` / ``goalkeeper`` / ``referee``);
    when omitted every row is a plain ``player`` (the COCO path, which has no role classes).
    """
    rows = []
    for i in range(len(track_ids)):
        rows.append(
            {
                "frame": int(frame_idx),
                "track_id": int(track_ids[i]),
                "role": "player" if roles is None else str(roles[i]),
                "team": int(teams[i]),
                "pitch_x": float(pitch_xy[i, 0]),
                "pitch_y": float(pitch_xy[i, 1]),
                "image_x": float(foot_xy[i, 0]),
                "image_y": float(foot_xy[i, 1]),
                "conf": float(confs[i]),
                "calib_error_m": float(calib_error_m),
            }
        )
    return rows


def build_ball_row(frame_idx: int, ball_image_xy, ball_pitch_xy, calib_error_m: float) -> dict:
    """Assemble the single ball row for one frame (pure)."""
    return {
        "frame": int(frame_idx),
        "track_id": -1,
        "role": "ball",
        "team": -1,
        "pitch_x": float(ball_pitch_xy[0]),
        "pitch_y": float(ball_pitch_xy[1]),
        "image_x": float(ball_image_xy[0]),
        "image_y": float(ball_image_xy[1]),
        "conf": 1.0,
        "calib_error_m": float(calib_error_m),
    }


def empty_positions() -> pd.DataFrame:
    """An empty positions frame with the full schema (written when a clip yields nothing)."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in POSITIONS_COLUMNS})


# === CV orchestration (lazy heavy imports) =======================================================
def _build_detector(device: str, name: str = "yolo", *, weights: str | None = None):
    """Construct the player/ball detector. ``name`` in {``football``, ``yolo``, ``rfdetr``}. Lazy.

    - ``football``: a football-trained YOLO with real ball/goalkeeper/player/referee classes (the
      ``tools/benchmark_detectors.py`` winner). The right choice for tactical work -- gives GK and
      referee roles instead of one undifferentiated ``person``. ``weights`` overrides the default.
    - ``yolo``: COCO ``yolov8s`` (person + sports-ball). Fast, dependency-light; no role classes.
    - ``rfdetr``: RF-DETR (DINOv2 transformer, NMS-free; SOTA on COCO, ICLR 2026). COCO-pretrained.
    """
    if name == "football":
        return _FootballRoleDetector(weights=weights, device=device)
    if name == "rfdetr":
        return _RFDetrDetector(device=device)
    from ultralytics import YOLO  # noqa: PLC0415

    yolo = YOLO(weights or "yolov8s.pt")
    yolo.to(device)
    return _CocoDetector(yolo)


class _CocoDetector:
    """COCO ``yolov8s`` detector: person -> players, 'sports ball' -> ball. The dependency-free path.

    No role classes, so every person is a plain ``player`` and the GK flag falls back to the
    positional heuristic in :mod:`generator.postprocess`. For real ball/GK/player/referee roles use
    :class:`_FootballRoleDetector` (``--detector football``), the benchmark winner.
    """

    PERSON, BALL = 0, 32

    def __init__(self, yolo):
        self.yolo = yolo

    def detect(self, frame_rgb):
        """Return ``(player_xyxy, player_conf, role_ids, ball_xy_or_None)`` (roles all ``player``)."""
        r = self.yolo(frame_rgb, verbose=False, conf=BALL_CONF, classes=[self.PERSON, self.BALL])[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        ppl = (cls == self.PERSON) & (conf >= DETECT_CONF)
        ball_sel = (cls == self.BALL) & (conf >= BALL_CONF)
        ball_xy = None
        if ball_sel.any():
            b = xyxy[ball_sel][int(np.argmax(conf[ball_sel]))]
            ball_xy = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        return xyxy[ppl], conf[ppl], np.full(int(ppl.sum()), ROLE_PLAYER), ball_xy

    def track(self, frame_rgb):
        """Detect + BoT-SORT (GMC) in one call -> ``(xyxy, conf, role_ids, track_ids, ball_xy)``."""
        r = self.yolo.track(frame_rgb, persist=True, tracker=_BOTSORT_CFG, verbose=False,
                            conf=BALL_CONF, classes=[self.PERSON, self.BALL])[0]

        def role_of(c):
            return "player" if c == self.PERSON else ("ball" if c == self.BALL else None)

        return _split_tracked_yolo(r, role_of)


class _RFDetrDetector:
    """RF-DETR detector (Roboflow, DINOv2 backbone, NMS-free; SOTA on COCO, ICLR 2026).

    COCO-pretrained: person -> players, 'sports ball' -> ball (COCO 91-class ids 1 and 37). For
    GK/referee/ball-as-distinct-classes, load a football-trained RF-DETR checkpoint instead. Weights
    download on first construction.
    """

    PERSON, BALL = 1, 37  # rfdetr COCO-91 ids

    def __init__(self, *, device: str = "cpu", size: str = "nano"):
        from rfdetr import RFDETRNano, RFDETRSmall  # noqa: PLC0415

        self.model = (RFDETRSmall if size == "small" else RFDETRNano)()
        self.device = device

    def detect(self, frame_rgb):
        """Return ``(player_xyxy, player_conf, role_ids, ball_xy_or_None)`` (roles all ``player``)."""
        det = self.model.predict(frame_rgb, threshold=BALL_CONF)
        if det.class_id is None or len(det) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None
        cls = np.asarray(det.class_id)
        conf = np.asarray(det.confidence) if det.confidence is not None else np.ones(len(det))
        xyxy = np.asarray(det.xyxy)
        ppl = (cls == self.PERSON) & (conf >= DETECT_CONF)
        ball_sel = (cls == self.BALL) & (conf >= BALL_CONF)
        ball_xy = None
        if ball_sel.any():
            b = xyxy[ball_sel][int(np.argmax(conf[ball_sel]))]
            ball_xy = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        return xyxy[ppl], conf[ppl], np.full(int(ppl.sum()), ROLE_PLAYER), ball_xy


class _FootballRoleDetector:
    """Football-trained YOLO giving ball / goalkeeper / player / referee in one pass (Pass C).

    Default weights are the HF ``uisikdag/yolo-v8-football-players-detection`` model, which won the
    ``tools/benchmark_detectors.py`` sweep on role coverage: the only candidate with a goalkeeper
    class, cleanly separates officials (so linesmen stop being counted as players), and is
    conservative on the ball (fewer false positives than soccana / RF-DETR). Class names are mapped by
    name (not a hard-coded index) so any compatibly-named football YOLO can be passed via ``weights``.
    Weights download once (cached) on first construction.
    """

    DEFAULT_REPO = "uisikdag/yolo-v8-football-players-detection"

    def __init__(self, *, weights: str | Path | None = None, device: str = "cpu"):
        from ultralytics import YOLO  # noqa: PLC0415

        self.yolo = YOLO(str(weights) if weights else self._download_default())
        self.yolo.to(device)
        # class id -> canonical role string ("player"/"goalkeeper"/"referee"/"ball" or None to ignore).
        self._role = {cid: _norm_role_name(nm) for cid, nm in self.yolo.names.items()}
        if "goalkeeper" not in self._role.values():
            logger.warning("football detector has no goalkeeper class; keeper falls back to position")

    @staticmethod
    def _download_default() -> str:
        # Prefer the already-cached snapshot (fully offline): a network outage must not break a
        # detector build when the .pt is on disk. Only list/download over the network on a cache miss.
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download  # noqa: PLC0415

        repo = _FootballRoleDetector.DEFAULT_REPO
        try:
            snap = Path(snapshot_download(repo, local_files_only=True, allow_patterns=["*.pt"]))
            local_pts = [p.name for p in snap.glob("*.pt")]
            if local_pts:
                pref = [f for f in local_pts if "best" in f.lower()] or local_pts
                return str(snap / pref[0])
        except Exception:  # noqa: BLE001 - no local snapshot -> fall through to the network path
            pass
        pts = [f for f in list_repo_files(repo) if f.endswith(".pt")]
        pref = [f for f in pts if "best" in f.lower()] or pts
        return hf_hub_download(repo, pref[0])

    def detect(self, frame_rgb):
        """Return ``(person_xyxy, person_conf, role_ids, ball_xy_or_None)`` with real roles."""
        r = self.yolo(frame_rgb, verbose=False, conf=BALL_CONF)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        roles = np.array([self._role.get(int(c)) for c in r.boxes.cls.int().cpu().tolist()], object)

        ball_xy = None
        ball_sel = (roles == "ball") & (conf >= BALL_CONF)
        if ball_sel.any():
            b = xyxy[ball_sel][int(np.argmax(conf[ball_sel]))]
            ball_xy = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

        person = np.isin(roles, ["player", "goalkeeper", "referee"]) & (conf >= DETECT_CONF)
        role_ids = np.array([ROLE_ID[r] for r in roles[person]], int)
        return xyxy[person], conf[person], role_ids, ball_xy

    def track(self, frame_rgb):
        """Detect + BoT-SORT (GMC) in one call -> ``(xyxy, conf, role_ids, track_ids, ball_xy)``."""
        r = self.yolo.track(frame_rgb, persist=True, tracker=_BOTSORT_CFG, verbose=False,
                            conf=BALL_CONF)[0]
        return _split_tracked_yolo(r, lambda c: self._role.get(int(c)))


def extract_positions(
    source: str | Path,
    out: str | Path,
    *,
    sample_every: int = 5,
    calibrator=None,
    max_error_m: float = MAX_REPROJ_ERROR_M,
    start_frame: int = 0,
    max_frames: int | None = None,
    auto_wide: bool = False,
    detector_name: str = "yolo",
    detector_weights: str | None = None,
    tracker_name: str = "bytetrack",
    calib_period: int = 1,
    calib_drift: float = 2.0,
) -> pd.DataFrame:
    """Run the CV pipeline over a video and write a positions parquet.

    Args:
        source: Input video path.
        out: Output parquet path.
        sample_every: Process every Nth frame.
        calibrator: Object with ``calibrate_frame(frame_rgb) -> CalibrationResult`` (e.g.
            :class:`~generator.calibrate.PnLCalibCalibrator`). If ``None``, pitch coords are left NaN
            (image-space tracks only) -- detection/tracking still run.
        max_error_m: Confidence-gate threshold passed through for logging.
        start_frame: Seek here before processing (skip broadcast pre-roll / smoke runs).
        max_frames: Stop after reading this many source frames past ``start_frame``; ``None`` = to end.
        auto_wide: Scan the clip for the highest-scoring wide tactical segment and centre the
            ``max_frames`` window on it (overrides ``start_frame``). Needs ``max_frames`` set.

    Returns:
        The post-processed positions DataFrame (also written to ``out``).
    """
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from generator.tracking import build_tracker  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("extract: source=%s device=%s stride=%d det=%s track=%s",
                source, device, sample_every, detector_name, tracker_name)
    detector = _build_detector(device, detector_name, weights=detector_weights)

    if calibrator is not None and calib_period > 1:  # Pass B: calibrate sparingly + reuse the pose
        from generator.temporal_calib import TemporalCalibrator  # noqa: PLC0415

        calibrator = TemporalCalibrator(calibrator, period=calib_period, drift_thresh=calib_drift)

    if auto_wide and max_frames:
        from generator.segments import best_wide_segment, scan_wide_segments  # noqa: PLC0415

        scan = scan_wide_segments(str(source), detector, start=max(start_frame, 0))
        best = best_wide_segment(scan)
        if best is not None:
            start_frame = max(best - max_frames // 2, 0)
            logger.info("extract: auto-wide picked frame %d -> start_frame=%d", best, start_frame)
        else:
            logger.warning("extract: auto-wide found no wide segment; using start_frame=%d", start_frame)

    # Fit the team classifier on early-clip crops.
    crops = _collect_team_crops(cv2, source, detector)
    if len(crops) < 2:
        logger.warning("extract: no player crops; writing empty output")
        empty_positions().to_parquet(out, index=False)
        return empty_positions()
    team_clf = JerseyColorTeamClassifier().fit(crops)

    tracker = build_tracker(tracker_name)
    rows: list[dict] = []
    rejected = 0
    cap = cv2.VideoCapture(str(source))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fi = start_frame
    while True:
        if max_frames is not None and fi - start_frame >= max_frames:
            break
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if fi % sample_every != 0:
            fi += 1
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detect first: the calibration gate needs this frame's own foot points to reject a
        # homography that fits the pitch keypoints yet projects every player off the pitch.
        xyxy, confs, role_ids, track_ids, ball_xy = tracker.update(detector, frame_rgb)
        if len(xyxy) == 0:
            fi += 1
            continue

        roles = np.array([ROLE_NAME.get(int(c), "player") for c in role_ids])
        # Crop from BGR (the classifier's expected colour order; was a real bug when RGB was used).
        crops = [_safe_crop(frame_bgr, b) for b in xyxy]
        teams = team_clf.predict(crops).astype(int)
        teams = np.where(roles == "referee", -1, teams)  # officials belong to no team
        foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])

        calib = None
        if calibrator is not None:
            try:  # calibrator's API expects BGR
                calib = calibrator.calibrate_frame(frame_bgr, foot[np.isin(roles, PLAYER_ROLES)])
            except Exception as exc:  # noqa: BLE001 - a failed calib just means NaN coords this frame
                logger.debug("frame %d: calibration failed: %s", fi, exc)
        if calib is not None and not calib.ok:
            rejected += 1
        err = calib.error_m if calib is not None else float("nan")

        pitch = transform_and_gate(foot, calib)
        rows.extend(build_player_rows(fi, track_ids, teams, foot, pitch, confs, err, roles=roles))

        if ball_xy is not None:
            ball_pitch = transform_and_gate(ball_xy.reshape(1, 2), calib)[0]
            rows.append(build_ball_row(fi, ball_xy, ball_pitch, err))
        fi += 1
    cap.release()

    df = postprocess(pd.DataFrame(rows)) if rows else empty_positions()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    valid = df.dropna(subset=["pitch_x", "pitch_y"])
    logger.info(
        "extract: %d rows (%d with pitch coords); %d frames rejected by the >%.1fm gate -> %s",
        len(df), len(valid), rejected, max_error_m, out,
    )
    if hasattr(calibrator, "n_full"):  # temporal calibrator: report how much full calibration it saved
        logger.info("extract: temporal calib ran %d full calibrations, reused %d frames",
                    calibrator.n_full, calibrator.n_reuse)
    return df


def _safe_crop(frame_rgb, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    c = frame_rgb[max(y1, 0):y2, max(x1, 0):x2]
    return c if c.size > 0 else np.zeros((32, 16, 3), np.uint8)


def _collect_team_crops(cv2, source, detector, n_max: int = 1024) -> list:
    """Grab player crops spread across the WHOLE clip to fit the team classifier.

    Fitting on a single early window collapses the 2-team KMeans when one team is off-screen / defending
    deep there (a follow-play broadcast often shows mostly one team for a stretch), which mislabels the
    entire chunk. Sampling short windows spread from 15%-90% of the clip lets both kits appear across
    varied phases, so the colour split is balanced. Outfield PLAYERS only (GK/referee kits would pull the
    KMeans onto extra colours).
    """
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    crops: list = []
    fracs = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
    per_window = max(n_max // len(fracs), 1)
    for frac in fracs:
        cap = cv2.VideoCapture(str(source))
        got = 0
        for i in range(int(total * frac), min(int(total * frac) + 240, total), 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            boxes, _, role_ids, _ = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keep = role_ids == ROLE_PLAYER if len(role_ids) else np.ones(len(boxes), bool)
            for b in boxes[keep].astype(int):
                crop = frame[max(b[1], 0):b[3], max(b[0], 0):b[2]]
                if crop.size > 0:
                    crops.append(crop)
                    got += 1
            if got >= per_window:
                break       # move to the next window so the fit isn't dominated by one phase
        cap.release()
        if len(crops) >= n_max:
            break
    return crops


def main() -> None:
    """CLI entry point (needs the ``[cv]`` stack)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="input video")
    ap.add_argument("--out", required=True, help="output positions parquet")
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--start-frame", type=int, default=0, help="seek here first (skip pre-roll)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames past start")
    ap.add_argument("--calibrate", action="store_true", help="enable PnLCalib pitch calibration")
    ap.add_argument("--auto-wide", action="store_true", help="auto-pick the widest tactical segment")
    ap.add_argument("--detector", choices=["football", "yolo", "rfdetr"], default="yolo",
                    help="detector backend (football = role classes; recommended for tactical work)")
    ap.add_argument("--weights", default=None, help="override detector weights (path or HF .pt)")
    ap.add_argument("--tracker", choices=["botsort", "bytetrack"], default="bytetrack",
                    help="bytetrack (default) won the persistence benchmark on this footage; botsort "
                         "adds GMC+ReID for heavy-pan footage (yolo/football only, not rfdetr)")
    ap.add_argument("--calib-period", type=int, default=1,
                    help="full PnLCalib every N frames + on cut/drift, reusing the pose between "
                         "(1=every frame; e.g. 25 for ~1s on a 25fps held camera)")
    ap.add_argument("--calib-drift", type=float, default=2.0,
                    help="recalibrate when camera shift (phase-corr px @320w) exceeds this; "
                         "lower=more accurate/more calibrations")
    args = ap.parse_args()
    calibrator = None
    if args.calibrate:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        calibrator = PnLCalibCalibrator()  # device=None -> auto-detect CUDA
    extract_positions(
        args.source, args.out, sample_every=args.sample_every,
        start_frame=args.start_frame, max_frames=args.max_frames, calibrator=calibrator,
        auto_wide=args.auto_wide, detector_name=args.detector, detector_weights=args.weights,
        tracker_name=args.tracker, calib_period=args.calib_period, calib_drift=args.calib_drift,
    )


if __name__ == "__main__":
    main()
