"""Temporal calibration: run full PnLCalib sparingly and reuse a stable pose in between (Pass B).

Calibrating every frame independently is the pipeline's slowest step and adds frame-to-frame jitter
to player coordinates. Within a continuous shot the broadcast camera barely moves, so this wrapper:

* runs the expensive base calibration only on the **first frame of a shot**, **every ``period``
  frames**, or when **camera drift** (phase-correlation shift vs the last calibrated frame) exceeds a
  threshold -- i.e. on real pan/zoom;
* **reuses the last accepted pose** for the frames in between (one stable homography -> no calibration
  jitter, much less compute);
* resets at **shot cuts** (grayscale-histogram break), so a pose never carries across a cut.

It honours the hard rule from the spec: a rejected frame is never resurrected. If the last full
calibration failed the gate, nothing is reused -- every following frame is freshly calibrated until an
accepted pose appears, and frames keep emitting ``ok=False`` until then.

The wrapper is drop-in: it exposes the same ``calibrate_frame(frame_bgr) -> CalibrationResult`` as
:class:`~generator.calibrate.PnLCalibCalibrator`.
"""

from __future__ import annotations

import numpy as np

from generator.calibrate import CalibrationResult


def decide_recalibration(
    since: int, period: int, drift: float, drift_thresh: float, is_cut: bool, has_last_ok: bool
) -> bool:
    """Pure policy: should this frame run a full calibration (vs reuse the last pose)?

    Recalibrate on a cut, when there is no accepted pose to reuse, once ``period`` frames have elapsed,
    or when camera drift since the last calibration crosses ``drift_thresh``. Otherwise reuse.
    """
    if is_cut or not has_last_ok:
        return True
    if since >= period:
        return True
    return drift >= drift_thresh


class TemporalCalibrator:
    """Wrap a per-frame calibrator with shot-aware periodic/drift-triggered recalibration + reuse."""

    def __init__(
        self,
        base,
        *,
        period: int = 25,
        drift_thresh: float = 2.0,
        cut_corr: float = 0.5,
        downscale_w: int = 320,
    ):
        self.base = base
        self.period = period
        self.drift_thresh = drift_thresh
        self.cut_corr = cut_corr
        self.downscale_w = downscale_w
        self._since = 10**9  # frames since last full calibration (force one on the first frame)
        self._last_ok: CalibrationResult | None = None  # last accepted result, eligible for reuse
        self._ref = None  # downscaled gray of the last calibrated frame (drift reference)
        self._prev = None  # previous frame's downscaled gray (cut detection)
        self.n_full = 0  # stats: full calibrations actually run
        self.n_reuse = 0  # stats: frames served from a reused pose

    # --- frame motion signals (overridable; isolated from the pure policy above) ---
    def _gray(self, frame_bgr):
        import cv2  # noqa: PLC0415

        h, w = frame_bgr.shape[:2]
        new_h = max(1, int(h * self.downscale_w / w))
        small = cv2.resize(frame_bgr, (self.downscale_w, new_h))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _is_cut(self, gray) -> bool:
        import cv2  # noqa: PLC0415

        if self._prev is None:
            return False
        h1 = cv2.calcHist([self._prev], [0], None, [64], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < self.cut_corr

    def _drift(self, gray) -> float:
        import cv2  # noqa: PLC0415

        if self._ref is None:
            return float("inf")
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(self._ref), np.float32(gray))
        return float(np.hypot(dx, dy))

    def calibrate_frame(self, frame_bgr) -> CalibrationResult:
        gray = self._gray(frame_bgr)
        is_cut = self._is_cut(gray)
        if is_cut:  # a cut invalidates the reusable pose and the drift reference
            self._last_ok = None
            self._ref = None
            self._since = 10**9
        drift = self._drift(gray)
        recal = decide_recalibration(
            self._since, self.period, drift, self.drift_thresh, is_cut, self._last_ok is not None
        )
        self._prev = gray

        if recal:
            res = self.base.calibrate_frame(frame_bgr)
            self.n_full += 1
            self._since = 1  # this frame counts as 1 toward the next period boundary
            self._ref = gray
            self._last_ok = res if res.ok else None  # never reuse a rejected calibration
            return res

        self.n_reuse += 1
        self._since += 1
        return self._last_ok  # decide_recalibration guarantees this is an accepted result
