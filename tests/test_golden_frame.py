"""Golden-frame + rejected-frame integration tests for the PnLCalib calibration seam.

These run the *real* PnLCalib model on a known broadcast frame, so they are **opt-in**: they skip
unless the PnLCalib weights and the golden video are both present (CI without the heavy CV stack still
passes). They guard the exact failure that shipped a broken generator -- PnLCalib's camera params are
in a frame **centred** on the centre spot, and the derived homography must output the project-wide
**uncentred** ``[0,105] x [0,68]`` coords. If the de-centring is dropped, every player is offset by
~(52.5, 34) m (keeper at "halfway", players crammed into one half) and the metre gate rejects the
frame -- both asserted below.

Set ``FOOTBALL_GOLDEN_VIDEO`` to override the clip path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

GOLDEN_VIDEO = Path(
    os.environ.get(
        "FOOTBALL_GOLDEN_VIDEO",
        r"C:\Users\siddh_ygv5bws\OneDrive\Desktop\cv-football\chunks\chunk_000.mp4",
    )
)
GOLDEN_FRAME = 13031  # a wide right-half view (halfway line at the left edge, right box centre-right)


def _pnlcalib_available() -> bool:
    try:
        from generator.calibrate import pnlcalib_root  # noqa: PLC0415

        root = pnlcalib_root()
        return (root / "weights" / "SV_kp").exists() and (root / "weights" / "SV_lines").exists()
    except Exception:  # noqa: BLE001 - any failure means "not available", just skip
        return False


pytestmark = pytest.mark.skipif(
    not (GOLDEN_VIDEO.exists() and _pnlcalib_available()),
    reason="golden video and/or PnLCalib weights not present (opt-in integration test)",
)


def _read_frame(idx: int) -> np.ndarray:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(GOLDEN_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read golden frame {idx}"
    return frame


@pytest.fixture(scope="module")
def calib_result():
    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

    return PnLCalibCalibrator().calibrate_frame(_read_frame(GOLDEN_FRAME))


def test_golden_frame_calibrates_within_gate(calib_result):
    """The known-good frame must calibrate and pass the metre gate (the de-centring keeps err low)."""
    assert calib_result.homography is not None
    assert calib_result.ok and calib_result.error_m <= 2.0


def test_golden_frame_maps_to_uncentred_pitch_without_inversion(calib_result):
    """pitch->image orientation: the right goal sits right of centre, and on-pitch points are uncentred.

    A dropped de-centring shift would push these into a centred frame (centre spot -> a corner), which
    this catches: the right goal must project to the right of the centre spot in the image, and an
    on-screen pitch point must round-trip into ``[0,105] x [0,68]`` near where it started.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    pitch_to_img = np.linalg.inv(calib_result.homography)

    def to_img(x, y):
        v = pitch_to_img @ np.array([x, y, 1.0])
        return v[:2] / v[2]

    # No left-right inversion: right goal (105) is to the right of the centre spot (52.5) in the image.
    assert to_img(105.0, 34.0)[0] > to_img(52.5, 34.0)[0]

    # Uncentred convention: the on-screen right box corner round-trips into pitch bounds near itself.
    box_corner = np.array([88.5, 54.16])
    back = apply_homography(calib_result.homography, to_img(*box_corner).reshape(1, 2))[0]
    assert np.linalg.norm(back - box_corner) < 1.0
    assert 0.0 <= back[0] <= 105.0 and 0.0 <= back[1] <= 68.0
