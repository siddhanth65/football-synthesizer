"""Temporal homography carry-over for ball projection (Phase-A PL coverage lever).

The Phase-A PL probe found post-``link_ball`` ball coverage stuck ~21% despite 87% held-out
detection recall. The binding constraint is *per-frame homography availability*: only frames with
>= 6 on-pitch player correspondences get a homography that :func:`tools.ball_possession.project_ball`
can fit, so a ball detected on a calibration-ambiguous midfield frame cannot be placed on the pitch.

This module implements a **camera-continuous temporal carry-over**: for a frame that lacks its own
usable homography we reuse (or interpolate between) the nearest frames that *do* have one, provided
no camera cut lies between them and they fall inside a short time window. Cuts are detected cheaply
from track-ID discontinuity in the dense parquet (a cut resets ByteTrack, so almost no track ids
survive it) -- no video re-decode.

Everything here is pure and unit-testable (only ``cv2.findHomography`` / ``perspectiveTransform`` are
used, both CPU). The carry-over is *off by default* at the call sites; the WC pipeline is unchanged.

Hard-won caveat (a WC retraction): temporal-H made zero post-link difference on the WC corpus, where
the gaps were camera cuts (uninterpolable). The PL failure mode is continuous-camera midfield
ambiguity, so carry-over is *plausible* here -- but the only number that counts is post-``link_ball``
usable-track coverage, measured end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from generator.postprocess import SRC_LEN, SRC_WID, clamp_to_pitch

PLAYER_ROLES = ("player", "goalkeeper")
DEFAULT_MIN_PTS = 6            # player correspondences needed to fit a trustworthy per-frame H
DEFAULT_MAX_REPROJ_PX = 12.0  # RANSAC inlier band (mirrors tools.ball_possession.project_ball)
DEFAULT_WINDOW_FRAMES = 50     # +/- carry window in native frames (~2 s at 25 fps)
DEFAULT_JACCARD_CUT = 0.30     # track-id Jaccard below this between sampled frames = camera cut


def fit_frame_homographies(
    players: pd.DataFrame, *, min_pts: int = DEFAULT_MIN_PTS,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
) -> dict[int, np.ndarray]:
    """Fit an image->pitch homography per frame from the on-pitch player correspondences.

    A frame yields a homography only if it has ``>= min_pts`` players carrying valid pitch coords and
    ``cv2.findHomography`` returns a non-degenerate matrix. These are the *known-good* homographies
    (the same ones :func:`tools.ball_possession.project_ball` fits) reused as carry-over sources.

    Args:
        players: player rows with ``frame, image_x, image_y, pitch_x, pitch_y``.
        min_pts: minimum player correspondences per frame.
        max_reproj_px: RANSAC reprojection band (pixels).

    Returns:
        ``{frame: (3, 3) homography}`` for every frame that fits.
    """
    import cv2  # noqa: PLC0415

    valid = players.dropna(subset=["image_x", "image_y", "pitch_x", "pitch_y"])
    out: dict[int, np.ndarray] = {}
    for fr, g in valid.groupby("frame"):
        if g["track_id"].nunique() < min_pts:
            continue
        src = g[["image_x", "image_y"]].to_numpy(np.float32)
        dst = g[["pitch_x", "pitch_y"]].to_numpy(np.float32)
        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, max_reproj_px)
        if h is not None:
            out[int(fr)] = np.asarray(h, dtype=float)
    return out


def camera_segment_ids(
    dense: pd.DataFrame, *, jaccard_cut: float = DEFAULT_JACCARD_CUT,
) -> dict[int, int]:
    """Assign a camera-continuity segment id to each sampled frame via track-id Jaccard.

    Between consecutive sampled frames we measure the Jaccard overlap of their player+GK track-id
    sets. A camera cut resets the tracker, so the overlap collapses toward zero; when it drops below
    ``jaccard_cut`` we start a new segment. Frames in the same segment are camera-continuous and may
    share a carried homography; a carry is never allowed to cross a segment boundary.

    Args:
        dense: dense positions table (needs ``frame, track_id, role``).
        jaccard_cut: overlap below which a boundary is treated as a camera cut.

    Returns:
        ``{frame: segment_id}`` (segment ids are 0-based and increasing in frame order).
    """
    players = dense[dense["role"].isin(PLAYER_ROLES)]
    ids = {int(fr): set(g["track_id"]) for fr, g in players.groupby("frame")}
    frames = sorted(int(f) for f in dense["frame"].unique())
    seg: dict[int, int] = {}
    cur = 0
    for i, fr in enumerate(frames):
        if i > 0:
            prev = frames[i - 1]
            a, b = ids.get(prev, set()), ids.get(fr, set())
            jac = len(a & b) / len(a | b) if (a | b) else 1.0
            if jac < jaccard_cut:
                cur += 1
        seg[fr] = cur
    return seg


def _nearest_sources(
    target: int, good_frames: np.ndarray, seg: dict[int, int], *, window: int,
) -> tuple[int | None, int | None]:
    """Nearest good frame on each side of ``target``, same camera segment, within ``window`` frames."""
    tseg = seg.get(target)
    left = right = None
    lo = np.searchsorted(good_frames, target)
    # scan left
    for j in range(lo - 1, -1, -1):
        g = int(good_frames[j])
        if target - g > window:
            break
        if seg.get(g) == tseg:
            left = g
            break
    # scan right
    for j in range(lo, len(good_frames)):
        g = int(good_frames[j])
        if g == target:
            continue
        if g - target > window:
            break
        if seg.get(g) == tseg:
            right = g
            break
    return left, right


@dataclass
class CarryStats:
    """Diagnostic counts for a carry-over projection pass (all PRE-link)."""

    detected: int = 0            # ball detections fed in
    own: int = 0                 # projected via the frame's own homography (baseline behaviour)
    carried: int = 0             # projected via a single carried neighbour homography
    interpolated: int = 0        # projected via a blend of two bracketing homographies
    offpitch_dropped: int = 0    # carried/own projection landed off-pitch -> rejected
    no_source: int = 0           # no own H and no in-window same-segment neighbour
    carried_frames: list = field(default_factory=list)  # frames whose ball came from a carry/interp

    def as_dict(self) -> dict:
        """Flat dict for table rendering."""
        return {"detected": self.detected, "own": self.own, "carried": self.carried,
                "interpolated": self.interpolated, "offpitch_dropped": self.offpitch_dropped,
                "no_source": self.no_source}


def project_ball_carry(
    ball_imgxy: dict[int, tuple[float, float]],
    dense: pd.DataFrame,
    *,
    carry: bool = True,
    interpolate: bool = True,
    window: int = DEFAULT_WINDOW_FRAMES,
    min_pts: int = DEFAULT_MIN_PTS,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
    jaccard_cut: float = DEFAULT_JACCARD_CUT,
    src_len: float = SRC_LEN,
    src_wid: float = SRC_WID,
) -> tuple[pd.DataFrame, CarryStats]:
    """Project ball image detections to pitch metres, optionally with temporal homography carry-over.

    With ``carry=False`` this reproduces the baseline projection (own-frame homography only, the
    :func:`tools.ball_possession.project_ball` behaviour). With ``carry=True`` a detection on a frame
    lacking its own homography reuses the nearest same-camera-segment homography within ``window``
    frames (or, with ``interpolate``, a temporal blend of the two bracketing ones). Every projected
    point is clamped to the pitch; off-pitch projections are rejected (so a wrong carried homography
    cannot plant a phantom on-pitch ball -- the physical guard, complemented downstream by
    ``link_ball``'s speed clamp).

    Args:
        ball_imgxy: ``{frame: (x, y)}`` ball image-pixel detections.
        dense: dense positions table (players supply correspondences + cut detection).
        carry: enable temporal carry-over (the lever); ``False`` = baseline.
        interpolate: blend the two bracketing homographies when both exist in-window.
        window: carry window in native frames each side.
        min_pts: player correspondences needed to fit a source homography.
        max_reproj_px: RANSAC band for homography fitting.
        jaccard_cut: track-id Jaccard cut threshold for camera segmentation.
        src_len: pitch length (m) for the off-pitch gate.
        src_wid: pitch width (m) for the off-pitch gate.

    Returns:
        ``(pitch_df, stats)`` where ``pitch_df`` has columns ``frame, x, y, source`` (``source`` in
        ``own``/``carried``/``interp``), sorted by frame, and ``stats`` is a :class:`CarryStats`.
    """
    import cv2  # noqa: PLC0415

    players = dense[dense["role"].isin(PLAYER_ROLES)]
    homs = fit_frame_homographies(players, min_pts=min_pts, max_reproj_px=max_reproj_px)
    good_frames = np.array(sorted(homs), dtype=int)
    seg = camera_segment_ids(dense, jaccard_cut=jaccard_cut) if carry else {}

    def _project(h: np.ndarray, bx: float, by: float) -> tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[[bx, by]]], np.float32), h)[0, 0]
        return clamp_to_pitch(float(p[0]), float(p[1]), src_len=src_len, src_wid=src_wid)

    rows: list[dict] = []
    stats = CarryStats(detected=len(ball_imgxy))
    for fr, (bx, by) in ball_imgxy.items():
        fr = int(fr)
        if fr in homs:  # own-frame homography (baseline path)
            x, y = _project(homs[fr], bx, by)
            if np.isnan(x):
                stats.offpitch_dropped += 1
                continue
            rows.append({"frame": fr, "x": x, "y": y, "source": "own"})
            stats.own += 1
            continue
        if not carry or good_frames.size == 0:
            stats.no_source += 1
            continue
        left, right = _nearest_sources(fr, good_frames, seg, window=window)
        if left is None and right is None:
            stats.no_source += 1
            continue
        if interpolate and left is not None and right is not None:
            xl, yl = _project(homs[left], bx, by)
            xr, yr = _project(homs[right], bx, by)
            if np.isnan(xl) or np.isnan(yr):
                stats.offpitch_dropped += 1
                continue
            w = (fr - left) / (right - left)
            x, y = (1 - w) * xl + w * xr, (1 - w) * yl + w * yr
            source = "interp"
        else:
            g = left if right is None else (right if left is None else
                                            (left if fr - left <= right - fr else right))
            x, y = _project(homs[g], bx, by)
            source = "carried"
        if np.isnan(x) or not (0 <= x <= src_len and 0 <= y <= src_wid):
            stats.offpitch_dropped += 1
            continue
        rows.append({"frame": fr, "x": x, "y": y, "source": source})
        stats.carried_frames.append(fr)
        if source == "interp":
            stats.interpolated += 1
        else:
            stats.carried += 1
    out = pd.DataFrame(rows, columns=["frame", "x", "y", "source"]).sort_values("frame")
    return out.reset_index(drop=True), stats
