"""Temporal homography carry-over for PLAYER projection (the geometry-yield lever).

The project's binding constraint is **player geometry yield**, not calibration: on
``brighton_manutd/h1_chunk_000`` 91.8% of detection-frames solve a homography with an excellent
*keypoint* reprojection error (median 0.21 m), yet only **37.4%** yield usable player pitch
coordinates. On the remainder the pose fits the pitch keypoints but is *globally wrong*: projected
through it, the detected players land off the pitch, and ``postprocess.reject_implausible_frames``
correctly NaNs the whole frame. Every downstream metric therefore lives on ~37% of frames.

This module applies the mechanism that already worked for the ball (:mod:`generator.ball_carry`,
post-link ball coverage 20.8 -> 38.8%, leave-one-out faithfulness median 0.33 m) to players:

    For a frame with no usable geometry, **borrow** a known-good homography from a
    camera-continuous neighbour within +-``window`` frames and project *that frame's own*
    detected players through it.

**Read this before assuming it is a re-run of a disproven idea.** ``results/pl_probe/diagnosis``
disproved "mechanism 2" = *reusing the frame's own (globally wrong) pose*. That cannot work: the
pose is wrong, so reusing it reproduces the same off-pitch garbage. Carry-over is a *different*
mechanism -- the pose comes from a **different, verified-good frame**, and the only thing taken from
the target frame is its player pixel coordinates. The two are unrelated.

Source poses come from :func:`generator.ball_carry.fit_frame_homographies`: a homography refit from
the player correspondences of a frame that *already produced valid pitch coords*. That definition is
load-bearing -- it selects poses that are known good **by their output** (players land on the pitch,
well spread), not merely poses that fit the keypoints. The alternative source pool ("any frame whose
calibration passed the <=2 m keypoint gate") is 91% of frames, of which ~59% are exactly the
globally-wrong poses this module exists to route around; it must not be used.

Two guards keep a wrong carry from planting phantom players:

1. **Off-pitch gate** (per player): :func:`generator.postprocess.clamp_to_pitch` -- anything more
   than 2 m past a line is dropped to NaN, never clamped onto it.
2. **Frame plausibility gate** (per frame): the carried frame must land ``>= min_onpitch`` players
   on the pitch spanning ``>= min_span_m`` -- the same test ``reject_implausible_frames`` applies to
   native poses. A frame that fails is discarded whole, exactly as a native bad pose would be.

Pure CPU (only ``cv2.findHomography`` / ``perspectiveTransform``). Off by default at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from generator.ball_carry import (
    DEFAULT_JACCARD_CUT,
    DEFAULT_MAX_REPROJ_PX,
    DEFAULT_MIN_PTS,
    DEFAULT_WINDOW_FRAMES,
    PLAYER_ROLES,
    _nearest_sources,
    camera_segment_ids,
    fit_frame_homographies,
)
from generator.postprocess import (
    MIN_ONPITCH_PLAYERS,
    MIN_PITCH_SPAN_M,
    SRC_LEN,
    SRC_WID,
    clamp_to_pitch,
    derive_keeper,
)

__all__ = [
    "PoseCarryStats",
    "carry_player_poses",
    "geometry_yield",
    "good_geometry_frames",
]


def good_geometry_frames(
    dense: pd.DataFrame, *, min_onpitch: int = MIN_ONPITCH_PLAYERS,
) -> np.ndarray:
    """Frames that already yield usable player geometry (``>= min_onpitch`` valid pitch coords).

    These are the *known-good* poses: their homography is validated by its own output (the players it
    projects land on the pitch and survive ``reject_implausible_frames``), not merely by keypoint
    reprojection error.

    Args:
        dense: dense positions table (``frame, role, pitch_x, pitch_y``).
        min_onpitch: players with valid pitch coords a frame needs to count as good.

    Returns:
        Sorted array of good frame indices.
    """
    players = dense[dense["role"].isin(PLAYER_ROLES)]
    counts = players.groupby("frame")["pitch_x"].apply(lambda s: int(s.notna().sum()))
    return np.array(sorted(int(f) for f in counts[counts >= min_onpitch].index), dtype=int)


def geometry_yield(dense: pd.DataFrame, *, min_onpitch: int = MIN_ONPITCH_PLAYERS) -> float:
    """Fraction of detection-frames that yield usable player pitch coordinates."""
    players = dense[dense["role"].isin(PLAYER_ROLES)]
    n_frames = players["frame"].nunique()
    if not n_frames:
        return float("nan")
    return len(good_geometry_frames(dense, min_onpitch=min_onpitch)) / n_frames


@dataclass
class PoseCarryStats:
    """Diagnostic counts for one pose carry-over pass."""

    det_frames: int = 0          # frames with player detections
    good_before: int = 0         # frames with usable geometry before the carry
    targets: int = 0             # frames lacking geometry (carry candidates)
    no_source: int = 0           # no in-window same-segment good neighbour
    carried: int = 0             # frames recovered via a single borrowed pose
    interpolated: int = 0        # frames recovered via a blend of two bracketing poses
    frame_gate_dropped: int = 0  # a source was found but the carried frame failed the plausibility gate
    players_added: int = 0       # player rows that gained valid pitch coords
    carried_frame_ids: list = field(default_factory=list)

    @property
    def good_after(self) -> int:
        """Frames with usable geometry after the carry."""
        return self.good_before + self.carried + self.interpolated

    @property
    def yield_before(self) -> float:
        """Player geometry yield before the carry."""
        return self.good_before / self.det_frames if self.det_frames else float("nan")

    @property
    def yield_after(self) -> float:
        """Player geometry yield after the carry."""
        return self.good_after / self.det_frames if self.det_frames else float("nan")

    def as_dict(self) -> dict:
        """Flat dict for table rendering."""
        return {
            "det_frames": self.det_frames, "good_before": self.good_before,
            "good_after": self.good_after, "targets": self.targets,
            "carried": self.carried, "interpolated": self.interpolated,
            "frame_gate_dropped": self.frame_gate_dropped, "no_source": self.no_source,
            "players_added": self.players_added,
            "yield_before": self.yield_before, "yield_after": self.yield_after,
        }


def project_players(
    h: np.ndarray, img: np.ndarray, *, src_len: float = SRC_LEN, src_wid: float = SRC_WID,
) -> np.ndarray:
    """Project ``(N, 2)`` image points through ``h`` and off-pitch-gate them (NaN outside).

    Args:
        h: ``(3, 3)`` image->pitch homography.
        img: ``(N, 2)`` image-pixel coordinates.
        src_len: pitch length (m).
        src_wid: pitch width (m).

    Returns:
        ``(N, 2)`` pitch coordinates; rows outside the pitch (+ tolerance) are NaN.
    """
    import cv2  # noqa: PLC0415

    pts = np.asarray(img, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, np.asarray(h, dtype=float)).reshape(-1, 2)
    out = np.empty_like(proj, dtype=float)
    for i, (px, py) in enumerate(proj):
        out[i] = clamp_to_pitch(float(px), float(py), src_len=src_len, src_wid=src_wid)
    return out


def carry_player_poses(
    dense: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW_FRAMES,
    interpolate: bool = True,
    min_pts: int = DEFAULT_MIN_PTS,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
    jaccard_cut: float = DEFAULT_JACCARD_CUT,
    min_onpitch: int = MIN_ONPITCH_PLAYERS,
    min_span_m: float = MIN_PITCH_SPAN_M,
    src_len: float = SRC_LEN,
    src_wid: float = SRC_WID,
    rederive_keeper: bool = True,
) -> tuple[pd.DataFrame, PoseCarryStats]:
    """Fill missing player pitch coords by borrowing a camera-continuous neighbour's good pose.

    Frames that already have usable geometry are left **byte-identical**; only frames with no valid
    player pitch coords are touched, so the baseline pipeline is exactly reproduced when no carry
    succeeds. Recovered frames are stamped with a ``pose_source`` provenance column and inherit their
    source frame's ``calib_error_m`` -- the geometry they carry *is* that source pose, so quoting the
    source's reprojection error is the honest provenance (and it is what lets the downstream
    ``calib_error_m <= 1 m`` fact-store gate see them at all; a carried frame's own keypoint error
    describes a pose that was never used).

    Args:
        dense: dense positions table with ``frame, track_id, role, pitch_x, pitch_y, image_x,
            image_y, calib_error_m``.
        window: carry window in native frames each side (convert seconds via the chunk's true fps).
        interpolate: blend the two bracketing poses when both exist in-window.
        min_pts: player correspondences needed to refit a source pose.
        max_reproj_px: RANSAC band (px) for the source-pose refit.
        jaccard_cut: track-id Jaccard below which a frame boundary is treated as a camera cut.
        min_onpitch: players a carried frame must land on the pitch to be accepted.
        min_span_m: pitch span (m) a carried frame's players must cover to be accepted.
        src_len: pitch length (m).
        src_wid: pitch width (m).
        rederive_keeper: re-run :func:`generator.postprocess.derive_keeper` over the result.

    Returns:
        ``(enriched, stats)``. ``enriched`` is ``dense`` plus a ``pose_source`` column
        (``own``/``carried``/``interp``/``none``), with carried frames' ``pitch_x``/``pitch_y``/
        ``calib_error_m`` filled in.
    """
    out = dense.copy()
    out["pose_source"] = "none"
    is_player = out["role"].isin(PLAYER_ROLES)
    players = out[is_player]

    good = good_geometry_frames(out, min_onpitch=min_onpitch)
    out.loc[is_player & out["frame"].isin(good), "pose_source"] = "own"

    det_frames = sorted(int(f) for f in players["frame"].unique())
    stats = PoseCarryStats(det_frames=len(det_frames), good_before=len(good))
    if not len(good):
        return out, stats

    homs = fit_frame_homographies(players, min_pts=min_pts, max_reproj_px=max_reproj_px)
    src_frames = np.array(sorted(f for f in homs if f in set(good.tolist())), dtype=int)
    if not src_frames.size:
        return out, stats
    seg = camera_segment_ids(out, jaccard_cut=jaccard_cut)
    err_by_frame = out.groupby("frame")["calib_error_m"].first().to_dict()

    good_set = set(good.tolist())
    targets = [f for f in det_frames if f not in good_set]
    stats.targets = len(targets)

    rows_by_frame = {int(f): g for f, g in players.groupby("frame")}
    fills: list[tuple[pd.Index, np.ndarray, str, float]] = []
    for fr in targets:
        left, right = _nearest_sources(fr, src_frames, seg, window=window)
        if left is None and right is None:
            stats.no_source += 1
            continue
        grp = rows_by_frame[fr]
        img = grp[["image_x", "image_y"]].to_numpy(float)
        if np.isnan(img).any():
            img = np.nan_to_num(img, nan=-1e6)  # a NaN pixel projects off-pitch -> gated out
        if interpolate and left is not None and right is not None:
            pl = project_players(homs[left], img, src_len=src_len, src_wid=src_wid)
            pr = project_players(homs[right], img, src_len=src_len, src_wid=src_wid)
            w = (fr - left) / (right - left)
            pitch = (1 - w) * pl + w * pr  # NaN in either bracket propagates -> that player is dropped
            source = "interp"
            cerr = max(err_by_frame.get(left, np.inf), err_by_frame.get(right, np.inf))
        else:
            g = left if right is None else (right if left is None else
                                            (left if fr - left <= right - fr else right))
            pitch = project_players(homs[g], img, src_len=src_len, src_wid=src_wid)
            source = "carried"
            cerr = err_by_frame.get(g, np.inf)
        valid = ~np.isnan(pitch[:, 0])
        n_valid = int(valid.sum())
        span = 0.0
        if n_valid:
            span = float(max(np.ptp(pitch[valid, 0]), np.ptp(pitch[valid, 1])))
        if n_valid < min_onpitch or span < min_span_m:  # the frame-plausibility guard
            stats.frame_gate_dropped += 1
            continue
        fills.append((grp.index, pitch, source, float(cerr)))
        stats.players_added += n_valid
        stats.carried_frame_ids.append(fr)
        if source == "interp":
            stats.interpolated += 1
        else:
            stats.carried += 1

    cerr_by_frame: dict[int, float] = {}
    for idx, pitch, source, cerr in fills:
        out.loc[idx, "pitch_x"] = pitch[:, 0]
        out.loc[idx, "pitch_y"] = pitch[:, 1]
        out.loc[idx, "pose_source"] = source
        cerr_by_frame[int(out.loc[idx[0], "frame"])] = cerr
    if cerr_by_frame:  # frame-level provenance: the carried pose's own reprojection error
        touched = out["frame"].isin(cerr_by_frame)
        out.loc[touched, "calib_error_m"] = out.loc[touched, "frame"].map(cerr_by_frame)

    if rederive_keeper and fills:
        carried_ids = set(stats.carried_frame_ids)
        sub = derive_keeper(out[out["frame"].isin(carried_ids)], src_len=src_len)
        out.loc[sub.index, "is_keeper"] = sub["is_keeper"]
    return out, stats
