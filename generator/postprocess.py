"""Pure post-processing of a raw positions table: smoothing + ball-carrier / keeper derivation.

These are the steps that turn raw per-frame detections into the model-ready positions schema
(``frame, track_id, role, team, pitch_x, pitch_y, is_actor, ...``). They were buried inside
``cv-football``'s ``extract_positions.postprocess``; here they are pure (numpy/pandas only), so they
run and are tested **without** the CV stack, and are reused by :mod:`generator.extract`.

Conventions (105x68 pitch, the ``cv-football`` source convention; the contract rescales to 120x80):
- ``role == "ball"`` rows carry the ball position (one per frame at most); ``role == "player"``
  rows carry players.
- The **actor** (ball-carrier) is the player nearest the ball in its frame, within
  :data:`ACTOR_MAX_DIST_M`.
- The **keeper** for each team is that team's most extreme player along the pitch's long axis
  (deepest for the attacking team, highest for the defending team) -- the same heuristic the
  contract's ``to_frames`` applies, lifted here so the persisted table already carries it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN as SRC_LEN, PITCH_WID as SRC_WID
SMOOTH_WINDOW = 5  # frames; rolling-median smooth of pitch coords per track (kills homography jitter)
ACTOR_MAX_DIST_M = 3.0  # a player within this of the ball is tagged the carrier (actor)
# Metres a ground point may sit past a line (foot-point + calibration noise) before it is OFF the
# pitch. Deliberately tight: officials/ball-boys outside the touchline must be *dropped*, not clamped
# onto the line as phantom players. (A touchline-hugging assistant referee still needs the detector's
# `referee` role to exclude -- position alone cannot separate them from a winger.)
_PITCH_EDGE_TOL_M = 2.0
KEEPER_GOAL_DEPTH_M = 16.5  # a positional keeper must be within this of a goal line (penalty-box depth)
NON_PLAYER_ROLES = ("ball", "referee")  # excluded from actor/keeper/team reasoning
PLAYER_ROLES = ("player", "goalkeeper")
# Rows a frame needs before its own homography can be re-derived from them by DLT (a genuine
# numerical requirement: 4 is the minimum, 8 makes the fit robust). NOT a trust threshold.
MIN_ONPITCH_PLAYERS = 8
MIN_PITCH_SPAN_M = 25.0  # retained for callers that ask for the original wide-shot definition
# Frame-calibration trust rule: a frame is trusted when its players project onto the pitch in a
# non-degenerate arrangement. **These are deliberately weak.** The original (8 players, 25 m span)
# encoded a wide-shot assumption and voided every zoomed broadcast frame: on the calibration-sick
# SoccerNet-GSR sequences 100% of the dropped frames had *every* player projecting on-pitch, at
# 96-98% precision against ground truth, and were thrown away only for showing 5 players or spanning
# 19 m (results/GSR_CALIBGATE.md). The keypoint-reprojection gate and per-point `clamp_to_pitch` are
# the real filters; this one only catches a projection collapsed into a point.
TRUST_MIN_PLAYERS = 3
TRUST_MIN_SPAN_M = 5.0


def clamp_to_pitch(
    px: float, py: float, *, src_len: float = SRC_LEN, src_wid: float = SRC_WID,
    tol: float = _PITCH_EDGE_TOL_M,
):
    """Clamp a near-line coordinate onto ``[0, len] x [0, wid]``; drop anything past ``tol`` to NaN.

    The tolerance is the projection/foot-point noise a *real* player may show just past a touchline,
    so they are snapped onto the line. Anything beyond it (officials outside the pitch, a bad
    homography) is **rejected**, not clamped -- clamping would plant a phantom player on the touchline,
    exactly the "linesman confused for a player" failure.
    """
    if px is None or py is None or np.isnan(px) or np.isnan(py):
        return float("nan"), float("nan")
    if not (-tol <= px <= src_len + tol and -tol <= py <= src_wid + tol):
        return float("nan"), float("nan")
    return float(np.clip(px, 0.0, src_len)), float(np.clip(py, 0.0, src_wid))


def onpitch_plausible(
    h: np.ndarray,
    image_pts: np.ndarray,
    *,
    min_onpitch: int = TRUST_MIN_PLAYERS,
    min_span_m: float = TRUST_MIN_SPAN_M,
) -> bool:
    """True iff ``h`` projects ``image_pts`` to a distribution :func:`reject_implausible_frames` keeps.

    The same on-pitch/spread test, moved one stage earlier so it can *choose* a homography instead of
    only voiding a frame after the fact. A homography that puts the frame's own player foot-points
    off the pitch (or collapses them to a point) is implausible however well it fits the pitch
    keypoints. Deliberately weak -- see :data:`TRUST_MIN_PLAYERS`.

    Args:
        h: A ``(3, 3)`` image->pitch homography (metres, uncentred 105x68).
        image_pts: ``(N, 2)`` image-space foot points of the frame's *players*.
        min_onpitch: Players that must land on the pitch (after :func:`clamp_to_pitch`).
        min_span_m: Metres those players must span along at least one pitch axis.

    Returns:
        Whether the projection is plausible. Pure: reads no ground truth, only our own detections.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415 - keeps this module pure

    pts = np.asarray(image_pts, dtype=float).reshape(-1, 2)
    if len(pts) < min_onpitch or not np.isfinite(h).all():
        return False
    proj = np.asarray([clamp_to_pitch(x, y) for x, y in apply_homography(h, pts)])
    good = proj[np.isfinite(proj).all(axis=1)]
    if len(good) < min_onpitch:
        return False
    return float(max(np.ptp(good[:, 0]), np.ptp(good[:, 1]))) >= min_span_m


def _players(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that count as on-pitch players (``player`` + ``goalkeeper``); excludes ball and referee.

    Falls back to "everything that isn't the ball" when there is no ``role`` column / only legacy
    roles, so older callers keep working.
    """
    if "role" not in df.columns:
        return df
    if df["role"].isin(PLAYER_ROLES).any():
        return df[df["role"].isin(PLAYER_ROLES)]
    return df[df["role"] != "ball"]  # legacy data: roles are just player/ball


def smooth_tracks(df: pd.DataFrame, *, window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """Rolling-median smooth ``pitch_x``/``pitch_y`` per ``track_id`` (player rows only).

    Returns a new frame; ball rows pass through untouched. NaNs are preserved by ``min_periods=1``
    only smoothing over present values.
    """
    if df.empty:
        return df
    out = df.copy()
    players = out["role"] != "ball"
    sub = out[players].sort_values(["track_id", "frame"])
    for col in ("pitch_x", "pitch_y"):
        smoothed = (
            sub.groupby("track_id")[col]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
        )
        # Only smooth where a value already exists: never fill a NaN (a gate-rejected frame) from its
        # neighbours -- smoothing must not resurrect a rejected coordinate into an accepted one.
        out.loc[sub.index, col] = smoothed.where(sub[col].notna())
    return out


def derive_actor(df: pd.DataFrame, *, max_dist_m: float = ACTOR_MAX_DIST_M) -> pd.DataFrame:
    """Add/refresh an ``is_actor`` column: the player nearest the ball per frame (within threshold).

    A frame with no ball row, or no player within ``max_dist_m`` of it, gets no actor. Returns a new
    frame.
    """
    out = df.copy()
    out["is_actor"] = False
    if out.empty or "role" not in out.columns:
        return out
    ball = out[out["role"] == "ball"].dropna(subset=["pitch_x", "pitch_y"]).set_index("frame")
    players = _players(out)
    for fr, grp in players.groupby("frame"):
        if fr not in ball.index:
            continue
        bx, by = float(ball.loc[fr, "pitch_x"]), float(ball.loc[fr, "pitch_y"])
        valid = grp.dropna(subset=["pitch_x", "pitch_y"])
        if valid.empty:
            continue
        d = np.hypot(valid["pitch_x"] - bx, valid["pitch_y"] - by)
        if float(d.min()) <= max_dist_m:
            out.loc[d.idxmin(), "is_actor"] = True
    return out


def derive_keeper(
    df: pd.DataFrame, *, src_len: float = SRC_LEN, goal_depth: float = KEEPER_GOAL_DEPTH_M
) -> pd.DataFrame:
    """Add/refresh an ``is_keeper`` column, one keeper per team per frame (when one is identifiable).

    Two ways a keeper is set, in order:

    1. **Detected ``goalkeeper`` role** (from a football detector): that team's GK-role player nearest
       its own goal line is the keeper -- the reliable signal.
    2. **Positional fallback** (COCO has no GK class): the team whose mean x is smaller defends the
       left goal, so its deepest (min-x) player is a keeper candidate; the other team's highest (max-x)
       player is the candidate. A candidate is tagged **only if within ``goal_depth`` of its own goal
       line** -- so when the real keeper is off-screen (e.g. a midfield view), the most-extreme
       outfielder is *not* falsely crowned (the old "keeper at halfway" bug).

    Returns a new frame.
    """
    out = df.copy()
    out["is_keeper"] = False
    if out.empty:
        return out
    players = _players(out).dropna(subset=["pitch_x", "pitch_y"])
    has_role = "role" in players.columns
    for _, grp in players.groupby("frame"):
        teams = grp["team"].unique()
        if len(teams) == 0:
            continue
        team_mean_x = {t: float(grp[grp["team"] == t]["pitch_x"].mean()) for t in teams}
        attacking = min(team_mean_x, key=team_mean_x.get)  # smaller mean-x defends the left (x=0) goal
        for t in teams:
            sub = grp[grp["team"] == t]
            own_goal = 0.0 if t == attacking else src_len
            gk = sub[sub["role"] == "goalkeeper"] if has_role else sub.iloc[0:0]
            if len(gk):  # (1) trust the detected GK role
                out.loc[(gk["pitch_x"] - own_goal).abs().idxmin(), "is_keeper"] = True
                continue
            # (2) positional fallback, gated to the penalty-box depth near the team's own goal.
            if t == attacking and float(sub["pitch_x"].min()) <= goal_depth:
                out.loc[sub["pitch_x"].idxmin(), "is_keeper"] = True
            elif t != attacking and float(sub["pitch_x"].max()) >= src_len - goal_depth:
                out.loc[sub["pitch_x"].idxmax(), "is_keeper"] = True
    return out


def _frame_homographies(df: pd.DataFrame, min_donor_rows: int) -> dict[int, np.ndarray]:
    """Recover each frame's image->pitch homography from its own already-projected rows (pure).

    A frame that survived the calibration gate carries >= 4 rows related by exactly one homography,
    so the transform can be re-derived from the positions table alone -- no calibrator, no video.
    """
    from generator.calibrate import estimate_homography  # noqa: PLC0415 - keeps this module pure

    out: dict[int, np.ndarray] = {}
    fin = df[np.isfinite(df["pitch_x"]) & np.isfinite(df["pitch_y"])
             & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    for fr, grp in fin.groupby("frame"):
        if len(grp) < min_donor_rows:
            continue
        h = estimate_homography(grp[["image_x", "image_y"]].to_numpy(),
                                grp[["pitch_x", "pitch_y"]].to_numpy())
        if h is not None and np.isfinite(h).all() and abs(h[2, 2]) > 1e-12:
            out[int(fr)] = h / h[2, 2]
    return out


def _donor_homography(frame: int, donors: list[int], hs: dict[int, np.ndarray],
                      max_gap: int | None) -> np.ndarray | None:
    """Homography for ``frame``: lerp between the bracketing donors, else carry the nearest (pure)."""
    i = np.searchsorted(donors, frame)
    lo = donors[i - 1] if i > 0 else None
    hi = donors[i] if i < len(donors) else None
    if lo is not None and max_gap is not None and frame - lo > max_gap:
        lo = None
    if hi is not None and max_gap is not None and hi - frame > max_gap:
        hi = None
    if lo is None and hi is None:
        return None
    if lo is None:
        return hs[hi]
    if hi is None:
        return hs[lo]
    w = (frame - lo) / (hi - lo)
    h = (1.0 - w) * hs[lo] + w * hs[hi]
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else None


def fill_calibration_gaps(
    df: pd.DataFrame,
    *,
    min_donor_rows: int = MIN_ONPITCH_PLAYERS,
    max_gap: int | None = None,
) -> pd.DataFrame:
    """Re-project rows the calibrator dropped, using a neighbouring frame's homography (pure).

    On SoccerNet-GSR the per-frame calibrator passes its own keypoint-reprojection gate while
    projecting every player off the pitch, so :func:`clamp_to_pitch` (and then
    :func:`reject_implausible_frames`) discard the whole frame: 25% of valid-split frames carry no
    pitch position at all although 84% of the ground-truth players were detected in image space.
    The camera barely moves, so a neighbouring frame's homography recovers them.

    Never overwrites a finite coordinate; filled points go through :func:`clamp_to_pitch` exactly
    like the calibrator's own, so an off-pitch recovery is still rejected.

    Args:
        df: A positions table (needs ``frame``, ``pitch_x/y``, ``image_x/y``).
        min_donor_rows: A frame may donate its homography only with this many projected rows.
        max_gap: Refuse to reach further than this many frames for a donor (``None`` = unlimited).

    Returns:
        A new table with the recoverable pitch coordinates filled in.
    """
    if df.empty or not {"image_x", "image_y"}.issubset(df.columns):
        return df
    hs = _frame_homographies(df, min_donor_rows)
    if not hs:
        return df
    donors = sorted(hs)
    out = df.copy()
    need = (~np.isfinite(out["pitch_x"]) | ~np.isfinite(out["pitch_y"])) & \
        np.isfinite(out["image_x"]) & np.isfinite(out["image_y"])
    for fr, idx in out[need].groupby("frame").groups.items():
        if int(fr) in hs:  # partial frame: the calibrator's own H rejected these points on purpose
            continue
        h = _donor_homography(int(fr), donors, hs, max_gap)
        if h is None:
            continue
        from generator.calibrate import apply_homography  # noqa: PLC0415

        proj = apply_homography(h, out.loc[idx, ["image_x", "image_y"]].to_numpy())
        out.loc[idx, ["pitch_x", "pitch_y"]] = [clamp_to_pitch(x, y) for x, y in proj]
    return out


def reject_implausible_frames(
    df: pd.DataFrame,
    *,
    min_onpitch: int = TRUST_MIN_PLAYERS,
    min_span_m: float = TRUST_MIN_SPAN_M,
) -> pd.DataFrame:
    """NaN the pitch coords of any frame whose on-pitch player distribution is degenerate.

    A frame is trusted only if it has ``>= min_onpitch`` players with valid pitch coords AND those
    players span ``>= min_span_m`` along at least one pitch axis. Otherwise the calibration is
    untrustworthy (a projection collapsed to a point) and we emit no positions for that frame.
    Returns a new frame; ``image_x/image_y`` are kept (detection is fine).

    The thresholds are :data:`TRUST_MIN_PLAYERS` / :data:`TRUST_MIN_SPAN_M`, weakened from the
    original wide-shot pair (8, 25 m) that was voiding correct calibrations on zoomed frames --
    ``results/GSR_CALIBGATE.md``. Pass the old values explicitly to reproduce a pre-2026.08 run.
    """
    if df.empty:
        return df
    out = df.copy()
    players = _players(out)
    for fr, grp in players.groupby("frame"):
        valid = grp.dropna(subset=["pitch_x", "pitch_y"])
        span = max(np.ptp(valid["pitch_x"]), np.ptp(valid["pitch_y"])) if len(valid) else 0.0
        if len(valid) < min_onpitch or span < min_span_m:
            frame_rows = out["frame"] == fr
            out.loc[frame_rows, ["pitch_x", "pitch_y"]] = np.nan
    return out


def postprocess(
    df: pd.DataFrame, *, window: int = SMOOTH_WINDOW, max_dist_m: float = ACTOR_MAX_DIST_M
) -> pd.DataFrame:
    """Full chain: smooth, drop implausibly-calibrated frames, then derive actor + keeper flags."""
    if df.empty:
        return df.assign(is_actor=False, is_keeper=False) if "is_actor" not in df else df
    out = smooth_tracks(df, window=window)
    out = reject_implausible_frames(out)
    out = derive_actor(out, max_dist_m=max_dist_m)
    out = derive_keeper(out)
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)
