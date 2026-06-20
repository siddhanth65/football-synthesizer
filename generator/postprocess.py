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

SRC_LEN, SRC_WID = 105.0, 68.0
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
# Frame-calibration plausibility gate: a trustworthy broadcast freeze frame puts a decent number of
# players on the pitch, spread over a real chunk of it. Frames that fit the pitch *keypoints* well but
# still cram players into a corner (sparse/clustered keypoints) fail this and are dropped -- the
# distribution check the keypoint-reprojection gate cannot do.
MIN_ONPITCH_PLAYERS = 8
MIN_PITCH_SPAN_M = 25.0


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
        out.loc[sub.index, col] = (
            sub.groupby("track_id")[col]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
        )
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


def reject_implausible_frames(
    df: pd.DataFrame,
    *,
    min_onpitch: int = MIN_ONPITCH_PLAYERS,
    min_span_m: float = MIN_PITCH_SPAN_M,
) -> pd.DataFrame:
    """NaN the pitch coords of any frame whose on-pitch player distribution is degenerate.

    A frame is trusted only if it has ``>= min_onpitch`` players with valid pitch coords AND those
    players span ``>= min_span_m`` along at least one pitch axis. Otherwise the calibration is
    untrustworthy (e.g. sparse/clustered keypoints projecting everyone into a corner) and we emit no
    positions for that frame. Returns a new frame; ``image_x/image_y`` are kept (detection is fine).
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
