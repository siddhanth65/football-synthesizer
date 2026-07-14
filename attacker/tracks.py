"""Dense, ID-persistent attacker tracks from the positions table.

Turns persistent track ids into per-player trajectories with smoothed position + velocity -- the dense
temporal signal StatsBomb 360 lacks. Velocity comes from a Savitzky-Golay-smoothed position series
(falls back to raw for short tracks), differentiated against real time (frame / fps), with velocity
**nulled across large gaps** so a re-acquired track doesn't fabricate a teleport speed. Feeds the real
per-player run targets and receiver labels in :mod:`attacker.labels`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAYER_ROLES = ("player", "goalkeeper")
DEFAULT_FPS = 50.0
MAX_GAP_STEPS = 3  # velocity across a gap longer than this many sample-steps is left NaN (re-acquired id)
SPEED_CAP_MS = 10.0  # no footballer sustains >10 m/s; faster = tracking jitter / ID switch -> clip


def _savgol(a: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth; gracefully shrinks the window (or no-ops) for short tracks."""
    a = np.asarray(a, float)
    n = len(a)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if n < 5 or w < 3 or w <= poly:
        return a
    from scipy.signal import savgol_filter  # noqa: PLC0415

    return savgol_filter(a, w, poly)


def build_tracks(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                 smooth_window: int = 5) -> pd.DataFrame:
    """Per-``(track_id, frame)`` rows for outfield+GK players with smoothed ``x_s, y_s, vx, vy, speed``.

    Args:
        positions: Dense positions table (``frame, track_id, role, team, pitch_x, pitch_y``...).
        fps: Source frame rate (``frame`` is the source-frame index, so dt = frame_gap / fps).
        smooth_window: Savitzky-Golay window (samples) for the position series.

    Returns:
        The input player rows (sorted by track then frame) plus smoothed position/velocity columns.
        Velocity is NaN at the start of each track and across gaps longer than ``MAX_GAP_STEPS``.
    """
    pl = positions[positions["role"].isin(PLAYER_ROLES)].dropna(subset=["pitch_x", "pitch_y"]).copy()
    if pl.empty:
        for c in ("x_s", "y_s", "vx", "vy", "speed"):
            pl[c] = pd.Series(dtype=float)
        return pl
    step = int(np.median(np.diff(np.sort(pl["frame"].unique())))) or 1
    out = []
    for _, g in pl.sort_values("frame").groupby("track_id", sort=False):
        g = g.copy()
        fr = g["frame"].to_numpy()
        xs, ys = _savgol(g["pitch_x"].to_numpy(), smooth_window), _savgol(g["pitch_y"].to_numpy(),
                                                                          smooth_window)
        if len(g) >= 2:
            t = fr / fps
            vx, vy = np.gradient(xs, t), np.gradient(ys, t)
            big = np.diff(fr, prepend=fr[0]) > MAX_GAP_STEPS * step  # spacing since previous sample
            vx[0] = vy[0] = np.nan  # no backward sample at a track's first frame
            vx[big], vy[big] = np.nan, np.nan
            spd = np.hypot(vx, vy)
            hot = spd > SPEED_CAP_MS  # clip physically impossible speeds (jitter / id switch)
            scale = np.where(hot, SPEED_CAP_MS / np.where(spd > 0, spd, 1.0), 1.0)
            vx, vy = vx * scale, vy * scale
        else:
            vx = vy = np.full(len(g), np.nan)
        g["x_s"], g["y_s"], g["vx"], g["vy"] = xs, ys, vx, vy
        g["speed"] = np.hypot(vx, vy)
        out.append(g)
    return pd.concat(out).reset_index(drop=True)
