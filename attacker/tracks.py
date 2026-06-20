"""Dense, ID-persistent attacker tracks from the positions table.

Turns persistent track ids into per-player trajectories with smoothed velocity (and orientation when
available) -- the dense temporal signal StatsBomb 360 lacks. Feeds real per-player run targets and
receiver labels in :mod:`attacker.labels`.
"""

from __future__ import annotations

import pandas as pd


def build_tracks(positions: pd.DataFrame, *, fps: float = 25.0, smooth_window: int = 5) -> pd.DataFrame:
    """Return per-(track_id, frame) rows with smoothed ``vx, vy`` (Savitzky-Golay). TODO."""
    raise NotImplementedError("TODO: per-track smoothing + velocity from persistent ids")
