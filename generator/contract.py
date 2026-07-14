"""The unified, substrate-aware freeze-frame contract (novelty C1).

Every observation substrate -- StatsBomb 360 (sparse, no IDs), broadcast CV (dense, ID-persistent,
oriented) and free tracking -- is normalised to ONE :class:`FreezeFrame` on a 120x80 pitch with the
attacking team playing left -> right. Each player node carries the *superset* of features plus a
*per-feature observability mask*: features a substrate cannot provide (e.g. velocity in 360, body
orientation when pose fails) are **masked, not zero-filled**, so a downstream model knows what it
cannot see.

The frame also **down-projects** to the exact ``(x, y, teammate, actor, keeper)`` shape the already
trained GAT consumes (``football-state-of-play`` ``eval/ood_demo.predict_from_freeze_frame`` and
``data/graphs.build_data``), so the existing model runs on generator output immediately.

Pure: depends only on numpy/pandas + stdlib. No torch, no CV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# StatsBomb convention (matches football-state-of-play)
from core.pitch import CONTRACT_LEN as PITCH_LENGTH, CONTRACT_WID as PITCH_WIDTH
_PITCH_DIAG = math.hypot(PITCH_LENGTH, PITCH_WIDTH)
_VELOCITY_NORM = 10.0  # m/s, for feature scaling


class Substrate(str, Enum):
    """Where a freeze-frame came from (drives the observability mask)."""

    SB360 = "statsbomb_360"
    BROADCAST_CV = "broadcast_cv"
    TRACKING = "tracking"


# Ordered node-feature names produced by :func:`node_feature_matrix` (and their mask).
FEATURE_NAMES: tuple[str, ...] = (
    "x_norm",
    "y_norm",
    "vx_norm",
    "vy_norm",
    "sin_orientation",
    "cos_orientation",
    "is_teammate",
    "is_actor",
    "is_keeper",
    "dist_to_ball_norm",
    "observed",
)


@dataclass
class PlayerNode:
    """One visible-or-imputed player on the pitch (120x80, attacking left -> right).

    Args:
        x: Pitch x in metres ([0, 120]).
        y: Pitch y in metres ([0, 80]).
        is_teammate: True if on the attacking team.
        is_actor: True if this player is on the ball.
        is_keeper: True if a goalkeeper.
        vx: Velocity x (m/s) or ``None`` if unobserved (e.g. StatsBomb 360).
        vy: Velocity y (m/s) or ``None``.
        orientation: Body orientation in radians (0 = facing +x / attacking goal), or ``None``.
        observed: True if directly detected; False if imputed by off-screen completion (C2).
    """

    x: float
    y: float
    is_teammate: bool = True
    is_actor: bool = False
    is_keeper: bool = False
    vx: float | None = None
    vy: float | None = None
    orientation: float | None = None
    observed: bool = True

    @property
    def has_velocity(self) -> bool:
        """True if velocity is known."""
        return self.vx is not None and self.vy is not None

    @property
    def has_orientation(self) -> bool:
        """True if body orientation is known."""
        return self.orientation is not None


@dataclass
class FreezeFrame:
    """A full freeze-frame: players + ball + graph-level context, tagged by substrate.

    Args:
        players: Visible (and optionally imputed) players. Must be non-empty.
        substrate: The source substrate (drives masks).
        ball: ``(x, y)`` of the ball, or ``None`` if not located.
        play_pattern: StatsBomb-style play pattern (graph-level context).
        from_counter: True for ``play_pattern == "From Counter"``.
        trigger_time_s: Period-relative seconds (graph-level context).
    """

    players: list[PlayerNode]
    substrate: Substrate
    ball: tuple[float, float] | None = None
    play_pattern: str = "Regular Play"
    from_counter: bool = False
    trigger_time_s: float = 600.0

    def __post_init__(self) -> None:
        if not self.players:
            raise ValueError("FreezeFrame requires at least one player")

    @property
    def n_players(self) -> int:
        """Number of player nodes."""
        return len(self.players)


# --- construction helpers ------------------------------------------------------------------------
def rescale_xy(x: float, y: float, src_len: float, src_wid: float) -> tuple[float, float]:
    """Rescale a point from a ``src_len`` x ``src_wid`` pitch to the 120x80 StatsBomb pitch."""
    return x * PITCH_LENGTH / src_len, y * PITCH_WIDTH / src_wid


def from_statsbomb(
    freeze_frame: list[dict],
    *,
    play_pattern: str = "Regular Play",
    from_counter: bool = False,
    trigger_time_s: float = 600.0,
    ball: tuple[float, float] | None = None,
) -> FreezeFrame:
    """Build a :class:`FreezeFrame` from a StatsBomb 360 ``freeze_frame`` list.

    Velocity and orientation are left unobserved (360 provides neither).

    Args:
        freeze_frame: List of ``{location: [x, y], teammate, actor, keeper}`` dicts.
        play_pattern: Graph-level context.
        from_counter: Counter-attack flag.
        trigger_time_s: Period-relative seconds.
        ball: Optional ball location.

    Returns:
        A 360-substrate :class:`FreezeFrame`.
    """
    players = [
        PlayerNode(
            x=float(p["location"][0]),
            y=float(p["location"][1]),
            is_teammate=bool(p.get("teammate", True)),
            is_actor=bool(p.get("actor", False)),
            is_keeper=bool(p.get("keeper", False)),
        )
        for p in freeze_frame
    ]
    return FreezeFrame(
        players=players,
        substrate=Substrate.SB360,
        ball=ball,
        play_pattern=play_pattern,
        from_counter=from_counter,
        trigger_time_s=trigger_time_s,
    )


def orient_left_to_right(frame: FreezeFrame) -> FreezeFrame:
    """Return a new frame flipped so the attacking team (teammates) plays towards x = 120.

    Mirrors ``cv_bridge._orient`` but also flips velocity and body orientation. A flip of both axes
    is a 180-degree rotation about the pitch centre, so velocities negate and orientation rotates pi.
    """
    tm_x = [p.x for p in frame.players if p.is_teammate]
    if not tm_x or float(np.mean(tm_x)) >= PITCH_LENGTH / 2:
        return frame  # already attacking left -> right (or no teammates to orient by)

    flipped: list[PlayerNode] = []
    for p in frame.players:
        flipped.append(
            PlayerNode(
                x=PITCH_LENGTH - p.x,
                y=PITCH_WIDTH - p.y,
                is_teammate=p.is_teammate,
                is_actor=p.is_actor,
                is_keeper=p.is_keeper,
                vx=None if p.vx is None else -p.vx,
                vy=None if p.vy is None else -p.vy,
                orientation=None if p.orientation is None else (p.orientation + math.pi) % (2 * math.pi),
                observed=p.observed,
            )
        )
    ball = None if frame.ball is None else (PITCH_LENGTH - frame.ball[0], PITCH_WIDTH - frame.ball[1])
    return FreezeFrame(
        players=flipped,
        substrate=frame.substrate,
        ball=ball,
        play_pattern=frame.play_pattern,
        from_counter=frame.from_counter,
        trigger_time_s=frame.trigger_time_s,
    )


# --- model bridges -------------------------------------------------------------------------------
def to_model_frame(frame: FreezeFrame) -> list[dict]:
    """Down-project to the ``predict_from_freeze_frame`` input (the trained GAT's contract)."""
    return [
        {
            "x": p.x,
            "y": p.y,
            "teammate": p.is_teammate,
            "actor": p.is_actor,
            "keeper": p.is_keeper,
        }
        for p in frame.players
    ]


def to_statsbomb_dataframe(frame: FreezeFrame) -> pd.DataFrame:
    """Down-project to the ``cv_bridge.frame_to_statsbomb`` output (for ``data.graphs.build_data``)."""
    return pd.DataFrame(
        {
            "location": [[p.x, p.y] for p in frame.players],
            "teammate": [p.is_teammate for p in frame.players],
            "actor": [p.is_actor for p in frame.players],
            "keeper": [p.is_keeper for p in frame.players],
        }
    )


def node_feature_matrix(frame: FreezeFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the masked node-feature matrix for the multi-substrate model (C1).

    Returns:
        ``(X, M)`` each ``[n_players, len(FEATURE_NAMES)]``. ``X`` holds features (unobserved entries
        are 0); ``M`` is 1.0 where a feature is observed/valid and 0.0 where it is masked.
    """
    n, d = frame.n_players, len(FEATURE_NAMES)
    x = np.zeros((n, d), dtype=np.float32)
    m = np.ones((n, d), dtype=np.float32)
    bx, by = frame.ball if frame.ball is not None else (None, None)
    for i, p in enumerate(frame.players):
        x[i, 0] = p.x / PITCH_LENGTH
        x[i, 1] = p.y / PITCH_WIDTH
        if p.has_velocity:
            x[i, 2] = p.vx / _VELOCITY_NORM
            x[i, 3] = p.vy / _VELOCITY_NORM
        else:
            m[i, 2] = m[i, 3] = 0.0
        if p.has_orientation:
            x[i, 4] = math.sin(p.orientation)
            x[i, 5] = math.cos(p.orientation)
        else:
            m[i, 4] = m[i, 5] = 0.0
        x[i, 6] = float(p.is_teammate)
        x[i, 7] = float(p.is_actor)
        x[i, 8] = float(p.is_keeper)
        if bx is not None:
            x[i, 9] = math.hypot(p.x - bx, p.y - by) / _PITCH_DIAG
        else:
            m[i, 9] = 0.0
        x[i, 10] = float(p.observed)
    return x, m
