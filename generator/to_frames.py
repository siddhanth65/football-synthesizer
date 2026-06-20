"""Positions table -> quality-gated :class:`FreezeFrame` objects.

Consumes the ``cv-football`` / tracking positions schema
(``frame, track_id, role, team, pitch_x, pitch_y, is_actor`` on a 105x68 pitch) and yields contract
frames on the 120x80 pitch, oriented attacking left -> right. Mirrors the logic in
``football-state-of-play`` ``eval/cv_bridge.frame_to_statsbomb`` but emits the richer contract
(velocity/orientation are passed through when present, masked otherwise) and applies the wide-shot
gate (>= ``min_players``) plus an optional frame-quality gate.

This is the seam between the CV generator and the trained relational model.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from generator.contract import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    FreezeFrame,
    PlayerNode,
    Substrate,
    orient_left_to_right,
)

SRC_LEN, SRC_WID = 105.0, 68.0  # cv-football / kloppy pitch convention
MIN_PLAYERS = 10  # mirror StatsBomb 360's >=10-visible wide-shot filter


def _attacking_team(team: np.ndarray, px: np.ndarray, actor: np.ndarray) -> int:
    """Attacking team = the ball-carrier's team if tagged, else the more-advanced team."""
    if actor.any():
        return int(team[actor][0])
    means = {int(t): float(px[team == t].mean()) for t in np.unique(team)}
    return max(means, key=means.get)


def frame_from_positions(
    grp: pd.DataFrame,
    *,
    substrate: Substrate = Substrate.BROADCAST_CV,
    min_players: int = MIN_PLAYERS,
) -> FreezeFrame | None:
    """Convert one frame's positions rows to a contract :class:`FreezeFrame`, or ``None`` if sparse.

    Args:
        grp: Rows for a single ``frame`` (players + optional ``role == "ball"`` row).
        substrate: Source substrate tag.
        min_players: Minimum visible players to accept the frame.

    Returns:
        An oriented :class:`FreezeFrame`, or ``None`` if fewer than ``min_players`` players.
    """
    players_df = grp[grp["role"] != "ball"].dropna(subset=["pitch_x", "pitch_y"])
    if len(players_df) < min_players:
        return None
    px = players_df["pitch_x"].to_numpy() * PITCH_LENGTH / SRC_LEN
    py = players_df["pitch_y"].to_numpy() * PITCH_WIDTH / SRC_WID
    team = players_df["team"].to_numpy()
    actor = (
        players_df["is_actor"].to_numpy().astype(bool)
        if "is_actor" in players_df.columns
        else np.zeros(len(players_df), bool)
    )
    atk = _attacking_team(team, px, actor)
    teammate = team == atk

    # Keeper heuristic: each team's most extreme player along the attack axis.
    keeper = np.zeros(len(players_df), bool)
    if teammate.any():
        keeper[np.where(teammate)[0][int(np.argmin(px[teammate]))]] = True
    if (~teammate).any():
        keeper[np.where(~teammate)[0][int(np.argmax(px[~teammate]))]] = True

    has_v = {"vx", "vy"}.issubset(players_df.columns)
    vx = players_df["vx"].to_numpy() if has_v else [None] * len(players_df)
    vy = players_df["vy"].to_numpy() if has_v else [None] * len(players_df)

    players = [
        PlayerNode(
            x=float(px[i]),
            y=float(py[i]),
            is_teammate=bool(teammate[i]),
            is_actor=bool(actor[i]),
            is_keeper=bool(keeper[i]),
            vx=None if vx[i] is None else float(vx[i]),
            vy=None if vy[i] is None else float(vy[i]),
        )
        for i in range(len(players_df))
    ]

    ball_rows = grp[grp["role"] == "ball"].dropna(subset=["pitch_x", "pitch_y"])
    ball = None
    if len(ball_rows):
        ball = (
            float(ball_rows.iloc[0]["pitch_x"] * PITCH_LENGTH / SRC_LEN),
            float(ball_rows.iloc[0]["pitch_y"] * PITCH_WIDTH / SRC_WID),
        )
    return orient_left_to_right(FreezeFrame(players=players, substrate=substrate, ball=ball))


def frames_from_positions(
    positions: pd.DataFrame, *, substrate: Substrate = Substrate.BROADCAST_CV
) -> Iterator[tuple[int, FreezeFrame]]:
    """Yield ``(frame_id, FreezeFrame)`` for every qualifying frame in a positions table."""
    for fr, grp in positions.groupby("frame"):
        frame = frame_from_positions(grp, substrate=substrate)
        if frame is not None:
            yield int(fr), frame
