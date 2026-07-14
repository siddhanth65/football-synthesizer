"""Dense positions table -> one analysis-ready **tactical-frame** row per accepted frame (Pass D/E).

The positions parquet is the dense per-detection table (one row per player/ball per frame). The metric
engine and report want a *per-frame* summary: how many players are on the pitch, each team's shape
(centroid, width, depth), where the ball and ball-carrier are, and how trustworthy the frame is. This
module flattens the dense table into that tactical-frame table.

It records spatial **facts** only. Attacking *direction* is deliberately NOT inferred from mean
x-position (a single frame can't tell which way a team attacks -- the review is explicit about this);
direction is resolved later from match-half metadata + possession continuity.

Pure: numpy/pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN as SRC_LEN, PITCH_WID as SRC_WID
MIN_PLAYERS = 10  # a frame needs this many on-pitch players to be a usable tactical frame
DEFAULT_FPS = 50.0

TACTICAL_COLUMNS = (
    "frame", "time_s", "accepted", "n_onpitch", "n_team0", "n_team1",
    "has_ball", "ball_x", "ball_y", "actor_team", "actor_track_id",
    "team0_cx", "team0_cy", "team0_xspan", "team0_yspan",
    "team1_cx", "team1_cy", "team1_xspan", "team1_yspan",
    "mean_calib_error_m",
)


def _team_shape(px: np.ndarray, py: np.ndarray) -> dict:
    """Centroid + extent (depth along x, width along y) of a set of player points."""
    if len(px) == 0:
        return {"cx": np.nan, "cy": np.nan, "xspan": np.nan, "yspan": np.nan}
    return {
        "cx": float(px.mean()), "cy": float(py.mean()),
        "xspan": float(np.ptp(px)), "yspan": float(np.ptp(py)),
    }


def build_tactical_frame(grp: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                         min_players: int = MIN_PLAYERS) -> dict:
    """Summarise one frame's positions rows into a single tactical-frame record."""
    fr = int(grp["frame"].iloc[0])
    players = grp[grp["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    rec = dict.fromkeys(TACTICAL_COLUMNS, np.nan)
    rec["frame"] = fr
    rec["time_s"] = fr / fps
    rec["n_onpitch"] = int(len(players))
    rec["mean_calib_error_m"] = (
        float(grp["calib_error_m"].dropna().mean()) if "calib_error_m" in grp else np.nan
    )
    if len(players) < min_players:
        rec["accepted"] = False
        return rec
    rec["accepted"] = True

    team = players["team"].to_numpy()
    px, py = players["pitch_x"].to_numpy(), players["pitch_y"].to_numpy()
    for t in (0, 1):
        sel = team == t
        rec[f"n_team{t}"] = int(sel.sum())
        shape = _team_shape(px[sel], py[sel])
        for k, v in shape.items():
            rec[f"team{t}_{k}"] = v

    if "is_actor" in players.columns and players["is_actor"].any():
        a = players[players["is_actor"]].iloc[0]
        rec["actor_team"] = int(a["team"])
        rec["actor_track_id"] = int(a["track_id"])

    ball = grp[grp["role"] == "ball"].dropna(subset=["pitch_x", "pitch_y"])
    rec["has_ball"] = bool(len(ball))
    if len(ball):
        rec["ball_x"] = float(ball.iloc[0]["pitch_x"])
        rec["ball_y"] = float(ball.iloc[0]["pitch_y"])
    return rec


def build_tactical_table(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                         min_players: int = MIN_PLAYERS) -> pd.DataFrame:
    """Build the per-frame tactical-frame table from a dense positions table."""
    if positions.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in TACTICAL_COLUMNS})
    rows = [build_tactical_frame(grp, fps=fps, min_players=min_players)
            for _, grp in positions.groupby("frame")]
    return pd.DataFrame(rows, columns=list(TACTICAL_COLUMNS)).sort_values("frame").reset_index(drop=True)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="dense positions parquet")
    ap.add_argument("--out", required=True, help="output tactical-frame parquet")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    tab = build_tactical_table(pos, fps=args.fps)
    tab.to_parquet(args.out, index=False)
    acc = int(tab["accepted"].sum()) if len(tab) else 0
    print(f"tactical table: {len(tab)} frames, {acc} accepted -> {args.out}")


if __name__ == "__main__":
    main()
