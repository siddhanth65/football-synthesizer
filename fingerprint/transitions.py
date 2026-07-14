"""Transition & press-trigger metrics: turnovers, counter-pressing, counter-attacks, high regains.

Matches turn on the moments **around a change of possession**. From the smoothed possession track + the
ball + player positions we read, per team:

* **turnovers** -- every possession switch, located by the ball (who lost it, who won it, where),
* **counter-press** -- after a team *loses* the ball, how fast it swarms the new carrier: the share of
  losses where a team-mate gets within ``press_radius_m`` of the ball inside the reaction window, and the
  mean time to that first pressure (Gegenpressing intensity),
* **counter-attack** -- after a team *wins* the ball, the xT it gains in the window (threat created in
  transition),
* **high regains** -- turnovers won in the attacking third (a press-trigger success signal).

Honest scope: built on the proximity possession proxy + a noisy ball, so read these as *tendencies*
(team A counter-presses harder than B), not exact event counts. Pure numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.structural_metrics import PITCH_LEN, attacking_coord
from fingerprint.xt import geometric_xt

WINDOW_FRAMES = 250     # legacy default only (5 s at 50 fps); production callers MUST pass an fps-derived
                        # window_frames = round(5 * native_fps) so the 5 s counter-press window (StatsBomb
                        # "within 5 s of an open-play turnover") is real-time-correct on 25 vs 59.94 fps.
PRESS_RADIUS_M = 5.0    # a losing-team player this close to the ball = a counter-press pressure
ATT_THIRD_X = 2 * PITCH_LEN / 3  # attacking-third line for "high regain"


def detect_turnovers(possession: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    """Possession switches located by the ball: ``frame, lost_team, won_team, x, y`` (one per turnover)."""
    cols = ["frame", "lost_team", "won_team", "x", "y"]
    if possession.empty or ball.empty:
        return pd.DataFrame(columns=cols)
    ballxy = {int(r.frame): (float(r.x), float(r.y)) for r in ball.itertuples(index=False)}
    p = possession.sort_values("frame").reset_index(drop=True)
    rows = []
    for a, b in zip(p.itertuples(index=False), p.iloc[1:].itertuples(index=False)):
        if int(a.team) == int(b.team):
            continue
        xy = ballxy.get(int(b.frame))
        if xy is None:
            continue
        rows.append({"frame": int(b.frame), "lost_team": int(a.team), "won_team": int(b.team),
                     "x": xy[0], "y": xy[1]})
    return pd.DataFrame(rows, columns=cols)


def transition_metrics(possession: pd.DataFrame, players: pd.DataFrame, ball: pd.DataFrame,
                       directions: dict[int, int], *, window_frames: int = WINDOW_FRAMES,
                       press_radius_m: float = PRESS_RADIUS_M) -> pd.DataFrame:
    """Per-team transition profile: turnovers won/lost, counter-press, counter-attack xT, high regains.

    Returns one row per team: ``team, won, lost, counterpress_rate, mean_recovery_frames, counter_xt,
    high_regains`` (``counterpress_rate`` / ``mean_recovery_frames`` describe the team *after losing* the
    ball; ``counter_xt`` / ``high_regains`` describe it *after winning*).
    """
    cols = ["team", "won", "lost", "counterpress_rate", "mean_recovery_frames", "counter_xt",
            "high_regains"]
    turnovers = detect_turnovers(possession, ball)
    if turnovers.empty:
        return pd.DataFrame(columns=cols)
    ballxy = {int(r.frame): (float(r.x), float(r.y)) for r in ball.sort_values("frame").itertuples(index=False)}
    ball_frames = np.array(sorted(ballxy))
    pbf = {fr: g for fr, g in players.dropna(subset=["pitch_x", "pitch_y"]).groupby("frame")}
    teams = sorted({int(t) for t in possession["team"].unique() if int(t) >= 0})
    acc = {t: {"won": 0, "lost": 0, "pressed": 0, "rec": [], "cxt": [], "high": 0} for t in teams}

    for tv in turnovers.itertuples(index=False):
        won, lost, f = int(tv.won_team), int(tv.lost_team), int(tv.frame)
        if won in acc:
            acc[won]["won"] += 1
        if lost in acc:
            acc[lost]["lost"] += 1
        win = ball_frames[(ball_frames > f) & (ball_frames <= f + window_frames)]
        # counter-press: nearest losing-team player to the ball across the window
        recovery = None
        for wf in win:
            g = pbf.get(int(wf))
            bx, by = ballxy[int(wf)]
            if g is None:
                continue
            d = g[g["team"] == lost]
            if len(d) and float(np.hypot(d["pitch_x"] - bx, d["pitch_y"] - by).min()) <= press_radius_m:
                recovery = int(wf) - f
                break
        if lost in acc and recovery is not None:
            acc[lost]["pressed"] += 1
            acc[lost]["rec"].append(recovery)
        # counter-attack: xT the winner gains over the window (its attacking direction)
        if won in directions and len(win):
            adir = directions[won]
            x0 = attacking_coord(np.array([ballxy[f][0]]), adir)[0] if f in ballxy else \
                attacking_coord(np.array([tv.x]), adir)[0]
            xt0 = float(geometric_xt(np.array([x0]), np.array([tv.y]))[0])
            lf = int(win[-1])
            x1 = attacking_coord(np.array([ballxy[lf][0]]), adir)[0]
            xt1 = float(geometric_xt(np.array([x1]), np.array([ballxy[lf][1]]))[0])
            if won in acc:
                acc[won]["cxt"].append(xt1 - xt0)
        # high regain: won in the winner's attacking third
        if won in directions:
            ac = attacking_coord(np.array([tv.x]), directions[won])[0]
            if ac > ATT_THIRD_X and won in acc:
                acc[won]["high"] += 1

    rows = []
    for t, v in acc.items():
        rows.append({"team": t, "won": v["won"], "lost": v["lost"],
                     "counterpress_rate": v["pressed"] / v["lost"] if v["lost"] else float("nan"),
                     "mean_recovery_frames": float(np.mean(v["rec"])) if v["rec"] else float("nan"),
                     "counter_xt": float(np.mean(v["cxt"])) if v["cxt"] else float("nan"),
                     "high_regains": v["high"]})
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--ball", required=True)
    ap.add_argument("--window", type=int, default=WINDOW_FRAMES)
    ap.add_argument("--team0", default="team 0")
    ap.add_argument("--team1", default="team 1")
    args = ap.parse_args()
    from fingerprint.structural_metrics import resolve_attack_directions  # noqa: PLC0415
    from generator.ball import assign_possession  # noqa: PLC0415

    pos = pd.read_parquet(args.positions)
    players = pos[pos["role"].isin(["player", "goalkeeper"])]
    ball = pd.read_parquet(args.ball)
    poss = assign_possession(ball, players, smooth=True)
    z = transition_metrics(poss, players, ball, resolve_attack_directions(players),
                           window_frames=args.window)
    names = {0: args.team0, 1: args.team1}
    z["team"] = z["team"].map(lambda t: names.get(int(t), f"team {t}"))
    print("Transition & press-trigger profile:\n")
    print(z.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
