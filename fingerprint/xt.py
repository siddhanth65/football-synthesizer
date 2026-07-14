"""Expected-threat (xT) surface + controlled-threat metrics (OBSO-lite), and ball-xT progression.

Two complementary reads of *danger*, built on the pitch-control backbone:

* **Controlled threat** (no ball needed). A geometric xT value per pitch cell (rises toward the goal,
  concentrated centrally) weighted by **who controls that cell** (``fingerprint.pitch_control``) gives
  each team's *off-ball* threat — where it manufactures danger by dominating dangerous space. This is the
  spatial term of Spearman's OBSO without the ball/finishing model.
* **Ball xT progression** (needs a ball track). The xT of the ball's location over time; summing the
  *positive* deltas while a team holds the ball measures how much threat that team **creates by moving the
  ball** — a possession-value / progression metric.

The grid value is a **geometric xT proxy** (distance + centrality to goal), not the learned Karun-Singh
grid — honest, reproducible, and good enough to rank locations. Direction is keeper-resolved per chunk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.tracks import DEFAULT_FPS, build_tracks
from fingerprint.pitch_control import control_field
from fingerprint.structural_metrics import PITCH_LEN, PITCH_WID, resolve_attack_directions

XT_DECAY_M = 18.0  # threat falls off with this length-scale (m) from the goal mouth


def geometric_xt(ax: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Geometric xT value in attacking coordinates (``ax``: 0 = own goal, 105 = attacking goal).

    ``exp(-distance_to_goal_mouth / XT_DECAY_M)`` -> ~1 at the goal, ~0 in the own half, with central
    locations valued above wide ones (the goal mouth is central). Returns values in (0, 1].
    """
    ax = np.asarray(ax, float)
    y = np.asarray(y, float)
    d = np.hypot(PITCH_LEN - ax, y - PITCH_WID / 2)
    return np.exp(-d / XT_DECAY_M)


def xt_grids(gx: np.ndarray, gy: np.ndarray, dir0: int) -> tuple[np.ndarray, np.ndarray]:
    """xT grids ``[ny, nx]`` for team 0 (attack sign ``dir0``) and team 1 (opposite), on grid ``gx, gy``."""
    gxx, gyy = np.meshgrid(gx, gy)
    ax0 = gxx if dir0 > 0 else (PITCH_LEN - gxx)          # team-0 attacking-x
    xt0 = geometric_xt(ax0, gyy)
    xt1 = geometric_xt(PITCH_LEN - ax0, gyy)              # team 1 attacks the other way
    return xt0, xt1


def controlled_threat_metrics(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS, sample: int = 600,
                              nx: int = 21, ny: int = 14) -> pd.DataFrame:
    """Per-team mean **controlled threat** = sum over the pitch of control-share x xT, sampled over a match.

    Returns ``team, controlled_threat, threat_share, frames`` where ``threat_share`` is a team's share of
    the total controlled threat that frame (0..1) -- a normalised "who owns the danger" number.
    """
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [(None, positions)]
    n_chunks = max(positions["chunk"].nunique(), 1) if "chunk" in positions.columns else 1
    acc: dict[int, dict[str, list]] = {0: {"t": [], "sh": []}, 1: {"t": [], "sh": []}}
    gx, gy = np.linspace(0, PITCH_LEN, nx), np.linspace(0, PITCH_WID, ny)
    for _, g in groups:
        dir0 = resolve_attack_directions(g).get(0, 1)
        xt0, xt1 = xt_grids(gx, gy, dir0)
        tr = build_tracks(g, fps=fps)
        frames = np.sort(tr["frame"].unique())
        if sample and len(frames) > sample // n_chunks:
            frames = frames[np.linspace(0, len(frames) - 1, sample // n_chunks).astype(int)]
        for fr in frames:
            fg = tr[(tr["frame"] == fr) & tr["team"].isin([0, 1])]
            if (fg["team"] == 0).sum() < 3 or (fg["team"] == 1).sum() < 3:
                continue
            ctrl, _ = control_field(fg["pitch_x"], fg["pitch_y"], fg["team"], fg["vx"], fg["vy"],
                                    nx=nx, ny=ny)
            t0 = float((ctrl * xt0).sum())
            t1 = float(((1 - ctrl) * xt1).sum())
            tot = t0 + t1 + 1e-9
            acc[0]["t"].append(t0); acc[0]["sh"].append(t0 / tot)
            acc[1]["t"].append(t1); acc[1]["sh"].append(t1 / tot)
    rows = [{"team": t, "controlled_threat": float(np.mean(v["t"])) if v["t"] else float("nan"),
             "threat_share": float(np.mean(v["sh"])) if v["sh"] else float("nan"),
             "frames": len(v["t"])} for t, v in acc.items() if v["t"]]
    return pd.DataFrame(rows)


def ball_xt_timeline(ball: pd.DataFrame, possession: pd.DataFrame,
                     directions: dict[int, int]) -> pd.DataFrame:
    """Per-team threat **created by moving the ball**: summed positive xT deltas while the team holds it.

    Args:
        ball: linked ball track ``frame, x, y`` (pitch metres).
        possession: ``frame, carrier, team`` (the holder per frame).
        directions: attacking direction per team (keeper-resolved).

    Returns:
        ``team, xt_created, n_moves`` -- ``xt_created`` sums only the positive xT gains (progression toward
        goal) credited to the team in possession; ``n_moves`` is how many ball samples contributed.
    """
    cols = ["team", "xt_created", "n_moves"]
    if ball.empty or possession.empty:
        return pd.DataFrame(columns=cols)
    poss_team = {int(r.frame): int(r.team) for r in possession.itertuples(index=False)}
    b = ball.sort_values("frame").reset_index(drop=True)
    acc: dict[int, list[float]] = {}
    prev = None
    for r in b.itertuples(index=False):
        team = poss_team.get(int(r.frame))
        if team is not None and team in directions:
            ax = r.x if directions[team] > 0 else (PITCH_LEN - r.x)
            xt = float(geometric_xt(np.array([ax]), np.array([r.y]))[0])
            if prev is not None and prev[0] == team:
                delta = xt - prev[1]
                if delta > 0:
                    acc.setdefault(team, []).append(delta)
            prev = (team, xt)
        else:
            prev = None
    return pd.DataFrame([{"team": t, "xt_created": float(np.sum(v)), "n_moves": len(v)}
                         for t, v in sorted(acc.items())], columns=cols)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--ball", default=None, help="linked ball track -> also report ball-xT progression")
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    z = controlled_threat_metrics(pos, sample=args.sample)
    print("Controlled threat (space x geometric xT):\n")
    print(z.round(3).to_string(index=False))
    if args.ball:
        from generator.ball import assign_possession  # noqa: PLC0415

        ball = pd.read_parquet(args.ball)
        poss = assign_possession(ball, pos, smooth=True)
        bt = ball_xt_timeline(ball, poss, resolve_attack_directions(pos))
        print("\nBall xT created (progression by moving the ball):\n")
        print(bt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
