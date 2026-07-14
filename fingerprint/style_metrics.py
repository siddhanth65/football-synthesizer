"""Two cheap, interpretable team-style axes from positions + velocity (no ball).

* **velocity synchrony** ("Team Mind") -- how aligned a team's players move (resultant length of their
  unit-velocity vectors, 0 = scattered, 1 = marching in lockstep); the corpus flags movement synchrony
  as the strongest chance-creation predictor.
* **style distance** ("Tactical DNA") -- a 1-D Wasserstein (earth-mover) distance between two teams'
  attack-normalised position marginals; a single number for *how differently two teams occupy space*.
  For one match it contrasts the two sides; the same function compares any two teams across matches
  (the seed for opponent matchups / C5).

Direction is resolved per chunk; velocity from :func:`attacker.tracks.build_tracks`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.tracks import DEFAULT_FPS, build_tracks
from fingerprint.structural_metrics import attacking_coord, resolve_attack_directions

MIN_SPEED_MS = 0.5   # ignore (near-)stationary players in the synchrony resultant
MIN_TEAM = 3


def velocity_synchrony(tracks: pd.DataFrame, *, min_speed: float = MIN_SPEED_MS,
                       min_team: int = MIN_TEAM) -> pd.DataFrame:
    """Per-team mean resultant length of moving players' unit-velocity vectors (0..1)."""
    pl = tracks.dropna(subset=["vx", "vy"])
    acc: dict[int, list] = {}
    for (_, team), g in pl.groupby(["frame", "team"]):
        if int(team) < 0:
            continue
        v = g[["vx", "vy"]].to_numpy()
        spd = np.hypot(v[:, 0], v[:, 1])
        v = v[spd > min_speed]
        if len(v) < min_team:
            continue
        u = v / np.hypot(v[:, 0], v[:, 1])[:, None]
        acc.setdefault(int(team), []).append(float(np.hypot(u[:, 0].mean(), u[:, 1].mean())))
    return pd.DataFrame([{"team": t, "velocity_synchrony": float(np.mean(v))} for t, v in acc.items()])


def team_synchrony(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Velocity synchrony per team over a multi-chunk match (builds tracks per chunk)."""
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [(None, positions)]
    parts = [velocity_synchrony(build_tracks(g, fps=fps)) for _, g in groups]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).groupby("team", as_index=False)["velocity_synchrony"].mean()


def style_distance(positions: pd.DataFrame) -> float:
    """1-D Wasserstein style distance between the two teams' attack-normalised position marginals (m)."""
    from scipy.stats import wasserstein_distance  # noqa: PLC0415

    groups = positions.groupby("chunk") if "chunk" in positions.columns else [(None, positions)]
    xs: dict[int, list] = {0: [], 1: []}
    ys: dict[int, list] = {0: [], 1: []}
    for _, g in groups:
        dirs = resolve_attack_directions(g)
        pl = g[g["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
        for team, gg in pl.groupby("team"):
            d = dirs.get(int(team))
            if int(team) not in (0, 1) or d is None:
                continue
            xs[int(team)].extend(attacking_coord(gg["pitch_x"].to_numpy(), d))
            ys[int(team)].extend(gg["pitch_y"].to_numpy())
    if not xs[0] or not xs[1]:
        return float("nan")
    return float(wasserstein_distance(xs[0], xs[1]) + wasserstein_distance(ys[0], ys[1]))
