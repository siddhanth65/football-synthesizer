"""Pitch-control surface + space-dominance metrics (the spatial backbone), positions + velocity only.

A Spearman/Fernandez-lite model: each player projects a Gaussian influence centred slightly ahead of
them (shifted by velocity x reaction time); a pitch cell is controlled by whichever team's summed
influence is larger. From this we derive per-team **space control** (share of the pitch) and **attacking-
third control** (territory in the dangerous third, direction-normalised) — the foundation the xT-Football-
Club tactical methods sit on, and it needs **no ball**.

Direction is resolved per chunk; velocity comes from :func:`attacker.tracks.build_tracks`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.tracks import DEFAULT_FPS, build_tracks
from fingerprint.structural_metrics import PITCH_LEN, PITCH_WID, resolve_attack_directions

SIGMA_M = 9.0       # influence radius (m)
REACTION_S = 0.5    # velocity look-ahead (s)
ATT_THIRD_X = 2 * PITCH_LEN / 3


def control_field(px, py, team, vx=None, vy=None, *, nx: int = 21, ny: int = 14,
                  sigma: float = SIGMA_M, reaction: float = REACTION_S):
    """P(team 0 controls) on an ``ny x nx`` grid over the pitch (0..1), with optional velocity look-ahead."""
    px, py, team = np.asarray(px, float), np.asarray(py, float), np.asarray(team, int)
    cx = px + (np.nan_to_num(np.asarray(vx, float)) * reaction if vx is not None else 0.0)
    cy = py + (np.nan_to_num(np.asarray(vy, float)) * reaction if vy is not None else 0.0)
    gx, gy = np.linspace(0, PITCH_LEN, nx), np.linspace(0, PITCH_WID, ny)
    gxx, gyy = np.meshgrid(gx, gy)  # [ny, nx]
    inf_a, inf_b = np.zeros_like(gxx), np.zeros_like(gxx)
    for i in range(len(px)):
        inf = np.exp(-((gxx - cx[i]) ** 2 + (gyy - cy[i]) ** 2) / (2 * sigma ** 2))
        if team[i] == 0:
            inf_a += inf
        elif team[i] == 1:
            inf_b += inf
    return inf_a / (inf_a + inf_b + 1e-9), gx


def _frame_space(ctrl: np.ndarray, gx: np.ndarray, dir0: int) -> dict[int, tuple[float, float]]:
    """Per-team (control_share, attacking_third_control) from a control field. dir0 = team-0 attack sign."""
    share0 = float(ctrl.mean())
    att0 = gx > ATT_THIRD_X if dir0 > 0 else gx < (PITCH_LEN - ATT_THIRD_X)  # team-0 attacking third cols
    att1 = ~att0
    a0 = float(ctrl[:, att0].mean()) if att0.any() else float("nan")
    a1 = float((1 - ctrl)[:, att1].mean()) if att1.any() else float("nan")
    return {0: (share0, a0), 1: (1 - share0, a1)}


def space_control_metrics(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                          sample: int = 600) -> pd.DataFrame:
    """Per-team mean space control + attacking-third control over a (sampled) match."""
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [(None, positions)]
    acc: dict[int, dict[str, list]] = {0: {"s": [], "a": []}, 1: {"s": [], "a": []}}
    n_chunks = max(positions["chunk"].nunique(), 1) if "chunk" in positions.columns else 1
    for _, g in groups:
        dirs = resolve_attack_directions(g)
        dir0 = dirs.get(0, 1)
        tr = build_tracks(g, fps=fps)
        frames = np.sort(tr["frame"].unique())
        if sample and len(frames) > sample // n_chunks:
            frames = frames[np.linspace(0, len(frames) - 1, sample // n_chunks).astype(int)]
        for fr in frames:
            fg = tr[tr["frame"] == fr]
            fg = fg[fg["team"].isin([0, 1])]
            if (fg["team"] == 0).sum() < 3 or (fg["team"] == 1).sum() < 3:
                continue
            ctrl, gx = control_field(fg["pitch_x"], fg["pitch_y"], fg["team"], fg["vx"], fg["vy"])
            for team, (share, att) in _frame_space(ctrl, gx, dir0).items():
                acc[team]["s"].append(share)
                acc[team]["a"].append(att)
    rows = [{"team": t, "space_control": float(np.nanmean(v["s"])) if v["s"] else float("nan"),
             "att_third_control": float(np.nanmean(v["a"])) if v["a"] else float("nan"),
             "space_frames": len(v["s"])} for t, v in acc.items() if v["s"]]
    return pd.DataFrame(rows)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()
    z = space_control_metrics(pd.read_parquet(args.positions), sample=args.sample)
    print("Pitch-control space dominance:\n")
    print(z.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
