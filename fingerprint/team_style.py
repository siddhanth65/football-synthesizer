"""Deterministic team-style fingerprint vector ``z_T``, structured to mirror the FIFA EFI report.

Built from positions + dense tracks only (no learned model), so it is reproducible and testable. Each
axis maps to a family in the FIFA Post-Match Summary (the project's oracle) that we can derive from
broadcast tracking:

* **Shape & territory** -> width, length, compactness, convex-hull surface area, five-lane occupation
  (FIFA "In Possession Line Height & Team Length"; attack-channel split).
* **Verticality** -> build-up height, defensive-line height, attacking-third share (FIFA "Line Height",
  "Defensive Line Height & Team Length").
* **Physical (rough)** -> speed-zone time share, sprint share, top speed (FIFA "Physical Data"). Broadcast
  tracking is partial and metre-scale-noisy, so only *fractions / maxima* are used (not absolute distance),
  and these are flagged rough.

Ball/event-dependent FIFA families -- possession %, phases of play, line breaks, passing networks,
pressures, offers/movement-to-receive -- need reliable ball tracking and are **deferred** (see
``docs/EFI_ALIGNMENT.md``). This vector is the deterministic core of ``z_T``; the relational-GAT
augmentation lives in :mod:`fingerprint.team_identity`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.tracks import DEFAULT_FPS, build_tracks
from fingerprint.structural_metrics import (
    compute_metrics_table,
    resolve_attack_directions,
    team_fingerprint,
)

KMH = 3.6  # m/s -> km/h
# FIFA physical speed zones (km/h): walk / jog / run / high-speed / sprint.
SPEED_ZONE_EDGES_KMH = (0.0, 7.0, 15.0, 20.0, 25.0, 1000.0)
SPEED_ZONE_NAMES = ("time_walk_z1", "time_jog_z2", "time_run_z3", "time_hispeed_z4", "time_sprint_z5")
SPRINT_KMH = 20.0  # zone 4+ = a sprint


def physical_profile(tracks: pd.DataFrame) -> pd.DataFrame:
    """Rough per-team physical style from track speeds (one row/team).

    Partial broadcast tracking biases absolute distance, so we report **time-share by FIFA speed zone**,
    **sprint share**, and a robust **top speed** (99th pct) instead. ``tracks`` must have ``team`` and
    ``speed`` (m/s; produced by :func:`attacker.tracks.build_tracks`).
    """
    rows = []
    for team, g in tracks.groupby("team"):
        spd = g["speed"].dropna().to_numpy() * KMH
        if int(team) < 0 or len(spd) == 0:
            continue
        frac = np.histogram(spd, bins=SPEED_ZONE_EDGES_KMH)[0] / len(spd)
        rec = {"team": int(team), "top_speed_kmh": float(np.percentile(spd, 99)),
               "sprint_share": float(np.mean(spd >= SPRINT_KMH)), "mean_speed_kmh": float(spd.mean())}
        rec.update({nm: float(frac[i]) for i, nm in enumerate(SPEED_ZONE_NAMES)})
        rows.append(rec)
    return pd.DataFrame(rows)


def globalize_chunk_ids(positions: pd.DataFrame) -> pd.DataFrame:
    """Make ``frame`` and ``track_id`` unique across chunks (they reset per chunk in the batch output).

    Without this, grouping by ``frame`` merges different moments and grouping by ``track_id`` merges
    different players across chunks. Offsets each chunk's ids into its own block. No-op without a
    ``chunk`` column.
    """
    if "chunk" not in positions.columns:
        return positions
    p = positions.copy()
    code = p["chunk"].astype("category").cat.codes.to_numpy().astype("int64")
    p["frame"] = code * 1_000_000 + p["frame"].to_numpy().astype("int64")
    p["track_id"] = code * 100_000 + p["track_id"].to_numpy().astype("int64")
    return p


def align_teams_by_defended_goal(positions: pd.DataFrame) -> pd.DataFrame:
    """Relabel per-chunk team ids to a consistent split: 0 = defends the left goal, 1 = the right.

    The jersey-colour classifier labels teams arbitrarily *per chunk*, so "team 0" is not the same side
    across chunks. Within a match half each team defends a fixed goal, so the median keeper x gives a
    stable identity: the team with the smaller median keeper-x defends the left goal -> 0. Needs a
    ``chunk`` column (else treats the input as one segment). **Valid within a half only** -- across
    half-time teams swap ends, so a full match needs a jersey-colour anchor or a second-half flip.
    """
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [("_", positions)]
    out = []
    for _, g in groups:
        is_gk = g["role"] == "goalkeeper"
        if "is_keeper" in g.columns:
            is_gk = is_gk | (g["is_keeper"] == True)  # noqa: E712
        med = g[is_gk].dropna(subset=["pitch_x"]).groupby("team")["pitch_x"].median().sort_values()
        remap = {int(t): i for i, t in enumerate(med.index)}  # smallest keeper-x -> 0
        gg = g.copy()
        gg["team"] = gg["team"].map(lambda t: remap.get(int(t), -1) if int(t) >= 0 else -1)
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def team_style_vector(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                      min_team: int = 4) -> pd.DataFrame:
    """The deterministic team-style fingerprint ``z_T`` -- one interpretable row per team.

    Combines the structural fingerprint (shape + verticality) with the rough physical profile and a
    couple of derived style indices. Returns an empty frame if no team clears ``min_team`` players.
    """
    fp = team_fingerprint(compute_metrics_table(positions, min_team=min_team))
    if fp.empty:
        return fp
    return _add_style_indices(fp.merge(physical_profile(build_tracks(positions, fps=fps)),
                                       on="team", how="left"))


def _add_style_indices(z: pd.DataFrame) -> pd.DataFrame:
    """Interpretable style indices on top of the raw lane axes."""
    z["wing_share"] = z["lane_left_wing"] + z["lane_right_wing"]
    z["halfspace_share"] = z["lane_left_halfspace"] + z["lane_right_halfspace"]
    z["lr_bias"] = ((z["lane_left_wing"] + z["lane_left_halfspace"])
                    - (z["lane_right_wing"] + z["lane_right_halfspace"]))  # +ve = left-leaning
    return z


def match_metrics_table(positions: pd.DataFrame, *, min_team: int = 4) -> pd.DataFrame:
    """Per-``(frame, team)`` metrics across chunks with **per-chunk** attack directions.

    Teams must already be globally consistent (e.g. via :func:`generator.team_anchor.anchor_teams`).
    Resolving direction per chunk keeps the direction-normalised heights correct across the half-time
    end swap (a global team attacks opposite goals in the two halves).
    """
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [(None, positions)]
    parts = [compute_metrics_table(g, min_team=min_team, attack_dirs=resolve_attack_directions(g))
             for _, g in groups]
    return pd.concat(parts, ignore_index=True) if parts else compute_metrics_table(positions)


def match_style_vector(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                       min_team: int = 4) -> pd.DataFrame:
    """Full-match ``z_T``: per-chunk-direction structural fingerprint + global physical profile.

    Expects globally-consistent team ids and a ``chunk`` column. Use this (not
    :func:`team_style_vector`) for a multi-chunk match spanning both halves.
    """
    fp = team_fingerprint(match_metrics_table(positions, min_team=min_team))
    if fp.empty:
        return fp
    phys = physical_profile(build_tracks(globalize_chunk_ids(positions), fps=fps))
    return _add_style_indices(fp.merge(phys, on="team", how="left"))


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="dense positions parquet")
    ap.add_argument("--out", default=None, help="optional parquet for the z_T vectors")
    ap.add_argument("--align", action="store_true",
                    help="align team ids across chunks by defended goal (multi-chunk match parquet)")
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    if args.align:  # multi-chunk match parquet: make ids unique, then align team identities
        pos = align_teams_by_defended_goal(globalize_chunk_ids(pos))
        z = match_style_vector(pos)
    else:
        z = team_style_vector(pos)
    pd.set_option("display.width", 260, "display.max_columns", 60)
    print("team-style fingerprint z_T:\n")
    print(z.round(2).to_string(index=False))
    if args.out:
        z.to_parquet(args.out, index=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
