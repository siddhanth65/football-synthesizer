"""Per-team metrics organised into the four FIFA facets: attacking, defending, passing, goalkeeping.

Combines the deterministic positional metrics (shape / verticality / lanes / physical) with **new**
goalkeeping (sweeper height, lateral range) and a **passing-connectivity proxy** (nearest-team-mate
distance + directness). Attacking direction is resolved **per chunk**, so the orientation is correct
across the half-time end swap. The relational GAT reads (attacking threat / defensive recovery /
receiver) layer on top via :mod:`generator.sop_bridge`.

**Honest scope (no ball yet):** these are orientation-normalised **overall** metrics per facet, not the
FIFA *in/out-of-possession phase split*; passing/goalkeeping here are **positional proxies**, not event
counts (pass completion, line breaks, GK distribution/saves need ball tracking). Run on the
colour-anchored full match (`generator.team_anchor`) for per-team-consistent numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.tracks import DEFAULT_FPS, build_tracks
from fingerprint.structural_metrics import attacking_coord, resolve_attack_directions, team_fingerprint
from fingerprint.team_style import (
    _add_style_indices,
    globalize_chunk_ids,
    match_metrics_table,
    physical_profile,
)

PLAYER_ROLES = ("player", "goalkeeper")


def goalkeeper_metrics(positions: pd.DataFrame, attack_dirs: dict[int, int]) -> pd.DataFrame:
    """Per team: ``gk_sweeper_height`` (median keeper distance up-pitch from its own goal -- higher =
    sweeper-keeper) and ``gk_lateral_range`` (keeper side-to-side spread, std of y)."""
    is_gk = positions["role"] == "goalkeeper"
    if "is_keeper" in positions.columns:
        is_gk = is_gk | (positions["is_keeper"] == True)  # noqa: E712
    gk = positions[is_gk].dropna(subset=["pitch_x", "pitch_y"])
    rows = []
    for team, g in gk.groupby("team"):
        d = attack_dirs.get(int(team))
        if int(team) < 0 or d is None or g.empty:
            continue
        height = attacking_coord(g["pitch_x"].to_numpy(), d)  # 0 = on own line, up = swept out
        rows.append({"team": int(team), "gk_sweeper_height": float(np.median(height)),
                     "gk_lateral_range": float(g["pitch_y"].std())})
    return pd.DataFrame(rows, columns=["team", "gk_sweeper_height", "gk_lateral_range"])


def passing_connectivity(positions: pd.DataFrame, *, min_team: int = 4) -> pd.DataFrame:
    """Per team: mean **nearest-team-mate distance** (shorter = tighter network / more short options)."""
    pl = positions[positions["role"].isin(PLAYER_ROLES)].dropna(subset=["pitch_x", "pitch_y"])
    acc: dict[int, list] = {}
    for (_, team), g in pl.groupby(["frame", "team"]):
        if int(team) < 0 or len(g) < min_team:
            continue
        xy = g[["pitch_x", "pitch_y"]].to_numpy()
        dist = np.hypot(xy[:, 0, None] - xy[None, :, 0], xy[:, 1, None] - xy[None, :, 1])
        np.fill_diagonal(dist, np.inf)
        acc.setdefault(int(team), []).append(float(dist.min(axis=1).mean()))
    return pd.DataFrame([{"team": t, "pass_nearest_mate_m": float(np.mean(v))} for t, v in acc.items()])


def _per_chunk(positions: pd.DataFrame, fn) -> pd.DataFrame:
    """Apply ``fn(chunk_df, attack_dirs)`` per chunk and average the result per team."""
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [(None, positions)]
    parts = [fn(g, resolve_attack_directions(g)) for _, g in groups]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return out.groupby("team", as_index=False).mean()


# Facet -> (source column, output name). Relational reads come from the GAT
# (generator.sop_bridge.per_team_relational), Territory from fingerprint.pitch_control, Coordination
# from fingerprint.style_metrics -- all merged in when supplied.
_FACETS: dict[str, list[tuple[str, str]]] = {
    "ATTACKING": [("buildup_height", "att_buildup_height"), ("attacking_third_share", "att_third_share"),
                  ("surface_area", "att_surface_m2"), ("lane_centre", "att_centre_share"),
                  ("wing_share", "att_wing_share"), ("att_xt", "att_threat_xt"),
                  ("att_success", "att_success_p"), ("att_option_richness", "att_option_richness")],
    "DEFENDING": [("def_line_height", "def_line_height"), ("compactness", "def_compactness_m"),
                  ("width", "def_block_width_m"), ("def_recovery", "def_recovery_p"),
                  ("def_press", "def_press_decisiveness"), ("def_lane_suppression", "def_lane_suppression")],
    "TERRITORY": [("space_control", "space_control"), ("att_third_control", "att_third_control")],
    "PASSING": [("pass_nearest_mate_m", "pass_nearest_mate_m"), ("length", "pass_depth_m")],
    "GOALKEEPING": [("gk_sweeper_height", "gk_sweeper_height_m"),
                    ("gk_lateral_range", "gk_lateral_range_m")],
    "COORDINATION": [("velocity_synchrony", "velocity_synchrony")],
    "PHYSICAL": [("top_speed_kmh", "phys_top_speed_kmh"), ("sprint_share", "phys_sprint_share")],
}


def match_facet_fingerprint(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS,
                            relational: pd.DataFrame | None = None, space: pd.DataFrame | None = None,
                            synchrony: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-team facet fingerprint over a colour-anchored, multi-chunk match.

    Optional per-team tables enrich the facets: ``relational`` (GAT — xT/success/option-richness/recovery/
    press/lane-suppression), ``space`` (pitch control — space & attacking-third control), ``synchrony``
    (movement synchrony).
    """
    shape = _add_style_indices(team_fingerprint(match_metrics_table(positions)))
    phys = physical_profile(build_tracks(globalize_chunk_ids(positions), fps=fps))
    gk = _per_chunk(positions, goalkeeper_metrics)
    pas = _per_chunk(positions, lambda g, _d: passing_connectivity(g))
    merged = shape.merge(phys, on="team", how="left").merge(gk, on="team", how="left") \
                  .merge(pas, on="team", how="left")
    rel_cols = ["att_xt", "att_success", "att_option_richness", "def_recovery", "def_press",
                "def_lane_suppression"]
    for extra, keep in ((relational, rel_cols), (space, ["space_control", "att_third_control"]),
                        (synchrony, ["velocity_synchrony"])):
        if extra is not None and not extra.empty:
            merged = merged.merge(extra[["team", *[c for c in keep if c in extra.columns]]],
                                  on="team", how="left")
    cols = {src: dst for facet in _FACETS.values() for src, dst in facet}
    out = merged[["team", *[c for c in cols if c in merged.columns]]].rename(columns=cols)
    out.insert(1, "frames", merged["frames"])
    return out


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="colour-anchored match positions parquet")
    ap.add_argument("--relational", default=None, help="per-team GAT relational parquet (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rel = pd.read_parquet(args.relational) if args.relational else None
    z = match_facet_fingerprint(pd.read_parquet(args.positions), relational=rel)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    for facet, pairs in _FACETS.items():
        names = [dst for _, dst in pairs if dst in z.columns]
        print(f"\n=== {facet} ===")
        print(z[["team", *names]].round(2).to_string(index=False))
    if args.out:
        z.to_parquet(args.out, index=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
