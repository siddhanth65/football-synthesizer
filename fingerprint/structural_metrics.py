"""Deterministic, auditable team-shape metrics over tactical frames (the metric engine, Section 3 v1).

The review is explicit: compute deterministic metrics *before* training any model. These need only
player positions + team labels from the dense/tactical tables -- no ball, no learned model -- so they
are fully reproducible and testable:

* **width** (spread across the pitch's short axis) and **length** (spread along the long axis),
* **compactness** (mean distance to the team centroid -- how tight the block is),
* **surface area** (convex hull of the outfield players -- territory occupied),
* **five-lane occupation** (share of players in left-wing / left-half-space / centre / right-half-space
  / right-wing channels, by pitch width).

These are deliberately **direction-agnostic** (lanes/width/compactness/area don't need to know which way
a team attacks). Direction-dependent metrics -- defensive-line/build-up *height* and attacking-third
share -- need a team's attacking goal; we resolve it from the **goalkeeper's defended goal** (read from
the footage, so it adapts to whichever match half a clip is from), never from outfield mean-x (which
the review warns against). Aggregated across a match, all of these give a first team fingerprint.

Pure: numpy/pandas (+ scipy for the hull, with a bbox fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID
# Five vertical channels across the pitch width (the standard 5-lane model).
LANE_EDGES = np.array([0.0, PITCH_WID / 5, 2 * PITCH_WID / 5, 3 * PITCH_WID / 5, 4 * PITCH_WID / 5,
                       PITCH_WID])
LANE_NAMES = ("left_wing", "left_halfspace", "centre", "right_halfspace", "right_wing")
MIN_TEAM_PLAYERS = 4  # need a few players for a meaningful shape


def lane_fractions(py: np.ndarray) -> np.ndarray:
    """Fraction of players in each of the five width-lanes (sums to 1, or zeros if empty)."""
    counts, _ = np.histogram(np.asarray(py, float), bins=LANE_EDGES)
    total = counts.sum()
    return counts / total if total else np.zeros(len(LANE_NAMES))


def _hull_area(px: np.ndarray, py: np.ndarray) -> float:
    """Convex-hull area (m^2) of the player cloud; 0 for < 3 non-degenerate points."""
    pts = np.column_stack([px, py])
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415

        return float(ConvexHull(pts).volume)  # 2-D ConvexHull.volume is the polygon area
    except Exception:  # noqa: BLE001 - degenerate (collinear) or scipy missing -> bbox fallback
        return float(np.ptp(px) * np.ptp(py))


def team_shape(px: np.ndarray, py: np.ndarray) -> dict:
    """Structural metrics for one team's outfield points (metres on the 105x68 pitch)."""
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    cx, cy = float(px.mean()), float(py.mean())
    compact = float(np.hypot(px - cx, py - cy).mean()) if len(px) > 1 else 0.0
    lanes = lane_fractions(py)
    out = {
        "n": int(len(px)),
        "centroid_x": cx, "centroid_y": cy,
        "width": float(np.ptp(py)), "length": float(np.ptp(px)),
        "width_std": float(py.std()), "length_std": float(px.std()),
        "compactness": compact,
        "surface_area": _hull_area(px, py),
    }
    out.update({f"lane_{nm}": float(lanes[i]) for i, nm in enumerate(LANE_NAMES)})
    return out


METRIC_COLUMNS = tuple(team_shape(np.array([0.0, 1.0, 2.0, 3.0]),
                                  np.array([0.0, 20.0, 40.0, 60.0])).keys())

# --- Direction-dependent metrics ------------------------------------------------------------------
# These need a team's attacking goal. We resolve it from the goalkeeper's defended goal (below), never
# from outfield mean-x. Everything here is normalised so 0 m = a team's own goal and 105 m = the goal
# it attacks, making the two teams directly comparable.
ATTACKING_THIRD_X = 2 * PITCH_LEN / 3  # the 70 m line; presence beyond it = in the attacking third
DEF_LINE_QUANTILE = 20  # robust "deepest line" percentile (ignores one straggler / partial broadcast view)
DIRECTION_COLUMNS = ("buildup_height", "def_line_height", "attacking_third_share")


def attacking_coord(px: np.ndarray, attack_dir: int) -> np.ndarray:
    """Map pitch x to metres *toward the attacking goal* (0 = own goal, 105 = attacking goal)."""
    px = np.asarray(px, float)
    return px if attack_dir > 0 else PITCH_LEN - px


def resolve_attack_directions(positions: pd.DataFrame) -> dict[int, int]:
    """Per team, the attacking-direction sign (+1 toward x=105, -1 toward x=0), from its keeper's goal.

    The goalkeeper sits by the goal the team *defends*, so the team attacks the opposite goal. Uses
    rows tagged ``role == 'goalkeeper'`` or (positionally) ``is_keeper``, taking the **median** keeper
    x to shrug off noisy positional tags. Because it reads the actual footage it adapts to whichever
    match half a clip comes from -- no hand-kept half table. Teams with **no keeper evidence are
    omitted** (``.get`` -> ``None`` downstream) so direction-dependent metrics stay NaN rather than
    being guessed from outfield mean-x.

    **Opposite-team constraint.** A follow-play broadcast often shows only one goalmouth, so both teams'
    keeper-tagged points cluster at the *same* end (the off-screen keeper's rows are misdetections) and a
    naive per-team resolve hands both teams the same direction -- which corrupts every direction-dependent
    metric. Since the two teams must attack opposite goals, we anchor on the **better-sampled keeper**
    (more keeper rows = more reliable) and force the other team opposite.
    """
    is_gk = positions["role"] == "goalkeeper"
    if "is_keeper" in positions.columns:
        is_gk = is_gk | (positions["is_keeper"] == True)  # noqa: E712
    kp = positions[is_gk].dropna(subset=["pitch_x"])
    stats = {}  # team -> (defends_left, keeper_row_count)
    for team, g in kp.groupby("team"):
        if int(team) < 0 or g.empty:
            continue
        stats[int(team)] = (float(g["pitch_x"].median()) < PITCH_LEN / 2, len(g))
    out = {t: (1 if left else -1) for t, (left, _) in stats.items()}
    # Two teams resolved to the same direction -> trust the better-sampled keeper, flip the other.
    if len(out) == 2 and len(set(out.values())) == 1:
        anchor = max(stats, key=lambda t: stats[t][1])
        for t in out:
            out[t] = (1 if stats[anchor][0] else -1) * (1 if t == anchor else -1)
    return out


def resolve_attack_directions_from_ball(positions: pd.DataFrame, ball: pd.DataFrame, *,
                                        radius_m: float = 4.0) -> dict[int, int]:
    """Attacking direction per team from **where each team has the ball** -- robust on follow-play footage.

    Keeper-based direction fails when the broadcast shows only one goalmouth (both teams' keepers cluster
    at the on-screen goal). The ball avoids this: credit each ball sample to the nearest player's team
    (the carrier), then compare the two teams' **mean ball-x during their own possession**. The team whose
    possessions sit further toward ``x = 105`` attacks that goal (+1); the other attacks ``x = 0`` (-1).
    A *relative* comparison (which team is further forward), so it doesn't assume a team spends most of its
    possession in the attacking half -- only that it is further forward than its opponent.

    Args:
        positions: one chunk's player positions (``frame, team, role, pitch_x, pitch_y``), colour-anchored.
        ball: linked ball track ``frame, x, y`` (pitch metres).
        radius_m: a player within this of the ball is treated as the carrier.

    Returns:
        ``{team: +1/-1}`` for the (up to two) teams with possession evidence; empty if the ball never
        falls near a player.
    """
    pl = positions[positions["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    by_frame = {int(fr): g for fr, g in pl.groupby("frame")}
    team_ballx: dict[int, list[float]] = {}
    for b in ball.itertuples(index=False):
        g = by_frame.get(int(b.frame))
        if g is None:
            continue
        d = np.hypot(g["pitch_x"].to_numpy() - b.x, g["pitch_y"].to_numpy() - b.y)
        j = int(np.argmin(d))
        if d[j] <= radius_m:
            t = int(g.iloc[j]["team"])
            if t >= 0:
                team_ballx.setdefault(t, []).append(float(b.x))
    means = {t: float(np.mean(xs)) for t, xs in team_ballx.items() if xs}
    if len(means) == 2:
        hi = max(means, key=means.get)               # further toward x=105 -> attacks +1
        return {t: (1 if t == hi else -1) for t in means}
    return {t: (1 if mx > PITCH_LEN / 2 else -1) for t, mx in means.items()}


def direction_metrics(px: np.ndarray, is_keeper: np.ndarray, attack_dir: int | None) -> dict:
    """Direction-normalised heights (metres toward the attacking goal); all NaN if ``attack_dir`` unknown.

    * ``buildup_height`` -- mean attacking-x of the outfield players (how high the team sits),
    * ``def_line_height`` -- a robust low-percentile attacking-x (where the deepest line holds),
    * ``attacking_third_share`` -- fraction of outfield players already in the attacking third.

    The keeper is excluded from all three (it would always be the deepest point).
    """
    if attack_dir is None:
        return {c: float("nan") for c in DIRECTION_COLUMNS}
    a = attacking_coord(px, attack_dir)
    keep = np.asarray(is_keeper, bool)
    outfield = a[~keep] if bool((~keep).any()) else a
    return {
        "buildup_height": float(outfield.mean()),
        "def_line_height": float(np.percentile(outfield, DEF_LINE_QUANTILE)),
        "attacking_third_share": float(np.mean(outfield > ATTACKING_THIRD_X)),
    }


def compute_metrics_table(positions: pd.DataFrame, *, min_team: int = MIN_TEAM_PLAYERS,
                          attack_dirs: dict[int, int] | None = None) -> pd.DataFrame:
    """Per ``(frame, team)`` structural metrics for every team with >= ``min_team`` on-pitch players.

    ``attack_dirs`` overrides the keeper-derived attacking directions (e.g. from known match-half
    metadata); pass ``{}`` to force every direction-dependent metric to NaN.
    """
    if attack_dirs is None:
        attack_dirs = resolve_attack_directions(positions)
    pl = positions[positions["role"].isin(["player", "goalkeeper"])].dropna(
        subset=["pitch_x", "pitch_y"])
    has_keeper = "is_keeper" in pl.columns
    rows = []
    for (fr, team), g in pl.groupby(["frame", "team"]):
        if int(team) < 0 or len(g) < min_team:
            continue
        adir = attack_dirs.get(int(team))
        rec = {"frame": int(fr), "team": int(team), "attack_dir": 0 if adir is None else int(adir)}
        rec.update(team_shape(g["pitch_x"].to_numpy(), g["pitch_y"].to_numpy()))
        kmask = g["is_keeper"].to_numpy() if has_keeper else np.zeros(len(g), bool)
        rec.update(direction_metrics(g["pitch_x"].to_numpy(), kmask, adir))
        rows.append(rec)
    cols = ["frame", "team", "attack_dir", *METRIC_COLUMNS, *DIRECTION_COLUMNS]
    return pd.DataFrame(rows, columns=cols)


def team_fingerprint(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame metrics into one row per team: frames observed + mean of each metric.

    A first, interpretable team fingerprint (mean shape over the match). Std is available per metric
    via ``metrics_table.groupby('team').std()`` when uncertainty is needed.
    """
    if metrics_table.empty:
        return pd.DataFrame()
    metric_cols = [c for c in (*METRIC_COLUMNS, *DIRECTION_COLUMNS) if c != "n"]
    agg = metrics_table.groupby("team")[metric_cols].mean()
    agg.insert(0, "frames", metrics_table.groupby("team").size())
    agg.insert(1, "avg_players", metrics_table.groupby("team")["n"].mean())
    if "attack_dir" in metrics_table.columns:
        agg.insert(2, "attack_dir", metrics_table.groupby("team")["attack_dir"].first())
    return agg.reset_index()


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="dense positions parquet")
    ap.add_argument("--out", default=None, help="optional parquet for the per-frame metrics table")
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    table = compute_metrics_table(pos)
    fp = team_fingerprint(table)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(f"team fingerprint over {table['frame'].nunique()} tactical frames:\n")
    print(fp.round(2).to_string(index=False))
    if args.out:
        table.to_parquet(args.out, index=False)
        print(f"\nper-frame metrics -> {args.out}")


if __name__ == "__main__":
    main()
