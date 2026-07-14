"""Positional-role and formation inference from broadcast tracks (no ball, no extra data).

Player *names* are infeasible from this footage (numbers are 3-5 px), but positional **roles**
(GK / full-back / centre-back / holding & central mid / wide & central attack) are recoverable from the
persistent tracks. We take each team's most-present tracks over a chunk, normalise their mean positions
into *attacking coordinates* (keeper-resolved, so independent of match half), and **Hungarian-match**
them against a small bank of formation templates; the lowest-cost template names the formation and
labels each track with a role.

Two payoffs:

* **Stable identity.** ``track_id`` resets per chunk and shatters on occlusion, so raw passing networks
  count fragments, not players (the "110 players for an 11-man team" bug). Collapsing tracks to roles
  gives ~11 stable slots per team -- a model-free re-identification good enough for per-role metrics.
* **Per-role analysis.** Line-breaking by the holding mid, build-up by the centre-backs, threat from the
  front three -- the FIFA-style reads that need to know *who* each player is, positionally.

Honest scope: broadcast shows only part of the pitch, so a chunk's mean positions are biased toward
wherever play happened; templates are coarse anchors, not ground truth. Roles are best read over a whole
chunk (or match), not a single frame. Pure numpy/pandas/scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from fingerprint.structural_metrics import (
    PITCH_LEN,
    PITCH_WID,
    attacking_coord,
    resolve_attack_directions,
)

# Formation templates as role anchors in **attacking coordinates**: x in [0, 105] toward the attacking
# goal (0 = own goal), y in [0, 68] across the pitch (0 = left touchline from the attacking POV). Each
# template is exactly 11 slots (1 keeper + 10 outfield).
FORMATIONS: dict[str, list[tuple[str, float, float]]] = {
    "4-3-3": [
        ("GK", 5, 34),
        ("LB", 25, 8), ("LCB", 20, 25), ("RCB", 20, 43), ("RB", 25, 60),
        ("LCM", 45, 20), ("CM", 45, 34), ("RCM", 45, 48),
        ("LW", 75, 12), ("ST", 80, 34), ("RW", 75, 56),
    ],
    "4-2-3-1": [
        ("GK", 5, 34),
        ("LB", 25, 8), ("LCB", 20, 25), ("RCB", 20, 43), ("RB", 25, 60),
        ("LDM", 40, 26), ("RDM", 40, 42),
        ("LAM", 62, 14), ("CAM", 64, 34), ("RAM", 62, 54),
        ("ST", 82, 34),
    ],
    "4-4-2": [
        ("GK", 5, 34),
        ("LB", 25, 8), ("LCB", 20, 25), ("RCB", 20, 43), ("RB", 25, 60),
        ("LM", 50, 10), ("LCM", 48, 28), ("RCM", 48, 40), ("RM", 50, 58),
        ("LST", 78, 27), ("RST", 78, 41),
    ],
    "3-5-2": [
        ("GK", 5, 34),
        ("LCB", 20, 18), ("CB", 18, 34), ("RCB", 20, 50),
        ("LWB", 50, 6), ("LCM", 45, 24), ("CM", 45, 34), ("RCM", 45, 44), ("RWB", 50, 62),
        ("LST", 78, 28), ("RST", 78, 40),
    ],
}

MIN_PRESENCE_FRAMES = 30  # a track must appear in at least this many frames to be a role candidate


def track_means(positions: pd.DataFrame, *, team: int, attack_dir: int | None,
                min_frames: int = MIN_PRESENCE_FRAMES) -> pd.DataFrame:
    """Mean attacking-coordinate position + presence for each outfield/keeper track of one team.

    Args:
        positions: a single chunk's positions (``frame, track_id, role, team, pitch_x, pitch_y,
            is_keeper``).
        team: which team label to summarise.
        attack_dir: that team's attacking direction (+1/-1); ``None`` leaves x un-normalised.
        min_frames: drop tracks present in fewer frames (transient / false tracks).

    Returns:
        ``track_id, ax, ay, frames, is_keeper`` sorted by presence (most-present first), where ``ax`` is
        the attacking-x (0 = own goal) and ``ay`` the cross-pitch coordinate.
    """
    g = positions[(positions["team"] == team) & positions["role"].isin(["player", "goalkeeper"])]
    g = g.dropna(subset=["pitch_x", "pitch_y"])
    if g.empty:
        return pd.DataFrame(columns=["track_id", "ax", "ay", "frames", "is_keeper"])
    has_kp = "is_keeper" in g.columns
    rows = []
    for tid, t in g.groupby("track_id"):
        if len(t) < min_frames:
            continue
        ax = attacking_coord(t["pitch_x"].to_numpy(), attack_dir).mean() if attack_dir is not None \
            else float(t["pitch_x"].mean())
        ay = float(t["pitch_y"].mean()) if attack_dir is None or attack_dir > 0 \
            else float(PITCH_WID - t["pitch_y"].mean())  # mirror y so "left" is attacking-relative
        kp = bool(t["is_keeper"].mean() > 0.5) if has_kp else (t["role"] == "goalkeeper").any()
        rows.append({"track_id": int(tid), "ax": float(ax), "ay": float(ay),
                     "frames": int(len(t)), "is_keeper": kp})
    out = pd.DataFrame(rows, columns=["track_id", "ax", "ay", "frames", "is_keeper"])
    return out.sort_values("frames", ascending=False).reset_index(drop=True)


def _match_formation(cands: pd.DataFrame, template: list[tuple[str, float, float]]
                     ) -> tuple[float, dict[int, str]]:
    """Hungarian-match candidate tracks to a template's 11 slots; return (mean cost, track->role).

    The keeper slot is constrained to the candidate with the strongest keeper evidence (its column),
    so the back line isn't mislabelled as the GK. Remaining slots match outfield candidates by distance.
    """
    if cands.empty:
        return float("inf"), {}
    roles = [r for r in template]
    # Keeper: force the best keeper candidate (if any) onto the GK slot.
    assign: dict[int, str] = {}
    pool = cands.copy()
    kp = pool[pool["is_keeper"]]
    if len(kp):
        gk_tid = int(kp.iloc[0]["track_id"])
        assign[gk_tid] = "GK"
        pool = pool[pool["track_id"] != gk_tid]
        roles = [r for r in roles if r[0] != "GK"]
    # Take up to len(roles) most-present outfield candidates.
    pool = pool.head(len(roles))
    if pool.empty:
        cost = np.mean([0.0]) if assign else float("inf")
        return (0.0 if assign else float("inf")), assign
    pts = pool[["ax", "ay"]].to_numpy(float)
    tpl = np.array([[x, y] for _, x, y in roles], float)
    # cost matrix [candidates x roles] of euclidean distance in metres
    cost = np.hypot(pts[:, None, 0] - tpl[None, :, 0], pts[:, None, 1] - tpl[None, :, 1])
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        assign[int(pool.iloc[r]["track_id"])] = roles[c][0]
    matched = cost[ri, ci]
    mean_cost = float(matched.mean()) if len(matched) else float("inf")
    return mean_cost, assign


def infer_formation(cands: pd.DataFrame) -> tuple[str, float, dict[int, str]]:
    """Pick the best-fitting formation for one team's candidate tracks.

    Returns ``(formation_name, mean_cost_m, {track_id: role})`` for the lowest-cost template.
    """
    best = ("?", float("inf"), {})
    for name, tpl in FORMATIONS.items():
        cost, assign = _match_formation(cands, tpl)
        if cost < best[1]:
            best = (name, cost, assign)
    return best


def assign_roles(positions: pd.DataFrame, *, min_frames: int = MIN_PRESENCE_FRAMES) -> pd.DataFrame:
    """Infer per-team formation + per-track role for every chunk in ``positions``.

    Args:
        positions: dense positions (multi-chunk ok; uses ``chunk`` if present). Needs ``role``,
            ``team``, ``is_keeper`` and pitch coordinates.
        min_frames: minimum track presence to be a role candidate.

    Returns:
        ``chunk, team, track_id, role, formation, fit_cost_m, frames`` -- one row per labelled track.
        ``fit_cost_m`` is the team-level mean matching cost (lower = cleaner formation fit).
    """
    has_chunk = "chunk" in positions.columns
    groups = positions.groupby("chunk") if has_chunk else [("_", positions)]
    rows = []
    for ck, g in groups:
        dirs = resolve_attack_directions(g)
        for team in sorted(t for t in g["team"].unique() if int(t) >= 0):
            cands = track_means(g, team=int(team), attack_dir=dirs.get(int(team)), min_frames=min_frames)
            if cands.empty:
                continue
            formation, cost, assign = infer_formation(cands)
            fr_by_tid = dict(zip(cands["track_id"], cands["frames"]))
            for tid, role in assign.items():
                rows.append({"chunk": ck, "team": int(team), "track_id": int(tid), "role": role,
                             "formation": formation, "fit_cost_m": round(cost, 2),
                             "frames": int(fr_by_tid.get(tid, 0))})
    return pd.DataFrame(rows, columns=["chunk", "team", "track_id", "role", "formation",
                                       "fit_cost_m", "frames"])


def role_lookup(roles: pd.DataFrame) -> dict[tuple[str, int], str]:
    """``(chunk, track_id) -> role`` map for tagging positions / possession with stable role labels."""
    return {(r.chunk, int(r.track_id)): r.role for r in roles.itertuples(index=False)}


# Canonical ordering of every role label across the templates, for stable integer ids.
ALL_ROLES = sorted({slot[0] for tpl in FORMATIONS.values() for slot in tpl})
_ROLE_INDEX = {r: i for i, r in enumerate(ALL_ROLES)}


def stable_role_id(team: int, role: str) -> int:
    """A stable integer id for ``(team, role)`` -- ``team * 100 + role_index`` (1-to-1 within a team)."""
    return int(team) * 100 + _ROLE_INDEX[role]


def role_centroids(players: pd.DataFrame, roles: pd.DataFrame, *, chunk: str | None = None
                   ) -> tuple[dict[int, list[tuple[int, float, float]]], dict[int, str]]:
    """Per team, the pitch-metre centroid of each of its 11 role slots (from the assigned backbone tracks).

    Returns ``({team: [(role_id, cx, cy), ...]}, {role_id: "team:role"})`` -- the spatial anchors used to
    map *every* track (not just the backbone 11) to a role.
    """
    rr = roles[roles["chunk"] == chunk] if chunk is not None and "chunk" in roles.columns else roles
    assigned = {int(r.track_id): (int(r.team), r.role) for r in rr.itertuples(index=False)}
    pl = players.dropna(subset=["pitch_x", "pitch_y"])
    means = pl[pl["track_id"].isin(assigned)].groupby("track_id")[["pitch_x", "pitch_y"]].mean()
    anchors: dict[int, list[tuple[int, float, float]]] = {}
    id2label: dict[int, str] = {}
    for tid, (team, role) in assigned.items():
        if tid not in means.index:
            continue
        rid = stable_role_id(team, role)
        cx, cy = float(means.loc[tid, "pitch_x"]), float(means.loc[tid, "pitch_y"])
        anchors.setdefault(team, []).append((rid, cx, cy))
        id2label[rid] = f"{team}:{role}"
    return anchors, id2label


def relabel_to_roles(possession: pd.DataFrame, players: pd.DataFrame, roles: pd.DataFrame, *,
                     chunk: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Collapse fragmented ``track_id``s to stable role ids so passing networks have <=11 nodes/team.

    The backbone 11 tracks (from :func:`assign_roles`) define each role's pitch centroid; **every** track
    -- including the minor fragments that are most often the ball carrier -- is then mapped to its team's
    **nearest role centroid**. Both the possession carriers and the player positions are relabelled to
    ``stable_role_id(team, role)``, so downstream :func:`~fingerprint.possession_metrics.extract_passes`
    / :func:`~fingerprint.possession_metrics.passing_network` produce role-keyed, <=11-node-per-team
    networks, and a pass between two fragments of the *same* player (same nearest role) correctly vanishes.

    Approximate by design: a fragment is assigned by its mean position, so a player drifting between two
    role zones can be mislabelled. It is a model-free re-identification, not ground truth.

    Args:
        possession: ``frame, carrier, team, dist_m`` (one chunk).
        players: positions for the same chunk (``frame, track_id, team, pitch_x, pitch_y`` + image cols).
        roles: output of :func:`assign_roles` (``chunk, team, track_id, role, ...``).
        chunk: restrict the role map to this chunk (track ids are only unique within a chunk).

    Returns:
        ``(possession2, players2, id_to_label)`` -- relabelled tables and a ``role_id -> "team:role"``
        display map.
    """
    anchors, id2label = role_centroids(players, roles, chunk=chunk)
    pl_xy = players.dropna(subset=["pitch_x", "pitch_y"])
    tmean = pl_xy.groupby("track_id").agg(team=("team", "first"), x=("pitch_x", "mean"),
                                          y=("pitch_y", "mean"))
    tid2id: dict[int, int] = {}
    for tid, row in tmean.iterrows():
        team_anchors = anchors.get(int(row["team"]))
        if not team_anchors:
            continue
        tid2id[int(tid)] = min(team_anchors, key=lambda a: (row["x"] - a[1]) ** 2 + (row["y"] - a[2]) ** 2)[0]

    poss = possession.copy()
    poss["carrier"] = poss["carrier"].map(tid2id)
    poss = poss.dropna(subset=["carrier"]).astype({"carrier": int}).reset_index(drop=True)

    pl = players.copy()
    pl["track_id"] = pl["track_id"].map(tid2id)
    pl = pl.dropna(subset=["track_id"]).astype({"track_id": int}).reset_index(drop=True)
    return poss, pl, id2label


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="dense positions parquet (multi-chunk ok)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--team0", default="team 0")
    ap.add_argument("--team1", default="team 1")
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    roles = assign_roles(pos)
    names = {0: args.team0, 1: args.team1}
    # Report the most common formation per team across chunks (the match-level read).
    print("Inferred formation (most common across chunks):\n")
    for team, g in roles.groupby("team"):
        mode = g.groupby("chunk")["formation"].first().mode()
        form = mode.iloc[0] if len(mode) else "?"
        cost = g["fit_cost_m"].mean()
        print(f"  {names.get(int(team), f'team {team}')}: {form}  (mean fit {cost:.1f} m, "
              f"{g['chunk'].nunique()} chunks)")
    if args.out:
        roles.to_parquet(args.out, index=False)
        print(f"\nper-track roles -> {args.out}")


if __name__ == "__main__":
    main()
