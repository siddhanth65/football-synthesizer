"""Possession-derived team metrics: passes, passing networks, and a PPDA pressing proxy.

These consume the ball-possession output (``generator.ball.assign_possession`` -> ``frame, carrier, team,
dist_m``) plus the player positions, so they need a working ball track (see the fine-tuned detector +
``tools/ball_possession.py``). Everything here is pure pandas/numpy and unit-tested.

* :func:`extract_passes` — consecutive same-team carrier hand-offs become directed passes (with end points
  in pitch metres).
* :func:`passing_network` — per team: nodes (a player's mean position + involvement) and weighted directed
  edges (pass counts); :func:`network_metrics` adds degree centrality and shape (compactness/width/depth).
* :func:`ppda` — passes-allowed-per-defensive-action: opponent passes in the pressing zone divided by a
  proximity-based **pressure proxy**. Honest scope: we don't detect tackles/interceptions, so the
  denominator is "off-ball defender within ``press_radius_m`` of the carrier", not Opta's event count —
  validate the magnitude against FIFA ``DefensivePressuresApplied`` (``outputs/fifaphy``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN
from fingerprint.structural_metrics import resolve_attack_directions

PRESS_ZONE_FRAC = 0.6   # PPDA presses outside the attacker's defensive 60% are ignored (Opta-style)
PRESS_RADIUS_M = 3.0    # an off-ball defender this close to the carrier counts as one pressure

# A direct pass completes within ~1 s; a longer carrier-to-carrier gap is a new spell / ball-out, not
# a single pass. Expressed in SECONDS so the physical window is identical across matches regardless of
# source fps (25 vs 59.94 Hz); the legacy ``max_gap_frames`` reproduces prior behaviour without an fps.
PASS_MAX_GAP_S = 0.9


def possession_share_proxy(team_passes: int, opp_passes: int) -> float:
    """Pass-count-share PROXY for possession share (Phatak et al. use *time*-based possession).

    Returns ``team_passes / (team_passes + opp_passes)`` -- a stand-in for the time-share the
    original possession-normalization method assumes. It is a PROXY, not the same quantity: a team can
    hold the ball long (high time share) while completing few passes (low pass share) and vice versa.
    Callers must label any downstream number ``*_proxy`` / "pass-share proxy", never "possession".

    Args:
        team_passes: the team's (completed) pass count.
        opp_passes: the opponent's pass count.

    Returns:
        Pass-share in ``[0, 1]``; ``nan`` when neither side has a pass.
    """
    total = team_passes + opp_passes
    return float(team_passes) / total if total else float("nan")


def normalize_kpi(kpi: float, poss_share: float) -> float:
    """Possession-normalize a KPI: ``KPI / (1 - poss_share)`` (Phatak et al., Sci Rep 2022).

    The correct denominator for any KPI accrued while the OPPONENT has the ball (shots conceded,
    passes allowed, defensive-phase shares): a team that only sees 30% of the ball defends for 70% of
    the match, so raw conceded counts understate its per-opportunity rate. Dividing by ``1 - poss``
    puts teams with different possession styles on a comparable footing.

    HONEST SCOPE (two caveats from the method's honesty trail):
      * ``poss_share`` here is typically the **pass-share proxy** (:func:`possession_share_proxy`),
        NOT the time-based possession the paper validated on -- label results as a proxy.
      * the paper validated only on **whole-season aggregates**; single-match / per-phase use is our
        extrapolation.

    Args:
        kpi: the raw metric (e.g. shots conceded, passes allowed, defensive-phase frame share).
        poss_share: the team's possession (or pass-share proxy) in ``[0, 1)``.

    Returns:
        ``kpi / (1 - poss_share)``; ``nan`` when ``poss_share`` is not in ``[0, 1)``.
    """
    if not 0.0 <= poss_share < 1.0:
        return float("nan")
    return float(kpi) / (1.0 - poss_share)


def _carrier_positions(players: pd.DataFrame) -> dict[tuple[int, int], tuple[float, float]]:
    """``(frame, track_id) -> (pitch_x, pitch_y)`` lookup for resolving carrier locations."""
    g = players.dropna(subset=["pitch_x", "pitch_y"])
    idx = {}
    for fr, tid, x, y in zip(g["frame"].to_numpy(int), g["track_id"].to_numpy(int),
                             g["pitch_x"].to_numpy(float), g["pitch_y"].to_numpy(float)):
        idx[(int(fr), int(tid))] = (float(x), float(y))
    return idx


def extract_passes(possession: pd.DataFrame, players: pd.DataFrame, *,
                   max_gap_frames: int = 50, max_gap_s: float | None = None,
                   fps: float | None = None) -> pd.DataFrame:
    """Turn possession hand-offs into directed passes.

    A pass is two consecutive possession samples of the **same team** with **different carriers** whose
    frame gap is within the allowed window (a longer gap = the ball went out / a new spell, not a direct
    pass).

    The window may be given either in native frames (``max_gap_frames``) or, preferred, in **seconds**
    (``max_gap_s`` + ``fps``). Seconds are fps-independent, so the same physical pass window applies to a
    25 fps and a 59.94 fps broadcast; a native-frame count silently means a different real duration per
    match (50 frames = 2.0 s at 25 fps but 0.83 s at 59.94 fps). When ``max_gap_s`` is given it overrides
    ``max_gap_frames`` via ``round(max_gap_s * fps)``.

    Args:
        possession: ``frame, carrier, team, dist_m`` from :func:`generator.ball.assign_possession`.
        players: positions with ``frame, track_id, pitch_x, pitch_y``.
        max_gap_frames: maximum *native-frame* gap between carriers (legacy; used when ``max_gap_s`` is
            ``None``).
        max_gap_s: maximum gap in **seconds**; requires ``fps`` and takes precedence when given.
        fps: native frames per second of this chunk's source video (needed only with ``max_gap_s``).

    Returns:
        ``frame, f_to, team, from_id, to_id, x0, y0, x1, y1, length_m`` (one row per completed pass).

    Raises:
        ValueError: if ``max_gap_s`` is given without a positive ``fps``.
    """
    if max_gap_s is not None:
        if fps is None or fps <= 0:
            raise ValueError("extract_passes: max_gap_s requires a positive fps")
        max_gap_frames = max(1, round(max_gap_s * fps))
    cols = ["frame", "f_to", "team", "from_id", "to_id", "x0", "y0", "x1", "y1", "length_m"]
    if possession.empty:
        return pd.DataFrame(columns=cols)
    pos = _carrier_positions(players)
    p = possession.sort_values("frame").reset_index(drop=True)
    rows = []
    for a, b in zip(p.itertuples(index=False), p.iloc[1:].itertuples(index=False)):
        if a.team != b.team or a.carrier == b.carrier:
            continue
        if b.frame - a.frame > max_gap_frames:
            continue
        pa, pb = pos.get((int(a.frame), int(a.carrier))), pos.get((int(b.frame), int(b.carrier)))
        if pa is None or pb is None:
            continue
        rows.append({"frame": int(a.frame), "f_to": int(b.frame), "team": int(a.team),
                     "from_id": int(a.carrier), "to_id": int(b.carrier),
                     "x0": pa[0], "y0": pa[1], "x1": pb[0], "y1": pb[1],
                     "length_m": float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))})
    return pd.DataFrame(rows, columns=cols)


def passing_network(passes: pd.DataFrame, players: pd.DataFrame, *,
                    team: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-team passing network: node table (mean position + involvement) and weighted edge table.

    Args:
        passes: output of :func:`extract_passes`.
        players: positions with ``frame, track_id, team, pitch_x, pitch_y``.
        team: which team's network to build.

    Returns:
        ``(nodes, edges)`` where ``nodes`` is ``track_id, mean_x, mean_y, touches`` and ``edges`` is
        ``from_id, to_id, count`` (directed).
    """
    tp = passes[passes["team"] == team]
    edges = (tp.groupby(["from_id", "to_id"]).size().reset_index(name="count")
             if not tp.empty else pd.DataFrame(columns=["from_id", "to_id", "count"]))
    involved = pd.unique(tp[["from_id", "to_id"]].to_numpy().ravel()) if not tp.empty else []
    pl = players[(players["team"] == team) & players["track_id"].isin(involved)].dropna(
        subset=["pitch_x", "pitch_y"])
    touches = (tp["from_id"].value_counts() + tp["to_id"].value_counts()).fillna(0) if not tp.empty \
        else pd.Series(dtype=float)
    nodes = (pl.groupby("track_id").agg(mean_x=("pitch_x", "mean"), mean_y=("pitch_y", "mean"))
             .reset_index())
    nodes["touches"] = nodes["track_id"].map(touches).fillna(0).astype(int)
    return nodes, edges


def network_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, float]:
    """Summarise a passing network: connectivity, top connector, and shape (compactness/width/depth).

    ``centrality`` is weighted degree (in+out pass count) normalised by the busiest node.
    """
    if nodes.empty:
        return {"n_players": 0, "n_passes": 0, "top_connector": None, "mean_pass_m": float("nan"),
                "compactness_m": float("nan"), "width_m": float("nan"), "depth_m": float("nan")}
    deg = {int(t): 0 for t in nodes["track_id"]}
    for e in edges.itertuples(index=False):
        deg[int(e.from_id)] = deg.get(int(e.from_id), 0) + int(e.count)
        deg[int(e.to_id)] = deg.get(int(e.to_id), 0) + int(e.count)
    top = max(deg, key=deg.get) if deg else None
    cx, cy = nodes["mean_x"].mean(), nodes["mean_y"].mean()
    compact = float(np.hypot(nodes["mean_x"] - cx, nodes["mean_y"] - cy).mean())
    return {"n_players": int(len(nodes)), "n_passes": int(edges["count"].sum()) if not edges.empty else 0,
            "top_connector": int(top) if top is not None else None,
            "compactness_m": compact,
            "width_m": float(np.ptp(nodes["mean_y"])) if len(nodes) > 1 else 0.0,
            "depth_m": float(np.ptp(nodes["mean_x"])) if len(nodes) > 1 else 0.0}


def ppda(possession: pd.DataFrame, players: pd.DataFrame, *, directions: dict[int, int] | None = None,
         press_radius_m: float = PRESS_RADIUS_M, zone_frac: float = PRESS_ZONE_FRAC) -> pd.DataFrame:
    """Passes-allowed-per-defensive-action (pressing intensity) proxy, per team.

    For each defending team D pressing attacker A: count A's completed passes that **originate** in the
    pressing zone (A's own defensive ``zone_frac`` of the pitch), and the **pressures** D applies there
    (possession frames in the zone with a D player within ``press_radius_m`` of A's carrier).
    ``PPDA = passes_allowed / pressures`` -- lower = more intense pressing. Needs keeper-resolved
    attack directions (auto via :func:`resolve_attack_directions`); teams without a direction are skipped.

    Returns:
        ``team, passes_allowed, pressures, ppda`` (``team`` is the **pressing/defending** team).
    """
    directions = directions or resolve_attack_directions(players)
    passes = extract_passes(possession, players)
    pos_by_frame = {fr: g for fr, g in players.dropna(subset=["pitch_x", "pitch_y"]).groupby("frame")}
    rows = []
    for d_team in sorted(t for t in players["team"].unique() if int(t) >= 0):
        a_team = next((t for t in directions if t != d_team), None)
        if a_team is None or a_team not in directions:
            continue
        adir = directions[a_team]
        # A's defensive zone: attacking-coord <= zone_frac (low = own half/build-up).
        def in_zone(x: float) -> bool:
            ac = x if adir == 1 else (PITCH_LEN - x)
            return ac <= zone_frac * PITCH_LEN
        ap = passes[(passes["team"] == a_team)]
        passes_allowed = int(sum(in_zone(x0) for x0 in ap["x0"].to_numpy()))
        pressures = 0
        carr = possession[possession["team"] == a_team]
        cpos = _carrier_positions(players)
        for r in carr.itertuples(index=False):
            cp = cpos.get((int(r.frame), int(r.carrier)))
            if cp is None or not in_zone(cp[0]):
                continue
            g = pos_by_frame.get(int(r.frame))
            if g is None:
                continue
            d = g[g["team"] == d_team]
            if len(d) and float(np.hypot(d["pitch_x"] - cp[0], d["pitch_y"] - cp[1]).min()) <= press_radius_m:
                pressures += 1
        rows.append({"team": int(d_team), "passes_allowed": passes_allowed, "pressures": pressures,
                     "ppda": passes_allowed / pressures if pressures else float("nan")})
    return pd.DataFrame(rows, columns=["team", "passes_allowed", "pressures", "ppda"])
