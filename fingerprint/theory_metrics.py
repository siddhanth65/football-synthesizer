"""P1 theory metrics: coaching-literature concepts made measurable on partial broadcast tracking.

The audit codex (docs/PROJECT_AUDIT_2026-07.md section 5) translates tactical theory into metrics that
survive our data reality — only ~6/11 players visible, ball in ~68% of frames. The rule it imposes, and
that every function here obeys: prefer **ball-anchored local** reads (visibility is highest near the ball)
and **ratios**, never single-frame whole-team shapes. Each metric names the FIFA EFI number that
validates it (see ``tools/validate_theory_c6.py``).

Metrics:
* ``pressing_intensity`` — Bielsa/Klopp pressing, via a time-to-intercept model on the ball carrier
  (positions+velocity only). FIFA: *Defensive Pressures* / pressure on the ball.
* ``line_breaks`` — the ball played from in front of the opponent's defensive line to behind it.
  FIFA: *Completed Line Breaks*.
* ``local_overload`` — numerical superiority near the ball (Juego de Posición). FIFA: *Forced Turnovers*.
* ``verticality`` — directness of ball progression (Bielsa). FIFA: *Long Ball %*.
* ``lane_occupation`` — the 5-lane × 3-third occupancy matrix + half-space share (Guardiola).
  FIFA: *Receptions in the Final Third*.
* ``counterpress_curve`` — regain probability within 3/4/5/6/8 s of a loss (Gegenpressing).
  FIFA: *Counter-Press %* / ball recovery time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID
from fingerprint.structural_metrics import (
    DEF_LINE_QUANTILE,
    LANE_EDGES,
    LANE_NAMES,
    attacking_coord,
)
from fingerprint.style_metrics import build_tracks

DEFAULT_FPS = 50.0

# --- pressing-intensity (time-to-intercept) parameters (Bekkers, arXiv 2501.04712) ---
REACT_S = 0.7          # reaction time before a defender accelerates toward the carrier (s)
MAX_SPEED_MS = 7.0     # cap on how fast a defender closes (m/s)
CONTROL_RADIUS_M = 1.5  # a defender this close already contests the ball (0 arrival time inside it)
T_REF_S = 1.5          # time-to-arrive at which interception prob = 0.5
STEEP = 2.0            # logistic steepness in (T_REF - t)

# --- local-overload parameter ---
OVERLOAD_RADIUS_M = 12.0  # radius around the ball for counting numerical superiority

# --- counterpress-curve parameters ---
REGAIN_WINDOWS_S = (3.0, 4.0, 5.0, 6.0, 8.0)
CP_PRESS_RADIUS_M = 5.0   # a losing-team player this close to the ball = pressure on it


def complete_directions(directions: dict[int, int], teams=(0, 1)) -> dict[int, int]:
    """Fill a missing team's attack direction as the opposite of the known one (opposite-goal constraint).

    Broadcast follow-play often frames only one goalmouth, so the keeper resolver returns a direction for
    just one team. Since the two teams attack opposite goals, a single known direction determines the
    other — this keeps direction-dependent metrics (line breaks, verticality, lanes) from silently
    dropping the unresolved team.
    """
    known = {int(t): int(d) for t, d in directions.items() if int(t) in teams}
    if len(known) == 1:
        (t, d), = known.items()
        other = next(x for x in teams if x != t)
        known[other] = -d
    return known


def _carrier_stream(ball: pd.DataFrame, possession: pd.DataFrame) -> pd.DataFrame:
    """``frame, team, x, y`` for frames where a team holds the ball at a known ball location."""
    if ball.empty or possession.empty:
        return pd.DataFrame(columns=["frame", "team", "x", "y"])
    bxy = ball.dropna(subset=["x", "y"]).set_index("frame")[["x", "y"]]
    poss = possession[["frame", "team"]].drop_duplicates("frame").set_index("frame")
    j = poss.join(bxy, how="inner").reset_index()
    return j[j["team"] >= 0].reset_index(drop=True)


def pressing_intensity(ball: pd.DataFrame, possession: pd.DataFrame, players: pd.DataFrame, *,
                       fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Per-team pressing intensity: mean probability a defender reaches the ball carrier (0..1).

    For every frame a team holds the ball, each opponent (the *defending* team) has a time-to-intercept
    ``t = REACT_S + max(0, dist - CONTROL_RADIUS_M) / MAX_SPEED_MS`` toward the carrier; a turning penalty
    scales it by how much the defender must change heading. Interception probability is
    ``p = sigmoid(STEEP·(T_REF_S − t))`` and the frame's pressure is ``P = 1 − Π(1 − p)`` (probability at
    least one defender arrives). The team score is the mean ``P`` over the frames it defends — high = an
    aggressive, ball-oriented press. Ball-anchored, so robust to partial views (defenders near the ball
    are the visible ones). Needs velocity (built via :func:`build_tracks`).

    Returns:
        ``team, pressing_intensity, press_frames`` where ``team`` is the **defending/pressing** side.
    """
    cols = ["team", "pressing_intensity", "press_frames"]
    carr = _carrier_stream(ball, possession)
    if carr.empty:
        return pd.DataFrame(columns=cols)
    tr = build_tracks(players, fps=fps)
    if tr.empty:
        return pd.DataFrame(columns=cols)
    by_frame = {int(f): g for f, g in tr.groupby("frame")}
    acc: dict[int, list[float]] = {}
    for r in carr.itertuples(index=False):
        g = by_frame.get(int(r.frame))
        if g is None:
            continue
        d = g[g["team"] != r.team]
        d = d[d["team"] >= 0].dropna(subset=["x_s", "y_s"])
        if d.empty:
            continue
        dx, dy = r.x - d["x_s"].to_numpy(), r.y - d["y_s"].to_numpy()
        dist = np.hypot(dx, dy)
        # turning penalty: 1 (already facing carrier) .. 2 (facing away); NaN velocity -> no penalty
        vx, vy = d["vx"].to_numpy(), d["vy"].to_numpy()
        spd = np.hypot(vx, vy)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (vx * dx + vy * dy) / (spd * dist)
        turn = np.where(np.isfinite(cos), 1.5 - 0.5 * cos, 1.0)  # 1..2
        t = REACT_S + turn * np.maximum(0.0, dist - CONTROL_RADIUS_M) / MAX_SPEED_MS
        p = 1.0 / (1.0 + np.exp(-STEEP * (T_REF_S - t)))
        # pressure is credited to the DEFENDING team(s) present this frame
        for dt in np.unique(d["team"].to_numpy()):
            mask = d["team"].to_numpy() == dt
            P = 1.0 - np.prod(1.0 - p[mask])
            acc.setdefault(int(dt), []).append(float(P))
    return pd.DataFrame(
        [{"team": t, "pressing_intensity": float(np.mean(v)), "press_frames": len(v)}
         for t, v in sorted(acc.items()) if v], columns=cols)


def _team_def_line_ax(g: pd.DataFrame, team: int, adir: int) -> float | None:
    """Attacking-x of a team's defensive line (robust deep quantile), in the ATTACKER's frame."""
    d = g[(g["team"] == team)].dropna(subset=["pitch_x"])
    if len(d) < 3:
        return None
    # the defending team's deepest line, expressed on the attacker's 0..105 axis (so a break = ball ax
    # exceeding this value). The defender's own-goal side is low on the attacker axis.
    ax = attacking_coord(d["pitch_x"].to_numpy(), adir)
    return float(np.percentile(ax, 100 - DEF_LINE_QUANTILE))


LINE_BREAK_BAND_M = 5.0   # hysteresis: ball must move from >band in front to >band behind to count


def line_breaks(ball: pd.DataFrame, possession: pd.DataFrame, players: pd.DataFrame,
                directions: dict[int, int], *, min_frames_between: int = 25,
                band_m: float = LINE_BREAK_BAND_M) -> pd.DataFrame:
    """Per-team completed line breaks: the ball played from clearly in front of the opponent's defensive line to clearly behind it.

    For each in-possession frame we place the ball on the possessing team's attacking axis and compare it
    to the opponent defensive line's attacking-x (robust deep quantile). A **hysteresis band** (``band_m``)
    makes a break a genuine penetration, not jitter: the ball must first be > ``band_m`` in *front* of the
    line, then cross to > ``band_m`` *behind* it (``min_frames_between`` debounces repeats). The band is
    essential on partial broadcast views, where deep defenders are often off-screen and the visible line
    reads high — so we count only unambiguous crossings. Ball-anchored, direction-normalised.

    FIFA validator: *Completed Line Breaks* (directional / rank, not a count match — partial view undercounts).

    Returns:
        ``team, line_breaks, in_poss_frames`` (``team`` = the attacking team making the break).
    """
    cols = ["team", "line_breaks", "in_poss_frames"]
    carr = _carrier_stream(ball, possession)
    if carr.empty or len(directions) < 2:
        return pd.DataFrame(columns=cols)
    pbf = {int(f): g for f, g in players.dropna(subset=["pitch_x"]).groupby("frame")}
    # state: "front" once the ball is clearly in front; a break fires on reaching clearly-behind
    acc = {t: {"breaks": 0, "frames": 0, "armed": False, "last": -10**9} for t in directions}
    for r in carr.itertuples(index=False):
        team = int(r.team)
        if team not in directions:
            continue
        opp = next((t for t in directions if t != team), None)
        g = pbf.get(int(r.frame))
        if opp is None or g is None:
            continue
        adir = directions[team]
        line_ax = _team_def_line_ax(g, opp, adir)
        if line_ax is None:
            continue
        diff = attacking_coord(np.array([r.x]), adir)[0] - line_ax
        acc[team]["frames"] += 1
        if diff < -band_m:
            acc[team]["armed"] = True
        elif diff > band_m and acc[team]["armed"] and (r.frame - acc[team]["last"]) >= min_frames_between:
            acc[team]["breaks"] += 1
            acc[team]["last"] = int(r.frame)
            acc[team]["armed"] = False
    return pd.DataFrame(
        [{"team": t, "line_breaks": v["breaks"], "in_poss_frames": v["frames"]}
         for t, v in sorted(acc.items())], columns=cols)


def local_overload(ball: pd.DataFrame, possession: pd.DataFrame, players: pd.DataFrame, *,
                   radius_m: float = OVERLOAD_RADIUS_M) -> pd.DataFrame:
    """Per-team numerical superiority near the ball while in possession (Juego de Posición overloads).

    For each in-possession frame, count the possessing team's players vs the opponent's within
    ``radius_m`` of the ball; the signed difference is the local overload. We report the mean overload
    and the share of frames at ``+1`` or better — where broadcast visibility is highest, so this is one of
    the more trustworthy partial-view metrics.

    FIFA validator: *Forced Turnovers* (overloads reduce losses / force opponent turnovers).

    Returns:
        ``team, mean_overload, overload_share, frames`` (``team`` = the team in possession).
    """
    cols = ["team", "mean_overload", "overload_share", "frames"]
    carr = _carrier_stream(ball, possession)
    if carr.empty:
        return pd.DataFrame(columns=cols)
    pbf = {int(f): g for f, g in players.dropna(subset=["pitch_x", "pitch_y"]).groupby("frame")}
    acc: dict[int, list[int]] = {}
    for r in carr.itertuples(index=False):
        g = pbf.get(int(r.frame))
        if g is None:
            continue
        near = g[np.hypot(g["pitch_x"] - r.x, g["pitch_y"] - r.y) <= radius_m]
        atk = int((near["team"] == r.team).sum())
        deff = int((near["team"] == (1 - r.team)).sum()) if r.team in (0, 1) else \
            int((near["team"] != r.team).sum())
        acc.setdefault(int(r.team), []).append(atk - deff)
    return pd.DataFrame(
        [{"team": t, "mean_overload": float(np.mean(v)),
          "overload_share": float(np.mean(np.array(v) >= 1)), "frames": len(v)}
         for t, v in sorted(acc.items())], columns=cols)


def verticality(ball: pd.DataFrame, possession: pd.DataFrame, directions: dict[int, int], *,
                min_run_frames: int = 10) -> pd.DataFrame:
    """Per-team ball directness: forward (goalward) displacement over total path length, per possession spell.

    Within each unbroken possession spell of a team we sum the ball's forward (attacking-axis) progress and
    its total travelled distance; ``verticality = forward / path`` (1 = perfectly direct/vertical, ~0 =
    sideways circulation). Bielsa-direct sides score high, van-Gaal-circulation sides low. Ball-only.

    FIFA validator: *Long Ball %* (direct sides play more long balls).

    Returns:
        ``team, verticality, spells`` (verticality is the goalward-progress fraction of ball path length).
    """
    cols = ["team", "verticality", "spells"]
    carr = _carrier_stream(ball, possession)
    if carr.empty or len(directions) < 2:
        return pd.DataFrame(columns=cols)
    acc: dict[int, dict[str, float]] = {}
    spell: list = []
    spell_team = None

    def flush(team, rows):
        if team not in directions or len(rows) < min_run_frames:
            return
        adir = directions[team]
        ax = attacking_coord(np.array([p[1] for p in rows]), adir)
        ay = np.array([p[2] for p in rows])
        fwd = float(ax[-1] - ax[0])
        path = float(np.sum(np.hypot(np.diff(ax), np.diff(ay))))
        a = acc.setdefault(team, {"fwd": 0.0, "path": 0.0, "spells": 0})
        a["fwd"] += max(0.0, fwd)
        a["path"] += path
        a["spells"] += 1

    for r in carr.itertuples(index=False):
        if spell_team is None or int(r.team) == spell_team:
            spell.append((int(r.frame), r.x, r.y))
            spell_team = int(r.team)
        else:
            flush(spell_team, spell)
            spell, spell_team = [(int(r.frame), r.x, r.y)], int(r.team)
    flush(spell_team, spell)
    rows = []
    for t, a in sorted(acc.items()):
        rows.append({"team": t, "verticality": a["fwd"] / a["path"] if a["path"] else float("nan"),
                     "spells": int(a["spells"])})
    return pd.DataFrame(rows, columns=cols)


def lane_occupation(players: pd.DataFrame, directions: dict[int, int]) -> pd.DataFrame:
    """Per-team 5-lane × 3-third occupancy shares + half-space share (Guardiola's Juego de Posición).

    Bins every visible outfield/keeper player-frame into one of 5 width lanes × 3 length thirds (15 cells),
    direction-normalised so "attacking third" and "left wing" are team-relative. Reported as **shares**
    (fractions summing to 1), which are partial-view robust: missing players thin the sample but don't bias
    the ratio if missingness is roughly lane-uniform. ``halfspace_share`` sums the two half-space lanes.

    FIFA validator: *Receptions in the Final Third* / receptions behind lines.

    Returns:
        One row per team: ``team, halfspace_share, wing_share, centre_share, final_third_share`` plus the
        15 ``lane_third`` cells (e.g. ``left_halfspace_att``).
    """
    thirds = ("def", "mid", "att")
    rows = []
    pl = players[players["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    for team, g in pl.groupby("team"):
        team = int(team)
        if team not in directions:
            continue
        adir = directions[team]
        ax = attacking_coord(g["pitch_x"].to_numpy(), adir)
        ay = g["pitch_y"].to_numpy() if adir > 0 else (PITCH_WID - g["pitch_y"].to_numpy())
        lane = np.clip(np.digitize(ay, LANE_EDGES[1:-1]), 0, 4)
        third = np.clip((ax / PITCH_LEN * 3).astype(int), 0, 2)
        row: dict = {"team": team}
        cell = {}
        for li, ln in enumerate(LANE_NAMES):
            for ti, tn in enumerate(thirds):
                cell[f"{ln}_{tn}"] = float(np.mean((lane == li) & (third == ti)))
        row.update(cell)
        row["halfspace_share"] = float(np.mean((lane == 1) | (lane == 3)))
        row["wing_share"] = float(np.mean((lane == 0) | (lane == 4)))
        row["centre_share"] = float(np.mean(lane == 2))
        row["final_third_share"] = float(np.mean(third == 2))
        rows.append(row)
    return pd.DataFrame(rows)


def counterpress_curve(ball: pd.DataFrame, possession: pd.DataFrame, players: pd.DataFrame, *,
                       fps: float = DEFAULT_FPS, windows_s: tuple[float, ...] = REGAIN_WINDOWS_S,
                       press_radius_m: float = CP_PRESS_RADIUS_M) -> pd.DataFrame:
    """Per-team counterpress success curve: P(regain within t s of losing the ball) for t in ``windows_s``.

    On each turnover (possession switches team) we look forward; the losing team "regains" if it wins the
    ball back (possession flips back) within the window. We report the cumulative regain fraction at each
    horizon — Klopp/Rangnick's 5-second rule made into a curve. Uses possession attribution + ball only.

    FIFA validator: *Counter-Press %* / ball recovery time.

    Returns:
        One row per team: ``team, losses`` + ``regain_<t>s`` columns (fraction of losses regained by ``t``).
    """
    from fingerprint.transitions import detect_turnovers  # noqa: PLC0415

    cols = ["team", "losses"] + [f"regain_{int(t)}s" for t in windows_s]
    tv = detect_turnovers(possession, ball)
    if tv.empty:
        return pd.DataFrame(columns=cols)
    poss = possession.sort_values("frame").reset_index(drop=True)
    pframe = poss["frame"].to_numpy()
    pteam = poss["team"].to_numpy()
    max_w = max(windows_s)
    acc: dict[int, dict] = {}
    for t in tv.itertuples(index=False):
        lost, f = int(t.lost_team), int(t.frame)
        a = acc.setdefault(lost, {"losses": 0, "hits": {w: 0 for w in windows_s}})
        a["losses"] += 1
        # first frame after f (within max window) where the losing team holds again
        lo, hi = f, f + int(max_w * fps)
        seg = (pframe > lo) & (pframe <= hi)
        back = pframe[seg & (pteam == lost)]
        if len(back):
            dt = (back[0] - f) / fps
            for w in windows_s:
                if dt <= w:
                    a["hits"][w] += 1
    rows = []
    for team, a in sorted(acc.items()):
        row = {"team": team, "losses": a["losses"]}
        for w in windows_s:
            row[f"regain_{int(w)}s"] = a["hits"][w] / a["losses"] if a["losses"] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)
