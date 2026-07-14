"""Off-screen player imputation (P2): reconstruct the missing ~5/11 players from the visible structure.

The audit's validity fix. Broadcast follow-play shows ~6/11 players and drops the **deep defenders** when
the camera tracks the ball forward, so the visible defensive line reads ~10 m too high — a bias the
pipeline currently papers over with a hand ``PARTIAL_VIEW_LINE_OFFSET``. This module removes the need for
that hand offset by *imputing* the missing players.

Approach (analytic, no GPU training — right-sized for our data): assign each backbone track a **role**
(``fingerprint.roles``) and its team's formation, then learn, per ``(team, role)``, that role's typical
**offset from its team's visible-outfield centroid** in the team's attacking frame, using only the frames
where the role IS visible (self-supervised). For a frame missing a formation slot, place a synthetic
player at ``centroid + offset``. A centre-back's offset-behind-centroid is stable across phases, so
imputing it during an attack (when it is off-screen and deep) restores the deep line — pulling the
line-height estimate back down toward the truth.

Validation lives in ``tools.impute_validate``: leave-one-visible-player-out accuracy (does the role offset
beat a centroid baseline?) and the resulting match-level line-height shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.roles import FORMATIONS, assign_roles
from core.pitch import PITCH_LEN
from fingerprint.structural_metrics import attacking_coord, resolve_attack_directions
from fingerprint.theory_metrics import complete_directions

PLAYERS = ("player", "goalkeeper")
MIN_VIS_FOR_CENTROID = 3   # need a few visible players to define a team centroid
# Outfield back-line role slots (GK excluded — the defensive LINE is the deepest outfielders).
DEF_ROLES = frozenset({"LB", "LCB", "CB", "RCB", "RB", "LWB", "RWB"})
LINE_PCTL = 20   # the deep-line quantile, matching structural_metrics.DEF_LINE_QUANTILE

# --- de-biased defensive line (the P2 line fix; see docs/PROJECT_AUDIT_2026-07.md P2) ---
# The line = mean of the deepest N outfielders (GK excluded) — the back line itself, not a percentile
# over all bodies (which sits ~12 m too shallow by definition). The censoring bias is then removed with
# ONE global slope: as fewer back-line defenders are visible the estimate reads higher, so we correct
# each frame to its "full back line visible" (n_back = BACK_REF) equivalent. The slope is fit
# (tools/fit_line_debias.py) over ALL processed matches — which INCLUDES the 3 FIFA France matches that
# tools/line_c6 validates against, so that FIFA validation is in-sample, not held-out (contamination note
# 2026-07-14). Measured: a clean refit on only the 2 non-FIFA matches gives -6.31 and raises the France
# per-phase mean |line - FIFA| from ~5.5 m (in-sample) to ~7.2 m (held-out). The scalar is stable
# (leave-one-match-out -5.95..-5.01, median -5.62); the constant is kept pending a METRICS_VERSION call.
DEEP_N = 4
BACK_REF = 4        # n_back_visible at which the line needs no correction (full back line seen)
LINE_DEBIAS_SLOPE = -5.62   # m per back-line defender seen (fit on ~168k pooled frames; nback predictor)


def _outfield(g: pd.DataFrame) -> pd.DataFrame:
    """Visible outfield players (role 'player', keeper excluded, with coordinates)."""
    o = g[g["role"] == "player"]
    if "is_keeper" in o.columns:
        o = o[o["is_keeper"] != True]  # noqa: E712
    return o.dropna(subset=["pitch_x", "pitch_y"])


def line_from_deepest(ax: np.ndarray, n: int = DEEP_N) -> float:
    """Defensive line = mean attacking-x of the deepest ``n`` outfielders (0 = own goal)."""
    a = np.sort(np.asarray(ax, float))
    return float(np.mean(a[:min(n, len(a))])) if len(a) else float("nan")


def line_estimates(aligned: pd.DataFrame, roles: pd.DataFrame | None = None,
                   slope: float = LINE_DEBIAS_SLOPE) -> pd.DataFrame:
    """Per (chunk, frame, team): the deepest-N line, its visibility, and the de-biased line.

    ``line_raw`` is the deepest-N back-line estimate (right *shape*, biased *scale*); ``line_debiased``
    removes the visibility-censoring inflation via the global slope, correcting to a full-back-line view
    while preserving the phase-to-phase variation. ``n_back`` is how many back-line role tracks were seen.
    """
    roles = assign_roles(aligned) if roles is None else roles
    role_map, _ = _chunk_meta(roles)
    rows = []
    for ck, g in (aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]):
        dirs = complete_directions(resolve_attack_directions(g))
        pl = _outfield(g)
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs or len(fg) < 3:
                continue
            ax = attacking_coord(fg["pitch_x"].to_numpy(), dirs[team])
            nback = sum(role_map.get((ck, int(t)), (None, None))[1] in DEF_ROLES for t in fg["track_id"])
            raw = line_from_deepest(ax)
            deb = raw + slope * (BACK_REF - min(nback, BACK_REF))
            rows.append({"chunk": ck, "frame": int(fr), "team": team, "n_vis": len(fg),
                         "n_back": nback, "line_raw": raw, "line_debiased": deb})
    return pd.DataFrame(rows)


def debias_slope(lines: pd.DataFrame, predictor: str = "n_back") -> float:
    """Global de-bias slope from a table of per-frame line estimates (pure regression).

    Regresses ``line_raw`` on the visibility predictor (``n_back`` clamped to ``BACK_REF``, or ``n_vis``
    clamped to 10). A negative slope means fewer visible defenders inflate the line — the coefficient we
    subtract to de-bias each frame to a full-back-line view.
    """
    cap = BACK_REF if predictor == "n_back" else 10
    x = np.clip(lines[predictor].to_numpy(float), 0, cap)
    b, _ = np.polyfit(x, lines["line_raw"].to_numpy(float), 1)
    return float(b)


def fit_line_debias(aligned_iter, predictor: str = "n_back") -> float:
    """Fit the single global de-bias slope over pooled frames from many matches (wraps :func:`debias_slope`)."""
    lf = pd.concat([line_estimates(a, slope=0.0) for a in aligned_iter], ignore_index=True)
    return debias_slope(lf, predictor)


def _formation_roles(name: str) -> list[str]:
    """The full set of role labels a formation template defines (its 11 slots)."""
    return [slot[0] for slot in FORMATIONS.get(name, [])]


def learn_role_offsets(aligned: pd.DataFrame, roles: pd.DataFrame | None = None) -> dict:
    """Per ``(team, role)``: mean attacking-frame offset ``(dax, day)`` from the team's visible centroid.

    Learned only from frames where the role's backbone track is visible (self-supervised). ``dax`` is the
    signed attacking-x offset (a deep defender has a large negative ``dax``); ``day`` the cross-pitch offset.
    """
    roles = assign_roles(aligned) if roles is None else roles
    role_map = {(r.chunk, int(r.track_id)): (int(r.team), r.role) for r in roles.itertuples(index=False)}
    acc: dict[tuple[int, str], list[tuple[float, float]]] = {}
    groups = aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]
    for ck, g in groups:
        dirs = complete_directions(resolve_attack_directions(g))
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs or len(fg) < MIN_VIS_FOR_CENTROID:
                continue
            adir = dirs[team]
            ax = attacking_coord(fg["pitch_x"].to_numpy(), adir)
            ay = fg["pitch_y"].to_numpy()
            cax, cay = float(ax.mean()), float(ay.mean())
            for tid, axi, ayi in zip(fg["track_id"].astype(int), ax, ay):
                tr = role_map.get((ck, int(tid)))
                if tr is None or tr[0] != team:
                    continue
                acc.setdefault((team, tr[1]), []).append((axi - cax, ayi - cay))
    return {k: (float(np.mean([o[0] for o in v])), float(np.mean([o[1] for o in v])), len(v))
            for k, v in acc.items()}


def _chunk_meta(roles: pd.DataFrame):
    """``(chunk,track_id)->(team,role)`` and ``(chunk,team)->formation-role-set`` from an assign_roles table."""
    role_map = {(r.chunk, int(r.track_id)): (int(r.team), r.role) for r in roles.itertuples(index=False)}
    formation = {(r.chunk, int(r.team)): r.formation for r in roles.itertuples(index=False)}
    return role_map, formation


def impute_positions(aligned: pd.DataFrame, roles: pd.DataFrame | None = None,
                     offsets: dict | None = None) -> pd.DataFrame:
    """Return ``aligned`` with synthetic rows added for missing formation slots (``imputed`` boolean column).

    For every (chunk, frame, team) we fill the formation roles not covered by a visible track, placing each
    at the team's visible-outfield centroid plus that role's learned offset. Real rows get ``imputed=False``.
    """
    roles = assign_roles(aligned) if roles is None else roles
    offsets = learn_role_offsets(aligned, roles) if offsets is None else offsets
    role_map, formation = _chunk_meta(roles)
    out = aligned.copy()
    if "imputed" not in out.columns:
        out["imputed"] = False
    synth = []
    groups = aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]
    for ck, g in groups:
        dirs = complete_directions(resolve_attack_directions(g))
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs:
                continue
            role_of = {int(t): role_map.get((ck, int(t)), (None, None))[1] for t in fg["track_id"]}
            synth.extend(impute_frame(fg, team=team, adir=dirs[team], role_of=role_of,
                                      froles=_formation_roles(formation.get((ck, team), "")),
                                      offsets=offsets, chunk=ck, frame=int(fr)))
    if synth:
        out = pd.concat([out, pd.DataFrame(synth)], ignore_index=True)
    return out


def reconstruct_defensive_line(aligned: pd.DataFrame, roles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (chunk, frame, team): the defensive line height, censored vs temporally reconstructed.

    The validity fix Fable flagged: this is an *offline* pipeline, so a back-line defender seen 10 s
    before or after an attack is unbiased evidence of where the line sat *during* the attack — evidence a
    per-frame spatial method throws away. For each team we take its back-line role tracks (``DEF_ROLES``),
    and **bidirectionally interpolate** each one's attacking-x across the frames it is off-screen
    (``np.interp`` holds the deep endpoints across the gap). The reconstructed line is the deep quantile of
    those de-censored back-line positions — which stays deep through an attack instead of inflating.

    Returns:
        ``chunk, frame, team, line_raw, line_recon, n_back_vis`` — ``line_raw`` is the shipped estimator
        (deep quantile of *visible* outfielders, the biased one); ``line_recon`` the reconstructed line.
    """
    roles = assign_roles(aligned) if roles is None else roles
    role_map, _ = _chunk_meta(roles)
    rows = []
    for ck, g in (aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]):
        dirs = complete_directions(resolve_attack_directions(g))
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for team in (0, 1):
            if team not in dirs:
                continue
            adir = dirs[team]
            tg = pl[pl["team"] == team]
            if tg.empty:
                continue
            frames = np.sort(tg["frame"].unique())
            # reconstruct each back-line role track across all frames of the chunk
            recon = []
            for tid in tg["track_id"].unique():
                if role_map.get((ck, int(tid)), (None, None))[1] not in DEF_ROLES:
                    continue
                t = tg[tg["track_id"] == tid].sort_values("frame")
                if t.empty:
                    continue
                ax = attacking_coord(t["pitch_x"].to_numpy(), adir)
                recon.append(np.interp(frames, t["frame"].to_numpy(), ax))  # bidirectional, endpoint-held
            recon = np.vstack(recon) if recon else None
            # per-frame raw (visible outfield deep quantile) + reconstructed back-line deep quantile
            vis_ax = {int(fr): attacking_coord(fg["pitch_x"].to_numpy(), adir)
                      for fr, fg in tg.groupby("frame")}
            back_vis = {int(fr): sum(role_map.get((ck, int(t)), (None, None))[1] in DEF_ROLES
                                     for t in fg["track_id"]) for fr, fg in tg.groupby("frame")}
            for j, fr in enumerate(frames):
                raw = float(np.percentile(vis_ax[int(fr)], LINE_PCTL))
                rec = float(np.percentile(recon[:, j], LINE_PCTL)) if recon is not None else raw
                rows.append({"chunk": ck, "frame": int(fr), "team": team,
                             "line_raw": raw, "line_recon": rec, "n_back_vis": back_vis[int(fr)]})
    return pd.DataFrame(rows)


def impute_frame(fg: pd.DataFrame, *, team: int, adir: int, role_of: dict[int, str],
                 froles: list[str], offsets: dict, chunk=None, frame=None) -> list[dict]:
    """Synthetic rows for the formation slots missing from one team's visible players in one frame.

    Args:
        fg: the team's visible outfield/keeper rows this frame (``track_id, pitch_x, pitch_y``).
        team, adir: team label and its attacking direction (+1/-1).
        role_of: ``track_id -> role`` for the visible tracks (chunk-level assignment).
        froles: the team's formation role slots (the target set of 11).
        offsets: ``(team, role) -> (dax, day, n)`` from :func:`learn_role_offsets`.

    Returns:
        A list of synthetic player dicts (``imputed=True``, ``track_id=-1``) — one per missing role that
        has a learned offset. Empty if too few visible players to anchor a centroid, or nothing missing.
    """
    if len(fg) < MIN_VIS_FOR_CENTROID or not froles:
        return []
    present = {role_of.get(int(t)) for t in fg["track_id"]}
    missing = [r for r in froles if r not in present]
    if not missing:
        return []
    ax = attacking_coord(fg["pitch_x"].to_numpy(), adir)
    cax, cay = float(ax.mean()), float(fg["pitch_y"].mean())
    rows = []
    for role in missing:
        off = offsets.get((team, role))
        if off is None:
            continue
        impute_ax = cax + off[0]
        pitch_x = impute_ax if adir > 0 else (PITCH_LEN - impute_ax)
        rows.append({"chunk": chunk, "frame": frame, "team": team, "role": "player",
                     "track_id": -1, "pitch_x": float(pitch_x), "pitch_y": cay + off[1],
                     "is_keeper": role == "GK", "calib_error_m": np.nan, "imputed": True})
    return rows
