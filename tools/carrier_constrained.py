"""Carrier identification as CONSTRAINED JOINT ASSIGNMENT (v2 of the closed-set probe).

``tools/carrier_attribution_probe.py`` picks the carrier's name by an independent per-moment argmax
over a per-team gallery. Sid's 90 hand labels score that at **end-to-end precision 0.35**
(``tools/score_carrier_labels.py``), and the error structure says why: the errors carry HIGH
appearance similarity (0.88-0.92), are WITHIN team, and are ROLE-IMPLAUSIBLE (CB -> CM, CB -> LW).
An independent argmax throws away everything else we know about the frame and about football.

This module rebuilds the decision as elimination + rescoring + joint assignment, with the two kinds
of knowledge kept strictly separate:

HARD (candidate removed from the set entirely -- it is provably or near-provably not that player):

* ``team`` -- the gallery is already restricted to the carrier's tracked team. Inherited from the
  v1 probe unchanged; reported here so the record is explicit (the observed errors are all
  within-team, so this constraint was never the missing piece).
* ``gallery`` -- a player with zero gallery crops can never be proposed. Also true by construction
  in v1. The labels' 0.00-precision "not gallery-covered" cell is the *converse* case (the TRUE
  carrier is absent), which no elimination can fix -- only abstention can. See ``territory_floor``.
* ``subwindow`` -- the player must be on the pitch at the estimated minute (Sofascore
  ``minutesPlayed`` windows via :func:`tools.make_carrier_labeller.roster_for`), with a generous
  +-:data:`MINUTE_TOL` because the broadcast clock is an estimate.
* ``exclusion`` -- a player the identity chain confidently placed on a DIFFERENT track in the same
  frame cannot also be the carrier.
* ``kinematic`` -- a player confidently identified nearby in time cannot have covered the distance
  to the carrier at more than :data:`MAX_SPEED_MS`. ``exclusion`` is the ``dt == 0`` case of this
  and shares its implementation; the two are counted separately.

SOFT (rescore only, never eliminate):

* ``territory`` -- per (player, half) pitch-position profile built from the player's own confident
  identifications; the candidate's Mahalanobis distance from it discounts the similarity. Aimed
  squarely at the CB -> winger confusions.
* ``hubness`` -- players with large galleries win the max-similarity argmax more often simply by
  having more chances (Bruno Fernandes has 151 gallery crops, others have 6). Subtracting a
  player's mean similarity over all queries removes that bias. Standard ReID hubness correction.
* ``joint`` -- consecutive carrier moments inside one possession string are solved TOGETHER with a
  Hungarian assignment, so two different tracks a second apart cannot both be the same player.

One knob deliberately goes beyond the "soft" brief and is labelled as such: ``territory_floor``
turns the positional prior into an ABSTENTION rule (if even the best candidate is positionally
absurd, answer nothing). That is the only mechanism available against the uncovered-gallery cell.

Evaluation discipline: everything is tuned on ``manutd_liverpool``'s 30 labels, frozen to
:data:`FROZEN_PATH`, and only then applied to the 60 held-out labels of ``manutd_tottenham`` +
``manutd_brighton``. Metrics are computed by :func:`tools.score_carrier_labels.score_labels`
verbatim, so the comparison against the 0.35 baseline is like-for-like.

Run (CPU except ``jersey``)::

    python -m tools.carrier_constrained --stage build      # candidate tables from cached embeddings
    python -m tools.carrier_constrained --stage tune       # dev-match grid -> frozen config
    python -m tools.carrier_constrained --stage score      # dev + held-out + trade-off curve
    python -m tools.carrier_constrained --stage jersey     # GPU: jersey numbers on carrier crops
    python -m tools.carrier_constrained --stage jersey_score
    python -m tools.carrier_constrained --stage selftest   # pure-seam asserts
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from core import registry
from tools.carrier_attribution_probe import OUT_DIR, PROBE_MATCHES
from tools.make_carrier_labeller import minute_estimate, roster_for
from tools.score_carrier_labels import DEFAULT_LABELS, load_labels, norm_name, score_labels

#: Threshold-selection match; the rest are held out.
DEV_MATCH = PROBE_MATCHES[0]
HELDOUT_MATCHES = PROBE_MATCHES[1:]
FROZEN_PATH = OUT_DIR / "frozen_constrained.json"
REPORT_PATH = Path("results/CARRIER_CONSTRAINED_v2.md")

#: Broadcast-clock slack on the substitution window (minutes). The minute is derived from fixed
#: 600 s chunk cuts, so stoppage time and the exact half split are unmodelled.
MINUTE_TOL = 5.0
#: Sprint ceiling for the kinematic constraint.
MAX_SPEED_MS = 10.0
#: Only anchors this close in time are used for the kinematic/exclusion check.
ANCHOR_DT_S = 3.0
#: Tracking/calibration slack subtracted from the anchor->carrier distance before the speed test.
ANCHOR_SLACK_M = 2.0
#: Floor on a territory profile's per-axis sigma (m) and the minimum anchor samples to trust one.
TERRITORY_SIGMA_FLOOR = 5.0
TERRITORY_MIN_N = 20
#: Mahalanobis distance is clipped here before it discounts similarity.
TERRITORY_CLIP = 5.0
#: Two carrier moments belong to the same possession string if they share a chunk, a team, and are
#: within this many seconds of each other.
POSSESSION_GAP_S = 8.0

#: Dev-match tuning grid (small on purpose: the dev split has ~15 assigned moments).
GRID_W_TERRITORY = (0.0, 0.01, 0.03)
GRID_TERRITORY_FLOOR = (None, 3.0, 2.5)
GRID_W_HUBNESS = (0.0, 1.0)
GRID_JOINT = (False, True)
#: A tuned point must keep at least this many dev assignments to be eligible (no degenerate n=1).
MIN_DEV_ASSIGNED = 8
#: Operating-point grid for the coverage/precision trade-off curve.
CURVE_SIM = (0.80, 0.84, 0.88, 0.92)
CURVE_MARGIN = (0.0, 0.01, 0.02, 0.04, 0.08)

HARD_CONSTRAINTS = ("team", "gallery", "subwindow", "exclusion", "kinematic")


# === inputs ======================================================================================
def named_positions(match_id: str) -> pd.DataFrame:
    """Pitch positions of every frame of every track the identity chain confidently named.

    These are the only "confident prior identifications" available, and they feed three things: the
    in-frame mutual-exclusion constraint, the kinematic constraint, and the territory prior.

    Args:
        match_id: registry match id.

    Returns:
        ``chunk, frame, track_id, player, half, pitch_x, pitch_y`` (rows without calibration
        dropped). Empty frame when the match has no named-tracks parquet.
    """
    path = Path(f"outputs/identity/{match_id}_named_tracks_both2_prtreid.parquet")
    cols = ["chunk", "frame", "track_id", "player", "half", "pitch_x", "pitch_y"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    named = pd.read_parquet(path)[["chunk", "track_id", "player_name"]]
    df = registry.get(match_id).load_aligned()
    df = df[df["role"].isin(["player", "goalkeeper"])]
    out = df.merge(named, on=["chunk", "track_id"], how="inner")
    out = out.dropna(subset=["pitch_x", "pitch_y"])
    out = out.rename(columns={"player_name": "player"})
    out["half"] = out["chunk"].str[:2]
    return out[cols].reset_index(drop=True)


def query_context(match_id: str) -> pd.DataFrame:
    """Per carrier-query context: pitch position, half, estimated match minute.

    Args:
        match_id: registry match id with cached probe artifacts.

    Returns:
        The resolvable carrier events plus ``half``, ``t_s``, ``minute``, ``pitch_x``, ``pitch_y``.
    """
    match = registry.get(match_id)
    ev = pd.read_parquet(OUT_DIR / f"{match_id}_events.parquet")
    q = ev[ev["resolvable"]].reset_index(drop=True).copy()
    df = match.load_aligned()[["chunk", "frame", "track_id", "pitch_x", "pitch_y"]]
    q = q.merge(df, on=["chunk", "frame", "track_id"], how="left")
    fps = {c: match.chunk_fps(c) for c in q["chunk"].unique()}
    q["fps"] = q["chunk"].map(fps)
    q["t_s"] = q["frame"] / q["fps"]
    q["minute"] = [minute_estimate(c, int(f), float(s))
                   for c, f, s in zip(q["chunk"], q["frame"], q["fps"])]
    q["half"] = q["chunk"].str[:2]
    q["qi"] = np.arange(len(q))
    return q


def candidate_table(match_id: str) -> pd.DataFrame:
    """Long ``(query, candidate player)`` table with the raw appearance similarity.

    Reproduces :func:`tools.carrier_attribution_probe.identify` exactly -- same per-team gallery,
    same leave-one-track-out, same "best crop of that player" similarity -- but keeps ALL candidates
    instead of collapsing to the argmax, which is what elimination needs.

    Args:
        match_id: registry match id with cached embeddings.

    Returns:
        ``qi, player, sim, n_crops`` (one row per surviving candidate player per query).
    """
    ev = pd.read_parquet(OUT_DIR / f"{match_id}_events.parquet")
    q = ev[ev["resolvable"]].reset_index(drop=True)
    gal = pd.read_parquet(OUT_DIR / f"{match_id}_gallery.parquet").reset_index(drop=True)
    qe = np.load(OUT_DIR / f"{match_id}_query_emb.npy")
    ge = np.load(OUT_DIR / f"{match_id}_gallery_emb.npy")
    ok_g = np.linalg.norm(ge, axis=1) > 0.5
    players = gal["player"].to_numpy()
    g_team, g_track, g_chunk = (gal["team"].to_numpy(), gal["track_id"].to_numpy(),
                                gal["chunk"].to_numpy())
    rows: list[dict] = []
    for i, r in enumerate(q.itertuples(index=False)):
        if np.linalg.norm(qe[i]) <= 0.5:
            continue
        mask = ok_g & (g_team == r.team) & ~((g_chunk == r.chunk) & (g_track == int(r.track_id)))
        if not mask.any():
            continue
        sims = ge[mask] @ qe[i]
        pl = players[mask]
        best: dict[str, float] = {}
        n_crops: dict[str, int] = {}
        for p, s in zip(pl, sims):
            n_crops[p] = n_crops.get(p, 0) + 1
            if s > best.get(p, -2.0):
                best[p] = float(s)
        for p, s in best.items():
            rows.append({"qi": i, "player": p, "sim": s, "n_crops": n_crops[p]})
    return pd.DataFrame(rows)


# === hard constraints ============================================================================
def subwindow_flags(cand: pd.DataFrame, ctx: pd.DataFrame, match_id: str) -> np.ndarray:
    """``True`` where the candidate was provably off the pitch at the query's estimated minute.

    Args:
        cand: long candidate table.
        ctx: :func:`query_context` output (indexed by ``qi`` order).
        match_id: registry match id (for the Sofascore on-pitch windows).

    Returns:
        Boolean array aligned with ``cand``.
    """
    roster = {norm_name(p["name"]): p for p in roster_for(match_id)["players"]}
    minute = ctx.set_index("qi")["minute"].to_dict()
    out = np.zeros(len(cand), bool)
    for k, (qi, player) in enumerate(zip(cand["qi"], cand["player"])):
        rec = roster.get(norm_name(player))
        if rec is None:
            continue
        m = float(minute[qi])
        if rec["unused"]:
            out[k] = True
        elif m + MINUTE_TOL < rec["on_from"] or m - MINUTE_TOL > rec["on_to"]:
            out[k] = True
    return out


def anchor_flags(cand: pd.DataFrame, ctx: pd.DataFrame,
                 anchors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mutual-exclusion (``dt == 0``) and kinematic (``dt > 0``) elimination flags.

    For every candidate player, the confident identifications of that player on OTHER tracks within
    :data:`ANCHOR_DT_S` of the query are checked: the implied speed from the anchor to the carrier's
    position must not exceed :data:`MAX_SPEED_MS` (after :data:`ANCHOR_SLACK_M` of tracking slack).
    A same-frame anchor more than the slack away is an outright contradiction -- the player is
    somewhere else in this very frame.

    Args:
        cand: long candidate table.
        ctx: :func:`query_context` output.
        anchors: :func:`named_positions` output for the same match.

    Returns:
        ``(exclusion, kinematic)`` boolean arrays aligned with ``cand``.
    """
    excl = np.zeros(len(cand), bool)
    kine = np.zeros(len(cand), bool)
    if anchors.empty:
        return excl, kine
    by_chunk: dict[str, dict] = {}
    for chunk, g in anchors.groupby("chunk"):
        g = g.sort_values("frame")
        by_chunk[str(chunk)] = {
            "frame": g["frame"].to_numpy(np.int64), "track": g["track_id"].to_numpy(np.int64),
            "player": g["player"].to_numpy(), "x": g["pitch_x"].to_numpy(float),
            "y": g["pitch_y"].to_numpy(float)}
    c = ctx.set_index("qi")
    by_q: dict[int, list[int]] = {}
    for k, qi in enumerate(cand["qi"]):
        by_q.setdefault(int(qi), []).append(k)
    names = cand["player"].to_numpy()
    for qi, idxs in by_q.items():
        row = c.loc[qi]
        a = by_chunk.get(str(row["chunk"]))
        if a is None or not np.isfinite(row["pitch_x"]):
            continue
        span = ANCHOR_DT_S * float(row["fps"])
        lo, hi = np.searchsorted(a["frame"], [row["frame"] - span, row["frame"] + span])
        if hi <= lo:
            continue
        sl = slice(int(lo), int(hi))
        keep = a["track"][sl] != int(row["track_id"])
        if not keep.any():
            continue
        dt = np.abs(a["frame"][sl][keep] - int(row["frame"])) / float(row["fps"])
        dist = np.hypot(a["x"][sl][keep] - float(row["pitch_x"]),
                        a["y"][sl][keep] - float(row["pitch_y"]))
        need = np.maximum(dist - ANCHOR_SLACK_M, 0.0)
        bad = need > MAX_SPEED_MS * dt
        if not bad.any():
            continue
        ap = a["player"][sl][keep]
        same_frame = {p for p, b, d in zip(ap, bad, dt) if b and d == 0.0}
        later = {p for p, b, d in zip(ap, bad, dt) if b and d > 0.0}
        for k in idxs:
            if names[k] in same_frame:
                excl[k] = True
            elif names[k] in later:
                kine[k] = True
    return excl, kine


# === soft priors =================================================================================
def territory_profiles(anchors: pd.DataFrame) -> dict[tuple[str, str], tuple[float, ...]]:
    """Per ``(player, half)`` pitch-position profile from that player's confident identifications.

    Half-specific because teams swap ends; a player with no anchors in one half inherits the other
    half's profile mirrored through the pitch centre.

    Args:
        anchors: :func:`named_positions` output.

    Returns:
        ``{(player, half): (mean_x, mean_y, sigma_x, sigma_y, n)}``.
    """
    from core.pitch import PITCH_LEN, PITCH_WID  # noqa: PLC0415

    out: dict[tuple[str, str], tuple[float, ...]] = {}
    if anchors.empty:
        return out
    for (player, half), g in anchors.groupby(["player", "half"]):
        if len(g) < TERRITORY_MIN_N:
            continue
        out[(str(player), str(half))] = (
            float(g["pitch_x"].mean()), float(g["pitch_y"].mean()),
            max(float(g["pitch_x"].std(ddof=0)), TERRITORY_SIGMA_FLOOR),
            max(float(g["pitch_y"].std(ddof=0)), TERRITORY_SIGMA_FLOOR), float(len(g)))
    for (player, half), prof in list(out.items()):
        other = "h2" if half == "h1" else "h1"
        if (player, other) not in out:
            out[(player, other)] = (PITCH_LEN - prof[0], PITCH_WID - prof[1],
                                    prof[2], prof[3], 0.0)
    return out


def territory_distance(cand: pd.DataFrame, ctx: pd.DataFrame,
                       profiles: dict[tuple[str, str], tuple[float, ...]]) -> np.ndarray:
    """Mahalanobis distance of each candidate's profile from the carrier's actual pitch position.

    Args:
        cand: long candidate table.
        ctx: :func:`query_context` output.
        profiles: :func:`territory_profiles` output.

    Returns:
        Float array aligned with ``cand``; ``0.0`` (i.e. no penalty) where no profile exists or the
        query has no calibrated position -- the prior abstains rather than guesses.
    """
    c = ctx.set_index("qi")
    px = c["pitch_x"].to_dict()
    py = c["pitch_y"].to_dict()
    half = c["half"].to_dict()
    out = np.zeros(len(cand), float)
    for k, (qi, player) in enumerate(zip(cand["qi"], cand["player"])):
        prof = profiles.get((player, half[qi]))
        x, y = px[qi], py[qi]
        if prof is None or not np.isfinite(x) or not np.isfinite(y):
            continue
        out[k] = float(np.hypot((x - prof[0]) / prof[2], (y - prof[1]) / prof[3]))
    return out


def hubness_bias(cand: pd.DataFrame) -> np.ndarray:
    """Per-player mean similarity over all this match's queries (the hubness attractor term)."""
    mean = cand.groupby("player")["sim"].transform("mean")
    return mean.to_numpy(float)


# === assignment ==================================================================================
def possession_groups(ctx: pd.DataFrame) -> np.ndarray:
    """Possession-string id per query: same chunk + team, consecutive within :data:`POSSESSION_GAP_S`.

    Args:
        ctx: :func:`query_context` output.

    Returns:
        Integer group id aligned with ``ctx`` row order.
    """
    order = ctx.sort_values(["chunk", "team", "t_s"])
    gid = np.zeros(len(ctx), np.int64)
    g = -1
    prev_key: tuple | None = None
    prev_t = -1e9
    for qi, chunk, team, t in zip(order["qi"], order["chunk"], order["team"], order["t_s"]):
        key = (chunk, int(team))
        if key != prev_key or t - prev_t > POSSESSION_GAP_S:
            g += 1
        gid[int(qi)] = g
        prev_key, prev_t = key, float(t)
    return gid


def solve(cand: pd.DataFrame, ctx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the configured constraints and produce one answer per query.

    Args:
        cand: long candidate table with ``kept``, ``score`` and ``sim`` columns already present.
        ctx: :func:`query_context` output.
        cfg: configuration dict; only ``joint`` is read here.

    Returns:
        ``qi, pred_player, best_sim, margin, terr_d`` -- one row per query that still has a
        candidate. ``best_sim`` is the RAW cosine similarity of the chosen candidate and ``margin``
        is the score gap between the query's own top-1 and top-2, so the frozen gate keeps its v1
        meaning even when the joint solver overrules the local argmax.
    """
    kept = cand[cand["kept"]].copy()
    if kept.empty:
        return pd.DataFrame(columns=["qi", "pred_player", "best_sim", "margin", "terr_d"])
    kept = kept.sort_values(["qi", "score"], ascending=[True, False])
    top = kept.groupby("qi").head(1).set_index("qi")
    second = kept.groupby("qi").nth(1).set_index("qi")["score"]
    choice = top["player"].to_dict()
    if cfg.get("joint"):
        gid = possession_groups(ctx)
        kept["gid"] = kept["qi"].map(lambda q: gid[int(q)])
        for _, g in kept.groupby("gid"):
            qis = sorted(g["qi"].unique())
            if len(qis) < 2:
                continue
            names = sorted(g["player"].unique())
            if len(names) < len(qis):
                continue
            cost = np.full((len(qis), len(names)), 1e3)
            qpos = {q: i for i, q in enumerate(qis)}
            npos = {n: j for j, n in enumerate(names)}
            for q, p, s in zip(g["qi"], g["player"], g["score"]):
                cost[qpos[int(q)], npos[p]] = -float(s)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e2:
                    choice[qis[i]] = names[j]
    sim = {(int(q), p): float(s) for q, p, s in zip(kept["qi"], kept["player"], kept["sim"])}
    terr = {(int(q), p): float(d) for q, p, d in zip(kept["qi"], kept["player"], kept["terr_d"])}
    rows = []
    for qi, player in choice.items():
        top2 = float(second.get(qi, -1.0))
        rows.append({"qi": int(qi), "pred_player": player,
                     "best_sim": sim[(int(qi), player)],
                     "margin": float(top.loc[qi, "score"]) - top2,
                     "terr_d": terr[(int(qi), player)]})
    return pd.DataFrame(rows)


def prepare(match_id: str) -> dict:
    """Load / build every per-match input the constraint stack needs (cached to parquet)."""
    cpath = OUT_DIR / f"{match_id}_candidates.parquet"
    if cpath.exists():
        cand = pd.read_parquet(cpath)
        ctx = pd.read_parquet(OUT_DIR / f"{match_id}_qctx.parquet")
        anchors = pd.read_parquet(OUT_DIR / f"{match_id}_anchors.parquet")
    else:
        cand, ctx = candidate_table(match_id), query_context(match_id)
        anchors = named_positions(match_id)
        cand.to_parquet(cpath, index=False)
        ctx.to_parquet(OUT_DIR / f"{match_id}_qctx.parquet", index=False)
        anchors.to_parquet(OUT_DIR / f"{match_id}_anchors.parquet", index=False)
    cand = cand.copy()
    cand["elim_subwindow"] = subwindow_flags(cand, ctx, match_id)
    excl, kine = anchor_flags(cand, ctx, anchors)
    cand["elim_exclusion"] = excl
    cand["elim_kinematic"] = kine
    cand["terr_d"] = territory_distance(cand, ctx, territory_profiles(anchors))
    cand["hub"] = hubness_bias(cand)
    return {"match": match_id, "cand": cand, "ctx": ctx, "anchors": anchors}


def run_config(prep: dict, cfg: dict) -> pd.DataFrame:
    """Score one configuration on one prepared match.

    Args:
        prep: :func:`prepare` output.
        cfg: ``subwindow``/``exclusion``/``kinematic`` (bool), ``w_territory``, ``territory_floor``
            (``None`` disables), ``w_hubness``, ``joint``.

    Returns:
        Prediction rows in :func:`tools.score_carrier_labels.load_predictions` schema
        (``match, chunk, frame, pred_player, best_sim, margin``) before the operating-point gate,
        plus ``qi`` and ``terr_d``.
    """
    cand, ctx = prep["cand"].copy(), prep["ctx"]
    kept = np.ones(len(cand), bool)
    for name in ("subwindow", "exclusion", "kinematic"):
        if cfg.get(name, True):
            kept &= ~cand[f"elim_{name}"].to_numpy()
    floor = cfg.get("territory_floor")
    if floor is not None:
        kept &= cand["terr_d"].to_numpy() <= float(floor)
    cand["kept"] = kept
    cand["score"] = (cand["sim"]
                     - float(cfg.get("w_territory", 0.0))
                     * np.clip(cand["terr_d"], 0.0, TERRITORY_CLIP)
                     - float(cfg.get("w_hubness", 0.0)) * cand["hub"])
    out = solve(cand, ctx, cfg)
    key = ctx.set_index("qi")[["chunk", "frame"]]
    out = out.merge(key, left_on="qi", right_index=True, how="left")
    out["match"] = prep["match"]
    out["frame"] = out["frame"].astype("Int64")
    return out


def gate(preds: pd.DataFrame, min_sim: float, min_margin: float) -> pd.DataFrame:
    """Apply the operating point, returning the scorer's prediction schema (``None`` = abstain)."""
    out = preds.copy()
    ok = (out["best_sim"] >= min_sim) & (out["margin"] >= min_margin)
    out["pred_player"] = out["pred_player"].where(ok, other=None)
    out["min_sim"], out["min_margin"] = min_sim, min_margin
    return out[["match", "chunk", "frame", "pred_player", "best_sim", "margin",
                "min_sim", "min_margin"]]


# === evaluation ==================================================================================
def gallery_sets() -> dict[str, set[str]]:
    """Per-match set of normalised gallery player names (for the coverage split)."""
    out: dict[str, set[str]] = {}
    for mid in PROBE_MATCHES:
        path = OUT_DIR / f"{mid}_gallery.parquet"
        if path.exists():
            out[mid] = {norm_name(p) for p in pd.read_parquet(path)["player"].unique()}
    return out


def evaluate(preds: pd.DataFrame, labels: pd.DataFrame, matches: tuple[str, ...],
             min_sim: float, min_margin: float) -> dict:
    """Score a prediction set against the hand labels on a match subset (like-for-like with v1)."""
    lab = labels[labels["match"].isin(matches)]
    return score_labels(lab, gate(preds[preds["match"].isin(matches)], min_sim, min_margin),
                        gallery_sets())


def _cfg_grid() -> list[dict]:
    """The dev-match tuning grid (hard constraints are always on)."""
    out = []
    for w_t, floor, w_h, joint in itertools.product(
            GRID_W_TERRITORY, GRID_TERRITORY_FLOOR, GRID_W_HUBNESS, GRID_JOINT):
        out.append({"subwindow": True, "exclusion": True, "kinematic": True,
                    "w_territory": w_t, "territory_floor": floor, "w_hubness": w_h,
                    "joint": joint})
    return out


def _predict_all(preps: dict[str, dict], cfg: dict) -> pd.DataFrame:
    return pd.concat([run_config(p, cfg) for p in preps.values()], ignore_index=True)


# === jersey-number route (independent of appearance ReID) ========================================
#: Best available jersey checkpoint (`results/jersey_model/JERSEY_MODEL.md`, Stage-1c torso crop).
JERSEY_CKPT = Path("outputs/jersey/jersey_torso_r224_acc417.pt")
#: Crops of the carrier's own track fed to the recognizer, nearest in time to the kick.
JERSEY_CROPS = 5
#: Confidence floors swept when reporting the legibility / precision trade-off.
JERSEY_CONF_GRID = (0.05, 0.20, 0.40, 0.60, 0.80)


def _carrier_crop_frames(match_id: str, ctx: pd.DataFrame) -> pd.DataFrame:
    """Up to :data:`JERSEY_CROPS` frames of each carrier's own track around the kick moment."""
    df = registry.get(match_id).load_aligned()
    df = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["image_x", "image_y"])
    idx = df.set_index(["chunk", "track_id"]).sort_index()
    rows: list[dict] = []
    for r in ctx.itertuples(index=False):
        try:
            sub = idx.loc[(r.chunk, int(r.track_id))]
        except KeyError:
            continue
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        sub = sub.iloc[(sub["frame"] - int(r.frame)).abs().argsort()[:JERSEY_CROPS]]
        for s in sub.itertuples(index=False):
            rows.append({"qi": int(r.qi), "chunk": r.chunk, "frame": int(s.frame),
                         "image_x": float(s.image_x), "image_y": float(s.image_y)})
    return pd.DataFrame(rows)


def run_jersey(match_id: str) -> Path:
    """Read a jersey number off each carrier's crops (single GPU job).

    Roster-constrained decoding: the softmax is masked to the shirt numbers of the carrier's TRACKED
    team, so a number maps to exactly one rostered player. Aggregation is the module's own
    confidence-weighted tracklet vote over the crops nearest the kick.

    Args:
        match_id: registry match id with cached probe artifacts.

    Returns:
        Path of the written ``<match>_jersey.parquet`` (``qi, number, conf, n_crops``).
    """
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from generator import jersey_id as J  # noqa: PLC0415
    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    ctx = pd.read_parquet(OUT_DIR / f"{match_id}_qctx.parquet")
    req = _carrier_crop_frames(match_id, ctx)
    rec = J.JerseyRecognizer.from_checkpoint(JERSEY_CKPT)
    roster = roster_for(match_id)["players"]
    masks = {t: J.roster_mask([p["shirt"] for p in roster if p["team"] == t]) for t in (0, 1)}
    q_team = ctx.set_index("qi")["team"].to_dict()
    probs: dict[int, list[np.ndarray]] = {}
    for chunk_key, g in req.groupby("chunk", sort=True):
        video = registry.VIDEO_ROOT / match_id / str(chunk_key)[:2] / \
            f"chunk_{str(chunk_key).split('_chunk_')[1]}.mp4"
        if not video.exists():
            print(f"WARN missing video {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        tensors: list = []
        owners: list[int] = []
        for r in g.sort_values("frame").itertuples(index=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, bgr = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0 or y2 - y1 < 24:
                continue
            tensors.append(rec.tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
            owners.append(int(r.qi))
        cap.release()
        if not tensors:
            continue
        out = []
        with torch.no_grad():
            for i in range(0, len(tensors), 256):
                xb = torch.stack(tensors[i:i + 256]).to(rec.device)
                with torch.amp.autocast(rec.device, enabled=rec.device == "cuda"):
                    raw = rec.model(xb)
                out.append(torch.softmax(raw.float(), dim=1).cpu().numpy())
        arr = np.concatenate(out)
        for k, qi in enumerate(owners):
            probs.setdefault(qi, []).append(arr[k])
        print(f"  {chunk_key}: {len(owners)} crops")
    rows = []
    for qi, plist in probs.items():
        pooled = J.tracklet_mean(np.stack(plist))
        for conf_floor in (0.0,):
            num, conf = J.decide(pooled, min_conf=conf_floor, mask=masks[int(q_team[qi])])
        rows.append({"qi": qi, "number": int(num), "conf": float(conf), "n_crops": len(plist)})
    out_path = OUT_DIR / f"{match_id}_jersey.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"{match_id}: jersey reads for {len(rows)} carriers -> {out_path}")
    return out_path


def jersey_predictions(match_id: str, min_conf: float) -> pd.DataFrame:
    """Number-based carrier identification at a confidence floor (scorer prediction schema)."""
    jd = pd.read_parquet(OUT_DIR / f"{match_id}_jersey.parquet")
    ctx = pd.read_parquet(OUT_DIR / f"{match_id}_qctx.parquet")
    roster = roster_for(match_id)["players"]
    by_num = {(p["team"], p["shirt"]): p["name"] for p in roster}
    j = jd.merge(ctx[["qi", "chunk", "frame", "team"]], on="qi", how="left")
    names = [by_num.get((int(t), int(n))) if n > 0 and c >= min_conf else None
             for t, n, c in zip(j["team"], j["number"], j["conf"])]
    return pd.DataFrame({"match": match_id, "chunk": j["chunk"],
                         "frame": j["frame"].astype("Int64"), "pred_player": names,
                         "best_sim": j["conf"], "margin": j["conf"],
                         "min_sim": -1.0, "min_margin": -1.0})


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["build", "tune", "score", "jersey", "jersey_score", "selftest"])
    ap.add_argument("--match", default=None, help="match id (jersey stage)")
    ap.add_argument("--min-sim", type=float, default=0.88)
    ap.add_argument("--min-margin", type=float, default=0.01)
    a = ap.parse_args()
    if a.stage == "selftest":
        _selftest()
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage == "jersey":
        for mid in ([a.match] if a.match else list(PROBE_MATCHES)):
            run_jersey(mid)
        return
    if a.stage == "jersey_score":
        labels = load_labels(DEFAULT_LABELS)
        rows = []
        for conf in JERSEY_CONF_GRID:
            preds = pd.concat([jersey_predictions(m, conf) for m in PROBE_MATCHES],
                              ignore_index=True)
            read = preds["pred_player"].notna()
            for tag, ms in (("dev", (DEV_MATCH,)), ("held", HELDOUT_MATCHES),
                            ("all3", PROBE_MATCHES)):
                h = score_labels(labels[labels["match"].isin(ms)],
                                 preds[preds["match"].isin(ms)], gallery_sets())["headline"]
                rows.append({"min_conf": conf, "split": tag,
                             "n_carriers": int(preds["match"].isin(ms).sum()),
                             "number_read_rate": round(
                                 float(read[preds["match"].isin(ms)].mean()), 3),
                             "n_labelled_assigned": h["n_assigned_all"],
                             "e2e": h["end_to_end_precision"],
                             "ident": h["identification_precision"]})
        out = pd.DataFrame(rows)
        out.to_csv(OUT_DIR / "jersey_carrier_scoring.csv", index=False, encoding="utf-8")
        print(out.to_string(index=False))
        return
    preps = {m: prepare(m) for m in PROBE_MATCHES}
    if a.stage == "build":
        for mid, p in preps.items():
            c = p["cand"]
            print(f"{mid}: {c['qi'].nunique()} queries, {len(c)} candidates, "
                  f"elim sub={int(c['elim_subwindow'].sum())} "
                  f"excl={int(c['elim_exclusion'].sum())} kin={int(c['elim_kinematic'].sum())} "
                  f"terr_d>3={int((c['terr_d'] > 3).sum())}")
        return
    labels = load_labels(DEFAULT_LABELS)
    if a.stage == "tune":
        rows = []
        for cfg in _cfg_grid():
            preds = _predict_all({DEV_MATCH: preps[DEV_MATCH]}, cfg)
            res = evaluate(preds, labels, (DEV_MATCH,), a.min_sim, a.min_margin)
            hl = res["headline"]
            rows.append({**{k: cfg[k] for k in
                            ("w_territory", "territory_floor", "w_hubness", "joint")},
                         "n_assigned": hl["n_assigned_all"],
                         "e2e": hl["end_to_end_precision"],
                         "ident": hl["identification_precision"]})
        sw = pd.DataFrame(rows)
        sw.to_csv(OUT_DIR / "constrained_dev_sweep.csv", index=False, encoding="utf-8")
        print(sw.to_string(index=False))
        ok = sw[sw["n_assigned"] >= MIN_DEV_ASSIGNED].copy()
        ok = ok.sort_values(["e2e", "n_assigned"], ascending=[False, False])
        best = ok.iloc[0]
        frozen = {"subwindow": True, "exclusion": True, "kinematic": True,
                  "w_territory": float(best["w_territory"]),
                  "territory_floor": (None if pd.isna(best["territory_floor"])
                                      else float(best["territory_floor"])),
                  "w_hubness": float(best["w_hubness"]), "joint": bool(best["joint"]),
                  "min_sim": a.min_sim, "min_margin": a.min_margin,
                  "dev_match": DEV_MATCH, "dev_e2e": float(best["e2e"]),
                  "dev_n_assigned": int(best["n_assigned"]),
                  "rule": f"max dev end-to-end precision s.t. n_assigned >= {MIN_DEV_ASSIGNED}; "
                          f"hard constraints always on; ties -> more assignments"}
        FROZEN_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print(f"frozen: {frozen}")
        return
    cfg = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    preds = _predict_all(preps, cfg)
    preds.to_parquet(OUT_DIR / "constrained_preds.parquet", index=False)
    for title, ms in (("DEV (tuned, do not quote)", (DEV_MATCH,)), ("HELD-OUT", HELDOUT_MATCHES)):
        res = evaluate(preds, labels, ms, cfg["min_sim"], cfg["min_margin"])
        hl, ab = res["headline"], res["abstention"]
        print(f"--- {title}: {', '.join(ms)}")
        print(f"  judged {hl['n_judged']}  wrong_box {hl['n_wrong_box']} "
              f"({hl['carrier_selection_error_rate']})")
        print(f"  assigned {hl['n_assigned_all']}  end-to-end {hl['end_to_end_precision']}  "
              f"ident {hl['identification_precision']} (n={hl['n_assigned_good_box']})")
        print(f"  abstain_rate {ab['abstain_rate']}  lo_sim {ab['abstain_below_similarity']} "
              f"lo_margin {ab['abstain_below_margin']}")
        for key in ("by_dist", "by_gallery"):
            print(f"  {key}: {res[key]}")
    rows = []
    for s, m in itertools.product(CURVE_SIM, CURVE_MARGIN):
        res = evaluate(preds, labels, HELDOUT_MATCHES, s, m)
        hl = res["headline"]
        n_all = int(gate(preds, s, m)["pred_player"].notna().sum())
        rows.append({"min_sim": s, "min_margin": m, "n_labelled_assigned": hl["n_assigned_all"],
                     "heldout_e2e": hl["end_to_end_precision"],
                     "all_assigned_3matches": n_all})
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT_DIR / "constrained_curve.csv", index=False, encoding="utf-8")
    print(curve.to_string(index=False))


def _selftest() -> None:
    """Smallest runnable check of the elimination + joint-assignment seams."""
    ctx = pd.DataFrame({"qi": [0, 1], "chunk": ["c", "c"], "frame": [100, 130], "track_id": [7, 8],
                        "team": [0, 0], "half": ["h1", "h1"], "fps": [25.0, 25.0],
                        "t_s": [4.0, 5.2], "minute": [4, 5],
                        "pitch_x": [20.0, 22.0], "pitch_y": [30.0, 32.0]})
    anchors = pd.DataFrame({"chunk": ["c"], "frame": [100], "track_id": [9], "player": ["A"],
                            "half": ["h1"], "pitch_x": [80.0], "pitch_y": [10.0]})
    cand = pd.DataFrame({"qi": [0, 0, 1, 1], "player": ["A", "B", "A", "B"],
                         "sim": [0.95, 0.90, 0.80, 0.94], "n_crops": [50, 5, 50, 5]})
    excl, kine = anchor_flags(cand, ctx, anchors)
    assert list(excl) == [True, False, False, False], list(excl)   # A is elsewhere at frame 100
    assert list(kine) == [False, False, True, False], list(kine)   # 60 m in 1.2 s is impossible
    prof = {("A", "h1"): (80.0, 10.0, 5.0, 5.0, 99.0)}
    d = territory_distance(cand, ctx, prof)
    assert d[0] > 5.0 and d[1] == 0.0, d
    assert abs(hubness_bias(cand)[0] - 0.875) < 1e-6
    # joint: both queries locally prefer nothing in conflict -> Hungarian keeps distinct names
    cand2 = cand.assign(kept=True, score=[0.95, 0.90, 0.94, 0.93], terr_d=0.0)
    solo = solve(cand2, ctx, {"joint": False})
    assert set(solo["pred_player"]) == {"A"}, solo["pred_player"].tolist()
    joint = solve(cand2, ctx, {"joint": True}).sort_values("qi")
    assert joint["pred_player"].tolist() == ["A", "B"], joint["pred_player"].tolist()
    gid = possession_groups(ctx)
    assert gid[0] == gid[1]
    print("selftest ok")


if __name__ == "__main__":
    main()
