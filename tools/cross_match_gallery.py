"""Cross-match POOLED gallery for carrier attribution: attack the REACHABILITY term.

``results/CARRIER_CONSTRAINED_v2.md`` decomposes end-to-end carrier attribution as::

    0.846 (correct box) x 0.719 (true player is a reachable candidate) x 0.609 (matcher picks him)

and shows the 0.609 matcher term is already at the gallery leave-one-track-out ceiling (0.647).
The remaining headroom is REACHABILITY: the gallery is built PER MATCH from that match's named
tracks, so a player the identity chain never named in *this* match cannot be proposed at all.

Most Man Utd players appear in all 12 processed matches and 9 of those have identity artifacts
(``outputs/identity/<match>_named_tracks_both2_prtreid.parquet``), and every labelled match's
OPPONENT also has a reverse fixture in the corpus. This module builds ONE gallery per player POOLED
ACROSS THE CORPUS (keyed by club + player name, not by the per-match team index) and re-runs the
carrier attribution unchanged around it.

Three gallery modes are compared like-for-like:

* ``per_match`` -- the shipped baseline (this match's named tracks only). Reproduces v1/v2 exactly.
* ``pooled`` -- every crop of that player from every corpus match of his club.
* ``pooled_kit`` -- pooled, but restricted to matches where the club had the SAME home/away status,
  i.e. wore the same kit. The fallback for the appearance-drift risk (different kit, lighting, hair).

Leave-one-track-out is preserved in every mode: the query's own ``(match, chunk, track_id)`` is
dropped, so an assignment is never a restatement of the named-fragment overlap. Crops from OTHER
matches are by construction not the query's track.

Evaluation discipline is identical to v2: thresholds are chosen on ``manutd_liverpool``'s 30 labels
only and applied unchanged to the 60 held-out labels of ``manutd_tottenham`` + ``manutd_brighton``.
Metrics come from :func:`tools.score_carrier_labels.score_labels` verbatim.

Run::

    python -m tools.cross_match_gallery --stage embed     # GPU, one job: gallery crops, 6 matches
    python -m tools.cross_match_gallery --stage pool      # CPU: per-player gallery table
    python -m tools.cross_match_gallery --stage drift     # CPU: cross-match appearance drift
    python -m tools.cross_match_gallery --stage score     # CPU: reachability + end-to-end
    python -m tools.cross_match_gallery --stage paired    # CPU: per-moment held-out comparison
    python -m tools.cross_match_gallery --stage selftest
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from tools.carrier_attribution_probe import (
    OUT_DIR,
    PROBE_MATCHES,
    gallery_spec,
    named_tracks_path,
)
from tools.carrier_constrained import DEV_MATCH, HELDOUT_MATCHES, gate, prepare, run_config
from tools.score_carrier_labels import DEFAULT_LABELS, load_labels, norm_name, score_labels

#: Every registry match with a PRTreID named-tracks artifact (the pooling corpus).
CORPUS = (
    "brighton_manutd", "fulham_manutd", "liverpool_manutd", "manutd_brighton",
    "manutd_liverpool", "manutd_palace", "manutd_southampton", "manutd_tottenham",
    "tottenham_manutd",
)
GALLERY_MODES = ("per_match", "pooled", "pooled_kit")
REPORT_DIR = Path("results/cross_match_gallery")
#: Operating point frozen by v1/v2 -- reused verbatim so the numbers are comparable.
FROZEN_SIM, FROZEN_MARGIN = 0.88, 0.01
#: Dev-only re-selection grid for the pooled modes (the pooled similarity scale differs).
SIM_GRID = (0.84, 0.86, 0.88, 0.90, 0.92)
MARGIN_GRID = (0.0, 0.005, 0.01, 0.02, 0.04)
#: A re-selected dev point must keep at least this many dev assignments.
MIN_DEV_ASSIGNED = 8

#: v2's two reference configurations.
CFG_BASELINE = {"subwindow": False, "exclusion": False, "kinematic": False,
                "w_territory": 0.0, "territory_floor": None, "w_hubness": 0.0, "joint": False}
CFG_HARD = {"subwindow": True, "exclusion": True, "kinematic": True,
            "w_territory": 0.0, "territory_floor": None, "w_hubness": 0.0, "joint": False}


# === corpus-wide gallery =========================================================================
@lru_cache(maxsize=None)
def _match(match_id: str) -> registry.Match:
    """Registry lookup, cached (``registry.get`` re-parses the YAML on every call)."""
    return registry.get(match_id)


def club_of(match_id: str, team_index: int) -> str:
    """Club name for a per-match team index (``registry`` kit-anchor order)."""
    return _match(match_id).teams[int(team_index)]


def is_home(match_id: str, club: str) -> bool:
    """True when ``club`` played this fixture at home (kit proxy: home side wears the home kit)."""
    return _match(match_id).home_team == club


def embed_gallery(match_id: str) -> None:
    """Build and embed the per-match gallery only (skips the carrier queries). Single GPU job."""
    from tools.carrier_attribution_probe import build_embedder, embed_requests  # noqa: PLC0415

    match = registry.get(match_id)
    gal = gallery_spec(match)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gal.to_parquet(OUT_DIR / f"{match_id}_gallery.parquet", index=False)
    n_pl = int(gal["player"].nunique()) if len(gal) else 0
    print(f"{match_id}: {len(gal)} gallery crops over {n_pl} players")
    if gal.empty:
        return
    ge = embed_requests(match_id, gal, build_embedder())
    np.save(OUT_DIR / f"{match_id}_gallery_emb.npy", ge)
    ok = int((np.linalg.norm(ge, axis=1) > 0.5).sum())
    print(f"{match_id}: embedded {ok}/{len(gal)} crops -> {OUT_DIR}")


def load_corpus_gallery() -> tuple[pd.DataFrame, np.ndarray]:
    """Concatenate every corpus match's gallery, keyed by CLUB (not the per-match team index).

    Returns:
        ``(rows, emb)`` where ``rows`` carries ``match, chunk, track_id, club, player, home`` and
        ``emb`` is the aligned ``(n, D)`` L2-normalised embedding matrix. Unreadable crops (zero
        embeddings) are dropped here so no downstream stage has to re-check.
    """
    frames: list[pd.DataFrame] = []
    embs: list[np.ndarray] = []
    for mid in CORPUS:
        gpath, epath = OUT_DIR / f"{mid}_gallery.parquet", OUT_DIR / f"{mid}_gallery_emb.npy"
        if not (gpath.exists() and epath.exists()):
            print(f"WARN no cached gallery for {mid}; run --stage embed")
            continue
        gal, ge = pd.read_parquet(gpath).reset_index(drop=True), np.load(epath)
        ok = np.linalg.norm(ge, axis=1) > 0.5
        gal = gal[ok].reset_index(drop=True)
        gal["match"] = mid
        gal["club"] = [club_of(mid, t) for t in gal["team"]]
        gal["home"] = [is_home(mid, c) for c in gal["club"]]
        frames.append(gal)
        embs.append(ge[ok])
    return pd.concat(frames, ignore_index=True), np.concatenate(embs).astype(np.float32)


def gallery_mask(rows: pd.DataFrame, *, mode: str, match_id: str, club: str) -> np.ndarray:
    """Rows of the corpus gallery usable for a query from ``match_id`` on ``club``.

    Args:
        rows: :func:`load_corpus_gallery` table.
        mode: one of :data:`GALLERY_MODES`.
        match_id: the query's match.
        club: the query's tracked club.

    Returns:
        Boolean array over ``rows``.
    """
    same_club = (rows["club"] == club).to_numpy()
    if mode == "per_match":
        return same_club & (rows["match"] == match_id).to_numpy()
    if mode == "pooled":
        return same_club
    if mode == "pooled_kit":
        return same_club & (rows["home"] == is_home(match_id, club)).to_numpy()
    raise ValueError(f"unknown gallery mode {mode!r}")


def pooled_candidate_table(match_id: str, rows: pd.DataFrame, emb: np.ndarray,
                           mode: str) -> pd.DataFrame:
    """Long ``(query, candidate player)`` table scored against the chosen gallery mode.

    Identical to :func:`tools.carrier_constrained.candidate_table` except that the gallery is the
    corpus-wide pool masked by :func:`gallery_mask`, and leave-one-track-out drops the query's own
    ``(match, chunk, track_id)``.

    Args:
        match_id: registry match id with cached carrier-query embeddings.
        rows: corpus gallery table.
        emb: aligned corpus gallery embeddings.
        mode: one of :data:`GALLERY_MODES`.

    Returns:
        ``qi, player, sim, n_crops`` -- one row per surviving candidate player per query.
    """
    ev = pd.read_parquet(OUT_DIR / f"{match_id}_events.parquet")
    q = ev[ev["resolvable"]].reset_index(drop=True)
    qe = np.load(OUT_DIR / f"{match_id}_query_emb.npy")
    g_match = rows["match"].to_numpy()
    g_chunk, g_track = rows["chunk"].to_numpy(), rows["track_id"].to_numpy()
    players = rows["player"].to_numpy()
    club_cache = {t: club_of(match_id, t) for t in q["team"].unique()}
    mask_cache = {t: gallery_mask(rows, mode=mode, match_id=match_id, club=c)
                  for t, c in club_cache.items()}
    out: list[dict] = []
    for i, r in enumerate(q.itertuples(index=False)):
        if np.linalg.norm(qe[i]) <= 0.5:
            continue
        mask = mask_cache[r.team] & ~((g_match == match_id) & (g_chunk == r.chunk)
                                      & (g_track == int(r.track_id)))
        if not mask.any():
            continue
        sims = emb[mask] @ qe[i]
        pl = players[mask]
        best: dict[str, float] = {}
        n_crops: dict[str, int] = {}
        for p, s in zip(pl, sims):
            n_crops[p] = n_crops.get(p, 0) + 1
            if s > best.get(p, -2.0):
                best[p] = float(s)
        out.extend({"qi": i, "player": p, "sim": s, "n_crops": n_crops[p]}
                   for p, s in best.items())
    return pd.DataFrame(out)


def prepare_mode(match_id: str, rows: pd.DataFrame, emb: np.ndarray, mode: str) -> dict:
    """:func:`tools.carrier_constrained.prepare` with the candidate table swapped for ``mode``.

    The constraint columns (substitution window, exclusion, kinematic, territory, hubness) are
    recomputed for the new candidate set by the v2 code paths, unchanged.
    """
    from tools.carrier_constrained import (  # noqa: PLC0415
        anchor_flags, hubness_bias, subwindow_flags, territory_distance, territory_profiles,
    )

    base = prepare(match_id)                      # ctx + anchors (cached parquet)
    ctx, anchors = base["ctx"], base["anchors"]
    cand = (base["cand"][["qi", "player", "sim", "n_crops"]].copy() if mode == "per_match"
            else pooled_candidate_table(match_id, rows, emb, mode))
    cand["elim_subwindow"] = subwindow_flags(cand, ctx, match_id)
    excl, kine = anchor_flags(cand, ctx, anchors)
    cand["elim_exclusion"], cand["elim_kinematic"] = excl, kine
    cand["terr_d"] = territory_distance(cand, ctx, territory_profiles(anchors))
    cand["hub"] = hubness_bias(cand)
    return {"match": match_id, "cand": cand, "ctx": ctx, "anchors": anchors}


# === reachability ================================================================================
def reachable_players(prep: dict, cfg: dict) -> dict[int, set[str]]:
    """Normalised candidate names surviving ``cfg``'s HARD constraints, per query index."""
    cand = prep["cand"]
    keep = np.ones(len(cand), bool)
    for name in ("subwindow", "exclusion", "kinematic"):
        if cfg.get(name, False):
            keep &= ~cand[f"elim_{name}"].to_numpy()
    sub = cand[keep]
    out: dict[int, set[str]] = {}
    for qi, player in zip(sub["qi"], sub["player"]):
        out.setdefault(int(qi), set()).add(norm_name(player))
    return out


def reachability(preps: dict[str, dict], preds: pd.DataFrame, labels: pd.DataFrame,
                 cfg: dict, matches: tuple[str, ...], min_sim: float,
                 min_margin: float) -> dict:
    """P(true player is a reachable candidate | assigned, correct box) -- the 0.719 term.

    Measured exactly where the v2 report measured it: over labelled moments with a CORRECT box on
    which the model ASSIGNED a name at the operating point. Also reports the label-set-wide
    "gallery has any entry for the true carrier" rate over all correctly boxed judged moments.

    Args:
        preps: ``{match: prepare_mode output}``.
        preds: ungated predictions for the same configuration.
        labels: the hand-label table.
        cfg: configuration whose hard constraints define the candidate set.
        matches: match subset to score.
        min_sim: operating-point similarity floor.
        min_margin: operating-point margin floor.

    Returns:
        ``n_assigned``, ``n_reachable``, ``reachability``, ``n_good_box``, ``n_covered``,
        ``coverage`` and the per-moment miss list.
    """
    lab = labels[labels["match"].isin(matches)].copy()
    lab["ans"] = lab["player_name"].str.strip()
    lab = lab[~lab["ans"].isin(["", "UNKNOWN", "NOT_A_PASS", "WRONG_PLAYER_BOXED"])]
    gated = gate(preds[preds["match"].isin(matches)], min_sim, min_margin)
    j = lab.merge(gated, on=["match", "chunk", "frame"], how="left")
    reach = {m: reachable_players(p, cfg) for m, p in preps.items()}
    key = {m: p["ctx"].set_index(["chunk", "frame"])["qi"].to_dict() for m, p in preps.items()}
    n_a = n_r = n_cov = 0
    misses: list[dict] = []
    for r in j.itertuples(index=False):
        qi = key.get(r.match, {}).get((r.chunk, int(r.frame)))
        cands = reach.get(r.match, {}).get(int(qi), set()) if qi is not None else set()
        ok = norm_name(r.ans) in cands
        n_cov += int(ok)
        if isinstance(r.pred_player, str):
            n_a += 1
            n_r += int(ok)
            if not ok:
                misses.append({"match": r.match, "truth": r.ans, "pred": r.pred_player})
    return {"n_assigned": n_a, "n_reachable": n_r,
            "reachability": round(n_r / n_a, 3) if n_a else None,
            "n_good_box": int(len(j)), "n_covered": n_cov,
            "coverage": round(n_cov / len(j), 3) if len(j) else None, "misses": misses}


# === scoring =====================================================================================
def gallery_sets_for(rows: pd.DataFrame, mode: str,
                     matches: tuple[str, ...] = PROBE_MATCHES) -> dict[str, set[str]]:
    """Per-match set of normalised player names the gallery can produce under ``mode``."""
    out: dict[str, set[str]] = {}
    for mid in matches:
        names: set[str] = set()
        for club in _match(mid).teams:
            m = gallery_mask(rows, mode=mode, match_id=mid, club=club)
            names |= {norm_name(p) for p in rows.loc[m, "player"].unique()}
        out[mid] = names
    return out


def score_mode(preps: dict[str, dict], rows: pd.DataFrame, mode: str, cfg: dict,
               labels: pd.DataFrame, min_sim: float, min_margin: float) -> dict:
    """End-to-end + reachability for one (gallery mode, configuration, operating point)."""
    preds = pd.concat([run_config(p, cfg) for p in preps.values()], ignore_index=True)
    gsets = gallery_sets_for(rows, mode)
    out: dict = {"preds": preds}
    for tag, ms in (("dev", (DEV_MATCH,)), ("held", HELDOUT_MATCHES), ("all3", PROBE_MATCHES)):
        res = score_labels(labels[labels["match"].isin(ms)],
                           gate(preds[preds["match"].isin(ms)], min_sim, min_margin), gsets)
        rc = reachability({m: preps[m] for m in ms}, preds, labels, cfg, ms, min_sim, min_margin)
        out[tag] = {**res["headline"], **{f"reach_{k}": v for k, v in rc.items()
                                          if k != "misses"},
                    "misses": rc["misses"]}
    out["n_assigned_total"] = int(gate(preds, min_sim, min_margin)["pred_player"].notna().sum())
    return out


def _dev_reselect(preps: dict[str, dict], rows: pd.DataFrame, mode: str, cfg: dict,
                  labels: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Re-select the operating point on the DEV match only (pooled sims live on a new scale)."""
    preds = pd.concat([run_config(p, cfg) for p in preps.values()], ignore_index=True)
    gsets = gallery_sets_for(rows, mode)
    lab = labels[labels["match"] == DEV_MATCH]
    grid = []
    for s in SIM_GRID:
        for m in MARGIN_GRID:
            h = score_labels(lab, gate(preds[preds["match"] == DEV_MATCH], s, m),
                             gsets)["headline"]
            grid.append({"min_sim": s, "min_margin": m, "n": h["n_assigned_all"],
                         "dev_e2e": h["end_to_end_precision"]})
    g = pd.DataFrame(grid)
    ok = g[(g["n"] >= MIN_DEV_ASSIGNED) & g["dev_e2e"].notna()]
    if ok.empty:
        return FROZEN_SIM, FROZEN_MARGIN, g
    best = ok.sort_values(["dev_e2e", "n"], ascending=[False, False]).iloc[0]
    return float(best["min_sim"]), float(best["min_margin"]), g


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n``."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def stage_paired() -> pd.DataFrame:
    """Held-out per-moment comparison of the gallery modes (the only test with any power).

    The three modes assign different numbers of moments, so the raw precision columns are not on the
    same denominator. This restricts to the held-out labelled moments where BOTH the per-match
    baseline and the pooled variant assigned a name, and reports the discordant pairs (the McNemar
    table) plus Wilson intervals on the marginals.
    """
    rows, emb = load_corpus_gallery()
    labels = load_labels(DEFAULT_LABELS)
    per_moment: dict[str, pd.DataFrame] = {}
    for mode in GALLERY_MODES:
        preps = {m: prepare_mode(m, rows, emb, mode) for m in PROBE_MATCHES}
        preds = pd.concat([run_config(p, CFG_HARD) for p in preps.values()], ignore_index=True)
        res = score_labels(labels[labels["match"].isin(HELDOUT_MATCHES)],
                           gate(preds[preds["match"].isin(HELDOUT_MATCHES)], FROZEN_SIM,
                                FROZEN_MARGIN), gallery_sets_for(rows, mode))
        r = res["rows"][["id", "assigned", "correct"]].set_index("id")
        per_moment[mode] = r
    out = []
    base = per_moment["per_match"]
    for mode in ("pooled", "pooled_kit"):
        alt = per_moment[mode]
        both = base.join(alt, lsuffix="_base", rsuffix="_alt", how="inner")
        both = both[both["assigned_base"] & both["assigned_alt"]]
        b, a = both["correct_base"].to_numpy(), both["correct_alt"].to_numpy()
        n01, n10 = int((~b & a).sum()), int((b & ~a).sum())
        out.append({"mode": mode, "n_both_assigned": int(len(both)),
                    "base_correct": int(b.sum()), "alt_correct": int(a.sum()),
                    "alt_right_base_wrong": n01, "base_right_alt_wrong": n10,
                    "discordant": n01 + n10})
    for mode, r in per_moment.items():
        a = r[r["assigned"]]
        k, n = int(a["correct"].sum()), int(len(a))
        out.append({"mode": mode, "n_both_assigned": None, "base_correct": None,
                    "alt_correct": None, "heldout_e2e": round(k / n, 3) if n else None,
                    "heldout_n": n, "wilson95": str(wilson(k, n))})
    tab = pd.DataFrame(out)
    tab.to_csv(REPORT_DIR / "paired_heldout.csv", index=False, encoding="utf-8")
    print(tab.to_string(index=False))
    return tab


# === reporting stages ============================================================================
def stage_pool() -> pd.DataFrame:
    """Per-player gallery table: crops available per match vs pooled, and the zero-sample players."""
    rows, _ = load_corpus_gallery()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per = rows.groupby(["club", "player", "match"]).size().rename("crops").reset_index()
    wide = per.pivot_table(index=["club", "player"], columns="match", values="crops",
                           fill_value=0).astype(int)
    wide["pooled"] = wide.sum(axis=1)
    wide.to_csv(REPORT_DIR / "per_player_gallery.csv", encoding="utf-8")
    print(f"corpus gallery: {len(rows)} crops, {rows['player'].nunique()} players, "
          f"{rows['match'].nunique()} matches")
    lines = []
    for mid in PROBE_MATCHES:
        for club in _match(mid).teams:
            here = set(rows.loc[(rows["match"] == mid) & (rows["club"] == club), "player"])
            pooled = set(rows.loc[rows["club"] == club, "player"])
            kit = set(rows.loc[gallery_mask(rows, mode="pooled_kit", match_id=mid, club=club),
                               "player"])
            lines.append({"match": mid, "club": club, "per_match_players": len(here),
                          "pooled_players": len(pooled), "pooled_kit_players": len(kit),
                          "new_players_pooled": len(pooled - here),
                          "new_players_kit": len(kit - here),
                          "per_match_crops": int(((rows["match"] == mid)
                                                  & (rows["club"] == club)).sum()),
                          "pooled_crops": int((rows["club"] == club).sum())})
    tab = pd.DataFrame(lines)
    tab.to_csv(REPORT_DIR / "pool_summary.csv", index=False, encoding="utf-8")
    print(tab.to_string(index=False))
    return tab


def stage_drift() -> pd.DataFrame:
    """Same-player appearance similarity WITHIN a match vs ACROSS matches (the kit-drift check)."""
    rows, emb = load_corpus_gallery()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for (club, player), g in rows.groupby(["club", "player"]):
        if g["match"].nunique() < 2:
            continue
        idx = g.index.to_numpy()
        sims = emb[idx] @ emb[idx].T
        same = g["match"].to_numpy()[:, None] == g["match"].to_numpy()[None, :]
        kit = g["home"].to_numpy()[:, None] == g["home"].to_numpy()[None, :]
        tri = ~np.eye(len(idx), dtype=bool)
        out.append({
            "club": club, "player": player, "n_matches": int(g["match"].nunique()),
            "n_crops": len(idx),
            "within_match": round(float(sims[same & tri].mean()), 4) if (same & tri).any() else None,
            "cross_match": round(float(sims[~same].mean()), 4) if (~same).any() else None,
            "cross_same_kit": round(float(sims[~same & kit].mean()), 4)
            if (~same & kit).any() else None,
            "cross_diff_kit": round(float(sims[~same & ~kit].mean()), 4)
            if (~same & ~kit).any() else None})
    df = pd.DataFrame(out)
    df.to_csv(REPORT_DIR / "appearance_drift.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False))
    for col in ("within_match", "cross_match", "cross_same_kit", "cross_diff_kit"):
        v = df[col].dropna()
        print(f"  mean {col:<16}: {v.mean():.4f}  (n players={len(v)})")
    return df


def stage_score() -> pd.DataFrame:
    """Reachability + held-out end-to-end for every (gallery mode, configuration)."""
    rows, emb = load_corpus_gallery()
    labels = load_labels(DEFAULT_LABELS)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table: list[dict] = []
    misses: list[dict] = []
    for mode in GALLERY_MODES:
        preps = {m: prepare_mode(m, rows, emb, mode) for m in PROBE_MATCHES}
        n_cand = int(sum(len(p["cand"]) for p in preps.values()))
        for cname, cfg in (("baseline", CFG_BASELINE), ("all_hard", CFG_HARD)):
            points = [("frozen_v1", FROZEN_SIM, FROZEN_MARGIN)]
            if mode != "per_match":
                s, m, grid = _dev_reselect(preps, rows, mode, cfg, labels)
                grid.to_csv(REPORT_DIR / f"dev_grid_{mode}_{cname}.csv", index=False,
                            encoding="utf-8")
                if (s, m) != (FROZEN_SIM, FROZEN_MARGIN):
                    points.append(("dev_reselected", s, m))
            for pname, s, m in points:
                res = score_mode(preps, rows, mode, cfg, labels, s, m)
                for tag in ("dev", "held", "all3"):
                    r = res[tag]
                    table.append({
                        "mode": mode, "config": cname, "point": pname, "min_sim": s,
                        "min_margin": m, "split": tag, "n_candidates": n_cand,
                        "n_judged": r["n_judged"],
                        "carrier_sel_err": r["carrier_selection_error_rate"],
                        "n_assigned": r["n_assigned_all"], "e2e": r["end_to_end_precision"],
                        "ident": r["identification_precision"],
                        "reachability": r["reach_reachability"],
                        "reach_n": f"{r['reach_n_reachable']}/{r['reach_n_assigned']}",
                        "gallery_coverage": r["reach_coverage"],
                        "cov_n": f"{r['reach_n_covered']}/{r['reach_n_good_box']}",
                        "assigned_all_passes": res["n_assigned_total"]})
                    if tag == "held":
                        misses += [{**x, "mode": mode, "config": cname, "point": pname}
                                   for x in r["misses"]]
        print(f"[{mode}] done ({n_cand} candidate rows)")
    df = pd.DataFrame(table)
    df.to_csv(REPORT_DIR / "mode_scores.csv", index=False, encoding="utf-8")
    pd.DataFrame(misses).to_csv(REPORT_DIR / "heldout_unreachable.csv", index=False,
                                encoding="utf-8")
    print(df[df["split"] != "all3"].to_string(index=False))
    return df


def _selftest() -> None:
    """Smallest runnable check of the pooling seams (pure, no artifacts touched)."""
    rows = pd.DataFrame({
        "match": ["manutd_liverpool", "manutd_liverpool", "liverpool_manutd", "brighton_manutd"],
        "chunk": ["h1_chunk_000"] * 4, "track_id": [1, 2, 3, 4],
        "club": ["Man Utd", "Liverpool", "Man Utd", "Man Utd"],
        "player": ["Bruno", "Salah", "Bruno", "Bruno"],
        "home": [True, False, False, False]})
    m_per = gallery_mask(rows, mode="per_match", match_id="manutd_liverpool", club="Man Utd")
    assert list(m_per) == [True, False, False, False], list(m_per)
    m_pool = gallery_mask(rows, mode="pooled", match_id="manutd_liverpool", club="Man Utd")
    assert list(m_pool) == [True, False, True, True], list(m_pool)
    # Man Utd is HOME in manutd_liverpool -> only home-kit matches pool in
    m_kit = gallery_mask(rows, mode="pooled_kit", match_id="manutd_liverpool", club="Man Utd")
    assert list(m_kit) == [True, False, False, False], list(m_kit)
    assert club_of("manutd_liverpool", 0) == "Man Utd"
    assert is_home("manutd_liverpool", "Man Utd") and not is_home("liverpool_manutd", "Man Utd")
    prep = {"cand": pd.DataFrame({"qi": [0, 0, 1], "player": ["Bruno", "Salah", "Bruno"],
                                  "elim_subwindow": [False, True, False],
                                  "elim_exclusion": [False] * 3,
                                  "elim_kinematic": [False] * 3})}
    assert reachable_players(prep, CFG_BASELINE)[0] == {"bruno", "salah"}
    assert reachable_players(prep, CFG_HARD)[0] == {"bruno"}
    print("selftest ok")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["embed", "pool", "drift", "score", "paired", "selftest"])
    ap.add_argument("--match", default=None, help="single match (embed stage)")
    a = ap.parse_args()
    if a.stage == "selftest":
        _selftest()
    elif a.stage == "embed":
        todo = [a.match] if a.match else [
            m for m in CORPUS if not (OUT_DIR / f"{m}_gallery_emb.npy").exists()]
        print(f"embedding galleries for: {todo}")
        for mid in todo:
            if not named_tracks_path(mid).exists():
                print(f"SKIP {mid}: no named-tracks artifact")
                continue
            embed_gallery(mid)
    elif a.stage == "pool":
        stage_pool()
    elif a.stage == "drift":
        stage_drift()
    elif a.stage == "paired":
        stage_paired()
    elif a.stage == "score":
        df = stage_score()
        (REPORT_DIR / "mode_scores.json").write_text(
            json.dumps(df.to_dict("records"), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
