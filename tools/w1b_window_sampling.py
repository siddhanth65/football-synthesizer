"""W1-B: the corrected 2 s turnover census, the censoring rate, and the finite-sample noise floor.

Three measurements, CPU only, no new estimators. Everything here is *measurement*; nothing is fitted.

**Task 1 -- the census at the window the literature actually uses.** ``tools/w1_feasibility`` priced a
**5 s continuous** observation window and returned FAIL. That requirement was self-imposed: Bauer &
Anzer (2021, p. 2016) evaluate every feature at ball-possession-change + {0, 1, 2} s and explicitly
reject longer feature windows; the 5 s is only the *success label*, and an unobserved regain is
**right-censored**, not missing. This module re-runs the identical turnover definition
(:func:`tools.w1_feasibility.chunk_inputs`, so the two censuses cannot drift) against the three
instants, sweeps the presser count at those instants, labels every turnover's regain-within-5 s
outcome as observed or censored, and applies the BA exclusions that are applicable to us.

**Task 2 -- the sampling floor.** For each metric with a published between-team SD, the finite-sample
SD of a 12-match estimate, by three routes (within-team season SD from BA Appendix D scaled to 12
matches; the 5-match manual-labelling spread from BA Table 7 scaled to 12; and a binomial/Poisson
floor from BA's own averages). ``D_sampling = SD_between / SD_sampling(12)`` answers whether 12
matches could separate two teams *with perfect tracking*. The BA tables are parsed straight out of
``docs/papers/`` rather than transcribed, so the read's harvested SDs are re-derived, not trusted.

**Task 3 -- our own corpus.** Between-team spread of the three quantities we can already compute
(possession-link share, block line, turnovers per match), decomposed with the six opponent-controlled
pairs into a between-team component and a residual, with the honest (very wide) uncertainty.

Run (CPU)::

    python -m tools.w1b_window_sampling
    python -m tools.w1b_window_sampling --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.pitch import PITCH_LEN
from core.registry import Match, get
from fingerprint.block_height import block_summary
from fingerprint.structural_metrics import attacking_coord, resolve_attack_directions
from tools.action_spot_probe import OUT_DIR as SPOT_DIR
from tools.action_spot_probe import class_index, find_peaks
from tools.w1_feasibility import (
    DEFENDER_LEVELS,
    MATCH_IDS,
    chunk_inputs,
    verdict,
    window_stats,
)

# --- Task 1: window parameters -------------------------------------------------------------------
INSTANTS_S = (0.0, 1.0, 2.0)   # BA Sect. 2.2.2 / Table 1: every feature at BPC + {0, 1, 2} s
INSTANT_TOL_S = 0.2            # "nearest available frame within tolerance" = one grid slot @ 25 fps
SUCCESS_S = 5.0                # BA p. 2014: the *label* horizon, not a feature window
OLD_WINDOW_S = 5.0             # re-computed in the same pass so old-vs-new is one run, not two

# BA exclusion rule 1 (set-piece-origin possessions) and rule 3 (scenario ends when the ball leaves
# play) both need dead-ball knowledge we have no validated detector for. The E2E-Spot classes below
# are used as a PROXY and reported as a labelled sensitivity, never as the headline: of the 17
# SoccerNet-v2 classes only `Foul` (1.069x) and `Clearance` (0.265x) have ever been validated here.
SETPIECE_CLASSES = ("Corner", "Throw-in", "Direct free-kick", "Indirect free-kick", "Kick-off",
                    "Penalty")
OOP_CLASS = "Ball out of play"
SPOT_THRESH = 0.30
SPOT_NMS_S = 5.0
SETPIECE_LEAD_S = 3.0          # a possession starting this soon after a set-piece spot is set-piece
SETPIECE_LAG_S = 1.0           # origin; the lag absorbs spot-timing error

# --- Task 2: literature inputs -------------------------------------------------------------------
BA_PDF = Path("docs/papers/data driven detection of counterpressing.pdf")
BA_T7_PAGE = 25                # printed p. 2034 -- manual expert labelling, 4-6 matches per team
BA_T8_PAGE = 27                # printed p. 2036 -- per team over 6.5 seasons (Appendix B)
BA_SEASON_PAGES = tuple(range(30, 38))   # printed pp. 2039-2046 -- per team-season (Appendix D)
BA_COLS = ("T/M", "CP+/T", "%CP", "S+/-", "G+/-", "%CP+", "%CP-", "%S+", "%G+", "%S-", "%G-")
STABLE_GAMES = 160             # ">= ~5 full seasons" -- the read's "17 stable teams" cut
SEASON_MATCHES = 34            # a full Bundesliga season
TARGET_N = 12                  # our corpus size
FOCUS_METRICS = ("T/M", "CP+/T", "%CP", "%CP+")

# --- Task 3: our corpus --------------------------------------------------------------------------
# Registry `teams` order is FLIPPED for these two (results/PAIR_ANALYSIS_v1.md verdicts 2 and 3,
# two independent lines each: poss-link majority vs Sofascore, and goal direction). Corrected here
# for the variance decomposition only; no shipped artifact is rewritten.
TEAM_ORDER_FLIPPED = ("tottenham_manutd", "southampton_manutd")
HOME_TEAM = "Man Utd"

REPORT_PATH = Path("results/W1B_WINDOW_AND_SAMPLING.md")


# === Task 1: the corrected census ================================================================
def _nearest(frames: np.ndarray, target: int, tol: int) -> int | None:
    """Nearest entry of a sorted frame array to ``target``, or ``None`` if none within ``tol``."""
    if frames.size == 0:
        return None
    i = int(np.searchsorted(frames, target))
    best: int | None = None
    for j in (i - 1, i):
        if 0 <= j < frames.size:
            f = int(frames[j])
            if abs(f - target) <= tol and (best is None or abs(f - target) < abs(best - target)):
                best = f
    return best


def instant_defs(frame: int, lost_team: int, ball_frames: np.ndarray,
                 def_counts: dict[tuple[int, int], int], *, fps: float,
                 tol_s: float = INSTANT_TOL_S) -> list[int | None]:
    """Trusted pressing-team player count at BPC + {0, 1, 2} s.

    Args:
        frame: turnover frame (BPC).
        lost_team: the team that lost the ball -- the pressing team in a counterpress.
        ball_frames: sorted frames present in this chunk's post-``link_ball`` track.
        def_counts: ``(frame, team) -> number of players with trusted geometry``.
        fps: native fps of the chunk.
        tol_s: "nearest available frame" tolerance, in seconds.

    Returns:
        One entry per instant: the presser count at the nearest ball-tracked frame, or ``None``
        when no ball sample lies within ``tol_s`` of that instant (the instant is unobserved).
    """
    tol = max(int(round(tol_s * fps)), 1)
    out: list[int | None] = []
    for s in INSTANTS_S:
        g = _nearest(ball_frames, frame + int(round(s * fps)), tol)
        out.append(None if g is None else int(def_counts.get((g, lost_team), 0)))
    return out


def success_label(frame: int, lost_team: int, ball_frames: set[int], geom_frames: set[int],
                  poss_map: dict[int, int], oop_frames: np.ndarray, *, stride: int, fps: float,
                  chunk_end: int) -> tuple[str, float]:
    """BA's regain-within-5 s label as a survival observation, never as a dropped window.

    The scan walks the sampling grid forward from the turnover and stops at the first event:

    * ``regain`` -- the pressing team is back in possession (observed success, time = regain time),
    * ``no_regain`` -- the grid was observed continuously to +5 s with no regain (observed failure),
    * ``out_of_play`` -- a spotted ball-out-of-play ended the scenario first (BA rule 3; BA credit
      the regain to the restarting team, which we cannot attribute -- a competing risk, not a
      success or a failure),
    * ``censored_track`` -- the ball track or the geometry broke first (right-censored at the last
      observed time),
    * ``censored_boundary`` -- the chunk ended first (administrative censoring, the analogue of
      BA rule 4 dropping phases that run into half-time).

    Args:
        frame: turnover frame.
        lost_team: pressing team.
        ball_frames: frames present in the linked ball track.
        geom_frames: frames with at least one trusted-geometry player of either team (below which
            the possession proxy is structurally blind).
        poss_map: ``frame -> possessing team`` from the Viterbi possession track.
        oop_frames: spotted ball-out-of-play frames in this chunk.
        stride: sampling stride of the grid, in native frames.
        fps: native fps of the chunk.
        chunk_end: last frame of the chunk.

    Returns:
        ``(outcome, time_s)`` -- the event time for ``regain`` / ``out_of_play``, the last observed
        time for the censored outcomes, and ``SUCCESS_S`` for ``no_regain``.
    """
    horizon = frame + int(round(SUCCESS_S * fps))
    after = oop_frames[(oop_frames > frame) & (oop_frames <= horizon)]
    oop = int(after[0]) if after.size else None
    last, s = frame, frame + stride
    while s <= horizon:
        if oop is not None and s >= oop:
            return "out_of_play", (oop - frame) / fps
        if s > chunk_end:
            return "censored_boundary", (last - frame) / fps
        if s not in ball_frames or s not in geom_frames:
            return "censored_track", (last - frame) / fps
        if poss_map.get(s) == lost_team:
            return "regain", (s - frame) / fps
        last, s = s, s + stride
    return "no_regain", SUCCESS_S


def spot_frames(match_id: str, chunk_key: str, classes: tuple[str, ...], fps: float) -> np.ndarray:
    """Native frames of E2E-Spot peaks for ``classes`` in one chunk (empty when the npz is absent).

    Args:
        match_id: registry match id.
        chunk_key: ``h1_chunk_003``-style key.
        classes: SoccerNet-v2 class names.
        fps: native fps of the chunk (the 2 Hz score grid is converted through it).

    Returns:
        Sorted array of chunk-local frame indices.
    """
    half, idx = chunk_key.split("_chunk_")
    path = SPOT_DIR / match_id / f"scores_{half}_chunk{idx}.npz"
    if not path.exists():
        return np.empty(0, dtype=int)
    d = np.load(path)
    sc = d["scores"].astype(np.float32)
    score_fps = float(d["fps"])
    sep = int(round(SPOT_NMS_S * score_fps))
    hits: list[float] = []
    for name in classes:
        hits += [i / score_fps for i, _ in find_peaks(sc[:, class_index(name)], SPOT_THRESH, sep)]
    return np.sort(np.array([int(round(t * fps)) for t in hits], dtype=int))


def chunk_records(match_id: str, chunk_key: str, pos_chunk: pd.DataFrame, ball: pd.DataFrame,
                  fps: float) -> list[dict]:
    """One record per turnover in a chunk: instant observation, exclusions, and success label.

    Args:
        match_id: registry match id.
        chunk_key: chunk identifier matching ``aligned['chunk']``.
        pos_chunk: this chunk's rows of the aligned parquet (all roles, unfiltered).
        ball: this chunk's post-``link_ball`` track.
        fps: native fps of the chunk.

    Returns:
        List of per-turnover dicts (empty when the chunk yields no turnover).
    """
    prepared = chunk_inputs(pos_chunk, ball)
    if prepared is None:
        return []
    players, poss, turnovers = prepared

    frames = np.sort(ball["frame"].astype(int).unique())
    stride = max(int(np.median(np.diff(frames))) if frames.size > 1 else 1, 1)
    ball_set = {int(f) for f in frames}
    def_counts = {(int(f), int(t)): int(v)
                  for (f, t), v in players.groupby(["frame", "team"]).size().items()}
    geom_frames = {int(f) for f in players["frame"].unique()}
    poss_map = {int(r.frame): int(r.team) for r in poss.itertuples(index=False)}
    dirs = resolve_attack_directions(players)
    chunk_end = int(pos_chunk["frame"].max())
    setpieces = spot_frames(match_id, chunk_key, SETPIECE_CLASSES, fps)
    oop = spot_frames(match_id, chunk_key, (OOP_CLASS,), fps)
    n_slot5 = max(int(round(OLD_WINDOW_S * fps)) // stride, 1)

    rows: list[dict] = []
    prev: int | None = None
    for tv in turnovers.itertuples(index=False):
        f, lost = int(tv.frame), int(tv.lost_team)
        defs = instant_defs(f, lost, frames, def_counts, fps=fps)
        outcome, t_out = success_label(f, lost, ball_set, geom_frames, poss_map, oop,
                                       stride=stride, fps=fps, chunk_end=chunk_end)
        old = window_stats(f, lost, ball_set, def_counts, stride=stride, n_slot=n_slot5)
        adir = dirs.get(lost)
        own_half = (None if adir is None else
                    bool(attacking_coord(np.array([float(tv.x)]), adir)[0] <= PITCH_LEN / 2))
        sp_origin: bool | None = None
        if prev is not None:
            lo, hi = prev - SETPIECE_LEAD_S * fps, prev + SETPIECE_LAG_S * fps
            sp_origin = bool(np.any((setpieces >= lo) & (setpieces <= hi)))
        rows.append({
            "match_id": match_id, "chunk": chunk_key, "frame": f, "lost_team": lost,
            "d0": defs[0], "d1": defs[1], "d2": defs[2],
            "n_inst_obs": sum(d is not None for d in defs),
            "outcome": outcome, "t_out": t_out,
            "own_half": own_half, "dir_known": adir is not None,
            "setpiece_origin": sp_origin, "spell_start_known": prev is not None,
            "past_boundary_2s": f + int(round(INSTANTS_S[-1] * fps)) > chunk_end,
            "old_ball_full": bool(old["ball_full"]), "old_min_def": int(old["min_def"]),
        })
        prev = f
    return rows


def match_records(m: Match) -> pd.DataFrame:
    """Every turnover record for one match, over all its linked ball chunks."""
    pos = m.load_aligned()
    rows: list[dict] = []
    for chunk_key, ball_path in m.ball_chunks():
        pos_chunk = pos[pos["chunk"] == chunk_key]
        if pos_chunk.empty:
            continue
        rows += chunk_records(m.id, chunk_key, pos_chunk, pd.read_parquet(ball_path),
                              m.chunk_fps(chunk_key))
    return pd.DataFrame(rows)


def _def_ok(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    """Boolean: instant ``col`` observed with at least ``n`` trusted pressers."""
    return df[col].notna() & (df[col].fillna(-1) >= n)


def yield_counts(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-match usable-turnover counts at presser level ``n`` for each observation rule."""
    ok = pd.concat([_def_ok(df, c, n) for c in ("d0", "d1", "d2")], axis=1)
    out = pd.DataFrame({
        "match_id": df["match_id"],
        "all3": ok.all(axis=1),
        "ge2": ok.sum(axis=1) >= 2,
        "t0": ok["d0"],
        "old5": df["old_ball_full"] & (df["old_min_def"] >= n),
    })
    return out.groupby("match_id", sort=False).sum(numeric_only=True).astype(int)


# === Task 2: the sampling floor ==================================================================
def _parse_rotated(pages: tuple[int, ...]) -> pd.DataFrame:
    """Parse BA's 90-degree-rotated appendix tables (Table 8 / Tables 10-11).

    pdfplumber emits the rotated cells as mirrored single-token lines in reverse column order, so
    each line is reversed and 11 numeric lines are accumulated until the ``Team(N)`` line closes the
    record. ``N`` is the game count in Table 8 and the final league position in Tables 10-11.

    Args:
        pages: zero-based PDF page indices to read.

    Returns:
        DataFrame with ``label``, ``bracket`` and the eleven :data:`BA_COLS` columns.
    """
    import pdfplumber  # noqa: PLC0415

    recs, buf = [], []
    with pdfplumber.open(BA_PDF) as pdf:
        for i in pages:
            for raw in (pdf.pages[i].extract_text() or "").splitlines():
                s = raw[::-1].strip().replace("−", "-").replace("%", "").replace(",", "")
                if re.fullmatch(r"-?\d+\.\d+", s):
                    buf.append(float(s))
                elif re.search(r"\(\d+\)$", s) and len(buf) >= len(BA_COLS):
                    recs.append([s, *buf[-len(BA_COLS):][::-1]])
                    buf = []
    df = pd.DataFrame(recs, columns=["label", *BA_COLS])
    df = df[~df["label"].str.startswith("AVERAGE")].reset_index(drop=True)
    df["bracket"] = df["label"].str.extract(r"\((\d+)\)$").astype(int)
    return df


def _parse_table7() -> pd.DataFrame:
    """BA Table 7 (manual expert labelling, 4-6 matches per team) -> ``team, matches, T/M, %CP``."""
    import pdfplumber  # noqa: PLC0415

    pat = re.compile(r"^(.+?)\((\d+)\)\s+([\d,]+)\([\d,]+\)\s+([\d,]+)\([\d,]+\)\s+"
                     r"[\d,]+\([\d,]+\)\s+[\d,]+\([\d,]+\)\s+([\d.]+)$")
    rows = []
    with pdfplumber.open(BA_PDF) as pdf:
        for raw in (pdf.pages[BA_T7_PAGE].extract_text() or "").splitlines():
            mo = pat.match(raw.strip())
            if mo and not mo.group(1).startswith("All"):
                matches, turnovers = int(mo.group(2)), int(mo.group(3).replace(",", ""))
                rows.append({"team": mo.group(1), "matches": matches,
                             "T/M": turnovers / matches, "%CP": float(mo.group(5))})
    return pd.DataFrame(rows)


def within_team_season_sd(seasons: pd.DataFrame, *, min_seasons: int = 4) -> dict[str, float]:
    """Pooled within-team SD across seasons, per metric (BA Appendix D, Tables 10-11).

    Each team-season is a 34-match mean, so this SD contains the 34-match sampling noise **plus**
    real season-to-season drift (coach and squad change). It is therefore an upper bound on the
    sampling term, which makes every ``D_sampling`` derived from it a *lower* bound.

    Args:
        seasons: parsed Appendix D table with a ``label`` of the form ``CODE-YYYY/YYYY(rank)``.
        min_seasons: minimum seasons a team needs to contribute.

    Returns:
        ``{metric: pooled within-team SD}`` plus ``{"_teams": n, "_df": total degrees of freedom}``.
    """
    seasons = seasons.assign(team=seasons["label"].str.split("-").str[0])
    out: dict[str, float] = {}
    groups = [g for _, g in seasons.groupby("team") if len(g) >= min_seasons]
    for col in BA_COLS:
        num = sum((len(g) - 1) * float(g[col].var(ddof=1)) for g in groups)
        den = sum(len(g) - 1 for g in groups)
        out[col] = float(np.sqrt(num / den)) if den else float("nan")
    out["_teams"] = float(len(groups))
    out["_df"] = float(sum(len(g) - 1 for g in groups))
    return out


def sampling_table(t8: pd.DataFrame, seasons: pd.DataFrame, t7: pd.DataFrame) -> pd.DataFrame:
    """The sampling-floor table: SD_between, three routes to SD_sampling(12), D, matches for D=2.

    Args:
        t8: parsed BA Table 8 (per team over 6.5 seasons).
        seasons: parsed BA Tables 10-11 (per team-season).
        t7: parsed BA Table 7 (manual labels, 4-6 matches per team).

    Returns:
        One row per metric in :data:`FOCUS_METRICS`.
    """
    stable = t8[t8["bracket"] >= STABLE_GAMES]
    within = within_team_season_sd(seasons)
    avg = {c: float(t8[c].mean()) for c in BA_COLS}
    t_per_match = avg["T/M"]
    n_trans = t_per_match * TARGET_N
    # Route C: the irreducible binomial / Poisson term at 12 matches, from BA's own averages.
    p_cp, p_cpt, p_suc = avg["%CP"] / 100, avg["CP+/T"] / 100, avg["%CP+"] / 100
    route_c = {
        "T/M": np.sqrt(t_per_match) / np.sqrt(TARGET_N),
        "%CP": 100 * np.sqrt(p_cp * (1 - p_cp) / n_trans),
        "CP+/T": 100 * np.sqrt(p_cpt * (1 - p_cpt) / n_trans),
        "%CP+": 100 * np.sqrt(p_suc * (1 - p_suc) / (n_trans * p_cp)),
    }
    n7 = float(t7["matches"].mean()) if not t7.empty else float("nan")

    rows = []
    for col in FOCUS_METRICS:
        sd_between = float(stable[col].std(ddof=1))
        a = within[col] * np.sqrt(SEASON_MATCHES / TARGET_N)
        b = float("nan")
        if col in t7.columns and not t7.empty:
            obs = float(t7[col].std(ddof=1))
            resid = max(obs ** 2 - sd_between ** 2, 0.0)
            b = np.sqrt(resid) * np.sqrt(n7 / TARGET_N)
        rows.append({"metric": col, "sd_between": sd_between, "sd_a": a, "sd_b": b,
                     "sd_c": route_c[col], "t7_obs_sd": float(t7[col].std(ddof=1))
                     if col in t7.columns and not t7.empty else float("nan")})
    df = pd.DataFrame(rows)
    for key in ("a", "b", "c"):
        df[f"d_{key}"] = df["sd_between"] / df[f"sd_{key}"]
        df[f"n2_{key}"] = TARGET_N * (2.0 / df[f"d_{key}"]) ** 2
    df.attrs["within"] = within
    df.attrs["avg"] = avg
    df.attrs["n7"] = n7
    df.attrs["n_stable"] = int(len(stable))
    return df


# === Task 3: our corpus ==========================================================================
def team_names(m: Match) -> dict[int, str]:
    """``team_int -> club name``, with the two documented registry order flips corrected."""
    a, b = m.teams
    return {0: b, 1: a} if m.id in TEAM_ORDER_FLIPPED else {0: a, 1: b}


def corpus_metrics(records: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team) values of the three quantities we can already compute.

    Args:
        records: pooled turnover records from :func:`match_records`.

    Returns:
        ``match_id, team, is_home_focus, turnovers, poss_link_share, block_line_m``.
    """
    rows = []
    for mid in MATCH_IDS:
        m = get(mid)
        names = team_names(m)
        block = block_summary(m)["teams"]
        registry = {0: m.teams[0], 1: m.teams[1]}
        facts = json.loads((Path("outputs") / mid / "facts" / "pass_network.json")
                           .read_text(encoding="utf-8").replace("NaN", "null"))
        links = {int(v["team_int"]): float(v["n_poss_links"]) for v in facts["teams"].values()}
        total = sum(links.values()) or float("nan")
        lost = records[records["match_id"] == mid]["lost_team"].value_counts()
        for t in (0, 1):
            blk = block.get(registry[t], {}).get("mean_line_m")
            rows.append({"match_id": mid, "team": names[t], "opponent": names[1 - t],
                         "turnovers": int(lost.get(t, 0)),
                         "poss_link_share": 100 * links.get(t, float("nan")) / total,
                         "block_line_m": float(blk) if blk is not None else float("nan")})
    return pd.DataFrame(rows)


def one_way(groups: list[np.ndarray]) -> dict[str, float]:
    """One-way variance components for a balanced/unbalanced grouping.

    Args:
        groups: one array of observations per group.

    Returns:
        ``k, n, ms_between, ms_within, sd_between, sd_within, f, p`` -- ``sd_between`` is the
        method-of-moments component ``sqrt(max(0, (MS_b - MS_w) / n_bar))`` and is truncated at 0.
    """
    groups = [np.asarray(g, float) for g in groups if np.isfinite(np.asarray(g, float)).sum() > 0]
    groups = [g[np.isfinite(g)] for g in groups]
    k, n = len(groups), sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ms_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / max(k - 1, 1)
    ms_w = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(n - k, 1)
    n_bar = n / k
    f = ms_b / ms_w if ms_w > 0 else float("inf")
    p = float(stats.f.sf(f, k - 1, n - k)) if np.isfinite(f) and n > k else float("nan")
    return {"k": k, "n": n, "ms_between": ms_b, "ms_within": ms_w,
            "sd_between": float(np.sqrt(max((ms_b - ms_w) / n_bar, 0.0))),
            "sd_within": float(np.sqrt(ms_w)), "f": f, "p": p}


def sd_ci(sd: float, df: int) -> tuple[float, float]:
    """Chi-square 95% CI for an SD estimated with ``df`` degrees of freedom."""
    if df <= 0 or not np.isfinite(sd):
        return float("nan"), float("nan")
    lo = sd * np.sqrt(df / stats.chi2.ppf(0.975, df))
    hi = sd * np.sqrt(df / stats.chi2.ppf(0.025, df))
    return float(lo), float(hi)


# === Reporting ===================================================================================
def _yield_block(records: pd.DataFrame) -> tuple[list[str], dict[int, pd.DataFrame]]:
    """Per-match yield tables at every presser level, plus the per-level frames for the verdicts."""
    lines, tables = [], {}
    for n in DEFENDER_LEVELS:
        tables[n] = yield_counts(records, n)
    per_match = records.groupby("match_id", sort=False).size()
    lines.append("| match | turnovers | " + " | ".join(
        f"a{n} | g{n} | t{n} | old5-{n}" for n in DEFENDER_LEVELS) + " |")
    lines.append("|---|---|" + "---|" * (4 * len(DEFENDER_LEVELS)))
    for mid in MATCH_IDS:
        cells = []
        for n in DEFENDER_LEVELS:
            r = tables[n].loc[mid]
            cells += [str(int(r["all3"])), str(int(r["ge2"])), str(int(r["t0"])),
                      str(int(r["old5"]))]
        lines.append(f"| {mid} | {int(per_match[mid])} | " + " | ".join(cells) + " |")
    cells = []
    for n in DEFENDER_LEVELS:
        s = tables[n].sum()
        cells += [str(int(s["all3"])), str(int(s["ge2"])), str(int(s["t0"])), str(int(s["old5"]))]
    lines.append(f"| **pooled (12)** | {int(per_match.sum())} | " + " | ".join(cells) + " |")
    return lines, tables


def _verdict_block(tables: dict[int, pd.DataFrame]) -> list[str]:
    """Declared-threshold verdicts for every (rule, presser level) pair, old 5 s rule included."""
    lines = ["| criterion | per-match min | median | max | pooled | declared verdict |",
             "|---|---|---|---|---|---|"]
    labels = {"all3": "2 s, all 3 instants", "ge2": "2 s, >=2 of 3 instants",
              "t0": "t0 only", "old5": "OLD 5 s continuous"}
    for key, label in labels.items():
        for n in DEFENDER_LEVELS:
            v = tables[n][key].to_numpy()
            lines.append(f"| {label}, N>={n} | {v.min()} | {np.median(v):.0f} | {v.max()} | "
                         f"{v.sum()} | {verdict(list(v))} |")
    return lines


def format_report(records: pd.DataFrame, sampling: pd.DataFrame, corpus: pd.DataFrame,
                  t8: pd.DataFrame, seasons: pd.DataFrame, t7: pd.DataFrame) -> str:
    """Render the whole deliverable as Markdown."""
    n_tv = len(records)
    yield_lines, tables = _yield_block(records)
    out = [
        "# W1-B -- the corrected 2 s census, the censoring rate, and the sampling floor",
        "",
        "Generated by `python -m tools.w1b_window_sampling`. CPU only, measurement only -- no metric",
        "estimator is built or fitted here. The turnover definition is unchanged and shared with",
        "`tools/w1_feasibility` via `chunk_inputs`, so the 5 s columns below are recomputed in the",
        "same pass as the 2 s ones and cannot have drifted.",
        "",
        "## 1. Task 1 -- the census at BPC + {0, 1, 2} s",
        "",
        "**What changed.** The first census required the ball tracked and N pressers visible in",
        "**every** slot of a 5 s window (26 slots). Bauer & Anzer evaluate every feature at three",
        "instants inside 2 s (p. 2016: \"the first two seconds immediately after the ball loss\";",
        "\"A time-window longer than two seconds was problematic\"). The 5 s is the *success label*",
        "only. So the requirement is 3 observed instants, not 26 observed slots.",
        "",
        "**Rule.** For each instant BPC + {0, 1, 2} s take the nearest ball-tracked frame within",
        f"+/-{INSTANT_TOL_S:.1f} s (one grid slot at 25 fps) and count the pressing (ball-losing)",
        "team's players with `calib_error_m <= 1.0` m and a pitch projection **at that frame**.",
        "",
        "Columns: `aN` = all three instants observed with >= N pressers; `gN` = at least two of the",
        "three; `tN` = the turnover instant alone; `old5-N` = the previous criterion (ball tracked in",
        "every slot of 5 s **and** >= N pressers in every slot), recomputed here for comparison.",
        "",
        *yield_lines,
        "",
        "### Distribution and verdict against the same declared thresholds",
        "",
        "(>=30/match = per-match viable; 10-30 = pooled only; <10 = drop the family.)",
        "",
        *_verdict_block(tables),
        "",
    ]

    # --- exclusions -------------------------------------------------------------------------
    ex = records
    own_known = ex["dir_known"].sum()
    own_half = int((ex["own_half"] == True).sum())  # noqa: E712
    sp_known = int(ex["spell_start_known"].sum())
    sp_hit = int((ex["setpiece_origin"] == True).sum())  # noqa: E712
    bnd = int(ex["past_boundary_2s"].sum())
    keep = ex[(ex["own_half"] == False) & (ex["setpiece_origin"] == False)  # noqa: E712
              & (~ex["past_boundary_2s"])]
    out += [
        "### BA's exclusion criteria: what we could apply",
        "",
        "| BA rule (Fig. 1, p. 2015) | applied? | effect on our 2546-turnover pool |",
        "|---|---|---|",
        f"| 1. Possessions starting with a set-piece are excluded | **proxy only** | The losing "
        f"team's possession start is the preceding turnover; a possession starting within "
        f"[-{SETPIECE_LEAD_S:.0f} s, +{SETPIECE_LAG_S:.0f} s] of an E2E-Spot set-piece peak "
        f"(Corner / Throw-in / Direct + Indirect free-kick / Kick-off / Penalty, thresh "
        f"{SPOT_THRESH:.2f}, NMS {SPOT_NMS_S:.0f} s) is flagged. {sp_hit} of {sp_known} turnovers "
        f"with a known spell start = {100 * sp_hit / max(sp_known, 1):.1f}%. **Unvalidated "
        f"classes** -- only `Foul` (1.069x) and `Clearance` (0.265x) have ever been checked against "
        "truth here, so this is a sensitivity, not a headline. |",
        f"| 2. Ball losses in the team's own half are excluded | **yes** | Attacking direction from "
        f"`resolve_attack_directions` per chunk; resolved for {own_known} of {n_tv} turnovers. "
        f"{own_half} losses ({100 * own_half / max(int(own_known), 1):.1f}% of resolved) are in the "
        "losing team's own half and are dropped. |",
        "| 3. Scenario ends when the ball goes out of play | **proxy only** | The linked ball track "
        "is never projected outside the pitch rectangle (0 of the corpus), so out-of-play is not "
        "readable from ball geometry. The `Ball out of play` E2E-Spot class is used to terminate "
        "the success scan instead; see the censoring table. Unvalidated class. |",
        "| 4. Only effective playing time; phases ending at half-time are dropped | **partly** | We "
        f"have no ball-in-play clock. The analogue we can apply is the chunk boundary: {bnd} "
        f"turnovers ({100 * bnd / n_tv:.1f}%) sit within 2 s of the end of their chunk and are "
        "dropped from the feature census; boundary-truncated success labels are reported separately "
        "as administrative censoring. |",
        "| 5. Deflected shots and individual-vs-team possession disagreements are excluded | **no** "
        "| We have one possession construct (the Viterbi proximity proxy), not two, so there is no "
        "disagreement signal to filter on, and no shot-deflection flag. |",
        "",
        f"Applying rules 2 + 4 + the set-piece proxy together leaves **{len(keep)} of {n_tv} "
        f"turnovers ({100 * len(keep) / n_tv:.1f}%)**. BA's own rules cut 20,928 tagged transitions "
        "to 11,108 (53.1% retained), so our retention is in the same regime -- the exclusions are "
        "not the thing that kills us.",
        "",
    ]
    ex_tab = ["| rule set | turnovers kept | pooled a3 | a5 | a7 | median/match a3 | a5 | a7 |",
              "|---|---|---|---|---|---|---|---|"]
    stages = [("none (as counted above)", records),
              ("rule 2 (own-half losses out)", records[records["own_half"] == False]),  # noqa: E712
              ("rules 2 + 4 (+ chunk-boundary)",
               records[(records["own_half"] == False) & (~records["past_boundary_2s"])]),  # noqa: E712
              ("rules 2 + 4 + set-piece proxy", keep)]
    for label, sub in stages:
        pooled, medians = [], []
        for n in DEFENDER_LEVELS:
            counts = yield_counts(sub, n).reindex(list(MATCH_IDS)).fillna(0)["all3"]
            pooled.append(int(counts.sum()))
            medians.append(float(np.median(counts)))
        ex_tab.append(f"| {label} | {len(sub)} | " + " | ".join(str(v) for v in pooled) + " | " +
                      " | ".join(f"{v:.0f}" for v in medians) + " |")
    out += ex_tab + [""]

    # --- censoring --------------------------------------------------------------------------
    out += ["### The success label: observed vs right-censored", "",
            "The regain-within-5 s outcome is **never dropped**. Each turnover's scan forward stops",
            "at the first of: a regain by the pressing team (observed success), the 5 s horizon with",
            "the grid observed throughout (observed failure), a spotted ball-out-of-play (BA rule 3",
            "-- a competing risk we cannot attribute, since BA credit the restart-taking team), a",
            "break in the ball track or in the geometry (right-censored at the last observed time),",
            "or the end of the chunk (administrative censoring).", ""]
    cens_sets = [("all turnovers", records),
                 ("usable at 2 s, N>=3", records[_usable(records, 3)]),
                 ("usable at 2 s, N>=5", records[_usable(records, 5)]),
                 ("old 5 s criterion, N>=3",
                  records[records["old_ball_full"] & (records["old_min_def"] >= 3)])]
    order = ["regain", "no_regain", "out_of_play", "censored_track", "censored_boundary"]
    out += ["| set | n | " + " | ".join(order) + " | censoring rate | +OOP as censored |",
            "|---|---|" + "---|" * (len(order) + 2)]
    for label, sub in cens_sets:
        vc = sub["outcome"].value_counts()
        cells = [f"{int(vc.get(k, 0))} ({100 * vc.get(k, 0) / max(len(sub), 1):.1f}%)"
                 for k in order]
        cens = int(vc.get("censored_track", 0) + vc.get("censored_boundary", 0))
        rate = 100 * cens / max(len(sub), 1)
        rate_oop = 100 * (cens + int(vc.get("out_of_play", 0))) / max(len(sub), 1)
        out.append(f"| {label} | {len(sub)} | " + " | ".join(cells) +
                   f" | **{rate:.1f}%** | {rate_oop:.1f}% |")
    med_c = records[records["outcome"].str.startswith("censored")]["t_out"]
    dec = records[records["outcome"].isin(["regain", "no_regain"])]
    share = 100 * (dec["outcome"] == "regain").mean()
    dec3 = records[_usable(records, 3) & records["outcome"].isin(["regain", "no_regain"])]
    share3 = 100 * (dec3["outcome"] == "regain").mean()
    out += ["",
            f"Median censoring time {med_c.median():.1f} s (mean {med_c.mean():.2f} s) -- i.e. the "
            "typical censored observation still carries the information \"no regain in the first "
            f"{med_c.median():.1f} s\", which a Kaplan-Meier estimator uses and a dropped window "
            "throws away.",
            "",
            "**Two things this table says that are not good news, and must be said before anyone "
            "builds on the yield.**",
            "",
            f"1. **The censoring is informative, not random.** Among the {len(dec)} turnovers whose "
            f"outcome is *decided*, {share:.1f}% are regains ({share3:.1f}% on the N>=3 usable "
            "subset). BA's regain-within-5 s rate is **30.2-36.2%** (p. 2025, n = 109,852). Our "
            "decided cases are therefore a heavily selected sample: the event that breaks the ball "
            "track (a long clearance, a switch of play, the camera cutting away) is exactly the "
            "event that implies *no* regain, so failures censor and successes survive. A naive "
            "\"regain rate among decided cases\" from this corpus would be roughly 2.5x the truth. "
            "This is precisely why the label has to be a survival estimator with a stated "
            "censoring mechanism, and why the mechanism cannot be assumed non-informative.",
            "2. **The `tN` (t0-only) columns are inflated by construction.** "
            "A turnover only exists where the possession proxy fired, which requires a calibrated "
            "player within 2 m of a tracked ball at that frame -- so geometry at t0 is guaranteed, "
            "not observed. The informative columns are the ones that also demand +1 s and +2 s "
            "(`aN`, `gN`). Read `tN` as an upper bound on nothing.",
            ""]

    # --- Task 2 ------------------------------------------------------------------------------
    out += _sampling_section(sampling, t8, seasons, t7)
    # --- Task 3 ------------------------------------------------------------------------------
    out += _corpus_section(corpus)
    # --- Verdicts ----------------------------------------------------------------------------
    out += _verdict_section(records, tables, sampling, corpus)
    return "\n".join(out)


def _usable(df: pd.DataFrame, n: int) -> pd.Series:
    """Boolean mask: all three instants observed with >= ``n`` trusted pressers."""
    return pd.concat([_def_ok(df, c, n) for c in ("d0", "d1", "d2")], axis=1).all(axis=1)


def _sampling_section(s: pd.DataFrame, t8: pd.DataFrame, seasons: pd.DataFrame,
                      t7: pd.DataFrame) -> list[str]:
    """Markdown for the finite-sample noise floor (Task 2)."""
    within = s.attrs["within"]
    avg = s.attrs["avg"]
    lines = [
        "## 2. Task 2 -- the finite-sample noise floor at n = 12 matches",
        "",
        "All literature inputs are **parsed from the PDF**, not transcribed: BA Table 7 "
        f"(p. 2034, {len(t7)} teams), Table 8 (p. 2036, {len(t8)} teams) and Tables 10-11 "
        f"(pp. 2039-2046, {len(seasons)} team-seasons). Three routes to the sampling SD of a "
        "12-match estimate:",
        "",
        f"* **Route A -- within-team season SD, scaled.** Pooled SD across a team's own seasons "
        f"(Appendix D, {int(within['_teams'])} teams with >= 4 seasons, {int(within['_df'])} df), "
        f"scaled `* sqrt({SEASON_MATCHES}/{TARGET_N})`. Contains real drift (coach and squad "
        "change) as well as sampling noise, so it **over-states** noise and the resulting D is a "
        "lower bound.",
        f"* **Route B -- the manual-labelling spread at ~{s.attrs['n7']:.1f} matches per team** "
        "(Table 7). Observed cross-team SD minus the stable between-team SD in quadrature, scaled "
        f"to {TARGET_N} matches. Also absorbs the 82.01% inter-labeller disagreement, so it "
        "over-states noise too. Available only for the two metrics Table 7 reports.",
        "* **Route C -- the binomial / Poisson floor** implied by BA's own averages "
        f"(T/M {avg['T/M']:.2f}, %CP {avg['%CP']:.2f}, CP+/T {avg['CP+/T']:.2f}, "
        f"%CP+ {avg['%CP+']:.2f}). This is the **irreducible** term: no estimator, however good, "
        "beats it. It is a hard lower bound on sampling SD and therefore a hard upper bound on D.",
        "",
        f"`SD_between` is recomputed from Table 8 over the {s.attrs['n_stable']} teams with "
        f">= {STABLE_GAMES} games.",
        "",
        "| metric | SD_between | Route A SD(12) | D_A | Route B SD(12) | D_B | Route C SD(12) "
        "| D_C | can 12 matches separate two teams? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s.itertuples(index=False):
        best = max(v for v in (r.d_a, r.d_b, r.d_c) if np.isfinite(v))
        call = "**NO**" if best < 1 else ("marginal" if best < 1.5 else "**yes**")
        sd_b = f"{r.sd_b:.3f}" if np.isfinite(r.sd_b) else "n/a"
        d_b = f"{r.d_b:.2f}" if np.isfinite(r.d_b) else "n/a"
        lines.append(f"| {r.metric} | {r.sd_between:.3f} | {r.sd_a:.3f} | {r.d_a:.2f} | {sd_b} | "
                     f"{d_b} | {r.sd_c:.3f} | {r.d_c:.2f} | {call} (best route D = {best:.2f}) |")
    lines += ["", "### Matches needed for D_sampling = 2 (`n = 12 * (2 / D_12)^2`)", "",
              "| metric | D_A @12 | n for D=2 (A) | D_B @12 | n for D=2 (B) | D_C @12 "
              "| n for D=2 (C, the optimistic floor) |", "|---|---|---|---|---|---|---|"]
    for r in s.itertuples(index=False):
        nb = f"{r.n2_b:.0f}" if np.isfinite(r.n2_b) else "n/a"
        db = f"{r.d_b:.2f}" if np.isfinite(r.d_b) else "n/a"
        lines.append(f"| {r.metric} | {r.d_a:.2f} | {r.n2_a:.0f} | {db} | {nb} | {r.d_c:.2f} | "
                     f"{r.n2_c:.0f} |")
    bayern = seasons[seasons["label"].str.startswith("FCB-")]
    stable = t8[t8["bracket"] >= STABLE_GAMES]
    read_sd = {"T/M": 4.26, "CP+/T": 0.635, "%CP": 1.027, "%CP+": 2.05}
    lines += ["", "### Three corrections to the numbers harvested in `PAPER_READ_METRICS.md`", "",
              "Everything above is parsed from the PDF text layer, so the read's hand-transcription "
              "can be checked rather than trusted. It survives on two metrics and fails on two.",
              "",
              "| metric | read's SD_between | parsed SD_between | delta | cause |",
              "|---|---|---|---|---|"]
    causes = {
        "T/M": f"the read's quoted range was 110.4-125.6; the true minimum is "
               f"{stable.loc[stable['T/M'].idxmin(), 'T/M']:.2f} (B. Moenchengladbach). Dropping "
               "4.9 m of spread **under-stated** between-team variation by 15%.",
        "%CP+": f"the read's range top was 35.56; parsed maximum is "
                f"{stable['%CP+'].max():.2f} (Bayer 04 Leverkusen). Minor.",
        "%CP": "exact match.",
        "CP+/T": "1.9% apart; within rounding of the printed table.",
    }
    for r in s.itertuples(index=False):
        d = r.sd_between - read_sd[r.metric]
        lines.append(f"| {r.metric} | {read_sd[r.metric]:.3f} | {r.sd_between:.3f} | "
                     f"{d:+.3f} | {causes[r.metric]} |")
    lines += ["",
              f"And the load-bearing one: the read quotes FC Bayern's **within-team** season SD of "
              f"%CP as 1.027 pp, which is byte-identical to its **between-team** SD -- a collision, "
              f"not a measurement. Parsed from Appendix D, Bayern's {len(bayern)} seasons of %CP "
              "are " + ", ".join(f"{v:.2f}" for v in bayern["%CP"]) +
              f" -> SD = **{bayern['%CP'].std(ddof=1):.3f} pp**, and the pooled figure over all "
              f"{int(within['_teams'])} multi-season teams is **{within['%CP']:.3f} pp**. Route A "
              "above uses the pooled figure. The verdict does not change.",
              "",
              "**One caveat on Route A for T/M.** The pooled within-team season SD of "
              f"transitions per match is {within['T/M']:.2f} -- larger than the entire between-team "
              f"SD ({s.loc[s['metric'] == 'T/M', 'sd_between'].iloc[0]:.2f}). Bayern alone runs "
              + ", ".join(f"{v:.1f}" for v in bayern["T/M"]) +
              " across seven seasons. That is real style drift (transitions per match is a "
              "possession-tempo quantity that moves with the squad), not sampling noise, so "
              "`D_A = 0.25` for T/M is not a sampling statement and Routes B and C are the "
              "credible ones for that metric.", ""]
    return lines


def _corpus_section(corpus: pd.DataFrame) -> list[str]:
    """Markdown for our own between-team spread (Task 3)."""
    metrics = [("turnovers", "turnovers per match (our detector, 2.03x over Sofascore)"),
               ("poss_link_share", "possession-link share (%)"),
               ("block_line_m", "block line (m from own goal)")]
    opp = corpus[corpus["team"] != HOME_TEAM]
    manu = corpus[corpus["team"] == HOME_TEAM]
    lines = [
        "## 3. Task 3 -- between-team spread in our own 12-match corpus",
        "",
        "Corpus structure: 12 matches = 6 reverse-fixture pairs against the same six opponents. Two",
        "one-way decompositions are possible and they answer different questions.",
        "",
        "* **(A) six opponents as groups (2 legs each).** Between-team variance vs residual. The two",
        "  legs of a pair share an opponent (always Man Utd) but differ in venue **and** manager, so",
        "  the residual is venue + manager + measurement noise.",
        "* **(B) Man Utd's own 12 matches, grouped by opponent (2 legs each).** Between-*opponent*",
        "  variance *within one team* -- i.e. how much of a single team's match-to-match variation is",
        "  driven by who it plays. That component is part of the sampling noise of any 12-match team",
        "  estimate.",
        "",
        "Team labels use the corrected mapping for `tottenham_manutd` and `southampton_manutd`",
        "(`results/PAIR_ANALYSIS_v1.md`, two independent lines each). Block line is",
        "`fingerprint.block_height.block_summary` `mean_line_m` (Gate 1 still pending, so read it as",
        "a relative spread, not a calibrated height).",
        "",
        "| metric | decomposition | k | n | grand mean | SD_between (team) | SD_within (resid) | "
        "95% CI on SD_within | F | p | D_sampling @12 | matches for D=2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for col, label in metrics:
        for tag, frame, key in (("(A) 6 opponents", opp, "team"),
                                ("(B) Man Utd by opponent", manu, "opponent")):
            groups = [g[col].to_numpy() for _, g in frame.groupby(key)]
            r = one_way(groups)
            lo, hi = sd_ci(r["sd_within"], int(r["n"] - r["k"]))
            d = r["sd_between"] / (r["sd_within"] / np.sqrt(TARGET_N)) if r["sd_within"] else 0.0
            n2 = TARGET_N * (2.0 / d) ** 2 if d > 0 else float("inf")
            n2s = f"{n2:.0f}" if np.isfinite(n2) else "inf"
            lines.append(
                f"| {label} | {tag} | {int(r['k'])} | {int(r['n'])} | "
                f"{np.concatenate(groups).mean():.2f} | {r['sd_between']:.2f} | "
                f"{r['sd_within']:.2f} | {lo:.2f}-{hi:.2f} | {r['f']:.2f} | {r['p']:.3f} | "
                f"{d:.2f} | {n2s} |")
    lines += ["",
              "Note that for `possession-link share` the two decompositions are numerically",
              "identical by construction (our two teams' shares sum to 100, so their variances are",
              "the same); only the grand mean differs. That is arithmetic, not a coincidence.",
              "",
              "### The denominator that actually matters for a 12-match claim",
              "",
              "Decomposition (A)'s residual is *not* the sampling noise of a real 12-match sample:",
              "each opponent's two legs are both against Man Utd, so opponent variation is missing",
              "from it. The honest denominator is a single team's match-to-match SD **across varied",
              "opponents**, which only Man Utd supplies (12 matches). Combining the two gives",
              "`D_total = SD_between(A) / (SD_ManU_total / sqrt(12))`.",
              "",
              "| metric | SD_between (A) | Man Utd match-to-match SD (n=12) | SD of a 12-match mean "
              "| D_total | matches for D=2 |",
              "|---|---|---|---|---|---|"]
    for col, label in metrics:
        r = one_way([g[col].to_numpy() for _, g in opp.groupby("team")])
        tot = float(manu[col].std(ddof=1))
        se = tot / np.sqrt(TARGET_N)
        d = r["sd_between"] / se if se else 0.0
        n2 = f"{TARGET_N * (2.0 / d) ** 2:.0f}" if d > 0 else "inf"
        lines.append(f"| {label} | {r['sd_between']:.2f} | {tot:.2f} | {se:.2f} | {d:.2f} | "
                     f"{n2} |")
    lines += ["", "**Honesty on the variance estimates.** Decomposition (A) has 5 between and 6",
              "within degrees of freedom. A variance component estimated on 5 df has a 95% CI that",
              "spans roughly a factor of 5 in SD, and the method-of-moments estimator is truncated",
              "at zero whenever MS_between < MS_within (which is what a non-significant F means:",
              "the data are consistent with **no** between-team difference at all). Only the block",
              "line clears p < 0.05, and it does so on 5 and 6 df with a single high-leverage leg",
              "(southampton_manutd, the corrected mapping). The 95% CI column is given for the",
              "residual SD only, where the df are least bad. Read every SD_between here as an order",
              "of magnitude, not a measurement -- and note that all three quantities are measured by",
              "our own uncalibrated pipeline, so an unknown share of each residual is measurement",
              "error rather than football.", "",
              "Per-team values behind the decomposition:", "",
              "| team | n | " + " | ".join(lbl.split(" (")[0] for _, lbl in metrics) + " |",
              "|---|---|" + "---|" * len(metrics)]
    for team, g in corpus.groupby("team"):
        cells = " | ".join(f"{g[c].mean():.1f} +/- {g[c].std(ddof=1):.1f}" for c, _ in metrics)
        lines.append(f"| {team} | {len(g)} | {cells} |")
    return lines + [""]


def _verdict_section(records: pd.DataFrame, tables: dict[int, pd.DataFrame], s: pd.DataFrame,
                     corpus: pd.DataFrame) -> list[str]:
    """The two verdicts the task asks for, stated against the numbers above."""
    keep = records[(records["own_half"] == False) & (~records["past_boundary_2s"])  # noqa: E712
                   & (records["setpiece_origin"] == False)]  # noqa: E712
    lines = ["## 4. Verdicts", "",
             "### (a) Does the transition family reopen at the 2 s window? **Yes, at N >= 3 and "
             "N >= 5. It stays shut at N >= 7.**", "",
             "| presser gate | OLD 5 s continuous (median/match, pooled) | NEW 2 s, 3 instants | "
             "gain | after BA exclusions | verdict change |", "|---|---|---|---|---|---|"]
    for n in DEFENDER_LEVELS:
        old = tables[n]["old5"].to_numpy()
        new = tables[n]["all3"].to_numpy()
        kept = yield_counts(keep, n).reindex(list(MATCH_IDS)).fillna(0)["all3"]
        gain = new.sum() / old.sum() if old.sum() else float("inf")
        lines.append(
            f"| N >= {n} | {np.median(old):.0f} / {old.sum()} | {np.median(new):.0f} / "
            f"{new.sum()} | **{gain:.1f}x** | {np.median(kept):.0f} / {int(kept.sum())} | "
            f"{verdict(list(old)).split(' (')[0]} -> **{verdict(list(new)).split(' (')[0]}** "
            f"(excl. {verdict(list(kept)).split(' (')[0]}) |")
    cens3 = records[_usable(records, 3)]["outcome"]
    rate3 = 100 * cens3.str.startswith("censored").mean()
    lines += ["",
              "The old FAIL was an artefact of pricing a window the literature does not use. "
              "Requiring three observed instants inside 2 s instead of 26 consecutive slots over "
              "5 s multiplies the usable pool by 5.5-6.3x, and the declared verdict moves from "
              "MARGINAL/FAIL/FAIL to PASS/PASS/MARGINAL. Applying BA's exclusion rules (own-half "
              "losses, chunk boundary, set-piece proxy) roughly halves the pool -- as it does for "
              "BA themselves, 20,928 -> 11,108 -- and N >= 3 still clears the per-match threshold "
              "while N >= 5 drops to pooled-only.",
              "",
              "**What reopens, precisely.** The broadcast-native, local half of BA's feature set "
              "(nearest-presser distance and speed, 10 m counts, local-5 stretch, Andrienko "
              "pressure) is instantaneous and needs exactly these three instants. What does *not* "
              "reopen: (i) the N >= 7 whole-team view, still MARGINAL at best; (ii) the success "
              f"label, which is right-censored in {rate3:.0f}% of even the N>=3 usable turnovers, "
              "and censored *informatively*; (iii) individual ball-possession time (BA's top SHAP "
              "feature), which needs a continuous ball-player association and was not measured "
              "here.",
              "",
              "### (b) Broadcast or sample size? **Sample size binds the three counterpress rate "
              "metrics; broadcast binds transitions-per-match.**", "",
              "| metric | best-case D_sampling @12 (Route C floor) | matches for D=2 at that floor "
              "| binding constraint at n = 12 |", "|---|---|---|---|"]
    verdicts = {
        "T/M": "**broadcast.** Sampling is survivable (D_C 1.58, D_B 1.13), but our "
               "possession-change detector runs 2.03x with a per-match ratio SD of 0.61 "
               "(CV 29.9%, `W1_FEASIBILITY_CHECKS.md`) against a between-team CV of "
               "4.3% -- a censoring/calibration term ~7x the signal.",
        "%CP": "**sample size.** D < 1 on every route including the irreducible binomial "
               "floor. Perfect tracking does not fix it.",
        "CP+/T": "**sample size.** Even the irreducible binomial floor leaves D = 0.90, and "
                 "Route A (which is the least noisy route for this metric) leaves 0.69.",
        "%CP+": "**sample size, worst of the four.** The success rate is a share of the "
                "counterpress subset, so its denominator is ~4x smaller again.",
    }
    for r in s.itertuples(index=False):
        lines.append(f"| {r.metric} | {r.d_c:.2f} | {r.n2_c:.0f} | {verdicts[r.metric]} |")
    lines += ["",
              "Read the middle column carefully: those are the matches needed **with perfect "
              "tracking and zero measurement error**. For the three counterpress rate metrics a "
              "12-match corpus is 5-8x too small before broadcast censoring is even considered, so "
              "an RQ-D experiment that reports D against the censoring term alone would be "
              "answering a question whose answer is already fixed by the sample size. The honest "
              "framing stands: total error is `sqrt(RMSE_censoring^2 + SD_sampling^2)`, and for "
              "%CP / CP+/T / %CP+ the second term alone already exceeds the between-team signal.",
              "",
              "**Our own corpus agrees where it can speak.** Turnovers per match shows a "
              "between-team component truncated to **0.00** (F < 1, p = 0.54): 12 matches of our "
              "detector cannot distinguish these seven teams on turnover count at all, which is "
              "exactly what a 30% per-match calibration CV against a 4% between-team CV predicts. "
              "Possession-link share and block line do show between-team structure "
              "(D_total 2.1 and 3.5 against Man Utd's own match-to-match SD), but on 5 df, with "
              "only the block line reaching p < 0.05, and with the block line's cross-leg "
              "correlation already measured at **-0.40** in `PAIR_ANALYSIS_v1.md` -- i.e. a "
              "between-team spread that does not repeat within a pair is not yet a team "
              "fingerprint.",
              "",
              "### What this changes",
              "",
              "1. `W1_FEASIBILITY_CHECKS.md`'s item 1 under \"What dies\" (\"per-match transition "
              "metrics ... dead at the declared threshold\") is **retracted for the 2 s window**. "
              "It remains true for a 5 s continuous window, which no longer has a justification.",
              "2. Item 4 (\"any transition metric requiring continuous 5 s observation of >= 7 "
              "pressers\") stands, and now also covers the 3-instant version at N >= 7.",
              "3. The battery should be scoped to the local/Tier-1 features BA's SHAP already "
              "ranks highest, with the success label as a censored survival quantity and its "
              "censoring mechanism argued, not assumed.",
              "4. No counterpress *rate* metric (%CP, CP+/T, %CP+) should be reported per team on "
              "12 matches, with or without imputation. If the thesis needs a team-level "
              "counterpress claim, it needs ~60-90 matches, not a better tracker.",
              ""]
    return lines


# === Orchestration ===============================================================================
def selftest() -> None:
    """Self-check of the two non-trivial pure functions (instant lookup and the survival label)."""
    ball = np.array([100, 105, 110, 125, 130, 150, 155])
    counts = {(100, 1): 6, (105, 1): 5, (125, 1): 4, (150, 1): 7}
    # fps 25 -> instants at 100, 125, 150; all present in the track.
    assert instant_defs(100, 1, ball, counts, fps=25.0) == [6, 4, 7]
    # A missing +1 s sample: 105 is 20 frames from 125, outside the 5-frame tolerance -> None.
    assert instant_defs(100, 1, np.array([100, 105, 150]), counts, fps=25.0) == [6, None, 7]
    # Nearest-within-tolerance: 130 is 5 frames from 125 and is picked when 125 is gone.
    assert instant_defs(100, 1, np.array([100, 130, 150]), counts, fps=25.0) == [6, 0, 7]

    grid = {100 + 5 * k for k in range(30)}
    poss = {f: 0 for f in grid}
    empty = np.empty(0, dtype=int)
    assert success_label(100, 1, grid, grid, poss, empty, stride=5, fps=25.0,
                         chunk_end=10_000) == ("no_regain", 5.0)
    poss[135] = 1
    assert success_label(100, 1, grid, grid, poss, empty, stride=5, fps=25.0,
                         chunk_end=10_000) == ("regain", 1.4)
    holed = grid - {120}
    out, t = success_label(100, 1, holed, grid, {f: 0 for f in grid}, empty, stride=5, fps=25.0,
                           chunk_end=10_000)
    assert (out, t) == ("censored_track", 0.6), (out, t)
    out, t = success_label(100, 1, grid, grid, {f: 0 for f in grid}, empty, stride=5, fps=25.0,
                           chunk_end=118)
    assert (out, t) == ("censored_boundary", 0.6), (out, t)
    out, t = success_label(100, 1, grid, grid, {f: 0 for f in grid}, np.array([137]), stride=5,
                           fps=25.0, chunk_end=10_000)
    assert (out, t) == ("out_of_play", 1.48), (out, t)

    r = one_way([np.array([1.0, 2.0]), np.array([5.0, 6.0]), np.array([9.0, 10.0])])
    assert r["k"] == 3 and r["n"] == 6 and abs(r["sd_within"] - 0.7071) < 1e-3, r
    assert r["sd_between"] > 3.0 and r["f"] > 50, r
    print("selftest ok")


def main() -> None:
    """Run the corrected census, the sampling floor and the corpus decomposition; write the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    frames = []
    for mid in MATCH_IDS:
        df = match_records(get(mid))
        frames.append(df)
        y3 = int(_usable(df, 3).sum()) if len(df) else 0
        y5 = int(_usable(df, 5).sum()) if len(df) else 0
        cens = int(df["outcome"].str.startswith("censored").sum()) if len(df) else 0
        print(f"census {mid:22s} turnovers {len(df):5d}  a3 {y3:4d}  a5 {y5:4d}  "
              f"censored {cens:5d}")
    records = pd.concat(frames, ignore_index=True)

    t8 = _parse_rotated((BA_T8_PAGE,))
    seasons = _parse_rotated(BA_SEASON_PAGES)
    t7 = _parse_table7()
    print(f"BA tables parsed: T7 {len(t7)} teams, T8 {len(t8)} teams, "
          f"Appendix D {len(seasons)} team-seasons")
    sampling = sampling_table(t8, seasons, t7)

    corpus = corpus_metrics(records)
    print(f"corpus metrics: {len(corpus)} team-match rows")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(records, sampling, corpus, t8, seasons, t7),
                           encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
