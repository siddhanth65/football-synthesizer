"""Feasibility probe for Lucey et al. (AAAI 2012) entropy maps on broadcast-derived ball tracks.

Replicates "Characterizing Multi-Agent Team Behavior from Partial Team Tracings" with the only two
inputs that method needs -- ball position and possessing team, at 1 Hz -- taken from our own CV
pipeline (``generator.ball.link_ball`` track + the Viterbi possession smoother). Everything the
paper assumes is dense (1 Hz ball for 90 min, 380 games) is *sparse* here, so the probe leads with a
census and reports every number that could inflate the result: how much of each possession string is
interpolated rather than observed, how many play-segments land in each grid cell, and a permutation
null for the identity test.

Stages (``--stage all`` runs them in order):
    census      -- possession-string supply per match at T in {3, 5, 8} s and gap policies.
    maps        -- entropy maps per team per match + aggregates, with per-cell sample counts.
    identity    -- leave-one-match-out 1-NN team identification + permutation null.
    deviation   -- per-match distance from the Man Utd leave-one-out aggregate map.

Parameters are DECLARED in the module constants below and swept for sensitivity; nothing is tuned to
the identity result. Run::

    python -m tools.entropy_map_probe --stage all
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID
from core.registry import Match, matches
from fingerprint.structural_metrics import (
    resolve_attack_directions,
    resolve_attack_directions_from_ball,
)
from fingerprint.theory_metrics import complete_directions
from generator.ball import assign_possession

CALIB_MAX_M = 1.0            # trusted-geometry gate on player rows, identical to report.facts
SAMPLE_HZ = 1.0              # Lucey's rate; our ball track is denser (5 Hz) and is decimated to this
GAP_S = 2                    # DECLARED gap policy: fill holes <= this many seconds inside a string
GAP_SWEEP = (0, 1, 2, 3, 5, 10)  # 0 = no interpolation at all (observed-only strings)
T_SWEEP = (3, 5, 8)          # play-segment window length in 1 Hz samples
T_PRIMARY = 3                # census-driven: T=5 (Lucey's optimum) leaves <100 segments/match-map
GRID_SWEEP = ((4, 3), (6, 4), (8, 6), (10, 8), (20, 16))
GRID_PRIMARY = (4, 3)        # census-driven (>= ~5 segments/cell/match-map), not results-driven
MIN_CELL_N = 5               # a cell needs this many starting segments before its entropy is used
SCRATCH = Path("results/entropy_map")


@dataclass(frozen=True)
class MapResult:
    """One team's entropy map plus the sample counts that produced it.

    Attributes:
        key: label of the map, e.g. ``"manutd_liverpool:Man Utd"``.
        team: team name.
        match_id: registry id the map came from (``"AGG"`` for pooled maps).
        entropy: per-cell Shannon entropy (bits), NaN where the cell is under-sampled.
        counts: per-cell number of play-segments starting there.
        n_segments: total play-segments in the map.
    """

    key: str
    team: str
    match_id: str
    entropy: np.ndarray
    counts: np.ndarray
    n_segments: int


# --------------------------------------------------------------------------------------------
# stage 0: 1 Hz ball + possession sequences
# --------------------------------------------------------------------------------------------
def sequence_table(m: Match) -> pd.DataFrame:
    """Decimate one match's linked-ball track + Viterbi possession to a 1 Hz sequence per chunk.

    Ball coordinates are stored raw (pitch metres, parquet frame) together with the possessing
    team's attacking-direction sign for that chunk, so a map can normalise to "attacking towards
    x = 105" without re-resolving directions.

    Args:
        m: registry match.

    Returns:
        ``match, chunk, t, x, y, team, dir, observed`` -- one row per second that carries a ball
        position; ``team`` is -1 where no player was within the possession radius.
    """
    aligned = m.load_aligned()
    rows: list[pd.DataFrame] = []
    for ck, path in m.ball_chunks():
        ball = pd.read_parquet(path)
        fps = m.chunk_fps(ck)
        pos = aligned[(aligned["chunk"] == ck) & (aligned["calib_error_m"] <= CALIB_MAX_M)].dropna(
            subset=["pitch_x", "pitch_y"])
        if ball.empty or pos.empty:
            continue
        dirs = resolve_attack_directions_from_ball(pos, ball)
        if len(dirs) < 2:
            dirs = resolve_attack_directions(pos)
        dirs = complete_directions(dirs)
        if len(dirs) < 2:
            continue
        poss = assign_possession(ball, pos, smooth=True)
        team_of = dict(zip(poss["frame"].astype(int), poss["team"].astype(int))) if len(poss) else {}
        b = ball.copy()
        b["t_exact"] = b["frame"] / fps
        b["t"] = b["t_exact"].round().astype(int)
        b["err"] = (b["t_exact"] - b["t"]).abs()
        b = b.sort_values("err").drop_duplicates("t").sort_values("t")
        b["team"] = [team_of.get(int(f), -1) for f in b["frame"]]
        b["dir"] = [dirs.get(int(t), 0) for t in b["team"]]
        rows.append(pd.DataFrame({
            "match": m.id, "chunk": ck, "t": b["t"].to_numpy(), "x": b["x"].to_numpy(),
            "y": b["y"].to_numpy(), "team": b["team"].to_numpy(), "dir": b["dir"].to_numpy(),
            "observed": b["observed"].to_numpy(),
        }))
    if not rows:
        return pd.DataFrame(columns=["match", "chunk", "t", "x", "y", "team", "dir", "observed"])
    return pd.concat(rows, ignore_index=True)


def build_cache(match_ids: list[str], cache: Path) -> pd.DataFrame:
    """Build (or reuse) the 1 Hz sequence cache for the given matches."""
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(df["match"].unique()) >= set(match_ids):
            return df[df["match"].isin(match_ids)].reset_index(drop=True)
    reg = {m.id: m for m in matches()}
    out = pd.concat([sequence_table(reg[mid]) for mid in match_ids], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


# --------------------------------------------------------------------------------------------
# stage 1: possession strings + play-segments
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class String:
    """One possession string: contiguous 1 Hz ball positions held by a single team.

    Attributes:
        match_id: registry id.
        team_id: 0/1 team index in the aligned parquet.
        x: ball x per second, normalised so the team attacks towards ``x = PITCH_LEN``.
        y: ball y per second (same 180-degree rotation applied as to x).
        observed: True only where the second carried a ball sample that ``link_ball`` marked
            observed; False for both ``link_ball``-interpolated samples and probe-filled holes.
        probe_filled: True where this probe invented the position (no 1 Hz ball sample at all).
    """

    match_id: str
    team_id: int
    x: np.ndarray
    y: np.ndarray
    observed: np.ndarray
    probe_filled: np.ndarray


def possession_strings(seq: pd.DataFrame, *, gap_s: int) -> list[String]:
    """Split a 1 Hz sequence into possession strings under a declared gap policy.

    A string is a maximal run of seconds labelled with the same possessing team. Holes -- seconds
    with no ball sample, or with a ball sample but no carrier within the possession radius -- are
    bridged when they are at most ``gap_s`` long, with ball position filled by linear interpolation
    between the flanking observed positions. A hole longer than ``gap_s``, or any second labelled
    with the *other* team, ends the string (turnover / stoppage / lost track).

    Args:
        seq: rows for one match from :func:`sequence_table`.
        gap_s: maximum bridged hole in seconds; 0 disables interpolation entirely.

    Returns:
        Possession strings, direction-normalised.
    """
    out: list[String] = []
    for (_, _), g in seq.groupby(["match", "chunk"], sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        lo, hi = int(t[0]), int(t[-1])
        n = hi - lo + 1
        xs = np.full(n, np.nan)
        ys = np.full(n, np.nan)
        lab = np.full(n, -1, dtype=int)
        src = np.zeros(n, dtype=bool)
        idx = t - lo
        xs[idx] = g["x"].to_numpy()
        ys[idx] = g["y"].to_numpy()
        lab[idx] = g["team"].to_numpy()
        src[idx] = g["observed"].to_numpy()
        dirs = {int(tm): int(d) for tm, d in zip(g["team"], g["dir"]) if int(tm) >= 0}
        for team in sorted({v for v in lab if v >= 0}):
            hits = np.flatnonzero(lab == team)
            if hits.size == 0:
                continue
            other = np.flatnonzero((lab >= 0) & (lab != team))
            runs: list[list[int]] = [[int(hits[0])]]
            for a, b in zip(hits[:-1], hits[1:]):
                hole = int(b - a) - 1
                blocked = bool(((other > a) & (other < b)).any())
                if hole <= gap_s and not blocked:
                    runs[-1].append(int(b))
                else:
                    runs.append([int(b)])
            sign = dirs.get(team, 1) or 1
            for run in runs:
                a, b = run[0], run[-1]
                if b == a:
                    continue
                sx, sy = xs[a:b + 1].copy(), ys[a:b + 1].copy()
                have = np.isfinite(sx)
                if not have.all():
                    k = np.arange(sx.size)
                    sx = np.interp(k, k[have], sx[have])
                    sy = np.interp(k, k[have], sy[have])
                if sign < 0:  # 180-degree rotation, not a mirror: attacking frame of reference
                    sx, sy = PITCH_LEN - sx, PITCH_WID - sy
                out.append(String(str(g["match"].iloc[0]), int(team), sx, sy,
                                  src[a:b + 1].copy(), ~have))
    return out


def cell_ids(x: np.ndarray, y: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Quantise pitch coordinates to flat grid-cell indices (clipped to the pitch)."""
    nx, ny = grid
    ix = np.clip((x / PITCH_LEN * nx).astype(int), 0, nx - 1)
    iy = np.clip((y / PITCH_WID * ny).astype(int), 0, ny - 1)
    return ix * ny + iy


def segments(strings: list[String], *, T: int, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-``T`` window over each string; return (start cell, destination cell) arrays.

    A play-segment is ``T`` consecutive 1 Hz samples, so the ball's travel horizon is ``T - 1`` s.
    A string of ``T1`` samples yields ``T1 - T + 1`` segments; shorter strings are discarded.
    """
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for s in strings:
        if s.x.size < T:
            continue
        c = cell_ids(s.x, s.y, grid)
        starts.append(c[:c.size - T + 1])
        ends.append(c[T - 1:])
    if not starts:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(starts), np.concatenate(ends)


def entropy_map(start: np.ndarray, end: np.ndarray, *, grid: tuple[int, int],
                min_n: int = MIN_CELL_N) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell Shannon entropy (bits) of the destination distribution, plus per-cell counts.

    Cells with fewer than ``min_n`` starting segments get NaN entropy: at small n the plug-in
    estimator is severely biased towards 0 and would fabricate "low entropy" structure.
    """
    ncell = grid[0] * grid[1]
    counts = np.bincount(start, minlength=ncell).astype(float)
    ent = np.full(ncell, np.nan)
    for c in np.flatnonzero(counts >= min_n):
        d = np.bincount(end[start == c], minlength=ncell).astype(float)
        p = d[d > 0] / d.sum()
        ent[c] = float(-(p * np.log2(p)).sum())
    return ent, counts


def map_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Mean absolute entropy difference over cells valid in both maps, and that cell count."""
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan"), 0
    return float(np.abs(a[m] - b[m]).mean()), int(m.sum())


def occupancy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon distance between two start-cell occupancy distributions (control feature)."""
    pa, pb = a / max(a.sum(), 1.0), b / max(b.sum(), 1.0)
    mm = 0.5 * (pa + pb)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        k = p > 0
        return float((p[k] * np.log2(p[k] / q[k])).sum())

    return float(np.sqrt(max(0.5 * _kl(pa, mm) + 0.5 * _kl(pb, mm), 0.0)))


# --------------------------------------------------------------------------------------------
# stage 2: census
# --------------------------------------------------------------------------------------------
def manutd_matches() -> list[Match]:
    """The 12 processed Man Utd EPL 24-25 matches, in date order."""
    ms = [m for m in matches(processed_only=True)
          if "Man Utd" in m.teams and m.ball_dir is not None and m.ball_chunks()]
    return sorted(ms, key=lambda m: m.date or "")


def census(cache: pd.DataFrame, ms: list[Match]) -> pd.DataFrame:
    """Per match: 1 Hz supply, possession-string lengths, interpolation share, segment counts."""
    rows = []
    for m in ms:
        seq = cache[cache["match"] == m.id]
        span = int(seq.groupby("chunk")["t"].agg(lambda s: s.max() - s.min() + 1).sum())
        mu_i = m.teams.index("Man Utd")
        row = {
            "match": m.id, "mgr": m.manager, "chunks": seq["chunk"].nunique(),
            "span_s": span, "ball_s": len(seq), "ball_cov": len(seq) / max(span, 1),
            "link_obs": float(seq["observed"].mean()) if len(seq) else float("nan"),
            "poss_s": int((seq["team"] >= 0).sum()),
            "poss_cov": float((seq["team"] >= 0).mean()) if len(seq) else float("nan"),
        }
        for gap in GAP_SWEEP:
            st = possession_strings(seq, gap_s=gap)
            lens = np.array([s.x.size for s in st]) if st else np.zeros(0, int)
            mu = [s for s in st if s.team_id == mu_i]
            row[f"g{gap}_strings"] = len(st)
            row[f"g{gap}_sec"] = int(lens.sum())
            row[f"g{gap}_med"] = float(np.median(lens)) if lens.size else 0.0
            row[f"g{gap}_p90"] = float(np.percentile(lens, 90)) if lens.size else 0.0
            row[f"g{gap}_max"] = int(lens.max()) if lens.size else 0
            row[f"g{gap}_obs"] = (float(np.concatenate([s.observed for s in st]).mean())
                                  if st else float("nan"))
            row[f"g{gap}_pfill"] = (float(np.concatenate([s.probe_filled for s in st]).mean())
                                    if st else float("nan"))
            for T in T_SWEEP:
                row[f"g{gap}_T{T}_strings"] = int((lens >= T).sum())
                row[f"g{gap}_T{T}_segs"] = int(np.clip(lens - T + 1, 0, None).sum())
                row[f"g{gap}_T{T}_segs_mu"] = int(sum(max(s.x.size - T + 1, 0) for s in mu))
        rows.append(row)
    return pd.DataFrame(rows)


def print_census(c: pd.DataFrame) -> None:
    """ASCII census tables (supply, string lengths, segment yield)."""
    print("\n== CENSUS A: 1 Hz supply per match (gap policy not yet applied) ==")
    print(f"{'match':<22}{'mgr':<9}{'chk':>4}{'span_s':>8}{'ball_s':>8}{'ballcov':>9}"
          f"{'linkobs':>9}{'poss_s':>8}{'posscov':>9}")
    for r in c.itertuples(index=False):
        print(f"{r.match:<22}{str(r.mgr):<9}{r.chunks:>4}{r.span_s:>8}{r.ball_s:>8}"
              f"{r.ball_cov:>9.3f}{r.link_obs:>9.3f}{r.poss_s:>8}{r.poss_cov:>9.3f}")
    print(f"{'POOLED':<22}{'':<9}{c.chunks.sum():>4}{c.span_s.sum():>8}{c.ball_s.sum():>8}"
          f"{c.ball_s.sum() / c.span_s.sum():>9.3f}{c.link_obs.mean():>9.3f}"
          f"{c.poss_s.sum():>8}{c.poss_s.sum() / c.ball_s.sum():>9.3f}")
    print("  ball_cov = 1 Hz seconds with a ball position / chunk span; link_obs = share of those")
    print("  samples link_ball marked observed; poss_cov = share of ball seconds with a carrier.")
    for gap in GAP_SWEEP:
        print(f"\n== CENSUS B: possession strings, gap policy G={gap} s ==")
        print(f"{'match':<22}{'strings':>8}{'secs':>7}{'med':>6}{'p90':>6}{'max':>6}{'obs':>7}"
              f"{'pfill':>7}"
              + "".join(f"{'T'+str(T)+'str':>9}{'T'+str(T)+'seg':>9}{'T'+str(T)+'MU':>8}"
                        for T in T_SWEEP))
        for r in c.itertuples(index=False):
            d = r._asdict()
            print(f"{r.match:<22}{d[f'g{gap}_strings']:>8}{d[f'g{gap}_sec']:>7}"
                  f"{d[f'g{gap}_med']:>6.1f}{d[f'g{gap}_p90']:>6.1f}{d[f'g{gap}_max']:>6}"
                  f"{d[f'g{gap}_obs']:>7.3f}{d[f'g{gap}_pfill']:>7.3f}"
                  + "".join(f"{d[f'g{gap}_T{T}_strings']:>9}{d[f'g{gap}_T{T}_segs']:>9}"
                            f"{d[f'g{gap}_T{T}_segs_mu']:>8}" for T in T_SWEEP))
        tot = {k: c[k].sum() for k in c.columns if k.startswith(f"g{gap}_")}
        print(f"{'POOLED':<22}{tot[f'g{gap}_strings']:>8}{tot[f'g{gap}_sec']:>7}{'':>6}{'':>6}{'':>6}"
              f"{c[f'g{gap}_obs'].mean():>7.3f}{c[f'g{gap}_pfill'].mean():>7.3f}"
              + "".join(f"{tot[f'g{gap}_T{T}_strings']:>9}{tot[f'g{gap}_T{T}_segs']:>9}"
                        f"{tot[f'g{gap}_T{T}_segs_mu']:>8}" for T in T_SWEEP))


# --------------------------------------------------------------------------------------------
# stage 3: entropy maps
# --------------------------------------------------------------------------------------------
def build_maps(cache: pd.DataFrame, ms: list[Match], *, gap_s: int, T: int,
               grid: tuple[int, int], min_n: int = MIN_CELL_N) -> list[MapResult]:
    """One entropy map per (match, team): 24 maps for the 12 Man Utd fixtures."""
    out: list[MapResult] = []
    for m in ms:
        st = possession_strings(cache[cache["match"] == m.id], gap_s=gap_s)
        for ti, name in enumerate(m.teams):
            s, e = segments([x for x in st if x.team_id == ti], T=T, grid=grid)
            ent, cnt = entropy_map(s, e, grid=grid, min_n=min_n)
            out.append(MapResult(f"{m.id}:{name}", name, m.id, ent, cnt, int(s.size)))
    return out


def pooled_map(cache: pd.DataFrame, ms: list[Match], team: str, *, gap_s: int, T: int,
               grid: tuple[int, int], exclude: str | None = None,
               min_n: int = MIN_CELL_N) -> MapResult:
    """Aggregate map for one team over several matches (optionally leaving one out)."""
    ss: list[np.ndarray] = []
    es: list[np.ndarray] = []
    for m in ms:
        if m.id == exclude or team not in m.teams:
            continue
        st = possession_strings(cache[cache["match"] == m.id], gap_s=gap_s)
        ti = m.teams.index(team)
        s, e = segments([x for x in st if x.team_id == ti], T=T, grid=grid)
        ss.append(s)
        es.append(e)
    s = np.concatenate(ss) if ss else np.zeros(0, int)
    e = np.concatenate(es) if es else np.zeros(0, int)
    ent, cnt = entropy_map(s, e, grid=grid, min_n=min_n)
    return MapResult(f"AGG:{team}", team, "AGG", ent, cnt, int(s.size))


def map_summary(maps: list[MapResult], grid: tuple[int, int]) -> pd.DataFrame:
    """Per-map summary: segments, filled cells, valid cells, mean/max entropy."""
    ncell = grid[0] * grid[1]
    rows = []
    for mp in maps:
        v = np.isfinite(mp.entropy)
        rows.append({
            "map": mp.key, "team": mp.team, "match": mp.match_id, "n_seg": mp.n_segments,
            "cells_hit": int((mp.counts > 0).sum()), "cells_valid": int(v.sum()),
            "cells_total": ncell, "med_cell_n": float(np.median(mp.counts[mp.counts > 0]))
            if (mp.counts > 0).any() else 0.0,
            "H_mean": float(np.nanmean(mp.entropy)) if v.any() else float("nan"),
            "H_max": float(np.nanmax(mp.entropy)) if v.any() else float("nan"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# stage 4: identity test
# --------------------------------------------------------------------------------------------
def distance_matrix(maps: list[MapResult], *, feature: str = "entropy") -> np.ndarray:
    """Pairwise map distance (NaN on the diagonal, NaN where no cell is valid in both)."""
    n = len(maps)
    d = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if feature == "entropy":
                d[i, j] = map_distance(maps[i].entropy, maps[j].entropy)[0]
            else:
                d[i, j] = occupancy_distance(maps[i].counts, maps[j].counts)
    return d


def loo_accuracy(d: np.ndarray, labels: np.ndarray, *, restrict: np.ndarray | None = None) -> dict:
    """Leave-one-out 1-NN accuracy over a precomputed distance matrix.

    Args:
        d: pairwise distances (diagonal NaN).
        labels: class label per map.
        restrict: optional boolean mask of which maps may act as queries *and* neighbours.

    Returns:
        ``n``, ``acc`` (top-1), ``bal_acc`` (mean per-class recall), ``majority`` (largest-class
        share) and the per-query hit vector.
    """
    idx = np.flatnonzero(restrict) if restrict is not None else np.arange(len(labels))
    hits = []
    for i in idx:
        row = d[i, idx].copy()
        row[idx == i] = np.inf
        row = np.where(np.isfinite(row), row, np.inf)
        if not np.isfinite(row).any():
            hits.append(False)
            continue
        hits.append(labels[idx[int(np.argmin(row))]] == labels[i])
    hits = np.array(hits)
    lab = labels[idx]
    recalls = [hits[lab == c].mean() for c in np.unique(lab)]
    _, sizes = np.unique(lab, return_counts=True)
    return {"n": int(idx.size), "acc": float(hits.mean()), "bal_acc": float(np.mean(recalls)),
            "majority": float(sizes.max() / sizes.sum()), "hits": hits}


def permutation_null(d: np.ndarray, labels: np.ndarray, *, restrict: np.ndarray | None = None,
                     n_perm: int = 2000, seed: int = 0) -> dict:
    """Null distribution of LOO 1-NN accuracy under random relabelling (class sizes preserved)."""
    rng = np.random.default_rng(seed)
    obs = loo_accuracy(d, labels, restrict=restrict)
    idx = np.flatnonzero(restrict) if restrict is not None else np.arange(len(labels))
    accs, bals = [], []
    for _ in range(n_perm):
        perm = labels.copy()
        perm[idx] = rng.permutation(labels[idx])  # shuffle WITHIN the tested subset only
        r = loo_accuracy(d, perm, restrict=restrict)
        accs.append(r["acc"])
        bals.append(r["bal_acc"])
    accs, bals = np.array(accs), np.array(bals)
    return {
        "acc": obs["acc"], "bal_acc": obs["bal_acc"], "n": obs["n"], "majority": obs["majority"],
        "null_acc_mean": float(accs.mean()), "null_acc_p95": float(np.percentile(accs, 95)),
        "p_acc": float(((accs >= obs["acc"]).sum() + 1) / (n_perm + 1)),
        "null_bal_mean": float(bals.mean()), "null_bal_p95": float(np.percentile(bals, 95)),
        "p_bal": float(((bals >= obs["bal_acc"]).sum() + 1) / (n_perm + 1)),
    }


def pair_test(d: np.ndarray, maps: list[MapResult], *, restrict: np.ndarray) -> dict:
    """Same-team-other-leg retrieval: rank of the sibling map among the restricted set."""
    idx = np.flatnonzero(restrict)
    ranks, top1 = [], []
    for i in idx:
        peers = idx[idx != i]
        order = peers[np.argsort(np.where(np.isfinite(d[i, peers]), d[i, peers], np.inf))]
        sib = [k for k in peers if maps[k].team == maps[i].team]
        if not sib:
            continue
        r = min(int(np.flatnonzero(order == s)[0]) + 1 for s in sib)
        ranks.append(r)
        top1.append(r == 1)
    return {"n": len(ranks), "top1": float(np.mean(top1)) if top1 else float("nan"),
            "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "chance_top1": 1.0 / (len(idx) - 1), "ranks": ranks}


# Man Utd scoreline per fixture -- the one datum the registry lacks (venue/manager/date come from
# core.registry). Source: results/PAIR_ANALYSIS_v1.md, itself Sofascore-checked.
RESULTS = {
    "manutd_fulham": "W 1-0", "brighton_manutd": "L 1-2", "manutd_liverpool": "L 0-3",
    "southampton_manutd": "W 3-0", "palace_manutd": "D 0-0", "manutd_tottenham": "L 0-3",
    "liverpool_manutd": "D 2-2", "manutd_southampton": "W 3-1", "manutd_brighton": "L 1-3",
    "fulham_manutd": "W 1-0", "manutd_palace": "L 0-2", "tottenham_manutd": "L 0-1",
}


# --------------------------------------------------------------------------------------------
# stage 5: deviation from the Man Utd aggregate
# --------------------------------------------------------------------------------------------
def match_segments(cache: pd.DataFrame, m: Match, team: str, *, gap_s: int, T: int,
                   grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(start cell, destination cell) play-segments for one team in one match."""
    st = possession_strings(cache[cache["match"] == m.id], gap_s=gap_s)
    ti = m.teams.index(team)
    return segments([x for x in st if x.team_id == ti], T=T, grid=grid)


def deviation_table(cache: pd.DataFrame, ms: list[Match], *, gap_s: int, T: int,
                    grid: tuple[int, int], n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Distance of each Man Utd match map from the leave-one-out Man Utd aggregate map.

    Each observed distance is scored against a **resampling null**: draw the same number of
    play-segments at random from Man Utd's pooled corpus and measure that pseudo-match's distance
    to the same aggregate. Without this null a match's distance is unreadable, because a map built
    from 43 segments is far from any aggregate purely through estimator noise.
    """
    rng = np.random.default_rng(seed)
    per_match = {m.id: match_segments(cache, m, "Man Utd", gap_s=gap_s, T=T, grid=grid) for m in ms}
    all_s = np.concatenate([v[0] for v in per_match.values()])
    all_e = np.concatenate([v[1] for v in per_match.values()])
    rows = []
    for m in ms:
        agg = pooled_map(cache, ms, "Man Utd", gap_s=gap_s, T=T, grid=grid, exclude=m.id)
        s, e = per_match[m.id]
        ent, _ = entropy_map(s, e, grid=grid)
        d, ncell = map_distance(ent, agg.entropy)
        null = []
        for _ in range(n_boot):
            k = rng.choice(all_s.size, size=max(s.size, 1), replace=False)
            be, _ = entropy_map(all_s[k], all_e[k], grid=grid)
            null.append(map_distance(be, agg.entropy)[0])
        null = np.array([v for v in null if np.isfinite(v)])
        opp = next(t for t in m.teams if t != "Man Utd")
        rows.append({
            "match": m.id, "opponent": opp, "venue": "H" if m.home_team == "Man Utd" else "A",
            "mgr": "TH" if m.manager == "ten_hag" else "AM", "result": RESULTS.get(m.id, "?"),
            "n_seg": int(s.size), "cells": ncell, "d_agg": d,
            "null_med": float(np.median(null)) if null.size else float("nan"),
            "null_p95": float(np.percentile(null, 95)) if null.size else float("nan"),
            "p_dev": float(((null >= d).sum() + 1) / (null.size + 1)) if null.size else float("nan"),
            "H_mean": float(np.nanmean(ent)) if np.isfinite(ent).any() else float("nan"),
        })
    df = pd.DataFrame(rows)
    mu, sd = df["d_agg"].mean(), df["d_agg"].std(ddof=1)
    df["z"] = (df["d_agg"] - mu) / sd
    return df.sort_values("d_agg", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
def run_identity(cache: pd.DataFrame, ms: list[Match], *, gap_s: int, T: int,
                 grid: tuple[int, int], min_n: int = MIN_CELL_N, n_perm: int = 2000,
                 quiet: bool = False) -> dict:
    """Full identity block: 1-NN LOO over 24 maps, opponent-only pair test, permutation nulls."""
    maps = build_maps(cache, ms, gap_s=gap_s, T=T, grid=grid, min_n=min_n)
    labels = np.array([mp.team for mp in maps])
    d_ent = distance_matrix(maps, feature="entropy")
    d_occ = distance_matrix(maps, feature="occupancy")
    is_opp = labels != "Man Utd"
    out: dict = {"gap_s": gap_s, "T": T, "grid": list(grid), "min_n": min_n,
                 "n_maps": len(maps),
                 "n_seg_min": int(min(mp.n_segments for mp in maps)),
                 "n_seg_med": float(np.median([mp.n_segments for mp in maps]))}
    for name, d in (("entropy", d_ent), ("occupancy", d_occ)):
        out[f"{name}_all"] = permutation_null(d, labels, n_perm=n_perm)
        out[f"{name}_opp"] = permutation_null(d, labels, restrict=is_opp, n_perm=n_perm)
        out[f"{name}_pair"] = pair_test(d, maps, restrict=is_opp)
    if not quiet:
        _print_identity(out)
    return out


def _print_identity(o: dict) -> None:
    """Print the identity block for one parameter setting."""
    print(f"\n== IDENTITY  G={o['gap_s']} T={o['T']} grid={o['grid'][0]}x{o['grid'][1]} "
          f"min_n={o['min_n']}  ({o['n_maps']} maps, segments/map min={o['n_seg_min']} "
          f"med={o['n_seg_med']:.0f}) ==")
    for feat in ("entropy", "occupancy"):
        a, b, p = o[f"{feat}_all"], o[f"{feat}_opp"], o[f"{feat}_pair"]
        print(f"  [{feat}] 7-class LOO 1-NN over all {a['n']} maps: acc={a['acc']:.3f} "
              f"(majority={a['majority']:.3f}, null mean={a['null_acc_mean']:.3f} "
              f"p95={a['null_acc_p95']:.3f}, p={a['p_acc']:.3f})")
        print(f"      balanced acc={a['bal_acc']:.3f} (null mean={a['null_bal_mean']:.3f} "
              f"p95={a['null_bal_p95']:.3f}, p={a['p_bal']:.3f})")
        print(f"      6-class opponent-only ({b['n']} maps): acc={b['acc']:.3f} "
              f"(chance=1/11=0.091, null mean={b['null_acc_mean']:.3f} p={b['p_acc']:.3f})")
        print(f"      same-opponent other-leg retrieval: top1={p['top1']:.3f} "
              f"(chance={p['chance_top1']:.3f}) mean rank={p['mean_rank']:.2f} of n={p['n']}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Lucey entropy-map feasibility probe")
    ap.add_argument("--stage", default="all",
                    choices=["census", "maps", "identity", "deviation", "metrica", "all"])
    ap.add_argument("--cache", type=Path, default=SCRATCH / "seq_1hz.parquet")
    ap.add_argument("--gap", type=int, default=GAP_S)
    ap.add_argument("--T", type=int, default=T_PRIMARY)
    ap.add_argument("--grid", type=str, default=f"{GRID_PRIMARY[0]}x{GRID_PRIMARY[1]}")
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()
    grid = (int(args.grid.split("x")[0]), int(args.grid.split("x")[1]))

    ms = manutd_matches()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache = build_cache([m.id for m in ms], args.cache)
    print(f"matches: {len(ms)}; 1 Hz rows cached: {len(cache)}")

    if args.stage in ("census", "all"):
        c = census(cache, ms)
        print_census(c)
        c.to_csv(SCRATCH / "census.csv", index=False)

    if args.stage in ("maps", "all"):
        _print_maps(cache, ms, gap_s=args.gap, T=args.T, grid=grid)

    if args.stage in ("identity", "all"):
        res = [run_identity(cache, ms, gap_s=args.gap, T=args.T, grid=grid, n_perm=args.perm)]
        print("\n== IDENTITY SENSITIVITY (entropy feature) ==")
        print(f"{'G':>3}{'T':>3}{'grid':>8}{'segmed':>8}{'acc':>7}{'bal':>7}{'p_acc':>7}"
              f"{'oppacc':>8}{'pair1':>7}")
        for gap in GAP_SWEEP:
            for T in T_SWEEP:
                for g in GRID_SWEEP:
                    o = run_identity(cache, ms, gap_s=gap, T=T, grid=g, n_perm=500, quiet=True)
                    a, b, p = o["entropy_all"], o["entropy_opp"], o["entropy_pair"]
                    print(f"{gap:>3}{T:>3}{str(g[0]) + 'x' + str(g[1]):>8}{o['n_seg_med']:>8.0f}"
                          f"{a['acc']:>7.3f}{a['bal_acc']:>7.3f}{a['p_acc']:>7.3f}"
                          f"{b['acc']:>8.3f}{p['top1']:>7.3f}")
                    res.append(o)
        (SCRATCH / "identity.json").write_text(
            json.dumps(res, default=_jsonable, indent=1), encoding="utf-8")

    if args.stage in ("deviation", "all"):
        dv = deviation_table(cache, ms, gap_s=args.gap, T=args.T, grid=grid)
        print(f"\n== DEVIATION from Man Utd LOO-aggregate  G={args.gap} T={args.T} "
              f"grid={args.grid} ==")
        print(f"{'match':<22}{'opp':<16}{'V':<3}{'mgr':<4}{'res':<7}{'n_seg':>7}{'cells':>6}"
              f"{'d_agg':>8}{'nullmed':>9}{'nullp95':>9}{'p_dev':>7}{'z':>7}{'Hmean':>7}")
        for r in dv.itertuples(index=False):
            print(f"{r.match:<22}{r.opponent:<16}{r.venue:<3}{r.mgr:<4}{r.result:<7}{r.n_seg:>7}"
                  f"{r.cells:>6}{r.d_agg:>8.3f}{r.null_med:>9.3f}{r.null_p95:>9.3f}"
                  f"{r.p_dev:>7.3f}{r.z:>7.2f}{r.H_mean:>7.3f}")
        dv.to_csv(SCRATCH / "deviation.csv", index=False)

    if args.stage in ("metrica", "all"):
        run_metrica(gap_s=args.gap, T=args.T, grid=grid, cache=cache)


def metrica_sequence(name: str, *, censor: str, ball_mask: np.ndarray | None = None,
                     half_w: float | None = None) -> tuple[pd.DataFrame, float]:
    """1 Hz ball + possession sequence from Metrica truth, optionally camera-censored.

    Same possession rule as the broadcast pipeline (Viterbi-smoothed nearest player within
    ``POSSESSION_RADIUS_M``), so the only thing that changes between conditions is what the
    "camera" is allowed to see.

    Args:
        name: Metrica game, e.g. ``"Sample_Game_1"``.
        censor: ``"full"`` (all 22 players, all ball frames), ``"camera"`` (only players inside
            the tuned broadcast window), or ``"camera+drop"`` (camera plus a ball-availability
            mask copied from our real matches).
        ball_mask: boolean per-second availability pattern to tile over the game (``camera+drop``).
        half_w: fixed camera half-width in metres; tuned on this game when ``None``.

    Returns:
        ``(sequence, half_w)`` with the sequence in :func:`sequence_table` schema.
    """
    from synthesizer.imputation import (  # noqa: PLC0415
        METRICA, load_metrica_match, smooth_camera, tune_window, visibility_mask,
    )

    truth, ball, period, fps, n_home = load_metrica_match(
        METRICA / f"{name}_RawTrackingData_Home_Team.csv",
        METRICA / f"{name}_RawTrackingData_Away_Team.csv")
    cam = smooth_camera(ball, fps)
    if half_w is None:
        half_w, _ = tune_window(truth, cam)
    vis = (visibility_mask(truth, cam, half_w, PITCH_WID / 2) if censor != "full"
           else np.isfinite(truth[:, :, 0]))
    step = max(1, int(round(fps)))                      # decimate to 1 Hz
    idx = np.arange(0, truth.shape[0], step)
    sec = idx // step                                   # TRUE second index; holes must stay holes
    keep = np.isfinite(ball[idx, 0])
    if ball_mask is not None:
        keep &= ball_mask[sec % ball_mask.size]
    idx, sec = idx[keep], sec[keep]
    team = np.zeros(truth.shape[1], int)
    team[n_home:] = 1
    rows = []
    for k, fr in enumerate(idx):
        m = vis[fr] & np.isfinite(truth[fr, :, 0])
        for slot in np.flatnonzero(m):
            rows.append((k, int(slot), int(team[slot]), truth[fr, slot, 0], truth[fr, slot, 1]))
    players = pd.DataFrame(rows, columns=["frame", "track_id", "team", "pitch_x", "pitch_y"])
    bdf = pd.DataFrame({"frame": np.arange(idx.size), "x": ball[idx, 0], "y": ball[idx, 1]})
    poss = assign_possession(bdf, players, smooth=True)
    lab = dict(zip(poss["frame"].astype(int), poss["team"].astype(int))) if len(poss) else {}
    # attacking direction per period from the deep (5th-percentile x) defensive line
    dirs = {}
    for p in np.unique(period[idx]):
        sel = idx[period[idx] == p]
        p5 = [np.nanpercentile(truth[sel][:, team == t, 0], 5) for t in (0, 1)]
        dirs[int(p)] = {0: 1 if p5[0] < p5[1] else -1, 1: -1 if p5[0] < p5[1] else 1}
    tm = np.array([lab.get(k, -1) for k in range(idx.size)])
    seq = pd.DataFrame({
        "match": name, "chunk": [f"p{int(p)}" for p in period[idx]],
        "t": sec, "x": ball[idx, 0], "y": ball[idx, 1], "team": tm,
        "dir": [dirs[int(p)].get(int(t), 0) for p, t in zip(period[idx], tm)],
        "observed": True,
    })
    # rebase the 1 Hz clock per period so strings never bridge half-time, gaps preserved
    seq["t"] -= seq.groupby("chunk")["t"].transform("min")
    return seq, half_w


def run_metrica(*, gap_s: int, T: int, grid: tuple[int, int], cache: pd.DataFrame) -> None:
    """Full-pitch vs camera-censored entropy maps on Metrica truth (the degradation control)."""
    seconds = cache.groupby(["match", "chunk"])["t"].apply(
        lambda s: np.isin(np.arange(int(s.min()), int(s.max()) + 1), s.to_numpy()))
    ball_mask = np.concatenate(seconds.to_list())  # real broadcast ball availability, burst structure
    print(f"\n== METRICA degradation control (ball_mask from our 12 matches: {ball_mask.mean():.3f} "
          f"of seconds present, {ball_mask.size} s) ==")
    half_w = None
    maps: dict[tuple[str, str, int], MapResult] = {}
    for name in ("Sample_Game_1", "Sample_Game_2"):
        for cond in ("full", "camera", "camera+drop"):
            seq, hw = metrica_sequence(name, censor=cond,
                                       ball_mask=ball_mask if cond == "camera+drop" else None,
                                       half_w=half_w)
            half_w = half_w or hw
            st = possession_strings(seq, gap_s=gap_s)
            lens = np.array([s.x.size for s in st]) if st else np.zeros(1)
            print(f"  {name} [{cond:<12}] 1Hz s={len(seq)} labelled={int((seq['team'] >= 0).sum())} "
                  f"({(seq['team'] >= 0).mean():.3f}) strings={len(st)} med_len={np.median(lens):.0f} "
                  f"p90={np.percentile(lens, 90):.0f}")
            for ti in (0, 1):
                s, e = segments([x for x in st if x.team_id == ti], T=T, grid=grid)
                ent, cnt = entropy_map(s, e, grid=grid)
                maps[(name, cond, ti)] = MapResult(f"{name}:{cond}:{ti}", str(ti), name, ent, cnt,
                                                   int(s.size))
                print(f"      team{ti}: segs={s.size:5d} valid={int(np.isfinite(ent).sum())}/"
                      f"{grid[0] * grid[1]} H_mean={np.nanmean(ent):.3f}")
    print("\n  distances (mean |dH| over commonly valid cells):")
    for name in ("Sample_Game_1", "Sample_Game_2"):
        for ti in (0, 1):
            f = maps[(name, "full", ti)].entropy
            for cond in ("camera", "camera+drop"):
                d_self = map_distance(f, maps[(name, cond, ti)].entropy)[0]
                d_other = map_distance(f, maps[(name, "full", 1 - ti)].entropy)[0]
                print(f"    {name} team{ti}: d(full, {cond:<12}) = {d_self:.3f}   "
                      f"vs d(full team{ti}, full team{1 - ti}) = {d_other:.3f}   "
                      f"ratio={d_self / d_other:.2f}")
    keys = [k for k in maps if k[1] == "full"] + [k for k in maps if k[1] == "camera+drop"]
    ml = [maps[k] for k in keys]
    d = distance_matrix(ml, feature="entropy")
    lab = np.array([f"{k[0]}:{k[2]}" for k in keys])
    hits = []
    for i in range(4, 8):  # censored maps query the 4 full maps
        row = d[i, :4]
        hits.append(lab[int(np.argmin(np.where(np.isfinite(row), row, np.inf)))] == lab[i])
    print(f"  censored -> full retrieval of the same team-game: {sum(hits)}/4 (chance 1/4)")


def _jsonable(o: object) -> object:
    """JSON fallback for numpy scalars/arrays in the identity dump."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def _print_maps(cache: pd.DataFrame, ms: list[Match], *, gap_s: int, T: int,
                grid: tuple[int, int]) -> None:
    """Per-map summary table plus the printed Man Utd aggregate entropy grid."""
    maps = build_maps(cache, ms, gap_s=gap_s, T=T, grid=grid)
    s = map_summary(maps, grid)
    print(f"\n== MAPS  G={gap_s} T={T} grid={grid[0]}x{grid[1]} min_n={MIN_CELL_N} ==")
    print(f"{'map':<34}{'n_seg':>7}{'hit':>5}{'val':>5}{'tot':>5}{'medN':>6}{'Hmean':>7}{'Hmax':>7}")
    for r in s.itertuples(index=False):
        print(f"{r.map:<34}{r.n_seg:>7}{r.cells_hit:>5}{r.cells_valid:>5}{r.cells_total:>5}"
              f"{r.med_cell_n:>6.1f}{r.H_mean:>7.3f}{r.H_max:>7.3f}")
    for team in ("Man Utd",):
        agg = pooled_map(cache, ms, team, gap_s=gap_s, T=T, grid=grid)
        print(f"\n  aggregate {team}: n_seg={agg.n_segments} "
              f"valid={int(np.isfinite(agg.entropy).sum())}/{grid[0] * grid[1]} "
              f"H_mean={np.nanmean(agg.entropy):.3f}")
        e = agg.entropy.reshape(grid)
        n = agg.counts.reshape(grid)
        print("  entropy grid (rows = y bands top..bottom, cols = x bands, attacking ->):")
        for j in range(grid[1] - 1, -1, -1):
            print("   " + " ".join(f"{e[i, j]:5.2f}" for i in range(grid[0]))
                  + "   n= " + " ".join(f"{int(n[i, j]):4d}" for i in range(grid[0])))
    s.to_csv(SCRATCH / "map_summary.csv", index=False)


if __name__ == "__main__":
    main()
