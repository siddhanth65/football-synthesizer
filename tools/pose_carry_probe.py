"""PROBE: temporal pose carry-over for PLAYER projection -- faithfulness + end-to-end impact.

The project's binding constraint is player *geometry yield*, not calibration: 91.8% of
``brighton_manutd`` detection-frames solve a homography with a 0.21 m median keypoint reprojection
error, but only 37.4% yield usable player pitch coordinates -- on the rest the pose fits the
keypoints yet projects players off the pitch, and is correctly discarded. Every metric we ship lives
on that 37%.

:mod:`generator.pose_carry` applies the *proven* ball mechanism (:mod:`generator.ball_carry`, post-
link ball coverage 20.8 -> 38.8%) to players: borrow a known-good homography from a camera-continuous
neighbour within +-2 s and project the target frame's own players through it. **This is NOT the
disproven "mechanism 2"** of ``results/pl_probe/diagnosis/DIAGNOSIS.md`` (which reused the frame's
*own*, globally-wrong pose); the pose here comes from a *different, verified-good* frame.

Sections (``--section``):

``faithfulness``
    Pre-committed leave-one-out. Hide a good frame's own pose, borrow a neighbour's, compare the
    projected players against their known pitch positions. ACCEPTANCE: median <= 1.0 m, p90 <= 3.0 m.
    Reported twice: ``naive`` (nearest good neighbour -- the literal pre-committed test, identical in
    construction to the ball probe's LOO) and ``blocked`` (a stricter variant that forces the borrow
    gap to match the gap distribution production actually faces on failing frames, since good frames
    cluster and the naive nearest neighbour is optimistically close).

``yield``
    Player geometry yield before/after, per chunk and whole match.

``downstream``
    Fact-store metrics recomputed on the enriched positions (written to a SCRATCH parquet -- never
    ``outputs/facts/``): pass counts vs the Sofascore oracle, possession share vs oracle (the acid
    test for the trackability bias), post-link ball coverage.

``wc``
    No-regression guard on the one FIFA-validated metric: pooled de-biased defensive-line error vs
    FIFA PMSR across the three WC France matches, carry OFF vs ON.

Run: ``python -m tools.pose_carry_probe --section all``. CPU-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import Match, get, matches
from fingerprint.phase_metrics import ball_phase_by_frame
from fingerprint.roles import assign_roles
from fingerprint.structural_metrics import (
    resolve_attack_directions,
    resolve_attack_directions_from_ball,
)
from fingerprint.theory_metrics import complete_directions
from generator.ball import assign_possession
from generator.ball_carry import (
    DEFAULT_JACCARD_CUT,
    DEFAULT_MAX_REPROJ_PX,
    DEFAULT_MIN_PTS,
    DEFAULT_WINDOW_FRAMES,
    PLAYER_ROLES,
    _nearest_sources,
    camera_segment_ids,
    fit_frame_homographies,
)
from generator.impute import line_estimates
from generator.pose_carry import PoseCarryStats, carry_player_poses, good_geometry_frames
from generator.pose_carry import project_players as _project
from generator.postprocess import MIN_ONPITCH_PLAYERS
from tools.line_c6 import PHASE_MAP, PHASE_ORDER

CALIB_MAX_M = 1.0  # the report.facts per-frame calibration gate
ACC_MEDIAN_M = 1.0  # pre-committed faithfulness acceptance
ACC_P90_M = 3.0
SCRATCH = Path("results/pose_carry_probe_scratch")
# Sofascore oracle, cached (tools/oracle.py): completed passes, brighton_manutd.
ORACLE_PASSES = {"Man Utd": 446, "Brighton": 407}
ORACLE_POSS_TEAM0 = 52.0  # Man Utd possession share (%)
BASELINE_PASSES = {"Man Utd": 213, "Brighton": 198}  # current CV, for the reported delta


# --------------------------------------------------------------------------------------------- #
# faithfulness
# --------------------------------------------------------------------------------------------- #
def _chunk_loo(
    dense: pd.DataFrame, *, seed: int, blocked: bool, window: int = DEFAULT_WINDOW_FRAMES,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Leave-one-out faithfulness for one chunk.

    For every frame that *does* have good geometry we drop its own pose from the source pool and
    project its players through a borrowed neighbour pose, then compare to their known pitch
    positions.

    Args:
        dense: one chunk of the aligned positions table.
        seed: RNG seed for the blocked variant's gap sampling.
        blocked: if True, also block every good frame closer than a gap drawn from the *real* carry-
            gap distribution, so the borrow distance matches what production faces on failing frames.
        window: carry window (native frames each side).

    Returns:
        ``(errors_m, real_gaps, n_offpitch_gated)`` -- ``errors_m`` excludes off-pitch-gated
        projections (those are never shipped).
    """
    players = dense[dense["role"].isin(PLAYER_ROLES)]
    good = good_geometry_frames(dense)
    homs = fit_frame_homographies(players, min_pts=DEFAULT_MIN_PTS,
                                  max_reproj_px=DEFAULT_MAX_REPROJ_PX)
    src = np.array(sorted(set(homs) & set(good.tolist())), dtype=int)
    if not src.size:
        return np.array([]), np.array([]), 0
    seg = camera_segment_ids(dense, jaccard_cut=DEFAULT_JACCARD_CUT)
    det = sorted(int(f) for f in players["frame"].unique())
    gset = set(good.tolist())

    gaps = []  # the gap distribution production really faces (targets = frames WITHOUT geometry)
    for f in (x for x in det if x not in gset):
        lft, rgt = _nearest_sources(f, src, seg, window=window)
        cand = [abs(f - x) for x in (lft, rgt) if x is not None]
        if cand:
            gaps.append(min(cand))
    gap_arr = np.array(gaps) if gaps else np.array([1])

    rng = np.random.default_rng(seed)
    rows = {int(f): g for f, g in players.groupby("frame")}
    errs: list[np.ndarray] = []
    n_gated = 0
    for f in good.tolist():
        block = int(rng.choice(gap_arr)) if blocked else 1
        pool = src[np.abs(src - f) >= block]
        lft, rgt = _nearest_sources(f, pool, seg, window=window)
        if lft is None and rgt is None:
            continue
        v = rows[f].dropna(subset=["pitch_x", "pitch_y"])
        if v.empty:
            continue
        img = v[["image_x", "image_y"]].to_numpy(float)
        true = v[["pitch_x", "pitch_y"]].to_numpy(float)
        if lft is not None and rgt is not None:
            w = (f - lft) / (rgt - lft)
            proj = (1 - w) * _project(homs[lft], img) + w * _project(homs[rgt], img)
        else:
            proj = _project(homs[lft if rgt is None else rgt], img)
        e = np.hypot(proj[:, 0] - true[:, 0], proj[:, 1] - true[:, 1])
        n_gated += int(np.isnan(e).sum())
        errs.append(e)
    e = np.concatenate(errs) if errs else np.array([])
    return e[~np.isnan(e)], gap_arr, n_gated


def _fmt_faith(tag: str, err: np.ndarray, n_gated: int) -> str:
    """One faithfulness row."""
    if not err.size:
        return f"{tag:<22} (no samples)"
    return ("%-22s n=%6d gated=%4d  median %5.2f  p90 %5.2f  max %6.1f  "
            "<=1m %3.0f%%  <=2m %3.0f%%  <=5m %3.0f%%"
            % (tag, err.size, n_gated, np.median(err), np.percentile(err, 90), err.max(),
               100 * (err <= 1).mean(), 100 * (err <= 2).mean(), 100 * (err <= 5).mean()))


def section_faithfulness(match_id: str, *, seeds: int = 5) -> dict:
    """Pre-committed LOO faithfulness (naive + gap-matched blocked)."""
    m = get(match_id)
    al = m.load_aligned()
    chunks = sorted(al["chunk"].unique())
    print(f"=== FAITHFULNESS (leave-one-out) -- {match_id} ===")
    print(f"acceptance: median <= {ACC_MEDIAN_M} m AND p90 <= {ACC_P90_M} m\n")

    naive, ng = [], 0
    for ck in chunks:
        e, _, g = _chunk_loo(al[al["chunk"] == ck], seed=0, blocked=False)
        naive.append(e)
        ng += g
    naive_e = np.concatenate(naive) if naive else np.array([])
    print("-- naive LOO (nearest good neighbour; the literal pre-committed test) --")
    print(_fmt_faith("naive POOLED", naive_e, ng))

    print("\n-- gap-matched blocked LOO (stricter: borrow gap matched to production) --")
    blocked_meds, blocked_p90s = [], []
    blocked_e = np.array([])
    for s in range(seeds):
        parts, g = [], 0
        for ck in chunks:
            e, _, gg = _chunk_loo(al[al["chunk"] == ck], seed=s, blocked=True)
            parts.append(e)
            g += gg
        e = np.concatenate(parts) if parts else np.array([])
        blocked_meds.append(float(np.median(e)))
        blocked_p90s.append(float(np.percentile(e, 90)))
        print(_fmt_faith(f"blocked seed {s}", e, g))
        if s == 0:
            blocked_e = e
    print(f"\n  blocked across {seeds} seeds: median {min(blocked_meds):.2f}-{max(blocked_meds):.2f} m"
          f"   p90 {min(blocked_p90s):.2f}-{max(blocked_p90s):.2f} m")

    verdict_naive = np.median(naive_e) <= ACC_MEDIAN_M and np.percentile(naive_e, 90) <= ACC_P90_M
    print(f"\n  PRE-COMMITTED (naive) VERDICT: {'PASS' if verdict_naive else 'FAIL'}")
    print(f"  stricter (blocked) p90 sits at the {ACC_P90_M} m bar -- reported, not tuned.")
    return {
        "naive": {"median": float(np.median(naive_e)), "p90": float(np.percentile(naive_e, 90)),
                  "n": int(naive_e.size), "pass": bool(verdict_naive)},
        "blocked": {"median_range": [min(blocked_meds), max(blocked_meds)],
                    "p90_range": [min(blocked_p90s), max(blocked_p90s)],
                    "n": int(blocked_e.size)},
    }


# --------------------------------------------------------------------------------------------- #
# enrichment + yield
# --------------------------------------------------------------------------------------------- #
def enrich(m: Match, *, window_s: float = 2.0) -> tuple[pd.DataFrame, dict[str, PoseCarryStats]]:
    """Run pose carry-over over every chunk of a match; returns the enriched table + per-chunk stats.

    The carry window is specified in **seconds** and converted per chunk through the registry's true
    fps (the WC corpus mixes 25 and 59.94 fps).
    """
    al = m.load_aligned()
    out, stats = [], {}
    for ck in sorted(al["chunk"].unique()):
        dense = al[al["chunk"] == ck].copy()
        fps = m.chunk_fps(ck)
        enr, st = carry_player_poses(dense, window=int(round(window_s * fps)))
        out.append(enr)
        stats[ck] = st
    return pd.concat(out, ignore_index=True), stats


def section_yield(match_id: str) -> dict:
    """Player geometry yield, before vs after carry-over."""
    m = get(match_id)
    _, stats = enrich(m)
    print(f"=== PLAYER GEOMETRY YIELD -- {match_id} ===\n")
    print(f"{'chunk':<15}{'det':>7}{'good':>7}{'yield':>8}{'carried':>9}{'interp':>8}"
          f"{'gated':>7}{'nosrc':>7}{'yield ON':>10}{'delta':>8}")
    tot_det = tot_before = tot_after = 0
    for ck, s in stats.items():
        print(f"{ck:<15}{s.det_frames:>7}{s.good_before:>7}{s.yield_before:>7.1%}"
              f"{s.carried:>9}{s.interpolated:>8}{s.frame_gate_dropped:>7}{s.no_source:>7}"
              f"{s.yield_after:>9.1%}{s.yield_after - s.yield_before:>+7.1%}")
        tot_det += s.det_frames
        tot_before += s.good_before
        tot_after += s.good_after
    yb, ya = tot_before / tot_det, tot_after / tot_det
    print(f"{'MATCH':<15}{tot_det:>7}{tot_before:>7}{yb:>7.1%}{'':>9}{'':>8}{'':>7}{'':>7}"
          f"{ya:>9.1%}{ya - yb:>+7.1%}")
    return {"det_frames": tot_det, "yield_off": yb, "yield_on": ya}


def section_ceiling(match_id: str) -> dict:
    """Why the yield lever is capped: what do the geometry-less frames actually contain?

    A frame needs ``MIN_ONPITCH_PLAYERS`` (8) players on the pitch to survive
    ``reject_implausible_frames``. No borrowed pose can invent players, so a frame with fewer than 8
    *detections* is unrecoverable by ANY pose mechanism. This measures that hard ceiling, and the
    ``n>=8 & image-x spread >= 800 px`` live-wide-play proxy from ``results/pl_probe/diagnosis``
    (both calibration-independent, so this is not circular).
    """
    m = get(match_id)
    al = m.load_aligned()
    print(f"=== ADDRESSABLE-POPULATION CEILING -- {match_id} ===\n")
    print(f"{'chunk':<15}{'targets':>9}{'med det':>9}{'>=8 det':>9}{'addressable':>13}"
          f"{'live wide':>11}{'yield cap':>11}")
    tot = {"det": 0, "tgt": 0, "addr": 0, "good": 0}
    for ck in sorted(al["chunk"].unique()):
        dense = al[al["chunk"] == ck]
        players = dense[dense["role"].isin(PLAYER_ROLES)]
        good = set(good_geometry_frames(dense).tolist())
        npl = players.groupby("frame")["track_id"].nunique()
        spread = players.groupby("frame")["image_x"].apply(
            lambda s: float(np.ptp(s.dropna())) if s.notna().any() else 0.0)
        tgts = [int(f) for f in npl.index if int(f) not in good]
        t_npl = npl.loc[tgts]
        addr = int((t_npl >= MIN_ONPITCH_PLAYERS).sum())
        wide = int(((npl.loc[tgts] >= 8) & (spread.loc[tgts] >= 800)).sum())
        n_det = int(npl.size)
        print(f"{ck:<15}{len(tgts):>9}{t_npl.median():>9.0f}{addr:>9}"
              f"{addr / max(1, len(tgts)):>12.1%}{wide / max(1, len(tgts)):>10.1%}"
              f"{addr / n_det:>+10.1%}")
        tot["det"] += n_det
        tot["tgt"] += len(tgts)
        tot["addr"] += addr
        tot["good"] += len(good)
    cap = tot["addr"] / tot["det"]
    print(f"\n{'MATCH':<15}{tot['tgt']:>9}{'':>9}{tot['addr']:>9}"
          f"{tot['addr'] / tot['tgt']:>12.1%}{'':>11}{cap:>+10.1%}")
    print("\n  Geometry-less frames are overwhelmingly close-ups / replays / tight shots:")
    print(f"  only {tot['addr']}/{tot['tgt']} ({tot['addr'] / tot['tgt']:.1%}) carry the >= "
          f"{MIN_ONPITCH_PLAYERS} detections a plausible frame needs.")
    print(f"  HARD CEILING on any pose-borrowing mechanism: yield "
          f"{tot['good'] / tot['det']:.1%} -> at most {(tot['good'] + tot['addr']) / tot['det']:.1%}"
          f" (+{cap:.1%} pp), even with a PERFECT pose on every addressable frame.")
    return {"targets": tot["tgt"], "addressable": tot["addr"], "yield_cap_pp": cap}


# --------------------------------------------------------------------------------------------- #
# downstream (facts on the enriched positions -- SCRATCH only)
# --------------------------------------------------------------------------------------------- #
def _possession_share(m: Match, aligned: pd.DataFrame) -> tuple[float, int]:
    """Team0 possession share (%) on the exact ``report.facts`` path, plus the sample count."""
    n0 = n1 = 0
    for ck, path in m.ball_chunks():
        ball = pd.read_parquet(path)
        pos = aligned[(aligned["chunk"] == ck) & (aligned["calib_error_m"] <= CALIB_MAX_M)].dropna(
            subset=["pitch_x"])
        if pos.empty or ball.empty:
            continue
        poss = assign_possession(ball, pos, smooth=True)
        if poss.empty:
            continue
        n0 += int((poss["team"] == 0).sum())
        n1 += int((poss["team"] == 1).sum())
    tot = n0 + n1
    return (100.0 * n0 / tot if tot else float("nan")), tot


def _ball_coverage(m: Match, aligned: pd.DataFrame) -> float:
    """Post-``link_ball`` usable ball-track coverage: linked ball frames / dense detection frames.

    This is the ONLY ball-coverage number that counts (never pre-link detections).
    """
    linked = dense = 0
    for ck, path in m.ball_chunks():
        ball = pd.read_parquet(path)
        d = aligned[aligned["chunk"] == ck]
        linked += int(ball.dropna(subset=["x", "y"])["frame"].nunique()) if not ball.empty else 0
        dense += int(d["frame"].nunique())
    return linked / dense if dense else float("nan")


def section_downstream(match_id: str) -> dict:
    """Recompute the fact store on enriched positions; compare passes/possession vs the oracle."""
    import report.facts as facts_mod  # noqa: PLC0415

    m = get(match_id)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    enriched, _ = enrich(m)
    scratch_parquet = SCRATCH / f"{match_id}_enriched.parquet"
    enriched.drop(columns=["pose_source"]).to_parquet(scratch_parquet, index=False)

    print(f"=== DOWNSTREAM IMPACT -- {match_id} (scratch: {scratch_parquet}) ===\n")
    results = {}
    real_get = facts_mod.get
    for tag, path in (("OFF", m.aligned), ("ON", scratch_parquet)):
        shim = dataclasses.replace(m, aligned=Path(path))
        facts_mod.get = lambda _mid, _s=shim, **_kw: _s  # noqa: B023 -- deliberate per-iteration bind
        try:
            f = facts_mod.build_facts(match_id)
        finally:
            facts_mod.get = real_get
        aligned = pd.read_parquet(path)
        poss, n_poss = _possession_share(m, aligned)
        results[tag] = {
            "passes": {t: int(v.get("n_passes", 0)) for t, v in f["cv"]["passing"].items()},
            "possession_team0": poss, "n_poss_samples": n_poss,
            "ball_coverage": _ball_coverage(m, aligned),
            "line_debiased": {t: v["def_line_debiased_m"]
                              for t, v in f["cv"].get("line_height", {}).items()},
        }
        (SCRATCH / f"{match_id}_facts_carry_{tag}.json").write_text(
            json.dumps(f, indent=1), encoding="utf-8")

    off, on = results["OFF"], results["ON"]
    print(f"{'metric':<28}{'OFF':>12}{'ON':>12}{'oracle':>12}{'delta':>12}")
    for team, orc in ORACLE_PASSES.items():
        o, n = off["passes"].get(team, 0), on["passes"].get(team, 0)
        print(f"{'passes ' + team:<28}{o:>12}{n:>12}{orc:>12}{n - o:>+12}")
        print(f"{'  pass-recall proxy':<28}{o / orc:>11.1%}{n / orc:>11.1%}{'-':>12}"
              f"{(n - o) / orc:>+11.1%}")
    print(f"{'possession team0 (%)':<28}{off['possession_team0']:>12.1f}"
          f"{on['possession_team0']:>12.1f}{ORACLE_POSS_TEAM0:>12.1f}"
          f"{on['possession_team0'] - off['possession_team0']:>+12.1f}")
    print(f"{'  |error| vs oracle (pp)':<28}{abs(off['possession_team0'] - ORACLE_POSS_TEAM0):>12.1f}"
          f"{abs(on['possession_team0'] - ORACLE_POSS_TEAM0):>12.1f}{'-':>12}"
          f"{abs(on['possession_team0'] - ORACLE_POSS_TEAM0) - abs(off['possession_team0'] - ORACLE_POSS_TEAM0):>+12.1f}")
    print(f"{'possession samples':<28}{off['n_poss_samples']:>12}{on['n_poss_samples']:>12}"
          f"{'-':>12}{on['n_poss_samples'] - off['n_poss_samples']:>+12}")
    print(f"{'post-link ball coverage':<28}{off['ball_coverage']:>11.1%}{on['ball_coverage']:>11.1%}"
          f"{'-':>12}{on['ball_coverage'] - off['ball_coverage']:>+11.1%}")
    return results


# --------------------------------------------------------------------------------------------- #
# WC no-regression (the one FIFA-validated metric)
# --------------------------------------------------------------------------------------------- #
def _phase_frames(m: Match, aligned: pd.DataFrame) -> pd.DataFrame:
    """France (team 0) per-frame ball phase -- the ``tools.line_c6`` path, on a supplied table."""
    parts = []
    for ck, path in m.ball_chunks():
        ball = pd.read_parquet(path)
        pos = aligned[(aligned["chunk"] == ck) & (aligned["calib_error_m"] <= CALIB_MAX_M)].dropna(
            subset=["pitch_x"])
        if pos.empty or ball.empty:
            continue
        dirs = complete_directions(resolve_attack_directions_from_ball(pos, ball)
                                   or resolve_attack_directions(pos))
        if len(dirs) < 2:
            continue
        poss = assign_possession(ball, pos, smooth=True)
        ph = ball_phase_by_frame(ball, poss, pos)
        ph = ph[ph["team"] == 0][["frame", "phase"]].copy()
        ph["chunk"] = ck
        parts.append(ph)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["frame", "phase", "chunk"])


def _line_errors(m: Match, aligned: pd.DataFrame) -> list[float]:
    """Per-phase |de-biased line - FIFA| for one France match."""
    lh = (m.load_pmsr() or {}).get("line_height")
    roster = json.loads(Path("data/france_roster.json").read_text())["matches"][m.id]
    fidx = 0 if roster["france_is_home"] else 1
    le = line_estimates(aligned, assign_roles(aligned))
    le = le[le["team"] == 0]
    ph = _phase_frames(m, aligned)
    if ph.empty or not lh:
        return []
    j = ph.merge(le, on=["chunk", "frame"], how="inner")
    errs = []
    for phase in PHASE_ORDER:
        grp = j[j["phase"] == phase]
        if grp.empty or phase not in PHASE_MAP:
            continue
        sec, key = PHASE_MAP[phase]
        v = lh.get(sec, {}).get(key)
        fifa = v[fidx] if isinstance(v, list) and v[fidx] is not None else None
        if fifa is None:
            continue
        errs.append(abs(float(grp["line_debiased"].mean()) - float(fifa)))
    return errs


def section_wc() -> dict:
    """No-regression guard: pooled de-biased line error vs FIFA, carry OFF vs ON."""
    print("=== WC NO-REGRESSION -- de-biased defensive line vs FIFA PMSR ===\n")
    print(f"{'match':<18}{'n phases':>9}{'OFF mean':>10}{'ON mean':>10}{'delta':>9}")
    pooled = {"OFF": [], "ON": []}
    for m in matches(processed_only=True):
        if "France" not in m.teams or m.pmsr is None:
            continue
        off = _line_errors(m, m.load_aligned())
        enriched, _ = enrich(m)
        on = _line_errors(m, enriched.drop(columns=["pose_source"]))
        if not off:
            continue
        pooled["OFF"] += off
        pooled["ON"] += on
        print(f"{m.id:<18}{len(off):>9}{np.mean(off):>10.1f}{np.mean(on):>10.1f}"
              f"{np.mean(on) - np.mean(off):>+9.1f}")
    po, pn = float(np.mean(pooled["OFF"])), float(np.mean(pooled["ON"]))
    print(f"\n{'POOLED mean |err|':<18}{len(pooled['OFF']):>9}{po:>10.1f}{pn:>10.1f}{pn - po:>+9.1f} m")
    print(f"  guard: ON must NOT worsen vs OFF -> {'PASS' if pn <= po + 0.1 else 'FAIL'}")
    return {"pooled_off_m": po, "pooled_on_m": pn, "pass": bool(pn <= po + 0.1)}


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description="Pose carry-over probe (players).")
    ap.add_argument("--section", default="all",
                    choices=["all", "faithfulness", "yield", "ceiling", "downstream", "wc"])
    ap.add_argument("--match", default="brighton_manutd")
    a = ap.parse_args()
    if a.section in ("all", "faithfulness"):
        section_faithfulness(a.match)
        print()
    if a.section in ("all", "yield"):
        section_yield(a.match)
        print()
    if a.section in ("all", "ceiling"):
        section_ceiling(a.match)
        print()
    if a.section in ("all", "downstream"):
        section_downstream(a.match)
        print()
    if a.section in ("all", "wc"):
        section_wc()


if __name__ == "__main__":
    main()
