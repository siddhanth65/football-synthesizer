"""W1 feasibility checks for RQ-D: is the transition/pressing metric family answerable at all?

Two measurements, both CPU-only, both reusing shipped machinery (nothing new is modelled here):

**W1-A -- turnover-window census.** On every processed EPL match, count the turnovers that survive
three simultaneous requirements: the ball is tracked at the loss moment, it stays tracked through the
following 5 s, and at least N of the *pressing* (= ball-losing) team's players carry trusted geometry
across the whole window. Turnover = the shipped possession-state change, i.e.
:func:`fingerprint.transitions.detect_turnovers` over
``generator.ball.assign_possession(..., smooth=True)`` (Viterbi team smoother, 2.0 m possession
radius, 1.5 m switch penalty), run per chunk on the **post-**:func:`generator.ball.link_ball` track --
never on raw detections. No new turnover definition is invented.

**W1-B -- defensive-action numerator check.** PPDA's denominator is tackles + interceptions + fouls.
Compare what the E2E-Spot action-spotting probe actually detects against the cached Sofascore oracle
(``outputs/oracle/sofascore/team_stats_<id>.parquet``, whole-match home+away), the same ratio-style
validation ``tools/bas_validate.py`` used for passes. The shipped probe summary used a 30 s NMS
window for every class, which is far longer than the gap between two fouls, so the peak picker is
re-run from the cached score npz over a small (threshold, NMS) sweep.

Run (CPU)::

    python -m tools.w1_feasibility
    python -m tools.w1_feasibility --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import Match, get
from fingerprint.transitions import detect_turnovers
from generator.ball import assign_possession
from tools.action_spot_probe import (
    FRAME_FPS,
    SOFASCORE_MATCH_ID,
    class_index,
    find_peaks,
    load_half_timeline,
)

# --- W1-A parameters -----------------------------------------------------------------------------
CALIB_ERROR_MAX_M = 1.0    # "trusted geometry" -- same gate tools/event_coverage.py uses
WINDOW_S = 5.0             # the post-turnover window the transition family is defined on
DEFENDER_LEVELS = (3, 5, 7)
LENIENT_COV = 0.80         # relaxed ball criterion: >=80% of window slots tracked AND the last slot

# --- W1-B parameters -----------------------------------------------------------------------------
ORACLE_DIR = Path("outputs/oracle/sofascore")
# E2E-Spot class -> Sofascore key. Only two of the 17 SoccerNet-v2 classes are defensive actions;
# tackles and interceptions have no class at all (that absence is the measurement).
DEFENSIVE_CLASS_TRUTH = {"Foul": "fouls", "Clearance": "totalClearance"}
PPDA_TRUTH_KEYS = ("totalTackle", "interceptionWon", "fouls")
SWEEP = ((0.30, 30.0), (0.30, 10.0), (0.30, 5.0), (0.50, 5.0))  # (threshold, NMS seconds)

REPORT_PATH = Path("results/W1_FEASIBILITY_CHECKS.md")
MATCH_IDS = tuple(SOFASCORE_MATCH_ID)

# Frozen reading of the 2026-07-26 run. The tables above it are recomputed on every run; this block
# is the interpretation written against those numbers and is dated so a later run can contradict it.
VERDICT_NOTES = """
## Verdicts (reading of the 2026-07-26 run)

### W1-A: FAIL at the declared criterion, MARGINAL only at its weakest setting

Against the pre-declared thresholds (>=30/match = per-match viable; 10-30 = pooled only; <10 = drop
the family), using the declared criterion (ball tracked for the whole 5 s **and** N pressing-team
players with trusted geometry in **every** slot of the window):

| N | per-match median | range | pooled | declared verdict |
|---|---|---|---|---|
| 3 | 12 | 0-28 | 139 | MARGINAL -- pooled-only, per-match claims die |
| 5 | 5 | 0-14 | 71 | FAIL |
| 7 | 1 | 0-7 | 21 | FAIL |

N=3 is not a defensible pressing measurement anyway (a counterpress described by three visible
players is a description of three players), so the operative verdict at any football-meaningful N is
**FAIL**: the transition family is not per-match estimable, and at N>=5 it is not even pooled-estimable
at the strict criterion (71 windows over 12 matches, one match contributing 0).

**The binding constraint is geometry, not the ball.** The funnel, pooled over 12 matches:

* 2546 possession-state changes detected (ball tracked at the loss moment by construction),
* 886 (34.8%) keep the ball tracked across the full 5 s,
* 139 (5.5%) additionally have >=3 pressing-team players calibrated in every slot,
* 21 (0.8%) have >=7.

Loosening the ball criterion changes nothing (the `lN` columns equal the `sN` columns: the 15
extra windows admitted at >=80% ball coverage all fail the defender gate anyway). On
brighton_manutd, 114 of the 130 ball-complete windows have **at least one slot with zero calibrated
pressing-team players** -- mean per-window geometry coverage is 0.51 across the corpus. Homography is
available on ~37% of frames in a chunk and the ~11 players it does place are split across two teams,
so demanding an uninterrupted 5 s view of one team's press is the expensive requirement.

**Sensitivity (why this is structural, not a knife-edge at N=3).** Replacing "in every slot" with
"median over the window's slots" lifts N>=3 to a median of 46/match (PASS) and N>=5 to 28/match
(MARGINAL), but N>=7 still fails at 9/match. So a *gappy* 5 s response function is measurable for a
half-team; a *continuous* one is not, at any N. Any transition metric that survives here has to be
defined on intermittent observation with imputation filling the gaps -- which is exactly the RQ-D
ablation, so this is a live design constraint rather than a dead end, but it must be designed for up
front, not assumed away.

### W1-B: PPDA as literally defined is NOT broadcast-estimable

| component | share of PPDA denominator | our detector | ratio vs Sofascore |
|---|---|---|---|
| fouls | 260/989 = 26% | E2E-Spot `Foul` | **1.069x** (shipped op) / 1.108x (5 s NMS) |
| tackles | 473/989 = 48% | none -- no SoccerNet-v2 class | **0.000x** |
| interceptions | 256/989 = 26% | none -- no SoccerNet-v2 class | **0.000x** |
| whole denominator | 989 | fouls only | **0.291x** |

The E2E-Spot 17 classes were checked, not assumed: `results/action_spotting_probe/*/summary.json`
contains exactly `Ball out of play, Clearance, Corner, Direct free-kick, Foul, Goal, Indirect
free-kick, Kick-off, Offside, Penalty, Red card, Shots off/on target, Substitution, Throw-in, Yellow
card, Yellow->red card`. Tackle and interception are absent from the label space, so no threshold
sweep can recover them; the 0.000x is structural, not tuning.

The one component we can see is genuinely good: fouls land at 1.069x pooled, inside the 0.97-1.09x
band that passed for passes, stable across 12 matches (per-match 0.955x-1.250x) and across the
(0.30, 30 s) -> (0.30, 5 s) NMS sweep. `Clearance` is 0.265x and should not be used. But 74% of the
PPDA denominator is invisible, so **PPDA as Opta defines it is out**, and it is out for a harder
reason than the 1-4% player-attribution regime: those events are not in the detector's vocabulary at
all.

Two further caveats on the foul number, both fatal to a naive PPDA:

1. It is a **match total with no team attribution**. PPDA is per-team; a spotted foul peak carries no
   "which team committed it" label, and nothing in the pipeline currently assigns one.
2. It is validated against a whole-match count, not against event timestamps -- count agreement is
   necessary, not sufficient, for using the events as a denominator located in space.

**The redesign the check was meant to price.** Our possession-change events run 2546 over 12 matches
= 212/match, against Sofascore `ballRecovery` 1256 = 105/match: **2.03x pooled**, per-match spread
0.96x-2.75x. So possession-change events are plentiful but over-detected by roughly 2x with a
2.9x spread across matches -- usable as a *rate* construct after calibration, not as a drop-in count.
That is the defensible finding the plan anticipated: "PPDA is not broadcast-estimable; here is a
possession-change-based pressing rate that is, with its calibration factor measured."

## What dies on this evidence

1. **Per-match transition metrics (5 s recovery rate, counterpress-vs-retreat per match).** Dead at
   the declared threshold. 12 usable windows/match at N=3, 5 at N=5, one match with 0. Cannot support
   a per-match claim, and therefore cannot support the opponent-controlled home/away pair comparison
   that made the corpus interesting.
2. **PPDA as literally defined.** Dead. 74% of the denominator has no detector.
3. **Clearance-based defensive-action counting.** Dead at 0.265x.
4. **Any transition metric requiring continuous 5 s observation of >=7 pressers.** Dead at every
   criterion tested (21 windows pooled, median 1/match).

## What survives

1. **Pooled transition statistics at N=3-5** (139 / 71 windows pooled over 12 matches) -- enough for a
   corpus-level statement with a wide interval, not for a per-team profile.
2. **Gap-tolerant transition metrics** (median-over-window observation): 46/match at N>=3, 28/match at
   N>=5. This is the version RQ-D would have to measure, and it makes the imputation arm load-bearing
   rather than decorative -- the metric family is only rescued if imputation fills those slots.
3. **Foul detection at 1.069x** -- a validated event class, once team attribution is solved.
4. **A possession-change-based pressing rate**, with the 2.03x over-detection factor now measured.
5. **The structure metrics** (line height, compactness), which need geometry at an instant rather than
   continuously across 5 s and are untouched by this census.
"""


# === W1-A ========================================================================================
def window_stats(frame: int, lost_team: int, ball_frames: set[int],
                 def_counts: dict[tuple[int, int], int], *, stride: int, n_slot: int) -> dict:
    """Per-turnover window diagnostics (pure; unit-checked by :func:`selftest`).

    Args:
        frame: turnover frame (first frame of the winning team's spell).
        lost_team: the team that lost the ball = the pressing team in a counterpress.
        ball_frames: frames present in the post-link ball track for this chunk.
        def_counts: ``(frame, team) -> number of trusted players`` for this chunk.
        stride: sampling stride of the ball/position grid, in native frames.
        n_slot: number of window slots after the turnover (window = ``n_slot * stride`` frames).

    Returns:
        ``frame, lost_team, ball_full, ball_cov, ball_end, min_def, med_def, geom_cov``.
        ``min_def`` is the declared criterion (N visible in *every* slot); ``med_def`` and
        ``geom_cov`` are diagnostics that say whether a failure is knife-edge or structural.
    """
    slots = [frame + k * stride for k in range(n_slot + 1)]
    present = sum(s in ball_frames for s in slots)
    seen = [def_counts.get((s, lost_team), 0) for s in slots]
    return {
        "frame": frame,
        "lost_team": lost_team,
        "ball_full": present == len(slots),
        "ball_cov": present / len(slots),
        "ball_end": slots[-1] in ball_frames,
        "min_def": min(seen),
        "med_def": float(np.median(seen)),
        "geom_cov": sum(1 for v in seen if v) / len(seen),
    }


def chunk_inputs(pos_chunk: pd.DataFrame,
                 ball: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Shared chunk preparation: trusted-geometry players, possession track, turnovers.

    This is the single definition of "what a turnover is" for every W1 census (the 5 s one below
    and the 2 s re-run in :mod:`tools.w1b_window_sampling`), so the two cannot silently drift apart.

    Args:
        pos_chunk: this chunk's rows of ``match_aligned.parquet`` (all roles, unfiltered).
        ball: this chunk's post-``link_ball`` track (``frame, x, y, observed``) in pitch metres.

    Returns:
        ``(players, possession, turnovers)`` or ``None`` when the chunk yields no turnover.
    """
    if ball.empty:
        return None
    ok = (pos_chunk["calib_error_m"] <= CALIB_ERROR_MAX_M) & pos_chunk["pitch_x"].notna()
    calib = pos_chunk[ok]
    players = calib[calib["role"].isin(["player", "goalkeeper"])]
    if players.empty:
        return None
    poss = assign_possession(ball, players, smooth=True)
    turnovers = detect_turnovers(poss, ball)
    return None if turnovers.empty else (players, poss, turnovers)


def chunk_census(pos_chunk: pd.DataFrame, ball: pd.DataFrame, fps: float) -> list[dict]:
    """Turnover-window diagnostics for one chunk (empty list when the chunk is unusable).

    Args:
        pos_chunk: this chunk's rows of ``match_aligned.parquet`` (all roles, unfiltered).
        ball: this chunk's post-``link_ball`` track (``frame, x, y, observed``) in pitch metres.
        fps: native source fps of the chunk (the 5 s window is converted through it).

    Returns:
        One :func:`window_stats` dict per detected turnover.
    """
    prepared = chunk_inputs(pos_chunk, ball)
    if prepared is None:
        return []
    players, _poss, turnovers = prepared

    frames = np.sort(ball["frame"].astype(int).unique())
    stride = max(int(np.median(np.diff(frames))) if frames.size > 1 else 1, 1)
    n_slot = max(int(round(WINDOW_S * fps)) // stride, 1)
    ball_frames = {int(f) for f in frames}
    def_counts = {(int(f), int(t)): int(v)
                  for (f, t), v in players.groupby(["frame", "team"]).size().items()}
    return [window_stats(int(tv.frame), int(tv.lost_team), ball_frames, def_counts,
                         stride=stride, n_slot=n_slot)
            for tv in turnovers.itertuples(index=False)]


def match_census(m: Match) -> dict:
    """Aggregate the turnover-window census over every ball chunk of one match.

    Args:
        m: registered match with an aligned parquet and linked ball chunks on disk.

    Returns:
        Match-level counts: raw turnovers, ball-complete turnovers, and usable windows per
        defender level under the strict and lenient ball criteria.
    """
    pos = m.load_aligned()
    rows: list[dict] = []
    for chunk_key, ball_path in m.ball_chunks():
        pos_chunk = pos[pos["chunk"] == chunk_key]
        if pos_chunk.empty:
            continue
        rows += chunk_census(pos_chunk, pd.read_parquet(ball_path), m.chunk_fps(chunk_key))

    cols = ["frame", "lost_team", "ball_full", "ball_cov", "ball_end", "min_def", "med_def",
            "geom_cov"]
    df = pd.DataFrame(rows, columns=cols)
    strict = df[df["ball_full"]] if not df.empty else df
    lenient = df[(df["ball_cov"] >= LENIENT_COV) & df["ball_end"]] if not df.empty else df
    out = {
        "match_id": m.id,
        "turnovers": int(len(df)),
        "ball_full": int(len(strict)),
        "ball_lenient": int(len(lenient)),
        "mean_ball_cov": float(df["ball_cov"].mean()) if not df.empty else float("nan"),
        "mean_geom_cov": float(df["geom_cov"].mean()) if not df.empty else float("nan"),
    }
    for n in DEFENDER_LEVELS:
        out[f"strict_n{n}"] = int((strict["min_def"] >= n).sum()) if not strict.empty else 0
        out[f"lenient_n{n}"] = int((lenient["min_def"] >= n).sum()) if not lenient.empty else 0
        out[f"median_n{n}"] = int((lenient["med_def"] >= n).sum()) if not lenient.empty else 0
    return out


def verdict(per_match: list[int]) -> str:
    """Declared W1-A decision rule applied to a per-match usable-window distribution."""
    med = float(np.median(per_match)) if per_match else 0.0
    if med >= 30:
        return f"PASS (median {med:.0f}/match >= 30)"
    if med >= 10:
        return f"MARGINAL (median {med:.0f}/match in 10-30: pooled-only)"
    return f"FAIL (median {med:.0f}/match < 10)"


# === W1-B ========================================================================================
def oracle_totals(match_id: str) -> dict[str, float]:
    """Whole-match (home + away) Sofascore team-stat totals, keyed by Sofascore stat key."""
    path = ORACLE_DIR / f"team_stats_{SOFASCORE_MATCH_ID[match_id]}.parquet"
    df = pd.read_parquet(path)
    d = df[df["period"] == "ALL"].drop_duplicates(subset=["key"])
    return {str(r.key): float(r.homeValue) + float(r.awayValue) for r in d.itertuples(index=False)}


def detected_counts(match_id: str, classes: tuple[str, ...]) -> dict[tuple[float, float],
                                                                    dict[str, int]]:
    """Re-run the E2E-Spot peak picker over :data:`SWEEP` from the cached per-chunk score npz.

    Args:
        match_id: registry match id (must have ``results/action_spotting_probe/<id>/``).
        classes: SoccerNet-v2 class names to count.

    Returns:
        ``{(threshold, nms_seconds): {class_name: count}}`` summed over both halves.
    """
    halves = [load_half_timeline(match_id, h)[0] for h in ("h1", "h2")]
    halves = [sc for sc in halves if sc.shape[0]]
    out: dict[tuple[float, float], dict[str, int]] = {}
    for thresh, sep_s in SWEEP:
        sep = int(round(sep_s * FRAME_FPS))
        out[(thresh, sep_s)] = {
            name: sum(len(find_peaks(sc[:, class_index(name)], thresh, sep)) for sc in halves)
            for name in classes
        }
    return out


# === Reporting ===================================================================================
def _fmt_ratio(n: float, truth: float) -> str:
    """``n/truth`` as ``x.xxx``, or ``-`` when truth is missing/zero."""
    return f"{n / truth:.3f}" if truth else "-"


def format_report(census: list[dict], defence: list[dict]) -> str:
    """Render both censuses as Markdown, then append the dated :data:`VERDICT_NOTES` reading."""
    lines = [
        "# W1 feasibility checks -- turnover-window census + defensive-action numerator",
        "",
        "Generated by `python -m tools.w1_feasibility`. CPU only, no new modelling.",
        "",
        "## W1-A -- turnover-window census",
        "",
        "**Turnover definition (shipped, not invented):** `fingerprint.transitions.detect_turnovers`"
        " over `generator.ball.assign_possession(ball, players, smooth=True)` -- the Viterbi team"
        " smoother (2.0 m possession radius, 1.5 m switch penalty). Possession runs per chunk on the"
        " **post-`link_ball`** track and on positions with `calib_error_m <= "
        f"{CALIB_ERROR_MAX_M:.1f}` m and a non-null pitch projection.",
        "",
        f"**Window:** the turnover frame plus {WINDOW_S:.0f} s, sampled on the ball/position grid"
        " (5 frame stride at 25 fps = 26 slots).",
        "",
        "* `turnovers` -- every possession-state change located by the ball (condition (a): the ball"
        " is tracked at the loss moment, which `detect_turnovers` already requires).",
        "* `ball_full` -- condition (b) strict: **every** window slot present in the linked track.",
        f"* `ball_len` -- condition (b) lenient: >= {LENIENT_COV:.0%} of slots present *and* the"
        " +5 s slot present.",
        "* `sN` / `lN` -- condition (c): at least N players of the **pressing (ball-losing)** team"
        " with trusted geometry in **every** slot of the window, under the strict / lenient ball"
        " criterion.",
        "* `mN` -- sensitivity variant: lenient ball, and the **median** (not minimum) over the"
        " window's slots is >= N. Says whether a failure is knife-edge or structural.",
        "* `geom cov` -- mean fraction of window slots that have *any* calibrated pressing-team"
        " player. This is the binding constraint (see below), not the ball.",
        "",
        "| match | turnovers | ball cov | geom cov | ball_full | s3 | s5 | s7 | ball_len | l3 | l5 "
        "| l7 | m3 | m5 | m7 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in census:
        lines.append(
            f"| {r['match_id']} | {r['turnovers']} | {r['mean_ball_cov']:.2f} | "
            f"{r['mean_geom_cov']:.2f} | {r['ball_full']} | "
            f"{r['strict_n3']} | {r['strict_n5']} | {r['strict_n7']} | {r['ball_lenient']} | "
            f"{r['lenient_n3']} | {r['lenient_n5']} | {r['lenient_n7']} | "
            f"{r['median_n3']} | {r['median_n5']} | {r['median_n7']} |"
        )
    skip = ("match_id", "mean_ball_cov", "mean_geom_cov")
    tot = {k: sum(r[k] for r in census) for k in census[0] if k not in skip}
    lines.append(
        f"| **pooled (12)** | {tot['turnovers']} | - | - | {tot['ball_full']} | {tot['strict_n3']} "
        f"| {tot['strict_n5']} | {tot['strict_n7']} | {tot['ball_lenient']} | {tot['lenient_n3']} "
        f"| {tot['lenient_n5']} | {tot['lenient_n7']} | {tot['median_n3']} | {tot['median_n5']} | "
        f"{tot['median_n7']} |"
    )
    lines.append("")
    lines.append("### Distribution and verdict per criterion")
    lines.append("")
    lines.append("| criterion | per-match min | median | max | pooled | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for label, key in [(f"strict, min N>={n}", f"strict_n{n}") for n in DEFENDER_LEVELS] + \
                      [(f"lenient, min N>={n}", f"lenient_n{n}") for n in DEFENDER_LEVELS] + \
                      [(f"lenient, median N>={n}", f"median_n{n}") for n in DEFENDER_LEVELS]:
        vals = [r[key] for r in census]
        lines.append(f"| {label} | {min(vals)} | {np.median(vals):.0f} | {max(vals)} | "
                     f"{sum(vals)} | {verdict(vals)} |")
    lines.append("")

    lines.append("## W1-B -- defensive-action numerator")
    lines.append("")
    lines.append(
        "Detected = E2E-Spot peaks re-picked from the cached per-chunk score npz "
        "(`results/action_spotting_probe/<match>/scores_*.npz`); truth = Sofascore whole-match "
        "home + away. The shipped `summary.json` used `(thresh 0.30, NMS 30 s)` for every class -- "
        "30 s is longer than the typical gap between two fouls, so the sweep re-picks at shorter "
        "NMS windows. Tackles and interceptions have **no SoccerNet-v2 class**, so their detected "
        "count is structurally 0."
    )
    lines.append("")
    header = " | ".join(f"{t:.2f}/{s:.0f}s" for t, s in SWEEP)
    lines.append(f"| match | class | truth | {header} |")
    lines.append("|---|---|---|" + "---|" * len(SWEEP))
    for r in defence:
        for name, key in DEFENSIVE_CLASS_TRUTH.items():
            truth = r["truth"].get(key, 0.0)
            cells = " | ".join(
                f"{r['counts'][(t, s)][name]} ({_fmt_ratio(r['counts'][(t, s)][name], truth)}x)"
                for t, s in SWEEP)
            lines.append(f"| {r['match_id']} | {name} | {truth:.0f} | {cells} |")
    lines.append("")
    lines.append("### Pooled ratios (sum detected / sum truth over the 12 matches)")
    lines.append("")
    lines.append(f"| class | truth total | {header} |")
    lines.append("|---|---|" + "---|" * len(SWEEP))
    for name, key in DEFENSIVE_CLASS_TRUTH.items():
        truth = sum(r["truth"].get(key, 0.0) for r in defence)
        cells = " | ".join(
            f"{sum(r['counts'][(t, s)][name] for r in defence)} "
            f"({_fmt_ratio(sum(r['counts'][(t, s)][name] for r in defence), truth)}x)"
            for t, s in SWEEP)
        lines.append(f"| {name} | {truth:.0f} | {cells} |")
    for label in ("totalTackle", "interceptionWon"):
        truth = sum(r["truth"].get(label, 0.0) for r in defence)
        cells = " | ".join("0 (0.000x)" for _ in SWEEP)
        lines.append(f"| {label} (no class exists) | {truth:.0f} | {cells} |")
    lines.append("")
    lines.append("### PPDA denominator, as literally defined")
    lines.append("")
    lines.append("| match | tackles | interceptions | fouls | PPDA denom (truth) | detected "
                 "(fouls only, 0.30/5s) | ratio | our possession-change events | ballRecovery "
                 "(truth) | ratio |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    cen = {r["match_id"]: r for r in census}
    tot_denom = tot_det = tot_to = tot_rec = 0.0
    for r in defence:
        t = r["truth"]
        denom = sum(t.get(k, 0.0) for k in PPDA_TRUTH_KEYS)
        det = r["counts"][(0.30, 5.0)]["Foul"]
        turn = cen[r["match_id"]]["turnovers"]
        rec = t.get("ballRecovery", 0.0)
        tot_denom, tot_det, tot_to, tot_rec = tot_denom + denom, tot_det + det, tot_to + turn, \
            tot_rec + rec
        lines.append(
            f"| {r['match_id']} | {t.get('totalTackle', 0):.0f} | "
            f"{t.get('interceptionWon', 0):.0f} | {t.get('fouls', 0):.0f} | {denom:.0f} | {det} | "
            f"{_fmt_ratio(det, denom)} | {turn} | {rec:.0f} | {_fmt_ratio(turn, rec)} |"
        )
    lines.append(f"| **pooled (12)** | - | - | - | {tot_denom:.0f} | {tot_det:.0f} | "
                 f"{_fmt_ratio(tot_det, tot_denom)} | {tot_to:.0f} | {tot_rec:.0f} | "
                 f"{_fmt_ratio(tot_to, tot_rec)} |")
    lines.append(VERDICT_NOTES)
    return "\n".join(lines)


def selftest() -> None:
    """Self-check of the window arithmetic (the only non-trivial logic in W1-A)."""
    ball = {100, 105, 110, 115, 120}
    counts = {(f, 1): 6 for f in ball}
    counts[(110, 1)] = 4
    full = window_stats(100, 1, ball, counts, stride=5, n_slot=4)
    assert full["ball_full"] and full["ball_cov"] == 1.0 and full["ball_end"], full
    assert full["min_def"] == 4 and full["med_def"] == 6.0 and full["geom_cov"] == 1.0, full
    holed = window_stats(100, 1, ball - {110}, counts, stride=5, n_slot=4)
    assert not holed["ball_full"] and abs(holed["ball_cov"] - 0.8) < 1e-9, holed
    assert holed["ball_end"], holed
    missing = window_stats(100, 0, ball, counts, stride=5, n_slot=4)
    assert missing["min_def"] == 0 and missing["geom_cov"] == 0.0, missing
    print("selftest ok")


def main() -> None:
    """Run both W1 measurements over the 12 processed EPL matches and write the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    census = []
    for mid in MATCH_IDS:
        m = get(mid)
        r = match_census(m)
        census.append(r)
        print(f"W1-A {mid:22s} turnovers {r['turnovers']:5d}  ball_full {r['ball_full']:5d}  "
              f"s3 {r['strict_n3']:4d}  s5 {r['strict_n5']:4d}  s7 {r['strict_n7']:4d}")

    defence = []
    for mid in MATCH_IDS:
        counts = detected_counts(mid, tuple(DEFENSIVE_CLASS_TRUTH))
        truth = oracle_totals(mid)
        defence.append({"match_id": mid, "counts": counts, "truth": truth})
        f5 = counts[(0.30, 5.0)]["Foul"]
        print(f"W1-B {mid:22s} Foul det(0.30/5s) {f5:4d}  truth {truth.get('fouls', 0):5.0f}  "
              f"tackles {truth.get('totalTackle', 0):4.0f}  int {truth.get('interceptionWon', 0):4.0f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(census, defence), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
