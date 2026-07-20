"""Assemble the score-state segmentation v3 (Plan B-5) across the full 6-match corpus.

Slices the validated style-fingerprint primitives by Man Utd score state (level / chasing / leading)
using :mod:`fingerprint.score_state`, and renders three markdown deliverables:

* ``results/SCORE_STATE_v3.md`` -- cross-match table: how Man Utd's shape and counter-press shift by
  score state across all six matches (small-n caveats throughout).
* ``results/WINS_VS_LOSSES.md`` -- the synthesis: is there a repeatable WIN-shape vs LOSS-shape at
  LEVEL state (0-0, before any goal decides posture) across the two wins and three losses?
* ``results/CASE_STUDY_manutd_liverpool.md`` -- the three losses (Liverpool, Tottenham, Brighton),
  each with named-player exemplars (observations, not stats): three different ways Man Utd lost.

No new science: every number is a re-bucketing of the B-4 fingerprint (validated tables) by the
validated goal timeline. The two expensive primitives (phase-frame table, turnover-press table) are
computed once per match and every view is derived from the annotated copies. Run (CPU)::

    python tools/run_score_state.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import registry  # noqa: E402
from core.pitch import ATT_THIRD_X  # noqa: E402
from fingerprint import score_state as ss  # noqa: E402
from fingerprint import style_fingerprint as sf  # noqa: E402

# southampton_manutd was excluded in v2 (team-anchor collapse, balance 0.004 -> no usable Man Utd
# side). It has since been re-anchored (balance 0.909: 67532 team-0 vs 61404 team-1 player rows) and
# now enters the two-team analytics -- the corpus is the full six. Ordered losses, wins, draw so the
# cross-match tables read result-group first.
MATCHES = ["manutd_liverpool", "manutd_tottenham", "brighton_manutd",
           "manutd_fulham", "southampton_manutd", "palace_manutd"]
SHORT = {"manutd_liverpool": "Liverpool", "manutd_tottenham": "Tottenham",
         "brighton_manutd": "Brighton", "manutd_fulham": "Fulham",
         "southampton_manutd": "Southampton", "palace_manutd": "Palace"}
RESULT = {"manutd_liverpool": "0-3 loss", "manutd_tottenham": "0-3 loss",
          "brighton_manutd": "1-2 loss", "manutd_fulham": "1-0 win",
          "southampton_manutd": "0-3 win", "palace_manutd": "0-0 draw"}
GROUP = {"manutd_liverpool": "loss", "manutd_tottenham": "loss", "brighton_manutd": "loss",
         "manutd_fulham": "win", "southampton_manutd": "win", "palace_manutd": "draw"}
# The three losses get the case study; the two wins + draw round out the corpus.
LOSSES = ["manutd_liverpool", "manutd_tottenham", "brighton_manutd"]
BALANCE_OK = {"manutd_liverpool", "manutd_fulham", "manutd_tottenham"}  # ledger split held (not Bri)
STATE_ORDER = {"level": 0, "chasing": 1, "leading": 2}
SCORELINE_CASE = ["0-0", "0-1", "0-2", "0-3"]
PHASE_COLS = list(sf.PHASE_METRIC_COLS)


def _fmt(df: pd.DataFrame, ndp: int = 1) -> str:
    return df.round(ndp).to_string(index=False)


# ------------------------------------------------------------------ derived views (reuse annotations)
def phase_by(ptab: pd.DataFrame, mi: int, by: str) -> pd.DataFrame:
    """Man Utd mean shape metrics per ``(<by>, phase)`` from an annotated phase-frame table."""
    mu = ptab[ptab["team"] == mi]
    g = mu.groupby([by, "phase"])
    agg = g[PHASE_COLS].mean()
    agg.insert(0, "frames", g.size())
    out = agg.reset_index()
    if by == "state":
        out = out.sort_values(["state", "phase"], key=lambda c: c.map(STATE_ORDER).fillna(c))
    return out.reset_index(drop=True)


def cp_by(ltab: pd.DataFrame, mi: int, by: str) -> pd.DataFrame:
    """Man Utd counter-press per ``<by>`` bucket from an annotated outside-third-loss table."""
    mu = ltab[ltab["team"] == mi]
    g = mu.groupby(by)
    out = g.agg(losses_outside_third=("pressed", "size"),
                counterpress_frac=("pressed", "mean"),
                regain_5s_frac=("regained", "mean")).reset_index()
    if by == "state":
        out = out.sort_values(by, key=lambda c: c.map(STATE_ORDER).fillna(c))
    return out.reset_index(drop=True)


def level_synthesis(ptab: pd.DataFrame, ltab: pd.DataFrame, mi: int) -> dict:
    """The level-state (0-0) win/loss-shape row: press, regain, block depth, territory, transition.

    ``block_height`` = mean out-of-possession deepest-line attacking-x (defensive block).
    ``att3_control`` = share of level in-possession frames whose mean outfield attacking-x is beyond
    the ``ATT_THIRD_X`` (70 m) line. ``trans_depth`` = mean attacking-transition build-up depth.
    Counts ride along so the (often small) level-state samples stay visible.
    """
    mp = ptab[(ptab["team"] == mi) & (ptab["state"] == "level")]
    ml = ltab[(ltab["team"] == mi) & (ltab["state"] == "level")]
    inp, outp, tpos = (mp[mp["phase"] == p] for p in ("in_poss", "out_poss", "trans_pos"))
    m = lambda s: float(s.mean()) if len(s) else float("nan")  # noqa: E731
    return {
        "lvl_losses": len(ml),
        "cp_frac": m(ml["pressed"]), "regain_5s": m(ml["regained"]),
        "block_height": m(outp["def_line_height"]),
        "att3_control": m(inp["buildup_height"] >= ATT_THIRD_X),
        "trans_depth": m(tpos["buildup_height"]),
        "in_frames": len(inp), "out_frames": len(outp), "trans_frames": len(tpos),
    }


def pass_balance(match, mi: int) -> pd.DataFrame | None:
    """Attributed-PASS balance (Man Utd vs opponent) by state from the ledger (floor).

    Only passes with an on-ball tracked carrier are counted, and per-state counts are not separately
    validated. Returns ``None`` when no ledger exists.
    """
    led_path = match.aligned.parent.parent / "ledger.parquet"
    if not led_path.exists():
        return None
    led = pd.read_parquet(led_path)
    p = led[(led["class"] == "PASS") & led["team"].notna()].copy()
    p["frame"] = p["frame_index"]
    p["state"] = ss.annotate(match.id, p[["chunk", "frame"]])["state"].to_numpy()
    tab = p.groupby(["state", "team"]).size().unstack(fill_value=0)
    rows = []
    for state in sorted(tab.index, key=lambda s: STATE_ORDER.get(s, 9)):
        rows.append({"state": state, "ManU_pass": int(tab.loc[state].get(mi, 0)),
                     f"{SHORT[match.id]}_pass": int(tab.loc[state].get(1 - mi, 0))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cross-match assembly
def build() -> dict:
    """Compute annotated phase + loss tables once per match; derive every score-state view."""
    out: dict = {"segments": {}, "phase": {}, "cp": {}, "balance": {}, "mi": {},
                 "level": {}, "ptab": {}, "ltab": {}}
    for mid in MATCHES:
        m = registry.get(mid)
        mi = ss.manu_index(m)
        ptab = ss.annotate(m.id, sf.phase_frame_table(m))
        losses, _poss, _turn = sf.turnover_press_table(m)
        ltab = ss.annotate(m.id, losses)
        out["mi"][mid] = mi
        out["ptab"][mid], out["ltab"][mid] = ptab, ltab
        out["segments"][mid] = ss.segments(m)
        out["phase"][mid] = phase_by(ptab, mi, "state")
        out["cp"][mid] = cp_by(ltab, mi, "state")
        out["level"][mid] = level_synthesis(ptab, ltab, mi)
        out["balance"][mid] = pass_balance(m, mi) if mid in BALANCE_OK else None
        print(f"[{mid}] mi={mi} states={len(out['cp'][mid])} "
              f"level_in={out['level'][mid]['in_frames']}")
    return out


def cross_cp_table(res: dict) -> str:
    """Cross-match Man Utd counter-press-by-state table (markdown)."""
    lines = ["| match (result) | state | outside-third losses | counter-press frac | 5s regain |",
             "|---|---|---|---|---|"]
    for mid in MATCHES:
        for _, r in res["cp"][mid].iterrows():
            lines.append(f"| {SHORT[mid]} ({RESULT[mid]}) | {r['state']} | "
                         f"{int(r['losses_outside_third'])} | {r['counterpress_frac']:.3f} | "
                         f"{r['regain_5s_frac']:.3f} |")
    return "\n".join(lines)


def money_table(res: dict) -> str:
    """The money comparison: Man Utd LEVEL-state counter-press per match, grouped by result."""
    lines = ["| result group | match | level-state losses | counter-press frac | 5s regain |",
             "|---|---|---|---|---|"]
    for grp in ("loss", "win", "draw"):
        for mid in MATCHES:
            if GROUP[mid] != grp:
                continue
            lv = res["level"][mid]
            if lv["lvl_losses"] == 0:
                lines.append(f"| {grp} | {SHORT[mid]} ({RESULT[mid]}) | 0 | - | - |")
                continue
            lines.append(f"| {grp} | {SHORT[mid]} ({RESULT[mid]}) | {lv['lvl_losses']} | "
                         f"{lv['cp_frac']:.3f} | {lv['regain_5s']:.3f} |")
    return "\n".join(lines)


def cross_phase_table(res: dict) -> str:
    """Cross-match Man Utd shape-by-state table for two readable phases (in_poss + trans_pos)."""
    lines = ["| match | state | phase | frames | def-line | width | buildup depth |",
             "|---|---|---|---|---|---|---|"]
    for mid in MATCHES:
        pp = res["phase"][mid]
        for phase in ("in_poss", "trans_pos"):
            for _, r in pp[pp["phase"] == phase].iterrows():
                lines.append(
                    f"| {SHORT[mid]} | {r['state']} | {phase} | {int(r['frames'])} | "
                    f"{r['def_line_height']:.1f} | {r['width']:.1f} | {r['buildup_height']:.1f} |")
    return "\n".join(lines)


def balance_table(res: dict) -> str:
    """Cross-match attributed-pass balance by state (matches whose ledger split held)."""
    lines = ["| match | state | ManU att. passes | opponent att. passes |", "|---|---|---|---|"]
    for mid in MATCHES:
        bal = res["balance"][mid]
        if bal is None:
            continue
        opp_col = f"{SHORT[mid]}_pass"
        for _, r in bal.iterrows():
            lines.append(f"| {SHORT[mid]} | {r['state']} | {int(r['ManU_pass'])} | "
                         f"{int(r[opp_col])} |")
    return "\n".join(lines)


def synthesis_table(res: dict) -> str:
    """The WINS_VS_LOSSES level-state synthesis: five tracking-native metrics, grouped by result."""
    lines = ["| group | match | level losses | cp frac | 5s regain | block height (out-poss) | "
             "att-3rd control | trans depth |", "|---|---|---|---|---|---|---|---|"]
    for grp in ("win", "loss", "draw"):
        for mid in MATCHES:
            if GROUP[mid] != grp:
                continue
            lv = res["level"][mid]

            def g(key: str, n: int, fmt: str) -> str:
                v = lv[key]
                return "-" if (n == 0 or np.isnan(v)) else format(v, fmt)  # noqa: B023
            lines.append(
                f"| {grp} | {SHORT[mid]} ({RESULT[mid]}) | {lv['lvl_losses']} | "
                f"{g('cp_frac', lv['lvl_losses'], '.3f')} | "
                f"{g('regain_5s', lv['lvl_losses'], '.3f')} | "
                f"{g('block_height', lv['out_frames'], '.1f')} | "
                f"{g('att3_control', lv['in_frames'], '.3f')} | "
                f"{g('trans_depth', lv['trans_frames'], '.1f')} |")
    return "\n".join(lines)


def synthesis_counts_table(res: dict) -> str:
    """Level-state frame counts behind the synthesis (so every small sample is visible)."""
    lines = ["| match | in-poss frames | out-poss frames | trans_pos frames | outside-third losses |",
             "|---|---|---|---|---|"]
    for mid in MATCHES:
        lv = res["level"][mid]
        lines.append(f"| {SHORT[mid]} ({RESULT[mid]}) | {lv['in_frames']} | {lv['out_frames']} | "
                     f"{lv['trans_frames']} | {lv['lvl_losses']} |")
    return "\n".join(lines)


def segments_block(seg: pd.DataFrame) -> str:
    return "```\n" + _fmt(seg) + "\n```"


# ------------------------------------------------------------------ SCORE_STATE_v3.md
def write_cross(res: dict, out: Path) -> None:
    """Render results/SCORE_STATE_v3.md."""
    lines = [
        "# Score-state segmentation v3 (Plan B-5) - Man Utd shape + press by scoreline, full 6-match "
        "corpus", "",
        "Every metric below is the B-4 style fingerprint (validated tracking-native primitives)",
        "re-bucketed by Man Utd's score state. States are **from Man Utd's perspective** "
        "(level / chasing / leading). Boundaries are the validated E2E-Spot goal peaks (per-half count",
        "matched to the Sofascore split); which team scored comes from the Sofascore per-half deltas.",
        "Brighton's two H2 goals are split by the final scoreline (90+' winner is Brighton's);",
        "Tottenham's fourth E2E H2 peak (the known replay false positive) is dropped to honour the true",
        "1H1/2H2 split. Engine: `fingerprint/score_state.py`. v1 (n=3) and v2 (n=5, southampton",
        "excluded) kept at `results/SCORE_STATE_v1.md` / `results/SCORE_STATE_v2.md`.", "",
        "**Corpus: all 6 matches** -- 3 losses (Liverpool 0-3, Tottenham 0-3, Brighton 1-2), 2 wins",
        "(Fulham 1-0, Southampton 0-3) and 1 draw (Palace 0-0). `southampton_manutd` was excluded at",
        "v2 (team-anchor collapse, balance 0.004) but is now re-anchored (balance 0.909) and included,",
        "giving the comparison a real second win.", "",
        "**n=6, each state a slice of an already ball-gap-limited base** - counts (`frames`,",
        "`outside-third losses`) are on every row so small samples stay visible. Two matches barely",
        "have a level state: Tottenham (opener ~3', level is only its first ~160 s) and Southampton",
        "(Man Utd ahead from ~35', so its level state is small: 33 losses). The leading state exists in",
        "Fulham (~5 min at 1-0) and Southampton (most of the match, ahead from ~35'). Absolute line",
        "heights carry the ~+11m partial-broadcast inflation: read across states, not against FIFA",
        "numbers.", "",
        "## The money comparison: level-state counter-press by result", "",
        "Man Utd's counter-press while the game is still level (0-0), grouped by how the match ended.",
        "This is the direct test of the Liverpool case-study finding -- was the press flat from",
        "kickoff a losses pattern, or Liverpool-specific? (Full five-metric level-state synthesis in",
        "`results/WINS_VS_LOSSES.md`.)", "",
        money_table(res), "",
        "## Score-state timeline (validated goal boundaries)", "",
    ]
    for mid in MATCHES:
        lines += [f"**{SHORT[mid]}** ({RESULT[mid]}):", "", segments_block(res["segments"][mid]), ""]
    lines += [
        "## Man Utd counter-press by state (cross-match)", "",
        "Counter-press fraction = pressure within 4.57 m of the ball within 5 s of an outside-third",
        "loss; 5s regain = ball won back in that window.", "",
        cross_cp_table(res), "",
        "## Man Utd shape by state (in-possession + attacking transition)", "",
        "`def-line` = deepest-line attacking-x, `buildup depth` = mean outfield attacking-x (0 = own",
        "goal). `trans_pos` = the win-it-and-go attacking transition.", "",
        cross_phase_table(res), "",
        "## Attributed-pass balance by state (ledger split held: liverpool, fulham, tottenham)", "",
        "Ledger PASS events with an on-ball tracked carrier, bucketed by state. A **floor** (only",
        "carried passes count; no per-state validation) and only for the matches whose match-level",
        "team split held; Brighton is excluded (its split inverts); Southampton/Palace have no ledger.",
        "", balance_table(res), "",
        "## Honest read", "", CROSS_READ, "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


# ------------------------------------------------------------------ WINS_VS_LOSSES.md
def write_wins(res: dict, out: Path) -> None:
    """Render results/WINS_VS_LOSSES.md -- the level-state win-shape vs loss-shape synthesis."""
    lines = [
        "# Wins vs losses: is there a repeatable win-shape at level state? (Plan B-5 synthesis)", "",
        "The corpus now has **two real wins** (Fulham 1-0, Southampton 0-3) against **three losses**",
        "(Liverpool 0-3, Tottenham 0-3, Brighton 1-2) and one draw (Palace 0-0). The question Sid",
        "asked originally: *how do they win vs how do they lose*, in the tracking-native metrics. To",
        "avoid the game-state confound we test at **LEVEL state only** (scoreline 0-0, before any goal",
        "moves Man Utd off level and changes their posture). Every number is the B-4 fingerprint",
        "re-bucketed by the validated goal timeline; engine `fingerprint/score_state.py`. Pre-committed",
        "metrics, all at 0-0: counter-press fraction, 5 s regain, defensive block height (out-of-",
        "possession deepest line), attacking-third control (share of in-possession frames with the",
        "team's mean line beyond the 70 m third), and attacking-transition depth.", "",
        "## Level-state comparison, all six (grouped win / loss / draw)", "",
        synthesis_table(res), "",
        "Level-state sample sizes (the honesty counters -- several slices are small):", "",
        synthesis_counts_table(res), "",
        "## The money question: is there a distinction, or is it n-limited noise?", "",
        WINS_READ, "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


# ------------------------------------------------------------------ CASE_STUDY (three losses)
def write_case(res: dict, out: Path) -> None:
    """Render results/CASE_STUDY_manutd_liverpool.md -- the three losses, three different shapes."""
    lines = [
        "# Case study: three ways Man Utd lost (Liverpool, Tottenham, Brighton)", "",
        "Three defeats, grounded **only in validated numbers**: the tracking-native style fingerprint",
        "(B-4) re-bucketed by the validated goal timeline (E2E-Spot goal peaks, per-half counts matched",
        "to the Sofascore split). States are from Man Utd's perspective. This file supersedes the",
        "single-match Liverpool case study -- the n=6 corpus retracted the 'counter-press collapses",
        "before the scoreline' reading as **Liverpool-specific** (STATUS 2026-07-20), so the three",
        "losses are framed as three *different* shapes, not one pattern. Every B-4 ceiling applies",
        "(ball-gap possession base, partial-broadcast line inflation ~+11m); `frames`/`losses` counts",
        "are shown so the samples stay visible.", "",
        "## Headline: three different ways to lose", "", CASE_HEADLINE, "",
        "## The three losses at level state (0-0), side by side", "",
        "From the level-state synthesis (`results/WINS_VS_LOSSES.md`) -- the shape *before* any goal:",
        "", loss_level_table(res), "",
    ]
    for mid in LOSSES:
        mi = res["mi"][mid]
        lines += [f"## {SHORT[mid]} ({RESULT[mid]})", "", CASE_SECTIONS[mid], "",
                  "Score-state timeline:", "", segments_block(res["segments"][mid]), ""]
        if mid == "manutd_liverpool":
            psl = phase_by(res["ptab"][mid], mi, "scoreline")
            csl = cp_by(res["ltab"][mid], mi, "scoreline")
            order = {s: i for i, s in enumerate(SCORELINE_CASE)}
            psl = psl[psl["scoreline"].isin(SCORELINE_CASE)].sort_values(
                ["scoreline", "phase"], key=lambda c: c.map(order).fillna(c))
            csl = csl[csl["scoreline"].isin(SCORELINE_CASE)].sort_values(
                "scoreline", key=lambda c: c.map(order))
            lines += [
                "Man Utd shape, scoreline by scoreline (`def_line_height` = deepest line, "
                "`buildup_height` = mean outfield line):", "",
                "```", _fmt(psl), "```", "",
                "Man Utd counter-press, scoreline by scoreline:", "",
                "```", _fmt(csl, 3), "```", "",
                "Attributed-pass balance by state (floor):", "",
                "```", _fmt(res["balance"][mid]), "```", "",
            ]
    lines += ["## Named-player exemplars (observations, not stats)", "", CASE_PLAYERS, "",
              "## Abstentions (stated proudly)", "", CASE_ABSTAIN, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


def loss_level_table(res: dict) -> str:
    """Level-state row for each of the three losses, plus the two wins for contrast."""
    lines = ["| match | level losses | cp frac | 5s regain | block height | att-3rd ctrl | "
             "trans depth |", "|---|---|---|---|---|---|---|"]
    for mid in (*LOSSES, "manutd_fulham", "southampton_manutd"):
        lv = res["level"][mid]
        cp = "-" if lv["lvl_losses"] == 0 else f"{lv['cp_frac']:.3f}"
        rg = "-" if lv["lvl_losses"] == 0 else f"{lv['regain_5s']:.3f}"
        a3 = "-" if lv["in_frames"] == 0 else f"{lv['att3_control']:.3f}"
        td = "-" if lv["trans_frames"] == 0 else f"{lv['trans_depth']:.1f}"
        bh = "-" if lv["out_frames"] == 0 else f"{lv['block_height']:.1f}"
        lines.append(f"| {SHORT[mid]} ({RESULT[mid]}) | {lv['lvl_losses']} | {cp} | {rg} | "
                     f"{bh} | {a3} | {td} |")
    return "\n".join(lines)


# Fixed prose, filled from the computed run (numbers verified against the tables above).
CROSS_READ = """\
1. THE Liverpool finding still does NOT repeat -- and the second win makes it starker. Liverpool's
   flat 0-0 counter-press (0.455) was Liverpool-specific: Brighton (also a loss) pressed 0.719 at 0-0,
   and the two WINS split the whole range -- Fulham 0.718 (high) vs Southampton 0.394 (low). Level-
   state counter-press does not separate win from loss, nor even the two wins from each other. The
   pre-scoreline collapse is one match (Liverpool), not a corpus law.
2. Tottenham stays unevaluable at level (conceded ~3', level = 1 loss). Southampton's level state is
   small too (33 losses -- ahead from ~35') but usable, and it is the low-press win, which is exactly
   what kills any 'wins press harder at 0-0' story.
3. Where the wins DO differ at 0-0 is not the press but the defensive BLOCK: both wins sat deeper
   (out-of-possession deepest line 32.9 Southampton / 38.0 Fulham vs 40.6-49.9 across the losses),
   with the goalless draw (34.8) in the win band. Territorial, not press, and small-n -- the full
   five-metric level-state synthesis is in `results/WINS_VS_LOSSES.md`.
4. The two 0-3 losses still carry the lowest match-level regain, but that is press while ALREADY
   behind (both spent almost the whole match chasing), confounded with game state -- not a 0-0 signal.
5. Leading = press-while-ahead now has TWO matches, not one. Fulham dropped its line in the ~5 min at
   1-0; Southampton led from ~35' and pressed MORE when ahead (leading 0.500-0.519 vs level 0.394) --
   matching the Southampton report's 'counter-press rose when ahead'. The wins' energy is a
   with-the-lead trait, not a from-kickoff one.
6. Shape shifts with state, confirming the segmentation tracks something real: trailing sides push the
   line and centroid higher (Liverpool build-up rises into 0-3; Tottenham chasing sits high). Ignore
   the Tottenham level rows (n=1/12 frames -- garbage from the ~3' opener).
7. Pass balance (floor; liverpool, fulham, tottenham) still follows the scoreline: the trailing team
   sees marginally more of the ball.
8. Honest n: 6 usable matches now. Wins = Fulham + Southampton, draw = Palace; level-state samples
   range from 1 (Tottenham) to 71 (Fulham) outside-third losses. The one robust cross-match statement
   is still the negative one -- the flat-press-from-kickoff is not a Man-Utd-in-losses law, it is what
   happened against Liverpool."""

WINS_READ = """\
**Short answer: no repeatable win-shape in the PRESS; a weak, consistent-direction win-shape in the
defensive BLOCK. Small n -- read as direction, not law.**

1. Counter-press fraction at 0-0 does NOT separate wins from losses. The two wins straddle the loss
   range: Fulham 0.718 sits right on top of the Brighton loss (0.719), while Southampton 0.394 falls
   BELOW the Liverpool loss (0.455). A team that pressed 0.394 at 0-0 won 3-0; a team that pressed
   0.719 at 0-0 lost 1-2. Level-state press is n-limited noise here.

2. 5 s regain and transition depth overlap the same way. Regain: wins 0.333-0.465, losses 0.227-0.406
   -- the Brighton loss (0.406) beats the Southampton win (0.333). Transition depth: wins 46.2-57.9,
   losses 45.3-61.4 -- total overlap. Neither is a win-shape.

3. The ONE metric that separates: defensive block height at 0-0 (out-of-possession deepest line).
   Both wins defended the deepest of the six -- Southampton 32.9 m and Fulham 38.0 m -- below all
   three losses (Liverpool 40.6, Brighton 46.9, Tottenham 49.9), with the goalless draw (34.8) sitting
   in the win band. Even dropping the unevaluable Tottenham level (32 out-poss frames), the two wins
   (32.9, 38.0) still sit under the two evaluable losses (40.6, 46.9).

4. Attacking-third control points the same way. The two wins held the ball in the attacking third the
   LEAST at 0-0 (Southampton 0.111, Fulham 0.173) while the Brighton loss was the most territorial
   (0.262). So the win-shape at level state is a DEEPER, LESS territorial block -- a control/counter
   posture -- not a front-foot press. That is consistent with the Southampton report: Man Utd's press
   rose after they went ahead, they did not out-press at 0-0.

5. The honest ceiling. This is a two-win signal, and one of the wins (Southampton) has a tiny level
   sample: 63 in-possession, 76 out-of-possession, 82 transition frames, 33 outside-third losses (the
   ~35 min before the opener). The block-height gap (~5-15 m) lives partly inside the ~+11 m partial-
   broadcast inflation band and is confounded with venue/opponent territory. And the goalless DRAW
   sits with the wins on block depth, so 'deep block' is better read as a **did-not-lose** shape than
   a **win** shape. Verdict: wins-vs-losses in the press is noise at this n; wins-vs-losses in
   defensive block depth (and attacking-third control) is a weak, consistent-direction territorial
   signal -- a real 'how they win vs how they lose' arrow, but one that needs more wins to confirm,
   not a validated law."""

CASE_HEADLINE = """\
Across the three defeats there is no single failure mode -- the n=6 corpus retracted that idea. Man
Utd lost to Liverpool, Tottenham and Brighton in three different shapes:

- **Liverpool (0-3):** beaten in the win-it/lose-it phase from kickoff. At 0-0 their counter-press
  fired on only 0.455 of outside-third losses (22 losses) and their attacking transition was shallow
  (build-up 45.3 m); the goals then pinned them deeper. This is the one match where the collapse
  preceded the scoreline -- and it did NOT generalise.
- **Tottenham (0-3):** behind from the ~3rd minute (162.5 s opener), so there is essentially no level
  state to judge (1 outside-third loss at 0-0). The loss shape is 'conceded early, chased all game' --
  the whole match is the chasing state, and the fingerprint can say nothing about their 0-0 posture.
- **Brighton (1-2):** pressed NORMALLY at 0-0 (0.719 -- identical to the Fulham win 0.718) and had
  MORE of the ball in the attacking third than any side (att-3rd control 0.262), yet still lost. They
  equalised for 1-1 and conceded a 90+' winner. This is a 'front-foot but couldn't hold on' defeat,
  the direct counter-example to the Liverpool press-collapse reading."""

CASE_SECTIONS = {
    "manutd_liverpool": """\
Manchester United did not need the scoreboard to start losing to Liverpool. At 0-0 across the first 34
minutes their counter-press fired on only 0.455 of outside-third losses (22 losses) -- barely two-
thirds of the ~0.72 they managed at level state against Brighton (0.719) and Fulham (0.718) -- and
their win-it-and-go attacking transition was the shallowest at level state of the three losses
(build-up 45.3 m). The two first-half goals (34:26, 41:58) then pinned United deeper still (in-
possession build-up 51.2 m at 0-0 -> 41.4 m at 0-1), and only once 0-3 down after 57' did they finally
push up and press hardest (0.690) -- energy that arrived when the game was already gone. Liverpool lost
the ball outside their own third the fewest of any side (58) and felt the least urgency to win it back
(0.241 five-second regain): United were beaten in the win-it/lose-it phase before the deficit forced
their hand. This is the single match behind the (retracted) 'press collapses before the scoreline'
reading.""",
    "manutd_tottenham": """\
Tottenham is the loss the fingerprint cannot dissect. Man Utd conceded at ~3' (162.5 s), so the level
state holds a single outside-third loss (0.000 counter-press on n=1 -- meaningless) and one in-
possession frame. From then on the entire match is the chasing state: 0-1, then 0-2 early in H2
(210.5 s), then 0-3 (2005.5 s). Chasing, Man Utd pushed the line and centroid up (in-possession
build-up in the high 50s m) and their counter-press ran 0.489 with a 0.255 regain -- among the corpus
lows, but that is press while already two/three down, confounded with game state. The honest read is
structural: when a side concedes in the third minute there is no 0-0 sample to ask how they set up,
and this defeat's shape is simply 'behind from minute three, never level again'.""",
    "brighton_manutd": """\
Brighton is the direct counter-example to the Liverpool reading. At 0-0 Man Utd pressed hard (counter-
press 0.719, essentially the Fulham-win number 0.718) and were the MOST territorial side of the whole
corpus in possession -- 0.262 of their level in-possession frames had the team's mean line beyond the
70 m third, more than any win. Their attacking transition at 0-0 was the deepest of the three losses
(build-up 61.4 m) and they defended from a higher block (46.9 m). None of that is a pre-scoreline
collapse. They fell behind 0-1, equalised for 1-1, and conceded a 90+' stoppage-time winner (Joao
Pedro) to lose 1-2. This is a front-foot, high-territory performance that lost late -- pressing and
possession at 0-0 looked like a win, and the result did not follow.""",
}

CASE_PLAYERS = """\
Floor observations from the attributed-pass ledger (`results/PLAYER_LEDGER.md`): a player is named
only when a named track is the on-ball carrier within 3 m of a filtered PASS, so coverage is sparse
(3.6-12.4% of team passes) and these are 'observed on at least N fragments' exemplars -- NOT a passing
ranking and NOT comparable to the Sofascore totals. Man Utd exemplars per loss:

- **Liverpool (coverage 5.6%):** Marcus Rashford (4 attributed fragments), with Lisandro Martinez and
  Noussair Mazraoui (2 each) also named on the pitch. On the Liverpool side Alexis Mac Allister drew
  the most fragments of anyone (5) and Mohamed Salah was identity-assigned but drew zero -- named
  presence, not an involvement count.
- **Tottenham (coverage 12.4%, the richest of the losses):** Amad Diallo, Alejandro Garnacho and Diogo
  Dalot (2 fragments each) for Man Utd; on the Tottenham side Dejan Kulusevski (5), Micky van de Ven,
  Timo Werner, James Maddison, Manuel Ugarte and Rodrigo Bentancur (4 each) were all named.
- **Brighton (coverage 3.6%):** Kobbie Mainoo (3 fragments) top for Man Utd, with Amad Diallo, Harry
  Maguire and Lisandro Martinez (2 each) -- Maguire is one of the players surfaced by the roster-mask
  levers. These are presence/involvement floors, never per-player rates."""

CASE_ABSTAIN = """\
- No goal-scorer identity is claimed from CV: goal *times* are the validated E2E spots, goal
  *ownership* is the Sofascore per-half delta. We do not name who scored from tracks.
- Per-scoreline slices (Liverpool 0-1, 0-2) rest on <100 in-possession frames and single-digit losses
  -- the arrows (deeper when pinned, higher when 0-3 down) are read as direction, not magnitude.
- Tottenham's level state (1 loss, 1 in-possession frame) is not evaluated; its 0-0 posture is unknown.
- Absolute line heights are not FIFA metres (partial-broadcast inflation ~+11m); only within-match,
  across-state comparisons are made.
- Player pass counts are a floor bounded by named-fragment coverage, never a ranking."""


def main() -> None:
    res = build()
    write_cross(res, Path("results/SCORE_STATE_v3.md"))
    write_wins(res, Path("results/WINS_VS_LOSSES.md"))
    write_case(res, Path("results/CASE_STUDY_manutd_liverpool.md"))


if __name__ == "__main__":
    main()
