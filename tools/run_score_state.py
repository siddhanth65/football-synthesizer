"""Assemble the score-state segmentation (Plan B-5): SCORE_STATE_v1.md + the Liverpool case study.

Slices the validated style-fingerprint primitives by Man Utd score state (level / chasing / leading)
using :mod:`fingerprint.score_state`, and renders two markdown deliverables:

* ``results/SCORE_STATE_v1.md`` -- cross-match table: how Man Utd's shape and counter-press shift by
  score state across the three matches (n=3 caveats throughout).
* ``results/CASE_STUDY_manutd_liverpool.md`` -- the 0-3 phase by phase, grounded only in validated
  numbers, with the pass-rate balance and named-player exemplars (observations, not stats).

No new science: every number is a re-bucketing of the B-4 fingerprint (validated tables) by the
validated goal timeline. Run (CPU)::

    python tools/run_score_state.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import registry  # noqa: E402
from fingerprint import score_state as ss  # noqa: E402

MATCHES = ["manutd_liverpool", "brighton_manutd", "manutd_fulham"]
SHORT = {"manutd_liverpool": "Liverpool", "brighton_manutd": "Brighton", "manutd_fulham": "Fulham"}
RESULT = {"manutd_liverpool": "0-3 loss", "brighton_manutd": "1-2 loss", "manutd_fulham": "1-0 win"}
STATE_ORDER = {"level": 0, "chasing": 1, "leading": 2}
SCORELINE_CASE = ["0-0", "0-1", "0-2", "0-3"]


def _fmt(df: pd.DataFrame, ndp: int = 1) -> str:
    return df.round(ndp).to_string(index=False)


def manu_phase(match, mi: int, by: str) -> pd.DataFrame:
    """Man Utd phase profile sliced by ``by`` (state/scoreline), ordered for reading."""
    sp = ss.segment_phase(match, by=by)
    sp = sp[sp["team"] == mi].drop(columns="team")
    if by == "state":
        sp = sp.sort_values([sp.columns[0], "phase"], key=lambda c: c.map(STATE_ORDER).fillna(c))
    return sp.reset_index(drop=True)


def manu_cp(match, mi: int, by: str) -> pd.DataFrame:
    """Man Utd counter-press sliced by ``by``, ordered for reading."""
    cp = ss.segment_counterpress(match, by=by)
    cp = cp[cp["team"] == mi].drop(columns="team")
    if by == "state":
        cp = cp.sort_values(cp.columns[0], key=lambda c: c.map(STATE_ORDER).fillna(c))
    return cp.reset_index(drop=True)


def pass_balance(match, mi: int) -> pd.DataFrame | None:
    """Attributed-PASS balance (Man Utd vs opponent) by state from the ledger (team-split matches).

    A FLOOR: only passes with an on-ball tracked carrier are counted, and per-state counts are not
    separately validated. ``brighton`` is excluded (its team split inverts). Returns ``None`` when no
    ledger exists.
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
        mu = int(tab.loc[state].get(mi, 0))
        op = int(tab.loc[state].get(1 - mi, 0))
        rows.append({"state": state, "ManU_pass": mu, f"{SHORT[match.id]}_pass": op})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cross-match assembly
def build() -> dict:
    """Compute per-match segments + Man Utd state slices + pass balances."""
    out: dict = {"segments": {}, "phase": {}, "cp": {}, "balance": {}, "mi": {}}
    for mid in MATCHES:
        m = registry.get(mid)
        mi = ss.manu_index(m)
        out["mi"][mid] = mi
        out["segments"][mid] = ss.segments(m)
        out["phase"][mid] = manu_phase(m, mi, "state")
        out["cp"][mid] = manu_cp(m, mi, "state")
        out["balance"][mid] = pass_balance(m, mi)
        print(f"[{mid}] segments={len(out['segments'][mid])} "
              f"cp-states={len(out['cp'][mid])}")
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
    """Cross-match attributed-pass balance by state (liverpool + fulham only)."""
    lines = ["| match | state | ManU att. passes | opponent att. passes |",
             "|---|---|---|---|"]
    for mid in MATCHES:
        bal = res["balance"][mid]
        if bal is None or mid == "brighton_manutd":
            continue
        opp_col = f"{SHORT[mid]}_pass"
        for _, r in bal.iterrows():
            lines.append(f"| {SHORT[mid]} | {r['state']} | {int(r['ManU_pass'])} | "
                         f"{int(r[opp_col])} |")
    return "\n".join(lines)


def segments_block(seg: pd.DataFrame) -> str:
    return "```\n" + _fmt(seg) + "\n```"


# ------------------------------------------------------------------ SCORE_STATE_v1.md
def write_cross(res: dict, out: Path) -> None:
    """Render results/SCORE_STATE_v1.md."""
    lines = [
        "# Score-state segmentation v1 (Plan B-5) - Man Utd shape + press by scoreline", "",
        "Every metric below is the B-4 style fingerprint (validated tracking-native primitives)",
        "re-bucketed by Man Utd's score state. States are **from Man Utd's perspective** "
        "(level / chasing / leading). Boundaries are the E2E-Spot goal peaks (validated 7/7 across",
        "these three matches with correct halves); which team scored each goal comes from the",
        "Sofascore per-half score deltas, with Brighton's two second-half goals split by the final",
        "scoreline (the 90+' winner is Brighton's). Engine: `fingerprint/score_state.py`.", "",
        "**n=3 matches, and each state is a slice of an already ball-gap-limited base** - counts",
        "(`frames`, `outside-third losses`) are shown on every row so small samples are visible. The",
        "leading state exists only in the Fulham match and only for the ~5 minutes after the 87'",
        "winner (tiny n - suggestive, not a claim). Absolute line heights carry the ~+11m",
        "partial-broadcast inflation from v1: read across states, not against FIFA numbers.", "",
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
        "## Attributed-pass balance by state (liverpool + fulham only)", "",
        "Ledger PASS events with an on-ball tracked carrier, bucketed by state. A **floor** (only",
        "carried passes count; no per-state validation) and only for the two matches whose match-level",
        "team split held; Brighton is excluded (its split inverts).", "",
        balance_table(res), "",
        "## Honest read", "", CROSS_READ, "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


# ------------------------------------------------------------------ CASE_STUDY_manutd_liverpool.md
def write_case(res: dict, out: Path) -> None:
    """Render results/CASE_STUDY_manutd_liverpool.md (scoreline-resolved)."""
    m = registry.get("manutd_liverpool")
    mi = res["mi"]["manutd_liverpool"]
    phase_sl = manu_phase(m, mi, "scoreline")
    cp_sl = manu_cp(m, mi, "scoreline")
    phase_sl = phase_sl[phase_sl["scoreline"].isin(SCORELINE_CASE)]
    cp_sl = cp_sl[cp_sl["scoreline"].isin(SCORELINE_CASE)]
    order = {s: i for i, s in enumerate(SCORELINE_CASE)}
    phase_sl = phase_sl.sort_values(["scoreline", "phase"], key=lambda c: c.map(order).fillna(c))
    cp_sl = cp_sl.sort_values("scoreline", key=lambda c: c.map(order))
    lines = [
        "# Case study: Manchester United 0-3 Liverpool (Plan B-5)", "",
        "How the 0-3 happened phase by phase, grounded **only in validated numbers**: the",
        "tracking-native style fingerprint (B-4) re-bucketed by the validated goal timeline",
        "(E2E-Spot goals, validated 3/3 with correct halves: 34:26 and 41:58 in H1, 12:19 in H2 --",
        "all Liverpool). States are from Man Utd's perspective. Every ceiling from B-4 still applies",
        "(ball-gap possession base, partial-broadcast line inflation ~+11m); per-scoreline slices are",
        "small - `frames` and `losses` are shown so the samples are visible.", "",
        "## Headline", "", CASE_HEADLINE, "",
        "## Score-state timeline", "",
        segments_block(res["segments"]["manutd_liverpool"]), "",
        "## Man Utd shape, scoreline by scoreline", "",
        "`def_line_height` = deepest-line attacking-x, `buildup_height` = mean outfield attacking-x.",
        "", "```", _fmt(phase_sl), "```", "",
        "## Man Utd counter-press, scoreline by scoreline", "",
        "```", _fmt(cp_sl, 3), "```", "",
        "## Attributed-pass balance by state (floor)", "",
        "```", _fmt(res["balance"]["manutd_liverpool"]), "```",
        "Near-even at level (69:74) and Man Utd marginally ahead when chasing (112:100) - the trailing",
        "side pushes and sees more of the ball. A floor (carried passes only); read as balance, not",
        "totals.", "",
        "## Named-player exemplars (observations, not stats)", "", CASE_PLAYERS, "",
        "## Abstentions (stated proudly)", "", CASE_ABSTAIN, "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


# Fixed prose, filled from the computed run (numbers verified against the tables above).
CASE_HEADLINE = """\
Manchester United did not need the scoreboard to start losing to Liverpool. At 0-0 across the first
34 minutes their counter-press already fired on only 0.455 of outside-third losses (22 losses) --
barely two-thirds of the ~0.72 they managed at level state against both Brighton (0.719) and Fulham
(0.718) -- and even their win-it-and-go attacking transition was the shallowest at level state of the
three, build-up 45.3 m versus 61.4 m (Brighton) and 57.9 m (Fulham). The two first-half goals (34:26,
41:58) then pinned United deeper still (in-possession build-up 51.2 m at 0-0 -> 41.4 m at 0-1, and the
attacking transition down to 32.5 m at 0-1 -- small samples), and only once 0-3 down after
57' did they finally push up (in-possession build-up 57.1 m, deepest-line 51.2 m) and press hardest
(0.690) -- energy that arrived when the game was already gone. Liverpool, for their part, lost the
ball outside their own third the fewest times of any side across the three matches (58) and felt the
least urgency to win it back (0.241 five-second regain): United were beaten in the win-it/lose-it
phase before the deficit ever forced their hand."""

CASE_PLAYERS = """\
- **Mohamed Salah** and **Alexis Mac Allister** were both identity-assigned on the pitch for this
  match (part of the 20-player Liverpool identity chain, confidence 0.78-1.00). These are named-track
  observations, not per-player metrics.
- In the attributed-pass ledger (a coverage floor: only ~5.6% of Liverpool's team passes carry a
  named track), Mac Allister appears on the most fragments of any player (5 attributed passes),
  Marcus Rashford next (4). These are "observed at least N" exemplars - NOT a passing ranking and not
  comparable to the Sofascore totals (Mac Allister's true totalPass was 49).
- Salah was named on the pitch but drew zero attributed pass fragments in the floor; we name his
  presence, we do not claim his involvement count."""

CASE_ABSTAIN = """\
- No goal-scorer identity is claimed from CV: goal *times* are the validated E2E spots, goal
  *ownership* is the Sofascore per-half delta (all three Liverpool). We do not name who scored from
  tracks.
- Per-scoreline slices (0-1, 0-2) rest on <100 in-possession frames and single-digit losses - the
  arrows (deeper when pinned, higher when 0-3 down) are read as direction, not magnitude.
- Absolute line heights are not FIFA metres (partial-broadcast inflation); only within-match,
  across-state comparisons are made.
- Player pass counts are a floor bounded by named-fragment coverage, never a ranking."""

CROSS_READ = """\
1. The counter-press collapse against Liverpool was already present at 0-0. Man Utd's level-state
   counter-press was 0.455 vs Liverpool but 0.719 (Brighton) and 0.718 (Fulham) - the press did not
   fail because United were chasing; it was the weakest of the three even while the game was level
   (Liverpool level n=22, Brighton 64, Fulham 71 - small but a wide gap).
2. Chasing lifts the press, not lowers it. In both losses Man Utd's counter-press rose from level to
   the trailing states as the match wore on (Liverpool 0.455 level -> 0.690 at 0-3; the numbers climb
   with the deficit) - the intensity arrives late, once the game is gone.
3. Leading = drop deep (one match, tiny n). In the ~5 minutes Man Utd led Fulham 1-0 they sat far
   back (out-of-possession build-up 28.5 m vs 43.1 m at level; deepest-line 23.4 vs 38.0) - a
   shut-up-shop signal, but n is single-digit-frames territory; suggestive only.
4. Shape shifts with state as expected: trailing sides push their line and centroid higher (Brighton
   trans_neg deepest-line 45.4 level -> 63.6 chasing; Liverpool build-up rises into 0-3) and widen
   slightly - the score state moves the block, confirming the segmentation is tracking something real.
5. Pass balance follows the scoreline: the trailing team sees marginally more of the ball
   (Liverpool: ManU 112:100 when chasing vs 69:74 level; Fulham: ManU behind on the ball 8:16 only
   while leading late) - a floor, but the direction is consistent.
6. n=3, small per-state slices, one leading state: this is a shape of behaviour, not a validated
   law. The one robust, cross-match claim is #1 - the Liverpool press was flat from kickoff, not just
   after the goals."""


def main() -> None:
    res = build()
    write_cross(res, Path("results/SCORE_STATE_v1.md"))
    write_case(res, Path("results/CASE_STUDY_manutd_liverpool.md"))


if __name__ == "__main__":
    main()
