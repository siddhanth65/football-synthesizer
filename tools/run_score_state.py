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

# Ordered by result group: the losses first, then the win, then the draw -- so the cross-match
# tables read as losses-vs-rest (the money comparison for the counter-press finding).
# southampton_manutd (a 3-0 Man Utd win) is processed but its team anchor collapsed (149084 team-0
# vs 670 team-1 player rows, balance 0.004) -- only one usable side, no Man Utd rows -- so it is
# excluded from the two-team score-state analytics (re-anchoring is a generator fix, out of scope).
EXCLUDED = {"southampton_manutd": "team-anchor collapse (0.004 balance) -> no usable Man Utd side"}
MATCHES = ["manutd_liverpool", "manutd_tottenham", "brighton_manutd",
           "manutd_fulham", "palace_manutd"]
SHORT = {"manutd_liverpool": "Liverpool", "manutd_tottenham": "Tottenham",
         "brighton_manutd": "Brighton", "manutd_fulham": "Fulham", "palace_manutd": "Palace"}
RESULT = {"manutd_liverpool": "0-3 loss", "manutd_tottenham": "0-3 loss",
          "brighton_manutd": "1-2 loss", "manutd_fulham": "1-0 win", "palace_manutd": "0-0 draw"}
# Result bucket for the money table (level-state counter-press, losses vs rest).
GROUP = {"manutd_liverpool": "loss", "manutd_tottenham": "loss", "brighton_manutd": "loss",
         "manutd_fulham": "win", "palace_manutd": "draw"}
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


def money_table(res: dict) -> str:
    """The money comparison: Man Utd LEVEL-state counter-press per match, grouped by result.

    Answers whether the 'counter-press flat from kickoff in losses' finding repeats across the losses
    versus the win + draw. Level state = scoreline 0-0 (before any goal moves Man Utd off level).
    """
    lines = ["| result group | match | level-state losses | counter-press frac | 5s regain |",
             "|---|---|---|---|---|"]
    for grp in ("loss", "win", "draw"):
        for mid in MATCHES:
            if GROUP[mid] != grp:
                continue
            lvl = res["cp"][mid][res["cp"][mid]["state"] == "level"]
            if lvl.empty:
                lines.append(f"| {grp} | {SHORT[mid]} ({RESULT[mid]}) | 0 | - | - |")
                continue
            r = lvl.iloc[0]
            lines.append(f"| {grp} | {SHORT[mid]} ({RESULT[mid]}) | "
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
        "# Score-state segmentation v2 (Plan B-5) - Man Utd shape + press by scoreline, 6-match corpus",
        "",
        "Every metric below is the B-4 style fingerprint (validated tracking-native primitives)",
        "re-bucketed by Man Utd's score state. States are **from Man Utd's perspective** "
        "(level / chasing / leading). Boundaries are the validated E2E-Spot goal peaks (per-half count",
        "matched to the Sofascore split); which team scored comes from the Sofascore per-half deltas.",
        "Brighton's two H2 goals are split by the final scoreline (90+' winner is Brighton's);",
        "Tottenham's fourth E2E H2 peak (the known replay false positive) is dropped to honour the true",
        "1H1/2H2 split. Engine: `fingerprint/score_state.py`. v1 (n=3) kept at",
        "`results/SCORE_STATE_v1.md`.", "",
        "**Corpus: 5 of 6 matches** -- 3 losses (Liverpool 0-3, Tottenham 0-3, Brighton 1-2), 1 win",
        "(Fulham 1-0) and 1 draw (Palace 0-0). `southampton_manutd` (a 3-0 Man Utd win) is processed",
        "but excluded: its team anchor collapsed (149084 team-0 vs 670 team-1 player rows, balance",
        "0.004), so it has no usable Man Utd side. That leaves the win column thinner than the corpus",
        "headline suggests -- stated so the n is honest.", "",
        "**n=5, each state a slice of an already ball-gap-limited base** - counts (`frames`,",
        "`outside-third losses`) are on every row so small samples are visible. Two matches barely",
        "have a level state: Tottenham (opener ~3', so level is only its first ~160 s) and Southampton",
        "(excluded). The leading state exists only in the Fulham match (~5 min after the 87' winner,",
        "tiny n). Absolute line heights carry the ~+11m partial-broadcast inflation: read across",
        "states, not against FIFA numbers.", "",
        "## The money comparison: level-state counter-press by result", "",
        "Man Utd's counter-press while the game is still level (0-0), grouped by how the match ended.",
        "This is the direct test of the Liverpool case-study finding -- was the press flat from",
        "kickoff a losses pattern, or Liverpool-specific?", "",
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
1. THE finding does NOT repeat. The Liverpool 'counter-press flat from kickoff' was Liverpool-
   specific, not a losses pattern. Level-state (0-0) counter-press: Liverpool 0.455 (flat, low) BUT
   Brighton - also a loss - 0.719, essentially identical to Fulham (win 0.718) and Palace (draw
   0.750). A losing side pressed exactly as hard at 0-0 as the win and the draw did. The collapse
   did not precede the scoreline against Brighton.
2. Tottenham cannot be tested for this. It conceded at ~3', so its level state holds a single
   outside-third loss (0.000 counter-press on n=1 - meaningless). When a team goes behind almost
   immediately there is no level-state sample to ask 'did the collapse precede the goal?'.
3. So across the three losses the pre-scoreline-collapse claim is 1 for, 1 against, 1 unevaluable:
   Liverpool shows it, Brighton contradicts it, Tottenham can't be judged. Not a repeatable pattern -
   report the Liverpool case as a single-match observation, and the wider corpus argues against
   generalizing it.
4. Where the two 0-3 losses DO stand out is the match-level / chasing press, not the level state.
   Tottenham's chasing counter-press 0.489 (regain 0.255) and Liverpool's match-level 0.600 are the
   corpus lows - but that is press while ALREADY behind, confounded with game state, and Liverpool's
   even rose with the deficit (0.455 level -> 0.660 chasing). Heavy losses show a weak press overall,
   not a weak press before the scoreline.
5. Leading = drop deep survives only as the one-match Fulham signal (tiny n): in the ~5 min at 1-0 up
   Man Utd sat back (in-possession build-up 46.1 m vs 56.9 at level; deepest-line 41.1 vs 49.8).
6. Shape shifts with state, confirming the segmentation tracks something real: trailing sides push
   the line and centroid higher (Tottenham chasing in_poss build-up 58.3 m; Liverpool build-up rises
   into 0-3). Ignore the Tottenham level rows (n=1/12 frames - garbage from the ~3' opener).
7. Pass balance (floor, liverpool + fulham only) still follows the scoreline: the trailing team sees
   marginally more of the ball (Liverpool ManU 112:100 chasing vs 69:74 level).
8. Honest n: 5 usable matches (southampton excluded, anchor collapse), so the 'wins' side of the
   comparison is Fulham alone plus the Palace draw; level-state samples range 1 (Tottenham) to 71
   (Fulham). The one robust cross-match statement is the negative one: the flat-press-from-kickoff is
   not a Man-Utd-in-losses law, it is what happened against Liverpool."""


def main() -> None:
    res = build()
    write_cross(res, Path("results/SCORE_STATE_v2.md"))
    write_case(res, Path("results/CASE_STUDY_manutd_liverpool.md"))


if __name__ == "__main__":
    main()
