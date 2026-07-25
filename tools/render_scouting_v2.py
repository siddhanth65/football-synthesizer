"""Narrative-first scouting pack v2 (Plan B-6): story -> style -> seams -> players -> validation.

A sibling of :mod:`tools.render_html_report` that REUSES its self-contained single-file aesthetic
(CSS, template, kit chips, tier badges, inline SVG pitch diagrams, section shell) but re-orders the
document around the *story* instead of the CV-vs-oracle validation. Section order:

1. **The story of the match** -- score-state phases (validated goal timeline) + the validated event
   layer.
2. **Man Utd style read** -- the B-4 tracking-native fingerprint (phase profile + counter-press) with
   cross-match context and the B-5 score-state slices.
3. **Seams -- how they were beaten / how they won** -- the case-study read per match.
4. **Players** -- lineup-prior named-on-pitch lists with confidence, exemplar observations, and the
   honest coverage banner.
5. **Validation appendix** -- the old CV-vs-oracle tables + gate live here now.

Built on the validated engines directly (``fingerprint.score_state``, ``fingerprint.style_fingerprint``,
lineup-assign parquets, the E2E summary, the Sofascore oracle) so it renders for all three matches;
the report_v2 tactical tables + gate are pulled into the appendix WHEN a fact store exists and
degrade to an honest banner when it does not (e.g. fulham). Tier labels sit on every section. v1 files
are untouched.

CLI::

    python -m tools.render_scouting_v2 --match all --out-dir results/reports
"""
from __future__ import annotations

import argparse
import html as _html
import json
from datetime import date
from pathlib import Path

import pandas as pd

from core.pitch import METRICS_VERSION
from core.registry import get
from fingerprint import score_state as ss
from fingerprint import style_fingerprint as sf
from tools import render_html_report as rh

OUT_DIR = Path("results/reports")
E2E_DIR = Path("results/action_spotting_probe")
LINEUP_DIR = Path("outputs/identity")

SHORT = {"manutd_liverpool": "Liverpool", "brighton_manutd": "Brighton", "manutd_fulham": "Fulham",
         "southampton_manutd": "Southampton", "palace_manutd": "Palace",
         "manutd_tottenham": "Tottenham"}
RESULT = {"manutd_liverpool": "0-3 loss", "brighton_manutd": "1-2 loss", "manutd_fulham": "1-0 win",
          "southampton_manutd": "3-0 win", "palace_manutd": "0-0 draw",
          "manutd_tottenham": "0-3 loss"}
STATE_ORDER = {"level": 0, "chasing": 1, "leading": 2}

# Validated event-layer summary per match (from results/action_spotting_probe/* + STATUS 2026-07-19).
EVENTS_VALIDATED = {
    "manutd_liverpool": "Goals 3/3 with exact halves (2 H1 / 1 H2). Yellows 5/5, red 0/0, "
    "corners 7/7 EXACT. Shots 21 vs 19, fouls 16 vs 14, offsides 1 vs 2.",
    "brighton_manutd": "Goals 3/3 with exact halves (1 H1 / 2 H2). Yellows 3/3, red 0/0. "
    "Shots 25/25 exact (on/off-target split is noisy zero-shot - aggregate only).",
    "manutd_fulham": "Goal 1/1 with exact half (87' winner). Fouls 21 vs 22, yellows 4 vs 5, "
    "corners 13 vs 15, offsides 2 vs 4. Shots 35 vs 24 OVER-FIRED in H2 (one flagged "
    "broadcast segment explains it - a single audit item, not a claim).",
    "southampton_manutd": "Goals 3/3 with exact halves (2 H1 / 1 H2, all Man Utd). Corners 7/7 "
    "EXACT. Shots 27 vs 26, fouls 29 vs 25, yellows 3 vs 5. The one red card (Southampton) was NOT "
    "detected - a measured zero-shot miss, aggregate events only.",
    "palace_manutd": "Goals 0/0 - negative control passed (no false-positive goal spots on a "
    "scoreless match). Yellows 3 vs oracle total 5, red 0/0. Corners 14 vs oracle total 15, fouls "
    "22 vs 19, shots 22 vs 24, offsides 0 vs 1. Team-level aggregate only (E2E-Spot does not split "
    "by team); no identity chain run on this match.",
    "manutd_tottenham": "Goals: 3 true positives with exact halves (1 H1 / 2 H2, all Tottenham) "
    "plus one flagged replay-window false positive (H2, score 0.338, the known FP - dropped, see "
    "fingerprint.score_state.GOALS). Yellows 8/8 and corners 8/8 EXACT. Red 0/1 (Man Utd's red card "
    "not detected - a measured zero-shot miss). Shots 34 vs oracle total 35, fouls 32 vs 30, "
    "offsides 2 vs 3.",
}

# Pundit-voice story prose per match. Paragraphs are separated by a blank line and rendered as
# separate <p>. Every claim here restates a number computed elsewhere in the same report (score-state
# table, counter-press table, WINS_VS_LOSSES level-state rows) in football language -- no new claims,
# and the caveated space-control/possession proxy is deliberately never narrated (it inverts the
# oracle; the report says so in the appendix and the palace/liverpool prose abstains explicitly).
STORY = {
    "manutd_liverpool":
        "For a little over half an hour this was a contest. United stayed level to 34 minutes and "
        "looked like they belonged in the game - and then it went in a hurry. Two Liverpool goals "
        "inside eight first-half minutes, on 34 and 42, sent United in at the break two down and "
        "already chasing, and when the third landed early in the second half the game was up. From "
        "that 34th minute onward the night was spent running after the score."
        "\n\n"
        "The uncomfortable part is that the wobble started before the scoreboard did. Even at 0-0 the "
        "one thing that is supposed to define this side - swarming the ball the instant they lose it "
        "- just was not there: the counter-press fired on well under half of their high turnovers, "
        "where in the Brighton and Fulham games it was going off nearly three times in four. They "
        "only found that intensity, and only shoved their line up the pitch, once they were three "
        "down and it no longer mattered. Liverpool, meanwhile, won it the way good sides win these - "
        "control, not chaos: they gave the ball away in dangerous areas less than anyone on show and "
        "never had to scramble to get it back.",
    "brighton_manutd":
        "This was a game United could, and probably should, have taken something from. They were "
        "level for the first half hour, and even after Brighton nicked the opener right before the "
        "interval they came back out like a side that fancied it - back on terms early in the second "
        "half, 1-1, with the momentum. For most of the night the margin was a single goal or nothing "
        "at all."
        "\n\n"
        "And then they lost it the way visiting sides so often come unstuck at Brighton: chasing the "
        "game late, committing bodies forward, and undone on the counter deep in stoppage time. This "
        "one stings because it was not a lack of effort or a pressing meltdown - if anything their "
        "work off the ball held up better here than it did against Liverpool. It was a shape problem "
        "the moment they went chasing, and Brighton had the composure to make them pay at the death.",
    "manutd_fulham":
        "Eighty-seven minutes of very little, and then the only moment that mattered. This was a "
        "grind - two sides cancelling each other out until United finally found the goal that won "
        "it, with the match all but over."
        "\n\n"
        "For almost the whole of it they went about the level game the right way: they hunted the "
        "ball back on better than seven of every ten losses high up the pitch and, at 0-0, won it "
        "back inside five seconds close to half the time - faster than in any of the other five "
        "matches we have tracked - from a higher and wider shape with the ball. The moment the "
        "goal went in the instinct was unmistakable: the whole side dropped a long way deeper to see "
        "it out. Say that one plainly, though - the passage in front is only a few minutes of "
        "football, so the shut-up-shop read is a glimpse, not a habit.",
    "southampton_manutd":
        "A proper away performance, and 3-0 barely does it justice. United were level for 35 "
        "minutes, went ahead just after the half hour, made it two before the break and three in "
        "the second half. The lead arrived early and never once looked like narrowing - from the "
        "moment they went in front they were in front for the rest of the night, with no "
        "backs-to-the-wall stretch to survive."
        "\n\n"
        "What separates this from the usual away smash-and-grab is that going ahead did not make "
        "them cautious. Most sides get their noses in front and settle into a block; United did the "
        "opposite - they chased the ball harder once they led than they had at 0-0, and won it back "
        "quicker with it. And they were on the front foot from the first whistle: at 0-0 this is the "
        "highest defensive line and the most territory of any of the six matches we have tracked. "
        "Two things to hold on to before anyone calls it a template - the level-state passage is "
        "short (barely 35 minutes before the opener, so a modest sample) and Southampton finished "
        "the game a man down, a red card the event oracle records and our video layer did not "
        "detect."
        "\n\n"
        "One correction on the record, because it matters: an earlier version of this report had the "
        "two teams the wrong way round and credited that high, front-foot game to Southampton. It "
        "was United's. The mapping was corrected on 2026-07-23 and every number above is recomputed "
        "from the fixed data (see results/PAIR_ANALYSIS_v1.md).",
    "palace_manutd":
        "Ninety minutes, and nobody blinked. This is the only goalless match in the set, which means "
        "there is no swing in the game state to read - nobody chasing, nobody protecting a lead, the "
        "same problem for the full ninety."
        "\n\n"
        "What the tracking will say is that United were relentless without the ball: at 0-0 this is "
        "the most aggressive counter-pressing of any of the six matches, going after the ball on "
        "three of every four losses high up the pitch and winning it back inside five seconds on "
        "better than a third of them. All that work and nothing to show for it - a stalemate, not an "
        "opponent broken. And one thing we will not pretend to know: our possession-style figure for "
        "this match is a coverage-biased proxy that disagrees with the event oracle, so from the "
        "footage we cannot fairly say who had the better of the ball - only who worked harder to win "
        "it back.",
    "manutd_tottenham":
        "There was barely a game to settle into. United were behind inside three minutes at Old "
        "Trafford and spent the next eighty-seven running after it. Tottenham added a second almost "
        "straight after the restart and a third with half an hour of the second half still to play - "
        "by then it had long stopped being a contest."
        "\n\n"
        "That early goal is why there is so little to say about how United wanted to play: they were "
        "level for under three minutes, which is not enough football to judge anybody on. What we "
        "can read is the chase, and it makes for grim watching - all but one of their dangerous "
        "give-aways came while behind, they went after the ball on roughly half of them, and got it "
        "back inside five seconds only about a quarter of the time. A side pushing forward because "
        "it had to, repeatedly failing to win it back before the next counter arrived. This was a "
        "from-behind performance almost from the first whistle, not a game undone by one seam that "
        "opened late.",
}

APPX_HINT = "the credibility spine -- every figure above, sourced (click to open)"

TRACKING = ("tracking-native", "cv")     # teal chip: position/fingerprint, ball-gap tolerant
VALID = ("validated", "oracle")          # blue chip: validated against the oracle
FLOOR = ("floor - not a ranking", "abst")  # red chip: coverage floor
ORACLE = ("oracle", "oracle")


# ==================================================================================================
# Small HTML helpers (reuse rh's CSS classes: table, tag-*, panel shell, seam boxes).
# ==================================================================================================
def _panel(title: str, chip: tuple[str, str], inner: str) -> str:
    """Section shell with a tier chip (reuses rh._panel + the shared tag CSS)."""
    return rh._panel(title, chip[1], chip[0], inner)


def _collapsed_panel(title: str, chip: tuple[str, str], hint: str, inner: str) -> str:
    """Section shell whose body is collapsed behind a ``<details>`` summary.

    Used for the validation appendix so the pundit narrative leads the document while the
    credibility spine stays one click away (never deleted, never hidden).

    Args:
        title: section heading text.
        chip: ``(label, css_class)`` tier chip, as in :func:`_panel`.
        hint: muted one-liner shown next to the summary.
        inner: pre-rendered HTML body.

    Returns:
        The section HTML.
    """
    return (f'<section class="prov-{chip[1]}"><details class="appx"><summary>'
            f'<span class="tag tag-{chip[1]}">{_html.escape(chip[0].upper())}</span> '
            f'{_html.escape(title)}<span class="hint">{_inline(hint)}</span></summary>'
            f'{inner}</details></section>')


def _table(df: pd.DataFrame, headers: dict[str, str], *, ndp: int = 1,
           caption: str = "") -> str:
    """Render a DataFrame as an HTML table; ``headers`` maps column -> display name (and order)."""
    cols = list(headers)
    head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers.values())
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.{ndp}f}"
            cells.append(f"<td>{_html.escape(str(v))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    cap = f"<figcaption>{_inline(caption)}</figcaption>" if caption else ""
    return (f"<figure><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"{cap}</figure>")


def _inline(text: str) -> str:
    """Escape prose for HTML, en-dashing ``--`` but letting literal ``<code>`` spans through."""
    return (_html.escape(text).replace("--", "&mdash;")
            .replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>"))


def _p(text: str) -> str:
    return f"<p>{_inline(text)}</p>"


# ==================================================================================================
# Section 1 -- the story of the match.
# ==================================================================================================
def _goal_timeline(match_id: str, focus: str, opp: str) -> str:
    """Validated goal timeline: each goal's half-clock time, scorer team, running scoreline."""
    rows = []
    manu =  0
    other = 0
    for half, t_s, who in sorted(ss.GOALS.get(match_id, []),
                                 key=lambda g: (0 if g[0] == "h1" else 1, g[1])):
        if who == "manu":
            manu += 1
            team = focus
        else:
            other += 1
            team = opp
        mm, secs = divmod(int(t_s), 60)
        rows.append(f"<tr><td>{half.upper()} {mm:02d}:{secs:02d}</td>"
                    f"<td>{_html.escape(team)}</td><td>{manu}-{other}</td></tr>")
    return ("<figure><table><thead><tr><th>half-clock (broadcast)</th><th>scorer team</th>"
            f"<th>Man Utd scoreline</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "<figcaption>Goal times are the validated E2E-Spot peaks; scorer team is the Sofascore "
            "per-half score delta. No goal-scorer is identified from video.</figcaption></figure>")


def _story_section(match_id: str, focus: str, opp: str) -> str:
    seg = ss.segments(get(match_id))
    seg_disp = seg.assign(window=seg["t0_s"].map(lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}")
                          + "-" + seg["t1_s"].map(lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"))
    tbl = _table(seg_disp, {"half": "half", "state": "Man Utd state", "scoreline": "score",
                            "window": "half-clock window", "stoppage": "reaches stoppage"})
    body = ("".join(_p(par) for par in STORY[match_id].split("\n\n"))
            + "<h3>Score-state phases (validated goal boundaries)</h3>" + tbl
            + "<h3>Validated event layer</h3>" + _goal_timeline(match_id, focus, opp)
            + _p(EVENTS_VALIDATED[match_id])
            + "<p class=\"src\">Zero-shot E2E-Spot vs the cached Sofascore oracle; full table in the "
              "validation appendix.</p>")
    return _panel("The story of the match", VALID, body)


# ==================================================================================================
# Section 2 -- Man Utd style read (B-4 fingerprint + B-5 score-state slices).
# ==================================================================================================
IDENTITY_READ = (
    "Watch United across these three games and the thing that repeats is not a formation or a shape "
    "- it is where they choose to live on the pitch. Line the matches up and every United "
    "performance looks most like another United performance; take away that territorial footprint "
    "and the resemblance falls apart. It is the real estate they occupy, not a fixed structure. The "
    "one on-ball habit that travels match to match is a mid-to-high counter-press - they hunt the "
    "ball in the opposition half, and across the set it fires on something like two of every three "
    "high turnovers. Three matches only, so read it as a tendency, not a law.")


def _style_section(match, match_id: str, focus: str) -> str:
    mi = ss.manu_index(match)
    pp = sf.phase_profile(match)
    pp = pp[pp["team"] == mi].drop(columns="team")
    phase_tbl = _table(pp, {"phase": "phase", "frames": "frames",
                            "def_line_height": "def-line", "width": "width",
                            "compactness": "compactness", "buildup_height": "build-up depth"},
                       caption="Man Utd shape per phase (metres; def-line = deepest-line "
                               "attacking-x, 0 = own goal). Partial-broadcast view inflates lines "
                               "~+11 m - read across states, not vs FIFA.")
    cp = sf.counterpress(match)
    cp = cp[cp["team"] == mi].drop(columns="team")
    cp_tbl = _table(cp, {"losses_outside_third": "outside-third losses",
                         "counterpress_frac": "counter-press frac", "regain_5s_frac": "5s regain",
                         "poss_frames": "poss frames (ball-gap base)", "turnovers": "turnovers"},
                    ndp=3, caption="Counter-press: pressure within 4.57 m of the ball within 5 s of "
                                   "an outside-third loss. poss_frames/turnovers are the ball-gap "
                                   "honesty base, not full-match totals.")
    cp_state = ss.segment_counterpress(match, by="state")
    cp_state = cp_state[cp_state["team"] == mi].drop(columns="team")
    cp_state = cp_state.sort_values("state", key=lambda c: c.map(STATE_ORDER).fillna(9))
    cps_tbl = _table(cp_state, {"state": "Man Utd state", "losses_outside_third": "losses",
                                "counterpress_frac": "counter-press frac",
                                "regain_5s_frac": "5s regain"}, ndp=3,
                     caption="Counter-press sliced by score state (B-5). Small per-state samples - "
                             "losses shown.")
    ph_state = ss.segment_phase(match, by="state")
    ph_state = ph_state[(ph_state["team"] == mi)
                        & (ph_state["phase"].isin(["in_poss", "trans_pos"]))].drop(columns="team")
    ph_state = ph_state.sort_values(["phase", "state"],
                                    key=lambda c: c.map(STATE_ORDER).fillna(c))
    phs_tbl = _table(ph_state, {"phase": "phase", "state": "state", "frames": "frames",
                                "def_line_height": "def-line", "width": "width",
                                "buildup_height": "build-up depth"},
                     caption="In-possession + attacking-transition shape by score state.")
    body = (_p(IDENTITY_READ)
            + "<h3>Phase profile (whole match)</h3>" + phase_tbl
            + "<h3>Counter-press (whole match)</h3>" + cp_tbl
            + "<h3>Counter-press by score state (B-5)</h3>" + cps_tbl
            + "<h3>Shape by score state (B-5)</h3>" + phs_tbl
            + "<p class=\"src\">Source: <code>results/style_fingerprint_v1.md</code> + "
              "<code>results/SCORE_STATE_v1.md</code>.</p>")
    return _panel(f"{focus} style read - tracking-native fingerprint", TRACKING, body)


# ==================================================================================================
# Section 3 -- seams (how beaten / how won). The case-study read per match.
# ==================================================================================================
# Pundit-voice seam read per match. Same discipline as STORY: restates numbers already in this
# report's own tables (or in results/WINS_VS_LOSSES.md, quoted there) as football language.
SEAMS = {
    "manutd_liverpool":
        "United were beaten in the moments the ball changed hands, and it started before Liverpool "
        "ever led. At 0-0 they went after their high turnovers less than half the time - against "
        "roughly three-in-four when level in the Brighton and Fulham games - and when they did win "
        "it back they broke from the shallowest starting point of the three matches. The first-half "
        "goals then squeezed them deeper still, and it was only at 3-0, with the game gone, that "
        "they finally pushed up and pressed with real venom - too late to matter. Liverpool simply "
        "never let them in: fewest give-aways in dangerous areas of any side on show, and the "
        "calmest of the lot about winning it back - control, not chaos. Full phase-by-phase read: "
        "<code>results/CASE_STUDY_manutd_liverpool.md</code>.",
    "brighton_manutd":
        "The soft spot was their shape the instant they went chasing. While the game was level "
        "United defended transitions from a sensible height; once they were behind and pouring "
        "forward, the line they tried to hold when they lost the ball leapt up the pitch - and a "
        "side as sharp on the break as Brighton needs no second invitation. That is exactly the "
        "picture of the stoppage-time winner: caught high, stretched, punished. Crucially this was "
        "not their pressing giving way - their work off the ball actually stood up better than it "
        "had against Liverpool. It was a shape-when-chasing problem, not a lack of legs.",
    "manutd_fulham":
        "How they won: patience, and the ball won back quickly. Eighty-seven minutes at 0-0, and "
        "through all of it United were the better-organised side without it - going after better "
        "than seven of every ten losses high up the pitch and recovering the ball inside five "
        "seconds close to half the time, quicker than in any other match in the set, while playing "
        "from a higher and wider shape on it. The goal came late and they immediately dropped deep "
        "to protect it. That worked here, but it tells us very little: the passage in front lasts "
        "only a few minutes of football.",
    "southampton_manutd":
        "How they won: not a smash-and-grab. United led from the 35th minute and only stretched it "
        "- 1-0, then 2-0 before the break, then 3-0 - so nearly the whole match was played from in "
        "front. The tell is what they did with that lead. The pressing did not relax: at 0-0 they "
        "went after roughly two in every five of their dangerous give-aways, and once ahead that "
        "rose to better than one in two, with the ball won back inside five seconds far more often "
        "than it had been before the opener. They defended from the highest line and spent more of "
        "the game in the opposition's territory than in any other match we have tracked. That is a "
        "control-through-pressure win, not a low-block heist. Two things before anyone calls it a "
        "template: the level-state passage is a modest sample, and Southampton played the closing "
        "stretch a man down (a red card the event oracle has and our video layer missed). And "
        "across the corpus the two wins we have look nothing like each other - there is no "
        "repeatable winning shape here yet "
        "(<code>results/WINS_VS_LOSSES.md</code>).",
    "palace_manutd":
        "There is no seam to name here - United never fell behind and never scored. What they did "
        "do was press: at 0-0 this is the most aggressive counter-pressing of any match in the set, "
        "chasing the ball on three of every four losses high up the pitch and winning it back "
        "inside five seconds on better than a third of them. It bought them nothing. As for who "
        "actually controlled the game, we abstain: our territory and possession figures for this "
        "match are coverage-biased proxies that disagree with the event oracle, so from the footage "
        "we cannot fairly settle that argument. A stalemate, not a beaten or broken opponent.",
    "manutd_tottenham":
        "The seam was open before the game had settled. United conceded at 2:42, which leaves "
        "effectively no level-state football to judge them on - a single dangerous give-away before "
        "they went behind is not a sample. From 0-1 onwards their counter-press fired on around "
        "half of those give-aways and turned into a regain inside five seconds only about a quarter "
        "of the time, all while the deficit grew; Tottenham's third arrived with half of the second "
        "half still to play. So from this footage we cannot fairly judge how this side plays when "
        "the game is level - that game barely existed here. Full phase-by-phase context in the "
        "validation appendix below.",
}


def _seams_section(match_id: str, focus: str, opp: str) -> str:
    result = RESULT.get(match_id, "")
    verb = "won" if result.endswith("win") else "drew" if result.endswith("draw") else "were beaten"
    seam = (f'<div class="seam"><p class="seam-claim">How {_html.escape(focus)} '
            f'{verb} vs '
            f'{_html.escape(opp)}</p><p class="seam-back">{_inline(SEAMS[match_id])}</p></div>')
    extra = ""
    secs = _report_v2_sections(match_id)
    if secs is not None:
        seam_sec = next((s for s in secs if s.title.startswith("Counter-structure")), None)
        if seam_sec is not None:
            extra = "<h3>Counter-structure seams (report_v2)</h3>" + "".join(
                rh._html_block(b) for b in seam_sec.blocks)
    return _panel("Seams - how they were beaten / how they won", TRACKING, seam + extra)


# ==================================================================================================
# Section 4 -- players (lineup-prior named-on-pitch lists + exemplars + coverage banner).
# ==================================================================================================
def _players_section(match, match_id: str, focus: str, opp: str) -> str:
    path = LINEUP_DIR / f"{match_id}_lineup_assign.parquet"
    if not path.exists():
        banner = (f'<div class="queued"><strong>Identity layer not run for '
                  f'{_html.escape(focus)} v {_html.escape(opp)}.</strong> The close-up '
                  f'jersey-number naming pipeline that names players on the Brighton and Liverpool '
                  f'broadcasts has not been run on this one - no players are named here. This is '
                  f'team-level only.</div>')
        return _panel("Players - named on pitch", FLOOR, banner)
    la = pd.read_parquet(path)
    assigned = la[la["assigned"]].copy()
    blocks = []
    for team_name, g in assigned.groupby("team_name"):
        g = g.sort_values("confidence", ascending=False)
        items = []
        for _, r in g.iterrows():
            sub = " (sub)" if bool(r["is_sub"]) else ""
            items.append(f"<li>{_html.escape(str(r['name']))} (#{int(r['shirt'])}){sub} "
                         f"&mdash; confidence {float(r['confidence']):.2f}, {_html.escape(str(r['method']))}, "
                         f"{int(r['n_fragments'])} fragments</li>")
        blocks.append(f"<h3>{_html.escape(str(team_name))} &mdash; {len(g)} named on pitch</h3>"
                      f"<ul>{''.join(items)}</ul>")
    exemplars = _exemplar_html(match)
    coverage = (
        '<div class="cmp-box"><strong>Coverage banner (read this first).</strong> Names come from '
        'a lineup-prior Hungarian assignment of the known 22 to tracks (jersey-vote + formation '
        'position + kit + GK-veto). Per-player EVENT counts are a coverage FLOOR: only ~3.6-5.6% of '
        'team passes carry a named track (broadcast names ~5-6 back-numbers per match via repeated '
        'close-ups, not a uniform XI). Confidence is the assignment posterior, not an accuracy '
        'guarantee. Treat the lists as "named on pitch", the counts as "observed at least N".</div>')
    return _panel("Players - named on pitch", TRACKING,
                  coverage + "".join(blocks) + exemplars)


def _exemplar_html(match) -> str:
    """Named-pass exemplar observations from the ledger (a floor, never a ranking)."""
    led_path = match.aligned.parent.parent / "ledger.parquet"
    if not led_path.exists():
        return ""
    led = pd.read_parquet(led_path)
    named = led[(led["class"] == "PASS") & led["player"].notna()]
    if named.empty:
        return ""
    top = named.groupby("player").size().sort_values(ascending=False).head(5)
    items = "".join(f"<li>{_html.escape(str(p))} &mdash; observed on {int(n)} attributed pass "
                    f"fragments</li>" for p, n in top.items())
    return ("<h3>Exemplar observations (floor, not a ranking)</h3>"
            f"<ul>{items}</ul><p class=\"src\">From the attributed-pass ledger "
            "(<code>results/PLAYER_LEDGER.md</code>); these are the players who surfaced most in the "
            "sparse named-carrier fragments, NOT a passing ranking or a Sofascore comparison.</p>")


# ==================================================================================================
# Section 5 -- validation appendix (report_v2 tables + gate + events, moved to the back).
# ==================================================================================================
def _report_v2_sections(match_id: str):
    """report_v2 sections + gate/audit, or None if no fact store exists for this match."""
    try:
        secs, gate, _facts, audit = rh.build_document(match_id)
        return secs
    except FileNotFoundError:
        return None


def _events_table(match_id: str) -> str:
    """E2E detected goal count vs the Sofascore oracle (the validated headline), per half."""
    summ_path = E2E_DIR / match_id / "summary.json"
    if not summ_path.exists():
        return ""
    from tools.action_spot_probe import goal_oracle  # noqa: PLC0415
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    got = summ["per_class"].get("Goal", {})
    orc = goal_oracle(match_id) or {}
    rows = (f"<tr><td>Goals (H1)</td><td>{got.get('by_half', {}).get('h1', '-')}</td>"
            f"<td>{orc.get('h1', '-')}</td></tr>"
            f"<tr><td>Goals (H2)</td><td>{got.get('by_half', {}).get('h2', '-')}</td>"
            f"<td>{orc.get('h2', '-')}</td></tr>"
            f"<tr><td>Goals (match)</td><td>{got.get('count', '-')}</td>"
            f"<td>{orc.get('total', '-')}</td></tr>")
    return ("<h3>Event spotting vs oracle (validated)</h3><figure><table><thead><tr><th>event</th>"
            f"<th>E2E detected</th><th>Sofascore</th></tr></thead><tbody>{rows}</tbody></table>"
            "<figcaption>Zero-shot E2E-Spot, official SoccerNet-v2 weights (no training on this "
            "broadcast). Broader validated counts in the story section.</figcaption></figure>")


def _validation_section(match_id: str) -> str:
    parts = [_events_table(match_id)]
    try:
        secs, gate, _facts, audit = rh.build_document(match_id)
    except FileNotFoundError:
        parts.append(
            '<div class="queued"><strong>report_v2 fact store not built for this match.</strong> '
            "The CV-vs-oracle tactical tables + evidence gate are not available; the validated layers "
            "shown above (score-state boundaries, event spotting, fingerprint) stand on their own "
            "artifacts.</div>")
        return _collapsed_panel("Validation appendix", ORACLE, APPX_HINT, "".join(parts))
    tlabel, tcls = rh.tier_badge(gate.tier)
    parts.append(
        f'<div class="gate"><span class="gbadge {tcls}">{tlabel}</span><div class="gtext">'
        f'<strong>Ball-evidence gate:</strong> post-link coverage {gate.coverage_pct:.0f}% '
        f'(needs &ge; 40%), pass-recall proxy {gate.recall_pct:.0f}% (needs &ge; 50% for absolute). '
        f'<strong>Body guardrail:</strong> {audit["precision"] * 100:.0f}% '
        f'({audit["n_backed"]}/{audit["n_numbers"]} numbers grounded).</div></div>')
    parts.append("<h3>CV-vs-oracle tactical tables (report_v2)</h3>")
    parts.append(rh._render_section(secs[-1], position_only=False))
    return _collapsed_panel("Validation appendix", ORACLE, APPX_HINT, "".join(parts))


# ==================================================================================================
# Assembly.
# ==================================================================================================
def render(match_id: str, generated: str) -> str:
    """Render the full narrative-first v2 scouting pack for one registered match."""
    m = get(match_id)
    # Scouting pack is Man-Utd-centric (style/seams/timeline all read Man Utd), so focus is Man Utd
    # regardless of registry team order -- teams[0] is the away side for e.g. southampton/palace.
    mi = ss.manu_index(m)
    focus, opp = m.teams[mi], m.teams[1 - mi]
    home, away = m.home_team or m.teams[0], m.away_team or m.teams[1]
    competition = getattr(m, "competition", "") or "match"

    ref_path = Path(f"outputs/eval/{match_id}_ball_eval.json")
    ref = (json.loads(ref_path.read_text(encoding="utf-8")).get("oracle_ref")
           if ref_path.exists() else None)
    score = rh.oracle_score(ref)
    hk, ak = rh._kit(m.home_team or focus), rh._kit(m.away_team or opp)
    score_html = (f'<span class="score">{score[0]}&ndash;{score[1]}</span>'
                  '<span class="score-src">Sofascore</span>' if score else
                  '<span class="score-src">score withheld</span>')
    header = (f'<header><div class="teams">{hk}<span class="vs">{score_html}</span>{ak}</div>'
              f'<p class="fixture">{_html.escape(home)} v {_html.escape(away)} &middot; '
              f'{_html.escape(competition)} &middot; narrative scouting pack v2</p></header>')

    body = [
        header,
        _story_section(match_id, focus, opp),
        _style_section(m, match_id, focus),
        _seams_section(match_id, focus, opp),
        _players_section(m, match_id, focus, opp),
        _validation_section(match_id),
        f'<footer>football-synthesizer scouting pack v2 &middot; match <code>'
        f'{_html.escape(match_id)}</code> &middot; metrics version {METRICS_VERSION} &middot; '
        f'generated {_html.escape(generated)}.<br>Narrative order: story &rarr; style &rarr; seams '
        f'&rarr; players &rarr; validation. Every number keeps its tier label; abstentions are shown, '
        f'not hidden.</footer>',
    ]
    title = f"{focus} v {opp} - scouting pack v2"
    return rh._TEMPLATE.format(title=_html.escape(title), css=rh._CSS, body="\n".join(body))


def generate(match_id: str, *, out_dir: Path = OUT_DIR, generated: str | None = None) -> Path:
    """Render one match to ``<out_dir>/<match_id>_v2.html`` and return the path."""
    gen = generated or date.today().isoformat()
    html = render(match_id, gen)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{match_id}_v2.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    """CLI: render narrative-first v2 scouting packs."""
    ap = argparse.ArgumentParser(description="Narrative-first scouting pack v2 renderer.")
    ap.add_argument("--match", default="all",
                    help="registered match id, comma list, or 'all' (every match with story prose)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    ids = list(STORY) if args.match == "all" else [s.strip() for s in args.match.split(",")]
    for mid in ids:
        out = generate(mid, out_dir=Path(args.out_dir), generated=args.date)
        print(f"[render_scouting_v2] {mid} -> {out}")


if __name__ == "__main__":
    main()
