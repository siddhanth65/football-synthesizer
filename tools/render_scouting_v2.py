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

# Story prose per match, grounded in the computed score-state numbers (verified against the tables).
STORY = {
    "manutd_liverpool":
        "Man Utd were level for 34 minutes, then chased for the rest of the match as the deficit "
        "deepened 0-1 (34:26) -> 0-2 (41:58) -> 0-3 (57'). The telling detail is that the collapse "
        "did not wait for the scoreboard (see the style read below): the counter-press was already "
        "the flattest of the three matches while the game was still 0-0.",
    "brighton_manutd":
        "Level for 31 minutes, then chasing after Brighton's opener; Man Utd equalised early in the "
        "second half to level it 1-1, before conceding a stoppage-time winner (chasing 1-2 from "
        "90+3'). Two-thirds of the match was spent either level or a single goal apart.",
    "manutd_fulham":
        "Level for 87 minutes, then leading 1-0 after a late winner. In the handful of minutes ahead "
        "Man Utd dropped noticeably deeper (out-of-possession build-up 43.1 -> 28.5 m) - a "
        "shut-up-shop reflex, though on a tiny sample.",
    "southampton_manutd":
        "The mirror image of the two heavy losses: level for 35 minutes, then leading for the rest as "
        "the lead only grew - 1-0 (35:04) -> 2-0 (41:03) -> 3-0. The lead arrived before half-time and "
        "never narrowed. And unlike a shut-up-shop win, Man Utd's counter-press did not drop off once "
        "ahead: the level-state fraction 0.39 rose to 0.50 while leading - they kept pressing on the "
        "front foot (small per-state samples).",
    "palace_manutd":
        "Level the entire 90 minutes - Man Utd's only scoreless match in the sample, so there is no "
        "score-state transition to slice. A single-phase read: Man Utd held the ball for most of the "
        "match (space-control 0.544 vs Palace 0.456) without turning it into a goal.",
    "manutd_tottenham":
        "The shortest 'level' window of the six matches: Man Utd conceded inside 2:42 and chased for "
        "the remaining 87 minutes as the deficit deepened 0-1 (02:42) -> 0-2 (H2 03:31) -> 0-3 (H2 "
        "33:25). With only a single outside-third loss recorded at level state, the chasing-state read "
        "is effectively the whole match: 47 of 48 outside-third losses came while chasing, counter-"
        "press firing on 0.49 of them (5s regain 0.26) - a from-behind performance almost start to "
        "finish, not a single seam that opened late.",
}

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
    return _html.escape(text).replace("--", "&mdash;")


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
    body = (_p(STORY[match_id])
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
    "Across the three matches Man Utd carry a repeatable TERRITORIAL signature, not a formation "
    "fingerprint: in the OT sliced-Wasserstein embedding every Man Utd side's nearest neighbour is "
    "another Man Utd side (mean intra-Man Utd distance 4.36 vs 6.38 to opponents), but removing the "
    "centroid collapses the separation - what repeats is WHERE they occupy the pitch, not a shape. "
    "The one habit that holds match to match is a mid/high counter-press (fraction 0.60-0.72, 5s "
    "regain 0.35-0.47). n=3, no significance claim.")


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
SEAMS = {
    "manutd_liverpool":
        "Man Utd were beaten in the win-it/lose-it phase before the deficit forced their hand. At "
        "0-0 their counter-press fired on only 0.455 of outside-third losses (22 losses) - against "
        "~0.72 at level state vs Brighton and Fulham - and their attacking transition was the "
        "shallowest at level state of the three (build-up 45.3 m vs 61.4/57.9). The first-half goals "
        "then pinned them deeper (in-possession build-up 51.2 -> 41.4 m at 0-1); only 0-3 down did "
        "they push up (57.1 m) and press hardest (0.690) - too late. Liverpool coughed the ball up "
        "outside their own third the fewest times of any side (58) with the lowest regain urgency "
        "(0.241): control, not chaos. Full phase-by-phase read: "
        "<code>results/CASE_STUDY_manutd_liverpool.md</code>.",
    "brighton_manutd":
        "The seam was the transition line when chasing: after Brighton's opener Man Utd's trans_neg "
        "deepest-line jumped from 45.4 m (level) to 63.6 m (chasing) - caught high and stretched on "
        "the counter, which is how the stoppage-time winner arrived. Their counter-press held up "
        "better here than vs Liverpool (0.632-0.719), so this was a transition-shape loss, not a "
        "pressing collapse.",
    "manutd_fulham":
        "How they won: a level-state grind (level for 87') with a solid counter-press (0.718, 5s "
        "regain 0.465, the best of the three at level) and a higher, wider in-possession block "
        "(build-up 56.9 m), converted late. Once 1-0 up they dropped deep to defend it (out-of-poss "
        "build-up 28.5 m) - effective here, but the leading sample is tiny (single-digit frames).",
    "southampton_manutd":
        "How they won: not a smash-and-grab. Man Utd led from the 35th minute and only extended the "
        "lead (1-0 -> 2-0 before half-time -> 3-0), so most of the match was played from ahead. The "
        "tell is that the counter-press did NOT relax with the lead - the level-state fraction 0.394 "
        "(33 outside-third losses) rose to 0.500 while leading (54 losses): they kept hunting the ball "
        "on the front foot rather than dropping into a block. A control-through-pressure win, not a "
        "low-block heist - though the per-state samples are small and Southampton played the closing "
        "stretch a man down (Sofascore red card, not CV-detected).",
    "palace_manutd":
        "No seam to name - Man Utd never fell behind and never scored. They pressed hard (counter-"
        "press 0.75 of 40 outside-third losses, 5s regain 0.375) and held more space (0.544 vs 0.456) "
        "and more attacking-third control (0.478 vs 0.326) than Palace, but the territorial edge did "
        "not convert into goals: a stalemate, not a beaten or broken opponent.",
    "manutd_tottenham":
        "The seam was open before kickoff finished settling: Man Utd conceded at 2:42, so there is "
        "almost no level-state sample to compare against (1 outside-third loss). From 0-1 on, their "
        "counter-press only fired on 0.49 of 47 outside-third losses (5s regain 0.26) while defending "
        "a growing deficit - Tottenham's third and final goal arrived at 33:25 of the second half, by "
        "which point the game state had been chasing for over an hour. Full phase-by-phase context in "
        "the validation appendix below.",
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
                  f'jersey-number naming pipeline that named 20 players for Brighton and Liverpool '
                  f'has not been run on this broadcast - no players are named here. This is team-level '
                  f'only.</div>')
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
        return _panel("Validation appendix", ORACLE, "".join(parts))
    tlabel, tcls = rh.tier_badge(gate.tier)
    parts.append(
        f'<div class="gate"><span class="gbadge {tcls}">{tlabel}</span><div class="gtext">'
        f'<strong>Ball-evidence gate:</strong> post-link coverage {gate.coverage_pct:.0f}% '
        f'(needs &ge; 40%), pass-recall proxy {gate.recall_pct:.0f}% (needs &ge; 50% for absolute). '
        f'<strong>Body guardrail:</strong> {audit["precision"] * 100:.0f}% '
        f'({audit["n_backed"]}/{audit["n_numbers"]} numbers grounded).</div></div>')
    parts.append("<h3>CV-vs-oracle tactical tables (report_v2)</h3>")
    parts.append(rh._render_section(secs[-1], position_only=False))
    return _panel("Validation appendix", ORACLE, "".join(parts))


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
                    help="registered match id, comma list, or 'all' (the three ManU matches)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    ids = (["manutd_liverpool", "brighton_manutd", "manutd_fulham"] if args.match == "all"
           else [s.strip() for s in args.match.split(",")])
    for mid in ids:
        out = generate(mid, out_dir=Path(args.out_dir), generated=args.date)
        print(f"[render_scouting_v2] {mid} -> {out}")


if __name__ == "__main__":
    main()
