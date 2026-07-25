"""Self-contained HTML scouting-pack renderer for a registered match.

Reuses the ``report.report_v2`` fact store + three-tier ball-evidence gate WHOLESALE (via
:func:`report.report_v2.build_document`) so this tool never recomputes or re-grounds a single number:
it renders exactly the gated Sections that report_v2 already audited at 100 % guardrail precision, plus
per-match identity/events panels drawn from their own validated artifacts, into one offline HTML file.

Cardinal rule (inherited from report_v2): only gated facts render. Every metric keeps its provenance
chip (CV / oracle) and its tier context (position-only always renders; ball families render in the
match's gate tier -- absolute totals, comparative shares, or an explicit abstention). Abstentions are
shown proudly in a dedicated "what we don't claim" panel.

The output is a single file with ALL CSS inline, no CDN / external requests, responsive, dark by
default with a ``prefers-color-scheme`` light variant, and small hand-rolled inline SVG pitch diagrams
(top-down 105x68) where they aid reading. No runtime JS clock: the generation date is passed in.

CLI::

    python -m tools.render_html_report --match manutd_liverpool
    python -m tools.render_html_report --match all --out-dir results/reports
"""
from __future__ import annotations

import argparse
import html as _html
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from core.pitch import METRICS_VERSION, PITCH_LEN, PITCH_WID
from core.registry import get
from report.report_v2 import _html_block, _inline_html, build_document

OUT_DIR = Path("results/reports")
ORACLE_DICTS = Path("outputs/oracle/sofascore/match_dicts_England_Premier_League_24_25.json")


# ==================================================================================================
# Kit chips. Man Utd red and Liverpool red are near-identical, so the accent palette stays NEUTRAL
# (slate/teal) and teams are disambiguated by an explicit labelled swatch, never by a red doc accent.
# ==================================================================================================
@dataclass(frozen=True)
class Kit:
    """One team's kit swatch: primary/secondary fill and whether it is striped."""

    primary: str
    secondary: str
    striped: bool = False


TEAM_KITS: dict[str, Kit] = {
    "Man Utd": Kit("#da291c", "#ffffff"),
    "Liverpool": Kit("#c8102e", "#00b2a9"),      # teal trim distinguishes it from Utd red on-screen
    "Brighton": Kit("#0057b8", "#ffffff", striped=True),
    "France": Kit("#1a2a6c", "#ffffff"),
    "Senegal": Kit("#00853f", "#fdef42"),
}
_DEFAULT_KIT = Kit("#5a6472", "#ffffff")


# ==================================================================================================
# Pure seams (unit-tested): tier badge mapping + pitch->SVG scaling math.
# ==================================================================================================
_TIER_BADGE = {
    "absolute": ("ABSOLUTE", "t-abs"),
    "comparative": ("COMPARATIVE - relative claims only", "t-cmp"),
    "abstain": ("ABSTAINED", "t-abst"),
}


def tier_badge(tier: str) -> tuple[str, str]:
    """Map a gate tier to its ``(label, css_class)`` badge (``absolute`` if unknown)."""
    return _TIER_BADGE.get(tier, _TIER_BADGE["absolute"])


def pitch_to_svg(x_m: float, y_m: float, w: float, h: float, margin: float) -> tuple[float, float]:
    """Map pitch metres (0..105 x, 0..68 y) to SVG px inside a ``w`` x ``h`` box with ``margin``.

    The drawable inner rectangle is ``[margin, w-margin] x [margin, h-margin]``. The y axis is flipped
    so pitch ``y=0`` (one touchline) sits at the bottom of the SVG, matching a broadcast top-down view.

    Args:
        x_m: Pitch x in metres (0 = left goal line, ``PITCH_LEN`` = right goal line).
        y_m: Pitch y in metres (0 = bottom touchline, ``PITCH_WID`` = top touchline).
        w: SVG viewBox width in px.
        h: SVG viewBox height in px.
        margin: Padding in px between the pitch rectangle and the viewBox edge.

    Returns:
        ``(px, py)`` pixel coordinates.
    """
    iw, ih = w - 2 * margin, h - 2 * margin
    px = margin + (x_m / PITCH_LEN) * iw
    py = margin + (1.0 - y_m / PITCH_WID) * ih
    return px, py


# ==================================================================================================
# Inline SVG pitch diagrams (hand-rolled, small; draw only position-only facts that always render).
# ==================================================================================================
_SVG_W, _SVG_H, _SVG_M = 420.0, 280.0, 14.0


def _svg_pitch_base() -> list[str]:
    """Return the SVG elements for a bare top-down pitch (outline, halfway, centre circle, boxes)."""
    x0, y0 = pitch_to_svg(0, 0, _SVG_W, _SVG_H, _SVG_M)
    x1, y1 = pitch_to_svg(PITCH_LEN, PITCH_WID, _SVG_W, _SVG_H, _SVG_M)
    mx, _ = pitch_to_svg(PITCH_LEN / 2, 0, _SVG_W, _SVG_H, _SVG_M)
    cx, cy = pitch_to_svg(PITCH_LEN / 2, PITCH_WID / 2, _SVG_W, _SVG_H, _SVG_M)
    els = [f'<rect class="pf" x="{x0:.1f}" y="{y1:.1f}" width="{x1 - x0:.1f}" '
           f'height="{y0 - y1:.1f}" rx="3"/>',
           f'<line class="pl" x1="{mx:.1f}" y1="{y1:.1f}" x2="{mx:.1f}" y2="{y0:.1f}"/>',
           f'<circle class="pl" cx="{cx:.1f}" cy="{cy:.1f}" r="{9.15 / PITCH_LEN * (_SVG_W - 2 * _SVG_M):.1f}" fill="none"/>']
    for gx in (0.0, PITCH_LEN):        # both 16.5 m penalty boxes
        bx = 16.5 if gx == 0.0 else PITCH_LEN - 16.5
        pxa, pya = pitch_to_svg(gx, (PITCH_WID - 40.3) / 2, _SVG_W, _SVG_H, _SVG_M)
        pxb, pyb = pitch_to_svg(bx, (PITCH_WID + 40.3) / 2, _SVG_W, _SVG_H, _SVG_M)
        els.append(f'<rect class="pl" x="{min(pxa, pxb):.1f}" y="{min(pya, pyb):.1f}" '
                   f'width="{abs(pxb - pxa):.1f}" height="{abs(pyb - pya):.1f}" fill="none"/>')
    return els


def _svg_wrap(inner: list[str], caption: str) -> str:
    """Wrap SVG element strings + a caption in a responsive figure."""
    body = "".join(inner)
    return (f'<figure class="pitch"><svg viewBox="0 0 {_SVG_W:.0f} {_SVG_H:.0f}" '
            f'role="img" preserveAspectRatio="xMidYMid meet">{body}</svg>'
            f'<figcaption>{_inline_html(caption)}</figcaption></figure>')


def svg_line_heights(line_focus: float, line_opp: float, focus: str, opp: str) -> str:
    """Two-team defensive-line diagram: each team's de-biased line drawn from its OWN goal.

    ``focus`` defends the left goal, ``opp`` the right, so the strip between the two lines is the
    compressed middle third both back lines leave open. Position-only (always renders).
    """
    els = _svg_pitch_base()
    xf, ytop = pitch_to_svg(line_focus, PITCH_WID, _SVG_W, _SVG_H, _SVG_M)
    _, ybot = pitch_to_svg(line_focus, 0, _SVG_W, _SVG_H, _SVG_M)
    xo, _ = pitch_to_svg(PITCH_LEN - line_opp, PITCH_WID, _SVG_W, _SVG_H, _SVG_M)
    els.append(f'<line class="lnf" x1="{xf:.1f}" y1="{ytop:.1f}" x2="{xf:.1f}" y2="{ybot:.1f}"/>')
    els.append(f'<line class="lno" x1="{xo:.1f}" y1="{ytop:.1f}" x2="{xo:.1f}" y2="{ybot:.1f}"/>')
    els.append(f'<text class="pt" x="{xf + 4:.1f}" y="{ytop + 14:.1f}">{_html.escape(focus)} '
               f'{line_focus:.0f} m</text>')
    els.append(f'<text class="pt" text-anchor="end" x="{xo - 4:.1f}" y="{ytop + 14:.1f}">'
               f'{_html.escape(opp)} {line_opp:.0f} m</text>')
    cap = (f"De-biased defensive lines, each measured up-pitch from that team's own goal: "
           f"{focus} {line_focus:.0f} m, {opp} {line_opp:.0f} m. Position-only, always rendered.")
    return _svg_wrap(els, cap)


def svg_lane_occupation(hs: float, ce: float, wg: float, team: str) -> str:
    """Lane-occupation diagram: five vertical channels shaded by wing/half-space/centre share.

    The three grounded shares (wing/half-space/centre) fill the corresponding channels (each wing and
    each half-space carries half its share, split symmetrically). Position-only (always renders).
    """
    els = _svg_pitch_base()
    # channel edges in metres across the width: wing | halfspace | centre | halfspace | wing
    edges = [0.0, 13.6, 24.0, 44.0, 54.4, PITCH_WID]
    shares = [wg / 2, hs / 2, ce, hs / 2, wg / 2]
    peak = max(shares) or 1.0
    for (y_lo, y_hi), sh in zip(zip(edges[:-1], edges[1:]), shares, strict=True):
        xa, ya = pitch_to_svg(0, y_hi, _SVG_W, _SVG_H, _SVG_M)
        xb, yb = pitch_to_svg(PITCH_LEN, y_lo, _SVG_W, _SVG_H, _SVG_M)
        op = 0.12 + 0.55 * (sh / peak)
        els.append(f'<rect class="lane" x="{xa:.1f}" y="{ya:.1f}" width="{xb - xa:.1f}" '
                   f'height="{yb - ya:.1f}" style="fill-opacity:{op:.2f}"/>')
    for label, share, y_mid in (("wing", wg, (edges[0] + edges[1]) / 2),
                                ("half-space", hs, (edges[1] + edges[2]) / 2),
                                ("centre", ce, PITCH_WID / 2)):
        tx, ty = pitch_to_svg(PITCH_LEN / 2, y_mid, _SVG_W, _SVG_H, _SVG_M)
        els.append(f'<text class="pt" text-anchor="middle" x="{tx:.1f}" y="{ty:.1f}">'
                   f'{label} {round(share * 100)}%</text>')
    cap = (f"Where {team}'s shape lives across the pitch width (lane occupation, whole match): "
           f"half-space {round(hs * 100)}%, centre {round(ce * 100)}%, wing {round(wg * 100)}%. "
           f"Position-only, always rendered.")
    return _svg_wrap(els, cap)


# ==================================================================================================
# Oracle score (Sofascore, cache-only). Grounded + explicitly labelled; never a CV output.
# ==================================================================================================
def oracle_score(oracle_ref: int | None) -> tuple[int, int] | None:
    """Final score ``(home, away)`` for a Sofascore match id from the cached season dicts, or ``None``.

    Reads only the already-cached ``match_dicts`` JSON (no network). Returns ``None`` if the cache or
    the fixture is missing so the header degrades to "score withheld".
    """
    if oracle_ref is None or not ORACLE_DICTS.exists():
        return None
    dicts = json.loads(ORACLE_DICTS.read_text(encoding="utf-8"))
    for d in dicts:
        if d.get("id") == oracle_ref:
            hs = d.get("homeScore", {}).get("current")
            aw = d.get("awayScore", {}).get("current")
            if hs is not None and aw is not None:
                return int(hs), int(aw)
    return None


# ==================================================================================================
# Per-match identity + events panels (drawn from their own VALIDATED artifacts, labelled as such).
# ==================================================================================================
def _identity_html(match_id: str, focus: str, opp: str) -> str:
    """Identity section: Brighton's 20 named players story, or Liverpool's honest queued banner."""
    if match_id == "brighton_manutd":
        body = (
            "<p>Close-up jersey-number reads (Koshkina PARSeq recognizer) were wired onto tracks to "
            "name players from the broadcast, with no roster leak. <strong>20 distinct players "
            "named</strong> (11 Man Utd, 9 Brighton) from <strong>1,887</strong> gated close-up "
            "anchors, attached to <strong>177</strong> named fragments through a ReID margin gate.</p>"
            "<p>The read precision is <strong>sample-verified at 39/40 tiles (~97.5%)</strong> by "
            "human spot-check &mdash; the one miss was a cut-off crop. Every named track survives the "
            "<strong>propagation guard</strong>: 18 fragments carried two disagreeing numbers and were "
            "named as neither, rather than guessing.</p>"
            "<ul>"
            "<li><strong>In-squad:</strong> 20/20 named players are in the matchday squad "
            "(no fabricated names).</li>"
            "<li><strong>GK exclusion:</strong> zero goalkeeper false positives &mdash; the reader "
            "never mislabels a keeper as an outfield number.</li>"
            "<li><strong>Hero-shot concentration:</strong> ~5-6 back-numbers surface legible close-up "
            "reads per match (#8 Bruno Fernandes alone is ~82% of anchors). This names the players who "
            "get repeated close-up shots, not a uniform XI.</li>"
            "</ul>"
            "<p class=\"src\">Source: <code>results/identity/NAMED_TRACKS_koshkina.md</code> "
            "(sample-verified precision verdict, 2026-07-17).</p>")
        return _panel("Identity - named players", "id", "sample-verified", body)
    banner = (
        f"<div class=\"queued\"><strong>Identity layer queued (GPU busy).</strong> The close-up "
        f"jersey-number naming pipeline that named 20 players for Brighton has not yet been run for "
        f"{_html.escape(focus)} v {_html.escape(opp)} &mdash; the GPU is committed to the current "
        f"batch. When it lands it will attach the same three artifacts: gated close-up anchors "
        f"(kit + OCR-agreement precision guard), ReID-attached named fragments, and a propagation "
        f"guard that names disagreeing fragments as neither. No player names are claimed for this "
        f"match until then.</div>")
    return _panel("Identity - named players", "id", "queued", banner)


def _events_html(match_id: str) -> str:
    """Events section: Brighton's validated action-spotting probe, or Liverpool's not-run notice."""
    if match_id == "brighton_manutd":
        summ_path = Path("results/action_spotting_probe/brighton_manutd/summary.json")
        rows = ""
        if summ_path.exists():
            summ = json.loads(summ_path.read_text(encoding="utf-8"))
            oracle = {"Goal": 3, "Yellow card": 3, "Red card": 0, "Foul": 22, "Corner": 8}
            for cls, orc in oracle.items():
                got = summ["per_class"].get(cls, {}).get("count")
                rows += (f"<tr><td>{_html.escape(cls)}</td><td>{got}</td><td>{orc}</td></tr>")
        body = (
            "<p>Zero-shot action spotting (E2E-Spot, official SoccerNet-v2 weights, no training on "
            "this broadcast) validated against the cached Sofascore oracle. It directly answers the "
            "reviewer attack \"your report can't see goals\".</p>"
            f"<figure><table><thead><tr><th>event</th><th>detected</th><th>Sofascore</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
            "<figcaption>Whole-match counts at the 0.30 gate vs Sofascore.</figcaption></figure>"
            "<ul>"
            "<li><strong>Goals:</strong> the 3 real goals are the 3 highest-confidence Goal peaks "
            "(0.76-0.95, cleanly above a &lt;0.08 noise floor), correct half split 1 H1 / 2 H2.</li>"
            "<li><strong>Cards exact:</strong> yellow 3/3, red 0/0.</li>"
            "<li><strong>Shots:</strong> total 25/25 exact; the on/off-target split is noisy "
            "zero-shot &mdash; report aggregate shots only, never the split.</li>"
            "<li><strong>Boundaries stated:</strong> goal replays re-trigger the Goal head "
            "(~90 s after the real goal); a confidence gate + post-goal refractory window removes "
            "them. Offside is under-detected (3 vs 6).</li>"
            "</ul>"
            "<p class=\"src\">Source: <code>results/action_spotting_probe.md</code> + "
            "<code>results/action_spotting_probe/brighton_manutd/summary.json</code>.</p>")
        return _panel("Events - action spotting", "ev", "Sofascore-validated", body)
    banner = ("<div class=\"queued\"><strong>Event layer not yet run.</strong> The zero-shot "
              "action-spotting probe (goals / cards / shots vs the Sofascore oracle) that validated "
              "Brighton has not been run for this match. No event counts are claimed until it is.</div>")
    return _panel("Events - action spotting", "ev", "not run", banner)


def _panel(title: str, cls: str, chip: str, inner: str) -> str:
    """Wrap a custom (non-report_v2) section in the same section shell with a provenance chip."""
    return (f'<section class="prov-{cls}"><h2><span class="tag tag-{cls}">'
            f'{_html.escape(chip.upper())}</span> {_html.escape(title)}</h2>{inner}</section>')


# ==================================================================================================
# "What we don't claim" panel -- abstentions shown proudly.
# ==================================================================================================
def _limits_html(match_id: str, gate, focus: str, opp: str) -> str:
    """Honest-limits panel: the withheld absolute claims + the known measurement biases + gaps."""
    items = []
    if gate.comparative_only:
        items.append(f"<li><strong>Absolute ball volumes</strong> (pass counts, xT totals, regain "
                     f"counts): withheld. The ball track clears the {gate.coverage_pct:.0f}% coverage "
                     f"bar but the pass-recall proxy is {gate.recall_pct:.0f}% (below the 50% absolute "
                     f"bar), team-symmetric (spread {gate.symmetry_spread:.3f}) &mdash; so ball "
                     f"families render as team-vs-team shares/ratios only, never as totals.</li>")
    elif not gate.passed:
        items.append(f"<li><strong>All ball-derived detail</strong> (passing, ball-xT, pressing, "
                     f"counterpress): withheld &mdash; the ball track fails the evidence gate "
                     f"(coverage {gate.coverage_pct:.0f}%, recall {gate.recall_pct:.0f}%).</li>")
    items.append("<li><strong>Shots:</strong> no CV shot detector &mdash; shot counts appear in the "
                 "oracle appendix for context only, never as a CV output.</li>")
    items.append("<li><strong>Possession %:</strong> our space-control and trackable-frame possession "
                 "are position/coverage-biased by construction and invert the event oracle; they are "
                 "reported caveated, never gated.</li>")
    if match_id != "brighton_manutd":
        items.append(f"<li><strong>Player names:</strong> the identity layer has not been run for "
                     f"{_html.escape(focus)} v {_html.escape(opp)} &mdash; no players are named.</li>")
        items.append("<li><strong>Events:</strong> the action-spotting probe has not been run for "
                     "this match &mdash; no goals/cards/shots are claimed from video.</li>")
    body = f"<ul>{''.join(items)}</ul>"
    return _panel("What we don't claim (and why)", "abst", "abstentions", body)


# ==================================================================================================
# Validation banner -- one line each per validation lineage.
# ==================================================================================================
def _validation_banner(gate, competition: str) -> str:
    """Three-line 'what is validated against what' banner (oracle / FIFA calibration / GS-HOTA)."""
    orc = gate.oracle_source or "Sofascore"
    lines = [
        f"<li><span class=\"vk\">Team aggregates</span> validated against the <strong>{_html.escape(orc)}"
        f"</strong> event oracle (possession, completed passes, shots) &mdash; the like-for-like pass "
        f"gate + the documented possession inversion.</li>",
        "<li><span class=\"vk\">Defensive-line de-biasing</span> calibrated against <strong>FIFA "
        "PMSR</strong> per-phase lines on the World Cup set (pooled ~5 m error); carried here as a "
        "method, since PL matches have no FIFA analogue.</li>",
        "<li><span class=\"vk\">Identity layer lineage</span> measured on the public <strong>SoccerNet "
        "GS-HOTA</strong> benchmark (14.8 &rarr; 19.8 with the jersey reader, zero sequences hurt) "
        "&mdash; the same reader that names players in the identity section.</li>",
    ]
    return (f'<div class="vbanner"><p class="vk-h">Validated against &mdash; {_html.escape(competition)}'
            f'</p><ul>{"".join(lines)}</ul></div>')


# ==================================================================================================
# Assembly.
# ==================================================================================================
def _render_section(sec, position_only: bool) -> str:
    """Render one report_v2 Section (heading + provenance chip + blocks) to HTML."""
    prov = sec.provenance.lower()
    chip = "position-only" if position_only else prov.upper()
    title = _html.escape(sec.title).replace(" -- ", " &mdash; ")
    parts = [f'<section class="prov-{prov}"><h2><span class="tag tag-{prov}">'
             f'{_html.escape(chip)}</span> {title}</h2>']
    parts += [_html_block(b) for b in sec.blocks]
    parts.append("</section>")
    return "".join(parts)


def render(match_id: str, generated: str) -> str:
    """Render the full self-contained HTML scouting pack for one registered match.

    Args:
        match_id: Registered match id (``core.registry``).
        generated: Generation date string baked into the footer (NOT a runtime JS clock).

    Returns:
        A single self-contained HTML document string.
    """
    m = get(match_id)
    focus, opp = m.teams[0], m.teams[1]
    sections, gate, facts, audit = build_document(match_id)
    cv = facts["cv"]
    ref = (json.loads(Path(f"outputs/eval/{match_id}_ball_eval.json").read_text(encoding="utf-8"))
           .get("oracle_ref") if Path(f"outputs/eval/{match_id}_ball_eval.json").exists() else None)
    score = oracle_score(ref)
    competition = getattr(m, "competition", "") or "match"

    # --- header ---
    hk, ak = _kit(m.home_team or focus), _kit(m.away_team or opp)
    home, away = m.home_team or focus, m.away_team or opp
    score_html = (f'<span class="score">{score[0]}&ndash;{score[1]}</span>'
                  f'<span class="score-src">Sofascore</span>' if score else
                  '<span class="score-src">score withheld (no oracle)</span>')
    header = (f'<header><div class="teams">{hk}<span class="vs">{score_html}</span>{ak}</div>'
              f'<p class="fixture">{_html.escape(home)} v {_html.escape(away)} '
              f'&middot; {_html.escape(competition)}</p></header>')

    # --- gate banner ---
    tlabel, tcls = tier_badge(gate.tier)
    gate_banner = (
        f'<div class="gate"><span class="gbadge {tcls}">{tlabel}</span>'
        f'<div class="gtext"><strong>Ball-evidence gate:</strong> post-link coverage '
        f'{gate.coverage_pct:.0f}% (needs &ge; 40%), pass-recall proxy {gate.recall_pct:.0f}% '
        f'(needs &ge; 50% for absolute). <strong>Body guardrail:</strong> '
        f'{audit["precision"] * 100:.0f}% precision ({audit["n_backed"]}/{audit["n_numbers"]} '
        f'numbers grounded). Gate inputs: '
        f'<code>outputs/eval/{_html.escape(match_id)}_ball_eval.json</code>.</div></div>')

    # --- body: report_v2 sections + injected SVG diagrams + identity/events panels ---
    body: list[str] = [header, _validation_banner(gate, competition), gate_banner]
    # sections[0]=focus setup, [1]=opponent structural, [2]=possession, [3]=defence, [4]=seams,
    # [-1]=validation appendix (body=False). Position-only = the two structural sections.
    body.append(_render_section(sections[0], position_only=True))
    body.append(_render_section(sections[1], position_only=True))
    body.append(_pitch_panel(cv, focus, opp))
    for sec in sections[2:5]:
        body.append(_render_section(sec, position_only=False))
    body.append(_identity_html(match_id, focus, opp))
    body.append(_events_html(match_id))
    body.append(_limits_html(match_id, gate, focus, opp))
    body.append(_render_section(sections[-1], position_only=False))

    body.append(
        f'<footer>football-synthesizer scouting pack &middot; match <code>'
        f'{_html.escape(match_id)}</code> &middot; metrics version {METRICS_VERSION} &middot; '
        f'generated {_html.escape(generated)}.<br>'
        f'Body = our computer-vision metrics, rendered only where the evidence gate clears; '
        f'oracle numbers appear in the appendix for validation only. Numbers not shown are '
        f'withheld on purpose &mdash; see "what we don\'t claim".</footer>')

    title = f"{focus} v {opp} - scouting pack"
    return _TEMPLATE.format(title=_html.escape(title), css=_CSS, body="\n".join(body))


def _pitch_panel(cv: dict, focus: str, opp: str) -> str:
    """Two inline SVG pitch diagrams (line heights + focus lane occupation), position-only facts."""
    diags = []
    lf = _get(cv, "line_height", focus, "def_line_debiased_m")
    lo = _get(cv, "line_height", opp, "def_line_debiased_m")
    if lf is not None and lo is not None:
        diags.append(svg_line_heights(lf, lo, focus, opp))
    hs = _get(cv, "theory", focus, "halfspace_share")
    ce = _get(cv, "theory", focus, "centre_share")
    wg = _get(cv, "theory", focus, "wing_share")
    if None not in (hs, ce, wg):
        diags.append(svg_lane_occupation(hs, ce, wg, focus))
    if not diags:
        return ""
    return (f'<section class="prov-cv"><h2><span class="tag tag-cv">position-only</span> '
            f'Pitch diagrams</h2><div class="pitchgrid">{"".join(diags)}</div></section>')


def _get(cv: dict, *path: str):
    """Safe nested lookup into the CV fact bundle."""
    node = cv
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def _kit(team: str) -> str:
    """Render a labelled kit chip (swatch + team name) for a team."""
    k = TEAM_KITS.get(team, _DEFAULT_KIT)
    if k.striped:
        fill = (f'background:repeating-linear-gradient(90deg,{k.primary} 0 6px,'
                f'{k.secondary} 6px 12px);')
    else:
        fill = f'background:{k.primary};border:2px solid {k.secondary};'
    return (f'<span class="kit"><span class="swatch" style="{fill}"></span>'
            f'{_html.escape(team)}</span>')


# ==================================================================================================
# Inline CSS (dark default, prefers-color-scheme light) + template.
# ==================================================================================================
_CSS = """
:root{color-scheme:dark light;
 --bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#98a2b3;--line:#262c35;
 --accent:#2dd4bf;--accent2:#7dd3fc;--abst:#f0787a;--cmp:#5eb3ea;--pass:#5bd88a;
 --pf:#12331f;--pl:#3a6b4c;}
@media (prefers-color-scheme:light){:root{
 --bg:#f4f6f8;--panel:#ffffff;--ink:#141922;--muted:#5a6472;--line:#e2e6ec;
 --accent:#0f766e;--accent2:#0369a1;--abst:#b03030;--cmp:#0a5a8a;--pass:#1a7a3a;
 --pf:#dcece1;--pl:#8fb79e;}}
*{box-sizing:border-box;}
html{-webkit-text-size-adjust:100%;}
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);
 margin:0;line-height:1.55;font-size:16px;}
main{max-width:860px;margin:0 auto;padding:32px 20px 64px;}
header{border-bottom:2px solid var(--line);padding-bottom:16px;margin-bottom:8px;}
.teams{display:flex;align-items:center;gap:18px;flex-wrap:wrap;}
.kit{display:inline-flex;align-items:center;gap:9px;font-size:22px;font-weight:700;
 letter-spacing:-.01em;}
.swatch{width:20px;height:26px;border-radius:3px;display:inline-block;}
.vs{display:flex;flex-direction:column;align-items:center;min-width:64px;}
.score{font-size:30px;font-weight:800;letter-spacing:.02em;}
.score-src{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.fixture{color:var(--muted);margin:10px 0 0;font-size:14px;}
h2{font-size:19px;margin:30px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line);
 letter-spacing:-.01em;}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:4px 20px 16px;margin:16px 0;}
section h2{margin-top:16px;}
p{margin:10px 0;}
a,code{color:var(--accent2);}
code{font-size:.86em;background:rgba(125,211,252,.10);padding:1px 5px;border-radius:4px;
 word-break:break-word;}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.06em;padding:2px 8px;
 border-radius:5px;vertical-align:middle;margin-right:8px;text-transform:uppercase;}
.tag-cv,.tag-id{background:rgba(45,212,191,.16);color:var(--accent);}
.tag-fifa,.tag-oracle{background:rgba(125,211,252,.14);color:var(--accent2);}
.tag-ev{background:rgba(125,211,252,.14);color:var(--accent2);}
.tag-abst{background:rgba(240,120,122,.16);color:var(--abst);}
.vbanner{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);
 border-radius:0 10px 10px 0;padding:12px 18px;margin:16px 0;}
.vk-h{font-weight:700;margin:2px 0 6px;font-size:14px;}
.vk{color:var(--accent);font-weight:600;}
.vbanner ul{margin:4px 0 2px;padding-left:20px;}
.vbanner li{font-size:13.5px;color:var(--muted);margin:4px 0;}
.gate{display:flex;gap:14px;align-items:flex-start;background:var(--panel);border:1px solid var(--line);
 border-radius:10px;padding:14px 18px;margin:16px 0;}
.gtext{font-size:13.5px;color:var(--muted);}
.gbadge{flex:none;font-size:11px;font-weight:800;letter-spacing:.04em;padding:5px 10px;
 border-radius:6px;text-transform:uppercase;white-space:nowrap;max-width:150px;line-height:1.25;}
.gbadge.t-abs{background:rgba(91,216,138,.18);color:var(--pass);}
.gbadge.t-cmp{background:rgba(94,179,234,.18);color:var(--cmp);}
.gbadge.t-abst{background:rgba(240,120,122,.18);color:var(--abst);}
ul{margin:8px 0 12px;padding-left:22px;}
li{margin:4px 0;}
.abstain-box{border-left:4px solid var(--abst);background:rgba(240,120,122,.08);padding:10px 14px;
 margin:12px 0;border-radius:0 8px 8px 0;font-size:14px;}
.cmp-box{border-left:4px solid var(--cmp);background:rgba(94,179,234,.09);padding:10px 14px;
 margin:12px 0;border-radius:0 8px 8px 0;font-size:14px;}
.queued{border-left:4px solid var(--accent2);background:rgba(125,211,252,.08);padding:12px 16px;
 border-radius:0 8px 8px 0;font-size:14.5px;}
.seam{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:0 8px 8px 0;
 padding:12px 16px;margin:12px 0;background:rgba(45,212,191,.05);}
.seam-claim{margin:0 0 6px;font-weight:600;}
.seam-back,.seam-fals{margin:4px 0;font-size:13.5px;color:var(--muted);}
table{border-collapse:collapse;width:100%;margin:10px 0 4px;font-size:14px;}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;}
th{background:rgba(125,211,252,.07);font-weight:600;}
figcaption{color:var(--muted);font-size:12.5px;font-style:italic;margin:6px 0 14px;}
.src{font-size:12.5px;color:var(--muted);}
.pitchgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
figure.pitch{margin:6px 0;}
figure.pitch svg{width:100%;height:auto;background:var(--pf);border-radius:6px;}
.pf{fill:var(--pf);stroke:var(--pl);stroke-width:1.5;}
.pl{stroke:var(--pl);stroke-width:1.3;fill:none;}
.lnf{stroke:var(--accent);stroke-width:2.6;}
.lno{stroke:var(--accent2);stroke-width:2.6;stroke-dasharray:5 4;}
.lane{fill:var(--accent);}
.pt{fill:var(--ink);font-size:11px;font-weight:600;font-family:'Segoe UI',sans-serif;}
footer{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);
 font-size:12.5px;}
details.appx{margin:8px 0 4px;}
details.appx>summary{cursor:pointer;list-style:none;padding:11px 2px;font-weight:600;
 font-size:19px;letter-spacing:-.01em;color:var(--ink);border-bottom:1px solid var(--line);}
details.appx>summary::-webkit-details-marker{display:none;}
details.appx>summary::before{content:"\25B8";color:var(--accent2);margin-right:9px;font-size:14px;}
details.appx[open]>summary::before{content:"\25BE";}
details.appx>summary .hint{color:var(--muted);font-weight:400;font-size:13px;margin-left:8px;}
@media (max-width:560px){.kit{font-size:18px;}.score{font-size:24px;}
 .gate{flex-direction:column;}main{padding:20px 14px 48px;}}
"""

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style></head>
<body><main>
{body}
</main></body></html>
"""


# ==================================================================================================
# CLI.
# ==================================================================================================
def generate(match_id: str, *, out_dir: Path = OUT_DIR, generated: str | None = None) -> Path:
    """Render one match to ``<out_dir>/<match_id>.html`` and return the path."""
    gen = generated or date.today().isoformat()
    html = render(match_id, gen)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{match_id}.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    """CLI: render one match (or a comma list) to a self-contained HTML scouting pack."""
    ap = argparse.ArgumentParser(description="Self-contained HTML scouting-pack renderer.")
    ap.add_argument("--match", default="manutd_liverpool",
                    help="registered match id, comma-separated ids, or 'all' (brighton + liverpool)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="generation date baked into the footer (default: today)")
    args = ap.parse_args()
    ids = (["brighton_manutd", "manutd_liverpool"] if args.match == "all"
           else [s.strip() for s in args.match.split(",")])
    for mid in ids:
        out = generate(mid, out_dir=Path(args.out_dir), generated=args.date)
        print(f"[render_html] {mid} -> {out}")


if __name__ == "__main__":
    main()
