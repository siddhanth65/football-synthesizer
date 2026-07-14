"""Report v2: a CV-PRIMARY grounded tactical read of a match (national-team or club).

The report is about the *focus team* -- ``registry.get(match_id).teams[0]`` (France for the WC
fixtures, Man Utd for the ``brighton_manutd`` PL pilot). National-team matches carry a France roster
and a FIFA PMSR oracle; club matches carry neither, so the report degrades to team-level prose and
validates against the Sofascore oracle instead (``tools.oracle``). The ball-evidence gate, the
CV-only body and the numeric guardrail behave identically in both cases.


The end-goal deliverable. Where the v1 pundit (``report/pundit.py``) built its narrative spine on FIFA
PMSR numbers (xG, build-up %, line breaks), **report v2 inverts the relationship**: the body is written
entirely from *our* computer-vision metrics (``outputs/facts/<match>.json`` -> ``facts["cv"]``) plus
roster names, and FIFA PMSR appears ONLY in a clearly separated validation appendix as an oracle we
check ourselves against -- never as content.

Three things make it honest rather than merely fluent:

* **Confidence gating.** Ball-derived families (passing, ball-xT, transitions, set pieces, counterpress)
  render only when the match clears a pre-declared evidence gate (post-``link_ball`` coverage >= 40 %
  AND pass-recall proxy >= 50 %). Otherwise the report prints an explicit *abstention* line. Position /
  shape families never depend on the ball, so they always render. Senegal clears the gate; Iraq and
  Norway abstain on the ball families -- which is the point.
* **Counter-structure seams.** The new analytical piece: concrete "how to play against France" ideas,
  each tied to one grounded CV number with a one-line falsification condition.
* **Numeric guardrail.** Every number in the body (sections 1-2) is audited against the grounded set
  (the CV fact store + this match's declared validation inputs) by ``report.guardrail``; the body must
  pass at 100 % precision or generation fails loudly.

CLI::

    python -m report.report_v2 --match france_senegal [--out results/report_v2_france_senegal.md]
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from core.registry import Match, get
from report import guardrail
from report.facts import load_facts

# --- Pre-declared evidence gate -------------------------------------------------------------------
# Ball-derived metric families render only when BOTH thresholds are met. These are policy constants,
# fixed before looking at any match, so gating cannot be tuned per result.
GATE_COVERAGE_MIN_PCT = 40.0   # post-link_ball usable-track coverage
GATE_RECALL_MIN_PCT = 50.0     # pass-recall proxy vs annotated volume

# --- Declared validation inputs (provenance, NOT re-derivable from the fact store) ----------------
# The ball-evidence gate (coverage + pass-recall proxy) is now read PER MATCH from a persisted eval
# artifact -- ``outputs/eval/<match>_ball_eval.json`` (see ``evaluate_gate``); GATE_INPUTS below is a
# fallback source for those two numbers AND the sole home of the FIFA-only validation constants that
# only national-team matches can cite (they have no PL analogue):
# post_link_coverage_pct / pass_recall_pct: fallback for the gate when the artifact is missing.
# phase_c6_mae_pp: tools phase-C6 mean-absolute-error of our phase classifier vs FIFA (percentage pts).
# def_line_error_m: pooled de-biased defensive-line error vs FIFA per-phase (P2 validation).
# c5_n_obs: team-matches pooled into the C5 opponent model.
# These are cited transparently and form part of the report's grounded set alongside facts["cv"].
GATE_INPUTS: dict[str, dict[str, float]] = {
    "france_iraq": {"post_link_coverage_pct": 36.0, "pass_recall_pct": 22.8,
                    "phase_c6_mae_pp": 16.2, "def_line_error_m": 5.2, "c5_n_obs": 8},
    "france_senegal": {"post_link_coverage_pct": 46.0, "pass_recall_pct": 55.0,
                       "phase_c6_mae_pp": 9.7, "def_line_error_m": 5.2, "c5_n_obs": 8},
    "france_norway": {"post_link_coverage_pct": 38.0, "pass_recall_pct": 10.6,
                      "phase_c6_mae_pp": 14.3, "def_line_error_m": 5.2, "c5_n_obs": 8},
}

# Per-match ball-evidence artifact directory (coverage + pass-recall proxy + oracle provenance).
BALL_EVAL_DIR = Path("outputs/eval")

# Metric families that depend on the ball track and are therefore gated.
BALL_FAMILIES = ("passing", "ball_xt", "transitions", "set_pieces", "counterpress",
                 "pressing", "verticality")

# Formation labels ("4-2-3-1") are categorical CV outputs, not quantitative metrics; strip them before
# the numeric guardrail so their digits are not mistaken for measured numbers.
_FORMATION_RE = re.compile(r"\d(?:-\d){2,3}")


# ==================================================================================================
# Structured document model -- rendered to both Markdown and self-contained HTML.
# ==================================================================================================
@dataclass
class Block:
    """One renderable unit within a section."""

    kind: str                       # "p" | "bullets" | "abstain" | "table" | "seam"
    text: str = ""
    items: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    # seam-only fields
    claim: str = ""
    backing: str = ""
    falsifies: str = ""


@dataclass
class Section:
    """A titled group of blocks with a provenance tag ('CV' body vs 'FIFA' appendix)."""

    title: str
    provenance: str                 # "CV" | "FIFA" | "ORACLE"
    blocks: list[Block] = field(default_factory=list)
    body: bool = True               # audited iff True (sections 1-2)


@dataclass(frozen=True)
class Ctx:
    """Per-match rendering context: who the report is about and which oracle validates it.

    ``focus`` is ``registry.teams[0]`` (France / Man Utd) and keys every CV family; ``opponent`` is
    ``teams[1]``. ``has_fifa`` selects the FIFA-PMSR path (roster prose, C5 seam, FIFA appendix) vs
    the club path (team-level prose, position-only fifth seam, Sofascore appendix).
    """

    match_id: str
    focus: str
    opponent: str
    has_fifa: bool
    roster: dict = field(default_factory=dict)
    oracle_name: str = "FIFA PMSR"
    oracle_ref: int | None = None
    home_team: str | None = None
    away_team: str | None = None


# ==================================================================================================
# Small formatting helpers (guardrail-friendly: numbers carry their correct unit token).
# ==================================================================================================
def _pct(x: float) -> str:
    """A 0-1 share as an integer percentage string (e.g. 0.239 -> '24%')."""
    return f"{round(x * 100)}%"


def _pctv(x: float) -> str:
    """A 0-100 value already in percent units as an integer percentage string."""
    return f"{round(x)}%"


def _m(x: float) -> str:
    """A metres value rounded to a whole metre (e.g. 30.04 -> '30 m')."""
    return f"{round(x)} m"


def _one(x: float) -> str:
    """A value to one decimal place (bare)."""
    return f"{x:.1f}"


# ==================================================================================================
# Gate.
# ==================================================================================================
@dataclass(frozen=True)
class GateResult:
    """Outcome of the pre-declared ball-evidence gate for one match."""

    passed: bool
    coverage_pct: float
    recall_pct: float
    oracle_source: str = ""
    provenance: str = "artifact"    # "artifact" | "constant" (fallback)

    @property
    def abstention(self) -> str:
        """The explicit withheld-evidence line shown in place of a gated family."""
        return (f"insufficient ball-track evidence (coverage {self.coverage_pct:.0f}%, "
                f"recall proxy {self.recall_pct:.1f}%) -- withheld.")

    @property
    def coverage_clear_recall_short(self) -> bool:
        """True when coverage clears its bar but recall alone (narrowly) misses -- the demo case."""
        return (not self.passed and self.coverage_pct >= GATE_COVERAGE_MIN_PCT
                and self.recall_pct < GATE_RECALL_MIN_PCT)


def gate_decision(coverage_pct: float, recall_pct: float) -> bool:
    """Pure gate rule: ball families render iff coverage AND recall clear the pre-declared minima."""
    return coverage_pct >= GATE_COVERAGE_MIN_PCT and recall_pct >= GATE_RECALL_MIN_PCT


def load_ball_eval(match_id: str, *, eval_dir: Path = BALL_EVAL_DIR) -> dict | None:
    """Read the persisted per-match ball-evidence artifact, or ``None`` if it is absent."""
    path = eval_dir / f"{match_id}_ball_eval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gate(match_id: str, *, eval_dir: Path = BALL_EVAL_DIR) -> GateResult:
    """Apply the pre-declared coverage/recall gate, reading the per-match eval artifact first.

    Coverage and pass-recall proxy come from ``outputs/eval/<match>_ball_eval.json`` (fractions in
    ``[0, 1]``). If that artifact is missing the function falls back to the declared ``GATE_INPUTS``
    constant with an ASCII warning, so an un-persisted match still renders rather than crashing.
    """
    art = load_ball_eval(match_id, eval_dir=eval_dir)
    if art is not None:
        cov = float(art["post_link_coverage"]) * 100.0
        rec = float(art["pass_recall_proxy"]) * 100.0
        return GateResult(passed=gate_decision(cov, rec), coverage_pct=cov, recall_pct=rec,
                          oracle_source=str(art.get("oracle_source", "")), provenance="artifact")
    gi = GATE_INPUTS[match_id]
    cov, rec = gi["post_link_coverage_pct"], gi["pass_recall_pct"]
    print(f"[report_v2] WARNING: no ball-eval artifact for {match_id}; "
          f"falling back to declared GATE_INPUTS constant.")
    return GateResult(passed=gate_decision(cov, rec), coverage_pct=cov, recall_pct=rec,
                      oracle_source="", provenance="constant")


# ==================================================================================================
# Grounded set for the guardrail = CV fact store + this match's declared validation inputs.
# ==================================================================================================
def _audit_facts(facts: dict, match_id: str) -> dict:
    """CV fact store augmented with declared validation constants, for the numeric guardrail.

    The report's honesty guarantee: every number in the body is either a persisted CV metric or one of
    a small, named set of validation inputs (coverage, recall, de-biased-line error, C5 obs count). We
    expose those inputs under ``cv`` so the guardrail (which grounds CV + FIFA) treats them as backed;
    fabricated numbers still match nothing.
    """
    gate = evaluate_gate(match_id)
    val = GATE_INPUTS.get(match_id)
    af = deepcopy(facts)
    cv = af.setdefault("cv", {})
    # 0-1 "share" labels auto-expand to a percentage form inside guardrail.flatten_facts.
    cv["_gate"] = {"coverage_share": gate.coverage_pct / 100.0,
                   "recall_share": gate.recall_pct / 100.0}
    # FIFA-only validation constants are cited in the body ONLY for national-team matches; expose them
    # to the guardrail only when they exist (club matches have no FIFA per-phase line / C5 model).
    if val is not None:
        cv["_val"] = {"def_line_error_m": val["def_line_error_m"], "c5_n_obs": int(val["c5_n_obs"])}
    af.pop("fifa", None)  # body must not be backed by FIFA -- audit against CV + validation only
    return af


def audit_body(sections: list[Section], facts: dict, match_id: str) -> dict:
    """Run the numeric guardrail over the body sections; return precision + unbacked findings."""
    text = _FORMATION_RE.sub("<formation>", body_text(sections))
    result = guardrail.audit(text, _audit_facts(facts, match_id))
    n, backed = result["n_numbers"], result["n_backed"]
    result["precision"] = (backed / n) if n else 1.0
    return result


def body_text(sections: list[Section]) -> str:
    """Concatenate every numeric-bearing string from the audited (body) sections."""
    parts: list[str] = []
    for s in sections:
        if not s.body:
            continue
        for b in s.blocks:
            parts += [b.text, b.claim, b.backing, b.falsifies, b.caption]
            parts += b.items
            parts += [c for row in b.rows for c in row]
    return "  ".join(p for p in parts if p)


# ==================================================================================================
# Body builders (CV-primary). Every number pulled from facts["cv"] or a declared validation input.
# ==================================================================================================
def _fr(cv: dict, *path: str, default=None):
    """Safe nested lookup for France's value within the CV bundle."""
    node = cv
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def _setup_section(cv: dict, ctx: Ctx) -> Section:
    """(a) How the focus team set up -- formation, de-biased line, size, lanes. Position-only."""
    sec = Section(f"How {ctx.focus} set up", "CV")
    ten = _fr(cv, "tendencies", ctx.focus) or {}
    form = _fr(cv, "formation", ctx.focus, "formation") or ctx.roster.get("formation", "")
    nominal = ctx.roster.get("formation", "")
    line = _fr(cv, "line_height", ctx.focus, "def_line_debiased_m")
    build = _fr(ten, "buildup_height", "mean")
    width = _fr(ten, "width", "mean")
    length = _fr(ten, "length", "mean")
    comp = _fr(ten, "compactness", "mean")
    note = "" if not nominal or nominal == form else f" (nominal {nominal})"
    sec.blocks.append(Block("p", text=(
        f"Our shape classifier reads {ctx.focus} in a **{form}**{note}. The structural read below is "
        f"measured directly from tracked positions and does not depend on the ball, so it renders for "
        f"every match regardless of ball-track quality.")))
    parts = []
    if line is not None and ctx.has_fifa:
        parts.append(f"their visibility-corrected defensive line sits **{_m(line)}** up the pitch "
                     f"(the raw broadcast line is censoring-inflated; the de-biased value is validated "
                     f"to within about **5 m** of FIFA's per-phase lines)")
    elif line is not None:
        parts.append(f"their visibility-corrected defensive line sits **{_m(line)}** up the pitch "
                     f"(the raw broadcast line is censoring-inflated; the visibility de-biasing was "
                     f"validated on the World Cup set, where FIFA per-phase lines exist)")
    if build is not None:
        parts.append(f"they build from a base line around **{_m(build)}**")
    if width is not None and length is not None:
        parts.append(f"the block is **{_m(width)}** wide and **{_m(length)}** deep")
    if comp is not None:
        parts.append(f"nearest-team-mate compactness holds near **{_one(comp)}** (a spread index, "
                     "lower is tighter)")
    if parts:
        sec.blocks.append(Block("p", text="In shape terms, " + "; ".join(parts) + "."))
    hs = _fr(cv, "theory", ctx.focus, "halfspace_share")
    ce = _fr(cv, "theory", ctx.focus, "centre_share")
    wg = _fr(cv, "theory", ctx.focus, "wing_share")
    if None not in (hs, ce, wg):
        sec.blocks.append(Block("bullets", text="Where the shape lives (lane occupation, whole match):",
                                items=[
            f"half-spaces **{_pct(hs)}** -- the dominant channel",
            f"centre **{_pct(ce)}**",
            f"wings **{_pct(wg)}** -- comparatively thin, a structural handle for opponents",
        ]))
    return sec


def _possession_section(cv: dict, ctx: Ctx, gate: GateResult) -> Section:
    """(b) In possession -- verticality, ball-xT, tempo/PPDA, phases, style. Mostly gated."""
    sec = Section("In possession", "CV")
    sync = _fr(cv, "style", "velocity_synchrony", ctx.focus)
    if sync is not None:
        sec.blocks.append(Block("p", text=(
            f"{ctx.focus} move as a unit -- velocity synchrony **{_pct(sync)}** on a 0-1 scale (this "
            f"is a position-only measure and always renders).")))
    if not gate.passed:
        sec.blocks.append(Block("abstain", text=(
            "Ball-dependent possession detail (tempo, PPDA, ball-xT, phase split, verticality): "
            + gate.abstention)))
        return sec
    ph = _fr(cv, "phases_pct", ctx.focus) or {}
    if ph:
        sec.blocks.append(Block("p", text=(
            f"By our own phase classifier {ctx.focus} spend **{_pctv(ph['build_up'])}** of "
            f"in-possession time in build-up, **{_pctv(ph['progression'])}** in progression and "
            f"**{_pctv(ph['final_third'])}** in the final third -- a side that wants to carry the ball "
            f"forward, not launch it.")))
    vert = _fr(cv, "theory", ctx.focus, "verticality")
    xt = _fr(cv, "ball_xt", ctx.focus, "xt_created")
    nmoves = _fr(cv, "ball_xt", ctx.focus, "n_moves")
    lb = _fr(cv, "theory", ctx.focus, "line_breaks")
    bits = []
    if vert is not None:
        bits.append(f"goalward directness (verticality) measures **{_pct(vert)}** of maximum")
    if lb is not None:
        bits.append(f"tracking caught **{lb} defensive-line breaks**")
    if xt is not None and nmoves is not None:
        bits.append(f"ball progression was worth **{xt:.1f} xG-equivalent** expected threat (xT) over "
                    f"**{nmoves}** tracked advances")
    if bits:
        sec.blocks.append(Block("p", text="With the ball, " + "; ".join(bits) + "."))
    pa = _fr(cv, "passing", ctx.focus) or {}
    if pa.get("n_passes") is not None:
        sec.blocks.append(Block("p", text=(
            f"Passing volume in the usable ball track: **{pa['n_passes']} passes** at a short mean of "
            f"**{_one(pa['mean_pass_m'])} m**, with a PPDA proxy of **{_one(pa['ppda'])}** opponent "
            f"build-up passes per defensive action ({pa['pressures']} pressures tracked) -- an "
            f"aggressive, high-tempo profile.")))
    return sec


def _defence_section(cv: dict, ctx: Ctx, gate: GateResult) -> Section:
    """(c) Out of possession -- pressing, counterpress decay, regains, line height. Mostly gated."""
    sec = Section("Out of possession", "CV")
    line = _fr(cv, "line_height", ctx.focus, "def_line_debiased_m")
    if line is not None:
        sec.blocks.append(Block("p", text=(
            f"{ctx.focus} defend from a high starting point -- the visibility-corrected line at "
            f"**{_m(line)}** is a front-foot posture (position-only, always rendered).")))
    if not gate.passed:
        sec.blocks.append(Block("abstain", text=(
            "Ball-dependent pressing detail (pressing intensity, counterpress decay curve, regains): "
            + gate.abstention)))
        return sec
    pi = _fr(cv, "theory", ctx.focus, "pressing_intensity")
    if pi is not None:
        sec.blocks.append(Block("p", text=(
            f"On the ball-tracked chunks their pressing intensity -- a 0-1 time-to-intercept index -- "
            f"runs at **{_pct(pi)}** of maximum.")))
    curve = _fr(cv, "theory", ctx.focus, "counterpress_regain_curve") or {}
    tr = _fr(cv, "transitions", ctx.focus) or {}
    if curve:
        sec.blocks.append(Block("p", text=(
            f"Their counterpress decays predictably: after a loss {ctx.focus} regain the ball within "
            f"3 s "
            f"**{_pctv(curve['3s'] * 100)}** of the time, **{_pctv(curve['5s'] * 100)}** within 5 s "
            f"and **{_pctv(curve['8s'] * 100)}** within 8 s -- the curve flattens after the 5 s "
            f"window.")))
    if tr.get("counterpress_rate") is not None:
        sec.blocks.append(Block("p", text=(
            f"They commit hard to it -- counterpress rate **{_pct(tr['counterpress_rate'])}** of "
            f"losses at a mean re-engagement of **{_one(tr['mean_recovery_s'])} s** -- but convert "
            f"only **{tr['high_regains']} high regains** from **{tr['turnovers_lost']} losses**, so "
            f"much of the pressure delays rather than wins the ball high.")))
    return sec


def _seam(title: str, claim: str, backing: str, falsifies: str, family: str,
          gate: GateResult) -> Block:
    """Build one counter-structure seam, fully withholding it if it rests on a gated ball family."""
    if family in BALL_FAMILIES and not gate.passed:
        return Block("abstain", text=f"Seam withheld ({title}, ball-derived): {gate.abstention}")
    return Block("seam", claim=claim, backing=backing, falsifies=falsifies)


def _seams_section(cv: dict, ctx: Ctx, gate: GateResult) -> Section:
    """(d) Counter-structure seams -- how to play against the focus team, grounded + falsifiable."""
    f = ctx.focus
    sec = Section(f"Counter-structure seams -- how to play against {f}", "CV")
    sec.blocks.append(Block("p", text=(
        "Each seam pairs a concrete idea with the exact CV number behind it and a one-line condition "
        "that would close it. Seams resting on ball-tracked families abstain when the match fails the "
        "evidence gate.")))
    curve = _fr(cv, "theory", f, "counterpress_regain_curve") or {}
    tr = _fr(cv, "transitions", f) or {}
    hs = _fr(cv, "theory", f, "halfspace_share")
    ce = _fr(cv, "theory", f, "centre_share")
    wg = _fr(cv, "theory", f, "wing_share")
    line = _fr(cv, "line_height", f, "def_line_debiased_m")
    ten = _fr(cv, "tendencies", f) or {}
    build = _fr(ten, "buildup_height", "mean")
    a3 = _fr(ten, "attacking_third_share", "mean")

    if curve:
        sec.blocks.append(_seam(
            title="counterpress-timing window",
            claim=(f"Retain through the first press and play forward within about 5 s. {f} recover "
                   f"**{_pctv(curve['5s'] * 100)}** of their losses inside 5 s but the curve flattens "
                   f"to **{_pctv(curve['8s'] * 100)}** by 8 s -- beating the initial counterpress buys "
                   "a clean progression window."),
            backing=(f"counterpress regain curve {_pctv(curve['3s'] * 100)} (3 s) -> "
                     f"{_pctv(curve['5s'] * 100)} (5 s) -> {_pctv(curve['8s'] * 100)} (8 s), over "
                     f"{tr.get('counterpress_losses', tr.get('turnovers_lost', 0))} tracked losses."),
            falsifies=(f"{f}'s 5 s regain rate rises well above {_pctv(curve['5s'] * 100)} "
                       "-- a counterpress that no longer plateaus."),
            family="counterpress", gate=gate))
    if None not in (hs, ce, wg):
        sec.blocks.append(_seam(
            title="wide-lane underload",
            claim=(f"Attack and switch into the wide lanes. {f} concentrate centrally and in the "
                   f"half-spaces (**{_pct(hs)}** half-space, **{_pct(ce)}** centre) with only "
                   f"**{_pct(wg)}** in the wings -- isolating the full-backs stretches a compact "
                   "block."),
            backing=(f"lane occupation: half-space {_pct(hs)}, centre {_pct(ce)}, wing {_pct(wg)} "
                     f"({f}, whole match, position-only)."),
            falsifies=(f"{f}'s wing occupation climbs well above {_pct(wg)} toward their "
                       "central load."),
            family="lane", gate=gate))
    if line is not None:
        extra = f" behind a base line near {_m(build)}" if build is not None else ""
        val_note = ("validated to within about 5 m of FIFA per-phase; position-only" if ctx.has_fifa
                    else "visibility de-biasing validated on the World Cup set; position-only")
        sec.blocks.append(_seam(
            title="in-behind the high line",
            claim=(f"Target the space behind the defensive line. Visibility-corrected, {f}'s line "
                   f"sits **{_m(line)}** up the pitch{extra}; early, direct balls in behind exploit "
                   "that depth before the block resets."),
            backing=f"de-biased defensive line {_m(line)} ({val_note}).",
            falsifies=(f"{f} drop the line well below {_m(line)} toward their own half."),
            family="line", gate=gate))
    if tr.get("counterpress_rate") is not None:
        sec.blocks.append(_seam(
            title="transition after beating the press",
            claim=(f"Commit to the counter the instant you win it back. {f} pour into the "
                   f"counterpress (**{_pct(tr['counterpress_rate'])}** rate, "
                   f"**{_one(tr['mean_recovery_s'])} s** mean re-engagement) yet convert only "
                   f"**{tr['high_regains']}** of **{tr['turnovers_lost']}** losses into high regains -- "
                   "survive first contact and the pitch opens."),
            backing=(f"counterpress rate {_pct(tr['counterpress_rate'])}, mean recovery "
                     f"{_one(tr['mean_recovery_s'])} s, {tr['high_regains']} high regains vs "
                     f"{tr['turnovers_lost']} losses."),
            falsifies=(f"{f}'s high regains rise toward a third of their losses -- a "
                       "counterpress that wins the ball high rather than just delaying."),
            family="transitions", gate=gate))
    # Fifth seam: the C5 opponent-model territory idea is WC-pooled and applies ONLY to the national
    # team; club matches get a position-only compactness/width seam in its place (no C5 claim).
    if ctx.has_fifa and a3 is not None:
        sec.blocks.append(_seam(
            title="territory denial (C5)",
            claim=("Deny territory with a compact mid-block, not a deep retreat. Our pooled opponent "
                   f"model (pooled 8-obs model -- directional) shows {f} push further up the more "
                   f"you drop off; against this opponent's mid-depth line {f}'s attacking-third "
                   f"share held at **{_pct(a3)}**."),
            backing=(f"attacking-third share {_pct(a3)} (C5 opponent model, pooled 8-obs -- "
                     "directional)."),
            falsifies=(f"{f}'s attacking-third share stays near {_pct(a3)} no matter how "
                       "deep you defend -- i.e. the opponent term flattens."),
            family="shape", gate=gate))
    elif not ctx.has_fifa:
        width = _fr(ten, "width", "mean")
        length = _fr(ten, "length", "mean")
        comp = _fr(ten, "compactness", "mean")
        if None not in (width, length, comp):
            sec.blocks.append(_seam(
                title="stretch the compact block",
                claim=(f"Stretch the block before entering it. {f} hold a tight, narrow shape -- "
                       f"**{_m(width)}** wide by **{_m(length)}** deep at a nearest-team-mate spread "
                       f"of **{_one(comp)}** -- so pinning both flanks and switching fast forces the "
                       "gaps a compact block does not want to open."),
                backing=(f"block size {_m(width)} wide x {_m(length)} deep, compactness {_one(comp)} "
                         f"({f}, whole match, position-only)."),
                falsifies=(f"{f}'s block width climbs well above {_m(width)} -- a shape already "
                           "spread enough to defend the flanks."),
                family="shape", gate=gate))
    return sec


# ==================================================================================================
# Validation appendix (FIFA oracle) -- the only place FIFA numbers may appear.
# ==================================================================================================
def _fifa(facts: dict, france_idx: int, *path: str, default=None):
    """Fetch France's FIFA value from a nested PMSR path (last key indexed by team)."""
    node = facts.get("fifa", {})
    for p in path[:-1]:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    v = node.get(path[-1]) if isinstance(node, dict) else None
    if isinstance(v, list) and len(v) > france_idx:
        return v[france_idx]
    return default


def _validation_section(facts: dict, cv: dict, ctx: Ctx, france_idx: int) -> Section:
    """FIFA-vs-CV side-by-side for the metrics where both exist. Oracle only, not used above."""
    sec = Section("Validation against FIFA PMSR -- oracle only, not used above", "FIFA", body=False)
    gi = GATE_INPUTS[ctx.match_id]
    sec.blocks.append(Block("p", text=(
        "None of the numbers below feed the analysis above. FIFA's official PMSR is used purely to "
        "check our CV output where the two measure the same thing.")))

    # Possession (CV space-control proxy vs FIFA possession).
    sc = _fr(cv, "space", ctx.focus, "space_control")
    poss = _fifa(facts, france_idx, "key_stats", "possession_pct")
    if sc is not None and poss is not None:
        sec.blocks.append(Block("table", caption="Possession",
            headers=["metric", "CV (ours)", "FIFA", "note"],
            rows=[["possession share", f"{sc * 100:.0f}% (space-control proxy)", f"{poss:.1f}%",
                   "proxy, not a like-for-like count"]]))

    # Phase distribution + declared C6 MAE.
    ph = _fr(cv, "phases_pct", ctx.focus) or {}
    if ph:
        fbu = (_fifa(facts, france_idx, "phases", "build_up_unopposed") or 0) + \
              (_fifa(facts, france_idx, "phases", "build_up_opposed") or 0)
        fpr = _fifa(facts, france_idx, "phases", "progression") or 0
        ff3 = _fifa(facts, france_idx, "phases", "final_third") or 0
        tot = (fbu + fpr + ff3) or 1
        rows = [
            ["build-up", f"{ph['build_up']:.0f}%", f"{100 * fbu / tot:.0f}%", ""],
            ["progression", f"{ph['progression']:.0f}%", f"{100 * fpr / tot:.0f}%", ""],
            ["final third", f"{ph['final_third']:.0f}%", f"{100 * ff3 / tot:.0f}%", ""],
        ]
        sec.blocks.append(Block("table",
            caption=(f"In-possession phase split (FIFA normalised over the same three buckets). "
                     f"Validated C6 mean-abs-error {gi['phase_c6_mae_pp']:.1f} pp."),
            headers=["phase", "CV (ours)", "FIFA", ""], rows=rows))

    # Defensive line (pooled).
    line = _fr(cv, "line_height", ctx.focus, "def_line_debiased_m")
    dl = facts.get("fifa", {}).get("line_height", {}).get("defensive", {})
    if line is not None and dl:
        vals = [v[france_idx] for v in dl.values() if isinstance(v, list) and len(v) > france_idx]
        if vals:
            fifa_mean = sum(vals) / len(vals)
            sec.blocks.append(Block("table",
                caption=("Defensive line (pooled). FIFA per-phase blocks: "
                         + ", ".join(f"{k} {v[france_idx]} m" for k, v in dl.items()) + "."),
                headers=["metric", "CV (ours)", "FIFA", "error"],
                rows=[["de-biased defensive line", f"{line:.1f} m",
                       f"{fifa_mean:.1f} m (block mean)",
                       f"{abs(line - fifa_mean):.1f} m (pooled ~{gi['def_line_error_m']:.1f} m)"]]))

    # Pass volume + recall proxy.
    npass = _fr(cv, "passing", ctx.focus, "n_passes")
    fpass = _fifa(facts, france_idx, "key_stats", "passes")
    if npass is not None and fpass is not None:
        sec.blocks.append(Block("table",
            caption="Pass volume in the usable ball track vs FIFA total.",
            headers=["metric", "CV (ours)", "FIFA", "recall proxy"],
            rows=[["passes", f"{npass}", f"{fpass:.0f}", f"{gi['pass_recall_pct']:.1f}%"]]))
    return sec


# ==================================================================================================
# Validation appendix (Sofascore oracle) -- club matches with no FIFA PMSR.
# ==================================================================================================
def _fetch_oracle(ctx: Ctx) -> dict | None:
    """Per-side Sofascore aggregates for a club fixture (cache-first), keyed by ``side``.

    Returns ``None`` on any failure (missing reference, cache miss, offline) so report generation
    never crashes on the oracle; the appendix then prints an explicit unavailable note.
    """
    if ctx.oracle_ref is None:
        return None
    try:
        from tools.oracle import get_match_aggregates  # noqa: PLC0415

        agg = get_match_aggregates(int(ctx.oracle_ref), source="sofascore")
    except Exception as exc:  # noqa: BLE001 -- oracle is best-effort; degrade to a note
        print(f"[report_v2] oracle fetch failed for {ctx.match_id}: {exc}")
        return None
    return {t["side"]: t for t in agg["teams"]}


def _validation_oracle_section(cv: dict, ctx: Ctx) -> Section:
    """CV-vs-Sofascore side-by-side for a club match. Oracle only, never used in the body above."""
    sec = Section(f"Validation against {ctx.oracle_name} -- oracle only, not used above", "ORACLE",
                  body=False)
    sec.blocks.append(Block("p", text=(
        f"None of the numbers below feed the analysis above; the {ctx.oracle_name} oracle only checks "
        "our CV output where the two measure the same thing. Two honesty notes carry over from the "
        "pilot gate: (1) our possession figures are measured on trackable ball frames only, so they "
        "are biased by ball coverage and here INVERT the oracle -- our space-control hands the "
        "territory to the side the event oracle has with less of the ball; the separately diagnosed "
        "trackable ball-possession share inverts identically (Man Utd ~44%; "
        "results/pl_pilot/possession_diagnosis.md). (2) We have no CV shot detector, so shots are an "
        "open gap shown for context only, never claimed as a CV output.")))
    oracle = _fetch_oracle(ctx)
    if oracle is None:
        sec.blocks.append(Block("p", text=(
            "Oracle aggregates unavailable (no cached response); see results/pl_pilot/fbref_gate.md "
            "for the persisted gate table.")))
        return sec
    side_of = {ctx.home_team: "home", ctx.away_team: "away"}
    teams = [ctx.focus, ctx.opponent]

    def _o(team: str, key: str):
        row = oracle.get(side_of.get(team, ""), {})
        return row.get(key)

    # Possession: CV space-control proxy vs oracle possession (the documented inversion).
    poss_rows = []
    for team in teams:
        sc = _fr(cv, "space", team, "space_control")
        op = _o(team, "possession_pct")
        if sc is not None and op is not None:
            poss_rows.append([team, f"{sc * 100:.0f}% (space-control proxy)", f"{op:.0f}%",
                              "trackable-frame bias; inverts the oracle"])
    if poss_rows:
        sec.blocks.append(Block("table",
            caption="Possession -- CV proxy vs Sofascore (biased-by-construction, reported caveated).",
            headers=["team", "CV (ours)", "Sofascore", "note"], rows=poss_rows))

    # Pass volume + recall proxy (the like-for-like gate metric).
    pass_rows = []
    for team in teams:
        npass = _fr(cv, "passing", team, "n_passes")
        oc = _o(team, "passes_cmp")
        if npass is not None and oc:
            pass_rows.append([team, f"{npass}", f"{oc:.0f}", f"{100 * npass / oc:.1f}%"])
    if pass_rows:
        sec.blocks.append(Block("table",
            caption="Pass volume in the usable ball track vs Sofascore completed passes.",
            headers=["team", "CV (ours)", "Sofascore", "recall proxy"], rows=pass_rows))

    # Shots: honest gap -- no CV shot detector.
    shot_rows = []
    for team in teams:
        os = _o(team, "shots")
        if os is not None:
            shot_rows.append([team, "-- (no CV shot detector)", f"{os:.0f}", "honest gap"])
    if shot_rows:
        sec.blocks.append(Block("table",
            caption="Shots -- shown for context; we do not detect shots from broadcast video.",
            headers=["team", "CV (ours)", "Sofascore", "note"], rows=shot_rows))
    return sec


# ==================================================================================================
# Assemble.
# ==================================================================================================
def _context_for(match_id: str, facts: dict) -> Ctx:
    """Assemble the per-match rendering context from the registry, roster and eval artifact."""
    m: Match = get(match_id)
    roster = _roster_for(match_id)
    has_fifa = bool(facts.get("fifa"))
    art = load_ball_eval(match_id) or {}
    return Ctx(
        match_id=match_id,
        focus=m.teams[0],
        opponent=m.teams[1],
        has_fifa=has_fifa,
        roster=roster,
        oracle_name="FIFA PMSR" if has_fifa else "Sofascore",
        oracle_ref=art.get("oracle_ref"),
        home_team=m.home_team,
        away_team=m.away_team,
    )


def build_document(match_id: str) -> tuple[list[Section], GateResult, dict, dict]:
    """Build the full section list for a match. Returns (sections, gate, facts, audit)."""
    facts = load_facts(match_id)
    if not facts:
        raise FileNotFoundError(f"no fact store for {match_id}; run report.facts first")
    cv = facts["cv"]
    ctx = _context_for(match_id, facts)
    gate = evaluate_gate(match_id)

    sections = [
        _setup_section(cv, ctx),
        _possession_section(cv, ctx, gate),
        _defence_section(cv, ctx, gate),
        _seams_section(cv, ctx, gate),
    ]
    audit = audit_body(sections, facts, match_id)
    if ctx.has_fifa:
        france_idx = 0 if ctx.roster.get("france_is_home", True) else 1
        sections.append(_validation_section(facts, cv, ctx, france_idx))
    else:
        sections.append(_validation_oracle_section(cv, ctx))
    return sections, gate, facts, audit


def _roster_for(match_id: str) -> dict:
    """France roster block for a match; ``{}`` for club matches with no France roster."""
    p = Path("data/france_roster.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["matches"].get(match_id, {})


# ==================================================================================================
# Renderers.
# ==================================================================================================
def _md_seam(b: Block) -> list[str]:
    out = [f"- **Seam.** {b.claim}"]
    if b.backing:
        out.append(f"  - *Backing:* {b.backing}")
    if b.falsifies:
        out.append(f"  - *Closes if:* {b.falsifies}")
    return out


def _oracle_meta(match_id: str, gate: GateResult) -> tuple[str, str, bool]:
    """Return ``(focus, opponent, is_fifa)`` plus the oracle name for a match's header prose."""
    m = get(match_id)
    is_fifa = (gate.oracle_source or "FIFA PMSR") == "FIFA PMSR"
    return m.teams[0], m.teams[1], is_fifa


def render_markdown(sections: list[Section], match_id: str, gate: GateResult, audit: dict) -> str:
    """Render the document to Markdown."""
    focus, opp, is_fifa = _oracle_meta(match_id, gate)
    oracle_name = gate.oracle_source or "FIFA PMSR"
    lines = [f"# {focus} v {opp} -- CV-primary tactical read", ""]
    if is_fifa:
        lines.append("*Body written entirely from our computer-vision metrics (the [CV] sections). "
                     "FIFA PMSR appears only in the validation appendix [FIFA], as an oracle we check "
                     "ourselves against -- never as content.*")
    else:
        lines.append(f"*Body written entirely from our computer-vision metrics (the [CV] sections). "
                     f"{oracle_name} appears only in the validation appendix [ORACLE], as an oracle we "
                     f"check ourselves against -- never as content.*")
    lines.append("")
    lines.append(f"**Ball-evidence gate:** post-link coverage {gate.coverage_pct:.0f}% "
                 f"(>= {GATE_COVERAGE_MIN_PCT:.0f}%), pass-recall proxy {gate.recall_pct:.1f}% "
                 f"(>= {GATE_RECALL_MIN_PCT:.0f}%) -> "
                 f"**{'PASS' if gate.passed else 'ABSTAIN on ball families'}**.  "
                 f"Body guardrail precision: {audit['precision'] * 100:.0f}% "
                 f"({audit['n_backed']}/{audit['n_numbers']} numbers CV-backed).")
    lines.append("")
    lines.append(f"*Gate inputs read from `outputs/eval/{match_id}_ball_eval.json` "
                 f"(oracle: {oracle_name}).*")
    lines.append("")
    if gate.coverage_clear_recall_short:
        lines.append(f"*Gate readout: coverage clears the {GATE_COVERAGE_MIN_PCT:.0f}% bar by "
                     f"{gate.coverage_pct - GATE_COVERAGE_MIN_PCT:.1f} pp, but the pass-recall proxy "
                     f"falls {GATE_RECALL_MIN_PCT - gate.recall_pct:.1f} pp short of the "
                     f"{GATE_RECALL_MIN_PCT:.0f}% bar -- a narrow miss. Ball families are withheld; "
                     f"the position-only structural sections render in full.*")
        lines.append("")
    for s in sections:
        tag = f"[{s.provenance}]"
        lines.append(f"## {tag} {s.title}")
        lines.append("")
        for b in s.blocks:
            if b.kind == "p":
                lines += [b.text, ""]
            elif b.kind == "bullets":
                lines.append(b.text)
                lines += [f"- {it}" for it in b.items]
                lines.append("")
            elif b.kind == "abstain":
                lines += [f"> **Withheld.** {b.text}", ""]
            elif b.kind == "seam":
                lines += _md_seam(b)
                lines.append("")
            elif b.kind == "table":
                lines += _md_table(b)
                lines.append("")
    lines.append("---")
    if is_fifa:
        lines.append("*football-synthesizer report v2. Body = CV metrics + roster names. Appendix = "
                     "FIFA PMSR oracle. C5 tendencies are a pooled 8-observation directional signal.*")
    else:
        lines.append(f"*football-synthesizer report v2. Body = CV metrics (team-level; no player "
                     f"roster for this club fixture). Appendix = {oracle_name} oracle. No C5 "
                     f"opponent-model claim (it is World-Cup-pooled).*")
    return "\n".join(lines)


def _md_table(b: Block) -> list[str]:
    out = []
    if b.caption:
        out.append(f"*{b.caption}*")
        out.append("")
    out.append("| " + " | ".join(b.headers) + " |")
    out.append("| " + " | ".join("---" for _ in b.headers) + " |")
    for row in b.rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def _inline_html(text: str) -> str:
    """Escape then convert **bold** and -> arrows for HTML."""
    esc = _html.escape(text)
    esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
    return esc.replace("--&gt;", "&rarr;").replace("-&gt;", "&rarr;").replace(" -- ", " &mdash; ")


def render_html(sections: list[Section], match_id: str, gate: GateResult, audit: dict) -> str:
    """Render a self-contained, print-friendly HTML document (embedded CSS, no CDN)."""
    focus, opp, is_fifa = _oracle_meta(match_id, gate)
    oracle_name = gate.oracle_source or "FIFA PMSR"
    gate_cls = "pass" if gate.passed else "abstain"
    gate_word = "PASS" if gate.passed else "ABSTAIN on ball families"
    body_html: list[str] = []
    for s in sections:
        tag = s.provenance
        body_html.append(f'<section class="prov-{tag.lower()}">')
        body_html.append(f'<h2><span class="tag tag-{tag.lower()}">{tag}</span> '
                         f'{_html.escape(s.title)}</h2>')
        for b in s.blocks:
            body_html.append(_html_block(b))
        body_html.append("</section>")
    prov_line = (f'<p class="legend">Gate inputs from '
                 f'<code>outputs/eval/{_html.escape(match_id)}_ball_eval.json</code> '
                 f'(oracle: {_html.escape(oracle_name)}).</p>')
    readout = ""
    if gate.coverage_clear_recall_short:
        readout = (f'<p class="legend">Gate readout: coverage clears the {GATE_COVERAGE_MIN_PCT:.0f}% '
                   f'bar by {gate.coverage_pct - GATE_COVERAGE_MIN_PCT:.1f} pp, but pass-recall falls '
                   f'{GATE_RECALL_MIN_PCT - gate.recall_pct:.1f} pp short of the '
                   f'{GATE_RECALL_MIN_PCT:.0f}% bar &mdash; a narrow miss; ball families withheld, '
                   f'structural sections render in full.</p>')
    if is_fifa:
        appendix_tag = '<span class="tag tag-fifa">FIFA</span> official PMSR (appendix only)'
        footer_note = ('Body = CV metrics + roster names &middot;\nAppendix = FIFA PMSR oracle '
                       '&middot; C5 tendencies are a pooled 8-observation directional signal')
    else:
        appendix_tag = (f'<span class="tag tag-oracle">ORACLE</span> {_html.escape(oracle_name)} '
                        '(appendix only)')
        footer_note = (f'Body = CV metrics (team-level; no player roster) &middot; Appendix = '
                       f'{_html.escape(oracle_name)} oracle &middot; no C5 opponent-model claim '
                       f'(World-Cup-pooled)')
    return _HTML_TEMPLATE.format(
        title=f"{_html.escape(focus)} v {_html.escape(opp)} - CV-primary tactical read",
        focus=_html.escape(focus), opponent=_html.escape(opp), oracle_name=_html.escape(oracle_name),
        gate_cls=gate_cls, gate_word=gate_word,
        coverage=f"{gate.coverage_pct:.0f}", recall=f"{gate.recall_pct:.1f}",
        cov_min=f"{GATE_COVERAGE_MIN_PCT:.0f}", rec_min=f"{GATE_RECALL_MIN_PCT:.0f}",
        precision=f"{audit['precision'] * 100:.0f}",
        n_backed=audit["n_backed"], n_numbers=audit["n_numbers"],
        prov_line=prov_line, readout=readout, appendix_tag=appendix_tag, footer_note=footer_note,
        css=_CSS, body="\n".join(body_html))


def _html_block(b: Block) -> str:
    if b.kind == "p":
        return f"<p>{_inline_html(b.text)}</p>"
    if b.kind == "bullets":
        items = "".join(f"<li>{_inline_html(it)}</li>" for it in b.items)
        return f"<p>{_inline_html(b.text)}</p><ul>{items}</ul>"
    if b.kind == "abstain":
        return f'<div class="abstain-box"><strong>Withheld.</strong> {_inline_html(b.text)}</div>'
    if b.kind == "seam":
        parts = [f'<div class="seam"><p class="seam-claim">{_inline_html(b.claim)}</p>']
        if b.backing:
            parts.append(f'<p class="seam-back"><em>Backing:</em> {_inline_html(b.backing)}</p>')
        if b.falsifies:
            parts.append(f'<p class="seam-fals"><em>Closes if:</em> {_inline_html(b.falsifies)}</p>')
        parts.append("</div>")
        return "".join(parts)
    if b.kind == "table":
        head = "".join(f"<th>{_html.escape(h)}</th>" for h in b.headers)
        rows = "".join("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>"
                       for r in b.rows)
        cap = f"<figcaption>{_inline_html(b.caption)}</figcaption>" if b.caption else ""
        return (f'<figure><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
                f"{cap}</figure>")
    return ""


_CSS = """
:root{--ink:#1a1a1a;--muted:#5a5a5a;--cv:#0b6b3a;--fifa:#8a5a00;--line:#e2e2e2;--abst:#b03030;}
*{box-sizing:border-box;}
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:var(--ink);max-width:820px;
 margin:0 auto;padding:40px 28px;line-height:1.55;font-size:16px;}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em;}
h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line);}
.lede{color:var(--muted);font-style:italic;margin:6px 0 18px;}
.gate{border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:14px 0 6px;
 background:#fafafa;font-size:14.5px;}
.gate .pass{color:var(--cv);font-weight:700;}
.gate .abstain{color:var(--abst);font-weight:700;}
.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.05em;padding:2px 7px;
 border-radius:4px;vertical-align:middle;margin-right:6px;}
.tag-cv{background:#e6f4ec;color:var(--cv);}
.tag-fifa{background:#f6ecd8;color:var(--fifa);}
.prov-fifa{opacity:.96;}
.prov-fifa h2{border-bottom-color:#e8d9b4;}
.tag-oracle{background:#f6ecd8;color:var(--fifa);}
.prov-oracle{opacity:.96;}
.prov-oracle h2{border-bottom-color:#e8d9b4;}
ul{margin:6px 0 14px;padding-left:22px;}
li{margin:3px 0;}
.abstain-box{border-left:4px solid var(--abst);background:#fdf3f3;padding:10px 14px;margin:12px 0;
 color:#7a2020;border-radius:0 6px 6px 0;font-size:14.5px;}
.seam{border:1px solid var(--line);border-left:4px solid var(--cv);border-radius:0 8px 8px 0;
 padding:12px 16px;margin:12px 0;background:#fbfdfb;}
.seam-claim{margin:0 0 6px;font-weight:600;}
.seam-back,.seam-fals{margin:4px 0;font-size:14px;color:var(--muted);}
table{border-collapse:collapse;width:100%;margin:8px 0 4px;font-size:14.5px;}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;}
th{background:#f4f4f4;font-weight:600;}
figcaption{color:var(--muted);font-size:13px;font-style:italic;margin:4px 0 16px;}
.legend{font-size:13px;color:var(--muted);margin:2px 0 0;}
footer{margin-top:36px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);
 font-size:13px;}
@media print{body{padding:0;font-size:12px;max-width:100%;}h2{page-break-after:avoid;}
 .seam,.abstain-box,figure{page-break-inside:avoid;}}
"""

_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style></head>
<body>
<h1>{focus} v {opponent}</h1>
<p class="lede">A CV-primary tactical read. The body is written entirely from our computer-vision
metrics; {oracle_name} appears only in the validation appendix as an oracle we check against.</p>
<div class="gate">
<strong>Ball-evidence gate:</strong> post-link coverage {coverage}% (needs &ge; {cov_min}%),
pass-recall proxy {recall}% (needs &ge; {rec_min}%) &rarr;
<span class="{gate_cls}">{gate_word}</span>.<br>
<strong>Body guardrail:</strong> {precision}% precision ({n_backed}/{n_numbers} numbers CV-backed).
{prov_line}{readout}
<p class="legend">Provenance legend: <span class="tag tag-cv">CV</span> our metrics (body) &middot;
{appendix_tag}.</p>
</div>
{body}
<footer>football-synthesizer report v2 &middot; {footer_note}.</footer>
</body></html>
"""


# ==================================================================================================
# CLI.
# ==================================================================================================
def generate(match_id: str, out: Path | None = None) -> dict:
    """Generate report v2 for a match; write .md and .html; return a summary dict."""
    sections, gate, facts, audit = build_document(match_id)
    md = render_markdown(sections, match_id, gate, audit)
    htm = render_html(sections, match_id, gate, audit)
    out_md = out or Path(f"results/report_v2_{match_id}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_html = out_md.with_suffix(".html")
    out_md.write_text(md, encoding="utf-8")
    out_html.write_text(htm, encoding="utf-8")
    return {"match": match_id, "gate": gate, "audit": audit,
            "md": out_md, "html": out_html, "sections": sections}


def _print_summary(res: dict) -> None:
    gate: GateResult = res["gate"]
    audit = res["audit"]
    verdict = "PASS" if gate.passed else "ABSTAIN(ball families)"
    print(f"[{res['match']}] gate={verdict} coverage={gate.coverage_pct:.0f}% "
          f"recall={gate.recall_pct:.1f}%  guardrail={audit['precision'] * 100:.0f}% "
          f"({audit['n_backed']}/{audit['n_numbers']})")
    if audit["n_unbacked"]:
        for u in audit["unbacked"]:
            print(f"   UNBACKED: {u['text']!r} ({u['unit']})")
    print(f"   wrote {res['md']}")
    print(f"   wrote {res['html']}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="CV-primary grounded France match report (v2).")
    ap.add_argument("--match", default="france_senegal",
                    help="france_iraq | france_senegal | france_norway | brighton_manutd | all "
                         "(all = the three France WC matches)")
    ap.add_argument("--out", default=None, help="output .md path (html written alongside)")
    args = ap.parse_args()
    ids = (list(GATE_INPUTS) if args.match == "all" else [args.match])
    fail = 0
    for mid in ids:
        res = generate(mid, Path(args.out) if args.out and args.match != "all" else None)
        _print_summary(res)
        if res["audit"]["n_unbacked"]:
            fail += 1
    if fail:
        raise SystemExit(f"{fail} match(es) had unbacked body numbers -- guardrail failed")


if __name__ == "__main__":
    main()
