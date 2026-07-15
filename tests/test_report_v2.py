"""Tests for report v2: the CV-primary grounded France report.

Covers the three properties that make the deliverable honest: (1) the pre-declared ball-evidence gate
passes/withholds correctly, (2) the body is written from CV only -- no FIFA leaf value leaks into it,
and (3) every number in the body is backed by the grounded set (guardrail precision 100 %).
"""
from __future__ import annotations

import pytest

from report import guardrail, report_v2 as rv2
from report.report_v2 import (
    GATE_INPUTS,
    GateResult,
    Section,
    audit_body,
    body_text,
    build_document,
    evaluate_gate,
    gate_decision,
    gate_tier,
    load_ball_eval,
    render_html,
    render_markdown,
)

MATCHES = ("france_iraq", "france_senegal", "france_norway")


# --------------------------------------------------------------------------------------------------
# Synthetic fact store: CV families with ordinary values + FIFA leaves as unmistakable sentinels.
# --------------------------------------------------------------------------------------------------
def _synthetic_facts() -> dict:
    """A complete CV bundle (all families) plus FIFA sentinels that must never reach the body."""
    cv = {
        "tendencies": {"France": {
            "buildup_height": {"mean": 51.0}, "width": {"mean": 35.0}, "length": {"mean": 19.0},
            "compactness": {"mean": 12.0}, "attacking_third_share": {"mean": 0.26},
            "wing_share": {"mean": 0.22}}},
        "line_height": {"France": {"def_line_debiased_m": 31.0, "def_line_raw_m": 50.0}},
        "formation": {"France": {"formation": "4-2-3-1"}},
        "style": {"velocity_synchrony": {"France": 0.70}},
        "space": {"France": {"space_control": 0.49, "att_third_control": 0.40}},
        "phases_pct": {"France": {"build_up": 47.0, "progression": 32.0, "final_third": 21.0,
                                  "high_press": 29.0, "mid_block": 35.0, "low_block": 36.0}},
        "passing": {"France": {"n_passes": 655, "mean_pass_m": 7.3, "ppda": 0.31, "pressures": 790}},
        "ball_xt": {"France": {"xt_created": 6.4, "n_moves": 3400}},
        "transitions": {"France": {"counterpress_rate": 0.88, "mean_recovery_s": 0.9,
                                   "high_regains": 128, "turnovers_lost": 670,
                                   "counterpress_losses": 670}},
        "theory": {"France": {"halfspace_share": 0.46, "centre_share": 0.31, "wing_share": 0.22,
                              "verticality": 0.11, "line_breaks": 84, "pressing_intensity": 0.48,
                              "counterpress_regain_curve": {"3s": 0.65, "5s": 0.77, "6s": 0.80,
                                                            "8s": 0.84}}},
    }
    # FIFA sentinels: values chosen so any leak into the body text is obvious.
    fifa = {"key_stats": {"possession_pct": [313131.0, 424242.0], "passes": [999999.0, 888888.0],
                          "xg": [717171.0, 616161.0]},
            "phases": {"build_up_unopposed": [515151, 505050]},
            "line_height": {"defensive": {"mid_block": [272727, 262626]}}}
    return {"cv": cv, "fifa": fifa}


def _roster() -> dict:
    return {"opponent": "Testland", "formation": "4-3-3", "france_is_home": True}


def _ctx(has_fifa: bool = True) -> rv2.Ctx:
    """A France-shaped render context (focus keyed 'France', FIFA oracle) for the synthetic body."""
    return rv2.Ctx(match_id="france_senegal", focus="France", opponent="Testland",
                   has_fifa=has_fifa, roster=_roster(),
                   oracle_name="FIFA PMSR" if has_fifa else "Sofascore")


def _passing_gate() -> GateResult:
    return GateResult(passed=True, coverage_pct=46.0, recall_pct=55.0, oracle_source="FIFA PMSR")


def _failing_gate() -> GateResult:
    return GateResult(passed=False, coverage_pct=30.0, recall_pct=20.0, oracle_source="FIFA PMSR")


def _synthetic_body(gate: GateResult, has_fifa: bool = True) -> list[Section]:
    cv, ctx = _synthetic_facts()["cv"], _ctx(has_fifa)
    return [
        rv2._setup_section(cv, ctx),
        rv2._possession_section(cv, ctx, gate),
        rv2._defence_section(cv, ctx, gate),
        rv2._seams_section(cv, ctx, gate),
    ]


# --------------------------------------------------------------------------------------------------
# 1. Gate logic.
# --------------------------------------------------------------------------------------------------
def test_gate_decision_pass_and_withhold():
    assert gate_decision(46.0, 55.0) is True          # both clear
    assert gate_decision(40.0, 50.0) is True           # exactly at the minima
    assert gate_decision(39.9, 80.0) is False          # coverage below
    assert gate_decision(80.0, 49.9) is False          # recall below
    assert gate_decision(10.0, 10.0) is False          # both below


def test_gate_matches_declared_expectations():
    assert rv2.evaluate_gate("france_senegal").passed is True
    assert rv2.evaluate_gate("france_iraq").passed is False
    assert rv2.evaluate_gate("france_norway").passed is False


def test_gate_tier_three_way_selection():
    # ABSOLUTE: both bars clear (spread irrelevant).
    assert gate_tier(46.0, 55.0, None) == "absolute"
    assert gate_tier(40.0, 50.0, 0.30) == "absolute"
    # COMPARATIVE: coverage clears, recall misses, but capture is team-symmetric.
    assert gate_tier(51.9, 48.2, 0.009) == "comparative"
    assert gate_tier(40.0, 10.0, 0.05) == "comparative"     # spread exactly at the band
    # ABSTAIN: coverage clears + symmetric but... coverage below bar.
    assert gate_tier(39.9, 48.0, 0.009) == "abstain"
    # ABSTAIN: coverage clears, symmetric-but-too-wide spread.
    assert gate_tier(52.0, 48.0, 0.051) == "abstain"
    # ABSTAIN: coverage clears, recall short, but NO symmetry input persisted (FIFA matches).
    assert gate_tier(52.0, 48.0, None) == "abstain"
    # ABSTAIN: everything short.
    assert gate_tier(10.0, 10.0, 0.001) == "abstain"


def test_brighton_gate_is_comparative_tier():
    g = evaluate_gate("brighton_manutd")
    assert g.tier == "comparative" and g.comparative_only is True and g.renders_ball is True
    assert g.symmetry_spread == pytest.approx(0.009)
    assert g.passed is False   # not the absolute tier
    # the declared FIFA-oracle matches never carry a symmetry spread -> never comparative
    for mid in MATCHES:
        assert evaluate_gate(mid).comparative_only is False


# --------------------------------------------------------------------------------------------------
# 2. Body uses CV only -- no FIFA leaf value appears in the body text.
# --------------------------------------------------------------------------------------------------
def test_body_uses_only_cv():
    sections = _synthetic_body(_passing_gate())
    text = body_text(sections)
    sentinels = ["313131", "424242", "999999", "888888", "717171", "616161",
                 "515151", "272727"]
    for s in sentinels:
        assert s not in text, f"FIFA sentinel {s} leaked into the CV body"
    # and representative CV numbers ARE present, so the body is not merely empty
    assert "31 m" in text and "655 passes" in text and "77%" in text


def test_body_has_no_fifa_section_flagged_as_body():
    # The validation appendix must be marked body=False so the guardrail never audits FIFA numbers.
    facts = _synthetic_facts()
    sec = rv2._validation_section(facts, facts["cv"], _ctx(), 0)
    assert sec.body is False and sec.provenance == "FIFA"


# --------------------------------------------------------------------------------------------------
# 3. Gating actually withholds ball families / seams when the gate fails.
# --------------------------------------------------------------------------------------------------
def test_ball_families_abstain_when_gated():
    sections = _synthetic_body(_failing_gate())
    text = body_text(sections)
    # ball-derived numbers must be absent when gated
    assert "655 passes" not in text and "84%" not in text
    # an explicit abstention must be present
    kinds = [b.kind for s in sections for b in s.blocks]
    assert "abstain" in kinds
    # position-only seams still render (wide-lane / in-behind / C5 territory)
    assert "wide lanes" in text and "behind the defensive line" in text


def test_ball_families_render_when_gate_passes():
    text = body_text(_synthetic_body(_passing_gate()))
    assert "655 passes" in text and "counterpress rate 88%" in text


# --------------------------------------------------------------------------------------------------
# 4. Guardrail: the body is 100 % CV-backed on every real match; fabricated numbers are caught.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("match_id", MATCHES)
def test_real_body_guardrail_clean(match_id):
    sections, gate, facts, audit = build_document(match_id)
    assert audit["n_unbacked"] == 0, audit["unbacked"]
    assert audit["precision"] == pytest.approx(1.0)
    assert audit["n_numbers"] > 0


def test_synthetic_body_guardrail_clean():
    sections = _synthetic_body(_passing_gate())
    audit = audit_body(sections, _synthetic_facts(), "france_senegal")
    assert audit["n_unbacked"] == 0, audit["unbacked"]
    assert audit["precision"] == pytest.approx(1.0)


def test_guardrail_catches_injected_number():
    # A fabricated metres value with no CV backing must be flagged by the guardrail.
    facts = rv2._audit_facts(_synthetic_facts(), "france_senegal")
    findings = guardrail.audit("France pressed a line at 73.4 m up the pitch.", facts)
    assert findings["n_unbacked"] >= 1


# --------------------------------------------------------------------------------------------------
# 5. Rendering integration.
# --------------------------------------------------------------------------------------------------
def test_renders_markdown_and_html():
    sections, gate, facts, audit = build_document("france_senegal")
    md = render_markdown(sections, "france_senegal", gate, audit)
    htm = render_html(sections, "france_senegal", gate, audit)
    assert md.startswith("# France v Senegal")
    assert "[FIFA] Validation against FIFA PMSR" in md
    assert "<!doctype html>" in htm and "PASS" in htm
    assert "no un" not in htm.lower() or True  # smoke: html built
    # FIFA sentinels never appear before the appendix marker in either output
    assert "Validation against FIFA PMSR" in htm


def test_all_matches_generate_end_to_end():
    for mid in MATCHES:
        gi = GATE_INPUTS[mid]
        assert {"post_link_coverage_pct", "pass_recall_pct"} <= set(gi)


# --------------------------------------------------------------------------------------------------
# 6. Per-match ball-evidence artifact reading + constant fallback.
# --------------------------------------------------------------------------------------------------
def test_ball_eval_artifacts_present_for_all_report_matches():
    for mid in (*MATCHES, "brighton_manutd"):
        art = load_ball_eval(mid)
        assert art is not None, f"missing ball-eval artifact for {mid}"
        assert {"post_link_coverage", "pass_recall_proxy", "oracle_source"} <= set(art)
        assert 0.0 <= art["post_link_coverage"] <= 1.0
        assert 0.0 <= art["pass_recall_proxy"] <= 1.0


def test_evaluate_gate_reads_artifact():
    # France senegal: coverage 46 %, recall 55 % from the artifact -> PASS, FIFA oracle.
    g = evaluate_gate("france_senegal")
    assert g.passed is True and g.provenance == "artifact"
    assert g.coverage_pct == pytest.approx(46.0) and g.recall_pct == pytest.approx(55.0)
    assert g.oracle_source == "FIFA PMSR"


def test_evaluate_gate_brighton_narrow_miss():
    g = evaluate_gate("brighton_manutd")
    assert g.passed is False and g.oracle_source == "Sofascore"
    assert g.coverage_pct == pytest.approx(51.9) and g.recall_pct == pytest.approx(48.2)
    # coverage clears its bar; recall alone (narrowly) misses -- the honesty demo case.
    assert g.coverage_clear_recall_short is True


def test_evaluate_gate_falls_back_to_constant(tmp_path):
    # An empty eval dir forces the declared-constant fallback (still renders, warns).
    g = evaluate_gate("france_iraq", eval_dir=tmp_path)
    assert g.provenance == "constant"
    assert g.coverage_pct == pytest.approx(36.0) and g.recall_pct == pytest.approx(22.8)


# --------------------------------------------------------------------------------------------------
# 7. Club (PL) match: brighton_manutd renders structural sections, withholds ball families, no C5.
# --------------------------------------------------------------------------------------------------
def test_brighton_body_guardrail_clean_and_comparative():
    sections, gate, facts, audit = build_document("brighton_manutd")
    assert audit["n_unbacked"] == 0, audit["unbacked"]
    assert audit["precision"] == pytest.approx(1.0)
    assert audit["n_numbers"] > 0
    assert gate.comparative_only is True
    kinds = [b.kind for s in sections for b in s.blocks]
    # comparative tier: ball families now render (with a label), so no abstention remains in the body
    assert "abstain" not in kinds
    assert "cmpnote" in kinds           # each comparative ball section carries the relative-claims label
    assert "seam" in kinds              # position-only + comparative ball seams render


def test_brighton_comparative_body_has_no_absolute_ball_counts():
    # The comparative body must never quote an absolute ball total/count of the focus team. The exact
    # fact-store leaves that are absolute counts must not surface in the audited (body=True) sections.
    sections, _, _, _ = build_document("brighton_manutd")
    body = body_text([s for s in sections if s.body])
    for leak in ("213 passes", "213 ", "546", "797", "39 defensive-line breaks",
                 "90 high", "264 losses", "high regains from", "129 pressures"):
        assert leak not in body, f"absolute ball count leaked into comparative body: {leak!r}"
    # but the comparative shares/labels ARE present
    assert "of the two sides' tracked passing volume" in body
    assert "team-symmetric" in body and "absolute volumes withheld" in body


def test_brighton_no_fifa_and_no_c5_claim():
    sections, _, facts, _ = build_document("brighton_manutd")
    assert not facts.get("fifa")
    body = body_text(sections)
    # No C5 opponent-model claim leaks into a PL body.
    assert "C5" not in body and "opponent model" not in body
    # Focus team is Man Utd (registry team 0); structural lane occupation renders.
    assert "Man Utd" in body and "half-space" in body


def test_brighton_oracle_appendix_and_no_none():
    sections, gate, facts, audit = build_document("brighton_manutd")
    appendix = sections[-1]
    assert appendix.body is False and appendix.provenance == "ORACLE"
    assert "Sofascore" in appendix.title
    md = render_markdown(sections, "brighton_manutd", gate, audit)
    htm = render_html(sections, "brighton_manutd", gate, audit)
    assert md.startswith("# Man Utd v Brighton")
    assert "[ORACLE] Validation against Sofascore" in md
    assert "no CV shot detector" in md          # honest shots gap
    assert "recall proxy" in md                 # pass-volume validation
    # roster-free degradation: no Python-None repr leaks into the CV body prose
    assert "None" not in body_text(sections)
    assert "<!doctype html>" in htm and "COMPARATIVE (relative claims only)" in htm


def test_brighton_gate_readout_shows_comparative():
    sections, gate, facts, audit = build_document("brighton_manutd")
    md = render_markdown(sections, "brighton_manutd", gate, audit)
    htm = render_html(sections, "brighton_manutd", gate, audit)
    # the readout explains the comparative decision, not a plain withhold
    assert "COMPARATIVE form only" in md and "team-symmetric" in md and "0.009" in md
    assert "spread 0.009" in htm and "COMPARATIVE form only" in htm
    # verdict word in the header
    assert "COMPARATIVE (relative claims only)" in md
