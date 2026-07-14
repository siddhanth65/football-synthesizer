"""Tests for the numeric guardrail — the honesty layer of the grounded report."""

from __future__ import annotations

from report.guardrail import annotate, audit, flatten_facts

FACTS = {
    "cv": {
        "line_height": {"France": {"def_line_debiased_m": 36.4}},
        "tendencies": {"France": {"attacking_third_share": {"mean": 0.32}}},
        "theory": {"France": {"counterpress_regain_curve": {"5s": 0.75}, "line_breaks": 41}},
    },
    "fifa": {"key_stats": {"xg": [2.30, 0.80], "completed_line_breaks": [117, 44],
                           "possession_pct": [58, 42]}},
}


def test_flatten_types_values_by_unit():
    g = {x.label: x for x in flatten_facts(FACTS)}
    # metres, xg, count, pct kinds inferred from labels
    assert any(v.kind == "m" and abs(v.value - 36.4) < 0.1 for v in g.values())
    assert any(v.kind == "xg" and abs(v.value - 2.30) < 0.01 for v in g.values())
    assert any(v.kind == "count" and v.value == 117 for v in g.values())
    # a 0..1 share/rate also grounds its percentage form
    assert any(v.kind == "pct" and abs(v.value - 32.0) < 0.1 for v in flatten_facts(FACTS))
    assert any(v.kind == "pct" and abs(v.value - 75.0) < 0.1 for v in flatten_facts(FACTS))


def test_grounded_numbers_pass():
    text = ("France's line sat at 36 m, they committed 32% forward, regained within 5 s 75% of the time, "
            "made 117 line breaks and 2.30 xG on 58% possession.")
    a = audit(text, FACTS)
    assert a["n_unbacked"] == 0, a["unbacked"]
    assert a["n_numbers"] >= 6


def test_fabrications_are_flagged_by_kind():
    # each number is out of range for its unit -> must be caught
    text = "a line of 88 m, worth 5.7 xG, some 999 line breaks, and 39 s recovery"
    a = audit(text, FACTS)
    flagged = {f["number"] for f in a["unbacked"]}
    assert {88.0, 5.7, 999.0, 39.0} <= flagged


def test_kind_typing_does_not_cross_units():
    # 117 is a grounded COUNT; as metres it must NOT be considered backed
    assert audit("a line height of 117 m", FACTS)["n_unbacked"] == 1
    # but 117 line breaks (count) is backed
    assert audit("117 line breaks", FACTS)["n_unbacked"] == 0


def test_annotate_wraps_only_ungrounded():
    out = annotate("line 36 m but a wild 88 m and 5.7 xG", FACTS)
    assert "[?88 m]" in out
    assert "[?5.7 xG]" in out
    assert "36 m" in out and "[?36 m]" not in out
