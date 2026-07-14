"""Numeric guardrail: every number a report emits must be traceable to a grounded fact.

A grounded pundit report's honesty is not the LLM's fluency — it is the guarantee that **no fabricated or
unbacked number reaches the reader**. This module flattens the per-match fact store + FIFA PMSR into a set
of grounded values, extracts every number from candidate report prose, and checks each against the
grounded set within a unit-appropriate tolerance. Any number with no grounding is flagged as a potential
fabrication.

Validation (``tools/guardrail_eval.py``): under adversarial injection of known-wrong numbers, the guardrail
must flag 100 % of them (recall) while leaving the genuinely grounded numbers alone (precision). See the
audit's section 6 and Fable's steer: *the guardrail's recall is the report's honesty.*
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A number optionally followed by a unit token. Handles "52 m", "32%", "2.30 xG", "0.4 s", "117".
_NUM_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|m|metres|xg|s\b|seconds)?",
    re.IGNORECASE,
)
# Unit-specific match tolerances (absolute), chosen so prose rounding passes but fabrications don't.
_TOL = {"pct": 1.6, "m": 1.6, "xg": 0.04, "s": 0.25, "count": 1.0}
# Tokens that are years / scores / list ordinals etc. — not grounded metrics; skip to cut false positives.
_SKIP_CONTEXT = re.compile(r"\b(20\d\d|first|second|third|half)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Grounded:
    """One grounded numeric value with its kind (unit) and where it came from."""

    value: float
    kind: str     # "m" | "pct" | "xg" | "s" | "count"
    label: str
    source: str   # "cv" | "fifa"


# Label -> kind, first matching rule wins. COUNT-distinctive names are checked before the phase-name
# PCT tokens, because count metrics embed phase words (receptions_final_THIRD, ball_PROGRESSIONs).
_KIND_RULES = [
    ("xg", ("xg", "xt_created", "counter_xt")),
    ("s", ("recovery_s", "mean_recovery")),
    ("count", ("reception", "line_break", "ball_progression", "forced_turnover", "turnover",
               "attempt", "n_passes", ".passes", "_passes", "pressure", "high_regains", "loss",
               "n_moves", "n_back", "goal", "set_pieces", "overload", "chunks",
               "n_ball", "crosses", "distance")),
    ("m", ("def_line", "line_debiased", "line_raw", "buildup_height", "line_height", "width",
           "depth", "style_distance", "mean_pass_m", "fit_cost_m", "length", "_m")),
    ("pct", ("share", "·pct", "control", "synchrony", "possession", "pass_completion", "phases_pct",
             "phases.", "pressing_intensity", "verticality", "counterpress_rate", "regain_curve",
             "counterpress_regain", "regain_", "build_up", "progression", "final_third",
             "high_press", "mid_block", "low_block", "high_block", "long_ball", "counter-press",
             "recovery", "transition", "_press", "_block")),
]


def _kind(label: str) -> str:
    """Infer a grounded value's unit-kind from its dotted fact-store label."""
    low = label.lower()
    for kind, toks in _KIND_RULES:
        if any(t in low for t in toks):
            return kind
    return "count"   # bare integers in the store are almost all counts


def _walk(obj, path, source, out):
    """Recursively collect numeric leaves from a fact dict as (typed) Grounded values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f"{path}.{k}" if path else str(k), source, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", source, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(Grounded(float(obj), _kind(path), path, source))


def flatten_facts(facts: dict | None) -> list[Grounded]:
    """All grounded numbers from a fact-store bundle: CV metrics + FIFA PMSR, with 0..1 shares also as %."""
    out: list[Grounded] = []
    if not facts:
        return out
    _walk(facts.get("cv", {}), "cv", "cv", out)
    _walk(facts.get("fifa", {}), "fifa", "fifa", out)
    # a share/rate in [0,1] is spoken as a percentage — ground the % form so "32%" matches 0.32
    for g in list(out):
        if g.kind == "pct" and 0.0 < g.value <= 1.0:
            out.append(Grounded(round(g.value * 100, 1), "pct", g.label + "·pct", g.source))
    # methodological constants the report legitimately cites (the gegenpressing 5-second window)
    for w in (3.0, 4.0, 5.0, 6.0, 8.0):
        out.append(Grounded(w, "s", "const.counterpress_window_s", "cv"))
    return out


def _text_kind(raw_unit: str | None) -> str:
    """Kind of a number appearing in prose, from its trailing unit token ('' -> count)."""
    u = (raw_unit or "").lower()
    if u == "%":
        return "pct"
    if u in ("m", "metres"):
        return "m"
    if u == "xg":
        return "xg"
    if u in ("s", "seconds"):
        return "s"
    return "count"


@dataclass(frozen=True)
class Finding:
    """A number found in the text and whether it is grounded."""

    number: float
    unit: str
    text: str
    backed: bool
    matched: Grounded | None


def check_text(text: str, grounded: list[Grounded]) -> list[Finding]:
    """Every number in ``text`` with whether a grounded fact backs it (within unit tolerance)."""
    findings = []
    for m in _NUM_RE.finditer(text):
        raw, unit_tok = m.group(1), m.group(2)
        # skip years / ordinals / score-like tokens by local context
        ctx = text[max(0, m.start() - 12):m.end() + 12]
        if _SKIP_CONTEXT.search(ctx) and not unit_tok:
            continue
        n = float(raw)
        kind = _text_kind(unit_tok)
        tol = _TOL[kind]
        best = None
        for g in grounded:
            if g.kind == kind and abs(g.value - n) <= tol:
                if best is None or abs(g.value - n) < abs(best.value - n):
                    best = g
        findings.append(Finding(n, kind, m.group(0).strip(), best is not None, best))
    return findings


def audit(text: str, facts: dict | None) -> dict:
    """Audit a candidate report against a match's fact store. Returns findings + an unbacked list."""
    grounded = flatten_facts(facts)
    findings = check_text(text, grounded)
    unbacked = [f for f in findings if not f.backed]
    return {
        "n_numbers": len(findings),
        "n_backed": sum(f.backed for f in findings),
        "n_unbacked": len(unbacked),
        "unbacked": [{"number": f.number, "unit": f.unit, "text": f.text} for f in unbacked],
        "findings": findings,
    }


def annotate(text: str, facts: dict | None, marker: str = "[?{}]") -> str:
    """Return ``text`` with every ungrounded number wrapped in ``marker`` so a reviewer sees fabrications.

    ``marker`` is a format string with one ``{}`` slot (default ASCII ``[?…]`` for terminal safety).
    """
    grounded = flatten_facts(facts)
    out, last = [], 0
    for m in _NUM_RE.finditer(text):
        n = float(m.group(1))
        kind = _text_kind(m.group(2))
        ctx = text[max(0, m.start() - 12):m.end() + 12]
        if _SKIP_CONTEXT.search(ctx) and not m.group(2):
            continue
        backed = any(g.kind == kind and abs(g.value - n) <= _TOL[kind] for g in grounded)
        if not backed:
            out.append(text[last:m.start()])
            out.append(marker.format(m.group(0).strip()))
            last = m.end()
    out.append(text[last:])
    return "".join(out)
