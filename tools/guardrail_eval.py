"""Validate the numeric guardrail — the single check that keeps a grounded report honest (Fable).

Two measurements over the France reports:

* **Precision** — audit each real pundit report; every number it emits should be backed by the fact
  store (a false positive means a grounded number was wrongly flagged, which would erode trust in the
  flags). Target: ~100 %.
* **Recall (adversarial injection)** — for each match and each unit-kind, inject a battery of numbers
  drawn from *outside* the grounded range for that kind (guaranteed fabrications) and require the guardrail
  to flag **100 %** of them. A grounded report's honesty is this recall: if one made-up number survives,
  nothing else matters.

Run: ``python -m tools.guardrail_eval``.
"""
from __future__ import annotations

import numpy as np

from core.registry import matches
from report.facts import load_facts
from report.guardrail import audit, flatten_facts
from report.pundit import generate

UNIT_TOKEN = {"m": "m", "pct": "%", "xg": "xG", "s": "s", "count": ""}
N_PER_KIND = 40


def _fabrications(grounded, kind, rng) -> list[float]:
    """Numbers clearly outside the grounded range for ``kind`` — guaranteed fabrications."""
    vals = [g.value for g in grounded if g.kind == kind]
    hi = max(vals) if vals else 10.0
    lo, span = hi + max(3.0, 0.3 * abs(hi)), max(20.0, abs(hi))
    return list(rng.uniform(lo, lo + span, N_PER_KIND))


def main() -> None:
    rng = np.random.default_rng(0)
    print("=== guardrail validation ===\n")
    tot_num = tot_backed = 0
    fps = []
    for m in matches(processed_only=True):
        if "France" not in m.teams:
            continue
        facts = load_facts(m.id)
        a = audit(generate(m.id), facts)
        tot_num += a["n_numbers"]
        tot_backed += a["n_backed"]
        fps += [(m.id, u["text"]) for u in a["unbacked"]]
    print(f"PRECISION (real reports): {tot_backed}/{tot_num} numbers backed = {tot_backed/tot_num:.1%}")
    if fps:
        print(f"  false positives (grounded numbers wrongly flagged): {fps}")

    caught = injected = 0
    per_kind: dict[str, list[int]] = {}
    for m in matches(processed_only=True):
        if "France" not in m.teams:
            continue
        facts = load_facts(m.id)
        grounded = flatten_facts(facts)
        for kind, tok in UNIT_TOKEN.items():
            fabs = _fabrications(grounded, kind, rng)
            text = " ".join(f"value {v:.2f} {tok}".strip() for v in fabs)
            a = audit(text, facts)
            flagged = {round(f["number"], 2) for f in a["unbacked"]}
            hit = sum(round(v, 2) in flagged for v in fabs)
            caught += hit
            injected += len(fabs)
            per_kind.setdefault(kind, [0, 0])
            per_kind[kind][0] += hit
            per_kind[kind][1] += len(fabs)
    print(f"\nRECALL (adversarial injection): {caught}/{injected} fabrications flagged = {caught/injected:.1%}")
    for kind, (h, n) in per_kind.items():
        print(f"  {kind:<6} {h}/{n} = {h/n:.0%}")
    print("\nHonest verdict: a grounded report's integrity is the recall above -- every un-grounded number "
          "gets bracket-annotated (report.guardrail.annotate) before a human or the LLM narration ships it.")


if __name__ == "__main__":
    main()
