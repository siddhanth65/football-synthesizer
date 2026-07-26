# B4 abstention policy — the layer-A defect, three policies measured, and the decision (2026-07-26)

Resolves the first of the two open issues left by the gate run (`docs/B4_IMPUTATION_PLAN.md`,
"TWO OPEN ISSUES"): the pre-registered abstention layer A is mis-specified and abstains on 100% of
the holdout. The supervisor deferred the decision to us **on the condition that the change must not
cost performance or accuracy**. This document measures that condition rather than asserting it.

**Nothing was refitted.** The frozen v1 protocol (`tools/imputation_b4_v1.py`, same splits, same
hyper-parameters, `random_state=0`) was replayed once to regenerate its deterministic predictions,
which are cached to `data/imputation/cache/b4_v1_frozen_preds.npz` (gitignored). Every policy below
is a *mask over that single frozen array*. Reproduction check, printed at the top of the run:
holdout n = 644 792, **v1 ALL 8.10 m vs the B7 bar 11.46 m** — the frozen record, to the last digit.

```
python -m tools.imputation_b4_abstention          # seconds from cache, ~15 min CPU to rebuild it
```

Code: `synthesizer/imputation_v1.py` (`emit`), `tools/imputation_b4_abstention.py`,
`tests/test_imputation_v1.py`. Verbatim console trail: `results/B4_ABSTENTION_runlog.md`.

---

## 1. The defect

Plan §3.3 layer A: *"the abstention horizon is the first bucket `b*` where v1's calibration-split
RMSE is not significantly below the frozen bar. Buckets at and beyond `b*` are declared no-assert."*
That wording imports the weather-forecasting assumption that skill **decays** with horizon. Ours
does the opposite — the only bucket without a significant win is the *shortest* one:

| horizon | v1 (CALIB) | B7 (CALIB) | diff [95% block CI] | significant? |
|---|---|---|---|---|
| 0-1s | 0.42 | 0.41 | [−0.01, +0.02] | **no** |
| 1-3s | 1.82 | 2.08 | [−0.32, −0.21] | yes |
| 3-5s | 3.70 | 4.36 | [−0.78, −0.54] | yes |
| 5-10s | 6.11 | 7.40 | [−1.61, −0.96] | yes |
| 10-30s | 9.52 | 13.73 | [−5.44, −2.93] | yes |
| 30s+ | 8.29 | 12.53 | [−5.09, −3.10] | yes |

So `b* = 0-1s`, "at and beyond" means every bucket, and the rule silences a model that wins by
5.2 m at 10-30s because of a 1 cm tie at 0-1s. The rule's error is conflating **"no skill"** with
**"no improvement over the anchor"**. At 0-1s v1 is accurate to 0.68 m and the anchor to 0.66 m:
there is nothing to *add*, not nothing to *say*.

---

## 2. Three policies on the frozen holdout

Same predictions throughout. "asserted" = a coordinate is emitted with `asserted=True`. "emitted
RMSE" scores whichever estimator the policy chose. "anchor RMSE" is B7 on the *same* rows, for
reference. PICP = empirical coverage of the calibrated 50%/90% regions of the emitted point.

### P0 — literal pre-registered rule (abstain at and beyond `b*`)

| horizon | n | asserted | emitted RMSE |
|---|---|---|---|
| every bucket | 644 792 | **0.0%** | nothing emitted |

Degenerate, exactly as predicted on CALIB before the holdout was opened. Recorded here as the
pre-registered outcome; it is not a candidate.

### P1 — per-bucket abstain (assert v1 only where it beats the anchor)

| horizon | n | asserted | emitted RMSE | p50 | anchor RMSE | anchor p50 | PICP50 | PICP90 |
|---|---|---|---|---|---|---|---|---|
| 0-1s | 51 113 | 0.0% | — | — | — | — | — | — |
| 1-3s | 82 286 | 100.0% | 1.90 | 1.15 | 2.09 | 1.18 | 49.9 | 89.8 |
| 3-5s | 62 854 | 100.0% | 3.81 | 2.61 | 4.37 | 2.86 | 48.3 | 90.1 |
| 5-10s | 107 320 | 100.0% | 6.07 | 4.23 | 7.55 | 5.18 | 50.9 | 90.6 |
| 10-30s | 190 552 | 100.0% | 9.74 | 6.57 | 14.91 | 9.90 | 47.8 | 89.4 |
| 30s+ | 150 667 | 99.7% | 11.06 | 5.86 | 14.99 | 8.95 | 45.6 | 89.4 |
| **ALL** | 644 792 | **92.0%** | **8.38** | 4.38 | 11.89 | 6.03 | 48.2 | 89.7 |

### P2 — defer-to-anchor (adopted; abstention left to layer B alone)

With the anchor's **own** CALIB-calibrated region wherever the anchor is emitted (see §3.3):

| horizon | n | asserted | emitted RMSE | p50 | anchor RMSE | anchor p50 | PICP50 | PICP90 | source |
|---|---|---|---|---|---|---|---|---|---|
| 0-1s | 51 113 | 100.0% | **0.66** | 0.15 | 0.66 | 0.15 | 52.4 | 90.0 | anchor |
| 1-3s | 82 286 | 100.0% | 1.90 | 1.15 | 2.09 | 1.18 | 49.9 | 89.8 | v1 |
| 3-5s | 62 854 | 100.0% | 3.81 | 2.61 | 4.37 | 2.86 | 48.3 | 90.1 | v1 |
| 5-10s | 107 320 | 100.0% | 6.07 | 4.23 | 7.55 | 5.18 | 50.9 | 90.6 | v1 |
| 10-30s | 190 552 | 100.0% | 9.74 | 6.57 | 14.91 | 9.90 | 47.8 | 89.4 | v1 |
| 30s+ | 150 667 | 99.7% | 11.06 | 5.86 | 14.99 | 8.95 | 45.6 | 89.4 | v1 |
| **ALL** | 644 792 | **99.9%** | **8.04** | 3.96 | 11.41 | 5.36 | 48.5 | 89.8 | — |

### Reference — unrestricted v1 (no policy at all)

| horizon | n | asserted | emitted RMSE | p50 | anchor RMSE | PICP50 | PICP90 |
|---|---|---|---|---|---|---|---|
| 0-1s | 51 113 | 100.0% | 0.68 | 0.21 | 0.66 | 49.6 | 89.3 |
| **ALL** | 644 792 | 100.0% | **8.10** | 3.96 | 11.46 | 48.2 | 89.7 |

(Other buckets identical to P2 by construction — v1 is emitted in all five skill buckets.)

### Side by side

| | P0 literal | P1 abstain | P2 defer (adopted) | unrestricted v1 |
|---|---|---|---|---|
| coverage (asserted) | **0.0%** | 92.00% | **99.93%** | 100% |
| positions emitted | 0 | 593 232 | 644 345 | 644 792 |
| emitted RMSE | n/a | 8.38 m | **8.04 m** | 8.10 m |
| emitted median error | n/a | 4.38 m | **3.96 m** | 3.96 m |
| anchor RMSE, same rows | n/a | 11.89 m | 11.41 m | 11.46 m |
| PICP 50 / 90 on asserted | n/a | 48.2 / 89.7 | 48.5 / 89.8 | 48.2 / 89.7 |
| buckets within ±5 pts of nominal | n/a | 5/5 · 5/5 | 6/6 · 6/6 | 6/6 · 6/6 |
| has a working "I don't know" | yes, always | yes (bucket + region) | yes (region only) | no |

---

## 3. Does P2 cost accuracy? **No — measured three ways.**

### 3.1 Against P1, where both speak: exactly zero

On the 593 232 rows P1 asserts, P1 and P2 emit **bit-identical coordinates** (verified by array
equality in the run log, not by argument). Difference: 0.0000 m. P2 differs from P1 only by
*additionally* emitting 51 113 positions that P1 throws away — at **0.66 m** RMSE, the most
accurate bucket the system has. P1's cost is 7.93 points of coverage bought for no accuracy at all.

### 3.2 Against unrestricted v1, on all 644 792 rows: −0.0001 m

| comparison | RMSE | diff [95% block-bootstrap CI] |
|---|---|---|
| P2 emitted point (all rows) | 8.0995 m | — |
| unrestricted v1 (all rows) | 8.0996 m | **−0.0001 [−0.0003, +0.0000]** |

Deferring is free to four decimal places. The 8.04 m in §2 is the same policy scored on the rows it
asserts (i.e. after layer B drops 447 rows); the 8.0995 m above is the apples-to-apples all-row
number.

### 3.3 Inside the defer bucket itself

| | anchor (emitted by P2) | v1 (emitted by P1's counterfactual) | diff [95% CI] |
|---|---|---|---|
| 0-1s RMSE | 0.6585 m | 0.6762 m | **−0.0177 [−0.0493, +0.0038]** |
| 0-1s median | 0.15 m | 0.21 m | — |
| mean \|v1 − anchor\| | — | — | 0.232 m |

The anchor is 1.8 cm better in the point estimate with a CI straddling zero: deferring neither helps
nor hurts measurably. That is the whole argument — the two estimators are interchangeable there, so
the sane behaviour is to emit one of them, labelled, rather than to go silent.

**The one real cost, found and fixed.** A naive P2 keeps v1's conformal multipliers while emitting
the anchor's point. That breaks calibration at 0-1s: **PICP50 = 59.3%**, outside the gate-2 ±5-point
band (over-coverage — the safe direction, but off-nominal). Cause: the anchor's residuals are more
concentrated (median 0.15 m vs 0.21 m), so v1's region is too generous for it. Fix, derived on
**CALIB only** exactly like every other conformal number: calibrate the region for the estimator
that produces the point.

| horizon | k50 v1 | k50 anchor | k90 v1 | k90 anchor | emitted by P2 |
|---|---|---|---|---|---|
| 0-1s | 0.523 | **0.431** | 1.335 | **1.351** | anchor |
| 1-3s | 0.758 | 0.827 | 1.603 | 1.792 | v1 |
| 3-5s | 0.756 | 0.918 | 1.583 | 1.864 | v1 |
| 5-10s | 0.791 | 0.985 | 1.630 | 1.995 | v1 |
| 10-30s | 0.853 | 1.307 | 1.816 | 2.699 | v1 |
| 30s+ | 0.914 | 1.639 | 1.953 | 2.871 | v1 |

With the anchor's own `k`, 0-1s coverage returns to **52.4% / 90.0%** — inside the band at both
levels, so P2 keeps 6/6 · 6/6 on the emitted point. This is the only implementation subtlety in the
change, and it is the reason `emit` takes `r90_anchor`: emitting one estimator's point with another
estimator's region is precisely the kind of quiet mismatch this project has already paid for once.

---

## 4. Is layer B still meaningful under P2? — the finding

Under P2, layer B is the *only* thing that can say "I don't know", so it had better work.

### 4.1 Why `R_max` accepts everything, and what it still rejects

`R_max = 30.4339 m` is not a chosen number: the SGR binary search returned its upper bracket
(`1.001 × max CALIB r90 = 1.001 × 30.40`) because the guarantee was already met at full coverage.
The root cause is the **target**, not the search: the risk bar was set to the *anchor's* CALIB RMSE
of 9.92 m, while v1's 95% upper bound at full coverage is **7.49 m**. Any target above ≈7.5 m is
inactive by arithmetic.

It is nevertheless **not literally dead on the holdout**: 447 rows (0.07%, all in 30s+, 0.30% of
that bucket) have `r90 > R_max` because the holdout's widest region (34.12 m) exceeds CALIB's
(30.40 m). And those rejections are not noise —

| on the 447 rejected rows | RMSE |
|---|---|
| v1, had it been asserted | **38.40 m** |
| the anchor, had it been asserted | 43.89 m |
| last-seen, which is what is actually emitted | 11.27 m |

The reject option, at 0.07% coverage cost, removes samples where the estimators are wrong by a
third of the pitch and last-seen is 3.4× better. The mechanism is sound; the threshold is slack.

### 4.2 The model does express doubt — the policy just doesn't act on it

`r90` distribution on the holdout:

| horizon | p50 | p90 | p99 | max | % > 10 m | % > 15 m | % > 20 m |
|---|---|---|---|---|---|---|---|
| 0-1s | 0.5 | 1.0 | 1.5 | 2.6 | 0.0 | 0.0 | 0.0 |
| 1-3s | 2.6 | 4.1 | 5.1 | 6.9 | 0.0 | 0.0 | 0.0 |
| 3-5s | 5.5 | 7.0 | 8.3 | 10.6 | 0.0 | 0.0 | 0.0 |
| 5-10s | 9.1 | 11.5 | 13.7 | 18.3 | 31.7 | 0.3 | 0.0 |
| 10-30s | 14.1 | 17.9 | 21.6 | 26.8 | **91.3** | **37.7** | 2.6 |
| 30s+ | 12.0 | 19.9 | 26.2 | 34.1 | **63.9** | **33.0** | 9.8 |

Long-horizon regions are genuinely wide: at 10-30s more than nine samples in ten carry a 90% region
wider than 10 m radius, and better than a third exceed 15 m. The uncertainty head is doing its job.

### 4.3 Where abstention would start to bite

Accept `r90 ≤ R` (holdout, **diagnostic only — no threshold may be chosen from this table**):

| R (m) | coverage | selective RMSE | coverage 10-30s | coverage 30s+ |
|---|---|---|---|---|
| 30.4 (frozen) | 99.9% | 8.04 | 100.0% | 99.7% |
| 25 | 99.6% | 7.90 | 100.0% | 98.4% |
| 20 | 96.9% | 7.63 | 97.4% | 90.2% |
| 15 | 81.1% | 5.98 | 62.3% | 67.0% |
| 12 | 64.5% | 4.47 | 23.4% | 49.7% |
| 10 | 52.8% | 3.79 | 8.7% | 36.1% |
| 6 | 29.5% | 2.35 | 1.2% | 6.0% |

First material bite is between 20 m and 15 m. And the same R_max re-derived by the *frozen* SGR
procedure from a target stated in metres (CALIB only; holdout columns disclosed as diagnostic):

| target (m) | R_max (m) | CALIB coverage | CALIB sel-RMSE (95% UB) | holdout coverage | holdout sel-RMSE |
|---|---|---|---|---|---|
| 9.92 (current) | 30.434 | 100.0% | 6.99 (7.49) | 99.9% | 8.04 |
| 8 | 30.434 | 100.0% | 6.99 (7.49) | 99.9% | 8.04 |
| 7 | 17.143 | 94.0% | 6.50 (7.00) | 91.2% | 7.09 |
| 6 | 14.686 | 83.1% | 5.68 (6.00) | 79.2% | 5.79 |
| 5 | 12.459 | 70.5% | 4.77 (5.00) | 67.0% | 4.68 |
| 4 | 10.118 | 57.9% | 3.83 (4.00) | 53.5% | 3.83 |
| 3 | 7.228 | 40.0% | 2.85 (3.00) | 36.6% | 2.88 |

Structural observation worth the thesis: because `r90` is so strongly horizon-driven, a tight
region threshold *is* approximately a horizon threshold (R = 10 m keeps 8.7% of 10-30s and 36.1% of
30s+). A working layer B would therefore deliver most of what layer A was originally trying to buy —
per-sample and calibrated, rather than per-bucket and by decree.

### 4.4 Options for making layer B active — reported, not chosen

| option | what it is | why it is defensible | why it might not be |
|---|---|---|---|
| **A. risk target in metres** | replace "the anchor's CALIB RMSE" with a target stated as *acceptable positional error for a downstream claim* (e.g. 5 m → `R_max` 12.46 m, 70.5% CALIB coverage) | the target comes from the use case, not from the error distribution; `R_max` is still derived on CALIB by the frozen SGR search; nothing about the model changes | the sweep above has now been computed, so a number picked today is informed by having *seen* that 5 m yields 70%; the choice must be justified downstream-first and re-frozen with sign-off |
| **B. sharpness cap** | abstain when `r90` exceeds a fixed footballing width (e.g. 15 m ≈ the depth of the penalty area, 81.1% holdout coverage) | needs no risk model at all — it is a statement about what region is too vague to support a claim; trivially auditable | the number is still a judgement call, and it ignores that a wide region can still be honest |
| **C. status quo** | keep `R_max = 30.4339 m` and report layer B as near-inactive (0.07%) | zero post-hoc anything; the guarantee is genuinely met at full coverage | leaves the thesis with an abstention mechanism that fires 447 times in 644 792 — thin as a demonstration |
| **D. per-consumer threshold** | ship no single `R_max`; every downstream metric declares its own tolerance and thresholds the calibrated `r90` | gate 2 says the region is calibrated 6/6, which is exactly the licence needed to let consumers threshold it; makes abstention a query-time property, which is what a scouting pack actually needs | no single headline "abstention rate" for the thesis; pushes the decision onto each consumer |

**Recommendation: A + D, with C left frozen for the gate record.** State the risk target in metres
from the downstream requirement (what positional error still supports the claim a scouting page
makes), re-derive `R_max` on CALIB by the unchanged SGR search, and let individual consumers tighten
further via D. That keeps a single auditable trail while giving the system a reject option that
actually fires. **This is a change to a pre-registered threshold and therefore needs supervisor
sign-off before any number ships** — unlike the layer-A restatement below, it is not free: it trades
coverage for selective accuracy, and the trade must be declared before it is measured on the
holdout again.

---

## 5. Decision

**Adopt P2 (defer-to-anchor), with the anchor's own CALIB-calibrated region.** Layer A stops being
an abstention rule and becomes an **estimator-selection** rule; abstention is layer B alone.

Justification, in the order that matters:

1. **Structural, not performance-driven.** The defect is a category error in the rule's wording
   ("no improvement over the anchor" treated as "no skill"), identified and disclosed **on CALIB
   before the holdout was opened** (`results/B4_MODEL_V1.md` §4). The correction restores the
   rule's intent; it does not rescue a number.
2. **It costs nothing measurable.** Zero difference from P1 where both speak; −0.0001 m
   [−0.0003, +0.0000] against unrestricted v1 over all rows; −0.0177 m [−0.0493, +0.0038] inside
   the defer bucket. The supervisor's condition is met with numbers, not adjectives.
3. **It buys 7.9 points of coverage** (92.00% → 99.93%, 51 113 extra positions) at 0.66 m accuracy.
4. **It removes the one thing a pundit report cannot tolerate** — a silent gap where the system
   knows the player's position to 66 cm and says nothing.
5. **It keeps calibration**: 6/6 within ±5 points at both 50% and 90% on the emitted point, once
   the anchor's region is calibrated for the anchor (§3.3).

**No gate number changes.** Gate 1 (v1 8.10 m vs the B7 bar 11.46 m, 5/6 BEAT + 1 tie) and gate 2
(6/6 at 50% and 90%, PICP 45.5–50.9 / 89.2–90.6) are properties of v1's predictions on the holdout.
This document changes only *which estimator is emitted where, and under what label*. The
reproduction check at the top of the run log re-derives 8.10 / 11.46 exactly. Gate 4 (abstention)
is the only gate this touches, and it was already recorded as "implemented and frozen, not
exercised".

---

## 6. What was implemented

`synthesizer/imputation_v1.emit` now takes the anchor and the CALIB-derived skill buckets and
returns a **source label per prediction**:

- `source="v1"` — v1's corrected point (593 232 rows, 92.00% of holdout, RMSE 8.38 m)
- `source="anchor"` — the B7 anchor's own point, in buckets where v1 has no edge
  (51 113 rows, 7.93%, RMSE 0.66 m)
- `source="abstained"` — layer B rejected it; the emitted coordinate is the **last-seen position**
  with `asserted=False` (447 rows, 0.07%; v1 would have scored 38.40 m there)

`asserted` is retained and is exactly `source != "abstained"`. `r90` is now the region of the point
that was *actually emitted*, and layer B thresholds that region. No downstream consumer can receive
an anchor-sourced or last-seen coordinate believing it came from v1.

Test: `tests/test_imputation_v1.py` (3 tests) — source/coordinate agreement, no silent fallback in
either degenerate case (layer B rejects everything; no skill bucket at all), and that the emitted
region follows the emitted point.

Callers updated: `tools/imputation_b4_v1.py` (prints the literal P0 outcome *and* the adopted P2
source counts; its gate-1/gate-2 code is untouched). Its committed run log was **not** regenerated —
it is the frozen artefact of the 2026-07-25 gate run.

---

## 7. What argues against the change

- **It is still a post-gate rule change.** The correction was disclosed on CALIB before the holdout
  was opened, and P2's numbers were pre-computed as the "restated variant" in the gate report — but
  the *decision* is being made after seeing holdout numbers. The only defence is that the decision
  is structural and the measured cost is −0.0001 m; if the supervisor prefers the literal rule, the
  literal outcome (0.0% asserted) is recorded in §2 and stands.
- **Deferring hides a tie behind a label.** A reader could mistake "the anchor speaks at 0-1s" for
  "the model is good at 0-1s". It is the *anchor* that is good there; v1 adds nothing. The `source`
  field exists so that this is legible in the data, not just in prose.
- **Layer A is now doing no abstention at all.** Under P2 the system's only "I don't know" fires on
  0.07% of samples. That is a genuine weakness of the current freeze (§4) and the reason the
  layer-B recommendation is attached to this decision rather than filed separately.
- **7.93% of asserted output is no longer the novel model.** The headline "our imputer" now emits a
  training-free baseline on one row in thirteen. Honest, and it should be stated whenever the
  coverage figure is quoted.
- **Transfer risk is unchanged and still outranks this.** Nothing here re-tests the censoring
  geometry finding (`results/B4_TRANSFER_M3.md`); a policy that emits *more* v1 predictions inherits
  whatever transfer risk those predictions carry.
