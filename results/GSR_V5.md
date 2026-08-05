# v5 completion — the half-tuned recipe, finished. The other half was worth nothing.

`results/CLUSTER_SESSION4.md` shipped v5 (the session-3 CLIP encoder in place of PRTreID) at valid
**TEST-38 37.0344** against a re-derived v4 control of **36.7090**, and declined to spend the one
official test-49 run because the recipe was *knowingly half-tuned*: `results/gsr_v5_frozen.json`'s
`not_retuned` block recorded that the solver's appearance term (`app_gain 10.0`, `sim_none 0.92`) is
in cosine units fitted to PRTreID and is therefore mis-scaled for CLIP's ~4.8x wider scale. It
estimated the equivalents at `app_gain ~3`, `sim_none ~0.67`, and called the 37.0344 a **lower
bound**.

**That premise is now measured, and it is wrong.** Retuning the solver on DEV-20 buys **+0.6118**
GS-HOTA there (37.0711 -> 37.6829, the largest DEV move available from any single knob left in this
recipe) and **-0.0527** on the held-out TEST-38 (37.0344 -> **36.9817**, 7 helped / 9 hurt,
Wilcoxon p = 0.438). The pre-declared gate (**TEST-38 >= 37.2 spends the test-49 run**) is **not
met** and **no test run was spent**. 37.0344 was not a lower bound; it was the number.

Two of the estimates in the `not_retuned` block also split: `app_gain ~3` is exactly the DEV argmax,
and **`sim_none ~0.67` is refuted** — 0.67 costs 0.28 GS-HOTA against the inherited 0.92, which is
itself the DEV optimum. So only one of the two "mis-scaled" knobs was mis-scaled at all.

---

## 1. The hook (the only code change)

`tools.gsr_v4.solve_v4_arm` hard-loaded the frozen `SOLVER_CONFIG` and offered no override, which is
the reason session 4 could not run this sweep. It now takes `app_gain` / `sim_none`, both defaulting
to `None`, through a pure helper:

```
tools.gsr_v4.solver_config(app_gain=None, sim_none=None) -> SolverConfig
```

Both `None` returns the on-disk config untouched, so every on-record arm is byte-identical — asserted
in `python -m tools.gsr_v4 --demo`. `config_key` is deliberately **not** extended with these two
values: they act strictly downstream of the connector and the identity bundles, so every solver arm
correctly re-uses one cached connector partition (which is also why arms 2..n cost 75 s each instead
of 3.5 min). The arm *tag* carries the suffix, so no two arms share a submission directory.
`--split {dev,t38}` was added so the verification arms run through the same harness as the sweep.

## 2. DEV-20 sweep — 21 arms, tuning split only

Bundle held at v5's frozen point (CLIP embedder, floor 0.80, connector **tau 0.450** + jersey gate,
no confusion refit); the ONLY things that move are the two solver scalars. The `10.00 / 0.92` row is
the re-derived half-tuned v5 control and it reproduces session 4's DEV number
(**37.0711**, DetA 23.9202, AssA 57.4538) at every printed digit.

| app_gain | sim_none | GS-HOTA | vs v5 | GS-DetA | GS-AssA | GS-LocA | IDF1 | identity | named |
|---|---|---|---|---|---|---|---|---|---|
| **10.0 (v5 control)** | 0.92 | **37.0711** | — | 23.9202 | 57.4538 | 92.2701 | 38.12 | 0.4473 | 502 |
| 3.0 | 0.55 | 37.4060 | +0.335 | 24.2784 | 57.6339 | 92.2718 | 38.64 | 0.4523 | 503 |
| 3.0 | 0.60 | 37.4060 | +0.335 | 24.2784 | 57.6339 | 92.2718 | 38.64 | 0.4523 | 503 |
| 3.0 | **0.67** (session-4 estimate) | 37.4075 | +0.336 | 24.2859 | 57.6208 | 92.2677 | 38.64 | 0.4524 | 504 |
| 3.0 | 0.75 | 37.5874 | +0.516 | 24.4959 | 57.6778 | 92.2000 | 38.92 | 0.4563 | 504 |
| 3.0 | 0.85 | 37.6200 | +0.549 | 24.5236 | 57.7126 | 92.2035 | 38.96 | 0.4568 | 503 |
| 3.0 | 0.90 | 37.6829 | +0.612 | 24.5796 | 57.7739 | 92.2367 | 39.02 | 0.4576 | 503 |
| **3.0** | **0.92 (FROZEN v5.1)** | **37.6829** | **+0.612** | **24.5796** | **57.7739** | 92.2367 | **39.02** | **0.4576** | 503 |
| 3.0 | 0.95 | 37.6829 | +0.612 | 24.5796 | 57.7739 | 92.2367 | 39.02 | 0.4576 | 503 |
| 3.0 | 1.00 | 37.4792 | +0.408 | 24.3800 | 57.6188 | 92.2216 | 38.75 | 0.4544 | 503 |
| 1.0 | 0.85 | 37.4250 | +0.354 | 24.2570 | 57.7430 | 92.3040 | 38.67 | 0.4517 | 502 |
| 2.0 | 0.85 | 37.6093 | +0.538 | 24.5037 | 57.7264 | 92.2346 | 38.96 | 0.4561 | 503 |
| 5.0 | 0.85 | 37.4854 | +0.414 | 24.3945 | 57.6035 | 92.2226 | 38.76 | 0.4547 | 502 |
| 2.0 | 0.92 | 37.6087 | +0.538 | 24.5000 | 57.7329 | 92.2380 | 38.96 | 0.4561 | 503 |
| 2.5 | 0.92 | 37.6201 | +0.549 | 24.5245 | 57.7109 | 92.2032 | 38.96 | 0.4568 | 504 |
| 3.5 | 0.92 | 37.4854 | +0.414 | 24.3945 | 57.6035 | 92.2226 | 38.76 | 0.4547 | 502 |
| 4.0 | 0.92 | 37.4854 | +0.414 | 24.3945 | 57.6035 | 92.2226 | 38.76 | 0.4547 | 502 |
| 5.0 | 0.92 | 37.4757 | +0.405 | 24.3919 | 57.5796 | 92.2548 | 38.73 | 0.4546 | 503 |
| 2.0 | 0.67 | 37.4005 | +0.329 | 24.2562 | 57.6694 | 92.3014 | 38.64 | 0.4516 | 504 |
| 5.0 | 0.67 | 37.4128 | +0.342 | 24.2959 | 57.6133 | 92.2680 | 38.65 | 0.4526 | 503 |
| 10.0 | 0.67 | **36.7808** | **-0.290** | 23.4184 | 57.7698 | 92.1848 | 37.49 | 0.4391 | 489 |

Three readings, all of which survived to the verification run only in part:

- **`app_gain` is the mis-scaled knob and `sim_none` is not.** Every `app_gain` in [2, 3] beats the
  inherited 10.0 by ~0.55-0.61 at any `sim_none` >= 0.85, while moving `sim_none` to the estimated
  0.67 *loses* 0.28 against leaving it at 0.92. The estimate's logic (place `sim_none` at the same
  relative point between within- and cross-track similarity) is the same logic that got tau right,
  so it is worth naming why it fails here: the solver's similarity is **mean-of-top-3 against an
  OCR-anchored gallery**, not a pairwise track distance, and it is compared against `sim_none` only
  through `exp(app_gain * (s - sim_none))`, in which identities with **no** gallery are also credited
  `sim_none`. Lowering `sim_none` therefore does not move a threshold — it inflates every galleried
  identity against every un-galleried one and against `unknown`, which is a different intervention
  from the one the estimate had in mind.
- **The `app_gain` optimum is a narrow ridge, not the broad plateau tau sat on.** 2.0-3.0 spans
  0.07 GS-HOTA, but 3.5 already costs 0.19 and 10.0 costs 0.61. `GSR_V4.md` §1.2 could freeze at
  the conservative end of a flat region; there is no equivalent defence here, and this is the
  strongest advance warning that the DEV win might not travel.
- **The whole DEV gain is DetA, i.e. the identity gate**: 23.92 -> 24.58 with AssA moving only
  +0.32. Lower `app_gain` makes appearance evidence less decisive, the OCR term relatively stronger,
  identity accuracy rises 0.4473 -> 0.4576, and more rows clear GS-HOTA's binary identity gate.

### 2.1 The DEV gain is two sequences

Only 11 of 20 DEV sequences change at all, and the pooled +0.61 is carried by two of them:

| sequence | delta GS-HOTA (v5.1 - v5) |
|---|---|
| SNGS-093 | **+5.50** |
| SNGS-090 | **+4.44** |
| SNGS-045 | +1.34 |
| SNGS-081 | +1.21 |
| SNGS-078 / 057 / 036 | +0.35 / +0.22 / +0.19 |
| SNGS-033 / 024 / 027 | -0.03 / -0.08 / -0.11 |

Sum of positives 13.25, sum of negatives -0.22. **Two sequences supply 9.94 of the 13.25.** A
20-sequence tuning split cannot distinguish "the knob is right" from "two sequences flipped".

## 3. The freeze

`results/gsr_v5_1_frozen.json`, declared **2026-08-04T19:36:48Z** (= 01:06:48 IST 2026-08-05),
recipe sha256 `41e254604b68e508...`, hash-linked to `results/gsr_v5_frozen.json`
(sha256 `92b4de6c89e05e81...`), which is left untouched as the audit record of session 4's own
pre-declaration.

```
embedder    = tools.clip_embedder.ClipEmbedder, epoch8.pt sha256 9d2d2058...  (v5, unchanged)
connector   = GTA, splitter OFF, tau 0.450, jersey-compatible merge gate ON   (v5, unchanged)
positions   = calibgate select_calibration(3 players / 5.0 m) + fill(10)      (v3, unchanged)
jersey      = per-crop OCR, crop_scale 1.0, aggregation floor 0.80            (v4, unchanged)
solver      = results/identity_solver_config_percrop.json (sha256 a7282d9b..., file NOT edited)
              with app_gain 10.0 -> 3.0 applied at load; sim_none stays 0.92
team map    = resolve_team_map_free (geometry); roster = self                 (v3, unchanged)
```

Ordering evidence (local mtimes): DEV sweep files 01:00 / 01:05 -> **frozen 01:06:48** -> TEST-38
CLIP arms 01:16 -> TEST-38 PRTreID control 01:23.

## 4. Valid TEST-38 — one run, both controls re-derived

`python -m tools.gsr_v4 --dev --split t38 --arms ...`, CLIP arms in one invocation, the PRTreID
control in a second (the embedder is an import-time env switch). Nothing below is quoted: all three
arms were derived in this session.

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 | identity | named | merges | merge prec |
|---|---|---|---|---|---|---|---|---|---|
| v4 control (PRTreID, tau 0.080) | **36.7090** | **24.1180** | 55.8787 | 92.1484 | 37.70 | 0.4421 | 1335/1941 | 1369 | 0.631 |
| v5 (CLIP, tau 0.450, app_gain 10) | **37.0344** | 23.9789 | **57.2018** | 92.1840 | **38.55** | 0.4394 | 999/1554 | 1785 | 0.542 |
| **v5.1 (CLIP, tau 0.450, app_gain 3)** | **36.9817** | 23.9589 | 57.0859 | **92.2152** | 38.47 | 0.4391 | 1003/1554 | 1785 | 0.542 |

**Both controls reproduce their on-record values at every printed digit** (36.7090 and 37.0344), so
the harness is session 4's and the deltas are like-for-like.

| comparison | pooled delta | paired mean | median | helped/hurt | best | worst | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| v5.1 vs v4 control | **+0.2727** | +0.4737 | +0.2758 | 21 / 17 | +4.77 (SNGS-058) | -4.38 (SNGS-088) | 0.1485 |
| v5 vs v4 control | +0.3254 | +0.5719 | +0.3941 | 23 / 15 | +4.75 | -4.38 | 0.0629 |
| **v5.1 vs v5** | **-0.0527** | **-0.0983** | 0.0000 | 7 / 9 | +0.70 (SNGS-050) | **-2.49 (SNGS-080)** | 0.4380 |

(The v5-vs-v4 row reproduces session 4's paired block to every digit it printed — mean +0.5719,
median +0.3941, 23/15, best +4.7475, worst -4.3751, p = 0.06293.)

16 of 38 sequences move at all; positives sum to +1.87 and negatives to -5.60. The DEV pattern
inverts: there is no +5 sequence, and the largest single move is a loss.

**Gate verdict: 36.9817 < 37.2. The gate is NOT met. No test-49 run was spent, no submission was
built, no GPU was booked.** (A test-49 CLIP run would also have needed a fresh ~78-minute laptop-GPU
pass to build `outputs/gsr_test/detembed_cache_clip` — 49 sequences, ~240 MB — since only the 58
valid sequences are cached. That cost was never incurred.)

## 5. What this closes

**The `not_retuned` caveat on `clip-s4-001` is resolved and it did not point the way the session
expected.** The tuned recipe is 0.05 *below* the half-tuned one on held-out data, so 37.0344 is the
honest measurement of the CLIP encoder in this pipeline, not a floor under it. Session 4's decision
to withhold the test run was still the right call procedurally — it just bought a negative instead of
the improvement it was waiting for, which is the point of running the check.

Combined with session 4's §6, the appearance lever now reads: **a +1.82 crop-retrieval mAP encoder is
worth about +0.33 GS-HOTA at p = 0.063, and no amount of rescaling the solver's appearance term adds
to that.** The remaining GS-HOTA on this pipeline is not in appearance.

## 6. Negatives

1. **The retune is a negative on held-out data** (-0.0527, p = 0.438). Reported as the headline, not
   a footnote: the DEV-20 +0.6118 was the largest single-knob move left in the recipe and it did not
   survive contact with 38 unseen sequences.
2. **`sim_none ~0.67` is refuted** (§2). The frozen record's own estimate would have cost 0.28
   GS-HOTA on DEV against changing nothing. The relative-position heuristic that correctly rescaled
   `tau` does not transfer to a parameter that enters an exponent shared with un-galleried identities.
3. **DEV-20 is now demonstrably too small for a 0.6-point decision** (§2.1): two sequences carried
   75% of the gain. Every DEV-only conclusion in this campaign inherits this caveat, including v4's
   tau pick — which was at least defended by a broad plateau and two later verifications.
4. **The v5.1 vs v4 comparison got *less* significant, not more** (p 0.063 -> 0.149). If the campaign
   ever spends a test-49 run on a CLIP arm, the arm to spend it on is **v5 (app_gain 10)**, not v5.1.
5. **Not measured:** whether `app_gain` interacts with `tau` (both were tuned one-at-a-time on DEV-20,
   tau first at the inherited `app_gain`, then `app_gain` at the frozen tau). A joint sweep is ~40
   CPU-minutes and is the only untried thing left in this bundle — but on the evidence of §4 it would
   be tuning a knob whose DEV wins do not transfer.
6. **`r_abstain = 0.0` and `pi_none = 0.7` were not touched.** They are also solver calibration and
   also fitted on PRTreID-era evidence; the brief scoped this task to `app_gain` and `sim_none`.

## 7. Reproduce

```
GSR_EMBEDDER=clip python -m tools.gsr_v4 --dev --tag v5_solver_dev20  --arms <stage A, 9 arms>
GSR_EMBEDDER=clip python -m tools.gsr_v4 --dev --tag v5_solver_dev20b --arms <stage B, 8 arms>
GSR_EMBEDDER=clip python -m tools.gsr_v4 --dev --tag v5_solver_dev20c --arms <stage C, 4 arms>
GSR_EMBEDDER=clip python -m tools.gsr_v4 --dev --split t38 --tag v51_t38_clip --arms \
    '[{"name":"v51_clip_a3s092","floor":"0.80","tau":0.45,"gate":true,"app_gain":3.0,
       "sim_none":0.92},{"name":"v5_clip_halftuned","floor":"0.80","tau":0.45,"gate":true}]'
python -m tools.gsr_v4 --dev --split t38 --tag v51_t38_prtreid --arms \
    '[{"name":"v4_prtreid","floor":"0.80","tau":0.08,"gate":true}]'
```

Self-check: `python -m tools.gsr_v4 --demo` (now asserts the override is default-preserving).
CPU only, ~55 minutes total, no GPU.

## 8. Files

- `tools/gsr_v4.py` — `solver_config()`, `solve_v4_arm(..., app_gain=, sim_none=)`, arm-tag suffix,
  `--split`, and the demo assertion. Default behaviour is byte-identical.
- `results/gsr_v5_1_frozen.json` (new); `results/gsr_v5_frozen.json` untouched.
- `results/gsr_benchmark/gsr_v4_v5_solver_dev20{,b,c}.json`,
  `results/gsr_benchmark/gsr_v4_v51_t38_{clip,prtreid}.json`.
- Logs `outputs/gsr/v5_solver_dev20{,b,c}.log`, `outputs/gsr/v51_t38_{clip,prtreid}.log`.
- No submission, no zip, no commit.
