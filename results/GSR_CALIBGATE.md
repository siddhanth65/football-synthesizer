# N2c — the calibration dropout was never a calibration failure

Campaign act N2c. `results/GSR_ASSOCIATION.md` §3.2 diagnosed the dropout as PnLCalib emitting
homographies that pass the 2.0 m keypoint-reprojection gate while projecting **every** detected
player off the pitch. `results/GSR_CALIBFILL_TEST.md` patched it *post hoc* by interpolating a
neighbour frame's homography: +1.49 GS-HOTA on the official test split, but **GS-LocA -0.56**. This
act was commissioned to fix that at the gate.

**Headline: the homographies were right. Frame-level measurement on the 8 calibration-sick valid
sequences shows 96.4% of the dead frames project every detected player ONTO the pitch, at 82-98%
precision against ground truth — they were discarded by our own post-hoc heuristic
`reject_implausible_frames`, whose ">= 8 players spanning >= 25 m" rule assumes a wide shot and
voids every zoomed one. §3.2 of `GSR_ASSOCIATION.md` is retracted below. Repairing the rule (3
players / 5 m, chosen on DEV-20) and restoring the coordinates from each frame's own recomputed
homography is worth **+1.99 GS-HOTA over the shipped fill arm on valid TEST-38 (34.00 vs 32.01)**
with **GS-LocA +0.18** — where the fill had to pay LocA for its gain.**

This document is written incrementally; each section is timestamped at the point it was written.

---

## 0. Pre-declaration (written 2026-08-01T09:38:37Z, before any DEV or valid scoring)

**The design.** The gate gains a second, GT-free term. `PnLCalib.heuristic_voting` already computes
**18 camera hypotheses** per frame (3 keypoint subsets x 6 RANSAC thresholds), sorts them by pixel
reprojection error, returns one and throws 17 away.
`generator.calibrate.PnLCalibCalibrator.candidates` returns all 18 in exactly that preference order;
`generator.calibrate.select_calibration` walks the list and takes the first hypothesis that passes
**both** the 2.0 m keypoint gate **and** `generator.postprocess.onpitch_plausible` — the same
">= 8 players inside the pitch polygon, spanning >= 25 m" test `reject_implausible_frames` already
applies *after* the fact, moved forward so it can *choose* a homography instead of only voiding a
frame. It is a pure function of our own detections and the homography; no ground truth anywhere.

The "re-solve with a different keypoint subset" is therefore **free** (the solves already happen),
and the fallback chain is: plausible hypothesis on this frame -> (none) temporal fill
(`fill_calibration_gaps`, frozen `max_gap = 10`) -> (none) frame stays dead.

**The bar, pre-declared.** Both baselines are on the frozen GT-free recipe (`free` team map, `self`
roster, solver config sha256 `a7282d9b...`), valid TEST-38:

| lineage | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| no-fill (`GSR_DELEAK.md` §2) | 30.7166 | 18.9799 | 49.7167 | **92.1229** | 29.672 |
| fill `max_gap=10` (`GSR_CALIBFILL_TEST.md` §2) | 32.0115 | 19.9234 | 51.4392 | **91.8767** | 30.824 |

The gate fix ships only if, on the full valid split at frozen settings, it **beats the fill arm on
GS-LocA and does not lose GS-HOTA against it**. A tie with the fill arm = report and stop: no test
spend, no package.

---

## 1. The frame-level diagnosis — and it refutes the premise

`python -m tools.gsr_calibgate --cache --worst 8` (GPU, 4,093 frames, 55 min) re-ran PnLCalib on
every frame the 8 calibration-sick valid sequences emit **no** pitch position for, storing all 18
hypotheses per frame; `--diag` then audits them on CPU. Ground truth is read only to score the
audit column. Raw: `results/gsr_benchmark/gsr_calibgate_diag_worst8.json`,
`gsr_calibgate_frames_worst8.parquet` (one row per frame).

| sequence | dead frames | no hypothesis at all | stock pick passes the 2.0 m gate | >half the players off-pitch | killed by "< 8 on pitch" | killed by "span < 25 m" | median players detected | median projected span | **stock-H precision vs GT** | **GT recall if kept** |
|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-024 | 550 | 53 | 488 | 3 | 287 | 203 | 6 | 21.6 m | 0.962 | 0.772 |
| SNGS-027 | 423 | 0 | 422 | 0 | 104 | 275 | 10 | 18.6 m | 0.964 | 0.914 |
| SNGS-028 | 376 | 54 | 318 | 4 | 121 | 191 | 9 | 21.0 m | 0.968 | 0.869 |
| SNGS-033 | 495 | 0 | 484 | 1 | 469 | 8 | 5 | 33.7 m | 0.872 | 0.678 |
| SNGS-034 | 645 | 0 | 645 | 0 | 13 | 628 | 16 | 19.2 m | **0.983** | 0.938 |
| SNGS-037 | 511 | 81 | 423 | 4 | 75 | 339 | 10 | 21.2 m | 0.932 | 0.902 |
| SNGS-055 | 465 | 320 | 122 | 33 | 144 | 0 | 6 | 27.0 m | 0.825 | 0.457 |
| SNGS-056 | 388 | 0 | 388 | 0 | 5 | 322 | 14 | 24.0 m | 0.957 | 0.930 |
| **total** | **3,853** | **508** | **3,290** | **45** | **1,218** | **1,966** | — | — | — | — |

Pooled over the 3,345 dead frames that have any hypothesis:

| | count | share |
|---|---|---|
| stock pick passes the 2.0 m keypoint gate | 3,290 | **0.984** |
| stock pick projects **100%** of the frame's players **onto** the pitch | 3,225 | **0.964** |
| stock pick projects more than half of them off-pitch | 45 | 0.013 |
| would pass the old `(>= 8 players, >= 25 m span)` trust rule | 161 | 0.048 |
| would pass a `(>= 3 players, >= 5 m span)` trust rule | 3,274 | 0.979 |
| **needs a hypothesis other than PnLCalib's own pick** | **26** | **0.008** |
| already-live control frames whose pick changes | 4 / 240 | 0.017 (RANSAC noise) |

**The premise of this task — and of `GSR_ASSOCIATION.md` §3.2 — is wrong.** The homographies do not
project players off the pitch: on 96.4% of the dead frames **every single detected player lands on
the pitch**, and those projections are **82.5-98.3% within the evaluator's 5 m tolerance of a real
ground-truth player**, recovering 45.7-93.8% of the GT rows on the frames they were denied. The
calibration was right; it was thrown away.

The thing throwing it away is `generator.postprocess.reject_implausible_frames`, whose rule
`>= 8 players on the pitch spanning >= 25 m` encodes a **wide-shot assumption**. A zoomed broadcast
frame shows 5-10 players spanning 19-24 m and fails it — 1,218 dead frames for having too few
players *visible*, 1,966 for spanning too little *pitch*, 3,184 of 3,345 in total. It is a heuristic
written to catch "sparse keypoints cramming everyone into a corner", and on this data it never fires
for that reason: **not one** of the 8 sequences has a meaningful population of collapsed
projections.

Two consequences for the design of §0:

1. **The plausibility term the task asked for is the defect, not the fix.** Adding
   `reject_implausible_frames`' own test to the gate changes nothing, because that test is what
   kills these frames. Measured directly: with the (8, 25) thresholds the new gate rescues 4.8% of
   dead frames; the fall-through re-solve is worth **26 frames out of 3,345 (0.8%)** at any
   threshold, because all 18 hypotheses come from the same keypoint detections and agree.
2. **The repair is a threshold, not a solver.** The gate machinery is still needed — the discarded
   coordinates are gone from the cached parquets, so a homography must be recomputed to restore them
   — but the thing being fixed is `reject_implausible_frames`, and it is fixed by weakening it to
   `TRUST_MIN_PLAYERS = 3`, `TRUST_MIN_SPAN_M = 5.0` (chosen on DEV-20, §2).

This also explains `GSR_CALIBFILL_TEST.md` mechanically: `fill_calibration_gaps` never re-runs the
trust rule, so its real mechanism was **bypassing the wide-shot rule**, not repairing a bad
homography — and it paid -0.56 GS-LocA for using a neighbour's interpolated geometry to do it.

**Cache faithfulness, and that the gate does not disturb healthy frames.** Every cached sequence
also stores 30 already-live *control* frames. Over the **1,739 controls in all 58 valid sequences**:
**19 (1.1%) change their pick and 6 (0.35%) are newly rejected** — the `cv2.RANSAC` noise floor —
and the recomputed homography reproduces the coordinates the original extraction wrote to a median
**0.257 m** (p90 0.875 m, p99 3.08 m; part of that is the parquet's rolling-median smoothing).
Recovered rows are therefore the same calibration the pipeline already trusted elsewhere, not a
different one, and the change is confined to the frames that had nothing.

**508 of the 3,853 dead frames (13.2%) have no hypothesis at all** — PnLCalib returns nothing (320
of them in SNGS-055 alone). No gate change can touch those; they remain the temporal fill's job,
which is why the shipped recipe keeps `fill_calibration_gaps(max_gap=10)` behind the new rule.

---

## 2. DEV-20 — choosing the trust rule (written 2026-08-01T12:15Z, before the valid run)

Partition: the declared DEV-20 (`eval.gsr_identity.split_sequences`, every third valid sequence).
`python -m tools.gsr_calibgate --cache --seqs <DEV-20>` then `--sweep`. Base arm (positions ->
submission, no jersey/connector/solver) — the same fast instrument `GSR_ASSOCIATION.md` §4.1 chose
`max_gap` with. Raw: `results/gsr_benchmark/gsr_calibgate_sweep_dev20.json`.

| arm | rows recovered | `loc_assoc` GS-HOTA | DetA | AssA | **LocA** | `gs_hota_full` |
|---|---|---|---|---|---|---|
| control: no fill | — | 49.02 | 61.56 | 39.06 | 92.69 | 16.32 |
| control: `fill(10)` | 12,919 | 50.73 | 64.58 | 39.89 | 92.44 | 16.53 |
| gate at the **old** (8, 25) | 3,948 | 49.66 | 62.62 | 39.42 | 92.66 | 16.48 |
| gate at the old (8, 25) + fill | 3,948 | 50.84 | 64.92 | 39.85 | 92.38 | 16.48 |
| **gate at (3, 5)** | **27,644** | **53.28** | **69.19** | 41.07 | **92.70** | 16.57 |
| **gate at (3, 5) + fill** | 27,644 | **53.47** | **69.44** | **41.21** | **92.64** | **16.61** |
| gate at (1, 0) | 27,840 | 53.32 | 69.24 | 41.10 | 92.70 | 16.56 |
| gate at (1, 0) + fill | 27,840 | 53.51 | 69.49 | 41.24 | 92.64 | 16.61 |

**Control check: the no-fill row (49.02 / 61.56 / 39.06) and the fill row (50.73 / 64.58 / 39.89)
reproduce `GSR_ASSOCIATION.md` §4.2 exactly** — same harness, the repair is the only variable.

Read:

* **The gate at the original thresholds is worth +0.64 HOTA** — it confirms the diagnosis rather
  than repairing anything. Almost all of that comes from the 4.8% of dead frames whose stock
  homography happened to satisfy the wide-shot rule.
* **The trust rule is the whole lever: +4.26 HOTA / +7.63 DetA over no fill, +2.55 / +4.61 over the
  fill arm.** It recovers 27,644 rows against the fill's 12,919, because it is not limited to frames
  within 10 of a live donor.
* **LocA moves the right way.** The gate at (3, 5) scores **92.70**, above the fill's 92.44 and even
  a hair above the untouched control's 92.69: a frame's own homography is not blurrier than its
  neighbour's, it is sharper. This is the column the post-hoc fill had to pay.
* **Removing the floor entirely (1, 0) buys +0.04** — nothing. **Frozen: `TRUST_MIN_PLAYERS = 3`,
  `TRUST_MIN_SPAN_M = 5.0`**, the conservative end, plus `fill_calibration_gaps(max_gap=10)` behind
  it for the 13.2% of dead frames PnLCalib cannot calibrate at all.

### 2.1 DEV-20 on the full GT-free solver arm

`--compare`: the frozen recipe (`free` team map + `self` roster + solver config `a7282d9b...`), all
three arms re-derived through one harness. `results/gsr_benchmark/gsr_calibgate_compare_dev20.json`.

| arm | GS-HOTA | GS-DetA | GS-AssA | **GS-LocA** | IDF1 |
|---|---|---|---|---|---|
| no fill | 30.8734 | 19.6501 | 48.5120 | 92.2643 | 29.5848 |
| `fill(10)` | 31.4041 | 19.6903 | 50.0921 | 92.1247 | 29.6184 |
| **gate(3, 5) + fill(10)** | **32.8276** | **20.6351** | **52.2272** | **92.3107** | **30.8786** |

Deltas: **+1.95 GS-HOTA over no-fill, +1.42 over the fill arm**, with **LocA above both** (+0.046 /
+0.186). Paired per sequence (n = 20, Wilcoxon): vs no-fill mean +2.70, helped 18, hurt 2,
p = 8.5e-4; vs fill mean +2.22, median +0.33, helped 15, hurt 5, worst **-0.18**, best +13.09,
p = 1.7e-3. Against the fill arm the worst case is -0.18 — the gate is a near-dominating change,
where the fill's own DEV win came with a -17.74 tail (a team-side flip) against the no-fill control.

**The pre-declared bar of §0 is met on DEV.** The frozen recipe is written to
`results/gsr_calibgate_frozen.json` (hash-linked to `gsr_calibfill_frozen.json`) before the valid
run below.

---

## 3. Valid TEST-38 — the one verification run at frozen settings (2026-08-01T12:38Z)

`python -m tools.gsr_calibgate --compare --seqs <TEST-38> --tag t38 --fill-gap 10`. One invocation,
three arms, all re-derived: nothing is quoted. Frozen recipe
`results/gsr_calibgate_frozen.json` (declared 12:08Z, **before** this run began at 12:26Z).
Raw: `results/gsr_benchmark/gsr_calibgate_compare_t38.json`.

| arm | GS-HOTA | GS-DetA | GS-AssA | **GS-LocA** | IDF1 |
|---|---|---|---|---|---|
| no fill (control) | 30.7166 | 18.9799 | 49.7167 | 92.1229 | 29.6720 |
| `fill(10)` (control) | 32.0115 | 19.9234 | 51.4392 | 91.8767 | 30.8238 |
| **gate(3, 5) + fill(10)** | **34.0028** | **21.1920** | **54.5640** | **92.0586** | **32.4898** |

**Both controls reproduce their on-record numbers to four decimals** — `GSR_DELEAK.md` §2's
30.7166 / 18.9799 / 49.7167 / 92.1229 / 29.672 and `GSR_CALIBFILL_TEST.md` §2's
32.0115 / 19.9234 / 51.4392 / 91.8767 / 30.824. Same harness; the repair is the only variable.

| against | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| the no-fill lineage | **+3.29** | +2.21 | +4.85 | **-0.06** | +2.82 |
| the fill lineage | **+1.99** | +1.27 | +3.12 | **+0.18** | +1.67 |

**The pre-declared bar is met: +0.18 GS-LocA against the fill arm and +1.99 GS-HOTA.** Against the
untouched no-fill control LocA is 0.06 lower — 20.7% more pitch rows for six hundredths of a
localisation point, where the fill paid 0.25 for half as many.

Paired per sequence (n = 38, Wilcoxon):

| comparison | mean | median | helped | hurt | worst | best | p |
|---|---|---|---|---|---|---|---|
| gate vs no fill | **+4.54** | +1.67 | **37** | 1 | **-0.02** | +22.25 | 2.2e-11 |
| gate vs `fill(10)` | **+2.63** | +0.42 | 25 | 12 | **-0.37** | +13.54 | 1.0e-4 |

The 12 sequences where the gate trails the fill lose **at most 0.37 GS-HOTA** (SNGS-091 -0.37,
SNGS-079 -0.31, then -0.13 and smaller); the gains are +13.54 (SNGS-037 18.24 -> 31.78), +13.21
(SNGS-034 10.00 -> 23.21), +12.11 (SNGS-025), +11.20 (SNGS-028), +10.81 (SNGS-086). Against the
no-fill control the worst case over 38 sequences is **-0.02**: this repair has no disaster tail,
where the fill's own test run had one (-15.34 on SNGS-190, `GSR_CALIBFILL_TEST.md` §4.1).

**Rows.** The gate recovers **65,482 of 425,084 rows (15.4%)** against the fill's 7.9%; pitch-carrying
rows go 341,971 -> 412,884 (**+20.7%**). It is not limited to frames within 10 of a live donor.

**The team-side map gets safer, not just luckier.** Free-map accuracy on TEST-38: no-fill 36/38,
fill 37/38, gate 37/38. Exactly one sequence moves — SNGS-034 — and the *margin* on it is what
matters: no-fill picks `right` (wrong) at 0.87 m, the fill picks `left` (right) at **0.26 m**, the
gate picks `left` at **2.99 m**. `GSR_DELEAK.md` negative #2 and `GSR_CALIBFILL_TEST.md` negative #1
both flagged that a sub-metre margin is the single largest risk in this recipe (it cost -15.34 on
test); the gate turns that coin-flip into a decision with 11x the evidence behind it.

---

## 4. The one test run — official test-49 GS-HOTA 33.37 -> 35.40 (2026-08-01T15:20Z)

`python -m tools.gsr_calibgate --solve-test --out-dir outputs/gsr_test`: a single invocation —
both controls, re-gate, connector, solve, score, legitimacy audit, package, zip re-score. The
candidate cache for the 49 test sequences took 2 h 16 m on the laptop GPU (16,510 frames, 49 npz,
8.1 MB); everything after it is CPU. Raw:
`results/gsr_benchmark/testsplit/gsr_calibgate_testsplit.json`.

| arm | GS-HOTA | GS-DetA | GS-AssA | **GS-LocA** | IDF1 |
|---|---|---|---|---|---|
| GT-free, no repair (control) | 31.8777 | 21.1107 | 48.1520 | 93.9336 | 31.0059 |
| GT-free + `fill(10)` (the shipped v2, control) | 33.3722 | 22.2578 | 50.0547 | 93.3688 | 32.3179 |
| **GT-free + gate(3, 5) + fill(10)** | **35.3991** | **24.0218** | **52.1825** | **93.4003** | **34.0911** |

**Both controls were re-derived, not copied, and reproduce their on-record numbers at every printed
digit** (`GSR_DELEAK.md` §5: 31.8777 / 21.1107 / 48.1520 / 93.9336 / 31.0059;
`GSR_CALIBFILL_TEST.md` §4: 33.3722 / 22.2578 / 50.0547 / 93.3688 / 32.3179). The +2.03 is
like-for-like.

| against | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| the no-repair lineage | **+3.52** | +2.91 | +4.03 | -0.53 | +3.09 |
| the shipped fill lineage | **+2.03** | +1.76 | +2.13 | **+0.03** | +1.77 |

Paired per sequence (n = 49, Wilcoxon):

| comparison | mean | median | helped | hurt | worst | best | p |
|---|---|---|---|---|---|---|---|
| gate vs no repair | +5.28 | +4.09 | 45 | 3 | -15.40 | +30.69 | 3.4e-8 |
| gate vs `fill(10)` | **+3.18** | +1.59 | **39** | 9 | **-1.06** | +19.48 | 1.0e-7 |

Best: SNGS-143 **27.06 -> 46.54** (+19.48), SNGS-144 +12.44, SNGS-116 +11.15, SNGS-147 +10.95,
SNGS-129 +8.94. Worst: SNGS-135 -1.06, then -0.19, -0.17, -0.16, -0.06. The -15.40 against the
no-repair control is SNGS-190, inherited unchanged from the fill lineage (see below), not caused
here.

**Rows.** The gate recovers **84,211 of 460,324 rows (18.3%)** where the fill recovered 9.6%;
pitch-carrying rows go 340,899 -> 435,432 (**+27.7%**).

**The team-side resolver: same 45/49, larger margins.** Free-map accuracy is 45/49 for all three
arms and the wrong set is identical ({126, 131, 190, 197}). Two sequences differ across arms, and
both are margin stories:

| sequence | no repair | + fill | + gate | GT |
|---|---|---|---|---|
| SNGS-129 | left, 0.75 m (wrong) | right, 3.44 m (right) | **right, 6.69 m** (right) | right |
| SNGS-190 | left, 0.41 m (right) | right, 0.36 m (wrong) | right, 0.84 m (wrong) | left |

SNGS-190 is the -15.34 the fill paid on record; the gate inherits it and does not deepen it
(-15.40). But every margin grows: the recipe's biggest single risk (`GSR_DELEAK.md` negative #2 —
a sub-metre side margin is not a confidence) is being answered with more geometry, not less.

**The identity columns.** Per-row identity accuracy 0.4515 (no repair) -> 0.4470 (fill) -> 0.4456
(gate); tracklets named 2214/3114 -> 2196/3095 -> 2135/3031; coverage @ jersey precision 0.85
0.4385 -> 0.4327 -> **0.4342**. Same direction as `GSR_CALIBFILL_TEST.md` negative #2: recovered
rows are worth more than the marginal naming accuracy they dilute, and coverage@0.85 actually
recovers slightly relative to the fill arm.

## 5. The package and its legitimacy audit

`results/gsr_submission/gsr_testphase_gtfree_calibgate_85f63db4.zip`

| check | result |
|---|---|
| entries | 49, all `tracklab/<SEQ>.json` (the verified official layout) |
| predictions | **424,654** (v2: 376,264) |
| size | 32.3 MB |
| **extract-and-score of the zip itself** | GS-HOTA **35.39913** / DetA 24.0218 / AssA 52.1825 / LocA 93.4003 / IDF1 34.0911 — **identical to the arm score at every digit** (`results/gsr_submission/zip_selfscore_gtfree_calibgate.json`) |
| **legitimacy audit** (`tools.gsr_calibfill.verify_gtfree`) | **0 violations** over 49 sequences and 424,654 predictions; flipped on exactly the 4 sequences {SNGS-126, SNGS-131, SNGS-190, SNGS-197} where the free map disagrees with the GT-agreement map |
| manifest | `manifest_gtfree_calibgate.json`, recipe hash `85f63db4...`, no `DO_NOT_UPLOAD` |
| prior packages | `gsr_testphase_gtfree_7dd2a50a.zip` (v1) and `..._calibfill_ab823742.zip` (v2) untouched |

The prediction chain still opens no `Labels-GameState.json`: the trust rule reads only our own
detections and the homography, and the team side comes from `resolve_team_map_free` (geometry).

Orientation only (nothing uploaded): 35.40 would sit between rank 9 and rank 10 on the 15-entry
codabench 4365 test-phase board, two places above where 33.37 sat.

---

## 6. Negatives, and what this does not say

1. **The task's premise was wrong and the design it specified is worth +0.64 GS-HOTA.** An on-pitch
   plausibility term at the gate, at the thresholds the codebase already used, changes almost
   nothing (§1, §2). The measured lever is the *weakening* of that same rule. Anyone reading only
   the headline should read §1 first: this act's main product is a **retraction**, not a feature.
2. **The re-solve lever is empty.** Walking PnLCalib's other 17 hypotheses rescues **26 of 3,345**
   dead frames (0.8%). All 18 come from the same keypoint heatmaps and agree; the code path stays
   because it costs nothing, but no one should expect anything from it. Same for the "different
   init" idea: PnLCalib's grid already spans 3 keypoint subsets x 6 RANSAC thresholds.
3. **GS-LocA is still below the untouched control on test** (93.400 vs 93.934, -0.53) and the
   margin over the fill arm there is only **+0.03** — much thinner than the +0.18 on valid TEST-38.
   The recovered frames are zoomed ones with intrinsically noisier geometry; the honest statement is
   "the gate does not pay LocA for its rows the way the fill did", not "the gate improves LocA".
4. **SNGS-190 is still lost** (-15.40 vs the no-repair control). The gate grows its margin from
   0.36 m to 0.84 m but does not flip it back. One binary side decision remains the largest
   single-sequence risk in this recipe.
5. **This is a repair applied to cached artifacts, not a re-extraction.** The shipped code path
   (`extract_positions` now calibrates *after* detection and passes the frame's foot points to
   `calibrate_frame`) is what a fresh run would do, but no split has been re-extracted end to end
   under it. Because `cv2.RANSAC` is randomised, a fresh extraction's coordinates would differ from
   these by a median 0.26 m (§1).
6. **The weakened thresholds now apply to the ManU broadcast pipeline too, unmeasured.** On the 17
   processed ManU matches **6.9% of frames currently carry no pitch position** (worst:
   `palace_manutd` 18.1%, `liverpool_manutd` 14.9%) — the same rule, the same zoomed frames. That
   upside is real but needs its own re-extraction (or the same candidate-cache trick) and its own
   validator before anyone claims it. `METRICS_VERSION` is bumped to `2026.08.1` accordingly.
7. **`MIN_ONPITCH_PLAYERS` / `MIN_PITCH_SPAN_M` were deliberately left at 8 / 25 m.** They are still
   the DLT donor rule for `fill_calibration_gaps` (8 points for a robust fit is a numerical
   requirement, not a trust heuristic) and the explicit defaults `generator/pose_carry.py` and
   `tools/pose_carry_probe.py` are on record with, so `results/pose_carry_probe.md` reproduces
   unchanged. Its conclusion ("a close-up is not recoverable at any pose") should nonetheless be
   re-read in light of §1: it was measuring a threshold, not the physics.
8. **`fill_calibration_gaps` is kept and still contributes.** 13.2% of dead frames have no
   hypothesis at all; on DEV-20 the fill behind the gate is worth +0.19 `loc_assoc` HOTA. Dropping
   it would trade 0.19 HOTA for 0.06 LocA — recorded as the alternative, not taken.
9. **Nothing was uploaded.** 35.40 is a locally-scored number on the official split, with the same
   scorer, the same layout, and a zip that re-scores to itself. The submission decision belongs with
   the orchestrator and the 1/day cadence.

## 7. Reproduce

```
python -m tools.gsr_calibgate --cache --worst 8                     # section 1 (GPU, 55 min)
python -m tools.gsr_calibgate --diag --worst 8                      # section 1 (CPU)
python -m tools.gsr_calibgate --cache --seqs <DEV-20 then TEST-38>  # GPU, ~1.4 h
python -m tools.gsr_calibgate --sweep --seqs <DEV-20> --tag dev20   # section 2 (CPU, ~30 min)
python -m tools.gsr_calibgate --compare --seqs <DEV-20> --tag dev20 # section 2.1
python -m tools.gsr_calibgate --freeze                              # section 3
python -m tools.gsr_calibgate --compare --seqs <TEST-38> --tag t38  # section 3
python -m tools.gsr_calibgate --cache --out-dir outputs/gsr_test --seqs <test-49>   # GPU, 2.3 h
python -m tools.gsr_calibgate --solve-test --out-dir outputs/gsr_test \
    --results-dir results/gsr_benchmark/testsplit                   # sections 4-5 (CPU, ~19 min)
```

Self-checks: `python -m tools.gsr_calibgate --demo`,
`pytest tests/test_calibrate.py tests/test_postprocess.py tests/test_pose_carry.py`.

## 8. Files

- `generator/postprocess.py` — `onpitch_plausible` (new), `TRUST_MIN_PLAYERS` / `TRUST_MIN_SPAN_M`
  (new; `reject_implausible_frames` now defaults to them), `MIN_ONPITCH_PLAYERS` /
  `MIN_PITCH_SPAN_M` unchanged for the donor + pose-carry callers.
- `generator/calibrate.py` — `CalibCandidate`, `select_calibration`,
  `PnLCalibCalibrator.candidates`; `calibrate_frame` gained an optional `foot_points`.
- `generator/extract.py` — detection now runs **before** calibration so the gate can see the
  frame's own players.
- `generator/temporal_calib.py` — pass-through for `foot_points`.
- `core/pitch.py` — `METRICS_VERSION` -> `2026.08.1`.
- `tools/gsr_calibgate.py` — the whole measurement (`--cache/--diag/--sweep/--compare/--freeze/
  --solve-test/--demo`).
- `tools/gsr_calibfill.py` — `gta_arm` gained `jersey_dir` (default unchanged).
- `tests/test_calibrate.py` — 4 new cases incl. the regression guard that `select_calibration`
  without foot points is byte-identical to the old behaviour.
- `results/gsr_calibgate_frozen.json`, `results/gsr_benchmark/gsr_calibgate_{diag,frames}_worst8.*`,
  `gsr_calibgate_sweep_dev20.json`, `gsr_calibgate_compare_{dev20,t38}.json`,
  `testsplit/gsr_calibgate_testsplit.json`.
- `results/gsr_submission/gsr_testphase_gtfree_calibgate_85f63db4.zip`,
  `manifest_gtfree_calibgate.json`, `zip_selfscore_gtfree_calibgate.json`.
- `outputs/gsr{,_test}/calib_candidates/`, `positions_gate/`, `eval_calibgate_*`,
  `identity_bundles_percrop_gate/`, `deleak_t{38,49}*` — artifacts. No original was overwritten.
