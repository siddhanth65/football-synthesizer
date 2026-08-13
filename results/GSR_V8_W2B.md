# v8 session W2b — the dead-frame calibration fallback (ours, line-based, paper-derived)

Campaign v8, session W2b. `results/GSR_V8_W2.md` closed the calibration-transfer question with a
negative (+0.4883 against a >= +0.50 bar) and left exactly one constructive residue: **the entire
transferable effect was coverage** — a real per-frame homography on the 1,693 probe rows where our
chain emits no pitch coordinate, worth 1.97 m -> 0.45 m against the interpolating fill on the
frames that have one. W2 §3 named the two candidate owners of that coverage term: (1) a **line-based
solve** (their Table 4 prices line-only calibration at 56.39 against keypoint-only 48.23, and W2
recorded "we have no line solver"), (2) the GPL `sn-banner` NBJW assets as a fallback-only
calibrator (~2 GPU-h).

**CLAIMS-HYGIENE.** The winner's code and weights are **not used in this session at all**. Nothing
of theirs is loaded, run, imported or read. Everything below is our implementation, on our
PnLCalib-based chain, using classical image processing and published ideas.

---

## 1. REGISTRATION (written 2026-08-13, before any arm was built or scored)

### 1.0 Two facts established before registration, and how they changed the plan

Both are population censuses of *our own* artifacts (no arm, no score, no GS-HOTA), and both
contradict a premise the task inherited. They are stated first because §1.4's arms are built on
them.

**(a) The sn-banner escalation is a mechanical no-op. Skipped, not deferred.**
`docs/SOCCERNET_REPO_SWEEP.md` §5 lists NBJW `SV_kp` / `SV_lines` as "weights we did not know
existed". We already run them:

| asset | NBJW release `Content-Length` | our `~/PnLCalib/weights/` |
|---|---|---|
| `SV_kp` | 264,964,645 B | **264,964,645 B** |
| `SV_lines` | 264,857,893 B | **264,857,893 B** |

PnLCalib and No-Bells-Just-Whistles are the same author (`mguti97`); PnLCalib's own README serves
`SV_kp`/`SV_lines` from its release of the *same* files. Build-order step 2 would have spent ~2
GPU-h running, as a "second fallback", the identical keypoint/line network that produced the failure
we are trying to recover from. **It is not run.** (Value of the sweep entry stands for the
Mask2Former weights and the cam-param filter; the calibration half of it is already in our chain.)

**(b) "PnLCalib emits nothing" is not why our probe frames are dead.** Census over the 10 W2 probe
sequences, on the shipped lineage (`positions_v6det` -> calibgate re-gate -> `fill(10)` =
`positions_gate_v6det`), reproducing W2's 1,693 empty rows exactly:

| | frames | rows |
|---|---|---|
| frames whose players carry a coordinate from **their own** homography | 7,210 | — |
| frames alive **only** because `fill_calibration_gaps` bridged them | 114 | — |
| **frames with no pitch coordinate at all (dead)** | **139** | **1,123** |
| rows empty on an otherwise-live frame (`clamp_to_pitch` rejected the point as off-pitch) | — | **570** |
| **total empty rows (W2 §2.7's number)** | | **1,693** |

And the 139 dead frames decompose into three populations, none of which is "no hypothesis":

| population | frames | rows | why it is dead | keypoint reproj err of the top hypothesis | its GT-anchored median error |
|---|---|---|---|---|---|
| **P1** cached, **fewer than 3 detected players** | **79** | 241 | `onpitch_plausible` returns `False` **whenever the frame has fewer than `TRUST_MIN_PLAYERS` foot points, whatever the homography is** | **0.17-0.21 m** | **0.46 m** (p75 0.93; **4 of 79 frames > 5 m**, worst 127 m) |
| **P2** cached, >= 3 players, hypotheses genuinely implausible | 1 | 16 | players project off-pitch / collapsed | 1.53 m | 68.3 m (correctly rejected) |
| **P3** not in the candidate cache at all | **59** | **866** | the cache was built for the *pre-v6det* lineage's dead set; these frames are dead in v6det and were never re-solved | unknown | unknown |
| frames with **zero** PnLCalib hypotheses | **0** | 0 | — | — | — |

Two consequences:

1. **P1 is the calibgate defect one level down.** `generator.postprocess.onpitch_plausible` opens
   with `if len(pts) < min_onpitch: return False` — a frame showing two players cannot pass, so a
   homography that fits the pitch keypoints to 0.17 m and lands within 0.46 m of ground truth is
   discarded for a reason that has nothing to do with the homography. But it is not a free
   relaxation either: **4 of those 79 frames carry a catastrophic homography** (up to 127 m), and
   with two players on screen there is no player-based evidence that can tell them apart.
2. **That is precisely the job for line evidence**, and it is why this session's line work is aimed
   where W2 §3 aimed it. On a player-sparse frame the only thing left in the image that knows where
   the camera is, is the pitch itself. Two uses, one primitive:
   * **support** — do the pitch model's lines, pushed through `H`, actually land on white line
     pixels? (an acceptance test that works with zero players)
   * **solve** — point-to-line correspondences give a linear system for `H` (`l^T H p = 0`), so the
     same evidence can *re-solve* a frame P3 leaves with nothing.

### 1.1 The probe — unchanged from W2

`SNGS-{024, 027, 039, 042, 045, 048, 051, 054, 057, 078}`, the 10 hardest DEV-20 sequences by the
rule registered in `GSR_V8_W2.md` §1.1. The control is re-derived on this machine and must reproduce
`results/gsr_benchmark/gsr_v7_control_dev.json` per sequence or the session is void.

Dead frames are concentrated: **SNGS-024 (64 frames / 335 rows), SNGS-057 (42 / 693), SNGS-051
(21 / 53), SNGS-054 (6 / 25), SNGS-078 (6 / 17)**. The other five probe sequences have **zero** dead
frames. Dead runs reach 28 consecutive frames (SNGS-024 579-606, SNGS-057 513-538), which is why
`fill_calibration_gaps(max_gap=10)` cannot bridge them: the nearest donor is 13 frames away.

### 1.2 The component — `generator/line_calib.py`, ours, CPU, classical

One primitive, three uses, all dead-frame-only:

1. **Line pixels.** Grass-restricted white-ridge mask of a frame: a morphological top-hat isolates
   structures thinner than the kernel (pitch lines) inside the green region, thresholded. Pure
   OpenCV, already a dependency.
2. **Pitch model lines.** The FIFA 105x68 line set in our uncentred metre frame (touchlines, goal
   lines, halfway, both penalty and goal areas, the centre circle as a polyline). Ours, written from
   the pitch spec, ~20 segments.
3. **`line_support(H, mask)`** — sample the model lines, push them into the image through `H^-1`,
   and measure the share of in-image samples that sit within `r` px of a ridge pixel. This is the
   acceptance evidence that replaces the player test when there are no players.
4. **`refit_homography(H0, mask)`** — from an initial `H0` (the neighbouring frame's, propagated
   sequentially through a dead run so each step sees one frame of camera motion), associate ridge
   pixels to the nearest projected model line and solve the linear point-to-line system
   `l^T (H p) = 0` by SVD; 3 iterations. Reports the mean point-to-line residual **in metres** —
   the same currency `MAX_REPROJ_ERROR_M` is denominated in.

**The one threshold, fixed before any arm was built, GT-free and score-free.** `min_support` is set
from the support distribution of homographies **the shipped chain already trusts**: 40 live frames
per probe sequence, each frame's own homography re-derived from its projected rows by the same DLT
`fill_calibration_gaps` uses to choose donors (`--support`, 400 frames). Pooled support **median
0.981, p10 0.857, p05 0.804, p01 0.679** — an accepted calibration puts ~98% of the visible model
lines on real white pixels, which is the instrument validating itself. **`min_support = 0.804`
(the p05)** is registered: a dead frame must look at least as line-consistent as the bottom 5% of
frames we already ship. Per sequence the p05 ranges 0.710 (SNGS-024) to 0.978 (SNGS-051); the
pooled value is used everywhere, no per-sequence tuning.

Every recovered homography goes through the **existing acceptance stack** before it may write a
coordinate: the metre error gate (`MAX_REPROJ_ERROR_M`), `onpitch_plausible` when the frame has
enough players to evaluate it, `clamp_to_pitch` per point, and now `line_support` on top. The fill
stays behind it as the final fallback. Default **OFF**; separate variant directories; nothing on
record is overwritten.

### 1.3 The arms — one variable, four pre-declared, everything else frozen

Nothing upstream of positions moves: the same v6 extraction (`positions_v6det`, S4b + ByteTrack),
the same track ids, the same cached OCR evidence, the same CLIP embeddings, the same box cache.
Only the pitch coordinates of dead-frame rows change. All arms are scored through
`tools.gsr_eiou.run_point` at the frozen v6 point (`e=0.3, rounds=1, w_app=0.5, app_max=0.30`,
embedder `clip_v6det`, tau 0.450, reader `_v6_v6det`), caches cleared per arm.

| arm | what it adds on dead frames |
|---|---|
| **CONTROL** | `positions_gate_v6det` — the shipped lineage, re-derived |
| **A relax** | the **free null hypothesis for all line work**: on a dead frame with fewer than `TRUST_MIN_PLAYERS` foot points, accept the best cached hypothesis on the metre gate alone. No lines. No new evidence. P1 only. |
| **B line-gate** | A, but every acceptance must also clear `line_support`; **plus P3** — the missing hypotheses are computed (PnLCalib on 59 frames, ~1 GPU-min) and subjected to the same stack |
| **C line-refit** | B, plus the donor-anchored **line solve** on every dead frame B still cannot accept — the actual "line-based homography estimation" W2 §3 named |

**A exists to price the line work honestly.** If B == A, the line evidence bought nothing that a
one-line relaxation does not, and this session ships the one-liner and bank the solver as a
negative. Reported per arm: dead-frame recovery rate, recovered-homography accuracy vs GT, per
sequence.

### 1.4 THE PRE-REGISTERED READ (as commissioned, verbatim)

> On the same 10-sequence probe as W2, the fallback arm achieves **mean paired GS-HOTA >= +0.40**
> with **>= 7/10 sequences helped**. Secondary: full DEV-20 >= +0.5 mean.
> Passed -> the component enters W5's bundle. Failed -> banked negative, the fill stays.

**A registered arithmetic note, stated before any scoring and derived from §1.0(b), not from any
score: the `>= 7/10 helped` clause cannot be met by a dead-frame-only stage.** Five of the ten probe
sequences (SNGS-027, -039, -042, -045, -048) contain **zero** dead frames, so their delta is
identically 0.000 and they cannot be "helped"; the ceiling is 5/10. This is a property of the
commissioned scope (dead-frame-only, never touching a frame PnLCalib solved), not of the method.
The clause is reported as it was written and it will be marked **FAIL** on that ground whatever the
means do. The quantity that carries decision weight is therefore stated now, before the fact:

* **primary:** mean paired GS-HOTA over the 10 probe sequences, best arm, bar `>= +0.40`;
* **helped:** reported against both denominators — 10 (as registered, ceiling 5) and 5 (the
  sequences the stage can reach), the second labelled as a secondary read that was **not** the
  commissioned bar;
* **secondary:** full DEV-20 `>= +0.5` mean, only if the probe passes.

### 1.5 Guards

1. Selection is on §1.4 alone. Every other number is descriptive.
2. Nothing of the winner's is used, run or read this session.
3. No TEST-38, no test-49, no submission, **no commit**.
4. Dead-frame-only: an arm that writes a coordinate on a frame the shipped chain already solved is a
   bug, and is asserted against.
5. One GPU job at a time; no pytest suite while the GPU pass runs.
6. If the result contradicts the task's premise, the contradiction is the deliverable — §1.0 is
   already two of those.

---

## 2. RESULTS — VERDICT: the registered read **FAILS at +0.3020** against a `>= +0.40` bar.
## The line solver works; it is worth 62% of what an external SFR network was worth, and that is
## not enough to clear the bar.

**The decisive number: best registered arm (`c`) mean paired gain = `+0.3020` GS-HOTA, 7 helped /
2 hurt, worst `-0.0056`, Wilcoxon `p = 0.098`.** The bar was `>= +0.40`. The component does **not**
enter W5's bundle; the negative is banked and `fill_calibration_gaps` stays. A post-registration
bug fix (§3.4) lifts it to `+0.3894` — still short, by 0.0106.

Four findings, in the order they change the picture:

1. **A line-based homography solver, written from the pitch spec and run on the CPU, recovers 90.6%
   of the rows our chain leaves empty on dead frames, at 0.35-0.57 m from ground truth.** On
   SNGS-057 it solves 62 frames the keypoint chain could not — **41 of them measured this session
   as returning zero hypotheses**, the other 21 alive only through the fill — and writes 719 rows
   at **median 0.35 m, p90 0.73 m, 100% inside the evaluator's 5 m gate** (570 GT-matched). That is more
   accurate than our whole shipped chain's pooled 0.475 m and than the benchmark leader's 0.536 m
   (`GSR_V8_W2.md` §2.4), on rows where we previously had nothing at all.
2. **On the frames the fill was bridging, a real homography beats an interpolated one again** —
   W2 §2.7's finding, reproduced with our own solver: SNGS-057 **1.53 m -> 0.39 m** (p90 2.43 ->
   0.84) over 272 paired GT-matched rows, SNGS-051 2.08 -> 0.53, SNGS-027 0.55 -> 0.23.
3. **The line evidence is worth more as a gate than the relaxation it guards.** The free arm `a`
   (accept a dead frame's hypothesis when the frame is too player-sparse to judge) is a coin toss:
   mean `+0.1125`, 2 helped / 2 hurt, and one clip at **-0.712**. Adding `line_support` (arm `b`)
   refuses 30 of arm `a`'s 88 frames and turns that -0.712 into +0.001, worst case -0.026. The
   line-support test is doing exactly the job §1.0 predicted: telling a good homography from a
   catastrophic one on a frame with two players on screen.
4. **`>= 7/10 helped` passed on the letter and means nothing.** Arm `c` helps 7 sequences, but four
   of those seven move by `<= +0.01` (fill-bridged frames whose coordinates got sharper without
   changing an identity decision). §1.4 registered the ceiling argument before the fact and it was
   half right: the stage does reach the "zero dead frame" sequences, through their bridged rows,
   but not by enough to matter.

### 2.1 Harness control — exact, and the rebuild is byte-identical

| check | result |
|---|---|
| re-derived CONTROL vs `gsr_v7_control_dev.json`, per sequence | **0.0 at every printed digit** (`max_abs 0.0`), in both scoring runs |
| rebuilt control chain (`apply_sequence` + `fill(10)`) vs the on-record `positions_gate_v6det` | **max 0.000000 m**, 0 rows gained, 0 rows lost, all 10 sequences |
| **guard #4** — coordinates on frames PnLCalib solved | **max delta 0.000000 m over 115,101 rows** in every arm |
| the one exception, declared | arms `c`/`d` newly fill **26 rows** on solved frames: the stage's recovered frames become *donors*, so `fill_calibration_gaps` reaches 26 rows it previously could not. The fill sits behind the stage by design; the stage itself still writes only dead frames. |

### 2.2 The P3 population, resolved: PnLCalib returns **nothing** on those frames

§1.0(b) listed 59 dead frames as "not in the cache, unknown". Measured this session (`--cache`,
GPU, 37 s of compute):

| sequence | dead frames run | hypotheses returned |
|---|---|---|
| SNGS-024 | 18 | **0** |
| SNGS-057 | 41 | **0** |

Both models load and run; `FramebyFrameCalib` simply cannot assemble a camera from the heatmaps.
This is `GSR_CALIBGATE.md` §1's "no hypothesis at all" population (13.2% of dead frames there), and
it carries **866 of the 1,123 dead-frame rows (77%)** on this probe. No keypoint-side change can
reach it — arms `a` and `b` recover **zero** of it by construction. It is the line solve's whole
reason to exist.

### 2.3 Dead-frame recovery rate

253 frames are dead **before** the fill (the fill bridges 114 of them; 139 stay dead).

| arm | frames recovered | share | by keypoint | by line solve | rows written by the stage | rows the control had empty |
|---|---|---|---|---|---|---|
| `a` relax | 88 | 34.8% | 88 | 0 | 244 | 214 |
| `b` line-gate | 58 | 22.9% | 58 | 0 | 168 | 158 |
| **`c` line-refit** | **166** | **65.6%** | 58 | **108** | 1,769 | **932** |
| `d` = `c` + §3.4 fix | **180** | **71.1%** | 58 | **122** | 1,916 | **1,018** |

Against the 1,693 rows W2 measured our chain leaving empty: arm `c` fills **932 (55.0%)**, arm `d`
**1,018 (60.1%)**, against the external calibrator's 1,324 (78.2%). Of the 1,123 empty rows that sit
on a *dead frame* — the only ones this stage may touch — arm `d` fills **90.6%**. The other 570
empty rows sit on live frames and were rejected one by one by `clamp_to_pitch` as off-pitch;
recovering those is a tolerance question, not a calibration one, and is out of scope.

Per sequence (frames recovered / rows written by the stage):

| sequence | dead (pre-fill) | `a` | `b` | `c` | `d` |
|---|---|---|---|---|---|
| SNGS-024 | 107 | 45 / 119 | 27 / 80 | 41 / 232 | **55 / 389** |
| SNGS-027 | 1 | 0 | 0 | 1 / 8 | 1 / 8 |
| SNGS-039 | 1 | 0 | 0 | 1 / 20 | 1 / 20 |
| SNGS-042 | 0 | — | — | — | — |
| SNGS-045 | 13 | 0 | 0 | 13 / 198 | 13 / 198 |
| SNGS-048 | 1 | 0 | 0 | 1 / 16 | 1 / 16 |
| SNGS-051 | 22 | 22 / 55 | 22 / 55 | 22 / 55 | 22 / 55 |
| SNGS-054 | 22 | 6 / 25 | 6 / 25 | 19 / 180 | 17 / 150 |
| SNGS-057 | 63 | 0 | 0 | **62 / 1,021** | **62 / 1,021** |
| SNGS-078 | 23 | 15 / 45 | 3 / 8 | 6 / 39 | 8 / 59 |

### 2.4 Accuracy of what was written — GT-anchored, on the rows each arm added

Our player/GK detections matched to the nearest GT person in image space (accepted under 60 px),
error in metres against that GT row's `bbox_pitch` — the instrument of `GSR_V8_W2.md` §1.6. Ground
truth is read only to score; no arm consumes it.

| sequence | arm | rows added | GT-matched | median | p90 | <= 5 m |
|---|---|---|---|---|---|---|
| SNGS-024 | `a` | 119 | 72 | 0.25 | 0.65 | 0.972 |
| SNGS-024 | `b` | 80 | 53 | **0.23** | **0.51** | **1.000** |
| SNGS-024 | `c` | 135 | 103 | 0.52 | 3.35 | 0.932 |
| SNGS-024 | `d` | 221 | 174 | 0.57 | 4.65 | 0.960 |
| SNGS-051 | all | 53 | 42 | 0.99 | 1.26 | 1.000 |
| SNGS-054 | all | 25 | 12 | 0.46 | 1.31 | 1.000 |
| **SNGS-057** | **`c`/`d`** | **719** | **570** | **0.35** | **0.73** | **1.000** |
| SNGS-078 | `a` | 17 | 11 | 0.52 | 1.56 | 1.000 |

The split is instructive: on SNGS-057 (steady camera, plenty of visible markings) the line solve is
**0.35 m**; on SNGS-024 (dead runs of up to 28 frames, sparse markings) it is 0.52-0.57 m with a
**p90 of 3.35-4.65 m** and a few rows past the 5 m gate. The keypoint route, where it is available
at all, is tighter (0.23-0.25 m).

### 2.5 The bridged frames — a real homography beats the interpolated one, our version

Rows the control carries **only** because `fill_calibration_gaps` borrowed a neighbour's geometry,
scored on the identical GT-matched rows for both coordinate sets:

| sequence | paired rows | fill median | arm `d` median | fill p90 | arm `d` p90 |
|---|---|---|---|---|---|
| **SNGS-057** | 272 | 1.53 | **0.39** | 2.43 | **0.84** |
| SNGS-051 | 2 | 2.08 | **0.53** | 2.26 | **0.65** |
| SNGS-027 | 8 | 0.55 | **0.23** | 0.89 | **0.39** |
| SNGS-045 | 143 | 0.39 | **0.30** | 0.88 | 0.97 |
| SNGS-039 | 16 | 0.86 | **0.71** | 1.79 | **1.13** |
| SNGS-054 | 163 | 0.53 | 0.56 | 1.68 | **1.23** |
| SNGS-078 | 84 | 0.74 | **0.71** | 3.48 | 3.29 |
| SNGS-048 | 14 | 0.58 | 0.77 | 0.75 | 0.96 |
| SNGS-024 | 458 | 1.97 | 1.98 | 3.98 | 4.63 |

Seven of nine improve, and the two that do not are the two the solver could not recover (SNGS-024,
whose coordinates stay the fill's, and SNGS-048's single frame). This is `GSR_CALIBGATE.md` §2's
"a frame's own homography is not blurrier than its neighbour's, it is sharper", measured a third
time and now against a solver of our own.

### 2.6 THE DECISIVE TABLE — paired GS-HOTA on the probe

Every arm re-derived through one harness: same detections, same track ids, same OCR evidence, same
embeddings, same connector, same solver. The variable is the pitch coordinate of dead-frame rows.

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| **CONTROL** | 42.9371 | 31.3141 | 58.8769 | 93.6346 | 45.8127 |
| `a` relax (no lines) | 42.9935 | 31.4043 | 58.8620 | 93.6430 | 45.8316 |
| `b` line-gate | 43.0264 | 31.3789 | 58.9997 | 93.6403 | 45.8899 |
| **`c` line-refit (registered)** | **43.1756** | **31.5052** | **59.1709** | **93.6922** | **45.9695** |
| `d` = `c` + seed-chain fix (post-registration) | **43.2222** | **31.5448** | **59.2240** | **93.7236** | **46.0019** |
| *(reference)* W2 `t3`, the external calibrator's coverage | 43.2713 | 31.5168 | 59.4120 | 93.5717 | 46.1202 |

**GS-LocA rises in every arm** (93.6346 -> 93.6922 for `c`, 93.7236 for `d`): the recovered rows are
not bought with localisation quality — the column `GSR_CALIBFILL_TEST.md`'s post-hoc fill had to pay
and the one W2's `t3` also lost (93.5717).

Per sequence, paired against the control:

| sequence | CONTROL | delta `a` | delta `b` | delta `c` | delta `d` |
|---|---|---|---|---|---|
| SNGS-024 | 47.343 | +1.617 | +1.160 | +1.150 | **+2.044** |
| SNGS-027 | 50.130 | 0.000 | 0.000 | +0.001 | +0.001 |
| SNGS-039 | 35.844 | 0.000 | 0.000 | +0.009 | +0.009 |
| SNGS-042 | 36.685 | 0.000 | 0.000 | 0.000 | 0.000 |
| SNGS-045 | 49.198 | 0.000 | 0.000 | -0.006 | -0.006 |
| SNGS-048 | 48.492 | 0.000 | 0.000 | -0.003 | -0.003 |
| SNGS-051 | 45.778 | +0.247 | +0.261 | +0.261 | +0.261 |
| SNGS-054 | 39.508 | -0.026 | -0.026 | +0.002 | -0.021 |
| SNGS-057 | 34.499 | 0.000 | 0.000 | **+1.601** | **+1.601** |
| SNGS-078 | 46.067 | **-0.712** | +0.001 | +0.004 | +0.007 |

| arm | mean | median | helped | hurt | worst | best | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| `a` | +0.1125 | 0.000 | 2 | 2 | **-0.7122** | +1.6165 | 0.875 |
| `b` | +0.1396 | 0.000 | 3 | 1 | -0.0259 | +1.1599 | 0.375 |
| **`c`** | **+0.3020** | +0.0029 | **7** | 2 | **-0.0056** | +1.6007 | 0.0977 |
| `d` (not registered) | +0.3894 | +0.0040 | 6 | 3 | -0.0207 | **+2.0439** | 0.2031 |

| the pre-registered read | required | measured | |
|---|---|---|---|
| mean paired GS-HOTA gain, best **registered** arm | **`>= +0.40`** | **`+0.3020`** (`c`) | **FAIL** |
| sequences helped | **`>= 7/10`** | 7/10, four of them by `<= +0.01` | pass, vacuously (§2 finding 4) |
| on the 5 sequences the stage can reach | *(secondary, not the bar)* | mean +0.6036, **5 helped / 0 hurt** | — |
| full DEV-20 | *(only if the probe passes)* | **not run** | — |

**VERDICT: FAIL. The component does not enter W5's bundle; the negative is banked and the fill
stays.** No arm was re-run to move it, no threshold was touched, and the DEV-20 confirmation was not
spent because the registration conditions it on a probe pass.

### 2.7 Where the missing 0.10 went, and it is one clip

Arm `c`'s shortfall against the bar is **0.098 GS-HOTA over 10 sequences = 0.98 points on one
sequence**, and SNGS-024 is exactly that sequence: W2's external calibration scored **+3.004** there
against our `c`'s +1.150 (`d`'s +2.044). SNGS-024 is the probe's worst case for line evidence — dead
runs of 19-28 frames, sparse visible markings (median line support of the frames the gate refused:
**0.465**, against 0.883 on the ones it accepted), and 66 of its 107 dead frames unrecovered by `c`.
A stage that recovered SNGS-024 the way it recovers SNGS-057 would clear the bar on this probe. It
does not, and §3.4 measures why rather than assuming it.

---

## 3. Mechanism and diagnostics

### 3.1 The line-support instrument validates itself on frames we already trust

400 live frames (40 per sequence), each judged by its own homography re-derived from its projected
rows: **support median 0.981, p10 0.857, p05 0.804, p01 0.679**. A calibration the chain already
ships puts ~98% of the visible model lines on real white pixels. `min_support = 0.804` (the pooled
p05) was fixed from this, before any arm existed, without ground truth and without a score.

### 3.2 What the gate refuses, and why that is the point

On the 79 player-sparse dead frames that have a hypothesis, the top hypothesis fits the pitch
keypoints to 0.17-0.21 m and lands a median **0.46 m** from ground truth — but **4 of the 79 are
catastrophic** (up to 127 m). Arm `a` accepts all of them and pays -0.712 on SNGS-078; arm `b`
refuses 30 frames and the disaster with them. That is the whole case for line evidence as an
acceptance test, and it is measured rather than argued.

### 3.3 The line solve is a real estimator, not a re-parameterisation of its seed

The ICP is seeded from a neighbouring frame's homography and could in principle just return it. It
does not: in the synthetic self-check a seed whose own mean point-to-line residual is **5.26 m**
converges to **0.083 m**; on real dead frames the accepted fits sit at a median residual of
**0.111-0.122 m** over 8-16 distinct model segments and several hundred point constraints. The
end-to-end evidence is stronger still — on SNGS-057 the seed *is* what the fill would have used, and
the fill's coordinates are 1.53 m from GT where the solved ones are 0.39 m.

### 3.4 The bug the diagnosis found — a refusal breaks the seed chain

`recover_dead_frames` documents "propagated forward through the dead run so each step sees one frame
of camera motion rather than the whole gap", but as first written it recorded a seed only when a
frame was **accepted**. One refusal therefore reset the chain and every later frame of that run was
initialised from ever further away. Measured on the frames arm `c` missed:

| outcome | median seed gap | median converged residual | median support |
|---|---|---|---|
| accepted | **1 frame** | 0.111 m | 0.883 |
| refused (support) | **15 frames** | 0.122 m | **0.465** |

A far seed lands the ICP in a locally-consistent but globally-wrong minimum: the residual stays
small (it has locked onto *some* lines) while support collapses (it does not explain the model).
The support test catches it — the system working — but the cause is the broken chain.

Arm `d` is `c` with `chain_seed=True`: a solve that is self-consistent (residual within the metre
gate) seeds the next frame even when it is refused for shipping. It recovers 180 frames instead of
166, 1,018 empty rows instead of 932, and scores `+0.3894` instead of `+0.3020`. **This is a
post-registration measurement and does not change the verdict** (§1.5 guard 1); it is reported
because the bug is in code that stays in the tree and the fixed behaviour is what the docstring
always claimed. `chain_seed=False` reproduces the arm that was scored.

---

## 4. Negatives, limits, and what was NOT done

1. **The registered read FAILED at +0.3020 against +0.40.** Even the post-registration bug fix
   (+0.3894) does not clear it — by 0.0106, which after W2's 0.0117 is this campaign's second
   hair-miss and should be read as "this effect is worth about 0.3-0.4 GS-HOTA on this probe", not
   as "it nearly passed".
2. **The escalation's step 2 was cancelled on a byte count, not on a measurement of its quality.**
   `sn-banner`'s NBJW `SV_kp`/`SV_lines` are the same files (264,964,645 B / 264,857,893 B) our
   PnLCalib already loads, so running them as a "second fallback" would re-run the network that
   produced the failure. Not spent: ~2 GPU-h. A genuinely independent keypoint opinion would have to
   come from a different architecture, and W2 §2.4 says its expected accuracy gain is negative.
3. **No DEV-20, no TEST-38, no test-49, no submission, no commit.** The registration conditions the
   DEV-20 read on a probe pass.
4. **`METRICS_VERSION` was NOT bumped.** The stage is default OFF and no shipped artifact changes;
   the rebuilt control is byte-identical to the on-record parquet (§2.1). Bumping it for a banked
   negative would falsely invalidate every on-record artifact. If the orchestrator ships the stage,
   the bump belongs with that decision.
5. **The line solve is only as good as its seed, and its seed is temporal.** Every recovery here
   starts from a neighbouring frame's homography. On a sequence that opens dead, or whose dead run
   exceeds the camera motion the ICP's 40 px coarse radius can absorb, there is no recovery at all —
   1 frame in SNGS-057 and 66 in SNGS-024 (arm `c`) failed exactly this way. A seedless solve
   (labelling detected line segments against the model outright) was **not** built; it is the honest
   next step and a materially bigger piece of work.
6. **`min_support` is one number from one distribution** (p05 of 400 live frames). It was not swept
   and deliberately not tuned against GS-HOTA — but it is a threshold, and on a different corpus or
   broadcast style it needs re-measuring. Per-sequence p05 ranges 0.710 to 0.978, so even the pooled
   value is a compromise.
7. **The image knobs are physical but untuned:** top-hat kernel 21 px and ridge threshold 24 are
   sized for 1080p frames with 4-8 px lines. Not swept. A different resolution needs the kernel
   rescaled, which is why it is a named constant and not a literal.
8. **Accuracy is population-dependent and the tail is real.** SNGS-024's recovered rows have a p90
   of 3.35-4.65 m with 4-7% outside the evaluator's 5 m gate, against SNGS-057's 0.73 m p90 and
   100%. Quoting "0.35 m" as the solver's accuracy would be quoting its best clip.
9. **The stage is dead-frame-only; the chain around it is not fully frozen.** Recovered frames
   become donors, so `fill_calibration_gaps` newly reaches 26 rows on solved frames (§2.1). No
   coordinate on a solved frame changes (max delta 0.000000 m), but the row count does.
10. **Detections, tracks and every GPU artifact below positions were held fixed**, as in W2. A fresh
    extraction with this stage enabled could move detection-time decisions this experiment cannot
    see (`GSR_CALIBGATE.md` negative #5 is the same caveat).
11. **Spend: 37 s of GPU** (59 frames of PnLCalib) plus ~50 min of CPU (arms, GT, support,
    diagnostics). The line solve costs **0.2-0.5 s per dead frame on one CPU core**, against the
    external network's 0.34 s/frame on a GPU for *every* frame.

## 5. What this means for the campaign (for the orchestrator, not a result)

- **The calibration coverage term is now priced from both sides.** An external SFR network was worth
  `+0.4883` on this probe (W2, failed a `>= +0.50` bar); our own CPU line solver is worth `+0.3020`
  registered / `+0.3894` fixed (failed a `>= +0.40` bar). Two independent implementations agree the
  effect is **real, small, and concentrated in two sequences**. Nothing further should be spent on
  calibration coverage without a reason bigger than 0.4 GS-HOTA.
- **What is worth keeping even though the arm failed:** `generator/line_calib.py` is a validated,
  dependency-free instrument that can (a) *score* any homography against the image it came from and
  (b) *solve* one from lines. The scoring half is the more useful one — the first GT-free per-frame
  calibration check in this repo that does not need players, and exactly the kind of evidence
  `GSR_CALIBGATE.md`'s player-count trust rule is a poor substitute for. That is a separate and
  larger question than dead-frame coverage.
- **The two things the campaign still has not priced remain what W2 named**: IDATR (+2.97 by the
  leader's own ablation) and the detector/tracker substrate. Three sessions have now confirmed by
  measurement that the leader's advantage is not in calibration or in identity reading.

## 6. Reproduce

```
python -m tools.gsr_w2b_linefall --census                                     # CPU ~2 min
python -m tools.gsr_w2b_linefall --cache                                      # GPU 37 s (59 frames)
python -m tools.gsr_w2b_linefall --support                                    # CPU ~2 min
python -m tools.gsr_w2b_linefall --positions --min-support 0.804 --arms-list a,b,c,d   # CPU ~10 min
python -m tools.gsr_w2b_linefall --gt --arms-list a,b,c,d                     # CPU ~1 min
python -m tools.gsr_w2b_linefall --arms --arms-list a,b,c,d                   # CPU ~7 min
python -m tools.gsr_w2b_linefall --demo && python -m generator.line_calib
pytest tests/test_line_calib.py tests/test_calibrate.py tests/test_postprocess.py
```

## 7. Files

- `generator/line_calib.py` (new): the pitch model, the ridge mask, `line_support`,
  `solve_point_on_line` / `refit_homography` (the line solve), the `recover_dead_frames` stage, and
  a `--demo` self-check that synthesises a frame from a known homography and requires exact recovery
  plus ICP convergence from a 5.26 m seed error.
- `tools/gsr_w2b_linefall.py` (new): the census, the GPU cache top-up, the `min_support` calibration
  on live frames, the four arms, the GT-anchored accuracy report, the scored comparison, its demo.
- `tests/test_line_calib.py` (new): 4 cases — model geometry, exact line-solve recovery, support/ICP
  behaviour, and that the stage touches only dead frames and that every gate bites.
- `results/GSR_V8_W2B.md` (this file);
  `results/gsr_benchmark/gsr_v8_w2b_{census,cache,support,positions,gt}_probe.json`,
  `gsr_v8_w2b_positions_probed.json`, `gsr_v8_w2b_arms.json`.
- `outputs/gsr/positions_w2b{a,b,c,d}_v6det{,_eiou}/`, `outputs/gsr/calib_candidates_w2b/`,
  `outputs/gsr/deleak_w2b_*`, `outputs/gsr/w2b_*.log` (gitignored). No on-record artifact was
  overwritten or deleted.
