# N1 de-leak — the price of a legitimate SoccerNet-GSR number

Campaign act N1. `results/GSR_TEST_FLAGPLANT.md` §6 found the scored chain reads the evaluation
labels twice and flagged the submission package `DO_NOT_UPLOAD`. This is the measurement of what
those two reads were worth, the label-free replacements, and the first GS-HOTA this project has
produced that no ground-truth file touched.

**Headline: official test-49 GS-HOTA 31.88 (DetA 21.11, AssA 48.15), zero label reads in the
prediction chain. The leaky flag-plant number on the same 49 sequences, reproduced exactly by this
harness, is 35.90. The price of legitimacy is -4.02 GS-HOTA, and 60% of it is four sequences whose
left/right assignment the geometric resolver gets wrong.**

The submission zip has been rebuilt without the `DO_NOT_UPLOAD` flag:
`results/gsr_submission/gsr_testphase_gtfree_7dd2a50a.zip`.

---

## 1. The two leaks

| | what it read | how big a prior |
|---|---|---|
| **team side** | `eval.gsr_score.resolve_team_map` — picks the KMeans-cluster -> `left`/`right` permutation by agreement with GT positions | 1 bit per sequence, but that bit gates **every** player row: get it wrong and the whole clip scores ~0 under `USE_TEAMS` |
| **roster** | `eval.gsr_identity._roster` — the exact `(team, jersey)` set in `Labels-GameState.json` | ~17 slots per sequence instead of the 198 of a free 1..99 x 2-side roster |

Every ablation below is a pure in-memory transform of the cached evidence bundles: the roster and
the leave-one-out gallery-similarity tensor are recomputed from the tracklets the bundle already
carries (`tools/gsr_deleak.py:rebuild_top`, verified to reproduce the cached tensor bit-for-bit on
SNGS-021), and the submitted `attributes.team` is flipped wherever the arm's map disagrees with the
one the cache was written under. Nothing was re-extracted; the whole matrix is CPU.

## 2. Leak cost — valid TEST-38, frozen 33.20 recipe

Solver config unchanged throughout (`results/identity_solver_config_percrop.json`,
sha256 `a7282d9b…` — the same file the flag-plant froze). `fixed` = the no-information map
`{0: left, 1: right}`, i.e. what you get if you refuse to resolve the permutation at all.

| team map | roster | GS-HOTA | Δ | GS-DetA | GS-AssA | identity acc | cov@0.85 | tracklets named | slots/seq |
|---|---|---|---|---|---|---|---|---|---|
| GT | GT | **33.20** | — | 21.81 | 50.55 | 0.4488 | 0.4141 | 2141/2541 | 17 |
| **free** | GT | 32.14 | **-1.06** | 20.64 | 50.04 | 0.4276 | 0.3991 | 2141/2541 | 17 |
| fixed | GT | 26.12 | -7.07 | 13.62 | 50.13 | 0.2830 | 0.3074 | 2168/2541 | 17 |
| GT | full | 29.58 | -3.62 | 17.44 | 50.18 | 0.3676 | 0.3716 | 2513/2541 | 198 |
| GT | **self** | 31.73 | **-1.47** | 20.05 | 50.22 | 0.4239 | 0.3265 | 1536/2541 | 10 |
| free | full | 28.66 | -4.54 | 16.49 | 49.80 | 0.3492 | 0.3716 | 2513/2541 | 198 |
| **free** | **self** | **30.72** | **-2.48** | 18.98 | 49.72 | 0.4033 | 0.3265 | 1536/2541 | 10 |

Paired per sequence against the leaky arm (Wilcoxon, n=38): `free/GT` 0 helped / 2 hurt (p=0.18),
`fixed/GT` 0/14 (p=9.8e-4), `GT/self` 10/28 (p=0.043), `free/self` 10/28 (p=0.023).

Three things the table says:

1. **The leaks are additive.** -1.06 (team) + -1.47 (roster) = -2.53 against a measured joint -2.48.
   They act on different parts of the pipeline and do not interact.
2. **The team map is worth 7.07 GS-HOTA in total** (fixed 26.12 -> GT 33.20) and the geometric
   resolver recovers **6.01 of it, 85%**. It is the single most valuable bit in the chain.
3. **The roster leak is real but small.** Handing the solver the label-derived slot set buys 1.47
   GS-HOTA over building the slots from the sequence's own reads. `AssA` barely moves in any arm
   (49.7-50.6): all of this lands in `DetA`, which is the identity gate.

## 3. The GT-free team side

`eval.gsr_score.resolve_team_map_free`: over all player rows of the clip, **the cluster with the
smaller mean pitch-x is `left`**. A team defends one goal, so whichever way the ball is running the
side being attacked has its defensive line deepest; the two clusters' centroids are therefore
ordered by the goal they defend. Goalkeepers are excluded — their kit is a third colour, so the
two-way kit clustering assigns them essentially at random.

Selection: five members of the same family were measured on the valid split and are flat.

| statistic | valid accuracy |
|---|---|
| row mean, players only (**chosen**) | 56/58 = **0.9655** |
| 5-95% trimmed row mean, players | 56/58 = 0.9655 |
| frame-paired Mann-Whitney U | 56/58 = 0.9655 |
| mean of per-frame centroid gaps | 56/58 = 0.9655 |
| median of per-frame centroid gaps | 55/58 = 0.9483 |
| row mean including goalkeepers | 55/58 = 0.9483 |
| mean of per-track mean x | 45/58 = 0.7759 |
| GK-only median x | 2/12 = 0.1667 (worse than chance) |
| most-extreme-x anchor points | 24/58 = 0.4138 |

The chosen statistic is the cheapest of the four co-leaders (one `groupby`, one comparison), and the
selection exposure is one binary choice among tied variants.

**Held-out accuracy: 45/49 = 0.9184 on the official test split** — sequences never used to choose
anything. Pooled over both splits, **101/107 = 0.9439**.

Negatives:

* **The margin is not a confidence.** On test the four wrong sequences have cluster-mean gaps of
  0.75, 2.48, 2.80 and 3.41 m while the correct ones start at 0.89 m. No abstention threshold
  separates them; a confidence gate was measured and does not exist.
* **The six failures are not degenerate clips.** GT cluster purity on them is 0.878-0.966 and kit
  minority share 0.434-0.486 — the clustering is healthy, the geometry genuinely inverts. These are
  clips where the attacking team's centroid sits behind the defending team's for the whole 30 s.

### The train split could not grade it (negative)

The plan was to grade the resolver on the 57 train sequences, which nothing else has ever touched.
It cannot be done as the pipeline stands: **18 of the 57 train sequences (SNGS-060 … SNGS-077, one
contiguous block) drive `JerseyColorTeamClassifier` to a degenerate fit** — minority cluster share
0.056-0.161 against a median of 0.448 on valid and 0.450 on test — and on those clips the GT
agreement map itself sits at chance (purity 0.504-0.537 measured on the seven that were extracted).
Grading a left/right resolver there measures noise. Valid: **0/58 degenerate. Test: 0/49.** Train:
**18/57**. The remaining 39 train sequences are healthy and would give a third measurement for
~4.5 GPU-hours of extraction; that was not spent.

Cost of finding this: 7 train sequences extracted at stride 1 (~50 min GPU, discarded) plus a
57-sequence 11 s/clip kit survey. Two cheaper probe designs were tried first and both failed
measurably — stride 10 collapses ByteTrack (119 rows/clip against ~11,000 at stride 1), and
truncating the clip destroys the signal the resolver depends on (accuracy 0.9655 over 750 frames,
0.9310 over 375, 0.8621 over 250, 0.7931 over 150; a short window is one attacking phase).

## 4. The GT-free roster

Chosen on the valid **DEV-20** partition — the same 20 sequences the solver's own calibration was
fitted on, so no new set was consumed. The team map is identical to the leaky one on all 20 DEV
sequences, so the roster arms are clean of the team question.

| roster | slots/seq | GS-HOTA (DEV-20) | Δ | identity acc | named |
|---|---|---|---|---|---|
| GT (leaky) | 16.6 | 31.29 | — | 0.4218 | 1022/1249 |
| **self** (**chosen**) | 10.2 | **30.87** | **-0.42** | 0.4022 | 751/1249 |
| self, number opened on both sides | 20.3 | 29.00 | -2.29 | 0.3721 | 1097/1249 |
| full (1..99 x 2 sides) | 198 | 27.01 | -4.28 | 0.3369 | 1217/1249 |

**`self`**: a slot `(team of the reading tracklet, number read)` for every number the sequence's own
per-crop OCR chain read at the frozen rule. Nothing else. It is the honest analogue of the
lineup-sheet prior — the numbers you can see are the numbers you may assign.

Why the full space loses: it names almost everything (2513 of 2541 tracklets on TEST-38 against
1536 for `self`) and is wrong far more often. Opening 198 slots dilutes the per-identity prior by
12x and shrinks the `unknown` background likelihood, so the solver stops abstaining; identity
accuracy falls from 0.4239 to 0.3676 while coverage at precision 0.85 barely moves. Opening each
read number on *both* sides is worse still — it doubles the mutual-exclusion pressure for no
evidence gain.

`self`'s real cost is the players it can never name: 1536 tracklets named against 2141 under the GT
roster. That is not a defect of the roster, it is the jersey-evidence density showing through
without a label to paper over it (`results/EVIDENCE_DENSITY_LAW.md`).

## 5. Pre-registration and the one test run

`results/gsr_deleak_frozen.json` was written **before** the test split was touched, naming
`resolve_team_map_free` + the `self` roster + the unchanged solver config and its sha256. The
official test split was then read once.

| | leaky flag-plant | GT-free | Δ |
|---|---|---|---|
| **GS-HOTA** | **35.90** | **31.88** | **-4.02** |
| GS-DetA | 26.74 | 21.11 | -5.63 |
| GS-AssA | 48.21 | 48.15 | -0.06 |
| GS-LocA | 94.00 | 93.93 | -0.07 |
| IDF1 | 37.96 | 31.01 | -6.95 |
| per-row identity accuracy | 0.5508 | 0.4515 | -0.099 |
| coverage @ jersey precision 0.85 | 0.5175 | 0.4385 | -0.079 |
| tracklets named | 2681/3114 | 2214/3114 | -467 |

The leaky column was **re-derived, not copied**: running the `GT`/`GT` arm through this same harness
on the cached test artifacts returns GS-HOTA 35.8968 / DetA 26.7359 / AssA 48.2121 / LocA 94.0038 /
IDF1 37.9619 — identical to `results/GSR_TEST_FLAGPLANT.md` §3 to four decimals. The -4.02 is
like-for-like.

Paired per sequence (n=49): mean -4.47, median -2.81, helped 11, hurt 38, Wilcoxon p = 1.75e-4.

### Where the 4.02 goes

| group | n | mean Δ GS-HOTA |
|---|---|---|
| sequences the free map gets right | 45 | **-1.94** (the roster leak alone) |
| sequences the free map flips | 4 | **-32.94** |

| flipped sequence | leaky | GT-free | Δ |
|---|---|---|---|
| SNGS-126 | 53.12 | 5.36 | -47.77 |
| SNGS-129 | 11.05 | 4.94 | -6.11 |
| SNGS-131 | 35.15 | 6.78 | -28.38 |
| SNGS-197 | 56.16 | 6.64 | -49.52 |

**Four clips out of 49 carry 60.2% of the total row-weighted loss.** A left/right flip is not a
degradation, it is an annihilation: the clip keeps its positions and its jersey numbers and scores
~5. Note that two of the four (SNGS-197 at 56.16, SNGS-126 at 53.12) were the *best* sequences in
the whole flag-plant table.

This relocates the binding constraint again. `results/GSR_TEST_FLAGPLANT.md` §5 moved it from
jersey evidence to association; for a legitimate number it is now **the team-side permutation**.
Taking the resolver from 91.8% to 98% recovers roughly +2.4 GS-HOTA on test — more than any roster
work available, and it is one bit per clip.

## 6. Package status

`results/gsr_submission/gsr_testphase_gtfree_7dd2a50a.zip` — 24.62 MB, 49 entries,
`tracklab/<SEQ>.json`, 333,274 predictions, the layout verified against the official example
submission. `manifest_gtfree.json` carries **no `DO_NOT_UPLOAD` field** and records instead:

> none. The cluster -> side map comes from `resolve_team_map_free` (geometry only) and the solver's
> roster from the sequence's own OCR reads. No `Labels-GameState.json` is opened anywhere in the
> prediction chain.

The old leaky package (`gsr_testphase_d084d11e.zip`, `manifest.json`) is left in place, flag intact,
as the record of what was measured before.

The GT-free chain's only inputs are `outputs/gsr_test/positions/*.parquet` (detector + tracker +
PnLCalib + kit KMeans), the PRTreID embedding cache, the per-crop OCR votes, and
`results/identity_solver_config_percrop.json`. `grade_teamside` does open the labels — it is a
local diagnostic run *after* the submission is written and feeds nothing.

One subtlety worth stating because a reviewer would ask. The cached base-arm JSONs were written
under the leaky map, so `write_arm` consults `team_map_src` (which *is* the GT map) to decide
whether to flip. The GT map cancels: the emitted side is `free_map[cluster]` either way, identical
to writing the submission from scratch with `free_map` and never opening a label. This was checked
on the artifact rather than argued — over all 49 sequences and 333,274 predictions, **every row's
`team` is the base arm's value, flipped exactly on the 4 sequences where `free != GT` and untouched
on the other 45: 0 violations.**

Where 31.88 sits on the codabench 4365 test-phase leaderboard (15 public entries): between rank 11
(lsmuqi, 33.12) and rank 12; DetA 21.11 would be 11th, AssA 48.15 remains the worst column in the
table. Offered as orientation — the number has not been uploaded.

## 7. Negatives, in one place

1. **The train split cannot grade the team resolver** — 18 of 57 sequences (SNGS-060…077) have a
   degenerate kit clustering (minority share 0.056-0.161, GT purity ~0.50). 39 healthy sequences
   remain unmeasured at a cost of ~4.5 GPU-hours.
2. **No confidence signal exists for the team map.** The cluster-mean gap does not separate right
   from wrong (wrong: 0.75-3.41 m; right: from 0.89 m up). Abstention is not available.
3. **Goalkeeper geometry is useless for the side question** — 2/12 on valid, worse than a coin
   flip, because the two-way kit KMeans assigns GK crops at random.
4. **The full-space roster is strictly worse than the self-roster** at every measurement point
   (DEV -4.28, TEST-38 -3.62). It is not a fallback for sequences the OCR chain reads poorly; it
   removes the solver's abstention pressure and it names indiscriminately.
5. **The valid-split price (-2.48) understates the test-split price (-4.02).** The difference is
   entirely the team resolver: 2 flips in 38 valid sequences against 4 in 49 test sequences, and
   the test flips landed on high-scoring clips. One number does not predict the other.
6. **A bug in the first matrix run inflated every GT-free arm.** `write_arm` compared the
   *retargeted* bundle's map against itself, so `attributes.team` was never actually flipped and the
   first pass reported 32.94 / 29.98 / 31.73 for arms that truly score 32.14 / 26.12 / 30.72. Fixed
   (`team_map_src`) and covered by the module self-check; every number in this document is from the
   corrected run.

## 8. Files

- `tools/gsr_deleak.py` — the whole study (`--train-probe --teamside --dev --freeze --matrix
  --testsplit --demo`).
- `eval/gsr_score.py` — `resolve_team_map_free` added beside `resolve_team_map`.
- `results/gsr_deleak_frozen.json` — the pre-declared GT-free config.
- `results/gsr_benchmark/gsr_deleak_dev.json` — the DEV-20 roster selection (7 arms).
- `results/gsr_benchmark/gsr_deleak_matrix.json` — the TEST-38 leak-cost matrix (7 arms, paired).
- `results/gsr_benchmark/gsr_deleak_teamside.json` — resolver grading per sequence.
- `results/gsr_benchmark/testsplit/gsr_deleak_testsplit.json` — the test-49 GT-free run + manifest.
- `results/gsr_benchmark/testsplit/gsr_deleak_test_repro.json` — the 35.90 reproduction.
- `results/gsr_submission/gsr_testphase_gtfree_7dd2a50a.zip`, `manifest_gtfree.json`.
- `outputs/gsr/deleak_t38_*`, `outputs/gsr_test/deleak_t49_*` — the arm directories.
