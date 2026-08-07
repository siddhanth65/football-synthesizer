# v7 session V4 step 1 — the association ceiling, re-measured on the v6 stack (2026-08-07)

`results/GSR_ASSOCIATION.md` measured the linking oracle on the **old** detector, on **unrepaired**
calibration, on a ByteTrack partition: proxy `assa_perfect_link` 0.627 (~68 AssA measured) against
a shipped 0.345. Everything under that number has since changed — the calibration gap repair, the
S4b detector, the EIoU re-association — and the v7 plan pre-registered a re-measurement before any
GPU is spent on a tracker. This is that read. CPU only, no GPU, no re-extraction, no tracker change.

**Headline: the ceiling is real and it is FRAGMENTATION, not coverage. On the current DEV-20 stack
a perfect linker on our own v6 detections reaches proxy AssA 0.8642 against the shipped chain's
0.6283 — +23.6 proxy points, +24.8 on the measured `loc_assoc` scale (66.07 -> ~90.9). A
merge-only oracle (what a connector alone can do — it can never split a contaminated tracklet)
reaches 0.7415, +11.9. Coverage is worth only +8.5 by the same instrument, and 19 of 20 sequences
are fragmentation-dominated. PATH VERDICT: CAMELTrack.**

The calibration/coverage lever that N2 chose has been spent well: GT pitch-space recall on DEV-20
is now **0.9155** (it was 0.745 on the valid split pre-repair) and the frame calibration rate is
**0.9807** (was 0.750). That is exactly why the fragmentation term now dominates.

---

## 1. What was measured, and on what

Instrument unchanged: `tools/gsr_assoc_diag.py` rebuilds HOTA's own `A(i,g) = n/(N_g + M_i - n)`
from a greedy nearest-GT assignment at the evaluator's 5 m tolerance, then re-scores
counterfactuals. Three additions this session, all default-preserving (`--diag`, `--coverage`,
`--merge` behave exactly as before):

| addition | why |
|---|---|
| `partition_report(..., positions_subdir=)` | the v6 stack's positions live in `positions_gate_v6det[_eiou]`, not `positions` |
| `submission_positions()` | reads a materialised submission back into the positions schema, so the **shipped** partition (tracker + connector + solver) is measurable on the same instrument |
| `assa_merge_only()` | `assa_perfect_link` lets the oracle SPLIT as well as merge; a connector cannot. This is the merge-only ceiling |

Five partitions, the same 20 DEV sequences, the same GT:

| | detector | tracker | source |
|---|---|---|---|
| A | shipped HF YOLO | ByteTrack | `positions_gate` |
| B | shipped HF YOLO | EIoU (v5 frozen point) | `positions_gate_eiou` |
| C | **S4b (v6det)** | ByteTrack | `positions_gate_v6det` |
| **D** | **S4b (v6det)** | **EIoU e0.3 r1 w_app0.5 a0.30** | `positions_gate_v6det_eiou` — **the v6 tracker output** |
| **E** | | + GTA connector tau 0.450 + MILP solver | `deleak_v6detdev_clip_e0.3r1w0.5a0.3` — **what is scored** |

**Harness control.** Re-scoring E's submission directory reproduces the on-record v7 control
exactly: `gs_hota_full` GS-AssA **68.28680940240496** against `gsr_v7_control_dev.json`'s
**68.28680940240496**, GS-HOTA 51.8083. The partition being diagnosed is the shipped one.

### 1.1 Proxy re-validation (the 1.083 factor is now 1.051)

`assa_hat` on the shipped partition against the official scorer's per-sequence GS-AssA, n = 20:

| measured config | Spearman | Pearson | proxy mean | measured mean | scale |
|---|---|---|---|---|---|
| `loc_assoc` | **+0.842** | +0.864 | 0.6283 | 0.6604 | **1.051** |
| `gs_hota_full` | +0.696 | +0.749 | 0.6283 | 0.6660 | 1.060 |

Weaker than N2's +0.971, and that is expected: N2 validated a geometry-only proxy against a
geometry-only arm (raw ByteTrack, no connector), while E's partition has been through an identity
solver. `loc_assoc` is the config the counterfactuals are quoted on, as in N2.

## 2. The ceiling — the table

Means over the 20 DEV sequences (proxy scale; x100x1.051 for the `loc_assoc` scale).

| partition | `assa_hat` | `assa_merge_only` | `assa_perfect_link` | `assa_perfect_det` | frag/GT | Herfindahl | dom. share | tracks |
|---|---|---|---|---|---|---|---|---|
| A base-det + ByteTrack | 0.3954 | 0.6336 | 0.8064 | 0.4583 | 6.40 | 0.3654 | 0.5101 | 79.9 |
| B base-det + EIoU | 0.3829 | 0.7003 | 0.8091 | 0.4446 | 11.26 | 0.3467 | 0.4974 | 170.9 |
| C v6det + ByteTrack | 0.4828 | 0.6896 | 0.8627 | 0.5445 | 4.78 | 0.4575 | 0.5950 | 53.6 |
| **D v6det + EIoU (v6 tracker)** | **0.4489** | **0.7415** | **0.8642** | 0.5063 | 8.08 | 0.4174 | 0.5618 | 109.9 |
| **E shipped (D + connector + solver)** | **0.6283** | 0.6936 | 0.8678 | 0.7093 | 5.06 | 0.5811 | 0.7209 | 43.0 |

*Anchor, same instrument, same 20 sequences, the pre-repair stack (`positions`, old detector, no
calibration fill): `assa_hat` **0.3362**, `assa_perfect_link` **0.6486**, `assa_perfect_det`
0.4673, matched share 0.7098, det recall 0.7037. N2's valid-58 figures were 0.3451 / 0.6272 /
0.4944, so DEV-20 is not a soft subset of the old measurement either.*

Three readings:

1. **The ceiling rose because of the DETECTOR + the calibration repair, not because of EIoU.**
   A -> C (detector swap, ByteTrack both sides) moves `assa_perfect_link` 0.8064 -> 0.8627; C -> D
   (EIoU) moves it 0.8627 -> 0.8642, i.e. nothing. The old 0.6486 -> 0.8064 step is the calibration
   repair (`positions` -> `positions_gate`). Linking headroom is created by coverage, exactly as
   N2 §3.3's quartile table predicted ("`assa_perfect_link` climbs 0.379 -> 0.821 with calibration
   health").
2. **EIoU trades raw AssA for merge-only ceiling.** C -> D drops `assa_hat` 0.4828 -> 0.4489 while
   raising `assa_merge_only` 0.6896 -> 0.7415. That is the S5 mechanism ("purity bought with
   fragmentation") measured on the new detector for the first time.
3. **The connector has already burned some of its own ceiling.** E's `assa_merge_only` (0.6936) is
   *below* D's (0.7415): the GTA connector's wrong merges (measured merge precision **0.5432** on
   this arm) lock contamination in that no later merge-only pass can undo. What is left on the
   shipped partition for a further merge-only pass is 0.6936 - 0.6283 = **+0.065 proxy (+6.9)**.

### 2.1 Capture fraction

| | proxy | `loc_assoc` scale |
|---|---|---|
| D, the tracker's own output | 0.4489 | 47.2 |
| E, after the connector + solver | 0.6283 | 66.07 *(measured, not scaled)* |
| merge-only oracle on D | 0.7415 | ~77.9 |
| **split+merge oracle on D (the linker ceiling)** | **0.8642** | **~90.9** |
| perfect detection on E (fragmentation kept) | 0.7093 | ~74.6 |

The connector realises **43.2%** of the split+merge headroom available at D (**61.3%** of the
merge-only headroom). N2 measured 24.5% for the same connector on the old stack, so the connector
is doing materially better on the new partition — and still leaves the larger half on the table.

## 3. Fragmentation or coverage? — FRAGMENTATION, 2.9 : 1

The decomposition the plan's path rule turns on, on the shipped partition E:

| counterfactual | proxy delta vs shipped | `loc_assoc` scale |
|---|---|---|
| **fix linking** (`assa_perfect_link` on D - `assa_hat` on E) | **+0.2359** | **+24.79** |
| fix linking, merge-only | +0.1132 | +11.89 |
| **fix coverage + contamination** (`assa_perfect_det` - `assa_hat`, both on E) | **+0.0810** | **+8.51** |

**Ratio 2.91 : 1 in favour of fragmentation, and the split is not carried by outliers: 19 of the
20 sequences are fragmentation-dominated** (the exception is SNGS-033, +0.100 link vs +0.185 det).
Per sequence:

| seq | `assa_hat` (E) | measured `loc_assoc` AssA | link headroom | det headroom | dominant |
|---|---|---|---|---|---|
| SNGS-021 | 0.7582 | 76.95 | 0.1490 | 0.0681 | FRAG |
| SNGS-024 | 0.5282 | 50.40 | 0.2294 | 0.1336 | FRAG |
| SNGS-027 | 0.5093 | 60.98 | 0.3985 | 0.0305 | FRAG |
| SNGS-030 | 0.6207 | 77.67 | 0.3328 | 0.0232 | FRAG |
| SNGS-033 | 0.6859 | 68.10 | 0.0998 | 0.1846 | **COV** |
| SNGS-036 | 0.8140 | 78.93 | 0.1452 | 0.0175 | FRAG |
| SNGS-039 | 0.4082 | 45.92 | 0.4327 | 0.0732 | FRAG |
| SNGS-042 | 0.5197 | 56.63 | 0.3564 | 0.0543 | FRAG |
| SNGS-045 | 0.6028 | 66.51 | 0.1888 | 0.1419 | FRAG |
| SNGS-048 | 0.5802 | 57.66 | 0.2963 | 0.0743 | FRAG |
| SNGS-051 | 0.6421 | 65.70 | 0.1880 | 0.1256 | FRAG |
| SNGS-054 | 0.5344 | 55.65 | 0.3218 | 0.0662 | FRAG |
| SNGS-057 | 0.5449 | 63.07 | 0.2039 | 0.1056 | FRAG |
| SNGS-078 | 0.6395 | 59.77 | 0.2482 | 0.0690 | FRAG |
| SNGS-081 | 0.7452 | 77.09 | 0.1401 | 0.0835 | FRAG |
| SNGS-084 | 0.6142 | 63.16 | 0.2253 | 0.0953 | FRAG |
| SNGS-087 | 0.7601 | 77.16 | 0.1256 | 0.0679 | FRAG |
| SNGS-090 | 0.7040 | 69.87 | 0.1958 | 0.0693 | FRAG |
| SNGS-093 | 0.6307 | 76.61 | 0.2791 | 0.0513 | FRAG |
| SNGS-096 | 0.7245 | 72.99 | 0.1606 | 0.0851 | FRAG |

### 3.1 The coverage side, measured directly

`coverage_split` on the **v6det gated + gap-filled** positions, GT-row weighted over DEV-20, with
N2's valid-58 pre-repair figures beside it:

| | N2, valid-58, pre-repair | **this session, DEV-20, v6 stack** |
|---|---|---|
| GT rows covered in image space | 0.8427 | 0.8904 |
| GT rows covered in **pitch space** (what the evaluator sees) | 0.7453 | **0.9155** |
| detected in image, then lost to the calibration | **0.1443** | **0.0101** |
| net image-minus-pitch recall difference | 0.0974 | **-0.0251** |
| frames with any pitch output | 0.7540 | **0.9807** |
| GT rows carrying no pitch label at all | — | 0.0064 |

**The calibration dropout is closed.** It was between 10 and 14 points of coverage loss; it is now
1.0 point, and the net difference has gone *negative* — image recall is now the smaller number,
because the image-space test is a foot-point-within-half-a-box-width proxy, not an IoU match, and
it now under-counts rather than over-counts. The residual coverage loss is 13.3% of GT rows
(`matched_share` 0.8672), i.e. genuine misses, not projection failures.

### 3.2 Contamination — the GT purity audit on the new detector

`tools.gsr_eiou.partition_purity`, DEV-20, before the connector. Ground truth read only to score.

| partition | row purity | tracklets | contaminated | frag/GT |
|---|---|---|---|---|
| base-det + ByteTrack | 0.8969 | 1,523 | 173 | 6.52 |
| base-det + EIoU | 0.9416 | 3,262 | 106 | 11.47 |
| **v6det + ByteTrack** | 0.9055 | 1,027 | 115 | 4.61 |
| **v6det + EIoU (current)** | **0.9395** | 2,116 | **92** | 7.91 |

The two base-det rows reproduce `GSR_EIOU.md` §5 **exactly** (0.8969 / 1,523 / 173 / 6.52 and
0.9416 / 3,262 / 106 / 11.47), so this is the same audit. On the new detector EIoU still buys
purity with fragmentation, at a smaller absolute cost (7.91 frags/GT against 11.47).

### 3.3 The fragment profile

Mean share of a GT identity held by its k-th largest predicted fragment, pooled over 428 DEV-20 GT
identities (427 on the base-det partitions):

| partition | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | >8 | unmatched |
|---|---|---|---|---|---|---|---|---|---|---|
| A base-det + ByteTrack | 0.512 | 0.178 | 0.073 | 0.031 | 0.014 | 0.007 | 0.004 | 0.003 | 0.005 | 0.173 |
| B base-det + EIoU | 0.500 | 0.166 | 0.070 | 0.031 | 0.017 | 0.011 | 0.007 | 0.005 | 0.021 | 0.173 |
| C v6det + ByteTrack | 0.596 | 0.179 | 0.056 | 0.018 | 0.008 | 0.004 | 0.002 | 0.001 | 0.003 | 0.133 |
| **D v6det + EIoU** | 0.563 | 0.175 | 0.064 | 0.026 | 0.013 | 0.008 | 0.005 | 0.003 | 0.010 | 0.133 |
| **E shipped** | **0.722** | 0.097 | 0.025 | 0.010 | 0.005 | 0.003 | 0.001 | 0.001 | 0.002 | 0.133 |

N2's equivalent row was 0.453 / 0.146 / 0.053 / ... / **0.302 unmatched**. Two structural changes:
the unmatched column halved (0.302 -> 0.133, the coverage repair), and the shipped chain now holds
0.722 of a GT identity in one track against N2's 0.453. **The prize is still one merge per player**:
on D, joining rank-1 to rank-2 alone moves the dominant share 0.563 -> 0.738, and on the shipped
partition rank-2 still holds 0.097 of every identity.

## 4. Is DEV-20 representative?

The v6det extraction exists for the 20 DEV sequences only (`positions_gate_v6det` = 20 files), so
**valid-58 on the current detector is not measurable without a ~4.4 GPU-hour re-extraction and was
not run.** What is cheap is the same partitions on the *old* detector, where all 58 exist:

| partition | split | `assa_hat` | `assa_perfect_link` | `assa_perfect_det` | frag/GT |
|---|---|---|---|---|---|
| `positions_gate` | valid-58 | 0.4042 | 0.7992 | 0.4695 | 7.11 |
| `positions_gate` | DEV-20 | 0.3954 | **0.8064** | 0.4583 | 6.40 |
| `positions_gate_eiou` | valid-58 | 0.3931 | 0.8016 | 0.4570 | 11.85 |
| `positions_gate_eiou` | DEV-20 | 0.3829 | **0.8091** | 0.4446 | 11.26 |

DEV-20 sits within **0.008** of valid-58 on the ceiling and within 0.013 on every column. The
subset is not soft on this quantity.

## 5. PATH VERDICT

Pre-registered thresholds (plan `floating-leaping-peacock.md`, V4): *ceiling < +5 AssA over current
=> EIoU param retune only (CPU); +5..+12 => CAMELTrack; > +12 => CAMELTrack anyway (SUSHI = v8).
If the loss is coverage not fragmentation, no tracker fixes it — EIoU retune regardless.*

| test | threshold | measured | band |
|---|---|---|---|
| ceiling over current, split+merge oracle | +5 / +12 | **+24.79** AssA (`loc_assoc` scale; +23.59 unscaled, +19.36 under a linear proxy->measured fit) | **> +12** |
| ceiling over current, merge-only oracle | +5 / +12 | **+11.89** AssA | **+5..+12** |
| coverage-dominated override? | link headroom vs det headroom | +24.79 vs +8.51 (**2.91 : 1**), 19/20 sequences fragmentation-dominated | **does NOT fire** |

**VERDICT: CAMELTrack.** Both readings of the ceiling clear +5 by a wide margin, and the two land
in the two different CAMELTrack bands (`+5..+12` and `> +12`) — which route to the same decision,
so the verdict does not depend on which oracle is the fair one. SUSHI stays v8 per the plan. The
EIoU-retune path is **not** indicated: the smallest defensible ceiling (+11.9, merge-only) is still
2.4x the +5 bar, and the coverage override does not fire because coverage is worth only +8.5 while
fragmentation is worth +24.8.

**One clause that can still override this.** The plan's budget contingency stands: *"if V2+V3
exceed 30 h combined, V4 takes the EIoU-retune path regardless of ceiling."* This document decides
the technical question only; the budget question belongs to the orchestrator after V2/V3 land.

**What step 2 should aim at, concretely.** Not "more merges" — the connector already merges 1,380
times at 0.543 precision and has spent 61% of the merge-only headroom while destroying 0.048 of
its own remaining ceiling with wrong merges. The measured target is the **rank-1/rank-2 join**
(§3.3) at a precision the current appearance model cannot reach; `GTA_LINK_STAGE1.md` §2 measured
PRTreID within-identity p90 0.093 against cross-identity p10 0.040-0.083, i.e. overlapping
distributions. That is the case for a learned association model rather than another threshold.

## 6. Negatives, limits, things deliberately not done

1. **The ceiling is an oracle, not an achievable target.** `assa_perfect_link` lets the oracle split
   contaminated tracks as well as merge fragments. `assa_merge_only` (this session's addition) is
   the honest connector-only bound, and it is **half** the headline number (+11.9 vs +24.8). Any
   claim built on the +24.8 figure must say "a tracker replacement", not "a connector".
2. **The proxy is a greedy stand-in for `trackeval`.** Per-frame greedy one-to-one matching at a
   hard 5 m gate against HOTA's Hungarian assignment over a Gaussian similarity averaged over
   alphas. Its Spearman on this stack is **+0.842**, not N2's +0.971. Levels should be read as
   +-3 AssA points here, not +-2, and certainly not to the decimal.
3. **The scale factor is extrapolated.** 1.051 is fitted at proxy ~0.63 and applied at 0.86. A
   linear (rather than multiplicative) fit of the same 20 points, `measured = 0.1446 + 0.8208 *
   proxy` (R2 0.747), gives a ceiling of 85.4 and a headroom of **+19.36** instead of +24.79. Both
   mappings, and the unscaled +23.59, are above +12; the verdict is invariant, the *level* is not.
4. **valid-58 was not measured on the current detector** (§4). It needs the v6det extraction for the
   38 held-out sequences, ~4.4 GPU-hours, and this session was pre-declared CPU-only.
5. **No tracker was run, no parameter was changed, no arm was re-scored except E** (and that
   reproduced the on-record control to 14 decimals). This is a read, not an experiment.
6. **`assa_perfect_det` mixes coverage and contamination.** It removes both false negatives and
   false positives at once, so the +8.51 "coverage" figure is an upper bound on the coverage half.
   The separate direct measurements are `matched_share` 0.8672 (13.3% of GT rows matched by
   nothing) and `pred_precision` 0.9678.
7. **The GTA connector's tau was not re-swept.** `GSR_EIOU.md` negative 4 still stands: tau 0.450
   was frozen on a different partition. A tau sweep is part of the EIoU-retune path this verdict
   declines, and if V4 step 2 is ever budget-forced back to that path, it is where it should start.

## 7. Reproduce

```
python -m tools.gsr_assoc_diag --demo                 # self-check (incl. the merge-only oracle)
python -m tools.gsr_assoc_diag --v7                   # sections 1-3, 5 (~28 min CPU)
python -m tools.gsr_assoc_diag --coverage --positions positions_gate_v6det   # section 3.1 alone
```

Section 4 and the §3.2 purity rows come from a scratch driver over the same public functions
(`partition_report`, `pooled_rank_table`, `tools.gsr_eiou.partition_purity`); its output is
`results/gsr_benchmark/gsr_assoc_diag_v7_extra.json`.

## 8. Files

- `tools/gsr_assoc_diag.py` — `assa_merge_only`, `rank_profile`, `pooled_rank_table`,
  `submission_positions`, `partition_report`, `v7_step1`, `--v7`/`--positions`/`--seqs`/
  `--pred-root`. `--diag`, `--coverage`, `--merge` and every existing function signature are
  default-preserving; three new asserts in `--demo`.
- `results/gsr_benchmark/gsr_assoc_diag_v7.json` — per-sequence rows for all five partitions, the
  fragment profiles, the coverage split, the measured scores, the proxy fit and the verdict.
- `results/gsr_benchmark/gsr_assoc_diag_v7_extra.json` — the old-stack anchor, the valid-58
  representativeness check and the purity audit.
- Claim: `v7-v4-001`.
