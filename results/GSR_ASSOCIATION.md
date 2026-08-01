# N2 — where GS-AssA dies, and the one lever that was worth pulling

Campaign act N2. `results/GSR_TEST_FLAGPLANT.md` §5 found our **GS-AssA 48.21 is the worst number
on the codabench 4365 leaderboard** while our GS-DetA (26.74) is 9th of 16. This is the diagnosis of
where the association is lost, the audit of what our GSR extraction actually ran (as opposed to what
the FOOTPASS work recommended), and one validated repair.

**Headline: association is not lost to the tracker's sampling rate, to crowding or to camera motion.
It is lost to (1) tracklet linking — an oracle linker on our own current detections would score raw
AssA ~68 against the 39.8 we measure and the 47.9 the GTA connector reaches, so the connector
realises 24% of the available headroom — and (2) a previously unmeasured calibration dropout that
silently discards 14.4% of all ground-truth player rows we had already detected. The lever taken is
the calibration repair, because it is CPU-only, needs no re-extraction, and on the frozen DEV-20
partition it helps 20 of 20 sequences.**

Development was on the valid split only. **The test split was not scored for this document**; the
one test-side number quoted (§6) is a property of our own cached predictions and reads no label.

---

## 1. The config audit — what our GSR runs actually did

Read from the code and the frozen recipe, not assumed
(`eval/gsr_score.py:extract_sequence`, `generator/extract.py`, `generator/tracking.py`,
`results/gsr_flagplant_frozen.json`).

| knob | GSR valid + test runs | FOOTPASS finding | transfers? |
|---|---|---|---|
| `sample_every` (stride) | **1** — every one of the 750 frames | stride 5 -> 2 is the big win | **NO. We already run at stride 1.** |
| tracker | `supervision.ByteTrack` | BoT-SORT wins at stride 2 | untested on GSR, see §5 |
| `minimum_consecutive_frames` | **3** (`build_tracker` default; never overridden) | `=1` buys +0.259 recall **at stride 5** | at stride 1 FOOTPASS measured +0.0135 recall and *worse* fragmentation (1.16 -> 1.23) |
| `minimum_matching_threshold` | **0.8** (supervision default; never set) | `=0.9` helps at stride 5 | same — a 200 ms-gap fix, and our gap is 40 ms |
| `lost_track_buffer` | 60 frames, with `frame_rate=30` left at default on 25 fps footage -> `max_time_lost` = 60 frames = **2.4 s** | measured null (<= 0.0005) | no |
| `track_activation_threshold` | 0.25 (default) | measured null | no |
| clip handling | **whole clip, no chunking** — `img1/%06d.jpg` read as one 750-frame sequence | — | — |
| GTA connector | within-clip, splitter OFF, **tau = 0.040** (the flag-plant freeze; the GS-HOTA-optimal on the jersey-null arm was 0.060) | — | — |
| calibration | PnLCalib every frame (`calib_period = 1`), gate `MAX_REPROJ_ERROR_M = 2.0` | — | — |

**The first finding of this act is a negative on the task's premise.** The stride/BoT-SORT result in
`results/FOOTPASS_E2E_PLAN.md` Appendix A is a fix for a **200 ms sampling gap**. GSR has no such
gap: `extract_sequence` hard-codes `sample_every=1` precisely so every GT frame gets a prediction.
Every ByteTrack knob FOOTPASS found worth turning was found worth turning *at stride 5*, and at
stride 1 the same table shows them buying +0.0135 recall for +6% fragmentation. **There is no stride
lever on GSR and the free-knob lever does not survive the regime change.**

## 2. The measurement instrument, and its validation

`tools/gsr_assoc_diag.py`. HOTA scores a matched (predicted track `i`, GT track `g`) pair at
`A(i,g) = n_ig / (N_g + M_i - n_ig)`, and `AssA` is the `n_ig`-weighted mean of `A`. Every term is
computable from a greedy nearest-GT assignment at the evaluator's own 5 m tolerance and centre
shift, so the same table yields counterfactuals the scorer cannot give:

* `assa_hat` — the proxy for the measured `loc_assoc` GS-AssA;
* `assa_perfect_link` — every fragment of one GT identity merged into a single predicted track,
  detections untouched (**the ceiling any linker/connector could reach**);
* `assa_perfect_det` — fragmentation untouched, every missed GT row found and every spurious
  predicted row removed.

**Validation: `assa_hat` vs the official scorer's per-sequence `loc_assoc` GS-AssA over all 58 valid
sequences: Spearman +0.971**, per-sequence means 0.3451 (proxy) against 0.3736 (measured), a
constant scale factor of 1.083. The proxy is a ranking- and level-faithful stand-in; the
counterfactuals below are reported on the proxy scale and, where scaled to the measured scale, the
1.083 factor is stated.

## 3. Where association dies — ranked

All numbers: SoccerNet-GSR **valid**, 58 sequences, cached positions parquets, CPU.
Raw rows: `results/gsr_benchmark/gsr_assoc_diag.json`, `..._merged.json`.

| counterfactual | AssA (proxy) | delta | scaled to the measured scale |
|---|---|---|---|
| as shipped (raw ByteTrack, no connector) | **0.3451** | — | 37.4 (measured 39.8 pooled) |
| + perfect linking (oracle connector, same detections) | **0.6272** | **+0.282** | **~67.9** |
| + perfect detection (no FN, no FP; fragmentation kept) | 0.4944 | +0.149 | ~53.5 |
| both | 1.0 | — | — |

### 3.1 Cause 1 — tracklet linking. Unrealised headroom: ~20 AssA points.

An oracle that merged our existing fragments correctly would take raw AssA from 39.8 to **~67.9** —
inside the leaderboard's 62-82 band — **without one extra detection**. The GTA connector at the
frozen `tau = 0.040` reaches 47.9, i.e. it realises **24.5%** of that headroom.

The fragment profile says why the remaining 75% is hard. Mean share of a GT identity held by its
k-th largest predicted fragment, pooled over 1,213 GT identities:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | >8 | unmatched |
|---|---|---|---|---|---|---|---|---|---|---|
| share | **0.453** | **0.146** | 0.053 | 0.022 | 0.010 | 0.005 | 0.003 | 0.002 | 0.004 | **0.302** |

* 5.75 fragments per GT identity, but only **2.40 of them carry >= 5%** of it. The tail is not the
  problem: tracks shorter than 25 frames are 30.6% of all tracks and **2.3% of all rows**.
* The prize is concentrated in one merge per player: joining rank-1 to rank-2 would move
  `sum p_i^2` from 0.29 to ~0.43 on its own.
* `GTA_LINK_STAGE1.md` §2 already measured why that merge is hard: PRTreID within-identity cosine
  distance p90 = 0.093 against cross-identity p10 = 0.040-0.083. **The distributions overlap**, and
  the connector's merge precision is already only 80.1% at tau 0.040 (69.7% at 0.060). Appearance is
  the binding constraint on linking, and `GTA_LINK_STAGE1.md` Addendum A3 measured the same
  embedding saturating at 0.62 LOTO top-1.

### 3.2 Cause 2 — coverage, and the calibration dropout inside it (NEW)

30.2% of GT player rows are matched by nothing. Splitting that by matching the same GT boxes in
**image** space (`image_x/image_y` vs `bbox_image`) instead of pitch space:

| | pooled over 58 valid sequences, 651k GT rows |
|---|---|
| GT rows covered by one of our tracks **in image space** | **0.8427** |
| GT rows covered **in pitch space** (what the evaluator sees) | 0.7453 |
| **detected in image space, then lost to the calibration** | **0.1443** |
| *the same thing as a net recall difference (a lower bound)* | *0.0974* |
| frames with *zero* finite pitch positions | **0.2498** |
| our own person rows carrying no pitch coordinate | **0.1979** |

(The two loss figures differ because the image match is a foot-point-within-half-a-box-width proxy,
not an IoU match: 4.7% of GT rows are matched in pitch space but not in image space. 0.1443 is the
direct count of "image hit AND pitch miss"; 0.0974 is the net difference of the two recalls. Both
are reported; the honest reading is that the projection loss is between 10 and 14 points.)

**Between a third and a half of all coverage loss is a projection failure on players we had already
detected and tracked** (9.7-14.4 points of the 25.5-30.6 point coverage loss, depending on which of
the two figures above you take). Eight of the 58 valid sequences emit pitch positions on fewer than
half their frames; the worst, SNGS-034, on 13.6%.

The mechanism is not a rejected calibration. On SNGS-034's 645 dead frames the calibrator's own
keypoint reprojection error is **0.365 m median — better than on its 102 live frames (0.434 m)** —
so `MAX_REPROJ_ERROR_M = 2.0` passes them. The homography is self-consistent on the pitch keypoints
and still puts **every** player more than 2 m outside the touchline, where
`generator.postprocess.clamp_to_pitch` NaNs them and `reject_implausible_frames` then voids what is
left. It is a silent, gate-passing failure.

### 3.3 What is NOT the cause (measured nulls)

| candidate | Spearman vs per-sequence measured `loc_assoc` GS-AssA |
|---|---|
| **crowding** (GT people per frame) | **+0.004** |
| **camera motion** (median per-frame GT box displacement, px) | **-0.138** |
| fragments per GT identity (raw count) | -0.530 |
| dominant-fragment share | +0.924 |
| Herfindahl `sum p_i^2` | +0.915 |
| rows with no pitch coordinate | -0.630 |
| detected-in-image-but-lost-to-calibration | **-0.663** |

Crowding and camera motion carry nothing. The raw *count* of fragments is a weak predictor; the
*share structure* is a strong one — which is the same statement as "the tail of short tracks does
not matter". And of the directly-measurable defects, the calibration loss is the strongest single
correlate of a bad sequence.

Sequences binned by calibration health (quartiles of the fraction of frames with any pitch output):

| quartile | calib frame rate | measured AssA (raw) | measured AssA (+GTA) | `assa_perfect_link` | `assa_perfect_det` | frags/GT |
|---|---|---|---|---|---|---|
| Q1 worst | 0.452 | 27.6 | 32.6 | 0.379 | 0.495 | 5.40 |
| Q2 | 0.732 | 40.5 | 47.8 | 0.626 | 0.525 | 4.78 |
| Q3 | 0.863 | 34.3 | 41.6 | 0.686 | 0.426 | 7.17 |
| Q4 best | 0.960 | 47.1 | 57.3 | 0.821 | 0.529 | 5.79 |

`assa_perfect_det` is **flat** across the quartiles (0.43-0.53) while `assa_perfect_link` climbs
0.379 -> 0.821. Read: fragmentation is a uniform property of the tracker; coverage is what separates
a good clip from a bad one, and on the broken clips it caps even a perfect linker at 0.38.

## 4. The lever taken — calibration gap repair

**Why this one and not the linker.** The linking headroom is larger (+20 AssA against the coverage
item's ~+14 on the same scale) but every route to it was already measured shut: appearance is
saturated (§3.1), the splitter half of GTA is a measured negative on this data, and a tracker swap
is a 6-GPU-hour re-extraction resting on an out-of-regime prior (§5). The calibration repair is
**CPU-only, re-uses every cached artifact, and its failure mode is bounded** (a wrong recovered
point is one FP row, not a corrupted track).

`generator.postprocess.fill_calibration_gaps` (+ `tools/gsr_calibfill.py` as the driver):

1. Any frame carrying >= 8 already-projected rows lets its exact image->pitch homography be
   re-derived by DLT from the positions table's own `(image_xy, pitch_xy)` pairs. Verified: the
   self-fit residual on its own frame is **0.035-0.131 m** across sequences.
2. A dead frame takes the linear interpolation of the two bracketing live frames' homographies
   (nearest carried at the clip ends), refusing donors further than `max_gap` frames away.
3. Its image points are re-projected and pass through the same `clamp_to_pitch`, so an off-pitch
   recovery is still rejected.

Nothing is re-extracted; the ground truth is used only to score the result. Track ids, jersey votes
and PRTreID embeddings are keyed by `(track_id, frame)` and are untouched, so the whole downstream
chain replays on CPU.

### 4.1 Choosing `max_gap` on the declared DEV-20

Partition: `eval.gsr_identity.split_sequences` — DEV = every third valid sequence (20), TEST-38 =
the rest. DEV-20's calibration health (0.743) matches the full split's (0.750), so it is not a soft
subset. GT audit of the recovered rows only:

| `max_gap` | rows recovered | precision of recovered rows (within 5 m of a GT player) | GT recall 0.785 -> |
|---|---|---|---|
| 25 | 17,854 | **0.891** | 0.865 |
| 75 | 23,976 | 0.755 | 0.878 |
| unlimited | 27,110 | 0.657 | 0.878 |

Scored on DEV-20 (base arm, no connector): `loc_assoc` GS-HOTA 49.02 -> **50.77** at gap 25,
50.14 at 75, 49.63 at unlimited. Unbounded carry-forward gives back most of the win; the frozen
pick is the conservative end.

**Frozen: `max_gap = 10` frames (0.4 s).** On the arm we actually ship (with the GTA connector) it
wins DEV on both HOTA and AssA; gap 25 is statistically indistinguishable there (55.26 vs 55.31) and
is recorded as the alternative.

### 4.2 DEV-20 result at the frozen setting

Arms rebuilt identically on both sides (positions -> submission -> cached Koshkina jersey votes ->
GTA connector tau 0.040, splitter OFF), so the repair is the only variable.

| arm | `loc_assoc` H / DetA / AssA | `role_only` H | `no_jersey` H | `gs_hota_full` H / DetA / AssA |
|---|---|---|---|---|
| baseline base | 49.02 / 61.56 / 39.06 | 44.68 | 42.72 | 16.32 / 6.76 / 39.40 |
| baseline + GTA tau 0.04 | 53.31 / 61.51 / 46.23 | 48.85 | 46.82 | 23.82 / 11.39 / 49.79 |
| fill base | 50.73 / 64.58 / 39.89 | 46.20 | 44.16 | 16.53 / 6.84 / 39.93 |
| **fill + GTA tau 0.04** | **55.31 / 64.54 / 47.43** | **50.64** | **48.51** | **24.46 / 11.73 / 51.03** |

Paired per sequence against `baseline + GTA` (n = 20, Wilcoxon):

| config | mean delta | median | helped | hurt | worst | best | p |
|---|---|---|---|---|---|---|---|
| `loc_assoc` | **+2.67** | +2.06 | **20** | 0 | +0.16 | +7.80 | 1.9e-6 |
| `role_only` | +2.41 | +1.62 | 20 | 0 | +0.16 | +7.26 | 1.9e-6 |
| `no_jersey` | +2.28 | +1.52 | 20 | 0 | +0.17 | +7.09 | 1.9e-6 |
| `gs_hota_full` | **+1.02** | +0.53 | 18 | 2 | -0.03 | +4.58 | 9.5e-6 |

No column is traded away: DetA +3.03, AssA +1.20 and HOTA +2.00 all move together on `loc_assoc`.

### 4.3 Full valid split (58 sequences), frozen `max_gap = 10` — one run

46,767 of 642,086 rows recovered (7.3%). `results/gsr_benchmark/gsr_calibfill_valid58.json`.

| arm | `loc_assoc` H / DetA / AssA | `role_only` H | `no_jersey` H | `gs_hota_full` H / DetA / AssA / IDF1 |
|---|---|---|---|---|
| baseline base | 48.91 / 60.18 / 39.79 | 44.95 | 43.06 | 14.76 / 5.98 / 36.43 / 8.07 |
| baseline + GTA tau 0.04 | 53.60 / 60.09 / 47.85 | 49.51 | 47.48 | **23.53** / 11.34 / 48.82 / 18.47 |
| fill base | 51.09 / 63.78 / 40.97 | 46.89 | 44.91 | 14.96 / 6.06 / 36.93 / 8.10 |
| **fill + GTA tau 0.04** | **55.99 / 63.66 / 49.29** | **51.64** | **49.52** | **24.35** / 11.77 / **50.38** / 19.07 |

**Control: the `baseline + GTA` row reproduces `GTA_LINK_STAGE1.md` §4 exactly** — 23.53 / 11.34 /
48.82 — so the two sides are the same harness and the repair is the only variable.

Deltas against `baseline + GTA`: **`loc_assoc` +2.39 HOTA, +3.57 DetA, +1.44 AssA; `gs_hota_full`
+0.82 HOTA (23.53 -> 24.35), +0.43 DetA, +1.56 AssA, +0.60 IDF1.** No column is traded away.

Paired per sequence (n = 58, Wilcoxon):

| config | mean delta | median | helped | hurt | worst | best | p |
|---|---|---|---|---|---|---|---|
| `loc_assoc` | **+3.16** | +2.30 | **56** | 2 | -0.67 | +12.17 | 6.2e-11 |
| `role_only` | +2.85 | +1.96 | 56 | 2 | -0.87 | +10.63 | 7.3e-11 |
| `no_jersey` | +2.73 | +1.73 | 56 | 2 | -0.83 | +10.21 | 7.6e-11 |
| `gs_hota_full` | **+1.18** | +0.55 | **53** | 5 | -2.29 | +8.27 | 8.5e-10 |

**The gain lands where the diagnosis said it would.** Spearman between a sequence's gain and its
calibration health is **-0.683**: the 8 sequences emitting pitch positions on under half their frames
gain **+6.11** `loc_assoc` HOTA on average, the 18 healthy ones (>0.90) gain **+1.33**. Best:
SNGS-056 +12.17 (`gs_hota_full` 21.79 -> 30.06), SNGS-038 +9.82, SNGS-026 +9.29. Worst:
SNGS-082 -0.67 (`gs_hota_full` -2.29) and SNGS-035 -0.04 — both already near-perfectly calibrated
(0.943 / 0.791), so the repair only added marginal rows there. This dose-response is the strongest
evidence that the mechanism identified in §3.2 is the one being repaired.

## 5. Negatives and things deliberately not done

1. **The stride lever does not exist on GSR** (§1). The task's premise that
   `stride-2 + BoT-SORT` transfers is half wrong: the stride half is already maxed, and the free
   ByteTrack knobs were validated in a regime (200 ms gaps) that GSR does not have.
2. **BoT-SORT was not run on GSR, and the prior points the wrong way.** FOOTPASS measured its win
   (frags 1.08 vs 1.29, contamination 0.037 vs 0.093) at stride 2 = 12.5 Hz. GSR runs at 25 Hz,
   where global motion compensation has strictly less to do, and `generator/tracking.py`'s own
   docstring records BoT-SORT fragmenting **more** than ByteTrack at 25 fps on our broadcast. A GSR
   trial costs a full re-extraction (~390 s/sequence, ~6.3 GPU-hours for the valid split) and is
   *not* recommended before the appearance model changes.
3. **The connector tau was not re-opened.** tau 0.060 is worth +1.26 GS-AssA over 0.040 on the
   jersey-null arm but drops merge precision 80.1% -> 69.7%, and `GTA_LINK_STAGE1.md` Addendum A4
   already concluded that any solver consuming merged tracklets must use 0.040. Changing it is an
   identity-chain decision, not an association one.
4. **The identity-solver arm was not re-run, so the 33.20 headline number is unmeasured here.** The
   numbers above stop at the GTA connector arm. `eval.gsr_identity.build_bundle` hard-codes
   `out_dir / "positions"`, so pointing the solver at the repaired parquets needs that one argument
   parameterised (CPU-cheap after that — the flag-plant log has the whole bundle+solve+score stage
   at 2 m 52 s for 49 sequences). It was left alone deliberately: it is a frozen-recipe module and
   re-running the solver is the same decision as spending a submission slot.
5. **The repair adds false positives by construction.** At `max_gap = 25` the recovered rows are
   89.1% within 5 m of a GT player; the other 10.9% are new FPs. At `max_gap = 10` the count is
   smaller and the precision higher, but it is never 1.0. This is why every arm is reported with
   DetA alongside AssA.
6. **The root cause in the calibrator was not fixed, only worked around.** PnLCalib is still
   emitting gate-passing homographies that project every player off the pitch. A real fix belongs in
   `generator/calibrate.py` (e.g. an on-pitch-players plausibility term inside the gate, of which
   `postprocess.MIN_ONPITCH_PLAYERS` is already the shape) and would need a GPU re-run to validate.
   Until then the repair is a post-hoc patch that also works on already-cached runs.
7. **The linking ceiling was not re-measured on the repaired positions.** The ~67.9 of §3.1 is
   conditional on the *unrepaired* coverage, and §3.3's quartile table shows `assa_perfect_link`
   climbing 0.379 -> 0.821 with calibration health — so after the repair the linking headroom is
   larger, not smaller, and the ranking in §3 does not change. Re-running `--diag` on
   `positions_filled` would put a number on it (~12 min CPU) and was not done.
8. **`assa_hat` is a greedy proxy, not `trackeval`.** It uses per-frame greedy one-to-one matching
   at a hard 5 m gate where HOTA runs a Hungarian assignment over a Gaussian similarity and averages
   over alphas. Its +0.971 Spearman and constant 1.083 scale are what justify the counterfactuals;
   the counterfactual *levels* should be read as +-2 AssA points, not to the decimal.

## 6. Projected test impact (no test label was read)

The calibration dropout is a property of our own cached predictions, so it is measurable on the test
positions without opening a single `Labels-GameState.json`:

| | valid (58) | test (49) |
|---|---|---|
| person rows with no pitch coordinate | 0.1979 | **0.2997** |
| frames with any pitch output | 0.7540 | **0.6291** |
| sequences below 0.50 frame coverage | 8 / 58 | **14 / 49** |
| rows recoverable at `max_gap = 10` | 7.9% | **10.5%** |

**The defect is ~1.5x worse on the test split than on valid**, so the repair should pay at least as
much there. It is *not* a reason to burn a submission slot on its own — that decision belongs with
the orchestrator and the 1/day cadence.

## 7. Reproduce

```
python -m tools.gsr_assoc_diag --diag        # §2, §3.1, §3.3  (~12 min CPU)
python -m tools.gsr_assoc_diag --coverage    # §3.2            (~15 min CPU)
python -m tools.gsr_assoc_diag --merge       # the correlation panel + quartiles (seconds)
python -m tools.gsr_calibfill --recover-audit --gaps 25,75,0   # §4.1 GT audit
python -m tools.gsr_calibfill --dev --gaps 10                  # §4.2 (~11 min CPU)
python -m tools.gsr_calibfill --run --max-gap 10               # §4.3 (~35 min CPU)
```

Self-checks: `python -m tools.gsr_assoc_diag --demo`, `python -m tools.gsr_calibfill --demo`,
`pytest tests/test_postprocess.py`.

## 8. Files

- `tools/gsr_assoc_diag.py` — the AssA decomposition and its counterfactuals (`--diag`, `--demo`).
- `tools/gsr_calibfill.py` — the repair driver (`--recover-audit`, `--dev`, `--run`, `--demo`).
- `generator/postprocess.py` — `fill_calibration_gaps` (+ `_frame_homographies`,
  `_donor_homography`); covered by two new cases in `tests/test_postprocess.py`.
- `results/gsr_benchmark/gsr_assoc_diag.json`, `gsr_assoc_diag_merged.json`,
  `gsr_coverage_split.json` — the diagnosis rows.
- `results/gsr_benchmark/gsr_calibfill_dev20.json`, `gsr_calibfill_valid58.json` — the arms.
- `outputs/gsr/positions_filled/`, `outputs/gsr/eval_calibfill_{base,koshkina,gta}/` — artifacts.
  The original `outputs/gsr/positions/` was never overwritten.
