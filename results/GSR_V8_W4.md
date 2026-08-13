# v8 session W4 — per-sequence BatchNorm test-time adaptation

Campaign v8, session W4. The cheapest domain-adaptation lever that exists: recompute every
BatchNorm layer's `running_mean` / `running_var` on the **test sequence's own unlabeled frames**
before inference. No labels, no gradients, no loss, no weight changes — one forward pass over frames
we are about to process anyway. Literature: Schneider et al. 2020, *Improving robustness against
common corruptions by covariate shift adaptation* (the "test-time BN" line); nothing was fetched for
this session.

**CLAIMS-HYGIENE.** Nothing of the benchmark winner's repo, weights or outputs is used, run, read or
needed here. Every component is ours or an already-installed dependency.

---

## 0. SCOPE CHECK (measured before registration; inventory only, no arm, no score)

### 0.1 What actually contains BatchNorm

| module | architecture | BN layers with running stats | adaptable channels | in scope |
|---|---|---|---|---|
| **detector** — YOLOv8s S4b fine-tune (`models/gsr_det/gsr_v3_ft_b_last.pt`, 11.1 M params) | conv net | **57** `BatchNorm2d` | **10,016** | **yes** |
| **embedder trunk** — CLIP ViT-B/16 visual tower (`tools/clip_embedder.py`) | transformer | 0 (LayerNorm only — no running statistics exist) | 0 | no |
| **embedder head** — the trained projection neck | `Linear(768->256) -> BatchNorm1d(256) -> L2` | **1** `BatchNorm1d` | **256** | **yes** |

**The task's scope premise is half wrong, and the half that is wrong is worth stating.** The brief
said: *"if it is a ViT/CLIP architecture it is LayerNorm-based, which has NO running stats, so
BN-stats does not apply. If so: scope = detector only."* The trunk is indeed LayerNorm-only, but the
checkpoint `outputs/gsr/clip_ckpt/epoch8.pt` carries `bn.weight`, `bn.bias`, `bn.running_mean`,
`bn.running_var`, `bn.num_batches_tracked = 10000` — a **BNNeck** (the standard re-ID head) sitting
between the projection and the L2 normalisation, i.e. the last operation before every appearance
similarity this pipeline computes. It has running statistics, they were estimated on the training
corpus, and they are adaptable. So the scope is **detector (primary) + embedder BNNeck (secondary)**,
not detector only. No Tent / entropy minimisation / gradient step is used anywhere in this session.

### 0.2 The fusion fact that makes or breaks the detector arm

Ultralytics folds every `Conv+BN` pair into a single convolution the first time a model predicts
(`AutoBackend` calls `model.fuse()`, **in place**). Measured: 57 BN modules before the first
`predict`, **0** after it — on the same object. Consequences, both handled in
`generator/bn_adapt.py::adapt_yolo`:

1. adaptation must run **before** the detector's first prediction, otherwise there is nothing left
   to adapt;
2. fusion is suppressed for the duration of the adaptation pass and the predictor is then dropped,
   so the next real prediction rebuilds it and **fuses the adapted statistics into the conv weights**
   — inference therefore runs the ordinary fused path, not a slower or numerically different one.

The adaptation pass goes through the detector's normal `__call__`, so it sees byte-identical
preprocessing (letterbox, scaling, channel order) to inference.

### 0.3 The precedent, stated accurately

`STATUS.md` and the brief both cite "+0.18 from BN-stats on the detector". The record does not say
that. `results/CLUSTER_SESSION1.md` §8 measured **+0.18 valid identity mAP** (56.93 -> 57.11) on the
**prtreid re-ID backbone**, as an accidental by-product of frozen-backbone training in train mode.
It was never measured on the detector and never in GS-HOTA. This session's prior is therefore weaker
than the brief implies: an mAP gain on a re-ID model is not evidence about a detector, and neither is
evidence about an end-to-end GS-HOTA chain with an OCR reader and a solver downstream.

### 0.4 The coupling that sets the price

`eval/gsr_jersey.py`, `tools/gsr_eiou.build_box_cache` and the embedding stage all **re-detect** each
frame and match the detections back to the persisted foot points ("MUST be the same football detector
the parquet was extracted with, so boxes reproduce exactly" — `eval/gsr_jersey.py`). A detector whose
statistics changed therefore cannot be swapped into extraction alone: **every GPU stage must build
the same per-sequence-adapted detector**, or the arm measures a box-matching artefact instead of
BN adaptation. This is why the stage is plumbed as a process-wide `generator.extract.BN_STATS`
rebinding (the mechanism `FOOTBALL_WEIGHTS` already uses) rather than a kwarg on one function.

---

## 1. REGISTRATION (written 2026-08-13, before any arm was built or scored)

### 1.1 Probe and control

Probe = the W2/W2b 10-sequence DEV probe, unchanged: **SNGS-{024, 027, 039, 042, 045, 048, 051, 054,
057, 078}** (the 10 hardest DEV-20 sequences by the rule registered in `GSR_V8_W2.md` §1.1; probe
mean control GS-HOTA 42.95 against DEV-20's 51.81).

Control = `results/gsr_benchmark/gsr_v7_control_dev.json` (the pinned-stack v6/v7 control; per
sequence 47.3428 / 50.1301 / 35.8442 / 36.6848 / 49.1977 / 48.4916 / 45.7777 / 39.5077 / 34.4992 /
46.0673). W2b re-derived it on this machine at **0.0 delta per sequence**, so pairing against the
file is valid. Same machine (this laptop), same weights, same seeds, same config apart from the BN
stage.

**One extra control obligation this session carries and W2b did not.** W2b changed only pitch
coordinates, so it could reuse the on-record detections. This session re-runs **extraction**, so the
on-record artifacts must be reproducible on today's stack for the pairing to be legitimate. A
flag-OFF re-extraction of one probe sequence is diffed against the on-record
`outputs/gsr/positions_v6det/SNGS-024.parquet` **before any arm is scored**; the result is reported
in §2.1 whatever it is. If it does not reproduce, the arm gets its own same-stack control
re-extraction and the extra cost is paid rather than the confound accepted.

### 1.2 THE REGISTERED ARM — `w4bn` (one arm, the verdict rests on it alone)

**Full cumulative recomputation of the detector's BatchNorm running statistics on the test
sequence's own frames.** Precisely:

* all 57 `BatchNorm2d` layers `reset_running_stats()` (training-corpus statistics discarded), then
  `momentum=None` (PyTorch's cumulative moving average — the result is the plain mean/variance over
  the frames seen, independent of frame order);
* those layers alone in train mode; every weight frozen; `torch.no_grad()`; no loss is formed;
* input = **every 5th frame** of the sequence (**stride 5 fixed a priori** — 150 of 750 frames;
  measured cost 3.7 s/sequence, so the stride is not a compute compromise either), each frame passed
  through the detector's ordinary call so preprocessing matches inference exactly;
* **statistics are reset per sequence** and cached per sequence
  (`outputs/gsr/bn_stats_w4/<SEQ>.pt`); no sequence ever sees another's statistics;
* the adapted detector is then used by **every** stage that detects — extract, per-crop OCR, box
  cache, CLIP embeddings — per §0.4, and everything downstream (calibration gate, EIoU association,
  reader, connector, solver, scorer) is the frozen v6 chain, unchanged.

Inputs to the adaptation are raw frames and nothing else: no labels, no ground truth, no scores, no
detections from another arm.

### 1.3 THE ONE SECONDARY ARM — `w4neck`

The same operation on the **embedder's `BatchNorm1d(256)` BNNeck** (§0.1): full reset,
`momentum=None` cumulative recomputation over the sequence's own crops (frame stride **10**, fixed a
priori; the shipped embedding pass runs at stride 2, so the adaptation pass costs a fifth of it),
reset per sequence, on the **shipped** detections (`positions_gate_v6det`), so it is an embedder-only
swap paired against the same control. Registered because §0.1 found the layer,
because it is the cheapest arm in the session (one GPU stage), and because it tests the same
hypothesis on the module whose train->test appearance shift is most direct. **The verdict stands on
`w4bn` alone**; `w4neck` is reported beside it and cannot rescue a failed registered arm.

### 1.4 Bars (fixed now)

| gate | threshold | consequence |
|---|---|---|
| **kill bar** | registered arm's mean paired GS-HOTA on the 10-sequence probe **< +0.10** | **registered FAIL**, stop, do **not** run DEV-20 |
| escalation | probe mean **>= +0.10** | run full DEV-20 |
| **ship gate** (entry to the W5 bundle) | DEV-20 mean **>= +0.5** AND **>= 12/20** sequences helped | otherwise banked negative |

### 1.5 Guards

1. **No score-peeking.** Stride (5), momentum (`None`, cumulative), reset policy (full), layer subset
   (all BN layers) are fixed in this section and are not tuned against any delta. If a variant is
   measured after the fact it is labelled NOT REGISTERED and cannot change the verdict.
2. **No GT leakage.** The adaptation sees only raw frames of the sequence. Ground truth is read by
   the scorer only.
3. **No cross-sequence leakage.** Statistics are reset per sequence; the cache is keyed by sequence.
4. **Stage default OFF.** `generator.extract.BN_STATS is None` ships; with it None the chain is the
   shipped chain, proven by the §1.1 re-extraction diff.
5. **No `METRICS_VERSION` bump** unless the DEV-20 ship gate passes.
6. No TEST-38, no test-49, no submission, **no commit**.
7. Only the end-of-chain GS-HOTA counts. Detection counts, box-match rates and statistic-shift
   magnitudes are descriptive only and are never quoted as the result.
8. One GPU job at a time; targeted tests only while the GPU is busy.
9. If the measurement contradicts the session's premise, the contradiction is the deliverable
   (§0.1 and §0.3 are already two of those).

---

## 2. RESULTS — VERDICT: **REGISTERED FAIL**, by the largest margin this campaign has recorded.
## The registered arm is worth **-17.56 GS-HOTA**, 0 sequences helped out of 10, against a +0.10 bar.

**The decisive number: `w4bn` mean paired GS-HOTA vs the same-stack control = `-17.5645`, 0 helped /
10 hurt, worst `-31.54`, best `-6.58`, Wilcoxon `p = 0.00195`.** The kill bar was `>= +0.10`. The
stage does not enter the W5 bundle, DEV-20 was not run (the registration forbids it after a kill-bar
failure), `METRICS_VERSION` was not bumped, and the stage ships default OFF.

The pre-registered secondary arm fails identically: the embedder BNNeck costs **-13.5219**, 0 helped
/ 10 hurt, `p = 0.00195`.

This is not a near miss and not a noise result. Four things are established:

1. **Test-time BN adaptation actively destroys this detector**: per-frame it drops 5-26% of
   detections, cuts median detection confidence from ~0.76 to 0.56-0.66, and shifts the role
   posterior so that 3-6% more people are called referees (§2.4). End to end that is -23.5% of all
   positions rows and +37% track fragments.
2. **The hypothesis's premise is largely absent on this benchmark.** The stage exists to close a
   train->test appearance shift. Our detector is the **S4b fine-tune, trained on SoccerNet GSR train
   + SoccerNet-v3** — the DEV sequences are the same corpus, same stadiums, same broadcast rig.
   There is little shift to close, and replacing well-estimated corpus statistics with 150-frame
   per-sequence estimates is a pure loss.
3. **The repo's "+0.18 BN-stats" precedent does not transfer and was mis-cited** (§0.3): it was a
   re-ID mAP number on a frozen backbone, not a detector, not GS-HOTA.
4. **A separate, load-bearing finding fell out of the control obligation**: the on-record v6det DEV
   extraction is **not reproducible on today's stack**, and re-extracting today costs **-1.14
   GS-HOTA** on this probe before any arm is applied (§2.1). Any future session that re-extracts
   must build its own control; pairing a fresh extraction against `gsr_v7_control_dev.json` would
   have charged 1.14 points of stack drift to the arm.

### 2.1 Control obligations — one failed, one passed, and both matter

| check | result |
|---|---|
| **flag-OFF re-extraction vs the on-record `positions_v6det`** | **NOT identical on any of the 10 sequences.** Rows +1.3% to +2.6% (120,763 vs 118,231 total); tracks systematically higher (e.g. SNGS-039 87 vs 58, SNGS-051 76 vs 50) |
| **cost of that drift, end to end** | `w4off` vs the on-record control: **paired mean -1.1395**, 3 helped / 7 hurt, worst -4.85, `p = 0.105` |
| **flag-OFF run-to-run, same stack, same sequence, twice** | **byte-identical** — `identical=True`, `max_abs 0.0` over 7,704 rows, 48/48 tracks (SNGS-024) |
| stage OFF = shipped chain | the `w4off` lineage runs the identical code path with `BN_STATS is None`; the only difference from the shipped chain is the stack it ran on, not the code |

So the measurement noise floor of this experiment is **exactly zero** (extraction is deterministic
run to run), the stack drift is **1.14**, and the registered effect is **-17.56**. The verdict does
not depend on which control is used: against the on-record control the arm is -18.70, against the
same-stack control -17.56.

The stack today is python 3.14.0, supervision 0.30.0, ultralytics 8.4.56, torch 2.11.0+cu128, RTX
3050; the on-record DEV artifacts were built on the supervision 0.28/0.29-era stack
(`results/gsr_v6_frozen.json` provenance).

### 2.2 THE DECISIVE TABLE — paired GS-HOTA on the probe

Same machine, same weights, same seeds, same frozen v6 chain (EIoU `e=0.3, rounds=1, w_app=0.5,
app_max=0.30`, reader PARSeq v6 arm-4t, floor 0.80, `tau=0.450`). The only variable is whether each
sequence's BatchNorm statistics were recomputed on its own frames.

| sequence | control `w4off` | arm `w4bn` | delta |
|---|---|---|---|
| SNGS-024 | 48.352 | 40.362 | **-7.990** |
| SNGS-027 | 48.384 | 37.931 | **-10.453** |
| SNGS-039 | 34.147 | 2.606 | **-31.541** |
| SNGS-042 | 33.994 | 20.897 | **-13.097** |
| SNGS-045 | 48.564 | 17.099 | **-31.464** |
| SNGS-048 | 48.299 | 20.642 | **-27.657** |
| SNGS-051 | 43.861 | 18.406 | **-25.455** |
| SNGS-054 | 40.448 | 33.866 | **-6.581** |
| SNGS-057 | 29.654 | 20.390 | **-9.264** |
| SNGS-078 | 46.445 | 34.303 | **-12.142** |

| arm | mean | median | helped | hurt | worst | best | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| **`w4bn` vs `w4off` (REGISTERED)** | **-17.5645** | -12.6195 | **0** | **10** | -31.5410 | -6.5814 | **0.00195** |
| `w4bn` vs on-record control (reference) | -18.7041 | -14.9484 | 0 | 10 | -33.2382 | -5.6412 | 0.00195 |
| `w4off` vs on-record control (stack drift) | -1.1395 | -1.1656 | 3 | 7 | -4.8453 | +1.0088 | 0.105 |
| **`w4neck` vs on-record control (SECONDARY)** | **-13.5219** | -13.6372 | **0** | **10** | -18.5708 | -6.4922 | **0.00195** |

Pooled over the probe, arm vs same-stack control:

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 | tracklets in -> out |
|---|---|---|---|---|---|---|
| `w4off` (control) | **41.6728** | **29.7643** | **58.3486** | **93.4706** | **43.4031** | 740 -> 1,422 |
| `w4bn` (registered) | 25.1791 | 14.4075 | 44.0168 | 91.8425 | 24.7889 | 1,019 -> 1,437 |
| `w4neck` (secondary, on the SHIPPED lineage) | 29.6221 | 23.6299 | 37.1446 | 93.3311 | 27.6183 | 584 -> **11,977** |

**GS-DetA more than halves (29.76 -> 14.41).** The damage is upstream of everything the campaign has
been tuning: it is in the detections themselves, so no downstream threshold could recover it.

### 2.3 The secondary arm fails too, but for a different and partly instrumental reason

The BNNeck statistics move enormously when recomputed per sequence — relative L2 shift of
`running_mean` **0.87 to 0.97** across the ten sequences (the training corpus spans thousands of
identities and dozens of matches; one sequence's crops are two kits in one stadium). Under the frozen
downstream point that is fatal in a specific, mechanical way: the EIoU appearance gate is
`app_max = 0.30` on the cosine distance of the L2-normalised embedding, and re-whitening every
feature moves every distance. The re-association shattered the partition — **584 tracklets in,
11,977 out**, against 1,422 for the control — i.e. the gate stopped matching almost anything.

**Honest limit on this arm:** part of that -13.52 is threshold coupling, not representation quality.
A fair test of BNNeck adaptation would re-calibrate `app_max` (and `tau`) on the new embedding
geometry. The registration forbids tuning anything against the probe, so that was not done and is
**not** claimed either way. What is established is narrower and still decisive for the campaign: **at
the frozen v6 operating point, per-sequence BNNeck adaptation cannot be dropped in.**

### 2.4 Mechanism — measured at the detector, not inferred

50 frames per sequence, the shipped detector built twice (original statistics vs that sequence's
adapted statistics), same thresholds, no tracking, no scoring:

| sequence | persons/frame | median detection confidence | share called referee |
|---|---|---|---|
| SNGS-024 | 12.38 -> 11.80 (**-4.7%**) | 0.805 -> 0.741 | — |
| SNGS-039 | 20.72 -> 18.68 (**-9.8%**) | 0.759 -> **0.562** | 0.157 -> 0.178 |
| SNGS-045 | 17.88 -> 13.28 (**-25.7%**) | 0.764 -> 0.659 | 0.141 -> **0.197** |
| SNGS-057 | 20.16 -> 17.78 (**-11.8%**) | 0.758 -> **0.583** | 0.150 -> **0.213** |

Three failures compound:

1. **Recall.** Fewer people are detected at all.
2. **Confidence collapse.** The median confidence falls by 0.06-0.20. Every threshold below the
   detector is frozen against the original distribution — `DETECT_CONF = 0.20`, ByteTrack's
   high/low split, the EIoU gate — so a uniform confidence drop is read downstream as "weak
   evidence" and tracks fragment: **1,029 raw tracks against the control's 750 (+37%)** on 23.5%
   fewer rows.
3. **Role corruption.** The referee share rises by 2-6 points. Under GS-HOTA a role error is a
   non-match, so a player relabelled referee is a double error (a miss plus a false positive).

Per-sequence positions written by the two lineages:

| sequence | rows control | rows arm | delta | tracks control | tracks arm |
|---|---|---|---|---|---|
| SNGS-024 | 7,704 | 6,903 | -10.4% | 48 | 68 |
| SNGS-027 | 12,185 | 11,190 | -8.2% | 77 | 109 |
| SNGS-039 | 14,258 | 11,644 | -18.3% | 87 | 110 |
| SNGS-042 | 14,219 | 9,580 | -32.6% | 89 | 111 |
| SNGS-045 | 11,732 | 7,988 | -31.9% | 87 | 134 |
| SNGS-048 | 14,992 | 8,725 | **-41.8%** | 70 | 114 |
| SNGS-051 | 11,298 | 8,415 | -25.5% | 76 | 91 |
| SNGS-054 | 11,088 | 8,876 | -19.9% | 66 | 85 |
| SNGS-057 | 13,802 | 10,642 | -22.9% | 93 | 125 |
| SNGS-078 | 9,485 | 8,470 | -10.7% | 57 | 82 |
| **total** | **120,763** | **92,433** | **-23.5%** | **750** | **1,029** |

**Why this is the expected result once stated properly, and why it was still worth measuring.** A
YOLO detector is trained with mosaic composition, HSV jitter, random scaling and flips. Its BatchNorm
running statistics are therefore the statistics of *augmented, mosaicked batches*, and every
convolution weight after them was fitted against those statistics. A clean, letterboxed broadcast
frame is a different distribution — not because the test domain shifted, but because the training
*input pipeline* is not the inference input pipeline. Recomputing the buffers on clean frames does
not "fix a domain shift"; it moves the network off the operating point its weights were fitted at.
The magnitude was not predictable a priori (the statistics only move a median 3.9% in mean and 19% in
variance, §2.5), which is exactly why the arm was registered and run rather than argued about.

### 2.5 How far the statistics actually moved

Per-sequence relative L2 shift of the detector's BN buffers against the checkpoint's, over all 57
layers:

| | median mean-shift | median var-shift | worst layer, mean | worst layer, var |
|---|---|---|---|---|
| SNGS-024 | 0.0387 | 0.1903 | 0.6466 | 0.7381 |
| SNGS-051 | 0.0514 | 0.2113 | 0.4793 | 1.3737 |
| SNGS-078 | 0.0405 | 0.1728 | 0.7563 | 0.9144 |
| **range over all 10** | **0.0387-0.0514** | **0.1728-0.2155** | 0.287-0.756 | 0.738-1.374 |

A ~4-5% shift in the means and a ~19% shift in the variances is enough to cost 17.6 GS-HOTA, and
individual layers move by 30-140%. The embedder's single BNNeck moves 20x further than the detector's
median (0.87-0.97) and costs 13.5. Full table: `results/gsr_benchmark/gsr_v8_w4_stats.json`.

### 2.6 Cost

| stage | wall clock | device |
|---|---|---|
| BN statistics (10 sequences, 150 frames each) | **82 s** | GPU |
| extraction, both lineages (20 sequence-extractions) | 8,989 s | GPU |
| per-crop OCR, both lineages | 3,254 s | GPU |
| box cache, both lineages | 610 s | GPU |
| CLIP embeddings, both lineages | 2,025 s | GPU |
| BNNeck arm embeddings (2-pass, 10 sequences) | 1,241 s | GPU |
| calibration re-gate, arm scoring, summaries, repro | ~780 s | CPU |
| **total** | **~4.6 h GPU + ~13 min CPU** | RTX 3050 (4 GB), one job at a time |

**The adaptation itself is as cheap as advertised: 8.2 s per sequence.** Everything else was the
price of measuring it honestly end to end — a detector change invalidates every cached artifact keyed
by track id, and the same-stack control had to be built from scratch because §2.1 failed.

---

## 3. Negatives, limits, and what was NOT done

1. **The registered arm FAILED at -17.5645 against a `>= +0.10` bar, 0/10 sequences helped,
   `p = 0.00195`.** The secondary arm FAILED at -13.5219, 0/10. Neither is a marginal call.
2. **DEV-20 was not run.** The registration conditions it on clearing the kill bar; it did not.
3. **`METRICS_VERSION` was NOT bumped** and no on-record artifact was overwritten. The stage is
   default OFF (`generator.extract.BN_STATS is None`) and the shipped chain is unchanged when it is.
4. **No unregistered rescue arm was run.** Momentum 0.1 partial update, a BN-layer subset (e.g. the
   first stage only), a larger adaptation batch, a re-calibrated `app_max` for the BNNeck arm — all
   are plausible and all would have been score-chosen. None was measured. The honest statement is
   that **full cumulative recomputation, the operation registered, fails**; a damped or partial
   variant is a different, untested hypothesis.
5. **Batch size 1 is a confound worth naming.** The adaptation pass runs one frame per forward, so
   each BatchNorm's per-batch estimate comes from a single image (over its spatial map). Cumulative
   averaging over 150 frames makes the *mean* estimate sound, but a per-image variance is not the
   per-batch variance a YOLO trains against. This was not varied because the stride and the pass
   structure were registered a priori.
6. **The premise is corpus-specific.** This measures BN adaptation on a detector already fine-tuned
   on the target corpus. It says nothing about the case the lever was originally invoked for — an
   EPL/ManU transfer where the target domain genuinely differs — except as a warning that the
   downstream thresholds are calibrated to the detector's confidence distribution, so any operation
   that shifts that distribution has to be paid for with a re-calibration.
7. **The cross-stack irreproducibility (§2.1) is reported, not diagnosed.** Which package changed
   (supervision's ByteTrack, ultralytics' NMS, torch's kernels) was not isolated; the session needed
   only to know that a fresh extraction may not be paired against on-record artifacts. Isolating it
   is a separate job and matters to the whole campaign, not just this session.
8. **The mechanism probe (§2.4) is 50 frames per sequence on 4 of 10 sequences**, at the detector
   only. It explains the direction and rough size of the failure; it is not a full detection
   benchmark and no mAP was computed.
9. **Nothing of the winner's was used, run or read.** No TEST-38, no test-49, no submission, no
   commit.

## 4. What this means for the campaign (for the orchestrator, not a result)

- **The cheapest lever on the list is dead, and it died cheap** — 8 s per sequence of actual
  adaptation, ~4.6 GPU-h of honest measurement, one afternoon. That is the intended outcome of a
  registered kill bar.
- **The detector substrate is fragile to anything that moves its confidence distribution.** The whole
  chain below it (ByteTrack thresholds, EIoU gate, connector `tau`, the solver) is calibrated against
  the shipped detector's output statistics. This is the first session to price that coupling: a 0.06-0.20
  drop in median confidence, with no other change, costs 17.6 GS-HOTA. Any future detector work —
  including the "detector/tracker substrate" route W2 named as one of the two unpriced options — must
  budget a re-calibration of the downstream thresholds, or it will measure the coupling instead of
  the detector.
- **A campaign-level hazard was uncovered by accident**: on-record DEV GPU artifacts cannot be
  reproduced on the current stack, and the drift is worth -1.14 GS-HOTA. Every session since the
  stack moved has been safe only because it reused cached detections (W2, W2b). The next session that
  re-extracts must build its own control, as this one did.
- **`generator/bn_adapt.py` is worth keeping** as a validated, tested instrument (a generic
  BatchNorm statistics context manager plus snapshot/apply and the ultralytics fuse dance). It is 
  what a future domain-transfer session would need on day one, and it now comes with a measured
  warning about what it costs in-domain.

## 5. Reproduce

```
python -m tools.gsr_w4_bnstats --stages demo                                  # self-check
python -m generator.bn_adapt                                                  # stage self-check
pytest tests/test_bn_adapt.py -q                                              # 8 cases, CPU

python -m tools.gsr_w4_bnstats --lineage w4bn  --stages stats                 # GPU 82 s
python -m tools.gsr_w4_bnstats --lineage w4bn  --stages extract               # GPU ~75 min
python -m tools.gsr_w4_bnstats --lineage w4bn  --stages gate,percrop,boxes,embed
python -m tools.gsr_w4_bnstats --lineage w4off --stages extract               # same-stack control
python -m tools.gsr_w4_bnstats --lineage w4off --stages gate,percrop,boxes,embed
python -m tools.gsr_w4_bnstats --stages neck                                  # GPU ~21 min

python -m tools.gsr_w4_bnstats --lineage w4off --stages arm                   # CPU
python -m tools.gsr_w4_bnstats --lineage w4bn  --stages arm
python -m tools.gsr_w4_bnstats --stages neckarm,repro
python -m tools.gsr_w4_bnstats --stages summary --summary w4bn:w4off          # THE registered read
```

Each stage must run as its own process: extraction of 9 sequences leaves the process near the host
RAM ceiling and the calibration re-gate's `.npz` loads then fail with `_ArrayMemoryError` (observed
once, at 5.1 GB free of 15.8 GB).

## 6. Files

- `generator/bn_adapt.py` (new): `recomputing` (the generic BatchNorm statistics context manager),
  `stats_of` / `apply_stats` (per-sequence snapshots), `adapt_yolo` (the ultralytics fuse dance), and
  a `--demo` self-check.
- `generator/extract.py`: **+9 lines** — the `BN_STATS` module flag (default `None`) and its
  application in `_build_detector`, mirroring the existing `FOOTBALL_WEIGHTS` rebinding mechanism.
- `tools/gsr_w4_bnstats.py` (new): the stats/extract/gate/percrop/boxes/embed/arm/neck/neckarm/
  repro/summary runner, the `w4bn` / `w4off` lineage switch, and its demo.
- `tests/test_bn_adapt.py` (new): 8 CPU cases — cumulative statistics equal the dataset statistics,
  no weight changes, no gradients, mode restoration, reset vs momentum, snapshot roundtrip, foreign
  snapshot rejection, per-sequence isolation, and that the stage is OFF by default.
- `results/GSR_V8_W4.md` (this file);
  `results/gsr_benchmark/gsr_v8_w4_{w4bn,w4off,w4neck}_probe.json`,
  `gsr_v8_w4_stats.json`, `gsr_v8_w4_repro.json`, `gsr_v8_w4_summary_{w4bn_vs_w4off,
  w4bn_vs_onrecord, w4off_vs_onrecord, w4neck_vs_onrecord}.json`.
- `outputs/gsr/{bn_stats_w4bn, positions_w4bn, positions_w4off, positions_w4off_rerun,
  positions_gate_w4bn, positions_gate_w4off, koshkina_percrop_v6_w4bn, koshkina_percrop_v6_w4off,
  detbox_cache_w4bn, detbox_cache_w4off, detembed_cache_clip_w4bn, detembed_cache_clip_w4off,
  detembed_cache_clip_w4neck}/`, `outputs/gsr/w4_*.log` (gitignored). **No on-record artifact was
  overwritten or deleted.**
