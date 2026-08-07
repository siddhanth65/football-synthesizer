# v7 session V4 step 2 — CAMELTrack, the learned-association arm (pre-registered)

Server `a100server1` GPU 1 (occupancy checked: GPU 0 585 MiB / another user, GPU 1 **4 MiB idle** at
session start) + laptop CPU for the scored chain. Control:
`results/gsr_benchmark/gsr_v7_control_dev.json` — DEV-20 GS-HOTA **51.808280888021784**, pinned stack
(`results/GSR_V7_V0.md`).

Routed here by `results/GSR_V7_V4S1.md`: the linking ceiling on the v6 stack is **+24.79 AssA**
(split+merge oracle) / **+11.89** (merge-only, what a connector alone can reach), 19 of 20 sequences
fragmentation-dominated, and the shipped GTA connector merges at **0.5432 precision** — destroying
0.048 of its own remaining ceiling. The plan's pre-registered thresholds route both readings to
CAMELTrack.

---

## 1. REGISTRATION (written before any arm was run or scored)

### 1.1 License record — Apache-2.0, recorded before use, nothing vendored

`git clone https://github.com/TrackingLaboratory/CAMELTrack.git` -> `a100server1:~/src/CAMELTrack`,
commit **`46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2`** (Sat 12 Jul 2025, "Update README.md" — the
repo's HEAD; the "Cleaning of the code for the training" item is still unchecked upstream).

| | |
|---|---|
| `LICENSE` present | **yes**, 201 lines, md5 `86d3f3a95c324c9479bd8986968f4327` |
| identified as | **Apache License, Version 2.0, January 2004** (verbatim boilerplate, unfilled `Copyright [yyyy] [name of copyright owner]` appendix) |
| posture | permissive; strictly more permissive than the GPL-2.0 (PnLCalib) and CC BY-NC 3.0 (Koshkina) already in the stack |
| what we do with it | **run it server-side, unmodified, from `~/src/CAMELTrack`; nothing is vendored into this repo.** No CAMELTrack source is copied, edited or re-licensed here. Our driver imports it. |
| upstream deps of note | `tracklab` (MIT, already installed server-side, editable at `~/src/tracklab`), `torchreid@git+victorjoos/keypoint_promptable_reidentification` (KPReID — **no license file**, same posture as PRTreID/Deep-EIoU already in the stack) |

`results/GSR_EIOU.md`'s clean-room rule is **not** invoked: EIoU was reimplemented from the paper
because Deep-EIoU ships no license. CAMELTrack ships Apache-2.0, so it is used as a dependency.

### 1.2 The input contract — checked BEFORE any arm, and it does not match our caches

CAMELTrack is a TrackLab `ImageLevelModule`. Per detection it consumes a dict of *cue* tensors; which
cues are required is set by the checkpoint's temporal encoders
(`cameltrack/architecture/det_tokenizers.py`):

| cue | tokenizer | required tensor | our cache |
|---|---|---|---|
| box | `BBoxLinProj` | `bbox_ltwh` (4) + `bbox_conf` (1) | **have** — `detbox_cache_v6det` (xyxy) + `conf` in `positions_gate_v6det` |
| pose | `KeypointsLinProj` | `keypoints_xyc` **(17, 3)**, COCO-17 | **derivable** — `generator.jersey_id` already runs torchvision KeypointRCNN (COCO-17) for the torso RoI, but only on legibility-gated crops, not per detection |
| appearance | `PartsEmbeddingsLinProj` | `embeddings` **(num_parts+1, emb_size) = (6, 128)** + `visibility_scores` (6) | **MISMATCH** — `detembed_cache_clip_v6det` is a **single global 256-d** vector, and it is **strided** (5,616 embeddings against 11,216 boxes on SNGS-021, `frame_stride=2`) |

**The appearance mismatch is structural, and it is the session's first finding.** The pretrained app
tokenizer is a `nn.Linear(128, 512)` fitted to **KPReID's part-based embedding space**
(`keypoint_promptable_reidentification`, 5 body parts + 1 global, 128-d each, each with a visibility
score). A 256-d CLIP vector cannot be reshaped, padded or truncated into that space and retain
meaning: the released weights would be reading a foreign basis. The task brief named
`detembed_cache_clip` as the appearance input; **it is not usable for a zero-shot arm** and this is
recorded as a deviation from the brief rather than papered over.

Two consequences, both pre-registered here:

1. The zero-shot arm must **compute KPReID part embeddings and COCO-17 keypoints on our own cached
   boxes** (a new GPU artifact), or drop those cues.
2. A trained-from-scratch arm is *free* of the contract — it can declare `emb_size=256`,
   `use_parts=false` and consume our CLIP cache directly. That is the shape the fine-tune arm takes
   if it is reached (§1.4 arm F).

**Two further contract facts that bind the integration:**

- **CAMELTrack drops detections.** `min_det_conf=0.4` filters at `preprocess`, `min_init_det_conf=0.6`
  gates track creation, and only `active`/`init` tracklets are emitted. Our chain requires **every**
  positions row to receive an id (`tools.gsr_eiou.associate`: "a detection is never dropped, so the
  row set — and therefore GS-DetA — is identical to the control's"). The driver therefore assigns any
  row CAMEL did not emit its **own fresh singleton id**, preserving the row set exactly. Recorded as a
  deliberate deviation from CAMELTrack's own tracklet management; without it the arm and the control
  would not be comparable on GS-DetA.
- **`Tracklet.count` is a class-level global.** It must be reset per sequence or ids leak across
  sequences.

### 1.3 Where the arm plugs in (and the V3 landmine)

`tools.gsr_eiou.relink_sequence` returns exactly one object the rest of the chain needs:
a remap `{(old_track_id, frame): new_track_id}`. `write_variant` then rewrites the three
`(track_id, frame)`-keyed artifacts (positions parquet, per-detection embedding npz, per-crop OCR
parquet) into `*_eiou` directories and `tools.gsr_v4.solve_v4_arm` runs the frozen connector + MILP
solver + official scorer over them.

**A CAMELTrack arm is therefore a drop-in replacement for `associate()` and nothing else.** The
association runs server-side (GPU); only the remap comes back to the laptop, where the *identical*
CPU chain that produced the control materialises and scores the arm. Jersey votes and the solver are
rebuilt per arm by `clear_arm_caches`, as the brief requires.

**The `GSR_V7_V3.md` §2.7 landmine is carried:** `tools.gsr_eiou.BOX_SUBDIR` is a module constant
defaulting to `detbox_cache` (the **base** detector's boxes). `tools.gsr_v6det.stage_arm` rebinds it
to `detbox_cache_v6det`; V3's harness did not, and silently matched `_v6det` positions against base
detector boxes for 55 minutes before being killed. Every driver in this session rebinds `BOX_SUBDIR`
explicitly and asserts the cache exists.

### 1.4 The arms

| arm | checkpoint | cues | new GPU artifact needed |
|---|---|---|---|
| **Z0** plumbing probe | `camel_bbox_bee24.ckpt` | bbox only | none |
| **Z1** | `camel_bbox_app_dancetrack.ckpt` | bbox + app | KPReID embeddings |
| **Z2** (the domain-closest zero-shot) | `camel_bbox_app_kps_sportsmot.ckpt` | bbox + app + kps | KPReID embeddings + COCO-17 keypoints |
| **Z3** | `camel_bbox_app_kps_global.ckpt` | bbox + app + kps | (same as Z2) |
| **F** fine-tune / from-scratch | trained on GSR train-57 | bbox + **our 256-d CLIP** (`emb_size=256`, `use_parts=false`) | train-57 detections + embeddings |

Z0 exists to validate the driver against a checkpoint that needs nothing we do not already hold; its
score is **not** a fair read on CAMELTrack (BEE24 = bees, bbox-only). Z2 is the honest zero-shot read.
Arm F runs only under the brief's rule — "only if zero-shot shows signal OR its failure is clearly
domain-shift" — and is budgeted at <= 4 GPU-h.

### 1.5 THE GATE (pre-registered, both halves required)

**`>= +1.0` DEV-20 GS-HOTA over the V0 control (51.8083) AND `>= 12/20` sequences helped.**
AssA gains that do not convert to GS-HOTA do not ship. Reported for every arm whether it passes or
fails, so pass/fail is mechanistically explained (the `GSR_V7_V4S1.md` decomposition quantities):

- `assa_hat`, `assa_merge_only`, `assa_perfect_link` on the arm's own partition
  (`tools.gsr_assoc_diag.partition_report`)
- fragments per GT identity, row purity, contaminated tracklets
  (`tools.gsr_eiou.partition_purity`)
- connector merge precision and merge count on the arm's partition
- named tracklets / total, admissible slots per sequence (the V1/V3 quantity)
- GS-DetA, to prove the row set was preserved

Guards: the `GSR_V5.md` §2.1 concentration guard (a gain whose top-2 sequences carry `>= 80%` of the
net is demoted); paired Wilcoxon reported with the arm count so multiplicity is visible.

### 1.6 Fallback (pre-registered, runs only if every CAMELTrack arm fails the gate)

**ONE** EIoU retune sweep on DEV-20, laptop CPU, over `e` / `rounds` / `w_app` around the frozen
point (e 0.3, rounds 1, w_app 0.5, app_max 0.30), same gate. Justification is on record: V4S1 §5
measured **+11.89 AssA** still available to a merge-only pass, and `GSR_EIOU.md` negative 4 records
that the connector's `tau = 0.450` was frozen on a *different* partition. Conservative-freeze rule:
the incumbent point ships unless a candidate clears both halves of the gate.

### 1.7 Not done, declared up front

No TEST-38 read, no test-49 read, no commit, no `METRICS_VERSION` bump (association is a pipeline
stage, not a metric). Weights and any trained checkpoint stay server-side under `~/data/camel/`.
The v6 chain, the detector, the reader and the solver are untouched in every arm.

### 1.8 Amendments made mid-session, each timestamped against what was known

1. **A `sim_threshold` sweep was added** after Z0 scored and after Z1's *track counts* were seen but
   before Z1's score. CAMELTrack's association threshold is per-dataset in the paper
   (DanceTrack/SportsMOT 0.1, MOT17 0.475, PoseTrack 0.45, BEE24 0.6) and no value exists for
   football; leaving it at one arbitrary value would have made a failure uninterpretable.
2. **A cheap partition screen replaced full scoring for the exploratory points** (`camel_screen.py`:
   purity, fragments/GT and the V4s1 counterfactuals on the arm's own partition, ~1 server-minute
   against ~15 laptop-minutes). **Selection for the gate is still end-to-end GS-HOTA**; the screen
   only decides which point is worth a full run, and it is validated against the laptop by exact
   agreement on `z1_sportsmot` (§2.1).
3. **Arm F became "train from scratch on our cues" rather than "fine-tune their weights"**, forced by
   §1.2: there is no checkpoint whose appearance tokenizer can read a 256-d global CLIP vector, so
   there is nothing to fine-tune. Declared before any training run.

---

## 2. RESULTS

### 2.1 The integration is sound — three harness checks before any number is believed

| check | result |
|---|---|
| **row set preserved** (GS-DetA comparability) | `positions_gate_v6det_eiou` holds **230,753** rows on the CAMEL arm against **230,753** on the source `positions_gate_v6det` — identical. `singleton_fallback = 0` on every sequence of every arm, i.e. CAMELTrack emitted an id for all 221,397 non-ball rows unaided |
| **`BOX_SUBDIR` rebound** (the `GSR_V7_V3.md` §2.7 landmine) | `tools.gsr_v7_camel.score_arm` rebinds to `detbox_cache_v6det` and raises if the directory is absent |
| **cross-machine agreement of the diagnostics** | the `z1_sportsmot` partition screened on the **server** and diagnosed on the **laptop** agrees to every printed digit: `assa_hat` 0.3342, `assa_merge_only` 0.3593, `assa_perfect_link` 0.8549, frag/GT 6.4648, purity 0.6097, 467 tracklets, 242 contaminated |

### 2.2 Zero-shot — every released checkpoint is FAR below the incumbent, and the reason is measured

The appearance cue could not be supplied (§1.2), so each multi-cue checkpoint runs through its own
trained `drop_app` / `drop_kps` path — bbox-only inference of a model trained with all cues.

**Partition screen** (`camel_screen.py`, DEV-20, on each arm's own tracker output, before the
connector). The comparison row is `results/GSR_V7_V4S1.md`'s partition **D** — the v6 tracker output
the control is built on:

| arm | ckpt / `sim_threshold` | tracklets | frag/GT | **row purity** | `assa_hat` | `assa_merge_only` | `assa_perfect_link` |
|---|---|---|---|---|---|---|---|
| **D — v6det + EIoU (the control's tracker)** | e0.3 r1 w0.5 a0.30 | **2,116** | **7.91** | **0.9395** | **0.4489** | **0.7415** | 0.8642 |
| Z0 `camel_bbox_bee24` | 0.6 | 2,194 | 10.01 | 0.8336 | 0.3078 | 0.5761 | 0.8635 |
| Z1 `camel_bbox_app_kps_sportsmot` | 0.1 | **467** | 6.47 | **0.6097** | 0.3342 | 0.3593 | 0.8549 |
| Z2 `camel_bbox_app_kps_global` | 0.5 | 6,088 | 50.19 | 0.6821 | **0.1710** | 0.3763 | 0.8607 |

`assa_perfect_link` is essentially constant across all four (0.855-0.864) — the *detections* are the
same, so the oracle ceiling is the same and only the linker differs. **No zero-shot arm reaches the
incumbent's `assa_hat` 0.4489; the best is 0.3342.**

**End-to-end**, through the identical v6 chain (connector tau 0.450, jersey votes and MILP solver
rebuilt per arm, official scorer), against the pinned control **51.8083**:

| arm | GS-HOTA | delta | GS-DetA | GS-AssA | IDF1 | helped | Wilcoxon p | connector merges / precision |
|---|---|---|---|---|---|---|---|---|
| **control (v6, EIoU)** | **51.8083** | — | 39.308 | 68.287 | 56.45 | — | — | 1,380 / 0.5432 |
| Z0 `bee24` @0.6 | 38.1984 | **-13.6098** | 27.729 | 52.623 | 41.27 | **0/20** | 1.9e-06 | 1,391 / 0.5623 |
| Z1 `sportsmot` @0.1 | 31.6038 | **-20.2045** | 22.602 | 44.192 | 33.86 | **0/20** | 1.9e-06 | **26 / 0.400** |
| Z2 `global` @0.5 | *declined* — screened at `assa_hat` 0.1710, the worst partition measured | | | | | | | |

**GATE: FAIL, by 14.6 and 21.2 GS-HOTA and on 0 of 20 sequences.**

**The mechanism, in one sentence:** with the appearance cue removed, CAMEL associates on box geometry
alone, and box geometry cannot tell 22 same-kit footballers apart — so at a permissive threshold it
merges them into 467 tracklets at **0.61 purity** (242 of 467 contaminated), and at a strict one it
shatters them into 6,088 fragments. Two independent confirmations that this, and not the score
plumbing, is what happened:

1. `assa_merge_only` on Z1 (0.3593) is barely above its own `assa_hat` (0.3342): **there is almost
   nothing left for a merge-only pass to recover**, because contamination cannot be un-merged. On the
   control's partition the same gap is 0.4489 -> 0.7415.
2. The GTA connector, handed Z1's partition, finds only **26** merges at 0.400 precision against
   1,380 at 0.5432 on the control's — it *correctly* refuses to merge tracklets that are already
   mixed. The downstream is not being fooled; it is being starved.

A third, quieter number says the same thing about the objective: Z1's admissible set collapses to
**236 named tracklets of 444, 11 slots per sequence** against the control's 610/870 at 13.25 — the
solver cannot name a tracklet that contains two players.

### 2.3 The `sim_threshold` sweep — the two failure modes are the two ends of one knob

CAMELTrack's association threshold is dataset-specific in the paper and undefined for football.
Swept on the domain-closest checkpoint (`sportsmot`, bbox-only), screened not scored (§1.8.2):

| `sim_threshold` | tracklets | frag/GT | row purity | `assa_hat` | `assa_merge_only` | end-to-end |
|---|---|---|---|---|---|---|
| 0.1 (the paper's SportsMOT value) | 467 | 6.47 | 0.6097 | 0.3342 | 0.3593 | **31.6038 (-20.20)** |
| **0.3** | **467** | **6.47** | **0.6097** | **0.3342** | **0.3593** | — |
| 0.5 | 653 | 6.85 | 0.7043 | **0.3825** | 0.4465 | **36.6114 (-15.20)** |
| 0.6 (`bee24`, different ckpt) | 2,194 | 10.01 | 0.8336 | 0.3078 | 0.5761 | 38.1984 (-13.61) |
| 0.7 | *killed after 6/20 to free the GPU for arm F; the 0.5/0.6 rows already bracket the strict end* | | | | | |

**`sim_threshold` 0.1 and 0.3 give a BIT-IDENTICAL remap** (`np.array_equal` on SNGS-021's
`new_tid`, 19 tracks both) while the driver logs confirm the override reached the model
(`sim_threshold: 0.1` / `sim_threshold: 0.3`). The learned similarity is bimodal on this data with an
empty band between 0.1 and 0.3, so the knob is **inert** there. Raising it to 0.5 does help — purity
0.6097 -> 0.7043, `assa_hat` 0.3342 -> 0.3825, end-to-end -20.20 -> **-15.20** — and the arm is still
15 GS-HOTA below the control. **The threshold is not what is wrong.**

The full curve says the two failures are one trade: purity climbs monotonically with the threshold
(0.61 -> 0.70 -> 0.83) and fragmentation climbs with it (467 -> 653 -> 2,194 -> 6,088 tracklets), and
**no setting reaches the incumbent's 0.9395 purity at 2,116 tracklets**. EIoU's appearance-gated
geometry buys purity *without* paying the fragmentation, because it has an appearance cue and this
CAMEL does not.

### 2.4 Fallback — the pre-registered EIoU retune

Laptop CPU, the same chain, the same gate, one sweep. `w_app` is the incumbent's appearance weight,
`e` its expansion scale.

| point | GS-HOTA | delta | GS-DetA | GS-AssA | IDF1 | helped | p | merges / precision |
|---|---|---|---|---|---|---|---|---|
| **control** e0.3 r1 w0.5 a0.30 | **51.8083** | — | 39.308 | 68.287 | 56.45 | — | — | 1,380 / 0.5432 |
| **e0.3 r1 w0.7 a0.30** | **52.2936** | **+0.4853** | 39.804 | 68.704 | 56.97 | **12/20** | **0.0401** | 1,350 / 0.5496 |
| e0.1 r1 w0.5 a0.30 | 52.0788 | +0.2705 | 39.468 | 68.722 | 56.68 | 13/20 | 0.227 | 1,414 / 0.563 |
| e0.5 r1 w0.5 a0.30 | 52.0118 | +0.2035 | 39.616 | 68.288 | 56.63 | 10/20 | 0.658 | 1,351 / 0.547 |
| e0.2 r1 w0.5 a0.30 | 51.8940 | +0.0858 | 39.373 | 68.400 | 56.51 | 11/20 | 0.193 | 1,400 / 0.569 |
| e0.3 r2 w0.5 a0.30 | 51.7822 | -0.0261 | 39.331 | 68.178 | 56.32 | 5/20 | 0.638 | 1,341 / 0.534 |
| e0.3 r1 w0.5 a0.20 | 44.0385 | **-7.7698** | 32.789 | 59.154 | 46.80 | 1/20 | 3.8e-06 | 10,175 / 0.612 |

**GATE: FAIL on the level, PASS on the breadth.** The best point, `w_app 0.7`, is **+0.4853** with
**12/20 helped** and a paired Wilcoxon p of **0.0401** — it clears `>= 12/20` and misses `>= +1.0` by
half a point. Six points were swept, so p = 0.0401 does not survive Bonferroni (0.0083). The
pre-registered conservative-freeze rule applies: **the incumbent e0.3 r1 w0.5 a0.30 ships unchanged.**

Two things the sweep does establish:

1. **The appearance weight is the live knob, not the geometry.** `w_app` 0.5 -> 0.7 is worth +0.49;
   every `e` move is worth `<= +0.27`; a second scale-up round is worth -0.03. The direction is
   consistent with §2.2's diagnosis — on this partition, association quality is bought with
   appearance.
2. **`app_max` is a cliff, not a knob.** Tightening the appearance gate 0.30 -> 0.20 explodes the
   partition from 2,289 to **11,723** tracklets and costs **-7.77 GS-HOTA**, even though its merge
   precision is the *highest* in the table (0.612). More merges at better precision on a shattered
   partition is still a much worse submission: 10,175 merges recover less than the fragmentation
   destroyed. That is the same lesson §2.2 draws from Z2 (`global` @0.5, 6,088 tracklets), from an
   independent direction.

### 2.5 Arm F — CAMEL trained from scratch on GSR train-57 with OUR cues

The zero-shot failure is unambiguously "no appearance cue" (§2.2), and there is no released
checkpoint whose appearance tokenizer can read a 256-d global CLIP vector, so the escalation the
brief authorises ("only if... its failure is clearly domain-shift") is a **from-scratch train on our
own cue set**: `PartsEmbeddingsLinProj(use_parts=False, emb_size=256)` + `BBoxLinProj(use_conf=True)`,
CAMELTrack's own GAFFE and Temporal Encoders unchanged.

**Rung 0 — the pre-training unit-check (`camel_train.py --demo`, before any real run): PASS**, and it
caught two things that would otherwise have been invisible:

| assertion | measured |
|---|---|
| a batch round-trips through `CAMEL.train_val_preprocess` with the masks the model expects | asserted (`feats_masks` bool, `dets.feats_masks.shape[2] == 1`, all detections unmasked) |
| association accuracy rises on a linearly-separable 8-identity toy | **0.141 -> 1.000** (1,000 steps) |
| *finding:* the NTXent loss is NOT a progress signal | it plateaus at **2.708 = ln 15** at every learning rate, including runs whose accuracy is at chance |
| *finding:* the paper's `init_lr = 1e-4` **collapses** the GAFFE embedding here | embedding std across objects **1e-4**, accuracy 0.125 = chance at 1e-4; 3e-5 trains cleanly. lr 3e-5 was used |

**The corpus.** GSR train-57 ground truth: 57 sequences, **690,624 person rows** (player /
goalkeeper / referee; the ball has no identity), 1,324 identities, CLIP-embedded on every 2nd frame
so that a row without appearance at training time is exactly a row without appearance at inference
time (`GtaParams.frame_stride = 2`). 51 sequences train / 6 held out (`SNGS-060..065`). 318 MB,
38 minutes of A100 time.

**The fit.** 10 epochs x 700 steps x batch 8 = 56,000 detection-tracklet groups, **24 minutes** on one
A100 — the paper's "~1 h on a consumer GPU" is credible.

| epoch | train loss | val loss | **val association accuracy** |
|---|---|---|---|
| 0 | 3.1664 | 3.4641 | 0.6906 |
| 2 | 2.1872 | 3.4611 | 0.8383 |
| 4 | 1.9020 | 3.4611 | 0.8980 |
| 7 | 1.7002 | 3.4665 | 0.9238 |
| **9** | **1.6737** | 3.4663 | **0.9221** |

**The model learns the task it was trained on.** On held-out GSR sequences it puts the right tracklet
first for **92.2%** of detections, up from 69.1% at initialisation.

**And it does not transfer.** Screened on the DEV-20 partition it produces from our *detector's*
boxes:

| arm | `sim_threshold` | tracklets | frag/GT | row purity | `assa_hat` | `assa_merge_only` |
|---|---|---|---|---|---|---|
| **D — v6det + EIoU (incumbent)** | — | 2,116 | 7.91 | **0.9395** | **0.4489** | **0.7415** |
| F | 0.3 | 566 | 7.07 | 0.6070 | 0.2963 | 0.3348 |
| F | 0.5 | 1,016 | 8.82 | 0.7122 | 0.3013 | 0.4242 |
| **F** | **0.7** | 1,612 | 11.25 | **0.7701** | 0.2848 | **0.4865** |
| best zero-shot (`sportsmot` @0.5) | 0.5 | 653 | 6.85 | 0.7043 | **0.3825** | 0.4465 |

Arm F is **not better than the zero-shot arms** on `assa_hat` and only marginally better on
`assa_merge_only` and purity — and it is far below the incumbent on every column. Both end-to-end
candidates were scored anyway, because the gate is end-to-end:

| arm | GS-HOTA | delta | GS-DetA | GS-AssA | IDF1 | helped | Wilcoxon p | merges / precision |
|---|---|---|---|---|---|---|---|---|
| control | **51.8083** | — | 39.308 | 68.287 | 56.45 | — | — | 1,380 / 0.5432 |
| F @0.7 (best `assa_merge_only`) | 33.2207 | **-18.5875** | 24.600 | 44.875 | 36.04 | **0/20** | 1.9e-06 | 875 / 0.426 |
| F @0.5 (best `assa_hat`) | 31.3951 | **-20.4131** | 23.165 | 42.561 | 33.48 | **0/20** | 1.9e-06 | 385 / 0.366 |

**GATE: FAIL, by 18.6 GS-HOTA, on 0 of 20 sequences.**

**This is the session's most interesting result and it is a negative:** a learned associator with
**92.2% held-out association accuracy on ground-truth tracklets** produces a partition at **0.77
purity** on detector output, against a hand-crafted EIoU's **0.94**. The training objective was
satisfied and the deployed task was not. The two candidate explanations, and what separates them:

1. **Train/test input mismatch.** The model was trained on GT boxes with GT identities — it never saw
   a false positive, a duplicated box, a box on the wrong player, or a detection whose foot point was
   mis-projected. Our augmentations (box shake sigma 0.05, appearance noise alpha 0.4, 40% appearance
   dropout, 20% observation dropout) simulate *noise*, not *wrong objects*. CAMELTrack's own recipe
   trains on **tracker states from a real detector** — that is what its `SwapSporadic` /
   `SwapOccluded` transforms exist to model, and neither was implemented here.
2. **The cue is genuinely too weak.** Our CLIP embedding is a single 256-d global vector at frame
   stride 2 and `GTA_LINK_STAGE1.md` §2 already measured its within-identity p90 (0.093) overlapping
   its cross-identity p10 (0.040-0.083). CAMELTrack's appearance cue is 6 part embeddings with
   per-part visibility from a ReID model trained for the purpose.

The evidence favours (1) being at least partly responsible: F@0.7's `assa_merge_only` (0.4865) is
higher than any zero-shot arm's, i.e. the learned appearance *is* doing something, but its raw
association `assa_hat` (0.2848) is the worst in the table — it merges confidently and wrongly, which
is the signature of a model that has never been shown a wrong object. Distinguishing them needs a
corpus built from our detector's boxes matched to GT (a ~4 GPU-hour re-extraction of train-57 that
this session did not have), and that is the honest next experiment, not another threshold.

---

## 3. VERDICT

**Every arm FAILS the pre-registered gate (`>= +1.0` DEV-20 GS-HOTA AND `>= 12/20` helped). The v6
association ships unchanged. No artifact from this session enters any freeze.**

| arm | best point | GS-HOTA | delta | helped | gate |
|---|---|---|---|---|---|
| zero-shot, released weights | `bee24` @0.6 | 38.1984 | -13.6098 | 0/20 | **FAIL** |
| zero-shot, domain-closest | `sportsmot` @0.5 | 36.6114 | -15.1969 | 0/20 | **FAIL** |
| trained on GSR train-57, our cues | F @0.7 | 33.2207 | -18.5875 | 0/20 | **FAIL** |
| **fallback: EIoU retune** | **e0.3 r1 w0.7 a0.30** | **52.2936** | **+0.4853** | **12/20** | **FAIL** (level) |
| **incumbent (ships)** | e0.3 r1 w0.5 a0.30 | 51.8083 | — | — | — |

**Mechanistic explanation, one paragraph.** `results/GSR_V7_V4S1.md` was right that the loss is
fragmentation, not coverage — the oracle ceiling `assa_perfect_link` is 0.855-0.864 on *every*
partition measured here, because they all consume the same detections. What this session adds is the
constraint V4S1 could not see: **on this benchmark association quality is bought with the appearance
cue, and every CAMELTrack arm we could actually run was appearance-starved.** The released weights
read KPReID's 6 x 128 part-embedding space, which our 256-d global CLIP cache cannot fill, so they ran
bbox-only and delivered either 0.61-purity merges (permissive threshold) or 6,088 fragments (strict).
The arm that *did* get our appearance cue learned it on ground-truth boxes and transferred at 0.77
purity against EIoU's 0.94. Meanwhile the one knob that moved the incumbent in the right direction
was `w_app` (+0.49), the appearance weight. **Fragmentation is the symptom; the appearance model is
the binding constraint** — V4S1 §5 said "that is the case for a learned association model rather than
another threshold", and the correction this session supplies is that the learned model needs a
learned *appearance*, not just a learned matcher.

**What this closes and what it opens.** It closes CAMELTrack as a v7 lever: the code works, the
integration is verified row-exact, the training converges, and the answer is no by 14-21 GS-HOTA. It
opens one concrete, pre-priced follow-up (v8, not v7): rebuild the arm-F corpus from **our detector's**
boxes matched to GT (~4 GPU-h re-extraction of train-57) and re-train, which is the only way to
separate "train/test input mismatch" from "the CLIP cue is too weak". A ReID upgrade (part embeddings
with visibility) is the other half and is a bigger project.

## 4. Negatives, limits, and what was NOT done

1. **The headline is four FAILs.** Nothing here is a candidate for the v7 bundle.
2. **The appearance cue was never tested zero-shot** (§1.2). Every released multi-cue checkpoint ran
   through its own `drop_app` path. This bounds the zero-shot verdict: it says "CAMELTrack *without
   appearance* fails on football", not "CAMELTrack fails on football". Running it faithfully needs
   KPReID (`torchreid@git+victorjoos/keypoint_promptable_reidentification`, no license file) plus a
   pose model for its keypoint prompts, in a fresh environment; that was scoped, priced and declined
   inside this session's budget, not overlooked.
3. **Only `sim_threshold` was swept on the zero-shot arms.** `max_wo_hits` (150) and
   `max_track_gallery_size` (50) stayed at CAMELTrack's defaults; the two confidence floors were set
   to 0.0 to preserve our row set (§1.2), which is itself a deviation from their tracklet management.
4. **`sim_threshold` 0.7 on `sportsmot` was killed at 6/20** to free the GPU for arm F. The 0.5 and
   0.6 rows bracket that end of the curve, but the point itself is missing.
5. **Z2 (`global` @0.5) was never scored end-to-end** — declined on a screen showing the worst
   `assa_hat` (0.1710) and 6,088 tracklets. Its row in §2.2 is a screen, not a score.
6. **Arm F's corpus is GT boxes, not detector boxes** (§2.5) — the single most likely reason it failed,
   and not fixed, because fixing it costs a train-57 re-extraction.
7. **Arm F's augmentations are a subset of CAMELTrack's.** `BBoxShake`, `AppEmbNoise`,
   `DropoutFeatures` and `DropoutSporadic` were implemented; **`SwapSporadic` and `SwapOccluded` were
   not**, and CAMELTrack's own ablation calls augmentation crucial. One seed, one architecture, one
   learning rate; no hyper-parameter search beyond the lr collapse the unit-check caught.
8. **The arm-F training loop is a reimplementation.** Upstream's is coupled to tracklab `TrackerState`
   pickles and its README still lists "cleaning of the code for the training" as an open TODO. The
   batch format here was reverse-engineered from `camel.py` and validated by the toy check and the val
   curve (0.69 -> 0.92); it is not upstream's loop and may differ from it.
9. **The fallback's best point is inside the multiplicity budget.** +0.4853 at p = 0.0401 over six
   swept points; the Bonferroni threshold is 0.0083. `results/GSR_V5.md`'s v5.1 lesson (+0.61 DEV ->
   -0.05 TEST-38) is exactly this size of effect, which is why the gate was set at +1.0.
10. **The connector `tau` was still not re-swept** (`GSR_EIOU.md` negative 4, restated by V4S1 §6.7).
    The fallback swept the tracker's parameters, not the connector's.
11. **Two GPU jobs overlapped on GPU 1** for ~20 minutes (the `z2_global` remap and the arm-F corpus
    pass). Both are small and the A100 has 40 GB, but per-stage timings in §5 are wall-clock and
    include that contention.
12. **Not done:** TEST-38, test-49, any commit, any `METRICS_VERSION` bump, any change to the shipped
    detector / reader / solver, any KPReID or pose model, SUSHI (still v8).

## 5. GPU spend

All on `a100server1` **GPU 1** (GPU 0 is another user's and was never touched; occupancy at session
start: GPU 0 585 MiB, GPU 1 4 MiB, both 0% utilisation).

| stage | GPU time |
|---|---|
| zero-shot remaps (Z0, Z1, Z2 + `sim_threshold` 0.3 / 0.5 / partial 0.7) | ~1.2 h |
| arm F corpus (CLIP over 690,624 GT boxes at stride 2) | ~0.6 h |
| arm F training (10 epochs x 700 steps x batch 8) | **0.4 h** |
| arm F inference (3 thresholds x 20 sequences) | ~0.4 h |
| unit-check + toy convergence diagnostics | ~0.4 h |
| **total** | **~3.0 GPU-h** of the 6-10 h budget |

Laptop CPU: 8 full end-to-end arms (6 EIoU sweep points + 2 CAMEL arms + 2 more CAMEL arms scored
earlier) at ~12 min each, plus 2 partition diagnoses. Disk: server `~/data/camel` **2.4 GB** (1.9 GB
released checkpoints, 318 MB corpus, 174 MB arm-F checkpoint); laptop `outputs/gsr/camel_remap` 6 MB.

## 6. Files

**Repo (code, both default-preserving):**
- `tools/gsr_eiou.py` — new `load_remap`; `build_arm_artifacts(..., remap_dir=)` and
  `run_point(..., remap_dir=)` substitute a precomputed association for `relink_sequence`. Every
  existing call site and default is unchanged; `--demo` still passes.
- `tools/gsr_v7_camel.py` (new) — `score_arm` (one remap through the frozen v6 chain, scored and
  paired against the V0 control), `score_eiou` (the fallback sweep), `diagnose` (the V4s1
  decomposition on an arm's partition). Rebinds `BOX_SUBDIR` to `detbox_cache_v6det` and raises if the
  cache is absent.

**Server (`a100server1`, nothing vendored into the repo):**
- `~/src/CAMELTrack` — commit `46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2`, **Apache-2.0**
  (`LICENSE` md5 `86d3f3a95c324c9479bd8986968f4327`), unmodified.
- `~/work/camel/camel_relink.py` — CAMEL association over our cached detections -> remap npz.
- `~/work/camel/camel_prep.py` — the train-57 GT + CLIP corpus.
- `~/work/camel/camel_train.py` — arm F training + the `--demo` unit-check.
- `~/work/camel/camel_screen.py` — the cheap partition screen.
- `~/data/camel/*.ckpt` — released weights, md5: `bee24` `6fcefbb2494dcca47efc201ad0025edb`,
  `app_dancetrack` `e22b3ace3f4cadf91497779f59554099`, `app_kps_sportsmot`
  `6074ce4f726652de70c0fd9dc3961652`, `app_kps_global` `2d646bad96cc4004cd80628e60eb8296`.
- `~/data/camel/camel_gsr_f.ckpt` — **arm F**, md5 `664df98b6191fd84e6c373c92cef5fab`, 174 MB, beside
  `camel_gsr_f.json` (the fit history).
- `~/data/camel/{traincorpus,remap,screen*.json}`, `~/logs/camel_*.log`.
- New pip packages in the `gsr` env: `pytorch-metric-learning 2.9.0`, `ordered-set 4.1.0`,
  `overrides 7.7.0` — dry-run verified first: exactly three installs, nothing else moved (torch,
  numpy, `supervision==0.30.0` untouched).

**Results:**
- `results/gsr_benchmark/gsr_v7_v4s2_{z0_bee24,z1_sportsmot,z1_sm_t0.5,f_t0.5,f_t0.7}.json` — the
  scored CAMEL arms.
- `results/gsr_benchmark/gsr_v7_v4s2_eiou_e*.json` — the six fallback points.
- `results/gsr_benchmark/gsr_v7_v4s2_z1_sportsmot_diag.json` — the laptop decomposition.
- `outputs/gsr/camel_remap/` — the per-arm remaps behind every CAMEL number.
- Claim: `v7-v4-002` (FAIL).

## 7. Reproduce

```
# server (GPU 1); CAMELTrack cloned to ~/src/CAMELTrack, weights auto-download from HF
python ~/work/camel/camel_train.py --demo                       # rung 0, minutes
python ~/work/camel/camel_relink.py --ckpt ~/data/camel/camel_bbox_app_kps_sportsmot.ckpt \
    --dest ~/data/camel/remap/z1_sm_t0.5 --sim-threshold 0.5 --drop app,kps --seqs <DEV-20>
python ~/work/camel/camel_prep.py                               # the train-57 corpus
python ~/work/camel/camel_train.py --epochs 10 --steps 700 --batch-size 8 --lr 3e-5
python ~/work/camel/camel_screen.py --arms <arm> --seqs <DEV-20> --out ~/data/camel/screen.json
# laptop (CPU)
python -m tools.gsr_eiou --demo                                 # the association self-check
python -m tools.gsr_v7_camel --arm z1_sm_t0.5 --diag
python -m tools.gsr_v7_camel --eiou-sweep "0.3:1:0.7:0.3"       # the fallback's best point
pytest tests/test_gsr_eiou.py tests/test_tracking.py tests/test_ocr_density.py
```
