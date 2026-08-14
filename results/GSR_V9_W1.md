# v9 W1 — the metric-coordinate associator: licence check + training-data factory (2026-08-14)

Campaign v9 opens with one target, set by Sid: **beat 61.48 GS-HOTA on the live SoccerNet GSR test
board.** Our public number is **53.09** (5 lifetime submissions left, ~4th). The vehicle is a
learned **tracklet associator on metric pitch coordinates** — every published associator in the
DanceTrack lineage (TWiX, MOTIP, SUSHI) works in image coordinates; our calibration already removes
camera motion, so the associator sees physically-bounded kinematics in metres instead.

W1 is foundations only: the model-lineage licence check (what W2 may build from), the training-data
factory (what W2 trains on), and the campaign registration. **No associator is trained here.**

---

## 1. Ceiling math — what +11.9 is, and what it is not

From `results/GSR_V7_V4S1.md` (measured on this exact stack: S4b detector -> ByteTrack -> PnLCalib
-> calibration gate/fill -> EIoU, DEV-20):

| quantity | value | reading |
|---|---|---|
| shipped chain (tracker + GTA connector + MILP solver), partition E | 66.07 `loc_assoc` AssA | what we score today |
| **merge-only oracle on the tracker output D** | **+11.89 AssA** | what a *connector* can reach: it may merge fragments, never split a contaminated tracklet |
| split+merge oracle on D | +24.79 AssA | a *tracker replacement*, not a connector |
| coverage+contamination oracle on E | +8.51 AssA | fragmentation beats coverage **2.91 : 1**; 19/20 sequences fragmentation-dominated |
| realised by the incumbent GTA connector (tau 0.450 + jersey-merge gate) | 61.3% of the merge-only headroom, 43.2% of the split+merge headroom | the connector merges 1,380 times at **0.5432** merge precision |

**What +11.9 is:** the AssA a *perfect merger* would add on top of the raw EIoU partition, on the
proxy instrument's `loc_assoc` scale.
**What +11.9 is not:** (a) it is not GS-HOTA — GS-HOTA is the geometric mean of GS-DetA and GS-AssA,
so a +11.9 AssA gain is worth **+4.24 GS-HOTA** at our DetA (arithmetic below), not +11.9; (b) it is not
additive with the connector's existing gain — the connector has already spent 61% of it, and its
wrong merges have destroyed a further 0.048 proxy of its own remaining ceiling (V4S1 §2); (c) the
proxy's Spearman against `trackeval` on this stack is +0.842, so levels carry ±3 AssA, not ±1.

**Honest arithmetic for the campaign target — do this before spending a GPU-week.** Our board entry
is GS-HOTA **53.08 = DetA 39.32 / AssA 71.67** (STATUS 2026-08-06; GS-HOTA is
`sqrt(DetA x AssA)`, and 39.32 x 71.67 reproduces 53.08 exactly). Adding the association oracles to
AssA and holding DetA fixed:

| scenario | AssA | GS-HOTA | delta |
|---|---|---|---|
| shipped | 71.67 | 53.08 | — |
| **merge-only oracle fully realised** (a *perfect* connector/associator) | 83.56 | **57.32** | **+4.24** |
| split+merge oracle (a tracker/linker replacement, may also split) | 96.46 | **61.59** | +8.51 |
| board leader | | 61.48 | +8.40 |

**So: a perfect merger lands at ~57.3, roughly 4.2 GS-HOTA short of 61.48.** The target is reachable
only in the split+merge band — i.e. the associator must be paired with a *splitter* for the
contaminated tracklets (13.3% of our tracklets audit below 0.8 purity, §4.1), or with a DetA lever,
or both. A merge-only associator that realised **100%** of its oracle would still miss the campaign
goal. Registered here so no later session reads +11.9 AssA as +11.9 GS-HOTA, and so W2/W3 plan for
the pairing rather than discovering the shortfall after training.

Assumption stated: the +11.89/+24.79 were measured on DEV-20's `loc_assoc` proxy scale and are
applied here as additive AssA deltas on test-49. Cross-split additivity is an assumption, not a
measurement; DEV-20 tracked valid-58 to within 0.008 on the ceiling (V4S1 §4), which is the only
support it has.

---

## 2. Task A — model-lineage licence check

| repo | licence | stars / activity | training code self-contained? | verdict |
|---|---|---|---|---|
| **TWiX** — `Guepardow/TWiX` (arXiv 2403.08018, *Pattern Recognition* 2024) | **MIT** (c) 2024 Mehdi Miah | 13 stars, 0 forks; last commit 2024-11-29 (quiet, but complete) | **Yes** — `src/association/twix/{data.py,train.py,twix.py,loss.py}`; the README's training recipe is exactly our pipeline shape: run detector -> IoU tracker -> build tracklet-pair batches (`data.py`) -> train (`train.py`). Detection code is *not* included (we have our own). | **USABLE-DIRECT** |
| **MOTIP** — `MCG-NJU/MOTIP` (CVPR 2025) | **Apache-2.0** | 553 stars, 51 forks; last commit 2026-07-30 (actively maintained) | **Yes** — `train.py`, `configs/`, `models/`, vendored `TrackEval`. Heavier: an end-to-end DETR-based ID-prediction tracker, not a tracklet post-processor. | **USABLE-DIRECT** (but architecture mismatch, see §5) |
| **SUSHI** — `dvl-tum/SUSHI` (CVPR 2023) | **MIT** (c) 2023 Dynamic Vision and Learning Group | 142 stars, 13 forks; last commit 2024-12-22 | **Yes** — `src/`, `configs/`, `scripts/`, `environment.yml`, vendored `TrackEval`. Hierarchical GNN over tracklets; needs precomputed detections + reID features, which we have. | **USABLE-DIRECT** |

All three are permissively licensed, so W2 may **build from the code** rather than clean-room from
the paper. Nothing was cloned into this repo tree; only the public metadata and two source files
were read over HTTPS.

Two technical notes taken while reading the source (they change W2's design, so they are recorded
now rather than rediscovered later):

1. **TWiX is not a plain pairwise scorer.** `src/association/twix/twix.py` projects 4-D boxes to
   `d_model`, adds a sin/cos *temporal* positional encoding normalised so the future tracklet starts
   at t=0, and — with `inter_pair=True` — attends **across all candidate pairs in the window**, so
   each pair is scored in the context of every other pair. That context term is most of what beats
   IoU heuristics, and it is cheap to keep.
2. **TWiX's `NormCoords` rescales each window's coordinates into [-1, 1].** That is a *fix* for
   image coordinates (scale changes with camera zoom) and a *bug* for ours: in metric pitch space
   the absolute scale is the signal (9 m/s is a hard physical bound). Our port must drop or
   condition that normalisation. This is exactly the seam the "metric coordinates" novelty claim
   lives on.

---

## 3. W1 registration — the factory design (fixed before the run)

Written and code-frozen (`tools/gsr_v9_factory.py`, self-check `--stage demo` green) **before** any
train-split artifact existed on disk.

**Principle (the CAMELTrack lesson, on record):** the associator trains on OUR pipeline's output
distribution, not on GT boxes.

1. **Chain (frozen v6 bundle, `results/gsr_v6_frozen.json`):** S4b detector (`gsr_v3_ft_b_last.pt`,
   md5 `2074d874…`) -> ByteTrack -> PnLCalib -> calibration re-gate + 10-frame gap fill -> EIoU
   re-association (e 0.3, rounds 1, w_app 0.5, app_max 0.30, CLIP embeddings) -> metric positions.
   **The GTA connector and the MILP solver are deliberately NOT run**: the fragmented EIoU partition
   is precisely what the associator must learn to merge. The PARSeq v6 arm-4t jersey reader IS run —
   its per-crop reads become features.
2. **Split:** GSR **train**, 57 sequences (`sequences_info.json`). GSR valid (DEV-20 + TEST-38),
   test-49 and challenge are not read at any point in W1.
3. **Labels:** per-row nearest-GT within the evaluator's 5 m tolerance
   (`eval.gsr_gta.gt_rows` — the same auditing rule as `tools.gsr_eiou.partition_purity`), then the
   majority GT id per tracklet. A tracklet with fewer than 5 audited rows gets **no** label (`-1`)
   rather than a guessed one. Referees are labelled too, through a referee-only GT map (the shared
   loader covers players/GKs only; referees are scored identities under GS-HOTA and fragment like
   everyone else). Per-tracklet purity is recorded alongside the label.
4. **Candidate rule:** an ordered pair (A ends before B starts) is a candidate iff the gap is
   positive, at most 750 frames (the whole sequence), and the endpoint distance is closable at
   **9 m/s + 2 m slack** — the exact physical constraint the incumbent connector applies at
   inference. **Team, role and jersey agreement are features, never filters** — they are the parts a
   learned model is meant to beat the connector on. The generator's recall (fraction of true
   same-identity pairs it keeps) is measured, not assumed.
5. **Features (per pair):** temporal gap (frames, seconds); both tracklets' lengths; endpoint metric
   positions; endpoint velocities over a 5-frame window; endpoint distance and the speed the gap
   demands; **forward and backward constant-velocity extrapolation residuals in metres** (the
   feature image-space associators cannot form); team ids + agreement; roles + agreement; dominant
   jersey number, its confidence mass and agreement flag on each side; mean detection confidence;
   per-side GT purity and ids; label.
6. **Serialisation:** `outputs/gsr/v9_assoc/v1/` — `tracklets/<seq>.parquet` (every non-ball row,
   full metric trajectories + per-row GT id, so W2 can consume pairwise *or* as a graph),
   `pairs/<seq>.parquet`, `manifest.json`, `per_sequence.csv`. Mirrored server-side.
7. **Validation slice (registered):** every 5th sequence of the sorted train split (12 sequences) is
   W2's model-selection set. Model selection never touches DEV-20 — **DEV-20 stays the frozen
   end-to-end gate for W3**, and TEST-38/test-49 stay frozen for the submission decision.
8. **Stack-drift note (kb v8-w4-004):** the train-split artifacts are a *new* extraction on the
   cluster stack. They are fine as training data, but any W3 evaluation that pairs an
   associator-on arm against an associator-off control must produce **both** on the same stack.

---

## 4. Results — the dataset

**Run:** cluster `a100server1`, GPU 1 only (GPU 0 checked idle and left alone), tmux `v9w1`,
`~/v9w1_chain.sh` -> `~/v9w1_chain.log`. Nothing for the train split existed server-side (0/57 in
every artifact directory), so all five stages were extracted fresh:

| stage | shards | wall clock | note |
|---|---|---|---|
| extract (S4b detector + ByteTrack + PnLCalib, `calib_period=1`) | 8 | 20:13 -> 21:03 (49.6 min) | 393-418 s per sequence per shard |
| gate (calibration re-gate + 10-frame fill) | 1 | 26 s | CPU |
| percrop (PARSeq v6 arm-4t jersey reader) | 6 | 25.0 min | |
| boxes (per-detection box cache) | 6 | 4.2 min | |
| embed (CLIP per-detection embeddings) | 6 | 18.4 min | |
| **total GPU** | | **97.6 min = 1.63 GPU-h** | of ~15 budgeted |
| EIoU re-association + dataset build | 1 | 4.7 min | CPU, laptop-identical module (md5 checked both sides) |

`5,787` EIoU tracklets from `3,061` ByteTrack tracks (**1.89x fragmentation**, the expected
direction — EIoU buys purity with fragments).

### 4.1 Dataset stats (`outputs/gsr/v9_assoc/v1/`, 33 MB, 57+57 parquets + manifest)

| quantity | value |
|---|---|
| sequences / detection rows | 57 / 657,776 |
| tracklets (>= 3 rows) | 4,709 — **82.6 per sequence** (min 44, max 160) |
| GT identities | 1,342 (23.5 per sequence) |
| tracklets with a trusted GT label (>= 5 audited rows) | 3,975 (84.4%); **734 unlabelled (15.6%)** — false positives and stubs, kept in the pair table with `label = -1` |
| **fragments per GT identity, dominant-label** | **2.98** (per-sequence 1.64 .. 5.28) |
| **fragments per GT identity, on-record "touch" definition** | **7.25** |
| row-weighted tracklet purity | **0.9576** (per-sequence 0.8717 .. 0.9951) |
| per-tracklet purity: median / p25 / p05 | 1.000 / 0.903 / 0.545; 47.4% are >= 0.99, 13.3% below 0.8 |
| tracklets carrying a jersey read (>= 0.80 crop confidence) | 1,651 (35.1%) |
| **candidate pairs** | **121,106** — 5,561 positive / 72,936 negative / 42,609 unlabelled |
| positive rate among labelled pairs | **7.08%** |
| **candidate-generator recall** | **0.9851** (per-sequence min 0.9388) — the physical 9 m/s + 2 m filter loses 1.5% of true pairs; that is the associator's recall ceiling |

**Premise check on fragmentation.** The task brief expected "~11.5" fragments per identity. That
number is from `GSR_EIOU.md` on the **old** detector; on the stack we actually ship,
`GSR_V7_V4S1.md` §3.2 measured **7.91** (DEV-20, touch definition). The train split gives **7.25**
by that same definition — i.e. the training distribution matches the evaluation stack to within 0.7
fragments, and the "11.5" figure would have been the wrong target. Under the stricter
dominant-label definition (a tracklet counts for the identity it mostly is) the number is **2.98**:
most "touches" are a handful of rows in passing, not a fragment worth merging.

### 4.2 Split, held out by sequence (registered in §3.7, no peeking)

| slice | sequences | tracklets | pairs | pos | neg | unlab |
|---|---|---|---|---|---|---|
| development | 45 | 3,614 | 89,356 | 4,039 | 53,725 | 31,592 |
| **validation (every 5th)** | **12** | 1,095 | 31,750 | 1,522 | 19,211 | 11,017 |

Validation slice: SNGS-060, -065, -070, -075, -099, -104, -109, -114, -154, -159, -164, -169.

**LEAKAGE CHECK — clean.** All 57 sequences are in the GSR train split
(`sequences_info.json`); overlap with the valid split = **[]**, with the test split = **[]**.
DEV-20, TEST-38, test-49 and challenge were not read, scored or extracted at any point in W1. The
check is code, not a claim: `leakage_check()` raises `SystemExit` before the manifest is written.

### 4.3 Feature separability, development portion only (45 sequences; the val slice was not looked at)

Medians, positive vs negative pairs:

| feature | same identity | different identity |
|---|---|---|
| temporal gap (frames) | 102 | 221 |
| endpoint distance (m) | 5.73 | 22.37 |
| **forward constant-velocity residual (m)** | **11.00** | **36.09** |
| backward constant-velocity residual (m) | 10.46 | 33.16 |
| CLIP tracklet-mean cosine distance | 0.290 | 0.613 |

Positive-pair gaps are long: quartiles 24 / 102 / 247 frames, p90 = 409, p99 = 635 — **a quarter of
the true merges span more than 10 seconds**, which is why a short-window motion heuristic cannot
harvest them and why the long-range jersey/appearance edges have to be in the model.

### 4.4 The finding that changes the W2 case: the connector's hard gates cost 19.8% of all true merges

The incumbent GTA connector may only merge tracklets that agree on **team** and **role**, and never
two that read different jersey numbers. Measured on the development portion's 4,039 true pairs:

| connector gate | share of TRUE merges it forbids |
|---|---|
| team must match | **16.59%** |
| role must match | 1.51% |
| team **or** role | 17.63% |
| jersey numbers disagree | 2.72% |
| **all three together** | **19.83%** |

So one merge in five is **structurally unreachable** by the incumbent connector at any tau — its
recall ceiling on this distribution is 0.802, before appearance is even consulted. Our candidate
generator keeps 0.985. Treating team/role/jersey as *features* rather than filters is therefore not
a stylistic choice: it is where a fifth of the merge headroom lives.

Two contextual baselines on the same rows (pairwise, so **not** the connector's agglomerative
procedure — precision is not comparable to its on-record 0.5432 merge precision):

| rule | pairs admitted | precision | recall of true merges |
|---|---|---|---|
| team+role+jersey gates, CLIP cos < 0.30 | 2,726 | 0.685 | 0.463 |
| team+role+jersey gates, **CLIP cos < 0.450** (the shipped tau) | 8,978 | 0.312 | 0.693 |
| metric only: forward residual < 2 m | 1,754 | 0.486 | 0.211 |
| metric only: forward residual < 10 m | 9,499 | 0.204 | 0.479 |

Neither single signal is close to sufficient — which is the point of learning the combination.
Jersey agreement is the sharpest evidence we hold (79.7% of positives agree vs 3.5% of negatives)
but it is available on only **11.2%** of labelled pairs.

### 4.5 Negatives, limits, hazards

1. **A laptop artifact directory has mixed provenance.** `outputs/gsr/positions_gate_v6det_eiou/`
   is written *in place* by every EIoU arm (`write_variant` has no arm key). The laptop's
   SNGS-021 copy audits at row purity 0.7098 and 35 tracklets against 47 ByteTrack tracks — i.e.
   *fewer* tracklets than the input, which EIoU never produces — so it is some later session's arm,
   not the frozen one (on-record DEV-20 purity is 0.9395). Nothing in W1 depends on it; the train
   artifacts were built fresh server-side. Any session that reads those laptop files as "the v6
   partition" is reading the wrong thing.
2. **Labels are GT-proximity labels, not identity truth.** A row is labelled by nearest GT within
   5 m (the evaluator's own tolerance), so a tracklet that shadows a nearby player inherits a wrong
   dominant id. 13.3% of tracklets audit below 0.8 purity; W2 should weight or filter by purity and
   report the sensitivity rather than assume clean supervision.
3. **35% of candidate pairs are unlabelled** (one side has no trusted GT identity). They are kept
   deliberately — at inference the model faces exactly these — but they cannot enter a supervised
   loss, and treating them as negatives would be a silent labelling decision.
4. **Referees are labelled through a private referee-only GT map.** The shared loader covers
   players/GKs only. This is 15 lines in the factory rather than a change to shared code, so the
   on-record purity instrument is untouched; the cost is that the two label paths could drift.
5. **New-stack extraction (kb v8-w4-004).** These artifacts were produced on the cluster stack. They
   are fine as *training* data, but any W3 comparison must build its own same-stack control.
6. **No associator was trained, no model was selected, no GS-HOTA number exists in this session.**

---

## 5. W2 plan (DRAFT — real registration is written by W2 before training)

**Model shape (from §2's verdict): a metric-space port of TWiX** (MIT, USABLE-DIRECT — W2 builds
from the code, no clean-room needed). One transformer layer, `d_model` 64-128, a per-pair CLS token,
sin/cos temporal encoding, `inter_pair=True` so every candidate pair is scored in the context of the
others in its window. Three deliberate deviations from the paper, all on the metric-coordinate seam:

1. **Drop `NormCoords`.** Metric positions are already in a fixed frame; the absolute scale (a 9 m/s
   speed bound, a 105x68 m pitch) is the signal. Normalise by a *constant* (pitch half-length), not
   per-window extrema.
2. **Input is `(x, y, vx, vy)` in metres/(m/s)**, not `(x1, y1, x2, y2)` in pixels — box geometry is
   camera-dependent, our positions are not.
3. **Side features on the pair token:** team ids + agreement, role agreement, jersey number
   agreement + confidence mass, and the CLIP tracklet-mean cosine distance (so the model can *use*
   appearance instead of being replaced by it, and so the incumbent connector's single signal is
   inside the model's input rather than outside it).

**Baseline and component gate (pre-registered before training in W2's own doc):** the incumbent
GTA connector (tau 0.450 + jersey-compatible-merge gate) run on the **held-out train slice** (the 12
registered validation sequences), never on DEV-20. Comparison **at matched merge count**: let the
connector make M merges on those sequences; take the model's top-M scoring merges; report merge
precision (fraction of merges joining the same GT identity), resulting fragments per GT identity,
and pair-ranking AP over the whole candidate set. **Gate to proceed to W3:** model merge precision
must beat the connector's on the same slice at the same M by a margin larger than the training-seed
spread (measure it — three seeds, exactly as v8 W3 had to learn to do), otherwise the associator
does not ship and W2 reports a registered negative.

**Then W3 (out of scope here):** end-to-end DEV-20 with the connector replaced, against a
same-stack connector-on control (kb v8-w4-004), and only then a submission decision.

**Carried forward from §1 (do not lose this):** a merge-only associator cannot reach 61.48 even at
its oracle. W2 should therefore also cost out the splitter arm — the GTA Splitter already exists in
`generator/gta_link.py` and the factory's per-row GT labels make its precision measurable on the
same 57 sequences without another extraction.

---

## 6. Files

**Laptop** (`c:\Users\siddh_ygv5bws\football-synthesizer\`)
- `tools/gsr_v9_factory.py` — the factory (self-check `--stage demo`, ruff-clean, 100 cols)
- `tests/test_gsr_v9_factory.py` — 2 tests, green
- `results/GSR_V9_W1.md` — this document
- `outputs/gsr/v9_assoc/v1/{manifest.json, per_sequence.csv, tracklets/*.parquet (57),
  pairs/*.parquet (57)}` — 33 MB, mirrored from the server

**Cluster** (`siddhanth23519@192.168.3.19`)
- `~/v9w1_chain.sh`, `~/v9w1_chain.log`, `~/train57.txt` (the 57 train sequence names)
- `~/football-synthesizer/tools/gsr_v9_factory.py` (md5-identical to the laptop copy)
- `~/football-synthesizer/outputs/gsr/{positions_v6det, positions_gate_v6det,
  koshkina_percrop_v6_v6det, detbox_cache_v6det, detembed_cache_clip_v6det}` — +57 train sequences
  each; `positions_gate_v6det_eiou`, `koshkina_percrop_v6_v6det_eiou`,
  `detembed_cache_clip_v6det_eiou` — 57 train sequences (created by this session)
- `~/football-synthesizer/outputs/gsr/v9_assoc/v1/` — the dataset (server-side original)
