# v9 W2 — the metric-space TWiX associator: registration, training, gates, splitter costing

Campaign target (Sid): **beat 61.48 GS-HOTA on the SoccerNet GSR test board**. W1 built the training
data (`outputs/gsr/v9_assoc/v1/`, GSR train split, 57 sequences) and registered the arithmetic that
governs this session: a **merge-only** oracle tops out at **GS-HOTA 57.32**, ~4.2 short of the
target; only the **split+merge** band (61.59) reaches it. W2 therefore delivers two things — the
trained merge model, and the splitter measurement on the same labels.

---

## 1. Registration (written before any training; the baseline numbers below were measured first)

### 1.1 Model — `generator/assoc_twix.py`, `TwixMetric`, version `v9-twixm-1.0`

Architecture follows **TWiX** (Miah et al., arXiv:2403.08018, `Guepardow/TWiX`, **MIT (c) 2024 Mehdi
Miah**; provenance header carried in the module). Re-implemented in metric space, not copied.

* **Stage 1, intra-pair.** One `nn.TransformerEncoderLayer` (`d_model 64`, `n_head 4`, `d_ff 128`,
  dropout 0.1, pre-norm) over `[CLS] + 8 past observations + 8 future observations` = 17 tokens.
* **Stage 2, inter-pair (KEPT, TWiX's `inter_pair=True`).** A second encoder layer over the CLS
  tokens of every candidate pair in the same window, so each pair is scored against its competitors.
* **Head.** LayerNorm -> Linear -> GELU -> Linear -> 1 logit. **72,385 parameters.**

**Observation token (`OBS_DIM = 5`), constant-scale normalisation — the registered deviation from
TWiX (`NormCoords` DROPPED):**

| slot | value | normaliser |
|---|---|---|
| 0 | pitch x (m) | `(x - 52.5) / 52.5` |
| 1 | pitch y (m) | `(y - 34.0) / 34.0` |
| 2 | vx (m/s), `np.gradient` over the tracklet's finite-pitch trajectory | `/ 9.0`, clipped to ±2 |
| 3 | vy (m/s) | `/ 9.0`, clipped to ±2 |
| 4 | side flag | `-1` past tracklet, `+1` future tracklet |

Absolute metric scale is preserved end to end (the model's self-check asserts that scaling all
coordinates by 2 changes the scores — i.e. the scale really is signal). Sampling: the **last 8**
observations of the past tracklet and the **first 8** of the future one, at a **stride of 5 frames**
(1.4 s of context per side at 25 fps); short tracklets are padded and masked. Time enters as a
signed sin/cos encoding of `(frame - first frame of the future tracklet) / 25` seconds, periods
0.2 s .. 60 s, added to the projected observation (TWiX's convention: the future tracklet starts at
t = 0).

**Pair-level side features (`SIDE_DIM = 9`), added to the CLS token** — team/role/jersey enter as
FEATURES, never as filters (kb v9-w1-004: the incumbent's hard gates forbid 19.83% of true merges):

`[team_same, team_unknown, role_same, jersey_match, jersey_conflict, log1p(min jersey mass)/3,
CLIP tracklet-mean cosine distance, clip_missing, gap_s/10 clipped at 4]`

Missing CLIP distance is filled with **0.6**, the negative-pair median from W1 §4.3 (i.e. "no
appearance evidence" defaults to the impostor prior, not to a neutral 0). Endpoint distance and the
constant-velocity residuals are **deliberately not** given as side features — the observation tokens
carry the positions, so the model must form them itself; that keeps the metric-kinematics novelty
claim clean.

**Windows (inter-pair grouping).** All candidate pairs whose *future* tracklet begins in the same
125-frame (5 s) bin of the same sequence form one window; windows larger than 384 pairs are chunked.
The W1 candidate set is used verbatim (physical 9 m/s + 2 m filter, recall 0.9851).

**Loss.** Plain `BCEWithLogits` (no `pos_weight`; the gates are ranking metrics) over **labelled
pairs only**. **Unlabelled pairs (`label = -1`, 35% of candidates) are EXCLUDED from the loss but
KEPT in the forward pass as inter-pair context** — they exist at inference and treating them as
negatives would be a silent labelling decision (W1 §4.5.3).

**Training config (registered):** AdamW, lr 3e-4, weight decay 1e-4, 8 windows per step, 30 epochs,
5% linear warmup then cosine decay, grad-norm clip 1.0. **Seeds 0, 1, 2** (3 minimum, noise floor =
max-min spread over seeds, computed before any verdict — the v8-W3 lesson).

**Splits.** Development 45 sequences -> **40 train / 5 inner-val** (every 9th of the sorted
development list: SNGS-061, -072, -102, -113, -160). The epoch is chosen by **inner-val** pair AP,
so the registered 12-sequence validation slice (SNGS-060, -065, -070, -075, -099, -104, -109, -114,
-154, -159, -164, -169) is only ever *measured*, never selected on. DEV-20, TEST-38, test-49 and
challenge are not read at any point in W2.

**Registered ablation arm (exactly one):** `use_side=False` — kinematics only, no team/role/jersey/
CLIP side features. This doubles as the novelty isolation: what metric kinematics alone buy.

### 1.2 Incumbent baseline — measured on the validation slice BEFORE training

`python -m tools.gsr_v9_train --stage baseline` — the shipped GTA connector (`generator.gta_link.
connect`, tau **0.450**, jersey-disagreement gate ON, team/role gates ON, splitter OFF) run on the
same EIoU partition the associator consumes, on the 12 registered validation sequences.

| quantity | no-merge partition | **incumbent connector** |
|---|---|---|
| tracklets (GT identities) | 1,095 (279) | 1,095 -> 450 components |
| merges | 0 | **645** |
| merge precision (pair-level, `track_relink.merge_precision`) | — | **0.7492** |
| fragments per GT identity | 3.315 | **1.799** |
| row-weighted purity | 0.9490 | **0.9313** |
| pair-ranking AP over the 20,733 labelled candidate pairs | — | **0.5586** |

The pair ranking the connector *implies* is `-clip_cos_dist` with its hard gates pushed below every
admissible pair (team mismatch, role mismatch, jersey conflict, missing embedding) — that is exactly
what the gates do at inference. Positive rate on the slice: 7.34%.

Note on comparability: 0.7492 is **not** the on-record 0.5432 from `GSR_V7_V4S1.md`. That number is
DEV-20, post-splitter, on the shipped chain; this one is train-split validation sequences,
connector-only, pair-level within components. The bars below are set from *this* measurement, on the
same rows the model is scored on, which is the only comparison that is like-for-like.

### 1.3 Component gates (fixed a priori; pass = all three, beyond the 3-seed noise floor)

* **G1 — ranking.** Model val pair-AP **> 0.5586**.
* **G2 — precision at matched merge count.** Per sequence, the model makes exactly the number of
  merges the incumbent made there (pooled 645), by accepting its top-scoring candidate pairs under
  single linkage with only the structural constraint a valid track must satisfy (segments temporally
  disjoint and reachable, `gta_link._chain_ok`). Pooled merge precision must be
  **>= 0.7492 + 0.05 = 0.7992**.
* **G3 — fragmentation at equal-or-better purity.** Fragments per GT identity **<= 1.799** with
  row-weighted purity **>= 0.9313**.

The verdict is PASS only if each margin exceeds the max-min spread over seeds 0/1/2.

### 1.4 Splitter costing (measurement, not training)

`generator.gta_link.split_tracklets` (DBSCAN over per-detection CLIP embeddings, `min_samples` 5,
`min_run` 5) at **eps 0.20 / 0.30 / 0.40** (0.30 = shipped default; the sweep is reported in full,
it is a measurement on train-split labels and selects nothing), plus a **per-row GT oracle splitter**
(cut wherever the audited GT identity changes, runs shorter than 5 audited rows absorbed) as the
upper bound. Reported over all 57 train sequences: split recall on the contaminated (<0.8 purity)
tracklets, split precision (share of split tracklets that were contaminated), false-split rate on
>= 0.99-pure tracklets, row-weighted purity before/after, and the fragment cost.

### 1.5 Anti-patterns named

No hyperparameter search on the validation slice beyond the registered arms (main + one ablation);
epoch selection runs on the inner-val slice only. No DEV-20/TEST-38/test-49/challenge contact of any
kind. Unlabelled pairs are never treated as negatives. Seed noise floor computed before any verdict.

---

## 2. Results — training

**Run:** cluster `a100server1`, GPU 1 only (GPU 0 checked, 585 MiB in use by someone else, left
alone), tmux `v9w2`, `~/work/v9/run_v9w2.sh` -> `~/work/v9/v9w2.log`. Six runs (2 arms x 3 seeds),
**2.6 min wall = 0.043 GPU-h** of the ~10 budgeted. Everything else in this session (baseline,
splitter costing, diagnostics) is laptop CPU.

### 2.1 Per-seed table (validation slice, 12 sequences, 1,095 tracklets, 279 GT identities)

Merges are matched per sequence to the incumbent's count (645 pooled) in every row.

| arm | seed | best epoch (inner-AP) | **val pair-AP** | **merge precision** | **frag/GT** | **purity** |
|---|---|---|---|---|---|---|
| main (side features) | 0 | 15 (0.8510) | 0.6760 | 0.7410 | 1.738 | 0.9343 |
| main | 1 | 3 (0.8401) | 0.6221 | 0.7163 | 1.760 | 0.9275 |
| main | 2 | 19 (0.8486) | 0.6758 | 0.7206 | 1.738 | 0.9309 |
| **main mean** | | | **0.6579** | **0.7260** | **1.746** | **0.9309** |
| **3-seed noise floor (max-min)** | | | **0.0540** | **0.0247** | **0.0215** | **0.0068** |
| ablation (kinematics only) | 0 | 12 (0.4143) | 0.2653 | 0.3424 | 2.455 | 0.8252 |
| ablation | 1 | 20 (0.4161) | 0.2769 | 0.3168 | 2.502 | 0.8223 |
| ablation | 2 | 20 (0.3977) | 0.2698 | 0.3236 | 2.459 | 0.8281 |
| **ablation mean** | | | **0.2707** | **0.3276** | **2.472** | **0.8252** |
| **incumbent GTA connector** | — | — | **0.5586** | **0.7492** | **1.799** | **0.9313** |
| no-merge partition | — | — | — | — | 3.315 | 0.9490 |

### 2.2 Gate verdicts

| gate | bar (registered a priori) | measured (3-seed mean) | margin vs noise floor | verdict |
|---|---|---|---|---|
| **G1** pair-ranking AP | > 0.5586 | **0.6579** (min seed 0.6221) | +0.0993 vs floor 0.0540 | **PASS** |
| **G2** merge precision at matched count | >= 0.7992 | **0.7260** (best seed 0.7410) | -0.0732 | **FAIL** |
| **G3** frag/GT <= 1.799 at purity >= 0.9313 | both | frag **1.746** (PASS, margin 0.053 vs floor 0.022); purity **0.9309** (FAIL by 0.0004, inside the 0.0068 floor) | — | **FAIL** (purity leg indistinguishable) |

**Registered verdict: FAIL.** Pass required all three. The associator ranks candidate pairs
substantially better than the incumbent's implied ranking, and it does *not* convert that into
better merges at the incumbent's operating point.

### 2.3 Why G1 passes and G2 fails (diagnostics, same val slice, no new training)

| quantity | incumbent | main arm (seeds 0/1/2) |
|---|---|---|
| pair-ranking AP, all candidates | 0.5586 | 0.676 / 0.622 / 0.676 |
| pair-ranking AP, same-team candidates only | 0.6668 | 0.732 / 0.685 / 0.731 |
| **raw precision of the top-645 pairs** (labelled ones) | **0.8088** | **0.9020 / 0.8539 / 0.8944** |
| merge precision at 324 merges (0.5x) | 0.8806 | 0.9168 / 0.8977 / 0.9237 |
| merge precision at 483 merges (0.75x) | 0.8410 | 0.8619 / 0.8296 / 0.8325 |
| merge precision at 645 merges (1.0x, the gate) | 0.7381 (**its own ranking under our greedy linkage**) / 0.7492 (its own agglomerative procedure) | 0.7410 / 0.7163 / 0.7206 |

Three things fall out:

1. **The ranking really is better** — top-645 pair precision 0.883 mean vs 0.809, and +0.065 AP even
   on the same-team subset where the incumbent is allowed to play. G1 is not an artifact of the
   incumbent's gates dumping blocked pairs at the bottom.
2. **The gain evaporates under transitive linkage.** `merge_precision` counts *all* pairs implied
   inside a component, so one wrong link inside a 6-tracklet component charges 5+ wrong pairs. The
   model's better pair ranking buys about +0.03 precision at 324 merges, +0.00 at 483 and +0.00 at
   645: the further down the ranking you force it, the more its extra recall is spent on links that
   glue two large components together.
3. **0.011 of the "incumbent 0.7492" is procedure, not ranking.** Running the incumbent's own scores
   through our greedy single-linkage merger gives 0.7381; its mean-recomputing agglomerative
   procedure is worth +0.011. The model beats *that* number by +0.003 (seed 0 +0.029) -- inside the
   0.0247 seed floor, i.e. indistinguishable.

**What the model does with its freedom from the gates (mean over seeds at 645 merges):** 56 of its
645 merges cross a team boundary (8.6%), 1 crosses role, 2 have conflicting jersey reads -- all
structurally impossible for the incumbent. Of the *labelled* cross-team merges it makes, 21 are
correct and 14 wrong (0.594 precision) -- worse than its overall labelled precision (0.883) but far
from noise, which is consistent with W1's finding that the team labels themselves are wrong often
enough to forbid 16.6% of true merges. That is real headroom the incumbent cannot touch; it is just
not yet worth more than what the model gives back elsewhere.

### 2.4 Ablation — metric kinematics alone are NOT competitive

The registered kinematics-only arm (no team/role/jersey/CLIP side features; the model sees only
metric positions, velocities and time) lands at **AP 0.2707 / merge precision 0.3276**, against the
incumbent's 0.5586 / 0.7492 and the full model's 0.6579 / 0.7260. Its 3-seed spread is 0.0116 AP, so
this is not a training accident.

**Honest reading of the v9 novelty premise:** the metric-coordinate signal by itself is roughly
*half* as informative as appearance-with-gates on this distribution. Metric kinematics pay only in
combination -- appearance + jersey + team + metric motion learned jointly is worth +0.099 AP over
the incumbent's appearance-with-gates. The claim "metric coordinates beat image-space association"
is not supported by anything measured here; the supported claim is "learning the combination beats
gating on one signal, at ranking".

---

## 3. Results — splitter costing (57 sequences, 4,709 tracklets, 632,710 audited rows)

`python -m tools.gsr_v9_train --stage split`, laptop CPU. Contaminated = per-tracklet purity < 0.8
over its audited rows: **772 tracklets (16.4%)**. Row-weighted purity before any split: **0.9547**.

| arm | tracklets split | of the 772 contaminated | share of splits that hit a contaminated tracklet | false splits on >= 0.99-pure tracklets | purity after | new fragments | contaminated tracklets after |
|---|---|---|---|---|---|---|---|
| DBSCAN **eps 0.40** | 0 | 0.0% | — | 0 | 0.9547 | +0 | 141 -> 141 |
| DBSCAN **eps 0.30 (shipped default)** | **2** | **0.3%** | 1.000 | 0 | 0.9548 | +3 | 141 -> 141 |
| DBSCAN **eps 0.20** | 597 | 10.0% | 0.129 | 280 (10.2% of clean tracklets) | 0.9593 | **+1,052 (+22.3%)** | 141 -> **183** |
| **per-row GT oracle cut** | 872 | 35.2% | 0.312 | 11 | **0.9871** | +2,642 (+56.1%) | 141 -> **4** |

**The shipped splitter is inert on this stack.** At its shipped `eps 0.30` it splits 2 tracklets in
57 sequences. Its `eps` was tuned for **PRTreID** embeddings; the chain now runs **CLIP**, whose
cosine distances are far larger (W1 §4.3: median 0.29 within identity, 0.61 across), so a 0.30
neighbourhood radius makes every tracklet one dense DBSCAN cluster. Anyone reading "splitter + 
connector" on the current chain should read "connector".

**Even loosened, it does not work.** At `eps 0.20` it finds 10% of the contaminated tracklets, but
87% of its 597 cuts land on tracklets that were not contaminated, it manufactures 1,052 new
fragments (+22%) for the connector to re-close at ~0.74 precision, and the count of *contaminated*
tracklets goes **up** (141 -> 183) because a cut in the wrong place turns one mixed tracklet into two
mixed ones. Net row-weighted purity gain: **+0.0046**.

**The oracle bounds the whole splitter class.** A splitter that knows every row's GT identity and is
allowed to cut wherever it changes (absorbing runs shorter than 5 rows, the same courtesy DBSCAN
gets) still only touches **35.2%** of the contaminated tracklets and removes **71.5%** of the
impurity mass. The other 28.5% is contamination that is *interleaved*, not contiguous -- a tracklet
that flickers between two players every few frames cannot be repaired by cutting it in two. **No
cut-based splitter, however good its appearance model, can recover the last 28.5%.**

### 3.1 Updated ceiling arithmetic

W1's oracles, carried onto the board identity `GS-HOTA = sqrt(DetA x AssA)` at our DetA 39.32:
merge-only oracle AssA 83.56 -> 57.32; split+merge oracle AssA 96.46 -> 61.59. The split leg is
therefore worth **+12.90 AssA** on top of a perfect merger. **Assumption (stated, and optimistic):**
that +12.90 is realised in proportion to the *fraction of impurity mass removed*, measured above.
It is optimistic because it charges nothing for the extra fragments a splitter creates, which a real
merger then has to re-close at its own precision.

| splitter | impurity mass removed | AssA with a **perfect merger** | GS-HOTA | vs target 61.48 |
|---|---|---|---|---|
| none (merge-only oracle) | 0% | 83.56 | 57.32 | -4.16 |
| DBSCAN eps 0.30 (shipped) | 0.2% | 83.59 | **57.33** | -4.15 |
| DBSCAN eps 0.20 | 10.2% | 84.87 | **57.77** | -3.71 |
| **per-row GT oracle cut** | **71.5%** | 92.79 | **60.40** | **-1.08** |
| W1's split+merge oracle (row-perfect partition) | 100% | 96.46 | 61.59 | +0.11 |

With the **realised** merger instead of a perfect one (our shipped AssA 71.67; the trained model did
not beat the incumbent at its operating point, so there is no merge gain to add): eps 0.20 gives
GS-HOTA **53.57** (+0.49 over the shipped 53.08) *before* charging for the +22% fragments, and the
GT-oracle cut would give 56.40.

**The conclusion this session is obliged to state.** W1 registered that the target is reachable only
in the split+merge band. This session measured that band's realistic top: **a splitter that is
perfect on ground truth, paired with a merger that is perfect, reaches ~60.4 GS-HOTA — still short
of 61.48.** The 61.59 figure requires a *row-level* partition correction (splitting interleaved
contamination frame by frame), which is a tracker replacement, not a splitter. On the association
axis alone, at DetA 39.32, the campaign target is **not reachable**. To hit 61.48 you need
DetA **45.23** with a perfect merger, **40.74** with perfect merge + oracle-cut split, or **52.74**
at today's AssA. **DetA is the binding constraint, and W3 should be a detection/coverage session,
not another association session.**

---

## 4. Negatives, limits and hazards

1. **The registered gate failed.** The associator does not ship on this evidence. G1 passed cleanly
   (+0.099 AP, 1.8x the seed floor); G2 missed its bar by 0.073 and does not even separate from the
   incumbent's own ranking under the same linkage; G3's purity leg failed by less than the noise.
2. **Inner-val is a noisy selector.** Seed 1's best inner-val epoch was epoch 3 (of 30) and it is
   the worst seed on the validation slice (val-AP 0.6221 vs 0.676). Five sequences is too small a
   selection set; a larger inner-val (or k-fold over the 45 development sequences) is the obvious
   fix, and it was not attempted here because the split was registered before training.
3. **Merge precision is a pair-level, transitive metric.** A single wrong link inside a large
   component is charged many times. That is the on-record definition
   (`generator.track_relink.merge_precision`) and it is applied identically to both arms, but it
   penalises long chains, and a linker that made the *same* links in a different order could score
   differently. The greedy-vs-agglomerative procedure alone moves the incumbent by 0.011.
4. **No GS-HOTA was measured in this session.** Every number here is on the W1 proxy instruments
   (pair AP, merge precision, fragments per identity, row purity) on train-split sequences. The
   §3.1 arithmetic inherits W1's cross-split additivity assumption *and* adds a linear-interpolation
   assumption of its own. Neither has been validated end to end.
5. **The splitter `eps` sweep is a measurement, not a tuning.** It selects nothing and no arm was
   carried forward; it exists because reporting "the shipped splitter does nothing" without checking
   whether a nearby setting works would have been a weaker negative.
6. **Label caveat carried from W1:** GT-proximity labels within 5 m, so a tracklet shadowing a
   nearby player inherits a wrong identity. Both arms and the incumbent are scored against the same
   labels, so the comparison is fair, but the absolute purity/precision levels are optimistic.
7. **Unlabelled pairs (35%) never entered the loss.** They were kept in the forward pass as
   inter-pair context. At 645 merges the model spends ~135 of them on unlabelled pairs (the
   incumbent spends 138), which are unauditable in either direction.

---

## 5. Files

**Laptop** (`c:\Users\siddh_ygv5bws\football-synthesizer\`)
- `generator/assoc_twix.py` — the metric-space TWiX port (MIT provenance header), self-check green
- `tools/gsr_v9_train.py` — baseline / windows / train / split driver, self-check green
- `tests/test_gsr_v9_assoc.py` — 4 tests, green; ruff clean at 100 cols
- `results/GSR_V9_W2.md` — this document
- `outputs/gsr/v9_assoc/baseline_incumbent.json`, `train_side.json`, `train_kin.json`,
  `splitter_cost.json`
- `outputs/gsr/v9_assoc/ckpts/twixm_{side,kin}_s{0,1,2}.pt` — 6 checkpoints, md5-verified against
  the cluster copies
- `outputs/gsr/v9_assoc/v1/detembed/*.npz` — 57 per-detection CLIP caches mirrored from the cluster
  (needed by the baseline and the splitter costing)

**Cluster** (`siddhanth23519@192.168.3.19`)
- `~/work/v9/run_v9w2.sh`, `~/work/v9/v9w2.log`
- `~/football-synthesizer/{generator/assoc_twix.py, tools/gsr_v9_train.py}` (md5-identical)
- `~/football-synthesizer/outputs/gsr/v9_assoc/{train_side.json, train_kin.json, ckpts/}`

DEV-20, TEST-38, test-49 and challenge were not read, scored or extracted at any point in W2.

