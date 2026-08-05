# The trained jersey head as a second OCR voter — REJECTED, and it fails harder than Qwen

Session 5B. `results/OCR_DENSIFICATION.md` §7 killed Qwen2-VL-2B as a second voter on one number:
**0.265 precision** on the crops where the shipped chain abstains, against a pre-declared floor of
**0.80**. The open question it left was whether a *football-trained* head would differ. Cluster
session 3 shipped one — the 42-way jersey head inside `outputs/gsr/clip_ckpt/epoch8.pt`, trained on
the same GSR crops our pipeline reads — so this act runs the identical trial design on it.

**Headline: it is worse than Qwen, by a lot, and the reason is diagnosable.** On DEV-20 crops where
the chain abstains the head reads at **0.151 precision** (Qwen 0.265, floor 0.80). On crops where
the chain *does* read, the chain is **0.703** precise per crop and the head **0.053**, with
**892 chain-only-right against 2 head-only-right**. The head has no regime in which it is useful.

**The mechanism: the head memorised identity -> number, it did not learn to read digits.** On its
own training split it is 0.863 precise per crop; on held-out valid sequences, built by the same crop
builder from the same dataset, it is **0.125** — a 6.9x collapse across a boundary that changes only
*which players* are in frame. Its top predictions on valid are the popular GSR-train numbers
(10 x1,127, 24 x504, 4 x339) and its precision (0.125) barely clears the "always guess the most
common number in this subset" baseline (0.102). **The second-voter question is now closed for
trained-head candidates as well as for VLMs**, and the finding retires a line in the roadmap: an
attribute head supervised on 1,343 identities is an identity classifier wearing a jersey label.

Step 3 (wiring, DEV-20 GS-HOTA) was **not run**, per the task's stop rule.

---

## 0. What was built, and the one thing that had to be reconstructed

Everything here is laptop-local; nothing was fetched from the cluster.

| piece | file | note |
|---|---|---|
| GSR GT crop builder | `tools/gsr_crops.py` | clean-room reimplementation of cluster session 2's `~/work/build_crops.py` |
| jersey-head reader | `tools/clip_embedder.py` (`jersey_probs`, `_forward`) | the checkpoint already on disk carries the head; `embed`'s contract is unchanged |
| the trial | `tools/jersey_head_trial.py` | head pass, chain pass, class-order recovery, every table |
| raw numbers | `results/gsr_benchmark/jersey_head_trial.json` | all three partitions |
| frozen class order | `results/jersey_head_vocab.json` | see §0.2 |

### 0.1 The crop builder reproduces the cluster's crop set exactly

The law (drop `w` or `h <= 30 px`; uniformly subsample each tracklet to 15 keeping first and last;
drop ids with `< 4` crops; downsize above 256x128; `min_vis 0.3` inapplicable — GSR carries no
visibility field) is asserted against the published counts in `tools/gsr_crops._demo()`:

| | ours (laptop) | cluster session 2 §2 |
|---|---|---|
| train crops | **20,067** | 20,067 |
| train identities | **1,343** | 1,343 |
| videos | **57** | 57 |
| roles | player 17,099 / referee 1,702 / GK 1,148 / other 60 / ball 58 | identical |
| teams | left 9,080 / right 9,167 / None 1,820 | identical |
| crops carrying a jersey | 13,994 (69.7%) | identical |

Exact on every published figure, including the 58 ball crops that a person-only category filter
silently drops (the first attempt read 20,009 / 1,337 for exactly that reason). Valid plans and
materialises **19,993** crops over **1,340** identities and 58 videos, nothing dropped.

### 0.2 The class order was not on disk and had to be recovered — on train, self-verifying

The checkpoint stores `jersey.weight (42, 256)` and `dims.n_jersey = 42` but no label mapping (that
lives in `~/work/crops_unified.json`). GSR train has exactly **41 distinct jersey numbers**, so
41 + 1 abstention = 42 confirms the vocabulary; only the order was unknown.

Recovered by optimal assignment (`scipy.linear_sum_assignment`) of predicted class against GT label
over all 20,067 **train** crops — the split the head was fitted on, so no evaluation data is spent.
The natural construction order (numbers ascending, `<none>` appended last) **is an optimal
assignment**, tying the free assignment to machine precision in both candidate input spaces, and
`recover_vocab` raises if it ever stops tying:

| input to the head | free-assignment train accuracy | natural-order accuracy |
|---|---|---|
| **L2-normalised shared embedding** (frozen) | **0.7137** | **0.7137** |
| pre-norm BN output | 0.6876 | 0.6876 |

Two rejected orders confirm the identification is not vacuous: `<none>` first scores **0.0051**, and
string-sorted numbers score **0.2815**. The 9 classes where the free assignment picked a different
permutation are rare classes the head never predicts correctly under *any* order (they contribute 0
to both totals) — the tie is degeneracy on dead classes, not ambiguity about the mapping.

---

## 1. Head evaluation — per-crop, against the split it was trained on

Grading: the GT jersey is **constant within a GSR track** (checked: 0 of 1,343 train and 0 of 1,340
valid tracks carry two values), so each crop inherits its track's number. `<none>` = the annotation
carries no jersey.

| quantity | TRAIN (fit set) | **VALID-58** | DEV-20 | TEST-38 |
|---|---|---|---|---|
| crops | 20,067 | 19,993 | 7,055 | 12,938 |
| crops with a GT number | 13,994 | 14,490 | 4,965 | 9,525 |
| ...GT number outside the head's 41-class vocab | 0 | **1,410 (9.7%)** | 525 | 885 |
| accuracy, all crops | 0.7137 | 0.2884 | 0.3093 | 0.2769 |
| **accuracy on numbered crops** | **0.6201** | **0.0251** | 0.0262 | 0.0246 |
| accuracy on no-number crops | 0.9295 | 0.9815 | 0.9818 | 0.9812 |
| emit rate on numbered crops | 0.7187 | 0.2008 | 0.1887 | 0.2070 |
| **read precision where it emits (numbered GT)** | **0.8627** | **0.1251** | 0.1387 | 0.1187 |
| ...same-subset "always the modal number" baseline | 0.0947 | **0.1018** | 0.1163 | — |
| ...same-subset independent-marginals baseline | 0.0455 | 0.0434 | 0.0535 | — |
| uniform over 41 classes | — | 0.0244 | — | — |

**The gap between column 1 and column 2 is the whole result.** Same crop builder, same dataset, same
preprocessing; the only difference is which sequences. Precision falls 0.863 -> 0.125 and accuracy
on numbered crops falls 0.620 -> 0.025. **0.125 is 1.23x the trivial "guess the most common number"
baseline of 0.102** — the head retains a prior over which numbers are common in GSR train and almost
nothing else.

### 1.1 Abstention quality — the design bet, measured

`CLUSTER_SESSION2.md` §3 made abstention an explicit class on purpose. It is not a useful abstention
signal on unseen sequences:

| | TRAIN | VALID-58 | DEV-20 |
|---|---|---|---|
| no-number class **recall** | 0.9295 | **0.9815** | 0.9818 |
| no-number class **precision** | 0.5892 | **0.3180** | 0.3375 |
| base rate of no-number crops | 0.3026 | **0.2752** | 0.2962 |
| precision / base rate (lift) | 1.95x | **1.16x** | 1.14x |

Recall 0.98 looks strong and means nothing: the head predicts `<none>` on **80% of all valid crops**,
so it catches almost every true abstention by abstaining almost always. Its precision, 0.318 against
a 0.275 base rate, is a **1.16x lift** — near-zero information. On its own training split the same
lift is 1.95x, so this too is memorised rather than learned.

### 1.2 Digit-level confusion structure

Wrong emitted reads on valid-58 (n = 2,545): **no digit shared with the true number 1,631 (64.1%)**,
first digit right 728 (28.6%), last digit right 271 (10.6%), length mismatch 934 (36.7%), of which
truncation (read is a prefix of the truth) 108 (4.2%).

Top confusions: `17->10` x168, `14->10` x144, `8->24` x137, `42->10` x125, `44->4` x101,
`11->10` x100, `34->10` x97, `19->10` x85. Top *predictions*: 10 (1,127), 24 (504), 4 (339), 8 (213),
3 (172).

**This is not OCR error structure.** A reader that sees glyphs and misreads them shares digits with
the truth (GSR's PARSeq errors are digit substitutions: `42->62`, `11->17`, `13->33` per
`GSR_V4.md` §5). Two thirds of this head's errors share **no digit at all** and the errors funnel
into a handful of frequent train numbers. It is emitting a prior, not a read. `42->10` x125 is the
tell: `42` is not in the head's vocabulary at all, so 9.7% of valid numbered crops are unreachable
by construction — but that ceiling is a rounding error next to a 0.125 precision.

### 1.3 The head is never confident

Max posterior on crops where it emits a number: valid p50 **0.222**, p90 0.412, p99 0.622, max
0.782. Only **4.6%** of emitted valid reads reach 0.50 and **0.27%** reach 0.70. Train: p50 0.367,
max 0.904. **No confidence floor rescues precision** — the floor sweeps in §2 empty out before
precision moves.

---

## 2. The Qwen-trial design, rerun — the kill criterion

Same design as `OCR_DENSIFICATION.md` §7: on crops the shipped chain **abstains** on, does the
candidate read at >= 0.80 precision? Both readers run over the **same 19,993 GT crops**. Precision
is over crops whose track carries a GT number (`OCR_DENSIFICATION.md` §3's definition, the one the
0.80 floor was declared against); the stricter all-tracks figure is reported beside it.

Three definitions of "abstains" are measured: the chain produced no read at all (legibility gate,
no pose torso, or no PARSeq output), and the two shipped aggregation rules' per-crop confidence
floors (`0.85-rule` = `min_crop_conf 0.99`, `0.80-rule` = `min_crop_conf 0.90`).

**DEV-20 — the gate.**

| chain-abstains definition | crops | head emits | rate | **precision (numbered GT)** | precision (all tracks) | bar |
|---|---|---|---|---|---|---|
| hard abstain (no read at all) | 5,659 | 415 | 0.073 | **0.1508** (57/378) | 0.1373 (415) | 0.80 — **FAIL** |
| abstain at the 0.85-rule floor 0.99 | 6,232 | 590 | 0.095 | **0.1214** (552) | 0.1136 (590) | **FAIL** |
| abstain at the 0.80-rule floor 0.90 | 6,012 | 518 | 0.086 | **0.1229** (480) | 0.1139 (518) | **FAIL** |

**TEST-38 — confirmation, no knob selected on it.**

| chain-abstains definition | crops | head emits | **precision (numbered GT)** |
|---|---|---|---|
| hard abstain | 10,286 | 784 | **0.1319** (728) |
| 0.85-rule floor | 11,364 | 1,213 | **0.1059** (1,152) |
| 0.80-rule floor | 10,963 | 1,049 | **0.1121** (990) |

Head-confidence floors on the added reads are empty, not better: at `>= 0.50` DEV keeps 14 reads at
**0.000** precision; at `>= 0.70`, zero reads.

**Against the precedent: Qwen 0.265, this head 0.151. The kill criterion is missed by 0.65, and the
football-trained candidate is 0.11 *worse* than the generic VLM it was meant to beat.**

### 2.1 Where the chain *does* read — the head loses there too

The tail that made Qwen interesting (`OCR_DENSIFICATION.md` §7: Qwen beat PARSeq per-crop, 0.800 vs
0.700, McNemar p = 0.031) has no analogue here.

| on crops the chain reads, numbered GT | DEV-20 | TEST-38 |
|---|---|---|
| crops | 1,370 | 2,595 |
| **chain (PARSeq) per-crop precision** | **0.7029** | **0.7252** |
| **head per-crop precision** | **0.0533** | **0.0532** |
| head emit rate on those crops | 0.408 | 0.479 |
| agreement | 0.064 | 0.066 |
| **chain-only-right / head-only-right** | **892 / 2** | **1,756 / 12** |

*Harness check:* the chain's per-crop precision here (0.703 DEV / 0.718 valid-58) reproduces the
Qwen trial's independently-measured chain per-crop precision of **0.700** on its own 100-crop
sample. The grader and the crop set are behaving.

### 2.2 The two most generous readings of the head, so the verdict is not an argmax artefact

Both are strictly more favourable than anything the pipeline could deploy.

| variant, DEV-20 | emit rate | n | precision |
|---|---|---|---|
| per-crop argmax | 0.138 | 937 | 0.1387 |
| per-crop + **oracle roster mask** (only numbers actually present in that sequence) | 0.116 | 794 | **0.2040** |
| track-level majority vote (473 tracks) | 0.414 | 185 | 0.1243 |
| track-level majority vote + oracle roster mask | 0.374 | 170 | **0.2000** |
| track-level argmax of the mean posterior | 0.072 | 33 | 0.1212 |

TEST-38 is the same picture (0.1187 / 0.1734 / 0.1065 / 0.1463 / 0.0972). An **oracle** roster —
better than the self-roster the pipeline builds — quadruples nothing: 0.20 against a bar of 0.80.
Track-level aggregation does not help either, which rules out "noisy per crop but consistent per
track". **There is no read of this head that survives.**

---

## 3. Why, and what it means for the roadmap

The jersey head's positive supervision came from **13,994 crops of 1,343 GSR-train identities**,
each identity carrying exactly one number, and `CLUSTER_SESSION3.md` §2 deliberately withheld
sn-reid's numbers (action-level labels -> hallucination) while keeping its 78,895 "never visible"
letters as abstention. So the head's number classes saw **1,343 player-number pairs**, and the
cheapest function that fits them is *recognise the player, recall his number* — appearance, kit,
posture, video-specific colour — not *read the glyphs on the shirt*. Session 2 already measured that
the same corpus overfits the identity head (peak epoch 10, decay to 49.32); the jersey head's
version of that failure is total because its label is a **property of the identity**, so
memorisation is not merely available, it is the direct route.

Three consequences worth carrying forward:

1. **The second-voter question is closed for now.** Two candidates from opposite directions — a
   generic VLM with real OCR ability (0.265) and a football-trained head with none (0.151) — both
   fail the same 0.80 floor on the same crops. `EVIDENCE_DENSITY_LAW.md` §4.2's warning holds:
   reads that are 85% wrong subtract from density-weighted precision. Anything proposed next should
   have to clear 0.80 on *abstain crops* before it gets a wiring budget.
2. **A jersey head trained this way should not be trained again.** The fix is not more epochs or a
   loss weight; it is a corpus where the number is not a function of the identity — i.e. the same
   `SOCCERNET_REPO_SWEEP.md` §15 item-4 sn-reid corpus that fixed the identity embedding, but with
   its jersey labels usable, which they are not per-crop. A digit-level OCR objective on crops with
   *per-crop* glyph labels (SoccerNet jersey-2023, already on disk at `data/soccernet/jersey-2023`)
   is the only version of this that could work, and it is a different build.
3. **The checkpoint is not damaged by this.** `clip-s4-001` wired the same checkpoint's *embedding*
   into the connector and it is in the shipped v5 recipe. This result is about the jersey head only;
   nothing here touches the embedding's measured value.

## 4. Negatives and limits

1. **The class order is reconstructed, not read from the cluster's manifest.** It is fixed on train
   (0.7137, tying the free optimal assignment, with two alternative orders at 0.005 and 0.282), and
   `recover_vocab` raises if the tie ever breaks — but if the true order differed on the 9
   degenerate classes, the valid numbers would move by at most a rounding error, because those
   classes are never correctly predicted under any order. A wrong mapping cannot explain 0.863 on
   train and 0.125 on valid *with the same mapping*.
2. **Per-crop GT is a track-level label.** A crop where the number is genuinely invisible still
   carries its track's number, so *every* per-crop precision here — chain and head alike — is a
   pessimistic bound. It biases both readers identically, so the comparison is unaffected.
3. **9.7% of valid numbered crops carry a number outside the head's 41-class vocabulary**
   (2, 37, 42, 43, 47, 59, 99). Even scoring those as free passes, precision could not exceed ~0.22.
4. **Step 3 was not run** — no votes were wired, no DEV-20 GS-HOTA was measured, no TEST-38
   verification was spent. That is the task's own stop rule at a failed step 2, and it is the right
   call: the pre-declared justification for a TEST-38 run (DEV GS-HOTA +0.3) cannot be reached from
   0.15-precision evidence.
5. **This is the GT-crop distribution, not the detector-crop distribution the pipeline runs on.**
   GT crops are cleaner (they are the annotator's boxes), so if anything this is the *easier*
   setting for both readers.
6. **The head was evaluated exactly as trained** (224x224 square, CLIP normalisation, the shared
   L2-normalised embedding). The pre-norm input was measured too and is worse (0.6876 vs 0.7137 on
   train). No temperature, calibration or fine-tune was attempted — the failure is not a calibration
   failure, it is a 0.05-precision-where-the-chain-reads failure.

## 5. Reproduce

```
python -m tools.gsr_crops                       # self-check: reproduces 20,067 / 1,343 / 57
python -m tools.gsr_crops --split train         # cut GT crops (CPU, ~20 min parallel)
python -m tools.gsr_crops --split validation
python -m tools.jersey_head_trial --head train        # GPU ~7 min
python -m tools.jersey_head_trial --recover           # freezes results/jersey_head_vocab.json
python -m tools.jersey_head_trial --head validation   # GPU ~7 min
python -m tools.jersey_head_trial --chain validation  # GPU ~9 min (legibility + pose + PARSeq)
python -m tools.jersey_head_trial --report            # CPU, writes the results JSON
```

Self-checks: `python -m tools.gsr_crops` (crop law + published counts),
`python -m tools.jersey_head_trial --demo` (grader), `python -m tools.clip_embedder` (embedding
contract unchanged by the `jersey_probs` addition).

**GPU etiquette note:** the laptop GPU was occupied by an unrelated foreground application (99%
util, 3.1-3.6 GB of 4 GB) for roughly 90 minutes mid-session. No job of ours competed with it; the
CPU fallback was ~7 crops/s against 50 crops/s on GPU and was abandoned once the GPU freed.

## 6. Files

- `tools/gsr_crops.py` — the crop law, label-only planner + cutter, `_demo` asserting the published
  counts.
- `tools/clip_embedder.py` — `jersey_probs`, `_forward` (one tower pass, both head inputs). `embed`
  and its 256-d L2-normalised contract are unchanged; `_demo` still passes.
- `tools/jersey_head_trial.py` — `run_head`, `run_chain`, `recover_vocab`, `head_eval`,
  `abstain_trial`, `generous_variants`, `_demo`.
- `results/jersey_head_vocab.json` — the frozen class order + its train-set verification.
- `results/gsr_benchmark/jersey_head_trial.json` — every number above, three partitions.
- Artifacts (gitignored): `outputs/gsr/gt_crops/{train,validation}/` (40,060 JPEGs, 205 MB),
  `outputs/gsr/jersey_head/{crops,head,chain}_*.{parquet,npz}`.
- Claim: `clip-s5-002`.
