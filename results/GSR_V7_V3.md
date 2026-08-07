# v7 session V3 — the Dirichlet evidential jersey head (pre-registered)

Server `a100server1` GPU 1 + laptop CPU. Control: `results/gsr_benchmark/gsr_v7_control_dev.json`
(DEV-20 GS-HOTA **51.808280888021784**, pinned stack, `results/GSR_V7_V0.md`).

---

## 1. REGISTRATION (written before any arm was trained or scored)

### 1.1 Why this session exists, in the campaign's own measurements

`results/GSR_V7_V1.md` §2.7 settled what binds the solver, and it is not calibration:

> 610 of 870 tracklets are named and the remaining 260 are not held back by calibration — they are
> held back by having no admissible slot.

Slots come from **reads**, because the GSR roster is `roster_self` — built from the sequence's own
reads. So the emit ceiling *is* the admissible set. And the emit ceiling has a name:
`results/GSR_S3_READER.md` §7.2 records that **12,814 of 67,224 DEV-20 crops (19.1%) pass the
shipped 2024 ResNet34 legibility gate** and everything downstream reads only those. Every density
number in the campaign is conditioned on a frozen 2024 classifier that nobody has retrained
(`CLUSTER_SESSION_S2.md` §4: arm 1 was never run).

This session attacks both at once with one model: a **Dirichlet evidential head** in the Grad
CVPRW2025 style (reimplemented — no public code) that learns abstention *inside* the recognizer, so
the external gate becomes a swept knob rather than a wall.

### 1.2 The design, and the two prior results it is built to respect

**GoMatching decoupling — the trunk is frozen.** The head sits on the S2 arm-4t PARSeq
(`~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt`, the shipped v6 reader). Its input per crop
is `[mean-pool(encoder memory) | max-pool(encoder memory) | decoder positional softmax at positions
0..2]` = `2 * 384 + 33 = 801` dims. The encoder half carries "is there a glyph on this shirt at
all"; the decoder half is arm-4t's actual read, so the head **recalibrates a 0.8344-precision
reader** rather than relearning OCR — and a head that cannot beat its own input on numbers is a
measurable failure, not a hidden one. Nothing but the head trains (`generator/evidential_jersey.py`
only ever sees cached features; the freeze is structural).

**JERSEY_HEAD_VOTER — the identity shortcut is designed against, then tested for.** The 5B head
collapsed 0.863 -> 0.125 across a player boundary because its label was a *property of the
identity*. Here: the trunk is frozen and is already a digit reader; supervision is per-crop
glyph/abstention only; and every train-vs-heldout number is split by
`player_disjoint_split` on a player key (`v3|<game>|<number>`, `j2023|<tracklet>`,
`gsr|<seq>|<track>`), so a memorisation collapse is visible by construction.

**Output.** `alpha = softplus(logits) + 1` over `{no-number} u {1..99}` (`NUM_CLASSES = 100`, the
same layout the whole chain already consumes). `S = sum(alpha)`; the Dirichlet mean `alpha / S`
replaces the folded PARSeq product; `u = 100 / S` is the total-evidence uncertainty. Loss = the
Bayes-risk MSE evidential loss + a uniform-Dirichlet KL annealed `0 -> kl_max` over 10 epochs.

### 1.3 Supervision

| half | source | rule | rows |
|---|---|---|---|
| numbers (classes 1..99) | S2 `plus_v3_torso` corpus | legibility > 0.7, torso RoI | 79,109 |
| **no-number (class 0)** | v3 digit crops that FAIL the 0.7 filter | the asymmetric-evidence rule of `CLUSTER_SESSION_S2.md` §3.2 — the *digit* label is unsound on those crops, the *illegible* marker is not, so take the negative evidence | 49,760 candidates |
| no-number (class 0) | GSR-train crops failing the same filter | the only **in-domain** negatives | 9,819 candidates |

The positive half is the S2 corpus reconstructed row-for-row (deterministic seeds), verified before
use against `corpus_stats_torso.json`: **79,109 rows, v3 40,823 / jersey2023 34,398 / gsr_train
3,888 — exact on every figure**. DEV-20 comes from GSR *validation* and enters no corpus.

*Not taken, recorded as not taken:* sn-reid letter crops (the S2 brief's "if cheap" item — it was
not resolvable from verifiable assets in S2 either, `CLUSTER_SESSION_S2.md` §4) and jersey-2023's
own illegible crops (~150k more negatives; the corpus is already balanced without them).

### 1.4 The gate ladder — pre-declared, in order, each blocking the next

**Rung 0 (blocking, before any A100 time): the Dirichlet loss unit-check on a toy split.**
`python -m generator.evidential_jersey` must assert (a) the evidential loss at least halves,
(b) mean uncertainty is higher on the illegible population than the legible one, (c) the abstention
ROC is sane (AUC > 0.80 on `p_none`, > 0.70 on `u`), (d) the head still reads (> 0.90 accuracy on
legible rows), (e) the Dirichlet fusion is additive, its uncertainty filter bites, and a
zero-evidence tracklet abstains.

**Rung 1 — component, on DEV-20 GT crops.** Per-crop precision **at matched emit** must be `>=` the
v6 reader's **0.8344 at emit 0.2761** (the S2 arm-4t bar; the incumbent bar is 0.7022 at 0.2759).
Reported alongside, not gated: the **invisible-class AUC** (the Grad claim is that the big gains
live on invisible detection) and the **player-disjoint holdout** precision against train.

**Rung 2 — tracklet, on DEV-20.**
1. *The Koshkina voting control.* Uncertainty-filtered Bayesian fusion (`fuse_tracklet`,
   `tau_u`/`tau_p_none` swept on DEV only) must beat the **existing** `percrop_votes` machinery run
   on the **same evidential rows**. If it does not, ship voting-on-evidential-reads and say so.
2. *d-at-floor.* `d >= 0.35` AT read precision `>= 0.92`. Arm A ships **d 0.3139 at 0.9256**;
   **both** axes must dominate, not one.

**Rung 3 — DEV-20 end-to-end.** `>= +1.0` GS-HOTA against the V0 control (**51.8083**) AND `>= 12/20`
sequences helped, through the v6 chain (`positions_gate_v6det`, EIoU e0.3/r1/w0.5/a0.30, CLIP,
tau 0.450, `team_src="free"`, `roster="self"`, the official scorer) with **only the votes swapped**
— the evidential arm lives in `_v7e`-suffixed variant directories so nothing on record can mix.

**Reported either way, because V1 predicts this is where any GS-HOTA comes from:** the
**admissible-set effect** — self-roster slots per sequence, and merged tracklets the solver names,
against the incumbent's **610 of 870** at **13.2 identities per sequence**.

### 1.5 Guards, and what a failure looks like

1. Selection at rung 3 is on end-to-end DEV-20 GS-HOTA only. Component metrics are reported, never
   selecting.
2. The `results/GSR_V5.md` §2.1 concentration guard applies: a gain whose two best sequences carry
   `>= 80%` of the net is demoted.
3. `results/GSR_S3_READER.md` §6 is the standing warning: the density-maximising operating point was
   **worse** end-to-end than the precision-holding one (41.80 vs 42.74). Removing the legibility gate
   is exactly a density move, so `min_legibility` is swept **including the shipped 0.5**, and the
   incumbent gate is a live arm, not a strawman.
4. The evidential percrop pass runs SERVER-side (the laptop has 30 GB free, `GSR_V7_V0.md` §5). The
   server does not hold the DEV-20 `positions_gate_v6det`; those 20 parquets are copied there and
   the resulting crop set is **verified against the on-record laptop parquet** (paired
   `(track_id, frame)` keys and legibility to 5 decimals, the `GSR_S3_READER.md` §2 check) before
   any number is believed. A failure of that check is reported, not worked around.
5. FAIL response, pre-declared: the v6 reader ships unchanged, the evidential head is recorded as a
   measured negative with its component numbers, and no `_v7e` artifact enters any freeze.
6. No TEST-38 read, no test-49 read, no commit. `METRICS_VERSION` moves only if a shipped metric
   changes (it does not: the reader is a pipeline stage).

### 1.6 Files this registration binds

- Head + loss + fusion + split: `generator/evidential_jersey.py` (new).
- Chain plumbing (all default-preserving): `tools/koshkina_str_sidecar.py` (`--edl-head`),
  `generator/jersey_id.py` (`edl_head`, `leg_thresh` reach the sidecar; `crop_reads` returns
  `alpha`), `eval/gsr_jersey.py` (`--edl-head`, `--leg-thresh`; `alpha`/`u` columns persisted),
  `tools/ocr_density.py` (`crop_alpha`; `max_u` / `max_p_none` / `fuse` rule keys).
- Server: `~/work/edl/{edl_corpus,edl_feats,edl_train}.py`, weights `~/data/edl/*.pt` (+ md5).
- Raw: `results/gsr_benchmark/gsr_v7_v3_*.json`.

---

## 2. RESULTS — VERDICT: **rungs 0/1/2 PASS, rung 3 FAILS by one sequence. The v6 reader ships
## unchanged; nothing enters a freeze.**

The best evidential arm scores **DEV-20 GS-HOTA 53.0645, +1.2562 over the pinned control, on 11 of
20 sequences** against a gate of `>= +1.0` **AND** `>= 12/20`. The arm with 13/20 breadth is
+0.8873. No single point clears both halves, so the pre-declared FAIL response applies.

Three findings are worth more than the leaderboard number, and two of them are negatives:

1. **At matched read density the evidential reader converts precision into GS-HOTA at a steep rate.**
   Incumbent `d 0.4114 @ 0.9365` -> evidential `d 0.4104 @ 0.9438`: **+0.0073 precision, +1.2562
   GS-HOTA**, carried by DetA +1.83.
2. **The Grad claim — "the big gains live on invisible detection" — does not reproduce.** The
   learned no-number class scores **0.6200** invisible-class AUC on DEV-20, *below* the frozen
   trunk's own 0.6429 and well below the 2024 legibility classifier it was meant to replace
   (**0.7038**). The total-evidence uncertainty is 0.5553, barely above chance.
3. **But the uncertainty channel still pays end-to-end, in isolation**: two arms differing in
   `max_u` alone (0.60 vs off) differ by **+0.3598 GS-HOTA**. `u` is a bad "is a number visible"
   detector and a useful "should this crop vote" weight.

And a fourth, which retires the session's own premise: **removing the 19.1% legibility gate is not
what won.** Every arm above +0.7 keeps it; the ungated arms score -0.37 and +0.03.

### 2.0 Rung 0 — the Dirichlet unit-check: PASS (before any A100 time)

`python -m generator.evidential_jersey`, CPU, seconds. Toy: 6 classes, class 0 = "no number" with
**no prototype direction at all** (an illegible crop is not a different glyph, it is the absence of
one), 3,000 train / 3,000 test rows from the same fixed glyph set.

| assertion | measured |
|---|---|
| the evidential loss at least halves | **0.9177 -> 0.1150** |
| mean uncertainty higher on illegible than legible | **0.3995 vs 0.2193** |
| abstention ROC sane (`p_none` AUC > 0.80, `u` AUC > 0.70) | **0.9962 / 0.7509** |
| the head still reads (> 0.90 on legible rows) | **0.9972** |
| fusion is additive, its `u` filter bites, an empty tracklet abstains | asserted |
| `KL(Dir(1)||Dir(1)) = 0`, `u(alpha=1) = 1`, tied-score AUC = 0.5 | asserted |

**Two real bugs the unit-check caught before they cost GPU time**, both recorded because the check
exists to catch exactly this class of thing:
1. `fit_head` built a 100-class head for a 6-class toy, so 94 dead classes contributed 94 to `S` and
   the KL term swamped the fit — the loss plateaued and accuracy sat at 0.001. `n_classes` is now a
   parameter.
2. The toy's test split regenerated its own prototypes, i.e. train and test had **different
   glyphs**. Every "collapse" before that fix was the harness, not the model.

**And one finding that transferred straight to the real run:** at `kl_max = 1.0` (Sensoy's value,
fitted on 10-class MNIST) the 100-class head collapses to `alpha = 1` everywhere. Measured on the
real corpus: `u = 0.9996`, invisible AUC **0.495 = chance**, precision 0.12. The KL weight is a
swept knob in this codebase, not a constant, and `kl_max` >= 0.05 is unusable at K = 100.

### 2.1 The corpus, and a leak that had to be found before the numbers meant anything

Manifest: **130,454 rows / 10,150 players / 90 classes**, of which **51,345 (39.4%) no-number**
(v3 42,830 + GSR-train 8,515 after an 86.2% torso-RoI survival on 59,579 candidates). Positive half
reconstructed exactly (§1.3).

Frozen-feature caching, the GoMatching decoupling made concrete: **130,454 crops in 129 s** and
DEV-20's 7,055 in 7 s, one A100 pass, after which every head config trains in ~40 s. Total trunk
GPU time for the whole session's head work: **136 s**.

**The leak: arm-4t was fine-tuned on this corpus.** On the player-disjoint corpus holdout the
*trunk* reads **0.9930 at 100% emit** and **1.0000 at emit 0.276** — it has memorised the rows. So
**the corpus holdout cannot measure number accuracy for anything built on arm-4t**, and every
number-class figure in this session is quoted on DEV-20 (GSR validation, unseen by both). This is
recorded rather than quietly dropped because a "0.997 holdout precision" line would otherwise read
as a triumph; it is a measurement of the trunk's memory.

What the corpus holdout *can* still say is that the head adds no identity shortcut: holdout
precision **exceeds** train precision in every config measured (free head: 0.377 holdout vs 0.340
train; gate head: 0.9969 vs 0.9967). The `JERSEY_HEAD_VOTER` 5B signature — 0.863 train collapsing
to 0.125 across a player boundary — **does not appear**. Frozen trunk + per-crop glyph supervision
was the right guard.

### 2.2 The first head design FAILED, and the failure is the interesting part

The pre-registered design (§1.2) was a free 100-way MLP on the pooled features. On DEV-20:

| arm | emit 0.20 | emit 0.2761 (the bar) | max emit |
|---|---|---|---|
| trunk arm-4t + shipped legibility gate (**the bar**) | 0.9718 | **0.8344** (1,371 reads) | 0.2761 |
| free evidential head | **0.4935** | unreachable | 0.2344 @ 0.4210 |

**The head is half its own trunk's precision at matched emit.** The diagnosis is structural, not a
training failure: PARSeq decodes a number autoregressively as `d0 * 10 + d1`, and an MLP asked to
reproduce that composition from a pooled 801-d vector smooths a sharp, combinatorial mapping. The
positional softmaxes are *in* the feature and the head still cannot match them. A free head on a
frozen reader **inherits the reader's ceiling and pays a smoothing tax to get there.**

**The fix, and what it says about the decoupling idea.** `EvidentialGateHead` emits exactly two
numbers per crop — total evidence `e` and no-number share `w` — and lays them over the trunk's own
renormalised ranking:

```
alpha = 1 + e * [ w , (1 - w) * trunk_numbers ]
```

The number argmax is then the trunk's **by construction** (asserted in `_demo`), `p_none = w` is a
learned calibrated abstention, and `u = 100 / (100 + e)` is the evidential channel. Abstention is
still learned inside the model; it is learned in the two dimensions the frozen reader does not
already supply. This is a deviation from the registered architecture, taken after the registered
one was measured and failed, and it is reported as a deviation.

### 2.3 Rung 1 — component gate on DEV-20 GT crops: **PASS**

Harness check first, because everything else rests on it: the trunk arm is recomputed here from the
**cached features' last 33 dims** (which *are* the positional softmaxes), and it reproduces the S2
bar exactly — **1,371 reads, emit 0.2761, precision 0.8344**, against `CLUSTER_SESSION_S2.md` §3.5's
`0.2761 / 0.8344`. Same crop set, same gate, same decode; the feature-derived read *is* the S2
harness.

7,055 DEV-20 GT crops, 4,965 numbered, 5,879 with a pose torso RoI (all 5,879 featurised).

| arm | emit 0.05 | emit 0.10 | emit 0.20 | **emit 0.2761** | max emit |
|---|---|---|---|---|---|
| incumbent Koshkina reader (S2 bar) | — | — | — | **0.7022** | 0.2759 |
| **trunk arm-4t + legibility gate (the bar)** | 0.9839 | 0.9798 | 0.9718 | **0.8344** | 0.2761 |
| trunk arm-4t, NO gate (stronger control) | 0.9677 | 0.9738 | 0.9617 | **0.8607** | 0.8550 @ 0.3404 |
| **gate head `kl 0.01 / none 0.3`, NO gate** | 0.9919 | 0.9940 | 0.9617 | **0.8738** | 0.3349 @ 0.7769 |
| gate head, + legibility gate | 0.9919 | 0.9919 | 0.9617 | — | 0.2570 @ 0.8824 |
| gate head `kl 0.01 / none 0.5`, NO gate | 0.9919 | 0.9899 | 0.9607 | — | 0.2638 @ 0.8710 |
| gate head `kl 0.0 / none 0.5`, NO gate | 0.9960 | 0.9940 | 0.9547 | — | 0.2274 @ 0.9238 |
| free head (§2.2), NO gate | 0.9839 | 0.9456 | 0.4935 | — | 0.2344 @ 0.4210 |

| criterion | required | measured | |
|---|---|---|---|
| per-crop precision at matched emit 0.2761 | >= 0.8344 | **0.8738** | **PASS** (+0.0394) |
| ...against the stronger no-gate trunk control | (reported) | 0.8738 vs **0.8607** | +0.0131 |
| player-disjoint holdout vs train | reported | 0.9969 vs 0.9967 (no collapse) | — |

**Head selection was made on the corpus holdout, not on this table**, by a criterion the design
forces: rung 2 sweeps `tau_u`, so a head whose `u` is degenerate is disqualified. At `kl_max = 0`
the evidence saturates at its ceiling and `u` is constant (0.1667, holdout AUC 0.18-0.22, *inverted*);
among the two live-`u` configs, `none_share 0.3` dominates `none_share 0.5` on every corpus-holdout
column (precision 0.9949 vs 0.9896, emit 0.9815 vs 0.9512, `u`-AUC 0.7329 vs 0.5156). `edlg_k0.01_n0.3`
is therefore selected. **Honest caveat:** all four DEV-20 rows above were computed before that
selection was written down, so the selection is *consistent with* a corpus-only criterion rather
than blind to DEV — the four candidates span 0.9547-0.9617 at emit 0.20, so the exposure is small,
but it is exposure and it is recorded.

### 2.4 The Grad claim does not reproduce: invisible detection is where the head is WEAKEST

The pre-registered expectation, from the research: *"the big gains live on invisible detection."*
Measured on DEV-20, separating crops whose GT track carries no number (2,090) from those that do
(4,965 numbered; on the 5,879 featurised rows):

| invisible-class detector | ROC AUC |
|---|---|
| the shipped 2024 ResNet34 **legibility classifier** (the thing the head was to replace) | **0.7038** |
| the frozen trunk's own `p_illegible` (PARSeq's end-token mass) | 0.6429 |
| **the evidential head's learned `p_none`** | **0.6200** |
| the evidential head's total-evidence uncertainty `u` | 0.5553 |

**The learned no-number class is worse than both things it was meant to beat**, and the
total-evidence uncertainty is barely above chance. On the *corpus* holdout the same head scores
0.9718 `p_none` AUC — a 0.35-AUC gap that has a clean explanation: **the head's no-number
supervision was defined by the legibility classifier** (`leg <= 0.7`), so it learned to predict its
own teacher's decision, and on DEV-20 — where "invisible" comes from GT rather than from that
classifier — it is a *lossy copy of the teacher*, 0.62 against the teacher's 0.70.

This is the session's sharpest negative and it retires an assumption, not just an arm: **the free
"illegible" supervision at scale is not free.** 56.8% of v3's digit crops failing a legibility
filter is a large, cheap, per-crop-sound signal about *that filter*, and only indirectly about
whether a number is visible. Any future abstention head needs a no-number label that does not come
from the gate it is replacing.

*(Scope: DEV-20's per-crop "invisible" label is inherited from the track, so a crop whose number is
genuinely hidden still counts as numbered — `JERSEY_HEAD_VOTER.md` §4.2. That biases all four rows
in the table identically and cannot explain the ordering.)*

### 2.5 The pipeline pass, and the crop-set equivalence check (§1.5.4): PASS

`tools.gsr_v6det --stages percrop --reader edl --leg-thresh 0.0`, server-side, 4 shards on GPU 1,
DEV-20 `positions_v6det` copied from the laptop. **20/20 sequences in 13 minutes**, 52,115 crops.

Removing the legibility gate is not a 5x but a **4.13x**: torso RoIs go **10,802 -> 44,659** of
52,115 crops (the gate passed 11,213; 85.7% of all crops yield a pose torso). Every one of the
44,659 carries a Dirichlet.

The check the registration demanded, against the on-record laptop `_v6_v6det` parquet:

| | |
|---|---|
| crops, on record / server | 52,113 / 52,115 (2 extra server-side) |
| **paired on `(track_id, frame)`** | **52,113 / 52,113 = 100%** |
| legibility \|delta\| > 0.01 | **79 crops (0.152%)**; > 0.05: 27 (0.052%) |
| the shipped `leg > 0.5` gate flips | **5 crops (0.0096%)** |
| torso flag agreement on crops legible on both | **11,206 / 11,210 = 0.9996** |

**Weaker than the `GSR_S3_READER.md` §2 standard** (which had legibility identical to 5 decimals on
every row, because both arms ran on one machine). The residual is a genuine cross-machine
difference — different CUDA/torch/python on the two hosts, and detector boxes that can differ by a
pixel. At 0.01%-of-crops magnitude it cannot carry a GS-HOTA effect, but it is a deviation and it is
recorded as one rather than described as "identical".

### 2.6 Rung 2 — tracklet aggregation: the Koshkina control PASSES, the d-at-floor bar passes
### its literal form and NOT its same-partition form

450 graded points (432 evidential x {voting, fusion}, 18 v6-reader), DEV-20, `_v6det` track ids,
GSR jersey GT. **Note the GT cache had to be re-keyed:** `tools.ocr_density.track_gt` read
`positions/` unconditionally, so grading a `_v6det` arm scored re-numbered tracks against the
original extraction's ids. Fixed and namespaced (`positions_subdir`); the on-record cache is
untouched. 1,072 original `_v6det` player/GK tracks, 1,025 auditable, 810 carrying a GT number.

**The density/precision frontier — max `d` at each precision floor:**

| precision floor | evidential / voting | **evidential / fusion** | v6 reader / voting |
|---|---|---|---|
| 0.88 | 0.4748 @ 0.8969 | 0.4618 @ 0.9135 | **0.4860** @ 0.8864 |
| 0.90 | 0.4487 @ 0.9154 | **0.4618** @ 0.9135 | 0.4608 @ 0.9066 |
| 0.92 | 0.4272 @ 0.9229 | **0.4496** @ 0.9201 | 0.4310 @ 0.9324 |
| 0.93 | 0.4049 @ 0.9430 | **0.4403** @ 0.9314 | 0.4310 @ 0.9324 |
| 0.94 | 0.4049 @ 0.9430 | **0.4198** @ 0.9404 | 0.3946 @ 0.9440 |
| 0.95 | 0.3713 @ 0.9509 | **0.3853** @ 0.9577 | **unreachable** |
| the shipped operating point (incumbent rule) | — | — | 0.4114 @ 0.9365 |

**(a) The Koshkina voting control: fusion WINS.** Additive Dirichlet fusion beats the incumbent
voting machinery run on the *same evidential rows* at every precision floor from 0.90 to 0.95
(+0.013 to +0.035 of `d`), losing only at 0.88 where precision is not the binding constraint. This
is the one place the evidential machinery earns its keep on its own terms, and it is the answer to
the pre-registered question: **fusion ships, not voting-on-evidential-reads.**

**(b) The `d >= 0.35 at precision >= 0.92` bar: PASS literally, INCONCLUSIVE honestly.**
Fusion reaches **d 0.4496 at precision 0.9201** — both above the bar, which was written against arm
A's **0.3139 @ 0.9256**. But arm A was measured on a *different track partition* (pre-`_v6det`), and
on **this** partition the v6 reader already sits at **d 0.4310 @ 0.9324** at the >= 0.92 floor and
**0.4114 @ 0.9365** at its shipped rule. Against that same-partition incumbent the evidential arm
does **not** dominate both axes at 0.92: it buys +0.019 of `d` for -0.012 of precision. It only
strictly extends the frontier at the **high-precision end** (0.94: +0.025 `d`; 0.95: the v6 reader
cannot reach that precision at all at any rule in the grid). The pre-registered bar was set against
a stale reference and is therefore reported both ways.

**(c) A transferability finding worth its own line: the incumbent aggregation rule emits ZERO reads
on evidential evidence.** `min_crop_conf = 0.90` is unreachable for a Dirichlet mean — the head's
peak mass is capped near `e(1-w)q / (100 + e) ~ 0.68` by construction — so the shipped rule gives
`d = 0.0000`. Evidential outputs are not drop-in for a chain calibrated on softmax confidences; the
rule must be re-swept, which is why this session swept it and why a future arm must not assume
otherwise.

**(d) The uncertainty channel is inert in the voting family.** `max_u` at 0.6 / 0.8 / 1.01 produces
*identical* `d` and precision for every voting rule, and `max_p_none` likewise — `percrop_votes`
already gates on `argmax != ILLEGIBLE` plus a confidence floor, which subsumes both. `u` earns its
place only inside fusion, and even there `max_p_none = 0.3` carries the effect while `max_u` moves
`d` by <= 0.012. Consistent with §2.4: `u` is a 0.555-AUC signal on DEV-20.

The two points carried to rung 3, chosen on DEV rung-2 numbers as the registration allows (rung 3
selection is on GS-HOTA; these are the *candidates*), bracketing the `GSR_S3_READER.md` §6 density/
precision trade rather than assuming which side wins:

| name | rule | d | read precision |
|---|---|---|---|
| `edl_fuse92` | fuse, conf 0.5, crops 2, leg 0.0, `max_u` 1.01, `max_p_none` 0.3 | 0.4496 | 0.9201 |
| `edl_fuse93` | fuse, conf 0.5, crops 1, leg 0.5, `max_u` 0.8, `max_p_none` 0.3 | 0.4403 | 0.9314 |

### 2.7 A bug that cost a wasted arm, found because the control would not reproduce

The first rung-3 attempt ran for **55 minutes on 7 of 20 sequences** and was killed. Cause:
`tools.gsr_eiou.BOX_SUBDIR` is a module constant defaulting to `detbox_cache` — the **original**
detector's per-detection box cache. `tools.gsr_v6det.stage_arm` rebinds it per detector arm; this
session's harness did not. Both caches exist on disk (`detbox_cache` 58 files, `detbox_cache_v6det`
20), so nothing warned: the EIoU re-link silently matched `_v6det` positions against the base
detector's boxes, produced a garbage fragment structure, and the connector's clustering went
super-linear on it.

After the fix (`BOX_SUBDIR` rebound + an explicit missing-cache `SystemExit`) the identical run does
the re-link in **1 minute** and the whole arm in **2.5 minutes**. Every number below is post-fix.
This is exactly the failure mode the harness check exists to catch, and it was caught *by* the
control being wrong, not by any assertion.

### 2.8 Rung 3 — DEV-20 end to end: **FAIL by one sequence, with a real effect underneath**

Harness check first: the v6 control re-derived through this session's harness gives GS-HOTA
**51.808280888021784** against the V0 control's **51.808280888021784** — **exact, 0 of 20 sequences
differ** — and reproduces V1's admissible-set numbers exactly (**610 / 870 named, 13.25 identities
per sequence**, against V1 §2.7's 610/870 and 13.2). The chain is the v6 chain.

Nine arms, DEV-20, everything frozen but the per-crop evidence and its aggregation rule.
`conc` = share of net gain carried by the two best sequences (the `GSR_V5.md` §2.1 guard, demote at
>= 0.80); `d` / `prec` are the rung-2 tracklet numbers of the same rule.

| arm | GS-HOTA | delta | DetA | AssA | IDF1 | helped | Wilcoxon p | conc | named | slots/seq | d | read prec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`edl_fuse94`** | **53.0645** | **+1.2562** | **41.135** | 68.458 | **58.24** | **11/20** | **0.0355** | 0.49 | 614/868 | 13.30 | 0.4104 | 0.9438 |
| `edl_p940` | 52.7047 | +0.8964 | 40.628 | 68.375 | 57.73 | 9/20 | 0.233 | 0.67 | 622/867 | 13.50 | 0.4198 | 0.9404 |
| `edl_fuse93` | 52.6955 | +0.8873 | 40.703 | 68.225 | 57.81 | **13/20** | 0.0642 | 0.52 | 625/869 | 13.65 | 0.4403 | 0.9314 |
| `edl_p958` | 52.5571 | +0.7488 | 40.479 | 68.244 | 57.57 | 8/20 | 0.551 | 1.10 | 609/868 | 13.00 | 0.3853 | 0.9577 |
| `edl_fuse95` | 51.8772 | +0.0690 | 39.244 | 68.581 | 56.34 | 7/20 | 0.691 | 97.5 | 599/868 | 12.80 | 0.3974 | 0.9444 |
| `edl_vote92` | 51.8405 | +0.0323 | 39.308 | 68.372 | 56.47 | 8/20 | 0.836 | 8.12 | 618/868 | 13.40 | 0.4272 | 0.9229 |
| **control (v6 reader)** | **51.8083** | — | 39.308 | 68.287 | 56.45 | — | — | — | **610/870** | **13.25** | 0.4114 | 0.9365 |
| `edl_fuse92` | 51.4382 | -0.3701 | 38.868 | 68.077 | 55.94 | 11/20 | 0.981 | — | 623/865 | 13.65 | 0.4496 | 0.9201 |
| `edl_p977` | 48.1736 | **-3.6347** | 33.748 | 68.767 | 50.31 | 6/20 | 0.0084 | — | 548/868 | 10.85 | 0.2910 | 0.9770 |

| gate criterion | required | best arm (`edl_fuse94`) | |
|---|---|---|---|
| DEV-20 GS-HOTA delta | >= +1.0 | **+1.2562** | PASS |
| sequences helped | >= 12/20 | **11/20** (8 hurt, 1 exactly tied) | **FAIL** |
| concentration | top-2 < 0.80 of net | 0.49 (top-2 +12.51 of net +25.50) | pass |

**VERDICT: FAIL.** The gate was "both, not either", and it misses the breadth half by one sequence.
The pre-declared FAIL response applies: the v6 reader ships unchanged and no `_v7e` artifact enters
any freeze. The arm with 13/20 breadth (`edl_fuse93`) is +0.8873, so no single point clears both.

**And the multiplicity is stated, not hidden:** nine DEV arms were scored. `edl_fuse94`'s p = 0.0355
is uncorrected; Bonferroni over the eight evidential arms puts the threshold at 0.00625, which it
does not clear. `results/GSR_V5.md`'s v5.1 lesson (+0.61 DEV -> -0.05 TEST-38) is the reason the
gate was sized at +1.0 with a breadth requirement, and the reason a near-miss is not promoted.

### 2.9 What the frontier actually says — and the one clean, isolated win

Ordering the arms by read precision instead of by score makes the mechanism obvious:

| read precision | d | GS-HOTA |
|---|---|---|
| 0.9201 | 0.4496 | 51.44 (**-0.37**) |
| 0.9229 | 0.4272 | 51.84 |
| 0.9314 | 0.4403 | 52.70 |
| **0.9365 (incumbent)** | **0.4114** | **51.81** |
| 0.9404 | 0.4198 | 52.70 |
| **0.9438** | **0.4104** | **53.06** |
| 0.9577 | 0.3853 | 52.56 |
| 0.9770 | 0.2910 | 48.17 (**-3.63**) |

**The headline comparison is a matched-density one.** The incumbent reads **d 0.4114 at precision
0.9365**; `edl_fuse94` reads **d 0.4104 at precision 0.9438** — density matched to 0.001 (441 vs 440
of 1,072 tracks) — and scores **+1.2562 GS-HOTA**, carried by **DetA +1.83**. On this chain, at
fixed density, **0.0073 of read precision is worth ~1.26 GS-HOTA**. That is a clean statement about
the objective and it holds `GSR_S3_READER.md` §6's direction: the scored objective wants precision,
and it wants it out of the reader, not out of the aggregation rule.

The optimum is interior on both sides: pushing density (0.4496 @ 0.9201) costs -0.37, and pushing
precision (0.2910 @ 0.9770) costs -3.63 by starving the roster (slots/sequence 13.25 -> 10.85,
named 610 -> 548).

**The isolated uncertainty-filter measurement.** `edl_fuse94` and `edl_p940` differ in **exactly one
field**, `max_u` 0.6 vs 1.01 — same reader, same crops, same fusion, same everything else:

| | `max_u` | d | read precision | GS-HOTA |
|---|---|---|---|---|
| `edl_p940` | 1.01 (off) | 0.4198 | 0.9404 | 52.7047 |
| `edl_fuse94` | **0.60** | 0.4104 | 0.9438 | **53.0645** |

**The total-evidence uncertainty filter is worth +0.3598 GS-HOTA on its own**, spending 0.0094 of
density to buy 0.0034 of precision. This is the one place the Dirichlet machinery pays for itself
end-to-end, and it pays *despite* `u` scoring only 0.555 AUC as an invisible-class detector (§2.4) —
`u` is a poor answer to "is a number visible" and a useful answer to "should this crop's vote
count", which are not the same question and were being conflated.

**The admissible set moved, and it is NOT where the GS-HOTA came from.** V1 predicted the gain would
arrive as roster slots. Measured: `edl_fuse93` has the *largest* admissible set (13.65 slots/seq,
625 named) and scores +0.89; `edl_fuse94` has a *smaller-than-incumbent-ish* set (13.30 slots/seq,
614 named, i.e. +0.05 slots and +4 names over the control) and scores +1.26. Across the nine arms
slots/sequence and GS-HOTA delta do not order together at all (the -3.63 arm is the only one where
the admissible set explains the score, and it explains a collapse). **V1's admissible-set hypothesis
is not supported by this session's evidence:** at this operating point GS-HOTA moves with *read
precision*, not with roster size.

### 2.10 Negatives, limits, and what was NOT done

1. **The headline is a FAIL** — 11/20 against a 12/20 requirement, and no arm clears both halves.
2. **The Grad invisible-detection claim does not reproduce** (§2.4): the learned no-number class is
   0.6200 AUC against the shipped legibility classifier's 0.7038 on DEV-20, and `u` is 0.5553.
3. **The registered architecture failed and was replaced mid-session** (§2.2). The free 100-way head
   reads DEV-20 at 0.49 where its own trunk reads 0.97.
4. **The "free illegible supervision at scale" is not free** (§2.4). Negatives defined by the gate
   you are replacing teach the model to imitate the gate.
5. **Removing the legibility gate is not what won.** `edl_fuse94`, `edl_fuse93`, `edl_p940`,
   `edl_p958` and `edl_p977` all keep `min_legibility = 0.5`; the ungated arms (`edl_fuse92`,
   `edl_vote92`) score -0.37 and +0.03. The 4.13x extra crops were *available* to every arm and the
   winning rules declined them. The session's premise — that the 19.1% legibility gate is the
   binding ceiling — is **not** supported end-to-end.
6. **Head hyperparameters were selected after all four DEV-20 rung-1 rows had been computed**
   (§2.3). The criterion used is corpus-holdout-only and is stated, but the selection was not blind.
7. **Nine DEV arms, one corpus, one trunk, one seed, one head architecture.** No repeat run, no
   second seed, no confidence interval beyond the paired Wilcoxon.
8. **The crop-set equivalence is 0.01%-level, not bit-exact** (§2.5), because the pass ran on a
   different machine than the on-record control.
9. **The corpus holdout cannot grade numbers** (§2.1) — the trunk memorised it.
10. **Not done:** sn-reid letter crops; jersey-2023's own illegible crops (~150k more negatives);
    any light adapter on the trunk (the plan allowed "optionally light adapters" and none was
    trained); a re-swept connector `tau` on the new evidence; a refit digit-confusion prior; a
    second seed; TEST-38; test-49; any commit; any `METRICS_VERSION` bump (no shipped metric moved).
11. **GPU spend: ~1.1 h of the 12-16 h budget.** The frozen-feature cache (136 s for 137k crops) is
    why: the expensive thing this session was *not* the training.

## 3. Files

- Code (new): `generator/evidential_jersey.py` (head, EDL loss, `EvidentialGateHead`, `trunk_dist`,
  `fuse_tracklet`, `player_disjoint_split`, `roc_auc`, `save_head`/`load_head`, the `_demo`
  unit-check), `tools/gsr_v7_edl.py` (rungs 2 + 3), `tests/test_evidential_jersey.py` (8 tests).
- Code (default-preserving edits): `tools/koshkina_str_sidecar.py` (`--edl-head`, `--device`;
  without the flag the output and the CPU device are unchanged), `generator/jersey_id.py`
  (`edl_head` reaches the sidecar; `crop_reads` returns `alpha` and the Dirichlet mean),
  `eval/gsr_jersey.py` (`--edl-head`, `--leg-thresh`; `alpha`/`u` columns), `tools/ocr_density.py`
  (`crop_alpha`, the `max_u`/`max_p_none`/`fuse` rule keys, and the `positions_subdir` fix to
  `track_gt`/`load_gt_cache`), `tools/gsr_v6det.py` (`percrop_variant`, `--reader edl`,
  `--edl-head`, `--leg-thresh`).
- Weights (server, not in the repo): `~/data/edl/edl_head_v7e.pt` (= `edlg_k0.01_n0.3.pt`),
  **md5 `4cfdaea31b0a9febf984d524da25097c`**, 1.66 MB. Frozen trunk:
  `~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt` (S2 arm 4t, unchanged).
- Server artifacts: `~/work/edl/{edl_corpus,edl_feats,edl_train,edl_curve,edl_dev20}.py`,
  `~/data/edl/{manifest.parquet,feats_corpus.npy,feats_dev20.npy,*.pt,*_fit.json,*_rung1.json}`
  (488 MB), `~/logs/v7v3_*.log`, `~/run_edl.sh`.
- Evidence: `outputs/gsr/koshkina_percrop_v7e_v6det/` (20 parquets, 8.7 MB, `alpha` + `u` per crop).
- Results: `results/gsr_benchmark/gsr_v7_v3_component.json` (rung 0/1 raw),
  `gsr_v7_v3_sweep.json` (450 rung-2 points), `gsr_v7_v3_arms.json` (rung 3).
- **Two new rule files, and what they do** (the `GSR_V7_V0.md` §7 lesson about
  `ocr_density_rule_v6_eiou.armB.json` — a rule file's *filename* silently re-points every future
  arm at that variant):
  - `results/ocr_density_rule_v6_v6det_eiou.json` — the incumbent 0.80-floor rule **verbatim**
    (`conf 0.9 / votes 3 / leg 0.5 / emit_all false`), written so the control arm runs through the
    same code path as the evidential arms. Verified byte-equal to `rule_for("0.80", "")`, so
    resolving it is a **provable no-op**; deleting it changes nothing.
  - `results/ocr_density_rule_v7e_v6det_eiou.json` — the evidential arm's rule. It resolves ONLY
    for the `_v7e_v6det_eiou` variant, which no on-record arm uses. It is rewritten by each
    `--arm` invocation, so it names whichever evidential point ran last: read
    `gsr_v7_v3_arms.json` for what a given number was produced with, never this file.

## 4. Reproduce

```
python -m generator.evidential_jersey                       # rung 0, CPU, seconds
# server (GPU 1):
python ~/work/edl/edl_corpus.py                             # corpus + negative pose pass
python ~/work/edl/edl_feats.py --paths ~/data/edl/manifest.parquet  --out ~/data/edl/feats_corpus
python ~/work/edl/edl_feats.py --paths ~/data/edl/dev20_paths.parquet --out ~/data/edl/feats_dev20
python ~/work/edl/edl_train.py --fit --gate --name edlg_k0.01_n0.3 --kl-max 0.01 --none-share 0.3
python ~/work/edl/edl_dev20.py --name edlg_k0.01_n0.3       # rung 1
python -m tools.gsr_v6det --detector s4b --stages percrop --reader edl \
    --edl-head ~/data/edl/edl_head_v7e.pt --leg-thresh 0.0 --seqs <DEV-20>
# laptop (CPU):
python -m tools.gsr_v7_edl --sweep                          # rung 2
python -m tools.gsr_v7_edl --arm '<rule json>' --reader edl --name <arm>   # rung 3
pytest tests/test_evidential_jersey.py tests/test_ocr_density.py tests/test_jersey_id.py
```
