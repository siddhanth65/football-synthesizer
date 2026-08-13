# v8 session W1 — the seam experiment: the leader's identity head, our aggregation

Campaign v8, session W1. The 61.48 leader (Broadcast2Pitch / KIST, WACV 2026) names its own
bottleneck in §6 of the paper: *"jersey number predictions are aggregated via majority voting, which
is particularly fragile when correct digits are sparsely observed"*. `docs/WINNER_REPO_RECON.md`
located that sentence in code: `predict_role_and_jersey_batch_clip` does `torch.max(logit, 1)` and
returns `(role: str, number: int, color: str)` — **every per-crop confidence is destroyed before any
aggregation sees it**, and both aggregators downstream are bare `Counter.most_common(1)`.

This session runs their identity head as a **private measurement instrument** on OUR DEV-20 GT crops
with the logits preserved, and then compares three aggregators on *the same per-crop outputs*: their
majority vote, our evidential/Bayesian fusion (`generator/evidential_jersey.fuse_tracklet`), and our
`percrop_votes` machinery. The comparison their paper cannot run, on their own module.

**CLAIMS-HYGIENE (plan v8, binding).** Their code and weights are measurement instruments only. The
clone lives at `<scratchpad>/winner`, the checkpoint at `<scratchpad>/winner_weights`, the extended
predictor and the per-crop outputs at `<scratchpad>/w1`. **Nothing of theirs enters the repo tree and
nothing of theirs enters any shipped artifact.** The two majority-vote functions in our harness are
reimplemented **clean-room from the prose description in `docs/WINNER_REPO_RECON.md` §3**, not copied
from their source. Everything under `tools/` and `results/` here is ours.

---

## 1. REGISTRATION (written before any comparison was run)

### 1.1 The instrument, and what is fetched

| item | source | size | hash |
|---|---|---|---|
| repo `SoccernetGSR` @ HEAD `dbfb65c0` | github.com/yinmayoo185/SoccernetGSR | 27 MB | see §2.1 |
| `CLIP_Jersey.pth` | the public Drive folder `1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh` | 937 MB | see §2.1 |
| `openai/CLIP` @ `d05afc4` | github.com/openai/CLIP (MIT) | 3 MB | see §2.1 |

Their `CLIPFinetune.__init__` calls `clip.load("ViT-L/14")`, which would fetch OpenAI's own 890 MB
checkpoint only to overwrite it a line later. The local extension therefore builds the tower with
`clip.model.build_model()` **directly from the `clip_model.*` half of `CLIP_Jersey.pth`**, so no
second download happens and the loaded weights are provably theirs (strict `load_state_dict`,
asserted). The architecture is rebuilt from their 81-line module verbatim (three
`Linear(768,256)-ReLU-Dropout-Linear(256,C)` heads with `C = 5 / 11 / 11`, plus `Linear(768,768)`
colour projection); the checkpoint's own key set is the proof it matches.

### 1.2 The seam, in code

Their predict path is extended **locally only** to return, per crop, the full
`role_logits [5]`, `digit1_logits [11]`, `digit2_logits [11]` and the 13 colour cosine similarities,
instead of four `argmax` indices. Everything else — CLIP preprocess, fp16 tower, `.float()` before
the heads, `get_number`'s `100` sentinel, the 13 hardcoded colour names, `F.normalize` on both sides
of the colour cosine — is theirs, unchanged. Their loop is per-image; ours batches. **Batching is
asserted numerically equivalent** (batch-1 vs batch-64 on 64 crops, max abs logit delta reported in
§2.2) before any result is believed, because that is the only edit that could move a number.

### 1.3 The crops and the graded universe

`outputs/gsr/gt_crops/validation` restricted to the DEV-20 partition
(`tools.jersey_head_trial.dev20_sequences`, `sorted(validation)[::3]`): **7,055 crops / 473 GT
tracks**, of which 4,965 crops sit on the **331 tracks carrying a GT jersey**. This is the crop set
every reader trial in the campaign used (`tools/gsr_crops.py`, the 20,067/19,993 crop law) and
exactly the 7,055/4,965 of `results/GSR_V7_V3.md` §2.3. DEV-20 is GSR **validation**; our arm-4t
trunk never saw it.

Graded universe: GT tracks with role `player` or `goalkeeper`. `d` = read tracks / universe;
`read_precision` = correct reads / auditable reads (auditable = the track carries a GT number) —
the `tools/ocr_density.py::measure` definitions verbatim, so every number here is comparable to
`GSR_V7_V3.md` §2.6.

Class layout for the comparison: **101 classes**, index 0 = "no number", index `k+1` = number `k`
for `k` in `0..99`. Their two digit heads give a joint over `11 x 11` cells, each mapped through
their own `get_number` (so `100` -> index 0 and single digits fold correctly); our reader's
100-class rows lift into the same layout with a zero at "number 0". DEV-20 GT carries no jersey `0`,
so a `0` emission is a wrong read and is counted as one.

### 1.4 The three aggregators (all on the SAME per-crop outputs)

**(a) THEIRS — argmax + tracklet majority vote.** Per crop: `argmax` each digit head, `get_number`.
Per tracklet: `Counter.most_common(1)`, with the two sentinel rules the recon describes —
`refine_tracklets`-style (if the modal value is `100` and the list is not entirely `100`, drop the
`100`s and re-vote) and `write_json_file_team`-style (`100` wins only at `>= 97%` share, else it is
deleted and the runner-up wins) — plus the plain no-sentinel rule as a third variant.

**(b) OURS — evidential / Bayesian fusion.** `generator.evidential_jersey.fuse_tracklet`, unchanged,
fed `alpha = 1 + e_i * p_i` from their probability vectors. Two evidence rules, both declared here:
`soft` (`e_i = 1`, the parameter-free Bayesian soft-count posterior `Dir(1 + sum_i p_i)`) and `conf`
(`e_i = 100 * max_c p_i(c)`, so the total-evidence uncertainty `u = K/S` is a live per-crop channel
and the V3 `max_u` filter — the one machinery that paid +0.3598 GS-HOTA on its own — is testable
here). A third, `logprob`, is reported as the classical naive-Bayes alternative (sum of log
probabilities) and is NOT part of the gate.

**(c) OURS — the votes machinery.** `generator.jersey_id.percrop_votes` at the same rule structure
the chain ships (`min_crop_conf`, `min_votes`, `emit_all`).

**Fairness, pre-declared.** All three families sweep a knob grid of comparable size on DEV-20, are
graded on the identical track universe with the identical grader, and the frontier — not a
hand-picked point — is compared. The majority-vote family is given a per-crop confidence floor and a
`min_crops` knob it does not have in their code, i.e. the baseline is **strengthened beyond what they
ship**, so a fusion win cannot be an artefact of knob budget.

### 1.5 THE PRE-REGISTERED GATE

> On their own per-crop outputs, at matched coverage, our fusion reads more precisely than their
> majority vote.

Formally, for coverage targets `T` in `{0.30, 0.40, 0.50, 0.60}` of `d`, let
`P_family(T) = max{ read_precision(rule) : rule in family, d(rule) >= T }`.

**PASS iff `P_fusion(T) > P_majority(T)` at `>= 3` of the 4 targets.** Reported alongside (not
gating): the single-point comparison at their shipped operating point (plain majority vote, no
confidence floor, `min_crops = 1`) against the fusion rule of nearest `d`; and `percrop_votes` on
the same frontier.

**FAIL response, pre-declared:** the result is written as a measured negative — "the leader's named
bottleneck is not recoverable by aggregation alone at the tracklet level on their own head's
outputs" — the claim is filed as FAIL, and the campaign's core claim is downgraded to whatever the
head-to-head still supports. No arm is re-swept to rescue it.

### 1.6 Head-to-head (reported, not gated)

Their head vs our v6 reader (arm-4t PARSeq, `~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt`)
on the identical 7,055 crops:

- per-crop **precision at matched emit** — the S2/V3 bar is `0.8344 at emit 0.2761`;
- tracklet `d` at read-precision floors `{0.88, 0.90, 0.92, 0.94}`, the `GSR_V7_V3.md` §2.6 table's
  own floors;
- disagreement structure: where each is right alone, and the oracle-union ceiling (an ensemble's
  upper bound). **No ensemble is built this session** — the plan says report only.

Stated asymmetry, because it is not controllable: their head consumes the whole person crop at
CLIP's 224x224; our chain applies a ResNet34 legibility gate and a KeypointRCNN torso RoI before
PARSeq. Each side is measured as its authors ship it, on the same source crops.

### 1.7 The leakage probe (their training data is unverifiable)

Their role head has exactly GSR's five categories in GSR's order, so it was fitted on GSR-format
annotations; the paper states the jersey fine-tune used the SoccerNet jersey dataset (Cioppa 2022).
Whether GSR **validation** entered their training set cannot be verified from the repo. Probe: run
the same head on a seeded 3,000-crop sample of GSR **train** GT crops and compare per-crop number and
role accuracy against DEV-20. A large train-over-validation gap is a memorisation tell; a small one
bounds the concern. Declared before running, and reported either way.

### 1.8 Guards and what this session may not do

1. Selection is on the pre-registered gate only; every other number is reported, never selecting.
2. No end-to-end GS-HOTA run, no TEST-38, no test-49, no submission, **no commit** (W-session
   commits are batched by the orchestrator).
3. `METRICS_VERSION` does not move: no shipped metric changes here.
4. Their artifacts stay outside the repo tree (§0). If any table below required their code to be
   vendored, the table would not be produced.
5. The GPU is a 4 GB laptop: one job at a time, no pytest suite while a pass runs.

---

## 2. RESULTS — VERDICT: the registered gate **FAILS**; the campaign's underlying claim survives in
## a stronger form than the gate asked for

**The pre-registered gate fails.** Additive Dirichlet fusion beats their majority vote at **1 of 4**
coverage targets (needed: 3 of 4). On their head it cannot even *reach* the two high-coverage
targets: 88% of tracklets abstain under unfiltered fusion, and no `max_p_none` value fixes it. The
reason is mechanistic and is this session's most useful negative — **evidence summation amplifies a
miscalibrated no-number mass that per-crop argmax discards** (§2.7).

**But the thing the gate was a proxy for is demonstrated, on a stricter comparison than the gate
asked for.** At **exactly** their shipped operating point — same head, same crops, same 350 read
tracklets, same 309 auditable reads, the only difference being `Counter.most_common(1)` versus our
confidence-weighted `percrop_votes` — read precision goes **0.3107 -> 0.4175, +0.1068 (+34%
relative), 96 -> 129 correctly named tracklets**. The gap is monotone in the per-crop confidence
floor and vanishes at floor 0.9, which locates it exactly where the paper says its bottleneck is:
the crops whose confidence `torch.max` threw away.

**And the head-to-head is lopsided.** At the matched emit rate 0.2761, our v6 reader reads
DEV-20 GT crops at **0.8330** per-crop precision against their CLIP head's **0.3406**. At every
tracklet read-precision floor from 0.90 up, our reader carries ~10x their coverage. Their head is
nonetheless *complementary in a measurable way*: 99 crops it reads correctly are ones ours gets
wrong or abstains on (§2.8).

### 2.1 What was fetched, with hashes

| artifact | identity | size |
|---|---|---|
| `yinmayoo185/SoccernetGSR` @ HEAD | commit `dbfb65c05c847e8e50baa47aa91ace960cc5e600` (2026-06-02, the post-issue-#1 fix) | 27 MB |
| `jersey_model/CLIPFinetune.py` | sha256 `f79638f3fc071dd42b1338d4ab4d536cb39ad2b00da40c79dcdcb4a5bd99477f` (81 lines) | 2 KB |
| **`CLIP_Jersey.pth`** | **sha256 `c068865c6ff28ff8b2dd7e90d2cfd37b72dfe4e117a3be79eb88a5f98e8afa1e`**, md5 `2a9e7fe2e2a6d98e6347bd9770d1f87f` | **936,987,722 B** |
| `openai/CLIP` (MIT) | commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`; `clip/model.py` sha256 `dc4981bb...0926`, `clip/clip.py` sha256 `3891eee0...ad0a` | 3 MB |

Nothing else was downloaded: the other checkpoints in their Drive folder were listed
(`skip_download=True`) and left alone.

**Checkpoint anatomy (new information, not in the recon).** 460 tensors: `clip_model.*` 446 —
**fp16** ViT-L/14 vision tower (`visual.positional_embedding [257, 1024]` = 224 px / patch 14) *and*
the full fp16 text tower — plus **fp32** heads exactly as the 81-line module declares:
`role_classifier` `(768->256->5)`, `digit1_classifier` / `digit2_classifier` `(768->256->11)`,
`color_projection` `(768, 768)`. The text tower ships inside the checkpoint, so the colour path is
self-contained.

**Recon correction (§2 of `docs/WINNER_REPO_RECON.md`).** The Drive folder holds **seven**
checkpoints plus the 8-file LoRA folder, not five. Two were not listed in the recon:
`ClipReid_B16.pth` and `osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth`
(the stock Market-1501 OSNet). File IDs are recorded in `results/gsr_benchmark/gsr_v8_w1_fetch.json`.

### 2.2 Instrument fidelity — the two things that could invalidate every number

**(a) It really is their model.** `load_state_dict` reports **zero unexpected keys** and the only
missing keys are `clip_model.*`, which `clip.model.build_model` supplies from the same file. Their
`clip.load("ViT-L/14")` would have pulled OpenAI's 890 MB checkpoint and overwritten it; skipping
that download changes nothing and is verified by the key set.

**(b) Batching is not a change.** Their loop is one image at a time. Batch-1 vs batch-64 on the same
crops, max absolute logit delta: **role 0.0311, digit1 0.0691, digit2 0.0250, colour-cosine 0.0014**
against a logit range of `[-63.3, +14.6]` — fp16 accumulation noise. On a 512-crop decision check:
**role 0/512, digit2 0/512, digit1 1/512 argmax flips**, i.e. **1 of 512 (0.20%) crop-level jersey
decisions** move. Recorded rather than described as "identical".

Throughput on the 4 GB laptop (RTX 3050, fp16, batch 32): **7,055 crops in 152 s** (46 crops/s);
the 29,765-crop full-density pass of §2.9 ran on the same card.

### 2.3 Harness check — our reader reproduces the on-record bar before anything is compared

`tools/gsr_w1_seam.py --reader` over the identical 7,055 crops with the arm-4t checkpoint,
`leg_thresh 0.0` (so the persisted rows cover every crop their head also saw; the shipped 0.5 gate is
re-applied offline from the `leg` column):

| | this session | on record (`GSR_V7_V3.md` §2.3 / `CLUSTER_SESSION_S2.md` §3.5) |
|---|---|---|
| crops | 7,055 | 7,055 |
| pose torso RoIs | 5,876 | 5,879 |
| reads at max emit, gate 0.5 | **1,371 @ emit 0.2761** | **1,371 @ 0.2761** |
| per-crop precision there | **0.8330** | **0.8344** |
| ungated trunk at emit 0.2761 | **0.8592** | 0.8607 |

Two crops of difference on a 1,371-read denominator (0.0014 of precision), from three crops where
KeypointRCNN produced no torso this time. The harness is the S2/V3 harness.

### 2.4 Their head on our DEV-20 GT crops — per-crop

7,055 crops / 473 GT tracks; 4,965 crops on the 331 tracks carrying a GT jersey.

| per-crop metric (numbered crops) | **theirs (CLIP ViT-L/14)** | **ours (arm-4t + gate)** | ours, ungated |
|---|---|---|---|
| max emit | 0.3754 | 0.2761 | 0.8544 |
| precision at max emit | **0.2666** | **0.8330** | 0.3392 |
| precision at emit 0.05 | 0.8105 | **0.9919** | 0.9921 |
| precision at emit 0.10 | 0.6250 | **0.9899** | 0.9879 |
| precision at emit 0.20 | 0.4340 | **0.9738** | 0.9627 |
| **precision at emit 0.2761 (the S2 bar)** | **0.3406** | **0.8330** | 0.8592 |

Their head reads *more often* (37.5% of numbered crops vs 27.6%) and is **2.4x less precise** at the
matched emit rate. Its confidence channel is nonetheless real and well-ordered — 0.81 precision on
its top 5% of reads against 0.27 overall — which is exactly the property that makes
confidence-weighted aggregation beat counting on it (§2.6).

Other attributes, same pass, same crops:

* **role**: 0.9015 accuracy over all 7,055 crops (5 classes). Confusions are structural, not noise:
  **153 player crops called goalkeeper, 355 called referee, 103 GK crops called player**. The GK/
  referee confusion is the same failure mode our own team module has to handle.
* **colour -> team, their exact rule** (lower-cased colour names, the two most frequent strings in
  the clip become team 0/1, per-track majority): **83.8% of player/GK tracks correct (357/426)**
  with an *oracle* left/right permutation per sequence, i.e. an upper bound on their §3.4 pitch-x
  side assignment. Per-sequence range 0.545 (SNGS-081) to 0.955. This is a measurement of the module
  their own Table 7 prices at **-5.94 GS-HOTA**, and it is consistent with that price.

### 2.5 THE GATE — three aggregators on THEIR per-crop outputs

`P_family(T) = max{read_precision : d >= T}` over the pre-registered grids (majority 45 rules,
fusion 60, votes 30; the `max_u` ablation adds 60 fusion rows, declared and reported separately).
473 player/GK tracklets, GSR jersey GT.

| coverage target `T` | **majority (theirs)** | **fusion (ours, gate grid)** | fusion + `max_u` (declared ablation) | votes (ours) |
|---|---|---|---|---|
| 0.30 | 0.6045 @ d 0.4202 | 0.5732 @ d 0.3920 | **0.6599 @ d 0.3498** | 0.5989 @ d 0.4202 |
| 0.40 | **0.6045 @ d 0.4202** | unreachable | 0.5870 @ d 0.4366 | 0.5989 @ d 0.4202 |
| 0.50 | 0.5064 @ d 0.5657 | unreachable | unreachable | **0.5277 @ d 0.5657** |
| 0.60 | 0.3958 @ d 0.7254 | unreachable | unreachable | **0.4549 @ d 0.7254** |

| gate criterion | required | measured | |
|---|---|---|---|
| fusion beats majority at coverage targets | `>= 3` of 4 | **1 of 4** (0.30 only, and only with the ablation `max_u`) | **FAIL** |
| — fusion on the registered grid alone | — | **0 of 4** | FAIL |
| (reported, not gating) votes beats majority | — | **2 of 4** (0.50, 0.60), tie-to-slight-loss at 0.30/0.40 | — |

**VERDICT: FAIL.** The pre-declared response applies: this is written as a measured negative, no arm
is re-swept to rescue it, and the campaign's core claim is downgraded to what the evidence below
actually supports.

*(A post-hoc diagnostic was run and also failed: the registered `max_p_none` grid held only
`{0.3, 1.01}`, so 90 extra fusion rules at `max_p_none in {0.4, 0.5, 0.6, 0.7, 0.9}` were graded to
test whether the coverage ceiling was a grid artefact. It is not — every post-hoc row is dominated,
and the frontier does not move by a single point. Labelled `posthoc` in the raw JSON and excluded
from every gate statistic.)*

### 2.6 The result that survives: at their OWN operating point, mass beats counting

The gate compared frontiers. This table compares something stricter — **the same tracklets**. The
`revote` majority rule and `percrop_votes` keep exactly the same crops (a crop votes iff its argmax
is a number and its confidence clears the floor), so at every knob setting the two families read the
**identical** tracklets and differ only in how the surviving votes are combined: **count** versus
**summed confidence mass**.

| per-crop conf floor | min votes | `d` (both) | tracks read | auditable | **majority** | **votes (ours)** | delta | correct: mv -> votes |
|---|---|---|---|---|---|---|---|---|
| **0.0** | **1** | **0.8216** | **350** | **309** | **0.3107** | **0.4175** | **+0.1068** | **96 -> 129** |
| 0.0 | 2 | 0.5423 | 231 | 218 | 0.3899 | 0.4404 | +0.0505 | 85 -> 96 |
| 0.0 | 3 | 0.3263 | 139 | 137 | 0.4745 | 0.5036 | +0.0292 | 65 -> 69 |
| 0.3 | 1 | 0.7254 | 309 | 288 | 0.3958 | 0.4549 | +0.0590 | 114 -> 131 |
| 0.3 | 2 | 0.3803 | 162 | 160 | 0.5250 | 0.5437 | +0.0187 | 84 -> 87 |
| 0.5 | 1 | 0.5657 | 241 | 235 | 0.5064 | 0.5277 | +0.0213 | 119 -> 124 |
| 0.5 | 2 | 0.2934 | 125 | 124 | 0.6371 | 0.6452 | +0.0081 | 79 -> 80 |
| 0.7 | 1 | 0.4202 | 179 | 177 | 0.6045 | 0.5989 | **-0.0056** | 107 -> 106 |
| 0.9 | 1 | 0.2394 | 102 | 101 | 0.8020 | 0.8020 | 0.0000 | 81 -> 81 |
| 0.9 | 3 | 0.0563 | 24 | 24 | 0.9583 | 0.9583 | 0.0000 | 23 -> 23 |

The top row **is their shipped configuration** — no confidence floor, one vote, the sentinel re-vote
(their `refine_tracklets.majority_vote`); their *other* rule, `write_json_file_team.majority_jersey`
with the 0.97 threshold, produces the **identical** 0.3107 on this crop set, because with `<= 15`
crops per tracklet "not entirely sentinel" and "under 97% sentinel" almost always coincide.

**The delta is monotone in the confidence floor and reaches zero at 0.7-0.9.** That is the
mechanism, measured: counting and mass agree once the low-confidence crops are gone, so the entire
gain lives in the crops that `torch.max` discarded — the paper's own diagnosis, quantified at
**+0.1068 read precision / +33 correctly named tracklets of 309**.

**Paired statistics at that operating point** (all 331 numbered player/GK tracklets, so tracklets
where both abstain count against both):

| | |
|---|---|
| correctly named tracklets | **96 -> 129 of 331** (0.290 -> 0.390) |
| sequences helped / hurt / tied | **13 / 1 / 6** of 20 |
| Wilcoxon on the 20 per-sequence deltas | **p = 0.00172** |
| tracklet-level flips (votes right alone vs majority right alone) | **37 vs 4**, binomial **p = 1.03e-7** |

Single comparison, pre-registered as a reported quantity, no multiplicity to correct.

The same comparison on **our** reader's per-crop outputs, same code, same tracks:

| per-crop conf floor | min votes | `d` | auditable | majority | votes | delta |
|---|---|---|---|---|---|---|
| 0.0 | 1 | 0.7160 | 291 | 0.8866 | **0.9141** | +0.0275 |
| 0.5 | 1 | 0.7066 | 290 | 0.9069 | **0.9241** | +0.0172 |
| 0.9 | 1 | 0.6643 | 276 | 0.9457 | **0.9565** | +0.0109 |
| 0.7 | 2 | 0.5610 | 236 | 0.9831 | **0.9915** | +0.0085 |

**The effect replicates on a second, much stronger head — and it is 4x smaller there** (+0.0275 vs
+0.1068). Confidence-weighted aggregation buys most where the per-crop reader is worst, which is a
statement about *when* the seam is worth attacking, and it says: on their head, a lot; on ours,
a little.

### 2.7 Why the Dirichlet fusion fails on their head — the negative worth keeping

`fuse_tracklet` sums per-crop evidence and then abstains if the fused no-number mass beats the best
number. On their per-crop rows:

| | theirs | ours (gated) |
|---|---|---|
| mean per-crop `p(no number)` | **0.5740** | 0.8020 |
| crops whose argmax IS no-number | **72.3%** | 80.9% |
| mean FUSED `p(no number)` per tracklet (soft evidence, no filter) | **0.5342** | — |
| tracklets where fused no-number wins -> abstain | **375 / 426 = 88.0%** | — |

Ours looks *worse* on the first two rows and fuses fine; theirs looks better and collapses. The
difference is **calibration, not magnitude**: our gated rows are one-hot on no-number for crops the
legibility gate blocked (a hard, correct abstention), whereas their sentinel mass is a *broad,
diffuse* 0.3-0.6 present on nearly every crop, including crops whose argmax is a number. Additive
evidence accumulates that diffuse mass across 15 crops until it outweighs any single number;
per-crop argmax never sees it, because argmax is invariant to how much mass the *loser* holds.

Three consequences, and they are the transferable part of this session:

1. **Bayesian evidence fusion is only as good as the per-crop calibration it is fed.** On a head
   whose abstention class is systematically over-weighted, summing is strictly worse than
   argmax-then-count. Our V3 machinery was fitted against a *learned, calibrated* `p_none`; drop-in
   reuse on a foreign head is not valid, and this session measured the failure rather than assuming
   it.
2. **The hybrid is what wins**: per-crop argmax as a *relative* gate (their robustness) plus
   confidence-mass accumulation over the survivors (our uncertainty awareness) — which is exactly
   what `percrop_votes` already is. The campaign's aggregation asset is the votes machinery, not the
   Dirichlet, when the reader is foreign.
3. **The `max_u` channel still earns its place**: on their head it is the only knob that raises
   fusion coverage at all (`d` 0.3967 -> 0.4366 at `max_u` 0.6, +0.010 precision as well), because
   dropping low-evidence crops removes exactly the diffuse rows that inflate the sentinel. Consistent
   with `GSR_V7_V3.md` §2.9's isolated +0.3598 GS-HOTA finding: `u` is a poor "is a number visible"
   detector and a useful "should this crop vote" weight.

### 2.8 Head-to-head — their identity head vs our v6 reader

Tracklet level, best rule of any family at each read-precision floor (auditable reads `>= 20`):

| read-precision floor | **theirs** | **ours (gate 0.5)** | ours, ungated |
|---|---|---|---|
| 0.88 | d 0.1127 (48 reads) | **d 0.7160** (291) | d 0.7559 (280) |
| 0.90 | d 0.0634 (27) | **d 0.7160** (291) | d 0.6995 (257) |
| 0.92 | d 0.0634 (27) | **d 0.7066** (290) | d 0.6761 (252) |
| 0.94 | d 0.0610 (26) | **d 0.6784** (280) | d 0.6362 (260) |
| 0.95 | d 0.0610 (26) | **d 0.6643** (276) | d 0.6362 (260) |

**Our reader carries 10-11x their tracklet coverage at every floor from 0.90 up.** Their head cannot
put more than 27 tracklets above 0.92 precision on this crop set.

Disagreement, per crop, on the 4,965 numbered crops:

| | gate 0.5 | ungated |
|---|---|---|
| crops read: theirs / ours | 1,864 / 1,371 | 1,864 / 4,242 |
| both read | 1,165 | 1,750 |
| agreement where both read | **0.3554** | 0.2571 |
| correct: theirs / ours | 497 / **1,142** | 497 / **1,439** |
| both right | 398 | 416 |
| **theirs only right** | **99** | 81 |
| ours only right | 744 | 1,023 |
| neither right | 3,724 | 3,445 |
| oracle-union recall | 0.2499 | **0.3061** |

**Ensemble assessment (report only, per the plan — nothing was built).** An oracle union would add
**99 crops (+8.7% relative on 1,142 correct)** over our gated reader, and 81 over the ungated one.
That is a real but small complementary signal, and it is bought at a bad exchange rate: their head
emits 1,864 reads at 0.267 precision, so any *realisable* (non-oracle) combiner has to find those 99
crops inside 1,367 wrong ones. A confidence-gated union is the only plausible form — their top 5% of
reads are 0.81 precise — and that top 5% is 248 crops, of which our reader already gets most. **Not
worth building on this evidence**; if W3 revisits it, the entry condition should be a measured
*disjoint* win at matched precision, not an oracle union.

### 2.9 The crop-density objection, tested — and the sharpest single number in the session

The obvious attack on §2.6: our GT crop law keeps **15 crops per tracklet**, while their pipeline
votes over **every frame** (~750 per 30 s sequence). Counting is sample-hungry; maybe majority vote
only looks bad because it is starved. So three DEV-20 sequences (SNGS-021 / 039 / 087, three
different kit pairs) were re-cut at **full density — every GT box, no 15-crop cap: 29,765 crops over
70 tracklets, ~425 crops/tracklet, 28x the evidence** — and run through the same instrument
(15.5 min on the laptop GPU).

Same 70 tracklets, same shipped rule (no confidence floor, one vote, sentinel re-vote):

| crops per tracklet | tracks read | auditable | **majority** | **votes (ours)** | delta |
|---|---|---|---|---|---|
| 15 (the crop law) | 55 | 47 | 0.3191 | **0.4681** | **+0.1489** |
| **~425 (every frame)** | 62 | **49** | 0.4286 | **0.4898** | **+0.0612** |

**The objection is real and it is not fatal.** Majority vote gains **+0.1095** from 28x more crops;
confidence-weighted votes gains **+0.0217** — because it had already extracted the information from
15. The gap therefore shrinks by ~60% and **survives**.

And the headline that follows from the same table: **our aggregation on 15 crops (0.4681) beats
their aggregation on 425 crops (0.4286)** — same tracklets, same head, same ground truth. Reading
the confidences of 15 crops is worth more than counting the argmaxes of 425.

Their per-crop precision also improves with density (0.2907 -> 0.2895 at max emit is flat, but the
top-5% precision rises 0.7838 -> 0.8423 and precision at emit 0.20 rises 0.4966 -> 0.5019), i.e. the
full-density crop pool contains genuinely easier crops, which is why both aggregators improve.

**Honest scale caveat:** 3 sequences, 49 auditable tracklets. The +0.0612 is **3 tracklets**
(21 -> 24). This probe answers "does the effect survive density", not "how big is it"; the
20-sequence, 331-tracklet measurement of §2.6 is the sized one.

### 2.10 Side measurements and the leakage probe

**The leakage probe (§1.7) returns a clear signal, and it is not validation leakage.** Same head,
same instrument, seeded 3,000-crop sample of GSR **train** GT crops vs the 7,055 DEV-20 crops:

| | GSR train (3,000 crops) | DEV-20 (validation, 7,055) | gap |
|---|---|---|---|
| role accuracy | **0.9877** | 0.9015 | -0.086 |
| per-crop read rate (numbered) | 0.4919 | 0.3718 | -0.120 |
| **per-crop read precision** | **0.4730** | **0.2681** | **-0.205 (1.76x)** |
| no-number recall on unnumbered crops | 0.9441 | 0.9555 | +0.011 |

*(This table applies their decision rule literally — per-digit `torch.max` then `get_number` — so
DEV-20 reads 0.3718 / 0.2681 where §2.4's folded-joint argmax reads 0.3754 / 0.2666. The two differ
on 0.4% of crops, where a number's mass is split across cells; every other table in this file uses
the folded form, which is the one the three aggregators consume.)*

Their head was fitted on GSR **train** (the 0.9877 role accuracy on GSR's own five categories in
GSR's own order is decisive) and generalises materially worse off it. **This is legitimate — train
is the training split — and it does not indicate validation leakage.** It does mean every number in
this file is a validation-generalisation number for them, and the same is true for us: our arm-4t
corpus contains 3,888 GSR-train rows and no validation row (`GSR_V7_V3.md` §1.3). Both heads are
train-fitted and validation-tested; the comparison is fair on that axis.

### 2.11 Negatives, limits, and what was NOT done

1. **The registered gate FAILED** (§2.5): 1 of 4 coverage targets, and 0 of 4 on the registered grid
   without the `max_u` ablation. No re-sweep was run to rescue it.
2. **Our Dirichlet fusion does not transfer to a foreign head** (§2.7). 88% of tracklets abstain;
   the post-hoc `max_p_none` diagnostic did not move the frontier by a single point. Any future
   claim of the form "our evidential fusion improves X" must first check X's abstention calibration.
3. **The whole session is a crop-level, tracklet-level study. There is no GS-HOTA number in it.**
   Nothing here has been shown to move an end-to-end score; §2.6's +0.1068 is read precision on GT
   tracklets, not a benchmark delta. The `GSR_S3_READER.md` §6 lesson (a denser/more precise
   tracklet rule scoring *worse* end-to-end) is exactly why that distinction is kept.
4. **The tracklet unit is not their tracklet unit.** We grade GT tracks; their pipeline grades
   DeepEIoU tracklets after IDATR split/merge. §2.9 closes the crop-density half of that gap on
   3 sequences; the tracking-error half is untested and untestable without running their tracker.
5. **Their identity head is only half their identity system.** The 61.48 headline uses the LLaMA
   variant (+1.35 over CLIP per their Table 5), which emits free text and carries no per-crop
   confidence at all — the seam this session exploits **does not exist on their shipping
   configuration**, only on the CLIP one. That is stated plainly because it bounds the claim.
6. **`d1` batching moves 1 crop decision in 512** (§2.2), and 3 of 7,055 crops changed pose-torso
   status vs the on-record V3 pass (§2.3). Both are recorded rather than called "identical".
7. **The colour->team figure (83.8%) uses an oracle side permutation** and is therefore an upper
   bound on their §3.4 assignment; and it is a measurement of their module on GT tracks, not of
   their pipeline.
8. **No ensemble was built** (plan says report only), no end-to-end run, no TEST-38, no test-49, no
   submission, **no commit**, no `METRICS_VERSION` change.
9. **Single seed, single crop law, one head each.** Their head was not re-run at a second
   preprocessing (the 12 of 7,055 crops that our 256x128 crop law aspect-distorts, 0.17%, were
   measured and judged not worth a control run).
10. **GPU spend: ~22 min total** (7,055-crop pass 152 s, 3,000-crop train probe 66 s, full-density
    29,765-crop pass 15.5 min) plus 11.5 min for our reader pass. No cluster time was used.

## 3. Files

**Ours (repo, commit-safe — but NOT committed this session):**

- `tools/gsr_w1_seam.py` (new): the three aggregator families, the grader, the frontier statistic,
  the per-crop matched-emit table, the disagreement table, our reader pass, and a `--demo`
  self-check on the aggregators + the digit fold.
- `results/GSR_V8_W1.md` (this file).
- `results/gsr_benchmark/gsr_v8_w1.json` (585 graded rules over 3 sources + per-crop tables +
  disagreement), `gsr_v8_w1_sub3_15crop.json`, `gsr_v8_w1_sub3_full.json` (§2.9),
  `gsr_v8_w1_fetch.json` (provenance + hashes + the corrected Drive listing).
- `outputs/gsr/w1_reader_arm4t_dev20.parquet` (gitignored): our reader's per-crop evidence on the
  7,055 DEV-20 GT crops, `leg_thresh 0.0`, arm-4t.

**Theirs / instrument (scratchpad only — never in the repo tree, never in a shipped artifact):**

- `<scratchpad>/winner/` — their repo @ `dbfb65c0`;
- `<scratchpad>/winner_weights/CLIP_Jersey.pth` — their checkpoint;
- `<scratchpad>/openai_clip/` — openai/CLIP (MIT) @ `d05afc4`;
- `<scratchpad>/w1/winner_percrop.py` — the extended predictor (the seam);
  `<scratchpad>/w1/full_density.py` — the §2.9 re-cut;
  `<scratchpad>/w1/{winner_dev20,winner_train3k,winner_full3}.npz` + their crop tables;
  `<scratchpad>/w1/fullcrops/` (29,765 JPEGs, 200 MB).

**Weights used, ours:** `~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt` (S2 arm 4t,
unchanged) and the shipped `legibility_resnet34_soccer_20240215.pth`.

## 4. Reproduce

```
# 1. instrument (scratchpad; NOT in the repo)
git clone --depth 1 https://github.com/yinmayoo185/SoccernetGSR   <scratchpad>/winner
git clone --depth 1 https://github.com/openai/CLIP                <scratchpad>/openai_clip
python -c "import gdown; gdown.download(id='11HA90V7h6GDeoqUAl2-4ZcOPmUJKK6nz', \
    output='<scratchpad>/winner_weights/CLIP_Jersey.pth')"
pip install ftfy
python <scratchpad>/w1/winner_percrop.py --split validation --dev20 --out dev20
python <scratchpad>/w1/winner_percrop.py --split train --sample 3000 --out train3k
python <scratchpad>/w1/full_density.py

# 2. ours
python -m tools.gsr_w1_seam --reader outputs/gsr/w1_reader_arm4t_dev20.parquet \
    --parseq-ckpt ~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt
python -m tools.gsr_w1_seam --compare --winner <s>/w1/winner_dev20.npz \
    --winner-crops <s>/w1/crops_dev20.parquet \
    --reader-parquet outputs/gsr/w1_reader_arm4t_dev20.parquet \
    --out results/gsr_benchmark/gsr_v8_w1.json
python -m tools.gsr_w1_seam --compare --winner <s>/w1/winner_full3.npz \
    --winner-crops <s>/w1/crops_full3.parquet --out results/gsr_benchmark/gsr_v8_w1_sub3_full.json
python -m tools.gsr_w1_seam --demo
```

## 5. What this means for the campaign (for the orchestrator, not a result)

- The seam is **real but narrower than the plan assumed**: it is a property of *confidence-weighted
  counting*, not of Dirichlet fusion, and it exists only on their CLIP configuration.
- The "build on the winner's identity module" path is **closed by the head-to-head**: their head
  reads DEV-20 GT crops at 0.34 where ours reads at 0.83. There is nothing to import.
- What survives as a defensible contribution is the *measurement*: on the leader's own module, at
  their own operating point, replacing `Counter.most_common(1)` with confidence-mass aggregation is
  worth +0.1068 read precision (p = 0.0017 paired over 20 sequences), and it still wins when their
  aggregator is given 28x more crops than ours.
- W2's premise is untouched by this session and remains the larger prize (their own ablation prices
  calibration at +10.28 against identity's +1.35).

