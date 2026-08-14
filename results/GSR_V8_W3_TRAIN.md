# v8 session W3-TRAIN — the purity-tiered jersey reader retrain

Campaign v8, session W3, training half. The corpus half (`GSR_V8_W3_PREP.md`) and the annotation
half (STATUS 2026-08-14, claims `v8-w3-007..013`) are complete; this session spends the cluster GPU.

**Everything in §1 was written and saved before a single training step ran.** Every number in it is
copied from an artifact that already existed (`GSR_V8_W3_PREP.md`, `CLUSTER_SESSION_S2.md`,
`GSR_S3_READER.md`, `knowledge/claims.json`, and the corpus parquets), never from a score produced
this session.

> **RESULT: the ladder stops at rung 1, and the registered hypothesis is REFUTED.**
> Both arms fail the component gate, and the purity arm fails *harder* than the volume arm.
> On the clean split neither arm beats the incumbent: **ARM V (volume) -0.0051 per-crop precision
> (p = 0.51, indistinguishable), ARM P (purity) -0.0211 (p = 0.0070, a real regression)**, and head
> to head **purity loses to volume by -0.0160 (p = 0.043)**. No rung 2, no rung 3, no TEST-38.
> **Banking the negative: tightening the legibility gate from 0.7 to 0.99 makes the reader worse.**
>
> **FOLLOW-UP (same day, §10 registration / §11 results), closing §7's limits 3 and 4:** the
> run-to-run seed spread of ARM V over three retrains is **0.0059**, so the -0.0160 above is 2.7x
> seed noise and **H1 stays refuted**, while ARM V's -0.0051 null is **inside** the noise floor. A
> volume-matched purity arm (ARM P100, 0.99 tier at 4x caps, 97,347 rows = 1.02x ARM V) still loses
> to ARM V by **-0.0255, p = 0.0020**: the purity loss is **not** a volume artifact.

---

## 1. Registration (a priori)

### 1.0 The hypothesis, and why it is not the one W3 was planned around

W3 was planned as a **volume** expansion: jersey-2023's full legibility pass lifted the admissible
pool from the 201,313-crop sample S2 trained on to **140,278 crops** (`v8-w3-006`), "3.7x what the
incumbent reader was trained on".

Sid's 2,520 per-crop labels (`v8-w3-010`) then measured the gate that produces that pool: at n=2,520
its **recall is 0.964 and its precision 0.728**. It barely loses readable glyphs; **~27% of what it
admits shows no readable number to a human.** Transferred band-by-band onto jersey-2023's score
histogram (`v8-w3-013`), ~38,300 of the 140,278 admitted crops are positive-class label noise, and
the threshold is a **purity dial**: `> 0.99` keeps 82,941 crops at an estimated precision of 0.889.

The campaign's own v6 lesson (`GSR_S3_READER.md` §6, claim `s3-reader-003`) is that reader headroom
should be **banked as precision, not spent on density**. So the registered primary hypothesis is:

> **H1: a purity-tiered corpus produces a better reader than the volume corpus, at the same recipe.**

### 1.1 A sizing fact that reshapes the experiment, recorded before it is used

The arm-4t recipe caps each source (`~/work/jersey/build_corpus.py`): **60 crops per jersey-2023
tracklet**, 60 per GSR-train track, 12 per (v3 game x number). Under that cap the "3.7x" headroom
mostly evaporates and, more importantly, **the purity tier costs far less volume than the raw crop
counts suggest**:

| jersey-2023 tier | admitted crops | tracklets | rows after cap 60 |
|---|---:|---:|---:|
| `> 0.7` (shipped) | 140,278 | 926 | **48,567** |
| `> 0.99` (purity) | 82,941 | 831 | **37,098** |

A 41% cut in crops is a **24%** cut in trainable rows. Two consequences, both registered now:

1. The two arms are a **purity contrast at roughly matched volume** on the dominant source — a
   cleaner experiment than the plan assumed.
2. **The 140,278 headline never reaches a corpus under this recipe.** S2's `planned` took 38,133
   jersey-2023 rows; the full pass at the same tier and cap yields 48,567, i.e. **+27%, not +268%**.
   The binding constraint is the cap, not the crop count. This corrects the framing carried in
   `GSR_V8_W3_PREP.md` §4.2 and STATUS 2026-08-13.

### 1.2 Corpus arms (at most two are trained; ARM P first)

Sources, tiers and caps. Every row is a `(crop, digit-string)` pair; **no abstention class, no
empty-label rows, no model-derived labels** anywhere (`v7-v3-002`).

| source | label provenance | cap | ARM P rows (`> 0.99`) | ARM V rows (`> 0.7`) |
|---|---|---:|---:|---:|
| jersey-2023 train, numbered tracklets | tracklet number, gate-carried | 60 / tracklet | **37,098** | **48,567** |
| GSR train, 45 non-holdout sequences | track number, gate-carried | 60 / track | **1,947** | **3,266** |
| SoccerNet-v3 digit crops | per-action ID, gate-carried | 12 / game x number | **18,375** | **42,479** |
| Sid tier-A human numbers, non-holdout | human tracklet number, gate-carried | 60 / tracklet | **~71** | **312** |
| Sid tier-B human-verified visible cells | **human, per crop** | none | **584** | **584** |
| **total (pre-torso, pre-val-split)** | | | **~58,075** | **95,208** |

**The tier rule, stated so it cannot be bent later:** the legibility tier is a purity dial on
**gate-carried** labels only. The 584 tier-B rows are the only crops in this project where a human
looked at *that exact crop* and confirmed the number is readable, so a model score cannot improve on
them — they enter **both** arms at their own provenance, unfiltered. Sid's tier-A rows are
tracklet-carried like every other source and are tiered like every other source.

**SoccerNet-v3 decision (the one §1 was required to make from inventory).** `~/data/v3_digits/legibility.parquet`
exists server-side with a per-crop `leg` column for all 105,702 crops, so **the same 0.99 tier IS
applied to v3 in ARM P**. It keeps 18,535 crops and **82 of the 88 distinct numbers**, so the
vocabulary that made v3 "the whole effect" in S2 (arm 2 -> arm 4 = +0.3078) survives the tier.

**Declared confound, not hidden:** ARM P is 61% of ARM V's size, so H1 is tested as a *package*
(purity + its volume cost) rather than purity alone. A volume-matched purity arm (0.99 tier with the
cap raised) would disentangle it and is **not** run — two arms is the registered budget.

### 1.3 Held-out slice (fixed before training, listed in full)

Rule, fixed a priori: **`sorted(57 GSR-train sequence ids)[::5]` = 12 sequences.**

```
SNGS-060 SNGS-065 SNGS-070 SNGS-075 SNGS-099 SNGS-104
SNGS-109 SNGS-114 SNGS-154 SNGS-159 SNGS-164 SNGS-169
```

They carry **3,045 of the 13,994 numbered GSR-train GT crops** (962 pass the eval-side legibility
gate at 0.5). Every row of every arm from a GSR-train source — including Sid's — is filtered against
this list before the corpus is built.

*Deviation declared:* `GSR_V8_W3_PREP.md` §6 proposed 8 sequences; 12 is registered instead, for
statistical power (~850 expected reads instead of ~570), at a cost of 1.3% of ARM V's rows.

**The contamination caveat §6 demanded, carried:** `build_corpus.gsr_train()` applies **no** sequence
filter, so the incumbent (arm 4t) trained on all 57 train sequences, including these 12. The rung-1
baseline is therefore biased **upward**, which makes the comparison conservative in the candidate's
favour. Stated now, not discovered later.

### 1.4 Recipe (the S3 arm-4t shape, deviations declared)

Verbatim from `~/work/jersey/run_4t.sh` / `run_arms.sh`:

```
python train.py +experiment=parseq dataset=real data.root_dir=<corpus> \
  trainer.max_epochs=25 pretrained=~/models/koshkina/parseq_shipped.ckpt \
  trainer.devices=1 trainer.val_check_interval=1.0 data.batch_size=128 data.max_label_length=2
```

* **init: the shipped Koshkina PARSeq**, not the S1 synthetic pretrain (S2 §3.5 measured synthetic
  init at -0.0217 against shipped on the same corpus).
* **geometry: pose torso RoI**, built by `~/work/jersey/torso_corpus.py`'s KeypointRCNN pass
  (S2 §3.4 measured a complete geometry crossover; a full-body corpus is not scoreable against this
  chain).
* **checkpoint selection: `last.ckpt`**, the uniform S2 rule — no arm is selected on any eval split.
* 95/5 train/val LMDB split, `seed=0`, `cap_per_group(random_state=0)` — deterministic row selection.
* **No architecture change, no abstention class, no new losses, no hyperparameter search.**

Deviations from arm 4t, both forced by the corpus and declared here: (a) 12 GSR-train sequences are
excluded from the corpus (§1.3); (b) jersey-2023 enters from the **full** legibility pass rather than
S2's 201,313-crop sample. Nothing else moves.

### 1.5 Gate ladder — all bars fixed now

**Rung 0 — corpus integrity (mechanical, must pass before training).** Zero `validation` / `test` /
`challenge` content in any row; every row's label is either gate-admitted at its arm's tier or
carries a human provenance string; zero rows whose label has a model origin; zero holdout-sequence
rows.

**Rung 1 — component: per-crop precision at matched emit, on the §1.3 held-out slice.**
Harness: `~/work/jersey/dev20_arm.py`, unmodified, pointed at a holdout torso set built by the same
`dev20_torso.py` RoI code. Eval-side legibility gate **0.5** (the S2 arm-scoring value), torso gate
on, PARSeq decode identical for both readers, same crops, same order.
Matched emit: raise the candidate's per-crop confidence floor to the smallest value whose read count
is **<=** the incumbent's read count; report the natural (floor 0) numbers too.

> **PASS: candidate per-crop precision >= incumbent's + 0.02 at matched emit.** FAIL = stop.

Reported alongside, registered now so it is not a post-hoc addition: **McNemar** on the paired
per-crop correctness (the two readers score the identical crops, so the paired test is the
informative one) and Wilson CIs on both precisions. Neither is a gate.
**Secondary, non-gating:** the same harness on the on-record DEV-20 torso set
(`~/data/dev20/`, 4,965 numbered crops), where the incumbent's on-record number is
**emit 0.2761 / precision 0.8344** (S2 §3.5) — context, not a bar.

**Rung 2 — tracklet: read density at the v6 precision-holding operating rule.**
The rule is **FROZEN as shipped**: the incumbent's 0.80-floor rule, `min_crop_conf 0.90,
min_votes 3, leg 0.50` (`GSR_S3_READER.md` §4.1). **No operating-point search, no sweep, no
`--sweep` rule written.** DEV-20, 1,596 player/GK tracks.

> **PASS: d >= 0.3139 at read precision >= 0.9256** (the on-record v6 numbers under that rule).

Because the DEV-20 per-crop pass re-detects frames (`eval.gsr_jersey.extract_track_crops`) and is
therefore stack-sensitive under `v8-w4-004`, the incumbent's DEV-20 evidence is **re-derived on the
cluster in the same env, same GPU, same session**, and the candidate must clear the same-stack
incumbent as well as the on-record absolute bar. Both are reported; **the absolute bar is the
registered verdict.**

**Rung 3 — end-to-end: DEV-20 weights-only swap.**
Frozen S5 EIoU partition (`e 0.3, rounds 1, w_app 0.5, app_max 0.30`, CLIP, `tau 0.450`), everything
upstream of PARSeq identical between control and arm, control produced in this session on the same
stack as the arm (`v8-w4-004`). Crop-level pairing verified (identical `(track_id, frame)` keys,
identical `torso` flags, legibility equal to within 1e-3) before any score is read.

> **PASS: mean paired GS-HOTA >= +1.0 AND >= 12/20 sequences helped.**

GS-DetA / GS-AssA decomposition and the Wilcoxon p reported. Only a full-ladder pass sends the
component to W5. **FAIL at any rung stops the ladder there and the negative is banked.**

**Rung 4 — TEST-38: not in this session at any outcome.** test-49 and the challenge split are not
opened.

### 1.6 Anti-patterns named, so they can be checked against the record

| anti-pattern | how §1 forecloses it |
|---|---|
| **score-peeking** | Every bar above is copied from an artifact predating this session. §1 was saved before the first `train.py` launch. |
| **sub-point DEV decisions** | Rung 3's bar is +1.0 GS-HOTA, not "positive". A +0.4 is a FAIL, and will be reported as one. |
| **teacher-imitation labels** | No abstention class, no empty-label rows, no model-derived negatives. The 1,749 human "hidden" cells and the 3,288 tier-A `none` crops are deliberately **excluded** from the reader corpus — they are gate supervision, and the gate is not retrained here. |
| **challenge-split data** | Not fetched, not prepared, not touched. Corpus = jersey-2023 train + GSR train (45 seqs) + SoccerNet-v3 labelled games only. |
| **operating-point search disguised as aggregation tuning** | Rung 2 runs the shipped rule and only the shipped rule. No `--sweep`. `GSR_S3_READER.md` §6's negative (the swept rule scored 0.94 GS-HOTA *worse*) is the reason. |
| **arm selection on the eval split** | `last.ckpt` for every arm, fixed by S2's uniform rule. |
| **quietly dropping the losing arm** | ARM V's result is reported whether it wins or loses; if budget stops it, the reason is recorded as a budget stop, not a result. |

---

## 2. Rung 0 — corpus integrity: PASS, both arms

Assembled server-side by `~/work/w3/build_w3_corpus.py --stage rows`, which reuses S2's
`build_corpus.cap_per_group` verbatim so the row selection is the arm-4t selection.

| | ARM P (`> 0.99`) | ARM V (`> 0.7`) |
|---|---:|---:|
| jersey-2023 (cap 60/tracklet) | 37,098 | 48,567 |
| SoccerNet-v3 (cap 12/game x number) | 18,375 | 42,479 |
| GSR train, 45 non-holdout seqs (cap 60/track) | 1,931 | 3,242 |
| Sid tier-B human-verified visible cells | 584 | 584 |
| Sid tier-A human numbers (cap 60/tracklet) | 71 | 312 |
| **rows** | **58,059** | **95,184** |
| distinct numbers | 82 | 89 |

GSR-train lands 16 / 24 rows under §1.2's projection because tier-B rows that also appear in the
GSR-train source are deduplicated to their human provenance (registered rule, `keep='last'`).

**Integrity checks, all zero:** rows from a `valid` / `test` / `challenge` path **0**; rows from a
held-out sequence **0**; empty labels **0**; missing files on disk **0**. `~/data/w3corpus/rung0.json`.

**Torso conversion** (KeypointRCNN, `dev20_torso.torso_from_keypoints`, one shared content-addressed
cache over the union of both arms): 118,136 source crops -> **109,316 RoIs, 92.5%**, 19 min 30 s on
GPU 1. S2 recorded 93.3% on the same code, so the geometry stage behaved as on record.

| final LMDB | train | val | numbers | torso survival |
|---|---:|---:|---:|---:|
| **ARM P** | **51,803** | 2,726 | 82 | 0.9392 |
| **ARM V** | **83,812** | 4,411 | 89 | 0.9269 |

## 3. Training — both arms, 38 min of A100

`~/work/w3/run_w3_chain.sh` under `tmux new -d -s w3chain`, `CUDA_VISIBLE_DEVICES=1`, GPU 1 verified
at 4 MiB immediately before launch (GPU 0 left to its other user throughout). The log confirms the
registered recipe: `[init] local checkpoint ~/models/koshkina/parseq_shipped.ckpt`,
`[adapt] pos_queries: (1, 26, 384) -> (1, 3, 384)`, `missing=0 unexpected=0`.

| arm | wall clock | epochs | final corpus-val accuracy | checkpoint md5 |
|---|---|---:|---:|---|
| ARM P | 13:32:12 -> 13:47:12 = **15 m 00 s** | 25 | 96.88 | `6be45878103243de191b179e6c8d0795` |
| ARM V | 13:47:12 -> 14:10:11 = **22 m 59 s** | 25 | 93.02 | `8b80904a099adafe8e3fe22a8a557f6b` |

**ARM P's higher corpus-val accuracy is not a result and must not be read as one.** Each arm's val
split is 5% of *its own* corpus, so ARM P is graded on the purer distribution it was trained on.
It is exactly the number a purity tier is guaranteed to inflate, and it is the reason the ladder
grades on held-out crops instead.

## 4. Rung 1 — the registered component gate: FAIL, both arms

Held-out slice (§1.3, 12 sequences), `~/work/w3/w3_component.py` wrapping `dev20_arm.py`'s decode
verbatim. 3,045 numbered GT crops, **877 pass the eval gate** (legibility > 0.5 AND a torso RoI).
Both readers emit on every gated crop, so the emit rates are identical by construction (0.288) and
the registered matched-emit floor is 0.

| arm | reads | emit | per-crop precision | 95% Wilson | delta vs incumbent | McNemar (cand-only / inc-only) | verdict |
|---|---:|---:|---:|---|---:|---|---|
| incumbent arm-4t | 877 | 0.2880 | **0.9692** | [0.9556, 0.9788] | — | — | bar |
| **ARM V** (volume) | 877 | 0.2880 | 0.9361 | [0.9180, 0.9505] | **-0.0331** | 12 / 41, p = 8.2e-5 | **FAIL** |
| **ARM P** (purity) | 877 | 0.2880 | 0.8940 | [0.8718, 0.9126] | **-0.0752** | 6 / 72, p = 1.8e-15 | **FAIL** |

The bar was `>= +0.02`. Both arms are the wrong side of it by a wide, significant margin.

### 4.1 The rung-1 baseline is 94.75% contaminated, measured — the FAIL is real but nearly uninformative

§1.3 declared that arm 4t trained on all 57 sequences. Measured rather than assumed
(`~/work/w3/contam.py`, re-running `build_corpus.gsr_train(0.7, cap 60)`): **831 of the 877
evaluation crops — 94.75% — are literally rows of the incumbent's own training corpus.** The
incumbent's 0.9692 here against its 0.8344 on DEV-20 is the size of that gap.

So rung 1 as registered measures "can a fresh reader beat a memorising baseline on the baseline's
memorised crops", and the answer is no, for any reader. The registered verdict stands — the bar was
fixed in advance and was not met — but the **informative** comparison is the one §1.5 registered
alongside it, on a split neither reader has ever seen.

## 5. The registered DEV-20 secondary — clean for both readers, and it carries the finding

DEV-20 comes from GSR *validation* and `CLUSTER_SESSION_S2.md` §3.3 records it never enters any
corpus; it is not in ARM P's, ARM V's or arm 4t's training data. Same harness, same 4,965 numbered
crops, 1,371 reads.

**Harness validity, first:** the incumbent scores **emit 0.2761 / precision 0.8344** — S2 §3.5's
on-record numbers to four decimals, nine days and two corpora later. The measurement below is made
on the instrument that set the campaign's reader bar.

| arm | reads | emit | precision | 95% Wilson | delta vs incumbent | McNemar | reading |
|---|---:|---:|---:|---|---:|---|---|
| incumbent arm-4t | 1,371 | 0.2761 | **0.8344** | [0.8138, 0.8532] | — | — | on record, reproduced |
| **ARM V** (volume) | 1,371 | 0.2761 | 0.8293 | [0.8085, 0.8483] | **-0.0051** | 38 / 45, **p = 0.51** | indistinguishable |
| **ARM P** (purity) | 1,371 | 0.2761 | 0.8133 | [0.7918, 0.8330] | **-0.0211** | 40 / 69, **p = 0.0070** | a real regression |

**H1 head to head, the comparison this session existed to make:**

| | precision | delta (P - V) | McNemar | verdict |
|---|---:|---:|---|---|
| ARM V (volume, tier 0.7) | 0.8293 | — | — | — |
| ARM P (purity, tier 0.99) | 0.8133 | **-0.0160** | 43 / 65, **p = 0.0428** | **H1 REFUTED** |

**The purity-tiered corpus makes the reader worse, significantly.** The v6 lesson "bank the
headroom as precision" is about the *aggregation operating point*, and it does not transfer to the
*training corpus admission threshold*. Those are different knobs, and this session is the
measurement that separates them.

### 5.1 The obvious artifact, ruled out by measurement

ARM P's tier costs 7 numbers (82 vs 89). Exactly **one** of them, `59`, appears in DEV-20's 37-number
GT vocabulary, on 45 of 4,965 crops — enough in principle to account for half the P-vs-V gap.
Re-running both comparisons with those 45 crops removed:

| comparison | full DEV-20 | vocabulary-controlled | McNemar counts |
|---|---:|---:|---|
| ARM P - ARM V | -0.0160 | **-0.0161** | 43 / 65, unchanged |
| ARM P - incumbent | -0.0211 | **-0.0212** | 40 / 69, unchanged |

The discordant-pair counts are **identical**, i.e. not one of the extra errors was an
out-of-vocabulary number. The regression is not a vocabulary artifact.

### 5.2 What the negative does and does not license

* **Does:** the shipped 0.7 legibility threshold is not leaving reader accuracy on the table.
  Tightening it to 0.99 removed ~38,000 estimated-noisy positives (`v8-w3-013`) and the reader got
  **worse**, so on this evidence that noise is either not harmful or is cheaper than the clean rows
  the tier discards with it.
* **Does not:** separate purity from the volume it costs. ARM P is 61% of ARM V's rows and roughly
  55% of its per-number support across DEV-20's vocabulary. The §1.2 confound was declared in
  advance and is **not** resolved here — a volume-matched purity arm (0.99 tier, cap raised to ~100)
  is the one-run experiment that would resolve it, and it was not run.
* **Does not:** say anything about the *gate* as an emit mechanism. Nothing in this session retrains
  the legibility classifier; the 19.1% DEV-20 emit ceiling (`GSR_S3_READER.md` §7.2) is untouched.

### 5.3 The other negative, and it is the more expensive one

**ARM V — the corpus the whole of W3 was built to produce — buys nothing.** It adds the full
jersey-2023 legibility pass (48,567 rows against S2's 38,133), Sid's 896 human-sourced rows, and it
gives up 12 GSR-train sequences, for **-0.0051 precision at p = 0.51**: a null. The corpus work is
sound and the leakage discipline held, but the *reader* does not move on it. Reading the two W3
sessions together, the measured value of W3 is its two findings (`v8-w3-007`, `v8-w3-010`) and its
corpus, not a better component.

## 6. Ladder state, and what was deliberately not spent

| rung | status |
|---|---|
| 0 — corpus integrity | **PASS**, both arms |
| 1 — component precision at matched emit | **FAIL**, both arms (-0.0331 / -0.0752 against a +0.02 bar) |
| 2 — tracklet `d` at the frozen v6 rule | **NOT RUN.** The registered ladder stops at the first failure. |
| 3 — DEV-20 end-to-end GS-HOTA | **NOT RUN.** |
| 4 — TEST-38 | **NOT RUN**, as registered at any outcome. |

No DEV-20 per-crop OCR pass, no EIoU re-score, no GS-HOTA number exists in this file. TEST-38,
test-49 and the challenge split were not opened. `METRICS_VERSION` unchanged. Nothing shipped: the
incumbent `parseq_v6_arm4t.ckpt` remains the reader.

Two pieces of plumbing were checked in advance and are recorded for whoever does reach rung 2/3:
the server checkout's `eval/gsr_jersey.py`, `tools/gsr_eiou.py`, `generator/jersey_id.py` and
`tools/koshkina_str_sidecar.py` are **byte-identical** to the laptop's (md5), but
**`tools/ocr_density.py` and `tools/gsr_v4.py` differ and must be synced first**. And because
`eval.gsr_jersey.extract_track_crops` re-detects frames, a DEV-20 per-crop pass is stack-sensitive
under `v8-w4-004`: the control must be re-derived on the same host.

## 7. Negatives, limits, and what was NOT done

1. **The registered hypothesis is refuted, not merely unsupported.** Purity is worse than volume at
   p = 0.043 and worse than the incumbent at p = 0.0070, on a split neither model saw.
2. **Rung 1 as registered is a weak instrument**, and this was discovered by measuring, not by
   reasoning: 94.75% of its evaluation crops are the incumbent's training rows (§4.1). A future
   component gate on GSR train must hold sequences out of *both* readers' corpora — which for the
   incumbent means retraining it — or must simply use DEV-20, as §5 did.
3. **Purity and volume are confounded (§5.2)** and the disentangling arm was not run.
   **[CLOSED in §11.3:** ARM P100 ran it — at matched volume purity still loses, -0.0255,
   p = 0.0020, so the confound resolved *against* purity.**]**
4. **Single seed, single recipe.** One `last.ckpt` per arm, no repeats, no variance estimate across
   training runs. The DEV-20 deltas (-0.005, -0.016, -0.021) are of the same order as run-to-run
   PARSeq variance plausibly is, and that variance is **unmeasured** in this project. The McNemar
   p-values quantify crop-level pairing, **not** training-seed noise; a -0.016 at p = 0.043 on one
   seed is directional evidence, not a settled ordering.
   **[CLOSED in §11.2:** two more ARM V retrains put the seed spread at **0.0059** — below the
   -0.016 (which therefore survives) and above the -0.0051 null (which is therefore inside the
   noise floor).**]**
5. **The legibility gate was not retrained**, so Sid's 5,037 per-crop human negatives — the most
   novel supervision W3 produced — are still unused. They are gate supervision, and the gate is the
   component the ladder never touched.
6. **Sid's positives are 0.9% of ARM V's rows and 0.1% of ARM P's**, so this session cannot say
   whether human labels help; it can only say that at this dose nothing moves.
7. **No torso-geometry sweep, no threshold sweep between 0.7 and 0.99.** The two tiers are the two
   registered points. The frontier between them is unmeasured, and 0.9 / 0.95 (125,120 / 113,917
   admitted jersey-2023 crops) are cheap unrun arms.
8. **The build scripts live only on the server** (`~/work/w3/`), the same gap `GSR_V8_W3_PREP.md` §2
   flagged for S2's scripts. They are unvendorable in any useful sense — SoccerNet-v3 is server-only
   — but the gap is real and is restated here rather than quietly repeated.

## 8. Files and provenance

**Repo (new, NOT committed):**
- `results/GSR_V8_W3_TRAIN.md` — this file.

**Checkpoints (gitignored, pulled back, md5 verified on both sides):**
- `outputs/gsr/w3_readers/parseq_w3_armP_purity.ckpt` — 381,471,766 B, md5
  `6be45878103243de191b179e6c8d0795`, server origin
  `~/src/jersey-number-pipeline/str/parseq/outputs/parseq/2026-08-14_13-32-17/checkpoints/last.ckpt`.
- `outputs/gsr/w3_readers/parseq_w3_armV_volume.ckpt` — 381,471,766 B, md5
  `8b80904a099adafe8e3fe22a8a557f6b`, server origin `.../2026-08-14_13-47-17/checkpoints/last.ckpt`.
- Incumbent, for the record: the server's `.../2026-08-05_21-44-57/checkpoints/last.ckpt` md5s
  `39c0c15defafda072ca0cefd0a4b8e18`, **identical to the laptop's**
  `~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt`. The comparison is against the shipped
  reader itself, verified, not against a re-export of it.

**Result JSONs (gitignored, `outputs/gsr/w3_readers/`):** `rung0.json`, `corpus_stats_w3.json`,
`rung1_armP.json`, `rung1_armV.json`, `rung1sec_armP.json`, `rung1sec_armV.json`, `H1_P_vs_V.json`,
`vocabctl_H1.json`, `vocabctl_INC.json`.

**Server (`siddhanth23519@a100server1`, nothing vendored):**
- Code: `~/work/w3/{build_w3_corpus.py, w3_component.py, holdout_manifest.py, contam.py, vocab.py,
  run_torso.sh, run_w3_chain.sh}`
- Data: `~/data/w3corpus/{rows/, _torso/, armP/, armV/}` (1.0 GB), `~/data/w3holdout/`,
  `~/data/w3in/` (the three uploaded artifacts), `~/data/w3out/` (the JSONs above)
- Logs: `~/work/w3/{torso.log, chain.log}`, `~/logs/w3_arm{P,V}.log`

**Uploads to the server this session:** `legibility_j2023_full.parquet` (3,325,410 B, md5
`bc90567805f3e203b1b40096601fb824`), `sid_labels.parquet` (136,918 B, md5
`e5e57de54f3ed6caa4e25bc2582efe12`), and 1,063 GSR GT crops (6,328,320 B tar; 1,017 new to the
server, 46 already present). All md5-verified on arrival.

**Cluster spend:** ~1.4 GPU-h on GPU 1 — torso 19 m 30 s, holdout torso ~1 m 30 s, training
37 m 59 s, seven scoring passes ~25 m. GPU 0 never touched; occupancy checked immediately before
every launch; every job under `tmux`. End state: GPU 1 at 4 MiB, no tmux sessions, quota
160 GB / 500 GB.

## 9. Reproduce

```
# server, in ~/work/w3, env gsr unless noted
python build_w3_corpus.py --stage rows                       # CPU, rung 0
python build_w3_corpus.py --stage torso --batch 64           # GPU 1, 19.5 min
python holdout_manifest.py
CUDA_VISIBLE_DEVICES=1 python ~/work/jersey/dev20_torso.py \
  --manifest ~/data/w3holdout/manifest.parquet --crop-root ~/data/gsr_gt_crops/train \
  --out-dir ~/data/w3holdout/torso --out ~/data/w3holdout/torso.parquet --batch 16
bash run_w3_chain.sh                                         # lmdb + both arms, 38 min
# env parseq, PYTHONPATH=~/src/jersey-number-pipeline/str/parseq
python w3_component.py --torso ~/data/w3holdout/torso.parquet --torso-dir ~/data/w3holdout/torso \
  --leg-parquet ~/data/gsr_crops/legibility.parquet --ckpt-a <arm4t> --ckpt-b <arm> --out <json>
python w3_component.py --torso ~/data/dev20/torso.parquet --torso-dir ~/data/dev20/torso \
  --leg-parquet ~/data/dev20/leg_shipped.parquet --ckpt-a <arm4t> --ckpt-b <arm> --out <json>
python contam.py && python vocab.py
```

---

## 10. Follow-up registration (a priori)

*This is the section the follow-up brief calls "§5". It is appended rather than inserted so the
§1-§9 numbering that `knowledge/claims.json` cites stays valid.*

**Everything in §10 was written and saved before a single GPU job of the follow-up session ran.**
The only computation that precedes it is the cap scan of §10.2, which is a CPU pass over the corpus
manifests and is required by the brief to be stated here before training.

This session closes the two limits §7 declared (items 3 and 4). It ships nothing: **no rung 2, no
rung 3, no GS-HOTA, no TEST read, at any outcome.** Both questions are read on the *same* DEV-20
component harness §5 used — `~/work/w3/w3_component.py`, unmodified, `~/data/dev20/torso.parquet`,
4,965 numbered crops, eval legibility 0.5, torso gate on — and every pass re-scores the incumbent
`parseq_v6_arm4t.ckpt` in the same run as its harness-validity check (**it must reproduce emit
0.2761 / precision 0.8344**; if it does not, that pass is reported as void, not as a result).

### 10.1 Q1 — seed variance of ARM V (the binding unknown)

ARM V is retrained **twice more on the byte-identical corpus** — the LMDB built on 2026-08-14 13:32
is reused, not rebuilt (`armV/train/real/shard{000,001,002}/data.mdb` md5
`b2f341e8f20872c3b7acbe231afdc656`, `016b42d78e7726be59b99a1581f68cbe`,
`f00d3cdbf18cd2859d7883775ef48c38`; `armV/val/shard000/data.mdb`
`ca230adeda9a86121207a773d81f0f54`) — at the identical §1.4 recipe and the identical command line.
**The only difference between the three runs is the training RNG.**

**Stated honestly, because it changes what "seed" means here:** the arm-4t recipe
(`~/src/jersey-number-pipeline/str/parseq/train.py` + `configs/main.yaml`) **sets no seed at all** —
there is no `seed_everything` call and no `seed` config key, so the original ARM V run drew
PyTorch's default non-deterministic seed and **did not record it**. The three runs are therefore:

| run | training seed | provenance |
|---|---|---|
| **ARM V seed U** (the original, §3/§5) | unset, unrecorded (torch default entropy) | ckpt md5 `8b80904a099adafe8e3fe22a8a557f6b`, run dir `2026-08-14_13-47-17` |
| **ARM V seed 1** (new) | `seed_everything(1)` | this session |
| **ARM V seed 2** (new) | `seed_everything(2)` | this session |

The seeding is applied by a 10-line wrapper, `~/work/w3/train_seeded.py`, which calls
`pytorch_lightning.seed_everything(int($W3_SEED))` and then execs `train.py` with the unchanged
argument list. `train.py` itself is not edited. cuDNN non-determinism is left as it is in the
recipe, so the spread measured below is **run-to-run variance of the shipped recipe**, which is
exactly the quantity the interpretation rules need — it is not a claim that seed 1 is reproducible
bit-for-bit.

**Readout:** DEV-20 per-crop precision at the incumbent's matched emit, for all three ARM V runs
(seed U's is on record at **0.8293**). Let `spread = max - min` across the three.

**Registered interpretation, fixed now:**

* `spread >= 0.016` -> the H1 head-to-head delta (ARM P - ARM V = **-0.0160**) is **DOWNGRADED to
  within-seed-noise**, and claim `v8-w3-015`'s H1 verdict (and `v8-w3-014`'s) carries that caveat.
* `spread < 0.016` -> **H1 stands as measured.**
* `spread >= 0.0051` -> ARM V's own null vs the incumbent (**-0.0051**) is "indistinguishable from
  the incumbent at seed noise", which **strengthens** the null claim rather than weakening it, and
  will be said plainly in those words.
* `spread < 0.0051` -> the null is a measured non-zero difference smaller than the noise floor is
  able to explain, and is reported as such.

No other quantity gates anything. The three-run spread is n=3; it is a coarse range, not a variance
estimate with an interval, and will be reported as a range.

### 10.2 Q2 — ARM P100, the purity/volume deconfound

**ARM P100 = the 0.99 legibility tier with every per-group cap raised by a common multiplier until
the total row count matches ARM V's 95,184 pre-torso rows to within +/-10%.**

Computed before training by `~/work/w3/p100.py --stage cap`, which drives `build_w3_corpus.build_rows`
unmodified (same tier logic, same `cap_per_group(random_state=0)`, same dedup, same holdout filter)
with the caps multiplied. **The full scan, recorded here in advance:**

| mult | cap j2023 / GSR / tier-A | cap v3 | rows | jersey-2023 | v3 | GSR | Sid | ratio to ARM V |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (= ARM P) | 60 | 12 | 58,059 | 37,098 | 18,375 | 1,931 | 655 | 0.610 |
| 2 | 120 | 24 | 79,399 | 58,278 | 18,535 | 1,931 | 655 | 0.834 |
| 3 | 180 | 36 | 91,120 | 69,999 | 18,535 | 1,931 | 655 | 0.957 |
| **4 (CHOSEN)** | **240** | **48** | **97,347** | **76,226** | **18,535** | **1,931** | **655** | **1.0227** |
| 5 | 300 | 60 | 100,585 | 79,464 | 18,535 | 1,931 | 655 | 1.057 |
| 6 | 360 | 72 | 102,367 | 81,246 | 18,535 | 1,931 | 655 | 1.076 |
| 8 | 480 | 96 | 103,853 | 82,732 | 18,535 | 1,931 | 655 | 1.091 |
| 10 / uncapped | 600 / inf | 120 / inf | 104,062 | 82,941 | 18,535 | 1,931 | 655 | 1.093 |

**Cap chosen: x4 (240 per jersey-2023 tracklet, 240 per GSR track, 240 per tier-A tracklet, 48 per
v3 game x number) -> 97,347 rows = 1.0227x ARM V.** Rule used: the in-band multiplier minimising
`|rows - 95,184|` (x4 is off by 2,163; x3 by 4,064). Volume-match is achievable — the uncapped 0.99
tier reaches 104,062 rows, 1.093x ARM V — so the brief's fallback branch does not apply.
`~/data/w3corpus/p100_cap.json`. Training seed for P100: **`seed_everything(1)`**, i.e. one of the
two seeds Q1 measures, so Q1's spread is the yardstick for reading Q2's delta.

**Three limits of P100, declared before it is trained:**

1. **Total volume is matched; the source mixture is not, and cannot be.** v3 at the 0.99 tier holds
   only 18,535 crops in total, so raising its cap adds 160 rows and nothing more. All the recovered
   volume comes from jersey-2023 (37,098 -> 76,226). P100 is 78.3% jersey-2023 / 19.0% v3 against
   ARM V's 51.0% / 44.6% (pre-torso row shares). Since S2 measured v3 as the dominant source of the
   reader's gain
   (arm 2 -> arm 4 = +0.3078), **a P100 loss cannot be attributed to purity alone** — it is
   "purity at matched row count with the mixture that the purity tier permits".
2. **Matched rows are not matched diversity.** P100 reaches volume by sampling up to 240 crops per
   jersey-2023 tracklet instead of 60, over 831 tracklets rather than ARM V's 926, so its rows are
   more correlated. This is unavoidable: deeper per-tracklet sampling is the *only* way to
   volume-match a purity tier, and it is the honest price of the deconfound.
3. **Vocabulary is 82 numbers, as ARM P's was** (the tier costs the same 7). §5.1 already measured
   that this is not the driver of ARM P's loss (identical discordant counts with the one affected
   DEV-20 number removed), so it is recorded, not re-litigated.

**Readout:** DEV-20 per-crop precision, and the **head-to-head paired exact McNemar against ARM V
seed U** (the same pairing `H1_P_vs_V.json` used: `--ckpt-a` = ARM V, `--ckpt-b` = P100). P100 vs
the incumbent is run as context, not as a gate.

**Registered interpretation, fixed now:**

* **P100 < ARM V at p < 0.05** -> **purity itself is harmful**: filtering at 0.99 discards useful
  hard examples, and the volume it costs was not the mechanism.
* **P100 ~ ARM V (p >= 0.05)** -> the ARM P loss was a **volume artifact**, and `v8-w3-015` /
  `v8-w3-014`'s "purity loses" is softened to **"purity buys nothing at matched volume"**.
* **P100 > ARM V at p < 0.05** -> reported **loudly**; it revives a purity direction, and anything
  built on it requires fresh registration in a later session. Nothing is shipped on it here.

### 10.3 Rung 0 for P100, and what else is fixed

P100 passes the same mechanical integrity check as §2 before it is trained (`p100.py --stage rows`
calls `build_w3_corpus.rung0`): zero rows from a `valid` / `test` / `challenge` path, zero rows from
any of the 12 held-out sequences (§1.3), zero empty labels, zero missing files. Torso RoIs come from
the **same shared content-addressed cache** as ARM P / ARM V (109,316 entries already written), so
crops the two earlier arms converted are not reconverted and P100's geometry is the geometry the
other arms were trained on. `last.ckpt` selection, 25 epochs, batch 128, shipped-Koshkina init,
95/5 split at `seed=0` — all unchanged from §1.4.

**Budget and stop rule:** ~2 GPU-h on GPU 1 of `a100server1`, occupancy checked immediately before
every launch, everything under `tmux`. Order of work is fixed now so a budget stop is not a
selection: P100 torso/LMDB, then **ARM V seed 1, ARM V seed 2** (Q1 first — it is the binding
unknown), then **P100**, then the scoring passes in the order Q1-seed-1, Q1-seed-2, Q2-vs-ARM-V,
Q2-vs-incumbent. Anything not reached is reported as a budget stop with the GPU-h spent, never
dropped.

---

## 11. Follow-up results

*The brief's "§6". Everything registered in §10 was run, in the registered order, and nothing else
was run. Ladder state is unchanged: rung 2, rung 3 and TEST-38 are still NOT RUN, no GS-HOTA number
exists, `METRICS_VERSION` unchanged, the incumbent `parseq_v6_arm4t.ckpt` is still the reader.*

> **RESULT: both declared limits close, and they close in opposite directions.**
> **Q1: the seed spread is 0.0059** — smaller than the 0.016 H1 delta (so **H1 stands as
> measured**) and larger than ARM V's 0.0051 null (so the null is **indistinguishable from the
> incumbent at seed noise**, which strengthens it).
> **Q2: purity itself is harmful.** At matched volume the 0.99 tier still loses to ARM V,
> **-0.0255, McNemar p = 0.0020** — the earlier purity loss was *not* a volume artifact.

### 11.1 Harness validity, checked in every pass

All four scoring passes re-scored the incumbent on `~/data/dev20` and every one returned
**emit 0.2761, precision 0.8344, 1,144 / 1,371 correct** — S2 §3.5's on-record numbers, identical
across passes to four decimals. No pass is void. Both checkpoints under comparison decode the same
1,397 gated crops in the same order, so all emit rates are 0.2761 by construction and the registered
matched-emit floor is 0 in every case, as it was in §5.

### 11.2 Q1 — seed variance of ARM V: spread = 0.0059

Three runs, identical corpus (LMDB md5s of §10.1, not rebuilt), identical recipe, identical command
line, differing only in the training RNG. Each was scored against the incumbent on DEV-20.

| ARM V run | training seed | wall clock | ckpt md5 | reads | DEV-20 precision | correct | vs incumbent | McNemar (cand / inc) |
|---|---|---|---|---:|---:|---:|---:|---|
| **seed U** (original, §5) | unset | 22 m 59 s | `8b80904a099adafe8e3fe22a8a557f6b` | 1,371 | **0.8293** | 1,137 | -0.0051 | 38 / 45, p = 0.51 |
| **seed 1** | 1 | 23 m 27 s | `6227f9f9b4a2b856d74194da52250a55` | 1,371 | **0.8352** | 1,145 | +0.0008 | 43 / 42, p = 1.00 |
| **seed 2** | 2 | 23 m 00 s | `e4a830da571716c0f7f70cf76f906e8a` | 1,371 | **0.8293** | 1,137 | -0.0051 | 42 / 49, p = 0.53 |

> **spread = max - min = 0.8352 - 0.8293 = 0.0059** (8 correct crops out of 1,371).

Applying §10.1's registered rules, both branches, in the order they were registered:

1. **`spread = 0.0059 < 0.016` -> H1 STANDS AS MEASURED.** The ARM P - ARM V delta of **-0.0160**
   is **2.7x** the full range of three same-recipe retrains, so it is not explained by training-seed
   noise. `v8-w3-014`'s H1 verdict needs no downgrade; the "single seed, unmeasured variance"
   caveat that §7 item 4 attached to it is **retired and replaced by a measurement**.
2. **`spread = 0.0059 >= 0.0051` -> ARM V's null is indistinguishable from the incumbent at seed
   noise.** Said plainly, as registered: **the -0.0051 gap between ARM V and the shipped reader is
   smaller than the range you get by retraining ARM V three times with nothing changed at all.**
   This **strengthens** `v8-w3-015`. The W3 corpus is not "slightly worse than the incumbent"; on
   this instrument it is **not distinguishable from it**, and the p = 0.51 McNemar now has a
   physical scale to go with it. Note the coincidence that makes the point concrete: seed 2 lands
   on exactly the incumbent-relative delta seed U did (-0.0051) while seed 1 lands on +0.0008 — the
   sign of the ARM V - incumbent difference is not stable across retrains.

**Corpus-val accuracy at epoch 24, for the record and not as a result:** seed U 93.02, seed 1 92.63,
seed 2 < 92.61 (its epoch 24 fell outside its own top-3 checkpoint list, so only a bound is
recoverable). The 0.4 pp corpus-val spread is an independent, coarser sighting of the same noise.

**What this does not license.** n = 3 is a range, not a variance estimate with an interval; 0.0059
is a *point estimate of the range*, and a fourth run could widen it. It is measured on one corpus
(ARM V) at one recipe, and there is no reason to assume a smaller corpus (ARM P at 58 k rows) has
the same noise floor. Two of the three runs are explicitly seeded and one is not (§10.1), so the
three are independent draws rather than a designed seed sweep.

### 11.3 Q2 — ARM P100, the deconfound: purity itself is harmful

**Corpus, built to the §10.2 cap and rung-0 clean:** 97,347 pre-torso rows (**1.0227x** ARM V's
95,184, inside the +/-10% band), torso survival 0.9343 -> **86,400 train / 4,547 val** against ARM V's
83,812 / 4,411 (**+3.1%** trainable rows — P100 is, if anything, marginally the *larger* corpus).
Integrity checks all zero: non-train-split rows 0, held-out-sequence rows 0, empty labels 0, missing
files 0 (`rung0_p100.json`). 28,303 new torso RoIs were converted into the shared cache; the other
62,644 rows reused the RoIs ARM P / ARM V trained on. Sources: jersey-2023 70,327, v3 18,154,
GSR-train 1,842, Sid 624; 82 distinct numbers.

| comparison | precision | delta | 95% Wilson | McNemar (cand-only / other-only) | p |
|---|---:|---:|---|---|---:|
| ARM V seed U (the registered `--ckpt-a`) | 0.8293 | — | [0.8085, 0.8483] | — | — |
| **ARM P100 vs ARM V** | **0.8038** | **-0.0255** | [0.7819, 0.8240] | **44 / 79** | **0.0020** |
| ARM P100 vs incumbent (context) | 0.8038 | -0.0306 | [0.7819, 0.8240] | 43 / 85 | 0.00026 |

> **Registered branch taken: P100 < V at p < 0.05 -> PURITY ITSELF IS HARMFUL.** Filtering the
> corpus at legibility 0.99 discards useful hard examples; the volume it cost was **not** the
> mechanism of ARM P's loss. `v8-w3-014`'s "purity loses" is **upheld and hardened**, not softened —
> the softening branch ("purity buys nothing at matched volume") did not fire.

The verdict is robust to which ARM V seed it is read against: P100's 0.8038 sits **0.0255 below the
lowest of the three ARM V runs**, i.e. **4.3x the §11.2 seed spread**, so no reachable seed of ARM V
would reverse it.

**The direction of the residual is the surprise.** Restoring the volume did not partially recover
ARM P's loss — P100 (-0.0306 vs the incumbent) is *further down* than ARM P (-0.0211), by 0.0095
against a 0.0059 seed spread. **This is an observation, not a registered comparison:** P100 and ARM P
were never paired against each other in a McNemar pass, the gap is only ~1.6x the seed range, and
nothing here is claimed from it beyond "the extra 28 k purity-tier rows did not buy back any of the
purity loss". If the tier's problem is that it removes hard examples, then sampling *more* crops
from the same easy tracklets is the one thing that cannot fix it, and that is consistent with what
the number does — but the session did not test that mechanism and does not claim it.

**The §10.2 limits still bind on the interpretation.** Volume is matched; the mixture is not
(post-torso, P100 is 77.3% jersey-2023 / 20.0% v3 against ARM V's 49.3% / 46.3%),
because the 0.99 tier holds only 18,535 v3 crops in total. A share of the -0.0255 belongs to the
thinner v3 support, which S2 measured as the dominant source of the reader's accuracy. The honest
statement of the finding is therefore: **at matched row count, the best corpus the 0.99 tier can
build is significantly worse than the 0.7 tier's** — which is the practically relevant question,
because that is the only corpus the tier can actually deliver.

### 11.4 What happened to the earlier claims

| claim | before | after this session |
|---|---|---|
| `v8-w3-014` (H1 refuted, purity -0.0160) | verdict + "single seed, variance unmeasured" caveat | **verdict stands, caveat retired**: 0.0160 is 2.7x the measured seed spread; and the P100 arm removes the confound caveat's escape hatch |
| `v8-w3-015` (ARM V null, -0.0051) | null with unmeasured noise floor | **strengthened**: the delta is *below* the seed spread — ARM V is indistinguishable from the incumbent, not slightly worse |
| `v8-w3-016` (rung-1 contamination) | untouched | untouched; this session read DEV-20 only |
| `v8-w3-017` (cap, not crop count, binds) | untouched | reinforced in passing: raising the cap 4x converted 37,098 -> 76,226 jersey-2023 rows exactly as its arithmetic predicted |
| **new `v8-w3-018`** | — | the seed-variance measurement (spread 0.0059 over three ARM V retrains) |
| **new `v8-w3-019`** | — | the purity/volume deconfound (P100 -0.0255 vs ARM V at matched volume, p = 0.0020) |

### 11.5 Spend, artifacts, and what was deliberately not done

**Cluster spend: ~1.29 GPU-h on GPU 1** (P100 torso 5 m 35 s; three trainings 23 m 27 s + 23 m 00 s
+ 23 m 30 s = 69 m 57 s; four DEV-20 scoring passes 18 s each = 1 m 12 s; one aborted scoring launch
~30 s). LMDB writing and the cap scan were CPU. GPU 0 (585 MiB, another user) was never touched;
GPU 1 was verified at 4 MiB immediately before each of the three launches and is at 4 MiB at session
end. No two GPU jobs ever overlapped. Budget was ~2 GPU-h.

**One thing broke and is recorded:** the first scoring launch failed instantly on all arms with
`InvalidModelError: Unable to find model class for '.../armV_seed1.ckpt'` — PARSeq's
`strhub.models.utils._get_model_class` infers the architecture from a **substring of the checkpoint
path**, and the pulled-back copies were not named `parseq*`. Renaming the copies fixed it; no
harness code was touched. Anyone copying a PARSeq checkpoint out of its `outputs/parseq/...` tree
must keep `parseq` in the filename.

**Checkpoints (gitignored, pulled back, md5 verified on both sides):**
- `outputs/gsr/w3_readers/parseq_w3_armV_volume_seed1.ckpt` — md5 `6227f9f9b4a2b856d74194da52250a55`,
  server origin `~/data/w3out/ckpt/parseq_armV_seed1.ckpt` = `.../outputs/parseq/2026-08-14_17-54-22/checkpoints/last.ckpt`
- `outputs/gsr/w3_readers/parseq_w3_armV_volume_seed2.ckpt` — md5 `e4a830da571716c0f7f70cf76f906e8a`,
  origin `.../2026-08-14_18-17-50/checkpoints/last.ckpt`
- `outputs/gsr/w3_readers/parseq_w3_armP100_purity_volmatched_seed1.ckpt` — md5
  `ee62d21a6a68d65a1bff86f8d14cbe0e`, origin `.../2026-08-14_18-40-51/checkpoints/last.ckpt`

**Result JSONs (gitignored, `outputs/gsr/w3_readers/`):** `fu_dev20_armV_seed1.json`,
`fu_dev20_armV_seed2.json`, `fu_P100_vs_armV.json`, `fu_dev20_armP100.json`, `p100_cap.json`,
`rung0_p100.json`, `corpus_stats_p100.json`.

**Server-side additions (`~/work/w3/`, nothing vendored):** `p100.py` (drives `build_w3_corpus.py`
with raised caps; does not modify it), `train_seeded.py` (10-line `seed_everything` wrapper around
the stock `train.py`), `run_p100_corpus.sh`, `run_w3fu_train.sh`, `run_w3fu_score.sh`. Data:
`~/data/w3corpus/armP100/` (327 MB), `~/data/w3out/ckpt/` (1.1 GB), logs `~/logs/w3fu_*.log`,
`~/work/w3/{p100_corpus.log, fu_chain.log, fu_score.log}`.

**Not done, on purpose:** no rung 2, no rung 3, no GS-HOTA, no TEST-38 / test-49 / challenge read,
no operating-point sweep, no seed-vs-seed McNemar pairing (the registered Q1 readout is the
precision spread; pairing two ARM V seeds against each other was not registered and was not run), no
third repeat of ARM P or P100, and no tier between 0.7 and 0.99 (§7 item 7 stands unmeasured).
Nothing shipped.
