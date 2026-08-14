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
4. **Single seed, single recipe.** One `last.ckpt` per arm, no repeats, no variance estimate across
   training runs. The DEV-20 deltas (-0.005, -0.016, -0.021) are of the same order as run-to-run
   PARSeq variance plausibly is, and that variance is **unmeasured** in this project. The McNemar
   p-values quantify crop-level pairing, **not** training-seed noise; a -0.016 at p = 0.043 on one
   seed is directional evidence, not a settled ordering.
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
