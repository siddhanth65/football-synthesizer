# Cluster session S0/S1/S2 — SoccerNet-v3 fetch, synthetic pretrain, jersey reader retrain

Date: 2026-08-05. Server `a100server1` (GPU 1 only, GPU 0 held by another user throughout).
Campaign v6 sessions S0, S1, S2 + the S2 confirmation run (arm 4t).

**Headline: the shipped jersey reader is beaten on its own chain.** A PARSeq fine-tune on a
legibility-filtered SoccerNet-v3 digit corpus reads **0.8344 per-crop precision at the incumbent's
own emit rate (0.2761 vs bar 0.2759)**, against the incumbent's **0.7022** — measured through an
identical harness that reproduces the incumbent's published DEV-20 numbers exactly. SoccerNet-v3
is the whole effect; the synthetic pretrain of S1 is not.

---

## 1. S0 — SoccerNet-v3 fetch + label audit

**Access route, verified by reading the SDK:** `SoccerNet/Downloader.py:1266` routes
`Frames-v3.zip` / `Labels-v3.json` through KAUST owncloud WebDAV with the credential pair
`okteXlk6jmDXNJc` / `SoccerNet_Reviewers_SDATA` **hardcoded as a public constant in the package
source**. Sid's NDA password was never used or written anywhere.

**The Session-1 landmine was pre-empted, not survived.** The SDK downloader still calls
`requests.get` with no timeout, so it was not used. `~/work/v3_fetch.py` (ours) does
HEAD-then-Range resume with per-request timeouts and size verification; range support was verified
first (`HTTP 206`).

| | result |
|---|---|
| `Labels-v3.json` | 400/400, 0 failures, 175 s |
| `Frames-v3.zip` | 400/400, 0 failures, 2,086 s (~35 min), ~21-25 MB/s |
| on disk | 45 GB, `~/data/soccernet-v3/` |

Disk gate before fetch: quota 89,385 MB / 500 GB, global free 844 GB (threshold 100 GB). Pass.

The SDK's `frames` task lists **443** games (290 train / 55 valid / 55 test / 43 challenge); the
400 labelled ones were fetched (challenge has no public labels).

### 1.1 Class inventory (400 games, 33,986 images, 371,599 boxes)

| v3 class | total | share | -> detector class |
|---|---:|---:|---|
| Player team left | 146,496 | 39.42% | 2 player |
| Player team right | 146,000 | 39.29% | 2 player |
| Ball | 26,376 | 7.10% | 0 ball |
| Main referee | 14,720 | 3.96% | 3 referee |
| Side referee | 11,482 | 3.09% | 3 referee |
| Goalkeeper team left | 10,944 | 2.95% | 1 goalkeeper |
| Goalkeeper team right | 10,063 | 2.71% | 1 goalkeeper |
| Staff members | 4,955 | 1.33% | DROP |
| Wall of players | 317 | 0.09% | DROP (group box, not a person) |
| Referee flag / Yellow card / Red card | 246 | 0.06% | DROP (objects) |

**Mapping onto `{0: ball, 1: goalkeeper, 2: player, 3: referee}` keeps 98.52% of boxes.**
Ball supervision is dense enough to need no upsampling: 26,346 images carry exactly one ball box,
15 carry two, 7,625 none (**77.6% coverage**).

Leagues: EPL 87, UCL 94, LaLiga 79, Serie A 66, Ligue 1 37, Bundesliga 37.
Boxes/game min 2 / median 934 / max 1,697. Box height median 91 px; 4.51% below the 30 px filter.

**Two facts that change S4's setup.** 67.42% of v3 images are **1280x720**, only 31.57% are
1920x1080 (GSR is uniformly 1080p) — the domain gap is resolution as well as era (2014-17 vs
2019). And v3 is **stills, not video**: 33,986 action/replay frames. It buys breadth, not
temporal data.

Full audit table: `~/logs/v3_audit.md`.

### 1.2 The `ID` field is the jersey number — verified against pixels

106,591 person boxes carry a digit-valued `ID` (91 distinct values; modal 9, 7, 5, 1, 10, 4, 11, 2
— a squad-number profile). 40 crops were cut from `england_epl/2014-2015/2015-02-21 Chelsea 1 - 1
Burnley` and inspected (`~/v3_ids.png`): **every legible shirt matched its label**.

**Caveat, and it is the reason for the filter below:** labels are present on crops where the number
is *not* visible (player turned away, occluded) — the annotator carried identity across an action.
These are per-player-within-action labels, **not per-crop-legible**.

---

## 2. S1 — synthetic digit pretrain

**License recorded before use:** `github.com/mkoshkina/jersey-number-pipeline` @ `007d54e5` is
**CC BY-NC 3.0 Unported** (`license.txt`, README badge). The vendored PARSeq at `str/parseq` is
**Apache-2.0**, Copyright 2022 Darwin Bautista. Clone lives only at
`~/src/jersey-number-pipeline`; nothing vendored into this repo.

**Synthetic set** (`~/work/jersey/synth_gen.py` -> `~/data/synth_jersey`, 515 MB): **200,000
images** (180k train / 20k val, disjoint seed ranges), 529 auto-validated fonts, 4,000 GSR train
frames as background source, PARSeq LMDB format, ~5 min on 48 cores.

Three corrections made by looking at the output rather than trusting the code:

1. ~20% of the first pass was humanly unreadable (unbounded downscale, unchecked contrast).
2. Still ~18%, because of a **modelling error**: bare glyphs pasted onto grass and crowd. A number
   always sits on a *shirt*. Complex2D now composites a kit panel over the broadcast patch.
3. A closed-loop `check_legible` measures ink-vs-shirt separation on the **degraded, JPEG-decoded**
   image and rejects failures. **Reject rate 49.9%.** This biases the kept set toward milder
   degradation — the deliberate trade for a meaningful gate.

**Environment.** A separate `parseq` conda env — Koshkina's own design (`str_env = 'parseq2'`) and
forced anyway, since their 2022-era code uses `validation_epoch_end` / `EPOCH_OUTPUT` /
`optimizer_idx`, all removed in the PL 2.6.5 that `gsr` carries. py3.11 + torch 2.5.1+cu121 +
pytorch-lightning 1.9.5 + timm 0.9.5. One blocker, the same one Session 1 documented:
`lightning_fabric` imports `pkg_resources` -> `setuptools<81`.

**Gate: PASSED at 0.9826** (25 epochs, 49 min, `val_accuracy` 96.145 -> 98.26 monotone after epoch
2). Control added unprompted, because the gate alone does not price the work:

| model | exact-match | per-char |
|---|---:|---:|
| base STR parseq, step 0 (same `pos_queries` crop) | 0.7226 | 0.7551 |
| after synthetic pretrain | **0.9826** | 0.9897 |

**+25.99 points**, not the full 98. See §5.3 — S2 later judged this pretrain as an *init*, and the
verdict is negative.

---

## 3. S2 — the component gate

### 3.1 The harness, validated by exact reproduction

`outputs/gsr/jersey_head/chain_validation.parquet` supplies the bar **and** its missing
denominator:

**Bar: 1,370 reads / 4,965 numbered-GT DEV-20 crops = emit 0.2759 at precision 0.7029.**

The server-side port reads **1,370 reads, emit 0.2759, precision 0.7022** — reads and emit exact,
precision off by 0.0007 (one crop, from re-encoding torso JPEGs at quality 95).

**Bug found getting there, and it is a pipeline observation worth keeping.** The first port fed
`keypoints_scores` through a sigmoid into the `POSE_CONF = 0.4` test. The pipeline passes
`out["keypoints"][0]`, whose third channel is torchvision's **visibility placeholder (always
1.0)** — so **`generator/jersey_id.py`'s `POSE_CONF` check is dead code**: a torso RoI is produced
whenever a person box is detected. The stricter version under-emitted (1,335 vs 1,370, torso rate
0.7821 vs 0.8333). Matching verbatim closed the gap exactly. **Flagged, not fixed** — the incumbent
numbers on record were all measured with the dead check, so changing it is a separate decision.

### 3.2 The v3 guard, measured

105,702 digit-ID crops cut from the 400 games (889 below min size), streaming one game's zip at a
time so peak extra disk was ~140 MB rather than another 45 GB.

| legibility threshold | kept | share | distinct numbers |
|---|---:|---:|---:|
| >0.0 | 105,702 | 1.0000 | 91 |
| >0.5 | 49,387 | 0.4672 | 89 |
| **>0.7** | **45,620** | **0.4316** | **88** |
| >0.9 | 37,783 | 0.3574 | 88 |

**The guard is not cosmetic: at 0.7 it discards 60,082 crops — 56.8% of v3's digit labels sit on
crops with no visible number.** Both sides checked visually (`~/leg_keep.png`, `~/leg_drop.png`):
DROP is overwhelmingly front-facing, occluded or blurred; KEEP is back-facing with legible numbers.
The filter is conservative — some faint-but-readable numbers land in DROP — which is the right
direction for precision.

### 3.3 Corpora

| corpus | train | val | sources | distinct numbers |
|---|---:|---:|---|---:|
| `planned` | 40,193 | 2,115 | jersey-2023 38,133 + GSR-train 4,175 | **44** |
| `plus_v3` | 80,548 | 4,239 | + v3 42,479 | **89** |
| `plus_v3_torso` | 75,154 | 3,955 | same rows, pose RoI (93.3% survive) | **89** |

All sources legibility-filtered at 0.7 (jersey-2023: 52,135 of 201,313 sampled; GSR train: 4,175 of
13,994). DEV-20 comes from GSR *validation* and never enters a corpus.

### 3.4 The geometry bug that invalidated the first arm numbers

First scoring pass gave arm 2 precision **0.0576** — below the synthetic-only model's 0.5308. A bug
signature, not a result. Cause: the corpora are **full-body crops**, the incumbent chain feeds
PARSeq a **pose torso RoI**. Tested rather than assumed:

| model (trained on) | scored on TORSO | scored on FULL-BODY |
|---|---:|---:|
| incumbent (torso) | **0.7022** | 0.0744 |
| arm 2 (full-body) | 0.0576 | 0.4230 |
| arm 4 (full-body) | 0.2743 | **0.7330** |

A complete crossover: each reader collapses in the geometry it was not trained on.

### 3.5 Arm results

Uniform pre-declared rule: `last.ckpt` for every arm, so no arm is selected on any eval split.
All gated by the same shipped legibility model at 0.5.

| arm | init | corpus | geometry | emit | precision | dominates |
|---|---|---|---|---:|---:|:--:|
| **INCUMBENT** | — | — | torso | 0.2759 | 0.7022 | bar |
| S1 synthetic only | — | synthetic | torso | 0.2743 | 0.5308 | no |
| arm 2 | synthetic | planned | full-body | 0.2866 | 0.4252 | no |
| arm 3 | shipped | planned | full-body | 0.2866 | 0.4469 | no |
| arm 4 | synthetic | planned+v3 | full-body | 0.2866 | 0.7330 | yes* |
| **arm 4t** | **shipped** | **planned+v3** | **torso** | **0.2761** | **0.8344** | **YES** |

\* arm 4's pass was geometry-asymmetric (its reader on full-body vs the incumbent on torso). Arm 4t
is the confirmation: **same chain, same RoI, same legibility gate, same emit rate — only the PARSeq
weights differ. +0.1322 precision over the incumbent.**

**v3 is the whole effect:** arm 2 -> arm 4 is **+0.3078** and arm 3 -> arm 4 is **+0.2861**, at
identical emit. Not a vocabulary artifact — `planned` already covers 44 numbers against DEV-20's
35-37.

**Synthetic init is a negative:** 0.4252 (synthetic) vs 0.4469 (shipped) on the same corpus,
**−0.0217**. S1's 0.9826 does not transfer into a better init; arm 4t therefore uses the shipped
init.

### 3.6 Secondary gate — jersey-2023 test tracklet accuracy

Chain: sample 100 crops/tracklet -> shipped legibility >0.5 -> top 30 -> pose torso -> PARSeq ->
majority vote at `min_conf` 0.30; no qualifying crop -> `-1`. Scored over all 1,211 tracklets with
`-1` a first-class class (the sn-jersey convention).

| model | overall | numbered (856) | illegible (355) | emitted |
|---|---:|---:|---:|---:|
| incumbent (shipped) | 0.8084 | 0.7967 | 0.8366 | 820 |
| **arm 4t** | **0.8365** | **0.8364** | 0.8366 | 819 |
| Koshkina published | 0.874 | — | — | — |

**The criterion as literally stated ("≥ Koshkina's published 87.4") FAILS — but it fails for the
incumbent too (80.84), i.e. for the very model that published 87.4.** That 6.6-point shortfall is a
harness-reproduction gap, not an arm regression: this harness omits their occlusion/outlier removal
(re-ID features + Gaussian fitting) and their consolidation rule. Judged relatively on the identical
harness, **arm 4t beats the incumbent by +2.81 points overall and +3.97 on numbered tracklets**.

---

## 4. Open gaps, stated

- **Arm 1 (legibility retrain) was NOT run.** The brief's "sn-reid letters as negatives" component
  was not resolvable from verifiable assets, and an arm trained on a guess is worse than a recorded
  gap. Emit rate in every row above is still the *shipped* legibility model's; arm 1 is the lever
  that could raise it.
- The secondary gate's absolute criterion is unreachable on this harness (see §3.6).
- Arm 4t is confirmed on DEV-20 only. TEST-38 is S3's job.
- `POSE_CONF` dead code (§3.1) is flagged, not fixed.

## 5. Two upstream bugs in the Koshkina pipeline

1. **The documented training command cannot init from either published checkpoint.** Base STR *and*
   their own SoccerNet fine-tune both carry `pos_queries` at `max_label_length=25` while their
   `main.yaml` sets 2, and `train.py` strict-loads. Patched in our server-side clone only
   (`train.py.orig` kept beside it).
2. **Hydra cannot parse `pretrained=<path>` when the path contains `=`** — and their own checkpoint
   filename (`parseq_epoch=24-step=2575-...`) does. This silently killed one arm launch. Worked
   around with a symlink.

Also: their command sets `trainer.val_check_interval=1`, which as an int means *validate every
batch*; changed to `1.0`.

## 6. Artifacts

Server (`siddhanth23519@a100server1`), nothing vendored into this repo:

- Code: `~/work/v3_fetch.py`, `~/work/v3_audit.py`, `~/work/jersey/{synth_gen.py, eval_synth.py,
  v3_crops.py, legibility.py, build_corpus.py, torso_corpus.py, dev20_torso.py, dev20_arm.py,
  jersey2023_test.py, build_test_torso.py, patch_train.py, patch_train2.py, run_arms.sh,
  eval_arms.sh, run_4t.sh}`
- Results: `~/data/dev20/{arms.jsonl, fullbody.jsonl}`, `~/data/jersey-2023/test_results.jsonl`,
  `~/data/s2corpus/{corpus_stats.json, corpus_stats_torso.json}`, `~/logs/v3_audit.md`
- Logs: `~/logs/{v3_labels.log, v3_frames.log, synth_gen.log, parseq_synth.log, v3_crops.log,
  dev20_torso.log, torso_corpus.log, arm2_syn_planned.log, arm3_shipped_planned.log,
  arm4_syn_plusv3.log, arm4t_shipped_plusv3torso.log, j2023_test_torso.log}`
- **Arm 4t checkpoint (the S3 candidate):**
  `~/src/jersey-number-pipeline/str/parseq/outputs/parseq/2026-08-05_21-44-57/checkpoints/last.ckpt`
- S1 synthetic pretrain: `.../outputs/parseq/2026-08-05_15-39-40/checkpoints/last.ckpt`

End state: GPU 0 and 1 idle; quota 157 GB / 500 GB; global free 774 GB. Server GPU spend across
S0+S1+S2 ≈ 3.5 h.
