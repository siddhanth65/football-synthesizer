# v7 session V2 — the side-aware overlay detector: training + component gate (2026-08-07)

Campaign v7's novel swing. `results/GSR_TEAMSIDE.md` closed the *positional* family: `meanx` is at
its own oracle ceiling (112/115 on ground-truth positions), the residual failures are clips whose
ground-truth geometry inverts, and the only rule that clears them (`gk_self`, 113/113 on GT) needs a
keeper-to-team link that two-way kit clustering does not supply. This session asks whether a
**learned** side assignment — a detector fine-tuned on SoccerNet-v3's discarded `team left` /
`team right` labels and run as an overlay on the shipped v6 detections — supplies that bit.

**Headline: it does not, and the reason is measurable. The overlay scores 91/97 = 0.9381 on the dev
side-pool — the same 91/97 = 0.9381 that the free incumbent `meanx` scores on the identical pool —
and it is WRONG on two of the three ground-truth-inversion clips it was built for (SNGS-092 and
SNGS-111), with healthy vote shares of 0.7123 and 0.7613. It agrees with `meanx` on 89 of 97
sequences; where they differ it fixes exactly the four clips where `meanx`'s margin is small
(0.01–1.84 m, i.e. `meanx`'s estimation errors) and breaks exactly four where `meanx` is confident.
It is a second estimator of the same quantity, not a second signal. Per the pre-declared rule this
is a KILL: the component is dropped and the V2 fusion session does not happen.**

The negative is sharp rather than mushy: the model is converged (val mAP@0.5 flat at ~0.822 from
epoch 17 of 30), it learned something real (per-box side accuracy 0.7713 in-domain against a
pure-image-x shortcut worth only 0.5983 on the same corpus), and it still inverts exactly where
pitch geometry inverts. Two side effects are worth banking: splitting the classes costs box quality
(person AP@0.5 0.9723 -> 0.9596, player 0.9669 -> 0.9503), and the learned goalkeeper side
classes independently reproduce `GSR_TEAMSIDE.md` §6's keeper/kit-cluster inversion rate to within
three points (0.7216 here vs 0.756 there) from a completely different signal path.

---

## 0. PRE-DECLARED COMPONENT GATE (rung 1 of the plan's 4-rung ladder)

Written **before** the overlay was run on any GSR sequence.

**Pool.** The dev side-pool of `tools/gsr_teamside.py::dev_names` — **97 sequences**: valid-58 plus
the 39 extracted non-degenerate train-probe sequences. The official test split is not read.

**Procedure.** Run the side-aware overlay on every frame of each sequence; match each overlay box to
our cached v6 positions rows (which carry the kit cluster `team`); aggregate one side vote per
matched row per kit cluster; take the cluster -> side assignment that maximises vote agreement;
compare with the ground-truth cluster -> side map (`eval.gsr_score.resolve_team_map`, the same
oracle map `gsr_teamside.py` grades against).

**PASS requires BOTH:**

- **(1a)** per-sequence side accuracy over the 97-sequence pool **>= 0.94** — `meanx`'s own dev
  level (63/67 = 0.9403 in `GSR_TEAMSIDE.md` §3); and
- **(1b)** a **correct** verdict on **all three** dev ground-truth-inversion clips — **SNGS-038,
  SNGS-092, SNGS-111**. These are the clips whose GT geometry inverts, i.e. the only clips where a
  learned appearance signal can beat the positional ceiling.

**KILL rule, as pre-declared in the plan:** missing the inversion clips means the model learned
geometry/depth again, in which case the component is killed, this document reports the negative,
and **the V2 fusion session never happens**. A pass on (1a) alone with (1b) failed is still a KILL —
an overlay that merely reproduces `meanx` is worth nothing, because `meanx` is free.

**Judged on `last.pt`** (final epoch). Per-epoch checkpoints are diagnostics and cannot open the
gate — the S4/S4b anti-selection rule, carried forward.

**Secondary numbers, reported regardless and gating nothing:** per-class detection quality on the
v3 holdout (did splitting the classes hurt box quality?), the v3 per-box side accuracy in-domain,
the training curve, wall time, GPU and disk state.

---

## 1. The dataset — the side labels v3 always had and S4 threw away

`~/work/det/build_side_labels.py` (new; the S4 builder `build_det_data.py` is untouched and its
`train_v3` split is untouched). Six classes, the plan's "8-class" list minus its two DROP rows:

| v3 class | boxes | -> side class |
|---|---:|---|
| Ball | 26,376 | 0 `ball` |
| Goalkeeper team left | 10,944 | 1 `gk_left` |
| Goalkeeper team right | 10,063 | 2 `gk_right` |
| Player team left | 146,496 | 3 `player_left` |
| Player team right | 146,000 | 4 `player_right` |
| Main + Side referee | 26,202 | 5 `referee` |
| **mapped** | **366,081 = 98.52%** | |
| Staff members | 4,955 | DROP |
| Wall of players | 317 | DROP (group box) |
| Referee flag / Yellow card / Red card | 246 | DROP (objects) |

Every count reproduces `CLUSTER_SESSION_S2.md` §1.1 and `CLUSTER_SESSION_S4.md` §1 to the box, and
collapsing the side reproduces S4's `V3_MAP` exactly (asserted in the builder's `--selfcheck`).

**No frames were re-extracted.** `images/train_v3` already holds all 33,986 stills pre-scaled to the
training long side, and YOLO labels are normalized, so the new labels sit beside a per-file symlink
farm pointing at the same JPEGs. Box pixels come from `Labels-v3.json` and the image size from its
own `imageMetadata`, so no zip was opened and no frame decoded: **2.5 min on 24 workers, +0.3 GB.**

| split | games | frames | ball | gk_left | gk_right | player_left | player_right | referee |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_v3s` | 380 | **32,532** | 25,237 | 10,478 | 9,642 | 140,207 | 139,745 | 25,131 |
| `valid_v3s` (holdout) | 20 | **1,454** | 1,139 | 466 | 421 | 6,289 | 6,255 | 1,071 |

Holdout = every 20th game of `gamelist.txt`, chosen by index before anything was trained. The same
holdout images are also written with S4's 4-class labels (`valid_v3s4`) so the S4b checkpoint can be
scored on identical pixels (§3).

**GSR train frames are NOT in this corpus.** GSR carries side attributes, but the dev side-pool
graded in §4 contains 39 GSR *train* sequences, so training on them would grade the model on its own
labels. v3-only is the only leak-free option and it is what was run.

### 1.1 The number that prices the experiment, measured before any GPU time

Over the 19,533 v3 frames carrying at least 3 `team left` and 3 `team right` player boxes, how often
is the left team's mean image-x actually smaller than the right team's — i.e. how far does a pure
image-geometry shortcut get on the training corpus?

| subset | n frames | image-x shortcut accuracy |
|---|---:|---:|
| all | 19,533 | **0.5983** |
| action frames | 11,231 | 0.6399 |
| replay frames | 8,302 | 0.5420 |
| holdout games | 813 | 0.6150 |

**The crudest geometric shortcut is worth only 0.60 on this corpus** — v3 stills are tight action
and replay crops where the visible players' mean x says almost nothing about which goal a team
defends. So the training signal is *not* degenerate: a model that reaches 0.77 per box (§3) has
learned something beyond mean image-x. What it has not learned is anything independent of pitch
geometry, which is what §4 measures.

## 2. Training

`~/work/det/train_det_ft.py` (S4b's script, plus two new pass-through arguments), init
**`~/runs/det/gsr_v3_ft_b/weights/last.pt`** (S4b, md5 `2074d8741f97a6892a1322d0f808244c`), data
`gsr_v3side_det.yaml`, imgsz 640, batch 64, SGD lr0 **0.002** / lrf 0.05 / momentum 0.937 / wd
0.0005, warmup 1.0, close_mosaic 3, **hsv_h 0.3 / hsv_s 0.9 / hsv_v 0.6**, seed 0, GPU 1 —
the S4b recipe. Ultralytics reported `Remapped 2/6 cls head rows from pretrained weights by class
name` (`ball` and `referee` transfer; the four side classes are fresh) and `Transferred 355/355
items`.

**Two declared deviations from S4b, both stated before the run:**

1. **`fliplr=0.0`** (S4b and ultralytics default 0.5). This is not a preference, it is a
   correctness requirement: a horizontal flip inverts left/right semantics and ultralytics has no
   paired-class swap for detection (only `flip_idx` for pose keypoints), so leaving it on would
   corrupt the label on the exact bit being learned, in half the samples. Mosaic was left at S4b's
   setting — it randomises where a box lands on the canvas and so removes the crudest absolute
   position shortcut, which is the right direction for this task.
2. **30 epochs, not ~10.** The head is re-initialised for 6 classes and the corpus is 2.4x smaller
   per epoch than S4b's GSR+v3 union, so 10 epochs here would be ~4 S4b-epochs of gradient steps.
   30 epochs = 15,270 iterations against S4 + S4b's cumulative 23,980. Still judged on `last.pt`.

**57.8 min wall (0.964 GPU-h)**, 509 iters/epoch, ~115 s/epoch. Loader: 32,532 images, 0 corrupt.

| epoch | 1 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| box | 0.978 | 0.975 | 0.964 | 0.954 | 0.947 | 0.939 | 0.931 | 0.922 | 0.914 | 0.910 | **0.869** |
| cls | 1.158 | 0.946 | 0.866 | 0.823 | 0.795 | 0.767 | 0.744 | 0.719 | 0.695 | 0.675 | **0.565** |
| dfl | 0.941 | 0.939 | 0.935 | 0.934 | 0.931 | 0.927 | 0.925 | 0.921 | 0.920 | 0.917 | **0.907** |
| holdout mAP@0.5 | .686 | .739 | .796 | .799 | .812 | .816 | .821 | .823 | .822 | **.823** | .819 |

**The run is converged.** Holdout mAP@0.5 is flat within 0.003 from epoch 17 to epoch 30 (0.821 →
0.819, peak 0.8230 at epoch 21/23/27). Losses keep falling — the step at epoch 28 is `close_mosaic`
— but the held-out metric does not move. This matters for how the §4 negative is read: unlike S4's
FAIL, this one cannot be explained as a dose shortfall. `best.pt` is epoch 27 (0.82261) against
`last.pt`'s 0.81945, a 0.003 difference on a split that is not the graded one; `last.pt` is used
throughout regardless.

## 3. Did splitting the classes hurt box quality? — the v3 holdout

Both checkpoints scored on the **same 1,454 holdout images** through `~/work/det/side_holdout_eval.py`,
with predictions and ground truth merged to a common taxonomy so a 6-class and a 4-class model
compare like for like (`gk_left|gk_right -> goalkeeper`, `player_left|player_right -> player`,
everything non-ball -> `person`). AP@0.5, all-point interpolation.

| merged class | GT boxes | S4b (4-class) | **side (6-class)** | delta |
|---|---:|---:|---:|---:|
| **person** (class-agnostic) | 14,502 | 0.9723 | **0.9596** | **-1.27** |
| player | 12,544 | 0.9669 | 0.9503 | -1.66 |
| goalkeeper | 887 | 0.8823 | 0.8529 | -2.94 |
| referee | 1,071 | 0.9350 | 0.9311 | -0.39 |
| ball | 1,139 | 0.7277 | 0.7238 | -0.39 |

At the pipeline's `conf 0.20`: person precision 0.9113 -> **0.8301** (-8.12), person recall 0.9563 ->
**0.9528** (-0.35). **Splitting the classes costs box quality** — most of it precision, from the
extra confusable class pair. Small in AP terms, but it is the price the overlay would have to repay
in the fusion, and it is one more reason the KILL is not a close call. (This says nothing about the
shipped detector: v6 keeps S4b untouched, exactly as the plan required.)

### 3.1 The side bit in-domain — 0.7713 per box

Among 12,799 holdout detections matched to a sided GT box at IoU >= 0.5 and conf >= 0.20:

| GT -> predicted | count |
|---|---:|
| left -> left | 5,004 |
| right -> right | 4,868 |
| right -> left | 1,500 |
| left -> right | 1,404 |
| left/right -> unsided class | 23 |
| **accuracy** | **9,872 / 12,799 = 0.7713** |

Comfortably above the 0.5983 image-x shortcut of §1.1 and comfortably below anything usable per box.
The errors are near-symmetric (1,500 vs 1,404), so there is no global left/right bias — the model is
locally uncertain, and §4 shows the residual uncertainty is correlated with frame geometry rather
than random, which is exactly why sequence-level aggregation does not rescue it.

## 4. THE COMPONENT GATE — the dev side-pool, 97 sequences

Overlay run on all **72,750 frames** of the 97 sequences at `imgsz 640`, `conf 0.20`, producing
**1,262,514 boxes** (`~/work/det/side_infer_gsr.py`, 26.6 min of GPU predict). Graded on the laptop
by `tools/gsr_sideoverlay.py`.

**Matching validated first.** Overlay boxes are matched to cached positions rows by foot point at
the repo's own tolerance `max(0.5 * box width, 12 px)`. Feeding the cached v6 boxes back in as
"overlay" boxes matches 100.0% of rows on three sequences, and on the 58 valid sequences where
`detbox_cache` exists, **566,800 matched pairs have mean IoU 0.8608 and 98.61% at IoU >= 0.5** — the
foot-point match is an IoU match in practice. Match rate over the pool: mean 0.9576 (min 0.7482),
919,378 of 961,845 person rows matched, 873,883 side votes cast.

### 4.1 The verdict

| | overlay | `meanx` (incumbent, free) |
|---|---:|---:|
| **dev pool, 97 sequences** | **91 / 97 = 0.9381** | **91 / 97 = 0.9381** |
| valid-58 | 56 / 58 = 0.9655 | 56 / 58 = 0.9655 |
| train-probe 39 | 35 / 39 = 0.8974 | 35 / 39 = 0.8974 |
| wrong | 052, **092**, **111**, 156, 165, 170 | 034, **092**, 097, 105, 106, **111** |
| agreement between the two | **89 / 97** | |

| criterion (§0) | threshold | measured | verdict |
|---|---|---:|---|
| **(1a)** per-sequence side accuracy | **>= 0.94** | **0.9381** (91/97; 92 needed) | **FAIL** |
| **(1b)** the three GT-inversion clips | all three correct | **1 of 3** | **FAIL** |

**VERDICT: KILL.** Both criteria fail. The component is dropped, `eval/gsr_score.py` is untouched,
and the V2 fusion session does not happen — as pre-declared.

### 4.2 The inversion clips, specifically — the reason this is a KILL and not a near miss

| clip | split | GT side of cluster 0 | overlay verdict | vote share | side votes | `meanx` | `meanx` right? |
|---|---|---|---|---:|---:|---:|---|
| SNGS-038 | valid | left | **left — correct** | 0.7921 | 7,581 | -1.514 | yes (by luck) |
| **SNGS-092** | valid | right | **left — WRONG** | **0.7123** | 10,298 | -1.129 | no |
| **SNGS-111** | train | right | **left — WRONG** | **0.7613** | 8,462 | -5.807 | no |

The one it gets right, SNGS-038, is the one `meanx` already gets right. The two it gets wrong are
the two `meanx` gets wrong — and it gets them wrong **with 71% and 76% of a 10,298- and 8,462-vote
majority**, i.e. tens of thousands of independent box-level decisions agreeing on the wrong answer.
That is `GSR_TEAMSIDE.md` §5's signature restated in a new modality: *an inversion is not weak
evidence, it is strong evidence pointing the wrong way.* A learned overlay was the plan's candidate
for evidence that is not geometric; it turns out to be geometric evidence learned end to end.

### 4.3 Where the two estimators differ — a wash, and a diagnostic one

Eight sequences separate them:

| seq | split | GT | overlay | vote share | `meanx` (m) | who is right |
|---|---|---|---|---:|---:|---|
| SNGS-034 | valid | left | left | 0.6636 | +0.866 | **overlay** |
| SNGS-097 | train | right | right | 0.7194 | -0.010 | **overlay** |
| SNGS-105 | train | right | right | 0.6931 | -1.843 | **overlay** |
| SNGS-106 | train | left | left | 0.7660 | +0.615 | **overlay** |
| SNGS-052 | valid | left | right | 0.5329 | -4.404 | `meanx` |
| SNGS-156 | train | left | right | 0.6346 | -1.316 | `meanx` |
| SNGS-165 | train | left | right | 0.5461 | -3.779 | `meanx` |
| SNGS-170 | train | left | right | 0.5908 | -7.136 | `meanx` |

Read the `meanx` column: **every clip the overlay fixes is one where `meanx`'s margin is under 1.9 m**
(0.01, 0.62, 0.87, 1.84) — `GSR_TEAMSIDE.md`'s *estimation-error* family, the ones a margin gate
already flags. **Every clip the overlay breaks, it breaks with a vote share of 0.53–0.63**, its own
low-confidence band, while `meanx` is confident there (up to 7.1 m). Four for four in both
directions. The overlay is a noisier estimator of the same latent quantity: it helps where the
positional estimate is noisy and hurts where it is not, and neither touches the inversions.

The overlay's own confidence is no better than the incumbent's, either: **AUC of vote share as a
correctness ranking = 0.8388, against 0.8425 for `|meanx|` on the same 97 sequences.** There is no
new gate to be had here.

### 4.4 One thing worth keeping: the goalkeeper classes replicate the keeper/kit-cluster inversion

Votes restricted to the `gk_left` / `gk_right` classes (median 185 keeper votes per sequence, all 97
sequences gradable), attributed to the kit cluster of the matched positions row:

| | value |
|---|---:|
| gk-only verdict accuracy, as attributed | **0.2784** (27 / 97) |
| the same with the cluster attribution inverted | **0.7216** |
| agreement between the gk-only verdict and the full overlay verdict | 0.2165 |

`GSR_TEAMSIDE.md` §6 measured, by matching our keeper detections to GT keepers, that the two-way kit
KMeans puts a keeper in the **opponent's** cluster in 0.756 of rows and 0.772 of sequences. Here a
completely independent signal path — a learned `gk_left`/`gk_right` class rather than pitch geometry
and GT matching — recovers **0.7216** for the same phenomenon. That is a cross-modality replication
of the keeper/kit-cluster inversion, and it re-confirms the same conclusion: at ~0.72–0.76 the
keeper route is well below `meanx`'s 0.9381 and is not shippable as an override. The named upgrade
path is unchanged and it is still an appearance-linking problem.

## 5. Negatives, in one place

1. **The gate FAILS on both criteria.** 0.9381 < 0.94, and 1 of 3 inversion clips.
2. **The overlay is not additive to the incumbent.** Same accuracy (91/97 both), 89/97 agreement,
   +4 / -4 where they differ. There is no bundle in which it and `meanx` are both worth having.
3. **The failure is not undertraining.** Held-out mAP@0.5 is flat from epoch 17 to 30 of 30. More
   epochs is not the answer this time; S4's dose-shortfall diagnosis does not transfer.
4. **The failure is not "it learned nothing".** 0.7713 per box in-domain against a 0.5983 image-x
   shortcut. It learned per-frame pitch geometry — landmarks, camera orientation, depth — which is
   the same information calibration already hands `meanx`, only estimated worse.
5. **Splitting the classes costs box quality:** person AP@0.5 -1.27, person precision at conf 0.20
   -8.12 points on an identical holdout.
6. **The vote-share margin is not a better gate** than the incumbent's metre margin (AUC 0.8388 vs
   0.8425).
7. **Scope of the claim.** One architecture (YOLOv8s detection head), one supervision form (a side
   class per box), one corpus (v3 stills, 2014-17). It does **not** show that no learned side model
   can work — it shows that *this* supervision teaches per-frame geometry, because per-frame
   geometry is what a single-frame box label can express. A model that could beat the ceiling would
   need supervision the frame does not contain: keeper identity linked across a kit, or temporal
   evidence over the clip.
8. **The v3 corpus itself is weak evidence for the target task.** 61% of its frames are replays
   (21,222 of 33,986) shot from angles a broadcast main camera never uses, and the left/right label
   is a match-level attribute that a replay frame frequently cannot express at all. That is visible
   in §1.1: the image-x shortcut is 0.6399 on action frames and 0.5420 on replays.

## 6. Artifacts

**Server (`siddhanth23519@a100server1`, GPU 1 only; GPU 0 was another user's throughout):**

- **Weights (parked, server only, nothing pulled to the laptop):**
  `~/runs/det/gsr_v3side/weights/last.pt` — 22,495,907 B, md5 **`977622f81faba4b1f4a7fb91b75fb29a`**.
  `best.pt` (epoch 27) md5 `45da3f97bf7fa58d5d354e531bf8e69b`, excluded from the gate by rule.
  `epoch{0,3,6,...,27}.pt` retained; run dir 471 MB. Class ids
  `{0: ball, 1: gk_left, 2: gk_right, 3: player_left, 4: player_right, 5: referee}` — **not** a
  drop-in for `generator/extract.py`, which expects S4b's 4-class scheme.
- Data: `~/work/det/data/{images,labels}/{train_v3s,valid_v3s,valid_v3s4}` (symlink farms over the
  existing `train_v3` JPEGs), `gsr_v3side_det.yaml`, `gsr_v3side_ho4.yaml`, `side_dataset_stats.json`.
- Scripts: `~/work/det/build_side_labels.py`, `side_infer_gsr.py`, `side_holdout_eval.py`;
  `train_det_ft.py` gained `--fliplr` and `--save-period` (defaults unchanged, so S4b's invocation is
  bit-identical; `.pres4` copy kept).
- Evals: `~/work/det/v7v2_holdout_{side,s4b}.json`; overlay dumps `~/work/det/side_overlay/*.npz`
  (97 files, 22 MB).
- Logs: `~/logs/v7v2_side_{build,train,infer}.log`, `~/runs/det/gsr_v3side/{results.csv,args.yaml}`.

**Laptop / repo:**

- `tools/gsr_sideoverlay.py` — the grading harness (`--grade`, `--demo`); ruff-clean at 100 cols,
  self-check passes.
- `results/gsr_benchmark/teamside/sideoverlay_dev.json` — the full per-sequence gate table.
- `outputs/gsr/side_overlay/*.npz` — the 97 overlay dumps (22 MB), gitignored.
- This document, and one claim (`v7-v2-001`). **Nothing under `eval/`, `generator/` or
  `synthesizer/` was touched. No commits.**

### 6.1 Spend and machine state

| | |
|---|---|
| dataset build | 2.5 min, 24 CPU workers, +0.3 GB |
| training | 57.8 min = **0.964 GPU-h**, 30 epochs |
| overlay inference | 26.6 min GPU predict (48 min wall, I/O-bound on the shared filesystem) |
| holdout evals | ~4 min GPU, both checkpoints, run twice after an evaluator fix |
| **total GPU 1 spend** | **~1.5 h** of the plan's 10-14 h V2 budget |
| end state | GPU 1 released (4 MiB, 0%); GPU 0 untouched |

**Disk hygiene (the V0 §5 flag).** `~/runs` pruned of superseded intermediate epoch checkpoints
before training: **31,306 MB -> 15,714 MB, 15,592 MB (15.2 GiB) reclaimed**, inside the 15-20 GB
target. Removed: `clip_s3/epoch{0_untrained,5,10,20,30,40}.pt`, `clip_s3_e12/epoch{1..12}.pt`,
`clip_uni/epoch{2,4,6,10,12}.pt`, and `epoch{0..9}.pt` from each of `det/gsr_ft`, `det/gsr_v3_ft`,
`det/gsr_v3_ft_b`. **Kept and md5-verified after the prune:** `det/gsr_v3_ft_b/weights/last.pt`
(`2074d8741f97a6892a1322d0f808244c`, the shipped v6 detector) and `clip_uni/epoch8.pt`
(`6f27a1ee2df1278c03e7d4c18fdc9a17`, named in `results/gsr_v5_frozen.json`), plus every `last.pt`
and `best.pt`. The PRTreID run directories (11.5 GB) were **not** touched — a superseded model
family is outside the "superseded epoch checkpoints only" scope. Global filesystem free after the
session: 779 GB (93% full), `~/runs` 16 GB.

## 7. Two defects in this session's own instruments, recorded

1. **The IoU cross-check was wrong on first run** (mean IoU 0.02): `vote_sequence` returned a
   within-frame box index and `iou_crosscheck` used it as a sequence-wide index. Found because the
   number was absurd, fixed, re-run — 0.8608. **The side votes were never affected** (they read the
   per-frame arrays consistently); the fix changed no vote count on the two sequences it was caught
   on, and the gate numbers below were all produced after the fix.
2. **The holdout evaluator's class-agnostic `person` recall exceeded 1.0** on first run: referees
   were matched but excluded from the denominator. Fixed and both models re-scored; §3's table is
   the corrected run.

Both are reported because they are the kind of instrument bug that silently flatters a result, and
in this session's case the corrected instruments still say KILL.
