# Cluster session 5A — detector fine-tune on GSR (2026-08-05)

The first GS-DetA lever. Our GS-DetA on the official test is 24-26 vs the winners' ~48. GS-DetA is
gated by BOTH the box and the identity attributes; the box half has never seen a frame of GSR
footage. `docs/SOCCERNET_REPO_SWEEP.md` §6 established that the org ships no SoccerNet-trained
detector at all (sn-tracking's "SoccerNet" YOLOX recipe trains on MOT20+CrowdHuman), so this is
unexplored ground rather than a re-run of someone else's recipe.

**Headline: the gate is met on its literal terms and the recommendation is still NO.** Fine-tuning
buys a large, uniform box win (+9.25 points of precision at the pipeline's operating confidence,
+2.40 recall at matched precision) and simultaneously destroys the ROLE attribute on one of the
three held-out games — 47.7% of that game's players are classed "referee". Role is a GS-HOTA-gated
attribute, so shipping these weights would very likely *lower* GS-HOTA despite better boxes. The
pre-declared gate did not include role; that omission is recorded rather than papered over (§7).

A second run (§8), identical but with hue/saturation/value augmentation raised, confirms the
diagnosed mechanism — it turns the -0.53 mAP@0.5 loss into **+5.29** and cuts the leak 27%, while
still leaving role coverage 6.05 points below the control. Augmentation is not the binding
constraint; **three games of training footage is** — the same wall sessions 1-3 hit for identity on
the same corpus.

---

## 0. PRE-DECLARED GATE (written before any evaluation was run)

Recommend the laptop-side re-extract + v6 eval **iff**, on the GSR valid split, the fine-tuned
detector clears EITHER:

- **person recall at matched precision: +2.0 points** — recall read off the fine-tune's PR curve at
  the precision the CONTROL achieves at the pipeline's operating confidence (`DETECT_CONF = 0.20`),
  minus the control's recall at that same point; or
- **box mAP@0.5: +3.0 points** over the control, same evaluator, same images.

Both models are evaluated at the pipeline's inference geometry (ultralytics default `imgsz=640`,
`iou=0.7`, `max_det=300`). The control is measured FIRST and through the same evaluator. If neither
threshold is cleared, the negative is the finding and no re-extract is recommended.

**Which checkpoint the gate judges (added before the fine-tune ran, after the control was measured):
the FINAL epoch (`last.pt`).** Ultralytics selects `best.pt` by fitness on its `val` set, which here
is the GSR valid split — that is selection on the split we then report, the exact trap session 1 §8
called out. `best.pt` is reported as an upper bound only and cannot open the gate on its own.
(Moot in the event: `best.pt` was epoch 10 = `last.pt`, differing md5 only through the strip/reload.)

Secondary numbers reported either way (not gates): per-class mAP@0.5, per-sequence recall spread,
recall by ground-truth box height (the small/distant-player question), and role-class accuracy on
matched detections.

---

## 1. The detector, exactly

`generator/extract.py:275` `_FootballRoleDetector`, the `--detector football` path that every GSR
run uses (`tools/gsr_flagplant.py:105`, `eval/gsr_score.py:608` default, `eval/gsr_gta.py:276`,
`eval/gsr_jersey.py:337`).

| | |
|---|---|
| family | **YOLOv8s**, ultralytics `detect` task |
| weights | HF `uisikdag/yolo-v8-football-players-detection` -> `best.pt`, 22,494,328 B, md5 `b3417a4c3228d7f932844e4e39ec5059` |
| size | 130 layers, **11,137,148 params**, 28.6 GFLOPs |
| classes | `{0: ball, 1: goalkeeper, 2: player, 3: referee}` |
| its own training | `model: yolov8s.yaml`, data `football-players-detection-1` (a Roboflow set), **25 epochs, imgsz 800**, batch 16, SGD `lr0 0.01` / `weight_decay 0.001`, mosaic 1.0, `close_mosaic 10`, `pretrained: False` |
| **inference geometry** | `self.yolo(frame_rgb, verbose=False, conf=BALL_CONF)` passes **no `imgsz`**, so ultralytics' default **640** — the model is *inferred 160 px below the resolution it was trained at*. NMS `iou=0.7`, `max_det=300` (defaults) |
| confidence handling | one forward at `conf >= BALL_CONF = 0.10`; ball = the single top-confidence `ball` box; persons = `player`/`goalkeeper`/`referee` boxes at `conf >= DETECT_CONF = **0.20**` |

The 800-vs-640 mismatch was worth a separate control (§3) since "just infer bigger" would be a free
win if it existed.

**Role is not a cosmetic label here.** The detector's class flows untouched through
`build_player_rows` -> the positions parquet's `role` column -> `eval/gsr_score.py:120`
`"attributes": {"role": role, ...}` in the submission. `generator/postprocess.py` never rewrites it.
And `_team_side` (`gsr_score.py:79`) returns `None` for any role outside `{player, goalkeeper}`, so
a person misclassified as `referee` also loses its **team** attribute. One class error costs two
gated attributes.

## 2. The GSR detection dataset

`~/work/det/build_det_data.py`. GSR `category_id` -> the detector's own class ids
(`{1: 2, 2: 1, 3: 3, 4: 0}`), so a fine-tuned checkpoint is drop-in with zero code change. Ball
boxes are **included**: excluding them would teach the model that balls are background and destroy
the class the pipeline depends on. Category and the `attributes.role` string agree 1:1 across the
train split (audited: 173,885 `(1, player)`, 16,711 `(3, referee)`, 10,705 `(4, ball)`,
5,071 `(2, goalkeeper)`, zero disagreements), so the mapping carries no ambiguity.

| | train (stride 1) | valid (stride 5) |
|---|---|---|
| sequences | 57 | 58 |
| frames | **42,750** | **8,700** |
| player boxes | 604,968 | 123,928 |
| goalkeeper | 24,107 | 5,203 |
| referee | 61,543 | 12,381 |
| ball | 40,928 | 8,179 |
| **person GT (the eval denominator)** | 690,618 | **141,512** |
| dropped degenerate (<=1 px after clipping) | 9 | 0 |
| frames skipped (`is_labeled`/`has_labeled_person` false) | 0 | 0 |
| frames with non-empty ignore regions | 0 | 0 |

Person box heights: train median **98 px**, p05 54, p95 156, **<40 px: 10,262 (1.6%)**; valid
median 94, p05 52, p95 157, <40 px 1,785 (1.4%). GSR is not a small-object dataset at native
1920x1080 — but at the pipeline's `imgsz=640` every height divides by 3, so the median player is
~33 px and the p05 is ~18 px in the network's input.

**Valid was never trained on.** It is ultralytics' `val` set, which only evaluates. Build time
65 s; 51,450 symlinks, no image copied.

## 3. Control — measured first, through the evaluator both models share

`~/work/det/det_eval.py`. Two evaluators: ultralytics `model.val` for box mAP, and a class-agnostic
PERSON evaluator (union of `goalkeeper|player|referee` — exactly `generator/extract.py`'s own
selection) matching greedily by confidence at IoU 0.5.

| control, `imgsz=640` | value |
|---|---|
| mAP@0.5 | **0.5003** |
| mAP@0.5:0.95 | 0.2300 |
| per-class mAP@0.5 | ball 0.0581 / goalkeeper 0.5705 / player 0.8472 / referee 0.5253 |
| person precision @ conf 0.20 | **0.7966** |
| person recall @ conf 0.20 | **0.9031** |
| detections @ conf 0.20 | 160,424 (vs 141,512 GT) |
| per-sequence recall | mean 0.9027, min 0.7883, p25 0.8868, median 0.9085, max 0.9555 |

Recall by ground-truth box height (at conf 0.20):

| height | GT | recall |
|---|---|---|
| <40 px | 1,997 | **0.3996** |
| 40-70 px | 24,389 | 0.8313 |
| 70-110 px | 70,497 | 0.9172 |
| >=110 px | 44,629 | 0.9425 |

Role on matched detections: per-detection accuracy 0.9306; goalkeeper 50.9% correct, referee
**44.2%** correct (5,889 of 10,578 referees are called "player"), player 99.3%. **Role-correct
coverage of all GT persons: 0.8404.**

Precision 0.7966 means ~20% of what the pipeline accepts as a person at conf 0.20 is not an
annotated GSR person — GSR labels only on-pitch players/keepers/officials, so bench, staff and
crowd detections are pure false positives that our tracker then has to carry.

### 3a. Resolution alone is not the lever (control at `imgsz=1280`)

| control | 640 | 1280 |
|---|---|---|
| person precision @ 0.20 | 0.7966 | **0.6806** |
| person recall @ 0.20 | 0.9031 | **0.8059** |
| recall, <40 px | 0.3996 | **0.5408** |
| recall, >=110 px | **0.9425** | 0.6342 |
| per-sequence recall min | 0.7883 | 0.4274 |

Doubling the input recovers small players (+14.1 points under 40 px) and **collapses large ones**
(-30.8 points at >=110 px) — the model was trained at 800 and cannot cope with players scaled past
that. Net recall falls 9.7 points. The 800-vs-640 mismatch is real but inference-side resolution is
not a free win; 640 is the right operating point and every number below is at 640.

## 4. The fine-tune

`~/work/det/train_det_ft.py`. Gentle LR on an already-converged, already-football-tuned init — the
lesson sessions 1 and 3 paid for twice.

`imgsz 640` (the pipeline's own geometry, so a win is drop-in), batch 64, SGD **`lr0 0.002`** (5x
below the base recipe's 0.01), `lrf 0.05`, `warmup_epochs 1.0`, `close_mosaic 3`, 10 epochs,
`save_period 1`, GPU 1, seed 0. `Transferred 355/355 items from pretrained weights`.

**32.3 min wall (0.539 h)**, 668 iters/epoch at 3.8-4.2 it/s, 13.1 GB peak. Far under the 14 A100-h
the RF-DETR soccer recipe needed.

Losses descend monotonically — no warmup blow-up, unlike session 1's run 1:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| box | 1.405 | 1.298 | 1.246 | 1.210 | 1.184 | 1.162 | 1.144 | 1.106 | 1.086 | 1.069 |
| cls | 0.642 | 0.593 | 0.562 | 0.540 | 0.521 | 0.507 | 0.498 | 0.478 | 0.467 | 0.457 |
| dfl | 0.926 | 0.891 | 0.879 | 0.872 | 0.867 | 0.864 | 0.861 | 0.870 | 0.866 | 0.863 |
| valid mAP@0.5 | .466 | .481 | .475 | .475 | .478 | .495 | .489 | .492 | .493 | **.494** |

**The training curve never reaches the control's 0.5003 mAP@0.5.** Training is healthy; the metric
simply does not go where a naive reading expects, and §6 explains why.

## 5. Control vs fine-tune (`last.pt`, epoch 10)

| metric, valid 58 seq / 8,700 frames / 141,512 person GT | control | fine-tune | delta |
|---|---|---|---|
| **mAP@0.5** | 0.5003 | 0.4950 | **-0.53** |
| mAP@0.5:0.95 | 0.2300 | 0.2718 | +4.18 |
| mAP@0.5 ball | 0.0581 | 0.1927 | **+13.5** |
| mAP@0.5 goalkeeper | 0.5705 | 0.7059 | **+13.5** |
| mAP@0.5 player | 0.8472 | 0.8104 | -3.7 |
| mAP@0.5 referee | 0.5253 | 0.2708 | **-25.5** |
| person precision @ conf 0.20 | 0.7966 | 0.8891 | **+9.25** |
| person recall @ conf 0.20 | 0.9031 | 0.9126 | +0.95 |
| **person recall @ MATCHED precision (P>=0.7966)** | 0.9031 | **0.9270** | **+2.40** |
| detections @ conf 0.20 | 160,424 | 145,255 | -15,169 |
| per-seq recall mean / min / median | .9027 / .7883 / .9085 | .9177 / .6893 / .9428 | 41/58 better |
| per-seq precision | — | — | **55/58 better**, mean +9.09 |
| role per-detection accuracy | 0.9306 | 0.8090 | **-12.16** |
| **role-correct coverage of GT** | **0.8404** | **0.7384** | **-10.20** |

Recall by ground-truth box height (conf 0.20):

| height | GT | control | fine-tune | delta |
|---|---|---|---|---|
| <40 px | 1,997 | 0.3996 | 0.4266 | +2.70 |
| 40-70 px | 24,389 | 0.8313 | 0.7882 | -4.31 |
| 70-110 px | 70,497 | 0.9172 | 0.9319 | +1.47 |
| >=110 px | 44,629 | 0.9425 | 0.9719 | +2.94 |

Role confusion on matched detections (`GT -> predicted`):

| | control | fine-tune |
|---|---|---|
| goalkeeper correct | 2,276 / 4,474 = 50.9% | 3,113 / 4,939 = **63.0%** |
| referee correct | 4,677 / 10,578 = 44.2% | 7,801 / 10,120 = **77.1%** |
| player correct | 111,966 / 112,744 = 99.3% | 93,572 / 116,091 = **80.7%** |
| **player -> referee** | **645** | **20,142** |

The fine-tune is better at both minority roles and catastrophically worse at the majority one.

### 5a. The leak is not epoch-dependent

| checkpoint | person P@0.20 | person R@0.20 | R @ matched P | role-correct coverage | player->referee |
|---|---|---|---|---|---|
| control | 0.7966 | 0.9031 | 0.9031 | 0.8404 | 645 |
| epoch 1 | 0.8505 | 0.9051 | 0.9144 | 0.7306 | 20,600 |
| epoch 3 | 0.8595 | 0.9097 | 0.9199 | 0.7348 | 19,655 |
| epoch 6 | 0.8814 | 0.9168 | 0.9283 | 0.7394 | 20,544 |
| epoch 10 | 0.8891 | 0.9126 | 0.9270 | 0.7384 | 20,140 |

The leak is fully present **after one epoch** and flat thereafter. Early stopping does not buy a
checkpoint with the box gain and without the role loss.

### 5b. Per-track majority voting does not repair it

Oracle test (`~/work/det/role_vote.py`): match detections to GT, group by **ground-truth track id**
(perfect association — an upper bound on what any tracker-side vote could buy), assign every
detection of a track its majority predicted class.

| | control | fine-tune |
|---|---|---|
| role accuracy, per detection | 0.9306 | 0.8090 |
| role accuracy, per-track vote | 0.9375 (+0.70) | **0.8053 (-0.38)** |
| player -> referee after voting | 0 | **21,166 (worse)** |
| tracks | 1,347 | 1,343 |

Voting *helps* the control (+0.70) and *hurts* the fine-tune. The fine-tune's errors are
track-consistent, not per-frame flicker: it is confidently and persistently wrong about the same
people. That closes the cheapest possible fix.

## 6. Diagnosis: it is one game, both teams, and the mechanism is colour

Per-sequence `player -> referee` rate: median **0.0002**, mean 0.173, **21 of 58 sequences over
10%**, max 0.563. Bimodal, not diffuse. Grouping by `info.game_id`:

| game | sequences | control recall | FT recall | delta R | control prec | FT prec | delta P | player->referee |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 0.8946 | 0.9532 | **+5.86** | 0.8184 | 0.8973 | +7.89 | **0.000** |
| 3 | 21 | 0.9139 | 0.8515 | **-6.24** | 0.7787 | 0.8524 | +7.37 | **0.477** |
| 5 | 19 | 0.8981 | 0.9572 | **+5.90** | 0.8040 | 0.9253 | +12.13 | **0.000** |

The valid split is three games (2, 3, 5); train is three other games (4, 6, 9). **The regression is
exactly game 3 and all 21 of its sequences.** On the other two games the fine-tune is an unambiguous
win on both axes.

It is not a single kit clash — inside game 3 **both** teams leak:

| game 3, GT players | -> player | -> referee |
|---|---|---|
| control, team left | 0.988 | 0.011 |
| control, team right | 0.986 | 0.012 |
| fine-tune, team left | 0.488 | **0.504** |
| fine-tune, team right | 0.548 | **0.440** |

Mean HSV over sampled frames per game:

| split | game | H | **S** | V |
|---|---|---|---|---|
| train | 4 | 47.9 | **167.4** | 131.7 |
| train | 6 | 47.8 | **107.7** | 102.1 |
| train | 9 | 61.9 | **97.9** | 120.0 |
| valid | 2 | 58.0 | 117.5 | 116.9 |
| valid | **3** | 68.9 | **72.4** | 103.2 |
| valid | 5 | 66.3 | 133.8 | 128.7 |

**Game 3 is markedly desaturated — below every training game.** A model fine-tuned on three
saturated games learns saturated kit appearance as the player signature and dumps washed-out
players into the class whose training examples are neutral/dark: referee. Both teams leak because
the confound is global to the broadcast, not specific to a kit.

**This is the same finding sessions 1-3 reached for identity, now confirmed for detection: 57 GSR
train clips = three games is not enough visual diversity.** Sessions 1/2 could not rebuild the
identity embedding from it; session 3 fixed that with 44.7x more identities from sn-reid. The
detector's role head fails the same way on the same corpus.

## 7. GATE VERDICT — met, and the recommendation is still no

| criterion (as declared in §0) | threshold | FT plain (the gated run) | FT colour (§8) |
|---|---|---|---|
| person recall at matched precision | +2.0 | **+2.40 MET** | **+2.73 MET** |
| box mAP@0.5 | +3.0 | -0.53 not met | **+5.29 MET** |

The gate was declared as an OR. **It is met** — by the gated run on one criterion, and by the
colour-hardened run on both.

**And I am not recommending the laptop re-extract.** The gate I wrote measured only the box, and
that was an error in the gate, made before §6 existed. GS-HOTA gates on role as well as position;
the same evaluation shows **role-correct coverage of GT persons falling 10.20 points** (0.8404 ->
0.7384), and because `_team_side` nulls `team` for non-player/GK roles, each of those 20,142
player->referee errors costs the team attribute too. Trading +2.40 points of box recall for -10.20
points of role coverage on an identity-gated metric is very unlikely to be positive, and burning a
~5 h laptop extract to discover that is the wrong bet. The honest form is: *the gate passed and the
gate was wrong*, recorded here rather than quietly re-drawn.

What is NOT in doubt: on 37 of 58 sequences (games 2 and 5) the fine-tune delivers **+5.6-5.9 points
of recall and +7.8-11.3 points of precision** with **zero** role leak. The box lever is real and
large. It is one game's colour statistics away from being shippable.

### 7a. What to do instead of the re-extract

1. **Close the role leak with training diversity, not recipe tuning** (§8 shows the recipe lever is
   already spent). SoccerNet-v3 is MIT-licensed, 400 games across 6 leagues, jersey-labelled
   bounding boxes whose `class_index` already separates player / goalkeeper / main referee / side
   referee / staff (`docs/SOCCERNET_REPO_SWEEP.md` §2, and session 3 §1 confirmed those class
   labels are richer than the sweep recorded). That is the same corpus that broke the identity wall
   in session 3, and it is the natural next cluster session. ~60 GB fetch, needs Sid's per-fetch
   approval.
2. **Measurable interim if a box-only win is wanted sooner:** run the fine-tuned detector for boxes
   and the shipped detector for the role class on those boxes. Keeps +9 precision and the control's
   0.9306 role accuracy at the cost of a second forward pass (~2x detector time, so ~8-10 h for the
   laptop extract rather than ~5 h). Not measured — do not quote it until it is.
3. Re-run the gate after (1). The gate must then include a role criterion; the version in §0 is not
   sufficient on its own and should not be reused as written.

## 8. Follow-up run — colour-hardened augmentation confirms the mechanism and does not fix it

Same recipe, same seed, same 32 min, one change: `hsv_h 0.015 -> 0.3`, `hsv_s 0.7 -> 0.9`,
`hsv_v 0.4 -> 0.6`, aimed directly at the §6 mechanism. Registered as a mechanism test, not a
second bite at the gate.

| valid, 8,700 frames | control | FT plain | **FT colour** |
|---|---|---|---|
| mAP@0.5 | 0.5003 | 0.4950 | **0.5532 (+5.29)** |
| mAP@0.5 ball | 0.0581 | 0.1927 | **0.2065** |
| mAP@0.5 goalkeeper | 0.5705 | 0.7059 | **0.7745** |
| mAP@0.5 player | 0.8472 | 0.8104 | **0.8675** |
| mAP@0.5 referee | 0.5253 | 0.2708 | 0.3644 |
| person precision @ 0.20 | 0.7966 | 0.8891 | 0.8643 |
| person recall @ 0.20 | 0.9031 | 0.9126 | **0.9218** |
| **recall @ matched precision** | 0.9031 | 0.9270 | **0.9304 (+2.73)** |
| per-seq recall mean | 0.9027 | 0.9177 | **0.9260** |
| **role-correct coverage of GT** | **0.8404** | 0.7384 | 0.7799 **(-6.05)** |
| player -> referee | 644 | 20,142 | **14,989** |

Per-epoch valid mAP@0.5: .508 .536 .525 .507 .527 .538 .543 .544 .555 **.554** — above the control
from **epoch 1**, where the un-augmented run never got there in ten.

Per-game (the §6 split), and the leak by game:

| game | seq | control R | colour-FT R | delta R | control P | colour-FT P | delta P | player->referee |
|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 0.8946 | 0.9522 | +5.76 | 0.8184 | 0.8963 | +7.79 | 0.0007 |
| 3 | 21 | 0.9139 | 0.8779 | **-3.59** | 0.7787 | 0.8003 | +2.16 | **0.348** |
| 5 | 19 | 0.8981 | 0.9543 | +5.62 | 0.8040 | 0.9174 | +11.34 | 0.0001 |

**The mechanism is confirmed and the fix is only partial.** Attacking the colour confound directly
turns a -0.53 mAP loss into a **+5.29 gain**, lifts every class including player, and cuts the
game-3 leak by 27% (0.477 -> 0.348). It does not close it: still 21 of 58 sequences over 10%, still
exactly game 3, still both teams. Per-track voting still makes it worse (0.8460 -> 0.8319).

Nor does collapsing the class away rescue it — if every FT `referee` prediction were relabelled
`player`, role-correct coverage would be 0.8256 (plain) / 0.8335 (colour), **both still below the
control's 0.8404**. There is no post-hoc remap that recovers the loss.

**Augmentation is not the binding constraint; three games of training footage is.** Same wall,
same corpus, as sessions 1-3 hit for identity — and session 3's fix there was 44.7x more identities
from outside GSR, not a better recipe.

## 9. Artifacts (server, `siddhanth23519@a100server1`; nothing vendored into this repo)

- Weights shipped up: `~/models/football_yolov8s_baseline.pt` (md5 verified identical to the laptop
  HF cache).
- Dataset: `~/work/det/data/` (`images/{train,valid}` symlinks, `labels/`, `gsr_det.yaml`,
  `dataset_stats.json`).
- Scripts: `~/work/det/build_det_data.py`, `det_eval.py`, `role_vote.py`, `train_det_ft.py`.
- **Fine-tuned weights (server only, nothing pulled to the laptop):**
  - gated run: `~/runs/det/gsr_ft/weights/last.pt` (22,491,747 B) + `epoch{0..8}.pt`
  - colour run: `~/runs/det/gsr_ft_colour/weights/last.pt` (22,491,747 B) — the one to use if this
    line is ever resumed; `save_period 0`, so no per-epoch checkpoints for this run
- Logs/runs: `~/logs/det_ft.log`, `~/runs/det/{gsr_ft,gsr_ft_colour}/results.csv`. 514 MB total.
- Evals: `~/work/det/eval_control_640.json`, `eval_control_1280.json`, `eval_ft_last_640.json`,
  `eval_ft_ep{0,2,5}.json`, `eval_ftcolour_last_640.json`, `rolevote_control.json`,
  `rolevote_ft.json`, `rolevote_ftcolour.json`.
- Server state at close: quota fine, volume 844 GB free, **GPU 1 released (4 MiB)**. GPU 0 was
  another user's throughout and was never touched.
- Repo-side changes this session: `results/CLUSTER_SESSION5A.md` and two claims
  (`gsr-det-001`, `gsr-det-002`) in `knowledge/claims.json`. No code touched, no commits.
