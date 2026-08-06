# Cluster session S4 — detector on SoccerNet-v3 + GSR, with the role-inclusive gate (2026-08-06)

Session 5A established the diagnosis: fine-tuning the pipeline detector on GSR train alone buys a
large box win and destroys the ROLE attribute on one held-out game (game 3, 47.7% of its players
classed "referee"), because 57 GSR train clips are three games and the model learns "saturated kit
= player". Colour-hardened augmentation cut the leak 27% and no further. 5A's conclusion:
*augmentation is not the binding constraint; three games of training footage is.*

S4 tests that conclusion by adding SoccerNet-v3 — 400 games, 6 leagues, 33,986 labelled stills —
to the training mix, and judges it on a gate that includes role, which 5A's did not.

**Headline: 5A's diagnosis is confirmed — and the gate FAILS by 0.0004.** Adding v3 breadth turns
every 5A regression around: box mAP@0.5 **+18.50** points, recall at matched precision **+5.41**,
ball mAP **+14.85**, and role-correct coverage **rises** 0.8403 -> 0.9038 (+6.35) where 5A's best
run lost 6.05. The game-3 `player -> referee` leak collapses from 0.348 (5A colour) to **0.0504**
— a 6.9x reduction — and lands **0.0004 above the 0.05 criterion (c)**: 2,322 leaked detections
of 46,053, where 2,302 would have passed. Twenty detections. Gate criterion (c) is a hard
threshold declared before the run, so the verdict is **FAIL** and v6 ships without A2.

The trajectory (§6) says the model was still improving at epoch 10 on both axes and had not
converged; this is a budget shortfall, not a wall. That is a recommendation for a future session
under a fresh gate, not a re-reading of this one.

> **Sequel: S4b (below, from "# S4b") re-ran the same fixed bar after 10 more epochs and PASSES
> all four criteria** — mAP@0.5 +20.66, recall@matched +5.58, role coverage 0.9130, game-3 leak
> 0.04576, ball 0.2222. The S4 FAIL recorded here stands exactly as written; S4b is a separate
> attempt under its own pre-declared gate (S4b.0), not a re-reading of this one. Read S4b.4 before
> trusting the (c) margin.

---

## 0. PRE-DECLARED GATE (written before any S4 evaluation was run)

Recommend the S7 laptop re-extract **iff ALL FOUR** hold on the GSR valid split (58 sequences,
8,700 stride-5 frames, 141,512 GT persons), measured through `~/work/det/det_eval.py` +
`~/work/det/role_vote.py` at the pipeline's inference geometry (`imgsz=640`, `iou=0.7`,
`max_det=300`, `DETECT_CONF=0.20`), with the CONTROL re-derived first through the same evaluator:

- **(a) box** — person recall at matched precision **>= +2.0 points** (recall read off the
  fine-tune's PR curve at the precision the control achieves at conf 0.20) **OR** box mAP@0.5
  **>= +3.0 points**;
- **(b) role, aggregate** — role-correct coverage of GT persons **>= 0.8404**, the control's value
  (no regression);
- **(c) role, per game** — `player -> referee` rate **<= 0.05 on EVERY valid game** (2, 3, 5).
  This is the game-3 criterion: 5A's failure was invisible in the aggregate box metrics and
  visible only per game;
- **(d) ball** — ball mAP@0.5 **>= 0.0581**, the control's value.

Judged on **`last.pt`** (final epoch). Ultralytics selects `best.pt` by fitness on its `val` set,
which here is the split we then report — selection on the reported split. `best.pt` may be quoted
as an upper bound only and cannot open the gate.

If any of (a)-(d) fails, the finding is the negative: v6 ships without A2 at zero sunk cost,
because the expensive laptop re-extract (valid ~5 h + test ~9 h) is deferred to S7 and is not run
here either way.

Secondary numbers reported regardless (not gates): per-class mAP@0.5, per-game person P/R,
recall by GT box height, per-track role vote, training curve.

---

## 1. The dataset — v3 ingester, mapping, and the resolution decision

`~/work/det/build_det_data.py` gained a `--v3-root` ingester (`build_v3`); the GSR path is
unchanged and `gsr_det.yaml` was not rebuilt, so control and fine-tune are evaluated on the same
`images/valid` symlinks as 5A. The original is kept beside it as `build_det_data.py.orig`.

**Class mapping (`V3_MAP`), exactly the audit's §9 decision.** Reproduced verbatim by the build:

| v3 class | boxes | -> detector class |
|---|---:|---|
| Player team left + right | 292,496 | 2 player |
| Ball | 26,376 | 0 ball |
| Main + Side referee | 26,202 | 3 referee |
| Goalkeeper team left + right | 21,007 | 1 goalkeeper |
| **mapped** | **366,081 = 98.52%** | |
| Staff members | 4,955 | DROP |
| Wall of players | 317 | DROP (group box, not a person) |
| Referee flag / Yellow card / Red card | 246 | DROP (objects) |

Every drop count matches `~/logs/v3_audit.md` exactly. Dropping Staff is not just "no target
class": GSR annotates only on-pitch players/keepers/officials, so touchline staff are pure false
positives there too — the drop teaches the same background convention the target domain uses.

**Resolution decision (the v3-vs-GSR mix).** 67.4% of v3 is 1280x720, GSR is uniformly 1920x1080;
both are ~16:9, so there is no letterbox asymmetry, only a scale one. Ultralytics' loader
rescales every training image so its long side equals `imgsz` and then letterboxes — i.e. it
already equalises the two corpora inside the network. v3 frames are therefore written to disk
**pre-scaled to that same long side (640, INTER_AREA, JPEG q90)**: the geometry ultralytics would
have produced anyway, for **1.9 GB instead of 45 GB** of PNG and a much cheaper decode. Labels are
normalized, so scaling leaves them bit-identical (asserted in the builder's `--selfcheck`). GSR
frames stay native symlinks and are scaled at load time. Build: 400 games, 24 workers, ~7 min,
resumable per game, peak scratch one game per worker.

| | GSR train | **v3 train** | GSR valid (eval) |
|---|---:|---:|---:|
| source | 57 clips / 3 games, video, stride 1 | **400 games / 6 leagues, stills** | 58 clips / 3 games |
| frames | 42,750 | **33,986** | 8,700 |
| player | 604,968 | 292,496 | 123,928 |
| goalkeeper | 24,107 | 21,007 | 5,203 |
| referee | 61,543 | 26,202 | 12,381 |
| ball | 40,928 | **26,376** (77.6% of frames) | 8,179 |
| person height p05/median/p95 px (native) | 54 / 98 / 156 | 46 / 94 / 498 | 52 / 94 / 157 |
| frames with no mapped box | 0 | 29 | 0 |

**Sampling ratio: plain union, 42,750 : 33,986 = 1.258 : 1 (55.7% GSR / 44.3% v3).** The brief's
example was a 1:1 epoch-level mix; the raw union is already 1.26:1, so no sampler was written —
the machinery would have moved the ratio by less than the complexity it added. The target domain
keeps the majority of gradient steps while the diversity term is nearly half. Worth stating
plainly: per *distinct scene* v3 dominates enormously (33,986 independent stills from 400 matches
vs 57 runs of 750 consecutive video frames), which is exactly the imbalance the session wanted.

**No leak into the eval split.** v3 is 2014-2017 (64 / 131 / 205 games by season); GSR clips are
2019-era and their `info` block carries no league or date, so overlap cannot be checked by
metadata — but the season ranges are disjoint. v3's own valid/test games were *not* excluded (all
400 labelled games are used); harmless, because nothing from v3 is ever evaluated on.

## 2. Training

`~/work/det/train_det_ft.py` (defaults updated to 5A §8's colour-hardened aug; `.orig` kept).
Init `~/models/football_yolov8s_baseline.pt`, data `gsr_v3_det.yaml`
(`train: [images/train, images/train_v3]`), imgsz 640, batch 64, SGD **lr0 0.002** / lrf 0.05 /
momentum 0.937 / wd 0.0005, warmup 1.0, close_mosaic 3, **hsv_h 0.3 / hsv_s 0.9 / hsv_v 0.6**,
10 epochs, save_period 1, seed 0, GPU 1. Loader: **76,736 images, 29 backgrounds, 0 corrupt.**

**48.2 min wall (0.803 h)**, 1,199 iters/epoch, ~286 s/epoch.

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| box | 1.317 | 1.231 | 1.192 | 1.165 | 1.144 | 1.127 | 1.113 | 1.077 | 1.062 | 1.049 |
| cls | 0.714 | 0.615 | 0.583 | 0.564 | 0.548 | 0.533 | 0.522 | 0.490 | 0.477 | 0.466 |
| dfl | 0.956 | 0.919 | 0.910 | 0.904 | 0.900 | 0.896 | 0.893 | 0.894 | 0.891 | 0.888 |
| valid mAP@0.5 | .607 | .633 | .618 | .646 | .659 | .653 | .674 | .669 | .681 | **.688** |

Losses descend monotonically; the valid curve is **above the control's 0.5003 from epoch 1** and
**still rising at epoch 10** (5A's colour run plateaued at .554 by epoch 9). `best.pt` = epoch 10
= `last.pt` (differs only by strip/reload; md5 `f53401dd...` vs `dc78f397...`), so the
anti-selection rule is moot in the event.

## 3. Control — re-derived first, same evaluator

`~/work/det/det_eval.py` + `role_vote.py` on `~/models/football_yolov8s_baseline.pt`, imgsz 640.
Every number reproduces 5A to the digit: mAP@0.5 **0.5003**, ball **0.0581**, person P@0.20
**0.7966** / R **0.9031**, role-correct coverage **0.8403457**, `player->referee` **645**. The
evaluator is unchanged apart from one additive field (`per_sequence_counts`, so per-game person
P/R can be summed from counts rather than averaged from ratios) and the per-game aggregation added
to `role_vote.py` for criterion (c).

## 4. Control vs fine-tune (`last.pt`, epoch 10)

| metric — valid 58 seq / 8,700 frames / 141,512 person GT | control | S4 fine-tune | delta |
|---|---:|---:|---:|
| **mAP@0.5** | 0.5003 | **0.6853** | **+18.50** |
| mAP@0.5:0.95 | 0.2300 | 0.3775 | +14.75 |
| mAP@0.5 **ball** | 0.0581 | **0.2066** | **+14.85** |
| mAP@0.5 goalkeeper | 0.5705 | 0.8635 | +29.29 |
| mAP@0.5 player | 0.8472 | 0.9418 | +9.45 |
| mAP@0.5 referee | 0.5253 | 0.7295 | +20.42 |
| person precision @ conf 0.20 | 0.7966 | 0.9049 | +10.83 |
| person recall @ conf 0.20 | 0.9031 | 0.9479 | +4.48 |
| **person recall @ MATCHED precision (P>=0.7966)** | 0.9031 | **0.9572** | **+5.41** |
| detections @ conf 0.20 | 160,424 | 148,233 | -12,191 |
| per-seq recall mean / min / median | .9027 / .7883 / .9085 | .9485 / .8894 / .9500 | **58/58 better** |
| per-seq precision | — | — | **57/58 better** |
| role accuracy, per detection | 0.9305 | 0.9535 | +2.30 |
| **role-correct coverage of GT** | **0.8403** | **0.9038** | **+6.35** |
| role coverage after per-track vote (GT tracks) | 0.8466 | **0.9197** | +7.31 |
| `player -> referee` (per detection) | 645 | 2,359 | +1,714 |
| `player -> referee` after per-track vote | 0 | 258 | +258 |

Recall by ground-truth box height (conf 0.20):

| height | GT | control | fine-tune | delta |
|---|---:|---:|---:|---:|
| <40 px | 1,997 | 0.3996 | **0.5493** | **+14.97** |
| 40-70 px | 24,389 | 0.8313 | 0.8929 | +6.16 |
| 70-110 px | 70,497 | 0.9172 | 0.9574 | +4.02 |
| >=110 px | 44,629 | 0.9425 | 0.9809 | +3.83 |

Every bucket improves — including the <40 px bucket that 5A could only move +2.70, and which
inference-side upscaling bought only by destroying large players (5A §3a).

Role confusion on matched detections (`GT -> predicted`):

| | control | S4 fine-tune |
|---|---|---|
| goalkeeper correct | 2,276 / 4,474 = 50.9% | **3,747 / 4,947 = 75.7%** |
| referee correct | 4,677 / 10,578 = 44.2% | **8,759 / 11,343 = 77.2%** |
| player correct | 111,966 / 112,744 = 99.3% | 115,393 / 117,849 = **97.9%** |
| player -> referee | 645 | 2,359 |

Compare 5A: its player-correct rate fell to 80.7% (plain) with 20,142 leaks. Here the majority
class is essentially intact (-1.4 points) while both minority roles gain 25-33 points.

**Per-track majority voting now REPAIRS the fine-tune, where in 5A it made it worse.** Over
ground-truth tracks (oracle association), coverage 0.9038 -> 0.9197 (+1.59) and the leak 2,359 ->
258 (-89%). 5A's errors were track-consistent — confidently and persistently wrong about the same
people; S4's residual errors are per-frame flicker, which is the failure mode a tracker can fix.

## 5. Per-game — the criterion-(c) table

| game | seq | control R | FT R | delta R | control P | FT P | delta P | control leak | **FT leak** | role coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 18 | 0.8935 | 0.9508 | **+5.73** | 0.8161 | 0.9128 | +9.67 | 0.0012 | **0.0010** | 0.8233 -> 0.9408 |
| 3 | 21 | 0.9138 | **0.9407** | **+2.69** | 0.7773 | 0.8798 | +10.25 | 0.0114 | **0.0504** | 0.8682 -> 0.8413 |
| 5 | 19 | 0.8984 | 0.9541 | **+5.57** | 0.8047 | 0.9294 | +12.47 | 0.0026 | **0.0001** | 0.8216 -> 0.9469 |

Game 3 — the game 5A *lost* 6.24 points of recall on — now **gains** 2.69 recall and 10.25
precision. Its leak is 2,322 of 46,053 matched players = 0.05042. Games 2 and 5 leak essentially
zero (34 and 3 detections). Per sequence: **1 of 58 above 10%** (SNGS-059 at 0.1006, a game-3
sequence) against 21 of 58 in 5A; median sequence leak 0.00055.

## 6. Epoch trajectory — the leak is falling and had not converged

Diagnostic only; `last.pt` is what the gate judges (§0).

| checkpoint | role coverage | coverage after track vote | leak game 2 | **leak game 3** | leak game 5 | `player->referee` |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.8403 | 0.8466 | 0.0012 | 0.0114 | 0.0026 | 645 |
| epoch 1 | 0.8634 | 0.8903 | 0.0015 | 0.0915 | 0.0003 | 4,224 |
| epoch 4 | 0.8867 | 0.9000 | 0.0013 | 0.0615 | 0.0001 | 2,870 |
| epoch 7 | 0.8988 | 0.9155 | 0.0011 | 0.0626 | 0.0002 | 2,941 |
| **epoch 10 = last** | **0.9038** | **0.9197** | 0.0010 | **0.0504** | 0.0001 | 2,359 |

This is the exact opposite of 5A §5a, where the leak was fully present after one epoch and flat
thereafter ("early stopping does not buy a checkpoint with the box gain and without the role
loss"). Here role coverage rises monotonically and the game-3 leak falls 0.0915 -> 0.0504 over ten
epochs, with the valid mAP curve still climbing at epoch 10. Nothing had converged when the
10-epoch budget ran out.

## 7. GATE VERDICT — FAIL on (c), by 0.0004

| criterion (as declared in §0) | threshold | measured | verdict |
|---|---|---:|---|
| (a) recall @ matched precision **OR** mAP@0.5 | +2.0 / +3.0 | **+5.41** / **+18.50** | **PASS** (both) |
| (b) role-correct coverage of GT persons | >= 0.8404 | **0.9038** | **PASS** (+6.35) |
| (c) `player -> referee` on EVERY valid game | <= 0.05 | 2: 0.0010, **3: 0.0504**, 5: 0.0001 | **FAIL** |
| (d) ball mAP@0.5 | >= 0.0581 | **0.2066** | **PASS** (+14.85) |

**All four are required. VERDICT: FAIL. v6 ships without A2.** Zero sunk cost: the expensive
laptop re-extract (valid ~5 h + test ~9 h) was deferred to S7 by design and was not run here.

The margin is 20 detections in 46,053 (2,322 leaked; 2,302 would have passed) — 0.04% of one
game's matched players. That is inside any reasonable noise band for a single seed, and it would
be trivially easy to talk past. The threshold was written down before the evaluator ran precisely
so that it could not be, and criterion (c) was chosen deliberately because 5A's failure was
invisible in aggregates and visible only per game. It is not re-drawn here.

**What the numbers do establish, and this is the session's real content:** 5A's causal claim was
correct. Training-set breadth, not augmentation, was the binding constraint. Adding 400 games of
stills to 3 games of video reversed every 5A regression — role coverage from -6.05 to **+6.35**,
game-3 recall from -6.24 to **+2.69**, the leak by **6.9x** — while multiplying the box win
(+5.29 -> **+18.50** mAP@0.5) and lifting ball mAP 3.6x. gsr-det-002's proposed fix is confirmed
as the right lever; only the dose was short.

### 7a. What a future session would need (NOT authorised here, NOT costed into v6)

1. **More of the same, longer.** §6 shows both role axes still moving at epoch 10 and §2 shows
   the valid curve still rising. 20-25 epochs on the identical mix is ~2 GPU-h. A re-run needs a
   *fresh* pre-declared gate — the epoch-10 checkpoint cannot be re-judged, and picking an epoch
   by its leak on the reported split is exactly the selection the anti-selection rule forbids.
2. **More v3 per epoch.** The mix is 55.7% GSR; the leak is a diversity term, so an oversampled
   v3 (say 1:2) is the other dose knob. Untested.
3. **The per-track vote is now a live option** (§4): over GT tracks it cuts the leak 89% and lifts
   coverage to 0.9197. Our pipeline has tracklets, so a real (non-oracle) tracker-side role vote
   is worth measuring — but it must be measured through the real association, not this oracle.
4. Do **not** quote the ball gain as ball coverage. mAP@0.5 0.0581 -> 0.2066 is a box-level number
   on stride-5 valid frames; the only ball figure that counts is the post-`link_ball` usable
   track, which was not run.

## 8. Artifacts (server `siddhanth23519@a100server1`; nothing vendored into this repo)

- **Weights (parked, server only, nothing pulled to the laptop):**
  `~/runs/det/gsr_v3_ft/weights/last.pt` (22,491,747 B, md5 `f53401ddee481e05756a282a92937fde`)
  plus `epoch{0..9}.pt` and `best.pt` (md5 `dc78f397fd20970a56f44d68464f3a90`). 471 MB total.
- Dataset: `~/work/det/data/` — `images/{train,valid}` symlinks + `images/train_v3` (33,986 JPEGs,
  1.9 GB), `labels/`, `gsr_det.yaml` (GSR only, the eval yaml), `gsr_v3_det.yaml` (the training
  yaml), `dataset_stats.json`, `_v3_shards/` (400 per-game resume shards). 2.5 GB.
- Scripts: `~/work/det/build_det_data.py` (+ `.orig`), `train_det_ft.py` (+ `.orig`),
  `role_vote.py` (+ `.orig`), `det_eval.py` (one additive field), `s4_gate.py`.
- Evals: `~/work/det/s4_eval_{control,ft_last}.json`,
  `~/work/det/s4_rolevote_{control,ft_last,ft_ep0,ft_ep3,ft_ep6}.json`.
- Logs: `~/logs/{v3_det_build.log, s4_control.log, s4_det_ft.log, s4_ft_eval.log,
  s4_epoch_leak.log}`, `~/runs/det/gsr_v3_ft/{results.csv,args.yaml}`.
- **Server GPU spend this session: ~1.4 h** (48 min train + ~35 min evals) of the 4-6 h budget.
  End state: GPU 1 released; GPU 0 was another user's throughout and was never touched. Volume
  772 GB free.
- Repo-side changes: this file and one claim (`gsr-det-003`) in `knowledge/claims.json`. No code
  under `core/ generator/ fingerprint/ synthesizer/ report/ tools/ tests/` touched. No commits.


---

# S4b — the same bar, a second dose (2026-08-06)

S4's verdict stands as recorded: FAIL on criterion (c), threshold un-redrawn. S4b is a NEW attempt
at the SAME fixed bar, justified by the dose-shortfall diagnosis in §6 (game-3 leak falling
monotonically 0.0915 -> 0.0504 across ten epochs, valid mAP still rising at epoch 10, nothing
converged). More training, not a moved goalpost.

## S4b.0 PRE-DECLARED GATE (written before any S4b training or evaluation was run)

**Identical to §0 — same four criteria, same thresholds, same original control.** Recommend the S7
laptop re-extract iff ALL FOUR hold on the GSR valid split (58 sequences, 8,700 stride-5 frames,
141,512 GT persons), through the same `det_eval.py` + `role_vote.py` at imgsz 640 / iou 0.7 /
max_det 300 / DETECT_CONF 0.20:

- **(a) box** — person recall at matched precision **>= +2.0 points** OR box mAP@0.5 **>= +3.0
  points**, both measured against the **ORIGINAL control** (`~/models/football_yolov8s_baseline.pt`:
  mAP@0.5 0.5003, person P 0.7966 / R 0.9031), not against the S4 fine-tune;
- **(b) role, aggregate** — role-correct coverage of GT persons **>= 0.8404**;
- **(c) role, per game** — `player -> referee` rate **<= 0.05 on EVERY valid game** (2, 3, 5);
- **(d) ball** — ball mAP@0.5 **>= 0.0581**.

Judged on the **new run's `last.pt`** (final epoch of the extension). No best-epoch selection: the
per-epoch role_vote trace is a diagnostic and cannot open the gate, exactly as in S4 §6.

**This is the last attempt.** If S4b fails, v6 ships without A2, final — no S4c; the detector lever
moves to the post-v6 backlog. The laptop re-extract is not run either way unless the gate passes.

Deviation stated up front, because it is the one thing that is not identical: ultralytics' `resume`
continues a run to its *original* epoch count and cannot extend a completed 10-epoch run without
editing the checkpoint's stored args. S4b therefore re-runs the **identical recipe** with
`--weights ~/runs/det/gsr_v3_ft/weights/last.pt` — a warm restart, so the LR restarts at 0.002 and
decays to 0.0001 again rather than continuing from 0.0001. Everything else (data, aug, batch,
seed 0, close_mosaic 3, 10 epochs, save_period 1) is unchanged.

## S4b.1 Training — the extension

`~/work/det/train_det_ft.py --weights ~/runs/det/gsr_v3_ft/weights/last.pt --name gsr_v3_ft_b`.
Same data (`gsr_v3_det.yaml`, 76,736 images / 29 backgrounds), same aug (hsv 0.3/0.9/0.6), same
lr0 0.002 / lrf 0.05 / warmup 1.0 / close_mosaic 3 / batch 64 / imgsz 640 / seed 0 / save_period 1
/ GPU 1. **48.6 min wall (0.810 h).** Cumulative over S4 + S4b: 20 epochs, 1.61 GPU-h.

| epoch (extension) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| box | 1.096 | 1.120 | 1.109 | 1.095 | 1.085 | 1.074 | 1.064 | 1.031 | 1.017 | **1.006** |
| cls | 0.507 | 0.528 | 0.520 | 0.512 | 0.503 | 0.495 | 0.488 | 0.458 | 0.448 | **0.439** |
| dfl | 0.890 | 0.895 | 0.892 | 0.890 | 0.888 | 0.886 | 0.884 | 0.885 | 0.882 | **0.880** |
| valid mAP@0.5 | .665 | .666 | .668 | .685 | .708 | .690 | .713 | .699 | .705 | **.710** |

The warm restart costs the first two epochs, exactly as the LR restart predicts: mAP@0.5 drops
0.688 -> 0.665 and box loss rises 1.049 -> 1.120 before both recover past their S4 values. Both
losses end below S4's (box 1.006 vs 1.049, cls 0.439 vs 0.466). The valid curve is noisier than
S4's — .708 at epoch 5, .690 at 6, .713 at 7, .710 at 10 — i.e. **this run is at or near its
plateau**, which S4's was not.

`best.pt` is **epoch 7** (.71288), not epoch 10 (.70992). Unlike S4, the anti-selection rule now
has teeth: `best.pt` would have been a better-looking checkpoint chosen on the very split the
gate reports, so it is excluded. Everything below is `last.pt` (md5
`2074d8741f97a6892a1322d0f808244c`, 22,491,747 B).

## S4b.2 Control vs S4b (`last.pt`, extension epoch 10)

Control unchanged and already on record (§3): mAP@0.5 0.5003, ball 0.0581, P 0.7966 / R 0.9031,
role coverage 0.8403457, leak 645.

| metric — valid 58 seq / 8,700 frames / 141,512 person GT | control | S4 | **S4b** | S4b vs control |
|---|---:|---:|---:|---:|
| **mAP@0.5** | 0.5003 | 0.6853 | **0.7069** | **+20.66** |
| mAP@0.5:0.95 | 0.2300 | 0.3775 | 0.3870 | +15.70 |
| mAP@0.5 **ball** | 0.0581 | 0.2066 | **0.2222** | **+16.41** |
| mAP@0.5 goalkeeper | 0.5705 | 0.8635 | 0.8764 | +30.59 |
| mAP@0.5 player | 0.8472 | 0.9418 | 0.9471 | +9.98 |
| mAP@0.5 referee | 0.5253 | 0.7295 | 0.7818 | +25.65 |
| person precision @ conf 0.20 | 0.7966 | 0.9049 | 0.9124 | +11.58 |
| person recall @ conf 0.20 | 0.9031 | 0.9479 | 0.9488 | +4.57 |
| **person recall @ MATCHED precision** | 0.9031 | 0.9572 | **0.9589** | **+5.58** |
| detections @ conf 0.20 | 160,424 | 148,233 | 147,156 | -13,268 |
| per-seq recall mean / min | .9027 / .7883 | .9485 / .8894 | .9493 / .8949 | **58/58 better** |
| per-seq precision | — | — | — | **57/58 better** |
| role accuracy, per detection | 0.9305 | 0.9535 | **0.9623** | +3.18 |
| **role-correct coverage of GT** | **0.8403** | 0.9038 | **0.9130** | **+7.27** |
| coverage after per-track vote (GT tracks) | 0.8466 | 0.9197 | **0.9286** | +8.20 |
| `player -> referee` (per detection) | 645 | 2,359 | 2,141 | +1,496 |
| `player -> referee` after per-track vote | 0 | 258 | 530 | +530 |

Recall by ground-truth box height (conf 0.20): <40 px 0.3996 -> **0.5528** (+15.32), 40-70 px
0.8313 -> 0.8976 (+6.63), 70-110 px 0.9172 -> 0.9576 (+4.04), >=110 px 0.9425 -> 0.9807 (+3.81).

Role confusion on matched detections: goalkeeper correct **3,938 / 4,971 = 79.2%** (control 50.9%),
referee correct **9,599 / 11,391 = 84.3%** (control 44.2%), player correct **115,662 / 117,905 =
98.1%** (control 99.3%).

## S4b.3 Per-game — the criterion-(c) table

| game | seq | ctl R | S4b R | delta R | ctl P | S4b P | delta P | ctl leak | **S4b leak** | leaked / matched | role coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 18 | 0.8935 | 0.9509 | +5.74 | 0.8161 | 0.9203 | +10.42 | 0.0012 | **0.00082** | 27 / 32,878 | 0.8233 -> 0.9409 |
| 3 | 21 | 0.9138 | 0.9424 | +2.86 | 0.7773 | 0.8905 | +11.32 | 0.0114 | **0.04576** | 2,110 / 46,114 | 0.8682 -> 0.8635 |
| 5 | 19 | 0.8984 | 0.9547 | +5.63 | 0.8047 | 0.9326 | +12.79 | 0.0026 | **0.00010** | 4 / 38,913 | 0.8216 -> 0.9483 |

Game 3 clears with 195 detections of headroom (2,110 leaked; 2,305 is the limit). Per sequence:
2 of 58 above 10% (SNGS-053 0.1047, SNGS-059 0.1035, both game 3); median sequence leak 0.00045.

## S4b.4 Epoch trace (diagnostic only — `last.pt` is what the gate judges)

| checkpoint | role coverage | after track vote | leak g2 | **leak g3** | leak g5 | `player->referee` |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.8403 | 0.8466 | 0.0012 | 0.0114 | 0.0026 | 645 |
| S4 epoch 10 | 0.9038 | 0.9197 | 0.0010 | 0.0504 | 0.0001 | 2,359 |
| S4b epoch 1 | 0.8912 | 0.9138 | 0.0010 | 0.0691 | 0.0001 | 3,207 |
| S4b epoch 4 | 0.9018 | 0.9138 | 0.0009 | 0.0447 | 0.0001 | 2,091 |
| S4b epoch 7 | 0.9153 | 0.9277 | 0.0005 | **0.0327** | 0.0001 | 1,528 |
| **S4b epoch 10 = last** | **0.9130** | **0.9286** | 0.0008 | **0.0458** | 0.0001 | 2,141 |

**Stated plainly, because it bounds how much the pass is worth:** the game-3 leak is NOT monotone
inside S4b. The warm restart pushes it back up to 0.0691, it falls to 0.0327 by epoch 7, then
returns to 0.0458 at epoch 10. The within-run spread over four sampled checkpoints is **0.0364**,
which is roughly **8x the 0.0042 margin by which the gate passes**. The verdict is correct by the
rule as declared — `last.pt`, no selection — but criterion (c) is measurably a noisy quantity at
this scale on one seed, and S4's FAIL at 0.0504 versus S4b's PASS at 0.0458 sits comfortably
inside that noise. Both verdicts are honest readings of the same declared bar; neither is strong
evidence about the other.

## S4b.5 GATE VERDICT — PASS on all four

| criterion (S4b.0, identical to §0) | threshold | measured | verdict |
|---|---|---:|---|
| (a) recall @ matched precision **OR** mAP@0.5 | +2.0 / +3.0 | **+5.58** / **+20.66** | **PASS** (both) |
| (b) role-correct coverage of GT persons | >= 0.8404 | **0.9130** | **PASS** (+7.27) |
| (c) `player -> referee` on EVERY valid game | <= 0.05 | 2: 0.00082, **3: 0.04576**, 5: 0.00010 | **PASS** |
| (d) ball mAP@0.5 | >= 0.0581 | **0.2222** | **PASS** (+16.41) |

**VERDICT: PASS. A2 is recommended for S7.**

The lever is now positive on every axis GS-HOTA gates: better boxes (+20.66 mAP@0.5, +5.58 recall
at matched precision, 58/58 sequences), better role (+7.27 coverage, and +8.20 after a track vote
our pipeline can actually perform), better ball boxes, and no per-game role catastrophe. Twenty
epochs of a 1.26:1 GSR:SoccerNet-v3 mix, 1.61 GPU-h total, on the same A100 slot.

### S4b.5a Weights parked

- **`~/runs/det/gsr_v3_ft_b/weights/last.pt`** — 22,491,747 B, md5
  `2074d8741f97a6892a1322d0f808244c`. Server only; nothing pulled to the laptop this session.
- Excluded on purpose: `best.pt` (epoch 7, md5 `09742df1117042a9c97ce039d5ffa81f`) — selected by
  fitness on the reported split.
- Per-epoch `epoch{0..9}.pt` retained; run dir 471 MB.
- Drop-in: class ids are unchanged `{0: ball, 1: goalkeeper, 2: player, 3: referee}`, so
  `generator/extract.py`'s `_FootballRoleDetector` takes it as a weights-path swap with no code
  change, at the same `imgsz=640` / `BALL_CONF=0.10` / `DETECT_CONF=0.20` operating point.

### S4b.5b Projected S7 cost (NOT run here)

Adopting A2 forces the cached-detection re-extract that S7 was designed to pay at most once:
**laptop GPU ~5 h for valid-58 and ~9 h for test, ~14 h total.** No re-extract was run in S4 or
S4b, per the brief. S7 must sequence it first in its dependency order (detector -> association /
embedder -> jersey votes -> solver) so it happens exactly once.

### S4b.5c Caveats the orchestrator should carry into S7

1. **The (c) margin is 0.0042 against a within-run spread of 0.0364** (S4b.4). Treat "leak <= 0.05
   on every game" as satisfied-but-marginal on game 3, not as a solved problem. Games 2 and 5 are
   solved (27 and 4 leaked detections).
2. `player -> referee` is still 3.3x the control in absolute count (2,141 vs 645); the aggregate
   role win comes from the minority classes (goalkeeper 50.9% -> 79.2%, referee 44.2% -> 84.3%)
   more than paying for a 1.2-point loss on players.
3. The per-track vote figure (0.9286) uses GROUND-TRUTH tracks — an oracle-association upper
   bound. The real tracker-side gain is unmeasured.
4. Ball mAP@0.5 0.0581 -> 0.2222 is a box-level number on stride-5 frames. Per the ball-metric
   rule, only the post-`link_ball` usable track counts as coverage; do not quote this as coverage.
5. Everything here is one seed, one mix ratio, one aug setting, and box-level. **No GS-HOTA number
   has been measured with these weights** — the DetA-to-GS-HOTA conversion is S7's job and the
   identity gate can still eat the box gain.

### S4b.6 Artifacts (server; nothing vendored into this repo)

- Weights: `~/runs/det/gsr_v3_ft_b/weights/{last,best,epoch0..9}.pt`; run
  `~/runs/det/gsr_v3_ft_b/{results.csv,args.yaml}`.
- Evals: `~/work/det/s4b_eval_ft_last.json`,
  `~/work/det/s4b_rolevote_ft_{last,ep0,ep3,ep6}.json`, `~/work/det/s4b_gate.py`.
- Logs: `~/logs/{s4b_det_ft.log, s4b_eval.log}`.
- Server GPU spend S4b: **~1.2 h** (48.6 min train + ~28 min evals). S4 + S4b total ~2.6 h.
  End state: GPU 1 released; GPU 0 was another user's throughout and was never touched.
- Repo-side: this section of `results/CLUSTER_SESSION_S4.md` and the `gsr-det-003` claim update.
  No code under `core/ generator/ fingerprint/ synthesizer/ report/ tools/ tests/`. No commits.
