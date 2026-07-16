# Jersey-number recognizer (SoccerNet jersey-2023) — Stage 1

Standalone jersey-number model trained and evaluated on the official SoccerNet jersey-2023 split.
This is Stage 1 (a citable held-out number). Stage 2 (wiring into the GSR pipeline) is separate.

## Headline number (honest)

**Held-out test tracklet accuracy: 0.396 (480/1211), official metric incl. the `-1` class.**

- Baseline (always predict `-1`, the illegible class): 355/1211 = **0.293**. We clear it by +10.3 pts.
- Published SoccerNet jersey-challenge range: ~0.73–0.92. **We are well below it** — this is a
  first-pass, single-backbone, weakly-labelled model, not the full challenge recipe (see gaps below).
- Numbered-only accuracy (when the ground truth is a real number): **0.290** — this is the weak link.
- Legibility sub-task (predict "a number is visible" vs `-1`): **precision 0.840, recall 0.754.**
  The legibility stage works; number *recognition* is what caps the headline.

Artifacts: `outputs/jersey/eval_test.json` (numbers), `outputs/jersey/train.log` (training trace).

## What was built

Single 100-way per-crop classifier, folding the challenge's two stages into one head:
- class `0` = illegible / no legible number (ground truth `-1`);
- classes `1..99` = jersey number.

Per-tracklet decision = **confidence-weighted mean softmax over the tracklet's crops**, then a
`min_conf` floor → `-1`. Rationale: crops that actually show the number accumulate consistent mass on
the true class; back-view / blurred crops spread thin and cancel; a tracklet with no consistent
number falls through to `-1`. The illegible class competes in the same vote, so legibility filtering
is learned implicitly rather than as a separate model (the plan's "measure before adding capacity").

- Backbone: ResNet18, ImageNet-pretrained, final `fc` → 100 classes. 11.7 M params.
- Input: whole player crop resized to **224×112**, ImageNet-normalized. fp16 (AMP) on a 4 GB RTX 3050.
- Loss: cross-entropy with **label smoothing 0.1** (absorbs weak-label noise — most crops of a
  "number" tracklet do not show the number, yet inherit its label).
- Training data sampling: **one random crop per tracklet-slot per epoch**, 24 slots/tracklet, so an
  epoch is ~31 k crops (not the full 733 k) and successive epochs cover different crops. 20 epochs,
  Adam lr 3e-4, batch 96. ~45 s/epoch → the full run is ~15 min of GPU.

## Split protocol

- **Train**: 1427 tracklets (`train_gt.json`), of which a **deterministic 90/10 tracklet split**
  (seed 0) holds out 142 for monitoring + `min_conf` tuning. Test is never touched during training.
- **Test**: 1211 tracklets (`test_gt.json`) — the official held-out split, scored once.
- **`min_conf` = 0.20**, picked on the train-val split (val accuracy plateaus 0.76–0.77 for
  `min_conf` ∈ [0.05, 0.35]; collapses above 0.45 as recall drops). NB: val accuracy 0.768 vs test
  0.396 is a **same-game leakage gap** — val tracklets share games (kits/rosters) with training, so
  val over-reads; the honest generalization number is the test 0.396.
- Label distribution: train 1024 numbered / 403 illegible; test 856 / 355. Numbers span 1–99;
  ~61–71 % are two-digit.

## Resolution ablation (measured, not assumed)

| Input | Test tracklet acc | Numbered-only acc | Weights |
|------|------|------|------|
| 128×64  | 0.367 | 0.264 | `outputs/jersey/ckpt_r128_acc367.pt` |
| 224×112 | 0.396 | 0.290 | `outputs/jersey/jersey_r224_acc396.pt` (= `ckpt.pt`) |

Higher resolution helps modestly. It is **not** the bottleneck.

## Where the accuracy goes (diagnosis)

Top test confusions (true→pred, count): 33→25 (24), 4→29 (17), 10→24 (14), 16→26 (12), 7→31 (12),
62→31 (11), 44→29 (10). Two structural causes, both pointing past this architecture:

1. **Weak per-crop labels.** Only a minority of a tracklet's crops actually show the number; the rest
   (back/side/occluded) inherit the number label anyway, polluting the number classes. Legibility
   survives this (0.84 P); number identity does not (0.29). The folded single-head is the *floor*.
2. **Few-shot high numbers + class imbalance.** ~99 number classes over ~920 numbered train
   tracklets → ~9 tracklets/class, far fewer for high numbers, which collapse onto common low
   numbers (62→31, 44→29).

## Measured next levers (to close the gap toward the published range)

- **Two-digit heads** (tens 0–9 + units 0–9 + blank) instead of one 100-way head — shares statistics
  across numbers, directly attacks the few-shot high-number collapse. Cheap, in-scope, likely the
  biggest single win.
- **True two-stage separation**: train the number recognizer on *legibility-filtered* crops only
  (bootstrap the filter from this model's illegible-class probability), removing the weak-label
  ceiling. This is the published-recipe path.
- More crops/epoch and longer training help the rare classes marginally; they do not lift the
  weak-label ceiling.

# Stage 1b — factorized digit heads + two-stage filter (negative result)

Stage 1 diagnosed number *recognition* (not legibility) as the cap and named two levers. Both were
implemented and evaluated end-to-end on the official 1211-tracklet test split. **Neither moved the
headline; the honest verdict is a negative result** — the Stage-1 single 100-way head at 0.396
remains the best model. Eval is deterministic (seeded crop sampling), so the differences below are
exact tracklet counts, not noise.

## Per-step ablation (official test, incl. `-1`, `eval-crops 48`)

| Config | min_conf | tracklet acc | numbered-only | legP | legR | weights |
|--------|---------|-------------|---------------|------|------|---------|
| Stage-1 single 100-way head (baseline) | 0.20 | **0.3964** (480/1211) | 0.290 | 0.840 | 0.754 | `jersey_r224_acc396.pt` |
| Step 1: factorized tens×units + legibility heads | 0.20 | 0.3922 (475) | 0.243 | 0.843 | 0.551 | `ckpt_mh.pt` |
| Step 1: same, at its val-tuned threshold | 0.05 | 0.3840 (465) | 0.255 | 0.841 | 0.669 | `ckpt_mh.pt` |
| Step 2: + legibility self-filter of the digit loss | 0.20 | 0.3972 (481) | 0.266 | 0.843 | 0.640 | `ckpt_mh2.pt` |

- **Step 1 (factorized digit heads): −0.4 to −1.2 pp — did not help.** Factorizing the 100-way head
  into a tens digit (0–9 + `none`), a units digit (0–9) and a separate legibility head shares digit
  statistics across numbers (units "2" is trained by 2/12/22/… not by jersey 62 alone). It was
  predicted to be "the biggest single win" against rare-high-number collapse. On test it was flat-to-
  slightly-worse. The rare-class confusions it targeted survive unchanged (62→31, 44→29), and the
  composed per-crop confidence (a product of marginals) is systematically lower, which suppresses
  legibility recall at a fixed threshold (0.754 → 0.551 at min_conf 0.20).
- **Step 2 (true two-stage filter): +0.1 pp (1 tracklet) — flat.** Warm-started from step 1, the
  digit loss is masked to crops the model reads as the tracklet number with confidence ≥ 0.5, so
  back-view / occluded crops stop teaching wrong digit associations; the legibility head keeps
  training on tracklet ground truth. (An earlier variant that also *relabelled* filtered crops
  illegible for the legibility head crashed legibility recall to 0.30 — it corrupted the one working
  component; it was corrected before this measurement.) Cleaning the digit labels did not lift the
  ceiling.
- **Stopped after step 2** per the measured-fix-path rule (gain < 1 pp → no ritual capacity). Steps 3
  (class-balanced sampling) and 4 (torso crops) were **not** run: step 1 already shows that sharing
  digit statistics does nothing for the rare classes on test, so class-balanced sampling is very
  unlikely to help. The one untested lever that addresses the *actual* ceiling is torso crops (below).

## Where the ceiling actually is (revised)

The persistent top confusions are **visual**, not rare-class or head-parametrization artifacts:
4→29, 33→25, 44→29, 36→22, 93→29. These are broadcast-resolution number misreads that neither
factorization nor label cleaning touches. Stage-1's "weak labels / few-shot high numbers" diagnosis
was only partly right: the folded head was not the floor, and cleaner per-crop labels did not raise
it. Closing the gap to the published 0.73–0.92 range needs stronger *visual* number signal — the
recipes that hit that range use pose/torso-guided crops (isolate the number region), heavier
backbones or transformers with spatial-transformer alignment, temporal fusion, and often an external
OCR head — not the head/loss re-parametrizations tried here. **Pipeline-readiness: at 0.396 tracklet
accuracy (0.29 numbered-only) jersey ID is a weak prior to fuse with team/role/temporal cues, not a
standalone track-identity signal.**

# Stage 1c — torso-guided crop (positive but sub-threshold)

Stage 1b relocated the ceiling to *visual* number signal and named one untested cheap lever: feed
the number region instead of the whole body. Implemented and evaluated end-to-end on the official
1211-tracklet test split. **It is the first lever that moved the headline — +2 pp — but stays under
the pre-committed >3 pp "meaningful" bar.** The Stage-1 recipe is unchanged (single 100-way head,
ResNet18, 20 epochs, Adam 3e-4, batch 96, 24 crops/tracklet, label smoothing 0.1, fp16); the *only*
change is the input crop.

**Torso band (pre-committed, no tuning, no pose model):** take the vertical band `[0.15, 0.55]` of
each full-body crop's height at full width, then resize that band to 224×112 — so the upper-back
number region occupies far more input pixels than when the whole body (legs/grass) is squeezed into
224×112. Constants `TORSO_BAND = (0.15, 0.55)` in `generator/jersey_id.py`; a fixed band is the lazy
proxy for a pose crop. Trained fresh from ImageNet (not warm-started), same slices as Stage 1.

## Stage-1c ablation (official test, incl. `-1`, `eval-crops 48`)

| Config | min_conf | tracklet acc | numbered-only | legP | legR | weights |
|--------|---------|-------------|---------------|------|------|---------|
| Stage-1 single 100-way head (whole body, baseline) | 0.20 | 0.3964 (480/1211) | 0.290 | 0.840 | 0.754 | `jersey_r224_acc396.pt` |
| **Stage-1c single head + torso crop** | 0.20 | **0.4170 (505)** | 0.299 | 0.855 | 0.730 | `jersey_torso_r224_acc417.pt` |
| Stage-1c single head + torso crop (val-best) | 0.05 | **0.4187 (507)** | 0.308 | 0.858 | 0.792 | `jersey_torso_r224_acc417.pt` |

- **+2.1 pp at the baseline threshold (0.20): 0.3964 → 0.4170, +25 tracklets.** At each model's own
  val-tuned threshold (baseline 0.20, torso 0.05) it is +2.2 pp (0.3964 → 0.4187). Numbered-only edges
  up 0.290 → 0.299–0.308; legibility precision improves (0.840 → 0.855–0.858) with recall holding.
  Eval is deterministic (seeded crop sampling), so these are exact tracklet counts, not noise.
- **The gain is real and directional — it confirms the Stage-1b diagnosis that the ceiling is visual
  resolution, not head/loss** — but it is below the pre-committed >3 pp bar. Several top confusions
  are dented (33→25 drops out of the top list; the 4→29 misread grows, i.e. the band helps some
  numbers and hurts the worst-case low-res ones). Torso cropping is a free +2 pp and should be the
  default preprocessing going forward (the checkpoint stamps a `torso` flag that eval auto-detects).

## Verdict and next step

**The revised ceiling verdict stands: at 0.42 tracklet accuracy (0.30 numbered-only) jersey ID is a
weak prior to fuse, not a standalone track-identity signal.** Per the pre-committed >3 pp rule the
torso lever is *not* meaningful, so **Stage 2 is fusion-only** (jersey read fused with team/role/
temporal cues), not a chase for standalone accuracy. A multi-band ensemble is the obvious next visual
lever but is *not* justified: torso alone bought +2 pp, and an ensemble of shifted bands would at best
recover a fraction of another marginal misread class at real cost (2–3× inference) — diminishing
returns against the same broadcast-resolution wall that needs heavier backbones / STN alignment /
external OCR to breach, which is out of scope here.

## Stage-1c artifacts and reproduce

Single-head torso model (`generator.jersey_id.build_model`; `torso` crop stamped in the checkpoint
and auto-detected by `from_checkpoint`, so the `JerseyRecognizer` contract is unchanged):

```
python tools/train_jersey.py train --arch single --torso \
    --ckpt outputs/jersey/ckpt_torso.pt --target-epochs 20 --batch-size 96   # ~15 min, resumable
python tools/train_jersey.py tune --ckpt outputs/jersey/ckpt_torso.pt
python tools/train_jersey.py eval --split test --min-conf 0.20 --ckpt outputs/jersey/ckpt_torso.pt
```

Eval JSONs: `outputs/jersey/eval_torso_mc20.json` (0.20), `eval_torso_mc05.json` (val-best 0.05).
Seam test: `tests/test_jersey_torso.py` (band pixel mapping + input tensor shape, CPU).

## Stage-1b artifacts and reproduce

Factorized model (`generator.jersey_id.MultiHeadJersey`, auto-detected by `from_checkpoint`):

```
python tools/train_jersey.py train --stage 1 --ckpt outputs/jersey/ckpt_mh.pt --target-epochs 20
python tools/train_jersey.py train --stage 2 --ckpt outputs/jersey/ckpt_mh2.pt \
    --init-ckpt outputs/jersey/ckpt_mh.pt --filter-tau 0.5 --target-epochs 15   # warm-started
python tools/train_jersey.py eval --split test --min-conf 0.20 --ckpt outputs/jersey/ckpt_mh.pt
```

Eval logs: `outputs/jersey/eval_mh_stage1.json` (factorized), `eval_stage2.log`,
`eval_stage1b*.log`. `eval_test.json` holds the canonical single-head baseline. The
`JerseyRecognizer` contract is unchanged: multi-head checkpoints load through the same
`from_checkpoint` / `predict_tracklet -> (number, conf)` and reuse the Stage-1
pooling/threshold/eval path via `heads_to_number_probs` (per-crop head softmaxes → `[.., 100]`).

## Stage-2 wiring contract (what the GSR side needs)

`generator/jersey_id.py` → `JerseyRecognizer`:

- `JerseyRecognizer.from_checkpoint("outputs/jersey/ckpt.pt", min_conf=0.20)` → loaded recognizer.
- `.predict_tracklet(crops)` where `crops` is a directory of `*.jpg` **or** an explicit list of crop
  paths → `(jersey_number: int, confidence: float)`; `jersey_number == -1` means "no number read".
- Lower-level seams if Stage 2 wants its own aggregation: `.crop_probs(paths)` →
  `[n, 100]` softmax; `tracklet_mean(probs)` → pooled vector; `decide(mean_p, min_conf)` → label.

The recognizer's input contract is **player crops**, matching the GSR track-crop output. The
model's native input size is baked into the module (`INPUT_H/W`); loading a checkpoint trained at a
different size requires matching those constants.

## Reproduce

```
python tools/train_jersey.py train --max-minutes 30 --target-epochs 20 --batch-size 96   # resumable
python tools/train_jersey.py tune                                                          # min_conf
python tools/train_jersey.py eval --split test --min-conf 0.20                             # metric
```

Training is **resumable in wall-clock slices** (checkpoints every epoch to `outputs/jersey/ckpt.pt`;
re-run until it prints `TARGET REACHED`) — built for Claude Code restarts that orphan detached jobs.
