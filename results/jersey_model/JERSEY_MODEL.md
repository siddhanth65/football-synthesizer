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
