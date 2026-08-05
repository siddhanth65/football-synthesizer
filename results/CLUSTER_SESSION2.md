# Cluster session 2 — the S3 CLIP+heads build (2026-08-03/04)

Session 1 (`results/CLUSTER_SESSION1.md`) closed the PRTreID flag-flip lever and fixed the floor at
the shipped `prtreid-soccernet-baseline.pth.tar`: **56.93 REID mAP / 78.17 team mAP** on the full
58-video GSR valid split. This session builds our own trainer per `docs/CLUSTER_RUNBOOK.md` §3 +
§3b + §4b: clean-room crop builder, CLIP ViT-B/16 + attribute heads, a verified evaluator, and a
full training run.

**Headline: everything was built and runs end to end, and this run does NOT clear the floor.**
Best identity mAP **53.75 (-3.18 vs 56.93)**; best team mAP **79.09 (+0.88 vs 78.17)**, but only at
2 epochs, decaying with training. Two secondary results are worth more than the headline: the
clean-room builder **reproduces sn-gamestate's crop set exactly**, and the evaluator caught a
colour-channel bug that **invalidated four numbers published in session 1** (corrected there).

---

## 1. Evaluator — built first, and it failed its verification twice before passing

Per the brief, the harness was wired and validated *before* any training. `~/work/eval_reid.py`
reimplements sn-gamestate's `mot_intra_video` protocol (market1501 AP/CMC, gallery restricted to
`g_camids == q_camid`) over a persisted split, and is scored against the known floor.

The split itself cannot be read off disk — it is built in memory by a seeded groupby sample — so
`~/work/dump_split.py` runs their builder **once** and persists
`(pid, camid, img_path, role, team, jersey)` to `~/work/gsr_reid_split.json`. Every later evaluation
reads the JSON and never touches their code. Dumped counts: **train 20,067 / query 2,845 / gallery
11,376**, matching the training runs exactly.

**Verification history — both failures were real bugs in my code, not tolerance issues:**

| attempt | identity mAP | team mAP | cause |
|---|---|---|---|
| target (their harness) | **56.93** | **78.17** | — |
| 1st | 35.31 | 75.25 | fed **BGR** arrays; `FeatureExtractor` only does `convert("RGB")` for `str` input and does no channel swap for ndarray |
| 2nd | 56.91 | 71.21 | identity fixed by passing paths; team metric still wrong |
| 3rd | **56.91** | **78.21** | their team metric drops `team == -1` (referee/ball/other) from **both** sides first (`part_based_engine.py:419`) |

Final agreement: identity **56.91 vs 56.93** (R1 74.06 vs 73.95), team **78.21 vs 78.17** (R1 96.89
vs 97.14). Residual deltas are argsort tie-ordering. **Evaluator verified.**

**This is the session's most valuable output.** The BGR bug also affected session 1's goalkeeper
probes, which fed `cv2.imread` arrays the same way. Every GK number in `CLUSTER_SESSION1.md` §10 has
been re-run under RGB and corrected in place; the verdict survived but **every number moved and the
ordering between two models flipped**. Without the "reproduce the floor first" gate, the CLIP
numbers below would have been quoted off a broken evaluator.

## 2. Clean-room crop builder — reproduces their set exactly

`~/work/build_crops.py`, written from the contract documented in `docs/SOCCERNET_REPO_SWEEP.md`
§1.1, not from `prtreid_dataset.py` (GPL-3.0): drop `visibility < 0.3`, drop `w` or `h <= 30 px`,
uniform-subsample each tracklet to 15 (first and last always kept), drop ids with `< 4` crops,
resize anything above 256x128 down. Resumable — crops already on disk are not re-cut.

| | ours | sn-gamestate's |
|---|---|---|
| crops | **20,067** | 20,067 |
| identities | **1,343** | 1,343 |
| videos | **57** | 57 |

**Exact match on all three.** That is an independent confirmation that the law recorded in the repo
sweep is complete and correct — the documented spec is sufficient to rebuild their dataset without
reading their code.

Label distribution: roles `player 17,099 / referee 1,702 / goalkeeper 1,148 / other 60 / ball 58`;
teams `left 9,080 / right 9,167 / None 1,820`; **13,994 of 20,067 crops (69.7%) carry a jersey
number**. Build time 5m58s, 97 MB on disk.

**One spec finding:** SoccerNet-GSR annotations carry **no `visibility` field** (keys are
`attributes / bbox_image / bbox_pitch / bbox_pitch_raw / category_id / id / image_id / supercategory
/ track_id`). The `min_vis 0.3` rule is therefore **inapplicable** on this dataset — every GT box is
fully visible by construction. The rule is implemented and recorded for parity, and filters nothing.

## 3. Model

`~/work/clip_model.py`. CLIP ViT-B/16 visual tower (`openai/clip-vit-base-patch16`, 85.8 M params,
hidden 768) via `transformers` — already installed, so no new dependency and no `open_clip` install
risk. One **shared 256-d embedding** (Linear 768->256 + BN + L2-norm) feeds every consumer, because
the harness scores identity *and* team retrieval off the same feature; the heads shape that
embedding rather than being read directly.

| head | form | loss weight |
|---|---|---|
| identity | ArcFace over 1,343 pids (scale 30, margin 0.3) | 1.0 |
| team | CE over **(video, team) composite** classes, 114 of them | 0.5 |
| jersey | CE over 48 observed numbers **+ an explicit no-number class** | 0.5 |
| role | CE over 5 roles | 0.5 |

Two design choices are direct consequences of recorded findings:

- **The team head is per-video, not global.** `CLUSTER_SESSION1.md` §10a documented PRTreID's
  6-class global team head as structurally mismatched to GSR's two per-video sides. Supervising on
  `(video, team)` makes the objective "which of the two kits *in this video*", which is what the
  team metric actually measures.
- **Jersey abstention is a class, not a dropped sample.** Consistent with the project's abstention
  discipline, the 6,073 crops without a number train the no-number class rather than being skipped.

Progressive unfreeze (`set_trainable`): heads only for the first N epochs, then the last 6
transformer blocks + `post_layernorm` at **1e-5** against the heads' **1e-3** — the probe-1 lesson
(a converged init degrades under a restarted high LR) applied to a much stronger init.

`transformers` 5.x note: `CLIPVisionModel` no longer nests the tower under `.vision_model`;
`set_trainable` handles both layouts. Also, `from_pretrained` requires `use_safetensors=True` here —
transformers 5 refuses `torch.load` on torch < 2.6 (CVE-2025-32434), and our env is pinned at 2.5.1.

## 4. Training

`~/work/train_clip.py`. bf16 autocast, AdamW (wd 0.05), batch 128, cosine LR with one epoch of
warmup, grad-clip 5.0, 8 dataloader workers, per-epoch checkpoints, `--resume` from `last.pt`.
Measured **13 s/epoch frozen, 26 s/epoch unfrozen** on one A100 shared with the co-tenant —
157 steps/epoch. GPU 1 only; GPU 0 never touched.

Run A (40 epochs, freeze 5, unfreeze 6): all four losses descend monotonically, no instability.

| epoch | 1 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| total | 20.05 | 11.23 | 6.12 | 2.42 | 1.09 | 0.90 |
| identity | 15.08 | 8.29 | 3.75 | 0.86 | 0.22 | 0.15 |
| team | 4.66 | 3.37 | 2.85 | 1.72 | 1.20 | 1.04 |
| jersey | 3.65 | 2.28 | 1.73 | 0.85 | 0.50 | 0.41 |
| role | 1.55 | 0.25 | 0.16 | 0.09 | 0.06 | 0.05 |

Run B (12 epochs, freeze 3, unfreeze 6) tests whether run A's peak was limited by an un-annealed LR
rather than by the data. Total run cost for both: ~25 min of GPU.

## 5. Results vs the floor

All numbers from the verified evaluator, full valid split, query 2,845 / gallery 11,376.

| model | identity mAP | R1 | team mAP | R1 |
|---|---|---|---|---|
| **floor — shipped PRTreID baseline** | **56.93** (ours: 56.91) | 73.95 | **78.17** (ours: 78.21) | 97.14 |
| untrained CLIP + random projection | 31.44 | 46.29 | 72.55 | 91.86 |
| run A, epoch 5 | 45.74 | 59.44 | 78.54 | 93.82 |
| run A, epoch 10 | **53.75** | 70.26 | 76.40 | 95.38 |
| run A, epoch 20 | 51.15 | 69.31 | 75.22 | 95.62 |
| run A, epoch 30 | 49.89 | 67.28 | 75.03 | 95.50 |
| run A, epoch 40 | 49.32 | 67.24 | 75.13 | 95.29 |
| run B, epoch 8 | 53.63 | **71.18** | 77.83 | 95.38 |
| run B, epoch 10 | 53.27 | 70.83 | 78.10 | 95.21 |
| run B, epoch 12 | 53.21 | 70.65 | 77.98 | 95.46 |
| (2-epoch smoke) | 48.99 | 63.09 | **79.09** | 94.19 |

**Identity: does not clear the floor.** Best is 53.75, **-3.18**. The curve peaks at epoch 10 and
declines monotonically thereafter — overfitting to 1,343 identities. Run B rules out the LR
schedule as the cause: annealed to a 12-epoch horizon it lands at 53.2-53.6, the *same* place.
**53.5 +/- 0.3 is where this configuration saturates.**

**Team: marginally above the floor, but only when barely trained.** 79.09 after two epochs and
78.54 at epoch 5 both beat 78.17; by epoch 10 it is below, and it decays to 75.1. Identity training
and team separation pull against each other in a shared embedding, and the identity loss wins.

**Training is doing real work.** Untrained CLIP + a random projection scores 31.44; our training
adds **+22.31 identity mAP** and +6.5 team mAP. The gap to the floor is not a broken pipeline — it
is a genuinely worse embedding than the shipped one.

## 6. Verdict, against the Table-5 hypothesis

The runbook §3 motivates S3 from Broadcast2Pitch Table 5: PRTreID+OCR 18.11 -> CLIP+heads **60.13**
GS-HOTA. **This run does not support that promise, and it also does not refute it.** Three reasons
to keep the two separate:

1. **Different metric.** Table 5's 60.13 is **GS-HOTA over a whole pipeline** (detector, tracker,
   calibration held fixed). Ours is **re-ID mAP on a crop-retrieval split**. A CLIP encoder can
   improve end-to-end GS-HOTA without beating PRTreID on this particular retrieval protocol, and
   vice versa. Nothing here licenses a 60.13-vs-53.75 comparison.
2. **One point in a five-dimensional unknown.** The runbook's UNKNOWN list #1-#5 (CLIP variant,
   frozen vs full fine-tune, per-head loss weights, epochs/LR, batch/augmentation/resolution) are
   all still unknown. I picked ViT-B/16, last-6-blocks, 1.0/0.5/0.5/0.5, 40 and 12 epochs, batch
   128 at square 224. That is one sample.
3. **The measured cause points elsewhere.** The failure mode is overfitting: identity peaks at
   epoch 10 and decays, on 1,343 identities and 20,067 crops. This is the *same* conclusion session
   1's probe 2 reached from the opposite direction — from-scratch hrnet32 on this split also
   plateaued below the floor (55.80). Two independent architectures now say **GSR train is too
   small to build a competitive identity embedding from a generic init**. The shipped PRTreID
   checkpoint's advantage is its SoccerNet re-ID pretraining, not its architecture.

**Highest-leverage next step, and it follows from the measurement rather than the paper:** more
identities, not more epochs. `docs/SOCCERNET_REPO_SWEEP.md` §15 item 4 already names the corpus —
SoccerNet-v3 / sn-reid's **340,993 jersey-labelled crops across 400 games in 6 leagues, MIT
licensed**, with a built-in illegible marker. That is ~17x our crop count and two orders more
identities, it directly attacks the measured failure mode, and it doubles as the EPL/ManU domain
answer. Training on GSR train alone should be considered closed.

Secondary lever, cheap: the 224-square resize discards the 2:1 person aspect ratio (marked
`ponytail:` in `clip_model.py`). 256x128 with interpolated position embeddings is the standard
CLIP-reID fix and is worth one run.

## 7. State and artifacts

Quota **71,609 MB / 500 GB**; volume 840 GB free (93% used globally — 20 GB of unevaluated
checkpoints pruned at end of session). GPU 0 untouched throughout; GPU 1 returned to the co-tenant's
2.1 GB / 40% baseline. No jobs of ours running.

Server-side (`siddhanth23519@a100server1`), nothing vendored into this repo:

- `~/work/dump_split.py` — persists their eval split once -> `~/work/gsr_reid_split.json`
- `~/work/build_crops.py` — clean-room crop builder -> `~/work/crops_train.json`, `~/data/gsr_crops/`
- `~/work/clip_model.py` — CLIP+heads, `load_for_eval`, `_self_check()` (`python clip_model.py`)
- `~/work/eval_reid.py` — the verified evaluator
- `~/work/train_clip.py` — trainer, resumable
- Logs `~/logs/clip_s3.log`, `~/logs/clip_s3_e12.log`
- Checkpoints `~/runs/clip_s3/epoch{5,10,20,30,40}.pt` + `epoch0_untrained.pt` (3.9 GB),
  `~/runs/clip_s3_e12/` (7.5 GB)

Self-checks left behind: `clip_model._self_check()` asserts the ArcFace margin actually lowers the
target logit (a sign error there trains silently and wrongly), the LR schedule endpoints, and the
transform's output shape. `build_crops.py` asserts the min/max samples-per-id law it implements and
that its manifest points at real files.
