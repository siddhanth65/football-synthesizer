# Cluster session 1 — PRTreID retrain campaign (2026-08-03)

First session on the college GPU server. Goal per `docs/CLUSTER_RUNBOOK.md` §3b item 1: flip
sn-gamestate's `training_enabled` and retrain PRTreID on SoccerNet-GSR train, to get "the measured
floor the CLIP work must beat" before writing any CLIP code.

**Headline: the lever is closed for re-ID.** Three 20-epoch runs and a floor evaluation say the
flag-flip retrain does not beat the checkpoint it initialises from, and GSR train alone cannot
rebuild that embedding from scratch. One thread stays open (a from-scratch team head, +4.71
retrieval mAP) and one closes hard (the GK->team link, negative in both estimators tested).

---

## 1. Hardware (probed — supersedes the runbook's OMNI/SLURM assumptions)

`a100server1`, direct process execution, **no SLURM**. Ubuntu 6.8.0-134, driver 595.71.05,
CUDA 13.2 runtime, 128 CPU cores, 251 GB RAM, 500 GB user quota.

- GPU 0: A100-PCIE-40GB — another user, 14.6-23.6 GB in use all session. **Never touched.**
- GPU 1: A100-PCIE-40GB — another user holds ~2.1 GB / ~38% continuously; we ran alongside at
  `CUDA_VISIBLE_DEVICES=1`, peaking 11.2 GB total (ours ~9.2 GB) at 95-97% util.
- `torch.cuda.get_device_capability()` -> `(8, 0)`. sm_80. **The runbook's H100/B200 torch-pin
  landmine (§1.2, §4a) does not apply here.** UNKNOWN #9 stays untested and is now moot for this
  machine.
- End-of-session: quota 57,382 MB / 500 GB; volume 875 GB free (92% used globally, unchanged);
  `~/runs` 13 GB, `~/data` 32 GB, `~/models` 536 MB.

## 2. Environment — modern torch, no legacy env needed

`~/miniconda3/envs/gsr`, python 3.11.15. **sn-gamestate's pinned stack (`torch==1.13.1`,
python <3.10, mmocr, mmdet, easyocr, openmim) was never installed.** The whole reid training path
runs on torch 2.5.1+cu121. Runbook §1.4's "legacy env half-day" is not needed for this task.

Installed: `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, `numpy 1.26.4`, `pandas 2.2.3`,
`opencv 4.11.0`, `scipy 1.14.1`, `albumentations 1.3.1`, `torchmetrics 0.10.3`,
`setuptools 80.10.2`, `hydra-core 1.3.4`, `omegaconf 2.3.1`, `scikit-learn 1.9.0`,
`tracklab 1.3.24` (editable), `prtreid 1.3.1` (editable, Cython `rank_cy` built),
`sn-gamestate 0.2.0` + `tracklab-calibration 1.0.2` (editable, `--no-deps`).

Five blockers, each with the fix that cleared it:

| Blocker | Fix |
|---|---|
| pip 26 resolved numpy 2.4 / pandas 3.0 / opencv 5.0 — 2022-era torchreid code breaks on all three | `~/constraints.txt` pinning the 2024-era scientific stack; `PIP_CONSTRAINT` set on every install |
| `requires-python = ">=3.9,<3.10"` blocks the editable install even with `--no-deps` | `sed` the pin to `>=3.9` in **their server-side clone only** (`pyproject.toml.orig` kept alongside); same for `plugins/calibration` |
| `albumentations 2.x` removed `albumentations.functional`, which `prtreid/data/data_augmentation/random_occlusion.py` imports | pin `albumentations==1.3.1` |
| `torchmetrics 0.10.3` imports `pkg_resources`, absent from pip-26 envs | `setuptools<81` |
| `MissingConfigException: Cannot find primary config 'soccernet'` | tracklab's `hydra_plugins/` namespace dir sits at repo root and is **not packaged** by its editable install -> `export PYTHONPATH=$HOME/src/tracklab` |

**Unfixed upstream landmine:** `tracklab.utils.download.download_file` calls
`requests.get(url, stream=True)` with **no timeout**. It hung on Zenodo for 11 minutes with zero
output; `curl -I` on the same URL answered in under a second. Pre-fetch the two weights with
`curl -L -C -` before launching anything:

- `prtreid-soccernet-baseline.pth.tar` (396 MB, md5 `9633825232bc89f23a94522c5561650e`)
- `hrnetv2_w32_imagenet_pretrained.pth` (166 MB, md5 `58ea12b0420aa3adaa2f74114c9f9721`)

Both md5s match the values hardcoded in `prtreid_api.download_models`, so it then short-circuits.

## 3. Data mirror — verified

No rsync on the Windows laptop. Used three parallel `tar -cf - SNGS-xxx | ssh ... tar -xf -`
streams, resumable per sequence.

- **30.85 GB, 164/164 sequences, 34m41s wall, ~15 MB/s aggregate.** Parallelism bought nothing over
  a single stream — link-capped, not disk-capped.
- Server-side integrity loop: every `SNGS-*` has exactly **750** `img1/*.jpg` and a
  `Labels-GameState.json`. Zero bad, zero missing, zero transfer failures.
- `~/data/SoccerNetGS/{train,valid,test}/` built as symlinks into the flat mirror:
  **train 57 / valid 58 / test 49**, 0 missing, 0 incomplete. Loader confirms
  `SoccerNetGameState= train set: 57; valid set: 58; test set: 49`.

## 4. Two config traps in the flag-flip path

Both silently produce a run that looks fine and answers a different question.

1. **`training_enabled: True` alone does not train.** `engine_run_kwargs` maps
   `test_only = cfg.test.evaluate` (`prtreid/scripts/default_config.py:340`), and `prtreid.yaml`
   ships `test.evaluate: True`. You get an eval-only run. You must also set
   `modules.reid.cfg.test.evaluate=False`.
2. **`dataset.nvid` defaults to `1`** in sn-gamestate's plugin dataset config (tracklab's own
   `default.yaml` says `-1`). Without `dataset.nvid=-1` you train on one video — 25 ids, 370 crops.
   Cost us one launch.

Two more that matter once you start tuning:

3. **`fixbase_epoch` is a no-op unless `open_layers` is set**, and `open_specified_layers` does
   `assert hasattr(model, layer)`. Their default `open_layers=['classifier']` is **not** a
   top-level `BPBreID` attribute — `self.classifier` only exists inside the inner
   `BNClassifier`/`PixelClassifier` submodules. The real heads are `global_identity_classifier`,
   `global_team_classifier`, `global_Role_classifier`.
4. **`stepsize=[40,70]` with `max_epoch=20` means the LR never decays.** prtreid's defaults assume
   `max_epoch=120`; sn-gamestate overrides `max_epoch` to 20 and leaves stepsize alone.

Also patched in their server-side clone (`avgmeter.py.orig` kept):
`prtreid/utils/avgmeter.py:273` read a CUDA event before it completed, hard-crashing **every**
eval on torch 2.5 with `RuntimeError: CUDA error: device not ready`. Their own
`# torch.cuda.synchronize()  # TODO Check if slows down computation` sits commented out one line
above. Fix is `self.end_event.synchronize()`; cost is invisible in the timings.

## 5. The dataset the flag-flip builds

Their filtering law, unmodified (`min_vis 0.3`, `min_h/min_w 30`, uniform tracklet sampling to 15,
`min_samples_per_id 4`, seeded id cap):

- 733k GT boxes -> **162,556 dropped for h/w < 30 px**, 550,359 dropped by uniform tracklet
  sampling, 19 dropped for < 4 samples/id.
- **train 1343 ids / 20,067 crops / 57 cameras; query 2845, gallery 11,376 over 58 valid videos.**

That 162k size-filter drop is a de-facto answer to the runbook's "GPU inference returns dummy boxes
[0,0,1,1]" landmine (§3b item 5) — box geometry varies normally here. That failure mode is
inference-side only and remains untested.

## 6. The floor

Shipped `prtreid-soccernet-baseline.pth.tar`, evaluated test-only on the **full valid split**
through the same harness every run below uses:

| metric | value |
|---|---|
| REID mAP | **56.93** |
| REID Rank-1 | **73.95** |
| Team-affiliation mAP | **78.17** |
| Team Rank-1 | 97.14 |
| Role accuracy | 75.56 |
| SSMD | 4.4455 |

## 7. Run 1 — their documented recipe, as shipped

`test.evaluate=False`, everything else at sn-gamestate's values (lr 3.5e-4, no fixbase,
stepsize [40,70] i.e. no decay). 20 epochs, 50m43s.

Loss **climbs monotonically** through the 10-epoch LR warmup and never returns:
1.197, 1.212, 1.238, 1.289, 1.301, 1.384, 1.395, 1.449, 1.437, 1.538, 1.565, ..., 1.461.

Valid REID mAP: 58.27, 58.07, 58.09, 55.80, 55.19, 56.82, 57.35, 56.67, 54.79, **55.14** (final).

| | floor | run 1 final | delta |
|---|---|---|---|
| REID mAP | 56.93 | 55.14 | **-1.79** |
| REID Rank-1 | 73.95 | 72.34 | -1.61 |
| Team mAP | 78.17 | 74.94 | **-3.23** |
| Role accuracy | 75.56 | 76.51 | +0.95 |
| SSMD | 4.4455 | 4.9168 | +0.47 |

**Run at their own documented recipe, the flag-flip retrain does not establish a floor — it lands
below the checkpoint it initialises from.** Cause is legible in the trace: the recipe restarts a
warmup to 3.5e-4 on an already-converged checkpoint, and mAP falls in lockstep with the rising
loss. This contradicts the runbook's §3b framing.

## 8. Probe 1 — LR-corrected fine-tune

Delta: `train.lr=3.5e-5` (10x down), `fixbase_epoch=5`,
`open_layers=[global_identity_classifier,global_team_classifier,global_Role_classifier]`,
`stepsize=[14,18]`. 44m24s.

Loss shape **fixed** — 3.052, 3.054, 3.016, 2.987, 2.957 (frozen backbone, heads only), then on
unfreeze 1.195, 1.201, 1.199, 1.204, 1.207, 1.207, 1.212, 1.203, 1.204, 1.195, 1.190, 1.192,
1.193, 1.187, 1.188. Flat-to-descending, never climbs.

Valid REID mAP: 57.11, 57.11, 57.11, 57.63, 56.80, 56.88, **58.26**, 57.87, 56.31, 56.46 (final).
Valid team mAP: 78.08, 78.08, 78.08, 76.22, 76.00, 78.64, **78.88**, 77.53, 75.49, 77.90 (final).

| | floor | final (ep 20) | best (ep 13) |
|---|---|---|---|
| REID mAP | 56.93 | 56.46 (-0.47) | **58.26 (+1.33)** |
| REID Rank-1 | 73.95 | 73.95 (0.00) | — |
| Team mAP | 78.17 | 77.90 (-0.27) | **78.88 (+0.71)** |
| Role accuracy | 75.56 | 76.51 (+0.95) | — |
| SSMD | 4.4455 | 4.4344 (-0.01) | — |

**The LR fix converts monotonic degradation into a noise band that straddles the floor.**
Post-unfreeze the series oscillates 56.31-58.26 with no trend; epoch 13's +1.33 is the top of that
band, not a demonstrated improvement, and the final model is still marginally below the floor.
Shipping epoch 13 would be selecting on the same valid split we would then report.

**Free gem: BatchNorm running-statistic adaptation is worth +0.18 mAP with zero gradient steps.**
Epochs 1/3/5 all read exactly **57.11** with the backbone frozen, versus **56.93** for the
untouched baseline. Identical weights — `requires_grad=False` does not stop BN running stats
updating in train mode. Cheapest domain-adaptation lever on record; relevant to the eventual
EPL/ManU transfer, where no labels exist at all.

## 9. Probe 2 — from-scratch floor (ImageNet hrnet32 init)

Delta: `model.load_weights=""`, `model.load_config=False`, `stepsize=[14,18]`, otherwise their
documented recipe (lr 3.5e-4). 47m56s.

Init verified in the log: `=> init weights from normal distribution` then
`=> loading pretrained model .../hrnetv2_w32_imagenet_pretrained.pth`, and **no**
`Successfully loaded pretrained weights from .../prtreid-soccernet-baseline.pth.tar` line (present
in every other run). Corroborated by the loss starting at **10.979** vs 1.197 when warm-started.

Loss descends monotonically and is still descending at epoch 20 (not converged): 10.979, 9.048,
7.795, 7.147, 6.629, 6.073, 5.621, 5.247, 5.271, 5.277, 4.674, 4.558, 4.278, 4.281, 3.721, 3.553,
3.506, ...

Valid REID mAP: 26.15, 47.90, **55.80**, 54.59, 42.15, 48.54, 48.17, 39.08, 47.72, 43.12 (final).
Valid team mAP: 71.40, 81.22, **82.88**, 78.21, 73.53, 78.72, 77.52, 78.01, 78.98, 76.78 (final).

| | floor | final | best (ep 5) |
|---|---|---|---|
| REID mAP | 56.93 | 43.12 (-13.81) | 55.80 (-1.13) |
| REID Rank-1 | 73.95 | 58.24 (-15.71) | — |
| Team mAP | 78.17 | 76.78 (-1.39) | **82.88 (+4.71)** |
| Role accuracy | 75.56 | 62.34 (-13.22) | — |
| SSMD | 4.4455 | 2.9187 | — |

**57 GSR train clips / 20,067 crops / 1343 ids is not enough to reproduce the shipped checkpoint's
identity embedding.** From-scratch tops out ~1.1 mAP below it at epoch 5, then degrades under the
un-decayed plateau. The shipped checkpoint is therefore not a 1%-subset artifact (cf.
`docs/SOCCERNET_REPO_SWEEP.md` §4 on sn-reid's `training_subset: 0.01` default) — it carries
supervision this split cannot replicate in 20 epochs, and there is no easy re-ID headroom for a
full-data retrain to claim.

**But the team head is not bottlenecked by the same thing.** From scratch it reaches **82.88 team
mAP at epoch 5, +4.71 over the floor** — the largest gain anywhere in this session — while its
re-ID is still poor (55.80). This is the one thread that clears the floor by more than noise.

Note: probe 2's *final* (epoch 20) checkpoint emits some zero-norm embeddings (NaN after L2
normalisation) — consistent with its collapsed mAP 43.12 / SSMD 2.92. Do not reuse it.

## 10. The GK->team link — negative in both estimators

Runbook §4b names this a mandatory eval target: `results/GSR_TEAMSIDE.md` has the keeper rule at
113/113 on ground truth but blocked at **0.244** link accuracy with kit-KMeans, and converting it
is worth ~+2.4 GS-HOTA via the side resolver. Two independent estimators tested; both negative.

**Data for both:** 1,124 goalkeeper crops from 57 of 58 valid videos, cut fresh from GT
`bbox_image` (min 30 px, uniform 15-per-track sampling — the same filtering law as
`prtreid_dataset`). GT distribution left 525 / right 599, so the **majority-class floor is 0.5329**.

### 10a. Team-head argmax (`~/gk_team_probe.py`)

`model_output[3]["globl"]` — the team head; `extract_test_embeddings` discards it, so the raw tuple
is read directly. Only the identity classifier is dropped on load (num_classes mismatch); the team
head loads intact. Scored with an **oracle class->team mapping** (each of the 6 head classes
assigned to whichever team is the majority among its own predictions) — a deliberate upper bound.

| model | GK team-head accuracy (oracle map) | pred class histogram (6 classes) |
|---|---|---|
| majority-class guess | 0.5329 | — |
| shipped baseline | **0.6343** | [36, 768, 13, 4, 59, 244] |
| probe 1 ep 13 | 0.5721 | — |
| probe 1 ep 20 | 0.5649 | — |
| probe 2 ep 5 | 0.5801 | — |

Best is +7.4 points over the majority floor on a two-class problem, with an oracle relabelling
already granted, and no trained checkpoint beats the shipped baseline. The head is also
structurally mismatched: `BPBreID.__init__` carries `num_teams=6,  # FIXME hardcoded` while GSR has
two teams, and predictions scatter across all six outputs.

> **CORRECTED 2026-08-04 (session 2).** The numbers above are the re-run under correct RGB input.
> The originally published figures (baseline 0.6343, probe1 ep13 0.5819, ep20 0.6343, probe2 ep5
> 0.6041) were computed feeding **BGR** crops to `FeatureExtractor`, which only applies
> `Image.open(..).convert("RGB")` for `str` input and does no channel swap for a numpy array. The
> bug was caught in session 2 when the new evaluator failed to reproduce the 56.93 floor (it read
> 35.31 until the channel order was fixed). Direction and verdict are unchanged; every number moved.

### 10b. Retrieval / unsupervised kit clusters (`~/gk_link_probe.py`)

The stronger test, because probe 2's 82.88 was a *retrieval* metric: the embedding space may
separate teams even where the head cannot. Per video: embed outfield-player and goalkeeper crops,
fit **KMeans k=2 on the outfield embeddings only** (no labels used to build the clusters — mirrors
what our kit-clustering pipeline can actually do), map each cluster to a team by outfield majority,
link each keeper crop to the nearer centroid. Per-tracklet is a majority vote over a keeper's
crops — the estimator our side resolver would consume.

| model | GK per-crop | GK per-tracklet | per-video mean / median | CONTROL: outfield cluster purity |
|---|---|---|---|---|
| majority-class floor | 0.5329 | 0.5329 | — | — |
| shipped baseline | 0.3443 | 0.3158 | 0.3363 / 0.2000 | **0.9266** (n=16,776) |
| probe 2 ep 5 (best team mAP) | **0.5142** | **0.5395** | 0.4947 / 0.5000 | **0.9637** (n=16,776) |
| probe 1 ep 20 | 0.3603 | 0.3421 | — | 0.9128 (n=16,776) |

n = 1,124 GK crops / 76 keeper tracklets / 57 videos. **All figures re-run under correct RGB input
on 2026-08-04** — see the correction note in §10a; the first published version of this table
(baseline 0.4546/0.4605, probe2 ep5 0.3630/0.3553, probe1 ep20 0.4555/0.4211, controls 0.9053 and
0.8917) fed BGR crops and is withdrawn. The RGB correction **changed every number and flipped the
ordering between the two models**, which is exactly why it is recorded rather than quietly patched.

**The control makes this decisive.** The *same* clusters, built the *same* way, separate outfield
teams at **91-96%** — the machinery is sound. The best keeper link is **0.5395 per-tracklet against
a 0.5329 majority-class floor: +0.0066, or about half a tracklet out of 76.** It is
indistinguishable from guessing whichever team is more common, and the shipped baseline is far
*below* chance (0.3158). Per-tracklet voting does not rescue it: nothing approaches the ~0.9 the
side-resolver path needs.

**Verdict: the +2.4 GS-HOTA side-resolver path does NOT reopen via a trained PRTreID embedding.**
A goalkeeper's kit is required to differ from both outfield kits, so an appearance-only nearest-kit
link carries no usable signal about which team he belongs to. This is a *reason*, not just a
number: any future GK->team work should use geometry/position or an explicitly GK-aware objective,
not appearance similarity to outfield players.

**What the correction did and did not change.** Unchanged: the verdict, the control's validity, and
the mechanism. Changed: the claim that the best team-mAP checkpoint is *systematically
anti-informative*. Under RGB it is the best of the three linkers (0.5395), and the *shipped
baseline* is the sub-chance one. "Anti-informative" was an artifact of the colour bug; the correct
statement is that no embedding tested carries GK->team signal above the majority floor.

**Comparability caveat, stated plainly:** none of these numbers is a reproduction of
`GSR_TEAMSIDE.md`'s 0.244. That figure is our own pipeline's end-to-end keeper->team link accuracy
over tracklets from detected (not GT) boxes with our own kit clustering. These are per-crop and
per-tracklet accuracies over GT-boxed crops with PRTreID embeddings. **The numbers must not be
subtracted.** What transfers is the direction and the mechanism, both of which agree with the 0.244
result rather than overturning it.

## 11. Artifacts

All on the server (`siddhanth23519@a100server1`); nothing from this session is vendored into this
repo, and the two upstream patches live only in the server-side clones with `.orig` copies beside
them.

- Logs: `~/logs/prtreid_train.log` (run 1), `~/logs/probe1_lr.log`, `~/logs/probe2_scratch.log`,
  `~/logs/baseline_eval.log` (the floor), `~/logs/smoke2.log`, `~/logs/mkenv.log`,
  `~/logs/mkdeps.log`, `~/logs/getw.log`
- Launcher: `~/run_prtreid.sh`; split builder: `~/mksplits.py`
- Probes: `~/gk_team_probe.py` (head argmax), `~/gk_link_probe.py` (retrieval + outfield control)
- Checkpoints, epochs 1,3,...,19,20, ~400 MB each:
  - run 1: `~/runs/outputs/prtreid_gsr_e20/2026-08-03/20-20-15/reid/0/2026_08_03_20_21_42_21S565da0e6-d4fc-432c-99d1-0df352692e90model/job-0_N_model.pth.tar`
  - probe 1: `~/runs/outputs/prtreid_lr35e6/2026-08-03/21-37-42/reid/0/2026_08_03_21_38_36_38S64a717c2-a95d-47ac-9d42-5ec63d6a3231model/job-0_N_model.pth.tar`
  - probe 2: `~/runs/outputs/prtreid_scratch/2026-08-03/22-24-58/reid/0/2026_08_03_22_26_13_26Se3283d8d-9351-4437-aaef-13b64f0b16b3model/job-0_N_model.pth.tar`

Timing for planning: 20 epochs + 10 valid evals = **~45-50 min** on one shared A100. Per-batch
240 ms (118 ms optimizer + 106 ms forward + 4 ms data loading — not input-bound at 8 workers);
each full-valid eval 50 s. A full recipe run fits comfortably in a 2-hour slot.

## 12. What this changes

- **Runbook §3b item 1 is answered and closed.** The flag-flip PRTreID retrain is not a floor to
  beat; it is below the shipped checkpoint at the documented recipe, and inside noise at a
  corrected one. Do not spend further GPU time on it.
- **Runbook §4b's mandatory GK->team eval target is answered: negative, with a mechanism.**
  Appearance-similarity-to-outfield cannot link keepers; the control run rules out an
  implementation fault.
- **The one open lever is the team head trained from scratch (+4.71 retrieval mAP at epoch 5).**
  Note it is *team affiliation retrieval*, not the argmax head and not the GK link — scope it
  carefully before spending a slot.
- Runbook §1.4's legacy-env contingency and §1.2's sm_90 landmine are both moot on this machine.
