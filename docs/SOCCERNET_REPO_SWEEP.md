# SoccerNet org - CODE-LEVEL repo sweep

Swept 2026-08-02. Scope: the 11 repos Sid listed, at **code level** (repo tree, README, key source files
via the GitHub API and raw.githubusercontent), plus a full scan of
https://github.com/orgs/SoccerNet/repositories?type=all for anything worth flagging.

This is **not** a paper survey. Method/paper findings live in `docs/GSR_METHODS_DEEP_DIVE.md` and
`docs/SOCCERNET_SURVEY_RAW.md`; this document does not repeat them and only cross-references where a
code finding corrects or extends one. Campaign needs it is judged against are in
`docs/GSR_CAMPAIGN_BRIEF.md`:

- **Need 1** - cluster training of a CLIP-encoder + attribute-heads identity model on GSR train
  (weights, training loops, data loaders, augmentation, crop utilities).
- **Need 2** - GSR benchmark climbing from official test 39.02 (DetA 26.47 / AssA 57.53): anything
  lifting detection, association, jersey, calibration.
- **Need 3** - later: player-centric action attribution (FOOTPASS / PCBAS).

All repo state (stars, last commit, open issues, licence) read from the GitHub API on 2026-08-02.

---

## 0. Verdict table

| Repo | What it really is | Last real commit | Open issues | Licence | Verdict |
|---|---|---|---|---|---|
| [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | GSR baseline + **GSR-to-crop dataset builder** + a live training path | 2026-05-02 | 20 (several load-bearing) | GPL-3.0 | **USE** - it already ships the crop/label pipeline need 1 requires |
| [SoccerNet-v3](https://github.com/SoccerNet/SoccerNet-v3) | Dataloader only, for a 400-game jersey-labelled bbox corpus | 2022-05-20 | 0 | MIT | **USE** - the out-of-domain identity training corpus GSR lacks |
| [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) | T-DEED + team head, full training code + checkpoint | 2025-08-26 | 2 | GPL-3.0 | **USE** (need 3) - the only org code that attributes actions to an agent |
| [sn-reid](https://github.com/SoccerNet/sn-reid) | torchreid fork; **no SoccerNet weights** | 2023-07-07 | 3 | MIT | **MINE-FOR-PARTS** - MIT train loop, sampler, augmentations, jersey-bearing filename parser |
| [sn-banner](https://github.com/SoccerNet/sn-banner) *(not on Sid's list)* | Banner replacement: NBJW calib + Mask2Former, **with direct weight URLs** | 2025-12-26 | 0 | GPL-3.0 | **MINE-FOR-PARTS** - weights we did not know existed + a stacked temporal cam-param filter |
| [sn-tracking](https://github.com/SoccerNet/sn-tracking) | Thin ByteTrack/DeepSORT wrappers; YOLOX Exp file | 2023-07-07 | 11 (worst health) | **none** | **MINE-FOR-PARTS** - YOLOX hyperparameters and sequence lists only; the "SoccerNet" recipe is mislabelled (S5.4) |
| [ActiveSpotting](https://github.com/SoccerNet/ActiveSpotting) | Active-learning loop over NetVLAD++ features | 2023-05-22 | 0 | MIT | **MINE-FOR-PARTS** - the labelled-budget sampler, if we ever hand-label crops |
| [sn-spotting](https://github.com/SoccerNet/sn-spotting) | Action spotting; committed checkpoints; ResNet-152+PCA feature extractor | 2024-02-07 | 3 (stale) | MIT (root) / Apache-2.0 (benchmarks) | **MINE-FOR-PARTS** - `Features/` extractor only; task is player-blind |
| [sn-calibration](https://github.com/SoccerNet/sn-calibration) | 2022/2023 calibration devkit; already vendored inside sn-gamestate | 2024-06-18 (README) | 3 (oldest 2022) | **none** | **IGNORE** - superseded by PnLCalib, which we already run |
| [PTS-baseline](https://github.com/SoccerNet/PTS-baseline) | **A fork** of jhong93/spot with SoccerNet ingestion bolted on | 2023-02-15 | 0 | BSD-3-Clause | **IGNORE** - superseded by sn-teamspotting, same codebase lineage |
| [sn-jersey](https://github.com/SoccerNet/sn-jersey) | **One file.** A README. | 2024-07-02 | 1 | **none** | **IGNORE** - contributes nothing beyond a download line we already have |
| [sn-grounding](https://github.com/SoccerNet/sn-grounding) | Replay grounding, feature-level, dead since 2022 | 2022-11-10 | 1 (2022) | MIT (root) / Apache-2.0 (benchmarks) | **IGNORE** - no player-level signal, task closed for us |

---

## 1. sn-gamestate - **USE**

https://github.com/SoccerNet/sn-gamestate | GPL-3.0 | 434 stars | last commit 2026-05-02 | 20 open issues

Baseline architecture and environment pins are already recorded in `GSR_METHODS_DEEP_DIVE.md` S3.2.
**Four things that pass did not surface, all found at code level this time.**

### 1.1 The GSR-to-labelled-crop dataset builder (the single most valuable artifact in the org)

`sn_gamestate/reid/prtreid_dataset.py` (26.8 KB) converts SoccerNet-GSR ground truth into a per-crop
identity dataset **carrying role, team and jersey_number as label columns**. That is exactly the
`(crop -> role/team/jersey)` supervision the Broadcast2Pitch Table-5 CLIP-encoder + attribute-heads
model consumes. We do not have to write a crop extractor for S3.

Call chain and what each stage does:

| Function (line) | Does |
|---|---|
| `build_reid_set` (161) | Walks a tracking split, precomputes crop/mask/annotation paths, resumes from an existing `reid_crops_anns.json` so re-running only generates newly-selected crops |
| `sample_detections_for_reid` (260) | The filtering law: drop `visibility < min_vis`; drop boxes with `w <= min_w` or `h <= min_h`; **uniform tracklet subsampling** to `max_samples_per_id` (`np.linspace` over the tracklet, first and last always kept); drop ids with fewer than `min_samples_per_id` crops; cap total ids under `np.random.seed(0)` so the split is reproducible across instantiations |
| `save_reid_img_crops` (322) | Crops from the source frame, resizes to `max_crop_size` with `INTER_CUBIC`, writes to disk, writes a JSON metadata sidecar |
| `query_gallery_split` (555) | Query/gallery split at `ratio_query_per_id` |
| `to_torchreid_dataset_format` (585) | Emits per-crop records `[pid, camid, img_path, masks_path, visibility, image_id, video_id] + columns`, factorising each attribute column to 0-based ints and keeping the inverse map. `camid = video_id`, so cross-video gallery filtering is free |
| `uniform_tracklet_sampling` (631) | The evenly-spaced tracklet sampler, standalone and liftable |

Driven by `sn_gamestate/configs/modules/reid/dataset/prtreid_dataset.yaml`:

```yaml
fig_size: [384, 128]          # PRTreID input; not binding on a CLIP head
max_crop_size: [256, 128]     # what actually lands on disk
mask_size: [64, 32]
masks_mode: "pose_on_img_crops"
columns: ["role", "team", "jersey_number"]
train: {min_vis: 0.3, min_h: 30, min_w: 30, min_samples_per_id: 4, max_samples_per_id: 15}
test:  {set_name: "valid", min_samples_per_id: 4, max_samples_per_id: 10, ratio_query_per_id: 0.2}
```

Note `min_h: 30` / `min_w: 30` - the organisers' own crop-size floor, which sits alongside Grad's
"exclude tracklets with average diagonal < 40 px" hygiene rule (`GSR_METHODS_DEEP_DIVE.md` S4.3.1).

**Licence caveat, load-bearing:** sn-gamestate is GPL-3.0. Lifting this file verbatim makes our repo
GPL-3.0. Options: (a) reimplement from the config semantics above - the filtering rules are six lines
of pandas and the crop writer is `cv2.resize` + `cv2.imwrite`; (b) keep it in an isolated,
clearly-GPL'd tool directory. Recommend (a); the value here is the *specification*, not the code.

### 1.2 The baseline identity model already has a training path

`sn_gamestate/configs/modules/reid/prtreid.yaml` carries `training_enabled: False`, and
`sn_gamestate/reid/prtreid_api.py:179` defines `def train(...)`. So retraining PRTreID on the GSR
train split is a config flip, not new code. Recorded config: `bpbreid` model, `hrnet32` backbone,
`gwap` pooling, `dim_reduce_output: 256`, loss `part_averaged_triplet_loss`, sampler `PrtreidSampler`
with `num_instances: 4`, `train.batch_size 32`, `train.max_epoch 20`, `data.transforms: ["rc", "re"]`
(random crop + random erasing), `data.height/width 256/128`, init from
`${model_dir}/reid/prtreid-soccernet-baseline.pth.tar`.

This is the cheapest possible S3 preliminary: a measured GSR-trained identity floor to beat, obtained
before a single line of CLIP code is written. It also sets the augmentation baseline (`rc` + `re`).

### 1.3 A committed test-split submission

`examples_predictions/SoccerNetGS-test.zip` - 69.3 MB, in-tree - is the baseline's own predictions on
the **test** split in submission format. Free submission-format validator and a floor to diff against,
without running the baseline for ~9 GPU-hours.

### 1.4 Open issues to read before the S0 cluster probe

- 2026-04-19 **"GPU inference returns dummy bounding boxes [0.0, 0.0, 1.0, 1.0] (CPU works fine)"**
- 2026-04-24 "calibration: use `.iloc[0]` so positional access works under pandas 2.x"
- 2025-11-07 "[Bug] Cannot run pipeline for inference/tracking without downloaded dataset splits (KeyError)"
- 2025-08-12 "RuntimeError: `_share_filename_`: only available on CPU; when running with mps"
- 2025-08-07 "Pytorch dependency conflicts"

The first one is a silent-failure class - a pipeline that "runs" and scores near zero. Worth an
explicit assertion in our S0 probe.

---

## 2. SoccerNet-v3 - **USE (as data), MINE-FOR-PARTS (dataloader)**

https://github.com/SoccerNet/SoccerNet-v3 | MIT | 76 stars | last commit 2022-05-20 | 0 open issues

**Contents: eight files.** `dataloader.py`, `visualize.py`, `statistics.py`, README, LICENSE, AUTHORS,
two PNGs. No model, no weights, no training, no benchmark. Dead as code since 2022.

**But the annotation schema is the point.** Per action frame *and its replay frames*, across
**400 games in 6 major leagues** (EPL included), ~60 GB of frames:

- `bboxes`: `(x_top, y_top, width, height, class_index, **jersey_number**)`
- `lines`: per-line polyline `(x1,y1,...,xn,yn)` + class index
- `links`: player-to-player correspondences across the live and replay views of the same action

Why this matters for need 1: SoccerNet-GSR is 200 clips of Swiss Super League 2019. The campaign brief
already flags domain transfer as a separate, later claim. SoccerNet-v3 is the same organisation's
**multi-league, jersey-labelled, broadcast-resolution** bbox corpus, MIT-licensed, and it is the source
the sn-reid crops were cut from. If the trained identity model is to generalise past Swiss 2019, this
is where the extra supervision lives.

Verdict: **USE** as a data source; the dataloader is a straight read of the JSON schema and is worth
copying (MIT) rather than reverse-engineering.

---

## 3. sn-teamspotting - **USE (need 3)**

https://github.com/SoccerNet/sn-teamspotting | GPL-3.0 | 19 stars | last commit 2025-08-26 | 2 open issues

The only actively-maintained challenge devkit besides sn-gamestate, and **the only code in the org that
attributes an action to an agent.**

- **Task:** 12 ball actions x which team (left/right) performed it, tolerance 1 second.
- **Ships full training code**, not a downloader: `train_tdeed_bas.py`, `model/model.py`,
  `model/modules.py`, `model/shift.py`, `model/impl/gsm.py`, `model/impl/gsf.py`,
  `dataset/frame.py` (21 KB), `extract_frames_sn.py`, `extract_frames_snb.py`, the metric itself in
  `util/eval.py` + `util/score.py`, and the split JSONs under `data/`.
- **Weights:** baseline checkpoint at
  https://drive.google.com/drive/folders/16IqSkctIGp76ZYKKvJvMB_ggHcQsessM?usp=sharing
- **Method:** T-DEED (arXiv 2404.05392, the 2024 Ball Action Spotting winner) plus **one extra sigmoid
  head and a BCE team loss**, joint-trained on SoccerNet Action Spotting (17 classes) and Ball Action
  Spotting (12 classes). Built on the E2E-Spot codebase.
- **Full recipe** (`config/SoccerNetBall/SoccerNetBall_baseline.json`): `feature_arch rny002_gsf`,
  `temporal_arch ed_sgp_mixer`, `n_layers 2`, `sgp_ks 9`, `sgp_r 4`, `clip_len 100`, `batch_size 4`,
  `learning_rate 8e-4`, `num_epochs 35`, `warm_up_epochs 3`, `start_val_epoch 15`,
  `epoch_num_frames 500000`, `mixup true`, `radi_displacement 4`, `crop_dim -1`, `event_team true`.
  Two-pass data prep: run once with `store_mode: "store"` to partition the untrimmed videos into clips,
  then `"load"` to train.
- **Scores:** Team mAP@1 **47.18** test / **51.72** challenge, against team-agnostic mAP@1 53.59 / 58.38.
  **Adding the team attribute costs ~6.4 mAP** - the same shape as GS-HOTA's attribute tax.
- **Evaluation servers:** test https://www.codabench.org/competitions/4418/ ,
  challenge https://www.codabench.org/competitions/4417/

Relevance to need 3: it stops at *team*, not *player*. But it is (a) the natural evaluation harness for
a FOOTPASS/PCBAS line, and (b) the exact architectural pattern we would extend - shared spotting
backbone + an additional attribute head + its own loss - swapping the binary team head for a
player-identity head fed by our GSR identity model. The 6.4-mAP team tax is the honest prior for what
a *player* head would cost.

---

## 4. sn-reid - **MINE-FOR-PARTS**

https://github.com/SoccerNet/sn-reid | MIT | 88 stars | last commit 2023-07-07 | 3 open issues

A fork of KaiyangZhou/deep-person-reid (torchreid). Full framework: `torchreid/models/` (osnet,
osnet_ain, resnet, resnet_ibn, pcb, hacnn, mlfn, densenet, ...), `torchreid/engine/` (the train loops:
`engine.py`, `image/softmax.py`, `image/triplet.py`), `torchreid/losses/`
(`cross_entropy_loss.py`, `hard_mine_triplet_loss.py`), `torchreid/data/` (datamanager, sampler,
transforms), `torchreid/metrics/` (incl. a Cython `rank_cylib`), `torchreid/utils/` (re-ranking +
a CUDA GPU re-ranking extension).

**Weights: there are no SoccerNet-trained weights here, and the org has not answered a request for
them.** Open issue 2026-05-13, still open: *"Request: Pre-trained weights for baseline ResNet50_fc512
(Inference only)"*. `docs/MODEL_ZOO.md` is torchreid's inherited zoo - Market-1501 / DukeMTMC / MSMT17
checkpoints on Google Drive (osnet_x1_0, osnet_ain_x1_0, resnet50_fc512, ...). Those are generic person
re-ID inits, useful as a starting point and nothing more.

Specifically reusable, all MIT:

- `torchreid/data/datasets/image/soccernetv3.py` - auto-download plus the **filename parser**:
  `<bbox_idx>-<action_idx>-<person_uid>-<frame_idx>-<clazz>-<id>-<UAI>-<WxH>.png`. Per the docstring,
  **`id` is the jersey number when the number is visible at least once in the action, and a letter when
  it is not.** That is free weak jersey supervision on 340,993 crops **and a built-in
  number-not-visible marker** - the abstention channel S4 needs, without annotating anything.
- `torchreid/data/sampler.py` - `RandomIdentitySampler` (P identities x K instances).
- `torchreid/data/transforms.py:233 build_transforms` - the augmentation menu: `random_flip`,
  `random_crop` (= `Random2DTranslation`: enlarge to 1.125x then crop back), `random_patch`,
  `color_jitter`, `random_erasing`, ImageNet normalisation. The MODEL_ZOO's own note is worth keeping:
  heavy augmentation **hurts cross-dataset generalisation** - relevant given we are training on Swiss
  2019 and want EPL transfer.
- `torchreid/engine/engine.py` - a plain, readable train/eval loop with LR warmup and per-epoch
  ranking evaluation.

The published baseline (`benchmarks/baseline/configs/baseline_config.yaml`): `resnet50_fc512`,
256x128, batch 32, `triplet` loss margin 0.3 (`weight_t 0.5`, `weight_x 0.5`), `RandomIdentitySampler`
`num_instances 4`, `max_epoch 10`, and **`training_subset: 0.01`** - the shipped default trains on 1%
of the training set. Anyone quoting "the sn-reid baseline" without changing that number is quoting a
1% model. Baseline leaderboard: mAP 59.11 / R-1 48.41.

README env (`python=3.7`, `cudatoolkit=9.0`) is ancient; only `python setup.py develop` and the
optional Cython rank extension actually constrain anything.

Caveat already on record (`GSR_METHODS_DEEP_DIVE.md` S4.2.3): SoccerNet re-ID is same-action
cross-viewpoint matching, so the 93.26 mAP headline does not transfer to long-horizon naming. Nothing
in the code changes that.

---

## 5. sn-banner - **MINE-FOR-PARTS** *(not on Sid's list; the surprise of this sweep)*

https://github.com/SoccerNet/sn-banner | GPL-3.0 | 3 stars | last commit 2025-12-26 | 0 open issues | 124 MB

Nominally an advertising-banner replacement pipeline. In practice it is a **weights-and-calibration
repo** that nobody has looked at.

**Weights, with direct URLs** (from the README table):

| Model | URL |
|---|---|
| NBJW keypoints (SV_kp) | https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_kp |
| NBJW lines (SV_lines) | https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_lines |
| Mask2Former fine-tuned on SoccerNet | https://huggingface.co/datasets/SoccerNet/BannerReplacement/resolve/main/best_mIoU_iter_10935.pth |

The first two are **GitHub release assets, not Google Drive** - they `curl` cleanly on a headless
cluster node. Every other calibration weight link in our notes is a Drive URL that needs a browser or a
cookie dance. Worth recording for the S0/S2 cluster runs regardless of whether we use NBJW itself
(we run PnLCalib; NBJW is its predecessor keypoint stage).

**`camera_calibration/filters.py`** - temporal camera-parameter filtering, applied as stacked layers by
`camera_calibration/evaluation/cam_params_filtering.py`:

- `to_valid_cam_params` - validity mask + erroneous-parameter positions
- `camParamsPerImage_to_camParamsPerType` / inverse - reshape to per-parameter time series
- layer 1 `linear_interpolation` over the erroneous positions
- layer 2 `outliers_remover`
- layer 3 `camParamsSmoothing(windowLength=23)`
- also `smoothing_using_banner_corners` (image-anchored smoothing, not applicable to us)
- `find_window_filter_length.sh` - a window-length sweep harness

This is the only public stacked cam-param filter *with a length sweep* in the org, and it is the same
family as our `generator/temporal_calib.py`. Compare against Constructor.Tech's Savitzky-Golay with a
15-frame delay and +/-2 deg / +/-2 m clamps (`GSR_METHODS_DEEP_DIVE.md` S1.1.1b).

**Semantic segmentation:** `semantic_segmentation/models/challenge_mask2former/` carries mmseg configs
including `mask2former_r50_1xb2-90k_soccernet-1080x1920.py` - a **SoccerNet-native 1080x1920 mmseg
training config** - plus benchmark confusion matrices (`.npy`) for mask2former / segformer / segmenter
/ knet / pointrend / ddrnet with and without TTA. If we ever want the HTWK Leipzig trick (GSR-13:
IoU between a pitch segmentation and the homography-reprojected pitch as a **ground-truth-free
calibration confidence**, worth up to 4% - `GSR_METHODS_DEEP_DIVE.md` S2.7), this is a segmentation
starting point, though the classes are banner/scene classes and would need remapping.

Dataset (public subset): https://huggingface.co/datasets/SoccerNet/BannerReplacement

GPL-3.0 - same vendoring caveat as sn-gamestate.

---

## 6. sn-tracking - **MINE-FOR-PARTS** (and one correction)

https://github.com/SoccerNet/sn-tracking | **no LICENSE file** | 107 stars | last commit 2023-07-07 | 11 open issues

Worst issue health in the sweep: 11 open, spanning 2023-05 to 2026-01, including *"Unable to reproduce
ByteTrack Results"* (2023-07-28) and *"SoccerNet-Tracking Challenge 2023 Submission Evaluation Issue"*
(2024-06-07). Unmaintained.

Contents are thin wrappers, not a framework:
`Benchmarks/ByteTrack/{demo_track.py, demo_track_no_gt.py, run_*_batch.sh, yolox_x_soccernet.py,
yolox_x_soccernet_no_gt.py}`; `Benchmarks/DeepSORT/` = PaddleDetection YAML configs plus three
`gen_*_SN.py` label converters; `Benchmarks/FairMOT/README.md` is a **109-byte stub** - an empty
benchmark. Plus `tools/evaluate_soccernet_v3_tracking.py`, `tools/zip_gt.py`, and the
`SNMOT-test.txt` / `SNMOT-challenge.txt` sequence lists.

**Correction worth recording.** `yolox_x_soccernet.py` reads like a SoccerNet detector recipe. It is
not. `get_data_loader` points `MOTDataset` at `mix_mot20_ch` (mixed MOT20 + CrowdHuman); only
`get_eval_loader` points at `SN_tracking`. **It trains on MOT20+CrowdHuman and merely evaluates on
SoccerNet.** That is a plausible root cause for the open reproduction issue, and it means the repo
contains no SoccerNet detector fine-tune at all.

The hyperparameters are still a usable YOLOX-X starting point: `num_classes 1`, `depth 1.33`,
`width 1.25`, input `1080x1920` when running with GT detections / `896x1600` without,
`random_size (20,36)`, `max_epoch 80`, `no_aug_epochs 10`, `warmup_epochs 1`,
`basic_lr_per_img 0.001/64`, `nmsthre 0.7`, `test_conf 0.001`, Mosaic + Mixup via `MosaicDetection`.
Consistent with the Deep-EIoU recipe already recorded (`GSR_METHODS_DEEP_DIVE.md` S4.4.1).

**Licence:** no LICENSE at repo root (default: all rights reserved). Sub-benchmarks carry their own -
`Benchmarks/ByteTrack/LICENSE` is MIT (Baidu USA LLC), `Benchmarks/DeepSORT/LICENSE` is Apache-2.0
(PaddlePaddle). Do not vendor from the root tree.

---

## 7. ActiveSpotting - **MINE-FOR-PARTS**

https://github.com/SoccerNet/ActiveSpotting | MIT | 3 stars | last commit 2023-05-22 | 0 open issues

Code for *"Towards Active Learning for Action Spotting in Association Football Videos"*
(arXiv 2304.04220, CVPRW 2023). Seven source files: `src/main.py` (24.7 KB, the AL loop), `train.py`,
`dataset.py`, `model.py`, `netvlad.py`, `loss.py`. Runs on pre-extracted ResNet-PCA512 features with a
NetVLAD++ head. **No weights.** The "Experiments with PTS" section of the README is literally `TBD` -
that half was never written.

The transferable content is the loop, not the task. CLI surface:
`--sampling_method {random, confidence_0.5, entropy}`, `--training_scheme faster`,
`--active_scheme increasing`, `--continue_training`, `--max_epochs`. The paper's finding is that
**fine-tuning from the previous AL iteration reaches target performance in far fewer epochs than
retraining from scratch**, and that confidence/entropy sampling beats random at a fixed label budget.

Why it is worth keeping: annotation budget is a live constraint. UPCxMobius reached 2nd place in GSR
2024 on roughly 3,000 hand-annotated frames total (`GSR_METHODS_DEEP_DIVE.md` S1.2), and Playbox's 2024
jersey work used 3,000 boxes. If S4's jersey head ever needs hand labels, this is the sampler schedule
- and an entropy/confidence sampler composes directly with the Dirichlet evidential uncertainty S4
already plans to output.

---

## 8. sn-spotting - **MINE-FOR-PARTS**

https://github.com/SoccerNet/sn-spotting | root LICENSE = MIT, benchmarks Apache-2.0 | 102 stars |
last commit 2024-02-07 | 3 open issues (newest 2024-01, all stale)

*(GitHub's licence detector reports "none" for this repo; the root `LICENSE` file is plain MIT,
copyright 2021 SoccerNet. Each `Benchmarks/*/LICENSE` is Apache-2.0.)*

258 MB, because **it commits trained weights into git**. In-tree checkpoints:
NetVLAD++ (5 seeds + a challenge model, 20.4 MB each), NetVLAD v1/v2, MaxPool v1/v2,
CALF (6.9 MB), CALF_Calibration VIS (6.9 MB) and NVIS (15.1 MB), CALF_Calibration_GCN (10 seeds x
9 MB), plus per-run training logs and zipped `results_spotting_{test,challenge}` predictions.

The task is frame/feature-level event spotting over 500 games. **No players, no boxes, no identity.**
Zero transfer to needs 1-2.

One genuinely reusable folder, `Features/`:
`ExtractResNET_TF2.py`, `VideoFeatureExtractor.py`, `ReduceFeaturesPCA.py`, `ConvertHQtoLQ.py`, plus the
fitted `pca_512_TF2.pkl` and `average_512_TF2.pkl`. That is a turnkey ResNet-152 + PCA-512 feature
extractor for arbitrary broadcast video, matching the `*_ResNET_TF2_PCA512.npy` features every
spotting model in the org (and ActiveSpotting) consumes. If need 3 ever wants a cheap global temporal
context feature alongside player tracks, this is it. `Annotation/` is a PyQt event-annotation tool.

For need 3 the benchmarks themselves (CALF, NetVLAD++) are superseded by E2E-Spot and T-DEED -
the README itself points at https://github.com/jhong93/spot as the Ball Action Spotting benchmark to
beat.

---

## 9. sn-calibration - **IGNORE**

https://github.com/SoccerNet/sn-calibration | **no LICENSE file** | 106 stars | last commit 2024-06-18
(README edit; last code change is older) | 3 open issues, oldest 2022-03 | branches: `main`,
`calibration-2023`

16 source files, no training code of any kind. `src/soccerpitch.py` (31 KB) is the canonical 3D pitch
model with the line and point class taxonomy; `src/camera.py` is the camera model;
`src/detect_extremities.py` runs inference with a DeepLabV3 segmentation network;
`src/evaluate_camera.py` / `evaluate_extremities.py` implement the JaC@5/10/20, completeness-rate and
final-score metrics; `evalai_camera.py` packages submissions. The `calibration-2023` branch is the same
16 files with a shorter README.

**Weights:** one Google Drive link, the 2022 segmentation baseline -
https://drive.google.com/file/d/1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM/view
Baseline results in the README: 11.7% / 68% / 7.96%. PnLCalib, which we already run, scores JaC@5 80.6
on the same benchmark (`GSR_METHODS_DEEP_DIVE.md` S4.5.1).

**And sn-gamestate already vendors this entire `src/` tree** at
`plugins/calibration/sn_calibration_baseline/`. There is nothing here we do not already have on disk.

Only conceivable use: `evaluate_camera.py` if we ever want per-frame JaC diagnostics decoupled from
GS-HOTA. No licence file, so reimplement rather than copy.

---

## 10. PTS-baseline - **IGNORE**

https://github.com/SoccerNet/PTS-baseline | BSD-3-Clause | 0 stars | **fork = true** | last commit
2023-02-15 | 0 open issues

**This is not SoccerNet's own work.** It is a fork of https://github.com/jhong93/spot - *"Spotting
Temporally Precise, Fine-Grained Events in Video"* (E2E-Spot, ECCV 2022, Hong et al.). The fork's only
additions are SoccerNet ingestion: `data/soccernetv2/`, `data/soccernet_ball/`, `parse_soccernet.py`,
`parse_soccernet_ball.py`, `frames_as_jpg_soccernet.py`, `frames_as_jpg_soccernet_ball.py`,
`eval_soccernetv2.py`, `eval_soccernet_ball.py`.

**Weights are not here** - upstream, at https://github.com/jhong93/e2e-spot-models/ (checkpoint +
`config.json` per model). TSP and 2D-VPD baseline features:
https://drive.google.com/drive/folders/1AQFd8JsvxdEG2jQfY5GDVSLEtc9r824W

Superseded on every axis by sn-teamspotting: same E2E-Spot codebase lineage, newer model (T-DEED),
maintained in 2025 rather than 2023, better scores, and it already has the agent-attribution head we
care about. The one thing it has that sn-teamspotting lacks - a SoccerNet-v2 (17-class) path - is
duplicated inside sn-teamspotting's joint-training config anyway.

---

## 11. sn-jersey - **IGNORE**

https://github.com/SoccerNet/sn-jersey | **no LICENSE file** | 30 stars | last commit 2024-07-02 |
1 open issue (2023-10, "Can't download the dataset")

**The repository contains exactly one file: `README.md`.** Repo size 18 KB. No dataloader, no baseline,
no evaluation script, no weights.

The README carries the 2023 leaderboard (ZZPM 92.85, UniBw Munich VIS 90.95, zzzzz 88.08, Mike Azatov
82.05, MT-IOT 81.70, justplay 77.77, AIBrain 75.18, SARG UWaterloo 73.77, Kalisteo 58.35, ... random
baseline 3.93), the task definition, the data format (one folder of thumbnails per player; ground truth
a JSON dict `player_id (str) -> jersey_number (int)`, **-1 = no number visible**), the download call
`mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])`, and an EvalAI link.
All of that is already in `GSR_METHODS_DEEP_DIVE.md` S4.3.6.

The *dataset* remains the right training corpus for the S4 jersey head, and -1-as-a-scored-class remains
the right abstention framing. The *repo* contributes nothing further.

---

## 12. sn-grounding - **IGNORE**

https://github.com/SoccerNet/sn-grounding | root MIT, `Benchmarks/LICENSE` Apache-2.0 | 9 stars |
last commit 2022-11-10 | 1 open issue from 2022-06

Replay grounding: given a replay clip, locate the live-action timestamp. Operates on pre-extracted
ResNet/Baidu features - no pixels, no boxes, no players. Ships three benchmarks
(`SoccerNetv2-ReplayGrounding-CALF`, `-CALF_more_negative`, `-NetVLAD-More-Negative`), each with a
committed `model.pth.tar` (1.6-13.1 MB), plus the PyQt annotation tool. Challenge deadline in the
README is 30 May 2022.

The one conceptually adjacent idea - matching two views of one action - is the same same-action
cross-view framing as sn-reid, which we have already closed as not transferring to long-horizon
identity. Nothing for needs 1-3.

---

## 13. Org scan - repos NOT on Sid's list

Full org listing (19 repos) taken from the API. Eight were outside the brief; here is what they are and
whether they matter.

| Repo | Push | Licence | What | Flag |
|---|---|---|---|---|
| [sn-banner](https://github.com/SoccerNet/sn-banner) | 2025-12-26 | GPL-3.0 | Banner replacement; **NBJW + Mask2Former weights, temporal cam-param filters** | **FLAGGED - see S5** |
| [sn-depth](https://github.com/SoccerNet/sn-depth) | 2025-03-27 | none | SoccerNet-Depth, 12.4K synthetic frames from NBA2K22/eFootball; ZoeDepth baseline **with weights**; extraction code. Codabench test 4876 / challenge 6864 | **WATCH** - Broadcast2Pitch++ is "Depth-Aware GSR"; if depth becomes a GSR lever, the SoccerNet depth model is here |
| [sn-trackeval](https://github.com/SoccerNet/sn-trackeval) | 2025-07-29 | MIT | GS-HOTA implementation | Already running locally |
| [SoccerNet](https://github.com/SoccerNet/SoccerNet) | 2025-07-16 | MIT | The pip downloader | Already in use |
| [sn-nvs](https://github.com/SoccerNet/sn-nvs) | 2025-12-02 | **none** | Novel view synthesis, 5 Blender scenes, COLMAP poses. **Three files**: README, `render_challenge.py`, `.gitmodules` | IGNORE |
| [sn-echoes](https://github.com/SoccerNet/sn-echoes) | 2025-05-29 | **none** | Whisper ASR transcripts of commentary (v1/v2, en) | IGNORE - commentary-as-identity closed at chance level (13.2% vs 12.5%) |
| [sn-mvfoul](https://github.com/SoccerNet/sn-mvfoul) | 2025-01-09 | GPL-3.0 | Multi-view foul classification (VARS) | IGNORE |
| [sn-caption](https://github.com/SoccerNet/sn-caption) | 2024-04-12 | **none** | Dense video captioning | IGNORE |

Nothing else in the org. No hidden GSR repo, no winner code (consistent with the campaign brief's "no
winner released code"). `sn-calibration` is the only repo with a second branch (`calibration-2023`),
and it is functionally identical to `main`.

---

## 14. Licence map (and the problems)

| Licence | Repos | Consequence |
|---|---|---|
| **No LICENSE file** | sn-calibration, sn-tracking, sn-jersey, sn-nvs, sn-depth, sn-echoes, sn-caption | Default is all-rights-reserved. **Do not vendor code from these.** Reimplement from the spec |
| **GPL-3.0** (copyleft) | sn-gamestate, sn-teamspotting, sn-banner, sn-mvfoul | Vendoring makes our repo GPL-3.0 |
| MIT | sn-reid, SoccerNet-v3, ActiveSpotting, sn-spotting (root), sn-grounding (root), sn-trackeval, SoccerNet pkg | Clean |
| Apache-2.0 | sn-spotting/Benchmarks/*, sn-grounding/Benchmarks/*, sn-tracking/Benchmarks/DeepSORT | Clean |
| BSD-3-Clause | PTS-baseline | Clean |

**The live problem:** our single best find (S1.1, `prtreid_dataset.py`) is GPL-3.0. Recommendation is to
treat it as a *specification* - the filtering law and the label-column contract are fully documented in
S1.1 above and are a few lines of pandas + `cv2` to reimplement - rather than to copy it. This is
separate from, and does not resolve, the open GSR **dataset** licence question (GPL-3.0 vs CC BY-4.0)
flagged in the campaign brief S7.

Two detector notes: GitHub's licence API reports `none` for sn-spotting and sn-grounding even though
both carry a plain MIT `LICENSE` at root (verified by fetching the file). Trust the file.

---

## 15. The five most valuable finds for the cluster training (need 1)

1. **`sn-gamestate/sn_gamestate/reid/prtreid_dataset.py` + `configs/modules/reid/dataset/prtreid_dataset.yaml`**
   - a complete SoccerNet-GSR to labelled-crop pipeline emitting **role / team / jersey_number** per
   crop, with a documented filtering law (`min_vis 0.3`, `min_h/min_w 30`, uniform tracklet sampling to
   15, `min_samples_per_id 4`, seeded id cap) and a resumable on-disk crop cache. This is the S3 data
   prep, already specified. **GPL-3.0 - reimplement from the spec, do not copy.** (S1.1)

2. **`sn-gamestate` PRTreID training path** - `training_enabled: False` in
   `configs/modules/reid/prtreid.yaml` plus `train()` at `sn_gamestate/reid/prtreid_api.py:179`.
   One flag retrains the baseline identity model on GSR train (hrnet32/bpbreid, 20 epochs, batch 32,
   part-averaged triplet, `rc`+`re` augmentation). Cheapest possible measured floor to beat before any
   CLIP code is written. (S1.2)

3. **`sn-reid` torchreid internals, MIT** - `engine/engine.py` train loop, `data/sampler.py`
   `RandomIdentitySampler(num_instances=4)`, `data/transforms.py:233` augmentation menu, and
   `data/datasets/image/soccernetv3.py`, whose filename parser yields **free weak jersey labels plus an
   explicit number-not-visible marker** across 340,993 crops. Also the zoo's own warning that heavy
   augmentation harms cross-dataset generalisation - directly relevant to Swiss-2019-to-EPL transfer.
   (S4)

4. **SoccerNet-v3, MIT** - 400 games across 6 leagues (EPL included) of jersey-labelled bounding boxes
   at broadcast resolution, with cross-view player correspondences. The multi-league identity training
   corpus that SoccerNet-GSR's 200 Swiss clips are not. Dataloader is MIT and copyable. (S2)

5. **`sn-banner`** - NBJW `SV_kp` / `SV_lines` as **GitHub release assets** (curl-able on a headless
   cluster node, unlike every Drive link in our notes), a SoccerNet-fine-tuned Mask2Former on
   HuggingFace with a native 1080x1920 mmseg config, and `camera_calibration/filters.py`: stacked
   temporal cam-param filtering (interpolate -> outlier-remove -> smooth, window 23) with a
   window-length sweep script - the same family as our `generator/temporal_calib.py`. (S5)

---

## 16. Surprises

- **`sn-banner` exists and nobody had looked at it.** It is the only repo in the org carrying
  calibration weights behind stable, browser-free download URLs, plus a SoccerNet Mask2Former and a
  temporal camera-parameter filter with a tuning sweep. It is filed under "advertising".
- **`sn-reid` publishes no SoccerNet-trained weights**, and the request for them (2026-05-13) is open
  and unanswered. Its shipped baseline config trains on **1%** of the training set by default.
- **`sn-tracking`'s "SoccerNet" YOLOX recipe trains on MOT20 + CrowdHuman** and only evaluates on
  SoccerNet - which likely explains the open "Unable to reproduce ByteTrack Results" issue. There is no
  SoccerNet detector fine-tune anywhere in the org.
- **`sn-gamestate` commits the baseline's own test-split submission** (`examples_predictions/
  SoccerNetGS-test.zip`, 69 MB) - a free submission-format validator and score floor.
- **`sn-jersey` is one README file.** 30 stars for a repo with no code.
- **`PTS-baseline` is a fork**, not SoccerNet's own work, and is superseded by `sn-teamspotting`.
- **`sn-teamspotting` measured the attribute tax independently**: adding just the *team* label to ball
  action spotting costs ~6.4 mAP@1 (53.59 -> 47.18 test). Same phenomenon as GS-HOTA's identity gate,
  in a different task - useful corroboration that agent attribution, not event detection, is the hard
  part.
- **`sn-spotting` stores ~200 MB of trained checkpoints in git**, including 10 seeds of one model.
