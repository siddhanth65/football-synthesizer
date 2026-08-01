# SoccerNet ecosystem survey - RAW findings

Surveyed 2026-07-28 by the research-search worker. Scope: anything bearing on PLAYER IDENTITY on
broadcast footage (tracking, re-ID, jersey OCR, game state reconstruction, commentary/ASR weak
labels). Raw collection only - no recommendations, no plan.

Our reference bottlenecks (used only to tag relevance, not to argue for anything):
naming factor 0.609 | appearance ceiling ~0.62 LOTO top-1 | silent OCR failure | GK-never-named
defect | GS-HOTA 24.18 vs SOTA 63.81.

Our current stack for comparison: YOLO detect, ByteTrack, PnLCalib homography, PRTreID embeddings,
Koshkina jersey OCR, GTA-Link tracklet connector.

---

## 0. Headline calendar correction

**GSR is finished as a SoccerNet challenge.** Game State Reconstruction ran 2024 and 2025 only. The
**SoccerNet 2026** edition (arXiv `2607.07320`) does NOT include GSR, tracking, re-ID, or jersey
number recognition as standalone tasks. The 2026 challenges are: Ball Action Anticipation,
Player-Centric Ball Action Spotting, Novel View Synthesis, Spiideo SoccerNet Synloc, Visual Question
Answering.

- SoccerNet 2023 results: https://arxiv.org/abs/2309.06006 (7 tasks incl. re-ID, tracking, jersey)
- SoccerNet 2024 results: https://arxiv.org/abs/2409.10587 (GSR introduced; NO jersey task)
- SoccerNet 2025 results: https://arxiv.org/abs/2508.19182 (GSR final edition)
- SoccerNet 2026 results: https://arxiv.org/abs/2607.07320 (GSR absent; PCBAS introduced)
- Challenge index: https://www.soccer-net.org/challenges

---

# (a) GSR challenge 2024 + 2025

## 1. SoccerNet-GSR benchmark + GS-HOTA metric (the task definition itself)

- **What:** The dataset/metric/baseline paper. 200 x 30-second broadcast clips, 9.37M line points for
  pitch localization, >2.36M athlete pitch positions. Introduces GS-HOTA.
- **Challenge/year:** CVPRW'24 CVsports. Baseline, not a competitor.
- **Paper:** https://arxiv.org/abs/2404.11335
- **PDF:** https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf
- **Code:** https://github.com/SoccerNet/sn-gamestate
- **Task:** localize + identify all athletes from a single moving camera onto a 2D top-view minimap.
  Must output: 2D pitch position, role (player/goalkeeper/referee/other), jersey number, team
  (left/right).
- **Metric detail (load-bearing for us):** `SimGS-HOTA(P,G) = LocSim(P,G) x IdSim(P,G)`. LocSim is an
  exponential distance decay on pitch coordinates. **IdSim is set to 1 only if ALL attributes match
  (role AND team AND jersey).** Any single attribute miss converts the detection to a False
  Positive. Detections are 2D points, not boxes.
- **Reported baseline:** GSR-Baseline = **22.26 GS-HOTA** on test set (2024 paper). Paper includes an
  ablation that disables each identity attribute in turn (Role / Team / Jersey), with IdSim=1 when
  all are disabled.
- **Relevance to us:** this is the exact metric definition behind our 24.18. The all-attributes-must-
  match rule means the GK-never-named defect and silent OCR failure each convert detections into FPs
  outright - GS-HOTA is not a graceful-degradation metric. The published per-attribute ablation is a
  ready-made decomposition of where score is lost.
- **Practicality:** GPL-3.0. Pretrained weights auto-download on first run.

## 2. GSR 2025 leaderboard (arXiv 2508.19182, Section 5)

Full table as published (GS-HOTA / GS-DetA / GS-AssA):

| Rank | Team | GS-HOTA | GS-DetA | GS-AssA |
|---|---|---|---|---|
| 1 | KIST-GSR | 63.90 | 51.36 | 79.55 |
| 2 | Constructor.Tech | 63.81 | 49.52 | 82.23 |
| 3 | lianyou | 62.76 | 46.17 | 85.33 |
| 4 | Playbox & MIXI | 61.64 | 45.04 | 84.36 |
| 5 | tyler_durden | 57.08 | 41.52 | 78.48 |
| 6 | SJTU Multi-Modal | 50.06 | 32.98 | 75.99 |
| 7 | eidos.ai | 46.24 | 31.06 | 68.85 |
| 8 | UTokyo Football Lab | 40.13 | 21.33 | 75.51 |
| 9 | hjkim | 36.90 | 17.56 | 77.58 |
| 10 | junkwang | 33.62 | 18.09 | 62.51 |
| 11 | lsmuqi | 32.62 | 17.03 | 62.50 |
| 12 | hayden97 | 32.59 | 16.96 | 62.61 |
| - | **Baseline (sn-gamestate)** | **29.01** | **13.71** | **61.41** |
| 13 | Lab for Biosignal | 28.45 | 13.50 | 59.99 |
| 14 | johnstreet | 14.26 | 9.67 | 21.10 |

Notes:
- The 2025 baseline (29.01) is higher than the 2024 paper baseline (22.26) - the repo baseline
  improved between editions.
- **Our 24.18 sits between johnstreet (14.26) and Lab for Biosignal (28.45), i.e. below the shipped
  baseline.**
- Across the board **GS-AssA is high (60-85) while GS-DetA is low (14-51)**. Every team, including
  the winner, loses far more on the detection/identity term than on association. The published
  spread says identity attributes, not tracklet continuity, are the dominant loss.
- **No GitHub URLs appear anywhere in the GSR section of the 2025 report**, and no code-availability
  statements. Confirmed by two independent passes over the HTML.
- Paper text explicitly states: "Our method had difficulties mostly with jersey number recognition",
  with jersey number recognition named as the bottleneck.

## 3. KIST-GSR - GSR 2025 WINNER (63.90)

- **What:** Korea Institute of Science and Technology. YOLO-X detection; Deep-EIoU + OSNet ReID
  embeddings for tracking; multi-frame keypoint detection model for pitch mapping.
- **Identity method (the distinctive part):** formulates role, jersey and team as **multimodal
  autoregressive generation with LLAMA-3.2-Vision**, driven by instruction-based prompts. Not an OCR
  head - a VLM asked to read the player.
- **Tracklet refinement:** split-and-merge strategy guided by identity consistency + ReID similarity.
- **Team assignment:** clusters jersey colors, then compares cluster-wise mean x-positions to decide
  left/right.
- **Post-processing:** majority voting within tracklets.
- **Paper:** no standalone paper found; method summary exists only inside arXiv `2508.19182` Sec 5.4.
- **Code:** **none found.** No repo referenced in the challenge report, no separate arXiv entry.
- **Relevance:** VLM-as-jersey-reader is an alternative to OCR that returns a refusal/uncertainty
  channel naturally (silent OCR failure); split-and-merge is the same family as our GTA-Link.
- **Practicality:** LLAMA-3.2-Vision 11B is the smallest vision variant - **not plausible on 4 GB
  VRAM** at fp16 (~22 GB); heavy quantization would be required and the weights are gated on HF
  (Llama 3.2 Community License, with an EU-territory restriction on the vision models). Inference-
  only in principle. No released code to reproduce the prompting.

## 4. Constructor.Tech - GSR 2025 runner-up (63.81) AND GSR 2024 WINNER

- **What:** Same team across both editions. The 2024 run is the paper the orchestrator named.
- **2024 result:** GS-HOTA **63.81**, vs 2nd place 43.15 and 3rd place 34.40 - a 20-point margin.
  The identical 63.81 in 2025 strongly suggests a resubmission of the 2024 system.
- **Paper (2024 winner):** "From Broadcast to Minimap: Achieving State-of-the-Art SoccerNet Game
  State Reconstruction", https://arxiv.org/abs/2504.06357 (CVPR 2025 CVsports Workshop).
  HTML: https://ar5iv.labs.arxiv.org/html/2504.06357
- **Authors:** Vladimir Golovkin, Nikolay Nemtsev, Vasyl Shandyba, Oleg Udin, Nikita Kasatkin, Pavel
  Kononov, Anton Afanasiev, Sergey Ulasen, Andrei Boiarov.
- **2024 components:** fine-tuned **YOLOv5m** detection; **SegFormer**-based camera parameter
  estimator; **DeepSORT**-based tracking with re-identification, **orientation prediction**, and
  jersey number recognition.
- **2025 components (per challenge report):** **RF-DETR** detection; **BoT-SORT + Global Tracklet
  Association**; optical flow for temporal calibration.
- **CODE: NOT FOUND.** Searched the arXiv abstract page, the ar5iv HTML, and the 2025 challenge
  report. No GitHub URL, no project page, no code-availability statement in any of them. This is the
  single most-cited GSR result and **it is unreproducible from published artifacts.** Flagging this
  as the clearest availability-contradicts-expectations finding.
- **Relevance:** orientation prediction as an explicit auxiliary signal is notable - it gates when a
  jersey number is even readable, which is the structural fix for silent OCR failure.

## 5. lianyou - GSR 2025 third (62.76)

- **What:** YOLOv12 detection; custom RT-DETR for field lines.
- **Calibration:** radial distortion correction combined with **PnLCalib** (our calibrator),
  simplified to a 4-DOF camera parameterisation `[pan, tilt, roll, focal]`.
- **Tracking:** enhanced IOF-Tracker using Farneback optical flow.
- **Jersey:** **fine-tuned ViT-L/14 CLIP model**, applied after aggregating tracklet features via a
  lightweight 6-layer Transformer. So: tracklet-level feature aggregation FIRST, then a single
  classification - not per-frame OCR then vote.
- **Highest GS-AssA in the top 5 (85.33).**
- **Code:** none found.
- **Relevance:** directly relevant that a top-3 team uses PnLCalib, i.e. our calibrator is not the
  thing separating us from 63. The CLIP-classifier-over-aggregated-tracklet-features design is a
  different shape to Koshkina's per-crop-OCR-then-vote and produces a calibrated confidence.
- **Practicality:** ViT-L/14 CLIP inference is ~1.7 GB fp16 - plausible on 4 GB. Fine-tuning is not.

## 6. Playbox & MIXI - GSR 2025 fourth (61.64)

- **What:** The most reproducible-looking top-5 entry, because it is built on public parts.
- **Framework:** **TrackLab** (the same framework sn-gamestate uses).
- **Detection:** RF-DETR. **Re-ID:** CLIP-ReID embeddings. **Tracking:** BoT-SORT + **Global Tracklet
  Association (GTA)** - the same GTA-Link family we just added.
- **Jersey:** **PARSeq OCR applied to torso regions extracted via ViTPose keypoints.** This is the
  Koshkina recipe (PARSeq + pose-guided crop) reassembled.
- **Calibration:** extends **BroadTrack** with optical-flow temporalisation.
- **Role:** heuristic goalkeeper assignment based on spatial positioning. **Directly relevant to the
  GK-never-named defect** - the 4th place team does not learn GK, they assign it geometrically.
- **Code:** none found for the submission itself, but every named component is public.
- **Relevance:** this entry is essentially our stack + GTA + a GK heuristic + BroadTrack, scoring
  61.64. It is the closest published upper bound on a stack we could actually assemble.

## 7. eidos.ai - GSR 2025 seventh (46.24) - VLM with zoom

- **What:** **Qwen2-VL-Instruct** combined with **function calling**. Processes detections in
  sequences using chain-of-thought reasoning, and employs a **zoom function** to inspect jerseys.
- **Relevance:** the zoom-as-a-tool pattern is the explicit acknowledgement that jersey crops at
  broadcast resolution are too small to read in one pass. Beat several conventional pipelines with no
  jersey OCR model at all.
- **Practicality:** Qwen2-VL-2B-Instruct exists and is Apache-2.0 - the only top-10 VLM approach with
  a genuinely small, openly-licensed variant. 2B at 4-bit is plausible on 4 GB.
- **Code:** none found for the submission.

## 8. UTokyo Football Lab - GSR 2025 eighth (40.13) - uses GTA-Link

- **What:** **DeblurGAN-v2** preprocessing; YOLOv8x detection; DeepLabV3 + MobileNetV3-Large
  segmentation; 10-dimensional color-encoding vectors; **GTA-Link for tracklet refinement**;
  **Gaussian blur applied before jersey recognition.**
- **Relevance:** the only published entry that both uses GTA-Link (as we do) and reports its score,
  giving a rough anchor for what GTA-Link alone buys in this pipeline shape. The
  deblur-then-reblur sequence around jersey recognition is a concrete preprocessing detail.
- **Code:** none found.

## 9. Lower-half 2025 entries (methods worth recording)

- **hjkim (36.90):** motion deblurring; optical flow for camera compensation; fine-tuned detector;
  **constraint enforcement of a maximum of 24 individuals**. A hard cardinality prior.
- **Laboratory for Biosignal Processing (28.45):** homography estimation with tracking-based
  refinement; **U-Net pitch segmentation compared via IoU as a confidence measure**; Lucas-Kanade
  optical flow tracking; RANSAC homography updates. The IoU-confidence idea is a calibration
  self-check.
- **Constructor.Tech, tyler_durden, SJTU Multi-Modal:** confirmed to have submitted **no method
  summary** in the report body or appendix.

## 10. Broadcast2Pitch - WACV 2026 (independent of the challenge)

- **What:** Modular GSR framework: multi-task keypoint AND line detection model + optimization-based
  homography estimation, using dense geometric cues from lines, circles and keypoints frame-by-frame.
  For identity: appearance-based re-ID plus a **vision-language-guided tracklet refinement strategy**
  to cut ID switches.
- **Authors:** Yin May Oo, Yewon Hwang, Muhammad Robbani, Vanyi Chao, Ankhzaya Jamsrandorj, Hoang
  Nguyen, Kyung-Ryoul Mun, Jinwook Kim (KIST-affiliated - same lab family as the 2025 GSR winner).
- **PDF:** https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
  (returns HTTP 403 to automated fetch; poster page https://wacv.thecvf.com/virtual/2026/poster/435 ,
  IEEE https://ieeexplore.ieee.org/document/11492373/ )
- **Follow-up exists:** "Broadcast2Pitch++: Depth-Aware Game State Reconstruction from Unconstrained
  Soccer Videos" (ResearchGate 405759501).
- **Code:** none found.
- **Relevance:** "unconstrained" here means replays, shot changes and close-ups - the failure regime
  our carrier attribution hits. No arXiv mirror located.

---

# (b) SoccerNet org repos - state of each

Org: https://github.com/orgs/SoccerNet/repositories (19 repos)

## 11. sn-gamestate - the GSR baseline

- **URL:** https://github.com/SoccerNet/sn-gamestate
- **State: ACTIVE.** 429 stars. Last updated **2026-05-02** (TrackLab 1.3.24 integration fix for
  broken GS-HOTA evaluation, plus visualization tooling). Dataset version 1.3.
- **License: GPL-3.0.** (Copyleft - matters if any of this is vendored.)
- **Baseline modules shipped:**
  - Detection: **YOLOv11**
  - Re-ID: **PRTReID** and **BPBreID** (both selectable)
  - Tracking: **StrongSORT**
  - Jersey number: **MMOCR**
  - Calibration / field localization: **TVCalib**, **PnLCalib**, or **NBJW_Calib** (all three)
- **Weights:** dataset and model weights **auto-download on first run**, or manually via the
  SoccerNet pip package. Pre-computed tracker states for val/test/challenge are on **Zenodo**.
- **Install:** UV or Conda.
- **Relevance:** PnLCalib and PRTReID - two of our components - are literally the shipped baseline
  options here. The baseline's jersey module is MMOCR, which is weaker than our Koshkina OCR. The
  pre-computed tracker states on Zenodo are a way to isolate identity performance from detection and
  tracking without running the GPU stages.
- **Practicality:** modular Hydra config means individual stages can be swapped/disabled - relevant
  under a 4 GB, one-heavy-job-at-a-time constraint.
- **Note:** the May 2026 commit says GS-HOTA evaluation was **broken** prior to TrackLab 1.3.24.
  Any GS-HOTA number computed against an older sn-gamestate/TrackLab pairing is suspect.

## 12. TrackLab - the framework underneath

- **URL:** https://github.com/TrackingLaboratory/tracklab
- **Docs:** https://trackinglaboratory.github.io/tracklab/
- **License: MIT** (note: more permissive than sn-gamestate's GPL-3.0 on top of it).
- **State: ACTIVE.** 1,153 commits. Public since 2024-02-05; last module additions 2025-05-22.
- **Ships:** detectors YOLO / YOLOX / RTMDet / RTDETR; pose RTMPose / RTMO / ViTPose / YOLOPose;
  re-ID **KPReID / BPBReID**; trackers DeepSORT / StrongSORT / OC-SORT.
- **Requirements:** Python **3.12**, PyTorch 2.6, TorchVision 0.21, **CUDA 12.4**.
- **Config:** Hydra, fully YAML-driven; online and offline tracking modes.
- **Datasets supported:** DanceTrack, MOTChallenge, SportsMOT, SoccerNet.
- **Relevance:** ViTPose is already in here, which is the pose model Koshkina's pipeline needs for
  torso cropping and which Playbox & MIXI (4th) used.
- **Practicality flags:** Python 3.12 vs our >=3.11 floor is fine. **CUDA 12.4 requirement** and
  PyTorch 2.6 are the things to check against the 4 GB laptop's driver. Related project
  **CAMELTrack** (Context-Aware Multi-cue ExpLoitation for Online MOT, released 2025-05-02) lives in
  the same org and is a learned-association tracker.

## 13. sn-reid - re-identification devkit

- **URL:** https://github.com/SoccerNet/sn-reid
- **License: MIT.** 86 stars. **State: STALE - last updated 2023-07-07.** README still says "will be
  actively maintained during the course of the challenge" (a 2023 challenge that is over).
- **Built on:** Torchreid.
- **Dataset (SoccerNet-v3 ReID):** 340,993 player thumbnails from 400 games across 6 major leagues.
  Splits: train 12.12 GB, valid 2.35 GB, test 2.41 GB, challenge 1.75 GB.
- **Baseline:** mAP **59.11**, rank-1 **48.41** (2022 leaderboard).
- **2023 leaderboard (Table 5 of arXiv 2309.06006):**
  | Team | mAP | R-1 |
  |---|---|---|
  | **UniBw Munich - VIS (winner)** | **93.26** | **91.26** |
  | Baseline (Inspur) | 91.68 | 89.41 |
  | sjtu-lenovo | 91.51 | 89.17 |
  | MTVACV | 90.11 | 87.04 |
  | ErxuanBridge | 85.76 | 82.33 |
  | cm_test | 42.60 | 28.73 |
- **Task caveat that matters:** SoccerNet re-ID is **same-action, cross-viewpoint** matching (match a
  player across camera angles *at one instant*), NOT long-term identity across a match. The 93.26 mAP
  headline does not transfer to our LOTO top-1 setting.
- **Relevance:** the paper explicitly names the difficulty as "soccer players from the same team have
  very similar appearances", plus few samples per identity and wide resolution diversity. That is
  precisely the ~0.62 appearance ceiling we measured, described by the benchmark authors as intrinsic.
- **Pretrained weights:** no explicit availability statement in the README.

## 14. PRTreID - our current embedder, upstream

- **URL:** https://github.com/VlSomers/prtreid
- **Paper:** "Multi-task Learning for Joint Re-identification, Team Affiliation, and Role
  Classification for Sports Visual Tracking", https://arxiv.org/abs/2401.09942 (MMSports'23)
- **Authors:** Amir M. Mansourian, Vladimir Somers, Christophe De Vleeschouwer, Shohreh Kasaei.
- **What:** part-based person representation doing **role classification + team affiliation + re-ID
  simultaneously**. Built on BPBReID (https://arxiv.org/abs/2211.03679), adding a role-prediction
  head and two extra training objectives.
- **Tested on:** SoccerNet-GSR and SoccerNet-Tracking.
- **Relevance to the GK-never-named defect:** PRTreID already has a **role head that predicts
  goalkeeper**. Worth recording that the role signal exists in the model we are already running.
- **Related from same author:** Keypoint Promptable Re-Identification (ECCV24),
  https://github.com/VlSomers/keypoint_promptable_reidentification - SOTA re-ID robust to occlusion
  and multi-person ambiguity (i.e. the crowded-box problem in player crops).

## 15. sn-jersey - jersey number devkit

- **URL:** https://github.com/SoccerNet/sn-jersey
- **State: STALE - last updated 2024-07-02.** 30 stars, 7 commits on main. Challenge ran in 2023 only.
- **License:** not specified in the repo.
- **Task:** given a tracklet of a player (up to a few hundred frames), classify the jersey number.
- **Dataset:** 2,853 player tracklets total; challenge set 1,211 tracklets with hidden annotations.
- **Data format:** one folder per player containing thumbnail images; ground truth is a JSON dict
  mapping player ID (string) -> jersey number (int), with **`-1` meaning no number visible**.
- **Classes:** 1-99 plus the `-1` "not visible" class.
- **Download:** `mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])`
  via the SoccerNet pip package.
- **Eval:** EvalAI.
- **Relevance - this is the single most direct fit for silent OCR failure.** The benchmark treats
  "no number visible" as a **first-class predicted class**, and scores it. Our OCR has no such
  channel. It also gives a labelled test set on which an abstention rate could be measured.

## 16. Other SoccerNet org repos (state snapshot)

| Repo | Stars | Last updated | License | Note |
|---|---|---|---|---|
| sn-gamestate | 429 | 2026-05-02 | GPL-3.0 | active, see #11 |
| sn-mvfoul | 62 | 2025-01-09 | GPL-3.0 | multi-view foul |
| sn-teamspotting | 19 | 2025-08-26 | GPL-3.0 | Team Action Spotting 2025 devkit |
| sn-trackeval | 5 | 2025-07-29 | MIT | eval code, incl. GS-HOTA |
| SoccerNet (pip pkg) | 47 | 2025-07-16 | MIT | the downloader |
| sn-echoes | 18 | 2025-05-29 | - | see #22 |
| sn-depth | 12 | 2025-03-27 | - | monocular depth |
| sn-nvs | 8 | 2025-12-02 | - | novel view synthesis |
| sn-banner | 3 | 2025-12-26 | GPL-3.0 | - |
| sn-jersey | 30 | 2024-07-02 | - | STALE, see #15 |
| sn-calibration | 106 | 2024-06-18 | - | STALE |
| sn-caption | 36 | 2024-04-12 | - | STALE, see #23 |
| sn-spotting | 101 | 2024-02-07 | - | STALE |
| **sn-reid** | 86 | **2023-07-07** | MIT | STALE, see #13 |
| **sn-tracking** | 106 | **2023-07-07** | - | STALE, see #19 |
| SoccerNet-v3 | 76 | 2022-05-20 | MIT | STALE |
| sn-grounding | 9 | 2022-11-10 | - | STALE |
| ActiveSpotting | 3 | 2023-05-22 | MIT | - |
| PTS-baseline | - | 2023-02-15 | BSD-3 | - |

**Availability flag:** the four repos most relevant to identity (sn-reid, sn-tracking, sn-jersey,
sn-calibration) are all **unmaintained since 2023-2024**. Only sn-gamestate is current. Licenses are
inconsistent across the org and several repos state none at all.

---

# (c) Jersey number recognition - challenge winners and beyond Koshkina

## 17. SoccerNet 2023 Jersey Number Recognition - full leaderboard

15 teams, 157 submissions. Metric: overall classification accuracy over tracklets (including
correctly predicting `-1`). Source: arXiv 2309.06006 Table 7.

| Team | acc | | Team | acc |
|---|---|---|---|---|
| **ZZPM (winner)** | **92.85** | | Kalisteo | 58.35 |
| UniBw Munich - VIS | 90.95 | | FindNum | 54.91 |
| zzzzz | 88.08 | | jn | 47.55 |
| Mike Azatov | 82.05 | | Surya | 28.40 |
| MT-IOT | 81.70 | | tony506672558 | 20.06 |
| justplay | 77.77 | | lfriend | 5.68 |
| AIBrain Global Team | 75.18 | | zbq | 4.07 |
| SARG UWaterloo | 73.77 | | **Baseline (Random)** | **3.93** |

Report's own summary of the field: most teams used a **standard three-stage approach** - text
detection (DBNet++, MMOCR, DeepSolo, YOLO) -> text recognition (fine-tuned PP-OCRv3, PaddleOCR) ->
**majority voting to aggregate image-level results within a tracklet**.

### 17a. ZZPM (winner, 92.85)
Rui Peng, Kexin Zhang, Junpei Zhang, Yanbiao Ma, Licheng Jiao (Xidian University).
Text detection with a **modified pre-trained DBNet++** used as a **filter**: images from the training
set without a detection box at confidence >= 90 were **removed**. Data augmentation: rotation,
flipping, scaling, cropping, colour scrambling, noise, and **multi-frame image overlay**. The paper
singles out multi-frame fusion as providing more information and increasing robustness to number
shape, colour and texture. Recognition: fusion of votes from **SVTR-tiny, SVTR-small, SATRN, NRTR and
ASTER**.
**Relevance:** the winner's core move is aggressive *training-set filtering by detection confidence* -
i.e. explicitly modelling legibility - plus a 5-model vote. No code found.

### 17b. UniBw Munich - VIS (90.95) - CLIP zero-shot labeling
Konrad Habel, Fabian Deuser, Norbert Oswald.
**Did not train on the challenge labels at all.** Citing "the high amount of label noise in the
dataset of the challenge", they used **ViT-L/14 CLIP (OpenAI) to zero-shot auto-label the larger, more
diverse SoccerNet Re-Identification dataset** with no human annotation, then fine-tuned the image
encoders of two CLIP models on those pseudo-labels. Tracklet prediction by **majority voting using
only images with classification probability > 70%** for numbers 1-99.
Test 90.09, Challenge 90.95.
**Relevance - most directly transferable idea in this whole survey for silent OCR failure:** an
explicit per-image **confidence gate at 70%** before a frame is allowed to vote. Also: the same team
won the 2023 re-ID challenge (see #20), and they report the official jersey labels are noisy.

### 17c. Other 2023 jersey methods worth recording
- **zzzzz (88.08)** (Tencent): DeepSolo text detection for initial OCR, then a **transformer that
  does sequential prediction over the tracklet** where image patches are replaced by image feature
  representations. Model ensemble with multi-input resolution.
- **MT-IOT (81.70)** (Meituan): video transformer backbone, multi-task classification head; splits the
  task into **predicting the two digits separately + predicting the permutation of the digits**,
  binary cross-entropy on the digit head. Designed for long-tailed number distribution.
- **AIBrain Global Team (75.18)**: **uses player body orientation.** Confidence sorting based on
  jersey visibility and image quality; ESRGAN-based upscaling for low resolution; pose-estimation
  keypoints to localize the number and get body orientation; ranks tracklet prediction confidence by
  orientation + image quality assessment. **Relevance: the only 2023 entry to make orientation an
  explicit gate - the same signal Constructor.Tech used in the GSR 2024 winner.**
- **justplay (77.77)**: manually reviewed 700+ folders to clean the crops - corroborates the label
  noise claim.

## 18. Koshkina & Elder - jersey-number-pipeline (our current OCR)

- **Paper:** "A General Framework for Jersey Number Recognition in Sports Video", arXiv
  **2405.13896**, CVPRW 2024 CVsports pp. 3235-3244.
  PDF: https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf
- **Code:** https://github.com/mkoshkina/jersey-number-pipeline
- **Reported accuracy:** **91.4% on the hockey dataset, 87.4% on SoccerNet tracklets.**
  (For scale: that is below the 2023 challenge winner's 92.85 and below UniBw's 90.95, though the
  splits are not necessarily identical.)
- **Pipeline stages:** legibility classifier -> pose-guided cropping (ViTPose) -> scene text
  recognition (PARSeq) -> re-ID features with **Gaussian outlier removal** for occlusions ->
  tracklet-level prediction consolidation.
- **Weights:** provided via **Google Drive** - original PARSeq, hockey-fine-tuned PARSeq,
  **SoccerNet-fine-tuned PARSeq**, legibility classifiers for both datasets, Centroid-ReID weights,
  ViTPose checkpoints.
- **LICENSE FLAG: Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0).**
  Non-commercial only. Recording this because it constrains anything beyond research use.
- **Note:** the pipeline **already contains a legibility classifier** - a built-in abstention stage.
  Worth checking whether our integration runs it.
- **Also relevant:** "Jersey Number Recognition using Keyframe Identification from Low-Resolution
  Broadcast Videos", MMSports'23, https://dl.acm.org/doi/abs/10.1145/3606038.3616162

---

# (d) Tracking challenge winners and the Deep-EIoU lineage

## 19. SoccerNet 2023 Multiple Player Tracking - leaderboard

7 teams, 83 submissions. **No ground-truth boxes provided** (unlike 2022) - participants had to do
both detection and association. Source: arXiv 2309.06006 Table 6.

| Team | HOTA | DetA | AssA |
|---|---|---|---|
| **Kalisteo (winner)** | **75.61** | **75.38** | **75.94** |
| MTIOT | 69.54 | 75.18 | 64.45 |
| MOT4MOT | 66.27 | 70.32 | 62.62 |
| ICOST | 65.67 | 73.07 | 59.17 |
| SAIVA_Tracking | 63.20 | 70.45 | 56.87 |
| ZTrackers | 58.69 | 68.69 | 50.25 |
| scnu | 58.07 | 64.77 | 52.23 |
| Baseline | 42.38 | 34.41 | 52.21 |

- **Repo:** https://github.com/SoccerNet/sn-tracking (STALE, 2023-07-07, 106 stars)
- **Kalisteo (CEA):** YOLO-X detection; **TrackMerger v2** - two successive Hungarian assignments
  using IoU and centre distance; Kalman filter with camera motion compensation; **tracklets split if
  they cross each other** to remove association errors; those non-ambiguous tracklets then fine-tune a
  **Multiple Granularity Network re-ID model with triplet loss** (positives from the same tracklet,
  negatives from concomitant tracklets); tracklets iteratively merged by re-ID vector distance,
  preventing duplication and teleportation.
  **Relevance: self-supervised re-ID fine-tuning from the video's own unambiguous tracklets - no
  labels needed. Directly addresses the 0.62 appearance ceiling by adapting embeddings per match.**
- **MOT4MOT (Amazon):** DeepOCSORT + fine-tuned YOLOv8; appearance model fine-tuned on a curated
  subset; post-processing with interpolation and **appearance-free track merging**. Tech report:
  https://arxiv.org/abs/2308.16651
- **ICOST:** image deblurring preprocessing; YOLOX; OC-SORT with **Buffered Complete IoU (BCIoU)**;
  camera motion compensation; **co-occurrence-aware hierarchical clustering** to merge tracklets -
  merging is *prohibited* between two tracklets that have co-occurrent detections.
  **Relevance: the co-occurrence constraint is the correct hard prior for tracklet merging and is
  exactly the kind of check a GTA-Link connector needs.**
- **CO-MOT / MOTRv2 end-to-end entry:** https://github.com/BingfengYan/CO-MOT (69.5 HOTA claimed).
- Report's conclusion: "the proposed methods perform much better than off-the-shelf open-source
  baselines. However, there is still some room for improvement in both detection and association."

## 20. CLIP-ReIdent (2023 re-ID winner, UniBw Munich - VIS)

- **Paper:** "CLIP-ReIdent: Contrastive Training for Player Re-Identification", MMSports'22,
  https://doi.org/10.1145/3552437.3555698
- **Result:** SoccerNet 2023 re-ID winner. Test mAP 93.51, Challenge mAP **93.26**, R-1 91.26.
  Also 1st place in the MMSports'22 Player Re-Identification challenge.
- **Method:** ensemble of three CLIP-based models (OpenCLIP + OpenAI CLIP); **custom sampling strategy
  that samples players of the same action together**; self-designed **per-action re-ranking** as
  post-processing; vision encoders fine-tuned with contrastive training and InfoNCE.
- **Report's summary of the field:** 2023 re-ID solutions were "mostly based on ViT, and most teams
  adopted the foundation model CLIP", with model ensembling, per-action re-ranking, and custom
  action-based training samplers.
- **Relevance:** same team as the 90.95 jersey result. CLIP-based embeddings beat the
  torchreid/OSNet family across the board in this benchmark.

## 21. Deep-EIoU and the GTA lineage (the tracker family that keeps winning)

### 21a. Deep-EIoU
- **Paper:** "Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking
  in Sports", arXiv **2306.13074**, WACV 2024 RWS Workshop.
  PDF: https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Huang_Iterative_Scale-Up_ExpansionIoU_and_Deep_Features_Association_for_Multi-Object_Tracking_WACVW_2024_paper.pdf
- **Code:** https://github.com/hsiangwei0903/Deep-EIoU
- **Authors:** Hsiang-Wei Huang, Cheng-Yen Yang, Jiacheng Sun, Pyong-Kun Kim, Kwang-Ju Kim, Kyoungoh
  Lee, Chung-I Huang, Jenq-Neng Hwang.
- **Results:** **85.4 HOTA on SoccerNet-Tracking test**, 77.2 HOTA on SportsMOT.
- **Key design:** **abandons the Kalman filter entirely**, using iterative scale-up ExpansionIoU plus
  deep features instead. Motivation: sports motion is irregular and non-linear, so Kalman's constant-
  velocity assumption hurts.
- **Relevance:** this is the tracker in the GSR 2025 winner (KIST-GSR) and the direct alternative to
  our ByteTrack. Its whole premise is that ByteTrack-style Kalman association is the wrong prior for
  football. Online, inference-only, no training needed. Small - plausible on 4 GB.

### 21b. GTA / gta-link (what we already added)
- **Paper:** "GTA: Global Tracklet Association for Multi-Object Tracking in Sports", arXiv
  **2411.08216**, ACCV 2024 MLCSA workshop.
  PDF: https://openaccess.thecvf.com/content/ACCV2024W/MLCSA2024/papers/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.pdf
  OpenReview: https://openreview.net/forum?id=hsi47B381b
- **Code:** https://github.com/sjc042/gta-link
- **What it does:** appearance-based global tracklet association - **splits tracklets containing
  multiple identities**, then **connects tracklets of the same identity**. Model-agnostic,
  plug-and-play, **offline post-processing**.
- **Results:** on SoccerNet, raised HOTA from **79.41 to 83.11** (+3.70) across multiple trackers.
  SOTA on SportsMOT at 81.04 HOTA.
- **Relevance:** confirms the measured headroom of the connector we just added, on this exact
  benchmark: roughly +3.7 HOTA, and the split half is as important as the merge half.

### 21c. GTATrack (2026) - Deep-EIoU + GTA combined
- **Paper:** "GTATrack: Winner Solution to SoccerTrack 2025 with Deep-EIoU and Global Tracklet
  Association", arXiv **2602.00484** (submitted 2026-01-31).
- **Code:** https://github.com/ron941/GTATrack-STC2025
- **Authors:** Rong-Lin Jian, Ming-Chi Luo, Chen-Wei Huang, Chia-Ming Lee, Yu-Fan Lin, Chih-Chung Hsu.
- **Result:** 1st place SoccerTrack Challenge 2025. HOTA 0.60, false positives reduced to 982.
- **What:** hierarchical two-stage - Deep-EIoU for motion-agnostic online association, GTA for
  trajectory-level refinement. Short-term matching + long-term identity consistency.
- **Relevance:** the exact composition (Deep-EIoU + GTA) that the GSR 2025 winner also used, now with
  **public code**. One of the few winner solutions in this survey that actually released.

### 21d. SoccerTrack v2 (adjacent dataset)
- **Paper:** "SoccerTrack v2: A Full-Pitch Multi-View Soccer Dataset for Game State Reconstruction",
  https://arxiv.org/abs/2508.01802

---

# (e) Commentary / ASR as weak labels

## 22. SoccerNet-Echoes

- **Paper:** https://arxiv.org/abs/2405.07354 (IEEE https://ieeexplore.ieee.org/document/10935951/)
- **Code/data:** https://github.com/SoccerNet/sn-echoes (18 stars, last updated 2025-05-29)
- **What:** automatically generated transcriptions of SoccerNet broadcast audio. Whisper ASR, with
  Google Translate to English where needed. Games narrated live in **10 different languages**.
- **Formats provided:** JSON per game with `{start, end, text}` segments, in versions
  **whisper_v1, whisper_v1_en, whisper_v2, whisper_v2_en, whisper_v3**.
- **Coverage:** organized by league/season - England EPL 2014-2015 and 2015-2016 visible, UEFA CL,
  and further leagues. **This is the SoccerNet 500/v2 game corpus, i.e. 2014-2017 era, NOT EPL
  2024-25.**
- **License:** **none stated in the repo.**
- **CRITICAL LIMITATION, stated by the authors:** "the model's capability to accurately identify
  entity names (players, teams, stadiums, etc.) is limited." Whisper systematically mangles player
  names. This is stated in the paper itself, not inferred.
- **Relevance:** the obvious weak-label source for naming, with an author-acknowledged failure mode on
  exactly the entity type we need. Also the wrong seasons for a ManU EPL 24-25 scope.

## 23. MatchTime / SN-Caption

- **MatchTime paper:** "MatchTime: Towards Automatic Soccer Game Commentary Generation", arXiv
  **2406.18530**, EMNLP 2024 Oral. HTML: https://arxiv.org/html/2406.18530v2
- **Code:** https://github.com/jyrao/MatchTime
- **Project page:** https://haoningwu3639.github.io/MatchTime/
- **Weights:** https://huggingface.co/Homie0609/MatchVoice
- **Data:** https://drive.google.com/drive/folders/14tb6lV2nlTxn3VygwAPdmtKm7v0Ss8wG
- **Datasets:** `MatchTime` (train/val, video features + caption labels) and
  **`SN-Caption-test-align`** - a test set with **manually aligned** ground-truth captions.
- **The contribution that matters here is the ALIGNMENT, not the generation.** Commentary in
  SoccerNet-Caption is temporally misaligned with the video. MatchTime fixes it in two stages:
  **coarse alignment** using WhisperX ASR + **LLaMA3 as an agent**, then **fine-grained alignment via
  contrastive learning**.
- **License:** **none stated in the repo.**
- **Requirements note:** README says "PyTorch >= 2.0.0 (If use A100)" - built for A100-class hardware.
- **SoccerNet-Caption (the underlying dataset):** CVPRW 2023,
  https://doi.org/10.1109/cvprw59228.2023.00536 ; repo https://github.com/SoccerNet/sn-caption
  (STALE, 2024-04-12).
- **Relevance:** if commentary is to be used as weak identity supervision, temporal misalignment is
  the first obstacle and this is the published, code-released solution to it.

## 24. SoccerNet 2026 Player-Centric Ball Action Spotting (PCBAS) + FOOTPASS

**This is the closest published task to "name the player who did the thing", and it is brand new.**

- **Challenge:** SoccerNet 2026, Section in arXiv 2607.07320.
- **Task:** identify and temporally localize an action AND determine **both the team affiliation and
  the jersey number of the acting player**. 8 classes: Drive, Pass, Cross, Shot, Header, Throw-in,
  Tackle, Block. Single-timestamp annotation per action.
- **Metric:** **Macro F1 at tau=0.15.** A true positive requires the prediction to be within
  **+/-12 frames** of ground truth AND match action class AND team affiliation AND **jersey number**.
  Score = average of class-wise F1 over foreground classes.
- **Includes replays and close-ups:** "all annotated actions ... including those occurring during
  replay segments or close-up shots, making the task particularly challenging under realistic
  broadcast conditions."
- **Baseline:** TAAD (Track-Aware Action Detection) + DST (Denoising Sequence Transduction Model) =
  **46.41** Macro F1@0.15.

**Leaderboard:**

| Rank | Team | Macro F1@0.15 |
|---|---|---|
| 1 | FSITAHAKOM (PAVE) | 58.94 |
| 2 | AISATSANZ (PC-SSAS) | 56.40 |
| 3 | TeamKIST | 55.69 |
| 4 | UniBW Munich VIS | 50.35 |
| 5 | WRF32010 | 46.06 |
| - | **Baseline** | **46.41** |
| 6 | Sarthi-GameChanger | 44.63 |

- **Winner PAVE** (Faisal Altawijri, Ismail Mathkour): temporal transformer over **ROI-aligned X3D
  player features** giving frame-level temporal context per player tracklet; **replaced flat
  role-vector encoding with per-player attention over 26 role slots**; 4 independently trained
  variants fused by weighted event fusion; **agreement filtering** suppresses single-model false
  positives; a tackle-specific exception preserves recall on the rarest class. +12.53 over baseline.
- **PC-SSAS (2nd, Sber AI):** fuses tactical player states, ball detections and **frozen full-frame
  visual embeddings**. Converts sparse tracking rows into dense frame-player tensors with kinematic
  features, visibility masks, ball geometry and **possession heuristics**. Actor tokens from tabular
  features cross-attend to global visual context. **Mamba** state-space temporal backbone.
  Event-centered sliding windows, Gaussian temporal targets.
- **WRF32010 (5th):** dual-backbone X3D-L (CNN) + Swin3D (Transformer), decision-level fusion with
  temporal Gaussian filtering and weighted averaging.
- **Report's own cross-cutting finding:** two recurring themes across all submissions were
  **(1) ball tracking for ball-centered visual features and player-ball interaction cues, and
  (2) handcrafted tactical features derived from the provided game-state information.** Tackle was
  the hardest class for everyone (rare + occluded).

### FOOTPASS dataset (the data behind PCBAS)

- **Paper:** "FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action
  Spotting in Soccer Broadcast Videos", arXiv **2511.16183**.
  HAL: https://hal.science/hal-05373478v1
  ScienceDirect (CVIU): https://www.sciencedirect.com/science/article/pii/S1077314226001578
- **Code/baselines:** https://github.com/JeremieOchin/FOOTPASS
- **Data:** https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026/tree/main
- **Eval server:** https://www.codabench.org/competitions/11232/
- **Contact:** jeremie.ochin@minesparis.psl.eu
- **Size:** **54 complete men's matches**, 2023/24 season - Ligue 1, Bundesliga, Serie A, La Liga,
  UEFA Champions League. 1920x1080 @ 25 fps, ~**81 hours** of broadcast video.
  **102,992 manually-validated play-by-play on-ball event annotations**, each a
  **`(frame, team, jersey, class)` tuple.**
- **Annotations also include:** lines, goal parts, players, referees, teams, salient objects, jersey
  numbers, player correspondences between views; plus extended game state - player positions and
  velocities on the pitch, team membership, jersey numbers, **roles (13 categories, Goalkeeper through
  Right Back)**, and single-player tracking data.
- **Baselines provided:** TAAD (visual), TAAD+GNN (spatio-temporal graph reasoning for multi-agent
  context), TAAD+DST (game-level reasoning).
- **Game-state inputs to the baselines are GROUND TRUTH**, not predicted - positions, velocities,
  tracking, team, jersey, role.
- **LICENSE / ACCESS:** annotations and baselines are **CC BY-NC 4.0** (non-commercial). **The videos
  require NDA approval** via
  https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform
- **Pretrained weights:** not mentioned in the repo. GPU requirements: not specified.
- **Relevance:** an 81-hour broadcast corpus where every on-ball action carries a **validated jersey
  number and team**, i.e. exactly the label type our naming factor 0.609 is measured against. Note
  the seasons are 2023/24 and the leagues exclude the EPL - it is Ligue 1 / Bundesliga / Serie A /
  La Liga / UCL only, so it is out-of-scope footage for a ManU EPL 24-25 demo but in-scope as a
  method benchmark. Also note the baselines consume ground-truth game state, so published PCBAS
  numbers are NOT end-to-end-from-pixels numbers.

---

# (f) Identity via commentary / play-by-play weak supervision

## 25. Weakly-supervised player identification - the prior literature

- **Lu, Okuma, Little et al., "Learning to Track and Identify Players from Broadcast Sports Videos",
  TPAMI 2012** - https://www.cs.ubc.ca/~murphyk/papers/weilwun-pami12.pdf
  The canonical result: **weak supervision from play-by-play text achieves accuracy comparable to
  strong supervision using ~200 labels vs the ~20,000 labels the strongly-supervised approach needs.**
  This is the ~100x label-efficiency claim that underwrites the whole commentary-as-weak-labels idea.
- **Gadde & Jawahar, "Transductive Weakly-Supervised Player Detection using Soccer Broadcast
  Videos", WACV 2022** -
  https://openaccess.thecvf.com/content/WACV2022/papers/Gadde_Transductive_Weakly-Supervised_Player_Detection_Using_Soccer_Broadcast_Videos_WACV_2022_paper.pdf
  Trains transductively using valid target-domain instances obtained by pruning noise.
- **Vats et al., "Player Identification in Hockey Broadcast Videos", arXiv 2009.02429** - hockey
  analogue; the sport where Koshkina's pipeline originates.
- **"Player-Centric Multimodal Prompt Generation for LLM-Based Identity-Aware Basketball Video
  Captioning", arXiv 2507.20163** - identity-aware captioning in basketball; the closest recent work
  on binding a name to an actor for narration purposes.
- **Survey context:** "A Survey of Deep Learning in Sports Applications: Perception, Comprehension,
  and Decision", arXiv 2307.03353 - notes that online commentaries could be used for action
  annotation in a weakly-supervised setting, and that matching faces with textual cues in soccer has
  been explored.

**Gap noted:** no paper found that does commentary-name -> tracklet binding on *soccer* broadcast
end to end. The 2012 TPAMI result is hockey/basketball-era play-by-play, and SoccerNet-Echoes
provides the transcripts but explicitly warns its entity names are unreliable. PCBAS/FOOTPASS
sidesteps commentary entirely by annotating jersey numbers directly.

---

# Cross-cutting availability flags

1. **Constructor.Tech's "From Broadcast to Minimap" (arXiv 2504.06357), the most-cited GSR result and
   the 63.81 SOTA we benchmark against, has NO public code.** No repo in the paper, no project page,
   nothing in the 2025 challenge report. Verified across the abs page, ar5iv HTML, and the challenge
   report.
2. **No GSR challenge participant, 2024 or 2025, released code.** The GSR sections of arXiv 2508.19182
   contain zero URLs. Every top-10 method is reconstruct-from-description only.
3. **The four SoccerNet identity repos are unmaintained:** sn-reid and sn-tracking last touched
   2023-07-07, sn-jersey 2024-07-02, sn-calibration 2024-06-18. Only sn-gamestate is current
   (2026-05-02).
4. **sn-gamestate GS-HOTA evaluation was broken** before the TrackLab 1.3.24 fix committed May 2026.
5. **License inconsistency across the org:** sn-gamestate GPL-3.0, TrackLab MIT, sn-reid MIT,
   sn-trackeval MIT, sn-jersey unspecified, sn-echoes unspecified, sn-caption unspecified.
6. **Koshkina's jersey-number-pipeline is CC BY-NC 3.0** (non-commercial), and its weights are on
   Google Drive rather than a durable host.
7. **FOOTPASS annotations are CC BY-NC 4.0 and its video requires an NDA** via Google Form.
8. **MatchTime and sn-echoes state no license at all.**
9. **LLAMA-3.2-Vision (the GSR 2025 winner's identity model) is gated on HuggingFace** and its
   community license carries an EU-territory restriction on the vision models. Smallest variant is
   11B - not runnable at 4 GB without heavy quantization.
10. **TrackLab pins CUDA 12.4 / PyTorch 2.6 / Python 3.12** - needs checking against the 4 GB laptop.
11. **SoccerNet-Echoes covers 2014-2017-era games, not EPL 2024-25**, and its authors state entity
    (player) name transcription is unreliable.
12. **Deep-EIoU, gta-link, GTATrack-STC2025, CO-MOT, prtreid, tracklab, sn-gamestate and
    jersey-number-pipeline are the only identity-relevant repos in this survey with code that is
    actually published and fetchable.**

# Source URL index

Papers: 2508.19182 | 2607.07320 | 2409.10587 | 2309.06006 | 2504.06357 | 2404.11335 | 2405.13896 |
2401.09942 | 2211.03679 | 2306.13074 | 2411.08216 | 2602.00484 | 2405.07354 | 2406.18530 |
2511.16183 | 2508.01802 | 2308.16651 | 2009.02429 | 2507.20163 | 2307.03353 (all arxiv.org/abs/...)

Repos: github.com/SoccerNet/sn-gamestate | .../sn-reid | .../sn-tracking | .../sn-jersey |
.../sn-echoes | .../sn-caption | .../sn-trackeval | github.com/TrackingLaboratory/tracklab |
github.com/VlSomers/prtreid | github.com/VlSomers/keypoint_promptable_reidentification |
github.com/mkoshkina/jersey-number-pipeline | github.com/hsiangwei0903/Deep-EIoU |
github.com/sjc042/gta-link | github.com/ron941/GTATrack-STC2025 | github.com/BingfengYan/CO-MOT |
github.com/JeremieOchin/FOOTPASS | github.com/jyrao/MatchTime | github.com/Spiideo/soccersegcal

Data/eval: huggingface.co/datasets/SoccerNet/SN-PCBAS-2026 | huggingface.co/Homie0609/MatchVoice |
codabench.org/competitions/11232 | eval.ai/web/challenges/challenge-page/2251/overview |
soccer-net.org/challenges
