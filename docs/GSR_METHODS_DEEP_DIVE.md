# SoccerNet GSR - METHODS DEEP DIVE (raw findings)

Surveyed 2026-07-29 by the research-search worker. Scope: everything that scores well on SoccerNet
Game State Reconstruction, with training-scale requirements, for a college-GPU-cluster roadmap.

Companion to `docs/SOCCERNET_SURVEY_RAW.md` (2026-07-28, shallow pass). This document does NOT
repeat that one; it goes deeper on methods, training recipes, and compute. Where the two disagree,
Section 0.2 records the correction.

Raw collection only. No recommendations. Every claim carries its URL.

Our stack, for relevance tagging only: fine-tuned YOLO detect, BoT-SORT/ByteTrack, PnLCalib
(0.13-0.20 m), PRTreID, Koshkina-lineage per-crop OCR at 0.85 floor, GTA connector, greedy namer;
GS-HOTA 33.20 on the public split.

---

## 0.1 What is new in this pass (headline)

1. **The GSR 2025 winner DOES have a full method paper.** The challenge-report entry "KIST-GSR"
   (GSR-1) is authored by Yin May Oo, Yewon Hwang, Vanyi Chao, Amrulloh Robbani Muhammad, Jinwook
   Kim - the exact author set of **Broadcast2Pitch (WACV 2026)**. The prior survey listed
   Broadcast2Pitch as an "independent" work and KIST-GSR as "no standalone paper found". They are
   the same system. Broadcast2Pitch contains full ablations, per-component comparisons, and
   training compute. See Section 2.1.
2. **Constructor.Tech's 2024 paper contains a complete training recipe** (per-model dataset sizes,
   epochs, LRs, input sizes, parameter counts, and the GPU: one A100 40 GB). See Section 1.1.
3. **The identity-model swap is the largest single published lever.** Inside one fixed GSR pipeline,
   Broadcast2Pitch Table 5 measures: PRTreID + EasyOCR + ResNet-18 colour = **18.11 GS-HOTA**;
   CLIP encoder + attribute heads = **60.13**; LLaMA-3.2-Vision = **61.48**. Same detector, same
   tracker, same calibration. See Section 2.1.4.
4. **The GSR test-phase evaluation server is still open.** Codabench competition **4365**
   ("2025 SoccerNet GSR - Test Phase") has phase status `current`, **no end date**, auto-run
   submissions enabled. The Challenge phase (4469) closed 2025-05-07. See Section 6.
5. **A new jersey SOTA exists that beats Koshkina on the challenge split and has an explicit
   abstention channel:** Grad, CVPRW 2025, 85.62% challenge-set tracklet accuracy vs Koshkina's
   79.31% on the same split, using Dirichlet evidential uncertainty. See Section 4.3.1.

## 0.2 Corrections to the prior survey (`SOCCERNET_SURVEY_RAW.md`)

| Prior claim | Correction | Evidence |
|---|---|---|
| "Constructor.Tech 2025 components: RF-DETR detection; BoT-SORT + GTA; optical flow for temporal calibration" | **Not sourced.** In arXiv 2508.19182 Table 4, Constructor.Tech carries **no GSR-n superscript**, meaning it submitted **no method summary** in 2025. RF-DETR/BoT-SORT/GTA/optical-flow appear only in the organisers' general Section 5.5 survey paragraph describing the field as a whole. | 2508.19182 p.8 Table 4 + Sec 5.5 |
| "KIST-GSR - no standalone paper found" | Broadcast2Pitch (WACV 2026) IS that paper; author sets match exactly. | 2508.19182 p.14 GSR-1 authors vs WACV 2026 paper authors |
| "Koshkina 87.4% on SoccerNet" | 87.45% is the **Test** split. On the **Challenge** split Koshkina scores **79.31%**. Comparisons must state the split. | Grad CVPRW 2025 Table 6 |
| "Broadcast2Pitch: code none found; no arXiv mirror" | Still true (no code, no arXiv), but the CVF open-access PDF is fetchable with a browser UA via curl (WebFetch gets 403). | see Section 8 |
| "TrackLab pins CUDA 12.4 / PyTorch 2.6 / Python 3.12" | That is **standalone TrackLab**. `sn-gamestate`'s own `pyproject.toml` pins `python >=3.9,<3.10`, `torch==1.13.1`, `tracklab==1.3.24`. These are two different environments. Section 3.2 - this is load-bearing for cluster hardware. | raw.githubusercontent.com/SoccerNet/sn-gamestate/main/pyproject.toml |
| "PnLCalib vs No Bells Just Whistles" as two works | Same arXiv id **2404.08401**. v1 = "No Bells, Just Whistles" (CVPRW 2024); v4 = "PnLCalib" (CVIU vol 267, April 2026). NBJW is the keypoint stage; PnLCalib adds the points-and-lines refinement on top. | arxiv.org/abs/2404.08401v1 vs v4 |

---

# 1. GSR 2024 challenge

Report: **arXiv 2409.10587** - https://arxiv.org/abs/2409.10587 (HTML: /html/2409.10587v1)

## 1.0 Full 2024 leaderboard (GS-HOTA / GS-DetA / GS-AssA), challenge split

| Team | GS-HOTA | GS-DetA | GS-AssA |
|---|---|---|---|
| Constructor tech | 63.81 | 49.52 | 82.23 |
| UPCxMobius | 43.15 | 30.46 | 61.16 |
| JAM | 34.40 | 19.38 | 61.08 |
| XJTU_MM (JNR) | 33.57 | 20.44 | 55.16 |
| VIPLab | 29.82 | 16.16 | 55.03 |
| JustTesting | 29.08 | 13.78 | 61.37 |
| Robo Space | 26.92 | 11.04 | 65.66 |
| sjc | 24.20 | 10.16 | 57.68 |
| ABL | 23.73 | 9.94 | 56.67 |
| testingGS | 23.73 | 9.94 | 56.67 |
| **Baseline** | **23.36** | **9.80** | **55.69** |
| SAIVA_Ball | 23.08 | 10.13 | 52.62 |
| Football AI Lab | 22.96 | 9.92 | 53.17 |
| playbox x NUSG | 21.26 | 7.55 | 59.90 |
| UWIPL | 20.33 | 7.85 | 52.64 |
| eidos.ai | 8.92 | 1.51 | 52.69 |
| Eidos | 8.92 | 1.51 | 52.69 |

Note the 2024 winner's margin: 63.81 vs 43.15 second place. The same 63.81 was resubmitted in 2025
and finished 2nd behind 63.90.

## 1.1 Constructor.Tech - GSR 2024 WINNER (63.81) - FULL METHOD

- **Paper:** "From Broadcast to Minimap: Achieving State-of-the-Art SoccerNet Game State
  Reconstruction", **arXiv 2504.06357** - https://arxiv.org/abs/2504.06357
  HTML: https://arxiv.org/html/2504.06357v1 | ar5iv: https://ar5iv.labs.arxiv.org/html/2504.06357
  (CVPR 2025 CVsports Workshop)
- **Authors:** Vladimir Golovkin, Nikolay Nemtsev, Vasyl Shandyba, Oleg Udin, Nikita Kasatkin,
  Pavel Kononov, Anton Afanasiev, Sergey Ulasen, Andrei Boiarov (Constructor Tech).
- **Code / weights: NONE.** No repository, no project page, no availability statement anywhere in
  the abs page, the HTML full text, or the challenge report.
- **Final scores:** GS-HOTA **63.81**, GS-DetA **49.52**, GS-AssA **82.23** (challenge split).
  Independently evaluated on the **test** split by Broadcast2Pitch: **55.82 / 41.67 / 74.86**.

### 1.1.1 Component-by-component

**(a) Object detection**
- **YOLOv5m**, fine-tuned. 35.5 M params.
- Input **1920x1080**. Two classes (athletes, ball).
- Training data: **66k images** with athlete+ball boxes (proprietary).
- Adam, initial LR 0.01, momentum 0.95, decay 5e-4, LambdaLR schedule, CIoU + BCE loss,
  **100 epochs**.
- Stated rationale: one-stage, lightweight, quality/speed tradeoff.

**(b) Camera parameter estimation (the distinctive part)**
- Custom **SegFormer** encoder-decoder, **5 M params**, input **512x288**.
- Regresses **seven camera parameters** directly: position (x, y, z), orientation (pan, roll, tilt),
  and FOV. Not a homography, not keypoints - camera parameters.
- Two heads: a *parameters* head (PSA layer -> Conv2D stack -> average pooling) and a *heatmaps*
  head (PSA -> Conv2D -> PixelShuffle) used **during training only** as an auxiliary task.
- Loss: `L = w1*L2_world + w2*L2_camera + w3*L1_parameters + w4*L2_heatmap` with tuned weights
  **w1=0.048, w2=2.49, w3=1.0, w4=10.0**.
- Training data: **22,000 real images** with keypoint annotations from a modified TVCalib framework
  + **40,000 synthetic images** from a customised **Google Research Football Simulator**
  (62k total).
- Schedule: **400k batches total**, batch size 8, AdamW, max LR 5e-4, cosine annealing with
  **20k warmup steps**; **phase 1 = 200k batches on real+synthetic mixed, phase 2 = 200k batches on
  real only.**
- **Refinement:** separate **ResNet18 Field Keypoints Model** detecting **74 keypoints**
  (36k-image keypoint dataset, 150 epochs, LR 0.01, OneCycleLR, AdaptiveWing loss, input 480x270,
  11 M params). Brute-force search over predefined delta combinations picks the camera parameters
  minimising L2 between projected keypoints and the ideal pitch model.
- **Temporal smoothing:** Savitzky-Golay filter with a **15-frame delay**, max adjustment
  +/-2 degrees on angles and +/-2 m on positions.

**(c) Tracking**
- **DeepSORT variant operating in real-world pitch coordinates**, not pixel coordinates. This is
  the structural difference from a stock tracker: association happens after projection.
- Additional gates: player-orientation restrictions (blocks physically implausible 180-degree
  direction flips between frames) and TeamID-embedding constraints.
- Deployed with **NVIDIA DeepStream SDK + TensorRT FP16**, reaching **up to 80 FPS on an RTX 3080Ti
  laptop GPU**.

**(d) Re-identification**
- **ResNet50**, 25.6 M params, input **256x128**, triplet loss, SGD, LR 1e-3.
- Training data: **280k crops from 370 unique players**.
- Triplet sampler: **28 classes x 7 samples** per batch.

**(e) Orientation prediction**
- **ResNet18**, 11 M params, input **62x32**, 4 discrete classes (left, up, right, down),
  CrossEntropy, SGD, LR 1e-4.
- Training data: **20k labelled images**.
- Used as a *consistency constraint inside the tracker*, not as a jersey-legibility gate in this
  paper (contrast with AIBrain 2023 and Broadcast2Pitch).

**(f) Jersey number recognition**
- **Modified ResNet18 with two classification heads**, 17 M params, input **32x32** crops of the
  jersey-visible upper half of the athlete box. **No text detection stage, no OCR.**
- Head 1: presence + value of the leading digit (1-9 or none) via **BinaryFocalLoss**.
  Head 2: second digit (0-9) via CrossEntropy.
- AdamW, LR 1e-4, ReduceLROnPlateau.
- Training data: **70k images**, 100 classes.
- Frame-level predictions; tracklets later merged by matching jersey numbers.

**(g) TeamID embeddings**
- **OSNet**, only **0.3 M params**, input **64x32**, trained as a **111-class classification** over
  111 unique uniform configurations. **550k images.** 40 epochs, Adam, LR 0.001, momentum 0.9,
  weight decay 5e-4, single-step LR schedule gamma=0.1.

**(h) Anomaly model** (not previously recorded anywhere)
- **ResNet18**, 11 M params, input 62x32, binary, **16k images**, Adam, LR 1e-3.

**(i) Role / team assignment logic**
- Five target clusters: left team, right team, referees, left GK, right GK.
- Query all athletes within **30 m of pitch centre**; cluster into three; two largest = field teams,
  smallest = referees. Fallback when only a penalty area is visible: use athletes outside it.
- **Goalkeepers:** query athletes inside the penalty area, filter by cosine distance to the TeamID
  embeddings of nearby clusters, then assign the GK cluster. (Geometric + embedding, not a learned
  GK class.)
- **Left/right:** per-frame mean x-coordinate of each cluster, votes collected across frames,
  majority wins.

**(j) Tracklet post-processing - four stages**
1. **Split** tracklets so each holds a single (jersey, team) label pair.
2. **Merge by jersey number**, gated on temporal non-overlap, physical feasibility of the distance
   within the time gap, and consistent team IDs.
3. **Merge by ReID similarity** for tracklets lacking a jersey number, using BOTH a mean-embedding
   cosine threshold AND a pairwise NxM max-similarity matrix. The paper states combining the two
   strategies is what raises merge accuracy.
4. **Linear interpolation** across short gaps.
- Reported effect: **"90% reduction in tracklets"** and significantly fewer tracklet swaps.

### 1.1.2 Training compute (Constructor.Tech)

- **All training on a single NVIDIA A100 40 GB.** Total GPU hours not disclosed.
- **No ablation table of any kind** in the paper - only final scores. There is no published
  attribution of the 63.81 to individual components.
- Stated future work: unify the camera and field-keypoint models; replace YOLOv5; model orientation
  as a 360-degree distribution instead of 4 classes; relax jersey matching to single-digit matching.

## 1.2 UPCxMobius - GSR 2024 runner-up (43.15)

Only the challenge-report summary exists (arXiv 2409.10587, GSR-2). No paper, no code.
- Replaces camera calibration with **hierarchical keypoint generation using an HRNetv2-based
  encoder-decoder**.
- Person detection **retrained on ~2,000 custom-annotated frames** to classify
  player / goalkeeper / referee directly at the detector.
- Dedicated **jersey digit detection model** trained on public data plus **~1,000 custom-annotated
  SoccerNet frames**; recognised digits are merged into numbers.
- Relevance note: this is the smallest annotation budget in the whole survey to reach 2nd place -
  ~3,000 hand-annotated frames total.

## 1.3 JAM - GSR 2024 third (34.40)

**No method summary submitted.** Nothing published. Flagged as unavailable.

## 1.4 Other documented 2024 entries

- **Robo Space (26.92):** TrackLab framework; fine-tuned YOLOv8 on a custom dataset; refined
  **BPBReID**; a "Global Temporal Attention" mechanism; temporal smoothing.
- **playbox x NUSG (21.26):** fine-tuned YOLOv8X; kept stock tracklab tracking; **corrected the
  TVCalib implementation** + temporal filtering; **two-stage CLIP-based jersey method** fine-tuned
  on **3,000 manually annotated boxes**, reaching **0.78 tracklet accuracy**; used **GPT-4o** to
  refine predicted roles, team affiliation and jersey numbers. (Same core team as 2025's
  Playbox & MIXI, which reached 61.64.)
- **eidos.ai (8.92):** Siamese network producing box embeddings, KMeans at varying k, then a
  **Random Forest** assigning roles/teams to clusters using game-context features from tracks and
  fieldmap position. (Same team reached 46.24 in 2025 after switching to a VLM.)

---

# 2. GSR 2025 challenge

Report: **arXiv 2508.19182** - https://arxiv.org/abs/2508.19182 (PDF is 15 pages; GSR = Section 5
on pp. 7-8, team summaries in Appendix 7.4 on pp. 14-15).
Task leads: Marc Gutierrez-Perez, Victor Joos, Vladimir Somers, Floriane Magera.

Participation: **14 teams, 66 submissions, 14 leaderboard entries** (report text says 76
participants; the Codabench challenge-phase API says 97 registered participants, 66 submissions).
12 teams beat the baseline.

## 2.0 Leaderboard (challenge split)

| Rank | Team | GS-HOTA | GS-DetA | GS-AssA | Summary in report? |
|---|---|---|---|---|---|
| 1 | KIST-GSR | 63.90 | 51.36 | 79.55 | yes (GSR-1) |
| 2 | Constructor.Tech | 63.81 | 49.52 | 82.23 | **no** |
| 3 | lianyou | 62.76 | 46.17 | 85.33 | yes (GSR-3) |
| 4 | Playbox & MIXI | 61.64 | 45.04 | 84.36 | yes (GSR-4) |
| 5 | tyler_durden | 57.08 | 41.52 | 78.48 | **no** |
| 6 | SJTU Multi-Modal Perception Group | 50.06 | 32.98 | 75.99 | **no** |
| 7 | eidos.ai | 46.24 | 31.06 | 68.85 | yes (GSR-7) |
| 8 | UTokyo Football Lab | 40.13 | 21.33 | 75.51 | yes (GSR-8) |
| 9 | hjkim | 36.90 | 17.56 | 77.58 | yes (GSR-9) |
| 10 | junkwang | 33.62 | 18.09 | 62.51 | **no** |
| 11 | lsmuqi | 32.62 | 17.03 | 62.50 | **no** |
| 12 | hayden97 | 32.59 | 16.96 | 62.61 | **no** |
| - | **Baseline [26]** | **29.01** | **13.71** | **61.41** | - |
| 13 | Laboratory for Biosignal Processing | 28.45 | 13.50 | 59.99 | yes (GSR-13) |
| 14 | johnstreet | 14.26 | 9.67 | 21.10 | **no** |

**No team reported training compute, GPU count, or epochs in the 2025 report.** Not one. The only
2025 training-scale numbers in this document come from the winner's separate WACV paper.

Organisers' own Section 5.5 summary of the field (verbatim substance): fine-tuned detectors are now
standard (YOLO-X, YOLOv8, YOLOv11, RF-DETR); tracking-by-detection still dominant (Deep-EIoU,
BoT-SORT); several teams added **GTA links** or two-stage tracking; calibration via landmark
detection then posterior calibration/homography with **NBJW, PnLCalib, BroadTrack**; **optical flow
for temporal smoothing was the novel development of this edition**; **CLIP- and OSNet-based features
emerged as a strong approach** for re-ID, role, team and jersey; VLMs (LLaMA-Vision, Qwen2-VL) were
applied successfully to jersey number recognition.

## 2.1 KIST-GSR / Broadcast2Pitch - 2025 WINNER (63.90) - FULL METHOD

- **Challenge summary:** arXiv 2508.19182, GSR-1 (pp. 7-8 and 14).
- **Full paper:** "Broadcast2Pitch: Game State Reconstruction from Unconstrained Soccer Videos",
  **WACV 2026**.
  PDF: https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
  Poster: https://wacv.thecvf.com/virtual/2026/poster/435 | IEEE: https://ieeexplore.ieee.org/document/11492373/
  (WebFetch returns 403; `curl` with a browser User-Agent succeeds.)
- **Authors:** Yin May Oo, Yewon Hwang, Muhammad Amrulloh Robbani, Vanyi Chao, Ankhzaya
  Jamsrandorj, Hoang Quoc Nguyen, Kyung-Ryoul Mun, Jinwook Kim. AI-Robotics, KIST School, UST /
  Korea Institute of Science and Technology, Seoul.
- **Follow-up:** "Broadcast2Pitch++: Depth-Aware Game State Reconstruction from Unconstrained
  Soccer Videos" (ResearchGate 405759501).
- **Code / weights: NONE released.**
- **Funding:** RS-2022-00164554 + KIST Institutional Program 2E33841.

### 2.1.1 Pipeline

Five components: (1) player detection and tracking, (2) sports field registration via keypoint and
line detection + homography estimation, (3) athlete identification, (4) identity-aware tracklet
refinement (IDATR), (5) post-processing.

**Detection:** YOLOX, trained on **SoccerNet-Tracking**.
**Tracking:** **Deep-EIoU** (expansion-IoU association).
**Re-ID:** **OSNet**, trained on **SportsMOT**.

**Sports field registration:**
- Multi-task encoder-decoder: **EfficientNetV2-S backbone with an attention-gated U-Net decoder**
  (Residual EfficientNet-Attention U-Net).
- Jointly predicts **97 pitch keypoint heatmaps** and **18 pitch line heatmaps** (each line
  represented as two Gaussian peaks at its extremities), trained as heatmap regression at half input
  resolution.
- **Homography estimation is a unified least-squares optimisation** over three residual families:
  confidence-weighted algebraic **line** residuals (sample up to N high-confidence points per line,
  threshold tau > 0.8), a **conic (centre-circle)** residual, and a **keypoint reprojection**
  fallback used when too few lines are detected.
- Initialised by **DLT** from >= 4 keypoint correspondences (identity matrix otherwise), then
  refined with **Levenberg-Marquardt**.

**Athlete identification - the load-bearing choice:**
- Identity recognition is reformulated as a **visual question answering / multimodal autoregressive
  generation** task. **LLaMA-3.2-Vision**, instruction-tuned, is prompted per cropped player box
  with a natural-language query (example prompt given in the paper: *"Classify role, jersey number,
  and jersey color"*) and jointly emits role, jersey number and jersey colour as a token sequence.
- **It is per-crop, not per-tracklet.** Tracklet-level values come from majority voting afterwards.
- Stated advantages: open-set generalisation to unseen jersey colours; no per-attribute annotation
  or architecture work; prompt-tuned multi-task inference.

**IDATR (Identity-Aware Tracklet Refinement):**
- Split/merge in the GTA family, but **splitting is driven by VLM identity predictions, not by ReID
  appearance features** - explicitly to avoid spurious splits caused by unreliable ReID under
  occlusion.
- **Split:** find frame gaps `delta_f > tau_gap`; segment there; keep adjacent segments split if an
  identity-inconsistency indicator `I(X,Y)` fires, i.e. if `Maj(role_X) != Maj(role_Y)` OR
  `Maj(jersey_X) != Maj(jersey_Y)` OR `Maj(colour_X) != Maj(colour_Y)`.
- **Merge** requires ALL of: ReID cosine distance `< tau_cos`; spatial continuity
  `||p_a_end - p_b_start||_2 < tau_spatial`; **temporal disjointness** `F_a ∩ F_b = empty`; and
  identity consistency `I(T_a,T_b) = 0`.
- Hyperparameters: **tau_gap = 25, tau_cos = 0.4, tau_spatial = 0.8**.

**Post-processing:** team left/right by clustering jersey colours and comparing cluster mean
x-positions; majority voting within each tracklet for jersey/role/team; **filtering of tracklets
shorter than 20 frames and of detections outside the pitch in pitch coordinates**.

### 2.1.2 Training compute (Broadcast2Pitch) - the only 2025 compute figure in existence

- **All experiments on a single NVIDIA RTX 4090, 24 GB, PyTorch.**
- Keypoint + line model: **25 epochs, LR 1e-4, input 384x384**, trained on the **SoccerNet-GSR train
  split**.
- **LLaMA-3.2-Vision fine-tuned for exactly ONE epoch at LR 1e-5 on the SoccerNet jersey dataset.**
- YOLOX and OSNet used off-the-shelf on SoccerNet-Tracking and SportsMOT respectively.
- This is the single most important compute fact in the survey: **a 63.90-scoring GSR system was
  trained end to end on one 24 GB consumer GPU.**

### 2.1.3 Broadcast2Pitch results (SoccerNet-GSR **test** split)

| Framework | GS-HOTA | GS-DetA | GS-AssA | IDF1 |
|---|---|---|---|---|
| GSR Baseline | 22.26 | 10.67 | 46.46 | - |
| Golovkin et al. (Constructor.Tech 2024) | 55.82 | 41.67 | 74.86 | - |
| **Broadcast2Pitch** | **61.48** | **48.47** | **78.00** | **64.20** |

Homography ablation (Table 4, detector held fixed):

| Keypoint HE | Line HE | IDATR | GS-HOTA | GS-DetA | GS-AssA | IDF1 |
|---|---|---|---|---|---|---|
| yes | no | no | 48.23 | 34.77 | 66.91 | 51.68 |
| no | yes | no | 56.39 | 42.77 | 74.40 | 61.31 |
| yes | yes | no | 58.51 | 44.63 | 76.72 | 61.57 |
| yes | yes | yes | **61.48** | 48.47 | 78.00 | 64.20 |

So: lines+circle beats keypoints+DLT+RANSAC by **+8.16 GS-HOTA**; adding the keypoint fallback
another **+2.12**; **IDATR alone is worth +2.97 GS-HOTA and +2.63 IDF1**.

### 2.1.4 Identity-model comparison inside one fixed pipeline (Table 5) - MOST TRANSFERABLE RESULT

| Identity model | GS-HOTA | GS-DetA | GS-AssA | IDF1 |
|---|---|---|---|---|
| **PRTreID (role) + EasyOCR (jersey) + ResNet-18 (colour/team)** | **18.11** | **6.54** | **50.14** | **10.62** |
| CLIP encoder + attribute-specific heads (role, jersey, team) | 60.13 | 46.88 | 77.15 | 62.20 |
| LLaMA-3.2-Vision (single backbone, all three attributes) | **61.48** | 48.47 | 78.00 | 64.20 |

Detector, tracker, ReID and calibration are identical across all three rows. The paper's own gloss:
"VLMs are significantly more effective than using separate, attribute-specific models", and
LLaMA's "modest accuracy gain over CLIP comes at higher computational cost."

### 2.1.5 GS-HOTA attribute decomposition (Table 7) - where the points actually go

Broadcast2Pitch's own system, test split, attributes enabled one at a time:

| Pitch | Role | Team | Jersey | GS-HOTA | GS-DetA | GS-AssA | IDF1 |
|---|---|---|---|---|---|---|---|
| yes | - | - | - | **79.52** | 85.97 | 73.60 | 83.40 |
| yes | yes | - | - | 79.20 | 85.18 | 73.68 | 83.14 |
| yes | - | yes | - | 73.58 | 74.46 | 72.77 | 76.97 |
| yes | - | yes* | - | 77.39 | 81.35 | 73.66 | 81.17 |
| yes | - | - | yes | **64.70** | 53.96 | 77.59 | 68.97 |
| yes | yes | yes | yes | 61.48 | 48.47 | 78.00 | 64.20 |

`*` = with three test clips (SNGS-126, 131, 197) excluded, which the authors state are **incorrectly
annotated for the team attribute in the SoccerNet-GSR test set**.

Reading: **role costs 0.32 points. Team costs 5.94 (2.13 after the annotation errors are removed).
Jersey costs 14.82.** Jersey number recognition is ~2.5x the cost of everything else combined.
The paper names it explicitly as "the primary bottleneck", and gives the mechanism: majority voting
over sparsely-observed tracklets is fragile, and residual ID switches surviving IDATR propagate
into wrong jersey numbers.

**Stated limitations:** the left/right team assumption breaks on corner kicks and sustained
high-pressing; LLaMA-3.2-Vision's compute cost blocks real-time deployment; IDATR needs
task-specific hyperparameter tuning and is not learnable (named as future work).

## 2.2 lianyou - 2025 third (62.76) - highest GS-AssA on the board (85.33)

Verbatim from arXiv 2508.19182, Appendix 7.4, GSR-3. Authors: **Youxing Lian, Tiancheng Wang**.

> "Our submission for the SoccerNet Game State Reconstruction challenge implements a four-stage
> process. We detect persons using YOLOv12 and 27 field line categories (16 points each) via a
> custom RT-DETR model. For field calibration, radial distortion correction (Wang et al.) is
> combined with PnLCalib (Gutierrez-Perez & Agudo); fixed camera parameters are simplified to
> [pan,tilt,roll,focal] and temporally smoothed. Our tracking, enhancing IOF-Tracker (Liu et al.),
> utilizes Farneback optical flow to improve Kalman filter-based motion prediction, especially
> during abrupt movements, applied to foreground objects identified via IoU and score thresholds.
> Features are extracted by CLIP-ReIdent (Habel et al., 0.5 cosine threshold). Finally,
> post-processing employs a lightweight 6-layer Transformer neural network to aggregate tracklet
> features, K-means for team assignment, and a fine-tuned ViT-L14 CLIP model (Habel et al.) for
> jersey number recognition."

Details the prior survey missed: **27 line categories at 16 points each** (432 line points);
camera model reduced to **4 DOF**; the **0.5 cosine threshold** on CLIP-ReIdent features; both the
re-ID *and* the jersey model come from the **UniBw Munich (Habel/Deuser/Oswald) CLIP lineage** -
i.e. one team runs the 2023 re-ID winner and the 2023 jersey runner-up together.
**Training data, epochs and GPUs: not stated. No code.**

Relevance: a top-3 team uses PnLCalib, the calibrator we already run. Calibration is not what
separates our 33.20 from their 62.76.

## 2.3 Playbox & MIXI - 2025 fourth (61.64) - most reproducible top-5 entry

Verbatim from arXiv 2508.19182, Appendix 7.4, GSR-4. Authors: Atom Scott, Calvin Yeung, Haruto
Nakayama, Ikuma Uchida, Masato Tonouchi, Pragyan Shrestha, Rio Watanabe, Shunzo Yamagishi, Takaya
Hashiguchi, Tomohiro Suzuki, Yuta Kamikawa (play-box.ai / Nagoya Univ. / MIXI / Tsukuba).

> "Our approach builds on the TrackLab framework, updating several modules with state-of-the-art
> techniques. For object detection, we use RF-DETR, outperforming the YOLO series. Feature
> extraction used CLIP-ReID embeddings, feeding our tracking pipeline consisting of BoT-SORT and
> Global Tracklet Association (GTA). Jersey number recognition closely follows previous work; we
> filter embeddings via Gaussian outlier rejection, geometrically extract torso regions using
> ViTPose keypoints, and recognize numbers using PARSeq OCR. Camera calibration extends BroadTrack
> by propagating calibration parameters temporally via optical flow, initialized from multiple
> stable frames. To address visual ambiguity or occlusion, we enhance role classification
> heuristically, assigning goalkeeper roles based on spatial positioning (furthest player in penalty
> areas), followed by greedy hill-climbing optimization penalizing invalid configurations, such as
> multiple goalkeepers. Finally, linear interpolation and a Savitzky-Golay filter are applied to
> ensure smoothness and handle short tracking gaps. This pipeline achieved 4th place (61.64
> GS-HOTA). Future work will leverage domain-specific knowledge to enhance overall robustness."

Details the prior survey missed:
- The **GK heuristic is two-stage**: (i) furthest player inside each penalty area, then (ii)
  **greedy hill-climbing over the whole assignment with an explicit penalty for invalid
  configurations such as two goalkeepers on one team.** It is a constrained global assignment, not
  a per-tracklet rule.
- Calibration is **BroadTrack extended with optical-flow parameter propagation, initialised from
  multiple stable frames** (not a single frame).
- Jersey is the **exact Koshkina recipe** - Gaussian outlier rejection on ReID embeddings + ViTPose
  torso crops + PARSeq.
- Every named component is public. This entry is the closest published upper bound for a stack that
  can be assembled entirely from released code.
- **Training data, epochs, GPUs: not stated. Submission code not released.**

## 2.4 eidos.ai - 2025 seventh (46.24) - VLM with a zoom tool

Verbatim (GSR-7). Authors: **Marcelo Ortega, Federico Mendez**.

> "We boost player-ID accuracy by combining a large foundational vision-language model
> (Qwen2-VL-Instruct) with function calling. Instead of classifying each detection in isolation, we
> feed the model a long sequence of detections arranged on a grid. Through chain-of-thought
> reasoning, the model can invoke a 'zoom' function to inspect specific frame ranges before choosing
> a jersey number, iterating until confident. We used this method also for the referee role
> recognition, and modified the clustering algorithm to search for a better balance between the
> number of tracks on the players clusters."

Notes: detections are presented **on a grid, as a long sequence** - i.e. the whole tracklet at once,
not per-crop. The zoom is over **frame ranges**, and the loop terminates on the model's own
confidence. This team jumped from 8.92 (2024, Siamese+RandomForest) to 46.24 (2025, VLM).
**No training at all is described - it reads as inference-only prompting.** No code.

## 2.5 UTokyo Football Lab - 2025 eighth (40.13)

Verbatim (GSR-8). Authors: Sadao Hirose, Kazuto Iwai, Ayaha Motomochi, Ririko Inoue.

> "We present an enhanced pipeline built upon the SoccerNet 2025 GSR baseline. Images in the dataset
> were deblurred using DeblurGAN-v2. We fine-tuned a YOLOv8x detector on the SoccerNet dataset for
> improved athlete detection. Detected regions were then segmented with DeepLabV3 and
> MobileNetV3-Large to isolate player silhouettes. Segmented images were used to extract a
> 10-dimensional vector encoding uniform colors and their corresponding areas. These vectors
> augmented inputs to downstream modules for re-identification, role classification, and team
> identification. We also introduced GTA-Link to refine tracklets post-baseline tracking. A Gaussian
> blur filter was applied prior to jersey number recognition. After passing through the pipeline, we
> applied linear interpolation to fill in gaps in detections and stabilize trajectories. Similarly
> to the original baseline, our method had difficulties mostly with jersey number recognition.
> Future work includes incorporating multimodal large language models for more accurate jersey
> number recognition."

This is the closest published analogue to our stack shape (baseline + fine-tuned detector +
GTA-Link + OCR): **40.13**. Note the deblur-then-Gaussian-blur sequence, and the authors' own
conclusion that jersey number recognition remained the bottleneck.

## 2.6 hjkim - 2025 ninth (36.90)

Verbatim (GSR-9). Authors: Hyungjung Kim, Joohyung Oh, Jae-Young Sim (UNIST).

> "We developed four modules to improve the performance of tracking and re-identification. First, we
> applied motion deblurring to reduce the blurring artifacts in images and improve the detection
> performance of the players and jersey numbers. Second, we used optical flow estimation to
> compensate for the camera movements and enhance the stability of tracking. Third, we fine-tuned
> the detector model on SoccerNet-GSR dataset to further improve the performance. Lastly, we used
> uniqueness property that no more than 24 individuals (22 players and 2 referees) have different
> identities for reliable identity matching. The proposed method achieves a GS-HOTA score of 36.90%
> in the challenge."

## 2.7 Laboratory for Biosignal Processing - 2025 thirteenth (28.45) - calibration only

Verbatim (GSR-13). Authors: Julian Ziegler, Patrick Frenzel, Daniel Matthes, Mirco Fuchs
(HTWK Leipzig).

> "This report presents our submission to the SoccerNet 2025 Game State Reconstruction Challenge,
> specifically addressing the camera calibration subtask. Our method refines homography estimation
> across video frames using a tracking-based approach. Reliable initialization frames are identified
> by comparing a U-Net pitch segmentation prediction with one derived from the NBJW-predicted
> homography, using their Intersection over Union (IoU) as a confidence measure. Keypoints from the
> NBJW model, along with auxiliary points sampled via geometric priors around randomly selected pitch
> locations, are projected into pitch coordinates and tracked using Lucas-Kanade optical flow. These
> tracked correspondences enable homography updates in subsequent frames via RANSAC. This iterative
> process improves temporal consistency and robustness. Evaluation using GS-HOTA and LocSim metrics
> shows improvements of up to 4% over the NBJW baseline, particularly in frames lacking prominent
> pitch features."

The IoU-between-segmentation-and-reprojected-homography trick is a **self-check on calibration
confidence with no ground truth needed**. Quantified benefit: up to 4% over NBJW.

## 2.8 Teams with NO published method

tyler_durden (57.08), SJTU Multi-Modal Perception Group (50.06), junkwang (33.62), lsmuqi (32.62),
hayden97 (32.59), johnstreet (14.26), **and Constructor.Tech (63.81)** submitted no summary in 2025.

---

# 3. The official stack: TrackLab + sn-gamestate

## 3.1 SoccerNet-GSR benchmark and metric (the task itself)

- **Paper:** "SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification
  on a Minimap", **arXiv 2404.11335** - https://arxiv.org/abs/2404.11335
  PDF: https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf
- **Authors:** Vladimir Somers, Victor Joos, Anthony Cioppa, Silvio Giancola, Seyed Abolfazl
  Ghasemzadeh, Floriane Magera, Baptiste Standaert, Amir M. Mansourian, Xin Zhou, Shohreh Kasaei,
  et al. Compute acknowledged: **CISM/UCLouvain supercomputing facilities**.
- **Dataset:** extends SoccerNet-Tracking. **200 clips x 30 s**. Splits per Broadcast2Pitch:
  **57 train / 58 valid / 49 test / 36 challenge**. Annotations: **>9.37 M line points**,
  **>2.36 M athlete pitch positions**. Pitch lines annotated in **26 classes** (including goal
  posts and crossbar), assuming a 105 x 68 m pitch.
- **Download size:** HuggingFace `SoccerNet/SN-GSR-2025`, total **35.1 GB**
  (train 9.76 GB, valid 9.76 GB, test 8.85 GB, challenge 5.31 GB).
  https://huggingface.co/datasets/SoccerNet/SN-GSR-2025
- **Metric:** `SimGS-HOTA(P,G) = LocSim(P,G) x IdSim(P,G)`;
  `LocSim = exp(ln(0.05) * ||P-G||^2 / tau^2)` with **tau = 5 metres**;
  `IdSim = 1` iff role AND team AND jersey all match, else 0.
  The alpha integral is discretised over **[0.05, 0.95] in 0.05 steps**, so pairs with similarity
  <= 0.05 are never matched, making **tau the hard maximum matching distance in metres**.
  Attributes absent from ground truth are ignored (e.g. jersey for referees); methods **must**
  suppress team/jersey for non-player roles and jersey when not visible in the video.

### 3.1.1 Baseline attribute decomposition (Table 1) - the second independent decomposition

GSR-Baseline, **test** split. "Pitch" disabled means LocSim is replaced by image-space box IoU.

| Pitch | Role | Team | Jersey | GS-HOTA |
|---|---|---|---|---|
| - | - | - | - | 57.64 (= plain HOTA) |
| yes | - | - | - | 42.65 |
| yes | yes | - | - | 40.76 |
| yes | - | yes | - | 37.03 |
| yes | - | - | yes | **25.65** |
| - | yes | yes | yes | 29.50 |
| yes | yes | yes | yes | **22.26** |

Valid split full: **18.05**. Challenge split full: **23.36**.
Ordering of difficulty stated by the authors: **jersey number > pitch localization > team
affiliation > role classification.** Same ordering as Broadcast2Pitch found two years later.

### 3.1.2 Baseline module oracle ablation (Table 2) - with speeds

Ground truth is used as an oracle for every module *except* the one named and its downstream
dependents.

| Module (+ downstream) | GS-HOTA | Batch size | FPS |
|---|---|---|---|
| Team side (left/right heuristic) | 92.00 | video | 1500 |
| ReID (PRTReID) | 87.42 | 16 | 14.5 |
| Jersey number (MMOCR) | 56.75 | 32 | 3.8 |
| Calibration (TVCalib) | 51.39 | 512 | 7.6 |
| Pitch localization (TVCalib) | 49.99 | 16 | 2.9 |
| BBox detection (YOLOv8) | 35.28 | 32 | 16.5 |
| **Full baseline** | **22.26** | - | **1.1** |

Timing hardware: **NVIDIA A100 32 GB**. **~11 minutes to process one 30-second clip** end to end.
For 49 test clips that is roughly 9 GPU-hours per full test-split evaluation with the stock
baseline.

### 3.1.3 Baseline implementation

- Detection: **YOLOv8, NOT fine-tuned on SoccerNet** ("it already provides decent performance"),
  filtered to the `person` class. Tracking is done in image space on boxes; projection to pitch
  happens later.
- Tracker: **StrongSORT**, appearance from PRTreID.
- Calibration: **TVCalib** (segmentation + iterative reprojection-error minimisation), authors'
  stock weights. Box bottom assumed on the ground plane.
- Identity: **PRTreID** (team/role/identity-aware embedding) + **MMOCR** for jersey
  (**DBNet** text detection -> **SAR** text recognition; highest-scoring numeric string wins).
- Tracklet consistency: majority voting for role and jersey.
- Team affiliation: average PRTreID embeddings per tracklet -> **K-means into 2 clusters** ->
  compare mean pitch x to label left/right.
- **The only components trained/fine-tuned on SoccerNet in the baseline are PRTreID (on the
  SoccerNet-GSR train set, original-paper hyperparameters) and TVCalib.**

## 3.2 sn-gamestate repository

- **URL:** https://github.com/SoccerNet/sn-gamestate
- **State (2026-07-29, GitHub API):** 430 stars, last push **2026-05-02**, **not archived**,
  repo size ~100 MB. **License: GPL-3.0** (copyleft).
- **Baseline modules shipped:** detection **YOLOv11**; re-ID **PRTReID** or **BPBreID**; tracking
  **StrongSORT**; jersey **MMOCR**; calibration/field localization **TVCalib**, **PnLCalib**, or
  **NBJW_Calib**.
- **Environment - CRITICAL for a cluster:** `pyproject.toml` pins
  `requires-python = ">=3.9,<3.10"`, `torch==1.13.1`, `tracklab==1.3.24`, `easyocr==1.7.1`,
  `soccernet==0.1.55`, `mmocr==1.0.1`, `mmdet~=3.1.0`, `openmim==0.3.9`, `lightning==2.0.9`,
  `numpy==1.26.4`, `transformers==4.35.2`, `tokenizers==0.15.2`, plus git installs of
  `prtreid` and `torchreid` (bpbreid) and `tracklab_calibration`.
  README's conda path installs `pytorch-cuda=11.7`.
  https://raw.githubusercontent.com/SoccerNet/sn-gamestate/main/pyproject.toml
  **Implication to verify on the cluster: torch 1.13.1 / CUDA 11.7 predates sm_90. A100 (sm_80) is
  fine; H100 / H200 / B200 class hardware is not supported by that pinned wheel.** Flagged as a
  hardware-compatibility risk, not verified by execution here.
- **Training:** README documents inference only. Weights auto-download on first run; manual download
  via the `SoccerNet` pip package. No training recipes shipped for the baseline modules.
- **Changelog:** 2026-05-01 TrackLab 1.3.24 update (**GS-HOTA fix**); 2025-05-02 UV install +
  visualisation tool; 2024-07-10 new calibration methods; 2024-05-13 dataset v1.3;
  2024-02-13 complete baseline released.
- **Submission target named in README:** the EvalAI server
  https://eval.ai/web/challenges/challenge-page/2251/overview (2024 edition; the page did not
  return content to automated fetch). Current live server is Codabench - see Section 6.

## 3.3 TrackLab framework

- **URL:** https://github.com/TrackingLaboratory/tracklab | Docs:
  https://trackinglaboratory.github.io/tracklab/
- **State (2026-07-29, GitHub API):** 243 stars, last push **2026-05-01**, **License: MIT**
  (more permissive than sn-gamestate's GPL-3.0 layered on top).
- **Ships:** detectors **YOLO, YOLOX, RTMDet, RTDETR**; pose **RTMPose, RTMO, ViTPose, YOLOPose**;
  re-ID **KPReID, BPBReID**; trackers **DeepSORT, StrongSORT, OC-SORT**.
- **Standalone requirements (README):** Python **3.12**, PyTorch **2.6**, torchvision 0.21,
  **CUDA 12.4** (with compatibility notes for other versions). Note the divergence from the
  sn-gamestate pin above.
- **Training IS supported:** the README states supervised training of the ReID model on the tracking
  training set; individual modules expose a `train` method invoked before inference. Hydra/YAML
  config; online and offline modes.
- **Datasets supported:** DanceTrack, MOTChallenge, SportsMOT, SoccerNet.
- **Companion tracker: CAMELTrack** - https://github.com/TrackingLaboratory/CAMELTrack ,
  arXiv **2505.01257**. Learned association (transformer Temporal Encoder + Group-Aware Feature
  Fusion Encoder) replacing hand-crafted matching; cue-agnostic (boxes, ReID features, pose
  keypoints). **Training takes one hour on a single consumer-grade GPU, 10 epochs, batch of 32
  detection-tracklet pairs. ~13 FPS on MOT17. Weights released.**
  **It does NOT report SoccerNet-Tracking or SoccerNet-GSR numbers** - evaluated on DanceTrack,
  SportsMOT, MOT17, PoseTrack21, BEE24 only.
- **Eval code:** https://github.com/SoccerNet/sn-trackeval - MIT, 5 stars, last push 2025-07-29.
  Contains the GS-HOTA implementation.

---

# 4. Component SOTA beyond the challenge

## 4.1 Detection

### 4.1.1 RF-DETR (used by Playbox & MIXI and named by the organisers as now-standard)

- **Repo:** https://github.com/roboflow/rf-detr - **8,783 stars, last push 2026-07-29, Apache-2.0**
  (the `rfdetr_plus` extension and RF-DETR-XL/2XL are **PML 1.0**, not Apache).
- DINOv2 vision-transformer backbone; detection, instance segmentation and keypoint (preview) in one
  API.
- Detection variants (COCO AP50 / AP50:95 / latency ms / params M / native resolution):
  **N** 67.6 / 48.4 / 2.3 / 30.5 / 384; **S** 72.1 / 53.0 / 3.5 / 32.1 / 512;
  **M** 73.6 / 54.7 / 4.4 / 33.7 / 576; **L** 75.1 / 56.5 / 6.8 / 33.9 / 704.
- **Soccer-specific fine-tune with published numbers:**
  https://huggingface.co/julianzu9612/RFDETR-Soccernet - RF-DETR-Large fine-tuned on
  **SoccerNet-Tracking 2023, 42,750 annotated images**, 4 classes (ball, player, referee,
  goalkeeper). **4 epochs, batch 4, LR 1e-4, AdamW, NVIDIA A100 40 GB, ~14 hours.**
  Results **mAP@50 85.7%, mAP@50:95 49.8%, mAP@75 52.0%**. 128 M params, 1.46 GB weights,
  Apache-2.0. Throughput 12-15 FPS on RTX 4070, 25-30 FPS on A100.
  This is the single most directly reusable detector artifact found in the survey.
- **Caveat:** these are third-party HuggingFace card claims, not a peer-reviewed result.

### 4.1.2 YOLO family on SoccerNet

- YOLOX (KIST-GSR, Broadcast2Pitch, Deep-EIoU), YOLOv8/v8x (baseline, UTokyo, Robo Space),
  YOLOv11 (current sn-gamestate baseline), YOLOv12 (lianyou), YOLOv5m (Constructor.Tech 2024).
- Deep-EIoU's YOLOX-X recipe (see 4.4.1) is the best-documented soccer detector training recipe with
  a paper behind it.
- A YOLO-version review on SoccerNet exists: https://ieeexplore.ieee.org/document/10756443/
  ("A Review of YOLO Models for Soccer-Based Object Detection") - reports YOLOv8 fine-tuned at 97% AP
  for player detection, with RT-DETR gaining AP but losing precision. Different protocol from the
  RF-DETR card above; the two numbers are not comparable.
- SoccerNet v3 detection subset in YOLO format: https://github.com/kmouts/SoccerNet_v3_H250

## 4.2 Re-identification

### 4.2.1 PRTreID (what we already run) - full training recipe

- **Paper:** "Multi-task Learning for Joint Re-identification, Team Affiliation, and Role
  Classification for Sports Visual Tracking", **arXiv 2401.09942**, MMSports'23.
  Authors: Amir M. Mansourian, Vladimir Somers, Christophe De Vleeschouwer, Shohreh Kasaei.
- **Code:** https://github.com/VlSomers/prtreid - 42 stars, last push 2025-04-11, **no license
  file** (GitHub reports NOASSERTION).
- **Architecture:** BPBReID with an **HRNet-W32** backbone, ImageNet pre-trained, initialised from
  **Market-1501** weights. **K = 5 body parts** (head, upper torso, lower torso, legs, feet) gave
  the best results.
- **Training recipe (verbatim substance):** images resized to **384 x 128**; augmentation = random
  crop with 10-px padding then random erasing at p=0.5; **trained end-to-end for 120 epochs with
  Adam on a SINGLE NVIDIA GeForce RTX 3090**; LR ramps linearly from 3.5e-5 to 3.5e-4 over the first
  10 epochs, then decays to 3.5e-5 at epoch 40 and 3.5e-6 at epoch 70.
- **Custom batch sampler, batch size 32:** each batch holds 4 identities from the left team,
  4 from the right team, and 3 from other roles, **4 images per identity, all drawn from one
  specific video**. For the team-affiliation objective only player-role identities are used
  (effective batch 24).
- Triplet margins: **0.3** for re-ID, **0.05** for team affiliation.
  Loss weights: lambda_pa=0.3, lambda_reid=1, lambda_team=0.1, lambda_role=1.5.
- **Multi-task ablation (Table 5) - the case for the role head being free:**

| # | ReID loss | Team loss | Role loss | ReID mAP | ReID R-1 | Team mAP | Team R-1 | Role Acc | Role Prec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | yes | - | - | 71.43 | 89.31 | 87.12 | 97.43 | - | - |
| 2 | - | yes | - | 12.34 | 4.28 | 91.48 | 96.68 | - | - |
| 3 | - | - | yes | 10.29 | 4.28 | 55.38 | 50.96 | 93.64 | 71.45 |
| 4 | yes | yes | - | 72.53 | 89.04 | 89.03 | 97.40 | - | - |
| 5 | yes | - | yes | 72.78 | 89.34 | 78.51 | 97.33 | 93.27 | 68.79 |
| 6 | **yes** | **yes** | **yes** | **72.59** | **89.57** | **92.89** | **97.60** | **94.27** | **74.36** |

  Role classification accuracy **94.27%** with precision **74.36%** - i.e. the model we already run
  has a role head that is accurate overall but noticeably less precise, which is the regime where a
  rare class like goalkeeper suffers. (Precision is macro across four classes; the paper does not
  break out GK.)
- **Sibling work, same author:** Keypoint Promptable Re-Identification (ECCV'24),
  https://github.com/VlSomers/keypoint_promptable_reidentification - for occlusion and multi-person
  ambiguity inside a crop.
- **Base method:** BPBReID, https://arxiv.org/abs/2211.03679

### 4.2.2 CLIP-ReIdent / CLIP-ReID - the embedding family that keeps winning

Two distinct things share similar names; both appear in the 2025 leaderboard.

**(a) CLIP-ReIdent (Habel, Deuser, Oswald - UniBw Munich)** - used by **lianyou (3rd)**.
- **Paper:** "CLIP-ReIdent: Contrastive Training for Player Re-Identification", MMSports'22,
  **arXiv 2303.11855** - https://arxiv.org/abs/2303.11855 |
  https://dl.acm.org/doi/abs/10.1145/3552437.3555698
- **Code:** https://github.com/KonradHabel/clip_reid - **MIT, 19 stars, last push 2024-01-05.**
- **Method:** reformulates CLIP's language-image contrastive objective into a **image-to-image**
  contrastive objective with **InfoNCE**. Only the CLIP **vision encoder** is kept; text encoder,
  embedding and projection layers are dropped (including the vision projection layer). Entirely
  class-agnostic.
- **Batch construction:** batch size **n = 16** with a custom sampler ensuring **only distinct
  players per batch** (a query/gallery pair of one player = one instance). Query and gallery are
  encoded in the same forward pass, doubling the effective batch. **Label smoothing 0.1** to absorb
  the case where a player appears in another player's images.
- **Training:** AdamW, polynomial LR schedule with warm-up; ViT max LR **4e-5** reached after
  2 epochs, decaying to 4e-6; CNN max LR 4e-4 -> 4e-5. **8 epochs**, best checkpoint by mAP.
  Only augmentation: random horizontal flip (the paper notes the risk to jersey-number reading and
  reports that mAP does not drop).
- **Compute:** "**GPU cluster Monacum One** at the University of the Bundeswehr Munich". **No GPU
  type, count or hours given.**
- **Results, DeepSportRadar / MMSports 2022 test split (mAP without / with re-ranking):**
  ViT-L/14 CLIP **96.9 / 98.2** (202 M params, 224 px);
  ViT-B/16 CLIP **95.0 / 98.1** (57 M params, 224 px);
  vit_large_patch16_224 ImageNet 92.5 / 97.1; convnext_large_in22ft1k 91.2 / 96.9;
  vit_base_patch16_224 89.9 / 95.4; convnext_base_in22ft1k 89.9 / 96.0; RN50x16 CLIP 88.5 / 94.9.
  **Challenge result 98.44% mAP.** CLIP-pretrained ViTs beat ImageNet-pretrained ViTs and all CNNs.
- **Key finding for jersey work:** the paper demonstrates **CLIP ViTs already have strong zero-shot
  OCR capability for shirt numbers with no fine-tuning**, verified by Score-CAM saliency on 100
  hand-annotated images across attributes {jersey number, jersey colour, sex, skin colour}. This is
  the ancestor of the 2023 jersey result (Section 4.3.2).
- **Split caveat:** 98.44 is **basketball (DeepSportRadar)**, not soccer. The same team's soccer
  number is the SoccerNet 2023 re-ID challenge: **mAP 93.26, R-1 91.26**.

**(b) CLIP-ReID (Li, Sun, Li - AAAI 2023)** - used by **Playbox & MIXI (4th)**.
- https://ojs.aaai.org/index.php/AAAI/article/view/25225 |
  **Code:** https://github.com/Syliz517/CLIP-ReID - **MIT, 512 stars, last push 2023-11-21.**
  Packaged form: https://github.com/therapml/clipreid
- Two-stage prompt-learning re-ID without concrete text labels. Checkpoints in the repo are named
  `ViT-B-16_60.pth`, implying 60 epochs. Person benchmarks (Market-1501, MSMT17, DukeMTMC,
  Occluded-Duke) and vehicle (VeRi-776, VehicleID). **No soccer-specific numbers published by the
  authors.**

### 4.2.3 Per-match self-supervised ReID fine-tuning (the Kalisteo / Maglo lineage)

- **Kalisteo (CEA), SoccerNet 2023 Tracking winner, HOTA 75.61 / DetA 75.38 / AssA 75.94**
  (arXiv 2309.06006 Table 6). Method: YOLO-X detection; **TrackMerger v2** (two successive Hungarian
  assignments on IoU and centre distance); Kalman with camera-motion compensation; **tracklets are
  split wherever they cross**, and the resulting non-ambiguous tracklets are used to fine-tune a
  **Multiple Granularity Network re-ID model with triplet loss** - positives from the same tracklet,
  negatives from concomitant tracklets - after which tracklets are iteratively merged by re-ID
  distance with anti-duplication and anti-teleportation constraints.
- **This is label-free per-match embedding adaptation.** No annotation is required; the video's own
  co-occurrence structure supplies the supervision. Directly relevant to any appearance ceiling
  measured on a fixed global embedding.
- Related: Maglo et al., cited in the GSR baseline paper's related work as introducing "test-time
  fine-tuning" for robust player tracking plus a football field registration technique, but
  **without identification or quantitative localization evaluation**.
- **ICOST (2023, HOTA 65.67)** contributed the complementary hard constraint:
  **co-occurrence-aware hierarchical clustering that forbids merging two tracklets which have
  co-occurrent detections.** Any connector (ours included) needs this to be sound.
- **sn-reid devkit:** https://github.com/SoccerNet/sn-reid - MIT, **stale since 2023-07-07**.
  SoccerNet-v3 ReID = 340,993 thumbnails from 400 games; train 12.12 GB / valid 2.35 GB /
  test 2.41 GB / challenge 1.75 GB. **Task is same-action cross-viewpoint matching, not long-term
  identity** - the 93.26 mAP headline does not transfer to a long-horizon naming setting.

## 4.3 Jersey number recognition

### 4.3.1 Grad, CVPRW 2025 - CURRENT PUBLISHED SOTA on the SoccerNet Challenge split (85.62)

- **Paper:** "Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer", CVPRW 2025
  (CVSports), pp. 6101-6108.
  https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.html
  PDF: https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf
- **Author:** Lukasz Grad (University of Warsaw / ReSpo.Vision, Poland). **No arXiv mirror found.
  No code repository found. No weights released.** Funded by Polish NCBR POIR.01.01.01-00-0718/21.
- **Shape:** single-stage. A crop goes in, a jersey number and a calibrated uncertainty come out.
  **No jersey detection, no text detection, no OCR, no pose model.**
- **Preprocessing:** crop the torso by **removing the top 1/6 and bottom 1/3 of the bounding box**,
  resize to a fixed **224x224**. (Compare: Koshkina uses ViTPose keypoints for the same purpose.)
- **Backbone:** ViT CLS token as global embedding. Ablations on **ViT-S/16** (ImageNet-21k,
  224x224, dim 384); final benchmarks on **ViT-B/8**.
- **Jersey number head - three variants:**
  - IND (independent): `s = Wx + b`, W in R^(101 x D) -> 101D + 101 params.
  - DA (digit-aware): separate linear layers for singles, tens, ones, plus an absent scalar;
    two-digit score `s_{10t+o} = s_tens,t + s_ones,o` -> 31D + 31 params.
  - **TDA (tied digit-aware, the proposal):** one shared weight matrix across all three positions
    with **learnable position embeddings** combined **additively (TDA-A)** or **multiplicatively
    (TDA-M)**: `x_p = x + e_p` or `x_p = x * e_p`, then `s_p = W x_p + b_p`.
    **14D + 31 params (per-digit bias) or 14D + 4 (per-position bias) - an 86% parameter reduction
    vs IND.**
- **Uncertainty - two variants:**
  - Softmax: an explicit **"absent" class at index 100**, cross-entropy with label smoothing
    epsilon = 1e-2 over N = 101 classes. Absent-class probability *is* the uncertainty measure.
  - **Dirichlet (evidential, the proposal):** head outputs evidence, `alpha_j = exp(s_j) + 1`,
    total evidence `S = sum alpha`, `p_j = alpha_j / S`, **uncertainty = 100 / S**. Trained with
    Type-II maximum-likelihood loss; samples with no visible number get a **KL regulariser toward a
    uniform Dirichlet**; total loss `L_Dir = L_ML + lambda_KL * L_KL`.
- **Tracklet aggregation - directly addresses silent failure:**
  - `tau_invisible`: if the **minimum** uncertainty across all frames in a tracklet exceeds this,
    the whole tracklet is declared "no visible jersey number".
  - `tau_uncertainty`: only frames **below** this contribute to the tracklet prediction.
  - Remaining frames combine as `p_tracklet = (sum_{i in F} p_i + alpha * p_prior) / (|F| + alpha)`
    with a uniform prior and Bayesian strength parameter alpha; the argmax is the answer.
- **Datasets (Table 1):** SoccerNet Train **1,427 tracklets / 733,000 images**;
  SoccerNet Test **1,211 / 564,500**; SoccerNet Challenge **1,426 / 748,600**;
  **SoccerNet ReID 350,986 images** (pseudolabelled); proprietary **200M: 594,977 tracklets /
  1,071,564 images** from >200 matches (professional, youth, women's), tracklet-level weak labels
  QA'd to **99.5% accuracy**; **Copa America 2024: 33,730 / 89,728** from 12 matches (held-out
  generalisation test).
- **Pseudolabelling recipe (why the public-data model works):** run the best proprietary model over
  the SoccerNet ReID dataset; keep low-uncertainty crops with their predicted number, label
  high-uncertainty crops **"absent"**, and **discard the intermediate band entirely**. Yields
  350,986 usable images from unlabelled data.
- **Data hygiene noted:** tracklets with average diagonal **< 40 px are excluded from training** (or
  assigned label 1 at inference) because small boxes are ball detections mislabelled as players.
- **Results (Table 6), tracklet accuracy:**

| Method | SoccerNet Test | SoccerNet Challenge |
|---|---|---|
| Gerke et al. | 32.57 | 35.79 |
| Vats et al. (2021) | 46.73 | 49.88 |
| Li et al. | 47.85 | 50.60 |
| Vats et al. (2022) | 52.91 | 58.45 |
| Balaji et al. | 68.53 | 73.77 |
| **Koshkina et al. (our current OCR lineage)** | **87.45** | **79.31** |
| Grad, ViT-S, SoccerNet public data only | 82.74 | - |
| **Grad, ViT-B, SoccerNet public data only** | **86.37** | **83.52** |
| Grad, ViT-B, 200M proprietary | 85.46 | **85.62** |
| Grad, ViT-B, 200M + 1 epoch finetune on SoccerNet train | **88.27** | 85.41 |

  **The public-data-only ViT-B beats Koshkina by +4.21 on the Challenge split** while losing 1.08 on
  Test - i.e. it generalises better, which is what matters off-benchmark.
- **Ablations:**
  - Uncertainty modelling (Table 4, SoccerNet Test, softmax -> Dirichlet delta):
    DA 80.15 -> **83.24 (+3.09)**; TDA-M 78.70 -> 82.91 (+4.21); TDA-MB 77.81 -> 82.11 (+4.30);
    TDA-A 75.81 -> 73.03 (-2.78); IND 73.47 -> 78.48 (+5.01).
    Broken out, **invisible-jersey accuracy** rises across the board, e.g. DA 71.55 -> 77.00,
    IND 70.70 -> 77.09. The paper's conclusion: **"the improvement in invisible jersey detection is
    particularly substantial across all head types."**
  - Held-out-number experiment (Table 3, balanced accuracy on 3 numbers never seen in training):
    TDA-M **77.93 softmax / 85.86 Dirichlet**; TDA-MB 74.38 / 84.11; TDA-A 69.21 / 68.00;
    DA 60.84 / 65.68; **IND 35.59 / 22.49**. Compositional heads generalise to unseen numbers;
    the independent-class head collapses.
  - Copa America (Table 5): all heads land at 25-28% - **unseen kit designs are far harder than the
    benchmark suggests**, and Dirichlet helps everywhere (+0.99 to +2.31).
- Experiments repeated with **three random seeds**; means and standard deviations reported.

### 4.3.2 UniBw Munich VIS - the CLIP-gating approach (90.95 in the 2023 challenge)

- Konrad Habel, Fabian Deuser, Norbert Oswald. Documented **only inside arXiv 2309.06006**
  (SoccerNet 2023 challenge report); Springer version:
  https://link.springer.com/article/10.1007/s12283-024-00466-4
- **Did not train on the challenge labels at all**, citing high label noise. Instead used
  **OpenAI ViT-L/14 CLIP zero-shot to auto-label the larger, more diverse SoccerNet Re-Identification
  dataset**, then fine-tuned the image encoders of two CLIP models on those pseudolabels.
- Tracklet prediction: **majority vote over only those images with classification probability >
  70%**, over classes 1-99.
- Results: Test **90.09**, Challenge **90.95**. Winner ZZPM scored 92.85.
- **No code released for the jersey model** (KonradHabel/clip_reid covers re-ID only). lianyou used
  "a fine-tuned ViT-L14 CLIP model (Habel et al.)" in 2025 - implying either a private
  reimplementation or a private weight transfer.
- **Split caveat:** the 2023 jersey challenge scored accuracy over tracklets **including correct
  prediction of -1 ("no number visible")**, so 90.95 and Grad's 85.62 are on the same task but
  different years' challenge sets.

### 4.3.3 Koshkina & Elder (our lineage)

- **Paper:** arXiv **2405.13896**, CVPRW 2024 pp. 3235-3244.
  https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf
- **Code:** https://github.com/mkoshkina/jersey-number-pipeline - **65 stars, last push
  2024-10-07, license = Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0).**
- **Two documented paths.** Image-level: legibility classifier -> scene text recognition.
  Tracklet-level: **occlusion/outlier removal (re-ID features + Gaussian fitting) -> legibility
  classifier -> pose-guided RoI cropping -> STR -> tracklet consolidation.**
- **Training IS supported and documented** for two components:
  - Legibility classifier: `python3 legibility_classifier.py --train --arch resnet34 --sam --data
    <dataset> --trained_model_path ./experiments/hockey_legibility.pth`, and a `--finetune` variant
    for SoccerNet starting from the hockey checkpoint.
  - PARSeq STR: `python3 main.py SoccerNet train --train_str` (and `Hockey` equivalent).
- **Weights provided (Google Drive + OneDrive):** original PARSeq, hockey-fine-tuned PARSeq,
  **SoccerNet-fine-tuned PARSeq**, legibility classifiers for both domains, Centroid-ReID weights,
  ViTPose checkpoints.
- **GPU requirements and training time: not specified in the repo.** No accuracy numbers in the
  README either.
- Accuracy from the paper: **91.4% hockey, 87.4% SoccerNet** - the latter being the **Test** split
  (Challenge = 79.31, per Grad Table 6).
- **The pipeline already contains a legibility classifier** - a built-in abstention stage.
- Related: "Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast
  Videos", MMSports'23, https://dl.acm.org/doi/abs/10.1145/3606038.3616162 (Balaji et al., the
  68.53 / 73.77 row above).

### 4.3.4 Orientation-guided multi-task jersey recognition

- "Generalized Jersey Number Recognition Using Multi-task Learning With Orientation-guided Weight
  Refinement", **arXiv 2406.01033** - https://arxiv.org/abs/2406.01033 . CC BY-SA 4.0.
- **Angle-Digit Refine Scheme (ADRS):** combines **human body orientation angles** with digit clues,
  multi-task. Built on a cross-sport dataset (soccer, football, basketball, volleyball, baseball).
- Reported: **Top-1 64.07%, Top-2 89.97%; F1 67.46 / 90.64.** These are cross-sport numbers, **not
  SoccerNet-comparable**, and are far below the SoccerNet-specific results above.
- **No GPU/epoch details, no code found.**
- Consistent with two other orientation findings: **AIBrain (2023, 75.18)** ranked tracklet
  confidence by pose-derived body orientation plus image-quality assessment with ESRGAN upscaling;
  **Constructor.Tech (2024)** used a 4-class orientation model as a tracking constraint.

### 4.3.5 PARSeq (the STR model underneath Koshkina and Playbox & MIXI)

- https://github.com/baudm/parseq - **728 stars, last push 2024-05-29, Apache-2.0.** Scene text
  recognition with permuted autoregressive sequence models (ECCV 2022).

### 4.3.6 sn-jersey devkit

- https://github.com/SoccerNet/sn-jersey - **stale since 2024-07-02**, no license file.
- 2,853 tracklets total; challenge set 1,211 tracklets, hidden annotations. Ground truth is a JSON
  dict {player id (str) -> jersey number (int)} with **-1 meaning "no number visible"**, scored as a
  first-class class alongside 1-99. Download:
  `mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])`. Eval on EvalAI.

## 4.4 Tracking and tracklet association

### 4.4.1 Deep-EIoU (tracker in the 2025 winner) - full training recipe

- **Paper:** arXiv **2306.13074**, WACV 2024 RWS Workshop.
  PDF: https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Huang_Iterative_Scale-Up_ExpansionIoU_and_Deep_Features_Association_for_Multi-Object_Tracking_WACVW_2024_paper.pdf
- **Code:** https://github.com/hsiangwei0903/Deep-EIoU - **72 stars, last push 2024-08-22,
  NO LICENSE FILE.**
- **Design:** abandons the Kalman filter entirely; iterative scale-up ExpansionIoU + deep features.
  Premise: sports motion is irregular, so constant-velocity priors hurt.
- **Detector:** **YOLOX-X**, COCO-pretrained, fine-tuned on **SportsMOT train+val for 80 epochs**,
  input **1440x800**, Mosaic + Mixup, SGD, weight decay 5e-4, momentum 0.9, initial LR 1e-3, 1-epoch
  warmup, cosine annealing (ByteTrack's schedule).
- **ReID:** OSNet, Market-1501-pretrained, fine-tuned **60 epochs, Adam, cross-entropy, LR 3e-4**.
  SportsMOT ReID data: 31,279 train / 133 query / 1,025 gallery images.
  **SoccerNet-Tracking ReID data: 100 GT boxes per player from randomly sampled videos, split
  65 train / 10 query / 25 gallery => 7,085 train / 1,090 query / 2,725 gallery images across
  109 identities.** That is a very small ReID training set.
- **Tracking thresholds:** high-score detections > 0.6; 0.6-0.1 treated as low score; < 0.1
  discarded. Cost filters **tau_A = 0.25, tau_EIoU = 0.5**. The bounding-box aspect-ratio constraint
  is **removed** because players lie on the ground in sports footage.
- **Compute: a single NVIDIA RTX 4080 for everything. Runs at ~14.6 FPS.**
- **Results:** SportsMOT **HOTA 77.2 / IDF1 79.8 / AssA 67.7**; **SoccerNet-Tracking HOTA 85.443 /
  DetA 99.236 / AssA 73.567.**
- **CRITICAL CAVEAT:** on SoccerNet-Tracking, **all evaluated methods use the oracle (ground-truth)
  detections provided by the dataset** - that is why DetA is 99.236. The 85.4 HOTA is an
  association-only number and does **not** transfer to a detect-then-track pipeline.

### 4.4.2 GTA / gta-link (what we already run) - measured headroom

- **Paper:** arXiv **2411.08216**, ACCV 2024 MLCSA workshop.
  PDF: https://openaccess.thecvf.com/content/ACCV2024W/MLCSA2024/papers/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.pdf
  OpenReview: https://openreview.net/forum?id=hsi47B381b
- **Code:** https://github.com/sjc042/gta-link - **MIT, 90 stars, last push 2025-12-12** (actively
  maintained, unlike most repos in this survey).
- **ReID model used in their experiments: OSNet trained on SportsMOT.** SoccerNet experiments use
  **oracle detections**, same caveat as above.
- **SoccerNet-Tracking 2023 test results (Table 2):**

| Method | HOTA | AssA | IDF1 | DetA | MOTA | IDs |
|---|---|---|---|---|---|---|
| SORT | 65.89 | 57.15 | 68.56 | 76.11 | 82.59 | 3281 |
| SORT + GTA | **72.73 (+6.84)** | 69.62 (+12.47) | 81.24 (+12.68) | 76.04 | 82.93 | 1374 (-1907) |
| ByteTrack | 67.30 | 60.38 | 73.22 | 75.14 | 84.66 | 4558 |
| ByteTrack + GTA | **71.97 (+4.67)** | 69.03 (+8.65) | 82.67 (+9.45) | 75.10 | 84.91 | 3149 (-1409) |
| Deep-EIoU | 79.41 | 71.55 | 78.40 | 88.14 | 87.92 | 2803 |
| Deep-EIoU + GTA | **83.11 (+3.70)** | 78.38 (+6.83) | 84.66 (+6.26) | 88.13 | 88.03 | 2188 (-615) |

- **Splitter vs Connector ablation on SoccerNet (Table 3):**
  SORT: connector only 71.74 (+5.85) -> both 72.73 (+6.84).
  ByteTrack: connector only 71.05 (+3.75) -> both 71.97 (+4.67).
  Deep-EIoU: connector only 82.01 (+2.60) -> both 83.11 (+3.70).
  **The connector supplies roughly 70-85% of the gain; the splitter adds the remainder.**
- **DetA is essentially unchanged in every row** - GTA is purely an association-side intervention.
- Hyperparameters: minimum cluster samples s = 5 (full list in the paper).

### 4.4.3 GTATrack (2026) - Deep-EIoU + GTA, with code

- arXiv **2602.00484** (submitted 2026-01-31); **code https://github.com/ron941/GTATrack-STC2025**.
- 1st place SoccerTrack Challenge 2025. HOTA 0.60; false positives reduced to 982.
- Hierarchical two-stage: Deep-EIoU online association, GTA trajectory-level refinement. The same
  composition the GSR 2025 winner used, but released.

### 4.4.4 Other trackers named in challenge reports

- **BoT-SORT:** arXiv 2206.14651 - used by Playbox & MIXI and named by the 2025 organisers as one of
  the two dominant trackers.
- **StrongSORT:** the sn-gamestate baseline tracker.
- **CO-MOT / MOTRv2 end-to-end:** https://github.com/BingfengYan/CO-MOT (69.5 HOTA claimed on
  SoccerNet 2023).
- **MOT4MOT (Amazon, 2023, HOTA 66.27):** DeepOCSORT + fine-tuned YOLOv8, appearance model
  fine-tuned on a curated subset, interpolation + **appearance-free track merging**.
  Tech report https://arxiv.org/abs/2308.16651

## 4.5 Camera calibration

### 4.5.1 PnLCalib / No Bells Just Whistles - one paper, two names

- **arXiv 2404.08401.** v1 = "No Bells, Just Whistles: Sports Field Registration by Leveraging
  Geometric Properties" (CVPRW 2024, pp. 3325-3334, https://ieeexplore.ieee.org/document/10678440/).
  v4 = "PnLCalib: Sports Field Registration via Points and Lines Optimization"
  (CVIU vol. 267, April 2026, https://www.sciencedirect.com/science/article/pii/S1077314226000792).
  SSRN 4998149. IRI: https://www.iri.upc.edu/publications/show/3100
- **Authors:** Marc Gutierrez-Perez, Antonio Agudo (Institut de Robotica i Informatica Industrial).
  **Gutierrez-Perez is also a GSR 2025 challenge task lead.**
- **Code:** https://github.com/mguti97/PnLCalib - **101 stars, last push 2026-03-17, GPL-2.0**
  (copyleft - matters for vendoring).
  Predecessor: https://github.com/mguti97/No-Bells-Just-Whistles - 59 stars, 2024-10-25, no license.
- **Architecture:** keypoint model and line model both **HRNetv2-w48** encoder-decoder producing
  half-resolution heatmaps. Keypoints are organised in **five hierarchical categories** (line-line
  intersections, extended intersections, line-ellipse intersections, ellipse tangent points,
  additional grid points). The line model outputs heatmaps with **two Gaussian peaks at each line's
  extremities** using sigmoid activation with a boundary channel.
- **Training:** SoccerNet-Calibration train split, **22,816 images**. Two variants:
  **multi-view (MV) 200 epochs, LR 1e-2, batch 2**; **single-view (SV) 200 epochs, LR 1e-5,
  batch 1**. Adam (0.9, 0.999), L2 heatmap regression loss. Augmentation: random horizontal flip,
  colour jitter, Gaussian noise. Fine-tuned separately on WC14 and TS-WorldCup.
- **Compute: a single NVIDIA RTX 2080 Ti (12 GB).** Inference: base MV model **7 Hz**, with PnL
  refinement **~4 Hz** (Intel Xeon Silver 4214).
- **SoccerNet-Calibration results (Table I):**
  Ours SV 75.8 / 89.7 / 91.9 (JaC@5/10/20), CR 98.1, FS 74.4;
  **Ours SV + PnL 80.6 / 89.9 / 92.4, CR 97.7, FS 78.7**;
  TVCalib 54.8 / 78.5 / 90.4, CR 100, FS 54.8; prior method [32] 63.9 / 80.7 / 86.3, FS 63.9.
- **WorldCup 2014 (Table II):** Ours SV 77.6 / 89.8 / 93.7; **Ours SV + PnL 85.2 / 94.0 / 96.1**.
  Homography (Table III): Ours SV + PnL **IoU_part 97.1, IoU_whole 93.2, proj. error 0.58 m,
  reproj. error 0.014**.
- **TS-WorldCup:** Ours SV + PnL IoU_part 97.3, IoU_whole 92.8, **proj. error 0.71 m**.
- **Successor from the same author:** **SoccerNet-v3D** (CVPR CVSPORTS 2025),
  https://github.com/mguti97/SoccerNet-v3D - **GPL-2.0, 46 stars, last push 2025-06-18.**
  Uses the PnLCalib pipeline to recover camera parameters, adds single-image 3D ball localization
  from camera calibration plus ball-size priors.

### 4.5.2 Calibration head-to-head on the SoccerNet-GSR **test** split (Broadcast2Pitch Table 3)

This is the only table that compares calibrators on the GSR data itself rather than on
SoccerNet-Calibration.

| Method | JaC@5 | JaC@10 | MRE (px) | MedRE (px) | Completeness Rate |
|---|---|---|---|---|---|
| TVCalib | 19.88 | 50.42 | 12.40 | - | 99.93 |
| NBJW | 37.14 | 68.24 | 10.28 | - | 93.67 |
| BroadTrack | 56.88 | 79.79 | 5.02 | 2.37 | 100 |
| Oo et al. | 40.44 | 68.34 | 9.36 | 3.70 | 99.98 |
| **Broadcast2Pitch** | **69.38** | **92.84** | **4.47** | **2.21** | **100** |

Note NBJW's **completeness rate of only 93.67%** - it fails to produce a homography on ~6% of GSR
frames, whereas the optimisation-based methods reach 100%.

### 4.5.3 BroadTrack (used by Playbox & MIXI)

- **Paper:** "BroadTrack: Broadcast Camera Tracking for Soccer", WACV 2025, **arXiv 2412.01721**.
  PDF: https://openaccess.thecvf.com/content/WACV2025/papers/Magera_BroadTrack_Broadcast_Camera_Tracking_for_Soccer_WACV_2025_paper.pdf
- **Authors:** Floriane Magera, Thomas Hoyoux, Olivier Barnich, Marc Van Droogenbroeck (EVS).
  Magera is also a GSR challenge task lead.
- **Code:** https://github.com/evs-broadcast/BroadTrack - **15 stars, last push 2025-02-17,
  no license file (NOASSERTION).**
- **Method:** combines open-source field detectors with explicit **camera and tripod models**
  including lens distortion; tracks calibration parameters through time rather than treating each
  frame independently. Demonstrated on a **20-minute** broadcast clip, not just 30-second cuts.
- **Claims:** halves mean reprojection error vs SOTA; **>15% improvement in Jaccard index**.
  See 4.5.2 for its measured GSR-test numbers.

---

# 5. Training recipes summary table

Everything with a published training recipe, assembled for cluster planning. "-" = not stated
anywhere public.

| Component | Model | Training data | Epochs / steps | Batch | LR | Input | GPU | Time | Source |
|---|---|---|---|---|---|---|---|---|---|
| **Constructor.Tech 2024 (63.81)** | | | | | | | **1x A100 40 GB (all models)** | - | arXiv 2504.06357 |
| detection | YOLOv5m (35.5 M) | 66k imgs, 2 cls | 100 ep | - | 0.01 Adam | 1920x1080 | A100 40 GB | - | " |
| field keypoints | ResNet18 (11 M) | 36k imgs, 74 kpts | 150 ep | - | 0.01 AdamW OneCycle | 480x270 | A100 40 GB | - | " |
| camera params | custom SegFormer (5 M) | 22k real + 40k synth | 400k batches (200k mixed + 200k real) | 8 | 5e-4 AdamW cosine, 20k warmup | 512x288 | A100 40 GB | - | " |
| TeamID | OSNet (0.3 M) | 550k imgs, 111 cls | 40 ep | - | 1e-3 Adam, step g=0.1 | 64x32 | A100 40 GB | - | " |
| ReID | ResNet50 (25.6 M) | 280k crops / 370 IDs | - | 28 cls x 7 | 1e-3 SGD, triplet | 256x128 | A100 40 GB | - | " |
| orientation | ResNet18 (11 M) | 20k imgs, 4 cls | - | - | 1e-4 SGD, CE | 62x32 | A100 40 GB | - | " |
| anomaly | ResNet18 (11 M) | 16k imgs, 2 cls | - | - | 1e-3 Adam | 62x32 | A100 40 GB | - | " |
| jersey | ResNet18 2-head (17 M) | 70k imgs, 100 cls | - | - | 1e-4 AdamW, plateau | 32x32 | A100 40 GB | - | " |
| **Broadcast2Pitch / KIST-GSR (63.90)** | | | | | | | **1x RTX 4090 24 GB (all)** | - | WACV 2026 |
| keypoints+lines | EfficientNetV2-S + attn U-Net | SoccerNet-GSR train (57 clips) | **25 ep** | - | 1e-4 | 384x384 | RTX 4090 | - | " |
| identity | **LLaMA-3.2-Vision fine-tune** | SoccerNet jersey dataset | **1 epoch** | - | **1e-5** | crops | RTX 4090 | - | " |
| detection | YOLOX | SoccerNet-Tracking | off-the-shelf | - | - | - | - | - | " |
| ReID | OSNet | SportsMOT | off-the-shelf | - | - | - | - | - | " |
| **Components (standalone papers)** | | | | | | | | | |
| PRTreID | BPBReID + HRNet-W32, K=5 | SoccerNet(-GSR) tracking, Market-1501 init | **120 ep** | 32 (4+4+3 IDs x 4 imgs, single video) | 3.5e-5 -> 3.5e-4 -> 3.5e-6 @40/70 | 384x128 | **1x RTX 3090** | - | arXiv 2401.09942 |
| Deep-EIoU detector | YOLOX-X, COCO init | SportsMOT train+val | **80 ep** | - | 1e-3 SGD, cosine, 1ep warmup | 1440x800 | **1x RTX 4080** | - | arXiv 2306.13074 |
| Deep-EIoU ReID | OSNet, Market-1501 init | 7,085 imgs / 109 IDs (SoccerNet) or 31,279 (SportsMOT) | **60 ep** | - | 3e-4 Adam, CE | - | 1x RTX 4080 | - | " |
| CLIP-ReIdent | CLIP ViT-L/14 vision enc. (202 M) | DeepSportRadar ReID | **8 ep** | **16** (custom sampler) | 4e-5 max AdamW poly, warmup 2ep | 224x224 | **"Monacum One" cluster (type/count not stated)** | - | arXiv 2303.11855 |
| PnLCalib keypoint+line | HRNetv2-w48 x2 | SoccerNet-Calib train, **22,816 imgs** | **200 ep** | 2 (MV) / 1 (SV) | 1e-2 (MV) / 1e-5 (SV), Adam | - | **1x RTX 2080 Ti 12 GB** | - | arXiv 2404.08401 |
| Grad jersey | ViT-B/8 (ImageNet-21k init) | SoccerNet train 733k imgs + ReID 350,986 pseudolabelled; or 200M 1.07 M imgs | (schedule in supplementary only) | - | - | 224x224 | **not stated** | - | CVPRW 2025 |
| RF-DETR soccer | RF-DETR-Large (128 M) | SoccerNet-Tracking 2023, **42,750 imgs**, 4 cls | **4 ep** | **4** | 1e-4 AdamW | 704x704 native | **1x A100 40 GB** | **~14 h** | HF julianzu9612 |
| CAMELTrack | transformer TE + GAFFE | DanceTrack/SportsMOT/MOT17/PoseTrack21/BEE24 | **10 ep** | 32 det-tracklet pairs | - | - | **1x consumer GPU** | **~1 h** | arXiv 2505.01257 |
| GSR baseline (PRTreID + TVCalib only) | - | SoccerNet-GSR train | per original papers | - | - | - | CISM/UCLouvain cluster | - | arXiv 2404.11335 |

**Inference-cost anchors (for planning eval runs, not training):**
- GSR baseline: **1.1 FPS end to end, ~11 min per 30 s clip, A100 32 GB** => ~9 GPU-h for the
  49-clip test split, ~6.6 GPU-h for the 36-clip challenge split.
- Per-module baseline speeds: TVCalib pitch 2.9 FPS, MMOCR jersey 3.8 FPS, TVCalib calib 7.6 FPS,
  PRTreID 14.5 FPS, YOLOv8 16.5 FPS.
- Deep-EIoU full pipeline: ~14.6 FPS on RTX 4080.
- PnLCalib: 7 Hz base, 4 Hz with PnL refinement.
- Constructor.Tech deployed: up to 80 FPS on an RTX 3080Ti laptop with TensorRT FP16.
- RF-DETR-Large: 25-30 FPS on A100, 12-15 FPS on RTX 4070.

**Dataset footprints:** SoccerNet-GSR **35.1 GB** (train 9.76 / valid 9.76 / test 8.85 /
challenge 5.31). SoccerNet-v3 ReID **18.6 GB** (12.12 / 2.35 / 2.41 / 1.75).

**Headline compute observation:** the two highest-scoring published GSR systems were trained on
**one A100 40 GB** and **one RTX 4090 24 GB** respectively. Nothing in the GSR literature reports
multi-GPU or multi-node training. The cluster advantage is in *parallel experiments and full-split
evaluations*, not in any single model needing large-scale distributed training.

---

# 6. Evaluation server status (2026-07-29)

| Competition | Codabench ID | Phase status | Start | End | Participants | Submissions |
|---|---|---|---|---|---|---|
| 2025 SoccerNet GSR - **Test Phase** | **4365** | **`current` - ACTIVE** | 2024-08-31 | **none (open-ended)** | 88 | 61 |
| 2025 SoccerNet GSR - Challenge Phase | 4469 | `previous` - CLOSED | 2024-08-31 | **2025-05-07** | 97 | 66 |

- https://www.codabench.org/competitions/4365/ (API: /api/competitions/4365/)
- https://www.codabench.org/competitions/4469/ (API: /api/competitions/4469/)
- Both created by user `fmagera` (Floriane Magera). Both `published: true`.

**Answers to the posed questions:**
1. **Is GSR codabench still accepting submissions in 2026?** The **Test Phase (4365) yes** - status
   `current`, no end date, `auto_run_submissions: true`, `registration_auto_approve: true`.
   The **Challenge Phase (4469) no** - it ended 2025-05-07 and is marked `previous`.
2. **What splits are scored?** 4365 scores the **test** split; 4469 scored the **challenge** split
   (hidden labels). Leaderboard columns for both: **GS-HOTA (primary, descending), GS-DetA,
   GS-AssA**.
3. **Can a non-challenge submission still be leaderboard-ranked?** On the test-phase leaderboard,
   yes, subject to the stated limits: **max 1 submission per day, max 10 submissions per person**,
   600 s execution time limit per submission.
4. Caveat on comparability: the 4365 test-phase leaderboard is **not** the table published in
   arXiv 2508.19182 - that table is the challenge split. Baseline reference points differ by split:
   test 22.26 (2024 paper), valid 18.05, challenge 23.36 (2024) / 29.01 (2025 repo baseline).
   **Any GS-HOTA number must state its split.**
5. GSR is **not** a SoccerNet 2026 task (2026 = Ball Action Anticipation, Player-Centric Ball Action
   Spotting, Novel View Synthesis, Spiideo SoccerNet Synloc, VQA). The GSR test server outliving the
   prize challenge is what makes continued benchmarking possible.
6. The **sn-gamestate README still points at EvalAI challenge 2251**
   (https://eval.ai/web/challenges/challenge-page/2251/overview) - that is the 2024 server with a
   2024-05-30 deadline in the README text. The page returned no content to automated fetch; treat
   the README's submission instructions as stale relative to Codabench.
7. **GS-HOTA evaluation was broken in sn-gamestate before the TrackLab 1.3.24 fix committed
   2026-05-01/02.** Any number computed against an older pairing is suspect.

---

# 7. Repository state snapshot (GitHub API, 2026-07-29)

| Repo | Stars | Last push | License | Note |
|---|---|---|---|---|
| roboflow/rf-detr | 8783 | 2026-07-29 | Apache-2.0 | XL/2XL under PML 1.0 |
| baudm/parseq | 728 | 2024-05-29 | Apache-2.0 | STR model |
| Syliz517/CLIP-ReID | 512 | 2023-11-21 | MIT | AAAI'23 CLIP-ReID |
| SoccerNet/sn-gamestate | 430 | 2026-05-02 | **GPL-3.0** | GSR baseline |
| TrackingLaboratory/tracklab | 243 | 2026-05-01 | MIT | framework |
| mguti97/PnLCalib | 101 | 2026-03-17 | **GPL-2.0** | our calibrator |
| sjc042/gta-link | 90 | 2025-12-12 | MIT | our connector, actively maintained |
| hsiangwei0903/Deep-EIoU | 72 | 2024-08-22 | **none** | winner's tracker |
| mkoshkina/jersey-number-pipeline | 65 | 2024-10-07 | **CC BY-NC 3.0** | our OCR, non-commercial |
| mguti97/No-Bells-Just-Whistles | 59 | 2024-10-25 | **none** | NBJW keypoints |
| mguti97/SoccerNet-v3D | 46 | 2025-06-18 | GPL-2.0 | PnLCalib successor |
| VlSomers/prtreid | 42 | 2025-04-11 | **none** | our embedder |
| KonradHabel/clip_reid | 19 | 2024-01-05 | MIT | CLIP-ReIdent (UniBw) |
| evs-broadcast/BroadTrack | 15 | 2025-02-17 | **none** | WACV'25 camera tracking |
| SoccerNet/sn-trackeval | 5 | 2025-07-29 | MIT | GS-HOTA implementation |
| TrackingLaboratory/CAMELTrack | - | - | - | learned association, weights released |
| ron941/GTATrack-STC2025 | - | - | - | Deep-EIoU + GTA, released |

**License risk concentrated in our current stack:** PnLCalib is **GPL-2.0**, sn-gamestate is
**GPL-3.0**, Koshkina's pipeline is **CC BY-NC 3.0 (non-commercial)**, and PRTreID / Deep-EIoU /
BroadTrack / NBJW ship **no license file at all**.

---

# 8. What is simply NOT public - flags

1. **Constructor.Tech has never released code or weights** for either the 63.81 (2024) or the
   resubmitted 63.81 (2025). Verified across the abs page, the arXiv HTML full text, ar5iv, and both
   challenge reports. Its training data (66k detection images, 62k calibration images, 550k TeamID
   images, 280k ReID crops, 70k jersey images) is entirely proprietary and unnamed.
2. **Constructor.Tech's paper contains no ablation table.** There is no published attribution of its
   63.81 to any component.
3. **Broadcast2Pitch / KIST-GSR released no code or weights** despite publishing full method,
   ablations and hyperparameters. No arXiv mirror; CVF PDF only, and CVF 403s automated fetchers.
4. **No 2025 GSR participant reported training compute, GPU count, or epochs in the challenge
   report.** The only 2025 compute figure in existence is the winner's separate WACV paper
   (1x RTX 4090).
5. **No GSR challenge participant in either 2024 or 2025 released submission code.** Zero URLs
   appear in the GSR sections of arXiv 2409.10587 and arXiv 2508.19182.
6. **The UniBw jersey model (90.95) has no public code or weights.** Only its re-ID sibling
   (KonradHabel/clip_reid) is released. lianyou's 2025 use of "a fine-tuned ViT-L14 CLIP model
   (Habel et al.)" is therefore not reproducible from public artifacts.
7. **Grad's 85.62 jersey model has no code, no weights, and no arXiv mirror.** Its training schedule,
   optimizer settings and threshold values live in a **supplementary PDF** (Sections 7, 8, 9.2,
   and 3.1.5) that is not in the CVF main-paper file. Its 200M dataset is proprietary. Its GPU is
   never stated.
8. **Koshkina's repo states no GPU requirement and no training time**, and hosts weights on Google
   Drive / OneDrive rather than a durable archive.
9. **PnLCalib does not report per-model training time**, only 200 epochs on one RTX 2080 Ti.
10. **CLIP-ReIdent names its cluster ("Monacum One") but never the GPU type, count, or hours.**
11. **BroadTrack's paper abstract does not state code state, speed, or training requirements**; its
    repo carries no license.
12. **The SoccerNet-GSR test set has known annotation errors:** Broadcast2Pitch names clips
    **SNGS-126, SNGS-131, SNGS-197** as incorrectly annotated for the team attribute, worth
    **3.81 GS-HOTA** on the team-only evaluation (73.58 -> 77.39). This is unfixed upstream.
13. **JAM (2024, 3rd, 34.40), tyler_durden (2025, 5th, 57.08) and SJTU Multi-Modal (2025, 6th,
    50.06) published nothing at all.**
14. **CAMELTrack, from the same lab as TrackLab and sn-gamestate, reports no SoccerNet numbers**
    despite SoccerNet being a supported TrackLab dataset.
15. **sn-gamestate's pinned environment (Python 3.9, torch 1.13.1, CUDA 11.7) has not been verified
    against modern datacentre GPUs by this survey.** Flagged as the first thing to test on the
    cluster.

---

# 9. URL index

**arXiv:** 2504.06357 (Constructor.Tech 2024 winner) | 2409.10587 (SoccerNet 2024 results) |
2508.19182 (SoccerNet 2025 results) | 2404.11335 (SoccerNet-GSR benchmark) |
2309.06006 (SoccerNet 2023 results) | 2405.13896 (Koshkina) | 2406.01033 (ADRS orientation jersey) |
2401.09942 (PRTreID) | 2211.03679 (BPBReID) | 2303.11855 (CLIP-ReIdent) | 2306.13074 (Deep-EIoU) |
2411.08216 (GTA) | 2602.00484 (GTATrack) | 2404.08401 (NBJW / PnLCalib) | 2412.01721 (BroadTrack) |
2505.01257 (CAMELTrack) | 2206.14651 (BoT-SORT) | 2308.16651 (MOT4MOT) | 2607.07320 (SoccerNet 2026)

**CVF open access (curl with browser UA; WebFetch gets 403):**
- Broadcast2Pitch WACV 2026: /content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
- Grad CVPRW 2025: /content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf
- GSR baseline CVPRW 2024: /content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf
- Koshkina CVPRW 2024: /content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf
- Deep-EIoU WACVW 2024: /content/WACV2024W/RWS/papers/Huang_Iterative_Scale-Up_ExpansionIoU_and_Deep_Features_Association_for_Multi-Object_Tracking_WACVW_2024_paper.pdf
- GTA ACCVW 2024: /content/ACCV2024W/MLCSA2024/papers/Sun_GTA_Global_Tracklet_Association_for_Multi-Object_Tracking_in_Sports_ACCVW_2024_paper.pdf
- BroadTrack WACV 2025: /content/WACV2025/papers/Magera_BroadTrack_Broadcast_Camera_Tracking_for_Soccer_WACV_2025_paper.pdf
(all under https://openaccess.thecvf.com)

**Repos:** github.com/SoccerNet/sn-gamestate | .../sn-trackeval | .../sn-jersey | .../sn-reid |
github.com/TrackingLaboratory/tracklab | .../CAMELTrack | github.com/VlSomers/prtreid |
.../keypoint_promptable_reidentification | github.com/mkoshkina/jersey-number-pipeline |
github.com/hsiangwei0903/Deep-EIoU | github.com/sjc042/gta-link | github.com/ron941/GTATrack-STC2025 |
github.com/mguti97/PnLCalib | .../No-Bells-Just-Whistles | .../SoccerNet-v3D |
github.com/evs-broadcast/BroadTrack | github.com/roboflow/rf-detr | github.com/baudm/parseq |
github.com/Syliz517/CLIP-ReID | github.com/KonradHabel/clip_reid | github.com/BingfengYan/CO-MOT |
github.com/kmouts/SoccerNet_v3_H250

**Data / models / eval:** huggingface.co/datasets/SoccerNet/SN-GSR-2025 |
huggingface.co/julianzu9612/RFDETR-Soccernet | codabench.org/competitions/4365 (OPEN) |
codabench.org/competitions/4469 (closed) | eval.ai/web/challenges/challenge-page/2251/overview |
soccer-net.org/challenges | trackinglaboratory.github.io/tracklab/
