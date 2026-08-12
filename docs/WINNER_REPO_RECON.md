# Winner repo recon — Broadcast2Pitch / KIST-GSR (GS-HOTA 61.48 test / 63.90 challenge)

Recon date: 2026-08-12. Read-only; nothing cloned, no weights downloaded.
Repo: https://github.com/yinmayoo185/SoccernetGSR

## 0. Correcting the prior surveys

Our surveys were *not* wrong about the paper — `docs/GSR_METHODS_DEEP_DIVE.md` already carries
Broadcast2Pitch's full ablation tables. The miss is narrow and specific:

- `docs/GSR_METHODS_DEEP_DIVE.md:46` states: *"Broadcast2Pitch: code none found; no arXiv mirror"*.
  The **code half of that line is now false.** The repo exists and is public.
- **It did not postdate our surveys.** GitHub API `created_at` = **2025-12-30T06:29:28Z**,
  `pushed_at` = **2026-06-02T02:26:14Z**. Both predate `GSR_METHODS_DEEP_DIVE.md` (2026-07-29) and
  `SOCCERNET_REPO_SWEEP.md` (2026-08-02). We simply missed it — the repo has no description, no
  topics, no README stars-bait, and the owner handle (`yinmayoo185`) does not match the paper's
  first author string most people would search (`Oo`, `Jinwook Kim`, `KIST`). Search hygiene note:
  it is findable by querying the *repo name* `SoccernetGSR`, not by the paper title.
- 11 stars, 3 forks, 23 commits, 18.4 MB. Owner: `yinmayoo185` (Yin May Oo, KIST).

## 1. LICENSE — the decisive finding

**There is no license.** GitHub API returns `"license": null`; there is no `LICENSE`,
`COPYING`, or `NOTICE` file at the repo root (confirmed against the full recursive tree —
243 non-`.pyc` blobs, listed in §2).

The only license file anywhere in the tree is `sn-trackeval/LICENSE` — **MIT, Copyright (c) 2020
Jonathon Luiten** — which covers the *vendored evaluation toolkit only*, not the authors' own code.
The vendored `yolox/` tree carries **no** license file either (upstream YOLOX is Apache-2.0, but
that grant is not reproduced here).

Consequences, stated plainly:

- Absent an explicit grant, default copyright applies. GitHub's ToS (§D.5) gives us the right to
  **view and fork within GitHub** and nothing more — no right to use, modify, redistribute, or
  incorporate into our own work.
- **Vendoring their code into `football-synthesizer` is not available to us.** Neither is shipping
  it server-side as part of a product, nor redistributing their weights.
- Running it locally as a **private benchmark reference** is the grey zone. The cheap fix is one
  email: the README names **Jinwook Kim <jinwook.kim21@gmail.com>** for inquiries. A short request
  for permission-to-use (and ideally a license addition) costs nothing and de-risks everything.
  Recommend sending it before any engineering time is spent on their code.
- `sn-trackeval` (MIT) **is** freely reusable — but we already have the official SoccerNet
  `sn-trackeval`, so this buys nothing.

## 2. Completeness — full pipeline, zero training code

**It is the complete inference pipeline, end to end**, not identity-only. Every stage of the paper
is present as runnable code:

| Stage | Files | Status |
|---|---|---|
| Detection | `exp/yolox_x_soccernet.py`, `yolox/` (full vendored YOLOX) | complete |
| Tracking | `yolox/EIoU_tracker/Deep_EIoU*.py`, `KF.py`, `matching.py` | complete (DeepEIoU) |
| ReID | `reid`/torchreid `FeatureExtractor`, OSNet x1_0 | complete (external dep) |
| Calibration / SFR | `kpts.py` (65 KB), `sfr/inference.py`, `sfr/models/attention_effv2sunet.py`, `sfr/template/soccernet_template_97.npy` | complete |
| Identity | `jersey_model/CLIPFinetune.py` + CLIP/LLaMA paths in `inference_soccernetGSR.py` | complete |
| IDATR | `IDATR/{rmv_doub_bbox,gen_tracklets,refine_tracklets,create_court_file}.py` | complete |
| Output/eval | `write_json_file_team.py`, `sn-trackeval/`, `visualize_prediction_results.py` | complete |

**What is absent — and this is the important half:**

- **No training code of any kind.** No `train.py`, no dataset class, no loss function, no
  `tools/` dir. `jersey_model/CLIPFinetune.py` is **81 lines of `nn.Module` definition only**.
  There is no trainer for the CLIP head, none for the SFR keypoint/line model, none for YOLOX,
  and none for the LLaMA LoRA.
- `unsloth_compiled_cache/` (18 × `Unsloth*Trainer.py`, ~1.5 MB) is **auto-generated Unsloth cache
  committed by accident** (commit `192c8485`, "Add unsloth_compiled_cache folder"). It is residue
  proving they fine-tuned with Unsloth; it is *not* their training script and contains none of
  their hyperparameters or data pipeline.
- **Linux-only.** `yolox/_C.cpython-{38,311,312}-x86_64-linux-gnu.so` are committed prebuilt
  extensions; README requires Linux + NVIDIA + Python 3.12. **The 4 GB Windows laptop cannot run
  this pipeline at all — cluster only.**
- Rough edges: `refine_tracklets.sh` hardcodes `PYTHON_EXEC="/home/ymo/anaconda3/envs/SoccernetGSR/bin/python"`;
  `checkpoints/osnet_...pth` is a **2-byte placeholder stub**; committed `.pyc` throughout;
  `README.md` has two corruption artifacts where "4: Format Conversion" was pasted into headings
  (lines 71, 81).

### Weights — all five, public, no gating

`download_properties.py` pulls from a **public Google Drive folder** (HTTP 200, listing readable
without auth):
https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh

Confirmed present in the folder listing:

- `yolox_soccernet.pth.tar` (detector)
- `sports_model.pth.tar-60` (OSNet ReID, SportsMOT-trained)
- `SoccernetGSR_EfficientNet_Best.pth` (SFR keypoint+line model)
- `CLIP_Jersey.pth` (the CLIP identity head)
- `lora_model_jersey_role_soccernet` (shared **folder** — the LLaMA-3.2-Vision LoRA)

No HuggingFace, no gated download, no NDA. **Caveat:** the LoRA is loaded via
`FastVisionModel.from_pretrained(model_dir, load_in_4bit=True)`. If that folder is an adapter
rather than a merged model, it will pull base **Llama-3.2-11B-Vision from HF, which *is* gated**
(Meta license acceptance). Unverified — we did not download. Also: a Drive folder is not an
archival host; it can vanish. If we intend to use it, mirror early (with permission).

## 3. The identity module — architecture, confirmed

### CLIP variant (`jersey_model/CLIPFinetune.py`, 81 lines)

Exactly as the paper describes, and simpler than expected:

- Backbone: **CLIP ViT-L/14**, `freeze_clip=True` by default — the CLIP weights are **frozen**;
  only the heads train. `clip_feature_dim = 768`.
- Three heads, all identical 2-layer MLPs (`Linear(768,256) → ReLU → Dropout(0.2) → Linear(256,C)`):
  - `role_classifier`, **C=5** → `{0:Player, 1:Goalkeeper, 2:Referee, 3:Ball, 4:Other}`
  - `digit1_classifier`, **C=11** (tens digit)
  - `digit2_classifier`, **C=11** (units digit)
  - class **10 = "null or invisible"** in both digit heads.
- **No team/colour classification head.** Colour is a single `Linear(768,768)` projection
  (`color_projection`) whose output is cosine-matched at inference against CLIP *text* embeddings of
  13 hardcoded colour names: `["Red","Blue","Green","Yellow","White","Black","Orange","Purple",
  "Pink","Brown","Grey","Charcoal","Neon yellow"]`.

Jersey number assembly (`inference_soccernetGSR.py:130`):

```python
def get_number(digit1, digit2):
    """10: null or invisible"""
    if digit1 == 10:
        number = 100 if digit2 == 10 else digit2
    else:
        number = digit1 * 10 + digit2
    return number
```

**`100` is the "no visible number" sentinel** and propagates all the way to the JSON writer, where
it becomes `None`. This sentinel is load-bearing in every aggregation function.

**The critical observation for us:** `predict_role_and_jersey_batch_clip` does
`torch.max(logit, 1)` and **keeps only the argmax index — every confidence is discarded at the
per-crop level.** The tuple returned is `(role: str, number: int, color: str)`. Nothing downstream
ever sees a probability. Despite the name, it loops one image at a time (`for image in images`),
`preprocess` → `unsqueeze(0)` — no real batching.

### LLaMA variant

Not a classifier — a **multimodal autoregressive generation** task (paper §3.2). Unsloth
`FastVisionModel`, 4-bit, LLaMA-3.2-Vision. Prompting is a single chat turn:

- instruction (default): `"Classify Role, Jersey Number and Jersey Color"`
- `apply_chat_template(messages, add_generation_prompt=True)` with `{"type":"image"}` + text
- generation: `max_new_tokens=128, temperature=1.0, min_p=0.1` — **stochastic decoding**, and a
  fresh `TextStreamer` per image
- output parsed as **comma-separated `role, number, color`** — `prediction.split("\n")[-1].strip().split(',')`

Fine-tune recipe (paper §4.3, *not* in the repo): **1 epoch, lr 1e-5, on the SoccerNet jersey
dataset** (Cioppa et al., Sci. Data 2022).

### Tracklet-level aggregation — the majority vote is confirmed, twice

The paper's majority vote is real and it is a **bare `Counter.most_common(1)`**. It appears in two
separate places:

1. `IDATR/refine_tracklets.py:209` — `majority_vote(arr, jersey=False)`, used **9×**. Its only
   sophistication is the `100` sentinel: if the modal value is `'100'` and the list is not
   *entirely* `'100'`, it drops all `'100'`s and re-votes.
2. `write_json_file_team.py:31` — `majority_jersey(series, threshold=0.97)`, a *different* rule for
   the final JSON: `100` wins only if it holds **≥97%** of the votes, otherwise `100` is deleted
   and the runner-up wins. Roles and colours use plain `Counter.most_common(1)`.

Team assignment is cruder still (`inference_soccernetGSR.py`):
`global_color_team_assignment` lower-cases predicted colour *names*, takes the **two most frequent
colour strings across the whole clip** → team 0 / team 1, everything else → `-1`; then
`unify_team_assignments` does a per-`track_id` majority vote over `{0,1}`. `AgglomerativeClustering`
is imported but used only once (not in the team path shown). Left/right side is then assigned by
mean pitch-x per colour cluster (paper §3.4).

## 4. Retraining on our corpora — possible, but we write the trainer

**Their code cannot retrain anything.** To put our S2/S3 jersey + v3 corpus through their CLIP
variant we would write the training loop ourselves. That is genuinely small — frozen ViT-L/14 +
three CE losses — but note two real gaps:

- **The colour head's training objective is not recoverable from the repo.** Inference cosine-matches
  `color_projection(feat)` against CLIP text embeddings of the 13 colour names, so the loss was
  presumably contrastive/cosine to the GT colour name's text embedding — but that is inference-time
  inference, not documented fact. We would be guessing.
- Label format we must produce: `role ∈ {0..4}` in their exact order, and jersey split into
  **two digit labels with 10 = absent** (not a flat 0-99 class, not a string). Our corpus labels
  will need remapping, and `100` must be preserved as the no-number sentinel end-to-end.

No dataset class, no expected on-disk format, no augmentation recipe ships. The only stated
hyperparameters are the paper's (§4.3): SFR model 25 epochs @ lr 1e-4, 384×384; LLaMA 1 epoch @
lr 1e-5; all on **a single RTX 4090 24 GB**.

## 5. IDATR — what it actually does

A **split-and-merge** pass over finished tracklets (paper §3.3; `IDATR/refine_tracklets.py`, 587
lines). Its stated novelty vs. the GTA/`gta-link` work it is "inspired by [41]": splitting is driven
by **VLM identity predictions instead of ReID appearance features**, which avoids spurious splits
when ReID degrades under occlusion.

- **Split** (`detect_id_switch`): cut at frame gaps `> τ_gap`, then keep a cut only if the
  majority-voted **role, jersey, or team differs** across the boundary.
- **Merge** (`merge_tracklets`): merge `T_a`,`T_b` when **all four** hold — ReID cosine distance
  `< τ_cos`; end-to-start pitch distance `< τ_spatial`; **no temporal overlap** (`F_a ∩ F_b = ∅`);
  and identity-consistency `I(T_a,T_b)=0`.
- Post-filter (paper §3.4): drop tracklets with **length < 20** and detections outside the pitch.

Thresholds — **paper vs shipped config disagree on one:**

| | paper §4.3 | `configs/config.yaml` |
|---|---|---|
| `τ_gap` | 25 | `GAP_LEN: 25` ✓ |
| `τ_cos` | 0.4 | `MERGE_DIST_THRES: 0.4` ✓ |
| `τ_spatial` | **0.8** | **`SPATIAL_FACTOR: 1`** ✗ |
| use jersey | — | `USE_JN: True` |

`τ_spatial=0.8` is **not implemented as a fixed threshold**. The code calls
`get_spatial_constraints(tid2track, factor)` to derive `max_x_range`/`max_y_range` empirically from
the data and scales them by `SPATIAL_FACTOR=1`. Different parameterization, different behaviour.
Anyone reproducing from the paper will not reproduce the code, and vice versa.

**Could IDATR consume our tracklets? Yes, cleanly.** Its input contract is a flat per-detection
table read by `IDATR/gen_tracklets.py` into `IDATR/Tracklet.py` — a 3.2 KB plain container:
`track_id, times[], scores[], bboxes[], features[] (512-d ReID), role[], jn[], jc[], team[]`. Any
tracker that emits those nine parallel arrays can drive it. The binding constraint is the
**512-d OSNet feature vector** and the identity strings in their vocabulary (roles capitalised,
jersey `100` sentinel).

## 6. Integration surface — the assessment

### 6a. Their identity module on our substrate — **cheap**

`predict_role_and_jersey_batch_clip(images, jersey_model, device) -> [(role, number, color), ...]`
takes a list of **PIL crops** and returns tuples. That is as clean a standalone interface as we
could ask for.

- Total surface: **`jersey_model/CLIPFinetune.py` (81 lines) + ~40 lines of the predict function +
  `CLIP_Jersey.pth`**. Dependencies: `torch`, OpenAI `clip`. **No YOLOX, no torchreid, no unsloth,
  no Linux `.so`** — the CLIP path is pure Python + PyTorch.
- ViT-L/14 at fp16 is ≈0.9 GB, so this is one of the few pieces that *might* run on the 4 GB
  laptop at batch 1. Everything else in their repo will not.
- The LLaMA path is the opposite: unsloth + 4-bit 11B-Vision, per-crop autoregressive generation
  with `max_new_tokens=128`. Cluster-only, slow, and worth **+1.35 GS-HOTA** (see §7).

### 6b. Our evidential fusion inside their code — **more expensive than it looks**

The seam is real and the paper names it (§6, verbatim): *"jersey number predictions are aggregated
via majority voting, which is particularly fragile when correct digits are sparsely observed"*, and
*"As future work, we aim to make the tracklet refinement module learnable... move beyond fixed
heuristics."* Swapping `majority_vote` is an invited change.

But the two functions are pure and trivially replaceable — **the cost is not in the swap, it's in
the plumbing.** Per-crop confidences are destroyed at the `torch.max` in §3. To feed an evidential
fusion you must thread softmax/logits through **five files**: `inference_soccernetGSR.py` (stop
discarding, write extra columns) → `IDATR/gen_tracklets.py` (parse) → `IDATR/Tracklet.py` (store) →
`IDATR/refine_tracklets.py` (consume in `majority_vote` *and* in `get_distance`'s identity
constraints) → `write_json_file_team.py` (consume in `majority_jersey`). And it works **only on the
CLIP path** — the LLaMA path emits free text and carries no calibrated confidence at all without
extracting generation logprobs.

**Verdict: 6a is decisively cheaper** (2 files + 1 checkpoint, no plumbing, laptop-feasible) and is
the right first move if we want a like-for-like identity comparison. 6b is where our
differentiation lives and it is the authors' own stated gap — but it is a ~5-file change inside
unlicensed code, which §1 says we should not be modifying until we have permission. **And §7 says
neither one is where our missing points are.**

## 7. The numbers that reframe P1

From the WACV paper (SoccerNet-GSR **test** split; PDF fetched with a browser UA — WebFetch gets
403 on CVF):
https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf

**Table 4 — homography estimation (HE) and IDATR ablation:**

| HE(Keypoint) | HE(Line) | IDATR | GS-HOTA | GS-DetA | GS-AssA | IDF1 |
|---|---|---|---|---|---|---|
| ✓ | ✗ | ✗ | 48.23 | 34.77 | 66.91 | 51.68 |
| ✗ | ✓ | ✗ | 56.39 | 42.77 | 74.40 | 61.31 |
| ✓ | ✓ | ✗ | 58.51 | 44.63 | 76.72 | 61.57 |
| ✓ | ✓ | ✓ | **61.48** | 48.47 | 78.00 | 64.20 |

**Table 5 — athlete identification model, same detector, same tracker:**

| Model | GS-HOTA | GS-DetA | GS-AssA | IDF1 |
|---|---|---|---|---|
| PRTReID + EasyOCR + ResNet-18 | 18.11 | 6.54 | 50.14 | 10.62 |
| **CLIP** | **60.13** | 46.88 | 77.15 | 62.20 |
| **LLaMA-3.2-Vision (theirs)** | **61.48** | 48.47 | 78.00 | 64.20 |

So: **calibration is worth +10.28 GS-HOTA** (48.23 → 58.51), **IDATR +2.97** (58.51 → 61.48), and
the entire CLIP→LLaMA identity upgrade is worth **+1.35**. The paper concedes it: LLaMA's *"modest
accuracy gain over CLIP comes at higher computational cost."*

**Table 7 — cost of each identity attribute** (Pitch-only is the upper bound):

| Pitch | Role | Team | Jersey | GS-HOTA |
|---|---|---|---|---|
| ✓ | | | | 79.52 |
| ✓ | ✓ | | | 79.20 |
| ✓ | | ✓* | | 73.58 |
| ✓ | | | ✓ | 77.39 |
| ✓ | ✓ | ✓ | | 64.70 |
| ✓ | ✓ | ✓ | ✓ | 61.48 |

Role is nearly free (−0.32). **Team is the single most expensive attribute** (−5.94, *even after*
excluding the mis-annotated clips), jersey next (−2.13 alone).

### The SNGS-126/131/197 claim — precisely scoped

Table 7's caption: *"Samples SNGS-126, 131, 197 are incorrectly annotated for the **team attribute**
in the SoccerNet-GSR test set. Asterisk (\*) indicates the exclusion of these samples."*

The exclusion applies **only to the Team row of Table 7**. The headline **61.48 in Table 2 is on the
full test set** — nothing excluded. This matters because it is the source of the one open-source
reproduction failure (§8).

### Reproduction: issue #1, and what it really shows

https://github.com/yinmayoo185/SoccernetGSR/issues/1 — "Performance reproduction", opened
2026-05-25 by `xiezhang666`, closed 2026-06-01. One issue total; zero open.

> *"Following the paper by excluding SNGS-126, 131 and 197, I got **48.32 GS-HOTA**. I failed to
> replicate the original performance metrics 61.48."*

Maintainer's reply gave two causes: (1) pull the latest `kpts.py` — *"we updated kpts.py with
improved sample point processing for homography computation"*, plus `opencv-contrib-python` is now
required; (2) *"The 61.48 GS-HOTA in the paper uses LLaMA... The default config uses CLIP."*

**Reason (2) is a red herring and Table 5 proves it:** CLIP scores 60.13, so jersey mode explains
**1.35 points of a 13.16-point shortfall.** The reproducer's **48.32 is a near-exact match for
Table 4's keypoint-only-homography row, 48.23** — their calibration was silently falling back to the
DLT/keypoint path. Reason (1) was the real cause, and the fix landed as the repo's most recent
commit, **`dbfb65c0`, 2026-06-02: "update keypoint extraction with robust line/circle fitting and
fix opencv dependency"** — one day after the issue was closed. That commit is HEAD and has **no
public reproduction against it.**

Net: **61.48 has never been independently reproduced in public.** The one attempt landed at 48.32
against pre-fix code.

## 8. What this changes about P1

1. **We cannot vendor their code.** No license (§1). Email `jinwook.kim21@gmail.com` before
   spending engineering time. Weights are freely downloadable but that is not a licence either.
2. **Our gap is calibration, not identity.** Their own ablation puts +10.28 GS-HOTA in the
   homography module and +1.35 in the entire CLIP→LLaMA identity upgrade. Our public-board 53.09
   vs their challenge-set 63.90 is a 10.81-point gap — the same size as their homography delta.
   The correct P1 target is **sports field registration** (their JaC₅ 69.38 vs next-best 56.88),
   not the identity module we came here to shop for.
3. **Chasing LLaMA is the worst available trade.** +1.35 GS-HOTA for an 11B 4-bit VLM doing
   per-crop autoregressive generation. The CLIP variant delivers 60.13 for ~120 lines of Python and
   a frozen ViT-L/14 — and is the only component of their stack that could plausibly run on the
   4 GB laptop.
4. **Team affiliation is the softest target in their system.** It costs them −5.94 GS-HOTA, and
   their method is top-2-most-frequent-colour-*name* matching against 13 hardcoded strings, then a
   majority vote. That is a far weaker mechanism than anything we would build, and it is worth more
   than jersey.
5. **Their headline number is unreproduced, and the reproduction trap is known.** Anyone (us
   included) who runs pre-`dbfb65c0` code, or whose line/circle fitting silently degrades, lands
   near 48. If we ever benchmark against their pipeline, pin HEAD `dbfb65c0`, install
   `opencv-contrib-python`, and sanity-check calibration first — a GS-HOTA near 48 means the
   homography fell back, not that identity failed.

## 9. URLs

- Repo: https://github.com/yinmayoo185/SoccernetGSR
- Issue #1 (reproduction): https://github.com/yinmayoo185/SoccernetGSR/issues/1
- HEAD commit `dbfb65c0`: https://github.com/yinmayoo185/SoccernetGSR/commit/dbfb65c05c847e8e50baa47aa91ace960cc5e600
- Weights (Google Drive, public): https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh
- WACV 2026 paper PDF: https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
- WACV poster page: https://wacv.thecvf.com/virtual/2026/poster/435
- IEEE Xplore: https://ieeexplore.ieee.org/document/11492373/
- Follow-up (Broadcast2Pitch++, depth-aware): https://www.researchgate.net/publication/405759501
- SoccerNet 2025 Challenges Results (omnibus): https://arxiv.org/abs/2508.19182
- Contact: Jinwook Kim <jinwook.kim21@gmail.com>, Human Data Intelligence Lab, KIST
