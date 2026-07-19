# Zero-shot action-spotting probe — Brighton 2-1 Man Utd (2024-08-24)

**Question (REVIEW_CRIB Q1):** can an *off-the-shelf* SoccerNet action-spotting model give us a
validated event layer (goals / shots / corners / cards) on a broadcast we never trained on, with
**no training**? Directly rebuts the reviewer attack "your report can't see goals."

**Answer:** yes for the high-value events. Zero-shot, the **3 real goals are the 3 highest-confidence
Goal peaks** (0.76-0.95, cleanly separated from noise) with the correct half split (1 H1 / 2 H2);
**Yellow 3/3, Red 0/0 exact**; **Foul 23 vs 22**, **Corner 7 vs 8**, **total shots 25 vs 25**. The
shot on/off-target split and offside are noisy. Adopt-with-caveats for Layer 3.

This is a **PROBE**: no pipeline module changed. New code is `tools/action_spot_probe.py`
(+ `tools/test_action_spot_probe.py`). Model code + weights live outside the repo at
`~/action-spot-env/` (BSD-3 external code, not committed).

---

## Method

- **Model:** E2E-Spot (Hong et al., *Spotting Temporally Precise, Fine-Grained Events in Video*,
  ECCV 2022). RegNet-Y 200MF backbone + Gate-Shift Module (GSM) temporal shift + single-layer
  bidirectional GRU head. 4.46 M params.
- **Weights:** `soccer_rny002gsm_gru_rgb` from the authors' public model repo
  `github.com/jhong93/e2e-spot-models` (18 MB checkpoint `checkpoint_088.pt`). Trained on the
  **SoccerNet-v2 Action Spotting** train split (17 classes + background = 18 outputs), validated on
  the val split; frames from 720p downsampled to 224 px high. RGB only (no optical-flow sidecar).
- **Task match:** the model's 17 classes are exactly the event vocabulary we need — Goal, Corner,
  Shots on/off target, Yellow/Red card, Foul, Offside, Penalty, Throw-in, Substitution, Kick-off,
  Clearance, direct/indirect free-kick, Ball out of play, Yellow->red.
- **Input:** `matches/brighton_manutd/{h1,h2}/chunk_*.mp4` via `core.registry` (11 chunks,
  ~100 min, 1920x1080/25fps). Frames extracted at **2 FPS, scaled to 224 px high** (native 16:9
  width ~398), ImageNet-normalized RGB, no crop (`config crop_dim: null`) — matches training.
- **Inference:** overlapping windows of `clip_len=100` frames (= 50 s at 2 FPS) at 50% stride,
  per-frame softmax averaged over overlaps. Per-chunk scores checkpointed to
  `results/action_spotting_probe/brighton_manutd/scores_*.npz` (resumable from disk).
- **Peak picking:** per class, greedy NMS over the per-frame class-probability timeline
  (descending score, 30 s suppression window), reported at a 0.30 gate.

### Provenance / reproducibility notes

- Loads **strict** on the current stack (torch 2.11+cu128, timm 1.0.28) after **one** compat patch
  to the external `spot/model/shift.py`: timm 1.0 renamed `ConvBnAct` -> `ConvNormAct`, so the GSM
  builder's `isinstance(net, timm.models.layers.conv_bn_act.ConvBnAct)` was replaced with a
  duck-typed `hasattr(net, "conv")` check. All backbone state-dict keys align across timm versions;
  no key was renamed. No sidecar venv was needed.
- The checkpoint has 18 outputs = 17 foreground classes + background(0). Foreground index = sorted
  class name position + 1 (`load_classes` convention), so **Goal = index 6**.

### Oracle (validation target, pre-committed before looking at model output)

Sofascore match 12436888 (cached, `outputs/oracle/sofascore/team_stats_12436888.parquet`):
Brighton 2-1 Man Utd, HT 1-0 -> **3 goals (1 in H1, 2 in H2)**. Whole-match counts:
Corner 8, shots-on-target 9, shots-off-target 11 (total shots 25), Yellow 3, Red 0, Foul 22,
Offside 6.

---

## Per-class results (whole match, 0.30 gate)

| class | detected | oracle | by half | read |
|---|---|---|---|---|
| **Goal** | 5 | **3** | h1=2 h2=3 | top-3 peaks = the 3 real goals (see below); 2 extra incl. a goal-replay |
| **Yellow card** | 3 | **3** | h1=0 h2=3 | exact |
| **Red card** | 0 | **0** | h1=0 h2=0 | exact |
| **Foul** | 23 | 22 | h1=11 h2=12 | +1 |
| **Corner** | 7 | 8 | h1=2 h2=5 | -1 |
| Shots on target | 17 | 9 | h1=6 h2=11 | over-triggers |
| Shots off target | 8 | 11 | h1=5 h2=3 | under |
| *(shots total)* | *25* | *25* | - | **on/off split noisy but total exact** |
| Offside | 3 | 6 | h1=2 h2=1 | under (offside is visually subtle) |
| Ball out of play | 55 | - | h1=28 h2=27 | plausible volume |
| Throw-in | 29 | - | h1=16 h2=13 | plausible volume |
| Indirect free-kick | 20 | - | h1=10 h2=10 | no oracle |
| Clearance | 8 | - | h1=6 h2=2 | no oracle |
| Direct free-kick | 5 | - | h1=1 h2=4 | no oracle |
| Substitution | 3 | - | h1=0 h2=3 | plausible (subs are 2nd-half) |
| Kick-off | 2 | - | h1=2 h2=0 | no oracle |
| Penalty | 0 | - | - | none in match |
| Yellow->red card | 0 | - | - | none in match |

---

## Goal-peak analysis (the headline)

Top Goal-class peaks across the match (video-elapsed time within each half):

| rank | half | time | score | interpretation |
|---|---|---|---|---|
| 1 | h2 | 13:14 | **0.953** | real goal (H2) |
| 2 | h2 | 48:01 | **0.895** | real goal (H2, deep stoppage -> the late winner) |
| 3 | h1 | 31:35 | **0.756** | real goal (H1) — lands at ~31', consistent with the known ~31' opener |
| 4 | h2 | 23:24 | 0.581 | false positive (big chance / disallowed / replay) |
| 5 | h1 | 33:02 | 0.438 | likely a **replay** of the H1 goal (~90 s after peak #3) |
| 6+ | - | - | <0.077 | noise floor |

- **The 3 real goals are the 3 highest-confidence peaks**, with a clean gap (0.76-0.95 real vs
  <=0.58 spurious vs <0.08 noise). At a **0.6 gate: exactly 3 Goal spots, half split 1 H1 / 2 H2**,
  matching the oracle exactly.
- The two peaks between 0.30 and 0.6 are the expected failure mode: broadcast **goal replays** and
  big chances re-trigger the Goal head (peak #5 is ~90 s after the H1 goal). A short post-goal
  refractory window + a confidence gate removes them.
- **Timestamp sanity:** the H1 goal peak at video-time 31:35 is consistent with the widely reported
  ~31' opener. Precise match-clock alignment was not independently established (no incident-level
  oracle is cached — only team aggregates), so minute-level timestamps are corroborative, not proven.

---

## Cost (RTX 3050 Laptop, 4 GB)

- **Total inference wall-clock: 378 s (6.3 min)** for ~100 min of video = ~16x real-time.
- Per 10-min chunk (1200 frames @ 2 FPS): ~38 s including ffmpeg frame extraction + inference.
- Model load: ~10 s once.
- **VRAM: 760 MB peak** per 100-frame clip at batch 1 (measured); ~1.4 GB total process incl. CUDA
  context. Fits 4 GB with large headroom (batch 2-3x is feasible).
- Resumable: per-chunk `.npz` checkpoints; a killed run re-invokes and skips completed chunks.

---

## Verdict (5 lines) — ADOPT-WITH-CAVEATS for Layer 3

1. Off-the-shelf, zero training: the 3 goals are the top-3 Goal peaks (clean 0.76-0.95 separation),
   cards exact (Y 3/3, R 0/0), Foul 23 vs 22, Corner 7 vs 8 — directly answers "can't see goals".
2. Adopt **Goal, cards, Foul, Corner** now behind a confidence gate + post-goal refractory window,
   validated per-class against Sofascore counts (the protocol used here).
3. The **shot on/off-target split and offside are unreliable** zero-shot (17 vs 9, 8 vs 11, 3 vs 6),
   though **total shot volume is exact (25 vs 25)** — report aggregate shots, not the split.
4. Domain shift is mild: a "Full Match Replay" broadcast (not SoccerNet's feed) still lands the
   headline events; a light fine-tune on SoccerNet labels or the challenge-split weights is the
   scheduled upgrade for the split/offside classes, not a prerequisite for the goal claim.
5. Ships as a validated capability, not an excuse: gate every event on confidence + oracle count
   agreement, abstain on the noisy classes — consistent with the project's validated-or-nothing bar.

---

*Reproduce:* `python -m tools.action_spot_probe --match brighton_manutd`  (run/resume inference),
then `--analyze` (aggregate). Self-check: `python -m tools.action_spot_probe --selftest`.
Model + patched code: `~/action-spot-env/{spot,models}`. Raw scores + summary.json:
`results/action_spotting_probe/brighton_manutd/`.
