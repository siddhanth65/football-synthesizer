# GSR with cluster GPUs — the synthesized roadmap (2026-07-29)

Synthesis of three sources produced today: docs/GSR_METHODS_DEEP_DIVE.md (methods, per-team),
docs/SOCCERNET_DATA_INVENTORY.md (data), and a 5-angle adversarially-verified sweep (5/5 core
claims confirmed 3-0 against primary sources before its later stages hit the session limit).
Audience: Sid + professor, deciding whether and how to pursue GSR. Not a commitment — a costed map.

## The verdict in one paragraph

A serious GSR attempt is feasible, and by cluster standards CHEAP. Verified fact: **no winning
GSR system ever used more than one GPU** — the 2024 winner (63.81) trained everything on a single
A100 40 GB; the 2025 winner (63.90, KIST-GSR = the Broadcast2Pitch paper, WACV 2026) on a single
RTX 4090, including fine-tuning LLaMA-3.2-Vision for exactly ONE epoch. A cluster's value here is
parallel experiments and fast evaluation loops (~9 GPU-h per full test-split eval at baseline
speed), not model scale. The eval server (codabench 4365, Test phase) is **open, undated,
auto-scoring, 1 submission/day** — leaderboard-ranked numbers are available to us at any time.

## The one table that is the whole roadmap

Broadcast2Pitch Table 5 — identical detector/tracker/calibration, ONLY the identity model varies:

| Identity model                              | GS-HOTA |
|---------------------------------------------|---------|
| PRTreID + EasyOCR + ResNet-18 colour        | 18.11   |
| CLIP encoder + attribute heads              | 60.13   |
| LLaMA-3.2-Vision (1 epoch fine-tune)        | 61.48   |

The top row is the shape of OUR current stack. The identity model is not a component of the
problem; it effectively IS the problem (their own attribute decomposition: jersey errors cost
~2.5x everything else, -14.82 GS-HOTA). And the step from CLIP-with-attribute-heads to an 11B VLM
buys only +1.35 — the paper itself calls the VLM's edge "modest... at higher computational cost."
Corroborating: both years' leaderboards show every team high on association, low on identity.

Caveat on comparability: their 18.11 is their re-implementation of the baseline stack on the test
split; our measured 33.20 is our stack on our declared valid-split TEST-38. Do not equate the
numbers; do trust the ordering and the gap.

## What this means for us, concretely

Our year of measured work already rebuilt most of a mid-table GSR system (BoT-SORT tracking
validated at stride 2, GTA connector, PnLCalib at 0.13-0.20 m on foreign stadiums, per-crop OCR
with a measured 2.4x densification, GS-HOTA 22.85 -> 33.20 on the public split). What we lack is
exactly what Table 5 prices: a TRAINED identity model. We could never train one — no GPU and no
labels. The cluster removes the first constraint; the GSR train split (9.76 GB, never fetched —
we hold only valid) removes the second, at jersey-GT density 0.762, far above our broadcast's
0.087-0.21 and above the 0.347 solvability bar our own evidence-density law measured.

## The staged plan (each step has a number attached)

- **S0 — environment probe (half a day, cluster).** sn-gamestate pins python<3.10 /
  torch 1.13.1 / CUDA 11.7 — predates sm_90, so H100 nodes likely fail; A100 fine. Verify before
  anything else. (Deep-dive flagged this; untested claim.)
- **S1 — data (one evening).** Fetch GSR train+test (18.6 GB; challenge split has no labels,
  skip). Optionally SoccerNet re-ID (+18.6 GB) for embedder training. Skip SoccerNet-Tracking:
  verified same 200 clips as GSR, re-annotated.
- **S2 — plant the flag (one day).** Submit our CURRENT pipeline to codabench 4365. Whatever it
  scores becomes our official, public, leaderboard-ranked baseline — and the thesis gets an
  externally-graded before/after arc.
- **S3 — the big lever (the core work).** Train the CLIP-encoder + attribute-heads identity
  model on GSR train (the 60.13 recipe; single-GPU scale). Integrate behind our existing
  tracking. This is the one step with a measured ~40-point ceiling attached.
- **S4 — jersey head with abstention (parallel to S3).** The 2025 jersey SOTA (Grad, CVPRW 2025:
  torso-crop ViT, digit-aware tied head, Dirichlet evidential abstention, 85.62 on the Challenge
  split vs Koshkina's 79.31) has NO code — but its core idea, uncertainty-gated abstention with
  "no number visible" as a first-class outcome, is the same insight our OCR-densification work
  measured independently. Reimplementation is moderate effort and doubles as a thesis chapter.
- **S5 — optional VLM pass.** LLaMA-3.2-Vision 1-epoch fine-tune (their exact recipe fits a
  4090). Gated HF access + regional restrictions need checking from India first. Expected gain
  over S3: ~+1.4. Do last, if at all.
- **Throughout:** 1/day submission cadence to the eval server as the honest progress meter.

## Risks and open items (verified where marked)

- **Nobody's code exists.** No GSR participant, 2024 or 2025, released code or weights
  (verified across both challenge reports + CVF papers). Everything above is reimplementation
  from method descriptions. Constructor.Tech's 63.81 is unreproducible-by-construction
  (proprietary training data: 550k team-ID, 280k ReID, 70k jersey crops).
- **License conflict on GSR data**: HF card says GPL-3.0, arXiv says CC BY-4.0. Resolve before
  publishing anything derived. (Unresolved.)
- **Domain**: all GSR footage is Swiss Super League 2019. Our EPL work is one domain step away;
  transfers both directions need measuring, not assuming.
- **The 2026 challenge is PCBAS, not GSR** (verified): GSR ranking = open test server, not a
  prize. The prof should know the distinction. PCBAS (the attribution task we are already
  running end-to-end) is where a 2026 challenge entry would live — and our FOOTPASS harness is
  already built for exactly that.
- Prior-survey correction (logged): the "Constructor.Tech 2025 = RF-DETR+BoT-SORT+GTA" recipe
  was the organisers' field survey, not a team submission — Constructor.Tech filed no 2025
  method summary.

## The thesis angle (why this is not just leaderboard chasing)

Our differentiators, all already on disk: the evidence-density law (what identity evidence is
REQUIRED — nobody else has measured this), pre-registered evaluation discipline with a public
correction trail, the from-pixels PCBAS number (in progress on game_18 — the official baseline
consumes ground-truth state; ours does not), and the ManU demo as the in-domain application no
public dataset can supply. A GSR pursuit slots under all of it as "the identity model our law
said we needed, trained at the density our law said was sufficient."
