# The GSR Project — Master Explainer

**Game State Reconstruction on SoccerNet: what the task is, what we did, what we found,
and what the research contribution actually is.**

*Prepared for Sid, 2026-08-17. Every number in this document is traceable to
`STATUS.md`, `results/*.json`, or `knowledge/claims.json` (301 versioned claims).
Where a number is an estimate or an inference, it says so.*

---

## How to read this document

- **§1** explains the problem from zero. If you have never done computer vision, start here and
  do not skip it — every later section leans on the vocabulary built in §1.
- **§2** is the story: 14.76 → 53.09 → 55.41 → the leaderboard shock → the geometry campaign.
- **§3** is the literature: every paper we used, with its real abstract, what we took, and what
  happened when we used it.
- **§4** is the defence. It is the answer to *"you neither topped the leaderboard nor invented a
  new architecture — so what was the research?"* It contains sentences written to be spoken aloud.
- **§5** is the honest self-critique: what is missing, what a professor will attack, and what
  the next real bets are.
- **§6** is the pivot menu.
- **§7** is a one-page cheat sheet: ten numbers, five sentences, a glossary.

A rule that runs through the whole document: **failures are reported as failures.** This project
has more registered failures than registered wins, and that ratio is a feature of the method, not
an embarrassment. §4 explains why.

---

# 1. What the task is

## 1.1 The one-sentence version

**Input:** a 30-second clip of a football match as it appeared on television — one moving camera,
zooms, pans, players walking off the edge of the frame.

**Output:** a *minimap*. For every frame, for every person on the pitch: where they are **on the
pitch in metres** (not where they are in the picture in pixels), which **team** they belong to,
what **role** they have (player / goalkeeper / referee), and their **jersey number**.

That is Game State Reconstruction (**GSR**). The "game state" is the answer to the question
*"who is standing where, right now?"* — the thing every football analytics product needs before it
can compute anything at all (pressing intensity, distance covered, formations, expected threat).

**Why it is hard:** a broadcast camera shows you maybe 60–70% of the pitch at any moment, from a
constantly changing viewpoint, at a resolution where a distant player is 30 pixels tall and the
number on his shirt is 6 pixels of blur. You have to do four jobs at once, and an error in any one
of them destroys the row:

1. **Find** everyone (detection).
2. **Keep them the same person** frame to frame (tracking, also called association).
3. **Convert pixels to pitch metres** (camera calibration).
4. **Say who they are** — team, role, number (identification).

## 1.2 The dataset: SoccerNet-GSR

The benchmark is **SoccerNet-GSR** (Somers et al., CVPRW 2024, arXiv:2404.11335) — part of the
SoccerNet family of open football-video benchmarks.

| Property | Value |
|---|---|
| Clips | 200, each 30 seconds, 1920×1080, 25 fps (= 750 frames per clip) |
| Source footage | Swiss Super League 2019 broadcasts |
| Splits | 57 train / 58 validation / 49 test / 36 challenge |
| Annotations | 2.36 M athlete positions with role, team and jersey number; 9.37 M pitch-line points for camera calibration |
| Access | HuggingFace, under NDA terms Sid signed |

Three splits matter to us and they are **not** interchangeable:

- **valid-58** — the public validation split. We can score it locally because its labels ship.
- **TEST-38** — *our own* invention: 38 of those 58 clips, held out and never tuned on. The other
  20 are **DEV-20**, our tuning playground.
- **test-49** — the official hidden-label test split, scored only by the competition server
  (Codabench), **one submission per day, ten in a lifetime**.

Any comparison across those splits is meaningless. Confusing them is the single easiest way to
fool yourself on this benchmark, and §4 explains the discipline we built to avoid it.

## 1.3 The metric, built up gently: GS-HOTA

### Step 1 — DetA: did you find the objects?

Take one frame. Ground truth says there are 14 visible people. You predicted 13 boxes. Match
predictions to truth; count true positives, false positives, false negatives. **DetA (Detection
Accuracy)** is roughly *true positives / (true positives + false positives + false negatives)*,
averaged over the clip. Perfect detection = 1.0.

### Step 2 — AssA: did you keep each identity continuous?

Finding people is not enough. If player A is "track 7" for one second, then the tracker loses him
behind an opponent and he comes back as "track 41", you have found him both times but you have
broken his identity. **AssA (Association Accuracy)** measures, over all the matched detections,
how consistently one predicted track corresponds to one real person over time. Perfect
association = 1.0.

Two failure names you will use constantly:

- **Fragmentation** — one real player split across many tracks. Ours: about 7.9 fragments per
  real identity.
- **Contamination** (opposite of **purity**) — two real players merged into one track. Ours: row
  purity 0.94, i.e. 94% of the rows inside a track really belong to that track's majority person.

### Step 3 — HOTA: the geometric mean

**HOTA (Higher Order Tracking Accuracy)**, Luiten et al., IJCV 2021, arXiv:2009.07736:

> **HOTA = √(DetA × AssA)**

The geometric mean means you cannot win by being excellent at one and terrible at the other. A
third number, **LocA (Localisation Accuracy)**, rides along as a diagnostic: how precisely
positioned are the detections you got right. LocA does not enter the ranking.

### Step 4 — the position Gaussian: every centimetre is charged

In normal tracking, a prediction "matches" ground truth if their boxes overlap enough — a yes/no
gate. GSR works in metres on the pitch, and it does **not** use a hard gate. The similarity
between a predicted position and a true position at distance *d* metres is:

> **similarity = exp( −0.5 × (d / σ)² )**,  with **σ = 2.0427 m**

The "5-metre tolerance" everyone quotes is simply the distance at which that similarity falls to
0.05. **It is not a pass/fail radius.** This distinction is not cosmetic — it is the single
premise correction that unlocked our v10 campaign (§2.5). Under a 5 m gate, our median error of
0.43 m looks like "already solved, stop investing". Under a Gaussian, *every centimetre of error
costs score*, all the way down. We had the wrong mental model for weeks, said so out loud, and the
correction was worth +2.1 points (§2.5).

### Step 5 — the identity gate: the merciless part

**GS-HOTA** (Game State HOTA) is HOTA computed in pitch coordinates *plus one extra rule*:

> A prediction matches a ground-truth person only if the position similarity is non-trivial
> **AND team, role, and jersey number are ALL correct. One wrong attribute scores zero, however
> perfect the position.**

That single rule reshapes the whole engineering problem. Three consequences you should be able to
recite:

1. **A wrong number is worse than no number.** A prediction with `jersey = null` can still match a
   ground-truth player whom the annotators left un-numbered. A prediction with the *wrong* number
   matches nothing at all. **Abstention is a strategy, not a weakness.** Our first jersey-reading
   module moved the score from 14.76 to 19.83 while abstaining on 91.3% of tracks and hurting
   **zero of 58 clips**.
2. **Jersey errors dominate**, at roughly **2.5×** the cost of any other attribute — measured
   independently by the leading published system and by us. The campaign brief's line for this:
   *GS-HOTA is an identity metric wearing a tracking metric's clothes.*
3. **Teams are ranked on GS-HOTA alone.** DetA/AssA/LocA are the dashboard that tells you *why*;
   they are never the score.

For calibration of expectations: on the published leader's own ablation, if you **disable** the
identity attributes their detection score is about **86**; switch the identity gate on and it
collapses to about **49**. Roughly **37 points of detection accuracy die at the identity gate**,
even at state of the art. That is the benchmark in one number.

## 1.4 Every abbreviation, expanded once

You will meet these throughout. Each gets a 2–3 sentence plain explanation here, so nothing later
is undefined.

**YOLO ("You Only Look Once")** — a family of fast object detectors. It looks at the whole image
once and directly outputs boxes with class labels, instead of proposing regions and then
classifying them. We run **YOLOv8s** ("s" = small), fine-tuned on football.

**Fine-tuning** — taking a model already trained on a big generic dataset and continuing its
training on your specific data for a short while. Much cheaper than training from scratch, and
usually better when your dataset is small.

**mAP (mean Average Precision)** — the standard detector quality score; higher is better.
`mAP@0.5` means a detection counts as correct if its box overlaps the true box by at least 50%.

**IoU (Intersection over Union)** — overlap between two boxes: shared area ÷ combined area. 1.0 =
identical, 0 = disjoint. The basic currency of "is this the same box?".

**Kalman filter** — a classical algorithm that predicts where a moving object will be next,
assuming smooth linear motion, and blends that prediction with the new measurement. Standard
inside trackers; it struggles when motion is jerky (footballers) or the camera itself moves.

**ByteTrack** — a simple, very strong tracker (arXiv:2110.06864) that associates *every* detection
box, including low-confidence ones, rather than throwing weak detections away. Our first tracker.

**EIoU / ExpansionIoU** — a motion-model-free alternative to the Kalman filter: instead of
predicting where the box will go, inflate both boxes about their centres and check overlap, so a
player who moved further than his own width still overlaps himself. **Deep-EIoU** adds appearance
features to the matching cost.

**ReID (Re-Identification)** — recognising that a person seen now is the same person seen earlier,
using appearance. In football this is brutally hard because ten players wear the identical kit.

**Embedding** — a list of numbers (a vector) summarising an image crop, arranged so that two crops
of the same person land close together and two different people land far apart. "Close" is usually
**cosine distance** (angle between the vectors).

**CLIP (Contrastive Language–Image Pre-training)** — OpenAI's model (arXiv:2103.00020) trained on
400 million image–caption pairs to put images and text in one shared embedding space. In practice
people use its image encoder as an extremely good general-purpose feature extractor. We trained
one on 160,070 football crops.

**ArcFace** — a loss function (arXiv:1801.07698) that trains embeddings by pushing classes apart
by a fixed angular margin. Standard in face recognition; we used it as our identity head.

**OCR (Optical Character Recognition) / STR (Scene Text Recognition)** — reading text from images.
STR is the harder "text in the wild" version: curved, blurred, angled text on real objects — like
a number on a moving shirt.

**PARSeq (Permuted Autoregressive Sequence model)** — the scene-text recogniser we use
(arXiv:2207.06966). It reads characters using an internal language model trained over many
permutations of character order, which makes it robust and fast. We fine-tuned it on football
jersey numbers.

**Legibility classifier** — a small network that answers one question before OCR runs: *"is a
number even visible in this crop?"* Cheap, and it is what makes abstention possible.

**Pose model / HRNet / ViTPose / KeypointRCNN** — networks that find human body joints in a crop.
We use one to locate the torso so the OCR reads the shirt and not the grass. **HRNet
(High-Resolution Network)** is a common backbone for such keypoint tasks; the leading systems also
use HRNet-style models to find *pitch* keypoints.

**Homography** — a 3×3 matrix that maps one flat plane to another flat plane in an image. Because
a football pitch is flat, a single homography converts any pixel on the pitch into a pitch
coordinate in metres. Getting it right per frame *is* the calibration problem.

**DLT (Direct Linear Transform)** — the classical least-squares method for computing a homography
from at least 4 point correspondences (e.g. "this pixel is the penalty spot").

**RANSAC (Random Sample Consensus)** — a robust fitting procedure: repeatedly fit a model on a
small random subset, keep the fit that most other points agree with. It is the standard defence
against a few wildly wrong correspondences. Our chain relies on gated least-squares and a robust
(Cauchy) loss in the batch refinement rather than classical RANSAC; the effect is the same
family — *do not let a handful of bad points steer the fit.*

**PnLCalib ("Points and Lines Calibration")** — our camera calibrator (arXiv:2404.08401). A
network detects pitch keypoints and lines; an optimisation fits the camera against a 3D pitch
model. Also known in its earlier form as **NBJW ("No Bells, Just Whistles")**.

**TVCalib** — an earlier calibration approach (arXiv:2207.11709) that treats field registration as
camera calibration with a differentiable objective over line segments, rather than as raw
homography fitting.

**Bundle adjustment** — the photogrammetry technique of jointly optimising *all* camera parameters
across *all* frames at once, against all observations, instead of solving each frame alone. It is
what you do when you are offline and can look at the whole clip.

**RTS smoother (Rauch–Tung–Striebel)** — the forward-backward version of a Kalman filter. It
smooths a time series using both past and future, which you can do when you are not required to
run live.

**MILP (Mixed-Integer Linear Programming)** — optimisation with some variables forced to be whole
numbers. We use it to assign names to tracks under the constraint that two tracks alive in the
same frame cannot be the same player. The simpler special case is the **Hungarian algorithm**
(a.k.a. linear sum assignment).

**VLM (Vision–Language Model)** — a large model that takes an image plus a text instruction and
answers in text (LLaMA-3.2-Vision, Qwen-VL). The current fashionable answer to "read this jersey
number". We tested it at scale and it lost (§2.5).

**TTA (Test-Time Adaptation) / BN (Batch Normalisation) statistics** — BatchNorm layers inside a
network carry running means and variances learned during training. TTA re-estimates them on the
*test* data, unlabelled, to adapt to a new domain. Cheap and popular; **Tent** (arXiv:2006.10726)
is the canonical paper. We tested it; it failed, informatively (§2.4).

**Oracle / ceiling analysis** — replace one component with ground truth, re-score, and see what
that component could ever be worth if perfected. The most useful technique in this whole campaign.

**Leave-one-out ablation** — remove one component from the shipped system and measure the loss.

**Pre-registration** — writing down the experiment's pass/fail threshold, in a timestamped file,
*before* running it. Borrowed from clinical trials. It is the reason our negative results are
credible.

**Wilcoxon signed-rank test** — a statistical test for paired data that does not assume a normal
distribution. We use it per-sequence: for each of 20 clips, did the change help or hurt, and is
the pattern beyond chance? Comparing two bare means, without pairing, is how people talk
themselves into non-existent gains.

## 1.5 Our pipeline as it stands

Seven stages. This is the thing whose score we have been moving.

| # | Stage | What it does | What we run |
|---|---|---|---|
| 1 | **Detector** | finds people and the ball, assigns role | YOLOv8s fine-tuned on GSR-train + SoccerNet-v3; role is a detector class |
| 2 | **Tracker** | links detections into short, pure tracklets | clean-room ExpansionIoU + CLIP appearance embeddings, no Kalman filter |
| 3 | **Connector** | merges tracklet fragments of the same player | appearance clustering at τ = 0.450 behind a jersey-conflict guard |
| 4 | **Calibration** | pixels → pitch metres | PnLCalib per frame + repaired acceptance gate + gap fill (+ v10 re-selection and batch refinement) |
| 5 | **Team & side** | which cluster defends which goal | kit-colour k-means + "smaller mean pitch-x defends left" + the keeper-side rule |
| 6 | **Jersey reader** | reads shirt numbers | legibility filter → torso crop → fine-tuned PARSeq → confidence-weighted tracklet vote |
| 7 | **Identity solver** | names tracklets from a self-built roster | MILP through SciPy/HiGHS, no ground-truth roster |

Two rules added in the v9 campaign now sit on top: **track-attribute voting** (a whole track gets
one role/team/jersey by weighted majority of its own rows) and the **keeper-side rule** (a
goalkeeper's team is the side of the pitch he stands on). Both are described in §2.3.

---

# 2. The story

## 2.1 The arc at a glance

| Date | Score | Split | What changed | Kind of fix |
|---|---|---|---|---|
| 07-16 | **14.76** | valid-58 | first end-to-end run; **no jersey numbers at all** | baseline |
| 07-17 | 19.83 | valid-58 | jersey reading attached (+5.07, 0 of 58 clips hurt) | new capability |
| 07-20 | 22.85 | valid-58 | track re-linking + number propagation | new capability |
| 07-27 | 24.18 | valid-58 | tracklet connector | new capability |
| 07-28 | **33.20** | TEST-38 | **the OCR aggregation bug fix** | our own bug |
| 08-01 | **31.88** | board #1 | **legitimacy reset**: two ground-truth leaks removed — score goes DOWN | honesty |
| 08-01 | 33.37 | board #2 | calibration gap fill | engineering patch |
| 08-01 | 35.40 | board #3 | calibration gate repair | our own threshold |
| 08-02 | 39.02 | board #4 | two thresholds behind a new safety guard | our own thresholds |
| 08-06 | **53.09** | board #5 | **the trained trio** (detector, reader, tracker) | trained models |
| 08-15 | **55.41** | test-49, local | v9 bundle: attribute voting + keeper-side rule (packaged, upload pending) | rules |
| 08-16 | **57.17** | DEV-20 | v10 geometry: hypothesis re-selection + batch refinement (freeze in flight) | selection + optimisation |

Read the right-hand column. Of the seven steps that took us from 14.76 to 53.09, **exactly one was
"we trained a model"**. Two were bugs in our own code, two were thresholds in our own code, one was
an engineering patch, one was an honesty correction. The trained trio is the biggest single jump
(+14.06), but the cheapest points in this project were always **defects we had shipped ourselves**.

## 2.2 The seven acts

### Act 1 — the floor: 14.76

The first complete pipeline: YOLO detector → ByteTrack → PnLCalib camera → colour clustering for
teams. It emitted **no jersey numbers**, so under the identity gate every numbered ground-truth
player was unmatchable.

*In plain terms: we could draw dots on a map of the pitch, but every dot was anonymous — and this
benchmark pays almost entirely for names.*

### Act 2 — learning to read: 19.83 → 22.85

We attached the jersey chain following Koshkina & Elder (CVPRW 2024): legibility filter → torso
crop via a pose model → PARSeq reads the digits → the track's frames vote. It read numbers on only
**8.7% of tracks** — but because it abstained everywhere else, **not one of the 58 clips got
worse** (+5.07). Then re-linking broken tracks and sharing a confident number across fragments of
the same player added three more points.

*In plain terms: read a number only when you're sure; silence is free, a wrong guess is poison.*

### Act 3 — the nine-point bug: 33.20

The best day of the campaign cost zero GPU-hours. The per-frame votes were being tallied with a
bug: **crops the legibility filter had already rejected as unreadable still cast votes — at maximum
confidence.** Garbage was out-voting the good reads. Fixing the tally alone more than doubled the
density of usable reads (0.088 → 0.208 at unchanged precision, a **2.38×** improvement for zero
extra compute) and moved TEST-38 from 24.19 to 33.20, helping **35 of 38 clips**.

*In plain terms: the pipeline was already reading numbers well; a tallying bug was throwing the
good reads away. Finding it was worth more than any model we ever trained.*

### Act 4 — the honesty reset: 31.88

Before our first public upload we audited ourselves and found **two places where the pipeline was
reading the answer key**: the team→side assignment picked whichever mapping agreed with ground
truth, and the identity solver was handed the true roster of (team, jersey) slots. Both were
replaced with ground-truth-free versions — side from mean pitch position, roster from the
sequence's *own* OCR reads. The measured cost of honesty was about **2.5 points**. Our first public
number, 31.88, was *lower* than our local scores because it was clean.

*In plain terms: we caught our own pipeline cheating in two places, removed it, took the score hit,
and only then went on the public board.*

### Act 5 — repairing calibration: 33.37 → 35.40

Two fixes. **Gap fill**: when the camera estimator fails on a frame, re-fit from the run's own good
frames and interpolate across short gaps — recovered 9.6% of all test rows. Then the bigger one:
our own frame-rejection safety rule (reject a frame unless ≥8 players spanning ≥25 m are visible)
was discarding **96.4% of "dead" frames that actually had perfectly good homographies**. Relaxing
it to (3 players, 5 m) with an on-pitch plausibility check recovered **84,000 rows (18.3% of the
test set)**.

*In plain terms: the camera maths was mostly fine; our own overcautious sanity check was throwing
away correct answers.*

### Act 6 — two knobs: 39.02

No new code paths. The jersey vote-confidence floor moved 0.85 → 0.80 (+1.09) and the connector's
merge threshold was loosened 2× — made safe by a new **jersey-conflict guard**: never merge two
fragments whose confidently-read numbers disagree (+1.65). **43 of 49 test clips improved.**

### Act 7 — the trained trio: 53.09

The college A100 cluster unlocked training, and one data discovery powered most of it:
**SoccerNet-v3** — an older, freely licensed dataset (400 games, 33,986 annotated stills) — carries
**106,591 pixel-verified jersey-number labels** that nobody in the GSR context was using.

- **Detector**: YOLOv8s fine-tuned on GSR-train + SoccerNet-v3, with role as a class.
  mAP@0.5 **+20.7**, role-correct coverage 0.84 → 0.91, **58/58 clips improved recall**. Its first
  training run failed its pre-declared gate *by 20 detections* and is recorded as a FAIL; a second
  run under a freshly declared gate passed.
- **Jersey reader**: PARSeq fine-tuned on those 106k labels — but only the **43%** that survive our
  legibility filter, because annotators had labelled numbers on crops where no number is visible
  and training on those teaches hallucination. Per-crop precision **0.70 → 0.83** at the same emit
  rate. The filtered labels were the *entire* effect; a synthetic-digit pre-train measured
  **negative**.
- **Tracker**: a clean-room reimplementation of ExpansionIoU (their repo carries no licence, so
  nothing was read from it — built from the paper), with appearance embeddings from a CLIP model we
  trained on 160,070 crops / 60,040 identities. The honest decomposition: **appearance features
  +3.49, the expansion trick itself +0.37**, and an appearance-free rebuild is *worse* than the
  ByteTrack baseline we already had.

Bundle result: TEST-38 49.50 against a pre-declared gate of 40.0, then one test-49 run and upload
#5 → **53.09, +14.06 in one submission**. Leave-one-out prices: detector −9.06, reader −5.52,
tracker −2.09.

*In plain terms: with real GPUs we retrained the three core models. The trick wasn't a clever
architecture — it was finding 106k free jersey labels in an older dataset, being picky about which
of them to trust, and rebuilding an unlicensed tracker legally from its paper.*

### The interlude: the benchmark audit

Five of the benchmark's own sequences have **side-swapped ground truth** — the two teams' left/right
labels are inverted. All five are **second-half clips still carrying their game's first-half side
convention**: a diagnosable annotation-process error, not random noise. The leader's paper names
three (SNGS-126, 131, 197); we found **two more** (SNGS-092, SNGS-111) with an instrument built
without reading their code. 159 of 164 sequences obey within-(game, half) consistency; exactly
these five dissent.

Consequence: about **2.45 points of our score loss is benchmark noise no correct method can
recover** — a correct pipeline must predict the *true* side and be punished for it. Under corrected
ground truth our 53.09 is about **55.5**.

## 2.3 The v9 campaign: association → detection

On 2026-08-14 Sid set a single goal — beat 61.48 — and the association rebuild, long deferred as
expensive, started immediately. Nine work-packages in about 36 hours of wall-clock, most of them
on CPU.

**W1 — the data factory, and an arithmetic correction that redirected the campaign.** We ran our
own chain over all 57 training clips and built a labelled association dataset: 657,776 rows, 4,709
tracklets, 1,342 identities, **121,106 candidate tracklet pairs** (5,561 positive, 72,936 negative,
42,609 unlabelled), candidate-generator recall 0.985, leakage checked by code that raises rather
than by a claim in a report.

Then the number that changed the plan: **a perfect merge-only associator reaches only 57.32** — 4.2
short of the leader. Only an oracle that may **split as well as merge** reaches 61.59, because
13.3% of tracklets are less than 80% pure and must be cut, not glued. Also measured: the incumbent
connector's hard team/role/jersey gates **forbid 19.8% of all true merges** — the case for using
those cues as *features* in a learned model rather than as filters.

**W2 — a learned associator that beat the incumbent at ranking and lost at the job.** We trained
**TwixMetric**, a 72k-parameter pairwise transformer over tracklet coordinates in *metric pitch
space* (the novel input representation: camera motion removed, physical speeds bounded).

- **Gate 1 PASS**: validation pair-AP **0.6579 vs the incumbent's 0.5586** — 1.8× the measured
  noise floor. The learned ranking really is better.
- **Gate 2 FAIL**: merge precision at matched merge count **0.7260 against a 0.7992 bar** — below
  the incumbent itself. The ranking edge dies once you chain pairwise decisions into transitive
  merges (if A–B and B–C both merge, A–C is forced).
- **Novelty correction, recorded against ourselves**: a kinematics-only ablation scored AP 0.2707,
  so **metric coordinates do not beat image space standalone**. The defensible claim shrank to
  "a learned combination beats hard gating, at ranking". We wrote that down rather than keeping the
  bigger claim.
- **The arithmetic that re-aimed everything**: an oracle per-row splitter removes only 71.5% of
  contamination — **28.5% of it is interleaved and uncuttable**. Oracle split + perfect merge lands
  at **~60.40**, still 1.08 short of 61.48. Therefore **the remaining gap is not association at
  all; it is DetA**, and the campaign turned to detection.

**W3 — the detection census and the first shipped win.** Against 235,174 ground-truth rows we reach
**89.93% positionally but only 56.47% attribute-correct**. Ranked oracles: jersey +24.74, detection
misses +6.99, role+team +6.15, localisation +3.74, dead-frame veto +1.12, false-positive dedup
+0.18 (dead).

The shipped result: **track-attribute voting** — let a whole track vote on its own role, team and
jersey instead of leaving per-row disagreements in place, plus a writer repair (numbers had been
written only onto rows labelled "player", so goalkeeper and referee rows of named tracks kept
`jersey = null`; that repair alone was +1.05). **DEV-20 51.81 → 54.80 (+2.99 GS-HOTA, 20 of 20
clips helped, p = 1.9 × 10⁻⁶)**, at zero GPU cost. Recorded qualifier: role-voting *alone* loses
DetA (−0.13) because it kills minority-correct goalkeeper rows.

**W4 — the keeper-side rule, and the cleanest single result in the project.** Of the rows whose
team is wrong after voting, **62% are goalkeepers**: 42 of 52 keeper tracks are clustered into the
*wrong* team by kit colour. This is structural, not accidental — goalkeeper kits are *required* by
the laws of the game to differ from both outfield kits, so "which team's shirts does this look
like?" runs backwards for keepers.

The replacement rule reads only our own predictions and has **zero fitted parameters**: *a
goalkeeper's team is the side of the pitch he stands on.* The sign of his mean pitch-x matches his
true team **52 times out of 52**, where kit matches 10 of 52.

Result: **+2.27 DetA / +1.46 GS-HOTA, 18 clips helped, 0 hurt, p = 1.96 × 10⁻⁴** — and the score is
**bit-identical to the goalkeeper-team oracle at 15 significant figures**. A definitional rule
reading only our own output is exactly as good as being told the answer for that bucket. Row audit:
5,171 wrong→right, **0 right→wrong**.

**W5 to W8 — four registered failures, each with a mechanism.**

| Session | Idea | Verdict | Measured mechanism |
|---|---|---|---|
| W5 | referee/player repair | **not built** | the error bucket is the *central* referee, and 51 correctly-labelled central referee tracks (7,831 rows) sit in the same region: any central-corridor rule breaks 7,831 rows to fix 1,496 |
| W5 | second-keeper rules | **FAIL** | +0.17/+0.21 against a bar honestly *reduced* to +0.35 before scoring (arithmetic showed +0.60 unreachable) |
| W6 | name-borrowing across tracks | **FAIL** | the +8.15 coverage bound is **structurally phantom**: a perfect oracle over the whole candidate set is +0.23 DetA (3% of the bound) — the connector had already consumed the reachable links; 82.1% of touched rows were over-named |
| W7 | detector retrain | **REFUSED on evidence** | of 19,933 misses, **66.1% are confident detector boxes the chain discards downstream**; only 9.7% are true detector misses. **A perfect detector is worth +0.57 GS-DetA.** The earlier "+2 to +4.4 from a retrain" pricing had blamed the detector for chain losses |
| W7 | keep-untracked / min-hits recovery | **FAIL ×2** | recall and GS-DetA move in *opposite* directions: recovering 12,103 rows lost 7.1 DetA; recovered detections' attribute yield was **0.0%** — the rows the chain loses are exactly the rows the identity machinery cannot name |
| W8 | attach discarded boxes to existing tracks | **STOP before building** | the mechanism works (65–68% attribute yield) but the GT-perfect ceiling is +0.78 DetA and the best ground-truth-blind rule reaches +0.1553 against a +0.75 bar |

W7 also produced a **freeze-critical measurement**: same machine, same code, same md5-verified
weights, **8 days apart → +1.02 GS-DetA / +1.08 GS-HOTA of drift** from dependency churn alone.
That is larger than most of the effects we were chasing. Pairing new arms against on-record numbers
would have *manufactured* passes. Every subsequent comparison happens inside one freshly extracted
lineage.

**W9 — the freeze.** Configuration frozen to a timestamped JSON *before* any held-out read. All
four pre-registered TEST-38 gates passed: **+3.8423 GS-HOTA (bar 2.0), 38 of 38 clips helped (bar
24), p = 7.3 × 10⁻¹², DetA +3.4740 (bar 1.5)**. One test-49 run: **53.0846 → 55.4062** (DetA
39.32 → 41.28, AssA 71.67 → 74.38), 46 of 49 clips helped, p = 4.9 × 10⁻¹³.

Transfer decay measured rather than hand-waved: DEV +4.45 → TEST-38 +3.70 (83%) → test-49 +2.32
(52%), and the *reason* is measured too — the flags rewrite 12.6% of DEV team rows but only 7.7% of
test-49's, because the test split was already the most attribute-consistent. Also on record: the
pre-declared fail-response (drop the keeper rule) **would have been wrong** — voting alone fails the
TEST-38 DetA bar; the keeper repair supplies +2.38 DetA there.

The 55.41 package sits in `results/gsr_submission/`, sha256 recorded, awaiting Sid's upload as
submission #6 of 10.

## 2.4 The leaderboard shock

On 2026-08-15 a new entry appeared at the top of the public GSR test board:

| Rank | Team | GS-HOTA | GS-DetA | GS-AssA |
|---|---|---|---|---|
| 1 | **betterdays** | **68.30** | 55.86 | 83.53 |
| 2 | myyyy / Broadcast2Pitch | 61.48 | 48.47 | 78.00 |
| 3 | Metrica-Sports | 58.17 | — | — |
| 4 | Playbox & MIXI | 58.06 | — | — |
| 5 | KIST | 56.56 | — | — |
| 6 | vladika | 55.82 | — | — |
| 7 | tyler_durden | 55.59 | — | — |
| 8 | SJTU | 54.77 | — | — |
| 9 | **us (v6)** | **53.09** | 39.32 | 71.67 |

68.3 is **+6.8 above the best published system anywhere** and above the highest number in any
paper (SoccerMaster's foundation model, 64.1). It was a single submission, posted 2026-08-14, from
an account with no paper and no repository.

**The unmasking.** Sid logged into Codabench (the profile is visible to participants but not to the
public), and "betterdays" resolved to **Oleg Baishev**, a Master's student at HSE, GitHub handle
`PogChamper`. Recon from **public sources only** then produced an unusually complete picture:

- His **public** GSR pipeline scores **55.68 / DetA 42.14 / AssA 73.59** — i.e. our twin, within a
  quarter of a point of our v9 55.41. The 68.3 method is private.
- His stack: **DEIMv2-DINOv3 detector** at 896 px, **BoT-SORT**, **OSNet ReID** + k-means teams,
  ShuffleNet legibility + **ConvNeXt-Tiny OCR** — and crucially **no VLM at all**. Identity is not
  his edge, which is consistent with our own VLM closure (§2.5).
- **The tell**: his fork of **BroadTrack** (the EVS broadcast-camera tracker) received **14 commits
  on 2026-08-09/10 — four days before the submission** — fixing, in his own commit messages: BGR
  frames fed to an RGB-trained keypoint HRNet; a broken heatmap decode; un-inverted forward radial
  distortion; non-subpixel line points; TVCalib-mismatched preprocessing; and "process every frame
  at native resolution", plus SoccerNet-GSR run/evaluate scripts.

Since GS-HOTA matches in **pitch coordinates**, calibration quality gates DetA *and* AssA. A camera
overhaul halving reprojection error is exactly a **+10–14 DetA-shaped change** off a baseline with
those bugs. That inference — a private calibration overhaul, not a new identity model — became the
hypothesis that the whole v10 campaign was built to test. His bug list became our **free audit
checklist**.

Two disciplines held here: we used **only public sources**; and his fork of EVS-licensed code is
*his* legal exposure, and it cautioned us off touching his fork at all.

## 2.5 The v10 geometry campaign

Goal set by Sid after the shock: **70 GS-HOTA, use as much GPU as needed.** Five sessions so far.

**W1 — the VLM trial: a registered FAIL, and three decisive answers.** We ran **Qwen3-VL-8B**
(Apache-2.0, 17.5 GB) at scale against Sid's own 499-tracklet human annotation as ground truth.
Pre-registered bars: precision ≥ 0.90 on human-numbered crops, false-positive rate ≤ 5% on
human-confirmed-none.

- Best VLM regime: **0.889 precision on 9 of 13 emitted** — unresolvable at n = 9, recorded as
  **FAIL** exactly as registered. False positives on "no number" 3.65%: PASS.
- **Our fine-tuned PARSeq chain crushes the VLM per-crop**: chain-only-correct 324 vs
  VLM-only-correct 54, **McNemar p = 4.8 × 10⁻⁴⁸**. Handing the VLM our torso crops made it
  *worse*. A general vision-language model loses to a small fine-tuned specialist on 6-pixel digits.
- Two findings outlived the failure. **(1) 94.8% of the unnamed pool is genuine glyph absence** —
  there is no number to read — confirmed now at VLM scale as well as by human eyes. **(2) Given 12
  good stratified views of one clean identity, our shipped reader names 92.4% of numbered
  tracklets**, yet on DEV 309 of 478 tracklets carry no confident read. **The coverage loss is
  upstream**: fragmented tracks never assemble good views. Reading is not the bottleneck;
  association and geometry are.

**W2 — the calibration audit: our stack was already clean.** We audited our PnLCalib integration
against Baishev's six-defect list (transfer limited to the *list*, no code read). Result: **four of
six bugs absent** (BGR feed absent — our code is line-for-line the reference; preprocessing
mismatch absent; frame skipping absent; radial distortion not applicable because PnLCalib models
none, residual measured at −0.58 px with no cubic signature), and reduced resolution is **by
design** (960×540 is the training resolution; native is out-of-distribution for the network).

The one real defect was genuinely subtle, and the textbook fix for it is *harmful*: the standard
sub-pixel peak refinement makes things worse (0.477 → 0.509 m) because PnLCalib is trained on
integer-cell targets. The true defect is that the decoder returns the cell's **low edge**, a
uniform 0–2 px bias towards the origin. A **derived** (not tuned) half-cell shift removes it:
pixel residuals dx +1.50 / dy +2.34 → −0.53 / +0.34, median error **0.478 → 0.427 m (−10.6%)**,
end-to-end **+0.4896 GS-HOTA, 17 clips helped / 3 hurt, p = 0.0049**. The registered bar was +1.0
DetA. **It FAILED, and the flag stays off** — held as a candidate rider for the next freeze.

Strategic verdict: **Baishev's +12.6 is not transferable as bug fixes — his baseline was broken,
ours never was.** Only about 4% of it reproduces on our substrate.

**W3 — the tripod camera class, refuted at −7.39.** The surviving hypothesis was *model class*: a
temporally-consistent tripod camera model (BroadTrack, WACV 2025) instead of per-frame solving. We
built and ran BroadTrack **strictly as a measurement instrument**, server-side, never vendored
(EVS licence is non-commercial; a clean-room hygiene log records that the algorithm files were not
read).

| Instrument | BroadTrack | Ours | Verdict |
|---|---|---|---|
| Median accuracy | 1.0001 m | **0.4024 m** | FAIL (2.5× worse) |
| Dead-frame coverage | 100% recovered, at 0.858 m | — | raw PASS, accuracy-conditioned FAIL |
| Frame-to-frame jitter | **0.0146 m** | 0.381 m | **PASS by 26×** |
| End-to-end swap | **−7.39 GS-HOTA, 10/10 clips hurt** | — | refuted |

And a mechanism worth remembering: **AssA fell 9.3 points** — *smoothness buys nothing once it
costs accuracy*. Their tripod estimator also raises an assertion error on 9 of 10 GSR clips: 30
seconds is outside the method's design envelope. Recommendation adopted: **do not clean-room
BroadTrack.** The surviving ember was much smaller and ours: smoothing our *own* homographies
scored +0.43 in an exploratory probe.

**W4 — THE SENSITIVITY CURVE.** This is the session that matters most for §4.

Nobody has published the **calibration-error → GS-HOTA sensitivity curve**. We built it: perturb
the positions in our own submission through controlled camera errors, and separately through
*oracle* homographies fitted to ground truth, with association held frozen, and re-score. The null
arm reproduces the baseline to 10⁻¹⁴.

*Perturbation families* (DEV-20, flags ON; baseline 55.066):

| Injected error (median displacement) | GS-HOTA | Loss |
|---|---|---|
| none (baseline / null) | 55.066 | 0.00 |
| 0.12 m, independent per row | 54.934 | −0.13 |
| 0.29 m | 54.217 | −0.85 |
| 0.59 m | 51.581 | −3.49 |
| 1.18 m | 42.758 | −12.31 |
| 2.35 m | 24.376 | −30.69 |
| 3.53 m | 13.661 | −41.41 |
| 0.29 m, **whole-frame correlated** | 54.219 | −0.85 |
| 1.17 m, whole-frame correlated | 42.801 | −12.27 |

*Oracle counterfactuals*:

| Oracle | GS-HOTA | Gain | Meaning |
|---|---|---|---|
| **O_acc** — perfect camera on the frames we already solve | **61.32** | **+6.25** | pure per-frame accuracy |
| O_comp — perfect completeness (every found person projected) | 55.56 | +0.49 | the dead-frame prize |
| O_fill — perfect geometry on interpolated frames only | 55.55 | +0.48 | the gap-fill prize |
| **O_all** — perfect camera *and* perfect completeness | **61.89** | **+6.82** | total geometry headroom |

Five findings, all novel as far as our literature sweep can tell:

1. **The geometry axis holds +6.82 GS-HOTA on our substrate, and 92% of it (+6.25 of 6.82) is
   per-frame accuracy** — not completeness, not temporal smoothness. Every method in the
   sports-calibration literature that optimises completeness or smoothness is optimising the 8%.
2. **Calibration stops mattering at about 0.20 m mean error. We sit at 0.762 m.** An order of
   magnitude of chargeable error remains — the opposite of the "median is past the bottleneck"
   inference we had written down in our own innovation map. **That inference is refuted, by an
   instrument whose decision rule we had pre-declared.** The rule fired correctly and the
   conclusion was wrong for the right reasons.
3. **Frame-correlated error and independent error cost the same per metre.** Camera error and
   detection error are the same currency. This is not obvious and, to our knowledge, nowhere
   published.
4. **Association gains more from geometry than detection does** (+8.24 AssA vs +4.75 DetA)
   *even with association frozen*. Better camera → better tracking, invisibly.
5. The registered temporal arms **FAILED** (camera-space smoothing: −5.77 and −0.41), and the root
   cause was measured: the per-frame camera decomposition is **degenerate** — camera position
   trades off against focal length, giving 1.2–2.6 m of frame-to-frame noise in parameters that
   describe the *same* mapping. PnLCalib also emits **non-square pixels** (fy/fx 0.94–0.97), so any
   code assuming square pixels is silently wrong. The identical smoother applied to the
   **homography (mapping) space** scored +0.5375. The transferable design law: **smooth the map,
   not the camera.**

**W5 — the prize we were throwing away.** PnLCalib does not emit one camera per frame; it emits a
pool of about **16 hypotheses** per frame, and our acceptance gate picks one. How often does the
gate pick the most accurate one? **25.4% of the time. Median accuracy rank: 4.** Simply choosing
the best admissible hypothesis already in the pool is worth a predicted **+2.32 GS-HOTA with no new
correspondences whatsoever.**

So all arms aimed at *selection*, not at new geometry:

| Arm | Method | DEV-20 GS-HOTA (flags ON) |
|---|---|---|
| control | shipped decode | 55.066 |
| R | RTS smoothing in mapping space (W4's arm E, properly registered) | +0.538 |
| B | full-clip batch refinement in mapping space | +1.685 |
| **D** | **temporal hypothesis re-selection → batch refinement** | **+2.102 → 57.168** |

Arm D: **+2.1024 GS-HOTA (DetA +1.57, AssA +2.81, LocA +0.94), 19 clips helped, 1 hurt,
p = 5.0 × 10⁻⁵.** Calibration accuracy: median 0.465 → 0.396 m, and the **mean collapses 1.600 →
0.590 m** — the method's real work is killing catastrophic frames, not shaving good ones.

Two riders on record: the 2024 winner's ±2 m clamp — a published guard we had adopted uncritically
— **actively hurts** here, because it blocks exactly the frames where the per-frame solve is
catastrophically wrong and the batch rescues it. And the per-clip **radial distortion term is real
and still unmodelled**: a **+1.88 px residual on the outer ring** of the image.

Arm D captures **33.6% of the +6.25 accuracy oracle, on CPU alone, with no new model and no new
correspondences.** The remaining ~66% is the honest headroom of the geometry axis (§5.1).

Status right now: the v10 freeze is in flight. Arm D needs a same-window TEST-38 confirmation,
which needs candidate-cache extraction on the cluster. An in-flight, not-yet-registered sub-gate
run stacks D with the half-cell decode fix at **+2.47 flags-OFF** on DEV-20, suggesting the two
riders compose; that number is not in the record yet and should not be quoted as a result.

## 2.6 What the other teams did

Our recon record, so you can answer "how does your approach compare to theirs?"

| Team | Score | Detector | Tracker | Calibration | Identity |
|---|---|---|---|---|---|
| **betterdays** (Baishev, HSE) | 68.30 test | DEIMv2-DINOv3 @896 | BoT-SORT | private overhaul; public fork = BroadTrack + 14 bug-fix commits | ShuffleNet legibility + ConvNeXt-Tiny OCR, OSNet ReID, k-means teams — **no VLM** |
| **Broadcast2Pitch / KIST** | 61.48 test, 63.90 challenge (2025 winner) | YOLOX | Deep-EIoU + OSNet | EfficientNetV2-S + U-Net, 97 keypoints + 18 lines, DLT + Levenberg–Marquardt | **LLaMA-3.2-Vision** fine-tuned 1 epoch; IDATR split-and-merge |
| **Broadcast2Pitch++** (June 2026) | 62.56 test | same | + depth-aware cost (Depth-Anything-V2) | same | **IDASTR soft-merge**: merge cost = ReID cosine + 0.5 × spatial + 0.2 × identity *disagreement* |
| **Constructor.Tech** | 63.81 challenge (2024 winner) | YOLOv5m @1080p, 66k images | DeepSORT **in pitch coordinates** + orientation gate | **SegFormer regressing 7 camera parameters directly** (not a homography), 22k real + 40k synthetic images, ResNet18 74-keypoint refinement, Savitzky-Golay smoothing with ±2°/±2 m clamps | ResNet18 two-digit-head classifier, **no OCR**; OSNet team embeddings over 111 kit classes |
| **lianyou** | 62.76 challenge, highest AssA on the board (85.33) | YOLOv12 + RT-DETR for 27 line categories | **IOF-Tracker with Farnebäck optical flow** feeding the Kalman filter | radial-distortion correction + PnLCalib, camera reduced to 4 DOF, temporally smoothed | CLIP-ReIdent + a **6-layer transformer aggregating tracklet features** + fine-tuned ViT-L/14 for numbers |
| **Playbox & MIXI** | 61.64 challenge | RF-DETR | BoT-SORT + GTA-link | **BroadTrack extended with optical-flow parameter propagation** | exact Koshkina recipe (ViTPose torso + PARSeq); goalkeeper by penalty-area position + hill-climbing over the whole assignment |
| **SoccerMaster** (arXiv:2512.11016) | 64.1 (highest published anywhere) | one multi-task **vision foundation model** for all soccer tasks | — | — | multi-task pretraining, automated data curation pipeline |
| **us** | 53.09 board / 55.41 packaged / 57.17 DEV | YOLOv8s fine-tuned | clean-room EIoU + our CLIP | PnLCalib + gate repair + **hypothesis re-selection + batch refinement** | legibility → torso → fine-tuned PARSeq → **confidence-weighted vote**; MILP solver; keeper-side rule |

Three observations worth saying out loud in a viva:

1. **Nobody's winning system used more than one GPU.** The 2024 winner trained everything on a
   single A100; the 2025 winner on a single RTX 4090, including a LLaMA-Vision fine-tune. Compute
   is not the moat here.
2. **No GSR participant in 2024 or 2025 released training code or weights.** Every comparison in
   this document required reimplementation or careful reading. The 61.48 has never been
   independently reproduced in public — the one logged attempt reached 48.32 against pre-fix code.
3. **The published field is converging on the same recipe** (fine-tuned detector, appearance
   tracker, keypoint calibration, CLIP or VLM identity). The 68.3 is not on that curve, which is
   why it matters.

---

# 3. The papers we drew from

Each entry: what it is, **its real abstract as fetched from arXiv on 2026-08-17**, what we took in
one plain sentence, and what happened. Where an abstract could not be fetched (no arXiv record) it
is explicitly marked **[PARAPHRASE — from our own recon notes, not the authors' words]**.

**Fetch record: 28 of 28 attempted arXiv abstracts retrieved successfully.** Four sources have no
arXiv record and are paraphrased: Broadcast2Pitch (WACV 2026), Broadcast2Pitch++ (Research Square
preprint), Grad (CVPRW 2025), and Qwen3-VL-8B (model card only).

## 3.1 Summary table

| # | Paper | Venue / year | arXiv | We took | Outcome |
|---|---|---|---|---|---|
| 1 | SoccerNet-GSR | CVPRW 2024 | 2404.11335 | the task, the data, the metric | **the whole project** |
| 2 | HOTA | IJCV 2021 | 2009.07736 | DetA/AssA decomposition as diagnosis | **adopted** |
| 3 | ByteTrack | ECCV 2022 | 2110.06864 | first tracker | adopted, later replaced |
| 4 | Deep-EIoU | WACVW 2024 | 2306.13074 | Kalman-free association | **adapted (clean-room)**; headline trick worth +0.37, appearance +3.49 |
| 5 | GTA-Link | ACCVW 2024 | 2411.08216 | tracklet connector | **adopted**; its splitter half **retired as inert** |
| 6 | CAMELTrack | 2025 | 2505.01257 | learned association | **refuted on our substrate**: −13.6 to −20.2 |
| 7 | TWiX | 2024 | 2403.08018 | coordinates-only pairwise transformer | **adapted → TwixMetric; FAILED gates 2 and 3** |
| 8 | MOTIP | 2024 | 2403.16848 | in-context ID prediction | licence-checked, **not built** (W2 arithmetic closed the axis) |
| 9 | SUSHI | CVPR 2023 | 2212.03038 | hierarchical graph association | scoped, **never run** |
| 10 | DanceTrack | CVPR 2022 | 2111.14690 | the same-appearance problem framing | context |
| 11 | Koshkina & Elder | CVPRW 2024 | 2405.13896 | the entire jersey chain | **adopted**; our aggregation fix is the delta |
| 12 | PARSeq | ECCV 2022 | 2207.06966 | the text recogniser | **adopted + fine-tuned**: 0.70 → 0.83 per-crop precision |
| 13 | CLIP | ICML 2021 | 2103.00020 | appearance backbone | **adopted** as association cue, **refuted** as a namer |
| 14 | ArcFace | CVPR 2019 | 1801.07698 | identity loss | adopted |
| 15 | PRTreID | MMSports 2023 | 2401.09942 | the baseline embedding we replaced | replaced (+1.82 retrieval → +0.33 GS-HOTA) |
| 16 | PnLCalib / NBJW | 2024 | 2404.08401 | the calibrator | **adopted**; measured 12% more accurate than the leader's |
| 17 | TVCalib | ACCV 2022 | 2207.11709 | calibration-as-camera-estimation framing | context; a defect class in the leader's checklist |
| 18 | BHITK | 2023 | 2311.10361 | Kalman over homographies | **the principled version of our gap fill**; motivated W4 |
| 19 | BroadTrack | WACV 2025 | 2412.01721 | tripod camera model | **REFUTED on GSR: −7.39 GS-HOTA** |
| 20 | Falaleev & Chen (sportlight) | MMSports 2024 | 2410.07401 | denser correspondences | **not built** — W5 found 37% unspent in the pool we already had; repo has no licence |
| 21 | Central-view geometry | 2025 | 2504.20052 | circle → line conversion | scoped; only 24 central-view dead frames remain |
| 22 | Tent (BN/TTA) | ICLR 2021 | 2006.10726 | test-time adaptation | **REFUTED in-domain: −17.56, 0/10 clips helped** |
| 23 | Evidential Deep Learning | NeurIPS 2018 | 1806.01768 | Dirichlet abstention | **FAILED by one sequence**; one constant refuted |
| 24 | Constructor.Tech | CVPRW 2025 | 2504.06357 | camera-parameter regression; the ±2 m clamp | clamp adopted, then **measured harmful for a batch solve** |
| 25 | SoccerNet 2025 results | 2025 | 2508.19182 | the competitive landscape | recon |
| 26 | SoccerNet 2026 results | 2026 | 2607.07320 | live-server landscape, pivot options | recon |
| 27 | SoccerMaster | 2025 | 2512.11016 | the highest published GS-HOTA (64.1) | context: 68.3 exceeds every paper |
| 28 | Qwen2.5-VL | 2025 | 2502.13923 | the VLM family we tested | **REFUTED for identity: p = 4.8 × 10⁻⁴⁸ against our chain** |
| 29 | Broadcast2Pitch | WACV 2026 | *(none)* | the ablation table that steered the campaign | **three of four claimed edges measured away** |
| 30 | Broadcast2Pitch++ | preprint 2026 | *(none)* | IDASTR soft-merge | corroborates our W1 hard-gate finding |
| 31 | Grad | CVPRW 2025 | *(none)* | illegible crops as free negatives | **refuted: teacher-imitation** |

## 3.2 The entries

### 1. SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap
Somers, Joos, Cioppa, Giancola et al. — CVPRW 2024 — **arXiv:2404.11335**

> Tracking and identifying athletes on the pitch holds a central role in collecting essential
> insights from the game, such as estimating the total distance covered by players or understanding
> team tactics. This tracking and identification process is crucial for reconstructing the game
> state, defined by the athletes' positions and identities on a 2D top-view of the pitch, (i.e. a
> minimap). However, reconstructing the game state from videos captured by a single camera is
> challenging. It requires understanding the position of the athletes and the viewpoint of the
> camera to localize and identify players within the field. In this work, we formalize the task of
> Game State Reconstruction and introduce SoccerNet-GSR, a novel Game State Reconstruction dataset
> focusing on football videos. SoccerNet-GSR is composed of 200 video sequences of 30 seconds,
> annotated with 9.37 million line points for pitch localization and camera calibration, as well as
> over 2.36 million athlete positions on the pitch with their respective role, team, and jersey
> number. Furthermore, we introduce GS-HOTA, a novel metric to evaluate game state reconstruction
> methods. Finally, we propose and release an end-to-end baseline for game state reconstruction,
> bootstrapping the research on this task. Our experiments show that GSR is a challenging novel
> task, which opens the field for future research.

**We took:** the task definition, the dataset and the GS-HOTA metric — everything this project is
scored on.
**What happened:** adopted wholesale, and then *audited*: we found six ground-truth errors in it,
two of which no published paper reports (§2.2, §4.3).

### 2. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking
Luiten, Ošep, Dendorfer, Torr et al. — IJCV 2021 — **arXiv:2009.07736**

> Multi-Object Tracking (MOT) has been notoriously difficult to evaluate. Previous metrics
> overemphasize the importance of either detection or association. To address this, we present a
> novel MOT evaluation metric, HOTA (Higher Order Tracking Accuracy), which explicitly balances the
> effect of performing accurate detection, association and localization into a single unified
> metric for comparing trackers. HOTA decomposes into a family of sub-metrics which are able to
> evaluate each of five basic error types separately, which enables clear analysis of tracking
> performance. We evaluate the effectiveness of HOTA on the MOTChallenge benchmark, and show that
> it is able to capture important aspects of MOT performance not previously taken into account by
> established metrics. Furthermore, we show HOTA scores better align with human visual evaluation
> of tracking performance.

**We took:** the DetA/AssA split as a *diagnostic instrument* — every session in this project reads
both, not just the headline.
**What happened:** confirmed and heavily used. It is how we know that our geometry gain of +2.10
lands mostly on association (+2.81 AssA vs +1.57 DetA), which is otherwise invisible.

### 3. ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Zhang, Sun, Jiang, Yu et al. — ECCV 2022 — **arXiv:2110.06864**

> Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in
> videos. Most methods obtain identities by associating detection boxes whose scores are higher
> than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown
> away, which brings non-negligible true object missing and fragmented trajectories. To solve this
> problem, we present a simple, effective and generic association method, tracking by associating
> almost every detection box instead of only the high score ones. For the low score detection
> boxes, we utilize their similarities with tracklets to recover true objects and filter out the
> background detections. When applied to 9 different state-of-the-art trackers, our method achieves
> consistent improvement on IDF1 score ranging from 1 to 10 points. […] ByteTrack also achieves
> state-of-the-art performance on MOT20, HiEve and BDD100K tracking benchmarks.

**We took:** our first tracker, and the "use the weak detections too" principle.
**What happened:** adopted, then replaced by EIoU — but with a sting in the tail. In v9-W7 we
measured that ByteTrack's own `update_with_detections` discards **8% of confident detection boxes**
by re-matching its Kalman boxes at IoU ≥ 0.5, and that discard accounts for **66% of our detection
misses**. It is also doing real false-positive suppression (40.1% of discards are phantoms), so it
is not simply a bug — it is a trade-off nobody had priced.

### 4. Iterative Scale-Up ExpansionIoU and Deep Features Association (Deep-EIoU)
Huang, Yang, Sun, Kim — WACVW 2024 — **arXiv:2306.13074**

> Deep learning-based object detectors have driven notable progress in multi-object tracking
> algorithms. Yet, current tracking methods mainly focus on simple, regular motion patterns in
> pedestrians or vehicles. This leaves a gap in tracking algorithms for targets with nonlinear,
> irregular motion, like athletes. Additionally, relying on the Kalman filter in recent tracking
> algorithms falls short when object motion defies its linear assumption. To overcome these issues,
> we propose a novel online and robust multi-object tracking approach named deep ExpansionIoU
> (Deep-EIoU) […] we abandon the use of the Kalman filter and leverage the iterative scale-up
> ExpansionIoU and deep features for robust tracking in sports scenarios. […] achieving a score of
> 77.2% HOTA on the SportsMOT dataset and 85.4% HOTA on the SoccerNet-Tracking dataset.

**We took:** the Kalman-free association idea — rebuilt **clean-room from the paper**, because
their repository ships no licence file.
**What happened:** **adapted, with an honest decomposition that contradicts the title.** On DEV-20:
appearance features **+3.49**, the expansion trick **+0.37**, iterative scale-up **−0.52**, and an
appearance-free rebuild is **−1.85 versus the ByteTrack baseline**. The half named in the paper's
title is worth a tenth of the half that is not. Held-out TEST-38: +2.51, p = 0.00092.

A second, counter-intuitive finding: EIoU **fragments 1.8× more** than ByteTrack (11.47 vs 6.52
fragments per identity) and is **purer** (row purity 0.9416 vs 0.8969). For us that is *better*,
because downstream we assign one jersey per merged tracklet: a contaminated tracklet mislabels
hundreds of rows, while an extra clean fragment is exactly what the connector exists to re-merge.
**"Better tracker" is a function of your objective, not a property of the tracker.**

### 5. GTA: Global Tracklet Association for Multi-Object Tracking in Sports
Sun, Huang, Yang, Jiang — ACCVW 2024 — **arXiv:2411.08216**

> Multi-object tracking in sports scenarios has become one of the focal points in computer vision
> […] challenges remain, such as accurately re-identifying players upon re-entry into the scene and
> minimizing ID switches. In this paper, we propose an appearance-based global tracklet association
> algorithm designed to enhance tracking performance by splitting tracklets containing multiple
> identities and connecting tracklets seemingly from the same identity. This method can serve as a
> plug-and-play refinement tool for any multi-object tracker […] achieved a new state-of-the-art
> performance on the SportsMOT dataset with HOTA score of 81.04%.

**We took:** the tracklet connector — cluster finished tracklets by appearance and merge.
**What happened:** **adopted (+1.33 when it landed, 51 of 58 clips helped)** — and its other half
**retired on measurement**. The splitter flipped sign from pilot (+0.95) to full run (−0.19), and
the paper's ε = 0.30 produces **2 splits out of 4,709 tracklets** on same-kit football. We also
measured that the connector **destroys part of its own ceiling**: at 0.5432 merge precision its
wrong merges lock in contamination that no later merge-only pass can undo.

### 6. CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online Multi-Object Tracking
Somers, Standaert, Joos, Alahi — 2025 — **arXiv:2505.01257**

> Online multi-object tracking has been recently dominated by tracking-by-detection (TbD) methods,
> where recent advances rely on increasingly sophisticated heuristics […] the extensive usage of
> human-crafted rules for temporal associations makes these methods inherently limited […] we
> introduce CAMEL, a novel association module for Context-Aware Multi-Cue ExpLoitation, that learns
> resilient association strategies directly from data […] CAMELTrack, achieves state-of-the-art
> performance on multiple tracking benchmarks.

**We took:** the strongest published *learned* association model, tried zero-shot and then
retrained from scratch on our data.
**What happened:** **refuted on our substrate, with the mechanism nailed.** Their released
checkpoints' appearance tokenizer is `nn.Linear(128, 512)` fitted to **KPReID part embeddings —
5 body parts + 1 global, 128-d each, with visibility scores**. Our cache is a **single global 256-d
CLIP vector**. You cannot reshape one basis into the other, so every checkpoint silently fell
through its own `drop_app` path and ran **appearance-less**: **−13.6 to −20.2 GS-HOTA, 0 of 20
clips helped**. Measured three ways (purity collapse; nothing left for merge-only recovery; the
connector correctly refusing to merge, 26 merges instead of 1,380). A from-scratch retrain on our
cues **learned the task** (validation association accuracy 0.69 → 0.92 on held-out sequences) and
still scored −18.6 end to end — trained on ground-truth boxes, never on detector output.

We reported this as **"we measured CAMELTrack-*without-its-appearance-cue*"**, not as
"CAMELTrack fails". That distinction is the difference between a result and a smear.

### 7. Learning Data Association for Multi-Object Tracking using Only Coordinates (TWiX)
Miah, Bilodeau, Saunier — 2024 — **arXiv:2403.08018**

> We propose a novel Transformer-based module to address the data association problem for
> multi-object tracking. From detections obtained by a pretrained detector, this module uses only
> coordinates from bounding boxes to estimate an affinity score between pairs of tracks extracted
> from two distinct temporal windows. This module, named TWiX, is trained on sets of tracks with
> the objective of discriminating pairs of tracks coming from the same object from those which are
> not. Our module does not use the intersection over union measure, nor does it requires any motion
> priors or any camera motion compensation technique. By inserting TWiX within an online cascade
> matching pipeline, our tracker C-TWiX achieves state-of-the-art performance on the DanceTrack and
> KITTIMOT datasets […]

**We took:** the architecture (a pairwise transformer over box trajectories with inter-pair
attention), reimplemented as **TwixMetric** with one deliberate change: we fed it **metric pitch
coordinates instead of image coordinates**, because our calibration removes camera motion.
**What happened:** **Gate 1 passed, Gates 2 and 3 failed** (§2.3). And the novelty claim was
corrected against us: a kinematics-only ablation showed metric coordinates do **not** beat image
space standalone. Design note banked: we kept their inter-pair attention and **dropped their
NormCoords**, because rescaling coordinates to [−1, 1] destroys the absolute metric scale that was
the whole point.

### 8. Multiple Object Tracking as ID Prediction (MOTIP)
Gao, Qi, Wang — 2024 — **arXiv:2403.16848**

> Multi-Object Tracking (MOT) has been a long-standing challenge in video understanding. A natural
> and intuitive approach is to split this task into two parts: object detection and association.
> Most mainstream methods employ meticulously crafted heuristic techniques […] we introduce a new
> perspective that treats Multiple Object Tracking as an in-context ID Prediction task,
> transforming the aforementioned object association into an end-to-end trainable task. […] MOTIP
> directly decodes the ID labels for current detections […] achieves state-of-the-art results
> across multiple benchmarks by solely leveraging object-level features as tracking cues.

**We took:** licence check (Apache-2.0, usable directly) and the framing of association as
sequence prediction.
**What happened:** **never built.** The W2 oracle arithmetic showed that even a *perfect* associator
lands at ~60.4, short of the target, so the axis was closed before spending GPU-hours. Deciding not
to run an experiment on measured grounds is a result too.

### 9. Unifying Short and Long-Term Tracking with Graph Hierarchies (SUSHI)
Cetintas, Brasó, Leal-Taixé — CVPR 2023 — **arXiv:2212.03038**

> Tracking objects over long videos effectively means solving a spectrum of problems, from
> short-term association for un-occluded objects to long-term association for objects that are
> occluded and then reappear […] we question the need for hybrid approaches and introduce SUSHI, a
> unified and scalable multi-object tracker. Our approach processes long clips by splitting them
> into a hierarchy of subclips […] We leverage graph neural networks to process all levels of the
> hierarchy […] we obtain significant improvements over state-of-the-art on four diverse datasets.

**We took:** the hierarchical offline framing (which matches our non-causal setting exactly).
**What happened:** scoped, licence-checked (MIT), **never run** — same reason as MOTIP. It remains
the strongest candidate if the association axis is ever reopened.

### 10. DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion
Sun, Cao, Jiang, Yuan — CVPR 2022 — **arXiv:2111.14690**

> A typical pipeline for multi-object tracking (MOT) is to use a detector for object localization,
> and following re-identification (re-ID) for object association. This pipeline is partially
> motivated by […] biases in existing tracking datasets, where most objects tend to have
> distinguishing appearance […] we propose a large-scale dataset for multi-human tracking, where
> humans have similar appearance, diverse motion and extreme articulation. […] we name it
> "DanceTrack". We expect DanceTrack to provide a better platform to develop more MOT algorithms
> that rely less on visual discrimination and depend more on motion analysis.

**We took:** the *framing*. Eleven players in identical kit is the same problem as dancers in
identical costumes, and the DanceTrack lineage is where appearance-free associators live.
**What happened:** it re-priced the association attack from "thesis-scale" to "hours-to-days on one
A100", which is what unlocked v9. Our own measurement then partially undercut the borrowed
intuition (see TWiX above) — that is what testing a borrowed intuition looks like.

### 11. A General Framework for Jersey Number Recognition in Sports Video
Koshkina & Elder — CVPRW 2024 — **arXiv:2405.13896**

> Jersey number recognition is an important task in sports video analysis, partly due to its
> importance for long-term player tracking. It can be viewed as a variant of scene text
> recognition. However, there is a lack of published attempts to apply scene text recognition
> models on jersey number data. Here we introduce a novel public jersey number recognition dataset
> for hockey and study how scene text recognition methods can be adapted to this problem. We
> address issues of occlusions and assess the degree to which training on one sport (hockey) can be
> generalized to another (soccer). For the latter, we also consider how jersey number recognition
> at the single-image level can be aggregated across frames to yield tracklet-level jersey number
> labels. We demonstrate high performance on image- and tracklet-level tasks, achieving 91.4%
> accuracy for hockey images and 87.4% for soccer tracklets.

**We took:** the whole chain — legibility filter → torso crop → scene-text model → tracklet
aggregation.
**What happened:** **adopted, and then improved at the seam the paper leaves open.** Our
contributions on top: the aggregation bug fix (2.38× read density for zero compute); a
confidence-weighted vote instead of a majority vote; a retrained PARSeq on filtered SoccerNet-v3
labels. Also on record: their pipeline's licence is CC BY-NC 3.0, which constrains what we could
reuse.

### 12. Scene Text Recognition with Permuted Autoregressive Sequence Models (PARSeq)
Bautista & Atienza — ECCV 2022 — **arXiv:2207.06966**

> Context-aware STR methods typically use internal autoregressive (AR) language models (LM).
> Inherent limitations of AR models motivated two-stage methods which employ an external LM. The
> conditional independence of the external LM on the input image may cause it to erroneously
> rectify correct predictions […] Our method, PARSeq, learns an ensemble of internal AR LMs with
> shared weights using Permutation Language Modeling. It unifies context-free non-AR and
> context-aware AR inference, and iterative refinement using bidirectional context. […] PARSeq is
> optimal on accuracy vs parameter count, FLOPS, and latency […] robust on arbitrarily-oriented
> text.

**We took:** the recogniser itself, fine-tuned on 106k filtered football labels.
**What happened:** **our second-largest single component** (leave-one-out −5.52). Per-crop precision
0.7022 → 0.8344 at identical emit rate; end-to-end +3.66 on DEV-20 (p = 0.00021), with **89% of the
movement in DetA** — exactly where a jersey lever must act under an identity-gated metric. A
by-product refutation: we tried to replace PARSeq's decode with a simple MLP over pooled features
and got 0.4935 against the trunk's own 0.8344, because **an MLP smooths the sharp combinatorial
`d₀×10 + d₁` map that permuted autoregressive decoding represents natively.**

### 13. Learning Transferable Visual Models From Natural Language Supervision (CLIP)
Radford, Kim, Hallacy, Ramesh et al. — ICML 2021 — **arXiv:2103.00020**

> State-of-the-art computer vision systems are trained to predict a fixed set of predetermined
> object categories. This restricted form of supervision limits their generality […] We demonstrate
> that the simple pre-training task of predicting which caption goes with which image is an
> efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400
> million (image, text) pairs […] After pre-training, natural language is used to reference learned
> visual concepts […] enabling zero-shot transfer of the model to downstream tasks. […] we match
> the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the
> 1.28 million training examples it was trained on.

**We took:** the ViT-B/16 backbone as the basis for a 256-d football appearance embedding, trained
on 160,070 crops / 60,040 identities (44.7× the identities in GSR-train alone).
**What happened:** **adopted where it works, refuted where it does not.** As an *appearance term
inside association*: **+3.49 GS-HOTA**. As a *namer* (retrieval against a gallery): saturated at
0.65–0.71 top-1 on same-kit broadcast; three pre-registered naming arms failed in three days. Two
independent architectures agreed that GSR-train's 1,343 identities cannot build a competitive
embedding — **scale was the fix**. And a scar worth repeating: **CLIP's cosine space is ~4.8× wider
than PRTreID's**, so at the inherited threshold τ = 0.080 the swap read as a 4-point *failure*
until τ was re-swept to 0.450. **Swap an embedding model and every distance threshold downstream is
now wrong.**

### 14. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Deng, Guo, Yang, Xue — CVPR 2019 — **arXiv:1801.07698**

> Recently, a popular line of research in face recognition is adopting margins in the
> well-established softmax loss function to maximize class separability. In this paper, we first
> introduce an Additive Angular Margin Loss (ArcFace), which not only has a clear geometric
> interpretation but also significantly enhances the discriminative power. Since ArcFace is
> susceptible to the massive label noise, we further propose sub-center ArcFace […] Extensive
> experiments demonstrate that ArcFace can enhance the discriminative feature embedding as well as
> strengthen the generative face synthesis.

**We took:** the identity head loss for our encoder.
**What happened:** adopted without drama — the encoder trained and its retrieval improved (+1.82
identity mAP over the shipped PRTreID floor). The interesting part is what that bought end-to-end:
**+0.33 GS-HOTA when swapped in directly**, versus +3.49 when used as an association cue. *Where*
you spend a representation matters more than how good it is.

### 15. Multi-task Learning for Joint Re-identification, Team Affiliation, and Role Classification (PRTreID)
Mansourian, Somers, De Vleeschouwer, Kasaei — MMSports 2023 — **arXiv:2401.09942**

> Effective tracking and re-identification of players is essential for analyzing soccer videos.
> But, it is a challenging task due to the non-linear motion of players, the similarity in
> appearance of players from the same team, and frequent occlusions. […] a multi-purpose part-based
> person representation method, called PRTreID, is proposed that performs three tasks of role
> classification, team affiliation, and re-identification, simultaneously. In contrast to available
> literature, a single network is trained with multi-task supervision to solve all three tasks,
> jointly. […] The proposed tracking method outperforms all existing tracking methods on the
> challenging SoccerNet tracking dataset.

**We took:** the baseline embedding shipped with the GSR baseline, and the multi-task idea.
**What happened:** replaced by our CLIP encoder. Its part-based cousin (KPReID) is also the
representation CAMELTrack demanded and we could not supply — the input-contract failure of §3.2/6.

### 16. PnLCalib: Sports Field Registration via Points and Lines Optimization
Gutiérrez-Pérez & Agudo — 2024 — **arXiv:2404.08401**

> Camera calibration in broadcast sports videos presents numerous challenges for accurate sports
> field registration due to multiple camera angles, varying camera parameters, and frequent
> occlusions of the field. Traditional search-based methods depend on initial camera pose
> estimates, which can struggle in non-standard positions and dynamic environments. In response, we
> propose an optimization-based calibration pipeline that leverages a 3D soccer field model and a
> predefined set of keypoints […] a novel refinement module that improves initial calibration by
> using detected field lines in a non-linear optimization process. This approach outperforms
> existing techniques in both multi-view and single-view 3D camera calibration tasks […]

**We took:** our camera calibrator, per frame.
**What happened:** **adopted, and it beat the leader's**: head-to-head on identical rows, our
calibration is **12% more accurate** (median 0.475 m vs 0.536 m). But we also found three things
its users should know: it emits **non-square pixels** (fy/fx 0.94–0.97); its heatmap decoder returns
the **low edge** of a cell, a systematic sub-pixel bias; and it emits about **16 hypotheses per
frame**, of which our gate picked the most accurate only **25.4%** of the time. That last one was
worth +2.10 GS-HOTA (§2.5).

### 17. TVCalib: Camera Calibration for Sports Field Registration in Soccer
Theiner & Ewerth — ACCV 2022 — **arXiv:2207.11709**

> Sports field registration in broadcast videos is typically interpreted as the task of homography
> estimation, which provides a mapping between a planar field and the corresponding visible area of
> the image. In contrast to previous approaches, we consider the task as a camera calibration
> problem. First, we introduce a differentiable objective function that is able to learn the camera
> pose and focal length from segment correspondences (e.g., lines, point clouds), based on
> pixel-level annotations for segments of a known calibration object. […] Compared to the typical
> solution, which subsequently refines an initial estimation, our solution does it in one step.

**We took:** the conceptual framing — treat this as camera calibration, not homography fitting — and
the knowledge that "TVCalib-mismatched preprocessing" is a real defect class (it is item 5 on the
leader's bug list).
**What happened:** context only; we did not run it. Our W4 result partially *inverts* its framing
for our purposes: the per-frame camera decomposition is degenerate, so **smoothing the homography
beats smoothing the camera parameters** on our substrate.

### 18. Video-based Sequential Bayesian Homography Estimation for Soccer Field Registration (BHITK)
Claasen & de Villiers — 2023 — **arXiv:2311.10361**

> A novel Bayesian framework is proposed, which explicitly relates the homography of one video
> frame to the next through an affine transformation while explicitly modelling keypoint
> uncertainty. The literature has previously used differential homography between subsequent
> frames, but not in a Bayesian setting. In cases where Bayesian methods have been applied, camera
> motion is not adequately modelled, and keypoints are treated as deterministic. The proposed
> method, Bayesian Homography Inference from Tracked Keypoints (BHITK), employs a two-stage Kalman
> filter and significantly improves existing methods. […] It enables less sophisticated and less
> computationally expensive methods to outperform the state-of-the-art approaches in most
> homography evaluation metrics.

**We took:** the principle that our dead-frame linear interpolation is a crude version of a
principled filter over homographies, with uncertainty.
**What happened:** **it is the reason W4's mapping-space arm exists, and it was right where the
camera-space arms were wrong.** Camera-space RTS smoothing failed (−5.77 / −0.41); the identical
smoother in mapping space passed every numeric bar (+0.5375). BHITK's choice of state space —
homographies, not camera parameters — is the part that transferred.

### 19. BroadTrack: Broadcast Camera Tracking for Soccer
Magera, Hoyoux, Barnich, Van Droogenbroeck — WACV 2025 — **arXiv:2412.01721**

> Camera calibration and localization […] enables many applications in the context of soccer
> broadcasting […] the research community has typically focused on single-view calibration methods
> […] but leaving all temporal aspects, if considered at all, to general-purpose tracking or
> filtering techniques. […] we present such a system capable of addressing the task of soccer
> broadcast camera tracking efficiently, robustly, and accurately, outperforming by far the most
> precise methods of the state-of-the-art. By combining the available open-source soccer field
> detectors with carefully designed camera and tripod models, our tracking system, BroadTrack,
> halves the mean reprojection error rate and gains more than 15% in terms of Jaccard index for
> camera calibration on the SoccerNet dataset.

**We took:** the tripod camera model — the strongest hypothesis for what the 68.3 leader might be
doing.
**What happened:** **REFUTED for GSR on our substrate, and the refutation is one of our cleanest
results.** Run as an instrument (never vendored; EVS licence is non-commercial): median accuracy
1.0001 m against our 0.4024 m; jitter 26× better; end-to-end **−7.39 GS-HOTA, 10 of 10 clips
hurt**, with **AssA falling 9.3 points**. Its own tripod estimator raises an assertion error on 9 of
10 GSR clips — 30-second clips are outside its design envelope (their paper's own long-form demo is
20 minutes). Recommendation adopted: do not clean-room it.

*The nuance a professor will appreciate:* BroadTrack is not a bad system. It is optimised for
**completeness and temporal consistency in live broadcast AR**, where a jittering graphic is
unacceptable. Our W4 curve says the GSR benchmark pays for **per-frame accuracy** and hardly pays at
all for completeness (+0.49) — different objective, opposite optimum.

### 20. Enhancing Soccer Camera Calibration Through Keypoint Exploitation
Falaleev & Chen (sportlight) — MMSports 2024 — **arXiv:2410.07401**

> Accurate camera calibration is essential for transforming 2D images from camera sensors into 3D
> world coordinates […] obtaining a sufficient number of high-quality point pairs remains a
> significant challenge […] This paper introduces a multi-stage pipeline that addresses this
> challenge by leveraging the structural features of the football pitch. Our approach significantly
> increases the number of usable points for calibration by exploiting line-line and line-conic
> intersections, points on the conics, and other geometric features. To mitigate the impact of
> imperfect annotations, we employ data fitting techniques. […] A voter algorithm iteratively
> selects the most reliable keypoints […] secured the top position in the SoccerNet Camera
> Calibration Challenge 2023.

**We took:** the plan to harvest denser correspondences (conic tangent points, line-conic
intersections) as the v10-W5 "accuracy attack".
**What happened:** **not built, and the reason is a result.** W5's hypothesis-pool analysis showed
**37% of the achievable accuracy gain was unspent inside the hypotheses PnLCalib already emits.**
Better selection came first, and it delivered +2.10. Their repository also carries **no licence**,
so anything shipped would have to be clean-room from the paper. Denser correspondences remain the
top-ranked next geometry bet (§5.1).

### 21. Can Geometry Save Central Views for Sports Field Registration?
Magera, Hoyoux, Castin, Barnich — 2025 — **arXiv:2504.20052**

> Single-frame sports field registration often serves as the foundation for extracting 3D
> information from broadcast videos […] because of the sparse and uneven distribution of field
> markings, close-up camera views around central areas of the field often depict only line and
> circle markings. On these views, sports field registration is challenging for the vast majority
> of existing methods, as they focus on leveraging line field markings and their intersections. It
> is indeed a challenge to include circle correspondences in a set of linear equations. In this
> work, we propose a novel method to derive a set of points and lines from circle correspondences,
> enabling the exploitation of circle correspondences […]

**We took:** the circle → line conversion as a route to rescuing centre-circle-only frames.
**What happened:** **scoped, then deprioritised on measurement.** Our post-fill dead pool is now
**216 frames (1.45%)**, of which only **24 are central-view**. The whole completeness prize is
+0.49. Correct idea, wrong bottleneck for us.

### 22. Tent: Fully Test-time Adaptation by Entropy Minimization
Wang, Shelhamer, Liu, Olshausen — ICLR 2021 — **arXiv:2006.10726**

> A model must adapt itself to generalize to new and different data during testing. In this setting
> of fully test-time adaptation the model has only the test data and its own parameters. We propose
> to adapt by test entropy minimization (tent): we optimize the model for confidence as measured by
> the entropy of its predictions. Our method estimates normalization statistics and optimizes
> channel-wise affine transformations to update online on each batch. Tent reduces generalization
> error for image classification on corrupted ImageNet and CIFAR-10/100 and reaches a new
> state-of-the-art error on ImageNet-C. […]

**We took:** the BatchNorm-statistics half — recompute the detector's normalisation statistics on
each test clip's own unlabelled frames. Free, no gradients, and never applied to GSR by anyone.
**What happened:** **registered FAIL at −17.5645 mean paired GS-HOTA, 0 of 10 clips helped.** The
mechanism was measured, and it is the useful part: BatchNorm buffers move only 4–5%, yet median
detector confidence drops 0.76 → ~0.60, detections fall 23.5% and fragments rise 37%. **Every
downstream threshold in our chain is calibrated to the shipped confidence distribution**, so a
uniform confidence deflation reads as weak evidence everywhere. Premise correction on record: our
detector is fine-tuned on GSR-train and DEV is in-domain — **there is no domain shift to adapt
away.** TTA is only warranted on genuinely out-of-domain footage (the ManU/EPL corpus).

### 23. Evidential Deep Learning to Quantify Classification Uncertainty
Sensoy, Kaplan, Kandemir — NeurIPS 2018 — **arXiv:1806.01768**

> Deterministic neural nets have been shown to learn effective predictors on a wide range of
> machine learning problems. However, as the standard approach is to train the network to minimize
> a prediction loss, the resultant model remains ignorant to its prediction confidence.
> Orthogonally to Bayesian neural nets […] we propose explicit modeling of the same using the
> theory of subjective logic. By placing a Dirichlet distribution on the class probabilities, we
> treat predictions of a neural net as subjective opinions […] We observe that our method achieves
> unprecedented success on detection of out-of-distribution queries and endurance against
> adversarial perturbations.

**We took:** the Dirichlet formulation for teaching the jersey reader to abstain.
**What happened:** **FAILED by one sequence** (+1.2562 cleared the +1.0 bar; 11 of 20 clips helped
missed the 12-of-20 bar). Two things survived: the uncertainty term alone pays +0.36 end to end;
and a hyperparameter refutation worth publishing in a footnote — their `kl_max = 1.0`, **fitted on
10-class MNIST, collapses a 100-class head to α = 1 everywhere.**

### 24. From Broadcast to Minimap: Achieving State-of-the-Art SoccerNet GSR
Golovkin, Nemtsev, Shandyba, Udin et al. (Constructor.Tech) — CVPRW 2025 — **arXiv:2504.06357**

> Game State Reconstruction (GSR), a critical task in Sports Video Understanding, involves precise
> tracking and localization of all individuals on the football field […] Achieving accurate GSR
> using a single-camera setup is highly challenging due to frequent camera movements, occlusions,
> and dynamic scene content. In this work, we present a robust end-to-end pipeline for tracking
> players across an entire match using a single-camera setup. Our solution integrates a fine-tuned
> YOLOv5m for object detection, a SegFormer-based camera parameter estimator, and a DeepSORT-based
> tracking framework enhanced with re-identification, orientation prediction, and jersey number
> recognition. […] securing first place in the SoccerNet Game State Reconstruction Challenge 2024

**We took:** two things. Their **±2°/±2 m clamp** on temporal smoothing, adopted as a safety guard;
and the confirmation that **association in pitch coordinates** (their DeepSORT runs after
projection) is a real design, not our invention.
**What happened:** the clamp was **measured harmful for our batch solve** — it halves the gain,
because it blocks exactly the frames where the per-frame solve is catastrophic and the batch
rescues it. A published guard, adopted uncritically, tested, and rejected with a mechanism. Note
also: their paper has **no ablation table at all**, so none of their 63.81 is attributable to a
component.

### 25–27. The landscape papers
**SoccerNet 2025 Challenges Results** (arXiv:2508.19182) and **SoccerNet 2026 Challenges Results**
(arXiv:2607.07320) gave us the leaderboards, team method summaries, and — decisively — the fact
that the 2026 season **closed on 25 April 2026** with 427 teams and 1,129 entries across five
tasks, while the **GSR test board remains live**. **SoccerMaster** (arXiv:2512.11016), a soccer
vision foundation model unifying detection, identification and event tasks through multi-task
pretraining, reports **64.1 GS-HOTA — the highest number in any paper anywhere**, which is what
makes the board's 68.3 remarkable.

### 28. Qwen2.5-VL Technical Report (the VLM family)
Bai, Chen, Liu, Wang et al. — 2025 — **arXiv:2502.13923**

> We introduce Qwen2.5-VL, the latest flagship model of Qwen vision-language series […] a major
> leap forward in understanding and interacting with the world through enhanced visual recognition,
> precise object localization, robust document parsing, and long-video comprehension. […] It
> provides robust structured data extraction from invoices, forms, and tables […] The flagship
> Qwen2.5-VL-72B model matches state-of-the-art models like GPT-4o and Claude 3.5 Sonnet,
> particularly excelling in document and diagram understanding.

**We took:** the family. Our trial used **Qwen3-VL-8B** (Apache-2.0, 17.5 GB) — *no arXiv record was
fetched for that specific model; it is described here from its model card and our own trial log*.
**What happened:** **registered FAIL** at 0.889 precision on 9 emitted (bar 0.90, unresolvable at
n = 9), and a decisive per-crop comparison: our fine-tuned PARSeq chain was correct on 324 crops
where the VLM was wrong; the VLM was correct on 54 where the chain was wrong. **McNemar
p = 4.8 × 10⁻⁴⁸.** Handing the VLM our torso crops made it worse. Identity-via-VLM is closed for
this project — and independently consistent with the 68.3 leader using no VLM at all.

### 29. Broadcast2Pitch: Game State Reconstruction from Unconstrained Soccer Videos
Oo, Hwang, Robbani, Chao, Jamsrandorj, Nguyen, Mun, Kim (KIST) — **WACV 2026** — no arXiv record.

**[PARAPHRASE — from our recon of the CVF PDF and repository, not the authors' abstract text.]**
The 2025 challenge winner (63.90 challenge / 61.48 test). Five components: YOLOX detection,
Deep-EIoU + OSNet tracking, sports-field registration via an EfficientNetV2-S + U-Net model over 97
keypoints and 18 lines, a fine-tuned LLaMA-3.2-Vision identity model, and **IDATR**, a
tracklet split-and-merge refinement driven by identity predictions rather than appearance.

**We took:** their **ablation tables** — which steered our entire v8 campaign — and the bottleneck
they name themselves.
**What happened:** **three of their four claimed edges fell under direct measurement.**

| Their claim | Our measurement |
|---|---|
| calibration worth +10.28 | our calibration is **12% more accurate** (0.475 m vs 0.536 m); their gain was against their own weak keypoint-only baseline |
| identity model is the edge | our reader covers **10–11× more tracks at 0.90+ precision** than their CLIP head |
| team assignment | theirs is top-2-most-frequent-colour-*name* matching against 13 hardcoded strings; ours is stronger |
| IDATR association +2.97 | **real, and never matched by us** — this is their genuine remaining edge |

Their paper also states its own primary weakness verbatim: *"jersey number predictions are
aggregated via majority voting, which is particularly fragile when correct digits are sparsely
observed."* We fixed exactly that, on their own model's outputs — see §4.3.

### 30. Broadcast2Pitch++: Depth-Aware Game State Reconstruction
KIST, Research Square preprint, June 2026, DOI 10.21203/rs.3.rs-9790440/v1 — no arXiv record.
**[PARAPHRASE — from the CC BY preprint text.]**

Extends the above to 62.56 on test. Adds depth-aware tracking (Depth-Anything-V2-Small foot depth,
EMA 0.5, cost weight 0.3) and **IDASTR**, a *soft* merge: cost = ReID cosine + 0.5 × spatial +
0.2 × identity-**disagreement** among majority votes, threshold 0.45.

**We took:** confirmation, from an independent group, of our own v9-W1 finding that **hard
identity gates forbid a large fraction of true merges** (we measured 19.8%). They moved identity
from a filter to a soft cost; so did our analysis, before we saw their preprint.
**What happened:** corroboration, not adoption. Their depth variant also carries an honest
trade-off on record: AssA 81.87 but DetA −6.5.

### 31. Grad: Single-Stage Uncertainty-Aware Jersey Number Recognition
CVPRW 2025 — no arXiv mirror, no code. **[PARAPHRASE — from the CVF PDF.]**

Reports 85.62 tracklet accuracy on the GSR challenge split against Koshkina's 79.31, using a
torso-crop ViT, a digit-aware tied head and Dirichlet abstention. Claims that illegible crops
provide **free negative supervision**.

**We took:** the free-negatives claim, and built it.
**What happened:** **refuted.** Defining the "no number" class by crops that *fail our legibility
filter* teaches the student to imitate its teacher: invisible-class AUC **0.6200**, below the frozen
trunk's own 0.6429 and well below the classifier it was meant to replace (0.7038). **A pseudo-label
transfers the labeller's decision boundary, not the truth.** That closed the cheapest imaginable
route to more jersey reads.

---

# 4. The research defence

> **"You neither topped the leaderboard nor invented a fundamentally new architecture. So what was
> the research?"**

## 4.1 The direct answer, in one paragraph

*The research is measurement-first science on a leaderboard problem. I did not set out to add
another entry to a converging field of near-identical pipelines; I set out to find out what
actually determines the score on this benchmark, and to establish it with a method that makes my
negative results as credible as my positive ones. Concretely I produced four measurements nobody
has published — the calibration-error-to-GS-HOTA sensitivity curve, an audit that found six errors
in the benchmark's own ground truth, the first human per-crop audit of a jersey-legibility gate,
and a head-to-head demonstration that our aggregation beats the state-of-the-art system's
aggregation on its own model's outputs. I shipped three method components that each passed a
pre-registered gate, one of which is provably equal to an oracle. And I mechanically closed five
directions the field is currently spending GPU-hours on — vision-language models for identity,
tripod camera models, test-time adaptation, label-purity filtering, and name-borrowing — each with
a measured mechanism, not an opinion. The score went from 14.76 to 57.17 on our development split
along the way, but the score is the by-product. The transferable output is the map: which subsystem
is worth how much, on what evidence, with what noise floor.*

## 4.2 Pillar 1 — the method is the contribution

Almost nobody in this benchmark's literature does the following. It is unglamorous and it is
exactly why our numbers can be trusted.

**Pre-registration.** Every headline experiment has a pass/fail bar written into a timestamped JSON
file *before* the experiment runs. Not a target — a bar with a kill rule. Consequences on record:

- The detector's first training run **failed its gate by 20 detections** (0.05042 against a ≤0.05
  bar). It is recorded as a FAIL and the threshold was not touched; a second run under a freshly
  declared identical gate passed. *A threshold you can move after seeing the data is not a
  threshold.*
- **The entire v7 campaign shipped nothing.** Five pre-registered ideas, five failures, 53.09 left
  untouched. Total GPU spend: ~5.6 hours of 42 budgeted — the gates killed everything at 13% of
  planned cost.
- In v9-W5, a bar was **honestly reduced** from +0.60 to +0.35 *before* scoring, because the
  stage-1 arithmetic proved +0.60 was physically unreachable — and the arm still failed at +0.21.
  The reduction and its reason are both in the registration file.

**Split hygiene.** Three splits, three roles: tune on DEV-20; read TEST-38 **once** per frozen
recipe; read test-49 **once** per version, then upload. Configuration frozen to a timestamped JSON
before any held-out read. Five public submissions used out of ten in a lifetime.

**Paired statistics.** Per-sequence Wilcoxon signed-rank tests, never bare means. Plus a
**concentration guard**: any gain whose top-2 sequences carry ≥80% of the net is demoted regardless
of its p-value. One arm was killed by that guard alone (concentration 1.92: top-2 = +5.91 on a net
of +3.07).

**Measured noise floors.** You cannot interpret a +0.4 result without knowing what zero looks like.
We measured:

| Noise source | Magnitude | How we found out |
|---|---|---|
| PARSeq training-seed spread | **0.0059** precision | three identical retrains, seed unset in the recipe (documented, not papered over) |
| Extraction drift, same code, 8 days apart | **+1.02 GS-DetA / +1.08 GS-HOTA** | v9-W7, from dependency churn alone |
| Within-one-stack re-extraction | **exactly 0.0** | bit-deterministic; verified |
| Ground-truth benchmark noise | **~2.45 GS-HOTA** | the side-swap audit |

The drift number is the sharpest one. **It is larger than most published GSR ablation deltas.**
Pairing a new arm against an on-record number 8 days old would *manufacture* a pass. Every
comparison after that discovery happens inside one freshly extracted lineage — and the freeze
session re-extracted DEV, TEST-38 and test-49 in one window purely to satisfy that rule.

**A versioned claims ledger.** `knowledge/claims.json` holds **301 claims**, each with a statement,
evidence paths and a status:

| Status | Count |
|---|---|
| confirmed | 242 |
| pending | 22 |
| **refuted** (our own hypothesis, killed by our own data) | 12 |
| **retracted** (we said it, then withdrew it) | 12 |
| refuted externally | 8 |
| superseded | 5 |

**Twelve retractions.** That is the number to point at. Example: an "anti-informative goalkeeper
appearance" mechanism claim turned out to be an artefact of a **BGR/RGB channel-order bug**; it was
withdrawn, re-run correctly, and the verdict downgraded from "mechanism" to "nothing". Another: a
framing explanation for one benchmark clip was retracted when a better instrument showed the ground
truth itself was wrong. **A ledger with zero retractions is a ledger nobody checked.**

## 4.3 Pillar 2 — four measurements nobody has published

This is the part to lead with. Each of these is a result about the *problem*, not about our code.

### (a) The calibration-error → GS-HOTA sensitivity curve

**What it is:** inject controlled camera error into a fixed submission and re-score, with
association frozen; then replace the camera with an oracle and re-score again. The output is a
curve: *how many GS-HOTA points does one metre of calibration error cost, on this benchmark?*

**Why nobody has it:** the sports-calibration literature evaluates calibration with calibration
metrics (reprojection error, Jaccard index at 5 px, completeness). The GSR literature reports
end-to-end scores. Nobody has connected the two axes. Our own literature sweep found the gap and
flagged the experiment as "cheap, novel, and it tells us exactly when to stop investing in
geometry".

**What it says:**

1. The geometry axis holds **+6.82 GS-HOTA**, of which **92% is per-frame accuracy** and only
   0.49 is completeness.
2. **Calibration stops mattering at ~0.20 m mean error; we sit at 0.762 m.**
3. **Frame-correlated and independent position error cost identically** per metre — camera error
   and detection error are the same currency.
4. **Association gains more from geometry than detection does**, with association frozen.
5. The "5 m tolerance" is a Gaussian with σ = 2.0427 m, so **every centimetre is charged** — which
   refutes the "our median is already inside tolerance, stop optimising" reasoning that we
   ourselves had written down two days earlier.

**Why it matters beyond us:** it is a *decision instrument* for the whole field. Any GSR team can
now ask "is my calibration budget spent?" and get a number instead of an intuition. It also
explains a published result: BroadTrack's headline on GSR clips is 100% completeness, and this
curve says completeness is worth +0.49 here.

**Say this:** *"I built the first curve relating camera-calibration error to the GSR score. It
shows the metric charges you for every centimetre — there is no tolerance zone — and that 92% of
the available geometry headroom is per-frame accuracy, not the completeness and smoothness that the
sports-calibration literature optimises. That result refuted my own prior, which is why I trust
it."*

### (b) The benchmark audit — six ground-truth errors, two beyond the state of the art's own list

**What it is:** an instrument that checks, across every sequence, whether kit-to-side assignment is
consistent within each (game, half).

**What it found:** **five side-swapped sequences**, all of them **second-half clips still carrying
their game's first-half side convention** — a *diagnosed annotation-process cause*, not a list of
anomalies. 159 of 164 sequences obey the consistency rule; exactly these five dissent. The
state-of-the-art paper names three (SNGS-126, 131, 197); **we found SNGS-092 and SNGS-111 in
addition**, with an instrument built without reading their code. A sixth error of a different type
(a player track and a referee track at IoU ≈ 0.85 in one frame) is filed pending.

**The methodological lesson inside it** (this is the part that impresses examiners): our two best
per-clip instruments were **structurally blind** to this error. When a whole clip is swapped, the
goalkeeper's label swaps along with everyone else's, so the clip is *internally flawless*. Only a
cross-clip consistency check catches it. The first pass said "ground truth is fine", then retracted
itself and built the third instrument.

**Consequence:** ~2.45 points of our score loss is unreachable by any correct method. Corrected-GT
score ≈ 55.5 at the 53.09 era.

**Say this:** *"I audited the benchmark and found six ground-truth errors, five of them with a
single diagnosed cause. The SOTA paper had found three of them; my instrument found two more. About
2.45 points of everyone's score is benchmark noise — including theirs."*

### (c) The human per-crop gate audit, and the purity-versus-volume refutation

**What it is:** Sid personally labelled **499 tracklets** and **2,520 individual crops** for the
question "is a number visible in this crop?" — the first human per-crop supervision of a
jersey-legibility gate that we can find anywhere.

**What it found:**

- The gate's **recall is 0.964** — it loses only 3.6% of readable glyphs — but its **precision is
  0.728**: about **27% of what it admits shows no readable number**. Transferred to the full
  corpus: ~38,300 of 140,278 admitted crops are positive-class label noise.
- **94.8% of the "unnamed" pool is genuine glyph absence** — there is no number to read, so the
  "unlabelled treasure" premise deflated about 20×. Independently re-confirmed at VLM scale in
  v10-W1.
- The benchmark contains **zero goalkeeper jersey numbers anywhere** (0 of 79 train GKs, 0 of 77
  valid GKs). **Every jersey-reading ceiling on this benchmark carries that cap** and, as far as we
  know, nobody had stated it.

**Then the refutation.** The obvious next move is to filter the 27% noise out and retrain. We did,
pre-registered:

| Arm | Corpus | Result vs incumbent |
|---|---|---|
| ARM V ("volume", legibility ≥ 0.7) | 48,567 rows | −0.0051, **p = 0.51 — a null** |
| ARM P ("purity", ≥ 0.99) | 61% of V's rows | −0.0211, **p = 0.007 — worse** |
| P100 (purity, **volume-matched**, +3.1% larger) | 97,347 rows | **−0.0255 vs V, McNemar p = 0.0020** |

**Purity loses to volume, and it still loses when volume is matched** — 4.3× the measured seed
noise. The 27% label noise is a true data-quality fact that **does not convert into a better model
by filtering.** This is a clean, noise-floored negative that speaks directly to the noisy-labels
literature, where robustness-at-high-volume is a known phenomenon but rarely tested this cleanly on
a real, human-audited industrial corpus.

**Say this:** *"I ran the first human per-crop audit of a jersey-legibility gate — 2,520 labels. The
gate admits 27% noise. Then I pre-registered the obvious fix, filtering that noise out, and it made
the model significantly worse, even at matched corpus size. The data-quality defect is real; the
intuitive remedy is wrong. I have the noise floor to prove the difference is not seed variance."*

### (d) The seam experiment — beating the leader's aggregation on the leader's own outputs

**What it is:** the state-of-the-art paper names its own primary bottleneck in its own words —
majority voting over jersey reads destroys confidence information. We ran **their** identity model
on **our** crops and swapped **only** the aggregation rule for our confidence-weighted fusion.

**What it found:** **+0.107 tracklet accuracy on their own model's outputs**, 13 of 20 sequences,
**p = 0.0017**, 37-versus-4 flips (p ≈ 10⁻⁷), monotone in the confidence floor, and the effect lives
exactly in the crops their `torch.max` discards. It survives the density objection: **our
aggregation on 15 crops beats theirs on 425.** On *our* reader the same effect is 4× smaller —
because we had already harvested it.

Bonus head-to-head from the same probe: **our jersey reader covers 10–11× more tracks at 0.90+
precision than theirs** (tracklet read density 0.716 vs 0.063).

**Why it is strong evidence:** it is not "our system beats theirs" — that comparison is confounded
by a hundred differences. It is a **controlled single-variable experiment inside their system**,
targeting a weakness they themselves published, and their own paper's future-work section invites
the change.

**Say this:** *"The state-of-the-art paper names its own bottleneck: it aggregates jersey reads by
majority vote and throws away the confidences. I ran their model, changed only the aggregation to
my confidence-weighted fusion, and beat their rule on their own model's outputs — p = 0.0017. One
variable, their substrate, their stated weakness."*

## 4.4 Pillar 3 — method components that passed their gates

Three shipped things. None is a new architecture; all three are new *rules*, and each one carries a
proof of value.

**(1) Track-attribute voting** — a whole track votes on its own role, team and jersey rather than
leaving per-row disagreements in place, plus a writer repair that was silently dropping numbers
onto goalkeeper and referee rows. **+2.99 GS-HOTA, 20 of 20 clips helped, p = 1.9 × 10⁻⁶**, zero
GPU, deterministic (noise floor exactly zero).

**(2) The keeper-side geometric rule** — *a goalkeeper's team is the side of the pitch he stands
on.* Zero fitted parameters. **+1.46 GS-HOTA, 18 helped, 0 hurt, p = 1.96 × 10⁻⁴** — and it is
**bit-identical to the goalkeeper-team oracle at 15 significant figures**. Row audit: 5,171
wrong→right, 0 right→wrong.

That oracle-equivalence is the rhetorically strongest single number in the project. *A rule that
reads only our own predictions is exactly as good as being told the answer for that bucket.* It
exists because we first measured *why* keepers were wrong — kit clustering puts a keeper in his own
team only 24.4% of the time, because the laws of the game *require* the keeper's kit to differ from
both outfield kits, which makes appearance systematically anti-informative for exactly this class.

**(3) Temporally-informed hypothesis re-selection + full-clip batch refinement in mapping space** —
the v10-W5 result. Our calibrator emits ~16 camera hypotheses per frame; our gate was picking the
most accurate one only 25.4% of the time (median rank 4). Re-select using temporal context, then
refine the whole clip jointly in homography space under a robust loss. **+2.10 GS-HOTA, 19 helped,
1 hurt, p = 5.0 × 10⁻⁵**, CPU only, no new model, no new correspondences. Mean calibration error
collapses 1.600 → 0.590 m.

Two design laws came out of it, both stated as transferable claims:

- **"Smooth the map, not the camera"** — camera-space smoothing failed at −5.77 because the
  per-frame camera decomposition is degenerate (position trades against focal length); the identical
  smoother in homography space passes.
- **A published safety clamp (the 2024 winner's ±2 m) is the wrong guard for a batch solve** — it
  blocks precisely the catastrophic frames the batch exists to rescue.

**Say this:** *"Three components passed pre-registered gates. The one I would defend hardest is the
keeper rule: a parameter-free geometric rule that scores bit-identically to the oracle for its
bucket, and it exists because I first measured why the appearance model fails on keepers — the laws
of the game guarantee it will."*

## 4.5 Pillar 4 — mechanised refutations that save the field GPU-time

A negative result with a *measured mechanism* is reusable. A negative result without one is a
rumour. We have five of the former, each closing a road the field is currently driving down.

| Direction | Verdict | The mechanism (this is the deliverable) |
|---|---|---|
| **VLMs for jersey identity** | FAIL, 0.889 vs 0.90 bar | a fine-tuned 20M-parameter specialist beats an 8B general VLM per-crop at **p = 4.8 × 10⁻⁴⁸**; giving the VLM better crops makes it worse. Corroborated externally: the 68.3 leader uses no VLM |
| **Tripod camera models (BroadTrack class)** | REFUTED, **−7.39**, 10/10 hurt | it optimises smoothness and completeness; this benchmark pays for per-frame accuracy (2.5× worse), and **AssA falls 9.3** — smoothness buys nothing once it costs accuracy. Their estimator also fails on 9/10 clips: 30 s is outside its design envelope |
| **Test-time adaptation (BN statistics)** | FAIL, **−17.56**, 0/10 helped | there is no domain shift in-domain to adapt away; and BN buffers moving 4–5% deflate detector confidence 0.76 → 0.60, which every downstream threshold reads as weak evidence |
| **Label-purity filtering** | REFUTED, p = 0.002 | 27% label noise is real and filtering it makes the reader worse at matched volume — noisy-label robustness at scale |
| **Name-borrowing across tracks** | FAIL, oracle +0.23 of a +8.15 bound | the bound was **structurally phantom**: the connector had already consumed the reachable links, so 97% of the "prize" has no lender anywhere in the clip |
| **Learned association transplant (CAMELTrack)** | REFUTED, −13.6 to −20.2 | an **input-representation contract** failure: their tokenizer expects 6×128 part embeddings with visibility; we hold one 256-d global vector; the model silently ran appearance-less |
| **Learned side detection** | KILL at rung 1 | the label is itself a function of pitch geometry, so the model becomes a **second estimator of the same latent quantity** — it ties the geometric rule exactly (91/97 vs 91/97) and is wrong on 2 of the 3 clips it was built for, at high confidence |
| **Teacher-imitation pseudo-labels** | REFUTED, AUC 0.620 vs teacher 0.704 | a pseudo-label transfers the labeller's decision boundary, not the truth |

Add the smaller ones on record: commentary as identity evidence (binds at chance), gait and face ID
(median face = **9.9 pixels**), appearance retrieval on same-kit broadcast (saturated at 0.65–0.71
top-1, three registered fails in three days), and a detector retrain **refused with numbers** after
the miss anatomy showed only 9.7% of misses were the detector's fault.

**Say this:** *"Eight directions were built and killed under pre-registered gates, each with a
measured mechanism. That is not a list of things that didn't work — it is a map of why they cannot
work on this problem, and any team that reads it saves the GPU-hours I spent."*

## 4.6 The honest limits of this defence

State these before the examiners do. Volunteering a limitation is what separates a researcher from
a salesperson.

1. **We are not on top of the leaderboard**, and 68.3 exists. Our best packaged score is 55.41 with
   a development-split 57.17 in flight.
2. **Most of our wins are selection, rules and post-processing — not learned models.** §5.5 answers
   this properly; the short version is that our learned attempts died at gates and the deaths are
   documented, which is honest but is not the same as having a learned contribution.
3. **Single benchmark, single dataset, 30-second clips from one league in one season.** Nothing here
   is validated as transferring to other footage.
4. **The sensitivity curve is a prediction-level instrument**, computed with association frozen. It
   is therefore a *lower bound* on the end-to-end cost of calibration error, and we labelled it that
   way in the registration file before running it.
5. **Two of our headline comparisons rest on the leader's published tables**, which have never been
   independently reproduced (the one public attempt reached 48.32).

---

# 5. Where the remaining points are, and what we lack

## 5.1 Geometry: the uncaptured 66%

The W4 curve says the geometry axis holds **+6.25 GS-HOTA of per-frame accuracy**. Arm D captured
**33.6%** of it. The remaining ~4.15 points, ranked by evidence:

**(a) The lens-distortion term — the strongest single piece of evidence we have for unmodelled
error.** PnLCalib models **no radial distortion at all**. We measured a **+1.88 px residual on the
outer ring** of the image after batch refinement — a systematic, radially structured error that no
homography can absorb, because a homography is a plane-to-plane map and lens distortion is not. A
per-clip `k₁` term (one extra parameter, shared across all 750 frames of a clip, fitted inside the
existing batch refinement) is the obvious next step. Cost: days, CPU. Note also that "un-inverted
forward radial distortion" is item 3 on the leader's own bug list.

**(b) Denser correspondences, clean-room.** Falaleev & Chen's conic tangent points and line-conic
intersections would raise the number of usable points per frame — more constraints, better fit.
Deprioritised in W5 only because 37% of the achievable gain was unspent *inside the existing
hypothesis pool*; that has now been partly spent, so this rises. Their repo carries **no licence**:
build from the paper.

**(c) Learned hypothesis selection.** Arm D's re-selection is a hand-built temporal rule and it
still leaves the best-of-pool oracle (+2.32 predicted) not fully captured. Choosing among ~16
hypotheses per frame, given temporal context and per-hypothesis features, is a small,
well-posed supervised learning problem with abundant training signal — and it is **the most
defensible learned bet in the project** (§5.5).

**(d) Rolling shutter.** Untouched anywhere in the sports-calibration literature, and broadcast
cameras have it. Speculative; listed for completeness, not recommended.

## 5.2 The association substrate

**The uncuttable 28.5%.** Our oracle analysis says a perfect per-row splitter removes only 71.5% of
contamination; the rest is **interleaved** — two identities alternating inside one track in a way no
single cut point can separate. Oracle split + perfect merge = ~60.40. **That is a ceiling on the
current architecture, not a to-do item.** It is also the single most important number for managing
expectations: *even a perfect associator on our substrate does not reach 61.48.*

**A learned splitter** is the untried half. The GTA splitter is inert on our chain (2 splits in
4,709 tracklets) and at any threshold it mostly cuts *clean* tracklets. Our own W1 factory already
contains the labels needed to train a splitter — 4,709 tracklets with per-row ground-truth identity.
This is a well-scoped supervised problem we costed and never built.

**The rank-1/rank-2 join.** Joining each identity's largest fragment to its second-largest alone
moves the dominant-fragment share from 0.563 to 0.738. It needs a merge precision our appearance
model cannot reach: within-identity distances at the 90th percentile (0.093) overlap
cross-identity distances at the 10th (0.040–0.083). **Overlapping distributions are the textbook
case for a learned model rather than another threshold.**

## 5.3 Detector modernisation — with an honest caveat

The 68.3 leader runs **DEIMv2-DINOv3** at 896 px; we run YOLOv8s. A modern detector is the obvious
upgrade, and there is a licence-clean path (his `dfine-cpp` is Apache-2.0).

**But our own miss anatomy says the detector is not our problem.** Of 19,933 DEV misses: 66.1% are
**confident detector boxes the chain discards**, 9.7% are true detector misses. **A perfect detector
is worth +0.57 GS-DetA.** We refused a detector retrain on those numbers, and that refusal is on
record with its arithmetic.

The honest reconciliation: a stronger detector helps mostly by *producing better tracks*, not by
finding more people — better boxes make better association and better crops for reading. That is a
plausible mechanism but we have **not** measured it, and we should not claim it. The measured
statement is: *finding more detections does not help us; two registered recovery arms proved it,
one of them with an attribute yield of exactly 0.0%.*

## 5.4 What 68.3 implies

DetA 55.86 against our 41.28 is the telling number. Working backwards through our own oracles:
identity-gated detection at that level is beyond what our jersey-evidence stack can reach — the
jersey-coverage oracle is largely **structurally phantom** on our substrate (94.8% genuine glyph
absence, and the benchmark never numbers goalkeepers at all).

The only reading consistent with everything we measured is that 68.3 rests on a **composite
substrate roughly five points beyond every published method**: a modern detector, a much cleaner
camera, and cleaner tracks feeding better votes — with the calibration overhaul as the load-bearing
piece, per the commit evidence. It is not one trick. That is simultaneously discouraging (no single
idea to copy) and clarifying (our +2.10 geometry result is on the right axis).

## 5.5 The gap a professor will actually probe: *where is the learning?*

Say it first, in your own words, before it is asked.

**The criticism, stated fairly:** of everything that shipped, the big wins are a bug fix, two
threshold repairs, a voting rule, a parameter-free geometric rule, and a classical optimisation.
The only trained components are three fine-tunes of existing architectures on new data. **There is
no novel learned model in the shipped system.** For a computer-vision thesis that is a real gap.

**The honest answer, in three parts:**

*First, this is a finding, not an accident.* The measured history of this project is that **data and
defects beat architecture at our scale, repeatedly and by large margins.** The single largest
non-training jump (+9) was a tallying bug; the single largest training jump (+14) was powered by
finding 106k free labels in an older dataset, not by a new architecture; the tracker paper's
headline mechanism was worth +0.37 while its unheadlined appearance term was worth +3.49. Every
architecture transplant we attempted failed its gate. Recording that pattern *is* a research
finding about this problem class.

*Second, the learned attempts exist and their deaths are documented.* We trained: a 6-class
side-aware detector (ties the geometric rule exactly — the label is a function of geometry); an
evidential Dirichlet jersey head (failed by one sequence; its published constant refuted); a learned
association model, twice (CAMELTrack transplant, −13 to −20 by input-contract failure;
TwixMetric from scratch, which **won its ranking gate and lost its precision gate**); a CLIP
identity encoder (won as an association cue at +3.49, lost as a namer). That is five learned models
built and measured. None shipped. **All five deaths have measured mechanisms**, and three of the
mechanisms are transferable claims about the problem rather than about our code.

*Third, here are the next learned bets in order of evidence* — and note that each is now aimed at a
measured gap rather than at a fashionable component, which is precisely the correction the record
demanded:

| Rank | Learned bet | Why the evidence supports it | Scale |
|---|---|---|---|
| 1 | **Learned hypothesis selection** (pick the best of ~16 PnLCalib camera hypotheses per frame using temporal context) | the gate picks the best only 25.4% of the time, median rank 4; best-of-pool oracle +2.32; a hand-built rule already banked +2.10, so the residual is real and the labels are free (accuracy against GT camera) | small model, hours on one A100 |
| 2 | **Learned tracklet splitter** | 13.3% of tracklets need splitting; the published splitter is inert on our chain; W1's factory already holds per-row identity labels for 4,709 tracklets | small model, hours |
| 3 | **Batch refinement as a differentiable layer** | arm D is a fixed-point optimisation with hand-chosen robust loss and weights; making it differentiable lets the selection, the distortion term and the refinement train against the kernel loss the metric actually uses | days |
| 4 | Faithful CAMELTrack (KPReID parts + pose + detector-output corpus) | the one trial we ran was explicitly *not* the method; the +11.9 merge-only headroom is still sitting there | 8–12 GPU-h |

**Say this:** *"Most of my shipped gains are not learned models, and I think that is a result rather
than a failing — at this scale, on this benchmark, data and defects beat architecture, and I have
the ledger to show it. But it is also a real gap in the thesis, so here is the honest accounting: I
built five learned models, all five died at pre-registered gates, and each death has a measured
mechanism. The next learned bet is hypothesis selection for the camera, because I have just proved
by hand that the signal is there — 25% selection accuracy today, a 2.32-point oracle, and free
training labels."*

**Two further gaps, volunteered:** there is **no end-to-end training** anywhere in this system — it
is a pipeline of independently trained parts with hand-set interfaces, and every threshold in it is
calibrated to a distribution some upstream component happens to produce (the TTA failure is exactly
this fragility made visible). And the **entire evaluation is single-benchmark**: 200 clips, one
league, one season, 30 seconds each.

## 5.6 The ranked list, if there were six more months

1. Learned hypothesis selection + per-clip distortion term inside the batch refinement (geometry:
   evidence-backed, ~+2 to +4 of the remaining 4.15).
2. Learned tracklet splitter (association: the untried half, ~+1 to +2 realistically against a
   28.5% hard ceiling).
3. Modern detector on a licence-clean path (unmeasured mechanism, honestly flagged).
4. Denser correspondences, clean-room from Falaleev & Chen.
5. Faithful CAMELTrack.
6. SNGS-082 — the worst single-clip regression of the whole campaign (−4 to −8), **never
   diagnosed**. One undiagnosed pathology on a 49-clip board is worth more than it looks, and it
   costs one afternoon of watching a video.

---

# 6. Pivot options

Four honest options, each with a fit assessment.

**(A) SynLoc 2027 — the strength-matched competition.** Spiideo SoccerNet SynLoc is single-frame
athlete localisation in world coordinates from a calibrated static camera. It is our strength triad
(detection, calibration, metric localisation) with our weakness (tracking and identity)
*surgically absent* — no association, no jersey numbers. The 2026 podium all detected in image
tiles and then ray-cast; **nobody detects directly in world space**, which the autonomous-driving
bird's-eye-view literature says works when the ground plane is known. Our shelved test-time
adaptation machinery would also finally meet a real domain gap (synthetic players composited onto
real backgrounds), which is exactly the condition under which our TTA failure said it *would* be
warranted. **Fit: excellent. Timing risk: the 2026 edition closed at 97.67 mAP-LocSim, so this is a
2027-season play** — servers historically open September–November, which would put an entry in
flight exactly at the December review. Prototype on the open sandbox server meanwhile.

**(B) PCBAS — the live, nearly-empty contest we are pre-adapted for.** Player-Centric Ball Action
Spotting: for each on-ball action, predict *when*, *which team*, *which jersey number*, *what
action* — and it counts only if **all of them** are right. That is GS-HOTA's identity logic at event
level, i.e. the exact problem we spent two months mastering. The dataset **provides tracklets and
22-player game state as inputs**, so our weakest subsystem is partially neutralised by the
organisers. The field is nearly empty: **6 teams ever, winner 58.94 macro-F1, servers open with no
end date.** New work required: temporal action spotting from video — a new subsystem, but
well-documented, with three baselines in the dev kit. Discipline note: the challenge phase allows
only 10 lifetime submissions, so all development goes on the 100-submission validation phase.
**Fit: strong. Realistic goal: top-3, plausibly top-1, on a live leaderboard by December.**

**(C) The methodology paper.** Take §4.3 and write it up as benchmark science rather than as a
leaderboard entry: the calibration-sensitivity curve, the six-error ground-truth audit with its
diagnosed cause, the human gate audit with the purity-versus-volume refutation, and the priced
graveyard of eight mechanised negatives. Venue: a CVsports/MMSports-class workshop, or the
SoccerNet organisers directly (the GT audit is a data contribution they should want, and the SOTA
paper corroborates three of our five). **Fit: this is the highest-value-per-hour option and the one
most likely to survive a December grilling intact, because it does not depend on a score.** It is
also the only option that turns our failures into first-class output. Risk: workshops prefer
methods, and a reviewer may want a method attached — the answer is to attach arm D (+2.10) as the
demonstration that the curve is actionable.

**(D) The original ManU scouting product.** Return to the pre-July scope: opposition scouting packs
for Manchester United from Premier League footage, using this whole stack as the tracking substrate.
**Fit: the strongest *product* story and the weakest *research* story.** It also re-opens problems
this campaign closed for GSR but not for real footage — full 90-minute matches, camera cuts,
replays, and a genuine domain shift where test-time adaptation would finally have a premise. If the
December review is graded on demonstrable application rather than on benchmark position, this is the
option; if it is graded on research contribution, it is not.

**The recommendation on the record** is A-then-B with C as the consolidator: build the
evidence-backed geometry work, enter PCBAS for a live leaderboard position at review time, and
write the methodology paper from material that already exists.

---

# 7. Cheat sheet

## The ten numbers to remember

| # | Number | What it is |
|---|---|---|
| 1 | **14.76 → 53.09 → 55.41 → 57.17** | public board floor → board best → packaged v9 (test-49, local) → current DEV-20 with the geometry work |
| 2 | **68.3 / 61.48 / 64.1** | current board leader (no paper) / best published system on test / highest GS-HOTA in any paper |
| 3 | **σ = 2.0427 m** | GS-HOTA's position kernel: similarity = exp(−0.5 (d/σ)²). The "5 m tolerance" is where it hits 0.05 — **not a gate** |
| 4 | **+6.82 / 92%** | total geometry headroom on our substrate, and the fraction of it that is per-frame accuracy, not completeness |
| 5 | **25.4%, median rank 4** | how often our gate picked the most accurate of the ~16 camera hypotheses we already had — the +2.10 came from fixing this |
| 6 | **52 of 52 vs 10 of 52** | keeper's pitch side predicts his team 52/52; kit colour manages 10/52. The rule is bit-identical to the oracle |
| 7 | **27% / 94.8%** | of admitted jersey crops carry no readable number (human audit, n = 2,520); of unnamed tracks the absence is genuine |
| 8 | **2.45 points** | benchmark ground-truth noise no correct method can recover — five side-swapped clips, two of which we found first |
| 9 | **0.0059 / +1.02** | training-seed noise floor / same-code extraction drift over 8 days — the numbers that make every other delta interpretable |
| 10 | **301 claims: 242 confirmed, 12 refuted, 12 retracted** | the ledger, including twelve things we said and then withdrew |

## The five sentences for the grilling

1. **"The contribution is measurement, not architecture: I built the first curve relating
   calibration error to GS-HOTA, and it refuted my own prior — 92% of the geometry headroom is
   per-frame accuracy, and the metric charges you for every centimetre."**
2. **"I audited the benchmark itself and found six ground-truth errors with a diagnosed
   annotation-process cause — two of them beyond the list the state-of-the-art paper publishes.
   About 2.45 points of everyone's score is benchmark noise."**
3. **"I took the leader's stated bottleneck — majority-vote jersey aggregation — ran their model,
   changed only the aggregation, and beat their rule on their own model's outputs at p = 0.0017."**
4. **"Eight directions were built and killed under pre-registered gates, each with a measured
   mechanism: VLM identity, tripod cameras, test-time adaptation, purity filtering, name-borrowing,
   learned association transplants, learned side detection, teacher-imitation pseudo-labels. Those
   negatives are reusable; a leaderboard position is not."**
5. **"My shipped gains are rules and selection rather than new learned models, and that is itself
   the finding at this scale — but it is a gap, so here is the next learned bet and the measurement
   that justifies it: camera-hypothesis selection, 25% accurate today, 2.32-point oracle, free
   training labels."**

## Glossary — one line each

**GSR** Game State Reconstruction: broadcast video in, minimap of who-is-where out.
**GS-HOTA** the score: HOTA in pitch metres, with team+role+jersey all-or-nothing.
**HOTA / DetA / AssA / LocA** √(DetA×AssA); did you find them / keep them / how precisely.
**DEV-20 / TEST-38 / test-49** tune here / read once per recipe / read once per version, then upload.
**Detection** finding people. **Association / tracking** keeping each one the same person.
**Tracklet** a short continuous track. **Fragmentation** one player split into many.
**Contamination / purity** two players inside one track.
**Homography** the 3×3 matrix mapping image pixels to pitch metres.
**Calibration** estimating that mapping per frame. **Dead frame** one where it failed.
**PnLCalib** our calibrator. **DLT** the classical way to fit a homography from ≥4 points.
**RANSAC** robust fitting: keep the fit most points agree with.
**Bundle adjustment** optimise all cameras and all observations jointly, offline.
**RTS smoother** forward-backward Kalman; uses the future as well as the past.
**YOLO** fast one-stage detector. **mAP** detector quality score. **IoU** box overlap.
**Kalman filter** predicts where a moving box goes next. **ByteTrack / EIoU** our two trackers.
**ReID** recognising the same person again by appearance. **Embedding** a vector summarising a crop.
**CLIP** image encoder trained on 400M image-caption pairs. **ArcFace** the identity loss.
**OCR / STR** reading text from images. **PARSeq** our text recogniser.
**Legibility classifier** "is a number visible here at all?" — what makes abstention possible.
**Pose model / HRNet** finds body joints, so we crop the torso not the grass.
**MILP / Hungarian** assignment optimisation: name tracks without naming two at once.
**VLM** vision-language model. **TTA / BatchNorm stats** adapting to test data without labels.
**Oracle** replace a component with ground truth to price it. **Leave-one-out** drop it and measure.
**Pre-registration** writing the pass/fail bar down, timestamped, before running.
**Wilcoxon signed-rank** the paired per-clip statistical test we use instead of comparing means.
**Noise floor** what "no change" measures — 0.0059 for a seed, +1.02 for 8 days of drift.

---

*End of document. Sources: `docs/GSR_JOURNEY_14_TO_53.md`, `docs/PROF_BRIEF_2026-08.md`,
`docs/CAMERA_INNOVATION_MAP.md`, `docs/GSR_NEXT_PLAN.md`, `docs/LEARNING_GUIDE.md`,
`docs/GSR_METHODS_DEEP_DIVE.md`, `docs/WINNER_REPO_RECON.md`, `STATUS.md` (v6 through v10),
`results/gsr_v9_*.json`, `results/gsr_v10_w*_registered.json`, `results/gsr_benchmark/*.json`,
`knowledge/claims.json` (301 claims). arXiv abstracts fetched 2026-08-17 from the arXiv API;
28 of 28 retrieved, 4 further sources paraphrased and marked.*
