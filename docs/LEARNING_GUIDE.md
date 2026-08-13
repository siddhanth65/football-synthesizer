# LEARNING GUIDE -- what we built, why it stops at 53, what to read next

For Sid, 2026-08-08, after the GSR campaign closed. You own every decision in this project but you
have not typed the internals. This is the internals: each idea gets its searchable NAME, a
plain-English explanation, where it lives in OUR system, and OUR measured number.

Every number comes from `STATUS.md`, `results/GSR_V6.md`, `results/GSR_V7_*.md`,
`results/EVIDENCE_DENSITY_LAW.md`, `results/GSR_{EIOU,TEAMSIDE,S3_READER}.md`,
`results/CLUSTER_SESSION*.md`, `docs/GSR_{CAMPAIGN_BRIEF,CLUSTER_ROADMAP,METHODS_DEEP_DIVE}.md` and
`knowledge/claims.json`. Where the source calls a number an estimate, so does this document.
Sections 1-3 = what we have; 4-5 = why it stops; 6-8 = what you do next.

---

## 1. The task and the score

**GSR = Game State Reconstruction.** Input: a 30-second broadcast clip (750 frames, 25 fps, 1080p,
Swiss Super League 2019). Output: **a minimap** -- for every frame and every person, their position
*on the 2D pitch* (not in the image) plus **role** (player / goalkeeper / referee), **team side**
(does this team defend the left or right goal), and **jersey number** (1..99, or `null` for "I do not
know"). Four jobs at once: find people, keep them the same person over time, map pixels to pitch
metres, and say WHO each one is. The minimap is the product; identity is the hard part.

**GS-HOTA.** HOTA (Higher Order Tracking Accuracy) is the standard multi-object-tracking metric:
`HOTA = sqrt(DetA x AssA)`, where **DetA** = did you find the objects and **AssA** = did you keep
each one's identity continuous. LocA (localisation) rides along as a diagnostic. **GS-HOTA adds one
merciless change:** a prediction matches ground truth only if the position is close enough AND
**team, role and jersey are ALL correct**. Wrong number = zero, however perfect the position.

Three consequences you should be able to recite:

1. **A wrong number is worse than no number.** `null` can still match a GT player the dataset left
   unnumbered; a wrong number matches nothing. Our first jersey attach moved valid-58 from 14.76 to
   19.83 with **0 sequences hurt**, because the reader abstained on 91.3% of tracks. Sparse-but-right
   beats dense-but-wrong.
2. **Jersey errors dominate**, measured twice (Broadcast2Pitch's attribute decomposition and our own):
   about **2.5x** the cost of anything else. The campaign brief's line: *the metric is an identity
   metric wearing a tracking metric's clothes.*
3. **Teams are ranked on GS-HOTA alone** (2024 and 2025 challenge reports). DetA/AssA are the
   dashboard that tells you WHY, never the score.

**Our number: 53.09 on the public test-phase board** (local scorer 53.0846 on the same 49 sequences;
the board rounds), roughly 4th. **Board leader 61.48.** The 2025 challenge-split winner (KIST-GSR =
Broadcast2Pitch, WACV 2026) scored 63.90 and the 2024 winner 63.81 -- different split, do not equate.

### The arc, and the taxonomy that IS the lesson

Read the right-hand column first. It is the most useful thing in this document.

| from -> to | what changed | KIND OF FIX |
|---|---|---|
| 14.76 -> 22.85 (valid-58) | jersey attach, ReID relink, jersey propagation inside merge groups | **new capability** |
| 22.85 -> 33.20 (TEST-38) | the OCR aggregation bug: illegible crops voted one-hot at MAX confidence, out-voting real reads | **bug in our own code** |
| 33.20 -> 31.88 (test-49) | de-leak: we read ground truth in two places (team map by GT agreement, roster from labels). Removed. | **honesty reset, not a jump** |
| 31.88 -> 33.37 | calibration gap fill (DLT re-fit + interpolation across dead frames, max_gap 10) | **engineering patch** |
| 33.37 -> 35.40 | root cause: our own `reject_implausible_frames` (a >=8-player, >=25 m wide-shot rule) killed 96.4% of dead frames that had good homographies. Relaxed to (3, 5 m). | **threshold in our own code** |
| 35.40 -> 39.02 | aggregation floor 0.85 -> 0.80 (+1.09, a DetA lever) + jersey-compatible merge gate at tau 0.080 (+1.65, an AssA lever) | **threshold** |
| 39.02 -> 53.09 | the trained trio: fine-tuned detector, retrained jersey reader, CLIP appearance inside association | **trained models** |

Of seven steps, ONE was "we trained a model". Two were bugs in our own code, two were thresholds in
our own code, one an engineering patch, one an honesty correction. The trained trio is the biggest
single jump (+14.06), but the cheapest points were always **defects we had shipped ourselves**. When
you read a paper, ask: is this a new mechanism, or a repair I could find in my own logs for free?

---

## 2. The pipeline, stage by stage

**SEARCH** = terms to type into a paper search. **OURS** = what we run. **NUMBER** = what it measured.

### 2.1 Detect

**SEARCH:** object detection, one-stage detector, YOLO fine-tuning, transfer learning, mAP@0.5, class
imbalance, domain shift, RF-DETR / RTMDet.

**OURS:** YOLOv8s fine-tuned on GSR train + SoccerNet-v3, classes `{ball, goalkeeper, player,
referee}`. **Role is a detector class here**, so the detector already does an identity job and one
role error costs two attributes under the gate.

**NUMBER (S4b, 20 epochs cumulative, 1.61 GPU-h on one A100):** mAP@0.5 **+20.66**, role-correct
coverage 0.8403 -> **0.9130**, ball mAP 0.2222, **58/58 valid sequences improved recall**, recall on
boxes under 40 px 0.40 -> 0.55. Leave-one-out in the v6 bundle: **-9.06 GS-HOTA** if dropped. Our
biggest component.

### 2.2 Track (associate detections into tracklets)

**SEARCH:** tracking-by-detection, data association, Kalman filter, IoU matching, ByteTrack, BoT-SORT,
OC-SORT, ExpansionIoU / Deep-EIoU, tracklet, track purity, fragmentation, ID switch.

**Plain English:** the detector gives boxes with no memory; a tracker decides which box in frame t+1
is the same person as which in frame t. Classical trackers predict where a box will be (a **Kalman
filter**, a linear motion model) and match on overlap (**IoU**). **ExpansionIoU** is the motion-free
alternative: do not predict, just inflate both boxes by `1+e` about their centres and take IoU, so a
player who moved further than his own width still overlaps himself. **Deep features** add an
appearance vector to the cost so overlapping same-kit players stay separable.

**OURS:** `tools/gsr_eiou.py`, a clean-room ExpansionIoU reimplementation (Deep-EIoU ships no license
file, so nothing was read from their repo). **No Kalman filter** -- a track's matching geometry is its
last *observed* box. A per-track EMA (momentum 0.9) over cached CLIP embeddings is fused into the cost
and used as a hard gate (`app_max`).

**NUMBER -- the decomposition matters more than the total.** DEV-20, from the ByteTrack control 37.0711:

| step | GS-HOTA | delta |
|---|---|---|
| EIoU with plain IoU (e=0), **no** appearance | 35.2242 | **-1.85** |
| + deep features | 38.7134 | **+3.49** |
| + expansion (e=0.3) | 39.0820 | **+0.37** |
| + iterative scale-up (3 rounds) | 38.5641 | **-0.52** |

Held-out TEST-38: 37.0344 -> **39.5403 (+2.51)**, 26/38 helped, Wilcoxon **p = 0.00092**.
Leave-one-out in v6: **-2.09**. Read that table again: the paper is called Deep-EIoU, and the half
named in the title is worth +0.37 while the appearance term is worth +3.49 -- and an appearance-free
rebuild is *worse* than the ByteTrack baseline we already had. That is what "read for the mechanism,
not the headline" means in practice.

**The counterintuitive part.** EIoU **fragments 1.8x more** than ByteTrack (11.47 vs 6.52 fragments
per GT identity) and is **purer** (row purity 0.9416 vs 0.8969; contaminated tracklets 106 vs 173).
That is *better* for us: our downstream assigns ONE jersey per merged tracklet, so a contaminated
tracklet mislabels hundreds of rows, while an extra clean fragment is exactly what the connector
exists to re-merge. **"Better tracker" is a function of your objective, not a property of the tracker.**

### 2.3 Repair tracklets (the connector)

**SEARCH:** tracklet association, offline / two-stage tracking, GTA-Link, agglomerative clustering,
graph-based tracking, merge precision.

**OURS:** `generator/gta_link.py` -- cluster tracklets by embedding distance, merge below `tau`, frozen
at `tau = 0.450` on the CLIP scale. The Splitter half of GTA was **retired**: it flipped sign from
pilot (+0.95) to full split (-0.19), and the paper's `eps = 0.30` produces zero splits on same-kit
football.

**NUMBER:** first landed 22.85 -> 24.18, 51/58 sequences helped. On v6 DEV-20 it makes **1,380 merges
at 0.5432 precision** and realises **43.2%** of the available split+merge headroom (24.5% on the old
stack).

### 2.4 Calibrate (image pixels -> pitch metres)

**SEARCH:** camera calibration, homography, DLT (direct linear transform), RANSAC, sports field
registration, keypoint-and-line calibration, PnLCalib / "No Bells, Just Whistles", reprojection error.

**Plain English:** a homography is the 3x3 matrix mapping image plane to pitch plane, fitted from
correspondences (line intersections, penalty spots). Bad fit or too few landmarks = no minimap for
that frame at all.

**OURS:** PnLCalib per frame, plus `generator/postprocess.fill_calibration_gaps` (re-fit a DLT from
the run's own image/pitch pairs, interpolate across dead frames, `max_gap=10` frozen on DEV-20).

**NUMBER:** PnLCalib lands 0.13-0.20 m on foreign stadiums. After repairs: GT rows recovered in pitch
space **0.7453 -> 0.9155**; frames emitting any pitch output **0.7540 -> 0.9807**; rows detected in the
image then lost at projection **0.1443 -> 0.0101**. Public board 33.37 -> 35.40. **The story to
remember:** the premise was "PnLCalib produces off-pitch homographies", and measurement overturned it
-- that was 1.3% of dead frames. **96.4% of dead frames had good homographies with 100% of players
on-pitch and were killed by our own filter.**

### 2.5 Read jerseys

**SEARCH:** scene text recognition (STR), PARSeq, permuted autoregressive sequence model, legibility
classification, region of interest, pose estimation / KeypointRCNN / ViTPose, torso crop, per-crop vs
per-tracklet aggregation, majority voting, abstention, evidential deep learning, Dirichlet uncertainty.

**Plain English, the chain in order:** crop the player -> a **legibility classifier** (ResNet34)
decides "is a number even visible" -> a **pose model** finds the torso so you crop shirt not grass ->
**PARSeq** reads the digits -> **aggregation** votes all crops of one tracklet into one number.

**OURS:** the Koshkina 2024 lineage with our retrained PARSeq (arm 4t) as a weights-only swap.

**NUMBERS, in the order they were won:**

- **The aggregation bug (the campaign's biggest free win).** Every ILLEGIBLE crop cast a one-hot vote
  at MAXIMUM confidence, out-voting legible reads, so a track needed a *majority* of legible crops to
  commit. Fixing it: read density **d 0.0877 -> 0.2083** at unchanged read precision, **2.38x for zero
  extra GPU**. GS-HOTA 24.19 -> 33.20.
- **Crop geometry.** `estimate_player_box` under-sized crops 0.814x everywhere (its two constants were
  fitted to nothing). Widening x1.25 gave 2.56x confident reads on one match and 1.31x on another,
  precision rising -- but on GSR it measured **-0.38** and was dropped, because GSR crops come from
  real detector boxes and there was nothing to repair. *Same fix, opposite sign, different corpus.*
- **The retrained reader (arm 4t).** Per-crop precision **0.7022 -> 0.8344** at the incumbent's own
  emit rate (0.2761 vs 0.2759): same legibility gate, same torso RoI, same decode, only the PARSeq
  weights differ. Tracklet read density **d 0.3026 -> 0.4148 at precision 0.8228**. End to end on
  DEV-20: **39.0820 -> 42.7449 (+3.66)**, 18/20 helped, **p = 0.00021**, **DetA +4.76 against AssA
  +0.57** -- 89% of the movement is detection accuracy, exactly where a jersey lever must act under an
  identity-gated metric. Leave-one-out in v6: **-5.52**.

**The evidence-density law** (`results/EVIDENCE_DENSITY_LAW.md`; simulated over FOOTPASS's 97,397
labelled events, every component calibrated to a number this project already measured, validated on a
held-out split): to attribute events at precision 0.85 with coverage >= 0.50 you need read density
**d* = 0.347** at read precision 0.86 (0.268 at precision 0.95). Read precision is a **1.3x** lever;
volume is the **4.0x** lever; **7x less fragmentation** buys the same thing -- an exchange rate
between two completely different engineering programmes. Our d = 0.4148 was the first measurement in
the project to clear d*.

**And the negative that reset the operating point.** The denser rule the tracklet gate *selected*
(d 0.4148 at precision 0.8228) scored **0.94 GS-HOTA worse end to end** than the precision-holding
rule (d 0.3139 at precision **0.9256**). Under GS-HOTA a wrong number is not a missing number: it
deletes a detection that `null` would have let match. **Bank the reader's headroom as precision, not
as coverage.**

### 2.6 Assign teams and sides

**SEARCH:** kit clustering, unsupervised colour clustering, k-means on appearance embeddings, team
assignment, oracle / ceiling analysis.

**Plain English:** two steps. (a) Cluster detections into two kit groups -- easy. (b) Decide which
cluster is "left" and which "right" -- one bit per clip, and getting it wrong flips everyone.

**OURS:** two-way kit KMeans, then `meanx`: the cluster with smaller mean pitch-x defends the left goal.

**NUMBER:** 45/49 on the official test split (0.918). **A flip costs about -32.94 GS-HOTA on that
clip** -- annihilation, not degradation. On v6 test-49 the flip set is SNGS-126/130/131/197, and the
solver's own identity metric is near-zero on exactly those four (0.0039 / 0.0244 / 0.0240 / 0.0279)
and normal everywhere else. Estimated headroom **~+3.1 GS-HOTA** (an estimate, banked nowhere).

### 2.7 Solve identities

**SEARCH:** assignment problem, Hungarian algorithm, linear sum assignment, integer linear programming
(ILP/MILP), mutual exclusion constraints, joint inference, message passing, abstention / reject option.

**Plain English:** you have N tracklets and a roster of candidate identities; each tracklet has
evidence (an OCR read, appearance similarity to a gallery). Instead of naming greedily, solve jointly
under a constraint: **two tracklets alive in the same frame cannot be the same player.** That is
bipartite assignment (Hungarian, polynomial time), or an ILP once you add extra constraints.

**OURS:** `generator/identity_solve.py`, a MILP through scipy/HiGHS (no new dependency). The roster is
`roster_self`, built from the sequence's OWN reads, because reading it from labels was one of the two
GT leaks we removed.

**NUMBER, and it is humbling.** On the v6 DEV-20 chain the solver names **610 of 870** merged tracklets
at **13.2 admissible roster slots per sequence**; the other 260 are blocked not by calibration nor by
the solve but by **having no admissible slot** -- nobody in the clip ever read their number. And,
measured three separate times: **the solve buys a dial, not a better frontier.** The greedy baseline
sits ON the solver's precision/coverage curve. Constraints redistribute evidence; they cannot create it.

### 2.8 Score

**SEARCH:** TrackEval, sn-trackeval, held-out evaluation, paired statistics, Wilcoxon signed-rank.

**OURS:** the official `sn-trackeval` scorer, run locally (audited). Three splits, and the discipline
is the point: **DEV-20** (tune here), **TEST-38** (held out inside the public valid split, read once
per frozen recipe), **test-49** (the official test split, read once per campaign version, then
uploaded to Codabench at 1/day).

---

## 3. The three trained components (the v6 trio)

The +14.06 jump. Three models, three different reasons they worked.

**3.1 The detector -- BREADTH.** Learned person/GK/referee/ball boxes across 400 games and 6 leagues,
from GSR train (57 sequences but only **3 distinct games**) plus **SoccerNet-v3** (400 games, 33,986
stills, 371,599 boxes, 98.52% mapping onto our four classes). Why it worked, in one sentence: session
5A had already proved augmentation was not the constraint (colour hardening cut the `player ->
referee` leak 27% and no further, while role coverage FELL 6-10 points on one desaturated game), and
adding real breadth reversed **every** 5A pathology at once, which is what a correct diagnosis looks
like. Riders on record: S4 FAILED its own gate by 20 detections (leak 0.05042 vs a <=0.05 bar,
recorded as FAIL, threshold untouched); S4b passed a freshly declared identical gate after 10 more
epochs; the leak criterion sits inside an ~8x noise band relative to its passing margin, so game 3 is
satisfied-but-marginal, not solved.

**3.2 The jersey reader -- A LABEL WINDFALL, FILTERED.** Learned to read digits off a torso crop, from
an unplanned find: SoccerNet-v3 carries **106,591 pixel-verified jersey-number labels** in its `ID`
field (40 crops were cut and eyeballed; every legible shirt matched its label). They entered training
ONLY through the shipped legibility filter at 0.7, which discards **56.8%** of them, because
annotators carried identity across an action and labelled crops where the number is not visible. Why
it worked, in one sentence: v3 is the *entire* effect -- adding it to the corpus is worth **+0.2861
to +0.3078 precision at identical emit** (arms 3 and 2 against arm 4) -- while the synthetic-digit
pretrain we built first measured **negative** as an init (0.4252 vs 0.4469). Two to internalise:
**the filter is the finding** (raw v3 labels teach hallucination on invisible shirts); and the earlier
jersey head (5B)
collapsed 0.863 train -> 0.125 valid across a boundary that changed only WHICH PLAYERS appear -- it
had learned *recognise-the-player-recall-his-number*, not *read-the-digits* (64% of its errors shared
no digit with the truth). Rule banked: **digit-level supervision decoupled from identity, or nothing.**

**3.3 The CLIP identity encoder -- WHERE it was used.** Learned a 256-d appearance embedding (CLIP
ViT-B/16 backbone, ArcFace identity head, plus per-video team, role and jersey-with-a-no-number-class
heads) from SoccerNet re-ID unified with GSR crops: **160,070 crops / 60,040 identities**, **44.7x**
the identities of GSR train alone. Why it worked, in one sentence: not as a namer (appearance
retrieval on same-kit broadcast is saturated at 0.65-0.71 top-1; three pre-registered naming arms
failed) but as the **appearance term inside association**, worth **+3.49 GS-HOTA** in EIoU's DEV
decomposition. Honest frame: identity mAP 58.75 / R1 76.70 at epoch 8 against the shipped PRTreID
floor of 56.93 -- a **+1.82** retrieval win that converted to **+0.33** GS-HOTA when swapped in
directly. Two independent architectures agreed **GSR train's 1,343 identities cannot build a
competitive embedding**; scale was the fix. Scale lesson: CLIP's cosine space is ~4.8x wider than
PRTreID's, so at the inherited `tau = 0.080` the swap read as a 4-point FAILURE until tau was
re-swept to 0.450. **Swap an embedding model and every distance threshold downstream is now wrong.**

---

## 4. Why we stopped at 53 -- four limits, each with its evidence

Not walls. Each is a precise research question with a price tag.

### 4.1 The four side-flip sequences (~3.1 points) -- TRIPLE-LOCKED

1. **Geometry is at its ceiling.** Run `meanx` on GROUND-TRUTH positions -- perfect detection,
   clustering and calibration -- and it scores **112/115 on dev and the same 45/49 on test**. The
   residual failures are not estimation errors, they are **clips whose ground-truth geometry inverts**.
   Also dead by construction: solve-both-permutations-and-vote, because the solver is
   permutation-invariant in cluster space (max posterior delta **0.0** on 6/6).
2. **Goalkeeper appearance is anti-informative.** The one rule above 0.98 is `gk_self` -- "a keeper's
   team defends the goal he stands in" -- **113/113 on ground truth**. Unshippable: two-way kit
   clustering puts a detected keeper in his OWN team's cluster only **24.4%** of the time
   (systematically the opponent's, ~77%). Keeper kits are *required* to differ from both outfield
   kits, so similarity-to-outfield runs backwards for keepers. (Correction on record: the first
   "anti-informative mechanism" claim was an artifact of a BGR channel-order bug and was WITHDRAWN;
   re-run under RGB the verdict survived as "nothing", 0.5395 against a 0.5329 majority floor.)
3. **A learned side detector is geometry re-derived.** V2 fine-tuned a 6-class side-aware detector on
   v3's `team left`/`team right` labels (0.96 GPU-h). It ties `meanx` **exactly** (91/97 vs 91/97),
   agrees on 89/97, and is **wrong on 2 of the 3 GT-inversion clips it was built for**, at confident
   vote shares 0.7123 and 0.7613. Margin AUC 0.839 vs meanx's 0.843. It learned something real
   (per-box side accuracy 0.7713 where a pure image-x shortcut gets 0.5983) and collapses to geometry
   at sequence level.

**The research question:** *is there any observable inside a 30-second clip that determines attacking
direction independently of visible-player pitch geometry?* Attack direction itself is 0.52 (a coin
flip) even on GT. Untried here: goal-mouth / net detection, scoreboard, crowd-side asymmetry,
pitch-landmark identification (which penalty box IS this).

### 4.2 Jersey coverage -- the frontier is INTERIOR on both sides

- **The emit ceiling.** **12,814 of 67,224 DEV-20 crops (19.1%)** pass the shipped 2024 ResNet34
  legibility gate; everything downstream reads only those. That classifier has never been retrained
  (the retrain arm was scoped and never run). Every density number in this campaign is conditioned on
  it.
- **The frontier is interior.** Denser is worse (arm B, -0.94, section 2.5) AND more precise is worth
  a lot: at matched density, +0.0073 read precision bought **+1.2562 GS-HOTA** on DEV-20. You are on a
  curve, and the optimum is inside it.
- **The teacher-imitation trap killed the free supervision.** Grad's CVPRW 2025 claim is that
  illegible crops give free negatives. We built it -- class 0 ("no number") defined by crops that FAIL
  the legibility filter -- and the head learns to **imitate its teacher**: invisible-class AUC
  **0.6200**, below the frozen trunk's own 0.6429 and well below the 2024 legibility classifier's
  **0.7038** it was meant to replace. **A pseudo-label transfers the labeller's decision boundary, not
  the truth.** That closes the cheapest imaginable route to more emits.

**The research question:** *what supervises "is a number visible in this crop" independently of the
2024 classifier?* Human annotation on a few thousand crops; multi-view consistency across a tracklet;
a self-supervised agreement signal.

### 4.3 Association -- a measured +11.9 we could not reach

Re-measured on the v6 stack (CPU only, no tracker run):

| counterfactual | AssA headroom |
|---|---|
| perfect linking (oracle may split AND merge) | **+24.79** |
| merge-only (what a connector alone can do) | **+11.89** |
| perfect coverage + contamination removal | +8.51 |

**Ratio 2.91:1 in favour of fragmentation; 19 of 20 sequences fragmentation-dominated.** The ceiling
rose because of the DETECTOR and the CALIBRATION repair, not EIoU (the detector swap moves it
0.8064 -> 0.8627; EIoU moves it 0.8627 -> 0.8642, i.e. nothing). **And the connector destroys part of
its own ceiling:** at **0.5432 merge precision** its wrong merges lock in contamination no later
merge-only pass can undo, so the shipped partition's merge-only bound (0.6936) is BELOW the tracker
output's (0.7415). It has spent 61% of the merge-only headroom and burned 0.048 of what remained.

**Why CAMELTrack failed HERE -- an input-representation story, not a method story.** Their released
checkpoints' appearance tokenizer is `nn.Linear(128, 512)` fitted to **KPReID part embeddings: 5 body
parts + 1 global, 128-d each, with per-part visibility scores**. Our cache is a **single global 256-d
CLIP vector**. You cannot reshape, pad or truncate one basis into the other and retain meaning, so
every multi-cue checkpoint fell through its own `drop_app` path and ran **appearance-less**:
**-13.6 to -20.2 GS-HOTA, 0/20 helped**. The mechanism is measured three ways: row purity collapses to
0.61-0.83; `assa_merge_only` sits barely above `assa_hat` (0.3593 vs 0.3342 -- nothing left to
recover, contamination cannot be un-merged); and the connector *correctly refuses* to merge, finding
26 merges instead of 1,380. A from-scratch retrain on our cues learned the task (val association
accuracy 0.6906 -> **0.9221** on held-out GSR sequences) and still scored -18.6 end to end --
**trained on ground-truth boxes, never on detector output**, named as the likely cause. Caveat on
record: this measured *CAMELTrack-WITHOUT-its-appearance-cue*; a faithful trial needs KPReID parts +
COCO-17 pose + a detector-output corpus, scoped at ~8-12 GPU-h and declined on budget.

**The research question:** *the rank-1/rank-2 join.* Joining each identity's largest fragment to its
second-largest alone moves the dominant share 0.563 -> 0.738. It needs a merge precision our
appearance model cannot reach: PRTreID's within-identity p90 is 0.093 against a cross-identity p10 of
0.040-0.083 -- **overlapping distributions**. That is the case for a learned association model, not
another threshold.

### 4.4 The statistics -- why DEV-20 cannot decide sub-point questions

- **v5.1:** a solver retune scored **+0.6118 on DEV-20** and **-0.0527 on held-out TEST-38**
  (p = 0.438); two sequences supplied 75% of the DEV gain. The pre-declared gate (>= 37.2) was not met,
  so no test-49 run and no submission were spent. The gate paid for itself in one session.
- **The transfer ratio we measured:** V3's evidential reader went **+1.2562 DEV -> +0.3753 TEST-38**,
  i.e. **30% transferred** -- and the *mechanism reversed sign*: the read-precision advantage V3 said
  caused the gain (+0.0073 on DEV) became **-0.0070** on TEST-38. A 450-point DEV sweep had fitted
  DEV's own density/precision frontier, not a property of the reader.
- **Multiplicity.** V3 scored 9 DEV arms: p = 0.0355 uncorrected against a Bonferroni bar of 0.00625.
  V4s2 swept 6 points: p = 0.0401 against 0.0083. The final push registered alpha 0.025 per arm and
  ARM 2's p = 0.0421 missed by 1.7x.
- **The concentration guard:** any gain whose top-2 sequences carry >= 80% of the net is demoted. V1's
  best arm had concentration **1.92** (top-2 = +5.91 on a net of +3.07).
- **The counter-example that keeps this honest:** EIoU's held-out gain was **larger** than its DEV gain
  (+2.51 vs +2.01, p = 0.00092). Gates are not pessimism. They are measurement.

**The research question:** *how large a dev split resolves a +0.5 GS-HOTA effect on 30-second clips?*
Nobody in this literature reports it. You have the harness to measure it.

---

## 5. The six v7 negatives, as six lessons

Five pre-registered gates failed, plus the two-arm confirmatory push. Total GPU spend: **~5.6 hours of
42 budgeted** -- the gates killed everything at ~13% of planned cost. That is the system working.

**1. V1, the solver retune -> CONFIGURATION-SPACE CORNERS.** *Tried:* 18 arms over 6 solver priors on
the now-dense (d ~ 0.31) chain. *Failed because:* at `r_abstain = 0` the solver sits in the
name-everything-admissible corner, and a label-change probe showed `p_correct`, `pi_none` and
`sim_none` move **0 of 870 tracklet labels** -- the unknown column's payoff is pinned, so priors scale
the whole identity block against one fixed entry. Nine of eighteen arms scored the control to 16
significant figures, and the only exit knob (`r_abstain`) exits in the losing direction because
GS-HOTA prices coverage above precision. *Concept:* **before sweeping a parameter, verify it changes a
DECISION, not just a score.**

**2. V2, the learned side detector -> LABEL-INFORMATION CEILINGS.** *Tried:* a 6-class side-aware
detector on v3's side labels. *Failed because:* the label is itself a function of pitch geometry, so
the model becomes a second estimator of the same latent quantity -- its +4 fixes are meanx's
low-margin estimation errors, its -4 breaks its own confident inversions. *Concept:* **ask what your
label is a function of.** A new estimator of an old quantity is not a new signal. (Side yield:
splitting the classes costs box quality, person AP 0.9723 -> 0.9596, which vindicates the overlay
design that left the shipped detector untouched.)

**3. V3, the evidential jersey head -> TEACHER-STUDENT SUPERVISION LEAKAGE.** *Tried:* a Dirichlet
evidential head learning abstention inside the recognizer, negatives harvested from crops the
legibility model rejects. *Failed because:* those negatives ARE the teacher's decisions, so the
student learns the teacher's boundary and lands below it (0.620 vs 0.704); rung 3 then failed by one
sequence (+1.2562 on 11/20 against >= +1.0 AND >= 12/20). *Concept:* **free supervision from another
model's outputs carries that model's ceiling.** (Two findings survived: the uncertainty term alone
pays +0.36 end to end; and the registered architecture -- a free 100-way MLP over pooled trunk
features -- scored 0.4935 against its own trunk's 0.8344, because an MLP smooths PARSeq's sharp
combinatorial `d0*10+d1` map. The fix was a 2-parameter GATE head preserving the trunk's argmax by
construction.)

**4. V4s2, CAMELTrack -> INPUT-REPRESENTATION CONTRACTS.** *Tried:* the strongest published
learned-association tracker, zero-shot then from scratch. *Failed because:* the pretrained appearance
tokenizer reads a KPReID 6x128 part basis and we handed it nothing, so it ran appearance-less; the
from-scratch arm trained on GT boxes did not transfer to detector output. *Concept:* **read the input
contract before the results table**, and if you cannot satisfy it, say out loud what you actually
measured. (We did: "this measured CAMELTrack-WITHOUT-its-appearance-cue".)

**5. THE PUSH -> OUT-OF-SAMPLE TRANSFER.** *Tried:* one confirmatory held-out TEST-38 read of V3's
near-miss, two arms, registration frozen 6m13s before the first artifact, controls re-derived to delta
**0.0** at full float precision. *Failed because:* 30% of the DEV effect transferred and V3's stated
mechanism reversed sign; ARM 2 cleared 1 of 3 criteria. Neither shipped; no test-49 spent. *Concept:*
**confirm the MECHANISM out of sample, not just the score.** A held-out win with a dead mechanism is a
coin that landed your way.

**6. The registration system itself -> PRE-REGISTRATION.** Every gate above was written down, with
thresholds and a kill rule, BEFORE the arm ran. That is why "FAIL by 0.0004" (S4's detector gate, 20
detections) is recorded as a FAIL with the threshold untouched, and why S4b is a separate attempt
under its own fresh gate rather than a re-reading of the old one. *Concept:* **a threshold you can
move after seeing the data is not a threshold.**

---

## 6. What to learn, ranked

In the order that pays for this project. arXiv ids only where they are on record in
`docs/GSR_METHODS_DEEP_DIVE.md` or certain; omitted rather than guessed.

**(i) The MOT literature stack -- the spine of the task.** *HOTA: A Higher Order Metric for Evaluating
Multi-Object Tracking* (Luiten et al., IJCV 2021), arXiv 2009.07736 -- read the DetA/AssA
decomposition until you can derive why a wrong jersey is worth less than an abstention. *ByteTrack*,
arXiv 2110.06864. *OC-SORT*, arXiv 2203.14360 and *BoT-SORT*, arXiv 2206.14651 -- two answers to "the
Kalman filter is the problem". *Iterative Scale-Up ExpansionIoU and Deep Features* (Deep-EIoU), arXiv
2306.13074 -- read against our decomposition in 2.2. *CAMELTrack*, arXiv 2505.01257 (Apache-2.0, code
released) -- read 4.3 first so you know what broke. *GTA-Link*, arXiv 2411.08216 (MIT) -- our
connector. *Unifying Short and Long-Term Tracking with Graph Hierarchies* (SUSHI, CVPR 2023) -- the
offline graph-hierarchy family, scoped as "v8" in our own plan and never run.

**(ii) Re-identification and metric learning.** *In Defense of the Triplet Loss for Person
Re-Identification*, arXiv 1703.07737. *ArcFace*, arXiv 1801.07698 -- what our encoder's identity head
uses. *BPBReID*, arXiv 2211.03679, and *PRTreID* (part-based re-ID + role + team, MMSports'23), arXiv
2401.09942 -- part embeddings are exactly what CAMELTrack wanted from us. *CLIP*, arXiv 2103.00020,
and *CLIP-ReIdent*, arXiv 2303.11855 -- our encoder's lineage. *A Metric Learning Reality Check*
(Musgrave et al.), arXiv 2003.08505 -- read for methodology as much as content; a subfield's
retrospective on unfair comparisons.

**(iii) Scene text recognition and uncertainty.** *PARSeq: Scene Text Recognition with Permuted
Autoregressive Sequence Models*, arXiv 2207.06966 (Apache-2.0) -- understand the permuted
autoregressive decode; it is why an MLP over pooled features could not reproduce it (lesson 3 above).
*A General Framework for Jersey Number Recognition in Sports Video* (Koshkina & Elder, CVPRW 2024),
arXiv 2405.13896 -- our entire OCR chain's lineage, code exists, CC BY-NC 3.0. *Evidential Deep
Learning to Quantify Classification Uncertainty* (Sensoy et al., NeurIPS 2018), arXiv 1806.01768 --
the Dirichlet formulation V3 implemented; note our finding that their `kl_max = 1.0`, fitted on
10-class MNIST, **collapses a 100-class head to alpha = 1 everywhere**. *Grad, CVPRW 2025* -- 85.62
tracklet accuracy on the GSR challenge split vs Koshkina's 79.31, via a torso-crop ViT, digit-aware
tied head and Dirichlet abstention; **no code, no arXiv mirror**, our V3 is the reimplementation and
it refuted one of its claims.

**(iv) Assignment problems.** The Hungarian method (Kuhn 1955; Munkres 1957) -- no arXiv, read a
modern lecture treatment; `scipy.optimize.linear_sum_assignment` is this. Integer linear programming
for tracking / joint identity inference -- our solver is a MILP through scipy/HiGHS; read the HiGHS
docs plus one MOT ILP formulation to see how mutual exclusion and abstention become constraints and
slack variables. *Learning a Neural Solver for Multiple Object Tracking* (Braso & Leal-Taixe, CVPR
2020) -- message passing on a tracking graph; this family could replace our connector AND our solver
with one learned object.

**(v) Experimental methodology -- do not skip; this is where the campaign's value is.**
Pre-registration as practised in clinical trials, and *False-Positive Psychology* (Simmons, Nelson &
Simonsohn, Psychological Science 2011) for what happens without it. *Controlling the False Discovery
Rate* (Benjamini & Hochberg, JRSS-B 1995) -- the right alternative to Bonferroni when you sweep many
arms. *Accounting for Variance in Machine Learning Benchmarks* (Bouthillier et al., MLSys 2021) -- how
much of a reported delta is seed noise. Paired non-parametric tests: Wilcoxon signed-rank, McNemar --
know why paired per-sequence statistics beat comparing two means.

---

## 7. Reading a new paper against OUR bottlenecks -- the innovation checklist

Five questions in order. Fail Q1, stop reading.

**Q1. Which of our four measured gaps does it touch?** (a) side flips, ~+3.1, needs a NEW OBSERVABLE;
(b) jersey emit ceiling 19.1%, needs supervision independent of the 2024 legibility model;
(c) association, +11.89 merge-only headroom, needs a high-precision rank-1/rank-2 join;
(d) statistics, needs a dev split that resolves sub-point effects. A paper improving something we are
already good at (LocA is 94.06) is worth nothing here.

**Q2. What does it need as INPUT -- do we have that representation?** The CAMELTrack lesson, which
cost 3 GPU-hours to learn. Find the tokenizer / feature spec before the results. If it wants
`(6, 128)` part embeddings with visibility scores and we hold one global 256-d vector, the honest
options are "build the representation" or "do not run the trial" -- never "run it degraded and report
the number".

**Q3. What SUPERVISION does it need -- do we have those labels, legally?** Do the labels exist (v3's
106,591 jersey labels sat undiscovered for weeks)? Are they trustworthy per-crop or only per-identity
(v3's were per-identity: 56.8% sit on illegible crops)? And the license: GSR's own is listed
inconsistently (GPL-3.0 vs CC BY-4.0, unresolved), Koshkina's pipeline is CC BY-NC 3.0, Deep-EIoU
ships none (hence our clean-room rebuild), CAMELTrack is Apache-2.0 (hence used as a dependency).
Record the license BEFORE using the code.

**Q4. What COMPUTE?** Verified: **no winning GSR system ever used more than one GPU.** The 2024 winner
trained everything on a single A100 40GB; the 2025 winner on a single RTX 4090, including a one-epoch
LLaMA-3.2-Vision fine-tune. A method needing 8 GPUs is out of scope AND probably not the reason anyone
won.

**Q5. Is there CODE, under what license?** No GSR participant in 2024 or 2025 released code or weights
(verified across both challenge reports and the CVF papers). Assume reimplementation; budget for it.

**The meta-rule: read for MECHANISMS, not headline numbers, and always demand the ablation table.**
Our whole campaign was steered by ONE table -- Broadcast2Pitch Table 5, detector/tracker/calibration
held fixed, only the identity model varying:

| identity model | GS-HOTA |
|---|---|
| PRTreID + EasyOCR + ResNet-18 colour | 18.11 |
| CLIP encoder + attribute heads | 60.13 |
| LLaMA-3.2-Vision, 1-epoch fine-tune | 61.48 |

Worth more than any full paper we read: it told us the identity model IS the benchmark, and that the
11B VLM buys only +1.35 over the CLIP recipe -- so we never spent a GPU-hour on the VLM.

**And remember what actually scored here:** a **bug fix** (the OCR aggregation one-hot, 2.38x density
for zero GPU), a **threshold fix** (crop scale, and the 0.85 -> 0.80 aggregation floor), and a **data
finding** (v3's jersey labels). "Simple new improvements" is not the consolation prize. It is exactly
how this project got to 53.

---

## 8. Where the residual map points

**1. `w_app`, the one living knob.** The appearance weight in EIoU association: **+0.2530 GS-HOTA**
isolated on identical evidence (20/38 helped, p = 0.1102), right-signed on two independent splits
(DEV +0.4853 at 12/20; TEST-38 +0.6282 at 23/38), **never singly tested with power.** *If you find a
paper about learned cue weighting, adaptive motion-vs-appearance gating, or uncertainty-weighted cost
fusion in MOT, it might unlock this.* Probably the cheapest legitimate win left: one pre-registered
confirmatory test of this single number at a dev scale that can resolve it.

**2. Faithful CAMELTrack.** ~8-12 GPU-h: build KPReID part embeddings + COCO-17 pose on our cached
boxes and retrain on **detector output** rather than GT boxes. *If you find a paper about part-based
or keypoint-promptable re-ID, or about the train-on-GT / infer-on-detections distribution gap in
tracking, it might unlock this.* The +11.89 merge-only headroom is still sitting there.

**3. SNGS-082.** The worst single-sequence regression of the final session: control 60.7732,
ARM 1 56.5568 (-4.216), ARM 2 52.9352 (-7.838). **Not diagnosed.** *If you find a paper about
per-sequence failure analysis or hard-case mining in tracking benchmarks, start by watching the clip.*
One undiagnosed pathology on a 49-sequence board is worth more than it looks.

**4. The side flips.** Triple-locked (4.1), awaiting a **genuinely new observable**. *If you find a
paper about attacking-direction inference, goal-mouth or net detection, scoreboard reading,
camera-pose priors, or pitch-landmark identification (which end am I looking at), it might unlock
this.* Estimated +3.1, concentrated in four known clips.

---

### Appendix: the four terms not defined above

**fragmentation** -- one real player split across many tracklets (ours: 7.91 per GT identity).
**contamination / purity** -- two real players inside one tracklet (ours: 0.9395 row purity).
**oracle / ceiling analysis** -- replace one component with ground truth and re-score, to price what
fixing it could ever be worth; the most useful technique in this whole campaign.
**leave-one-out ablation** -- drop one component from the shipped bundle and measure the loss
(ours: detector -9.06, reader -5.52, association -2.09).
