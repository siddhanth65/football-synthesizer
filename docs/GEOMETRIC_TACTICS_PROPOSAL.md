# GEOMETRIC_TACTICS_PROPOSAL -- team-level tactics, restored

Written 2026-07-26 for Sid and his supervisor, ahead of the December 2026 BTP review.
Reads with `docs/PROJECT_OPTIONS_FINAL.md` (which it extends, not replaces) and
`results/MANUTD_IDENTITY_PROFILE.md` (which is the evidence base).

The original goal -- **team-level tactical metrics: pressing, ball recovery, shape** -- was never
refuted. What was refuted is a *different question* that happened to use the same words. This document
separates the two, then builds the project back around the half that is still open.

---

## 1. The correction, stated first

There are two questions in this project and they have been sharing a name.

> **DESCRIBE.** In *this* match, what shape did each team hold, in and out of possession, and how did
> the two structures sit against each other?

> **DISCRIMINATE.** Does Manchester United press higher / sit deeper / stay more compact **than other
> teams do**, as a stable property of the team?

Every closed result in this repository closed the second question. Not one of them closed the first.

### 1.1 The reason is the unit of analysis, and it differs by three orders of magnitude

| question | independent unit | how many we have | typical SE |
|---|---|---|---|
| DISCRIMINATE (team A vs team B on a rate or a mean) | **the match** | 12 (16-18 planned) | between-match SD 4.25 m on a 28.3 m block line -> a team mean carries +/- 1.2 m at n=12 |
| DESCRIBE (this team, this phase, this match) | **the window / the frame** | thousands per match (block coverage 0.815-0.977 of trusted frames per match) | a 2-minute window aggregates ~1,200-6,000 player-frames |

`results/W1B_WINDOW_AND_SAMPLING.md` computed the discrimination floor honestly: separating two teams
on a counterpress **rate** needs **58-90 matches even with perfect tracking and zero measurement
error**. That number is a property of binomial variance and between-match spread. It is not a
statement about the CV pipeline, and no amount of pipeline work moves it. At n=12 it is closed; at
n=18 it is still closed. Accept it once and stop paying for it.

But that floor prices the *contrast between teams*. It does not price a measurement made inside one
match, where the same machinery has thousands of independent-ish windows instead of twelve matches.
The profile already demonstrates the difference in both directions:

- **Description works.** `results/MANUTD_IDENTITY_PROFILE.md` reports the block line per match with
  enough resolution to order twelve matches across a 16.6 m range (20.8 m at Anfield to 37.4 m at
  Southampton) and to show a +7.5 m climb from loss-likely to win-likely game state. The per-match
  numbers separate; that is a descriptive claim and it stands.
- **Discrimination fails on the same data.** United's block-line SD is 4.25 m, the pooled six-opponent
  set's is 3.09 m; F = 1.89, p = 0.305 -- cannot be separated. Compactness: 6.17 +/- 0.39 m for
  United, 5.93 +/- 0.29 m for the opposition, F = 1.81, p = 0.34 -- cannot be separated. The
  measurement was fine. The comparison was underpowered.

### 1.2 Which negative result closes which question

| measured negative | what it actually closes | what it leaves open |
|---|---|---|
| Per-player event attribution 0.36-0.40 precision (95% CI 0.20-0.59) at 13-14% coverage; elimination constraints +0.04; jersey OCR **0.000** and silent (`results/PLAYER_ANALYSIS_v2.md`, `results/CARRIER_CONSTRAINED_v2.md`) | Anything of the form *named player + event verb*: pass counts, involvements, per-player pressing actions, passing networks with real names | **Everything positional.** Roles, shape, formation, block geometry, coupling -- none of these need a name. `fingerprint/roles.py` already assigns positional roles with zero identity |
| Team-vs-team **rate** comparison needs 58-90 matches (`results/W1B_WINDOW_AND_SAMPLING.md`) | Every cross-team claim of the form "United press more than X"; every league ranking; any unpaired between-team mean or variance test at n<=18 | **Within-match** contrasts (in-possession vs out-of-possession shape), **within-pair** contrasts (same opponent, both legs), and per-match description. These are paired designs and they are powered -- see 1.3 |
| Ball-movement style identity at chance (0.168 vs a 0.191 null); needs >=8 s possession strings, we have 3 s (`results/ENTROPY_MAP_FEASIBILITY.md`) | The *ball-trajectory* style fingerprint, and any representation that needs long uninterrupted possession strings | **Player-geometry** representations. Player positions are available continuously at 11.57 trusted rows/frame of 22, in and out of possession; they do not depend on possession-string length at all. This failure was about the ball layer, not the shape layer |
| Ball post-link coverage 32-52% (pooled 38.8% at 1 Hz), 47.4% of ball frames carry zero trusted player row | Anything requiring a continuous ball track: entry-channel chains (43 in 12 matches), ball-relative metrics at frame resolution | Shape and formation, which need no ball; and phase splitting, which needs only *which team has it* -- the Viterbi possession label, already validated by pass counts at 0.97-1.09x official on 10/12 matches |
| "United are rigid / attack down the left / hold a compact block" all dissolved by opponent control | Team **identity** claims | The same measurements as **per-match descriptions**. "In this match United defended at 29.6 m and Brighton at 24.1 m, and the gap moved by X after the goal" is not an identity claim and does not need the opponent set to fail to reproduce it |

### 1.3 What is powered at 16-18 matches, quantitatively

The corpus is not useless across matches -- it is useless for *unpaired* comparisons. Paired designs
on the same footage are strong, and the profile already proves it:

- **Within-match paired contrast**, sign test over matches. The own-third-to-final-third funnel came
  back positive in 11/12 matches, two-sided exact sign test **p = 0.0063**; the opposition's in 12/12,
  **p = 0.0005**. The same design applied to "in-possession shape is longer/wider than out-of-possession
  shape" gives p = 0.0005 at 12/12 and p ~ 8e-6 at 18/18. This is the legitimate cross-match claim
  and it costs nothing extra.
- **Within-pair contrast** (same opponent, home and away legs). Six pairs now, eight or nine with the
  planned additions. All-same-direction gives p = 0.031 at 6 pairs, p = 0.0039 at 9. Enough for a
  consistent effect; not enough for a small one. Say which in advance.
- **Unpaired between-team means or variances.** F = 1.89 / p = 0.305 at n=12 was not bad luck. Do not
  run these again.

### 1.4 The literature agrees the descriptive target is the right one

Two independent confirmations, both citable in the thesis:

- Cross-provider **formation-label agreement is 30%** (13 of 44 formations agree across StatsBomb,
  Wyscout and Stats Perform); tactical-position labels agree 79% (Sotudeh 2024, survey, below). Chasing
  classification accuracy against one provider's label taxonomy is chasing a coin-flip target. The
  defensible claim is "this shape, with this confidence", not "correct per Opta".
- Even with full tracking, the best-validated formation method (SoccerCPD, KDD 2022) matches expert
  annotation on **72.4% of playing time**. That is the ceiling with perfect data. Any framing that
  needs 90%+ label accuracy is dead before broadcast occlusion is considered.

**So: the original goal is intact.** Pressing, ball recovery and shape as *rates that separate teams*
are closed. Pressing, ball recovery and shape as *geometry measured in a match and described with an
interval* are open, unsolved in the literature under broadcast conditions, and buildable on what is
already on disk.

---

## 2. The reading list

Eight papers, ranked by how much they change what gets built. Read **#1 and #2 first** -- one gives
the estimator, the other gives the argument for why the target is a distribution and not a label.

### #1 Bekkers (2025), EFPI: Elastic Formation and Position Identification -- READ FIRST

*Joris Bekkers, "EFPI: Elastic Formation and Position Identification in Football (Soccer) using
Template Matching and Linear Assignment", arXiv:2506.23843, June 2025.*
<https://arxiv.org/abs/2506.23843> -- code: <https://github.com/UnravelSports/unravelsports>

Matches observed player positions against 65 static formation templates (scaled to actual pitch
dimensions) by linear sum assignment, minimising total assignment distance; the lowest-cost template
names the formation and the assignment names each player's role. No training, no event data, no ball.
It runs at any granularity -- single frame, per possession, per window, per period -- with an optional
stability parameter to stop frame-to-frame flip-flopping.

**Data requirement:** outfield (x, y) only, single match, goalkeeper excluded from the assignment.
Critically, its template bank **includes 8-, 9- and 10-outfield templates specifically to handle
missing players**.

**How it applies here:** this is the same algorithm `fingerprint/roles.py` already implements
(Hungarian match against four templates in absolute attacking coordinates), with a 65-template bank
and a variable-N story. Two things to take: the wider template bank, and the honest gap -- EFPI's
reduced-N templates were designed for *red cards and injuries*, i.e. a team that genuinely has nine
players, **not** for a full team of which nine happen to be in shot. It reports **zero accuracy
numbers**. That gap is the project: the first validation of variable-N template matching under real
broadcast occlusion would be Sid's, not theirs.

### #2 Sotudeh (2024), The Principles of Tactical Formation Identification -- READ SECOND

*Hadi Sotudeh, "The Principles of Tactical Formation Identification in Association Football (Soccer)
-- A Survey", Frontiers in Sports and Active Living, 2024.*
<https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1512386/full>

A two-decade taxonomy of formation identification -- preprocessing, team-level (template/clustering on
average positions), position-level (per-player relative-location methods) -- with an explicit audit of
how rigorously the field reports itself.

**Data requirement:** none, it is a survey.

**How it applies here:** it supplies the two load-bearing sentences of the thesis introduction.
(1) Cross-provider formation agreement is **30%** and position-label agreement **79%** -- the ground
truth everyone validates against is itself provider-dependent, which is why the deliverable must be a
*distribution over shapes with a confidence*, not a label with an accuracy. (2) The survey states,
field-wide, that assuming a fixed outfield count "fails to address scenarios with fewer players" and
that **none** of the reviewed methods handle reduced or partial player counts, and that most do not
report accuracy, runtime or storage at all. That is an independent, citable confirmation that the gap
this project targets is genuinely open. Take this one to the supervisor meeting.

### #3 Kim, Kim, Chung, Yoon & Ko (2022), SoccerCPD

*"SoccerCPD: Formation and Role Change-Point Detection in Soccer Matches Using Spatiotemporal Tracking
Data", ACM SIGKDD 2022.* <https://arxiv.org/abs/2206.10926> -- code:
<https://github.com/hyunsungkim-ds/soccercpd>

Two-stage change-point detection: frame-level role assignment (Bialkowski-style) produces a sequence of
role-adjacency matrices; change-point detection on that sequence segments the match into stable
formation periods; a second pass catches role permutations (two centre-backs swapping sides) inside a
formation period. Frames with abnormally high role-switch rates (>0.7, typically set pieces) are
dropped automatically as noise.

**Data requirement:** (x, y) at 10 Hz; **at least one moment per session where all ten outfielders are
simultaneously measured** to seed the role assignment, and near-complete tracking thereafter.

**How it applies here:** three things. It is the **only** method in this space with real validated
numbers (809 match halves, 864 formation periods; 72.4% of playing time matching expert annotation,
86.5% of player-minutes on position labels) -- so it sets the realistic ceiling and stops anyone in the
viva expecting 95%. Its change-point framing is exactly "how did the two structures evolve through the
match", which is the descriptive question. And its noise filter -- drop frames whose role-switch rate
is anomalous -- is the template for dropping windows where too many players are off-screen. Its
seeding requirement (one fully-observed frame per half) is a real risk to check against the corpus in
week 7.

### #4 Bialkowski, Lucey, Carr, Yue, Sridharan & Matthews (2014), role-based representation

*"Large-Scale Analysis of Soccer Matches Using Spatiotemporal Tracking Data", IEEE ICDM 2014.*
<https://www.iainm.com/assets/pdf/Bialkowski-2014c.pdf>

The origin of the role-based representation: at every frame players are dynamically re-labelled to the
*role* (not the shirt) that minimises the entropy of role-specific occupancy maps, solved as a min-cost
assignment. Template-free, usable per-frame or aggregated. Everything downstream -- Shaw & Glickman,
SoccerCPD, EFPI -- descends from it.

**Data requirement:** full-season multi-camera tracking (~400M points); the entropy-minimising
assignment is global over all players every frame, so it implicitly assumes a fixed, known outfield
count. Degraded conditions are never tested.

**How it applies here:** it is the citation the supervisor will expect, and it is the *contrast* case
in the thesis -- the method whose central assumption (fixed known N every frame) is precisely what
broadcast breaks. Worth implementing on the Metrica full-truth data as the "what it looks like with
complete tracking" reference arm, and not worth fighting to make work at 11.6 of 22.

### #5 Omidshafiei, Hennes, Garnelo et al. (2022), Graph Imputer

*"Multiagent off-screen behavior prediction in football", Scientific Reports 12:8638, 2022 (DeepMind +
Liverpool FC).* <https://www.nature.com/articles/s41598-022-12547-0>

Graph network + VAE that reconstructs off-screen player trajectories from the visible subset,
conditioning on past **and** future observed context, learning a distribution rather than a point
estimate, evaluated against a simulated broadcast camera that masks players in and out of view.

**Data requirement:** full ground-truth tracking for all 22 for training, with a synthetic view cone
applied; 105 EPL matches, and sequences are kept only when **all 22 players' ground truth** is present.
Not trainable here -- see `docs/PROJECT_OPTIONS_FINAL.md` Part 0.1 for the full audit.

**How it applies here:** two specific uses, and one of them has probably not been extracted yet.
First, its camera model yields **12.76 +/- 3.70 visible of 22**, an independent confirmation of this
project's 11.57/22. Second, and more important for this proposal, it is **the one paper that measures a
team-level spatial metric (pitch control) computed on imputed positions against the same metric on
ground truth** -- i.e. it already answers, for one metric, the question this project asks for a whole
battery. Re-read it specifically for the pitch-control-vs-N-missing result; `fingerprint/pitch_control.py`
already exists and can be pointed at the same comparison in week 4.

### #6 Memmert, Raabe, Schwab & Rein (2019), geometric metric battery

*"A tactical comparison of the 4-2-3-1 and 3-5-2 formation in soccer: a theory-oriented, experimental
approach based on positional data in an 11 vs. 11 game set-up", PLOS ONE, 2019.*
<https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0210191>

Defines and computes the full geometric battery: Effective Playing Space (convex-hull area, GK
excluded), Player Length-per-Width ratio, Team Separateness (mean distance to nearest opponent),
Voronoi space-control gain, Pressure Passing Efficiency. Compares two formations in staged 11v11.

**Data requirement:** 22-player optical/Kinexon tracking at 25 Hz downsampled to 1 Hz. Reported
completeness was **0.88 +/- 0.13** -- about 12% of positions missing *even with a purpose-built rig* --
and those gaps were repaired by plain linear interpolation with **no sensitivity analysis**.

**How it applies here:** it is the formula sheet -- take the definitions verbatim so the battery is
citable rather than invented (PLpW 0.64 vs 0.59, p = .012; Pressure Passing Efficiency 4.56 vs 2.3,
p = .019; EPS and separateness showed no formation difference, which is itself useful: two of the five
metrics did not discriminate even in a designed experiment). It is also the strongest evidence that
nobody has done the robustness work: a professional rig had 12% missingness and the authors
interpolated and moved on.

### #7 Duarte, Araujo, Correia, Davids, Marques & Richardson (2013), team-team synchrony

*"Competing together: Assessing the dynamics of team-team and player-team synchrony in professional
association football", Human Movement Science 32(4):555-66, 2013.*
<https://pubmed.ncbi.nlm.nih.gov/24054894/>

Cluster-phase analysis (coupled-oscillator physics) applied to both teams' centroid trajectories
simultaneously, quantifying how synchronised the two collectives are longitudinally and laterally, and
how individual players couple to their own team's centroid.

**Data requirement:** full-match (x, y) for all 22 outfielders -- **of one single match**. The method
was validated on one game.

**How it applies here:** this is the most literal published answer to "how do two teams' structures
interact in a match", it is explicitly a single-match method (so the n=12 closure never touches it),
and it is a few dozen lines of numpy (Hilbert transform plus circular statistics) on top of the
existing 105x68 output. Reported values give something to sanity-check against: longitudinal synchrony
0.89 +/- 0.12 vs lateral 0.73 +/- 0.16 (p<.01); cross-team coupling Cross-SampEn 0.02 +/- 0.01;
possession did **not** significantly change synchronisation. One extra reason it is well-suited to
broadcast: the camera bias pulls *both* centroids the same way, so it partially cancels in the
inter-team quantity -- a testable claim, and worth testing in week 4.

### #8 Muller-Budack, Theiner, Rein & Ewerth (2019), "Does 4-4-2 exist?"

*"'Does 4-4-2 exist?' -- An Analytics Approach to Understand and Classify Football Team Formations in
Single Match Situations", MMSports workshop, ACM Multimedia 2019.* <https://arxiv.org/abs/1910.00412>

Formation classification deliberately scoped to **single match situations** rather than season
averages, plus a visualisation scheme comparing formations across possession states and match segments
*within one game*, with an expert annotation study on the task's difficulty.

**Data requirement:** positional data for one match; no multi-match corpus, by design.

**How it applies here:** it is the published defence of the unit of analysis this proposal is built on.
Its central finding -- nominal formation labels poorly describe actual in-play shape variability -- is
the argument for reporting a shape distribution per phase instead of one label per match, from a paper
rather than from Sid's own convenience.

### Supporting shelf (read on demand, not before week 4)

- **Corsie & Swinton (2023)**, "Reliability of spatial-temporal metrics used to assess collective
  behaviours in football: an in-silico experiment", Science and Medicine in Football 7(3):297-305 --
  <https://www.tandfonline.com/doi/full/10.1080/24733938.2022.2100460>. Injects 0.5 / 2 / 4 m
  positional noise into real tracking. Metrics stay reliable at <=0.5 m error; at **>=2 m several
  become unreliable**, especially over short (<20 s) windows. This is the error budget for the
  homography fix in section 6 -- it converts "the homography is slightly off" into a pass/fail number.
- **Penn, Donnelly & Bhatt (2025)**, "Continuous football player tracking from discrete broadcast
  data", Royal Society Open Science 12(10) / arXiv:2311.14642 --
  <https://royalsocietypublishing.org/rsos/article/12/10/251175/236076/>. A non-neural (ARMAX +
  Hungarian) imputer under a broadcast visibility model of ~10.60 of 20 outfielders visible; overall
  error 3.45 m, off-camera 7.33 m mean / 5.37 m median, <10 m mean after 20 s off-screen. Free
  competing baseline for the imputation arm, and its "team-level structure remained qualitatively
  identifiable as individual error grew" is exactly the hypothesis under test here.
- **Bauer, Anzer & Shaw (2023)**, "Putting team formations in association football into context",
  Journal of Sports Analytics 9(1):39-59 -- <https://doi.org/10.3233/JSA-220620>. Formation measured
  per phase of play rather than per match. The usable number: **>=60 seconds of aggregated data per
  phase** for a stable formation estimate. The existing Viterbi possession state substitutes for their
  CNN phase classifier.
- **Rico-Gonzalez et al. (2021/22)**, "Reference values for collective tactical behaviours based on
  positional data in professional football matches", Biology of Sport --
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8805357/>. External sanity ranges: team length 31-46 m,
  width 35-48 m, stretch index 7-16 m, surface area 750-1831 m2, all higher in possession than out.
  Same role the FIFA PMSR PDFs played for the physical metrics.
- **Brandes, Sotudeh, Parlak, Laffranchi & Erkul (2025)**, "Shape Graphs and the Instantaneous
  Inference of Tactical Positions in Soccer", npj Complexity 2:25 --
  <https://www.nature.com/articles/s44260-025-00047-x>. Per-frame Delaunay shape graph, keeping only
  angularly stable edges (threshold pi/4); reproduces provider position labels in "over two-thirds" of
  ~1M frames across 7 matches. The stable-edge idea is a natural defence against homography jitter, and
  Delaunay is well-defined for any point set >= 3, so it degrades gracefully with missing players --
  but that is an untested hypothesis in the paper, so treat it as a week-14 stretch arm, not a plan.

---

## 3. Formation and shape from partial observation

This is the technical core. It has three parts: **the estimator** (mostly already written), **the bias**
(named precisely, and specific to this codebase), and **the correction** (two arms, one of which may
turn out to be unnecessary -- which is itself a result).

### 3.1 The estimator: role-occupancy template assignment

Already implemented at `fingerprint/roles.py`. For a team over a window:

1. Take every track present for at least `MIN_PRESENCE_FRAMES` (currently 30).
2. Convert to **attacking coordinates** in absolute metres (x from own goal, y mirrored so "left" is
   attack-relative) -- note this uses the homography's absolute frame and is **not** centroid-normalised,
   which already sidesteps the worst failure mode of naive partial-data formation detection.
3. Force the strongest keeper candidate onto the GK slot; Hungarian-assign the remaining tracks to
   template slots (`scipy.optimize.linear_sum_assignment`, rectangular cost matrix, so N < 11 is already
   tolerated); score the template by mean matched distance; take the lowest-cost template.

Changes needed: widen `FORMATIONS` from 4 templates to EFPI's bank (or install `unravelsports` and call
it -- same algorithm, 65 templates, already maintained); window the assignment by phase instead of by
chunk; and replace the hard label with a distribution.

### 3.2 The bias, named twice, because there are two of them

**Bias 1 -- ball-conditioned missingness (MNAR occupancy shrinkage).** The camera follows the ball, so
*which* players are observed depends on where they are relative to the ball -- which is the very
quantity being estimated. This is missing-not-at-random in the textbook sense. Its concrete effect on
the estimator above: a track's mean position is an average over exactly the frames in which that track
was in shot, i.e. frames where the ball was near it. Every player's estimated mean position is
therefore pulled toward the ball's own occupancy centre, the team's estimated shape **compresses**, and
the template comparison is biased toward whichever template is most compact. Frame averaging does not
fix it -- the bias points the same way in every frame. Independent confirmation of the magnitude in a
neighbouring metric: the training-free imputation paper (arXiv:2607.11548, PDF already in the repo
root) measures **25-27 percentage points** of pitch-control error in hidden zones under exactly this
condition. And the effect is comparable in size to the signal: inter-role spacing is roughly 8-12 m
inside a 35-45 m block.

**Bias 2 -- mean-cost bias under variable N.** Specific to the code as written. `_match_formation`
scores a template by `matched.mean()`, the mean assignment distance over however many candidates
exist. With six visible players the mean cost falls for *every* template, but it falls fastest for
templates whose slots happen to lie where the visible players are -- near the ball. So the reported
formation drifts with the number of visible players, silently, and the drift correlates with match
phase. This is measurable on masked full-truth data in week 2 and it is a good week-2 result whichever
way it comes out.

### 3.3 The correction, as a three-arm ablation

Validated against **full-truth tracking censored by the measured camera footprint**, never against a
provider's formation label (30% cross-provider agreement makes that a meaningless target). Ground truth
is *the same estimator run on the unmasked data* -- so the question is "does occlusion change the
answer", which is well-posed and provider-independent.

**Arm A -- inverse-probability weighting (Horvitz-Thompson).** Weight each observation of player i at
frame t by `1 / pi_hat(visible | position, ball position)`, with `pi_hat` taken from the camera
footprint already measured from SkillCorner open data (**98.4%** of detected players fall inside the
reported polygon, `results/B4_TRANSFER_M3.md` §0). Players who are rarely visible where they actually
spend time get up-weighted when they *are* seen. No model, no training, ~40 lines on top of
`synthesizer.imputation.visibility_mask`. This arm directly targets bias 1 and it should be run first,
because if it works the imputation layer is not needed for this purpose and the thesis gets simpler.

**Arm B -- imputation completion with multiple imputation.** Complete the 10-outfield set with the
frozen B4 v1 quantile model (`tools/imputation_b4_external.fit_frozen_v1`), drawing **K ~ 20 samples**
from its predictive distribution, running the assignment once per draw, and reporting the
**distribution over formation labels** ("4-2-3-1 out of possession in 83% of draws") plus role-assignment
agreement across draws. This also fixes bias 2 by construction: every template is scored on the same
10 slots every time.

**Arm C -- the strawman.** A single fitted scalar correction (one expansion factor applied to the
observed shape), fitted on masked truth. It goes in *first* on the plot, per `docs/PROJECT_OPTIONS_FINAL.md`:
if a scalar beats both real arms, the thesis needs to know in week 3, not in November.

### 3.4 Is the imputation layer actually needed? Honest answer: for half the battery

State the hypothesis before measuring it, then let week 3 decide:

- **For time-averaged, per-player quantities** -- role occupancy, mean position, template assignment
  over a >=60 s window -- IPW should be **sufficient**, because each player accumulates *some* direct
  observations within any window (the camera pans across the whole team over two minutes) and
  re-weighting marginal observations is a legitimate estimator of a marginal quantity. Prediction: arm A
  closes most of the gap to full-truth here.
- **For simultaneity-dependent functionals** -- convex-hull area, stretch index, team length/width,
  pitch control, cluster-phase synchrony, pressing intensity -- IPW **cannot** work, because the metric
  is a nonlinear function of the configuration of all ten players *at one instant*, and no re-weighting
  of marginal observations reconstructs a joint configuration. These need arm B.

That split is a clean, falsifiable claim, it explains exactly where the existing calibrated imputer
earns its place, and either outcome is publishable in the thesis. The caveat that must be printed next
to every arm-B number: the imputer's 90% regions are ~8-11 m, comparable to inter-role spacing, so a
single imputed frame is nearly uninformative about that player's role. Confidence comes from
aggregation across the window and across draws, never from a frame -- and windows whose direct-observation
coverage is too low get **refused**, not labelled. `tools/tactical_clip.py` already implements exactly
this refusal behaviour for rendering; reuse the pattern.

### 3.5 Uncertainty, done properly

Two nested sources: imputation draws (within-window) and window-to-window autocorrelation (off-screen
spells last tens of seconds and imputer errors are autocorrelated). Per-frame counts would fake
precision by an order of magnitude. Use the K-draw spread for within-window uncertainty and a **block
bootstrap over windows** for the match-level interval. This is the same discipline
`synthesizer/imputation_v1.py::paired_rmse_ci` already applies.

---

## 4. The commentary channel

**Verdict: viable, worth one boxed week, but re-scoped -- and it is not on the critical path.**

The idea is sound and was probed on the real footage this session, not merely surveyed: the pipeline
chunks carry AAC 48 kHz audio; `faster-whisper` small, CPU int8, no denoising, transcribed two 3-minute
slices at ~6x realtime with usable output; roster-name density measured at **7.7 hits/min** in a
live play-by-play stretch (manutd_liverpool) against **1.7/min** in a pundit-chatter stretch
(brighton_manutd), versus ~10 pass events/min; and live names land within **0-2 s** of the touch (the
5-15 s lag MatchTime documents applies to website text timestamps, not live audio). Name mangling is
tractable because the vocabulary is two known lineups, ~50 surnames -- "Masawawi" -> Mazraoui,
"Dallow" -> Dalot, "Casamiro" -> Casemiro are exactly what roster-constrained fuzzy matching fixes
(Whisper-Courtside, arXiv:2602.18966, 21.7% -> 18.0% WER by biasing a second pass with a roster).

What it is **not**: a rescue of per-event attribution. A name labels "a touch near time t", not a
passer versus a receiver ("Alexander-Arnold... does find Salah"), and the purely visual state of the art
with a face database Sid does not have tops out at **71.1%** (GameSight, arXiv:2604.00057, versus 96.3%
human). Do not reopen the 0.36-0.40 closure through this door.

**What it is worth, inside a geometry project** -- two uses, both cheap, both aligned with section 3:

1. **Names attached to role slots, not to events.** A role slot persists for minutes. Associating a
   commentary name with the track occupying a role over a 2-minute window is a many-to-one aggregation
   where individual errors average out, rather than a per-event decision where they do not. This makes
   the tactics board legible ("the left centre-back is Martinez") without a single per-event claim.
2. **Free change-point ground truth.** Substitutions and announced tactical switches are stated on air
   with a time. A formation change-point detector (section 3, SoccerCPD framing) needs labelled moments
   where the shape genuinely changed, and there is no other free source of them for this corpus. This
   is the higher-value use and nobody in the surveyed literature does it.

### The first cheap experiment (1-2 days), with go/no-go numbers

Run against the **existing** 90-label harness so the number is directly comparable to the closed one:

1. `ffmpeg`-extract audio for the chunks behind `results/carrier_attr/labelpack/labels_filled.csv`,
   plus two full halves.
2. Whisper-small, `word_timestamps=True`, `vad_filter=True`, `initial_prompt` seeded with both lineups.
3. `rapidfuzz` match against both 25-man squads, `score_cutoff ~80`.
4. Classify utterances play-by-play vs chatter (segment length <= 4 words, or bare-name heuristic).
5. Snap each play-by-play name to the Viterbi carrier interval overlapping `[t-3s, t+1s]`, sweeping the
   lag offset; score with `tools/score_carrier_labels.py`.

**GO if all three hold:** name-to-true-carrier precision **>= 0.60** at the best fixed lag; usable
synchronous names **>= 1.5/min averaged over a full half** (not a busy stretch); lag jitter **IQR <= 4 s**.
**NO-GO on any one**, and the channel is dropped in a day with a measured paragraph for the thesis --
which is a perfectly good outcome, because "commentary does not rescue attribution either, and here is
the density and jitter that prove it" closes the last open door on the identity question honestly.

If it goes, spend week 10 on use (2) only -- substitution and tactical-switch timestamps as change-point
labels -- and leave use (1) as a rendering nicety. Do **not** let it grow into an attribution project.

---

## 5. Team-level metrics that survive

The battery, with the honest observability caveat and the sample size each claim needs. "Matches
needed" is for the *claim*, not the computation -- every one of these computes on a single match.

| # | metric | definition source | needs | observability caveat | matches for a per-match description | matches for a cross-match claim |
|---|---|---|---|---|---|---|
| 1 | Defensive line height (per phase) | `fingerprint/block_height.py` (shipped) | rearmost outfielder, out-of-possession frames | broadcast frames the block deeper when the ball is forward; ~7 m held-out scale uncertainty, Gate 1 hand-annotation pending. **Ordering usable, metre label not certified** until IPW + gate | **1** | paired within-match or within-pair only |
| 2 | Vertical compactness / inter-line distance | shipped, same module | full defensive unit | measured at 6.17 +/- 0.39 m for United and 5.93 +/- 0.29 m for six opponents -- a constant across seven teams is a property of the *measurement*, not of a team. Must be re-derived under IPW; if masked-Metrica shows occlusion manufactures the constant, that is a finding | **1** | paired only; never as identity |
| 3 | Team length, width, length-per-width (PLpW) | Memmert 2019 | all 10 outfield simultaneously | simultaneity-dependent -> **needs arm B** (imputation). Reference ranges: length 31-46 m, width 35-48 m | **1** (with K-draw interval) | 12-18 for a paired in/out-possession contrast |
| 4 | Effective Playing Space / convex-hull area | Memmert 2019; `structural_metrics.team_shape` (shipped) | all 10 simultaneously | **explicitly not N-invariant** (Rumpf review: area and stretch move with player count) -- a hull over 8 visible players reads differently from one over 10 regardless of true shape. Report only with the masked-truth bias curve attached | **1** | 12-18 paired |
| 5 | Stretch index (mean centroid-to-player distance) | Rico-Gonzalez ranges 7-16 m | centroid + all players | centroid is the single most MNAR-exposed quantity in the battery; IPW-correct it before anything downstream uses it | **1** | 12-18 paired |
| 6 | Formation label per phase, as a distribution | EFPI / `fingerprint/roles.py` | >= 60 s per phase (Bauer) | ceiling is 72.4% agreement *with full tracking*; report K-draw label shares, refuse windows below a coverage threshold | **1** | not a cross-match claim at all -- 16-18 descriptions |
| 7 | Formation change-points within a match | SoccerCPD framing | label sequence + stability | needs >= 1 well-observed frame per half to seed roles; commentary substitution times as free labels (section 4) | **1** | n/a |
| 8 | Inter-team centroid distance and cluster-phase synchrony (longitudinal + lateral) | Duarte 2013 | both teams' centroids | **camera bias is common-mode across both centroids and partially cancels in the difference** -- a testable claim, test it in week 4. Validated on one match in the source paper | **1** | 12-18 paired |
| 9 | Pressing intensity surface (time-to-intercept -> probability) | Bekkers, arXiv:2501.04712 | positions + velocities, no events | velocity is the jitter-sensitive quantity; Corsie & Swinton says >= 2 m positional error breaks short-window metrics -> **gated on the section 6 homography fix**. Zero dependence on event attribution, which is the point | **1** | 12-18 paired |
| 10 | Pitch control / space occupation | `fingerprint/pitch_control.py` (shipped) | full sets | the one metric with a published degradation number: 25-27 pp error in hidden zones (arXiv:2607.11548); DeepMind measured the imputed-vs-truth version. Needs arm B and the degradation curve printed alongside | **1** | 12-18 paired |

**What is deliberately absent, and stays absent:** any per-player event count; any unpaired between-team
comparison; any team-identity claim; ball-trajectory style; entry-channel chains (43 in 12 matches).

**"Pressing" and "ball recovery" in the restored form.** Pressing survives as metric 9 -- a continuous
geometric surface per frame, opponent-conditioned by construction, needing no names and no event
attribution -- plus block height and compactness at the moment of loss. Ball recovery survives as
*where on the pitch possession changed hands and what both shapes looked like at that instant*, which
is geometry plus the already-validated Viterbi possession label, not an attributed event. What does not
survive is "United counterpress at rate R and Brighton at rate R'". That is the 58-90 match wall, it is
real, and it is the only thing being given up.

---

## 6. Pipeline quality fixes

Three, in ascending cost. The first is free and already written.

### 6.0 Promote the direction resolver that already works (half a day)

`fingerprint/roles.py` calls `structural_metrics.resolve_attack_directions` (keeper-based), which
`results/MANUTD_IDENTITY_PROFILE.md` measured as "correct but noisy per chunk" -- 3 of 6 chunks
backwards on `tottenham_manutd` h1. The profile then built and validated a better one: a chunk votes on
direction only when the two keepers separate cleanly (medians >= 40 m apart), votes are summed per half
weighted by keeper support, and the half-time end swap is enforced. Graded on an external position
oracle across 9 matches it gives **positive rank correlation in 8/9 matches (mean rho +0.610) versus
7/9 (+0.361)** for the shipped resolver. It lives in `tools/manutd_identity.py` and is not in the
library. Every formation number in section 3 depends on attacking direction being right, so this is the
cheapest quality win available -- move it into `fingerprint/structural_metrics.py` and make it the
default. (Also: `resolve_attack_directions_from_ball` is *systematically inverted* -- it flips the sign
of the rank correlation in 5/5 matches tested -- and is still the primary resolver inside
`fingerprint/block_height.py::_directions`. Fix or delete it.)

### 6.1 Homography drift: per-shot temporal smoothing of the camera parameters (1-2 days)

**Diagnosis.** The pipeline estimates the camera independently per frame. BroadTrack (Magera et al.,
WACV 2025, arXiv:2412.01721, <https://arxiv.org/abs/2412.01721>) documents this exact symptom on
SoccerNet broadcast: per-frame pan/focal estimates jitter frame to frame and the implied camera
position wanders **up to 20 m within a single 30 s shot**. Their numbers on sn-gamestate: per-frame
SOTA NBJW **10.28 px** mean reprojection error (JaC5 37.14), TVCalib 12.4 px, BroadTrack's temporal
tracking **5.02 px** (JaC5 56.88) -- error halved. Crucially, their ablation shows the *minimal*
temporal move -- warm-starting each frame's optimisation from the previous frame, no optical flow, no
tripod constraint -- already reaches **5.74 px**. Most of the gain comes from any temporal coupling at
all.

**The cheapest intervention, and it is pure Python.** `generator/calibrate.py:315` already computes
`h = ground_homography_from_cam_params(final["cam_params"])` and then **throws the camera parameters
away**. Keep them, and:

1. Emit `cam_params` (focal, rotation, `position_meters`) per frame from
   `PnLCalibCalibrator.calibrate_frame`.
2. Segment frames by the cut detector that `tools/tactical_clip.py` already implements.
3. Within a shot, fix `position_meters` to the shot median (the tripod constraint -- a broadcast main
   camera does not translate) and smooth pan/tilt/roll/focal with a rolling median or Savitzky-Golay
   over ~0.5-1 s, interpolating across gate-failed gaps shorter than 1 s only.
4. Rebuild H per frame with the existing `ground_homography_from_cam_params`.
5. Keep the 2 m keypoint gate as a post-check on the smoothed H, falling back to the per-frame H where
   the smoothed one fails it.

**Pre-register the two metrics before running.** JITTER = mean inter-frame displacement of four fixed
projected pitch points within a shot. ACCURACY = existing per-frame keypoint reprojection error in
metres, plus 5-10 hand-annotated frames as an independent check. **Success: jitter down > 50% with
reprojection error flat or better.** **Kill: jitter down but hand-annotated reprojection unchanged** --
that means the error is per-frame *bias* (a systematically misidentified line), not variance, and
smoothing merely makes the lines steadily wrong; the next step then is a better per-frame calibrator,
not more filtering. Also check gap statistics on one match first (half a day): if gate-pass frames
within shots are regularly more than 1-2 s apart, interpolation cannot bridge them and the fix is dead
before it is written.

Why this matters beyond tidiness: Corsie & Swinton put the reliability cliff at **>= 2 m positional
error**, especially for short windows. Metric 9 (pressing intensity, which needs velocities) sits on the
wrong side of that cliff if the jitter is real. Do this before week 11.

**Do not attempt to run BroadTrack itself.** It ran at 16 fps on *two RTX 4090s*, needs CUDA Docker
under WSL2, and 4 GB VRAM fit is unverified. Cite it, take its ablation, skip its code.

### 6.2 Ball tracking: measure the sensitivity before improving it (1 day to decide)

**First, the honest scoping.** The battery in section 5 needs the ball for exactly one thing -- the
possession/phase split -- and that needs only *which team has it*, which is the Viterbi label, already
validated indirectly by pass counts at 0.97-1.09x official on 10/12 matches and directly cross-checked
against the attributed-pass ledger to a mean absolute 5.3 pp on channel splits. Formation, shape,
coupling and pressing geometry need no ball at all. So before spending weeks on the ball: **run the
whole battery with the phase labels perturbed** (shift phase boundaries by +/- 1, 2, 5 s and re-run) and
see whether the descriptions move. If they do not, the ball is off the critical path and the remaining
32-52% coverage is a caveat, not a defect.

**If it does move, the cheapest intervention** is not a better detector. `generator/ball.py::link_ball`
is a *forward-only* greedy nearest-to-prediction linker with `MAX_INTERP_GAP = 8` samples, so it
systematically loses the frames immediately *before* each re-acquisition (nothing to predict from yet).
Run the same linker on the reversed frame sequence and union the two tracks, dropping frames where they
disagree beyond the speed gate. ~30 lines, no model, no GPU. Acceptance criterion, pre-registered:
post-`link_ball` usable coverage rises from the current per-match 0.360-0.561 **without** the possession
label degrading against the validated pass ledger. (Remember the standing rule: the only ball number
that counts is post-`link_ball` usable track.)

---

## 7. The proposed project

> **Broadcast football video shows about 11.6 of 22 players, and which 11.6 depends on where the ball
> is -- so every team-shape metric ever computed from broadcast has been estimated from a ball-biased
> sample of the team, and nobody has measured what that does. This project measures it and then
> corrects it. It assembles a battery of team-level geometric tactical metrics (formation by template
> assignment, in- and out-of-possession shape, length/width/area/stretch, inter-team centroid coupling,
> and a time-to-intercept pressing surface), calibrates each one's bias by censoring full-truth
> tracking with a broadcast camera footprint measured from real data, corrects that bias with two
> arms -- inverse-probability weighting against the measured footprint, and calibrated multiple
> imputation of the off-screen players -- and applies the corrected battery to 16-18 Manchester United
> broadcasts to produce per-match, per-phase descriptions of how two teams set up against each other,
> each carried with an interval and an explicit refusal where visibility is too low to answer. The
> contribution is not another formation classifier: it is the first measurement of how far broadcast
> occlusion bends team-shape metrics, the first estimator that corrects for it, and a tactical replay
> that shows its own uncertainty on screen.**

Why this is the right project rather than a consolation prize: every closed result feeds it (the
attribution failure is *why* the battery is name-free; the n=12 wall is *why* the unit is the match and
the window; the entropy failure is *why* the representation is player geometry rather than ball
trajectories); every surviving asset is load-bearing (the camera footprint becomes the censoring
operator, the calibrated imputer becomes one arm of an ablation instead of a scooped contribution, the
clip renderer becomes the instrument that displays the result); and the field-wide gap statement comes
from a survey rather than from us.

### Week by week to December (19 weeks, corpus 16-18 matches)

Hardware rule respected throughout: match ingest is the only GPU-heavy work and is scheduled one at a
time in the background; the analysis is CPU numpy/scipy and can run alongside.

| week | work | done when (the number) |
|---|---|---|
| **1** | Masking harness on full truth. Reuse `synthesizer.imputation.load_metrica_match` / `visibility_mask` / `smooth_camera`; apply the SkillCorner-measured footprint to Metrica Games 1-2; run `fingerprint/roles.py` unchanged on masked and unmasked, per 2-minute in-play window. Promote the validated direction resolver (6.0) on the way past. | paired (masked, truth) formation labels exist for every window in 2 games; direction resolver merged and its 8/9 rho reproduced |
| **2** | **Measure both biases.** Formation-label disagreement vs visible-player count; per-player mean-position shift vs ball distance (bias 1); mean-cost drift vs N (bias 2). Widen the template bank to EFPI's (or install `unravelsports`). | two bias curves with magnitudes in metres and in label-disagreement rate; **decision gate: if masked-vs-truth agreement is already > 85% the correction work shrinks and weeks 3-4 move to the battery** |
| **3** | Three-arm correction: C (scalar strawman) first, then A (IPW), then B (K=20 multiple imputation). Score against unmasked-truth labels, plus calibration of the draw distribution (when draws disagree, is the truth label in the disputed set?). | three agreement curves vs visibility; an explicit verdict on the 3.4 hypothesis (IPW enough for aggregates, imputation needed for simultaneity) |
| **4** | Extend to the simultaneity metrics: length/width/PLpW, hull area, stretch index, inter-team centroid distance, cluster-phase synchrony. Same three arms. Test the common-mode cancellation claim for inter-team quantities. | one table row per metric: truth value, censored bias, censored RMSE, corrected RMSE per arm; the cancellation claim confirmed or refuted with a number |
| **5** | Phase conditioning: in-possession / out-of-possession / transition windows from the existing Viterbi state, Bauer's >= 60 s minimum enforced; re-run weeks 3-4 per phase. Define the refusal rule (coverage threshold below which a window is not labelled). | per-phase versions of the week-4 table; refusal threshold derived from data, not chosen |
| **6** | **Homography fix (6.1)** with its pre-registered JITTER / ACCURACY metrics, and the gap-statistics check first. Re-run week 4 on the smoothed geometry to quantify what it bought. | jitter down > 50% with reprojection flat or better -- or a documented kill, in which case the calibrator, not the filter, is the problem |
| **7** | Port to the existing 12 ManU matches. **Coverage census**: how much of each match yields 2-minute windows above the refusal threshold, and does each half contain at least one near-complete frame (SoccerCPD's seeding requirement)? | per-match usable-window counts. **Go/no-go: if a majority of match time has no usable window, the method survives on paper and dies on this corpus** -- fall back to the coarse already-validated geometry (line height, block width, compactness) phrased role-free |
| **8-9** | Ingest the 4-6 additional matches (GPU, one at a time, in the background). Selection criteria, in priority order: (a) completes a home/away pair, taking paired designs from 6 to 8-9; (b) an opponent that presses high -- **all 12 current legs have the opposition block between 19.9 and 29.7 m, so the corpus contains no high-press example at all**; (c) any match with a red card, which is a genuine reduced-N formation and a near-free check on the variable-N estimator. Foreground work: run the corrected battery on the 12 existing matches. | 16-18 matches in `data/matches.yaml`, all passing the existing pipeline gates; battery outputs for the original 12 |
| **10** | **Boxed commentary week (section 4).** Run the 1-2 day experiment, apply the go/no-go, and either take substitution/tactical-switch timestamps as change-point labels or write the negative paragraph and stop. Nothing here is allowed to slip into week 11. | one number vs the 0.36-0.40 closure, and a GO/NO-GO decision on the record |
| **11** | Cross-team interaction on the full corpus: inter-centroid distance and cluster-phase synchrony per match; pressing-intensity surface (metric 9) now that the geometry is smoothed. | per-match coupling series for 16-18 matches, with intervals |
| **12** | Paired cross-match analysis -- the only cross-match inference in the project: in- vs out-of-possession shape (sign test over matches), and home vs away legs against the same opponent (sign test over 8-9 pairs). Pre-register the direction of each test before looking. | two sign tests with pre-registered directions and honest p-values, or an honest "underpowered, declined" |
| **13** | Formation change-point detection per match (SoccerCPD framing on the label-distribution sequence), validated against the week-10 substitution labels if commentary passed, and against visible personnel changes if it did not. | change-point series per match with a stated validation source |
| **14** | The demo: opposition scouting pack for one fixture pair, rendered through `tools/tactical_clip.py` with the shape overlay, the K-draw uncertainty and the refusal state visible on screen. | a rendered pack an examiner can watch, including at least one refused passage |
| **15** | Consolidation: the bias table, the correction curves, the per-match description set, the reference-range cross-check against Rico-Gonzalez (length 31-46 m, width 35-48 m, stretch 7-16 m, area 750-1831 m2). | every headline figure regenerable by one command; anything outside the reference ranges explained or flagged |
| **16-18** | Writing. Chapter 2 is the reading list in section 2 (the 30% cross-provider number and the "no method handles reduced player counts" statement carry the motivation); chapter 3 is the pipeline as apparatus; chapters 4-5 are the bias measurement and the correction; chapter 6 is the applied descriptions and the demo. | full draft with every number traceable to a results file |
| **19** | Slack. Deliberately empty. | -- |

**Decision gates, restated so they cannot be quietly skipped:** end of week 2 (is there a bias worth
correcting?), end of week 3 (which arm wins, and is imputation needed at all?), end of week 6 (did the
homography fix work, or is the error bias not variance?), end of week 7 (does this corpus have enough
usable windows?), end of week 10 (commentary in or out?). Four of the five have a fallback that still
produces a thesis; week 7 is the one that would force the fallback to coarse role-free geometry, which
is why it is deliberately early.

---

## What this proposal deliberately does not promise

Said once, clearly, so it does not have to be re-litigated: **per-player event statistics are not
coming back.** Not from better embeddings (0.36-0.40 end to end), not from elimination constraints
(+0.04), not from jersey OCR (0.000, and it fails silently), and not from commentary (which at best
names a touch, not a passer, and whose fully-supervised published ceiling with a face database is
71.1%). And **team-vs-team rate comparisons are not coming back at this corpus size** -- 58-90 matches
is a variance floor, not an engineering target.

Everything else the original goal asked for -- how a team sets up, how high it defends, how compact it
stays, how it presses, how its shape answers the opponent's -- is measurable from geometry, needs no
names, computes on a single match, and is currently unvalidated under broadcast occlusion by anyone in
the field. That is the project.
