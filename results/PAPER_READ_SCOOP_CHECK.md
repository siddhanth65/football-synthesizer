# Paper read + scoop check: does RQ-D survive?

Harsh full-text read of two papers, done 2026-07-26. Read-only pass; no code touched.

Papers:

1. **Choi, S. (2026).** *Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial
   Football Analytics.* arXiv:2607.11548v1, 13 Jul 2026. Single author, "Independent researcher".
   Preprint, not peer-reviewed. Code+data: github.com/nowayfootball/offscreen-impute.
   10 pages incl. references.
2. **Theiner, J., Gritz, W., Muller-Budack, E., Rein, R., Memmert, D., Ewerth, R. (2021).**
   *Extraction of Positional Player Data from Broadcast Soccer Videos.* arXiv:2110.11107v1,
   21 Oct 2021. L3S Research Center (Leibniz Universitat Hannover), TIB Hannover, Institute of
   Exercise and Sport Informatics (German Sport University Cologne). This is the WACV 2022 paper.
   13 pages incl. appendix and references. Funded by BMBF 01IS20021A/B.

---

## 0. VERDICT ON RQ-D (up front)

**RQ-D SURVIVES, NARROWED. Neither paper does the resolution/discriminability analysis. Not one
between-team variance, not one signal-to-noise ratio, not one abstain threshold appears in either
paper.** But the narrowing is real and non-negotiable on three axes, and one clause of RQ-D has to
be demoted from contribution to instrument.

**What is dead / must be dropped:**

- **Imputation-as-contribution is dead.** Choi's B4 "role-anchored centroid voting" (Sec 4, Eq. 1,
  p.4) is the same method family as the B6/B7 anchor Sid's v1 sits on: each visible player votes
  for the team centroid by subtracting its running role offset, offsets updated by EMA against the
  voted centroid. Published, benchmarked on the same Metrica open data, with open code. Sid cannot
  present "an off-screen imputation model" as the novel object. He can present it as an instrument
  he validated.
- **D computed on pitch control / control share is dead on arrival.** Choi Table 1 (p.5) and
  Table 2 (p.6) already give the censoring-induced error for exactly those two quantities across
  three matches and four viewport widths (B0 ignore: hidden-zone control MAE 25.1-26.9 pp, share
  error 11.1-13.4 pp at W=44 m). A reviewer will say the denominator of D is already in his table.
  If Sid runs D on control share, he has written a ratio wrapper around a published number.

**What is unclaimed and load-bearing (this is the surviving RQ-D):**

1. **The metric family: pressing and transition.** Choi measures pitch control, team control share,
   and one possession-quality score (SCI). Zero pressing metrics. Zero transition metrics. No PPDA,
   no counterpress recovery, no defensive-line height, no block compactness (he *names* block
   compactness in the intro, p.1, and never measures it), no rest-defence. Theiner measures no
   tactical metric at all. This whole family is open, and it is the family where Sid's own
   feasibility census says geometry binds (84% of usable turnover windows lost to visibility vs 65%
   to ball tracking).
2. **Calibrated predictive uncertainty and the abstain rule derived from it.** Choi reports point
   estimates only. His only uncertainty is a one-minute **block-bootstrap 95% confidence interval on
   the aggregate error statistic** (Sec 3, p.4) - sampling uncertainty on a mean, not a predictive
   region on a player. There is no coverage check, no calibration table, no per-sample interval, no
   abstention anywhere in the paper. Theiner has no uncertainty either (his rho/q filters are
   frame-discard heuristics, Appendix D-E, p.10). Sid's 50/90% regions passing 6/6 coverage are
   genuinely unclaimed territory, and they are the only mechanism that can *produce* a D=1 abstain
   threshold rather than assert one.
3. **Measured broadcast visibility, not a simulated rectangle.** Choi's viewport is a horizontally
   panning band, full pitch height, ball-EMA lag, width W (Sec 3, p.3). He names its failure himself
   in Limitations (Sec 7, p.9): *"The simulated viewport pans but does not zoom or tilt; zoom changes
   the number of visible players, so real-broadcast occlusion statistics (including the 50-57%
   long-occlusion share) may differ from the simulated ones."* Sid's SkillCorner-measured trapezoid
   footprint (11.0 m half-width near edge, 24.1 m far edge, 98.4% of detected players inside,
   detection crossing 0->94% within ~2 m of the edge) is the direct empirical answer to a limitation
   the incumbent author wrote down. Keep it, promote it, do not bury it in a methods appendix.

**One-line statement of the surviving question:** *Under a measured (not simulated) broadcast
footprint, which pressing and transition metrics retain between-team discriminability, and does
calibration-derived abstention - not raw imputation accuracy - move that threshold?*

That is defensible. The word doing the work is **calibrated**, and the metric family doing the work
is **pressing/transition**. Strip either and the RQ collapses into Choi's Table 1.

---

## 1. Choi 2026: what it actually does and claims

### Dataset

Metrica Sports open tracking (ref [8]), **three matches**, all 22 players + ball, 25 fps. Games 1-2
ship as full-match CSVs; game 3 is EPTS-FIFA format, one half only. He uses **the first 45 minutes of
each**, evaluated at 5 fps. Games 1-2 for method and hyperparameter development; **game 3 plus the
B3E/B3V/B5 ablation variants were specified and frozen before game 3 was first evaluated** (Sec 3,
p.3). That is a clean held-out-match protocol. Note: Sid's protocol holds out *Game 2 H2* after
calibrating on *Game 2 H1* - same match, same players. On this one axis Choi's protocol is stronger
and Sid should say so before a supervisor does.

### Censoring model

Virtual main camera, **pans horizontally only**, following exponentially smoothed ball position
(alpha = 0.06 per frame at 25 fps) to mimic pan lag. Visible region = width-W window spanning the
**full pitch height**. At W = 44 m the viewport shows 14.6-15.0 of 22 players (Sec 3, p.3), which he
justifies against 10-16 in his own World Cup clips and 12.8 +/- 3.7 in Omidshafiei et al. Sensitivity
swept over W in {36, 44, 52, 60} m.

**No zoom, no tilt, no perspective, rectangular.** Flagged by him as a limitation (Sec 7, p.9).

### Method

A ladder of **training-free, causal (no future observations), online, closed-form** heuristics using
only observations from the current match, running faster than real time on one CPU core (Sec 4, p.4):

- B0 ignore (hidden players absent from the metric - the visible-only baseline)
- B1 last-seen with decay (tau = 8 s toward visible-team mean)
- B2 formation anchor (stored offset from visible-team centroid)
- B5 fixed formation template (cumulative-mean offset)
- B3 = B2 + EMA offsets + constant-velocity extrapolation (blend weight exp(-dt/1.5 s)); B3E and B3V
  isolate the two ingredients
- **B4 role-anchored centroid voting** - the headline. Each visible player votes for the full-team
  centroid as its position minus its role offset; offsets are then EMA-updated against the voted
  centroid (EMA weight 0.1 per 5 fps step). Self-consistent bootstrap, seeded by the visible mean
  until three offset-bearing voters exist. Falls back to B2 (14-30% of estimates at W=44 m) and B1.

All hyperparameters (tau = 8 s, EMA 0.1, three-voter threshold, 1.5 s blend) are "round values fixed
once during development on games 1-2 and **not optimised by search**" (Sec 7, p.9).

*Relevant finding from Sid's own work:* his sweep of the vote EMA half-life found monotone
improvement over three orders of magnitude down to 0.04 s (one frame), i.e. **the optimum of Choi's
"EMA role offset" is no EMA at all** on this data. That is a substantive, citable finding about a
published method and it is currently sitting in `results/B4_MODEL_V1.md` as an internal note. Use it.

### Metrics reported

Three, "chosen to reflect the downstream analytics rather than trajectory fidelity alone" (Sec 3,
p.3):

1. **Hidden-player position error (m)** - over all (frame, player) pairs where the player is outside
   the viewport *and has been observed at least once earlier in the half*. Never-observed players are
   excluded from position scoring.
2. **Pitch-control map MAE (pp)** - 3 m grid, arrival-time + sigmoid contest model (Spearman [14]),
   scored full-pitch and hidden-zone separately. **Control computed with zero velocities in all
   conditions** - explicitly, to isolate imputation from velocity estimation. Velocity-aware control
   is untested and left open (Sec 7, p.8-9).
3. **Team control-share error (pp)** - mean absolute per-frame deviation of "team A controls x% of
   the pitch".

He designates (1) and (3) as *decision-relevant* and selects methods on them; (2) is "for
completeness" (Sec 3, p.4).

### The error numbers, and MEAN vs MEDIAN

**This matters and is easy to get wrong.** Table 1 (p.5), W = 44 m, games g1/g2/g3:

| method | hidden ctrl MAE (pp) | share err (pp) | **median** pos err (m) |
|---|---|---|---|
| B0 ignore | 26.9 / 25.6 / 25.1 | 13.4 / 12.5 / 11.1 | - (places no players) |
| B1 last-seen | 22.1 / 20.0 / 19.5 | 10.6 / 9.5 / 8.2 | 19.6 / 17.9 / 18.4 |
| B2 anchor | 15.7 / 14.6 / 14.3 | 6.2 / 5.4 / 4.4 | 13.6 / 12.8 / 12.5 |
| B5 fixed template | 23.0 / 18.8 / 19.1 | 10.3 / 7.7 / 6.9 | 22.7 / 20.7 / 21.9 |
| B3 EMA+velocity | **12.8 / 11.8 / 13.2** | 5.7 / 4.6 / 4.7 | 14.6 / 14.0 / 15.1 |
| **B4 centroid vote** | 13.3 / 12.2 / 13.8 | **4.7 / 4.5** / 4.7 | **11.6 / 10.0 / 9.7** |

- The **position column is MEDIAN** (column header: "median pos. err. (m)"). The map and share
  columns are **MEANs** (MAE, mean absolute per-frame deviation). So the headline "9.7-11.6 m" is a
  median over a right-skewed distribution. A mean or RMSE on the same data would be materially
  larger.
- **Sid's 8.10 m is an RMSE.** RMSE >= mean >= median for a right-skewed error distribution, so 8.10
  RMSE against 9.7-11.6 median is a *bigger* gap than the raw digits suggest - but it is measured on
  a different split (Game 2 H2 holdout vs Choi's Game 3), so **it is not a like-for-like comparison
  and must not be presented as one.** State the metric type every time.
- Choi's B2 anchor (12.5-13.6 m median) is roughly the strength of Sid's B7 causal bar (11.46 m
  RMSE). The baseline is in the literature too.

### Occlusion-duration stratification (Sec 5.1, p.7) - the most useful table in the paper

| occlusion gap | <=2 s | 2-9.6 s | >9.6 s |
|---|---|---|---|
| B4 median position error | 3.3-3.7 m | 7.2-8.9 m | **15.6-16.9 m** |
| share of hidden samples (frame-weighted) | <=9.6 s: 43-50% | | **>9.6 s: 50-57%** |

He states plainly that **half or more of hidden observations lie beyond the 9.6 s window** that the
closest prior work (DeepMind Graph Imputer, Omidshafiei et al. 2022, ref [10]) can even operate in,
that long far-side-defender occlusions during sustained attacking phases are "precisely where spatial
metrics need imputation most", and that his 15.6-16.9 m there "show the remaining headroom, which we
expect learned long-horizon models to attack."

**That sentence is an open invitation, written by the incumbent.** See section 5 below.

### Downstream analytics evaluated

- Pitch control map and team control share (on simulated censoring, with full-pitch ground truth).
- **Space-Creation Index (SCI)** on **two** junk-possession windows from one real broadcast
  (FIFA WC 2026 R32, Netherlands-Morocco), inside an end-to-end video->GSR->metrics pipeline
  (PnLCalib calibration, BoT-SORT, colour team assignment, ghosts with 20 s lifetime).
  Window 1: SCI +15.6 -> +32.8. Window 2: SCI -10.9 -> +4.7, flipping the verdict class from "dead
  junk" to "weak progression" (Sec 6, p.8).

### What he explicitly says he does NOT / cannot claim

Unusually well-disciplined on this front. Verbatim or near:

- **"Absent full-pitch ground truth for broadcast footage we make no claim that the imputed scores
  are more accurate; the point is a sensitivity result."** (Sec 6, p.8). His real-broadcast
  contribution is N=2 windows with **no ground truth at all**.
- **"we do not claim superiority over learned bidirectional models within their own evaluation
  regime; the point is that a zero-training method is usable there at all."** (Sec 5.1, p.7).
- "learned imputers remain preferable when training data and future context are available - our
  contribution is the training-free floor, the harsher online/long-occlusion benchmark, and the
  downstream-metric framing." (Sec 7, p.9).
- Protocol differences with Graph Imputer "preclude direct numeric comparison" (Sec 2, p.2;
  Table 3, p.7).
- GS-HOTA 35.2 is on 11 downloadable SoccerNet-GSR sequences and is "for context only", explicitly
  not comparable to the official 22.26 on the full test set (Sec 6, p.7-8).
- Velocity-aware control untested; jersey-number OCR ~45% so ghosts anchor to roles not verified
  identities; three matches from a single provider; viewport pans but does not zoom or tilt
  (Sec 7, p.8-9).

### Uncertainty / calibration: NONE

The only uncertainty quantity in the entire paper is a **one-minute block-bootstrap 95% confidence
interval** on the hidden-zone MAE and the share error (Sec 3, p.4), used to say e.g. that the
game-3 B4-B2 share-error difference [-0.4, +1.3] pp includes zero and "this gap is not resolved at
this sample size" (Sec 5, p.5). That is a CI on an aggregate error statistic across temporal blocks.

There is **no predictive distribution, no per-sample interval, no coverage test, no calibration
table, no conformal anything, and no abstention** in the paper. B4 emits a point per hidden player.

### Scoop verdict on Sid's imputation work

**Genuinely scooped:**

1. The problem framing (broadcast shows 10-16 of 22; team spatial metrics computed on a biased
   subset; imputation as the fix) - fully articulated and quantified, Abstract + Sec 1.
2. The evaluation protocol (simulated viewport applied to open full-pitch tracking, score the
   invisible remainder against full-pitch truth) - identical design, Sec 3.
3. The specific method family (role/formation anchor + centroid voting, causal, online,
   training-free) - Sec 4, Eq. 1.
4. The causal baseline ladder (last-seen-with-decay, formation anchor) that Sid's 11.46 m bar sits
   in - B1/B2, Table 1.
5. Position error in metres as the headline, with a viewport-width sweep - Tables 1-2.
6. Occlusion-gap stratification of the error - Sec 5.1. (Sid's is finer: six buckets vs three, and
   RMSE vs median. Finer, not new.)
7. "Imputation changes a downstream tactical verdict" as a motivating result - Sec 6.

**Genuinely NOT scooped, and worth defending:**

1. **Calibrated predictive regions with verified coverage** (Sid: 50/90% passing 6/6 buckets, with
   per-sample quantile widths needed to fix the 30s+ bucket that a per-bucket constant width
   under-covered at 39%). Choi has zero of this. This is a different *kind* of output, not a better
   number of the same kind.
2. **Abstention derived from calibration** - a policy that says "do not report this metric here".
   Absent from both papers.
3. **Transfer validation against a measured real broadcast footprint** (SkillCorner opendata
   1886347, 37 530 frames, per-frame `image_corners_projection` + detected/extrapolated flags).
   Choi's censoring model is a rectangle he designed; Sid's is a trapezoid he measured. Choi names
   this exact gap as his own limitation.
4. **The finding that the published EMA is not an EMA at its optimum** (half-life 0.04 s dominates
   over three orders of magnitude on this data).
5. Long-gap performance: Sid's 30s+ bucket is 11.24 m RMSE and his 10-30s is 9.74 m RMSE, against
   Choi's >9.6 s at 15.6-16.9 m median. Suggestive that the residual-correction model helps exactly
   where Choi says the headroom is - **but different splits, different error statistics; this is a
   hypothesis to test on Game 3 under Choi's protocol, not a claim to make.**

**Unsentimental summary:** the imputation *model* is a reproduction with an uncertainty layer bolted
on. The uncertainty layer, the abstention it enables, and the measured-footprint transfer test are
the parts that are his.

---

## 2. Theiner et al. 2021: what it actually does

### The task

A **fully-automated modular pipeline for player position estimation from broadcast video**, and -
explicitly - the *evaluation* of it end to end, module by module, against ground-truth positional
data. Their claimed novelty is not any single module; it is (a) integrating all required sub-tasks,
(b) evaluating each module *and* the compounding of module errors on the final output, (c)
generalisation testing on datasets the constituent models were not trained on, and (d) proposing
evaluation metrics for the global task (Abstract, p.1; Contributions, p.2).

Pipeline (Sec 3, Fig. 1, p.2-4): shot boundary detection (TransNetV2) -> shot type classification
(main-camera shots identified via homography stability, threshold tau = 0.35) -> field mask ->
sports field registration (Chen & Little synthetic-edge-image nearest-neighbour + Lucas-Kanade
refinement) -> player detection (CenterTrack) -> team assignment (DBScan on HSV colour of upper-half
bounding box, exactly two clusters, goalkeepers and referees deliberately discarded) -> position via
inverse homography on the bottom-centre of each box.

### Data

Four halves from four **Bundesliga** matches, 25 Hz, with synchronised ground-truth positions from a
calibrated multi-camera system: **TV12** (2012, SD), **TV14** (2014, HD), **TC14** (tactic-cam, HD),
**TV14-S** (broadcast of the same matches as TC14). Ground truth itself "can be inaccurate in some
cases. An error of one meter is to be assumed" (Sec 4.1, p.5). Auxiliary: WC14 for registration,
ISSIA-CNR and Soccer Player for detection, SSET for shot detection.

### Numbers

- **Player detection** (Table 1, p.5): CenterTrack AP 90.1 (ISSIA-CNR) / 90.2 (Soccer Player);
  chosen over Faster R-CNN (87.4 / 92.8) and FootAndBall (92.1 / 88.5) for cross-dataset stability.
- **Team assignment** (Sec 4.3, p.6): macro accuracy 0.91 over three classes, micro 0.93 over the
  two team classes, on 200 manually annotated frames from 20 matches. Most errors are players
  assigned to *other* (goalkeepers, referees).
- **Field registration** (Table 2, p.6), IoU_part: 92.2-96.6 on WC14, but only 59.5-66.5 on TV12,
  85.2-87.4 on TV14, 82.4-92.5 on TC14, 84.5-91.0 on TV14-S. Registration is the weak link and they
  say so.
- **End-to-end player position error** (Table 3, p.7) - the headline. With self-verification (sv,
  rho = 3 m) and player-mismatch (pm, zeta = 0.3) filters, no team-assignment constraint:

  | dataset | frames kept | d_mean | d_med | acc<=2m | acc<=3m |
  |---|---|---|---|---|---|
  | TV12 | 0.79 | 2.10 | 1.55 | 0.64 | 0.80 |
  | TV14 | 0.72 | 2.29 | 1.64 | 0.60 | 0.77 |
  | TC14 | 0.78 | 1.66 | 1.13 | 0.79 | 0.88 |
  | TV14-S | 0.75 | 1.73 | 1.27 | 0.75 | 0.87 |

  **With** the team-assignment constraint the errors roughly double (TV12 3.39/2.99, TV14 3.17/1.71,
  TC14 2.16/1.34, TV14-S 2.78/2.32). Team assignment is a bigger source of positional error than
  detection.

  **Read the fine print:** "d_mean" is not a mean. It is the **average over the best 80 percent of
  position estimates** (q = 0.8), a trimmed statistic they propose in Sec 4.5 (p.7) and justify in
  Appendix E (p.10) as outlier rejection. Table 6 (p.10) gives the honest comparison on TV12: plain
  mean with no filters = **8.90 m**; plain mean with sv+pm = 2.27 m; median with sv+pm = 1.83 m;
  their proposed trimmed aggregate with sv+pm = 1.77 m. So the famous "~1.7-2.3 m broadcast
  accuracy" is **filtered (72-79% of frames kept) and trimmed (best 80% of players)**. Cite it with
  those qualifiers or you are quoting a number that does not mean what it looks like.

### Does it measure how tactical/positional metrics degrade under broadcast conditions?

**No. Not once. Zero tactical metrics are computed anywhere in the paper.**

The closest it gets is a forward-looking sentence in the Conclusions (Sec 5, p.8): *"we claim that
our system outputs promising results in many cases providing a first baseline to conduct various
automatic analyses, for instance, regarding formation detection [2, 36] or space control [12, 41]"*
and *"A relatively small error in meters should allow sports analysts to study team behavior."*
Those are proposals citing Bialkowski et al. and Fernandez & Bornn. They are not experiments.

### Does it compare broadcast-derived values against full-pitch ground truth?

**Positions: yes, but only for players VISIBLE in the frame. Metrics: no.**

Sec 4.5 (p.7): *"Most of the time only a subset of players is visible in the broadcast videos and
there is no information about which player is visible at a certain frame - making evaluation
complex."* Their solution is a per-frame **Hungarian assignment between detections and ground-truth
positions of visible field players**, plus a discard rule (Eq. 1, p.7): drop any frame where the
ratio of detected to expected players deviates by more than zeta = 0.3.

**So the off-screen population - the entire subject of Choi's paper and of RQ-D - is defined out of
their evaluation by construction.** They handle invisibility by *rejecting frames*, not by imputing
or by measuring what the invisibility costs downstream.

### What it explicitly does not do

- No player tracking or re-identification across shots: "player tracking is not covered which would
  lead to more stable predictions across multiple frames" and "additional steps for player
  re-identification (within and across shots) are necessary to allow player-based analysis across a
  match" (Sec 5, p.8).
- Temporal consistency not evaluated quantitatively (Sec 5, p.8; Appendix C, p.9 - the smoothing
  step of ref [47] was not implemented, on runtime grounds).
- Goalkeepers excluded from team assignment by design (Sec 3.3, p.4).
- No uncertainty of any kind. The rho (self-verification) and q (trim) parameters are error-control
  heuristics, and Appendix D Table 5 (p.10) shows the whole rho sweep barely moves d_mean
  (2.12-2.20 m for rho in [1, 5]) while discarding 8-19% of frames - i.e. their abstain knob buys
  almost nothing on the trimmed statistic. Worth knowing before Sid designs his own.

### Scoop verdict

**Zero overlap with RQ-D.** This is an *upstream measurement-error* paper. It is the canonical
citation for "broadcast-derived positions of visible players carry roughly 1.7-2.3 m trimmed error,
degrading to ~3 m once team identity is required, on 72-79% of frames that survive geometric
sanity checks". That is the *instrument noise floor* Sid should be quoting, and it is a gift: it
gives him a published, peer-reviewed number for the non-censoring component of his error budget,
from a group (Memmert, Rein) whose tactical-analytics credibility is unimpeachable.

---

## 3. Why RQ-D is still standing: what neither paper contains

Explicit checklist against RQ-D's components:

| RQ-D component | Choi 2026 | Theiner 2021 |
|---|---|---|
| between-team variance of any metric | absent | absent |
| discriminability / SNR / reliability ratio | absent | absent |
| abstain threshold on metric usability | absent | frame-level geometric abstain only (rho, zeta) |
| pressing metrics | absent | absent |
| transition metrics | absent | absent |
| censoring-induced error on a tactical metric | yes, for pitch control + control share only | absent |
| tactical-verdict flip under censoring | yes, SCI, N=2 windows, no ground truth | absent |
| measured (non-simulated) camera footprint | no - pan-only rectangle, flagged as limitation | n/a (real video, but no footprint model) |
| predictive uncertainty / calibration | no - bootstrap CI on aggregate error only | no |

**Two live risks to name before a supervisor does:**

1. **D is a signal-to-noise ratio, and there is a reliability literature it must be reconciled
   with.** Between-team SD over an EPL season divided by a censoring RMSE is structurally an ICC /
   discriminability argument. Rein & Memmert (cited throughout Theiner) and the "does 4-4-2 exist?"
   line (Muller-Budack, Theiner et al., ref [36] in Theiner) are in that space. Frame D explicitly
   as a *censoring-specific* SNR and relate it to ICC, or it looks invented.
2. **D=1 as "derived abstain threshold" is only derived if the derivation is stated.** D=1 means the
   censoring noise equals the between-team spread - a reasonable convention, but a convention. The
   thing that makes it *derived* rather than asserted is Sid's calibrated regions: propagate the
   predictive region through the metric and you get a per-window uncertainty that can be compared to
   the between-team SD directly, without picking 1. Do that, and D becomes a consequence rather than
   a choice. This is also the single clearest way the calibration work earns its place in the RQ.

---

## 4. The honest framing paragraph (usable as-is in a background chapter)

> The distortion that broadcast framing imposes on team-level spatial metrics has been documented
> from both ends. Theiner et al. [WACV 2022] built the first fully-evaluated broadcast-to-positions
> pipeline and measured its accuracy against synchronised multi-camera ground truth, reporting
> roughly 1.7-2.3 m error for visible players (trimmed to the best 80% of estimates, on the 72-79% of
> frames surviving their geometric self-verification), rising to about 3 m once team identity is
> required; they evaluate only players visible in frame, handle invisibility by discarding frames,
> and compute no tactical metric, closing with the suggestion that such data "should allow sports
> analysts to study team behavior". Choi [arXiv:2607.11548] took the complementary half: simulating a
> ball-tracking viewport over full-pitch Metrica tracking, he showed that simply ignoring off-screen
> players inflates hidden-zone pitch-control error to 25-27 pp and team control-share error to
> 11-13 pp, and that a training-free, causal role-anchored centroid vote roughly halves both, with a
> median hidden-player position error of 9.7-11.6 m at a 44 m viewport. Our imputation model belongs
> to the same method family as Choi's and we make no claim of methodological novelty over it; where
> our work differs is in what it emits and how it was tested. Choi's estimator returns a point per
> hidden player, and his only uncertainty is a block-bootstrap confidence interval on aggregate error;
> ours returns a calibrated predictive region whose 50% and 90% nominal coverage we verify empirically
> across every occlusion-gap bucket, which is what permits a principled abstention rather than an
> unconditional estimate. Choi's censoring model is a rectangular, pan-only window and he identifies
> its inability to zoom or tilt as a threat to the external validity of his occlusion statistics; we
> therefore measured a real broadcast footprint from SkillCorner open tracking - a trapezoid widening
> from 11.0 m half-width at the near touchline to 24.1 m at the far one, inside which 98.4% of
> detected players fall, with detection probability collapsing within about two metres of its edge -
> and re-ran the transfer test against that shape. Neither paper asks the question this thesis asks:
> what the residual censoring error, with or without imputation, does to the *discriminability* of
> pressing and transition metrics between teams.

Two things that paragraph deliberately does: concedes the method, and states the two differentiators
in the incumbents' own terms (Choi's stated limitation, Theiner's stated non-coverage). Do not soften
the concession - a supervisor who has read Choi will find it in thirty seconds, and volunteering it
is worth more than the claim it costs.

---

## 5. Better research questions suggested by the papers

Ranked. All three exploit assets Sid already has.

### RQ-D' (strongest): occlusion-duration-conditioned abstention

**"At what occlusion duration does off-screen imputation stop being informative for pressing and
transition metrics, and can calibrated predictive regions locate that boundary automatically?"**

Why this is better than RQ-D as stated:

- It is **invited by the incumbent**. Choi (Sec 5.1, p.7) establishes that **50-57% of hidden
  observations exceed 9.6 s**, that his error there is 15.6-16.9 m median (vs 3.3-3.7 m at <=2 s),
  that the closest learned prior work (Graph Imputer) *cannot operate* in that regime by
  construction, and writes that this "shows the remaining headroom, which we expect learned
  long-horizon models to attack". Sid's v1 is exactly that and already reports 9.74 m (10-30 s) and
  11.24 m (30 s+) RMSE.
- It turns D from a scalar into **D(gap)**, a curve. A curve with a crossing point is a result; a
  scalar with an asserted threshold is a convention.
- It is the one thing a point-estimate ladder structurally *cannot* do. Choi cannot answer it
  without inventing the uncertainty layer Sid already built and validated.
- Sid already has the abstention policy artefact (`results/B4_ABSTENTION_POLICY.md`) and the
  risk-coverage machinery, so the marginal work is applying it to pressing/transition metrics rather
  than to position error.

### RQ-D'' : two-stage abstention (geometry x resolution) with a measured yield curve

**"Composing a frame-level geometric abstain with a metric-level resolution abstain: what fraction
of broadcast match time yields a tactically usable pressing/transition metric, and at what fidelity?"**

Theiner's sv/pm filters (rho, zeta) are a *geometry* abstain that keeps 72-79% of frames (Table 3,
p.7) - and Appendix D Table 5 (p.10) shows the rho sweep buys almost nothing on the trimmed error,
which is itself a finding worth citing. Sid's feasibility census says geometry costs 84% of usable
turnover windows and ball tracking 65%. **Nobody has composed the two stages and measured the
yield/fidelity trade-off.** This is a systems contribution, it directly consumes the census (which is
otherwise an internal document), and it is the honest answer to "can you actually do this on real
broadcast" - which is the question a BTP examiner will ask.

### RQ-D''' : do imputed players carry usable velocity?

Choi computes pitch control with **zero velocities in all conditions**, deliberately, and lists
velocity-aware control with imputed velocities as open (Sec 3, p.3; Sec 7, p.8-9). His own ablation
found constant-velocity extrapolation "mildly helpful, not harmful" for position (B3V) but he never
tested its effect on a velocity-dependent metric. **Pressing metrics are intrinsically
velocity-dependent** - approach speed, closing rate, pressure intensity, counterpress reaction time.
So there is a stated open question that happens to be a precondition for Sid's chosen metric family.
Smallest scope of the three, and it would slot in as a component result inside RQ-D' rather than
standing alone.

---

## 6. Bibliographic details for the thesis

- Seongjin Choi. *Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial Football
  Analytics.* arXiv:2607.11548v1 [cs.CV], 13 July 2026. Preprint; single author; independent
  researcher; ORCID 0009-0001-3193-7424. Code: github.com/nowayfootball/offscreen-impute (benchmark
  code public; case-study pipeline and footage not included). Not peer-reviewed as of this reading -
  state that when citing.
- Jonas Theiner, Wolfgang Gritz, Eric Muller-Budack, Robert Rein, Daniel Memmert, Ralph Ewerth.
  *Extraction of Positional Player Data from Broadcast Soccer Videos.* arXiv:2110.11107v1 [cs.CV],
  21 October 2021. Published at WACV 2022. Peer-reviewed; cite the WACV version.

Key third-party works both papers lean on, worth reading next if the background chapter needs depth:
Omidshafiei et al., *Multiagent off-screen behavior prediction in football*, Scientific Reports
12:8638, 2022 (the Graph Imputer - the learned incumbent, 105 proprietary EPL matches, bidirectional,
9.6 s windows); Somers et al., *SoccerNet Game State Reconstruction*, CVPRW 2024 (GS-HOTA, scores
visible players only); Penn, Donnelly, Bhatt, *Continuous football player tracking from discrete
broadcast data*, Royal Society Open Science 12(10):251175, 2025 (ARMAX reconstruction with ball as
exogenous input - temporal rather than spatial sparsity).
