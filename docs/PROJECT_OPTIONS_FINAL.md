# PROJECT_OPTIONS_FINAL — four candidate BTP projects, graded on the measured assets (2026-07-26)

Audience: Sid, and the two examiners in December 2026. Written after reading the two new papers
(`docs/papers/gnn approach.pdf`, `docs/papers/A_team_ball_game_tactical_recognition_and_result_p.pdf`)
and re-auditing what the repository actually holds. Supersedes the ranking in
`docs/RESEARCH_QUESTION_OPTIONS.md` §3 only where it contradicts it; the critique in that document's
§1 and §2 stands and is assumed here.

---

## Part 0 — the GNN papers, decided first because they change nothing

### 0.1 `gnn approach.pdf` is not an event-detection paper

It is **Omidshafiei et al., "Multiagent off-screen behavior prediction in football", Scientific
Reports 12:8638 (2022)** — DeepMind + Liverpool FC. The task is **off-screen player trajectory
imputation**: given the on-screen subset of players, estimate where the off-screen ones are. That is
the B4 problem this project already built, gated and transfer-tested. There is no event detection,
no event stream, no action recognition anywhere in it. It is already cited in
`docs/RESEARCH_QUESTION_OPTIONS.md` §1.5 as the closest prior work to the imputation module.

| dimension | what the paper requires |
|---|---|
| task | multiagent time-series imputation (fill missing player (x, y) between observations) |
| input | **full 25 fps tracking of all 22 players + ball**, proprietary provider, downsampled to 6.25 fps |
| training filter | "we retain only trajectories with **all 22 players' ground truth**" — partial-truth sequences are discarded |
| corpus | **105 English Premier League matches**, cut into 9.6 s / 240-frame sequences -> 30,838 train + 4,024 eval trajectories |
| occlusion | *simulated* by a synthetic view cone over the full truth (12.76 +/- 3.70 of 22 in frame, 4.94 +/- 3.49 s in-frame duration) |
| architecture | bidirectional LSTM (2 layers, 64 hidden) + GraphNet encoder/prior/decoder (2-layer MLPs, 64 units, 23 nodes = 22 players + ball) + VRNN latents (16-d), forward/backward fusion |
| training | 10^5 iterations, batch 24, Adam 1e-3, 5 seeds per hyperparameter cell, wide sweep |
| headline | L2 imputation loss **0.153** vs GVRNN 0.865, bidirectional Social LSTM 0.198, cubic spline 0.193; pitch-control MAE **0.0207** vs 0.0418 (GVRNN) |
| baselines | spline (linear/quad/cubic), autoregressive LSTM, Social LSTM, role-invariant VRNN (hand-crafted for football by the same authors), GVRNN |
| data availability | **"not publicly available due to licensing restrictions"** |

**Trainable on what Sid holds? No — and the blocker is data, not GPU.** The model itself is tiny
(64-unit LSTMs and MLPs, 16-d latents) and would fit a 4 GB card comfortably. What it cannot be given
is its input: it trains on sequences where *all 22 players' true positions are known at every frame*.
Sid has 12 broadcast matches with ~11.8 of 22 players visible and no ground truth at all on them; the
only full-truth football tracking in the repo is Metrica Sample Games 1-2 (`data/imputation/metrica`)
plus SkillCorner opendata, i.e. **~2 matches of truth against their 105** — roughly 2% of the
training set, from a single source, with no held-out league. A Graph Imputer trained on Metrica would
be a demonstration of the code, not a model, and the existing frozen quantile-GBM v1 already clears a
pre-registered causal bar (8.10 m vs 11.46 m, 50%/90% regions holding 6/6) at a fraction of the
complexity. Building it would be re-deriving a scooped contribution with 2% of the data.

**Usable at all?** Two things, both free:
1. **As a citation and a number.** Their camera model produces 12.76 +/- 3.70 visible of 22 — an
   independent third confirmation of this project's 11.8/22 and of the 11.57/22 measured in
   `results/ENTROPY_MAP_FEASIBILITY.md`. Quote it in Chapter 2.
2. **As the missing gap statement.** They imputed with a synthetic view cone and never validated on
   real broadcast; Sid's SkillCorner-derived footprint (98.4% of detected players inside the reported
   camera polygon, `results/B4_TRANSFER_M3.md` §0) is a *measured* censoring operator, which theirs
   is not. That asymmetry is the one defensible thing left in this territory and it is Option A/B
   below.

**Verdict on "a GNN might make event detection work": the paper does not support it, because the
paper is not about event detection, and no version of its method is trainable on 12 broadcast
matches with no ground truth.**

### 0.2 `A_team_ball_game_tactical_recognition_...pdf` — do not build on it, do not cite it as support

Yan, Liang & Qiao, *Discover Applied Sciences* (2026), "Tactic Graph Net": GATv2 + ST-GCN over a
dynamic player-ball graph, 40-frame (4 s) segments, node features = (x, y, vx, vy, ball-possession
flag, team id), multi-task heads for tactic class + xG/EPV regression. Trained on 26,450 football
segments, 4x A100.

It is the right *shape* of thing (tactical recognition on trajectories) and it is a bad paper:

- **The football input it describes does not exist in the source it names.** §4.1 says the football
  data is **StatsBomb Open Data** sampled at **10 Hz with player positions, ball positions and
  velocities**. StatsBomb Open Data is an *event* stream (plus 360 freeze-frames); it contains no
  continuous 10 Hz positional track for any player. The stated features cannot be derived from the
  stated source.
- **The numbers do not agree with themselves.** Abstract: 91% football / 92.3% basketball. §4.2: "F1
  0.89 soccer and 0.87 basketball", then "classification accuracy for soccer and basketball is 0.91
  and 0.93". Abstract says 14 tactic categories; Table 1 lists 8 football attack types; Figure 6's
  confusion matrix has 12. Table 3 has two differently-valued columns both labelled "Tactic Graph
  Net". Latency is reported as "less than 8 mm".
- **Labels are private and hand-made** (three analysts, Krippendorff alpha 0.82) and the trained
  model is the only artifact; nothing is reproducible.

Its one useful contribution to Sid's decision is the input-feature ablation (Table 4): position-only
84.2% -> +velocity 87.6% -> +possession flag 89.3% -> +team id 89.8%. Even taken at face value that
says the marginal value of the ball-possession flag over pure geometry is ~1.7 points — i.e. a
tactical-recognition system does **not** need the per-player attribution layer that measured 0.36-0.40
here. That is a mildly encouraging data point for Option C, and nothing more.

---

## Part 1 — the asset audit the options are graded against

Everything below is a measured number from this repository, not an aspiration.

| asset | state | number that matters |
|---|---|---|
| CV pipeline (detect/track/homography/team/ball/events) | works, 12 matches processed | passes 0.97-1.09x official on 10/12; goals 14/16 |
| Ball track (post-`link_ball`) | works, censored | **32-52% coverage**, pooled 38.8% at 1 Hz |
| Player geometry | works, censored | **11.57 trusted rows/frame of 22**; 47.4% of ball frames have *zero* trusted player row |
| Calibrated camera simulator | validated | **98.4%** of SkillCorner-detected players inside the reported footprint; footprint shape measured, not assumed |
| Imputation B4 v1 (frozen) | gated + transfer-tested | 8.10 m vs 11.46 m causal bar; 50%/90% regions hold 6/6; transfer 105-112% on a measured footprint; **scooped by Choi 2026 as a contribution** |
| Per-player attribution | **CLOSED — negative** | held-out end-to-end precision **0.36-0.40** (95% CI 0.20-0.59) at 13-14% coverage; constraints add +0.04 = one moment; jersey OCR **0.000**, and fails silently |
| Counterpress *rate* metrics | **CLOSED — underpowered** | need **58-90 matches** at the irreducible binomial floor; we have 12 |
| Transition *situations* (2 s window) | **OPEN** | 767 usable pooled / **median 67 per match** at N>=3 pressers; 421 / 34 at N>=5 |
| Ball-movement style identity (Lucey entropy maps) | **CLOSED — at chance** | 0.168 vs a 0.191 null over 90 settings; needs >=8 s possession strings, we have 3 s |
| "Team identity" claims | mostly dissolved by opponent control | wide-attack funnel is league geometry (12/12 opponents too); compactness constant holds for all 7 sides; line height leg-to-leg r = **-0.40** |
| Tactical clip renderer | **NEW, working** | `tools/tactical_clip.py`, 1037 lines, self-check green, two rendered passages in `results/tactical_clips/_test/`; refusal gate fires on quality; draws 50%/90% calibrated regions |
| Free ground truth held | — | Metrica Games 1-2 (25 Hz, 22 players), SkillCorner opendata, SoccerNet GSR-2024 + jersey-2023, 90 human carrier labels, FIFA PMSR PDFs |

**Three hard rules the options obey** (from the constraints, and each one is a measured
consequence): no option depends on per-player event volume (0.36 precision); no option compares two
teams on a rate at n=12 (needs 58-90); no option needs data that is not already on disk or free.

---

## Part 2 — the options

### OPTION A — The resolution limit of broadcast tactical measurement
*(the measurement/methods study that cannot fail into nothing)*

**QUESTION.** For each tactical metric computable from positions, how large is the error that
broadcast censoring induces, **relative to the effect the metric exists to detect** — and at what
level of player visibility does that ratio cross 1? Does calibrated imputation move the crossing?

```
D(m, v) = between-team SD of metric m  /  RMSE of m induced by censoring at visibility v
```

`D > 1`: the metric can still discriminate under broadcast conditions. `D < 1`: measurement noise
swamps the football. The abstain threshold becomes a derived quantity, not an engineering constant.

**METHOD.**
1. Metric battery on Metrica full truth, all geometric, none event-count-based: defensive line height
   (absolute and ball-relative), vertical/horizontal compactness, team surface area / stretch index,
   inter-line distances, and the *instantaneous* Bauer-Anzer transition features at BPC+{0,1,2} s
   (nearest-presser distance and speed, players within 10 m of the ball, local-5 stretch, Andrienko
   pressure). Every one is a per-window geometric functional — no passer names, no per-player counts.
2. Censor with the SkillCorner-calibrated simulator over a visibility sweep (v = 8, 10, 11.8, 14, 16,
   18, 22 of 22), recompute, and pair each censored window against its own truth. The unit of
   analysis is **the window, of which there are thousands**, not the match, of which there are two —
   this is what makes n=2 games acceptable and it must be said out loud in the write-up.
3. Four recovery arms as an explicit ablation: (a) ignore off-screen players (what SoccerNet GSR
   effectively does), (b) linear interpolation (what challenge winners do), (c) a *fitted scalar
   correction* with no model at all — the baseline that would embarrass the thesis if it won, so it
   goes in first, (d) frozen B4 v1 with its 50%/90% regions Monte-Carlo-propagated to a metric-level
   interval. Nobody in this literature propagates imputation uncertainty into the metric.
4. The denominator (between-team SD) comes from **published tables and free full-tracking data, never
   from our 12 matches**: Bauer & Anzer's Tables 7/8/10-11 are already PDF-parsed
   (`results/W1B_WINDOW_AND_SAMPLING.md` §2, 126 team-seasons), Frencken/Bekkers give structural SDs,
   SkillCorner opendata supplies a second free estimate. This is the clause that keeps the study
   inside the n=12 rule.
5. Real-broadcast transfer, stated as a consistency check and not as a team comparison: do the
   `D > 1` metrics show between-match structure in the 12-match corpus, and do the `D < 1` ones show
   none? `results/W1B_WINDOW_AND_SAMPLING.md` §3 already has the machinery and the honest df caveat.

**DELIVERABLE the examiner touches.** *Table 1*: one row per metric — truth value, censored bias,
censored RMSE, between-team SD, D under each of the four arms, verdict
REPORTABLE / REPORTABLE-WITH-CORRECTION / ABSTAIN. *Figure 1*: D against visible-player fraction, one
curve per metric, a horizontal line at D = 1 — the single figure the thesis is about. *Artifact*: the
report gate re-wired so its thresholds are the measured D = 1 crossings, demonstrated live by
regenerating a report and watching a claim get withdrawn when visibility drops.

**CONSUMES.** Camera simulator (the censoring operator, and its 98.4% validation becomes a result
rather than a footnote); frozen B4 v1 + conformal multipliers + abstention policy; `fingerprint`
block-height / compactness / structural code; the W1B parsed literature tables; the 12-match corpus
as a transfer set; the failed marginal style fingerprint as the motivating negative control.

**STILL NEEDS BUILT.** The metric battery on Metrica (~5 metrics x ~150 lines); the visibility-sweep
harness (the censoring call already exists in `synthesizer.imputation`); Monte-Carlo propagation from
the calibrated regions to a metric interval; the four-arm ablation runner; the D table/figure.

**RISK, AND WHAT FAILURE YIELDS.** Low. The failure modes are (i) *every* metric returns D < 1 — that
is the thesis, stated as "broadcast video cannot resolve team-level tactical differences below X
visibility; here is the visibility you would need", a hard negative with a number; (ii) the fitted
scalar correction matches the imputation model — that is a finding and a good one; (iii) literature
denominators are unavailable for some metric — that metric is reported as bias/RMSE only and named as
such. There is no branch where the work yields nothing.

**EFFORT to December (~19 weeks available).** Battery on Metrica 4 wk; censoring sweep + four arms
3 wk; denominators + propagation 3 wk; real-broadcast transfer 2 wk; demo chapter (Option B) 3 wk;
writing 4 wk. Fits, with two weeks of slack.

---

### OPTION B — The glass-box tactical replay: the first broadcast-native validation of off-screen imputation
*(the demo-first option, centred on the working renderer)*

**QUESTION.** Off-screen imputation is validated only on simulators (DeepMind, Choi, and this
project's own B4). **Can it be validated on real broadcast footage, where no ground truth exists —
and what does a tactical visualisation owe a viewer when the positions it draws are inferred?**

The trick that makes it answerable: **re-appearance is free ground truth.** When a hidden player
walks back into frame, the tracker measures where he actually is. Comparing the model's last
prediction (and its 50%/90% region) against that measured re-entry position gives a real error and a
real coverage number **on our own footage**, which `results/B4_TRANSFER_M3.md` currently marks as a
"smell test" only. The obvious objection — re-appearance-conditioned error is a biased sample of all
occlusions — is answerable rather than fatal: run the *identical* re-appearance-only protocol on
Metrica, where the full truth is known, measure the bias between re-appearance-conditioned error and
true error, and carry that correction to the broadcast number.

**METHOD.**
1. Re-appearance census on all 12 matches: how many hidden slots re-enter the frame within the
   renderer's 20 s cap **and** are linkable to the same slot? Distribution per match, not the mean.
   This is the go/no-go and it is one day of work.
2. Metrica bias calibration: the same protocol on full truth -> `E[err | reappears]` vs `E[err]`, per
   occlusion-duration bucket, plus coverage of the 50%/90% regions under both conditionings.
3. Broadcast measurement: error and region coverage at re-entry across the 12 matches, corrected by
   step 2, reported per horizon bucket with intervals.
4. The renderer becomes the instrument that displays it: every clip carries the measured broadcast
   coverage of the regions it is drawing, and the abstention state is shown, not hidden.
5. A shortlisted clip set across the corpus (build-up vs low block, transitions, the refusal cases),
   each with its quality numbers printed on the frame.

**DELIVERABLE.** A rendered clip library plus one number that does not currently exist anywhere in
this literature: *the empirical coverage of a calibrated imputation region on real broadcast video*.
Plus the refusal demo — passages the system declines to render, with the failing numbers on screen.
`tools/tactical_clip.py` already implements the refusal gate, the three visual states and the
provenance banner; this option turns its honesty banner into a measured claim.

**CONSUMES.** The renderer (working, self-check green); frozen B4 v1 + conformal + abstention; the
identity gate at >=3 agreeing anchors (which is the *correct* use of a 0.36-precision identifier —
name almost nobody, band everybody); `make_demo` drawing machinery; block-height and phase detectors
for the caption strip; Metrica as the bias calibrator.

**STILL NEEDS BUILT.** The re-appearance linker and its census; the Metrica conditioning study; the
correction; a clip batch runner across all 12 matches; a short evaluation write-up.

**RISK, AND WHAT FAILURE YIELDS.** **Medium, and the risk is concentrated in one measurable place.**
If linkable re-appearances are rare — plausible, because re-id churn is exactly what the 20 s cap
exists to contain, and the per-track team label is already wrong 17% of the time — the broadcast
validation collapses to a small sample and the headline number carries a useless interval. Failure
still yields: the census itself (a publishable-shaped statement about how often broadcast tracking
offers self-validation), the Metrica conditioning result (how biased re-appearance-conditioned error
is, which nobody has measured), and the renderer as a demo artifact. It does **not** yield a thesis
on its own — which is why it is ranked as a chapter, not a project.

**EFFORT.** Census 1 wk; Metrica conditioning 2 wk; broadcast measurement 2 wk; clip library +
write-up 2 wk. ~7 weeks standalone, ~3 weeks if it sits on top of Option A's harness (it reuses the
same censoring/propagation code).

---

### OPTION C — Counterpress or retreat: a per-situation decision from broadcast, validated against human labels

**QUESTION.** Bauer & Anzer detect counterpressing from full tracking and report it as a team *rate*
— which this corpus provably cannot support (58-90 matches needed). **Can the underlying per-loss
decision — did this team counterpress or retreat? — be made from broadcast geometry at the three
instants the literature actually uses, with a validated accuracy and a principled abstention?**

This is deliberately a *classification* question with human ground truth, not a rate comparison. It
is the one live opening `results/W1B_WINDOW_AND_SAMPLING.md` left: the 2 s window multiplied the
usable pool 5.5-6.3x and moved the verdict from FAIL to PASS at N>=3 and N>=5 (median 67 and 34
usable turnovers per match).

**METHOD.**
1. Sid labels ~250-300 turnovers across 3-4 matches as PRESS / RETREAT / UNCLEAR from the video
   alone, using a labelling tool in the mould of `tools/make_carrier_labeller.py`. **Only the
   behaviour is labelled, never the 5 s outcome** — that dodges the 73.4% informative censoring on
   the success label, which is documented and would otherwise sink the study.
2. Feature set restricted to what is measurable at BPC+{0,1,2} s: nearest-presser distance and
   closing speed, count within 10 m, local-5 stretch, Andrienko pressure, block-line displacement.
   No per-player identity, no event counts.
3. A rule-based detector first (BA's own construct), then a small classifier, both scored against the
   labels with precision/recall/abstention curves, plus inter-labeller agreement on a re-labelled
   subset so the ceiling is known.
4. The derived abstention rule: refuse the decision where the visible-presser count or the geometry
   quality puts the classifier below a stated accuracy. Each decision renders as a clip via
   `tools/tactical_clip.py`, which is what an examiner watches.

**DELIVERABLE.** A precision/recall/abstention table against human labels, an agreement ceiling, and
a clip reel of correct calls, wrong calls and refusals.

**CONSUMES.** The W1B census and its turnover definition; the labelling-tool pattern and scorer; the
2 s instant machinery; the clip renderer; the block-height detector.

**STILL NEEDS BUILT.** The labelling tool for turnovers; 250-300 human labels (Sid's own time, ~2-3
weeks elapsed, not compressible); the feature extractor; the detector; the scoring harness.

**RISK, AND WHAT FAILURE YIELDS.** **Medium-high.** Three separate exposures: (i) the labelling
burden is Sid's calendar time and it is the same failure mode that capped the carrier study at 90
labels and left every precision with a +-0.18 interval — 250 labels is the minimum for a usable
operating point and it is a real commitment; (ii) the decision may simply not be visible at N>=3
pressers, in which case the abstention rule swallows most of the corpus; (iii) it partly re-treads
Bauer & Anzer, so the novelty rests entirely on "from broadcast, with a validated refusal". Failure
yields the labelled dataset (genuinely reusable and citable), the agreement ceiling, and a measured
statement of what broadcast geometry cannot decide — decent, but thinner than Option A's failure.

---

### OPTION D — Tracking continuity as the binding constraint: diagnose it, fix it, state the acceptance criterion

**QUESTION.** Three separate studies in this repository died on the same number: **47.4% of ball
frames carry no calibration-trusted player row.** It killed the entropy-map representation (3 s
possession strings vs the 8 s the method needs), it is the dominant term in the possession-label rate
(0.315 vs Metrica's 0.668), and it caps every windowed metric. **What causes the frame-level dropout,
what does fixing it recover, and what is the acceptance criterion?**

**METHOD.** Decompose the 47.4% into its causes (detection miss, calibration failure, replay/cut,
close-up shot, track fragmentation) on a stratified sample; fix the cheapest large term; re-run the
entropy-map and W1B censuses as the acceptance test with their pre-declared criteria (median
possession string >= 8 s; usable 2 s windows per match).

**DELIVERABLE.** A dropout attribution table, one engineering fix, and a before/after on two censuses
that already have pre-declared thresholds — i.e. it cannot be graded generously.

**CONSUMES.** The entropy-map probe, the W1B census, the calibration and tracking stack.

**RISK, AND WHAT FAILURE YIELDS.** Low technical risk, **high grading risk**: this is systems work,
and `docs/RESEARCH_QUESTION_OPTIONS.md` §1.1 is right that the pipeline layer is not a research
contribution — Theiner 2022 and SoccerNet GSR own it. If the fix works, the reward is that other
options get better inputs; the finding itself is "we improved our own tracker". Ranked last for that
reason, and listed because it is the highest-leverage *enabler* if any option stalls on data volume.

**EFFORT.** 4-6 weeks, and it is the one option worth doing in parallel with another if a GPU week
frees up.

---

## Part 3 — the ranking

Scored on: defensibility in front of two strong technical examiners x feasibility on the measured
constraints x reuse of existing work. 1-5 each, product shown.

| # | option | defensibility | feasibility | reuse | product | one-line reason |
|---|---|---|---|---|---|---|
| **1** | **A — resolution limit** | 5 | 5 | 5 | **125** | Turns two scooped assets into instruments, needs no new data, and its worst outcome is a hard negative with a number. |
| **2** | **B — glass-box replay** | 4 | 3 | 5 | **60** | The only route to a validated imputation number on real broadcast, and the most demo-able artifact already runs — but the re-appearance census can come back thin. |
| **3** | **C — counterpress decision** | 4 | 2 | 3 | **24** | Scientifically clean and on the professor's stated interest, but gated on 250-300 hand labels of Sid's own time and partly re-treads Bauer & Anzer. |
| **4** | **D — continuity** | 2 | 5 | 4 | **40 (defensibility-capped)** | Highest leverage, lowest novelty; an enabler, not a thesis. Ranked below C despite the higher product because "we fixed our tracker" is not a finding. |

---

## RECOMMENDATION

**Commit to Option A as the thesis, and run Option B as its demonstration chapter. That is one
project, not two.**

> **Under measured broadcast visibility, which tactical metrics retain enough resolution to
> distinguish one team from another — and does calibrated off-screen imputation move that threshold?**

Why this and not the others:

1. **A and B compose without duplicated work.** A builds the censoring sweep and the propagation from
   calibrated regions to metric-level intervals; B is that same machinery pointed at real footage,
   with re-appearance supplying the truth A's transfer chapter otherwise lacks. B costs ~3 weeks on
   top of A instead of ~7 standalone.
2. **It is the only framing where the two scoops become strengths.** The imputation model stops being
   a contested contribution and becomes one arm of a four-arm ablation; the DeepMind paper Sid just
   read stops being a threat and becomes the row above his in Chapter 2's comparison table.
3. **It obeys all three hard rules by construction.** Every metric in the battery is a geometric
   functional of positions — no per-player attribution anywhere. The between-team denominator comes
   from published tables and free full-tracking data, never from n=12. Nothing needs buying.
4. **The demo an examiner touches is already rendering.** `results/tactical_clips/_test/` holds two
   finished passages today, with the refusal gate and the three visual states working.
5. **It cannot fail into nothing**, and the two things most likely to go wrong (all-ABSTAIN; the
   scalar correction beating the model) are both reportable results rather than dead ends.

Kept from the current framing: the corpus design, the pipeline as Chapter 3 apparatus, the scouting
pack as the applied demo. Dropped: "validated-or-abstain" as a contribution in its own right; the
r = +0.83 / -0.40 stability numbers as findings; anything that needs a named passer.

### First two weeks under the recommendation

Nothing here is exploratory. Each item ends in a number that changes the plan if it comes back wrong.

**Week 1 — the loop, on one metric, end to end.**

| day | work | done when |
|---|---|---|
| 1 | Metric battery skeleton on Metrica: **ball-relative defensive line height** only, computed per 1 s window over Games 1-2, both teams, with an assert-based self-check on a hand-worked frame. | a per-window truth series exists and the self-check passes |
| 2 | Censoring harness: wire `synthesizer.imputation`'s camera window to a visibility sweep (8, 10, 11.8, 14, 16, 18, 22 of 22), emitting the *paired* (truth window, censored window) set. | paired sets exist at 7 visibility levels; visible-count per level verified against target |
| 3 | Arms (a) ignore and (b) linear interpolation; bias and RMSE per visibility level. | two curves of RMSE vs v |
| 4 | Arm (c): the fitted scalar correction with no model. This one goes in before the imputation arm deliberately — if it wins, the thesis needs to know in week 1. | third curve, and an explicit statement of whether it beats (a)/(b) |
| 5 | Arm (d): frozen B4 v1 via `tools.imputation_b4_external.fit_frozen_v1`, plus Monte-Carlo propagation of the 50%/90% regions into a metric-level interval. | fourth curve **with** interval bands |

**Week 2 — the denominator, the first table row, and the go/no-go for the demo chapter.**

| day | work | done when |
|---|---|---|
| 6 | Between-team SD for line height from free sources: SkillCorner opendata across its matches, plus any published SD. **Explicitly not from the 12-match corpus.** | one denominator with a stated provenance and an interval |
| 7 | Compute `D(v)` for all four arms. Produce **row 1 of Table 1 and curve 1 of Figure 1**, with the D = 1 crossing read off. | the figure exists as a png and the crossing has a number |
| 8 | Extend the battery to compactness and stretch (cheap: same window machinery, different functional) and re-run days 3-7 as a batch. | three rows of Table 1 |
| 9 | Sanity transfer: do the three metrics show between-match structure in the 12-match corpus in the direction `D` predicts? Report as a consistency check with the df caveat from `W1B` §3 — not as a team comparison. | one paragraph with numbers, or an honest "underpowered, declined" |
| 10 | **Re-appearance census** on all 12 matches: how many hidden slots re-enter within 20 s and link back to the same slot? Distribution per match. This is the go/no-go for Option B. | a per-match count; >= ~30 linkable re-entries per match keeps chapter B, < 10 kills it |

**Decision gate at the end of week 2.** Three rows of Table 1 + one clean D curve -> Option A is a
matter of repetition; commit and expand the battery. Scalar correction beating imputation -> keep it
as the headline finding and reduce the imputation arm to a comparison. Re-appearance census healthy
-> chapter B proceeds; thin -> B degrades to the Metrica-conditioning result plus the clip library as
a pure demo, and the freed weeks go to the transition-instant half of the battery.
