# B4 model plan — v1 architecture, calibration, abstention

Written 2026-07-24. Extends `docs/B4_IMPUTATION_PLAN.md` (problem statement, truth data, censoring
simulator, baseline tables, gates 1-4) — that doc stays the source of truth for everything already
measured. This doc only covers what happens **after** the supervisor's gate-1 decision: which model
we build, how it reports uncertainty, how it abstains, and when.

---

## 1. The decision, and the exact numbers v1 must beat

Supervisor's ruling (2026-07-24, option b, recorded in `B4_IMPUTATION_PLAN.md` line 61): **the
pass/fail line is the best CAUSAL baseline `B5_blend`.** Offline linear interpolation is reported
alongside as an **oracle ceiling only** — it consumes the future re-sighting, which no causal
imputer has. This is an information-asymmetry argument, not a performance rescue: it was made
before any learned-model number existed.

The bar, per time-since-last-seen bucket (held-out Game 2, RMSE metres):

| horizon | B1_hold | **B5_blend = THE BAR** | B2_offline (oracle ceiling, not the bar) |
|---|---|---|---|
| 0-1s | 1.38 | **0.55** | 0.91 |
| 1-3s | 4.48 | **2.27** | 2.63 |
| 3-5s | 8.35 | **5.08** | 4.69 |
| 5-10s | 13.58 | **9.05** | 7.35 |
| 10-30s | 21.80 | **17.27** | 10.17 |
| 30s+ | 19.20 | **15.95** | 12.46 |
| ALL | 16.16 | **12.63** | 8.66 |

What this means concretely for the model target:

- **Gate 1 is now six separate tests, not one.** v1 must be at or below 0.55 / 2.27 / 5.08 / 9.05 /
  17.27 / 15.95 in *every* bucket, on the frozen holdout, with the win surviving a 1-minute
  block-bootstrap CI (i.i.d. bootstrap is invalid on 25 fps trajectory data).
- **The short buckets are nearly closed.** 0-1s is 0.55 m — that is the velocity-extrapolation noise
  floor. There is no headroom there and we should not spend effort chasing it; matching the bar in
  0-1s and 1-3s is a pass.
- **The money is at 5-10s, 10-30s, 30s+.** The bar leaves a 7-9 m gap to the oracle ceiling at
  10-30s (17.27 vs 10.17). That gap is where structure information lives, and it is exactly where
  our own ablation says memory-alone and structure-alone both fail (30s+ improves 18.58 -> 15.95
  only when blended). External corroboration: the closest published analogue
  (https://arxiv.org/html/2607.11548v1) finds error roughly doubles past ~9.6 s of occlusion
  (3.3-3.7 m at <=2 s vs 15.6-16.9 m at >9.6 s) and that 50-57% of hidden observations sit in that
  hard regime. Caveat carried forward: **their numbers are medians, ours are RMSE** — they are not
  directly comparable and must not be quoted as "they beat our bar".
- **Non-monotonicity is real and must be preserved in reporting.** 30s+ (15.95) is *better* than
  10-30s (17.27) for every baseline, because 30s+ gaps concentrate on low-mobility situations
  (stoppages, a keeper, a deep full-back). Do not "fix" this; explain it.

---

## 2. Recommended model v1 — pick, fallback, stretch

### 2.1 The pick (build this)

**B4-v1 = quantile gradient-boosted residual correction on the frozen `B5_blend` anchor, in an
attack-aligned frame, with per-bucket conformal calibration.**

Concretely: `sklearn.ensemble.HistGradientBoostingRegressor(loss="quantile", quantile=q)`, one model
per axis per quantile, q in {0.05, 0.25, 0.50, 0.75, 0.95} — 10 small CPU models total. The target
is **not** the player's position; it is the residual `truth - B5_blend_prediction`, expressed in a
frame rotated so +x is the player's team's attacking direction.

Justification, in order of weight:

1. **Residual-on-anchor makes the bar the floor by construction.** If the learner finds nothing, it
   predicts ~0 residual and we tie `B5_blend` rather than losing to it. With two matches of training
   data, guaranteeing "no worse than the bar" matters more than headroom.
2. **The literature explicitly warns against a heavier model here.** MIDAS
   (https://www.arxiv.org/pdf/2408.10878v3) shows Graph Imputer (3.5181) losing to plain cubic
   spline (1.9209) and linear interpolation (3.1083) on the *same Metrica 3-match dataset with the
   same 2/0.5/0.5 split we use*, and attributes the gap to small-data regime (2-3 soccer matches vs
   70 basketball games, where the gap closes). Agent Imputer
   (https://arxiv.org/pdf/2302.06569) puts a plain XGBoost at 9.26 m against a full LSTM+GNN at
   6.88 m on the same engineered features — a tree model on tabular features gets most of the way.
3. **Zero new dependencies.** scikit-learn 1.8 is already installed and already a declared
   dependency. LightGBM, NGBoost, MAPIE and PyTorch-for-this all buy a few points at the cost of a
   new install on a 4 GB laptop; sklearn's quantile HistGBM plus ~10 lines of conformal arithmetic
   covers the gate.
4. **Our features are already tabular.** `synthesizer/imputation.py:collect_samples` already emits
   `tsls / hold / linear / v0 / slot` per hidden sample. v1 is a feature-widening of an existing
   collector, not a new pipeline.

**Uncertainty head:** the quantile heads *are* the head — pinball loss at 0.05/0.25/0.75/0.95 gives
per-axis, locally adaptive half-widths that widen where the model is locally worse, and then a
single conformal correction per bucket makes coverage finite-sample valid
(Conformalized Quantile Regression, https://candes.su.domains/publications/downloads/CQR.pdf). No
Gaussian assumption, which matters because our residuals at 30s+ are visibly not Gaussian.

**Loss:** pinball (sklearn `loss="quantile"`) for the interval heads; pinball at 0.5 (i.e. MAE) for
the point head. Deliberately **not** MSE for the point head — MAE at the median is robust to the
long-tail 10-30s outliers that would otherwise drag the median prediction. RMSE is still the gate
metric; we report both RMSE (for the gate) and median error (for comparability with
arXiv 2607.11548).

**Inputs (concrete, ~34 columns).** All positions in the attack-aligned frame; all "structure"
features computed from **visible players only** — never from hidden truth.

*Last-seen memory (8):* `last_x`, `last_y`, `v0_x`, `v0_y`, `speed0 = |v0|`, `tsls`,
`log1p(tsls)`, `dist_last_to_nearest_boundary` (min distance to touchline/goal-line — a player last
seen on the touchline cannot drift outward, this is free geometry).

*Ball (7):* `ball_x`, `ball_y` now; `ball_x_at_last_seen`, `ball_y_at_last_seen`;
`ball_disp_x`, `ball_disp_y` since last seen; `dist_last_to_ball_now`. Rationale: the ball moving
40 m upfield since the player vanished is the single strongest cue that the player moved too.

*Visible-team structure (11):* own-team visible centroid (x, y) now and its displacement since the
player's last-seen frame (2+2); opponent visible centroid (x, y) (2); own-team visible x-spread and
y-spread (2); own defensive-line height (deepest visible outfielder x) (1); `n_visible_teammates`
(1) — a structure estimate from 3 visible teammates deserves less trust than one from 8, and the
model needs that column to learn when to ignore structure.

*Role (5):* role group one-hot collapsed to an ordinal + the fitted slot-prior prediction
(`slot_x`, `slot_y` from `fit_slot_model`), and `off_role_x`, `off_role_y` = how far the player
was from their own slot prior at the moment they were last seen (a full-back caught 20 m upfield
tends to be recovering, not staying).

*Role-anchored centroid vote (3):* `vote_x`, `vote_y`, `vote_n` — the training-free estimator from
https://arxiv.org/html/2607.11548v1: each visible player votes for the **full-team** centroid by
subtracting its own EMA role offset (correcting the bias in the visible-only centroid), then the
hidden player's predicted position is `voted_centroid + that player's EMA role offset`. It is zero
learned parameters, current-match-only, single-CPU-core real-time. We add it **both** as a
standalone pre-registered baseline `B6_vote` (so we can honestly report whether a training-free
method already beats our blend) **and** as three input columns to v1.

*Anchors (2):* `blend_x`, `blend_y` — the anchor's own prediction, so the tree can modulate its
correction by where the anchor put the player.

### 2.2 Fallback (build this FIRST, in week 1)

**Uncertainty-only wrapper on the frozen `B5_blend`, zero learning.** Take the existing baseline,
compute per-bucket empirical residual quantiles on the calibration split, emit 50%/90% regions and
the abstention flag. One afternoon of work, no model.

Cost/benefit, honestly: it **cannot pass gate 1** (it ties the bar, it does not beat it). It **does
pass gates 2 and 4** (calibrated regions, abstention horizon), and it makes gate 3 (downstream bias
reduction) runnable. If everything else slips, this is a shippable, defensible December deliverable
— "we calibrated and gave a reject option to the best causal baseline, which is more than the
published prior art does" is a true statement (Graph Imputer,
https://pmc.ncbi.nlm.nih.gov/articles/PMC9126960/, reports sample diversity but **no** PICP or
coverage numbers; the training-free B4 paper uses a hard 20 s ghost lifetime, not a calibrated
region; SkillCorner, https://skillcorner.com/us/products/football/xy-tracking-data, publishes no
uncertainty for its off-screen extrapolated segments at all).

### 2.3 Stretch (only if v1 plateaus AND M3 is done)

**Small GRU over the last 5 s of visible context + set-pooled teammate encoder, trained with
beta-NLL** (https://github.com/martius-lab/beta-nll, beta=0.5), predicting *velocity* and
integrating to position with a direct-position head ensembled in (the MIDAS derivative-accumulation
trick, https://www.arxiv.org/pdf/2408.10878v3 Table 3 — adding velocity+acceleration as inputs *and*
targets improves monotonically), 5-member deep ensemble for epistemic spread.

Cost/benefit: ~2-3 weeks including debugging, fits in 4 GB, and the honest expected gain over a
tuned GBM on 2 matches is small — MIDAS's own gap over simple baselines is largest in the small-data
soccer setting precisely because it uses the derivative trick, not because it is a transformer.
**Do not start this before the frozen-holdout run of v1 has happened.**

### 2.4 Explicitly rejected, with reasons

| Option | Rejected because |
|---|---|
| Graph Imputer / GVRNN (https://arxiv.org/abs/2106.04219) | Bidirectional — uses the future re-sighting. It is an *oracle-class* method, not a causal one. Trained on 105 EPL matches; we have 2. It also loses to cubic spline on Metrica. |
| NAOMI (https://arxiv.org/pdf/1901.10946) | Non-causal by design, GAN-style training, and MIDAS reports it structurally "cannot handle" the camera-occlusion missingness pattern — our exact pattern. |
| SelectiveNet (https://arxiv.org/pdf/1901.09192) | Three-headed architecture + full retrain to get abstention. Our model is small enough that post-hoc thresholding on a calibrated score is the pragmatic choice. |
| Deep Evidential Regression (https://arxiv.org/abs/1910.02600) | Its selling point for us was a principled epistemic signal for abstention — and that is the part the literature disputes (Meinert et al., https://arxiv.org/abs/2205.10060, call the disentanglement unsound and DER "a heuristic rather than an exact UQ"). Not worth the 4-parameter NIG loss. |
| NGBoost, MAPIE | Both fine libraries, both new dependencies for work that is ~10 lines of numpy on top of an already-installed sklearn estimator. |

---

## 3. Calibration and abstention (pre-registered 2026-07-24, before any v1 number exists)

### 3.1 Producing the 50% / 90% regions

The region is a **2-D ellipse**, not a pair of independent per-axis bars. Per-axis intervals would
give marginal coverage per axis but not the joint coverage the gate asks for.

1. Point estimate `mu = blend + residual_q50`.
2. Per-axis scale from the quantile heads: `w_x = (q95_x - q05_x)/2`, `w_y = (q95_y - q05_y)/2`
   (floored at 0.25 m so the score is finite at 0-1s).
3. Normalized radial nonconformity score on the calibration split:
   `s = sqrt(((y_x - mu_x)/w_x)^2 + ((y_y - mu_y)/w_y)^2)`.
4. **Per-bucket** conformal quantile: `k_b(alpha) = the ceil((n_b+1)(1-alpha))/n_b empirical
   quantile of s within bucket b`, for alpha in {0.50, 0.10}.
5. Region = ellipse centred at `mu` with semi-axes `k_b(alpha) * w_x`, `k_b(alpha) * w_y`.

Per-bucket (not global) calibration is deliberate: residual scale *and* bias differ by horizon, and
a single global correction under-covers long horizons while over-covering short ones
(https://arxiv.org/pdf/2604.13253). We already stratify everything by bucket, so this is free.

### 3.2 Validating coverage

Report the standard two-number scorecard per bucket
(https://arxiv.org/pdf/2508.17761), never one without the other:

- **PICP** — empirical fraction of held-out truths inside the region. Gate 2 target: within
  +/-5 points of nominal (50% and 90%), i.e. PICP in [45, 55] and [85, 95].
- **MPIW** — mean region area expressed as an equivalent radius in metres. Sharpness. A region that
  passes PICP by being 40 m wide has failed.

CIs on both via **1-minute block bootstrap** (same protocol as arXiv 2607.11548), because
consecutive 25 fps frames are not exchangeable — which is also the theoretical wrinkle in split
conformal here. Documented mitigation: per-bucket calibration + block-bootstrapped coverage CIs; if
a bucket misses PICP on the holdout, the **pre-declared** upgrade is Adaptive Conformal Inference
(https://arxiv.org/pdf/2106.00170), one running scalar per bucket, replayed sequentially over the
holdout — declared now so it cannot be presented later as a post-hoc fix.

### 3.3 The abstention rule (no parameter chosen after seeing holdout results)

Two layers, both fixed before the holdout is touched:

**Layer A — skill horizon (bucket level).** Borrowing the weather-forecasting definition of forecast
skill horizon (skill ends where the forecast stops beating the trivial baseline): the abstention
horizon is *derived, not chosen* — it is the first bucket `b*` where v1's calibration-split RMSE is
not significantly below the frozen `B5_blend` bar (95% block-bootstrap CI overlapping the bar).
Buckets at and beyond `b*` are declared no-assert. If v1 beats the bar everywhere, `b*` does not
exist and layer A abstains nowhere — that is a legitimate outcome and we say so.

**Layer B — per-sample reject option (SGR).** Confidence score = the calibrated 90% region's
equivalent radius `r90`. Threshold `R_max` is chosen by binary search on the **calibration split
only** (Selection with Guaranteed Risk, https://arxiv.org/pdf/1705.08500, delta=0.05): the smallest
`R_max` such that selective RMSE on the accepted subset is <= the `B5_blend` bar with 95%
confidence. `R_max` is then **frozen and written into this document before the holdout run** (the
number goes in section 6's milestone log the day it is computed). Nothing about it is re-tuned
afterwards.

**What abstention means downstream.** The Hawk-Eye "umpire's call" pattern
(https://www.itsonlycricket.com/cricket-ball-tracking): inside the uncertainty zone the system does
not assert a coordinate. On abstain we emit the last-seen position **explicitly labelled as
last-seen, not estimated**, with `asserted=False`. Every downstream metric consumer must branch on
that flag — either exclude the player or count them as unknown. A silent fallback to hold-last
would be exactly the kind of quiet failure that costs a retraction.

**Reporting abstention honestly.** Risk-coverage curve per bucket (RMSE on the accepted subset vs
fraction accepted). Report the **curve**, plus AUGRC rather than bare AURC
(https://arxiv.org/pdf/2407.01032 — AURC hides how errors distribute and does not penalise
confident silent failures), and explicitly list the worst high-confidence failures (small `r90`,
large error) per bucket. A confident 25 m miss at 10-30s is the finding that must not be buried.

---

## 4. Training and evaluation protocol

### 4.1 Splits — declared 2026-07-24, frozen

| Split | Data | Used for |
|---|---|---|
| TRAIN | Metrica Game 1 (full) | GBM fit; slot-model fit; EMA role offsets; feature scaling. Baseline `tau=4.75` and blend weights already fitted here — unchanged. |
| CALIB / DEV | Metrica Game 2, **first half** | conformal `k_b`, SGR `R_max`, skill horizon `b*`, early stopping, any hyperparameter |
| HOLDOUT (frozen) | Metrica Game 2, **second half** | gates 1-4, run **once** |
| EXTERNAL (stretch) | Metrica Game 3 | audit format in M2; if it passes, second frozen holdout |

We copy the pre-registration wording of the closest prior work verbatim in spirit
(https://arxiv.org/html/2607.11548v1: "Games 1-2 were used for method and hyperparameter
development; game 3 ... was specified and frozen before game 3 was first evaluated"). Our split is
specified today, before v1 exists.

Half-split honesty: calibration and holdout share a match, so they share teams, players and tactical
context. This is a weaker holdout than a whole unseen match and we will say so in the thesis. It is
the price of a 2-match corpus; Game 3 passing its audit upgrades this and is the single highest-value
data win available.

### 4.2 Leakage risks and the specific guard for each

| Risk | Guard |
|---|---|
| **Structure features computed from hidden truth.** The slot prior and centroid features must use visible players only. `fit_slot_model` trains on truth (fine — that is training), but at inference every structure column must be derivable from the visibility mask. | Unit assertion in the feature builder: recompute every structure column with hidden rows set to NaN and require bit-identical output. This is the one check that, if it fails, invalidates every number. |
| **Future leakage via `next_visible_idx`.** `collect_samples` computes `linear` from the *next* sighting. That column is the oracle and must never enter the feature matrix. | Feature matrix is built by explicit allowlist, never `df.drop(target)`. `linear` lives in a separate eval-only frame. |
| **EMA role offsets updated with hidden players.** The centroid-vote offsets are current-match-only EMAs; they must update only on visible frames. | Same visibility-masked recompute assertion. |
| **Calibration set reused for model selection.** | Hyperparameters and `R_max` both come from CALIB — acceptable (that is what a dev split is for), but then CALIB numbers are **not** reportable as results. Only HOLDOUT numbers go in the thesis tables. |
| **Autocorrelation inflating significance.** 25 fps means ~25 near-identical rows per second. | 1-minute block bootstrap everywhere; never i.i.d. resampling. Also report effective sample size per bucket. |
| **Bucket boundary tuning.** | Buckets are already frozen from M1/M2a (0-1/1-3/3-5/5-10/10-30/30s+). They do not move. |

### 4.3 What gets frozen, when

- **2026-07-24 (today):** splits, buckets, feature allowlist, gate thresholds, abstention rule
  *form*. This document is the record.
- **At end of the CALIB phase (target Sep 5):** GBM hyperparameters, conformal `k_b`, `R_max`, `b*`
  — all written into section 6's log with their computed values before the holdout is opened.
- **Holdout run (target Sep 12):** one run. If we discover a bug afterwards, the fix is disclosed and
  the run is reported as run #2 with the reason, not silently replaced.

---

## 5. How uncertainty reaches the pundit-voice reports

Two tiers, because the audiences tolerate different things. Analyst/coach tier gets numbers; the
fan-facing pundit tier gets one calibrated adjective. (Precedent: Bayes-xG publishes 95% HDIs for
the analytics audience, https://pubmed.ncbi.nlm.nih.gov/38947867/; Second Spectrum collapses a full
predictive distribution to a single familiar-looking number for broadcast,
https://www.nbastuffer.com/analytics101/quantified-shot-quality-qsq/.)

### 5.1 Sentence templates (pundit tier)

Never emit a bare percentage. Probability-of-precipitation research shows comprehension only
improves when the complementary outcome is stated explicitly
(https://journals.ametsoc.org/view/journals/bams/90/2/2008bams2509_1.xml), so every confidence
statement carries its complement or a plain-language qualifier.

- **Short horizon (asserted, tight):** "Dalot was last on camera 2 seconds ago pressing the near
  touchline — we can place him within about 3 metres of there."
- **Mid horizon (asserted, wide):** "Casemiro has been off camera for 8 seconds. Our best read has
  him around the centre circle, and 9 times out of 10 he'd be within roughly 12 metres of that spot
  — but a 1-in-10 chance says he's somewhere else in that half."
- **Long horizon (abstained):** "We last saw Shaw 34 seconds ago on the far side. That's too long
  for us to call his position — we're not going to guess. Treat the far-side numbers in this section
  as covering 10 outfield players, not 11."
- **Aggregate caveat line (every report that uses imputed positions):** "N of the 22 positions in
  this graphic are estimates rather than sightings, and M players were too long off camera to
  estimate at all."
- **Banned phrasings:** "the player is at X" for any imputed position; a bare "68% confident"; any
  number quoted to more than 1 decimal place; describing the region as where the player "can" be.

### 5.2 The visual convention: graded quantile dotplot, never a hard-edged blob

One convention, used everywhere:

- **Dots, not a density cloud.** Scatter 20 dots sampled from the calibrated region, each dot = 5%
  of the probability. Non-experts read frequency framing more accurately than density framing
  (quantile dotplots beat density curves and ribbons for decision quality,
  https://www.mjskay.com/papers/chi2018-uncertain-bus-decisions.pdf). Pure matplotlib scatter, no
  new dependency.
- **No solid boundary, ever.** A hard-edged polygon triggers the documented containment / boundary
  misconceptions from hurricane-cone research (https://escholarship.org/uc/item/96s9t7g5): viewers
  read the edge as a guarantee and the width as the object's size, not the estimate's uncertainty.
  This is the same false-confidence failure that ordinary pitch-control graphics already have
  (https://iopscience.iop.org/article/10.1088/2632-072X/acb67d) — hard team-coloured territories over
  an inherently probabilistic model. We will not repeat it. Plain error bars are out for the same
  reason (read as min/max containers).
- **Fade with horizon.** Dot opacity decreases with time-since-last-seen — tight and dark at 0-1s,
  pale and scattered at 10-30s. This is the Bank of England fan-chart grammar
  (https://www.bankofengland.co.uk/working-paper/2026/anchors-aweigh-the-effect-of-communicating-forecast-uncertainty),
  already validated for exactly the "uncertainty grows with horizon" shape our RMSE curve has.
- **Abstained players are a separate layer,** drawn as a hollow marker at the last-seen point with a
  dashed leader line and the label "last seen 34 s ago — position not estimated". Following the NHC
  cone redesign (https://noaanhc.wordpress.com/2024/08/12/nhc-cone-heads-in-a-new-direction/), the
  "we can't say" layer is a distinct labelled layer, not a wider version of the same blob.
- **Legend text is mandatory and fixed:** "Dots show where our model thinks the player is, not the
  area they cover. 20 dots = the 90% region; 1 dot missing the truth in 10 is expected."

Reuse note: marker *size* is already spoken for by the xG-shot-map convention our audience knows
(size = probability/danger). We encode uncertainty with dot *spread and opacity* only, so a big
scattered cloud never reads as "big important player".

---

## 6. Milestones and risk table

Today 2026-07-24. BTP review December 2026. These slot inside the existing M2/M3/M4 in
`B4_IMPUTATION_PLAN.md` — they do not replace them.

| Window | Deliverable | Done-when |
|---|---|---|
| Jul 25 - Aug 1 | **Fallback wrapper first.** Per-bucket empirical regions + abstention flag on frozen `B5_blend`. Splits frozen in this doc. | PICP/MPIW table exists for the baseline; gate 2 answerable without a model |
| Aug 2 - Aug 16 | Feature builder (section 2.1 allowlist) + visibility-masked leakage assertion + `B6_vote` role-anchored centroid baseline scored as a standalone baseline | `B6_vote` row added to the baseline table; leakage assertion green |
| Aug 17 - Sep 5 | v1 GBM quantile residual model; train on Game 1, all tuning on Game 2 H1; conformal `k_b`, SGR `R_max`, skill horizon `b*` computed and **written into this file** | Frozen-params block appended below with actual numbers |
| Sep 6 - Sep 12 | **Frozen holdout run — once.** Gates 1-4 scored. Risk-coverage curves + AUGRC + silent-failure list | Holdout table in `results/` |
| Sep 13 - Sep 15 | M2 gate report to supervisor (pass or honest negative) | Existing M2 deadline met |
| Sep 16 - Oct 31 | M3: transfer onto our real ManU tracks; dotplot renderer; pundit templates wired into report generator | Imputed + abstained players visible in a real scouting report |
| Nov 1 - Nov 30 | M4: gate 3 downstream bias reduction (line-height 16.4 m bias, possession undercount); negative results written up; thesis section | Thesis section drafted |
| Dec 1 | Code freeze. Only slides and prose after this. | — |

Stretch (GRU + beta-NLL) has **no window**. It is only started if v1 clears the holdout before Sep 12
and M3 is on schedule — otherwise it never happens, by design.

### What could go wrong

| Risk | Likelihood | Fallback |
|---|---|---|
| v1 fails to beat the bar at 10-30s / 30s+ | Medium-high — this is the hard regime everyone reports | Report it as the headline honest negative, and let gate 4 carry it: abstain past `b*`. "Calibrated regions + a principled reject option on the best causal baseline" is still a defensible novelty, since neither Graph Imputer nor the training-free B4 paper nor SkillCorner does calibrated abstention. |
| v1 loses to the training-free `B6_vote` baseline | Medium | Excellent outcome, report it loudly. Then `B6_vote` becomes the anchor for the residual model instead of `B5_blend` — the architecture is unchanged, only the anchor column swaps. Re-freeze and re-run once. |
| Coverage misses on the holdout (PICP outside +/-5) | Medium | Pre-declared (section 3.2): Adaptive Conformal Inference replay, one scalar per bucket. Declared before the run, so it is not a post-hoc rescue. |
| Split conformal invalid because frames are not exchangeable | Certain in theory, unclear in magnitude | Block bootstrap quantifies it; ACI removes the exchangeability assumption entirely if it bites. |
| Metrica Game 3 fails its format audit | Medium | We stay on the Game-2-half-split holdout and state the weaker-holdout caveat explicitly. No number changes. |
| Half-split holdout criticised as too weak at review | Medium | Pre-empt it: state the limitation ourselves in the thesis before anyone asks, and show the Game-1-fit-to-Game-2 baseline transfer (already measured, zero degradation) as evidence the pipeline does not overfit. |
| Downstream gate 3 shows no bias reduction | Medium | Report the negative. It is a real finding: it would mean the residual bias is not driven by off-screen players, which changes what B5 should attack. |
| Disk / time pressure from the parallel GPU chain | Low-medium | Everything here is CPU-only and small (10 sklearn models, <2 GB data). It is the one workstream that can run while the GPU is busy. |
| Name collision with arXiv 2607.11548's "B4" | Certain (cosmetic) | Rename ours in prose to "B4 module" and theirs to "role-anchored voting (RAV)". Flagged to the orchestrator: confirm this is genuine outside work and not our own output being indexed. |

---

## 7. Honesty trail — research claims that did not survive checking

Kept visible rather than dropped, per the validated-or-nothing rule.

**Refuted outright:**

1. *"A 2025 Velocity Completion paper benchmarks against NAOMI and Capellera-2024 as its main deep
   baselines."* **Wrong.** The paper exists (https://arxiv.org/pdf/2505.16199, Umemoto & Fujii 2025)
   but NAOMI and Capellera are cited in related work only. Its sole external baseline is a rule-based
   velocity method; everything else compared is the authors' own MLP/VAE/GNN/RNN variants, scored in
   RMSE m/s. More importantly its task is the *inverse* of ours — all 22 positions are known and only
   velocities are missing — so it cannot calibrate anything in our metre-scale position problem. Its
   only residual value is a citation trail: Everett et al. 2023 and Penn et al. 2023 are the closest
   position-completion works it names. Downgrade from "with_adaptation" to "not relevant".

2. *"A NeurIPS 2023 companion paper shows beta-NLL under-covers in the tails and proposes post-hoc
   variance recalibration."* **Fabricated mechanism.** Immer et al. 2023 exists and does beat beta-NLL
   with a single network, but it never analyses tail coverage, proposes no post-hoc recalibration, and
   its method is the natural *parametrization* of the Gaussian plus a Laplace approximation and
   empirical Bayes — "natural gradient" appears only in its bibliography. It reports test
   log-likelihood, not interval coverage. **Consequence for us:** it is not a coverage-calibration
   lever, so it is not the fallback for an under-covering long-horizon bucket. Conformal (section 3.2)
   is, and that is why conformal is the primary mechanism in this plan rather than a nicer likelihood.

**Corrected while otherwise standing:**

3. *beta-NLL gradient direction was stated backwards* in our notes: beta-NLL **up**-weights
   high-variance (hard) points — plain NLL's 1/variance factor starves them, and `var^beta` partially
   cancels it, interpolating toward MSE at beta=1. The formula we recorded is correct; the
   explanation was not. Also: beta>0 is not a proper scoring rule, so a beta-NLL model would still
   need its own calibration step — another reason the stretch option does not remove the conformal
   layer.

4. *Deep Evidential Regression's "unreasonable effectiveness" follow-up is a critique, not a
   validation.* Meinert et al. (https://arxiv.org/abs/2205.10060, AAAI 2023) show the
   aleatoric/epistemic disentanglement is mathematically unsound (nu under-constrained by the NLL;
   the "epistemic" output largely tracks scaled aleatoric uncertainty) and call DER "a heuristic
   rather than an exact uncertainty quantification". The ensemble-competitive OOD numbers come from
   the 2020 paper, not the follow-up. Since the epistemic head was DER's entire appeal for our
   abstention gate, DER is rejected (section 2.4).

5. *Agent Imputer's baseline ordering was stated wrong* on our side: XGBoost (9.26 m) is the **worst**
   ML baseline, not the GNN (8.32 m). The "structure alone is weaker than temporal alone" replication
   still holds (GNN 8.32 < LSTM 7.09 in quality terms), but "GNN worst of three" does not. Bigger
   catch: Agent Imputer is **bidirectional interpolation** — its features include `nextAgentTime/X/Y`
   and its LSTM window spans t-2..t+2 — so *all* its numbers, XGBoost's included, live in our oracle
   regime, not our causal regime. A causal-features-only XGBoost is untested by that paper. This is
   why section 2.1 cites it for the *tabular-model-is-competitive* point only, and not as a number to
   beat.

6. *MIDAS's errors are in metres, not normalized units* (Fig 4 annotates PEs as e.g. "2.4728m"),
   contrary to our note — though Table 2 never states units. And MIDAS is an **offline bidirectional**
   imputer whose backward derivative accumulation anchors on the future re-sighting, so its 1.23 m
   compares to our 8.66 m oracle ceiling, **not** our 12.63 m causal bar. Its gaps are also bounded
   inside 20 s windows with the first/last 5 frames always observed, so it has no 10-30s or 30s+
   analogue at all. Only the derivative-accumulation trick transfers.

7. *arXiv 2607.11548 uses 3 Metrica matches and reports MEDIAN position error*; we use 2 matches and
   report RMSE. Their 15.6-16.9 m median at >9.6 s must not be compared to our 17.27 m RMSE at
   10-30 s. If we want a real comparison we recompute both statistics on a common split — that is a
   named task inside the Aug 2-16 window when `B6_vote` is implemented.

8. *SoccerNet GSR novelty framing needs scoping.* GSR pipelines genuinely do not impute off-screen
   players — the task definition itself (https://arxiv.org/abs/2404.11335) limits evaluation to
   players in the camera's field of view, so no entrant could address it. But off-screen imputation
   *is* addressed elsewhere in the literature (Graph Imputer, MIDAS, RAV). Our novelty claim must
   therefore be scoped to "causal, calibrated, abstaining, from a broadcast pipeline" — not
   "nobody has done off-screen imputation".

9. *Quantile-dotplot precision figure unverified.* The "~1.15x lower variance than density plots"
   number came from a secondary summary; the primary PDF blocked text extraction. We use the
   qualitative finding (frequency framing beats density framing) and will not quote the number.
