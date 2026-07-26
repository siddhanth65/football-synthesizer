# Paper read: metric battery for RQ-D

Source PDFs in `docs/papers/`. Read in full (Bauer & Anzer, Bialkowski) / skimmed (Herold).
Page numbers below are the **printed journal page numbers**, with the PDF page index in brackets
where they differ.

- **[BA]** Bauer P, Anzer G (2021). "Data-driven detection of counterpressing in professional
  football." *Data Mining and Knowledge Discovery* 35:2009-2049. DOI 10.1007/s10618-021-00763-7.
- **[BI]** Bialkowski A, Lucey P, Carr P, Yue Y, Sridharan S, Matthews I (2014). "Large-Scale
  Analysis of Soccer Matches using Spatiotemporal Tracking Data." *ICDM 2014*, pp. 725-730.
- **[HE]** Herold M, Goes F, Nopp S, Bauer P, Thompson C, Meyer T (2019). "Machine learning in
  men's professional football." *Int J Sports Sci Coaching* 14(6):798-817.

Everything numeric in Section 4 is either quoted from the paper or computed from a table in the
paper; computed values are labelled `[derived]` and the input table is named.

---

## 1. Bauer & Anzer: the counterpressing operational definition

### 1.1 The definition chain (BA Sect. 2.1.2, pp. 2013-2014)

The paper does *not* define counterpressing as a rule. It defines a **situation class** plus a
**human label**, then learns the label. Four nested definitions:

**(a) Ball possession** (p. 2013). "Ball possession for one team [...] from that time point a
player of that team touches the ball with ball control, until the ball is out of play, or an
opponent player touches the ball with ball control." *Ball control* = "the ability to conduct a
contrived action with the ball." A pass between two players of one team keeps possession with that
team "as long as no opposing player intercepted that pass or won the ball within an individual
duel." On interception, "the ball possession change is detected exactly at the time of the first
ball touch of the intercepting player."

**(b) Defensive transition phase** (p. 2014). "the time-window when a team loses full ball
possession, but is not yet into their ideal defensive formation."

**(c) Counterpressing trigger** (p. 2014, the operative sentence):

> "a team conducts *counterpressing* if at least one player exerts (spatio and/or temporal)
> pressure on the ball carrier, or on the opponents close to the ball."

Note explicitly what this is *not*: the authors reject the radius rule as the definition. They cite
StatsBomb's pressing definition ("a defensive player being within a five-yard radius of the
ball-carrying opponent", p. 2014) and Andrienko's continuous pressure score, and use **both only as
rule-based baselines to beat** (Table 3, rows 5-6). Their stated reason (p. 2014): "according to the
match-analysts involved in this project, being close to the player in ball possession is not the
only way to exert pressure. Attacking or blocking the easiest pass options could, for instance, also
be seen as applying pressure."

**(d) Success** (p. 2014):

> "we consider it as successful if the ball is regained within **five seconds** and shots and goals,
> scored or received, are accredited to the previous counterpressing phase if they occur within the
> following **20 seconds**."

Both thresholds were *chosen by expert review of video*, not optimised (p. 2020): the authors
"queried relevant video scenes with possession regains after 3, 4, 5, 6, 7 and 8 seconds and
discussed them with a group of professional match-analysts. The same procedure was conducted to
investigate the follow-up goal-scoring opportunities. Here scenes with shots 10, 15, 17, 20 and 25
seconds after the initial ball loss were discussed." The 5 s figure aligns with Guardiola's "five
second rule" (p. 2010, fn 1).

### 1.2 Inclusion / exclusion criteria (BA Fig. 1, p. 2015)

1. All ball possessions starting with a **set-piece are excluded**.
2. All ball losses in the team's **own half are excluded**.
3. Scenarios **end when the ball goes out of play**; possession is deemed regained at the moment the
   ball goes out of play by the team that will take the set-piece.
4. Only **effective (net) playing time** counts; phases ending at half-time/full-time/referee ball
   are dropped (p. 2014).
5. Out-of-bounds touches not flagged as a possession change (deflected shots), and situations where
   the individual-possession model disagrees with the team possession flag, are excluded (p. 2014).

Effect of the criteria: 20,928 tagged defensive transitions over 97 matches reduce to **11,108
"relevant"** ones (Table 7, p. 2034) -- i.e. **47% of raw turnovers are discarded before modelling**.

### 1.3 The time window: 2 seconds, not 5 (BA Sect. 2.2.2, p. 2016) -- KEY FOR US

> "The features describe the location of a ball possession change, several relevant factors
> describing both teams' exact positioning at the time of turnover and their movements in the
> **first two seconds** immediately after the ball loss. A time-window longer than two seconds was
> problematic, because it would cut off too many situations where the ball possession changed within
> that time."

Every feature is evaluated at **BPC + {0 s, 1 s, 2 s}** (Table 1, pp. 2017-2018). The 5 s window
appears **only in the success label** (did the ball come back), and the 20 s window **only in the
outcome label** (shot/goal). **The detector never needs 5 s of continuous observation.** This
directly contradicts the window our own W1 census priced.

### 1.4 The 134 features (BA Table 1, pp. 2017-2018)

Grouped by what they demand of the data:

| Group | Features | Spatial extent |
|---|---|---|
| Ball | turnover position (x,y); **ball height (z)** at 0/1/2 s | ball only |
| Nearest-player | distance of closest player to ball (both teams); speed of closest player (both teams) | local |
| Local density | number of players within 10 m / 20 m / 30 m circles of the ball (both teams); players closer to ball than the nearest opponent | local |
| Local shape | **local stretch index** of the 3, 4 and 5 players closest to the ball (Santos et al. 2018 normalisation), both teams | local |
| Global shape | team centre (x,y, GK excluded); covered area (widest player-to-player distance in x and y); global stretch index (Bourbousson & Sève 2010); **effective playing space** = convex hull of all outfield players | whole team |
| Ball-relative counts | players in front of / behind the ball (relative to own goal centre), and the compactness (stretch index) of each of those two sub-groups | whole team |
| Pressure | Andrienko (2017) normalised pressure on the ball, and on the ball-possessing player, exerted by the regaining team | local |
| Speed | average speed of each team at 0/1/2 s | whole team |
| Possession history | duration of previous ball possession phase (out-of-play sequences excluded); **individual ball possession time**, absolute and relative (Link & Hoernig 2017) | ball + identity |

### 1.5 Labelling procedure (BA Sect. 2.2.1, p. 2015; Appendix A, p. 2034)

- Trained **student analysts** with a football-tactics background, not the expert analysts.
- **97 matches** = the first 11 matchdays of Bundesliga 2018/19, **home team perspective only**.
- 20,928 defensive transitions tagged -> 11,108 pass the inclusion criteria -> **3,196 labelled
  counterpressing (28.77%)**, 970 fallback, **6,942 (62.50%) labelled "undefined"** ("uncontrolled
  transition situations", e.g. very short possessions, headers after a corner).
- Fallback = "a defensive transition phase, where all players' intention is to either react
  inactively, or move backwards to their defensive line-up without exerting pressure on the ball."
  More than **95% of fallback situations start with a goalkeeper catching the ball**, so fallback was
  merged into "not counterpressing" for modelling (p. 2034-2035).
- **Inter-labeller reliability: 20 matches labelled by 3 students; pairwise accuracy 82.01%**
  (p. 2015). The authors call this "the basic limitation to achieve further accuracy" (p. 2030).
- Transition durations (p. 2015, p. 2035): all transitions **9.34 s**; counterpressing **9.89 s**;
  non-counterpressing **9.11 s**; fallback **18.30 s**; undefined **7.83 s**.

### 1.6 Model and performance (BA Sects. 2.3, 3.1; Table 3, p. 2021)

XGBoost (Chen & Guestrin 2016), 75/25 split taking 25% of transitions **from every match** so no
match/team/result is over-represented, 5-fold CV on train, Bayesian TPE hyperparameter search,
`nrounds` capped at 400 (Table 2, p. 2019).

| # | Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 1 | **XGBoost (Model 1)** | 0.72 | 0.63 | 0.67 | **0.874** |
| 2 | Logistic regression | 0.69 | 0.51 | 0.59 | 0.841 |
| 3 | Random forest | 0.74 | 0.55 | 0.63 | 0.867 |
| 4 | XGBoost + class balancer (Model 2) | 0.60 | **0.80** | **0.69** | 0.865 |
| 5 | Naive rule-based (5-yard radius) | 0.31 | 0.87 | 0.46 | 0.602 |
| 6 | Andrienko pressure > 0.74 | 0.37 | 0.37 | 0.37 | 0.568 |

Read row 5 carefully: **the StatsBomb-style radius rule labels 72.41% of all turnovers as
counterpressing** (p. 2021) -- precision 0.31, AUC 0.602, i.e. barely above chance. Any battery of
ours that implements "defender within 5 yards" and calls it counterpressing is implementing a
detector with AUC 0.60.

Yield: "Per team and per match, we detect around **20 to 30 counterpressing situations**, out of
around **90 to 200 transition situations**" (p. 2021).

Top SHAP features (Fig. 2, p. 2022), in order: `IndividualBallPossession(abs)`,
`SpeedRegainingTeamPlayer(2s)`, `IndividualBallPossession(rel)`, `OpposingTeamLocal5_stretch(2s)`,
`DistanceRegainingTeam(2s)`, `SpeedRegainingTeam(1s)`, `OpposingTeamsCenter(2s)`,
`PlayersOpposingTeamBehindBall(1s)`, `PlayersRegainingTeamCloseThenOpposing(2s)`,
`RegainingTeamCenter(2s)`. Documented non-linearity: **>= 4 opposing players behind the ball at
+1 s** flips the feature's contribution positive ("The number of four defenders -- almost half of the
team -- seems to be a decisive threshold", p. 2022).

Expert validation on two national-team matches (Table 6, p. 2029): of 25/24/33/25 detections, the
analysts manually excluded 6/3/5/2 and added 0/2/0/1. **Ten of the eleven manual exclusions were
situations with only one player exerting pressure** -- i.e. the "at least one player" clause is the
main source of expert-model disagreement.

### 1.7 Frame-by-frame data requirement, and can it be made gap-tolerant?

**What the paper needs, literally:** 25 Hz optical tracking of all 22 players + ball with **ball
z-coordinate**, plus a live human operator flagging ball possession and ball status **every frame**
(p. 2013), plus manually annotated event data synchronised to the tracking (p. 2013), plus player
identity for individual ball possession time.

**Gap tolerance, decomposed.** This is the good news and it is specific:

| Requirement | Continuous? | Gap-tolerant redefinition |
|---|---|---|
| Feature extraction | **No** -- 3 instants (0/1/2 s) | Needs 3 observed slots inside 2 s, not 26 slots over 5 s. Slot-level, so a censored/observed indicator per slot is natural. |
| Success label (regain within 5 s) | **No** -- 2 instants | Needs "who has the ball at t0" and "who has it at t0+5 s". Unobserved regain time is a **right-censored** observation -> Kaplan-Meier / interval-censored survival, not a dropped window. |
| Outcome label (shot/goal in 20 s) | No | Event-level; needs a shot detector with team attribution, not tracking. |
| Local features (10/20/30 m counts, local-5 stretch, distance/speed of closest player, Andrienko pressure) | Instantaneous | **Broadcast-favourable**: they only require players *near the ball*, which is exactly where the camera points. Missingness is close to MCAR within the frame. |
| Global features (team centre, covered area, global stretch, convex hull, players in front/behind ball, team mean speed) | Instantaneous | **Broadcast-hostile**: every one is a statistic over the *whole* team. With k of 10 outfielders visible, convex hull and covered area are biased **downwards** and team centre is biased **towards the ball**. These are the imputation-dependent ones. |
| Ball height z | Instantaneous | **Not recoverable** from monocular broadcast without a dedicated model. Drop the feature. |
| Individual ball possession time | Continuous over ~3 s | Needs ball + a stable player association for 3 s. Our post-`link_ball` track supports this only sometimes; it is the **top SHAP feature**, so its loss is not cosmetic. |

So: **a gap-tolerant counterpressing detector is definable**, but it is not the same model. It would
be the BA model restricted to the local + ball-relative feature families, over a 2 s window, with a
censored success label. That is a defensible derivative, and it is close to what the W1 census's
"median over slots" variant was groping towards -- except the window should be **2 s, not 5 s**.

### 1.8 Between-team variation reported by BA

See Section 4 for the full harvest. Headline statements from the text:

- "The percentage of counterpressings detected per transition differs significantly per team.
  Borussia Moenchengladbach presented the highest value (**40.07%**), whereas only **21.80%** of
  Hannover 96's transitions have been labeled as counterpressing" (p. 2015, manual labels).
- "teams within the German Bundesliga follow appreciably different transition strategies (RQ4)"
  (p. 2032) and "The experts' expectations of which coaches use counterpressing more often and/or
  more efficiently were underpinned by the results" (p. 2026).
- Counter-evidence on time trends: "No consistent tendency over the considered seasons is
  detectable: games do not get more intensive in terms of total in-play transitions per match, nor
  are there significant changes in teams average counterpressing behaviour" (p. 2025).

---

## 2. Bialkowski: role-aligned representation

### 2.1 Why marginal position distributions are inadequate (BI Sect. I, Figs. 1-2, pp. 1-2)

Three distinct failures, all stated in the paper:

1. **The mean is a position no one occupied.** Fig. 1(a): a player who starts left-wing and switches
   to right-wing at half time. "Current approaches just give the mean position which neglects the
   context" -- the mean lands mid-pitch, describing neither behaviour. The role representation
   (Fig. 1(b)) recovers two coherent modes.
2. **Identity-indexed marginals overlap almost completely.** Fig. 2(a)/(b): over a match, per-player
   2D Gaussians "continuously change position throughout a match causing heavy overlap in their
   spatial probability density functions." A representation whose components overlap is a
   representation with little discriminative power.
3. **The static assumption breaks structurally**: "constant interchanging of player positions [...]
   making the dimensionality of the resulting subspace much higher. Additionally, this assumption
   breaks down when there is a substitution, an expulsion of a player [...] or when comparing
   different teams (i.e., different identities)" (p. 2).

Note (3) is decisive for *us* independently of the paper's argument: broadcast re-ID gives us weak
and intermittent identity, so an **identity-indexed representation is not merely suboptimal, it is
not constructible**. Role-based is the only option available to us, which is a reason to invest in
it rather than a cost.

### 2.2 The method (BI Sects. I.A, III, pp. 2-3)

**Definition 1.1** (p. 2): "A formation F is an arbitrarily ordered set of N roles {R1, R2, ..., RN}
which describes the spatial arrangement of N players." A role is "a space or area that each player
is assigned responsibility for relative to the other teammates." Formation is **shift-invariant**
and allows non-rigid deformation. Each role is unique at any instant; players swap roles over time
but hold exactly one role per frame.

**Representation** (Fig. 3, p. 2): `r_t = P_t x_t`, where `x_t` is the identity-ordered position
vector and `P_t` is a **permutation matrix, re-estimated every frame**.

**Objective** (Sect. III.A, p. 3). Model the team heat map as a mixture over roles,
`P(x) = (1/N) sum_n P_n(x)` (eq. 1-2). Roles should overlap minimally, so penalise negative KL
divergence between each role's density and the team's, `V_n = -KL(P_n(x) || P(x))` (eq. 3), maximise
`V` (eq. 4). Written in entropy terms this collapses to

> `F* = argmin_F sum_{n=1..N} H(x|n)`  (eq. 9, p. 3)

i.e. **minimum entropy data partitioning** (Roberts et al.; Lee & Choi).

**Solver** (Sect. III.B, p. 3): EM, "similar to k-means clustering. However, instead of assigning
each data point to its closest cluster, we solve a **linear assignment problem** between identities
and roles using the **Hungarian algorithm**."
Loop: (i) arbitrarily assign each player a fixed role label, initialise each role's PDF from that
player's data; (ii) for each frame, build a cost matrix from the **log probability of each position
under each role label** and solve the assignment; (iii) recompute role PDFs; iterate to convergence.
Critically: "We **normalize the tracking data in each frame to have zero mean** in order to negate
the effects of translation."

**Formation detection** (Sect. IV.B, p. 4): run the above per team per match-half; each formation =
a set of **ten** role PDFs (GK excluded). Cluster the resulting formations agglomeratively using
**Earth Mover's Distance** (eq. 10), with the ground distance between two formations set to the sum
of the role-to-role distances between corresponding roles.

**Visualisation** (Sect. IV.C, p. 5): role PDFs recomputed over a **5-minute sliding window** give a
film-strip of formation evolution (Fig. 7). Roles in Fig. 9: 1=LB, 2=LCB, 3=RCB, 4=RB, 5=LCM, 6=RCM,
7=LW, 8=RW, 9=LF, 10=RF.

### 2.3 Numbers (BI Table I p. 4, Figs. 5-6 pp. 4-5)

| Quantity | Value | Source |
|---|---|---|
| Teams | 20 | Table I |
| Games | 375 (of 380; 5 omitted for data errors) | Table I, Sect. IV.A |
| Data points | 480 M | Table I |
| Ball events | 981 K, 43 event types | Table I, Sect. IV.A |
| Sampling rate | 10 fps (Prozone) | Sect. IV.A |
| Formations detected | **1411** (per team per half, send-off halves excluded) | Sect. IV.B |
| Formation clusters | 6 | Fig. 5 |
| Cluster frequencies | 50.37% (4-4-2), 20.76% (4-2-3-1), 12.52% (C4), 5.40% (C3), 3.15% (C6), 0.90% (C5) | Fig. 5 |
| **Overall correct classification vs expert formation labels** | **75.33%** | Sect. IV.B, p. 5 |
| Per-cluster diagonal (Fig. 6 heat map, read-off, %) | ~84 (C1 / 4-2-3-1), ~83 (C2 / 4-4-2), ~80 (C3 / 3-4-3), ~67 (C4 / 4-3-3), ~73 (C5 / 4-1-4-1), ~48 (C6 / other) | Fig. 6 |

Main confusion: cluster 6 (a 4-4-2 "diamond") splits into 4-4-2 and 4-3-3; "sometimes the formation
appears in between two clusters, e.g. there is some confusion between the 4-4-2 and 4-2-3-1
formations when the second striker is positioned slightly behind the other" (p. 5).

**There are no between-team pressing, transition, or discriminability effect sizes in this paper.**
It is a representation paper; the only quantitative team-comparison output is the formation
clustering above. Anyone quoting BI as a source of between-team SD is quoting something that is not
there.

### 2.4 Is role assignment computable when only ~half the players are visible?

Split into three claims, because they have different answers.

**(a) The per-frame assignment: YES, mechanically.** The Hungarian algorithm solves rectangular
assignment (k visible players to N=10 roles) at the same cost; unmatched roles are simply left
unassigned in that frame. Nothing in eq. 9 requires k = N at a given frame. This is the good news
and it is why role alignment is worth attempting on broadcast at all.

**(b) Learning the role PDFs from broadcast data: NO, not without correction.** The EM step
re-estimates `P_n(x)` from the assigned positions. Broadcast visibility is **strongly correlated
with pitch location** -- the camera follows the ball, so the far-side full-back and the far winger
are missing far more often than the players near the ball. That is missing-not-at-random on the very
variable being estimated. The learned role PDFs will be pulled towards the ball and their variances
will be under-estimated, systematically, and no amount of data fixes it. This is where calibrated
imputation is genuinely load-bearing rather than cosmetic.

**(c) The per-frame zero-mean normalisation: BROKEN, and it is a silent bug.** The paper normalises
each frame to zero mean "to negate the effects of translation" (Sect. III.B, p. 3). On broadcast,
the mean of the *visible subset* is not the team centroid; it is biased toward the ball by exactly
the amount the camera is biased. Applying the paper's normalisation naively to visible-only
positions injects a ball-correlated translation into every frame -- which would then show up as
"tactical" structure. Any implementation must either (i) impute first and then normalise, or
(ii) normalise against a pitch-anchored reference (e.g. the defensive line or the goal-to-goal axis)
rather than the visible centroid, and say which. Same issue applies to BA's `team centre` feature.

---

## 3. Computability on our broadcast pipeline

Legend: **AS-IS** = computable now; **GAP** = computable under a gap-tolerant redefinition;
**IMPUTE** = needs the off-screen imputation model to be unbiased; **NO** = not computable.

| # | Metric (source) | Verdict | Justification |
|---|---|---|---|
| 1 | Turnover / ball-possession-change location (BA Table 1) | **GAP** | `detect_turnovers` fires 2546 times over 12 matches = 212/match against Sofascore ballRecovery 105/match (**2.03x**, per-match 0.96-2.75x, `W1_FEASIBILITY_CHECKS.md`). BA's own T/M is 115.6. Counts are unusable; a *rate* construct after per-match calibration is usable. Note BA discard 47% of raw turnovers by rule (set-piece, own half) -- applying their exclusions would cut our over-count substantially and should be tried before blaming the detector. |
| 2 | Defensive reaction time (Vogelbein; BA p. 2014) | **GAP** | Needs possession identity at loss and at regain, not the trajectory between. Unobserved regains are **right-censored**, not missing: use Kaplan-Meier over the 2546 turnovers instead of discarding the 65% with broken ball track. This converts our worst funnel loss into an estimator assumption. |
| 3 | Counterpress **success** = regain within 5 s (BA p. 2014) | **GAP** | Two instants (t0, t0+5 s), not 26 slots. Same censoring argument as #2. |
| 4 | Counterpress **outcome** = shot/goal balance in 20 s (BA p. 2014, Table 4) | **GAP, unvalidated** | Needs a shot detector with **team attribution**. `Shots on/off target` exist in the E2E-Spot 17-class space but were never validated against truth in W1 (only Foul 1.069x and Clearance 0.265x were). Team attribution is unsolved for every spotted event. Must be measured before use. |
| 5 | Ball height z at 0/1/2 s (BA Table 1) | **NO** | Monocular broadcast, no ball-height model. Drop the feature; BA's SHAP does not rank it in the top 10, so the loss is tolerable. |
| 6 | Distance / speed of closest player to ball, both teams (BA Table 1) | **AS-IS** | Local to the ball = inside the camera frame by construction. `SpeedRegainingTeamPlayer(2s)` and `DistanceRegainingTeam(2s)` are SHAP ranks 2 and 5. This is the strongest broadcast-native family in the paper. |
| 7 | Players within 10/20/30 m of the ball, per team (BA Table 1) | **AS-IS** at 10 m, **GAP** at 20-30 m | The 10 m circle is inside the frame. The 30 m circle exceeds typical broadcast coverage, so its count is censored downward; treat as "count among visible" plus a visibility covariate, or impute. |
| 8 | Numerical superiority within 10 m of ball (BA RQ3, p. 2025) | **AS-IS** | Purely local. BA report the effect: superiority -> regain within 5 s **36.2%** vs **30.2%** otherwise, over 109,852 detections. A small effect (6 pp) but on a broadcast-friendly quantity. |
| 9 | Local stretch index of the 3/4/5 players closest to the ball (Santos et al.; BA Table 1) | **AS-IS** | `OpposingTeamLocal5_stretch(2s)` is SHAP rank 4 and it needs only the five nearest players. Best signal-to-visibility ratio in the whole BA feature set. |
| 10 | Andrienko normalised pressure on ball / ball-carrier (BA Table 1) | **AS-IS** | Local. Caveat: as a *standalone detector* it scores AUC 0.568 (Table 3 row 6). Use as a feature, never as the metric. |
| 11 | Team centre (x,y), GK excluded (BA Table 1) | **IMPUTE** | Statistic over the whole team; visible-subset mean is biased toward the ball. SHAP ranks 7 and 10. See Sect. 2.4(c) -- this is the same bug as BI's frame normalisation. |
| 12 | Covered area (widest x/y span) and global stretch index (BA Table 1) | **IMPUTE** | Extremal statistics: a single missing wide player collapses the estimate. Biased **downwards**, monotonically, with visibility. Worst-behaved family under censoring. |
| 13 | Effective playing space = convex hull of outfielders (Santos et al.; BA Table 1) | **IMPUTE** | Same as #12 and worse: hull area is a function of exactly the peripheral players broadcast omits. |
| 14 | Players in front of / behind the ball, and their compactness (BA Table 1) | **IMPUTE** | Whole-team counts. `PlayersOpposingTeamBehindBall(1s)` is SHAP rank 8 with a decision threshold at **>= 4 players**; an undercount of 2 flips the classification. Redefining as a *fraction of visible* changes the threshold's meaning and must be re-fit, not re-labelled. |
| 15 | Team average speed (BA Table 1) | **IMPUTE** | Whole-team mean over a ball-biased sample; players near the ball move differently from those away from it, so the visible-only mean is biased, not just noisy. |
| 16 | Individual ball possession time, abs and rel (Link & Hoernig; BA Table 1) | **GAP, hard** | **SHAP rank 1 and 3.** Needs ball plus a stable ball-player association over ~3 s. Post-`link_ball` we hold the ball through a full 5 s window in 886/2546 = 34.8% of cases; over 3 s the yield will be higher but has not been measured. Measure it before designing around it. |
| 17 | Duration of previous ball possession phase (BA Table 1) | **GAP** | Requires looking backwards through possession state; broken ball track truncates it. Right-censored again (a lower bound is always observable). |
| 18 | The BA counterpressing **label** (XGBoost over all 134 features) | **NOT REPRODUCIBLE; a restricted variant is** | Two blockers: (i) it is *supervised* and needs hand labels -- BA used 11,108 labelled turnovers over 97 matches and state "In our case we found 100 labeled matches to be sufficient" (p. 2031); we have 12 matches. (ii) the global features (#11-15) are the imputation-dependent half. A defensible derivative = BA features restricted to #6-10 + #16-17 over the 2 s window, trained on whatever labels we can afford, reported as **"BA-local"**, never as "counterpressing per Bauer & Anzer". |
| 19 | PPDA (not from these papers, in our plan) | **NO** | 74% of the denominator (tackles 48%, interceptions 26%) has no class in the E2E-Spot 17-class label space. Structural, not a threshold issue. Already established in W1. |
| 20 | Per-frame role assignment, k visible players -> 10 roles (BI Sect. III.B) | **AS-IS (mechanically)** | Hungarian handles rectangular assignment. Cost matrix from log-probability under each role PDF is well-defined for any k. |
| 21 | Learning the role PDFs from broadcast (BI Sect. III.B) | **IMPUTE** | Visibility is correlated with pitch position -> MNAR on the estimand itself. Role PDFs learned from visible-only data will be ball-biased with under-estimated variance. |
| 22 | Per-frame zero-mean normalisation (BI Sect. III.B) | **NEEDS REDEFINITION** | Visible-subset centroid != team centroid; the bias is ball-correlated, so the artefact will look tactical. Impute-then-normalise, or anchor to pitch geometry. |
| 23 | Formation detection (10 role PDFs, EMD clustering; BI Sect. IV.B) | **IMPUTE** | Requires the full role set per half. A marginalised EMD over observed roles only is possible in principle but is not what BI validated at 75.33%, so that accuracy number would not transfer. |
| 24 | Role-swap timeline / 5-min sliding-window formation (BI Figs. 7, 9) | **GAP** | Tolerant to gaps by construction (a 5-min window has many frames); the binding constraint is #21, not frame coverage. |
| 25 | Line height, team compactness at an instant (our existing structural metrics) | **AS-IS to IMPUTE** | Instantaneous rather than 5 s-continuous, so untouched by the W1 census; but line height uses the defensive line's extremes and inherits #12's downward bias in a milder form. |

---

## 4. Between-team effect sizes (the numerator of D)

### 4.1 Quoted directly

| # | Quantity | Value | Source |
|---|---|---|---|
| E1 | %CP (share of transitions that are counterpressing), **manual expert labels**, range over 18 teams | **21.80% (Hannover 96) to 40.07% (B. Moenchengladbach)**, all-team mean 28.77% | BA p. 2015 + Table 7 p. 2034 |
| E2 | %CP, **model-detected, 6.5 seasons**, range over 25 teams | **18.65% (Union Berlin) to 24.87% (Dortmund)**, average 23.08% | BA Table 8 p. 2036 |
| E3 | Counterpressing situations detected per team per match | **20 to 30**, out of **90 to 200** transitions | BA p. 2021 |
| E4 | Home vs away counterpressing propensity | model classifies **27.24%** of home-team transitions as CP; labelled home data 28.77% ("teams playing at home tend to conduct counterpressing slightly more often") | BA p. 2025 |
| E5 | Numerical superiority within 10 m at turnover -> regain within 5 s | **36.2%** with superiority vs **30.2%** without (n = 109,852 detections) | BA p. 2025 |
| E6 | Shot/goal yield of counterpressing (all 4118 matches) | all CP: 3.22% shots for, 5.15% shots against, 0.35% goals for, 0.78% against. Successful CP: **6.27% / 1.76% / 0.75% / 0.25%**. Unsuccessful CP: **1.83% / 6.70% / 0.16% / 1.02%** | BA Table 4 p. 2023 |
| E7 | Correlation of CP metrics with final league ranking (season level) | successful-CP / total-transitions **r = -0.44**; shot balance **r = -0.36**; goal balance **r = -0.42** (negative = better rank) | BA p. 2024 |
| E8 | Coaches: CP+/T range over 30 coaches with >= 34 matches | **5.82% (Schuster) to 8.61% (Guardiola)**, average 7.40% | BA Table 9 pp. 2037-2038 |
| E9 | Coaches: %CP range | **16.88% (Schuster) to 25.98% (Veh)**, average 23.06% | BA Table 9 |
| E10 | Team-season %CP, extremes across all team-seasons | **16.88% (Darmstadt 2015/16) to 27.65% (Wolfsburg 2014/15)** | BA Tables 10-11 pp. 2039-2046 |
| E11 | Formation-cluster prevalence across 20 EPL teams | 4-4-2 **50.37%**, 4-2-3-1 **20.76%**, then 12.52 / 5.40 / 3.15 / 0.90% | BI Fig. 5 p. 4 |
| E12 | Formation classification accuracy vs expert labels | **75.33%** overall; per-cluster ~48-84% | BI Sect. IV.B p. 5 |
| E13 | Home vs away team identification from entropy maps + 23 match statistics | **47% classification accuracy** | HE p. 810, citing Ratke/entropy work |
| E14 | Top vs bottom team discriminating features (6396 games, 10 M events) | more passes and shots than the opponent; fewer fouls, tackles and goalkeeping actions. Final ranking predictable; **single-match victory/defeat "difficult to detect"** | HE p. 800 + p. 806, citing Pappalardo & Cintia |
| E15 | Team-specific defensive ghosting: expected goal value of a team's own defensive pattern vs "league average ghosts" | Swansea **69.1%** vs league-average ghosts **71.8%** | HE p. 806, citing Le et al. |

### 4.2 Derived: between-team SD from BA Appendix B, Table 8 (p. 2036)

`[derived]` -- computed over the 25 team rows (AVERAGE row excluded). Values transcribed from the
printed table, which is rotated 90 degrees in the PDF; two cells (`%CP+` for Hannover 96 and
Hamburger SV) fail the `%CP+ + %CP- = 100` check by 1.7 pp and 0.1 pp, so those two rows carry
transcription risk. Removing them does not move any SD by more than 0.05.

The right-hand block restricts to the **17 teams with >= 160 games** (>= ~5 full seasons), which is
the cleaner estimate of stable between-team variation: the promoted/relegated teams have 25-68 games
and their means are dominated by sampling noise.

| Metric | mean (25) | **SD (25)** | CV% | mean (17 stable) | **SD (17 stable)** | CV% | range (17) |
|---|---|---|---|---|---|---|---|
| Transitions per match (T/M) | 115.85 | 8.93 | 7.7 | 115.48 | **4.26** | 3.7 | 110.4-125.6 |
| Successful CP per transition (CP+/T, %) | 7.13 | 0.70 | 9.8 | 7.28 | **0.635** | 8.7 | 6.14-8.51 |
| Counterpress share (%CP) | 22.76 | 1.60 | 7.0 | 23.17 | **1.027** | 4.4 | 20.87-24.87 |
| CP success rate (%CP+) | 31.52 | 2.50 | 7.9 | 31.62 | **2.05** | 6.5 | 28.79-35.56 |
| Shots for per CP (%S+) | 3.15 | 0.405 | 12.9 | 3.22 | **0.396** | 12.3 | 2.58-3.90 |
| Shots against per CP (%S-) | 7.47 | 1.046 | 14.0 | 7.49 | **0.702** | 9.4 | 6.29-9.07 |
| Goals for per CP (%G+) | 0.32 | 0.136 | 42.4 | 0.35 | **0.120** | 34.2 | 0.14-0.66 |
| Goals against per CP (%G-) | 1.17 | 0.374 | 31.9 | 1.14 | **0.238** | 20.8 | 0.80-1.66 |

`[derived]` from BA Table 7 (manual labels, 4-6 matches per team, 18 teams): **SD of %CP = 4.73 pp**
(mean 28.83, range 21.80-40.07); **SD of turnovers per match = 9.83** (mean 117.0, range 97.6-133.4).

### 4.3 Derived: RMSE budgets implied by RQ-D

If D = SD_between / RMSE_censoring, then for a target D the censoring RMSE must satisfy
RMSE <= SD_between / D. Using the 17-stable-team SDs:

| Metric | SD_between | RMSE budget @ D=1 | RMSE budget @ D=2 | as % of the metric's mean @ D=2 |
|---|---|---|---|---|
| Transitions per match | 4.26 | 4.26 | **2.13** | **1.8%** |
| %CP | 1.027 pp | 1.03 pp | **0.51 pp** | **2.2%** |
| CP+/T | 0.635 pp | 0.64 pp | **0.32 pp** | **4.4%** |
| %CP+ (success rate) | 2.05 pp | 2.05 pp | **1.03 pp** | **3.2%** |
| %S+ | 0.396 pp | 0.40 pp | **0.20 pp** | 6.2% |
| %S- | 0.702 pp | 0.70 pp | **0.35 pp** | 4.7% |

These budgets are the single most decision-relevant output of this read. **Counterpressing metrics
are low-CV metrics.** Teams differ by 4-9% of the metric's own mean, so a broadcast pipeline needs
**1.8-4.4% relative accuracy** to reach D=2. For comparison, our measured possession-change detector
runs at **2.03x with a per-match ratio SD of 0.61 (CV 29.9%)** -- roughly **8x** the between-team CV
of transitions-per-match, giving `[derived]` **D ~ 0.12** for transitions-per-match today. It would
need per-match calibration to within a few percent to become a discriminative metric, and the W1
per-match spread (0.96x-2.75x) says that calibration constant is not stable.

### 4.4 Derived: the sampling floor -- a second denominator RQ-D does not currently include

`[derived]`, two independent routes:

- **Route A (season-to-season, cleanest).** FC Bayern's six seasons in BA Appendix D give
  within-team SD of %CP = **1.027 pp at 34 matches/season**. Scaled as sampling noise to a 12-match
  sample: 1.027 * sqrt(34/12) = **1.73 pp**. Against SD_between = 1.027 pp, that is
  **D_sampling = 0.59**. (Bayern's season SD also contains real coaching change -- Guardiola,
  Ancelotti, Kovac, Flick -- so this over-states noise and 0.59 is a *lower* bound on D_sampling.)
- **Route B (BA Table 7, 4-6 matches per team).** Observed cross-team SD at ~5 matches = 4.73 pp
  vs stable between-team SD 1.027 pp, implying a within-team term of 4.62 pp at 5 matches -> 2.98 pp
  at 12 matches -> **D_sampling = 0.34**. (This over-states noise too: it also absorbs the 82.01%
  inter-labeller disagreement and the manual-vs-model definition gap.)

Both routes land **D_sampling < 1 for %CP at 12 matches, with perfect tracking**. The implication is
uncomfortable and should be stated in the BTP rather than discovered by a reviewer:

> Even with commercial tracking, a 12-match sample cannot separate two Premier League teams on
> counterpress share, because the sampling SD of the estimate is 1.7-3.0 pp against a between-team
> SD of 1.0 pp. Broadcast censoring is a *second* error term added to a floor that is already above
> the signal.

Total error is `sqrt(RMSE_censoring^2 + SD_sampling^2)`. Reporting D against the censoring term
alone will produce a D that is real but not sufficient for the claim "distinguishes one team from
another". **The battery should report both, and the headline discriminability should use total
error.** This does not kill RQ-D -- it sharpens it. RQ-D's honest question becomes "does imputation
move the *censoring* term below the sampling floor, i.e. is broadcast the binding constraint or is
sample size?" -- and for %CP the answer looks like *sample size*, which is itself a publishable
negative result, and is a cheap one to establish before building anything.

---

## 5. Methodological warnings that should change the battery design

**W1. The radius rule is a bad detector, and it is the one everyone reaches for.**
"defender within 5 yards of the ball carrier" labels **72.41%** of all turnovers as counterpressing,
precision 0.31, AUC **0.602**; Andrienko-pressure thresholding gets AUC **0.568** (BA Table 3,
p. 2021). If the battery contains a rule-based pressing metric, it must be reported with that
provenance, not as a proxy for counterpressing.

**W2. The window is 2 seconds, not 5.** BA extract every feature at BPC + {0, 1, 2 s} and explicitly
reject longer windows (p. 2016). Our W1 census priced a **5 s continuous** window and got 5.5% yield
at N>=3. The census should be re-run at 2 s before the transition family is declared dead -- the
5 s requirement was ours, not the literature's. This is the highest-value cheap action from this
read.

**W3. Success and outcome labels are censored observations, not missing data.** Regain-within-5 s
and shot-within-20 s need two instants, not a trajectory. Discarding windows where the ball track
breaks throws away information *and* biases the estimate (track breaks are not independent of what
happens -- a long clearance both breaks the track and implies no regain). Use survival/censored
estimators.

**W4. Human labels cap the achievable accuracy at ~82%.** BA's three-student pairwise agreement is
**82.01%** (p. 2015), and they call it "the basic limitation to achieve further accuracy" (p. 2030).
Any validator we build against hand-annotated pressing labels inherits this ceiling; an F1 of 0.67
against such labels (BA's own model) is not a weak result. Set expectations before running the
validator, not after.

**W5. Exclusion criteria are half the method.** BA discard 47% of raw turnovers by rule (set-piece
starts, own-half losses) and a further 62.5% of the remainder as "undefined" during labelling. Our
2.03x over-detection versus Sofascore ballRecovery is being compared against a construct that has
had none of those exclusions applied. Apply BA's rules to our turnovers before quoting a
calibration factor.

**W6. Whole-team statistics and local statistics have opposite censoring behaviour.** Convex hull,
covered area, global stretch, team centre and players-behind-ball are all biased (not merely noisy)
under broadcast visibility, and biased in a direction correlated with the ball. Local statistics
(nearest-player distance/speed, 10 m counts, local-5 stretch) are approximately unbiased. **Split the
battery along that line and report the two families separately** -- otherwise a single aggregate D
will average a well-behaved family with a pathological one.

**W7. The visible-centroid trap.** BI normalises every frame to zero mean (Sect. III.B) and BA uses
team centre as a feature. On broadcast, the visible-subset mean is biased toward the ball, so this
normalisation injects ball-correlated translation that will read as tactical structure. This is a
silent failure -- it produces plausible numbers. Impute first, or anchor to pitch geometry.

**W8. Role-PDF learning is MNAR, not MCAR.** Visibility depends on pitch position, which is the
quantity being estimated (Sect. 2.4b). Imputation here is not variance reduction, it is bias
removal, and its validity has to be argued (calibrated virtual broadcast camera on Metrica is
exactly the right instrument -- use it to measure the *bias*, not just the RMSE).

**W9. Cross-system comparability is already a known problem in the literature.** HE p. 812:
"scenarios involving quick, unpredictable movements including frequent occlusions between players
provide challenges [...] in regards to the accuracy of the information [...] it is recommended that
practitioners use caution when comparing results between different tracking systems." Our
between-team SDs come from Tracab/Prozone; our RMSEs come from our own pipeline against Metrica.
That mismatch belongs in the limitations section explicitly.

**W10. Counterpressing metrics are confounded with team quality.** BA warn their own benchmarks
"should be used carefully since they are based on small sample size and could contain confounding
effects with the overall offensive or defensive qualities of a team. Loosing the ball, increases the
probability of an opponent conducting an offensive action" (p. 2024), and that fallback is not a
clean control ("all non-counterpressing situations consist of myriad of different circumstances,
they do not serve as reasonable baseline", p. 2024). A between-team difference in %CP is not
automatically a difference in *style*.

**W11. Single-match discrimination is hard even with full event data.** Pappalardo & Cintia over
6396 games and 10 M events: final ranking is predictable from performance features, but victory or
defeat in an individual game "is difficult to detect" (HE p. 806). A 12-match corpus sits closer to
the single-match regime than the season regime.

**W12. Bialkowski reports no between-team effect sizes.** If the BTP needs role-based
discriminability numbers, they are not in this paper. The two follow-ups that would have them are
Lucey et al., "Characterizing Multi-Agent Team Behavior from Partial Team Tracings: Evidence from
the English Premier League", AAAI 2012 (**BI ref [16] / HE ref [61]** -- literally about partial
observation, and the single most relevant uncited paper for this project), and Bialkowski et al.,
"Identifying team style in soccer using formations learned from spatiotemporal tracking data",
ICDMW 2015. Neither is in `docs/papers/`.

---

## 6. Recommended battery, ordered by (effect size) / (censoring exposure)

Tier 1 -- broadcast-native, local, defensible now:
1. Nearest-presser distance and speed at BPC + {0,1,2 s} (BA SHAP 2, 5).
2. Local-5 stretch index of the losing team near the ball at +2 s (BA SHAP 4).
3. Players within 10 m of the ball per team, and the numerical-superiority indicator (BA RQ3;
   effect size E5 = 36.2% vs 30.2%).
4. Andrienko pressure on the ball -- as a feature, with its AUC 0.568 provenance attached.

Tier 2 -- censored-estimator redefinitions, cheap and high value:
5. Defensive reaction time as a Kaplan-Meier survival curve over all 2546 turnovers.
6. Regain-within-5 s probability from the same curve (this is BA's success label).
7. Counterpress share %CP restricted to BA's inclusion rules (no set-pieces, no own-half losses).

Tier 3 -- imputation-dependent, i.e. the actual RQ-D experiment:
8. Team centre displacement over the first 2 s (the fallback-vs-counterpress discriminator in BA
   Fig. 7, p. 2033).
9. Players behind the ball at +1 s, with the >= 4 threshold (BA Fig. 2).
10. Covered area / global stretch / convex hull change over 2 s.
11. Role-aligned formation via BI minimum-entropy partitioning.

Report Tier 1 vs Tier 3 D-values separately. The RQ-D headline is precisely the gap between them,
and the honest framing is "imputation is what moves a metric from Tier 3-unusable to Tier
3-usable", measured against a **total** error that includes the sampling floor from Sect. 4.4.

---

## 7. First actions implied (not done here; read-only task)

1. **Re-run the W1 turnover-window census at a 2 s window** instead of 5 s. The 5 s requirement was
   ours; BA's detector needs 2 s. This may change the FAIL verdict on the transition family and
   costs one CPU run.
2. **Apply BA's exclusion rules** (drop set-piece-origin possessions, drop own-half losses) to
   `detect_turnovers` output and re-measure the 2.03x ratio against Sofascore.
3. **Measure post-`link_ball` ball retention over 3 s** (not 5 s) -- that is what individual ball
   possession time, the top SHAP feature, actually needs.
4. **Validate the `Shots on/off target` E2E-Spot classes** against truth the way `Foul` was
   (1.069x); the 20 s shot-balance outcome metric depends entirely on it.
5. **Measure imputation bias, not just RMSE**, on team centre / covered area / convex hull using the
   calibrated virtual broadcast camera on Metrica. Those three are biased, not noisy, and RMSE alone
   will not show it.
