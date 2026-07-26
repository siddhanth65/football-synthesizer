# RESEARCH_QUESTION_OPTIONS — scientific advice, not encouragement (2026-07-26)

Audience: Sid. Context: BTP, IIIT Delhi, reviewed December 2026 by two technically strong
professors. Target style: Rahimian, Davis & Toka (Machine Learning, 2026) — one clear question,
real baselines, ablations, honest limits. Hard constraint: no paid football data.

This document supersedes the research-question framing in `docs/PROJECT_DEFINITION.md`. It does not
supersede `docs/SCOUTING_PACK_PLAN.md`, which is a *demo* plan and survives intact under a new
question (see §4).

---

## 1. Blunt assessment of the current framing

### 1.1 Would the CV pipeline alone satisfy two strong examiners?

**No.** Not close. Say it plainly to yourself now rather than hear it in December.

Broadcast video → detection → tracking → homography → team assignment → position projection was
published as an end-to-end pipeline by Theiner et al. at WACV 2022
(https://arxiv.org/abs/2110.11107), four years before you built yours. The camera-calibration layer
is a solved, competed sub-problem: TVCalib (https://arxiv.org/abs/2207.11709), BroadTrack (WACV
2025), Cioppa et al. (CVPR-W 2021). The whole stack *including identity* is a standing SoccerNet
challenge track — Game State Reconstruction (https://arxiv.org/abs/2404.11335) — with a published
2024 winner that does detection + camera parameters + DeepSORT re-ID + jersey OCR end to end
(https://arxiv.org/abs/2504.06357). Your GS-HOTA 14.76 → 22.85 is an honest number on a public
benchmark, and it is not competitive with that leaderboard. An examiner who spends ten minutes on
Google Scholar finds all of this.

So: the pipeline is a **research instrument**, and a good one. It is not a research contribution.
Frame it in Chapter 3 as apparatus, cite Theiner and SoccerNet GSR as the prior art you are
standing on, and spend your novelty budget one layer up.

### 1.2 What in the existing work is genuinely research

Three things, and only three.

1. **The camera simulator calibrated against a *measured* broadcast footprint.** You did not assume
   a viewport — you derived one from SkillCorner open broadcast tracking and verified 98.4% of
   detected players fall inside it. Every comparable paper hand-specifies the viewport: DeepMind
   uses a synthetic view cone (https://www.nature.com/articles/s41598-022-12547-0), Choi 2026 uses
   a 36–60 m assumed window (https://arxiv.org/abs/2607.11548). A *validated* censoring operator is
   the load-bearing piece of any simulate-and-measure study, and yours is better validated than the
   published ones. This is the single most defensible asset you own.
2. **Calibrated uncertainty on the imputation, not just a point estimate.** 50%/90% regions holding
   6/6 is a different kind of claim from "8.10 m mean error". Choi's method is a point estimator
   with no calibration story; DeepMind reports error, not coverage. Calibrated intervals are what
   let you propagate measurement error into a *metric-level* confidence interval, which is where
   the actual contribution lives.
3. **The corpus design.** Six same-opponent home/away pairs is an opponent-controlled repeat
   measurement. Buldu et al. (Sci Rep 2020, https://pmc.ncbi.nlm.nih.gov/articles/PMC7661721/)
   aggregate over a season; Bialkowski et al. (2014) compare home/away with uncontrolled opponents.
   Nobody controls the opponent. At n=6 it is underpowered (see §1.4), but the *design* is right and
   it is the thing your imputation competitors do not have.

### 1.3 What is engineering, not research

The gate. The abstain tier. The fact store and its staleness guard. The registry. The correction
trail. The pre-registration protocol. Version-stamped artifacts.

All of this is **good practice**, and it is exactly why your results will hold up. None of it is a
finding. "Validated-or-abstain" is not a contribution — it is what a competent measurement study
does by default, and no reviewer gives credit for the absence of a mistake. `PROJECT_DEFINITION.md`
currently lists it as contribution #1. Delete that.

The one way this becomes research: **if the abstain thresholds are derived rather than chosen.**
Right now `coverage ≥ 40`, `recall proxy ≥ 50`, `spread ≤ 0.05` are engineering constants somebody
picked. If instead you *derive* the visibility level at which each metric stops being able to
distinguish two teams, and the gate fires on that derived threshold, then the gate is the output of
a measurement-theory result and the whole abstention layer becomes a deliverable. That is the pivot
this document recommends (§3, RQ-D).

### 1.4 What is presentation, and the numbers that will not survive contact

Be honest with yourself about the empirical section of `PROJECT_DEFINITION.md`:

- **"Possession control is stable, r = +0.83."** n=6 pairs. Fisher-z 95% CI ≈ **[0.06, 0.98]**. That
  interval contains "barely correlated" and "almost deterministic". It supports no claim at all.
- **"Defensive line height is NOT stable, r = −0.40."** n=6. 95% CI ≈ **[−0.92, +0.61]**. This is
  not a negative result; it is *no result*. Reporting it as a finding is the kind of thing a strong
  examiner opens with.
- **"Style fingerprint separation decays 2.0 → 0.74 → 0.09."** This is very likely a construct
  artifact, not a discovery (§2).
- **"7 of 9 players sit higher under Amorim."** Confounded with venue, opponent, score state, and
  your own tracker's per-fixture bias, on n=1 managerial change. Bekkers & Dabadghao needed 8,219
  matches to make a manager-effect claim (https://journals.sagepub.com/doi/10.3233/JSA-190290).
- **The forecast (7.92 pp MAE beating 10.63 / 12.18).** This one is real and pre-registered, but it
  is a *demo* result, not a research finding — you are forecasting one number from one prior meeting
  on a 6-pair set. Keep it in the demo chapter; do not put it in the abstract.

**The scouting pack is a demo, not a result.** It is excellent viva material — a generated artifact
an examiner can hold, with a live abstention on Southampton — and it should stay exactly as
`SCOUTING_PACK_PLAN.md` specifies. But zero lines of it are a research contribution, and if it is
the centrepiece of the *thesis* rather than of the *defence*, the thesis has no centrepiece.

### 1.5 The scoop you have to deal with, up front

Thirteen days before this review, Seongjin Choi posted *Training-Free Off-Screen Player Imputation
for Broadcast-Based Spatial Football Analytics* (https://arxiv.org/abs/2607.11548): Metrica open
tracking, simulated broadcast viewport, off-screen imputation, downstream metric degradation,
real-broadcast case study, public code. Same dataset, same problem statement, same closest prior
work (Omidshafiei et al.), comparable numbers (3.3–8.9 m median error vs your 8.10 m mean).

Also: Penn, Donnelly & Bhatt reconstruct continuous tracks from discrete broadcast detections with
~6.9–7.3 m off-camera error, validated on Metrica
(https://royalsocietypublishing.org/doi/10.1098/rsos.251175). Everett et al. (AAMAS 2023) impute to
~6.9 m from event data alone (https://arxiv.org/abs/2302.06569). MIDAS (ECML-PKDD 2025,
https://arxiv.org/abs/2408.10878) evaluates imputation on downstream distance/pass-success/pitch
control.

**Consequence, non-negotiable: off-screen imputation can no longer be a contribution.** It is your
instrument. Put a table in Chapter 2 with your 8.10 m next to Choi's 3.3–8.9 m, Penn's ~6.9 m and
Everett's ~6.9 m, note that yours is causal/online and calibrated where theirs are not, and move on.
Discovering Choi in the viva instead of in Chapter 2 is a survivable but avoidable wound.

---

## 2. The identity-measurement critique

**Was the marginal / optimal-transport style embedding the wrong construct? Yes, almost certainly.**

The method characterises a team by the *marginal* distribution of player positions over frames —
where players stand on average, unconditioned on possession phase, opponent action, or score state
(Baouan, Pulido & Rosenbaum, https://arxiv.org/abs/2501.10299; and the JRSSC Serie A touch-map
paper, https://academic.oup.com/jrsssc/article/75/3/624/8321559, which does the same thing with
touch locations). A marginal occupancy map moves whenever the opponent's block moves. It cannot
separate *"we changed how we play"* from *"they made us stand somewhere else"*. Your "identity
decays as matches are added" result is exactly what you would expect from a construct that is
mostly measuring the opponent: pool more opponents, the opponent-driven component averages toward
the league mean, apparent separation collapses. **The decay is diagnostic of the instrument, not of
football.**

This critique is not yours alone, and that is good — it means it is citable. Bialkowski et al.
(ICDM-W 2014, https://ieeexplore.ieee.org/document/7022571/) argued a decade ago that *role-aligned*
descriptors, not raw positional marginals, carry team-style signal. Buldu et al.
(https://pmc.ncbi.nlm.nih.gov/articles/PMC7661721/) flag in their own limitations that passing-network
consistency captures positional tendency, not context-dependent behaviour. The scoping review of 40
playing-style studies (Plakias et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC10123610/) lists
counterpressing and build-up patterns as *unmeasured* styles and notes that test-retest designs are
essentially absent from the field.

**What a defensible conditional definition looks like.** Identity is a *response function*, not an
occupancy map: a distribution over behaviour conditioned on a trigger.

| Trigger (conditioning event) | Response measured | Literature anchor |
|---|---|---|
| Ball lost in opponent half | Regain within 5 s / 5 m; time-to-first-pressure | Merckx et al. MLSA 2021 (https://dtai.cs.kuleuven.be/events/MLSA21/papers/MLSA21_paper_merckx.pdf) |
| Ball lost anywhere | Counterpress vs retreat (binary, per loss) | Bauer & Anzer, DMKD 2021 (https://link.springer.com/article/10.1007/s10618-021-00763-7) |
| Goalkeeper in possession | Short build-up vs long, first-pass length distribution | practitioner-standard; flagged as academically unmeasured by Plakias et al. |
| Opponent in settled possession | Defensive line height **relative to ball**, not absolute | arXiv:2511.06191 — relative line height beats absolute as a predictor |

Each of these is (a) a conditional distribution, (b) invariant to where the opponent happens to
stand when nothing is happening, and (c) already a recognised construct in the literature so you are
not inventing a metric nobody will accept.

**Use the failed fingerprint as a negative control, not as dead weight.** "Marginal embedding
separation collapses with n; conditional signature separation does not (or does — report either
way)" is a clean, publishable-shaped comparison, and it converts your most embarrassing result into
the motivating experiment of the thesis. That reframing is free and you should take it regardless of
which question you pick.

---

## 3. Candidate research questions, ranked

Ranked by: novelty that survives an adversarial examiner × feasibility on your actual data ×
distance from what is already scooped.

---

### RANK 1 — **RQ-D (new; the survey found this, we did not ask it): the resolution-limit question**

> **At what level of broadcast visibility does each pressing/transition tactical metric stop being
> able to distinguish one team from another — and does calibrated off-screen imputation move that
> threshold?**

This is the question the survey surfaced and it is stronger than all three we checked. It ranks
first, and the reason is one sentence: **every existing paper reports metric error; none reports
error relative to the effect the metric is supposed to resolve.** A 3 m bias on line height is
catastrophic if teams differ by 2 m and irrelevant if they differ by 12 m. Nobody has published that
ratio for any football metric.

**The quantity.** For each metric *m*:

```
D(m, v) = between-team SD of m  /  RMSE of m induced by broadcast censoring at visibility v
```

`D > 1` means the metric can still tell two teams apart under broadcast conditions. `D < 1` means
the measurement noise swamps the football. **The abstain threshold is `D = 1`** — derived, not
chosen. Sweep *v* and you get a resolution curve per metric.

**Method.**
1. Implement a metric battery on Metrica full-pitch truth: PPDA and a tracking-based pressing
   intensity (Bekkers, https://arxiv.org/abs/2501.04712), 5 s/5 m recovery rate (Merckx), counterpress
   vs retreat rate (Bauer & Anzer construct, rule-based not ML), line height absolute *and*
   relative-to-ball, stretch/compactness (Frencken et al.,
   https://onlinelibrary.wiley.com/doi/10.1080/17461391.2010.499967).
2. Censor with your calibrated simulator at a sweep of visibility levels. Recompute.
3. Three recovery strategies as an explicit ablation: **(a)** ignore off-screen players (what
   SoccerNet GSR effectively does), **(b)** linear interpolation (what challenge winners do), **(c)**
   your calibrated imputation with uncertainty propagated to metric-level intervals.
4. Estimate between-team SD from your 12-match corpus + public season data, with the corpus's
   opponent-controlled pairs used to separate team variance from opponent variance.
5. Transfer-check on real broadcast: do the metrics whose `D > 1` actually separate teams in the
   12-match corpus, and do the `D < 1` ones actually fail to?

**Data:** Metrica (have), calibrated simulator (have, validated 98.4%), imputation + calibration
(have), 12 real matches (have), event pipeline for PPDA/recovery (have, partially validated). **No
new data, no paid data, no new CV work.** Everything is an instrument you already built.

**Deliverable.**

*Table 1 — the resolution table.* One row per metric; columns: full-truth value, censored bias,
censored RMSE, between-team SD, `D` under ignore / interpolate / impute, verdict
(REPORTABLE / REPORTABLE-WITH-CORRECTION / ABSTAIN).

*Figure 1 — the money figure.* `D` on the y-axis against visible-player fraction on the x-axis, one
curve per metric, horizontal line at `D = 1`. Reading straight off it: "counterpress rate is
recoverable above 9/22 visible; absolute line height never is." That figure is the thesis.

*Figure 2 — transfer.* The 12 real matches with propagated metric-level CIs, showing which
between-team differences survive and which vanish into the error bars.

**Novelty verdict: strong, with an honest boundary.** The simulate-and-measure *paradigm* is prior
art — Omidshafiei 2022, Bassek et al. 2025
(https://www.tandfonline.com/doi/full/10.1080/24733938.2025.2533808), Choi 2026. What is not prior
art: (i) the pressing/transition metric family — Bassek stops at physical metrics + formation
detection, Choi stops at pitch control and says so in his own repo README; (ii) normalising error by
between-team effect size to yield a *decision rule* rather than a catalogue; (iii) propagating
calibrated imputation uncertainty into metric-level intervals — nobody does this, all published
imputation is point-estimate; (iv) Crang et al. (https://arxiv.org/abs/2508.19477) explicitly
*assert* that broadcast-derived tactical metrics survive better than physical ones **without testing
a single tactical metric.** You would be turning a stated-but-untested assumption in the literature
into a measured result, which is a clean thing to say in an abstract.

**Risk of failure: low, and it fails informatively.** If every metric comes back `D < 1`, that is the
result — "broadcast video cannot resolve team-level pressing differences, here is the visibility you
would need" — and it is a *better* thesis than a mixed table, because it is a hard negative with a
number attached. There is no outcome where you have nothing.

**Effort to December (~20 weeks):** metric battery on Metrica 4 wk; censoring sweep + `D` 3 wk;
between-team variance from the corpus 3 wk; real-broadcast transfer 3 wk; scouting-pack demo (already
planned, reuses everything) 4 wk; writing 3 wk. Tight but real, and every week of it uses an
instrument that already exists.

**The examiner attack you must pre-empt:** *"Isn't this just RQ-A with a denominator?"* Answer, and
have it ready: the denominator changes what you must measure. A bias table needs only ground truth;
a discriminability table needs the between-team variance, which is why the opponent-controlled
12-match corpus is load-bearing rather than decorative, and it is why the output is a threshold a
practitioner can act on rather than a number they cannot interpret.

---

### RANK 2 — **RQ-C: pressing and ball-recovery metrics from broadcast, with quantified error**

> Can PPDA-like pressing intensity, counter-press success and 5 s ball recovery be estimated from
> broadcast video with quantified error, and what do they say about a team across a season?

This is RQ-D minus the discriminability framing, and it is the safe fallback. Same data, same
instruments, same battery. It is directly your professor's stated interest.

**Why it ranks below RQ-D:** the second half ("what do they say about a team across a season") is
application, not research, and at n=12 with honest error bars it will very likely say nothing
detectable — which then reads as a failed thesis rather than as a measured limit. And the
pressing-intensity leg alone is structurally "Choi with the downstream metric swapped" (both are
time-to-intercept constructs over pitch control), which an examiner can fairly call incremental.
RQ-D absorbs both problems: the null result becomes the deliverable, and the discriminability
criterion is not something Choi has.

**Feasibility flag you must resolve either way:** PPDA's denominator is tackles/interceptions/fouls.
Your per-player attribution is 1–4% of truth and only *pass counts* are validated (0.97–1.09×). You
either demonstrate the defensive-action numerator works, or you redesign the estimator around
possession-change events, which you can detect. Do not assume this. Measure it in week 2 (§5).

**Effort:** ~16 weeks. **Risk:** medium — the "what does it say about a team" half can come back
empty and there is no pre-registered way to make that interesting.

---

### RANK 3 — **RQ-B: conditional tactical identity from broadcast**

> Is a team's tactical identity measurable from broadcast video when identity is defined
> *conditionally* (post-turnover response, GK build-up route, line height given score state) rather
> than *marginally*, and is it stable under an opponent-controlled repeat design?

Intellectually the most interesting question here, and the one that best matches §2's critique. It
ranks third on **statistical power**, not on ideas.

**The problem is n.** 12 matches, 6 pairs, 1 managerial change. Slice a conditional signature by
score state and you are estimating a distribution from a handful of events. Buldu et al. measured
identity across a 380-match season; Bekkers & Dabadghao ran the manager experiment on 8,219 matches;
Bauer & Anzer profiled counterpressing over six Bundesliga seasons. As a *football discovery*
question you lose to all three on data. Worse, ball trackability of 32–52% caps the number of usable
turnover windows before you even start — and that number is currently unknown (§5).

**It can be rescued** by reframing as measurement science — "how much of each conditional signature
survives broadcast-grade observation, reported as reliability intervals not hypothesis tests" — but
that reframing *is* RQ-D with a smaller metric battery. If you like RQ-B's framing better, run RQ-D
and put the conditional-vs-marginal comparison in as a chapter. You get both.

**Effort:** ~18 weeks. **Risk:** high — depends entirely on the turnover-window census coming back
healthy, and there is no fallback if it does not.

---

### RANK 4 (REJECT AS WORDED) — **RQ-A: which tracking-derived metrics survive broadcast censoring**

> Compute metrics on full-pitch truth, recompute under censoring, measure bias, try to correct it.

**Do not use this sentence.** It is the title-level claim of two recent papers. Bassek, Theiner,
Ewerth, Memmert & Raabe (Science and Medicine in Football 9(4), 2025) emulate broadcast tracking
from real feeds and measure degradation of physical metrics *and* formation detection against
official TRACAB, with released code. Choi (arXiv:2607.11548, July 2026) censors Metrica with a
simulated viewport, quantifies pitch-control distortion, corrects with imputation, and closes with a
real-broadcast case where the verdict flips. Between them they own both halves. An examiner who
knows either one ends the viva in a single question.

The *residual gap* inside RQ-A — the pressing/defensive metric family, the correction-strategy
head-to-head, calibrated uncertainty at metric level — is real, and RQ-D is exactly that gap stated
as its own question rather than as a subset of somebody else's. Keep the design, discard the title.

---

## 4. Recommendation

**Commit to RQ-D.**

> **Under measured broadcast visibility, which pressing and transition metrics retain enough
> resolution to distinguish one team from another — and does calibrated off-screen imputation move
> that threshold?**

**Why this one:**

1. **It needs no new data and no new CV work.** Every instrument exists and is validated. Five
   months is not enough time to build a new capability *and* answer a question; it is enough to
   answer a question with the capability you have.
2. **It is the only framing where your two scooped assets become strengths.** The imputation model
   stops being a contested contribution and becomes one arm of a three-arm ablation. The
   SkillCorner-validated simulator stops being a supporting detail and becomes the study's censoring
   operator — the most rigorous one in this literature.
3. **It converts the abstention layer from hygiene into a result.** The gate thresholds stop being
   engineering constants and become the `D = 1` crossings you derived. That is the single biggest
   upgrade available to the existing work.
4. **It cannot fail into nothing.** A table of all-ABSTAIN is a hard negative with a number attached,
   and hard negatives with numbers are what strong examiners reward.
5. **It lands on your professor's actual interest** — pressing, ball recovery within 5 s, deriving
   metrics — while being a measurement study rather than a football-discovery study, which is the
   only kind of study 12 matches can support.

**What is preserved as the instrument (all of it, relabelled):**

| Existing work | New role |
|---|---|
| CV pipeline (detect/track/homography/teams/events/identity) | Chapter 3 apparatus. Cite Theiner 2022, SoccerNet GSR as prior art you build on. No novelty claimed. |
| Calibrated camera simulator (98.4% SkillCorner footprint) | **The censoring operator.** Chapter 4. This is the study's methodological backbone and its validation is a genuine result. |
| Off-screen imputation (8.10 m, 6/6 calibration) | **Ablation arm (c)**, benchmarked against Choi / Penn / Everett in a Chapter 2 table. The *calibration* is what you claim, not the error. |
| Event spotting validated at 0.97–1.09× official | Feasibility evidence for the event-based half of the battery (PPDA, recovery). |
| 12 matches / 6 opponent-controlled pairs | **The between-team variance estimate** — the denominator of `D`, and the real-broadcast transfer set. |
| Gate + abstain layer | The mechanism that *implements* the derived `D = 1` thresholds. Now downstream of a result. |
| Scouting pack (`SCOUTING_PACK_PLAN.md`) | **Keep exactly as planned.** It becomes the applied demonstration chapter: what a pack looks like when every claim is filtered by a derived resolution threshold. Southampton's abstention is now *principled*, not just cautious. |
| Style fingerprint decay (2.0 → 0.74 → 0.09) | **Negative control / motivating experiment** (§2). Marginal construct fails; that is why the thesis measures conditional ones. |

**What gets dropped:** "validated-or-abstain" as contribution #1; the r = ±0.83/−0.40 stability
claims as findings (report with CIs, in a limitations section, as underpowered); the manager-change
narrative as anything more than a described confound.

---

## 5. First two weeks — the cheapest experiments that tell you the question is answerable

Three checks. All use existing code. Any one of them coming back badly changes the plan, which is
the point. **Do not start building the metric battery before these land.**

### W1-A — The turnover-window census (2 days). *Kills or confirms half the battery.*

On all 12 real matches, count: turnovers where the ball is tracked at the loss moment **and** stays
tracked through +5 s, with ≥ N defenders visible. Report the distribution per match, not the mean.

- **Pass:** ≥ 30 usable windows/match → the transition metrics (recovery, counterpress) are viable.
- **Marginal:** 10–30 → viable pooled across matches only; per-match claims die.
- **Fail:** < 10 → drop the transition family, restrict the battery to structure metrics (line
  height, compactness) and event-only pressing (PPDA-like), and say so in the proposal.

Given 32–52% ball trackability this is genuinely uncertain, and it is the single largest unknown in
the whole plan. It costs two days. Run it first.

### W1-B — The defensive-action numerator check (1 day). *Kills or confirms PPDA.*

Count detected tackles/interceptions/fouls per match against official public counts, the same way
you validated passes. If the ratio is anywhere near the 0.97–1.09× you got on passes, PPDA is in. If
it is at the 1–4% attribution level, PPDA as literally defined is out and the estimator must be
redesigned around possession-change events — which is fine, but you need to know now, and the
redesign is itself a defensible methods contribution ("PPDA is not broadcast-estimable; here is what
is").

### W2 — The one-metric end-to-end dry run (5 days). *Proves the whole design in miniature.*

Pick **one** metric — recommend **defensive line height, relative to ball** (arXiv:2511.06191 says
relative beats absolute, it is computable from visible defenders, and you already have a partial
implementation). Run the complete RQ-D loop on it, on Metrica alone:

full-truth value → censor at 3 visibility levels → bias + RMSE → between-team SD → `D` under
ignore / interpolate / impute → one row of Table 1 and one curve of Figure 1.

If that row and that curve exist by end of week 2, the thesis is a matter of repetition and the
question is answered. If the loop stalls — the metric will not compute on censored data, the
between-team SD is unestimable, the propagation does not close — you have found the structural
problem in week 2 instead of week 14, with time to switch to RQ-C.

**Decision point, end of week 2:** W1-A pass + W2 loop closes → commit to RQ-D with the full
battery. W1-A marginal/fail → commit to RQ-D with a structure-only battery and state the transition
family as future work. W2 loop does not close → fall back to RQ-C.

---

## 6. Report outline, mapped to the reference paper's shape

Rahimian, Davis & Toka works because it has: one question stated in the abstract, baselines that a
sceptic would have proposed, ablations that isolate each component's contribution, and limitations
that name what the method cannot do. Mirror that structure literally.

**1. Introduction** — the constraint (broadcast shows ~11.8/22, independently confirmed by
SkillCorner and by a 5%-full-visibility audit, https://medium.com/@arnaud_santin/football-data-finally-under-scrutiny-arnaudsantin-58c14ffe2d52),
the gap (Goes et al.'s review of 73 tactical-tracking studies, all assuming full tracking,
https://onlinelibrary.wiley.com/doi/10.1080/17461391.2020.1747552), and the untested assertion you
are testing (Crang et al. claim tactical metrics survive broadcast better than physical ones without
testing one). One-paragraph statement of `D` and the `D = 1` rule. Contributions as three bullets.

**2. Related work** — four subsections, each ending in an explicit "what this leaves open":
(2.1) broadcast CV pipelines — Theiner, TVCalib, SoccerNet GSR: *your apparatus, not your claim*;
(2.2) off-screen imputation — Omidshafiei, **Choi**, Penn, Everett, MIDAS, with **a numeric
comparison table containing your 8.10 m**;
(2.3) tactical metrics under full tracking — Spearman, Bekkers, Merckx, Bauer & Anzer, Frencken,
Link;
(2.4) degradation studies — Bassek, Crang, Mills. State in one sentence which cell of the design
matrix each leaves empty.

**3. Apparatus** — pipeline, with its validated numbers and its measured limits (ball 32–52%,
attribution 1–4%, ~11.8/22). Written as a methods section, not as an achievement.

**4. The censoring operator** — the simulator, the SkillCorner footprint derivation, the 98.4%
validation, and the visibility sweep. This is the chapter that makes the study credible; give it the
space.

**5. Metric battery and the discriminability criterion** — formal definition of each metric with its
citation, formal definition of `D`, and the uncertainty-propagation scheme from calibrated imputation
regions to metric-level intervals.

**6. Experiments**
- 6.1 **Baselines** (the sceptic's list): ignore-off-screen; linear interpolation; nearest-neighbour
  role fill; a fitted per-metric correction factor with no imputation at all. The last one matters —
  if a scalar correction does as well as your model, that is a finding and you must report it.
- 6.2 **Main result**: Table 1 + Figure 1.
- 6.3 **Ablations**: visibility level; occlusion duration bucket (short vs long — Choi shows >9.6 s
  occlusions are where prior learned work fails, so this axis is pre-motivated); metric family;
  with/without uncertainty propagation.
- 6.4 **Transfer**: the 12 real matches; do `D > 1` metrics actually separate teams and do `D < 1`
  metrics actually fail to.
- 6.5 **Negative control**: marginal OT fingerprint vs a conditional signature, same matches.

**7. Applied demonstration** — the scouting pack, the derived thresholds driving the gate, Brighton
as the working case and Southampton as the principled abstention. Pre-registered protocol and
one-shot scoring exactly as `SCOUTING_PACK_PLAN.md` specifies.

**8. Limitations** — written as claims you are actively refusing to make, not as apologies:
2 Metrica matches vs Omidshafiei's 105; n=6 pairs is proof-of-signal, never significance, with the
CIs printed; ball trackability caps the transition battery; no per-player claims at 1–4% attribution;
venue and manager are inseparable in this corpus; the simulator is calibrated to one broadcast style.

**9. Conclusion** — the resolution curve, restated as a practitioner rule, plus the one sentence
that makes it citable: which metrics a broadcast-only analyst may report, which they must correct,
and which they must refuse.

---

## Bottom line

**Recommended question:** *At what level of broadcast visibility does each pressing and transition
metric lose the resolution to distinguish one team from another, and does calibrated off-screen
imputation move that threshold?*

**Why:** it is the one framing that needs no new data, converts both of your scooped assets
(imputation, simulator) into instruments of a question nobody has asked, derives your abstention
thresholds instead of choosing them, lands squarely on your professor's pressing/recovery interest,
and cannot fail into a non-result.

**Two-week de-risking:** (1) census the usable post-turnover windows across all 12 matches — ≥30/match
keeps the transition battery alive, <10 kills it; (2) validate the defensive-action count against
official numbers to see whether PPDA is estimable at all; (3) run the entire loop end-to-end on one
metric (ball-relative line height) on Metrica alone, producing one row of Table 1 and one curve of
Figure 1. If that row and curve exist on day 14, the thesis is repetition. If they do not, you have
lost two weeks instead of fourteen.
