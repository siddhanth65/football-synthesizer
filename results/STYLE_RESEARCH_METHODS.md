# STYLE_RESEARCH_METHODS — the synthesis

**Deliverable 3 of 3** for the style-analysis deep research (brief: `docs/STYLE_RESEARCH_BRIEF_2026-07-22.md`).
Companion catalogues: `results/STYLE_RESEARCH_XGFC.md`, `results/STYLE_RESEARCH_SOURCES.md`.

**The question.** Which published methods beat what we already run for a stylistic analysis of
Manchester United (EPL 24-25), given that we hold **censored broadcast tracking** — ~11.8/22 players
visible per frame, ~26.4% of frames with usable homography, ball track usable 32-52% — not full-pitch
commercial tracking? Every method below is scored on whether it survives that data regime under the
project's **validated-or-nothing** rule.

**The bar to beat (do not re-recommend).** OT sliced-Wasserstein territorial embedding (Baouan 2025);
rule-based 4-phase segmentation; StatsBomb counter-press primitive; score-state segmentation from
validated goal boundaries; E2E-Spot 17-class events; BAS lRomul PASS/DRIVE (0.97-1.09x Sofascore);
lineup-prior Hungarian identity + PRTreID + jersey OCR; GS-HOTA 22.85; planned shape graphs / phase CNN
/ line-breaking passes.

**How to read the tables.** `runs on our data` ∈ {yes = event/geometry we already emit; adapt = needs
adaptation to censoring; no = needs full-pitch tracking we cannot produce}. Recommendation ∈
{**build**, **probe**, skip}. Rows corrected or refuted by the adversarial verification pass are struck
through in an **honesty trail** under each axis, with the reason — that trail is part of the deliverable,
not an appendix.

---

## Axis 1 — Formations & team structure

Formation detection, role assignment, shape metrics, rest-defence. Our stack has **nothing** here beyond
the territorial OT embedding (label-free) — no named formations, no role labels, no rest-defence.

| # | Method | Source | Runs on our data | Expected lift | Cost | Rec |
|---|--------|--------|:---:|---------------|------|:---:|
| 1 | Training-free off-screen player imputation (role-anchored centroid voting, **B4**) | Choi, *Training-Free Off-Screen Player Imputation…*, [arXiv 2607.11548](https://arxiv.org/abs/2607.11548); code [nowayfootball/offscreen-impute](https://github.com/nowayfootball/offscreen-impute); learned ceiling: DeepMind Graph Imputer, [Sci Rep 2022](https://www.nature.com/articles/s41598-022-12547-0) | adapt | The **enabler** — turns visible-subset frames into approximate full-team shapes so every method below becomes runnable. Published error 9.7-11.6 m median hidden-player, hidden-zone control error ~halved. **But see honesty trail: those bars were measured on clean simulated-viewport tracking, not our noisy CV tracks — treat as ceiling.** | Low (CPU, open, Metrica-format) | **build (conditional)** |
| 2 | EFPI elastic formation & position ID (template matching + linear assignment) | Bekkers, *EFPI*, [arXiv 2506.23843](https://arxiv.org/abs/2506.23843); open in `unravelsports` | adapt | First named formations (4-2-3-1 vs 3-4-3) + per-player position labels the OT embedding can't give. **But: built/validated only on full 11-outfield commercial tracking; reduced 8-10 templates are red-card states, not per-frame occlusion — can only run on method 1's *output*.** | Low-mod (pip pkg, reuses our Hungarian machinery) | ~~build~~ → **probe** |
| 3 | Shaw-Glickman dynamic formation analysis (phase role distributions + agglomerative clustering) | Shaw & Glickman, Barça Sports Analytics Summit 2019 | adapt | Gold standard for in-poss vs out-of-poss splits + mid-match switches — formations *discovered* not template-matched; separates attack/defence shape ("defend 4-4-2, attack 3-2-5"). | Moderate (no code; numpy/scipy reimpl) | **probe** |
| 4 | SoccerCPD formation & role change-point detection (Delaunay adjacency + g-segmentation) | Kim et al., KDD 2022, [arXiv 2206.10926](https://arxiv.org/abs/2206.10926); code [pientist/soccercpd](https://github.com/pientist/soccercpd) | adapt | Detects **when** shape/roles change ("after going behind, moved to a back three") — a narrative hook nothing of ours has. | Mod-high (change-point stats assume dense sequences; our gaps + imputation noise inflate false positives) | **probe** |
| 5 | Rest-defence structure KPIs (players behind ball at loss, deep-space control) | *Success Factors of Rest Defense*, [JSSM 2023](https://www.jssm.org) (jssm-22-707); *Rule-based counterpressing & rest defence*, [IJPAS 2025](https://doi.org/10.1080/24748668.2025.2473799) | adapt | Fills the one named sub-axis we have **zero** coverage of; pairs with our counter-press primitive. | Moderate, **honesty-limited**: rest-defence players are exactly whom the camera excludes while following attack | **probe** |
| 6 | Pass-network average-position formation proxy (event-only fallback) | Standard event practice (Between the Posts); our validated pass events | yes | Modest but **unconditional** — survives when geometry/ball drop out entirely; independent event-channel cross-check on EFPI/Shaw-Glickman. | Trivial (aggregation + plot) | **build** |
| 7 | DefR zone-responsibility defensive shape proxy | Hudl StatsBomb DefR via xGFC TI-031 (proprietary/paywalled) | no | Marginal — value is per-defender season attribution our identity noise + n=6 can't support; team-level slice already delivered by EFPI role labels. | High relative to yield (no ground truth to validate) | skip |

**Honesty trail (Axis 1).**
- **#1 B4 imputation — corrected, not refuted.** The paper, repo, and DeepMind alternative are all real and correctly cited; the mechanism runs on our trusted-frame subset (visible x,y + pitch geometry only, CPU, causal, no ReID/calibration). What was overstated: the 9.7-11.6 m error bars were measured on a **simulated broadcast viewport over clean full-pitch Metrica tracking** — the "visible" players had zero position error. Our visible players are noisy CV detections on ~26.4% usable frames. The paper's only real-broadcast test (two World Cup clips) reports downstream possession-score shifts, **not** position error vs ground truth. So: budget for degraded accuracy and **re-validate imputation error on our own annotated frames before trusting any downstream metric.** Also confirm lineup-prior identity is stable frame-to-frame (B4 uses an EMA per-player role offset).
- **#2 EFPI — downgraded build → probe.** Real paper, real shipped code, correctly described; it does reuse assignment machinery we already have. But it is validated only on full 11-outfield commercial tracking, reports **no quantitative accuracy** (qualitative WC2022 only), and its reduced templates model red-card states, not frame-level occlusion. It cannot ingest our ~5.9/11-per-team shifting-subset frames — only method 1's imputed output — so any EFPI label inherits 100% of the (still-unvalidated) imputation error. Integrate **only after** the imputation layer has its own accuracy validation.

---

## Axis 2 — Phases of play & game-state

Build-up/attack/block classification, low block vs high press, counter vs possession, score-state
conditioning. Our score-state segmentation uses **crude scoreline buckets** (1-0 at 5' == 1-0 at 85').

| # | Method | Source | Runs on our data | Expected lift | Cost | Rec |
|---|--------|--------|:---:|---------------|------|:---:|
| 1 | Bayesian in-game win-probability as the game-state covariate | Robberechts, Van Haaren & Davis, *A Bayesian Approach to In-Game Win Probability*, [KDD'21](https://dl.acm.org/doi/10.1145/3447548.3467194); [DTAI blog](https://dtai.cs.kuleuven.be/static/sports/blog/a-bayesian-approach-to-in-game-win-probability/) | yes (base subset) | Replaces raw score buckets with a **continuous, time- and strength-aware** WP(t) to condition every phase/style metric and the synthesizer. Our validated goal boundaries are exactly the input. Paper's calibration: ECE 0.011 vs 0.023-0.027; end-of-game ECE 0.002 vs 0.101-0.174. | Low (fit once on open events, CPU ~1-2 d; inference = lookup) | **build** |
| 2 | Defensive block-height + compactness on trusted-geometry frames (low/mid/high) | Pressing-tracking lit ([pressure characteristics](https://www.researchgate.net/publication/366180183); [Barça IH PPDA overview](https://barcainnovationhub.fcbarcelona.com/blog/high-pressing-ppda-metrics/)); own censored-broadcast adaptation | adapt | The single most-requested claim we **cannot** make today: "defends in a low block at X m vs presses high". Note PPDA is unavailable — our spotters emit no tackles/interceptions — so geometry is the only honest route. | Mod-low (pure geometry on existing outputs, CPU) | **build** |
| 3 | Counter vs sustained-possession attack typing (event-only, transition-inspired) | Hobbs, Power, Sha, Ruiz & Lucey, *Quantifying the Value of Transitions…*, [MIT Sloan 2018](https://www.semanticscholar.org/paper/01bdc98b7260ce99189ca7a4100a13626312575b) — full method needs tracking; ours is an event heuristic | adapt | Upgrades the counter-press primitive into **attack-type labels** (fast transition / sustained build-up / direct) — the "quick counters vs possession domination" axis of the brief. With #1: "United counter more when leading." | Low (rule-based over possession segments, CPU ~1 d) | **build** |
| 4 | Event-count playing-style factor profile (direct vs possession, press height, width) | Fernández-Navarro et al., *Attacking and defensive styles of play*, [J Sports Sci 2016](https://www.tandfonline.com/doi/full/10.1080/02640414.2016.1169309); xGFC TI-026 | adapt | Named, literature-backed style axes the territorial OT embedding lacks; season-scale factors stabilise claims n=6 can't. | Low (PCA/FA on public FBref-grade counts + ours, CPU ½ d) | **build** |
| 5 | Possession normalization of defensive/phase KPIs | Phatak et al., *Context is key*, [Sci Rep 2022](https://doi.org/10.1038/s41598-022-05089-y); xGFC 2026-07-16 | yes | Makes shots-conceded / passes-allowed / phase shares comparable across opponents with different possession styles; correct denominator for per-phase rates. | **Trivial** (one line: `KPI / (1 - poss_share)`) | **build** |
| 6 | Flow motifs for possession-style fingerprinting | xGFC TI-029; Bekkers & Dabadghao, *Flow motifs in soccer*, J Sports Analytics 5(1) 2019 | adapt | Circulation vs one-two combination profiles per team/phase; texture the count-based profile (#4) misses. | Low-mod (trivial classifier; real cost = identity-confidence gate) | **probe** |
| 7 | Supervised phase/situation labeling from positional data | *Toward Automatically Labeling Situations in Soccer*, [Front Sports Act Living 2021](https://www.frontiersin.org/articles/10.3389/fspor.2021.725431/full); DFB-Akademie | no | None over our stack — the already-planned Bauer/Anzer/Shaw phase CNN covers it and both need full tracking. | High + blocked on data we can't produce | skip |
| 8 | Game-state-conditioned sequence generation (ScoutGPT-style tokens) | xGFC 2026-07-02; Hong et al., [arXiv 2603.15212](https://arxiv.org/abs/2603.15212) | no | Portable idea (game-state token) already slotted in our possession-archetype plan; #1's WP is the better conditioning variable to feed it. | High (external corpus training; transformer marginal on 4 GB) | skip |

**Honesty trail (Axis 2).**
- **#1 Bayesian WP — verified, data_req corrected.** Method and lift are well-supported by the paper's own numbers (PyMC3 ADVI, CPU ~76 min, 8 seasons). Two corrections: (a) the strength prior is **scraped Elo (clubelo.com)**, not odds/table — buildable but a distinct input; (b) "no tracking" understates it — **4 of 10 features (attacking passes, xT, chance quality, duel strength) need Opta-grade located events**, which our 26.4% geometry can't reliably supply. Build the **base subset** (score, clock, Elo, goals, reds, yellows); label the located-event features **optional-and-currently-infeasible**. The paper shows the simplified model still beats baselines (RPS 0.138 vs 0.134), so this is fine.
- **#3 Counter-vs-possession — verified, reframed.** The Hobbs et al. paper is real but its innovation is **defensive-disorganization scoring from full opponent positions** — which our event heuristic drops entirely. Do **not** call ours a "proxy" of their clustering; frame it as *inspired by the practitioner fast-break heuristic*. Confident 3-way typing needs progression speed + regain location (ball 32-52%, geometry 26.4%), so ~a quarter-to-half of possessions get a full label; the rest fall back to a time-to-shot/pass-count-only heuristic.
- **#5 Possession normalization — verified, two caveats.** Formula exact (`NormKPI = KPI/(1-poss%/100)`). (a) Paper's possession is **time-based**; we substitute a **pass-count-share proxy** — flag it, don't treat as equivalent. (b) Paper validated only on **whole-season aggregates** for blocks and tackles+interceptions predicting xGA (+7.9pp R²); single-match granularity and shots/passes/phase-shares are our extrapolation. The "small but immediate" hedge is honest. (xGFC article's real title is *"You Should Make Football Metrics Smarter Before Modelling"*.)

---

## Axis 3 — Player-player interactions

Passing networks, pitch control, off-ball runs, duels, synchrony. Our stack is **territorial + event-count
only** — no interaction structure at all.

| # | Method | Source | Runs on our data | Expected lift | Cost | Rec |
|---|--------|--------|:---:|---------------|------|:---:|
| 1 | Passing-network analysis w/ network-science metrics (coaching-signature variant) | Buldú et al., *Defining a historic football team*, [Sci Rep 2019](https://doi.org/10.1038/s41598-019-49969-2); *Guardiola, Klopp, Pochettino…* [PMC8569793](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8569793/); Frontiers Psych 2018; Between the Posts | yes | **First interaction layer we have.** Centrality/betweenness/clustering/edge-asymmetry → who-combines-with-whom, build-up hubs, Amorim-vs-opponent coaching-signature. Feeds report v3. | Low (networkx on pass events, CPU 1-2 sessions) — ship with identity-confidence gate | **build** |
| 2 | Flow motifs (3-pass ABAB/ABAC/ABCD profiles) | Bekkers & Dabadghao, J Sports Analytics 5(1) 2019, via xGFC TI-029 | yes | One-two/triangle vs circulation — the quick-passes-vs-domination dimension raw counts miss; compact per-team fingerprint. | Low (counting pass over segmented chains; shares infra with #1) | **build** |
| 3 | Possession-normalized defensive/duel KPIs (context-adjusted) | Phatak et al., [Sci Rep 2022](https://doi.org/10.1038/s41598-022-05089-y), via xGFC 2026-07-16 | yes | Comparability fix across possession styles; honest substitute for duel analysis given we emit no duel events. | Trivial (one line) — do **not** fabricate tackle/interception counts | **build** |
| 4 | Visible-set pitch control + training-free off-screen imputation | Spearman/Shaw pitch control (Friends of Tracking / Metrica) + [arXiv 2607.11548](https://arxiv.org/abs/2607.11548) B4; canonical control needs full commercial tracking (flagged) | adapt | First spatial-dominance surface — control shares at pass/shot moments, "who controls the half-space" vocabulary. Paper reports control error 28-48% of naive baseline in our regime. | Moderate — **validation is the cost**: control over imputed players can't self-certify → **SkillCorner A-League opendata is the gate** | **probe** |
| 5 | Off-ball run identification & classification | Gregory, *Ready Player Run* (Barça IH/OptaPro); SkillCorner run taxonomy; OBSO (Spearman 2018) needs full tracking (flagged) | adapt | Off-ball dimension (runner-passer pairs, who attacks depth when Bruno receives between lines) — invisible to our event stack. | Mod-high — run heuristic + clustering, CPU; **needs 30-50 annotated clips** (Sid's time) | **probe** |
| 6 | Team coordination/synchrony (centroid coupling, stretch index, relative phase) | Duarte et al., HMS 2013; Folgado; Frencken — all assume full-pitch data | adapt | Compact "do teams move as coupled blocks" signal — but largely redundant with OT embedding + planned 5×5 shape graphs. | Low to compute but **confounded**: both teams share one camera frame → synchrony partly measures the cameraman | skip |
| 7 | Learned multiagent off-screen imputation (Graph Imputer class) | Omidshafiei et al., [Sci Rep 2022](https://www.nature.com/articles/s41598-022-12547-0); *Inferring Player Location*, [arXiv 2302.06569](https://arxiv.org/abs/2302.06569); continuous-tracking 2025 (~6.9 m) | adapt | Our `imputation_probe.md`: closed-form floor 3.4 m @2 s / 6.8 m @8 s vs 3.6 m interp floor — learned model has room only at gaps ≥2 s. | High (graph-VAE occupies the one heavy-job slot; 10 external matches thin; win over B4 unproven) | skip |
| 8 | Defensive Responsibility / per-defender duel-zone attribution (DefR) | Hudl StatsBomb DefR via xGFC TI-031 (proprietary/paywalled) | no | Value is per-defender season attribution our identity censoring can't support; team-zone proxy adds little over #3. | High + unvalidatable | skip |

**Honesty trail (Axis 3).** #1's three academic sources all verified real and accurately characterised
(Buldú 2019 = doi:10.1038/s41598-019-49969-2). Ship #1 with an **identity-confidence gate** (edges only
where both endpoints are high-confidence) and an honest edge-coverage stat; validate network pass totals
against the frozen **Sofascore 0.97-1.09x** band. #4 inherits Axis 1's imputation caveat — the probe's
own verdict stands: compute on trusted frames, **validate on SkillCorner (genuine off-screen ground
truth)**, ship only event-anchored snapshots with stated coverage. #6 fails the bar-to-beat test twice
(camera-framing bias + redundancy) → skip.

---

## Axis 4 — Coaching style & team identity

Style fingerprints, manager effects, cross-match consistency, opponent-conditioned behaviour. Our OT
embedding's identity **decays with n** — it cannot make a falsifiable "team identity" claim. Our corpus
has a **structural gift**: reverse fixtures vs the same six opponents, straddling the **Ten Hag → Amorim**
change (sacked Oct 2024; Amorim 3-4-2-1 from Nov 2024).

| # | Method | Source | Runs on our data | Expected lift | Cost | Rec |
|---|--------|--------|:---:|---------------|------|:---:|
| 1 | Pass-network consistency & identifiability (who imposes style on whom) | Herrera-Diestra, …, Buldú, *Consistency and identifiability of football teams*, [Sci Rep 2020](https://doi.org/10.1038/s41598-020-76835-3) | adapt | Answers 3 sub-axes we don't touch: **consistency** (how stable match-to-match), **identifiability** (falsifiable identity, unlike OT), **opponent imposition** (which team imposed its style, per fixture). Our reverse-fixture corpus is near-ideal for it. | Low (networkx, no geometry dependency — sidesteps 26.4% entirely) | **build** |
| 2 | Manager-regime intervention split (Ten Hag vs Amorim), opponent-controlled | Managerial-change lit (Audas/Dobson/Goddard 2002; ter Weel) as a split-sample over our own metrics; xGFC formation piece | yes | Turns **every existing metric** into a manager-effect statement for near-zero marginal modelling. Same-opponent pre/post pairs control for strength. **If fingerprints don't separate the regimes, that's a failed validation of the fingerprints** — a free discriminative-validity test. | Trivial (one metadata field + paired report); n≈3/regime → report **directions, not significance** | **build** |
| 3 | Flow motifs (3-pass) as style fingerprint | xGFC TI-029; Gyarmati, Kwak, Rodríguez, [arXiv:1409.0308](https://arxiv.org/abs/1409.0308); Bekkers & Dabadghao 2019 | adapt | Sequence-structure identity (one-twos vs circulation) neither OT nor phase segmentation captures; most-replicated event-only style discriminator (Barça's ABAB). | Low (shares pass-chain extraction with #1); team-level only, null-model baseline | **build** |
| 4 | Possession-normalized (context-adjusted) defensive & tempo KPIs | xGFC 2026-07-16; Phatak et al., [Sci Rep 2022](https://doi.org/10.1038/s41598-022-05089-y) | yes | Not a new fingerprint — a **correctness fix** to every existing one: raw counter-press/shots-conceded/phase shares are possession-confounded, biasing the whole consistency axis. Apply **first**. | Near-zero (few lines) | **build** |
| 5 | SoccerMix soft-clustered action prototypes (location-direction mixtures + opponent deflection) | Decroos, Van Roy, Davis, *SoccerMix*, ECML-PKDD 2020, KU Leuven DTAI | adapt | Closest published fit to the **full** axis spec: prototype-usage vectors (fingerprint), match-to-match distances (consistency), opponent deflection from season norm (opponent-conditioning) — semantically richer than territorial OT. | Moderate — EM is cheap but 26.4% geometry → small per-match samples; bootstrap stability first | **probe** |
| 6 | Opponent-conditioned event-token transformer (ScoutGPT-style) as synthesizer backbone | xGFC 2026-07-02; Hong et al., [arXiv:2603.15212](https://arxiv.org/abs/2603.15212) | no | Only candidate that generates opponent-conditioned counterfactual behaviour — the synthesizer's end-goal. **Position-token masking trick is portable to our player embeddings today** regardless of the full build. | High (external corpus + training; can't validate outputs against our pipeline events) | **probe** (lift the masking idea; defer training) |
| 7 | Match-level playing-style factor axes (Fernández-Navarro typology, reduced) | Fernández-Navarro et al., J Sports Sci 2016; review: Gu et al. 2025; xGFC TI-026 | adapt | Named style axes — but factor extraction needs **far more than 6 matches**; loadings unstable at our n. Overlaps #4 of Axis 2 (do it there at season scale, not here). | Mod (needs public-feed supplementation for crosses/duels) | skip (subsumed by Axis 2 #4) |

**Honesty trail (Axis 4).** #1 and #3 both build on Buldú-lineage network science (all sources verified);
their strength is that they are **event-only, no geometry dependency**, so the 26.4% constraint doesn't
touch them. #2 is the cheapest manager-effects design possible and exploits a feature unique to our exact
corpus — do it **before** adding any new embedding. #6 is strategically aligned with the synthesizer
roadmap but not with this milestone's data reality; it is effectively a full-event-data method that
**cannot run on our censored broadcast output** — probe = steal the masking idea, defer the rest.

---

## Cross-axis note on CV / footage analysis

The brief's fifth axis (CV methods to improve footage analysis) resolves, for *this* milestone, into a
**single lever**: off-screen player imputation (Axis 1 #1 / Axis 3 #4 / Axis 3 #7). Every spatial method
across all four axes is gated on it, and the honest verdict is uniform — **training-free B4 first,
re-validated on our own noisy CV tracks and on SkillCorner, before any learned imputer or any pitch-control
number ships.** Our detection/tracking/identity stack (E2E-Spot, lRomul, PRTreID, jersey OCR, GS-HOTA
22.85) is already the input those methods consume; the research finding is that the **next CV win is
imputation of the invisible, not more accuracy on the visible.**

---

## United style-analysis v2 — proposed architecture

Combining only the **surviving build recommendations**, sequenced for a 4 GB-GPU laptop (one heavy job at
a time; event-only work is CPU and parallel-safe). Each phase ends with a validation gate; nothing
downstream ships until its gate passes.

### Phase 0 — Event-only foundation (CPU, no GPU, days not weeks)
The cheapest, highest-coverage layer — no geometry dependency, survives when tracking drops out.

1. **Possession normalization** (`KPI/(1-poss_share)`, pass-share proxy flagged) — *apply first*; it is
   the correct denominator for every metric that follows.
2. **Passing-network + consistency/identifiability** (Axis 3 #1 + Axis 4 #1, one build) — centrality,
   edge-asymmetry, cross-match stability, opponent imposition. Identity-confidence gate; validate pass
   totals vs Sofascore 0.97-1.09x.
3. **Flow motifs (3-pass)** (Axis 2 #6 / Axis 3 #2 / Axis 4 #3, one extraction) — circulation vs one-two
   fingerprint, with a randomized-network null.
4. **Pass-network average-position formation proxy** (Axis 1 #6) — event-channel formation visual and
   independent cross-check for Phase 2.
5. **Manager-regime split** (Axis 4 #2) — one metadata field in `data/matches.yaml`; re-express every
   metric above as Ten Hag vs Amorim. Doubles as the discriminative-validity test for the whole stack.
6. **Bayesian WP base subset** (Axis 2 #1) — fit goal-intensity model once on open events; replaces
   scoreline buckets as the game-state covariate for everything.
7. **Counter vs sustained-possession typing** (Axis 2 #3) + **event-count style factor profile**
   (Axis 2 #4, season-scale public data).

**Gate 0:** network pass totals within Sofascore band; motif profiles stable across a team's matches;
manager split shows *some* separation (else the fingerprints themselves are suspect).

### Phase 1 — Geometry on trusted frames (CPU, no imputation yet)
8. **Defensive block-height + compactness** (Axis 2 #2) — visible-outfielder rearmost line, spread,
   ball-to-block distance on the 26.4% usable frames. **Gate 1:** block-height vs hand-annotated
   sequences + per-match coverage stat (broadcast frames the block preferentially during forward passes —
   report the bias).

### Phase 2 — Imputation, the one gated GPU-adjacent investment
9. **Training-free B4 imputation** (Axis 1 #1) — CPU, but its output is the foundation for all remaining
   spatial methods, so it gets its own validation. **Gate 2 (hard):** re-measure imputation position
   error on our *own* annotated frames (not the paper's clean-viewport numbers) and against SkillCorner
   A-League opendata. Confirm lineup-prior identity is stable frame-to-frame (B4's EMA role offset needs
   it). **Only if Gate 2 passes** do the following unlock:
   - **EFPI named formations** (Axis 1 #2) on imputed frames — now a supported input, not out-of-envelope.
   - **Visible-set pitch control** (Axis 3 #4) — event-anchored snapshots at pass/shot instants only,
     coverage stated.

### Deferred probes (decide build/skip from one-match spikes, not committed)
Shaw-Glickman (Axis 1 #3, only if EFPI templates prove too coarse) · SoccerCPD change-points (Axis 1 #4,
gap-robustness spike) · rest-defence KPIs (Axis 1 #5, quantify turnover visibility first) · SoccerMix
(Axis 4 #5, bootstrap stability on the two Liverpool fixtures) · off-ball runs (Axis 3 #5, needs Sid's
30-50 annotated clips) · ScoutGPT position-token masking (Axis 4 #6, lift the idea into the existing
synthesizer, defer training).

---

## What we should NOT build, and why

| Do not build | Axis | Why |
|--------------|:---:|-----|
| **EFPI as a low-cost drop-in** | 1 | Cannot ingest our shifting visible-subset frames; runs only on validated imputation output. Build **after** Gate 2, never before. |
| **DefR / per-defender responsibility** | 1, 3 | Value is per-defender season attribution; our identity censoring + n=6 + paywalled matrices make it unvalidatable. Team-zone slice already covered by EFPI labels + normalized counts. |
| **Supervised full-tracking phase labeling** | 2 | Redundant with the already-planned Bauer/Anzer/Shaw phase CNN; both need full tracking we can't produce. Block-height (#2) is the affordable slice. |
| **ScoutGPT full training** | 2, 4 | Needs thousands of located-event matches (we have 6, no start/end locations); transformer marginal on 4 GB; outputs unvalidatable against our events. Steal the masking trick only. |
| **Team synchrony metrics** | 3 | Inter-team synchrony on one shared camera frame partly measures the cameraman; defensible residue duplicates OT + planned shape graphs. |
| **Learned Graph Imputer (now)** | 3 | Sequencing, not merit: training-free B4 is the floor it must beat, and our probe shows headroom only at gaps ≥2 s. Occupies the one heavy-job slot for an unproven win. Revisit only if Gate 2 shows long-gap extrapolation dominates control error. |
| **Match-level factor typology on n=6** | 4 | Factor loadings unstable below tens of matches; do the season-scale version once in Axis 2 #4, not twice. |

**One-line summary.** Ten event-only builds (Phase 0-1) give United a passing-network + motif + manager-split
+ game-state + block-height fingerprint with **no new models and no GPU risk**; the only heavy bet is
training-free imputation, and it does not ship a single spatial number until it is re-validated on our own
tracks and on SkillCorner. Everything richer is a gated probe, and seven tempting methods are explicitly
off the table because they need full-pitch tracking, per-defender identity, or an external corpus we cannot
honestly supply.
