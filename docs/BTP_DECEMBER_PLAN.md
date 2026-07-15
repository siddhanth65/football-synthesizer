# BTP December Plan — research-grade by Review 1 (2026-07-14)

Fine-grained plan for the December 2026 review, built on: the verified project state (STATUS.md,
`docs/CAPABILITY_LEDGER.md`), the 2026-07-14 external-audit verification, and a cited research sweep
of academic SOTA + industry practice. Confidence labels: **[verified]** = adversarially verified
against primary sources; **[captured]** = extracted from a primary source, formal verification
pending (research run was quota-interrupted; resume queued). Companion: `docs/PL_PIVOT_PLAN.md`
(semester frame), `docs/CV_EXPLAINER.md` (how the pipeline works).

---

## 1. The thesis positioning (the single most important decision)

**The field has already formalized our exact problem.** SoccerNet **Game State Reconstruction
(GSR)** — single moving broadcast camera → 2D pitch positions + role + team + jersey number for
every person, i.e. "a video-game-like minimap" — is a recognized academic task with a public
dataset, an official baseline, and a dedicated metric (**GS-HOTA**) [verified: arXiv 2404.11335,
CVPRW'24; arXiv 2409.10587].

This is a gift, not a threat. It gives us:

- **A task name reviewers recognize** — we are a GSR pipeline extended with validated tactical
  metrics, opponent conditioning, and guardrailed reporting.
- **A calibration of credible claims** — the 2024/2025 challenge SOTA is **GS-HOTA 63.8–63.9 vs
  baselines of 23–29** [verified: arXiv 2508.19182]. Even the best published broadcast-only
  pipeline recovers well under two-thirds of the joint localization+identity score. Our "~37% of
  frames yield trusted geometry" story is the same physics, honestly measured.
- **External validation of our architecture** — the winning 2025 GSR pipeline is modular and
  mirrors ours (YOLO detection, motion+ReID tracking, multi-frame keypoint calibration,
  jersey-colour clustering for team ID), and reads jersey numbers with a **vision-language model
  prompted on crops** — which independently validates our close-up-anchored Layer-2 jersey plan
  [verified: arXiv 2508.19182].
- **A defensible novelty direction** — published GSR winners handle off-screen/occluded players
  with **plain linear interpolation**; a *validated* off-screen imputation/uncertainty module goes
  beyond the published SOTA [verified 3-0: arXiv 2504.06357]. Industry (FIFA-co-authored validation
  study) independently names off-screen estimation as *the* open problem: commercial broadcast
  trackers hit 0.44–1.14 m RMSE on detected players but degrade to **4.6–12.2 m when the player is
  off-screen**, against an industry bar of ~1 m [captured: arXiv 2508.19477]. FIFA runs a formal
  **Broadcast EPTS** certification (validation chain: broadcast system → VisionKit → VICON);
  SkillCorner's single-camera product holds FIFA Basic certification [captured: skillcorner.com,
  inside.fifa.com].

**Thesis claim (one sentence):** *Uncertainty-aware tactical inference from partially observed
broadcast video: a validated, evidence-gated pipeline that measures exactly what a broadcast
supports, proves where it stops, and reports nothing it cannot ground.* Every existing asset —
the gates, the abstentions, the possession-bias proof, the retraction discipline — is a feature
of this claim, not an embarrassment.

## 2. Where we stand (post-verification, honest)

| Asset | State |
|---|---|
| CV pipeline (detect→track→calibrate→ball→metrics→facts→gated report) | Working end-to-end, WC + PL; ~260 tests |
| De-biased defensive line vs FIFA | **Restated 2026-07-14: raw 16.4 m → ~7.2 m held-out** (5.5 m was in-sample; contamination caught, measured +1.7 m, docstrings fixed) |
| Pass volume (PL) | recall proxy ~48%, team-symmetric (0.009) — relative claims defensible |
| Possession | proven biased-uncorrectable without events; renamed + caveated (thesis material) |
| Ball coverage | 51.9% post-link full match (carry-over lever, LOO-validated 0.33 m) |
| C3 run/receiver module | **bug fixed + re-evaluated 2026-07-15: fix RESTORES the headline** — run RMSE 4.35/3.83 m, receiver top3 0.93/0.81 (brighton/senegal), clearing old baselines; pre-fix numbers withdrawn; small receiver test sets (48/182), half-time direction confound queued |
| C5 opponent model | reframed honestly: explanatory post-match regression (n=8), NOT yet a pre-match forecast — upgrade path in §4 |
| Jersey identity | probe says viable (close-ups carry legible numbers + names); SoccerNet jersey-2023 data local (2,638 labelled tracklets), unconsumed yet |
| Eval stubs | `eval/gsr_score.py` (GS-HOTA) and `eval/fifa_validate.py` are stubs — **filling the GS-HOTA one is the top credibility item** (§4, B1) |
| Repo | pushed to private GitHub (siddhanth65/football-synthesizer), checkpoint bf150f7 |

## 3. Data plan (all free, no paid licenses)

| Source | What | Size / access | Use |
|---|---|---|---|
| Own PL Archive footage (user-supplied) | Man Utd 24/25 full replays | ~38 matches, local | The corpus. Semester 1: 10–15 matches |
| **SoccerNet-GSR** | 200×30 s broadcast clips, 2.36 M positions with role/team/jersey, 9.37 M pitch-line points | **Free, NO NDA/password** (pip `SoccerNet` or TrackLab) [verified] | **External benchmark**: run OUR pipeline on their valid split → report GS-HOTA vs baseline (29) and SOTA (63.9). Also: calibration training data if needed |
| SoccerNet jersey-2023 | 1,427 train / 1,211 test jersey tracklets | Already local (`data/soccernet/`, gitignored) | Layer-2 jersey model training/eval |
| SoccerNet NDA content (raw full broadcasts) | videos | Password on file (never committed) | NOT needed now; only if event-spotting (Layer 3) requires video |
| **StatsBomb open data** (`statsbombpy`) | events + **360 freeze-frames** (`data/three-sixty/<match_id>.json`): every player **visible in the broadcast frame** at event moments, teammate/actor/keeper flags | Free for research [captured] | **Like-for-like freeze-frame validation**: SB360 has the SAME visibility censoring as broadcast CV → validate our freeze-frame *distributions* (count-visible, shape stats) against a pro annotator's. NOT full-pitch ground truth |
| RSOS "continuous tracking from discrete 360" | code converting SB360 → approximate continuous tracking, ~200 matches | Free [captured: RSOS 12:251175] | Substrate/prior for the imputation module (B4) |
| **SkillCorner opendata** | **10 matches** A-League 24/25 **broadcast tracking** (supersedes the 2019/20 9-match release) | Free on GitHub [captured] | The closest thing to full-pitch ground truth from broadcast: off-screen imputation validation + cross-checking our tracking error norms |
| Sofascore (via `tools/oracle.py`) | per-fixture aggregates (+ per-player stats, shot maps) | Free, cache-first, proven | Match-level oracle gates; per-player validation once Layer 2 lands |
| Understat | shot-level xG, PL covered | Free | Shot-detector validation later (Layer 3) |

## 4. The workplan to December (fine-grained)

### B0 — Integrity closeout (now → Jul 20) *mostly done*
- [x] External-audit verification (all 7 claims adjudicated with evidence)
- [x] Line-height headline restated (7.2 m held-out); false docstrings fixed
- [x] C3 chunk-aware fix implemented + regression tests green
- [x] C3 before/after re-evaluation on brighton + france_senegal — fix restores the headline
      (receiver top3 0.16→0.93 / 0.14→0.81; run RMSE 6.15→4.35 / 4.43→3.83 m)
- [x] README rewrite (GSR positioning, honest numbers, verified quickstart)
- [ ] Commit + push the integrity fixes with an honest message (correction trail = thesis hygiene)

### B1 — External benchmark + live-play filter (Jul 20 → Aug 15)
1. **GS-HOTA harness** (fills `eval/gsr_score.py`): download SoccerNet-GSR valid split; adapt our
   pipeline output to GSR format; score with the official TrackLab evaluator — never reimplement
   the metric. Report per-component (localization vs identity) so the no-jersey-yet state is
   explicit: expect decent LocSim, near-zero jersey IdSim before Layer 2 → the Layer-2 lift then
   becomes *measurable on an external benchmark*. Deliverable: one table, our score vs baseline
   (29.01) vs SOTA (63.90). GPU: batchable on the laptop (200×30 s), faster on cluster.
2. **Live-play filter** (Phase B ingestion stage 0): shot-type classifier (wide/close/replay/
   graphic) gating extraction. Already re-priced honestly: it fixes denominators and cuts wasted
   compute (~59% of frames), it cannot create geometry. Measure: reported yields on live-play
   denominator; pass-recall on live-play-conditioned footage (does it clear 50%? — pre-registered
   question, either answer is a result).
3. **C5 → true pre-match forecast v0**: predict opponent def-line from *their prior matches* (Utd
   corpus gives every opponent twice; league-wide priors from fact stores), then forecast Utd
   tendencies from predicted-not-realized opponent depth; LOMO-backtest vs Tier-A. This converts
   the audit's "explanatory regression" criticism into the Semester-2 arc.

### B2 — Layer 2: player identity (Aug 1 → Sep 15, overlaps B1)
1. Jersey-number model on SoccerNet jersey-2023 (local): baseline = the challenge-winning recipe
   class; ALSO trial the 2025 GSR winner's approach — a small **VLM prompted on crops** (free-tier
   via Ollama; cluster for batch) [verified this is SOTA practice]. Held-out on their test split
   (1,211 tracklets) → a citable standalone number.
2. Close-up anchor propagation (the probe's strong path): general detector on close-up frames
   (the football-YOLO is blind there — known gotcha), read number+name, propagate along track ids
   across cuts. The cross-cut linking problem is the research meat; ReID (OSNet-class) is the tool.
3. Wire number → squad list → name (Utd roster is one squad all season); per-player metrics gated
   exactly like ball metrics (pre-declared ID-confidence gate + abstention).
4. Measure: GS-HOTA IdSim lift on the external benchmark + per-player validation vs Sofascore
   player stats on our corpus.

### B3 — Season scale (Sep 1 → Oct 31, overlaps B2)
- Batch ingestion of 10–15 Utd matches (resumable runner exists); one fine-tune amortizes.
- Every match: fact store + oracle gate + report v2. Error bars across N matches; home/away
  repeat-measurement consistency (validation axis needing zero external data).
- 4 GB laptop: one match ≈ overnight; cluster shifts this to days-not-weeks. Risk hedge: 10 is
  enough for Review 1; 15 is stretch.

### B4 — The novelty module: validated off-screen imputation (Oct 1 → Nov 30)
The one place we can exceed published GSR SOTA on a real research axis [captured: SOTA uses linear
interpolation off-screen; FIFA study names it the open problem]:
1. Revive `generator/complete.py` (currently a stub) as a *probabilistic* imputer: predict
   off-screen player positions with uncertainty (start: velocity/formation-prior Kalman or the
   Graph-Imputer/DASE family; broadcast shows on average only ~12.8/22 players [captured: Nature
   s41598-022-12547-0] — our measured ~37% yield is the same physics).
2. Validate three ways, all free: (a) held-out masking on our own tracks (hide visible players,
   predict them); (b) SkillCorner 10-match broadcast tracking; (c) SB360 freeze-frame
   distributional agreement (same-censoring comparison) + the RSOS discrete→continuous code as a
   prior/baseline.
3. Ship only if it beats linear interpolation with honest error bars — evidence-gated like
   everything else. Downstream: recompute shape metrics on imputed frames, report the delta
   (this was P2's original intent, audit §4.1).

### B5 — Thesis assembly (Nov 15 → Dec review)
- Review-1 pack: WC-validated methodology chapter (incl. the correction trail: ball retraction,
  possession reclassification, line restatement — the *discipline* is the story), PL season
  results with error bars, GS-HOTA external benchmark table, Layer-2 named-player demo, C5
  forecast v0 backtest, demo video (tooling exists).
- Venue targets [captured]: **CVSports @ CVPR 2026** (exact topical match: tracking, calibration,
  position estimation, tactics), **MLSA @ ECML-PKDD 2026**, MIT Sloan abstract (long shot,
  worth one evening). Also the StatsBomb conference. A workshop paper draft doubles as the thesis
  core — write once.
- LLM-report angle if space permits: our guardrail (100%/100%) is well-motivated by the measured
  LLM factuality gap (high FactScore, entailment only 60–72%) [captured] and aligns with the
  wordalisation literature (arXiv 2504.00767).

## 5. What makes December read as "research-grade" (checklist)

1. **External benchmark, not just self-report**: GS-HOTA on SoccerNet-GSR with the official
   evaluator, positioned against the published baseline and SOTA. *(B1 — the single highest-value
   item.)*
2. **Baselines for every claim**: linear interpolation vs our imputer; Tier-A vs Tier-B(true
   forecast); raw vs de-biased line (held-out); v5 vs v6 detector.
3. **Pre-registered gates** (we already do this — keep receipts): the 50%-recall gate question for
   relative-vs-absolute claims must be settled with the prof *in advance* — it is on the open list.
4. **Error bars + negative results**: possession-bias proof, pose-carry negative, contamination
   restatement — presented as findings, not confessions.
5. **The correction trail in the open**: git history now exists; commit fixes with honest messages.
6. **A demo a reviewer can touch**: one command → one match → gated report with named players.

## 6. Pivot options (pre-decided fallbacks, not surprises)

| Risk | Trigger | Pivot |
|---|---|---|
| GS-HOTA harness heavier than expected | >2 weeks | Report LocSim-only GSR eval + our internal gates; still external |
| Jersey model underperforms on our footage | held-out <~60% tracklet accuracy | Ship close-up-anchor-only identity (per-match partial naming, gated); the probe already supports it |
| Cluster access slips | Sep | Laptop-scale: 10 matches, defer imputation training to Sem 2 (v5 fine-tune precedent: laptop-feasible) |
| Imputation doesn't beat linear interp | B4 validation | Publish the negative honestly + keep the uncertainty quantification (still novel vs GSR practice) |
| Live-play recall stays <50% | B1 measurement | Adopt the pre-committed relative-claims bar (symmetric capture) — decision needed from prof EITHER WAY |
| C5 true-forecast overfits at n≈20–30 | LOMO | Report explanatory + forecast side by side; forecasting matures in Sem 2 |

## 7. Standing constraints (unchanged)

4 GB GPU laptop, one heavy job at a time; cluster = accelerator not dependency; free-tier LLMs;
lawful user-supplied footage only, never redistribute; validated-or-nothing; commit only when
asked; author Sid, no AI attribution.
