# Backup project plans (2026-07-24, requested by Sid)

Fallback scopes if the main thesis (opposition scouting pack for ManU) under-delivers by December.
Each is scored on: reuse of what exists, models needed, risk, and fit with the review-1 checklist
(external validation, honest error bars, touchable demo, novelty).

## Asset inventory (what any backup inherits)

- 12 processed EPL matches: tracks on 105x68, ball 32-52%, Viterbi possession, validated events
  (E2E goals 14/16 + splits; BAS passes 10/12 in frozen 0.97-1.09 band), identity 3->9 matches
  (PRTreID precision arm + koshkina recall arm), 6 home/away pairs spanning a manager change
- Truth/benchmark data: Metrica full-pitch (2 games) + calibrated censoring simulator +
  imputation baselines; SkillCorner opendata; StatsBomb open events; FBref/Elo/Sofascore caches;
  SoccerNet GSR valid split + official evaluator (our arc 14.76 -> 22.85)
- Models in hand: football-YOLO, ByteTrack, PnLCalib, TrackNet v6 (fine-tuned), E2E-Spot, BAS
  (lRomul), Koshkina jersey OCR (86.13%), PRTreID, OSNet, Bayesian WP (ECE 0.042)
- Method assets: frozen-threshold validation discipline, comparative-claims tier, team-mapping
  screen, claims-audit workflow, retraction trail, style-research synthesis (105 xGFC articles,
  107 sources, ranked methods)
- Constraints: 4 GB GPU, free LLMs only, lawful supplied footage only

## The backups, ranked

### 1. Censoring-bias correction atlas (strongest standalone science)
**Claim:** every standard tactical metric (possession, block height, compactness, PPDA-like
counts, pass shares) is systematically biased by broadcast censoring; we measure each bias
against full-pitch truth and publish correction factors + an open tool.
**How:** Metrica truth + our calibrated virtual camera -> compute each metric on censored vs
full data -> bias curves by metric x game-state x camera style; validate transfer on SkillCorner;
cross-check with our own measured biases (block +5.2 m broadcast bias, possession undercount,
16.4->7.2 m line de-bias). **Models:** none new (CPU stats). **Risk: LOW** — data local, method
proven on two metrics already. **Novelty:** real; nobody publishes correction factors for
broadcast-derived analytics. Fits CVSports/MLSA. Reuse ~80%.

### 2. B4 imputation as the whole thesis
**Claim:** validated probabilistic off-screen imputation with calibrated uncertainty + abstention,
beating published practice (linear interp) under a pre-registered causal bar.
**How:** M1/M2a already done (simulator, 5 baselines, held-out); build model v1 (memory+structure),
calibration, SkillCorner transfer, downstream bias-reduction demo on our matches.
**Models:** GBM/small-MLP first; Graph-Imputer-style GNN as stretch. **Risk: MEDIUM** — the model
may not beat the causal blend at all horizons; but even a negative result + abstention map is a
defensible thesis under our methodology. Reuse ~60%. Blocked only on the prof gate answer.

### 3. Same-kit identity paper (results already exist)
**Claim:** breaking the same-kit re-ID wall on a public benchmark: jersey-gate + PRTreID relink +
propagation arc 14.76 -> 22.85 GS-HOTA (externally graded), gate recalibration with pre-committed
operating points, disagreement-flag collapse 18/20/44 -> 2/2/1.
**How:** consolidate + extend (T-DEED probe, propagation variants, per-component ablation table).
**Models:** already in hand. **Risk: LOW** — the results are on disk today; work = ablations +
writing. Reuse ~95%. Weakness: less "product", pure CV-benchmark story.

### 4. Rematch forecasting study (C5 expanded to the core)
**Claim:** how much of a rematch is predictable from one prior broadcast meeting? Forecast rematch
metrics + result distribution from first-leg features (possession identity r=+0.83 is the
already-proven signal), scored against actuals + Elo/bookmaker baselines.
**How:** 6 pairs backtest + season-level features (FBref/Elo/WP); honest n-small framing
(proof-of-signal). **Models:** logistic/GBM, CPU. **Risk: MEDIUM-HIGH** — n=6 pairs limits
claims; but pairs well with backup 1 or the main thesis. Reuse ~70%.

### 5. Pipeline-as-product on a new domain (e.g. ISL/college footage)
**Claim:** the whole stack generalizes to any lawful broadcast — demo on Indian football.
**How:** re-run chain on 2-3 new-domain matches Sid supplies; measure what transfers (detector,
kits, calibration) and what breaks. **Risk: HIGH** — new kits/cameras can break detection,
calibration, OCR; no oracle as rich as Sofascore for validation. Reuse ~85% of code, 0% of
validated numbers. Only worth it as an add-on chapter, not a pivot.

## Recommendation

- Primary fallback = **#1 (bias atlas)**: lowest risk, real novelty, CPU-only, and it UPGRADES
  the main thesis even if nothing fails (the correction factors feed the scouting pack's error
  bars). Can start any time the GPU is busy.
- If the December crunch hits: **#3 (identity paper)** is the "already done, just write it"
  parachute.
- #2 stays on its current track (gated on prof). #4 is a natural chapter of the main thesis
  rather than a standalone. #5 only if Sid wants a demo-wow chapter and supplies footage.

None of these require abandoning the ManU corpus — every backup consumes it.
