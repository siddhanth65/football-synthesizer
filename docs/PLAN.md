# From Broadcast Video to Tactical Foresight — Final Plan

### A relational, multi-substrate model of attacker instinct and opponent-conditioned team identity,
### delivering an auto-generated FIFA-style team report.

> Combined single pipeline (the two earlier "parallel tracks" are merged). **First focus: the
> freeze-frame generator.** Novelty pursued: **C3 (fixed attacker model)** and **C5 (opponent-
> conditioned synthesizer)** — both if time allows. End deliverable: a FIFA-EFI-style report,
> generated for *any* team, that both describes and *predicts* how they play.

---

## Thesis

> *A single relational model, fed by a broadcast freeze-frame generator and grounded in public
> StatsBomb 360 + FIFA Enhanced Football Intelligence, can repair the attacker-prediction failures
> that freeze-frames alone caused, and be aggregated into opponent-conditioned, calibrated team-style
> predictions — packaged as a FIFA-style report — validated against how teams actually play.*

The CV generator is the **enabling front-end** (reproduce-and-adapt from open SoccerNet work). The
**novelty budget goes to the attacker + synthesizer layer.**

---

## Why combine the tracks (the footage reason)

The WC26 is underway (live footage is sparse/rights-locked), so we source **completed** competitions
with both broadcast video **and** public ground truth:

| Source | Footage | StatsBomb 360 | FIFA EFI report | Role |
|---|---|---|---|---|
| **FIFA World Cup 2022** | broadcast (YouTube) | partial | **yes** (per-match PDFs) | primary train+validate (EFI oracle) |
| **UEFA Euro 2020 / 2024** | broadcast | **yes** | no | train + style |
| **Copa América** | broadcast | partial | no | breadth / transfer |
| **Friendlies / WC26 as it airs** | broadcast | no | (some) | demo / live |

Because data sources differ in completeness, the pipeline is **one flow with a substrate-aware,
masked freeze-frame contract** (novelty **C1**) — not separate code paths.

---

## The six contributions

| # | Contribution | New (on open data) | Validated by |
|---|---|---|---|
| **C1** | One model, three substrates (360 / broadcast-CV / tracking) via a masked freeze-frame contract | yes | leave-one-substrate-out + cross-gender transfer |
| **C2** | Orientation/velocity-conditioned off-screen completion with a learned formation prior | yes | beat AgentImputer ~6.9 m; downstream EFI/pitch-control fidelity |
| **C3** ★ | Per-player **run + receiver** heads trained on real dense video tracks + EFI off-ball labels | yes | run RMSE / hit@3m & receiver top-k **vs old 360 baselines** |
| **C4** | Instinct as a calibrated population finding + counterfactual run optimisation | yes | deviation calibration; success-gradient ablation |
| **C5** ★ | Opponent-conditioned, distributional tendency prediction (the report's forecast) | yes | leave-one-match/team-out backtest; CRPS/Brier + reliability; **skill over team-average** |
| **C6** | FIFA EFI as a public validation oracle for CV/relational metrics | yes | per-metric CV-vs-EFI correlation |

★ = the two the supervisor prioritised.

---

## Technical architecture

### 1. Unified freeze-frame contract (C1) — `generator/contract.py` (built first)
Every substrate → one frame on a 120×80 pitch, attacking left→right. Each player node carries the
**superset** of features + a **per-feature observability mask** + a **substrate tag**:
`(x,y)` always; `(vx,vy)` for CV/tracking (masked for 360); `(sinθ,cosθ)` body orientation from pose
(masked when unavailable); role flags; distances/angles; occlusion-aware shot-window angle. Missing =
**masked, not zero-filled**. Down-projects to the existing model's `(x,y,teammate,actor,keeper)`
frame so the **already-trained GAT** runs immediately via the `cv_bridge`/`ood_demo` contract.

### 2. Freeze-frame generator (FIRST FOCUS) — `generator/`
Reproduce-and-adapt `sn-gamestate`/TrackLab: detection + **BoT-SORT** tracking + **re-ID/jersey#** +
**PnLCalib** calibration with a **per-frame confidence gate** (reject high-reprojection-error frames →
"no wrong frames" is *measurable*) + pose→orientation. `to_frames.py` emits contract frames; quality
gate keeps trustworthy ones. Honest anchor: **GS-HOTA** on SoccerNet-GSR.

### 3. Off-screen completion (C2) — `generator/complete.py`
Conditional set-imputation of the unseen players from visible players' **position + orientation +
velocity** and a learned **team formation prior**; Gaussian-mixture per missing node + Sinkhorn set
loss; trained on full 360/tracking with simulated broadcast masking (30 m-of-ball).

### 4. Rebuilt attacker heads (C3 ★) — `attacker/`
Per-player **node-level run head** = displacement distribution at t+1.5 s on *real* tracks (fixes the
360 nearest-neighbour label noise that capped the old head at 6.33 m). **Receiver head** = node
softmax with orientation+velocity + EFI movement-to-receive/offers labels. Plus **instinct (C4)**:
predicted-optimal-vs-actual deviation + counterfactual run optimisation.

### 5. Team identity fingerprint — `fingerprint/team_identity.py`
Aggregate per-possession GAT reads → the `team_metrics` schema + classical style axes (formation/role
occupancy, line height, width, directness, L/C/R attack share, PPDA). One vector per team-match → a
team-history encoder → `z_T`.

### 6. Opponent-conditioned synthesizer (C5 ★) — `synthesizer/`
`ŷ = g(z_T, z_O, ctx)`; outputs **distributions** over tendencies (block height/press, attack-channel
split, directness, formation). Tier A marginal `P(y|z_T)` → Tier B matchup `P(y|z_T,z_O)`. Headline
test: **does Tier B beat Tier A** (does the opponent term carry signal)?

### 7. The FIFA-style report (the product) — `report/`
Render a per-team PDF/HTML mirroring FIFA EFI (phases of play, line breaks, defensive line height,
receptions between lines, pressure, attack channels) **plus** the synthesizer's forecast for a chosen
opponent, with calibrated uncertainty. This is what the supervisor sees.

---

## Evaluation (rigor first)
Match-level splits (no same-match leakage); **temporal** split for the synthesizer (predict future
from past); leave-one-team-out for generalisation. Metrics: GS-HOTA (generator); run RMSE/hit@3m +
receiver top-1/3 vs old 360 & nearest-teammate (attacker); imputation position error (beat 6.9 m) +
downstream fidelity; **CRPS/Brier + reliability + skill-over-team-average** (synthesizer);
split-conformal prediction sets for calibrated uncertainty; multi-seed CIs; report nulls.

## Honest ceilings (state up front — no "Opta-accurate" overclaim)
GS-HOTA SOTA **63.81** (arXiv:2504.06357); broadcast position **RMSE 1.68–16.39 m** vs TRACAB **0.08 m**
(arXiv:2508.19477); **~10–14 of 20** players visible (our real clip ~5.5); imputation **~6.9 m**
(arXiv:2302.06569); orientation **~27°** (arXiv:2003.00943). Freeze frames are **partial, metre-scale**
→ great for on-screen attacker + aggregated style, **not** a per-frame Opta replacement.

## Change triggers
Own-footage **GS-HOTA < 50** or homography fails **>30%** → make public **360 the primary input**, CV a
demonstrator. Imputation error too high → rely on **multi-match distribution aggregation**.

## Reuse map
`football-state-of-play`: `results/checkpoints/gnn.pt`, `eval/cv_bridge.py`, `eval/ood_demo.py`,
`data/graphs.py`, `models/{gnn,heads}.py`, `eval/team_metrics.py`, `eval/tracking_bridge.py`.
`cv-football`: `extract_positions.py` (replace homography with PnLCalib).
External: `SoccerNet/sn-gamestate`, PnLCalib, BoT-SORT, TrackNet/WASB, `sn-reid`, `socceraction`,
FootBots, AgentImputer; oracle = FIFA EFI Explanation Document.

## Milestones (generator-first)
1. **Contract + tests** (done first — the foundation). 2. **Generator** reproduce-and-adapt + GS-HOTA.
3. **360 backbone** wired to the trained GAT through the contract. 4. **Attacker rebuild (C3)** vs old
360. 5. **FIFA EFI parse + validate (C6)**. 6. **Fingerprints**. 7. **Synthesizer (C5)** + backtest.
8. **FIFA-style report** generation. 9. Off-screen completion (C2) + instinct (C4) as time allows.

## Caveats
SoccerNet 2022 vs 2023 HOTA not comparable; several cites are preprints; broadcast-accuracy from a
single WC22 study; 360 is itself broadcast-derived; opponent prediction → calibrated distributions,
not point predictions.
