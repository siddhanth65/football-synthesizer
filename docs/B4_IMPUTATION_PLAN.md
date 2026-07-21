# B4 — validated off-screen imputation (started early 2026-07-22, per Sid)

The declared novelty module of `docs/BTP_DECEMBER_PLAN.md`. Original window Oct 1 → Nov 30;
pulled forward with Sid's explicit approval because B1/B2 finished ~2 months early and B4 is
compute-compatible with the running video chain (it trains on tracking data, not video — CPU-first).

## Problem statement

At any trusted broadcast frame we see ~11.8 of 22 players (whole-broadcast ~7.4/22). Published
SOTA practice (GSR winners, arXiv 2504.06357) imputes off-screen players with **linear
interpolation** — unvalidated. FIFA's own study says off-screen estimation fails industry bars.
Novelty claim: a **probabilistic, structure-aware imputer with calibrated uncertainty**, graded
against real full-pitch tracking, that beats published practice — with abstention where the
horizon is too long.

**Regime: EXTRAPOLATION ONLY (pre-declared).** We impute players who left the frame and answer
"where are they now, with what uncertainty". Identity of the 22 comes from lineup priors (B2);
we never invent players.

## Truth data (milestone 1 audits this before anything is built)

| Source | Role | To verify |
|---|---|---|
| Metrica Sports open data (3 matches, fixed camera) | PRIMARY truth candidate: full-pitch, all 22, 25 fps | confirm truly full-pitch, schema, license |
| SkillCorner opendata (9 matches, broadcast) | domain-transfer check on real broadcast tracking | their points are flagged extrapolated/detected — NOT raw truth; audit flags |
| StatsBomb 360 freeze frames | censoring-geometry calibration (visible-area polygon) | same-censoring as our footage |
| SoccerNet GSR valid split (in hand) | our-pipeline transfer target | broadcast-visible only — never truth for off-screen |

Validated-or-nothing: if the audit shows Metrica is not genuinely full-pitch, that finding gates
everything downstream and we re-plan the truth source before writing a model.

## Design

Censoring simulator: virtual broadcast camera over full-pitch truth (pan/zoom following the ball,
calibrated against our measured visibility stats: ~11.8/22 trusted-frame, 26.4% whole-broadcast
geometry yield, SB360 visible-area polygons). Mask → impute → score vs the hidden truth.

Baselines (pre-registered, all four before any model):
1. last-seen hold
2. **linear interpolation between sightings** — the published-SOTA bar to beat
3. velocity extrapolation with decay
4. formation-slot prior (role mean position | ball position)

Model v1 (structure-aware): predict off-screen player position from last seen state (pos, vel,
time-since-seen), ball position, visible-team structure (centroid, spread, line heights, phase),
and role slot. Probabilistic output (mean + covariance, or grid heatmap). Start with the smallest
thing that can beat linear interp (GBM / small MLP); literature anchor and upgrade path:
Graph Imputer (Omidshafiei et al., soccer trajectory imputation).

Metrics: RMSE binned by time-since-last-seen; calibration = empirical coverage of 50%/90%
predictive regions; downstream = does imputation reduce the measured biases (line-height 16.4 m
raw, possession undercount) on the oracle matches.

## Gates — to pre-register with the professor BEFORE model results exist

1. Beat linear interpolation RMSE in every time-horizon bucket on a held-out match.
2. Predictive-region coverage within ±5 points of nominal (50%/90%).
3. Downstream: measurable bias reduction on ≥1 oracle-validated team metric.
4. Abstention horizon: beyond the horizon where gate 1 fails, the imputer must say "don't know".

## Milestones

- **M1 (now → Aug 8):** data acquisition + audit; censoring simulator; baselines 1–2 scored on
  one Metrica match. CPU-only — runs alongside the reverse-fixture GPU chain.
- **M2 (→ Sep 15):** baselines 3–4; model v1 + calibration; held-out gates run.
- **M3 (→ Oct 31):** transfer into our pipeline — impute from our real tracks, uncertainty labels
  rendered in scouting reports (per-player "last seen 12 s ago, position ±8 m").
- **M4 (Nov):** downstream validation, negative results written up, thesis section.

Storage constraint (Sid, 2026-07-21: disk is tight): audit free space before download; both open
datasets are small (<2 GB total); land under `data/imputation/` (gitignored), never in the repo.
