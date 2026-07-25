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

**Gate-1 amendment: ACCEPTED by the supervisor 2026-07-24.** Sid put the choice to his professor
and the answer was option (b): **the model's pass/fail line is the best CAUSAL baseline
(B5_blend), with offline-linear reported alongside as an oracle ceiling.** Rationale accepted as
given ("go for b as we can validate claims"). Gates 2-4 (calibration, downstream bias reduction,
abstention horizon) stand unchanged. Original proposal text preserved below for the
pre-registration trail.

**BAR RE-FROZEN 2026-07-24 (same day, before any v1 model exists) — two corrections, both
raising the difficulty:**

1. *Wrong split.* The numbers quoted above (0.55 / 2.27 / 5.08 / 9.05 / 17.27 / 15.95, ALL 12.63)
   are full-Game-2. The frozen protocol (B4_MODEL_PLAN 4.1) holds out **Game 2 second half only**,
   where the same frozen B5_blend scores **0.66 / 2.26 / 4.98 / 8.95 / 17.72 / 16.80, ALL 13.24**.
   Comparing v1 against the full-game column would have been apples-to-oranges.
2. *A better causal baseline exists, so it becomes the bar.* Week-1 measured the published
   training-free role-anchored vote (**B6_vote**) — it **LOSES** to B5_blend (ALL 15.64 vs 13.24;
   ahead only at 10-30 s) and is reported as a negative result. But replacing B5's weakest
   component (the slot-OLS prior, 21-35 m alone) with the vote field — same veldecay, same frozen
   tau=4.75 s, blend weights refit on TRAIN only — gives **B7 = veldecay (+) vote**, which beats
   B5_blend in 5/6 buckets with non-overlapping 95% block-bootstrap CIs, on **both** CALIB
   (9.92 vs 12.04, wins 6/6) and HOLDOUT.

**GATE RESULT 2026-07-24 (run once, after freezing): v1 PASSES gates 1 and 2.**
v1 = quantile-GBM residual correction on B7. Holdout ALL **8.10 m vs the 11.46 m bar**
[95% CI -4.27, -2.53]; beats in 5/6 buckets, ties at 0-1s (+0.02 m, CI includes 0 — the anchor is
already accurate to 0.66 m there, so there is nothing to add). Biggest wins where the plan
predicted: 10-30s -35%, 30s+ -26%. Gate 2 **PASS 6/6 at BOTH 50% and 90%** (PICP 45.5-50.9 /
89.2-90.6); the pre-declared ACI remedy was not needed and not applied.
**Independently confirmed** (Fable pass, 2026-07-24): the bar was re-derived from raw Metrica CSVs
and reproduced all seven frozen numbers exactly; anchor and bar are literally the same array (no
inflation possible); every threshold (hyperparameters, conformal k, b*, R_max) traced to
TRAIN/CALIB only. Standing caveats it attached, which travel with the claim: the 0-1s tie, and
that the holdout was opened more than once across the *exploration* (disclosed, and v1's config
was identical across runs — no tuning followed).

**TWO OPEN ISSUES FROM THE GATE RUN (neither silently patched):**

- **Abstention Layer A is mis-specified.** The rule "abstain at and beyond the first bucket where
  v1 stops beating the bar" assumed skill decays with horizon. Ours fails only at the *shortest*
  bucket, so b* = 0-1s and the literal rule abstains on **100% of the holdout** — absurd. The
  defect: the rule conflates "no skill" with "no improvement over the anchor". At 0-1s we have
  excellent accuracy (0.68 m) and merely nothing to add. **Proposed restatement, to be approved by
  the supervisor before it is used for any claim:** where v1 does not beat the anchor, *defer* to
  the anchor (emit its prediction, labelled) rather than abstain; abstention is reserved for
  Layer B (uncertainty too wide). Literal-rule and restated-rule outcomes are both reported in
  results/B4_MODEL_V1.md (restated variant asserts 92.0%, RMSE 8.38 vs bar 11.89).
- **Provenance mismatch:** `VOTE_HALFLIFE_S = 10.0` in code contradicts the 0.04 s that produced
  the frozen bar (and its own "selected on TRAIN" comment). The bar is correct and reproducible at
  0.04 s; the constant needs fixing. Related honest note: shorter half-life is monotonically better
  to the grid edge — at our optimum the published "EMA role offset" is not an EMA.

**TRANSFER RISK — the finding that outranks the win.** v1 places 14.9% of predictions inside the
visible band where a hidden player cannot be (anchor 18.7%, truth 11.5%): it has partly learned the
*simulator's rectangular camera window*. Holdout numbers therefore may not transfer to real
broadcast footage. **M3 (transfer validation on SkillCorner + our own tracks) is promoted ahead of
any model stretch work** — no B4 number ships into a scouting report until it survives that.

**THE GATE-1 BAR IS THEREFORE B7, HOLDOUT (Game 2 H2), per horizon:**

| 0-1s | 1-3s | 3-5s | 5-10s | 10-30s | 30s+ | ALL |
|---|---|---|---|---|---|---|
| 0.66 | 2.09 | 4.37 | 7.55 | 14.91 | 15.16 | **11.46** |

Rationale: the supervisor's decision was "score against the best CAUSAL baseline". B7 is causal and
training-free, so honouring that decision means the bar moves with our knowledge of what causal
methods achieve. Keeping the weaker B5 bar while letting v1 sit on the stronger B7 anchor would let
v1 clear the gate by construction — inflation, not evidence. The anchor swap is justified on the
**CALIB** split (where B7 wins 6/6); the holdout was also inspected during this exploration and that
is disclosed here rather than hidden — the "run once" discipline applies to the v1 gate run, which
has not happened.

**Gate-1 amendment (2026-07-22, dated BEFORE any learned-model result exists — baselines only):**
M2a exposed that offline linear interpolation uses the FUTURE sighting, which a causal imputer
never has; no causal baseline beats it past 3 s and the gap widens with horizon (7.1 m at
10-30 s). Proposed restatement, pending professor sign-off: **gate 1 is scored against the best
CAUSAL baseline (B5_blend) per bucket; offline-linear is reported alongside as an oracle
ceiling, not the pass/fail line.** Rationale is structural (information asymmetry), not
performance-rescue: the model being gated does not exist yet and no model numbers informed this.
If the professor prefers the literal offline bar, we keep it and expect gate 4 (abstention) to
carry the long horizons.

## Prior work reconciliation (found 2026-07-22)

A pre-registered closed-form baseline probe already existed: `ccf63c5` (2026-07-16),
`tools/imputation_probe.py` + `results/imputation_probe.md`, run on OUR OWN tracking as
self-truth (brighton + france_senegal parquets, gaps <= 8 s). Its findings AGREE with M1/M2a:
interpolation is near the noise floor at short gaps; extrapolation is where the gap opens;
structure (centroid_rel there, slot prior here) starts paying at the longest horizons. The new
Metrica setup SUPERSEDES it as the grading harness — external full-pitch truth instead of
self-truth (its own stated caveat), and 30 s+ horizons instead of <= 8 s. The probe still
contributes two things going forward: the **re-entry error** metric (error at the moment the
player reappears — adopt in M2), and a domain-matched self-truth harness for the M3 transfer
check on our real footage. Both code paths stay: probe = measurement on our parquets,
`synthesizer/imputation.py` = the build.

## Baseline results (M1+M2a, frozen fits on Game 1, held-out Game 2; RMSE metres)

| horizon | B1_hold | B2_offline (oracle) | B3_veldecay | B4_slot | B5_blend (causal bar) |
|---|---|---|---|---|---|
| 0-1s | 1.38 | 0.91 | 0.55 | 21.92 | **0.55** |
| 1-3s | 4.48 | 2.63 | 2.42 | 22.63 | **2.27** |
| 3-5s | 8.35 | 4.69 | 5.71 | 23.57 | **5.08** |
| 5-10s | 13.58 | 7.35 | 11.00 | 24.47 | **9.05** |
| 10-30s | 21.80 | 10.17 | 19.76 | 26.91 | **17.27** |
| 30s+ | 19.20 | 12.46 | 18.58 | 33.36 | **15.95** |
| ALL | 16.16 | 8.66 | 14.61 | 26.69 | **12.63** |

Fitted params (Game 1 only): veldecay tau = 4.75 s; blend weight on B3 per bucket =
[1.0, .95, .85, .75, .60, .65]. Findings: slot prior alone is the WORST baseline everywhere
(22-33 m — a linear structural guess has no last-seen memory), yet blending 35-40% of it into
veldecay cuts 30s+ RMSE 18.58 -> 15.95: model v1 needs last-seen memory AND structure jointly.
Frozen fits transferred to Game 2 with zero degradation (held-out slightly better than
in-sample) — no overfitting on the fit set. Game 2 audit: PASS, same full-pitch regime as Game 1.

## Milestones

- **M1 (now → Aug 8):** data acquisition + audit; censoring simulator; baselines 1–2 scored on
  one Metrica match. CPU-only — runs alongside the reverse-fixture GPU chain.
- **M2 (→ Sep 15):** baselines 3–4; model v1 + calibration; held-out gates run.
- **M3 (→ Oct 31):** transfer into our pipeline — impute from our real tracks, uncertainty labels
  rendered in scouting reports (per-player "last seen 12 s ago, position ±8 m").
- **M4 (Nov):** downstream validation, negative results written up, thesis section.

Storage constraint (Sid, 2026-07-21: disk is tight): audit free space before download; both open
datasets are small (<2 GB total); land under `data/imputation/` (gitignored), never in the repo.
