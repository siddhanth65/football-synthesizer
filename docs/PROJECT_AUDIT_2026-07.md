# PROJECT AUDIT — 2026-07-07

Full-stack audit: architecture (Sonnet agent, whole-repo sweep), SOTA landscape (Opus agent, web
research), football-theory→metric codex (Opus agent, web research). This document is the **anchor** —
every future work item should trace to a section here or explicitly amend this file.

---

## 1. Verdict (one paragraph)

The pipeline is real and validated (CV front-end ≈ 2022-baseline tier using strong external models;
defensive-phase line heights within 2–4 m of FIFA; C5 opponent model +43%/+40% LOMO skill on n=8), and
the report layer is **already architecturally SOTA-aligned** (grounded "wordalisation" + LLM narration is
the published credible pattern — arXiv 2504.00767). The two things holding it back are **not** "more
matches": (a) **validity** — every shape metric is computed from ~6/11 visible players with no
imputation, a systematic bias, and (b) **capability** — no event layer, which locks us out of the entire
VAEP/EPV/OBSO value ecosystem. Secondary: ~80% of the implemented metric inventory is orphaned from the
product path; the predictive layer is one-feature OLS; there is no match registry, so every new match
means hand-editing 5 files.

## 2. Current state — honest levels

| Layer | What we have | Tier vs SOTA |
|---|---|---|
| Detection/tracking | YOLO + ByteTrack + jersey-colour KMeans | 2022 baseline (SOTA: Deep-EIoU + GTA, PRTreID joint reID+team+role) |
| Calibration | PnLCalib + hand-rolled homography | Good (SOTA: No Bells Just Whistles — pretrained, 3D, better central views) |
| Ball | Fine-tuned TrackNetV2 v4 (P .99 / R .68 / ~0.2 m) | Solid for consumer HW |
| Off-screen players | **Nothing** (generator/complete.py stub) | **The validity gap.** SOTA: DeepMind Graph Imputer (Nature SR 2022), DASE (arXiv 2408.10878) |
| Events | **Nothing** | **The capability gap.** SOTA: T-DEED spotting + ELASTIC sync (arXiv 2508.09238) → socceraction VAEP |
| Metrics | Rich (pitch control, xT/OBSO-lite, transitions, PPDA proxy, roles, synchrony, set pieces) but **mostly orphaned** | Definitions are ~2019-tier; wiring is the bigger problem |
| Phases/formation | Hand-tuned thresholds (one-match calibration, hardcoded) | SOTA: EFPI (2506.23843), SoccerCPD (2206.10926) — principled, CPU |
| Synthesizer | 1-feature OLS, n=8, LOMO-validated (+43%) | Honest small-data baseline; field is at opponent-conditioned transformers/GNNs (TacticAI, LEM) — **not reachable at n=8; our edge must be grounding + report quality** |
| Report/LLM | Template pundit + Ollama/OpenAI/Anthropic narration, grounded | **Aligned with SOTA pattern** (RAG-grounded frontier model + wordalisation). Missing: retrieval layer, numeric fact-checker |

## 3. Architecture debt (from full-repo inventory; top items ranked by scaling impact)

1. **No central match registry** — match list + paths duplicated by hand in 5 files
   (`synthesizer/opponent_model.py`, `report/pundit.py`, `tools/france_profile.py`,
   `tools/phase_pct_c6.py`, `tools/style_matrix.py`) across **2 incompatible path conventions**
   (`outputs/matches/<name>/` vs `outputs/<match>/final/`). Fix: one `data/matches.yaml` +
   `registry.py`; everything reads it.
2. **Orphaned metric families**: `fingerprint/transitions.py`, `fingerprint/set_pieces.py` — zero
   importers; `xt.py` controlled-threat/ball-xT CLI-only; possession/PPDA/roles only via
   `tools/possession_report.py`. The pundit report consumes **4 numbers**.
3. **Duplicated, contradictory backtests**: `synthesizer/backtest.py` (n=3, Tier-B loses) vs
   `opponent_model.py` (n=8, Tier-B wins) — deprecate the former with a marker.
4. **Two disconnected report pipelines** (`build_report.py` HTML vs `pundit.py`/`narrate.py`); rich
   facet metrics never reach the product.
5. **Unversioned metrics + hand-tuned global thresholds** (`PARTIAL_VIEW_LINE_OFFSET=11` calibrated on
   one match, applied to all). Fix: stamp `metrics_version` in every parquet; make offsets per-match.
6. **Pitch-dimension sprawl** (105×68 redefined in ≥6 files; 120×80 in contract.py) → one constants module.
7. **Sibling-repo GAT via sys.path + re-exec hack**, unpinned — record checkpoint hash or vendor it.
8. ~20 files with `sys.path.insert`; 6 files with hardcoded `C:/Users/...` paths (ball_eval,
   test_golden_frame, benchmark_detectors, diag_calib, probe_convention, validate_detection).

## 4. SOTA adoption list (everything open-weights / analytic, 4GB-GPU or CPU inference)

Priority-ordered ("do only 5 things" list from the research):

1. **Off-screen imputation** — Graph Imputer (GNN+VAE, Nature SR 2022) or simpler DASE
   (arXiv 2408.10878). Recompute all shape metrics on imputed 22-player frames, report the delta.
   *Validity, not vanity.* Directly attacks our measured +11 m line bias.
2. **Event layer**: T-DEED ball-action spotting (SoccerNet sn-teamspotting, pretrained) + ELASTIC
   event↔tracking sync (88.4% exact alignment) → SPADL rows → **socceraction VAEP** replaces the xT
   grid as headline value metric.
3. **PRTreID** (arXiv 2401.09942) — one model: reID + team + role. Kills kit-clash/ID-switch failures;
   supersedes jersey-colour KMeans.
4. **EFPI** (2506.23843) or **SoccerCPD** (2206.10926) formation/role detection + **Pressing
   Intensity** (2501.04712 — time-to-intercept → P(≥1 defender arrives); positions+velocity only,
   validates vs FIFA "Pressure on the Ball").
5. **Wasserstein style embedding** (OT playing-style distance, 2501.10299 — extends our existing
   style_metrics EMD) + formalize RAG-grounded wordalisation (2504.00767) + post-generation numeric
   fact-checker.

Also noted: No Bells Just Whistles calibration swap (low-med effort), Deep-EIoU tracker,
jersey-number-pipeline (2405.13896) if footage resolution ever allows. **Do NOT chase**: TacticAI
open-play (unreproducible), MatchVoice video commentary (different product), fine-tuning a small LLM
(hallucinates more, hardware-gated — RAG + frontier model wins).

## 5. Theory→metric codex (summary; full agent report in section 8 sources)

Partial-view rule that governs everything: **prefer ratios, per-line/relative distances, ball-anchored
local metrics (visibility highest near ball), and role-centroid time aggregation. Avoid single-frame
whole-team metrics (convex hull, absolute team length) — PARTIAL-FRAGILE.**

New metrics to implement, each with FIFA EFI validator:

| Concept (school) | Metric | Data | FIFA validator |
|---|---|---|---|
| Sacchi 25 m block | inter-line distances (CB↔MF↔FW line centroids) + <25 m flag | POS | Team Length |
| Sacchi synchrony | velocity correlation of adjacent line centroids | POS | Line-height time series |
| Guardiola 5 lanes | 5-lane × 3-third occupancy matrix; half-space share | POS (ratio→robust) | Receptions Behind Lines |
| Guardiola staggering | no-2-same-lane / no-3-same-line violation index (lower bound) | POS | — |
| Overloads (numerical superiority) | attackers−defenders within 10–15 m of ball; time-share at +1/+2 | POS+BALL (robust: near-ball visible) | Forced Turnovers, Final-Third Entries |
| Between-lines reception (positional superiority) | occupancy/receptions in pockets between opponent line y-bands | POS(+BALL) | Receptions Behind Mid/Def Line |
| +1 build-up | build-up outfielders vs first-line pressers count | POS+BALL | Build-Up phase % |
| Pressing intensity (Bielsa/Klopp) | time-to-intercept → P(≥1 defender reaches carrier), avg out-of-possession | POS+velocity | **Pressure on the Ball** (direct) |
| Man-orientation index | defender tracks specific opponent (velocity-pair correlation) vs holds zone | POS (pair co-visible) | — |
| Counterpress 5 s rule | regain within 3/4/5/8 s of loss (curve) — transitions.py already close | BALL+POS | **Ball Recovery Time** (direct) |
| Rest defence | own players goal-side of ball at final-third entry (role-centroid lower bound) | POS+BALL (fragile — deep players off-screen) | opponent Counter-Attack % |
| Verticality / directness (Bielsa) | forward ball displacement ÷ total path per possession | BALL | Long Ball %, Line Breaks |
| Line breaks | ball crosses opponent line-centroid y | BALL+POS | **Line Breaks** (direct) |
| Zone-14 occupation/entries | occupancy + ball entries in central-outside-box zone | POS+BALL | xG proxy |
| Pitch tilt | final-third possession-time share | BALL | Final-Third Entries |
| De Zerbi bait detection | low tempo + rising opponent press intensity + maintained +1 → line-break spike | POS+BALL | Forced Turnovers |

**Counter-structure pairings for the C5 matchup model** (structure signature ↔ counter signature ↔ seam):
1. High line ↔ runs in behind/direct — seam: controlled space behind line + fast forward.
2. Low block ↔ wide overloads + cutbacks + zone 14 — seam: wide overload count × zone-14 occupation.
3. Man-press ↔ rotations/third man — seam: space vacated by committed marker + runner into it.
4. Possession dominance ↔ transition — seam: **rest-defence count/spread vs opponent transition speed**.
5. Build-up vs press ↔ De Zerbi bait — seam: provocation pattern → imminent line break.
6. Box midfield/inverted FB ↔ central traps vs half-space release — seam: central overload vs forced turnover.

This is the vocabulary the opponent model should eventually forecast in — not just 4 shape numbers.

## 6. The synthesizer question ("LLM trained on loads of match data?")

Research verdict: **no fine-tuned LLM.** On 4 GB you can't train anything competitive, and the
literature consensus (TacticalGPT → RAG papers 2025) is **hybrid: deterministic metric engine → grounded
fact store → RAG → frontier LLM with numeric guardrails**. The "trained synthesizer" ambition is served
by the *metric+model layer* (C5, pressing intensity, VAEP once events land), not by LLM weights.
Roadmap for the report product:
1. Fact store: per-match JSON of every metric with provenance (metric version, source: CV|FIFA).
2. Retrieval: narration pulls only relevant facts (matchup seams from §5C) instead of one big template.
3. Guardrail: post-generation checker — every number in the LLM output must match the fact store, else
   regenerate/flag.
4. Structure the prose around the counter-structure seams (§5C) — that's what makes it read like a
   pundit: threats, weaknesses, marquee players (roster), opportunities (seams), all cited.

## 7. Anchored roadmap (phases; each has a definition-of-done + validator)

- **P0 — Plumbing week (no new science). ✅ DONE 2026-07-07.** Match registry (`data/matches.yaml` +
  `core/registry.py`); constants module (`core/pitch.py` + `METRICS_VERSION`); deprecated
  `synthesizer/backtest.py`; fact store (`report/facts.py` → `outputs/facts/<match>.json`) wires the
  orphaned metrics; pundit/narrate read it. Result: adding a match = 1 yaml entry; fact store carries
  110–126 facts (was 4). 165 tests.
- **P1 — Ball-anchored theory metrics (codex B-priority). ✅ DONE 2026-07-08.** Implemented in
  `fingerprint/theory_metrics.py`: pressing intensity (time-to-intercept), counterpress curve, local
  overloads, line breaks (hysteresis-banded), verticality, 5-lane matrix + half-space share. Wired to the
  fact store (`cv.theory`) + pundit. Validated by `tools/validate_theory_c6.py`: **3/3 clean-mapped
  metrics direction-correct vs FIFA** (pressing↔defensive_pressures, line_breaks↔completed_line_breaks,
  overload↔forced_turnovers); 3 more carry honest "weak validator" caveats (no like-for-like FIFA field).
  173 tests. Deferred to later: zone-14, pitch tilt, inter-line synchrony, man-orientation (not yet needed
  by the report). DoD met: each metric validated vs its FIFA counterpart, error reported honestly like C6.
- **P2 — Validity: off-screen imputation.** DASE or Graph-Imputer-style model; recompute all P1/shape
  metrics on imputed frames; publish before/after delta vs FIFA line heights. DoD: in-possession phase
  bias (currently ~+11 m) measurably shrinks without hand offsets.
- **P3 — Event layer.** T-DEED spotting + ELASTIC sync → SPADL → socceraction VAEP + real PPDA + real
  third-man/pass networks. DoD: VAEP per possession on one France match, sanity-checked vs FIFA xG/line
  breaks.
- **P4 — Matchup model v2 + report v2.** C5 forecasts counter-structure seams (§5C) with role/formation
  states (EFPI) as conditioning; report restructured as fact-store + RAG + guardrail (§6). DoD: a
  France-vs-X preview naming threats/weaknesses/opportunities, every claim traced, seams quantified.

**Workflow anchors (to stop wandering):** (1) nothing merges without a validator (FIFA EFI number, LOMO
backtest, or annotated ground truth); (2) every metric registered in one place with version + data needs;
(3) new ideas get a line in this file's roadmap before code; (4) prefer pretrained/analytic over training
anything; (5) partial-view rule of §5 governs every new metric design.

## 8. Sources (key)

Imputation: Graph Imputer (Nature SR 2022), DASE arXiv 2408.10878 · Events: T-DEED / sn-teamspotting,
ELASTIC 2508.09238, socceraction VAEP · Tracking: Deep-EIoU 2306.13074, PRTreID 2401.09942, GSR
2504.06357, No Bells Just Whistles (CVPRW'24) · Tactical models: Pressing Intensity 2501.04712,
SoccerCPD 2206.10926, EFPI 2506.23843, OBSO (Spearman), EPV U-Net 2502.02565, OT style 2501.10299,
TacticAI 2310.10553, LEM/foundation 2407.14558 · LLM: wordalisation 2504.00767, MatchTime 2406.18530,
TacticalGPT (StatsBomb 2023) · Theory: Spielverlagerung (Marić, half-spaces/rest defence), Sacchi 25 m
block, Juego de Posición (Breaking The Lines / Touchline Theory), gegenpressing detection (Springer
s10618-021-00763-7), FIFA EFI Data Reference (efidatareference.com) + FIFA Training Centre phases.
