# GSR campaign brief (2026-07-31) — the pre-GPU consolidation Sid asked for

Goal (Sid, verbatim intent): score well on the SoccerNet GSR benchmark (GS-HOTA). ManU-corpus
adaptation comes later. Cluster access approved; Sid books slots on request. Companion docs:
docs/GSR_CLUSTER_ROADMAP.md (the S0-S5 plan), docs/GSR_METHODS_DEEP_DIVE.md,
docs/SOCCERNET_DATA_INVENTORY.md, STATUS.md (the day-by-day record).

## 1. The metric — and whether GS-HOTA is "the only score"

GS-HOTA is HOTA (the standard multi-object tracking metric) with one merciless change: a
predicted player only counts as matching a ground-truth player if the position is close ENOUGH
**and** the identity attributes (team, jersey number, role) are ALL correct. Wrong jersey =
the match scores zero, however perfect the position.

- GS-HOTA = sqrt(GS-DetA x GS-AssA). GS-DetA = detection quality under that gate; GS-AssA =
  association (track continuity) quality under it.
- **Ranking: the challenge reports (2024, 2025) rank teams by GS-HOTA alone.** GS-DetA/GS-AssA
  are published alongside as diagnostics, not ranked. So yes: GS-HOTA is the score that matters;
  DetA/AssA are the dashboard that tells you WHY you scored it. (Codabench page is a JS app —
  ranking column being confirmed via API by the fetch worker; the challenge reports are the
  primary source either way.)
- Practical consequence, measured twice (their Table 7, our factorisation): jersey errors cost
  ~2.5x anything else. The metric is an identity metric wearing a tracking metric's clothes.

## 2. Where we stand (all measured, kb-claimed)

- **Our score: 33.20** on the 38-sequence held-out subset we declared inside the public valid
  split (arc: 14.76 -> 22.85 -> 24.18 -> 33.20). Same official scorer (sn-trackeval, audited),
  DIFFERENT sequences than the leaderboard (they rank on the withheld-labels test/challenge
  splits) — so 33.20 is internally rock-solid, externally indicative only. First campaign act:
  get an official number (Section 4).
- **Leaderboard: 63.90** (KIST-GSR 2025, = Broadcast2Pitch, WACV 2026), 63.81 (Constructor 2024),
  podium 61-63. Every top entry: single-GPU training. No winner released code (verified).
- **The measured lever (their Table 5, components held fixed):** identity model swap:
  PRTreID+OCR-shaped stack 18.11 -> CLIP-encoder+attribute-heads 60.13 -> LLaMA-3.2-Vision 61.48.
  The identity model IS the benchmark.

## 3. What we already own that transfers

Detection+tracking (BoT-SORT validated at stride 2: recall 0.947, contamination 0.037), GTA
connector (+GS-HOTA, 51/58 sequences), PnLCalib (0.13-0.20 m, zero rejects, foreign stadiums),
per-crop OCR with two validated fixes (aggregation debug: 2.45x reads; crop x1.25: 1.37x at held
precision), the frozen-split/pre-registration evaluation machinery, and the official scorer
running locally. Data: valid split held; train+test fetching now (~18.6 GB); FOOTPASS separate.

## 4. What is closed, with the number that closed it

- Appearance-gallery retrieval as the namer: saturated 0.65-0.71; three pre-registered fails.
- The joint solver as the namer: dial not frontier (3 measurements); r_abstain=0 poisons
  (p~1e-99); the abstention point that fixes naming kills GK coverage.
- Commentary as identity evidence: chance-level binding at real lag (13.2% vs 12.5%).
- Gait/faces: pixel-starved (median face 9.9 px). Splitter half of GTA: sign-flips OOD.
- Hand-tuned OCR at broadcast density: even double-fixed, d = 0.158 (EPL) / 0.070 (Serie A) vs
  the d* = 0.347 the law requires. Inference-side OCR is out of headroom.

**The synthesis of a month of negatives: evidence, not inference, is the binding constraint —
and the only measured path to dense evidence is a TRAINED identity model (Table 5).**

## 5. The campaign (= roadmap S0-S5, operationalized)

1. **S0 cluster probe** (first slot, half day): sn-gamestate env pins (torch 1.13/CUDA 11.7 —
   suspect H100 incompatibility; A100 target), our repo's env alongside.
2. **S1 data**: train+test splits (in flight on laptop; mirror to cluster storage on arrival).
3. **S2 flag-plant**: run OUR current pipeline on the official test split, submit to codabench
   (1/day) -> the first official leaderboard number. Laptop-feasible (~9-11 GPU-h) if cluster
   slot lags.
4. **S3 the big lever**: train CLIP-encoder + attribute-heads identity on GSR train (their 60.13
   recipe; single-GPU scale), integrate behind our tracking, re-submit.
5. **S4 jersey head with evidential abstention** (Grad-style; our abstention findings converge on
   it) — parallel track to S3.
6. **S5 optional VLM pass** (LLaMA-3.2-V 1-epoch; gated access + ~+1.4 expected — last, if at all).

## 6. How we decide a method works (the evaluation constitution)

1. **The market price**: GS-HOTA on the official codabench test split. A method that does not
   move it does not work, whatever the local numbers say. Cadence: submit at most 1/day.
2. **Local law**: tune ONLY on train/valid; hold the declared local test frozen; single-run
   evaluations at frozen configs; paired per-sequence statistics (Wilcoxon; McNemar for paired
   binary), not bare means.
3. **Claims discipline**: pre-registered one-shot tests for any headline claim (registration file
   timestamped before artifacts — three precedents on record); every result, pass or FAIL, into
   knowledge/claims.json; identity claims report coverage-at-a-precision-floor, never bare
   accuracy (the Stage-2 lesson).
4. **Diagnostics**: GS-DetA vs GS-AssA split, per-sequence deltas, and the attribute
   decomposition (pitch/role/team/jersey) to locate every gain or loss.

## 7. Known gaps / risks going in

No winner code exists (all reimplementation); GSR license listed inconsistently (GPL-3.0 vs
CC BY-4.0 — resolve before publishing); footage is Swiss Super League 2019 (domain transfer to
EPL is a later, separate claim); cluster env untested (S0 exists for this); test-split labels are
withheld (the server is the only test-split oracle — local dev stays on train/valid).
