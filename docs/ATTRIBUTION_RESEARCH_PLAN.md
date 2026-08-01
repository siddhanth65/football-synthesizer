# Player-event attribution: research plan (2026-07-27)

Owner: orchestrator (planned on Fable 5 at Sid's direction). Status: PLAN — no code started.
Problem: events (passes, shots, tackles) cannot be pinned to named players because identity does
not persist — players leave frame, tracklets fragment, and naming collapses.
Measured baseline (results/CARRIER_CONSTRAINED_v2.md): carrier attribution precision
**0.846 (gate-hit) x 0.719 (team) x 0.609 (naming) = 0.350** (best variant 0.400).

## The organizing insight (from the basketball paper Sid supplied)

Lu, Ting, Little, Murphy, *Learning to Track and Identify Players from Broadcast Sports Videos*
(TPAMI 2013), `basketball paper.pdf`:

- A per-image identity classifier scored **50-55%**. The SAME features inside a joint solve —
  (a) identity constant within a tracklet, (b) a player appears at most once per frame (mutual
  exclusion), (c) play-by-play text restricting who is on court — scored **85-89%**.
- Weak labels from play-by-play cut supervision from **20,000 labels to 200**.
- They abandoned jersey OCR as unreliable (2013) and identified players as whole entities.

Mapping to us: we already own every *unary* ingredient (PRTreID embeddings, Koshkina OCR 86.13%,
team assignment, lineup priors, homography). What we never built is the **joint inference layer**.
Our 0.609 naming factor and the silent-OCR failure mode are exactly what constraints repair:
today one wrong OCR read poisons a tracklet and nothing contradicts it; under mutex + roster +
temporal consistency, a wrong name must *fight* every other assignment in the half.

## Current-literature confirmation (2024-2026)

1. **GTA / GTA-Link** (Sun et al., ACCV 2024 W; arXiv 2411.08216): training-free tracklet
   **Splitter** (removes multi-identity contamination) + **Connector** (hierarchical clustering of
   fragments by appearance embedding under spatio-temporal constraints). Plug-and-play on any
   tracker; SOTA on SportsMOT (HOTA 81.04). **SoccerTrack 2025 winner = Deep-EIoU + GTA**
   (arXiv 2602.00484). Training-free -> fits the 4 GB GPU constraint; clustering is CPU work.
2. **SoccerNet GSR winner** (Golovkin et al., CVPRW 2025; arXiv 2504.06357): GS-HOTA **63.81**
   with detector + camera + tracking + ReID + jersey + *tracklet-level* aggregation. Our measured
   22.85 on the same benchmark says the gap is engineering headroom with known recipes, not magic.
3. **Commentary as weak labels — Sid's idea, now with a literature**: SoccerNet-Echoes (arXiv
   2405.07354) ran Whisper ASR over 1,100 broadcast halves; MatchTime (arXiv 2406.18530) showed
   the hard part is *alignment* (they hand-corrected timestamps). Commentary is football's
   play-by-play: it names the carrier at ball-event moments. Our pre-declared go/no-go gate
   (precision >=0.60, >=1.5 names/min over a full half, lag IQR <=4 s) guards exactly the failure
   MatchTime documents.

## Verdicts on the attached leads

- **Reddit thread (4 y old)**: describes the pipeline we already built (YOLO + ByteTrack + OCR +
  roster logic). Its open question — "how do SkillCorner do it?" — is answered by the GSR-winner
  recipe above. Nothing new to adopt; useful as confirmation the naive route dead-ends where ours did.
- **Kaggle football-match-actions dataset**: solves the wrong half. Detecting *that* a pass/tackle
  happened is not our bottleneck (pass counts already 0.97-1.09x official on 10/12 matches; goals
  13/13 recall). The bottleneck is *who*, and action clips carry no identity labels. SKIP for
  attribution. Revisit only if we later want event classes we lack (e.g. tackles) — as a detector
  training set, decoupled from this plan.
- **Gait recognition**: dead on arrival at broadcast scale — our measured median face height is
  9.9 px (results/CROSS_MATCH_GALLERY.md); silhouettes at that scale carry no gait signal. Do not pursue.

## The program — three stages, each gated

**Stage 1 — Fragmentation repair (GTA-Link port).** Run Splitter+Connector over our existing
tracklets + PRTreID embeddings (already computed: results/carrier_attr/*_gallery_emb.npy).
No training. Metrics: tracklets-per-player-half before/after; GS-HOTA on the SoccerNet GSR public
split (baseline 22.85); re-run the attribution factorisation. Expected lift: naming factor (0.609)
rises as fragments inherit confident names; gate-hit (0.846) may rise too (fewer dead tracks at
kick moments).

**Stage 2 — Global identity solve (the Lu-style ILP, modern parts).** Per half: variables =
(merged tracklet -> roster name | unknown). Evidence terms: OCR reads as **soft votes with a
confusion prior** (kills the silent-failure mode — today the roster mask makes every wrong read
look valid), PRTreID gallery similarity, team posterior, role priors (fixes the GK-never-named
defect: de Ligt/Onana/van Dijk/Vicario at 0 mentions in 1,664 named tracks). Constraints: mutex
per frame, <=11 per team on pitch, substitution windows from lineups. Solver: OR-tools CP-SAT or
scipy linprog — CPU only. **Gate (pre-declared): carrier-naming precision on labels_filled.csv
from 0.400 (current best) to >=0.55, with coverage reported alongside.**

**Stage 3 — Commentary weak labels.** Whisper (small/medium, local, free) on our own match audio;
extract (timestamp, player-name) mentions; attach as unary evidence on the tracklet nearest the
ball at mention time; feed Stage 2. Boxed to 1 week with the go/no-go gate above. This is the
play-by-play prior from the basketball paper, in the only form football broadcasts provide it.

**Ceiling arithmetic (pre-declared so nobody is surprised).** Naming is one factor of three.
If naming reaches 0.85 (basketball-paper level) and the others hold: 0.846 x 0.719 x 0.85 ~= **0.52**.
Going beyond requires lifting gate-hit and team too — Stage 1 is the only stage that touches them.
If Stage 1+2 land under 0.50 composite, the honest conclusion is that broadcast attribution caps
near half-right, and that is itself a reportable finding with a measured cause per factor.

## Data to fetch (all free/licensed; no copyrighted footage)

- SoccerNet GSR split — already held (credentials in memory). Re-eval harness exists.
- SportsMOT (eval only, if we want the GTA port sanity-checked against its home benchmark).
- SoccerNet-Echoes transcripts (text only, small) — reference distribution for Stage 3 ASR quality.
- NOT the Kaggle actions dataset (see verdict).

## Post-survey revision (2026-07-28, from docs/SOCCERNET_SURVEY_RAW.md)

Survey of SoccerNet challenges/repos (searcher on requested-Opus route) changes four things:

1. **Validation target upgraded: FOOTPASS.** SoccerNet retired GSR in 2026 and replaced it with
   *Player-Centric Ball Action Spotting* — name the acting player by team+jersey, Macro-F1, jersey
   must match. Its dataset FOOTPASS: 54 matches, 81 h broadcast, **102,992 validated
   (frame, team, jersey, action) tuples** (annotations CC BY-NC 4.0 on HuggingFace; video via NDA).
   This dissolves our label-starvation problem (78 hand moments) AND confirms the thesis question is
   the field's current frontier. Note: the official baseline (46.41) consumes ground-truth game
   state — an honest end-to-end-from-pixels number is open territory.
2. **The field validates Stage 1's negative.** Across the entire GSR 2025 leaderboard, association
   scores (GS-AssA 60-85) far exceed identity scores (GS-DetA 14-51); the winner states its failures
   were "mostly jersey number recognition". Tracklet continuity is not where anyone is stuck —
   identity is. Matches our measured result (connector helps tracking, not naming).
3. **No winner's code exists to borrow** — zero public repos across all GSR top-10s, both years.
   But the 4th-place recipe (61.64) is nearly our stack (TrackLab-style + GTA + PnLCalib-class
   calibration) plus: better detector (RF-DETR), PARSeq jersey OCR on pose-cropped torsos, and a
   **geometric goalkeeper heuristic**. The gap to 60+ is identity engineering, not architecture.
4. **Integrity check required:** sn-gamestate's GS-HOTA scoring was broken before TrackLab 1.3.24
   (May 2026). Must verify which pairing produced our 22.85/24.18 before quoting them against 63.81.
   (Relative on/off deltas used one harness and likely survive; absolute comparability may not.)

Concrete adoptions (queued, in order):
- **A — DONE 2026-07-28, cleared.** Our harness never imports tracklab/sn-gamestate at all:
  `eval/gsr_score.py` drives SoccerNet's own `sn-trackeval` fork directly (installed 2026-07-15
  from the repo's still-current HEAD, static since 2025-07). The TrackLab 1.3.24 fix touched
  packages outside our execution path, and the pairing logic we DO execute matches by image_id
  (the safe pattern). All three numbers (14.76 / 22.85 / 24.18) came from one unchanged scorer;
  comparability to the leaderboard (also scored on the official evaluator) stands.
- **B — annotations acquired 2026-07-28** (data/footpass/raw/, 2.6 GB, gitignored; schema in
  data/footpass/README.md). Better than advertised: DENSE per-frame per-player tracking
  (frame, player_id, jersey, role, x, y, velocity, ROI box, action class), not just event tuples.
  97,397 visible events in train+val (the 5.4% shortfall vs 102,992 = the 3 withheld challenge
  games — checks out); 54/54 games present. Caveats: fully anonymized (game_0..53, teams 1/2, no
  club names, no roster file), and NO video (separate NDA form — Sid's decision, required before
  FOOTPASS can grade our pipeline end-to-end from pixels). Interim use without video: fragment
  FOOTPASS's dense tracks synthetically (mirror our measured tracklet-length distribution) and
  grade the Stage 2 SOLVER's constraint machinery at 97k-event scale — identity-inference-given-
  fragmentation, the identity analog of the Metrica virtual-camera trick.
- **C (post-Stage-2, folds into solver):** OCR hardening from the winners' pattern — verify
  Koshkina's legibility classifier actually runs in our path; per-read confidence gate (~0.70) on
  votes; explicit "no number visible" class (the benchmark scores -1 as first-class; we have no
  such channel). UniBw hit 90.95% largely on gating discipline.
- **D (post-Stage-2):** GK naming — PRTreID (which we already run) HAS a role head that predicts
  goalkeeper; check whether we discard its output; add the 4th-place geometric GK prior as backstop.
- **E (candidate, GPU window needed):** per-match self-supervised re-ID fine-tune (Kalisteo: train
  on the video's own non-crossing tracklets, no labels) — the only surveyed idea that attacks the
  0.62 appearance ceiling directly. F: Qwen2-VL-2B (Apache-2.0) zoom-and-read jersey trial — the
  one VLM path viable at 4 GB. G: Deep-EIoU/GTATrack tracker swap (code released) — bigger surgery,
  only if C/D/E plateau.
- **Stage 3 note:** SoccerNet-Echoes is wrong-era footage and its authors call player-name ASR
  unreliable; nobody has shipped commentary-name -> tracklet binding on soccer. Our boxed
  experiment keeps its gates unchanged — and its novelty claim strengthens if it passes.

## The measured law (2026-07-28) — Stage 3 retired, levers re-ranked

`results/EVIDENCE_DENSITY_LAW.md` (simulator calibration-gated against Stage 2's measured reality:
0.2576 vs 0.2604 with nothing fitted to it; kb ident-013..016). On FOOTPASS truth, 97,397 events:

- **The bar:** event-attribution precision 0.85 at coverage >=0.50 requires OCR-like read density
  **d = 0.347 vs today's 0.087 (4.0x)** at measured read precision — or **~7x less fragmentation**
  (~1 fragment/30 s on-screen -> 0.505 coverage at today's read volume). Read *precision* is a weak
  lever (1.3x). VAL reproduces d* = 0.347 exactly.
- **Commentary as a naming channel: FAIL, retired.** At the pre-declared go/no-go point
  (1.5 names/min, prec 0.60, lag IQR 4 s): coverage-at-0.85 = 0.000; all 27 commentary-only
  conditions indistinguishable from no evidence. Measured cause: at 4 s lag the mention binds to
  the right player 13.2% vs 12.5% CHANCE (ball moves 24 m in 4 s). Binding needs IQR <=0.5-1 s to
  carry signal, and even ORACLE binding at 3/min tops out at 0.295 coverage — below the bar.
  Stacked on today's OCR it is net-HARMFUL (0.124 -> 0.042 at 1.5/min), replicated on VAL.
  The 1-week Stage 3 build is cancelled by measurement, at simulator cost.
- **Solve-unit insight:** the same evidence anchors 0.176 of events on 30 s units but 0.395 on
  2-min units — Stage 2's 0.365 ceiling was partly clip-length artifact. Real matches should solve
  half-wide or in long windows.
- **Hard ceiling:** 18.5% of PCBAS events have an OFF-SCREEN actor — no pixel pipeline exceeds
  0.81 coverage; full-coverage arms already sit there.
- **Low-precision evidence is worse than none** when mixed with good evidence; OCR
  confidence-gating (adoption C) is near a wash and net-negative if it discards >~25% of reads.

**Re-ranked build order:** (1) OCR densification toward 4x — persist per-crop reads, pose/torso
crops (PARSeq pattern), Qwen2-VL-2B zoom trial (survey F), long solve windows; (2) fragmentation
toward ~1/30 s — within-chunk association is the prize (perfect within-chunk = 0.505 today);
(3) end-to-end from-pixels PCBAS run on the 3 VAL games once (1)/(2) move. Commentary: only as
report color, never as attribution evidence.

## Execution notes

- All code via deep-worker; GTA-Link port and CP-SAT solve are CPU-side, so GPU stays free for
  any embedding recompute (one job at a time).
- Every stage writes its result into knowledge/claims.json via tools/kb.py, pass or fail.
