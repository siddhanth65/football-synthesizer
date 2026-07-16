# STATUS

**Last updated:** 2026-07-15 (C3 eval complete — the fix RESTORES the receiver headline)

## 2026-07-16 — B2 STAGE 2b (anchors): probe positive, then TWO honest negatives; anchors stay unwired

Probe (brighton, full match): 5,558 high-conf close-up anchors, 78% of shots covered, transfer
POSITIVE on clean back-views — but naive conf-gating is ~80% hallucination (crowd→"1"@0.94;
front-view players→"20/29/11"; illegible head never saw non-player crops). Fix attempt 1
(negatives retrain, weak-labeled crowd/ref crops): **DOMAIN SHORTCUT — SoccerNet gate improves
(0.417→0.448) while close-up recall collapses 100%→0.1-3.2%** (reject head keys on close-up domain,
not the number patch; the task premise "negatives = core fix" is measured WRONG). Fix attempt 2
(per-shot consensus): 5 anchors match-wide at 20% precision — consensus REINFORCES consistent
front-view hallucinations, only cancels random scatter. **Anchors remain un-wired (80% bar not
met).** Next levers (measured next, not assumed): kit-color gate (kills crowd/ref mass) + digit-
evidence/OCR check on the torso band (kills in-kit front-view mass — the dominant error, which a
kit gate alone cannot touch). Retrained weights kept but NOT promoted.

## 2026-07-16 — B2 STAGE 2a: ReID track-relinking — GS-AssA +4.2..+5.4 external lift; 35% merge precision caveat

Post-hoc fragment merging (deep-worker requested: opus; `generator/track_relink.py`, OSNet
embeddings + team/role/temporal/motion constraint gates + greedy merge; threshold 0.80 frozen on 3
pilot seqs before scoring the other 55). Re-scored all 58 GSR sequences, official evaluator:
**GS-AssA +4.2 to +5.4 in every config** (loc_assoc 39.8→44.7, HOTA 48.9→51.7, IDF1 51.9→58.5;
official full 14.8→15.8); DetA/LocA flat as expected (relabeling only). Fragments/seq 79→36.
**HONEST CATCH: true merge precision on pilot GT = 35%** (vs ~9% random) — the lift is
CONSTRAINT-driven, not appearance-driven: ImageNet OSNet is kit-dominated (median cosine 0.81 on
constraint-valid pairs; AssA near-flat over thresholds 0.50-0.80) and cannot separate same-kit
players — exactly what close-up jersey anchors (Stage 2b) attack. **PRODUCTION GUARD: relink is
benchmark-side ONLY — do NOT wire into facts/report per-player metrics until merge precision
clears a pre-committed bar (propose >=80%);** 65% wrong merges would corrupt player attribution.
Worker also caught+fixed a real union-find bug mid-build (interval components). 12 targeted tests
green. Baseline artifacts untouched (`gsr_scores_relink.json` separate).

## 2026-07-16 — B2 JERSEY STAGE 1+1b: model trained + honest negative on the cheap levers

Stage 1 (deep-worker requested: opus; `generator/jersey_id.py`, `tools/train_jersey.py`, resumable
slice trainer, 4 GB fp16): ResNet18 100-way head + tracklet voting on local SoccerNet jersey-2023.
Official test (1,211 tracklets): **0.396 tracklet accuracy** (baseline 0.293; published 0.73-0.92
— we are honestly below). Legibility works (P 0.84 / R 0.75); number recognition is the weak link.
Stage 1b ablation (second worker run): factorized tens×units heads **-0.4 to -1.2 pp**; legibility-
filtered digit loss **+0.1 pp** — both flat, stopped per the pre-set <1 pp rule; steps 3-4 skipped
with cause. **Diagnosed ceiling: visual broadcast-resolution misreads** (4→29, 44→29 confusions);
published-range recipes use pose/STN alignment, temporal fusion, heavier backbones — not head
surgery. Stage 1c (torso-band crop, pre-committed band 0.15-0.55): **+2.1 pp → 0.417/0.419** — the first
lever that moved the headline, confirming the visual-resolution diagnosis, but below the >3 pp
bar; adopted as free default preprocessing, multi-band ensemble rejected with cause. Jersey line
CLOSED at 0.42 tracklet / 0.31 numbered-only. Strategic read: at 0.29
numbered-only, jersey is a **weak prior to FUSE** (team+role+position+close-up anchors), not a
standalone signal — which matches the close-up-anchored Layer-2 architecture from the July probe.
Per-player Sofascore oracle cached meanwhile (40 players × 84 stats, brighton) — validation target
ready. Also: per-crop mojibake in scraper names queued for the number→name matcher.

## 2026-07-16 — GS-HOTA EXTERNAL BENCHMARK SHIPPED: first public-metric grade in project history

Full SoccerNet-GSR **valid split (58 seqs)** scored with the **official** evaluator (sn-trackeval
0.4.0; GT-copy sanity = 100.0; deep-worker requested: opus + slice-runner driven by main session
across 3 session restarts — resumable-by-disk-state design absorbed every kill).
`results/gsr_benchmark/GSR_BENCHMARK.md` + `eval/gsr_score.py` (stub now real; 6 tests).
**Headline: official gs_hota_full 14.8 | no_jersey 43.1 | role_only 45.0 | loc_assoc 48.9 |
GS-LocA 92.5.** Reading: when we report a player, the position is right (LocA ~92.5 — the
calibration/projection stack externally validated); the official composite is a **pre-Layer-2
floor by construction** (we emit jersey=null → every numbered GT player unmatchable; full GS-HOTA
tracks per-seq jersey-annotation density almost linearly). AssA ~37-40 = the track-churn/re-ID gap
— the same problem jersey Layer 2 attacks. Split caveat stated (valid ≠ challenge; published
baseline 29.01 / SOTA 63.90 are context, not ranking). Calib: period-25 measured equivalent to
per-frame (-0.9 LocA, 6× faster). **December story now has its external anchor: pre-identity 14.8
→ post-Layer-2 X, on a public benchmark.**

## 2026-07-15 — LIVE-PLAY FILTER SHIPPED; the pre-registered 50%-gate question is answered: NO

Phase-B B1.2 done (deep-worker requested: opus; `generator/live_play.py`, `tools/live_play_probe.py`,
`tests/test_live_play.py` 17/17 green, `results/live_play_probe.md` + audit montages). Rule-based,
CPU-only, thresholds pre-committed BEFORE measurement, flag-gated, no shipped metric touched.
Brighton full grid (30,011 frames): live_wide 32.1% / close_up 24.1% / replay 16.6% /
zero-detection 27.1% — reconciles with the known ~41%-of-detected-frames live fraction and the
~37% geometry yield (measured 36.1%). **Live-play-conditional: geometry yield 80.5%, post-link
ball coverage 71.2%** (vs 37.3% whole-grid) — the honest denominators for every report.
**Pre-registered question answered NO: pass-recall stays 47.8/48.6% — the 50% event gate cannot
be cleared by denominator conditioning** (two conditioning variants computed and explicitly
rejected as over-corrections; per-pass timestamps would be needed). Consistent with the re-pricing:
filter = honesty + compute win (~61% of ball frames land on live_wide), NOT a recall lever; the
remaining recall lever is the learned event/identity layer (plan B2+). Classifier honesty:
live_wide ~100% visual precision (10/10, the class that matters); the non-live sub-classes are
really "detector under-populated" buckets — treat the split as binary. Queued: box-height signal
(not persisted in dense parquets today) if a semantic 4-way shot classifier is ever needed.
**Consequence for the prof decision:** the relative-claims bar (symmetric ~48% capture) is now the
ONLY path to PL ball-family claims — the decision cannot be deferred behind "the filter will fix it".
**DECIDED (user, 2026-07-15): adopt the relative-claims regime.** Pre-committed bar: ball families
may render in COMPARATIVE form only (shares/ratios/team-vs-team differences, never absolute
totals) when coverage >=40% AND recall-proxy team-symmetry spread <=0.05; absolute-claim rendering
still requires the original 50% recall gate. Implementation delegated to report_v2 (in flight).

## 2026-07-14 — AUDIT VERIFIED (2 corrections shipped), REPO PUSHED, GSR POSITIONING, DECEMBER PLAN

**External-LLM audit adjudicated** (deep-worker requested: opus, read-only; every claim checked
against code/artifacts — its structural findings were real, its magnitudes were not):

1. **LINE-HEIGHT HEADLINE RESTATED (the big one).** The de-bias slope was fit on all 5 processed
   matches INCLUDING the 3 FIFA validation matches (docstrings claimed the opposite — now fixed in
   `tools/fit_line_debias.py` + `generator/impute.py`). Held-out refit (club matches only): slope
   -6.31 vs shipped -5.62; clean validation on the France matches: **raw 16.4 m → 5.5 m in-sample
   → ~7.2 m held-out (contamination cost +1.7 m)**. LOMO slope stable (-5.9..-5.0); the club-only
   slope sits outside that envelope (broadcast-domain difference, not noise). DECISION: keep the
   shipped constant (swapping to the domain-mismatched club slope ships a *worse* number), quote
   **7.2 m held-out** everywhere or label 5.5 m explicitly in-sample. Ledger + CV explainer updated.
   Real fix queued: one more non-FIFA WC-broadcast match for a domain-matched clean fit.
2. **C3 CROSS-CHUNK TRACK BUG CONFIRMED + FIXED.** `attacker/tracks.py`/`labels.py` grouped by
   `track_id` only, but ids are chunk-local → different players merged across chunks in EVERY
   multi-chunk artifact (brighton 7,428 dup (track_id,frame) rows; senegal 46,193; audit's exact
   counts didn't reproduce but the defect is real). Affected: C3 attacker path only
   (`eval/attacker_eval.py`, `heads.py train_run_head`); fingerprint paths were already safe
   (pre-grouped or `globalize_chunk_ids`). **Pre-fix C3 run/receiver numbers are WITHDRAWN.**
   Fix: new `attacker.tracks.track_keys` helper — `(chunk, track_id)` grouping when a chunk column
   exists — through `build_tracks`/`run_targets`/`receiver_labels`/`receiver_candidates`/the
   run-head split; single-chunk callers byte-identical; 2 regression tests (9/9 green, ruff clean).
   **2026-07-15 EVAL (attack_dirs held identical, only chunk-awareness differs): the fix RESTORES
   the C3 headline rather than shrinking it.** Run head RMSE 6.15→4.35 m / hit@3m 0.365→0.571
   (brighton), 4.43→3.83 m / 0.594→0.671 (senegal) — the bug had been fabricating teleport
   velocities (outlier drop 26.2→3.2%). Receiver head: pre-fix was near-random (top3 0.156/0.140,
   with fake cross-chunk "passes" inflating event counts); fixed = **top1 0.600/0.391, top3
   0.933/0.812** — clearing the old-360 baselines (run 6.33 m/0.244; receiver 0.41/0.78). Honest
   caveats: receiver test sets are small (48/182 events, wide error bars); a pre-existing
   half-time direction confound in `attack_dirs` (global sign per team, teams swap ends) affects
   dir_cos interpretability — orthogonal to this bug, queued. No cached models/labels existed on
   disk (heads are in-memory sklearn fits at eval time) — nothing to delete; older STATUS entries
   quoting pre-fix C3 numbers stay as history, superseded by this entry.
3. **C5 honestly reframed**: explanatory post-match regression (feature = opponent's REALIZED line
   from the same match; `predict.py` Tier-B still raises NotImplementedError; cache stale at 8 obs
   vs 5 matches). Upgrade path (predict opponent line from THEIR prior matches → true pre-match
   forecast) is now B1.3 in `docs/BTP_DECEMBER_PLAN.md`.
   Also confirmed: `complete.py`/`gsr_score.py`/`fifa_validate.py` stubs; SoccerNet jersey-2023
   local (2,638 tracklets) + consumed by nothing; README stale (rewrite queued).

**REPO ON GITHUB.** Checkpoint commit `bf150f7` (195 files, author Sid, no AI attribution) pushed
to **private** `siddhanth65/football-synthesizer`. Excluded: outputs/, SoccerNet zips (2.2 GB),
FIFA PMSR PDFs (copyright), user-local files — .gitignore hardened first (fast-worker requested:
sonnet). gh CLI installed; user authed as siddhanth65.

**RESEARCH SWEEP (partial — quota-interrupted, resume queued): the project's task has a name.**
SoccerNet **Game State Reconstruction** (CVPRW'24) is exactly our problem; **GS-HOTA** is the
field's metric; challenge SOTA 63.8-63.9 vs baselines 23-29 [verified 3-0 votes, arXiv 2404.11335,
2409.10587, 2508.19182]. The winning 2025 GSR pipeline mirrors our architecture AND reads jerseys
with a VLM on crops — independently validating the Layer-2 plan. Published GSR SOTA handles
off-screen players by linear interpolation only [VERIFIED 3-0, 2026-07-15], and a FIFA-co-authored
study measures the off-screen cliff (0.44-1.14 m detected → 4.6-12.2 m off-screen vs ~1 m industry
bar) [captured] — so **validated off-screen imputation is a real novelty axis**. Research total:
13 claims verified 3-0, 12 captured-unverified (FIFA/commercial quotes); synthesis folded into
`docs/BTP_DECEMBER_PLAN.md`; research loop CLOSED (no further resumes — diminishing returns).
Data routes: SoccerNet-GSR is free/no-NDA (external benchmark for us); StatsBomb 360 freeze-frames
= VISIBLE players only (like-for-like validation of our freeze frames, not full-pitch truth);
SkillCorner opendata now 10 A-League 24/25 broadcast-tracking matches.

**NEW DOCS:** `docs/CV_EXPLAINER.md` (stage-by-stage pipeline teaching doc for Sid — incl. two
corrections to our own folklore: calibration gate is >=4 keypoints, the >=6 gate is the
ball-projection path; production ball net is TrackNetV2, not WASB) and `docs/BTP_DECEMBER_PLAN.md`
(fine-grained plan to Review 1: B0 integrity closeout → B1 GS-HOTA external benchmark + live-play
filter + C5 forecast v0 → B2 jersey identity → B3 season scale → B4 validated imputation →
B5 thesis assembly; pivot table pre-decided; venue targets CVSports/MLSA/Sloan).

**IN FLIGHT (2026-07-15):** research workflow verification+synthesis resume; SoccerNet-GSR valid
split download (external GS-HOTA benchmark, plan B1.1); live-play filter build + measurement
(plan B1.2, pre-registered 50%-recall question); README rewritten (GSR framing, honest numbers).

## JERSEY-NUMBER FEASIBILITY — LAYER 2 IS VIABLE; close-ups are the identity goldmine

Probe on brighton footage (deep-worker requested: opus; `tools/jersey_feasibility.py`,
`results/jersey_probe/JERSEY_PROBE.md` + montages). Content mix: wide 45% / medium 29% / close 26%.

**Path 1 (direct OCR on wide play): marginal.** Median tactical torso ~38 px → back number ~13-14 px
— at the legibility floor; the tallest wide crops show readable numbers but the median needs a
TRAINED model (SoccerNet-scale), and numbers are backs-only (fronts = sponsor). Passes the
pre-committed >=20% bar (71% clear ~12 px) but only as a per-track vote, never per-frame.

**Path 2 (close-up-anchored identity): the strong path.** 26% of the match is close-up/replay —
frames we DISCARD for geometry — and **30-38% of those frames carry a plainly legible number, often
with the SURNAME nameplate** (evidence: "GILMOUR 11" at ~200 px, `closeup_diag/`). Legibility is
SOLVED there; the binding constraint is **cross-cut track linking** (carrying the identity from the
close-up back onto the tactical track). **Recommended architecture: both together** — close-up
anchors propagated along track ids + a SoccerNet-trained number model voting over wide-play frames.

**Detector gotcha discovered (matters beyond this probe):** the football-trained YOLO used by
extract.py is BLIND to close-up players (box height caps ~157 px; COCO yolov8s sees ~1050 px on the
same frames). Harmless for tactical extraction (close-ups yield no geometry anyway) but any Layer-2
work must use a general detector on close-up content. First probe pass was wrong because of this —
caught and redone with COCO.

Parallel: SoccerNet jersey-2023 dataset download (fast-worker requested: sonnet) in flight — the NDA
only gates raw broadcast videos; jersey tracklets/labels, action-spotting labels + 2fps features,
calibration/re-ID/tracking data are ALL freely downloadable via the pip package. NDA (user filed,
awaiting reply) is now a nice-to-have, not a blocker.

## POSE-CARRY PROBE — negative, and it RE-PRICES the whole plumbing programme

Probe (deep-worker requested: opus; `generator/pose_carry.py`, `tools/pose_carry_probe.py`,
`results/pose_carry_probe.md`; 263 tests, ruff clean, `outputs/` untouched).

**Faithfulness PASSES** (borrowed poses are geometrically sound): naive LOO median **0.20 m** / p90
1.04 m; a stricter gap-matched *blocked* LOO (the worker's own addition, forcing the borrow gap to
match production's median 15-frame gap) gives median 0.39-0.41 m / p90 ~3.0 m — exactly at the
pre-committed bar, reported as found, nothing tuned.

**But the lever does NOT fire: geometry yield 36.1% -> 37.7% (+1.7 pp)** vs the ball's +18.0 pp.

**THE RE-ATTRIBUTION (this supersedes my previous STATUS correction, which got the DENOMINATOR right
and the CAUSE wrong).** The ~63% of detection-frames with no geometry are NOT mostly "globally wrong
poses". Anatomy of brighton's 21,875 detection-frames: 46.2% are calibration-accepted-but-zero-output,
and **90.3% of THOSE are CLOSE-UPS (<8 players detected)** — geometry-less frames carry a median of
**4** detected players (good frames: 11). Genuinely bad poses are only **4.5%**. A borrowed pose is
fine; four players is not eleven. **Hard ceiling on ANY pose-borrowing mechanism: +9.3 pp.**
**Therefore: the ~37% geometry yield is not a bug — it is approximately the live-wide-play fraction
of a broadcast.** The camera simply is not showing a tactical view most of the time. Carry-over
worked for the BALL because the ball needs one point projected and faces no min-player frame gate.

**Downstream with carry ON** (OFF column reproduces known baselines exactly — harness validated):
passes 213/198 -> 218/203 (recall proxy 47.8/48.6% -> **48.9/49.9%**, still under the 50% gate);
possession Utd 44.2% -> **44.4%** (oracle 52%). Acid test: the **121 newly recovered possession
samples carry Utd at 47.9%** vs 44.3% pre-existing — directionally exactly what trackability-bias
theory predicts, but n=121 and the 95% CI (+-8.9 pp) contains the pre-existing share:
**corroborates, does not overturn, `possession_debias_probe.md`.**
WC no-regression: de-biased line pooled 5.5 -> 5.5 m (every match a hair better). **DECISION: do NOT
enable pose-carry by default** (+1.7 pp yield is not worth the complexity + an untested zoom/pan
failure mode — faithfulness was only validated on wide shots). Code stays: validated, tested,
flag-gated, revisit after the live-play filter.

**SECOND CORRECTION — the FIFA-validated line number was stale.** Re-measured with `tools/line_c6`:
**raw 16.4 m -> de-biased 5.5 m** (we have been quoting "17.1 -> 5.2"). The claim "validated to about
5 m" still stands; the exact figures were out of date (likely pre-2026.07.3). Use 16.4 -> 5.5.

**STRATEGIC CONSEQUENCE — the CV plumbing programme is now COMPLETE.** Detection, tracking,
calibration, ball detection, ball linking, ball carry-over, de-biasing: all levers pulled, each one
measured end-to-end. What remains is not plumbing — the broadcast physically does not contain more
tactical geometry than we are already extracting. **The live-play filter is RE-PRICED: it cannot
create positions on close-ups; it can only remove them from the denominator (an honesty/reporting
fix, NOT a data fix). It will NOT by itself push pass recall past the 50% gate.** Every further gain
must come from LEARNED MODELS ON LABELLED DATA — player identity (jersey numbers) and event spotting
— i.e. Layers 2 and 3 of `docs/CAPABILITY_LEDGER.md`, trained on SoccerNet. That is now the project.

**OPEN DECISION for the user/prof (do NOT slip this through silently):** the 50%-recall gate was set
for EVENT-level work (VAEP/xT need most events). Our pass capture is ~49% but **highly symmetric
between teams** (47.8 vs 48.6), which makes *relative/comparative* claims defensible even at partial
capture. That is a different, weaker claim needing a different, pre-committed bar — legitimate as a
principled distinction, illegitimate as goalpost-moving. Must be decided explicitly and in advance.

## CORRECTION — "calibration 91%" was the wrong denominator; true usable geometry yield is ~37%

Caught while fact-checking the demo video (the render forced a per-frame audit). **What 91% actually
measured:** 91.8% of *detection-frames* SOLVE a homography, and those solutions are genuinely
excellent (median keypoint reprojection **0.21 m**, 99.2% <= 2 m). **What it does NOT measure:**
usable output. Only **37.4% of detection-frames (28.8% of all sampled frames) yield player pitch
coordinates** — on the remainder the pose is *globally wrong* (fits the keypoints, projects players
off-pitch), and the pipeline correctly discards it. This is the SAME error class as the ball-coverage
retraction: quoting an intermediate instead of the usable end product. Corrected in
`docs/CAPABILITY_LEDGER.md`; earlier STATUS entries quoting "calib 91%" as yield are hereby
superseded. **Nothing downstream is invalidated** — every metric was always computed only on frames
with valid pitch coords (i.e. on the honest ~37%); the error was in REPORTING, not in the pipeline.
The whole-match geometry yield reconciles with the known live-play fraction (~41% live x ~65-90%
calibration on live play ~= 27-33%).

**THE LEVER THIS EXPOSES (now the biggest available win):** the ball carry-over mechanism (borrow a
good homography from a camera-continuous neighbour within +-2 s) lifted BALL coverage 20.8 -> 38.8%
and was faithfulness-validated (median 0.33 m). **The identical mechanism has never been applied to
PLAYER projection** — yet player geometry yield (37.4%) is the bottleneck under EVERY metric. If
pose-carry-over works for players as it did for the ball, it lifts possession, passes, phases,
pressing and structure simultaneously, and may reduce the possession bias itself (more transition
frames become trackable). NOTE the distinction from the disproven "mechanism 2" in
`results/pl_probe/diagnosis/DIAGNOSIS.md`: that tried to REUSE the frame's own (globally wrong) pose;
this BORROWS a known-good pose from a neighbouring frame. Different mechanism. Probe delegated.

## 5-MINUTE CV DEMO VIDEO SHIPPED (prof deliverable)

`results/demo/cv_pipeline_demo.mp4` (5:30, 1080p, 161 MB) + `_720p.mp4` (45 MB); tool
`tools/make_demo.py` (segment plan is a data table; `--preview`). Sections: detection+tracking ->
calibration (pitch model projected onto the broadcast, INCLUDING two real failures) -> top-down
reconstruction -> ball tracking (with explicit "ball not tracked" / hollow-marker-for-inferred
provenance) -> validated structures -> honest scorecard. Every on-screen number read from artifacts
at render time. The worker independently caught the calibration-denominator problem above while
sourcing its figures. 254 tests pass; ruff clean.

## FIRST PL REPORT V2 SHIPPED — brighton_manutd, gate honestly enforced (narrow ABSTAIN)

`results/report_v2_brighton_manutd.{md,html}` (deep-worker requested: opus). Report_v2 generalized
to club matches: focus team = registry `teams[0]`, no-roster degradation (team-level prose),
**oracle appendix generalized** (FIFA PMSR for WC / Sofascore for PL — "oracle only, not used
above"), no C5 anywhere in a PL body (WC-pooled model, not applicable). **Gate inputs now read from
persisted per-match eval artifacts** (`outputs/eval/<match>_ball_eval.json`, provenance-stamped;
the queued GATE_INPUTS-constant follow-up, done properly; France reports regenerated byte-similar
apart from the provenance line). Brighton gate readout: coverage 51.9% clears by 11.9 pp; **pass-
recall proxy 48.2% misses the pre-declared 50% bar by 1.8 pp → ball families WITHHELD; gate not
bent.** Structural sections render in full (3-5-2 shape read, de-biased line 33 m, lane occupation
49/33/18, compactness 11.4, synchrony 72%) with position-only seams. Guardrail: 100% on all four
bodies (38/77/38/38). Full suite 244; ruff clean. **Open validation item:** our shape classifier
reads Utd 3-5-2 vs official lineup 4-2-3-1 — in-play shape vs nominal formation can differ
legitimately, but PL formation output is unvalidated; queue a formation-vs-lineup check across
Phase-B matches. **Phase-B lever confirmed by the near-miss: the live-play filter is what pushes
recall past 50% and un-gates the PL ball families.**

## POSSESSION DE-BIAS PROBE — NEGATIVE: estimator declared uncorrectable without event data

Probe (deep-worker requested: opus; `tools/possession_debias.py`,
`results/possession_debias_probe.md`): can trackability bias be corrected with position/timing
geometry, validated LOO against oracle possession? **Oracle base = 4 matches** (3 FIFA PMSR
re-normalised + brighton Sofascore; mun_mci has NO pmsr and NO linked ball — excluded, stated
plainly). Raw errors: iraq +5.9, senegal +0.7, norway +3.1, brighton -7.8 pp (median 4.53) — the
bias consistently inflates the POSITIONAL side, whichever team that is. All 3 pre-committed
candidates REJECTED: zone post-stratification is null (max move 0.67 pp — `true_time` and
`tracked_poss` coincide within every zone); chunk-time weighting null-to-negative; spell
gap-extrapolation improves the median (3.74) but breaches the no-match-worse->2pp guard on norway
and is mechanically a centre-pull toward 50/50, not a bias fix. **Load-bearing evidence:** in
brighton, Utd sits below its oracle share in EVERY third — the missingness is informative *within*
every stratum (whole untrackable transition spells), which no reweighting can recover; only
touch/possession EVENTS could. **Decision: keep "trackable-frame possession share" as a caveated
descriptive metric; no correction ships; the validated fix is the future event layer.** This
negative is thesis material for the uncertainty-aware framing (we can measure exactly what the
broadcast supports, and we can prove where it stops).

## PHASE A CLOSED — Sofascore oracle built (FBref-independent); final gate: pass-volume PASSES

**`tools/oracle.py` (deep-worker requested: opus)** — per-fixture team aggregates with Sofascore as
PRIMARY (adapted from the user's proven ScraperFC route in the read-only sibling
`../mufc-rodri-search`), cache-first under `outputs/oracle/`, FBref cached pages as cross-check
only (live FBref is 403-dead). Empirically verified API surface: `get_match_dicts(year, league)`
(381 PL 24/25 events) + `scrape_team_match_stats(match_id)` (125-row stats table; keys
ballPossession/passes/accuratePasses/totalShotsOnGoal/shotsOnGoal/expectedGoals). Brighton fixture
= Sofascore id 12436888. **Cross-validation Sofascore vs FBref: EXACT agreement** on possession
(48/52) and shots (14/11), zero flags; passes + xG are Sofascore-only (FBref passing page was never
cached). 20 new tests; full suite 236; ruff clean.

**Final Phase-A gate table (`results/pl_pilot/fbref_gate.md`):** pass-volume (like-for-like) —
our CV 213/198 passes vs oracle 446/407 completed = **recall proxy 47.8% (Utd) / 48.6% (Brighton),
team-symmetry spread 0.009 (band <=0.05) -> PASS** (symmetric capture keeps relative pass features
unbiased). Absolute recall ~48% whole-match sits just under the WC 50% event bar BEFORE any
live-play filtering — the Phase-B filter should lift it; event-layer viability for PL is therefore
promising but not yet claimed. Possession reported caveated outside pass/fail (both our proxies
invert the oracle's 52/48 — the documented trackability bias, and precisely why it left the gate).

**PHASE A VERDICT: PASSED** — footage quality excellent, calibration passes live-play-conditional,
detector fine-tuned (87% held-out), coverage lever proven (51.9% full-match), oracle independent of
FBref, pass-volume gate green, possession estimator honestly reclassified. Next: Phase B decision
point with the prof (docs/PL_PIVOT_PLAN.md) — batch ingestion of the Man Utd 24/25 season with the
live-play filter as the first engineering task.

## BRIGHTON PILOT COMPLETE — best coverage in project history; possession gate FAILED with cause

Full-match numbers (11 chunks / ~95 min, extraction died twice mid-batch and resumed cleanly both
times): calibration 91% on probe chunk, ~12 players/frame, **post-link ball coverage 51.9% mean
(carry-over +19.6 pp; every chunk 40.2-63.9% — ALL clear report-v2's 40% ball gate)** — better than
any WC match, on a full match. Team anchor Man Utd=team0 (visually verified, frames in
`results/pl_pilot/anchor_check/`). Fact store `outputs/facts/brighton_manutd.json` born under
2026.07.3 semantics; FBref oracle fetched via soccerdata/selenium (direct scraping is 403-blocked —
confirmed; cached pages in ~/soccerdata).

**Possession gate: FAILED end-to-end, cause diagnosed (deep-worker requested: opus;
`results/pl_pilot/possession_diagnosis.md`).** CV ball-tracked possession Utd 44.2% vs FBref 52%
(~8 pp; both halves skew identically). Ruled out: anchor flip (visual proof); carry-over lever
(contributes ZERO possession samples — facts.py's calib<=1m player gate excludes exactly the frames
the lever adds ball on; coverage win and possession bias are decoupled). Verdict: **genuine
trackability bias** — possession is only measured on well-calibrated ball-tracked frames, which
over-represent settled build-up (Brighton) and under-represent direct/transition play (Utd); same
mechanism as mun_mci's 82/18 over-skew. Note honestly: senegal's WC "possession 48 vs FIFA 49.4"
agreement was likely luck, not validation. **Protocol decision:** ball-possession-share is a
biased-by-construction estimator ("trackable-frame possession share") — it leaves the pass/fail
oracle gate (reported caveated instead), and the metric gets renamed in the fact store. This is NOT
tolerance-widening-to-pass: the estimator measures a different quantity than Opta possession; the
replacement like-for-like oracle gates for PL are pass volume (recall proxy) and, once the event
benchmark exists, event-level agreement. De-biasing (phase-weighted coverage correction) is a
candidate metric but ships only with a validator, per house rules.

## FPS=50 HARDCODE FIXED (METRICS_VERSION 2026.07.3) — modest corrections, no claim overturned

The brighton pilot exposed `report/facts.py` hardcoding `FPS = 50.0` for ALL time-based metrics
(set pieces, counterpress curve, pressing intensity, synchrony, space, ball-xT) — the same bug
class as the pass-window fix, live since the fact store was built. Fixed (deep-worker requested:
opus; the worker was session-killed twice mid-verification and the main session completed the
verification): facts.py now uses per-chunk `Match.chunk_fps()` for chunk-level metrics and the
per-match median fps for aligned-level metrics; **METRICS_VERSION 2026.07.2 → 2026.07.3**; all 4
fact stores regenerated + integrity-checked (287/287/286/124 leaves, no torn writes from the
double regen).

**Honest impact (senegal, 59.94 fps native, worst-case ~17% window error):** counterpress regain
curve 66/77/84 → **70/81/86%** (3/5/8 s), counterpress rate 89 → 91.4%, mean recovery 0.9 → 0.82 s;
pressing intensity ~unchanged (48.3%); synchrony unchanged (71%). Directionally consistent
(windows were too short before), **no sign flips, no FIFA-validated claim overturned** — but the
published seam numbers changed, so all 3 report-v2 outputs AND the 3 v1 pundit reports were
regenerated (report-v2 bodies re-audited: 100/100/100%). Guardrail eval: precision 150/151 = 99.3%
— the single "miss" is an eval false-positive on v1 pundit's C5 text ("~64%", a validation
constant, not a fact-store leaf; same known class as the count-kind edge; report-v2 handles these
via declared non-body inputs). Queued: the worker's full per-consumer fps-semantics trace
(in its transcript, undelivered due to the session kill) — re-request after limits reset.

**Brighton pilot status:** extraction died at 3/11 chunks (process kill, not an error); relaunched
detached via `tools/pl_pilot_run` (resumable, auto-continues align → ball → facts → FBref gate)
with a fresh stall-watch monitor. The brighton fact store will now be generated under the FIXED
2026.07.3 semantics — born correct.

## COVERAGE LEVER PROVEN — temporal homography carry-over ~doubles PL post-link coverage

**Post-`link_ball`, lever OFF vs ON, all 9 probe segments (deep-worker requested: opus):
pooled 20.8% → 38.8% (+18.0 pp); every segment improves** (brighton 29.3→47.3, fulham 19.2→36.5,
liverpool 13.8→32.5 match means; best segment 68.4%). Faithfulness by leave-one-out xval (n=656):
carried projections reproduce own-frame ball positions to **median 0.33 m, 90% within 2 m**; all
carried samples off-pitch-gated + speed-clamped (0 off-pitch survivors). Implementation
(`generator/ball_carry.py`, flag-gated, DEFAULT OFF — WC path untouched): reuse the nearest
known-good player-correspondence homography within ±2 s, never across a camera cut (cut detector =
track-ID Jaccard < 0.30 between sampled frames — a cut resets ByteTrack; validated). 9 new tests;
full suite 216; ruff clean. Why this succeeded where the retracted WC temporal-H failed: WC gaps
were camera cuts (corr=0, uninterpolable); PL failures are continuous-camera midfield ambiguity
with a good homography a fraction of a second away — different mechanism, and this time measured
end-to-end post-link. **Premise correction from the probe:** the diagnosis's "calibrated-but-<6-
projected frame-edge" bucket was actually globally-wrong homography poses (players project 2 m+
off-pitch, bimodal 0-or->=8) — mechanism 2 (reuse PnLCalib calibration) recovers zero frames,
disproven by construction. Caveat: fulham/liverpool still sit under report-v2's 40% ball gate at
the raw-slice level; the Phase-B live-play filter should lift them (non-live frames dilute the
denominator). Artifacts: `results/pl_probe/carry_probe.md`, feasibility.md Stage 4, cached
detections `outputs/pl_probe/<m>/<seg>_ball_imgxy.parquet`. **Verdict: Phase B ingestion includes
carry-over.**

## v6 DETECTOR SHIPPED — recall gate smashed; PL bottleneck is HOMOGRAPHY YIELD, not detection

**Stage 1 — zero-shot recall gate PASSED decisively (deep-worker requested: opus).** v5 on the
user's 120 hand-labelled PL frames, scored with the unchanged `validate_ball.evaluate_annotated`
(tol 8 px in the 512x288 grid — the exact v5 protocol): **67.8% pooled** (brighton 77.5 / fulham
70.0 / liverpool 55.3), localization ~4 px native. Zero-shot PL already beats v5's own WC held-out
senegal/norway — the single-production-style bet is paying out.

**Stage 2 — v6 fine-tune** (15 ep from v5, full rehearsal + PL; `tracknetv2_v6.pth`, v5 untouched):
PL held-out **87.0%** (100/75/85.7 per match, 23 frames). WC forgetting check on identical holdouts:
iraq +0.8, norway +4.5, mun +3.2, fra_sen 0.0, **senegal -5.0 pp** (26→24 of 40 — the only breach of
the 3 pp bar; 2 frames on a 40-frame holdout, no broad forgetting, but recorded loudly). Decision:
**v6 is the PL detector; WC parquets stay v5-generated** (no silent swap). OOM note: the first run
was commit-limit-killed at ep8 (~12 GB in-RAM dataset); fixed via optional `--store-dtype float16`
in `finetune_ball.py` (default float32 = v5 recipe unchanged; fp16 verified loss-identical).

**Stage 3 — the honest end-to-end number: post-link coverage moved only 20.2→21.4%** despite
fire-rate +7 pp and 87% held-out recall. **Detection is NOT the PL bottleneck.** Every detection
that reaches projection projects cleanly; the binding constraint is the **>=6-correspondence /
calibrated-frame yield on raw broadcast slices** (27% pooled raw; 68% on live play per the
diagnosis). The lever is Phase B's live-play filter + temporal homography carry-over — NOT more
ball labels. Caution recorded: temporal-H was tried once on WC and retracted (zero post-link
difference there — but that failure mode was camera-cut gaps with corr=0; the PL failure mode is
continuous midfield stretches where calibration is ambiguous, a genuinely different mechanism).
Any yield claim must again be post-link, end-to-end. 207 tests pass (incl. the worker's pending
targeted set, run after the classifier outage). Artifacts: `results/pl_probe/feasibility.md`
(v6 column; zero-shot table preserved), `results/pl_probe/v6_finetune.log`, updated
`manifest.csv`/`holdout_split.csv` (WC holdout byte-identical, backups kept).
**Phase A: COMPLETE except the FBref agreement check, which rides on the first full-match process.**

## PHASE A FEASIBILITY: PL FOOTAGE PASSES (calibration "failure" was a sampling artifact)

User supplied 3 full Man Utd 24/25 PL Archive replays (Brighton a, Fulham h, Liverpool h) —
**1920x1080 @ uniform 25 fps, ~6.1 Mbps** (2.25x WC pixels, 2x bitrate; one production style; the
uniform fps also removes the 25-vs-59.94 time-semantics class of bugs). Probe: 9 x 2-min segments
through the EXACT existing pipeline (`tools/pl_feasibility.py`, deep-worker requested: opus;
artifacts `results/pl_probe/feasibility.md`, dense+ball parquets under `outputs/pl_probe/`).

Raw gate table looked mixed: players/frame 7.3, calibrated 80%, **>=6-corr yield 27% (vs the >=41%
pre-committed gate — the norway-killer metric)**, zero-shot ball fire-rate 53%, zero-shot post-link
coverage 20.2%. A CPU diagnostic (`tools/pl_probe_diagnose.py`, deep-worker requested: opus;
`results/pl_probe/diagnosis/DIAGNOSIS.md`) settled the calibration question: **sampling artifact,
not geometry.** Raw 2-min broadcast slices are only ~41% live wide-camera play (replays, close-ups,
graphics), while the WC gate was measured on hand-picked wide-play chunks. **Live-play-conditional:
>=6-corr yield 68% (gate 41% — PASS on all 3 matches: 68/56/81%), PnLCalib solve 81% (vs iraq 88%).**
Residual live-play failures: 59% midfield-centered shots (halfway line + center circle only — few
line features; production temporal-reuse recovers most), 41% calibrated-but-<6-players-projected
(frame-edge projection coverage, not calibration). Two weak segments (brighton seg_2, fulham seg_3)
are the same midfield phenomenon, not a distinct failure. Gate-table lesson recorded in the plan:
apply Phase A gates to play-filtered footage.

**Ball chain read:** zero-shot post-link 20.2% pooled (brighton seg_1: 42%) vs senegal's 10%
PRE-fine-tune (which became 46% after) — the proven fine-tune recipe should clear the bar.
**Still pending (honestly, not proxied): true ball recall needs ~100 user hand-labels
(`tools/annotate_ball.py` on the probe segments); FBref aggregate agreement needs one full-match
process.** Phase A verdict: **effectively PASSES** pending the recall label check — proceed to the
annotation batch + one-time PL fine-tune.

## REPORT V2 SHIPPED (CV-primary, gated, FIFA = oracle only) + BTP/PL PIVOT PLANNED

**Report v2 (`report/report_v2.py`, deep-worker requested: opus; cosmetic fix fast-worker requested:
sonnet)** — the audit's end-goal deliverable, and the WC-France closing artifact for the prof demo.
Architecture: the BODY uses ONLY CV fact-store numbers (formation, de-biased line, our phase
classifier, pressing/counterpress, ball-xT, lane occupation, counter-structure seams); **ball-derived
families render only past a pre-declared evidence gate** (coverage ≥40% AND pass-recall proxy ≥50%)
and otherwise print explicit abstention lines; **FIFA PMSR appears ONLY in a validation appendix**
("oracle only, not used above") — fixing v1's habit of quoting FIFA in the narrative spine.
Results: senegal **PASSES** the gate (46% / 55.0%) — the showcase; iraq (36% / 22.8%) and norway
(38% / 10.6%) abstain on ball families, structural sections render everywhere. Guardrail on every
body: **100%** (77/77, 38/38, 38/38). Five falsifiable "how to play against France" seams on senegal
(counterpress plateau after 5 s, thin 22% wing occupation, 30 m high line, 130/677 high-regain
conversion, C5 territory tendency — labeled pooled/directional). 202 tests pass; ruff clean.
Artifacts: `results/report_v2_<match>.{md,html}` (self-contained HTML). Known follow-up queued: gate
inputs (coverage/recall/C6/line-error) are a declared provenance-commented constant, not re-derived —
should read from a persisted per-match eval artifact once one exists.

**BTP / PREMIER LEAGUE PIVOT (user + prof decision).** The project is now a 2-semester IIIT Delhi
BTP (12 credits, 8+4; reviews **Dec 2026** and **May 2027**). After WC-France closes, pivot to the
**PL Archive 24/25 full replays — Manchester United season** (~38 matches → ~76 team-match obs,
fixing C5's N=8). Full roadmap: **`docs/PL_PIVOT_PLAN.md`** — Phase A feasibility gate (3 replays,
pre-committed thresholds) BEFORE scale; Semester 1 = validated season-scale Utd fingerprint (oracle:
FBref/Understat aggregates + self-built event benchmark + home/away consistency — no PMSR for PL);
Semester 2 = real opponent-conditioned synthesizer + scored seam forecasts. College GPU cluster
likely. Licensing hard line unchanged: lawful access, local academic processing, never redistribute.

## TIME-SEMANTICS FIX SHIPPED — measurement verdict NEGATIVE, premise inverted; P3 stays parked

**The code (deep-worker, requested: opus; measurement run by main session after the worker hit the
session limit):** possession debounce and pass max-gap are now expressed in **seconds**, converted
per-chunk from the video's true fps (`generator/ball.py assign_possession(debounce_s=0.3, fps=...)`;
`fingerprint/possession_metrics.py extract_passes(max_gap_s=PASS_MAX_GAP_S=0.9, fps=...)`;
`core/registry.py Match.chunk_fps()`); wired through `report/facts.py` and `tools/event_coverage.py`;
legacy frame/sample args kept for back-compat. 189 tests pass; touched files ruff-clean.

**The verdict (pre-committed setting, NOT tuned to FIFA): the gate FAILS and the premise inverts.**
Pass-recall vs FIFA at a uniform 0.9 s window: **iraq 39.1→22.8%, senegal 55.0→55.0% (held),
norway 17.1→10.6%.** The old frame-based window (50 native frames) meant **2.0 s** at iraq/norway's
25 fps but **0.83 s** at senegal's 59.94 fps — so iraq was never "compressed" by time semantics; it
was **subsidized** by a 2.4× longer pass window. Under consistent physics, senegal is the only
event-viable match and iraq/norway's true event yield is *lower* than previously reported. The
possession-debounce half of the fix is a no-op for these numbers (the pipeline's possession path uses
the Viterbi smoother, which bypasses debounce) — spell counts are unchanged (iraq 143, sen 440).
Phase-% C6 no-regression confirmed: 16.2 / 9.7 / 14.3 pp, mean **13.4 pp** — identical to baseline.

**Decision:** the seconds-based semantics STAY (they are the honest, cross-match-consistent
measurement; the old iraq/norway recalls were apples-to-oranges) and the headline recall numbers are
corrected to **iraq 22.8% / senegal 55.0% / norway 10.6%**. P3 event/value layer remains **parked**
(1/3 matches viable; the bar is ≥2). All 4 fact stores + 3 pundit reports regenerated under the new
semantics; guardrail re-check on the fresh reports: precision **100% (151/151)**, adversarial recall
99.0% (the known count-kind eval edge, still queued). **Next deliverable: report v2** (RAG + counter-structure seams — the audit's end-goal), per the standing
branch decision. Remaining honest lever for iraq/norway events: ball-track *continuity* (their
sparse effective sampling breaks carrier hand-offs), not window tuning — widening the window to
chase FIFA totals would be fitting to the validator and is explicitly rejected.

## BALL CHAIN HEALED — v5 detector (user annotations) + linker runaway-death fix; P3 re-test running

Two-stage unblock, each stage measured honestly (all coverage numbers are **post-`link_ball`**):

1. **v5 detector — the domain gap is closed.** The user hand-annotated 510 frames (431 ball-visible)
   across 3 senegal + 3 norway chunks; a deep-worker (requested: opus) fine-tuned v5 from v4 with the
   full iraq/mun/fra_sen corpus mixed in as rehearsal (no forgetting), held-out split persisted
   (`data/ball_annotations/holdout_split.csv`). **Held-out recall v4→v5: senegal 45→65%, norway 25→50%,
   iraq 70→80%, mun 75→86%** (localization ~1 px). Gate (sen/nor ≥50%, iraq within 3 pp) **passed** —
   confirming the forensic verdict that senegal/norway were detector-domain-gapped, not footage-bound.
2. **Linker runaway-death fix — the collapse the honest probe exposed.** At correct per-video fps the
   probe showed senegal 2.9% coverage despite 69% detection. deep-worker diagnosis (instrumented, not
   guessed): a candidate gap lets `dt` grow unboundedly while a jitter-inflated stale velocity (41.9 m/s,
   above the 40 m/s physical cap) extrapolates the prediction ~627 m off-pitch — the greedy tracker never
   re-acquires (704–1956-frame death streaks). My fps-mismatch hypothesis was INVERTED: fps=50 had been
   accidentally masking the bug. **Fix (generator/ball.py `link_ball`): re-seed after a long gap +
   velocity clamp** — minimal, physical, with 2 decisive regression tests.
   **Probe coverage: senegal 2.9→54.4%, norway 16.8→37.9%, iraq 39.9→47.0% (no regression).** 185 tests.

**REGENERATION DONE + P3 VERDICT (2026-07-10).** All 37 chunks regenerated (`tools/regen_ball.py`,
registry-driven, v5 + fixed linker + true fps; 29 backed up as `.v4bak`, 8 chunks gained ball data for
the first time). Post-link coverage: iraq 23.3→**35.9%**, senegal 10.1→**45.8%**, norway 13.2→**38.0%**.
**Event pass-recall vs FIFA: iraq 24→39.1%, senegal 12→55.0%, norway 1.3→17.1%** (mean 12.4→37.1%).
Phase-% C6 improved to mean **13.4 pp** (senegal 9.7 — best yet). 185 tests pass.

**Honest P3 verdict:** senegal **clears** the ~50% event-viability bar; iraq is ~10 pp shy; norway is
sub-viable — its blocker is now **per-frame homography yield** (min-6-correspondence fails on several
chunks; broadcast quality, not the detector). Per the 3-match FIFA-agreement rule, the event/value layer
stays **parked** until ≥2 matches clear — but the ball layer itself (possession, phases, counterpress,
ball-xT, set pieces) is now dramatically richer on ALL three matches and flows into the fact store.
Pass-count caveat: recall is a **volume proxy** (carrier hand-offs vs FIFA totals), not per-event matched.

**Iraq +10pp probe — clean NEGATIVE result, real lever found (deep-worker, no code shipped).** The
projection-yield hypothesis was disproven with data: on *calibrated* frames iraq's ball coverage is
already **83.3% (highest of the three)**; correspondence counts are bimodal (0 or ≥8, so min-pts
relaxation recovers exactly zero frames; corr=0 = camera cuts, not interpolable); projection in the
shipped path is lossless. iraq h2_005 = truncated 344 KB broadcast stub — correctly empty. **The actual
recall gate is possession→pass conversion**: iraq 143 possession spells vs senegal 440 at similar
coverage (mean pass 8.3 m vs 5.5 m) — sample-based params (`POSSESSION_DEBOUNCE`=3 samples,
`max_gap_frames`=50) enforce ~0.75 s physics at iraq's ~4 Hz sampling vs ~0.3 s at senegal's ~10 Hz.
Fix direction: uniform TIME-based semantics (seconds), justified by cross-match consistency. Norway has
the same compression disease → the fix could lift both. Next scoped task delegated.

**Downstream refreshed on the new ball data (fast-worker):** all 4 fact stores + 3 pundit reports
regenerated. France's ball-tracked facts are transformed — e.g. senegal: 680 France passes detected
(was ~71 pre-v5), ball-xT over 3,598 tracked advances, counterpress 89%. Guardrail re-check on the new
reports: **precision 100% (151/151)**; adversarial recall **99.0% (594/600)** — 6 count-category
synthetic fabrications slipped (pre-existing eval-harness edge, precision unaffected). **Known issue
queued:** tighten the count-kind matching/eval generator. 185 tests pass.



## BALL-COVERAGE: attempted a "fix", RETRACTED it — measurement error caught (no coverage win)

Revisited "can we get events (P3) to work without new footage?" Tried two levers (temporal-homography
projection + lower detector threshold) and **initially claimed a ~2× coverage lift — that was wrong, and
I retracted it.** The claim rested on measuring **raw projected-frame count**, not the **usable track
after `link_ball`** (the physics/speed gate that produces the real ball trajectory). The end-to-end
regeneration told the truth: on iraq chunk_002 the existing parquet has **610 clean detections → 950
linked (40%)**, while the lever pipeline gave **1407 noisy detections → only 128 linked** — the physics
gate correctly rejects 87% of them, and **temporal-H made zero difference post-link (128 either way)**.
Root cause: my "baseline" used `ball_possession`'s 512×288 *downscaled* detection (noisy); the existing
parquets already use **native-resolution** detection, which is cleaner and better. So **there is no easy
coverage win — the existing pipeline is already the best we have.** Reverted the threshold change, removed
`project_ball` + the tools/tests built on the flawed premise. Lesson: for ball coverage, always measure
the **post-`link_ball`** track, never the pre-link projection count.

**What survives (genuine, still-unproven):** the *forensic domain-gap* finding for senegal/norway. Their
low ball coverage is NOT footage-bound — all 3 matches are **1280×720**, and the visual pilot confirmed
the ball is **human-locatable (~6-10 px)** in senegal/norway active play. The smoking gun: the v4 detector
was fine-tuned on **iraq/mun only — zero senegal/norway frames** (`data/ball_annotations/manifest.csv`),
and `finetune_ball.py` states pretrained weights don't transfer. So senegal/norway run an **out-of-domain
detector** → low, *noisy* detection. The real (unproven) lever is **fine-tuning the detector on
senegal/norway** (annotate their footage — no new footage; combined training with iraq/mun rehearsal to
avoid forgetting, per Fable). That targets detection *quality* (clean detections that link), which is the
actual bottleneck — unlike the projection lever, which was a dead end. Not yet attempted.



## P3 EVENT LAYER — SHELVED (evidence-backed) + P4 GROUNDED-REPORT GUARDRAIL shipped

**P3 (event layer) shelved on measured evidence, not a guess.** The plan was T-DEED/tracking-derived
events → VAEP. Before building, measured actual event yield ([tools/event_coverage.py](tools/event_coverage.py),
via a Sonnet subagent): **pass recall averages ~12%** (iraq 24%, senegal 12%, norway 1.3% = 7 passes),
gated almost linearly by **ball-track coverage ~33%**; shots are undetectable (nothing separates a pass
from a shot in tracking). Fable proposed a ball-coverage interpolation spike as the unblock; I measured the
gap structure first ([the go/no-go]) — the residual loss is **~100% long blackouts** (>3 samples: only
8/2/1 short-gap frames remain across entire matches — `link_ball` already fills those). Long blackouts
(broadcast cuts / far-camera / tiny-ball) are not honestly interpolable and no ball-anchored gain remains
to bank. Verdict, confirmed with Fable: **shelve the event layer** (blocked on ball detection in
blackouts, not fixable by interpolation on 4 GB); an event-based possession value at 12% recall with
non-random missingness would be noise, violating the project's validated-or-nothing rule. Revisit only if
ball detection during blackouts improves. VAEP/EPV/OBSO stay out of reach — honestly.

**P2.5 consolidation — C5 migrated onto the de-biased line, and it IMPROVED** (Fable's "land the sure
thing" sequencing). [synthesizer/opponent_model.py](synthesizer/opponent_model.py) `build_observations`
now takes `opp_def_depth` + the `def_line_height` target from the P2 de-biased estimator
(`generator.impute.line_estimates`) instead of the biased 20th-pctile line. LOMO skill went **2/4 → 3/4
tendencies**: attacking-third +43%→**+48%**, def-line +40%→**+64%**, wing_share −14%→**+1%** (flipped
positive), width −14%→−8%. The more accurate opponent-depth feature strengthens the opponent-conditioning
— the report is now coherent (de-biased line everywhere). `results/c5_opponent_model.png` regenerated.

**P4 GROUNDED-REPORT NUMERIC GUARDRAIL — the honesty layer** ([report/guardrail.py](report/guardrail.py)).
Fable's steer: *a grounded report's honesty is the guardrail's recall, not the LLM's fluency.* The guardrail
flattens the fact store + FIFA PMSR into unit-typed grounded values (m / pct / xG / s / count — with 0–1
shares also grounded as %), extracts every number from candidate report prose, and flags any that no
grounded fact backs within a unit-appropriate tolerance. **Validated**
([tools/guardrail_eval.py](tools/guardrail_eval.py)): **precision 148/148 = 100%** on the 3 real pundit
reports (zero grounded numbers wrongly flagged) and **recall 600/600 = 100%** under adversarial injection
(fabrications out-of-range per unit — none survive). Wired into [report/narrate.py](report/narrate.py):
the LLM narration is post-checked, ungrounded numbers `[?…]`-annotated inline + summarised, so a
hallucinated stat cannot silently reach the reader. Unit-typing was essential — a flat match against the
126-fact store let 7/8 fabrications through (dense-net false negatives) until numbers were typed by unit.
**183 tests pass** (+5 guardrail). Also extracted per-phase FIFA line heights in P2 remain the ground
truth. **Next:** structure the report around the counter-structure seams (theory codex) + retrieval.



## P2 OFF-SCREEN VALIDITY FIX DONE — de-biased defensive line, validated vs FIFA to ~5 m (audit roadmap phase 2)

The validity cornerstone: broadcast follow-play shows ~6/11 players and drops the deep defenders, so the
line reads ~+15 m too high in attacking phases (the pipeline used to hide this with a hand
`PARTIAL_VIEW_LINE_OFFSET`). Now removed with a data-driven, FIFA-validated estimator.

- **Bias proven** ([tools/impute_diagnose.py](tools/impute_diagnose.py)): line reads 52 m at ≤4 visible →
  38 m at ≥9 visible (Spearman −0.21) — pure censoring artifact, not a metric error.
- **FIFA per-phase line-height ground truth extracted** ([tools/parse_pmsr.py](tools/parse_pmsr.py), via a
  Sonnet subagent): the PMSR "In Possession / Defensive Line Height" pitch-graphics → per-phase line height
  for both teams incl. the **in-possession** phases where the bias lives. Verified against known anchors
  (France final_third 59–62, mid_block 38, low_block 18). This is now a permanent validation asset.
- **Final estimator** ([generator/impute.py](generator/impute.py) `line_estimates`): line = **mean of the
  deepest-4 outfielders** (GK excluded — the back line itself, not a 20th-pctile-of-all that sits ~12 m
  shallow), then **de-biased by ONE global slope** on *back-line visibility* (`n_back`, b = −5.6 m per
  back defender seen, fit on 153 k pooled frames via [tools/fit_line_debias.py](tools/fit_line_debias.py),
  NOT the FIFA matches). Correcting each frame to a "full back line seen" view removes the censoring
  inflation while **keeping the phase-to-phase shape**.
- **Validated** ([tools/line_c6.py](tools/line_c6.py)): per-phase |line − FIFA| pooled over 3 France
  matches **17.1 m → 5.2 m**; de-biased phase spread **43 m ≈ FIFA's ~40 m** (shape preserved). Iraq/Senegal
  nearly all phases <8 m; Norway noisier (634 frames, 10× smaller). Chose `n_back` over `n_visible` because
  it agrees better with *independent* FIFA truth, decisively on high_press (11→4 m) — the censoring phase it
  measures; confirmed with the reviewer.
- **Approaches tried + rejected, honestly**: (a) centroid-offset imputation — improves per-player LOO
  (12.1 vs 14.5 m) so **kept for local metrics**, but can't fix the line (biased anchor); (b) temporal
  back-line reconstruction — over-flattens (back line 96% censored in final_third) so the line loses phase
  variation; superseded by the de-bias for the line metric, retained as a reference.
- **Wired**: de-biased line in the fact store (`cv.line_height.<team>.def_line_debiased_m`) and surfaced in
  the pundit report ("visibility-corrected"). Also fixed a StatsBomb-spec conformance bug: counterpress
  window 1 s → 5 s ([transitions.py](fingerprint/transitions.py)); metrics bumped to v2026.07.1.
- **178 tests pass** (+5 P2). **Follow-up (P4):** migrate the C5 opponent model + tendencies onto the
  de-biased line (they still use the biased 20th-pctile line).

## P1 THEORY METRICS DONE — coaching concepts made measurable + FIFA-validated (audit roadmap phase 1)

Six ball-anchored tactical metrics from the audit codex, each mapped to a FIFA EFI validator.
[fingerprint/theory_metrics.py](fingerprint/theory_metrics.py), wired into the fact store,
validated by [tools/validate_theory_c6.py](tools/validate_theory_c6.py).

- **Metrics:** `pressing_intensity` (Bielsa/Klopp — time-to-intercept model on the ball carrier,
  positions+velocity, P=1−Π(1−p)), `line_breaks` (ball played clearly front→behind the opponent's
  defensive line, hysteresis-banded), `local_overload` (numerical superiority near the ball —
  Juego de Posición), `verticality` (goalward directness of ball progression — Bielsa),
  `lane_occupation` (Guardiola's 5-lane × 3-third matrix + half-space share), `counterpress_curve`
  (regain P at 3/4/5/6/8 s — Gegenpressing). All obey the partial-view rule: ball-anchored / ratios /
  role-relative, never single-frame whole-team shapes.
- **FIFA C6 validation (France, 3 matches):** the **3 clean-mapped metrics validate directionally 3/3** —
  pressing_intensity vs defensive_pressures (rho +0.5), line_breaks vs completed_line_breaks (+0.5,
  absolute undercounts 33 vs 117 as the partial view predicts), mean_overload vs forced_turnovers (+0.5).
  The other 3 have honest validator-mapping caveats (long_ball has ~0 variance for France; lane-share is a
  ratio vs a volume count; our counterpress *success* ≠ FIFA counterpress *frequency*), flagged "weak" in
  the harness — a validator limitation, not a metric fault. Honest scope: n=3, directional not calibrated.
- **Wired everywhere:** fact store `cv.theory.<team>` (pressing, line breaks, overload, verticality, lane
  matrix, counterpress curve for both teams); the pundit "By the numbers" section now leads with pressing
  intensity + counterpress + line breaks + verticality + half-space share. **173 tests pass** (+8 theory).

## P0 PLUMBING DONE — registry + constants + fact store (audit roadmap phase 0)

Post-audit foundation so the project scales past a handful of matches without hand-editing.
Anchor: [docs/PROJECT_AUDIT_2026-07.md](docs/PROJECT_AUDIT_2026-07.md).

- **Central match registry** ([data/matches.yaml](data/matches.yaml) + [core/registry.py](core/registry.py)):
  one YAML, one loader. Killed the hand-duplicated match lists / 2 path conventions across 5 files —
  `opponent_model.py`, `pundit.py`, `france_profile.py`, `phase_pct_c6.py`, `style_matrix.py` all read the
  registry now. Adding a match = one YAML entry. C5 model reproduces exactly through it (8 obs, +43%/+40%).
- **Constants module** ([core/pitch.py](core/pitch.py)): single source for pitch dims (105×68 / 120×80
  contract) + `METRICS_VERSION` stamp. Removed 9 per-file `PITCH_LEN=105.0` copies.
- **FACT STORE** ([report/facts.py](report/facts.py) → `outputs/facts/<match>.json`): runs the WHOLE metric
  inventory over a match — tendencies, phase-time %, transitions/counter-press, passing/PPDA, ball-xT,
  set pieces, velocity synchrony, style distance, space control, formation — CV + FIFA side by side,
  version-stamped. **110–126 numeric facts/match (was 4).** All 4 matches built. Fixed the P0 audit's #1
  finding ("the pundit report consumes 4 numbers") — previously-orphaned transitions/set_pieces/xt/PPDA/
  roles are now wired into the product.
- **Pundit + narrate read the fact store**: new "By the numbers (CV, ball-tracked)" section surfaces
  counter-press %, recovery time, PPDA, ball-xT, synchrony, space control; narrate's evidence bundle
  carries the full ball-tracked block for both teams.
- **Deprecated** `synthesizer/backtest.py` (the n=3 Tier-B-loses artifact) with a warning → use
  `opponent_model.py`. **165 tests pass** (+8 new: registry, fact store). ruff-clean on authored files.

## C5 OPPONENT PREDICTOR — Tier-B BEATS Tier-A (+43% skill)

The headline novelty, working. Key move: pool across **every** processed team-match (each match = 2
observations), giving **8 team-matches** (France x3 + Iraq + Senegal + Norway + Man Utd + Man City) with
**CV-derived features on both sides** (no reliance on FIFA PMSRs → extends to any match).
[synthesizer/opponent_model.py](synthesizer/opponent_model.py), `results/c5_opponent_model.png`,
`outputs/c5_observations.parquet`.

- **Model:** attacking-third share = 0.79 − 0.012·(opponent defensive-line height). Slope < 0 → teams
  commit further forward vs deeper-sitting opponents. Clean fit across all 8 points (France, City attack
  deep blocks; reactive sides vs France's high line stay back).
- **Backtest (leave-one-match-out):** Tier-B (opponent-conditioned) MAE **0.054** vs Tier-A (team avg)
  **0.095** → **+43% skill, Tier-B WINS**. The earlier n=3 France-only failure was overfitting; pooling
  across teams gave the opponent term enough data to generalise. Honest: n=8 is small — a validated
  positive signal + framework, not a significance claim; strengthens as matches are ingested.
- **Wired in:** `synthesizer.opponent_model.forecast(opp_depth)` (full tendency vector); the pundit report
  carries a **C5 matchup-forecast** section (expected vs actual). Demo: France vs 25 m low-block → att3rd
  ~0.48; vs 60 m high line → ~0.06.
- **Multi-tendency (which dimensions are opponent-driven):** LOMO skill per tendency — attacking-third
  share **+43%**, defensive-line height **+40%** (Tier-B wins) but width **-14%**, wing share **-14%**
  (Tier-A wins). **Insight:** the *vertical* game (how high/how committed) adapts to the opponent; the
  *horizontal* game (width, wing focus) is fixed team identity. `results/c5_opponent_model.png` (4-panel).

## SYNTHESIZER v1 — phase calibration + C5 Tier-B backtest + LLM narration

Three follow-ups done:
1. **Phase-threshold calibration** ([fingerprint/phase_metrics.py](fingerprint/phase_metrics.py)):
   principled partial-view correction (+11 m line offset from the measured bias; build-up = own half).
   Phase-% C6 mean abs error **27 -> 16 pp** (Senegal 26->12). Validates the bias diagnosis; not fit-to-FIFA.
2. **C5 Tier-B implemented** ([synthesizer/backtest.py](synthesizer/backtest.py)): leave-one-match-out,
   opponent-conditioned (linear on opp low-block %) vs Tier-A mean. **Honest verdict: Tier B does NOT beat
   Tier A at n=3** (skill -106% to -172%) — the opponent term overfits fitting a line through 2 points.
   The signal exists (the scatter) but out-of-sample prediction is data-gated; the backtest quantifies it.
3. **LLM narration layer** ([report/narrate.py](report/narrate.py)): assembles the grounded evidence
   bundle -> a SYSTEM-prompted narration that forbids any un-grounded claim; calls Anthropic API if
   `ANTHROPIC_API_KEY` set, else writes the ready-to-send prompt. Hand-verified target output:
   `results/narrate_france_iraq.md` (natural pundit prose, every claim traced to a number).

## CLEAN SAME-MATCH C6 (phase distribution) — systematic bias diagnosed

`tools/phase_pct_c6.py` (`results/phase_pct_c6.png`): our France phase-time distribution vs each match's
OWN FIFA PMSR phase % (apples-to-apples). Mean abs error 21-33 pp — **not tight, but the error is
systematic and explained**: (1) we **under-count build-up / over-count progression** (deep build-up ball
reads mid-pitch), and (2) we **over-count high-press / under-count mid-block** because France's defensive
line reads ~10 m too high (deep defenders off-screen, the known partial-view bias). It's a **calibratable
threshold bias, not a broken engine** — the structural layer (line height, shape) is the trustworthy
output; the phase classifier needs partial-view-corrected thresholds (lower HIGH_PRESS_MIN_LINE, raise
BUILDUP_MAX_X) before its % distribution matches FIFA. Ball layer now dense (Senegal 5298, Norway 2295
samples). Honest C6 verdict: structural metrics validate; phase-% classification is directionally right
but biased by broadcast framing.

## FRANCE 3-MATCH PROFILE + C5 OPPONENT SIGNAL (overnight)

All 3 France matches processed (Iraq, Senegal, Norway) — structural CV read + FIFA ground truth.
[tools/france_profile.py](tools/france_profile.py) (`outputs/france_profile.parquet`,
`results/france_opponent_signal.png`):

| match | France line (CV) | att-3rd (CV) | opp low-block (FIFA) |
|---|---|---|---|
| Iraq | 52 m | 0.32 | 46% |
| Norway | 55 m | 0.41 | 36% |
| Senegal | 47 m | 0.24 | 17% |

**France's marginal line = 51 ± 4 m** (consistent identity), with a real **opponent-conditioning signal**:
France commits further forward (attacking-third share) against deeper-sitting opponents — the evidence the
**C5 Tier-B** synthesizer learns from (directional on 3 points). All 3 pundit reports CV-enriched
(`results/pundit_france_*.md`). Ball detection (v4) running on Senegal+Norway -> per-match C6 + possession
next.

## GROUNDED PUNDIT REPORT — the end-goal deliverable, working

Three France group-stage matches in hand with footage + FIFA PMSRs: Iraq (M42, processed), Senegal (M17),
Norway (M61). FIFA PDFs parsed ([tools/parse_pmsr.py](tools/parse_pmsr.py) -> `outputs/pmsr/*.json`:
phases %, key stats, score). France roster ([data/france_roster.json](data/france_roster.json)): squad +
per-match XI/formation + key attackers, for player-ID by role.

**[report/pundit.py](report/pundit.py) writes a grounded pundit analysis** — pundit-style prose where every
claim cites a CV metric or an official FIFA number, naming France's attackers. Demonstrated on all 3:
`results/pundit_france_{iraq,senegal,norway}.md`. France-Iraq sample: "the wide creators Olise and Barcola
had to break Iraq's 46% low block — exactly the matchup France's high, wide shape is designed for"
(grounded: line 52 m, 117 line breaks, 2.30 xG). This is the user's vision (e.g. "Mbappe/Olise/Doue offer
a creative threat to low-blocking teams") working. LLM-narration is the next layer on this grounded bundle.

**Overnight (autonomous):** GPU batch_match processing Senegal + Norway footage (background, task b4w03xwu0);
on completion -> align (kit-anchor) -> v4 ball -> CV-enriched reports + clean per-match C6 (now possible:
own-match FIFA phase % to compare against). 9 new tests pass (parse_pmsr, pundit, synthesizer Tier-A).

## LABEL FIX + FRANCE FOCUS — the "fra-sen" footage is actually France vs IRAQ

The WC footage processed as France-Senegal is **France vs Iraq** (source mislabel). France data is
correct; opponent was wrong. Renamed: `matches/france_iraq/`, `outputs/france_iraq/{h1,h2,final}/`,
`data/ball_annotations/france_iraq/`; scripts `tools/analyze_match.py`, `tools/phase_c6.py`. **France is
now the project focus** — ingesting all group-stage France matches: Iraq (done), Senegal + Norway
(incoming). 3 France matches → unlocks **C5 Tier-B** + a **clean same-match C6** (real Senegal footage vs
the France-Senegal PMSR). The per-phase C6 below was **cross-opponent** (France/Iraq CV vs France/Senegal
FIFA) — defensive-block match still meaningful (opponent-stable), but the clean C6 awaits Senegal footage.
New-match drop: `matches/france_{senegal,norway}/{h1,h2}/`. 150 tests pass.

## PER-PHASE C6 — defensive blocks VALIDATE vs FIFA (within 2-4 m)

The real FIFA validation, finally with a ball (v4 detector, recall 0.25->0.68 after the 714-ball corpus).
France per-phase defensive-line height vs FIFA PMSR (`results/fra_sen_phase_c6.png`,
`tools/fra_sen_phase_c6.py`):

| phase | ours | FIFA | |
|---|---|---|---|
| Low Block | 22.7 | 19 | Δ4 ✅ |
| Mid Block | 40.4 | 38 | Δ2 ✅ |
| Build-up | 24.9 | 44 | Δ19 ⚠️ |
| Final Third | 73.7 | 59 | Δ15 ⚠️ |

**Sharp finding:** **defensive (out-of-possession) phases match FIFA within 2-4 m** — when a team defends
compactly it is fully framed. In-possession phases are biased because the **camera follows the ball**:
build-up frames the deep build (line reads low), final-third frames the attack (deep defenders off-screen,
line reads high). The metric engine is correct; the in-possession bias is a diagnosable camera-framing
artifact, not an error. First genuine external validation the project has produced. v4 ball tracks ~2x
denser (950 samples/chunk). 150 tests pass.

## FIRST CROSS-MATCH STYLE COMPARISON (C5 seed) — works

Two matches ingested → `tools/style_matrix.py` (`results/style_matrix.png`): Wasserstein style distance
between all four teams. **France ≈ Man City (3.2), Senegal ≈ Man Utd (6.5)** — dominant sides cluster,
reactive sides cluster. The Tactical-DNA / opponent-matchup seed working on real data. Pitch control:
France/Senegal ~even space (0.49/0.52). Formation inference still loose (~18m fit) — partial-view ceiling.

## FULL MATCH ANALYZED — France-Senegal both halves, complete labels, line height stable

Both halves re-extracted with the fixed classifier ([generator/extract.py](generator/extract.py)) →
complete balanced labels (no `-1`), kit-anchored France=navy/Senegal=white
([generator/team_anchor.py](generator/team_anchor.py) `align_teams_by_color`). Analysis:
[tools/analyze_fra_sen.py](tools/analyze_fra_sen.py), graphic `results/fra_sen_c6_final.png`.

- **Line height STABLE:** half 2 CV **5%** (half 1 noisier only from kickoff/end boundary chunks).
- **Coherent fingerprint:** France line 53 m, build-up 59 m, **35% in attacking third** (high, dominant);
  Senegal line 32 m, **9% in attacking third** (deep block). France won 3-1. ✅
- **C6:** France overall line 53 m sits **within** FIFA's published France range (19-59 m by phase); width
  34 m just under FIFA's 35-57 m. Reads high vs build-up/mid because **broadcast shows ~6 of 11 players**
  (deep defenders off-frame) — a fundamental partial-view bias, not a labeling/calibration error.
- **Remaining for a clean per-phase C6:** the ball (phase split) — needs ~200 annotated frames on these
  halves → v4 detector. Projection already 45%. 146 tests pass.

## (Earlier) BREAKTHROUGH — clean France-Senegal halves work; root team-classifier bug fixed

New footage: the **two clean halves** (`fra-sen half 1/2.mp4`, 720p/25fps, 57+50 min) — a proper broadcast,
not the earlier 140-min replay package. This turned the project around.

- **Calibration density fixed by footage hygiene:** clean halves + stride 5 → **688-1149 calibrated
  frames/chunk** (replay had 264). MUN-grade. (`matches/fra_sen2/h1`, `outputs/fra_sen2/`.)
- **ROOT-CAUSE bug found + fixed:** `_collect_team_crops` fit the jersey classifier on **one early
  ~80-frame window** then broke out → on navy(France)-vs-white(Senegal) kits it collapsed to **18003 vs
  1142**. That degenerate split was what broke possession AND direction all along. Fixed to sample 6
  windows across the whole clip ([generator/extract.py](generator/extract.py)). `team_anchor` corrects
  existing data post-hoc.
- **With correct labels everything resolves:** possession 50-60/40 (was 100/0); **keeper direction is now
  consistent across the half** ({France:+1, Senegal:-1} every chunk); **line-height CV dropped 24% -> ~3%**
  on core chunks (001-004). France high line ~59 m (dominant), Senegal deep ~28 m, France won 3-1 -
  coherent. `resolve_attack_directions_from_ball` added as a cross-check.
- **Ball layer usable but recall-limited:** projection yield 45% (was 7-17%), 450 linked samples/chunk
  (was 38-132); detection recall 25% (v3 never trained on this 25fps broadcast — annotation lever).
- **Honest C6 status:** line height now stable + directionally correct, but absolute values read high
  (~3-4 of 11 players labelled/frame -> missing deep defenders biases the line up). Full per-phase C6
  needs fewer `-1` unknowns + denser ball. 145 tests pass.

## (Earlier) C6 on the replay: INCONCLUSIVE (direction/identity noise), C6 retracted

Ingested the FIFA PMSR oracle match (France 3-1 Senegal, 720p/30fps) **locally, no Colab**. Calibration
+ direction-agnostic metrics are solid; **direction-dependent metrics are NOT reliable** on this footage,
so the earlier "C6 passed" claim was withdrawn (it matched the **wrong** team — user confirmed
**France = cluster 1**, not 0 — so the 0.4 m "match" was Senegal's numbers coincidentally near FIFA's
France figures).

- **Pipeline generalizes to 720p (real win):** play chunks (004-013) calibrate at 0.18-0.32 m median.
  Chunked via [tools/chunk_video.py](tools/chunk_video.py); outputs in `outputs/fra_sen/` (filter
  `calib_error_m <= 1.0`). Direction-agnostic read (correct labels): **France 56% of the space**, bigger
  surface area, won 3-1 — coherent. Width 35 / length 28-30 m (length in FIFA range; width under-reads).
- **Two fixes shipped (both genuine improvements, 144 tests):**
  1. `resolve_attack_directions` handed **both teams the same direction** (off-screen keeper misdetected).
     Fixed with an opposite-team constraint ([fingerprint/structural_metrics.py](fingerprint/structural_metrics.py)).
  2. Cross-chunk team anchoring rebuilt: `apply_chunkwise_team_labels` clusters within each chunk then
     matches centroids across chunks ([generator/team_anchor.py](generator/team_anchor.py)) — steadier
     identity than pooling every track.
- **But line height is STILL unstable, and the diagnostic pins why:** direction-AGNOSTIC metrics are
  stable across chunks (width CV 8%, length 13%), direction-DEPENDENT is not (line CV 24%). So the data /
  calibration / identity are fine — the blocker is **per-chunk attacking-direction ambiguity**, which is
  fundamental to a follow-play broadcast (one goalmouth visible, keeper-team identity noisy). No keeper
  heuristic fully fixes it.
- **Conclusion:** on follow-play broadcasts, **directional metrics (line height, formation orientation,
  xT) need the BALL to anchor direction**; direction-agnostic metrics (width/length/compactness/surface/
  space control) are trustworthy now. So **C6 line-height on France-Senegal runs *through* the ball
  layer** (detector re-tune), not around it. Wide-framed broadcasts (like MUN) don't hit this.

## Prior session (ball/identity/phase layer + Colab)

Ball layer now end-to-end on all 11 chunks with the fine-tuned **TrackNetV2 v2** detector, plus the
identity + phase prerequisites that were gating the FIFA-style reads:

- **Ball detector validated** ([eval/ball_eval.py](eval/ball_eval.py)): P/R ≈ 0.99, localization
  **~0.2 m** on the chunk_006 holdout. The ~78% "detection rate" is recall + homography yield loss, not
  accuracy — now reported separately in [tools/ball_possession.py](tools/ball_possession.py).
- **Roles/formation inference** ([fingerprint/roles.py](fingerprint/roles.py)): Hungarian-match tracks
  to formation templates → 11 stable role slots/team (Utd 3-5-2, City 4-2-3-1). `relabel_to_roles`
  collapses carrier fragments to nearest role centroid → passing networks now **≤11 role-keyed nodes**
  (was 110/126). Top connector reads "LCB", not "track 64".
- **Ball-driven phase split** ([fingerprint/phase_metrics.py](fingerprint/phase_metrics.py)): FIFA-style
  build-up/progression/final-third + high-press/mid/low-block, per-phase structural metrics. Line climbs
  15→45→77 m through the phases; City presses 14 m higher than Utd ([results/phase_dashboard.png](results/phase_dashboard.png)).
- **Possession smoothing** ([generator/ball.py](generator/ball.py)): Viterbi DP over the team sequence
  (`smooth=True`) — 24% fewer switches, drops the unknown-team noise. Renamed "territorial possession proxy".
- **Colab runner** ([notebooks/football_synthesizer_colab.ipynb](notebooks/football_synthesizer_colab.ipynb)):
  offloads fine-tune + batch_match to a T4. See [docs/COLAB.md](docs/COLAB.md).
- **xT / controlled threat** ([fingerprint/xt.py](fingerprint/xt.py)): geometric xT surface x pitch
  control = off-ball threat (City owns 62% of dangerous space on chunk_006), plus ball-xT progression.
- **Set-piece detection** ([fingerprint/set_pieces.py](fingerprint/set_pieces.py)): gap-restart heuristic
  → boundary-anchored corners/throw-ins/goal-kicks are reliable (12 across the match); free-kick residual
  flagged as low-confidence candidate (dominated by tracking dropouts, not real FKs).
- **Transitions / press triggers** ([fingerprint/transitions.py](fingerprint/transitions.py)): turnovers,
  counter-press rate + recovery time, counter-attack xT, high regains. City counter-presses harder
  (0.77-0.84) than Utd (0.68) — a real Gegenpressing read.
- Results graphics: [results/dashboard.png](results/dashboard.png), [results/verify_topdown.png](results/verify_topdown.png), [results/phase_dashboard.png](results/phase_dashboard.png).
- 142 tests pass. **CV/analytics feature set complete for single-match analysis.** Remaining before the
  end goal: ingest a real WC match (generalization test) → then ≥2 matches for the C5 synthesizer.

## Current focus

**CV foundation rebuilt and validated end-to-end on a full broadcast chunk.** Pipeline: clip →
**football-role detection** (player/GK/referee/ball) → ByteTrack → **jersey-colour teams** →
**PnLCalib full-camera calibration** (centred→uncentred fixed) → quality gates → dense positions →
**tactical-frame table** → **deterministic team-shape metrics**. Validated on all of chunk_000 (10 min
@ 10 Hz): 1803 accepted tactical frames, 15.1 players/frame, 0.26 m calibration. The earlier GAT clip
readout was computed on the **broken** calibration and is invalid — to be re-run on corrected coords.
**Metric engine now live and verified**: [fingerprint/structural_metrics.py](fingerprint/structural_metrics.py)
produces a real per-team fingerprint over the chunk — both direction-agnostic shape *and*
direction-dependent height metrics (attacking direction resolved from the keeper's defended goal, so no
hand-kept match-half table is needed). **C3 attacker run head rebuilt and beats the old 360 baseline**
(RMSE 5.11 vs 6.33 m, hit@3m 0.566 vs 0.244 on chunk_000). **Team-style fingerprint `z_T` built**
([fingerprint/team_style.py](fingerprint/team_style.py)), structured to mirror the **FIFA EFI report**
(now adopted as the oracle/target — see [docs/EFI_ALIGNMENT.md](docs/EFI_ALIGNMENT.md)). Match batch
got **3/11 chunks** before a CUDA-OOM (fix landed). Next: ball/phase layer (the big EFI unlock), then
`report/` v1, then the grounded game-plan synthesizer (C5).

## Done this session

- **Prepped #4 (ball annotation/fine-tune) + #5 (multi-match ingestion).** All wired; how-to in
  [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).
  - [tools/annotate_ball.py](tools/annotate_ball.py): interactive click-annotator (left-click ball /
    Enter=not visible / resumable) over auto-picked live-play frames.
  - [tools/finetune_ball.py](tools/finetune_ball.py): fine-tune TrackNetV2 from annotations (3-frame
    windows + Gaussian heatmap targets) → adapted weights for `WASBBallDetector`.
  - [tools/ingest_match.py](tools/ingest_match.py): one-command per-match pipeline (chunks → extract →
    colour-anchor → facet fingerprint, `--enrich` adds GAT/pitch-control/synchrony) → `outputs/matches/<name>/`.
  - **Footage:** can't download copyrighted video; user provides full-match broadcast as ~10-min
    `chunk_*.mp4`. **Player recognition:** names infeasible (ball/numbers ~3-5 px); **positional roles
    feasible** (formation/Hungarian on tracks) — clean next feature, not built.
  - *Pending the classifier outage:* MUN-vs-MCI team-colour identification run (method ready) + the
    report relabel.
- **Richer team-style read — full GAT heads + pitch control + style axes (roadmap items 1-3).**
  Driven by [docs/ANALYSIS_ROADMAP.md](docs/ANALYSIS_ROADMAP.md) (synthesised from the xG-Football-Club
  corpus + SoP `eval/team_metrics.py`).
  - **#1 Full GAT heads (free):** [generator/sop_bridge.py](generator/sop_bridge.py) `full_reads` now
    extracts **option richness** (receiver entropy → attacking), **press decisiveness** (presser_probs)
    and **lane suppression** (1-xPass → defending) from the same checkpoint; merged per team.
  - **#2 Pitch control:** [fingerprint/pitch_control.py](fingerprint/pitch_control.py) (2 tests) —
    Spearman/Fernández-lite control surface from positions+velocity (no ball) → **space control** &
    **attacking-third control**.
  - **#3 Style axes:** [fingerprint/style_metrics.py](fingerprint/style_metrics.py) (2 tests) — velocity
    **synchrony** + **style distance** (1-D Wasserstein "Tactical DNA").
  - All folded into [fingerprint/facet_metrics.py](fingerprint/facet_metrics.py) (new TERRITORY +
    COORDINATION facets) → `outputs/match_facets.parquet`, report regenerated. **Coherent story:**
    Team 1 = dominant/high-press/territorial (space 0.58, att-third control 0.49, press 0.18); Team 0 =
    efficient counter-puncher (xT 0.038, success 0.56, less space). Style distance 7.56 m. 113 tests.
- **Section A — deepened this match (A1 instinct, A2 phase-split, A3 report).**
  - **A1 — C4 'Instinct'** ([attacker/instinct.py](attacker/instinct.py), 5 tests): counterfactual
    off-ball run optimisation — move an off-ball attacker, re-score team xT via the GAT, take the
    value-optimal run, compare to the actual run (matched in the +1.5 s frame). Engine works end-to-end
    on the real GAT.
    - *v1 finding:* the GAT's graph xT is *ball-dominated* — relocating one off-ball player barely moves
      it (value-gain ~0.001), so xT-as-value is a weak signal.
    - *v2 fix (done):* swapped in the **receiver head** as the player-specific run value via
      `sop_bridge.receiver_probs` ("run to where you'd be the best passing option"); the engine takes a
      generic `player_value(frame, idx)` callable and `run_match_instinct` drives it end-to-end. Value-gain
      jumped **0.001 → ~0.025 (25× more sensitive)** — a real per-team **off-ball receiving-potential**
      metric. Decision-quality (actual-vs-optimal-run cosine) stays near zero: good off-ball movement isn't
      only about *immediate* receiving (runs in behind reduce it), so the value-gain is the useful output.
  - **A2 — possession phase-split** ([fingerprint/phase_metrics.py](fingerprint/phase_metrics.py),
    2 tests): carry-forward possession from sparse ball-carrier frames → per-team in/out-of-possession
    metrics. Reveals real tactics: **Team 0 builds high (46 m) but defends deep (line 36 m = mid/low
    block); Team 1 builds high *and* presses high (line 42 m)** — the FIFA in-possession-vs-block
    contrast. `outputs/match_phases.parquet`.
  - **A3 — descriptive report** ([report/build_report.py](report/build_report.py), 1 test): no-dep HTML,
    EFI-style mirrored two-team comparison grouped by the 4 facets + the phase split + honest-scope
    footer → `results/reports/match_report.html`.
- **GAT re-run on corrected coords + per-team 4-facet fingerprint.** Re-ran the trained GAT
  ([generator/sop_bridge.py](generator/sop_bridge.py)) on the calibration-corrected positions — valid
  relational reads (P(success) 0.361, Dynamic-xT 0.0168, P(def recovers) 0.418; the old broken-calib
  0.506/0.0312/0.461 is retired). Then built [fingerprint/facet_metrics.py](fingerprint/facet_metrics.py)
  (2 tests): per-team metrics organised into **ATTACKING / DEFENDING / PASSING / GOALKEEPING** (+physical)
  on the colour-anchored full match, with **new** goalkeeping (sweeper height, lateral range) and a
  **passing-connectivity** proxy (nearest-team-mate distance). Direction resolved per chunk. Result
  (`outputs/match_facets.parquet`): Team 1 = higher line/wider/more wing-oriented attack + **sweeper
  keeper (13 m off line vs 8 m)**; Team 0 = deeper/compact + tighter passing network (8.6 vs 9.6 m).
  **Honest scope:** orientation-normalised *overall* metrics, not the FIFA in/out-of-possession phase
  split; passing/GK are positional proxies (pass completion, line breaks, GK distribution need ball
  tracking). The GAT's attacking-threat / defensive-recovery reads are the relational layer to attribute
  **per team via possession** next.
- **Per-team GAT attribution done** ([generator/sop_bridge.py](generator/sop_bridge.py)
  `per_team_relational`): for each ball-carrier (`is_actor`) frame the carrier's team is attacking
  (its `success`/`dynamic_xt`), the other team is defending (`p_defstop`); aggregated per anchored team
  over the match (5493 actor frames; team0 attacked 1735, team1 3107). Result: **team 0 = the clinical
  attacker** (xT 0.038, success 0.56) but less possession; **team 1 = more possession + better recovery**
  (def 0.48 vs 0.43) yet lower threat (xT 0.028, success 0.44). Merged into
  [fingerprint/facet_metrics.py](fingerprint/facet_metrics.py) (`att_threat_xt`, `att_success_p`,
  `def_recovery_p`) → `outputs/match_facets.parquet` now carries positional + relational per facet.
- **Fix 2 — ball-tracking foundation + WASB wired (but pretrained model does NOT transfer).**
  [generator/ball.py](generator/ball.py) (3 tests): the **deterministic, detector-agnostic** layer —
  `link_ball` (greedy nearest-to-prediction + physical speed gate + short-gap interpolation) and
  `assign_possession` (nearest-player, debounced), pure + tested, validated on chunk_000's sparse ball
  (11% raw → 496-frame linked trajectory → 213 possession frames).
  - **WASB fully sourced + wired**: cloned `nttcom/WASB-SBDT` to `~/WASB-SBDT`, downloaded soccer weights
    (`wasb_soccer_best.pth.tar`), reimplemented a no-Hydra HRNet forward in `WASBBallDetector` (loads
    cleanly — 428 keys, 0 missing; runs 18.6 fps GPU @512, native-res option added).
  - **Finding: the pretrained soccer model does NOT transfer to this 1024×576 broadcast.** @512: weak
    heatmaps (~0.18), ~300 px from the ball. @native: stronger heatmaps (~0.54) but ~350 px off, and over
    a contiguous window only 4% fire and those are **stationary** (locks onto a fixed background feature,
    not the moving ball). Weights verified loaded → genuine **domain gap**, not a bug.
  - **Tried the zoo's other models (TrackNetV2, DeepBall-Large soccer weights downloaded).** TrackNetV2
    is better than WASB but **still only ~15% recall on *verified* live play** (an early "99% recall"
    result was the model tracking an animated **Man City pre-match graphic**, not the ball — a
    methodological trap; the YOLO ball is also contaminated by false positives on graphics). Zoomed
    visual: TrackNetV2 finds the ball when it's clearly at a player's feet, but **fires on the persistent
    scoreboard overlay** and misses most frames. Two root causes: low-res 1024×576 footage (domain gap)
    **and persistent broadcast graphics** the detectors mistake for the ball.
  - **Honest conclusion: no off-the-shelf model works well on this footage.** The ball is the project's
    real CV bottleneck. Cheap next step: **mask the scoreboard/graphic regions** (cuts false positives);
    real fix: **fine-tune TrackNetV2 (best base) on manual ball annotations**. The deterministic
    link/possession layer + seam are ready for whatever detector lands.
- **Fix 1 — match-consistent team identity (full-match fingerprint unlocked).**
  [generator/team_anchor.py](generator/team_anchor.py) anchors teams by **jersey colour clustered
  globally across all chunks** (estimates a torso box from the foot point — no boxes in the parquet —
  reuses `teams.jersey_color`, then one KMeans(2) over every track's colour). This is invariant across
  the half-time end swap, unlike per-chunk KMeans + defended-goal. [fingerprint/team_style.py](fingerprint/team_style.py)
  `match_style_vector` aggregates with **per-chunk** attack directions (a global team attacks opposite
  goals each half). Validation: the full match now splits **25,901 vs 25,560 frames** (vs the lopsided
  10k/7k first-half-only) — the anchoring tracks the two physical teams across both halves. Full-match
  `z_T` → `outputs/match_anchored_dense.parquet`; Team 1 higher/wider/wing-oriented, Team 0 central.
  2 tests.
- **Team-style fingerprint `z_T` + FIFA-EFI alignment.**
  [fingerprint/team_style.py](fingerprint/team_style.py) (3 tests): one interpretable vector per team —
  shape (width/length/compactness/surface area/5 lanes) + verticality (build-up/def-line height,
  attacking-third share) + rough physical (speed-zone share, sprint share, top speed). Adopted the
  **FIFA EFI Post-Match Summary as the oracle** and wrote [docs/EFI_ALIGNMENT.md](docs/EFI_ALIGNMENT.md):
  capability map (we own the spatial/structural families; ball/event families are gated on ball
  tracking), a **C6 validation opportunity** (our chunk_000 widths 33–36 m / line heights 40–53 m land
  in the same range as FIFA's mid-block 38 m / build-up 44 m), the **predictive report** vision, and the
  **grounded game-plan synthesizer** reframe (LLM narrates computed metrics — "how A should play B").
  Multi-chunk combining fixed: `globalize_chunk_ids` (frame/track ids reset per chunk) +
  `align_teams_by_defended_goal` (KMeans labels teams arbitrarily per chunk; align within a half).
  3-chunk aligned `z_T` produced (`outputs/match_zT.parquet`).
- **Match batch — all 11 chunks now done** ([tools/batch_match.py](tools/batch_match.py)). First run
  OOM'd at chunk 4 (`extract_positions` builds a fresh detector per chunk → VRAM accumulates); **fixed**
  with `gc.collect()` + `torch.cuda.empty_cache()` between chunks, and a `--skip-existing` resume
  finished 3–10. `match_dense.parquet` = **11 chunks, 25,648 accepted tactical frames**.
  - **Data quality:** median calib **~0.25 m on every chunk**, **0% player points off-pitch** — the
    accepted positions are clean. A few frames per chunk (esp. 1/4/9) had catastrophic calibration
    (mean calib 64–948 m) that the **keypoint gate let through but the player-distribution gate
    rejected** (positions NaN'd) — vindicates that backstop; the keypoint gate alone is insufficient.
  - **Real blocker for a *full-match* fingerprint:** teams **swap ends at half-time**, and per-chunk
    KMeans + `align_teams_by_defended_goal` only link teams **within a half**. Naively aggregating all
    11 chunks blends both teams. Next fix: **jersey-colour team anchoring across chunks/halves**.
  - **Clean first-half `z_T`** (chunks 0–3, ~40 min, `outputs/match_h1_zT.parquet`): same story as one
    chunk with 4× data — Team 1 higher (build-up 52 vs 48 m, def-line 46 vs 42 m) and more forward
    (att-third 0.25 vs 0.18); Team 0 more central/territorial (surface 506 vs 431 m²).
  - **C3 over 4 chunks (`outputs/match_h1_dense.parquet`):** run head **RMSE 4.64 / hit@3m 0.572**
    (robust, beats old-360 6.33/0.244); **receiver head now 161 pass events** — learned **0.31/0.72
    top1/3 beats nearest-teammate (0.25/0.59) but trails old-360 (0.41/0.78)**: ball-recall-limited,
    the WASB ball-tracking upgrade is the unlock.
- **C3 attacker rebuild — run head beats the old 360 baseline (the headline novelty)**
  ([attacker/tracks.py](attacker/tracks.py), [attacker/labels.py](attacker/labels.py),
  [attacker/heads.py](attacker/heads.py), [eval/attacker_eval.py](eval/attacker_eval.py); 6 tests).
  Per-player run + receiver heads trained on the dense, ID-persistent **video** tracks (not sparse 360):
  - *tracks* — Savitzky-Golay smoothed position + velocity, velocity nulled across track gaps and
    **clipped to 10 m/s** (kills jitter/ID-switch teleports).
  - *labels* — run displacement at t+1.5 s read straight off the persistent track (fixes the old 360
    nearest-neighbour label noise); receiver = consecutive same-team ball-carrier transitions.
  - *heads* — attack-direction-normalised (every team → attacks +x, via the keeper-derived directions),
    Ridge run head + Gaussian residual vs **no-move / constant-velocity** baselines; logistic receiver
    head vs **nearest-team-mate**.
  - **Result on chunk_000** (drop 3.8% >10 m/s ID-switch labels): learned run head **RMSE 5.11 m /
    hit@3m 0.566**, beating old-360 (**6.33 m / 0.244**) on *both* — dense motion data more than doubles
    hit@3m. const-vel blow-up (16→6.7 m) fixed by the velocity cap. **Receiver head: only 12 pass events
    on one chunk → not yet meaningful** (ball-recall-limited; the full-match batch + native-fps ball
    tracking are the unlock).
- **Run-head sharpening — recovered run magnitude + a calibrated distribution.** Eyeballing showed the
  Ridge head's arrows were too short: it is the **RMSE-optimal conditional mean**, which shrinks
  magnitude when run direction is uncertain (median predicted 1.16 m vs true 2.81 m). Added two things
  ([attacker/heads.py](attacker/heads.py)): a **speed × heading** head (scalar speed doesn't cancel) that
  restores realistic length (median 3.82 m, dir-cos 0.75) at a small RMSE cost (5.76 vs 5.11), and a
  **Gaussian distribution** on the mean head (sigma ≈ 3.2–4.0 m, 2σ coverage 0.87 vs 0.91 ideal — roughly
  calibrated). The honest takeaway: you cannot minimise point-RMSE *and* match run length for stochastic
  runs — the mean head is point-optimal (the headline-vs-360 number), speed×heading / the distribution
  are for realism + downstream uncertainty. **Surprise:** plain `const_vel` is the best *realistic*
  predictor (dir-cos 0.80, median 2.75 m ≈ true) — at a 1.5 s horizon current velocity carries the run.
  The inspection gallery's predicted arrow now uses speed×heading so it shows realistic length.
- **Whole-match batch running** ([tools/batch_match.py](tools/batch_match.py)): all 11 MUN–MCI chunks →
  per-chunk dense+tactical parquets (partial-safe, `--skip-existing` resumable, one calibrator reused)
  + concatenated `match_dense/tactical.parquet`. Feeds a whole-match fingerprint and many more receiver
  events. Background (`outputs/batch_match.log`); user chose full per-frame-quality run.
  - **Finding — temporal calib's ~8× speedup does NOT hold on a panning broadcast.** chunk_000 took
    **31 min** (3891 full calibrations, only 2184 reused = 64% full): at sample-every 5 the camera pans
    >2 px between samples, so the drift gate fires almost every frame. **GPU idle (0%), CPU-bound on
    PnL refinement.** The earlier "12 full / 100" was a near-static window. Levers for next time: relax
    `--calib-drift` (~6 px) and/or coarsen `--sample-every`. Output quality is good: 0.35 m calib,
    14.4 players/frame, 1799 accepted tactical frames/chunk.
- **Deterministic metric engine built + verified (the review's Section 3, v1)**
  ([fingerprint/structural_metrics.py](fingerprint/structural_metrics.py), 8 tests passing). Model-free,
  auditable team metrics from player positions only.
  - *Direction-agnostic shape:* width, length, compactness, convex-hull surface area, five-lane
    occupation.
  - *Direction-dependent (added this session):* buildup_height, def_line_height, attacking-third share —
    normalised toward each team's attacking goal. **Attacking direction is resolved from the
    goalkeeper's defended goal** (`resolve_attack_directions`, median keeper-x), which reads the footage
    so it auto-adapts to whichever match half a clip is from — *not* from outfield mean-x (the review's
    warning). Overridable via `attack_dirs=` when match-half metadata is known.
  - Produced the first **real MUN–MCI fingerprint** on chunk_000 (1994 tactical frames). team 0 = high
    line (def_line 46.7 m, buildup 53.5 m), central (0.33 centre lane), vertically stretched (25.0 m),
    24% in attacking third. team 1 = deeper (def_line 39.5 m, buildup 46.3 m), wider (36.3 m),
    wing-oriented (0.27 wing share), larger footprint (511 m²), 15% forward. Directions resolved
    correctly opposite (team 0 → x=105, team 1 → x=0).
  - Caveats: `avg_players ≈ 7.5/frame` (broadcast frames a partial team) → shape metrics are comparative
    lower bounds; and the ball-following camera biases **absolute** heights toward the ball side, so
    direction metrics are reliable team-vs-team contrasts, not survey-grade absolutes. Multi-camera /
    wide-shot weighting is the eventual fix.
- Created the standalone repo (separate from `football-state-of-play` and `cv-football`).
- Wrote the final combined plan: [docs/PLAN.md](docs/PLAN.md) — one pipeline, generator-first,
  novelty C3 + C5, end goal = auto-generated FIFA-style team report.
- Built the foundation the whole pipeline plugs into: the **unified, substrate-aware freeze-frame
  contract** ([generator/contract.py](generator/contract.py)) + tests — pure, dependency-light,
  down-projects to the trained GAT's input format.
- Stubbed every package (`ingest`, `generator`, `attacker`, `fingerprint`, `synthesizer`, `report`,
  `eval`) with docstrings + typed signatures + TODOs.
- Registered the data sources in [ingest/sources.py](ingest/sources.py).
- **Wired the loop end-to-end** ([generator/sop_bridge.py](generator/sop_bridge.py)): positions
  parquet → `frames_from_positions` → trained GAT → clip-level relational readout (P(success),
  Dynamic-xT, P(def recovers)). Verified on the real `vid1_full.parquet` (25-frame run).
- **Unified the interpreter (the two-stack split is gone)**: installed `torch_geometric` into py3.14,
  which already had the CV stack + torch. The whole pipeline now runs in one process (see Decisions).
  `sop_bridge` keeps an auto-re-exec fallback (tested) but no longer needs it here.
- **Validated `extract.py` end-to-end on real gameplay** (py3.14, `chunk_000.mp4`): detection +
  ByteTrack ID persistence + ball detection all work; added `--start-frame`/`--max-frames` after
  finding the clip's first ~3 s is broadcast pre-roll (0 players).
- **Wired PnLCalib** — [generator/calibrate.py](generator/calibrate.py) `PnLCalibCalibrator`: loads
  PnLCalib's two HRNet models (SV_kp/SV_lines), detects pitch keypoints/lines. Cloned to `~/PnLCalib`
  (override `$FOOTBALL_PNLCALIB_PATH`); weights under `<repo>/weights`; deps (`shapely`, `lsq-ellipse`,
  `munkres`) installed + declared.
- **CRITICAL CALIBRATION BUG — two stages, both now fixed and verified.**
  - *Stage 1 (my-own-homography):* the first integration used PnLCalib only as a keypoint detector,
    then fit *my own planar homography* (`cv2.findHomography`) on the ground keypoints. Those cluster
    in a thin image band (~192 px), under-constraining a planar homography → players projected to
    nonsense. **Fixed** by switching to PnLCalib's full 3D **camera calibration** (`heuristic_voting`
    → points+lines+PnL refinement) and deriving the image→pitch homography from its 3×4 projection
    `P` (`ground_homography_from_cam_params`).
  - *Stage 2 (centred-vs-uncentred — the one that still looked "inverted"):* PnLCalib's camera params
    are in a frame **centred on the centre spot** (its `keypoint_world_coords_2D` are re-centred by
    `[x−52.5, y−34]`, `utils_calib.py:41`). I treated `P`'s output as **uncentred** `[0,105]×[0,68]`,
    a constant **(52.5, 34) m offset**: the centre spot landed at a corner, ~half the players were
    dropped as "off-pitch", and a far-goal keeper plotted at "halfway". This hid because the gate's
    own `obj` points were *also* centred, so the error read ~0.27 m while the **output** was offset by
    ~62 m. **Fix:** `ground_homography_from_cam_params` now inverts to centred metres then shifts the
    origin to the corner (×`to_uncentred`), and `calibrate_frame`'s gate compares against uncentred
    `obj`. **Verified** (`tools/probe_convention.py`): production homography error vs true uncentred
    coords **62.55 m → 0.27 m**; the right-goal keypoint maps to **(105, 43)**, not (52, 9); pitch
    geometry projects correctly (halfway line at the image's left edge, right box centre-right, left
    box off-screen); `corr(image_x, pitch_x)=+0.85` (no inversion). Locked by a synthetic centred-cam
    unit test (`test_calibrate.py`) **and** an opt-in golden-frame integration test
    (`tests/test_golden_frame.py`). User confirmed the camera-model pitch lines are correct.
  - **Regenerated the displaced outputs** (Delivery #1): `seg1_fixed` now **44/47 frames, ~16 players,
    ~41 m span** (right-half view, x 56–97); `seg2_fixed` now **54 frames, 14–18 players** spread round
    the centre circle (x 26–59) — previously the infamous "4 dots in a corner".
- **Player-distribution plausibility gate** ([generator/postprocess.py](generator/postprocess.py)
  `reject_implausible_frames`): a frame is trusted only if ≥8 players land on-pitch spanning ≥25 m on
  an axis; otherwise its pitch coords are NaN'd. Kept as a backstop. NB: seg2's old "drop all 57" was a
  *symptom of the Stage-2 bug* (centred coords crammed players into a corner), not a bad angle — with
  calibration fixed, seg2 now passes. Unit-tested.
- **Pass-C groundwork — roles, keepers, pitch boundary** ([generator/postprocess.py](generator/postprocess.py),
  [generator/extract.py](generator/extract.py)):
  - *Pitch boundary (linesman fix):* `clamp_to_pitch` tolerance cut 5 m → **2 m**; points past it are
    **dropped, not clamped onto the touchline** (no phantom officials-as-players). On seg1 this dropped
    only ~1 row while keeping all real players.
  - *Keeper tags:* `derive_keeper` now (1) trusts a detected `goalkeeper` role, else (2) tags a team's
    extreme player **only if within 16.5 m of its own goal** — killing the "keeper at halfway" false
    tag. Verified: seg1 went from false midfield keepers to **3 tags, all at x≈99.7 (right goal),
    ≤1/frame**.
  - *Role plumbing:* every detector returns a per-box role id carried through ByteTrack; `referee`
    rows get `team=-1` and are excluded from actor/keeper/plausibility. COCO/RF-DETR emit all `player`
    (no role classes), so a football-trained detector slots in unchanged. Unit-tested.
  - **Football-role detector adopted (Pass C).** Benchmarked 4 candidates on 6 held-out frames
    ([tools/benchmark_detectors.py](tools/benchmark_detectors.py)): COCO YOLOv8s, HF `uisikdag` v8,
    HF `soccana` v11, RF-DETR. **Winner: `uisikdag/yolo-v8-football-players-detection`** — the only
    one with a goalkeeper class, cleanly separates officials (others count them as players), and is
    conservative on the ball (soccana/RF-DETR over-detect). Wired as `--detector football`
    ([generator/extract.py](generator/extract.py) `_FootballRoleDetector`, class names mapped by name,
    weights download cached from HF). End-to-end on seg1: **GK at x≈97 (right goal), tagged keeper via
    role; central referee identified at x≈75 and excluded (team=-1); ball in 10/44 frames**; team
    classifier now fits on player crops only (GK/ref kits no longer pollute the 2-team KMeans).
  - **Open refinements:** GK *team* still comes from jersey KMeans (better: assign by defended goal);
    ball recall ~23% of frames (needs native-fps ball tracking); role *precision* not yet measured on
    labelled frames.
- **Tracking — swappable backends, ByteTrack kept as default** ([generator/tracking.py](generator/tracking.py),
  `--tracker {bytetrack,botsort}`). Added Ultralytics **BoT-SORT** (GMC `sparseOptFlow` + appearance
  ReID, tuned in [generator/botsort_tuned.yaml](generator/botsort_tuned.yaml)) via the YOLO detector's
  `track()`. **Benchmarked vs ByteTrack** on a 2 s dense clip and a 50 s sampled window: ByteTrack won
  both (≈22 IDs ≈ true player count, mean track 57 fr, frag 1.69) vs tuned BoT-SORT (28–34 IDs, p50
  18–27 fr, frag 2.32–2.66) — the held tactical camera pans too little for GMC to help, and
  ByteTrack's 3-frame confirmation suppresses the spurious tracks BoT-SORT keeps. **Evidence-based
  call: stay on ByteTrack**; BoT-SORT stays available for heavy-pan sources (re-benchmark per source).
  NB: `supervision.ByteTrack` is deprecated (removed in sv 0.30) — migration needed eventually.
- **Detection validation harness** ([tools/validate_detection.py](tools/validate_detection.py),
  tested box-matching core). Two GT modes: a **multi-detector consensus proxy** (a box ≥2/3 detectors
  agree on = a person) and **manual YOLO labels** (`--labels`). Consensus over 10 held-out frames:
  football detector **person recall 0.979 / precision 0.898** (rfdetr 0.995/0.942, coco 0.936/0.985),
  consensus head-count ≈19.5/frame. Proxy can't judge roles/ball — exported the 10 frames + draft YOLO
  labels + README ([outputs/validation/annotate/](outputs/validation/annotate/)) for correction →
  true role accuracy + ball recall via `--labels`.
- **Temporal calibration (Pass B)** ([generator/temporal_calib.py](generator/temporal_calib.py),
  `--calib-period N --calib-drift PX`). Full PnLCalib only on shot-cut (hist break) + every N frames +
  on camera drift (phase-correlation); **reuses the last accepted pose** between, and **never
  resurrects a rejected frame** (unit-tested policy). On 100 frames, period=25 with the tuned **2 px
  drift gate** ran **12 full calibrations vs 100** (~8× fewer) at the **same frame coverage**, adding
  **1.27 m median** length-axis error (within the ≤2 m budget; width 0.19 m). The 95th-pct tail (~4 m)
  is reuse lagging a panning camera — so **default stays per-frame (`--calib-period 1`)** and temporal
  is the opt-in speed lever for full-match runs (#4), tunable via `--calib-period`/`--calib-drift`.
- **Full-chunk dense tracking + tactical-frame table (Pass D/E, #4)**
  ([generator/tactical_frames.py](generator/tactical_frames.py)). Ran the whole pipeline over **all of
  chunk_000 (10 min @ 10Hz)** with football detector + temporal calib: 6075 frames → 3199 with
  detections → **1803 accepted tactical frames (~3 tactical min), mean 15.1 players/frame, 0.26 m mean
  calib error**, 3341 non-tactical frames correctly rejected. `tactical_frames.build_tactical_table`
  flattens the dense positions into one analysis-ready row/frame (counts, per-team centroid/width/
  depth, ball, ball-carrier, quality) — the metric-engine input; attack *direction* deliberately left
  to match-half metadata, not mean-x. Scales to the full 11-chunk match as a batch (same command).
- **Fixed a smoothing-resurrection bug** ([generator/postprocess.py](generator/postprocess.py)
  `smooth_tracks`): the centered rolling median (`min_periods=1`) was **filling gate-rejected NaN
  positions from neighbours** (found via 20 inf-calib-error rows that still had coordinates) — exactly
  the "smoothing must not resurrect a rejected frame" rule. Now masks NaNs back; unit-tested.
- **Visual verification upgraded** ([tools/visualize.py](tools/visualize.py) `--pitch-lines`): draws
  the reconstructed pitch model on the video panel so each overlay self-verifies (lines must sit on
  the real markings). [tools/diag_calib.py](tools/diag_calib.py) compares calibration methods on a frame.
- **Fixed the team-classifier collapse** — [generator/teams.py](generator/teams.py)
  `JerseyColorTeamClassifier`: grass-masked CIELAB-chrominance per crop → KMeans(2). Deterministic,
  no SigLIP/UMAP. Diagnosis showed the old {304:7} collapse had two causes, both removed: (1) a
  **RGB/BGR crop bug** in extract (predict crops were RGB, the classifier expects BGR — fixed), and
  (2) UMAP's `.transform` destabilising across camera regions (in-sample both classifiers split fine;
  the colour one has no `.transform` step). 5 unit tests.
- **Fixed the wide-shot problem** — [generator/segments.py](generator/segments.py): a `wide_shot_score`
  (players × horizontal spread) + `scan_wide_segments`, plus an `extract --auto-wide` flag that picks
  the widest tactical segment automatically. 4 unit tests.
- **Closed the loop with the STANDARD ≥10 wide-shot gate (no lowering):** `--calibrate` extract on a
  scanned wide segment → **32 frames, 17–20 valid players/frame, both teams ({1: 317, 0: 272}),
  0.22 m mean calibration error** → `sop_bridge` ran the GAT on all 32 frames →
  **P(success) 0.506, Dynamic-xT 0.0312, P(def recovers) 0.461**. The full video→GAT path now
  produces a non-degenerate tactical readout from real footage.
- **GPU enabled (RTX 3050 Laptop, CUDA 12.8).** Replaced the CPU torch with `torch 2.11.0+cu128` +
  `torchvision 0.26.0+cu128` in the unified py3.14 env (cp314 CUDA wheels exist; minor 2.12→2.11
  downgrade, all deps fine). `PnLCalibCalibrator` and YOLO **auto-detect CUDA** (calibrator
  `device=None` → cuda). **PnLCalib dropped from ~13 s/frame (CPU) to ~0.35 s/frame warm (~38× faster)**,
  ~2 GB VRAM, same accuracy (0.247 m). All 50 tests pass on the new torch. End-to-end GPU run
  (`extract --calibrate --auto-wide`, full-clip scan + 50 calibrated frames + bridge): **~66 s wall**
  (was ~8 min for a smaller CPU run); `--auto-wide` auto-picked frame 13050; 47 frames cleared the
  ≥10 gate through the GAT.
- **Built the CV front-end as reproduce-and-adapt of `cv-football`, with the trustworthy work in
  tested, pure modules:**
  - [generator/calibrate.py](generator/calibrate.py): homography estimation (numpy normalised-DLT,
    or `cv2.findHomography`/RANSAC when present) + the **reprojection-error confidence gate**
    ("no wrong frames"). PnLCalib is a lazy seam (weights are an external download). 5 tests.
  - [generator/postprocess.py](generator/postprocess.py): pure track smoothing + ball-carrier
    (actor) and keeper derivation + pitch-bounds clamping — lifted out of the CV script so they run
    and are tested without the CV stack. 6 tests.
  - [generator/extract.py](generator/extract.py): video → positions parquet orchestration (YOLO +
    ByteTrack + team classifier + calibrate + post-process), **heavy CV imported lazily** (module
    imports without `[cv]`), COCO fallback when Roboflow `inference` is absent. Pure seams
    (transform+gate, row building, schema) are tested. 8 tests.
- **Test suite: 50 tests, all passing in the unified py3.14 env** (the GAT-bridge test runs here now,
  not skipped). Still skips cleanly on any interpreter lacking torch-geometric. PnLCalib config is
  unit-tested; the heavy HRNet run is an opt-in integration check.

## Decisions made

- **`football-state-of-play` dependency = runtime path-inject bridge** (not pip-install / not
  vendored). [generator/sop_bridge.py](generator/sop_bridge.py) puts the sibling repo root on
  `sys.path` at call time and imports only `data.graphs.build_data` + `models.gnn.GAT`; we run our
  own thin predict loop. Avoids the top-level `eval` package collision an editable install would
  cause, and keeps a single source of truth (no model-code drift). Override the sibling location
  with `$FOOTBALL_SOP_PATH`.
- **One unified interpreter (FIXED).** The pipeline previously needed two interpreters (CV stack vs
  torch-geometric). Resolved by installing `torch_geometric` (2.8.0, pure-python wheel) into the
  **default py3.14**, which already had the full CV stack + torch. That interpreter now imports
  `cv2`, `ultralytics`, `supervision`, `sports`, `torch`, `torch_geometric` together, builds the GAT,
  loads the checkpoint, and runs `extract.py` AND `sop_bridge.py` in one process. Verified: GAT runs
  directly in py3.14 (no re-exec); all 41 tests pass with the bridge test running (not skipped).
  - The GAT checkpoint was saved on CUDA; `sop_bridge.load_model` now loads with
    `map_location="cpu"` so a CPU-only torch works.
  - `sop_bridge`'s auto re-exec into the SoP `.venv` is kept as a portability fallback (no-op now),
    overridable with `$FOOTBALL_SOP_PYTHON`.
  - Reproducible from clean: `pip install -e ".[cv,dev]"` (torch + torch-geometric are core deps;
    `sports` added to the `[cv]` extra as a git dependency).
- **Calibration: RESOLVED via PnLCalib.** Cloned `mguti97/PnLCalib` to `~/PnLCalib` with SV_kp/SV_lines
  weights; `PnLCalibCalibrator` runs its HRNet detectors and feeds correspondences to our gate.
  Verified sub-metre reprojection error on real frames. (Roboflow `inference` not needed.)

## Research-driven upgrades (this session)

- **RF-DETR detector option** ([generator/extract.py](generator/extract.py) `_RFDetrDetector`,
  `--detector rfdetr`). RF-DETR (Roboflow, DINOv2 backbone, NMS-free; SOTA on COCO, ICLR 2026).
  A/B vs YOLOv8s on 30 real frames ([tools/ab_detectors.py](tools/ab_detectors.py)):
  **ball-detection 0.37 → 1.00**, players/frame 19.3 → 20.2, mean conf 0.58 → 0.66, 0.056 → 0.085
  s/frame on GPU. The ball win directly fixes actor-tagging. Currently COCO-pretrained; the
  football-trained RF-DETR (player/GK/ref/ball, ~83% mAP) is the next upgrade.
- **Visual verification tool** ([tools/visualize.py](tools/visualize.py)): video frame + top-down
  freeze frame side by side, plus track-trajectory plots, for manual inspection.
- **Known weakness found via 2nd-clip viz:** the calibration confidence gate scores *keypoint*
  reprojection error only. A bad camera angle (seg2 / frame 21720) fit keypoints fine yet projected
  most players into one pitch corner (5 valid tracks vs seg1's 19) — and **passed the gate**. Need a
  *player-distribution* sanity check (on-pitch count, spread, both-box coverage), not just keypoint error.

## Tracking upgrade (researched, not yet implemented)

- SoccerNet-GSR SOTA uses **BoT-SORT + global motion compensation (GMC)** + re-ID, not plain
  ByteTrack. GMC matters for panning broadcast cameras. Candidate swap in `detect_track`/extract.

## Open questions for Sid / supervisor

- Confirm footage list + how many FIFA EFI PDFs we can collect (WC22 has them per match).
- Is GS-HOTA an acceptable generator-accuracy anchor?

## Next steps (in priority order)

1. **Make RF-DETR default + load football-trained weights** (player/GK/ref/ball classes) for a
   single-pass role+ball detector; re-A/B.
2. **Strengthen the calibration gate** with a player-distribution sanity check (fixes the seg2 failure
   mode that slipped through).
3. **Attacker rebuild (C3)** — `attacker/{tracks,labels,heads}`: per-player run + receiver heads on
   the dense, ID-persistent tracks the generator now produces, vs the old 360 baselines.
4. **Tracking: BoT-SORT + GMC + re-ID** for ID persistence under camera motion.
5. **Team fingerprints + synthesizer (C5)**; stand up `sn-gamestate` + GS-HOTA; `ingest/fifa_efi.py`.

## Blockers

- None. (GPU now in use — PnLCalib ~0.35 s/frame; the earlier CPU-speed blocker is resolved.)

## Environment (reproduce from clean)

- Unified interpreter: **py3.14**, `torch 2.11.0+cu128` (install:
  `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`),
  plus `pip install -e ".[cv,dev]"`. PnLCalib: clone `mguti97/PnLCalib` to `~/PnLCalib` + SV_kp/SV_lines
  weights under `<repo>/weights`. GPU: RTX 3050 Laptop, CUDA 12.8, ~2 GB VRAM used.
  Detection/tracking are also CPU-bound here.
