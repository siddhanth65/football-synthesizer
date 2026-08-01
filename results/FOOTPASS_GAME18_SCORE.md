# FOOTPASS game_18 — the from-pixels attribution number, end to end

Date: 2026-07-30. The completion of the run planned and logged in `results/FOOTPASS_E2E_PLAN.md`
(Appendix B). Everything below starts at pixels: detect -> track -> team -> homography -> ball ->
per-crop OCR -> GTA connector -> identity solve -> carrier -> `(team, jersey)` at each of
**1,879 labelled VAL events** of game_18. The official SN-PCBAS baseline (Macro-F1 46.41) is handed
ground-truth game state; this is not comparable to it and is not claimed to be.

**Headline, both arms, on the primary metric (attribution-given-event, coverage over ALL events):**

| arm | coverage@precision>=0.85 | coverage@0.60 | full coverage | at precision |
|---|---|---|---|---|
| **Stage-2 solver** (frozen config, per-crop evidence, MILP posterior as the dial) | **0.0005** | **0.1575** | 0.4428 | 0.3173 |
| **Stage-1 greedy** (connector + unanimous propagation on per-crop reads) | **0.0000** | **0.1293** | 0.1293 | 0.7119 |

**Neither arm reaches a 0.85 precision floor at any usable coverage.** The solver's dial gets it to
coverage 0.1575 at the 0.60 floor; the greedy arm has no dial at all (every confidence is 1.0 by
construction) and is a single point at (0.1293, 0.7119).

Against the ceilings: game_18's actor-on-screen rate is **0.7584** (reproduced exactly, 1,425 /
1,879) and the carrier gate reaches **0.6562**. So 0.1575 is 24% of what the gate delivers and 21%
of the on-screen ceiling.

---

## 1. The two frontiers

### 1.1 Solver arm — a real dial with a knee just below the bar

Selected points (full 60-point curve in `results/footpass/val_attribution_game_18_solver.json`):

| threshold | coverage | precision |
|---|---|---|
| 1.000000 | 0.0005 | 1.0000 |
| 0.996449 | 0.0154 | 0.7586 |
| 0.932394 | 0.0303 | 0.7895 |
| **0.923805** | **0.0378** | **0.8169** |
| 0.918214 | 0.0452 | 0.8118 |
| 0.393417 | 0.0527 | 0.7879 |
| 0.337373 | 0.0750 | 0.7872 |
| **0.249443** | **0.0905** | **0.8000** |
| 0.024159 | 0.0979 | 0.7717 |
| 0.019556 | 0.1128 | 0.7358 |
| 0.018839 | 0.1277 | 0.7000 |
| **0.016700** | **0.1575** | **0.6250** |
| 0.015909 | 0.1879 | 0.5297 |
| 0.013535 | 0.2624 | 0.4422 |
| 0.010834 | 0.3901 | 0.3465 |
| 1.3e-10 | 0.4428 | 0.3173 |

The curve tops out at **precision 0.8169 at coverage 0.0378** (71 events) and **0.8000 at 0.0905**
(170 events). It never crosses 0.85 except at the single highest-confidence event, which is why
`coverage@0.85` reads 0.0005 (1 of 1,879) rather than 0. The bar is missed by 0.03-0.05 of
precision, not by an order of magnitude.

### 1.2 Greedy arm — one point, and its "frontier" is an artifact

`coverage_at` returns 0.1293 at the 0.60 floor, which is simply the arm's own operating point. Its
60 tabulated frontier rows all carry `threshold = 1.0`: with every confidence tied, `frontier()`
walks the answered events in table order, so the intermediate rows are a **tie-ordering artifact and
carry no information**. Reported as one point, as `RETEST_ABSTENTION.md` said it must be:

| | coverage | precision |
|---|---|---|
| overall | 0.1293 (243 / 1,879) | 0.7119 (173 / 243) |
| on-screen | 0.1488 (212 / 1,425) | 0.7406 |
| off-screen | 0.0683 (31 / 454) | 0.5161 |

## 2. Split by actor visibility

| split | events | arm | answered | correct | full cov | precision | cov@0.85 | cov@0.60 |
|---|---|---|---|---|---|---|---|---|
| overall | 1,879 | solver | 832 | 264 | 0.4428 | 0.3173 | 0.0005 | 0.1575 |
| on-screen | 1,425 | solver | 683 | 231 | 0.4793 | 0.3382 | 0.0007 | 0.1951 |
| off-screen | 454 | solver | 149 | 33 | 0.3282 | 0.2215 | 0.0022 | 0.0066 |
| overall | 1,879 | greedy | 243 | 173 | 0.1293 | 0.7119 | 0.0000 | 0.1293 |
| on-screen | 1,425 | greedy | 212 | 157 | 0.1488 | 0.7406 | 0.0000 | 0.1488 |
| off-screen | 454 | greedy | 31 | 16 | 0.0683 | 0.5161 | 0.0000 | 0.0220 |

Best on-screen point at a relaxed floor: solver **coverage 0.1137 at precision 0.8148**.

**The off-screen split is a surprise and is reported as one.** 33 of 149 off-screen answers are
*correct* (0.2215) — the actor was flagged as carrying no broadcast ROI at the labelled frame, yet
the pipeline named them right. Against a modal-guess control (always answer the single most frequent
true identity in that slice) of **0.0940**, this is 2.4x chance, so it is not noise. Two causes,
neither of which we can separate here: (a) `roi = NaN` is FOOTPASS's own annotation of visibility and
is not a perfect "outside the frame" signal at one exact frame; (b) the predictor evaluates at the
nearest **sampled** frame, up to 1 frame away, at which the actor can be back in shot.
**The 0.7584 ceiling is therefore soft in practice, not hard** — a correction to
`FOOTPASS_E2E_PLAN.md` section 5's framing, which treated it as absolute.

## 3. The factorisation — where the coverage actually goes

Both arms share every stage up to naming, so the gate columns are identical.

| stage | count | rate over 1,879 events |
|---|---|---|
| actor on screen (annotation) | 1,425 | 0.7584 |
| ball AND players present at the nearest sampled frame | 1,619 | **0.8616** |
| carrier found within 3.0 m (the gate) | 1,233 | **0.6562** |
| carrier track carries a name — **solver** | 832 | **0.4428** |
| carrier track carries a name — **greedy** | 243 | **0.1293** |

| conditional | solver | greedy |
|---|---|---|
| team correct given answered | 649 / 832 = **0.7800** | 207 / 243 = **0.8519** |
| shirt correct given answered | 266 / 832 = **0.3197** | 173 / 243 = **0.7119** |
| both correct (end-to-end over all 1,879 events) | 264 = **0.1405** | 173 = **0.0921** |

Median carrier distance **1.25 m**, median tracked players in frame **13**.

**The carrier gate is at its structural ceiling, and the loss is entirely in naming.** The gate's
ceiling is (actor on screen) x (actor covered by one of our boxes at the event frame) =
0.7584 x 0.9046 = **0.686**; measured 0.6562, i.e. **95.7% of the ceiling**. Detection, tracking,
calibration, ball and the carrier rule together give up 4.3% of what was available. Everything below
0.6562 is the identity channel.

The two arms make opposite trades from the same 1,233 gated events: the solver names 67.5% of them at
0.32 shirt precision, the greedy rule names 19.7% at 0.71. The solver ends with 1.53x as many correct
answers (264 vs 173) from 3.4x as many attempts.

## 4. Read density `d` on FOOTPASS — the negative that explains the rest

Same definition as `OCR_REALMATCH.md` section 2 (fraction of player/GK tracklets carrying at least
one read under the frozen 0.85-floor rule):

| corpus | d | tracks read / tracks | crops |
|---|---|---|---|
| manutd_liverpool | 0.1150 | 960 / 8,345 | 94,244 |
| manutd_tottenham | 0.1053 | 941 / 8,934 | 95,284 |
| manutd_brighton | 0.1263 | 1,174 / 9,294 | 117,041 |
| **footpass_game_18** | **0.0304** | **352 / 11,575** | **166,930** |

**d = 0.0304, 3.5-4.2x BELOW the 0.105-0.126 real-match band** — and it is not a crop-budget
problem: game_18 got **43% more crops than manutd_brighton and 77% more than manutd_liverpool**, and
produced **352 reads against their 1,174 and 960** (70% and 63% fewer). The cause is at the crop
level:

| | legibility >= 0.5 | p_number >= 0.99 |
|---|---|---|
| manutd_liverpool | 20,925 / 94,244 = **0.2220** | 12,726 / 94,244 = **0.1350** |
| **footpass_game_18** | 16,040 / 166,930 = **0.0961** | 5,913 / 166,930 = **0.0354** |

Legibility is 2.3x worse and confident digit reads 3.8x worse per crop. Combined with the frozen
rule's `min_votes = 5`, only 352 of 11,575 tracklets clear the bar. `FOOTPASS_E2E_PLAN.md` section 8
risk 4 asked whether GSR-selected OCR precision transfers to real broadcast; `OCR_REALMATCH.md`
answered "yes on our own footage". **On Serie A broadcast it does not transfer as *density*** — the
rule holds its shape and simply fires 4x less often.

Two consequences visible downstream: **0 of 163 goalkeeper-role tracklets carry a read** (as always —
a keeper's number is never legible), and of the 352 reads only **282** land on a shirt that exists on
the roster of the team the track was assigned, which is the greedy arm's entire direct evidence
before propagation (282 -> 1,537 named tracks, 1,255 propagated, 2 groups disagreed; fragments
9,833 -> 4,736).

Against `EVIDENCE_DENSITY_LAW.md`'s d* = 0.347 for coverage 0.50 at precision 0.85, game_18 runs at
**8.8% of the required density**. The law predicted this outcome; this is the first time it has been
measured end to end from pixels.

## 5. Tracked-recall sanity vs the probe's prediction

Appendix A.5 predicted **0.9882** (ByteTrack stride 2 + knobs) / **0.9912** (BoT-SORT stride 2)
tracked recall on game_18, from 18 windows on cached detections. Measured on the **real full-match
extraction**, 3,000 sampled frames drawn uniformly across both halves, 35,688 GT visible player
boxes:

| criterion | all rows | tracked rows (track_id >= 0) |
|---|---|---|
| IoU >= 0.3 (probe's criterion) | 0.8860 | **0.8855** |
| GT box contains our foot point | 0.5995 | 0.5991 |

and at the 1,425 on-screen event frames specifically, the **actor's own** box:

| criterion | actor covered |
|---|---|
| IoU >= 0.3 | **0.9046** |
| GT box contains our foot point | 0.5726 |

**Read this as a floor, not a refutation of the probe.** The probe matched GT against the detector's
own boxes; the aligned table persists only a foot point, so these numbers go through
`generator.team_anchor.estimate_player_box`, a coarse depth-scaled reconstruction — the 0.60
point-in-box figure is mostly a statement about that estimator's vertical placement, not about
tracking. The 0.8855 IoU figure is 0.10 below the probe's 0.988; part of that gap is the estimator
and part is that the full match contains close-ups and replays that the probe's 18 tactical windows
did not.

**Two clean sub-results.** Untracked rows are negligible — 13.535 of 13.550 rows per frame carry a
track id — so BoT-SORT at stride 2 gives an id to essentially every detection, and the
tracker-association failure that A.1 measured at stride 5 (0.4876) is gone. And 0.9046 actor coverage
at the event frames is exactly the number the carrier-gate ceiling arithmetic in section 3 needs.

Ball, for completeness (the only number that counts, post-`link_ball`): **33,741 ball rows over
59,593 sampled frames = 0.566**, reproducing Appendix B.4's 57.3%.

## 6. Wall clock, measured, against the plan's estimates

From process start times and artifact mtimes. GPU stages ran one at a time; nothing else was on the
GPU.

| stage | window | wall | plan estimate for this game | delta |
|---|---|---|---|---|
| extract (GPU, stride 2, BoT-SORT, calib_period 62) | 07-29 15:48:40 - 21:58:41 | **6 h 10 m** | ~9.2 h (A.6) | **-33%** |
| align (CPU) | -> 07-30 11:39:21 | not separately timed | 10-20 min | — |
| ball (GPU, TrackNet v6 + link_ball) | 11:39:21 - 13:07:42 | **1 h 28 m** | ~1.7 h (A.6) | -14% |
| GTA per-detection embeddings (GPU, stride 5) | ~13:08 - 13:49:27 | **~41 m** | ~0.42 h (A.6) | on estimate |
| per-crop OCR (GPU, 20 crops/track) | 13:50:17 - 15:57:00 | **2 h 07 m** | ~1.7 h (A.6) | +25% |
| **GPU subtotal** | | **10 h 26 m** | ~11.9 h | **-12%** |
| teammap (CPU) | 16:16:52 - 16:17:06 | 14 s | — | |
| identity solve (CPU, 11 chunks, HiGHS) | 16:17:15 - 16:18:50 | **95 s** | 2-4 h for 35 chunks | **~40x faster** |
| predict, solver arm (CPU) | 16:19:49 - 16:20:12 | 23 s | minutes | |
| predict, greedy arm (CPU) | 16:20:21 - 16:21:15 | 54 s | minutes | |
| score, both arms (CPU) | | ~15 s | minutes | |

Extract per chunk: mean **2,018 s** over 11 chunks (1,897-2,569 s; h2 ran ~20% slower than h1).

**The identity solve's 95 s is a symptom, not a win.** The MILP is trivial here because there are
only 352 read tracklets to build galleries from, so most tracklets have no appearance evidence and
almost no candidates survive the 6-candidate prune. The plan's 2-4 h estimate was priced against
manutd-density evidence.

## 7. The team bit, resolved twice and agreeing

`FOOTPASS_E2E_PLAN.md` section 4 required this bit to be measured twice, independently:

* **before the solve** — `footpass_prep --stage teammap` over our own 352 OCR reads: votes
  **+112 / -54**, giving FOOTPASS team 1 -> our team 1, team 2 -> our team 0.
* **at scoring time** — `footpass_score.resolve_team_map` over the answered events' *team* column:
  identity map 183 hits, flipped map **649** hits, choosing our 0 -> FOOTPASS 2, our 1 -> FOOTPASS 1.

**The same bit, from two disjoint mechanisms, 3.5:1 in favour.** Not a near-tie, so the kit anchor is
genuinely separating the two teams. Team accuracy on answered events is 0.7800 (solver) / 0.8519
(greedy).

## 8. Per action class (diagnostic only — classes 3-8 are too thin to read)

| class | n | solver ans / corr | solver full-cov @ prec | greedy ans / corr | greedy full-cov @ prec |
|---|---|---|---|---|---|
| 1 (pass) | 738 | 345 / 113 | 0.4675 @ 0.3275 | 95 / 71 | 0.1287 @ 0.7474 |
| 2 (reception) | 936 | 434 / 148 | 0.4637 @ 0.3410 | 137 / 98 | 0.1464 @ 0.7153 |
| 3 | 40 | 13 / 1 | 0.3250 @ 0.0769 | 2 / 1 | 0.0500 @ 0.5000 |
| 4 | 36 | 11 / 0 | 0.3056 @ 0.0000 | 0 / 0 | 0.0000 @ n/a |
| 5 | 27 | 6 / 0 | 0.2222 @ 0.0000 | 3 / 0 | 0.1111 @ 0.0000 |
| 6 | 61 | 11 / 1 | 0.1803 @ 0.0909 | 2 / 1 | 0.0328 @ 0.5000 |
| 7 | 10 | 6 / 0 | 0.6000 @ 0.0000 | 1 / 0 | 0.1000 @ 0.0000 |
| 8 | 31 | 6 / 1 | 0.1935 @ 0.1667 | 3 / 2 | 0.0968 @ 0.6667 |

Only classes 1 and 2 (1,674 of 1,879 events) carry enough mass to read. Both arms answer the two
dense classes at essentially the same rate, so nothing here is class-specific.

## 9. Negatives, surprises and limits

1. **Neither arm clears a 0.85 precision floor at any coverage worth reporting.** That is the
   headline. The solver's curve peaks at 0.8169 (coverage 0.0378) and 0.8000 (coverage 0.0905).
2. **Read density on FOOTPASS broadcast is 0.0304, 3.5-4.2x below our own footage** and 8.8% of the
   law's d* = 0.347. Not a crop-budget problem: 78% more crops, 70% fewer reads. Raising `max_crops`
   20 -> 60 would attack the wrong term.
3. **The off-screen "ceiling" is soft.** 0.2215 precision on 149 off-screen answers, 2.4x a
   modal-guess control. `roi = NaN` is an annotation of visibility, not a guarantee of absence.
4. **The measured tracked recall (0.8855 IoU) is 0.10 below the probe's 0.9882 prediction**, but the
   comparison is not like-for-like: ours goes through a foot-point box reconstruction and covers the
   whole match including close-ups. The probe is not refuted; it is also not confirmed.
5. **The greedy arm cannot be moved.** Every confidence is 1.0, its 60-row frontier is a tie-order
   artifact, and it therefore scores 0.0000 at the 0.85 floor despite a 0.7119 operating precision.
   Any use of it must be at its single point.
6. **The solver ran at `r_abstain = 0`** (the frozen `results/identity_solver_config.json`, as the
   task specified), i.e. name-everything with the posterior as a post-hoc dial. That is the
   operating point `RETEST_ABSTENTION.md` measured as catastrophic for LOTO retrieval, and its
   full-coverage precision here (0.3173) is consistent with that. The dial recovers a usable curve,
   which is the convention `EVIDENCE_DENSITY_LAW.md` section 3 uses, but the arm should not be
   shipped at full coverage.
7. **h2 rosters are 27 slots wide against h1's 23**, weakening mutual exclusion in the second half
   (`FOOTPASS_E2E_PLAN.md` section 4). Not split out here.
8. **One game.** game_18 is the worst of the three on the on-screen ceiling (0.758 vs 0.881 / 0.832)
   and its OCR density may or may not be representative. Games 24 and 47 were not run.
9. **A predictor bug was fixed before this run** (section 10) and no pre-fix number exists to
   compare against, so its size is unmeasured.

## 10. One code change made during this run

`tools/footpass_predict.py` mapped the labelled event frame onto the sampling grid with
`event_ledger.bas_to_grid_frame`, which rounds to `event_ledger.STEP = 5`. **This run extracted at
stride 2** (Appendix A.6's decision), so that rounding evaluated the carrier at a frame up to 2
further from the label than necessary. `_nearest_row` already snaps to the nearest frame that
actually exists, so the fix is to hand it the label's own frame:

```python
g = int(gt_frame) + shift - f0     # was: el.bas_to_grid_frame(local)
```

Stride-agnostic, and the selftest now pins the snap on a stride-2 grid instead of pinning the
stride-5 rounding. Median frame offset actually used at scoring time: **0 frames**. `ruff` clean,
`footpass_predict --selftest` and `footpass_score --selftest` both pass. No other module changed;
`SCORER_VERSION` and `METRICS_VERSION` are unchanged because no metric definition moved.

## 11. Files

* `outputs/footpass/preds_game_18_solver.parquet`, `outputs/footpass/preds_game_18_greedy.parquet`
* `results/footpass/val_attribution_game_18_solver.json`,
  `results/footpass/val_attribution_game_18_greedy.json` (full 60-point frontiers, per-split and
  per-action)
* `results/identity_match_footpass_game_18.json` — the solve (9,163 of 11,575 tracks named, 133 of
  163 GK-role tracks, both keeper slots filled: T1#1, T2#16)
* `results/footpass/team_map.json` — the pre-solve team bit
* `outputs/identity/solver/footpass_game_18_solver_names.parquet`,
  `outputs/identity/footpass_game_18_lineup_assign.parquet`
* `outputs/footpass_game_18/final/ocr_percrop/*.parquet` — 11 chunks, 166,930 crops


---

# v2-widened rerun — the crop-box fix of `OCR_DOMAIN_SHIFT.md`, measured end to end

Date: 2026-07-30, evening. `results/OCR_DOMAIN_SHIFT.md` §7 diagnosed our own crop geometry as the
largest single term in game_18's OCR blindness and *projected* the effect of widening the OCR crop
x1.25. This section replaces every projection with a measurement: the whole per-crop OCR pass was
re-run at `--crop-scale 1.25`, and the CPU tail (connector `tau = 0.040` -> frozen Stage-2 solve ->
predict -> score) was re-run on the result. Nothing else changed — same aligned table, same ball,
same PRTreID caches, same `results/identity_solver_config.json`, same lineup, same scorer.

**Headline: the projection did not hold — it was too pessimistic by 1.8x.** Projected
`d = 0.0384` at read precision 0.849; **measured `d = 0.0704` at read precision 0.8673**. The
0.85 precision floor on the attribution metric is still **not** cleared overall.

## v2.1 What was changed in code

The widening is a **parameter, default 1.0**, applied in the OCR crop path only
(`tools/ocr_match.write_chunk_crops` -> `scale_box`, the function `tools/ocr_box_trial.py` already
used and now imports). `generator.team_anchor.estimate_player_box` is **untouched**, so no geometric
consumer moved. Output goes to a variant directory (`--variant _w125` ->
`outputs/footpass_game_18/final/ocr_percrop_w125/`), so the v1 parquets are intact and every arm
below is re-runnable. `OCR_PERCROP_VERSION` 1.0 -> **1.1** and each row carries its `crop_scale`.

**Queued, NOT done (out of scope here):** the root cause is that the detector's real boxes exist at
extract time and are discarded, so `estimate_player_box` reconstructs them from two constants fitted
to nothing (0.814x true on this game). Persisting a per-track median box height — or fitting the two
constants per match from one detector pass — removes the guess for *every* consumer
(`tools/gta_match.py`, the identity chain, the tracked-recall numbers of section 5 above). x1.25 is a
calibration knob standing in for that measurement.

## v2.2 The GPU pass

| | v1 (shipped box) | v2 (`crop_scale 1.25`) |
|---|---|---|
| chunks | 11 | 11 |
| crops | 166,930 | **166,931** |
| compute (sum of per-chunk wall) | 2 h 07 m | **2 h 37 m 38 s** (+24%) |

The crop *set* is identical to within one crop (one box clears `MIN_BOX_H` at 1.25x that did not at
1.0x), and per-chunk track counts match exactly (e.g. `h1_chunk_000`: 18,839 crops / 1,222 tracks in
both). The pass is therefore **paired at the crop level over the whole match**, not sampled.

*Operational note, no effect on data:* the first launch died at chunk 5 of 11 with
`CalledProcessError ... exit status 3221225794` (`0xC0000142`, DLL init failure) when the PARSeq
sidecar was spawned — the launching background shell had been torn down under it. `run_match` is
resumable per chunk, so the run was restarted detached and the four finished chunks were kept. The
+24% is compute time, not wall clock.

## v2.3 Crop level — measured over all 166,931 crops

| quantity | v1 | v2 (x1.25) | ratio |
|---|---|---|---|
| legibility >= 0.5 | 0.0961 | **0.1903** | 1.98x |
| `p_number` >= 0.99 | 0.0354 | **0.0735** | 2.08x |
| per-crop read precision vs GT shirt | 0.6516 (5,058 graded) | **0.7170** (10,544) | **+0.065** |

The paired 1,751-crop trial of `OCR_DOMAIN_SHIFT.md` §7.1 predicted 2.56x on confident reads and
+0.039 precision. Over the full corpus the gain is **2.08x at +0.065 precision** — the trial's
frame-sample was optimistic on volume and pessimistic on precision, and both land in the same place:
**more reads, and better ones.**

## v2.4 Read density `d` and read precision vs ground truth — the pre-declared measurement

Denominator 11,575 player/GK `(chunk, track_id)` pairs, exactly as section 4 above. Read precision =
dominant read of a read tracklet against its GT dominant shirt.

| arm | `d` | tracks read | read precision | graded | projection |
|---|---|---|---|---|---|
| v1, `min_votes 5` (shipped rule) — **control** | 0.0304 | 352 | 0.7860 | 299 | — |
| **v2 x1.25, `min_votes 5` — PRIMARY** | **0.0704** | **815** | **0.8673** | 701 | 0.0384 @ 0.849 |
| v1, `min_votes 4` | 0.0402 | 465 | 0.7481 | 397 | — |
| *v2 x1.25, `min_votes 4` — EXPLORATORY, post-hoc* | *0.0918* | *1,063* | *0.8350* | *921* | *0.0577 @ 0.848* |

**The projection did not hold, in the conservative direction.** `d` is **1.83x** the projected value
on the primary arm (1.59x on the exploratory one) and read precision beat its projection too
(+0.018). The projection's error is structural and worth recording: it modelled each currently
non-confident crop as *independently* gaining a confident read with probability `q = 0.0683`. The
extra reads are not independent — they **cluster on the same tracklets** (a player whose crops become
legible gains several at once), and `min_votes = 5` is precisely a bar on per-tracklet clustering.
A binomial model over crops must under-count a tracklet-level threshold, and it did.

Two controls make this readable:

* the grader reproduces v1's `d = 0.0304 (352 / 11,575)` **exactly**;
* its read precision on v1 is **0.7860** where `OCR_DOMAIN_SHIFT.md` §5 recorded **0.8054**. The
  difference is one convention: this grader builds each tracklet's GT shirt from **every** aligned
  player row (7,050 labelled tracklets) rather than from the crop rows of one particular OCR pass
  (6,162), so the label cannot move when the crop geometry moves. 299 vs 298 tracklets are graded and
  ~5 flip label. Every before/after number in this section uses the same grader.

**`d = 0.0704` is still 20% of `EVIDENCE_DENSITY_LAW.md`'s `d* = 0.347`** — 4.9x short, against 11.4x
short before. It is now inside the 0.105-0.126 band of our own EPL matches by a factor of 1.5-1.8
rather than 3.5-4.2.

## v2.5 The new frontiers — attribution-given-event, all 1,879 events

Everything up to naming is unchanged and reproduces exactly: ball+players at the nearest sampled
frame **1,619 (0.8616)**, carrier within 3.0 m **1,233 (0.6562)**, actor on screen **0.7584**.

| arm | full cov | precision | cov@0.85 | cov@0.60 | best knee (cov >= 0.02) |
|---|---|---|---|---|---|
| solver, on record | 0.4428 | 0.3173 | 0.0005 | 0.1575 | 0.8169 @ 0.0378 |
| **solver, v2 x1.25 (PRIMARY)** | **0.4742** | **0.3838** | 0.0005 | **0.2012** | **0.7970 @ 0.1048** |
| *solver, v2 + votes 4 (exploratory)* | *0.4758* | *0.3792* | *0.0005* | ***0.2661*** | *0.7766 @ 0.1453* |
| greedy, on record | 0.1293 | 0.7119 | 0.0000 | 0.1293 | single point |
| **greedy, v2 x1.25 (PRIMARY)** | **0.1921** | **0.6371** | 0.0000 | **0.1921** | single point |
| *greedy, v2 + votes 4 (exploratory)* | *0.2406* | *0.6283* | *0.0000* | *0.2406* | *single point* |

On/off-screen, same splits as section 2:

| split | arm | answered | correct | full cov | precision | cov@0.85 | cov@0.60 |
|---|---|---|---|---|---|---|---|
| on-screen (1,425) | solver on record | 683 | 231 | 0.4793 | 0.3382 | 0.0007 | 0.1951 |
| on-screen | **solver v2** | 731 | 305 | **0.5130** | **0.4172** | 0.0007 | **0.2519** |
| on-screen | *solver v2 votes 4* | 732 | 299 | 0.5137 | 0.4085 | *0.0091* | *0.3137* |
| on-screen | greedy on record | 212 | 157 | 0.1488 | 0.7406 | 0.0000 | 0.1488 |
| on-screen | **greedy v2** | 316 | 208 | **0.2218** | 0.6582 | 0.0000 | 0.2218 |
| off-screen (454) | solver on record | 149 | 33 | 0.3282 | 0.2215 | 0.0022 | 0.0066 |
| off-screen | **solver v2** | 160 | 37 | 0.3524 | 0.2313 | 0.0000 | 0.0066 |
| off-screen | greedy on record | 31 | 16 | 0.0683 | 0.5161 | 0.0000 | 0.0220 |
| off-screen | **greedy v2** | 45 | 22 | 0.0991 | 0.4889 | 0.0000 | 0.0441 |

**The best on-screen solver point is now `precision 0.8448 at coverage 0.1221`** (on record:
0.8148 @ 0.1137). That is **0.005 below the 0.85 bar**, at 1.07x the coverage. Overall the bar is
still missed: `cov@0.85 = 0.0005` (one event), unchanged from the on-record run.

The off-screen surprise of section 2 survives the change: 37 of 160 off-screen answers are correct
(0.2313 vs the 0.0940 modal control), so `roi = NaN` remains a soft ceiling.

## v2.6 The factorisation, re-measured

| conditional | solver on record | **solver v2** | greedy on record | **greedy v2** |
|---|---|---|---|---|
| answered | 832 | **891** | 243 | **361** |
| team correct given answered | 0.7800 | 0.7677 | 0.8519 | 0.8116 |
| shirt correct given answered | 0.3197 | **0.3984** | 0.7119 | 0.6371 |
| both correct, over all 1,879 events | 0.1405 | **0.1820** | 0.0921 | **0.1224** |

**End-to-end correct answers: solver 264 -> 342 (+30%), greedy 173 -> 230 (+33%).** The gain is
entirely in the shirt channel, as it must be — the team bit and the carrier gate did not move. The
team bit still resolves the same way from the scorer's own side (identity 207 vs flipped 684, 3.3:1;
on record 183 vs 649).

Solve-side counts: named tracks **9,163 -> 9,387** (GK-role 133 -> 129 of 163). Greedy-side: direct
roster-matching reads **282 -> 725**, named tracks **1,537 -> 2,245** (propagated 1,255 -> 1,520,
groups disagreeing 2 -> **14**), fragments 9,833 -> 4,736 in both (the connector is OCR-independent).

## v2.7 Verdict on the fix

**Bake it in.** For +24% GPU time and no model change it delivers, on this game, **2.3x the read
density at +0.08 read precision**, **+30% end-to-end correct attributions**, and `cov@0.60`
0.1575 -> 0.2012. It is the cheapest measured lever in the identity chain and it is a pure inference
change with a `1.0` default, so no shipped artifact moves until a rerun is asked for.

**It does not change the conclusion of this document.** `cov@0.85` is still 0.0005; `d = 0.0704` is
still 4.9x below `d*`; and the wall named in `docs/GSR_CLUSTER_ROADMAP.md` S3/S4 — a trained identity
model — is still the thing that crosses it. The fix moves the *starting point* of that work, not the
verdict about it.

**On games 24 and 47:** if they are run, run them at `--crop-scale 1.25`. The fix should **not** be
the reason to run them — nothing here suggests either game would clear 0.85 — but there is no reason
to spend 10 GPU-hours a game reproducing a known 0.814x crop-box error.

## v2.8 Negatives and limits of this rerun

1. **The 0.85 floor is still not cleared at any usable coverage.** `cov@0.85 = 0.0005` overall, and
   the best on-screen point (0.8448) misses by 0.005. The headline of this document stands.
2. **The greedy arm lost precision: 0.7119 -> 0.6371 (-0.075)** while coverage rose 1.49x. It gains
   57 correct answers but anyone who wanted its 0.71-precision operating point no longer has it. The
   mechanism is visible in the propagation counts: 2.6x the direct reads feed 1.2x the propagations
   through the same merge groups, and disagreeing groups went 2 -> 14. Reported as a real regression
   on precision, not netted out against the coverage gain.
3. **The solver's knee precision barely moved** (0.8169 -> 0.7970 at 2.8x the coverage). The fix buys
   coverage at a given precision; it does not buy precision.
4. **The projection in `OCR_DOMAIN_SHIFT.md` §7.3 was wrong by 1.8x** (conservative). Its binomial
   crop model cannot represent a per-tracklet vote threshold. Any future `d` projection from a
   crop-level `q` should be treated as a lower bound.
5. **`min_votes = 4` is post-hoc.** It was graded against this game's own ground truth in the same
   document that proposed it, and it is reported in italics throughout for that reason. It is
   *interesting* — `cov@0.60` 0.2012 -> 0.2661 on the solver at flat full-coverage precision — and it
   is **not** adopted.
6. **The grader convention differs from `OCR_DOMAIN_SHIFT.md` §5** (0.7860 vs 0.8054 on the same v1
   artifact), for the arm-independence reason given in v2.4. Comparisons *within* this section are
   like-for-like; a cross-document precision comparison is not.
7. **No EPL rerun.** The 1.31x crop-level gain measured on brighton was not converted into a `d`, and
   there is no EPL ground truth to grade it with. `manutd_*` still run at `crop_scale 1.0`.
8. **One game.** Same limit as the rest of this document.
9. **The root cause is untouched** (see v2.1). x1.25 is 93% of the annotated-ROI oracle on this
   game's crops; on another broadcast the right constant will differ, which is exactly why the knob
   exists and why the per-match fit is queued.

## v2.9 Files (v2 arm)

* `outputs/footpass_game_18/final/ocr_percrop_w125/*.parquet` — 11 chunks, 166,931 crops,
  `ocr-percrop-1.1`, `crop_scale = 1.25`
* `results/ocr_box_grade_footpass_game_18{,_w125}_v{4,5}.json` — the GT-graded `d` / read precision
* `results/gt_track_shirts_footpass_game_18.json` — the arm-independent GT shirt per tracklet
* `results/identity_match_footpass_game_18_w125{,_v4}.json`,
  `outputs/identity/solver_w125{,_v4}/footpass_game_18_solver_names.parquet`
* `outputs/footpass/preds_game_18_{solver,greedy}_w125{,_v4}.parquet`
* `results/footpass/val_attribution_game_18_{solver,greedy}_w125{,_v4}.json` — full frontiers
* Code: `tools/ocr_match.py` (`--crop-scale`, `--variant`, `scale_box`), `tools/ocr_box_trial.py`
  (`--grade`, `gt_track_shirts`), `tools/identity_match.py` and `tools/footpass_predict.py`
  (`--percrop-variant`, `--min-votes`), `generator/jersey_id.py` (`OCR_PERCROP_VERSION` 1.1)
* Claim: `ident-030`.
