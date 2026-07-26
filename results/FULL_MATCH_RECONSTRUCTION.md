# Full-match 22-player reconstruction, validated on real re-appearances (2026-07-26)

One match, both halves, the **frozen** B4 v1 (`results/B4_MODEL_V1.md`) under the adopted P2
defer-to-anchor policy (`results/B4_ABSTENTION_POLICY.md`). Nothing was refitted, no threshold was
moved, no hyper-parameter was touched. Verbatim console trail:
`results/FULL_MATCH_RECONSTRUCTION_runlog.md`. **This** file is the hand-written analysis.

Reproduce (CPU only, ~8 min wall clock, no GPU):

```
python -m tools.full_match_reconstruction --match tottenham_manutd \
       --runlog results/FULL_MATCH_RECONSTRUCTION_runlog.md
```

Code: `tools/full_match_reconstruction.py`, `tests/test_full_match_reconstruction.py` (3 tests).
Artifacts: `results/reconstruction/tottenham_manutd_reconstruction.parquet` (426 808 rows, 14.4 MB)
and `results/reconstruction/tottenham_manutd_reappearance_events.parquet` (1 270 scored events).

**Headline, stated before the tables so it cannot be missed: the frozen predictive regions do not
hold on real broadcast.** They passed 6/6 at both levels on Metrica; here they cover **20.5%** where
they promise 50% and **55.3%** where they promise 90%, failing 0/4 testable buckets at both levels.
The point estimate is statistically indistinguishable from hold-last on the only ground truth we can
measure — but that truth is a biased sample which *cannot* demonstrate skill, and we quantify by how
much. Both of those are negative results and neither is smoothed below.

---

## 0. Match choice, and why

**`tottenham_manutd`** (Tottenham 1-0 Man Utd, 2025-02-16, Sofascore 12436952), over
`manutd_liverpool`. The deciding criterion was Task 2, not Task 1: the validation is what makes this
research, so the match with more measurable re-appearances wins.

| | `manutd_liverpool` | `tottenham_manutd` |
|---|---|---|
| tracks >= 2 s (all chunks) | 3 035 | 4 606 |
| usable re-appearance events (within-track, liveness-guarded) | 892 | **1 371** |
| … of which at 5-10s | 15 | **34** |
| name-gated fragments (>= 3 anchors, on the team sheet) | 59 (12 players) | **79 (15 players)** |
| Sofascore lineup | 11 a side throughout | 11 a side throughout |

The trade is explicit and against Task 1: `tottenham_manutd` fragments *worse* (4 606 track ids for
22 players), so its reconstruction is built from more re-identification debris. We took the worse
reconstruction to get the better validation, and both effects are visible below.

Squad arithmetic is confirmed from the Sofascore oracle's offline cache, not assumed: Tottenham
11 starters / 5 used subs / 5 withdrawn, Man Utd 11 / 1 / 1 — balanced, **no red card**, so the
on-pitch cap is 11 for both teams for the whole match. Substitution windows therefore never change
the arithmetic in this fixture; they would only matter for naming, and naming is dead anyway (§1.3).

---

## 1. Task 1 — the reconstruction

### 1.1 A defect found while building it, and fixed at the assembly layer

`tools/imputation_b4_external.load_ours` ghosts a track **only after its final sighting**, and
`imputation.collect_samples` needs a finite `truth` entry to emit a hidden sample. The consequence,
which nobody had noticed: **the pipeline emitted no prediction at all for a player who dropped out
for two seconds and came straight back** — precisely the occlusions that are validatable, and a
third of all imputable player-time. On the pilot chunk, 100% of accepted ghosts were dead
re-identification fragments before the fix and 34% were live-track occlusions after it.

`tools.full_match_reconstruction.fill_internal_gaps` extends `load_ours`'s own never-scored
placeholder device to gaps *between* two sightings. This is a data-assembly change, not a model
change: every structural field, the velocity and `hold` are all computed from
`np.where(visible, truth, nan)` or at the last **visible** index, so the placeholder is read by
nothing except the decision of which slot-frames get a prediction. Unit-tested
(`test_fill_only_touches_frames_between_two_sightings`).

**This defect is live in `tools/tactical_clip.py`.** The shipped clip renderer draws ghosts only for
dead track fragments and never for a genuinely occluded live player. Its uncertainty panel is
therefore showing the *least* trustworthy class of imputation. Flagged for the orchestrator; not
patched here, because `tactical_clip` is a committed demo artifact.

### 1.2 Coverage profile

Match grid: 24 343 timesteps at 5 fps = **81.1 min** of chunked video. A timestep is *trusted* when
at least 4 gated players are tracked on it (the same liveness notion the validation uses):
**7 293 timesteps = 30.0%**, i.e. 24.3 min. Everything below is per team on those trusted timesteps
(14 578 scored team-timesteps of a possible 14 586; 8 carried no row of either kind).

| players on the pitch | observed only | with reconstruction |
|---|---|---|
| 0 | 0.6% | 0.0% |
| 1 | 2.8% | 0.1% |
| 2 | 8.0% | 0.6% |
| 3 | 13.1% | 1.2% |
| 4 | 15.8% | 1.5% |
| 5 | 15.8% | 3.0% |
| 6 | 14.5% | 3.5% |
| 7 | 10.8% | 4.8% |
| 8 | 7.7% | 5.7% |
| 9 | 5.2% | 6.3% |
| 10 | 3.8% | 7.7% |
| **11** | **1.2%** | **65.1%** |
| >11 (fragmentation artefact) | 0.5% | 0.5% |

| | observed | reconstruction |
|---|---|---|
| mean per team | 5.30 / 11 | **9.82 / 11** |
| all 11 | 1.8% | **65.6%** |
| >= 10 | 5.6% | 73.3% |
| >= 9 | 10.8% | 79.6% |
| >= 8 | 18.5% | 85.3% |
| h1 / h2 split (mean, %all-11) | — | 10.10, 72.5% / 9.36, 54.3% |

The 0.5% ">11" column is honest bookkeeping: the tracker occasionally projects twelve or more gated
same-team tracks onto one frame, which is a re-identification duplicate the squad cap cannot remove
because it caps *ghosts*, not observations.

### 1.3 What the reconstruction is made of

| | count | share of imputed rows |
|---|---|---|
| observed | 78 162 | — |
| imputed, `source="v1"` | 312 004 | 89.5% |
| imputed, `source="anchor"` (0-1s defer bucket) | 36 642 | 10.5% |
| imputed, `source="abstained"` | **0** | 0.0% |
| … of the imputed, `ghost_kind="gap"` (a live track, genuinely occluded) | 104 014 | **29.8%** |
| … of the imputed, `ghost_kind="terminal"` (dead re-id fragment) | 244 632 | **70.2%** |

Three things to take from this table.

1. **Layer B never fires.** Not one row in 348 646 exceeded `R_MAX = 30.4339 m`. The abstention
   machinery is present, frozen and completely inert on real broadcast — the same degeneracy the
   gate report recorded on Metrica (447 rows in 644 792), now at exactly zero.
2. **70% of the reconstruction is not a re-appearing player.** A `terminal` ghost is a track id that
   is never seen again; it is imputed because the tracker lost the player permanently, and the
   player is usually still on screen under a new id. The `DUP_M = 6 m` de-duplication removes the
   ones the model places near a live track, but a ghost that drifts further than 6 m survives as a
   phantom. This is the single biggest structural doubt about the reconstruction and it is a
   **tracking** problem, exactly as `results/B4_TRANSFER_M3.md` §3 predicted.
3. **Identity is absent.** 1 477 of 143 150 trusted rows (1.0%) carry a name. The reconstruction is
   "eleven anonymous shapes per team", not eleven named players.

Sharpness of the imputed rows (trusted timesteps only): median `r90` **6.5 m**, p90 14.9 m; median
`r50` 3.1 m. Horizon mix of the imputed rows: 14.5% / 23.6% / 17.7% / 24.8% / 19.3% / 0.0% across
0-1s … 30s+. **44% of the reconstruction sits at horizons >= 5 s** — and §2 has 35 truth events
there in total. Remember that when reading the validation.

Tidy output schema (`results/reconstruction/tottenham_manutd_reconstruction.parquet`):
`match, half, chunk, frame, time_s, team, team_name, track_id, player_name, x, y, source, observed,
asserted, ghost_kind, tsls_s, r50_m, r90_m, keeper, trusted_ts`.

---

## 2. Task 2 — re-appearance as ground truth

### 2.1 The two linking rules and their false-link risk

**Within-track (primary).** The tracker itself carried the id across the gap: slot visible at `t0`,
absent, visible again at `t1`. Extra guard: **every** frame inside the gap must still carry >= 4
gated players, so a whole-frame calibration/replay outage cannot masquerade as an occlusion. This
guard is not cosmetic — before it, `manutd_liverpool` shows 4 873 within-track "gaps", of which only
892 survive it; the other 3 981 overlap frames where the calibration gate dropped the whole frame,
so the player's absence is a measurement hole rather than an occlusion.

False-link risk: ByteTrack (`supervision.ByteTrack`, `lost_track_buffer=60`) re-associates by IoU
and motion in *image* space. Two consequences, both measured:

* it caps the achievable occlusion length. The longest liveness-guarded within-track gap in the
  whole match is **10.8 s** (8.2 s in `manutd_liverpool`), which is why the 30s+ bucket is empty and
  the 10-30s bucket has 2 events. This is a hard structural ceiling, not a sampling accident.
* it selects re-appearances that came back near where they left (§2.5).

Residual mis-association shows up as physically impossible truth: **26 of 1 259 events (2.1%)**
imply a speed above 12 m/s between `t0` and `t1`. Those are either id switches or gross projection
failures. They are kept in the primary numbers (removing them would remove the largest errors and
flatter the model) and reported separately as a sensitivity.

**Cross-track (secondary, reported and NOT trusted).** Two name-gated fragments of the same player,
each with >= `MIN_ANCHORS = 3` agreeing close-up name reads and a name on the match team sheet, with
no third gated fragment of that name alive in between. False-link risk = per-fragment naming
precision, **which this project has never measured**. It does not matter, because the rule fails for
a different reason: only 79 fragments in the match are name-gated, out of 4 606 tracks (1.7%), so
"the next NAMED fragment" is nowhere near "the next appearance". The 11 events it yields have a
median gap of 200 s with only **21% of the gap frames carrying live tracking** — these are broadcast
outages between two sightings minutes apart, not occlusions. They are scored below and must not be
quoted as validation.

### 2.2 Census — read this before the error table

| link rule | 0-1s | 1-3s | 3-5s | 5-10s | 10-30s | 30s+ | total |
|---|---|---|---|---|---|---|---|
| **within-track (primary)** | 782 | 351 | 91 | 33 | 2 | **0** | **1 259** |
| cross-track (not trusted) | 0 | 1 | 0 | 0 | 0 | 10 | 11 |

**The validation is thick where the model is trivial and empty where the model earns its claim.**
v1's headline wins on Metrica were −5.17 m at 10-30s and −3.92 m at 30s+. Those two buckets carry
**2 and 0** measurable re-appearances here. Above the ~30-per-bucket bar we have 0-1s (782), 1-3s
(351) and 3-5s (91); 5-10s (33) is marginal; 10-30s and 30s+ are **untestable by this method on this
match**, and — because the ceiling is ByteTrack's association buffer, not this fixture — untestable
on any match in the corpus without an identity layer.

### 2.3 Error by occlusion duration, in the Metrica format

RMSE in metres. `emitted` is the P2 policy output (anchor at 0-1s, v1 elsewhere) — the number a
consumer actually receives. `hold` is last-seen-position-held, the trivial baseline.

| horizon | n | emitted | p50 | p90 | anchor | hold | v1 | **Metrica emitted** |
|---|---|---|---|---|---|---|---|---|
| 0-1s | 782 | 2.55 | 0.5 | 2.5 | 2.55 | 2.62 | 2.55 | **0.66** |
| 1-3s | 351 | 3.75 | 2.0 | 5.5 | 3.66 | 3.70 | 3.75 | **1.90** |
| 3-5s | 91 | 5.80 | 2.9 | 8.6 | 5.71 | 5.71 | 5.80 | **3.81** |
| 5-10s | 33 | 7.58 | 4.8 | 11.5 | 7.18 | 8.74 | 7.58 | **6.07** |
| 10-30s | 2 | 10.10 | 8.4 | 12.9 | 8.49 | 3.02 | 10.10 | 9.74 |
| 30s+ | 0 | — | | | | | | 11.24 |
| **ALL** | **1 259** | **3.47** | 0.9 | 4.9 | 3.41 | 3.53 | 3.47 | 8.10 (different mix) |

Paired 1-minute block-bootstrap CI on the RMSE difference, within-track events:

| horizon | vs anchor | vs hold |
|---|---|---|
| 0-1s | +0.00 [+0.00, +0.00] (identical by policy) | −0.07 [−0.16, +0.04] |
| 1-3s | **+0.09 [+0.00, +0.17]** | +0.06 [−0.30, +0.42] |
| 3-5s | +0.09 [−0.20, +0.36] | +0.08 [−0.82, +1.34] |
| 5-10s | +0.39 [−0.24, +1.01] | −1.16 [−2.07, +0.06] |
| **ALL** | +0.07 [−0.01, +0.14] | −0.06 [−0.22, +0.16] |

**On real broadcast re-appearances the model beats nothing.** It is a statistical tie with its own
anchor everywhere except 1-3s, where it is marginally *worse* with a CI that just touches zero, and
a tie with hold-last overall (−0.06 m, CI straddling zero). The only bucket where it is directionally
better than hold is 5-10s (−1.16 m, CI touching zero, n=33).

Errors are 2-4x the Metrica figure at every comparable horizon. §2.4 and §2.5 attribute that gap.

### 2.4 Calibration on real broadcast — the major finding

Empirical coverage of the **frozen** 50% and 90% conformal regions, on real re-appearances:

| horizon | n | PICP50 (nominal 50) | PICP90 (nominal 90) | mean r50 | mean r90 | pass 50 | pass 90 |
|---|---|---|---|---|---|---|---|
| 0-1s | 782 | **17.3** | **51.3** | 0.2 | 0.7 | no | no |
| 1-3s | 351 | **20.5** | **56.7** | 1.2 | 2.6 | no | no |
| 3-5s | 91 | **40.7** | **73.6** | 2.6 | 5.5 | no | no |
| 5-10s | 33 | **39.4** | **81.8** | 4.3 | 8.9 | no | no |
| 10-30s | 2 | 50.0 | 100.0 | 6.3 | 13.5 | (n=2) | (n=2) |
| **ALL** | **1 259** | **20.5** | **55.3** | 0.8 | 1.8 | **no** | **no** |

Metrica holdout, same emitted policy, for comparison: 45.5–52.4 / 89.2–90.6, **6/6 pass at both
levels**. Here it is **0/4 at both levels** on the buckets with enough events to judge.

This is a major finding and it is stated plainly: **a "calibrated 50/90% predictive region" is a
false claim on our own footage.** Anything downstream that quotes the region as calibrated — the
tactical clip's ellipses, any scouting page built on them — is currently over-confident by roughly a
factor of two in area at short horizons.

**Attribution, measured rather than asserted.** Two mechanisms, in order of size:

1. **Our "truth" is not truth.** Metrica truth is exact; ours is a homography projection of a
   detection box. Measured from midpoint deviations of 65 714 consecutive-sighting triples (which
   cancel constant-velocity motion and leave ~0.06 m of acceleration plus noise), the per-axis
   observation noise is **sigma = 0.59 m**, i.e. ~0.83 m RMS radially. The frozen 90% region at 0-1s
   is 0.7 m in radius: **the noise floor alone exceeds it.** Widening the same frozen regions in
   quadrature by that measured sigma — a diagnostic, not a re-calibration, nothing in the model
   changed — moves coverage to:

   | horizon | PICP50 -> noise-widened | PICP90 -> noise-widened |
   |---|---|---|
   | 0-1s | 17.3 -> **68.3** | 51.3 -> **74.7** |
   | 1-3s | 20.5 -> 31.3 | 56.7 -> 62.7 |
   | 3-5s | 40.7 -> 44.0 | 73.6 -> 73.6 |
   | 5-10s | 39.4 -> 39.4 | 81.8 -> 81.8 |
   | ALL | 20.5 -> 55.4 | 55.3 -> 71.4 |

   Observation noise explains most of the 0-1s failure (and over-corrects the 50% level there), and
   explains almost none of the 1-3s and 3-5s failure. It is a necessary correction, not a sufficient
   one.

2. **A heavy tail the regions cannot see.** At 0-1s the median error is 0.5 m but the RMSE is
   2.55 m; at 1-3s, 2.0 m median against 3.75 m RMSE. 2.1% of events are physically impossible
   (>12 m/s implied). Excluding them (diagnostic only — it deletes the largest errors and flatters
   the model) gives RMSE 2.98 m, PICP 20.9 / 56.4: the tail moves the RMSE by half a metre and the
   coverage by less than 1.5 points. So the tail is **not** the main cause of under-coverage; the
   under-coverage is broad, not outlier-driven. The conformal regions are marginal per bucket, so
   they cannot adapt to a per-sample noise level in any case — the limitation §5 of the gate report
   already recorded, now with a measured cost.

Note the shape of the failure: coverage improves monotonically with horizon (51 -> 57 -> 74 -> 82 at
the 90% level). The regions fail worst where they are tightest and the noise floor dominates, and
approach nominal where they are wide. The reconstruction's actual operating regime is wide (median
`r90` 6.5 m, 44% of rows at >= 5 s), so the ALL-row 55.3% is *pessimistic* for the reconstruction —
but the best-tested bucket still misses nominal by 8 points, and there is no evidence at all beyond
10 s.

### 2.5 Selection bias — quantified, with the direction stated

Re-appearance is emphatically not a random sample of occlusions. Two measurements.

**(a) Who comes back.** 12 462 track-losses in the match. 63.0% re-appear under the same id at all;
only **10.1% become usable events** after the liveness guard. Terminal losses had a median 268 s of
chunk remaining — they had ample opportunity to return and did not.

| at the last sighting (medians) | re-appearing | terminal | usable events |
|---|---|---|---|
| distance from the ball | 17.56 m | 19.40 m | 18.34 m |
| normalised distance from the nearest image edge | 0.34 | 0.31 | 0.35 |
| last sighting within 10% of the image edge (a genuine frame exit) | **6.1%** | **11.3%** | **6.4%** |

So the players we can validate on are **nearer the ball and about half as likely to have left the
frame** as the ones we cannot. Our truth set is dominated by *detector dropouts on players the
camera never left*, not by off-camera exits — which is the case the model is actually for.

**(b) How far they moved.** The decisive number. `|truth(t1) - last seen(t0)|`, against the same
statistic on the Metrica holdout, plus the Metrica holdout re-scored on only those samples that
moved no further than our events' p90 (like for like):

| horizon | n | our p50 disp | our mean disp | Metrica p50 disp | displacement-matched Metrica: n / RMSE / PICP50 / PICP90 |
|---|---|---|---|---|---|
| 0-1s | 782 | 0.52 | 1.20 | 0.73 | 47 420 / **0.40** / 55 / 92 |
| 1-3s | 351 | **1.05** | 2.18 | **3.08** | 68 066 / **1.64** / 52 / 92 |
| 3-5s | 91 | **1.88** | 3.67 | **6.10** | 43 262 / **3.15** / 54 / 94 |
| 5-10s | 33 | **4.02** | 6.32 | **10.10** | 74 037 / **4.99** / 58 / 95 |

**Direction of the bias, stated explicitly: our truth sample is far EASIER than a real occlusion.**
Players in our events move 2-3x less than Metrica players over the same elapsed time, because
ByteTrack only re-associates a track that came back near where it left. On such a sample hold-last
is near-optimal by construction, and any model that predicts movement is penalised for doing its
job. That is why §2.3 shows no skill and it is **not** evidence that the model has none.

It cuts the other way for calibration, and this is the part that cannot be explained away: on the
*matched* Metrica subset — same short displacements — the frozen regions still cover 52-58% and
92-95%. On our broadcast events, on comparably easy samples, they cover 17-41% and 51-82%. The
calibration failure survives the bias correction; the accuracy comparison does not.

---

## 3. Task 3 — does reconstructing the missing players change the tactical picture?

Per (trusted timestep, team), outfielders only, in the team's own attacking frame (own goal = 0 m).
Line height = the second-deepest player (the frozen `_team_stats` convention). Depth/width spread =
standard deviation along attacking-x and along y. Restricted to the team-timesteps where the
reconstruction actually adds a player.

**Full reconstruction** — 12 740 team-timesteps scored, >= 1 player added on 11 230 (88%), mean 3.97
added:

| metric | observed only | reconstructed | mean delta | p50 delta | p90 abs delta |
|---|---|---|---|---|---|
| defensive line height | 48.75 m | 44.07 m | **−4.67 m** | −2.48 m | 12.79 m |
| depth spread | 6.58 m | 8.23 m | **+1.65 m** | +1.21 m | 4.87 m |
| width spread | 9.89 m | 11.22 m | **+1.32 m** | +0.86 m | 5.45 m |
| x-span | 17.85 m | 26.02 m | **+8.17 m** | +6.53 m | 19.30 m |

**Gap-ghosts only** (dead re-identification fragments excluded — the conservative read), >= 1 player
added on 7 951 (62%), mean 1.29 added:

| metric | observed only | reconstructed | mean delta | p50 delta |
|---|---|---|---|---|
| defensive line height | 48.75 m | 46.22 m | **−2.53 m** | 0.00 m |
| depth spread | 6.59 m | 7.28 m | +0.69 m | +0.16 m |
| width spread | 9.82 m | 10.38 m | +0.56 m | −0.01 m |
| x-span | 17.62 m | 21.36 m | +3.74 m | +0.53 m |

Scaling with how much is missing (full reconstruction):

| players added | n | line | depth | width |
|---|---|---|---|---|
| +1 | 1 095 | −0.90 m | +0.49 m | +0.29 m |
| +2 to +3 | 2 687 | −2.54 m | +1.39 m | +0.83 m |
| +4 to +11 | 7 448 | **−6.00 m** | +1.92 m | +1.65 m |

**The answer to the payoff question is yes, materially.** A 4.7 m shift in the block line is 28% of
the entire span between `fingerprint.block_height`'s low-block ceiling (33 m) and its high-block
floor (50 m) — enough to reclassify a spell. The visible-only x-span of 17.9 m is not a football
shape at all; the reconstruction turns it into 26.0 m, which is still short of a real team's 40-60 m
but is no longer absurd. Physical sanity holds: 0.04% of reconstructed line heights fall off the
pitch (observed-only 0.00%).

**Two mandatory caveats.**

1. **The sign of the delta is mechanical, not evidential.** Adding players can only lower the
   second-deepest x and only widen a spread. The tables prove that reconstruction *matters*; they do
   not prove it is *right*. Given §2.4 they should be read as "the observed-only numbers are
   biased by at least this much", not as "the reconstructed numbers are correct".
2. **The full-reconstruction delta is inflated by phantoms.** 70% of imputed rows are dead re-id
   fragments; excluding them halves the line shift (−4.67 -> −2.53 m) and cuts the spread shifts by
   ~60%. The truth lies between the two tables and cannot be located without an identity layer.

---

## 4. Threats to validity (all disclosed, none fixed post-hoc)

1. **The masking protocol perturbs the frame it measures.** To obtain a prediction at `t1` the
   validation hides the re-appearing slot at that single frame (and, for cross-track events, the
   successor fragment too, or the model would see the player it is predicting). All such frames are
   masked in one pass, so at a scored frame the visible structure is 1-2 players short of what the
   reconstruction pass had. That makes the validation mildly **pessimistic** and it is not corrected.
2. **`tsls` is measured in track-loss time, not off-camera time.** Only 6.4% of usable events follow
   a sighting near the image edge, so most of our "occlusions" are detector failures on players who
   stayed on screen. Metrica's are true off-camera exits. The two horizon axes are not the same
   physical quantity, which is a second reason the RMSE columns are adjacent rather than comparable.
3. **The liveness guard uses >= 4 gated players.** Chosen a priori as "the camera is tracking
   something"; it is not swept, and no result below depends on the exact value (the failure mode it
   removes — zero players tracked — is unambiguous).
4. **One match.** Both halves, 81 min of grid, 24 min trusted. sigma, PICP and the shape delta are
   single-match measurements with no replication.
5. **The cross-track rule was pre-registered as conservative and is reported as failed.** Its 11
   events are in the tables purely so the attempt is on record.
6. **The 10-30s and 30s+ rows of every table are empty or n<=2.** No statement in this document about
   long-horizon behaviour on real broadcast is supported by evidence, in either direction.

---

## 5. Verdict — should this be extrapolated to the corpus?

**No, not as a shipping product. Yes, as a cheap replication of the two measurements that decide it.**

What the experiment settled:

* **The reconstruction is buildable and dense.** 5.30 -> 9.82 players per team on trusted timesteps,
  a full XI on 65.6% of them, physically sane, one tidy parquet, 8 CPU minutes per match.
* **The regions are not calibrated on our footage.** 20.5% / 55.3% against 50% / 90%, 0/4 buckets,
  and the failure survives correction for the measured 0.59 m observation-noise floor and for the
  selection bias. This is the blocking finding. Until it is resolved, no product may describe the
  imputed positions as carrying a calibrated 50/90% region — and `tools/tactical_clip.py` currently
  does exactly that on every frame it renders.
* **Re-appearance cannot validate the point model.** The linking rule that gives clean truth also
  guarantees the player barely moved (2-3x less displacement than Metrica at the same horizon), so
  the sample cannot distinguish a good model from hold-last. The tie in §2.3 is a property of the
  test, not a measured property of the model.
* **The horizon ceiling is structural.** ByteTrack's association buffer caps measurable occlusions at
  ~10 s. The buckets carrying v1's entire published advantage (10-30s, 30s+) are unreachable by this
  method on *any* match in the corpus. Extrapolating the harness to twelve more matches would
  multiply the 0-3s evidence by twelve and add nothing at all where it is needed.

What to do next, in order, and none of it needs a GPU:

1. **Replicate sigma and PICP on 2-3 more matches** (`--match manutd_liverpool`, `liverpool_manutd`,
   `manutd_brighton`; ~8 min each, one at a time). If the observation-noise floor and the
   under-coverage replicate, the finding is a property of our pipeline and the region claim must be
   retracted project-wide rather than caveated. That is the cheapest decisive experiment available
   and it is the one worth running.
2. **Do not re-conformalise on re-appearances.** It is tempting — 1 259 events with measured truth —
   but §2.5 shows the sample is a biased, near-stationary subset of occlusions, so multipliers fitted
   on it would be tight where the model is actually used. Re-calibration needs either annotated
   off-screen truth or a genuine identity layer, and it is a change to a frozen artifact that needs
   sign-off in any case.
3. **The unblocker is identity, not the model.** Cross-track linking would deliver exactly the
   10-30s and 30s+ events the validation is missing, and would collapse the 70% phantom mass in the
   reconstruction. At 79 name-gated fragments out of 4 606 tracks it is nowhere near able to. Every
   line of this report points at the same dependency that `results/B4_TRANSFER_M3.md` §3 and
   `results/CARRIER_CONSTRAINED_v2.md` already named.
4. **Fix `tools/tactical_clip.py`** to use `fill_internal_gaps`, so it draws genuinely occluded live
   players rather than only dead fragments — and, until item 1 reports, relabel its ellipses as
   "Metrica-calibrated region, not validated on broadcast".

**One-line answer to the question as asked:** the reconstruction works and it moves the tactical
numbers by 2.5-4.7 m of block line, but its advertised uncertainty is wrong on real broadcast by a
factor of about two, and the only ground truth this footage can yield is structurally incapable of
testing the horizons the model is used at — so extrapolate the *diagnostic* to a few more matches,
not the *product* to the corpus.
