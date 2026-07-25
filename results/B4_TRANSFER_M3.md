# B4 M3 — transfer validation of the frozen v1 (2026-07-25)

Answers the question raised by the gate run's own disclosure (`results/B4_MODEL_V1.md` §6.1): v1
places 14.9% of its predictions inside the band where a hidden player cannot be, against 18.7% for
the anchor and 11.5% for the truth — so how much of its **−3.36 m margin over the B7 anchor** is
the simulator's rectangular camera window rather than football?

**Nothing was refitted for sections 1–3.** The v1 heads, the B7 anchor (`tau`, blend weights), the
slot model and the window-tuning target are all the frozen artefacts of the gate run. Every
geometry is scale-tuned **on TRAIN (Game 1)** to the same 11.80 visible players/frame, so the
sweep compares *shapes*, not *amounts of information*.

Harness validation: under the training geometry the harness reproduces the gate run **exactly** —
B7 `0.66 / 2.09 / 4.37 / 7.55 / 14.91 / 15.16`, ALL 11.46; v1 `0.68 / 1.90 / 3.81 / 6.07 / 9.74 /
11.24`, ALL 8.10. Same numbers as `results/B4_MODEL_V1.md`, to the last digit.

Code: `tools/imputation_b4_transfer.py` (geometry sweep + mitigation),
`tools/imputation_b4_external.py` (SkillCorner + our own tracks),
`tests/test_imputation_transfer.py`. Verbatim console output for every run, with the exact
commands: `results/B4_TRANSFER_M3_runlog.md`.

---

## 0. What a real broadcast camera window actually looks like (measured, not assumed)

Before choosing test geometries, we measured one. SkillCorner opendata match 1886347 reports a
per-frame camera footprint (`image_corners_projection`, 37 530 frames) **and** a detected /
extrapolated flag per player. Measured there:

| statistic | value |
|---|---|
| players reported per live frame | 22.0 (13.04 of them **detected**) |
| detected players inside the reported footprint | **98.4%** |
| extrapolated players inside the footprint | **5.3%** |
| detection rate at 5–10 m *inside* the footprint edge | 97.7% |
| detection rate at 2–5 m *outside* the edge | 1.9% |
| footprint shape (median) | trapezoid: half-width **11.0 m** at the near edge (y = −30.1), **24.1 m** at the far edge (y = +39.0) |

Three consequences, all of which reshape the transfer question:

1. **Real censoring is also a hard geometric window.** The detection probability crosses 0% → 94%
   within about 2 m of the footprint edge. "Hidden ⇒ outside the visible region" is not a
   simulator artefact; it is true of real broadcast tracking too. What the simulator gets wrong is
   the window's *shape*, not the existence of the constraint.
2. **The real window is a trapezoid that widens away from the camera**, spanning 22 m of pitch
   width at the near touchline and 48 m at the far one.
3. **Its aspect ratio is close to ours by accident.** Mean footprint width ≈ 32 m over a y-extent
   of ≈ 64 m, i.e. ≈ 0.50 — the same as our simulator's 33.8 m × 68 m rectangle. That single
   number turns out to decide whether v1 transfers (§1).

Caveat carried throughout: the *positions* of undetected players in that file are SkillCorner's own
estimates, so the "5.3% inside" figure is partly a property of their extrapolator. The 98.4% for
detected players is not — that side is hard evidence about what the footprint means.

---

## 1. Task 1 — the frozen v1 under nine censoring geometries

Margin = `RMSE(v1) − RMSE(B7)` on the holdout (Game 2 H2) re-censored by each geometry, paired
1-minute block-bootstrap CI, 400 replicates. Negative = v1 better. Absolute RMSE is **not**
comparable across rows (each geometry hides a different set of player-frames); the margin is.

| geometry | window | aspect (x:y) | B7 | v1 | **margin [95% CI]** | kept | v1 inside the visible region |
|---|---|---|---|---|---|---|---|
| `rect_base` (training) | 33.8 × 68 m rectangle | 0.50 | 11.46 | 8.10 | **−3.36 [−4.27, −2.53]** | 100% | 4.0% |
| `lag_1s` | same, camera 1.0 s late | 0.49 | 11.59 | 7.87 | **−3.72 [−4.67, −2.76]** | 111% | 3.7% |
| `lag_2s` | same, camera 2.0 s late | 0.50 | 11.56 | 7.89 | **−3.67 [−4.61, −2.64]** | 109% | 4.0% |
| `trapezoid_sc` | SkillCorner footprint | 0.50 | 12.08 | 8.32 | **−3.76 [−4.69, −2.76]** | 112% | 5.9% |
| `sc_realistic` | footprint + 0.5 s lag + ~2 m feathered edge | 0.50 | 11.64 | 8.10 | **−3.54 [−4.48, −2.53]** | 105% | 9.4% |
| `soft_edge` | rectangle, per-player edge jitter N(0, 4 m) | 0.51 | 10.73 | 7.95 | **−2.78 [−3.69, −2.02]** | 83% | 11.8% |
| `ellipse` | elliptical / vignette window | 0.60 | 11.01 | 8.46 | **−2.55 [−3.31, −1.86]** | 76% | 4.5% |
| `aspect_25` | 42.9 × 50 m rectangle | 0.86 | 10.60 | 8.60 | **−2.00 [−2.69, −1.38]** | 60% | 3.9% |
| `aspect_15` | 110 × 30 m band | 3.67 | 11.04 | 10.81 | **−0.24 [−1.28, +0.85]** | **7% (n.s.)** | 6.2% |

"kept" = margin as a percentage of the training geometry's margin. "v1 inside" = fraction of v1's
predictions falling inside that geometry's visible region, where a hidden player cannot be; the
truth scores 0.0% by construction for every hard geometry (10.3% / 5.1% for the two feathered ones,
because a feathered edge legitimately hides players inside the nominal window).

**Answer to the question as asked: yes, the margin survives a camera v1 never trained on — for
every camera shaped like a real broadcast camera — and it shrinks to nothing only for a window
aspect no broadcast uses.**

Per-bucket detail for the two extremes (metres):

| horizon | B7 `trapezoid_sc` | v1 | margin | | B7 `aspect_15` | v1 | margin |
|---|---|---|---|---|---|---|---|
| 0-1s | 0.68 | 0.68 | +0.00 [−0.01, +0.03] | | 0.49 | 0.59 | **+0.10** [+0.07, +0.13] |
| 1-3s | 2.40 | 2.13 | −0.27 [−0.34, −0.21] | | 2.18 | 2.30 | **+0.11** [+0.03, +0.22] |
| 3-5s | 4.75 | 4.02 | −0.73 [−0.95, −0.56] | | 4.54 | 4.67 | +0.13 [−0.07, +0.42] |
| 5-10s | 7.84 | 6.08 | −1.76 [−2.32, −1.23] | | 7.67 | 7.85 | +0.17 [−0.29, +0.81] |
| 10-30s | 15.46 | 9.93 | −5.53 [−7.17, −3.57] | | 12.47 | 13.33 | +0.86 [−0.36, +2.00] |
| 30s+ | 15.69 | 11.28 | −4.41 [−5.13, −3.56] | | 18.87 | 16.53 | −2.33 [−4.94, +0.45] |
| **ALL** | 12.08 | 8.32 | **−3.76** | | 11.04 | 10.81 | **−0.24** |

Under the real broadcast footprint v1 wins 5/6 buckets with non-overlapping CIs and ties at 0-1s —
the identical verdict shape to the gate run. Under the 110 × 30 m band it *loses* significantly in
the two shortest buckets and wins nothing significantly.

### Diagnosis

The dependence is on **window aspect**, and it is monotone: 0.50 → 100%, 0.60 → 76%, 0.86 → 60%,
3.67 → 7%. The mechanism is visible in the feature set. v1's strongest inputs are ball-relative
(`ball_x`, `ball_disp_x`, `dist_last_ball_now`), and the camera follows the ball in x, so "this
player is hidden" carries the learnt implication "this player is displaced **in x** from the ball".
In the 110 × 30 m band, hiddenness is instead a statement about **y**, and the learnt implication is
simply false — so the correction is applied in the wrong axis and v1 falls back to (slightly worse
than) its anchor.

Two secondary effects, both mild:

- **Edge softness costs ~17%** (`soft_edge`, 83%). When the visibility boundary is not a step
  function, "hidden ⇒ beyond the edge" becomes probabilistic and part of the correction is wasted.
  The measured real edge (§0) is much sharper than the 4 m jitter tested here, so this is a
  pessimistic bound: `sc_realistic`, which uses a ~2 m feathered edge on the measured shape, keeps
  105%.
- **Camera lag costs nothing** (109–111%, i.e. slightly *better* than baseline). A late operator
  makes the anchor worse faster than it makes v1 worse, which is the expected direction: the
  structure features are computed from where the visible players actually are, not from where the
  camera is pointing.

The "learnt censoring geometry" concern from the gate run is therefore **real but bounded**: v1 has
learnt a geometric prior, that prior is *correct* for broadcast-shaped windows (including the real
measured one), and it only misfires when the window's aspect is changed by more than ~2x.

---

## 2. Task 2 — SkillCorner real-tracking check, and its honest limits

Setup: SkillCorner's **detected** points are treated as observed and its own extrapolated segments
as the hidden set. The frozen v1 and the frozen B7 anchor are fed SkillCorner-derived features
(10 fps, 22 players, real broadcast camera, A-League) and scored against SkillCorner's reported
positions for the hidden players. 386 763 hidden samples; horizon mix 10/21/13/17/20/19%.

| horizon | n | B7 | v1 | margin [95% block CI] |
|---|---|---|---|---|
| 0-1s | 40 426 | **0.42** | 1.06 | **+0.64** [+0.57, +0.70] |
| 1-3s | 79 330 | **1.90** | 2.37 | **+0.48** [+0.34, +0.61] |
| 3-5s | 50 553 | 4.18 | 4.15 | −0.03 [−0.24, +0.19] |
| 5-10s | 65 465 | 7.31 | **6.77** | −0.53 [−1.01, −0.15] |
| 10-30s | 78 368 | 11.61 | **10.17** | −1.44 [−1.91, −0.94] |
| 30s+ | 72 621 | 13.13 | **10.84** | −2.30 [−2.99, −1.44] |
| **ALL** | 386 763 | 8.47 | **7.37** | **−1.10** [−1.43, −0.76] |

**What this can establish.** Both estimators are measured against the *same* third-party reference
on real broadcast data. v1 agrees with a commercial vendor's off-screen estimates better than the
training-free anchor does, and the advantage grows monotonically with horizon (−0.53 → −1.44 →
−2.30 m). That is the same qualitative shape as the Metrica result, on data with a real camera, real
detection dropout, a different league and a different frame rate.

**What this cannot establish, and must not be quoted as.** SkillCorner's extrapolated positions are
**not ground truth** — they are another estimator's output, and for a delivered data product almost
certainly a *bidirectional* one (it can use the re-sighting). So:

- No absolute accuracy claim is licensed. "v1 is 7.37 m accurate on real broadcast" is **false**;
  the correct statement is "v1 sits 7.37 m from SkillCorner's own estimate".
- The short-horizon losses are largely an artefact of that reference. At 0-1s their reference is
  essentially a smooth continuation of the detected track, which is what B7's velocity-decay term
  computes — so B7 scores 0.42 m almost by construction, and any method that predicts *differently*
  is penalised regardless of being right. The bias favours the anchor exactly where the anchor
  already had no headroom.
- Conversely at long horizons their estimator must use structure (or the future), so agreement
  there is more informative — but a residual "both use structure" affinity cannot be ruled out.
- One match, one competition, one vendor. n = 386 763 rows is not 386 763 independent facts; the CIs
  are 1-minute block-bootstrapped for that reason.

**The strongest thing this dataset does establish is not the RMSE table at all** — it is §0: on real
broadcast tracking, detection *is* essentially a geometric window with a ~2 m soft edge. That is
what makes the §1 sweep an honest proxy rather than a guess.

Distributional check on the same real data (see §3 for the metric definitions):

| method | off-pitch | inside the real camera footprint | implied speed p50 / p90 / max (m/s) |
|---|---|---|---|
| SkillCorner's own estimate | 0.4% | **5.2%** | 1.17 / 3.09 / 13.01 |
| last-seen | 0.4% | 11.0% | 0 / 0 / 0 |
| B7 anchor | 1.0% | 11.4% | 0.90 / 2.60 / 10.63 |
| **v1** | **0.3%** | **7.5%** | 1.11 / 3.03 / **36.11** |

v1 is closer than the anchor to the vendor's own behaviour on every column: fewer impossible
placements (7.5% vs 11.4%, vendor 5.2%), fewer off-pitch placements, and a displacement
distribution within 5% of theirs at both the median and the 90th percentile. The 36.1 m/s maximum is
a single physically impossible outlier and is the kind of failure the uncertainty head is supposed
to flag; it is not caught by the marginal conformal regions (the same limitation §5 of the gate
report already recorded).

---

## 3. Task 3 — smell test on our own footage (NOT a validation)

No truth exists off-camera in our pipeline, so nothing here is scored for accuracy. Slots are
**tracks**, a hidden sample is a track that has no detection at that frame, and a dead track is
followed for up to 60 s. Numbers are distributional plausibility only.

| | brighton_manutd (3 chunks) | manutd_liverpool (2 chunks) |
|---|---|---|
| tracks ≥ 2 s | 1 351 | 894 |
| frames with any trusted sighting | 2 442 of 8 731 (28%) | 1 334 of 5 849 (23%) |
| visible players on those frames | mean 10.06 (p50 10, p90 15) | mean 9.81 (p50 9, p90 17) |
| hidden samples scored | 372 577 | 200 086 |
| horizon mix | 1 / 4 / 4 / 9 / 34 / 49% | 1 / 4 / 4 / 9 / 34 / 49% |
| v1 off-pitch rate | **0.2%** (anchor 0.9%) | **0.3%** (anchor 1.3%) |
| v1 inside the visible bounding box | 50.4% (anchor 44.4%, last-seen 28.6%) | 47.4% (anchor 44.5%, last-seen 27.9%) |
| … restricted to genuine frame exits | **34.2%** | **36.8%** |
| implied speed p50 / p90 / max | 0.53 / 1.39 / 22.76 m/s | 0.55 / 1.43 / 15.14 m/s |
| team x-span, visible-only → completed | 17.5 → 37.4 m (never > 90 m) | 17.1 → 36.5 m (never > 90 m) |

**Reads as sane:** off-pitch placement is rare and *rarer than the anchor's*; completing the team
with imputed players turns a 17 m visible span into a 37 m team length, which is a plausible
football shape and never explodes; median implied speed is below walking pace, consistent with a
sample dominated by long horizons.

**Reads as a warning:** v1 puts a third to a half of its predictions inside the region the camera
can see, against 7.5% on SkillCorner. Three causes, in order of size, none of which is the camera
*shape*:

1. **Track fragmentation dominates the sample.** Only 12–14% of hidden samples follow a sighting
   near the image edge; the rest are re-identification drops, where the player really is still on
   screen under a new track id. For those, "inside the visible region" is the *correct* answer, so
   this statistic is an upper bound by construction. The edge-exit subset (34–37%) is the honest
   figure, and it is still ~5x SkillCorner's.
2. **The horizon mix is far longer than anything v1 was gated on** — 83% of samples are ≥ 10 s,
   against 53% on the Metrica holdout, because a dead track is ghosted for 60 s. At long horizons v1
   correctly regresses toward team structure, which sits inside the camera view.
3. **The visible bounding box is a lower bound on the true footprint** (it stops at the outermost
   detected player), so the true rate is somewhat lower than printed. We store no per-frame
   homography, so an exact footprint test like §0's is not currently possible on our own data.

**What this actually flags for the pipeline:** the obstacle to using imputation on our footage is
not the camera model — it is that our tracker delivers ~10 trusted players on ~25% of frames with
heavy re-id churn, i.e. a much sparser and more fragmented visible structure than either Metrica
(11.8/22 every frame) or SkillCorner (13.0/22 every frame). Any B4 number that ships into a
scouting report needs B2 identity to be applied first so that a "hidden player" means a player, not
a dead track.

---

## 4. Task 4 — mitigation (v1.1, camera-shape randomised training)

**v1 stands as gated. v1.1 is a separate, clearly-labelled model and does not replace it.** It is
the cheapest mitigation the plan allows: *same* architecture, *same* hyper-parameters, *same*
frozen B7 anchor and slot model, *same* 36-column feature allowlist — only the TRAIN censoring
changes. TRAIN (Game 1) is censored four ways instead of one (`rect_base`, `aspect_25`,
`aspect_15`, `lag_1s`), every second sample kept from each, so v1.1 sees ~2x v1's rows spread over
four window shapes. No feature was dropped: the window-edge information lives in the ball columns,
which carry most of v1's long-horizon signal, so deleting them was not tested — randomising the
window teaches the same lesson without paying that price.

The evaluation geometries split into **in-family** (a shape v1.1 trained on) and **out-of-family**
(a shape it never saw — different functional form, not just different parameters). Only the
out-of-family rows are evidence of robustness; the in-family rows show the mitigation worked at all.

| geometry | family | v1 RMSE | v1.1 RMSE | v1 margin | **v1.1 margin [95% CI]** | v1.1 inside |
|---|---|---|---|---|---|---|
| `rect_base` (training) | in | 8.10 | **7.98** | −3.36 | **−3.48** [−4.39, −2.65] | 3.8% |
| `aspect_25` | in | 8.60 | **7.85** | −2.00 | **−2.75** [−3.35, −2.20] | 3.9% |
| `aspect_15` | in | 10.81 | **8.68** | −0.24 (n.s.) | **−2.36** [−3.18, −1.57] | 4.0% |
| `trapezoid_sc` | **OUT** | **8.32** | 8.69 | −3.76 | −3.38 [−4.45, −2.39] | 5.3% |
| `sc_realistic` | **OUT** | **8.10** | 8.30 | −3.54 | −3.34 [−4.33, −2.37] | 9.1% |

**What it bought, in one line: the aspect failure is fixed, and it costs 6–10% of the margin on the
cameras we actually have.**

- The collapse is repaired. `aspect_15` goes from a statistical tie (−0.24 m, CI straddling zero) to
  a significant −2.36 m — 70% of the training-geometry margin recovered on the window shape that
  broke v1. `aspect_25` improves from −2.00 to −2.75 m (60% → 82% of base).
- It is not free on realistic cameras. On the two out-of-family broadcast-shaped geometries v1.1 is
  *worse* than v1 by 0.37 and 0.20 m of margin (−3.38 vs −3.76; −3.34 vs −3.54) — the usual
  domain-randomisation trade: a flatter model across shapes, slightly weaker on any single one. Both
  remain significant wins over the anchor.
- Robustness spread, the number that decides adoption: v1's margin ranges over **−0.24 to −3.76 m**
  across the nine geometries; v1.1's over the five measured here ranges **−2.36 to −3.48 m**.
- It does not degrade physical plausibility: impossible placements 3.8–9.1%, at or below v1's.

**Should we adopt it? Not yet, and not for the thesis claim.** Three reasons, all procedural rather
than performance:

1. v1.1 has been evaluated on its **point head only**. Gate 2 (conformal coverage), the SGR
   threshold `R_max` and the skill horizon `b*` have not been re-derived for it, so it has no
   calibrated uncertainty and no abstention — the parts that make the novelty claim. Adopting it
   means re-running the whole frozen protocol from `docs/B4_MODEL_PLAN.md` §3–4 and re-freezing.
2. Its advantage is insurance against a camera framing we have measured and do **not** have.
   SkillCorner's real footprint sits at aspect ≈ 0.50, the same as the training rectangle; on that
   family v1 is the better model.
3. Two of the three geometries where v1.1 wins are geometries it trained on, which is the weakest
   possible evidence. The honest generalisation test — out-of-family — has v1.1 *behind*.

**Recommendation.** Keep frozen v1 as the gated, shipping model for broadcast footage. Hold v1.1 as
the pre-tested remedy with its numbers on record, and promote it (with a full re-freeze) the moment
B4 is pointed at footage whose framing is not standard broadcast — tactical cam, vertical/social
crops, or any source whose measured window aspect is more than ~1.5x from 0.50. Measuring that
aspect is cheap: it is the §0 procedure, and it needs one afternoon per new source.

---

## 5. Verdict

**Does the gain transfer? Yes, conditionally — and the condition is met by real broadcast cameras.**

1. **Not simulator-specific.** Under the SkillCorner-measured broadcast footprint, with camera lag
   and a feathered edge, the frozen v1 keeps **105–112%** of its training-geometry margin
   (−3.54 / −3.76 m vs −3.36 m), winning 5/6 horizon buckets with non-overlapping CIs. Camera lag
   costs nothing.
2. **But the model has learnt a geometric prior, and its validity is aspect-dependent.** Change the
   window aspect by 1.7x and 40% of the margin goes; change it by 7x and all of it goes
   (−0.24 m, CI straddles zero). The prior "hidden ⇒ displaced in x from the ball" is what carries
   the long-horizon gain, and it is only true for windows shaped like broadcast windows.
3. **Real censoring is geometric too**, measured, not assumed: 98.4% of detected players inside the
   reported footprint, 94.7% of undetected ones outside, transition width ~2 m. So the learnt prior
   is a *correct* piece of physics about broadcast football, not a simulator exploit — the earlier
   worry that the model was cheating off a rectangle is answered in the negative, with the caveat
   that it would misfire on a differently-framed camera (a tactical wide shot, a vertical crop).
4. **The real transfer obstacle is our own tracking, not the camera.** On our footage v1 stays
   physically plausible (0.2–0.3% off-pitch, sane speeds, sane team shape) but works from ~10
   trusted players on ~25% of frames with heavy re-id fragmentation, and places 34–37% of genuine
   frame-exit predictions inside the visible region against 7.5% on SkillCorner.
5. **A mitigation exists, is tested, and is not needed yet.** Train-time camera randomisation
   (v1.1) turns the `aspect_15` collapse into a significant −2.36 m win and improves `aspect_25`
   from −2.00 to −2.75 m, at a cost of 0.20–0.37 m of margin on the out-of-family broadcast-shaped
   geometries. It is held in reserve, not adopted: it has no calibrated uncertainty or abstention
   yet, and the cameras we have are the ones v1 is best on (§4).

**Recommended standing caveat for any B4 claim:** "validated under broadcast-shaped censoring
(aspect ≈ 0.5, measured on SkillCorner opendata); the margin degrades for camera framings more than
~2x from that aspect and has not been validated on tactical-cam or vertical-crop footage."

**What still cannot be claimed:** any absolute accuracy figure on real broadcast footage. The
SkillCorner comparison is against another estimator, and our own footage has no truth at all. The
only accuracy numbers that remain licensed are the Metrica holdout ones, now with a measured
statement about which cameras they generalise to.
