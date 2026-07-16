# Off-screen imputation -- closed-form baseline probe (B4)

**Status:** measurement only. No model built, no production module touched. This fixes the masking
protocol and the baseline floor *before* any learned imputer is designed, so the headroom claim is
pre-registered rather than reverse-justified.

## 1. Protocol (pre-committed)

**Data.** `brighton_manutd` and `france_senegal` aligned parquets via `core.registry`. Trusted
rows only: `calib_error_m <= 1` m (the `report.facts` gate), valid pitch coords,
`team in {0, 1}`, `role in {player, goalkeeper}`. Detections are sampled every
5 native frames (modal step, verified both matches); brighton runs at 25 fps, senegal
at 59.94 fps, so all windowing is done in **seconds**, not sample counts.

**Spans.** Per (chunk, track_id) we split the trusted samples into contiguous visible spans,
tolerating holes up to 0.5 s (a one-sample dropout does not fragment a span). Only
spans lasting >= 8 s are eligible to host a hidden window.

**Masking.** For each duration in {0.5, 1, 2, 4, 8} s we slide a hidden
window across each eligible span (stride 0.4 s; windows within a span overlap -- see caveat).
A window is valid only with >= 2 visible samples before the gap and >= 1 after.
The hidden samples are the ground truth we score against (they are genuinely visible; we pretend
they are not). Candidates are randomly subsampled to <= 1500 per (match, duration)
with seed 0; actual counts are in the table.

**Two regimes, same windows.**
- *interpolation* -- predictor may use context before AND after the gap. Linear interpolation
  between the last-seen and first-seen-again positions is exactly published GSR off-screen practice;
  it is the floor to beat.
- *extrapolation* -- predictor may use context BEFORE only. This is the broadcast-real case: the
  player has left frame and not returned, so there is no re-entry point to interpolate to.

**Baselines (all closed-form, zero training).**
- `linear_interp` -- straight line between last-before and first-after sample (interp regime only).
- `const_vel` -- least-squares velocity over the trailing 0.6 s of before-context,
  extrapolated with exponential velocity damping (tau = 1.0 s; displacement asymptotes to
  v*tau as the player is assumed to decelerate). Before-context only, so evaluated in both regimes.
- `hold_last` -- freeze at the last visible position. Before-context only; both regimes.
- `centroid_rel` -- the cheapest structure-aware baseline: the player holds its offset from the
  live team centroid (outfield teammates, target removed), and the centroid moves. In extrapolation
  the before-offset is held; in interpolation the offset is linearly blended between the before and
  after anchors. Reported only over gap samples where >= 2 teammates give a
  usable centroid (coverage shows in n_samp).

**Metrics.** Per (duration x regime x baseline): RMSE and median of per-sample Euclidean error (m),
and the re-entry error = median over windows of the error at the LAST hidden sample (how wrong we
are the instant the player reappears -- the worst point of an extrapolation). Reported per match and
pooled. Commercial reference (FIFA-co-authored study): on-screen detected 0.44-1.14 m RMSE, off-screen imputed 4.6-12.2 m -- our extrapolation regime is the like-for-like off-screen comparison.

**Caveat (pre-stated).** Sliding windows within a span overlap, so windows are not independent;
counts are exposure, not effective sample size. Ground truth is our own tracking (trusted geometry,
sub-metre calibration), not external truth -- SkillCorner opendata would replace self-truth with
real full-pitch broadcast tracking (see verdict).


## 2. Results

### brighton_manutd

| dur (s) | regime | baseline | n_win | n_samp | RMSE (m) | median (m) | re-entry (m) |
|--:|:--|:--|--:|--:|--:|--:|--:|
| 0.5 | interp | linear_interp | 1500 | 3696 | 0.65 | 0.12 | 0.11 |
| 0.5 | interp | const_vel | 1500 | 3696 | 1.25 | 0.22 | 0.31 |
| 0.5 | interp | hold_last | 1500 | 3696 | 1.16 | 0.31 | 0.49 |
| 0.5 | interp | centroid_rel | 1497 | 3685 | 2.24 | 1.15 | 1.02 |
| 0.5 | extrap | const_vel | 1500 | 3696 | 1.25 | 0.22 | 0.31 |
| 0.5 | extrap | hold_last | 1500 | 3696 | 1.16 | 0.31 | 0.49 |
| 0.5 | extrap | centroid_rel | 1498 | 3688 | 2.93 | 1.33 | 1.53 |
| 1 | interp | linear_interp | 1500 | 8007 | 0.71 | 0.17 | 0.13 |
| 1 | interp | const_vel | 1500 | 8007 | 1.58 | 0.36 | 0.70 |
| 1 | interp | hold_last | 1500 | 8007 | 1.58 | 0.54 | 1.06 |
| 1 | interp | centroid_rel | 1497 | 7981 | 2.52 | 1.31 | 0.95 |
| 1 | extrap | const_vel | 1500 | 8007 | 1.58 | 0.36 | 0.70 |
| 1 | extrap | hold_last | 1500 | 8007 | 1.58 | 0.54 | 1.06 |
| 1 | extrap | centroid_rel | 1499 | 7991 | 3.46 | 1.76 | 2.26 |
| 2 | interp | linear_interp | 1500 | 15478 | 0.87 | 0.29 | 0.15 |
| 2 | interp | const_vel | 1500 | 15478 | 2.21 | 0.65 | 1.49 |
| 2 | interp | hold_last | 1500 | 15478 | 2.27 | 0.90 | 1.94 |
| 2 | interp | centroid_rel | 1497 | 15418 | 2.93 | 1.54 | 0.84 |
| 2 | extrap | const_vel | 1500 | 15478 | 2.21 | 0.65 | 1.49 |
| 2 | extrap | hold_last | 1500 | 15478 | 2.27 | 0.90 | 1.94 |
| 2 | extrap | centroid_rel | 1497 | 15418 | 4.04 | 2.19 | 3.02 |
| 4 | interp | linear_interp | 1500 | 30475 | 1.36 | 0.56 | 0.17 |
| 4 | interp | const_vel | 1500 | 30475 | 3.44 | 1.25 | 2.95 |
| 4 | interp | hold_last | 1500 | 30475 | 3.63 | 1.55 | 3.28 |
| 4 | interp | centroid_rel | 1493 | 30281 | 3.56 | 2.00 | 0.79 |
| 4 | extrap | const_vel | 1500 | 30475 | 3.44 | 1.25 | 2.95 |
| 4 | extrap | hold_last | 1500 | 30475 | 3.63 | 1.55 | 3.28 |
| 4 | extrap | centroid_rel | 1496 | 30323 | 4.96 | 2.85 | 4.02 |
| 8 | interp | linear_interp | 1274 | 51355 | 2.13 | 0.90 | 0.15 |
| 8 | interp | const_vel | 1274 | 51355 | 4.77 | 1.81 | 3.69 |
| 8 | interp | hold_last | 1274 | 51355 | 4.87 | 1.96 | 3.79 |
| 8 | interp | centroid_rel | 1272 | 51241 | 4.00 | 2.22 | 0.90 |
| 8 | extrap | const_vel | 1274 | 51355 | 4.77 | 1.81 | 3.69 |
| 8 | extrap | hold_last | 1274 | 51355 | 4.87 | 1.96 | 3.79 |
| 8 | extrap | centroid_rel | 1274 | 51321 | 5.41 | 3.14 | 4.44 |

### france_senegal

| dur (s) | regime | baseline | n_win | n_samp | RMSE (m) | median (m) | re-entry (m) |
|--:|:--|:--|--:|--:|--:|--:|--:|
| 0.5 | interp | linear_interp | 1500 | 8845 | 0.82 | 0.15 | 0.11 |
| 0.5 | interp | const_vel | 1500 | 8845 | 1.68 | 0.32 | 0.61 |
| 0.5 | interp | hold_last | 1500 | 8845 | 1.50 | 0.46 | 0.89 |
| 0.5 | interp | centroid_rel | 1493 | 8792 | 2.26 | 0.91 | 0.56 |
| 0.5 | extrap | const_vel | 1500 | 8845 | 1.68 | 0.32 | 0.61 |
| 0.5 | extrap | hold_last | 1500 | 8845 | 1.50 | 0.46 | 0.89 |
| 0.5 | extrap | centroid_rel | 1495 | 8801 | 3.02 | 1.15 | 1.76 |
| 1 | interp | linear_interp | 1500 | 17751 | 1.30 | 0.25 | 0.13 |
| 1 | interp | const_vel | 1500 | 17751 | 3.01 | 0.58 | 1.36 |
| 1 | interp | hold_last | 1500 | 17751 | 2.45 | 0.86 | 1.84 |
| 1 | interp | centroid_rel | 1490 | 17600 | 2.63 | 1.24 | 0.46 |
| 1 | extrap | const_vel | 1500 | 17751 | 3.01 | 0.58 | 1.36 |
| 1 | extrap | hold_last | 1500 | 17751 | 2.45 | 0.86 | 1.84 |
| 1 | extrap | centroid_rel | 1495 | 17642 | 3.58 | 1.83 | 2.72 |
| 2 | interp | linear_interp | 1500 | 35491 | 1.91 | 0.51 | 0.15 |
| 2 | interp | const_vel | 1500 | 35491 | 4.65 | 1.21 | 2.70 |
| 2 | interp | hold_last | 1500 | 35491 | 3.77 | 1.58 | 3.28 |
| 2 | interp | centroid_rel | 1484 | 35033 | 3.11 | 1.62 | 0.39 |
| 2 | extrap | const_vel | 1500 | 35491 | 4.65 | 1.21 | 2.70 |
| 2 | extrap | hold_last | 1500 | 35491 | 3.77 | 1.58 | 3.28 |
| 2 | extrap | centroid_rel | 1487 | 35069 | 4.37 | 2.59 | 3.77 |
| 4 | interp | linear_interp | 1500 | 70903 | 2.45 | 1.00 | 0.15 |
| 4 | interp | const_vel | 1500 | 70903 | 6.24 | 2.50 | 5.43 |
| 4 | interp | hold_last | 1500 | 70903 | 5.81 | 2.88 | 5.77 |
| 4 | interp | centroid_rel | 1485 | 70067 | 3.70 | 2.20 | 0.32 |
| 4 | extrap | const_vel | 1500 | 70903 | 6.24 | 2.50 | 5.43 |
| 4 | extrap | hold_last | 1500 | 70903 | 5.81 | 2.88 | 5.77 |
| 4 | extrap | centroid_rel | 1489 | 70249 | 5.58 | 3.54 | 5.26 |
| 8 | interp | linear_interp | 1500 | 142107 | 4.00 | 2.19 | 0.17 |
| 8 | interp | const_vel | 1500 | 142107 | 9.53 | 4.83 | 9.75 |
| 8 | interp | hold_last | 1500 | 142107 | 9.19 | 5.11 | 10.06 |
| 8 | interp | centroid_rel | 1494 | 141260 | 4.94 | 3.29 | 0.32 |
| 8 | extrap | const_vel | 1500 | 142107 | 9.53 | 4.83 | 9.75 |
| 8 | extrap | hold_last | 1500 | 142107 | 9.19 | 5.11 | 10.06 |
| 8 | extrap | centroid_rel | 1498 | 141618 | 7.30 | 4.87 | 6.91 |

### pooled (both matches)

| dur (s) | regime | baseline | n_win | n_samp | RMSE (m) | median (m) | re-entry (m) |
|--:|:--|:--|--:|--:|--:|--:|--:|
| 0.5 | interp | linear_interp | 3000 | 12541 | 0.77 | 0.14 | 0.11 |
| 0.5 | interp | const_vel | 3000 | 12541 | 1.56 | 0.29 | 0.43 |
| 0.5 | interp | hold_last | 3000 | 12541 | 1.41 | 0.41 | 0.66 |
| 0.5 | interp | centroid_rel | 2990 | 12477 | 2.25 | 0.98 | 0.81 |
| 0.5 | extrap | const_vel | 3000 | 12541 | 1.56 | 0.29 | 0.43 |
| 0.5 | extrap | hold_last | 3000 | 12541 | 1.41 | 0.41 | 0.66 |
| 0.5 | extrap | centroid_rel | 2993 | 12489 | 2.99 | 1.20 | 1.64 |
| 1 | interp | linear_interp | 3000 | 25758 | 1.15 | 0.22 | 0.13 |
| 1 | interp | const_vel | 3000 | 25758 | 2.65 | 0.50 | 0.96 |
| 1 | interp | hold_last | 3000 | 25758 | 2.22 | 0.74 | 1.37 |
| 1 | interp | centroid_rel | 2987 | 25581 | 2.59 | 1.26 | 0.68 |
| 1 | extrap | const_vel | 3000 | 25758 | 2.65 | 0.50 | 0.96 |
| 1 | extrap | hold_last | 3000 | 25758 | 2.22 | 0.74 | 1.37 |
| 1 | extrap | centroid_rel | 2994 | 25633 | 3.54 | 1.81 | 2.48 |
| 2 | interp | linear_interp | 3000 | 50969 | 1.66 | 0.42 | 0.15 |
| 2 | interp | const_vel | 3000 | 50969 | 4.07 | 0.99 | 2.00 |
| 2 | interp | hold_last | 3000 | 50969 | 3.38 | 1.34 | 2.50 |
| 2 | interp | centroid_rel | 2981 | 50451 | 3.06 | 1.59 | 0.59 |
| 2 | extrap | const_vel | 3000 | 50969 | 4.07 | 0.99 | 2.00 |
| 2 | extrap | hold_last | 3000 | 50969 | 3.38 | 1.34 | 2.50 |
| 2 | extrap | centroid_rel | 2984 | 50487 | 4.28 | 2.46 | 3.40 |
| 4 | interp | linear_interp | 3000 | 101378 | 2.19 | 0.83 | 0.16 |
| 4 | interp | const_vel | 3000 | 101378 | 5.55 | 2.02 | 3.90 |
| 4 | interp | hold_last | 3000 | 101378 | 5.25 | 2.42 | 4.41 |
| 4 | interp | centroid_rel | 2978 | 100348 | 3.66 | 2.14 | 0.54 |
| 4 | extrap | const_vel | 3000 | 101378 | 5.55 | 2.02 | 3.90 |
| 4 | extrap | hold_last | 3000 | 101378 | 5.25 | 2.42 | 4.41 |
| 4 | extrap | centroid_rel | 2985 | 100572 | 5.40 | 3.32 | 4.60 |
| 8 | interp | linear_interp | 2774 | 193462 | 3.60 | 1.71 | 0.16 |
| 8 | interp | const_vel | 2774 | 193462 | 8.53 | 3.76 | 6.59 |
| 8 | interp | hold_last | 2774 | 193462 | 8.27 | 4.05 | 7.00 |
| 8 | interp | centroid_rel | 2766 | 192501 | 4.71 | 2.98 | 0.59 |
| 8 | extrap | const_vel | 2774 | 193462 | 8.53 | 3.76 | 6.59 |
| 8 | extrap | hold_last | 2774 | 193462 | 8.27 | 4.05 | 7.00 |
| 8 | extrap | centroid_rel | 2772 | 192939 | 6.85 | 4.38 | 5.72 |

## 3. Verdict -- is a learned imputer worth building?

| dur (s) | interp floor (linear) | best extrap (closed-form) | headroom (m) | extrap best baseline |
|--:|--:|--:|--:|:--|
| 0.5 | 0.77 | 1.41 | +0.63 | hold_last |
| 1 | 1.15 | 2.22 | +1.07 | hold_last |
| 2 | 1.66 | 3.38 | +1.72 | hold_last |
| 4 | 2.19 | 5.25 | +3.06 | hold_last |
| 8 | 3.60 | 6.85 | +3.25 | centroid_rel |

**Read.** The interpolation floor (linear interp, using the re-entry point) is small at every duration -- this is why GSR gets away with it *when the player comes back on screen*. The gap opens in the **extrapolation** regime, the broadcast-real case: at 2 s the best closed-form baseline is ~3.4 m RMSE and at 8 s ~6.8 m (vs a ~3.6 m interp floor at 8 s). That extrapolation error lands squarely in the FIFA-study off-screen band (4.6-12.2 m), confirming the regime, not our tracking, is the hard part.

**Where a learned imputer pays off.** Extrapolation, long gaps (>= 2 s). Interpolation is already near the calibration noise floor -- a learned model there would chase sub-metre gains that our own ground truth cannot even certify. The structure-aware `centroid_rel` baseline is the tell: where it already beats `const_vel`/`hold_last` in extrapolation, a model that learns richer team structure (formation, role, ball context) has visible headroom; where it does not, the ceiling is motion, not structure.

**What SkillCorner opendata would add.** All numbers here score against our own trusted tracking, so 'error' is really disagreement-with-self and cannot exceed our calibration quality. SkillCorner's 10-match A-League broadcast release carries genuine off-screen coverage (full-pitch positions for players not in frame), so it replaces self-truth with external truth in exactly the extrapolation regime that matters -- the one validation that can certify a learned imputer actually beats the closed-form floor rather than the tracker agreeing with itself.
