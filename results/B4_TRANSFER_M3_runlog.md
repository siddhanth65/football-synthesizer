# B4 M3 transfer -- verbatim run logs

Machine-written console output for every run behind `results/B4_TRANSFER_M3.md` (the hand-written
analysis, which is NOT overwritten by a re-run). The sweep was split into several invocations so a
kill could not lose it again; the fitted heads are cached (joblib, deterministic, `random_state=0`),
so splitting changed no number -- `rect_base` reproduces the gate run exactly in every run that
includes it.

Reproduce:

```
python -m tools.imputation_b4_transfer --geoms rect_base,soft_edge,aspect_25 --cache-dir CACHE
python -m tools.imputation_b4_transfer --geoms aspect_15,lag_1s,lag_2s      --cache-dir CACHE
python -m tools.imputation_b4_transfer --geoms ellipse,trapezoid_sc,sc_realistic --cache-dir CACHE
python -m tools.imputation_b4_transfer --mitigate --geoms rect_base,aspect_15 --cache-dir CACHE
python -m tools.imputation_b4_transfer --mitigate --geoms trapezoid_sc,sc_realistic,aspect_25     --cache-dir CACHE
python -m tools.imputation_b4_external --source skillcorner --cache CACHE
python -m tools.imputation_b4_external --source ours --match brighton_manutd  --chunks 3 --cache CACHE
python -m tools.imputation_b4_external --source ours --match manutd_liverpool --chunks 2 --cache CACHE
```


---

## Geometry sweep A: rect_base (control), soft_edge, aspect_25

```
====================================================================================================
B4 M3 -- TRANSFER: the FROZEN v1 under censoring geometries it never trained on
====================================================================================================
loaded both Metrica games in 16 s; fps=25.0
training geometry: half-width 16.902 m (11.80 visible/frame) -- identical to the frozen protocol's tune_window
frozen anchor: tau=4.75s, B7 weights [0.95, 0.75, 0.6, 0.5, 0.4, 0.5] (TRAIN only)
v1: all 10 heads ready in 860 s -- all five quantiles are needed because predict_heads SORTS across them to remove crossing, so the point estimate is not the raw 0.50 head

--- rect_base: TRAINING geometry: hard rectangle, full pitch height (control)
    scale=16.902 tuned on TRAIN -> 11.80 visible/frame; holdout n=644792
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  51113 |   0.66 |   0.68 |   +0.02 [ -0.00, +0.05] |   +2.7%
1-3s      |  82286 |   2.09 |   1.90 |   -0.19 [ -0.25, -0.13] |   -8.9%
3-5s      |  62854 |   4.37 |   3.81 |   -0.56 [ -0.72, -0.41] |  -12.9%
5-10s     | 107320 |   7.55 |   6.07 |   -1.48 [ -2.01, -0.98] |  -19.6%
10-30s    | 190552 |  14.91 |   9.74 |   -5.17 [ -6.80, -3.35] |  -34.7%
30s+      | 150667 |  15.16 |  11.24 |   -3.92 [ -4.60, -3.08] |  -25.8%
ALL       | 644792 |  11.46 |   8.10 |   -3.36 [ -4.27, -2.53] |  -29.3%
    inside the impossible visible region: truth 0.0% | last-seen 22.5% | B7 7.3% | v1 4.0%

--- soft_edge: rectangle with a feathered edge (per-slot N(0, 4 m), 10 s blocks)
    scale=17.231 tuned on TRAIN -> 11.80 visible/frame; holdout n=659068
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  55740 |   0.60 |   0.62 |   +0.02 [ +0.00, +0.05] |   +4.0%
1-3s      |  88986 |   2.04 |   1.87 |   -0.17 [ -0.22, -0.12] |   -8.2%
3-5s      |  67758 |   4.32 |   3.81 |   -0.51 [ -0.69, -0.35] |  -11.8%
5-10s     | 117802 |   7.54 |   6.22 |   -1.32 [ -1.86, -0.84] |  -17.5%
10-30s    | 185922 |  14.21 |   9.66 |   -4.55 [ -6.16, -2.99] |  -32.0%
30s+      | 142860 |  14.50 |  11.36 |   -3.14 [ -3.88, -2.37] |  -21.6%
ALL       | 659068 |  10.73 |   7.95 |   -2.78 [ -3.69, -2.02] |  -25.9%
    inside the impossible visible region: truth 10.3% | last-seen 22.3% | B7 14.9% | v1 11.8%

--- aspect_25: rectangle, half-height 25 m (squarer window, wider in x)
    scale=21.468 tuned on TRAIN -> 11.80 visible/frame; holdout n=669016
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  49810 |   0.45 |   0.52 |   +0.07 [ +0.04, +0.10] |  +14.6%
1-3s      |  83821 |   2.00 |   1.90 |   -0.10 [ -0.16, -0.02] |   -4.9%
3-5s      |  65305 |   4.21 |   3.87 |   -0.34 [ -0.50, -0.19] |   -8.0%
5-10s     | 118017 |   7.30 |   6.35 |   -0.95 [ -1.37, -0.58] |  -13.0%
10-30s    | 204934 |  12.95 |   9.98 |   -2.98 [ -4.26, -1.78] |  -23.0%
30s+      | 147129 |  14.97 |  12.50 |   -2.47 [ -3.16, -1.64] |  -16.5%
ALL       | 669016 |  10.60 |   8.60 |   -2.00 [ -2.69, -1.38] |  -18.9%
    inside the impossible visible region: truth 0.0% | last-seen 15.9% | B7 5.8% | v1 3.9%

====================================================================================================
SUMMARY -- margin = RMSE(v1) - RMSE(B7 anchor), metres. Negative = v1 better.
Absolute RMSE is NOT comparable across geometries (different hidden sets); the MARGIN is.
====================================================================================================
geometry     | vis/fr |      n |    B7 |    v1 | margin [95% CI]       | kept | v1 inside
----------------------------------------------------------------------------------------------------
rect_base    |  11.80 | 644792 | 11.46 |  8.10 |  -3.36 [-4.27,-2.53] |  100% |   4.0%
soft_edge    |  11.80 | 659068 | 10.73 |  7.95 |  -2.78 [-3.69,-2.02] |   83% |  11.8%
aspect_25    |  11.80 | 669016 | 10.60 |  8.60 |  -2.00 [-2.69,-1.38] |   60% |   3.9%
'kept' = this geometry's margin as a percentage of the training geometry's margin.
```


---

## Geometry sweep B: aspect_15, lag_1s, lag_2s

```
====================================================================================================
B4 M3 -- TRANSFER: the FROZEN v1 under censoring geometries it never trained on
====================================================================================================
loaded both Metrica games in 11 s; fps=25.0
training geometry: half-width 16.902 m (11.80 visible/frame) -- identical to the frozen protocol's tune_window
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
frozen anchor: tau=4.75s, B7 weights [0.95, 0.75, 0.6, 0.5, 0.4, 0.5] (TRAIN only)
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
v1: all 10 heads ready in 0 s -- all five quantiles are needed because predict_heads SORTS across them to remove crossing, so the point estimate is not the raw 0.50 head

--- aspect_15: rectangle, half-height 15 m (tight zoom, much wider in x)
    scale=54.967 tuned on TRAIN -> 11.80 visible/frame; holdout n=679787
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  53183 |   0.49 |   0.59 |   +0.10 [ +0.07, +0.13] |  +20.0%
1-3s      |  90671 |   2.18 |   2.30 |   +0.11 [ +0.03, +0.22] |   +5.1%
3-5s      |  72448 |   4.54 |   4.67 |   +0.13 [ -0.07, +0.42] |   +2.9%
5-10s     | 133073 |   7.67 |   7.85 |   +0.17 [ -0.29, +0.81] |   +2.3%
10-30s    | 221917 |  12.47 |  13.33 |   +0.86 [ -0.36, +2.00] |   +6.9%
30s+      | 108495 |  18.87 |  16.53 |   -2.33 [ -4.94, +0.45] |  -12.4%
ALL       | 679787 |  11.04 |  10.81 |   -0.24 [ -1.28, +0.85] |   -2.2%
    inside the impossible visible region: truth 0.0% | last-seen 8.4% | B7 6.5% | v1 6.2%

--- lag_1s: rectangle, camera reacts 1.0 s late
    scale=16.786 tuned on TRAIN -> 11.80 visible/frame; holdout n=650175
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  50256 |   0.88 |   0.89 |   +0.01 [ -0.01, +0.05] |   +1.1%
1-3s      |  81349 |   2.11 |   1.93 |   -0.17 [ -0.24, -0.11] |   -8.3%
3-5s      |  62495 |   4.25 |   3.70 |   -0.55 [ -0.71, -0.39] |  -12.9%
5-10s     | 107415 |   7.35 |   5.79 |   -1.56 [ -2.12, -1.06] |  -21.2%
10-30s    | 193947 |  15.04 |   9.60 |   -5.44 [ -7.16, -3.55] |  -36.2%
30s+      | 154713 |  15.27 |  10.66 |   -4.61 [ -5.33, -3.71] |  -30.2%
ALL       | 650175 |  11.59 |   7.87 |   -3.72 [ -4.67, -2.76] |  -32.1%
    inside the impossible visible region: truth 0.0% | last-seen 24.5% | B7 8.5% | v1 3.7%

--- lag_2s: rectangle, camera reacts 2.0 s late
    scale=17.047 tuned on TRAIN -> 11.80 visible/frame; holdout n=648638
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  50480 |   1.18 |   1.18 |   +0.00 [ -0.02, +0.04] |   +0.0%
1-3s      |  82569 |   2.47 |   2.29 |   -0.18 [ -0.25, -0.11] |   -7.3%
3-5s      |  61462 |   4.52 |   3.93 |   -0.59 [ -0.76, -0.39] |  -13.0%
5-10s     | 106611 |   7.73 |   6.15 |   -1.57 [ -2.09, -1.07] |  -20.4%
10-30s    | 194278 |  15.06 |   9.61 |   -5.45 [ -7.19, -3.43] |  -36.2%
30s+      | 153238 |  15.00 |  10.53 |   -4.47 [ -5.24, -3.53] |  -29.8%
ALL       | 648638 |  11.56 |   7.89 |   -3.67 [ -4.61, -2.64] |  -31.7%
    inside the impossible visible region: truth 0.0% | last-seen 27.3% | B7 9.5% | v1 4.0%

====================================================================================================
SUMMARY -- margin = RMSE(v1) - RMSE(B7 anchor), metres. Negative = v1 better.
Absolute RMSE is NOT comparable across geometries (different hidden sets); the MARGIN is.
====================================================================================================
geometry     | vis/fr |      n |    B7 |    v1 | margin [95% CI]       | kept | v1 inside
----------------------------------------------------------------------------------------------------
aspect_15    |  11.80 | 679787 | 11.04 | 10.81 |  -0.24 [-1.28,+0.85] |      |   6.2%
lag_1s       |  11.80 | 650175 | 11.59 |  7.87 |  -3.72 [-4.67,-2.76] |      |   3.7%
lag_2s       |  11.80 | 648638 | 11.56 |  7.89 |  -3.67 [-4.61,-2.64] |      |   4.0%
'kept' = this geometry's margin as a percentage of the training geometry's margin.
```


---

## Geometry sweep C: ellipse, trapezoid_sc, sc_realistic

```
====================================================================================================
B4 M3 -- TRANSFER: the FROZEN v1 under censoring geometries it never trained on
====================================================================================================
loaded both Metrica games in 11 s; fps=25.0
training geometry: half-width 16.902 m (11.80 visible/frame) -- identical to the frozen protocol's tune_window
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
frozen anchor: tau=4.75s, B7 weights [0.95, 0.75, 0.6, 0.5, 0.4, 0.5] (TRAIN only)
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
v1: all 10 heads ready in 1 s -- all five quantiles are needed because predict_heads SORTS across them to remove crossing, so the point estimate is not the raw 0.50 head

--- ellipse: elliptical / vignette window
    scale=20.398 tuned on TRAIN -> 11.80 visible/frame; holdout n=648186
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  45640 |   0.52 |   0.57 |   +0.04 [ +0.01, +0.07] |   +8.1%
1-3s      |  77662 |   1.99 |   1.87 |   -0.13 [ -0.20, -0.06] |   -6.4%
3-5s      |  63218 |   4.16 |   3.78 |   -0.38 [ -0.55, -0.21] |   -9.2%
5-10s     | 113008 |   7.29 |   6.09 |   -1.20 [ -1.69, -0.74] |  -16.5%
10-30s    | 196743 |  13.65 |   9.86 |   -3.79 [ -5.20, -2.42] |  -27.8%
30s+      | 151915 |  15.07 |  12.01 |   -3.06 [ -3.69, -2.37] |  -20.3%
ALL       | 648186 |  11.01 |   8.46 |   -2.55 [ -3.31, -1.86] |  -23.1%
    inside the impossible visible region: truth 0.0% | last-seen 19.6% | B7 7.0% | v1 4.5%

--- trapezoid_sc: SkillCorner-measured broadcast footprint (widens with distance)
    scale=0.910 tuned on TRAIN -> 11.80 visible/frame; holdout n=668671
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  48110 |   0.68 |   0.68 |   +0.00 [ -0.01, +0.03] |   +0.6%
1-3s      |  79704 |   2.40 |   2.13 |   -0.27 [ -0.34, -0.21] |  -11.2%
3-5s      |  63183 |   4.75 |   4.02 |   -0.73 [ -0.95, -0.56] |  -15.4%
5-10s     | 110799 |   7.84 |   6.08 |   -1.76 [ -2.32, -1.23] |  -22.5%
10-30s    | 205775 |  15.46 |   9.93 |   -5.53 [ -7.17, -3.57] |  -35.8%
30s+      | 161100 |  15.69 |  11.28 |   -4.41 [ -5.13, -3.56] |  -28.1%
ALL       | 668671 |  12.08 |   8.32 |   -3.76 [ -4.69, -2.76] |  -31.1%
    inside the impossible visible region: truth 0.0% | last-seen 27.0% | B7 8.8% | v1 5.9%

--- sc_realistic: SkillCorner footprint + 0.5 s camera lag + ~2 m feathered edge
    scale=0.913 tuned on TRAIN -> 11.80 visible/frame; holdout n=670763
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  50097 |   0.82 |   0.82 |   +0.00 [ -0.02, +0.05] |   +0.6%
1-3s      |  80815 |   2.36 |   2.14 |   -0.23 [ -0.30, -0.16] |   -9.7%
3-5s      |  63564 |   4.68 |   4.02 |   -0.66 [ -0.86, -0.47] |  -14.1%
5-10s     | 111973 |   7.74 |   6.10 |   -1.64 [ -2.18, -1.10] |  -21.2%
10-30s    | 208747 |  15.16 |   9.76 |   -5.40 [ -7.02, -3.51] |  -35.6%
30s+      | 155567 |  14.85 |  10.93 |   -3.93 [ -4.75, -2.95] |  -26.4%
ALL       | 670763 |  11.64 |   8.10 |   -3.54 [ -4.48, -2.53] |  -30.4%
    inside the impossible visible region: truth 5.1% | last-seen 26.3% | B7 13.2% | v1 9.4%

====================================================================================================
SUMMARY -- margin = RMSE(v1) - RMSE(B7 anchor), metres. Negative = v1 better.
Absolute RMSE is NOT comparable across geometries (different hidden sets); the MARGIN is.
====================================================================================================
geometry     | vis/fr |      n |    B7 |    v1 | margin [95% CI]       | kept | v1 inside
----------------------------------------------------------------------------------------------------
ellipse      |  11.80 | 648186 | 11.01 |  8.46 |  -2.55 [-3.31,-1.86] |      |   4.5%
trapezoid_sc |  11.80 | 668671 | 12.08 |  8.32 |  -3.76 [-4.69,-2.76] |      |   5.9%
sc_realistic |  11.80 | 670763 | 11.64 |  8.10 |  -3.54 [-4.48,-2.53] |      |   9.4%
'kept' = this geometry's margin as a percentage of the training geometry's margin.
```


---

## Mitigation v1.1: fit + rect_base, aspect_15

```
====================================================================================================
B4 M3 -- TRANSFER: the FROZEN v1 under censoring geometries it never trained on
====================================================================================================
loaded both Metrica games in 11 s; fps=25.0
training geometry: half-width 16.902 m (11.80 visible/frame) -- identical to the frozen protocol's tune_window
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
frozen anchor: tau=4.75s, B7 weights [0.95, 0.75, 0.6, 0.5, 0.4, 0.5] (TRAIN only)
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
v1: all 10 heads ready in 0 s -- all five quantiles are needed because predict_heads SORTS across them to remove crossing, so the point estimate is not the raw 0.50 head

MITIGATION v1.1 -- train-time camera-shape randomisation (a NEW model; v1's gate
record stands untouched). TRAIN censorings pooled: ['rect_base', 'aspect_25', 'aspect_15', 'lag_1s']
  rect_base    scale=16.90 visible=11.80 n=729534 (stride 2)
  aspect_25    scale=21.47 visible=11.80 n=731077 (stride 2)
  aspect_15    scale=54.97 visible=11.80 n=736083 (stride 2)
  lag_1s       scale=16.79 visible=11.80 n=729053 (stride 2)
  fitted v1.1 (10 heads) on n=2925747 in 1155 s

--- rect_base: TRAINING geometry: hard rectangle, full pitch height (control)
    scale=16.902 tuned on TRAIN -> 11.80 visible/frame; holdout n=644792
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  51113 |   0.66 |   0.68 |   +0.02 [ -0.00, +0.05] |   +2.7%
1-3s      |  82286 |   2.09 |   1.90 |   -0.19 [ -0.25, -0.13] |   -8.9%
3-5s      |  62854 |   4.37 |   3.81 |   -0.56 [ -0.72, -0.41] |  -12.9%
5-10s     | 107320 |   7.55 |   6.07 |   -1.48 [ -2.01, -0.98] |  -19.6%
10-30s    | 190552 |  14.91 |   9.74 |   -5.17 [ -6.80, -3.35] |  -34.7%
30s+      | 150667 |  15.16 |  11.24 |   -3.92 [ -4.60, -3.08] |  -25.8%
ALL       | 644792 |  11.46 |   8.10 |   -3.36 [ -4.27, -2.53] |  -29.3%
    inside the impossible visible region: truth 0.0% | last-seen 22.5% | B7 7.3% | v1 4.0%
    v1.1 (camera-randomised training): 7.98 m, margin -3.48 [-4.39,-2.65], inside 3.8%  [in-family]

--- aspect_15: rectangle, half-height 15 m (tight zoom, much wider in x)
    scale=54.967 tuned on TRAIN -> 11.80 visible/frame; holdout n=679787
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  53183 |   0.49 |   0.59 |   +0.10 [ +0.07, +0.13] |  +20.0%
1-3s      |  90671 |   2.18 |   2.30 |   +0.11 [ +0.03, +0.22] |   +5.1%
3-5s      |  72448 |   4.54 |   4.67 |   +0.13 [ -0.07, +0.42] |   +2.9%
5-10s     | 133073 |   7.67 |   7.85 |   +0.17 [ -0.29, +0.81] |   +2.3%
10-30s    | 221917 |  12.47 |  13.33 |   +0.86 [ -0.36, +2.00] |   +6.9%
30s+      | 108495 |  18.87 |  16.53 |   -2.33 [ -4.94, +0.45] |  -12.4%
ALL       | 679787 |  11.04 |  10.81 |   -0.24 [ -1.28, +0.85] |   -2.2%
    inside the impossible visible region: truth 0.0% | last-seen 8.4% | B7 6.5% | v1 6.2%
    v1.1 (camera-randomised training): 8.68 m, margin -2.36 [-3.18,-1.57], inside 4.0%  [in-family]

====================================================================================================
SUMMARY -- margin = RMSE(v1) - RMSE(B7 anchor), metres. Negative = v1 better.
Absolute RMSE is NOT comparable across geometries (different hidden sets); the MARGIN is.
====================================================================================================
geometry     | vis/fr |      n |    B7 |    v1 | margin [95% CI]       | kept | v1 inside
----------------------------------------------------------------------------------------------------
rect_base    |  11.80 | 644792 | 11.46 |  8.10 |  -3.36 [-4.27,-2.53] |  100% |   4.0%
aspect_15    |  11.80 | 679787 | 11.04 | 10.81 |  -0.24 [-1.28,+0.85] |    7% |   6.2%
'kept' = this geometry's margin as a percentage of the training geometry's margin.

v1.1 (camera-shape randomised training) vs v1, same geometries:
geometry     | family |    v1 |  v1.1 | v1 margin | v1.1 margin | v1.1 inside
------------------------------------------------------------------------------------
rect_base    | in     |  8.10 |  7.98 |     -3.36 |       -3.48 |        3.8%
aspect_15    | in     | 10.81 |  8.68 |     -0.24 |       -2.36 |        4.0%
```


---

## Mitigation v1.1: trapezoid_sc, sc_realistic, aspect_25

```
====================================================================================================
B4 M3 -- TRANSFER: the FROZEN v1 under censoring geometries it never trained on
====================================================================================================
loaded both Metrica games in 17 s; fps=25.0
training geometry: half-width 16.902 m (11.80 visible/frame) -- identical to the frozen protocol's tune_window
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
frozen anchor: tau=4.75s, B7 weights [0.95, 0.75, 0.6, 0.5, 0.4, 0.5] (TRAIN only)
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
v1: all 10 heads ready in 1 s -- all five quantiles are needed because predict_heads SORTS across them to remove crossing, so the point estimate is not the raw 0.50 head
loaded v11_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v11_heads.joblib

--- trapezoid_sc: SkillCorner-measured broadcast footprint (widens with distance)
    scale=0.910 tuned on TRAIN -> 11.80 visible/frame; holdout n=668671
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  48110 |   0.68 |   0.68 |   +0.00 [ -0.01, +0.03] |   +0.6%
1-3s      |  79704 |   2.40 |   2.13 |   -0.27 [ -0.34, -0.21] |  -11.2%
3-5s      |  63183 |   4.75 |   4.02 |   -0.73 [ -0.95, -0.56] |  -15.4%
5-10s     | 110799 |   7.84 |   6.08 |   -1.76 [ -2.32, -1.23] |  -22.5%
10-30s    | 205775 |  15.46 |   9.93 |   -5.53 [ -7.17, -3.57] |  -35.8%
30s+      | 161100 |  15.69 |  11.28 |   -4.41 [ -5.13, -3.56] |  -28.1%
ALL       | 668671 |  12.08 |   8.32 |   -3.76 [ -4.69, -2.76] |  -31.1%
    inside the impossible visible region: truth 0.0% | last-seen 27.0% | B7 8.8% | v1 5.9%
    v1.1 (camera-randomised training): 8.69 m, margin -3.38 [-4.45,-2.39], inside 5.3%  [OUT of-family]

--- sc_realistic: SkillCorner footprint + 0.5 s camera lag + ~2 m feathered edge
    scale=0.913 tuned on TRAIN -> 11.80 visible/frame; holdout n=670763
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  50097 |   0.82 |   0.82 |   +0.00 [ -0.02, +0.05] |   +0.6%
1-3s      |  80815 |   2.36 |   2.14 |   -0.23 [ -0.30, -0.16] |   -9.7%
3-5s      |  63564 |   4.68 |   4.02 |   -0.66 [ -0.86, -0.47] |  -14.1%
5-10s     | 111973 |   7.74 |   6.10 |   -1.64 [ -2.18, -1.10] |  -21.2%
10-30s    | 208747 |  15.16 |   9.76 |   -5.40 [ -7.02, -3.51] |  -35.6%
30s+      | 155567 |  14.85 |  10.93 |   -3.93 [ -4.75, -2.95] |  -26.4%
ALL       | 670763 |  11.64 |   8.10 |   -3.54 [ -4.48, -2.53] |  -30.4%
    inside the impossible visible region: truth 5.1% | last-seen 26.3% | B7 13.2% | v1 9.4%
    v1.1 (camera-randomised training): 8.30 m, margin -3.34 [-4.33,-2.37], inside 9.1%  [OUT of-family]

--- aspect_25: rectangle, half-height 25 m (squarer window, wider in x)
    scale=21.468 tuned on TRAIN -> 11.80 visible/frame; holdout n=669016
horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7
----------------------------------------------------------------------------
0-1s      |  49810 |   0.45 |   0.52 |   +0.07 [ +0.04, +0.10] |  +14.6%
1-3s      |  83821 |   2.00 |   1.90 |   -0.10 [ -0.16, -0.02] |   -4.9%
3-5s      |  65305 |   4.21 |   3.87 |   -0.34 [ -0.50, -0.19] |   -8.0%
5-10s     | 118017 |   7.30 |   6.35 |   -0.95 [ -1.37, -0.58] |  -13.0%
10-30s    | 204934 |  12.95 |   9.98 |   -2.98 [ -4.26, -1.78] |  -23.0%
30s+      | 147129 |  14.97 |  12.50 |   -2.47 [ -3.16, -1.64] |  -16.5%
ALL       | 669016 |  10.60 |   8.60 |   -2.00 [ -2.69, -1.38] |  -18.9%
    inside the impossible visible region: truth 0.0% | last-seen 15.9% | B7 5.8% | v1 3.9%
    v1.1 (camera-randomised training): 7.85 m, margin -2.75 [-3.35,-2.20], inside 3.9%  [in-family]

====================================================================================================
SUMMARY -- margin = RMSE(v1) - RMSE(B7 anchor), metres. Negative = v1 better.
Absolute RMSE is NOT comparable across geometries (different hidden sets); the MARGIN is.
====================================================================================================
geometry     | vis/fr |      n |    B7 |    v1 | margin [95% CI]       | kept | v1 inside
----------------------------------------------------------------------------------------------------
trapezoid_sc |  11.80 | 668671 | 12.08 |  8.32 |  -3.76 [-4.69,-2.76] |      |   5.9%
sc_realistic |  11.80 | 670763 | 11.64 |  8.10 |  -3.54 [-4.48,-2.53] |      |   9.4%
aspect_25    |  11.80 | 669016 | 10.60 |  8.60 |  -2.00 [-2.69,-1.38] |      |   3.9%
'kept' = this geometry's margin as a percentage of the training geometry's margin.

v1.1 (camera-shape randomised training) vs v1, same geometries:
geometry     | family |    v1 |  v1.1 | v1 margin | v1.1 margin | v1.1 inside
------------------------------------------------------------------------------------
trapezoid_sc | OUT    |  8.32 |  8.69 |     -3.76 |       -3.38 |        5.3%
sc_realistic | OUT    |  8.10 |  8.30 |     -3.54 |       -3.34 |        9.1%
aspect_25    | in     |  8.60 |  7.85 |     -2.00 |       -2.75 |        3.9%
```


---

## SkillCorner opendata check

```
====================================================================================================
B4 M3 -- FROZEN v1 on external tracking: skillcorner
====================================================================================================
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
frozen v1 ready in 0 s

SkillCorner 1886347 (A-League, 10 fps): 59061 frames x 36 slots, fps=10.0
  visible/frame mean 13.04 of 22.0 active (over 43458 live frames); hidden samples scored 386763
  horizon mix: 0-1s=10% 1-3s=21% 3-5s=13% 5-10s=17% 10-30s=20% 30s+=19%
  vs the reference positions (NOT ground truth -- see module docstring):
  horizon   |      n |   B7   |   v1   |  margin  [95% block CI]
  0-1s      |  40426 |   0.42 |   1.06 |   +0.64 [ +0.57, +0.70]
  1-3s      |  79330 |   1.90 |   2.37 |   +0.48 [ +0.34, +0.61]
  3-5s      |  50553 |   4.18 |   4.15 |   -0.03 [ -0.24, +0.19]
  5-10s     |  65465 |   7.31 |   6.77 |   -0.53 [ -1.01, -0.15]
  10-30s    |  78368 |  11.61 |  10.17 |   -1.44 [ -1.91, -0.94]
  30s+      |  72621 |  13.13 |  10.84 |   -2.30 [ -2.99, -1.44]
  ALL       | 386763 |   8.47 |   7.37 |   -1.10 [ -1.43, -0.76]
  method    | off-pitch | inside visible region | implied speed m/s p50 / p90 / max
  reference |      0.4% |                  5.2% |   1.17 /   3.09 /   13.01
  last-seen |      0.4% |                 11.0% |   0.00 /   0.00 /    0.00
  B7        |      1.0% |                 11.4% |   0.90 /   2.60 /   10.63
  v1        |      0.3% |                  7.5% |   1.11 /   3.03 /   36.11
  team x-span at the query frames: visible-only 21.1 m -> completed with v1 51.5 m (> 90 m in 0.0% of team-frames)
```


---

## Own-footage smell test: brighton_manutd

```
====================================================================================================
B4 M3 -- FROZEN v1 on external tracking: ours
====================================================================================================
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
frozen v1 ready in 0 s

brighton_manutd (3 chunks, 1351 tracks >= 2s): 8731 frames x 460 slots, fps=5.0
  visible/frame over frames with any sighting: mean 10.06 (p50 10, p90 15) on 2442 of 8731 frames; hidden samples scored 372577
  horizon mix: 0-1s=1% 1-3s=4% 3-5s=4% 5-10s=9% 10-30s=34% 30s+=49%
  method    | off-pitch | inside visible region | implied speed m/s p50 / p90 / max
  last-seen |      0.0% |                 28.6% |   0.00 /   0.00 /    0.00
  B7        |      0.9% |                 44.4% |   0.39 /   1.12 /   12.49
  v1        |      0.2% |                 50.4% |   0.53 /   1.39 /   22.76
  last sighting within 10% of the image edge (a genuine frame exit rather than a tracker drop): 13.7% of samples
  edge-exit subset only: v1 inside the visible box 34.2%, implied speed p50 0.52 m/s
  team x-span at the query frames: visible-only 17.5 m -> completed with v1 37.4 m (> 90 m in 0.0% of team-frames)
```


---

## Own-footage smell test: manutd_liverpool

```
====================================================================================================
B4 M3 -- FROZEN v1 on external tracking: ours
====================================================================================================
loaded train_meta from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\train_meta.joblib
loaded v1_heads from cache C:\Users\SIDDH_~1\AppData\Local\Temp\claude\c--Users-siddh-ygv5bws-football-synthesizer\9b756e0f-4677-4dfe-ab15-755490eabda1\scratchpad\b4cache\v1_heads.joblib
frozen v1 ready in 0 s

manutd_liverpool (2 chunks, 719 tracks >= 2s): 5849 frames x 371 slots, fps=5.0
  visible/frame over frames with any sighting: mean 9.81 (p50 9, p90 17) on 1334 of 5849 frames; hidden samples scored 200086
  horizon mix: 0-1s=1% 1-3s=4% 3-5s=4% 5-10s=9% 10-30s=34% 30s+=49%
  method    | off-pitch | inside visible region | implied speed m/s p50 / p90 / max
  last-seen |      0.0% |                 27.9% |   0.00 /   0.00 /    0.00
  B7        |      1.3% |                 44.5% |   0.35 /   1.11 /   11.55
  v1        |      0.3% |                 47.4% |   0.55 /   1.43 /   15.14
  last sighting within 10% of the image edge (a genuine frame exit rather than a tracker drop): 12.2% of samples
  edge-exit subset only: v1 inside the visible box 36.8%, implied speed p50 0.49 m/s
  team x-span at the query frames: visible-only 17.1 m -> completed with v1 36.5 m (> 90 m in 0.0% of team-frames)
```
