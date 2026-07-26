```
================================================================================================
FULL-MATCH 22-PLAYER RECONSTRUCTION (frozen B4 v1, P2 policy) -- nothing is refitted
================================================================================================
match manutd_brighton (Man Utd vs Brighton), 11 chunks, 116469 gated position rows, 41 name-gated fragments
  [oracle] team 30: 11 starters, 5 used subs, 5 withdrawn -> on-pitch cap 11 throughout
  [oracle] team 35: 11 starters, 4 used subs, 4 withdrawn -> on-pitch cap 11 throughout
loaded train_meta from cache data\imputation\cache\train_meta.joblib
loaded v1_heads from cache data\imputation\cache\v1_heads.joblib
frozen v1 ready in 0 s
  h1_chunk_000: 2898 timesteps x 495 slots, 148050 hidden samples, 174 within-track + 1 cross-track events, 48534 estimate rows [48 s]
  h1_chunk_001: 2980 timesteps x 611 slots, 187431 hidden samples, 315 within-track + 0 cross-track events, 54826 estimate rows [60 s]
  h1_chunk_002: 2080 timesteps x 340 slots, 94825 hidden samples, 138 within-track + 3 cross-track events, 33308 estimate rows [30 s]
  h1_chunk_003: 2952 timesteps x 664 slots, 211904 hidden samples, 311 within-track + 1 cross-track events, 58585 estimate rows [67 s]
  h1_chunk_004: 2629 timesteps x 524 slots, 156227 hidden samples, 222 within-track + 1 cross-track events, 46181 estimate rows [50 s]
  h2_chunk_000: 2990 timesteps x 401 slots, 123137 hidden samples, 193 within-track + 1 cross-track events, 41861 estimate rows [39 s]
  h2_chunk_001: 2726 timesteps x 319 slots, 94068 hidden samples, 104 within-track + 0 cross-track events, 35491 estimate rows [30 s]
  h2_chunk_002: 2978 timesteps x 305 slots, 100311 hidden samples, 120 within-track + 0 cross-track events, 36121 estimate rows [33 s]
  h2_chunk_003: 2660 timesteps x 440 slots, 130729 hidden samples, 188 within-track + 0 cross-track events, 40901 estimate rows [42 s]
  h2_chunk_004: 2935 timesteps x 477 slots, 147879 hidden samples, 187 within-track + 0 cross-track events, 46478 estimate rows [48 s]
  h2_chunk_005: 786 timesteps x 152 slots, 37146 hidden samples, 82 within-track + 0 cross-track events, 12539 estimate rows [12 s]

wrote results\reconstruction\manutd_brighton_reconstruction.parquet (454825 rows)

================================================================================================
TASK 1 -- COVERAGE PROFILE
================================================================================================
timesteps on the chunk grid: 24622 (82.1 min); trusted (>= 4 gated players): 9136 (37.1%)

players per team per trusted timestep (share of team-timesteps):
  n  | observed only | with reconstruction
   0 |          0.3% |                0.0%
   1 |          1.7% |                0.2%
   2 |          4.9% |                0.2%
   3 |         10.6% |                0.4%
   4 |         15.9% |                1.5%
   5 |         16.4% |                2.7%
   6 |         16.6% |                3.5%
   7 |         13.0% |                4.7%
   8 |          9.2% |                6.2%
   9 |          5.4% |                7.8%
  10 |          3.1% |                7.6%
  11 |          1.7% |               64.2%
  >11 |          1.1% |                1.1%

mean per team: observed 5.66 -> reconstruction 9.91 of 11
  fraction of team-timesteps with >= 11: observed   2.8%  reconstruction  65.3%
  fraction of team-timesteps with >= 10: observed   5.9%  reconstruction  72.9%
  fraction of team-timesteps with >=  9: observed  11.4%  reconstruction  80.7%
  fraction of team-timesteps with >=  8: observed  20.6%  reconstruction  86.9%

imputed rows by source: {'v1': np.int64(314361), 'anchor': np.int64(36558)}
imputed rows by provenance: {'terminal': np.int64(248494), 'gap': np.int64(102425)} -- 'terminal' means the slot is never seen again, i.e. a dead re-identification fragment that may be a
phantom rather than a genuinely occluded player (results/B4_TRANSFER_M3.md section 3).

================================================================================================
TASK 2 -- RE-APPEARANCE VALIDATION
================================================================================================
census by linking rule and horizon bucket:
              0-1s  1-3s  3-5s  5-10s  10-30s  30s+
link                                               
cross-track      0     0     0      1       1     5
within-track  1269   520   157     84       3     0

observation-noise floor of our own projected positions: sigma = 0.61 m per axis (88133 midpoint triples). Metrica truth has none; the frozen regions were calibrated
against exact truth, so this much error is unmodellable by construction on our footage.

  PRIMARY -- within-track linking (tracker id carried across the gap)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90 | 50+n | 90+n
  0-1s      | 1269 |    2.26 |  0.5 |  2.3 |   2.26 |   2.26 |   2.27 |   17.9 |   51.5 |  0.2 |  0.6 | 71.5 | 76.8
  1-3s      |  520 |    4.03 |  1.9 |  5.9 |   3.93 |   4.19 |   4.03 |   24.2 |   56.9 |  1.1 |  2.4 | 36.9 | 61.3
  3-5s      |  157 |    5.20 |  3.6 |  7.5 |   4.87 |   5.78 |   5.20 |   34.4 |   75.2 |  2.7 |  5.7 | 34.4 | 73.9
  5-10s     |   84 |    6.40 |  5.3 |  9.6 |   5.84 |   6.81 |   6.40 |   36.9 |   86.9 |  4.2 |  8.7 | 39.3 | 85.7
  10-30s    |    3 |    6.70 |  3.1 |  9.3 |   5.70 |   5.95 |   6.70 |   66.7 |  100.0 |  6.0 | 12.8 | 66.7 | 100.0
  30s+      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  ALL       | 2033 |    3.35 |  0.9 |  5.2 |   3.23 |   3.50 |   3.35 |   21.6 |   56.3 |  0.8 |  1.8 | 58.4 | 73.0
  (50+n / 90+n = the SAME frozen regions widened in quadrature by the noise floor -- a
   diagnostic of why they miss, NOT a re-calibration; nothing in the model changed.)

  SECONDARY -- cross-track identity linking (NOT trusted; see report)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90
  0-1s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  1-3s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  3-5s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  5-10s     |    1 |    3.13 |  3.1 |  3.1 |   4.21 |   1.40 |   3.13 |  100.0 |  100.0 |  4.5 |  9.3
  10-30s    |    1 |   17.08 | 17.1 | 17.1 |  15.99 |   7.13 |  17.08 |    0.0 |    0.0 |  6.2 | 13.3
  30s+      |    5 |   10.73 |  8.4 | 15.1 |  10.91 |  20.55 |  10.73 |   40.0 |  100.0 |  8.0 | 17.1
  ALL       |    7 |   11.20 |  8.4 | 16.5 |  11.14 |  17.59 |  11.20 |   42.9 |   85.7 |  7.2 | 15.4
  cross-track gap frames with live tracking: median 22% -- a broadcast outage, not an occlusion

  Metrica holdout (frozen, same emitted policy) for comparison:
  horizon   |       n |   RMSE | PICP50 | PICP90 | |truth-lastseen| p50 / mean
  0-1s      |   51113 |   0.66 |   52.4 |   90.0 |   0.73 /   1.00
  1-3s      |   82286 |   1.90 |   49.9 |   89.8 |   3.08 /   3.68
  3-5s      |   62854 |   3.81 |   48.2 |   90.1 |   6.10 /   7.13
  5-10s     |  107320 |   6.07 |   51.0 |   90.6 |  10.10 /  11.62
  10-30s    |  190552 |   9.74 |   47.8 |   89.4 |  17.16 /  19.27
  30s+      |  150667 |  11.24 |   45.5 |   89.2 |  15.17 /  17.24

  SELECTION: displacement between the last sighting and the measured truth, and the
  Metrica holdout restricted to samples that moved no further (like for like):
  horizon   |    n | p50 disp | mean disp | Metrica p50 | matched n | RMSE | P50 | P90
  0-1s      | 1269 |     0.51 |      1.03 |        0.73 |     45377 | 0.38 |  57 |  92
  1-3s      |  520 |     1.33 |      2.41 |        3.08 |     66300 | 1.62 |  52 |  92
  3-5s      |  157 |     2.61 |      3.82 |        6.10 |     46917 | 3.21 |  53 |  94
  5-10s     |   84 |     3.11 |      4.50 |       10.10 |     46709 | 4.89 |  59 |  95
  10-30s    |    3 |     4.37 |      5.42 |       17.16 |     33253 | 6.29 |  64 |  95

  paired block-bootstrap CI on RMSE(emitted) - RMSE(anchor) and - RMSE(hold):
  0-1s      | vs anchor  +0.00 [ +0.00,  +0.00] | vs hold  +0.00 [ -0.07,  +0.09]
  1-3s      | vs anchor  +0.10 [ +0.01,  +0.19] | vs hold  -0.16 [ -0.53,  +0.20]
  3-5s      | vs anchor  +0.33 [ +0.15,  +0.56] | vs hold  -0.58 [ -1.25,  +0.47]
  5-10s     | vs anchor  +0.56 [ +0.14,  +1.07] | vs hold  -0.41 [ -1.41,  +0.92]
  ALL       | vs anchor  +0.11 [ +0.08,  +0.16] | vs hold  -0.15 [ -0.35,  +0.04]

  physically impossible links (implied speed > 12 m/s, i.e. an id switch or a gross projection failure): 31 of 2033 (1.5%)
  SENSITIVITY (diagnostic only -- excluding them removes the largest errors and so flatters the model): n 2002, RMSE 2.98 m, PICP50 22.0, PICP90 57.1

================================================================================================
TASK 2b -- SELECTION BIAS: WHO COMES BACK?
================================================================================================
track-losses in the match: 13485; re-appear under the same id: 64.9%; survive the liveness guard and become usable events: 15.1%
  gap length of re-appearing losses: p50 1.4 s, p90 10.2 s, max 77.8 s
  terminal losses have 259 s of chunk left on average (median) -- they had the opportunity to return and did not
  distance from the ball at the last sighting (m): re-appearing 15.65 | terminal 17.96 | usable events 15.66  (medians)
  normalised distance from the nearest image edge: re-appearing 0.34 | terminal 0.31 | usable events 0.36  (medians)
  last sighting within 10% of the image edge (a genuine frame exit): re-appearing 8.0%, terminal 13.9%, usable 8.3%

================================================================================================
TASK 3 -- SHAPE METRICS, OBSERVED VS RECONSTRUCTED
================================================================================================

  FULL reconstruction
  team-timesteps scored: 16901 (>= 1 player added on 14751, 87%); mean added 3.85
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    50.10 |         45.62 |       -4.48 |     -2.46 |     11.90
  depth spread  |     6.48 |          8.22 |       +1.74 |     +1.34 |      4.88
  width spread  |     9.97 |         11.24 |       +1.27 |     +0.82 |      5.18
  x-span        |    17.85 |         26.34 |       +8.48 |     +7.03 |     19.10
  +1-1 players (n=1567): line -1.41 m, depth +0.72 m, width +0.42 m
  +2-3 players (n=3547): line -2.54 m, depth +1.28 m, width +0.62 m
  +4-11 players (n=9637): line -5.69 m, depth +2.08 m, width +1.64 m
  line height off the pitch (<0 or >105 m): reconstruction 0.00%, observed-only 0.00%

  gap-ghosts only (dead re-id fragments excluded)
  team-timesteps scored: 16901 (>= 1 player added on 10307, 61%); mean added 1.19
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    50.46 |         47.96 |       -2.50 |     +0.00 |      8.31
  depth spread  |     6.45 |          7.24 |       +0.79 |     +0.34 |      3.04
  width spread  |     9.91 |         10.34 |       +0.43 |     -0.09 |      3.29
  x-span        |    17.59 |         21.59 |       +3.99 |     +1.21 |     11.49
  +1-1 players (n=4589): line -1.35 m, depth +0.61 m, width +0.25 m
  +2-3 players (n=4697): line -3.05 m, depth +0.91 m, width +0.53 m
  +4-11 players (n=1021): line -5.18 m, depth +1.00 m, width +0.81 m
  line height off the pitch (<0 or >105 m): reconstruction 0.00%, observed-only 0.00%
```
