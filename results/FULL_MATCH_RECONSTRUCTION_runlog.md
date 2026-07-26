```
================================================================================================
FULL-MATCH 22-PLAYER RECONSTRUCTION (frozen B4 v1, P2 policy) -- nothing is refitted
================================================================================================
match tottenham_manutd (Man Utd vs Tottenham), 11 chunks, 93882 gated position rows, 79 name-gated fragments
  [oracle] team 33: 11 starters, 5 used subs, 5 withdrawn -> on-pitch cap 11 throughout
  [oracle] team 35: 11 starters, 1 used subs, 1 withdrawn -> on-pitch cap 11 throughout
loaded train_meta from cache data\imputation\cache\train_meta.joblib
loaded v1_heads from cache data\imputation\cache\v1_heads.joblib
frozen v1 ready in 0 s
  h1_chunk_000: 2965 timesteps x 661 slots, 213385 hidden samples, 209 within-track + 1 cross-track events, 57798 estimate rows [69 s]
  h1_chunk_001: 2667 timesteps x 587 slots, 184887 hidden samples, 216 within-track + 1 cross-track events, 50250 estimate rows [59 s]
  h1_chunk_002: 2985 timesteps x 549 slots, 171432 hidden samples, 158 within-track + 2 cross-track events, 47488 estimate rows [54 s]
  h1_chunk_003: 2954 timesteps x 475 slots, 140783 hidden samples, 120 within-track + 2 cross-track events, 42052 estimate rows [44 s]
  h1_chunk_004: 2602 timesteps x 491 slots, 155145 hidden samples, 135 within-track + 0 cross-track events, 45723 estimate rows [49 s]
  h1_chunk_005: 573 timesteps x 99 slots, 26989 hidden samples, 21 within-track + 0 cross-track events, 8174 estimate rows [8 s]
  h2_chunk_000: 2961 timesteps x 543 slots, 171384 hidden samples, 131 within-track + 1 cross-track events, 52306 estimate rows [54 s]
  h2_chunk_001: 2850 timesteps x 338 slots, 110657 hidden samples, 81 within-track + 1 cross-track events, 34178 estimate rows [34 s]
  h2_chunk_002: 2995 timesteps x 460 slots, 145175 hidden samples, 103 within-track + 2 cross-track events, 45079 estimate rows [45 s]
  h2_chunk_003: 2759 timesteps x 302 slots, 91992 hidden samples, 60 within-track + 1 cross-track events, 33208 estimate rows [28 s]
  h2_chunk_004: 850 timesteps x 101 slots, 22608 hidden samples, 25 within-track + 0 cross-track events, 10552 estimate rows [7 s]

wrote results\reconstruction\tottenham_manutd_reconstruction.parquet (426808 rows)

================================================================================================
TASK 1 -- COVERAGE PROFILE
================================================================================================
timesteps on the chunk grid: 24343 (81.1 min); trusted (>= 4 gated players): 7293 (30.0%)

players per team per trusted timestep (share of team-timesteps):
  n  | observed only | with reconstruction
   0 |          0.6% |                0.0%
   1 |          2.8% |                0.1%
   2 |          8.0% |                0.6%
   3 |         13.1% |                1.2%
   4 |         15.8% |                1.5%
   5 |         15.8% |                3.0%
   6 |         14.5% |                3.5%
   7 |         10.8% |                4.8%
   8 |          7.7% |                5.7%
   9 |          5.2% |                6.3%
  10 |          3.8% |                7.7%
  11 |          1.2% |               65.1%
  >11 |          0.5% |                0.5%

mean per team: observed 5.30 -> reconstruction 9.82 of 11
  fraction of team-timesteps with >= 11: observed   1.8%  reconstruction  65.6%
  fraction of team-timesteps with >= 10: observed   5.6%  reconstruction  73.3%
  fraction of team-timesteps with >=  9: observed  10.8%  reconstruction  79.6%
  fraction of team-timesteps with >=  8: observed  18.5%  reconstruction  85.3%

imputed rows by source: {'v1': np.int64(312004), 'anchor': np.int64(36642)}
imputed rows by provenance: {'terminal': np.int64(244632), 'gap': np.int64(104014)} -- 'terminal' means the slot is never seen again, i.e. a dead re-identification fragment that may be a
phantom rather than a genuinely occluded player (results/B4_TRANSFER_M3.md section 3).

================================================================================================
TASK 2 -- RE-APPEARANCE VALIDATION
================================================================================================
census by linking rule and horizon bucket:
              0-1s  1-3s  3-5s  5-10s  10-30s  30s+
link                                               
cross-track      0     1     0      0       0    10
within-track   782   351    91     33       2     0

observation-noise floor of our own projected positions: sigma = 0.59 m per axis (65714 midpoint triples). Metrica truth has none; the frozen regions were calibrated
against exact truth, so this much error is unmodellable by construction on our footage.

  PRIMARY -- within-track linking (tracker id carried across the gap)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90 | 50+n | 90+n
  0-1s      |  782 |    2.55 |  0.5 |  2.5 |   2.55 |   2.62 |   2.55 |   17.3 |   51.3 |  0.2 |  0.7 | 68.3 | 74.7
  1-3s      |  351 |    3.75 |  2.0 |  5.5 |   3.66 |   3.70 |   3.75 |   20.5 |   56.7 |  1.2 |  2.6 | 31.3 | 62.7
  3-5s      |   91 |    5.80 |  2.9 |  8.6 |   5.71 |   5.71 |   5.80 |   40.7 |   73.6 |  2.6 |  5.5 | 44.0 | 73.6
  5-10s     |   33 |    7.58 |  4.8 | 11.5 |   7.18 |   8.74 |   7.58 |   39.4 |   81.8 |  4.3 |  8.9 | 39.4 | 81.8
  10-30s    |    2 |   10.10 |  8.4 | 12.9 |   8.49 |   3.02 |  10.10 |   50.0 |  100.0 |  6.3 | 13.5 | 50.0 | 50.0
  30s+      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  ALL       | 1259 |    3.47 |  0.9 |  4.9 |   3.41 |   3.53 |   3.47 |   20.5 |   55.3 |  0.8 |  1.8 | 55.4 | 71.4
  (50+n / 90+n = the SAME frozen regions widened in quadrature by the noise floor -- a
   diagnostic of why they miss, NOT a re-calibration; nothing in the model changed.)

  SECONDARY -- cross-track identity linking (NOT trusted; see report)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90
  0-1s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  1-3s      |    1 |    5.93 |  5.9 |  5.9 |   5.03 |   4.98 |   5.93 |    0.0 |    0.0 |  1.8 |  3.9
  3-5s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  5-10s     |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  10-30s    |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  30s+      |   10 |   17.44 | 13.0 | 25.3 |  18.54 |  28.76 |  17.44 |   30.0 |   70.0 |  9.2 | 19.7
  ALL       |   11 |   16.73 | 12.1 | 24.2 |  17.74 |  27.46 |  16.73 |   27.3 |   63.6 |  8.6 | 18.3
  cross-track gap frames with live tracking: median 21% -- a broadcast outage, not an occlusion

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
  0-1s      |  782 |     0.52 |      1.20 |        0.73 |     47420 | 0.40 |  55 |  92
  1-3s      |  351 |     1.05 |      2.18 |        3.08 |     68066 | 1.64 |  52 |  92
  3-5s      |   91 |     1.88 |      3.67 |        6.10 |     43262 | 3.15 |  54 |  94
  5-10s     |   33 |     4.02 |      6.32 |       10.10 |     74037 | 4.99 |  58 |  95
  10-30s    |    2 |     2.14 |      2.14 |       17.16 |     10890 | 6.18 |  65 |  94

  paired block-bootstrap CI on RMSE(emitted) - RMSE(anchor) and - RMSE(hold):
  0-1s      | vs anchor  +0.00 [ +0.00,  +0.00] | vs hold  -0.07 [ -0.16,  +0.04]
  1-3s      | vs anchor  +0.09 [ +0.00,  +0.17] | vs hold  +0.06 [ -0.30,  +0.42]
  3-5s      | vs anchor  +0.09 [ -0.20,  +0.36] | vs hold  +0.08 [ -0.82,  +1.34]
  5-10s     | vs anchor  +0.39 [ -0.24,  +1.01] | vs hold  -1.16 [ -2.07,  +0.06]
  ALL       | vs anchor  +0.07 [ -0.01,  +0.14] | vs hold  -0.06 [ -0.22,  +0.16]

  physically impossible links (implied speed > 12 m/s, i.e. an id switch or a gross projection failure): 26 of 1259 (2.1%)
  SENSITIVITY (diagnostic only -- excluding them removes the largest errors and so flatters the model): n 1233, RMSE 2.98 m, PICP50 20.9, PICP90 56.4

================================================================================================
TASK 2b -- SELECTION BIAS: WHO COMES BACK?
================================================================================================
track-losses in the match: 12462; re-appear under the same id: 63.0%; survive the liveness guard and become usable events: 10.1%
  gap length of re-appearing losses: p50 1.8 s, p90 11.0 s, max 58.6 s
  terminal losses have 268 s of chunk left on average (median) -- they had the opportunity to return and did not
  distance from the ball at the last sighting (m): re-appearing 17.56 | terminal 19.40 | usable events 18.34  (medians)
  normalised distance from the nearest image edge: re-appearing 0.34 | terminal 0.31 | usable events 0.35  (medians)
  last sighting within 10% of the image edge (a genuine frame exit): re-appearing 6.1%, terminal 11.3%, usable 6.4%

================================================================================================
TASK 3 -- SHAPE METRICS, OBSERVED VS RECONSTRUCTED
================================================================================================

  FULL reconstruction
  team-timesteps scored: 12740 (>= 1 player added on 11230, 88%); mean added 3.97
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    48.75 |         44.07 |       -4.67 |     -2.48 |     12.79
  depth spread  |     6.58 |          8.23 |       +1.65 |     +1.21 |      4.87
  width spread  |     9.89 |         11.22 |       +1.32 |     +0.86 |      5.45
  x-span        |    17.85 |         26.02 |       +8.17 |     +6.53 |     19.30
  +1-1 players (n=1095): line -0.90 m, depth +0.49 m, width +0.29 m
  +2-3 players (n=2687): line -2.54 m, depth +1.39 m, width +0.83 m
  +4-11 players (n=7448): line -6.00 m, depth +1.92 m, width +1.65 m
  line height off the pitch (<0 or >105 m): reconstruction 0.04%, observed-only 0.00%

  gap-ghosts only (dead re-id fragments excluded)
  team-timesteps scored: 12740 (>= 1 player added on 7951, 62%); mean added 1.29
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    48.75 |         46.22 |       -2.53 |     +0.00 |      8.69
  depth spread  |     6.59 |          7.28 |       +0.69 |     +0.16 |      3.12
  width spread  |     9.82 |         10.38 |       +0.56 |     -0.01 |      3.74
  x-span        |    17.62 |         21.36 |       +3.74 |     +0.53 |     11.04
  +1-1 players (n=3359): line -1.16 m, depth +0.40 m, width +0.41 m
  +2-3 players (n=3543): line -3.02 m, depth +0.84 m, width +0.66 m
  +4-11 players (n=1049): line -5.25 m, depth +1.11 m, width +0.69 m
  line height off the pitch (<0 or >105 m): reconstruction 0.05%, observed-only 0.00%
```
