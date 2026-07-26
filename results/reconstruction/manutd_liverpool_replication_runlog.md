```
================================================================================================
FULL-MATCH 22-PLAYER RECONSTRUCTION (frozen B4 v1, P2 policy) -- nothing is refitted
================================================================================================
match manutd_liverpool (Man Utd vs Liverpool), 11 chunks, 63954 gated position rows, 59 name-gated fragments
  [oracle] team 35: 11 starters, 4 used subs, 4 withdrawn -> on-pitch cap 11 throughout
  [oracle] team 44: 11 starters, 4 used subs, 4 withdrawn -> on-pitch cap 11 throughout
loaded train_meta from cache data\imputation\cache\train_meta.joblib
loaded v1_heads from cache data\imputation\cache\v1_heads.joblib
frozen v1 ready in 0 s
  h1_chunk_000: 2945 timesteps x 361 slots, 110409 hidden samples, 88 within-track + 0 cross-track events, 42509 estimate rows [36 s]
  h1_chunk_001: 2904 timesteps x 358 slots, 115045 hidden samples, 143 within-track + 2 cross-track events, 41214 estimate rows [36 s]
  h1_chunk_002: 2542 timesteps x 345 slots, 102488 hidden samples, 80 within-track + 2 cross-track events, 33994 estimate rows [32 s]
  h1_chunk_003: 2997 timesteps x 210 slots, 63363 hidden samples, 51 within-track + 1 cross-track events, 23942 estimate rows [20 s]
  h1_chunk_004: 2338 timesteps x 243 slots, 69779 hidden samples, 46 within-track + 0 cross-track events, 26446 estimate rows [22 s]
  h2_chunk_000: 3007 timesteps x 282 slots, 81821 hidden samples, 66 within-track + 0 cross-track events, 31646 estimate rows [26 s]
  h2_chunk_001: 2991 timesteps x 287 slots, 84205 hidden samples, 81 within-track + 2 cross-track events, 32713 estimate rows [26 s]
  h2_chunk_002: 2910 timesteps x 193 slots, 57124 hidden samples, 24 within-track + 0 cross-track events, 21069 estimate rows [17 s]
  h2_chunk_003: 2563 timesteps x 344 slots, 108979 hidden samples, 98 within-track + 2 cross-track events, 36365 estimate rows [34 s]
  h2_chunk_004: 3005 timesteps x 314 slots, 93191 hidden samples, 81 within-track + 0 cross-track events, 35025 estimate rows [29 s]
  h2_chunk_005: 455 timesteps x 98 slots, 19045 hidden samples, 33 within-track + 0 cross-track events, 9129 estimate rows [6 s]

wrote results\reconstruction\manutd_liverpool_reconstruction.parquet (334052 rows)

================================================================================================
TASK 1 -- COVERAGE PROFILE
================================================================================================
timesteps on the chunk grid: 22505 (75.0 min); trusted (>= 4 gated players): 5169 (23.0%)

players per team per trusted timestep (share of team-timesteps):
  n  | observed only | with reconstruction
   0 |          0.6% |                0.0%
   1 |          2.7% |                0.2%
   2 |          9.5% |                1.2%
   3 |         15.2% |                2.2%
   4 |         17.2% |                4.4%
   5 |         18.4% |                6.4%
   6 |         14.1% |                6.6%
   7 |          9.2% |                8.7%
   8 |          5.4% |                9.1%
   9 |          3.9% |               10.1%
  10 |          2.8% |               10.8%
  11 |          1.0% |               40.1%
  >11 |          0.2% |                0.2%

mean per team: observed 4.96 -> reconstruction 8.74 of 11
  fraction of team-timesteps with >= 11: observed   1.2%  reconstruction  40.3%
  fraction of team-timesteps with >= 10: observed   4.0%  reconstruction  51.1%
  fraction of team-timesteps with >=  9: observed   7.8%  reconstruction  61.2%
  fraction of team-timesteps with >=  8: observed  13.2%  reconstruction  70.3%

imputed rows by source: {'v1': np.int64(258859), 'anchor': np.int64(23030)}
imputed rows by provenance: {'terminal': np.int64(209085), 'gap': np.int64(72804)} -- 'terminal' means the slot is never seen again, i.e. a dead re-identification fragment that may be a
phantom rather than a genuinely occluded player (results/B4_TRANSFER_M3.md section 3).

================================================================================================
TASK 2 -- RE-APPEARANCE VALIDATION
================================================================================================
census by linking rule and horizon bucket:
              0-1s  1-3s  3-5s  5-10s  30s+
link                                       
cross-track      0     0     0      0     9
within-track   532   209    36     14     0

observation-noise floor of our own projected positions: sigma = 0.28 m per axis (45025 midpoint triples). Metrica truth has none; the frozen regions were calibrated
against exact truth, so this much error is unmodellable by construction on our footage.

  PRIMARY -- within-track linking (tracker id carried across the gap)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90 | 50+n | 90+n
  0-1s      |  532 |    2.25 |  0.5 |  1.9 |   2.25 |   2.34 |   2.30 |   20.9 |   56.2 |  0.2 |  0.7 | 47.6 | 67.7
  1-3s      |  209 |    3.19 |  2.1 |  4.4 |   3.01 |   3.26 |   3.19 |   22.5 |   58.4 |  1.2 |  2.4 | 24.9 | 61.7
  3-5s      |   36 |    5.26 |  3.7 |  8.7 |   5.18 |   7.00 |   5.26 |   27.8 |   69.4 |  2.5 |  5.2 | 27.8 | 66.7
  5-10s     |   14 |    6.38 |  5.2 |  9.8 |   5.29 |   5.61 |   6.38 |   35.7 |   78.6 |  4.2 |  8.7 | 42.9 | 78.6
  10-30s    |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  30s+      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  ALL       |  791 |    2.84 |  0.8 |  3.6 |   2.74 |   3.05 |   2.87 |   21.9 |   57.8 |  0.6 |  1.5 | 40.6 | 66.2
  (50+n / 90+n = the SAME frozen regions widened in quadrature by the noise floor -- a
   diagnostic of why they miss, NOT a re-calibration; nothing in the model changed.)

  SECONDARY -- cross-track identity linking (NOT trusted; see report)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90
  0-1s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  1-3s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  3-5s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  5-10s     |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  10-30s    |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  30s+      |    9 |   17.73 | 11.4 | 28.2 |  18.95 |  33.51 |  17.73 |   22.2 |   66.7 |  7.8 | 16.6
  ALL       |    9 |   17.73 | 11.4 | 28.2 |  18.95 |  33.51 |  17.73 |   22.2 |   66.7 |  7.8 | 16.6
  cross-track gap frames with live tracking: median 23% -- a broadcast outage, not an occlusion

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
  0-1s      |  532 |     0.52 |      0.93 |        0.73 |     44873 | 0.38 |  57 |  92
  1-3s      |  209 |     1.20 |      2.04 |        3.08 |     60299 | 1.58 |  53 |  92
  3-5s      |   36 |     3.12 |      4.68 |        6.10 |     49548 | 3.27 |  52 |  93
  5-10s     |   14 |     3.31 |      4.12 |       10.10 |     30865 | 4.98 |  58 |  95

  paired block-bootstrap CI on RMSE(emitted) - RMSE(anchor) and - RMSE(hold):
  0-1s      | vs anchor  +0.00 [ +0.00,  +0.00] | vs hold  -0.09 [ -0.16,  -0.04]
  1-3s      | vs anchor  +0.18 [ +0.06,  +0.31] | vs hold  -0.08 [ -0.44,  +0.30]
  3-5s      | vs anchor  +0.08 [ -0.32,  +0.46] | vs hold  -1.74 [ -4.00,  +0.35]
  5-10s     | vs anchor  +1.09 [ +0.05,  +1.86] | vs hold  +0.78 [ -1.59,  +2.99]
  ALL       | vs anchor  +0.10 [ +0.02,  +0.19] | vs hold  -0.21 [ -0.63,  +0.06]

  physically impossible links (implied speed > 12 m/s, i.e. an id switch or a gross projection failure): 8 of 791 (1.0%)
  SENSITIVITY (diagnostic only -- excluding them removes the largest errors and so flatters the model): n 783, RMSE 2.24 m, PICP50 22.1, PICP90 58.4

================================================================================================
TASK 2b -- SELECTION BIAS: WHO COMES BACK?
================================================================================================
track-losses in the match: 7908; re-appear under the same id: 61.6%; survive the liveness guard and become usable events: 10.0%
  gap length of re-appearing losses: p50 1.4 s, p90 11.4 s, max 56.0 s
  terminal losses have 213 s of chunk left on average (median) -- they had the opportunity to return and did not
  distance from the ball at the last sighting (m): re-appearing 16.64 | terminal 17.55 | usable events 16.54  (medians)
  normalised distance from the nearest image edge: re-appearing 0.33 | terminal 0.30 | usable events 0.33  (medians)
  last sighting within 10% of the image edge (a genuine frame exit): re-appearing 8.2%, terminal 13.3%, usable 8.5%

================================================================================================
TASK 3 -- SHAPE METRICS, OBSERVED VS RECONSTRUCTED
================================================================================================

  FULL reconstruction
  team-timesteps scored: 8934 (>= 1 player added on 7251, 81%); mean added 3.33
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    49.62 |         44.87 |       -4.75 |     -2.84 |     13.19
  depth spread  |     6.70 |          8.18 |       +1.49 |     +0.87 |      4.73
  width spread  |     9.51 |         10.42 |       +0.91 |     +0.40 |      4.70
  x-span        |    18.01 |         25.48 |       +7.46 |     +5.56 |     17.91
  +1-1 players (n=982): line -1.46 m, depth +0.44 m, width +0.36 m
  +2-3 players (n=2021): line -3.28 m, depth +0.98 m, width +0.65 m
  +4-11 players (n=4248): line -6.21 m, depth +1.97 m, width +1.16 m
  line height off the pitch (<0 or >105 m): reconstruction 0.00%, observed-only 0.00%

  gap-ghosts only (dead re-id fragments excluded)
  team-timesteps scored: 8934 (>= 1 player added on 4624, 52%); mean added 0.90
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    50.42 |         47.58 |       -2.84 |     -0.05 |      8.83
  depth spread  |     6.61 |          7.28 |       +0.66 |     +0.16 |      3.10
  width spread  |     9.46 |          9.83 |       +0.37 |     -0.26 |      3.38
  x-span        |    17.63 |         20.97 |       +3.34 |     +0.00 |     10.17
  +1-1 players (n=2547): line -1.79 m, depth +0.45 m, width +0.26 m
  +2-3 players (n=1745): line -3.83 m, depth +0.92 m, width +0.62 m
  +4-11 players (n=332): line -5.76 m, depth +0.93 m, width -0.14 m
  line height off the pitch (<0 or >105 m): reconstruction 0.00%, observed-only 0.00%
```
