```
================================================================================================
FULL-MATCH 22-PLAYER RECONSTRUCTION (frozen B4 v1, P2 policy) -- nothing is refitted
================================================================================================
match liverpool_manutd (Liverpool vs Man Utd), 10 chunks, 60913 gated position rows, 48 name-gated fragments
  [oracle] team 35: 11 starters, 3 used subs, 3 withdrawn -> on-pitch cap 11 throughout
  [oracle] team 44: 11 starters, 4 used subs, 4 withdrawn -> on-pitch cap 11 throughout
loaded train_meta from cache data\imputation\cache\train_meta.joblib
loaded v1_heads from cache data\imputation\cache\v1_heads.joblib
frozen v1 ready in 0 s
  h1_chunk_000: 2932 timesteps x 486 slots, 151680 hidden samples, 139 within-track + 1 cross-track events, 44192 estimate rows [49 s]
  h1_chunk_001: 2848 timesteps x 381 slots, 117660 hidden samples, 105 within-track + 1 cross-track events, 41226 estimate rows [37 s]
  h1_chunk_002: 2414 timesteps x 277 slots, 90225 hidden samples, 58 within-track + 0 cross-track events, 30548 estimate rows [28 s]
  h1_chunk_003: 2777 timesteps x 443 slots, 146468 hidden samples, 157 within-track + 0 cross-track events, 44569 estimate rows [46 s]
  h1_chunk_004: 2698 timesteps x 318 slots, 91609 hidden samples, 90 within-track + 0 cross-track events, 33963 estimate rows [29 s]
  h2_chunk_000: 2751 timesteps x 243 slots, 67090 hidden samples, 50 within-track + 0 cross-track events, 27066 estimate rows [21 s]
  h2_chunk_001: 2761 timesteps x 205 slots, 58234 hidden samples, 51 within-track + 1 cross-track events, 23938 estimate rows [18 s]
  h2_chunk_002: 1709 timesteps x 143 slots, 41990 hidden samples, 44 within-track + 0 cross-track events, 16056 estimate rows [13 s]
  h2_chunk_003: 2538 timesteps x 198 slots, 57172 hidden samples, 30 within-track + 0 cross-track events, 20932 estimate rows [18 s]
  h2_chunk_004: 2964 timesteps x 326 slots, 101604 hidden samples, 93 within-track + 0 cross-track events, 36461 estimate rows [32 s]

wrote results\reconstruction\liverpool_manutd_reconstruction.parquet (318951 rows)

================================================================================================
TASK 1 -- COVERAGE PROFILE
================================================================================================
timesteps on the chunk grid: 20673 (68.9 min); trusted (>= 4 gated players): 4849 (23.5%)

players per team per trusted timestep (share of team-timesteps):
  n  | observed only | with reconstruction
   0 |          0.5% |                0.0%
   1 |          3.6% |                0.2%
   2 |          8.0% |                0.8%
   3 |         13.5% |                1.9%
   4 |         18.3% |                3.5%
   5 |         16.4% |                5.5%
   6 |         13.7% |                5.2%
   7 |         10.1% |                7.0%
   8 |          6.6% |                9.0%
   9 |          4.4% |                9.8%
  10 |          2.7% |               10.1%
  11 |          1.5% |               46.2%
  >11 |          0.7% |                0.7%

mean per team: observed 5.12 -> reconstruction 9.08 of 11
  fraction of team-timesteps with >= 11: observed   2.2%  reconstruction  47.0%
  fraction of team-timesteps with >= 10: observed   4.9%  reconstruction  57.1%
  fraction of team-timesteps with >=  9: observed   9.3%  reconstruction  66.9%
  fraction of team-timesteps with >=  8: observed  15.9%  reconstruction  75.9%

imputed rows by source: {'v1': np.int64(243939), 'anchor': np.int64(24526)}
imputed rows by provenance: {'terminal': np.int64(194170), 'gap': np.int64(74295)} -- 'terminal' means the slot is never seen again, i.e. a dead re-identification fragment that may be a
phantom rather than a genuinely occluded player (results/B4_TRANSFER_M3.md section 3).

================================================================================================
TASK 2 -- RE-APPEARANCE VALIDATION
================================================================================================
census by linking rule and horizon bucket:
              0-1s  1-3s  3-5s  5-10s  30s+
link                                       
cross-track      0     0     0      0     3
within-track   513   238    52     14     0

observation-noise floor of our own projected positions: sigma = 0.52 m per axis (41739 midpoint triples). Metrica truth has none; the frozen regions were calibrated
against exact truth, so this much error is unmodellable by construction on our footage.

  PRIMARY -- within-track linking (tracker id carried across the gap)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90 | 50+n | 90+n
  0-1s      |  513 |    1.88 |  0.5 |  2.4 |   1.88 |   1.88 |   1.91 |   17.0 |   49.1 |  0.2 |  0.6 | 60.6 | 68.6
  1-3s      |  238 |    3.85 |  2.2 |  5.5 |   3.70 |   3.92 |   3.85 |   22.3 |   49.6 |  1.1 |  2.4 | 30.7 | 54.2
  3-5s      |   52 |    5.26 |  2.8 |  8.3 |   4.89 |   4.62 |   5.26 |   40.4 |   73.1 |  2.8 |  5.9 | 48.1 | 73.1
  5-10s     |   14 |    5.91 |  3.7 |  9.6 |   5.07 |   4.23 |   5.91 |   57.1 |   85.7 |  4.3 |  8.9 | 57.1 | 78.6
  10-30s    |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  30s+      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  ALL       |  817 |    2.98 |  1.0 |  4.2 |   2.86 |   2.89 |   2.99 |   20.7 |   51.4 |  0.7 |  1.6 | 51.0 | 64.9
  (50+n / 90+n = the SAME frozen regions widened in quadrature by the noise floor -- a
   diagnostic of why they miss, NOT a re-calibration; nothing in the model changed.)

  SECONDARY -- cross-track identity linking (NOT trusted; see report)
  horizon   |    n | emitted |  p50 |  p90 | anchor |   hold |     v1 | PICP50 | PICP90 |  r50 |  r90
  0-1s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  1-3s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  3-5s      |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  5-10s     |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  10-30s    |    0 |      - |    - |    - |     - |     - |     - |     - |     - |   - |   -
  30s+      |    3 |   19.84 | 13.5 | 27.1 |  19.19 |  38.33 |  19.84 |   33.3 |   66.7 |  9.9 | 21.2
  ALL       |    3 |   19.84 | 13.5 | 27.1 |  19.19 |  38.33 |  19.84 |   33.3 |   66.7 |  9.9 | 21.2
  cross-track gap frames with live tracking: median 8% -- a broadcast outage, not an occlusion

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
  0-1s      |  513 |     0.53 |      1.01 |        0.73 |     46810 | 0.39 |  56 |  92
  1-3s      |  238 |     1.25 |      2.33 |        3.08 |     66110 | 1.62 |  52 |  92
  3-5s      |   52 |     2.13 |      3.08 |        6.10 |     39453 | 3.10 |  55 |  94
  5-10s     |   14 |     1.98 |      3.12 |       10.10 |     36725 | 4.95 |  58 |  95

  paired block-bootstrap CI on RMSE(emitted) - RMSE(anchor) and - RMSE(hold):
  0-1s      | vs anchor  +0.00 [ -0.00,  +0.00] | vs hold  -0.00 [ -0.08,  +0.08]
  1-3s      | vs anchor  +0.16 [ +0.05,  +0.26] | vs hold  -0.07 [ -0.67,  +0.59]
  3-5s      | vs anchor  +0.36 [ -0.11,  +0.72] | vs hold  +0.63 [ -0.51,  +1.64]
  5-10s     | vs anchor  +0.83 [ -0.05,  +2.00] | vs hold  +1.67 [ -1.27,  +4.46]
  ALL       | vs anchor  +0.13 [ +0.06,  +0.19] | vs hold  +0.09 [ -0.21,  +0.36]

  physically impossible links (implied speed > 12 m/s, i.e. an id switch or a gross projection failure): 10 of 817 (1.2%)
  SENSITIVITY (diagnostic only -- excluding them removes the largest errors and so flatters the model): n 807, RMSE 2.82 m, PICP50 20.9, PICP90 52.0

================================================================================================
TASK 2b -- SELECTION BIAS: WHO COMES BACK?
================================================================================================
track-losses in the match: 8515; re-appear under the same id: 64.5%; survive the liveness guard and become usable events: 9.6%
  gap length of re-appearing losses: p50 1.6 s, p90 11.2 s, max 54.4 s
  terminal losses have 257 s of chunk left on average (median) -- they had the opportunity to return and did not
  distance from the ball at the last sighting (m): re-appearing 16.46 | terminal 18.18 | usable events 16.33  (medians)
  normalised distance from the nearest image edge: re-appearing 0.33 | terminal 0.30 | usable events 0.35  (medians)
  last sighting within 10% of the image edge (a genuine frame exit): re-appearing 7.5%, terminal 14.3%, usable 7.7%

================================================================================================
TASK 3 -- SHAPE METRICS, OBSERVED VS RECONSTRUCTED
================================================================================================

  FULL reconstruction
  team-timesteps scored: 8459 (>= 1 player added on 7159, 85%); mean added 3.55
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    52.30 |         47.54 |       -4.75 |     -2.93 |     12.71
  depth spread  |     5.91 |          7.51 |       +1.60 |     +1.15 |      4.66
  width spread  |    10.10 |         11.14 |       +1.04 |     +0.61 |      4.71
  x-span        |    15.98 |         23.75 |       +7.77 |     +6.38 |     18.05
  +1-1 players (n=1000): line -1.46 m, depth +0.66 m, width +0.40 m
  +2-3 players (n=1878): line -3.07 m, depth +1.25 m, width +0.57 m
  +4-11 players (n=4281): line -6.26 m, depth +1.98 m, width +1.40 m
  line height off the pitch (<0 or >105 m): reconstruction 0.00%, observed-only 0.00%

  gap-ghosts only (dead re-id fragments excluded)
  team-timesteps scored: 8459 (>= 1 player added on 4812, 57%); mean added 1.16
  metric        | observed | reconstructed |  mean delta | p50 delta | p90 |delta|
  line height   |    52.54 |         49.50 |       -3.04 |     -0.66 |      9.25
  depth spread  |     5.84 |          6.79 |       +0.95 |     +0.38 |      3.65
  width spread  |    10.09 |         10.46 |       +0.37 |     -0.12 |      3.42
  x-span        |    15.57 |         19.97 |       +4.40 |     +1.35 |     12.30
  +1-1 players (n=2038): line -1.58 m, depth +0.50 m, width +0.27 m
  +2-3 players (n=2178): line -3.57 m, depth +1.16 m, width +0.37 m
  +4-11 players (n=596): line -6.07 m, depth +1.73 m, width +0.73 m
  line height off the pitch (<0 or >105 m): reconstruction 0.00%, observed-only 0.00%
```
