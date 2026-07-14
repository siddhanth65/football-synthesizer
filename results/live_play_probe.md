# Live-play filter -- measurement on brighton_manutd (Phase-B B1.2)

Full match, 11 chunks, sampling grid 30011 frames (every 5th native frame). Classifier: `generator.live_play` (rule-based, pre-committed thresholds). All numbers ADDITIONAL -- no shipped metric or gate is changed.

Re-priced honestly (STATUS): the filter CANNOT create geometry (missing frames are close-ups/graphics); it fixes DENOMINATORS and cuts wasted compute.

## 1. Per-class fractions (full sampling grid)

`graphic` = grid frames absent from the dense parquet (zero detections at extraction: full-screen graphics, hard cuts, black, and extreme close-ups the football-YOLO cannot see -- this bucket conflates those, stated plainly).

| class | frames | fraction |
|-------|-------:|---------:|
| live_wide | 9647 | 32.1% |
| close_up | 7234 | 24.1% |
| replay_or_other | 4994 | 16.6% |
| graphic | 8136 | 27.1% |

Sanity/reconciliation: `live_wide` is **32.1%** of the full grid, but **44.1%** of DETECTED frames (excluding the 27% zero-detection `graphic` bucket) -- the latter matches the known ~41% live wide-camera fraction (measured on detected frames in the pl_probe diagnosis). Whole-match >=6-corr geometry yield over detected frames = **36.1%** (vs the ~37% reference) -- consistent.

### Per-chunk

| chunk | grid | live_wide | close_up | replay/other | graphic | ge6@live | cov@live |
|-------|-----:|----------:|---------:|-------------:|--------:|---------:|---------:|
| h1/chunk_000 | 3000 | 31% | 26% | 20% | 23% | 92% | 76% |
| h1/chunk_001 | 3000 | 29% | 27% | 25% | 19% | 88% | 81% |
| h1/chunk_002 | 3000 | 35% | 26% | 17% | 21% | 79% | 75% |
| h1/chunk_003 | 3000 | 34% | 21% | 17% | 28% | 81% | 67% |
| h1/chunk_004 | 2476 | 40% | 20% | 15% | 24% | 90% | 79% |
| h2/chunk_000 | 3000 | 40% | 25% | 19% | 16% | 75% | 62% |
| h2/chunk_001 | 3000 | 24% | 23% | 14% | 38% | 86% | 73% |
| h2/chunk_002 | 3000 | 33% | 26% | 12% | 29% | 68% | 60% |
| h2/chunk_003 | 3000 | 29% | 30% | 13% | 29% | 74% | 67% |
| h2/chunk_004 | 3000 | 29% | 17% | 15% | 39% | 74% | 77% |
| h2/chunk_005 | 535 | 23% | 14% | 9% | 54% | 90% | 81% |

## 2. Live-play-conditional numbers

- **Geometry yield on live_wide** (>=6 pitch correspondences): **80.5%** (7766/9647); PnLCalib solve on live_wide 89.8%. Expected band 65-90% -- in band.
- **Post-link ball coverage on live_wide**: **71.2%** (6873/9647) vs whole-grid 37.3% (11184/30011). 61% of all post-link ball frames fall on live_wide -- our pass-forming signal is overwhelmingly live-sourced.

### Pre-registered question: does live-play-conditioned pass recall clear the 50% gate?

Airtime shares: live_wide **32.1%**, live_wide+replay (action on screen) **48.8%**.

| team | our passes | oracle cmp | raw recall | cond. (/action) | ceiling (/live) |
|------|-----------:|-----------:|-----------:|----------------:|----------------:|
| Man Utd | 213 | 446 | 47.8% | 97.9% | 149% |
| Brighton | 198 | 407 | 48.6% | 99.7% | 151% |

**Raw recall 48.2% (both teams) -- does NOT clear the 50% gate** (reproduces fbref_gate 47.8/48.6%). The airtime-scaled 'conditioned' recall (98.8% mean) and the /live ceiling (>100%) are reported for completeness but REJECTED as honest answers: scaling the oracle by airtime assumes passes distribute uniformly over airtime, when passes are far denser in live_wide -- so both over-correct. A rigorous live-play-conditioned denominator needs per-pass timestamps we do not have.

**Straight verdict:** the live-play filter does NOT by itself push pass recall past 50% -- exactly as STATUS re-priced it. It is an honesty/denominator + compute win, not a recall lever. The remaining lever is the learned event/identity layer, not camera filtering.

## 3. Classifier audit (visual precision)

Stratified montages (`results/live_play_probe/montage_<class>.jpg`, 10 frames/class). Precision below is the worker's visual audit of those 40 crops (small n; directional).

| class | visual precision | what the sample actually contained |
|-------|-----------------:|-------------------------------------|
| live_wide | ~100% (10/10) | all genuine wide tactical main-camera views -- the load-bearing class is clean |
| close_up | ~50% (5/10) | true face/body close-ups AND distant wide shots where the detector found only 1-2 players (sparse-detection wide views mislabelled) |
| replay_or_other | ~20% as 'replay' | dominated by live MEDIUM-wide tactical frames (5-7 detections just under the live_wide gate); very few actual replays -- an ambiguous medium-count bin, not a replay detector |
| graphic | ~0% as literal 'graphic' | close-ups, celebration huddles, set-piece scrambles, crowd/stadium shots where the football-YOLO detected ZERO players -- NOT scoreboard/lineup graphics (none appeared); really a 'zero-detection / detector-blind' bucket |

**Honest takeaway:** the reliable distinction is BINARY -- `live_wide` (clean, ~100% precision) vs 'not geometry-yielding'. The 4-way naming over-promises: the non-live sub-classes are approximate (`close_up`/`replay_or_other`/`graphic` all really mean 'the detector under-populated this frame', for varied reasons). For the DENOMINATOR purpose this is fine -- what matters is that live_wide is precise, so the live-play-conditional yields and coverage above are trustworthy. For a semantic shot-type product the non-live split would need box-height (not persisted) or a colour/motion cue.

## Method + caveats

- Signals: detected players+GK count, image x-spread, calibration error, >=6-corr count -- all from the dense parquet (CPU, no re-decode). Box HEIGHT is not persisted (extract.py collapses each box to a foot point), so tallness is proxied by low detection count; this matches the STATUS finding that geometry-less frames carry a median of 4 detected players.
- The `graphic` bucket is a residual (grid frames with zero detections); it cannot be sub-split into graphic vs extreme-close-up from the parquet alone.
- Thresholds were fixed in `generator/live_play.py` BEFORE any recall measurement (no tuning on the gate outcome).
