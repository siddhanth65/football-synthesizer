# BAS pass/drive validation (B-3 stage 1)

Operating point (frozen on brighton_manutd h1): PASS confidence >= 0.4, same-class peaks merged within 1 s.
Replay filter (pre-committed hypothesis): keep peaks in live_wide segments (gap-merge 2 s, pad 1 s).

`raw` = every BAS peak. `live` = replay-filter arm (+dedup). `op` = confidence+dedup arm. `ratio_*` vs Sofascore attempted; a half's ratio is shown only when BAS covers >= 90% of the half's frame span (else PARTIAL).

## brighton_manutd

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 729 | 428 | 538 | 543 | 1.343 | 0.788 | 0.991 |
| h2 | 654 | 384 | 443 | 445 | 1.470 | 0.863 | 0.996 |
| **match** | 1383 | 812 | 981 | 988 | 1.400 | 0.822 | 0.993 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 667 | 420 | 424 |
| h2 | 600 | 410 | 343 |
| **match** | 1267 | 830 | 767 |

## manutd_liverpool

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 698 | 343 | 510 | 524 | 1.332 | 0.655 | 0.973 |
| h2 | 668 | 343 | 471 | 447 | 1.494 | 0.767 | 1.054 |
| **match** | 1366 | 686 | 981 | 971 | 1.407 | 0.706 | 1.010 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 621 | 326 | 397 |
| h2 | 571 | 328 | 361 |
| **match** | 1192 | 654 | 758 |

## manutd_fulham

Coverage -- h1: 5/5 chunks (100% of frames), h2: 5/5 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 694 | 363 | 500 | 485 | 1.431 | 0.748 | 1.031 |
| h2 | 661 | 304 | 445 | 381 | 1.735 | 0.798 | 1.168 |
| **match** | 1355 | 667 | 945 | 866 | 1.565 | 0.770 | 1.091 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 636 | 390 | 392 |
| h2 | 541 | 286 | 286 |
| **match** | 1177 | 676 | 678 |

## palace_manutd

Coverage -- h1: 6/6 chunks (100% of frames), h2: 5/6 chunks (99% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 639 | 278 | 447 | 467 | 1.368 | 0.595 | 0.957 |
| h2 | 721 | 301 | 518 | 485 | 1.487 | 0.621 | 1.068 |
| **match** | 1360 | 579 | 965 | 952 | 1.429 | 0.608 | 1.014 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 532 | 273 | 345 |
| h2 | 625 | 284 | 374 |
| **match** | 1157 | 557 | 719 |

## manutd_tottenham

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 743 | 343 | 541 | 526 | 1.413 | 0.652 | 1.029 |
| h2 | 746 | 334 | 538 | 505 | 1.477 | 0.661 | 1.065 |
| **match** | 1489 | 677 | 1079 | 1031 | 1.444 | 0.657 | 1.047 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 657 | 328 | 416 |
| h2 | 622 | 326 | 398 |
| **match** | 1279 | 654 | 814 |

## southampton_manutd

Coverage -- h1: 5/5 chunks (100% of frames), h2: 6/6 chunks (100% of frames)

### PASS

| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |
|------|-----|------|----|-----------|-----------|------------|----------|
| h1 | 706 | 436 | 520 | 492 | 1.435 | 0.886 | 1.057 |
| h2 | 756 | 473 | 567 | 599 | 1.262 | 0.790 | 0.947 |
| **match** | 1462 | 909 | 1087 | 1091 | 1.340 | 0.833 | 0.996 |

### DRIVE (no ground truth -- raw vs filtered only)

| half | raw | live | op |
|------|-----|------|----|
| h1 | 630 | 426 | 411 |
| h2 | 676 | 465 | 451 |
| **match** | 1306 | 891 | 862 |

## Fulham h2 outlier -- audit

CPU-only, no new inference. Buckets `op` PASS events and the E2E-Spot shot-spotting probe
(`results/action_spotting_probe/manutd_fulham/summary.json`, thresh 0.30, min_sep 30 s) by chunk for
manutd_fulham h2, against a reference distribution built from the other 11 halves in the corpus.

### BAS PASS(op), per chunk (reference: 29 h2 chunks from the other 5 matches)

| chunk | pass_op | span (s) | rate (pass/min) | corpus h2 rate range |
|-------|---------|----------|------------------|-----------------------|
| h2_chunk_000 | 85 | 600.8 | 8.49 | mean 9.96, median 10.00, 7.31-13.94 |
| h2_chunk_001 | 83 | 589.4 | 8.45 | (same) |
| h2_chunk_002 | 89 | 598.0 | 8.93 | (same) |
| h2_chunk_003 | 90 | 589.6 | 9.16 | (same) |
| h2_chunk_004 | 98 | 575.6 | 10.22 | (same) -- contains the 87' goal (elapsed h2 clock ~84:26-94:13) |

All 5 chunks sit inside the corpus range; none approaches the corpus max (13.94). No chunk is a
localized PASS outlier -- the 1.168x match-level over-fire (op 445 vs truth 381) is a diffuse
uplift spread evenly across the half, not one noisy segment. `h2_chunk_004` (the goal chunk) has the
highest rate but is still below the corpus mean + 1 spread and well inside the observed range.

### E2E-Spot shots (Shots on target + Shots off target summed), per chunk (reference: 30 h2 chunks,
other 5 matches)

| chunk | shots (on+off) | on-target only | span (s) | rate (shots/min) |
|-------|-----------------|-----------------|----------|-------------------|
| h2_chunk_000 | 4 | 2 | 601.5 | 0.40 |
| h2_chunk_001 | 3 | 0 | 600.5 | 0.30 |
| h2_chunk_002 | 7 | 5 | 598.5 | 0.70 |
| h2_chunk_003 | 7 | 5 | 601.5 | 0.70 |
| h2_chunk_004 | 3 | 2 | 597.0 | 0.30 -- contains the 87' goal |

Corpus reference (on+off summed): mean 0.26/min, median 0.30, max 0.70/min (n=30 chunks). Corpus
reference (on-target alone, 61 chunks across both halves of the other 5 matches): mean 1.34, median
1.0, max 4 anywhere in the corpus.

`h2_chunk_002` and `h2_chunk_003` are the outliers: each fires 5 "Shots on target" peaks, tying for
the highest on-target count anywhere in the corpus (max elsewhere is 4), and both tie the corpus-wide
max combined rate (0.70/min). These two chunks span elapsed h2 clock ~64:50-84:26 -- i.e. **before**
the 87' goal, not the post-goal celebration window (`h2_chunk_004`, which is normal on every measure
above).

### Characterization (cached artifacts only)

- **Shot-type at each peak timestamp** (`outputs/manutd_fulham/live_play/h2_chunk_00{2,3}.json`,
  nearest tracking-fps grid frame to each spot-probe peak, spot fps 2 Hz vs tracking fps 25 Hz):
  chunk_002's 5 distinct peak times are all `close_up` (2) or `replay_or_other` (3) -- zero
  `live_wide`. chunk_003's 6 distinct peak times are 2 `live_wide`, 1 `close_up`, 3
  `replay_or_other`. Baseline chunks (000/001/004) skew more toward `live_wide` (1-3 of 3-4 peaks
  each). Sample sizes (3-7 peaks/chunk) are too small to call this a clean statistical separation,
  and per the frozen finding above the shot-type classifier conflates wide/tight *framing* with
  live/replay -- a `close_up` or `replay_or_other` tag at a peak does **not** by itself prove the
  peak is a re-shown replay.
- **On/off-target class coincidence** (framing-independent, exact-timestamp check): at
  `h2_chunk_002` t=363.5s and t=527.0s, and `h2_chunk_003` t=272.5s, the *same* frame fires both the
  "Shots on target" and "Shots off target" channels above the 0.30 threshold simultaneously -- 3
  moments double-counted by the probe's `on-count + off-count` summing. Checked across every other
  chunk in this match (all 5 h1 chunks, h2_chunk_000/001/004): zero such coincidences. This pattern
  is unique to the two outlier chunks.
- **Tracking quality** (aligned parquet, same two chunks): mean players/frame 6.6 (chunk_002) / 7.0
  (chunk_003), tracked frames 2159 / 1856 -- both in line with fulham's other h2 chunks (6.5-7.7,
  1662-2111 frames) and the whole 6-match corpus (5.8-9.3 mean players/frame). No tracking
  degradation in the outlier chunks.

### Verdict

- **BAS PASS: not explained by a localized segment.** Every fulham h2 chunk sits inside the 11-half
  corpus range; the over-count is a diffuse ~1.0-1.2x uplift, not one noisy stretch. There is no
  chunk whose exclusion is defensible: dropping the highest-rate chunk (`h2_chunk_004`, 98 events)
  would leave 347 op events against the same 381-pass whole-half truth (0.91x, now an under-count) --
  not a meaningful correction, since the truth total covers all 5 chunks, not 4. **Hypothetically**,
  no single-chunk exclusion moves the frozen-threshold corpus ratios materially; the fulham h2 case
  would remain the corpus outlier at 1.168x (or worse) under any one-chunk removal.
- **E2E-Spot shots: partially explained, and it is chunk-localized -- but not to the celebration
  window.** `h2_chunk_002`/`h2_chunk_003` (pre-goal) are genuine, corpus-wide outliers on
  "Shots on target"; `h2_chunk_004` (post-goal) is normal on every measure checked. At least 3 of the
  h2-level excess shots are a **measurement artifact**: the probe sums on-target and off-target peaks
  as independent classes, and in these two chunks the same real moment fires both channels at once.
  Deduplicating those 3 coincidences would drop h2 detected shots from 24 to 21 (h2 truth reference,
  Sofascore `totalShotsOnGoal` 2ND = 15). The remaining gap (21 vs 15) is **unexplained** by any
  artifact examined here: the shot-type signal is small-sample and confounded (framing, not
  live/replay), and tracking density is normal, so this residual could be genuine elevated shot
  activity in that real 20-minute window (both sides pushing before the 87' winner) or a model
  false-positive rate specific to this broadcast's camera style -- not distinguishable with existing
  artifacts. **Hypothetically**, if `h2_chunk_002`/`h2_chunk_003` were excluded entirely, h2 detected
  shots would fall to 10 (well under the 15 truth) -- not a meaningful correction either, since real
  match time (and real shots) fall inside those chunks.

### Recommendation (not implemented)

1. E2E-Spot: de-duplicate "detected shots" by merging on-target/off-target peaks that land within
   the model's own ~1 s temporal tolerance into a single shot event before comparing to oracle counts
   (same pattern as the `DEDUP_S` merge already frozen for BAS). This mechanically removes the
   3-event double-count found above; it does not address the unexplained residual.
2. A goal-triggered refractory window (the original hypothesis on record) is **not supported** by
   this data for either model: the outlier chunks (BAS: none really outlying; E2E-Spot: chunks
   002/003) sit *before* the 87' goal, and the actual post-goal chunk (`h2_chunk_004`) is normal on
   every measure checked here. This fix should not be adopted on the strength of this audit.
