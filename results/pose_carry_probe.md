# Pose carry-over for PLAYERS — probe

**Verdict: DON'T SHIP as a coverage lever. The mechanism is sound and faithfulness-validated, but it
cannot deliver the hypothesised win — because the premise that motivated it is wrong.**

The probe's most valuable output is not the lever. It is the **correction to our diagnosis of the
37% geometry yield**: the frames we lose are not frames with a bad pose. They are close-ups.

Code: `generator/pose_carry.py` (reusable, off by default), `tools/pose_carry_probe.py`,
`tests/test_pose_carry.py` (9 tests). CPU-only. Nothing was written to `outputs/`; the enriched
positions and both fact stores live in `results/pose_carry_probe_scratch/`.

---

## 1. The mechanism (and what it is NOT)

For a frame with no usable player geometry, **borrow** a known-good homography from a
camera-continuous neighbour within ±2 s and project *that frame's own* detected players through it.
Cuts are detected with `ball_carry`'s track-ID Jaccard < 0.30 detector; a carry never crosses one.
Every projection is off-pitch-gated (`clamp_to_pitch`, 2 m tolerance — dropped, never clamped onto
the line), and the recovered frame must still pass the same plausibility gate a native pose faces
(≥ 8 players on the pitch, spanning ≥ 25 m).

**This is not the disproven "mechanism 2".** `results/pl_probe/diagnosis/DIAGNOSIS.md` disproved
*reusing the frame's own (globally wrong) pose* — which cannot work, since reusing a wrong pose
reproduces the same off-pitch garbage. Here the pose comes from a **different, verified-good frame**;
the only thing taken from the target frame is its player pixel coordinates. The two are unrelated
mechanisms, and this one is faithfulness-validated below.

**Source of the "known-good" homographies — (a) beats (b), and (b) is not merely worse, it is the
bug.** Two candidates were specified:

- **(a) refit from the frame's own player correspondences** (`ball_carry.fit_frame_homographies`;
  needs ≥ 6 players carrying valid pitch coords). This selects poses that are known good **by their
  output** — the players they project land on the pitch and survive the plausibility gate.
- **(b) the calibration's stored homography wherever the pose was ACCEPTED.** First, it is not
  available: `generator/extract.py` persists only `calib_error_m`, never the homography, and no
  calibration artifact exists on disk — evaluating (b) would require re-running PnLCalib over the
  video (a GPU job; this probe is CPU-only). Second, and decisively, **(b)'s source pool is
  precisely the polluted one.** "Accepted" means keypoint reprojection ≤ 2 m, which 91.8% of frames
  pass — including all the frames whose pose is globally wrong. On brighton, 10,116 detection-frames
  (46.2%) pass the calibration gate and still emit **zero** player coordinates. Feeding those poses
  in as carry sources would inject exactly the error class the mechanism exists to route around.
  **(a) is used throughout.**

**One semantic change is load-bearing.** A carried frame inherits its **source frame's**
`calib_error_m`. This is the honest provenance (the geometry it carries *is* that source pose), and
it is what lets the fact store see the frame at all: `report/facts.py` gates on
`calib_error_m <= 1.0`, which is exactly why the *ball* carry-over contributed zero possession
samples (STATUS: "the coverage win and the possession bias are decoupled"). Without this relabelling
the player carry would be invisible downstream too.

---

## 2. Faithfulness — PASS (pre-committed, leave-one-out)

Hide a good frame's own pose, borrow a neighbour's, compare the projected players against their known
pitch positions. **Acceptance: median ≤ 1.0 m and p90 ≤ 3.0 m.** Whole match, 11 chunks.

| variant | n | median | p90 | max | ≤1 m | ≤2 m | ≤5 m | off-pitch gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **naive LOO** (nearest good neighbour — the literal pre-committed test, identical in construction to the ball probe's LOO) | 92,059 | **0.20 m** | **1.04 m** | 73.4 m | 89% | 95% | 98% | 0.1% |
| **gap-matched blocked LOO** (stricter, see below; 5 seeds) | ~88,600 | 0.39–0.41 m | 2.99–3.09 m | 75.4 m | 71% | 84% | 95% | 0.2% |

**Pre-committed verdict: PASS** (0.20 m / 1.04 m, comfortably inside the bar).

The *blocked* variant is a stricter test I added, not the pre-committed one. Good frames cluster, so
the naive nearest good neighbour is only ~5 frames away — optimistically close, since production
borrows across *runs* of bad frames (real carry gap: median 15 frames, p90 40). The blocked variant
forces the borrow distance to match that real gap distribution. It lands **exactly at the p90 bar**
(2.99–3.09 m across seeds). Reported as found; **nothing was tuned.**

Sanity: only 0.1–0.2% of carried player projections are off-pitch-gated (the ball probe saw
outliers to 756 m). Borrowed player poses are physically sane — the camera is near-static across
±2 s, so a neighbour's pose is a valid stand-in. **The mechanism works.**

---

## 3. Geometry yield — the mechanism works, and it barely moves

`brighton_manutd`, whole match (21,875 detection-frames).

| chunk | det | good | yield OFF | carried | interp | gate-dropped | no source | yield ON | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h1_chunk_000 | 2314 | 865 | 37.4% | 1 | 15 | 584 | 849 | 38.1% | +0.7 |
| h1_chunk_001 | 2435 | 783 | 32.2% | 23 | 18 | 670 | 941 | 33.8% | +1.7 |
| h1_chunk_002 | 2359 | 855 | 36.2% | 17 | 30 | 506 | 951 | 38.2% | +2.0 |
| h1_chunk_003 | 2167 | 834 | 38.5% | 17 | 14 | 592 | 710 | 39.9% | +1.4 |
| h1_chunk_004 | 1887 | 897 | 47.5% | 27 | 17 | 490 | 456 | 49.9% | +2.3 |
| h2_chunk_000 | 2515 | 904 | 35.9% | 23 | 28 | 543 | 1017 | 38.0% | +2.0 |
| h2_chunk_001 | 1849 | 649 | 35.1% | 41 | 9 | 376 | 774 | 37.8% | +2.7 |
| h2_chunk_002 | 2140 | 706 | 33.0% | 21 | 24 | 323 | 1066 | 35.1% | +2.1 |
| h2_chunk_003 | 2143 | 639 | 29.8% | 3 | 10 | 420 | 1071 | 30.4% | +0.6 |
| h2_chunk_004 | 1821 | 646 | 35.5% | 4 | 18 | 534 | 619 | 36.7% | +1.2 |
| h2_chunk_005 | 245 | 109 | 44.5% | 0 | 2 | 71 | 63 | 45.3% | +0.8 |
| **MATCH** | **21875** | **7887** | **36.1%** | 177 | 185 | 5109 | 8517 | **37.7%** | **+1.7** |

**+1.7 pp**, against the ball's +18.0 pp. The lever does not fire.

---

## 4. WHY — the premise is wrong (the real finding)

STATUS and the probe brief both assert that on the ~63% of detection-frames without player geometry
"the pose is GLOBALLY WRONG (fits keypoints, projects players off-pitch)". **The data says
otherwise.** Anatomy of `brighton_manutd`'s 21,875 detection-frames:

| bucket | frames | share |
|---|---:|---:|
| yield usable player geometry | 7,887 | 36.1% |
| **no player geometry** | **13,988** | **63.9%** |
| ├─ calibration FAILED (> 2 m / inf) | 3,872 | 17.7% |
| └─ calibration ACCEPTED, yet zero player output | 10,116 | 46.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;├─ … with **≥ 8 players detected** → genuinely a bad pose | **985** | **4.5%** |
| &nbsp;&nbsp;&nbsp;&nbsp;└─ … with **< 8 players detected** → a CLOSE-UP | **9,131** | **41.7%** |

**90.3% of the frames we have been calling "globally wrong pose" are simply close-ups with too few
players to form a plausible frame.** The geometry-less frames carry a **median of 4 detected
players**; the good frames carry 11. Only 14.6% of them have the ≥ 8 detections that
`reject_implausible_frames` requires — and no borrowed pose can invent a player.

Two independent confirmations:

- **The borrowed pose is working; there is nothing to project.** Among target frames that *did* get a
  source but failed the frame gate, the median frame landed **5 of its 5 detected players on the
  pitch**. The pose is fine. Five is not eight.
- **The calibration-independent live-wide-play proxy** (≥ 8 detections AND image-x spread ≥ 800 px,
  from `DIAGNOSIS.md`) flags **97.9%** of good frames but only **5.2%** of target frames (h1_chunk_000).
  The geometry-less frames are replays, close-ups, crowd and tight shots — exactly the ~59% non-live
  content `DIAGNOSIS.md` already measured in a raw PL slice, which is why 13,988/21,875 = 63.9%
  reconciles with it.

**Hard ceiling on ANY pose-borrowing mechanism** (perfect pose on every addressable frame):

| | frames | yield |
|---|---:|---:|
| current | 7,887 | 36.1% |
| + every addressable frame (≥ 8 detections) recovered | +2,038 | **45.4% (+9.3 pp)** |
| what we actually got | +362 | 37.7% (+1.7 pp) |

Even a *perfect* pose oracle buys +9.3 pp, not +18. The 37.4% yield is **not a pose-quality bug — it
is approximately the live-wide-play fraction of the broadcast.** The pose is not the bottleneck; the
camera is.

---

## 5. Downstream impact — end-to-end, on the enriched positions

Fact store recomputed on the enriched parquet (scratch; `outputs/facts/` untouched). The carry-OFF
column reproduces the known baselines exactly (passes 213/198, possession 44.2%), which validates the
harness.

| metric | OFF | ON | oracle (Sofascore) | Δ |
|---|---:|---:|---:|---:|
| passes Man Utd | 213 | **218** | 446 | +5 |
| &nbsp;&nbsp;pass-recall proxy | 47.8% | **48.9%** | — | +1.1 pp |
| passes Brighton | 198 | **203** | 407 | +5 |
| &nbsp;&nbsp;pass-recall proxy | 48.6% | **49.9%** | — | +1.2 pp |
| possession Man Utd | 44.2% | **44.4%** | 52.0% | +0.2 pp |
| &nbsp;&nbsp;\|error\| vs oracle | 7.8 pp | **7.6 pp** | — | −0.2 pp |
| possession samples | 4,157 | 4,278 | — | +121 |
| post-link ball coverage | 51.1% | 51.1% | — | 0.0 (control) |
| de-biased line, Man Utd | 33.0 m | 32.9 m | — | −0.1 m |

Pass recall rises ~1.2 pp but still misses report-v2's 50% ball gate (48.9 / 49.9%). Ball coverage is
unchanged, as expected: pose carry-over adds no ball detections, and the homographies it borrows are
the same ones `ball_carry` already had.

### The acid test — does it reduce the possession bias?

Possession moves **toward** the oracle (44.2 → 44.4%, oracle 52.0%) — the right direction, closing
2.6% of the 7.8 pp gap. Sharper form: the **newly recovered frames alone** carry Man Utd possession at

| sample | n | Man Utd share |
|---|---:|---:|
| pre-existing possession samples | 4,157 | 44.3% |
| **newly recovered by carry-over** | **121** | **47.9%** |
| oracle | — | 52.0% |

The recovered frames *are* more Man-Utd-heavy than the ones we already had, and sit between the
biased estimate and the oracle — **directionally exactly what the trackability-bias theory predicts**
(the missing frames are Man Utd's untrackable transition play). But **n = 121**: the binomial 95% CI
on 47.9% is ±8.9 pp, which contains the pre-existing 44.3%. So this is *consistent with* the theory,
**not statistically decisive**. It corroborates `results/possession_debias_probe.md`'s conclusion
without overturning it: the missing possession is real and informatively missing, and recovering a
couple of hundred frames of it is not enough. Only an event layer fixes this.

---

## 6. No-regression on the FIFA-validated metric — PASS

De-biased defensive line vs FIFA PMSR, 3 WC France matches, per-chunk fps from the registry (25 /
59.94 Hz). Carry ON must not worsen the pooled error.

| match | phases | OFF mean \|err\| | ON mean \|err\| | Δ |
|---|---:|---:|---:|---:|
| france_iraq | 6 | 4.9 m | 4.8 m | −0.1 |
| france_senegal | 6 | 5.7 m | 5.7 m | −0.0 |
| france_norway | 6 | 6.1 m | 5.9 m | −0.2 |
| **POOLED** | **18** | **5.5 m** | **5.5 m** | **−0.1 m** |

**PASS** — no regression; if anything a hair better on every match. The enriched frames are not
garbage. (The OFF pooled figure is 5.5 m, reproducing the current `tools/line_c6` output; the "5.2 m"
quoted in the brief is a stale number.)

---

## 7. Verdict

**DON'T SHIP as a coverage lever.**

- The mechanism is **sound and validated** — faithfulness passes the pre-committed bar (0.20 m median
  / 1.04 m p90), off-pitch rejections are ~0.2%, and it regresses nothing (WC line 5.5 → 5.5 m).
- But the **impact is negligible**: yield +1.7 pp, pass recall +1.2 pp (still under the 50% gate),
  possession +0.2 pp of a 7.8 pp gap. That does not justify a new code path in the pipeline, and the
  `calib_error_m` provenance relabelling it requires is a semantic hazard to carry for a ~2% gain.
- The hypothesis "the identical mechanism will lift every metric at once" is **falsified, with cause**:
  it worked for the ball because the ball needs *one* point projected and faces no frame-plausibility
  gate. Players need ≥ 8 on-pitch points spanning ≥ 25 m, and **90% of the frames we were trying to
  recover are close-ups holding 4 players**. There is nothing to project.

Keep `generator/pose_carry.py` as a validated, off-by-default capability (it is the natural partner to
a live-play filter, where every remaining frame *is* a wide shot and the addressable fraction is high).

**Redirect the effort.** This probe says the geometry bottleneck is **broadcast content, not pose
quality**, so:

1. **The live-play filter (Phase B) is the real lever** — and this probe re-prices it. It cannot
   create positions on close-ups, but it removes ~42% of frames from the *denominator*, which is what
   lifts conditional yield and pass recall past the 50% gate. It was already the plan; it is now the
   *only* plumbing lever with headroom.
2. **Possession bias stays uncorrectable without events**, exactly as `possession_debias_probe.md`
   concluded — now with a second, independent piece of evidence (the frames we *can* recover carry
   possession at ~48%, between the biased 44% and the oracle 52%, but there are only 121 of them).
3. **STATUS needs correcting.** The "91.8% solve / 37.4% yield" gap is not 54 pp of globally-wrong
   poses. It is 4.5 pp of bad poses and ~42 pp of close-ups. The correction that fixed the
   *denominator* mistake still mis-attributed the *cause*.
