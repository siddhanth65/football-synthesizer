# Why per-crop OCR went 4x blind on FOOTPASS game_18 — and what of it is ours

Date: 2026-07-30. Diagnosis of the negative recorded in `results/FOOTPASS_GAME18_SCORE.md` section 4
and claim `ident-027`: read density `d = 0.0304` on game_18 (Serie A 2019, Napoli-Milan) against
0.105-0.126 on our three ManU EPL matches, from **more** crops (166,930).

Everything below is measured on artifacts already on disk plus three small trials on **re-cut cached
crops** (no match re-run; the `_crops` working dirs had been deleted by `tools/ocr_match.py`, so
1,751-5,327 crops per match were re-cut from the chunk videos on CPU). Raw numbers:
`results/ocr_domain_shift.json`.

**One new capability made this diagnosis sharp:** FOOTPASS ships per-frame annotated player ROIs
*and* shirt numbers (`val_tactical_data.h5`, extracted from `data/footpass/raw/tactical_data_VAL.zip`
for this run). Matching our crops to those ROIs gives **the first ground-truth OCR precision this
project has ever had on a real broadcast**. The matcher is validated by an independent bit: our
kit-anchored team label agrees with the annotation's team on **96.1-97.7% of crops within each half**
(the label flips between halves, which is itself the two-mechanism team-bit check of
`FOOTPASS_GAME18_SCORE.md` section 7 reappearing).

---

## 0. Verdict first

| candidate cause | verdict | size |
|---|---|---|
| **legibility threshold shift** (mass just under 0.5, recoverable by recalibration) | **NO** | mass is at ~0; only 1.5% of crops sit in [0.3, 0.5) |
| **legibility mass collapse** (the model genuinely cannot see) | **YES** | 80.5% of crops score < 0.01 vs 60-67% on EPL |
| **crop physics** (Serie A source is softer at torso scale) | **YES, real, ~1.5-2x** | Laplacian variance 1.5-2.1x lower at *matched* crop height; 4.80 Mbps Constrained-Baseline source vs 5.97 Mbps High |
| **kit** (Milan red-black stripes) | **YES, strongly team-concentrated** | legibility 4.73% (Milan) vs 15.57% (Napoli), 3.29x; EPL teams differ by < 1.2x |
| **2-digit shirt numbers** | **YES, under-appreciated** | 86.8% of game_18 GT shirts are 2-digit; read precision 0.663 on those vs **0.961** on 1-digit, and 44% of wrong confident reads are exactly the *first digit* of the true number |
| **crop budget** | NO | game_18 got *more* crops per tracklet (median 20 vs 13/10) |
| **our own crop geometry** | **YES — and it is the single largest term** | `estimate_player_box` returns 0.814x the annotated player height; fixing it **2.56x**es confident reads on game_18 *at higher precision* |

**Headline: roughly half of the 3.5x crop-level gap is our bug, not the domain.** At the crop level
game_18 produces confident digits at 0.29x brighton's rate under the shipped crop box; with the box
corrected, at **0.57x**. The remaining gap is genuine domain (softness + striped kit + 2-digit
shirts).

**A rerun-worthy inference fix exists** (crop geometry — no model change, no retraining, and it helps
EPL too). It does **not** rescue the attribution result: projected `d` after the fix is 0.038-0.058
against the law's `d* = 0.347`. The trained identity model (`docs/GSR_CLUSTER_ROADMAP.md` S3/S4)
remains required.

---

## 1. Crop quality physics — the Serie A source really is softer, but only ~1.5-2x

3,114 game_18 and 5,327 brighton crops re-cut from the chunk videos at the exact
`(chunk, frame, track_id)` of the OCR pass. Both matches are 1920x1080 @ 25 fps.

| quantity (median) | game_18 | brighton |
|---|---|---|
| crop box height (px) | **91** | 85 |
| Laplacian variance, crop resized to 48x96 | **314** | 467 |
| torso-band RMS contrast | **34.6** | 47.3 |
| torso-band brightness | 104.9 | 123.4 |
| torso-band saturation | 134.3 | 108.5 |

game_18's crops are **larger** and still softer. Matched on box height the gap is monotone and
*widens* with player size — the opposite of a camera-distance effect:

| box height (px) | n (g18 / bri) | Laplacian var (g18 / bri) | torso RMS (g18 / bri) |
|---|---|---|---|
| 70-85 | 830 / 1,907 | 259 / 400 (1.55x) | 36.0 / 46.4 |
| 85-100 | 923 / 1,547 | 351 / 477 (1.36x) | 35.5 / 48.6 |
| 100-120 | 724 / 829 | 379 / 667 (1.76x) | 33.3 / 49.9 |
| 120-145 | 339 / 316 | 451 / 924 (**2.05x**) | 32.2 / 50.0 |

Source encodes (chunks are `ffmpeg -c copy` stream copies, so this is the *delivered* encode, not
ours): game_18 **4.80 Mbps, H.264 Constrained Baseline, I/P only** (5 I + 115 P in 120 frames);
brighton **5.97 Mbps, High profile, B-frames present** (96 B + 5 I + 19 P). Lower bitrate in a
markedly less efficient profile.

Radially-averaged full-frame FFT (4 frames each) shows **no resolution truncation on either** (no
cutoff knee, so game_18 is not an upscale); relative to its own low-frequency content game_18 carries
**35-40% less mid-band energy (0.15-0.50 Nyquist)** and slightly *more* above 0.55 Nyquist, the
signature of compression noise rather than detail.

**Read: physics is real and is worth ~1.5-2x, not 4x.**

## 2. Legibility model behaviour — mass collapse, not a threshold shift

| corpus | leg < 0.01 | 0.3 <= leg < 0.5 | leg >= 0.5 |
|---|---|---|---|
| **footpass_game_18** | **0.8045** | **0.0149** | **0.0963** |
| manutd_brighton | 0.6026 | 0.0333 | 0.2127 |
| manutd_liverpool | 0.6703 | 0.0222 | 0.2220 |
| manutd_tottenham | 0.6661 | 0.0244 | 0.2072 |

The distribution is bimodal in both domains and game_18's extra blindness is crops moving *into the
zero mode*, not crops piling up just under the bar. Dropping the offline floor 0.5 -> 0.3 would add
1.49% of crops (+15% relative), 0.5 -> 0.1 adds 3.7%. **A threshold recalibration cannot buy 4x.**

Worse, it cannot be done offline at all: the shipped reader *drops* crops at `leg <= 0.5` **before**
pose and PARSeq, so sub-threshold crops carry no read in the parquet. Testing it required re-running
the chain.

**Measured** (`leg_thresh` 0.5 -> 0.05, same 3,114 cached crops, same models):

| arm | crops legible | confident reads (p >= 0.99) | added-read precision vs GT |
|---|---|---|---|
| control (shipped 0.5) | 0.1024 | 0.0376 | — |
| **`leg_thresh = 0.05`** | 0.1477 | **0.0495 (+32%)** | **0.846** (22/26, at p >= 0.99) |

The added reads are good — but there are few of them, and `min_votes = 5` eats the gain. Binomial
projection over the real corpus (each non-confident crop gains a confident read with the measured
probability 0.0124, correct with probability 0.846): **`d` 0.0304 -> 0.0315.** Not worth a re-run on
its own.

## 3. The OCR head is *also* degraded, so it is not a single broken layer

PARSeq never abstains: every torso crop yields a number in both domains (torso|legible >= 0.997,
number|torso >= 0.998 everywhere). The degradation shows up in *confidence*:

| corpus | leg >= 0.5 | P(p_number >= 0.99 \| a number was read) | p99 over all crops | median confident votes per read tracklet |
|---|---|---|---|---|
| **footpass_game_18** | 0.0963 | **0.371** | 0.0356 | **3** |
| manutd_brighton | 0.2127 | 0.617 | 0.1310 | 5 |
| manutd_liverpool | 0.2220 | 0.610 | 0.1350 | 5 |
| manutd_tottenham | 0.2072 | 0.640 | 0.1322 | 5 |

**Both layers are broken**: 2.21x (legibility) x 1.66x (confidence) = **3.68x** at crop level; the
`min_votes = 5` tracklet bar amplifies that to the observed **4.15x** in `d`, because game_18 lands
3 confident votes per read tracklet where EPL lands 5.

### 3.1 The mechanism: 2-digit shirt numbers truncate

97,462 crops carry a GT shirt. On the 4,275 confident (p >= 0.99) reads with a strict GT match:

| GT shirt | n | per-crop read precision |
|---|---|---|
| 1-digit | 566 | **0.961** |
| 2-digit | 3,709 | **0.663** |

**86.8% of game_18's GT crops carry a 2-digit shirt** (Serie A 2019: 68, 77, 80, 81 in this fixture).
Of the 1,271 wrong confident reads, **562 (44.2%) are exactly the first digit of the true number**
(81 -> 8 alone accounts for 445) and 134 (10.5%) the second digit. This is a resolution-limited
second-digit drop, and it is a domain property: it is why 70 of game_18's 352 reads (20%) land on no
roster shirt, against 32/1,174 on brighton.

## 4. Kit — the blindness is strongly team-concentrated, and it suppresses rather than corrupts

Teams identified by median torso colour on the re-cut crops: our **team 0** = BGR (58, 93, 108),
hue 42 deg, V 107 — dark red/black = **Milan**; **team 1** = BGR (145, 137, 89), hue 188 deg, V 144 —
sky blue = **Napoli**.

| corpus | team | crops | legible | p >= 0.99 | tracklet read precision (shipped rule, GT) |
|---|---|---|---|---|---|
| **game_18** | **0 — Milan, red/black stripes** | 90,126 | **0.0473** | **0.0146** | **0.883** (60) |
| **game_18** | **1 — Napoli, sky blue** | 75,077 | **0.1557** | **0.0612** | 0.786 (238) |
| brighton | 0 | 58,577 | 0.2265 | 0.1530 | — |
| brighton | 1 | 56,956 | 0.2042 | 0.1119 | — |
| liverpool | 0 | 47,309 | 0.2500 | 0.1500 | — |
| liverpool | 1 | 46,080 | 0.1973 | 0.1222 | — |

**3.29x between the two Serie A kits; < 1.27x between any EPL pair.** Even Napoli alone (0.1557) is
below every EPL team, so the striped kit is a second, additive effect on top of the source physics —
not the whole story. Crucially Milan's *surviving* reads are the **more** precise ones (0.883 vs
0.786), i.e. the stripes cost density, not correctness. Goalkeepers: 0 of 163 GK tracklets read, as
in every previous match.

## 5. Rule recalibration is a dead end — measured against ground truth

126 aggregation rules (`min_crop_conf` x `min_votes` x `min_legibility`) applied to the real
166,930-crop corpus and graded against the GT dominant shirt of each tracklet (6,162 tracklets carry
one). Denominator 11,575 tracklets, as in `ident-027`.

| operating point | d | tracks | graded | **read precision** |
|---|---|---|---|---|
| **shipped 0.85-floor rule** (conf 0.99, votes 5, leg 0.5) | **0.0304** | 352 | 298 | **0.8054** |
| 0.80-floor rule (conf 0.90, votes 3) | 0.0952 | 1,102 | 940 | 0.6181 |
| 0.87-floor rule (conf 0.99, votes 5, leg 0.9) | 0.0173 | 200 | 167 | 0.7844 |

The Pareto frontier (best `d` at each precision floor, `graded >= 50`):

| precision floor | max d | rule |
|---|---|---|
| >= 0.85 | 0.0223 | conf 0.999, votes 4 |
| >= 0.80 | **0.0352** | conf 0.995, votes 4 |
| >= 0.75 | 0.0508 | conf 0.995, votes 3 |
| >= 0.70 | 0.0778 | conf 0.995, votes 2 |
| >= 0.60 | 0.1200 | conf 0.95, votes 2 |

The shipped rule is already within 16% of the frontier at its own precision. **No `min_legibility`
above 0.5 appears anywhere on the frontier.** For scale, the same `conf 0.995 / votes 4` rule on
brighton gives `d = 0.1432` — **4.07x**, so the whole frontier is shifted, not merely the operating
point.

**Two things this pins that were previously assumed:** the shipped rule's read precision on
FOOTPASS broadcast is **0.8054** (n = 298) — statistically of a piece with GSR TEST-38's 0.858 and
the EPL anchor-agreement 0.9217, so *the reads that survive are fine; there are simply 4x fewer of
them*. And the density claim of `ident-027` is confirmed with a precision number attached, which it
did not have.

## 6. Crop preprocessing — measured, and it is not the fix

Same 3,114 cached crops, same chain, one arm per preprocessing:

| arm | crops legible | p >= 0.99 | added reads where control abstained (precision) |
|---|---|---|---|
| control | 0.1024 | 0.0376 | — |
| CLAHE (clip 3.0, 4x4 on L) | 0.1092 | 0.0408 | 21 (0.33; 6 at p >= 0.99, 0.67) |
| 2x bicubic + unsharp | 0.1140 | **0.0373** | 29 (0.28; 7 at p >= 0.99, 0.43) |

Contrast enhancement moves legibility by +7-11% relative and moves confident reads **not at all**.
Upsampling does not create information that the encoder removed. **Neither is adopted.**

## 7. The finding: our crop box is the largest single term

`generator.team_anchor.estimate_player_box` reconstructs a player box from the persisted foot point
with two fixed constants (`0.035` far, `0.130` near, as fractions of frame height). Against 94,276
GT-matched crops on game_18 it is **too small**:

* median estimated height **90 px** vs annotated **110 px**; median ratio **0.814**;
* least-squares refit of the same linear model on the annotations: `h/H = 0.0615 + 0.0805 t`
  against the shipped `0.035 + 0.095 t` — the under-size is worst far from camera (1.30x at
  `t ~ 0.4`, 1.17x at the bottom of frame). Aspect: GT `w/h` median 0.465 vs shipped 0.45 (fine).

Legibility tracks box fit almost perfectly:

| IoU(our box, GT box) | n | legible | p >= 0.99 |
|---|---|---|---|
| 0.3-0.5 | 32,238 | 0.0320 | 0.0108 |
| 0.5-0.6 | 22,832 | 0.0904 | 0.0343 |
| 0.6-0.7 | 18,316 | 0.1718 | 0.0655 |
| **>= 0.7** | 15,807 | **0.3156** | **0.1212** |

At IoU >= 0.7 game_18 crops are *more* legible than the EPL corpus average. That is a strong hint,
not a proof (IoU also correlates with crowding), so it was tested directly.

### 7.1 The paired trial — same frames, same players, three crop geometries

1,751 crops, one OCR chain, three boxes: shipped `estimate_player_box`; the same box scaled **x1.25**
about the foot point; the **annotated ROI** (oracle).

| arm | legible | p >= 0.99 | precision of confident reads (GT) |
|---|---|---|---|
| `est` (shipped) | 0.1194 | 0.0388 | 0.691 (68) |
| **`est x1.25`** | **0.2439** | **0.0994 (2.56x)** | **0.730 (174)** |
| `gtbox` (oracle) | 0.2616 | 0.1062 (2.74x) | 0.742 (186) |

**2.56x the confident reads at *higher* precision, and x1.25 recovers 93% of the oracle's gain.**
Paired: 115 crops gain a confident read (precision **0.722**), 9 lose one (those 9 were 0.33
precise — i.e. the losses were mostly wrong anyway).

### 7.2 The EPL control — it is our estimator, not the Serie A camera

Same trial on 1,800 brighton crops (no GT there, so precision is not gradeable):

| arm | legible | p >= 0.99 |
|---|---|---|
| `est` (shipped) | 0.2106 | 0.1328 |
| **`est x1.25`** | **0.3372** | **0.1739 (1.31x)** |

Where both arms read confidently (n = 209), they return the **identical number 209/209** — the bigger
box does not change *what* is read, only *how often*. `P(x1.25 confident | shipped not)` is 0.0666 on
brighton and 0.0683 on game_18: **the absolute gain is the same in both domains**, so the estimator
under-sizes everywhere; the *relative* gain is 2.6x on game_18 only because its base is 3.4x lower.

Consequence for the diagnosis: the crop-level confident-read ratio game_18 : brighton is
**0.0388 / 0.1328 = 0.29** as shipped and **0.0994 / 0.1739 = 0.57** with the geometry fixed. **The
box accounts for about half the log gap.**

### 7.3 Projected `d` if the box is fixed

Binomial projection over the real corpus: each currently-non-confident crop gains a confident read
with the measured `q = 0.0683`, correct with the measured 0.7217; rule otherwise unchanged; the
`q = 0` arm reproduces the true 0.0304 / 0.8054 exactly as a control.

| arm | projected d | projected read precision |
|---|---|---|
| shipped box, shipped rule (control, reproduces reality) | 0.0304 | 0.8054 |
| **box x1.25, `min_votes = 5`** | **0.0384** | **0.849** |
| **box x1.25, `min_votes = 4`** | **0.0577** | **0.848** |
| box x1.25, `min_votes = 6` | 0.0278 | 0.860 |
| box x1.25 + `leg_thresh 0.05` | 0.0407 | 0.859 |
| sensitivity: added-read precision 0.60 instead of 0.7217 | 0.0363 | 0.839 |
| sensitivity: q = 0.05 instead of 0.0683 | 0.0349 | 0.833 |

**At a matched 0.85 precision floor the frontier moves from `d = 0.0223` to `d ~ 0.058` — 2.6x.**
Precision *rises* rather than falls, because the extra crops thicken existing majorities.

## 8. Recommendation

**A rerun-worthy inference fix exists: the crop box.** Not the x1.25 constant — that is the *probe*.
The right fix is to stop reconstructing the box at all where we do not have to, or to calibrate the
reconstruction:

1. **Cheapest, largest, safest:** widen the crop that `tools/ocr_match.py` cuts (it is a crop for a
   classifier, not a geometric estimate — nothing downstream consumes it). x1.25 is measured; the
   oracle says x1.25 is 93% of the ceiling. Cost: one OCR re-run per match (2 h GPU each).
   Expected: **+31% confident crop reads on EPL**, **+156% on game_18**, precision unchanged or
   better (209/209 read agreement on EPL, +0.04 measured precision on game_18).
2. **Root cause, wider blast radius:** `estimate_player_box`'s constants are a global two-number
   depth model fitted to nothing, and it is used well beyond OCR (`tools/gta_match.py`, the identity
   chain). It is 0.814x true on game_18. The detector's real boxes exist at extract time and are
   discarded; persisting a per-track median box height, or fitting the two constants per match from
   one detector pass, removes the guess. **Leave the knob per match — camera height/zoom differs by
   broadcast, which is exactly what this measurement shows.**
3. **Do not** spend on: legibility threshold recalibration (+0.001 `d`), CLAHE or upsampling
   (+0.000), a wider crop budget (game_18 already sits at the 20-crop cap with more crops per
   tracklet than any EPL match), or aggregation-rule re-tuning (+16% `d` at matched precision).

**What the fix does not do.** Projected `d ~ 0.04-0.06` is still **6-9x below** the `d* = 0.347` that
`results/EVIDENCE_DENSITY_LAW.md` requires for attribution coverage 0.50 at precision 0.85. On
`FOOTPASS_GAME18_SCORE.md`'s factorisation the naming channel would improve materially but the 0.85
precision floor would still not be reached at usable coverage. **The verdict of
`docs/GSR_CLUSTER_ROADMAP.md` S3/S4 — that a trained identity model is what crosses this wall —
stands, and is unaffected by this diagnosis.** What changes is that ~half of the *measured* game_18
deficit was ours, so the "domain wall" is smaller than `ident-027` implied, and the same fix is worth
having on our own EPL footage where nothing about Serie A applies.

## 9. Negatives, limits and things not measured

1. **The projections in 2 and 7.3 are simulations, not measurements.** The crop-level numbers
   (2.56x, 1.31x, precision 0.730) are measured on paired crops; the tracklet-level `d` numbers come
   from a binomial model whose only validation is that its `q = 0` arm reproduces reality exactly.
   Nothing downstream (attribution coverage, the solver, LOTO) was re-run.
2. **`n = 26` on the best sub-result of the legibility-floor arm** (added confident reads at
   precision 0.846). Wilson 95% is 0.66-0.94. It does not change the verdict because the projected
   `d` gain is 0.001 either way.
3. **The GT matcher is validated by team agreement (0.961-0.977 within half), not by identity.**
   If a crop is matched to the wrong player of the *same* team the read is graded against the wrong
   shirt; ~2-4% of matches are cross-team so the same-team error rate is plausibly of that order.
   All precision numbers here therefore carry a few points of matcher noise.
4. **43% of game_18's crops match no annotation at all** (replays, close-ups, unannotated frames)
   and those crops are half as legible (0.063 vs 0.122). They are counted in `d`'s denominator and in
   every crop-level rate, exactly as `ident-027` counted them.
5. **No EPL ground truth exists**, so section 7.2's control reports yield only. The claim "the fix is
   safe on EPL" rests on 209/209 read agreement, not on precision.
6. **The kit effect is confounded with the physics effect** — Milan's dark kit is also the
   lower-contrast one, and this diagnosis cannot separate "stripes confuse the classifier" from
   "dark kit + soft source = no edges". The team split is reported as measured.
7. **One game, one EPL comparison match.** games 24 and 47 were not run; tottenham/liverpool appear
   only in the corpus-level cascade, where they agree with brighton to within 8%.
8. `val_tactical_data.h5` (530 MB) was extracted to scratch for this run and deleted afterwards;
   it re-extracts from `data/footpass/raw/tactical_data_VAL.zip` in ~2 minutes.

## 10. Files

* `results/ocr_domain_shift.json` — every table above as raw numbers, including the full 126-rule
  GT-graded frontier.
* `tools/ocr_box_trial.py` — the paired crop-geometry trial of section 7.1/7.2 (the validator for
  recommendation 1; the oracle arm activates only when the FOOTPASS h5 is present).
* Inputs, unchanged: `outputs/{footpass_game_18,manutd_*}/final/ocr_percrop/*.parquet`,
  `data/footpass/raw/tactical_data_VAL.zip`.
* Claim: `ident-029`.
