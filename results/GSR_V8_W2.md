# v8 session W2 — the calibration-transfer probe: does any of their +10.28 apply to us?

Campaign v8, session W2. The leader (Broadcast2Pitch / KIST, WACV 2026, GS-HOTA 61.48 test) prices
its own homography module at **+10.28 GS-HOTA** (48.23 keypoint-only -> 58.51 keypoint+line,
`docs/WINNER_REPO_RECON.md` §7 Table 4). W1 closed the identity route: their reader is far weaker
than ours (0.34 vs 0.83 per-crop precision at matched emit), so their 61.48 must rest on
calibration, IDATR and the LLaMA shipping config. This session tests the calibration half by
measurement, not inference.

**The premise this session must not force.** Their +10.28 is measured off *their* weak
keypoint-only baseline. Ours is not weak: `results/GSR_CALIBGATE.md` already re-fought the
calibration battle (the trust-rule repair, +2.03 GS-HOTA on test-49, GS-LocA 93.40) and
`results/GSR_V7_V4S1.md` §3.1 measured the dropout closed (GT pitch-space recall 0.9155, frame
calibration rate 0.9807). The honest possibility — and the one the pre-registered read is built to
accept — is that our calibgate campaign already captured whatever their module is worth.

**CLAIMS-HYGIENE (plan v8, binding).** Their code and weights are **private measurement instruments
only**. The clone lives at `<scratchpad>/winner` (HEAD `dbfb65c0`, the post-Issue-#1 fix), the
checkpoint at `<scratchpad>/winner_weights`, the runner and every homography they produce at
`<scratchpad>/w2`. **Nothing of theirs enters the repo tree and nothing of theirs enters any shipped
artifact.** `tools/gsr_w2_calibswap.py` is ours: it consumes a directory of per-frame `3x3` `.npy`
files and knows nothing about who produced them.

---

## 1. REGISTRATION (written 2026-08-13, before any arm was built or scored)

### 1.1 The probe — 10 of DEV-20, and why these

Selection rule, fixed before it was applied: rank the 20 DEV sequences by (a) control GS-HOTA
(`results/gsr_benchmark/gsr_v7_control_dev.json`, the v6/v7 pinned-stack control, pooled 51.8083)
and (b) measured `loc_assoc` GS-AssA (`results/GSR_V7_V4S1.md` §3, partition E — the shipped
partition), **1 = worst on each**; take the 10 smallest rank sums. The two rankings agree on 9 of
10 members, so the composite is not doing subtle work; it exists only to break the one disagreement
(GS-HOTA alone would take SNGS-045, AssA alone SNGS-084) with a stated rule instead of a taste.

| seq | control GS-HOTA | rank | `loc_assoc` AssA | rank | identity acc | rank sum |
|---|---|---|---|---|---|---|
| SNGS-039 | 35.84 | 2 | 45.92 | 1 | 0.414 | **3** |
| SNGS-042 | 36.68 | 3 | 56.63 | 4 | 0.378 | **7** |
| SNGS-054 | 39.51 | 4 | 55.65 | 3 | 0.484 | **7** |
| SNGS-057 | 34.50 | 1 | 63.07 | 8 | 0.404 | **9** |
| SNGS-024 | 47.34 | 7 | 50.40 | 2 | 0.739 | **9** |
| SNGS-078 | 46.07 | 6 | 59.77 | 6 | 0.527 | **12** |
| SNGS-048 | 48.49 | 8 | 57.66 | 5 | 0.735 | **13** |
| SNGS-051 | 45.78 | 5 | 65.70 | 10 | 0.667 | **15** |
| SNGS-027 | 50.13 | 10 | 60.98 | 7 | 0.489 | **17** |
| SNGS-045 | 49.20 | 9 | 66.51 | 11 | 0.650 | **20** |
| — cut — | | | | | | |
| SNGS-084 | 51.34 | 13 | 63.16 | 9 | 0.586 | 22 |
| SNGS-033 | 50.77 | 12 | 68.10 | 12 | 0.656 | 24 |

**Probe = SNGS-{024, 027, 039, 042, 045, 048, 051, 054, 057, 078}.** Probe mean control GS-HOTA
**42.95** against DEV-20's 51.88 — by construction the hard half.

**Why worst-AssA/identity is the right probe for a calibration question.** GS-HOTA gates an identity
match at a **5 m** pitch distance. A position error smaller than 5 m is invisible to GS-LocA's
tolerance curve in the sense that matters here but can still flip *which* GT row a detection binds
to, which is where association and identity errors are manufactured. If their calibration carries
information ours lacks, the sequences whose association is already failing are where a 1-3 m
position shift is most likely to change a match. The counter-argument is registered too: these are
also the sequences with the most fragmentation, which no calibration fixes — so a null result on
this probe is *not* automatically a null result on DEV-20. §4 addresses that explicitly.

### 1.2 The instrument

| item | identity | size |
|---|---|---|
| repo `yinmayoo185/SoccernetGSR` @ HEAD | `dbfb65c05c847e8e50baa47aa91ace960cc5e600` (2026-06-02, the post-Issue-#1 fix) | 27 MB |
| **`SoccernetGSR_EfficientNet_Best.pth`** | sha256 `3998b0a734c4c595c896d5491ee2f3e062ac24dc0e08ee2f19a00cbaded00cc9`, md5 `7a50bfb8b733bca19e4567e4cddfccb9` | **148,893,763 B** |
| `opencv-contrib-python` | `4.10.0.84`, installed to a scratchpad `--target` dir (their `dbfb65c0` fix needs `cv2.ximgproc.thinning`; the repo env has plain `opencv-python`) | 45 MB |

Their `kpts.predict` is called **unmodified**. Two local interventions, both declared:

1. `numpy.save` is redirected. Their code writes each frame's homography *into the SoccerNet
   `img1/` folder* next to the frames; the redirect sends it to `<scratchpad>/w2/homog/<SEQ>/`
   instead. No numeric effect.
2. A per-frame branch tag is recorded by wrapping
   `estimate_inv_homography_line_circle_points_middle_pair` and testing the emitted matrix for
   exact identity, giving three categories per frame: **line** (their line+circle+point
   Levenberg-Marquardt path), **point** (the DLT `cv2.findHomography` fallback = the Issue-#1
   failure mode, Table 4's 48.23 row), **identity** (`np.eye(3)`, their hard failure).

### 1.3 The coordinate convention, determined once and declared

Their template frame is metres with a margin: `x in [10, 115]`, `y in [5, 73]`
(`kpts.pitch_line_endpoints`), so pitch-corner coordinates are `(tx - 10, ty - 5)` and GSR's centred
frame is `(tx - 62.5, ty - 39)`. The axis *direction* is not documented, so it was measured once
before anything else: GT `bbox_image` bottom-middle points of SNGS-024 frames 1-20 (120 rows)
pushed through their homography land **median 0.414 m / p90 0.953 m** from the GT `bbox_pitch` under
`(tx - 62.5, ty - 39)` and **median 13.35 m** under a flipped `y`. The convention is fixed at the
unflipped form for the rest of the session. This is an axis check, not a tuning knob, and it is the
only number in this file produced before §1 was written.

### 1.4 The arms — one variable, three pre-declared swaps

Nothing upstream of positions moves: the same v6 extraction (`positions_v6det`, S4b detector +
ByteTrack), the same track ids, the same cached per-crop OCR evidence, the same CLIP per-detection
embeddings, the same box cache. Only the **pitch coordinates of each detection row** change, and
everything below them — EIoU re-association, the GTA connector at tau 0.450, the jersey votes, the
free team map, the MILP identity solver — is ours and is re-derived per arm.

| arm | positions |
|---|---|
| **CONTROL** | `positions_gate_v6det` — our shipped lineage: PnLCalib + the calibgate trust rule (3 players / 5 m) + `fill_calibration_gaps(max_gap=10)` |
| **T1 theirs-faithful** | their per-frame homography wherever it is usable (non-identity, non-singular), `NaN` elsewhere. No fill. |
| **T2 theirs + our fill** | T1, then `fill_calibration_gaps(max_gap=10)` |
| **T3 union** | our on-record coordinate wherever it is finite; theirs on the rows ours left `NaN` — the "does their calibration reach frames our gate cannot" arm |

All four go through `tools.gsr_eiou.run_point` at the frozen v6 point (`e=0.3, rounds=1,
w_app=0.5, app_max=0.30`, embedder `clip_v6det`, tau 0.450, reader variant `_v6_v6det`), which
clears the connector/bundle caches per arm, so no arm can inherit another's artifacts. The CONTROL
is **re-derived in this session on this machine**, not quoted; its per-sequence GS-HOTA must
reproduce `gsr_v7_control_dev.json` on all 10 probe sequences or the session is void.

### 1.5 THE PRE-REGISTERED READ

> **Mean paired per-sequence GS-HOTA gain over the 10 probe sequences, best of the three swap arms
> vs the re-derived CONTROL.**
>
> * **`>= +0.5`** — the gap is real: their calibration carries something our calibgate campaign did
>   not capture. W2 escalates to the paper-derived SFR build on the GPL `sn-banner` assets
>   (`docs/SOCCERNET_REPO_SWEEP.md`), **reported as a plan, not built this session**.
> * **`< +0.5`** — our calibgate already captured it. The question **CLOSES**, the negative is
>   banked, and W2's cost stays a probe.

Taking the **maximum over three arms** biases the read *toward* escalation; that is deliberate, so
that a negative verdict is the stronger of the two possible outcomes. Reported alongside, never
gating: the per-arm table, helped/hurt counts, Wilcoxon p, and the GS-DetA/AssA/LocA decomposition.

### 1.6 Reported, not gating

1. **Position deltas.** Per detection row, `||our pitch - their pitch||` for every row where both
   are finite: median / p90 per sequence, and the pooled agreement map.
2. **GT-anchored accuracy, paired.** Our player/GK detections matched to the nearest GT person in
   image space (accepted under 60 px); error in metres against that GT row's `bbox_pitch`, computed
   for our coordinate and theirs **on the identical matched rows**. This is the only table that can
   say which calibration is *right*, as opposed to different.
3. **Their failure behaviour on our probe:** the line / point / identity branch split per sequence
   (does the Issue-#1 fallback fire?), and their runtime per sequence.
4. **The bridged frames:** rows where our chain has no pitch coordinate (or has one only because
   `fill_calibration_gaps` interpolated a neighbour's homography) — does theirs have a real
   homography there, and is it any good against GT?

### 1.7 Guards

1. Selection is on §1.5 alone. Every other number is descriptive.
2. No TEST-38, no test-49, no submission, **no commit**, no `METRICS_VERSION` change.
3. Their artifacts stay outside the repo tree (§1.2). If a table required vendoring their code, the
   table would not be produced.
4. One GPU job at a time on the 4 GB laptop; no pytest suite while a pass runs.
5. If the probe result contradicts the task's premise, the contradiction is the deliverable.

---

## 2. RESULTS — VERDICT: the registered read **FAILS by 0.0117**, and the reason is not that their
## calibration is bad. It is that **ours is more accurate than theirs on this probe.**

**The decisive number: best swap arm mean paired gain = `+0.4883` GS-HOTA against a `>= +0.5` bar.**
The question CLOSES with the negative banked. It failed by a hair, so §2.9 states exactly what the
hair is and what it does and does not license.

Three findings, in order of how much they change the campaign's picture:

1. **Their calibration is not better than ours here — it is worse.** On 95,927 GT-matched player/GK
   detections, our on-record coordinate sits **0.475 m** from ground truth (p90 1.379, 99.72% inside
   the evaluator's 5 m gate); the same detections through *their* homography sit **0.536 m** away
   (p90 1.554, 99.05% inside the gate). 844 rows clear the 5 m gate under ours alone against 199
   under theirs alone. Wilcoxon over the paired rows: `p < 1e-300`.
2. **Where they beat us is coverage, not geometry, and it is worth `+0.4883`.** Their model emits a
   usable homography on **7,500 of 7,500** probe frames. Our chain leaves 1,693 detection rows
   (1.43%) with no pitch coordinate at all; theirs supplies 1,324 of them. Taking *only* those rows
   (arm `t3`) is the entire transferable effect: 8 sequences helped, 1 hurt by **0.002**, `p = 0.0078`
   — and it lands 0.0117 under the escalation bar.
3. **The per-sequence sign of the swap is fully explained by which calibration is more accurate on
   that clip.** Spearman between the GT-anchored p90 advantage and the end-to-end GS-HOTA delta over
   the 10 sequences: **+0.903 (p = 0.0003)**. There is no hidden systematic advantage in their
   method that our metric is failing to see; where their homography is better (SNGS-024, -048) the
   swap wins big, where ours is better (SNGS-039, -027, -054) it loses.

### 2.1 Harness control — exact

The control was re-derived this session on this machine through
`tools.gsr_eiou.run_point` at the frozen v6 point, restricted to the 10 probe sequences. Against
`results/gsr_benchmark/gsr_v7_control_dev.json` (the v7 V0 pinned-stack DEV-20 control):

| | |
|---|---|
| per-sequence GS-HOTA delta, all 10 | **0.0 at every printed digit** (`max_abs 0.0`) |

The probe is a strict subset of a 20-sequence run and reproduces it exactly, so the arm chain is
per-sequence separable and every delta below is like-for-like.

### 2.2 The instrument ran clean — and the Issue-#1 fallback never fired

7,500 frames, 10 sequences, laptop RTX 3050:

| | |
|---|---|
| frames with a homography | **7,500 / 7,500 (100%)** |
| **line-based path** (their LM over lines + circle + point pairs) | **7,500 (100%)** |
| **DLT point fallback** (the Issue-#1 failure mode, Table 4's 48.23 row) | **0** |
| **identity fallback** (`np.eye(3)`) | **0** |
| wall clock | **2,527 s = 42.1 min** (219-287 s/sequence, **0.34 s/frame**) |

Their `dbfb65c0` fix is doing what the maintainer said it does. Anyone reproducing their 48.32 is
not looking at the code that ships now — but note that "the calibration path never degrades" is a
statement about *robustness*, not accuracy, and §2.4 measures accuracy separately.

Load fidelity: `Unet(out_ch=98, num_lines=21)` loads `SoccernetGSR_EfficientNet_Best.pth` at
`strict=True` with no missing or unexpected key. Their code was called through `kpts.predict`
unmodified (§1.2).

### 2.3 Position deltas — the two calibrations agree to a third of a metre

Our detections through both homographies, every row where both are finite (116,422 rows):

| sequence | rows | median | p90 | p99 | share > 1 m | share > 5 m |
|---|---|---|---|---|---|---|
| SNGS-024 | 7,034 | 0.60 | 2.26 | 5.89 | 0.322 | 0.0142 |
| SNGS-027 | 11,743 | 0.47 | 1.12 | 4.90 | 0.127 | 0.0100 |
| SNGS-039 | 13,878 | 0.53 | 2.45 | 10.02 | 0.291 | 0.0414 |
| SNGS-042 | 13,958 | 0.22 | 0.55 | 1.45 | 0.027 | 0.0000 |
| SNGS-045 | 11,355 | 0.26 | 1.11 | 2.91 | 0.122 | 0.0016 |
| SNGS-048 | 14,750 | 0.29 | 1.12 | 2.11 | 0.121 | 0.0000 |
| SNGS-051 | 10,910 | 0.36 | 0.85 | 1.75 | 0.062 | 0.0004 |
| SNGS-054 | 10,807 | 0.44 | 1.51 | 9.83 | 0.186 | 0.0272 |
| SNGS-057 | 12,764 | 0.39 | 1.65 | 4.19 | 0.179 | 0.0067 |
| SNGS-078 | 9,223 | 0.34 | 1.08 | 3.62 | 0.112 | 0.0028 |
| **pooled** | **116,422** | **0.359** | **1.325** | **5.137** | **0.1489** | **0.0105** |

**Two independently-trained sports-field-registration systems, on broadcast frames neither trained
on, disagree by a third of a metre at the median.** The disagreement has a long tail — 1.05% of rows
differ by more than the evaluator's whole 5 m tolerance, concentrated in SNGS-039 (4.1%) and
SNGS-054 (2.7%), and §2.4 says those tails are theirs, not ours.

### 2.4 GT-anchored accuracy — the table that says which one is *right*

Our player/GK detections matched to the nearest GT person in image space (accepted under 60 px);
error in metres against that GT row's `bbox_pitch`, computed for both coordinate sets **on the
identical matched rows**.

| sequence | paired rows | ours median | theirs median | ours p90 | theirs p90 | ours <= 5 m | theirs <= 5 m |
|---|---|---|---|---|---|---|---|
| SNGS-024 | 6,046 | 0.78 | **0.50** | 2.49 | **1.45** | 0.9838 | **1.0000** |
| SNGS-027 | 10,715 | **0.40** | 0.52 | **1.20** | 1.47 | **0.9997** | 0.9897 |
| SNGS-039 | 11,362 | **0.35** | 0.59 | **1.00** | 2.74 | **0.9984** | 0.9583 |
| SNGS-042 | 11,841 | **0.36** | 0.40 | **1.12** | 1.16 | 0.9992 | **0.9994** |
| SNGS-045 | 8,231 | **0.40** | 0.41 | **1.02** | 1.13 | 0.9983 | **1.0000** |
| SNGS-048 | 11,944 | 0.52 | **0.48** | 2.05 | **1.31** | 0.9969 | 0.9969 |
| SNGS-051 | 8,797 | **0.66** | 0.86 | **1.41** | 1.74 | 0.9982 | **0.9998** |
| SNGS-054 | 8,861 | **0.53** | 0.68 | **1.26** | 1.83 | **0.9970** | 0.9745 |
| SNGS-057 | 10,108 | 0.62 | 0.62 | **1.38** | 2.22 | **0.9978** | 0.9945 |
| SNGS-078 | 8,022 | **0.40** | 0.41 | 1.08 | **0.94** | 0.9973 | **1.0000** |
| **pooled** | **95,927** | **0.475** | 0.536 | **1.379** | 1.554 | **0.99722** | 0.99049 |

| | |
|---|---|
| rows inside the 5 m gate under **ours only** | **844** |
| rows inside the 5 m gate under **theirs only** | 199 |
| paired Wilcoxon on the absolute errors | `p ~ 0` (mean 0.692 m ours vs 0.814 m theirs) |

**This is the finding that reframes the session.** The premise the campaign inherited — "their
+10.28 means their calibration is the thing we lack" — does not survive contact with our own probe.
Their homography module is a good one, and on 3 of 10 sequences it is the better one; pooled over
95,927 GT-matched detections it is **12% worse in mean error and 0.7 points worse at the 5 m gate**
than the PnLCalib + calibgate chain `results/GSR_CALIBGATE.md` shipped. Their +10.28 is real for
*their* pipeline because their baseline was keypoint-only DLT; it is not a gap we have.

### 2.5 THE DECISIVE TABLE — paired GS-HOTA on the probe

Every arm re-derived through one harness, same detections, same track ids, same OCR evidence, same
embeddings, same connector, same solver. **The EIoU re-association is provably unchanged by the
swap** (`relink_sequence` consumes boxes and appearance embeddings only, never `pitch_x`): all four
arms report **584 -> 1,323 tracks**. The variable is the pitch coordinate and nothing else.

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| **CONTROL** (ours) | **42.9371** | 31.3141 | 58.8769 | **93.6346** | 45.8127 |
| T1 theirs-faithful | 42.9191 | 31.0248 | **59.3747** | 93.2560 | 45.7413 |
| T2 theirs + our fill(10) | 42.9188 | 31.0245 | 59.3745 | 93.2539 | 45.7395 |
| **T3 union** | **43.2713** | **31.5168** | **59.4120** | 93.5717 | **46.1202** |

Per sequence, and paired:

| sequence | CONTROL | T1 | T3 | delta T1 | delta T3 | rows T3 adds |
|---|---|---|---|---|---|---|
| SNGS-024 | 47.343 | **59.178** | 50.346 | **+11.836** | +3.004 | 336 |
| SNGS-027 | 50.130 | 47.842 | 50.165 | -2.288 | +0.035 | 46 |
| SNGS-039 | 35.844 | 29.639 | 35.848 | **-6.205** | +0.004 | 5 |
| SNGS-042 | 36.685 | 36.444 | 36.685 | -0.241 | **0.000** | **0** |
| SNGS-045 | 49.198 | 49.485 | 49.245 | +0.288 | +0.047 | 9 |
| SNGS-048 | 48.492 | 52.397 | 48.574 | +3.906 | +0.082 | 24 |
| SNGS-051 | 45.778 | 44.533 | 46.049 | -1.245 | +0.271 | 59 |
| SNGS-054 | 39.508 | 37.317 | 39.506 | -2.191 | -0.002 | 49 |
| SNGS-057 | 34.499 | 32.878 | 35.019 | -1.621 | +0.519 | 731 |
| SNGS-078 | 46.067 | 48.517 | 46.989 | +2.449 | +0.921 | 65 |

| arm | mean | median | helped | hurt | worst | best | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| T1 | +0.4687 | **-0.7430** | 4 | 6 | -6.205 | +11.836 | 1.000 |
| T2 | +0.4687 | -0.7430 | 4 | 6 | -6.205 | +11.836 | 1.000 |
| **T3** | **+0.4883** | +0.0649 | **8** | 1 | **-0.002** | +3.004 | **0.0078** |

| the pre-registered read | required | measured | |
|---|---|---|---|
| mean paired GS-HOTA gain, best of three arms | **`>= +0.50`** | **`+0.4883`** (T3) | **FAIL** |

**VERDICT: `< +0.5`. Our calibgate campaign already captured it. The question CLOSES, the negative
is banked, and W2's cost stays a probe** (1.2 GPU-h of a 3-5 h budget). No SFR build is
commissioned. §2.9 records what the 0.0117 margin does and does not license.

**T3 is internally validated by its own dose-response.** SNGS-042 is the one probe sequence where
their homography adds **zero** rows our chain lacks, and its T3 delta is exactly **0.000**; SNGS-039
adds 5 rows and moves +0.004. An arm that only writes rows we had none for moves the score only
when it writes rows.

### 2.6 Mechanism — the swap's per-sequence outcome is a calibration-accuracy read-out

Rank correlation over the 10 probe sequences between each sequence's GT-anchored accuracy advantage
(theirs minus ours) and its T1 end-to-end GS-HOTA delta:

| accuracy statistic | Spearman | p | Pearson |
|---|---|---|---|
| median error advantage | +0.794 | 0.0061 | +0.933 |
| **p90 error advantage** | **+0.903** | **0.0003** | +0.903 |
| share-within-5 m advantage | +0.855 | 0.0016 | +0.800 |

The two big movers decompose exactly as that predicts:

| sequence | arm | GS-DetA | GS-AssA | GS-LocA | GT p90 |
|---|---|---|---|---|---|
| SNGS-024 | control | 42.53 | 52.71 | 90.44 | 2.49 |
| SNGS-024 | **T1 (theirs)** | **52.87** | **66.24** | **94.34** | **1.45** |
| SNGS-039 | control | **26.25** | **48.95** | **95.90** | **1.00** |
| SNGS-039 | T1 (theirs) | 21.33 | 41.20 | 92.71 | 2.74 |

**The team-side map is not the mechanism.** `resolve_team_map_free` returns the GT-correct side on
**10 of 10** probe sequences in *every* arm, with margins that move by at most 0.34 m
(the one exception, SNGS-054, moves 0.42 -> 0.08 m under T1 and still resolves correctly). The
+11.8 and -6.2 swings are ordinary localisation quality propagating into the 5 m identity gate, not
the coin-flip failure `GSR_DELEAK.md` negative #2 and `GSR_CALIBGATE.md` §4 have been watching.

### 2.7 The bridged frames — the one place their calibration is clearly better than ours

Rows on frames whose control coordinate does **not** come from that frame's own homography (i.e.
frames `positions_v6det` left dead, whose coordinate exists only because `fill_calibration_gaps`
borrowed a neighbour's geometry):

| sequence | GT-matched people | ours: n / median / <=5 m | theirs: n / median / <=5 m |
|---|---|---|---|
| SNGS-024 | 710 | 469 / **1.97** / 0.9424 | **703 / 0.45 / 1.0000** |
| SNGS-051 | 44 | **2** / 2.08 / 1.0000 | **44 / 0.57 / 1.0000** |
| SNGS-078 | 138 | 122 / 0.65 / 0.9754 | **134 / 0.39 / 1.0000** |
| SNGS-045 | 170 | 160 / 0.40 / 1.0000 | 160 / **0.27** / 1.0000 |
| SNGS-048 | 46 | 45 / 0.52 / 1.0000 | 45 / **0.45** / 1.0000 |
| SNGS-039 | 17 | 16 / 0.86 / 1.0000 | 16 / **0.78** / 1.0000 |
| SNGS-027 | 9 | 8 / 0.55 / 1.0000 | 8 / **0.46** / 1.0000 |
| SNGS-054 | 198 | 174 / **0.51** / 0.9828 | 186 / 0.84 / **1.0000** |
| SNGS-057 | 943 | 289 / **1.47** / **0.9896** | 849 / 2.57 / 0.9377 |
| SNGS-042 | 0 | — | — |

**A real per-frame homography beats an interpolated neighbour's on 8 of 9 sequences that have
bridged frames**, and on SNGS-024 it is not close: 1.97 m -> 0.45 m and 94.2% -> 100% inside the 5 m
gate, over 703 rows. This is the same effect `GSR_CALIBGATE.md` §2 measured from the other side
(the gate beat the post-hoc fill on GS-LocA because "a frame's own homography is not blurrier than
its neighbour's"), now measured against an external calibrator that has a homography where we have
none.

Coverage, pooled over the probe: 118,224 detection rows, of which our control carries a pitch
coordinate on 116,531 (98.57%) and T3 on 117,855 (99.69%). Their calibration reaches **1,324 of the
1,693 rows (78.2%)** our chain leaves empty. It also *loses* 109 rows our control has, because
`clamp_to_pitch` rejects them as off-pitch: 478 of their 118,224 projections (0.40%) land outside
the pitch polygon plus tolerance, against our chain's own rejections already applied upstream.

### 2.8 Their calibration's failure behaviour on our probe — there isn't one

| | |
|---|---|
| frames with no homography | 0 |
| frames on the DLT point fallback | 0 |
| frames on the identity fallback | 0 |
| projections rejected off-pitch by our `clamp_to_pitch` | 478 of 118,224 (0.40%) |
| worst sequence for off-pitch rejection | SNGS-027, 197 rows (1.6%) |
| runtime | 0.34 s/frame on an RTX 3050 (4 GB), 100% single-GPU, Windows-clean |

Robustness is the honest half of their +10.28: the module never falls over, where PnLCalib returns
nothing on a real fraction of frames (`GSR_CALIBGATE.md` §1: 508 of 3,853 dead frames on the valid
worst-8 had **no hypothesis at all**). §2.7 is what that robustness is worth to us.

### 2.9 What the 0.0117 margin does and does not license

The registered bar was `>= +0.5`; T3 measured `+0.4883`. Three things are worth stating plainly so
the orchestrator is not deciding on a headline:

1. **The verdict stands as registered.** No arm is re-run, no threshold is moved, no fourth arm is
   invented to cross the line. The bar was set with a deliberate pro-escalation bias (max over three
   arms) and it was still not cleared.
2. **The effect T3 measures is not "their SFR model".** It is *any* calibration that emits a
   homography on the 1.43% of rows where PnLCalib returns nothing. Their model is one way to get
   that; the sn-banner NBJW assets are another; a second-pass re-solve on our own dead frames with
   a relaxed keypoint minimum is a third and costs us no new dependency. If the orchestrator wants
   the +0.49, **the cheapest route is not the SFR build the escalation clause names.**
3. **This probe is the campaign's hard half, and it may flatter the coverage effect.** Probe mean
   control GS-HOTA is 42.94 against DEV-20's 51.88, and 2 of the 10 sequences (SNGS-024, SNGS-057)
   carry 80% of the recoverable rows. On the other 10 DEV sequences — untested — the coverage
   headroom is likely smaller, not larger, because they are the ones whose calibration is already
   healthy.

---

## 3. The escalation clause, and why it is not triggered

The plan's escalation was: *"if a real gap appears: the SHIP path is paper-derived SFR via the
never-used sn-banner assets (curl-able NBJW/Mask2Former weights, GPL = server-runnable,
`docs/SOCCERNET_REPO_SWEEP.md`) — not their code."* **It is not triggered** (§2.5). For the record,
what the measurement says about that path if the orchestrator ever revisits it:

* **A paper-derived SFR build would be aiming at the wrong target.** §2.4 says their *estimator* is
  not more accurate than ours on this data. Rebuilding it from the paper buys a second estimator of
  the same quantity — the exact shape of the negative campaign v7 V2 recorded for the learned
  side-detector ("a second estimator of the SAME per-frame pitch-geometry quantity, not a new
  signal"). The expected value of that build is the *coverage* term, not the geometry term.
* **The coverage term has a cheaper owner.** T3's `+0.4883` comes entirely from rows where PnLCalib
  returns nothing. `GSR_CALIBGATE.md` §1 already located and sized that population (13.2% of dead
  frames have **no hypothesis at all**; 508 of 3,853 on the valid worst-8, 320 of them in one
  sequence) and §6 negative #2 recorded that walking PnLCalib's other 17 hypotheses does not reach
  them, because all 18 come from the same keypoint heatmaps. So the honest follow-up, in order of
  cost:
  1. **Free:** on frames with no hypothesis, fall back to a *line-fit* solve rather than a keypoint
     DLT — the structural difference between their module and ours, and the one their Table 4
     prices at +8.16 on its own (48.23 keypoint-only -> 56.39 line-only). We have no line-based
     solver at all. This is the single named, measured, unexploited idea from their paper.
  2. **~2 GPU-h:** the sn-banner NBJW keypoint/line assets used *only* as a fallback calibrator on
     dead frames, not as a replacement — a coverage patch with a measurable ceiling of about +0.49
     on this probe.
  3. **Not recommended:** a full paper-derived SFR retrain to replace PnLCalib. §2.4 is the reason.
* **A note for the thesis, not for the board.** The comparison in §2.4 is itself a result: our
  calibration chain, assembled from a public PnLCalib checkpoint plus a trust-rule repair we
  derived and validated ourselves, is **more accurate on GSR validation broadcast frames than the
  sports-field-registration module of the current benchmark leader** — measured on 95,927
  GT-matched detections, on the leader's own weights, with their code run unmodified. That is a
  defensible, checkable statement and it did not exist before this session.

## 4. Negatives, limits, and what was NOT done

1. **The registered read FAILED, by 0.0117 GS-HOTA.** It is a hair, and a hair is a fail. Nothing
   was re-run to move it (§2.9).
2. **T1 — the faithful swap — is a coin toss, not a loss and not a win.** Mean `+0.4687` but median
   `-0.7430`, 4 helped / 6 hurt, Wilcoxon `p = 1.000`. Its mean is one sequence (SNGS-024, +11.84)
   against another (SNGS-039, -6.21). **Anyone quoting T1's `+0.47` as "their calibration is worth
   half a point to us" is quoting a single clip.** T3's `+0.4883` is the one with a distribution
   behind it (8/1, `p = 0.0078`).
3. **The probe is 10 sequences chosen for being hard.** §1.1 registered the reason and §2.9 the
   consequence. **No DEV-20 number, no TEST-38 number, no test-49 number, no submission** was
   produced. A DEV-20 read on the same instrument would cost another ~0.7 GPU-h and was not spent,
   because the verdict does not change sign at the probe's own bar.
4. **Their calibration was never given our trust rule or our fill on its own terms.** T1/T2 apply
   *our* `clamp_to_pitch` to *their* geometry, which rejects 478 of their projections. That is the
   correct treatment for an arm inside our chain, but it is not a measurement of their pipeline's
   own coverage, and this document makes no claim about their end-to-end score.
5. **The GT matcher is an image-space nearest-neighbour under 60 px, not the evaluator's Hungarian
   assignment.** It is identical for both coordinate sets by construction (§1.6), so the *paired*
   comparison is safe; the absolute levels (0.475 m / 0.536 m) inherit whatever bias a 60 px gate
   has. Rows with no GT partner (about 19% of people rows) are excluded from both sides.
6. **The axis convention was fixed on 120 rows of one sequence** (§1.3). A 13.35 m vs 0.414 m
   separation makes it unambiguous, but it is one check, not a sweep.
7. **Detections, tracks and every GPU artifact below positions were held fixed.** A calibration
   change in a *fresh extraction* could move detection-time decisions this experiment cannot see
   (`generator/extract.py` now calibrates after detection and feeds the gate its foot points). The
   re-extraction was not run; `GSR_CALIBGATE.md` negative #5 is the same caveat.
8. **Process fault, on the record.** The first instrument launch used `nohup ... &` inside a
   backgrounded shell; the shell returned, the child survived, and a second launch ran
   **concurrently with the first** for ~25 minutes, both writing the same `homog/` tree. Both were
   killed, `homog/` was deleted, and **every homography in this document comes from a single clean
   run** (§2.2's 2,527 s). The wasted GPU is counted in the spend below. No result was computed from
   the contaminated tree.
9. **GPU spend: 1.2 h** (42 min clean instrument run + ~25 min of the killed concurrent pair + a
   20-frame smoke test) of a 3-5 h budget. CPU: 4 min 55 s for all four scored arms, ~70 s for the
   positions/deltas pass. No cluster time; the a100server1 host was unreachable this session
   (`ssh 192.168.3.19` timed out), and nothing here needed it.
10. **No commit, no `METRICS_VERSION` change, no TEST-38, no test-49, no submission**, and nothing
    of theirs anywhere in the repo tree.

## 5. Files

**Ours (repo, NOT committed this session):**

- `tools/gsr_w2_calibswap.py` (new): the external-calibration swap (`swap_positions`, `build_arms`),
  the GT-anchored paired accuracy instrument (`gt_people`, `gt_matched_error`), the delta report,
  the four-arm runner, the on-record control check, and a `--demo` self-check on the swap arithmetic,
  the origin shift, the identity-sentinel filter and the GT matcher.
- `results/GSR_V8_W2.md` (this file).
- `results/gsr_benchmark/gsr_v8_w2_positions.json` — per-sequence row accounting for all three swap
  arms.
- `results/gsr_benchmark/gsr_v8_w2_deltas.json` — position deltas, GT-anchored accuracy (all rows
  and bridged-frame subsets), their per-frame branch split and runtime.
- `results/gsr_benchmark/gsr_v8_w2_arms.json` — the four scored arms, per-sequence GS-HOTA, paired
  statistics.
- `outputs/gsr/positions_w2t{1,2,3}_v6det{,_eiou}/`, `outputs/gsr/deleak_w2_{control,t1,t2,t3}/`,
  `outputs/gsr/w2_arms.log` (gitignored artifacts). No on-record artifact was overwritten.

**Theirs / instrument (scratchpad only — never in the repo tree, never in a shipped artifact):**

- `<scratchpad>/winner/` — their repo @ `dbfb65c0` (already present from W1);
- `<scratchpad>/winner_weights/SoccernetGSR_EfficientNet_Best.pth` — sha256 `3998b0a7...d00cc9`;
- `<scratchpad>/oc/` — `opencv-contrib-python 4.10.0.84`, isolated `--target` install;
- `<scratchpad>/w2/winner_calib.py` — the runner (np.save redirect + branch tagging);
  `<scratchpad>/w2/homog/<SEQ>/{000001..000750}.npy` + `_branches.json` — 7,500 homographies;
  `<scratchpad>/w2/w2_calib.log`.

## 6. Reproduce

```
# 1. instrument (scratchpad; NOT in the repo)
pip install --target <scratchpad>/oc --no-deps opencv-contrib-python==4.10.0.84
python -c "import gdown; gdown.download(id='1xnWOQDizjrjyUoEwwLssjaGv8LkEpuQl', \
    output='<scratchpad>/winner_weights/SoccernetGSR_EfficientNet_Best.pth')"
python <scratchpad>/w2/winner_calib.py --seqs SNGS-024,...,SNGS-078          # GPU, 42 min

# 2. ours
python -m tools.gsr_w2_calibswap --positions --deltas --homog <scratchpad>/w2/homog   # CPU, ~70 s
python -m tools.gsr_w2_calibswap --arms                                              # CPU, ~5 min
python -m tools.gsr_w2_calibswap --demo
pytest tests/test_calibrate.py tests/test_postprocess.py
```

## 7. What this means for the campaign (for the orchestrator, not a result)

- **The calibration route is closed by measurement, and it closed in the most useful way**: not
  "we tried and it did not transfer" but "we are already ahead on this component". W1 closed
  identity the same way. Their 61.48 therefore rests on the two things this campaign has *not*
  priced: **IDATR (+2.97 by their own ablation)** and the **detector/tracker substrate** their
  ablation holds fixed and never prices at all.
- **The one idea of theirs that is still unexploited is line-based homography estimation**, not the
  SFR network: their Table 4 prices line-only at 56.39 against keypoint-only 48.23, and our chain
  has no line solver. §3 lists it as the free follow-up, aimed at the frames where PnLCalib returns
  nothing rather than at replacing PnLCalib.
- **The largest remaining gap is still association** (`GSR_V7_V4S1.md`: +11.89 GS-AssA from a
  merge-only oracle, +24.79 from a perfect linker, 19/20 sequences fragmentation-dominated). Two
  sessions have now confirmed by measurement that the leader's advantage is not in the two
  components we borrowed their instruments to test.

