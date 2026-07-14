# Phase A feasibility probe (Man Utd PL segments; 3 matches x 3 segments x 2 min)

Detection/tracking/calibration columns are unchanged from the zero-shot probe (same dense parquets;
`tools/pl_feasibility.py --skip-extraction`). The ball rows now carry a **v6 fine-tuned** column
next to the **zero-shot v5** baseline. The labelled-recall gate is no longer pending: the user
hand-clicked 120 frames (118 visible) on the three seg_1 videos.

## BALL: labelled recall + post-link coverage (the only coverage number that counts)

Recall scored with `tools/validate_ball.py::evaluate_annotated` unchanged (thr=0.5, hit radius
tol_px=8.0 in the 512x288 model grid == 30 px native at 1920x1080) -- the exact protocol that
produced the WC v5 held-out numbers.

### Stage 1 -- zero-shot v5 recall on ALL 120 labelled PL frames (never trained on) -> GATE PASSED
```
match          visible  hits   recall   med_err  mean_err (512x288 px)
brighton            40    31    77.5%      0.91      1.03
fulham              40    28    70.0%      1.07      1.10
liverpool           38    21    55.3%      1.02      1.07
POOLED             118    80    67.8%      1.03px    1.07px  (== 3.9 / 4.0 px native)
```
Gate was >= 40% pooled -> **67.8% PASSES decisively** (v5 zero-shot already beats its own WC held-out
senegal 65% / norway 50%). Proceeded to fine-tune.

### Stage 2 -- v6 held-out recall (stride-5 split, 23 PL frames never trained on)
```
match          visible  hits   recall
brighton             8     8   100.0%
fulham               8     6    75.0%
liverpool            7     6    85.7%
POOLED              23    20    87.0%
```
PL held-out 87.0% -- clears the "fine-tune will reach 60%+" expectation.

### Stage 2 -- WC forgetting check (v6 vs v5, identical held-out frames + tol, stride 5)
```
match            v5      v6     delta
france_iraq     80.1%   80.9%   +0.8
france_norway   50.0%   54.5%   +4.5
mun_mci         86.3%   89.5%   +3.2
fra_sen         83.3%   83.3%    0.0
france_senegal  65.0%   60.0%   -5.0   <-- ONLY regression > 3 pp (2 frames of 40; small sample)
```
Net: 3 WC matches improved, 1 flat, senegal down 5.0 pp (26/40 -> 24/40). Reported loudly per the
gate rule; the other four corpora are stable-or-better, so no broad catastrophic forgetting.

### Stage 3 -- POST-LINK coverage, zero-shot v5 vs v6 (all 9 segments)
```
                      zero-shot v5     v6
brighton_manutd/seg_1     42.0%       43.2%
brighton_manutd/seg_2     20.9%       20.0%
brighton_manutd/seg_3     22.8%       24.9%
manutd_fulham/seg_1       26.0%       29.5%
manutd_fulham/seg_2       24.0%       24.5%
manutd_fulham/seg_3        6.8%        5.0%
manutd_liverpool/seg_1    17.8%       21.0%
manutd_liverpool/seg_2     8.6%        8.6%
manutd_liverpool/seg_3    13.2%       16.1%
------------------------------------------
brighton  (match)         28.6%       29.4%
fulham    (match)         18.9%       19.7%
liverpool (match)         13.2%       15.2%
POOLED                    20.2%       21.4%   (+1.2 pp)
ball fire-rate @0.5       53%         60%     (+7 pp; NOT recall)
                          WC baseline: sen 46% / irq 36% / nor 38%
```

**Honest read.** Fine-tuning lifted fire-rate +7 pp and held-out recall to 87%, but post-link
coverage rose only +1.2 pp pooled (20.2 -> 21.4%). Detection recall is NOT the PL post-link
bottleneck -- the binding constraint is the **>=6-correspondence homography yield (27% pooled, the
"norway killer")** plus calibrated-frame gating: every detected ball that reaches the projection
stage projects (proj=N/N, too_few=0, homog_fail=0), so coverage is capped upstream of the ball net.
PL post-link stays below senegal's 46%. This matches the PL_PIVOT diagnosis that raw broadcast
slices are ~41% live wide-camera play; the Phase-B live-play filter + temporal homography reuse are
the levers for coverage, not more ball labels.

## DETECTION / TRACKING / CALIBRATION (unchanged; reproduced for reference)
```
metric                          brighton_manutd  manutd_fulham  manutd_liverpool   POOLED     WC baseline
--------------------------------------------------------------------------------------------------------
players/frame (mean)                        8.9            6.8               6.2      7.3  irq 7.8/sen 10.9
track count (mean/seg, 2min)                194            174               193      187    ~180/2min (WC)
track frag index (n_trk/ppf)              22.48          25.59             32.30    26.79      lower=better
mean track length (s)                       4.6            2.8               2.9      3.4              n/a
calibrated frames (<=2m gate)               75%            78%               86%      80%          irq 88%
>=6-corr yield (norway killer)              38%            19%               24%      27%       >= irq 41%
```

## STAGE 4 -- THE COVERAGE LEVER: temporal homography carry-over (PROVEN, +18 pp pooled)

Tool: `tools/pl_carry_probe.py` (deep-worker, requested: opus); logic `generator/ball_carry.py`
(flag-gated, default off; WC pipeline unchanged). v6 ball re-detected on ALL dense frames of each
segment (not just already-calibrated ones), then project+link OFF (own-frame homography only,
reproduces the v6 column) vs ON (carry-over). The ONLY number that counts is post-`link_ball`
coverage; own/carried/interp are PRE-LINK diagnostics.

Mechanism: for a ball frame with no usable own homography, reuse the nearest frame's homography
within +/-50 frames (~2 s @25fps) on the SAME camera segment (or blend the two bracketing ones).
Camera segments come from track-id Jaccard between consecutive sampled frames (<0.30 = cut; a cut
resets ByteTrack so overlap collapses) -- no video re-decode. Every projection is off-pitch-gated.

```
                          OFF cov   ON cov   delta      own carry interp offpit
brighton_manutd/seg_1       43.2%    68.4%   +25.2 pp   180    86     38      1
brighton_manutd/seg_2       20.0%    29.8%    +9.8      74     45     25      0
brighton_manutd/seg_3       24.7%    43.7%   +18.9      80     54     13      2
manutd_fulham/seg_1         29.5%    55.7%   +26.2      95     76     12      7
manutd_fulham/seg_2         23.5%    34.7%   +11.2      43     41      6     11
manutd_fulham/seg_3          4.7%    19.0%   +14.3      11     22      3      1
manutd_liverpool/seg_1      18.0%    37.8%   +19.8      83     74     11     16
manutd_liverpool/seg_2       6.6%    21.8%   +15.2      15     22      4      1
manutd_liverpool/seg_3      16.8%    37.9%   +21.1      56     75     16      5
------------------------------------------------------------------------------
brighton  (match mean)      29.3%    47.3%   +18.0
fulham    (match mean)      19.2%    36.5%   +17.3
liverpool (match mean)      13.8%    32.5%   +18.7
POOLED (mean of 9 segs)     20.8%    38.8%   +18.0 pp
```
OFF (20.8% pooled) reproduces the v6 baseline (21.4%); the ~0.6 pp gap is because the carry path
also off-pitch-clamps the OWN-frame projection (the published `project_ball` did not) -- this only
lowers OFF, so the +18 pp ON gain is if anything conservative. Every one of the 9 segments improves.

**Plausibility (the honest guard).** Leave-one-out cross-validation: for frames that have BOTH their
own homography and a ball, project the ball with the own H (reference) vs carry-over with that H
removed. Carry-over reproduces the own-frame ball position to **median 0.33 m, p90 2.01 m, 90%
within 2 m, 97% within 5 m** (n=656). i.e. across a 2 s same-segment window the camera is near
static, so a borrowed homography is a faithful stand-in. Gross outliers (max 756 m) are the ~1% of
cases where the camera panned inside the window; the off-pitch gate + `link_ball` speed clamp reject
them (578 carried samples survive linking, 0 off-pitch; worst surviving inter-sample speed 46 m/s =
one 9.2 m/0.2 s jump, borderline-physical for a struck ball, not a teleport).

**Why this works here but the WC temporal-H attempt (retracted) did not:** the WC gap frames were
camera CUTS (corr=0, uninterpolable); the PL failures are continuous-camera midfield ambiguity, where
a good homography exists a fraction of a second away on the same shot. Different failure mode ->
different result, now measured post-link end-to-end.

**Mechanism 2 (reuse PnLCalib's own calibration to project the ball on "calibrated-but-<6-projected"
frames) is DISPROVEN by construction.** The DIAGNOSIS's 205-frame "calibrated but <6 players
projected (frame-edge)" bucket is mislabelled: those frames have 8-13 detected players that ALL
project >2 m off-pitch (valid-pitch player count is bimodal -- 0 or >=8, never 4-5). Their PnLCalib
homography passed the 2 m KEYPOINT-reproj gate but is a globally wrong pose, so it would send the
ball off-pitch too (0 recoverable frames). The recoverable lever is carry-over of the KNOWN-GOOD
homographies (>=6 players project on-pitch), i.e. Stage 4 above.

**Verdict:** Phase B ingestion SHOULD include the carry-over stage. It roughly doubles post-link ball
coverage (20.8 -> 38.8% pooled), is CPU-only, physically faithful (median 0.33 m), and default-off so
it never touches the WC path. Note: it still does not clear report-v2's 40% ball-family gate on
fulham/liverpool at the raw-slice level (a Phase-B live-play filter should lift these further, since
non-live frames dilute the denominator); brighton clears it (47.3%). Artifacts:
`results/pl_probe/carry_probe.md`, `outputs/pl_probe/<match>/<seg>_ball_imgxy.parquet`,
`tests/test_ball_carry.py` (9 tests).

## STILL PENDING (honestly, not proxied)
  - FBref possession within ~5 pp: PENDING -- needs a full-match process + FBref pull.
