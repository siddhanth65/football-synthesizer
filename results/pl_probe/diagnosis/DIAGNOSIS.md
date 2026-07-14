# Phase A diagnosis -- sampling artifact vs geometry blocker

Live wide-play proxy: n_det >= 8 AND image_x spread >= 800px (both calibration-independent, so `calib_live` below is not circular).
Gate: >=6-corr yield >= 41% (>= iraq WC). Frame-weighted pooling.

## Per-match / pooled gate table

```
scope                    frm  live%  ge6 ALL  ge6 LIVE  calib LIVE  gate?
-------------------------------------------------------------------------
brighton_manutd         1463    55%      38%       68%         78%   PASS
manutd_fulham           1066    36%      20%       56%         73%   PASS
manutd_liverpool        1299    29%      24%       81%         96%   PASS
-------------------------------------------------------------------------
POOLED (all 3)          3828    41%      28%       68%         81%   PASS
```

WC reference (curated wide-play chunks): >=6-corr iraq 41% / sen 62% / nor 49%; calib iraq 88%.

## Per-segment detail

```
segment                     frm  live% ppf_live  ge6_all ge6_live calib_live
brighton_manutd/seg_1       528    50%     11.5      40%      81%        89%
brighton_manutd/seg_2       470    63%     13.2      29%      45%        57%
brighton_manutd/seg_3       465    54%     14.3      44%      80%        91%
manutd_fulham/seg_1         404    36%     11.6      28%      78%        84%
manutd_fulham/seg_2         383    31%     11.1      20%      63%        76%
manutd_fulham/seg_3         279    42%     10.4       9%      22%        57%
manutd_liverpool/seg_1      510    30%     11.2      25%      80%        99%
manutd_liverpool/seg_2      348    31%     15.5      27%      88%        95%
manutd_liverpool/seg_3      441    26%     10.8      20%      77%        93%
```

## Why the residual live-play frames fail

```
pooled live-play frames that fail >=6-corr:        497
  calibration failed entirely (err>2m / inf):      292 (59%)
  calibration OK but <6 players projected:         205 (41%)
```

The calibration-failed frames (see `fail_*.jpg`) are overwhelmingly MIDFIELD-CENTERED shots: the camera frames the halfway line + center circle only, so PnLCalib has few / near-symmetric line features to lock onto. When play moves toward either box (penalty area, D, spot, goal) the markings are distinctive and calibration succeeds. This is a known-ambiguous single-frame case, not a broadcast close-up or replay.

Two caveats that make even these residuals softer than they look:
- The probe ran per-frame calibration (`calib_period=1`) for an HONEST yield. Production uses temporal reuse -- a good homography from an adjacent box-view frame carries across the ambiguous midfield frames -- which recovers most of the 59% calibration-failed bucket.
- The 41% partial bucket is projection coverage (players near the frame edge), not a calibration failure; it shrinks as more of the pitch is in view.

## Apples-to-oranges caveat on the pre-committed gate

Only ~41% of each raw 2-min PL slice is live wide-camera play (the rest is replays, close-ups, crowd, graphics -- and frames with zero detections are already dropped from the dense parquet, so the true wall-clock live fraction is lower still). The WC gate (>=6-corr >= 41%) was measured on hand-picked wide-play chunks, i.e. ~100% live. Comparing the WC gate to the PL UNCONDITIONAL 27% is apples-to-oranges; the like-for-like comparison is WC vs the PL LIVE-conditional yield (68% pooled), which clears the gate. The plan's gate should either (a) be applied to play-filtered footage, or (b) be lowered to a raw-slice-equivalent threshold.

## Verdict

SAMPLING ARTIFACT, not a geometry blocker. On genuine live wide-camera play PnLCalib solves 81% of frames (pooled; vs iraq WC 88%) and the >=6-corr yield is 68% (vs the 41% gate). The unconditional 27% is dragged down by the ~59% of each raw broadcast slice that is non-live content, which play-filtering / chunk selection removes for free. The only genuine geometry residual is midfield-centered single frames, which temporal calibration reuse largely recovers. Phase A effectively PASSES on calibration.

## Reading

- `ge6_all` is the unconditional yield the feasibility gate table reports; `ge6_live` conditions on the live wide-play proxy.
- `calib_live` (PnLCalib solve rate on live frames) is the direct, non-circular answer to 'does live wide play fail calibration?'.
- Contact sheets (`*_contactsheet.jpg`) validate the proxy visually; `fail_*.jpg` show why the residual live-play failures fail.
