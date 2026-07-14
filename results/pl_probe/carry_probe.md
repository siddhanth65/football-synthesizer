# PL coverage lever: temporal homography carry-over

```
PL COVERAGE LEVER -- temporal homography carry-over for the ball (v6 detections)
window=+/-50 frames (~2.0s @25fps)  cut=track-id Jaccard<0.3  interpolate=True
The ONLY success metric is post-link coverage OFF vs ON. own/carried/interp are
PRE-LINK diagnostics (how many frames each mechanism projected, before linking).

segment                    n_dense  OFF cov   ON cov  d(pp)   own carry interp offpit
-------------------------------------------------------------------------------------
brighton/seg_1                 528   43.2%    68.4%  +25.2   180    86     38      1
brighton/seg_2                 470   20.0%    29.8%   +9.8    74    45     25      0
brighton/seg_3                 465   24.7%    43.7%  +18.9    80    54     13      2
  brighton_manutd (match mean)           29.3%    47.3%  +18.0

manutd_f/seg_1                 404   29.5%    55.7%  +26.2    95    76     12      7
manutd_f/seg_2                 383   23.5%    34.7%  +11.2    43    41      6     11
manutd_f/seg_3                 279    4.7%    19.0%  +14.3    11    22      3      1
  manutd_fulham (match mean)           19.2%    36.5%  +17.3

manutd_l/seg_1                 510   18.0%    37.8%  +19.8    83    74     11     16
manutd_l/seg_2                 348    6.6%    21.8%  +15.2    15    22      4      1
manutd_l/seg_3                 441   16.8%    37.9%  +21.1    56    75     16      5
  manutd_liverpool (match mean)           13.8%    32.5%  +18.7

POOLED (mean of 9 segs)              20.8%    38.8%  +18.0

Baseline check: OFF should reproduce the v6 feasibility.md column (brighton 29.4 / fulham 19.7 / liverpool 15.2 / pooled 21.4).

PLAUSIBILITY of carried ball positions (post-link, observed carried samples):
  carried samples surviving link: 578   off-pitch: 0   max frame-to-frame speed: 46 m/s (clamp 40)

CARRY FAITHFULNESS (leave-one-out: carried vs own-frame ball position, metres):
  n=656  median=0.33m  p90=2.01m  within2m=90%  within5m=97%  max=756.4m
  (a sub-metre median means the camera is near-static across the window, so a
   carried homography is a valid stand-in; gross outliers hit the off-pitch gate.)
```
