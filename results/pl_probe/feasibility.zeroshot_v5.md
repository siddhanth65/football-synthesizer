# Phase A feasibility probe

```
PHASE A FEASIBILITY GATE TABLE (Man Utd PL segments; 3 segments x 2 min each)

metric                             brighton_manutd   manutd_fulhammanutd_liverpool          POOLED           WC baseline
------------------------------------------------------------------------------------------------------------------------
players/frame (mean)                           8.9             6.8             6.2             7.3      irq 7.8/sen 10.9
track count (mean/seg, 2min)                   194             174             193             187        ~180/2min (WC)
track frag index (n_trk/ppf)                 22.48           25.59           32.30           26.79          lower=better
mean track length (s)                          4.6             2.8             2.9             3.4                   n/a
calibrated frames (<=2m gate)                  75%             78%             86%             80%               irq 88%
>=6-corr yield (norway killer)                 38%             19%             24%             27%            >= irq 41%
ball fire-rate @0.5 (NOT recall)               57%             58%             43%             53%       no direct WC eq
POST-LINK coverage (the number)              28.6%           18.9%           13.2%           20.2%sen 46%/irq 36%/nor 38%

GATES NEEDING EXTERNAL DATA (honestly pending, not proxied):
  - True ball recall on ~100 hand-labelled frames: PENDING -- needs hand labels.
    (fire-rate above is a precision-eyeball proxy only; see results/pl_probe/<m>/).
  - FBref possession within ~5pp: PENDING -- needs a full-match process + FBref pull.

PER-SEGMENT DETAIL:
  brighton_manutd/seg_1: frames=528 ppf=7.7 tracks=239 frag=31.17 calib=80% >=6corr=40% fire=80% proj=170/170 (too_few=0 homog_fail=0) post-link=42.0%
  brighton_manutd/seg_2: frames=470 ppf=9.6 tracks=184 frag=19.20 calib=64% >=6corr=29% fire=54% proj=74/74 (too_few=0 homog_fail=0) post-link=20.9%
  brighton_manutd/seg_3: frames=465 ppf=9.4 tracks=160 frag=17.06 calib=82% >=6corr=44% fire=36% proj=75/75 (too_few=0 homog_fail=0) post-link=22.8%
  manutd_fulham/seg_1: frames=404 ppf=6.7 tracks=193 frag=28.66 calib=89% >=6corr=28% fire=75% proj=85/85 (too_few=0 homog_fail=0) post-link=26.0%
  manutd_fulham/seg_2: frames=383 ppf=6.7 tracks=177 frag=26.47 calib=78% >=6corr=20% fire=64% proj=48/48 (too_few=0 homog_fail=0) post-link=24.0%
  manutd_fulham/seg_3: frames=279 ppf=7.0 tracks=151 frag=21.65 calib=68% >=6corr=9% fire=35% proj=9/9 (too_few=0 homog_fail=0) post-link=6.8%
  manutd_liverpool/seg_1: frames=510 ppf=6.0 tracks=227 frag=37.70 calib=97% >=6corr=25% fire=56% proj=70/70 (too_few=0 homog_fail=0) post-link=17.8%
  manutd_liverpool/seg_2: frames=348 ppf=7.2 tracks=132 frag=18.40 calib=72% >=6corr=27% fire=18% proj=17/17 (too_few=0 homog_fail=0) post-link=8.6%
  manutd_liverpool/seg_3: frames=441 ppf=5.4 tracks=219 frag=40.80 calib=89% >=6corr=20% fire=56% proj=49/49 (too_few=0 homog_fail=0) post-link=13.2%
```
