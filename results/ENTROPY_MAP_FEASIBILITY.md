# Entropy-map feasibility on broadcast-derived ball tracks (Lucey et al., AAAI 2012)

**Verdict up front: the representation is computable but the identity test is at chance, and the
representation is not viable on our data as it stands.** Across 90 declared parameter settings
(6 gap policies x 3 window lengths T x 5 grids) the leave-one-match-out 1-NN team-identification
accuracy averages **0.168 against a permutation null of 0.191** — i.e. below the null on average.
Exactly **1 of 90** settings reaches p < 0.05 (4.5 expected by chance; family-wise p = 0.86 for that
minimum). The same-opponent other-leg retrieval — the cleanest test our design permits — averages
**top-1 = 0.094 against a chance rate of 0.091**, mean rank **5.92 of a chance 6.0**. At the declared
primary setting the entropy maps score 0.208 (null mean 0.256), and the opponent-only 6-class test
scores **0.000** (chance 0.091). There is no signal to report.

The census explains why, and the explanation is specific rather than generic: our **possession
strings are a median of 3 seconds long** where Metrica ground truth over the same possession rule
gives 8 seconds. Lucey's method needs strings long enough to slide a T-second window over; ours
mostly are not. The binding constraint is **not** ball coverage and **not** the broadcast camera's
field of view — the camera-censoring control below costs essentially nothing. It is frame-level
dropout in our own player track: **47.4% of 1 Hz ball samples have no calibration-trusted player row
at that frame at all**.

Artifacts: `tools/entropy_map_probe.py` (all stages), `tests/test_entropy_map_probe.py` (6 checks,
green), `results/entropy_map/runlog.txt` (full captured run), plus `census.csv`, `map_summary.csv`,
`deviation.csv`, `identity.json` in `results/entropy_map/`.

---

## 1. Declared parameters and gap policy (fixed before any identity result was computed)

| parameter | primary | sweep | why |
|---|---|---|---|
| sample rate | 1 Hz | — | Lucey's rate; our ball track is 5 Hz and is decimated (nearest sample to each integer second) |
| window `T` | **3 samples** (2 s travel) | 3, 5, 8 | census-driven: at Lucey's T=5 the median match-map has 68 play-segments, at T=3 it has 146 |
| grid | **4 x 3** (26.3 x 22.7 m cells) | 4x3, 6x4, 8x6, 10x8, 20x16 | census-driven: 4x3 is the finest grid giving >= ~5 segments/cell in a per-match map. Lucey's 20x16 = 320 cells against 146 segments is not estimable and is reported, not smoothed |
| gap policy `G` | **2 s** | 0, 1, 2, 3, 5, 10 | see below |
| `min_n` per cell | 5 | — | below this the plug-in entropy estimator is biased hard towards 0; such cells are set NaN, never smoothed or zero-filled |
| distance | mean abs. entropy difference over cells valid in *both* maps | — | mean, not sum, so maps with different valid-cell counts stay comparable |

**A play-segment** is `T` consecutive 1 Hz samples inside one possession string, so the ball's travel
horizon is `T - 1` s; a string of `T1` samples yields `T1 - T + 1` segments and shorter strings are
discarded. **The entropy map** takes, per grid cell, every segment *starting* in that cell and forms
the distribution over the *destination* cell `T - 1` s later; the cell's value is that distribution's
Shannon entropy in bits. (The alternative reading — entropy over the whole quantised sequence — is
not estimable at any sample size we or Lucey have: 320^5 possible sequences.)

**Gap policy, declared before use.** A possession string is a maximal run of seconds carrying the
same possessing team. A *hole* is a second with no ball sample, or with a ball sample but no carrier
within the 2.0 m possession radius. Holes of at most `G` seconds are bridged, with the ball position
linearly interpolated between the flanking observed positions; a longer hole, or any second carrying
the *other* team's label, ends the string. `G = 0` disables all probe interpolation. Direction is
normalised by a **180-degree rotation** (`x -> 105-x`, `y -> 68-y`) for the team attacking `-x`, not
a mirror, so both legs of a fixture and both halves land in the same attacking frame.

**How much is interpolated.** Two layers, reported separately (Census B columns `obs` / `pfill`):

| layer | share of string-seconds at G=2 | share at G=0 | share at G=10 |
|---|---|---|---|
| observed by the detector *and* kept by `link_ball` | 0.770 | 0.829 | 0.647 |
| `link_ball`'s own gap fill (<= 8 samples = 1.6 s), already in the shipped track | 0.203 | 0.171 | 0.197 |
| invented by **this probe** (`pfill`) | **0.027** | 0.000 | 0.156 |

At the declared G=2 the probe fabricates 2.7% of string seconds; the pre-existing `link_ball` fill is
an order of magnitude larger and is a property of the shipped artifact, not of this analysis. Every
result below was also computed at G=0 (zero probe interpolation) — the sensitivity table shows the
conclusion is identical.

---

## 2. Census (task 1)

### A. 1 Hz supply per match

`span_s` = seconds spanned by the match's chunks; `ball_cov` = 1 Hz seconds carrying a ball position;
`link_obs` = share of those samples `link_ball` marked observed; `poss_cov` = share of ball seconds
with a carrier within 2.0 m.

| match | mgr | chunks | span_s | ball_s | ball_cov | link_obs | poss_s | poss_cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| manutd_fulham | ten_hag | 10 | 5488 | 1927 | 0.351 | 0.746 | 580 | 0.301 |
| brighton_manutd | ten_hag | 11 | 5817 | 2529 | 0.435 | 0.809 | 915 | 0.362 |
| manutd_liverpool | ten_hag | 11 | 5737 | 1994 | 0.348 | 0.781 | 573 | 0.287 |
| southampton_manutd | ten_hag | 11 | 5670 | 2132 | 0.376 | 0.726 | 677 | 0.318 |
| palace_manutd | ten_hag | 10 | 5356 | 1735 | 0.324 | 0.782 | 468 | 0.270 |
| manutd_tottenham | ten_hag | 11 | 5752 | 1999 | 0.348 | 0.775 | 575 | 0.288 |
| liverpool_manutd | amorim | 10 | 5476 | 1754 | 0.320 | 0.744 | 516 | 0.294 |
| manutd_southampton | amorim | 10 | 5783 | 2377 | 0.411 | 0.669 | 655 | 0.276 |
| manutd_brighton | amorim | 11 | 5915 | 2473 | 0.418 | 0.729 | 1012 | 0.409 |
| fulham_manutd | amorim | 10 | 5725 | 2731 | 0.477 | 0.742 | 972 | 0.356 |
| manutd_palace | amorim | 11 | 5746 | 2248 | 0.391 | 0.695 | 703 | 0.313 |
| tottenham_manutd | amorim | 11 | 5501 | 2466 | 0.448 | 0.730 | 647 | 0.262 |
| **POOLED** | | 127 | 67966 | 26365 | **0.388** | 0.744 | 8293 | **0.315** |

Ball coverage (38.8%) is consistent with the 32-52% figure on record. **Possession coverage is the
new number and it is the killer: only 31.5% of ball-covered seconds carry a possession label.**
Decomposed over all 26,376 1 Hz ball samples:

- 13,861 (**0.526**) have at least one calibration-trusted player row *at that exact frame*;
- of those, 8,296 (**0.599**) have a player within 2.0 m -> 0.526 x 0.599 = **0.315** overall.

The 2 m rule is not the problem: on Metrica ground truth the identical rule labels 0.668 of seconds
(see section 6), i.e. 0.599 vs 0.668 is a minor difference. **The dominant loss is the 47.4% of ball
frames with no trusted player row.** Our mean trusted player rows per frame is 11.57/22 — close to the
11.8 broadcast figure the censoring simulator is tuned to — so the players we do see are the right
number; they are simply absent on half the frames the ball is present.

### B. Possession strings and play-segment yield

Diagnostic ceiling first: **contiguous ball runs ignoring the possession label** are median 4 s,
p90 13 s, max 47 s (4,598 runs, 26,365 s). **Possession-labelled runs are median 1 s, p90 3 s, max
15 s** (4,569 runs, 8,293 s). The representation's raw material is destroyed by possession labelling,
not by ball tracking.

Per match at the declared **G = 2 s** (`T{k}str` = strings of length >= k; `T{k}seg` = play-segments;
`T{k}MU` = the Man Utd share of those segments):

| match | strings | secs | med | p90 | max | obs | pfill | T3str | T3seg | T3MU | T5str | T5seg | T5MU | T8str | T8seg | T8MU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| manutd_fulham | 112 | 452 | 3.0 | 7.0 | 14 | .726 | .029 | 78 | 228 | 132 | 31 | 102 | 66 | 10 | 32 | 26 |
| brighton_manutd | 202 | 808 | 3.0 | 8.0 | 19 | .837 | .012 | 129 | 404 | 163 | 59 | 195 | 80 | 22 | 73 | 32 |
| manutd_liverpool | 110 | 456 | 3.0 | 7.1 | 14 | .825 | .015 | 74 | 236 | 142 | 38 | 109 | 74 | 11 | 28 | 24 |
| southampton_manutd | 146 | 585 | 3.0 | 6.0 | 32 | .718 | .074 | 97 | 293 | 150 | 39 | 135 | 69 | 12 | 59 | 25 |
| palace_manutd | 94 | 389 | 3.0 | 7.0 | 15 | .810 | .013 | 63 | 201 | 142 | 27 | 92 | 67 | 8 | 35 | 26 |
| manutd_tottenham | 124 | 473 | 3.0 | 7.0 | 11 | .784 | .019 | 78 | 225 | 43 | 34 | 98 | 13 | 11 | 21 | 1 |
| liverpool_manutd | 101 | 440 | 4.0 | 7.0 | 13 | .745 | .034 | 70 | 238 | 85 | 37 | 115 | 38 | 9 | 31 | 6 |
| manutd_southampton | 138 | 536 | 3.0 | 7.0 | 27 | .733 | .024 | 81 | 260 | 105 | 37 | 125 | 45 | 12 | 43 | 7 |
| manutd_brighton | 229 | 929 | 3.0 | 8.0 | 15 | .749 | .029 | 158 | 471 | 162 | 67 | 216 | 55 | 24 | 66 | 14 |
| fulham_manutd | 210 | 920 | 3.5 | 7.0 | 21 | .791 | .022 | 157 | 500 | 263 | 74 | 238 | 127 | 20 | 78 | 51 |
| manutd_palace | 154 | 609 | 3.0 | 8.0 | 19 | .755 | .033 | 94 | 301 | 182 | 36 | 144 | 91 | 19 | 57 | 39 |
| tottenham_manutd | 140 | 555 | 3.0 | 7.1 | 12 | .764 | .014 | 90 | 275 | 59 | 42 | 123 | 16 | 14 | 33 | 1 |
| **POOLED** | 1760 | 7152 | | | | .770 | .027 | 1169 | 3632 | 1628 | 521 | 1692 | 741 | 172 | 556 | 252 |

Pooled totals at the other gap policies:

| G | strings | string-seconds | median len | T3 segments | T5 segments | T8 segments | probe-filled |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1647 | 4718 | 2.0 | 1424 | 422 | 95 | 0.000 |
| 1 | 1741 | 6080 | 3.0 | 2598 | 1022 | 285 | 0.012 |
| **2** | **1760** | **7152** | **3.0** | **3632** | **1692** | **556** | **0.027** |
| 3 | 1746 | 8098 | 4.0 | 4606 | 2477 | 976 | 0.043 |
| 5 | 1725 | 9851 | 4.0 | 6401 | 4079 | 2032 | 0.076 |
| 10 | 1703 | 13233 | 6.0 | 9827 | 7309 | 4695 | 0.156 |

**Census verdict.** Computable at **T = 3** (median 146 segments per match-map, ~12 per cell on a 4x3
grid) and marginally at T = 5 (median 68 segments, ~6 per cell). **Not computable at T = 8** (median
25 segments per match-map at G=2; 2 at G=0). Lucey's own parameters — T=5 on a 20x16 grid — are not
computable at any gap policy: 320 cells against at most a few hundred segments. The aggregate
(12-match pooled) Man Utd map at T=3/4x3 has 1,628 segments and fills all 12 cells; **per-match maps
fill a median of 9 of 12 cells and leave 3 empty**, and two matches (manutd_tottenham, n=43;
tottenham_manutd, n=59) leave 8 of 12 cells under-sampled.

---

## 3. The maps (task 3)

Man Utd aggregate, G=2 / T=3 / 4x3, 1,628 segments, all 12 cells valid, mean entropy 1.485 bits
(rows = width bands, columns = length bands, attacking left-to-right; `n` = segments starting there):

```
 1.17  1.54  1.24  1.49    n=   69  135  215   80
 1.77  2.14  1.72  1.40    n=   45  148  372  142
 1.04  1.36  1.51  1.44    n=   49  170  149   54
```

Per-map summary is in `results/entropy_map/map_summary.csv`. Range over the 24 match-maps: 43-309
segments, 4-11 of 12 cells valid, mean entropy 0.797-1.512 bits.

**The estimator confound, stated plainly.** Across the 24 maps, mean entropy correlates with the map's
segment count at **r = 0.49**, and the number of valid cells at **r = 0.68**. A map's entropy level is
therefore substantially a readout of how much data that match produced, not of how that team played.
Any apparent similarity structure would be partly sample-size similarity — which is one more reason
not to read anything into the (absent) identity signal.

Grid sensitivity: see the sweep in section 4. Going finer than 4x3 raises the empty-cell count
monotonically; at 20x16 the median match-map has 146 segments for 320 cells and almost every cell is
NaN, so the distance is computed over a handful of survivors.

---

## 4. Identity test (task 4)

Design at n=12 matches: **24 maps** (each match yields a Man Utd map and an opponent map), 7 classes
(Man Utd n=12; Brighton, Liverpool, Fulham, Crystal Palace, Southampton, Tottenham n=2 each). Three
tests, each with its own permutation null (labels shuffled *within the tested subset* so class sizes
are preserved; 2,000 draws at the primary setting, 500 in the sweep).

### Primary setting (G=2, T=3, 4x3, min_n=5), entropy feature

| test | n | observed | correct chance baseline | null mean | null p95 | p |
|---|---:|---:|---|---:|---:|---:|
| (a) 7-class LOO 1-NN, all maps | 24 | **0.208** | majority class = 0.500 | 0.256 | 0.417 | 0.731 |
| (a) balanced accuracy (mean per-class recall) | 24 | **0.060** | 1/7 = 0.143 | 0.104 | 0.214 | 0.802 |
| (b) 6-class opponent-only LOO 1-NN | 12 | **0.000** | 1/11 = 0.091 | 0.089 | — | 1.000 |
| (b) same-opponent other-leg retrieval, top-1 | 12 | **0.000** | 0.091 | — | — | — |
| (b) same-opponent other-leg mean rank | 12 | **7.75** | 6.0 | — | — | — |

Control feature (start-cell **occupancy** distribution, Jensen-Shannon distance — i.e. "where does
this team hold the ball", ignoring entropy): 7-class accuracy 0.292 (null 0.263, p = 0.477),
opponent-only 0.000, retrieval top-1 0.000, mean rank 7.67. Also at chance, so the failure is not
specific to the entropy transform.

Note the correct reading of (a): the majority-class baseline is 0.500 (always answer "Man Utd") and
the permutation null mean is 0.256. The observed 0.208 is **below both**.

### (c) Sensitivity and the null, over all 90 declared settings

| statistic | value |
|---|---|
| settings tested | 90 (6 gap policies x 3 T x 5 grids) |
| mean 7-class accuracy | **0.168** |
| mean permutation-null accuracy | **0.191** |
| settings where accuracy exceeds its own null | 28 / 90 |
| settings with p < 0.05 | **1** (4.5 expected by chance) |
| minimum p | 0.022 (G=5, T=3, 10x8); family-wise p = **0.86** |
| opponent-only accuracy: mean / max | 0.077 / 0.500 (chance 0.091); exactly 0.000 in 52 / 90 settings |
| other-leg retrieval top-1: mean / max | 0.094 / 0.500 (chance 0.091) |
| other-leg retrieval mean rank across settings | 5.92 (chance 6.0) |

The full 90-row table is in `results/entropy_map/runlog.txt`. The single p < 0.05 hit is what 90
independent tests produce by construction; it is not reported as a finding.

---

## 5. Deviation detection (task 5)

Distance of each Man Utd match map from the **leave-one-out** Man Utd aggregate (that match excluded
from the aggregate). Because a map built from 43 segments is far from any aggregate through estimator
noise alone, each distance is scored against a **resampling null**: 400 draws of the same number of
segments from Man Utd's pooled corpus, each measured against the same aggregate. `p_dev` is the
share of null draws at least as far as the observed map.

| match | opponent | venue | mgr | result | n_seg | valid cells | d_agg | null median | null p95 | p_dev | z | H_mean |
|---|---|:--:|:--:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| palace_manutd | Crystal Palace | A | TH | D 0-0 | 142 | 7 | **0.818** | 0.376 | 0.562 | **0.002** | 2.24 | 0.810 |
| southampton_manutd | Southampton | A | TH | W 3-0 | 150 | 10 | **0.656** | 0.378 | 0.552 | 0.005 | 1.10 | 0.879 |
| manutd_fulham | Fulham | H | TH | W 1-0 | 132 | 7 | 0.587 | 0.391 | 0.573 | 0.040 | 0.62 | 1.082 |
| manutd_southampton | Southampton | H | AM | W 3-1 | 105 | 9 | 0.557 | 0.426 | 0.616 | 0.140 | 0.41 | 1.019 |
| tottenham_manutd | Tottenham | A | AM | L 0-1 | 59 | 4 | 0.500 | 0.477 | 0.800 | 0.469 | 0.00 | 1.221 |
| manutd_tottenham | Tottenham | H | TH | L 0-3 | 43 | 4 | 0.493 | 0.508 | 0.941 | 0.529 | -0.05 | 1.077 |
| manutd_palace | Crystal Palace | H | AM | L 0-2 | 182 | 9 | 0.446 | 0.329 | 0.457 | 0.075 | -0.37 | 1.229 |
| fulham_manutd | Fulham | A | AM | W 1-0 | 263 | 11 | 0.434 | 0.286 | 0.413 | 0.037 | -0.46 | 1.248 |
| brighton_manutd | Brighton | A | TH | L 1-2 | 163 | 10 | 0.430 | 0.344 | 0.494 | 0.180 | -0.49 | 1.315 |
| liverpool_manutd | Liverpool | A | AM | D 2-2 | 85 | 9 | 0.417 | 0.454 | 0.675 | 0.618 | -0.58 | 1.103 |
| manutd_brighton | Brighton | H | AM | L 1-3 | 162 | 10 | 0.383 | 0.337 | 0.481 | 0.312 | -0.82 | 1.302 |
| manutd_liverpool | Liverpool | H | TH | L 0-3 | 142 | 8 | 0.270 | 0.364 | 0.516 | 0.858 | -1.61 | 1.325 |

Numbers, not narrative:

- The raw distance `d_agg` is strongly confounded with sample size — the null median runs from 0.286
  at n=263 to 0.508 at n=43. The raw ordering (and the `z` column) is therefore not interpretable on
  its own; `p_dev` is.
- Holm-corrected over the 12 tests, **exactly one match is an outlier**: `palace_manutd` (away at
  Crystal Palace, ten Hag, 0-0; p_dev = 0.002, 0.002 x 12 = 0.024). `southampton_manutd`
  (p_dev = 0.005) does not survive (0.005 x 11 = 0.055). Nothing else is close.
- Direction of the effect for `palace_manutd`: its map mean entropy is the **lowest** of the 12
  (0.810 bits vs 1.134 averaged over the 12 Man Utd maps) on only 7 of 12 valid cells.
- Cross-reference without interpretation: the two lowest `p_dev` matches are both **away, ten Hag**;
  the three lowest-`d_agg` matches are Liverpool home (TH), Brighton home (AM) and Brighton away (TH).
  Venue, manager and result are each split across the table's top and bottom, and with one surviving
  outlier out of 12 there is no basis for attributing it to any of them.

---

## 6. Degradation control: full-pitch Metrica vs the same track censored (added on request)

Metrica Sample Games 1 and 2 (genuine 25 Hz, 22-player, full-pitch truth) run through the *identical*
pipeline — same 1 Hz decimation, same Viterbi possession smoother, same 2.0 m radius, same gap policy,
same T and grid. Three conditions: `full` (all 22 players), `camera` (only players inside the
broadcast window from `synthesizer.imputation`, half-width tuned so mean visible players = 11.8/22),
`camera+drop` (camera plus a per-second ball-availability mask copied verbatim from our 12 real
matches — 38.8% present, real burst structure).

| game | condition | 1 Hz s | labelled | strings | med len | p90 | segs team0 | segs team1 | H_mean t0 / t1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Game 1 | full | 3531 | 0.668 | 247 | 8 | 25 | 1217 | 1119 | 1.707 / 1.799 |
| Game 1 | camera | 3531 | 0.668 | 247 | 8 | 25 | 1217 | 1118 | 1.707 / 1.794 |
| Game 1 | camera+drop | 1504 | 0.645 | 178 | 4 | 13 | 363 | 410 | 1.572 / 1.574 |
| Game 2 | full | 3334 | 0.673 | 237 | 8 | 26 | 1236 | 1080 | 1.764 / 1.813 |
| Game 2 | camera | 3334 | 0.673 | 237 | 8 | 26 | 1236 | 1080 | 1.764 / 1.813 |
| Game 2 | camera+drop | 1579 | 0.668 | 194 | 5 | 12 | 456 | 414 | 1.705 / 1.676 |

Map displacement, against the between-team difference in the same game as the yardstick:

| game / team | d(full, camera) | d(full, camera+drop) | d(full teamA, full teamB) | ratio (drop / between-team) |
|---|---:|---:|---:|---:|
| G1 team0 | 0.000 | 0.236 | 0.253 | 0.93 |
| G1 team1 | 0.005 | 0.309 | 0.253 | 1.22 |
| G2 team0 | 0.000 | 0.194 | 0.258 | 0.75 |
| G2 team1 | 0.000 | 0.161 | 0.258 | 0.62 |

Censored-map -> full-map retrieval of the same team-game: **3 / 4** (chance 1/4).

Three readings, all load-bearing for the pivot:

1. **The camera window costs nothing.** d(full, camera) = 0.000-0.005 against a between-team
   difference of 0.25. Off-screen censoring is irrelevant to this representation, because the
   broadcast camera follows the ball, so whoever is within 2 m of the ball is on screen by
   construction. Any framing of this pivot as "can we recover team style from a restricted field of
   view" is answered *yes, trivially* — and that is not our problem.
2. **Ball-track dropout at our measured rate is what does the damage.** It halves the usable seconds,
   cuts median string length from 8 s to 4-5 s, cuts segments per team-game from ~1200 to ~400, and
   moves the map by 0.16-0.31 — **0.62 to 1.22 times the true between-team difference**. When the
   censoring artefact is the same size as the signal, identification at n=12 is hopeless, which is
   exactly what section 4 measured.
3. **Our real data is worse than `camera+drop`.** That condition retains a 0.65 possession-label rate
   and 4-5 s median strings; ours are 0.315 and 3 s, because the simulation censors *ball* seconds but
   still hands the possession rule perfect player positions on every surviving frame. It also injects
   no positional noise (our ball projection and player positions both carry metre-scale error). The
   Metrica censored numbers are therefore an **optimistic bound** on our situation, and even that
   bound sits at the edge of unusable.

---

## 7. Verdict

**Not viable as it stands, and the deficit is quantified rather than vague.** The gap to a working
regime is: median possession string 3 s vs 8 s, possession-label rate 0.315 vs 0.668, per-match-map
segments ~146 vs Lucey's per-team corpus built from 380 games. Two things follow.

- **This is not a field-of-view problem.** The calibrated broadcast-camera censor costs 0.005 bits.
  Anyone arguing that broadcast footage inherently cannot support Lucey-style representations because
  of the camera is wrong, and we have the control that says so.
- **It is a frame-level tracking-continuity problem, with one dominant term.** 47.4% of 1 Hz ball
  samples carry no calibration-trusted player row. Fix that and the possession-label rate moves
  towards the 0.599 conditional rate we already achieve (which is close to Metrica's 0.668), string
  length follows, and the representation becomes estimable at Lucey's own T=5. That is a
  continuity/calibration engineering target with a measurable acceptance criterion — median
  possession string >= 8 s — not another metric to invent.

Until that criterion is met, do not build on entropy maps: at n=12 matches with our current
continuity, the representation carries no team identity that survives its own null, and the
censoring artefact is the same magnitude as the between-team signal it would have to detect.

**What was not done and why:** no matched-n resampling control on the identity test (its purpose is
to check whether an *observed* signal is a sample-size artefact — there is no observed signal to
check); no Metrica event-based possession (we deliberately used the identical nearest-player rule so
the comparison isolates censoring rather than the possession definition); Metrica positional noise not
simulated, which makes the section-6 numbers an optimistic bound as stated.
