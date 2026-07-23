# Game-state v2 - win-probability bands + manager-regime attack typing

> **2026-07-23 CORRECTION:** `southampton_manutd` + `tottenham_manutd` team mappings corrected (see `results/PAIR_ANALYSIS_v1.md`). The Southampton (0-3 win) Man Utd counter-press / shape / attack-mix rows are recomputed with team0 = Man Utd; the prior Southampton-high-press and wins=deeper-block claims are retracted/revised. `tottenham_manutd`, now processed, enters the Amorim attack-typing pool (6 vs 6).

Two upgrades to the game-state layer, both event-only (CPU):

1. **Bayesian-style in-game win probability, base subset** (Robberechts, Van Haaren & Davis,
   KDD'21) replaces the crude scoreline states (level / chasing / leading). WP(t) is
   continuous and **strength- and time-aware**: covariates = minutes remaining, score
   differential, clubelo Elo prior, goals, red cards, yellow cards. Engine
   `fingerprint/win_probability.py`; fit `tools/build_win_probability.py`.
2. **Attack typing** (fast transition / sustained build-up / direct), rule-based over Viterbi
   possession spells (`fingerprint/attack_typing.py`), split by manager regime.

## The WP model + calibration (the mandatory gate)

Fit on **StatsBomb open-data Premier League 2015/2016** (380 matches, all 20 clubs, goal + card
minutes from events; the same competition family as our 24/25 target and a held-out era, so no
leakage). Team strength = **clubelo** Elo at the match date (cached `outputs/oracle/elo/`).
Model: multinomial logistic regression over the six covariates + two interactions
(`score_diff/sqrt(min_left+1)`, `score_diff^2`); a gradient-boosted alternative was rejected
for fitting Elo **non-monotonically** (a stronger team getting a lower win prob).
Only the base subset is built -- the paper's four located-event features (attacking passes, xT,
chance quality, duel strength) need Opta-grade events our 26% geometry cannot supply.

* Held-out win-prob **ECE = 0.0415** (random match-level 80/20 split); last-10-min
  ECE = 0.0472; temporal end-of-season split ECE = 0.0487 (stricter,
  regime-shifted). Paper full model ECE 0.011 (10 features, 8 seasons) -- ours is a 6-feature
  base subset on 1 season, so a higher ECE is expected and honestly reported.
* Sanity: kickoff home P(win) at equal Elo = ~0.42 (PL home-win base rate 0.41); a 1-goal lead
  with 5 min left = ~0.84; monotone increasing in Elo. The WP(t) series for our matches
  behave correctly (e.g. Liverpool away-underdog kickoff 0.30 -> 0.00 in the
  0-3; Fulham 0.48 -> 0.996 on the 87' winner).

WP(t) series (Man Utd perspective) are cached at `outputs/oracle/wp/<match_id>.parquet`.

### Reliability table (held-out)

```
 bin_lo  bin_hi    n  mean_p  frac_win
    0.0     0.1 2893  0.0298    0.0207
    0.1     0.2 1846  0.1527    0.1056
    0.2     0.3 2387  0.2495    0.1877
    0.3     0.4 1907  0.3451    0.3760
    0.4     0.5 1049  0.4512    0.4833
    0.5     0.6  653  0.5487    0.5972
    0.6     0.7  773  0.6457    0.7296
    0.7     0.8  742  0.7462    0.6887
    0.8     0.9  354  0.8397    0.8644
    0.9     1.0 1228  0.9753    0.9161
```

## Every score-state metric, re-expressed by WP band

Bands on Man Utd's win probability: **loss-likely** (WP<0.35), **balanced** (0.35-0.65),
**win-likely** (WP>0.65). Unlike the raw scoreline states, these fold in opponent strength and
time remaining. Counts (`losses`, `frames`) are on every row; all B-4 ceilings still apply
(ball-gap possession base, partial-broadcast line inflation ~+11 m -- read across bands, not
against FIFA metres). n = 6 validated ten-Hag matches.

### Man Utd counter-press by WP band (pooled over 6 matches)

Counter-press fraction = pressure within 4.57 m of the ball within 5 s of an outside-third
loss; 5 s regain = ball won back in that window.

```
       band  losses  cp_frac  regain_5s
loss-likely     279    0.642      0.341
   balanced      83    0.614      0.434
 win-likely      85    0.541      0.341
```

### Man Utd in-possession shape by WP band (pooled)

`def_line` = deepest-line attacking-x, `buildup` = mean outfield attacking-x (0 = own goal).

```
       band  frames  def_line  buildup  width
loss-likely    2300      46.4     53.1   33.0
   balanced     626      53.3     60.2   34.1
 win-likely     603      51.8     57.5   34.4
```

### Why WP bands are not just the scoreline states (frames: state x band)

The cross-tab shows the raw `level`/`chasing`/`leading` state splitting across WP bands -- e.g.
a 0-0 (`level`) against a stronger side sits in `balanced` or `loss-likely`, not a single
bucket. This is the whole point of the upgrade.

```
      match   state  loss-likely  balanced  win-likely
  Liverpool chasing         1133         0           0
  Liverpool   level          708         0           0
  Tottenham chasing         1768         0           0
  Tottenham   level            0        60           0
   Brighton chasing          770         0           0
   Brighton   level         1320       344           0
     Fulham leading            4         0         129
     Fulham   level          533      1051           0
Southampton leading            0         0        1355
Southampton   level            0       309           0
     Palace   level         1442         0           0
```

### Per-match counter-press by WP band

```
      match        band  losses  cp_frac  regain_5s
  Liverpool loss-likely      75    0.600      0.347
  Tottenham    balanced       1    0.000      0.000
  Tottenham loss-likely      47    0.489      0.255
   Brighton    balanced      14    0.500      0.286
   Brighton loss-likely      88    0.716      0.409
     Fulham    balanced      42    0.786      0.643
     Fulham loss-likely      29    0.621      0.207
     Fulham  win-likely       8    0.750      0.500
Southampton    balanced      26    0.423      0.192
Southampton  win-likely      77    0.519      0.325
     Palace loss-likely      40    0.750      0.375
```

## Manager-regime attack-type mix (Ten Hag vs Amorim)

Rule-based typing over Viterbi possession spells: a **3-way** label (`fast_transition` /
`sustained_build_up` / `direct`) is given to a deep-start possession that reaches the
attacking third with a placed entry time (ball+geometry); everything else falls back to a
2-way fast/sustained tier or is a high-turnover start (excluded). **Coverage is low** -- only
~10-20% of build attempts get a full 3-way label at our ball coverage (32-52%) -- so the mix
is a **tendency over a small n**, never an event count. `cov_3way` and `n_3way` are shown so
the resolution is visible. Practitioner-heuristic-inspired, NOT a Hobbs et al. proxy (their
method scores defensive disorganisation from full opponent tracking we do not have).

### Pooled 3-way mix by regime (shares over the 3-way-labelled Man Utd attacks)

```
manager  matches  n_3way  fast_transition_n  fast_transition  sustained_build_up_n  sustained_build_up  direct_n  direct
ten_hag        6      71                  7            0.099                    42               0.592        22   0.310
 amorim        6      77                  8            0.104                    36               0.468        33   0.429
```

### Per-match detail (n_attack = build attempts; cov_3way = 3-way label coverage)

```
manager              match       date  n_attack  n_3way  cov_3way  high_n  fast_transition  sustained_build_up  direct
ten_hag    brighton_manutd 2024-08-24        91      10     0.110      57                3                   5       2
ten_hag   manutd_liverpool 2024-09-01        65      11     0.169      36                0                   8       3
ten_hag      manutd_fulham 2024-08-16        75      16     0.213      24                3                   9       4
ten_hag      palace_manutd 2024-09-21        40      18     0.450      17                0                  14       4
ten_hag   manutd_tottenham 2024-09-29        47       3     0.064      29                0                   0       3
ten_hag southampton_manutd 2024-09-14        89      13     0.146      35                1                   6       6
 amorim   liverpool_manutd 2025-01-05        68      12     0.176       8                1                   6       5
 amorim    manutd_brighton 2025-01-19        95       9     0.095      32                0                   7       2
 amorim      fulham_manutd 2025-01-26        99      20     0.202      40                4                   9       7
 amorim      manutd_palace 2025-02-02        97      10     0.103      33                0                   3       7
 amorim manutd_southampton 2025-01-16       102      10     0.098      38                2                   5       3
 amorim   tottenham_manutd 2025-02-16        89      16     0.180      41                1                   6       9
```

## Honest read

1. The WP(t) upgrade is real and validated: the base-subset model is calibrated (held-out ECE ~0.04)
   and gives sensible, monotone win probabilities, so the WP bands are a defensible strength-/time-
   aware replacement for the raw scoreline states. The state x band cross-tab confirms they carry
   information the scoreline does not (a 0-0 is split across bands by opponent Elo and clock).
2. The band re-expression is over the SAME six validated matches and the SAME ball-gap-limited B-4
   primitives -- it re-buckets, it does not add data. Small per-band samples remain; the pooled rows
   are the honest unit.
3. Cards are unavailable for our own matches, so red_diff/yellow_diff = 0 there; the card covariates
   are exercised only in fitting/calibration. This is a documented base-subset gap, not a silent one.
4. Attack typing has LOW 3-way coverage (~10-20% of build attempts) at our ball coverage; the manager
   split is a direction-only read over small n (6 ten-Hag vs 6 Amorim matches, tens of typed attacks
   per regime). Do not report it as significance. The 2-way fallback tier is dominated by short
   fragments and is not used for the headline mix.
5. Nothing here claims a tactical law. It claims: a calibrated game-state covariate now exists, and a
   first, honest manager-regime attack-mix read is on the table for the corpus to grow into.
