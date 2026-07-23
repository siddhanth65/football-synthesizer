# Score-state segmentation v3 (Plan B-5) - Man Utd shape + press by scoreline, full 6-match corpus

> **2026-07-23 CORRECTION:** `southampton_manutd` + `tottenham_manutd` team mappings corrected (see `results/PAIR_ANALYSIS_v1.md`). Every Southampton (0-3 win) Man Utd row is recomputed with team0 = Man Utd; the prior wins-sat-deeper block read is retracted/revised.

Every metric below is the B-4 style fingerprint (validated tracking-native primitives)
re-bucketed by Man Utd's score state. States are **from Man Utd's perspective** (level / chasing / leading). Boundaries are the validated E2E-Spot goal peaks (per-half count
matched to the Sofascore split); which team scored comes from the Sofascore per-half deltas.
Brighton's two H2 goals are split by the final scoreline (90+' winner is Brighton's);
Tottenham's fourth E2E H2 peak (the known replay false positive) is dropped to honour the true
1H1/2H2 split. Engine: `fingerprint/score_state.py`. v1 (n=3) and v2 (n=5, southampton
excluded) kept at `results/SCORE_STATE_v1.md` / `results/SCORE_STATE_v2.md`.

**Corpus: all 6 matches** -- 3 losses (Liverpool 0-3, Tottenham 0-3, Brighton 1-2), 2 wins
(Fulham 1-0, Southampton 0-3) and 1 draw (Palace 0-0). `southampton_manutd` was excluded at
v2 (team-anchor collapse, balance 0.004) but is now re-anchored (balance 0.909) and included,
giving the comparison a real second win.

**n=6, each state a slice of an already ball-gap-limited base** - counts (`frames`,
`outside-third losses`) are on every row so small samples stay visible. Two matches barely
have a level state: Tottenham (opener ~3', level is only its first ~160 s) and Southampton
(Man Utd ahead from ~35', so its level state is small: 33 losses). The leading state exists in
Fulham (~5 min at 1-0) and Southampton (most of the match, ahead from ~35'). Absolute line
heights carry the ~+11m partial-broadcast inflation: read across states, not against FIFA
numbers.

## The money comparison: level-state counter-press by result

Man Utd's counter-press while the game is still level (0-0), grouped by how the match ended.
This is the direct test of the Liverpool case-study finding -- was the press flat from
kickoff a losses pattern, or Liverpool-specific? (Full five-metric level-state synthesis in
`results/WINS_VS_LOSSES.md`.)

| result group | match | level-state losses | counter-press frac | 5s regain |
|---|---|---|---|---|
| loss | Liverpool (0-3 loss) | 22 | 0.455 | 0.227 |
| loss | Tottenham (0-3 loss) | 1 | 0.000 | 0.000 |
| loss | Brighton (1-2 loss) | 64 | 0.719 | 0.406 |
| win | Fulham (1-0 win) | 71 | 0.718 | 0.465 |
| win | Southampton (0-3 win) | 26 | 0.423 | 0.192 |
| draw | Palace (0-0 draw) | 40 | 0.750 | 0.375 |

## Score-state timeline (validated goal boundaries)

**Liverpool** (0-3 loss):

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 2066.0     False
  h1 chasing       0-1 2066.0 2518.0     False
  h1 chasing       0-2 2518.0 2940.0      True
  h2 chasing       0-2    0.0  739.0     False
  h2 chasing       0-3  739.0 3116.5      True
```

**Tottenham** (0-3 loss):

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0  162.5     False
  h1 chasing       0-1  162.5 2940.0      True
  h2 chasing       0-1    0.0  210.5     False
  h2 chasing       0-2  210.5 2005.5     False
  h2 chasing       0-3 2005.5 3156.4      True
```

**Brighton** (1-2 loss):

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 1895.5     False
  h1 chasing       0-1 1895.5 2895.0      True
  h2 chasing       0-1    0.0  794.5     False
  h2   level       1-1  794.5 2881.0      True
  h2 chasing       1-2 2881.0 3086.8      True
```

**Fulham** (1-0 win):

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 2786.0      True
  h2   level       0-0    0.0 2512.5     False
  h2 leading       1-0 2512.5 2978.2      True
```

**Southampton** (0-3 win):

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 2103.5     False
  h1 leading       1-0 2103.5 2462.5     False
  h1 leading       2-0 2462.5 2940.0      True
  h2 leading       2-0    0.0 3069.5      True
  h2 leading       3-0 3069.5 3152.6      True
```

**Palace** (0-0 draw):

```
half state scoreline  t0_s   t1_s  stoppage
  h1 level       0-0   0.0 3129.6      True
  h2 level       0-0   0.0 3033.0      True
```

## Man Utd counter-press by state (cross-match)

Counter-press fraction = pressure within 4.57 m of the ball within 5 s of an outside-third
loss; 5s regain = ball won back in that window.

| match (result) | state | outside-third losses | counter-press frac | 5s regain |
|---|---|---|---|---|
| Liverpool (0-3 loss) | level | 22 | 0.455 | 0.227 |
| Liverpool (0-3 loss) | chasing | 53 | 0.660 | 0.396 |
| Tottenham (0-3 loss) | level | 1 | 0.000 | 0.000 |
| Tottenham (0-3 loss) | chasing | 47 | 0.489 | 0.255 |
| Brighton (1-2 loss) | level | 64 | 0.719 | 0.406 |
| Brighton (1-2 loss) | chasing | 38 | 0.632 | 0.368 |
| Fulham (1-0 win) | level | 71 | 0.718 | 0.465 |
| Fulham (1-0 win) | leading | 8 | 0.750 | 0.500 |
| Southampton (0-3 win) | level | 26 | 0.423 | 0.192 |
| Southampton (0-3 win) | leading | 77 | 0.519 | 0.325 |
| Palace (0-0 draw) | level | 40 | 0.750 | 0.375 |

## Man Utd shape by state (in-possession + attacking transition)

`def-line` = deepest-line attacking-x, `buildup depth` = mean outfield attacking-x (0 = own
goal). `trans_pos` = the win-it-and-go attacking transition.

| match | state | phase | frames | def-line | width | buildup depth |
|---|---|---|---|---|---|---|
| Liverpool | level | in_poss | 194 | 44.1 | 33.2 | 51.2 |
| Liverpool | chasing | in_poss | 470 | 45.9 | 32.1 | 52.7 |
| Liverpool | level | trans_pos | 136 | 39.4 | 31.1 | 45.3 |
| Liverpool | chasing | trans_pos | 212 | 45.4 | 28.0 | 51.3 |
| Tottenham | level | in_poss | 1 | 16.3 | 25.9 | 22.3 |
| Tottenham | chasing | in_poss | 229 | 52.2 | 27.7 | 58.3 |
| Tottenham | level | trans_pos | 12 | 18.7 | 27.5 | 24.4 |
| Tottenham | chasing | trans_pos | 184 | 49.8 | 27.2 | 55.3 |
| Brighton | level | in_poss | 344 | 47.5 | 38.6 | 54.2 |
| Brighton | chasing | in_poss | 147 | 50.9 | 38.5 | 57.4 |
| Brighton | level | trans_pos | 387 | 56.6 | 32.0 | 61.4 |
| Brighton | chasing | trans_pos | 154 | 53.5 | 33.6 | 59.4 |
| Fulham | level | in_poss | 549 | 49.8 | 31.3 | 56.9 |
| Fulham | leading | in_poss | 7 | 41.1 | 28.8 | 46.1 |
| Fulham | level | trans_pos | 343 | 52.0 | 32.5 | 57.9 |
| Fulham | leading | trans_pos | 26 | 37.6 | 31.5 | 43.5 |
| Southampton | level | in_poss | 85 | 58.9 | 42.0 | 64.6 |
| Southampton | leading | in_poss | 596 | 51.9 | 34.5 | 57.7 |
| Southampton | level | trans_pos | 88 | 45.5 | 31.7 | 51.1 |
| Southampton | leading | trans_pos | 302 | 54.0 | 32.3 | 59.2 |
| Palace | level | in_poss | 907 | 45.9 | 32.6 | 52.8 |
| Palace | level | trans_pos | 192 | 42.8 | 32.9 | 48.8 |

## Attributed-pass balance by state (ledger split held: liverpool, fulham, tottenham)

Ledger PASS events with an on-ball tracked carrier, bucketed by state. A **floor** (only
carried passes count; no per-state validation) and only for the matches whose match-level
team split held; Brighton is excluded (its split inverts); Southampton/Palace have no ledger.

| match | state | ManU att. passes | opponent att. passes |
|---|---|---|---|
| Liverpool | level | 69 | 74 |
| Liverpool | chasing | 112 | 100 |
| Tottenham | level | 4 | 8 |
| Tottenham | chasing | 116 | 210 |
| Fulham | level | 158 | 128 |
| Fulham | leading | 8 | 16 |

## Honest read

_2026-07-23 CORRECTION: southampton_manutd + tottenham_manutd team mappings corrected (see
results/PAIR_ANALYSIS_v1). Every Southampton (0-3 win) Man Utd row below is recomputed with team0 =
Man Utd; the prior "wins sat deeper" block read is retracted (point 3)._

1. THE Liverpool finding still does NOT repeat -- and the second win makes it starker. Liverpool's
   flat 0-0 counter-press (0.455) was Liverpool-specific: Brighton (also a loss) pressed 0.719 at 0-0,
   and the two WINS split the whole range -- Fulham 0.718 (high) vs Southampton 0.423 (low). Level-
   state counter-press does not separate win from loss, nor even the two wins from each other. The
   pre-scoreline collapse is one match (Liverpool), not a corpus law.
2. Tottenham stays unevaluable at level (conceded ~3', level = 1 loss). Southampton's level state is
   small too (26 losses -- ahead from ~35') but usable, and it is the lower-press win, which is exactly
   what kills any 'wins press harder at 0-0' story.
3. Block height at 0-0 does NOT separate wins from losses either (CORRECTED): the Southampton win
   defended the HIGHEST out-of-possession line of the six (53.7 m, above every loss 40.6-49.9), while
   the Fulham win was among the deepest (38.0 m) -- the two wins bracket the whole range. The prior
   'both wins sat deeper (32.9 / 38.0)' read was the southampton_manutd mapping flip and is retracted;
   the full five-metric level-state synthesis is in `results/WINS_VS_LOSSES.md`.
4. The two 0-3 losses still carry the lowest match-level regain, but that is press while ALREADY
   behind (both spent almost the whole match chasing), confounded with game state -- not a 0-0 signal.
5. Leading = press-while-ahead now has TWO matches, not one. Fulham dropped its line in the ~5 min at
   1-0; Southampton led from ~35' and pressed MORE when ahead (leading 0.519 vs level 0.423) --
   matching the Southampton report's 'counter-press rose when ahead'. The wins' energy is a
   with-the-lead trait, not a from-kickoff one.
6. Shape shifts with state, confirming the segmentation tracks something real: trailing sides push the
   line and centroid higher (Liverpool build-up rises into 0-3; Tottenham chasing sits high). Ignore
   the Tottenham level rows (n=1/12 frames -- garbage from the ~3' opener).
7. Pass balance (floor; liverpool, fulham, tottenham) still follows the scoreline: the trailing team
   sees marginally more of the ball.
8. Honest n: 6 usable matches now. Wins = Fulham + Southampton, draw = Palace; level-state samples
   range from 1 (Tottenham) to 71 (Fulham) outside-third losses. The one robust cross-match statement
   is still the negative one -- the flat-press-from-kickoff is not a Man-Utd-in-losses law, it is what
   happened against Liverpool.
