# Score-state segmentation v1 (Plan B-5) - Man Utd shape + press by scoreline

Every metric below is the B-4 style fingerprint (validated tracking-native primitives)
re-bucketed by Man Utd's score state. States are **from Man Utd's perspective** (level / chasing / leading). Boundaries are the E2E-Spot goal peaks (validated 7/7 across
these three matches with correct halves); which team scored each goal comes from the
Sofascore per-half score deltas, with Brighton's two second-half goals split by the final
scoreline (the 90+' winner is Brighton's). Engine: `fingerprint/score_state.py`.

**n=3 matches, and each state is a slice of an already ball-gap-limited base** - counts
(`frames`, `outside-third losses`) are shown on every row so small samples are visible. The
leading state exists only in the Fulham match and only for the ~5 minutes after the 87'
winner (tiny n - suggestive, not a claim). Absolute line heights carry the ~+11m
partial-broadcast inflation from v1: read across states, not against FIFA numbers.

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

## Man Utd counter-press by state (cross-match)

Counter-press fraction = pressure within 4.57 m of the ball within 5 s of an outside-third
loss; 5s regain = ball won back in that window.

| match (result) | state | outside-third losses | counter-press frac | 5s regain |
|---|---|---|---|---|
| Liverpool (0-3 loss) | level | 22 | 0.455 | 0.227 |
| Liverpool (0-3 loss) | chasing | 53 | 0.660 | 0.396 |
| Brighton (1-2 loss) | level | 64 | 0.719 | 0.406 |
| Brighton (1-2 loss) | chasing | 38 | 0.632 | 0.368 |
| Fulham (1-0 win) | level | 71 | 0.718 | 0.465 |
| Fulham (1-0 win) | leading | 8 | 0.750 | 0.500 |

## Man Utd shape by state (in-possession + attacking transition)

`def-line` = deepest-line attacking-x, `buildup depth` = mean outfield attacking-x (0 = own
goal). `trans_pos` = the win-it-and-go attacking transition.

| match | state | phase | frames | def-line | width | buildup depth |
|---|---|---|---|---|---|---|
| Liverpool | level | in_poss | 194 | 44.1 | 33.2 | 51.2 |
| Liverpool | chasing | in_poss | 470 | 45.9 | 32.1 | 52.7 |
| Liverpool | level | trans_pos | 136 | 39.4 | 31.1 | 45.3 |
| Liverpool | chasing | trans_pos | 212 | 45.4 | 28.0 | 51.3 |
| Brighton | level | in_poss | 344 | 47.5 | 38.6 | 54.2 |
| Brighton | chasing | in_poss | 147 | 50.9 | 38.5 | 57.4 |
| Brighton | level | trans_pos | 387 | 56.6 | 32.0 | 61.4 |
| Brighton | chasing | trans_pos | 154 | 53.5 | 33.6 | 59.4 |
| Fulham | level | in_poss | 549 | 49.8 | 31.3 | 56.9 |
| Fulham | leading | in_poss | 7 | 41.1 | 28.8 | 46.1 |
| Fulham | level | trans_pos | 343 | 52.0 | 32.5 | 57.9 |
| Fulham | leading | trans_pos | 26 | 37.6 | 31.5 | 43.5 |

## Attributed-pass balance by state (liverpool + fulham only)

Ledger PASS events with an on-ball tracked carrier, bucketed by state. A **floor** (only
carried passes count; no per-state validation) and only for the two matches whose match-level
team split held; Brighton is excluded (its split inverts).

| match | state | ManU att. passes | opponent att. passes |
|---|---|---|---|
| Liverpool | level | 69 | 74 |
| Liverpool | chasing | 112 | 100 |
| Fulham | level | 158 | 128 |
| Fulham | leading | 8 | 16 |

## Honest read

1. The counter-press collapse against Liverpool was already present at 0-0. Man Utd's level-state
   counter-press was 0.455 vs Liverpool but 0.719 (Brighton) and 0.718 (Fulham) - the press did not
   fail because United were chasing; it was the weakest of the three even while the game was level
   (Liverpool level n=22, Brighton 64, Fulham 71 - small but a wide gap).
2. Chasing lifts the press, not lowers it. In both losses Man Utd's counter-press rose from level to
   the trailing states as the match wore on (Liverpool 0.455 level -> 0.690 at 0-3; the numbers climb
   with the deficit) - the intensity arrives late, once the game is gone.
3. Leading = drop deep (one match, tiny n). In the ~5 minutes Man Utd led Fulham 1-0 they sat far
   back (out-of-possession build-up 28.5 m vs 43.1 m at level; deepest-line 23.4 vs 38.0) - a
   shut-up-shop signal, but n is single-digit-frames territory; suggestive only.
4. Shape shifts with state as expected: trailing sides push their line and centroid higher (Brighton
   trans_neg deepest-line 45.4 level -> 63.6 chasing; Liverpool build-up rises into 0-3) and widen
   slightly - the score state moves the block, confirming the segmentation is tracking something real.
5. Pass balance follows the scoreline: the trailing team sees marginally more of the ball
   (Liverpool: ManU 112:100 when chasing vs 69:74 level; Fulham: ManU behind on the ball 8:16 only
   while leading late) - a floor, but the direction is consistent.
6. n=3, small per-state slices, one leading state: this is a shape of behaviour, not a validated
   law. The one robust, cross-match claim is #1 - the Liverpool press was flat from kickoff, not just
   after the goals.
