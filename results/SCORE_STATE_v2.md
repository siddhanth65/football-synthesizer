# Score-state segmentation v2 (Plan B-5) - Man Utd shape + press by scoreline, 6-match corpus

Every metric below is the B-4 style fingerprint (validated tracking-native primitives)
re-bucketed by Man Utd's score state. States are **from Man Utd's perspective** (level / chasing / leading). Boundaries are the validated E2E-Spot goal peaks (per-half count
matched to the Sofascore split); which team scored comes from the Sofascore per-half deltas.
Brighton's two H2 goals are split by the final scoreline (90+' winner is Brighton's);
Tottenham's fourth E2E H2 peak (the known replay false positive) is dropped to honour the true
1H1/2H2 split. Engine: `fingerprint/score_state.py`. v1 (n=3) kept at
`results/SCORE_STATE_v1.md`.

**Corpus: 5 of 6 matches** -- 3 losses (Liverpool 0-3, Tottenham 0-3, Brighton 1-2), 1 win
(Fulham 1-0) and 1 draw (Palace 0-0). `southampton_manutd` (a 3-0 Man Utd win) is processed
but excluded: its team anchor collapsed (149084 team-0 vs 670 team-1 player rows, balance
0.004), so it has no usable Man Utd side. That leaves the win column thinner than the corpus
headline suggests -- stated so the n is honest.

**n=5, each state a slice of an already ball-gap-limited base** - counts (`frames`,
`outside-third losses`) are on every row so small samples are visible. Two matches barely
have a level state: Tottenham (opener ~3', so level is only its first ~160 s) and Southampton
(excluded). The leading state exists only in the Fulham match (~5 min after the 87' winner,
tiny n). Absolute line heights carry the ~+11m partial-broadcast inflation: read across
states, not against FIFA numbers.

## The money comparison: level-state counter-press by result

Man Utd's counter-press while the game is still level (0-0), grouped by how the match ended.
This is the direct test of the Liverpool case-study finding -- was the press flat from
kickoff a losses pattern, or Liverpool-specific?

| result group | match | level-state losses | counter-press frac | 5s regain |
|---|---|---|---|---|
| loss | Liverpool (0-3 loss) | 22 | 0.455 | 0.227 |
| loss | Tottenham (0-3 loss) | 1 | 0.000 | 0.000 |
| loss | Brighton (1-2 loss) | 64 | 0.719 | 0.406 |
| win | Fulham (1-0 win) | 71 | 0.718 | 0.465 |
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
| Palace | level | in_poss | 907 | 45.9 | 32.6 | 52.8 |
| Palace | level | trans_pos | 192 | 42.8 | 32.9 | 48.8 |

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

1. THE finding does NOT repeat. The Liverpool 'counter-press flat from kickoff' was Liverpool-
   specific, not a losses pattern. Level-state (0-0) counter-press: Liverpool 0.455 (flat, low) BUT
   Brighton - also a loss - 0.719, essentially identical to Fulham (win 0.718) and Palace (draw
   0.750). A losing side pressed exactly as hard at 0-0 as the win and the draw did. The collapse
   did not precede the scoreline against Brighton.
2. Tottenham cannot be tested for this. It conceded at ~3', so its level state holds a single
   outside-third loss (0.000 counter-press on n=1 - meaningless). When a team goes behind almost
   immediately there is no level-state sample to ask 'did the collapse precede the goal?'.
3. So across the three losses the pre-scoreline-collapse claim is 1 for, 1 against, 1 unevaluable:
   Liverpool shows it, Brighton contradicts it, Tottenham can't be judged. Not a repeatable pattern -
   report the Liverpool case as a single-match observation, and the wider corpus argues against
   generalizing it.
4. Where the two 0-3 losses DO stand out is the match-level / chasing press, not the level state.
   Tottenham's chasing counter-press 0.489 (regain 0.255) and Liverpool's match-level 0.600 are the
   corpus lows - but that is press while ALREADY behind, confounded with game state, and Liverpool's
   even rose with the deficit (0.455 level -> 0.660 chasing). Heavy losses show a weak press overall,
   not a weak press before the scoreline.
5. Leading = drop deep survives only as the one-match Fulham signal (tiny n): in the ~5 min at 1-0 up
   Man Utd sat back (in-possession build-up 46.1 m vs 56.9 at level; deepest-line 41.1 vs 49.8).
6. Shape shifts with state, confirming the segmentation tracks something real: trailing sides push
   the line and centroid higher (Tottenham chasing in_poss build-up 58.3 m; Liverpool build-up rises
   into 0-3). Ignore the Tottenham level rows (n=1/12 frames - garbage from the ~3' opener).
7. Pass balance (floor, liverpool + fulham only) still follows the scoreline: the trailing team sees
   marginally more of the ball (Liverpool ManU 112:100 chasing vs 69:74 level).
8. Honest n: 5 usable matches (southampton excluded, anchor collapse), so the 'wins' side of the
   comparison is Fulham alone plus the Palace draw; level-state samples range 1 (Tottenham) to 71
   (Fulham). The one robust cross-match statement is the negative one: the flat-press-from-kickoff is
   not a Man-Utd-in-losses law, it is what happened against Liverpool.
