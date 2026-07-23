# Case study: three ways Man Utd lost (Liverpool, Tottenham, Brighton)

Three defeats, grounded **only in validated numbers**: the tracking-native style fingerprint
(B-4) re-bucketed by the validated goal timeline (E2E-Spot goal peaks, per-half counts matched
to the Sofascore split). States are from Man Utd's perspective. This file supersedes the
single-match Liverpool case study -- the n=6 corpus retracted the 'counter-press collapses
before the scoreline' reading as **Liverpool-specific** (STATUS 2026-07-20), so the three
losses are framed as three *different* shapes, not one pattern. Every B-4 ceiling applies
(ball-gap possession base, partial-broadcast line inflation ~+11m); `frames`/`losses` counts
are shown so the samples stay visible.

## Headline: three different ways to lose

Across the three defeats there is no single failure mode -- the n=6 corpus retracted that idea. Man
Utd lost to Liverpool, Tottenham and Brighton in three different shapes:

- **Liverpool (0-3):** beaten in the win-it/lose-it phase from kickoff. At 0-0 their counter-press
  fired on only 0.455 of outside-third losses (22 losses) and their attacking transition was shallow
  (build-up 45.3 m); the goals then pinned them deeper. This is the one match where the collapse
  preceded the scoreline -- and it did NOT generalise.
- **Tottenham (0-3):** behind from the ~3rd minute (162.5 s opener), so there is essentially no level
  state to judge (1 outside-third loss at 0-0). The loss shape is 'conceded early, chased all game' --
  the whole match is the chasing state, and the fingerprint can say nothing about their 0-0 posture.
- **Brighton (1-2):** pressed NORMALLY at 0-0 (0.719 -- identical to the Fulham win 0.718) and were
  highly territorial (att-3rd control 0.262, behind only the Southampton win's 0.353), yet still lost. They
  equalised for 1-1 and conceded a 90+' winner. This is a 'front-foot but couldn't hold on' defeat,
  the direct counter-example to the Liverpool press-collapse reading.

## The three losses at level state (0-0), side by side

From the level-state synthesis (`results/WINS_VS_LOSSES.md`) -- the shape *before* any goal:

| match | level losses | cp frac | 5s regain | block height | att-3rd ctrl | trans depth |
|---|---|---|---|---|---|---|
| Liverpool (0-3 loss) | 22 | 0.455 | 0.227 | 40.6 | 0.180 | 45.3 |
| Tottenham (0-3 loss) | 1 | 0.000 | 0.000 | 49.9 | 0.000 | 24.4 |
| Brighton (1-2 loss) | 64 | 0.719 | 0.406 | 46.9 | 0.262 | 61.4 |
| Fulham (1-0 win) | 71 | 0.718 | 0.465 | 38.0 | 0.173 | 57.9 |
| Southampton (0-3 win) | 26 | 0.423 | 0.192 | 53.7 | 0.353 | 51.1 |

## Liverpool (0-3 loss)

Manchester United did not need the scoreboard to start losing to Liverpool. At 0-0 across the first 34
minutes their counter-press fired on only 0.455 of outside-third losses (22 losses) -- barely two-
thirds of the ~0.72 they managed at level state against Brighton (0.719) and Fulham (0.718) -- and
their win-it-and-go attacking transition was the shallowest at level state of the three losses
(build-up 45.3 m). The two first-half goals (34:26, 41:58) then pinned United deeper still (in-
possession build-up 51.2 m at 0-0 -> 41.4 m at 0-1), and only once 0-3 down after 57' did they finally
push up and press hardest (0.690) -- energy that arrived when the game was already gone. Liverpool lost
the ball outside their own third the fewest of any side (58) and felt the least urgency to win it back
(0.241 five-second regain): United were beaten in the win-it/lose-it phase before the deficit forced
their hand. This is the single match behind the (retracted) 'press collapses before the scoreline'
reading.

Score-state timeline:

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 2066.0     False
  h1 chasing       0-1 2066.0 2518.0     False
  h1 chasing       0-2 2518.0 2940.0      True
  h2 chasing       0-2    0.0  739.0     False
  h2 chasing       0-3  739.0 3116.5      True
```

Man Utd shape, scoreline by scoreline (`def_line_height` = deepest line, `buildup_height` = mean outfield line):

```
scoreline     phase  frames  def_line_height  width  compactness  buildup_height
      0-0   in_poss     194             44.1   33.2         13.3            51.2
      0-0  out_poss     231             40.6   27.8         10.2            45.6
      0-0 trans_neg     147             53.5   28.7         10.8            58.1
      0-0 trans_pos     136             39.4   31.1         12.3            45.3
      0-1   in_poss      98             33.6   32.8         13.7            41.4
      0-1  out_poss       8             50.9   28.0          9.8            54.5
      0-1 trans_neg      25             30.9   25.9         10.3            35.6
      0-1 trans_pos      22             27.9   23.4          9.7            32.5
      0-2   in_poss      87             42.3   35.4         14.9            51.1
      0-2  out_poss      67             40.6   26.7         10.7            45.4
      0-2 trans_neg      57             50.8   28.1         11.0            55.9
      0-2 trans_pos      50             39.1   25.2         11.3            45.3
      0-3   in_poss     285             51.2   30.9         12.2            57.1
      0-3  out_poss     130             48.3   28.4         10.4            52.4
      0-3 trans_neg     164             55.4   25.8         10.5            60.2
      0-3 trans_pos     140             50.3   29.7         12.2            56.3
```

Man Utd counter-press, scoreline by scoreline:

```
scoreline  losses_outside_third  counterpress_frac  regain_5s_frac
      0-0                    22              0.455           0.227
      0-1                     2              0.500           0.000
      0-2                     9              0.556           0.333
      0-3                    42              0.690           0.429
```

Attributed-pass balance by state (floor):

```
  state  ManU_pass  Liverpool_pass
  level         69              74
chasing        112             100
```

## Tottenham (0-3 loss)

Tottenham is the loss the fingerprint cannot dissect. Man Utd conceded at ~3' (162.5 s), so the level
state holds a single outside-third loss (0.000 counter-press on n=1 -- meaningless) and one in-
possession frame. From then on the entire match is the chasing state: 0-1, then 0-2 early in H2
(210.5 s), then 0-3 (2005.5 s). Chasing, Man Utd pushed the line and centroid up (in-possession
build-up in the high 50s m) and their counter-press ran 0.489 with a 0.255 regain -- among the corpus
lows, but that is press while already two/three down, confounded with game state. The honest read is
structural: when a side concedes in the third minute there is no 0-0 sample to ask how they set up,
and this defeat's shape is simply 'behind from minute three, never level again'.

Score-state timeline:

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0  162.5     False
  h1 chasing       0-1  162.5 2940.0      True
  h2 chasing       0-1    0.0  210.5     False
  h2 chasing       0-2  210.5 2005.5     False
  h2 chasing       0-3 2005.5 3156.4      True
```

## Brighton (1-2 loss)

Brighton is the direct counter-example to the Liverpool reading. At 0-0 Man Utd pressed hard (counter-
press 0.719, essentially the Fulham-win number 0.718) and were among the most territorial sides in
possession -- 0.262 of their level in-possession frames had the team's mean line beyond the 70 m third
(behind only the corrected Southampton win's 0.353). Their attacking transition at 0-0 was the deepest
of the three losses (build-up 61.4 m) and they defended from a higher block (46.9 m). None of that is a
pre-scoreline collapse. They fell behind 0-1, equalised for 1-1, and conceded a 90+' stoppage-time
winner (Joao Pedro) to lose 1-2. This is a front-foot, high-territory performance that lost late --
pressing and possession at 0-0 looked like a win, and the result did not follow.

Score-state timeline:

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 1895.5     False
  h1 chasing       0-1 1895.5 2895.0      True
  h2 chasing       0-1    0.0  794.5     False
  h2   level       1-1  794.5 2881.0      True
  h2 chasing       1-2 2881.0 3086.8      True
```

## Named-player exemplars (observations, not stats)

Floor observations from the attributed-pass ledger (`results/PLAYER_LEDGER.md`): a player is named
only when a named track is the on-ball carrier within 3 m of a filtered PASS, so coverage is sparse
(3.6-12.4% of team passes) and these are 'observed on at least N fragments' exemplars -- NOT a passing
ranking and NOT comparable to the Sofascore totals. Man Utd exemplars per loss:

- **Liverpool (coverage 5.6%):** Marcus Rashford (4 attributed fragments), with Lisandro Martinez and
  Noussair Mazraoui (2 each) also named on the pitch. On the Liverpool side Alexis Mac Allister drew
  the most fragments of anyone (5) and Mohamed Salah was identity-assigned but drew zero -- named
  presence, not an involvement count.
- **Tottenham (coverage 12.4%, the richest of the losses):** Amad Diallo, Alejandro Garnacho and Diogo
  Dalot (2 fragments each) for Man Utd; on the Tottenham side Dejan Kulusevski (5), Micky van de Ven,
  Timo Werner, James Maddison, Manuel Ugarte and Rodrigo Bentancur (4 each) were all named.
- **Brighton (coverage 3.6%):** Kobbie Mainoo (3 fragments) top for Man Utd, with Amad Diallo, Harry
  Maguire and Lisandro Martinez (2 each) -- Maguire is one of the players surfaced by the roster-mask
  levers. These are presence/involvement floors, never per-player rates.

## Abstentions (stated proudly)

- No goal-scorer identity is claimed from CV: goal *times* are the validated E2E spots, goal
  *ownership* is the Sofascore per-half delta. We do not name who scored from tracks.
- Per-scoreline slices (Liverpool 0-1, 0-2) rest on <100 in-possession frames and single-digit losses
  -- the arrows (deeper when pinned, higher when 0-3 down) are read as direction, not magnitude.
- Tottenham's level state (1 loss, 1 in-possession frame) is not evaluated; its 0-0 posture is unknown.
- Absolute line heights are not FIFA metres (partial-broadcast inflation ~+11m); only within-match,
  across-state comparisons are made.
- Player pass counts are a floor bounded by named-fragment coverage, never a ranking.
