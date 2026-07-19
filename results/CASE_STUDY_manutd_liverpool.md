# Case study: Manchester United 0-3 Liverpool (Plan B-5)

How the 0-3 happened phase by phase, grounded **only in validated numbers**: the
tracking-native style fingerprint (B-4) re-bucketed by the validated goal timeline
(E2E-Spot goals, validated 3/3 with correct halves: 34:26 and 41:58 in H1, 12:19 in H2 --
all Liverpool). States are from Man Utd's perspective. Every ceiling from B-4 still applies
(ball-gap possession base, partial-broadcast line inflation ~+11m); per-scoreline slices are
small - `frames` and `losses` are shown so the samples are visible.

## Headline

Manchester United did not need the scoreboard to start losing to Liverpool. At 0-0 across the first
34 minutes their counter-press already fired on only 0.455 of outside-third losses (22 losses) --
barely two-thirds of the ~0.72 they managed at level state against both Brighton (0.719) and Fulham
(0.718) -- and even their win-it-and-go attacking transition was the shallowest at level state of the
three, build-up 45.3 m versus 61.4 m (Brighton) and 57.9 m (Fulham). The two first-half goals (34:26,
41:58) then pinned United deeper still (in-possession build-up 51.2 m at 0-0 -> 41.4 m at 0-1, and the
attacking transition down to 32.5 m at 0-1 -- small samples), and only once 0-3 down after
57' did they finally push up (in-possession build-up 57.1 m, deepest-line 51.2 m) and press hardest
(0.690) -- energy that arrived when the game was already gone. Liverpool, for their part, lost the
ball outside their own third the fewest times of any side across the three matches (58) and felt the
least urgency to win it back (0.241 five-second regain): United were beaten in the win-it/lose-it
phase before the deficit ever forced their hand.

## Score-state timeline

```
half   state scoreline   t0_s   t1_s  stoppage
  h1   level       0-0    0.0 2066.0     False
  h1 chasing       0-1 2066.0 2518.0     False
  h1 chasing       0-2 2518.0 2940.0      True
  h2 chasing       0-2    0.0  739.0     False
  h2 chasing       0-3  739.0 3116.5      True
```

## Man Utd shape, scoreline by scoreline

`def_line_height` = deepest-line attacking-x, `buildup_height` = mean outfield attacking-x.

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

## Man Utd counter-press, scoreline by scoreline

```
scoreline  losses_outside_third  counterpress_frac  regain_5s_frac
      0-0                    22              0.455           0.227
      0-1                     2              0.500           0.000
      0-2                     9              0.556           0.333
      0-3                    42              0.690           0.429
```

## Attributed-pass balance by state (floor)

```
  state  ManU_pass  Liverpool_pass
  level         69              74
chasing        112             100
```
Near-even at level (69:74) and Man Utd marginally ahead when chasing (112:100) - the trailing
side pushes and sees more of the ball. A floor (carried passes only); read as balance, not
totals.

## Named-player exemplars (observations, not stats)

- **Mohamed Salah** and **Alexis Mac Allister** were both identity-assigned on the pitch for this
  match (part of the 20-player Liverpool identity chain, confidence 0.78-1.00). These are named-track
  observations, not per-player metrics.
- In the attributed-pass ledger (a coverage floor: only ~5.6% of Liverpool's team passes carry a
  named track), Mac Allister appears on the most fragments of any player (5 attributed passes),
  Marcus Rashford next (4). These are "observed at least N" exemplars - NOT a passing ranking and not
  comparable to the Sofascore totals (Mac Allister's true totalPass was 49).
- Salah was named on the pitch but drew zero attributed pass fragments in the floor; we name his
  presence, we do not claim his involvement count.

## Abstentions (stated proudly)

- No goal-scorer identity is claimed from CV: goal *times* are the validated E2E spots, goal
  *ownership* is the Sofascore per-half delta (all three Liverpool). We do not name who scored from
  tracks.
- Per-scoreline slices (0-1, 0-2) rest on <100 in-possession frames and single-digit losses - the
  arrows (deeper when pinned, higher when 0-3 down) are read as direction, not magnitude.
- Absolute line heights are not FIFA metres (partial-broadcast inflation); only within-match,
  across-state comparisons are made.
- Player pass counts are a floor bounded by named-fragment coverage, never a ranking.
