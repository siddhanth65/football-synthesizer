# Wins vs losses: is there a repeatable win-shape at level state? (Plan B-5 synthesis)

The corpus now has **two real wins** (Fulham 1-0, Southampton 0-3) against **three losses**
(Liverpool 0-3, Tottenham 0-3, Brighton 1-2) and one draw (Palace 0-0). The question Sid
asked originally: *how do they win vs how do they lose*, in the tracking-native metrics. To
avoid the game-state confound we test at **LEVEL state only** (scoreline 0-0, before any goal
moves Man Utd off level and changes their posture). Every number is the B-4 fingerprint
re-bucketed by the validated goal timeline; engine `fingerprint/score_state.py`. Pre-committed
metrics, all at 0-0: counter-press fraction, 5 s regain, defensive block height (out-of-
possession deepest line), attacking-third control (share of in-possession frames with the
team's mean line beyond the 70 m third), and attacking-transition depth.

## Level-state comparison, all six (grouped win / loss / draw)

| group | match | level losses | cp frac | 5s regain | block height (out-poss) | att-3rd control | trans depth |
|---|---|---|---|---|---|---|---|
| win | Fulham (1-0 win) | 71 | 0.718 | 0.465 | 38.0 | 0.173 | 57.9 |
| win | Southampton (0-3 win) | 33 | 0.394 | 0.333 | 32.9 | 0.111 | 46.2 |
| loss | Liverpool (0-3 loss) | 22 | 0.455 | 0.227 | 40.6 | 0.180 | 45.3 |
| loss | Tottenham (0-3 loss) | 1 | 0.000 | 0.000 | 49.9 | 0.000 | 24.4 |
| loss | Brighton (1-2 loss) | 64 | 0.719 | 0.406 | 46.9 | 0.262 | 61.4 |
| draw | Palace (0-0 draw) | 40 | 0.750 | 0.375 | 34.8 | 0.127 | 48.8 |

Level-state sample sizes (the honesty counters -- several slices are small):

| match | in-poss frames | out-poss frames | trans_pos frames | outside-third losses |
|---|---|---|---|---|
| Liverpool (0-3 loss) | 194 | 231 | 136 | 22 |
| Tottenham (0-3 loss) | 1 | 32 | 12 | 1 |
| Brighton (1-2 loss) | 344 | 535 | 387 | 64 |
| Fulham (1-0 win) | 549 | 441 | 343 | 71 |
| Southampton (0-3 win) | 63 | 76 | 82 | 33 |
| Palace (0-0 draw) | 907 | 182 | 192 | 40 |

## The money question: is there a distinction, or is it n-limited noise?

**Short answer: no repeatable win-shape in the PRESS; a weak, consistent-direction win-shape in the
defensive BLOCK. Small n -- read as direction, not law.**

1. Counter-press fraction at 0-0 does NOT separate wins from losses. The two wins straddle the loss
   range: Fulham 0.718 sits right on top of the Brighton loss (0.719), while Southampton 0.394 falls
   BELOW the Liverpool loss (0.455). A team that pressed 0.394 at 0-0 won 3-0; a team that pressed
   0.719 at 0-0 lost 1-2. Level-state press is n-limited noise here.

2. 5 s regain and transition depth overlap the same way. Regain: wins 0.333-0.465, losses 0.227-0.406
   -- the Brighton loss (0.406) beats the Southampton win (0.333). Transition depth: wins 46.2-57.9,
   losses 45.3-61.4 -- total overlap. Neither is a win-shape.

3. The ONE metric that separates: defensive block height at 0-0 (out-of-possession deepest line).
   Both wins defended the deepest of the six -- Southampton 32.9 m and Fulham 38.0 m -- below all
   three losses (Liverpool 40.6, Brighton 46.9, Tottenham 49.9), with the goalless draw (34.8) sitting
   in the win band. Even dropping the unevaluable Tottenham level (32 out-poss frames), the two wins
   (32.9, 38.0) still sit under the two evaluable losses (40.6, 46.9).

4. Attacking-third control points the same way. The two wins held the ball in the attacking third the
   LEAST at 0-0 (Southampton 0.111, Fulham 0.173) while the Brighton loss was the most territorial
   (0.262). So the win-shape at level state is a DEEPER, LESS territorial block -- a control/counter
   posture -- not a front-foot press. That is consistent with the Southampton report: Man Utd's press
   rose after they went ahead, they did not out-press at 0-0.

5. The honest ceiling. This is a two-win signal, and one of the wins (Southampton) has a tiny level
   sample: 63 in-possession, 76 out-of-possession, 82 transition frames, 33 outside-third losses (the
   ~35 min before the opener). The block-height gap (~5-15 m) lives partly inside the ~+11 m partial-
   broadcast inflation band and is confounded with venue/opponent territory. And the goalless DRAW
   sits with the wins on block depth, so 'deep block' is better read as a **did-not-lose** shape than
   a **win** shape. Verdict: wins-vs-losses in the press is noise at this n; wins-vs-losses in
   defensive block depth (and attacking-third control) is a weak, consistent-direction territorial
   signal -- a real 'how they win vs how they lose' arrow, but one that needs more wins to confirm,
   not a validated law.
