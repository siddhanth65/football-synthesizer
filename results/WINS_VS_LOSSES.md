# Wins vs losses: is there a repeatable win-shape at level state? (Plan B-5 synthesis)

> **2026-07-23 CORRECTION:** `southampton_manutd` + `tottenham_manutd` team mappings corrected (see `results/PAIR_ANALYSIS_v1.md`). The Southampton (0-3 win) Man Utd numbers below are recomputed with team0 = Man Utd; the prior **wins = deeper-block** claim is retracted/revised (United's true level-state block in that win is high, 53.7 m, not deep).

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
| win | Southampton (0-3 win) | 26 | 0.423 | 0.192 | 53.7 | 0.353 | 51.1 |
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
| Southampton (0-3 win) | 85 | 79 | 88 | 26 |
| Palace (0-0 draw) | 907 | 182 | 192 | 40 |

## The money question: is there a distinction, or is it n-limited noise?

_2026-07-23 CORRECTION: southampton_manutd + tottenham_manutd team mappings corrected (see
results/PAIR_ANALYSIS_v1). The Southampton (0-3 win) level-state row previously carried the OPPONENT's
numbers -- Man Utd's true level-state block is HIGH (53.7 m), not deep (32.9 m). The prior
"wins = deeper block" arrow is retracted below._

**Short answer: after the mapping correction there is NO repeatable win-shape at level state -- not in
the press, not in the block, not in territory. The earlier "wins defend a deeper block" finding was an
artifact of the mislabelled Southampton leg. Small n -- read as direction, not law.**

1. Counter-press fraction at 0-0 does NOT separate wins from losses. The two wins straddle the loss
   range: Fulham 0.718 sits right on top of the Brighton loss (0.719), while Southampton 0.423 falls
   BELOW the Liverpool loss (0.455). A team that pressed 0.423 at 0-0 won 3-0; a team that pressed
   0.719 at 0-0 lost 1-2. Level-state press is n-limited noise here.

2. 5 s regain and transition depth overlap the same way. Regain: wins 0.192-0.465, losses 0.000-0.406
   -- the Brighton loss (0.406) sits well ABOVE the Southampton win (0.192). Transition depth: wins
   51.1-57.9, losses (evaluable) 45.3-61.4 -- total overlap. Neither is a win-shape.

3. Block height at 0-0 does NOT separate wins from losses -- the corrected headline. The Southampton
   win defended the HIGHEST block of all six (53.7 m), ABOVE every loss (Liverpool 40.6, Brighton 46.9,
   Tottenham 49.9); the Fulham win (38.0 m) was among the deepest. The two wins sit at OPPOSITE ends of
   the block-height range. The pre-correction claim that "both wins defended the deepest" was the
   Southampton mapping flip: United's true level-state line in that win is high, not deep.

4. Attacking-third control fails to separate too. The Southampton win was the MOST territorial side at
   0-0 (0.353, above the Brighton loss 0.262) while the Fulham win was mid (0.173). One win pressed
   high and territorial, the other sat deeper and less territorial -- no shared win posture.

5. The honest ceiling. Two wins, opposite shapes; the Southampton level sample is modest (85 in-
   possession, 79 out-of-possession, 88 transition frames, 26 outside-third losses -- the ~35 min
   before the opener) and every line carries the ~+11 m partial-broadcast inflation. Verdict after
   correction: there is NO win-shape vs loss-shape at level state in this corpus -- the prior
   "wins = deeper, less-territorial block" arrow is RETRACTED, having rested entirely on the
   mislabelled Southampton leg. More wins are needed before any 'how they win vs how they lose' claim.
