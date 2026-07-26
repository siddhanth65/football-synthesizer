# Manchester United 2024-25 -- a season identity read from twelve broadcasts

Twelve Premier League matches, six opponents played twice, one manager change in the middle. This is what the footage will say about who this team is -- and, just as often, what it refuses to say.

**The one substitution that makes this honest.** Everything below is built on *position* and *territory*: where the ball travels, where players stand, how the defensive block is shaped. Nothing is built on per-player event counts, because naming the man on the ball works on only 1-4% of United's passes (`results/PLAYER_ANALYSIS_v2.md`). So you will read "he operates between the lines" and never "he made 62 passes". And there are no *we-do-this-more-than-they-do* rate comparisons either: separating two teams on a pressing rate needs 58-90 matches and we have twelve (`results/W1B_WINDOW_AND_SAMPLING.md`). Where the honest answer is "we cannot tell", it is written as "we cannot tell".

---

## The reads this corpus actually supports

1. **They come out through the middle and finish down the sides -- and so does everybody else.** 46.9% of United's ball time in their own third is in the central channel; by the final third that is down to 23.9%, with 76.1% of it out wide. The swing repeats in 11 of 12 matches (median 25.0 percentage points, sign test p = 0.0063). But the same six opponents do it in 12 of 12 (p = 0.0005) and finish *more* wide-heavy than United do (22.6% central to United's 23.9%). So: yes, United attack down the flanks -- and no, that is not a United trait. It is what a Premier League attack looks like.

2. **There is no favoured flank.** Pooled, the final third splits 39.9% left to 36.2% right, which reads like a tilt until you count matches: 5 of 12 lean left, the rest lean right (median tilt -8.0 percentage points, sign test p = 0.7744). Whatever side a given afternoon runs down, it is the afternoon, not the team. We decline the left-side / right-side read.

3. **The territory map is real; the volume map is not.** Harry Maguire is the floor of the build (36.5 m from his own goal, tenth percentile 8.7 m, 42.3% of his time in his own third, on 168 tracked frames). Alejandro Garnacho is the ceiling (64.3 m, 48.6% of his frames in the final third, and a deepest tenth of 38.6 m -- he does not come back). Bruno Fernandes is the man the cameras find most (2404 named frames under Amorim, close to double the next), and his ground covers all three zones (19.7% own / 54.6% middle / 25.7% final). That is the honest sentence about him: we can see where he plays, not how much he does.

4. **Rigid? Not on the one axis we can measure.** United's defensive line averages 28.3 m from their own goal with a between-match SD of **4.25 m** (CV 15.0%) and a range of 20.8-37.4 m across 12 matches -- a 16.6 m swing. Against the same opponent in two legs it moves 5.4 m on average, and the leg-to-leg correlation is **-0.40** (`results/PAIR_ANALYSIS_v1.md`) -- knowing where they defended at home tells you nothing about the away leg. The whole opponent set pooled sits at 3.09 m SD, i.e. no more variable than United despite containing six different clubs (F = 1.89, p = 0.305, so the two cannot be separated at n = 12). Their height is set by the afternoon, not by a principle.

5. **The one thing that never moves is not theirs.** Block vertical compactness sits at 6.17 +/- 0.39 m (CV 6.3%) across 12 matches, six opponents, two managers, two venues and three game states. That looks like the rigidity claim -- until the same measurement on the opposition comes back at 5.93 +/- 0.29 m (CV 4.9%), statistically indistinguishable (F = 1.81, p = 0.34). A constant that holds for all seven sides is a property of the measurement, or of the league, not a fingerprint of one of them. Declined.

6. **Nobody comes to press them.** In all 12 legs the opposition block sits between 19.9 m and 29.7 m from its own goal -- low to mid, every time, home and away, under both managers. Whatever United are, teams do not fancy going after them high up the pitch.

7. **A lead pushes the line up and leaves the shape alone.** On the six matches with a calibrated win-probability series, United defend at 29.1 m when a loss is likely and 36.6 m when a win is likely (+7.5 m), while compactness moves 0.19 m (6.06 -> 5.87). Hold that one loosely: the win-likely band is 879 frames and most of it is one afternoon at Southampton.

8. **The manager change shows up in the players, not in the team's shape.** 7 of the 9 players tracked in both eras stand higher up the pitch under Amorim, Garnacho by +15.1 m. The block line goes 29.7 -> 26.9 m -- a 2.8 m move inside a 4.25 m match-to-match SD, which is not a separation. And United played three of their six ten Hag matches at home and three of six Amorim matches at home in *the reverse* fixtures, so venue and manager flip together in every pair: no number below can tell the two apart.

---

## 1. How they build

Follow the ball rather than the formation and United's attack has a clear shape to it, and it is not the one the phrase "they build through the wingers" suggests. They start in the middle. In their own third the ball spends 46.9% of its time in the central channel, against 26.5% left and 26.6% right -- centre-backs and goalkeeper circulating in front of their own box, which is exactly what you would expect and is partly the goalkeeper's doing, since he is central by definition.

Then it drifts outwards. By the middle third the centre is down to 38.1%, and in the final third it is 23.9% -- more than three quarters of United's attacking third ball time is in the two wide channels. Central at the back, wide at the front. It is a proper pattern, not a pooled illusion: it shows up in 11 of 12 individual matches (median swing 25.0% points, sign test p = 0.0063).

And here is where a season profile has to be careful, because the same measurement run on the other side of the same twelve broadcasts gives the opposition 47.8% central at the back and 22.6% central at the front -- the identical funnel, in 12 of 12 legs, and finishing marginally *wider* than United do. The wide attacking third is Premier League geometry, not Manchester United's identity. If you want the honest headline: **they attack down the flanks like everyone attacks down the flanks, and there is nothing in this footage that makes it theirs.**

**United's ball map (12 matches, 18483 possession-credited ball samples)**

| zone | left | central | right | wide total |
|------|-----:|--------:|------:|-----------:|
| own third | 26.5% | 46.9% | 26.6% | 53.1% |
| middle third | 33.5% | 38.1% | 28.4% | 61.9% |
| final third | 39.9% | 23.9% | 36.2% | 76.1% |

**The same twelve broadcasts, the opposition's ball (19955 samples)**

| zone | left | central | right | wide total |
|------|-----:|--------:|------:|-----------:|
| own third | 24.5% | 47.8% | 27.8% | 52.3% |
| middle third | 28.3% | 36.7% | 35.0% | 63.3% |
| final third | 38.6% | 22.6% | 38.9% | 77.5% |

**Left or right?** No. Pooled it looks like a left lean (39.9% to 36.2% in the final third), but match by match it is 5 leaning left and 7 leaning right (sign test p = 0.7744). United do not have a side. We say so rather than pick the one that reads better.

**What we cannot say here.** Where the ball *crosses into* the final third by channel -- the sharpest version of the question -- rests on only 43 cleanly-chained crossings across twelve matches, four or five a game. That is not a sample; it is an anecdote with a decimal point. Declined.

---

## 2. Who occupies what space

Nine of the twelve matches have the identity layer run on them, which means named players on named tracks. What follows is a map of ground occupied. It is not a ranking, it is not an involvement table, and it never will be from this footage.

The spine reads exactly as a spine should. **Harry Maguire is the floor of the build** -- mean position 36.5 m from his own goal, tenth percentile 8.7 m, 42.3% of his tracked time inside his own third, on 168 tracked frames. **Alejandro Garnacho is the ceiling** -- 64.3 m, 48.6% of his frames in the final third, and even his deepest tenth is 38.6 m: he is not a winger who tracks all the way back. Joshua Zirkzee sits further forward still on the table below -- 83.9 m, 94% of his frames in the final third -- but on 83 frames, about seventeen seconds of tracked football, so he is listed and not leaned on.

**Bruno Fernandes** is the player the profile most wants to talk about and can say least about. He is the most-tracked United outfielder in the corpus by a distance (2404 named frames under Amorim, close to double the next man), his mean position is 52.4 m, and his ground spans all three zones (19.7% own third, 54.6% middle, 25.7% final). Laterally he is almost perfectly even (0.34 left / 0.40 central / 0.26 right), so the fashionable "right half-space" line is not available -- he does not sit in one. He is everywhere, and being seen everywhere is partly a fact about him and partly a fact about where the camera points. **We can see where he plays. We cannot see how much he does.** Calling him the primary playmaker off this evidence would be a guess in a lab coat.

**Lateral labels come with a warning.** Left and right are anchored on players whose side is known rather than assumed: Alejandro Garnacho, Lisandro Martínez, Marcus Rashford pool to y = 27.8 m and Amad Diallo, Antony, Diogo Dalot, Noussair Mazraoui to y = 41.5 m, so the labels are the right way round (check: PASS). At the individual level it is directional only -- 8 of 10 known-sided player-eras land on the correct flank, which is good enough for "he plays wide left" and not good enough for a half-space claim.

### Territory map -- Amorim era

| player | frames | mean advance (m) | p10 | p90 | front-back spread | side-to-side spread | own/mid/final | L/C/R |
|--------|-------:|-----------------:|----:|----:|------------------:|--------------------:|---------------|-------|
| Harry Maguire | 168 | 36.5 | 8.7 | 63.6 | 20.7 | 10.0 | 0.42/0.54/0.04 | 0.14/0.77/0.09 |
| Noussair Mazraoui | 557 | 42.7 | 24.2 | 70.6 | 18.3 | 16.2 | 0.41/0.46/0.12 | 0.07/0.39/0.54 |
| Kobbie Mainoo | 178 | 43.9 | 24.0 | 70.6 | 16.9 | 15.6 | 0.33/0.57/0.11 | 0.45/0.40/0.15 |
| Manuel Ugarte | 437 | 44.7 | 23.6 | 70.4 | 18.1 | 12.4 | 0.32/0.57/0.11 | 0.11/0.60/0.29 |
| Lisandro Martínez | 177 | 46.8 | 18.4 | 81.1 | 24.1 | 10.6 | 0.40/0.37/0.23 | 0.65/0.32/0.03 |
| Diogo Dalot | 1314 | 49.1 | 22.7 | 77.6 | 19.7 | 21.3 | 0.26/0.55/0.19 | 0.24/0.31/0.44 |
| Bruno Fernandes | 2404 | 52.4 | 21.7 | 82.2 | 21.0 | 17.4 | 0.20/0.55/0.26 | 0.34/0.40/0.26 |
| Casemiro | 33 | 54.5 | 19.9 | 72.4 | 20.0 | 4.3 | 0.12/0.67/0.21 | 0.00/1.00/0.00 |
| Leny Yoro | 432 | 54.8 | 41.7 | 74.2 | 13.7 | 12.0 | 0.03/0.76/0.22 | 0.60/0.32/0.07 |
| Amad Diallo | 1336 | 55.0 | 27.7 | 81.3 | 18.1 | 14.9 | 0.15/0.64/0.22 | 0.15/0.36/0.50 |
| Rasmus Højlund | 195 | 55.9 | 45.7 | 75.1 | 11.1 | 18.5 | 0.01/0.82/0.16 | 0.48/0.33/0.19 |
| Alejandro Garnacho | 547 | 64.3 | 38.6 | 82.1 | 18.2 | 18.6 | 0.08/0.44/0.49 | 0.61/0.22/0.17 |
| Joshua Zirkzee | 83 | 83.9 | 84.5 | 87.7 | 11.9 | 1.8 | 0.02/0.04/0.94 | 0.00/1.00/0.00 |

### Territory map -- ten Hag era

| player | frames | mean advance (m) | p10 | p90 | front-back spread | side-to-side spread | own/mid/final | L/C/R |
|--------|-------:|-----------------:|----:|----:|------------------:|--------------------:|---------------|-------|
| Kobbie Mainoo | 103 | 38.3 | 25.8 | 59.9 | 11.7 | 16.2 | 0.58/0.42/0.00 | 0.69/0.18/0.13 |
| Lisandro Martínez | 90 | 39.6 | 19.8 | 58.2 | 18.1 | 11.0 | 0.48/0.44/0.08 | 0.62/0.33/0.04 |
| Noussair Mazraoui | 413 | 40.2 | 24.3 | 56.9 | 13.4 | 9.4 | 0.41/0.55/0.03 | 0.01/0.34/0.65 |
| Casemiro | 146 | 46.7 | 30.6 | 61.6 | 11.9 | 10.1 | 0.20/0.77/0.02 | 0.08/0.60/0.32 |
| Bruno Fernandes | 300 | 48.0 | 31.3 | 69.3 | 14.4 | 11.7 | 0.20/0.70/0.10 | 0.29/0.55/0.16 |
| Alejandro Garnacho | 592 | 49.2 | 27.5 | 74.0 | 17.8 | 22.6 | 0.28/0.55/0.17 | 0.37/0.11/0.52 |
| Diogo Dalot | 209 | 49.5 | 29.7 | 70.2 | 16.2 | 11.0 | 0.19/0.70/0.11 | 0.45/0.54/0.01 |
| Antony | 30 | 50.5 | 46.4 | 54.4 | 3.6 | 6.6 | 0.00/1.00/0.00 | 0.00/0.00/1.00 |
| Manuel Ugarte | 109 | 52.8 | 35.6 | 78.3 | 17.1 | 11.3 | 0.10/0.70/0.20 | 0.06/0.49/0.44 |
| Amad Diallo | 121 | 54.4 | 28.3 | 73.3 | 19.2 | 10.7 | 0.35/0.28/0.36 | 0.04/0.40/0.56 |
| Toby Collyer | 28 | 56.4 | 45.1 | 65.3 | 8.2 | 7.6 | 0.00/0.93/0.07 | 0.07/0.86/0.07 |
| Marcus Rashford | 169 | 61.5 | 49.0 | 82.9 | 13.7 | 11.8 | 0.02/0.69/0.29 | 0.83/0.12/0.05 |
| Mason Mount | 26 | 61.8 | 37.6 | 71.5 | 15.0 | 7.1 | 0.00/0.31/0.69 | 0.31/0.69/0.00 |
| Christian Eriksen | 60 | 67.2 | 47.5 | 91.4 | 18.4 | 10.6 | 0.03/0.62/0.35 | 0.65/0.35/0.00 |

Mean advance is metres from United's own goal; p10 and p90 are the deepest and highest tenth of the player's tracked frames; spreads are standard deviations; the last two columns are the share of frames by vertical zone and by lateral channel. Frames are tracked-and-named frames, which track minutes played in *rank* but not in scale.

---

## 3. How they defend

United defend deep and they defend at whatever height the afternoon demands. Across the twelve matches their block line averages 28.3 m from their own goal, but the range is 20.8 to 37.4 m: from a 20.8 m hunker at Anfield to a 37.4 m front-foot line in the 3-0 at Southampton, the highest they defended in the whole corpus and their best result in it.

The other half of the picture is that nobody comes to meet them. In every one of the twelve legs the opposition block sits between 19.9 and 29.7 m from its own goal -- low or mid, home and away, against both managers. There is no high-press exception in this corpus. Whatever else is true of this side, opponents do not think going after them high up the pitch is the way to beat them.

**Read the block heights as relative, not absolute.** The de-biased line carries about 7 m of held-out scale uncertainty and the hand-annotation gate is still pending (`results/BLOCK_AND_STYLE_v1.md`), and the broadcast frames the block preferentially when the ball is further forward, which biases every observed block to read deeper than it truly is. The ordering across matches is the usable signal; the metre label is not certified.

---

## 4. Are they rigid?

This is the claim a fan makes about this side more than any other, so it gets the sharpest treatment. "Rigid" is a statement about variance, and variance is measurable.

**On defensive height, they are the opposite of rigid.** Between-match standard deviation of the block line is **4.25 m** on a mean of 28.3 m -- a coefficient of variation of 15.0% over 12 matches, with a 16.6 m spread from lowest to highest. Play the same opponent twice and the line moves 5.4 m on average between the legs, and the leg-to-leg correlation is **-0.40**: the home leg does not predict the away leg, and if anything predicts it backwards. Whatever sets United's defensive height, it is not a rule the manager has drilled in.

The comparison to keep honest: the pooled opponent set -- six different clubs across the same twelve broadcasts -- sits at 3.09 m SD (CV 12.5%), which is *lower* than United's despite containing between-club variation United's number does not. That is suggestive and it is not significant: F = 1.89 on 11 and 11 degrees of freedom, p = 0.305. So the claim that survives is the absolute one -- United's line is not fixed -- not a league ranking of who wobbles most.

**Leg-to-leg swing against the same opponent**

| opponent | block-line gap between legs (m) | compactness gap (m) |
|----------|-------------------------------:|--------------------:|
| Brighton | 3.6 | 0.29 |
| Crystal Palace | 3.4 | 0.79 |
| Fulham | 1.9 | 0.48 |
| Liverpool | 8.7 | 0.21 |
| Southampton | 12.2 | 1.11 |
| Tottenham | 2.5 | 0.33 |

**On shape, the number that looks rigid is not theirs to own.** The block's front-to-back compactness -- how squeezed the defending unit is between its deepest and highest man -- sits at 6.17 m with an SD of 0.39 m (CV 6.3%), range 5.71-6.94 m. Twelve matches, six opponents, two managers, two venues, three game states, and it barely twitches. That is a rigidity number -- until the same measurement on the opposition returns 5.93 +/- 0.29 m (CV 4.9%), statistically indistinguishable from United's (F = 1.81, p = 0.34). Seven teams, one constant. That is the measurement talking, or the league, not Manchester United. **Declined as an identity claim.**

**Does the shape change with the state of the game?** Only the height does. On the six matches with a calibrated win-probability series:

| game state | frames | block line (m) | compactness (m) |
|------------|-------:|---------------:|----------------:|
| loss-likely | 6126 | 29.1 | 6.06 |
| balanced | 1174 | 29.0 | 6.12 |
| win-likely | 879 | 36.6 | 5.87 |

The line climbs 7.5 m from loss-likely to win-likely while compactness moves 0.19 m. United get braver about where they defend when the game is safe and change nothing about how they defend. Caveat, stated rather than buried: the win-likely row is 879 frames drawn from effectively two matches, most of it the Southampton away win.

**And within a single match?** We cannot tell. The spell-to-spell scatter of the block line inside a match is 20.6 m -- larger than the quantity we would be trying to measure. At that resolution any "they changed shape after the goal" read is measurement noise with a story attached. Declined.

---

## 5. What the manager change did

Ruben Amorim replaced Erik ten Hag between the two halves of this corpus, and the design has a hole in it that has to be said before any number: United played the *reverse* fixture of each pair under the new manager, so **venue and manager flip together in all six pairs**. Nothing below can separate "Amorim" from "away at Anfield instead of home to Liverpool". Every number in this section is a direction with n = 6 a side.

**It shows in the players.** 7 of the 9 players tracked in both eras stand higher up the pitch under Amorim. The mover is Alejandro Garnacho, +15.1 m further forward, and the direction holds through the spine -- Casemiro, Martinez, Mainoo and Fernandes all take a step up.

**It does not show in the team's shape.** The block line goes 29.7 m under ten Hag to 26.9 m under Amorim, a 2.8 m move against a 4.25 m match-to-match SD -- inside the noise, and the direction is dominated by a single high leverage leg (the 37.4 m at Southampton, which sits in the ten Hag column). Compactness goes 5.99 to 6.36 m, which is nothing.

**Nor in where the ball goes.** Own-third central share 49.1% under ten Hag to 44.8% under Amorim; final-third central share 27.9% to 21.1%. The funnel is the same funnel.

The one place a manager signal survives the confound at all is the attack-type mix already on file (`results/GAME_STATE_v2.md`): sustained build-ups fall from 0.592 to 0.468 of typed attacks and direct attacks rise from 0.310 to 0.429. That is 71 and 77 typed attacks respectively at 10-20% labelling coverage -- a direction on a small n, and it is offered as a direction. The formation question (4-2-3-1 to 3-4-3) is a measured negative: the named-player structure shows no wing-back signature at all (`results/PLAYER_ANALYSIS_v2.md`).

---

## What this profile cannot tell you

Bluntly, and with the reason measured:

- **Anything about how much a player did.** No passes, touches, involvements, chances created, plus-minus, or impact ranking. Naming the man on the ball works on **1.0-4.2%** of United's attributed passes -- two to five named passes a match -- and pooling all nine identity matches leaves the busiest player on **6**, against a floor of 20 before Poisson noise drops under 22% (`results/PLAYER_ANALYSIS_v2.md`). The event layer is dead at the player level and no amount of aggregation revives it.
- **Anything of the form "United press more than X".** Team-level counter-press *rate* metrics need **58-90 matches** to separate two sides even with perfect tracking and zero measurement error; we have 12 (`results/W1B_WINDOW_AND_SAMPLING.md`). Any rate comparison against another club would be answering a question whose answer is already fixed by sample size.
- **Which flank they favour.** Sign test p = 0.7744 over 12 matches. There is no tilt to report.
- **Which channel they enter the final third through.** 43 cleanly-chained crossings in 12 matches.
- **Whether the shape changes inside a match.** Spell-level line scatter (20.6 m) exceeds the effect.
- **The formation.** Seven to eleven named players a match cannot distinguish a 4-2-3-1 from a 3-4-3, and the wing-back test came back negative.
- **Set pieces.** We hold corner counts and nothing else.
- **Absolute block heights in FIFA metres.** ~7 m of held-out scale uncertainty, hand-annotation gate pending; the ordering is usable, the label is not.

---

# Evidence appendix

## Provenance

- **Ball layer.** Post-`link_ball` usable ball track per chunk, credited to a team by the Viterbi possession smoother (`generator.ball.assign_possession`, `smooth=True`), oriented so United attack +x. Post-link ball coverage per match: min 0.360, median 0.465, max 0.561 of sampled frames. Channels are equal 22.67 m thirds of the 68 m width; zones are equal 35 m thirds of the 105 m length.
- **Cross-check against the validated event layer.** In the 4 matches that carry an attributed-pass ledger (649 United PASS events, on the frozen threshold that holds 0.97-1.09x Sofascore), the lateral channel split of pass origins matches the ball-track split to a mean absolute 5.3 percentage points (worst cell 11.9 pp). Two independent views of the same geometry agree.
- **Player layer.** `fingerprint.player_profiles.oriented_positions` on the 9 PRTreID identity matches, filtered to United by the PRTreID roster (not the colour anchor, which leaks opponents), pooled per player per era, minimum 25 named frames.
- **Block layer.** `fingerprint.block_height` -- de-biased rearmost-outfielder line on trusted-geometry frames (`calib_error_m <= 1.0`), out-of-possession resolved by the same Viterbi smoother. Reproduces `results/BLOCK_AND_STYLE_v1.md` leg for leg.
- **Game state.** `outputs/oracle/wp/<id>.parquet`, the calibrated base-subset win-probability model (held-out ECE 0.0415, `results/GAME_STATE_v2.md`); available for 6 of 12 matches.

## A direction bug found and fixed here (read this before reusing the numbers)

Left/right and own-half/final-third are meaningless without the attacking direction, and both shipped resolvers turned out to be unreliable on this corpus:

- **`fingerprint.structural_metrics.resolve_attack_directions_from_ball` is systematically inverted.** It credits each ball sample to the nearest player within 4 m and calls the team with the further-forward mean the attacking one -- but in a crowded defensive third the nearest player is usually a *defender*, so the team defending a goal reads as the team attacking it. Graded against the Sofascore D<M<F position oracle it flips the sign of the rank correlation in 5 of 5 matches tested (`manutd_brighton` +0.926 -> -0.926). It is the **primary** resolver inside `fingerprint/block_height.py::_directions`, so the `ball_to_block_m` and `broadcast_bias_m` columns of `results/BLOCK_AND_STYLE_v1.md` are computed on an inverted frame and should be re-derived. `mean_line_m`, the vertical spread, coverage and the low/mid/high spell classes are **not** affected (the line comes from `generator.impute.line_estimates`, and a standard deviation is invariant under the flip), which is why the block numbers in this profile match the shipped ones exactly.
- **The keeper-based resolver is correct but noisy per chunk.** On `tottenham_manutd` h1 it resolves 3 of 6 chunks backwards against the goal-direction evidence in `results/PAIR_ANALYSIS_v1.md`.

**What this profile uses instead.** A chunk only votes on direction when the two goalkeepers separate cleanly (medians >= 40 m apart, the same test the pair-analysis mapping adjudication used); votes are summed per half weighted by keeper support; and the half-time end swap is enforced, because a real team cannot attack the same goal in both halves. Graded on the same external oracle across all 9 identity matches this beats the shipped per-chunk keeper resolve: **positive rank correlation in 8/9 matches versus 7/9, mean rho +0.610 versus +0.361**. It repairs both of the "advanced full-back inversion" matches flagged in `results/PLAYER_ANALYSIS_v2.md` (`brighton_manutd` -0.507 -> +0.676; `tottenham_manutd` +0.397 -> +0.850). The per-match column below shows how many chunks supplied a clean vote -- two matches rest on only 2 clean votes and their lateral read should be read accordingly.

## Per-match evidence

| match | mgr | venue | opponent | post-link ball cov | clean direction votes | United ball samples | block line (m) | block spread (m) | block cov |
|-------|-----|-------|----------|-------------------:|----------------------:|--------------------:|---------------:|-----------------:|----------:|
| manutd_fulham | ten_hag | H | Fulham | 0.400 | 10/10 | 1330 | 25.1 | 6.08 | 0.969 |
| brighton_manutd | ten_hag | A | Brighton | 0.511 | 2/11 | 1848 | 33.2 | 6.17 | 0.960 |
| manutd_liverpool | ten_hag | H | Liverpool | 0.445 | 8/11 | 1478 | 29.5 | 5.92 | 0.966 |
| southampton_manutd | ten_hag | A | Southampton | 0.417 | 2/11 | 1636 | 37.4 | 5.83 | 0.815 |
| palace_manutd | ten_hag | A | Crystal Palace | 0.366 | 5/12 | 1548 | 26.7 | 5.78 | 0.902 |
| manutd_tottenham | ten_hag | H | Tottenham | 0.419 | 4/11 | 739 | 26.0 | 6.16 | 0.967 |
| liverpool_manutd | amorim | A | Liverpool | 0.360 | 9/11 | 1013 | 20.8 | 5.71 | 0.962 |
| manutd_southampton | amorim | H | Southampton | 0.491 | 9/10 | 1511 | 25.2 | 6.94 | 0.956 |
| manutd_brighton | amorim | H | Brighton | 0.514 | 10/11 | 1961 | 29.6 | 5.88 | 0.954 |
| fulham_manutd | amorim | A | Fulham | 0.561 | 4/11 | 2456 | 27.0 | 6.56 | 0.933 |
| manutd_palace | amorim | H | Crystal Palace | 0.465 | 7/11 | 1865 | 30.1 | 6.57 | 0.939 |
| tottenham_manutd | amorim | A | Tottenham | 0.496 | 2/11 | 1098 | 28.5 | 6.49 | 0.977 |

## Statistical tests as run

- **Funnel (own third vs final third central share), United:** paired per-match difference positive in 11/12, median +0.250, two-sided exact sign test p = 0.0063.
- **Funnel, the same twelve broadcasts' opposition:** positive in 12/12, median +0.181, p = 0.0005.
- **Left-vs-right tilt in the final third, United:** left-heavy in 5/12, median -0.080, p = 0.7744 -- **no effect**.
- The match is the independent unit throughout. Pooled ball samples sit 0.2 s apart and are heavily autocorrelated, so no test is run on the sample count; the pooled shares are descriptive and the sign tests carry the inference.

## Known limitations of this profile specifically

- Ball coverage is 36-56% of sampled frames, and the broadcast follows the ball, so the observed ball map is biased toward the phases television chooses to show.
- The own-third central share is partly the goalkeeper, who is central by construction. The own-to-final funnel is a *within-match contrast*, so a static lateral bias in the homography cancels; a zone-dependent one would not.
- Nine of twelve matches carry named players; the ten Hag era contributes only three of those nine, so every era-split player number leans on the Amorim side.
- Block heights are un-gated (Gate 1 pending) and read relatively, never as FIFA metres.
- Everything here is Manchester United measured against Manchester United and against the six opponents inside these twelve broadcasts. It is not a league-wide comparison and cannot become one at n = 12.

Generated by `python -m tools.manutd_identity`; evidence cached at `outputs/manutd_identity/evidence.json`.
