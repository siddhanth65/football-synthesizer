# Player analysis v2 (Phase A) -- Manchester United, across the manager change

Extends v1 (`results/PLAYER_ANALYSIS_v1.md`, kept) from 3 to **9 named matches spanning the ten Hag -> Amorim change** (3 ten Hag, 6 Amorim). Same PRTreID PRECISION identity artifacts + validated event ledger; Man Utd oriented to attack +x on the 105x68 pitch. v2 adds the first PLAYER-level manager comparison, a formation proxy, and pooled-event test. The v1 oracle-join bug (6 reverse-fixture matches printed `rho None` because their Sofascore player-stats parquet was never mapped) is fixed -- **all 9 matches now validate**.

## Validation (all 9 matches -- read first)

Two independent oracle checks per match against Sofascore. **Visibility** (tracked-named frames vs minutes) is positive in all 9 (0.14-0.59): the more a player is tracked-and-named, the more he actually played -- the positional layer's load-bearing validator. **Position ordering** (mean advance x vs D<M<F) is positive in 7/9 and negative in 2/9.

| match | manager | pos-order rho | n | vis rho | n | flag |
|-------|---------|-------------:|--:|--------:|--:|------|
| brighton_manutd | ten_hag | -0.507 | 6 | 0.268 | 8 | advanced full-back inversion |
| manutd_liverpool | ten_hag | 0.766 | 9 | 0.425 | 10 |  |
| manutd_tottenham | ten_hag | 0.529 | 8 | 0.586 | 11 |  |
| liverpool_manutd | amorim | 0.804 | 10 | 0.365 | 10 |  |
| manutd_brighton | amorim | 0.926 | 6 | 0.136 | 8 |  |
| fulham_manutd | amorim | -0.956 | 6 | 0.505 | 7 | advanced full-back inversion |
| manutd_palace | amorim | 0.394 | 8 | 0.547 | 11 |  |
| manutd_southampton | amorim | 0.894 | 5 | 0.299 | 7 |  |
| tottenham_manutd | amorim | 0.397 | 7 | 0.412 | 8 |  |

The two negatives are the same advanced-full-back artifact flagged in v1, not a mapping error: in `brighton_manutd` Mazraoui (nominally D) is an overlapping full-back reading highest at 72 m; in `fulham_manutd` (rho -0.96, n=6) the inversion is near-total -- the only two D are the bombing full-backs Dalot (58 m) and Mazraoui (61 m) while the striker Zirkzee (F) drops deepest to 20 m. With only 5-6 trusted players carrying a position label and full-backs pushed to wing-back, a single dropping striker flips the rank. Mean-x recovers role directionally (7/9 positive) but is fooled by advanced full-backs and sparse frames -- read it with the frame count.

## Coverage statement (bounds every claim below)

- **What has real support: the positional layer, not events.** Where players play (mean advance, line band) and how visible they are rest on hundreds of tracked-named frames per player, validated in all 9 matches above.
- **What is DEAD: per-player event involvement.** 1.0-4.2% of Man Utd's attributed passes carry a named player (2-5 per match); the pooled-across-9 test below shows even summing does not rescue it. No plus-minus, no rate cards, no involvement-based impact ranking.

## Manager-split player table (the first PLAYER-level manager comparison)

Man Utd players named in 2+ matches, geometry split by era. `x` / `f3` (final-third share) are means over TRUSTED (>=25-frame) per-match positions in that era; `frames` are totals (all named frames, trusted or not). `dx` = Amorim mean advance minus ten Hag mean advance (positive = higher under Amorim), defined only for the 9 players trusted in both eras -- these are the only rows carrying a manager delta.

| player | th n | th frames | th x | th f3 | am n | am frames | am x | am f3 | dx (am-th) |
|--------|----:|---------:|----:|-----:|----:|---------:|----:|-----:|-------:|
| Alejandro Garnacho | 2 | 578 | 48 | 12% | 6 | 547 | 64 | 48% | +15.1 |
| Noussair Mazraoui | 3 | 413 | 53 | 33% | 6 | 554 | 42 | 17% | -11.0 |
| Bruno Fernandes | 3 | 290 | 50 | 13% | 6 | 2321 | 55 | 29% | +5.6 |
| Diogo Dalot | 3 | 209 | 50 | 12% | 5 | 1314 | 50 | 16% | +0.3 |
| Casemiro | 2 | 145 | 47 | 2% | 1 | 33 | 54 | 21% | +7.7 |
| Amad Diallo | 2 | 121 | 49 | 23% | 5 | 1315 | 57 | 26% | +8.3 |
| Manuel Ugarte | 1 | 109 | 45 | 6% | 4 | 430 | 44 | 9% | -1.4 |
| Kobbie Mainoo | 2 | 103 | 46 | 1% | 2 | 178 | 53 | 25% | +7.3 |
| Lisandro Martínez | 2 | 90 | 39 | 6% | 3 | 177 | 48 | 26% | +9.2 |
| Marcus Rashford | 3 | 169 | 60 | 27% | 0 | 0 | - | - | - |
| Christian Eriksen | 2 | 60 | 60 | 7% | 1 | 23 | - | - | - |
| Mason Mount | 2 | 26 | 62 | 72% | 0 | 0 | - | - | - |
| Harry Maguire | 0 | 0 | - | - | 4 | 166 | 41 | 14% | - |
| Rasmus Højlund | 0 | 0 | - | - | 3 | 195 | 58 | 23% | - |
| Leny Yoro | 0 | 0 | - | - | 3 | 432 | 54 | 19% | - |

**Read.** Of 9 both-era players, 7 sit higher up the pitch under Amorim and 2 deeper. Largest advance shift: Alejandro Garnacho (+15.1 m). This is a positional shift with n and frame support stated per row; at 2-5 trusted matches per player it is directional, not a stable per-90 trait.

## Line composition x manager (formation proxy)

Mean Man Utd line-band centroids per era (m from own goal, all named + unnamed outfield fragments clustered per match, then averaged over the era's matches):

| era | matches | def line | mid line | att line |
|-----|--------:|--------:|--------:|--------:|
| ten_hag | 3 | 27.2 | 51.9 | 79.5 |
| amorim | 6 | 23.6 | 50.5 | 78.9 |

**Formation proxy (honest test).** ten Hag's nominal 4-2-3-1 vs Amorim's 3-4-3 should, if the named-player x-structure can see it, push the wide defenders higher under Amorim (full-back -> wing-back) and lift the defensive line. The recurring full-backs' advance delta:
- Noussair Mazraoui: ten Hag 53 m -> Amorim 42 m (dx -11.0 m, frames 413/554).
- Diogo Dalot: ten Hag 50 m -> Amorim 50 m (dx +0.3 m, frames 209/1314).

Defensive-line centroid moves -3.6 m ten Hag -> Amorim and the recurring full-backs average -5.3 m. **The proxy does NOT support the wing-back read**: under Amorim the defensive line sits, if anything, deeper and the wide defenders do not push higher (Mazraoui markedly deeper, Dalot flat). The named-player x-structure cannot see a 3-4-3 wing-back signature here -- opponent mix (6 different Amorim opponents), scoreline, and the broadcast-sparse full-back frame counts dominate any formation effect. Reported as a negative result, not forced into the tactical narrative.

## Named-pass involvement pooled across 9 matches

Threshold stated first: a per-player attributed-pass claim needs >= 20 pooled attributed passes before Poisson noise (relative SE ~ 1/sqrt(N)) drops below ~22%. Pooled over all 9 matches, the busiest named players are:

| player | matches | pooled attr pass | pooled touch proxy |
|--------|--------:|----------------:|------------------:|
| Diogo Dalot | 8 | 6 | 15 |
| Bruno Fernandes | 9 | 5 | 18 |
| Kobbie Mainoo | 4 | 5 | 7 |
| Alejandro Garnacho | 8 | 4 | 11 |
| Noussair Mazraoui | 9 | 3 | 4 |

Peak pooled attributed passes = 6 (< 20). **Pooling does NOT rescue the event layer** -- even summed over 9 matches no player clears the threshold, so no defensible per-player pass-involvement claim exists. The involvement columns stay a floor, exactly as in v1; the validated content remains positional.

## Who is the impact player? (the honest answer, unchanged)

**Straight answer: this corpus cannot name an impact player, and it would be dishonest to.** Impact needs involvement volume or on/off value, and the event layer is dead here -- 2-5 named passes per match (coverage note above). Ranking anyone 'biggest impact' off three attributed passes would be noise dressed as a finding. What the data DOES support, and what checks out against the oracle, is *where each player operates and how central their zone is to the shape* -- so that is what is reported.

What the validated positional layer supports (players in 2+ matches, trusted positions):

- **Deepest builders (rearmost mean advance):** Harry Maguire (41 m), Lisandro Martínez (43 m), Manuel Ugarte (44 m) -- these anchor the defensive/first line of the build-up.
- **Highest / most territorial:** Mason Mount (62 m, 72% final-3rd), Christian Eriksen (60 m, 8% final-3rd), Marcus Rashford (60 m, 27% final-3rd) -- the players carrying Man Utd furthest up the pitch on trusted frames.
- **Most on-ball-visible (a biased proxy for centrality, NOT impact):** Bruno Fernandes (2611 frames), Diogo Dalot (1523 frames), Amad Diallo (1436 frames) -- read as 'the pipeline sees them on the ball most', which conflates true involvement with broadcast/tracking visibility; do not read it as most valuable.

So the closest defensible statement to Sid's question is territorial, not a rating: Man Utd's build-up is anchored deep by Harry Maguire and carried highest by Mason Mount. Naming a single 'biggest impact player' needs the plus-minus this coverage forbids -- flagged, not faked.

## What this CANNOT claim

- **Not a plus-minus or a rating.** Attribution coverage (1-4%) forbids goals/assists-added or on/off splits. A low attributed count can be low involvement OR low broadcast visibility.
- **The manager deltas are positional, not tactical verdicts.** `dx` says a player's tracked mean advance moved; it does not attribute that to the manager alone (opponent, scoreline, and the specific matches sampled all move it). No opponent is measured twice under both managers.
- **Formation is NOT visible in this data.** The named-player x-structure did not show the expected 3-4-3 wing-back signature (defensive line no higher, full-backs no higher under Amorim) -- a negative result. 7-11 named players per match, dominated by opponent mix and broadcast-sparse full-back frames, cannot distinguish 4-2-3-1 from 3-4-3 as a shape.
- **Pooling does not create event coverage.** Summing 9 matches leaves the busiest player below the 20-pass floor; the event layer stays dead at the player level.
- **Line bands are geometric, not tactical roles**; **possession-link % over-counts**; **frames != played minutes** (tracks minutes in rank only, validated above). See v1 for the full form of these caveats.
