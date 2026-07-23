# Player analysis v1 (Phase A) -- Manchester United named players

Per-named-player profiles from the PRTreID PRECISION identity artifacts (`outputs/identity/<id>_named_tracks_both2_prtreid.parquet`) + the validated event ledger. Man Utd is oriented to attack +x on the 105x68 pitch. Auto-extends to any match that gains a PRTreID parquet.

## Coverage statement (read first -- this bounds every claim below)

- **Identity matches: 3** (brighton_manutd, manutd_liverpool, manutd_tottenham -- all correct team-mapping, none of the two known flips).
- **What has real support: the positional layer, not events.** Where players play (mean advance, line band) and how visible they are rest on hundreds of tracked-named frames per player. Visibility is validated in all three matches -- Spearman(frames, minutes played) = 0.27-0.59 (positive every leg: the more a player is tracked-and-named, the more he actually played). Position ordering Spearman(mean advance x, D<M<F) = -0.51-0.77: positive in 2/3 matches but NEGATIVE in brighton_manutd, where the few trusted frames of an attacking full-back (Mazraoui, nominally D) landed him at 72 m and inverted the naive defender-deep ordering. So mean-x recovers role directionally but is fooled by advanced full-backs and sparse-frame players -- read it with the frame count.
- **What is DEAD: per-player event involvement.** Only 1.0-4.2% of Man Utd's attributed passes carry a named player (the known 4-12% floor, a tracking limit not an identity error): 2-5 named passes per match. The attributed-pass / touch / possession-link counts are a **floor an order of magnitude below the truth**, and their Spearman vs the oracle is computed over a near-all-zero vector -- it is reported but is NOT a meaningful ranking. This confirms the `PASS_NETWORKS_v1` negative result at the player level.
- **No plus-minus, no per-90 rate cards, no involvement-based impact ranking.** At this coverage on/off-ball impact cannot be estimated. The impact section below leads with what the positional layer CAN say (role, territory, line anchoring) and states plainly what it cannot.

## Per-match named-player tables

### brighton_manutd  (ten_hag, home=Brighton)

Man Utd named players: 8 | attributed Man Utd passes: 191 | named-attributed: 2 (coverage 1.0%) | Man Utd poss-links: 84.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = -0.507 over 6 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = 0.268 over 8 players.
- event involvement (DEAD -- 1 players with any attributed pass): Spearman(attr pass, totalPass) = 0.412, top-5 overlap 2/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1609 entities clustered): def 28.5 / mid 55.1 / att 82.7.
- defensive: Kobbie Mainoo (36m)
- midfield: Amad Diallo (49m), Marcus Rashford (49m), Bruno Fernandes (51m), Antony (54m)
- attacking: Noussair Mazraoui (72m)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Kobbie Mainoo | 36.0 | 26.2 | 15.5 | 3% | 78 | yes |
| Mason Mount | 47.0 | 52.0 | 0.0 | 0% | 1 | thin |
| Amad Diallo | 48.6 | 35.5 | 22.5 | 23% | 106 | yes |
| Marcus Rashford | 48.7 | 54.2 | 16.4 | 4% | 84 | yes |
| Bruno Fernandes | 51.2 | 23.8 | 22.7 | 30% | 63 | yes |
| Antony | 54.5 | 13.4 | 7.5 | 0% | 30 | yes |
| Noussair Mazraoui | 72.1 | 16.4 | 13.2 | 75% | 73 | yes |
| Diogo Dalot | 100.5 | 30.4 | 0.4 | 100% | 4 | thin |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Kobbie Mainoo | 2 | 45 | 2 | 65 | 1.2% |
| Amad Diallo | 0 | 41 | 0 | 53 | 0.0% |
| Bruno Fernandes | 0 | 40 | 1 | 49 | 0.0% |
| Antony | 0 | 2 | 0 | 3 | 0.0% |
| Diogo Dalot | 0 | 59 | 0 | 81 | 0.0% |
| Marcus Rashford | 0 | 18 | 0 | 23 | 0.0% |
| Mason Mount | 0 | 16 | 0 | 22 | 0.0% |
| Noussair Mazraoui | 0 | 34 | 0 | 48 | 0.0% |

### manutd_liverpool  (ten_hag, home=Man Utd)

Man Utd named players: 10 | attributed Man Utd passes: 181 | named-attributed: 3 (coverage 1.7%) | Man Utd poss-links: 75.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = 0.766 over 9 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = 0.425 over 10 players.
- event involvement (DEAD -- 3 players with any attributed pass): Spearman(attr pass, totalPass) = 0.648, top-5 overlap 2/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1077 entities clustered): def 27.2 / mid 53.0 / att 79.8.
- defensive: Lisandro Martínez (35m)
- midfield: Noussair Mazraoui (44m), Bruno Fernandes (45m), Diogo Dalot (46m), Casemiro (47m), Alejandro Garnacho (50m), Kobbie Mainoo (56m), Toby Collyer (56m)
- attacking: Marcus Rashford (71m)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Lisandro Martínez | 35.3 | 23.3 | 19.1 | 0% | 32 | yes |
| Noussair Mazraoui | 44.1 | 46.0 | 13.9 | 5% | 278 | yes |
| Bruno Fernandes | 45.0 | 33.6 | 14.0 | 4% | 146 | yes |
| Diogo Dalot | 46.4 | 27.4 | 17.1 | 15% | 93 | yes |
| Casemiro | 46.8 | 39.5 | 14.6 | 2% | 142 | yes |
| Alejandro Garnacho | 49.6 | 43.6 | 19.7 | 7% | 45 | yes |
| Kobbie Mainoo | 55.9 | 36.1 | 25.6 | 0% | 25 | yes |
| Toby Collyer | 56.4 | 34.2 | 9.5 | 7% | 28 | yes |
| Marcus Rashford | 71.1 | 25.2 | 15.2 | 51% | 63 | yes |
| Christian Eriksen | 81.4 | 30.0 | 14.3 | 90% | 20 | thin |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Lisandro Martínez | 1 | 65 | 1 | 77 | 1.3% |
| Kobbie Mainoo | 1 | 41 | 2 | 57 | 1.3% |
| Noussair Mazraoui | 1 | 54 | 1 | 85 | 1.3% |
| Alejandro Garnacho | 0 | 21 | 0 | 34 | 0.0% |
| Bruno Fernandes | 0 | 43 | 0 | 60 | 0.0% |
| Casemiro | 0 | 37 | 0 | 45 | 0.0% |
| Diogo Dalot | 0 | 43 | 0 | 70 | 0.0% |
| Christian Eriksen | 0 | 3 | 0 | 4 | 0.0% |
| Marcus Rashford | 0 | 22 | 0 | 33 | 0.0% |
| Toby Collyer | 0 | 19 | 0 | 33 | 0.0% |

### manutd_tottenham  (ten_hag, home=Man Utd)

Man Utd named players: 11 | attributed Man Utd passes: 120 | named-attributed: 5 (coverage 4.2%) | Man Utd poss-links: 42.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = 0.529 over 8 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = 0.586 over 11 players.
- event involvement (DEAD -- 3 players with any attributed pass): Spearman(attr pass, totalPass) = -0.39, top-5 overlap 0/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1036 entities clustered): def 25.9 / mid 47.5 / att 75.9.
- defensive: (no named anchor)
- midfield: Lisandro Martínez (42m), Noussair Mazraoui (42m), Manuel Ugarte (45m), Alejandro Garnacho (47m), Bruno Fernandes (53m), Diogo Dalot (54m), Christian Eriksen (60m)
- attacking: Mason Mount (62m)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Amad Diallo | 28.1 | 24.6 | 2.5 | 0% | 15 | thin |
| Lisandro Martínez | 42.0 | 21.1 | 18.7 | 12% | 58 | yes |
| Noussair Mazraoui | 42.4 | 42.0 | 23.6 | 18% | 62 | yes |
| Casemiro | 43.7 | 17.9 | 8.6 | 0% | 3 | thin |
| Manuel Ugarte | 45.0 | 32.8 | 18.4 | 6% | 109 | yes |
| Alejandro Garnacho | 47.3 | 35.3 | 28.0 | 17% | 533 | yes |
| Marcus Rashford | 51.8 | 10.7 | 6.6 | 9% | 22 | thin |
| Bruno Fernandes | 53.1 | 32.3 | 12.0 | 6% | 81 | yes |
| Diogo Dalot | 53.8 | 16.3 | 15.2 | 8% | 112 | yes |
| Christian Eriksen | 60.0 | 10.7 | 10.4 | 8% | 40 | yes |
| Mason Mount | 61.9 | 19.7 | 15.2 | 72% | 25 | yes |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Alejandro Garnacho | 3 | 29 | 8 | 50 | 4.8% |
| Mason Mount | 1 | 9 | 1 | 17 | 0.0% |
| Christian Eriksen | 1 | 16 | 1 | 27 | 2.4% |
| Bruno Fernandes | 0 | 23 | 0 | 29 | 0.0% |
| Amad Diallo | 0 | 12 | 0 | 16 | 0.0% |
| Diogo Dalot | 0 | 47 | 1 | 60 | 0.0% |
| Casemiro | 0 | 30 | 0 | 42 | 0.0% |
| Lisandro Martínez | 0 | 46 | 0 | 66 | 0.0% |
| Manuel Ugarte | 0 | 25 | 0 | 37 | 0.0% |
| Marcus Rashford | 0 | 20 | 2 | 32 | 0.0% |
| Noussair Mazraoui | 0 | 47 | 0 | 65 | 0.0% |

## Cross-match aggregation (Man Utd players named in 2+ matches)

`frames` and volume columns are totals; `x` / `final-3rd %` are means over the player's TRUSTED (>=25-frame) per-match positions only (`trust` = how many of the matches gave a trusted position). Event columns stay near zero -- see the coverage note; the positional columns are the validated content.

| player | matches | n | trust | frames | x (m) | final-3rd % | attr pass | touch proxy | poss-link % |
|--------|---------|--:|------:|-------:|------:|------------:|----------:|------------:|------------:|
| Noussair Mazraoui | brighton,manutd,manutd | 3 | 3 | 413 | 53 | 33% | 1 | 1 | 0.4% |
| Bruno Fernandes | brighton,manutd,manutd | 3 | 3 | 290 | 50 | 13% | 0 | 1 | 0.0% |
| Diogo Dalot | brighton,manutd,manutd | 3 | 2 | 209 | 50 | 12% | 0 | 1 | 0.0% |
| Marcus Rashford | brighton,manutd,manutd | 3 | 2 | 169 | 60 | 27% | 0 | 2 | 0.0% |
| Alejandro Garnacho | manutd,manutd | 2 | 2 | 578 | 48 | 12% | 3 | 8 | 2.4% |
| Casemiro | manutd,manutd | 2 | 1 | 145 | 47 | 2% | 0 | 0 | 0.0% |
| Amad Diallo | brighton,manutd | 2 | 1 | 121 | 49 | 23% | 0 | 0 | 0.0% |
| Kobbie Mainoo | brighton,manutd | 2 | 2 | 103 | 46 | 1% | 3 | 4 | 1.3% |
| Lisandro Martínez | manutd,manutd | 2 | 2 | 90 | 39 | 6% | 1 | 1 | 0.6% |
| Christian Eriksen | manutd,manutd | 2 | 1 | 60 | 60 | 8% | 1 | 1 | 1.2% |
| Mason Mount | brighton,manutd | 2 | 1 | 26 | 62 | 72% | 1 | 1 | 0.0% |

## Who is the impact player? (the honest answer)

**Straight answer: this corpus cannot name an impact player, and it would be dishonest to.** Impact needs involvement volume or on/off value, and the event layer is dead here -- 2-5 named passes per match (coverage note above). Ranking anyone 'biggest impact' off three attributed passes would be noise dressed as a finding. What the data DOES support, and what checks out against the oracle, is *where each player operates and how central their zone is to the shape* -- so that is what is reported.

What the validated positional layer supports (players in 2+ matches, trusted positions):

- **Deepest builders (rearmost mean advance):** Lisandro Martínez (39 m), Kobbie Mainoo (46 m), Casemiro (47 m) -- these anchor the defensive/first line of the build-up.
- **Highest / most territorial:** Mason Mount (62 m, 72% final-3rd), Christian Eriksen (60 m, 8% final-3rd), Marcus Rashford (60 m, 27% final-3rd) -- the players carrying Man Utd furthest up the pitch on trusted frames.
- **Most on-ball-visible (a biased proxy for centrality, NOT impact):** Alejandro Garnacho (578 frames), Noussair Mazraoui (413 frames), Bruno Fernandes (290 frames) -- read as 'the pipeline sees them on the ball most', which conflates true involvement with broadcast/tracking visibility; do not read it as most valuable.

So the closest defensible statement to Sid's question is territorial, not a rating: Man Utd's build-up is anchored deep by Lisandro Martínez and carried highest by Mason Mount. Naming a single 'biggest impact player' needs the plus-minus this coverage forbids -- flagged, not faked.

## What this CANNOT claim

- **Not a plus-minus or a rating.** No goals/assists-added, no on/off splits -- the attribution coverage (4-12%) forbids it. A player's low attributed count can be low involvement OR low broadcast visibility; the two are not separable here.
- **Line bands are geometric, not tactical roles.** The clustering splits Man Utd outfield entities by mean advance on trusted frames; a full-back bombing on reads as higher, a dropping striker as lower. It is a shape descriptor, not a formation call.
- **Possession-link % over-counts.** An edge links the next *attributed* carrier and skips unlabeled touches; and a named carrier can be the opponent's nearest player. Read it as relative involvement, not a pass-completion figure.
- **Frames != played minutes.** The `frames` column counts only frames where the player's fragment is named and on-screen (broadcast follows the ball), so it tracks minutes only in *rank* (validated: frames-vs-minutes Spearman above), not in scale.
- **n=3 matches, distinct opponents.** No opponent is measured twice, so nothing here separates a player-stable trait from a single-match matchup.
