# Player analysis v1 (Phase A) -- Manchester United named players

Per-named-player profiles from the PRTreID PRECISION identity artifacts (`outputs/identity/<id>_named_tracks_both2_prtreid.parquet`) + the validated event ledger. Man Utd is oriented to attack +x on the 105x68 pitch. Auto-extends to any match that gains a PRTreID parquet.

## Coverage statement (read first -- this bounds every claim below)

- **Identity matches: 9** (brighton_manutd, manutd_liverpool, manutd_tottenham -- all correct team-mapping, none of the two known flips).
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

### liverpool_manutd  (amorim, home=Liverpool)

Man Utd named players: 10 | attributed Man Utd passes: 113 | named-attributed: 4 (coverage 3.5%) | Man Utd poss-links: 31.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = None over 0 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = None over 0 players.
- event involvement (DEAD -- 3 players with any attributed pass): Spearman(attr pass, totalPass) = None, top-5 overlap None/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1141 entities clustered): def 21.7 / mid 46.5 / att 76.9.
- defensive: Harry Maguire (19m), Lisandro Martínez (34m)
- midfield: Noussair Mazraoui (41m), Manuel Ugarte (47m), Diogo Dalot (50m), Amad Diallo (51m), Kobbie Mainoo (53m), Rasmus Højlund (55m), Alejandro Garnacho (61m), Bruno Fernandes (61m)
- attacking: (no named anchor)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Harry Maguire | 19.5 | 24.9 | 15.8 | 2% | 41 | yes |
| Lisandro Martínez | 33.6 | 21.4 | 17.4 | 18% | 90 | yes |
| Noussair Mazraoui | 41.1 | 54.5 | 17.8 | 4% | 122 | yes |
| Manuel Ugarte | 47.5 | 37.3 | 16.1 | 0% | 98 | yes |
| Diogo Dalot | 50.3 | 4.3 | 7.8 | 0% | 89 | yes |
| Amad Diallo | 51.2 | 36.8 | 22.6 | 20% | 533 | yes |
| Kobbie Mainoo | 53.2 | 26.8 | 21.1 | 25% | 158 | yes |
| Rasmus Højlund | 54.5 | 29.1 | 3.8 | 2% | 49 | yes |
| Alejandro Garnacho | 61.1 | 29.7 | 26.6 | 45% | 31 | yes |
| Bruno Fernandes | 61.2 | 20.6 | 28.4 | 35% | 228 | yes |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Kobbie Mainoo | 2 | - | 3 | - | 6.5% |
| Harry Maguire | 1 | - | 2 | - | 6.5% |
| Noussair Mazraoui | 1 | - | 1 | - | 3.2% |
| Alejandro Garnacho | 0 | - | 0 | - | 0.0% |
| Amad Diallo | 0 | - | 1 | - | 0.0% |
| Diogo Dalot | 0 | - | 1 | - | 0.0% |
| Bruno Fernandes | 0 | - | 3 | - | 0.0% |
| Lisandro Martínez | 0 | - | 0 | - | 0.0% |
| Manuel Ugarte | 0 | - | 1 | - | 0.0% |
| Rasmus Højlund | 0 | - | 0 | - | 0.0% |

### manutd_brighton  (amorim, home=Man Utd)

Man Utd named players: 8 | attributed Man Utd passes: 187 | named-attributed: 3 (coverage 1.6%) | Man Utd poss-links: 75.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = None over 0 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = None over 0 players.
- event involvement (DEAD -- 2 players with any attributed pass): Spearman(attr pass, totalPass) = None, top-5 overlap None/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1949 entities clustered): def 23.2 / mid 50.2 / att 77.4.
- defensive: (no named anchor)
- midfield: Noussair Mazraoui (44m), Leny Yoro (47m), Manuel Ugarte (50m), Bruno Fernandes (50m), Amad Diallo (56m)
- attacking: Alejandro Garnacho (65m)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Noussair Mazraoui | 43.9 | 56.3 | 15.2 | 12% | 108 | yes |
| Leny Yoro | 47.5 | 20.1 | 9.1 | 0% | 178 | yes |
| Manuel Ugarte | 49.7 | 35.3 | 22.0 | 19% | 153 | yes |
| Bruno Fernandes | 50.5 | 25.3 | 20.6 | 20% | 546 | yes |
| Amad Diallo | 55.9 | 49.7 | 16.9 | 18% | 500 | yes |
| Harry Maguire | 57.8 | 42.8 | 1.6 | 0% | 9 | thin |
| Alejandro Garnacho | 64.9 | 12.6 | 18.2 | 61% | 123 | yes |
| Diogo Dalot | 75.2 | 19.7 | 21.4 | 75% | 16 | thin |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Bruno Fernandes | 2 | - | 2 | - | 2.7% |
| Harry Maguire | 1 | - | 2 | - | 1.3% |
| Amad Diallo | 0 | - | 1 | - | 0.0% |
| Alejandro Garnacho | 0 | - | 1 | - | 0.0% |
| Diogo Dalot | 0 | - | 0 | - | 0.0% |
| Leny Yoro | 0 | - | 0 | - | 0.0% |
| Manuel Ugarte | 0 | - | 1 | - | 0.0% |
| Noussair Mazraoui | 0 | - | 0 | - | 0.0% |

### fulham_manutd  (amorim, home=Fulham)

Man Utd named players: 7 | attributed Man Utd passes: 233 | named-attributed: 5 (coverage 2.1%) | Man Utd poss-links: 89.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = None over 0 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = None over 0 players.
- event involvement (DEAD -- 4 players with any attributed pass): Spearman(attr pass, totalPass) = None, top-5 overlap None/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1662 entities clustered): def 22.6 / mid 51.4 / att 80.5.
- defensive: Joshua Zirkzee (20m)
- midfield: Rasmus Højlund (53m), Amad Diallo (53m), Bruno Fernandes (54m), Diogo Dalot (58m), Noussair Mazraoui (61m)
- attacking: (no named anchor)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Joshua Zirkzee | 19.6 | 41.0 | 3.3 | 0% | 83 | yes |
| Rasmus Højlund | 52.6 | 9.8 | 5.6 | 0% | 100 | yes |
| Amad Diallo | 52.8 | 28.9 | 15.1 | 10% | 95 | yes |
| Bruno Fernandes | 54.0 | 42.7 | 23.0 | 32% | 497 | yes |
| Diogo Dalot | 57.5 | 38.9 | 18.4 | 28% | 385 | yes |
| Noussair Mazraoui | 61.3 | 21.6 | 17.3 | 46% | 77 | yes |
| Alejandro Garnacho | 82.3 | 14.8 | 4.3 | 100% | 17 | thin |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Bruno Fernandes | 2 | - | 5 | - | 1.1% |
| Alejandro Garnacho | 1 | - | 1 | - | 0.0% |
| Diogo Dalot | 1 | - | 2 | - | 2.2% |
| Noussair Mazraoui | 1 | - | 1 | - | 0.0% |
| Amad Diallo | 0 | - | 0 | - | 0.0% |
| Joshua Zirkzee | 0 | - | 0 | - | 0.0% |
| Rasmus Højlund | 0 | - | 0 | - | 0.0% |

### manutd_palace  (amorim, home=Man Utd)

Man Utd named players: 11 | attributed Man Utd passes: 207 | named-attributed: 4 (coverage 1.9%) | Man Utd poss-links: 74.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = None over 0 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = None over 0 players.
- event involvement (DEAD -- 2 players with any attributed pass): Spearman(attr pass, totalPass) = None, top-5 overlap None/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1359 entities clustered): def 24.1 / mid 51.0 / att 79.9.
- defensive: Noussair Mazraoui (34m), Manuel Ugarte (34m)
- midfield: Diogo Dalot (39m), Harry Maguire (43m), Leny Yoro (60m), Lisandro Martínez (62m), Bruno Fernandes (63m)
- attacking: Amad Diallo (68m)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Noussair Mazraoui | 34.1 | 56.7 | 11.8 | 0% | 51 | yes |
| Manuel Ugarte | 34.2 | 40.7 | 15.7 | 0% | 70 | yes |
| Diogo Dalot | 38.9 | 14.6 | 23.1 | 10% | 241 | yes |
| Alejandro Garnacho | 43.1 | 30.5 | 28.5 | 27% | 22 | thin |
| Harry Maguire | 43.4 | 28.0 | 10.3 | 9% | 35 | yes |
| Kobbie Mainoo | 46.6 | 12.4 | 22.2 | 15% | 20 | thin |
| Christian Eriksen | 48.7 | 25.2 | 29.5 | 44% | 23 | thin |
| Leny Yoro | 60.5 | 20.3 | 17.3 | 38% | 248 | yes |
| Lisandro Martínez | 61.9 | 20.5 | 23.4 | 35% | 66 | yes |
| Bruno Fernandes | 62.6 | 28.7 | 23.3 | 47% | 214 | yes |
| Amad Diallo | 67.6 | 38.9 | 17.2 | 48% | 134 | yes |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Diogo Dalot | 3 | - | 8 | - | 1.4% |
| Amad Diallo | 1 | - | 1 | - | 0.0% |
| Alejandro Garnacho | 0 | - | 0 | - | 0.0% |
| Bruno Fernandes | 0 | - | 2 | - | 0.0% |
| Christian Eriksen | 0 | - | 0 | - | 0.0% |
| Harry Maguire | 0 | - | 0 | - | 0.0% |
| Kobbie Mainoo | 0 | - | 0 | - | 0.0% |
| Leny Yoro | 0 | - | 0 | - | 0.0% |
| Lisandro Martínez | 0 | - | 0 | - | 0.0% |
| Manuel Ugarte | 0 | - | 0 | - | 0.0% |
| Noussair Mazraoui | 0 | - | 0 | - | 0.0% |

### manutd_southampton  (amorim, home=Man Utd)

Man Utd named players: 7 | attributed Man Utd passes: 195 | named-attributed: 2 (coverage 1.0%) | Man Utd poss-links: 65.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = None over 0 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = None over 0 players.
- event involvement (DEAD -- 2 players with any attributed pass): Spearman(attr pass, totalPass) = None, top-5 overlap None/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1200 entities clustered): def 24.6 / mid 49.8 / att 78.5.
- defensive: Noussair Mazraoui (23m)
- midfield: Manuel Ugarte (43m), Bruno Fernandes (53m), Amad Diallo (57m)
- attacking: Alejandro Garnacho (71m)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Noussair Mazraoui | 22.5 | 30.4 | 19.2 | 16% | 25 | yes |
| Leny Yoro | 34.1 | 37.5 | 3.4 | 0% | 6 | thin |
| Manuel Ugarte | 43.2 | 39.7 | 19.5 | 16% | 109 | yes |
| Bruno Fernandes | 52.7 | 21.8 | 15.2 | 18% | 276 | yes |
| Lisandro Martínez | 55.6 | 27.2 | 12.3 | 10% | 21 | thin |
| Amad Diallo | 57.0 | 51.8 | 21.0 | 34% | 53 | yes |
| Alejandro Garnacho | 70.5 | 23.1 | 27.0 | 64% | 184 | yes |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Bruno Fernandes | 1 | - | 1 | - | 1.5% |
| Manuel Ugarte | 1 | - | 2 | - | 1.5% |
| Amad Diallo | 0 | - | 0 | - | 0.0% |
| Alejandro Garnacho | 0 | - | 1 | - | 0.0% |
| Christian Eriksen | 0 | - | 0 | - | 0.0% |
| Leny Yoro | 0 | - | 0 | - | 0.0% |
| Lisandro Martínez | 0 | - | 0 | - | 0.0% |
| Noussair Mazraoui | 0 | - | 0 | - | 0.0% |

### tottenham_manutd  (amorim, home=Tottenham)

Man Utd named players: 8 | attributed Man Utd passes: 156 | named-attributed: 2 (coverage 1.3%) | Man Utd poss-links: 61.

Validation vs Sofascore (Man Utd):
- position ordering: Spearman(mean advance x, position D<M<F) = None over 0 trusted players.
- visibility: Spearman(tracked-named frames, minutes played) = None over 0 players.
- event involvement (DEAD -- 1 players with any attributed pass): Spearman(attr pass, totalPass) = None, top-5 overlap None/5 -- computed over a near-all-zero vector, NOT meaningful.

**Lines** (centre m from own goal, 1692 entities clustered): def 25.2 / mid 54.4 / att 80.2.
- defensive: (no named anchor)
- midfield: Noussair Mazraoui (48m), Bruno Fernandes (52m), Casemiro (55m), Diogo Dalot (55m), Alejandro Garnacho (57m), Harry Maguire (60m), Rasmus Højlund (66m)
- attacking: (no named anchor)

**Visible geometry** (Man Utd, oriented to attack +x):

| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |
|--------|------:|------:|-------------:|------------:|-------:|:-------:|
| Patrick Dorgu | 25.6 | 33.8 | 15.5 | 12% | 8 | thin |
| Noussair Mazraoui | 48.3 | 28.9 | 22.2 | 22% | 171 | yes |
| Bruno Fernandes | 51.8 | 32.2 | 26.7 | 22% | 560 | yes |
| Casemiro | 54.5 | 35.2 | 13.9 | 21% | 33 | yes |
| Diogo Dalot | 54.7 | 30.3 | 28.9 | 24% | 583 | yes |
| Alejandro Garnacho | 57.5 | 26.8 | 18.0 | 22% | 170 | yes |
| Harry Maguire | 59.5 | 39.2 | 20.7 | 32% | 81 | yes |
| Rasmus Højlund | 66.4 | 47.9 | 19.3 | 67% | 46 | yes |

**Involvement + oracle validation:**

| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |
|--------|----------:|------------:|------------:|-------------:|------------:|
| Diogo Dalot | 2 | - | 3 | - | 1.6% |
| Alejandro Garnacho | 0 | - | 0 | - | 0.0% |
| Bruno Fernandes | 0 | - | 4 | - | 0.0% |
| Casemiro | 0 | - | 0 | - | 0.0% |
| Harry Maguire | 0 | - | 0 | - | 0.0% |
| Noussair Mazraoui | 0 | - | 1 | - | 0.0% |
| Patrick Dorgu | 0 | - | 0 | - | 0.0% |
| Rasmus Højlund | 0 | - | 1 | - | 0.0% |

## Cross-match aggregation (Man Utd players named in 2+ matches)

`frames` and volume columns are totals; `x` / `final-3rd %` are means over the player's TRUSTED (>=25-frame) per-match positions only (`trust` = how many of the matches gave a trusted position). Event columns stay near zero -- see the coverage note; the positional columns are the validated content.

| player | matches | n | trust | frames | x (m) | final-3rd % | attr pass | touch proxy | poss-link % |
|--------|---------|--:|------:|-------:|------:|------------:|----------:|------------:|------------:|
| Bruno Fernandes | brighton,manutd,manutd,liverpool,manutd,fulham,manutd,manutd,tottenham | 9 | 9 | 2611 | 54 | 24% | 5 | 18 | 0.6% |
| Noussair Mazraoui | brighton,manutd,manutd,liverpool,manutd,fulham,manutd,manutd,tottenham | 9 | 9 | 967 | 46 | 22% | 3 | 4 | 0.5% |
| Diogo Dalot | brighton,manutd,manutd,liverpool,manutd,fulham,manutd,tottenham | 8 | 6 | 1523 | 50 | 14% | 6 | 15 | 0.6% |
| Alejandro Garnacho | manutd,manutd,liverpool,manutd,fulham,manutd,manutd,tottenham | 8 | 6 | 1125 | 58 | 36% | 4 | 11 | 0.6% |
| Amad Diallo | brighton,manutd,liverpool,manutd,fulham,manutd,manutd | 7 | 6 | 1436 | 56 | 25% | 1 | 3 | 0.0% |
| Manuel Ugarte | manutd,liverpool,manutd,manutd,manutd | 5 | 5 | 539 | 44 | 8% | 1 | 4 | 0.3% |
| Lisandro Martínez | manutd,manutd,liverpool,manutd,manutd | 5 | 4 | 267 | 43 | 16% | 1 | 1 | 0.3% |
| Kobbie Mainoo | brighton,manutd,liverpool,manutd | 4 | 3 | 281 | 48 | 9% | 5 | 7 | 2.2% |
| Harry Maguire | liverpool,manutd,manutd,tottenham | 4 | 3 | 166 | 41 | 14% | 2 | 4 | 1.9% |
| Christian Eriksen | manutd,manutd,manutd,manutd | 4 | 1 | 83 | 60 | 8% | 1 | 1 | 0.6% |
| Leny Yoro | manutd,manutd,manutd | 3 | 2 | 432 | 54 | 19% | 0 | 0 | 0.0% |
| Rasmus Højlund | liverpool,fulham,tottenham | 3 | 3 | 195 | 58 | 23% | 0 | 1 | 0.0% |
| Casemiro | manutd,manutd,tottenham | 3 | 2 | 178 | 51 | 12% | 0 | 0 | 0.0% |
| Marcus Rashford | brighton,manutd,manutd | 3 | 2 | 169 | 60 | 27% | 0 | 2 | 0.0% |
| Mason Mount | brighton,manutd | 2 | 1 | 26 | 62 | 72% | 1 | 1 | 0.0% |

## Who is the impact player? (the honest answer)

**Straight answer: this corpus cannot name an impact player, and it would be dishonest to.** Impact needs involvement volume or on/off value, and the event layer is dead here -- 2-5 named passes per match (coverage note above). Ranking anyone 'biggest impact' off three attributed passes would be noise dressed as a finding. What the data DOES support, and what checks out against the oracle, is *where each player operates and how central their zone is to the shape* -- so that is what is reported.

What the validated positional layer supports (players in 2+ matches, trusted positions):

- **Deepest builders (rearmost mean advance):** Harry Maguire (41 m), Lisandro Martínez (43 m), Manuel Ugarte (44 m) -- these anchor the defensive/first line of the build-up.
- **Highest / most territorial:** Mason Mount (62 m, 72% final-3rd), Christian Eriksen (60 m, 8% final-3rd), Marcus Rashford (60 m, 27% final-3rd) -- the players carrying Man Utd furthest up the pitch on trusted frames.
- **Most on-ball-visible (a biased proxy for centrality, NOT impact):** Bruno Fernandes (2611 frames), Diogo Dalot (1523 frames), Amad Diallo (1436 frames) -- read as 'the pipeline sees them on the ball most', which conflates true involvement with broadcast/tracking visibility; do not read it as most valuable.

So the closest defensible statement to Sid's question is territorial, not a rating: Man Utd's build-up is anchored deep by Harry Maguire and carried highest by Mason Mount. Naming a single 'biggest impact player' needs the plus-minus this coverage forbids -- flagged, not faked.

## What this CANNOT claim

- **Not a plus-minus or a rating.** No goals/assists-added, no on/off splits -- the attribution coverage (4-12%) forbids it. A player's low attributed count can be low involvement OR low broadcast visibility; the two are not separable here.
- **Line bands are geometric, not tactical roles.** The clustering splits Man Utd outfield entities by mean advance on trusted frames; a full-back bombing on reads as higher, a dropping striker as lower. It is a shape descriptor, not a formation call.
- **Possession-link % over-counts.** An edge links the next *attributed* carrier and skips unlabeled touches; and a named carrier can be the opponent's nearest player. Read it as relative involvement, not a pass-completion figure.
- **Frames != played minutes.** The `frames` column counts only frames where the player's fragment is named and on-screen (broadcast follows the ball), so it tracks minutes only in *rank* (validated: frames-vs-minutes Spearman above), not in scale.
- **n=3 matches, distinct opponents.** No opponent is measured twice, so nothing here separates a player-stable trait from a single-match matchup.
