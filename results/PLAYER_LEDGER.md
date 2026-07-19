# Player-action ledger (B-3 stage 2)

Team attribution: ball-carrier (nearest tracked player to the Viterbi ball) team at the aligned frame nearest each filtered PASS event; abstain if no ball within +-1 s or the nearest player is > 3 m away. Player attribution (identity matches only): nearest NAMED track to the ball within 3 m. Truth = Sofascore attempted passes (team) / totalPass (player).

## Team gate (all matches)

### brighton_manutd -- team gate

| team | side | h1 attr/truth | h2 attr/truth | match attr/truth | ratio |
|------|------|---------------|---------------|------------------|-------|
| Man Utd | away | 109/284 | 82/227 | 191/511 | 0.374 |
| Brighton | home | 126/259 | 97/218 | 223/477 | 0.468 |

Filtered PASS events: 981 | team-attributed: 414 | abstained (no on-ball carrier): 567 (57.8%)

### manutd_liverpool -- team gate

| team | side | h1 attr/truth | h2 attr/truth | match attr/truth | ratio |
|------|------|---------------|---------------|------------------|-------|
| Man Utd | home | 88/266 | 93/241 | 181/507 | 0.357 |
| Liverpool | away | 92/258 | 82/206 | 174/464 | 0.375 |

Filtered PASS events: 981 | team-attributed: 355 | abstained (no on-ball carrier): 626 (63.8%)

### manutd_fulham -- team gate

| team | side | h1 attr/truth | h2 attr/truth | match attr/truth | ratio |
|------|------|---------------|---------------|------------------|-------|
| Man Utd | home | 105/297 | 61/185 | 166/482 | 0.344 |
| Fulham | away | 77/188 | 67/196 | 144/384 | 0.375 |

Filtered PASS events: 945 | team-attributed: 310 | abstained (no on-ball carrier): 635 (67.2%)

## Player attribution (identity matches)

### brighton_manutd -- named-player passes vs oracle

Team-attributed passes: 414 | with a named player: 15 (coverage 3.6%) | Spearman(attr, totalPass) = 0.194 over 8 named players (N too small to be conclusive; counts are a floor)

| player | attr pass | oracle totalPass |
|--------|-----------|------------------|
| Kobbie Mainoo | 3 | 45 |
| Amad Diallo | 2 | 41 |
| Jan Paul van Hecke | 2 | 86 |
| Harry Maguire | 2 | 45 |
| Lisandro Martínez | 2 | 73 |
| Joël Veltman | 2 | 40 |
| Diogo Dalot | 1 | 59 |
| Kaoru Mitoma | 1 | 15 |

### manutd_liverpool -- named-player passes vs oracle

Team-attributed passes: 355 | with a named player: 20 (coverage 5.6%) | Spearman(attr, totalPass) = 0.009 over 9 named players (N too small to be conclusive; counts are a floor)

| player | attr pass | oracle totalPass |
|--------|-----------|------------------|
| Alexis Mac Allister | 5 | 49 |
| Marcus Rashford | 4 | 22 |
| Andy Robertson | 2 | 42 |
| Lisandro Martínez | 2 | 65 |
| Dominik Szoboszlai | 2 | 39 |
| Noussair Mazraoui | 2 | 54 |
| Bruno Fernandes | 1 | 43 |
| Kobbie Mainoo | 1 | 41 |
| Ryan Gravenberch | 1 | 44 |

## Honest limits

- Coverage ceiling is tracking, not the method: ~35-42% of filtered passes get an on-ball carrier within 3 m; the rest abstain because no tracked player is on the ball at the kick (broadcast detects players in a minority of frames -- the industry regime, cf. the plan doc). Where a carrier IS found the fit is tight (median ~1.3 m).
- The per-team split holds for manutd_liverpool (attr 181:174 vs truth 507:464) and manutd_fulham (166:144 vs 482:384), but INVERTS for brighton_manutd (attr over-weights Brighton 223 vs Man Utd 191, while truth has Man Utd ahead 511:477) -- a real ceiling of the nearest-carrier heuristic on this match, not smoothed over.
- Team attribution rests on the nearest-player-to-ball heuristic, not a possession model; a loose ball between two players attributes to whoever is closest. The gate ratio is the check that this holds in aggregate.
- Player coverage is bounded by named-fragment coverage (sparse): only passes whose carrier is a named track within the radius get a player, so counts are a floor, not a total.
- `manutd_fulham` has no identity artifacts -> team-level only.
- The event stream itself is BAS PASS/DRIVE; E2E-Spot goals/shots/cards live in `results/action_spotting_probe/` and are not merged here (needs a shared 2 fps<->25 fps half-clock; add when the ledger needs multi-class rows).
