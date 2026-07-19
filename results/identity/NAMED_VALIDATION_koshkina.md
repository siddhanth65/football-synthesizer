# Named-track validation -- Koshkina arm vs Sofascore oracle (brighton_manutd)

Source: `outputs/identity/brighton_manutd_named_tracks_koshkina.parquet` (177 named-track rows, 20
distinct players). Visible-minutes computation reused verbatim from
`tools/wire_anchors.visible_seconds_by_player` (union of per-fragment frame spans, merged within a
chunk via `generator.anchor_wire.merge_intervals`, summed across chunks, divided by chunk fps),
applied directly to the parquet's `(chunk, track_id, player_name, jersey_number)` rows in place of
the in-memory `ResolvedFragment` list the original function consumes. Oracle: `player_stats_df` for
`brighton_manutd` (Sofascore match 12436888), joined on `shirtNumber` (not `jerseyNumber`, per the
task's given key).

## 1-2. Full per-player table (all 20, no cherry-picking)

Sorted by our visible minutes, descending.

| player | team | # | fragments | anchors | our visible min | oracle min | oracle touches | abs error (min) |
|---|---|---|---|---|---|---|---|---|
| Joel Veltman | Brighton | 34 | 35 | 85 | 6.4 | 90.0 | 65.0 | 83.6 |
| Bruno Fernandes | Man Utd | 8 | 32 | 52 | 5.1 | 79.0 | 49.0 | 73.9 |
| Marcus Rashford | Man Utd | 10 | 25 | 52 | 4.2 | 65.0 | 23.0 | 60.8 |
| Diogo Dalot | Man Utd | 20 | 15 | 21 | 3.7 | 90.0 | 81.0 | 86.3 |
| Danny Welbeck | Brighton | 18 | 4 | 4 | 2.6 | 79.0 | 25.0 | 76.4 |
| Noussair Mazraoui | Man Utd | 3 | 6 | 9 | 2.5 | 90.0 | 48.0 | 87.5 |
| Amad Diallo | Man Utd | 16 | 10 | 13 | 2.5 | 89.0 | 53.0 | 86.5 |
| Kaoru Mitoma | Brighton | 22 | 8 | 10 | 1.5 | 89.0 | 24.0 | 87.5 |
| James Milner | Brighton | 6 | 5 | 5 | 1.1 | 73.0 | 42.0 | 71.9 |
| Yankuba Minteh | Brighton | 17 | 4 | 9 | 1.1 | 89.0 | 49.0 | 87.9 |
| Mason Mount | Man Utd | 7 | 5 | 5 | 0.9 | 45.0 | 22.0 | 44.1 |
| Lisandro Martinez | Man Utd | 6 | 6 | 9 | 0.8 | 90.0 | 80.0 | 89.2 |
| Harry Maguire | Man Utd | 5 | 4 | 4 | 0.7 | 79.0 | 54.0 | 78.3 |
| Kobbie Mainoo | Man Utd | 37 | 4 | 7 | 0.7 | 90.0 | 65.0 | 89.3 |
| Casemiro | Man Utd | 18 | 2 | 3 | 0.5 | 90.0 | 69.0 | 89.5 |
| Jan Paul van Hecke | Brighton | 29 | 3 | 8 | 0.4 | 90.0 | 100.0 | 89.6 |
| Julio Enciso | Brighton | 10 | 3 | 7 | 0.3 | 11.0 | 12.0 | 10.7 |
| Lewis Dunk | Brighton | 5 | 3 | 3 | 0.2 | 90.0 | 86.0 | 89.8 |
| Antony | Man Utd | 21 | 2 | 4 | 0.1 | 8.0 | 3.0 | 7.9 |
| Simon Adingra | Brighton | 24 | 1 | 2 | 0.0 | 8.0 | 8.0 | 8.0 |

All 20 players have an oracle row (`in_squad = True` for all, `oracle_min` present for all).
Absolute error is large across the board (~8-90 min) because the visible-minutes proxy is
sparse ReID-limited screen time, not played minutes -- as documented in `NAMED_TRACKS.md`, the
comparison is meant to be **ordinal**, not calibrated.

Touch/event proxy counts: **not computed** -- the parquet carries no per-frame ball-possession
association, same limitation `wire_anchors.py` states ("Touch-count proxy: skipped"). Oracle
`touches` are reported in the table above for reference only, not compared.

## 3. Spearman gates (pre-committed, all reported)

| gate | n at gate | n with oracle | Spearman(our_visible_min, oracle_min) |
|---|---|---|---|
| no gate | 20 | 20 | 0.207 |
| visible_min >= 5 | 2 | 2 | None (n<3) |
| visible_min >= 10 | 0 | 0 | None (n<3) |
| visible_min >= 20 | 0 | 0 | None (n<3) |
| fragments >= 3 | 17 | 17 | -0.033 |
| anchors >= 10 | 6 | 6 | 0.029 |

No gate improves on the no-gate 0.207; two gates (`fragments>=3`, `anchors>=10`) flip it to
~zero/negative. None of the minutes-based gates (>=5/10/20) retain enough players even to compute
Spearman (our visible-minutes proxy tops out at 6.4 min, so `>=10` and `>=20` are empty by
construction).

**3 worst rank disagreements** (our rank vs oracle rank, both descending, n=20):

| player | our visible min | oracle min | our rank | oracle rank | rank diff |
|---|---|---|---|---|---|
| Lewis Dunk | 0.2 | 90.0 | 18 | 4.5 | 13.5 |
| Marcus Rashford | 4.2 | 65.0 | 3 | 16 | 13.0 |
| Jan Paul van Hecke | 0.4 | 90.0 | 16 | 4.5 | 11.5 |

(Runners-up: Bruno Fernandes rank diff 11.0, Casemiro 10.5, Kobbie Mainoo 9.0 -- all full-90
players our proxy ranks near the bottom because they simply got few close-up hero shots.)

## 4. Sanity checks

- **Named player not in oracle squad:** none. All 20 `(team, shirtNumber)` keys resolve to an
  oracle row (`in_squad = True` for every row) -- the roster mask is doing its job as designed.
- **Goalkeeper handling:** neither starting keeper (Andre Onana #1/24 Man Utd 90 min, Jason Steele
  #23 Brighton 90 min) appears among the 20 named players, and no anchor read #1, #23, or #24 for
  the goalkeeper role. No GK was mis-named or falsely excluded from a match -- they simply never
  produced a survivor anchor (consistent with the goalkeeper kit standing out from the OSNet
  candidate pool, or just not getting close-up shots in this survivor set). Confirmed clean, no
  action needed.
- **Substitutes vs weak correlation:** 3 of the 20 named players are clear subs by oracle minutes
  (Julio Enciso 11 min, Antony 8 min, Simon Adingra 8 min); the other 17 all played >=45 min
  (15 of those played the full 90). Restricting to the 17 near-full-match starters, Spearman is
  **-0.288** -- *worse* than the all-20 figure of 0.207. **The substitute hypothesis does not
  explain the weak correlation.** The noise is dominated by uneven hero-shot concentration among
  players who all played comparable (mostly full 90) minutes: Veltman (85 anchors) and Bruno
  Fernandes (52 anchors) get broadcast attention Casemiro (3 anchors) and Lewis Dunk (3 anchors)
  do not, despite all four playing the full match. This matches the caveat already logged in
  `NAMED_TRACKS.md` ("names the players who get repeated close-up hero shots, not a uniform XI").

## Summary (5 lines, honest)

1. All 20 named players are legitimately in the oracle squad; no roster-mask leak, no GK
   mis-handling -- the naming pipeline is structurally clean at n=20.
2. Spearman(visible_min, oracle_min) = 0.207 at no gate (n=20), confirming the quick number; every
   pre-committed sub-gate either kills the sample size (<3) or drives correlation to ~0/negative
   (fragments>=3: -0.033; anchors>=10: 0.029) -- no gate rescues the signal.
3. The weak correlation is NOT a substitute-minutes artifact: restricted to the 17 near-full-match
   starters, Spearman is -0.288, worse than the full set.
4. Root cause is hero-shot concentration, not naming error: full-90 players differ 3-to-85x in
   anchor count (Casemiro 3 vs Veltman 85), so visible-minutes-by-close-up-count is not a proxy for
   played minutes once you're past "did this player get broadcast attention at all."
5. Net read: the Koshkina arm's precision claim (98.6%, N=2-agreement) is about *who the anchor
   says it is*, which this validation does not contest -- what it contests is treating close-up
   anchor volume as a minutes/usage proxy; it isn't, at n=20 or at n=6.
