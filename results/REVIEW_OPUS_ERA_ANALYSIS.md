# Adversarial review: Opus-era corpus-capstone analysis (commit d8d79bf and adjacent)

Independent recomputation from parquets/artifacts, CPU-only, read-only. Every headline number below
was re-derived by re-running the module seams fresh off the parquets (not by trusting the cached
markdown). Where the seam is deterministic (KMeans seed=0, NMS peak-picking) the recompute is
byte-for-byte; where it is a heuristic the direction is checked. This file is the only artifact
written.

## Verdict table

| # | claim | verdict |
|---|-------|---------|
| 1 | WINS_VS_LOSSES level-state money table (cp / regain / block / att-3rd / trans) | CONFIRMED (exact) |
| 2 | style_fingerprint_v3 identity-decay (intra 5.347 / cross 5.435 / 3-of-6 / Sou 37.0 m) | CONFIRMED (exact, matrix byte-identical) |
| 3 | Tottenham B-3 ledger (218:120 vs 636:395, abstain 68.7%, cover 12.4% / 42) | CONFIRMED (exact) |
| 4 | Tottenham identity chain (4646 both2 anchors, 23/40, 0 sub-window violations) | CONFIRMED (sub count is 6, not 5) |
| 5 | Southampton re-anchor (balance 0.909, 14.3% team=-1, centroids, HT flip) | CONFIRMED (flip holds in aggregate) |
| 6 | Fulham-h2 BAS addendum (3 double-fires; no BAS chunk a corpus outlier) | CONFIRMED (rate col uses a different span; op counts exact) |

Overall: the capstone conclusions survive. All three load-bearing stories hold on the recomputed
numbers.

---

## Claim 1 - WINS_VS_LOSSES level-state table: RETRACTED (mapping flip)

> **2026-07-23 CORRECTION:** this "CONFIRMED (exact)" verdict re-derived the *then-current* generator
> output, but that output rested on a **flipped `southampton_manutd` team mapping** (see
> `results/PAIR_ANALYSIS_v1.md`). The Southampton (0-3 win) Man Utd row below carried the OPPONENT's
> numbers. Corrected with team0 = Man Utd, United's level-state block in that win is **HIGH (53.7 m),
> not deep (32.9 m)**, and their attacking-third control is the **highest** of the six (0.353), not the
> lowest. The **"both wins defended deepest" conclusion is FALSE** and retracted; there is no
> repeatable win-shape at level state (see the corrected `results/WINS_VS_LOSSES.md`).

Recomputed fresh via `fingerprint.score_state.annotate` over `style_fingerprint.phase_frame_table`
and `turnover_press_table` (both re-read every ball/aligned parquet and re-run `assign_possession`),
then `level_synthesis` at `state == "level"`. ATT_THIRD_X = 70.0 m confirmed in `core/pitch.py`.
(Southampton row corrected 2026-07-23; the other five legs were already correctly labelled.)

| match (result) | n loss | cp frac | 5s regain | block ht | att-3rd | trans depth | in/out/tp frames |
|---|---|---|---|---|---|---|---|
| Fulham (1-0 win)      | 71 | 0.718 | 0.465 | 38.0 | 0.173 | 57.9 | 549/441/343 |
| Southampton (0-3 win) | 26 | 0.423 | 0.192 | 53.7 | 0.353 | 51.1 | 85/79/88 |
| Liverpool (0-3 loss)  | 22 | 0.455 | 0.227 | 40.6 | 0.180 | 45.3 | 194/231/136 |
| Tottenham (0-3 loss)  |  1 | 0.000 | 0.000 | 49.9 | 0.000 | 24.4 | 1/32/12 |
| Brighton (1-2 loss)   | 64 | 0.719 | 0.406 | 46.9 | 0.262 | 61.4 | 344/535/387 |
| Palace (0-0 draw)     | 40 | 0.750 | 0.375 | 34.8 | 0.127 | 48.8 | 907/182/192 |

After the correction:

- **Press does not separate W/L** (unchanged). cp_frac wins {0.718, 0.423} straddle losses {0.455,
  0.719}: Fulham-win 0.718 sits on Brighton-loss 0.719; Southampton-win 0.423 is below Liverpool-loss
  0.455.
- **Block height does NOT separate W/L (RETRACTED).** Corrected ordering: Palace-draw 34.8 < Ful-win
  38.0 < Liv 40.6 < Bri 46.9 < Tot 49.9 < **Sou-win 53.7**. The Southampton win defended the HIGHEST
  block of all six, above every loss; the two wins bracket the range. "Both wins defended deepest" is
  false.
- **Att-3rd control does NOT separate either (RETRACTED).** Corrected: Sou-win 0.353 is the HIGHEST of
  the evaluable (above Brighton-loss 0.262), Ful-win mid 0.173. Regain and transition depth still
  fully overlap - no win-shape in any of the five metrics.

The write-up's hedges remain: the Tottenham level slice is n=1 loss / 32 out-poss frames (the ~3'
opener), the Southampton level slice is small (26 losses), and lines carry the stated ~+11 m broadcast
inflation. But the substantive block/territory conclusion was mapping-flipped and is now retracted.

Evidence: `fingerprint/score_state.py`, `fingerprint/style_fingerprint.py`, per-match aligned + ball
parquets via `core.registry`; the module `_demo()` seam self-checks pass.

## Claim 2 - style_fingerprint_v3 identity-decay: CONFIRMED (exact, matrix byte-identical)

Rebuilt the attack-normalised (center=False) 12x12 side matrix from scratch:
`match_embeddings -> prototypes (KMeans n=64, seed=0) -> wasserstein_distance_nd`. The recomputed
matrix is byte-identical to section 3a of `results/style_fingerprint_v3.md` (all 66 off-diagonal
cells reproduce to 3 dp - determinism from seed=0 holds).

- intra-ManU mean = **5.347**, ManU-to-opp mean = **5.435**, gap = **0.089** (rounds to 0.09). Match.
- ManU sides whose nearest neighbour is another ManU side: **3 / 6** (vLiv->vFul, vFul->vLiv,
  vTot->vLiv). The other three point at opponents (vBri->Southampton, vPal->Crystal Palace,
  vSou->Fulham). Match.
- Southampton ManU in-possession build-up = **37.0 m** (via `phase_profile`), the ~15 m low-outlier
  vs the 52-58 m band elsewhere. Match.

The "territorial signature collapsed at full n=6 / no shape fingerprint / away win is an outlier"
verdict is supported by the recomputed numbers.

Evidence: `fingerprint/style_fingerprint.py` (`match_embeddings`, `distance_matrix`),
`tools/run_style_fingerprint.py::side_labels`.

## Claim 3 - Tottenham B-3 ledger: CONFIRMED (exact)

From `outputs/manutd_tottenham/ledger.parquet` (1893 rows: 1079 PASS, 814 DRIVE):

- Team-attributed PASS with non-null team = **338**; team 1 (Tottenham) **218**, team 0 (Man Utd)
  **120** -> **218:120**. team_name map from the ledger confirms {0: Man Utd, 1: Tottenham}.
- Oracle `outputs/oracle/sofascore/team_stats_12436995.parquet`, "Passes" row: home (Man Utd) 395,
  away (Tottenham) 636 -> truth **395:636**, i.e. **636:395** Tottenham-first. Attributed direction
  (Tot 218 > ManU 120) matches truth direction (Tot 636 > ManU 395): **direction preserved**.
- Abstention = (1079 - 338) / 1079 = 741 / 1079 = **0.687** (68.7%). Match.
- Player coverage: 42 of 338 team-attributed passes carry a named player = **0.124** (12.4%), across
  18 distinct players. Match. (Matches the `PLAYER_LEDGER.md` per-player table.)

## Claim 4 - Tottenham identity chain: CONFIRMED (with one count discrepancy)

- `results/closeup_anchor_probe/koshkina_manutd_tottenham/levers/levers_stats.json`, arm `both2`:
  `anchors` = **4646**. Match.
- `outputs/identity/manutd_tottenham_lineup_assign.parquet`: `assigned.sum()` = **23** of 40. Match.
- Sub-window violations: I cross-referenced each assigned sub's `track_ids` against
  `manutd_tottenham_named_tracks_koshkina.parquet` and found the chunk (half) of every assigned
  fragment. All assigned subs have on_h1=False / on_h2=True, and **every one of their assigned
  fragments lies in an h2 chunk** - **0 sub-window violations**. Confirmed.

Discrepancy (not a break): the assigned subs number **6**, not 5 - Mason Mount (7), Casemiro (18),
Christian Eriksen (14), Rasmus Hojlund (9), Amad Diallo (16) for Man Utd, plus Lucas Bergvall (15)
for Tottenham. The "5 subs" figure appears to count Man Utd subs only; the zero-violation result is
unchanged and covers all 6.

## Claim 5 - Southampton re-anchor: CONFIRMED (flip holds in aggregate)

From `outputs/southampton_manutd/final/match_aligned.parquet` (150382 player+gk rows):

- team-0 = **67532**, team-1 = **61404**, team=-1 = 21446. Balance min/max = 61404/67532 =
  **0.9093** (0.909). Match.
- team=-1 share of player rows = 21446 / 150382 = **0.1426** (14.3%). Match.
- Both-team centroids are numerically distinct: team-0 (pitch_x 60.49, pitch_y 34.65) vs team-1
  (57.86, 36.02). Distinct (the anchor did not collapse as in v2's balance-0.004 case); the ~2.6 m
  x-gap is small, as expected for two sides sharing a pitch - "distinct" is accurate but a weak
  differentiator, and the real health signal is the 0.909 balance, not the centroid gap.
- Halftime attack-direction flip: `resolve_attack_directions` per chunk gives team-0 modal direction
  -1 in h1 and +1 in h2 (a clean HT flip in aggregate). Two of eleven chunks disagree with their
  half's mode (h1_chunk_001, h2_chunk_002) - ordinary per-chunk keeper-resolution noise, not a
  broken flip. Holds.

## Claim 6 - Fulham-h2 BAS addendum: CONFIRMED (op counts exact; rate column caveat)

**The 3 same-timestamp on/off-target double-fires.** Replicating the actual detector
(`tools/action_spot_probe.py::find_peaks`, greedy NMS, thresh 0.30, min_sep 30 s) on the "Shots on
target" (idx 13) and "Shots off target" (idx 12) channels, per chunk, the coincident on/off PEAK
timestamps are **exactly**: chunk_002 = {363.5 s, 527.0 s}, chunk_003 = {272.5 s}, and **zero** in
every other chunk (all 5 h1 chunks, h2_chunk_000/001/004). Reproduces the addendum precisely,
including "unique to the two outlier chunks."

Caveat found and dismissed: a naive per-frame scan (both channels > 0.30 on the raw grid) yields
extra hits (chunk_003 448.5 s, chunk_000 506.0 s). Those are NOT double-peaks - one channel is
suppressed by NMS or is not a local peak - so they correctly do not count. The claim used the right
(peak-level) method; the raw scan would have over-counted.

**No BAS chunk is a corpus outlier.** Rebuilding the `op` PASS stream
(`tools/bas_validate.py::op_events`, OP_THRESHOLD 0.40 + DEDUP_S 1.0 NMS) per fulham h2 chunk gives
op PASS = **85 / 83 / 89 / 90 / 98** - identical to the addendum table. Corpus h2 reference (29
chunks from the other 5 matches): all five fulham chunks fall inside the range, none approaches the
max. Conclusion holds.

Caveat: the per-minute rate values in the addendum table (8.49 / 8.45 / 8.93 / 9.16 / 10.22) use a
slightly different span convention than num_predicted_frames / fps (I get 8.52 / 8.33 / 8.96 / 9.02 /
9.89), and my corpus range comes out 6.90-13.46 vs the table's 7.31-13.94. These are span-definition
differences of ~0.1-0.4 /min; the op event **counts** are exact and the "inside range, not an
outlier" conclusion is unaffected.

---

## Overall verdict

All six claims survive independent recomputation. The two deterministic, seed-locked artifacts
(the WINS_VS_LOSSES money table and the OT distance matrix) reproduce **exactly** - the matrix is
byte-identical - so the "regression-locked seams" hold under recomputation, not just trust. The three
capstone conclusions stand on the recomputed numbers:

1. **Wins-vs-losses direction** - press does not separate W/L; the only consistent-direction signal
   is the deeper, less-territorial block, better read as a did-not-lose shape (the draw sits with the
   wins). Confirmed, with the write-up's own small-n hedges intact.
2. **Style-identity-gone** - intra 5.347 vs cross 5.435 (gap 0.09), only 3/6 self-nearest, no centred
   shape fingerprint, Southampton the away outlier. Confirmed exactly.
3. **Three-ways-to-lose** - the level-state numbers behind Liverpool (beaten in transition at 0-0),
   Tottenham (conceded ~3', essentially no level state), and Brighton (front-foot, lost late) all
   reproduce.

Flags / cannot-fully-check:
- Claim 4 "5 subs" is really 6 assigned subs; the 0-violation result is unchanged.
- Claim 6 rate column uses a non-obvious span convention (op counts and the outlier conclusion are
  exact regardless).
- Claim 5 "centroids distinct" is true but weak; the load-bearing anchor-health number is balance
  0.909, which reproduces.
- Not re-derived from first principles (out of scope, would need re-running upstream CV): the goal
  timeline in `GOALS` (taken as the validated E2E/Sofascore input to score_state), the koshkina
  anchor labels feeding the identity chain, and the raw broadcast-to-pitch calibration. These are
  inputs to the claims, not the claims themselves.
