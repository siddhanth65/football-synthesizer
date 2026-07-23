# Pair analysis v1 -- 6 same-opponent home/away legs across the ten Hag -> Amorim change

Corpus = 12 processed matches = 6 reverse-fixture pairs vs the same six opponents, straddling the
managerial change (ten Hag Aug-Sep 2024 / Amorim Jan-Feb 2025). This deliverable (a) adjudicates the
two flagged degenerate-kit mappings, (b) tabulates each pair leg-by-leg, (c) measures repeat-measurement
consistency of the team-stable metrics, and (d) states the venue/manager confound precisely.

All team-attributed numbers below use the **corrected** ManU<->opponent mapping. Two of the twelve
matches were mislabelled; the corrections are applied *in this analysis only* -- per the task, no shipped
artifact was rewritten (affected-artifact list in the last section, for the orchestrator's decision).

---

## Task 1 -- Kit-mapping adjudication (do first; two matches gated)

Method: a match's parquet team labels (`team` 0/1) come from `generator.team_anchor.align_teams_by_color`
(darker-kit cluster -> team 0). For degenerate near-identical kits it can assign the wrong cluster. Two
independent oracles test the assignment:
- **(b) possession/pass split** -- BAS carrier `n_poss_links` per team (`fingerprint.pass_network`) vs the
  Sofascore possession/pass majority. The dominant-link cluster must be the dominant-possession team.
- **(a) goal direction** -- an E2E goal enters the *conceding* team's defended end. The defended end per
  team is read from the goalkeeper median-x in the chunks where the two keepers separate cleanly
  (>40 m apart); the goal end is read from the linked-ball trajectory at the goal frame.

### VERDICT 1 -- `manutd_southampton` (ManU 3-1 Southampton, Amorim home): mapping CORRECT

- **Possession (agree).** parquet team0 (labelled Man Utd) = 65 poss-links vs team1 (Southampton) = 51
  -> ManU 56% link share. Sofascore ManU 60% possession (594 vs 406 passes). Dominant cluster = team0 =
  ManU. **Consistent.**
- **Goal direction (consistent, weak).** Clean half-time keeper end-swap: team0 defends x=0 in h1 and
  x=105 in h2 (a single physical team switching ends), so labels are globally consistent. The 3 detected
  h2 goals sit in the x=0 half where ManU (team0) attacks after the swap; ball tracking was too sparse at
  the goal frames to pin the goal line, so this line corroborates without contradicting.
- **Confidence: HIGH.** Possession is decisive and matches the registry; label consistency verified.

### VERDICT 2 -- `tottenham_manutd` (Tottenham 1-0 ManU, Amorim away): mapping FLIPPED

- **Possession (says flip).** parquet team1 (labelled Man Utd) = 112 poss-links vs team0 (Tottenham) = 61
  -> the team currently *called* ManU has 65% of the links. But Sofascore says **Tottenham** had the ball
  (56% possession, 521 vs 412 passes). The dominant-link cluster is therefore Tottenham, not ManU:
  parquet team1 = Tottenham, team0 = Man Utd.
- **Goal direction (says flip, independently).** Maddison's goal (0.88 E2E peak, h1 12:13) put the ball
  physically at x~=3, y~=33 -- the x=0 goalmouth. Keeper evidence in the chunks that separate cleanly
  (h1_chunk_003: team0 kpr x~=12.6, team1 x~=88.7; clean half-time swap in h2_chunk_003: team0 x~=91,
  team1 x~=4) shows **team0 defends x=0 in h1**. Tottenham scored, so ManU conceded at x=0 -> ManU = team0.
- **Both lines agree: parquet team0 = Man Utd, team1 = Tottenham. Registry `teams: [Tottenham, Man Utd]`
  is FLIPPED; correct order is `[Man Utd, Tottenham]`. Confidence: HIGH** (two independent lines agree;
  the half-time keeper end-swap confirms labels are globally consistent, not scrambled per chunk).

### VERDICT 3 -- `southampton_manutd` (Southampton 0-3 ManU, ten Hag away): mapping FLIPPED (NOT flagged by the task)

A whole-corpus screen (ManU poss-link share vs Sofascore possession, per match) flags exactly the
matches where the two disagree on *who had the ball*. Only three matches cross the line by more than BAS
noise, and they are exactly the red/striped-kit ones: the two above **plus `southampton_manutd`**
(Southampton's home kit is red/white stripes -- the "stripe collapse" the anchor code documents). It was
not in the task's list, but the evidence is as strong as Verdict 2, so it is reported here.

- **Possession (says flip).** parquet team0 (labelled Southampton) = 69 poss-links vs team1 (Man Utd) =
  39 -> the team *called* ManU has only 36% of the links, yet Sofascore has **ManU at 56%** possession
  (the away side dominated the ball in a 0-3 win). The dominant-link cluster is ManU: team0 = ManU.
- **Goal direction (says flip, independently, 3 goals).** Labels are globally consistent (clean half-time
  swap: team1 defends x=0 in h1, x=105 in h2). Southampton scored 0, so all three ManU goals enter
  team1's end: the two h1 goals (35:03, 41:02) tracked to x~=1-2 (the x=0 goal, team1's h1 end) and the
  h2 goal (51:09) tracked to x~=88 (the x=105 goal, team1's h2 end). Every goal enters **team1**'s end ->
  team1 conceded -> team1 = Southampton, team0 = Man Utd.
- **Both lines agree: parquet team0 = Man Utd, team1 = Southampton. Registry `teams: [Southampton, Man
  Utd]` is FLIPPED; correct order is `[Man Utd, Southampton]`. Confidence: HIGH** (three goals + possession
  all agree).

Control checks: `manutd_tottenham` (ManU 0-3 Tottenham, ten Hag home) verified CORRECT by goal direction
(Tottenham's h2 goal enters team0's end and team0 conceded 0-3 = ManU = team0, matching registry);
`manutd_liverpool` / `liverpool_manutd` have no ball at the goal frames but possession direction agrees
with Sofascore and their kits are distinct (Liverpool white away / red home vs ManU) -- no red flag. The
eight distinct-kit matches all pass the possession-direction screen.

**Downstream impact of the two flips -- reported, NOT rewritten (orchestrator decision):**

`fingerprint/score_state.py:65` resolves ManU by `match.teams.index("Man Utd")`, so for both flipped
matches every score_state-derived "ManU" number is actually the **opponent's**. Affected shipped
artifacts:

| Artifact | What is wrong | Corrected direction |
|----------|---------------|---------------------|
| `data/matches.yaml` | `tottenham_manutd` and `southampton_manutd` `teams` order flipped | `[Man Utd, Tottenham]`, `[Man Utd, Southampton]` |
| `outputs/southampton_manutd/facts/pass_network.json` | team names + poss-links swapped | ManU 69 / Southampton 39 (was 39/69) |
| `results/BLOCK_AND_STYLE_v1.md` | southampton_manutd rows swapped; **"Southampton pressed high 37.4 m and lost 0-3" is false** -- that 37.4 m high block (34.5% high spells) is **ManU's**; Southampton sat at 24.0 m. ten-Hag ManU mean line 27.4 -> ~29.7 m | ManU=37.4 m high; opponents ALL sit low/mid vs Utd (no high-press exception) |
| `results/GAME_STATE_v2.md` | southampton_manutd ManU counter-press + attack-typing rows are Southampton's; corrupts ten-Hag pooled counter-press, attack-mix, and the WP-band per-match rows | recompute with team0 |
| `results/WINS_VS_LOSSES.md` | the **Southampton 0-3 win** row (cp 0.394, block 32.9, att-3rd 0.111, trans 46.2) is Southampton's, not ManU's -- and ManU actually pressed HIGH (37.4 m) in that win, which **contradicts the "wins = deeper block" arrow** | recompute; the deep-block win-shape weakens |
| `results/PASS_NETWORKS_v1.md` | southampton_manutd poss-link labels swapped (structural metrics are empty/NaN, so only labels+counts) | ManU 69 / SOU 39 |

E2E summaries, `bas_validation.md`, WP(t) parquets and the FBref season profile are team-agnostic (or
scoreline-only) and are **not** affected. `tottenham_manutd` was not yet in the block/pass-network/game-
state reports (processed after them), so its only shipped exposure is the registry `teams` order.

---

## Task 2 -- Pair tables (home leg vs away leg, ManU perspective, corrected mapping)

Columns: `blk` = de-biased block line (m from own goal), `lo/hi` = low/high block-spell share, `vspr` =
vertical compactness (m, smaller=tighter), `poss%` = ManU poss-link share, `sofa%` = Sofascore ManU
possession, `opPASS/band` = BAS op-PASS total / Sofascore ratio, `cov` = post-link ball coverage
(pooled rows/dense, approx), `attack f/s/d` = fast-transition/sustained/direct 3-way counts (low cov),
`shots/xG` = Sofascore ManU, `res` = ManU result. Block/`poss%` for the two flipped legs are corrected;
their attack-mix and counter-press are **corrupted in the shipped engine** (opponent's) and marked `[opp]`.

Opponent block line (`opp blk`) is shown to confirm the opponents-sit-deep pattern in both legs.

### 1. Brighton -- both losses
| leg | venue | mgr | res | blk | lo/hi | vspr | poss% | sofa% | opPASS/band | cov | attack f/s/d | shots/xG | opp blk |
|-----|-------|-----|-----|----:|-------|-----:|------:|------:|-------------|----:|--------------|----------|--------:|
| manutd_brighton | H | AM | L 1-3 | 29.6 | .57/.11 | 5.9 | 47.8 | 52 | 960 / 1.067 | 51.4 | 0/7/2 | 10 / 1.48 | 22.4 |
| brighton_manutd | A | TH | L 1-2 | 33.2 | .52/.22 | 6.2 | 47.5 | 52 | 981 / 0.993 | 51.1 | 3/5/2 | 11 / 1.43 | 23.8 |

### 2. Liverpool -- loss / draw
| leg | venue | mgr | res | blk | lo/hi | vspr | poss% | sofa% | opPASS/band | cov | attack f/s/d | shots/xG | opp blk |
|-----|-------|-----|-----|----:|-------|-----:|------:|------:|-------------|----:|--------------|----------|--------:|
| manutd_liverpool | H | TH | L 0-3 | 29.5 | .59/.15 | 5.9 | 52.8 | 53 | 981 / 1.010 | 44.5 | 0/8/3 | 8 / 1.36 | 25.8 |
| liverpool_manutd | A | AM | D 2-2 | 20.8 | .77/.09 | 5.7 | 34.8 | 47 | 886 / 1.079 | 36.0 | 1/6/5 | 13 / 1.05 | 27.6 |

### 3. Fulham -- both 1-0 wins
| leg | venue | mgr | res | blk | lo/hi | vspr | poss% | sofa% | opPASS/band | cov | attack f/s/d | shots/xG | opp blk |
|-----|-------|-----|-----|----:|-------|-----:|------:|------:|-------------|----:|--------------|----------|--------:|
| manutd_fulham | H | TH | W 1-0 | 25.1 | .73/.11 | 6.1 | 58.0 | 55 | 945 / 1.091* | 40.0 | 3/9/4 | 14 / 2.43 | 21.2 |
| fulham_manutd | A | AM | W 1-0 | 27.0 | .65/.11 | 6.6 | 51.7 | 49 | 1017 / 1.030 | 56.1 | 4/9/7 | 4 / 0.25 | 21.7 |

### 4. Crystal Palace -- loss / draw
| leg | venue | mgr | res | blk | lo/hi | vspr | poss% | sofa% | opPASS/band | cov | attack f/s/d | shots/xG | opp blk |
|-----|-------|-----|-----|----:|-------|-----:|------:|------:|-------------|----:|--------------|----------|--------:|
| manutd_palace | H | AM | L 0-2 | 30.1 | .57/.23 | 6.6 | 74.0 | 67 | 943 / 1.108* | 46.5 | 0/3/7 | 17 / 1.12 | 24.7 |
| palace_manutd | A | TH | D 0-0 | 26.7 | .62/.07 | 5.8 | 84.4 | 67 | 965 / 1.014 | 36.6 | 0/14/4 | 15 / 1.70 | 28.4 |

### 5. Southampton -- both wins  (away leg = the FLIPPED match)
| leg | venue | mgr | res | blk | lo/hi | vspr | poss% | sofa% | opPASS/band | cov | attack f/s/d | shots/xG | opp blk |
|-----|-------|-----|-----|----:|-------|-----:|------:|------:|-------------|----:|--------------|----------|--------:|
| manutd_southampton | H | AM | W 3-1 | 25.2 | .71/.14 | 6.9 | 56.0 | 60 | 1066 / 1.066 | 49.1 | 2/5/3 | 23 / 3.42 | 29.7 |
| southampton_manutd | A | TH | W 3-0 | **37.4** | .50/.35 | 5.8 | **63.9** | 56 | 1087 / 0.996 | 41.7 | [opp] | 20 / 2.67 | **24.0** |

### 6. Tottenham -- both losses  (away leg = a FLIPPED match)
| leg | venue | mgr | res | blk | lo/hi | vspr | poss% | sofa% | opPASS/band | cov | attack f/s/d | shots/xG | opp blk |
|-----|-------|-----|-----|----:|-------|-----:|------:|------:|-------------|----:|--------------|----------|--------:|
| manutd_tottenham | H | TH | L 0-3 | 26.0 | .68/.11 | 6.2 | 30.7 | 39 | 1079 / 1.047 | 41.9 | 0/0/3 | 11 / 0.96 | 27.1 |
| tottenham_manutd | A | AM | L 0-1 | **28.5** | .55/.20 | 6.5 | **35.3** | 44 | 1036 / 1.110* | 49.6 | [opp] | 16 / 1.52 | **19.9** |

`*` = op-PASS ratio outside the frozen 0.97-1.09x Sofascore band (fulham/palace home over; tottenham away
1.110). `[opp]` = attack-mix corrupted by the mapping flip in the shipped engine (would be the opponent's).
`opp blk` bold in pairs 5-6 = the corrected opponent value after the flip.

**Which side of each metric flips with venue vs holds with team identity (per pair):**
- **Result**: holds (same both legs) in 4/6 pairs (Brighton LL, Fulham WW, Southampton WW, Tottenham LL);
  the other two differ only by draw-vs-loss (Liverpool L/D, Palace L/D). No pair flips win<->loss.
- **Poss-link share**: holds strongly -- the team ManU out-passed at home it out-passed away and vice
  versa (see Task 3 correlation).
- **Block line / high-share**: does NOT hold -- swings 2-12 m within a pair (see Task 3).

---

## Task 3 -- Repeat-measurement consistency (the honest error bar on "team identity")

Because a pair's two legs differ in **both** venue and manager against the **same** opponent, cross-leg
agreement measures how much of a metric is opponent/team-driven (stable) vs venue/manager/noise-driven.

| metric (ManU) | mean abs-diff between legs | cross-leg corr r (n=6 pairs) | read |
|---------------|--------------------------:|-----------------------------:|------|
| poss-link share | 7.9 pp (median 7.1) | **+0.83** | **stable / opponent-driven** |
| block line (m) | 5.4 m (median 3.5) | **-0.40** | **not stable** |
| block high-spell share | 10.5 pp | (n/a, small counts) | not stable |
| result (W/D/L) | -- | 4/6 identical, 6/6 same category | **stable / opponent-driven** |

- **Possession identity is the one repeatable fingerprint.** Cross-leg r = +0.83: ManU dominated the
  ball against Palace (74/84%) and Fulham (58/52%) in both legs, was roughly even vs Brighton/Liverpool,
  and was out-possessed by Tottenham (31/35%) in both. The mean 7.9 pp leg-to-leg wobble is the honest
  error bar; the *ordering* of opponents by how much ManU controls the ball is preserved.
- **Block height is NOT a stable team fingerprint at this n.** The between-leg spread (5.4 m mean, 12.2 m
  for Southampton) is *larger* than the whole-corpus between-leg-median and the cross-leg correlation is
  **negative** (-0.40) -- i.e. knowing ManU's block line in the home leg tells you nothing (or the wrong
  sign) about the away leg vs the same opponent. This sits on top of the acknowledged ~7 m de-bias scale
  uncertainty + forward-pass broadcast bias (BLOCK_AND_STYLE Gate 1 pending). Read block height as a
  match-state descriptor, not a team-identity claim.
- **Counter-press consistency is NOT reported** for the pairs: it exists only for the 6 ten-Hag matches
  (GAME_STATE/WINS), and one of those (southampton_manutd) is corrupted by the flip. With one leg per pair
  missing or corrupted, no pair has two clean counter-press legs. This is a genuine gap, not an omission.

---

## Task 4 -- The venue / manager confound (stated precisely)

The naive worry ("all six pairs confound venue with the manager change") is **half right**. Within a
single pair the two legs differ in **both** venue and manager, so a single pair's home-vs-away gap cannot
be split into a venue effect and a manager effect. But across the six pairs the design is a **balanced
2x2** (venue x manager), n=3 opponents per cell:

| | ten Hag | Amorim |
|---|---|---|
| **Home** | liverpool, fulham, tottenham | brighton, palace, southampton |
| **Away** | brighton, palace, southampton | liverpool, fulham, tottenham |

Consequences:
- **Estimable (direction only, n=3/cell, venue-balanced):** the pooled **manager** main effect. Corrected
  aggregates: ten Hag block 29.7 m / poss-link 56.2% / high-share 16.7%; Amorim 26.9 m / 49.9% / 14.6%.
  So *after the flip correction* ten Hag ManU reads a shade **higher / more possession-heavy**, reversing
  the shipped "similar depth" read -- but this swing is driven largely by the one corrected leg
  (southampton_manutd 37.4 m into the ten-Hag cell), so treat as fragile direction, n=6/regime.
- **Estimable (direction only, manager-balanced):** the pooled **venue** main effect -- and it is
  essentially **null**: home block 27.6 m / poss 53.2% vs away 28.9 m / 52.9%. ManU's possession and block
  depth barely move with venue; they move with the opponent.
- **NOT estimable:** the venue x manager interaction, and any single pair's home-away decomposition.
- **What the design genuinely buys:** each opponent measured **twice** under different venue+manager, which
  is exactly the repeat-measurement test in Task 3 (and it says: possession identity and result are
  opponent-stable; block height is not).

---

## Task 5 -- Five sharpest honest findings

1. **A second, unflagged mapping flip (`southampton_manutd`) falsifies a shipped headline.** The
   "Southampton pressed high (37.4 m) and lost 0-3 vs United" claim is backwards: that high block is
   **United's** (a genuine 37.4 m high-press away win, United's highest line in the corpus), Southampton
   actually sat at 24.0 m. Corrected, **all 11 opponents sit low/mid vs United -- no high-press
   exception.** Root cause: `score_state.py:65` trusts the registry team order, which the red/striped-kit
   anchor got wrong. Both flips were caught by one cheap screen: poss-link majority vs Sofascore
   possession majority.

2. **Possession identity is the only repeatable team fingerprint here (cross-leg r = +0.83).** United's
   control-of-ball ordering vs each opponent survives a venue *and* manager change; block height does not
   (r = -0.40). If the thesis needs one falsifiable "team identity" number from this corpus, it is the
   possession/pass-share profile, not the block line.

3. **Results are opponent-determined, not venue- or manager-determined.** 4/6 pairs share the exact result
   across the change of venue and manager (beat Fulham and Southampton twice, lost to Brighton and
   Tottenham twice); no pair flips win<->loss. United's outcome vs a given side was locked in regardless
   of Old Trafford/away or ten Hag/Amorim.

4. **Venue has near-zero effect on United's shape; the opponent sets it.** Pooled home vs away is 27.6 vs
   28.9 m block and 53.2 vs 52.9% possession -- flat. Opponents drop into a low/mid block vs United in
   **both** legs (opp block 20-30 m in 11/12 legs). The "who imposes on whom" answer is opponent-driven
   territory, not a United home-away swing.

5. **The corrected manager read reverses the shipped one, but rests on one leg.** With southampton_manutd's
   37.4 m moved into the ten-Hag column, ten Hag now reads higher-block / more-possession than Amorim
   (29.7 vs 26.9 m; 56.2 vs 49.9%), opposite to BLOCK_AND_STYLE's "similar depth, Amorim deeper." One
   corrected match drives most of the swing, so this is a fragile direction (n=6/regime, one high-leverage
   leg) -- honest verdict: the block metric still does **not** cleanly separate the two managers, and
   whatever separation exists is sensitive to a single mapping fix. Do not ship a manager-block claim
   until GAME_STATE/WINS are recomputed with the corrected mappings.

---

### Provenance
Adjudication: `fingerprint/pass_network.py` (poss-links), `outputs/oracle/sofascore/team_stats_*.parquet`
(possession/passes/shots/xG), `results/action_spotting_probe/<id>/summary.json` (goal peaks),
`outputs/<id>/final/ball/` + `outputs/<id>/final/match_aligned.parquet` (goal-end + keeper ends via
`fingerprint.block_height` / `structural_metrics.resolve_attack_directions`). Block geometry:
`fingerprint.block_height.block_summary`. Sofascore id map: brighton_manutd 12436888, manutd_liverpool
12436920, manutd_fulham 12436870, palace_manutd 12436962, manutd_tottenham 12436995, southampton_manutd
12436949, liverpool_manutd 12436514, manutd_brighton 12436883, fulham_manutd 12436899, manutd_palace
12436925, manutd_southampton 12436516, tottenham_manutd 12436952. Ball coverage is a pooled
rows/dense-frames approximation (a few pp off the per-chunk-mean figures in STATUS). No artifact was
rewritten; no commits.
