# Brighton 2-1 Man Utd — CV pipeline vs oracle, side-by-side (data extraction only)

Fixture: England Premier League 24/25 MW2, Brighton & Hove Albion (home) 2-1 Manchester United
(away), 2024-08-24 (Amex). Sofascore fixture id `12436888`.

Sources read:
- `outputs/facts/brighton_manutd.json` — CV fact store (`metrics_version: "2026.07.3"`, generated
  2026-07-11T12:20:34+00:00; `cv.sources.cv = outputs\brighton_manutd\final\match_aligned.parquet`)
- `outputs/oracle/sofascore/team_stats_12436888.parquet` (125 rows, Sofascore raw team-stat feed)
- `outputs/oracle/sofascore/match_dicts_England_Premier_League_24_25.json` (381-match fixture-metadata
  cache; entry `id==12436888`)
- `outputs/eval/brighton_manutd_ball_eval.json`
- `results/pl_pilot/fbref_gate.md`, `results/pl_pilot/possession_diagnosis.md`
- `results/report_v2_brighton_manutd.md`
- `outputs/brighton_manutd/final/match_aligned.parquet` (171,953 rows; per-frame per-track table)

No files above were edited. This document is a new file.

---

## A. Statistical table

Team-id note: fact store keys are `Man Utd` (team0, CV-anchored, confirmed red shirts) and
`Brighton` (team1). Sofascore/FBref list Brighton as home. Both orderings shown below use the
official team name so there is no ambiguity.

| metric | our CV value | oracle value (Sofascore) | ratio / gap | verdict |
|---|---|---|---|---|
| Possession share | Man Utd 43.5% / Brighton 56.5% (`cv.space.*.space_control`, `outputs/facts/brighton_manutd.json:126-133`); separately, ball-proximity possession Man Utd 44.2% / Brighton 55.8% (`results/pl_pilot/possession_diagnosis.md:8-9`, 4157 samples) | Man Utd 52% / Brighton 48% (`outputs/oracle/sofascore/team_stats_12436888.parquet` row 0, `homeValue=48.00` Brighton, `awayValue=52.00` Man Utd) | gap 8-12 pp, **and the leader inverts** (oracle: Utd majority; ours: Brighton majority) | **biased** — ours is "trackable-frame share" (proximity-to-ball on covered frames) / "position-only territory control", not Opta time-possession; explicitly dropped from the pass/fail gate (`results/pl_pilot/fbref_gate.md:42-49`, `results/pl_pilot/possession_diagnosis.md:160-163`) |
| Passes (volume) | Man Utd 213 / Brighton 198 detected (`cv.passing.*.n_passes`, fact store lines 199,205) | Man Utd 446 / Brighton 407 completed passes (`results/pl_pilot/fbref_gate.md:11-12`, Sofascore `passesAccurate`-style aggregate) | recall proxy: Man Utd 213/446=47.8%, Brighton 198/407=48.6% (`outputs/eval/brighton_manutd_ball_eval.json`: `pass_recall_proxy: 0.482`) | **validated proxy (partial)** — recall is team-symmetric (spread 0.9 pp, passes gate protocol accepts symmetry <=5pp) so relative pass volume is usable for fingerprinting, but absolute recall (~48%) falls short of the report-v2 50% evidence-gate threshold by 1.8 pp -> ball-family sections **abstain** in the actual report (`results/report_v2_brighton_manutd.md:5,9`) |
| Shots | **NOT MEASURED** — no CV shot detector exists in this pipeline (confirmed in `results/report_v2_brighton_manutd.md:56,72-77`: "We have no CV shot detector, so shots are an open gap shown for context only") | Man Utd 11 / Brighton 14 total shots (`outputs/oracle/sofascore/team_stats_12436888.parquet` row 4, `homeValue=14.00` Brighton, `awayValue=11.00` Man Utd); shots on target Brighton 5 / Man Utd 4 (`results/pl_pilot/fbref_gate.md:11`) | n/a | **absent** — honest gap, not a biased/wrong measurement, a non-existent one |
| xG | ours = `ball_xt` (Expected Threat accumulated along ball-carry moves): Man Utd `xt_created=2.720` over `n_moves=546`, Brighton `xt_created=2.934` over `n_moves=797` (`outputs/facts/brighton_manutd.json:211-219`) | Sofascore xG: Man Utd 1.43 / Brighton 2.09 (`outputs/oracle/sofascore/team_stats_12436888.parquet` row 2, `homeValue=2.09` Brighton, `awayValue=1.43` Man Utd) | not comparable — different quantities and different scales (xT is a possession-value integral over ball moves, not a shot-based scoring-probability sum) | **not comparable, mislabel risk** — the values happen to sit in a similar numeric range (~1.4-2.9) purely by coincidence of scale; report-v2 correctly withholds ball-xT under the current ball-evidence gate (`results/report_v2_brighton_manutd.md:26`, "ball-xT ... withheld") and never states it against Sofascore xG in the appendix table |

CV-only metrics with **no public-report / Sofascore equivalent** (all from
`outputs/facts/brighton_manutd.json`, `cv.*`):

| metric | Man Utd | Brighton | source path |
|---|---|---|---|
| De-biased defensive line height (m) | 33.0 | 25.4 | `cv.line_height.*.def_line_debiased_m` |
| Raw (censoring-inflated) defensive line height (m) | 53.6 | 46.1 | `cv.line_height.*.def_line_raw_m` |
| Formation read | 3-5-2 (mean fit cost 21.88 m, 5/11 chunks agree) | 4-3-3 (mean fit cost 21.89 m, 4/11 chunks agree) | `cv.formation.*` |
| Block width / length (m) | 30.6 / 18.6 | 33.0 / 21.6 | `cv.tendencies.*.width.mean`, `.length.mean` |
| Compactness (nearest-teammate spread index, lower = tighter) | 11.43 | 12.76 | `cv.tendencies.*.compactness.mean` |
| Lane occupation — half-space / centre / wing | 48.6% / 33.0% / 18.4% | 45.0% / 35.3% / 19.8% | `cv.theory.*.halfspace_share`, `centre_share`, `wing_share` |
| Counterpress rate | 70.1% | 77.4% | `cv.transitions.*.counterpress_rate` |
| Pressing intensity | 0.364 | 0.437 | `cv.theory.*.pressing_intensity` |
| PPDA (CV-derived) | 1.13 | 0.99 | `cv.passing.*.ppda` |
| Velocity synchrony (0-1) | 0.720 | 0.703 | `cv.style.velocity_synchrony.*` |
| Space control / attacking-third control | 43.5% / 43.3% | 56.5% / 56.5% | `cv.space.*` |
| Phases split (build-up / progression / final-third %, of their own possession windows) | 33.9 / 27.4 / 38.7 | 49.5 / 22.5 / 28.0 | `cv.phases_pct.*` (WITHHELD in report-v2 — ball-dependent, gate fails) |
| Verticality | 0.175 | 0.191 | `cv.theory.*.verticality` |
| Overload share | 58.7% | 53.2% | `cv.theory.*.overload_share` |
| Counterpress regain curve (3s/4s/5s/6s/8s) | 42.0/46.6/52.3/57.2/66.3% | 49.4/54.3/61.9/66.4/72.1% | `cv.theory.*.counterpress_regain_curve` |

---

## B. Tactical paragraph data (what our structural sections actually claim)

Extracted from `results/report_v2_brighton_manutd.md` (the rendered report, Man Utd side only — the
report as generated has no equivalent Brighton prose section, only the Man Utd team is narrated;
see caveat below) and cross-checked against `outputs/facts/brighton_manutd.json`.

- **Formation read (rendered):** "Our shape classifier reads Man Utd in a **3-5-2**." — matches fact
  store `cv.formation."Man Utd".formation = "3-5-2"` (`results/report_v2_brighton_manutd.md:13`,
  `outputs/facts/brighton_manutd.json:137`).
- **Line height (rendered):** "their visibility-corrected defensive line sits **33 m** up the pitch
  ... they build from a base line around **56 m**" — matches `def_line_debiased_m: 33.0`
  (`cv.line_height`) and `buildup_height.mean: 56.28` (`cv.tendencies`)
  (`results/report_v2_brighton_manutd.md:15`).
- **Block dimensions (rendered):** "the block is **31 m** wide and **19 m** deep; nearest-team-mate
  compactness holds near **11.4**" — matches `width.mean: 30.58`, `length.mean: 18.59`,
  `compactness.mean: 11.43` (`results/report_v2_brighton_manutd.md:15`).
- **Lane occupation (rendered):** "half-spaces **49%** — the dominant channel; centre **33%**; wings
  **18%** — comparatively thin" — matches `cv.theory."Man Utd".halfspace_share: 0.486`,
  `centre_share: 0.330`, `wing_share: 0.184` (`results/report_v2_brighton_manutd.md:18-20`).
- **In-possession unit cohesion (rendered):** "velocity synchrony **72%**" — matches
  `cv.style.velocity_synchrony."Man Utd": 0.7197` (`results/report_v2_brighton_manutd.md:24`).
- **Counter-structure seams (rendered, all position-only, non-withheld):**
  1. "Attack and switch into the wide lanes ... isolating the full-backs stretches a compact block" —
     backed by the same 49/33/18 lane split above.
  2. "Target the space behind the defensive line ... line sits 33 m up the pitch behind a base line
     near 56 m" — backed by `def_line_debiased_m`/`buildup_height` above.
  3. "Stretch the block before entering it ... 31 m wide by 19 m deep at a nearest-team-mate spread of
     11.4" — backed by block-size/compactness above.
     (`results/report_v2_brighton_manutd.md:40-52`)
- **Withheld ball-dependent tactical content:** possession/tempo, PPDA, ball-xT, phase split,
  verticality (In-possession section); pressing intensity, counterpress decay curve, regains
  (Out-of-possession section); a counterpress-timing seam and a transition-after-press seam — all
  explicitly withheld under the failed pass-recall gate (48.2% < 50% threshold)
  (`results/report_v2_brighton_manutd.md:26,32,38,48`). These values DO exist in the fact store
  (see the CV-only table in Section A: counterpress rate, pressing intensity, PPDA, phases_pct,
  verticality, regain curve) but the report correctly does not narrate them for this match.

**Caveat — no Brighton tactical narrative in report-v2:** `results/report_v2_brighton_manutd.md`
only writes prose sections ("How Man Utd set up", "In possession", "Out of possession", "Counter-
structure seams") for **Man Utd**. Brighton's formation (4-3-3), line height (25.4 m), lane split
(45.0/35.3/19.8), compactness (12.76) etc. all exist in the fact store (Section A CV-only table)
but are not rendered as prose anywhere found in this file. This is worth flagging to the
orchestrator: the fact store is symmetric across both teams, but the rendered report is currently
Man-Utd-only single-team output, not a two-team comparative report — I checked but could not find a
second-team prose block in this 79-line file. If the orchestrator expected a Brighton section, it is
MISSING from `results/report_v2_brighton_manutd.md` as currently generated.

---

## C. Per-player readiness

### What already exists in our parquets (track-level)

Source: `outputs/brighton_manutd/final/match_aligned.parquet` — 171,953 rows, columns:
`frame, track_id, role, team, pitch_x, pitch_y, image_x, image_y, conf, calib_error_m, is_actor,
is_keeper, chunk`.

- **Position (pitch_x/pitch_y) per frame:** present, dense.
- **Role tag:** present (`role` in `{player, referee, ball, goalkeeper}`); `is_keeper` boolean also
  present.
- **Team assignment:** present (`team` int 0/1, jersey-colour-anchored per `STATUS.md` team-anchor
  discussion — not read from this file directly but consistent with fact-store team labels).
- **Confidence / calibration error:** present per row (`conf`, `calib_error_m`) — usable to filter
  reliable frames, as the gate already does (`calib_error_m <= 1.0`).
- **Minutes visible, distance covered, speed, position centroid, role cluster (per named player):
  NOT COMPUTABLE from this file as-is.** Checked directly: `track_id` does **not** persist across
  chunks as a stable player identity. Each of the 11 chunks independently allocates roughly
  750-1,045 unique `track_id` values (`df.groupby('chunk')['track_id'].nunique()` — e.g.
  `h1_chunk_000`: 1,005 ids, `h1_chunk_001`: 1,045 ids) even though only ~22-23 players + 1 referee
  are ever on the pitch at once — i.e. tracker re-identifies/reassigns ids constantly within and
  across chunks (ByteTrack-style track churn, no cross-chunk re-identification layer). So a
  `track_id` is a short-lived per-chunk track fragment, not a stable "Bruno Fernandes" identity.
  Aggregating `track_id`-level distance/speed/centroid today would produce per-*fragment* stats, not
  per-*player* season/match stats, and could not be matched to a jersey number or player name.
  Distance/speed/centroid ARE mechanically computable per fragment (pitch_x/pitch_y + frame number
  give displacement and implied speed at the extraction fps), just not yet rolled up to a persistent
  player.
- This matches the project's own framing in `STATUS.md` (roadmap "B2 jersey identity") and the
  jersey-feasibility probe (`results/jersey_probe/JERSEY_PROBE.md`) — persistent player identity via
  jersey-number OCR is explicitly the next phase ("Layer 2"), not yet built.

### What Sofascore's cached data offers per player

- `outputs/oracle/sofascore/team_stats_12436888.parquet` is **team-level only** (125 rows, all
  `group` values are team-aggregate categories: "Match overview", "Defending", "Goalkeeping", etc.,
  split only by `period` ALL/1ST/2ND and `home`/`away`). No player-row granularity exists in this
  file — confirmed by column list (`name, home, away, compareCode, statisticsType, valueType,
  homeValue, awayValue, renderType, key, period, group, homeTotal, awayTotal`), none of which is a
  player identifier.
- `outputs/oracle/sofascore/match_dicts_England_Premier_League_24_25.json` — 381-match fixture
  metadata cache. The Brighton-Man Utd entry (`id == 12436888`) carries boolean flags
  `hasEventPlayerStatistics: True` and `hasEventPlayerHeatMap: True`, confirming Sofascore's API
  *offers* per-player ratings/touches/passes/duels/heatmaps for this fixture, but **the actual
  per-player statistics payload was never fetched/cached** — only fixture metadata (team names,
  score, venue, timestamps, tournament/round info) is stored locally. Per-player Sofascore fields
  (rating, touches, passes, duels, etc.) are therefore **MISSING** from the local cache; they exist
  upstream but would need a fresh API pull (`hasEventPlayerStatistics=True` is the signal that a pull
  would succeed).

### Net gap

Both sides are currently team-level only for this fixture: our CV lacks persistent player identity
(track fragments only, no jersey bridge), and Sofascore's cached data lacks the per-player payload
(only the "available" flag was cached, not the data). Closing the jersey-ID layer (roadmap B2) closes
our half of the gap; a separate oracle re-pull with player-stat endpoints would be needed to close the
Sofascore half if per-player CV-vs-oracle comparison is wanted later.
