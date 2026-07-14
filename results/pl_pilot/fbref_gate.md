# Phase-A oracle-agreement gate -- Brighton vs Manchester Utd (FINAL)

Fixture: **Brighton 2-1 Manchester Utd**, England Premier League 24/25, MW2, 2024-08-24 (Amex). Sofascore match id `12436888`.

Oracle: **Sofascore primary** (`tools.oracle`, raw response cached under `outputs\oracle\sofascore\team_stats_12436888.parquet`), cached FBref as cross-check. Our numbers are CV-derived from the aligned + linked-ball parquets.

## Oracle aggregates (Sofascore)

| team | possession % | passes cmp | passes att | shots | shots on target | xG |
|------|-------------:|-----------:|-----------:|------:|----------------:|---:|
| Brighton & Hove Albion | 48 | 407 | 477 | 14 | 5 | 2.09 |
| Manchester United | 52 | 446 | 511 | 11 | 4 | 1.43 |

## Cross-validation: Sofascore vs FBref (cached)

FBref passing/xG matchlogs were never cached, so those cells are `-` (Sofascore-only).

| team (side) | metric | Sofascore | FBref | delta |
|-------------|--------|----------:|------:|------:|
| Brighton & Hove Albion (home) | possession % | 48.0 | 48.0 | +0.0 |
| Brighton & Hove Albion (home) | shots | 14 | 14 | +0.0 |
| Brighton & Hove Albion (home) | shots on tgt | 5 | 5 | +0.0 |
| Manchester United (away) | possession % | 52.0 | 52.0 | +0.0 |
| Manchester United (away) | shots | 11 | 11 | +0.0 |
| Manchester United (away) | shots on tgt | 4 | 4 | +0.0 |

**No material disagreement** (possession within 2 pp, passes within 5%): Sofascore and FBref agree exactly on possession and shots for this fixture.

## Gate: pass volume (like-for-like)

Ball-possession is dropped from pass/fail (biased-by-construction; see caveat below). The gate metric is pass volume: pass-recall proxy = our n_passes / oracle passes_cmp.

| team | oracle passes cmp | our n_passes | pass-recall proxy |
|------|------------------:|-------------:|------------------:|
| Man Utd | 446 | 213 | 0.478 |
| Brighton | 407 | 198 | 0.486 |

**Team-symmetry check:** recall 0.478 (Man Utd) vs 0.486 (Brighton), spread = 0.009 (band <= 0.05) -> **PASS**.

Interpretation: our CV captures ~48% of completed passes (both teams), consistent with 11 linked-ball chunks of partial-match coverage. What matters for fingerprint features is that capture is team-symmetric so relative pass volumes are unbiased -- it is. The absolute-recall acceptance threshold is orchestrator-owned (STATUS.md).

## Possession proxies -- REPORTED CAVEATED (outside pass/fail)

| team | oracle possession % | our ball-poss % (trackable) | our space-control % |
|------|--------------------:|----------------------------:|--------------------:|
| Man Utd | 52 | 44.2 | 43.5 |
| Brighton | 48 | 55.8 | 56.5 |

Ball-possession share is nearest-player-to-ball on trackable frames only (biased by ball coverage); space-control is position-only territory. Both diverge from event-based possession by construction -- informative, not gated.
