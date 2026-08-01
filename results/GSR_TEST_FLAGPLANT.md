# S2 flag-plant — the frozen recipe on the official SoccerNet-GSR test split

Campaign act S2 of `docs/GSR_CAMPAIGN_BRIEF.md` §5. One score-once run of the recipe that measured
**33.20** on our internal TEST-38 subset of the public *valid* split, applied unchanged to the
**49-sequence official test split**. No tuning, no variant arms, no post-hoc knobs.

**Headline: GS-HOTA 35.90 (GS-DetA 26.74, GS-AssA 48.21) over all 49 official test sequences,
official scorer, single run.**

**Blocking caveat, stated up front (§6): the pipeline reads the test labels twice, so the
submission package must NOT be uploaded as built.** The score is a valid like-for-like extension of
our 33.20 protocol; it is *not* a legitimate blind-leaderboard number.

---

## 1. The frozen recipe

Written to `results/gsr_flagplant_frozen.json` **before** the first test sequence was read.

| | |
|---|---|
| recipe hash | `d084d11eea0426aa3c1d82bcdf025d6a696357d3997851f6b0ba2e70238a4276` |
| frozen at | `2026-07-31T18:01:34Z` (first test frame read 2026-07-31T18:16Z) |
| git HEAD | `29dc04b` |
| detect / track / calibrate | football-YOLO + ByteTrack + PnLCalib every frame, `sample_every = 1` |
| embeddings | PRTreID per detection, `frame_stride = 2` |
| per-crop OCR | 60 crops/track, `crop_scale = 1.0`, legibility+pose+PARSeq chain |
| OCR aggregation | frozen DEV-0.85-floor rule: `min_crop_conf 0.99, min_votes 5, min_legibility 0.5, emit_all False` |
| tracklet repair | GTA connector `tau = 0.040`, **splitter OFF** |
| identity solver | `results/identity_solver_config_percrop.json` — `p_correct 0.8837, app_gain 10, sim_none 0.92, topk 3, pi_none 0.7, r_abstain 0.0, max_concurrent 11`, digit-confusion prior refit on the valid split's DEV-20 (348 reads, 55 wrong, 573 aligned digit observations) |
| file hashes | solver config `a7282d9b…`, OCR rule `ff4a56b2…` |

`crop_scale 1.25` is deliberately **not** in this run (never validated on GSR data). The 33.20
config, not the 33.32 `p_correct`-refit variant, is the one that was frozen — verified by
re-solving the valid arm from the same file and reproducing 33.20 to the second decimal (§4).

Reproduce with `python -m tools.gsr_flagplant --freeze` then `--run`.

## 2. Wall times (RTX 3050 laptop, 4 GB, one stage per process)

| stage | what | wall | rate |
|---|---|---|---|
| extract | detect + track + calibrate + team + submissions | **4 h 42 m** (+ ~59 m in a killed first attempt that completed 8 sequences) | ~390 s/seq |
| base_arm | seed the GTA template from the jersey-null arm | 0.4 s | — |
| embed | PRTreID per-detection embeddings | **45 m** | 55 s/seq |
| ocr | per-crop OCR, 60 crops/track | **2 h 0 m** | 147 s/seq |
| connect | GTA connector + 4-config scoring | 2 m 51 s | — |
| votes | densified per-track votes at the frozen rule | 55 s | — |
| solve | bundles + frozen solver + 2 GS-HOTA passes | 2 m 52 s | — |
| package | zip | 5 s | — |
| **total** | | **≈ 8 h 30 m** | |

`results/gsr_benchmark/testsplit/gsr_flagplant_times.json`. The first launch was killed by the
agent harness at ~1 h; every stage is resumable per sequence, so the restart lost only the
in-flight sequence and the run continued from sequence 10.

## 3. Scores — official test split, 49 sequences, `gs_hota_full`

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| positions only, `jersey = null` (no identity model) | 16.72 | 7.09 | 39.46 | 93.69 | 9.61 |
| + GTA connector `tau 0.040`, still `jersey = null` | 17.85 | 7.08 | 44.98 | 93.68 | 10.65 |
| **+ frozen identity solver (the flag-plant)** | **35.90** | **26.74** | **48.21** | **94.00** | **37.96** |
| *diagnostic:* same predictions, jersey attribute off | 45.20 | 50.54 | 40.44 | 94.53 | 47.44 |
| *diagnostic:* position + association only | 49.15 | 57.06 | 42.37 | 94.34 | 52.31 |

Paired per sequence, solver vs connector base: **helped 49, hurt 0**, mean +18.88, Wilcoxon
p = 3.55e-15. Per-row identity accuracy (team AND jersey, 307,891 auditable rows)
0.1610 → **0.5508**; 2,681 of 3,114 merged tracklets named; coverage at jersey precision 0.85 =
**0.5175**, at 0.60 = 0.8219.

Jersey read density on test at the frozen rule: 961 of 3,666 original player/GK tracks carry a
read, **d = 0.2621** (144,493 crops, 39.4 per track, 20.6% producing a PARSeq number).

### 3.1 Per-sequence

`base` = GTA connector arm with null jerseys; `solver` = the flag-plant arm.

| sequence | base | solver | DetA | AssA | LocA | IDF1 | delta |
|---|---|---|---|---|---|---|---|
| SNGS-116 | 12.93 | **22.28** | 19.53 | 25.43 | 93.72 | 31.21 | +9.35 |
| SNGS-117 | 8.83 | **26.12** | 21.21 | 32.18 | 94.10 | 30.03 | +17.29 |
| SNGS-118 | 10.42 | **33.01** | 26.19 | 41.62 | 92.18 | 38.63 | +22.59 |
| SNGS-119 | 2.57 | **53.16** | 38.67 | 73.08 | 98.11 | 55.24 | +50.59 |
| SNGS-120 | 26.18 | **31.89** | 21.30 | 47.77 | 95.28 | 32.81 | +5.71 |
| SNGS-121 | 12.81 | **36.65** | 30.70 | 43.77 | 92.64 | 45.65 | +23.84 |
| SNGS-122 | 11.46 | **23.28** | 18.33 | 29.60 | 93.45 | 24.99 | +11.82 |
| SNGS-123 | 7.31 | **40.02** | 34.98 | 45.79 | 94.97 | 46.80 | +32.72 |
| SNGS-124 | 31.30 | **36.86** | 24.19 | 56.17 | 95.35 | 37.55 | +5.56 |
| SNGS-125 | 12.46 | **38.95** | 23.54 | 64.44 | 97.79 | 38.50 | +26.49 |
| SNGS-126 | 30.63 | **53.12** | 50.35 | 56.05 | 95.88 | 61.45 | +22.49 |
| SNGS-127 | 44.12 | **49.03** | 37.93 | 63.40 | 94.37 | 51.48 | +4.90 |
| SNGS-128 | 23.29 | **41.61** | 33.15 | 52.24 | 93.97 | 43.46 | +18.31 |
| SNGS-129 | 7.99 | **11.05** | 8.22 | 14.86 | 95.26 | 13.92 | +3.06 |
| SNGS-130 | 10.02 | **35.07** | 33.84 | 36.35 | 92.95 | 47.76 | +25.05 |
| SNGS-131 | 23.27 | **35.15** | 31.18 | 39.71 | 92.02 | 40.30 | +11.88 |
| SNGS-132 | 19.76 | **29.06** | 19.17 | 44.07 | 93.74 | 28.58 | +9.31 |
| SNGS-133 | 18.25 | **35.94** | 31.86 | 40.56 | 88.18 | 40.99 | +17.69 |
| SNGS-134 | 26.10 | **37.87** | 34.06 | 42.13 | 94.82 | 42.02 | +11.77 |
| SNGS-135 | 7.86 | **29.02** | 28.24 | 29.88 | 93.78 | 30.92 | +21.16 |
| SNGS-136 | 12.34 | **28.91** | 22.92 | 36.47 | 93.71 | 32.12 | +16.57 |
| SNGS-137 | 15.84 | **40.28** | 35.94 | 45.15 | 94.56 | 42.62 | +24.44 |
| SNGS-138 | 11.98 | **31.78** | 28.37 | 35.63 | 92.87 | 36.24 | +19.80 |
| SNGS-139 | 4.63 | **31.79** | 27.38 | 36.91 | 91.64 | 41.21 | +27.16 |
| SNGS-140 | 3.38 | **33.40** | 32.92 | 33.89 | 94.18 | 41.19 | +30.03 |
| SNGS-141 | 17.93 | **29.42** | 22.70 | 38.15 | 92.03 | 35.93 | +11.49 |
| SNGS-142 | 6.04 | **15.33** | 12.73 | 18.46 | 93.84 | 19.94 | +9.29 |
| SNGS-143 | 3.54 | **16.20** | 9.75 | 27.05 | 92.51 | 16.73 | +12.66 |
| SNGS-144 | 1.96 | **21.66** | 18.73 | 25.10 | 89.76 | 27.84 | +19.70 |
| SNGS-145 | 1.25 | **23.53** | 15.07 | 36.80 | 93.29 | 24.69 | +22.28 |
| SNGS-146 | 20.15 | **23.88** | 17.82 | 32.00 | 94.94 | 28.38 | +3.73 |
| SNGS-147 | 8.93 | **9.71** | 5.74 | 16.42 | 96.27 | 9.82 | +0.77 |
| SNGS-148 | 10.63 | **33.11** | 28.79 | 38.12 | 91.98 | 41.48 | +22.48 |
| SNGS-149 | 15.80 | **49.81** | 39.70 | 62.54 | 94.81 | 53.16 | +34.01 |
| SNGS-150 | 8.47 | **20.92** | 14.44 | 30.30 | 89.55 | 23.64 | +12.45 |
| SNGS-187 | 4.94 | **44.93** | 34.24 | 58.97 | 96.38 | 46.02 | +39.99 |
| SNGS-188 | 28.48 | **42.40** | 24.52 | 73.32 | 93.30 | 38.40 | +13.91 |
| SNGS-189 | 12.37 | **38.60** | 31.71 | 47.03 | 95.17 | 38.05 | +26.23 |
| SNGS-190 | 13.28 | **21.77** | 17.93 | 26.42 | 95.14 | 27.53 | +8.49 |
| SNGS-191 | 7.62 | **37.43** | 28.21 | 49.68 | 94.72 | 41.73 | +29.82 |
| SNGS-192 | 18.30 | **40.82** | 27.48 | 60.65 | 96.14 | 43.73 | +22.52 |
| SNGS-193 | 16.18 | **18.84** | 10.92 | 32.55 | 95.17 | 18.02 | +2.66 |
| SNGS-194 | 22.34 | **36.22** | 20.26 | 64.79 | 95.94 | 29.20 | +13.88 |
| SNGS-195 | 23.53 | **42.28** | 28.11 | 63.62 | 93.52 | 41.63 | +18.75 |
| SNGS-196 | 10.14 | **34.99** | 24.25 | 50.49 | 97.16 | 36.46 | +24.85 |
| SNGS-197 | 19.47 | **56.16** | 51.19 | 61.63 | 94.26 | 62.68 | +36.69 |
| SNGS-198 | 4.77 | **29.03** | 19.98 | 42.24 | 93.91 | 28.13 | +24.26 |
| SNGS-199 | 24.31 | **26.50** | 21.93 | 32.14 | 92.11 | 27.19 | +2.19 |
| SNGS-200 | 5.47 | **47.79** | 35.04 | 65.18 | 95.15 | 48.60 | +42.32 |

Spread: min 9.71 (SNGS-147), median 33.40, max 56.16 (SNGS-197) — a 5.8x range across clips of the
same league and camera. `results/gsr_benchmark/testsplit/gsr_flagplant_scores.json`.

## 4. Valid vs test

| | valid TEST-38 (internal) | official test-49 | delta |
|---|---|---|---|
| GS-HOTA | **33.20** | **35.90** | +2.70 |
| GS-DetA | 21.81 | 26.74 | +4.93 |
| GS-AssA | 50.55 | 48.21 | -2.34 |
| GS-LocA | 92.01 | 94.00 | +1.99 |
| per-row identity accuracy | 0.4488 | 0.5508 | +0.102 |
| coverage @ jersey precision 0.85 | 0.4141 | 0.5175 | +0.103 |
| jersey read density `d` | 0.2083 | 0.2621 | +0.054 |
| jersey-null floor (same positions) | 14.76 (58 seqs) | 16.72 | — |
| position + association only | 48.91 (58 seqs) | 49.15 | +0.24 |

The valid figures were re-derived, not copied: re-solving the valid bundles from the same frozen
config file reproduces **GS-HOTA 33.20** exactly and supplies the DetA/AssA the on-record payload
never stored (the on-disk valid arm directory had been overwritten by the 33.32 `p_correct`-refit
variant, which is a different config and is not the comparator).

**The test split is easier for this recipe, and the reason is measurable.** Localisation and
association are the same on both splits (loc_assoc 49.15 vs 48.91 — a 0.24 difference), so nothing
about the tracking improved. What moved is evidence: read density 0.2083 → 0.2621 (+26%), which
propagates almost linearly into coverage (+0.103) and identity accuracy (+0.102), exactly the
relation `results/EVIDENCE_DENSITY_LAW.md` §3 predicts. **+2.70 GS-HOTA is a property of the split,
not of the method.**

## 5. Where 35.90 sits

Official codabench 4365 *Test Phase* leaderboard (15 public entries, fetched from
`api/phases/7455/get_leaderboard/`, same 49 sequences):

| rank | GS-HOTA | GS-DetA | GS-AssA | entrant |
|---|---|---|---|---|
| 1 | 61.48 | 48.47 | 78.00 | myyyy |
| 2 | 58.17 | 44.44 | 76.17 | Metrica-Sports |
| 3 | 58.06 | 41.33 | 81.58 | as |
| … | | | | |
| 9 | 44.42 | 24.74 | 79.80 | lianyou |
| 10 | 35.98 | 20.63 | 62.74 | hjkim |
| — | **35.90** | **26.74** | **48.21** | **ours (not submitted — see §6)** |
| 11 | 33.12 | 19.35 | 56.72 | lsmuqi |
| 14 | 23.09 | 11.11 | 48.01 | fmagera (organiser baseline) |

Two things the columns say plainly:

1. **Our GS-DetA (26.74) is the 9th best of the 16 and beats every entry below rank 8** — including
   rank 9's 24.74. Under the identity gate we detect-and-name as well as the upper-middle of the
   field.
2. **Our GS-AssA (48.21) is the worst number in the entire table.** Every other entry is 48.01+ and
   the top nine are 62-82. Association — keeping one identity on one person across the clip — is
   where the deficit is concentrated, not detection and not the jersey reader.

That inverts the campaign brief's working assumption (§1: "jersey errors cost ~2.5x anything else").
Jersey evidence was the binding constraint at 24.19; at 35.90 the binding constraint has moved to
**association**. The next lever named in the brief (S3, the CLIP identity model) targets DetA, which
is already our relatively strong column.

## 6. NEGATIVE — the submission package cannot be uploaded as built

The chain consumes the test-split ground truth in two places, both inherited from the valid-split
harness where the leak was harmless (we were not submitting anything):

1. **`eval.gsr_score.resolve_team_map`** picks the KMeans cluster → `left`/`right` permutation by
   agreement with GT positions. Defensible as unsupervised-label resolution, but it is GT, and a
   blind split would need a geometric substitute (which does not exist in the code).
2. **`eval.gsr_identity._roster`** builds the solver's identity slots from the exact set of
   `(team, jersey)` pairs present in that sequence's `Labels-GameState.json` — typically ~15-20
   slots instead of the 198 an unconstrained 1..99 × 2-team roster would give. This is a strong
   prior handed to the assignment solver, documented in the code as "the lineup-sheet analogue".
   In deployment a team sheet is a legitimate input; here it is read off the evaluation labels.

Codabench 4365 scores **these same 49 labelled sequences**, so uploading the zip would post a
GT-informed number to a public leaderboard. `results/gsr_submission/manifest.json` carries a
`DO_NOT_UPLOAD` field saying so.

This does not invalidate 35.90 as an internal number: the valid-split 33.20 was measured under the
identical two conditions, so §4's comparison is like-for-like and every arm in §3 shares the same
handicap-or-help. It does mean **35.90 is not comparable to §5's leaderboard column** and the
comparison there is offered as orientation, not as a rank claim.

The size of the leak was **not measured** — a roster-free / team-map-free ablation is a variant arm,
and the protocol forbids variant arms on test. It must be measured on train/valid before any
submission (recommended as the first act of S3).

## 7. Other negatives and honest limits

1. **The aggregated 20-crop Koshkina reader was not run on test.** Its only downstream effect is
   `attributes.jersey` on player rows of the GTA arm, and `write_solver_submissions` overwrites that
   field on every player row unconditionally. Verified empirically on the valid split: of 229 tracks
   the base arm named, the solver **nulled 19 and changed the value on 53** — the base arm's jersey
   cannot reach the scored artifact. Consequence: the `base` column of §3.1 is the jersey-null
   connector arm (17.85), not the propagated-jersey arm (23.53) the valid-split tables used, so the
   +18.88 deltas are larger than the valid-split +9.01 by construction and the two are not
   comparable.
2. **One-run discipline held, but one process was killed.** The harness killed the first background
   run at ~1 h (8 sequences complete). Restart was from disk state at the identical frozen config;
   no sequence was processed twice with different settings.
3. **Two sequences got worse identity accuracy** under the solver than the baseline (worst
   -0.050), against 47 better. Every sequence improved on GS-HOTA.
4. **SNGS-147 barely moves** (8.93 → 9.71, DetA 5.74). Its GT is 45.1% jersey-labelled and our
   pitch-coverage there is poor; it is the floor of the distribution and was not diagnosed.
5. **GSR labels no goalkeeper numbers**, so nothing here speaks to the GK-naming defect.
6. **Extract ran ~390 s/sequence against the ~200 s/sequence in the valid-split log.** The cause was
   not investigated; same code, same settings, same machine. It doubled the dominant cost.

## 8. Files

- `tools/gsr_flagplant.py` — the driver (`--freeze` / `--run` / `--stage` / `--package` / `--demo`).
- `eval/gsr_score.py` — `--seqs` added to `run_benchmark` (the data dir holds all four splits flat).
- `results/gsr_flagplant_frozen.json` — the frozen recipe + hash + timestamp.
- `results/gsr_benchmark/testsplit/` — `gsr_scores.json` (jersey-null arm), `gsr_scores_gta.json`
  (connector arm), `gsr_identity_testsplit_percrop.json` (solver, gates + coverage curve),
  `gsr_flagplant_scores.json` (DetA/AssA per arm and per sequence), `gsr_flagplant_times.json`.
- `results/gsr_submission/gsr_testphase_d084d11e.zip` (24.6 MB, 49 entries, 333,274 predictions),
  `manifest.json`, `zip_selfscore.json`.
- `outputs/gsr_test/` — positions, PRTreID cache, per-crop OCR, votes, bundles, all arms.

### Submission format (verified, not guessed)

Downloaded the official example, `sn-gamestate/examples_predictions/SoccerNetGS-test.zip`
(69,319,587 bytes, 49 entries). Layout: **one top-level folder** (`tracklab/`) holding
`<SEQ>.json`, one per test sequence, each `{"predictions": [...]}` in the Labels-GameState
prediction schema (`image_id`, `track_id`, `supercategory`, `category_id`, `bbox_pitch`,
`bbox_image`, `attributes {role, jersey, team}`). **No metadata file.** Our zip reproduces that
layout exactly, including the folder name, and re-scoring the zip *as a zip* — unpacked into a
trackers folder with `TRACKER_SUB_FOLDER=""`, the way the server's scoring program consumes it —
returns **GS-HOTA 35.90 / DetA 26.74 / AssA 48.21**, identical to the arm directory. The package is
mechanically valid; §6 is why it is not upload-ready.

Competition 4365 is titled *"2025 SoccerNet Game State Reconstruction Competition - Test Phase"* and
its phase 7455 evaluates the test split, i.e. the labels we hold. The separate *challenge* split
(36 clips, labels withheld) is a different competition and was never fetched.
