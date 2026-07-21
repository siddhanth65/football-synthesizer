# SoccerNet-GSR re-score: Koshkina jersey identities attached (official GS-HOTA lift)

The shipped benchmark (`GSR_BENCHMARK.md`) emitted `jersey = null` for every detection, so the
official `gs_hota_full` config -- which zeroes similarity unless role AND team AND jersey all match --
could never match a GT player carrying a labelled number. That produced the **14.8** floor: a direct
measurement of the pre-Layer-2 identity gap. This re-score fills the jersey field with the newly wired
Koshkina reader and re-runs the **same official evaluator** (sn-trackeval 0.4.0) on the **same** valid
split, with the geometry/tracking untouched.

**Headline: official `gs_hota_full` 14.8 -> 19.83 on the 58-sequence valid split (+5.07, +34%).**

## Method (bridge)

Code: `eval/gsr_jersey.py` (reader/vote/patch/score) on top of the existing `eval/gsr_score.py`
harness. Per valid-split sequence:

1. **Crop recovery.** For each player/GK `track_id` in the cached positions parquet, up to
   **20 crops** are sampled evenly across the track; each sample's persisted foot point is matched to
   the nearest re-detected box (same football detector, <= 6 px) and cropped. This is the exact
   crop-recovery of `generator.track_relink.fragment_embeddings`, writing crop files for the reader
   instead of embedding them -- the parquet stores only the projected foot point, so the same
   detector reproduces the source box.
2. **Tracklet vote (no roster prior).** All of a sequence's crops go through the Koshkina chain in one
   pass (`KoshkinaRecognizer.crop_probs`: ResNet34 legibility gate -> KeypointRCNN torso RoI ->
   SoccerNet-fine-tuned PARSeq in the py3.11 sidecar), then the `[N, 100]` softmax rows are grouped
   per track and consolidated with `generator.jersey_id.aggregate_votes` (confidence-weighted mean
   softmax, thresholded at `min_conf = 0.30`). A crop that fails legibility/pose/OCR reads illegible,
   so a track without a consistent legible number **abstains** (`-1`). **No roster mask** is applied:
   GSR sequences ship no squad list (unlike our Brighton-ManU match, where both squads' numbers constrain the
   decode), so every number 1..99 is admissible and the reader has no roster prior to lean on -- a
   harder setting than the anchor funnel on our own match.
3. **Patch.** Only `attributes.jersey` is filled, on `role == "player"` rows, with the voted number as
   a **string** (`"7"`, matching GT's storage and the evaluator's `str == str` comparison; an `int`
   would silently never match). Every other field -- pitch position, role, team, track id, confidence
   -- is copied verbatim from the baseline submission, so the ONLY quantity that moves is jersey.

Resumable by disk state: one vote checkpoint per sequence (`outputs/gsr/koshkina_jersey/<seq>.json`),
long GPU job. Patched submissions live in `outputs/gsr/eval_koshkina/` (baseline `eval/` untouched);
scores in `results/gsr_benchmark/gsr_scores_koshkina.json` (baseline `gsr_scores.json` untouched).

## Both arms (combined over the valid split)

Two arms: **attach** (jersey-attached, this work) vs **abstain** (jersey-null everywhere, == the
shipped baseline `gs_hota_full`). The non-jersey configs are jersey-independent and are shown as an
invariant check.

| Config | attributes gated | abstain (baseline) GS-HOTA | attach GS-HOTA | delta |
|---|---|---|---|---|
| `gs_hota_full` | role + team + **jersey** (official) | **14.76** | **19.83** | **+5.07** |
| `no_jersey` | role + team | 43.06 | 43.06 | 0.00 (invariant) |
| `role_only` | role only | 44.95 | 44.95 | 0.00 (invariant) |
| `loc_assoc` | none (loc + assoc) | 48.91 | 48.91 | 0.00 (invariant) |

Shipped baseline for continuity: full **14.8** / no-jersey **43.1** / loc+assoc **48.9** / LocA
**92.5**. The `no_jersey`/`role_only`/`loc_assoc` numbers are **identical bit-for-bit** across arms
because those configs set `USE_JERSEY_NUMBERS=False` and ignore the field -- confirmation that the
bridge perturbs nothing but jersey.

## Delta decomposition (where the +5.07 comes from)

`gs_hota_full` sub-metrics, abstain -> attach:

| Sub-metric | abstain | attach | delta | reading |
|---|---|---|---|---|
| GS-DetA | 5.98 | **9.89** | **+3.91** | the driver: correct numbers make previously-unmatchable numbered GT players matchable |
| GS-AssA | 36.43 | 39.77 | +3.34 | a correct read stitches a track's identity to the GT id across frames |
| GS-LocA | 91.35 | 91.85 | +0.50 | essentially flat -- geometry unchanged (matched set shifts slightly) |
| IDF1 | 8.07 | 14.73 | +6.66 | identity F1 nearly doubles on the numbered subset |

GS-HOTA = sqrt(DetA x AssA), so the composite lift is driven by DetA (the newly matched numbered
detections) multiplied through AssA. This is the identity half the benchmark was built to expose,
recovered without touching localization.

## Abstention analysis

- **425 of 4,870 player/GK tracks read a number = 8.73% read coverage; 91.27% abstain.** Mean 7.3
  reads/sequence (range 0-22). This is expected and honest: on wide broadcast framing the back number
  is a handful of pixels, the legibility gate rejects most crops, and the tracklet vote only commits
  when a track carries consistently legible reads. Coverage is the ceiling on the lift, not a defect.
- **49,067 of 481,038 player prediction rows (10.2%) carry an attached number** -- the read tracks
  are the longer, better-framed ones, so they cover proportionally more rows than tracks.
- The lift is large *given* the coverage: +5.07 GS-HOTA from numbering under 9% of tracks means each
  correct read is worth a lot under the jersey gate, because it flips a hard-zero similarity to a
  match. Raising coverage (higher-res crops, closer framing, roster priors) is the headroom.

Best per-sequence deltas (all from a handful of confident reads):

| Seq | abstain -> attach | delta | reads/tracks |
|---|---|---|---|
| SNGS-051 | 2.8 -> 24.8 | +22.04 | 13/66 |
| SNGS-038 | 3.2 -> 22.9 | +19.62 | 10/84 |
| SNGS-085 | 6.9 -> 22.6 | +15.71 | 14/76 |
| SNGS-056 | 6.5 -> 21.5 | +14.97 | 15/112 |
| SNGS-041 | 6.8 -> 21.4 | +14.63 | 6/79 |

Distribution over 58 sequences: **56 improved, 2 flat, 0 regressed.** Mean delta **+6.17**, median
**+4.61**. The 2 flat sequences (SNGS-043, SNGS-048) had **zero** reads -> their submissions are
byte-identical to baseline -> exactly the baseline score.

## The zero-hurt finding and why

**No sequence regressed** (`n_seqs_hurt = 0`). This is not luck; it is the interaction of two things:

1. **The `null == null` match mechanism.** In the official metric, a GT player who carries **no**
   labelled jersey has `jersey = null`, and our abstaining prediction also has `jersey = null`;
   `null == null` is **True**, so that pair can still match on role+team+position. Attaching a
   *wrong* number to a track overlapping such a GT player would flip `null == null` (match) into
   `"5" == null` (miss) -- that is the concrete way jerseys can HURT under GS-HOTA. (Verified directly
   against the evaluator's numpy object-array comparison.)
2. **Abstention discipline.** The confidence-weighted tracklet vote only commits a number when a track
   is dominated by consistently legible, agreeing crops; every uncertain track stays `null` and keeps
   its `null == null` safety. So the reader trades recall (91% abstain) for precision, and precision is
   exactly what protects the composite. The reader was strong enough that the committed 8.7% did not
   contain a net-harmful wrong-number cluster on any single sequence.

Because attaching can only help (correct read: miss -> match) or hurt (wrong read on a GT-null player:
match -> miss), and abstention removes the hurt path wherever the reader is unsure, the aggregate is a
one-directional ratchet here. If a future, higher-coverage reader commits more borderline numbers, the
hurt path reopens and per-sequence regressions become possible -- worth re-checking then.

## Caveats (honest)

- **Valid split, not the challenge leaderboard.** Published references (baseline GS-HOTA 29.01, 2024
  SOTA 63.90) are test/challenge-split numbers with a full jersey+ReID identity stack; 19.83 is our
  valid-split `gs_hota_full`. Not a like-for-like ranking -- context only.
- **Reader precision is sample-verified on a different match, not on GSR.** [CORRECTED 2026-07-20,
  claims-audit] The Koshkina chain reproduces at 86.13% tracklet accuracy on the SoccerNet
  jersey-2023 test split (1043/1211, two substitutions disclosed) -- that figure is confirmed. The
  "97.5% sample precision on our Brighton-ManU match's close-ups" previously stated here is an
  unpersisted directional human spot-check (39/40 on a 40-tile montage), not a stored measurement:
  the `verdict` fields in every relevant `levers_stats.json` are empty strings, and there exist two
  coincidentally-identical 39/40 brighton verdicts on two different reader arms (one persisted, for
  the torso/easyocr arm; one prose-only, for the Koshkina arm) that cannot be disambiguated from
  artifacts on disk. Treat 97.5% as directional, not as a measured precision figure for the Koshkina
  reader specifically. On GSR wide broadcast we have the aggregate GS-HOTA signal and the zero-hurt
  distribution, but not a per-read GT audit of the 425 committed numbers. The +5.07 is the end-to-end
  metric effect, which is the number that counts; a per-read precision audit against GSR GT jersey
  labels is the natural next validator.
- **8.73% read coverage** is the binding constraint. The lift measures what a precision-first reader
  extracts from wide broadcast without roster priors; it is a floor on what a coverage-focused reader
  (higher-res crops, close-up anchors, roster-constrained decode) could add.
- **No roster mask** by design (GSR has no squad list). On our Brighton-ManU match the roster prior lifts the
  anchor funnel; that lever is unavailable here, so this is the harder, roster-free setting.

## Provenance

- Reader: `generator.jersey_id.KoshkinaRecognizer` (legibility ResNet34 `legibility_resnet34_soccer_20240215`,
  KeypointRCNN torso RoI, PARSeq `parseq_epoch=24-...-val_accuracy=95.60`), tracklet vote
  `aggregate_votes(min_conf=0.30)`, `max_crops=20/track`, no roster mask.
- Metric: official sn-trackeval 0.4.0 `trackeval.datasets.SoccerNetGS` + stock HOTA/Identity, Gaussian
  similarity (5 m tolerance), unchanged from the shipped benchmark. Non-jersey configs identical across
  arms (invariant held) -- the plumbing is confirmed sound.
- Artifacts: `results/gsr_benchmark/gsr_scores_koshkina.json` (this re-score; baseline
  `gsr_scores.json` untouched), submissions `outputs/gsr/eval_koshkina/`, vote checkpoints
  `outputs/gsr/koshkina_jersey/`.
