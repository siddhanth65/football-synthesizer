# Claims audit -- adversarial re-derivation of headline numbers (2026-07-21)

Every number below was **recomputed from raw artifacts** (parquets, npz score files, JSON stats,
crop images), not read back out of the document that asserts it. Where an earlier verifier reached
a conclusion, that conclusion was re-tested independently; two of its findings are amended here.

## Verdict summary

| Verdict | Count | Notes |
|---|---:|---|
| VERIFIED | 41 | From the earlier pass. **Single-verifier, not panel-confirmed** -- see limitation below. |
| CONTRADICTED -- already fixed | 2 | A (STATUS BAS swap), B (CRIB players/frame category error). |
| CONTRADICTED -- confirmed by this pass | 3 | C-1 (geometry yield denominator), C-2 (goal FP count), C-3 (identity numerator provenance). |
| PARTIAL / prior verifier amended | 1 | C-4 (Koshkina 97.5% provenance) -- defect real, prior attribution not established. |
| **Total adjudicated** | **47** | |

**Limitation, stated honestly.** The audit was interrupted by quota mid-run. The 41 VERIFIED entries
were confirmed by a single verifier and have NOT been re-derived a second time. Two of the six
contradictions this pass re-tested had a partly wrong supporting story (C-3, C-4) even though the
core defect was real -- so a single-verifier VERIFIED should be read as "no defect found by one
pass", not "panel-confirmed". Re-derive before quoting anything load-bearing in a review.

---

# CONTRADICTED findings (most load-bearing first)

## C-1 -- "~37% usable-geometry yield, whole broadcast" -- CONFIRMED CONTRADICTED

**Claimed** (`docs/REVIEW_CRIB.md` line 73): `| ~37% / 80.5% | usable-geometry yield: whole
broadcast / live-wide conditional |`

**Recomputed** (`outputs/brighton_manutd/final/match_aligned.parquet` +
`outputs/brighton_manutd/live_play/*.json`, frames keyed on `(chunk, frame)`):

| Quantity | Count | Source |
|---|---:|---|
| Sampling-grid frames (every 5th native frame, 11 chunks) | 29,849 | sum of `grid` in the 11 live_play JSONs |
| Detection-frames (>= 1 detection persisted) | 21,880 | `h1/h2 match_dense.parquet`, unique (chunk, frame) |
| Frames yielding player pitch coordinates | 7,891 | `match_aligned.parquet`, `pitch_x` notna |
| ... of those, with team in {0,1} | 7,887 | same, matches `pose_carry_probe.md` exactly |
| live_wide grid frames | 9,647 | live_play JSONs |
| geometry AND live_wide | **7,766** | set intersection |

Derived yields:

- **/ live_wide: 7766 / 9647 = 80.50%** -- exact, reproduced to the decimal. The 80.5% half of the
  claim is sound.
- **/ detection-frames: 7891 / 21880 = 36.1%** -- this is the origin of "~37%".
- **/ whole sampling grid: 7891 / 29849 = 26.4%** (26.3% against the nominal 30,011 grid quoted in
  `results/live_play_probe.md`, which counts 162 tail grid points the JSONs do not persist).

**Adjudication: CONFIRMED.** The prior verifier is right. "~37%" is a **conditional-on-detection**
yield relabelled as a whole-broadcast yield. The excluded 7,969 frames (26.7% of the grid) are the
zero-detection `graphic` bucket -- which `results/live_play_probe.md` itself documents as
"close-ups, celebrations, crowd, set-piece scrambles the detector could not populate", i.e. real
broadcast time, not an artefact. Dropping them from the denominator inflates the yield by 9.7 pp.

**Which denominator is defensible?** Both, for different sentences -- but only if labelled:
- 26.4% is the only defensible "**whole broadcast**" number. The grid is a uniform 1-in-5 sample of
  every frame of the match; nothing is excluded.
- 36.1% is defensible as "**of frames where the detector saw at least one player**". It is the right
  denominator for the pose-borrowing ceiling argument (`results/pose_carry_probe.md`) because a
  zero-detection frame is un-addressable by any pose lever.
- 80.5% is defensible as "of live-wide frames" and is the strongest of the three. Note **7766 / 7891
  = 98.4% of all usable-geometry frames are live_wide** -- the geometry we have is almost entirely
  live wide-camera, which is the honest version of the "yield ~= live-wide fraction" story.

The narrative "yield ~= live-wide-play fraction" survives but shifts: on the whole grid live_wide is
32.3% and yield is 26.4% (ratio 0.82 = the 80.5% conditional). It is NOT an equality; the filter's
own 32.1% figure and the 26.4% yield are 5.7 pp apart.

**RECOMMENDATION -- correct the number.** In `docs/REVIEW_CRIB.md` line 73 replace with:
`| 26.4% / 36.1% / 80.5% | usable-geometry yield: whole broadcast / of detection-frames / live-wide
conditional |`. `docs/CAPABILITY_LEDGER.md` and `docs/CV_EXPLAINER.md` already say
"detection-frames" and need no change; `docs/BTP_DECEMBER_PLAN.md` line 26 says "our ~37% of"
without a denominator and should be scoped. Do NOT retract -- the measurement is fine, only the
label on the crib line is wrong.

---

## C-2 -- "13/13 real goals detected across 6 matches, 1 FP" -- CONFIRMED CONTRADICTED

**Claimed** (`STATUS.md` line 98, 2026-07-19 entry).

**Recomputed** by re-running the probe's own peak picker at the documented operating point.
Threshold and dedup taken from `tools/action_spot_probe.py` argparse defaults -- `--thresh 0.30`,
`--min-sep-s 30.0`; `analyze()` converts that to `min_sep = round(30.0 * 2.0 FPS) = 60` samples of
greedy NMS on the per-frame Goal-class (index 6) probability, per half, over
`results/action_spotting_probe/<match>/scores_{h1,h2}_chunk*.npz` concatenated in chunk order.
Oracle from `goal_oracle()` -> `outputs/oracle/sofascore/match_dicts_England_Premier_League_24_25.json`.

| Match | Oracle goals (h1/h2) | Peaks >= 0.30 | Real | FP |
|---|---|---:|---:|---:|
| brighton_manutd | 3 (1/2) | 5 | 3 | **2** |
| manutd_liverpool | 3 (2/1) | 3 | 3 | 0 |
| manutd_fulham | 1 (0/1) | 1 | 1 | 0 |
| palace_manutd | 0 (0/0) | 0 | 0 | 0 (negative control holds) |
| manutd_tottenham | 3 (1/2) | 4 | 3 | **1** |
| southampton_manutd | 3 (2/1) | 3 | 3 | 0 |
| **TOTAL** | **13** | **16** | **13** | **3** |

The three false peaks, with the timestamps and scores the earlier verifier named -- all three
reproduce exactly:

| Match | half | t_s | mm:ss | score |
|---|---|---:|---|---:|
| brighton_manutd | h1 | 1982.5 | 33:02 | 0.4382 |
| brighton_manutd | h2 | 1404.0 | 23:24 | 0.5811 |
| manutd_tottenham | h2 | 1692.5 | 28:12 | 0.3379 |

**Adjudication: CONFIRMED.** Recall is right: **13/13 real goals, 100%** -- every true goal appears,
and in every match the true goals are the highest-scoring peaks (0.75-0.96) with the FPs strictly
below (0.34-0.58). Precision is wrong: **13/16 = 81.3%**, three false positives, not one.

Two aggravating details:
- The two brighton FPs are not hidden repo-wide -- `results/action_spotting_probe.md` states
  "Goal | 5 | 3 ... 2 extra incl. a goal-replay" -- but the STATUS **six-match rollup** silently
  counts only the tottenham FP (which STATUS line 69 describes as dropped by a replay-window fix).
  The rollup sentence is the one a reviewer would quote.
- The "replay-window fix" that removes the tottenham FP is a **post-hoc** rule applied after seeing
  the peak. If it is applied to brighton too it must be applied uniformly and re-measured; as of the
  persisted npz files, at the documented operating point, there are 16 peaks.

**RECOMMENDATION -- correct the number.** STATUS should read: *"13/13 real goals detected across 6
matches (100% recall); 16 peaks at the documented 0.30 / 30 s operating point = 3 false positives,
precision 81.3%. The two brighton FPs are replay/celebration windows and were already disclosed in
results/action_spotting_probe.md."* If the replay-window filter is to be claimed as the operating
point, it must be implemented in `tools/action_spot_probe.py`, applied to all six matches, and the
FP count re-derived -- not asserted for one match.

---

## C-3 -- Identity precision "brighton 39/40, liverpool 40/40, tottenham 40/40" -- CONFIRMED (provenance), prior attribution AMENDED

This finding is about **where the numbers live and what they are attached to**, not about doubting
Sid's eyeball verdicts. Sid looked at the montages and reported what he saw; the defect is that the
project never wrote those verdicts down, so nothing but prose carries them.

**Claimed** (`STATUS.md` lines 13-14, 93, 144-145; banners in
`results/identity/NAMED_TRACKS_koshkina.md`, `..._manutd_liverpool_koshkina.md`,
`..._manutd_tottenham_koshkina.md`).

**Part 1 -- denominators verify, numerators are not persisted. CONFIRMED.**
Read all four `levers_stats.json`:

| File | match | picks per arm | `verdict` values |
|---|---|---:|---|
| `results/closeup_anchor_probe/koshkina/levers/levers_stats.json` | brighton_manutd | 40 (roster/agree2/agree3/both2/both3) | all `""` |
| `results/closeup_anchor_probe/koshkina_manutd_liverpool/levers/levers_stats.json` | manutd_liverpool | 40 | all `""` |
| `results/closeup_anchor_probe/koshkina_manutd_tottenham/levers/levers_stats.json` | manutd_tottenham | 40 | all `""` |
| `results/closeup_anchor_probe/levers/levers_stats.json` (torso arm) | brighton_manutd | 40 (agree3: 44) | all `""` |

The 40 denominators check out (`_finalize_levers(..., sample_n=40)`, stratified by predicted number
via `np.linspace`; the agree3 arm's 44 is the `keep_all_max=60` branch keeping every new anchor).
The montage IS the pick set -- `_montage([r["path"] for r in picks], ...)` and `spotcheck_picks` are
built from the same `picks` list, so tile <-> record correspondence is exact and a verdict *could*
have been stored. But `"verdict": ""` is a hardcoded literal at `tools/closeup_anchor_probe.py:1158`
and **no tool in the repo ever writes it back** (grep for `"verdict"`: two writers, both emitting
`""`; one reader, `tools/make_identity_demo.py`, which reads a different file).

**One important amendment: a 39/40 IS persisted -- but for a different reader.**
`results/closeup_anchor_probe/spotcheck_step3/verdicts.json` (mtime 2026-07-16 19:40) has fully
filled verdicts: **39 x "CORRECT (legible back, kit+OCR agree)" + 1 x "WRONG (melee front-crop, OCR
fired on ref FIFA badge)"**, `n_survivors: 226`. That is the torso-ResNet18 + kit + easyocr arm
(STATUS lines 326/404, "Yield: 226"). So the earlier verifier's blanket "no numerator is persisted
anywhere" is **too strong** -- one is, for the easyocr/torso arm. The three numbers in the STATUS
identity headline (koshkina arm, brighton/liverpool/tottenham) are the ones with no persisted
numerator. **Net: CONFIRMED for the claim under audit, with the blanket statement narrowed.**

**Part 2 -- the brighton miss is mis-sourced. CONFIRMED, prior attribution OVERTURNED.**
`results/identity/NAMED_TRACKS_koshkina.md` says the verdict was taken on
`results/closeup_anchor_probe/koshkina/levers/montage_new_both2.png` and that "the one miss was a
cut-off crop with no visible number read as 4".

The koshkina `both2` pick list contains **no** prediction of 4 (raw preds: 34, 10, 28, 20, 29, 3, 5,
6, 8, 7, 16, 17, 18, 22, 37, 41; masked preds `pred_m` identical modulo the roster mask; zero files
matching `*n04_*`). So the described miss cannot be in that montage. Confirmed.

Where does it come from? Two candidates, and the prior verifier picked the weaker one:

- *Prior verifier's answer:* the koshkina **roster** arm, file
  `roster_survivors/h2_chunk_005_f845_n04_c0.937.jpg`. That file does exist and IS in the roster
  arm's 40 picks with `pred_m = 4`. But I opened the crop: it shows a Brighton striped shirt with a
  **clearly legible 41** on the back. That is a truncation misread, not "a cut-off crop with no
  visible number".
- *Better match:* the **torso/easyocr** `both2` arm,
  `results/closeup_anchor_probe/levers/new/h1_chunk_003_f9420_n04_c0.902.jpg` (in that arm's 40
  picks, `pred = pred_m = 4`, mtime 2026-07-17 11:37 -- 7 h before the koshkina run). I opened it:
  a front-on crop of Casemiro (#18), torso cut off at the frame edge, **no number visible anywhere**,
  read as 4. That is the sentence, verbatim, in the wrong document.

**Adjudication: the defect is CONFIRMED; the prior verifier's file attribution is OVERTURNED.** The
"cut-off crop read as 4" miss belongs to the **torso/easyocr both2 montage**, and its description was
carried over into the Koshkina banner -- which is a same-arm-name, different-reader transcription
error, exactly the class of mistake as contradictions A and B.

**RECOMMENDATION -- mark unverifiable-as-stored, then make it storable.**
1. Add a `--verdict-file` (or a two-line `verdicts.json` sidecar per arm, same shape as
   `spotcheck_step3/verdicts.json`, which already works) so a human verdict lands on disk keyed to
   the exact pick record. Until then, annotate the three STATUS lines: *"human spot-check, verdict
   recorded in prose only -- the per-tile verdicts were not persisted (levers_stats.json ships
   `verdict: ""`)."*
2. Fix the miss description in `results/identity/NAMED_TRACKS_koshkina.md`: the koshkina both2
   sample contains no 4-read; the cut-off/no-number/read-as-4 crop is
   `results/closeup_anchor_probe/levers/new/h1_chunk_003_f9420_n04_c0.902.jpg` from the torso arm.
   Either restate what the actual koshkina miss was, or downgrade the brighton koshkina figure to
   "39/40 reported, tile not identifiable from artifacts".
3. The 40/40s for liverpool and tottenham are unfalsifiable from disk as they stand. Do not present
   them as measured precision in a review without re-running the spot-check with verdict capture.

---

## C-4 -- GSR rescore "86.13% tracklet accuracy and 97.5% sample precision" -- PARTIAL; prior verifier's attribution NOT established

**Claimed** (`results/gsr_benchmark/GSR_RESCORE_KOSHKINA.md`, Caveats): *"The Koshkina chain was
measured at 86.13% tracklet accuracy on SoccerNet jersey-2023 test and 97.5% sample precision on our
Brighton-ManU match's close-ups."*

**86.13%: VERIFIED as Koshkina.** `STATUS.md` line 352 records the full-split reproduction --
1043/1211 tracklets on the SoccerNet jersey-2023 test split under the upstream eval convention, with
two disclosed substitutions (torchvision KeypointRCNN for ViTPose; no Centroid-ReID outlier filter),
86.13% vs the paper's 87.45%. Drivers live outside the repo (`~/jersey-number-pipeline/`), so this
is not re-derivable in-repo, but the number is internally consistent and attributed to the right
chain.

**97.5%: ambiguous between two different measurements on two different readers.** There are exactly
two 39/40 brighton close-up verdicts in the project, and they coincidentally give the same 97.5%:

| # | Artifact | Reader | Persisted? | Stated miss | mtime |
|---|---|---|---|---|---|
| (a) | `results/closeup_anchor_probe/spotcheck_step3/verdicts.json` | torso ResNet18 (`outputs/jersey/jersey_torso_r224_acc417.pt`) + kit gate + easyocr agreement | **yes**, verdict strings filled | melee front-crop, OCR fired on ref FIFA badge | 2026-07-16 19:40 |
| (b) | `results/identity/NAMED_TRACKS_koshkina.md` banner, on `koshkina/levers/montage_new_both2.png` | Koshkina chain (legibility ResNet34 -> KeypointRCNN torso -> SoccerNet PARSeq) + kit + agreement gates | **no**, all `verdict: ""` | "cut-off crop, no visible number, read as 4" -- shown in C-3 to belong to the torso arm | 2026-07-17 18:54 |

`GSR_RESCORE_KOSHKINA.md` was written 2026-07-17 21:18, i.e. **after** (b), in a Koshkina-reader
context. The most likely referent is therefore (b), the Koshkina spot-check -- not (a).

**Adjudication: PARTIAL. The prior verifier's specific claim -- "the 97.5% was measured on the
ResNet18 torso recognizer" -- is NOT established and is probably wrong on referent.** Nothing in the
timeline or the document forces the ResNet18 reading; it only follows if you assume the *persisted*
39/40 is the only 39/40, which C-3 shows it is not. Calling this a "provenance conflation between
two readers" overstates what the artifacts support.

**But the underlying defect is real and is arguably worse than stated:** the sentence cites a number
that is **not resolvable from artifacts to a specific reader**, and the two candidate readers are
very far apart on the one benchmark where both were measured on the same data --
Koshkina 86.13% vs the torso checkpoint's **41.70%** tracklet accuracy (both on n = 1211,
`outputs/jersey/eval_torso_mc20.json`, `accuracy: 0.41701`). A reviewer who asks "which reader is
the 97.5%?" cannot be answered from disk. Two further scope caveats the sentence omits:
- Either way, 97.5% is the precision of **gated anchors** (kit-colour + OCR/roster agreement), not
  of a bare recognizer -- the gates do a large share of the work.
- Either way, it is a 40-tile sample on close-ups, cited inside a document about **wide-broadcast**
  GSR sequences where those close-up conditions do not hold. The document does say this; the number
  is still doing rhetorical work it cannot carry.

**RECOMMENDATION -- rescope the sentence.** Replace with: *"The Koshkina chain reproduces at 86.13%
tracklet accuracy on the SoccerNet jersey-2023 test split (1043/1211, two substitutions disclosed).
Its precision on our Brighton-ManU close-ups was spot-checked by eye at 39/40 on a 40-tile montage;
that verdict was not persisted per-tile, so treat it as a directional human check, not a stored
measurement. Neither number transfers to GSR wide framing, where the only precision evidence is the
aggregate GS-HOTA lift and the zero-hurt distribution."* Keep 86.13%; do not quote 97.5% as a
measurement until C-3's verdict capture is in place.

---

# ALREADY-FIXED contradictions (logged for completeness)

## A -- STATUS BAS pass-ratios for southampton / tottenham were transcribed swapped -- FIXED

`STATUS.md` (2026-07-20 BAS entry) had the two ratios the wrong way round. Truth from the oracle:
southampton_manutd (Sofascore 12436949) attempted 1091 -> 1087/1091 = **0.996**;
manutd_tottenham (12436995) attempted 1031 -> 1079/1031 = **1.047**.
`results/bas_validation.md` (lines 98, 118) was **always correct** -- the defect was in STATUS prose
only, and the aggregate (mean |dev| 2.9%, 5/6 within 5%) is unaffected. STATUS corrected and
annotated 2026-07-20.

## B -- REVIEW_CRIB "12.76 / 22 players visible" was a category error -- FIXED

12.76 is a **compactness value in metres** (`cv.tendencies.*.compactness.mean`,
`results/brighton_cv_vs_oracle.md` line 43, sibling value 11.43), relabelled as a player count with
an invented `/22` denominator. Recomputed from
`outputs/brighton_manutd/final/match_aligned.parquet`: mean players per trusted frame (team in
{0,1}, `pitch_x` notna, per-frame mean over the 7,887 qualifying frames) = **11.75**. I reproduced
this independently in this pass: 11.749968 over 7,887 frames. CRIB corrected to ~11.8/22.

---

# What Sid can say in a review

## Confirmed-safe headline numbers (recomputed this pass, from raw artifacts)

| Number | Exact statement that is safe |
|---|---|
| **80.5%** | "Of live wide-camera frames, 80.5% yield usable player geometry (7,766 of 9,647, brighton_manutd)." Reproduced to the decimal. |
| **26.4%** | "Over the whole broadcast -- a uniform 1-in-5 frame grid, 29,849 frames -- 26.4% yield usable geometry." |
| **36.1%** | "Of frames where the detector saw at least one player, 36.1% yield geometry." Always attach the denominator. |
| **98.4%** | "98.4% of all our usable-geometry frames come from live wide-camera play." (7,766 / 7,891.) |
| **11.75 / 22** | "A trusted frame carries ~11.8 of 22 players with a pitch position." |
| **13/13, 100% recall** | "Zero-shot E2E-Spot finds every real goal across 6 matches -- 13 of 13 -- and the palace 0-0 negative control fires zero peaks." |
| **81.3% precision** | "...at 16 peaks / 3 false positives on the documented 0.30 / 30 s operating point." Say this in the same breath as 13/13. |
| **86.13%** | "We reproduced the Koshkina jersey chain at 86.13% tracklet accuracy on SoccerNet jersey-2023 test (1043/1211), vs the paper's 87.45%, with two substitutions disclosed." |
| **41.70%** | Our own torso ResNet18 on the same 1211 tracklets -- useful as the honest contrast that motivated adopting Koshkina. |
| **+5.07 / +34%** | GS-HOTA 14.76 -> 19.83 on the 58-sequence valid split, 56 improved / 2 flat / 0 regressed. (Not re-derived this pass -- from the earlier single-verifier set.) |
| **2.9% mean \|dev\|** | 6-match BAS pass-count deviation from a single frozen threshold; per-match table in `results/bas_validation.md` (which was always correct). |

## DO NOT SAY

- **"~37% of the broadcast yields geometry."** Wrong denominator by 9.7 pp. Say 26.4% (broadcast) or
  36.1% (of detection-frames), never "~37%" bare.
- **"13/13 goals with 1 false positive."** It is 3 FPs at the documented operating point. Two
  brighton FPs (33:02 h1, 23:24 h2) are undisclosed in the STATUS rollup though disclosed in
  `results/action_spotting_probe.md`.
- **"Identity precision is 39/40 / 40/40 / 40/40 (measured)."** These are unpersisted eyeball
  verdicts -- the `verdict` fields in all three `levers_stats.json` are empty strings. Say "human
  spot-check, reported 39-40 of 40; per-tile verdicts not stored" or say nothing.
- **"The brighton miss was a cut-off crop read as 4."** Not in the koshkina both2 sample -- that crop
  is from the torso/easyocr arm.
- **"The Koshkina reader was measured at 97.5% precision on our close-ups."** Cannot be resolved from
  artifacts to a reader; two different 39/40s exist, on two readers 44 pp apart on the same test set.
- **"12.76 of 22 players visible."** Retracted -- 12.76 m is a compactness value. Use 11.8/22.
- **Southampton 1.047 / tottenham 0.996.** Swapped. Southampton 0.996, tottenham 1.047.
- **Any 41 VERIFIED number as "audited".** Single-verifier only; the audit was quota-interrupted
  before a second pass.

---

## Method notes

- Frame identity throughout is the `(chunk, frame)` pair; `chunk` strings in the aligned parquet
  match the live_play JSON basenames (`h1_chunk_000`), so the set intersections are exact.
- Goal peaks were re-derived by importing `find_peaks`, `load_half_timeline`, `class_index` and
  `analyze` from `tools/action_spot_probe.py` -- the probe's own code at its own argparse defaults,
  so no re-implementation risk. `min_sep = 60` samples at 2 FPS = the documented 30 s.
- Crops in C-3/C-4 were opened and read visually; no OCR was re-run.
- CPU only. No GPU touched, no pipeline module modified, no test suite run.
