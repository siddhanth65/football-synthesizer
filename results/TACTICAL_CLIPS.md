# Tactical clip generator — annotated broadcast + tactics board, with uncertainty drawn

The analyst-breakdown deliverable: for one passage of play, a two-panel video with the broadcast on
the left (tracked, team-coloured, role-banded, sparsely named) and the real 105 × 68 m tactics board
on the right — where the players we **cannot see** are drawn as calibrated uncertainty rather than
as confident dots.

Code: `tools/tactical_clip.py`. Tests: `tests/test_tactical_clip.py` (24 checks) plus
`python tools/tactical_clip.py --self-check`. Outputs: `results/tactical_clips/`.

---

## 1. Rendered examples (2026-07-26)

Four passages, four different situation types, all 14 s at 25 fps / 1080p. Each ships an `.mp4` and
a six-frame contact sheet `_sheet.png`.

| file | fixture | clock | situation | tracking | imputation |
|---|---|---|---|---|---|
| `tottenham_manutd_h1_0343s.mp4` | Man Utd @ Tottenham | H1 05:43–05:57 | **Tottenham build-up (97% of possession frames) vs a Man Utd HIGH block, line 66 m** — the Carrick-video theme | 13.7 players/frame, ball 100%, geometry 92% | 1 124 imputed samples over 70 frames; up to 11 ghosts on screen at once |
| `fulham_manutd_h2_0636s.mp4` | Man Utd @ Fulham | H2 10:36–10:50 | Man Utd build-up (100%), opponent block **not measurable** | 15.4 players/frame, ball 100%, geometry 99% | 177 samples over 65 frames |
| `manutd_southampton_h2_2486s.mp4` | Southampton @ Man Utd | H2 41:26–41:40 | Man Utd progression (66%) vs a **mid** block | 14.1 players/frame, ball 100%, geometry 100% | 86 samples over 48 frames |
| `fulham_manutd_h1_1578s.mp4` | Man Utd @ Fulham | H1 26:18–26:32 | Man Utd build-up (100%) vs a **low** block — an attacking set piece; 18.1 players/frame is the best-tracked passage in the corpus | 18.1 players/frame, ball 100%, geometry 100% | 25 samples over 17 frames |

`tottenham_manutd_h1_0343s.mp4` is the one to watch first: it is the only passage in the shortlist
where the uncertainty panel carries real weight (11 simultaneous ghosts, horizons from 1 s to 10 s,
so the regions visibly grow across the clip).

Named players appear in `fulham_manutd_h2_0636s` (Joshua Zirkzee, followed on both panels).

---

## 2. Marker semantics — the visual argument

| marker | meaning | where it comes from |
|---|---|---|
| solid team-coloured disc | **OBSERVED** — detected this frame and projected through a homography fitted on this frame's own pitch↔image correspondences | `outputs/<match>/final/match_aligned.parquet`, gated at `calib_error_m ≤ 1.0 m` |
| hollow diamond inside two nested shaded ellipses | **IMPUTED** — the frozen B4 v1 model's point estimate with its calibrated **50%** (inner) and **90%** (outer) predictive regions. Tight at 1 s occlusion, a wide smear at 10 s | `synthesizer.imputation_v1.emit()`; `source ∈ {"v1", "anchor"}` |
| faded tilted cross + "? N s" | **ABSTAINED** — the model declined to assert (region wider than the frozen SGR threshold `R_MAX = 30.43 m`); the coordinate shown is the last sighting, never a prediction | `emit()` with `source == "abstained"`, `asserted=False` |
| solid white dot / amber ring | ball **detected** / ball **carried over** from a nearby frame | post-`link_ball` parquet's `observed` flag |
| yellow text | a player **name** | only where ≥ 3 close-up reads agree AND the name is on the match team sheet |
| white text (`GK/DEF/MID/ATT`) | a **role band**, the fallback whenever a name is not earned | median position along that team's own attack direction over the passage |

**Ellipse geometry.** The regions are axis-aligned ellipses with semi-axes `k[bucket] × w`, where
`w` is the quantile heads' per-sample half-width and `k` is the frozen per-horizon split-conformal
multiplier from `results/B4_MODEL_V1_runlog.md`:

```
k(50%) v1  = [0.523, 0.758, 0.756, 0.791, 0.853, 0.914]
k(90%) v1  = [1.335, 1.603, 1.583, 1.630, 1.816, 1.953]
k(50%) anc = [0.431, 0.827, 0.918, 0.985, 1.307, 1.639]
k(90%) anc = [1.351, 1.792, 1.864, 1.995, 2.699, 2.871]
```

over horizon buckets `0-1s / 1-3s / 3-5s / 5-10s / 10-30s / 30s+`. The adopted P2 defer-to-anchor
policy (`results/B4_ABSTENTION_POLICY.md`) emits the **anchor** in the 0-1 s bucket and **v1** in the
other five, each with its own multipliers, so the drawn region always belongs to the estimator that
actually produced the dot. Nothing is refitted at render time: the heads are loaded from
`data/imputation/cache/v1_heads.joblib` (a 583 s one-off CPU fit on Metrica Game 1).

**What the imputation panel is NOT.** The B4 v1 model was trained and gated on Metrica tracking
data. There is **no ground truth off camera in broadcast footage**, so applying it to our tracks is
a smell test, not a validation — `results/B4_TRANSFER_M3.md` says exactly this, and the sentence
"Imputed positions have no broadcast ground truth (smell test only)" is burned into every rendered
frame.

**Two filters applied before any ghost is drawn** (`Passage.ghosts_at`):

1. **Re-identification duplicates.** Our tracks fragment, so a "hidden" slot is often a live player
   under a new id. Any ghost within `DUP_M = 6 m` of an observed same-team player is dropped — we
   are not entitled to draw a second player where we can already see one.
2. **Squad arithmetic.** A team has eleven players, so if the broadcast shows eight we can be
   missing at most three. Ghosts per team are capped at that remainder, most-recently-seen first.
   Without this, fragmentation puts a crowd on the board that does not exist.

Ghosts hidden longer than `MAX_IMPUTE_S = 20 s` are never drawn at all.

---

## 3. Refusal conditions

The tool refuses to render rather than produce a plausible-looking frame it cannot support. Every
rate is measured against the samples the passage *should* have had, not the ones that survived — a
window half-eaten by a replay must not score on the quality of its surviving half.

| gate | threshold | message |
|---|---|---|
| frame coverage | `< 85%` of expected samples yield trusted geometry | "broken passage: only N% of its frames yield trusted geometry — replay, cut or calibration failure" |
| tracking density | `< 12` gated players/frame | "tracking too sparse: N players/frame" |
| ball coverage | `< 60%` post-`link_ball` samples | "ball not tracked enough: N% of frames" |
| both teams visible | `< 50%` of frames with ≥ 4 players per team | "only one team on screen" |

Inside an accepted passage, individual bad frames are still shown *as failures*: a frame with no
pipeline output gets `NO OUTPUT: BROADCAST CUT / REPLAY / GRAPHIC` and the tactics board is blanked
with `NO PITCH GEOMETRY`. Nothing is interpolated across a cut.

Two situation labels also refuse rather than guess:

* **Block class.** `generator.impute.line_estimates` de-biases the visible defensive line, and on a
  passage showing too little of the block that correction can land the line *behind its own goal*.
  A real read of **−7 m** turned up in `fulham_manutd` h2 10:36 and was originally printed as
  "OPP BLOCK: low (−7 m)". An out-of-range line is now reported as **"not measurable"** — it is a
  failed measurement, not a low block. (Block-height Gate 1 is still `pending` in
  `fingerprint/block_height.py`; even an in-range class is descriptive, not certified.)
* **Phase / possession.** Possession is a *proximity* proxy. In a crowded box the modal owner flips
  every few frames, and "Man Utd build-up" for a corner they are defending is worse than no label.
  When the modal team holds `< 60%` of the passage's possession frames the caption reads
  **"contested (possession flips)"** and no possessing team is named. The chip now always carries
  the share, e.g. `POSSESSION: Tottenham (97%)`.

---

## 4. Passage selection

`--shortlist` scans every registered, processed match in 14 s windows at 7 s stride and scores each
(all terms in [0, 1]):

| weight | term |
|---|---|
| 0.30 | mean gated players/frame, normalised at 18 |
| 0.25 | post-`link_ball` coverage |
| 0.20 | frame coverage (expected samples that yield trusted geometry) |
| 0.10 | at least one name-gated track on the pitch |
| 0.10 | **interest**: build-up from own third against a `mid`/`high` opponent block — the Carrick theme |
| 0.05 | both teams on screen |

Windows below the refusal thresholds are dropped before ranking, overlapping windows are collapsed
to their best member, and no single fixture contributes more than four rows. `--auto N` then renders
N passages chosen to cover **different `(phase, block)` situation types**, not the N highest scores.

### Shortlist, all 16 processed matches (2026-07-26)

Full CSV: `results/tactical_clips/all_shortlist.csv`.

| # | match | half | clock | window (s) | players/frame | ball | geom | named frames | possession | phase | opp block | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fulham_manutd | h2 | 10:36 | 636.9–650.9 | 15.4 | 100% | 99% | 70 | Man Utd | build-up | not measurable | 0.853 |
| 2 | manutd_southampton | h2 | 41:26 | 2486–2500 | 14.1 | 100% | 100% | 71 | Man Utd | progression | mid | 0.835 |
| 3 | tottenham_manutd | h1 | 05:43 | 343–357 | 13.7 | 100% | 92% | 0 | Tottenham | **build-up** | **high** | 0.806 |
| 4 | fulham_manutd | h1 | 26:18 | 1578–1592 | 18.1 | 100% | 100% | 0 | Man Utd | build-up | low | 0.800 |
| 5 | tottenham_manutd | h2 | 00:21 | 21–35 | 16.9 | 77% | 90% | 21 | Tottenham | progression | low | 0.800 |
| 6 | fulham_manutd | h2 | 18:25 | 1105.9–1119.9 | 12.3 | 100% | 97% | 28 | Fulham | progression | low | 0.798 |
| 7 | manutd_southampton | h1 | 34:33 | 2073–2087 | 13.1 | 100% | 92% | 0 | Southampton | **build-up** | mid | 0.793 |
| 8 | fulham_manutd | h2 | 29:20 | 1760.8–1774.8 | 18.6 | 62% | 92% | 54 | Man Utd | progression | high | 0.784 |
| 9 | tottenham_manutd | h1 | 17:49 | 1069–1083 | 12.7 | 100% | 86% | 0 | Tottenham | **build-up** | **high** | 0.771 |
| 10 | manutd_brighton | h2 | 52:15 | 3135–3149 | 16.7 | 100% | 96% | 0 | Brighton | attack | high | 0.768 |
| 11 | brighton_manutd | h2 | 08:03 | 483–497 | 14.9 | 70% | 97% | 0 | Brighton | **build-up** | **high** | 0.767 |
| 12 | manutd_brighton | h1 | 35:36 | 2136–2150 | 16.4 | 100% | 97% | 0 | Brighton | progression | low | 0.766 |
| 13 | france_senegal | h1 | 28:38 | 1718.5–1732.5 | 15.1 | 100% | 100% | 0 | Senegal | progression | low | 0.752 |
| 14 | france_senegal | h1 | 11:59 | 719.4–733.4 | 17.0 | 88% | 99% | 0 | Senegal | build-up | low | 0.750 |
| 15 | france_senegal | h1 | 21:32 | 1292.1–1306.1 | 15.0 | 100% | 100% | 0 | France | progression | low | 0.750 |
| 16 | tottenham_manutd | h1 | 41:24 | 2484–2498 | 16.3 | 100% | 92% | 0 | Tottenham | attack | mid | 0.748 |
| 17 | brighton_manutd | h1 | 31:24 | 1884–1898 | 12.9 | 82% | 93% | 0 | Brighton | **build-up** | **high** | 0.745 |
| 18 | france_senegal | h2 | 01:37 | 97.9–111.9 | 16.7 | 86% | 100% | 0 | France | progression | low | 0.744 |

Bold rows are the Carrick-video theme (build-up under a mid/high press). Five of the eighteen
qualify; four of those five are Tottenham or Brighton building against Man Utd.

Per-match candidate counts (windows clearing every refusal gate, capped at 4 each):
`france_iraq 4 · france_senegal 4 · france_norway 4 · mun_mci 0 · brighton_manutd 4 ·
manutd_liverpool 2 · manutd_fulham 1 · palace_manutd 3 · manutd_tottenham 3 ·
southampton_manutd 3 · liverpool_manutd 1 · manutd_brighton 4 · fulham_manutd 4 ·
manutd_palace 4 · manutd_southampton 4 · tottenham_manutd 4`.

`mun_mci` contributes nothing: it has no linked-ball artifacts, so no window can clear the ball gate.

---

## 5. How to run

```bash
# rank candidate passages across every processed match (~8 min CPU, renders nothing)
python tools/tactical_clip.py --match all --shortlist --top 18

# one match only
python tools/tactical_clip.py --match manutd_liverpool --shortlist

# render one passage (seconds into the half)
python tools/tactical_clip.py --match tottenham_manutd --half h1 --start 343 --end 357

# render the top N shortlisted passages, spread across situation types
python tools/tactical_clip.py --match all --auto 4

# skip the uncertainty panel (much faster; the legend then says "imputation off")
python tools/tactical_clip.py --match fulham_manutd --half h2 --start 636 --end 651 --no-impute

python tools/tactical_clip.py --self-check     # geometry / labelling seams, no artifacts needed
```

Cost: ~40–70 s wall per 14 s clip at 1080p on CPU (rendering, mp4v write, x264 re-encode). The
first run of any imputed clip pays a one-off 583 s CPU fit of the frozen v1 heads, cached
thereafter in `data/imputation/cache/`. GPU is never touched.

`--start` / `--end` are seconds into the named half, resolved through the true per-chunk video
durations (chunks are not all 600 s — `manutd_liverpool` h1 ends on a 540.1 s chunk), so the clock
on screen is the real one. A passage that would straddle a chunk cut is clipped to the first chunk.

---

## 6. Reused, not rebuilt

`tools/make_demo.py` already owned the drawing machinery, so this tool imports it rather than
copying: `TopDown` (the 105 × 68 board), `FrameState` / `_lerp_state` (sample interpolation),
`_pitch_polylines`, `text` / `shade` / `chip`, the team colours and the ffmpeg encode pattern.
Situation labels come from the shipped detectors — `fingerprint.block_height.block_frames`,
`generator.ball.assign_possession` (via that module), `fingerprint.structural_metrics`
attack directions — and events from `outputs/<match>/ledger.parquet`.

One upstream change was needed: `tools/imputation_b4_external.load_ours` now accepts an explicit
`chunks` list and additionally returns `slot_track` / `frame_map`, so a caller can build a bundle
for one passage's chunk and map imputed slots back to source track ids. Both changes are additive
and default-preserving; `tests/test_imputation_transfer.py` and `tests/test_imputation_v1.py` still
pass.

---

## 7. Known limits

* **No off-camera ground truth.** The uncertainty regions are calibrated on Metrica, not on
  broadcast. Their *coverage* (PICP 48.5 / 89.8 at nominal 50 / 90) is a Metrica-holdout number.
* **Block height is un-validated.** Gate 1 (hand-annotated line heights) is still pending; low /
  mid / high are descriptive.
* **Names are rare by design.** With the ≥ 3-anchor gate, `manutd_liverpool` yields 59 named track
  fragments across the whole match, present on 10–133 sampled frames per chunk. Most passages show
  no name at all, which is the correct outcome given 35% per-moment attribution precision
  (`results/CARRIER_CONSTRAINED_v2.md`).
* **Role bands are bands.** `GK / DEF / MID / ATT` is the third of the pitch a track occupied over
  the passage, measured toward its own team's attacking goal — not a formation position.
* **Possession is proximity.** See §3; the share is now always printed so the reader can discount it.
