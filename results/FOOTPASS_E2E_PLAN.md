# FOOTPASS VAL end-to-end from-pixels run — setup, frame-alignment proof, GPU plan

Date: 2026-07-29. Task 2 setup for the approved next step in `docs/ATTRIBUTION_RESEARCH_PLAN.md`
("Re-ranked build order", item 3). **This phase was CPU-only by instruction — no GPU inference was
run.** Every stage below that needs the GPU is listed, costed and left unstarted.

Goal: produce the honest number the official PCBAS baseline (Macro-F1 46.41) avoids. That baseline
consumes ground-truth game state; ours starts at pixels — detect → track → team → homography → ball
→ per-crop OCR → GTA connector → identity solver → carrier → `(team, jersey)` at each of the
**6,070 labelled VAL events**.

Code: `tools/footpass_prep.py` (probe/align/chunk/offsets/verify/lineup/extract/teammap),
`eval/footpass_score.py` (the metric), `tools/footpass_predict.py` (pipeline artifacts → prediction
table), `tests/test_footpass_score.py` (6 tests, green). Registry: three new entries in
`data/matches.yaml`. Raw prep results: `results/footpass/{probe,align,verify}.json`.

---

## 1. Extraction and disk

`data/footpass/raw/videos_fullHD_VAL.zip` (10.57 GiB, AES) extracted with 7-Zip, password passed on
the command line only — it appears in **no** repo file. All three games came out:

| file | bytes | frames | fps | size | duration |
|---|---|---|---|---|---|
| `game_18.mp4` | 3,669,542,565 | 149,650 | 25 | 1920x1080 | 5,986.0 s |
| `game_24.mp4` | 3,973,913,396 | 162,075 | 25 | 1920x1080 | 6,483.0 s |
| `game_47.mp4` | 3,750,346,755 | 152,950 | 25 | 1920x1080 | 6,118.0 s |

Total video 464,675 frames = **18,587 s = 5.16 h**.

**Disk.** Free space went 38 GB (before) → 28 GB (after extraction) → 5.7 GB (after chunking, which
duplicates the video) → **17 GB** after deleting the now-redundant full-match mp4s and the temporary
`val_tactical_data.h5`. Both are re-derivable in minutes from the zips that remain on disk. Current
footprint: `matches/footpass_game_*` = 10.6 GB (chunks), `data/footpass/raw` = 13.8 GB (zips),
`data/footpass/digest` = 100 MB. `git status` on `matches/`, `outputs/` and `data/footpass/` is
clean — everything is covered by the existing `*.mp4` / `outputs/` / `data/footpass/` ignore rules.

**Headroom warning for the GPU phase: 17 GB free is enough** (the pipeline's own artifacts are
~200 MB for all three games, plus an ~80 MB transient crop directory per chunk that `ocr_match`
deletes as it goes) **but it is not comfortable.** If anything else lands on the disk, drop
`data/footpass/raw/videos_fullHD_VAL.zip` (10.57 GiB) — the chunks are already cut and verified.

## 2. The frame-alignment proof — VERDICT: annotation frame k == mp4 frame k, exact to ±1 frame

The `data/footpass/README.md` schema note says `frame` is the "frame index within that half's
video". **That is wrong, and it matters.** The digests show H2's frame indices *continue* from H1
(game_18: H1 ends 75,307, H2 begins 75,525), so the index is global over one file that is H1 and H2
concatenated with half-time cut out. A harness built on the README's reading would have been off by
~75,000 frames on every H2 event — half the corpus.

Three independent checks, all CPU:

**(a) Direct ROI overlay — the decisive one.** The annotations carry `roi_x/y/w/h` in broadcast
pixels. At annotation frame 41,703 of `game_18_H1` the 15 GT boxes were drawn on the mp4's frame
41,703 (0-based `cv2` index): every box lands on a player, and the annotated shirt numbers match the
numbers legible on the kits (Napoli 7/20/21/22/77/81 in sky blue, Milan 9/14/19/23/31/33/80 in
red-black, the green GK as 16). Zero shift.

**(b) ROI occupancy over a shift grid, both halves, all three games.** Mean non-grass pixel fraction
inside the GT boxes, at 8 well-populated frames per game spread across H1 and H2:

| game | −2500 | −250 | −25 | −5 | −1 | **0** | +1 | +5 | +25 | +250 | +2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| game_18 | 0.299 | 0.189 | 0.088 | 0.247 | 0.360 | **0.352** | 0.324 | 0.224 | 0.115 | 0.219 | 0.416 |
| game_24 | 0.270 | 0.089 | 0.051 | 0.203 | 0.307 | **0.310** | 0.306 | 0.213 | 0.060 | 0.148 | 0.486 |
| game_47 | 0.225 | 0.062 | 0.084 | 0.248 | 0.395 | **0.371** | 0.329 | 0.196 | 0.098 | 0.069 | 0.103 |

Sharp plateau at {−1, 0, +1}, ~35% lower at ±5, ~75% lower at ±25. No per-half and no per-game
offset, and no drift between the start and end of a file (the samples span both halves). The ±2500
column is noise — a random distant frame sometimes lands on another crowded shot.

**(c) Camera-cut train correlation over the whole file.** ffmpeg's `scene` detector (threshold 0.25,
192x108) against the cuts implied by the annotations' per-frame visible-player count (`roi_*` is NaN
exactly when a player is off-screen, so the count steps at a cut). Zero-tolerance matching, offsets
searched −60…+60:

| game | best offset | exact matches | video cuts | annotation cuts | runner-up offset's matches |
|---|---|---|---|---|---|
| game_18 | **+1** | 181 | 365 | 500 | 4 |
| game_24 | **0** | 234 | 450 | 362 | 4 |
| game_47 | **+1** | 147 | 319 | 244 | 3 |

A single-frame-wide peak 45-58x above the runner-up. The ±1 disagreement between games is a detector
convention difference (whether the scene score fires on the first frame of the new shot or the last
of the old), not a data offset — check (a) settles the sign at 0.

**Conclusion used by the code:** `tools/footpass_predict.measured_frame_shift` returns **0** and
warns if `align.json` ever reports a cut offset outside ±1. The residual ±1 frame is 40 ms and is
absorbed by the pipeline's stride-5 sampling grid anyway.

## 3. Chunking and the global-frame bookkeeping

One ffmpeg stream-copy segmentation per game, cuts forced at every 600 s **and** at the half
boundary (midpoint of the unannotated gap between H1's last and H2's first annotated frame).
Segments keep their global index in the filename, so chunk keys are unique across halves:
`matches/footpass_game_18/h1/chunk_000.mp4 … h2/chunk_010.mp4`.

| game | chunks | half boundary (frame) | boundary chunk starts | annotation gap | frames lost | bad joins |
|---|---|---|---|---|---|---|
| game_18 | 11 | 75,416 | h2_chunk_006 @ 75,425 | 75,307…75,525 | **0** | none |
| game_24 | 12 | 78,205 | h2_chunk_006 @ 78,225 | 77,221…79,189 | **0** | none |
| game_47 | 12 | 73,946 | h2_chunk_005 @ 73,950 | 73,804…74,088 | **0** | none |

Every boundary cut lands **inside** the unannotated gap, so no chunk holds frames from both halves
and no annotated frame is orphaned. Sum of per-chunk container frame counts equals the source frame
count exactly for all three games, and every adjacent pair joins with a zero gap.

**One bug found and fixed here.** The first implementation derived offsets from ffmpeg's
`segment_list` start times. Two of the three files carry a container `start_time` of 0.022969 s, so
`round(start * 25)` lands **one frame late** from the second chunk onward — `verify` caught it as a
`-1` join on game_18 and game_47. Offsets are now cumulative container frame counts
(`footpass_prep.write_offsets`), which is exact by construction and cross-checked against the
source's own frame count. `outputs/footpass_game_*/chunk_offsets.json` carries the offsets, the
per-chunk frame counts and the accounting.

## 4. Registry, rosters, and the one bit that is measured rather than declared

Three entries added to `data/matches.yaml`: `footpass_game_18` / `_24` / `_47`, artifacts under
`outputs/footpass_game_N/final/`, ball under `.../final/ball` (so `run_align_ball` works unchanged).
`teams` is a placeholder pair — FOOTPASS is anonymised and **no club names were invented**, even
though the broadcasts themselves are not anonymised at all (game_18 = Napoli–Milan, Serie A;
game_24 = Nice–Lille, Ligue 1; game_47 = Real Madrid–Napoli, UCL, all legible from the scoreboard).

Rosters come from the annotations themselves — the distinct `(player_id, shirt_number, role_id)` per
half — written in the schema `tools/identity_match.py` already reads
(`outputs/identity/footpass_game_N_lineup_assign.parquet`):

| game | roster slots | available h1 | available h2 | goalkeepers |
|---|---|---|---|---|
| game_18 | 32 | 23 | 27 | 2 |
| game_24 | 31 | 22 | 30 | 2 |
| game_47 | 31 | 22 | 30 | 2 |

`role_id == 1` is the goalkeeper (verified: exactly two per half, one per team). A player's "name" is
`T<footpass_team>#<shirt>` — unique within a game, and exactly the `(team, jersey)` identity PCBAS
scores. Shirt numbers **overlap between the two teams** within a match (game_24 H1: both teams field
a 4, a 7 and a 9), so the team half of the tuple is load-bearing, not decoration.

**Caveat on the h2 roster.** "Available in h2" means "appears anywhere in the h2 annotations", which
is 27-30 slots for 11 on-pitch places. The solver's mutual-exclusion constraint is therefore weaker
in h2 than in h1 (22-23 slots). This is a real handicap on the second half and is not fixable from
the data as distributed — there are no substitution timestamps.

**The team-id bit.** Our kit anchor emits 0/1 (`dark_is_team0`); FOOTPASS uses 1/2. This is resolved
**twice, independently, from evidence**, never declared:

* before the solve — `footpass_prep --stage teammap` counts jersey reads from our own per-crop OCR
  that hit a number appearing on exactly one FOOTPASS roster, and rewrites the lineups if flipped.
  Uses the rosters (an input) and our OCR (our output); never the event labels.
* at scoring time — `footpass_score.resolve_team_map` picks the mapping whose *team* column agrees
  with more answered events, and **prints both arms' hit counts** so the choice is auditable and a
  near-tie (which would mean the kit anchor is not separating the teams at all) is visible.

## 5. The scorer

`eval/footpass_score.py`. PRIMARY metric = **attribution-given-event**: the event frames are given
(this is not action spotting), and the pipeline must name the acting player's `(team, jersey)`.
Coverage is over **all** events, precision over answered ones, and the dial is the identity solver's
posterior — the exact convention of `EVIDENCE_DENSITY_LAW.md` §3 (it reuses
`tools.evidence_sim.frontier` / `coverage_at` rather than re-deriving them), so the number is
directly comparable to the simulator's law and to `OCR_DENSIFICATION.md` §6.

Reported per split (all / actor-on-screen / actor-off-screen), per game and per action class:
`coverage@precision>=0.85`, `coverage@0.60`, full-coverage precision, and the whole frontier.

An event with no pipeline row within ±2 frames stays **unanswered and in the denominator** — the
honest treatment, and it is pinned by a test.

**The off-screen ceiling, re-measured on VAL specifically** (the corpus-wide figure in
`EVIDENCE_DENSITY_LAW.md` §1 is 0.8151):

| split | events | actor on screen at the labelled frame |
|---|---|---|
| **VAL, all three games** | **6,070** | **0.8250** |
| game_18 | 1,879 | 0.7584 |
| game_24 | 1,976 | 0.8811 |
| game_47 | 2,215 | 0.8316 |

**17.5% of VAL events have an off-screen actor. No pixel pipeline can exceed coverage 0.825 here**,
and game_18 caps at 0.758. Event count 6,070 reproduces `data/footpass/README.md` exactly. Action
class mix: 2,470 / 3,059 / 111 / 97 / 67 / 162 / 26 / 78 for classes 1-8 — classes 3-8 are thin
enough that per-class numbers will be noisy and should be read as diagnostics only.

Tests: `tests/test_footpass_score.py`, 6 green — a synthetic 8-event fixture that pins the whole
metric's arithmetic (coverage@0.85 = 3/8, coverage@0.60 = 6/8, full coverage 6/8 @ precision 4/6),
plus unanswered-events-still-cost-coverage, the empty-prediction case, the visibility-run reader,
and the alignment estimator recovering a known shift with a clean runner-up gap.

## 6. Pipeline adapter audit — what breaks on FOOTPASS input

Walked end to end. Findings, with the CPU-side fixes already made:

| # | Concern | Verdict | Action |
|---|---|---|---|
| 1 | **fps mismatch** (25 vs our matches') | **NON-ISSUE, measured.** Our PL chunks are already 1920x1080 @ 25 fps, 15,000 frames per 600 s chunk — identical to FOOTPASS. `sample_every=5` gives the same 5 Hz grid on both. | none |
| 2 | **Chunking expectations** | Fine, with one adaptation: our splitter assumes one file per half. FOOTPASS is one file for both. | `footpass_prep.chunk_game` cuts at the half boundary and keeps a global chunk index; offsets measured, not assumed (§3) |
| 3 | **Global-frame bookkeeping** | Did not exist — nothing in the repo ever needed a frame index across chunks. | `chunk_offsets.json` per game + `footpass_predict` maps GT global frame → (chunk, grid frame) |
| 4 | **Team-colour KMeans on unknown kits** | Low risk. Sky-blue vs red-black (g18), white vs red-black (g24), white vs navy (g47) — all well separated in lightness, so `dark_is_team0` is well defined. Smoke run produced both team ids. | none; verify from the `align` stage's colour report |
| 5 | **Homography on these stadiums** | **PASS, measured on CPU.** PnLCalib produced median reprojection error **0.15 m** (game_18) and **0.26 m** (game_47), with **0 frames rejected** by the 2.0 m gate. game_47 is a notably high/wide UCL camera and still calibrated cleanly. | none |
| 6 | **`pl_pilot_align` hardcodes Brighton/Man Utd** | Cosmetic only — the kit-name strings in its printout. `run_align_ball` already repoints every path via the registry. | none (misleading log line accepted) |
| 7 | **Identity chain needs `<match>_named_tracks_both2_prtreid.parquet`** | **Real blocker.** That artifact comes from the close-up *name-caption* anchor chain. FOOTPASS players are anonymised — there are no names to read, only numbers. | `tools/identity_match.py` already carries a `--percrop` reads seam (added concurrently by another worker); `main()` now skips the Stage-1 anchor baseline when the artifact is absent instead of crashing |
| 8 | **The solver threw away its posterior** | Real gap — a coverage-at-precision frontier needs a dial, and `solve_match` computed the MILP's assigned-identity probability and discarded it. | `solve_match` now returns `(names, posteriors, stats)` and writes a `confidence` column into `<match>_solver_names.parquet`. Additive; the one in-repo caller is updated |
| 9 | **Carrier identification is bound to Sofascore + hand labels** | The existing `carrier_attribution_probe` chain needs rosters with minute windows and BAS-detected events. Neither exists here — but neither is needed: PCBAS *gives* the event frames. | `tools/footpass_predict.py` reuses `event_ledger`'s carrier rule verbatim (Viterbi ball track + aligned detections, nearest player, abstain beyond 3.0 m) at the labelled frame. **No search window** — `event_ledger` biases its window before the BAS peak because that detector fires mid-flight; a window scanned for "the frame where somebody is closest to the ball" would be a knob fitted against the thing being graded |
| 10 | **Ball model** | `tracknetv2_v6.pth` present; `pl_pilot_ball` is registry-generic once repointed. Untested on Serie A/Ligue 1/UCL footage — the fine-tune was on PL broadcast. | GPU-side risk, flagged in §8 |
| 11 | **Roster/lineup format** | Built (§4). | done |
| 12 | **`data/footpass/README.md` says frames are per-half** | **Wrong** (§2). | corrected in this document; the README should be amended |

### 6.1 CPU smoke test (GPU hidden via `CUDA_VISIBLE_DEVICES=-1`, asserted before any model load)

The full `extract_positions` chain (football YOLO + ByteTrack + PnLCalib at `calib_period=25`,
`sample_every=5`), 10 s per window, on CPU, at windows the annotations say are wide tactical shots:

| | game_18 h1_chunk_002 | game_47 h1_chunk_001 |
|---|---|---|
| **tracked players per frame** | **6.5** (min 1, max 10) | **10.8** (min 5, max 18) |
| **GT visible players per frame, same frames** | **14.7** (12-17) | **16.4** (9-19) |
| **ratio** | **0.44** | **0.66** |
| median calibration error | **0.20 m** | **0.13 m** |
| frames rejected by the 2 m gate | 0 | 0 |
| rows carrying pitch coordinates | 0.45 | 0.83 |
| distinct track ids / ball rows | 18 / 7 | 33 / 18 |
| CPU wall clock | 413 s | 277 s |

**Calibration and detection pass; tracked coverage is the open question.** A detector-only control
(no tracker, no calibration; 20 frames at 1 s spacing in the same windows) settles which stage is
responsible:

| | game_18 | game_47 |
|---|---|---|
| detector boxes per frame (referees excluded) | **7.4** | **16.3** |
| GT visible per frame, same frames | 5.8 | 13.2 |
| **detector / GT-visible** | **1.29** | **1.23** |

The detector finds *more* people than FOOTPASS annotates as visible (it also catches officials near
the touchline and bench figures — an eyeballed overlay at game_18 frame 41,700 puts 21 clean boxes
on the frame, including the goalkeeper and the far-side group). **Detection is not the bottleneck.**
The gap between 1.25x at the detector and 0.44-0.66x after the chain is the **tracker**: at
`sample_every=5` the association step sees 200 ms between processed frames, and a large share of
detections never confirm into a track id.

For calibration: PnLCalib produced 0.13-0.20 m median reprojection error with **zero** frames
rejected by the 2.0 m gate, on a Serie A stadium and on game_47's notably high/wide UCL camera.
**Homography on these stadiums is a solved problem, not a risk.**

Caveat, stated plainly: two 10-second windows on CPU. This is enough to rank the risks, not enough
to quote a tracked-coverage figure for the corpus. **The cheapest high-value action before spending
19 GPU-hours is to extend exactly this comparison** — tracked players per frame against the
annotations' visible count, across a few hundred sampled frames per game. FOOTPASS is the first
corpus this project has had that can measure it at all.

## 7. GPU launch plan — DO NOT EXECUTE (orchestrator greenlights separately)

Throughputs are **measured on this machine** on `manutd_liverpool` (1080p/25 fps, 11 chunks,
6,600 s of video), from stage logs and artifact timestamps — not vendor numbers:

| stage | measured | per second of video |
|---|---|---|
| extract (YOLO-football + ByteTrack + PnLCalib@25, stride 5) | 1,222 s per 600 s chunk (8 timed chunks: 1069-1303 s) | **2.04 s/s** |
| ball (TrackNet v6 + carry-over + `link_ball`) | ~4 min per chunk | **0.40 s/s** |
| GTA per-detection PRTreID embeddings (stride 2) | ~2.6 min per chunk | **0.26 s/s** |
| per-crop jersey OCR @ 20 crops/track | ~10 min per chunk (1 h 46 m for 11, `OCR_DENSIFICATION.md` §8) | **1.00 s/s** |

Applied to 18,587 s of FOOTPASS video:

| stage | GPU hours | resume point |
|---|---|---|
| 1. extract, all 3 games, both halves | **10.5** | per-chunk `chunk_NNN_dense.parquet` (`run_batch --skip-existing`) |
| 2. ball (after align, CPU) | **2.1** | per-chunk `ball_<half>_chunk<NNN>.parquet` |
| 3. GTA embeddings | **1.3** | per-chunk `detemb_<chunk>.npz` |
| 4. per-crop OCR @ 20 crops | **5.2** | per-chunk `ocr_percrop/<chunk>.parquet` |
| **total** | **19.1 GPU-hours** | every stage resumable by disk state |

**Honest band: 19-26 GPU-hours.** The upside risks are (a) ±20% throughput variance across footage,
(b) game_47's very wide camera producing more small boxes, (c) any re-run of a stage. If the OCR
crop budget is raised 20 → 60 (the arm that measured d = 0.2083 at precision 0.858 on GSR), stage 4
goes 5.2 h → ~13 h and the total to ~27 h; on GSR that 2.5x crop spend bought only +35% density, so
**start at 20 and re-price after the first game's `d` is measured** — that is exactly the lesson
`OCR_DENSIFICATION.md` §9.3 records.

CPU stages, which do not consume the GPU budget: align (~10-20 min/game), identity solve
(HiGHS MILP, 60 s cap per chunk plus gallery similarities — **2-4 h CPU for 35 chunks**), predict
and score (minutes).

**Order, with the exact commands** (one GPU job at a time — never two, never with the full pytest
suite):

```
# --- GPU ---
python -m tools.footpass_prep --stage extract --games game_18            # ~3.4 h
# --- CPU ---
python -m tools.run_align_ball footpass_game_18                          # align (CPU) then...
# --- GPU (the same command continues into the ball stage) ---           # ~0.7 h
python -m tools.gta_match --match footpass_game_18 --build-cache --stride 2   # ~0.4 h
python -m tools.ocr_match --match footpass_game_18 --max-crops 20            # ~1.7 h
# --- CPU ---
python -m tools.footpass_prep --stage teammap --games game_18
python -m tools.identity_match --matches footpass_game_18 --percrop --floor 0.85
python -m tools.footpass_predict --games game_18
python -m eval.footpass_score --preds outputs/footpass/preds.parquet --games game_18
```

Then repeat for `game_24` and `game_47` and score all three together. Doing **game_18 first, end to
end, and scoring it alone** is the recommended order: it is the cheapest complete signal (6.2 GPU-h)
and it is the game with the worst off-screen ceiling (0.758), so a bad number there is informative
rather than fatal.

Transient disk during the GPU phase: ~200 MB of durable artifacts for all three games (dense
parquets ~10 MB/game, GTA caches ~45 MB/game, per-crop OCR ~4 MB/game, ball ~250 KB/game) plus an
~80 MB per-chunk crop directory that `ocr_match` deletes as it goes. Against 17 GB free this is not
a constraint; the constraint is the 10.6 GB of chunks already on disk.

## 8. Risks, ranked

0. **Tracked coverage, not detection, may be the binding ceiling.** §6.1 measures 0.44-0.66 tracked
   players per annotated-visible player on two wide-shot windows, against a detector that finds
   1.25x the annotated visible set. If that holds corpus-wide it is a multiplier on the carrier
   gate — the actor must be *tracked*, not merely detected — stacking on top of the 0.825 on-screen
   ceiling and putting the achievable coverage nearer 0.4-0.55 before naming is even attempted.
   It is also, unlike most items here, **cheap to measure on CPU before the GPU spend**, and if it
   is real the right response is a tracker/stride change, not 19 GPU-hours at the current settings.
1. **The ball model has never seen this footage.** `tracknetv2_v6` was fine-tuned on PL broadcast.
   Post-`link_ball` usable coverage on PL is 45.7%; on Serie A / Ligue 1 / UCL it is unmeasured. The
   carrier rule needs a ball position within ±1 s of the event, so **ball coverage multiplies
   directly into the coverage ceiling** — and the only number that counts is post-`link_ball`, never
   detection or projection counts (`CLAUDE.md`'s standing lesson). Mitigation: report the
   post-`link_ball` rate per game before reading any attribution number, and report the fraction of
   events for which the predictor found both a ball and a player (`ball_found` is persisted in the
   prediction table for exactly this).
2. **17.5% of events are unanswerable by construction.** Coverage cannot exceed 0.825 overall, 0.758
   on game_18. Any headline must be quoted against that ceiling, not against 1.0.
3. **h2 rosters are 27-30 slots wide**, weakening the solver's mutual exclusion in the second half.
   Expect an h1/h2 asymmetry and report the split.
4. **The frozen OCR rule was selected on GSR**, whose crops come from re-detected boxes; ours come
   from `estimate_player_box` on a foot point. `OCR_DENSIFICATION.md` §8 already flags that GSR
   precision does not automatically transfer. FOOTPASS lets us finally *measure* read precision on
   real broadcast, which no previous run could — that is a genuine bonus of this corpus.
5. **game_47's camera is much wider than our PL footage**, so torso crops will be smaller and read
   density likely lower than the 0.115 measured on `manutd_liverpool`. Expect the worst OCR arm
   there.
6. **The carrier gate is unconditioned.** The dial is the identity posterior only; a confidently
   named player who was not the actor scores as a confident error. That is correct for an end-to-end
   number, and it means the result is a product of gate x naming, as
   `CARRIER_CONSTRAINED_v2.md` factorised it (0.846 x 0.719 x 0.609 there).
7. **Concurrency.** `tools/identity_match.py` was being edited by another worker during this task;
   the changes here (posterior persistence, anchor-baseline skip) are additive and ruff-clean, but a
   merge conflict is possible if that worker holds an older copy.

## 9. What is done, and what the next phase inherits

Done (CPU): video extracted and verified; three registry entries; chunked with exact measured global
frame offsets and zero frame loss; rosters built; frame alignment proven three ways; scorer built
and tested; predictor written; two adapter fixes in `identity_match`; GPU plan costed from measured
throughputs.

Not done (needs the GPU window): stages 1-4 of §7 — 19.1 GPU-hours point estimate, 19-26 h honest.
Nothing else blocks the run.

---

# Appendix A — tracker-coverage pre-check (2026-07-29, CPU only, GPU untouched)

Approved follow-up to §6.1 / risk 0. Code: `tools/footpass_track_probe.py` (`--stage cache`, then
`--stage sweep`, plus `--stage selftest`); raw numbers `results/footpass/track_probe.json`.

**Method.** 18 windows — 3 per half per game, starts fixed at the 0.15/0.45/0.75 quantiles of each
half's annotated span, so shot type is whatever happens to be there and cannot be cherry-picked.
175 consecutive frames each; the detector runs **once per frame at stride 1** and every box (down to
confidence 0.05) is cached, so every stride and every ByteTrack setting is replayed on **identical
detections** — nothing in the comparison can move except the association. The first 25 frames of
each window are discarded as tracker warm-up, and every condition is scored on the **same stride-5
evaluation grid**, so a lower stride is credited only for the association it buys, never for having
more rows.

**Metric.** Not a count ratio. FOOTPASS ships a broadcast ROI box **and a player id** per visible
player, so this is a real recall: greedy IoU matching (hit at IoU >= 0.3) of GT visible boxes against
our boxes — "was the player the carrier gate would have to name actually covered by a track". The
player id also makes fragmentation measurable directly: distinct track ids handed to one annotated
player, and track ids covering two of them.

**Scale.** 4,149 GT player-boxes, 178 player-windows, both halves of all three games.

## A.1 The table

| condition | tracked recall | 95% CI (Wilson) | fragments / player / window | contaminated tracks | GPU cost vs shipped |
|---|---|---|---|---|---|
| **stride 5 (shipped)** | **0.4876** | 0.4724-0.5028 | 1.54 | 0.142 | 1.00x |
| stride 5, `lost_buffer=150` | 0.4876 | 0.4724-0.5028 | 1.54 | 0.142 | free |
| stride 5, `activation=0.10` | 0.4881 | 0.4729-0.5033 | 1.54 | 0.142 | free |
| stride 5, `det_conf=0.10` | 0.4881 | 0.4729-0.5033 | 1.54 | 0.142 | free |
| stride 5, `matching=0.9` | 0.5529 | 0.5377-0.5680 | 1.56 | 0.184 | free |
| stride 5, `min_hits=2` | 0.5797 | 0.5646-0.5946 | 1.86 | 0.153 | free |
| stride 5, `min_hits=1` | 0.6896 | 0.6753-0.7035 | 2.09 | 0.147 | free |
| **stride 5, `min_hits=1` + `matching=0.9`** | **0.7474** | 0.7340-0.7604 | **1.94** | **0.201** | **free** |
| stride 5, + also `det_conf=0.10` | 0.7484 | 0.7349-0.7613 | 1.95 | 0.201 | free |
| stride 2, default knobs | 0.8678 | 0.8526-0.8817 | 1.31 | 0.089 | 1.93x |
| **stride 2, `min_hits=1` + `matching=0.9`** | **0.9469** | 0.9364-0.9558 | **1.29** | **0.093** | **1.93x** |
| stride 1, default knobs | 0.9564 | 0.9497-0.9622 | 1.16 | 0.076 | 3.47x |
| stride 1, `min_hits=1` + `matching=0.9` | 0.9699 | 0.9642-0.9747 | 1.23 | 0.075 | 3.47x |

**Detector recall is 0.9759 in every arm** (0.9788 at `det_conf=0.10`) — it does not move, because it
cannot: the detector is not the bottleneck. Per game it is 0.997 / 0.981 / 0.947
(game_18 / game_24 / game_47).

Per-game tracked recall, shipped vs the recommended arm:

| game | GT boxes | shipped (stride 5) | stride 2 + knobs |
|---|---|---|---|
| game_18 | 679 | 0.5243 | 0.9882 |
| game_24 | 2,512 | 0.5478 | 0.9566 |
| game_47 | 958 | **0.3038** | 0.8937 |

game_47 — the very wide UCL camera — is by far the worst, exactly as §8 risk 5 predicted, and it is
where the fix matters most (0.30 -> 0.89).

## A.2 Diagnosis

**§6.1's 0.44-0.66 estimate is confirmed and lands at the low end: 0.4876 [0.4724-0.5028].** Half of
every player FOOTPASS marks as visible never receives a track id at the shipped settings, while the
detector finds 97.6% of them. The loss is entirely in association.

1. **The cause is the 200 ms gap, not any threshold.** `lost_track_buffer`,
   `track_activation_threshold` and the detection confidence floor move tracked recall by
   **<= 0.0005** — dead knobs here. The two that move it are the ones governing IoU association
   across the gap (`minimum_matching_threshold`) and how long a track must survive before it is
   emitted (`minimum_consecutive_frames`). At 5 Hz a running player displaces most of their own box
   between processed frames, IoU matching fails, and the 3-frame confirmation then discards the
   restart.

2. **The free knobs are not free.** `min_hits=1` + `matching=0.9` buys +0.259 recall at zero GPU
   cost, but buys it by **creating more tracks**: fragments per player per window 1.54 -> 1.94
   (+26%), contaminated tracks 0.142 -> 0.201 (+41%). `results/EVIDENCE_DENSITY_LAW.md` §5 prices
   fragmentation as a first-order lever (a ~7x reduction is worth about the same as 4x the OCR
   reads) and §4.2 shows contaminated evidence is worse than none, because a wrong read poisons the
   appearance gallery for that identity. This arm trades one lever for another rather than winning
   outright.

3. **Lowering the stride improves all three axes at once.** Stride 2 + knobs is a strict Pareto
   improvement over the shipped configuration: recall 0.4876 -> 0.9469, fragments 1.54 -> 1.29,
   contamination 0.142 -> 0.093. Stride 1 adds only +0.023 recall over stride 2 for another 1.8x of
   GPU.

Sanity note: the shipped arm's 14.2% contaminated tracks sits near the 10.96% of fragments
`GTA_LINK_STAGE1.md` measured as contaminated on GSR. The definitions differ (a track covering >= 2
annotated players over a 6 s window vs. their fragment definition), so this corroborates the order of
magnitude, not a like-for-like number.

## A.3 Revised GPU price

The extract stage does **not** scale cleanly with stride, because `cap.read()` runs on every source
frame regardless. Measured decode cost on this machine: **6.02 ms per source frame = 90 s of the
1,222 s per 600 s chunk (7.4%)**; the remaining 1,132 s is 377 ms per *sampled* frame. So
`extract(stride) = 90 s + sampled_frames x 0.377 s`, i.e. **2.39x at stride 2 and 4.70x at stride 1**
for that stage.

The ball stage also scales: `pl_pilot_ball` detects on **every dense frame**, so 2.5x more dense
frames at stride 2 costs 2.5x. The GTA embedding stage does **not** have to scale — raising its own
`--stride` from 2 to 5 keeps the crop count identical. Per-crop OCR does not scale (`max_crops` is
per track, and stride 2 produces slightly *fewer* tracks).

| stage | shipped (stride 5) | stride 2 | stride 1 |
|---|---|---|---|
| extract | 10.5 h | **25.1 h** | 49.4 h |
| ball | 2.1 h | **5.3 h** | 10.5 h |
| GTA embeddings (`--stride` 2 -> 5) | 1.3 h | 1.3 h | 1.3 h |
| per-crop OCR @ 20 crops | 5.2 h | 5.2 h | 5.2 h |
| **total** | **19.1 h** | **36.9 h** | **66.4 h** |
| tracked recall bought | 0.4876 (0.7474 with the free knobs) | **0.9469** | 0.9699 |

**Revised price for the recommended configuration** (stride 2, `minimum_consecutive_frames=1`,
`minimum_matching_threshold=0.9`, GTA `--stride 5`): **36.9 GPU-hours, +17.8 h over the shipped
plan; honest band 37-48 h** (same ±20% throughput variance as §7, plus the option of raising the OCR
crop budget). One unquantified saving in our favour: `calib_period` counts *sampled* frames, so it
should go 25 -> 62 at stride 2 to hold PnLCalib at one calibration per 5 s of video. Without a GPU
measurement I cannot say what share of the 377 ms is PnLCalib, so **that saving is real but not
priced in** — 36.9 h is an upper estimate for the extract stage, not a floor.

## A.4 Verdict — what is worth what

1. **Take the free knobs regardless** — `minimum_consecutive_frames=1`, `minimum_matching_threshold=0.9`
   in `generator/tracking.py`. +0.259 tracked recall for zero GPU. But ship them *with* a stride
   change, not instead of one: alone they raise fragmentation 26% and contamination 41%, and the
   measured law says that is a real cost to the identity channel, not a rounding error.
2. **Stride 5 -> 2 is the fix, and it is worth its price.** +17.8 GPU-hours (19.1 -> 36.9) converts
   tracked recall 0.488 -> 0.947 *and* improves fragmentation and contamination below the shipped
   baseline. Because the carrier gate needs the actor **tracked**, this moves the run's achievable
   coverage ceiling from ~0.40 (0.825 on-screen x 0.488 tracked) to ~**0.78** (0.825 x 0.947) before
   naming is attempted. Nothing else in this project buys a 2x coverage ceiling for 18 GPU-hours.
3. **Stride 1 is not worth it.** +29.5 h beyond stride 2 for +0.023 recall.
4. **Do not bother with** `lost_track_buffer`, `track_activation_threshold`, or a lower detection
   confidence floor. Measured null: <= 0.0005 in every case.

**The one check that could still halve this bill, and was not run (~20 min CPU on the same 18
windows).** The loss is a motion-compensation problem, and this repo already ships a
motion-compensated tracker: `BotSortBackend` with `gmc_method: sparseOptFlow`
(`generator/botsort_tuned.yaml`). `generator/tracking.py`'s own docstring records that BoT-SORT
fragmented *more* than ByteTrack on our PL footage — but that was at an effective 25 fps on a held
tactical camera, the regime where GMC has least to do, and is not evidence about a 5 Hz gap. If
BoT-SORT reaches stride-2-like recall at stride 5 it saves **17.8 GPU-hours**. It cannot be replayed
from the cached detections (Ultralytics couples tracking to the detector call), so it needs one more
CPU detector pass over the same windows. **Recommend running it before committing the 36.9 h.**

**Standing caveats.** 18 windows of 7 s each: long-lived tracks are under-represented, so
fragments-per-player is a within-condition comparison only and is *not* comparable to the law's
per-30 s figure. Recall is measured against FOOTPASS's *visible* set, so the 17.5% off-screen ceiling
of §5 still applies on top of every number here. And tracked recall is necessary, not sufficient — a
tracked player still has to be named.

**No code outside `tools/footpass_track_probe.py` was changed for this appendix.** In particular
`generator/tracking.py` still ships `min_hits=3` / default matching, and `tools/identity_match.py`
was not touched again (the other worker still holds it).

## A.5 BoT-SORT + sparseOptFlow GMC — the check A.4 asked for

Run 2026-07-29, CPU only, `tools/footpass_track_probe.py --stage botsort`; raw numbers
`results/footpass/track_probe_botsort.json`. **Same 18 windows, same GT boxes, same stride-5
evaluation grid, same warm-up rule, same metric** as A.1 — only the tracker changes. All 18 windows
were kept (no cut to 9 was needed): BoT-SORT only needs the detector on *sampled* frames, so a
stride-5 pass over the whole set costs 333 s and a stride-2 pass 842 s — far cheaper than the
stride-1 detection cache A.1 required.

Two implementation facts worth recording, because they affect how the table reads:

* **Ultralytics' `track()` replaces the result's boxes with the tracked ones**, so an untracked
  detection is invisible to this path and `n_det` is essentially `n_trk` by construction. The
  comparable detector recall is the **0.9759** the stride-1 cache measured with the identical model
  and thresholds; BoT-SORT at stride 5 is therefore discarding ~21% of real detections as untracked.
* A fresh detector is built per window so tracker state never leaks across windows, matching the
  ByteTrack sweep's per-window reset.

A **permissive** arm was also run (`new_track_thresh` 0.60 -> 0.25, `track_high_thresh` 0.50 -> 0.25,
`track_low_thresh` 0.10 -> 0.05) because `botsort_tuned.yaml`'s thresholds were fitted on held-camera
25 fps footage and could have produced a negative that was about thresholds rather than about GMC.
On game_18 it scored **identically** (0.7644 vs 0.7644, fragmentation slightly *worse* at 1.79 vs
1.67), so the thresholds are not the lever and the arm was dropped from the full run.

### The table (BoT-SORT rows added to A.1; 4,149 GT player-boxes throughout)

| condition | tracked recall | 95% CI | frags / player / window | contaminated tracks |
|---|---|---|---|---|
| ByteTrack stride 5 (shipped) | 0.4876 | 0.4724-0.5028 | 1.54 | 0.142 |
| ByteTrack stride 5, `min_hits=1` + `matching=0.9` | 0.7474 | 0.7340-0.7604 | 1.94 | 0.201 |
| **BoT-SORT tuned, stride 5** | **0.7674** | 0.7543-0.7800 | **1.65** | **0.122** |
| ByteTrack stride 2, default | 0.8678 | 0.8526-0.8817 | 1.31 | 0.089 |
| ByteTrack stride 2, `min_hits=1` + `matching=0.9` | 0.9469 | 0.9364-0.9558 | 1.29 | 0.093 |
| **BoT-SORT tuned, stride 2** | **0.9469** | 0.9364-0.9558 | **1.08** | **0.037** |
| ByteTrack stride 1, default | 0.9564 | 0.9497-0.9622 | 1.16 | 0.076 |
| ByteTrack stride 1, `min_hits=1` + `matching=0.9` | 0.9699 | 0.9642-0.9747 | 1.23 | 0.075 |

Per game:

| game | BoT-SORT stride 5 | BoT-SORT stride 2 | ByteTrack stride 2 + knobs |
|---|---|---|---|
| game_18 | 0.7644 | 0.9912 | 0.9882 |
| game_24 | 0.8272 | 0.9686 | 0.9566 |
| game_47 | **0.6127** | 0.8609 | 0.8937 |

### Verdict against the pre-declared rule

> *BoT-SORT@stride5 is adopted only if recall >= 0.852 (0.90 x the stride2+knobs figure) with
> fragmentation and contamination no worse than the shipped stride-5 arm.*

**REJECTED, on two of the three criteria.**

| criterion | bar | measured | |
|---|---|---|---|
| tracked recall | >= 0.8520 | **0.7674** [0.7543-0.7800] | **FAIL** — the CI does not come near the bar |
| fragments / player / window | <= 1.54 | **1.65** | **FAIL** |
| contaminated tracks | <= 0.142 | **0.122** | pass |

**The stride-2 ByteTrack plan stands.** GMC is not a substitute for the sampling rate: it recovers
about the same ground as the free ByteTrack knobs (0.767 vs 0.747 — a real but small edge, and the
CIs barely separate) and it does so at ~1.3x the per-sampled-frame cost rather than free. The 5 Hz
gap is a sampling problem, not a camera-motion problem, and the honest reading of the docstring
verdict this check was meant to re-examine is that it was **directionally right for the wrong
reason**: BoT-SORT is not worse than ByteTrack here, it simply cannot close a gap that GMC does not
address.

### The unanticipated result: at stride 2, BoT-SORT wins on the axis the law prices highest

Not part of the pre-declared rule, so it is reported as a finding and a decision for the
orchestrator, not adopted here.

At stride 2 the two trackers reach **exactly the same recall (0.9469, identical CI)**, but BoT-SORT
delivers it with **1.08 fragments per player per window against 1.29** (-16%) and **0.037
contaminated tracks against 0.093** (-60%). 1.08 is within touching distance of the ideal 1.0, and
these are the two quantities `EVIDENCE_DENSITY_LAW.md` prices most highly: §5 makes fragmentation
worth about as much as a 4x change in OCR read density at matched read volume, and §4.2 shows a
contaminated track poisons the appearance gallery of the identity it pollutes. It also beats
ByteTrack at **stride 1** on both (1.16 / 0.076) while costing half as many detector calls.

**Cost, priced honestly and with its uncertainty stated.** Measured on CPU, BoT-SORT costs
0.41-0.49 s per sampled frame against ByteTrack's ~0.344 s, i.e. **+0.07 to +0.14 s of mostly
CPU-side sparseOptFlow work per sampled frame**. That overhead does **not** shrink when the detector
moves to the GPU, so as a fraction it may be larger there. At stride 2 (7,500 sampled frames per
600 s chunk, 31 chunk-equivalents in the corpus) that is **+4.5 to +9.0 GPU-hours** on top of the
stride-2 extract stage. This band cannot be narrowed without a GPU measurement; **one chunk of
game_18 (~10 GPU-minutes) would collapse it to a number**, and that is the cheapest way to decide.

### A.6 Final priced recommendation

| configuration | tracked recall | frags/player | contam | GPU hours |
|---|---|---|---|---|
| shipped (ByteTrack stride 5) | 0.4876 | 1.54 | 0.142 | 19.1 |
| ByteTrack stride 5 + free knobs | 0.7474 | 1.94 | 0.201 | 19.1 |
| BoT-SORT stride 5 | 0.7674 | 1.65 | 0.122 | ~22-24 |
| **ByteTrack stride 2 + knobs (the recommendation)** | **0.9469** | 1.29 | 0.093 | **36.9** |
| ByteTrack stride 2 + knobs, BoT-SORT instead | 0.9469 | **1.08** | **0.037** | 41.4-45.9 |
| ByteTrack stride 1 + knobs | 0.9699 | 1.23 | 0.075 | 66.4 |

**Final price: 36.9 GPU-hours** for stride 2 + ByteTrack with `minimum_consecutive_frames=1`,
`minimum_matching_threshold=0.9` and GTA `--stride 5` — unchanged from A.3, because the BoT-SORT
check did not displace it. Honest band **37-48 h** (throughput variance plus the OCR crop-budget
option), and 36.9 h remains an upper estimate for the extract stage because the `calib_period`
25 -> 62 adjustment is real but unpriced.

**Open decision for the orchestrator, with a 10-GPU-minute way to settle it:** whether to pay an
estimated **+4.5 to +9.0 h (total 41.4-45.9 h)** to swap ByteTrack for BoT-SORT at stride 2 and buy
16% less fragmentation and 60% less contamination at identical recall. My read: on the measured law
that is a good trade — fragmentation is one of only two levers that move attribution coverage, and
this is the cheapest movement on it anyone has found in this project — **but the price band is 2x
wide, so measure the real GPU overhead on one chunk before committing.**

**What did not change.** No code outside `tools/footpass_track_probe.py` was touched;
`generator/tracking.py` still ships `min_hits=3`, `generator/botsort_tuned.yaml` is unmodified, and
`tools/identity_match.py` was not edited. No GPU was used: the BoT-SORT arms ran on CPU
(1,175 s total wall clock for both strides).

---

# Appendix B — the game_18 GPU run (live log; updated as stages complete)

Greenlit 2026-07-29. Scope: **game_18 only**, staged, one GPU job at a time. Games 24 and 47 are
NOT started — that decision returns to Sid with the number below.

Pre-run reconciliation: `tools/identity_match.py` holds both this worker's edits (posterior
persistence, anchor-baseline skip) and the other worker's (`--percrop` reads seam) — verified by
inspection, ruff clean, and **488/488 pytest green** with the GPU idle before launch.

## B.1 BoT-SORT GPU price — MEASURED, and the pick

`tools/footpass_track_probe.py --stage gpuprice`, `results/footpass/gpu_price.json`. One chunk of
game_18 (`h1_chunk_000`), stride 2, tracked stage only (detector + tracker; calibration and team
classification are identical between arms and would only dilute the difference), 690 timed sampled
frames per arm after a 60-frame warm-up, RTX 3050 Laptop 4 GB.

| arm | s per sampled frame |
|---|---|
| ByteTrack | **0.0355** |
| BoT-SORT (tuned yaml, sparseOptFlow GMC + ReID) | **0.0906** |
| **delta** | **+0.0551** |

Projected over the 3-game corpus at stride 2 (232,338 sampled frames): **+3.56 GPU-hours.**

> Pre-declared rule: adopt BoT-SORT if the projected 3-game overhead is <= +6 h.
> **3.56 <= 6 -> BoT-SORT ADOPTED.**

Two things worth recording. First, the CPU-derived band in A.5 (+4.5 to +9.0 h) was **too
pessimistic**: the real figure is below its lower bound, and the GPU measurement was worth the ten
minutes. Second, the *relative* multiplier went the way A.5 predicted and then some — BoT-SORT costs
**2.55x** ByteTrack per sampled frame on GPU against ~1.3x on CPU, because the sparseOptFlow GMC is
CPU-bound and does not shrink when the detector moves to the GPU. It is only affordable because the
GPU makes the detector ~10x cheaper in absolute terms (0.0355 s vs 0.344 s on CPU).

**Configuration locked for the run:** stride 2, `--tracker botsort` with
`generator/botsort_tuned.yaml` **exactly as shipped** (no edit), `calib_period` 25 -> 62 (holds
PnLCalib at one calibration per 5 s of video), GTA `--stride 5`, per-crop OCR at the 0.85-floor
frozen rule with 20 crops/track, connector `tau=0.040`, frozen solver config. Net shipped-module
edits required by this configuration: **none** — the A.4 ByteTrack knobs are not needed, because the
tracker they applied to is not the one being used.

## B.2 Extract (GPU) — in progress

Launched 15:48:40 with the locked configuration. First chunk (`h1_chunk_000`, 15,025 source frames)
completed in **1,897 s (31.6 min)**, projecting **~5.3 h** for game_18's 11 chunks — against the
~9.2 h A.6 implied for this game. The difference is the `calib_period` 25 -> 62 adjustment A.3
flagged as "real but not priced in": it is real, and it is worth roughly a third of the extract bill.

First-chunk health (`outputs/footpass_game_18/h1/match/chunk_000_dense.parquet`):

| quantity | value |
|---|---|
| sampled frames with rows | 6,147 |
| players per frame | **13.1** (vs 6.5 at stride 5 in §6.1) |
| distinct track ids | 1,222 |
| rows carrying pitch coordinates | **0.759** (vs 0.45 at stride 5) |
| calibration error, median / p90 | **0.178 m / 0.429 m** |
| frames inside the 2 m gate | 0.808 |
| both teams present | yes (42,886 / 37,176 rows) |

One log line needs reading correctly: `batch_match` prints the *mean* calibration error, which for
this chunk is 8.1e6 m. That is not a broken calibration — it is a handful of frames whose homography
blew up (max 3.25e10) and which the 2 m gate has already rejected, dragging an unguarded mean. The
median (0.178 m) and the 80.8% gate-pass rate are the honest numbers.

The detector's own ball is nearly absent (66 rows in 6,147 frames), which is expected — the ball
track comes from the TrackNet v6 stage, not from the person detector. It does mean the ball stage is
load-bearing for the carrier rule, as §8 risk 1 says.

**B.2 COMPLETE.** All 11 chunks, both halves, 15:48:40 -> 21:59:00 = **6 h 10 m** wall.
Against the ~9.2 h A.6 implied for this game: **33% under**. Per-chunk 1,897-2,575 s (h2 ran
~20% slower than h1). Totals: **719,280 dense rows over 59,593 sampled frames**, players/frame
12.8-16.4 per chunk, pitch-coordinate rate **0.768**, calibration median **0.169 m** with **88.0%**
of frames inside the 2 m gate. Zero chunks failed. (The session that launched it died; the job
survived and finished on its own, and the per-chunk artifacts made the resume free.)

## B.3 Align (CPU) — COMPLETE

`tools/run_align_ball footpass_game_18`. Kit-colour anchoring separated the two teams cleanly
(mean torso hue **35** for team 0 vs **78** for team 1; the printed club names are
`pl_pilot_align`'s hardcoded Brighton/Man Utd strings and mean nothing here — the real kits are
Milan red-black and Napoli sky-blue).

**The attack-direction check passes perfectly and is the first end-to-end validation of the chunking
work in §3:** team 0 attacks `-1` in **all six** h1 chunks and `+1` in **all five** h2 chunks. That
simultaneously confirms the team anchor is stable across the whole match, the half boundary was cut
in the right place, and the homographies are consistently oriented.

## B.4 Ball (GPU) — COMPLETE

TrackNet v6 + camera-continuous carry-over + `link_ball`, 11 chunks, ~1 h 50 m wall (vs the 1.7 h
A.6 implied for this game). **The only number that counts is the post-`link_ball` usable track**, per
`CLAUDE.md`'s standing rule, and it is better here than on our own broadcast:

| | post-link coverage, carry-over OFF | ON | delta |
|---|---|---|---|
| **footpass_game_18 (11 chunks)** | 44.3% | **57.3%** | +13.0 pp |
| *manutd_liverpool, for scale* | 25.1% | 45.7% | +20.6 pp |

Per-chunk ON coverage ranges 44.7%-66.5%; the weakest is `h2_chunk007` (44.7%). §8 risk 1 asked
whether a PL-fine-tuned ball model would transfer to Serie A/Ligue 1/UCL footage — on this game it
transfers **better than it does at home**, and the ~11 pp improvement over `manutd_liverpool` is at
least partly the stride-2 sampling giving `link_ball` a denser sequence to link.

This caps the carrier gate: an event whose instant has no ball position within +-1 s cannot be
attributed at all.

## B.5 GTA per-detection embeddings (GPU) — COMPLETE

`tools/gta_match --build-cache --stride 5` (the stride-5 setting that holds the crop count identical
to what stride-5 extraction + GTA-stride-2 would have produced, per A.3). 11 chunks, **~40 min**
wall (~200 s/chunk) against the 0.42 h A.6 implied — on estimate. 809-1,149 tracks per chunk,
12.5k-16.3k crops per chunk, **99 MB** of cache.

## B.6 Per-crop jersey OCR (GPU) — COMPLETE

`tools/ocr_match --match footpass_game_18 --max-crops 20`, 13:50:17 → 15:57:00 = **2 h 07 m** wall
(11 chunks, ~11.5 min/chunk) against the ~1.7 h A.6 implied — **25% over**. 166,930 crops over
11,575 player/GK tracklets.

**The stage's own output is the run's biggest negative: read density `d = 0.0304`**, against the
0.105-0.126 our own three matches produce (`OCR_REALMATCH.md` §2). 352 tracklets read of 11,575.
It is not a crop-budget problem — game_18 got 43% more crops than manutd_brighton (77% more than
manutd_liverpool) and produced 352 reads against their 1,174 and 960, i.e. 70% and 63% *fewer*.
Crop-level: legibility >= 0.5 on **9.61%** of crops vs 22.20% on manutd_liverpool;
a confident digit (`p_number >= 0.99`) on **3.54%** vs 13.50%. The frozen 0.85-floor rule holds its
shape on Serie A broadcast and simply fires ~4x less often. 0 of 163 goalkeeper-role tracklets read.

## B.7 The CPU tail (teammap → solve → predict → score) — COMPLETE

| step | wall | result |
|---|---|---|
| `footpass_prep --stage teammap` | 14 s | votes **+112 / -54** → FOOTPASS 1 = our 1 |
| `identity_match --percrop --floor 0.85` | **95 s** | 9,163 of 11,575 tracks named; 133 of 163 GK-role; both keeper slots (T1#1, T2#16) |
| `footpass_predict --arm solver` | 23 s | 832 of 1,879 events named |
| `footpass_predict --arm greedy` | 54 s | 282 direct reads → 1,537 named tracks; 243 events named |
| `footpass_score` x2 | ~15 s | see B.8 |

The solve's 95 s against the plan's "2-4 h CPU for 35 chunks" is a **symptom of B.6**, not a win: with
only 352 read tracklets the galleries are tiny and the 6-candidate prune leaves the MILP almost
nothing to decide.

**One bug fixed before the run.** `footpass_predict` mapped the event frame onto the grid with
`event_ledger.bas_to_grid_frame` (rounds to `STEP = 5`), but this run extracted at **stride 2** per
A.6, so it evaluated the carrier up to 2 frames further from the label than necessary. Now it hands
the label's own frame to `_nearest_row`, which snaps to whatever frame actually exists — stride
agnostic. Median frame offset used at scoring: **0**.

## B.8 THE NUMBER — full write-up in `results/FOOTPASS_GAME18_SCORE.md`

1,879 labelled events, attribution-given-event, coverage over ALL events:

| arm | cov@precision>=0.85 | cov@0.60 | full coverage | at precision |
|---|---|---|---|---|
| **Stage-2 solver** (frozen config, per-crop evidence, MILP posterior as the dial) | **0.0005** | **0.1575** | 0.4428 | 0.3173 |
| **Stage-1 greedy** (connector + unanimous propagation) | **0.0000** | **0.1293** | 0.1293 | 0.7119 |

**Neither arm clears the 0.85 floor.** The solver's curve peaks at precision 0.8169 (coverage 0.0378)
and 0.8000 (coverage 0.0905) — missed by 0.03-0.05 of precision, not by an order of magnitude. The
greedy arm has no dial (all confidences 1.0) and is a single point.

The gate factorisation says where it went:

| stage | rate over 1,879 events |
|---|---|
| actor on screen (annotation) | 0.7584 |
| ball AND players at the nearest sampled frame | 0.8616 |
| carrier found within 3.0 m | **0.6562** |
| carrier named — solver / greedy | 0.4428 / 0.1293 |
| team correct \| answered — solver / greedy | 0.7800 / 0.8519 |
| shirt correct \| answered — solver / greedy | 0.3197 / 0.7119 |

**The carrier gate is at 95.7% of its structural ceiling** (0.7584 on-screen x 0.9046 actor-covered
= 0.686 available, 0.6562 delivered). Detection, tracking, calibration, ball and the carrier rule
give up 4.3% between them. **Every remaining loss is the identity channel**, and B.6 says why.

Two results that revise this document:

* **Risk 0 (tracked coverage) is retired.** Measured on the real full-match extraction over 3,000
  sampled frames / 35,688 GT boxes: IoU>=0.3 tracked recall **0.8855**, and **0.9046** for the actor
  at the 1,425 on-screen event frames. Untracked rows are 0.1% of all rows. The stride-5 association
  collapse A.1 measured (0.4876) is gone. It is 0.10 under A.5's 0.9882 prediction, but ours goes
  through a foot-point box reconstruction and covers close-ups the probe's 18 windows did not, so it
  is a floor rather than a refutation.
* **§5's off-screen ceiling is SOFT, not hard.** 33 of 149 off-screen answers are correct
  (precision 0.2215) against a 0.0940 modal-guess control — 2.4x chance. `roi = NaN` annotates
  visibility, not absence, and the predictor reads a sampled frame up to 1 away.

The team bit resolved twice and agreed (teammap votes +112/-54; scorer hits 649 flipped vs 183
identity), so the kit anchor is genuinely separating Milan from Napoli.

## B.9 What this leaves

**Do not spend GPU on games 24 and 47 at these settings.** The binding constraint is measured and it
is not tracking, not calibration, not the ball and not the solver: it is **read density on this
footage, 0.0304 against the law's d\* = 0.347** — 8.8% of what coverage 0.50 at precision 0.85
requires. Raising `max_crops` 20 → 60 attacks the crop budget, which is not the failing term (78%
more crops already bought 70% fewer reads than manutd). The open levers, in the order the law prices
them, are the legibility/STR stage itself on non-PL broadcast, and fragmentation (9,833 → 4,736 under
the connector).

game_18's total spend for this number: **10 h 26 m GPU** (extract 6 h 10 m, ball 1 h 28 m, GTA
~41 m, OCR 2 h 07 m) — 12% under the ~11.9 h A.6 implied for this game — plus ~4 min of CPU tail.
