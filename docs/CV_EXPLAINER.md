# The CV pipeline, explained — from broadcast video to a grounded report

**Audience:** Sid (and anyone reviewing the BTP). You know high-level ML; this explains what
*actually happens* to a match video inside this repo, stage by stage, in plain English, with the
honest failure modes and where each piece lives in the code. Companion docs:
`docs/CAPABILITY_LEDGER.md` (what's validated), `docs/PL_PIVOT_PLAN.md` (roadmap),
`STATUS.md` (log). Sourced from a full code walk on 2026-07-14.

---

## The one-paragraph version

We slice a broadcast match into ~10-minute chunks, run a football-trained YOLO detector on every
5th frame to find players, stitch detections into per-player tracks with ByteTrack, split the 22
players into two teams by jersey *colour* (not numbers — we don't know who anyone is), fit the TV
camera with a neural pitch-line detector (PnLCalib) to get a **homography** (a 3×3 matrix mapping
screen pixels → metres on a 105×68 pitch), project every player's feet through it, detect the ball
with a fine-tuned TrackNetV2 heatmap network, chain the ball dots into one physically plausible
trajectory, decide who "has" the ball by smoothed nearest-player logic, and compute all tactical
metrics only on frames whose geometry survived every sanity gate. The result per match is a JSON
**fact store** of 100–300 numbers, from which a report is generated whose every number is checked
by a guardrail against the fact store — nothing a language model invents can reach the reader.

## The data spine

```
match.mp4
  └─ tools/chunk_video.py            → matches/<id>/<half>/chunk_NNN.mp4   (ffmpeg, no re-encode)
       └─ tools/batch_match.py       → <chunk>_dense.parquet               (players: box, track_id, team, pitch_x/y)
            └─ align (team_anchor)   → match_aligned.parquet               (chunk-consistent team labels)
                 └─ ball stage       → ball/ball_<chunk>.parquet           (linked ball trajectory + possession)
                      └─ report/facts.py → outputs/facts/<match>.json      (the fact store)
                           └─ report/report_v2.py → results/report_v2_<match>.{md,html}
```

Everything is driven by one registry: `data/matches.yaml` + `core/registry.py`. Adding a match is
one YAML entry. `core/pitch.py` holds the pitch constants and `METRICS_VERSION`, stamped into every
artifact so stale numbers are detectable.

---

## Stage by stage

### 1. Chunking (`tools/chunk_video.py`)

The full match mp4 is cut into ~10-minute chunks with ffmpeg **stream copy** — no re-encoding, so
it's fast and lossless; cuts snap to keyframes, which is fine because every chunk is analysed
independently. Chunks keep their **native frame rate** — our corpus mixes 25 fps (PL, Iraq, Norway)
and 59.94 fps (Senegal). That heterogeneity caused a whole class of bugs (a hardcoded `FPS=50`
survived for weeks); the rule now is that *all* time-based logic converts seconds → frames using the
chunk's true fps (`Match.chunk_fps()`).

**Why chunks?** A 4 GB GPU can process one at a time, and a crashed run resumes at the chunk
boundary instead of restarting the match.

### 2. Player detection (`generator/extract.py`)

Every **5th frame** goes through a YOLOv8 model trained on football broadcast imagery (HuggingFace
`uisikdag/yolo-v8-football-players-detection`). Being football-trained matters: it has separate
classes for *player / goalkeeper / referee / ball*, so referees are excluded rather than counted as
players. Confidence thresholds: 0.20 for players, 0.10 for the ball (deliberately lax — the ball is
tiny and fast).

**Known blind spot (matters for the jersey work):** this detector was trained on wide tactical
shots, so on close-up/replay frames it barely sees anyone — box heights cap around 157 px, while a
generic COCO YOLO sees 1000+ px people in the same frames. Harmless for geometry (close-ups have no
tactical view anyway), fatal if you try to read jersey numbers with it. Any Layer-2 identity work
must use a general detector on close-up content — we learned this the hard way in the jersey probe.

### 3. Tracking (`generator/tracking.py`)

ByteTrack associates boxes frame-to-frame (by predicted position overlap) into persistent
`track_id`s: a player keeps one ID as long as the camera holds them. Two tuning choices: a track
needs 3 consecutive frames to be born (kills flicker), and a lost track survives 60 frames before
being killed (survives brief occlusions).

**Two facts to internalize:** (a) IDs are **chunk-local and camera-shot-local** — a camera cut kills
essentially every track (we later *exploit* this as a free camera-cut detector); (b) a `track_id`
is *not* a person. It's "the same jersey blob the tracker managed to follow for a while". Cross-referencing
IDs across chunks without also keying on the chunk is a bug — exactly this bug lived in the
attacker/C3 code until 2026-07-15 (found by an external audit, confirmed, fixed with chunk-aware
grouping; the fix restored the receiver head from near-random to top3 ~0.9).

### 4. Team assignment (`generator/teams.py`, `generator/team_anchor.py`)

For each player crop we take the torso window, mask out grass (green in HSV) and dark pixels, and
summarise the remaining jersey pixels as a median **CIELAB colour** (a colour space where distance ≈
perceived difference, and brightness is a separate axis — so shadows don't flip a player's team).
KMeans with k=2 splits the match's crops into two kits. A second clustering layer
(`team_anchor.py`) makes the labels *consistent across chunks and halves* — cluster within each
chunk, then cluster the chunk-centroids, and anchor "team 0 = darker kit" so the mapping can't flip
at half-time.

**What we do NOT have:** player identity. No jersey numbers, no names. The names in the France
reports come from a curated roster file and are attached to prose, never to tracks. Closing this gap
is Layer 2 of the roadmap (jersey-number recognition, probed viable via close-ups).

### 5. Camera calibration (`generator/calibrate.py`, `generator/temporal_calib.py`)

The hardest stage. **PnLCalib** (a 2024 SOTA method) runs two HRNet neural networks over the frame —
one detects pitch *keypoints* (line intersections, penalty spots), one detects pitch *lines* — then
fits a full 3D camera model by "heuristic voting" over those correspondences, from which we derive
the ground-plane **homography**: the 3×3 matrix that maps any pixel on the grass to metres on the
105×68 pitch.

Because HRNet inference is expensive, `temporal_calib.py` only runs it on the first frame of each
camera shot, every 25th processed frame, or when phase-correlation detects camera drift — otherwise
it **reuses the last accepted pose** (a broadcast camera barely moves between nearby frames). It
never reuses a *rejected* pose, and resets at cuts.

**Two separate quality gates — don't confuse them (our own docs sometimes have):**

- **Fit gate** (`calibrate.py`): the homography must be built from **≥4 keypoint correspondences**
  with mean reprojection error **≤2 m**. 91.8% of detection-frames pass, and the passing fits are
  excellent (median error 0.21 m).
- **Plausibility gate** (`generator/postprocess.py::reject_implausible_frames`): a homography can
  fit the visible keypoints perfectly and still be **globally wrong** — e.g. a midfield shot showing
  only the halfway line and centre circle is geometrically ambiguous, and the "best fit" may project
  every player into a corner or off the pitch. So a frame's projections are only *trusted* if ≥8
  players land on the pitch spanning ≥25 m. Fail → the frame's coordinates are NaN'd.

The famous **~37% usable-geometry yield** is the product of both gates, and it is *not a bug*:
90.3% of the frames that fail are close-ups/replays with a median of 4 visible players — there is no
tactical view in them to reconstruct. ~37% ≈ the fraction of a broadcast that actually shows live
wide-camera play. Every downstream metric is computed only on those trusted frames.

*(Note: the "≥6 correspondences" figure that appears throughout STATUS is a different, later gate —
it belongs to the ball-projection path, stage 8, which re-fits its own homography from ≥6 player
correspondences. The pitch-calibration gate is ≥4 keypoints.)*

### 6. Projection to pitch coordinates (`generator/extract.py`, `generator/postprocess.py`, `generator/to_frames.py`)

Each player's **foot point** (bottom-centre of their box) is multiplied through the frame's
homography → position in metres. Points more than 2 m outside a touchline are dropped rather than
clamped (so a linesman never becomes a defender). Tracks are then smoothed with a rolling median.
A separate module (`to_frames.py`) can re-render these positions as StatsBomb-style **freeze
frames** (120×80 grid, attack always left→right, ≥10 players visible) — that's the seam where
StatsBomb-360-like data or a future relational model plugs in.

### 7. Ball detection (`generator/ball.py`, `tools/ball_possession.py`)

The ball is too small and fast for the player detector, so it gets its own network: **TrackNetV2**
(from the WASB-SBDT sports-ball repo), which reads **3 consecutive frames** (motion is the signal)
downscaled to 512×288 and outputs a heatmap; the peak above 0.5 is the ball. We fine-tuned it three
times on hand-labelled frames: v4 (iraq/mun) → **v5** (+ Senegal/Norway labels; WC production) →
**v6** (+ PL labels; 87% held-out recall on PL, PL production). The lineage exists because ball
detectors do **not** transfer across broadcast production styles — this "domain gap" cost weeks
before it was diagnosed.

### 8. Ball linking (`generator/ball.py::link_ball`) — the physics gate

Raw per-frame ball dots are noisy (false peaks, misses). `link_ball` chains them into one
trajectory with a constant-velocity motion model: the next dot is accepted only if a real ball
could have travelled there (**≤40 m/s**); short gaps (≤8 samples) are linearly interpolated; after
a long gap the tracker **re-seeds** instead of extrapolating a stale velocity (a missing clamp once
let a jittery velocity estimate fling the prediction 600 m off-pitch and kill entire chunks — the
54%→2.9% coverage collapse in STATUS).

**House rule born from a retraction:** the only ball-coverage number that counts is the
**post-`link_ball` usable track**. Pre-link detection counts flatter enormously (1407 noisy
detections once linked to just 128 usable frames).

### 9. Ball carry-over (`generator/ball_carry.py`) — borrowed geometry

For ball projection, a frame needs its own good homography (re-fit from ≥6 player
correspondences). Many PL frames narrowly lack one while a camera-continuous neighbour half a
second away has an excellent one. Carry-over **borrows the neighbour's homography** (within ±2 s),
but **never across a camera cut** — cuts are detected for free by track-ID survival (Jaccard
overlap < 0.30 between sampled frames = the tracker got reset = cut). Every carried point still
passes the off-pitch gate.

This lever roughly **doubled** PL ball coverage (20.8% → 38.8% pooled; Brighton full match 51.9%)
and was validated by leave-one-out cross-checking (borrowed geometry reproduces true ball positions
to a median 0.33 m). The same idea applied to *player* geometry (pose-carry) fires at only +1.7 pp
— because player metrics need ≥8 visible players and the missing frames are close-ups, which no
borrowed homography can fix. That asymmetry is why the "plumbing programme" is declared complete.

### 10. Possession (`generator/ball.py::assign_possession`)

The carrier is the nearest player within 2 m of the ball. Raw nearest-player flips constantly on
projection jitter, so the production path smooths the *team* sequence with a *Viterbi pass*: a
dynamic program where staying with the current team is free and switching costs the equivalent of
1.5 m — the ball must be *decisively* closer to the other team to flip possession.

**Honest label:** this is a proximity proxy, not event possession. It systematically
over-represents settled build-up (trackable) and under-represents transitions (untrackable) — we
proved this bias is uncorrectable without event data and renamed the metric "trackable-frame
possession share", reported caveated only.

### 11. Metrics (`fingerprint/*`)

All computed only on trusted frames. Position-only families (formation, width/compactness,
de-biased defensive line, lane occupation, synchrony, space control) survive any ball-coverage
level — they're the system's spine. Ball-dependent families (passes = carrier hand-offs, PPDA,
ball-xT, pressing intensity, counterpress curve, line breaks, verticality, phases, set pieces)
inherit ball coverage and are proxies, not events. The single most validated number in the project:
the **visibility-de-biased defensive line height** — broadcast cameras crop out deep defenders, so
the naive line reads ~16 m too high; a global correction slope (fit on pooled frames, per
back-line-player visible) brings pooled error vs FIFA's ground truth from **16.4 m to ~7.2 m
held-out** (5.5 m if measured in-sample — restated 2026-07-14 after a contamination check) while
preserving phase-to-phase shape.

### 12. Fact store, report, guardrail (`report/facts.py`, `report/report_v2.py`, `report/guardrail.py`)

`facts.py` runs the whole metric inventory over a match → `outputs/facts/<match>.json` (100–300
version-stamped numbers, CV and oracle side by side). `report_v2.py` writes the report **only from
CV facts**; ball-derived sections render only if the match passes a pre-declared evidence gate
(coverage ≥40% AND pass-recall proxy ≥50%) and otherwise print an explicit abstention; FIFA/oracle
numbers appear only in a validation appendix. The **guardrail** then extracts every number from the
prose and verifies it against the fact store with unit-typed tolerances — validated at 100%
precision and 100% adversarial recall. A hallucinated statistic cannot silently ship.

---

## What runs where (4 GB GPU laptop)

| GPU (sequential, never overlapping) | CPU (everything else) |
|---|---|
| YOLOv8 player detection | Team KMeans, anchoring |
| PnLCalib HRNet ×2 (calibration) | All homography algebra, projection |
| TrackNetV2 ball detection | `link_ball`, carry-over, Viterbi possession |
| (fine-tunes, when we run them) | All `fingerprint/*` metrics, facts, guardrail, report |

Sampling rates: players every 5th frame; full calibration every 25th processed frame (+ on
cuts/drift), reused between; ball detection on every player-bearing frame (reading 3 contiguous
native frames each time); all time windows converted per-chunk from true fps.

Weights: player YOLO auto-downloads from HuggingFace; PnLCalib HRNets live in `~/PnLCalib/weights`
(~265 MB each, env `FOOTBALL_PNLCALIB_PATH`); ball weights in `outputs/ball_finetuned/`
(`tracknetv2_v5.pth` = WC production, `v6` = PL production); ball annotation ground truth in
`data/ball_annotations/` (tracked in git — it's the permanent corpus).

## Processing a new match (the actual commands)

1. `python tools/chunk_video.py …` — split the mp4 into chunks.
2. Add one entry to `data/matches.yaml`.
3. `python -m tools.pl_pilot_run` — turnkey: extract → align → ball → facts → gate (resumable,
   `--from <stage>` to continue). WC-era pieces: `tools/batch_match.py`, `tools/analyze_match.py`,
   `tools/regen_ball.py`.
4. `python tools/verify_team_mapping.py --match <id>` — **team-mapping screen** (do this before any
   per-team analysis). Compares the BAS Man Utd poss-link majority against the Sofascore possession
   majority; on disagreement it prints a loud WARNING and writes `outputs/<id>/facts/
   team_mapping_flag.json`, meaning the `teams` order in `data/matches.yaml` is likely flipped (the
   degenerate red/striped-kit anchor failure that mislabelled southampton/tottenham — see
   `results/PAIR_ANALYSIS_v1.md`). Fix the registry order and rerun before trusting any per-team number.
5. `python -m report.report_v2 --match <id>` — the gated report.

## Glossary

- **Homography** — 3×3 matrix mapping image pixels to pitch metres, valid for points on the ground
  plane. One per (frame, camera pose).
- **Reprojection error** — after fitting, project known pitch points back into the image and
  measure the miss distance; our fit gate is ≤2 m mean.
- **Globally wrong pose** — a homography that fits the visible keypoints but places the camera
  wrongly, projecting players off-pitch; caught by the plausibility gate, not the fit gate.
- **Coverage (ball)** — % of a match's frames with a usable *linked* ball position (post-`link_ball`,
  never raw detections).
- **Recall proxy (passes)** — our detected carrier hand-offs ÷ oracle completed passes; a volume
  proxy, not per-event matching.
- **ByteTrack / track_id** — frame-to-frame box association; an ID is a followed blob, not a person.
- **Viterbi smoothing** — dynamic programming over a frame sequence that trades per-frame evidence
  against a switching penalty; here used to stop possession flip-flopping.
- **Freeze frame** — a StatsBomb-360-style snapshot of visible player positions at an event moment,
  normalized to 120×80 attack-left-to-right.
- **HOTA / GS-HOTA** — the field's standard tracking / game-state-reconstruction accuracy scores
  (we do not yet compute these — a known gap in `eval/gsr_score.py`).

## The honest numbers to keep in your head

| Number | Meaning |
|---|---|
| ~37% | detection-frames yielding trusted player geometry ≈ live-wide-play fraction of a broadcast |
| 91.8% / 0.21 m | homography solve rate / median keypoint reprojection error |
| 51.9% | best post-link ball coverage (Brighton, full match, carry-over ON) |
| 16.4 → ~7.2 m | defensive-line error vs FIFA, raw → de-biased, held-out (5.5 m in-sample; the headline validated result) |
| ~48% | pass-recall proxy on PL (symmetric between teams → relative claims OK, absolute not) |
| 87% | v6 ball detector held-out recall on PL frames |
| 100% | guardrail precision on shipped report bodies |
