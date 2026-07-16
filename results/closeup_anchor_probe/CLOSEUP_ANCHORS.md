# Close-up jersey-anchor probe -- brighton_manutd (B2 Stage-2b opening probe, 2026-07-16)

PROBE ONLY -- nothing wired into the pipeline. Question: end-to-end, how many **high-confidence
(number, track-attachable) anchors** does a full match yield from close-up shots, before any fusion
is designed? Tool: `tools/closeup_anchor_probe.py`. Artifacts: this file,
`closeup_anchor_stats.json`, `closeup_reads.csv`, `anchors/` (5,554 anchor crops), `spotcheck/`
(24 verified crops + `VERDICTS.md`).

## Method (all reused modules)

1. **Shot selection** -- `generator.live_play.classify_dense` on the aligned parquet; keep
   `shot_type == close_up` (1-4 football-YOLO detections). 7,234 close-up frames match-wide.
2. **Re-detection** -- COCO `yolov8s` person detection (conf 0.20) on those frames (football-YOLO is
   blind on close-ups; established by the July feasibility probe).
3. **Crop + read** -- each person box with `box_h >= 100 px` (pre-committed crop-eligibility floor;
   below it a back number is < ~14 px even in a close-up) is read by the trained torso recognizer
   `outputs/jersey/jersey_torso_r224_acc417.pt` (`generator.jersey_id.JerseyRecognizer`).
4. **Anchor rule (FROZEN before the full run)** -- per-crop peak-class confidence `>= 0.70`, frozen
   on a manual 20-crop spot-check of ONE chunk (`spotcheck/VERDICTS.md`): every confirmed failure
   there sat at conf <= 0.40, every confirmed-correct read at conf >= 0.53; 0.70 is the conservative
   high-confidence floor. Applied unchanged to all 11 chunks.

## Yield funnel (full match, threshold 0.70)

| stage | count | note |
|---|---|---|
| close-up frames | 7,234 | `shot_type == close_up`, all 11 chunks |
| persons detected (COCO) | 94,314 | incl. crowd, bench, referees, small background |
| eligible crops (box_h >= 100 px) | 43,020 | crop pool fed to the recognizer |
| "legible" reads (argmax != illegible) | 35,950 | recognizer emitted a number |
| **high-conf anchors (conf >= 0.70)** | **5,558** | per-frame |
| distinct close-up shots | 802 | contiguous close-up runs (gap <= 15 frames) |
| **shots with >= 1 anchor** | **626 (78%)** | per-shot |

Per-chunk table is in `closeup_anchor_stats.json`. Raw volume looks healthy: 78% of close-up shots
carry a high-confidence read, ~5.6 k frame-level anchors.

## The catch: that yield is mostly HALLUCINATION (the decisive finding)

The high-confidence anchor population is pathologically concentrated on a few number classes:

| number | anchors | share | what the crops actually are (eyeballed) |
|---|---|---|---|
| 20 | 2,059 | 37% | Man Utd reds, **mostly front/side/motion, no back number visible** (+ some correct backs) |
| 29 | 1,030 | 18% | Brighton, **side/front, no legible number** (0/3 confirmed correct) |
| 1  | 837 | 15% | **crowd, referees, back-of-head close-ups -- not jerseys at all** (3/3 confirmed false) |
| 11 | 538 | 10% | Brighton front/side hallucination |
| 10 | 293 | 5% | includes **confirmed-correct** Man Utd "10" backs |
| 8  | 242 | 4% | **confirmed-correct** Man Utd "8" (Bruno) backs |

**Top-4 classes = 80% of all anchors.** A real match has ~30 distinct players roughly uniformly;
this distribution is the population signature of a recognizer defaulting to attractor classes on
out-of-distribution crops.

**Eyeball precision audit of high-conf (>= 0.70) anchor crops** (`anchors/`, ~14 adjudicable):

- CORRECT clean back-number reads: **3** -- all Man Utd, clear digits (`8`@0.85, `10`@0.90, `8`@0.82).
- WRONG: **~11** -- crowd->`1`@0.93, referee->`1`@0.94, back-of-head->`1`@0.90, front/side player
  ->`20`@0.97 / `11`@0.92 / `29`@0.98 (no number present), and a genuine misread `20`->`24`@0.83.
- => **high-confidence anchor precision ~20-25%.** The confidence gate does NOT separate true reads
  from hallucinations: a crowd shot scores `1`@0.94.

Root cause: the recognizer's illegible head was trained only on *player* tracklets, so on
non-player / front-facing / crowd crops -- which COCO happily detects on close-up frames -- it has
never learned to say "no number" and instead emits a confident attractor number.

## Propagation feasibility (no fusion attempted)

For each anchor frame, is there a `live_wide` tactical frame within +/- 2 s across the cut?

- **2,658 / 5,558 = 47.8% of frame-anchors** have a tactical frame within +/- 2 s (the "could
  attach" number, as asked).
- **Team-match is a no-op filter**: `anchors_team_matchable == anchors_attachable_2s` exactly in
  every chunk -- both teams are present on essentially every wide frame, so once a wide window
  exists the anchor's kit team always appears. The hard part of propagation is *which track*, not
  *which team* -- and that is out of scope here (no fusion). (Kit-team guess is a naive red-share
  heuristic and is itself unreliable -- it labelled clear Man Utd reds as team 1 -- so treat the
  team column as illustrative only.)
- Frame-level 47.8% is a lower bound on shot-level attachability (a shot's boundary frames are
  cut-adjacent even when its middle is not); refining it is moot until anchor precision is fixed.

## Transfer verdict (recognizer -> close-up domain)

**It transfers.** On crops whose number is human-legible (clean back views), the torso recognizer
reads correctly at high confidence -- confirmed `8` (x3), `10`, `20` on Man Utd backs, vs 1/99
chance. It is **not** garbage on this domain, and downscaling the high-res close-up crop to its
224x112 input does not break it. No OCR baseline was run (easyocr/tesseract absent; not adding a
dependency for the optional sanity leg -- the visual verification IS the transfer check).

The failure is not the reader on clean backs; it is the **absence of an out-of-distribution reject**
plus per-frame (not per-shot) decisions.

## Verdict + recommended Stage-2b architecture

**Anchor-propagation is the right architecture** -- genuine, correct, high-confidence close-up
back-number reads exist and are unambiguous -- **but the raw "recognizer + confidence gate on
close-up crops" pipeline ships ~80% hallucinated anchors and must NOT be wired as-is** (same lesson
as the Stage-2a relink 35% merge-precision production guard: a wrong anchor propagated along a whole
track corrupts identity worse than no anchor).

Before propagation, Stage 2b needs, in priority order:

1. **An out-of-distribution / content gate upstream of the reader.** Reject non-players
   (crowd/referee/coach) and non-back views. Cheapest levers: a team-kit gate (only red or
   blue-white kit crops pass -- crowd/ref fail) + a back-facing orientation gate. This alone kills
   the class-1 (crowd/ref) and much of the class-20/29/11 (front/side) hallucination mass.
2. **Retrain the recognizer's legibility/illegible head with hard NEGATIVES** -- crowd, referees,
   front/side player crops -- so "no number present" fires on OOD instead of defaulting to an
   attractor. (Cheap head-only fine-tune; the number head already transfers.)
3. **Per-close-up-shot consensus, not per-frame.** A true player reads the same number across a
   shot's consecutive frames; hallucinations scatter. Apply `jersey_id.aggregate_votes` over each
   close-up shot (802 shots) -- this is the shot analogue of tracklet voting and suppresses
   inconsistent noise. Report anchors per shot, gated on within-shot agreement, not 5,558 per-frame.
4. Residual `20<->24`-style visual misreads remain the known SoccerNet number-recognition ceiling
   (Stage-1 line closed at 0.31 numbered-only) -- fuse the anchor as a weak per-track prior with
   team/role/position, never as a standalone identity.

**Redirect, not abandonment:** the identity source (trained recognizer) is fine on clean backs;
Stage 2b = *gated* anchor-propagation (OOD/back-facing filter + per-shot voting), not OCR-from-scratch
and not the naive per-frame conf-gate this probe measured.

## Scope boundary (stated, not hidden)

Extreme close-ups that yield **zero** football-YOLO detections are `SHOT_GRAPHIC` (absent from the
aligned parquet) and were not sampled here -- so the biggest, most legible face-cam frames are
partly missed. Those are predominantly front/face content (no back number), so the miss is largely
non-anchoring; a full-grid re-decode would recover the minority of back-facing extreme close-ups.
