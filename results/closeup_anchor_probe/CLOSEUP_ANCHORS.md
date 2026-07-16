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

---

# Stage-2b step 1 (negatives retrain) + step 2 (per-shot consensus) -- 2026-07-16

**Both levers were implemented, run end-to-end, and MEASURED. Both FAIL the pre-committed >=80%
propagation-precision bar. The negatives retrain is worse than a miss -- it destroys the anchor
source. Honest negative result; the remaining untested lever is the kit-color gate (step 3 / the
probe's own priority-1 recommendation).** Weights (kept for the record, NOT promoted):
`outputs/jersey/ckpt_torso_neg.pt` (aggressive), `outputs/jersey/ckpt_torso_neg_gentle.pt` (gentle).
New code: `tools/train_jersey.py` `--neg-dir` + `_negative_tracklets` (illegible pseudo-tracklets);
`tools/closeup_anchor_probe.py` `--mode harvest_neg` (+`_zero_detection_frames`,
`_has_player_inside`), `--consensus` (+`_shots`, `_consensus_anchors`). CPU seam tests:
`tests/test_closeup_anchor_seams.py` (4, green). Baseline stats preserved:
`closeup_anchor_stats_baseline.json`. Evidence montages: `neg_set_sample.png`,
`step1_survivors_sample.png`, `step2_consensus_anchors.png`.

## Step 1 -- negatives retrain of the illegible/reject head

**Negative harvest (weak-labeled, no manual labels), two documented sources**
(`closeup_anchor_probe.py --mode harvest_neg`):
- *Zero-detection (SHOT_GRAPHIC) frames* -- grid frames where football-YOLO fired nothing
  (`_zero_detection_frames`): crowd, referees, benches, coaches, graphics, extreme face close-ups.
- *Close-up frames* -- COCO person boxes (`box_h >= 100`) with **no** football-YOLO player/keeper
  point inside (`_has_player_inside`): crowd/ref behind play + front/side pitch players (no number).

5,500 raw crops harvested (500/chunk). **Weak-label purity, measured by eyeball (not assumed):** the
raw containment set leaks real numbered backs -- football-YOLO under-detects on close-ups, so many
pitch players fall into the "no player point" bucket, and broadcast close-ups are full of player
hero/celebration shots with legible numbers even on zero-detection frames. A recognizer post-filter
(drop crops the torso model reads as a confident non-attractor number, conf>=0.35) removed
897 (16%) likely-genuine backs; the kept 4,603-crop set (`data/jersey_negatives`) still carries
~10% residual legible-back leakage (dominated by the attractor classes 1/11/20/29 we intend to
suppress). This residual is the documented ceiling of the free weak-label.

**Training:** warm-start from the torso checkpoint (`jersey_torso_r224_acc417.pt`), negatives added
as illegible pseudo-tracklets (~13% of each epoch), 5 epochs, lr 1e-4. A second, gentler run used
the pure zero-detection subset only, post-filtered (`data/jersey_negatives_graphic_clean`, 2,592
crops, ~8%/epoch), 2 epochs, lr 5e-5.

**SoccerNet regression gate (official 1211-tracklet test, the gate = must not drop below 0.41):**

| model | min_conf | tracklet acc | numbered-only | legP | legR |
|---|---|---|---|---|---|
| torso baseline (Stage-1c) | 0.20 | 0.4170 | 0.299 | 0.855 | 0.730 |
| **+ negatives (aggressive)** | 0.20 | **0.4476** | 0.308 | 0.855 | 0.530 |
| **+ negatives (aggressive)** | 0.05 | **0.4500** | 0.322 | 0.855 | 0.590 |

The gate **PASSES with margin -- the retrain even *improves* SoccerNet by +3.1 pp.** But that
improvement is the tell: it comes entirely from rejecting more of the 355 illegible test tracklets
(legibility recall drops 0.730 -> 0.530, i.e. the model now says "-1" far more often), and SoccerNet
crops are broadcast-wide -- a **different visual domain** from the close-up crops the negatives came
from.

**Close-up anchor recall -- COLLAPSE (the decisive measurement).** Re-reading the 5,554 baseline
high-conf anchor crops (`anchors/`) with the retrained model, counting those still read as a
confident number (conf>=0.70):

| model | anchors surviving | share | 3 verified-correct backs (8,8,20) |
|---|---|---|---|
| torso baseline | 5,554 (all, by construction) | 100% | read 8/8/20 correctly |
| + negatives (aggressive) | **4** | **0.1%** | all three now read **illegible** |
| + negatives (gentle) | **180** | **3.2%** | all three now read **illegible** |

All 24 saved threshold-freezing spot-check crops (`spotcheck/sc_*.jpg`) -- including the three
confirmed-correct Man Utd backs -- now read **illegible** under both retrained models. And the 180
gentle-model survivors are **still ~30% precision** (eyeball of 48: a referee read as `10`, front/
side reds labelled `20`/`16`/`10`, Brighton `29` -- the same attractor mass, just a random 3%
subset): `step1_survivors_sample.png`.

**Root cause -- domain shortcut.** Every harvested negative is a close-up-domain crop; at the
recognizer's 224x112 upscaled input the discriminative signal it latches onto is *domain*
(close-up blur / motion / scale artifacts), not the ~14-40 px number patch. So the reject head
learns "close-up-domain crop -> illegible" wholesale and rejects the entire close-up domain,
genuine back numbers included. SoccerNet (broadcast-wide) is a disjoint domain, so it is spared --
which is exactly why the gate *improves* while close-up recall goes to zero. Cleanly separating a
non-player crop from a back-number crop of the same domain needs close-up **positives** (labelled
back numbers from close-ups), which the "no manual labelling" constraint of step 1 cannot provide.

**Step-1 verdict: FAILS the 80% bar, and is counterproductive.** It does not clean the anchor
population; it deletes it (recall 100% -> 0.1-3.2%) while the few survivors stay ~30% precise. The
recognizer is fine on clean broadcast backs; retraining its reject head on cheap same-domain
negatives is the wrong lever.

## Step 2 -- per-shot consensus (original torso model)

Ran the full funnel with `--consensus` (each close-up shot = one `aggregate_votes` over all its
crops; `_shots` + `_consensus_anchors`), threshold 0.70 pooled, on the *original* torso model (the
retrained model has no recall to aggregate). Per-frame numbers reproduce the baseline exactly
(5,558 anchors, 802 shots); consensus adds:

| unit | count match-wide | number histogram | est. precision |
|---|---|---|---|
| per-frame anchors (baseline) | 5,558 | 20/29/1/11 = 80% of mass | ~20-25% (probe audit) |
| **per-shot consensus anchors** | **5** | {20: 3, 24: 2} | **1/5 = 20%** (per-crop verified) |

Per-crop verdicts of all 5 consensus anchors (`step2_consensus_anchors.png`): `20`@0.72 on a clear
Man Utd "20" back = **CORRECT**; `24`@0.83 on a clear "20" back = WRONG (20->24 misread); `24`@0.91
on Casemiro's **front** (no back number) = WRONG; `20`@0.72 on Garnacho's "17" back = WRONG;
`20`@0.72 on a motion-blur turn = WRONG.

**Step-2 verdict: FAILS the 80% bar on both yield and precision.** Two independent failure modes:
(1) pooling *all* of a shot's crops (foreground player + crowd + other players) dilutes any single
number so hard that only 5 shots match-wide clear the 0.70 pooled floor -- consensus over a shot
without per-person association inside the shot is the wrong aggregation unit; (2) the surviving
reads are still attractor hallucinations because a front/side player reads the *same* attractor
number on every frame of a shot, so pooling **reinforces** the consistent hallucination instead of
cancelling it. Consensus suppresses random scatter (crowd), not consistent within-shot hallucination
-- which is the majority of the false mass.

## Bottom line vs the >=80% bar, and what is left

Neither step 1 (negatives retrain) nor step 2 (per-shot consensus) produces clean gated anchors;
both leave precision at ~20-30% or destroy recall. **The pre-committed >=80% propagation bar is NOT
cleared -- do not wire close-up anchors.** The one lever not yet tried is the probe's *own*
priority-1 recommendation, deferred by this task to step 3: the **kit-color gate** (require the
crop's torso colour to match one of the two match kits). Unlike a same-domain reject head it keys on
colour, not domain, so it can drop crowd/referee/coach crops without rejecting pitch players, and
unlike consensus it acts per-crop so it is not diluted -- it is the correct next thing to measure.

---

# Stage-2b step 3 (kit-color gate + OCR digit-evidence gate) -- 2026-07-16

**Both remaining cheap levers were implemented as flags, frozen on the 20-crop spot-check set, and
run end-to-end in one 4-arm pass. The combined gate CLEARS the >=80% bar with enormous margin:
verified precision 69/70 = 98.6% at a non-trivial yield of 226 frame-anchors over 95 close-up shots
(161 propagation-feasible). VERDICT: WIRE the gated extractor -- with the concentration caveat
below.** New code (flags, no deletions): `tools/closeup_anchor_probe.py` `--kit-gate`
(`kit_centroids`, `_kit_dist_ok`) + `--ocr-gate` (`_ocr_reader`, `_ocr_tokens`, `_digit_agreement`),
single-pass 4-arm funnel in `run_full`, `_finalize_step3`. New dep (OCR lever only):
**easyocr 1.7.2** (CPU; cp314 wheels for scikit-image 0.26.0, pyclipper 1.4.0, python-bidi 0.6.11,
etc.; torch/torchvision untouched). CPU seam tests: `tests/test_closeup_anchor_seams.py`
(+2 = 6 green). Evidence: `spotcheck_step3/` (226 survivors in `_survivors/`, 40-crop sample
`verdicts.json`, montages `montage_eights.png` / `montage_others_labeled.png` / `zoom_p01.png`),
`kit_centroids.json`, `closeup_anchor_stats.json`.

## The two levers (each targets one error mass, frozen before the run)

- **LEVER A -- kit-color gate** (`_kit_dist_ok`, `KIT_DIST_MAX = 22.0`): a candidate's median-torso
  CIELAB (`generator.teams.jersey_color`) must sit within 22 of one of the two match-kit centroids,
  recomputed per match from wide-play crops (`kit_centroids`; Brighton `[69.1, 4.5, -10.7]`, Man Utd
  red `[42.8, 29.9, 7.3]`). Frozen on the spot-check set: the 5 confirmed-correct Man Utd back crops
  sit at dmin <= 18.7, the referee (only clear non-player) at 40.5 -> 22 keeps players, drops
  crowd/ref/coach. (Single-negative caveat: only one non-player example existed in the frozen set.)
- **LEVER B -- OCR digit-evidence + agreement gate** (`_digit_agreement`): easyocr on the 3x-upscaled
  torso band (`OCR_BAND = (0.15,0.55,0.15,0.85)`); anchor valid iff OCR returns a digit token
  (conf >= 0.5) whose value AGREES with the classifier's number. Frozen on the spot-check set: reads
  the "8" and "20" backs at conf 1.0, returns nothing on the referee / illegible Brighton / 2-body
  crops, and reads `20` on the classifier's `20->24` misread (disagreement -> correctly rejected).
  This is the precision play: requiring an independent OCR read of the *same* number both proves a
  digit region exists (kills the front/side no-number attractors) and cross-checks the value (kills
  `20<->24`-style misreads).

## 4-arm funnel (full match, threshold 0.70)

| stage | count | note |
|---|---|---|
| eligible crops (box_h >= 100) | 43,020 | unchanged crop pool |
| **baseline** anchors (conf >= 0.70) | **5,558** | ~20-25% precision (step-0 audit) |
| **+A** kit-color gate | **4,281** | drops 1,277 off-kit (crowd/ref/coach); in-kit front/side survive |
| **+B** OCR-agreement gate | **234** | drops 96% of baseline -- the dominant filter |
| **+A+B** both | **226** | +A trims 8 off-kit OCR-agreements off +B (small precision insurance) |
| +A+B propagation-feasible (wide within +/-2s) | 161 | 71% of survivors could attach |
| close-up shots with an +A+B anchor | 95 / 802 | distinct shots covered |

## Measured precision per arm vs the 80% bar

- **baseline ~20-25%** and **+A alone still fails** (~25-30%): the kit gate removes non-kit
  crops but the dominant error mass -- IN-KIT front/side players reading attractor 20/29/11 -- passes
  colour untouched. Kit-color alone is necessary-not-sufficient, exactly as predicted.
- **+B alone (234) and +A+B (226): PASS.** Every +A+B survivor was audited (226 > 60, so a 40-crop
  stratified sample is the requirement; in practice ALL 40 non-`8` survivors were verified
  individually plus a 30-crop sample of the 186 `8`s):
  - 186 pred=`8`: sampled 30, **30/30 correct** -- every crop a clear Man Utd red back with a legible
    white "8" (Bruno), `montage_eights.png`.
  - 40 non-`8` (preds 10, 20, 11, 6, 1 -- the entire non-`8` population): **39/40 correct** legible
    backs (Man Utd `20`/`10`/`6`, Brighton `10`/`11`), `montage_others_labeled.png`. The single
    failure is a touchline-melee front-crop of a Man Utd player beside the FIFA referee read as `1`
    @0.768 -- OCR fired on the referee's "FIFA REFEREE" badge digits (`zoom_p01.png`).
  - **Verified precision 69/70 = 98.6%** (stratified 40-crop sample: 39/40 = 97.5%). **>> 80% bar.**

## Yield (WIRE decision inputs)

- **226 high-precision frame-anchors**, **95 distinct close-up shots covered**, **161
  propagation-feasible** (a `live_wide` tactical frame within +/-2 s across the cut).
- **Concentration caveat (honest):** the yield is skewed to hero-shot players. Number histogram of
  the 226 survivors: `8`x186 (82%), `10`x27, `20`x8, `11`x3, `1`x1, `6`x1 -- roughly 5-6 distinct
  player-numbers, dominated by Bruno #8. This is not a uniform per-player anchor source; it is a
  high-confidence source for the handful of players who repeatedly get close-up hero shots.

## Verdict: WIRE (gated close-up anchor extraction)

The gate clears the pre-committed >=80% bar by ~19 pp (98.6% vs 80%) at a usable yield. **Wire the
`--kit-gate --ocr-gate` extractor** (kit-color per-crop gate + easyocr digit-agreement), not the raw
conf-gate this probe opened with. Ordering of contribution: **LEVER B (OCR-agreement) is the
decisive filter** (5,558 -> 234); LEVER A (kit-color) removes the residual 8 off-kit OCR-agreements
(the melee-badge failure mode) and is cheap insurance -- keep both.

Wiring implications for the identity plan:
1. Anchors are a **weak per-track number+team prior**, applied per close-up shot (report the 95
   covered shots), never a standalone identity -- unchanged from the step-0 conclusion, but now the
   prior is trustworthy (~99% vs ~22%).
2. Because coverage is concentrated (Bruno #8 etc.), close-up anchors **supplement** roster-level
   priors and relink constraints for hero-shot players; they do NOT replace a broad close-up-domain
   reader. The uniform-coverage gap (most squad numbers never surface a legible close-up back) still
   needs the cluster-trained / VLM route flagged in step 0.
3. Step-1 (negatives retrain) and step-2 (consensus) stay PARKED; step 3 supersedes them as the
   shippable gate.
