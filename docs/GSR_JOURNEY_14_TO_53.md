# How the GSR score went from 14.76 to 53.09

*A plain-language technical record of the SoccerNet Game State Reconstruction campaign,
2026-07-16 to 2026-08-07. Companion doc: `GSR_NEXT_PLAN.md` (what we do next and why).
Deep detail per step lives in `results/GSR_*.md`; day-by-day in `STATUS.md`.*

---

## 1. The task and the metric

**The task (GSR):** given a broadcast clip of a football match, output — for every frame —
where every player is *on the pitch* (2D coordinates in metres, not pixels), which team
they play for, their role (player / goalkeeper / referee / ball), and their jersey number.

**The metric (GS-HOTA):** HOTA is the standard multi-object-tracking score,
`HOTA = sqrt(DetA x AssA)` — DetA asks "did you find the objects?", AssA asks "did you
keep each one's identity continuous over time?". GS-HOTA adds one merciless change:

> A prediction matches ground truth only if the position is close enough **AND team,
> role, and jersey number are ALL correct. A wrong number scores zero, however perfect
> the position.**

Three consequences that shaped every decision:

1. **A wrong number is worse than no number.** A `null` jersey can still match a GT
   player the annotators left unnumbered; a wrong number matches nothing. Abstaining
   when unsure is a strategy, not a weakness.
2. **Jersey errors dominate** — measured at ~2.5x the cost of any other attribute, both
   by the current leader's paper and by our own decomposition. GS-HOTA is an identity
   metric wearing a tracking metric's clothes.
3. Teams are ranked on GS-HOTA alone; everything else is diagnostics.

**Where we ended:** public board **53.09** (local evaluator says 53.0846; the board
rounds), roughly 4th on the open test-phase leaderboard. The leader (Broadcast2Pitch,
WACV 2026, KIST) holds 61.48.

---

## 2. The score arc at a glance

Careful with the eval sets — three different ones appear below. **valid-58** = the public
validation split (58 clips, scored locally). **TEST-38** = our own held-out 38 of those
58 (never tuned on). **test-49 / board** = the official hidden-GT test split on the
codabench server (1 submission/day, 10 max — we used 5).

| Date | Score | Split | What changed |
|---|---|---|---|
| 07-16 | **14.76** | valid-58 | First official scoring ever. Detector + ByteTrack + calibration; **no jersey numbers at all** (every row `jersey=null`) |
| 07-17 | 19.83 | valid-58 | Jersey-reading layer attached (+5.07, **0 of 58 clips hurt**) |
| 07-20 | 22.85 | valid-58 | Track re-linking + propagating numbers inside merged tracks |
| 07-27 | 24.18 | valid-58 | Tracklet connector (merging fragments by appearance) |
| 07-28 | **33.20** | TEST-38 | **The OCR aggregation bug fix** — the single biggest jump of the whole campaign, zero new models |
| 08-01 | **31.88** | board #1 | **Legitimacy reset**: two accidental ground-truth leaks removed. Score goes DOWN; the number becomes real |
| 08-01 | 33.37 | board #2 | Calibration gap fill |
| 08-01 | 35.40 | board #3 | Calibration gate repair (our own safety check was the villain) |
| 08-02 | 39.02 | board #4 | Two knobs: lower jersey-vote floor + looser merging behind a jersey-conflict guard |
| 08-06 | **53.09** | board #5 | **The trained trio**: fine-tuned detector + retrained jersey reader + new tracker (GPU-cluster week) |

Everything after 08-06 (the v7 and v8 sessions) is verification, negative results, and
groundwork — nothing has shipped since, *by our own pre-registered rules* (section 5).

---

## 3. The story, act by act

### Act 1 — the floor: 14.76 (07-16)

The first end-to-end pipeline: a YOLO detector finds people, ByteTrack links them across
frames, PnLCalib estimates the camera so pixel positions become pitch metres, and a
colour-clustering step splits the two teams. It emitted **no jersey numbers at all** —
and under the identity gate, every numbered GT player is therefore unmatchable. 14.76 is
what "we can track and localise but cannot say who anyone is" is worth.

*In plain terms: we could draw dots on a map of the pitch, but every dot was anonymous —
and this benchmark pays almost entirely for names.*

### Act 2 — learning to read: 19.83 → 22.85 (07-17 to 07-20)

We attached a jersey-number reading chain (following Koshkina 2024): a small classifier
first asks "is a number even visible on this crop?" (legibility filter), a pose model
crops the torso, a scene-text model (PARSeq) reads the digits, and the track's frames
vote. It read numbers on only 8.7% of tracks — but because it **abstained** everywhere
else, not one of the 58 clips got worse (+5.07). Then re-linking broken tracks and
propagating a confidently-read number to the other fragments of the same player added
three more points.

*In plain terms: read a number only when you're sure; silence is free, a wrong guess is
poison. Then, if two track fragments are provably the same player, share the name.*

### Act 3 — the nine-point bug: 33.20 (07-28)

The best day of the campaign cost zero GPU-hours. The per-frame votes for a track's
number were being aggregated with a bug: **crops the legibility filter had rejected as
unreadable still cast votes — at maximum confidence.** Garbage was out-voting the good
reads. Fixing the aggregation more than doubled the density of usable reads (d 0.088 →
0.208) and moved TEST-38 from 24.19 to 33.20, helping 35 of 38 clips.

*In plain terms: the pipeline was already reading numbers well; a tallying bug was
throwing the good reads away. Finding it was worth more than any model we ever trained.*

### Act 4 — the honesty reset: 31.88 (08-01, board #1)

Before first upload we audited ourselves and found **two places where the pipeline read
the answer key**: the team→side assignment picked whichever mapping agreed with GT, and
the identity solver was handed the true roster of (team, jersey) slots. Both were
replaced with GT-free versions — side from mean pitch position (the team defending left
sits at smaller x), roster from the sequence's *own* OCR reads. The measured cost of
honesty: about 2.5 points. Our first public number, 31.88, was lower than our local
scores *because it was clean.*

*In plain terms: we caught our own pipeline cheating in two places, removed it, took the
score hit, and only then went on the public board.*

### Act 5 — repairing calibration: 33.37 → 35.40 (08-01)

Two submissions in one day (the daily limit forced the split). First, **gap fill**: when
the camera estimator fails on a frame, re-fit from the run's own good frames and
interpolate across short gaps — recovered 9.6% of all test rows. Second, the bigger one:
the frame-rejection safety rule **we ourselves had written** (reject unless ≥8 players
spanning ≥25 m) was discarding 96.4% of "dead" frames that actually had good
homographies. Relaxing it to (3 players, 5 m) with an on-pitch plausibility check
recovered 84,000 rows (18.3% of the test set).

*In plain terms: the camera-maths was mostly fine; our own overcautious sanity check was
throwing away correct answers. Both fixes together: +3.5 points, no new models.*

### Act 6 — two knobs: 39.02 (08-02)

No new code paths — two frozen thresholds moved, each behind a validator. The jersey
vote-confidence floor went 0.85 → 0.80 (+1.09: slightly braver naming). The tracklet
connector's merge threshold was loosened 2x — made safe by a new **jersey-conflict
guard**: never merge two fragments whose confidently-read numbers disagree (+1.65).
43 of 49 test clips improved.

*In plain terms: be a little braver about naming, and a lot braver about gluing track
fragments together — but install a tripwire so two different shirt numbers can never be
glued into one player.*

### Act 7 — the trained trio: 53.09 (08-05 to 08-06)

The college A100 server unlocked training, and one data discovery powered most of it:
**SoccerNet-v3** (an older, freely licensed dataset: 400 games, 33,986 annotated stills)
turned out to carry **106,591 pixel-verified jersey-number labels** nobody in the GSR
context was using. Three components were built in a week of pre-gated sessions:

- **Detector (S4b):** YOLOv8s fine-tuned on GSR-train + SoccerNet-v3, with
  role-as-a-class (player/GK/referee/ball). mAP@0.5 +20.7 points; role-correct coverage
  0.84 → 0.91. Its first training run (S4) FAILED its pre-declared gate by 20 detections
  and was recorded as a FAIL; 10 more epochs passed a freshly declared gate.
- **Jersey reader (S3):** PARSeq fine-tuned on those 106k labels — but only the 43%
  that survive our legibility filter (annotators had labelled numbers on crops where no
  number is visible; training on those teaches hallucination). Per-crop precision 0.70 →
  0.83 at the same emit rate. The filtered v3 labels were the ENTIRE effect; a synthetic
  digit pre-train measured *negative*.
- **Tracker (S5):** a clean-room reimplementation of ExpansionIoU (their repo has no
  license, so nothing was read from it — built from the paper), with appearance
  embeddings from a CLIP model we trained on 160,070 crops / 60,040 identities. The
  honest decomposition: the *deep appearance features* carry the gain (+3.49); the
  expansion trick itself is worth only +0.37. EIoU fragments tracks 1.8x MORE than
  ByteTrack but keeps them purer — which is exactly right for a pipeline that names
  whole tracklets.

Bundle result: TEST-38 49.50 against a pre-declared gate of 40.0; one test-49 run;
upload #5 → **53.09, +14.06 in one submission**. Leave-one-out prices on DEV-20:
detector −9.06, reader −5.52, tracker −2.09 if dropped.

*In plain terms: with real GPUs we retrained the three core models. The trick wasn't a
clever architecture — it was finding 106k free jersey labels in an older dataset, being
picky about which of them to trust, and rebuilding an unlicensed tracker legally from
its paper.*

---

## 4. What the pipeline is now (one line each)

1. **Detector** — fine-tuned YOLOv8s; finds people + ball and assigns role.
2. **Tracker** — clean-room EIoU + CLIP appearance embeddings; short, pure tracklets.
3. **Connector** — merges tracklets by appearance (tau 0.450) behind the
   jersey-conflict guard.
4. **Calibration** — PnLCalib per frame + repaired acceptance gate + gap fill; ~0.5 m
   median position accuracy (measured MORE accurate than the current leader's, v8-W2).
5. **Team side** — kit-colour KMeans + "smaller mean x defends left" (GT-free);
   918 of 949 clip-level calls correct; a single flipped clip costs ~33 points on it.
6. **Jersey reader** — legibility filter → torso crop → fine-tuned PARSeq →
   confidence-weighted tracklet vote (the fixed aggregation).
7. **Identity solver** — a small assignment optimisation naming tracklets from the
   self-built roster; frozen since 07-28 (measured three times: it buys a dial, not a
   better frontier).

---

## 5. The discipline that made the numbers trustworthy

- **Three splits, three roles:** tune on DEV-20 only; TEST-38 read once per frozen
  recipe; test-49 read once per version, then uploaded. Configs frozen to timestamped
  JSON *before* any held-out read.
- **Pre-registration:** headline claims get a registered pass/fail bar before the
  experiment runs. Failures are recorded as failures — v7 ran five pre-registered ideas,
  all five failed their gates, and v7 shipped **nothing**; 53.09 stands untouched.
- **Paired statistics only:** per-sequence Wilcoxon tests, never bare means.
- **Everything in the ledger:** ~200 claims in `knowledge/claims.json`, each with
  evidence paths — including the retractions.

The graveyard is long and priced: learned association transplants (CAMELTrack, −13 to
−20 zero-shot), VLM jersey reading (0.27 precision where our chain abstains), commentary
as identity evidence (binding at chance), gait/face ID (median face = 9.9 pixels),
appearance retrieval on same-kit broadcast (saturated — three registered fails in three
days), teacher-imitation pseudo-labels (the head learns to imitate its teacher, not the
world). Each negative closed a road so the next session didn't drive down it again.

And one more find that reframes the scoreboard itself (v8-W0, 08-13): **five of the
benchmark's own sequences have side-swapped ground truth** — all five are second-half
clips still carrying their game's first-half side convention. Three are named in the
leader's paper; two (SNGS-092, SNGS-111) we found beyond their list. About 2.45 points
of our loss is benchmark noise no correct method can recover; under corrected GT our
score is ~55.5.

---

## 6. Where the remaining points are

Measured, not guessed (see `GSR_NEXT_PLAN.md` for the plan built on this):

- **Association is the big one:** a connector-oracle ceiling of **+11.9 GS-HOTA** exists
  in merging tracklet fragments correctly; we currently realise 43% of it.
- **Naming coverage:** 260 of 870 tracklets per-DEV-run have *no admissible name* —
  nobody ever read their number; more/better reads (not a better solver) is the lever.
- **Dead-frame calibration coverage** — partially closed by W2b's probe; remaining gain
  small but real.
- **GT noise:** ~2.45 points unrecoverable by honest methods (see the audit above).
