# What we do next, and why

*The v8 campaign plan in plain technical language, as of 2026-08-13. Companion doc:
`GSR_JOURNEY_14_TO_53.md` (how we got here). We stand at public 53.09 (~4th); the
leader, Broadcast2Pitch (WACV 2026), holds 61.48. Five of ten submissions used.*

---

## 1. The problem v8 exists to solve

The score is decent, but a reviewer in December will ask: **"what was YOUR innovation?"**
A pipeline of well-executed known parts earns engineering credit, not research credit.
So the v8 campaign was designed around three moves:

1. **Build upon the best published system** — take the leader's paper apart, measure
   which of its claimed advantages actually transfer to our pipeline, and attack the
   bottleneck *they themselves name*.
2. **Own the data** — improve the labels themselves (our annotations, our filtering
   discipline, and an audit of the benchmark's own ground truth).
3. **Say only what we measured** — every experiment pre-registers its pass/fail bar
   before running; failures get recorded and published, not buried.

One legal-hygiene rule sits under all of it: the leader's code and weights are
unlicensed, so they are used **only as private measurement instruments** (never copied,
never shipped); everything we ship derives from their *paper* via our own
implementations. Ideas are fair game in academia; code is not.

---

## 2. What v8 has already found (four sessions, one week)

These are the results the plan's remaining steps are built on:

**W0 — the benchmark audit.** Five GSR sequences have side-swapped ground truth — all
five are second-half clips still carrying the first-half side convention, a diagnosable
annotation-process error. The leader's paper names three; we found two more (SNGS-092,
SNGS-111) with an instrument built without reading their code. ~2.45 of our lost points
are benchmark noise; corrected-GT score ~55.5. *This is a publishable data contribution
in its own right.*

**W1 — the seam experiment.** The leader names their own primary bottleneck: they
aggregate jersey reads by simple majority vote, destroying confidence information. We
ran *their* jersey model on our crops and swapped only the aggregation: our
confidence-weighted voting beats their majority vote **on their own model's outputs**
(+0.107 tracklet accuracy, p=0.0017). Bonus measurement: our jersey reader covers
10-11x more tracks than theirs at 0.90+ precision. Their named weakness is real, and
our machinery fixes it.

**W2 — the calibration inversion.** Their ablation credits calibration with +10.28. We
measured head-to-head on identical rows: **our calibration is 12% more accurate than
theirs.** Their gain was relative to their own weak baseline. What they DO have is
coverage — their calibrator answers on 78% of the frames where ours goes silent. The
conclusion redirected effort from "adopt their calibrator" to "fix our dead frames".

**W2b — the line-solver fallback (registered FAIL).** We built the one idea in their
paper nobody exploited — solving the camera from the pitch's painted *lines* when
keypoints fail. It recovers 66% of dead frames accurately, but landed +0.30 against a
pre-registered +0.40 bar, so by our own rules it does not ship. Banked with it: two
premise corrections (a planned fallback model turned out to be byte-identical to what
we already run; and a third of our "uncalibratable" frames actually hold a correct
answer that our own gate vetoes for having too few visible players — a cheap future fix).

**Net effect:** three of the leader's four supposed edges are measured away (reader,
calibration accuracy, team assignment). Their remaining real advantages: their
association module (+2.97 in their ablation) and substrate we haven't priced. Our
largest measured headroom is **association** (+11.9 ceiling) and **naming coverage**
(260 of 870 tracklets never get a readable number).

---

## 3. What is running and what comes next

### W4 — test-time adaptation (running now)

Each clip has its own stadium, lighting, and camera. The detector's internal
normalisation statistics (BatchNorm running means/variances) were computed on the
*training* clips; test-time adaptation re-estimates them on each test clip's own
unlabeled frames before inference — no labels, no gradients, nearly free. This is a hot
technique in the broader ML literature that nobody has applied to GSR. Pre-registered:
killed if the 10-clip probe shows <+0.10; enters the final bundle only at DEV-20
>= +0.5 with 12/20 clips helped. Cheap, novel-for-the-benchmark, dies cheap if it dies.

### W3 — the data track (needs Sid: "a few evenings")

The measured naming gap is evidence starvation, and more model tuning cannot create
evidence — only more supervision can. Three sources, one discipline:

- the 106k SoccerNet-v3 jersey labels (already exploited via the legibility filter),
- GSR-train's own legibility-passed digit crops,
- **Sid's annotation evenings**: labelling at *tracklet* level — one human decision
  propagates to hundreds of frames through our existing anchor-propagation machinery —
  aimed exactly at the measured failure inventory (the unnamed-tracklet pool).

The discipline: **glyph-only supervision** — a crop enters training only if a human (or
a pixel-verified label) says the number is *visible in that crop*. No model-generated
labels for the "unreadable" class: we measured (v7-V3) that such labels teach the
student to imitate its teacher's blind spots instead of learning the world. Retrain the
reader and its fusion on this corpus; gate it like every other component.

### W5 — the endgame

When W3/W4 verdicts are in: leave-one-out ablation of the full bundle, freeze the
configuration to a timestamped JSON, one TEST-38 read (gate: >= 51.0), one test-49 run,
package with the legitimacy audit, and spend **submission #6 of 10**. Pre-declared fail
response: drop the weakest component, re-gate once; if still failing, ship nothing —
53.09 stands. We plan to spend at most 2 of the 5 remaining submissions, keeping
reserve for December.

---

## 4. Why this plan and not something bigger

- **The association gap (+11.9) is the largest prize but the highest risk** — v7
  already measured a transplanted learned associator (CAMELTrack) at −13 to −20 without
  its native appearance features. Attacking association properly means training an
  associator on our own cue basis: that is a thesis-scale effort, noted as future work,
  not a two-week gamble before reviews.
- **Data beats architecture at our scale.** Every large win in this project's history
  was data or a bug (the 106k labels, the aggregation fix, the gate repair); every
  architecture transplant failed its gate. W3 doubles down on the thing that has
  actually worked.
- **The negatives are deliverables too.** For a BTP defence, "we pre-registered five
  ideas, four failed, here is the measured mechanism of each failure" is a stronger
  research posture than an unverifiable win. The claims ledger (~200 entries),
  the GT audit, and the seam experiment are each defensible on their own.

**The December story, as it stands:** (1) the seam claim — the leader's named
bottleneck, fixed by our uncertainty-aware fusion, demonstrated on their own model's
outputs; (2) the curated-corpus story — glyph-only discipline plus human tracklet
annotation; (3) the benchmark audit — five ground-truth errors, two beyond the SOTA
paper's own list; (4) whatever score lands on the board via submission #6.

---

## 5. Open decisions (Sid's)

1. **Organizer report** on the five GT-swapped sequences — recommended (it is this
   project's most legible data contribution, and the SOTA paper corroborates 3 of 5).
   Draft goes out on your word.
2. **Author email** to the Broadcast2Pitch authors asking for a license clarification —
   optional; a permissive answer upgrades what we may build on directly.
3. **Annotation evenings** — W3 starts when you schedule the first one.
