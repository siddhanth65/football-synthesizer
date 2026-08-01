# The evidence-density law for player-event attribution, measured on FOOTPASS

Date: 2026-07-28. Follow-up to `results/IDENTITY_SOLVER_STAGE2.md` (Gate 1 FAIL) and
`results/GTA_LINK_STAGE1.md`. Plan: `docs/ATTRIBUTION_RESEARCH_PLAN.md`, post-survey revision B.

Stage 2 measured that the Lu-style joint solve cannot beat the appearance-only arm **because the
identity evidence is starved** — 8.7% of tracklets carry an OCR read, and the evidence-restricted
oracle caps at 0.365. That is a claim about inputs, and real footage does not let us vary inputs.
This experiment varies them, on top of FOOTPASS's dense per-frame ground truth (54 matches,
**97,397 validated action events**), with every invented component pinned to a number this project
already measured.

Code: `tools/footpass_digest.py` (HDF5 -> compact per-half arrays), `tools/evidence_sim.py`
(simulator + thin adapter onto the **unmodified** `generator.identity_solve`),
`tools/evidence_sweep.py` (77-condition grid, tables, figures), `tests/test_evidence_sim.py`.
Figures: `results/evidence_density/*.png`. Raw metrics: `results/evidence_density/sweep.json`,
per-event outcome vectors under `results/evidence_density/raw/`. CPU only; the GPU was untouched.

**Headline, in one line each.**

1. **OCR-like channel:** attribution reaches precision 0.85 at coverage >= 0.50 at a read density of
   **d = 0.347** (read precision 0.86) or **d = 0.268** (read precision 0.95) — **4.0x and 3.1x
   today's 0.087**. Today's density delivers coverage **0.124** at that precision floor.
2. **Commentary at the pre-declared go/no-go point (1.5 names/min, precision 0.60, lag IQR 4 s):
   VERDICT FAIL, and not marginally.** It buys **nothing**: coverage at precision 0.85 is
   **0.000**, coverage at 0.60 is **0.000**, and full-coverage precision is **0.142** against a
   no-evidence control of **0.139**. Stacked on today's OCR it is actively **harmful** (coverage at
   precision 0.85 falls 0.124 -> 0.042). The cause is measured: at a 4 s lag IQR a mention binds to
   the correct player **13.2%** of the time, against a **12.5%** chance rate.

---

## 1. The substrate: what FOOTPASS actually contains, and one hard ceiling

The HDF5 schema matches `data/footpass/README.md` exactly — no contradiction found, so the
stop-and-report condition did not fire. Verified independently here: 102 half-datasets (96 TRAIN +
6 VAL), **97,397 rows with `class != 0`** (91,327 + 6,070), matching the README's counts to the row.

**Off-screen encoding (asked for explicitly).** `roi_x/y/width/height` are `NaN` — all four together,
never zero-filled, never absent — exactly when the player is not visible in the broadcast frame;
`x, y` tracking coordinates are always present. Off-screen-ness is therefore a clean binary signal
and is what the simulator uses for visibility.

| corpus fact | value |
|---|---|
| half-datasets / games | 102 / 54 (3 CHALLENGE games carry no labels) |
| action events (`class != 0`) | **97,397** |
| players per half (incl. substitutes) | 26.1 (22-32) |
| half length | 74,606 frames = 49.7 min @ 25 fps |
| **player-frames on screen** | **33.4%** (per-half range 0.201-0.517) |
| on-screen run length (frames) | p10 44, p25 80, **p50 164**, p75 327, p90 564, mean 249 |
| on-screen runs per player-half | 97 |
| **actor on screen at the annotated event frame** | **0.8151** |

That last row is a ceiling nothing in this report can move: **18.5% of PCBAS events are committed by
a player who is not in the broadcast frame at that instant.** No tracker, no OCR, no commentary and
no solver can name them from pixels. Every coverage number below is over *all* events, so 0.815 is
the maximum coverage any arm can reach, and the measured full-coverage arms land at 0.81 — i.e. they
already answer for essentially every event that is physically answerable.

## 2. Simulator calibration — the gate, before any sweep

Nothing here is a free parameter. Six components, each fitted to a measured number, then checked.

| component | measured target (source) | simulator achieved |
|---|---|---|
| fragments per 30 s of on-screen time | **4.72** (GSR connector, `GTA_LINK_STAGE1.md` §4) | **4.27** (chunk 3000) / **4.31** (chunk 750) |
| tracklet span distribution (frames) | GSR `tau=0.040` partition: p10 5, p25 33, **p50 147**, p75 407, p90 670, mean 240 | p10 16, p25 44, **p50 102**, p75 219, p90 380, mean 162 |
| appearance top-1 at 4.55 candidates | **0.6366** (Stage 1 A3 LOTO) / 0.6403 (Stage 2) | **0.6413** |
| OCR reads land on longer tracklets | **1.63x** mean span (321 vs 196, measured here on the Stage-2 bundles) | **1.60x** |
| read precision | 0.875 (Stage 2 §2) | **0.859** |
| read density per **merged** tracklet | 0.103 (390/3,790; the 8.7% in Stage 2 is per *original* track) | 0.103 by construction |

The span distribution is the one imperfect fit and it is reported as such: the simulator reproduces
the median and the fragmentation rate but under-represents both tails — it produces no 1-5 frame
detector blips (FOOTPASS visibility runs have p10 = 44 frames) and no 750-frame full-sequence
tracks. Direction of the bias: fewer worthless micro-tracklets *and* fewer very long ones.

### 2.1 The gate: reproduce Stage 2's measured reality

Run at GSR geometry — a **750-frame solve unit**, the same 30 s window `eval/gsr_identity.py`
solves — with the measured operating point (d = 0.103, read precision 0.86, no commentary,
measured fragmentation). At this unit the fragmenter barely cuts at all (fitted span scale 38):
**visibility breaks plus the chunk boundary already produce the measured 4.7 fragments per 30 s of
on-screen time on their own.**

| quantity | Stage 2, measured on GSR | simulator, gate arm | verdict |
|---|---|---|---|
| identity accuracy, solver | 0.2604 (of which ~0.035 is credit for correctly abstaining on GT players GSR leaves unnumbered) | **0.2576** per attributable event / 0.2144 per event | **match** |
| appearance-only arm, units named correctly | 0.136 (= 0.154 coverage x 0.882) | **0.1246** | match (-8%) |
| greedy direct-read arm | 0.154 coverage @ **0.882** precision | 0.110 coverage @ **0.842** precision | precision matches; coverage 29% low |
| evidence-anchored fraction | 0.216 per tracklet / **0.253 per row** | **0.176** per event | 20-30% low |
| no-evidence control | — | 0.133 precision, 0.000 coverage at any floor | sane |

**GATE: PASS.** The two numbers that matter — what the solver scores and what the appearance-only
arm scores — land within 1-8% of Stage 2's measured values, on a completely different corpus, with
no parameter fitted to them. The two that run 20-30% low are both denominator effects: Stage 2's
are **row**-weighted (rows accrue to long tracklets, which are precisely the ones OCR reads),
FOOTPASS's are **event**-weighted. The simulator is therefore, if anything, slightly pessimistic
about how much today's evidence buys.

### 2.2 One thing the gate exposed that Stage 2 did not: the solve window is an evidence multiplier

The same 8.7-10.3% read density anchors **0.176** of events at a 750-frame solve unit and **0.395**
at a 3,000-frame (2 min) unit — because a player has ~4x more tracklets in the window, so ~4x more
chances that one of them was read. Stage 2's 0.365 evidence ceiling is therefore **partly an
artifact of grading on 30 s GSR clips**, not a property of the OCR. Everything below runs at the
2-minute unit, which is the match-scale regime our own pipeline solves in (11 chunks per match in
`tools/identity_match.py`).

## 3. The OCR-like channel — the law

20 TRAIN halves (`sorted(paths)[::5]`, 19,603 events), fragmentation 4.27/30 s, 2-min solve unit.
Coverage is over all events; precision is over answered events; the dial is the posterior of the
identity the solver assigned. Figure: `results/evidence_density/ocr_frontier.png`.

| read density d | read prec | coverage @ prec >= 0.85 | coverage @ prec >= 0.60 | precision at full coverage | anchored | VAL cov@0.85 |
|---|---|---|---|---|---|---|
| **0.087 (today)** | 0.86 | **0.124** | 0.358 | 0.361 | 0.395 | 0.082 |
| 0.15 | 0.86 | 0.222 | 0.542 | 0.476 | 0.557 | 0.219 |
| 0.25 | 0.86 | 0.359 | 0.814 | 0.601 | 0.698 | 0.343 |
| 0.40 | 0.86 | **0.577** | 0.810 | 0.737 | 0.796 | 0.585 |
| 0.60 | 0.86 | 0.796 | 0.810 | 0.847 | 0.823 | 0.802 |
| 1.00 | 0.86 | 0.808 | 0.808 | 0.907 | 0.831 | 0.804 |
| 0.087 | 0.95 | 0.166 | 0.414 | 0.392 | 0.403 | 0.178 |
| 0.15 | 0.95 | 0.278 | 0.626 | 0.519 | 0.565 | 0.262 |
| 0.25 | 0.95 | 0.470 | 0.816 | 0.667 | 0.713 | 0.511 |
| 0.40 | 0.95 | **0.720** | 0.817 | 0.805 | 0.798 | 0.738 |
| 0.60 | 0.95 | 0.819 | 0.819 | 0.912 | 0.825 | 0.813 |
| 1.00 | 0.95 | 0.823 | 0.823 | 0.972 | 0.832 | 0.817 |

**The law.** Linear interpolation on the coverage-at-0.85 column:

> **d\* = 0.347 at read precision 0.86; d\* = 0.268 at read precision 0.95.**
> Held-out VAL gives **0.347** and **0.246** — the law transfers to the 3 unseen games unchanged.

In units that a detector engineer can act on, at our measured fragmentation that is
**0.74 -> 2.96 reads per minute that a player spends on screen** (or 2.29/min if OCR precision is
first raised to 0.95). Raising read precision from 0.86 to 0.95 is worth about a **1.3x** reduction
in the required density — real, but far smaller than the 4x density gap itself. **Volume, not
purity, is the binding constraint.**

Two structural readings of the table:

- **The precision-0.60 floor is cheap and the 0.85 floor is expensive.** d = 0.25 already answers
  81% of events at precision >= 0.60; the same density answers only 36% at >= 0.85. Anything
  downstream that can tolerate a 60%-right name gets it at ~3x today's density; per-player *facts*
  need the 0.85 floor and 4x.
- **Saturation at 0.81 coverage** across every high-density row is the 18.5% off-screen ceiling of
  §1, not a solver limit.

## 4. The commentary channel — verdict on the pre-declared go/no-go point

OCR off, so the channel is measured alone. Every arm below is at lag IQR 4 s unless stated.
Figures: `commentary_frontier.png`, `commentary_alignment.png`.

| names/min | name precision | cov @ 0.85 | cov @ 0.60 | precision at full coverage | effective evidence precision (binding x naming) |
|---|---|---|---|---|---|
| no-evidence control | — | 0.000 | 0.034 | **0.139** | — |
| 0.5 | 0.60 | 0.000 | 0.000 | 0.141 | 0.10 |
| 1.0 | 0.60 | 0.000 | 0.000 | 0.143 | 0.10 |
| **1.5 (the gate)** | **0.60** | **0.000** | **0.000** | **0.142** | **0.105** |
| 2.0 | 0.60 | 0.000 | 0.000 | 0.144 | 0.11 |
| 3.0 | 0.60 | 0.000 | 0.000 | 0.145 | 0.11 |
| 5.0 | 0.60 | 0.000 | 0.000 | 0.148 | 0.11 |
| 1.5 | 0.80 | 0.000 | 0.000 | 0.138 | 0.14 |
| 5.0 | 0.80 | 0.000 | 0.000 | 0.150 | 0.15 |
| 1.5 | 0.95 | 0.000 | 0.000 | 0.143 | 0.16 |
| 5.0 | 0.95 | 0.000 | 0.000 | 0.143 | 0.17 |

**Every commentary-only condition in the grid — 27 of them, spanning 0.5 to 5.0 names/min, 0.60 to
0.95 name precision, 4 s and 8 s lag — is statistically indistinguishable from having no evidence at
all.** Held-out VAL reproduces this exactly (1.5/min at 0.60: precision 0.148 vs no-evidence 0.145).

### 4.1 Why: the alignment, not the channel

The mention itself is fine; the **binding** is the failure. Measured on 3,300 mentions per row over
6 TRAIN halves, with a perfect (precision 1.0) name:

| lag IQR (s) | binds to the named player, uncompensated | ...after subtracting the lag's median |
|---|---|---|
| 0.5 | 0.542 | 0.721 |
| 1 | 0.368 | 0.578 |
| 2 | 0.230 | 0.427 |
| **4 (the gate)** | **0.132** | 0.283 |
| 8 | 0.085 | 0.183 |

Chance is **0.125** (8.0 tracklets alive at an average instant). **At the pre-declared 4 s lag IQR,
binding a commentary name to the tracklet nearest the ball is not distinguishable from assigning it
at random**, because the ball proxy travels a median of **24 m in 4 s**. Multiply by the gate's 0.60
name precision and the channel delivers evidence at precision **0.105** — below chance-corrected
zero — at a volume of **0.0176 reads per tracklet**, one fifth of OCR's 0.087.

Three controls separate "commentary is worthless" from "our binding rule is naive":

| arm (3 names/min, name precision 0.95) | evidence precision | cov @ 0.85 | cov @ 0.60 | precision at full coverage |
|---|---|---|---|---|
| lag IQR 4 s, uncompensated | 0.140 | 0.000 | 0.000 | 0.143 |
| lag IQR 4 s, **median-lag compensated** | 0.277 | 0.000 | 0.000 | 0.195 |
| lag IQR 2 s, compensated | 0.534 | 0.013 | 0.147 | 0.245 |
| lag IQR 1 s, compensated | 0.722 | 0.040 | 0.296 | 0.302 |
| lag IQR 0.5 s, compensated | 0.845 | 0.215 | 0.390 | 0.363 |
| **oracle binding** (alignment solved) | 0.948 | 0.295 | 0.497 | 0.435 |
| *(today's OCR, for scale)* | 0.863 | 0.124 | 0.358 | 0.361 |

- Calibrating out the median lag — which any real implementation would do — roughly **doubles**
  evidence precision and still clears no floor.
- The requirement is **sub-second alignment**: only at IQR <= 0.5 s does commentary reach
  today's-OCR-grade usefulness, and even then it needs 3 names/min at 0.95 precision to do it.
- Even with **perfect** binding, 3 names/min at 0.95 precision reaches coverage 0.295 at precision
  0.85 — better than today's OCR (0.124) but still short of the 0.50 bar. The volume a commentator
  produces is simply small: 3 names/min over a half is ~150 anchors against ~3,800 tracklets.

### 4.2 Commentary stacked on today's OCR makes it worse

| arm | cov @ 0.85 | cov @ 0.60 | precision at full coverage |
|---|---|---|---|
| OCR alone (d 0.087, p 0.86) | **0.124** | **0.358** | **0.361** |
| + 1.0 names/min @ 0.60 | 0.055 | 0.263 | 0.345 |
| + **1.5 names/min @ 0.60 (the gate)** | **0.042** | **0.222** | **0.337** |
| + 2.0 names/min @ 0.60 | 0.000 | 0.166 | 0.323 |
| + 1.5 names/min @ 0.80 | 0.000 | 0.221 | 0.328 |

Monotone damage, dose-dependent, replicated on VAL. The mechanism is visible in the diagnostic
columns: the direct-read baseline's precision collapses from 0.844 to 0.542 as mis-bound names
contaminate the read pool, and because galleries are anchored *by reads*, a wrong name also
poisons the appearance channel for that identity. Low-precision evidence is not free.

**VERDICT on the pre-declared operating point (precision >= 0.60, >= 1.5 names/min, lag IQR <= 4 s):
FAIL.** Not "small gain", not "promising direction" — zero coverage at both pre-declared floors,
indistinguishable from no evidence alone, and net-negative when added to OCR. The gate's own
thresholds are the problem: **it never specified a timestamp-alignment precision, and alignment is
the only variable that matters.** A gate worth running would read: *>= 3 names/min at name precision
>= 0.95 with binding accuracy >= 0.70* (i.e. lag IQR <= 0.5 s after median compensation) — and even
that buys coverage 0.30 at precision 0.85, not 0.50.

## 5. The fragmentation interaction, and the confound in it

Figure: `fragmentation_interaction.png`.

Read density `d` is defined **per tracklet**, so a more fragmented arm silently receives more reads.
That confound inverts the result, and both parameterisations are reported:

| arm | fragments / 30 s on-screen | d | reads / tracklet | precision at full coverage | cov @ 0.85 |
|---|---|---|---|---|---|
| *density fixed per tracklet (confounded)* | | | | | |
| none (1 tracklet per player-chunk) | 0.63 | 0.087 | 0.088 | 0.222 | 0.065 |
| connector (measured) | 4.27 | 0.087 | 0.087 | 0.361 | 0.124 |
| GSR baseline | 5.31 | 0.087 | 0.087 | 0.395 | 0.125 |
| fulham baseline | 6.19 | 0.087 | 0.086 | 0.417 | 0.138 |
| **matched read VOLUME (controlled)** | | | | | |
| none | 0.63 | 0.587 | 0.588 | **0.610** | **0.505** |
| connector (measured) | 4.27 | 0.087 | 0.087 | 0.361 | 0.124 |
| GSR baseline | 5.31 | 0.070 | 0.070 | 0.351 | 0.096 |
| fulham baseline | 6.19 | 0.060 | 0.061 | 0.346 | 0.083 |

Fragmenting a track does not create legible frames, so **read volume per unit of on-screen time is
the physically invariant quantity** and the controlled rows are the ones to believe. On those,
fragmentation is monotonically harmful, and the size is startling:

> **At today's read volume, a perfect connector — one tracklet per player per 2-minute chunk —
> reaches coverage 0.505 at precision 0.85 (VAL 0.501). Today's fragmentation reaches 0.124.**

So there are two roads to the same place, and they are worth about the same: **4x the OCR reads, or
a 7x reduction in fragmentation.** This substantially qualifies Stage 1's negative result. Stage 1
correctly measured that its connector bought tracking association and *nothing measurable* for
naming — but that was measured at 4.72 -> 5.87 fragments, a 24% move on a curve whose payoff lives
between 4.7 and 1.0. The perfect-association arm is an upper bound no tracker will hit; the
exchange rate between the two levers is the usable result.

Commentary is flat at ~0.14 across every fragmentation arm — you cannot fix a mis-bound name by
fixing the tracker.

## 6. The two headline answers

**(1) Minimum evidence for event-attribution precision 0.85 at coverage >= 0.50.**

| channel | requirement | today | factor |
|---|---|---|---|
| OCR-like, read precision 0.86 | **d = 0.347** reads per tracklet (2.96 reads per on-screen minute per player) | 0.087 | **4.0x** |
| OCR-like, read precision 0.95 | **d = 0.268** (2.29 per on-screen minute) | 0.087 | 3.1x |
| commentary-like, lag IQR 4 s | **unreachable at any rate up to 5/min and any precision up to 0.95** | — | — |
| commentary-like, perfect binding | unreachable at 3/min @ 0.95 (reaches 0.295); would need ~5-6/min at >= 0.95 with sub-0.5 s alignment | — | — |
| fragmentation, at today's read volume | **<= ~1 fragment per 30 s on-screen** (perfect within-chunk association) reaches 0.505 | 4.27 | 7x fewer |

**(2) The commentary operating point (1.5 names/min, precision 0.60, lag IQR 4 s): FAIL.** It buys
coverage 0.000 at precision 0.85, coverage 0.000 at precision 0.60, full-coverage precision 0.142
against a 0.139 no-evidence control, and it *degrades* the OCR channel it is added to
(0.124 -> 0.042 coverage at precision 0.85). Whisper-grade ASR at 4 s alignment is noise for this
purpose. The single lever that could change the verdict is timestamp alignment to sub-second
accuracy — which is exactly the problem MatchTime hand-corrected rather than solved.

## 7. Negatives, surprises, and what this simulator cannot say

1. **The premise survives but shifts.** Stage 2's "evidence starvation" is confirmed and quantified
   (4x), *but* a large part of what looked like starvation is dilution: the same reads spread over
   4.3x more tracklets than a perfect associator would need. And Stage 2's 0.365 ceiling is partly
   an artifact of its 30 s solve unit — at a 2-minute unit the same density anchors 0.395 of events
   rather than 0.176.
2. **Higher read precision is a weak lever** (1.3x) compared to volume (4x). Adoption C in the
   research plan (OCR confidence gating) trades volume for precision — on this curve that is close
   to a wash and could be net-negative if the gate discards more than ~25% of reads.
3. **Low-precision evidence is worse than none** when mixed with good evidence (§4.2), which is a
   direct argument against ever shipping the commentary channel at its planned operating point.
4. **The 18.5% off-screen ceiling** is the quietest important number here: even a perfect system
   answers at most 0.815 of PCBAS events from broadcast pixels. FOOTPASS's own baseline (46.41)
   consumes ground-truth game state and so does not pay this cost; any from-pixels number must.
5. **What the simulator cannot say.** It hands the solver the acting player's *tracklet* (an oracle
   gate) — the 0.846 gate-hit factor of `results/CARRIER_CONSTRAINED_v2.md` is not simulated, so
   every number here is a **naming** number and an end-to-end figure would be ~0.85x it. Appearance
   draws are independent across a player's fragments (real embeddings are correlated), team error is
   a flat 5% rather than a measured confusion, and the tracklet-span tails are under-represented
   (§2). The fragmenter also never produces a *contaminated* tracklet (two identities in one track);
   Stage 1 measured 10.96% of real fragments as contaminated, so the simulator is optimistic there.
6. **Conditions dropped for time:** none from the declared grid. Scale was bounded instead — 20 of
   96 TRAIN halves (19,603 events per condition, standard error < 0.005 on precision) and all 6 VAL
   halves. Every headline was checked on VAL and transferred (d\* 0.347 train / 0.347 val).

## 8. Files

- `tools/footpass_digest.py`, `tools/evidence_sim.py`, `tools/evidence_sweep.py`,
  `tests/test_evidence_sim.py` (7 tests, green).
- `results/evidence_density/sweep.json` (77 conditions x 2 splits), `raw/*.npz` (per-event
  outcomes), `report.txt` (all tables), `ocr_frontier.png`, `commentary_frontier.png`,
  `commentary_alignment.png`, `coverage_at_floor.png`, `fragmentation_interaction.png`.
- `data/footpass/digest/*.npz` (102 halves, 100 MB, gitignored). The 8.8 GB TRAIN HDF5 was
  extracted, digested and deleted; peak transient disk was ~9 GB.
