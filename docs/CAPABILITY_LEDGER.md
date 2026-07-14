# Capability ledger — what this system can and cannot measure (2026-07-11)

The honest inventory. Every row is either **validated** (checked against an external oracle),
**unvalidated** (we produce it, nobody has checked it), or **absent** (we do not produce it at all).
Written so nothing in a report, a demo, or a review can be over-claimed. Companion to
`docs/PL_PIVOT_PLAN.md` (roadmap) and `STATUS.md` (living log).

---

## 1. What "honest reclassification" meant (the possession story, in plain terms)

We used to print a number and call it **"possession %"**. We proved it is not possession %.

Our number is computed only on frames where the ball was tracked *and* a well-calibrated player was
near it. Those frames are **not a random sample of the match** — tracking succeeds during slow,
settled, wide-camera play and fails during fast transitions, long balls, aerial duels and scrambles.
So a team that circulates patiently gets **over-counted**, and a team playing direct gets
**under-counted**. This is systematic bias, not noise.

Evidence (Brighton 2-1 Man Utd): official possession Man Utd 52%. Our number: 44%. Man Utd sat below
their true share **in every third of the pitch** — so the missing possession is not concentrated in a
zone we could reweight; whole untrackable spells are simply absent. We tried three principled
corrections (zone post-stratification, chunk-time weighting, spell gap-extrapolation), validated
leave-one-match-out against 4 oracle-backed matches: **all three rejected**
(`results/possession_debias_probe.md`).

So we **renamed the metric to what it actually is** — *trackable-frame possession share* — kept it as
a descriptive, caveated figure, and **removed it from any pass/fail validation gate**. That is the
whole of the "reclassification": calling the number by its true name instead of a name it had not
earned. It is not a failure; it is the boundary of what broadcast video supports, measured.

**The only real fix is an event layer** (touch/possession events), not more geometry.

---

## 2. The ledger

### STRONG — position-only, survives poor ball coverage
These need no ball, so they render on every match regardless of ball-track quality. This is the
system's spine.

| Capability | Status | Evidence |
|---|---|---|
| Player detection + tracking | Validated (visually + density) | ~12 players/frame on PL (vs 7.8-10.9 on WC) |
| Pitch calibration — *solve rate + quality* | Validated | **91.8%** of detection-frames solve a homography, and those solutions are excellent: median keypoint reprojection **0.21 m**, 99.2% under 2 m |
| Pitch calibration — **usable geometry yield** | **~37% — and this is a property of broadcast, not a bug** | Only ~37% of detection-frames produce player pitch coordinates. Cause (measured, `results/pose_carry_probe.md`): **90.3% of the geometry-less frames are CLOSE-UPS** (median 4 players detected vs 11 on good frames) — you cannot reconstruct a tactical view from a shot of one player's face. Only **4.5%** are genuinely bad poses. **The ~37% yield ≈ the live-wide-play fraction of the broadcast.** Every downstream metric is computed on it. Pose-borrowing has a hard ceiling of +9.3 pp and was measured at +1.7 pp — **the plumbing is exhausted.** |
| **Defensive line height (headline validated result)** | **Validated vs FIFA** | **raw 16.4 m → de-biased 5.5 m** pooled error, 3 WC matches (re-measured 2026-07-11; the previously quoted "17.1 → 5.2" was stale) |
| Team identity (which team a player is on) | Validated per match | jersey-colour clustering + cross-chunk anchoring; Man Utd=red confirmed visually |
| **Defensive line height** | **Validated vs FIFA** | visibility-de-biased; see the headline row below (16.4 m → 5.5 m) |
| Block width / depth / compactness | Unvalidated (plausible) | no oracle publishes these |
| Lane occupation (wing/half-space/centre) | Unvalidated (plausible) | internally consistent across matches |
| Space control / territory | Unvalidated | correlates with possession but is not possession |
| Velocity synchrony, physical (speed/distance) | Unvalidated | derived from tracks; oracle has physical data for WC only |
| In-play formation (e.g. "3-5-2") | **Unvalidated — and currently suspect** | our PL read (3-5-2) disagrees with the official lineup (4-2-3-1). In-play shape *can* legitimately differ from the team sheet, but we have never checked this. **Open item.** |

### PARTIAL — ball-dependent, coverage-gated, proxy-quality
Real signal, but every number inherits ball-track coverage and none is an event-grade measurement.

| Capability | Status | Honest description |
|---|---|---|
| Ball tracking | Validated as detection | v6 detector 87% held-out recall; **post-link usable coverage 51.9%** (Brighton full match) |
| Passes | **Proxy, ~48% recall** | we detect *carrier hand-offs*, not tagged pass events. No pass outcome, no pass type, no intended receiver. Recall is **symmetric** between teams (47.8% / 48.6%), so *relative* pass features are usable; absolute counts are not |
| Possession | **Biased — caveated only** | see §1. Not a validation metric |
| PPDA, ball-xT, verticality, line breaks | Proxy | all derived from the pass/possession proxies — they inherit the same ceiling |
| Pressing intensity, counterpress curve | Proxy | time-to-intercept geometry; plausible, never validated against an oracle |

### ABSENT — we do not produce these at all
This list is the honest answer to "are we getting tackles, shots, runs?"

| Capability | Reality |
|---|---|
| **Tackles / interceptions / duels / blocks** | **Not implemented. Zero detection.** Nothing in the pipeline distinguishes a tackle from two players being near the ball |
| **Shots** | **Not implemented.** We cannot distinguish a shot from a pass — both are a carrier releasing the ball |
| **xG** | Impossible without shots |
| **Player identity from video** | **Not implemented.** No jersey-number OCR, no face/player re-ID. See §3 |
| **Off-ball runs as tactical events** | We have raw movement (speed, direction, distance — real). "A run" as a labelled event is not detected or validated |
| Set pieces | Heuristic candidates only (ball-speed based); flagged unreliable in the fact store |

---

## 3. Player identification — the honest position (answering "are we doing jersey/player ID?")

**No. We have team identity, not player identity.**

- What we *do*: cluster jersey **colour** (CIELAB chrominance) to split the 22 players into two teams,
  anchored consistently across chunks and halves (`generator/teams.py`, `generator/team_anchor.py`).
  Plus a persistent **track id** per player (a number like `track_17`) that survives while the tracker
  holds them — it is not a person, it is a track.
- What we *do not* do: read jersey **numbers**, recognise faces, or link a track to a named human.
- Where the names in the France reports come from: a **curated roster JSON** (`data/france_roster.json`)
  plus FIFA PMSR — the report even discloses this in its footer ("Player names: FIFA PMSR"). The names
  are attached to *prose*, not to tracked players. **We have never claimed a tracked player is a named
  person, and we must never start.**
- Consequence: on the PL corpus (no roster), `report_v2` correctly degrades to **team-level prose**.

**This is the single biggest gap between what we have and a "pundit report".** A pundit says
"Bruno drops between the lines"; we can currently only say "the left half-space is occupied 49% of
the time". Closing it requires **jersey-number recognition** — which is a solvable, well-defined CV
problem with public labelled data (SoccerNet has a jersey-number task).

---

## 4. Data routes (answering "can we get better data?")

Two different needs, often confused. Keep them separate.

### (a) Oracle / validation data — SOLVED, no action needed
- **Sofascore via ScraperFC** (the user's proven route from `../mufc-rodri-search`) is our oracle:
  possession, passes attempted/completed, shots, shots on target, xG per fixture, cache-first.
  Cross-validated against cached FBref on the Brighton fixture: **exact agreement**.
- FBref direct scraping is **403-dead**; the soccerdata/selenium route is a slow fallback only.
- **Worth extending (cheap, high value):** Sofascore also exposes **per-player match stats** and
  likely **shot maps with coordinates + per-shot xG**. Per-shot data would let us validate a future
  shot detector; per-player stats would let us validate per-player metrics — *but only once we have
  player identification* (§3). These two upgrades are coupled.

### (b) Training labels — THE REAL GAP
We do not need better *validation* data. We need **labelled video** to train the things we cannot do
(events, jersey numbers). Options:

| Source | What it gives | Verdict |
|---|---|---|
| **SoccerNet** (research access) | **Action spotting** (500 matches, 17 action classes, 300k labels), **jersey-number recognition**, **camera calibration** (20k labelled images), game-state reconstruction | **The unlock.** Free for research. Use it as a *training corpus*, then run the trained models on our PL footage |
| StatsBomb Open Data | Full event data, free — but **no PL 24/25** | Useful for method prototyping only |
| Understat | Shot-level events + xG, free, PL covered | Cheap add for shot validation |
| Opta / StatsBomb / Wyscout (commercial) | Everything | **Ask the prof** — an institutional/academic licence would change the project's ceiling |
| Our own annotation | Video-aligned event benchmark | Needed regardless, as the *test set* (see PL_PIVOT_PLAN Phase B) |

**Key strategic point:** an earlier external analysis suggested *abandoning* this project for a
SoccerNet benchmark task. That is the wrong trade. The right move is to **use SoccerNet as labelled
training data** for the two missing models (jersey numbers, event spotting) and **apply them to our
own Man Utd corpus** — keeping the pipeline, the validation discipline, the fact store and the
gated-report architecture, which are the actual assets.

---

## 5. The four layers (the "from the basics" plan)

Everything below is ordered by dependency, not by ambition.

**Layer 0 — Foundation (DONE).** Tracking, calibration, team anchoring, ball detection+linking,
carry-over, structural metrics, fact store, coverage gating, numeric guardrail, oracle module,
validation discipline. This is real, and it is more than most undergraduate CV projects ship.

**Layer 1 — Live-play filter (cheap, but RE-PRICED — read this carefully).** Raw broadcast is only
~41% live wide-camera play. Filtering the rest out gives **honest denominators** — it stops us
reporting "we only reconstruct 28% of frames" when the truthful statement is "we reconstruct most of
the frames that *contain a tactical view*". But it is now proven that it **cannot create data**: the
missing frames are close-ups, which contain no tactical geometry to recover (see the calibration row
above, and `results/pose_carry_probe.md`). **It will NOT by itself push pass recall past the 50%
gate.** Do it for reporting integrity, not for capability.

**The plumbing is finished.** Detection, tracking, calibration, ball detection, linking, carry-over,
de-biasing — every lever has been pulled and measured end-to-end. The broadcast does not contain more
tactical geometry than we already extract. **From here, every gain requires a learned model trained
on labelled data.** That is Layers 2 and 3, and it is what the rest of the BTP is.

**Layer 2 — Player identity (jersey-number recognition).** Train on SoccerNet's jersey dataset, run on
our corpus, link number → squad list → name. Unlocks: named-player metrics, per-player validation
against Sofascore player stats, and the actual *pundit* report. This is the difference between a shape
analyser and a football analyst.

**Layer 3 — Event spotting (tackles, shots, duels).** Train on SoccerNet action spotting; validate
detected event *counts* against Sofascore aggregates and against our own hand-labelled benchmark.
Fixes the biggest hole in the ledger (§2, ABSENT) and is the only real cure for the possession bias.

**Layer 4 — The synthesizer, at scale.** With ~38 Man Utd matches (≈76 team-match observations vs
today's 8), the opponent-conditioned model becomes learnable rather than anecdotal. LOMO-backtested,
with the counter-structure seams turned into *forecasts* scored against held-out matches.

Layers 1 + 2 are Semester 1 (December review). Layers 3 + 4 are Semester 2 (May review). The GPU
cluster matters from Layer 2 onward.
