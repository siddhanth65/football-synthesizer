# Review crib sheet — the seven attacks and your answers (2026-07-15)

Grill-session outcome (Sid + orchestrator). Each entry: the attack a reviewer will make → the
answer you chose. Memorize the *shape*: concede what's true, state the measured boundary, pivot to
the validated strength. Companion: `docs/CV_EXPLAINER.md` (mechanics), `docs/BTP_DECEMBER_PLAN.md`
(plan), `results/brighton_cv_vs_oracle.md` (the worked example).

## Q1 — "Your report can't see goals. Why trust it?"
Positional geometry cannot distinguish a shot from a pass — both are a carrier releasing the ball.
We pre-declared that boundary, measured it, and gate every claim behind validated evidence. The
report's value is structural reads no media report has (de-biased line to 7.2 m vs FIFA, lane
occupation, counterpress curves) with abstention where evidence is thin. Trust comes FROM the
refusal to fake events. (Layer 3 action-spotting is planned — trained on SoccerNet labels,
validated vs Sofascore counts — as capability, not as an excuse.)

## Q2 — "Your possession picks the wrong leader. Why publish?"
It measures a different quantity — trackable-frame share; tracking fails precisely in transitions,
so direct teams get under-counted. We PROVED uncorrectability: 3 pre-registered corrections, LOMO
vs 4 oracle matches, all rejected; missingness informative in every zone. Renamed, caveated,
out of gates. **Then pivot to the strength:** Sofascore says Utd had 52% and lost; we measure WHY
it didn't help — final-third tilt 39/28 against them, xT share 52/48, Brighton counterpress faster
at every horizon (62% vs 52% at 5 s). Sterile possession, measured. Every absent stat gets
partnered with a rendered structural strength.

## Q3 — "You scored ~X on GS-HOTA; SOTA is 63.9."
GS-HOTA = localization × ALL-OR-NOTHING identity (jersey+role+team). We emit no jersey numbers yet
— by design, pre-Layer-2 — so identity zeroes the composite. Read the localization component
(competitive) and treat the composite as our external BASELINE: when jersey ID lands, the lift is
measurable on a public benchmark instead of self-graded. We score the valid split; 63.9 is the
challenge split — stated, not hidden. Never chase the leaderboard; labs optimize the benchmark,
we use it as an anchor.

## Q4 — "Comparative tier = moving the goalposts."
The gate didn't move — the CLAIM changed. Absolute claims still require 50% recall and still
abstain. Ratios between teams need a different sufficiency condition: symmetric capture (47.8 vs
48.6, spread 0.009 ≤ pre-committed 0.05) — like comparing two polls with equal sampling rates.
Decision dated + logged before implementation; every section carries the disclosure label.
Goalpost-moving is silently relabeling failure; this is a documented weaker claim type behind a
pre-declared bar.

## Q5 — "Jersey numbers are 13 px. How do you name anyone?"
Two sources vote: close-ups (26% of frames, discarded for geometry, legible numbers 30-38% incl.
surnames) read by number-model/VLM — what 2025 GSR winners did; plus a SoccerNet-trained model
voting per-TRACK on wide play. The hard part is NOT digits — it's carrying identity across camera
cuts: tracks churn 750-1,045 ids/chunk for 22 players, so cross-cut re-ID (appearance embeddings +
team/role/position constraints) is the research problem. Names ship behind a per-player confidence
gate with abstention; validated vs Sofascore per-player stats.

## Q6 — "C5 is a tautology on 8 points."
Concede accurately: today's C5 is explanatory (opponent depth explains tendency variance, LOMO
+48% over team-average), not a deployable forecast — we said so in writing the day we verified it.
Upgrade is mechanical + scheduled: predict opponent line from THEIR prior matches (n→30+ with the
season corpus), forecast from the prediction, same LOMO protocol. n=8 stays a proof-of-signal,
never significance. May deliverable: pre-match seam forecasts scored on held-out matches.

## Q7 — "StatsBomb/SkillCorner do this better. What's novel?"
(1) Independence: any lawful broadcast becomes analysis — methodology is the product.
(2) The thesis is uncertainty-aware tactical inference: pre-registered gates,
proof-of-uncorrectability, validated bias correction, abstention as output — FIFA's own study says
off-screen estimation fails industry bars; nobody grades commercial trackers this honestly.
(3) Beyond-published-SOTA axis: GSR winners impute off-screen players with linear interpolation
[verified 3-0, arXiv 2504.06357]; a validated probabilistic imputer exceeds published practice —
and we hold the free validation assets (SkillCorner tracking, SB360 same-censoring freeze frames).

## Numbers to have cold
| # | Meaning |
|---|---|
| ~37% / 80.5% | usable-geometry yield: whole broadcast / live-wide conditional |
| 16.4 → 7.2 m | line height raw → de-biased, HELD-OUT (5.5 m = in-sample; say "held-out" first) |
| 47.8 / 48.6 / 0.009 | pass recall Utd / Brighton / symmetry spread — the comparative-tier license |
| 44.2 vs 52 | our possession share vs Sofascore — the bias exhibit, leader inverts |
| 51.9% / 71.2% | post-link ball coverage: full match / live-wide conditional |
| 14.8 / 43.1 / 48.9 / 92.5 | OUR GS-HOTA valid-split: official full / no-jersey / loc+assoc / LocA — jersey gap measured externally |
| 29.01 / 63.90 | GSR baseline / SOTA (challenge split — context, not like-for-like ranking) |
| 12.76 / 22 | players visible on average in broadcast (why imputation is THE problem) |
| 100% | guardrail precision — no ungrounded number reaches a reader |

## The one-sentence thesis
*Uncertainty-aware tactical inference from partially observed broadcast video: measure exactly
what the broadcast supports, prove where it stops, forecast only what survives a held-out test,
and report nothing that cannot be grounded.*
