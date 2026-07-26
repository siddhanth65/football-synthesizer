# What this project is — the one-page definition (2026-07-26)

Written because the work had fragmented into modules and the through-line stopped being obvious.
This document is the answer to "what is your project?" and it supersedes any framing that
contradicts it. Audience: two technically strong examiners at IIIT Delhi, December 2026.

---

## The one-sentence version

**We turn ordinary TV football footage into tactical measurements that are graded against external
truth, and use them to ask which parts of a team's playing style are genuinely the team — and which
are just the opponent.**

## The research question (the thing being asked, not built)

> Football clubs are described as having a "DNA" — a way of playing. From broadcast video alone,
> **which style dimensions are stable properties of the team, and which are artefacts of who they
> played, where, and the state of the game?** And did Manchester United's measurable identity change
> when the manager changed?

This is answerable with what we hold and it has a real answer, including negative parts. It is not
a claim that we can read a club's soul off the TV; it is a measurement of how much of "style" is
durable at all.

## Why that question is defensible (the design that makes it work)

The corpus is deliberately built as a **repeat-measurement design**: 12 Manchester United matches,
EPL 2024-25, arranged as **six home/away pairs against the same six opponents**, straddling the
ten Hag -> Amorim managerial change. Measuring the same team against the same opponent twice is what
lets us separate "this is United" from "this is what Brighton did to United".

## The three contributions

**1. Engineering — a validated broadcast-to-metrics pipeline.**
Video -> player/ball detection -> tracking -> homography to a 105x68 m pitch -> team assignment ->
possession -> events -> player identity -> a gated report. What makes it a contribution rather than
an integration exercise: **every layer is graded against an external source**, and the report layer
**abstains** when the evidence is thin instead of guessing.
- Pass counts from the event model: within **0.97-1.09x official** on 10 of 12 matches, on a
  threshold frozen at the first match and never retuned.
- Goals: 14/16 detected at the right time, every half-boundary independently confirmed.
- Player identity on a public benchmark (SoccerNet GS-HOTA): **14.76 -> 22.85**, externally graded.

**2. Method — validated off-screen player imputation (the novel module).**
TV shows ~11.8 of 22 players. Published practice (SoccerNet challenge winners) fills the missing
players by straight-line interpolation and never measures the error. We:
- built a camera simulator over full-pitch ground truth, calibrated to reproduce our real visibility;
- **pre-registered** the bar (beat the best causal baseline) before building the model;
- when a better baseline appeared mid-way, **raised our own bar** rather than keep the easy one;
- **passed**: 8.10 m vs an 11.46 m bar, with 50%/90% uncertainty regions that hold **6/6**;
- **transfer-tested** against a camera footprint *measured* from real broadcast data, where the gain
  holds at 105-112% — and stated the aspect range where it would not.

**3. Empirical — what is and is not a team's identity.**
Findings so far, all measured, including the negative ones:
- **Possession control is stable.** Across repeat meetings, r = **+0.83**. It is predictable: a
  pre-match forecast from one prior meeting hits **7.92 pp MAE**, beating an uninformative baseline
  (10.63) and Elo (12.18).
- **Defensive line height is NOT stable** (r = **-0.40**) — it swings with the opponent.
- **Results are opponent-determined**: no home/away pair flipped win<->loss; venue effect ~null.
- **A single-team "style fingerprint" degrades as matches are added** (separation 2.0 -> 0.74 -> 0.09)
  — the honest verdict is that much of what looks like identity at n=3 is opponent and game-state.
- **The manager change is visible in players, not in team shape**: 7 of 9 players common to both eras
  sit higher up the pitch under Amorim (Garnacho +15.1 m), while the block-height metric does not
  cleanly separate the two managers.

## The demo (the artifact an examiner can touch)

A **pre-match opposition scouting pack** for a fixture we have played twice: generated using only
information available *before* the second meeting (the first meeting's broadcast analysis + public
season data + Elo), then **scored against what actually happened**.
- **Brighton pair = the primary demo.** Brighton's manager was stable across both legs, so it is a
  fair test of the design.
- **Southampton pair = the honest stress case.** Southampton changed their own manager between the
  legs, so the pack is run to the point where the evidence gate refuses. *What the system declines
  to say, and why, is the thesis claim being demonstrated live.*

## What we deliberately do NOT claim

- No per-player statistics (passes/touches per player). Player-level event attribution is **1-4%** of
  truth: naming a passer needs the ball tracked, a carrier within 3 m at the kick moment, **and** that
  track named — and those conditions rarely coincide. Stated as a limit, not hidden.
- No set-piece analysis. We hold corner counts and nothing else.
- No claim of accuracy parity with commercial tracking providers.
- No claim that imputation numbers are reliable on our own footage yet — our tracker fragments too
  much; that is written down as the next dependency, not glossed.

## Why an examiner should find this credible

1. Every headline number is reproducible from the repository, and one was independently re-derived
   from raw source data and matched to the last digit.
2. Bars were written down before results, and when a harder bar appeared we adopted it.
3. Negative results are reported as findings (style fingerprint decays; attack-direction forecasting
   is no better than chance; a published method we implemented **lost** to our baseline).
4. There is a public correction trail: we published a claim ("Southampton pressed high"), discovered
   two teams were mislabelled, retracted it loudly, corrected nine artifacts, and added an automatic
   screen so the error class cannot recur.
5. The system refuses to answer when it cannot ground an answer — and we demo that refusal.

## One-line thesis statement

*Measure exactly what the broadcast supports, prove where it stops, forecast only what survives a
held-out test, and report nothing that cannot be grounded — then use that instrument to ask how much
of a football team's "identity" is real.*
