# Prof-meeting brief — where the BTP stands and where it should go

*Prepared 2026-08-14 for Monday's meeting. Companion docs: `GSR_JOURNEY_14_TO_53.md` (how we
got to 53.09), `GSR_NEXT_PLAN.md` (the just-closed v8 campaign), `STATUS.md` (day-by-day log),
`knowledge/claims.json` (228 versioned claims, every number in this brief traceable).*

---

## 0. The 30-second version

We hold **53.09 GS-HOTA** on SoccerNet Game State Reconstruction (~4th on the public test
board; leader 61.48). The gap to the leader is not mysterious: it is **one subsystem —
tracklet association — worth a measured +11.9 ceiling** that we deferred as too large. Two
things changed this week: (1) a literature sweep showed the association attack is **much
cheaper than we thought** (small, coordinates-only learned associators from the DanceTrack
lineage, trainable in hours-to-days on the college A100s), with a **genuinely novel angle
nobody has published** — association in *metric pitch coordinates* instead of image
coordinates; (2) a full scrape of the SoccerNet 2026 ecosystem found the 2026 challenge
season is over (closed April 25), but **two servers accept submissions today** — including
a nearly-empty leaderboard (PCBAS, 6 teams ever, SOTA 58.94 F1) whose input format is
literally our pipeline's output format.

**Proposed plan: (A) build the world-coordinate associator and push our own GSR score toward
the leader on the still-live test board — this is the thesis's novel method; (B) enter PCBAS,
the live low-competition challenge that our jersey-identity work is purpose-built for;
(C) monitor the 2027 season opening (~Sept-Nov) for SynLoc, the task that fits our measured
strengths best.**

---

## 1. What we have (the honest asset sheet)

**The score.** 14.76 → 53.09 in seven weeks of gated work, five public submissions, each
step frozen-then-verified. Full history in `GSR_JOURNEY_14_TO_53.md`.

**The method discipline** (this is a thesis asset, not overhead): pre-registered pass/fail
bars written before every experiment; DEV/TEST/challenge split hygiene; paired per-sequence
statistics; a 228-claim versioned ledger including retractions; measured noise floors
(training-seed spread = 0.0059, established this week).

**Original findings already banked:**
1. **Benchmark audit** — six ground-truth errors in SoccerNet-GSR, five side-swapped
   sequences with a *diagnosed annotation-process cause* (second-half clips carrying
   first-half side conventions); two errors beyond what the SOTA paper itself reports.
   ~2.45 of our score loss is benchmark noise; corrected-GT score ≈ 55.5.
2. **The seam experiment** — the leader's paper names its own primary bottleneck (majority-
   vote jersey aggregation); our confidence-weighted fusion beats it **on their own model's
   outputs** (+0.107 tracklet accuracy, p=0.0017).
3. **The purity result** — first human per-crop audit of a jersey-legibility gate (n=2,520;
   from Sid's 499-tracklet annotation): the gate admits ~27% noise, yet **filtering that
   noise makes the reader worse, even at matched corpus size** (p=0.002, 4.3× seed noise).
   A clean, noise-floored negative that speaks to the noisy-labels literature.
4. **A priced graveyard** — five v7 + three v8 pre-registered failures, each with a measured
   mechanism (e.g. test-time adaptation fails *because* the benchmark has no domain shift;
   transplanted trackers fail *because* their appearance basis can't be reproduced).

## 2. Why not 61 — the direct answer

Under direct measurement, three of the leader's four claimed edges are **ours already**:
our jersey reader covers 10× more tracks at 0.90+ precision, our camera calibration is 12%
*more* accurate (0.475 m vs 0.536 m median), and their team-assignment module is weaker than
ours. Their 61.48 has also never been publicly reproduced (the one attempt logged in their
own repo reached 48.32).

What remains is **association**: keeping each player's track unbroken through occlusions and
camera cuts. Our oracle measurement: perfect tracklet merging is worth **+11.9 GS-HOTA**; we
realise ~43% of it. Every cheap and medium-priced idea has now been tried under registered
gates — v6 sits near the ceiling of its current architecture. The remaining points require
the one component we never rebuilt. That was a deliberate scheduling call when it looked like
a multi-month build; the DanceTrack literature has since shrunk it (see §4A).

## 3. The 2026 landscape — verified this week

The 2026 season **closed April 25, 2026**; the results paper (arXiv:2607.07320, July 2026)
reports 427 teams / 1,129 entries across five tasks. What accepts submissions **today**:

| Server | Status | Notes |
|---|---|---|
| **GSR test phase** | live (our 5 submissions, Aug 1-6) | where 53.09 sits, ~4th; 5 of 10 lifetime submissions left |
| **PCBAS validation + challenge** | open, no end date | 6 teams ever; SOTA 58.94 macro-F1; challenge phase capped at 10 lifetime submissions |
| **SynLoc test phase** | open, no end date | 2026 winner hit 97.67 mAP-LocSim; open phase = sandbox for 2027 |
| Ball Action Anticipation benchmark | open since July 2026 | poor fit (pure temporal forecasting) |
| VQA / NVS / FIFA Skeletal | closed | VQA saturated at 98% by Gemini scaffolding |

A 2027 season is unconfirmed but the pattern (servers open Sept-Nov, deadline April) plus the
organizers' stated intent implies openings within ~1-3 months — i.e. **a live 2027 entry
would be in flight exactly at the December review**.

## 4. The three paths

### A. GSR: the world-coordinate associator — the thesis's novel method

**The idea.** The best current associators for visually-identical targets (DanceTrack: dancers
in identical outfits — the same-kit problem) are small, *appearance-free*, coordinates-only
models: TWiX (a pairwise transformer over box trajectories), MOTIP (in-context identity
prediction), SUSHI (hierarchical graph). All published work runs them in **image coordinates,
where broadcast camera panning corrupts every motion cue.** Our calibrated pipeline can feed
them trajectories in **metres on the pitch — camera motion removed, physically bounded speeds
and accelerations, plus sparse high-precision jersey evidence as long-range edges.** Nobody
has published this input space.

**Why it's credible:** it attacks our largest measured headroom (+11.9 ceiling) with the cue
basis we are best at producing; it sidesteps the exact failure mode of our CAMELTrack
transplant (appearance-basis mismatch — these models use none); the models are small
(hours-to-days per training on one A100).

**Cost & risk:** 6-8 weeks including gated evaluation; the live GSR test board scores it
publicly; realistic landing 56-60+ (honest range — the +11.9 is an oracle ceiling, not a
promise). Fails safe: even a partial gain is a defensible novel-method chapter.

### B. PCBAS: the live contest we are pre-adapted for

Player-Centric Ball Action Spotting: for each on-ball action, predict *when*, *which team*,
*which jersey number*, *what action* — a prediction counts only if **all of them** are right.
That gating is GS-HOTA's identity logic at event level, i.e. the problem we spent two months
mastering. The dataset **provides tracklets and 22-player game-state as inputs** (it is
GSR-shaped data — our weakness partially neutralised by the organisers), Sid already holds
the SoccerNet NDA, and the field is nearly empty: **6 teams ever, winner 58.94 F1, servers
open indefinitely.** New work needed: temporal action spotting from video (a new but
well-documented subsystem; the dev kit ships three baselines). Discipline note: the challenge
phase allows only 10 lifetime submissions — all development on the 100-submission validation
phase first, exactly our GSR constitution.

**Realistic goal: a top-3, plausibly top-1, position on a live leaderboard by December.**

### C. SynLoc: the 2027 build

Single-frame world-coordinate player localization — our strength triad (detection,
calibration, metric localization) with our weakness (tracking/identity) surgically absent.
The 2026 podium all detected in image tiles then ray-cast; **nobody detects directly in
world space** — the autonomous-driving BEV literature (BEVHeight, MVDet lineage) says that
works when the ground plane is known, and even the winner's frame accuracy capped at ~82%.
Our shelved test-time-adaptation machinery also finally meets a real domain gap (synthetic
players composited onto real backgrounds). But the 2026 edition is closed at 97.67, so this
is a **2027-season play**: prototype on the open sandbox server, enter when the season opens.

## 5. Recommendation

**A then B, C monitored.** September-October: build and gate the world-coordinate associator
(Path A) — it is the novel method the thesis needs, it answers "why not 61" by *doing the
missing thing*, and its output (better tracking + identity) is also PCBAS-relevant substrate.
November: PCBAS entry (Path B) for a live leaderboard position at review time. If the 2027
season opens meanwhile, decide then whether SynLoc replaces or follows PCBAS. Budget-wise all
of it fits the college cluster (Path A trainings are hours each; we have used <10 of ~60
budgeted GPU-hours to date).

**The thesis narrative this produces:** Chapter 1 — a 53.09 system built with pre-registration
discipline, plus a benchmark audit the SOTA paper corroborates. Chapter 2 — a novel
world-coordinate association method attacking the measured bottleneck, scored on the public
board. Chapter 3 — a live-challenge campaign (PCBAS/SynLoc) transferring the stack. The
negatives stay in as evidence of method, each with mechanism and noise floor.

## 6. Answers to the direct questions

- **"Why can't we at least reach 61?"** — Because one subsystem (association) was never
  rebuilt; everything else is measured at-or-above the leader. It is now the explicit target,
  and cheaper than we believed (§4A).
- **"What's our problem finding innovative ideas?"** — Not finding them: eight pre-registered
  ideas were built and tested in v7/v8; most failed *informatively* (that is what the gates
  are for). The correctable fault was aiming them at saturated components instead of the one
  expensive gap — this brief fixes the aim.
- **"What are the CV problems here?"** — (1) association of visually-identical small targets
  (DanceTrack literature); (2) tiny low-resolution text recognition (scene-text literature —
  our purity result contributes here); (3) single-frame camera calibration from pitch
  markings (sports-field registration); (4) sim-to-real transfer (AD/BEV + TTA literatures,
  relevant to SynLoc).
- **"Did we scrape the ecosystem properly?"** — Partially fair criticism: our deep recon
  covered the GSR-era ecosystem, and the 2026 lineup was noted but never ingested. That is
  now done (this week's sweep: all six 2026 tasks, rules, dev kits, results paper, live
  server status via the codabench API), and monitoring the 2027 openings is a standing task.
