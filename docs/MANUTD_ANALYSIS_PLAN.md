# ManU analysis plan — research-backed build order (2026-07-18)

Scope pivot (Sid, 2026-07-17): Manchester United, EPL 2024-25. Demo = opposition scouting pack.
This doc turns Sid's product questions into a concrete, cited build plan. Sources: the xG FC corpus
distillation (92 articles in `football-state-of-play/docs/xg_club_articles/`, digest in the
2026-07-18 research sweep) + verified web research on lineup-prior identity and ball-action
spotting.

## The reframe that anchors everything

Independent validation of COMMERCIAL broadcast trackers (arXiv 2508.19477): position RMSE
0.44-2.23 m, players detected in only **36-64% of frames on standard broadcast**; 19+ players fully
visible only **~5% of the time** (EPL gameweek study). SkillCorner's own methodology disclosure:
identity comes from **lineup priors + appearance/jersey fusion**, not per-match training. Our
coverage numbers are the industry regime, not a defect — the fix is the same one the industry uses:
priors + event models + extrapolation, with our differentiator being honest uncertainty labels.

## Three pillars

### Pillar 1 — Lineup-prior identity (assignment, not recognition)
Team sheets are public pre-match (names, shirt numbers, starting positions, subs + minutes — all in
the Sofascore oracle we already cache). Identity becomes a **bipartite assignment**: 22 known
players x our tracks, costs = jersey-read posterior (Koshkina votes, 97.5% sample precision) +
formation-position prior (shape-graph tactical positions) + kit/team + GK flag; solve Hungarian
(arXiv 2110.11107 formulation; GSR winners fuse the same signals — arXiv 2504.06357).
Sub events (known minutes) re-key the assignment mid-match.
- Target: most of 22 assigned per match with per-player confidence, vs 20 open-set names today.
- Validation: per-player touches/shots vs Sofascore player stats (Bruno-style), per match.

### Pillar 2 — Event layer via spotting models (counts from events, geometry for context)
Pass counts from possession-chain geometry land at ~30% of truth (coverage artifact — measured).
Counts must come from **event spotting**, where we already validated the approach (E2E-Spot:
3/3 goals, shots 25/25, cards exact).
- **Ball Action Spotting**: adopt **lRomul/ball-action-spotting** first — 1st place 2023,
  **86.47% mAP@1**, public weights (verified in hand 2026-07-18, ~53 MB, 6.8M params, 4 GB-easy).
  CORRECTION (2026-07-18): the 2023 task/winner is **2-class: PASS + DRIVE** — the 12-class list
  previously here belongs to the 2024 task (T-DEED). Pass/drive is precisely the pass-count fix;
  richer event classes already come from the validated E2E-Spot layer (shots/goals/cards/corners).
  **T-DEED (2024, 12-class, checkpoints public, VRAM unverified) = the upgrade probe.**
  **sn-teamspotting** (2025 task) adds team attribution per action — what attribution needs.
- Probe protocol (validated-or-nothing): pass counts vs Sofascore on brighton + liverpool +
  fulham. If BAS can't get close on our footage, that is the finding.
- Then: **event time x ball position x Pillar-1 assignment = player-action ledger** (who did what).

### Pillar 3 — Style + situation analytics (the actual product)
From the xG FC distillation, **Tier-1 = tracking-native, works on our data now** (no pass stream
needed, ball-gap tolerant):
1. **OT sliced-Wasserstein style embedding** (Baouan et al. 2025) — team fingerprint from player
   positions alone; team identity recognisable at 82% top-1; ~70% from 300 frames. Our per-match
   ManU signature + "how did Liverpool distort it" comparisons.
2. **Shape graphs -> 5x5 tactical positions** (npj Complexity 2025) + **phase-of-play CNN**
   (Bauer/Anzer/Shaw 2023: build-up / attack / high-mid-low block) — build-up channel
   (wing vs central), block height, width, per-phase formations, coach fingerprint.
3. **Counter-press primitive** (StatsBomb defs): pressure = defender within 5 yd of carrier;
   counter-press = pressure within 5 s of turnover. Computable from tracks + possession today.
4. **Line-breaking passes + SBR** (arXiv 2506.06666): defender-band clustering (doubles as block
   height), LBP detection, space-build-up ratio — per-player once Pillar 1 lands.
5. **Score-state conditioning** (Kent-Brandes 2025 Cox survival + the standard score_diff/time/
   red-card covariates): every metric sliced by level/chasing/leading + stoppage-time — Sid's
   "who turns up when losing" becomes a query. Goal spots (validated) give the score state.
6. **Possession-archetype fingerprint** (verified: mixture of marked spatio-temporal point
   processes, arXiv 2511.14297): cluster possessions into archetypes (fast transition / contested
   carry / patient build-up...) — team style = cluster proportions, trackable across matchdays
   (how ManU's mix shifts when losing, vs top-6, home/away).
Tier-2 (needs Pillar-2 pass stream): xT 16x12 grid + action credit, PPDA (passes/defensive actions,
x>40), VPEP, passing networks + motifs, decision rDP/gDP.

## Build order (each step validated before the next leans on it)

B-1. Lineup module + assignment layer (Pillar 1) — re-run brighton, liverpool. [worker: opus]
B-2. BAS probe (lRomul weights) — pass/action counts vs Sofascore, 3 matches. [GPU probe]
B-3. Player-action ledger (events x assignment) — per-player validation vs Sofascore. [opus]
B-4. Style fingerprint v1: OT embedding + shape/phase + counter-press + possession archetypes,
     computed per match for all processed matches; ManU identity doc with cross-match consistency.
B-5. Score-state segmentation of everything + the Liverpool 0-3 case study (how the press/shape
     collapsed, phase by phase — the "how do they get outplayed" exhibit).
B-6. Scouting report v2: narrative-first (style, seams, players, situations), CV-vs-oracle
     validation moved to appendix. HTML renderer already exists.

## What Sid's questions map to
- "why no shot tracking" -> solved (E2E-Spot validated); wired in B-2/B-3.
- "passes half of Sofascore" -> B-2 (count from events, not geometry).
- "player identification does nothing" -> B-1 + B-3 (assignment + action ledger).
- "style of play: wingers/central/counter/low block" -> B-4 (phase CNN + channels + archetypes).
- "how do they lose / win" -> B-5 (score-state + case studies).
- "team-level metrics and identity" -> B-4 fingerprint, stable across matches = identity.
