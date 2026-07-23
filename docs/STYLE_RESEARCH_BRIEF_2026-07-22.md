# Style-analysis deep research — brief (Sid, 2026-07-22 evening)

Sid's ask, verbatim intent: individually go through ALL articles on thexgfootballclub.substack.com
ONLINE (not just the 92 archived locally in the READ-ONLY sibling repo
`football-state-of-play/docs/xg_club_articles/`), then find similar sources — substacks, coaches,
clubs, institutes, university research papers — at the intersection of tech / computer vision /
football tactics-formations / player data analysis. xG FC translates academic papers for readers;
find the papers BEHIND that genre too.

## Goal

Find methods BETTER than ours for a **stylistic analysis of Manchester United** that builds a
proper understanding of them as a team (extensible to other teams later):
- every action, every tactic, every formation, every structure
- coaching style, player-player interactions
- game-state understanding: low blocks, quick counters, possession domination, quick passes
- CV methods we can use or improve for better game-footage analysis

## What we already have (the bar to beat — do not re-recommend what we run)

- OT sliced-Wasserstein style embedding (Baouan 2025) — territorial only, identity decays with n
- Rule-based 4-phase segmentation; StatsBomb counter-press primitive (4.57 m / 5 s)
- Score-state segmentation from validated goal boundaries; possession-archetype plan (mixture of
  marked spatio-temporal point processes, arXiv 2511.14297)
- E2E-Spot 17-class events (goal recall 100%/13; precision 81.3%); BAS lRomul 2-class PASS/DRIVE
  (pass counts 0.97-1.09x Sofascore, frozen threshold)
- Lineup-prior Hungarian identity assignment; PRTreID re-ID; jersey OCR (Koshkina 86.13%)
- GS-HOTA 22.85 valid split; shape graphs -> 5x5 tactical positions planned (npj Complexity 2025);
  phase CNN (Bauer/Anzer/Shaw 2023) planned; line-breaking passes/SBR (arXiv 2506.06666) planned

## Deliverables (write to results/, one file each)

1. `results/STYLE_RESEARCH_XGFC.md` — xG FC online catalogue: every article found online, one-line
   method + source paper + applicability-to-us verdict (esp. articles NOT in the 92 local ones)
2. `results/STYLE_RESEARCH_SOURCES.md` — the wider source map: substacks/blogs (e.g. analytics
   FC-adjacent), coaches' writing, club/institute research groups (e.g. DFB-Akademie, KU Leuven,
   Liverpool/Brentford-style data teams), university labs, key recurring authors
3. `results/STYLE_RESEARCH_METHODS.md` — THE synthesis: ranked methods better than ours per axis
   (formations/structure, phases/game-state, player interactions, coaching style, CV/footage
   analysis), each with: paper/source, data requirements (does it run on OUR broadcast tracking
   with its censoring?), expected lift over what we have, integration cost, and a build/skip
   recommendation. End with a proposed United style-analysis v2 architecture.

## Constraints

- Research only — no pipeline code changes from this task.
- Validated-or-nothing lens: a method that needs full-pitch commercial tracking must be flagged as
  such (we have censored broadcast tracking, ~11.8/22 visible).
- France/WC material is reference only; scope is Man Utd EPL 24-25.

## Status

- [x] xG FC online catalogue complete (2026-07-23: 105 posts, ~50 newer than local archive)
- [x] wider source map complete (107 sources, 6 modalities)
- [x] methods synthesis complete (19 build-recs adversarially verified: 8 survive, 10 corrected/refuted, 1 verify lost to connection error)
