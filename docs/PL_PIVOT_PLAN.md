# Premier League pivot — BTP roadmap (2026-07 → 2027-05)

**Context.** The project becomes a 12-credit BTP (8+4) at IIIT Delhi, reviewed twice: **end of
December 2026** (semester 1) and **May 2027** (semester 2). The prof has agreed that footage scarcity
is the binding constraint and endorsed a pivot to the **Premier League Archive (full 90-minute
replays, 2024/25 season)**, starting with **Manchester United 24/25**. College **GPU cluster** access
is likely. The WC2026 France work is finished first (report v2 = the closing deliverable) and becomes
the *validated methodology* the PL work scales.

This plan is ours — the Codex/GPT analysis was read for reference only. Where it suggested abandoning
the project for a SoccerNet benchmark, we explicitly do NOT: the pipeline, the validation discipline,
and the fact-store/guardrail architecture are the asset. What changes is the corpus and the oracle.

---

## Why PL footage fixes the right problem (and what it can't fix)

Three of our four hard blockers are **corpus-size or footage-quality** problems, which the pivot
addresses directly; one is **physics of broadcast** and no footage fixes it:

| Blocker (measured, WC corpus) | PL archive effect |
|---|---|
| C5 synthesizer has 8 team-match observations — can't learn opponent-conditioning | 38 league matches for one team → **~76 team-match observations**, same-production footage |
| Ball detector needed per-corpus fine-tune (domain gap, v5 lesson) | ONE production style across the season → one fine-tune amortizes over 38 matches |
| Norway-style homography failure (min-6 correspondence yield) | Consistent main-camera production should raise calibration yield — **must be measured, not assumed** |
| Ball-following camera hides off-ball players / cuts hide events | **Not fixed.** Stays a partial-observability limit; handled by gating + abstention (report v2's mechanism) |

Club football also removes two WC frictions: stable squad + fixed kit per match (team anchoring,
roster mapping get easier), and every opponent appears twice (home/away pairs = natural
repeat-measurement validation we never had).

**The oracle changes.** No FIFA PMSR for PL. Replacement validation stack, in order of strength:
1. **FBref / Understat public match-level aggregates** (possession, passes completed, shots, xG,
   pressures-era proxies) via `soccerdata` — the new "PMSR-lite" for every match, free.
2. **Self-built video-aligned event benchmark** — stratified clips, human-verified labels
   (pipeline proposes candidates, human confirms; ~the same annotation muscle as the 510-frame ball
   effort), double-annotated subset with agreement stats. This is itself a citable deliverable.
3. **Home/away repeat-measurement consistency** — the same team measured twice under the same
   methodology should agree within error bars; a validation axis that needs no external data at all.

Licensing boundary (standing rule, unchanged): the user supplies footage from their lawful Archive
access; local processing for academic analysis only; never redistribute footage or scrape streams.
The thesis ships **derived metrics and annotations**, never video.

---

## Phase 0 — close out WC France (now, ~1 week)

- Report v2 (CV-primary, gated, FIFA-as-oracle appendix) on all 3 France matches; senegal = demo.
- STATUS/docs consolidated so the WC chapter is a self-contained "methodology validated" story:
  line height 17.1→5.2 m vs FIFA, phase MAE 13.4 pp, honest event verdict (55/22.8/10.6% recall),
  guardrail 100% precision. This chapter is Review-1 evidence that the method works before scaling.

## Phase A — feasibility gate (~2-3 weeks, before ANY scale work)

Take **3 lawful full-match PL replays** (Man Utd matches, ideally different opponents/stadia) and run
the exact WC probes on 2-3 wide-camera chunks each. **Pre-committed gates** (decided before
measuring, per project discipline):

| Probe | Gate (vs WC baseline) |
|---|---|
| Calibration/homography yield (frames with ≥6 correspondences) | ≥ iraq's yield; norway-style failure <20% of chunks |
| Player detection + track stability | ≥ WC corpus |
| Ball recall on ~100 hand-labelled frames (pretrained v5, NO new fine-tune yet) | ≥ 40% zero-shot → fine-tune will clear 60%+ |
| Post-`link_ball` coverage (the ONLY coverage number that counts) | ≥ senegal's 46% after fine-tune projection |
| FBref aggregate agreement on one fully-processed match | possession within ~5 pp |

**Decision point with prof:** gates pass → Phase B. Gates fail on calibration → the honest fallback
is scoping semester 1 to shape/structural metrics only (they survive low ball coverage). Either way
the gate table itself is Review-1 material.

**2026-07-11 result — Phase A effectively PASSES.** Footage: 1080p @ uniform 25 fps, ~6.1 Mbps,
one production style. The raw ≥6-corr yield (27%) initially failed the gate, but diagnosis
(`results/pl_probe/diagnosis/DIAGNOSIS.md`) proved a sampling artifact: raw broadcast slices are
only ~41% live wide-camera play, whereas the WC baseline was measured on hand-picked wide-play
chunks. **Live-play-conditional yield: 68% pooled (all 3 matches pass); PnLCalib solve 81% vs iraq
88%.** Lesson folded into the method: **Phase A gates apply to play-filtered footage**, and Phase B
ingestion needs a live-play/broadcast-content filter stage (the residual failure modes — midfield
line-symmetric shots and frame-edge projection loss — are handled by temporal homography reuse in
production). Zero-shot post-link ball coverage 20.2% (vs senegal 10% pre-fine-tune → 46% post).
Pending before Phase B: ~100 hand-labelled frames for true zero-shot recall, then the one-time PL
detector fine-tune; FBref agreement check rides on the first full-match process.

**2026-07-11 (later) — Phase A CLOSED: PASSED.** v5 zero-shot recall on the user's 120 labels:
67.8% (gate 40%). v6 fine-tune: 87% PL held-out. Carry-over lever proven end-to-end (brighton full
match 51.9% mean post-link coverage — project record). Oracle rebuilt FBref-independent
(`tools/oracle.py`, Sofascore primary via the user's mufc-rodri-search route; FBref live = 403).
Final gate: pass-volume recall proxy ~48% both teams, symmetry 0.009 → PASS; possession estimator
reclassified as "trackable-frame share" and dropped from pass/fail (diagnosed bias:
`results/pl_pilot/possession_diagnosis.md`). Amendments for Phase B: (1) ingestion adds a
live-play/broadcast-content filter stage; (2) oracle gates = pass volume + (later) event benchmark,
possession caveated-only; (3) facts.py time metrics now per-chunk-fps (METRICS_VERSION 2026.07.3).

## Phase B — scale infrastructure (Aug–Oct 2026) → Review 1 (Dec 2026)

Semester-1 deliverable: **"Manchester United 24/25 — a season-scale, validated tactical fingerprint
from broadcast video"**, 10-15 matches processed.

1. **Registry generalizes to club football** (`data/matches.yaml`): per-match kit colours, opponent,
   home/away, competition; Man Utd roster (one squad, whole season — far easier than WC).
2. **One PL detector fine-tune** on the cluster: ~500 annotated frames across 3-4 matches
   (the proven v5 recipe: rehearsal mix with WC corpus, held-out split, post-link measurement).
   4 GB-laptop rules stay for local work; heavy jobs move to the cluster.
3. **Batch ingestion**: chunking → tracking → calibration → ball → fact store, resumable
   (regen_ball's `--resume-from` pattern), one match end-to-end first, then batches.
4. **Validation at scale**: every match's fact store scored against FBref/Understat aggregates;
   error bars from N matches (a thing we could never do with N=3). Home/away consistency check.
5. **Review-1 demo**: season dashboard + per-match report v2 + the WC validation chapter.

## Phase C — the synthesizer proper (Jan–May 2027) → Review 2 (May 2027)

With ~2×15-38 team-match observations, C5 graduates from pooled-linear-demo to the actual product:

1. **Opponent-conditioned model**: P(Utd style metrics | opponent fingerprint), leave-one-match-out
   backtested across the season; honest comparison vs season-mean baseline (the C5 protocol, at a
   sample size where it means something).
2. **Counter-structure seam forecasts**: pre-match "where Utd will be exploitable / can exploit"
   predictions, scored against what actually happened in the held-out match — the falsifiable
   version of pundit punditry.
3. **Event layer, if Phase A/B ball numbers clear the ≥50% recall bar on PL footage** — reopened on
   evidence, exactly as it was parked on evidence. Plus the self-built event benchmark (2b above).
4. **Review-2 demo**: pick a real upcoming/held-out fixture → generate the pre-match forecast report
   → show the post-match scoring of last season's forecasts.

## Risks & mitigations

- **Archive quality/availability** ≠ assumed: Phase A measures actual replay bitrate/resolution
  before committing. If 24/25 full replays are region-gated, prof's data-access help is ask #1.
- **Oracle weaker than PMSR** (aggregates only, no phase splits): mitigated by the 3-layer stack
  above; phase-level validation carries over from the WC chapter's FIFA-validated methodology.
- **Cluster access slips**: the v5 fine-tune ran on the 4 GB laptop; slow but proven. Cluster is an
  accelerator, not a dependency.
- **Scope creep**: the two review dates are the spec. Anything not on a review's deliverable list is
  parked by default.

## Asks for the prof (concrete, not "help")

1. Scope sign-off on the two-review deliverable split above.
2. GPU cluster account + storage quota (~200 GB for 15 matches of chunked footage + parquets).
3. A second annotator (any labmate) for the double-annotated event-benchmark subset.
4. Guidance on whether the thesis claim should be framed as *uncertainty-aware tactical inference
   from partially observed broadcast video* (our report-v2 gating story) — the defensible framing.
