# EFI Alignment — mapping our pipeline to the FIFA Post-Match Summary (and what the product becomes)

**The oracle.** The FIFA "Enhanced Football Intelligence" Post-Match Summary Report (the France 3–1
Senegal PMSR, 16 Jun 2026) is the exact detail level and metric vocabulary we are aiming at. It is
both our **validation target** (do our numbers match FIFA's for a known match?) and the **template**
for the report product. This doc maps what our video pipeline can produce today vs what needs the
ball/event layer, then reframes the report + synthesizer (C5) around it.

## 1. Capability map (FIFA family → our status → source)

| FIFA family (report pages) | Status | Where it comes from |
|---|---|---|
| **Line height & team length**, in/out of possession (pp.6–7, 27–28) | ✅ **have** | `fingerprint.structural_metrics` (width/length) + `direction_metrics` (buildup/def-line height) |
| **Team shape**: compactness, surface area, L/C/R channels | ✅ have (extra vs FIFA) | `structural_metrics.team_shape` (hull area, 5 lanes) |
| **Physical**: speed-zone share, sprints, top speed (pp.50–51) | ⚠️ **rough** | `fingerprint.team_style.physical_profile` (partial-tracking + jitter biased → fractions/maxima only) |
| **Formation** (4-2-3-1 / 4-3-3) (p.2) | ⚠️ derivable | centroid cloud → role/formation inference (not built) |
| **Movement to receive**: In Front/Behind/Out-to-In… (pp.22–23) | ⚠️ partial | `attacker` run/receiver heads classify run *type* relative to ball/goal (needs ball) |
| **Possession %** (p.3) | ❌ needs ball | sparse `is_actor` only (~17% ball recall) |
| **Phases of play**: build-up/progression/press/block split (p.4) | ❌ needs ball + possession | phase classifier (not built) |
| **Line breaks** by unit/line/direction (pp.8–11) | ❌ needs ball + lines | pass tracking + line definitions |
| **Passing networks** (pp.12–13) | ❌ needs ball | pass-event detection |
| **Offering to receive / receptions** (pp.20–21) | ❌ needs ball | requires possession + pass events |
| **Defensive pressure / actions** (pp.25–26, 29) | ❌ needs ball/events | pressure & duel detection |
| **Attempts at goal, crosses, set plays, goalkeeping** (pp.14–19, 31–40) | ❌ needs events | event detection |

**Headline:** the families we own outright are exactly the **spatial/structural** ones — line height,
team length, compactness, territory, channels — plus a rough physical profile. Everything FIFA derives
from **on-ball events** is gated behind reliable ball/event tracking (currently our weakest link).

## 2. Validation opportunity (C6) — we can check our numbers against FIFA's

FIFA prints ground-truth structural numbers we directly compute. For **France** (attacking direction
normalised), FIFA gives:

| phase | FIFA width × length, line | what we compute |
|---|---|---|
| Build-up Mid | 57 × 29 m, line 44 m | `width`, `length`, `buildup_height` |
| Final Third | 46 × 29 m, line 59 m | ″ (final-third phase bucket) |
| Mid Block | 38 × 27 m, line 38 m | `width`, `def_line_height` |
| Low Block | 35 × 25 m, line 19 m | ″ |

Our chunk_000 numbers (width 33–36 m, build-up height 46–53 m, def-line 40–47 m) already land in the
**same range** as these EFI figures — strong external sanity for the metric engine. **C6 task:** when we
have a match with a published PMSR, run our pipeline on its broadcast and report per-phase error vs the
PMSR (the honest accuracy anchor). This needs the **phase split** (§4) to bucket frames like FIFA does.

## 3. What `z_T` captures today

`fingerprint.team_style.team_style_vector` → one interpretable row per team:
shape (width/length/compactness/surface area/5 lanes) + verticality (build-up & def-line height,
attacking-third share) + rough physical (speed-zone share, sprint share, top speed) + indices
(wing/half-space share, L/R bias). Multi-chunk matches: `globalize_chunk_ids` (frame/track ids reset
per chunk) + `align_teams_by_defended_goal` (KMeans labels teams arbitrarily per chunk) make the chunks
combinable within a half. This is the **model-free core of `z_T`**; the relational-GAT read augments it
later (`fingerprint.team_identity`).

## 4. The gap that unlocks most of FIFA: the ball/event layer

Possession, phases, line breaks, passing networks, pressures, offers — **all need the ball and basic
events**. Priorities to close it:
1. **Native-fps ball tracking** (TrackNet/WASB) → possession sequences (who has it, where). Unlocks
   possession %, and a **phase classifier** (build-up/progression/final-third by ball location + team
   block height; press/block by our line height when out of possession — which we already measure).
2. **Pass-event detection** (carrier change + ball trajectory) → passing networks, line breaks
   (intersect pass segment with our learned positional "lines"), offers/receptions. Our `attacker`
   receiver labels are the seed of this.
3. **Phase split** then lets us report line height/width **per phase** exactly like FIFA pp.6–7/27–28,
   and enables the C6 validation in §2.

## 5. The report product (refined) — a *predictive*, EFI-structured two-team analysis

The FIFA PMSR is *post-match*. Our product is the **pre-match predictive** version with the same
layout: for an upcoming `A vs B`, render each EFI section as a **forecast** from the fingerprints
(`z_A`, `z_B`) with calibrated uncertainty. `report/` builds this; sections we can fill **now** (line
height/team length, team shape, physical) ship first; ball/event sections (phases, line breaks,
passing) fill in as §4 lands. Each rendered number is **traceable to a computed metric** (no invented
figures).

## 6. The synthesizer (C5), reframed per the brief — generate a grounded game plan

Not just "predict tendencies" but **write how Team A should play Team B**, grounded in the data:

```
inputs:  z_A, z_B  (FIFA-aligned style vectors)  +  relational GAT reads  +  matchup context
Tier A:  P(tendencies | team)            -- each team's marginal style
Tier B:  P(tendencies | team, opponent)  -- adjusted for the specific opponent (does the opp term help?)
   ->    a GROUNDED generator (LLM) turns the predicted matchup numbers into a tactical report:
         "Senegal sit in a mid-block ~38 m wide with a 38 m line; attack the wide channels where the
          block is narrowest, use in-behind runs vs their 41 m high press; their LB DIOUF over-commits
          to crosses (2) — overload that flank."
```

The **novelty + safety** is grounding: every prescriptive claim cites a number from our deterministic
metrics / GAT read, so the LLM narrates evidence rather than inventing tactics. Headline test stays:
**does Tier B beat Tier A** (does knowing the opponent change the plan), scored with CRPS/Brier +
reliability. **Caveat:** the matchup model needs *many* team-matches to learn the opponent term — one
match (MUN–MCI) only supports the marginal/report path until more matches are ingested.

## 7. Roadmap refinement (priority order)
1. **Ball tracking (native-fps)** → possession + **phase classifier** (the single biggest unlock; gates
   half of FIFA + the C6 validation).
2. **`report/` v1** on the families we own (line height/team length, shape, physical), EFI layout,
   traceable numbers — shippable now on `z_T`.
3. **Pass/line-break/passing-network** detection on the ball track → fill the remaining EFI sections.
4. **More matches ingested** → train the Tier-B matchup synthesizer; then the **grounded game-plan
   generator** (§6).
5. **C6 validation** vs a published PMSR once the phase split exists.
