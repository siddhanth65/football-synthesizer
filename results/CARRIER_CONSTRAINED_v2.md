# Carrier identification as constrained joint assignment (v2) -- NEGATIVE RESULT

Rebuild of `results/CARRIER_ATTRIBUTION_PROBE.md`'s per-moment argmax as elimination + rescoring +
joint assignment, scored against Sid's 90 hand labels. Code: `tools/carrier_constrained.py`
(stages `build` / `tune` / `score` / `jersey` / `jersey_score` / `selftest`). Metrics come from
`tools.score_carrier_labels.score_labels` **verbatim**, so every number below is like-for-like with
the 0.35 baseline.

## Verdict first

**Elimination does not fix a weak appearance model. Per-player carrier attribution is still not
defensible at any useful coverage.**

| configuration | held-out end-to-end precision | n | 95% CI (Wilson) |
|---|---:|---:|---|
| v1 baseline (independent argmax) | **0.360** | 9/25 | 0.20 - 0.56 |
| all HARD constraints | **0.400** | 10/25 | 0.23 - 0.59 |
| frozen tuned config (hard + hubness) | **0.348** | 8/23 | 0.19 - 0.55 |
| jersey-number assignment (alone) | **0.000** | 0/25 | 0.00 - 0.13 |

The best constrained variant moves held-out end-to-end precision by **+0.04 (one moment)** against a
baseline whose own confidence interval is 0.36 wide. Nothing here is statistically distinguishable
from doing nothing. The one configuration that looked like a real win on the tuning split
(hubness correction: dev 0.333 -> 0.500) **did not transfer** (held-out 0.360 -> 0.348) -- a textbook
overfit to 15 assigned dev moments.

Wiring check first: with every constraint disabled the module reproduces v1 **exactly** --
`carrier-selection error 0.154 (12/78), identification precision 0.438 (n=32), end-to-end 0.350
(n=40), abstain rate 0.487`. The rebuild is not a different pipeline that happens to score similarly.

## Why constraints cannot help: the precision decomposition

The label set says end-to-end precision factorises cleanly, and only one factor is an
*identification* problem:

```
end-to-end = P(box on right player) x P(true player is even a candidate | assigned)
                                    x P(matcher picks him | he is a candidate)
```

Measured on the 66 correctly-boxed labelled moments (all 3 matches, frozen operating point):

| factor | value | who can fix it |
|---|---:|---|
| P(box on right player) | 0.846 (66/78) | ball/carrier resolution, not identity |
| P(true player is a candidate \| assigned) | **0.719** (23/32) | gallery completeness, or abstention |
| P(matcher picks him \| he is a candidate) | **0.609** (14/23) | appearance model, or constraints |
| product | 0.438 identification / 0.350 end-to-end | |

`0.609` is the entire space constraints can operate in, and it already matches the gallery-LOTO
top-1 of 0.647 -- PRTreID at ~90 px simply is a ~0.6 identifier. Held-out, the hard constraints lift
this term from 0.643 (9/14) to 0.714 (10/14): **one moment**.

The other 9/32 assignments are *unreachable* -- the true carrier has no gallery entry on the tracked
team, so no re-ranking of the candidate set can produce him. Only abstention helps, and no constraint
built here detects that state.

## Constraint set: what each one is, and what it removed

Hard (candidate deleted from the set) vs soft (rescore only), kept strictly separate in the code.

### HARD

| # | constraint | share of candidates removed (lvp / tot / bhn) | true answer wrongly removed |
|---|---|---|---:|
| 1 | **gallery coverage** | n/a -- true by construction | 0 |
| 2 | **substitution window** | 17.7% / 24.6% / 15.2% | **0 / 45** |
| 3 | **team** | n/a -- already in v1 | (see below) |
| 4 | **mutual exclusion in-frame** | 2.3% / 2.6% / 1.2% | 1 / 45 |
| 5 | **kinematic feasibility** | 1.2% / 0.7% / 0.6% | **0 / 45** |

1. **Gallery coverage was already a hard constraint in v1** and needs no code: the candidate set *is*
   the gallery, so a player with zero crops can never be proposed. The labels' `gallery_covered =
   False -> precision 0.00 (7 assignments)` cell is the **converse** case -- the TRUE carrier is
   missing -- which elimination cannot address by construction. Only abstention can, and the
   territory floor (below) was the one mechanism tried; it failed.
2. **Substitution window** (Sofascore `minutesPlayed` via `tools.make_carrier_labeller.roster_for`,
   estimated minute +-5 min because the broadcast clock is derived from fixed 600 s chunk cuts).
   This is the workhorse of the hard set: it deletes **15-25% of all candidates and never once
   removed the true labelled player**. It is the only constraint with a measurable held-out effect
   (+0.04 end-to-end alone, and it is the whole of the "all hard" gain).
3. **Team was already enforced in v1** -- `identify()` masks the gallery to the carrier's per-track
   majority team. Confirmed: that is why the observed errors are all within-team. It is *not* free of
   defects, though: on the labelled moments the tracked team disagrees with the true player's real
   team on **5 of 30 reads (17%)**, and the team restriction is exactly what drops the true player
   from 51/66 gallery-covered moments to **45/66 reachable candidates**. Six of the twenty-one
   unreachable moments are caused by our own team label, not by a thin gallery.
4. **Mutual exclusion in-frame** and 5. **kinematic feasibility** are one mechanism (`anchor_flags`):
   a candidate is deleted if the identity chain confidently placed that player on a *different* track
   within 3 s and the implied speed to the carrier exceeds 10 m/s (2 m of tracking slack; `dt == 0`
   is the mutual-exclusion case). They are **safe but tiny** -- together 1.8-3.5% of candidates, with
   one false elimination in 45, and no measurable effect on precision.

Second-order effect worth noting: eliminating candidates *widens* the top-1/top-2 margin, so the hard
stack **increases** assignments at a fixed operating point (394 -> 425 across the three matches,
+7.9%) at flat precision. If anything ships, this is the defensible part: same precision, ~8% more
coverage, and two provably-impossible-assignment classes closed.

The hard stack changes **75 of 1095** carrier answers; 5 of those fall on labelled moments and show
the whole mechanism honestly:

| match | truth | baseline | all-hard |
|---|---|---|---|
| tottenham | WRONG_PLAYER_BOXED | Christian Eriksen | *abstain* |
| tottenham | Noussair Mazraoui | Marcus Rashford | *abstain* |
| brighton | Harry Maguire | Manuel Ugarte | **Harry Maguire** |
| brighton | Andre Onana | *abstain* | Yasin Ayari |
| brighton | Bruno Fernandes | *abstain* | Alejandro Garnacho |

The substitution window kills exactly the Eriksen-in-the-first-half error the v1 probe flagged
label-free, and one wrong answer becomes right. But the same margin-widening that buys coverage also
converts two safe abstentions into two new wrong assignments. Net: +1 correct out of 25 held-out
assignments.

### SOFT (rescore only)

| prior | dev e2e (n) | held-out e2e (n) | verdict |
|---|---|---|---|
| none (hard only) | 0.333 (15) | 0.400 (25) | reference |
| territory, w=0.01 | 0.375 (16) | 0.385 (26) | flat |
| territory floor 2.5 (abstain) | 0.385 (13) | 0.417 (24) | flat, and unsafe (below) |
| hubness correction | **0.500** (10) | **0.348** (23) | **overfit; did not transfer** |
| joint (Hungarian, possession) | 0.357 (14) | 0.375 (24) | flat |
| everything on | 0.455 (11) | 0.333 (18) | worse |

6. **Positional / role plausibility.** Per `(player, half)` pitch-position profile built from that
   player's own confident identifications (mean + sigma floored at 5 m, other half mirrored),
   candidate scored by Mahalanobis distance from the carrier's actual pitch position. This was the
   prior aimed at the CB->winger confusions and it is the **most disappointing result here**: a
   floor at 2.5 sigma removes ~48% of candidates but **false-eliminates 12 of the 45 true answers
   (27%)**. The profiles are built from the same hero-shot-biased named tracks as the gallery, so a
   defender who was only ever named during two attacking set pieces has an attacking territory. The
   prior is not wrong in principle; the evidence it is estimated from is too thin and too biased to
   act on. As a soft weight it moves held-out precision by -0.015.
7. **Joint assignment over a possession.** Carrier moments sharing a chunk + team within 8 s are
   solved together with a Hungarian assignment, so two different tracks in one possession string
   cannot both be the same player. Effect: -0.025 held-out. The reason is structural and was already
   visible in v1 -- **carrier tracks are ~1 pass long** (333 of 344 dev tracks carry exactly one
   pass), and possession strings that survive the ball-coverage funnel rarely contain 2+ resolvable
   carriers. There is very little joint structure left to exploit.

**Hubness correction** (not in the brief; added because "Bruno Fernandes has 151 gallery crops and
others have 6" is a mechanical argmax bias): subtract each player's mean similarity over all queries.
On dev it is by far the biggest mover (0.333 -> 0.500). On held-out it is *negative*. Reported in
full because it is the clearest illustration of how little 15 dev assignments can decide.

## Tuning protocol and its (small) honesty

Tuned on `manutd_liverpool`'s 30 labels only; the grid is 36 points over
`w_territory x territory_floor x w_hubness x joint` with all hard constraints forced on, rule
"max dev end-to-end precision subject to >= 8 dev assignments", written to
`results/carrier_attr/frozen_constrained.json`, then applied unchanged to the 60 held-out labels.
Full grid: `results/carrier_attr/constrained_dev_sweep.csv`.

**The dev split has 29 judged moments and 15 assignments.** A 36-point grid over 15 outcomes selects
noise, and it did: the frozen point is the single worst-transferring cell in the grid. Quote only the
held-out column.

## Coverage / precision trade-off (held-out labels; coverage over all 3 matches' PASS events)

| min_sim | min_margin | config | held-out e2e | n labelled | 95% CI | assignments (of 3020 PASS) | coverage |
|---:|---:|---|---:|---:|---|---:|---:|
| 0.84 | 0.00 | baseline | 0.277 | 47 | 0.17-0.42 | 991 | 32.8% |
| 0.88 | 0.00 | baseline | 0.295 | 44 | 0.18-0.44 | 815 | 27.0% |
| 0.88 | 0.01 | baseline | 0.360 | 25 | 0.20-0.56 | 394 | 13.0% |
| 0.88 | 0.01 | **all hard** | **0.400** | 25 | 0.23-0.59 | **425** | **14.1%** |
| 0.90 | 0.01 | baseline | 0.450 | 20 | 0.26-0.66 | 322 | 10.7% |
| 0.90 | 0.02 | frozen | 0.556 | 9 | 0.27-0.81 | 214 | 7.1% |
| 0.92 | 0.01 | baseline | 0.455 | 11 | 0.21-0.72 | 224 | 7.4% |
| 0.92 | 0.01 | frozen | 0.667 | 9 | 0.35-0.88 | 241 | 8.0% |
| 0.92 | 0.02 | frozen | 0.750 | 4 | 0.30-0.95 | 150 | 5.0% |
| 0.94 | 0.01 | baseline | 0.600 | 5 | 0.23-0.88 | 133 | 4.4% |

Full grid: `results/carrier_attr/constrained_curve.csv`.

The curve slopes the right way -- precision rises monotonically-ish as coverage falls -- but **the
90-label set cannot resolve an operating point**. By the time nominal precision reaches 0.6-0.75 the
labelled sample is 4-9 moments and the interval spans 0.27-0.95. The honest statement is: *if* the
high-threshold points hold, ~5-8% of passes could be attributed at ~0.6-0.7 precision; verifying that
needs roughly 200 more labels drawn **at that operating point**, not at this one.

## Jersey numbers: fails, and fails SILENTLY

The previous worker flagged number recognition as "likely to beat appearance ReID outright" because
"it fails loudly rather than silently". **Both halves of that are wrong on this footage.**

Setup: the best available checkpoint (`outputs/jersey/jersey_torso_r224_acc417.pt`, Stage-1c torso
crop, SoccerNet test tracklet accuracy 0.417 / numbered-only 0.30), run on up to 5 crops of the
carrier's own track nearest the kick, aggregated with the module's own confidence-weighted tracklet
vote, and decoded under a **roster mask restricted to the carrier's tracked team** -- so a number
maps to exactly one rostered player.

| min_conf | carriers | number read | held-out e2e | all-3 e2e | n labelled |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 1095 | 44.4% | **0.000** | 0.054 | 37 |
| 0.20 | 1095 | 43.1% | **0.000** | 0.057 | 35 |
| 0.40 | 1095 | 26.8% | **0.000** | 0.048 | 21 |
| 0.60 | 1095 | 10.8% | 0.000 | 0.000 | 7 |
| 0.80 | 1095 | 2.8% | -- | -- | 0 |

On the 66 correctly-boxed labelled moments a number is emitted for 30, and **2 of 30 match the true
player's shirt (0.067)**. Held-out: **0 of 25**. Raising the confidence floor does not help -- the
error rate is flat while coverage collapses.

The failure is silent by construction: the roster mask guarantees the model lands on a number some
player on that team actually wears, so every read *looks* valid. The reads collapse onto a handful of
low numbers -- on `manutd_liverpool`, 33 reads of `20`, 30 of `10`, 21 of `11` out of 136 -- which is
the exact low-number prior collapse `results/jersey_model/JERSEY_MODEL.md` diagnosed on SoccerNet,
amplified by a
~90 px domain the model never saw. The v1 report's observation that numbers are "legible on several
players at exactly this zoom" was a human reading a montage; this recognizer cannot do it.

Artifacts: `results/carrier_attr/<match>_jersey.parquet`,
`results/carrier_attr/jersey_carrier_scoring.csv`.

## Verdict on shipping per-player attribution

**Not defensible at any coverage worth having.** Concretely:

* At the frozen v1 operating point (13-14% of passes attributed) the honest precision is
  **0.36-0.40 with a CI of 0.20-0.59**. Six in ten named passes are wrong. No per-player claim
  survives that.
* The constrained rebuild is worth keeping only for the parts that are *free and safe*: the
  substitution window and the exclusion/kinematic checks delete 17-27% of the candidate space,
  never removed a true answer in 45 chances, close two classes of provably-impossible output, and
  buy ~8% more assignments at unchanged precision. Ship those; do not claim a precision gain.
* The remaining error is **not** in the assignment logic. It is (a) a ~0.6-accuracy appearance model
  at 90 px, (b) a gallery that cannot represent 21 of 66 labelled carriers, (c) a per-track team
  label that is wrong 17% of the time, and (d) 15% carrier-selection error before identification
  starts. Constraints touch none of them.
* The next lever with real headroom is therefore **not** a better matcher and **not** jersey numbers.
  It is candidate reachability: 0.719 x 0.609 is the product to attack, and 0.719 (gallery + team
  label) is both the larger deficit and the cheaper one.

## Caveats

* 90 labels, 78 judged, ~25 assigned per split. Every precision here has a +-0.18 interval. The
  tuning split has 15. Treat all deltas below 0.15 as unmeasured.
* Held-out is 2 matches, not 2 independent samples of football; match-level effects
  (0.273 tottenham vs 0.429 brighton at baseline) are larger than every constraint effect measured.
* The territory prior, the exclusion constraint and the kinematic constraint all trust the identity
  chain's named tracks, whose own read precision is marked UNVERIFIED upstream. One false elimination
  in 45 is consistent with that, and it is the mechanism by which those constraints could silently
  cost precision at larger scale.
* Minute estimation assumes fixed 600 s chunk cuts starting at kickoff; +-5 min slack absorbs it, but
  the substitution constraint is correspondingly loose (a tighter clock would remove more candidates).
* The jersey result is one checkpoint. It is the best one we have, and it was never trained on this
  broadcast domain; a recognizer fine-tuned on PL wide-shot crops is untested here.
