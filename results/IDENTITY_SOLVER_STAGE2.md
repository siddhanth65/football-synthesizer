# Stage 2 — the global identity solve (Lu-style joint inference), measured

Date: 2026-07-28. Plan: `docs/ATTRIBUTION_RESEARCH_PLAN.md` Stage 2. Input partition: Stage 1's
connector at `tau = 0.040` (`results/GTA_LINK_STAGE1.md`, 80.1% GT-audited merge precision).
Method source: Lu, Ting, Little, Murphy, *Learning to Track and Identify Players from Broadcast
Sports Videos* (TPAMI 2013) — per-image identity 50-55% -> 85-89% inside a joint solve.

Code: `generator/identity_solve.py` (`SOLVER_VERSION = "identity-solve-1.0"`),
`eval/gsr_identity.py` (GSR gates), `tools/identity_match.py` (real matches),
`tests/test_identity_solve.py` + `python -m generator.identity_solve` (self-check).
Solver: `scipy.optimize.milp` (HiGHS). **No new dependency** — OR-tools and PuLP are not installed;
scipy 1.17 was already here. CPU only.

**Headline: GATE 1 FAILS.** On the held-out 38 sequences the solver moves per-row identity accuracy
from **0.2559 to 0.2604 (+0.0045)**; the pre-declared paired-per-sequence test does not separate the
arms (Wilcoxon **p = 0.566**, 20 helped / 18 hurt, worst -0.229, best +0.193). Gate 2 passes:
GS-HOTA **23.37 -> 24.19** on the same sequences, no regression, 24 helped / 14 hurt (p = 0.243).
The reason is measured below and it is not the solver: **only 24.1% of prediction rows belong to a
player whose number OCR ever read**, so the evidence ceiling for any method fed by this OCR is
**0.365** identity accuracy, and the appearance link that would spread those reads runs at 0.640
top-1 against a mean of 4.6 candidates.

---

## 1. Three blockers, found before building, none improvised around

**B1 — there is no GSR train or test split on this machine.** `data/soccernet/gamestate-2024` holds
58 sequences (SNGS-021..059, SNGS-078..096): the **valid** split. The plan's "fit on train/valid,
evaluate once on the public test split" is therefore not executable as written. Options priced:
(a) download the train + test splits and re-run detection, calibration, tracking, Koshkina OCR and
the PRTreID per-detection pass over them — on the measured per-sequence costs that is a multi-hour
GPU job per stage, i.e. a day; (b) declare a partition of the 58 in code, fit on one part, run the
other once. **(b) was taken and is stated in `eval/gsr_identity.py`:** `DEV = sorted(seqs)[::3]`
(20 sequences), `TEST` = the remaining 38. The frozen config was written to
`results/identity_solver_config.json` *before* TEST was scored, and TEST was scored once.
Caveat that must travel with this: the input tracklet partition (`tau = 0.040`) was itself chosen in
Stage 1 with the full 58 in view, so TEST is clean for the Stage-2 weights only. Both arms consume
the identical partition, so it cancels in the paired comparison.

**B2 — per-crop OCR output was never persisted, so the digit confusion prior is estimated at the
number level.** `outputs/gsr/koshkina_jersey/*.json` holds one aggregated `(number, confidence)` per
track, not the per-crop softmax. Recovering per-crop reads means re-running the Koshkina STR chain
over 79,053 crops (GPU). The prior was therefore fitted from **(true number, read number)** pairs on
DEV: 129 reads, 15 of them wrong, 214 aligned digit observations. That is enough to see the shape
(`9 -> 0` 0.176, `4 -> 6` 0.125, `7 -> 9` 0.086 after add-1 smoothing) and nowhere near enough for a
10x10 matrix; the smoothing dominates, and the ablation in §6 confirms the prior is inert.

**B3 — SoccerNet-GSR labels no goalkeeper numbers at all.** 77 of 77 GT goalkeeper tracks across the
58 sequences carry `jersey = null`. The GK-naming defect this stage is meant to fix therefore cannot
be exhibited *or* graded on GSR; on GSR the correct action for a keeper is exactly the abstention
the baseline already produces. Gate 4 is measured on the real matches only (§7).

A fourth, smaller one: `core.registry` carries no lineups (`roster: false` for every PL fixture, and
`data/france_roster.json` is France-only). The real-match roster, shirt numbers, positions and
substitution availability come from `outputs/identity/<match>_lineup_assign.parquet`.

## 2. What the evidence can possibly support (measured first, all 58 sequences)

| quantity | value |
|---|---|
| GT persons per sequence | 21.1 (14-23); 966 numbered player tracks, 256 unnumbered |
| Koshkina read rate | 425 / 4,870 tracks = **8.7%** |
| read precision vs numbered GT | 328 / 375 = **0.875** |
| merged tracklets at `tau = 0.040` | 3,790 (65.3 per sequence) |
| prediction rows with an auditable GT match | 469,357 |
| rows whose GT player is **unnumbered** (abstaining is correct) | 15.7% |
| rows whose GT player has **>= 1 OCR read anywhere in the sequence** | **24.1%** |

Ceilings on per-row identity accuracy `(team AND jersey)`, computed by handing each merged tracklet
its best possible label:

| ceiling | identity | jersey |
|---|---|---|
| oracle (dominant GT label per merged tracklet) | 0.8349 | 0.8993 |
| **oracle restricted to OCR-anchored identities** | **0.3649** | 0.4037 |
| abstain on everything | 0.1281 | 0.1583 |

The middle row is the one that binds. A perfect Stage-2 solver, fed this OCR and this partition,
tops out at 0.365 — the other 0.470 of the oracle is players nobody ever read, whose slot no amount
of joint inference can fill.

And the link that would spread the reads is weak. For every merged tracklet whose GT player *is*
OCR-anchored somewhere in the sequence, scoring it against the anchored galleries (own tracklet
excluded, mean-of-top-k):

| top-k | tracklet top-1 | row-weighted | mean candidates |
|---|---|---|---|
| 1 | 299/467 = 0.6403 | 0.6764 | 4.55 |
| 3 | 299/467 = 0.6403 | 0.6752 | 4.55 |
| 5 | 299/467 = 0.6403 | 0.6721 | 4.55 |
| 10 | 297/467 = 0.6360 | 0.6560 | 4.55 |

0.64 against 4.55 candidates (chance 0.22). A further **1,348 of 1,815 queries** have no anchored
identity to match at all. This replicates the ~0.62 LOTO top-1 measured on real matches in
`results/GTA_LINK_STAGE1.md` A3 on a completely different corpus.

## 3. The model

Per sequence (or per match half), variables `x[tracklet, identity | unknown]`.

**Evidence -> per-tracklet posterior.** `likelihood(i) = P_ocr * P_app * gate_team * gate_role`:

- *OCR soft votes.* Each read `(n, conf)` contributes `p_c` to the identity wearing `n` and
  `(1 - p_c) * P(n | j)` to every other roster number `j`, with `P(n | j)` from the digit prior of
  B2 and `p_c = p_correct * conf + (1 - p_correct) * 0.5`. The `unknown` class gets the background
  `1/K`. Nothing is masked: a read that names a number nobody's appearance supports loses the
  argument instead of deciding it (the silent-failure mode of `results/MANUTD_IDENTITY_PROFILE.md`).
- *Appearance.* `exp(app_gain * (sim - sim_none))`, `sim` = mean of the best `topk` cosine
  similarities against the identity's OCR-anchored gallery (mean-of-top-k, not max — Stage 1 A3
  measured max-over-crops as the rule most exposed to a contaminated merge).
- *Team / role gates.* Multiplicative factors in `[eps, 1]`. The role gate is what lets a keeper
  slot with no number read be filled by a keeper-looking tracklet.
- *Substitution windows.* An identity outside its availability window has likelihood 0.

**Objective.** `maximise sum_t n_rows(t) * payoff(t, label)`, where `payoff` is the identity's
posterior probability and the abstain payoff is a fitted constant `r_abstain`. The abstain payoff is
deliberately **not** the posterior mass on `unknown`: that mass mixes "this player carries no number"
(abstaining is right) with "this player is on the roster but left no evidence" (abstaining is exactly
as wrong as a wrong name). Conflating them made the first fit abstain on 92% of tracklets and lose to
the baseline at every one of 375 grid points; separating them is what produced the numbers below.

**Constraints.** One label per tracklet; two tracklets sharing **any** frame may not share an
identity (0-frame slack — the SoccerNet evaluator rejects a repeated id in one timestep); at most
`max_concurrent` named tracklets per team in any frame. Exclusion groups are built from actual
shared frames, not interval envelopes, so two fragments of one player that interleave without
touching are still allowed to share a name (pinned in `tests/test_identity_solve.py`).

## 4. Frozen configuration

Fitted on DEV only (20 sequences, 378 grid points), written to
`results/identity_solver_config.json` before TEST was touched:

```
p_correct   0.8837   (measured: 129 DEV reads, 15 wrong)
confusion   digit prior from those 129 reads (214 aligned digit observations, add-1 smoothed)
app_gain    10.0     sim_none 0.92     topk 3
pi_none     0.70     r_abstain 0.0
team_eps    0.02     role_eps 0.05     max_concurrent 11
```

`r_abstain = 0.0` sits on the edge of its grid and means *never reward abstention*: the DEV optimum
is a name-everything arm. It won by 0.005 identity accuracy over a conservative regime in the same
grid (`r_abstain = 0.05`, 160 of 1,249 tracklets named, DEV identity 0.2573 / jersey 0.2994). That
margin is not a real preference, and §5 shows what the two regimes actually are.

## 5. Gate 1 (PRIMARY) - FAIL

TEST, 38 sequences, 304,132 auditable rows, one run from the frozen file.

| arm | identity (team+jersey) | jersey | tracklets named |
|---|---|---|---|
| appearance-only baseline (connector `tau` 0.040 + unanimous propagation) | **0.2559** | 0.2816 | 229 / 2,541 |
| **Stage-2 solver (frozen)** | **0.2604** | 0.2737 | 2,129 / 2,541 |
| evidence ceiling (§2) | 0.3649 | 0.4037 | — |

Paired per sequence (the pre-declared test): mean **+0.0046**, median +0.0075, **helped 20, hurt 18**,
worst -0.229 (SNGS-035), best +0.193 (SNGS-086), **Wilcoxon p = 0.566**. Not significant — **Gate 1
fails as declared.**

Secondary, row level (McNemar over discordant rows): base-only-right 26,468 vs solver-only-right
27,838, p = 4.2e-09. That p is not to be believed as an independent-sample result: all rows of a
tracklet flip together, so the effective n is tracklets, not rows. It is reported because it was
pre-declared, and it says the same thing the per-sequence test says about *size*: +1,370 rows out of
304,132 = +0.45 percentage points.

DEV, for the record: 0.2450 -> 0.2625 (+0.0174), Wilcoxon p = 0.294 (14 helped / 6 hurt). The DEV
gain is 4x the TEST gain — the usual shrinkage of a boundary-selected grid point.

### 5b. The number the accuracy metric hides

| split / arm | rows given a number | jersey precision on those rows |
|---|---|---|
| TEST baseline | 46,917 / 304,132 = 15.4% | **0.8817** |
| TEST solver | 259,783 / 304,132 = **85.4%** | **0.2638** |
| DEV baseline | 12.4% | 0.8566 |
| DEV solver | 81.3% | 0.2483 |

The two arms reach the same accuracy by opposite routes: the baseline answers rarely and is right
88% of the time; the solver answers about everything and is right 26% of the time. Per-row identity
accuracy cannot tell them apart, because on a *numbered* GT player a wrong number scores exactly the
same as an abstention — the metric only punishes over-naming on the 15.7% of rows GT leaves
unnumbered. **For this project's actual use (per-player facts) the baseline's operating point is the
useful one and the solver's is not**, and the pre-declared gate metric is blind to that difference.
This is a finding about the gate, not an excuse for the result: the solver still fails its gate.

### 5c. POST-HOC sweep of the abstain floor — reported, NOT claimed

Chosen after seeing TEST, therefore not a result (same status as the eps 0.08 splitter setting in
`results/GTA_LINK_STAGE1.md` §5.5). Every row below is a point that existed in the DEV grid:

| `sim_none` | `r_abstain` | identity | named tracklets | named-row coverage | jersey precision | paired p |
|---|---|---|---|---|---|---|
| 0.92 (frozen) | **0.00 (frozen)** | 0.2604 | 2,129 | 0.854 | 0.264 | 0.566 |
| 0.92 | 0.02 | 0.2443 | 1,109 | 0.482 | 0.387 | 0.946 |
| 0.92 | 0.05 | 0.2568 | 222 | 0.147 | 0.932 | 0.717 |
| 0.92 | 0.10 | 0.2541 | 219 | 0.144 | 0.932 | 0.609 |
| 0.85 | 0.00 | 0.2670 | 2,131 | 0.851 | 0.274 | 0.373 |
| 0.85 | 0.02 | 0.2548 | 1,156 | 0.487 | 0.404 | 0.780 |
| 0.85 | **0.05** | **0.2742** | 319 | 0.183 | 0.853 | **0.0029** |
| 0.85 | 0.10 | 0.2539 | 219 | 0.145 | 0.926 | 0.609 |
| baseline | — | 0.2559 | 229 | 0.154 | 0.882 | — |

One of the eight beats the baseline significantly, and only by moving *two* parameters away from the
frozen values; picked out of a 378-point grid after the fact, that is noise mining and it is
labelled as such. The durable reading of the table is the last two columns: the solver traces a
coverage/precision curve (0.15 @ 0.93 -> 0.18 @ 0.85 -> 0.49 @ 0.40 -> 0.85 @ 0.26) and **the greedy
baseline sits on that curve, not below it** (0.154 @ 0.882). The joint solve buys a *dial*, which the
greedy rule does not have. It does not buy a better frontier.

## 6. Gate 2 - PASS (no regression), and the ablations

GS-HOTA (official `trackeval` SoccerNetGS, `gs_hota_full`), same 38 sequences, submissions differing
only in `attributes.jersey` on `player` rows:

| arm | GS-HOTA (TEST-38) |
|---|---|
| baseline `tau` 0.040 connector | 23.37 |
| **Stage-2 solver** | **24.19** |
| (on record, same arm over all 58: 23.53; Stage-1 best over all 58: 24.18) | — |

Paired per sequence: mean **+0.709**, median +1.365, **helped 24, hurt 14**, worst -13.94
(SNGS-035), best +9.91 (SNGS-041), Wilcoxon p = 0.243. No regression against the 23.53 reference —
**Gate 2 passes** — but the improvement is not significant either, and the per-sequence spread
(-13.9 to +9.9) is far larger than the mean.

Ablations on TEST (frozen config, one term removed at a time):

| arm | identity | delta vs full solver |
|---|---|---|
| full solver | 0.2604 | — |
| no OCR term | 0.1397 | **-0.1207** |
| no mutual exclusion | 0.2317 | -0.0287 |
| no appearance term | 0.2379 | -0.0225 |
| no digit-confusion prior (uniform) | 0.2564 | -0.0040 (row McNemar p = 0.52) |

Read this honestly. The joint-inference machinery *is* load-bearing relative to its own parts —
mutual exclusion alone is worth more (+0.029) than the whole solver's margin over the baseline
(+0.0045), and appearance is worth +0.023. What it is not is *additive on top of* the greedy
propagation: the constrained solve reaches the same place the greedy rule already reached, by a
different and much less precise route. And the digit-confusion prior, the piece B2 forced to be
thin, does nothing (p = 0.52) — it is 40 lines that buy 0.004 and should be dropped unless per-crop
OCR output is ever persisted.

## 7. Real matches: the goalkeepers are named (gate 4), and the spot-check (gate 3)

`tools/identity_match.py`, frozen GSR config, three labelled matches, roster and substitution
windows from `outputs/identity/<match>_lineup_assign.parquet`. Artifacts:
`results/identity_match_solver.json`, `outputs/identity/solver/*_solver_names.parquet`.

### Gate 4 — yes, and it is the one unambiguous win

| match | arm | tracks named | GK-role tracks named | ...of which carry a keeper's name | keeper slots filled |
|---|---|---|---|---|---|
| manutd_liverpool | baseline | 934 | 7 / 182 | **0** | **0 of 4** |
| manutd_liverpool | **solver** | 7,398 | 166 / 182 | 67 | **2** (Onana, Alisson) |
| manutd_tottenham | baseline | 867 | 7 / 186 | **0** | **0 of 4** |
| manutd_tottenham | **solver** | 8,095 | 169 / 186 | 59 | **2** (Onana, Vicario) |
| manutd_brighton | baseline | 873 | 3 / 280 | **0** | **0 of 4** |
| manutd_brighton | **solver** | 7,690 | 237 / 280 | 118 | **2** (Onana, Verbruggen) |

The baseline never names a goalkeeper in any of the three matches — and the handful of
goalkeeper-role tracks it does name, it names with an *outfielder* (0 of 7, 0 of 7, 0 of 3 carry a
keeper's name). The solver fills both keeper slots in all three matches and gets the right starters
every time (Onana for United; Alisson, Vicario, Verbruggen for the opponents — the actual starting
keepers). It also names the outfielders OCR never read: on `manutd_liverpool` the name set goes from
18 players to 29, including Matthijs de Ligt (384 tracks) and Virgil van Dijk (242), two of the four
players `results/MANUTD_IDENTITY_PROFILE.md` recorded as never named. The mechanism is exactly the
one Stage 2 was built for: a keeper carries no legible number, so the *role gate* plus the roster
slot is the only thing that can name him.

Caveats that must travel with this: it is measured at the `r_abstain = 0` operating point, which
names 87-95% of all tracklets, so "named" is cheap here; only 35-50% of goalkeeper-role tracks
actually receive a keeper's name; and there is no per-track ground truth on these matches, so this
counts *coverage of the keeper slot*, not naming accuracy. Two implementation ceilings are flagged
in the code: candidate identities are pruned to the best 6 per tracklet and HiGHS gets 60 s per
chunk (the unpruned program over ~700 merged tracklets did not close in 10 minutes), so the
real-match assignments are incumbents, not proven optima.

### Gate 3 — the 78-moment factorisation (report only, nothing claimed)

`tools/identity_carrier.py` reuses `tools/gta_carrier.py` unchanged and swaps only the name set.

| arm | gate-hit | team / candidate | naming | Wilson 95% | ident | end-to-end | assigned |
|---|---|---|---|---|---|---|---|
| baseline (on record) | 66/78 = 0.846 | 23/32 = 0.719 | 14/23 = **0.609** | 0.408-0.778 | 0.438 | 0.350 | 40 |
| Stage-2 solver names | 66/78 = 0.846 | 12/12 = **1.000** | 7/12 = **0.583** | 0.320-0.807 | 0.583 | 0.438 | 16 |

Per match (naming, n): baseline liverpool 5/9, tottenham 3/4, brighton 6/10; solver liverpool 3/5,
tottenham **0/2**, brighton 4/5. Gate-hit is identical (0.846), as in every previous arm — this
stage cannot touch it either. The team factor reaching 12/12 is an artifact, not a win: with the
whole lineup in the gallery the true carrier is trivially inside any surviving candidate set, and
the price is visible next to it — assigned moments collapse **40 -> 16** and abstention rises
0.487 -> 0.795, because a 29-player gallery destroys the margin the v1 operating point
(`min_sim 0.88`, `min_margin 0.01`) requires. The naming factor moves 0.609 -> 0.583 on a
denominator that fell from 23 to 12. `results/GTA_LINK_STAGE1.md` A4 pre-declared that this label
set cannot resolve anything smaller than ~0.25; it does not resolve this either. **Nothing is
claimed from this table.**

## 8. Verdict

- **Gate 1: FAIL.** +0.0045 identity accuracy on 38 held-out sequences, Wilcoxon p = 0.566. The
  Lu-style joint solve does not beat the appearance-only arm on this data.
- **Gate 2: PASS.** GS-HOTA 23.37 -> 24.19 on the same sequences, no regression vs 23.53.
- **Gate 4: PASS, and it is the concrete win.** The goalkeepers are named — 0 of 4 keeper slots
  filled in every match before, 2 of 4 (both starters, correct) in every match after — and de Ligt
  and van Dijk with them. Measured at a name-everything operating point, with no per-track truth.
- **Gate 3: not resolvable.** 0.609 -> 0.583 naming on a denominator that fell 23 -> 12; assigned
  moments 40 -> 16. Reported, not claimed, exactly as pre-declared.
- **The binding cause is evidence, not inference.** 8.7% of tracks carry an OCR read; 24.1% of rows
  belong to a player read anywhere; the appearance link runs at 0.64 top-1 against 4.6 candidates.
  The ceiling for any solver on these unaries is 0.365, and the baseline is already at 0.256 of it.
  The basketball paper's 85-89% came with play-by-play naming who was on court at every moment —
  dense weak labels. We have a number legible on one track in eleven.
- **Consequence for Stage 3.** This is the strongest argument yet for the commentary stage: the
  missing ingredient is *more identity anchors per half*, not a better decision layer over the few
  we have. Stage 3's gate (precision >= 0.60, >= 1.5 names/min) would roughly triple the anchored
  fraction if it lands, which is the only lever §2 leaves open.
- **The DEV selection metric was the wrong one.** Per-row identity accuracy is indifferent between
  a 15%-coverage / 88%-precision namer and an 85%-coverage / 26%-precision one, so the grid picked
  the reckless regime by 0.005 on DEV. Any re-run of this stage should pre-declare
  **coverage at a precision floor** (e.g. maximise named-row coverage subject to jersey precision
  >= 0.80) and redo the DEV/TEST discipline from scratch. That is a re-run, not a re-scoring: the
  numbers above stand as measured.
- **Do not ship the `r_abstain = 0` operating point** into anything that produces per-player facts.
  If the solver is used at all it must abstain, and that choice has to be validated against a
  precision target.
- **What is worth keeping from this stage**: the GK/role slot (gate 4 is real and cheap), the
  substitution-window gating, and the fact that the solver exposes a coverage dial the greedy rule
  does not have. What is worth deleting: the digit-confusion prior (inert, p = 0.52).
