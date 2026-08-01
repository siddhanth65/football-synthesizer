# Confirmatory retest: a DEV-tuned abstention regime, pre-registered, one shot

Date: 2026-07-29. Follows `results/OCR_REALMATCH.md` section 6, whose closing note asked for exactly
this: *"add when someone pre-declares a coverage-at-precision-floor gate"*, and
`results/IDENTITY_SOLVER_STAGE2.md` section 8, which says the same thing in more words
(*"pre-declare coverage at a precision floor ... and redo the DEV/TEST discipline from scratch"*).

Yesterday's run measured two things that both point at the same fix. (a) The Stage-2 solver at its
frozen `r_abstain = 0` name-everything operating point is catastrophically worse than either control
on real-match LOTO (p ~ 1e-99 to 1e-129). (b) The greedy Stage-1 rule on per-crop evidence gained
+0.0302 at p = 0.00656 against a Bonferroni alpha of 0.00625 — an unclaimable near-miss. The fix
both point to is a properly tuned abstention regime, tuned on data independent of the test.

**Everything below ran on CPU from existing parquets and caches. No GPU work.**

**Result, up front: the registered arm FAILS. One-sided McNemar p = 0.9958 against alpha 0.05; the
arm is 0.0329 *below* the control on the paired keys (two-sided p = 0.0114, i.e. the direction is
significantly wrong). One run, no second attempt.**

---

## 1. The DEV frontier (tuning, GSR DEV-20 only, TEST-38 never touched)

`tools/retest_abstention.py`, per-crop evidence bundles (`outputs/gsr/identity_bundles_percrop`,
`koshkina_percrop_votes`), config frozen at `results/identity_solver_config_percrop.json` with
**only** `r_abstain` and `sim_none` reopened. Artifact:
`results/gsr_benchmark/retest_dev_frontier.json`.

Two coverage/precision pairs are reported per point:

- **own operating point** — coverage and row-weighted jersey precision of the tracklets the arm
  itself names. This is the only pair that transfers: a real match has no ground truth with which to
  calibrate a posterior threshold, so the shipping arm must abstain by itself.
- **dial** — `eval.gsr_identity.coverage_curve`, i.e. the best coverage reachable by thresholding the
  posterior post hoc, at precision floors 0.85 and 0.60. Reported for continuity with
  `results/OCR_DENSIFICATION.md` section 6; **not** used for selection (it would be a third knob,
  and the brief froze the config at two).

DEV-20, 165,225 auditable rows.

| arm | `sim_none` | `r_abstain` | coverage | precision | named tracklets | dial cov @0.85 | dial cov @0.60 |
|---|---|---|---|---|---|---|---|
| solver | 0.85 | 0.00 | 0.7932 | 0.4625 | 1020 | 0.3335 | 0.5244 |
| solver | 0.85 | 0.02 | 0.5502 | 0.5700 | 662 | 0.3335 | 0.5197 |
| solver | 0.85 | 0.05 | 0.3717 | 0.7929 | 308 | 0.3346 | 0.3717 |
| solver | 0.85 | 0.10 | 0.3367 | **0.8554** | 234 | 0.3367 | 0.3367 |
| solver | 0.85 | 0.15 | 0.3305 | 0.8738 | 228 | 0.3305 | 0.3305 |
| solver | 0.85 | 0.20 | 0.3271 | 0.8756 | 226 | 0.3271 | 0.3271 |
| solver | 0.85 | 0.30 | 0.1672 | 0.9037 | 125 | 0.1672 | 0.1672 |
| solver | 0.85 | 0.50 | 0.1206 | 0.9304 | 82 | 0.1206 | 0.1206 |
| solver | 0.92 | 0.00 | 0.7953 | 0.4686 | 1022 | 0.3455 | 0.5306 |
| solver | 0.92 | 0.02 | 0.5297 | 0.6091 | 583 | 0.3463 | 0.5297 |
| **solver** | **0.92** | **0.05** | **0.3447** | **0.8572** | **247** | **0.3447** | **0.3447** |
| solver | 0.92 | 0.10 | 0.3342 | 0.8714 | 232 | 0.3342 | 0.3342 |
| solver | 0.92 | 0.15 | 0.3300 | 0.8751 | 228 | 0.3300 | 0.3300 |
| solver | 0.92 | 0.20 | 0.3240 | 0.8786 | 224 | 0.3240 | 0.3240 |
| solver | 0.92 | 0.30 | 0.1467 | 0.9256 | 106 | 0.1467 | 0.1467 |
| solver | 0.92 | 0.50 | 0.0602 | 0.9481 | 37 | 0.0602 | 0.0602 |
| **greedy** (Stage-1 rule, per-crop reads) | — | — | 0.3675 | **0.8425** | 271 | — | 0.3675 |

**Selection, by the pre-declared rule (max coverage subject to own precision >= 0.85):**
`sim_none = 0.92`, `r_abstain = 0.05` — **coverage 0.3447 at precision 0.8572**, coverage@0.60 also
0.3447 (at this operating point the arm's whole named set already clears 0.60, so the two floors
coincide). 11 of the 17 points qualify.

**Greedy vs solver, decided on DEV alone.** The greedy rule reaches **higher** coverage (0.3675) at
**0.8425** precision — *below* the floor, by 0.0075. It therefore does not qualify and the solver arm
is the registered one. This is a knife-edge and is recorded as such: on this evidence the greedy rule
and the tuned solver sit on essentially the same frontier (`IDENTITY_SOLVER_STAGE2.md` section 5c's
durable finding, reproduced on per-crop evidence), and the floor decides between them by less than a
percentage point of precision. What the solver has that the greedy rule does not is the dial: the
greedy arm's confidences are all 1.0, so its "dial @0.85" column is meaningless (0.0073) — it cannot
be moved to any other operating point at all.

Three things the frontier shows plainly:

1. **`r_abstain = 0` is not a coverage/precision trade, it is a collapse.** 0.79 coverage at 0.46-0.47
   precision. The 0.05 -> 0.10 band is where the knee is.
2. **`sim_none` barely matters** at the knee (0.8572 vs 0.8554 for 0.92 vs 0.85 at their respective
   qualifying points); the abstain payoff does all the work.
3. **The post-hoc dial and the arm's own abstention reach the same place.** At `r_abstain = 0` the
   dial recovers 0.3455 @ 0.85 and the tuned arm reaches 0.3447 @ 0.8572 by itself. Tuning
   `r_abstain` buys transferability, not a better frontier.

## 2. Registration (echoed; the file is `results/retest_registration.json`)

Written **2026-07-29T10:16:11.852630+00:00** = 15:46:11 local, before any real-match computation for
this arm. Ordering evidence, file mtimes:

| file | mtime (local) |
|---|---|
| `results/gsr_benchmark/retest_dev_frontier.json` (DEV only) | 15:44:28 |
| `results/identity_solver_config_retest.json` | 15:45:36 |
| **`results/retest_registration.json`** | **15:46:11** |
| first arm artifact (`outputs/identity/solver_retest/manutd_liverpool_solver_names.parquet`) | 15:47:18 |
| `results/identity_match_solver_retest.json` | 15:49:39 |
| `results/identity_carrier_retest.json` | 15:51:24 |
| `results/retest_loto.json` (the test) | 15:52:01 |

The registration asserted in code that none of the five arm artifacts existed when it was written.

- **Arm (one only):** `solver_retest` = Stage-2 identity MILP on per-crop OCR evidence (frozen 0.85
  floor rule), `sim_none = 0.92`, `r_abstain = 0.05`, everything else frozen from
  `results/identity_solver_config_percrop.json`; partition = GTA connector, splitter OFF,
  `tau = 0.040`.
- **Config hash:** `results/identity_solver_config_retest.json`, sha256
  `8da5cddd8992222f477d6a5c55ce7640001559b4f58fe85769b08a0bc4fd86f4`.
- **Hypothesis, verbatim:** *"The registered arm (Stage-2 solver, per-crop evidence, sim_none 0.92,
  r_abstain 0.05) achieves HIGHER paired leave-one-group-out top-1 gallery accuracy than the
  tau = 0.040 Stage-1 connector control, on the three labelled matches (manutd_liverpool,
  manutd_tottenham, manutd_brighton), graded by the anchor-truth grader of
  tools.gta_carrier.loto_query_hits."*
- **Test:** exact McNemar (binomial on discordant pairs, p0 = 0.5), **one-sided**
  (`arm_only_right > control_only_right`), **alpha = 0.05**, single pre-registered comparison, no
  multiplicity correction.
- **Denominator rule:** paired on the exact query keys `(match, chunk, src_track, frame)` present in
  **both** arms' hit dictionaries; the test denominator is the **discordant** pairs among those
  shared keys (`control_only_right + arm_only_right`). Unpaired rates are reported, not tested.
- **Grader:** `tools.gta_carrier.loto_query_hits` **unmodified** — queries are gallery crops whose
  *source* tracklet carries a close-up-anchor OCR read, graded against that anchor's player name,
  never the name the arm assigned.
- **Declared bias:** the control labels its gallery from the same anchor chain that defines the
  truth; the registered arm never sees that chain. The test is biased **against** the arm.
- **Stopping rule:** one run, verdict read once, no knob-turning afterwards.

## 3. THE SINGLE TEST — FAIL

`python -m tools.gta_carrier --loto --taus 0.04 --extra retest=outputs/identity_carrier_retest/solver
--controls tau0.040`. Artifacts `results/retest_loto.json`, `results/retest_verdict.json`.

Unpaired LOTO top-1 (context only, not the test — each arm's gallery samples different crops):

| arm | LOTO top-1 |
|---|---|
| baseline (on record) | 2421 / 3877 = 0.6245 |
| `tau0.040` control | 2273 / 3671 = 0.6192 |
| **`retest` (registered arm)** | **1079 / 1646 = 0.6555** |

**The registered comparison:**

| | value |
|---|---|
| paired keys | **1,491** |
| control (`tau0.040`) top-1 on those keys | **0.6895** |
| arm (`retest`) top-1 on those keys | **0.6566** |
| delta | **-0.0329** |
| control-only-right | **205** |
| arm-only-right | **156** |
| **one-sided McNemar p (registered)** | **0.9958** |
| alpha | 0.05 |
| **VERDICT** | **FAIL** |

Two-sided p on the same table is **0.0114**: the arm is not merely no better, it is worse by more
than chance would explain. That reverse reading is *not* a registered claim and nothing is claimed
from it; it is reported because it is the same number and hiding it would be dishonest. The declared
bias runs against the arm, so -0.0329 is a generous estimate of its deficit, not a harsh one.

**No second attempt was run and no knob was turned after this number was seen.**

### 3.1 What the failure actually says

The DEV-tuned abstention regime does exactly what DEV said it would — 0.20 real-match row coverage at
a regime that scored 0.857 precision on DEV, against 0.79 coverage at 0.47 precision for
`r_abstain = 0` — and it *still* does not beat the greedy connector arm at appearance retrieval. So:

- **The `r_abstain = 0` catastrophe is fixed** (LOTO 0.3591 -> 0.6566 on paired keys, both against
  the same control). Tuning the abstention regime recovered ~0.30 of top-1. The instruction in
  `IDENTITY_SOLVER_STAGE2.md` section 8 was right about the direction and the size.
- **And it was not enough.** Fixed, the solver lands *at* the greedy arm's neighbourhood and slightly
  below the control. Two independent tunings (yesterday's greedy arm on the same evidence, today's
  solver at a DEV-selected floor) both land in the 0.65-0.71 band that the connector control already
  occupies. **On this corpus the joint solve buys a dial, not a better frontier** — Stage 2's own
  section 5c conclusion, now measured a third time and on real broadcast.
- The one thing that has never been contradicted: **gate-hit is 0.846 in every arm, for the sixth
  analysis running.** Nothing in the identity chain touches it.

## 4. Report-only (no claims)

### 4.1 The 78-moment factorisation

`tools/identity_carrier.py --arm solver --names-dir outputs/identity/solver_retest`. The baseline arm
reproduced **byte-identically** (66/78 = 0.846, 23/32 = 0.719, 14/23 = 0.609, e2e 0.350, 40 assigned)
before the registered arm was scored.

| arm | gate-hit | team / candidate | naming | Wilson 95% | ident | end-to-end | assigned | abstain |
|---|---|---|---|---|---|---|---|---|
| baseline (on record) | 66/78 = 0.846 | 23/32 = 0.719 | 14/23 = 0.609 | 0.408-0.778 | 0.438 | 0.350 | 40 | 0.487 |
| **retest (registered arm)** | 66/78 = 0.846 | 22/35 = 0.629 | 17/22 = **0.773** | 0.566-0.899 | 0.486 | **0.386** | **44** | 0.436 |
| *(yesterday) percrop_names* | 0.846 | 25/36 = 0.694 | 20/25 = 0.800 | 0.609-0.911 | 0.556 | 0.444 | 45 | 0.423 |
| *(yesterday) solver @ r_abstain 0* | 0.846 | 13/15 = 0.867 | 8/13 = 0.615 | 0.355-0.823 | 0.533 | 0.421 | 19 | 0.756 |

Per match (naming, n): liverpool 8/12, tottenham 2/2, brighton 7/8. Gallery players
18 / 14 / 26 per match (before: 18 / 24 / 20) — tottenham loses players again, as it did under
`percrop_names`, and remains undiagnosed.

17/22 vs the baseline's 14/23 is Fisher exact **p = 0.34**. `results/GTA_LINK_STAGE1.md` A4
pre-declared that 23 usable naming moments cannot resolve anything smaller than ~0.25. **Nothing is
claimed from this table**, and note that it points the *opposite way* to the registered test, which
is precisely why A4 forbade reading it.

### 4.2 Coverage on the real matches

There is **no per-track ground truth on these matches**, so a real-match coverage-at-precision floor
cannot be measured — which is the whole reason the floor was tuned on GSR DEV. What can be reported
is coverage:

| match | named rows / player+GK rows | named tracks / tracks | posterior median |
|---|---|---|---|
| manutd_liverpool | 22,372 / 123,837 = 0.1807 | 1,091 / 8,345 = 0.1307 | 0.741 |
| manutd_tottenham | 18,683 / 121,097 = 0.1543 | 931 / 8,934 = 0.1042 | 0.799 |
| manutd_brighton | 44,121 / 173,889 = 0.2537 | 1,684 / 9,294 = 0.1812 | 0.949 |
| **pooled** | **85,176 / 418,823 = 0.2034** | **3,706 / 26,573 = 0.1395** | — |

Against DEV's 0.3447 row coverage — the shortfall is the read-density gap already on record (GSR
per-crop `d = 0.208` vs real-match `d = 0.114`, `OCR_REALMATCH.md` section 2). Applying the DEV
0.85-floor dial threshold (posterior >= 0.0520) to the real-match posteriors retains 3,705 of 3,706
named tracks, i.e. the arm's own abstention has already done that filtering; the DEV 0.60 threshold
is the same number at this operating point.

For scale: the same solver at `r_abstain = 0` on the same per-crop evidence named
7,420 / 8,180 / 7,575 tracks. The registered arm
names 1,091 / 931 / 1,684 — the same order as the Stage-1 baseline's 934 / 867 / 873.

### 4.3 Goalkeepers — CHANGED, and the change is a loss

| match | arm | GK-role tracks named (of) | ...carrying a keeper's name | keeper slots filled |
|---|---|---|---|---|
| manutd_liverpool | solver @ `r_abstain` 0, per-crop | 165 / 182 | 65 | **2** (Onana, Alisson) |
| manutd_liverpool | **retest** | 11 / 182 | **0** | **0 of 4** |
| manutd_tottenham | solver @ `r_abstain` 0, per-crop | 169 / 186 | 59 | **2** (Onana, Vicario) |
| manutd_tottenham | **retest** | 9 / 186 | **0** | **0 of 4** |
| manutd_brighton | solver @ `r_abstain` 0, per-crop | 237 / 280 | 115 | **2** (Onana, Verbruggen) |
| manutd_brighton | **retest** | 16 / 280 | **0** | **0 of 4** |

Zero goalkeeper-role merged tracklets are named in any match (`gk_named = 0` in all three), and the
handful of GK-role *original* tracks that do carry a name inherit it from an outfielder merged with
them. **The abstention regime that fixes the LOTO collapse destroys Stage 2's one unambiguous win.**
It has to: a keeper's number is never read, so his likelihood comes from the role gate alone, and
`r_abstain = 0.05` is above whatever the role gate can produce without an OCR term. Gate 4 and the
LOTO result are in direct conflict, and there is no operating point in this grid that has both.

## 5. Negatives and limits

1. **The registered arm failed, one-sided p = 0.9958, direction wrong at two-sided p = 0.0114.** That
   is the headline.
2. **The greedy-vs-solver decision was decided by 0.0075 of DEV precision.** Had the floor been 0.84
   the greedy arm would have been registered — and its comparison against this same control is
   already on record at +0.0302, p = 0.00656 two-sided. A pre-registration whose arm selection hinges
   on that small a margin is fragile, and the honest reading is that the DEV frontier does **not**
   separate the two rules.
3. **Two independent knife-edges have now landed on opposite sides of their bar** (`percrop_names`
   p = 0.00656 vs alpha 0.00625, and this arm's DEV precision 0.8572 vs greedy's 0.8425). Both were
   recorded as declared. No claim survives either.
4. **Gate 4 (goalkeepers) is lost at the tuned operating point.** 2 of 4 keeper slots -> 0 of 4 in all
   three matches. Anything downstream that assumed the solver names keepers must use the
   `r_abstain = 0` names and inherit their 0.359 LOTO.
5. **The 78-moment table points the other way** (naming 0.773 vs 0.609) and is not evidence; A4's
   power caveat holds for the sixth time.
6. **DEV precision is row-weighted over merged tracklets**, so its effective n is tracklets (247
   named), not rows. The 0.8572 vs 0.8425 gap is well inside that noise.
7. **The paired key sets are small subsets of each arm's queries** (1,491 shared of 1,646 arm /
   3,671 control), because each arm's gallery sampler picks different frames. This is the
   pre-declared denominator rule, unchanged from yesterday, and it is what makes the comparison fair;
   it also means the test is powered on ~1.5k crops, not on 3.7k.
8. **No claim is made about GSR TEST-38.** It was not touched by this run; the DEV frontier is
   selection data, and the one confirmatory measurement was spent on the real matches.

## 6. Files

- `tools/retest_abstention.py` — DEV `r_abstain` x `sim_none` sweep on per-crop bundles
  (`greedy_percrop` scores the Stage-1 rule on the identical bundles), and `--verdict`, which reads
  the one-sided p of the single registered comparison out of a `--loto` payload. Nothing else was
  modified: the grader, the solver, the connector and the carrier harness are untouched.
- `results/retest_registration.json` — the pre-registration (timestamps above).
- `results/identity_solver_config_retest.json` — the registered config (sha256 above).
- `results/gsr_benchmark/retest_dev_frontier.json` — the DEV frontier.
- `results/identity_match_solver_retest.json`, `results/identity_carrier_retest.json`,
  `results/retest_loto.json`, `results/retest_verdict.json` — the run.
- `outputs/identity/solver_retest/*.parquet`, `outputs/identity_carrier_retest/solver/` — arm
  artifacts. Nothing on record was overwritten.
- Tests: `tests/test_gta_link.py`, `tests/test_identity_solve.py`, `tests/test_ocr_density.py` —
  14 passed.
