# The widened OCR crop on our own EPL footage — a pre-registered one-shot test

Date: 2026-07-31. Carries the crop-box fix of `results/OCR_DOMAIN_SHIFT.md` §7 and
`results/FOOTPASS_GAME18_SCORE.md` §v2 — measured on FOOTPASS game_18 — onto the three labelled
Manchester United matches, and spends **one** pre-registered confirmatory test on it.

**Result, up front: the registered arm FAILS.** One-sided McNemar **p = 0.9644** against alpha 0.05;
on the 1,762 paired query keys the widened-crop arm is **0.0085 below** the control (two-sided
p = 0.115, i.e. not distinguishable in either direction). One run, no second attempt.

**And the fix itself works exactly as advertised at the read level, which is the useful finding:**
read density `d` **0.1157 -> 0.1581 pooled (+37%)** at **identical** agreement with the close-up
anchor chain (0.9217 -> 0.9227). It buys evidence; it does not buy appearance retrieval.

---

## 1. Registration (echoed; the file is `results/retest2_registration.json`)

Written **2026-07-30T16:17:07Z = 21:47:07 local**, before any `crop_scale = 1.25` artifact existed
for any EPL match. The writer **asserted in code** that all ten arm artifacts were absent and
refuses to write the file otherwise; it also records the sha256 of `tools/identity_carrier.py`,
`tools/gta_carrier.py` and `tools/ocr_match.py` as of registration.

| file | mtime (local) |
|---|---|
| **`results/retest2_registration.json`** | **2026-07-30 21:47:07** |
| first arm artifact (`outputs/manutd_liverpool/final/ocr_percrop_w125/h1_chunk_000.parquet`) | 2026-07-30 22:01:01 |
| last arm artifact (`outputs/manutd_brighton/.../h2_chunk_005.parquet`) | 2026-07-31 04:03:49 |
| `outputs/identity_carrier_w125/percrop_names/*` | 2026-07-31 04:05:59 |
| `results/identity_carrier_w125.json` | 2026-07-31 04:06:41 |
| `results/epl_cropfix_loto.json` (the test) | 2026-07-31 04:08:44 |
| `results/epl_cropfix_verdict.json` (the verdict) | 2026-07-31 04:09:14 |

* **Arm (one only):** `percrop125` = the **shipped greedy Stage-1 namer**
  (`tools.identity_carrier.percrop_namer` -> `tools.gta_carrier.propagate_names`: dominant per-crop
  read per tracklet, team/roster/half-gated, then unanimous propagation inside a merge group with
  any disagreeing group left untouched) consuming per-crop OCR evidence produced at
  `--crop-scale 1.25`. Aggregation = the frozen 0.85 floor (`min_crop_conf 0.99`, `min_votes 5`,
  `min_legibility 0.5`, `emit_all False`). Partition = GTA connector, splitter OFF, `tau = 0.040`.
* **Control:** `percrop10` = the identical rule on the identical matches at `crop_scale 1.0` — the
  `percrop_names` arm of `results/OCR_REALMATCH.md` §6, scored from its **on-record** gallery
  artifacts, which were neither rebuilt nor overwritten.
* **Hypothesis (verbatim):** *"The greedy namer consuming per-crop reads at crop_scale 1.25 achieves
  HIGHER paired leave-one-group-out top-1 gallery accuracy than the same rule at crop_scale 1.0, on
  manutd_liverpool + manutd_tottenham + manutd_brighton, graded by the anchor-truth grader of
  tools.gta_carrier.loto_query_hits."*
* **Test:** exact McNemar on discordant pairs, **one-sided** (`arm_only_right > control_only_right`),
  **alpha 0.05**, single comparison, no multiplicity correction.
* **Denominator:** the exact query keys `(match, chunk, src_track, frame)` present in **both** arms.
* **Declared bias:** unlike `OCR_REALMATCH.md` §0 and `RETEST_ABSTENTION.md`, this comparison is
  **symmetric** in the truth source — neither arm's names come from the anchor chain that defines
  the truth, and both run the identical rule and partition. No direction of bias was claimed.
* **Only code change made for the run:** an evidence *path* parameter (`percrop_namer(..., variant)`
  plumbed from a new `--percrop-variant` flag) so the namer can read `ocr_percrop_w125`. The rule,
  the floor, `min_votes = 5`, the connector, the grader and the scorer are untouched.
* **Stopping rule:** one run, verdict read once, no knob turned afterwards. Honoured.

## 2. The GPU passes

One match at a time, 20 crops per track, `--crop-scale 1.25 --variant _w125`, resumable by disk
state. `OCR_PERCROP_VERSION = ocr-percrop-1.1`; every row carries its `crop_scale`.

| match | wall clock (1.25) | v1 (1.0, on record) | chunks | crops 1.25 | crops 1.0 |
|---|---|---|---|---|---|
| manutd_liverpool | 21:47:28 -> 23:58:48 = **2 h 11 m 20 s** | 1 h 46 m | 11 | 94,244 | 94,244 |
| manutd_tottenham | 23:58:48 -> 01:54:08 = **1 h 55 m 20 s** | 2 h 01 m | 11 | 95,284 | 95,284 |
| manutd_brighton | 01:54:08 -> 04:03:53 = **2 h 09 m 45 s** | 1 h 56 m | 11 | 117,042 | 117,041 |
| **total** | **6 h 16 m 25 s** | 5 h 43 m | 33 | **306,570** | 306,569 |

**The crop set is identical to within one crop** (one brighton box clears `MIN_BOX_H` at 1.25x that
did not at 1.0x — the same +1 game_18 showed), so the two passes are **paired at the crop level over
all three matches**, not sampled.

*Operational note, no effect on data:* a CPU-heavy game process (`WatchDogs2`, 100% CPU) was running
on the machine during part of the liverpool pass; `h1_chunk_001` took 1,598 s against 461-936 s for
every other chunk. That contention, not the crop scale, is most of the +9.7% total. Chunk time is
dominated by random-access video seeking; the widening itself cost roughly what game_18 measured
(+24% there, on an uncontended machine).

## 3. Read density and read agreement — the shipped 0.85 rule

Denominator = each match's player/GK `(chunk, track_id)` pairs. "Agreement" is the 318/345 = 0.9217
statistic of `OCR_REALMATCH.md` §2.1: on tracklets that **both** the close-up anchor chain and the
per-crop pass read, do they return the same number?

| match | `d` @1.0 | `d` @1.25 | ratio | tracks read | GK-role tracks read (of) | agreement @1.0 | agreement @1.25 |
|---|---|---|---|---|---|---|---|
| manutd_liverpool | 0.1150 | **0.1505** | 1.31x | 960 -> **1,256** | 11 -> **13** (of 182) | 107/114 = 0.9386 | 133/142 = **0.9366** |
| manutd_tottenham | 0.1053 | **0.1455** | 1.38x | 941 -> **1,300** | 9 -> **13** (of 186) | 105/116 = 0.9052 | 134/146 = **0.9178** |
| manutd_brighton | 0.1263 | **0.1771** | 1.40x | 1,174 -> **1,646** | 7 -> **19** (of 280) | 106/115 = 0.9217 | 115/126 = **0.9127** |
| **pooled** | **0.1157** | **0.1581** | **1.37x** | 3,075 -> **4,202** | 27 -> **45** (of 648) | **318/345 = 0.9217** | **382/414 = 0.9227** |

Wilson 95% on the pooled agreement: 1.0 = 0.889-0.946, 1.25 = **0.893-0.945**. **Flat, on a 20%
larger graded set.** This is the EPL analogue of game_18's `d` 0.0304 -> 0.0704 at precision
0.786 -> 0.867: the same mechanism, one third the size, because EPL crops were never as broken.

### 3.1 The reads that already existed do not move; the new ones are as good

Paired over the **2,918** tracklets read at both scales: **2,909 (99.69%) return the identical
dominant number**. (The game_18 EPL control in `OCR_DOMAIN_SHIFT.md` §7.2 measured 209/209 at the
crop level; this is the same statement at the tracklet level, on 14x the sample.)

Split the anchor-graded set three ways:

| subset | n | agree | rate |
|---|---|---|---|
| read at both scales | 335 | 309 @1.0 / **311 @1.25** | 0.9224 / 0.9284 |
| read only at 1.25 (the marginal reads) | **79** | **71** | **0.8987** |
| read only at 1.0 (lost by widening) | 10 | 9 | 0.900 |

Discordant pairs on the shared 335: **0 where 1.0 is right and 1.25 wrong, 2 the other way.** The
widened crop never corrupted a read this evidence can see; the extra reads land at 0.899, i.e.
within noise of the base rate. **The fix is clean at the read level on EPL.**

### 3.2 Crop level, all 306,570 crops

| match | legibility >= 0.5 @1.0 | @1.25 | `p_number` >= 0.99 @1.0 | @1.25 | ratio |
|---|---|---|---|---|---|
| manutd_liverpool | 0.2220 | 0.2869 | 0.1350 | 0.1756 | 1.30x |
| manutd_tottenham | 0.2072 | 0.2971 | 0.1322 | 0.1802 | 1.36x |
| manutd_brighton | 0.2127 | 0.3325 | 0.1310 | 0.1848 | 1.41x |
| **pooled** | **0.2139** | **0.3075** | **0.1326** | **0.1806** | **1.36x** |

`OCR_DOMAIN_SHIFT.md` §7.2 projected this from 1,800 brighton crops: legibility 0.2106 -> 0.3372,
confident reads 0.1328 -> 0.1739 = 1.31x. **Measured over the full brighton corpus: 0.2127 -> 0.3325
and 1.41x.** The 1,800-crop probe replicated to within 0.005 on legibility and slightly *under*-called
the confident-read gain — the opposite error to the one game_18's binomial `d` projection made.

### 3.3 Exploratory (post-hoc, not the registered rule): the 0.80 floor

| floor | `d` @1.0 | `d` @1.25 | ratio | agreement @1.0 | agreement @1.25 |
|---|---|---|---|---|---|
| **0.85 (shipped, registered)** | 0.1157 | **0.1581** | 1.37x | 318/345 = 0.9217 | 382/414 = **0.9227** |
| 0.80 (exploratory) | 0.2075 | 0.2785 | 1.34x | 426/481 = 0.8857 | 482/548 = 0.8796 |

Both floors gain the same ~1.35x. Nothing was selected on this table; it is here because
`ocr_match.py --report` defaults to the 0.80-floor rule and the first pass of this section was
computed at that default — see §6.4.

## 4. THE REGISTERED TEST — FAIL

`python -m tools.gta_carrier --loto --taus 0.04 --extra percrop10=outputs/identity_carrier_percrop/percrop_names --extra percrop125=outputs/identity_carrier_w125/percrop_names --controls percrop10`.
Artifacts `results/epl_cropfix_loto.json`, `results/epl_cropfix_verdict.json`.

**Reproduction requirements, all met inside the same run, byte-identically:**

| arm | unpaired LOTO top-1 | on record |
|---|---|---|
| baseline | 2421 / 3877 = 0.6245 | 0.6245 (`OCR_REALMATCH.md` §6) |
| `tau0.040` | 2273 / 3671 = 0.6192 | 0.6192 |
| **`percrop10` (the control)** | **1317 / 1931 = 0.6820** | **0.6820** |
| **`percrop125` (the arm)** | **1399 / 2117 = 0.6608** | new |

The registered comparison:

| | value |
|---|---|
| paired keys | **1,762** |
| control (`percrop10`) top-1 on those keys | **0.7128** |
| arm (`percrop125`) top-1 on those keys | **0.7043** |
| delta | **-0.0085** |
| control-only-right | **47** |
| arm-only-right | **32** |
| **one-sided McNemar p (registered)** | **0.9644** |
| alpha | 0.05 |
| **VERDICT** | **FAIL** |

Two-sided p on the same table is **0.115** — unlike `RETEST_ABSTENTION.md`'s arm, this one is not
significantly *worse* either; it is indistinguishable. Only 79 of 1,762 paired keys are discordant
at all, which is what the 99.69% read stability of §3.1 predicts: **denser reads of the same players
name mostly the same tracklets, so the gallery the appearance model retrieves against barely moves.**

Reference comparisons emitted by the same driver, **not** the test and not claimed:

| comparison | n paired | control | arm | ctrl-only | arm-only | two-sided p |
|---|---|---|---|---|---|---|
| `tau0.040` vs `percrop10` | 1,756 | 0.7079 | 0.6777 | 210 | 157 | 0.00656 |
| `baseline` vs `percrop10` | 519 | 0.7283 | 0.7052 | 66 | 54 | 0.315 |

The first is the mirror image of `OCR_REALMATCH.md` §6's `percrop_names vs tau0.040` (157/210,
p = 0.00656) with the roles swapped — **it reproduces to the digit**, which is the strongest
available check that the control arm is the on-record one.

**No second attempt was run and no knob was turned after this number was seen.**

## 5. Report-only (no claims)

### 5.1 The 78-moment factorisation

`tools/identity_carrier.py --arm percrop_names --percrop-variant _w125`. The baseline arm
**reproduced byte-identically** (66/78, 23/32, 14/23, e2e 0.350, 40 assigned) in the same run.

| arm | gate-hit | team / candidate | naming | Wilson 95% | ident | end-to-end | assigned | abstain |
|---|---|---|---|---|---|---|---|---|
| baseline (on record) | 66/78 = 0.846 | 23/32 = 0.719 | 14/23 = 0.609 | 0.408-0.778 | 0.438 | 0.350 | 40 | 0.487 |
| `percrop_names` @1.0 (on record) | 0.846 | 25/36 = 0.694 | 20/25 = **0.800** | 0.609-0.911 | 0.556 | **0.444** | **45** | 0.423 |
| **`percrop_names` @1.25 (new)** | 0.846 | 21/32 = 0.656 | 14/21 = **0.667** | 0.454-0.828 | 0.438 | **0.350** | 40 | 0.487 |

Per match (naming, n): @1.25 liverpool 4/9, tottenham 3/3, brighton 7/9 (@1.0: 8/12, 3/3, 9/10).
**The 78-moment table moves DOWN at 1.25**, by 0.133 of a naming factor on 21-25 moments —
`GTA_LINK_STAGE1.md` A4 pre-declared that this label set cannot resolve anything below ~0.25, and
this is the seventh analysis in which it points somewhere the powered test does not. Nothing is
claimed from it. (Curiosity, not a finding: the 1.25 arm's `ident`, `e2e`, `assigned` and `abstain`
coincide exactly with the baseline's at 14 correct of 40 assigned, while its per-match naming
differs from baseline's — a coincidence in the aggregate, not a redirect failure.)

**Gate-hit is 66/78 = 0.846 for the eighth analysis running.** Nothing in the identity chain touches
it.

### 5.2 The game_18 pathology replicates: name-disagreeing merge groups rise with density

| match | groups disagreeing @1.0 | @1.25 | direct reads named @1.0 -> @1.25 | named after propagation |
|---|---|---|---|---|
| manutd_liverpool | 11 | **17** | 625 -> 833 | 1,338 -> 1,725 |
| manutd_tottenham | 21 | **23** | 585 -> 820 | 1,104 -> 1,473 |
| manutd_brighton | 48 | **84** | 1,142 -> 1,607 | 2,378 -> 2,760 |
| **total** | **80** | **124 (+55%)** | 2,352 -> **3,260** | 4,820 -> **5,958** |

game_18 went 2 -> 14 on the same quantity. A disagreeing group is a **proven bad merge** and
`build_gallery` drops it, so denser evidence both fills more groups and exposes more broken ones.
Reads landing on no available roster shirt: 335 -> 423 (liverpool), 356 -> 480 (tottenham),
32 -> 39 (brighton) — the per-match roster-filter asymmetry recorded in `OCR_REALMATCH.md` §7.4 is
unchanged and still undiagnosed.

### 5.3 Goalkeepers

GK-role tracklets carrying a jersey **read** go 27 -> 45 of 648 (§3). As in every previous analysis
this buys nothing downstream: the greedy namer fills a keeper slot only through the roster, and a
keeper's number is what this pipeline does not read.

## 6. Negatives, limits, and one process error

1. **The registered arm failed: one-sided p = 0.9644, delta -0.0085.** That is the headline. The
   crop fix does not improve appearance retrieval on our own footage.
2. **This is the third pre-registered real-match identity arm in three days to fail** (percrop_names
   vs tau0.040 at p = 0.00656 against alpha 0.00625; the DEV-tuned solver at p = 0.9958; this one at
   p = 0.9644). All three land in the same 0.65-0.71 LOTO band. The band, not the arm, is the
   finding: **on this corpus, appearance retrieval is insensitive to everything the identity chain
   has done to it.**
3. **The 78-moment factorisation points the other way** (0.800 @1.0 vs 0.667 @1.25) and is not
   evidence; A4's power caveat holds for the seventh time.
4. **PROCESS ERROR, caught before it entered any table.** `tools/ocr_match.py --report` without
   `--rule-floor` applies the **top-level** rule of `results/ocr_density_rule.json`
   (`min_crop_conf 0.90`, `min_votes 3` — the 0.80 floor), not the shipped 0.85 floor. The first
   pass of §3 was computed at that default and reported `d = 0.276/0.269/0.290`, which is a
   different *rule*, not a different *crop scale*. It was caught by cross-checking against
   `tools.identity_match.percrop_reads` (which correctly asks for floor `"0.85"`, so **the
   registered arm was never affected**), and §3 was recomputed with `--rule-floor 0.85`. The
   0.80-floor numbers are kept as the exploratory row in §3.3 and in
   `results/ocr_match_<match>_w125_floor0.80.json`. The default is a footgun and should be changed.
5. **Agreement is not precision.** The 0.9227 is agreement with the 98.6%-verified anchor chain on
   the subset both mechanisms read — the easy subset — and says nothing about the 3,788 tracklets
   only the per-crop pass reads. There is still **no per-track ground truth on these matches**;
   that is why the only measured precision statement in the crop-fix chain remains game_18's.
6. **The arm's gallery is bigger and its unpaired LOTO is lower** (2,117 queries @ 0.6608 vs 1,931
   @ 0.6820). The paired analysis controls for this exactly; the unpaired rates are not comparable
   and neither is claimed.
7. **`min_votes = 4` was not run.** It is post-hoc in `FOOTPASS_GAME18_SCORE.md` v2.5 and stays that
   way here.
8. **The root cause is still untouched.** `generator.team_anchor.estimate_player_box` still
   reconstructs the box from two constants fitted to nothing; x1.25 is a calibration knob that was
   fitted on *game_18's* annotations and is merely *transferred* here. EPL has no annotated ROIs, so
   the right EPL constant is unknown — the measured 1.37x is a lower bound on what a per-match fit
   could give, and an upper bound on nothing.
9. **The timing comparison is contaminated** by a CPU-heavy game process during the liverpool pass
   (§2). The honest cost statement is game_18's uncontended +24%.

## 7. Files

* `results/retest2_registration.json` — the pre-registration (timestamps and asserts in §1).
* `results/epl_cropfix_loto.json`, `results/epl_cropfix_verdict.json` — the single test.
* `results/identity_carrier_w125.json` — the 78-moment factorisation and repair stats of the arm.
* `results/ocr_match_manutd_{liverpool,tottenham,brighton}_w125.json` — `d`, GK, anchor agreement at
  the shipped 0.85 floor; `*_w125_floor0.80.json` — the exploratory floor.
* `results/epl_cropfix_reads.json` — crop-level rates and the paired read analysis of §3.1/§3.2.
* `outputs/{manutd_liverpool,manutd_tottenham,manutd_brighton}/final/ocr_percrop_w125/*.parquet` —
  33 chunks, 306,570 crops, `ocr-percrop-1.1`, `crop_scale = 1.25`. Nothing on record was
  overwritten.
* `outputs/identity_carrier_w125/percrop_names/` — the arm's gallery, candidates and anchors.
* Code: `tools/identity_carrier.py` — `percrop_namer(..., variant)` and `--percrop-variant` only.
  The grader (`tools/gta_carrier.py`), the aggregation rule, the connector and the OCR pass are
  unmodified. Tests: `tests/test_gta_link.py`, `tests/test_identity_solve.py`,
  `tests/test_ocr_density.py` — 14 passed.
* Claim: `ident-031`.
