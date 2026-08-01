# Per-crop OCR on our own broadcast -- the real-match extension of the densification result

Date: 2026-07-29. Extends `results/OCR_DENSIFICATION.md` section 8 (which ran one match) to all three
labelled matches, and asks the question that file listed as *not measured* (section 9.7): **does the
densified evidence move the carrier-attribution factorisation on our own footage?**

Inputs, all frozen elsewhere and unchanged here:

- per-crop OCR rule = the **0.85 DEV floor** of `results/ocr_density_rule.json`
  (`min_crop_conf 0.99`, `min_votes 5`, `min_legibility 0.5`, `emit_all False`) -- the arm that
  passed the OCR gate on GSR TEST-38 at read precision 0.8578;
- Stage-2 solver = `results/identity_solver_config.json`, **only** the digit-confusion prior
  swapped for the DEV-20 per-crop refit (348 reads, 55 wrong, 573 aligned digit observations),
  persisted to `results/identity_solver_config_percrop.json`;
- tracklet partition = GTA connector, splitter OFF, `tau = 0.040` (the 80.1%-merge-precision
  operating point).

---

## 0. Pre-declaration (written to disk BEFORE the arm it grades existed)

**The claimable test is the LOTO gallery proxy, not the 78 hand moments.**
`results/GTA_LINK_STAGE1.md` A4 already established that the label set carries 23 usable naming
moments and cannot resolve any intervention of realistic size; those numbers are reported with
Wilson intervals and **nothing is claimed from them**.

**Statistic.** Gallery leave-one-group-out top-1, Stage-1 A3 machinery
(`tools/gta_carrier.py --loto`): every gallery crop is scored against the rest of its own team's
gallery with its whole merge group removed, and is counted correct when the nearest player is the
crop's true one. Queries are restricted to crops whose **source** tracklet carries a direct
close-up-anchor OCR read, and the label they are graded against is that anchor's player name -- not
the name the arm itself assigned. (Grading against the arm's own gallery label measures
self-consistency, which an arm that relabels a whole match can win for free; this is a change to
`loto_top1`/`loto_query_hits` made for this run and re-verified to reproduce the on-record baseline
figure 2421/3877 = 0.6245 byte-identically.)

**Test.** Exact paired McNemar on the **identical query keys** `(match, chunk, src_track, frame)`
shared by the two arms.

**Pre-declared comparisons (the claimable ones), two-sided, Bonferroni alpha = 0.025 each:**

1. `solver_percrop` vs `tau0.000` -- the **zero-merge control** (no connector, gallery redrawn from
   the same stride-2 cached frame grid).
2. `solver_percrop` vs `tau0.040` -- the **Stage-1 connector arm** (connector + unanimous OCR-name
   propagation), i.e. the best previously-measured naming arm.

Reported alongside, not claimed: `baseline` (on record) and `solver_anchor` (the same Stage-2
solver consuming the on-record close-up evidence instead of per-crop evidence), so that "solver"
and "per-crop evidence" can be separated.

**AMENDMENT, 00:45 (file mtime), before any arm below it was scored.** A fourth arm, `percrop_names`, is added:
the **Stage-1** naming rule (connector + unanimous propagation, the greedy rule the on-record 0.609
was measured with) fed the **per-crop** read set instead of the close-up anchors. It exists because
the three arms above confound two changes -- denser evidence *and* the MILP's
`r_abstain = 0` name-everything operating point, which `results/IDENTITY_SOLVER_STAGE2.md` section 8
explicitly says not to ship. `percrop_names` changes only the evidence. Its two pre-declared
comparisons are the same two controls (`tau0.000`, `tau0.040`), same statistic, same alpha; with
four arms x two controls the Bonferroni alpha becomes **0.00625** and every p below is reported
against that. Ordering evidence: at the time of this amendment the only `percrop_names` quantity
computed was its *repair* count on one match (625 directly-read tracklets -> 1,338 after
propagation on `manutd_liverpool`); no LOTO, factorisation or GK number for any arm existed, and
`manutd_tottenham` / `manutd_brighton` per-crop OCR had not finished (1 of 11 and 0 of 11 chunks on
disk).

**Direction of the bias, stated in advance.** For `baseline`, `tau0.000` and `tau0.040` the gallery
labels are *derived from the same anchor chain that defines the truth*, so those arms are graded
against their own evidence source. `solver_percrop` consumes per-crop reads only and never sees the
anchor chain, so its labels are independent of the truth. **The test is therefore biased against
the arm under test**, and a win for it is conservative.

**Secondary, report-only:** the 78-moment factorisation (gate-hit / team / naming / end-to-end /
assigned) with Wilson 95% intervals; `d` per match before/after; GK naming per match.

**Reproduction requirement:** the on-record baseline arm (0.846 / 0.719 / 0.609, e2e 0.350,
40 assigned) must reproduce byte-identically before any new arm is scored.

---

## 1. The GPU passes

One match at a time, 20 crops per track, resumable by disk state, `tools/ocr_match.py`.

| match | wall clock | chunks | crops | player/GK tracklets |
|---|---|---|---|---|
| manutd_liverpool (2026-07-28) | 1 h 46 m | 11 | 94,244 | 8,345 |
| **manutd_tottenham** | 00:15 - 02:16 = **2 h 01 m** | 11 | 95,284 | 8,934 |
| **manutd_brighton** | 02:18 - 04:14 = **1 h 56 m** | 11 | 117,041 | 9,294 |

**306,569 crops over 26,573 tracklets, ~5.8 GPU-hours in total.** Artifacts:
`outputs/<match>/final/ocr_percrop/<chunk>.parquet`, `ocr_version = ocr-percrop-1.0`. Nothing was
overwritten; the tottenham/brighton directories did not exist before this run.

Cost note for anyone repeating it: with a second CPU-heavy job on the machine a chunk takes
25 min instead of 9-10. Chunk time is dominated by random-access video seeking, not by the GPU.

## 2. Read density `d` on all three matches

Denominator = that match's player/GK tracklets. "Before" = the shipped close-up-anchor chain
(`outputs/identity/<match>_named_tracks_both2_prtreid.parquet`), a different mechanism reading only
gated hero shots; both sides are "fraction of tracklets carrying >= 1 read", so the comparison is
like-for-like on the quantity `results/EVIDENCE_DENSITY_LAW.md` cares about, not on the method.

At the **shipped 0.85-floor rule**:

| match | d before | d after | ratio | tracks before -> after | GK-role tracks read | overlap with anchors |
|---|---|---|---|---|---|---|
| manutd_liverpool | 0.0265 | **0.1150** | **4.34x** | 221 -> 960 | 3 -> **11** (of 182) | 114 / 221 |
| manutd_tottenham | 0.0316 | **0.1053** | **3.34x** | 282 -> 941 | 1 -> **9** (of 186) | 116 / 282 |
| manutd_brighton | 0.0179 | **0.1263** | **7.07x** | 166 -> 1,174 | 0 -> **7** (of 280) | 115 / 166 |
| **pooled** | **0.0253** | **0.1143** | **4.60x** | 669 -> 3,075 | 4 -> **27** (of 648) | 345 / 669 |

All three floors, for the record:

| match | floor 0.80 | floor **0.85** | floor 0.87 |
|---|---|---|---|
| manutd_liverpool | 0.2171 (1,812) | **0.1150 (960)** | 0.0714 (596) |
| manutd_tottenham | 0.1942 (1,735) | **0.1053 (941)** | 0.0688 (615) |
| manutd_brighton | 0.2117 (1,968) | **0.1263 (1,174)** | 0.0804 (747) |

**The one-match result replicates 3/3.** It is not uniform: brighton's anchor chain is the weakest
(0.0179) so its ratio is the largest, and the *post* densities are far tighter across matches
(0.105-0.126) than the *pre* densities (0.018-0.032) -- the per-crop pass is the more stable
mechanism.

### 2.1 A precision number on our own footage, which the densification file said did not exist

The close-up-anchor chain's jersey numbers are 98.6%-verified. On the tracklets **both** mechanisms
read, do they say the same number?

| match | agree / graded | rate |
|---|---|---|
| manutd_liverpool | 107 / 114 | 0.9386 |
| manutd_tottenham | 105 / 116 | 0.9052 |
| manutd_brighton | 106 / 115 | 0.9217 |
| **pooled** | **318 / 345** | **0.9217** (Wilson 95% 0.889-0.946) |

This is an *agreement* rate, not a precision -- the anchor chain is itself fallible and this is the
subset both mechanisms could read, which is the easy subset. Taken for what it is, it is **above**
the 0.8578 read precision the same rule measured on GSR TEST-38, so the transfer worry recorded in
`results/OCR_DENSIFICATION.md` section 8 ("real-match crops come from `estimate_player_box` rather
than a re-detected box... GSR precision does not automatically carry over") does not show up as a
collapse.

### 2.2 Correction to `results/OCR_DENSIFICATION.md` section 8 and claim `ident-021`

Both say "only 89 of the 221 close-up-anchor tracks are also read by the per-crop pass" next to the
**0.85**-floor row. Re-measured, 89/221 is the **0.87**-floor number. At the floors as labelled:
0.80 -> 158/221, **0.85 -> 114/221**, 0.87 -> 89/221. The complementarity conclusion is unchanged
and if anything understated (at the shipped rule 51.6% of anchor tracks are re-read, and 846 of the
960 per-crop-read tracks are ones the anchor chain never saw); the number itself was quoted off by
one row.

## 3. Solve unit: the brief's premise is half right

`tools/identity_match.py` pools **galleries** half-wide, but the MILP itself is solved **per
chunk** -- frame numbering restarts at each chunk, so the mutual-exclusion groups are only defined
inside one. A chunk is 15,000 frames at 25 fps = **10 minutes**, i.e. 5x the 2-minute unit
`results/EVIDENCE_DENSITY_LAW.md` measured as the good end of its solve-unit sweep (0.176 at 30 s
-> 0.395 at 2 min). Going half-wide would mean a ~45-minute unit over ~3,700 merged tracklets;
HiGHS already needs a 6-candidate prune and a 60 s budget to close ~700. **Not changed, and the
law's solve-unit lever is already spent at this granularity.** Recorded here because the brief
asked for a confirmation and the answer is "galleries yes, solve no, and it does not matter".

## 4. Goalkeepers, all three matches

Two separate questions. At the **read** level (section 2) GK-role tracklets carrying a jersey read
go 3/1/0 -> 11/9/7. At the **naming** level the per-crop evidence changes nothing, because the role
gate -- not OCR -- is what fills a keeper slot:

| match | arm | tracks named | GK-role tracks named (of) | ...carrying a keeper's name | keeper slots | which |
|---|---|---|---|---|---|---|
| manutd_liverpool | baseline | 934 | 7 / 182 | 0 | **0 of 4** | -- |
| manutd_liverpool | solver (anchor) | 7,398 | 166 / 182 | 67 | 2 | Onana, Alisson |
| manutd_liverpool | **solver (per-crop)** | 7,420 | 165 / 182 | 65 | **2** | Onana, Alisson |
| manutd_tottenham | baseline | 867 | 7 / 186 | 0 | **0 of 4** | -- |
| manutd_tottenham | solver (anchor) | 8,095 | 169 / 186 | 59 | 2 | Onana, Vicario |
| manutd_tottenham | **solver (per-crop)** | 8,180 | 169 / 186 | 59 | **2** | Onana, Vicario |
| manutd_brighton | baseline | 873 | 3 / 280 | 0 | **0 of 4** | -- |
| manutd_brighton | solver (anchor) | 7,690 | 237 / 280 | 118 | 2 | Onana, Verbruggen |
| manutd_brighton | **solver (per-crop)** | 7,575 | 237 / 280 | 115 | **2** | Onana, Verbruggen |

Same two starters, same counts to within 3 tracks. **Denser jersey evidence buys nothing for
goalkeeper naming** -- as it must, since a keeper's number is what the pipeline cannot read. The
Stage-2 caveats travel unchanged: this is measured at `r_abstain = 0`, where 85-92% of tracklets
are named, so "named" is cheap, and there is no per-track truth.

## 5. The carrier factorisation, four arms

`tau = 0.040` connector throughout, v1 frozen operating point (`min_sim` 0.88, `min_margin` 0.01),
all constraints OFF -- the configuration the on-record 0.846 / 0.719 / 0.609 was measured at.
The baseline arm **reproduced byte-identically** before anything else was scored.

| arm | gate-hit | team / candidate | naming | Wilson 95% (naming) | ident | end-to-end | assigned | abstain |
|---|---|---|---|---|---|---|---|---|
| **baseline** (on record) | 66/78 = 0.846 | 23/32 = 0.719 | 14/23 = **0.609** | 0.408-0.778 | 0.438 | **0.350** | 40 | 0.487 |
| solver, anchor evidence (2026-07-28) | 0.846 | 12/12 = 1.000 | 7/12 = 0.583 | 0.320-0.807 | 0.583 | 0.438 | 16 | 0.795 |
| **solver, per-crop evidence** | 0.846 | 13/15 = 0.867 | 8/13 = 0.615 | 0.355-0.823 | 0.533 | 0.421 | 19 | 0.756 |
| **percrop_names** (Stage-1 rule, per-crop evidence) | 0.846 | 25/36 = 0.694 | 20/25 = **0.800** | 0.609-0.911 | 0.556 | **0.444** | **45** | **0.423** |

Per match (naming, n): baseline liverpool 5/9, tottenham 3/4, brighton 6/10; solver+per-crop
liverpool 2/5, tottenham 1/2, brighton 5/6; **percrop_names liverpool 8/12, tottenham 3/3,
brighton 9/10**.

- **Gate-hit is 0.846 in every arm**, for the fifth analysis running. Nothing in the identity chain
  can touch it; it is a detector/gate problem.
- `percrop_names` is the first arm in this project's history to raise the naming factor **without
  paying in coverage**: assigned moments 40 -> **45** and abstention 0.487 -> 0.423, where every
  previous naming gain came with a coverage collapse (tau 0.040: 40 -> 31; solver: 40 -> 16).
- **It is still not a result on this label set.** 20/25 vs 14/23 is Fisher exact **p = 0.207**, and
  `results/GTA_LINK_STAGE1.md` A4 pre-declared that 23 usable moments cannot resolve anything
  smaller than ~0.25. Wilson intervals overlap heavily. Nothing is claimed from this table.
- The solver arm's `team` 1.000 -> 0.867 and its recovery of a few assigned moments (16 -> 19) is
  the same artifact A4 named: a whole-lineup gallery makes "true carrier inside the candidate set"
  nearly free and destroys the margin the v1 operating point needs.

## 6. THE POWERED TEST -- LOTO gallery leave-one-group-out, as pre-declared in section 0

1,931-3,877 query crops per arm (23-40x the label set), queries fixed to anchor-read source
tracklets, graded against the anchor player name.

| arm | LOTO top-1 (unpaired) | what it is |
|---|---|---|
| baseline | 2421 / 3877 = 0.6245 | on record (reproduced byte-identically) |
| `tau0.000` | 2226 / 3528 = 0.6310 | **control A**: zero merges, same frame grid |
| `tau0.040` | 2273 / 3671 = 0.6192 | **control B**: Stage-1 connector arm |
| **`solver_percrop`** | 1179 / 3230 = **0.3650** | Stage-2 MILP on per-crop evidence |
| **`percrop_names`** | 1317 / 1931 = **0.6820** | Stage-1 rule on per-crop evidence |
| `solver_anchor` | 1111 / 3367 = 0.3300 | Stage-2 MILP on close-up evidence (reference) |

Paired exact McNemar on identical query keys (Bonferroni alpha **0.00625**):

| comparison | n paired | control | arm | control-only-right | arm-only-right | McNemar p | verdict |
|---|---|---|---|---|---|---|---|
| **`solver_percrop` vs `tau0.000`** | 2,021 | 0.6373 | **0.3523** | 698 | 122 | **8.2e-99** | **FAIL -- significantly WORSE** |
| **`solver_percrop` vs `tau0.040`** | 2,910 | 0.6251 | **0.3591** | 949 | 175 | **4.5e-129** | **FAIL -- significantly WORSE** |
| **`percrop_names` vs `tau0.000`** | 1,087 | 0.7001 | 0.7093 | 131 | 141 | 0.585 | FAIL (no difference) |
| **`percrop_names` vs `tau0.040`** | 1,756 | 0.6777 | **0.7079** | 157 | 210 | **0.00656** | **FAIL by 0.0003** |
| `solver_anchor` vs `tau0.000` | 2,089 | 0.6333 | 0.3308 | 770 | 138 | 4.6e-107 | reference |
| `solver_anchor` vs `tau0.040` | 3,107 | 0.6125 | 0.3238 | 1070 | 173 | 3.3e-158 | reference |
| `tau0.040` vs `tau0.000` | 2,232 | 0.6366 | 0.6232 | 207 | 177 | 0.139 | reproduces Stage 1 A3 exactly |

### 6.1 Verdict, in the order the pre-declaration asks for it

1. **The arm the brief named -- the frozen Stage-2 solver on per-crop evidence -- FAILS, and not
   narrowly: it is 0.27-0.29 *below* both controls at p < 1e-98.** The MILP's `r_abstain = 0`
   name-everything operating point relabels the gallery so badly that appearance can no longer
   retrieve a known player. Per-crop evidence helps it slightly (0.3300 -> 0.3650) and nowhere near
   enough. `results/IDENTITY_SOLVER_STAGE2.md` section 8's instruction -- *"Do not ship the
   `r_abstain = 0` operating point into anything that produces per-player facts"* -- is now measured
   at n = 3,230 instead of asserted from 12 hand moments.
2. **The amended arm -- the greedy Stage-1 rule on per-crop evidence -- misses the declared alpha by
   0.0003** (p = 0.00656 vs 0.00625) against the Stage-1 connector control, on +0.0302 top-1
   (210 arm-only-right vs 157 control-only-right). It is recorded as a **FAIL**, not rounded up,
   for the same reason the OCR gate's PRIMARY arm was recorded as a fail at 0.7993 vs 0.800.
3. Against the zero-merge control the same arm shows **nothing** (p = 0.585). The two comparisons
   disagree because they ask different questions: `tau0.040` differs from `percrop_names` only in
   the read set (evidence isolated), while `tau0.000` also differs in the partition.
4. **Direction of the bias held as declared**: `tau0.000`/`tau0.040` label their galleries from the
   same anchor chain that defines the truth, `percrop_names` and `solver_percrop` never see it. The
   +0.030 is therefore a conservative estimate and the -0.27 a generous one.

**So: on the claimable test, attribution on our own footage does not move.** The single positive
signal is +0.030 LOTO top-1 at p = 0.0066 against a 0.00625 bar, one comparison of four.

## 7. Negatives, limits, raw numbers

1. **The pre-declared primary arm failed by 100+ orders of magnitude in the wrong direction.** That
   is the headline of this run.
2. **The label set still cannot referee anything** (naming 20/25 vs 14/23, p = 0.207). A4's power
   caveat holds for the fifth time.
3. **`percrop_names` scores on a smaller, easier query set** -- 1,931 unpaired queries against
   3,528-3,877 for the controls, because its gallery only contains tracklets per-crop OCR named.
   The paired analysis controls for this exactly (shared keys only); the unpaired 0.6820 does not
   and should not be compared with the unpaired 0.6192.
4. **The roster lookup that turns a read into a name behaves very differently per match**: reads
   landing on no available shirt are 335/960 (liverpool), 356/941 (tottenham), **32/1,174**
   (brighton). Lineup sizes are near-identical (22 slots in h1, 28-31 in h2), so this is a property
   of the read distributions, not of the rosters, and it means the roster acts as a strong filter
   on two matches and almost none on the third. Not diagnosed here.
5. **Per-crop evidence buys goalkeepers nothing** (section 4). The role gate already did that work.
6. **Gate-hit 0.846 is untouched, again.** Every factorisation arm run since 2026-07-27 -- five of
   them -- reports the identical 66/78.
7. **No per-track ground truth exists on these matches.** Section 2.1's 0.92 is agreement with the
   anchor chain on the subset both mechanisms read; it is not a precision estimate for the other
   2,730 per-crop-read tracklets.
8. `manutd_tottenham`'s gallery loses players under `percrop_names` (24 -> 13 distinct named
   players), while liverpool holds (18 -> 19) and brighton grows (20 -> 27). The 0.694 team factor
   and the 3/11 team term on tottenham come from that. Not diagnosed.

## 8. Files

- `tools/ocr_match.py` -- per-crop OCR on real broadcast; `--report` now also measures agreement
  with the anchor chain's numbers on the tracklets both mechanisms read.
- `tools/identity_match.py` -- `anchor_reads` / `percrop_reads` (the evidence source is now a
  parameter), `--percrop`, `PERCROP_FLOOR`.
- `tools/identity_carrier.py` -- `percrop_namer` (Stage-1 rule on per-crop evidence), `--arm`,
  `--names-dir`.
- `tools/gta_carrier.py` -- LOTO now grades against the **anchor truth** rather than the arm's own
  gallery label (the self-consistency trap), `loto_top1` derives from `loto_query_hits`,
  `run_loto` takes extra arms and multiple controls; `propagate_names` takes an explicit base set.
- `eval/gsr_identity.py` -- `--dump-config` persists the DEV-refit digit prior.
- `tests/test_gta_link.py` -- one new test pinning the anchor-truth grading.
- Artifacts: `outputs/{manutd_tottenham,manutd_brighton}/final/ocr_percrop/*.parquet`,
  `outputs/identity/solver_percrop/*_solver_names.parquet`, `outputs/identity_carrier_percrop/`,
  `results/identity_solver_config_percrop.json`, `results/identity_match_solver_percrop.json`,
  `results/identity_carrier_percrop.json`, `results/identity_carrier_percrop_names.json`,
  `results/ocr_realmatch_loto.json`, `results/ocr_match_manutd_{tottenham,brighton}.json`.
