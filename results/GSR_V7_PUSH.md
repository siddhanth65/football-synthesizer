# v7 THE PUSH — the confirmatory TEST-38 retest of the V3 near-miss (pre-registered)

Server `a100server1` GPU 1 for the evidential vote passes; laptop CPU for the arms, the pairing and
the scoring. Same-machine pairing throughout (`results/CLUSTER_MIGRATION.md`): every TEST-38 GPU
artifact — the control's and both arms' — comes off the server; every CPU stage runs on the laptop.

Registration frozen at **2026-08-07T17:21:47Z**, `results/gsr_benchmark/gsr_v7_push_registration.json`,
written before the evidential TEST-38 per-crop pass was launched and before any TEST-38 artifact of
either arm existed.

---

## 1. REGISTRATION (echo of the frozen JSON — nothing below was re-tuned)

### 1.1 Why this session exists

`results/GSR_V7_V3.md` §2.8 measured the evidential arm `edl_fuse94` at **DEV-20 +1.2562 GS-HOTA on
11 of 20 sequences** against a pre-registered gate of `>= +1.0` AND `>= 12/20` — a FAIL by one
sequence, with an honest multiplicity note (nine DEV arms scored, p = 0.0355 uncorrected against a
Bonferroni threshold of 0.00625) and an honest selection note (§2.10.6: head hyperparameters were
chosen after the DEV-20 rung-1 rows had been computed). `results/GSR_V7_V4S2.md` §2.4 measured the
EIoU retune point `e0.3 r1 w_app0.7 app_max0.30` at **DEV-20 +0.4853 on 12 of 20** — the mirror
failure, breadth without level, p = 0.0401 over six swept points against a Bonferroni threshold of
0.0083.

Both numbers are DEV numbers chosen from sweeps. `results/GSR_V5.md`'s v5.1 lesson (+0.61 DEV ->
-0.05 TEST-38) is exactly this size of effect. **The only thing that settles it is a held-out read at
a frozen configuration, and that is what this session spends.**

### 1.2 The two arms, frozen as measured

| arm | reader evidence | aggregation rule | association |
|---|---|---|---|
| **ARM 1** `edl_fuse94` | `_v7e_v6det` (evidential gate head on the frozen arm-4t trunk) | `conf 0.5 / votes 2 / leg 0.5 / max_u 0.60 / max_p_none 0.30 / fuse true` | e 0.3, rounds 1, **w_app 0.5**, app_max 0.30 |
| **ARM 2** `edl_fuse94_w07` | same | same | e 0.3, rounds 1, **w_app 0.7**, app_max 0.30 |

Everything else is the v6 chain verbatim: S4b detector (`gsr_v3_ft_b_last.pt`, md5
`2074d874...`), PARSeq v6 arm-4t trunk (md5 `39c0c15d...`), evidential head
`~/data/edl/edl_head_v7e.pt` (md5 `4cfdaea31b0a9febf984d524da25097c`, re-verified on the server at
registration time), connector `tau 0.450`, jersey-compatible merge on, `team_src="free"`,
`roster="self"`, aggregation floor 0.80, the official scorer.

### 1.3 Hypotheses, test, gate, ship rule

- **H1 / H2:** each arm beats the v6 TEST-38 control (49.4970, re-derived on this stack first).
- **Test:** one-sided paired Wilcoxon signed-rank (`alternative="greater"`) on the 38 per-sequence
  GS-HOTA deltas; **Bonferroni alpha 0.025 per arm**.
- **SHIP GATE, per arm, all three required:** TEST-38 GS-HOTA `>= 50.5` **AND** corrected-significant
  (`p <= 0.025`) **AND** `>= 22/38` sequences helped.
- **Ship rule:** the higher-scoring arm that clears its full gate; if both clear, ARM 2 only if it
  beats ARM 1, else the simpler arm. **If neither clears, the push ends** — v7 ships nothing, this
  document is the final registered negative, and no third arm and no threshold surgery follows.
- **If an arm ships:** ONE test-49 run against the v6 test-49 control (53.0846), packaged to the full
  v6 standard (zip self-score identical, materialized GT-free legitimacy audit, manifest with every
  component md5). Nothing is uploaded.

### 1.4 Guards carried in

1. `tools.gsr_eiou.BOX_SUBDIR` is rebound to `detbox_cache_v6det` and the harness raises if the cache
   is absent — the `GSR_V7_V3.md` §2.7 landmine that cost a 55-minute wasted arm.
2. Every arm rebuilds identity bundles, the GTA connector arm and the densified votes
   (`clear_arm_caches`), so no arm inherits another's partition.
3. Same-machine pairing (§ header). The evidential per-crop pass runs at `--leg-thresh 0.0`, exactly
   as V3 ran it; the rule's `min_legibility 0.5` applies the gate at aggregation time.
4. No `METRICS_VERSION` bump: the reader and the association are pipeline stages, not shipped metrics.

---

## 2. RESULTS — VERDICT: **NEITHER ARM CLEARS ITS GATE. The push ends. v7 ships nothing.**

ARM 1 scores **TEST-38 49.8722 (+0.3753, 16/38 helped, one-sided p 0.2203)** against a gate of
`>= 50.5` AND `p <= 0.025` AND `>= 22/38`: it fails all three halves. ARM 2 scores **50.1252
(+0.6282, 23/38 helped, one-sided p 0.0421)**: it clears the breadth half alone. Per the
pre-declared ship rule, **no test-49 run was spent, no package was built, and no v7 artifact enters
any freeze.**

The DEV-20 near-miss did not survive the held-out read: **+1.2562 on DEV became +0.3753 on TEST-38,
30% of the effect.** That is the `results/GSR_V5.md` v5.1 pattern (+0.61 DEV -> -0.05 TEST-38)
repeating at a larger DEV effect size, and it is the reason the confirmatory retest was registered.

### 2.0 Control re-derivation — EXACT, to every digit

Re-derived on this session's stack through the *same harness the arms run through*
(`tools.gsr_v7_edl --arm <incumbent rule> --reader v6`), not by re-reading the on-record file:

| metric | on record (`gsr_v6det_t38_v6det.json`) | re-derived | delta |
|---|---|---|---|
| GS-HOTA | 49.49695486974895 | **49.49695486974895** | **0** |
| GS-DetA | 35.63306799324153 | 35.63306799324153 | 0 |
| GS-AssA | 68.75904488927765 | 68.75904488927765 | 0 |
| GS-LocA | 93.33778047383011 | 93.33778047383011 | 0 |
| IDF1 | 52.682973889697415 | 52.682973889697415 | 0 |

**Max |per-sequence delta| over the 38 sequences: 0.0** — 0 of 38 differ. The re-derived payload is
`results/gsr_benchmark/gsr_v7_push_t38_control.json` and it, not the on-record file, is what both
arms are paired against.

### 2.1 The crop-set equivalence check — STRONGER than V3's

The evidential per-crop pass ran server-side (GPU 1, 4 shards, 38 sequences); the on-record v6
per-crop artifact for TEST-38 was produced on that same server. `GSR_V7_V3.md` §2.5 could only
reach 0.15%-of-crops agreement because its two arms straddled two machines. Here both are
server-side, and the check is correspondingly tight
(`results/gsr_benchmark/gsr_v7_push_cropcheck_t38.json`):

| | |
|---|---|
| crops, `_v6_v6det` / `_v7e_v6det` | 99,932 / 99,932 |
| **paired on `(track_id, frame)`** | **99,932 / 99,932 = 100.00%** |
| max \|delta legibility\| | **0.000244** (one fp16 ulp) |
| crops with \|delta legibility\| > 0.01 | **0** |
| the shipped `leg >= 0.5` gate flips | 102 crops (**0.102%**) — all on crops sitting exactly on the 0.5 boundary, which one fp16 ulp is enough to move |
| torso-flag agreement on crops legible in both | 19,844 / 19,857 = **0.99935** |

### 2.2 The two arms on TEST-38

38 held-out sequences, everything frozen but the per-crop evidence (both arms) and the appearance
weight (ARM 2). Paired against the §2.0 re-derivation; `p1` is the registered one-sided signed-rank
p, `p2` the two-sided p every earlier session quoted, shown so the two are never confused.

| arm | GS-HOTA | delta | GS-DetA | GS-AssA | GS-LocA | IDF1 | helped/hurt/tied | **p1** | p2 |
|---|---|---|---|---|---|---|---|---|---|
| **control (v6)** | **49.4970** | — | 35.633 | 68.759 | 93.338 | 52.68 | — | — | — |
| **ARM 1** `edl_fuse94` | 49.8722 | **+0.3753** | 36.199 | 68.714 | 93.284 | 53.29 | **16 / 14 / 8** | **0.2203** | 0.4405 |
| **ARM 2** `+ w_app 0.7` | **50.1252** | **+0.6282** | 36.526 | 68.793 | 93.303 | 53.56 | **23 / 14 / 1** | **0.0421** | 0.0841 |

| gate criterion | required | ARM 1 | ARM 2 |
|---|---|---|---|
| level | TEST-38 `>= 50.5` | 49.8722 **FAIL** (-0.63) | 50.1252 **FAIL** (-0.37) |
| significance | one-sided p `<= 0.025` (Bonferroni) | 0.2203 **FAIL** | 0.0421 **FAIL** (1.7x over) |
| breadth | `>= 22/38` helped | 16/38 **FAIL** | 23/38 **PASS** |
| **verdict** | all three | **FAIL (0 of 3)** | **FAIL (1 of 3)** |

**Ship rule applied:** neither arm clears its full gate, so the push ends. No third arm, no threshold
surgery, no test-49 spend, no package. The v6 bundle ships unchanged.

Secondary numbers, for the record:

| | control | ARM 1 | ARM 2 |
|---|---|---|---|
| EIoU tracks before -> after | 2,511 -> 4,398 | 2,511 -> 4,398 | 2,511 -> **4,362** |
| connector merges / precision | 2,622 / 0.4885 | 2,619 / 0.4923 | 2,564 / **0.4959** |
| named tracklets / total | 1,197 / 1,714 | 1,206 / 1,717 | **1,212 / 1,733** |
| admissible slots per sequence | 13.26 | 13.34 | 13.34 |
| pooled identity / jersey | 0.5660 / 0.6207 | 0.5730 / 0.6298 | **0.5773 / 0.6352** |
| concentration (top-2 share of net) | — | 0.756 | 0.456 |

The `GSR_V5.md` §2.1 concentration guard does not fire on either arm (both < 0.80), but ARM 1's
0.756 is worth stating plainly: **two of 38 sequences (SNGS-029 +9.16, SNGS-022 +6.14) carry 76% of
its net +20.22**, and eight sequences move by exactly 0.0000 — on those the evidential votes changed
nothing the solver could use.

### 2.3 The mechanism the V3 session claimed does NOT transfer — and this is the sharpest finding

`GSR_V7_V3.md` §2.9 made a specific, falsifiable mechanistic claim: at matched read density the
evidential reader buys read *precision*, and on this chain 0.0073 of read precision is worth ~1.26
GS-HOTA. TEST-38 falsifies the premise. Graded against the GSR jersey GT on the identical
`positions_gate_v6det` track partition, each reader at its own arm's rule
(`results/gsr_benchmark/gsr_v7_push_readgrade_t38.json`):

| split | reader | d | read precision |
|---|---|---|---|
| DEV-20 (on record, `GSR_V7_V3.md` §2.9) | v6 + incumbent rule | 0.4114 | 0.9365 |
| DEV-20 (on record) | **evidential + `fuse94`** | 0.4104 | **0.9438 (+0.0073)** |
| **TEST-38 (this session)** | v6 + incumbent rule | 0.3512 (834/2,375) | **0.9079** |
| **TEST-38 (this session)** | **evidential + `fuse94`** | 0.3528 (838/2,375) | **0.9009 (-0.0070)** |

**The precision advantage does not merely shrink — it changes sign.** On DEV-20 the evidential arm
read +0.0073 more precisely at matched density; on TEST-38 it reads **0.0070 less** precisely, at a
density matched to 0.0016. The 450-point DEV rung-2 sweep that chose `fuse94` was fitting DEV's
density/precision frontier, not a transferable property of the reader.

Which leaves ARM 1's +0.3753 needing a different explanation than the one V3 gave, and the arm's own
numbers supply it: **GS-DetA +0.566 with GS-AssA -0.045**, named tracklets 1,197 -> 1,206, slots per
sequence 13.26 -> 13.34. The gain that did transfer is a small *coverage* gain in the admissible set
— which is `GSR_V7_V1.md`'s admissible-set hypothesis, the very thing V3 §2.9 declared "not
supported by this session's evidence". On the held-out split the two sessions' conclusions swap
places. Neither effect is large enough to ship.

### 2.4 What the appearance weight is worth on the evidential partition

ARM 2 minus ARM 1 isolates the V4s2 association point on identical evidence:

| | GS-HOTA | GS-DetA | merges / precision | tracks after |
|---|---|---|---|---|
| ARM 1 (`w_app 0.5`) | 49.8722 | 36.199 | 2,619 / 0.4923 | 4,398 |
| ARM 2 (`w_app 0.7`) | 50.1252 | 36.526 | 2,564 / 0.4959 | 4,362 |
| **isolated effect** | **+0.2530** pooled | +0.327 | -55 merges at +0.0036 precision | -36 |

Per-sequence: mean **+0.1874**, **20/38 helped**, one-sided p **0.1102**. On DEV-20 the same knob
was worth +0.4853 over the v6 reader (`GSR_V7_V4S2.md` §2.4); on TEST-38, over the evidential
reader, it is worth **+0.2530** — about half, and not significant on its own. It is nevertheless the
larger of the two halves' surviving effects in *breadth* terms: ARM 2 is the only arm in this
session that clears any half of the gate, and it does so on the association knob, not the reader.

### 2.5 Per-sequence table (all 38, both arms)

| seq | control | ARM 1 | d1 | ARM 2 | d2 |
|---|---|---|---|---|---|
| SNGS-022 | 60.8877 | 67.0249 | **+6.137** | 67.1636 | **+6.276** |
| SNGS-023 | 38.6204 | 41.7371 | +3.117 | 41.7924 | +3.172 |
| SNGS-025 | 55.1205 | 53.7644 | -1.356 | 53.9419 | -1.179 |
| SNGS-026 | 63.0974 | 63.7257 | +0.628 | 63.6717 | +0.574 |
| SNGS-028 | 62.6519 | 62.6519 | 0.000 | 59.6193 | -3.033 |
| SNGS-029 | 41.0420 | 50.1997 | **+9.158** | 47.2238 | +6.182 |
| SNGS-031 | 54.2382 | 56.7215 | +2.483 | 57.1500 | +2.912 |
| SNGS-032 | 50.3525 | 52.9970 | +2.644 | 52.9501 | +2.598 |
| SNGS-034 | 43.2237 | 44.4759 | +1.252 | 44.9567 | +1.733 |
| SNGS-035 | 76.8387 | 76.8387 | 0.000 | 76.8387 | 0.000 |
| SNGS-037 | 59.2923 | 58.9919 | -0.300 | 58.9919 | -0.300 |
| SNGS-038 | 62.4207 | 62.0323 | -0.388 | 63.7804 | +1.360 |
| SNGS-040 | 28.3623 | 28.2067 | -0.156 | 29.6939 | +1.332 |
| SNGS-041 | 39.5807 | 39.5807 | 0.000 | 41.1916 | +1.611 |
| SNGS-043 | 42.1454 | 41.9169 | -0.228 | 41.8902 | -0.255 |
| SNGS-044 | 38.2962 | 35.5661 | -2.730 | 35.9892 | -2.307 |
| SNGS-046 | 39.4703 | 40.1414 | +0.671 | 39.8360 | +0.366 |
| SNGS-047 | 38.7102 | 34.4010 | **-4.309** | 36.2370 | -2.473 |
| SNGS-049 | 43.7106 | 46.8840 | +3.173 | 45.6168 | +1.906 |
| SNGS-050 | 29.9484 | 29.8402 | -0.108 | 30.7436 | +0.795 |
| SNGS-052 | 48.9485 | 48.9527 | +0.004 | 52.3400 | +3.391 |
| SNGS-053 | 44.5431 | 47.8178 | +3.275 | 48.6194 | +4.076 |
| SNGS-055 | 23.1964 | 25.5399 | +2.343 | 26.6239 | +3.428 |
| SNGS-056 | 58.1116 | 57.2437 | -0.868 | 58.4071 | +0.296 |
| SNGS-058 | 39.5815 | 40.3647 | +0.783 | 40.8315 | +1.250 |
| SNGS-059 | 44.9534 | 44.9534 | 0.000 | 47.8110 | +2.858 |
| SNGS-079 | 37.6431 | 37.6431 | 0.000 | 33.8232 | -3.820 |
| SNGS-080 | 64.1422 | 64.1422 | 0.000 | 64.0827 | -0.060 |
| SNGS-082 | 60.7732 | 56.5568 | -4.216 | 52.9352 | **-7.838** |
| SNGS-083 | 64.2567 | 62.7298 | -1.527 | 65.6140 | +1.357 |
| SNGS-085 | 62.9989 | 63.0129 | +0.014 | 60.4048 | -2.594 |
| SNGS-086 | 53.2853 | 53.1007 | -0.185 | 56.9568 | +3.671 |
| SNGS-088 | 71.9780 | 71.9780 | 0.000 | 71.0209 | -0.957 |
| SNGS-089 | 52.4292 | 53.7287 | +1.300 | 53.3702 | +0.941 |
| SNGS-091 | 49.0182 | 54.8603 | **+5.842** | 54.8193 | +5.801 |
| SNGS-092 | 18.3716 | 18.3716 | 0.000 | 18.3702 | -0.001 |
| SNGS-094 | 72.3508 | 69.4488 | -2.902 | 69.9688 | -2.382 |
| SNGS-095 | 61.2749 | 57.9443 | -3.331 | 57.9303 | -3.345 |

---

## 3. VERDICT AND WHAT IT CLOSES

**Both registered arms FAIL. v7 ships nothing. The v6 bundle — TEST-38 49.4970, test-49 53.0846,
package `gsr_testphase_gtfree_v6_28f3986b.zip` — remains the campaign's shipped submission,
untouched.**

| arm | TEST-38 | delta | helped | one-sided p | gate |
|---|---|---|---|---|---|
| ARM 1 `edl_fuse94` | 49.8722 | +0.3753 | 16/38 | 0.2203 | **FAIL** (0 of 3) |
| ARM 2 `+ w_app 0.7` | 50.1252 | +0.6282 | 23/38 | 0.0421 | **FAIL** (breadth only) |
| v6 (ships) | **49.4970** | — | — | — | — |

**What this closes.** The v7 reader lever is closed by measurement, not by budget. Three sessions
priced it: V3 measured the DEV effect and honestly flagged its selection exposure; this session
spent the held-out read and found that (a) two-thirds of the DEV effect is gone, (b) the *mechanism*
V3 named — a transferable read-precision advantage — is not merely absent but reversed in sign on
held-out data (§2.3), and (c) what little survives is a coverage effect the same session had
declared unsupported. A DEV gain of +1.26 chosen from nine end-to-end arms over a 450-point
aggregation sweep buys about +0.38 held out. **That ratio, measured, is the transferable result of
this session**, and it is the number to price the next DEV sweep against.

**What it does not close.** The appearance weight (§2.4) is the only knob in the whole v7 campaign
that has now moved the score in the right direction on *two independent splits* (DEV +0.4853 on the
v6 reader, TEST-38 +0.2530 on the evidential reader, and +0.6282 in ARM 2's total over the control
at 23/38 breadth). It is under-powered at every measurement and it was never the registered
headline. If anything from v7 is worth a v8 registration, it is a properly powered single-knob test
of `w_app` — one arm, one gate, no reader change — and not another reader.

## 4. Negatives, limits, and what was NOT done

1. **The headline is two FAILs.** Nothing in this session is a candidate for any bundle.
2. **The V3 mechanism does not reproduce** (§2.3): the read-precision advantage flips sign
   (+0.0073 DEV -> -0.0070 TEST-38) at matched density. The DEV rung-2 sweep was fitting DEV.
3. **ARM 1 is concentrated**: 76% of its net comes from two of 38 sequences, and it is exactly
   0.0000 on eight — the median per-sequence delta is **0.000**.
4. **ARM 2 was never measured on DEV-20 as a combination.** Its two halves were measured separately
   in two different sessions. The registration says so, and it means ARM 2's TEST-38 read is its
   *first* read of any kind — which cuts both ways: no DEV selection exposure, but also no prior.
5. **ARM 2's -7.838 on SNGS-082** is the worst single-sequence regression in the session, and
   SNGS-082 is hurt by both arms (-4.216 / -7.838). Not diagnosed.
6. **One-sided vs two-sided.** Every earlier v7 session quoted the two-sided Wilcoxon p. This
   registration specified one-sided, which is the *more permissive* choice, and ARM 2 still misses
   0.025 by 1.7x. Under the two-sided convention of V3/V4s2 it would miss by 3.4x.
7. **`min_legibility` was fixed at the arms' registered 0.5** and the per-crop pass ran at
   `--leg-thresh 0.0` exactly as V3 ran it, so the arms consume the identical superset of crops.
   No legibility sweep was run on TEST-38 and none was permitted.
8. **Not done:** test-49 (not earned — the gate closed it), any package, any commit, any
   `METRICS_VERSION` bump, any third arm, any re-tuned threshold, any change to the detector, the
   trunk reader, the connector `tau` or the solver.
9. **The connector `tau` is still not re-swept** on any of these partitions (`GSR_EIOU.md`
   negative 4, restated by V4S1 §6.7 and by V4S2 §4.10). Carried forward again.
10. **One seed, one head, one run per arm.** No repeat, no confidence interval beyond the paired
    signed-rank.

## 5. Spend

| stage | where | cost |
|---|---|---|
| evidential per-crop pass, 38 sequences, `--leg-thresh 0.0`, 4 shards | a100server1 GPU 1 | **~35 min wall, ~0.6 GPU-h** (GPU 0 585 MiB / another user and never touched; GPU 1 was 4 MiB idle at launch) |
| v6 TEST-38 control re-derivation | laptop CPU | 299 s |
| ARM 1 end-to-end | laptop CPU | 288 s |
| ARM 2 end-to-end | laptop CPU | 291 s |
| crop-equivalence + read grading | laptop CPU | ~2 min |
| **total GPU** | | **~0.6 h** |

Disk: `outputs/gsr_srv/koshkina_percrop_v7e_v6det` 17 MB (laptop),
`~/football-synthesizer/outputs/gsr/koshkina_percrop_v7e_v6det` 25 MB (server, now 58 sequences).
No test-49 artifact was created.

## 6. Files

- `results/gsr_benchmark/gsr_v7_push_registration.json` — the frozen registration
  (2026-08-07T17:21:47Z), written before the per-crop pass was launched.
- `results/gsr_benchmark/gsr_v7_push_t38.json` — the control re-derivation and both arms, full
  payloads (per-sequence GS-HOTA, paired stats one- and two-sided, EIoU, connector, admissible set).
- `results/gsr_benchmark/gsr_v7_push_t38_control.json` — the re-derived control, extracted as the
  payload both arms are paired against.
- `results/gsr_benchmark/gsr_v7_push_cropcheck_t38.json` — the crop-set equivalence check.
- `results/gsr_benchmark/gsr_v7_push_readgrade_t38.json` — the TEST-38 read density/precision
  grading behind §2.3.
- `outputs/gsr_srv/koshkina_percrop_v7e_v6det/` — 38 evidential per-crop parquets (`alpha`, `u`).
- Code (one file, default-preserving): `tools/gsr_v7_edl.py` — `run_arm` gained `w_app` and
  `keep_arm_dir`; `paired_vs_control` gained the one-sided p and a `control` argument; the CLI
  gained `--split`, `--w-app`, `--control`, `--arms-path`, `--keep-arm-dir`. Every existing default
  reproduces the V3 behaviour; `--demo` passes; post-edit md5 `a257a2934a6d560352d1268df888cda0`.
- Server: `~/run_edl_t38.sh`, `~/football-synthesizer/logs/v7push_edl_t38_{1..4}.log`.
- Rule files (the `GSR_V7_V0.md` §7 hazard — a rule file's *filename* silently re-points every
  future arm at that variant): `results/ocr_density_rule_v7e_v6det_eiou.json` is rewritten by each
  `--arm` invocation and now names `fuse94`; read `gsr_v7_push_t38.json` for what any number was
  produced with, never this file. `results/ocr_density_rule_v6_v6det_eiou.json` was rewritten by
  the control run and came out **byte-identical** (git reports it unmodified) — the "provable
  no-op" V3 claimed for it, now re-verified.
- Claims: `v7-push-001` (ARM 1, FAIL), `v7-push-002` (ARM 2, FAIL), `v7-push-003` (the mechanism
  reversal, §2.3).

## 7. Reproduce

```
# server (GPU 1)
bash ~/run_edl_t38.sh                       # the evidential per-crop pass over TEST-38
# laptop (CPU); copy the 38 parquets into outputs/gsr_srv/koshkina_percrop_v7e_v6det first
python -m tools.gsr_v7_edl --arm '{"min_crop_conf":0.9,"min_votes":3,"min_legibility":0.5,
    "emit_all":false}' --reader v6 --name v6_control_t38 --split t38 --out-dir outputs/gsr_srv \
    --control results/gsr_benchmark/gsr_v6det_t38_v6det.json \
    --arms-path results/gsr_benchmark/gsr_v7_push_t38.json
python -m tools.gsr_v7_edl --arm '{"min_crop_conf":0.5,"min_votes":2,"min_legibility":0.5,
    "emit_all":false,"max_u":0.6,"max_p_none":0.3,"fuse":true}' --reader edl \
    --name arm1_edl_fuse94 --split t38 --out-dir outputs/gsr_srv --w-app 0.5 \
    --control results/gsr_benchmark/gsr_v7_push_t38_control.json \
    --arms-path results/gsr_benchmark/gsr_v7_push_t38.json
#   ...the same with --w-app 0.7 --name arm2_edl_fuse94_w07 for ARM 2
pytest tests/test_gsr_eiou.py tests/test_ocr_density.py tests/test_evidential_jersey.py \
    tests/test_jersey_id.py        # 28 passed
```
