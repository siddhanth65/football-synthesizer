# v7 session V1 — the dense-regime solver retune (pre-registered)

Laptop, CPU only, no GPU stage. Control: `results/gsr_benchmark/gsr_v7_control_dev.json`
(DEV-20 GS-HOTA **51.808280888021784**, pinned stack, `results/GSR_V7_V0.md`).

---

## 1. REGISTRATION (written before any arm was run)

### 1.1 Hypothesis

`results/EVIDENCE_DENSITY_LAW.md` measures that the value of identity *constraints* rises with
evidence density: at d = 0.087 the solver's joint solve buys a dial and nothing else
(`IDENTITY_SOLVER_STAGE2.md` §5c, `RETEST_ABSTENTION.md` §3.1, both measured three times), while the
frontier only opens up above d* = 0.347 (read precision 0.86) / d* = 0.268 (0.95). The v6 GSR chain
now runs at **d ~ 0.31 at 0.926 measured read precision** — the densest regime this project has ever
solved in, and materially denser than the d = 0.087-0.103 regime every existing solver calibration
was fitted on.

Every shipped solver prior is PRTreID-era and demonstrably stale against that measurement:

| knob | shipped (`results/identity_solver_config_percrop.json`) | what has since been measured |
|---|---|---|
| `p_correct` | 0.8837209302325582 | per-crop read precision **0.926** (arm A, 0.9256) |
| `r_abstain` | 0.0 | the DEV frontier's knee is 0.05-0.10 (`RETEST_ABSTENTION.md` §1) |
| `pi_none` | 0.7 | never re-fitted since the sparse regime |
| `topk` | 3 | never swept end-to-end on CLIP embeddings |
| `app_gain` | 10.0 | fitted to PRTreID cosine units, arm now runs on CLIP |
| `sim_none` | 0.92 | ditto |

**H1 (the law's dense-regime prediction):** at d ~ 0.31 the solver's priors are mis-specified in the
direction of *under*-trusting the OCR channel and *over*-abstaining-by-accident, so a coordinate
retune moves end-to-end DEV-20 GS-HOTA by >= +1.0.

**H0 (the law-consistent null):** d = 0.31 is still below d* = 0.347, so the constraint machinery is
still in the regime where the joint solve buys a dial, not a frontier — and no solver knob moves
GS-HOTA by >= +1.0. **A null here is a PASS for the law, not a failure of the session**, and it is
pre-declared as such: the outcome of this session is a *measurement of where the law's knee sits on
the real GSR chain*, whichever way it lands.

### 1.2 The sweep grid (coordinate, around the incumbent)

Incumbent = the frozen config, unchanged. Each coordinate arm changes **exactly one** field.

| knob | values | incumbent | non-incumbent arms |
|---|---|---|---|
| `p_correct` | 0.8837, 0.926, 0.95 | 0.8837 | 2 |
| `r_abstain` | 0, 0.02, 0.05, 0.10 | 0 | 3 |
| `pi_none` | 0.5, 0.7, 0.85 | 0.7 | 2 |
| `topk` | 3, 5 | 3 | 1 |
| `app_gain` | 5, 10 | 10 | 1 |
| `sim_none` | 0.90, 0.92, 0.95 | 0.92 | 2 |

**11 coordinate arms + 1 control.** Then ONE small joint grid (<= 6 arms) over the two knobs with
the largest |mean per-sequence delta| in the coordinate pass, crossing each knob's incumbent value
with its best coordinate value(s).

### 1.3 What an arm is

Full DEV-20 through the v6 chain, everything except the solver config frozen: the same
`positions_gate_v6det_eiou` partition, the same `_v6_v6det_eiou` per-crop evidence, the same
0.80-floor aggregation rule, the same GTA connector at tau 0.450 with the jersey gate, the same
`team_src="free"` / `roster="self"` legitimacy standard, the same official scorer. Only
`generator.identity_solve.SolverConfig` moves, via a new default-preserving override hook in
`tools.gsr_v4.solver_config`.

The GPU-cached and CPU-cached upstream stages (EIoU relink, densified votes, evidence bundles, GTA
arm) are **identical by construction across every arm** — no solver knob is an input to any of them —
so they are computed once and shared. The equivalence is not assumed: the control arm is run through
the sweep harness and must reproduce **51.808280888021784 exactly** before any arm is scored. If it
does not, the session stops.

### 1.4 Selection rule and guards (the v5.1 anti-pattern)

1. **Selection is on end-to-end DEV-20 GS-HOTA and nothing else.** No component metric (jersey
   precision, coverage, named-tracklet count, pooled identity) may select a point. They are reported.
2. **Gate: delta >= +1.0 GS-HOTA vs the V0 control AND >= 12/20 sequences helped.** Both, not either.
3. **Concentration demotion:** a point whose two largest per-sequence gains sum to >= 80% of its
   total net gain is demoted (it is one or two lucky sequences, not a regime effect).
4. **ONE point is carried.** If several pass, the one with the largest mean per-sequence delta;
   ties broken by sequences-helped.
5. **One confirmatory paired re-run** of the selected point against the V0 control on the same
   machine and stack before anything is declared.
6. **FAIL response, pre-declared:** ship the incumbent config untouched, record the null as the
   law's dense-regime measurement, and write no override file into any frozen bundle.
7. No TEST-38 read, no test-49 read, no GPU, no commit.

### 1.5 The known GK coupling (pre-declared, reported either way)

`results/RETEST_ABSTENTION.md` §4.3: on real matches, `r_abstain > 0` **zeroed goalkeeper naming**
(2 of 4 keeper slots -> 0 of 4, in all three matches) because a keeper's number is never read and his
likelihood comes from the role gate alone, which cannot clear a positive abstain floor. GS-HOTA is
this session's objective, but any winning point must state what it spent: **GK-role named tracklet
counts are reported per arm** beside GS-HOTA, and a point that wins GS-HOTA while zeroing GKs gets
that trade-off written into the verdict explicitly.

Structural note recorded at registration time (verified in code, `tools/gsr_deleak.py:roster_self`
and `write_arm`): on the GSR chain the roster is built from the sequence's own reads and every slot
it contains is `role="player"`; `write_arm` only writes a jersey onto predictions whose role is
`player`. So the GSR arm has no GK roster slot to lose. The reported GK column is therefore
"GK-role-dominant tracklets that the solver named at all", which is the closest observable analogue
on this chain, and its expected value under the incumbent is already low. This is stated **before**
the numbers so the column cannot be reinterpreted after the fact.

### 1.6 Files this registration binds

- Harness: `tools/gsr_v7_solver.py` (new), override hook in `tools/gsr_v4.py`.
- Raw: `results/gsr_benchmark/gsr_v7_v1_sweep.json` (every arm, per-sequence GS-HOTA, paired stats).
- Selected point (only if the gate passes): `results/gsr_v7_solver_override.json`.

---

## 2. RESULTS — VERDICT: **FAIL against the gate. The incumbent solver config ships unchanged.**

18 arms (control + 11 coordinate + 6 joint), each a full DEV-20 run through the v6 chain, ~100 s
each, CPU only. **The best arm in the whole sweep is +0.0577 GS-HOTA on 7/20 sequences helped,
against a gate of >= +1.0 AND >= 12/20 — and it is demoted by the concentration guard anyway.**
Nine of the eighteen arms score *exactly* the control, to 16 significant figures.

### 2.1 The harness equivalence check (run before any arm)

The sweep shares the (solver-independent) upstream stages across arms. That shortcut was verified,
not assumed: the frozen config through the sweep harness gives GS-HOTA **51.808280888021784**
against the V0 control's **51.808280888021784** — **EXACT**, and per-sequence identical on all 20.
The confirmatory repeat run at the end of the session reproduced it again, per-sequence identical
(and 610 named tracklets, 25 GK-named, both runs).

### 2.2 The sweep table (sorted by GS-HOTA; every arm paired vs the V0 control)

`named` = merged tracklets the solver named (of 870); `GK` = GK-role-dominant tracklets named
(of 39); `conc` = share of the arm's net per-sequence gain carried by its two best sequences
(the >= 0.80 demotion guard).

| arm | GS-HOTA | delta | DetA | AssA | LocA | IDF1 | help/hurt | Wilcoxon p | conc | named | GK | jersey |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pi_none` 0.5 + `r_abstain` 0.05 | **51.8660** | **+0.058** | 39.138 | 68.736 | 93.703 | 56.48 | 7/13 | 0.841 | **1.92** | 278 | 1 | 0.6665 |
| `sim_none` 0.90 | 51.8083 | **+0.0000** | 39.308 | 68.287 | 93.652 | 56.45 | 0/0 | 1 | — | 610 | 25 | 0.6518 |
| `pi_none` 0.5 | 51.8083 | **+0.0000** | 39.308 | 68.287 | 93.652 | 56.45 | 0/0 | 1 | — | 610 | 25 | 0.6518 |
| `p_correct` 0.95 | 51.8083 | **+0.0000** | 39.308 | 68.287 | 93.652 | 56.45 | 0/0 | 1 | — | 610 | 25 | 0.6518 |
| `p_correct` 0.926 | 51.8083 | **+0.0000** | 39.308 | 68.287 | 93.652 | 56.45 | 0/0 | 1 | — | 610 | 25 | 0.6518 |
| **control (incumbent)** | **51.8083** | — | 39.308 | 68.287 | 93.652 | 56.45 | — | — | — | 610 | 25 | 0.6518 |
| `topk` 5 | 51.8081 | -0.0002 | 39.307 | 68.288 | 93.652 | 56.45 | 1/2 | 0.285 | — | 611 | 25 | 0.6518 |
| `app_gain` 5 | 51.8081 | -0.0002 | 39.307 | 68.288 | 93.652 | 56.45 | 1/2 | 0.285 | — | 609 | 25 | 0.6517 |
| `r_abstain` 0.02 | 51.7864 | -0.022 | 39.008 | 68.755 | 93.708 | 56.34 | 7/13 | 0.784 | 8.41 | 283 | 1 | 0.6648 |
| `pi_none` 0.5 + `r_abstain` 0.10 | 51.7685 | -0.040 | 38.962 | 68.787 | 93.706 | 56.38 | 7/13 | 0.648 | 6.35 | 271 | 0 | 0.6663 |
| `pi_none` 0.5 + `r_abstain` 0.02 | 51.7685 | -0.040 | 39.148 | 68.461 | 93.700 | 56.41 | 7/13 | 0.784 | 18.92 | 296 | 1 | 0.6673 |
| `sim_none` 0.95 | 51.7364 | -0.072 | 39.213 | 68.263 | 93.643 | 56.36 | 0/1 | 0.317 | — | 610 | 26 | 0.6486 |
| `pi_none` 0.85 | 51.7364 | -0.072 | 39.213 | 68.263 | 93.643 | 56.36 | 0/1 | 0.317 | — | 610 | 26 | 0.6486 |
| `pi_none` 0.85 + `r_abstain` 0.02 | 51.6966 | -0.112 | 38.867 | 68.765 | 93.697 | 56.28 | 7/13 | 0.648 | — | 273 | 0 | 0.6651 |
| `r_abstain` 0.05 | 51.5833 | -0.225 | 38.708 | 68.744 | 93.690 | 56.12 | 7/13 | 0.498 | — | 270 | 0 | 0.6631 |
| `r_abstain` 0.10 | 51.3082 | -0.500 | 38.212 | 68.895 | 93.675 | 55.78 | 7/13 | 0.261 | — | 262 | 0 | 0.6569 |
| `pi_none` 0.85 + `r_abstain` 0.05 | 51.2917 | -0.517 | 38.161 | 68.942 | 93.673 | 55.74 | 7/13 | 0.261 | — | 262 | 0 | 0.6561 |
| `pi_none` 0.85 + `r_abstain` 0.10 | **50.4571** | **-1.351** | 36.557 | 69.645 | 93.670 | 54.05 | 5/15 | **0.044** | — | 249 | 0 | 0.6351 |

The joint grid was `r_abstain` x `pi_none` — the two knobs with the largest |mean per-sequence
delta| in the coordinate pass (0.4237 and 0.0612). `sim_none` tied `pi_none` at 0.0612 exactly and
was dropped from the joint grid because §2.4 shows the two are **the same intervention**, not two.

### 2.3 The gate, applied

| criterion | required | best arm (`pi_none` 0.5 + `r_abstain` 0.05) | verdict |
|---|---|---|---|
| DEV-20 GS-HOTA delta | >= +1.0 | **+0.0577** | FAIL |
| sequences helped | >= 12/20 | **7/20** | FAIL |
| concentration | top-2 share < 0.80 of net gain | **1.92** (top-2 = +5.91 on a net of +3.07) | demoted |
| Wilcoxon p (reported, not a gate) | — | 0.841 | — |

That arm's per-sequence gain is +3.81 (SNGS-033) and +2.10 (SNGS-096) against losses of -2.12
(SNGS-036), -1.68 (SNGS-039), -1.17 (SNGS-081); positives sum to +10.03 and negatives to -6.96.
This is the `results/GSR_V5.md` §2.1 signature *exactly* — v5.1's +0.61 DEV win was two sequences
(SNGS-093 +5.50, SNGS-090 +4.44) and it turned into -0.05 on TEST-38. Here the same pattern appears
at a tenth of the effect size and does not even clear its own DEV gate, so it is not carried.

**Selection: the incumbent config.** No knob is changed. `results/gsr_v7_solver_override.json` is
written as the incumbent's values verbatim, so consuming it is a **provable no-op**
(verified: `load_solver_override(...)` then `solver_config() == SolverConfig.load(frozen)` -> True).

### 2.4 Why nine arms scored *exactly* the control — the mechanism, measured

`results/gsr_benchmark/gsr_v7_v1_inert_probe.json`: the same 870 merged tracklets re-solved under
each knob, counting **tracklet labels changed**, not score.

| knob change | tracklet labels changed (of 870) | newly named | newly abstained |
|---|---|---|---|
| `p_correct` 0.8837 -> 0.926 | **0** | 0 | 0 |
| `p_correct` 0.8837 -> 0.95 | 2 | 1 | 1 |
| `pi_none` 0.7 -> 0.5 | **0** | 0 | 0 |
| `pi_none` 0.7 -> 0.85 | 5 | 1 | 1 |
| `sim_none` 0.92 -> 0.90 | **0** | 0 | 0 |
| `sim_none` 0.92 -> 0.95 | 5 | 1 | 1 |
| `topk` 3 -> 5 | 6 | 1 | 0 |
| `app_gain` 10 -> 5 | 21 | 3 | 4 |
| `r_abstain` 0 -> 0.02 | **330** | 1 | **328** |
| `r_abstain` 0 -> 0.05 | 341 | 0 | 340 |
| `r_abstain` 0 -> 0.10 | 349 | 0 | 348 |

The null is not "the knobs are mis-measured", it is **structural, and it is visible in ten lines of
`generator/identity_solve.py`**:

1. `solve_bundle_scored` overwrites the unknown column of the payoff matrix with the constant
   `r_abstain` (`probs[:, -1] = cfg.r_abstain`). The *posterior* mass on unknown never competes —
   only the flat floor does.
2. `pi_none` enters `posterior` as a uniform prior `(1 - pi_none)/n` over **every** identity, and
   `sim_none` enters as `exp(app_gain * (s - sim_none))` with the unknown class credited `sim_none`
   itself, i.e. `exp(0) = 1`. Both therefore scale the **whole identity block** against the unknown
   entry by one factor per tracklet, and leave the ordering *within* the block untouched.
3. With the unknown entry's payoff pinned to `r_abstain = 0`, a common factor on the identity block
   cannot change any argmax and cannot change any MILP optimum. `pi_none` and `sim_none` are
   **the same knob** at this operating point — which is why `pi_none` 0.85 and `sim_none` 0.95
   produced *identical* numbers on every column of the table (51.7364, 610 named, 26 GK, jersey
   0.6486), and why both moved exactly 5 of 870 labels.
4. `p_correct` only sharpens `pc = p_correct*conf + (1-p_correct)*0.5` inside `read_likelihoods`; at
   `conf ~ 1` that is 0.942 -> 0.963 -> 0.975. It re-scales the read's own slot against its
   confusable neighbours, and on this evidence it never once re-ordered them (0 of 870 at 0.926).

So at `r_abstain = 0` the solver already sits in the **name-everything-admissible corner**, and the
only knob that can leave that corner (`r_abstain`) leaves it in the losing direction: it removes
328-348 names and adds at most 1.

### 2.5 What abstention actually buys, and what it costs (the r_abstain frontier, end-to-end)

`RETEST_ABSTENTION.md` measured this frontier on tracklet precision. This is the first time it has
been measured **end to end on GS-HOTA**:

| `r_abstain` | named / 870 | jersey precision | GS-DetA | GS-AssA | GS-HOTA |
|---|---|---|---|---|---|
| 0 (incumbent) | 610 | 0.6518 | **39.308** | 68.287 | **51.8083** |
| 0.02 | 283 | 0.6648 | 39.008 | 68.755 | 51.7864 |
| 0.05 | 270 | 0.6631 | 38.708 | 68.744 | 51.5833 |
| 0.10 | 262 | **0.6569**† | 38.212 | **68.895** | 51.3082 |

† precision peaks at `r_abstain` 0.02 and then falls again — beyond the knee, abstention starts
discarding correct names too.

Abstention is doing exactly what the frontier says: **+1.3 points of jersey precision and +0.6 AssA,
paid for with -1.1 DetA**, and GS-HOTA's per-row identity gate weights the coverage loss more
heavily than the precision gain. On this objective the trade is never worth taking.

### 2.6 Goalkeepers — the RETEST_ABSTENTION collapse reproduces on GSR

| `r_abstain` | GK-role tracklets named (of 39) | GK rows named |
|---|---|---|
| 0 (incumbent) | **25** | 3,268 |
| 0.02 | **1** | 507 |
| 0.05 | **0** | 0 |
| 0.10 | **0** | 0 |
| 0.02-0.10 with `pi_none` 0.85 | **0** | 0 |

`RETEST_ABSTENTION.md` §4.3 measured GK naming going 2-of-4 keeper slots -> 0-of-4 in all three ManU
matches at `r_abstain` 0.05. **The same collapse, at the same thresholds, now reproduces on a
completely different corpus** (SoccerNet GSR DEV-20, 39 GK-role tracklets): 25 -> 1 -> 0. The
mechanism claimed there — a keeper's number is never read, so his likelihood comes from the role
gate alone, which cannot clear a positive abstain floor — is corpus-independent, and this is its
second independent measurement.

**Honest scope of that GK number, as pre-declared in §1.5:** on the GSR chain the roster is
`roster_self`, whose slots are all `role="player"`, and `write_arm` writes a jersey only onto
`role="player"` predictions. Those 25 GK-role tracklets are being named with *player* roster slots
and their names are dropped before submission, so **the GK collapse is not what costs the r_abstain
arms their GS-HOTA** — that comes from the ~328 player tracklets that stop being named. The GK
column is a mechanism measurement here, not a cost line. It is reported because the pre-registration
required it either way, and because the reproduction is worth more than the GSR arm is.

### 2.7 The evidence-density law — the dense-regime test, and what it actually returned

Measured on the very bundles the arms solve (`gsr_v7_v1_inert_probe.json`):

| quantity | v6 GSR chain, DEV-20 | the law's reference |
|---|---|---|
| merged tracklets | 870 | — |
| tracklets carrying >= 1 read | 336 | — |
| **read density d (per merged tracklet)** | **0.386** | today's real-match 0.087-0.103 |
| reads per tracklet | 0.533 | — |
| row-weighted read coverage | 0.658 | — |
| read precision | ~0.926 (S3 arm A, 0.9256) | 0.86 / 0.95 rows of the law |
| identities offered per sequence | 13.2 | — |

**The chain is above d\*.** `EVIDENCE_DENSITY_LAW.md` puts d\* at 0.347 (read precision 0.86) and
0.268 (0.95); at 0.926 precision the interpolated bar is ~0.28-0.30, and this chain sits at
**0.386**. So H1's premise held — this genuinely is the densest regime this project has ever solved
in, and it is on the far side of the law's knee.

**And the retune still bought nothing.** H1 as stated (a coordinate retune moves DEV-20 by >= +1.0
in the dense regime) is **falsified**, not merely unconfirmed: 18 arms, best +0.058.

The reconciliation is the useful result, and it sharpens the law rather than contradicting it:

- The law's d\* is the density at which **coverage at a precision floor of 0.85** becomes reachable.
  It is a statement about the *evidence*, and it is about the precision/coverage **dial**.
- GS-HOTA does not sit anywhere on that dial. Its per-row identity gate rewards coverage so heavily
  that the optimum is pinned at the maximum-coverage end (§2.5: buying +1.3 precision for -1.1 DetA
  is a net loss). A solver whose abstain floor is already 0 is **already at that corner**, so
  crossing d\* changes what the dial *could* deliver and changes nothing about where the objective
  wants to sit.
- What binds instead is the **admissible set**: 13.2 identities per sequence, built from the
  sequence's own reads, and the mutex/concurrency constraints. 610 of 870 tracklets are named and
  the remaining 260 are not held back by calibration — they are held back by having no admissible
  slot. That is a roster/association problem, which is exactly where V2 (side classifier), V3
  (evidential reader -> more and better reads -> more slots) and V4 (association) act.

**So the law's dense-regime prediction is upheld in the form it was actually measured in (density
crossed d\*; the precision/coverage dial did move — jersey precision 0.6518 -> 0.6648 is available
for the taking) and refuted in the form this session pre-registered (that the dial's availability
converts into end-to-end GS-HOTA via solver priors). The conversion step is the missing link, and it
is missing because the objective does not want the trade.** This is a real, banked negative result
about the *interface* between the law and a coverage-weighted metric.

### 2.8 Negatives, limits, and what was NOT done

1. **The headline is a FAIL.** 18/18 arms below the +1.0 gate; 17/18 at or below the control.
2. **`app_gain` 3.0 was not in the pre-registered grid.** `GSR_V5.md` measured app_gain 3 as the
   DEV optimum (+0.61) on the *v5* chain, and this grid only carried {5, 10}. app_gain 5 moved 21 of
   870 labels for -0.0002 GS-HOTA, so the knob is nearly flat here — but the v5.1 optimum itself was
   never re-tested on the v6 chain, and this session did not close that. (It also never travelled:
   v5.1 was -0.05 on TEST-38.)
3. **Knobs deliberately left out of the registered grid:** `team_eps`, `role_eps`,
   `max_concurrent`, `use_*` ablations. §2.4 argues the interesting direction is *naming more*, and
   those three are the knobs that would do it — `role_eps` in particular is what stops a GK-role
   tracklet taking a player slot. Un-swept, and now the obvious follow-up if anyone re-opens this.
4. **20 sequences cannot resolve 0.058.** The best arm's Wilcoxon p is 0.841. Nothing in the
   positive column of this table is distinguishable from noise, and the gate was set at +1.0
   precisely so that it could not be.
5. **No TEST-38 read, no test-49 read, no GPU, no commit, no `METRICS_VERSION` bump** (no metric
   changed).
6. **The sweep shares upstream stages across arms.** Legitimate because no solver knob is an input
   to the EIoU re-link, the votes, the bundles or the connector — and verified by the exact control
   reproduction (§2.1) — but it does mean these arms were not each run through
   `tools.gsr_v6det --stages arm` end to end from cold caches. The one arm that was (the control,
   in V0) is byte-identical.
7. **The 2026-08-07 mid-session process death** killed the joint grid after 2 of 6 arms; the sweep
   is resumable by arm name and the remaining 4 were re-run from disk state. No arm was scored
   twice except the control (deliberately, §2.1).

## 3. Files

- `results/gsr_benchmark/gsr_v7_v1_sweep.json` — all 18 arms, per-sequence GS-HOTA, paired stats,
  role diagnostics, the override each carried.
- `results/gsr_benchmark/gsr_v7_v1_inert_probe.json` — the label-change probe of §2.4 and the
  density measurement of §2.7.
- `results/gsr_v7_solver_override.json` — the selected point = the incumbent, verbatim; loading it
  is a provable no-op.
- `tools/gsr_v7_solver.py` (new), `tools/gsr_v4.py` (`SOLVER_OVERRIDE` + `load_solver_override`,
  default-preserving, covered by `python -m tools.gsr_v4 --demo`).
- Claim: `v7-v1-001` (FAIL).

