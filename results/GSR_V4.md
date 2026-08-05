# v4 — the bundle that beat 35.40, and the lever that was not there

Campaign act v4, built on the frozen v3 recipe of `results/GSR_CALIBGATE.md` (official test-49
GS-HOTA **35.3991**). Four candidate levers were developed on the declared **DEV-20** partition
only; two were kept, two were dropped on their own measurements.

**Headline: official test-49 GS-HOTA 35.3991 -> 39.0215 (+3.62)**, with the v3 arm re-derived inside
the same run and reproducing its on-record number at every printed digit. GS-DetA 24.02 -> 26.47,
GS-AssA 52.18 -> 57.53, **GS-LocA 93.400 -> 93.595 (up, not paid)**, IDF1 34.09 -> 40.03. Paired over
49 sequences: mean +3.41, 43 helped, 6 hurt, worst -2.96, Wilcoxon p = 8.6e-10. The shipped zip
re-scores to itself at every digit and the legitimacy audit is **0 violations over 424,654
predictions**.

**The bundle is two knobs, and neither is new code in the model:** the jersey-aggregation floor moves
0.85 -> **0.80** (more reads at 0.81 read precision instead of fewer at 0.87), and the connector runs
at **tau 0.080** — twice as loose as the on-record 0.040 — behind a **new jersey-compatibility merge
gate** that forbids merging two tracklets whose confident reads disagree. The solver config,
calibration gate, team-side resolver and roster are byte-identical to v3.

**And a lever that turned out not to exist: `crop_scale 1.25` does nothing on GSR** (§2). It is the
single largest OCR fix on our own broadcast (1.37x reads) and here, on 67,224 exactly-paired crops,
it moves confident reads by **1.006x** and moves read density the *wrong* way. The reason is
mechanical and was checkable in advance: on GSR the OCR crop is the **detector's own box**, not the
`estimate_player_box` reconstruction that the 1.25 constant was fitted to repair.

---

## 0. What was measured, and where

| | |
|---|---|
| development partition | valid **DEV-20** (`eval.gsr_identity.split_sequences`, every third valid sequence) |
| verification partition | valid **TEST-38**, one run at the frozen bundle |
| market price | official **test-49**, one run, packaged |
| base recipe | v3 = `results/gsr_calibgate_frozen.json` (gate(3, 5) + `fill(10)`, free team map, self roster, solver config sha256 `a7282d9b...`) |
| GPU spent | 51 min (the DEV-20 `crop_scale 1.25` OCR pass — the only GPU work in this act, and it bought a negative) |
| CPU spent | ~2 h (24 DEV arms, TEST-38, test-49, packaging) |

Ordering evidence for the freeze (file mtimes, local):

| artifact | written |
|---|---|
| DEV-20 arm that chose the bundle (`gsr_v4_dev20_iso.json`) | 02:01:49 |
| **`results/gsr_v4_frozen.json`** | **02:02:32** |
| valid TEST-38 (`gsr_v4_valid_t38.json`) | 02:17:30 |
| official test-49 (`testsplit/gsr_v4_testsplit.json`) | 02:36:56 |
| the aggregation rules themselves (`results/ocr_density_rule.json`) | **2026-07-28 21:37** (frozen by `OCR_DENSIFICATION.md`, not fitted here) |

---

## 1. Component table — DEV-20, one variable at a time

Every row is the full chain (re-gated positions -> submissions -> jersey attach -> GTA connector ->
GT-free solver -> official scorer) with exactly the named knobs changed. **The v3 row reproduces
`GSR_CALIBGATE.md` §2.1 to four decimals** (32.8276 / 20.6351 / 52.2272 / 92.3107 / 30.8786), so the
harness is the same one and the deltas are like-for-like.

| arm | floor | tau | jersey gate | prior refit | GS-HOTA | vs v3 | DetA | AssA | LocA | IDF1 | merges | merge prec | cov@0.85 | helped/hurt | worst | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v3 (control)** | 0.85 | 0.040 | - | - | **32.8276** | — | 20.635 | 52.227 | 92.311 | 30.879 | 392 | 0.766 | 0.2693 | — | — | — |
| **C1** `crop_scale 1.25` | 0.85 | 0.040 | - | - | 32.4472 | **-0.380** | 20.106 | 52.370 | 92.085 | 30.322 | 392 | 0.766 | 0.2354 | 8/8 | -2.20 | 0.469 |
| **C2** floor 0.80 | 0.80 | 0.040 | - | - | 33.9219 | **+1.094** | 22.370 | 51.444 | 92.159 | 33.413 | 392 | 0.766 | 0.0013 | 12/8 | -6.31 | 0.133 |
| C2 + confusion refit | 0.80 | 0.040 | - | Y | 34.4102 | +1.583 | 22.751 | 52.049 | 92.190 | 33.983 | 392 | 0.766 | 0.0013 | 13/7 | -6.31 | 0.053 |
| **C3** tau 0.030 | 0.85 | 0.030 | - | - | 32.2518 | **-0.576** | 20.282 | 51.291 | 92.202 | 29.908 | 257 | 0.835 | 0.2687 | 6/13 | -4.87 | 0.027 |
| C3 tau 0.045 | 0.85 | 0.045 | - | - | 33.2019 | +0.374 | 20.667 | 53.341 | 92.375 | 31.201 | 434 | 0.747 | 0.2666 | 8/5 | -1.09 | 0.152 |
| C3 tau 0.050 | 0.85 | 0.050 | - | - | 33.2242 | +0.397 | 20.357 | 54.228 | 92.309 | 31.293 | 477 | 0.726 | 0.2653 | 9/9 | -2.16 | 0.586 |
| C3 tau 0.060 | 0.85 | 0.060 | - | - | 32.8962 | +0.069 | 19.872 | 54.460 | 92.236 | 31.043 | 556 | 0.659 | 0.2652 | 8/11 | -3.32 | 0.936 |
| **C3** jersey gate only | 0.85 | 0.040 | **Y** | - | 32.9149 | **+0.087** | 20.700 | 52.342 | 92.299 | 31.112 | 389 | **0.786** | 0.2762 | 3/2 | -1.45 | 0.500 |
| C3 gate + tau 0.050 | 0.85 | 0.050 | Y | - | 33.5113 | +0.684 | 20.637 | 54.422 | 92.285 | 31.709 | 474 | 0.741 | 0.2768 | 12/6 | -2.16 | 0.145 |
| C3 gate + tau 0.060 | 0.85 | 0.060 | Y | - | 33.6746 | +0.847 | 20.804 | 54.511 | 92.250 | 31.950 | 548 | 0.680 | 0.2644 | 13/6 | -1.22 | 0.016 |
| C3 tau 0.080 | 0.85 | 0.080 | - | - | 33.0041 | +0.177 | 19.611 | 55.547 | 92.271 | 31.466 | 634 | 0.612 | 0.2299 | 10/10 | -3.41 | 0.701 |
| **C3 gate + tau 0.080** | 0.85 | 0.080 | **Y** | - | **34.4734** | **+1.646** | 21.268 | 55.880 | 92.285 | 33.342 | 626 | 0.632 | 0.2423 | 16/4 | **-0.38** | 1.3e-4 |

### 1.1 Combinations

| bundle | GS-HOTA | vs v3 | DetA | AssA | merges | merge prec | helped/hurt | worst | p |
|---|---|---|---|---|---|---|---|---|---|
| floor 0.80 + refit, tau 0.050, no gate | 35.4092 | +2.582 | 23.313 | 53.788 | 477 | 0.726 | 14/6 | -4.45 | 0.0094 |
| floor 0.80 + refit + gate, tau 0.045 | 34.9842 | +2.157 | 23.223 | 52.707 | 427 | 0.759 | 15/5 | -6.31 | 0.0073 |
| floor 0.80 + refit + gate, tau 0.050 | 35.5921 | +2.765 | 23.615 | 53.650 | 468 | 0.743 | 16/4 | -4.45 | 0.0027 |
| floor 0.80 + refit + gate, tau 0.060 | 35.6590 | +2.831 | 23.567 | 53.960 | 539 | 0.686 | 17/3 | -4.44 | 0.0017 |
| floor 0.80, tau 0.080, no gate | 34.9721 | +2.145 | 22.255 | 54.961 | 634 | 0.612 | 14/6 | -6.05 | 0.027 |
| **floor 0.80 + gate, tau 0.080 (FROZEN)** | **36.4488** | **+3.621** | 24.265 | 54.754 | 617 | 0.636 | **17/3** | -3.34 | **3.2e-4** |
| floor 0.80 + refit + gate, tau 0.080 | 36.4435 | +3.616 | 24.251 | 54.770 | 617 | 0.636 | 17/3 | -3.34 | 3.2e-4 |

**The two kept levers are complementary and neither is a substitute for the other.** The floor buys
**DetA** (more jersey evidence -> more predictions clear GS-HOTA's identity gate: 20.6 -> 24.3); the
loose tau behind the gate buys **AssA** (fewer fragments: 52.2 -> 54.8). Alone they are worth +1.09
and +1.65; together +3.62, i.e. **additive to within 0.9 of a point** and slightly super-additive.

### 1.2 The shape check that decided tau, and why 0.080 and not 0.250

The DEV curve does not turn inside the range swept (floor 0.80 + gate + refit):

| tau | 0.040 | 0.045 | 0.050 | 0.060 | 0.070 | **0.080** | 0.100 | 0.150 | 0.250 |
|---|---|---|---|---|---|---|---|---|---|
| GS-HOTA | 34.59 | 34.98 | 35.59 | 35.66 | 36.11 | **36.44** | 36.49 | 36.72 | 36.90 |
| merge precision | 0.784 | 0.759 | 0.743 | 0.686 | 0.663 | **0.636** | 0.574 | 0.530 | 0.495 |

Past 0.080 the curve is flat (+0.05 to 0.100, +0.45 total to 0.250) while merge precision falls off a
cliff (0.636 -> 0.495: at tau 0.250 **half the merges join two different players** and GS-HOTA still
rises, because AssA's fragmentation penalty outweighs the contamination). **0.080 is frozen as the
conservative end of the flat region**, the same rule `GSR_CALIBGATE.md` §2 used when the (1, 0) trust
rule bought +0.04 over (3, 5). Chasing the argmax at 0.250 would have bought +0.45 more on DEV at a
merge precision no one should ship.

### 1.3 The confusion-prior refit was dropped because it is inert at the operating point

Refitting the digit-confusion prior on DEV per-crop reads is worth **+0.49 GS-HOTA at tau 0.040** and
**-0.005 at tau 0.080** (36.4435 refit vs 36.4488 not). It is a DEV-fitted quantity evaluated on DEV,
so its apparent value there is optimistic anyway; at the frozen operating point it is measurably
nothing. **Dropped**, which leaves the shipped solver configuration byte-identical to v3's.

---

## 2. Component 1 — `crop_scale 1.25` on GSR: measured, and it is not a lever here

`results/EPL_CROPFIX.md` measured **1.37x reads at held precision** on our three ManU matches and
`FOOTPASS_GAME18_SCORE.md` §v2 **2.3x** on game_18. This act ran the same widening on GSR: one GPU
pass over DEV-20 at 60 crops/track, `crop_scale 1.25` (51 min), into a variant directory. Because the
crop *set* is chosen from the same tracks, the same sampled frames and the same detections, the two
passes are **exactly paired: 67,224 crops, 67,224 shared `(track_id, frame)` keys, zero unmatched.**

| quantity, DEV-20 | crop_scale 1.0 | crop_scale 1.25 | ratio |
|---|---|---|---|
| crops | 67,224 | 67,224 | 1.000 |
| legibility >= 0.5 | 12,814 | **11,433** | **0.892** |
| confident reads (`p_number` >= 0.99) | 7,220 | 7,264 | **1.006** |
| ...gained by widening / lost | — | **1,189 / 1,145** | — |
| ...read by both, agreeing | 6,075 | 6,057 / 6,075 = **0.997** | — |
| `d` at the 0.85-floor rule | **0.2218** (354/1,596) | 0.2137 (341/1,596) | **0.963** |
| read precision at 0.85 floor | 0.8678 (n=348) | 0.8690 (n=336) | +0.001 |
| `d` at the 0.80-floor rule | **0.3026** (483/1,596) | 0.2939 (469/1,596) | 0.971 |
| read precision at 0.80 floor | 0.8103 (n=464) | 0.8220 (n=455) | +0.012 |
| **downstream: GS-HOTA on DEV-20** | **32.8276** | **32.4472** | **-0.380** |

**Density moves the wrong way at unchanged precision, and the solver arm loses 0.38 GS-HOTA.**
Widening swaps roughly equal numbers of reads in and out (1,189 gained, 1,145 lost) and *reduces*
legibility by 11% — the classifier is being handed more background.

**The mechanism, and why this was predictable:** on our own broadcast the OCR crop is *reconstructed*
by `generator.team_anchor.estimate_player_box` from a foot point and two constants fitted to nothing,
and `OCR_DOMAIN_SHIFT.md` §7 measured it at **0.814x** the annotated player height. x1.25 is a repair
of that undersizing, and its oracle arm (the annotated ROI) capped the gain at 1.07x beyond it. On
GSR, `eval.gsr_jersey.extract_track_crops` crops **the football detector's own `xyxy`** — there is no
undersizing to repair, so the same multiplier only adds background. **The 1.25 constant is a
calibration for one crop *estimator*, not a property of jersey OCR**, and it does not transfer to a
pipeline that already has real boxes. Recorded as the sharpest available instance of
`EPL_CROPFIX.md` negative #8 ("x1.25 was fitted on game_18's annotations and is merely transferred").

---

## 3. Component 2 — the aggregation floor, and the FACTS/BENCHMARK split it forces

The shipped rule is the **0.85 floor** (`min_crop_conf 0.99`, `min_votes 5`): DEV-20 `d = 0.2218` at
read precision 0.868. The **0.80 floor** (`min_crop_conf 0.90`, `min_votes 3`) is denser and dirtier:
`d = 0.3026` (**1.36x**) at read precision 0.810. On the official test-49 the same swap moves
`d` **0.2621 -> 0.3243** (961 -> 1,189 of 3,666 tracks read).

*Control on the vote emitter:* re-emitting the votes at the **0.85** floor through this act's code
path reproduces the shipped `outputs/gsr/koshkina_percrop_votes/*.json` **byte-identically on all 58
valid sequences (58 identical, 0 differing)**, so the only thing the 0.80 arm changes is the floor.

For **GS-HOTA the denser point wins outright: +1.09 alone on DEV, and it is the DetA half of the
frozen bundle's +3.62.** GS-HOTA's identity gate is binary per row — a track with no jersey scores
zero on every one of its rows, exactly like a wrong one — so evidence that is right 81% of the time
is strictly better than no evidence, and the solver's confusion prior + mutual exclusion absorb part
of the remaining 19%.

**For per-player FACTS the same swap is a disaster, and this must not propagate.** Coverage at a
jersey-precision floor of 0.85 — the number `IDENTITY_SOLVER_STAGE2.md` insists identity claims be
reported at — collapses on DEV-20 from **0.2693 to 0.0013** and on valid TEST-38 from **0.3118 to
0.0282**: with reads that are 81% precise, no posterior threshold recovers an 85%-precise operating
point at usable coverage. Coverage at 0.60 rises (0.4996 -> 0.6159 on DEV).

**So the 0.80 floor is adopted for the GSR benchmark arm ONLY.** `tools/prtreid_probe.py`'s 80%
per-player-fact bar and the report/facts chain keep the 0.85-floor rule
(`results/ocr_density_rule.json` is unchanged; the benchmark arm names its own votes directory
`koshkina_percrop_votes_f080` and never overwrites the shipped one).

*Curiosity, not a claim:* on test-49 v4's coverage@0.85 is **higher** than v3's (0.4707 vs 0.4342),
the opposite of DEV and valid-38. Coverage-at-a-floor is a threshold crossing on a precision curve;
the test split runs easier (LocA 93.6 vs 92.1, identity 0.483 vs 0.442) and its curve clears 0.85
where DEV's does not. The DEV/valid evidence is the one the facts decision rests on, because it is
the pessimistic one and there are two of it.

---

## 4. Component 3 — the linking ceiling did rise on repaired positions

`GSR_ASSOCIATION.md` negative #7 recorded that the linking ceiling was never re-measured after the
calibration repair, and §5.3 recorded that tau 0.060 dropped merge precision 80.1% -> 69.7% on the
*unrepaired* positions and was therefore refused. On the CALIBGATE-repaired positions (20.7% more
pitch rows, so the connector's speed/reachability constraint has far more geometry to work with) the
picture is different: **every tau from 0.045 to 0.250 beats 0.040, and 0.030 is worse than 0.040**
(-0.58, 6 helped / 13 hurt, p = 0.027). The tighter direction is measurably wrong.

**The jersey-compatibility merge gate** (new, `generator.gta_link.connect(..., numbers=)`): two
components that both carry a confident read and disagree may never merge, whatever appearance says; a
component with no read is unconstrained and inherits the number of whatever it joins. It costs 4
lines and it is orthogonal to appearance, so it buys **merge precision back at every tau**:

| tau (floor 0.85, so the gate is the only variable) | 0.040 | 0.050 | 0.060 | 0.080 |
|---|---|---|---|---|
| GS-HOTA, no gate | 32.8276 | 33.2242 | 32.8962 | 33.0041 |
| GS-HOTA, gate | 32.9149 | 33.5113 | 33.6746 | **34.4734** |
| **gain from the gate** | +0.09 | +0.29 | +0.78 | **+1.47** |
| merge precision, no gate | 0.766 | 0.726 | 0.659 | 0.612 |
| merge precision, gate | **0.786** | **0.741** | **0.680** | **0.636** |

**The gate is nearly worthless at the tight tau it was not needed at (+0.09) and worth +1.5 at the
loose tau it enables** — which is exactly the "a gate allowing looser tau" hypothesis, confirmed. On
the official test-49 the frozen arm merges 1,253 tracklets at **77.1% merge precision** against v3's
650 at 86.8%, and fragments per GT identity fall 5.51 -> 4.97.

---

## 5. Component 4 — NOT BUILT, because the mechanism it models is not present

`OCR_DOMAIN_SHIFT.md` §3.1 measured that **44.2% of wrong confident FOOTPASS reads are exactly the
first digit of the true number** (81 -> 8 alone accounts for 445 of them), and proposed extending the
confusion prior so a 1-digit read can be a truncated 2-digit number. Measured on GSR DEV reads before
any code was written (`python -m tools.gsr_v4 --trunc-probe`):

| | 0.80-floor rule | 0.85-floor rule |
|---|---|---|
| graded reads | 464 | 348 |
| wrong | 88 | 46 |
| ...length mismatch | 18 | 8 |
| ...**truncation (read is a prefix of the true number)** | **7 (8.0% of wrong)** | **3 (6.5% of wrong)** |

74.9% of GSR's auditable GT jerseys are 2-digit (2,658 of 3,549), so the *opportunity* exists — the
reader simply does not truncate here. The dominant GSR error is digit substitution on both positions
(42->62 x14, 11->17 x9, 13->33 x8), not second-digit loss. Fitting a truncation parameter on **3
observations** would be worse than not having one, and the whole confusion prior it lives inside is
worth -0.0128 identity accuracy on the record and **-0.005 GS-HOTA at the frozen operating point**
(§1.3). **Not built.** The premise is a domain property of FOOTPASS's soft 4.8 Mbps Serie A encode,
not of jersey OCR.

---

## 6. The frozen bundle

`results/gsr_v4_frozen.json`, declared 2026-08-02T02:02:32 (before TEST-38 at 02:17 and test-49 at
02:36), hash-linked to `results/gsr_calibgate_frozen.json`:

```
positions      = calibgate: select_calibration(trust 3 players / 5.0 m) + fill_calibration_gaps(10)
jersey reads   = per-crop OCR, crop_scale 1.0, 60 crops/track, aggregation floor 0.80
                 (min_crop_conf 0.90, min_votes 3, min_legibility 0.50, emit_all False)
connector      = GTA, splitter OFF, tau 0.080, jersey-compatible merge gate ON
solver         = results/identity_solver_config_percrop.json, UNCHANGED (no refit)
team map       = resolve_team_map_free (geometry); roster = self
```

Everything except the two bold knobs is byte-identical to v3. No GPU-side artifact changed, so the
whole bundle replays on CPU from the caches v3 already built.

## 7. Valid TEST-38 — the verification run (02:17Z)

`python -m tools.gsr_v4 --valid`. Both arms derived in one invocation; nothing quoted.

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 | identity | tracklets named | merge prec |
|---|---|---|---|---|---|---|---|---|
| v3 (control) | **34.0028** | 21.1920 | 54.5640 | 92.0586 | 32.4898 | 0.3985 | 1,463 / 2,458 | 0.794 |
| **v4** | **36.7090** | 24.1180 | 55.8787 | **92.1484** | 37.7039 | 0.4421 | 1,335 / 1,941 | 0.631 |
| delta | **+2.71** | +2.93 | +1.31 | **+0.09** | +5.21 | +0.044 | — | — |

**The v3 control reproduces `GSR_CALIBGATE.md` §3 at every printed digit** (34.0028 / 21.1920 /
54.5640 / 92.0586 / 32.4898). Paired (n = 38, Wilcoxon): mean **+2.58**, median +2.12, **31 helped,
7 hurt**, worst **-6.17** (SNGS-086), best +15.37 (SNGS-094), **p = 8.7e-6**.

The pre-declared bar — *beat v3's 34.00 on valid, or stop and spend no test slot* — is met by +2.71.

## 8. The one test run — official test-49 GS-HOTA 35.3991 -> 39.0215 (02:36Z)

`python -m tools.gsr_v4 --test --out-dir outputs/gsr_test`. One invocation: control, re-gate, votes,
connector, solve, score, legitimacy audit, package, zip re-score. **CPU only, 19 minutes** — the
frozen bundle moved no GPU-side knob, so every cached test-49 artifact was consumed unchanged.

| arm | GS-HOTA | GS-DetA | GS-AssA | **GS-LocA** | IDF1 |
|---|---|---|---|---|---|
| v3 = the shipped 35.40 (control, re-derived) | **35.3991** | 24.0218 | 52.1825 | 93.4003 | 34.0911 |
| **v4** | **39.0215** | **26.4711** | **57.5334** | **93.5948** | **40.0277** |
| delta | **+3.62** | +2.45 | +5.35 | **+0.19** | +5.94 |

**The control reproduces `GSR_CALIBGATE.md` §4 at every printed digit.** Paired (n = 49, Wilcoxon):
mean **+3.41**, median +3.00, **43 helped, 6 hurt**, worst **-2.96**, best +15.27, **p = 8.6e-10**.

Best: SNGS-199 +15.27, SNGS-146 +11.32, SNGS-196 +10.93, SNGS-200 +10.34, SNGS-198 +8.95.
Worst: SNGS-141 -2.96, SNGS-150 -2.48, SNGS-147 -1.02, then -0.29 and smaller. **No disaster tail:**
the worst sequence loses 2.96 GS-HOTA where v3's own test run inherited a -15.40.

**GS-LocA rises (+0.19).** This is the column both prior repairs had to pay (the fill -0.53, the gate
+0.03 over it); v4 does not move a single coordinate — the positions are v3's — and the small gain is
a re-weighting effect: the rows that newly clear the identity gate are the well-localised ones.

Diagnostics: identity accuracy 0.4456 -> **0.4834**; tracklets 3,031 -> 2,450 (the looser connector),
named 2,135 -> 1,851 (75.6% of tracklets named, up from 70.4%); merges 650 -> 1,253 at merge
precision 0.869 -> 0.771; fragments per GT identity 5.51 -> 4.97; roster slots per sequence 10.8 ->
12.6; **coverage @ jersey precision 0.85: 0.4342 -> 0.4707** (see §3's caveat — on DEV and valid-38
this number collapses).

**The team-side map is untouched**: both arms run `resolve_team_map_free` on the same
`positions_gate`, so free-map accuracy is v3's 45/49 and the disagreeing set is identically
{SNGS-126, SNGS-131, SNGS-190, SNGS-197}. SNGS-190 remains lost.

## 9. The package and its legitimacy audit

`results/gsr_submission/gsr_testphase_gtfree_v4_24b4b67e.zip`

| check | result |
|---|---|
| entries | 49, all `tracklab/<SEQ>.json` (the verified official layout) |
| predictions | **424,654** (identical to v3's package — only attributes differ) |
| size | 32.3 MB |
| **extract-and-score of the zip itself** | GS-HOTA **39.02152** / DetA 26.4711 / AssA 57.5334 / LocA 93.5948 / IDF1 40.0277 — **identical to the arm score at every digit** (`results/gsr_submission/zip_selfscore_gtfree_v4.json`) |
| **legitimacy audit** (`tools.gsr_calibfill.verify_gtfree`) | **0 violations** over 49 sequences and 424,654 predictions; flipped on exactly {SNGS-126, SNGS-131, SNGS-190, SNGS-197} |
| manifest | `manifest_gtfree_v4.json`, recipe hash `24b4b67e...`, no `DO_NOT_UPLOAD` |
| prior packages | v1 `..._gtfree_7dd2a50a.zip`, v2 `..._calibfill_ab823742.zip`, v3 `..._calibgate_85f63db4.zip` untouched |

The prediction chain opens no `Labels-GameState.json`: the merge gate reads only our own OCR votes,
the team side comes from geometry, the roster from our own reads.

Orientation only (nothing uploaded): 39.02 would sit between rank 8 and rank 9 on the 15-entry
codabench 4365 test-phase board, one place above where 35.40 sat.

## 10. Negatives, and what this does not say

1. **`crop_scale 1.25` is not a GSR lever and the task's premise for it was wrong** (§2). 1.006x
   confident reads on 67,224 exactly-paired crops, `d` down 3.7%, GS-HOTA -0.38. The 51 GPU-minutes
   spent on it bought a negative, and the negative is the useful part: **the 1.25 constant calibrates
   `estimate_player_box`, not jersey OCR**, so it should never be transferred to a pipeline that
   crops from real detector boxes. Not measured: whether a *narrower* crop helps GSR.
2. **Component 4 was not built** (§5). 3 truncation errors in 348 DEV reads against FOOTPASS's 44%.
   Reported as a measurement of the premise, not as an implementation.
3. **The 0.80 floor is a benchmark-only adoption and it destroys facts-grade coverage** (§3):
   coverage@0.85 0.2693 -> 0.0013 on DEV, 0.3118 -> 0.0282 on valid-38. Anything that reports
   per-player facts must keep the 0.85 rule. The test-49 number moves the other way and is not
   evidence against this.
4. **Merge precision is now 0.771 on test-49 and 0.636 on DEV — far below the 80% bar
   `GTA_LINK_STAGE1.md` A4 set for any solver consuming merged tracklets.** GS-HOTA rewards the
   trade; the facts chain must not inherit tau 0.080 without its own validator. The DEV curve is
   still rising at tau 0.250 (merge precision 0.495), which is a warning about the metric as much as
   about the connector.
5. **The tau pick is a DEV selection over 10 swept values on 20 sequences.** It is defended by the
   TEST-38 (+2.71) and test-49 (+3.62) verifications, both of which came *after* the freeze, but the
   DEV number (+3.62) and the test number (+3.62) agreeing is partly luck: valid-38 gave +2.71.
   Honest interval for the bundle: **+2.5 to +3.6 GS-HOTA**.
6. **The v4 gain is bigger than the +0.5 to +2.5 the act expected.** The reason is that the two
   levers hit different halves of the metric (DetA via evidence, AssA via linking) and neither had
   been opened since the positions changed under them. Nothing here says the next act will find
   anything comparable — both knobs are now at their measured operating points.
7. **The confusion-prior refit is dropped as inert** (§1.3), which retires the "+0.49 on DEV" number
   as an artifact of the tight-tau regime rather than a property of the prior.
8. **SNGS-190 is still lost.** The team-side coin-flip is untouched by this act and remains the
   largest single-sequence risk in the recipe.
9. **Nothing was uploaded.** 39.02 is a locally-scored number on the official split with the official
   scorer, the verified layout, and a zip that re-scores to itself. The submission decision belongs
   with the orchestrator and the 1/day cadence.
10. **Not re-measured:** the linking *ceiling* (`gsr_assoc_diag --diag`) on these positions at tau
    0.080. §4 shows the realised gain, not how much headroom is left; `GSR_ASSOCIATION.md` negative
    #7 is therefore only half-answered.

## 11. Reproduce

```
python -m eval.gsr_jersey --percrop --max-crops 60 --crop-scale 1.25 --variant _w125 \
    --seqs <DEV-20>                                       # section 2 (GPU, 51 min)
python -m tools.gsr_v4 --cropscale                        # section 2 (CPU)
python -m tools.gsr_v4 --trunc-probe                      # section 5 (CPU)
python -m tools.gsr_v4 --dev --tag dev20                  # section 1 (CPU, ~25 min)
python -m tools.gsr_v4 --dev --tag dev20_combo --arms ... # section 1.1
python -m tools.gsr_v4 --dev --tag dev20_shape  --arms ... # section 1.2
python -m tools.gsr_v4 --dev --tag dev20_shape2 --arms ... # section 1.2
python -m tools.gsr_v4 --dev --tag dev20_iso    --arms ... # section 1.3
python -m tools.gsr_v4 --freeze '{"name":"f080_tau080_jg_norefit","floor":"0.80","tau":0.080,
    "gate":true,"refit":false,"variant":""}' --tag dev20_iso        # section 6
python -m tools.gsr_v4 --valid                                     # section 7 (CPU, ~10 min)
python -m tools.gsr_v4 --test --out-dir outputs/gsr_test \
    --results-dir results/gsr_benchmark/testsplit                  # sections 8-9 (CPU, 19 min)
```

Self-checks: `python -m tools.gsr_v4 --demo`, `python -m generator.gta_link`,
`pytest tests/test_gta_link.py tests/test_ocr_density.py tests/test_identity_solve.py
tests/test_jersey_id.py tests/test_gsr_score.py`.

## 12. Files

- `generator/gta_link.py` — `connect(..., numbers=)`: the jersey-compatibility merge gate (+ 5 new
  assertions in `_demo`, which `tests/test_gta_link.py` runs).
- `eval/gsr_gta.py` — `repair_sequence(..., numbers=)` and `process_sequence(..., numbers=)`
  pass-through; sub-tracklets inherit their parent's number.
- `eval/gsr_identity.py` — `dominant_numbers` (new), `build_bundle`/`load_bundles` gained `params`
  (so a non-default tau partition can be reproduced) and `jersey_gate`.
- `eval/gsr_jersey.py` — `extract_track_crops(..., crop_scale)`, `read_sequence_percrop`,
  `run_percrop(..., crop_scale, variant, only)`, CLI `--crop-scale/--variant/--seqs`. Default 1.0
  reproduces every on-record artifact.
- `tools/ocr_density.py` — `load_percrop(..., variant)`, `emit_votes(..., votes_subdir, variant)`.
- `tools/gsr_calibfill.py` — `gta_arm(..., gate_votes_dir)` and it now returns per-sequence stats.
- `tools/gsr_v4.py` — the whole act (`--trunc-probe/--cropscale/--dev/--freeze/--valid/--test/
  --demo`).
- `results/gsr_v4_frozen.json`; `results/gsr_benchmark/gsr_v4_{dev20,dev20_combo,dev20_shape,
  dev20_shape2,dev20_iso,cropscale_dev20,truncation_probe,valid_t38}.json`;
  `results/gsr_benchmark/testsplit/gsr_v4_testsplit.json`.
- `results/gsr_submission/gsr_testphase_gtfree_v4_24b4b67e.zip`, `manifest_gtfree_v4.json`,
  `zip_selfscore_gtfree_v4.json`.
- Artifacts: `outputs/gsr/koshkina_percrop_w125/` (20 DEV sequences, the negative),
  `outputs/{gsr,gsr_test}/koshkina_percrop_votes_f080/`, `eval_v4_gta_*/`,
  `identity_bundles_v4_*/`, `deleak_t49_v4campaign_*`. No on-record artifact was overwritten.
