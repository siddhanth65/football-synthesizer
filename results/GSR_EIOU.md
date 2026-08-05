# S5 — clean-room ExpansionIoU association, and what actually carried it

Campaign v6, session S5. `results/GSR_ASSOCIATION.md` measured ~20 unrealised GS-AssA points sitting
in tracklet linking: an oracle linker on our own detections reaches AssA ~68 while the shipped chain
reaches 47.9. This session replaces the ByteTrack-era association on GSR with a clean-room
implementation of Deep-EIoU (arXiv:2306.13074) and measures what that buys, end to end, through the
frozen v4/v5 arm machinery.

**Headline: the pre-declared DEV gate PASSES and the held-out TEST-38 check is the first
significant association win of the campaign — GS-HOTA 37.0344 -> 39.5403 (+2.51), GS-AssA +2.54,
GS-DetA +2.19, 26 of 38 sequences helped, Wilcoxon p = 0.00092.** For comparison, every prior v5
arm stalled at p ~= 0.06.

**But the paper's headline idea is not what paid.** Decomposed on DEV-20, the *deep-features* half
of Deep-EIoU is worth **+3.49 GS-HOTA** and the *ExpansionIoU* half is worth **+0.37**; the
*iterative scale-up* is worth **-0.15** (a small negative). A from-scratch EIoU re-tracker with the
appearance term switched off scores **3.54 points BELOW** the ByteTrack control. The honest name for
this result is "deep-feature association, with expansion as a tiebreak", not "ExpansionIoU works".

The mechanism is measured, not inferred: the EIoU partition **fragments 1.8x more** than ByteTrack
(11.47 vs 6.52 fragments per GT identity) and is **purer** (row-weighted purity 0.9416 vs 0.8969;
contaminated tracklets 106 vs 173). Our downstream assigns ONE jersey per merged tracklet, so
contamination costs and fragmentation is what the GTA connector is for. §5.

Development was on DEV-20 only. TEST-38 was run **once**, after the freeze
(`results/gsr_eiou_frozen.json`, declared 2026-08-05T12:09:23Z). **The test-49 split was not
touched.** This is a component check; the v6 combined recipe belongs to S7.

---

## 1. Clean-room provenance, and the one thing the cache did not hold

The Deep-EIoU reference repository carries **no license file** -> all rights reserved. Nothing was
read from it. `tools/gsr_eiou.py` implements three ideas from the paper text:

1. **ExpansionIoU** — scale both boxes about their centres by `1 + e`, then take a standard IoU.
   `e = 0` is plain IoU, so the parameterisation is self-checking.
2. **Iterative scale-up** — associate in rounds, raising `e` by `e_step` for whatever is still
   unmatched.
3. **Deep features** — a per-track appearance state (EMA, momentum 0.9) over the cached
   per-detection embeddings, fused into the assignment cost and used as a hard gate.

There is **no Kalman filter**: a track's matching geometry is its last *observed* box.

Two deviations, both forced by the cache and both stated up front:

* **The cached positions rows carry no bounding box.** `generator.extract.POSITIONS_COLUMNS`
  persists the foot point `(image_x, image_y)` and `conf` only — verified by reading the schema and
  the parquets. EIoU is a box method, so `--build-boxes` re-runs the same football YOLO at stride 1
  and attaches each row its own detection box (greedy nearest foot point at the 6 px tolerance
  `generator.gta_link.detection_embeddings` already uses), falling back to
  `generator.team_anchor.estimate_player_box` when there is no detection within tolerance.
  **Measured fallback rate: 0.46% on DEV-20 (99.54% real detector boxes) and 0.51% on TEST-38
  (99.49%).** This is the only GPU stage: 37-47 s/sequence, 12 min for DEV-20, 25 min for TEST-38.
  Reconstructing boxes from the foot point instead would have made EIoU a deterministic function of
  foot distance and the negative unfalsifiable; that route was rejected.
* **ByteTrack's low-score second association is omitted.** Our cached rows are already ByteTrack's
  *output*. Every one of them must receive an id, because dropping a row would move GS-DetA and
  destroy the like-for-like against the control, so a confidence split cannot do its job here.

**Consequence worth stating: this is a re-association, not a re-detection.** Detections ByteTrack
deleted (`minimum_consecutive_frames = 3`) are not recoverable from here. The ceiling of this lever
is the oracle-linker ceiling of `GSR_ASSOCIATION.md` §3.1, not perfect tracking.

## 2. The output contract (why nothing on record can mix)

The downstream chain is untouched: connector `tau` frozen per embedder, solver frozen, aggregation
floor 0.80, jersey-compatible merge gate on, no confusion refit. Only the track partition changes.

Three cached artifacts are keyed by `(track_id, frame)` and are rewritten into `*_eiou` variant
directories by the exact same relabelling map:

| artifact | consumer |
|---|---|
| `positions_gate/<seq>.parquet` | submission geometry, connector fragments, solver, free team map |
| `detembed_cache_{clip,prtreid}/<seq>.npz` | connector distance + solver appearance gallery |
| `koshkina_percrop/<seq>.parquet` | jersey evidence (re-aggregated into fresh votes per point) |

`koshkina_jersey` is deliberately **not** remapped, and this is verified rather than assumed: it
only ever fills `attributes.jersey` on `role == "player"` rows
(`eval.gsr_jersey.patch_submission`, `eval.gsr_prtreid_relink.propagation_fill`), and every such row
is overwritten by `tools.gsr_deleak.write_arm`. It cannot reach the scored submission.

Arm isolation uses the mechanism `results/CLUSTER_SESSION4.md` §1 built for exactly this: the
embedder name drives both the cache subdirectory and the arm key together, so an EIoU arm can never
reuse the control's bundle cache. Every consumer imports that constant *inside* the function, so
`set_embedder()` switches the whole chain in one rebinding. One new keyword was added to
`tools.gsr_v4.solve_v4_arm` (`positions_subdir`, defaulting to the on-record value) — that is the
entire diff to existing modules.

**Control validation.** The control arm is re-derived in-session, never quoted, and it reproduces
the on-record numbers to the fourth decimal: DEV-20 CLIP **37.0711** and TEST-38 CLIP **37.0344**,
both exactly `results/CLUSTER_SESSION4.md` §4-§5; DEV-20 PRTreID **36.4488**, exactly the on-record
v4 arm. The two sides are the same harness.

## 3. Self-check

`python -m tools.gsr_eiou --demo` asserts the paper's motivating case on a synthetic 3-box scenario:
a player moving further than his own width between two frames (plain IoU **0.000**, so IoU
association breaks the track) plus a far-away distractor.

```
IoU(e=0)   -> 0.000 for every pair, association yields 3 ids for 2 objects
EIoU(e=0.7)-> 0.133 for the true pair, 0.000 for the distractor, 2 ids
```

It further asserts: expansion is about the centre and monotone in `e`; a self-pair is 1.0 at any
`e`; iterative scale-up reaches the same match from a tighter start in a later round; deep features
veto a geometrically-plausible but visually-wrong match; and a row with no cached embedding is
matched on geometry rather than refused. `tests/test_gsr_eiou.py` adds the two contract invariants
(EIoU monotone in `e`; every detection gets an id, deterministically).

## 4. DEV-20 sweep — both embedder arms

Grid pre-declared before any result: `e in {0, 0.3, 0.5, 0.7, 1.0, 1.4}`, `rounds in {1, 3}`,
`w_app in {0, 0.5}`, `app_max` per embedder. All 20 DEV sequences, paired per sequence.

### 4.1 CLIP arm (connector tau 0.450, `app_max` 0.30)

| arm | GS-HOTA | GS-DetA | GS-AssA | IDF1 | mean d | median d | helped | hurt | p |
|---|---|---|---|---|---|---|---|---|---|
| **control (ByteTrack)** | **37.0711** | 23.9202 | 57.4538 | 38.12 | — | — | — | — | — |
| e 0.0, r 1, w_app 0 | 35.2242 | 22.5953 | 54.9134 | 36.32 | -2.11 | -1.95 | 5 | 15 | 0.0441 |
| e 0.7, r 3, w_app 0 | 33.5283 | 21.4808 | 52.3337 | 34.47 | -4.01 | -3.83 | 3 | 17 | 0.00085 |
| e 0.0, r 1, w_app 0.5 | 38.7134 | 25.7161 | 58.2819 | 39.55 | +1.50 | +0.27 | 11 | 9 | 0.261 |
| **e 0.3, r 1, w_app 0.5 (FROZEN)** | **39.0820** | **25.7403** | **59.3412** | **39.85** | **+1.89** | +1.46 | **15** | 5 | 0.0532 |
| e 0.7, r 1, w_app 0.5 | 38.9320 | 25.8467 | 58.6433 | 40.12 | +1.67 | +1.49 | 13 | 7 | 0.202 |
| e 1.0, r 1, w_app 0.5 | 38.6550 | 25.2665 | 59.1403 | 39.59 | +1.37 | +1.66 | 13 | 7 | 0.165 |
| e 1.4, r 1, w_app 0.5 | 38.9391 | 25.5365 | 59.3776 | 39.67 | +1.71 | +1.67 | 15 | 5 | 0.0826 |
| e 0.3, r 3, w_app 0.5 | 38.5641 | 25.1259 | 59.1910 | 39.28 | +1.21 | +1.58 | 14 | 6 | 0.143 |
| e 0.5, r 3, w_app 0.5 | 38.5897 | 25.3543 | 58.7358 | 39.54 | +1.33 | +1.56 | 15 | 5 | 0.114 |
| e 0.7, r 3, w_app 0.5 | 38.6432 | 25.4873 | 58.5914 | 39.59 | +1.35 | +1.71 | 13 | 7 | 0.189 |
| e 1.0, r 3, w_app 0.5 | 38.7693 | 25.4450 | 59.0727 | 39.64 | +1.48 | +1.91 | 13 | 7 | 0.105 |

### 4.2 The decomposition — which third of the paper paid

Read down the `rounds = 1` column at `e = 0`:

| step | GS-HOTA | delta |
|---|---|---|
| ByteTrack control | 37.0711 | — |
| EIoU, plain IoU (e=0), **no** deep features | 35.2242 | **-1.85** |
| + deep features (e=0, w_app 0.5) | 38.7134 | **+3.49** |
| + expansion (e=0.3) | 39.0820 | **+0.37** |
| + iterative scale-up (e=0.3, rounds 3) | 38.5641 | **-0.52** |

**A from-scratch re-tracker is worse than ByteTrack until appearance is added.** Expansion then adds
a real but small amount, and the scale-up is a small negative — consistent with
`GSR_ASSOCIATION.md` §1's finding that GSR has no sampling gap (stride 1, 25 fps) and §3.3's
measured nulls on crowding and camera motion. The paper's motivating regime is large inter-frame
motion; we do not have it. What we have is an appearance-free tracker, and that is the hole EIoU
plugs.

### 4.3 PRTreID arm (connector tau 0.080)

The `app_max` gate had to be rescaled: measured EMA-to-next-detection cosine distance quantiles
over 18,051 pairs are p90 **0.0703** / p95 0.0993 for PRTreID against p90 **0.2107** / p95 0.2563
for CLIP. The initial estimate of 0.06, extrapolated from the *tracklet-mean* statistics in
`CLUSTER_SESSION4.md` §3 (a 4.8x scale ratio), was **wrong** — the tails scale by ~2.1x, not 4.8x —
and it cost 2.8 GS-HOTA. This is recorded because it is the same class of error as v5's `tau`.

| arm (e 0.3, r 1, w_app 0.5) | GS-HOTA | GS-DetA | GS-AssA | mean d | helped | hurt | p |
|---|---|---|---|---|---|---|---|
| **control (ByteTrack)** | **36.4488** | 24.2649 | 54.7536 | — | — | — | — |
| app_max 0.06 | 33.6100 | 23.1158 | 48.8754 | -3.14 | 5 | 15 | 0.0362 |
| app_max 0.10 | 36.0317 | 23.9920 | 54.1169 | -0.55 | 11 | 9 | 0.841 |
| **app_max 0.14 (best)** | **37.5541** | 24.9237 | 56.5888 | +0.70 | 12 | 8 | 0.498 |
| app_max 0.20 | 37.0011 | 24.2429 | 56.4763 | +0.37 | 10 | 10 | 0.985 |
| app_max 0.30 | 37.4406 | 25.1164 | 55.8129 | +0.86 | 13 | 7 | 0.368 |
| app_max 0.06, w_app 0 | 33.9368 | 21.9104 | 52.5669 | -2.65 | 5 | 15 | 0.0073 |

**The PRTreID arm passes the gate at exactly one point and never significantly.** Its best is
+1.11 GS-HOTA against CLIP's +2.01, which is the expected ordering: the association win is an
appearance win, so the better appearance model wins more. CLIP is the arm carried forward.

## 5. The mechanism, measured — purity bought with fragmentation

GT audit of the **tracker's own partition**, before the connector (`--purity`, ground truth read
only to score). Pooled over DEV-20, ~190k auditable rows:

| partition | row purity | tracklets | contaminated | fragments per GT identity |
|---|---|---|---|---|
| ByteTrack (control) | 0.8969 | 1,523 | 173 | 6.52 |
| **EIoU e 0.3, w_app 0.5** | **0.9416** | 3,262 | **106** | **11.47** |
| EIoU e 0.3, **w_app 0** | 0.8519 | 1,294 | 202 | 6.26 |
| EIoU e 0.0, w_app 0.5 | 0.9434 | 3,512 | 108 | 12.14 |

**The EIoU tracker is not a better tracker by the usual reading — it fragments 1.8x more.** It is
better by the reading our pipeline cares about: **39% fewer contaminated tracklets**. The identity
solver assigns one jersey to every row of a merged tracklet, so one contaminated tracklet mislabels
hundreds of rows, while an extra clean fragment is exactly what the GTA connector exists to merge
(merges rise 826 -> 2,328 on DEV). Switch appearance off and purity drops *below* ByteTrack
(0.8519) — the same ordering as the scores. This is why GS-DetA moves at all: GS-DetA is gated on
team+jersey+role, so a purer partition converts into detections that pass the gate.

## 6. Gate verdict

Pre-declared (plan `floating-leaping-peacock.md`, S5): **AssA >= +1.5 AND GS-HOTA >= +1.0 AND
>= 12/20 sequences helped AND DetA loss <= 0.3**, against the same-session control.

At the frozen point (CLIP, e 0.3, rounds 1, w_app 0.5, app_max 0.30):

| criterion | required | measured | |
|---|---|---|---|
| GS-AssA | >= +1.50 | 57.4538 -> 59.3412 = **+1.887** | PASS |
| GS-HOTA | >= +1.00 | 37.0711 -> 39.0820 = **+2.011** | PASS |
| sequences helped | >= 12/20 | **15/20** | PASS |
| GS-DetA | loss <= 0.30 | 23.9202 -> 25.7403 = **+1.820** (a gain) | PASS |

**GATE PASSES.** Honest caveats: the paired Wilcoxon on DEV is **p = 0.053**, not significant at
0.05; and the AssA criterion is the noisy one — across the ten appearance-enabled DEV points the
GS-HOTA criterion is met **10/10** (+1.21 to +2.01) while the AssA criterion is met **5/10**,
bouncing +1.14 to +1.89 between neighbouring `e` values. The robust DEV statement is the HOTA one.

**Freeze rule applied.** The `rounds = 1` family is flat over `e in [0, 1.4]` (38.66-39.08, spread
0.43). Taking the conservative (smallest-expansion) end *of the gate-passing points* gives
`e = 0.3`, which is also the DEV maximum — both rules agree, so no choice had to be made between
them. Frozen: `results/gsr_eiou_frozen.json`, declared **2026-08-05T12:09:23Z**, module sha256
`a7a8e3d0...`, before the TEST-38 run.

## 7. Valid TEST-38 — the verification, once

Control re-derived in the same harness (**37.0344**, matching the on-record v5 CLIP TEST-38 number
exactly), one arm at the frozen setting, 38 held-out sequences.

| | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 | tracklets |
|---|---|---|---|---|---|---|
| control (CLIP, tau 0.450) | 37.0344 | 23.9789 | 57.2018 | 92.1840 | 38.5485 | 3,399 |
| **EIoU (frozen)** | **39.5403** | **26.1697** | **59.7437** | 92.2192 | **41.2789** | 6,818 |
| delta | **+2.5059** | **+2.1908** | **+2.5419** | +0.035 | **+2.7304** | +3,419 |

Paired across the 38 sequences: mean **+2.535**, median **+2.406**, **helped 26 / hurt 12**, worst
-4.30, best +14.66, **Wilcoxon p = 0.00092**.

**The held-out gain is larger than the DEV gain (+2.51 vs +2.01) and significant at p < 0.001.**
That ordering is the opposite of the v5 and v5.1 pattern, where a DEV win shrank to +0.33 / -0.05 on
TEST-38 at p ~= 0.06-0.44. Twelve sequences still get worse and one by 4.3 points, so this is a
distributional win, not a uniform one.

No test-49 run. No submission. No commits.

## 8. Negatives and things deliberately not done

1. **ExpansionIoU itself is worth +0.37 GS-HOTA, not +2.0.** The session's win is the deep-feature
   association term. Reporting this as "Deep-EIoU works on GSR" would be true only in the sense
   that the bundle works; the named mechanism of the paper's title is the small half.
2. **Iterative scale-up is a measured small negative** (-0.52 at e 0.3, -0.29 at e 0.7). Frozen at
   `rounds = 1`, i.e. the scale-up is switched OFF in the shipped setting. The knob is retained
   because the DEV grid is 20 sequences and the effect is inside the noise band.
3. **The `app_max` mis-scaling cost the PRTreID arm 2.8 GS-HOTA** before it was measured directly
   (§4.3). The tracklet-mean cosine statistics of `CLUSTER_SESSION4.md` §3 do **not** transfer to a
   per-detection gate; the tail quantiles must be measured. The estimate-then-verify order was
   wrong, and the correct instrument (EMA-to-next-detection distance quantiles) is now in the log.
4. **The connector `tau` was not re-swept on the EIoU partition.** The partition changed
   substantially (2x the tracklets), so the frozen `tau = 0.450` is very likely off its new optimum
   and the numbers here are a **lower bound** on this lever. Re-sweeping it is an S7 decision, not
   an S5 one — the brief froze the downstream so that the association is the single variable.
5. **The two embedder arms were not run at a common `app_max` quantile.** CLIP 0.30 sits at ~p97.5
   of its own EMA-distance distribution; the PRTreID ladder brackets the equivalent point but the
   matched-quantile experiment (CLIP at 0.45 / 0.20) was started and lost to a host restart, and was
   not re-run because the PRTreID ladder already documents the sensitivity shape.
6. **This does not recover deleted detections.** ByteTrack's `minimum_consecutive_frames = 3` ran
   before our cache existed; re-association cannot undo it. A true re-track from raw detections
   needs the S7 re-extract.
7. **The box cache is a new 58-sequence artifact** (`outputs/gsr/detbox_cache`, 616k rows). It is
   reusable by S7 and was built once at stride 1; it is the only GPU cost of this session
   (~37 min total).
8. **`max_lost`, `ema` and `e_step` were not swept.** They were fixed at 30 frames / 0.9 / 0.35 from
   the paper's regime and left alone; a 20-sequence DEV split cannot resolve three more knobs.

## 9. Reproduce

```
python -m tools.gsr_eiou --demo                                  # the self-check
python -m tools.gsr_eiou --build-boxes --split dev               # GPU, 12 min
python -m tools.gsr_eiou --run --split dev --embedder clip \
    --grid "0.0:1:0.0,0.3:3:0.5,0.5:3:0.5,0.7:3:0.5,1.0:3:0.5,0.7:1:0.5,0.7:3:0.0"
python -m tools.gsr_eiou --run --split dev --embedder clip --tag _iso \
    --grid "0.0:1:0.5,0.3:1:0.5,1.0:1:0.5,1.4:1:0.5"             # the decomposition
python -m tools.gsr_eiou --run --split dev --embedder prtreid --tag _b \
    --grid "0.3:1:0.5:0.14,0.3:1:0.5:0.20,0.3:1:0.5:0.30"
python -m tools.gsr_eiou --purity --split dev --embedder clip \
    --grid "0.3:1:0.5:0.30,0.3:1:0.0:0.30,0.0:1:0.5:0.30"        # section 5
python -m tools.gsr_eiou --build-boxes --split t38               # GPU, 25 min
python -m tools.gsr_eiou --run --split t38 --embedder clip --grid "0.3:1:0.5:0.30"
pytest tests/test_gsr_eiou.py
```

## 10. Files

- `tools/gsr_eiou.py` — the association (`expand_boxes`, `eiou_matrix`, `associate`), the box cache,
  the artifact rewriter, the arms, `--demo`, `--purity`.
- `tests/test_gsr_eiou.py` — the self-check plus the two contract invariants.
- `tools/gsr_v4.py` — one added keyword, `solve_v4_arm(positions_subdir=...)`, default-preserving.
- `results/gsr_eiou_frozen.json` — the pre-TEST-38 freeze record.
- `results/gsr_benchmark/gsr_eiou_dev_clip{,_iso}.json`, `gsr_eiou_dev_prtreid_{a,b}.json`,
  `gsr_eiou_purity_dev_clip.json`, `gsr_eiou_t38_clip.json`.
- `outputs/gsr/detbox_cache/` (58 sequences), `outputs/gsr/{positions_gate,koshkina_percrop,
  detembed_cache_clip}_eiou/`, logs `outputs/gsr/eiou_*.log`.
