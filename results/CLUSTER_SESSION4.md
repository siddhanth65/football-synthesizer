# Cluster session 4 — does the CLIP encoder convert to GS-HOTA? (2026-08-04/05)

Session 3 (`results/CLUSTER_SESSION3.md`) produced an encoder that beats the shipped PRTreID
checkpoint on **crop-retrieval mAP** (58.75 vs 56.93). That is not the benchmark. This session
wires it into the real pipeline and asks the only question that matters: does it move **GS-HOTA**?

**Headline: yes, but not decisively.** On the held-out valid **TEST-38** partition, swapping only
the appearance embedder takes GS-HOTA **36.7090 -> 37.0344 (+0.33)**, with the gain landing exactly
where the mechanism predicts — **GS-AssA +1.32** — at a small DetA cost. Paired across the 38
sequences: mean **+0.57**, median +0.39, **23 helped / 15 hurt**, **Wilcoxon p = 0.063**. That is
not significant at 0.05.

**I did not spend the one test-49 run.** The reason is not the p-value alone — it is that the
recipe is knowingly *half-tuned* (§5), and the campaign's single test budget should not be spent on
a recipe I can already name an untested improvement to. That is a deviation from the brief's step 4
and is flagged for the orchestrator in §7.

---

## 1. The swap is controlled — and two silent-mixing bugs were closed first

The existing per-detection cache (`outputs/gsr/detembed_cache_prtreid/*.npz`) stores
`version / track_ids / frames / embeddings`, with embeddings already **(N, 256) float32** — the
same width as our CLIP projection head, so **no consumer needed a dimension change**.

Crops are **not** stored: `generator.gta_link.detection_embeddings` re-detects each strided frame,
matches every positions row to the nearest re-detected box by foot distance, and crops from that.
So the cheapest correct route was not to ship boxes anywhere — it was to reuse that function
verbatim and swap only its `embedder` argument, whose contract is already
`embed(crops) -> (N, D) L2-normalised`. `tools/clip_embedder.py` implements that contract against
`tools.prtreid_probe.PrtreidEmbedder`'s exact signature.

**This makes the arms row-identical rather than merely comparable.** Verified on SNGS-021: the CLIP
and PRTreID caches agree on `track_ids` and `frames` element-for-element (5,515 rows each), so the
only thing that differs between the two runs is the 256 floats per row.

Two path bugs would have silently mixed embedding spaces, and both are the class of error that
produces a plausible wrong number:

1. **`eval/gsr_identity.py:155` hardcoded `"detembed_cache_prtreid"`** and ignored the module
   constant. The connector would have used CLIP while the solver's gallery used PRTreID. Fixed to
   resolve through the shared constant, so both consumers read one space by construction.
2. **`tools/gsr_v4.config_key` did not include the embedder**, so a CLIP arm at a tau an on-record
   PRTreID arm already used (0.080) would have silently reused that arm's bundle cache and
   submission directory. Fixed by appending the embedder to the key, with the default contributing
   no suffix so every on-record directory name is byte-identical.

Selection is one environment variable, `GSR_EMBEDDER=clip`, which drives the cache subdirectory and
the arm key together — two embedding spaces can no longer land in one cache.

## 2. Extraction cost and transfer

Run on the **laptop** RTX 3050, not the server. The dominant cost is YOLO re-detection plus JPEG
decode, which is identical either way, and running locally avoids shipping boxes and embeddings
across the link entirely. Deviation from the brief's step 1, taken because it is strictly less
machinery for the same result.

| | |
|---|---|
| transfer server -> laptop | `epoch8.pt`, **923,890,202 B (881 MB)**, 45 s |
| transfer laptop -> server | **none** |
| DEV-20 cache | 20 sequences, **32 min** (~96 s/seq) |
| TEST-38 cache | 38 sequences, **58 min** |
| full valid cache | 58 sequences, **283 MB** on disk |

## 3. The similarity scale moved ~4.8x — the whole reason a retune was mandatory

Measured over the DEV-20 caches (cosine distance):

| embedding | within-track p50 | within-track p90 | cross-track p10 | cross-track p50 |
|---|---|---|---|---|
| PRTreID | 0.0169 | 0.0395 | 0.0931 | 0.1900 |
| CLIP | 0.1325 | 0.2311 | 0.4287 | 0.6803 |

The *separation ratio* is comparable (within-p90 to cross-p10: 2.36x for PRTreID, 1.85x for CLIP),
but the **absolute** scale is ~4.8x wider, and the connector's `tau` is an absolute ceiling. v4's
`tau = 0.080` sits 76% of the way from within-p90 to cross-p10 on PRTreID's scale; the same relative
point on CLIP's scale is **0.38**. That prediction was made before the sweep and landed inside the
optimum.

## 4. DEV-20 sweep (tuning split only)

Floor 0.80 + jersey-compatibility merge gate, i.e. the v4 bundle with only the embedder and tau
changed. Control is the on-record v4 DEV-20 arm `f080_tau080_jg_norefit`.

| arm | tau | GS-HOTA | DetA | AssA | LocA | IDF1 | merges | merge prec |
|---|---|---|---|---|---|---|---|---|
| **v4 control (PRTreID)** | 0.080 | **36.4488** | 24.2649 | 54.7536 | 92.2615 | — | 867 | 0.636 |
| CLIP | 0.080 | 32.7085 | 23.7820 | 44.9957 | 92.0972 | 31.19 | 77 | 0.880 |
| CLIP | 0.250 | 36.0583 | 23.8473 | 54.5254 | 92.1673 | 36.34 | 593 | 0.670 |
| CLIP | 0.320 | 36.8913 | 23.9189 | 56.9028 | 92.1615 | 37.59 | 694 | 0.640 |
| CLIP | 0.380 | 36.9169 | 23.8910 | 57.0465 | 92.2472 | 37.78 | 768 | 0.585 |
| **CLIP (chosen)** | **0.450** | **37.0711** | 23.9202 | 57.4538 | 92.2701 | 38.12 | 826 | 0.548 |
| CLIP | 0.520 | 37.0461 | 23.8433 | 57.5611 | 92.2652 | 38.24 | 876 | 0.511 |

Two things worth reading off this table:

- **Not retuning would have been a disaster, not a wash.** At the inherited `tau = 0.080` the CLIP
  arm scores **32.71**, nearly 4 points *below* the control, because the ceiling is so tight only 77
  merges survive. An embedding swap without a scale retune would have looked like a decisive failure.
- **The optimum is a broad plateau, not a knife-edge**: 0.320-0.520 all land within 0.18 GS-HOTA.
  That is what makes 0.450 safe to freeze on a 20-sequence tuning split.

Recipe frozen at `results/gsr_v5_frozen.json` (declared 22:52, checkpoint sha256
`9d2d205885ab...`) **before** the TEST-38 run.

## 5. Valid TEST-38 — the verification, once

Control re-derived in the same harness rather than quoted: it returns **36.7090**, matching the
on-record 36.71 exactly.

| | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 | merges | merge prec |
|---|---|---|---|---|---|---|---|
| v4 control (PRTreID, tau 0.080) | 36.7090 | **24.1180** | 55.8787 | 92.1484 | 37.70 | 2,009 | 0.631 |
| **v5 CLIP (tau 0.450)** | **37.0344** | 23.9789 | **57.2018** | 92.1840 | **38.55** | 2,920 | 0.542 |
| delta | **+0.3254** | -0.1391 | **+1.3231** | +0.036 | +0.85 | +911 | -0.089 |

Paired across the 38 sequences: mean **+0.5719**, median **+0.3941**, **helped 23 / hurt 15**, best
+4.75, worst **-4.38**, **Wilcoxon p = 0.0629**.

**The mechanism checks out even though the significance does not.** The entire gain is association
(+1.32 AssA) and the small DetA loss (-0.14) is the expected price of 911 extra merges at lower
precision — the same trade v4 itself took. A better appearance embedding should show up in AssA and
nowhere else, and it does. But 15 of 38 sequences got *worse*, one by 4.4 points, and p = 0.063
means this cannot be called a real improvement at the conventional bar.

## 6. Why crop mAP converted so weakly

Session 3's +1.82 crop-retrieval mAP became +0.33 GS-HOTA. The two metrics measure different things
and the pipeline damps the difference:

- The connector consumes **tracklet-mean** embeddings, not per-crop ones. Averaging 15-50 detections
  suppresses exactly the per-crop noise that retrieval mAP rewards reducing.
- GS-HOTA is gated by detection and calibration, which this swap does not touch. GS-DetA is
  ~24 in both arms; only the association half of the metric is even reachable from here.
- The merge-precision/GS-HOTA trade means the metric rewards *more* merging almost regardless of
  quality (v4 recorded the same effect), which compresses the advantage of a better embedding.

## 7. What I did NOT do, and why

**The solver's appearance parameters were not retuned, and they are mis-scaled by construction.**
`generator.identity_solve` scores its OCR-anchored gallery with `app_gain = 10.0` and
`sim_none = 0.92`, both in **cosine similarity units** fitted to PRTreID's distribution. On CLIP's
wider scale the equivalents are roughly `sim_none ~ 0.67` and `app_gain ~ 3` (same relative position
between within- and cross-track similarity; same gain x spread product). The TEST-38 number above
therefore runs a correctly-tuned connector against a **mis-tuned solver**, and is a **lower bound**
on this embedder, not its optimum. This is recorded in the frozen file's `not_retuned` block.

**Consequently I did not spend the one test-49 run**, despite the brief's step 4 gate (37.03 >
36.71) being numerically met. Spending the campaign's scarcest, once-only resource on a recipe whose
own frozen record names an untested improvement would be the wrong trade — especially at p = 0.063.
The orchestrator may of course overrule this; the recipe is frozen and the run is one command.

Recommended order before any test-49 spend:

1. Retune `sim_none` / `app_gain` on **DEV-20 only** (needs a small hook: `solve_v4_arm` currently
   hard-loads `SOLVER_CONFIG` and does not accept an override).
2. Re-run TEST-38 once at the fully-tuned bundle. If the paired p drops below 0.05 and the gain
   grows, spend test-49; if it stays at ~+0.3 and p ~ 0.06, the honest conclusion is that a better
   appearance embedding is worth ~a third of a GS-HOTA point on this pipeline and the lever is
   nearly exhausted.

## 8. Repo changes (all additive or default-preserving)

- `tools/clip_embedder.py` (new) — the embedder; `python -m tools.clip_embedder` runs a contract
  self-check (shape, L2 norm, identical crops embed identically).
- `eval/gsr_gta.py` — `EMBEDDER` env switch + `CACHE_SUBDIR` derived from it; `build_cache` picks the
  embedder class. Default `prtreid` keeps every on-record path byte-identical.
- `eval/gsr_identity.py` — the hardcoded cache literal now resolves through `CACHE_SUBDIR`.
- `tools/gsr_v4.py` — `config_key` includes the embedder (no suffix for the default).
- `results/gsr_v5_frozen.json` (new), `results/gsr_benchmark/gsr_v5_{clip,prtreid}_t38.json` (new),
  `results/gsr_benchmark/gsr_v4_clip_dev20{,b}.json` (new).

Targeted tests green: `pytest tests/ -k "gta or identity or gsr"` -> **22 passed, 1 skipped**.

Artifacts: `outputs/gsr/detembed_cache_clip/` (58 seqs, 283 MB),
`outputs/gsr/clip_ckpt/epoch8.pt` (881 MB), logs `outputs/gsr/clip_cache_{dev20,t38}.log`,
`outputs/gsr/clip_dev_sweep{,2}.log`, `outputs/gsr/{clip,prtreid}_t38.log`.

No test-49 run. No submission. No commits.

---

**Postscript (2026-08-05): §7's recommended retune was run and it is a negative.** See
`results/GSR_V5.md`. `app_gain 10 -> 3` is worth **+0.6118 on DEV-20** (37.0711 -> 37.6829) and
**-0.0527 on held-out TEST-38** (37.0344 -> 36.9817, 7 helped / 9 hurt, p = 0.438); the pre-declared
gate of 37.2 is not met and no test-49 run was spent. The `sim_none ~0.67` estimate in §7 is
**refuted** — 0.67 costs 0.28 GS-HOTA against leaving it at the inherited 0.92, which is itself the
DEV optimum. So **37.0344 was not a lower bound on this embedder; it was the number.**
