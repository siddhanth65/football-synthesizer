# N2b — the calibration repair inside the GT-free recipe: official test 31.88 -> 33.37

Campaign act N2b. `results/GSR_ASSOCIATION.md` froze `fill_calibration_gaps(max_gap=10)` and measured
it on the **connector** arm only (valid 23.53 -> 24.35); its negative #4 named the reason the
identity-solver arm was left unmeasured — `eval.gsr_identity.build_bundle` hard-codes
`out_dir / "positions"`. `results/GSR_DELEAK.md` froze the GT-free recipe and scored **31.88** on the
official 49-sequence test split. This document folds the two together.

**Headline: official test-49 GS-HOTA 33.37 (DetA 22.26, AssA 50.05, IDF1 32.32), still zero label
reads in the prediction chain. The unrepaired GT-free arm, re-derived through this same harness in
the same run, returns 31.88 / 21.11 / 48.15 — identical to `GSR_DELEAK.md` §5 to four decimals. The
repair is worth +1.49 GS-HOTA on test and +1.29 on valid TEST-38, and no column is traded away.**

Package: `results/gsr_submission/gsr_testphase_gtfree_calibfill_ab823742.zip` (28.23 MB, 49 entries,
376,264 predictions, 0 legitimacy violations). The v1 `gsr_testphase_gtfree_7dd2a50a.zip` is
untouched.

---

## 1. The one-argument change

`eval.gsr_identity.build_bundle` and `load_bundles` gained `positions_subdir: str = "positions"`.
The default is byte-identical to the old behaviour, so nothing on record shifts. Two pass-through
kwargs were added downstream (`tools.gsr_deleak.run_arm`: `positions_subdir`, `base_arm`; and
`package_free`: `stem`, so a new package can never overwrite a prior one), also default-identical.

That is the whole enabling diff. Everything else in this document is a run.

## 2. Valid TEST-38 — the arm the roster was never fitted on

`python -m tools.gsr_calibfill --solve-valid`. Both arms are the frozen GT-free recipe
(`resolve_team_map_free` + the `self` roster + `results/identity_solver_config_percrop.json`,
sha256 `a7282d9b…`); the only variable is which positions table feeds bundle construction, the free
team map and the base arm the solver rewrites.

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 | identity acc | named | cov@0.85 |
|---|---|---|---|---|---|---|---|---|
| no fill (control) | **30.7166** | 18.9799 | 49.7167 | 92.1229 | 29.672 | 0.4033 | 1536/2541 | 0.3265 |
| + `fill(max_gap=10)` | **32.0115** | 19.9234 | 51.4392 | 91.8767 | 30.824 | 0.4012 | 1533/2534 | 0.3252 |
| Δ | **+1.29** | +0.94 | +1.72 | -0.25 | +1.15 | -0.0021 | -3 | -0.0013 |

**Control check: the no-fill arm reproduces `GSR_DELEAK.md` §2's `free`/`self` row exactly**
(30.72 / 18.98 / 49.72). The two sides are the same harness.

Paired per sequence (n = 38, Wilcoxon): mean **+1.91**, median +1.04, **helped 35, hurt 3**,
worst -1.47 (SNGS-082), best +9.04 (SNGS-034), **p = 7.78e-9**.

Best: SNGS-034 +9.04 (0.96 -> 10.00), SNGS-056 +8.71, SNGS-038 +6.82, SNGS-026 +5.59. Worst:
SNGS-082 -1.47, SNGS-035 -0.11 — the same two sequences `GSR_ASSOCIATION.md` §4.3 found hurt on the
connector arm, and both are already well calibrated, so the repair only adds marginal rows there.

**A second-order win the connector arm could not show.** SNGS-034 is the worst-calibrated sequence
in the split (pitch positions on 13.6% of frames). On the repaired positions the geometric team-side
resolver gets it *right*, where on the raw positions it flipped: **free-map accuracy on valid TEST-38
goes 36/38 -> 37/38**. A left/right flip is an annihilation (`GSR_DELEAK.md` §5), so repairing the
geometry the resolver reads is worth more than the rows it recovers. This is the only sequence in
the 38 whose side map moved.

## 3. The freeze

`results/gsr_calibfill_frozen.json`, written **2026-08-01T08:32:27Z** (14:02:27 IST), before the test
split was touched — the test run was launched after it and its first result line is 14:05:12 IST.
It names its parent
(`results/gsr_deleak_frozen.json`, declared 03:45:35Z), the unchanged solver config and its sha256,
and the fill provenance (`generator.postprocess.fill_calibration_gaps`, `max_gap = 10`,
`min_donor_rows = 8`, chosen on valid DEV-20).

Recipe hash (sha256 of the canonical record, the same function `package_free` uses):
**`ab8237427e510e790e9feca88dc6aa34b49d01cdff26d0bdfa2d55cc5bb0182d`** — `ab823742` in the package
name.

## 4. The one test run

`python -m tools.gsr_calibfill --solve-test` — a single invocation: control solve, fill, connector,
solve, score, legitimacy audit, package. Fill recovered **44,244 of 460,324 test rows (9.6%)**
against the 7.3% of valid, consistent with `GSR_ASSOCIATION.md` §6's projection that the defect is
~1.5x worse on test.

| | GT-free (on record) | GT-free + calibfill | Δ |
|---|---|---|---|
| **GS-HOTA** | **31.8777** | **33.3722** | **+1.49** |
| GS-DetA | 21.1107 | 22.2578 | +1.15 |
| GS-AssA | 48.1520 | 50.0547 | +1.90 |
| GS-LocA | 93.9336 | 93.3688 | -0.56 |
| IDF1 | 31.0059 | 32.3179 | +1.31 |
| per-row identity accuracy | 0.4515 | 0.4470 | -0.0045 |
| coverage @ jersey precision 0.85 | 0.4385 | 0.4327 | -0.0058 |
| tracklets named | 2214/3114 | 2196/3095 | -18 / -19 |

The control column was **re-derived, not copied**: the no-fill arm run through this harness returns
GS-HOTA 31.8777 / DetA 21.1107 / AssA 48.1520 / LocA 93.9336 / IDF1 31.0059, identical to
`GSR_DELEAK.md` §5 at every printed digit. The +1.49 is like-for-like.

Paired per sequence (n = 49): mean **+2.10**, median +1.81, **helped 46, hurt 2**, worst -15.34,
best +11.20, **p = 3.23e-8**.

### 4.1 Where the test gain goes — and the one clip it cost

Two of the 49 sequences had their free team-side map *change* under the repair, and they are the
whole tail of the distribution:

| sequence | side map raw -> filled | correct? | GS-HOTA raw -> filled | Δ |
|---|---|---|---|---|
| SNGS-129 | left -> right | wrong -> **right** | 4.94 -> 12.34 | **+7.40** |
| SNGS-190 | left -> right | right -> **wrong** | 20.12 -> 4.77 | **-15.34** |

**Free-map accuracy on test is 45/49 both ways** — the repair swapped which four clips it loses, it
did not improve the resolver. Wrong before: {126, 129, 131, 197}. Wrong after: {126, 131, **190**,
197}. Both clips that moved had the smallest raw margins in the split (0.75 m and 0.41 m), which is
`GSR_DELEAK.md` negative #2 restated: the margin is not a confidence, and it does not become one
after the repair.

**On the 47 sequences whose side map did not move, the mean gain is +2.36 GS-HOTA.** The two flips
net -7.94 between them, dragging the pooled paired mean down to +2.10. Read: the repair's real,
mechanism-consistent effect is +2.36/sequence; one unlucky bit flip on a 0.4 m margin took a quarter
of it back. Best gains otherwise: SNGS-143 +11.20 (15.86 -> 27.06), SNGS-144 +7.36, SNGS-150 +6.69.
The other "hurt" sequence is SNGS-197 at -0.01.

## 5. The package and its legitimacy audit

`results/gsr_submission/gsr_testphase_gtfree_calibfill_ab823742.zip`

| check | result |
|---|---|
| entries | 49, all `tracklab/<SEQ>.json` (the verified official layout) |
| predictions | 376,264 |
| size | 28,234,496 bytes (28.23 MB) |
| zip bytes vs arm JSONs | **byte-identical for all 49 sequences** |
| extract-and-score of the zip itself | GS-HOTA **33.3722** / DetA 22.2578 / AssA 50.0547 — matches the arm score exactly (`results/gsr_submission/zip_selfscore_gtfree_calibfill.json`) |
| prediction schema | `attributes, bbox_image, bbox_pitch, category_id, confidence, id, image_id, supercategory, track_id`; roles {player, goalkeeper, referee}; teams {left, right, None} |
| manifest | `manifest_gtfree_calibfill.json`, no `DO_NOT_UPLOAD`, carries the recipe hash and the fill block |
| v1 package | `gsr_testphase_gtfree_7dd2a50a.zip` + `manifest_gtfree.json` untouched (mtime 09:52, before this work) |

**Legitimacy audit (`tools.gsr_calibfill.verify_gtfree`), the same check `GSR_DELEAK.md` §6 ran.**
The cached base arm's `attributes.team` is written under the GT-agreement map, so the shipped rows
must equal `free_map[cluster]`: the base value, flipped on exactly the sequences where the free map
disagrees with the GT one, untouched everywhere else. Over **49 sequences and 376,264 predictions:
0 violations**, flipped on exactly the 4 sequences {SNGS-126, SNGS-131, SNGS-190, SNGS-197}. The GT
map cancels; the emitted side is identical to writing the submission from scratch under the free map
with no label opened.

Everything else in the chain is the GT-free one `GSR_DELEAK.md` §6 documents — positions, PRTreID
embeddings, per-crop OCR votes, the frozen solver config — plus the repair, which reads only the
positions table's own `(image_xy, pitch_xy)` pairs.

Orientation only (nothing uploaded): 33.37 would sit between rank 11 (lsmuqi, 33.12) and rank 10 on
the 15-entry codabench 4365 test-phase board, one place above where 31.88 sat.

## 6. Negatives

1. **The repair did not improve the team-side resolver, it reshuffled it.** 45/49 on test either
   way; SNGS-190 was lost (-15.34) as the price of SNGS-129 (+7.40). The valid split saw the
   friendlier version of the same coin (36/38 -> 37/38). **One binary flip on a 0.4 m margin is
   still the largest single-sequence risk in this recipe**, and the repair does not touch it.
2. **The identity columns move the wrong way, slightly.** Per-row identity accuracy 0.4515 ->
   0.4470 on test, coverage@0.85 0.4385 -> 0.4327, 18 fewer tracklets named. The repair changes the
   connector's merge (3114 -> 3095 tracklets), so the solver is answering a slightly different
   question; GS-HOTA rises because the recovered *rows* are worth more than the marginal identity
   accuracy they dilute. Anyone quoting the identity accuracy alone would read this as a regression.
3. **GS-LocA falls on both splits** (-0.25 valid, -0.56 test). Recovered rows are re-projected
   through an interpolated homography, so they are on average slightly less precise than the
   calibrator's own. The DetA gain more than pays for it, but the repair is not free in localisation.
4. **`max_gap` was not re-opened.** 10 is the DEV-20 freeze; `GSR_ASSOCIATION.md` §4.1 records 25 as
   statistically indistinguishable there. Re-picking it on this arm would be fitting on verification
   data.
5. **The root cause is still unfixed.** PnLCalib still emits gate-passing homographies that project
   every player off the pitch (`GSR_ASSOCIATION.md` negative #6). This is a post-hoc patch on cached
   artifacts; a real fix in `generator/calibrate.py` needs a GPU re-run.
6. **The valid delta does not predict the test delta.** +1.29 on valid TEST-38, +1.49 on test —
   close here, but the mechanism differs (one favourable flip on valid, one favourable and one
   unfavourable on test). The agreement is partly luck, as `GSR_DELEAK.md` negative #5 warned.
7. **Nothing was re-extracted and nothing was uploaded.** Every number above is CPU work on cached
   artifacts. The submission decision belongs with the orchestrator and the 1/day cadence.

## 7. Reproduce

```
python -m tools.gsr_calibfill --solve-valid                    # section 2 (~9 min CPU)
python -m tools.gsr_calibfill --freeze --max-gap 10            # section 3
python -m tools.gsr_calibfill --solve-test \
    --out-dir outputs/gsr_test --results-dir results/gsr_benchmark/testsplit   # section 4 (~13 min)
```

Self-checks: `python -m tools.gsr_calibfill --demo` (now also covers `verify_gtfree`),
`python -m tools.gsr_deleak --demo`, `pytest tests/test_postprocess.py` (13 passed).

## 8. Files

- `eval/gsr_identity.py` — `positions_subdir` on `build_bundle` / `load_bundles` (default unchanged).
- `tools/gsr_deleak.py` — `positions_subdir` + `base_arm` on `run_arm`/`write_arm`/`_team_maps`,
  `stem` on `package_free` (all defaults unchanged).
- `tools/gsr_calibfill.py` — `solve_arm`, `verify_gtfree`, `freeze_combined`, `run_testsplit`, and
  the `--solve-valid / --freeze / --solve-test` CLI.
- `results/gsr_calibfill_frozen.json` — the pre-declared combined recipe (hash `ab823742…`).
- `results/gsr_benchmark/gsr_calibfill_solve_valid38.json` — the valid TEST-38 arms + paired stats.
- `results/gsr_benchmark/testsplit/gsr_calibfill_testsplit.json` — the test-49 run, its control,
  the fill stats, the legitimacy audit and the manifest.
- `results/gsr_submission/gsr_testphase_gtfree_calibfill_ab823742.zip`,
  `manifest_gtfree_calibfill.json`, `zip_selfscore_gtfree_calibfill.json`.
- `outputs/gsr_test/positions_filled/`, `outputs/gsr_test/eval_calibfill_{base,koshkina,gta}/`,
  `outputs/gsr{,_test}/deleak_t*_{no,cali}fill_free_self/`,
  `outputs/gsr_test/identity_bundles_percrop_filled/` — artifacts. No original was overwritten.
