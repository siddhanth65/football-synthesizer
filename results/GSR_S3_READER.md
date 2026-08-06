# S3 — the retrained jersey reader, dropped in and measured

Campaign v6, session S3. `results/CLUSTER_SESSION_S2.md` trained a PARSeq fine-tune (arm 4t) that
beats the shipped Koshkina reader **0.8344 vs 0.7022 per-crop precision at the incumbent's own emit
rate**, on the server, through a harness that reproduces the incumbent's DEV-20 numbers exactly.
This session swaps those weights into the laptop chain and measures what the whole pipeline does
with them.

**Headline: both pre-declared gates PASS, and this is the largest single-lever DEV gain of the
campaign.** On the S5-frozen EIoU partition, a weights-only swap moves **GS-HOTA 39.0820 ->
42.7449 (+3.66)**, 18 of 20 sequences helped, Wilcoxon p = 0.00021, with **GS-DetA carrying it
(+4.76 against GS-AssA +0.57)** — exactly the decomposition the gate asked for. Read density on
DEV-20 tracklets goes **0.3026 -> 0.4148 at read precision 0.8228**, the first time the
`EVIDENCE_DENSITY_LAW.md` target of `d = 0.347` has been cleared.

**And a negative that changes the operating point:** the denser aggregation rule the v6 sweep
freezes (`d = 0.4148`) is **worse end-to-end** than reusing the incumbent's own rule on the same
evidence (`d = 0.3139` at precision 0.9256) — 41.8014 vs 42.7449 GS-HOTA. The tracklet gate and the
scored objective disagree about where to sit on the density/precision frontier, and the scored
objective wants precision. §6.

No TEST-38 scoring was spent. The test-49 split was not touched. S7 owns the combined freeze.

---

## 1. What changed in the repo (weights-path-only, incumbent-default)

`KoshkinaRecognizer` already took `parseq_ckpt` as a constructor argument, so the swap is not a
model change at all — the only hard-coded reference was the sidecar's resolution glob.

| file | change |
|---|---|
| `eval/gsr_jersey.py` | `resolve_parseq_ckpt(spec=None)` + `KOSHKINA_PARSEQ_GLOB` replace the inline `next(KOSHKINA_REPO.glob(...))`; `_build_recognizer(device, parseq_ckpt)`; threaded through `read_sequence_percrop` -> `run_percrop` -> `--parseq-ckpt` |
| `generator/jersey_id.py` | `OCR_PERCROP_VERSION` `ocr-percrop-1.1` -> **`ocr-percrop-1.2`** (the stamp now depends on the reader weights as well as the crop geometry) |
| `tools/ocr_density.py` | `--variant`; new `rule_path(variant)` so a variant sweep writes `results/ocr_density_rule_v6.json`, never the on-record freeze |
| `tools/gsr_eiou.py` | `run_point(percrop_variant=...)`, `sweep(percrop_variants=...)`, `--percrop-variants`; `clear_arm_caches` is variant-aware |
| `tools/gsr_v4.py` | `rule_for(floor, variant="")` prefers a variant's own frozen rule when one exists, else the on-record file |
| `tests/test_gsr_eiou.py` | one added test pinning arm isolation (a v6 arm may not reuse the control's bundle/vote caches) |

**Nothing on record moves.** Every reader defaults to the incumbent: the seven other
`_build_recognizer()` call sites (`tools/ocr_match.py`, `tools/jersey_head_trial.py`,
`tools/ocr_box_trial.py`, `read_sequence_jerseys`) pass no arguments, and `rule_for` falls back to
`results/ocr_density_rule.json` for every variant that has no sweep of its own.

**Default-preservation verified on the real chain, not just in a unit test.** The S5 frozen point
was re-derived through the refactored code before the new reader was introduced:

| | GS-HOTA | DetA | AssA | IDF1 | helped/hurt | p |
|---|---|---|---|---|---|---|
| control (ByteTrack, CLIP tau 0.450) | 37.0711 | 23.9202 | 57.4538 | 38.1219 | — | — |
| EIoU frozen, shipped reader | 39.0820 | 25.7403 | 59.3412 | 39.8526 | 15/5 | 0.0532 |

Every figure matches `GSR_EIOU.md` §4.1/§6 to four decimals (`gsr_eiou_dev_clip_s3repro.json`).

### 1.1 A landmine caught before it cost a GPU pass

`strhub.models.utils.load_from_checkpoint` selects the model class by substring-testing the
checkpoint **path**:

```python
def _get_model_class(key):
    ...
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
```

The trainer writes `.../checkpoints/last.ckpt`. Copied into `~/jersey-number-pipeline/models/` under
that name, the resolved path contains no `parseq` and the sidecar raises `InvalidModelError` on the
first crop of a 50-minute pass. `resolve_parseq_ckpt` now refuses such a path up front with an
actionable message. This is the same class of defect as the two upstream Koshkina bugs
`CLUSTER_SESSION_S2.md` §5 logged.

## 2. The GPU pass

`python -m eval.gsr_jersey --percrop --variant _v6 --max-crops 60 --parseq-ckpt <arm4t>` over
DEV-20: **55 minutes**, 20/20 sequences, `outputs/gsr/koshkina_percrop_v6/`. `--max-crops 60` is
load-bearing — the on-record cache was cut at 60 (verified: max 60 crops/track) while the CLI
default is 20, and a 20-crop pass would not be comparable to the incumbents.

**The comparison is paired at the crop, not sampled.** Because the crop extraction, the ResNet34
legibility gate and the KeypointRCNN torso RoI all run *upstream* of PARSeq and are unchanged, the
two evidence caches align 1:1:

- **67,224 paired crops** over DEV-20, identical `(track_id, frame)` keys;
- `legibility` identical to 5 decimals on every row, `torso` flag identical on every row;
- 12,814 crops pass legibility, 12,279 of those produce a torso RoI.

**Only the PARSeq weights differ.** That is the strongest form of the "weights-only" claim available
and it is measured, not asserted.

## 3. Crop-level: where the gain comes from

| per-crop confidence floor | incumbent emits | v6 emits | ratio | both | agree |
|---|---:|---:|---:|---:|---:|
| >= 0.50 | 11,663 | 11,803 | 1.012x | 11,290 | 8,528 (0.755) |
| >= 0.90 | 9,088 | 10,355 | **1.139x** | 8,446 | 7,591 (0.899) |
| >= 0.99 | 7,220 | 9,305 | **1.289x** | 6,654 | 6,383 (0.959) |

The v6 reader is not merely more accurate, it is **more decisive**: at the 0.99 floor it commits to
a number on 1.29x as many crops. The pre-registered risk for this session was that arm 4t's win,
measured at *matched emit*, would not convert into density because the legibility model is
unchanged. It converts — through the confidence floor, which is the knob the frozen rules actually
use.

## 4. Tracklet gate (pre-declared: d >= 0.35 at precision >= 0.80 AND d >= 0.26 at >= 0.85)

DEV-20, 1,596 original player/GK tracks in both arms (identical denominator), GSR jersey GT grading,
`tools/ocr_density.py --sweep --variant _v6`.

| floor | arm | d | tracks read | read precision | reads graded | rule chosen |
|---|---|---:|---:|---:|---:|---|
| 0.80 | incumbent | 0.3026 | 483/1596 | 0.8103 | 464 | conf 0.90, votes 3, leg 0.50 |
| 0.80 | **v6** | **0.4148** | 662/1596 | **0.8228** | 615 | conf 0.50, votes 1, leg 0.50 |
| 0.85 | incumbent | 0.2218 | 354/1596 | 0.8678 | 348 | conf 0.99, votes 5, leg 0.50 |
| 0.85 | **v6** | **0.3872** | 618/1596 | 0.8620 | 587 | conf 0.90, votes 1, leg 0.50 |
| 0.87 | incumbent | 0.1905 | 304/1596 | 0.8800 | 300 | conf 0.99, votes 5, leg 0.90 |
| 0.87 | v6 | 0.3628 | 579/1596 | 0.8745 | 550 | conf 0.50, votes 2, leg 0.50 |

| criterion | required | measured | |
|---|---|---|---|
| d at the 0.80 floor | >= 0.35 | **0.4148** | PASS |
| read precision there | >= 0.80 | **0.8228** | PASS |
| d at the 0.85 floor | >= 0.26 | **0.3872** | PASS |
| read precision there | >= 0.85 | **0.8620** | PASS |

**GATE PASSES on both floors**, by +0.065 and +0.127 of density respectively. `d = 0.4148` is the
first measurement in this project to clear `EVIDENCE_DENSITY_LAW.md`'s `d = 0.347` bar.

The mechanism is visible in the chosen rules: the v6 reader lets the sweep collapse `min_votes` from
3-5 to **1** and `min_crop_conf` from 0.99 to 0.50-0.90 while *holding* precision. A single confident
v6 crop is now worth more than five incumbent ones.

### 4.1 The same evidence under the incumbent's own rule (the single-variable arm)

Holding the aggregation rule fixed and changing only the weights:

| rule | incumbent evidence | v6 evidence |
|---|---|---|
| 0.80-floor rule (conf 0.90, votes 3) | d 0.3026 at prec 0.8103 | d **0.3139** at prec **0.9256** |
| 0.85-floor rule (conf 0.99, votes 5) | d 0.2218 at prec 0.8678 | d **0.2807** at prec **0.9406** |

**+0.115 read precision for free, at slightly higher density.** This is the arm carried into §5,
because it changes exactly one thing.

## 5. DEV gate (pre-declared: >= +1.0 GS-HOTA, >= 12/20 helped, DetA carrying it)

Base = the **S5-frozen EIoU partition** (`e 0.3, rounds 1, w_app 0.5, app_max 0.30`, CLIP,
tau 0.450), the campaign's best. Control and arms share the identical association, positions and
embeddings — `run_point` rewrites the same relabelling map and only the per-crop OCR parquet
differs. 20 DEV sequences, 190,899 auditable rows per arm.

| arm | GS-HOTA | GS-DetA | GS-AssA | IDF1 | tracklets | named | identity acc | jersey acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack control (for scale) | 37.0711 | 23.9202 | 57.4538 | 38.1219 | 818 | — | — | — |
| **EIoU + incumbent reader (control)** | **39.0820** | 25.7403 | 59.3412 | 39.8526 | 1,138 | 715 | 0.4756 | 0.5302 |
| **EIoU + v6, incumbent rule (arm A)** | **42.7449** | **30.4994** | **59.9098** | **45.5482** | 1,140 | 729 | **0.5463** | **0.6050** |
| EIoU + v6, v6-frozen rule (arm B) | 41.8014 | 29.7939 | 58.6506 | 44.7755 | 1,176 | 875 | 0.5316 | 0.5931 |

Paired per sequence against the EIoU + incumbent control:

| arm | dGS-HOTA | dDetA | dAssA | helped | hurt | worst | best | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A (weights only)** | **+3.6629** | **+4.7592** | +0.5686 | **18/20** | 2 | -1.74 | +14.02 | **0.00021** |
| B (weights + denser rule) | +2.7194 | +4.0536 | **-0.6905** | 14/20 | 6 | -4.52 | +8.01 | 0.015 |

| criterion | required | measured (arm A) | |
|---|---|---|---|
| GS-HOTA gain | >= +1.00 | **+3.6629** | PASS |
| sequences helped | >= 12/20 | **18/20** | PASS |
| DetA carrying the gain | qualitative | **DetA +4.76 vs AssA +0.57** | PASS |

**GATE PASSES.** The DetA criterion is met emphatically: 89% of the movement is detection accuracy,
which is what a jersey-reading lever must move under a metric that deletes any detection whose
number is wrong. Two sequences regress (SNGS-090 -1.74, SNGS-033 -0.88); the gain is concentrated but
not carried by outliers — the median sequence gains +1.99 and 18 of 20 improve.

## 6. The negative: denser is worse (arm B)

Arm B is the operating point the tracklet gate *selected* — `d = 0.4148`, +32% more read tracklets
than arm A. It scores **0.94 GS-HOTA lower**, and it is worse on both terms: DetA -0.71 against arm
A, and **AssA actually goes negative against the control (-0.69)**. It names 875 tracklets against
arm A's 729 and gets a *lower* identity accuracy (0.5316 vs 0.5463) for them.

Two things follow, and both matter for S7.

1. **The tracklet gate's objective (`d` at a precision floor) is not the scored objective.** Under
   GS-HOTA a wrong number is not a missing number — it deletes a detection that a `null` would have
   let match. The frontier point that maximises density subject to precision >= 0.80 is therefore
   the wrong point to ship; the reader's precision headroom should be banked as precision, not spent
   on density. Arm A takes precision 0.9256 at d 0.3139; arm B takes 0.8228 at 0.4148; arm A wins.
2. **The extra votes also cost association.** Arm B's mean identities per sequence rises 13.25 ->
   16.85 and merge precision is flat (0.495 vs 0.496), i.e. the denser evidence is splitting
   identities across more slots rather than consolidating them. This is the jersey-compatible merge
   gate doing what it is told with noisier numbers.

Arm B vs arm A is not statistically separated on 20 sequences (mean -0.850, 7 helped / 13 hurt,
Wilcoxon p = 0.216), so this is a directional finding on DEV, not a proven ordering. **Both arms
clear the DEV gate**; arm A is the recommendation.

## 7. Negatives, limits, and what was not done

1. **Arm B loses (§6)** — the gate-selected aggregation rule is not the best end-to-end rule. The
   density/precision trade is re-opened, not settled, and 20 sequences cannot resolve the 0.94-point
   gap (p = 0.216).
2. **The legibility model is untouched, and it is still the binding emit gate.** 12,814 of 67,224
   DEV-20 crops (19.1%) pass it; everything downstream reads only those. `CLUSTER_SESSION_S2.md` §4
   records that arm 1 (the legibility retrain) **was never run**. Every density number in this file
   is conditioned on the shipped 2024 legibility classifier.
3. **`POSE_CONF` is still dead code** (`gsr-pipeline-003`), left as is by instruction: every
   incumbent number was measured with the gate inert, so activating it is a re-baselining decision.
4. **No TEST-38 scoring.** The +3.66 is DEV-20 only. The v5/v5.1 pattern (a DEV win shrinking to
   +0.33/-0.05 on TEST-38 at p ~ 0.06-0.44) is the reason the gate was sized at +1.0 and the reason
   this number is not a projection of the held-out gain. S5's EIoU lever is the counter-example
   (held-out gain *larger* than DEV); neither precedent settles this one.
5. **The connector `tau` was not re-swept.** Frozen at 0.450 from the CLIP arm; the evidence
   changed substantially, so as in `GSR_EIOU.md` §8.4 these numbers are a lower bound on the lever.
6. **The digit-confusion prior was not refit** on v6 reads (`refit=False` throughout, as in the
   frozen EIoU downstream). The v6 reader's error pattern is different — 0.755 crop-level agreement
   with the incumbent at the 0.50 floor — so a refit is a live S7 candidate, not a measured one.
7. **The two `koshkina_percrop_votes_f080_v6_eiou` on-disk states are not distinguishable.** Arms A
   and B share a `config_key` (the key encodes the floor, not the rule), so running B overwrote A's
   vote cache. Both results are persisted in their own JSONs; the on-disk vote cache currently holds
   **arm B's** votes and must be wiped before A is rebuilt. Cheap to fix in S7 by keying on the rule.
8. **GSR labels no goalkeeper numbers**, so nothing here speaks to the GK-naming defect.

## 8. Reproduce

```
python -m eval.gsr_jersey --percrop --variant _v6 --max-crops 60 \
    --parseq-ckpt ~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt --seqs <DEV-20>
python -m tools.ocr_density --sweep --variant _v6                        # the tracklet gate
python -m tools.gsr_eiou --run --split dev --embedder clip \
    --grid "0.3:1:0.5:0.3" --percrop-variants ",_v6" --tag _s3reader     # arm A + control
python -m tools.gsr_eiou --run --split dev --embedder clip \
    --grid "0.3:1:0.5:0.3" --percrop-variants "_v6" --tag _s3readerB     # arm B (v6 rule)
pytest tests/test_gsr_eiou.py tests/test_ocr_density.py
```

## 9. Files

- Code: `eval/gsr_jersey.py`, `generator/jersey_id.py`, `tools/ocr_density.py`,
  `tools/gsr_eiou.py`, `tools/gsr_v4.py`, `tests/test_gsr_eiou.py`.
- Weights: `~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt` (363.8 MB, S2 arm 4t; server
  origin `~/src/jersey-number-pipeline/str/parseq/outputs/parseq/2026-08-05_21-44-57/`). Not in the
  repo.
- Evidence: `outputs/gsr/koshkina_percrop_v6/` (version-stamped `ocr-percrop-1.2`, `reader` column
  per row), `outputs/gsr/koshkina_percrop_v6_eiou/`, `outputs/gsr/koshkina_percrop_votes_f080_v6_eiou/`.
- Results: `results/ocr_density_rule_v6.json`, `results/ocr_density_rule_v6_eiou.json`,
  `results/gsr_benchmark/gsr_ocr_density_dev_v6.json`,
  `results/gsr_benchmark/gsr_eiou_dev_clip_{s3repro,s3reader,s3readerB}.json`.
- Logs: `outputs/gsr/{percrop_v6_dev20,percrop_v6_valid58,ocr_density_v6_sweep,s3_repro_dev20,
  s3_reader_dev20,s3_readerB_dev20}.log`.
- Claims: `s3-reader-001` (tracklet gate), `s3-reader-002` (DEV gate), `s3-reader-003` (the
  density-vs-precision negative).
