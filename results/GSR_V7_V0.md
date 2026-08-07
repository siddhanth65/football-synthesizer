# v7 session V0 — the pin + hygiene precursor (2026-08-07)

Every later v7 gate pairs against a DEV-20 control. This session pins the one library that was
measured to move the track partition across machines, re-derives that control on the pinned stack,
cleans the vote cache that two S3 arms shared, re-probes the unexplained zero-torso fault, and
audits disk on both machines. No model, metric or solver knob changed; the only repository edit is
one comment block in `requirements.txt`.

**Headline: the control re-derivation is EXACT.** GS-HOTA **51.808280888021784** against the
on-record **51.808280888021784** — delta **0.0000000000** on a +-1.5 gate — and the regenerated
20-sequence submission is **byte-identical** to the on-record arm directory, 20/20 files by sha256.

---

## 1. The pin — `supervision==0.30.0`, both machines

**Decision: 0.30.0.** The server (`a100server1`, conda env `gsr`) already ran 0.30.0 and holds every
TEST-38 / test-49 GPU artifact; the laptop was on 0.29.0.post0. Pinning DOWN would have invalidated
the newest and most expensive caches, so the laptop was brought UP.

| | before | after |
|---|---|---|
| laptop (`C:\Python314`, py 3.14.0, RTX 3050) | supervision 0.29.0.post0 | **0.30.0** |
| a100server1 (`~/miniconda3/envs/gsr`, py 3.11.15) | supervision 0.30.0 | **0.30.0** (unchanged) |

`pip install supervision==0.30.0 --dry-run` on the laptop reported exactly one action —
`Would install supervision-0.30.0` — so nothing else in the environment moved (numpy 2.4.0,
scipy 1.17.0, torch 2.11.0+cu128 all untouched). The wheel is `py3-none-any`, `Requires-Python
>=3.10`.

Recorded in:
- `requirements.txt` (laptop / repo), with the reason and the ceiling.
- `~/constraints.txt` (server), the file `PIP_CONSTRAINT` points at for every server-side install.

### 1.1 How the pin was verified

**(a) EIoU self-check.** `python -m tools.gsr_eiou --demo` ->
`gsr_eiou self-check OK: IoU 0.000 -> EIoU(e=0.7) 0.133 on the paper's motivating case`, the
documented value.

**(b) Targeted tests.** `pytest tests/test_gsr_eiou.py tests/test_tracking.py
tests/test_ocr_density.py` -> **13 passed**, 1 warning (the deprecation below).

**(c) 2-sequence tracked smoke.** Full GPU re-extraction (S4b detector, PnLCalib at
`calib_period=1`, ByteTrack) of SNGS-021 and SNGS-024 on the pinned stack, written to a scratch
`--out-dir` so no on-record cache was touched, then compared against the on-record
`outputs/gsr/positions_v6det` (extracted on the laptop under 0.29):

| sequence | supervision | rows | tracks | frames | mean track len | median |
|---|---|---:|---:|---:|---:|---:|
| SNGS-021 | 0.29 (on record) | 11,757 | 48 | 747 | 244.9 | 229 |
| SNGS-021 | **0.30 (pinned)** | **11,893** | **53** | **748** | 224.4 | 205 |
| SNGS-024 | 0.29 (on record) | 7,484 | 45 | 747 | 166.3 | 117 |
| SNGS-024 | **0.30 (pinned)** | **7,704** | **48** | **748** | 160.5 | 124 |

The version difference is real and reproduces on the laptop: **+10.4% and +6.7% more tracks**,
+1.2% / +2.9% more rows, one extra tracked frame, shorter tracks. This is the mechanism behind
`GSR_V6.md` negative 3 (-3.90 GS-HOTA on TEST-38 when arms were mixed across versions), now measured
on one machine with the detector held fixed. Runtime was 404 s and 376 s per sequence against the
on-record 0.29 extraction's ~390 s/sequence — no performance regression.

### 1.2 Two things the pin does NOT do (both must be carried into V1-V5)

1. **`sv.ByteTrack` is deprecated since 0.28.0 and is REMOVED in 0.31.0** (the pytest warning,
   raised from `generator/tracking.py:40`). 0.30.0 is the last release that carries it. The pin
   therefore has a hard expiry: moving past 0.30.x needs `generator/tracking.py` ported off
   `sv.ByteTrack` **and** a full re-extraction. Do not float this dependency.
2. **The pin is forward-looking; it does not retro-fix any cache.** Every DEV-20 GPU artifact was
   extracted on the laptop under 0.29; every TEST-38 / test-49 GPU artifact was extracted
   server-side under 0.30. Each split is internally same-stack, which is what the pairing rule
   requires, and the DEV control below is unaffected because it consumes those caches. The operative
   rule for v7: **no re-extraction of the DEV positions without re-deriving this whole control.**
   V2 (overlay detector), V3 (reader) and V4 (association) all read cached boxes and do not
   re-extract, so none of them trips it.

---

## 2. The DEV-20 FULL-bundle control — GATE PASS, exactly

Gate: within +-1.5 GS-HOTA of `results/gsr_benchmark/gsr_v6det_dev.json` (51.8082808880), else stop
the session. Re-derived CPU-only on the pinned stack (`python -m tools.gsr_v6det --stages arm
--split dev`) from the cached GPU artifacts `positions_gate_v6det`, `detbox_cache_v6det`,
`detembed_cache_clip_v6det`, `koshkina_percrop_v6_v6det`. `run_point`'s `clear_arm_caches` wiped the
arm's identity bundles, GTA arm and vote cache first, so everything downstream of the GPU caches was
genuinely rebuilt: EIoU relink (1,137 -> 2,289 tracks), vote densification, connector, MILP solver,
submission materialisation, official scorer.

| metric | on record | v7 control | delta |
|---|---|---|---|
| GS-HOTA | 51.808280888021784 | **51.808280888021784** | **+0.0000000000** |
| GS-DetA | 39.308307987997765 | 39.308307987997765 | +0.0000000000 |
| GS-AssA | 68.28680940240496 | 68.28680940240496 | +0.0000000000 |
| GS-LocA | 93.65233939165817 | 93.65233939165817 | +0.0000000000 |
| IDF1 | 56.452551003353804 | 56.452551003353804 | +0.0000000000 |

**VERDICT: PASS**, |delta| = 0 against a +-1.5 bar. Saved as
`results/gsr_benchmark/gsr_v7_control_dev.json` (with a `v7_v0` block recording the gate, the
reference and the stack). The on-record `gsr_v6det_dev.json` was not touched.

**Whole-JSON diff:** 259 leaves compared, **3 differ, none numeric** —
`/arm/tag` (`v6detdev_clip_e0.3r1w0.5a0.3` -> `v6dev_v6det_clip_e0.3r1w0.5a0.3`),
`/arm/detector/arm_variant` (added), `/arm/detector/origin` (dropped). All three are naming drift in
`tools/gsr_v6det.py` since the on-record run (`stage_arm` now folds `VARIANT` into the stem so the
`--detector control` arm gets its own directory). Max numeric delta across every differing leaf:
**0**. Per-sequence GS-HOTA is identical on all 20 sequences.

**Stronger, independent check.** Because the tag changed, the regenerated arm landed in a
*different* directory name and could be compared file-by-file against the on-record one:
`deleak_v6dev_v6det_clip_e0.3r1w0.5a0.3` vs `deleak_v6detdev_clip_e0.3r1w0.5a0.3` — same 20-file
keyset, **20/20 byte-identical by sha256**. The reproduction is not merely metric-equal; the shipped
rows are the same bytes. The duplicate was then deleted (124.5 MB reclaimed).

---

## 3. Vote-cache hygiene — arm B evicted, arm A rebuilt and re-verified

`GSR_S3_READER.md` §7.7: arms A and B share a `config_key` (it encodes the aggregation *floor*, not
the *rule*), so running B overwrote A's votes in
`outputs/gsr/koshkina_percrop_votes_f080_v6_eiou`.

**Pre-state found on disk (before any action this session):** the directory already carried a
`_rule.json` stamp reading `{"min_crop_conf": 0.9, "min_votes": 3, "min_legibility": 0.5,
"emit_all": false}` — **arm A's** (incumbent) rule, mtime 2026-08-06 11:33. Two S7 fixes had already
landed:
1. `tools.gsr_v4.votes_dir` now persists the rule beside the votes and wipes the cache on any
   mismatch (rule-keyed, as the §7.7 note asked for).
2. `results/ocr_density_rule_v6_eiou.json` — the variant sweep file that made
   `rule_for("0.80", "_v6_eiou")` resolve to arm B's denser rule — was renamed to
   `results/ocr_density_rule_v6_eiou.armB.json`, so the lookup falls back to the on-record
   0.80-floor freeze. **This rename is load-bearing: restore that filename and every future
   `_v6_eiou` arm silently becomes arm B.**

The directory was wiped and rebuilt anyway, and arm A re-derived end to end
(`eiou.run_point`, e 0.3 / rounds 1 / w_app 0.5 / app_max 0.30, CLIP, tau 0.450, `positions_gate`,
percrop variant `_v6`):

| | on record (`gsr_eiou_dev_clip_s3reader.json`) | rebuilt | delta |
|---|---|---|---|
| GS-HOTA | 42.74492210340575 | **42.74492210340575** | 0 |
| GS-DetA | 30.49944109110281 | 30.49944109110281 | 0 |
| GS-AssA | 59.90976080218577 | 59.90976080218577 | 0 |
| GS-LocA | 92.5648245755026 | 92.5648245755026 | 0 |
| IDF1 | 45.54818598618869 | 45.54818598618869 | 0 |
| n_merges / frags_after | 2,326 / 1,140 | 2,326 / 1,140 | 0 |
| pooled identity / jersey | 0.5463203055018623 / 0.6050005500290729 | identical | 0 |
| tracklets / named | 1,139 / 729 | 1,139 / 729 | 0 |

Per-sequence max |delta| over the 20 DEV sequences: **0**. The rebuilt cache holds 59 files with the
incumbent rule stamp.

**Determinism:** a second wipe + rebuild of the vote cache produced **59/59 byte-identical files**,
so `emit_votes` is reproducible from the per-crop evidence.

**Honest defect in this step's evidence.** The pre-wipe sha256 snapshot was written to the shared
scratchpad under a name a previous run also used and got clobbered, so "rebuilt bytes == pre-wipe
bytes" was *not* measured. The determinism check and the exact end-to-end score reproduction stand
in its place; they are the stronger evidence anyway, but the weaker one is missing and is recorded
as missing.

---

## 4. Zero-torso re-probe (30-minute box) — NOT REPRODUCIBLE

`GSR_V6.md` negative 2: two sequences produced zero torso RoIs, cause not established. Re-ran the
per-crop pass on the pinned stack for both (`eval.gsr_jersey --percrop --variant _v7probe
--max-crops 60 --parseq-ckpt parseq_v6_arm4t.ckpt --seqs SNGS-055,SNGS-056`), 201 s and ~200 s:

| sequence | rows | torso RoIs | legible >= 0.5 | numbered | tracks |
|---|---:|---:|---:|---:|---:|
| SNGS-055 on record `_v6` | 4,152 | 599 | 636 | 599 | 110 |
| SNGS-055 `_v7probe` (0.30) | 4,152 | **599** | 636 | 599 | 110 |
| SNGS-056 on record `_v6` | 3,876 | 882 | 890 | 882 | 112 |
| SNGS-056 `_v7probe` (0.30) | 3,876 | **882** | 890 | 882 | 112 |

Joined on `(track_id, frame)`: **100.00% agreement on the torso flag and on the read number over
every one of the 4,152 / 3,876 matched rows, max |delta legibility| = 0.** The reads are bit-identical
to the on-record artifact.

**Verdict: the fault was transient.** It does not reproduce on the pinned stack, and the pose stage
is byte-deterministic on exactly the two sequences that failed.

**Hypothesis, recorded as a hypothesis.** The reader shells out to a per-sequence STR sidecar in a
separate virtualenv (`jersey-str-env`) which writes torso crops and `parseq_positions.json` into
`eval.gsr_jersey.koshkina_work_dir()`. `KoshkinaRecognizer` **wipes that directory at
construction**. Before the per-PID fix that directory was shared, so a concurrently-running shard
could delete another shard's torso crops between the pose stage and the sidecar — producing exactly
this signature: legible crops present, zero torso RoIs, PARSeq never invoked, whole sequence unread.
That makes this the same root cause as `GSR_V6.md` negative 1 (the OCR sharding concurrency bug),
recorded there as a separate finding. Both mitigations are live in the tree today: the per-PID work
directory, and the `n_leg > 0 and not torso.any()` guard that discards the artifact so the resumable
pass rebuilds it.

**What was NOT established.** The original failing run's log is not on the laptop —
`outputs/gsr/percrop_v6_valid58.log` is the successful *re-run* (SNGS-055 in 1,271 s, SNGS-056 in
205 s) — so whether a second process was actually writing the shared scratch at the moment of the
fault is unproven. No frame-level pose dump was captured because the fault did not recur; there was
no failing frame to dump. The box was respected and the investigation stopped here. The item stays
**open-but-mitigated**, not solved.

---

## 5. Disk audit

### Laptop (`C:`)

| | GB |
|---|---:|
| volume total | 476 |
| used | 447 |
| **free** | **30 (94% used)** |
| `outputs/` total | 43.8 |
| — `outputs/gsr` | 31.7 |
| — `outputs/gsr_test` | 5.5 |
| — `outputs/gsr_srvtest` | 1.8 |
| `data/` | 38.8 |
| `matches/` | 10.4 |
| `.git` | 0.2 |

Reclaimed this session: 124.5 MB (the duplicate arm directory of §2). Net session growth: ~0.

**Flag for the campaign plan.** The v7 plan's disk line ("< 25 GB new against 334 free") is the
*server quota*. The laptop has **30 GB free**, and `CLUSTER_MIGRATION.md` §6.3 notes that a laptop
test-49 CLIP arm would need a cold embedding cache. Any laptop-side cold cache in V2-V5 must be
sized against 30 GB, not 334.

### Server (`a100server1`, user `siddhanth23519`)

| | value |
|---|---|
| quota used / limit | **167 GB / 500 GB** (headroom 333 GB) |
| underlying filesystem | `/dev/mapper/ubuntu--vg-ubuntu--lv`, 11 TB, 9.6 TB used, **765 GB avail (93% full)** |
| `~/data` | 99 GB |
| `~/runs` | 31 GB |
| `~/miniconda3` | 15 GB |
| `~/src` | 7.7 GB |
| `~/football-synthesizer` | 5.3 GB |
| `~/verify_t38` | 461 MB |
| `~/fs-frozen` | 114 MB |

**Second flag.** The quota headroom (333 GB) is larger than what the *shared* filesystem can
actually give (765 GB free across all users, 93% full). `~/data` has grown 32 GB -> 99 GB since
`CLUSTER_MIGRATION.md` §3.1. Budget against the filesystem, and clear `~/runs` (31 GB of past
fine-tune checkpoints) before V2/V3 training if space tightens.

---

## 6. What changed on disk

- **Repo (1 file):** `requirements.txt` — the `supervision==0.30.0` pin plus its reason and its
  0.31-removal ceiling. No Python module was edited; no `METRICS_VERSION` change (no metric moved).
- **New results:** `results/gsr_benchmark/gsr_v7_control_dev.json`, this document.
- **Server:** `~/constraints.txt` gained the same pin line.
- **Caches:** `koshkina_percrop_votes_f080_v6_eiou` (wiped + rebuilt, 59 files),
  `koshkina_percrop_votes_f080_v6_v6det_eiou` and the v6det identity-bundle / GTA arms (wiped +
  rebuilt by the control run, byte-identical output), `outputs/gsr/koshkina_percrop_v7probe/`
  (0.2 MB, the zero-torso probe evidence), `outputs/gsr/deleak_v6dev_v6det_clip_e0.3r1w0.5a0.3`
  (created, verified byte-identical to the on-record arm, deleted).
- **Nothing committed.**

## 7. Files

- `results/gsr_benchmark/gsr_v7_control_dev.json` — the pinned-stack DEV-20 control every v7 gate
  pairs against.
- `results/gsr_benchmark/gsr_v6det_dev.json` — the on-record reference (untouched).
- `results/gsr_benchmark/gsr_eiou_dev_clip_s3reader.json` — arm A's on-record numbers.
- `results/ocr_density_rule_v6_eiou.armB.json` — arm B's rule, parked under a name `rule_for` will
  not resolve. Do not rename it back.
- `outputs/gsr/koshkina_percrop_v7probe/{SNGS-055,SNGS-056}.parquet` — the zero-torso re-probe.
- `requirements.txt`, server `~/constraints.txt` — the pin.
- Claim: `v7-v0-001`.
