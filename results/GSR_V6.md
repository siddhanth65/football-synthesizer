# S7 — the v6 combined bundle: DEV-20 ablation, TEST-38 gate, test-49 spend

Campaign v6, session S7. Three components were built and gated separately (S2/S3 jersey reader,
S4/S4b detector, S5 EIoU association). This session combines them, prices each one's inclusion on
DEV-20, runs the single pre-declared TEST-38 gate, and spends the campaign's fourth test-49 run.

**Headline: all three components clear the DEV-20 inclusion bar (detector -9.06, reader -5.52,
EIoU -2.09 GS-HOTA if dropped), the combined bundle passes the raised TEST-38 gate at 49.50 vs
40.0, and the once-only test-49 run scores GS-HOTA 53.0846 — +14.06 over the public board's 39.02.**
Package `gsr_testphase_gtfree_v6_28f3986b.zip` (467,425 predictions, 49 sequences) is built and
self-score-verified; upload is Sid's action.

Development was DEV-20 and TEST-38 only until the single test-49 run in §4. No result in this
document mixes GPU artifacts from different machines within a paired comparison (§1, provenance).

---

## 1. The bundle and the freeze

Frozen: `results/gsr_v6_frozen.json`, declared **2026-08-06T05:53:08Z** (amended 08:39:16Z to add
machine/stack provenance only — the bundle definition itself did not change), before any TEST-38
read of this bundle.

| component | name | artifact | md5 / sha256 |
|---|---|---|---|
| detector | YOLOv8s GSR+SoccerNet-v3 fine-tune (S4b) | `models/gsr_det/gsr_v3_ft_b_last.pt` | md5 `2074d8741f97a6892a1322d0f808244c` |
| jersey reader | PARSeq v6 arm 4t (S2) | `parseq_v6_arm4t.ckpt` | md5 `39c0c15defafda072ca0cefd0a4b8e18` |
| association | clean-room ExpansionIoU (S5) | `tools/gsr_eiou.py` | module sha256 `44176b1e...051d2b` |

Reader aggregation rule: the **incumbent 0.80-floor freeze (S3 arm A)** — `min_crop_conf 0.9,
min_votes 3, min_legibility 0.5, emit_all false` — explicitly NOT the v6 sweep's denser rule
(S3 §6 measured that rule worse end to end). Downstream frozen and untouched: aggregation floor
0.80, connector `tau = 0.45`, jersey-compatible merge on, digit-confusion refit off (S7 re-measured
this as inert, §2).

**Provenance (the same-machine pairing rule, `results/CLUSTER_MIGRATION.md`).** DEV-20 ablation:
laptop (RTX 3050, py3.14, `supervision 0.28.x`) — every DEV arm and its controls came off that one
stack. TEST-38 and test-49 GPU stages (detector inference, box cache, embeddings): a100server1
GPU 1 (py3.11, torch 2.5.1+cu121, `supervision 0.30.0`), moved there 2026-08-06 to free the laptop
GPU. TEST-38 and test-49 CPU stages (re-association, connector, solver, official scorer): laptop.
Every paired comparison inside a split uses arms whose GPU artifacts came from the same machine —
the detector-OUT TEST-38 control was therefore re-extracted server-side rather than read from the
laptop's DEV-era cache. Cross-stack check on SNGS-021 (S4b detector, both stacks): 94.4% paired
detections, pitch-disagreement median 0.0657 m / p90 0.307 m, role and team agreement both 1.0 —
close but not identical, which is why §5's cross-stack tracker item is escalated rather than
absorbed. `tools/gsr_eiou.py` in the working tree differs from `gsr_eiou_frozen.json`
(S3 added per-variant plumbing); AST-checked identical on `expand_boxes`, `eiou_matrix`, `_match`,
`associate`, `relink_sequence`, `attach_boxes` — only `clear_arm_caches`, `run_point`, `sweep`,
`main` changed.

---

## 2. DEV-20 leave-one-out — the inclusion bar

Source: `results/gsr_benchmark/gsr_v6det_dev.json` (full bundle), `gsr_v6det_dev_incumbent.json`
(v6 reader dropped, incumbent Koshkina reader in its place), `gsr_v6det_dev_noeiou.json` (EIoU
dropped, ByteTrack in its place), and the S5/S3 on-record numbers for the detector-out arm
(`results/gsr_eiou_dev_clip_s7armA.json`, matching `GSR_EIOU.md` §4.1 and `GSR_S3_READER.md` to
four decimals). All 20 DEV sequences, one detector/reader/tracker stack (laptop).

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| ByteTrack + incumbent reader + old detector (base control) | 37.0711 | 23.9202 | 57.4538 | 92.2701 | 38.1219 |
| + EIoU (old detector, incumbent reader) | 39.0820 | 25.7403 | 59.3412 | 92.4160 | 39.8526 |
| + EIoU + v6 reader (old detector = **detector dropped from FULL**) | 42.7449 | 30.4994 | 59.9098 | 92.5648 | 45.5482 |
| ByteTrack + v6 reader + S4b detector (**EIoU dropped from FULL**) | 49.7221 | 37.3018 | 66.2821 | 93.5240 | 54.1853 |
| EIoU + incumbent reader + S4b detector (**v6 reader dropped from FULL**) | 46.2903 | 31.9676 | 67.0320 | 93.4547 | 48.2951 |
| **FULL bundle (EIoU + v6 reader + S4b detector)** | **51.8083** | **39.3083** | **68.2868** | **93.6523** | **56.4526** |

Leave-one-out deltas versus FULL (51.8083):

| dropped component | delta vs FULL | inclusion bar (>= 1.0) |
|---|---|---|
| detector | **-9.0634** | PASS, by a wide margin |
| v6 jersey reader | **-5.5179** | PASS |
| EIoU association | **-2.0862** | PASS |

**Inclusion rule** (`gsr_v6_frozen.json`): a component is included when dropping it costs
`>= 1.0` GS-HOTA on DEV-20. All three components clear the bar with headroom; none is a marginal
call.

**Two negatives priced on the same DEV-20 base, both closed before the freeze.**

*Connector tau re-sweep* (`results/gsr_benchmark/gsr_eiou_dev_clip_s7tau0.320.json`,
`...s7tau0.380.json`, `...s7tau0.520.json`, `...s7armA.json`, all on the FULL-minus-detector arm
`eioudev_clip_e0.3r1w0.5a0.3_v6`):

| tau | GS-HOTA |
|---|---|
| 0.320 | 42.3333 |
| 0.380 | 42.8814 |
| **0.450 (frozen)** | **42.7449** |
| 0.520 | 42.7226 |

Flat; best alternative (+0.14 at tau 0.380) is below the 1.0-point bar. S5's "lower-bound" caveat
on tau (never re-swept on the EIoU partition) is closed: tau stays 0.450.

*Digit-confusion prior refit* (`results/gsr_benchmark/gsr_v6_dev_refit.json`): GS-HOTA 42.7394
against the un-refit 42.7449 (`paired_vs_armA`: mean -0.0040, 0 helped / 2 hurt, worst -0.0565).
Inert — stays off.

---

## 3. TEST-38 — the single pre-declared gate

Gate (`gsr_v6_frozen.json.pre_declared_gate`, from STATUS 2026-08-06, raised from the plan's 37.5
because S5 had already put 39.54 on record): **valid TEST-38 GS-HOTA >= 40.0**, controls v4 36.71
/ EIoU v5 39.54.

Result (`results/gsr_benchmark/gsr_v6det_t38_v6det.json`, 38 held-out sequences, server GPU +
laptop CPU stages): **GS-HOTA 49.4970 / GS-DetA 35.6331 / GS-AssA 68.7590 / GS-LocA 93.3378 / IDF1
52.6830 — PASS by +9.50 over the 40.0 gate.**

**Control 1 — the EIoU-only re-derivation, to the digit.** Re-run in this session
(`results/gsr_benchmark/gsr_eiou_t38_clip_s7controls.json`): control GS-HOTA **37.0344**, EIoU-only
arm GS-HOTA **39.5403** (39.54032402174176), paired mean +2.535 / median 2.406, 26 helped / 12
hurt, Wilcoxon **p = 0.00092286903236527** — the frozen S5 number
(`GSR_EIOU.md` §7) to the digit, re-verified end to end (`CLUSTER_MIGRATION.md` §4 additionally
confirms this on the server-migrated stack to 5-6 significant figures). This is the harness-
fidelity evidence for the whole session, not a fresh claim.

**Control 2 — same-stack pairing, detector in vs. out.** Both arms are v6-reader + EIoU, differing
only in the detector, both re-extracted on the a100server1 stack so the pairing is apples-to-apples
(`results/gsr_benchmark/gsr_v6det_t38_base.json` = old/base detector, `gsr_v6det_t38_v6det.json` =
S4b detector):

| | GS-HOTA | GS-DetA | GS-AssA | GS-LocA |
|---|---|---|---|---|
| base detector (old, pretrained) | 39.3818 | 25.8041 | 60.1047 | 93.1242 |
| S4b detector | 49.4970 | 35.6331 | 68.7590 | 93.3378 |
| **pooled delta** | **+10.115 (~"+10.12")** | +9.829 | +8.654 | +0.214 |

Per-sequence paired (n=38, same 38 sequences both arms): mean **+11.60**, **34 helped / 4 hurt**,
Wilcoxon **p = 4.66e-09** (rounds to the STATUS-quoted 4.7e-09). The pooled GS-HOTA delta (+10.12)
and the per-sequence paired mean (+11.60) differ because GS-HOTA is not a simple per-sequence
average of its pooled inputs — both numbers are reported so neither is mistaken for the other.

---

## 4. test-49 — the single run, once

Gate cleared, test-49 spent once. Source: `results/gsr_benchmark/gsr_v6det_test_v6det.json`, 49
sequences (the official SoccerNet GSR test-phase split, `SNGS-116..150, 187..200`).

| metric | v6 (once) | public board | delta |
|---|---|---|---|
| GS-HOTA | **53.0846** | 39.02 | **+14.06** |
| GS-DetA | **39.3231** | — | — |
| GS-AssA | **71.6704** | — | — |
| GS-LocA | **94.0597** | — | — |
| IDF1 | 56.3703 | — | — |

Public arc: 31.88 -> 33.37 -> 35.40 -> 39.02 -> (pending Sid's upload) **53.08**. Internal arc from
campaign start: 22.85 -> 53.08 across ~6 weeks, every step frozen then verified.

**Legitimacy.** `manifest_gtfree_v6.json`: `label_reads: "none. ... No Labels-GameState.json is
opened anywhere in the prediction chain."` — the team map comes from `resolve_team_map_free`
(geometry only) and the roster from the sequence's own OCR reads, the same standard as
`GSR_DELEAK.md` §6. The row-level audit is materialized at
`results/gsr_benchmark/gsr_v6_legitimacy_audit.json`, via `tools.gsr_calibfill.verify_gtfree`
(every shipped row's `team` must equal `free_map[cluster]` — the cached base arm's value, flipped
only on the sequences where the free map disagrees with the GT-agreement map; any other row is a
violation). Run independently on the laptop against
`outputs/gsr_srvtest/deleak_v6test_v6det_clip_e0.3r1w0.5a0.3` (the exact directory the shipped zip
was packaged from): **0 violations over 467,425 rows, 49 sequences**, matching
`results/gsr_benchmark/gsr_v6_testsplit_package.json`'s embedded `legitimacy` block exactly (same
counts, same flip set) — this is a re-derivation, not a re-read of that file. **Flip set: SNGS-126,
SNGS-130, SNGS-131, SNGS-197** (the 4 sequences where the geometric free-map resolver disagrees
with the GT-agreement map) — corroborated independently by the raw solver `identity` metric in
`gsr_v6det_test_v6det.json`, which is near-zero on exactly these four sequences (0.0039 / 0.0244 /
0.0240 / 0.0279) and normal everywhere else, including SNGS-190 (0.757). Note for the record: an
earlier verbal restatement of this flip set as `{126, 131, 190, 197}` was a transcription slip —
190 is not a flip sequence; 130 is. The violation count (0) is unaffected either way.

**Package.** `results/gsr_submission/gsr_testphase_gtfree_v6_28f3986b.zip` — 35,483,700 bytes, 49
sequences, layout `tracklab/<SEQ>.json`, built 2026-08-06T15:00:23Z from
`outputs/gsr_srvtest/deleak_v6test_v6det_clip_e0.3r1w0.5a0.3/predictions/data`.

**Zip self-score verification** (`results/gsr_submission/zip_selfscore_gtfree_v6.json`): re-scoring
the packaged zip reproduces the combined numbers **identical to 4 decimal places** — GS-HOTA
53.08461849762875 both places, GS-DetA 39.323050301425525, GS-AssA 71.67039386681004, GS-LocA
94.05967148264726. The package is exactly what was measured.

Competition target: codabench 4365 (2025 SoccerNet GSR — Test Phase). **AWAITING SID'S CODABENCH
UPLOAD.**

---

## 5. Negatives, verbatim-faithful to the record

1. **A worker-introduced sharding concurrency bug in the OCR scratch directory.** Caught by
   measurement (32.4% read-agreement, well below the expected range), not by inspection. The
   contaminated batch was discarded and redone; a per-PID fix shipped. No on-record artifact was
   affected — the bug was caught before anything downstream consumed the bad batch.
2. **A zero-torso fault on 2 sequences.** Repaired, a guard added against recurrence. Cause not
   established — recorded as an open item, not a solved one.
3. **A cross-stack tracker difference: `supervision 0.29` (laptop, DEV-era) vs `0.30` (server,
   TEST-38/test-49 stage) produce different track partitions** — 37% more fragments on the newer
   version, and a measured **-3.90 GS-HOTA on TEST-38** when the versions are mixed. This is why
   §3's controls are explicitly same-stack: every paired claim in this document uses arms from one
   machine. Escalated as a pin-the-version backlog item, not fixed this session.
4. **Team-map flips are now the dominant per-sequence failure mode**, roughly **+3.1 GS-HOTA of
   estimated headroom** concentrated in the same small number of known-bad clips (the audited flip
   set itself, §4: SNGS-126/130/131/197) — stated as an *estimate, not a measurement*, and not
   banked anywhere in the numbers above.
5. **The detector dose decision (S4b's second 10-epoch run) touched valid, not test.** test-49 is
   its clean, never-touched split — the dose tuning that happened on DEV/valid does not contaminate
   the once-only test-49 spend.
6. **v4 is quoted, not re-derived, in this session.** The 36.71 TEST-38 control in the gate table
   comes from the on-record `CLUSTER_SESSION4.md` number, not a fresh run here. The EIoU control's
   re-derivation to the digit (§3, Control 1) is the harness-fidelity evidence that stands in for
   re-deriving v4 as well — the two share the same scorer, splits and CPU stack.

---

## 6. Public arc and backlog

Public leaderboard arc across the campaign: 31.88 -> 33.37 -> 35.40 -> 39.02 -> pending 53.08.
Backlog carried out of this session: pin `supervision` to one version across laptop and server
(item 3 above); resolve the team-map flip headroom on the known clips (item 4, now the audited
flip set SNGS-126/130/131/197, §4).

## 7. Files

- `results/gsr_v6_frozen.json` — the freeze record (declared + amended).
- `results/gsr_benchmark/gsr_v6det_dev.json`, `gsr_v6det_dev_incumbent.json`,
  `gsr_v6det_dev_noeiou.json` — the DEV-20 leave-one-out arms.
- `results/gsr_benchmark/gsr_eiou_dev_clip_s7armA.json`, `...s7tau0.320.json`, `...s7tau0.380.json`,
  `...s7tau0.520.json` — the tau re-sweep.
- `results/gsr_benchmark/gsr_v6_dev_refit.json` — the digit-confusion prior refit check.
- `results/gsr_benchmark/gsr_v6det_t38_base.json`, `gsr_v6det_t38_v6det.json` — the same-stack
  detector-in/out TEST-38 pairing.
- `results/gsr_benchmark/gsr_eiou_t38_clip_s7controls.json` — the EIoU-only TEST-38 re-derivation.
- `results/gsr_benchmark/gsr_v6det_test_v6det.json` — the once-only test-49 run.
- `results/gsr_submission/manifest_gtfree_v6.json`, `zip_selfscore_gtfree_v6.json`,
  `gsr_testphase_gtfree_v6_28f3986b.zip` — the submission package and its self-score verification.
- `results/gsr_benchmark/gsr_v6_legitimacy_audit.json` — the independently re-derived row-level
  legitimacy audit (0 violations / 467,425 rows / flip set SNGS-126,130,131,197).
- `results/GSR_S3_READER.md`, `results/CLUSTER_SESSION_S4.md`, `results/GSR_EIOU.md` — component
  provenance (jersey reader, detector, association).
- `results/CLUSTER_MIGRATION.md` — GPU/CPU stack provenance and the same-machine pairing rule.
