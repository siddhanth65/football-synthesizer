# v9 W3 — where GS-DetA bleeds, and the first repair of the largest reachable bucket

Campaign target (Sid): **beat 61.48 GS-HOTA on the SoccerNet GSR test board** (ours: 53.09 =
sqrt(39.32 x 71.67) on test-49). W2 closed the association axis: even an oracle splitter paired with
a perfect merger lands at ~60.4, so **GS-DetA is the binding constraint** and 61.48 needs DetA
**40.74** (with oracle split + perfect merge) to **45.23** (perfect merge only) against today's
**39.32**. W3 is therefore a detection/attribute session: decompose the DetA loss, rank the levers by
oracle value AND by what is actually reachable, then attack the top reachable one.

All of W3 runs on **DEV-20**, the tuning split, and on **laptop CPU only** (`0.000 GPU-h`).
TEST-38, test-49 and challenge were not read, scored or extracted at any point.

---

## 1. Registration — the instruments, written before the measurement

### 1.1 What GS-DetA actually gates on (read from the evaluator, not assumed)

`trackeval.datasets.SoccerNetGS.get_raw_seq_data` computes a gaussian similarity on the pitch
bottom-middle point, `exp(-0.5 (d/sigma)^2)` with `sigma = sqrt(25 / -2 ln 0.05) = 2.0428` m (so
5.00 m -> 0.05), and then **zeroes it whenever role, team or jersey disagree**. `HOTA.eval_sequence`
Hungarian-matches on that similarity and reports `DetA = TP / (TP + FN + FP)` averaged over the 19
thresholds `alpha = 0.05 .. 0.95` — i.e. a matched pair only scores at every alpha if it is inside
**0.654 m**, and scores at none if it is beyond 5 m. Attribute preprocessing (verbatim from the
evaluator): **team is compared only for `player`/`goalkeeper` rows and jersey only for `player`
rows**, on each side independently, and `None == None` **matches** — so a GT row the annotators left
unnumbered is free to us as long as we also emit `null`.

### 1.2 Two instruments, deliberately separate

* **Census** (`tools/gsr_v9_deta.py --census`) — an **attribute-blind** per-frame Hungarian match of
  our submission against GT on position alone (same sigma, cut at 5 m). Every GT row and every
  prediction row falls into exactly one bucket. This is a row census, not a metric.
* **Oracles** (`--oracles`) — counterfactual submissions that fix **exactly one** bucket, scored with
  the **official machinery** (`eval.gsr_score.gs_hota`, `gs_hota_full` config). The DetA delta is
  that bucket's point cost. Isolated arms are **not additive**; leave-one-out arms are reported
  alongside so each bucket is bracketed by [isolated, marginal].

**Harness fidelity.** The unmodified arm re-scores to **GS-DetA 39.3083 / GS-HOTA 51.8083**, the
on-record `GSR_V6.md` §2 DEV-20 numbers to four decimals, and the `all` oracle returns **DetA
100.0000** exactly. The arm read is `outputs/gsr/deleak_v6detdev_clip_e0.3r1w0.5a0.3` — the shipped
v6 bundle's DEV-20 submission, whose DetA (39.31) tracks test-49's (39.32) to 0.01.

### 1.3 Stage-2 registration (written BEFORE the component was built or measured)

Decided by §3's table: the top *reachable* lever is **role + team repair** (perfect role+team alone
is worth **+6.15 DetA**, more than the 5.91 the no-splitter path needs), and the cheapest bite at it
is that our submission lets **role and team flicker inside a single track** while a GT identity
carries one role and one team for the whole clip.

* **Component.** `eval.gsr_score.vote_track_attributes` — replace each track's per-row `role`/`team`
  (and, in one declared arm, `jersey`) with that track's majority value. Pure, deterministic, no GPU,
  no re-extraction. Ships **default OFF** (`VOTE_TRACK_ATTRS = ()`), one call site in
  `tools.gsr_deleak.write_arm` so every submission writer (DEV arms and the test package) routes
  through it.
* **Control.** The shipped DEV-20 arm itself. Only JSON attribute values change; the GPU artifacts
  are byte-identical, so the same-stack rule (kb v8-w4-004) is satisfied by construction and the
  experimental noise floor is **exactly 0** (no training, no seed, deterministic tie-break).
* **Arms, fixed a priori (4, no others):** `role`, `team`, `role+team`, `role+team+jersey`.
* **Gate to call it a bundle candidate (all three legs):**
  1. pooled DEV-20 **GS-DetA >= +0.50** over the control;
  2. pooled **GS-HOTA not worse** than the control;
  3. per-sequence paired GS-HOTA **helped > hurt** with **Wilcoxon p < 0.05** (n = 20).
* The +0.50 bar is a third of the smallest campaign-relevant DetA step (+1.42); a lever worth less
  than that does not earn a bundle slot.
* No TEST-38 / test-49 / challenge contact in this session whatever the verdict.

---

## 2. The census — 235,174 GT rows, 213,571 prediction rows, DEV-20

Exclusive buckets, checked in the order role -> team -> jersey (a row that fails role is charged to
role only). `%` is of all GT rows.

| bucket | rows | % of GT | note |
|---|---|---|---|
| **correct** (position-matched, all attributes right) | 132,808 | 56.47% | still loses 3.76 points to the alpha sweep, below |
| **(a) detection miss, live frame** | 20,075 | 8.54% | no prediction within 5 m |
| **(a) detection miss, dead frame** | 3,602 | 1.53% | the 263 frames (of 14,520) where our submission emits nothing at all |
| **(b) role wrong** | 12,547 | 5.34% | |
| **(c) team wrong** (role right) | 13,532 | 5.75% | |
| **(d) jersey wrong** (role+team right) | 52,610 | 22.37% | = coverage loss + named-wrong + over-naming |
| ... **coverage loss** (GT numbered, we emit `null`) | 36,138 | 15.37% | the single largest bucket |
| ... **named-wrong** (both numbered, different) | 13,360 | 5.68% | |
| ... **over-naming** (GT `null`, we emit a number) | 3,112 | 1.32% | |
| **(f) false positives** | 2,074 | 0.88% | 1,142 duplicates (a GT within 5 m was taken by another prediction) + 932 phantoms |

**(e) Localization.** On the 132,808 fully-correct rows the mean fraction of the 19 alpha thresholds
cleared is **0.9333**, i.e. **8,853 row-equivalents (3.76% of GT) are lost to the tolerance sweep
alone** — those rows are matched, correctly attributed, and still not TPs at the tighter alphas.

**Role confusion**, the whole of bucket (b): `player -> referee` **7,032**, `referee -> player`
3,377, `goalkeeper -> player` 1,792, `player -> goalkeeper` 313, `referee -> goalkeeper` 32,
`goalkeeper -> referee` 1. The dominant single failure is calling a player a referee.

**Detection misses by GT box height** (the S4b small-box hypothesis, re-measured end to end):

| GT box height | miss | hit | recall | share of all GT | share of all misses |
|---|---|---|---|---|---|
| < 30 px | 599 | 80 | **0.118** | 0.29% | 2.5% |
| < 40 px | 1,708 | 1,992 | **0.538** | 1.57% | 7.2% |
| < 50 px | 3,317 | 7,611 | 0.697 | 4.65% | 14.0% |
| < 60 px | 5,278 | 20,950 | 0.799 | 11.15% | 22.3% |
| < 80 px | 10,918 | 70,380 | 0.866 | 34.57% | 46.1% |
| >= 141 px (p90+) | 1,886 | 22,412 | 0.922 | 10.33% | 8.0% |
| **all** | 23,677 | 211,497 | **0.899** | 100% | 100% |

The `<40 px recall 0.538` reproduces S4b's on-record 0.55 end to end — the instrument agrees with
the record. **But small boxes are 1.57% of GT rows and 7.2% of all misses**: even perfect recall
below 40 px recovers 1,708 rows. Recall is 0.92 at the top decile and 0.90 overall, so the miss mass
is spread across every size bin, not concentrated in the far ones.

**Crowding** (best image-box IoU with another GT box in the same frame, a cheap occlusion proxy):
recall **0.809** on crowded rows (IoU > 0.3, 16,528 rows) vs **0.906** on isolated rows. Crowding
costs ~0.10 recall on 7% of GT.

---

## 3. The lever table — oracle cost, DEV-20, official evaluator

Each arm fixes one bucket and nothing else; `nomiss*` insert a prediction exactly on every missed GT
row **with GT attributes** (a detection you do not have has no attributes, so this is an upper
bound). Baseline **GS-DetA 39.3083**.

| # | oracle arm (fix only this) | rows touched | **GS-DetA** | **delta** | GS-HOTA delta |
|---|---|---|---|---|---|
| 1 | **jersey, all three sub-buckets** | 65,465 | 64.05 | **+24.74** | +10.60 |
| 2 | ... jersey **coverage loss** | 44,866 | 55.67 | **+16.36** | +7.67 |
| 3 | ... coverage loss **bounded by our own roster** | 24,518 | 47.46 | **+8.15** | +3.83 |
| 4 | **detection misses** (all 23,677, GT attributes) | 23,677 | 46.30 | **+6.99** | +1.40 |
| 5 | **role + team together** (`attrs_no_jersey`) | 38,149 | 45.45 | **+6.15** | +4.46 |
| 6 | ... jersey **named-wrong** | 15,406 | 44.37 | **+5.06** | +1.87 |
| 7 | **localization** (snap matched rows onto GT) | 211,497 | 43.05 | **+3.74** | +4.85 |
| 8 | **team** alone | 25,602 | 42.77 | **+3.47** | +2.46 |
| 9 | **role** alone | 12,547 | 40.73 | **+1.42** | +1.08 |
| 10 | ... jersey **over-naming** | 5,193 | 40.47 | **+1.16** | +0.61 |
| 11 | **dead-frame rows only** (subset of #4) | 3,602 | 40.43 | **+1.12** | +0.06 |
| 12 | **false-positive removal** | 2,074 | 39.49 | **+0.18** | +0.21 |
| — | all attributes (role+team+jersey) | | 80.03 | +40.72 | +19.59 |
| — | everything (`all`) | | **100.00** | +60.69 | +28.37 |

**Isolated arms understate, badly.** A row that is role-wrong is usually also team- or jersey-wrong,
so fixing one attribute alone leaves it a miss. Leave-one-out gives the other bracket:

| attribute | isolated | **marginal** (cost of leaving only this broken, from `attrs` 80.03) |
|---|---|---|
| role | +1.42 | **+8.13** |
| team | +3.47 | **+13.84** |
| jersey | +24.74 | **+34.58** |

**Reading the HOTA column.** `nomiss*` insert rows as fresh track ids, which is perfect detection with
zero association, so their GS-HOTA gain (+1.40, +0.06) is not the value of the lever — DetA is the
registered read for those arms.

### 3.1 Ranked levers with an honest achievable fraction

Campaign need: **+1.42 DetA** (to 40.74, with an oracle splitter + perfect merger) or **+5.91** (to
45.23, perfect merger only).

| lever | oracle DetA | what bounds the achievable share | honest estimate |
|---|---|---|---|
| **jersey coverage** (#2) | +16.36 | Our own emitted roster caps it at **+8.15** (#3) — the chain is GT-free, the solver may only name numbers its own reader produced. The abstention dial is **already fully open**: the frozen config carries `r_abstain = 0.0` (`results/identity_solver_config_percrop.json`), so the solver abstains for lack of a slot, not for want of payoff. On the 10 cached DEV bundles, **309 of 478 tracklets carry no confident read at all** and hold 24,543 numbered GT rows (29% of the numbered mass) — those need a better reader or more crops, not a better policy. | a few DetA at best, and it needs a reader/roster gain, not an operating point. The v6 lesson stands: the coverage/precision dial is already spent. |
| **role + team** (#5) | +6.15 | Purely a labelling problem on rows we already detect. A free first bite exists (§4): role/team **flicker inside single tracks** — 4.60% of rows carry a minority role and 6.36% a minority team within their own track. The rest needs a better role/team classifier. | flicker repair measured in §4; the remainder is a real but ordinary classifier problem. |
| **jersey named-wrong** (#6) | +5.06 | Attackable by a precision floor — but banked already: at full coverage the shipped reader runs at 0.7907 row precision and the coverage curve at an 0.85 floor buys precision by giving back 4.8 points of coverage. Under GS-DetA a *wrong* number costs exactly what `null` costs, so trading coverage for precision on numbered GT is **DetA-neutral at best**. | ~0 without a better reader. |
| **detection misses** (#4) | +6.99 | The oracle also grants perfect attributes on the inserted rows; only **62.8%** of the rows we do detect are attribute-perfect, so a perfect detector with our attribute quality is worth roughly **+4.4**. Needs a detector retrain (priced in §5). Small boxes are 7.2% of the miss mass, crowding ~13%. | +2 to +4.4, at 6-10 GPU-h plus a full same-stack control re-extraction. |
| **localization** (#7) | +3.74 | Calibration accuracy, not detection. Our GS-LocA is already 93.65; the residual is homography noise on rows we already match. W2b priced the classical line solver at 0.35 m median on dead frames, but it was measured as **+0.30 GS-HOTA against a +0.40 bar** end to end. | small; the cheap version already failed its gate. |
| **jersey over-naming** (#10) | +1.16 | Requires predicting which GT rows the annotators left unnumbered. v8-w3-007: that pool is **94.8% genuinely unreadable**, so our own reader is weakest exactly there — but the coverage curve says buying that precision costs more coverage than it saves. | ~0 as a standalone dial. |
| **dead frames / the `onpitch_plausible` veto** (#11) | +1.12 | The whole dead-frame population is worth 1.12 DetA even if perfectly recovered, and W2b's actual line-solve arm delivered **+0.30 GS-HOTA vs a +0.40 bar** (kb v8-w2b-001). | +0.3 GS-HOTA, already measured and already failed. |
| **false positives / dedup** (#12) | +0.18 | Nothing to win: 2,074 FP rows in 213,571. | **dead lever — do not build** |

---

## 4. Stage 2 — the per-track attribute vote (registered in §1.3 before it was built)

`eval.gsr_score.vote_track_attributes` forces every track's `role`/`team`/(`jersey`) to its own
majority value over that track's rows. Ties keep the first-seen value, which makes the operation
deterministic. Applied to the shipped DEV-20 submission; the control is that same submission
untouched. Scored with the official evaluator.

### 4.1 Why the operation is not a heuristic

The submission our chain writes lets a single track disagree with itself: **4.60% of prediction rows
carry a minority role** within their own track, **6.36% a minority team** and **2.22% a minority
jersey** (285 / 510 / 163 of the 896 DEV-20 tracks are internally inconsistent). The jersey case has
a mechanical cause: `tools.gsr_deleak.write_arm` writes the solver's number **only onto rows whose
per-row role is `player`**, so every row the detector called goalkeeper or referee inside a named
track keeps `jersey = null`. A GT identity has one role and one team for the whole clip, so every
minority row is a guaranteed error *unless the majority is the wrong value*.

### 4.2 Results — DEV-20, official evaluator, control = the same submission untouched

| arm | rows changed | **GS-DetA** | **delta** | GS-AssA delta | **GS-HOTA** | delta | paired per-seq HOTA | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| control (shipped v6 DEV-20) | — | 39.3083 | — | — | 51.8083 | — | — | — |
| `vote_role` | 10,046 | 39.1804 | **-0.128** | +1.06 | 52.1248 | +0.317 | +0.360, 15 helped / 4 hurt | 0.0100 |
| `vote_team` | 22,094 | 39.9390 | **+0.631** | +1.89 | 52.9391 | +1.131 | +1.115, 20 / 0 | 1.9e-06 |
| `vote_roleteam` | 32,140 | 40.0573 | **+0.749** | +3.23 | 53.5207 | +1.712 | +1.719, 20 / 0 | 1.9e-06 |
| **`vote_all` (role+team+jersey)** | **43,485** | **41.1082** | **+1.800** | **+4.76** | **54.7970** | **+2.989** | **+2.984, 20 / 0** | **1.9e-06** |

**Registered verdict: PASS on `vote_all`.** All three legs of the §1.3 gate: DetA **+1.7999** vs the
+0.50 bar; GS-HOTA **+2.9887**, not worse; **20 sequences helped, 0 hurt**, Wilcoxon **p = 1.9e-06**.
Per-sequence GS-DetA: 18 helped / 2 hurt (worst -1.68 on SNGS-078, best +6.47 on SNGS-048); per
sequence GS-HOTA the *worst* sequence is **+0.177**, i.e. nothing regressed. The noise floor is
exactly 0 — the operation is deterministic and the GPU artifacts are byte-identical to the control's.

Three honest notes on the arms:

1. **Role voting alone LOSES DetA (-0.128)** while gaining AssA (+1.06). Collapsing a track onto its
   majority role destroys the rows where the minority was right — the census's 1,792
   `goalkeeper -> player` rows are exactly that shape (a GK detected correctly on part of a track
   that is mostly labelled `player`). Role voting only pays *in combination*, because a consistent
   role is what lets the team and jersey values reach every row.
2. **Most of the gain is the jersey leg** (+1.05 DetA on top of `vote_roleteam`), which is not a new
   number: it is the solver's own number reaching the rows `write_arm` skipped. That is a **writer
   bug being repaired**, and it lands inside the largest bucket of §2 (jersey coverage).
3. **AssA rises by 4.76** as a by-product: the evaluator zeroes similarity on attribute mismatch, so
   an internally inconsistent track is *already* being cut into pieces by the metric. This is a DetA
   session, so AssA is reported and not claimed.

### 4.3 Legitimacy and the audit

The vote reads **only our own predictions** — no `Labels-GameState.json`, no GT — so the GT-free
standard of `GSR_V6.md` §4 is preserved. `tools.gsr_calibfill.verify_gtfree`, however, compares each
shipped row's team against the *base* arm's, so a voted submission would read as 22,094 violations.
The audit is therefore extended to apply the same pure vote to its expectation before comparing — a
**no-op while the flag is empty**, verified twice: `tools.gsr_calibfill` self-check green (it
exercises the flipped-map branch), and the on-record test-49 audit re-derived through the refactored
function returns **0 violations / 467,425 rows / flip set {SNGS-126, -130, -131, -197}**, identical
to `GSR_V6.md` §4.

The same audit could **not** be re-derived with the flag ON on DEV-20: the laptop's cached base arm
(`outputs/gsr/eval_v4_gta_f080_v6_v6det_eiou_t0.450_jg_clip_v6det_eiou`) holds a different row count
from the shipped DEV submission (SNGS-024: 6,915 vs 6,701), i.e. it is some later session's arm —
a fresh instance of the in-place-overwrite hazard kb v9-w1-006. Stated, not worked around.

### 4.4 What this is and is not

It is **+1.80 GS-DetA and +2.99 GS-HOTA on DEV-20 for zero GPU**, from a 30-line pure function, and
it takes DEV-20 DetA from 39.31 to **41.11** — past the 40.74 that W2's arithmetic needs on the
oracle-split path, and 4.1 short of the 45.23 the no-splitter path needs. It is **not** a board
number: DEV-20 is the tuning split, the transfer assumption is W1's (DEV-20 DetA tracked test-49 to
0.01 before this change, which is the only support it has), and the component ships **default OFF**
until a TEST-38 confirmation is run and a bundle decision is taken. No submission was built.

---

## 5. What a detector retrain would actually cost (priced, not built)

Registered in the brief as the STOP-and-price branch. Lever #4 is worth at most **+6.99 DetA**
(oracle, perfect attributes) and realistically **+2 to +4.4**, against these costs:

* **Training.** The S4b recipe (`gsr_v3_ft_b_last.pt`, YOLOv8s, GSR + SoccerNet-v3) was a 10-epoch
  fine-tune plus a second 10-epoch dose. A small-object-focused rerun (higher `imgsz`, a tiled or
  copy-paste small-object augmentation) is ~2-4 GPU-h per arm on the cluster A100, 3 arms = 6-12
  GPU-h, comfortably inside the 12 GPU-h W3 budget.
* **The control lineage is the expensive part, not the training.** kb v8-w4-004: any re-extraction
  must be paired against a **same-stack** control, because a flag-OFF re-extraction of the on-record
  artifacts already differs by **-1.14 GS-HOTA** on the 10-sequence probe. A detector arm therefore
  needs a full DEV-20 re-extraction of *both* arms (extract + gate + percrop + boxes + embeddings),
  which W1 measured at **1.63 GPU-h for 57 sequences** — call it ~0.6 GPU-h per arm for 20
  sequences, plus CPU re-association and a re-run of the connector/solver chain per arm.
* **Expected value.** Even the oracle (+6.99) barely clears the no-splitter requirement (+5.91), and
  the realistic band (+2 to +4.4) does not reach it alone. It is a *contributor*, not a solution.

---

## 6. Negatives, limits and hazards

1. **The census matcher is not the evaluator's matcher.** It is attribute-blind and unweighted; the
   evaluator's Hungarian runs on attribute-zeroed similarity weighted by the global alignment score.
   The census answers "is there a prediction near this GT row at all"; every point cost in §3 comes
   from the official machinery instead.
2. **Isolated oracles are not additive** and understate every attribute bucket (§3). Both brackets
   are reported; neither alone is the bucket's "value".
3. **`nomiss*` oracles grant perfect attributes and perfect association on inserted rows.** Their
   DetA is a hard upper bound; their GS-HOTA is meaningless (new track ids destroy AssA).
4. **DEV-20 is the tuning split** and everything here is measured on it, repeatedly and deliberately.
   Nothing in this session was read from TEST-38, test-49 or challenge.
5. **The DetA target itself inherits W1/W2's assumptions** (cross-split additivity of DEV-20 proxy
   deltas onto test-49 levels, and the linear apportionment of the split leg). DEV-20's DetA 39.31
   sits 0.01 from test-49's 39.32, which is the only support that transfer has.
6. **The jersey-coverage roster bound (#3) uses our own submission's emitted numbers**, which is a
   proxy for the solver's self-roster, not the solver's actual slot set.
7. **The vote's largest leg is a writer repair, not a modelling gain** (§4.2 note 2), and role
   voting on its own is *negative* on DetA. Anyone quoting "+1.80 DetA from a majority vote" without
   those two facts is quoting it wrong.
8. **The "coverage/precision dial is already spent" reading rests on the frozen config**
   (`r_abstain = 0.0`) and on 10 of 20 cached DEV bundles (478 tracklets). The other 10 bundles were
   not rebuilt.

---

## 7. Files

**Laptop** (`c:\Users\siddh_ygv5bws\football-synthesizer\`)
- `tools/gsr_v9_deta.py` — census + oracle counterfactual driver (`--demo` self-check green)
- `eval/gsr_score.py` — `vote_track_attributes` + the `VOTE_TRACK_ATTRS` flag (default OFF)
- `tools/gsr_deleak.py` — the one call site in `write_arm`
- `tests/test_gsr_v9_deta.py` — targeted tests
- `results/gsr_benchmark/gsr_v9_w3_census.json` — the per-sequence row census
- `results/gsr_benchmark/gsr_v9_w3_oracles.json` — every oracle arm's official score
- `results/gsr_benchmark/gsr_v9_w3_vote.json` — the stage-2 arms
- `results/GSR_V9_W3.md` — this document

**GPU used: 0.000 GPU-h.** The cluster was not needed and was not touched.
