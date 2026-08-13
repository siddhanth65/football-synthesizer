# v8 session W3-PREP — the glyph-only corpus, audited, and Sid's first evening made turnkey

Campaign v8, session W3, preparation half. **No training, no scoring, no GPU job longer than three
minutes.** The job was to do everything the annotation evening does not need Sid for, and to audit
the corpus he will be adding to before he adds to it.

**Three measured results reframe the W3 plan, and two of them contradict a premise the plan was
carrying.**

1. **The whole non-Sid corpus except SoccerNet-v3 is already on this laptop**, and the laptop
   reproduces the server-side S2 corpus figure **exactly**: `CLUSTER_SESSION_S2.md` §3.3 records
   "GSR train: 4,175 of 13,994" passing the legibility filter at 0.7; a fresh local pass over the
   same crop law returns **4,175 of 13,994**. SoccerNet-v3 (45 GB, the largest single source at
   42,479 corpus rows) is server-only and **cannot come here** — 33 GB free on C:.
2. **GSR's jersey label is 100% identity-carried, measured, not assumed.** Of 1,224 GSR-train
   player/GK tracklets, **0 mix numbered and un-numbered boxes** and **0 carry two different
   numbers**. A number is a property of the tracklet, never an observation about a frame. The task's
   premise is confirmed at the strongest available resolution, and the shipped legibility filter is
   the only thing standing between that label and a corpus full of invisible glyphs.
3. **SoccerNet-GSR never labels a goalkeeper's number — anywhere.** 0 of 79 train GKs and 0 of 77
   validation GKs carry one, against 933 of 1,145 train players. **27% of the unnamed-tracklet pool
   is not a legibility failure at all; it is an annotation policy.** No filter, no reader and no
   amount of GPU can recover those numbers from this dataset. A human is the only possible source.

And a fourth, produced while eyeballing the deliverable: **`SNGS-065` player track 6 and referee
track 22 are the same body at frame 000254** (IoU ≈ 0.85, both boxes smooth across their
neighbours). One more GSR ground-truth error, found the same way W0's five were. Not audited
systematically — see §7.

Deliverables: `tools/gsr_w3_corpus.py`, `outputs/gsr/w3_annotation/` (499 contact sheets + the
labelling CSV + the HOWTO + the corpus manifest), and §6's draft registration.

---

## 1. Split hygiene — the rules this session ran under

Stated up front because the whole value of the corpus is that these were not bent.

| rule | how it is enforced |
|---|---|
| Label sources are **GSR train** and **jersey-2023** only | `gsr_w3_corpus._assert_train_only` raises on any other split; every label-producing entry point calls it. Covered by `--demo`. |
| **DEV-20 is a failure-mode reference only**, never a label source | DEV-20 is `sorted(validation)[::3]`; no `validation` sequence appears in any artifact this session produced. The manifest carries an explicit leakage check that prints the intersection with `validation + test`: **NONE**. |
| **TEST-38 and test-49 untouched** | Neither was opened. No scoring of any kind was run. |
| **Challenge-split self-training BLOCKED** pending a codabench-rules check | Not prepared, not planned, not mentioned in the draft registration below. |
| **Glyph-only**: a `(crop, number)` pair enters only if the number is plausibly visible in THAT crop | GSR-train and jersey-2023 numbers are both identity-carried (§3), so both are admitted **only** through the shipped ResNet34 legibility classifier at its shipped **0.7** threshold — the S2/S3 discipline, same weights, same value. |
| **Anti-teacher-imitation**: no model-derived abstention labels | The manifest emits **no** no-number rows at all. Every abstention label in the corpus comes from a human: Sid's tier-A `none` verdicts, or jersey-2023's 403 human-marked illegible tracklets (§3.2). |

One deliberate, declared use of a model that is **not** a label: the tier-B queue is *ranked* by the
legibility model's own uncertainty (§5.3). Selecting which tracklets a human is shown is not
supervision; nothing the model says is recorded as a label anywhere.

## 2. Inventory — what is here, what is on the server, what is dead

`docs/SOCCERNET_DATA_INVENTORY.md` is dated 2026-07-29 and is **out of date in our favour**: it
records that we hold only the GSR *validation* split. `data/soccernet/gamestate-2024/README_SPLITS.md`
records the 2026-07-31 fetch of train + test, and disk confirms it.

| asset | where | size | state |
|---|---|---|---|
| **GSR train, 57 sequences, full labels** | `data/soccernet/gamestate-2024/` | part of 30.85 GB | **LOCAL**. 629,078 player/GK boxes over 1,224 tracklets. |
| **GSR-train GT crops, the S2 crop law** | `outputs/gsr/gt_crops/train/` | 20,067 JPEGs | **LOCAL**, and `tools/gsr_crops.py --demo` still asserts the published 20,067 / 1,343 / 57. |
| **GSR-train legibility scores** | `outputs/gsr/w3_annotation/legibility_train.parquet` | 20,067 rows | **BUILT THIS SESSION** (107 s GPU). Did not exist locally before. |
| **jersey-2023 train** | `data/soccernet/jersey-2023/train/` | 733,001 crops, 1,427 tracklets | **LOCAL**, complete, with `train_gt.json`. |
| **jersey-2023 legibility scores** | — | — | **NOT BUILT.** A 20,000-crop seeded sample was scored (§4.2); the full pass is 1 h 47 m of laptop GPU and is left to the training session. |
| shipped legibility ResNet34 | `~/jersey-number-pipeline/models/legibility_resnet34_soccer_20240215.pth` | 85 MB | **LOCAL** |
| arm-4t PARSeq (the incumbent reader) | `~/jersey-number-pipeline/models/parseq_v6_arm4t.ckpt` | 382 MB | **LOCAL** |
| **SoccerNet-v3** (the 106k digit labels) | `a100server1:~/data/soccernet-v3/` | **45 GB** | **SERVER ONLY.** 33 GB free on C: — it does not fit, and it is the single largest corpus source. §4.3. |
| S2's built corpora (`planned`, `plus_v3`, `plus_v3_torso`) | `a100server1:~/data/s2corpus/` | — | **SERVER ONLY**, and only the `corpus_stats.json` summaries are documented. The corpora themselves are not reproducible here without v3. |
| S2's build scripts (`v3_crops.py`, `legibility.py`, `build_corpus.py`, `torso_corpus.py`) | `a100server1:~/work/jersey/` | — | **SERVER ONLY.** Not vendored into this repo, then or now. |

**Dead paths checked, not assumed:** every `~/…` path in `CLUSTER_SESSION_S2.md` §6 was tested. The
two model checkpoints resolve on this laptop (they were copied down for S3); everything under
`~/data/` and `~/work/` does not exist here.

## 3. Label-provenance audit — the part the task asked for, and it changes the plan

### 3.1 GSR train: identity-carried, with zero exceptions

Every player/GK annotation of all 57 train sequences, grouped by `track_id`:

| | tracklets | boxes |
|---|---:|---:|
| all boxes numbered | **933** | 512,009 |
| all boxes un-numbered | **291** | 117,069 |
| **mixed (some frames numbered, some not)** | **0** | — |
| carrying two different numbers | **0** | — |
| total player/GK | 1,224 | 629,078 |

**There is no per-frame visibility information in GSR's jersey field.** The label is attached to the
identity and replicated to every box of it, exactly as SoccerNet-v3's `ID` field is
(`CLUSTER_SESSION_S2.md` §1.2 found the same by inspecting pixels). 81.4% of train player/GK boxes
carry a number, and a large share of those crops show the player's front, a shoulder, or a blur.
The 0.7 legibility gate is therefore not a nicety — it is the entire glyph-only guarantee, and it
discards **9,819 of 13,994** numbered GSR-train crops (70.2%).

### 3.2 jersey-2023: also identity-carried, but its abstention class is human and usable

`train_gt.json` is a flat `{tracklet_id: int}` map — **one integer per tracklet**, `-1` = illegible.
Median 532 crops per tracklet. So a jersey-2023 number is carried across ~532 crops from however
many the annotator actually read it in: the same hazard, and it gets the same 0.7 treatment.

The `-1` class is different in kind, and this is the useful finding:

| jersey-2023 train | tracklets | crops |
|---|---:|---:|
| numbered | 1,024 | 560,744 |
| **illegible (`-1`)** | **403** | **172,257** |

A `-1` is a human saying *no number was legible anywhere in this tracklet* — a stronger claim than
any per-crop label, and it is the only large human-sourced abstention set that exists without Sid.
Cross-checked against the shipped legibility model on the seeded sample: of the illegible-tracklet
crops, **0.51% score above 0.7**. The model and the human annotators agree on the negative class
**99.5%** of the time, on 4,722 independently-sampled crops. That is a validation of the 0.7 gate as
a glyph-only proxy that this project did not previously have, and it means the anti-teacher-imitation
rule can be honoured without waiting for Sid: **~171,000 human-sourced no-number crops are already on
this disk.**

### 3.3 The goalkeeper policy — the finding that resizes the W3 target

| split | player numbered | player un-numbered | **GK numbered** | GK un-numbered |
|---|---:|---:|---:|---:|
| GSR train | 933 | 212 | **0** | **79** |
| GSR validation | 966 | 179 | **0** | **77** |

`GSR_S3_READER.md` §7.8 noted in passing that "GSR labels no goalkeeper numbers". It is exact, it
holds on both splits, and it means the unnamed-tracklet pool the W3 plan targets is **two
populations, not one**:

* **212 train players (73%)** — a legibility/visibility failure, in principle recoverable by a
  better reader or a better gate;
* **79 train goalkeepers (27%)** — a *label* absence. No model can learn a number nobody wrote down.
  On DEV-20 the same policy caps the ceiling of every reader we will ever build.

This is why 79 of the 289 tier-A sheets are goalkeepers, and why they are worth Sid's time more than
any player sheet: they are supervision the benchmark structurally cannot contain.

## 4. The corpus — what was assembled, and what could not be

### 4.1 GSR train, assembled and validated against the server number

`python -m tools.gsr_w3_corpus --legibility train` (20,067 crops, **107 s**, RTX 3050) then
`--manifest`:

| | crops | tracklets |
|---|---:|---:|
| planned player/GK GT crops | 18,247 | 1,224 |
| — carrying a GT number | 13,994 | 933 |
| — **admitted, glyph-only (legibility > 0.7)** | **4,175** | **861** |
| — carried but illegible (rejected) | 9,819 | — |
| — no GT number (the unnamed pool -> Sid's queue) | 4,253 | 291 |
| distinct numbers admitted | **40** | |

| cross-check | S2 (server, 2026-08-05) | this session (laptop) |
|---|---|---|
| GSR-train crops passing legibility 0.7 | **4,175 of 13,994** | **4,175 of 13,994** |

**Exact to the crop**, on a different machine, through an independently written scorer, against a
number recorded in a server-side session eight days earlier. The 0.7 filter is reproducible and the
harness is the S2 harness.

**Leakage check, run inside `--manifest`:** intersection of the manifest's sequences with
`validation + test` = **NONE**. No dedup was needed — GSR crop names are
`{sequence}_{track:04d}_{frame}.jpg`, globally unique by construction, and 18,247 rows carry 18,247
distinct names.

Manifest schema (`outputs/gsr/w3_annotation/corpus_manifest.parquet`): `crop_path`, `label`,
`source`, `legibility`, `sequence`, `tracklet_id`, `frame`, `role`, `label_provenance`, `admitted`,
`reason`.

### 4.2 jersey-2023 — measured, not assembled

Scoring 733,001 crops is a 1 h 47 m GPU pass and is not prep work, so a **20,000-crop seeded sample**
(`seed=0`) was scored instead (179 s) to measure the pass rate rather than inherit it:

| | sample crops | pass @ 0.7 | projected over the full split |
|---|---:|---:|---:|
| numbered-tracklet crops | 15,278 | **0.2533** | **~142,000 of 560,744** (±~2,000, 1 s.e.) |
| illegible-tracklet crops | 4,722 | **0.0051** | ~880 of 172,257 |

S2 sampled 201,313 jersey-2023 crops and kept 52,135 = **0.2590**. Our 0.2533 on an independent
15,278-crop sample is within 1.6 standard errors — the two agree.

**The headroom this exposes is the practical point.** S2's `planned` corpus took **38,133**
jersey-2023 rows from a 201,313-crop sample. The full split holds **~142,000** admissible glyph-only
crops, i.e. **3.7x what the incumbent reader was trained on, from a source already on this disk, at
the cost of one 1 h 47 m legibility pass.** That is the single cheapest corpus expansion available
to W3 and it needs neither Sid nor the cluster.

### 4.3 SoccerNet-v3 — the extraction plan (NOT run; it cannot run here)

v3 is the source S2 measured as *the whole effect* (arm 2 -> arm 4 = **+0.3078** precision at
identical emit). It is also the only source that is not local.

| | |
|---|---|
| where | `a100server1:~/data/soccernet-v3/` (`Frames-v3.zip` per game + `Labels-v3.json`), **45 GB**, 400 labelled games |
| what to cut | the **106,591** person boxes carrying a digit-valued `ID`; 105,702 crops survive the 30 px floor |
| filter | shipped legibility ResNet34 at **0.7** -> **45,620 kept (43.2%)**, 88 distinct numbers (S2 §3.2, measured) |
| corpus contribution | 42,479 train + 4,239 val rows in `plus_v3`; 93.3% survive the pose-torso conversion |
| GPU time | ~35 min to fetch (already fetched, resident), legibility over 105,702 crops ≈ **15 min** on an A100, ~15 min more for torso RoIs. The extraction itself is CPU + disk. |
| scripts | `~/work/jersey/{v3_crops.py, legibility.py, build_corpus.py, torso_corpus.py}` — server-side, already written, already validated by S2 |
| peak extra disk | ~140 MB (S2 streams one game's zip at a time; this was a deliberate S2 design and it holds) |

**Laptop-vs-server recommendation: SERVER, and it is not a preference.** C: has **33 GB free**
against v3's 45 GB. The corpus cannot be assembled on this machine at all, and by extension the
retrain that consumes it should run where its largest source already lives — as S2's did. The
laptop's role in W3 is the two things it did this session: the GSR-train half of the corpus, and
Sid's queue.

**A second reason to prefer the server that is not about disk:** an arm-4t-comparable retrain needs
the *torso* geometry (S2 §3.4 measured a complete crossover — each reader collapses in the geometry
it was not trained on: incumbent 0.7022 on torso vs 0.0744 on full-body). Torso conversion is a
KeypointRCNN pass over the whole corpus. On ~190,000 crops that is hours on a 4 GB card.

## 5. Task B — Sid's annotation evening #1

### 5.1 The target, and why the queue is the size it is

The failure inventory the plan names is DEV-20's **260 of 870** tracklets that never get a readable
number (`GSR_V7_V3.md` §1, `docs/GSR_NEXT_PLAN.md` §2). DEV-20 is reference-only, so the queue is
built from the **trainable analogue on GSR train: the 291 player/GK tracklets carrying no GT number
at all** (§3.1) — 23.8% of train tracklets against DEV-20's 29.9%, the same failure mode on the side
of the split we are allowed to label.

289 of the 291 survive a 12-box minimum. **That is the entire population** — the task's "top
~300-500 by leverage" is not a sampling choice here, it is the whole trainable unnamed pool, and it
is smaller than 500. The queue is filled to 499 with a second, different tier (§5.3).

### 5.2 Tier A — 289 tracklets, "what number is this player?"

| | |
|---|---|
| tracklets | **289** (210 player, **79 goalkeeper**) |
| sequences covered | **57 of 57** |
| GT boxes behind the labels | **117,060** (median 409/tracklet, max 750) |
| of those, at readable box size (>= 89 px tall) | **38,909** |
| ranking | by readable-size box count, descending — a 750-frame tracklet filmed at 45 px is unlabelable *and* un-trainable, so raw length is the wrong leverage measure |

Each sheet shows **12 views: the tallest box in each of 12 equal time windows** of the tracklet,
drawn from the boxes at least 89 px tall. 89 px is measured, not chosen: it is the 10th percentile of
the height of GSR-train crops that pass legibility 0.7, and below 60 px the shipped model fires on
0.4% of crops.

| box height | crops | legibility pass rate |
|---|---:|---:|
| < 60 px | 556 | **0.004** |
| >= 60 px | 17,691 | 0.239 |
| >= 100 px | 9,565 | 0.353 |
| >= 150 px | 1,326 | 0.552 |

**The first version of this queue was wrong and the sheets caught it.** Ranking views by box height
alone put all 12 cells of `A0001` inside frames 706-742 of a 750-frame tracklet — the one stretch
where the player ran past the camera. A `none` verdict on that sheet would have meant "no number in
1.4 seconds", not "no number in this tracklet". Time-stratifying first is what makes the abstention
label defensible; 221 of the 289 tier-A sheets now have a **median** view clearing 89 px.

**The design is validated by the first sheet it produced.** `A0001` (`SNGS-169` track 4) is in the
unnamed pool — GSR gives it no number — and cells 8, 9 and 10 show **`28`** plainly. The premise that
the unnamed pool contains readable numbers the benchmark's own annotators did not record is not an
assumption; it is visible in the first item of the queue.

### 5.3 Tier B — 210 tracklets, "which of these 12 views show the number?"

The one thing GSR, jersey-2023 and SoccerNet-v3 all lack is **per-crop glyph visibility**, and it is
the supervision the *legibility model* needs. `CLUSTER_SESSION_S2.md` §4 records that arm 1 — the
legibility retrain — **was never run**, and `GSR_S3_READER.md` §7.2 records that the shipped 2024
classifier is still the binding emit gate: **12,814 of 67,224 DEV-20 crops (19.1%) pass it**, and
every density number in the campaign is conditioned on it.

| | |
|---|---|
| tracklets | 210, over 56 sequences, all with a known GT number |
| **per-crop labels offered** | **2,520** (12 cells x 210), positive and negative |
| ranking | **legibility-model uncertainty**: mean per-crop `abs(legibility - 0.5)`, ascending — the tracklets sitting closest to the gate's decision boundary |

Ranking by model uncertainty is model-derived **selection**, never a model-derived label. The model
decides which tracklets a human is shown; the human decides everything that is recorded. Stated here
because it is the one place the anti-teacher-imitation rule could be read as touched, and it is not:
no model output enters `sid_labels.parquet`.

`B0001` (`SNGS-163` track 4, GT `9`) is the intended shape: cell 1 shows the `9` unambiguously,
cells 7-12 are backs and shoulders with nothing readable on them. Those are exactly the rows a
legibility retrain has never had.

### 5.4 The propagation machinery — and a premise correction

The task asks the labels to feed the existing propagation machinery via the
`tools/wire_anchors.py` + `generator/anchor_wire.py` pattern. **They do not need to, and they should
not.**

`anchor_wire` solves the ManU-corpus problem: a close-up crop with a readable number has to be
*attached* to a ByteTrack fragment by an OSNet/PRTreID margin gate, because nothing tells us which
track the close-up belongs to. `wire_anchors.py` §caveats records what that costs — the margin gate
rejects most same-kit disambiguations, and attachment yield is "far below the anchor count".

**On GSR train, that whole problem is absent.** Sid labels a GT `track_id`, and the GT `track_id`
already enumerates every box of that identity. Propagation is exact, lossless and free; running a
ReID attachment gate over it could only *lose* rows. So `--ingest` is a direct adapter, and it
carries the one asymmetry that matters:

* a tier-A **number** is an identity property -> carried to all of the tracklet's boxes, then
  admitted only through the same 0.7 legibility gate the machine labels pass;
* a tier-A **`none`** is a statement about the 12 views he was shown -> labelled on **those 12 crops
  only**. We do not claim a number never appeared in a frame he was never shown.

Verified end-to-end on a synthetically-filled CSV before Sid is asked to spend an evening on it
(13,295 crop rows produced, every row's admission reason accounted for). Full-density crops come
back `admitted=False, reason="unscored (needs a full-density legibility pass)"` rather than being
silently rejected — the legibility pass so far covers only the 15-crop-per-tracklet re-ID crop set.

### 5.5 Expected yield and pace

| tier | rows | s/row | evening time | direct label yield |
|---|---:|---:|---|---|
| A | 289 | ~12 | **1 h 00 m** | one tracklet-level verdict each |
| B | 210 | ~18 | **1 h 03 m** | **2,520 per-crop glyph-visibility labels** |
| **total** | **499** | | **~2 h** | |

Tier-A crop yield, stated as a range because the read rate is the unknown Sid resolves:

* the 289 tracklets carry **38,909** readable-size boxes;
* if **r** is the fraction he can name, the corpus gains `r x 38,909` crops carrying a human-sourced
  number, of which the 0.7 gate admits roughly 24-35% at those box heights (§5.2 table);
* **r = 0.4** -> ~15,600 carried crops -> **~4,000-5,500 admitted glyph-only crops**, which
  **roughly doubles the GSR-train contribution to the corpus (4,175 today)**;
* the complement, `(1 - r) x 289` tracklets, yields `12 x (1 - r) x 289` ≈ **2,100 human abstention
  crops** at r = 0.4 — small in count, but the only crop-level human "no number visible here" labels
  the project will own, and the class the v7-V3 refutation says must not come from a model.

**These are projections and they are labelled as such.** The measured quantities are 38,909 readable
boxes, 2,520 tier-B cells offered, and the height-conditioned pass rates. `r` is not estimable from
here — that is what the evening measures.

## 6. Planned training registration (DRAFT, NOT YET REGISTERED)

> **This section is a DRAFT and registers nothing.** The real registration is written by the
> training session, before it trains, in its own results file. It is sketched here only so the
> orchestrator can price W3 and so the corpus above is built against a known target. Any number in
> this section may move when the real registration is written; none of it constrains the session
> that writes it.

**Gate ladder, following the S2 -> S3 pattern the campaign already ran end to end.**

**Rung 0 — corpus integrity (must pass before a single step of training).**
No `validation` or `test` sequence in any corpus row; every `(crop, number)` row either passes
legibility 0.7 or carries a human provenance string; zero rows whose no-number label has a model
origin. Mechanical, auditable from the manifest, no discretion.

**Rung 1 — component: per-crop precision at matched emit, on a held-out GSR-train slice.**
Hold out a fixed slice of train sequences (proposed: 8 of 57, chosen by a rule fixed before it is
applied) from the corpus. Score incumbent (arm-4t) vs candidate on that slice through the S2/V3
harness — same crops, same legibility gate, same pose torso RoI, only the PARSeq weights differ —
and compare **at the incumbent's own emit rate**, as S2 §3.5 did.

*Proposed bar: candidate precision > incumbent precision by a margin declared before the run.*
**Caveat that must be carried into the real registration:** arm 4t's own corpus contains GSR-train
rows and *which sequences* is not recorded locally, so the incumbent may have trained on the holdout.
That biases the baseline **upward**, making the comparison conservative in the right direction — but
it must be stated, not discovered later.

**Rung 2 — d at a precision floor** (`tools/ocr_density.py --sweep`), the `GSR_S3_READER.md` §4
gate. Reported at the 0.80 and 0.85 read-precision floors against the on-record `d = 0.4148 / 0.3872`.

**Rung 2 carries a known trap and the real registration must handle it explicitly.**
`GSR_S3_READER.md` §6 measured that the operating point the density gate *selects* scored **0.94
GS-HOTA worse** end-to-end than the incumbent's own rule on the same evidence (41.80 vs 42.74),
because under GS-HOTA a wrong number deletes a detection that a `null` would have let match. **Rung 2
must not select the shipped rule.** Both arms — candidate weights under the incumbent's rule, and
candidate weights under their own swept rule — go to rung 3, as S3 did.

**Rung 3 — DEV-20 end-to-end: `>= +1.0` GS-HOTA with `>= 12/20` sequences helped**, paired against a
control re-derived in the same session on the same machine, with the GS-DetA/GS-AssA decomposition
reported. This is the campaign's standard bar and the one S3 cleared at +3.66 / 18-of-20.

**Rung 4 — TEST-38: not in this session, at any outcome. W5 owns it**, inside the leave-one-out
ablation and the single frozen bundle. test-49 and the challenge split are not touched by W3 at all.

**Two prior negatives the real registration should carry as declared risks**, because both were
measured on this exact component: our Dirichlet fusion does **not** transfer to a head whose
abstention class is miscalibrated (W1 §2.7 — 88% of tracklets abstained), so any fusion arm must
check the candidate's `p(no number)` calibration before it is believed; and the legibility model is
**unchanged** in everything above, so a reader-only retrain inherits the 19.1% emit ceiling. If the
tier-B labels are used to retrain the gate as well, that is a **second** component and needs its own
rung 1 on its own held-out slice, not a shared one.

## 7. Negatives, limits, and what was NOT done

1. **No training, no scoring, no end-to-end number.** There is no GS-HOTA in this file and nothing
   here has been shown to move one. Every claim is a corpus count, a label-provenance fact or a
   projection explicitly marked as such.
2. **SoccerNet-v3 — the largest corpus source and the one S2 measured as "the whole effect" — is not
   local and cannot be made local** (45 GB against 33 GB free). §4.3 is a plan, not an assembly, and
   the W3 retrain is therefore a cluster job unless the source is dropped.
3. **jersey-2023's legibility scores were sampled, not computed.** 20,000 of 733,001 crops. The
   ~142,000-crop projection carries a ±2,000 sampling interval and nothing more; the full pass
   (1 h 47 m laptop GPU) was not spent because prep is not the place for it.
4. **No torso RoIs were built.** Every corpus row in the manifest is a *full-body* crop. S2 §3.4
   measured a complete geometry crossover (each reader collapses in the geometry it was not trained
   on), so the manifest is **not** directly trainable against the arm-4t chain until a KeypointRCNN
   pass converts it. That pass is hours on this card and belongs on the server.
5. **The `SNGS-065` GT error was found by eye, not by an audit.** One instance, spotted while
   checking that the contact sheets were usable. No systematic duplicate-box scan was run across the
   57 train sequences, so this is an existence proof and not a count. The `note` column in the
   labelling CSV is where Sid's evenings will surface more of them; a real audit is a separate,
   cheap, unrun job (all-pairs IoU between player/GK and referee tracks per frame).
6. **`r`, the tier-A read rate, is unmeasured**, and it is the quantity that decides whether W3's
   annotation half is worth the evenings. Two sheets were inspected: `A0001` yields a clear `28`,
   `A0150` yields an honest `none` (12 front/side views, no back). Two sheets estimate nothing.
7. **Tier-A `none` verdicts are bounded to 12 crops each** by construction (§5.4), which is correct
   but means the human abstention yield is ~2,100 crops, not ~70,000. The larger human abstention
   set available to W3 is jersey-2023's 403 illegible tracklets (§3.2), and it costs nothing.
8. **The 89 px view floor and the 0.7 legibility threshold are both inherited or measured, but
   neither was swept.** 0.7 is S2's shipped value, kept deliberately so the corpus is comparable;
   89 px is the 10th percentile of passing-crop heights, a defensible cut rather than an optimised
   one.
9. **No DEV-20, TEST-38, test-49 or challenge data was opened. No submission. No commit. No
   `METRICS_VERSION` change.** Targeted tests only: `tests/test_jersey_id.py`,
   `tests/test_ocr_density.py` — 15 passed.
10. **GPU spend: ~5 minutes total** (107 s GSR-train legibility, 179 s jersey-2023 sample). CPU:
    ~6 minutes for the queue builds and the label audits.

## 8. Files

**Repo (new, NOT committed):**

- `tools/gsr_w3_corpus.py` — the whole session: the legibility scorer, the glyph-only manifest
  builder with its leakage check, the annotation-queue builder (contact sheets + CSV), the
  `--ingest` adapter, and a `--demo` self-check covering the split guard, both answer-grammar
  parsers, the view picker (time stratification, readable-size floor, fallback) and the admission
  rule. `ruff` clean at 100 cols.
- `results/GSR_V8_W3_PREP.md` — this file.

**Artifacts (gitignored, `outputs/gsr/w3_annotation/`):**

- `legibility_train.parquet` — 20,067 GSR-train GT crops with their legibility score.
- `legibility_j2023_sample.parquet` — the 20,000-crop seeded jersey-2023 sample.
- `corpus_manifest.parquet` — 18,247 rows, the glyph-only corpus manifest.
- `sheets/` — **499 contact sheets** (289 tier A + 210 tier B), 52 MB.
- `labelling_manifest.csv` — the labelling queue: `queue_id, tier, sheet, sequence, tracklet_id,
  n_frames, n_readable_frames, role, gt_number, n_cells, median_view_h, max_view_h, cell_frames,
  label, note`.
- `HOWTO.md` — Sid's instructions: what to type, the propagation promise, the pace table.
- `sid_labels.parquet` — written by `--ingest` once the CSV comes back (the smoke-test copy was
  deleted).

## 9. Reproduce

```
python -m tools.gsr_w3_corpus --demo                    # self-check, CPU, instant
python -m tools.gsr_w3_corpus --legibility train        # GPU, 107 s
python -m tools.gsr_w3_corpus --j2023-sample 20000      # GPU, 179 s
python -m tools.gsr_w3_corpus --manifest                # CPU, ~20 s
python -m tools.gsr_w3_corpus --queue                   # CPU, ~2 m 40 s (resumable)
python -m tools.gsr_w3_corpus --ingest outputs/gsr/w3_annotation/labelling_manifest.csv
pytest tests/test_jersey_id.py tests/test_ocr_density.py
```

## 10. Addendum (2026-08-13) — full jersey-2023 legibility pass, interrupted, 21/28 shards local

`tools/gsr_w3_corpus.py` gained `--full-jersey23` (scores every jersey-2023 train crop of a
**numbered** tracklet — 560,744 crops, the population §4.2/§7.3 sampled — in resumable 20,000-crop
shards under `outputs/gsr/w3_annotation/j2023_full_shards/shard####.parquet`, skip-existing so a
rerun only fills gaps) and `--j2023-full-manifest` (glyph-only manifest over the completed pass, a
sibling of `corpus_manifest.parquet` for the training session to `pd.concat`). The illegible pool
(403 tracklets / 172,257 crops, §3.2) is intentionally **not** rescored by `--full-jersey23` — it is
already an admissible human-abstention class on its own label, per the glyph-only rule.

The pass was launched (harness-tracked background job, batch 128, RTX 3050) and ran cleanly at
~1.5-2 min/shard. **Killed by request at 21/28 shards** (Sid needed the laptop GPU immediately, not
a fault) — landed cleanly mid-shard-21, no partial `shard0021.parquet` was left on disk.

**Verified before writing this addendum:**
- No orphan process: `tasklist` / `Get-CimInstance Win32_Process` show no `python.exe` running
  `gsr_w3_corpus` after the kill. GPU memory 0 MiB used.
- `shard0000.parquet` .. `shard0020.parquet` (21 files, `outputs/gsr/w3_annotation/j2023_full_shards/`)
  all load with `pandas.read_parquet`, each exactly 20,000 rows, columns
  `tracklet, file, label, legibility` — **420,000 crops scored, 0 bad/truncated shards.**
  `shard0021.parquet` does not exist (nothing to delete).
- The manifest and the claims-file update were **deliberately not done this session** — the pass is
  incomplete (420,000 of 560,744 numbered crops, 75%) and neither `build_j2023_full_manifest()` nor
  a claim should be written against a partial count.

**To resume** (laptop, once the GPU is free again, or on the college cluster — copy
`data/soccernet/jersey-2023/train/` and `~/jersey-number-pipeline/models/legibility_resnet34_soccer_20240215.pth`,
same relative paths):

```
python -m tools.gsr_w3_corpus --full-jersey23      # resumes at shard 21/28 automatically
python -m tools.gsr_w3_corpus --j2023-full-manifest
```

Remaining: shards 21-27 (7 shards, ~140,000 crops), est. 12-18 min GPU at the rate observed above.

## 11. Addendum (2026-08-13) — pass FINISHED on the cluster, corpus deliverable complete

The laptop GPU stayed off-limits; the remainder ran on `a100server1` GPU 1 (A100-PCIE-40GB, 4 MiB
resident at launch, GPU 0 left to its other user).

**Shard arithmetic correction.** §10 says "21/28 shards"; the real partition is **29** shards —
560,744 = 28 x 20,000 + 744, so `shard0028.parquet` holds 744 rows. Remaining work was shards
**21-28 (8 shards, 160,744 crops)**, not 7.

**Portability verdict: the shards are machine-independent, so only the missing 8 were recomputed.**
`_j2023_numbered_rows()` keys every row by `(tracklet, file)` — relative ids, never an absolute path
— and builds the list by `sorted(gt.items())` then `sorted(dir.iterdir())`, i.e. a deterministic
enumeration of the same dataset. Verified rather than assumed: the same function run on both machines
returns **560,744 rows with identical MD5 `5b77db76f8de97da53cbe80802a4b6dd`** over the
`tracklet/file|label` stream (`train_gt.json` also md5-identical, `8a1913d1...`), first row
`('0','0_1.jpg',10)`, last `('999','999_750.jpg',55)`, and the local `shard0020.parquet`'s own first
and last rows (`627/627_550.jpg`, `663/663_670.jpg`) match the server's shard-20 boundary exactly.
Route taken: **upload the 21 local shards (3.3 MB) to the server so skip-existing resume applies,
score shards 21-28 there, copy those 8 parquets back.** No code change, no re-scoring of the 420,000
crops the laptop already paid for.

**Server spend.** Sync: `tools/gsr_w3_corpus.py` (32,520 B) only — `tools/gsr_crops.py` and
`generator/jersey_id.py` were already byte-identical server-side (md5 `90a0cdcc...`, `a8edbea8...`),
the legibility weights were already at `~/jersey-number-pipeline/models/` (symlink to
`~/models/koshkina/`), and `~/data/jersey-2023` was linked in as `data/soccernet/jersey-2023` so the
tool's relative paths resolve. Plus the 21 shards (3.3 MB) and a 30-line enumeration-hash probe.
Run: `~/run_w3_j23.sh` under `tmux new -d -s w3j23` (no bare nohup), env `gsr`
(torch 2.5.1+cu121), `CUDA_VISIBLE_DEVICES=1`, batch 128, 1,424 MiB peak.
**Wall clock 5 min 44 s for 160,744 crops (~43 s/shard, ~2.2x the laptop's rate);** session killed
and GPU 1 released afterwards.

**Final measurement (all 29 shards, merged and rebuilt locally, CPU only).** The server's merge and
the local merge agree to the crop:

| population | crops | tracklets | provenance |
|---|---:|---:|---|
| **admissible numbered** (legibility > 0.7) | **140,278** | 926 | identity-carried GT, glyph-gated |
| model-rejected numbered (leg <= 0.7) | 420,466 | — | *rejected*, NOT an abstention label |
| human-abstention illegible pool (label -1) | 172,257 | 403 | human tracklet verdict, unscored |

* 560,744 numbered crops scored, **0 decode failures, 0 unscored**; 560,744 + 172,257 = 733,001 =
  the whole jersey-2023 train split, so every crop is accounted for in exactly one population.
* **Pass rate 0.2502** against the 20,000-crop sample estimate **0.2533** (n = 15,278 numbered
  crops in that sample) — the sample was optimistic by **0.31 pp / 1.2%**, and §7.3's "~142,000-crop
  projection, ±2,000" lands 1,722 crops high, inside its own stated interval.
* 926 of 1,024 numbered tracklets keep at least one admitted crop; **98 lose every crop** (the gate
  says no view of that tracklet shows its number). Admitted crops per surviving tracklet:
  min 1, p25 59, median 121, p75 217, max 702, mean 151.5; 909 tracklets have >= 4 and 871 have
  >= 15, so the standard 4-per-id / 15-per-tracklet sampling laws are satisfiable on almost all of it.
* Per-number distribution: **all 44 distinct numbers in the numbered pool survive the gate**, but the
  tail is thin — median 2,313 crops per number, max 14,286 (number 10), min 5; **7 numbers have < 100
  admitted crops** and **9 are backed by a single tracklet**, so those classes are identity-shortcut
  risk, not digit supervision. 30.5% of admitted crops carry a 1-digit label (42,766) and 69.5% a
  2-digit one (97,512).
* The rejected population is explicitly **not** relabelled as no-number: v7-V3's refutation
  (`v7-v3-002`) is that a head trained on `leg <= 0.7` learns its teacher. The abstention class stays
  the 172,257 human-marked crops.

`jersey23_full_manifest.parquet` (560,744 rows) carries the same 11 columns as
`corpus_manifest.parquet` — `crop_path, label, source, legibility, sequence, tracklet_id, frame,
role, label_provenance, admitted, reason` — and is deliberately a **sibling**, not a merge: the two
sources have different identity spaces (GSR `sequence+track_id` vs a jersey-2023 tracklet folder),
so the concat is the training session's call. Split hygiene is unchanged: jersey-2023 train and GSR
train only.

**Not done / unchanged:** no torso RoIs (§7.4 still stands — these are full-body crops); no
training, no GS-HOTA; no `METRICS_VERSION` change; no commit; `STATUS.md` untouched. Laptop GPU never
used this session.

**Reproduce (from a state with all 29 shards on disk, CPU only — `--full-jersey23` re-scores nothing
when every shard exists):**

```
python -m tools.gsr_w3_corpus --full-jersey23 --j2023-full-manifest   # CPU, ~60 s
```
