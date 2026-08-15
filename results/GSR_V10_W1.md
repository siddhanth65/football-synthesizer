# v10 W1 - can an open-weights VLM read the jersey numbers our chain calls unreadable?

Campaign context (Sid, 2026-08-15): the board leader `betterdays` posts **68.3 GS-HOTA with
GS-DetA 55.86**, against our 41.28 - identity-gated detection far beyond what our substrate's
oracles said was reachable. The largest priced bucket in the v9-W3 DetA census is **jersey
(+24.74 oracle)** and every conventional attack on it failed. W1 tries the one direction never
tried at scale: **an open-weights VLM reading jersey numbers**, scored against the only
per-tracklet human ground truth that exists - Sid's 499-tracklet W3 annotation.

Everything below section 1 was written **after** the runs; section 1 was written and committed to
disk **before any model weight was downloaded or any inference was run**.

---

## 1. Registration (written before inference)

### 1.1 The question

Can a modern open-weights VLM read jersey numbers on tracklets our shipped PARSeq + legibility
chain calls unreadable, at **precision >= 0.90** and coverage materially above zero, on the
human-audited pool? The competing hypothesis is kb `v8-w3-007`: 94.8% of the unnamed-tracklet pool
is *genuinely* unreadable - but that verdict was measured at **contact-sheet resolution, by a human,
single-pass**. A VLM sees the same 12 views with arbitrary zoom and no fatigue, so the hypothesis is
testable, not assumed.

### 1.2 Model, license, weights

| | |
|---|---|
| **Primary** | `Qwen/Qwen3-VL-8B-Instruct`, **Apache-2.0** (research use permitted), 17.55 GB repo, 4 bf16 safetensors shards |
| **Fallback arm** (runs only if the primary FAILS a registered bar) | `Qwen/Qwen2.5-VL-7B-Instruct`, **Apache-2.0**, 16.60 GB repo |
| Total download plan | **<= 34.2 GB** to `~/.cache/huggingface` on `a100server1` (quota 168 GB / 500 GB before the session). Open weights, not NDA footage. |
| Serving | HF `transformers` 5.14.1 + torch 2.5.1+cu121, bf16, single A100-40GB (**GPU 1 only**), greedy decode (`do_sample=False`, `max_new_tokens=24`). No vLLM (would drag its own torch into the `gsr` env). |
| Prior to beat | the v7 `Qwen2-VL-2B` trial, kb `ident-020`: **0.265 precision** on chain-abstain crops. |

Why an 8B primary and not the 2B that already failed: `ident-020`'s failure was a small-model
failure, and Broadcast2Pitch's 61.48 row replaced its OCR head with a fine-tuned LLaMA-3.2-Vision,
so the class of model is the variable under test.

### 1.3 Input regimes (three tracklet regimes, fixed; one crop-level instrument)

All three tracklet regimes see **exactly the evidence the human saw** - the same 12 stratified
closest views (`cell_frames` in the queue manifest) - so the comparison isolates *presentation*, not
information.

| regime | what the model gets |
|---|---|
| **sheet** | the byte-identical annotation contact sheet Sid read (4x3, 170x250 cells, each crop `INTER_CUBIC`-scaled to fit), one image |
| **multi12** | the 12 cell crops as **12 separate images**, each Lanczos-enlarged to 256 px tall (cap 4x) |
| **zoom1** | the **tallest** of the 12 cells alone, Lanczos-enlarged to 512 px tall (cap 4x) - the registered "crop + upscale" arm, at 4x rather than the brief's 2x because 4x is the strictly stronger zoom test |
| *(instrument)* **percrop** | every tier-B cell alone, same enlargement as `zoom1` - 2,520 single-crop reads for the tier-B visibility read |

Processor bounds: `min_pixels = 64*28*28`, `max_pixels = 1024*28*28`.

### 1.4 Prompts (verbatim, `tools/gsr_v10_w1.py`)

Tracklet prompt - the opening sentence varies by regime, everything after it is byte-identical:

```
{intro}

Read the jersey number printed on this player's shirt.

Rules:
- Give a number ONLY if you can actually see the printed digits in at least one view.
- Jersey numbers are 1 to 99. Never infer a number from the player's role, position, kit colour,
  body shape or anything other than digits you can see.
- If no digits are legible in any view, answer null. Answering null is correct and expected: in
  this dataset most players' numbers are not visible.

Reply with one line of JSON and nothing else:
{"number": <integer 1-99, or null>}
```

`{intro}` is, per regime:

* **sheet**: `This image is a 4x3 contact sheet of 12 cropped views of ONE football (soccer)
  player, taken at 12 different moments of the same 30-second broadcast clip. The small yellow
  digits 1-12 in the top-left corner of each cell are cell indices, NOT jersey numbers.`
* **multi12**: `These images are 12 cropped views of ONE football (soccer) player, taken at 12
  different moments of the same 30-second broadcast clip.`
* **zoom1**: `This image is the closest available cropped view of ONE football (soccer) player from
  a 30-second broadcast clip, enlarged.`

Per-crop prompt (`percrop`) is the same text with `This image is one cropped view of a football
(soccer) player from a broadcast match, enlarged.` and "in THIS view" / "not legible here".

Parsing (`parse_read`): a 1-99 integer from the JSON field, else an explicit null/refusal, else the
first bare integer; **anything outside 1-99 is an abstain, never a clamp**. Unit-tested.

### 1.5 Evaluation pool and the registered bars

Sid's tier A holds **289** GSR-train player/GK tracklets SoccerNet left unnumbered: **15** carry a
human number, **274** a human `none`, **0** `unsure`. Two of the 15 (`A0085` = 30, `A0103` = 36)
carry a stray `u` in the note column; kb `v8-w3-012` reads those as a mis-keyed `unsure`, so the
**confident pool is 13 numbers**. Both reads are reported; the bars are judged on the 13.

**PASS requires both, on the primary model, on at least one registered regime:**

1. **precision >= 0.90** on the 13 human-numbered tracklets (correct reads / reads emitted there);
2. **false-positive rate <= 5%** on the 274 human-confirmed-`none` tracklets (any number emitted is
   a false positive - under GS-DetA an invented number costs exactly what a wrong one costs).

Coverage "materially above zero" is reported as recall on the 13, with no bar attached: a
0-emission run trivially satisfies bar 1 and is reported as a degenerate PASS, not a win.

### 1.6 Secondary reads, all from the same runs

* **tier-B tracklet** (210 tracklets whose GT number SoccerNet *does* know): precision and recall.
  This is the positive control - if the VLM cannot read numbers the benchmark itself asserts are
  there, a tier-A abstention means nothing.
* **tier-B per-crop** (2,520 cells, 771 human-visible / 1,749 human-hidden, kb `v8-w3-009`):
  agreement with `gt_number` on visible cells, FP rate on hidden cells.
* **PARSeq agreement**: on the tier-B cells the shipped incumbent reader
  (`parseq_v6_arm4t.ckpt`) names confidently, how often the VLM agrees. Disagreement is a
  calibration signal, not a verdict.
* **mean token log-probability** is stored for every read, so a precision/coverage curve exists
  even if the raw operating point fails.

### 1.7 Discipline, budget, stop rule

* Sid's 499 labels are **GSR-train** tracklets; no training happens here, so using them as
  evaluation GT is legitimate. `load_queue` hard-refuses any sequence outside `~/train57.txt`.
  **DEV-20, TEST-38, test-49 and challenge are not opened at any point this session.**
* All GPU on **GPU 1** of `a100server1`, occupancy checked immediately before each launch, every
  job under `tmux`. Budget ~10 GPU-h; anything not reached is reported as a budget stop.
* Order of work fixed now so a budget stop is not a selection: cells (CPU) -> `zoom1` -> `sheet` ->
  `multi12` -> `percrop` -> PARSeq agreement -> fallback arm (only on FAIL).
* If the bars fail, the report states plainly whether kb `v8-w3-007`'s 94.8%-genuine-absence
  finding is **confirmed at VLM scale**. If they pass, the report sketches the v10-W2 DEV
  integration and runs none of it.

---

## 2. Verdict against the registered bars

> **FAIL on bar 1 by a single read; PASS on bar 2.** The best registered regime (`multi12`) scores
> **8 correct of 9 emitted = 0.889 precision** on the 13 human-numbered tracklets (bar 0.90) and
> **10 false positives of 274 = 3.65%** on the human-confirmed-`none` pool (bar <= 5%).
> The pool cannot resolve the bar: the 95% Wilson interval on 8/9 is **[0.565, 0.980]**.
>
> **And the VLM does not beat the shipped chain.** Handed the identical 12 views, the incumbent
> PARSeq + legibility chain scores **exactly the same 8 of 9** on tier A, and **beats every VLM
> regime on tier B** - 0.9848 precision at 0.9238 recall against the best VLM's 0.9579 / 0.8667.
> Per crop the gap is not close: chain-only-correct 324 vs VLM-only-correct 54 over 2,520 paired
> cells, exact McNemar **p = 4.8e-48**.

The registered premise ("identity evidence can be scaled past our PARSeq + legibility stack, and a
VLM is the way") is **not supported on this pool**. What the session did produce is three harder
results: the genuine-absence hypothesis is confirmed by a second, independent reader; the incumbent
reader is far stronger at tracklet level than the DEV coverage numbers implied; and reader
*agreement* is a perfect-precision gate.

---

## 3. The confusion table on the human-audited pool

Tier A = the 289 GSR-train tracklets SoccerNet left unnumbered. `precision` is over reads emitted on
the 13 confident human numbers; `FP` is any number emitted on the 274 human-confirmed-`none`
tracklets. Wilson 95% intervals in brackets. All reads greedy, no threshold.

| reader | emit / 13 | correct | **precision** | recall of 13 | **FP / 274** | FP rate |
|---|---|---|---|---|---|---|
| VLM `zoom1` | 5 | 4 | 0.800 [0.376, 0.964] | 0.308 | 8 | 2.92% |
| VLM `sheet` (human-parity view) | 7 | 6 | 0.857 [0.487, 0.974] | 0.462 | 4 | **1.46%** |
| **VLM `multi12`** (best) | 9 | 8 | **0.889** [0.565, 0.980] | 0.615 | 10 | 3.65% |
| VLM `multi12` on torso RoIs (post-hoc) | 7 | 6 | 0.857 [0.487, 0.974] | 0.462 | 10 | 3.65% |
| **shipped chain**, same 12 views | 9 | 8 | **0.889** [0.565, 0.980] | 0.615 | 14 | **5.11%** |
| human (Sid, W3) | - | 13 by definition | - | 1.000 | 0 by definition | - |

The 15-number read (including the two stray-`u` rows `A0085`/`A0103`) changes nothing: no reader
emits on either row, so precision is identical and recall falls to 8/15 = 0.533.

**Both readers get the same tracklets right and wrong, and they are not the same misses.** Side by
side on the 13 (`chain` / `multi12`):

```
A0001 29 -> 28 / 29     A0027  1 -> --- / ---    A0034 36 -> 36 /  6
A0054  1 ->  1 /  1     A0061 18 -> --- /  18    A0077  1 ->  1 /  1
A0082  9 ->  9 / ---    A0128  1 ->  1 /  1      A0144  1 ->  1 /  1
A0146 17 -> 17 / 17     A0177 14 -> 14 / 14      A0192  1 -> --- / ---
A0204  1 -> --- / ---
```

Each reader's single error is the other's success (`A0001`: chain reads 28, VLM 29; `A0034`: chain
reads 36, VLM 6), and each covers one tracklet the other abstains on. That is the ensemble signal
in section 6.

**Caveat carried in the open:** 7 of the 13 human numbers are `1` and 6 of those are goalkeepers.
Every reader has a GK-`1` prior. Excluding all `read == 1` rows, `multi12` scores 4/5 and the chain
4/5 - the same conclusion on a smaller pool, so the prior is not what drives the verdict.

---

## 4. Tier B - the positive control, where a number provably exists

210 tracklets whose number SoccerNet itself annotates, shown through the same 12 views.

| reader | emit / 210 | correct | precision | **recall** |
|---|---|---|---|---|
| VLM `zoom1` | 71 | 55 | 0.775 | 0.262 |
| VLM `sheet` | 174 | 166 | 0.954 | 0.790 |
| VLM `multi12` | 190 | 182 | 0.958 | 0.867 [0.814, 0.906] |
| VLM `multi12` on torso RoIs | 171 | 157 | 0.918 | 0.748 |
| **shipped chain** | 197 | 194 | **0.985** | **0.924** [0.880, 0.953] |

Paired over the 210: chain-only correct **14**, VLM-only correct **2**, both 180, neither 14. The
union is 196/210 = 0.933, i.e. the VLM adds **two tracklets** to a reader that already has 194.

**The single most campaign-relevant number in this session is the chain's 0.9238.** The v9-W3
census priced jersey coverage on the assumption that our reader cannot name most tracklets (309 of
478 DEV bundle tracklets carry no confident read). On GT tracklets, given twelve stratified
closest views, the same reader names **92.4% of numbered tracklets correctly**. The coverage loss
therefore lives in **which crops reach the reader** - tracker fragmentation, box quality, view
selection - not in the reader's ability to read.

### 4.1 Per-crop, the same 2,520 cells

| pool | shipped chain emit / correct / prec / recall | VLM `percrop` emit / correct / prec / recall |
|---|---|---|
| all 2,520 | 990 / 954 / 0.964 / 0.379 | 920 / 684 / 0.743 / 0.271 |
| human-**visible** 771 | 716 / 704 / **0.983** / **0.913** | 699 / 602 / 0.861 / 0.781 |
| human-**hidden** 1,749 | 274 / 250 / **0.912** / 0.143 | 221 / 82 / 0.371 / 0.047 |

Paired per-crop correctness over 2,520: **chain-only 324, VLM-only 54, exact McNemar p = 4.8e-48**.

**Both readers beat single-view human reading, and the chain beats it by 3x.** On the 1,749 cells
where Sid judged the digits illegible, the chain returns the *correct* number on **250** (14.3% of
them, at 0.912 precision) and the VLM on 82 (4.7%). Sid's per-cell visibility labels are therefore
a lower bound on what is machine-readable, exactly as kb `v8-w3-010` suspected from the other side.

### 4.2 Is the VLM's deficit just localization? No.

The chain feeds PARSeq a pose torso RoI; the registered VLM regimes see the whole body. Handing the
VLM the **identical torso RoIs** (post-hoc arm, 5,319 of 5,988 cells have one) makes it **worse**,
not better: tier-B tracklet recall 0.748 (from 0.867), per-crop recall 0.201 (from 0.271). The VLM
uses whole-body context; a general 8B VLM is simply a weaker digit reader than a jersey-fine-tuned
STR specialist, and no amount of cropping closes it.

---

## 5. The 94.8%-genuine-absence hypothesis: CONFIRMED

kb `v8-w3-007` reported that a human reading 12 stratified closest views recovered a number on only
15 of 289 unnamed tracklets (5.2%) and returned `none` on 274 (94.8%). Two independent machine
readers now re-read the identical views:

* the VLM emits a number on **10 of 274** (3.65%) of the human-`none` tracklets;
* the shipped chain emits on **14 of 274** (5.11%);
* they **both** emit on 5 of them and **agree on the number** on **2** - `A0026` (a goalkeeper, both
  say `1`, which is precisely the prior both readers carry and therefore weak evidence) and
  **`A0057` (SNGS-100 track 18, a player, where all three reads - chain, `sheet`, `multi12` - say
  33)**;
* per crop, the chain's legibility gate passes and PARSeq emits on **68 of 3,468** tier-A cells
  (2.0%), against **990 of 2,520** (39.3%) on tier-B cells from the same builder.

So the correction to 94.8% is at most a fraction of a percentage point: **one** of 274 `none`
tracklets carries strong independent evidence of a real, human-missed number (0.36%), and one more
is a GK-`1` coin flip. **The unnamed pool of SoccerNet-GSR train is genuinely unnumbered**, and the
v9-W3 lever table's judgement that jersey **over-naming** (+1.16 DetA oracle) is unreachable stands
- reinforced now by a reader that had never seen the data.

---

## 6. Agreement is a perfect-precision gate (the one shippable idea in here)

Requiring the two independent readers to emit the **same** number:

| pool | emit | correct | precision | recall |
|---|---|---|---|---|
| tier B (210 numbered tracklets) | 180 | **180** | **1.0000** | 0.857 |
| tier A, 13 human numbers | 6 | 6 | 1.0000 | 0.462 |
| tier A, 274 human `none` | 2 | - | - | FP rate **0.73%** |

180 of 180 correct is a stronger precision figure than either reader alone produces at any operating
point, at 86% of the chain's own recall. Under GS-DetA a wrong number costs exactly what `null`
costs, so a rule that trades 6.7 points of tracklet recall for the *elimination* of named-wrong is
DetA-neutral at worst; where it earns is the **identity solver's roster**, whose slot assignments
are corrupted by a single confident wrong number. Not a claim about DetA - a v10-W2 hypothesis with
a measured prior.

**Not registered, so not a PASS:** a post-hoc threshold on the VLM's own mean token log-probability
would clear both bars (`mean_logprob >= -0.1`: 8 of 8 correct = 1.000 precision on the numbered
pool, FP 10/274 = 3.65%; at `>= -0.005`, 8/8 with FP 6/274 = 2.19%). The threshold was chosen after
seeing the labels. It is reported as a curve, not as a result.

---

## 7. Cost, and the state the cluster was left in

| item | measured |
|---|---|
| weights downloaded | `Qwen/Qwen3-VL-8B-Instruct`, **17.0 GiB on disk** (4 bf16 shards + tokenizer/processor), 0 failed files. The fallback `Qwen2.5-VL-7B-Instruct` was **not** downloaded (see below) |
| cell cutting (CPU) | 5,988 cells from 5,988 GT boxes, **0 missing**, 94.6 s |
| VLM inference, registered regimes | `zoom1` 164.5 s (499 items) / `sheet` 271.4 s (499) / `multi12` 393.3 s (499) / `percrop` 168.1 s (2,520) |
| VLM inference, post-hoc torso arms | `percrop-torso` 122.4 s (2,366) / `multi12-torso` 153.9 s (492) |
| shipped-chain re-run | legibility 5,988 crops + pose torso (5,319 RoIs) + PARSeq, twice (tier B, then all cells) |
| **total GPU-h, GPU 1** | **~0.48** of a ~10 GPU-h budget (28.5 min wall, all under `tmux`) |
| session wall clock | ~1 h 5 m including the 6.5-minute weight fetch |
| end state | GPU 1 at **4 MiB**, no tmux sessions, quota **185 GB / 500 GB** (+17 GB for the weights) |

**Occupancy note, reported not hidden:** for roughly the first four minutes of the `zoom1`/`sheet`
runs GPU 1 also carried a *sibling session's* six-process `tools.gsr_v10_w2_calib` job (17.6 GiB);
peak was 36.4 GiB of 40.9 GiB and neither job OOMed. GPU 0 (another user's long-running `app.py`)
was never touched.

**Registered fallback arm not run, and why that is not a budget stop:** the fallback
(`Qwen2.5-VL-7B-Instruct`) was registered to fire on a FAIL, to separate "this model" from "VLMs".
It is not informative here, because the primary did **not** fail by being a weak reader - it matched
the incumbent exactly on tier A and reached 0.958 precision on tier B. An older, smaller sibling of
the same family cannot overturn the finding that the incumbent is *better*; the 2.7 GPU-h it would
cost buys nothing the v10-W2 questions do not need more. Stated as a deliberate deviation, not an
omission.

**Two infrastructure facts worth carrying forward.** (1) `a100server1` resolves AAAA records for
`huggingface.co` but has no IPv6 route, so `hf download` / any `requests`-based fetch **hangs
silently**; `curl -L` (happy-eyeballs) works at ~48 MB/s. Weights were fetched with a 12-line curl
loop and loaded with `HF_HUB_OFFLINE=1`. (2) `transformers` 5.14.1 in the `gsr` env serves
Qwen3-VL out of the box on torch 2.5.1+cu121.

---

## 8. What this changes, and the honest surprises

1. **The v10 opening premise is wrong in its specifics.** `betterdays`' DetA 55.86 is not explained
   by "a VLM reads jerseys our chain cannot" - our chain, on GT tracklets with good views, already
   names 92.4% of numbered tracklets at 0.985 precision. Whatever they do, it is not simply a
   stronger jersey OCR.
2. **The jersey bucket's remaining value is upstream of the reader.** v9-W3 priced jersey coverage
   at +16.36 DetA oracle and blamed the reader. This session's tier-B control says the reader is
   fine when it is given twelve stratified closest views of one clean identity. The loss is in
   getting those views: tracker fragmentation, ID switches, box quality, and the per-crop gate that
   admits 27% unreadable crops (kb `v8-w3-010`). **v10-W2 should attack view selection, not
   reading.**
3. **Sid's `none` labels are vindicated and his per-cell `hidden` labels are not.** The tracklet-
   level `none` verdicts survive two independent readers (one genuine miss in 274). The per-cell
   visibility labels are a *lower bound*: the shipped chain correctly reads 250 of the 1,749 cells
   he marked unreadable, at 0.912 precision.
4. **Surprise, measured:** giving a VLM a tighter crop makes it worse. Torso RoIs - the geometry
   that makes PARSeq work - cost `multi12` 12 points of tier-B recall. Whole-body context is load
   bearing for a general VLM.
5. **Surprise, measured:** the two readers' errors are almost disjoint, and their agreement is a
   1.000-precision gate over 180 tracklets. That is the only artifact of this session with a path
   to the board.

### 8.1 v10-W2 design sketch (proposed, nothing run)

The bars failed, so the registered "if PASS, sketch DEV integration" branch does not fire. What the
numbers *do* justify proposing:

* **Instrument first (0 GPU):** rebuild the v9-W3 coverage census on DEV-20 asking not "did the
  reader emit?" but "how many stratified closest views of >= 89 px did each *predicted* tracklet
  ever get?". If our tracklets rarely accumulate 12 good views, the tier-B 0.924 is unreachable by
  construction and that is the number to fix.
* **Then view selection (cheap GPU):** feed the incumbent reader the same `_pick_views` law used to
  build Sid's sheets (tallest box per twelfth of the track) instead of the current per-crop gate,
  and re-measure tracklet naming rate on DEV-20 against the frozen v9 arm.
* **Agreement gate, only if a second reader is free:** the VLM costs 0.79 s per tracklet at
  `multi12`; over ~478 DEV tracklets that is 6 minutes. Worth measuring *after* view selection,
  scored on DEV-20 GS-DetA with the standard flags-on/flags-off pairing.
* Explicitly **not** proposed: fine-tuning a VLM on jersey crops. The measured gap is 324:54 per
  crop against a specialist that already exists and is already ours.
