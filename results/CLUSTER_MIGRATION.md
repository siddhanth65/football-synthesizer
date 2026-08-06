# Eval-pipeline migration to a100server1 (2026-08-06)

The CPU eval/scoring half of the GSR campaign now runs server-side. This document is the machine
map, the sync flow, and the verification that the move did not change any number.

**Headline: the migration is VERIFIED.** The frozen EIoU TEST-38 arm, re-derived end to end on the
server through the full CPU harness and the official scorer, reproduces the on-record
`results/GSR_EIOU.md` §7 numbers to **GS-HOTA 39.5403 vs 39.5403 (delta -0.0000292)**, with the
control at **37.0344 vs 37.0344 (delta -0.0000068)**. Every integer in the chain is identical and
the pitch geometry is bit-identical. §4.

**One real defect was caught by the move:** the `sn-trackeval 0.4.0` already installed in env `gsr`
was **not** commit `9c25232` — it was missing a line from the GS-HOTA dataset class. Had the gate
not been run, every server-side score would have come from a different scorer build than every
number on record. §3.2.

---

## 1. Machine map

| Role | Machine | Location | Notes |
|---|---|---|---|
| Canonical git | GitHub `siddhanth65/football-synthesizer` (private) | remote | Laptop is the only pusher |
| Working tree / authoring | laptop `c:\Users\siddh_ygv5bws\football-synthesizer` | laptop | All edits and commits happen here |
| GPU training | a100server1 GPU 1 | `~/runs`, `~/src` | PRTreID/detector fine-tunes; GPU 0 is another user's, never touched |
| GPU inference (embeds, box cache) | a100server1 GPU 1 | `~/football-synthesizer/outputs/gsr` | Cold-cache only; the caches below make eval CPU-pure |
| **CPU eval / scoring** | **a100server1 (128 cores, no GPU)** | **`~/fs-frozen`, `~/football-synthesizer`** | **New this session** |
| CPU eval / scoring (fallback) | laptop | as before | Unchanged; still authoritative for on-record numbers |
| Scorer | a100server1 `~/sn-trackeval-9c25232` | PYTHONPATH-injected | Pinned commit, byte-verified against the laptop (§3.2) |
| Dataset | a100server1 `~/data/gamestate-2024` (32 GB) | symlinked into the repo | 166 entries, mirrors the laptop exactly |
| Packaging / submissions | laptop | `results/gsr_submission` | Not moved |

## 2. Sync flow (and its one limitation)

```
laptop  --git push-->  GitHub origin (canonical)
laptop  --tar|ssh-->   a100server1:~/football-synthesizer   (code + .git + results + caches)
server  --scp/tar-->   laptop                                (logs, result JSONs, checkpoints)
```

- **The server has no GitHub credentials and no network path to origin.** It cannot `git pull`.
  The server clone is a *snapshot pushed from the laptop*, refreshed by re-running the tar stream
  in §3.1. `.git` came across intact (HEAD `31be41a`, full history), so `git log`, `git show`,
  `git worktree` and `git diff` all work server-side — which is what made the frozen-commit
  checkout in §4.1 possible. Only fetch/push are unavailable.
- **The laptop has no `rsync`** (confirmed again this session; `results/CLUSTER_SESSION1.md` §3
  found the same). All transfers are `tar -cf - ... | ssh ... tar -xf -`. The server *does* have
  `rsync`, so server->laptop pulls can use it if a laptop-side daemon is ever wanted; it is not.
- **`core.autocrlf` must be `true` on the server clone.** The laptop checkout carries CRLF; without
  the matching setting `git status` on the server reports **413** phantom modifications. With it,
  status shows the 37 genuinely-uncommitted entries. Already set.
- Refresh command (code only, no caches):

```
tar -cf - --exclude=./outputs --exclude=./data --exclude=./matches --exclude='*.pdf' \
  --exclude='*.mp4' --exclude=node_modules --exclude=__pycache__ --exclude='*.pyc' . \
  | ssh siddhanth23519@192.168.3.19 'tar -xf - -C ~/football-synthesizer'
```

## 3. What moved

### 3.1 Sizes (measured, not estimated)

| Item | Size | Shipped? |
|---|---|---|
| Repo tree incl. `.git`, excl. outputs/data/matches/PDFs/MP4s | 636 MB (`.git` 146 MB) | Yes |
| `outputs/gsr` **total on laptop** | **33 GB** | Partially — 3.7 GB |
| `outputs/gsr_test` **total on laptop** | **5.6 GB** | Partially — 208 MB |
| `~/fs-frozen` (git worktree at the frozen commit) | 114 MB | Created server-side |
| `~/sn-trackeval-9c25232` (pinned scorer) | 1.6 MB | Installed server-side |
| `~/verify_t38` (this session's evidence) | 461 MB | Created server-side |
| **Total added to quota** | **~5.2 GB** | |

**Quota: 160 GB -> 166 GB of 500 GB. Headroom 334 GB.**

**The task brief estimated 15-25 GB for the two subtrees; the real total is 38.6 GB, and shipping
all of it would have been waste.** Of `outputs/gsr`'s 33 GB, roughly 31 GB is ~120 *derived* arm
directories (`deleak_*`, `eval_v4_gta_*`, `identity_bundles_v4_*`) at 104-350 MB each — the output
of past sweeps, every one of them regenerable from the inputs by the command that made it. What was
actually shipped is the **input** set the chain reads:

`positions`, `positions_filled`, `positions_gate`, `detembed_cache_clip`, `detembed_cache_prtreid`,
`koshkina_percrop`, `koshkina_percrop_votes{,_f080}`, `koshkina_jersey`, `detbox_cache`,
`calib_candidates`, `clip_ckpt`, `jersey_head`, plus `eval_gta_tau0.040_nosplit` (see below) and
the two on-record TEST-38 submissions kept as a diff reference. Same for `gsr_test`.

To ship any specific arm directory later:

```
tar -cf - outputs/gsr/<ARM_DIR> | ssh siddhanth23519@192.168.3.19 'tar -xf - -C ~/football-synthesizer'
```

**`outputs/manutd_*` and `outputs/footpass_*` were deliberately not shipped** (paused ManU thread —
stays local/Drive). The consequence is visible in the test run, §5.

**A hidden coupling worth knowing:** `eval.gsr_identity.split_sequences` defines the DEV-20 /
TEST-38 partition by listing the JSON filenames inside `outputs/gsr/eval_gta_tau0.040_nosplit`
(`BASE_ARM`) — a 294 MB legacy arm directory. Omit it and every split resolves to **0 sequences**,
and the failure surfaces far downstream as a `FileNotFoundError` on
`data/soccernet/gamestate-2024/valid` from inside trackeval (empty `SEQ_INFO` makes the scorer fall
back to listing a split folder that this flat dataset layout does not have). That cost one aborted
gate run. With it present the split resolves to **dev 20 / t38 38**, and the t38 list is identical
to the sequence set of the on-record TEST-38 submission.

### 3.2 The scorer — a real defect caught

`docs`/`results` pin the metric to **sn-trackeval 0.4.0, commit
`9c25232f6f2b56c9f203f1eb55784ff1e97df683`**. Env `gsr` already had "sn-trackeval 0.4.0" installed
from the Session-1 `sn-gamestate` dependency resolution. A per-file SHA-256 comparison against the
laptop's audited install (46 `.py` files, line endings normalised) found **one real difference**:

```
trackeval/datasets/soccernet_gs.py, line 424:
-         data['seq'] = raw_data['seq']          # present on the laptop, ABSENT in env gsr
```

Fetching the file from GitHub at the pinned commit settled it: **upstream 9c25232 has the line**,
so the laptop is correct and **the env-`gsr` build was a different, older sn-trackeval 0.4.0**.
That is the GS-HOTA dataset class itself. Every server score would have come from a scorer that is
not the one on record.

Fix, chosen so that **nothing in the shared `gsr` env was mutated** (the S7 worker was live on the
server at the time):

```
PIP_CONSTRAINT=$HOME/constraints.txt pip install --no-deps --no-cache-dir \
  -t ~/sn-trackeval-9c25232 \
  "git+https://github.com/SoccerNet/sn-trackeval@9c25232f6f2b56c9f203f1eb55784ff1e97df683"
export PYTHONPATH=$HOME/sn-trackeval-9c25232:$PWD      # precedes site-packages
```

Re-verified after install: **all 46 files byte-identical to the laptop's audited copy** (modulo
line endings). The stale copy is still in `site-packages` and will be picked up by any process that
does not export `PYTHONPATH` — treat the export as mandatory for anything that scores.

### 3.3 Environment

Env `gsr` (`~/miniconda3/envs/gsr`, python 3.11.15, torch 2.5.1+cu121) already carried numpy,
pandas, pyarrow, scipy, sklearn, cv2, yaml, tqdm, matplotlib, jinja2, transformers. Nothing from
`requirements.txt` had to be installed for the eval chain; `mplsoccer` (viz), `open_clip` (only
needed on a cold embedding cache), `lxml` and `torchreid` remain absent. All twelve eval modules
import: `core.registry`, `core.pitch`, `generator.gta_link`, `eval.gsr_score`, `eval.gsr_gta`,
`eval.gsr_identity`, `eval.gsr_jersey`, `tools.gsr_v4`, `tools.gsr_deleak`, `tools.gsr_calibgate`,
`tools.gsr_calibfill`, `tools.gsr_eiou`.

Data: `~/football-synthesizer/data/soccernet/gamestate-2024 -> ~/data/gamestate-2024` (symlink,
verified through the harness, not just by `ls`).

## 4. The verification gate

### 4.1 Running the *frozen* code, not the working tree

`tools/gsr_eiou.py` in the working tree has **drifted** from the module that produced the on-record
TEST-38 number — S7 is editing it. Hashes:

```
frozen record (results/gsr_eiou_frozen.json)  a7a8e3d09c10d0e1db37de318b04c823f9b6e5a27916d7314e6b34de8082c0ec
working tree at migration time                44176b1e65b73cb7e05cf0bb5096f6c586356673479a2d9414e1c499b1051d2b
```

Walking the file's git history, exactly one commit matches the frozen hash: **`0abe5c8`**
("gsr v5 arc: CLIP embedder wiring, retune negative, eiou tracker -- test-38 39.54"). The gate
therefore runs from a detached worktree at that commit:

```
git worktree add --detach ~/fs-frozen 0abe5c8
cd ~/fs-frozen && rm -rf outputs data
ln -s ~/football-synthesizer/outputs outputs && ln -s ~/football-synthesizer/data data
find . -name '*.py' -not -path './outputs/*' -not -path './data/*' -exec sed -i 's/\r$//' {} +
```

The `sed` is required: with `core.autocrlf=true` git checks the tree out CRLF, and the frozen hash
was taken over LF bytes. After it, `tools/gsr_eiou.py` hashes to **`a7a8e3d0...`** — the frozen
module, exactly. `python -m tools.gsr_eiou --demo` reproduces the documented self-check
(`IoU 0.000 -> EIoU(e=0.7) 0.133`) and `pytest tests/test_gsr_eiou.py` is 3 passed.

### 4.2 Result

```
python -m tools.gsr_eiou --run --split t38 --embedder clip --grid "0.3:1:0.5:0.30"
```

Control and arm both re-derived in-session, CPU-only, ~16 min wall for the pair.

| | on record (`GSR_EIOU.md` §7) | server, re-derived | delta |
|---|---|---|---|
| control GS-HOTA | 37.0344 | **37.0344** | **-0.0000068** |
| control GS-DetA | 23.9789 | 23.9788 | -0.0000412 |
| control GS-AssA | 57.2018 | 57.2018 | +0.0000755 |
| control GS-LocA | 92.1840 | 92.1840 | +0.0000786 |
| control IDF1 | 38.5485 | 38.5485 | 0.0000000 |
| **EIoU GS-HOTA** | **39.5403** | **39.5403** | **-0.0000292** |
| EIoU GS-DetA | 26.1697 | 26.1695 | -0.0001753 |
| EIoU GS-AssA | 59.7437 | 59.7440 | +0.0003113 |
| EIoU GS-LocA | 92.2192 | 92.2192 | +0.0000195 |
| EIoU IDF1 | 41.2789 | 41.2789 | 0.0000000 |
| paired mean / median | +2.535 / +2.406 | +2.535 / **+2.406 (exact)** | mean -0.000045 |
| helped / hurt | 26 / 12 | **26 / 12** | 0 |
| Wilcoxon p | 0.00092286903236527 | **0.00092286903236527** | 0 (17 digits) |

A field-by-field diff of the whole result JSON (server vs the on-record
`results/gsr_benchmark/gsr_eiou_t38_clip.json`) finds **25 differing leaves in total**, of which
**two are wall-clock timings** and the largest genuine metric difference is **0.0019881** — one
sequence's per-sequence GS-HOTA (SNGS-028). No non-numeric field differs anywhere.

**Everything integer-valued is identical**, which is the part that matters: `n_merges` 1785 /
4568, `frags_after` 1525 / 2048, `frag_per_gt_after` to 15 significant figures, `helped`, `hurt`,
`worst`, `best`. The track partition and the connector's merge decisions are reproduced exactly.

### 4.3 Where the residual 1e-5 comes from

Byte-diffing the 76 submission JSONs against the on-record ones localises it precisely:

| arm | byte-identical files | rows with a differing identity attribute | max pitch-coordinate delta |
|---|---|---|---|
| control | 27 / 38 | **23 / 403,386 (0.0057%)** | **0.000e+00** |
| EIoU | 29 / 38 | **10 / 403,386 (0.0025%)** | **0.000e+00** |

The row keyset (`image_id`, `track_id`) is identical in every sequence and **the pitch geometry is
bit-identical** — zero difference across every coordinate of every differing file. The only
divergence is a few dozen jersey/identity assignments landing on the other side of a numerical tie
in the MILP objective, from float summation order differing between the laptop's numpy 2.4.0 on
Windows and the server's numpy 1.26.4 on Linux. This is *smaller* than the documented RANSAC-jitter
tolerance, and it is not RANSAC — calibration never re-runs here, the positions are cached.

**Verdict: identical, within tie-break noise of 33 rows in 806,772. The server is usable for CPU
eval and scoring.**

Evidence kept server-side at `~/verify_t38/`: `ctrl_server/`, `arm_server/` (the regenerated
submissions) and `gsr_eiou_t38_clip.server.json`. The canonical on-record copies were moved back
into `outputs/gsr/` after the diff, so the server mirror matches the laptop.

## 5. Test suite server-side

`pytest tests/` from `~/football-synthesizer`: **457 passed, 21 failed, 8 skipped**, plus 2 modules
blocked at collection. Every failure is explained; none is a broken migration.

| Count | Cause | Class |
|---|---|---|
| 2 modules (collection error) | `np.trapezoid` at `tools/evidence_sim.py:238` needs numpy >= 2.0; env `gsr` is pinned to **numpy 1.26.4** (Session-1's constraint, required by the 2022-era torchreid training code). Blocks `test_evidence_sim.py` and `test_footpass_score.py`. `np.trapz` is the 1.x spelling. | Real, one call site |
| 15 | Missing `outputs/manutd_*` / `outputs/france_*` fact stores and `matches/*.mp4` — deliberately not shipped (§3.1) | Expected |
| 2 | `lxml` not installed (`test_make_demo.py`) | Trivial |
| 2 | `torchreid` not installed (`test_track_relink.py`, the OSNet relink path) | Expected |
| 1 | `test_kb.py::test_evidence_paths_exist` — **fails identically on the laptop**; pre-existing dead evidence links, not a migration issue | Pre-existing |
| 1 | `test_identity_solve.py` — see below | Real, server-only |

### The `milp` landmine (server-only, found and characterised)

`scipy.optimize.milp` in env `gsr` (scipy 1.14.1 against numpy 1.26.4) raises

```
ValueError: Buffer dtype mismatch, expected 'int' but got 'long'   (_highs_wrapper.pyx:240)
```

for programs with **exactly one** `LinearConstraint`. With two or more it works, because `_milp`'s
internal `vstack` rebuilds the index arrays and downcasts them to int32; a single constraint is
passed through with its int64 indices straight to HiGHS. Confirmed by direct experiment:

```
1 constraint  (45 x 855)  -> FAIL ValueError: Buffer dtype mismatch
2 constraints (45 x 855 + 10 x 855) -> OK status=0
```

`generator.identity_solve.solve_assignment` builds `cons = [eq]` alone when `r == 0` (no group with
>= 2 members and no squad-size constraint) and `cons = [eq, groups]` otherwise. **Real GSR bundles
always produce `r > 0`** — instrumented on SNGS-028, the solver calls `milp` once with constraint
matrices `(45, 855)` and `(1674, 855)` — which is why the TEST-38 gate ran clean over 38 sequences.
The `_demo` in `identity_solve.py` hits the single-constraint branch and dies, which is the whole of
the `test_identity_solve.py` failure.

**Consequence for future server work:** any arm that switches the mutex off (`use_mutex=False`) or
runs bundles with no multi-member group will crash on the server and run fine on the laptop. Fix
when it bites: bump numpy/scipy in a *separate* eval env, or coerce the index dtype before the call.
Do not silently work around it — it is a version-skew bug, not a modelling choice.

## 6. What broke / notes for the next session

1. The stale `sn-trackeval` in `site-packages` (§3.2) is still there. **`export
   PYTHONPATH=$HOME/sn-trackeval-9c25232:$PWD` before anything that scores.**
2. `results/GSR_EIOU.md` §7 reports "tracklets 3,399 / 6,818" for the TEST-38 control/arm. The
   result JSON's `v4.frags_after` reads **1,525 / 2,048** on *both* the laptop and the server, so
   the doc's tracklet column is a different quantity (pre-connector, presumably). Not a migration
   discrepancy — flagged only so nobody chases it.
3. `outputs/gsr_test` has **no `detembed_cache_clip`**. A future test-49 CLIP arm will need a cold
   GPU embedding pass over 49 sequences; the PRTreID cache (144 MB) is the only one present.
4. Nothing was committed and no repo code was modified. Files changed on the laptop: this document
   only. Server-side new paths: `~/football-synthesizer`, `~/fs-frozen`, `~/sn-trackeval-9c25232`,
   `~/verify_t38`, `~/logs/migrate_gate_t38.log`.
5. S7's uncommitted edits rode along in the snapshot (`tools/gsr_eiou.py`, `tools/gsr_v4.py`,
   `eval/gsr_gta.py`, `tools/ocr_density.py`, `knowledge/claims.json`, `tools/gsr_v6det.py` and
   ~30 more). No file was observed changing underfoot during the transfer. The server copy of those
   files is a **snapshot of work in progress**, not a reviewed state — re-sync before using them.
