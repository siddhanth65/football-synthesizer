# Storage audit — football-synthesizer (2026-07-24)

Read-only audit. **Nothing was deleted, moved, or rewritten.** Every reclaim action below is a
command written out for you to run yourself, later, at your discretion.

---

## 0. FIRST — an accidental 2.25 GB I created, and how to remove it

While probing git history I ran `git fsck --lost-found`. I did not realise that flag *writes*
unreachable objects out as decompressed files. It created:

```
c:\Users\siddh_ygv5bws\football-synthesizer\.git\lost-found\   2,250 MB / 271 files
```

This is pure duplicate junk (decompressed copies of objects already in `.git/objects`). It is safe
to remove and removing it changes nothing about your repo or history. I did not remove it because
your instruction was an absolute "delete nothing". Sorry — this is on me.

```powershell
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\.git\lost-found"
```

Every size below is stated *excluding* this 2.25 GB unless labelled "as measured now".

---

## 1. Headline numbers

| Scope | Size |
|---|---|
| Repo total (pre-fsck baseline) | **136.4 GB** |
| Repo total (as measured now, incl. my lost-found) | 138.6 GB |
| Free space on C: | **5.1 GB** |
| External project clones + envs (outside repo) | **7.8 GB** |
| Project grand total | **~144 GB** |

### Repo top-level

| Path | Size | Files |
|---|---|---|
| `matches/` | **60.2 GB** | 201 |
| repo-root `*.mp4` (12 full-match replays) | **51.2 GB** | 12 |
| `data/` | **16.7 GB** | 1,362,601 |
| `outputs/` | **4.42 GB** | 2,370 |
| `results/` | **1.99 GB** | 200,657 |
| `.git/` | **1.85 GB** | 1,620 |
| everything else (code, docs, PDFs, `yolov8s.pt`) | 0.03 GB | ~460 |

### Correction to the audit premise

You said "~140 GB, of which ~90 GB is NOT video." That is not what the disk says.

- Video is **111.4 GB of 136.4 GB = 82%** of the repo.
- Non-video is **25.0 GB** (`data/` 16.7 + `outputs/` 4.4 + `results/` 2.0 + `.git/` 1.85).

The "90 GB that isn't video" is almost entirely video after all — specifically **51.2 GB of video
stored twice**. See §5.

---

## 2. `matches/` — 60.2 GB

| Path | Size | Contents |
|---|---|---|
| 12 × `matches/<manutd_fixture>/` | **51.2 GB** | `h1/chunk_NNN.mp4` + `h2/chunk_NNN.mp4`, 10–11 chunks each (~4.3 GB/match) |
| `matches/fra_sen/` | 3.25 GB | `chunks/` (14 files, 1.95 GB) **+ `play/` (10 files, 1.30 GB — bytes already in `chunks/`)** |
| `matches/france_senegal/` | 1.97 GB | `h1/` + `h2/` chunks — same fixture as `fra_sen`, registry-canonical |
| `matches/france_norway/` | 1.65 GB | h1/h2 chunks (PMSR ground-truth match) |
| `matches/france_iraq/` | 1.35 GB | h1/h2 chunks (PMSR ground-truth match) |
| `matches/pl_probe/` | 0.77 GB | 9 × ~88 MB `seg_N.mp4` feasibility clips, 3 fixtures |

Registry (`data/matches.yaml`) contains 16 match ids. `fra_sen` and `pl_probe` are **not** among
them — they are pre-registry scratch.

---

## 3. `data/` — 16.7 GB

| Path | Size | Files | What |
|---|---|---|---|
| `data/soccernet/gamestate-2024/` | **11.54 GB** | 43,559 | 58 × `SNGS-0NN/` sequences, 750 frames + labels each. GSR benchmark corpus. |
| `data/soccernet/jersey-2023/` | **4.28 GB** | 1,297,552 | Jersey-number crop dataset (the 1.3M tiny files that make every dir scan slow). |
| `data/jersey_negatives_experimental/` | 0.40 GB | 13,877 | Generated negative crops (experimental sweep). |
| `data/imputation/` | 0.20 GB | 9 | Metrica + SkillCorner open tracking data (B4 imputation truth). |
| `data/jersey_negatives/` | 0.14 GB | 4,603 | Generated negative crops (production). |
| `data/jersey_negatives_graphic_clean/` | 0.12 GB | 2,592 | Generated negative crops (graphic-cleaned variant). |
| `data/ball_annotations/` | ~0 | 22 | **Hand-annotated ball ground truth. Tiny. Precious.** |

---

## 4. `.git/` — 1.85 GB, and the .npz theory is wrong

`git count-objects -vH`: **1,595 loose objects, 1.85 GiB, zero packs.** This repo has never been
gc'd — nothing is delta-compressed.

**Where the 1.85 GB actually is:**

| Category | Size | Share of `.git` |
|---|---|---|
| Two **dangling** (unreachable) zip blobs | **1.92 GB on disk** | **>99%** |
| All blobs reachable from any ref, all history, all branches | **29.7 MB** (808 blobs) | ~1.5% |

The two blobs are `0dd0cfab…` (1,249,810,960 B) and `8c598716…` (1,074,009,748 B). Byte-sniffing
them shows ZIP headers with first entries `train/` and `test/` — these are the **SoccerNet
jersey-2023 train.zip and test.zip**, `git add`-ed once and then never committed. Your own
`.gitignore` documents the moment:

```
# SoccerNet downloads (4.4 GB incl. two ~1 GB zips — never commit; licensing + size)
data/soccernet/
```

The ignore rule went in *after* the `git add`. The staged blobs were orphaned, not deleted.
`git rev-list --objects --all` confirms neither blob is reachable from any commit, and
`git ls-files` confirms neither is in the index.

**Your `.npz` concern — measured, and it is a non-issue.** All `.npz` blobs in the entire
reachable history: **4.06 MB across 131 blobs.** That is 0.2% of `.git`.

Full reachable-history rollup by extension:

| Ext | Size in history | Blobs |
|---|---|---|
| `.json` | 14.48 MB | 87 |
| `.md` | 5.10 MB | 191 (mostly ~40 `STATUS.md` revisions at 100–174 KB each) |
| `.npz` | 4.06 MB | 131 |
| `.py` | 3.32 MB | 313 |
| `.csv` | 1.99 MB | 22 |
| `.html` | 0.61 MB | 40 |
| everything else | <0.15 MB | 24 |

No `.mp4`, no `.parquet`, no `.jpg`, no `.pt/.pth` in history — `.gitignore` held.

**Verdict: do NOT rewrite history.** `filter-repo`/BFG would buy you 4 MB and cost you every commit
hash, every clone, and a force-push you have standing orders never to do. The 1.85 GB comes back
from a plain **`git gc --prune=now`**, which touches no commit and rewrites no history:

```powershell
cd "c:\Users\siddh_ygv5bws\football-synthesizer"; git gc --prune=now --aggressive
```

Expect `.git` to drop from ~1.85 GB to roughly **15–25 MB**.

---

## 5. Duplicates and waste

### 5.1 Every ManU match is stored TWICE — 51.2 GB

Yes. `tools/chunk_video.py` runs ffmpeg with `-c copy` (stream-copy, **no re-encode, lossless**), so
`matches/<id>/h1|h2/chunk_*.mp4` is a byte-for-byte re-container of the repo-root
`<Fixture> 2024-25 Full Match Replay.mp4`. Sizes match to within container-header overhead:

| Root .mp4 | Root GB | `matches/<id>` | Chunks GB | Chunks |
|---|---|---|---|---|
| Crystal Palace v Man Utd | 4.394 | `palace_manutd` | 4.396 | 12 |
| Southampton v Man Utd | 4.363 | `southampton_manutd` | 4.364 | 11 |
| Man Utd v Brighton | 4.345 | `manutd_brighton` | 4.346 | 11 |
| Man Utd v Crystal Palace | 4.320 | `manutd_palace` | 4.322 | 11 |
| Man Utd v Liverpool | 4.309 | `manutd_liverpool` | 4.310 | 11 |
| Fulham v Man Utd | 4.286 | `fulham_manutd` | 4.288 | 11 |
| Liverpool v Man Utd | 4.282 | `liverpool_manutd` | 4.283 | 11 |
| Brighton v Man Utd | 4.267 | `brighton_manutd` | 4.269 | 11 |
| Man Utd v Tottenham | 4.255 | `manutd_tottenham` | 4.255 | 11 |
| Man Utd v Southampton (1) | 4.203 | `manutd_southampton` | 4.205 | 10 |
| Man Utd v Fulham (1) | 4.122 | `manutd_fulham` | **4.121** | 10 |
| Tottenham v Man Utd | 4.065 | `tottenham_manutd` | 4.065 | 11 |
| **total** | **51.211** | | **51.224** | |

Only `manutd_fulham` has chunks summing *below* its source (by ~1 MB), which is the one case worth
verifying before you trust the chunks alone. Everything else is chunks ≥ source, as stream-copy
should be.

The pipeline reads chunks, never the root files. **Keep the chunks, retire the root .mp4s.**
Verify duration parity first:

```powershell
# for each fixture: source duration vs sum of chunk durations (should match to <1 s)
ffprobe -v error -show_entries format=duration -of csv=p=0 "Man Utd v Fulham 2024-25 Full Match Replay (1).mp4"
Get-ChildItem "matches\manutd_fulham" -Recurse -Filter *.mp4 | ForEach-Object { ffprobe -v error -show_entries format=duration -of csv=p=0 $_.FullName } | Measure-Object -Sum
```

### 5.2 `matches/fra_sen` duplicates itself AND `matches/france_senegal` — 3.25 GB

- Internal: `fra_sen/play/chunk_004..013.mp4` (1.30 GB) are byte-copies of `fra_sen/chunks/chunk_004..013.mp4`.
- External: `fra_sen/chunks/` (1.95 GB, 14 chunks) covers the same France–Senegal fixture as the
  registry-canonical `matches/france_senegal/` (1.97 GB, h1/h2 layout). `fra_sen` is the old
  flat-chunks generation, superseded by the h1/h2 convention and absent from `data/matches.yaml`.

### 5.3 `outputs/gsr/` — six generations of the same eval — 1.76 GB

Six directories, each ~293.4 MB, each holding `predictions/data/SNGS-0NN.json` for the same 58
sequences under a different re-linking config:

| Dir | MB | Config |
|---|---|---|
| `eval/` | 293.5 | baseline |
| `eval_koshkina/` | 293.5 | Koshkina anchors |
| `eval_relink/` | 293.4 | OSNet relink |
| `eval_prtreid_relink_0.960/` | 293.4 | PRTreID @0.960 |
| `eval_prtreid_relink_0.965/` | 293.4 | PRTreID @0.965 |
| `eval_prtreid_relink_0.960_propagate/` | 293.4 | PRTreID @0.960 + propagate |

Spot-check of `SNGS-096.json` across all six: sizes 8,560,074–8,563,159 B, six distinct MD5s — so
they are genuinely different runs, not literal copies. But **every scored result from all six is
already distilled into `results/gsr_benchmark/`** (`gsr_scores*.json`, 40–151 KB each, plus
`GSR_BENCHMARK.md`, `GSR_RESCORE_KOSHKINA.md`, `GSR_RESCORE_PRTREID.md`). The 1.76 GB of raw
per-frame prediction JSON is intermediate.

### 5.4 Checkpoint zoos — 2.09 GB

- `outputs/jersey/`: **10 × 128.2–128.6 MB = 1.28 GB**. Canonical is
  `jersey_torso_r224_acc417.pt` (hardcoded at `tools/closeup_anchor_probe.py:53`). The other 9 are
  superseded epoch/arch snapshots (`ckpt.pt`, `ckpt_mh.pt`, `ckpt_mh2.pt`, `ckpt_r128_acc367.pt`,
  `ckpt_torso.pt`, `jersey_r224_acc396.pt`, `jersey_mh_r224_acc396.pt`, + the two `_neg` variants).
- `outputs/ball_finetuned/`: **19 × 43.3 MB = 0.82 GB**. Canonical per `README.md:70` is
  `tracknetv2_v5.pth` (WC production) and `tracknetv2_v6.pth` (PL production). The other 17 are `_ep5`
  / `_ep10` / `_ep15` intermediates and v2/v3/v4 generations.

### 5.5 `results/closeup_anchor_probe/` — 1.40 GB, 200,031 files

| Ext | Size | Files |
|---|---|---|
| `.jpg` | **1,355.5 MB** | 199,826 |
| `.png` | 68.4 MB | 65 |
| `.json` | 7.3 MB | 128 |
| `.csv` | 1.9 MB | 2 |
| `.log` / `.md` | 0.2 MB | 10 |

The 200k jpgs are per-detection jersey/close-up crop caches under `koshkina*/levers/` etc. The
**analysis output** (`parseq_positions.json`, `kit_centroids.json`, `closeup_reads.csv`,
`spotcheck_step3/verdicts.json`, the `.md` writeups) is 9.4 MB and includes the human-verified
verdicts. The crops are the disposable 99.3%.

Also stale scratch inside it: `_tmp_spot/` (19.7 MB), `_tmp_lev/` (6.2 MB), `_tmp_full/` (5.7 MB).

### 5.6 Not a duplicate — `outputs/<match>/` is cheap and precious

All 36 non-model dirs under `outputs/` total **0.50 GB** (~28 MB per match: 40 parquets, 11 npz, 23
json each, in `final/`, `h1/`, `h2/`, `ball_action/`, `live_play/`, `facts/`). This is the entire
validated CV product of the project and it is 0.4% of the repo. Do not touch it.

---

## 6. External clones and environments — 7.8 GB

| Path | Size | Files | What | Re-obtainable |
|---|---|---|---|---|
| `~/ball-action-env` | **4.98 GB** | 30,383 | Python venv (torch + CUDA wheels) for ball-action-spotting | yes — `python -m venv` + `pip install -r`, ~20 min + bandwidth |
| `~/jersey-str-env` | 1.10 GB | 23,863 | Python venv for the PARSeq/STR jersey reader | yes — same |
| `~/jersey-number-pipeline` | 0.98 GB | 109,870 | clone; `models/` 445 MB + `out/` 371 MB (109,688 generated crops) + `str/` 92 MB + `.git/` 86 MB | clone free; `models/` re-downloadable; `out/` regenerates from the tool |
| `~/prtreid` | 0.39 GB | 300 | clone; `weights/` = 378 MB single checkpoint | yes — clone + weight download |
| `~/.cache/torch` | 0.22 GB | 1 | torch hub weight cache | yes — auto-refetch |
| `~/ball-action-spotting` | 0.07 GB | 115 | clone; `data/` 53 MB + a 16 MB `bas_run.log` | yes |
| `~/action-spot-env` | 0.04 GB | 98 | small venv | yes |
| `~/.cache/huggingface` | 0.03 GB | 9 | HF cache — negligible, this project barely used it | yes |

Everything in this section is **100% re-obtainable** (clone + pip + public weight download). None of
it contains project-unique state. Note `~/football-state-of-play` (23.5 GB) and
`~/manutd-stylistic-evolution` (1.26 GB) are separate repos and outside this audit's scope.

---

## 7. Full classification table

Ordered by size. "Regen cost" uses your measured numbers where they exist: `tools/batch_match.py`
chunk_000 took **31 min** on this laptop (STATUS.md:1843 — CPU-bound on PnL refinement, GPU idle),
so a full 11-chunk match extract is **~5.5 h**; the identity/closeup chain is **~2 h GPU per match**
(STATUS.md:7).

| path | size | what it is | REGENERABLE? | what breaks if it's gone | recommendation |
|---|---|---|---|---|---|
| repo-root `*.mp4` ×12 | **51.2 GB** | Full-match PL replays Sid supplied | **free** — `ffmpeg -f concat` from `matches/<id>/h1+h2` chunks (lossless stream-copy both ways), ~10 min for all 12 | Nothing in the pipeline. Only re-chunking with a *different* chunk length, which can also be done from the chunks. | **SAFE TO DELETE** (verify duration parity first, esp. `manutd_fulham`) |
| `matches/<12 manutd ids>/` | **51.2 GB** | Chunked broadcast video, the pipeline's actual input | **NOT regenerable** if root .mp4s also go — PL replays, licence forbids re-download | Every CV re-run, every future metric, every new match-level artifact | **KEEP** (this is the copy to keep) |
| `data/soccernet/gamestate-2024/` | 11.5 GB | 58 SoccerNet GameState sequences (frames + GT) | **cheap** — re-download w/ NDA password (memory: `soccernet-credentials.md`), ~1–2 h bandwidth | Re-running the GSR benchmark / any new re-ID sweep. Scores already saved in `results/gsr_benchmark/`. | **ARCHIVE ELSEWHERE** (or delete + re-download when next needed) |
| `data/soccernet/jersey-2023/` | 4.28 GB | Jersey-number crop dataset, 1.3M files | **cheap** — re-download w/ NDA password, ~1 h | Re-training the jersey recognizer. Trained weights already exist. | **ARCHIVE ELSEWHERE** |
| `matches/fra_sen/` | 3.25 GB | Superseded flat-chunk generation of France–Senegal; `play/` duplicates `chunks/` internally | **free** — re-chunk from `matches/france_senegal/`; not in `data/matches.yaml` | Nothing — no registry entry, no code path references it | **SAFE TO DELETE** |
| `matches/france_senegal/` | 1.97 GB | France–Senegal chunks, registry-canonical, PMSR ground truth | **NOT regenerable** (WC source video) | PMSR method calibration; the only fixture with a FIFA PMSR PDF cross-check | **KEEP** |
| `.git/` dangling blobs | 1.85 GB | Two orphaned SoccerNet zips staged pre-`.gitignore` | **free** — `git gc --prune=now` | Nothing. Unreachable from every ref and absent from the index. | **SAFE TO DELETE** (via gc, not history rewrite) |
| `matches/france_norway/` | 1.65 GB | WC chunks, PMSR ground truth | **NOT regenerable** | PMSR calibration | **KEEP** |
| `outputs/gsr/eval_*` (5 superseded of 6) | 1.47 GB | Per-frame GSR prediction JSON, one dir per relink config | **expensive-GPU-hours** — full 58-sequence re-eval per config | Nothing scored: all six configs' numbers live in `results/gsr_benchmark/gsr_scores*.json` + 3 `.md` files | **SAFE TO DELETE** (keep `results/gsr_benchmark/`) |
| `results/closeup_anchor_probe/**/*.jpg` | 1.36 GB | 199,826 cached crops from the identity probe | **expensive-GPU-hours** — ~2 h GPU/match × 9 matches ≈ 18 h to regenerate | Nothing downstream: the reads (`parseq_positions.json`, `closeup_reads.csv`) and verdicts are separate JSON/CSV. Only re-inspecting a specific crop by eye. | **SAFE TO DELETE** (keep every `.json` / `.csv` / `.md` in that tree) |
| `matches/france_iraq/` | 1.35 GB | WC chunks, PMSR ground truth | **NOT regenerable** | PMSR calibration | **KEEP** |
| `outputs/jersey/*.pt` (9 superseded of 10) | 1.16 GB | Jersey recognizer training zoo | **expensive-GPU-hours** — full re-train per checkpoint | Nothing — `jersey_torso_r224_acc417.pt` is the only one referenced in code. Metrics for all variants are in `results/jersey_model/JERSEY_MODEL.md`. | **ARCHIVE ELSEWHERE** (2 `_neg` variants are documented experiments; the other 7 are **SAFE TO DELETE**) |
| `~/ball-action-env` | 4.98 GB | venv | **cheap-CPU** — pip reinstall ~20 min | Ball-action-spotting reruns until you rebuild it | **SAFE TO DELETE** |
| `outputs/ball_finetuned/*.pth` (17 superseded of 19) | 0.74 GB | TrackNetV2 epoch/generation snapshots | **expensive-GPU-hours** — a fine-tune run each | Nothing — `v5` (WC prod) and `v6` (PL prod) are the referenced ones (`README.md:70`) | **SAFE TO DELETE** (keep `tracknetv2_v5.pth`, `tracknetv2_v6.pth`) |
| `matches/pl_probe/` | 0.77 GB | 9 × 88 MB feasibility clips, 3 fixtures | **free** — re-cut from the corresponding full matches, minutes | Nothing — pre-registry probe, superseded by the 12 full matches | **SAFE TO DELETE** |
| `~/jersey-str-env` | 1.10 GB | venv | **cheap-CPU** ~20 min | STR jersey reads until rebuilt | **SAFE TO DELETE** |
| `~/jersey-number-pipeline/out` + `/models` | 0.82 GB | 109,688 generated crops + downloaded weights | **cheap** — regenerate/redownload | Nothing project-side | **SAFE TO DELETE** |
| `outputs/<36 match dirs>/` | 0.50 GB | Per-chunk dense/tactical parquets, match-level aligned parquets, ball dirs, facts, npz | **expensive-GPU-hours** — ~5.5 h/match × 15 ≈ **80 h** | **Everything.** Every metric, every report, every synthesizer input, every registry `aligned:` / `ball_dir:` path | **KEEP — highest value per byte in the project** |
| `data/jersey_negatives_experimental/` | 0.40 GB | Generated negative crops (sweep) | **cheap-CPU** — regenerate from the tool | Retraining the `_neg` jersey variants | **SAFE TO DELETE** |
| `~/prtreid/weights` | 0.38 GB | PRTreID checkpoint | **cheap** — public download | PRTreID re-linking until redownloaded | **ARCHIVE ELSEWHERE** |
| `results/demo/` full-res mp4 | 0.20 GB | `cv_pipeline_demo.mp4` (153 MB) + `identity_demo.mp4` (47 MB) | **cheap-CPU** — `tools/make_demo.py` / `make_identity_demo.py` | Nothing — 720p versions (43 + 19 MB) are what the demo page serves | **SAFE TO DELETE** (keep the `_720p` files + poster jpg) |
| `data/imputation/` | 0.20 GB | Metrica + SkillCorner open tracking data | **cheap** — public download | B4 imputation validation | **KEEP** (small, and finding the exact release again is a chore) |
| `.git/` reachable history | 0.03 GB | All commits, all code, 191 `.md` revisions, 131 `.npz` (4 MB) | **NOT regenerable** | Your entire project history | **KEEP — do not rewrite** |
| `data/ball_annotations/` | ~0 | Hand-annotated ball ground truth CSVs | **NOT regenerable** — human labour | Every ball-metric validation; the only thing standing between you and another ball retraction | **KEEP — precious** |
| `PMSR-*.pdf` ×3 | 0.015 GB | FIFA PMSR match reports | **NOT regenerable** — copyrighted, not re-downloadable | Method calibration ground truth | **KEEP — precious** |

---

## 8. (a) Ranked reclaim list

In-repo first, biggest and safest first.

| # | Item | GB | Cumulative | Risk |
|---|---|---|---|---|
| 0 | `.git/lost-found` (my accident, §0) | 2.25 | 2.25 | none |
| 1 | 12 repo-root `*.mp4` (chunks are lossless copies) | 51.21 | 53.46 | low — verify durations first |
| 2 | `data/soccernet/` (NDA re-download) | 15.82 | 69.28 | none — archive or refetch |
| 3 | `git gc --prune=now` (dangling zip blobs) | 1.85 | 71.13 | none — no history change |
| 4 | `matches/fra_sen/` (superseded, self-duplicating) | 3.25 | 74.38 | none |
| 5 | `outputs/gsr/eval_*` — 5 superseded of 6 | 1.47 | 75.85 | none — scores retained |
| 6 | `results/closeup_anchor_probe/**/*.jpg` (199,826 crops) | 1.36 | 77.21 | none — reads/verdicts retained |
| 7 | `outputs/jersey/` — 7 superseded checkpoints | 0.90 | 78.11 | none |
| 8 | `outputs/ball_finetuned/` — 17 superseded checkpoints | 0.74 | 78.85 | none |
| 9 | `matches/pl_probe/` (pre-registry probe clips) | 0.77 | 79.62 | none |
| 10 | `data/jersey_negatives_experimental/` | 0.40 | 80.02 | none |
| 11 | `results/demo/` full-res mp4s (720p kept) | 0.20 | 80.22 | none |
| 12 | `results/closeup_anchor_probe/_tmp_*` scratch | 0.03 | 80.25 | none |
| — | **in-repo subtotal** | | **80.3 GB** | |
| 13 | `~/ball-action-env` | 4.98 | 85.23 | rebuild ~20 min |
| 14 | `~/jersey-str-env` | 1.10 | 86.33 | rebuild ~20 min |
| 15 | `~/jersey-number-pipeline/{out,models}` | 0.82 | 87.15 | redownload |
| 16 | `~/prtreid/weights` | 0.38 | 87.53 | redownload |
| — | **grand total** | | **~87.5 GB** | |

Doing just #0 through #3 gets you **71 GB back in about ten minutes** and touches nothing the
pipeline reads.

### Top 5 by size, with regeneration cost

1. **Repo-root `*.mp4` — 51.2 GB.** Free to regenerate (`ffmpeg -f concat` from the chunks, ~10 min
   total); or simply never regenerate, because nothing reads them.
2. **`data/soccernet/` — 15.8 GB.** Cheap: re-download with the NDA password, ~2–3 h of bandwidth.
3. **`matches/fra_sen/` — 3.25 GB.** Free: re-chunk from `matches/france_senegal/`, minutes. Not in
   the registry at all.
4. **`.git` dangling blobs — 1.85 GB.** Free: `git gc --prune=now`, seconds. Not a history rewrite.
5. **`outputs/gsr/eval_*` (5 dirs) — 1.47 GB.** Expensive-GPU-hours to regenerate, but the scores
   are already extracted to `results/gsr_benchmark/`, so nothing of value is lost.

## 8. (b) DO-NOT-DELETE — the non-regenerable list

1. `matches/brighton_manutd/`, `fulham_manutd/`, `liverpool_manutd/`, `manutd_brighton/`,
   `manutd_fulham/`, `manutd_liverpool/`, `manutd_palace/`, `manutd_southampton/`,
   `manutd_tottenham/`, `palace_manutd/`, `southampton_manutd/`, `tottenham_manutd/` — **51.2 GB.**
   After you delete the root .mp4s these become the only copy of footage you cannot legally
   re-download. Back these up before anything else.
2. `matches/france_senegal/`, `matches/france_norway/`, `matches/france_iraq/` — 4.97 GB. WC source
   video paired with the PMSR PDFs.
3. `PMSR-M17-FRA-V-SEN.pdf`, `PMSR-M42-FRA-V-IRQ.pdf`, `PMSR-M61-NOR-V-FRA.pdf` — 15 MB. Copyrighted
   FIFA ground truth, no reproduction path.
4. `data/ball_annotations/` — hand annotations. The validator behind every ball claim.
5. `outputs/<match>/` for all 36 dirs — 0.50 GB. ~80 GPU/CPU-hours of validated CV product; every
   registry `aligned:` and `ball_dir:` path points here.
6. `results/gsr_benchmark/` (0.6 MB) — the distilled scores that make 1.76 GB of eval dirs disposable.
7. `results/closeup_anchor_probe/**/*.json`, `closeup_reads.csv`, `spotcheck_step3/verdicts.json`,
   `spotcheck/VERDICTS.md`, `CLOSEUP_ANCHORS.md` — 9.4 MB. **Human-verified identity verdicts.**
8. `results/identity/`, `results/reports/`, `results/possession*`, `results/jersey_model/`,
   `results/CLAIMS_AUDIT.md` — <1 MB total, all validated conclusions.
9. `outputs/jersey/jersey_torso_r224_acc417.pt`, `outputs/ball_finetuned/tracknetv2_v5.pth`,
   `outputs/ball_finetuned/tracknetv2_v6.pth` — 215 MB, the three production weights.
10. `outputs/pmsr/*.json` — parsed PMSR ground truth.
11. `.git/` reachable history — 30 MB. Never rewrite it; there is nothing in there worth 4 MB of
    savings and a broken remote.

## 8. (c) Policy changes that stop this recurring

**`.gitignore` — one gap to close.** It is doing its job for the working tree (no video, parquet,
image, or weight blob ever reached history). But the zip rule came *after* the `git add`. Add a
belt-and-braces line and never repeat the 1.9 GB orphan:

```
data/soccernet/**
*.zip
```

plus a habit: after any `git reset` that unstages something huge, run `git gc --prune=now`.

**Actual policy changes worth making (not .gitignore):**

1. **Never keep both the source .mp4 and its chunks.** Make `tools/chunk_video.py` print, on
   success, the exact command to remove the source — or add a `--delete-source` flag that only fires
   after verifying total chunk duration equals source duration. This one rule is worth 51 GB.
2. **Checkpoint retention: keep-2.** Training tools write `_ep5/_ep10/_ep15` snapshots and never
   prune. Adopt "keep the best + the last, delete the rest on run completion" — worth ~1.9 GB across
   `outputs/jersey/` and `outputs/ball_finetuned/`.
3. **Eval-run retention: keep the scores, drop the predictions.** `outputs/gsr/eval_*` shows the
   pattern — six 293 MB prediction dumps whose entire value is a 60 KB scores JSON. Make the eval
   harness delete `predictions/` after scoring unless `--keep-predictions` is passed.
4. **Crop caches under `results/` should live under a `_cache/` prefix** that a single
   `tools/clean_caches.py` can purge. 199,826 jpgs (1.36 GB) accumulated with no expiry, and they
   also make every `Get-ChildItem -Recurse` in the repo crawl.
5. **`_tmp_*` dirs need an owner.** `_tmp_spot`, `_tmp_lev`, `_tmp_full` are small now but they are
   evidence that scratch is never swept.
6. **Registry as the source of truth for `matches/` too.** `fra_sen` and `pl_probe` (4.0 GB) survived
   only because nothing checks `matches/` against `data/matches.yaml`. A 10-line
   `tools/orphan_matches.py` that lists `matches/*` dirs with no registry entry would have flagged
   both.
7. **Archive target.** Move `data/soccernet/` and the checkpoint zoo to an external drive rather than
   deleting — they cost hours to reacquire and nothing to store offline.

## 8. (d) One command per item — WRITTEN, NOT EXECUTED

Nothing below was run. Read each before you run it. Consider `-WhatIf` on the `Remove-Item` calls.

```powershell
# 0. my accidental lost-found (2.25 GB)
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\.git\lost-found"

# 1a. VERIFY FIRST: source duration vs summed chunk duration for one fixture
ffprobe -v error -show_entries format=duration -of csv=p=0 "c:\Users\siddh_ygv5bws\football-synthesizer\Man Utd v Fulham 2024-25 Full Match Replay (1).mp4"
(Get-ChildItem "c:\Users\siddh_ygv5bws\football-synthesizer\matches\manutd_fulham" -Recurse -Filter *.mp4 | ForEach-Object { [double](ffprobe -v error -show_entries format=duration -of csv=p=0 $_.FullName) } | Measure-Object -Sum).Sum

# 1b. then, and only then: the 12 root replays (51.2 GB)
Get-ChildItem "c:\Users\siddh_ygv5bws\football-synthesizer\*.mp4" | Remove-Item -Force

# 2. SoccerNet corpora (15.8 GB) -- archive to an external drive instead if you have one
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\data\soccernet"

# 3. dangling git blobs (1.85 GB) -- no history rewrite, no force-push
cd "c:\Users\siddh_ygv5bws\football-synthesizer"; git gc --prune=now --aggressive

# 4. superseded France-Senegal generation (3.25 GB)
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\matches\fra_sen"

# 5. superseded GSR eval dirs (1.47 GB) -- keeps outputs\gsr\eval as the baseline
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\outputs\gsr\eval_koshkina","c:\Users\siddh_ygv5bws\football-synthesizer\outputs\gsr\eval_relink","c:\Users\siddh_ygv5bws\football-synthesizer\outputs\gsr\eval_prtreid_relink_0.960","c:\Users\siddh_ygv5bws\football-synthesizer\outputs\gsr\eval_prtreid_relink_0.965","c:\Users\siddh_ygv5bws\football-synthesizer\outputs\gsr\eval_prtreid_relink_0.960_propagate"

# 6. closeup crop cache (1.36 GB) -- jpgs only; every json/csv/md survives
Get-ChildItem "c:\Users\siddh_ygv5bws\football-synthesizer\results\closeup_anchor_probe" -Recurse -Filter *.jpg | Remove-Item -Force

# 7. superseded jersey checkpoints (0.90 GB) -- keeps acc417 + both _neg experiments
Get-ChildItem "c:\Users\siddh_ygv5bws\football-synthesizer\outputs\jersey\*.pt" | Where-Object { $_.Name -notin @('jersey_torso_r224_acc417.pt','ckpt_torso_neg.pt','ckpt_torso_neg_gentle.pt') } | Remove-Item -Force

# 8. superseded ball checkpoints (0.74 GB) -- keeps v5 (WC prod) + v6 (PL prod)
Get-ChildItem "c:\Users\siddh_ygv5bws\football-synthesizer\outputs\ball_finetuned\*.pth" | Where-Object { $_.Name -notin @('tracknetv2_v5.pth','tracknetv2_v6.pth') } | Remove-Item -Force

# 9. pre-registry probe clips (0.77 GB)
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\matches\pl_probe"

# 10. experimental jersey negatives (0.40 GB)
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\data\jersey_negatives_experimental"

# 11. full-res demo videos (0.20 GB) -- 720p + poster survive
Remove-Item -Force "c:\Users\siddh_ygv5bws\football-synthesizer\results\demo\cv_pipeline_demo.mp4","c:\Users\siddh_ygv5bws\football-synthesizer\results\demo\identity_demo.mp4"

# 12. probe scratch dirs (0.03 GB)
Remove-Item -Recurse -Force "c:\Users\siddh_ygv5bws\football-synthesizer\results\closeup_anchor_probe\_tmp_spot","c:\Users\siddh_ygv5bws\football-synthesizer\results\closeup_anchor_probe\_tmp_lev","c:\Users\siddh_ygv5bws\football-synthesizer\results\closeup_anchor_probe\_tmp_full"

# 13-16. external envs and caches (7.3 GB) -- all rebuild from pip/clone
Remove-Item -Recurse -Force "$env:USERPROFILE\ball-action-env"
Remove-Item -Recurse -Force "$env:USERPROFILE\jersey-str-env"
Remove-Item -Recurse -Force "$env:USERPROFILE\jersey-number-pipeline\out","$env:USERPROFILE\jersey-number-pipeline\models"
Remove-Item -Recurse -Force "$env:USERPROFILE\prtreid\weights"

# BEFORE ANY OF THE ABOVE: back up the irreplaceable footage to an external drive
robocopy "c:\Users\siddh_ygv5bws\football-synthesizer\matches" "E:\fs-backup\matches" /E /XD fra_sen pl_probe
```

---

## 9. Method notes

- Sizes from `Get-ChildItem -Recurse -Force -File | Measure-Object Length -Sum`; GB = 1024^3.
- Git analysis from `git count-objects -vH`, `git cat-file --batch-all-objects --batch-check`, and
  `git rev-list --objects --all` joined on object id. No `filter-repo`, no `gc`, no `prune` was run.
- Duplication of root .mp4 vs chunks established from `tools/chunk_video.py` (`-c copy`, documented
  lossless) plus per-fixture byte totals agreeing to within 0.05%.
- Runtime figures cited from `STATUS.md:7` (~2 h GPU per match, identity chain) and `STATUS.md:1843`
  (chunk_000 = 31 min, 11 chunks per match).
- Everything in this document is an observation or a recommendation. No file was created, removed,
  or relocated by this audit except `.git/lost-found`, disclosed in §0.
