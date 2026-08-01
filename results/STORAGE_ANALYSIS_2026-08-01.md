# Storage analysis — football-synthesizer (2026-08-01)

Read-only. Nothing deleted, moved, or renamed. This updates the prior audit
(`results/STORAGE_AUDIT_2026-07-24.md`) — most of that report's Tier-A items (root-level
`*.mp4`, `.git` dangling blobs) are already gone, confirming `git gc` and the root-.mp4 cleanup
happened since. The items that were flagged but NOT executed (jersey/ball checkpoint zoos,
`closeup_anchor_probe` crops, `pl_probe`, `fra_sen`, demo full-res mp4s) are still present at
essentially unchanged sizes and are re-verified below.

**Current free space on C: 25.72 GB** (27,620,724,736 B). Used 450.15 GB.

## Live / untouchable, per the campaign brief (checked first, not re-analyzed for reclaim)

- `data/soccernet/gamestate-2024/` (all 3 splits) — **30.85 GB** — GSR campaign core (STATUS
  2026-08-01: official test-split submission in flight). Untouchable.
- `outputs/gsr/` + `outputs/gsr_test/` — **8.46 GB** (6.972 + 1.488) — live campaign eval/deleak
  caches (64 sub-dirs: GTA-tau sweeps, deleak arms, identity-solver bundles). Untouchable while
  the campaign is iterating on these numbers, even though several are literal re-run duplicates
  (see Tier D note below — flagged, not sized into any reclaim tier).
- `matches/` ManU corpus (12 registered fixtures, 51.2 GB) and `footpass_game_18/24/47`
  (10.6 GB) and `france_senegal/france_norway/france_iraq` (4.97 GB, PMSR ground truth) —
  irreplaceable broadcast footage. Flagged only as an **external-drive backup candidate**, never
  a delete candidate.
- `data/ball_annotations/`, `knowledge/`, `results/gsr_benchmark/`, `results/jersey_model/`,
  small validated-conclusion dirs under `results/` — all <1 MB combined, never.

## Ranked table

| # | Item | GB | What it is | Re-derivable how | Risk if deleted | Tier |
|---|---|---|---|---|---|---|
| 1 | `data/footpass/raw/videos_fullHD_VAL.zip` | **10.565** | NDA-downloaded raw video zip, 3 entries: `game_18.mp4` (3499.5 MB), `game_24.mp4` (3789.8 MB), `game_47.mp4` (3576.6 MB) | Re-download via SoccerNet FOOTPASS NDA (credentials on file, per memory) — but not actually needed: `matches/footpass_game_{18,24,47}/{h1,h2}/chunk_*.mp4` is already the extracted, working copy (verified: 1920x1080, 25 fps, h264, chunk numbering h1=000-005 + h2=006-010 spans the full match, sizes agree to within ~4%, consistent with container/audio-track overhead). `tools/footpass_digest.py` (the only code that touches `data/footpass/raw/`) reads only `tactical_data_{TRAIN,VAL}.zip`, never this file. | None — unreferenced by any code path | **A** |
| 2 | `results/closeup_anchor_probe/**/*.jpg` + `*.png` | **1.424** (1355.5+68.4 MB) | 199,826+65 cached per-detection jersey/close-up crop images from the identity probe | Re-run the probe (`tools/closeup_anchor_probe.py`), ~2 h GPU/match x 9 matches | None on results — the analysis outputs (`parseq_positions.json`, `kit_centroids.json`, `closeup_reads.csv`, `spotcheck_step3/verdicts.json`, all `.md`) are separate files (9.4 MB) and are retained; only re-eyeballing a specific crop is lost | **A** |
| 3 | `outputs/jersey/*.pt` — 7 of 10 checkpoints | **0.877** (7 x 128.2-128.6 MB) | Superseded jersey-recognizer training snapshots (`ckpt.pt`, `ckpt_mh.pt`, `ckpt_mh2.pt`, `ckpt_r128_acc367.pt`, `jersey_r224_acc396.pt`, `jersey_mh_r224_acc396.pt`, `ckpt_torso.pt`) | Re-train per `results/jersey_model/JERSEY_MODEL.md` | None — verified `jersey_torso_r224_acc417.pt` is the only path hardcoded in code (`tools/carrier_constrained.py`, `tools/closeup_anchor_probe.py`); keep it + the two `_neg` variants (`ckpt_torso_neg.pt`, `ckpt_torso_neg_gentle.pt`) as documented experiments per the prior audit | **A** |
| 4 | `outputs/ball_finetuned/*.pth` — 17 of 19 checkpoints | **0.719** (17 x 43.3 MB) | TrackNetV2 epoch/generation snapshots (`_ep5`/`_ep10`/`_ep15`, v2/v3/v4 generations) | Re-run the fine-tune | None — verified `tracknetv2_v5.pth`/`tracknetv2_v6.pth` are the only paths referenced in code (`tools/pl_pilot_run.py`, `tools/pl_pilot_ball.py`, `tools/pl_carry_probe.py`, `tools/pl_ball_recall.py`, `tools/pl_feasibility.py`, `tools/regen_ball.py`) | **A** |
| 5 | `matches/pl_probe/` | **0.772** | 9 x ~88 MB feasibility clips cut from 3 fixtures, pre-registry | Re-cut from the corresponding full `matches/<fixture>/` chunks, minutes | None — absent from `data/matches.yaml`, no registered code path reads it | **A** |
| 6 | `results/demo/cv_pipeline_demo.mp4` + `identity_demo.mp4` | **0.195** (153.3+46.6 MB) | Full-res demo videos | `tools/make_demo.py` / `make_identity_demo.py` | None — `_720p` versions (43+19.3 MB) plus the poster jpg already exist and are what the demo page serves | **A** |
| 7 | pip HTTP cache (`%LOCALAPPDATA%\pip\cache\http-v2`) | **0.144** | 74 cached package index/wheel responses | `pip cache purge` re-fetches on next install | None | **A** |
| 8 | `%USERPROFILE%\.cache\huggingface\hub` (Qwen2-VL-2B-Instruct 4.125 GB + faster-whisper-small 0.453 GB) | **4.611** | HF model weight cache — Qwen2-VL is used by `tools/qwen_jersey_trial.py` (a jersey-OCR-via-VLM trial, not part of the frozen GSR/FOOTPASS pipeline) | Auto re-download from HF Hub on next run of that script (public weights, no NDA) | Low — only breaks `qwen_jersey_trial.py` until it re-fetches; **verify no run of that script is mid-flight before deleting** (the CPU worker noted as active in this task reads `outputs/`/`results/` caches, not this) | **B** — condition: confirm no in-flight `qwen_jersey_trial.py` run |
| 9 | `data/soccernet/jersey-2023/` | **4.281** | Jersey-number crop dataset (1.3M tiny files), the training corpus behind `outputs/jersey/*.pt` | Re-download with NDA password (memory: `soccernet-credentials.md`), ~1 h bandwidth | Blocks retraining the jersey recognizer until re-fetched; the trained weights (`jersey_torso_r224_acc417.pt`) already exist and its numbers are distilled in `results/jersey_model/JERSEY_MODEL.md` — nothing measured is lost | **B** — condition: confirm no further jersey-model retraining is planned this cycle; **archive externally rather than delete**, since it's the slowest re-fetch on this list |
| 10 | `matches/fra_sen/` | **1.908** (chunks 0.569 + play 1.339) | Old flat-chunk generation of the France-Senegal fixture. Absent from `data/matches.yaml`; the registry-canonical copy is `matches/france_senegal/` (1.973 GB, h1/h2 layout). Update from the 07-24 audit: `chunks/` no longer internally duplicates `play/` (now 4 files vs 10, i.e. they're complementary halves, not copies) | Content-match to `france_senegal` was NOT re-verified byte-for-byte this pass (sizes are close: 1.908 vs 1.973 GB, consistent but not proof) | If it turns out to be non-duplicate French WC footage, it is irreplaceable (never re-download copyrighted footage per standing rule) | **C — Sid's decision** (downgraded from the prior audit's "safe to delete": that call was made without a fresh byte-level check, and this rule explicitly forbids improvising the call) |
| 11 | Loose root PDFs + 2 stray mp4s (git status: `02072026_CSAS...pdf`, `A_team_ball_game...pdf`, `Untitled document (23).pdf`, `basketball paper.pdf`, `data driven detection...pdf`, `gnn approach.pdf`, `herold-et-al-2019...pdf`, `large svale analysi...pdf`, `lucey paper.pdf`, `positional data from broadcast footage.pdf`, `training free offscreen...pdf`, `Carrick's Preseason Tactics Are Unusual.mp4`, `what i wanted.mp4`) | **~0.76** | Reference papers + 2 misc clips dropped at repo root, all untracked (`git status` shows `??`) | N/A — these are Sid's own reference material, not pipeline output | None to the pipeline; only clutter (git status noise) | **C — Sid's decision** (relocate to `docs/papers/` or elsewhere, not a delete candidate — small and possibly wanted for the attribution/GSR reading list) |

## Tier D — flagged, not sized into reclaim (do not touch)

- `outputs/gsr/` (64 sub-dirs, 6.97 GB) / `outputs/gsr_test/` (1.49 GB): several eval-config
  dirs are near-identical in size to each other (many at exactly 0.287/0.186/0.1 GB — same
  sequence count, different re-linking/deleak config), which is the same "predictions are
  disposable, scores are distilled" pattern the 07-24 audit found in the old `outputs/gsr/eval_*`
  set. **Not sized into any reclaim tier here** because the task brief explicitly names
  `outputs/gsr*` as live campaign caches. If the campaign later declares a batch of these configs
  closed (the way `results/gsr_benchmark/gsr_scores*.json` already distills the old six), that
  would move roughly this much from D to A — but that call belongs to the campaign, not this scan.
- `.git/` — 0.064 GB, already `gc`'d since the 07-24 audit (was 1.85 GB). No action.

## Totals

| Tier | GB | Notes |
|---|---|---|
| A (safe now) | **14.70** | items 1-7, verified re-derivation for each |
| B (condition-gated) | **8.89** | items 8-9, need a one-line confirmation before acting |
| C (Sid's decision) | **2.67** | items 10-11, judgment calls this scan should not make |
| **A+B+C if all actioned** | **26.26** | would roughly double current free space (25.72 -> ~52 GB) |

## EXECUTED 2026-08-01

Tiers A and B actioned per Sid's authority ("do a+b then..."). Full row-by-row trail, including
re-verification results and skip reasons, is in `results/STORAGE_RECLAIM_LOG_2026-08-01.md`.

| # | Item | Tier | Status |
|---|---|---|---|
| 1 | `data/footpass/raw/videos_fullHD_VAL.zip` | A | DELETED |
| 2 | `results/closeup_anchor_probe/**/*.jpg`+`*.png` | A | DELETED (analysis files retained) |
| 3 | `outputs/jersey/*.pt` (7 of 10) | A | DELETED |
| 4 | `outputs/ball_finetuned/*.pth` (17 of 19) | A | DELETED |
| 5 | `matches/pl_probe/` | A | DELETED |
| 6 | `results/demo/cv_pipeline_demo.mp4`+`identity_demo.mp4` | A | DELETED |
| 7 | pip HTTP cache | A | DELETED |
| 8 | HF cache (Qwen2-VL + faster-whisper-small) | B | DELETED |
| 9 | `data/soccernet/jersey-2023/` | B | **SKIPPED** — report's own condition says archive externally rather than delete; no external drive connected this session |
| 10 | `matches/fra_sen/` | C | Not actioned (Sid's decision, out of scope) |
| 11 | Loose root PDFs/mp4s | C | Not actioned (Sid's decision, out of scope) |

**Freed: ~19.33 GB.** Free space on C: 30.45 GB -> 49.78 GB.

## Part 2 — full disk map (2026-08-01, post-reclaim, analysis only, nothing deleted here)

Top-level, `Get-ChildItem -Recurse -Force -File | Measure-Object Length -Sum` per directory.
Repo total ~120 GB post-reclaim (`matches` 68.50, `data` 38.78, `outputs` 12.03, `results` 0.71,
rest <0.1 GB combined). Every dir >= 0.5 GB, classified:

### ACTIVE (current campaign reads/writes)

| Dir | GB | Note |
|---|---|---|
| `data/soccernet/gamestate-2024/` | 30.848 | GSR campaign core, official test-split submission in flight |
| `outputs/gsr/` | 8.518 | live eval/deleak caches |
| `outputs/gsr_test/` | 1.752 | live eval/deleak caches |

### KEEP-LOCAL (needed, not active daily)

| Dir | GB | Note |
|---|---|---|
| `data/footpass/` | 2.676 | `tactical_data_{TRAIN,VAL}.zip` + digest outputs — the raw `videos_fullHD_VAL.zip` component was already removed this pass (Tier A row 1); what remains is read by `tools/footpass_digest.py` and backs the frozen `results/FOOTPASS_*` reports |
| `outputs/jersey/` | 0.377 | 3 checkpoints post-cleanup, all referenced or documented-experiment; below the 0.5 GB map threshold but listed for continuity with Tier A row 3 |

### MOVE-TO-EXTERNAL candidates (safe to relocate once Sid's drive connects)

| Dir | GB | Note |
|---|---|---|
| `matches/` ManU fixtures (12 registered, both filename orders) | **50.229** | `palace_manutd` 4.396, `southampton_manutd` 4.364, `manutd_brighton` 4.346, `manutd_palace` 4.322, `manutd_liverpool` 4.310, `fulham_manutd` 4.288, `liverpool_manutd` 4.283, `brighton_manutd` 4.269, `manutd_tottenham` 4.255, `manutd_southampton` 4.205, `manutd_fulham` 4.121, `tottenham_manutd` 4.065 — measured total is higher than the task brief's ~45 GB estimate; ManU thread is paused, irreplaceable broadcast footage |
| `matches/footpass_game_{18,24,47}/` | 10.393 | 3.347+3.625+3.421 GB — NDA footage, extracted working copies (see Tier A row 1) |
| `matches/france_senegal/` + `france_norway/` + `france_iraq/` | 4.972 | 1.973+1.647+1.352 GB — PMSR ground truth, France-era reference-only per pivot |

### DELETE-CANDIDATE (name the condition)

| Dir | GB | Condition |
|---|---|---|
| `matches/fra_sen/` | 1.909 | Old flat-chunk generation, likely superseded by `matches/france_senegal/` (1.973 GB) — sizes are close but **not byte-verified** this pass (per the original report's Tier-C downgrade: "that call was made without a fresh byte-level check"). Condition to delete: run a byte/frame-level diff against `france_senegal/` and confirm true duplication. Sid's decision, not actioned. |
| `data/soccernet/jersey-2023/` | 4.281 | Confirmed this pass: jersey tracklets are **free, no NDA needed** (per `soccernet-credentials` memory: "Labels, jersey tracklets, 2fps features, calibration/re-ID data are free and need no password") — re-downloadable directly, no NDA gate to reconfirm. Condition to delete outright (vs. archive): no further jersey-model retraining planned this cycle. Currently **skipped** in this pass because the original report recommends archive-over-delete and no external drive was connected — remains open as either an archive-and-delete or a straight delete once Sid decides. |
| Loose root PDFs/mp4s (git status `??`) | ~0.76 | Not pipeline output, Sid's reference material — relocate to `docs/papers/`, not a delete candidate; listed here only for completeness (untracked clutter) |

### Not sized into any DELETE class (explicitly live, repeated from Part 1 guard)

`outputs/gsr/` and `outputs/gsr_test/` contain same-size sibling sub-dirs (several eval configs
at near-identical GB) that look like re-run duplicates, but the campaign brief names these paths
untouchable while iterating — no sub-dir breakdown taken this pass to avoid any risk of touching
live state.

## Method

Sizes via `Get-ChildItem -Recurse -Force -File | Measure-Object Length -Sum` (GB = 1024^3). Code
references checked via `Grep` across `*.py`. Video duplication (item 1) verified via `ffprobe`
(codec/resolution/duration on sample chunks) and Python `zipfile` entry listing (no extraction).
Nothing was extracted to disk, moved, or deleted.
