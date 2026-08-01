# Storage reclaim execution log — 2026-08-01

Source: `results/STORAGE_ANALYSIS_2026-08-01.md`, Tiers A and B only. Authority: Sid ("do a+b
then but ensure that it doesnt affect our current work and also make sure we have a record of
all these changes").

**Free space before: 30.45 GB.** **Free space after: 49.78 GB.** **Freed: ~19.3 GB.**

## Live-work guard (checked before any deletion)

- Confirmed via `Get-CimInstance Win32_Process` (all `python.exe`/`pythonw.exe` command lines
  inspected) that no process was reading `data/soccernet/gamestate-2024/`, `outputs/gsr/`,
  `outputs/gsr_test/`, or running `tools/qwen_jersey_trial.py` at execution time. Running
  processes were: VS Code's black-formatter LSP, 4x Claude windows-mcp helpers, and an unrelated
  scratchpad script (`run_train_probe.py`, a different subagent's temp file, left untouched).
- None of the executed rows fall under `data/soccernet/gamestate-2024/`, `outputs/gsr/`, or
  `outputs/gsr_test/` — verified by path inspection before each deletion.

## Executed rows

| # | Path | Size | Tier | Justification (quoted) | Re-derivation route | Verification | Result |
|---|---|---|---|---|---|---|---|
| 1 | `data/footpass/raw/videos_fullHD_VAL.zip` | 10.565 GB | A | "not actually needed: `matches/footpass_game_{18,24,47}/{h1,h2}/chunk_*.mp4` is already the extracted, working copy" | Re-download via SoccerNet FOOTPASS NDA | `grep` of `tools/footpass_digest.py`: `RAW_DIR` only reads `tactical_data_{TRAIN,VAL}.zip`, never this file (confirmed fresh). `ffprobe` on `matches/footpass_game_{18,24,47}/h1/chunk_000.mp4`: all three report `h264,1920,1080,25/1` — consistent broadcast-quality extracted copies present. | **DELETED** |
| 2 | `results/closeup_anchor_probe/**/*.jpg` + `*.png` | 1.424 GB (199,891 files) | A | "Re-run the probe... the analysis outputs... are separate files (9.4 MB) and are retained" | `tools/closeup_anchor_probe.py`, ~2h GPU/match x 9 matches | Confirmed post-delete: 140 non-image files remain (`.json`, `.md`, `.csv`, `.log` — `CLOSEUP_ANCHORS.md`, `closeup_reads.csv`, `kit_centroids.json`, all `levers/`/`koshkina*/`/`spotcheck*/` dirs), 0 `.jpg`/`.png` remain. | **DELETED** |
| 3 | `outputs/jersey/*.pt` (7 of 10) | 0.877 GB | A | "verified `jersey_torso_r224_acc417.pt` is the only path hardcoded in code... keep it + the two `_neg` variants" | Re-train per `results/jersey_model/JERSEY_MODEL.md` | `grep` of `tools/carrier_constrained.py` (line 543) and `tools/closeup_anchor_probe.py` (line 53): both hardcode only `jersey_torso_r224_acc417.pt`. | **DELETED** (`ckpt.pt`, `ckpt_mh.pt`, `ckpt_mh2.pt`, `ckpt_r128_acc367.pt`, `ckpt_torso.pt`, `jersey_mh_r224_acc396.pt`, `jersey_r224_acc396.pt`). **KEPT**: `jersey_torso_r224_acc417.pt`, `ckpt_torso_neg.pt`, `ckpt_torso_neg_gentle.pt`. |
| 4 | `outputs/ball_finetuned/*.pth` (17 of 19) | 0.719 GB | A | "verified `tracknetv2_v5.pth`/`tracknetv2_v6.pth` are the only paths referenced in code" | Re-run the fine-tune | `grep -l` across the 6 named tools confirmed only `tracknetv2_v5.pth`/`tracknetv2_v6.pth` referenced. | **DELETED** 17 files. **KEPT**: `tracknetv2_v5.pth`, `tracknetv2_v6.pth`. |
| 5 | `matches/pl_probe/` | 0.772 GB | A | "absent from `data/matches.yaml`, no registered code path reads it" | Re-cut from `matches/<fixture>/` chunks | `grep pl_probe data/matches.yaml` returned no match (exit 1 = absent). | **DELETED** |
| 6 | `results/demo/cv_pipeline_demo.mp4` + `identity_demo.mp4` | 0.195 GB | A | "`_720p` versions... plus the poster jpg already exist and are what the demo page serves" | `tools/make_demo.py` / `make_identity_demo.py` | Confirmed `cv_pipeline_demo_720p.mp4`, `identity_demo_720p.mp4`, `identity_demo_poster.jpg` present before deletion; `grep` of `demo/` for the full-res filenames found no references. | **DELETED** |
| 7 | `%LOCALAPPDATA%\pip\cache\http-v2` | 0.144 GB (0.134 GB measured) | A | "74 cached package index/wheel responses... `pip cache purge` re-fetches on next install" | `pip` auto re-fetches | Directory existed at stated path, size matched report (~0.13-0.14 GB). | **DELETED** |
| 8 | `%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct` + `models--Systran--faster-whisper-small` | 4.578 GB (4.125 + 0.453) | B | "only breaks `qwen_jersey_trial.py` until it re-fetches; verify no run of that script is mid-flight before deleting" | HF Hub auto-download (public weights, no NDA) | Process list checked (see guard above): no `qwen_jersey_trial.py` process running. Sizes matched report exactly. Two other unrelated cached models in the same `hub/` dir (`models--timm--regnety_002.pycls_in1k`, 0.012 GB; `models--uisikdag--yolo-v8-football-players-detection`, 0.021 GB) were **left untouched** — not named in the report, out of scope. | **DELETED** (Qwen2-VL + faster-whisper-small only) |

## Skipped rows

| # | Path | Size | Tier | Reason skipped |
|---|---|---|---|---|
| 9 | `data/soccernet/jersey-2023/` | 4.281 GB | B | Report's own condition text says "**archive externally rather than delete**, since it's the slowest re-fetch on this list." No external drive is connected in this session (Part 2 of this task notes the drive connects later for the MOVE-TO-EXTERNAL map). Deleting without the archive step contradicts the report's stated safety condition, so per the execution rule ("if a verification fails, SKIP the row and record why") this row was skipped. Re-derivation remains valid (NDA re-download, ~1h) if Sid decides to delete outright instead of archiving. |

Rows 10-11 (Tier C, `matches/fra_sen/` and loose root PDFs/mp4s) were explicitly out of scope
("Tier A and B only") and were not touched.

## Totals

- Tier A executed: rows 1-7 = 10.565 + 1.424 + 0.877 + 0.719 + 0.772 + 0.195 + 0.144 = **14.696 GB**
- Tier B executed: row 8 = **4.578 GB**
- **Total freed (measured): 19.33 GB** (30.45 GB -> 49.78 GB free on C:)
- Tier B skipped: row 9 = 4.281 GB not freed (see reason above)

Measured free-space delta (19.33 GB) is close to the sum of executed rows (19.27 GB); the small
excess is consistent with filesystem cluster overhead differences reported vs. actual on-disk
allocation for the ~200k small image files in row 2.
