# Google Drive upload manifest (2026-08-01)

Read-only prep. Nothing uploaded, nothing deleted, nothing modified outside this file. rclone is
installed (`rclone v1.75.0`, via `winget install Rclone.Rclone`) but **not configured** — Sid runs
`rclone config` himself to create the `gdrive` remote (interactive step, out of scope here).

## Rclone install status

```
> rclone version
rclone v1.75.0
- os/version: Microsoft Windows 11 Home 25H2 25H2 (64 bit)
- os/kernel: 10.0.26200.8875 (x86_64)
- os/arch: amd64
```

Installed via `winget install Rclone.Rclone --accept-source-agreements --accept-package-agreements
--disable-interactivity` (non-interactive, succeeded first try — no manual zip download needed).
`rclone` is on PATH (winget added the shim). `rclone config` was **not** run — remote `gdrive` does
not exist yet; that step is Sid's.

## Scope excluded per instruction

`matches/footpass_game_{18,24,47}/` is **excluded from this upload batch** — it is NDA-downloaded
SoccerNet FOOTPASS footage, re-downloadable from HuggingFace/SoccerNet with the credentials already
on file (`soccernet-credentials` memory), so it doesn't need to occupy Drive quota as a backup. (It
is still measured below for completeness / in case Sid wants numbers, but no upload command is given
for it.)

## Per-fixture table (a) — 12 ManU EPL 24-25 fixtures, `matches/<fixture>/`

Measured fresh this pass (`os.walk` + `os.path.getsize`, byte-exact) — supersedes the rounded
`STORAGE_ANALYSIS_2026-08-01.md` figure of 50.229 GB (that scan predates a small number of files
added since; current total is higher, see below).

| Fixture | Files | Size (GB) | Size (bytes) |
|---|---|---|---|
| brighton_manutd | 11 | 4.269 | 4,583,777,375 |
| fulham_manutd | 11 | 4.288 | 4,603,879,856 |
| liverpool_manutd | 11 | 4.283 | 4,598,959,042 |
| manutd_brighton | 11 | 4.346 | 4,666,695,364 |
| manutd_fulham | 10 | 4.121 | 4,425,048,517 |
| manutd_liverpool | 11 | 4.310 | 4,628,073,677 |
| manutd_palace | 11 | 4.322 | 4,640,533,376 |
| manutd_southampton | 10 | 4.205 | 4,514,729,033 |
| manutd_tottenham | 11 | 4.255 | 4,568,814,249 |
| palace_manutd | 12 | 4.396 | 4,719,960,999 |
| southampton_manutd | 11 | 4.364 | 4,686,150,441 |
| tottenham_manutd | 11 | 4.065 | 4,365,163,165 |
| **Total** | **131** | **51.224 GB** | 55,001,785,094 |

(All 12 are registered in `data/matches.yaml`, the full ManU EPL 2024-25 corpus per the project
pivot. `manutd_fulham` and `manutd_southampton` have 10 files each — one fewer than the h1/h2 +
mp4 + parquet siblings; not investigated here, out of scope for an upload-prep pass, flagged for
Sid/orchestrator if it matters.)

## Per-fixture table (b) — France-era reference footage, ~5.0 GB

Identified from `data/matches.yaml` (`france_iraq`, `france_norway`, `france_senegal` — the three
matches with `pmsr:` ground truth and `roster: true`, i.e. the FIFA-PMSR-calibrated reference set,
France/WC = reference-only per the 2026-07-17 pivot).

| Fixture | Files | Size (GB) | Size (bytes) |
|---|---|---|---|
| france_iraq | 12 | 1.352 | 1,451,311,680 |
| france_norway | 13 | 1.647 | 1,768,890,526 |
| france_senegal | 12 | 1.973 | 2,118,160,664 |
| **Total** | **37** | **4.972 GB** | 5,338,362,870 |

## Item (c) — `matches/fra_sen/` (1.909 GB) — duplicate check vs `france_senegal/`

**Verdict: NOT a duplicate of `matches/france_senegal/`.** Three independent checks, all
disagreeing:

1. **Structure differs.** `fra_sen/` = `chunks/` (4 files) + `play/` (10 files), 14 files total,
   2,049,288,961 B (1.909 GB). `france_senegal/` = `h1/` (6 files) + `h2/` (6 files), 12 files
   total, 2,118,160,664 B (1.973 GB). Different
   directory layout, different file count, different chunk-numbering convention (`chunk_000`
   through `chunk_013` spanning both an old "chunks" and "play" folder, vs a clean h1/h2 split).

2. **Encoding differs.** `ffprobe` on `chunk_000.mp4` in each:
   - `fra_sen/chunks/chunk_000.mp4`: h264, 1280x720, **30 fps**, duration 605.07s
   - `france_senegal/h1/chunk_000.mp4`: h264, 1280x720, **59.94 fps (60000/1001)**, duration 600.60s
   Same resolution, different frame rate — not a re-encode of the same source at a different
   bitrate; a genuinely different capture/transcode.

3. **Total duration differs by ~27%.** Summed `ffprobe` durations across all files:
   - `fra_sen` (chunks + play): 8,390.93 s = 139.85 min
   - `france_senegal` (h1 + h2): 6,604.46 s = 110.07 min
   `fra_sen` covers ~30 more minutes of footage than `france_senegal` — not the same match segment
   re-encoded, but a longer capture (likely includes broadcast pre/post-match coverage or a
   different cut).

4. **Byte hash (first 1 MB) of paired chunks — all distinct**, consistent with 1-3:
   - `fra_sen/chunks/chunk_000.mp4` -> `6f10af832701b5ff...`
   - `france_senegal/h1/chunk_000.mp4` -> `525e7a4e85ab3965...`
   - `fra_sen/play/chunk_004.mp4` -> `927be97d5c9c336c...`
   - `france_senegal/h1/chunk_003.mp4` -> `6f5faf83b2f5a986...`

Conclusion: `fra_sen/` is an older, differently-chunked, differently-encoded, longer capture of
France-Senegal — not a byte duplicate, not even a content duplicate at a different bitrate. Per the
"never re-download copyrighted footage" rule and the prior audit's retraction (STATUS.md: "the
audit's 'delete fra_sen' recommendation was RETRACTED — two of its clips are live ball-detector
training inputs"), **treat as irreplaceable and back up, not skip.**

| Dir | Files | Size (GB) | Size (bytes) |
|---|---|---|---|
| `matches/fra_sen/` | 14 | 1.909 | 2,049,288,961 |

## Excluded — FOOTPASS (for reference only, not uploaded)

| Fixture | Files | Size (GB) | Reason excluded |
|---|---|---|---|
| footpass_game_18 | 11 | 3.347 | Re-downloadable, SoccerNet NDA on file |
| footpass_game_24 | 12 | 3.625 | Re-downloadable, SoccerNet NDA on file |
| footpass_game_47 | 12 | 3.421 | Re-downloadable, SoccerNet NDA on file |
| **Total (not uploaded)** | 35 | 10.393 | — |

## Grand total for this upload batch

51.224 (ManU) + 4.972 (France) + 1.909 (fra_sen) = **58.105 GB** across 182 files, 16 directories.

## Drive folder layout

```
gdrive:football-synthesizer-archive/
  matches/
    brighton_manutd/
    fulham_manutd/
    liverpool_manutd/
    manutd_brighton/
    manutd_fulham/
    manutd_liverpool/
    manutd_palace/
    manutd_southampton/
    manutd_tottenham/
    palace_manutd/
    southampton_manutd/
    tottenham_manutd/
    france_iraq/
    france_norway/
    france_senegal/
    fra_sen/
```

One `rclone copy` per fixture directory (not one giant `matches/` copy) so each fixture can be
verified and freed independently, and a failed/interrupted transfer only has to resume one fixture.

## Upload commands (run after `rclone config` creates the `gdrive` remote)

Run from the repo root (`c:\Users\siddh_ygv5bws\football-synthesizer`). One fixture at a time —
default to sequential; `--transfers 2 --checkers 4` is the per-command parallelism cap, not
across-command.

```powershell
# ManU fixtures (12)
rclone copy matches/brighton_manutd      gdrive:football-synthesizer-archive/matches/brighton_manutd      --progress --transfers 2 --checkers 4
rclone copy matches/fulham_manutd        gdrive:football-synthesizer-archive/matches/fulham_manutd        --progress --transfers 2 --checkers 4
rclone copy matches/liverpool_manutd     gdrive:football-synthesizer-archive/matches/liverpool_manutd     --progress --transfers 2 --checkers 4
rclone copy matches/manutd_brighton      gdrive:football-synthesizer-archive/matches/manutd_brighton      --progress --transfers 2 --checkers 4
rclone copy matches/manutd_fulham        gdrive:football-synthesizer-archive/matches/manutd_fulham        --progress --transfers 2 --checkers 4
rclone copy matches/manutd_liverpool     gdrive:football-synthesizer-archive/matches/manutd_liverpool     --progress --transfers 2 --checkers 4
rclone copy matches/manutd_palace        gdrive:football-synthesizer-archive/matches/manutd_palace        --progress --transfers 2 --checkers 4
rclone copy matches/manutd_southampton   gdrive:football-synthesizer-archive/matches/manutd_southampton   --progress --transfers 2 --checkers 4
rclone copy matches/manutd_tottenham     gdrive:football-synthesizer-archive/matches/manutd_tottenham     --progress --transfers 2 --checkers 4
rclone copy matches/palace_manutd        gdrive:football-synthesizer-archive/matches/palace_manutd        --progress --transfers 2 --checkers 4
rclone copy matches/southampton_manutd   gdrive:football-synthesizer-archive/matches/southampton_manutd   --progress --transfers 2 --checkers 4
rclone copy matches/tottenham_manutd     gdrive:football-synthesizer-archive/matches/tottenham_manutd     --progress --transfers 2 --checkers 4

# France-era reference (3)
rclone copy matches/france_iraq          gdrive:football-synthesizer-archive/matches/france_iraq          --progress --transfers 2 --checkers 4
rclone copy matches/france_norway        gdrive:football-synthesizer-archive/matches/france_norway        --progress --transfers 2 --checkers 4
rclone copy matches/france_senegal       gdrive:football-synthesizer-archive/matches/france_senegal       --progress --transfers 2 --checkers 4

# fra_sen (confirmed NOT a duplicate of france_senegal, above)
rclone copy matches/fra_sen              gdrive:football-synthesizer-archive/matches/fra_sen              --progress --transfers 2 --checkers 4
```

Or, to run all 16 sequentially in one shot (PowerShell):

```powershell
$fixtures = @(
  "brighton_manutd","fulham_manutd","liverpool_manutd","manutd_brighton","manutd_fulham",
  "manutd_liverpool","manutd_palace","manutd_southampton","manutd_tottenham","palace_manutd",
  "southampton_manutd","tottenham_manutd","france_iraq","france_norway","france_senegal","fra_sen"
)
foreach ($f in $fixtures) {
  rclone copy "matches/$f" "gdrive:football-synthesizer-archive/matches/$f" --progress --transfers 2 --checkers 4
}
```

## Post-upload verification

Per fixture, after its `rclone copy` finishes:

```powershell
rclone check matches/<fixture> gdrive:football-synthesizer-archive/matches/<fixture> --one-way
```

`--one-way` checks every local file exists and matches on Drive (doesn't fail on Drive-side extras).
Only delete the local copy after this returns clean (0 differences) — deletion is explicitly out of
scope for this task and for Sid to decide/run separately.

## Restore command template (for later, if a fixture is deleted locally after upload)

```powershell
rclone copy gdrive:football-synthesizer-archive/matches/<fixture> matches/<fixture> --progress --transfers 2 --checkers 4
```

Then re-run the same `rclone check ... --one-way` to confirm the restore is byte-complete before
resuming any pipeline work on that fixture.

## Bandwidth throttling note

- **Daytime (Sid working, needs headroom):** add `--bwlimit 8M` (roughly 8 MB/s = 64 Mbps cap,
  adjust to taste) so uploads don't starve foreground use.
- **Overnight / unattended:** omit `--bwlimit` (full speed) or set a generous cap like `--bwlimit
  50M`; also consider `rclone copy ... --bwlimit "08:00,512k 23:00,off"` (schedule syntax) to
  auto-throttle during work hours and run free overnight, if this becomes a recurring upload.

## Honest time estimates (58.105 GB total, no live bandwidth test — Drive requires the `gdrive`
remote/auth to test, which is Sid's step)

| Upload speed | Time for 58.105 GB |
|---|---|
| 10 Mbps (1.25 MB/s) | ~46,484 s = ~12.9 hours |
| 20 Mbps (2.5 MB/s) | ~23,242 s = ~6.5 hours |
| 50 Mbps (6.25 MB/s) | ~9,297 s = ~2.6 hours |

(58.105 GB = 62,395 MB; time = MB / (Mbps/8). Real-world rclone-to-Drive throughput is usually
somewhat below raw line speed due to per-file API overhead — expect the low end of each bracket in
practice, especially with 131+ small-ish per-fixture files.)

## Confirmation

Nothing deleted. Nothing uploaded (no `gdrive` remote configured — `rclone config` intentionally
not run, per instruction). Nothing modified outside: this file
(`results/GDRIVE_UPLOAD_MANIFEST.md`, new) and the rclone install itself (winget package, does not
touch the repo).

## Next command for Sid to run

```powershell
rclone config
```

then create a remote named `gdrive` (type `drive`, follow the OAuth browser prompt), then run the
per-fixture `rclone copy` commands above.
