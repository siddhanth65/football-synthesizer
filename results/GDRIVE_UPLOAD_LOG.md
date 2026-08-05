# Google Drive upload log

Batch started: 08/01/2026 14:34:33  (remote: gdrive:football-synthesizer-archive)

| dir | GB | copy exit | check | duration |
|---|---|---|---|---|
| brighton_manutd | 4.27 | 0 | OK | 12.1 min |
| fra_sen | 1.91 | 0 | OK | 40.2 min |
| france_iraq | 1.35 | 0 | OK | 62.6 min |
| france_norway | 1.65 | 0 | OK | 50.4 min |
| france_senegal | 1.97 | 0 | OK | 15.1 min |
| fulham_manutd | 4.29 | 0 | OK | 63 min |
| liverpool_manutd | 4.28 | 0 | OK | 48.7 min |
| manutd_brighton | 4.35 | 0 | OK | 49.2 min |
| manutd_fulham | 4.12 | 0 | OK | 33.7 min |
| manutd_liverpool | 4.31 | 0 | OK | 20.1 min |
| manutd_palace | 4.32 | 0 | OK | 57.3 min |
| manutd_southampton | 4.2 | 0 | OK | 38.8 min |
| manutd_tottenham | 4.26 | 0 | OK | 46.8 min |
| palace_manutd | 4.4 | 0 | OK | 6 min |
| southampton_manutd | 4.36 | 0 | OK | 15.3 min |
| tottenham_manutd | 4.07 | 0 | OK | 22.7 min |

Batch finished: 08/02/2026 00:16:41  total 582 min, failures: 0
Nothing was deleted locally. Restore template: rclone copy gdrive:football-synthesizer-archive/matches/<dir> matches/<dir>

## LOCAL DELETION 2026-08-02

Authority: Sid, on record ("can we now delete them from the system?") after the verified upload
batch above (16/16 dirs, rclone check OK, 0 failures). Protocol: verify-then-delete, abort on any
mismatch.

### 1. Spot-restore test (SHA-256, local original vs fresh download from gdrive)

| file | local sha256 | downloaded sha256 | match |
|---|---|---|---|
| matches/brighton_manutd/h2/chunk_005.mp4 (78.4 MiB) | 724378e2...9cce3b7d | 724378e2...9cce3b7d | YES |
| matches/france_iraq/h1/chunk_004.mp4 (116.6 MiB) | 33932efd...d458abd9482 | 33932efd...d458abd9482 | YES |

Both files identical byte-for-byte. Verdict: PASS.

### 2. Per-dir size confirmation (local `du -sb` vs `rclone size --json`, bytes)

| dir | local bytes | remote bytes | match |
|---|---|---|---|
| brighton_manutd | 4583777375 | 4583777375 | YES |
| fra_sen | 2049288961 | 2049288961 | YES |
| france_iraq | 1451311680 | 1451311680 | YES |
| france_norway | 1768890526 | 1768890526 | YES |
| france_senegal | 2118160664 | 2118160664 | YES |
| fulham_manutd | 4603879856 | 4603879856 | YES |
| liverpool_manutd | 4598959042 | 4598959042 | YES |
| manutd_brighton | 4666695364 | 4666695364 | YES |
| manutd_fulham | 4425048517 | 4425048517 | YES |
| manutd_liverpool | 4628073677 | 4628073677 | YES |
| manutd_palace | 4640533376 | 4640533376 | YES |
| manutd_southampton | 4514729033 | 4514729033 | YES |
| manutd_tottenham | 4568814249 | 4568814249 | YES |
| palace_manutd | 4719960999 | 4719960999 | YES |
| southampton_manutd | 4686150441 | 4686150441 | YES |
| tottenham_manutd | 4365163165 | 4365163165 | YES |

All 16 dirs match byte-for-byte, total 62,389,436,925 bytes (62.39 GB). Verdict: PASS.

Pre-delete process check: `wmic process where "name='python.exe'" get CommandLine` showed only an
MCP tool-server process and a VSCode black-formatter LSP running -- neither touches `matches/`.
Cleared to proceed.

### 3. Deletion (all 16 dirs, `rm -rf matches/<dir>`, verified removed after each)

| dir | deleted at (UTC) |
|---|---|
| brighton_manutd | 2026-08-02T01:54:32Z |
| fra_sen | 2026-08-02T01:54:32Z |
| france_iraq | 2026-08-02T01:54:32Z |
| france_norway | 2026-08-02T01:54:32Z |
| france_senegal | 2026-08-02T01:54:32Z |
| fulham_manutd | 2026-08-02T01:54:32Z |
| liverpool_manutd | 2026-08-02T01:54:32Z |
| manutd_brighton | 2026-08-02T01:54:32Z |
| manutd_fulham | 2026-08-02T01:54:32Z |
| manutd_liverpool | 2026-08-02T01:54:32Z |
| manutd_palace | 2026-08-02T01:54:32Z |
| manutd_southampton | 2026-08-02T01:54:32Z |
| manutd_tottenham | 2026-08-02T01:54:33Z |
| palace_manutd | 2026-08-02T01:54:33Z |
| southampton_manutd | 2026-08-02T01:54:33Z |
| tottenham_manutd | 2026-08-02T01:54:33Z |

`matches/footpass_game_18`, `matches/footpass_game_24`, `matches/footpass_game_47` NOT touched
(not part of this upload batch, separate lifecycle). Spot-check scratchpad downloads deleted after
hash comparison.

`data/matches.yaml` NOT edited -- registry entries stay; the fixtures still exist, they live in
Drive now. A worker resuming any of these 16 threads must restore first (see restore template
above).

### Free space (C: drive)

- Before deletion: 44,376,911,872 bytes free (41.3 GiB)
- After deletion: 106,972,069,888 bytes free (99.6 GiB)
- Reclaimed: ~62.6 GB, consistent with the 62.39 GB total dir size above.

### Restore instructions (repeated)

```
rclone copy gdrive:football-synthesizer-archive/matches/<dir> matches/<dir>
```
Applies to any of the 16 dirs listed above. All are confirmed present and byte-identical on Drive.
