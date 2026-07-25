# Deletion verification -- repo-root .mp4 vs matches/<id> chunks (2026-07-24)

**READ-ONLY. Nothing was deleted, moved, or modified.** This document is the evidence Sid asked for
before irreversibly removing ~51 GB of root .mp4 files. Every number below was measured today with
`ffprobe` / `ffmpeg` from
`C:\Users\siddh_ygv5bws\OneDrive\Desktop\ffmpeg-2026-05-28-git-7b46c6a2a3-essentials_build\bin\`.

Scope check first: **the repo root contains exactly 12 video files, all `.mp4`, all Man Utd 24/25
replays.** There is no `fra-sen half1.mp4`, no other France/WC material, and no `.mkv/.mov/.ts/.avi`
in the root. Every one of the 12 maps to a chunk set. **No root .mp4 is an only-copy orphan.**

---

## 1. Method and pass criteria

Per fixture, five independent lines of evidence:

1. **Container/stream probe** -- `ffprobe -show_entries format=... :stream=...` on the source and on
   every chunk: duration, codec, profile, resolution, `r_frame_rate`, `nb_frames`, stream count,
   audio codec, bitrate.
2. **Frame arithmetic** -- `sum(chunk nb_frames) - source nb_frames`. This is the decisive number:
   a gap in coverage makes it **negative**; an overlap makes it positive.
3. **Container-box walk** -- top-level MP4 boxes read by seek (no media read) to separate `mdat`
   (media payload) from `moov`+`ftyp`+`free` (index/overhead). Answers "missing footage or container
   overhead?" directly, in bytes.
4. **Pixel identity at the boundaries** -- md5 of decoded rgb24 for the source's first 25 frames vs
   the first chunk's first 25 frames, and the source's last 1 s vs the last chunk's last 1 s.
5. **Halftime seam** -- is the last frame of `h1`'s final chunk present inside `h2/chunk_000`'s
   opening (overlap) or absent (gap)?

Plus integrity: head (`-t 2`) and tail (`-sseof -2`) decode of **every** chunk with `-v error`, and
the corroborating fact of whether the fixture ran end-to-end through the CV pipeline.

**PASS criteria (all four must hold):** `|sum(chunks) - source| < 2 s` AND `< 0.1 %`; codec /
profile / resolution / fps / stream layout identical; chunk count consistent with the outputs tree;
no chunk 0-length or unreadable.

Chunking convention (`tools/chunk_video.py`): `ffmpeg -c copy -map 0 -f segment -segment_time 600
-reset_timestamps 1` -- stream copy, no re-encode. Each fixture was split into halves first (manual
halftime cut, see `data/matches.yaml` comments and STATUS.md), then each half chunked, giving
`matches/<id>/h1/chunk_NNN.mp4` + `h2/chunk_NNN.mp4`.

---

## 2. Per-fixture verdict table

Durations in seconds (container `format=duration`, h1+h2 summed). `d_frames` =
`sum(chunk nb_frames) - source nb_frames`. `d_bytes` = total chunk bytes - source bytes.

| Source .mp4 (repo root) | Size (GiB) | Chunk dir | Chunks | Src dur | Chunk dur sum | Delta (s) | Delta % | d_frames | d_bytes | codec/res/fps | Integrity | Pipeline-processed | VERDICT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Brighton v Manchester United 2024-25 Full Match Replay.mp4` | 4.267 | `matches/brighton_manutd` (5+6) | 11 | 6001.94 | 6003.19 | +1.25 | +0.0209 % | +28 | +1,958,326 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Crystal Palace v Manchester United 2024-25 Full Match Replay.mp4` | 4.394 | `matches/palace_manutd` (6+6) | 12 | 6376.92 | 6378.09 | +1.17 | +0.0183 % | +28 | +2,030,473 | identical | clean | yes, 12/12 dense | **SAFE TO DELETE** |
| `Fulham v Manchester United 2024-25 Full Match Replay.mp4` | 4.286 | `matches/fulham_manutd` (5+6) | 11 | 6033.94 | 6034.20 | +0.26 | +0.0043 % | +3 | +1,424,167 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Liverpool v Manchester United 2024-25 Full Match Replay.mp4` | 4.282 | `matches/liverpool_manutd` (5+6) | 11 | 6008.96 | 6009.22 | +0.26 | +0.0043 % | +3 | +1,446,668 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Manchester United v Brighton 2024-25 Full Match Replay.mp4` | 4.345 | `matches/manutd_brighton` (5+6) | 11 | 6113.94 | 6114.20 | +0.26 | +0.0042 % | +3 | +1,629,725 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Manchester United v Crystal Palace 2024-25 Full Match Replay.mp4` | 4.320 | `matches/manutd_palace` (5+6) | 11 | 6085.95 | 6086.21 | +0.26 | +0.0042 % | +3 | +1,445,063 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Manchester United v Fulham 2024-25 Full Match Replay (1).mp4` | 4.122 | `matches/manutd_fulham` (5+5) | 10 | 5784.96 | 5785.20 | +0.24 | +0.0042 % | +3 | **-602,696** | identical | 1 warning, resolved (S4) | yes, 10/10 dense | **SAFE TO DELETE** (see S4) |
| `Manchester United v Liverpool 2024-25 Full Match Replay.mp4` | 4.309 | `matches/manutd_liverpool` (5+6) | 11 | 6065.94 | 6066.20 | +0.26 | +0.0043 % | +3 | +1,343,223 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Manchester United v Southampton 2024-25 Full Match Replay (1).mp4` | 4.203 | `matches/manutd_southampton` (5+5) | 10 | 5905.94 | 5906.20 | +0.26 | +0.0044 % | +3 | +1,628,649 | identical | clean | yes, 10/10 dense | **SAFE TO DELETE** |
| `Manchester United v Tottenham Hotspur 2024-25 Full Match Replay.mp4` | 4.255 | `matches/manutd_tottenham` (5+6) | 11 | 6100.96 | 6101.08 | +0.12 | +0.0020 % | +3 | +382,747 | identical (video-only source, 1 stream; chunks also 1 stream) | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Southampton v Manchester United 2024-25 Full Match Replay.mp4` | 4.363 | `matches/southampton_manutd` (5+6) | 11 | 6129.94 | 6130.20 | +0.26 | +0.0042 % | +3 | +1,419,529 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |
| `Tottenham Hotspur v Manchester United 2024-25 Full Match Replay.mp4` | 4.065 | `matches/tottenham_manutd` (6+5) | 11 | 5708.97 | 5710.11 | +1.14 | +0.0199 % | +28 | +16,952 | identical | clean | yes, 11/11 dense | **SAFE TO DELETE** |

**Every fixture meets every PASS criterion.**

- Max duration delta **+1.25 s** (brighton_manutd), max relative delta **+0.0209 %** -- both an order
  of magnitude inside the 2 s / 0.1 % bar. **All deltas are positive**: the chunk sets are longer
  than their sources, never shorter.
- `d_frames` is **positive in all 12** (+3 or +28). No fixture is missing a single video frame.
- Stream parameters are **byte-for-byte identical** in all 12: `h264 / High / 1920x1080 / 25 fps`,
  same stream count, same audio codec (`aac`, except `manutd_tottenham` whose source has no audio
  track at all -- its chunks correctly also have none).
- No chunk is 0-length; no chunk is unreadable; all 130 chunks probe and decode.
- Chunk counts match the outputs tree **exactly**: 130 chunks across 12 fixtures, 130
  `outputs/<id>/<half>/match/chunk_NNN_dense.parquet` files. Per-half counts agree in every case.

### 2.1 Boundary pixel identity (the strong form of "same footage")

For all 12 fixtures, decoded-pixel md5 of:

- source frames 0-24 == `h1/chunk_000` frames 0-24 -> **MATCH, 12/12**
- source last 1 s == last chunk's last 1 s -> **MATCH, 12/12**

So each chunk set starts on the source's exact first frame and ends on the source's exact last
frame. Combined with the positive frame deltas, that leaves the halftime seam as the only possible
place footage could have gone missing.

### 2.2 Halftime seam -- overlap, never gap

`h1`'s final decoded frame was found **inside `h2/chunk_000`'s opening in all 12 fixtures**. The
halves overlap at the cut; they do not abut with a hole. (The reported overlap index is an upper
bound -- the seam region contains repeated identical frames, e.g. a frozen graphic, so a pixel-hash
lookup can land early. The exact overlap is given by `d_frames`: 3 frames = 0.12 s for nine
fixtures, 28 frames = 1.12 s for brighton_manutd / palace_manutd / tottenham_manutd.)

### 2.3 Integrity decode

167 chunks checked (the 12 ManU fixtures = 130, plus `france_senegal` 12 and `fra_sen` 25 for
section 4): head `-t 2` and tail `-sseof -2` decoded with `ffmpeg -v error`. **166 fully silent,
return code 0.** One warning, resolved in S4 below. Zero truncated or unreadable files.

### 2.4 Two source-side oddities that are NOT chunking defects

- **`palace_manutd`**: the *source* has container duration 6376.92 s but a video stream of only
  6167.64 s -- roughly 209 s of audio-only tail. The chunks reproduce this faithfully
  (`h2/chunk_005`: video 33.64 s, audio 242.96 s). Video frames: source 154,188, chunks 154,216.
  Nothing lost.
- **`manutd_tottenham`**: the source has **one stream, no audio**. Its chunks likewise. Not a
  stripping defect -- the source never had audio.

---

## 3. Corroboration: these chunks have already been decoded end-to-end

All 12 fixtures have `outputs/<id>/h1/match_dense.parquet` **and** `outputs/<id>/h2/match_dense.parquet`,
plus a per-chunk `chunk_NNN_dense.parquet` for **every one of the 130 chunks**. The CV pipeline has
therefore fully decoded every chunk in the deletion set at least once, frame by frame, and produced
validated tracking output from it. This is stronger evidence of decodability than any probe.

---

## 4. S4 -- the `manutd_fulham` case, resolved definitively

**Question:** the storage audit found `manutd_fulham` chunks summing ~1 MB *below* the source (the
only fixture that way round). Missing footage, or container overhead?

**Answer: container overhead. Specifically, an oversized `moov` index in the source. No footage is
missing -- the chunks in fact carry MORE media payload than the source file does.**

Measured (top-level MP4 box walk, seek-only):

| | source | 10 chunks combined | delta |
|---|---|---|---|
| `mdat` (media payload) | 4,420,988,198 B | **4,421,457,185 B** | **+468,987 B** |
| `moov` + `ftyp` + `free` (index/overhead) | 4,663,015 B | 3,591,332 B | -1,071,683 B |
| file total | 4,425,651,213 B | 4,425,048,517 B | -602,696 B |

The source's `moov` is **4.66 MB** where its sibling replays carry ~2.7 MB (`tottenham_manutd`'s
source is the other 4.6 MB outlier -- a different muxer generation among Sid's downloads). Splitting
into 10 files rebuilds ten smaller sample-index tables totalling 3.59 MB. That 1.07 MB index saving
outweighs the 0.47 MB of extra media, producing the apparent -0.6 MB. **The byte deficit is index
metadata, not video.**

Supporting facts, all pointing the same way:

- Frames: source 144,624, chunks **144,627 (+3)**. Not one frame short.
- Duration: source 5784.96 s, chunks 5785.20 s (**+0.24 s**, +0.0042 %).
- **Time gap between consecutive chunks: none.** Within each half the segment muxer writes a
  continuous stream. At the halftime seam, `h1/chunk_004`'s last frame reappears inside
  `h2/chunk_000`'s opening -- an overlap of 3 frames (0.12 s), not a gap. Per-chunk h1 durations are
  600/600/600/600/386.12 (= 2786.12 s) and h2 601.68/600.44/598.72/601.28/596.96 (= 2999.08 s);
  h1 + h2 = 5785.20 s vs source 5784.96 s.
- Pixel identity: source's first 25 frames == `h1/chunk_000`'s first 25 frames; source's last second
  == `h2/chunk_004`'s last second.
- Bit-for-bit stream parity: h264 High 1920x1080 25 fps, 2 streams, aac -- identical in source and
  all 10 chunks.

**The one integrity warning, also resolved.** `matches/manutd_fulham/h2/chunk_003.mp4` emitted, on
decode:

```
[h264] mmco: unref short failure
[h264] number of reference frames (0+5) exceeds max (4; probably corrupt input), discarding one
```

Return code 0. A **full** decode of that chunk (all 15,032 frames) emits exactly the same two
warnings and nothing else, rc 0. Decoding the **source file** at the corresponding timestamp
(`-ss 4586.8 -t 3`) emits **the identical pair of warnings**. The defect is in Sid's original
download's H.264 reference-frame management and was copied verbatim by the stream copy. Deleting the
source removes nothing that would fix it. This chunk also processed successfully through the
pipeline (`outputs/manutd_fulham/h2/match/chunk_003_dense.parquet` exists).

**`manutd_fulham` verdict: SAFE TO DELETE.**

---

## 5. `matches/fra_sen/` -- the audit's "superseded, safe to delete" call is WRONG

**VERDICT: DO NOT DELETE `matches/fra_sen/` wholesale. It contains live, referenced training data
that is NOT reproducible from `matches/france_senegal/`.**

### 5.1 Reference audit -- every hit for "fra_sen" outside `.git/`

| file:line | reference | live or historical? |
|---|---|---|
| **`data/ball_annotations/manifest.csv:5`** | `fra_sen,chunk_004,data/ball_annotations/fra_sen/chunk_004.csv,`**`matches/fra_sen/play/chunk_004.mp4`**`,outputs/fra_sen/match/chunk_004_dense.parquet` | **LIVE -- code reads this video** |
| **`data/ball_annotations/manifest.csv:6`** | same for **`matches/fra_sen/play/chunk_005.mp4`** | **LIVE -- code reads this video** |
| **`data/ball_annotations/holdout_split.csv:155-166`** | 12 held-out annotated rows, all citing `matches/fra_sen/play/chunk_004.mp4` / `chunk_005.mp4` | **LIVE -- the reproducible validation split** |
| `data/ball_annotations/manifest.pre_v6.bak:5-6` | same two rows, pre-v6 backup | historical backup of a live file |
| `data/ball_annotations/holdout_split.pre_v6.bak:155-166` | same 12 rows | historical backup |
| `tools/chunk_video.py:7` | `--out-dir matches/fra_sen/chunks` in the module docstring | docstring example only, not executed |
| `tools/c6_validate.py:68` | writes `results/fra_sen_c6.png` | output filename only, no `matches/` path |
| `tools/c6_phase_graphic.py:37` | writes `results/fra_sen_phase_c6.png` | output filename only |
| `tools/fra_sen_c6_final.py:11,48` | hard-coded per-chunk def_line numbers + `results/fra_sen_c6_final.png` | numbers already inlined; no video read |
| `tools/analyze_match.py:8` | docstring mentions `tools/analyze_fra_sen.py` | stale docstring, that file no longer exists |
| `tools/parse_pmsr.py:9` | `--out outputs/pmsr/fra_sen.json` in docstring | docstring example only |
| `docs/BALL_CORPUS.md:14` | `fra_sen/chunk_004.csv` in a layout illustration | doc |
| `STATUS.md:1140,1250,1542,1543,1571,1588,1613` | historical log entries | historical |
| `results/pl_probe/feasibility.md:41` | a benchmark row | historical |
| `results/STORAGE_AUDIT_2026-07-24.md` (9 hits) | the audit that proposed deleting it | the claim under review |

**The live path:** `tools/finetune_ball.py::read_manifest` keeps every manifest row **whose
annotation CSV exists on disk**; `data/ball_annotations/fra_sen/chunk_004.csv` (860 B) and
`chunk_005.csv` (1,916 B) both exist. Those rows are then fed to `build_dataset`, which opens the
row's `video` with OpenCV and decodes windows around each annotated frame. So
`matches/fra_sen/play/chunk_004.mp4` and `chunk_005.mp4` are **input files to the ball fine-tune
that produced `tracknetv2_v5.pth` / `tracknetv2_v6.pth`** -- the two production ball weights.
Deleting them silently removes two annotated chunks from the ball training corpus and makes the
recorded `holdout_split.csv` unreproducible.

Every video path in `manifest.csv` currently resolves on disk (18/18), including three
`matches/pl_probe/<fixture>/seg_1.mp4` entries -- **so the audit's "pl_probe: safe to delete,
nothing references it" is wrong for the same reason.** Flagged, out of scope here.

### 5.2 Is `fra_sen` reproducible from `france_senegal`? NO -- they are different source encodes

| | `matches/fra_sen/chunks/` | `matches/france_senegal/` |
|---|---|---|
| layout | flat `chunk_000..013` | `h1/` 6 + `h2/` 6 |
| resolution | 1280x720 | 1280x720 |
| frame rate | **30/1** | **60000/1001 (59.94)** |
| total duration | **8390.94 s (139.8 min)** | **6604.46 s (110.1 min)** |
| bytes | 2,049,288,961 (14 files) | 2,118,160,664 (12 files) |

STATUS.md:1584-1585 records exactly this: the `france_senegal` material is "the **two clean halves**
(`fra-sen half 1/2.mp4`) -- a proper broadcast, **not the earlier 140-min replay package**". The
140-min replay package is `fra_sen`. Different encode, different frame rate, different length,
different chunk boundaries. **The annotated frame indices in
`data/ball_annotations/fra_sen/chunk_004.csv` (e.g. frames 1544, 14392, 16712 in an 18,000-frame
30 fps chunk) have no meaning against a 59.94 fps `france_senegal` chunk.** The audit's claim
"free -- re-chunk from `matches/france_senegal/`, minutes" is false: the source `.mp4` for `fra_sen`
is gone from the repo root and cannot be regenerated from anything still present.

### 5.3 Is the internal `play/` vs `chunks/` duplication real? YES

`play/chunk_004..013.mp4` (10 files, 1,438,154,992 B) vs `chunks/chunk_004..013.mp4`: identical
sizes, identical durations, identical frame counts. MD5 spot-check on three pairs:

| chunk | `chunks/` MD5 | `play/` MD5 | identical |
|---|---|---|---|
| 004 | 77D481778760D70F49C3F04A44EB392C | 77D481778760D70F49C3F04A44EB392C | yes |
| 005 | 1EEA2838E8C251F8AACC854EF23D5075 | 1EEA2838E8C251F8AACC854EF23D5075 | yes |
| 013 | 0C86EDE55447DB77E44439529EA7283C | 0C86EDE55447DB77E44439529EA7283C | yes |

So `play/` is a byte-identical second copy of `chunks/004-013`; `chunks/000-003` are unique to
`chunks/`. Integrity: all 25 `fra_sen` files decode head and tail clean.

### 5.4 Is `matches/france_senegal/` a separate, intact, registered fixture? YES -- KEEP IT

- Registered in `data/matches.yaml` (`france_senegal`, teams France/Senegal, `roster: true`,
  `pmsr: outputs/pmsr/france_senegal.json` -- that file exists, 2,760 B).
- 12 chunks, all decode clean, all uniform 1280x720 @ 59.94 fps.
- Fully processed: `outputs/france_senegal/h1/match_dense.parquet` and `h2/match_dense.parquet`
  exist, plus 6 + 6 per-chunk dense parquets -- one per chunk.
- Referenced by `data/ball_annotations/manifest.csv` for `h1/chunk_001`, `h1/chunk_004`,
  `h2/chunk_002`.
- It is the fixture behind `PMSR-M17-FRA-V-SEN.pdf`, the FIFA ground truth. **Non-regenerable.**

### 5.5 What `fra_sen` disposition is actually defensible

| item | bytes | GiB | verdict |
|---|---|---|---|
| `matches/fra_sen/play/chunk_004.mp4`, `chunk_005.mp4` | 294,980,830 | 0.275 | **DO NOT DELETE** -- live ball-training inputs |
| the other 23 files in `matches/fra_sen/` | 3,192,463,123 | 2.973 | *probably* redundant -- but see caveat |

Caveat, and why this document does not sign off on the 2.97 GB: `chunks/chunk_004.mp4` and
`chunks/chunk_005.mp4` are byte-identical backups of the two live files, and
`outputs/fra_sen/match/chunk_006..013_dense.parquet` exist, meaning chunks 006-013 were once
processed and could be annotated again -- that is the WC-footage corpus the ball detector was built
on, and it is not re-downloadable. **A conservative call keeps `matches/fra_sen/play/` (10 files,
1.339 GiB) and deletes only `matches/fra_sen/chunks/` (14 files, 1.909 GiB), which loses nothing
except chunks 000-003 (pre-match/warm-up material, never annotated, never in the manifest).** That
is the version this document is willing to call safe. Deleting `matches/fra_sen/` wholesale is NOT
safe.

---

## 6. (a) Root .mp4 paths PROVEN safe to delete

All 12. Each has a complete, stream-identical, frame-complete, fully decodable, already-pipeline-processed
chunk set under `matches/<id>/`.

| # | path | bytes | GiB | cumulative GiB |
|---|---|---|---|---|
| 1 | `c:\Users\siddh_ygv5bws\football-synthesizer\Crystal Palace v Manchester United 2024-25 Full Match Replay.mp4` | 4,717,930,526 | 4.394 | 4.394 |
| 2 | `c:\Users\siddh_ygv5bws\football-synthesizer\Southampton v Manchester United 2024-25 Full Match Replay.mp4` | 4,684,730,912 | 4.363 | 8.757 |
| 3 | `c:\Users\siddh_ygv5bws\football-synthesizer\Manchester United v Brighton 2024-25 Full Match Replay.mp4` | 4,665,065,639 | 4.345 | 13.102 |
| 4 | `c:\Users\siddh_ygv5bws\football-synthesizer\Manchester United v Crystal Palace 2024-25 Full Match Replay.mp4` | 4,639,088,313 | 4.320 | 17.422 |
| 5 | `c:\Users\siddh_ygv5bws\football-synthesizer\Manchester United v Liverpool 2024-25 Full Match Replay.mp4` | 4,626,730,454 | 4.309 | 21.731 |
| 6 | `c:\Users\siddh_ygv5bws\football-synthesizer\Fulham v Manchester United 2024-25 Full Match Replay.mp4` | 4,602,455,689 | 4.286 | 26.017 |
| 7 | `c:\Users\siddh_ygv5bws\football-synthesizer\Liverpool v Manchester United 2024-25 Full Match Replay.mp4` | 4,597,512,374 | 4.282 | 30.299 |
| 8 | `c:\Users\siddh_ygv5bws\football-synthesizer\Brighton v Manchester United 2024-25 Full Match Replay.mp4` | 4,581,819,049 | 4.267 | 34.566 |
| 9 | `c:\Users\siddh_ygv5bws\football-synthesizer\Manchester United v Tottenham Hotspur 2024-25 Full Match Replay.mp4` | 4,568,431,502 | 4.255 | 38.821 |
| 10 | `c:\Users\siddh_ygv5bws\football-synthesizer\Manchester United v Southampton 2024-25 Full Match Replay (1).mp4` | 4,513,100,384 | 4.203 | 43.024 |
| 11 | `c:\Users\siddh_ygv5bws\football-synthesizer\Manchester United v Fulham 2024-25 Full Match Replay (1).mp4` | 4,425,651,213 | 4.122 | 47.146 |
| 12 | `c:\Users\siddh_ygv5bws\football-synthesizer\Tottenham Hotspur v Manchester United 2024-25 Full Match Replay.mp4` | 4,365,146,213 | 4.065 | 51.211 |
| | **total** | **54,987,662,268** | **51.211 GiB** (54.99 GB) | |

Plus, from section 5.5, defensible at the same time:
`c:\Users\siddh_ygv5bws\football-synthesizer\matches\fra_sen\chunks\` -- 14 files, 2,049,288,961 B =
**1.909 GiB**, of which 10 files (1,438,154,992 B) are byte-identical duplicates of `fra_sen/play/`
and 4 (`chunk_000..003`, 611,133,969 B) are unannotated pre-match footage.
**Combined reclaim: 53.120 GiB.**

**One standing condition before executing any of this:** these chunk sets become the only copy of
footage that cannot be legally re-downloaded. Back up `matches/` to external media first. That is a
policy point, not a defect in the evidence.

## 6. (b) MUST BE KEPT, and why

1. **`matches/<id>/h1|h2/` for all 12 ManU fixtures -- 51.2 GiB.** After the root .mp4s go, this is
   the only copy of the PL 24/25 replay footage. Non-regenerable (licence forbids re-download).
2. **`matches/fra_sen/play/chunk_004.mp4` and `chunk_005.mp4` -- 0.275 GiB.** LIVE inputs to
   `tools/finetune_ball.py` via `data/ball_annotations/manifest.csv:5-6`; their annotations and the
   recorded `holdout_split.csv` rows depend on this exact 30 fps encode. Not reproducible from
   `matches/france_senegal/` (59.94 fps, different length, different cut points). **This contradicts
   `results/STORAGE_AUDIT_2026-07-24.md` sections 5.2, 7 and 8(a) item 4, which say `matches/fra_sen`
   is unreferenced and safe -- that claim is retracted here.**
3. **`matches/fra_sen/play/chunk_006..013.mp4` -- 1,143,174,162 B = 1.065 GiB.** Conservative keep: processed WC
   replay footage (`outputs/fra_sen/match/chunk_00N_dense.parquet` exist), non-re-downloadable,
   candidate material for further ball annotation. Would be settled by a decision that no further
   WC ball annotation will ever be done.
4. **`matches/france_senegal/`, `matches/france_iraq/`, `matches/france_norway/` -- 4.97 GiB.**
   Registered in `data/matches.yaml`, all with PMSR ground truth, all fully processed, all cited by
   the ball-annotation manifest. Non-regenerable.
5. **`matches/pl_probe/<fixture>/seg_1.mp4` (3 files).** Cited live by
   `data/ball_annotations/manifest.csv` -- same failure mode as `fra_sen`. The audit's "pl_probe is
   safe to delete" is wrong for at least these three files. (`seg_2`/`seg_3` are not referenced.)
6. **`outputs/<id>/` for all 12 fixtures.** The proof in section 3 depends on them, and they are
   ~80 GPU/CPU-hours of validated product.

### What would settle the remaining ambiguity

The only genuinely ambiguous item in this whole exercise is item 3 (`fra_sen/play/chunk_006..013`,
1.065 GiB) and the unannotated `fra_sen/chunks/chunk_000..003` (0.569 GiB). Both become clearly
disposable the moment Sid confirms the ball detector will not be re-fine-tuned on WC footage. Nothing
about the 12 root .mp4 files is ambiguous.

---

## 7. Method notes

- `ffprobe -v error -show_entries format=duration,size,bit_rate,format_name:stream=...` -of json,
  one call per file, 179 files.
- Integrity: `ffmpeg -v error -t 2 -i <f> -f null -` and `ffmpeg -v error -sseof -2 -i <f> -f null -`,
  167 chunks x 2.
- Box walk: pure `seek`+16-byte-header reads of top-level MP4 boxes (handles 64-bit `largesize`);
  no media bytes read.
- Pixel hashes: `ffmpeg -f rawvideo -pix_fmt rgb24` piped to md5; seam check downscaled to 320x180
  (deterministic scaler, both sides identical settings).
- Sizes: `Path.stat().st_size`; GiB = 1024^3.
- **No file in the repository was created, deleted, moved or modified by this verification** other
  than this document. All scratch scripts live outside the repo, in the session scratchpad.
