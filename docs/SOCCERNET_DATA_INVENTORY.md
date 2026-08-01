# SoccerNet data family vs. our own corpus — complete inventory

Date: 2026-07-29. Read-only survey. **Nothing was fetched for this document.**
Companion to `docs/SOCCERNET_SURVEY_RAW.md` (which surveys *methods*); this one surveys *data*.

Every number below is either (a) measured locally by counting the files/annotations we hold, or
(b) quoted from a named web source. Measured numbers are marked **[measured]**; everything else is
sourced and carries the uncertainty of its source. Where a source contradicts another, both are
shown.

---

# Part A — SoccerNet's data family

## A.0 The one structural fact that reframes the whole family

**SoccerNet-Tracking, SoccerNet-GSR and (in part) the re-ID set are the same pixels re-annotated.**

- SoccerNet-Tracking: 200 clips of 30 s, 1080p, 25 fps, from **12 Swiss Super League games (2019)**,
  split 57 train / 49 test / 58 challenge (arXiv 2204.06918).
- SoccerNet-GSR: 200 clips of 30 s, 1080p, 25 fps, split **57 train / 58 valid / 49 test**
  (`sequences_info.json` **[measured]**), and the GSR paper states it "expands SoccerNet-Tracking
  with additional pitch localization, camera calibration and athlete pitch position annotations".

The split sizes are the *same three numbers permuted* (tracking's 58-clip challenge became GSR's
58-clip valid; tracking's 49-clip test stayed the test). High-confidence inference, not a quoted
fact: **GSR is SoccerNet-Tracking's clips with a richer label layer.** Consequence for planning —
fetching SoccerNet-Tracking after GSR buys **almost no new video**, only a different (poorer) label
format. It is not a second corpus.

Second consequence, and it matters for us: **the GSR/tracking video is Swiss Super League 2019**,
not a top-5 league and not 2024-25. Every GSR number this project has measured is measured on a
domain one step further from ManU EPL 24-25 than we may have been assuming.

## A.1 Per-dataset table

| Dataset | Raw data | Annotations (exact) | Density / QC | Size on disk | License | Access route |
|---|---|---|---|---|---|---|
| **SoccerNet-GSR** (`SN-GSR-2025`, also `SN-GSR-2024`) | 200 clips x 30 s, 1920x1080, 25 fps = 150,000 frames total; Swiss Super League 2019 (inherited from Tracking) | Per frame, per person: `bbox_image`, `bbox_pitch` (+`bbox_pitch_raw`), `track_id`, `role` (player/GK/referee/other), `team` (left/right), **`jersey` number**; plus ball boxes; plus **26-class pitch-line point series** and camera params. Paper totals: 2.36 M athlete positions, 9.37 M line points | Manual annotation; keyframes + interpolation; pitch-visibility check before a frame is admitted. **[measured]** on the 58 valid clips: 76.2% of person boxes carry a jersey-number label | **35.1 GB** on HF (train 9.76 / valid 11.2 / test 8.85 / challenge 5.31) | HF card says **GPL-3.0**; the arXiv listing for the paper is CC BY-4.0. Inconsistent — treat as GPL-3.0 for the data | HuggingFace `SoccerNet/SN-GSR-2025`, **not gated**; also `sn-gamestate` auto-downloader |
| **SoccerNet-Tracking** | 200 clips x 30 s @1080p/25fps + 1 full 45-min half + 12 full games; Swiss Super League 2019; 225,375 frames in the annotated portion | ~**3.6 M** bounding boxes, **5,009** tracklets: player 4,005 / GK 262 / referee 432 / ball 297 / other 13. MOT-style 10-column CSV. Team and jersey shipped as *metadata where visible*, not as a primary label | **88.5% manually annotated**, 11.5% keyframe-interpolated. Professional annotation via SuperAnnotate | Not published. Same clips as GSR, so ~35 GB of pixels + small labels; the 12 full games are extra and unsized | Not stated in repo | `pip install SoccerNet` -> `downloadDataTask(task="tracking")`. Repo unmaintained since 2023-07-07 |
| **SoccerNet Re-Identification** | 340,993 player thumbnails from **400 matches, 6 major leagues**; highly variable crop resolution | Identity labels **valid only within one action** — "a given player has a different identity for each action he has been spotted in". No jersey number, no name | Derived from SoccerNet-v3 within-action bbox links, not from a persistent-identity pass | **18.63 GB** (train 12.12 / valid 2.35 / test 2.41 / challenge 1.75) | MIT (repo) | SoccerNet pip. Unmaintained since 2023-07-07 |
| **SoccerNet Jersey Number Recognition (`jersey-2023`)** | Player tracklets, each a folder of crops. **[measured]** locally: 1,427 train + 1,211 test tracklets, ~1.30 M crop images | **One integer label per tracklet** (`-1` = illegible). Not per-frame, not per-crop | Repo states 2,853 tracklets total and a 1,211-tracklet hidden challenge set; our train+test = 2,638 **[measured]**, so the 2,853 figure and the released splits do not reconcile cleanly — flagged, not resolved. Illegible share **[measured]**: 28.2% train, 29.3% test | **4.28 GB** extracted+zips **[measured]** | Not stated in repo | SoccerNet pip `task="jersey-2023"` |
| **SoccerNet Camera Calibration (`calibration-2023`)** | 25,506 images from the 500-match SoccerNet video pool; includes non-main cameras (behind goal, in goal, spider-cam) | Extremities of pitch semantic elements, 26 classes, as 2D point pairs; homography derivable | 16,463 train / 3,212 valid / 3,141 test / 2,690 challenge (challenge unannotated) | Not published | Not stated | SoccerNet pip `task="calibration-2023"`. Unmaintained since 2024-06-18 |
| **SoccerNet Action Spotting (v2)** | 500 untrimmed broadcast matches (550 in some counts), **~764 hours**, 720p and 224p, 25 fps, 6 European leagues 2014-2017 | ~**300,000** single-timestamp action labels, **17 classes** (goal, foul, corner, card, substitution, …). Plus replay grounding and camera-shot segmentation layers | Timestamp-level, manually annotated | Video needs NDA; features/labels are small | NDA for video | SoccerNet pip; **video requires the NDA password** |
| **Ball Action Spotting (`SN-BAS-2025`)** | **7 English Football League games**, 720p | **11,041** single-timestamp annotations across **12 classes** (Pass, Drive, Header, High Pass, Out, Cross, Throw-in, Shot, Ball-Player Block, Successful Tackle, Free Kick, Goal). 2025 adds a **team side (left/right)** requirement | Dense on-ball annotation of complete games | **19.3 GB** on HF (train 8.45 / valid 2.04 / test 4.53 / challenge 4.23). One source cites ~1.2 TB for the full-resolution video pool | Not stated | HF `SoccerNet/SN-BAS-2025`, **not gated** |
| **PCBAS / FOOTPASS (`SN-PCBAS-2026`)** | **54 complete men's matches**, 2023/24, Ligue 1 / Bundesliga / Serie A / La Liga / UCL. 1920x1080 @25 fps, **~81 hours** | **102,992** claimed `(frame, team, jersey, class)` on-ball events, 8 classes; **[measured]** 97,397 present in the public TRAIN+VAL (CHALLENGE labels withheld). Underneath them, **dense per-frame per-player tracking**: 177 M player-frame rows with pitch x/y, velocity, team, **shirt number**, 13-way role, and broadcast-frame ROI (NaN when off-screen) | Manually validated events; game state supplied as **ground truth**, so published PCBAS scores are *not* end-to-end-from-pixels | **235 GB total** on HF. Annotations only = 2.6 GB. Video: fullHD VAL 10.57 GiB, fullHD TRAIN 5 parts ~171 GiB, fullHD CHALLENGE 10.74 GiB; 352x640 variants 1.26/21.24/1.45 GiB | Annotations + baselines **CC BY-NC 4.0**; **video needs a separate NDA** (Google Form) | HF `SoccerNet/SN-PCBAS-2026` — **GATED** ("agree to share contact information"); video additionally NDA-gated |
| **SoccerNet-Caption** (dense video captioning) | 471 untrimmed broadcast games, 720p, 25 fps | **36,894** anonymised timestamped captions | Anonymised — **player names are stripped**, which kills it as an identity source | Small (text) | Not stated | `sn-caption` repo. Stale since 2024-04-12 |
| **SoccerNet-Echoes** | ASR transcripts of **550 games** (the 2014-2017 v2 pool), multilingual + English translation | Whisper large-v1/v2/v3 transcripts, "original" and "en" variants, timestamped | Machine transcription. **Authors state entity/player-name transcription is unreliable** | Small (text), HF size class 10M-100M rows | Not stated | HF `SoccerNet/SN-echoes`, **not gated** |
| **SN-MVFouls (2024, 2025)** | **3,901** foul actions from 500 games, 6 leagues, 2014-2017, multi-camera clips | 10 properties per foul (offence type, severity, body part, …) | Refereeing-expert annotation | Not published | Not stated | HF `SoccerNet/SN-MVFouls-2025`, **not gated** |
| **SN-Depth-2025** | Broadcast frames with depth | Monocular depth GT | — | **44 GB** | Not stated | HF, **not gated** |
| **SN-NVS-2026** | Multi-camera broadcast footage for novel-view synthesis | Multi-view frames + poses | — | Not published | Not stated | HF `SoccerNet/SN-NVS-2026`, **not gated** |
| **SN-VQA-2026** | Text + image + video across **14 soccer understanding tasks** | QA pairs | — | Not published | Not stated | HF `SoccerNet/SN-VQA-2026` — the 2026 challenge page calls it **gated**, while the HF API reports `gated: false`. Contradiction, flagged |
| **ActionAnticipation / BannerReplacement / SoccerNet_raw_HQ** | Auxiliary sets. `SoccerNet_raw_HQ` is the high-quality raw video pool | — | — | Not published | — | `SoccerNet_raw_HQ` is **gated: manual** (per-request approval). Others not gated |

## A.2 What is new for 2026

The 2026 challenge lineup is: **Spiideo SynLoc** (1st edition — single-frame world-coordinate
athlete localisation, synthetic + real), **Ball Action Anticipation** (new task — predict what and
when in a 5 s window), **VQA**, **Player-Centric Ball Action Spotting** (PCBAS/FOOTPASS),
**Novel View Synthesis** (new), and the **FIFA Skeletal Tracking Light** challenge (2nd edition).
Deadline was 2026-04-24; USD 1,000 per track, plus World Cup Final tickets for the FIFA track.

**Two absences worth naming:**

1. **There is no GSR challenge in the 2026 lineup.** GSR ran 2024 and 2025. Its data
   (`SN-GSR-2025`) is still up and un-gated; the competition around it has ended.
2. **There has never been a jersey-number dataset after `jersey-2023`.** No 2024, 2025 or 2026
   edition exists. `jersey-2023` — which we already hold in full — is the *entire* public
   jersey-number-recognition corpus. There is nothing to upgrade to.

---

# Part B — What we hold locally

All sizes **[measured]** 2026-07-29 by recursive file-size sum.

## B.1 Third-party data on disk

| Asset | On disk | Files | What it is | What's missing |
|---|---|---|---|---|
| `data/soccernet/gamestate-2024/` | **11.54 GB** | 43,559 | **The complete 58-sequence GSR *validation* split** (SNGS-021…059, 078…096) — verified against `sequences_info.json` v1.3: 58 of 58 valid clips present, 0 missing | **All 57 train clips** and **all 49 test clips**. We have benchmarked GSR on a split whose official role is model selection |
| `data/soccernet/jersey-2023/` | **4.28 GB** | 1,297,552 | Full train (1,427 tracklets, 403 illegible) + full test (1,211 tracklets, 355 illegible), both with GT JSON | Only the hidden challenge set — which has no public labels. **Effectively complete** |
| `data/footpass/raw/` | **13.14 GB** | 12 | FOOTPASS annotations, all 3 splits (2.6 GB) + `videos_fullHD_VAL.zip` (10.57 GiB, games 18/24/47, encrypted) | TRAIN video (~171 GiB fullHD) and CHALLENGE video. TRAIN *annotations* are present |
| `data/footpass/digest/` | **0.10 GB** | 102 | Our own compact per-half arrays derived from the HDF5 | — |
| **Third-party total** | **≈29 GB** | | | |

## B.2 Our own corpus

`data/matches.yaml` holds **19 registry entries**: **12 ManU EPL 2024-25 matches**, 1 older
`mun_mci`, 3 France WC reference matches (the only ones with FIFA PMSR ground truth), and 3 FOOTPASS
VAL games registered as first-class matches.

The 12 ManU EPL 24-25 matches (all with `aligned` parquet + `ball_dir`, none with PMSR):

`manutd_fulham` (2024-08-16), `brighton_manutd` (08-24), `manutd_liverpool` (09-01),
`southampton_manutd` (09-14), `palace_manutd` (09-21), `manutd_tottenham` (09-29),
`liverpool_manutd` (2025-01-05), `manutd_southampton` (01-16), `manutd_brighton` (01-19),
`fulham_manutd` (01-26), `manutd_palace` (02-02), `tottenham_manutd` (02-16).

At ~100 min per replay that is **≈20 hours of EPL 24-25 broadcast** — comparable in raw volume to a
quarter of FOOTPASS, and **80x** the GSR validation split's 29 minutes.

Per-match derived artifacts (machine output, not ground truth): `final/match_aligned.parquet`,
`final/h1|h2_aligned.parquet`, `final/ball/ball_*_chunk*.parquet` (11 chunks on the sampled match),
`final/gta/`, `final/ocr_percrop/`.

### The hand-made ground truth — the whole of it

| Artifact | Count | What it is | Ceiling |
|---|---|---|---|
| `labels_filled.csv` (= `results/carrier_attr/labelpack/labels_filled.csv`) | **90 rows** | Hand-named ball carrier at a single frame: `(match, chunk, frame, team_tracked, carrier_dist_m, player_name, confidence_1to3)`. Backed by a 6-image context pack per moment (crop + ctx-2..+2) | 90 moments across 3 matches |
| `results/carrier_attr/label_scoring.csv` | **78 rows** | The subset actually judged against predictions | `GTA_LINK_STAGE1.md` A4 already recorded that these carry **23 usable naming moments** and "cannot resolve any intervention of realistic size". Every precision from them has a ±0.18 interval |
| `results/carrier_attr/*_gallery.parquet` (9 matches) | **9,646 rows**, **93 unique player names** | Close-up-anchored identity gallery: `(chunk, frame, track_id, team, player, image_x, image_y)` | **Machine-derived**, not hand ground truth — OCR/close-up anchors propagated by the connector. Usable as a LOTO proxy, not as truth |
| FIFA PMSR JSON (3 France matches) | 3 matches | The only external, official ground truth in the project | Wrong competition, wrong era, team-aggregate metrics only — no per-player-per-frame anything |

**That is the complete list.** There is no other hand-made label set in this repository.

---

# Part C — Comparison

## C.1 The table

| Asset | Hours video | Matches / clips | Label types | Label density | External-truth quality | License | Held? | GB to fetch |
|---|---|---|---|---|---|---|---|---|
| **GSR valid** (ours) | 0.48 h | 58 clips x 30 s | box + pitch-xy + track_id + role + team + **jersey** + 26-class lines | **[measured]** 792,166 annotations / 43,500 frames; 707,766 person boxes; **76.2% carry a jersey GT** | Gold — manual, QC'd, interpolation-audited | GPL-3.0 (HF) | **YES** 11.54 GB | 0 |
| **GSR train** | 0.48 h | 57 clips | identical schema | ~same density | Gold | GPL-3.0 | **NO** | **9.76** |
| **GSR test** | 0.41 h | 49 clips | identical schema | ~same density | Gold | GPL-3.0 | **NO** | **8.85** |
| **GSR challenge** | 0.29 h | 36 clips (5.31 GB) | images only, labels withheld | n/a | n/a (eval server) | GPL-3.0 | **NO** | 5.31 |
| **SoccerNet-Tracking** | ~1.9 h + 12 games | 200 clips + 1 half + 12 games | box + track_id + class; team/jersey as loose metadata | 3.6 M boxes, 5,009 tracklets, 88.5% manual | Gold but **poorer schema than GSR on the same pixels** | unstated | **NO** | unpublished (mostly redundant with GSR) |
| **SoccerNet re-ID** | n/a (crops) | 400 matches | within-action identity only | 340,993 thumbnails | Weak — identity does not persist across actions | MIT | **NO** | **18.63** |
| **jersey-2023** | n/a (crops) | 2,638 tracklets | **one integer per tracklet** | 71.8% / 70.7% legible **[measured]** | Gold, but tracklet-level not frame-level | unstated | **YES** 4.28 GB | 0 |
| **calibration-2023** | n/a (stills) | 25,506 images | 26-class pitch point sets | one label set per image | Gold | unstated | **NO** | unpublished |
| **Action Spotting v2** | **764 h** | 500 matches | 17-class timestamps | ~300 k labels / 764 h ≈ 390/match | Gold, but **no actor identity at all** | NDA for video | **NO** | NDA-gated |
| **Ball Action Spotting** | ~11 h | 7 EFL games | 12-class timestamps + team side | 11,041 labels / 7 games ≈ 1,577/match | Gold, no player identity | unstated | **NO** | 19.3 (HF) |
| **FOOTPASS annotations** | — | 54 matches | `(frame, team, **jersey**, class)` + **dense per-frame tracking, role, velocity, ROI** | **[measured]** 97,397 events over 177 M player-frame rows | Gold, manually validated | CC BY-NC 4.0 | **YES** 2.6 GB | 0 |
| **FOOTPASS video** | 81 h total | 54 matches | (pixels for the above) | — | — | NDA | **VAL only** (3 games, 10.57 GiB) | TRAIN ~171 GiB fullHD (NDA) |
| **Our ManU EPL 24-25** | **≈20 h** | 12 matches | *derived*: tracks, pitch xy, ball, per-crop OCR | **8.7% of tracklets carry an OCR read** (`EVIDENCE_DENSITY_LAW.md`) | **None** — no external truth exists for these matches | user-supplied replays | **YES** | 0 |
| **Our hand labels** | — | 3 matches | carrier name at a frame | **90 labels / 78 judged / 23 usable naming moments** | Human, single-annotator, ±0.18 intervals | ours | **YES** | 0 |

## C.2 What their labels give that ours cannot

**1. Per-frame, all-player identity ground truth exists nowhere in our corpus, and cannot be
created without annotation labour.**
On the 58 GSR valid clips we hold, **707,766 person boxes** each carry role + team + track_id, and
**539,624 of them (76.2%) carry a hand-verified jersey number** — that is one identity-resolved
athlete-frame every 0.06 s of video, for every visible player simultaneously. Our own 20 hours of
EPL footage carry **90** hand-named moments, each naming **one** player at **one** frame. The ratio
is roughly **6,000:1 in their favour on identity labels**, on 1/40th of the video.
No amount of compute closes this. It is annotation labour, and it is the reason every identity
claim we make is measured on somebody else's Swiss Super League clips.

**2. Nobody's labels — theirs or ours — answer the question our demo asks.**
GSR's identity GT is on 30-second clips of Swiss Super League 2019. FOOTPASS's is on 2023/24
continental football, teams anonymised as "team 1 / team 2", **no club names anywhere**. Action
Spotting has no actor identity at all. Caption is deliberately **name-anonymised**. Echoes has names
but its authors warn they are unreliable. So the label type "this named Manchester United player did
this thing in this EPL 24-25 match" exists in **no public dataset**, ours included. Everything we
can borrow is a *method benchmark* in a shifted domain; none of it is in-domain supervision.

**3. Their evidence density is 4-9x ours, and we have measured exactly what that costs.**
`EVIDENCE_DENSITY_LAW.md` establishes that attribution reaches precision 0.85 at coverage ≥0.50 only
at read density **d = 0.347** (at 0.86 read precision). Our real footage runs at **d = 0.087**.
GSR's *ground truth* jersey density is **0.762** **[measured]** — 8.8x our achieved OCR density and
2.2x the density the law says we need. This is the cleanest statement of the gap: their labels sit
comfortably above the threshold where the problem becomes solvable; our evidence sits at a quarter
of it. The GSR train split is the only public asset that lets us *train* against that density rather
than merely evaluate against it.

---

# Part D — Fetch-plan inputs (nothing fetched)

| # | Asset | Size | Route | Gated? | Note |
|---|---|---|---|---|---|
| 1 | **GSR train** (`train.zip`) | **9.76 GB** | HF `SoccerNet/SN-GSR-2025` | **No** | The single highest-value missing asset. 57 clips, same schema as the 58 we hold, ~700 k more identity-labelled person boxes. Extraction ratio measured at ~1.03x, so budget **~20 GB peak** if the zip is kept alongside |
| 2 | **GSR test** (`test.zip`) | **8.85 GB** | same | **No** | Needed for any honest headline number — we currently report on a split whose official role is model selection. ~**18 GB peak** |
| 3 | GSR challenge (`challenge.zip`) | 5.31 GB | same | No | Images only, labels withheld. Useless without a Codabench submission |
| 4 | **SoccerNet re-ID** | **18.63 GB** (train 12.12 / valid 2.35 / test 2.41 / challenge 1.75) | `pip install SoccerNet` -> `downloadDataTask(task="reid")` | No | MIT. But identity is **within-action only** — it trains an embedder, it cannot supervise persistent identity. Fetch train only (12.12 GB) if the goal is a better embedder |
| 5 | SoccerNet-Tracking | unpublished | SoccerNet pip | No | **Recommend skipping.** Same clips as GSR with a poorer label schema. The 12 full games are the only genuinely new pixels, and they are unsized |
| 6 | **jersey-2024+** | — | — | — | **Does not exist.** No jersey dataset after `jersey-2023`, which we hold complete. Dead end — close this line |
| 7 | calibration-2023 | unpublished (25,506 images) | SoccerNet pip `task="calibration-2023"` | No | Only relevant if the retracted calibration claim is revisited. Repo unmaintained since 2024-06-18 |
| 8 | FOOTPASS TRAIN video | ~171 GiB fullHD (5 parts, 31-37 GiB each) or **21.24 GiB at 352x640** | HF `SN-PCBAS-2026` | **Yes, gated** + separate video NDA | The 352x640 TRAIN zip at 21 GiB is the sane cluster option if end-to-end-from-pixels on FOOTPASS is ever wanted. Requires Sid's per-fetch approval per `CLAUDE.md` |
| 9 | SN-BAS-2025 | 19.3 GB | HF | No | 7 EFL games. No player identity — low value for our line of work |

**Total for a complete GSR pursuit: 23.92 GB to download** (train 9.76 + test 8.85 + challenge
5.31), or **18.61 GB** for the useful part (train + test, skipping the label-less challenge split).
Peak disk with zips retained alongside extraction: **~38 GB**. Added to the 11.54 GB we already
hold, a complete local GSR corpus lands at **~36 GB extracted**.

Realistic cluster-storage line item for "GSR pursuit + a better embedder":
**GSR train+test (18.61 GB) + re-ID train (12.12 GB) ≈ 31 GB download, ~62 GB peak.**

## Access surprises

1. **`SN-PCBAS-2026` is gated on HuggingFace** ("you need to agree to share your contact
   information"), and its **video is NDA-gated on top of that**. Two independent gates. Total repo
   is **235 GB**.
2. **`SoccerNet_raw_HQ` is `gated: manual`** — per-request human approval, the strictest gate in the
   org.
3. **`SN-VQA-2026`: the 2026 challenge page says the HuggingFace data is gated; the HF API reports
   `gated: false`.** Unresolved contradiction — do not plan around either answer without checking.
4. **The GSR license is inconsistent**: the HF dataset card says **GPL-3.0**; the arXiv paper
   listing is CC BY-4.0. GPL-3.0 on a *dataset* is unusual and would be worth a second look before
   anything derived from it is published.
5. **Four of the identity-relevant repos are unmaintained** (sn-reid and sn-tracking since
   2023-07-07, sn-jersey since 2024-07-02, sn-calibration since 2024-06-18). Their pip-based
   downloaders are the only documented route to reid/jersey/calibration/tracking data and have not
   been touched in one to three years. **No dead link was confirmed in this survey** — but none was
   exercised either, because nothing was fetched.
6. **No jersey dataset exists after 2023**, and **GSR has no 2026 challenge**. Both lines have
   stopped upstream.
7. `soccer-net.org/data` is thin: it lists the datasets but omits sizes, leagues and seasons for
   most of them. The real numbers live in the papers and on HuggingFace, not on the project site.

## Numbers this survey could not establish

Stated plainly rather than estimated: **on-disk sizes for SoccerNet-Tracking, calibration-2023 and
jersey-2023 as distributed**, and **license text for jersey, calibration, tracking, caption, echoes
and MVFouls**. None are published in the repos or on the project site. They would have to come from
starting a download and reading the reported size, which was out of scope here.
