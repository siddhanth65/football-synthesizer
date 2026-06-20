# STATUS

**Last updated:** 2026-06-19

## Current focus

**The full pipeline is closed end-to-end on real video, with the standard wide-shot gate**: clip →
detect/track → **robust jersey-colour teams** → **PnLCalib calibration** → contract → trained GAT, in
one interpreter. On a real wide tactical segment: 32 frames, ~18 players/frame, both teams, sub-metre
calibration → the GAT produced a clip readout (P(success) 0.506, Dynamic-xT 0.0312). Generator core is
complete. Next milestones are the attacker rebuild (C3) and the synthesizer (C5).

## Done this session

- Created the standalone repo (separate from `football-state-of-play` and `cv-football`).
- Wrote the final combined plan: [docs/PLAN.md](docs/PLAN.md) — one pipeline, generator-first,
  novelty C3 + C5, end goal = auto-generated FIFA-style team report.
- Built the foundation the whole pipeline plugs into: the **unified, substrate-aware freeze-frame
  contract** ([generator/contract.py](generator/contract.py)) + tests — pure, dependency-light,
  down-projects to the trained GAT's input format.
- Stubbed every package (`ingest`, `generator`, `attacker`, `fingerprint`, `synthesizer`, `report`,
  `eval`) with docstrings + typed signatures + TODOs.
- Registered the data sources in [ingest/sources.py](ingest/sources.py).
- **Wired the loop end-to-end** ([generator/sop_bridge.py](generator/sop_bridge.py)): positions
  parquet → `frames_from_positions` → trained GAT → clip-level relational readout (P(success),
  Dynamic-xT, P(def recovers)). Verified on the real `vid1_full.parquet` (25-frame run).
- **Unified the interpreter (the two-stack split is gone)**: installed `torch_geometric` into py3.14,
  which already had the CV stack + torch. The whole pipeline now runs in one process (see Decisions).
  `sop_bridge` keeps an auto-re-exec fallback (tested) but no longer needs it here.
- **Validated `extract.py` end-to-end on real gameplay** (py3.14, `chunk_000.mp4`): detection +
  ByteTrack ID persistence + ball detection all work; added `--start-frame`/`--max-frames` after
  finding the clip's first ~3 s is broadcast pre-roll (0 players).
- **Wired PnLCalib** — [generator/calibrate.py](generator/calibrate.py) `PnLCalibCalibrator`: loads
  PnLCalib's two HRNet models (SV_kp/SV_lines), detects pitch keypoints/lines. Cloned to `~/PnLCalib`
  (override `$FOOTBALL_PNLCALIB_PATH`); weights under `<repo>/weights`; deps (`shapely`, `lsq-ellipse`,
  `munkres`) installed + declared.
- **CRITICAL CALIBRATION BUG — two stages, both now fixed and verified.**
  - *Stage 1 (my-own-homography):* the first integration used PnLCalib only as a keypoint detector,
    then fit *my own planar homography* (`cv2.findHomography`) on the ground keypoints. Those cluster
    in a thin image band (~192 px), under-constraining a planar homography → players projected to
    nonsense. **Fixed** by switching to PnLCalib's full 3D **camera calibration** (`heuristic_voting`
    → points+lines+PnL refinement) and deriving the image→pitch homography from its 3×4 projection
    `P` (`ground_homography_from_cam_params`).
  - *Stage 2 (centred-vs-uncentred — the one that still looked "inverted"):* PnLCalib's camera params
    are in a frame **centred on the centre spot** (its `keypoint_world_coords_2D` are re-centred by
    `[x−52.5, y−34]`, `utils_calib.py:41`). I treated `P`'s output as **uncentred** `[0,105]×[0,68]`,
    a constant **(52.5, 34) m offset**: the centre spot landed at a corner, ~half the players were
    dropped as "off-pitch", and a far-goal keeper plotted at "halfway". This hid because the gate's
    own `obj` points were *also* centred, so the error read ~0.27 m while the **output** was offset by
    ~62 m. **Fix:** `ground_homography_from_cam_params` now inverts to centred metres then shifts the
    origin to the corner (×`to_uncentred`), and `calibrate_frame`'s gate compares against uncentred
    `obj`. **Verified** (`tools/probe_convention.py`): production homography error vs true uncentred
    coords **62.55 m → 0.27 m**; the right-goal keypoint maps to **(105, 43)**, not (52, 9); pitch
    geometry projects correctly (halfway line at the image's left edge, right box centre-right, left
    box off-screen); `corr(image_x, pitch_x)=+0.85` (no inversion). Locked by a synthetic centred-cam
    unit test (`test_calibrate.py`) **and** an opt-in golden-frame integration test
    (`tests/test_golden_frame.py`). User confirmed the camera-model pitch lines are correct.
  - **Regenerated the displaced outputs** (Delivery #1): `seg1_fixed` now **44/47 frames, ~16 players,
    ~41 m span** (right-half view, x 56–97); `seg2_fixed` now **54 frames, 14–18 players** spread round
    the centre circle (x 26–59) — previously the infamous "4 dots in a corner".
- **Player-distribution plausibility gate** ([generator/postprocess.py](generator/postprocess.py)
  `reject_implausible_frames`): a frame is trusted only if ≥8 players land on-pitch spanning ≥25 m on
  an axis; otherwise its pitch coords are NaN'd. Kept as a backstop. NB: seg2's old "drop all 57" was a
  *symptom of the Stage-2 bug* (centred coords crammed players into a corner), not a bad angle — with
  calibration fixed, seg2 now passes. Unit-tested.
- **Pass-C groundwork — roles, keepers, pitch boundary** ([generator/postprocess.py](generator/postprocess.py),
  [generator/extract.py](generator/extract.py)):
  - *Pitch boundary (linesman fix):* `clamp_to_pitch` tolerance cut 5 m → **2 m**; points past it are
    **dropped, not clamped onto the touchline** (no phantom officials-as-players). On seg1 this dropped
    only ~1 row while keeping all real players.
  - *Keeper tags:* `derive_keeper` now (1) trusts a detected `goalkeeper` role, else (2) tags a team's
    extreme player **only if within 16.5 m of its own goal** — killing the "keeper at halfway" false
    tag. Verified: seg1 went from false midfield keepers to **3 tags, all at x≈99.7 (right goal),
    ≤1/frame**.
  - *Role plumbing:* every detector returns a per-box role id carried through ByteTrack; `referee`
    rows get `team=-1` and are excluded from actor/keeper/plausibility. COCO/RF-DETR emit all `player`
    (no role classes), so a football-trained detector slots in unchanged. Unit-tested.
  - **Football-role detector adopted (Pass C).** Benchmarked 4 candidates on 6 held-out frames
    ([tools/benchmark_detectors.py](tools/benchmark_detectors.py)): COCO YOLOv8s, HF `uisikdag` v8,
    HF `soccana` v11, RF-DETR. **Winner: `uisikdag/yolo-v8-football-players-detection`** — the only
    one with a goalkeeper class, cleanly separates officials (others count them as players), and is
    conservative on the ball (soccana/RF-DETR over-detect). Wired as `--detector football`
    ([generator/extract.py](generator/extract.py) `_FootballRoleDetector`, class names mapped by name,
    weights download cached from HF). End-to-end on seg1: **GK at x≈97 (right goal), tagged keeper via
    role; central referee identified at x≈75 and excluded (team=-1); ball in 10/44 frames**; team
    classifier now fits on player crops only (GK/ref kits no longer pollute the 2-team KMeans).
  - **Open refinements:** GK *team* still comes from jersey KMeans (better: assign by defended goal);
    ball recall ~23% of frames (needs native-fps ball tracking); role *precision* not yet measured on
    labelled frames.
- **Tracking — swappable backends, ByteTrack kept as default** ([generator/tracking.py](generator/tracking.py),
  `--tracker {bytetrack,botsort}`). Added Ultralytics **BoT-SORT** (GMC `sparseOptFlow` + appearance
  ReID, tuned in [generator/botsort_tuned.yaml](generator/botsort_tuned.yaml)) via the YOLO detector's
  `track()`. **Benchmarked vs ByteTrack** on a 2 s dense clip and a 50 s sampled window: ByteTrack won
  both (≈22 IDs ≈ true player count, mean track 57 fr, frag 1.69) vs tuned BoT-SORT (28–34 IDs, p50
  18–27 fr, frag 2.32–2.66) — the held tactical camera pans too little for GMC to help, and
  ByteTrack's 3-frame confirmation suppresses the spurious tracks BoT-SORT keeps. **Evidence-based
  call: stay on ByteTrack**; BoT-SORT stays available for heavy-pan sources (re-benchmark per source).
  NB: `supervision.ByteTrack` is deprecated (removed in sv 0.30) — migration needed eventually.
- **Detection validation harness** ([tools/validate_detection.py](tools/validate_detection.py),
  tested box-matching core). Two GT modes: a **multi-detector consensus proxy** (a box ≥2/3 detectors
  agree on = a person) and **manual YOLO labels** (`--labels`). Consensus over 10 held-out frames:
  football detector **person recall 0.979 / precision 0.898** (rfdetr 0.995/0.942, coco 0.936/0.985),
  consensus head-count ≈19.5/frame. Proxy can't judge roles/ball — exported the 10 frames + draft YOLO
  labels + README ([outputs/validation/annotate/](outputs/validation/annotate/)) for correction →
  true role accuracy + ball recall via `--labels`.
- **Temporal calibration (Pass B)** ([generator/temporal_calib.py](generator/temporal_calib.py),
  `--calib-period N --calib-drift PX`). Full PnLCalib only on shot-cut (hist break) + every N frames +
  on camera drift (phase-correlation); **reuses the last accepted pose** between, and **never
  resurrects a rejected frame** (unit-tested policy). On 100 frames, period=25 ran **4 full
  calibrations vs 100** (~25× fewer) with the **same frame coverage**. Accuracy caveat: at a loose 6 px
  drift gate it added ~1.9 m median in the *length* axis (depth is tilt-sensitive); drift gate
  tightened to **2 px default** to keep reuse honest. **Default stays per-frame (`--calib-period 1`)**;
  temporal is the opt-in speed lever for full-match runs (#4), to be tuned against the ≤2 m budget.
- **Visual verification upgraded** ([tools/visualize.py](tools/visualize.py) `--pitch-lines`): draws
  the reconstructed pitch model on the video panel so each overlay self-verifies (lines must sit on
  the real markings). [tools/diag_calib.py](tools/diag_calib.py) compares calibration methods on a frame.
- **Fixed the team-classifier collapse** — [generator/teams.py](generator/teams.py)
  `JerseyColorTeamClassifier`: grass-masked CIELAB-chrominance per crop → KMeans(2). Deterministic,
  no SigLIP/UMAP. Diagnosis showed the old {304:7} collapse had two causes, both removed: (1) a
  **RGB/BGR crop bug** in extract (predict crops were RGB, the classifier expects BGR — fixed), and
  (2) UMAP's `.transform` destabilising across camera regions (in-sample both classifiers split fine;
  the colour one has no `.transform` step). 5 unit tests.
- **Fixed the wide-shot problem** — [generator/segments.py](generator/segments.py): a `wide_shot_score`
  (players × horizontal spread) + `scan_wide_segments`, plus an `extract --auto-wide` flag that picks
  the widest tactical segment automatically. 4 unit tests.
- **Closed the loop with the STANDARD ≥10 wide-shot gate (no lowering):** `--calibrate` extract on a
  scanned wide segment → **32 frames, 17–20 valid players/frame, both teams ({1: 317, 0: 272}),
  0.22 m mean calibration error** → `sop_bridge` ran the GAT on all 32 frames →
  **P(success) 0.506, Dynamic-xT 0.0312, P(def recovers) 0.461**. The full video→GAT path now
  produces a non-degenerate tactical readout from real footage.
- **GPU enabled (RTX 3050 Laptop, CUDA 12.8).** Replaced the CPU torch with `torch 2.11.0+cu128` +
  `torchvision 0.26.0+cu128` in the unified py3.14 env (cp314 CUDA wheels exist; minor 2.12→2.11
  downgrade, all deps fine). `PnLCalibCalibrator` and YOLO **auto-detect CUDA** (calibrator
  `device=None` → cuda). **PnLCalib dropped from ~13 s/frame (CPU) to ~0.35 s/frame warm (~38× faster)**,
  ~2 GB VRAM, same accuracy (0.247 m). All 50 tests pass on the new torch. End-to-end GPU run
  (`extract --calibrate --auto-wide`, full-clip scan + 50 calibrated frames + bridge): **~66 s wall**
  (was ~8 min for a smaller CPU run); `--auto-wide` auto-picked frame 13050; 47 frames cleared the
  ≥10 gate through the GAT.
- **Built the CV front-end as reproduce-and-adapt of `cv-football`, with the trustworthy work in
  tested, pure modules:**
  - [generator/calibrate.py](generator/calibrate.py): homography estimation (numpy normalised-DLT,
    or `cv2.findHomography`/RANSAC when present) + the **reprojection-error confidence gate**
    ("no wrong frames"). PnLCalib is a lazy seam (weights are an external download). 5 tests.
  - [generator/postprocess.py](generator/postprocess.py): pure track smoothing + ball-carrier
    (actor) and keeper derivation + pitch-bounds clamping — lifted out of the CV script so they run
    and are tested without the CV stack. 6 tests.
  - [generator/extract.py](generator/extract.py): video → positions parquet orchestration (YOLO +
    ByteTrack + team classifier + calibrate + post-process), **heavy CV imported lazily** (module
    imports without `[cv]`), COCO fallback when Roboflow `inference` is absent. Pure seams
    (transform+gate, row building, schema) are tested. 8 tests.
- **Test suite: 50 tests, all passing in the unified py3.14 env** (the GAT-bridge test runs here now,
  not skipped). Still skips cleanly on any interpreter lacking torch-geometric. PnLCalib config is
  unit-tested; the heavy HRNet run is an opt-in integration check.

## Decisions made

- **`football-state-of-play` dependency = runtime path-inject bridge** (not pip-install / not
  vendored). [generator/sop_bridge.py](generator/sop_bridge.py) puts the sibling repo root on
  `sys.path` at call time and imports only `data.graphs.build_data` + `models.gnn.GAT`; we run our
  own thin predict loop. Avoids the top-level `eval` package collision an editable install would
  cause, and keeps a single source of truth (no model-code drift). Override the sibling location
  with `$FOOTBALL_SOP_PATH`.
- **One unified interpreter (FIXED).** The pipeline previously needed two interpreters (CV stack vs
  torch-geometric). Resolved by installing `torch_geometric` (2.8.0, pure-python wheel) into the
  **default py3.14**, which already had the full CV stack + torch. That interpreter now imports
  `cv2`, `ultralytics`, `supervision`, `sports`, `torch`, `torch_geometric` together, builds the GAT,
  loads the checkpoint, and runs `extract.py` AND `sop_bridge.py` in one process. Verified: GAT runs
  directly in py3.14 (no re-exec); all 41 tests pass with the bridge test running (not skipped).
  - The GAT checkpoint was saved on CUDA; `sop_bridge.load_model` now loads with
    `map_location="cpu"` so a CPU-only torch works.
  - `sop_bridge`'s auto re-exec into the SoP `.venv` is kept as a portability fallback (no-op now),
    overridable with `$FOOTBALL_SOP_PYTHON`.
  - Reproducible from clean: `pip install -e ".[cv,dev]"` (torch + torch-geometric are core deps;
    `sports` added to the `[cv]` extra as a git dependency).
- **Calibration: RESOLVED via PnLCalib.** Cloned `mguti97/PnLCalib` to `~/PnLCalib` with SV_kp/SV_lines
  weights; `PnLCalibCalibrator` runs its HRNet detectors and feeds correspondences to our gate.
  Verified sub-metre reprojection error on real frames. (Roboflow `inference` not needed.)

## Research-driven upgrades (this session)

- **RF-DETR detector option** ([generator/extract.py](generator/extract.py) `_RFDetrDetector`,
  `--detector rfdetr`). RF-DETR (Roboflow, DINOv2 backbone, NMS-free; SOTA on COCO, ICLR 2026).
  A/B vs YOLOv8s on 30 real frames ([tools/ab_detectors.py](tools/ab_detectors.py)):
  **ball-detection 0.37 → 1.00**, players/frame 19.3 → 20.2, mean conf 0.58 → 0.66, 0.056 → 0.085
  s/frame on GPU. The ball win directly fixes actor-tagging. Currently COCO-pretrained; the
  football-trained RF-DETR (player/GK/ref/ball, ~83% mAP) is the next upgrade.
- **Visual verification tool** ([tools/visualize.py](tools/visualize.py)): video frame + top-down
  freeze frame side by side, plus track-trajectory plots, for manual inspection.
- **Known weakness found via 2nd-clip viz:** the calibration confidence gate scores *keypoint*
  reprojection error only. A bad camera angle (seg2 / frame 21720) fit keypoints fine yet projected
  most players into one pitch corner (5 valid tracks vs seg1's 19) — and **passed the gate**. Need a
  *player-distribution* sanity check (on-pitch count, spread, both-box coverage), not just keypoint error.

## Tracking upgrade (researched, not yet implemented)

- SoccerNet-GSR SOTA uses **BoT-SORT + global motion compensation (GMC)** + re-ID, not plain
  ByteTrack. GMC matters for panning broadcast cameras. Candidate swap in `detect_track`/extract.

## Open questions for Sid / supervisor

- Confirm footage list + how many FIFA EFI PDFs we can collect (WC22 has them per match).
- Is GS-HOTA an acceptable generator-accuracy anchor?

## Next steps (in priority order)

1. **Make RF-DETR default + load football-trained weights** (player/GK/ref/ball classes) for a
   single-pass role+ball detector; re-A/B.
2. **Strengthen the calibration gate** with a player-distribution sanity check (fixes the seg2 failure
   mode that slipped through).
3. **Attacker rebuild (C3)** — `attacker/{tracks,labels,heads}`: per-player run + receiver heads on
   the dense, ID-persistent tracks the generator now produces, vs the old 360 baselines.
4. **Tracking: BoT-SORT + GMC + re-ID** for ID persistence under camera motion.
5. **Team fingerprints + synthesizer (C5)**; stand up `sn-gamestate` + GS-HOTA; `ingest/fifa_efi.py`.

## Blockers

- None. (GPU now in use — PnLCalib ~0.35 s/frame; the earlier CPU-speed blocker is resolved.)

## Environment (reproduce from clean)

- Unified interpreter: **py3.14**, `torch 2.11.0+cu128` (install:
  `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`),
  plus `pip install -e ".[cv,dev]"`. PnLCalib: clone `mguti97/PnLCalib` to `~/PnLCalib` + SV_kp/SV_lines
  weights under `<repo>/weights`. GPU: RTX 3050 Laptop, CUDA 12.8, ~2 GB VRAM used.
  Detection/tracking are also CPU-bound here.
