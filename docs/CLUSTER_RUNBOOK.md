# Cluster runbook — spend the first GPU slot training, not configuring

Written 2026-08-01 for an unknown-but-typical university SLURM/Linux GPU cluster. Every step names
its source; every step ends with a one-line check. Companion docs: `docs/GSR_CLUSTER_ROADMAP.md`
(S0-S5 plan), `docs/GSR_CAMPAIGN_BRIEF.md` (why/what), `docs/GSR_METHODS_DEEP_DIVE.md` (methods),
`docs/SOCCERNET_DATA_INVENTORY.md` (data), `data/soccernet/gamestate-2024/README_SPLITS.md`
(what's fetched), `results/GSR_TEST_FLAGPLANT.md` + `results/GSR_DELEAK.md` (pipeline's measured
state).

**Cluster assumptions (stated, not guessed at the terminal):** SLURM scheduler, shared filesystem
with a `$HOME` quota + larger `$SCRATCH`/`$WORK`, module system (`module load cuda/...`), compute
nodes may have restricted internet egress, GPU model unknown until S0 runs.

---

## 1. Session 0 — probe (~half day)

1.1 **GPU model + CUDA.** `nvidia-smi` on an interactive allocation (`srun --gres=gpu:1 --pty
bash`). `nvidia-smi --query-gpu=name,compute_cap --format=csv` for compute capability.
   - Check: prints a GPU name + driver/CUDA version, exit 0.

1.2 **Landmine: sn-gamestate's torch 1.13.1/CUDA 11.7 pin vs H100 (sm_90).** Per
`docs/GSR_METHODS_DEEP_DIVE.md` §3.2, `sn-gamestate`'s `pyproject.toml` pins `python>=3.9,<3.10`,
`torch==1.13.1`, `pytorch-cuda=11.7` — predates sm_90. A100 (sm_80) fine; H100/H200/B200 flagged
untested by that survey. **30 s check** (only needed if `sn-gamestate` itself runs, i.e. only for
1.4):
   ```bash
   python -c "import torch; print(torch.cuda.get_device_capability()); \
     x=torch.randn(4,4).cuda(); print((x@x).sum())"
   ```
   Incompatible pairing raises `CUDA error: no kernel image is available` inside that call.
   - Check: prints a finite number, no kernel-image error.

1.3 **Our env: py3.11 + our requirements.** Repo `pyproject.toml` pins `requires-python
>=3.11`, `torch>=2.2`, `torch-geometric>=2.5`, no CUDA pin — takes whatever CUDA the cluster's
torch wheel targets. This is the env S3 training uses.
   ```bash
   conda create -n footsynth-cluster python=3.11 -y && conda activate footsynth-cluster
   pip install -e ".[cv,dev]"   # from repo root, after the rsync in step 2
   ```
   - Check: `python -c "import torch; print(torch.cuda.is_available())"` -> `True`.

1.4 **Legacy env: only if `sn-gamestate` itself is needed.** Our `generator/`/`eval/` code does
not import `sn-gamestate` or `tracklab` (grep-confirmed) — skip unless a task explicitly needs the
official baseline code.
   ```bash
   conda create -n sn-gamestate python=3.9 -y && conda activate sn-gamestate
   pip install torch==1.13.1 --index-url <cluster's cu117 wheel index, if offered>
   ```
   - Check: only build if a task needs it; otherwise skip, save the half-day.

1.5 **Disk quota + scratch vs home.**
   ```bash
   df -h $HOME; df -h $SCRATCH; quota -s 2>/dev/null || true
   ```
   Repo clone + conda envs on `$HOME` (small, backed up); GSR data mirror (~30 GB) + checkpoints
   on `$SCRATCH`/`$WORK` (large, often purged after N days — read the retention policy first).
   - Check: `$SCRATCH` reports >= 60 GB free.

---

## 2. Data mirror

Per `README_SPLITS.md`, **train+test are already fetched and extracted locally** (train 9.76 GB /
57 clips, test 8.85 GB / 49 clips, valid 11.54 GB / 58 clips — 30.85 GB extracted, zips deleted
post-verification). This is a mirror plan (laptop -> cluster), not a fresh download.

2.1 **rsync from the laptop:**
   ```bash
   rsync -avz --progress \
     "C:/Users/siddh_ygv5bws/football-synthesizer/data/soccernet/gamestate-2024/" \
     <user>@<cluster-host>:$SCRATCH/football-synthesizer/data/soccernet/gamestate-2024/
   ```
   No native rsync on Windows: use `scp -r` or WSL/Git-Bash rsync. Budget for ~30 GB over a
   residential line — the original SoccerNet-side fetch of this volume stalled once at 3.5 h and
   needed a restart (`README_SPLITS.md`), so treat this as multi-hour, not a quick copy.

2.2 **Integrity check vs `README_SPLITS.md`:**
   ```bash
   find $SCRATCH/football-synthesizer/data/soccernet/gamestate-2024 -maxdepth 1 -type d -name 'SNGS-*' | wc -l
   # expect 164 (57 train + 58 valid + 49 test; challenge not fetched)
   find .../SNGS-021/img1 -name '*.jpg' | wc -l   # expect 750
   ```
   - Check: dir count = 164, one sampled `img1/` = 750 frames, one `Labels-GameState.json` parses
     with `info/images/annotations/categories` keys.

2.3 **SoccerNet pip fallback — download directly on the cluster** (faster than uploading from
home internet):
   ```bash
   pip install SoccerNet
   python -c "
   from SoccerNet.Downloader import SoccerNetDownloader
   d = SoccerNetDownloader(LocalDirectory='$SCRATCH/football-synthesizer/data/soccernet')
   d.password = <PASSWORD>   # env var only -- see rule below
   d.downloadDataTask(task='gamestate-2024', split=['train','valid','test'])
   "
   ```
   **Password rule (per `CLAUDE.md`'s credentials constraint): read from an env var set at the
   shell, never write it into a repo file or a logged job script.** `README_SPLITS.md` notes
   `gamestate-2024` uses the SDK's public-data default password, not the NDA one — confirm which
   password a task needs, but either way it's `export SOCCERNET_PASSWORD=...` at the interactive
   shell only.
   - Check: same integrity check as 2.2 against the pip-downloaded copy.

---

## 3. The training target (S3): CLIP-encoder + attribute-heads identity model

Source: `docs/GSR_METHODS_DEEP_DIVE.md` §2.1.4 (Broadcast2Pitch Table 5), §2.1.1-2.1.2. This is
the **60.13 GS-HOTA row** (PRTreID+OCR 18.11 -> **CLIP+heads 60.13** -> LLaMA-3.2-V 61.48), same
detector/tracker/calibration held fixed across all three rows.

**Stated directly:** all Broadcast2Pitch training on **one RTX 4090, 24 GB**; the keypoint/line
model has a full recipe (25 epochs, LR 1e-4, 384x384) but **the CLIP+heads row has no separate
compute paragraph** — only the LLaMA fine-tune (1 epoch, LR 1e-5) is numbered. Heads are
"attribute-specific" (role, jersey, team), contrasted with LLaMA's single multi-task head. CLIP
variant/size is unnamed.

**Reconstructed from our side:** training data = GSR train (57 clips, mirrored per §2),
jersey-GT density **0.762** (`docs/SOCCERNET_DATA_INVENTORY.md` A.1, measured on valid; train
presumed comparable, not separately measured) — 8.8x our broadcast OCR density (0.087,
`results/EVIDENCE_DENSITY_LAW.md`). Crop pipeline already exists:
`generator/jersey_id.py`'s `build_transform`/`torso_crop` (Stage-1c torso band,
`TORSO_BAND=(0.15, 0.55)`) and `KoshkinaRecognizer`'s pose-guided torso extraction reuse directly
for building CLIP-head training pairs from `Labels-GameState.json` labels; `generator/gta_link.py`
`detection_embeddings` shows the existing crop-then-batch-embed call shape a CLIP encoder would
fill. Jersey head can reuse `jersey_id.py`'s exact `NUM_CLASSES=100` (0-99 + illegible) layout.
Team head is worth testing against our already-validated geometric resolver
(`eval.gsr_score.resolve_team_map_free`, 91.8%/98%+ accuracy, `results/GSR_DELEAK.md`) before
committing GPU time — it may be redundant.

**Single-GPU expectation:** every winning GSR system (2024, 2025) trained on one GPU
(`docs/GSR_CLUSTER_ROADMAP.md`, verified). A CLIP-ViT + 3 small heads is smaller than either full
pipeline; should fit one cluster GPU slot. No published GPU-hours for this row — order-of-magnitude
estimate by analogy to the keypoint model's recipe (single-digit GPU-hours), **not sourced,
confirm with the first run.**

**Checkpointing:** no source states a cadence. Recommendation (ours): checkpoint every epoch to
`$SCRATCH`, keep last-3 + best-on-valid.

**Eval-on-valid loop:** reuse the existing harness — `eval/gsr_identity.py`
(`--build-bundles`/`--fit`/`--test`) and `eval/gsr_score.py` (`gs_hota`). A trained model's
per-crop predictions need to land in the same shape `generator/jersey_id.py`'s
`percrop_votes`/`aggregate_votes` already consume, so bundle-building and the solver stay
unchanged downstream — only the evidence *source* swaps.

---

## 4. Integration path

4.1 **Module boundary.** Identity evidence layer = `eval/gsr_identity.py`'s bundle-building
(`build_bundle`, `_tracklet_reads`, `_roster`) + `generator/identity_solve.py`'s solver
(`solve_assignment`, `posterior`, `digit_confusion_prior`). A trained CLIP+heads model is a new
evidence source feeding the same contract the OCR chain feeds today — per-crop distributions in,
tracklet votes out. Solver's confusion prior likely needs re-fitting for the new evidence source's
error pattern (a re-fit task, not a code-structure change).

4.2 **Recompute location.** **Cluster:** the training run itself; full-test-split extraction at
scale (the flag-plant's `extract` stage took 4h42m on a laptop RTX 3050 for 49 sequences,
`results/GSR_TEST_FLAGPLANT.md` §2 — detector/tracker/calibration are CPU/light-GPU bound and may
not scale linearly with a bigger GPU; budget the same order until measured). **Laptop:** solver
fitting (CPU-only per `eval/gsr_identity.py`), bundle-building from cached evidence, GS-HOTA
scoring (CPU), reports/docs. Only evidence extraction through a GPU model needs the cluster.

4.3 **Valid-first-then-one-test-run discipline** (`docs/GSR_CAMPAIGN_BRIEF.md` §6): tune on
train/valid only; freeze the recipe (hash + timestamp, the pattern `gsr_flagplant_frozen.json` /
`gsr_deleak_frozen.json` already set); run test exactly once at frozen config; submit to codabench
4365 at most 1/day. Follow `GSR_TEST_FLAGPLANT.md` / `GSR_DELEAK.md` as precedent, including their
negative sections — the flag-plant's package was found non-uploadable post hoc for reading test
labels via `resolve_team_map` and roster construction; check both the same way before any S3
submission.

4.4 **A finding that should reorder S3's priority.** `GSR_TEST_FLAGPLANT.md` §5: on the official
test split our **GS-AssA (48.21) is the worst column on the leaderboard**, while GS-DetA (26.74,
where identity accuracy lives) is mid-table — the opposite of the campaign brief's working
assumption that jersey/identity dominates. The CLIP identity model targets DetA, not our measured
deficit (association/tracklet continuity). **Before the first GPU slot goes to S3, confirm it's
still the highest-leverage target for our pipeline specifically — an orchestrator call, not
resolved here.**

---

## 3b. Revisions from the code-level repo sweep (2026-08-02, docs/SOCCERNET_REPO_SWEEP.md)

1. **First training experiment is now CHEAPER: retrain PRTreID before writing any CLIP code.**
   sn-gamestate ships a complete live training path (`training_enabled: False` in prtreid.yaml,
   `train()` in prtreid_api.py) — 20 epochs, batch 32, single GPU. Flipping that flag on GSR train
   is the measured floor the CLIP work must beat, and it trains role/team/jersey-aware parts —
   directly attacking the embedder ceiling AND the GK->team link.
2. **S3 data prep is already SPECIFIED (but GPL-3.0 — reimplement, do not copy):**
   sn-gamestate's `prtreid_dataset.py` documents the full GSR->labelled-crop law (min_vis 0.3,
   min 30 px, uniform 15-per-tracklet sampling, min 4 samples/id). The sweep doc records the
   contract; rewrite it in ~50 lines of pandas+cv2 under our license. torchreid internals
   (sn-reid, MIT) are safe to reuse: engine loop, RandomIdentitySampler(num_instances=4),
   transforms menu.
3. **Data upgrade for domain robustness: SoccerNet-v3** (MIT, 400 games, 6 leagues INCLUDING EPL,
   jersey-labelled boxes; sn-reid's 340,993 crops carry weak jersey labels + a built-in
   illegible marker in the filename id field). Train the encoder on GSR train + these — answers
   the Swiss-league domain risk AND pre-builds the eventual ManU adaptation.
4. **Calibration weights are curl-able**: sn-banner ships NBJW SV_kp/SV_lines as GitHub release
   assets (headless-friendly, unlike Drive links) + temporal filter recipes akin to our
   temporal_calib. Optional LocA lever.
5. **S0 probe addition (load-bearing)**: sn-gamestate open issue — "GPU inference returns dummy
   boxes [0,0,1,1] (CPU works fine)". Add an assertion to the probe: run 10 frames, assert box
   variance > 0. A silent near-zero-score failure mode.
6. **Free artifacts**: sn-gamestate commits the baseline's own test-split submission zip (69 MB)
   — a submission-format validator + score floor with zero GPU spent.
7. **License wall, recorded**: GPL-3.0 on sn-gamestate/sn-teamspotting/sn-banner (reimplement or
   isolate); sn-calibration/sn-tracking/sn-jersey have NO license (do not vendor anything).

## 4a. OMNI cluster specifics (IIIT Delhi, policy v2.0 June 2026 — supplied by Sid 2026-08-01)

Hardware: 2x B200 (180-192 GB), 2x H200 (144 GB; Short queue = MIG slices 3g.71gb ~70 GB),
1x A100 DGX (DOWN). SLURM with -A GroupName (NEEDS a faculty group — the one thing Sid must get
from the prof). 5 TB group storage (our ~100 GB ask is trivial).

**Queue strategy (token costs matter: Short 0.1/job, Medium 0.5, Long 1.0):**
- Session 0 probe + crop extraction + eval loops -> **Short** (MIG 70 GB, 2 h default / 6 h max,
  3 jobs/group, cheapest). A 70 GB MIG slice ~ A100-class throughput: fits everything we do.
- Training runs -> Short (6 h max, resume-per-epoch) or **Medium** (up to 1 day, dedicated
  H200/B200) for uninterrupted runs. Long (3 days) likely never needed.
- Template: the policy's sbatch skeleton + `--gres=gpu:3g.71gb:1 --mem=64G` (Short) or
  `--gres=gpu:1 --nodelist=hgxh200 --mem=60G` (Medium).

**Compatibility ordering (updates the S0 landmine):** prefer **H200 first** (sm_90: torch >=2.3
cu121 wheels fine). **B200 is Blackwell (sm_100): needs very recent torch (>=2.6 / CUDA 12.8
wheels)** — verify with the 30-s probe before scheduling training there. The legacy sn-gamestate
env (torch 1.13.1/cu117) will run on NEITHER — confirmed irrelevant since our code does not
import it.

**Policy compliance notes:** academic workload (fine); checkpoint discipline doubles as
anti-idle compliance; NDA data lives in group storage (policy: data private, only usage metrics
public — consistent with SoccerNet NDA research use). Do not share login credentials — Sid runs
`sbatch`; agents prepare scripts.

**Colab/Kaggle fallback verdict (if group approval drags):** free Colab (T4 16 GB, ~4 h
sessions) fits crop extraction and SMALL training probes only with aggressive per-epoch
checkpointing; Kaggle (30 h/week, 12 h sessions) is the better free stopgap. Both are stopgaps —
one OMNI Short slot outclasses either. Do not architect for them.

## 4b. Orchestrator's resolution of the S3-vs-AssA tension (Fable, 2026-08-01)

Section 4 flags that our worst official column is GS-AssA (association), arguing against S3 (an
identity/DetA lever) as the first GPU spend. Resolution: **the tension is apparent, not real,
because both columns are blocked by the same component — the appearance encoder.** Measured
chain: GSR_ASSOCIATION.md shows oracle linking on our own detections reaches raw AssA ~68 and
that the GTA connector captures only 24.5% of that headroom, blocked by PRTreID's same-kit
embedding overlap (within-identity p90 0.093 vs cross-identity p10 0.040-0.083, GTA_LINK_STAGE1);
three pre-registered tests confirm that ceiling. The CLIP encoder S3 trains IS the replacement
for that embedder. Therefore: train the SHARED encoder first, with both consumers in mind —
(a) its embeddings feed the tracklet connector (association -> AssA), (b) its attribute heads
feed the solver (identity -> DetA). Evaluate BOTH effects on valid after training. The team head
vs geometric-resolver question (UNKNOWN #7) stays open; the GK->team link (GSR_TEAMSIDE.md:
keeper rule 113/113 on GT, blocked at 0.244 link accuracy) is a mandatory eval target for the
trained encoder — it converts directly to ~+2.4 GS-HOTA via the side resolver.

## 5. Slot-size guidance

**4-hour slot:** S0 probe end to end; data-mirror verification; a small CLIP+heads training smoke
test (few epochs, train subset, confirm the loop runs and loss is sane) — not a full run (epoch/LR
count is UNKNOWN, see below). Does not fit: full test-split extraction (4h42m alone on a laptop
GPU, likely still multi-hour on cluster given CPU-bound stages) or full training to convergence.

**Day slot:** a full CLIP+heads training run (single-digit GPU-hours by analogy) plus a valid-split
eval pass; or a full test-split extraction + one frozen scored run, mirroring the flag-plant's
~8.5 h total wall-clock (extract+embed+OCR+connect+solve+package).

**Checkpoint discipline on unexpected slot end:** checkpoint every epoch to `$SCRATCH` (not
`$HOME`) with the frozen-recipe hash in the filename or an adjacent JSON, written before the run
starts. Resume from the last complete epoch, never restart at 0. If a slot dies mid-extraction, the
flag-plant precedent is directly applicable — that pipeline is resumable per-sequence (a kill at
8/49 sequences resumed cleanly, losing only the in-flight one); confirm the same holds for any new
CLIP-based extraction path before relying on it under slot pressure.

---

## UNKNOWN list — needs the first experiment, not more reading

1. CLIP variant/size (ViT-B/32, B/16, L/14, other) for the 60.13 row — not stated.
2. Frozen backbone + linear-probe heads, vs full fine-tune — not stated.
3. Per-head loss weights; joint vs staged head training — not stated.
4. Epochs and LR for the CLIP+heads row specifically — not stated (only the keypoint model and
   the LLaMA row have numbers).
5. Batch size, augmentation, input crop resolution for this row — not stated.
6. Real single-GPU wall-clock for this training run — not stated, only inferable by analogy.
7. Whether a trained team head beats our validated 91.8-98%+ geometric resolver — a local
   ablation, answerable without more literature.
8. Whether S3 (identity/DetA lever) is still the right first GPU spend given GS-AssA, not DetA, is
   our worst column on the official test split (§4.4) — an orchestrator decision, not a data gap.
9. H100/H200/B200 compatibility of `sn-gamestate`'s pinned torch 1.13.1/CUDA 11.7 env — flagged,
   never executed; only matters if the legacy env (§1.4) is ever needed.
