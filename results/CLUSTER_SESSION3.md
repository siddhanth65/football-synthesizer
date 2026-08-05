# Cluster session 3 — scaling the identity corpus (2026-08-04)

Session 2 (`results/CLUSTER_SESSION2.md`) ended with a measurement, not a guess: CLIP ViT-B/16 +
heads on GSR train alone peaked at **53.75** identity mAP and decayed, and session 1's from-scratch
hrnet32 hit the same wall at 55.80 — both below the shipped PRTreID floor of **56.93 / 78.17**. Two
architectures agreeing pointed at the data, so the recommendation was *more identities, not more
epochs*. This session executes that.

**Headline: it works. Identity mAP 58.75, +1.82 over the floor and +5.00 over the GSR-only run,
with R1 76.70 vs 73.95.** The bar set for this session — clear 56.93 decisively, i.e. >= 58, outside
the +/-1 noise band — is met at **four consecutive checkpoints (epochs 6, 8, 10, 12)**, so this is a
plateau rather than a lucky epoch. Team mAP is marginally sacrificed (77.3-78.0 vs 78.17).

The second arm — fine-tuning the shipped PRTreID checkpoint on the same corpus — **hit a wall** and
is reported as such, with the full diagnosis chain, in §6.

---

## 1. Data fetch — verified

Disk gate cleared before downloading: 845-862 GB free globally (threshold was 100 GB), quota 71.6 /
500 GB.

Per `docs/SOCCERNET_DATA_INVENTORY.md`'s own recommendation ("Fetch train only (12.12 GB) if the
goal is a better embedder"), only the **train** split was fetched. Our evaluation is the GSR valid
split, so sn-reid's valid/test/challenge (6.5 GB) buy nothing here.

| | expected | measured |
|---|---|---|
| download | 12.12 GB | **12,112,383,260 B = 12.11 GB** |
| extracted crops | — | **248,234 PNG**, 13 GB |
| matches | 400 (all splits) | **290** in train, 6 leagues |

No password needed (`task="reid"` is public). The SDK's no-timeout stall did **not** recur —
sustained 7-10 MB/s, 23 min, one pass. Zip deleted after count verification.

`clazz` in the filename turned out to be richer than the repo sweep recorded — it encodes **role and
team side together**:

```
Player_team_left 85,204 | Player_team_right 84,967 | Main_referee 8,545
Goalkeeper_team_left 6,468 | Goalkeeper_team_right 5,986 | Side_referee 5,888 | Staff_members 2,942
```

So sn-reid supervises identity, role **and** team, not identity alone. That was not in the plan and
materially improved the merge.

## 2. Unified training set

`~/work/build_unified.py`. sn-reid filenames carry everything needed, so **no image is opened** —
the whole merge is a filename parse, 4 seconds. Field split follows sn-reid's own
`extract_sample_info` exactly (plain split on `-`, requiring 8 fields; the trailing size field is
**HxW**, not WxH as the schema comment implies).

| | GSR | sn-reid | unified |
|---|---|---|---|
| crops | 20,067 | 140,003 | **160,070** |
| identities | 1,343 | 58,697 | **60,040** |

**44.7x the identities, 8.0x the crops.**

**The crop law had to differ per source, and that is the load-bearing decision of this session.**
Applying our GSR law (`min 4 crops/id`) to sn-reid kept only 21,420 crops over 4,676 ids — 91% of
the corpus discarded. The reason is structural: SoccerNet re-ID identities are **same-action
cross-view pairs**, not tracklets. Measured distribution over 150,803 identities:

| crops/id | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|
| ids | 92,106 | 43,480 | 10,541 | 2,912 | 1,121 | 643 |

`min_per_id = 2` is the correct operating point: it keeps every identity that has a genuine positive
pair (58,697 ids / 140,003 crops) and drops the 92,106 singletons, which can contribute no positive
and would only inflate the ArcFace class count. Size floor (30 px) dropped 16,125 crops (6.5%).

**Jersey supervision is taken asymmetrically, on purpose.** sn-reid's `id` field is the jersey number
if it is visible *at least once in the action*, and a letter if it is never visible.

- **letter -> abstention class.** Sound: if the number is invisible in every frame of the action it
  is invisible in this crop. 78,895 crops.
- **number -> ignored (-100).** "Visible once in the action" does **not** mean visible in *this*
  crop, and feeding it to a per-crop head teaches hallucination. 61,108 crops ignored rather than
  used.

This is the abstention discipline applied to weak labels: we take the negative evidence, which is
per-crop valid, and refuse the positive evidence, which is only action-level.

**Identity spaces are disjoint and unioned**, asserted in code. sn-reid identities are within-action
only, so two crops of the same real player in different actions are different classes. Every
positive pair is genuine; the cost is false negatives (the same player pushed apart across actions).
That is a real limitation and it is *not* worked around — it is arguably mild for our target, since
the eval protocol (`mot_intra_video`) is itself within-video retrieval, not full-match persistence.
It would matter far more for long-horizon naming, which `GSR_METHODS_DEEP_DIVE` §4.2.3 already flags.

## 3. Arm (a) — CLIP ViT-B/16 + heads on the unified set

Session 2's recipe unchanged except the manifest: batch 128, bf16, 12 epochs, 2 frozen then last 6
blocks at 1e-5 vs heads 1e-3, cosine LR. `n_ids 60,040 / n_teamvid 17,295 / n_jersey 42 / n_roles 5`.
110 s/epoch frozen, 204 s/epoch unfrozen — **41 min total** on one shared A100.

Losses descend monotonically throughout: total 26.75 -> 4.84, identity 19.97 -> 1.14, jersey
2.83 -> 0.24, role 1.09 -> 0.07. Team loss falls only 9.64 -> 7.09, which is expected — 17,295
per-action team classes with ~2 crops each is a very sparse target.

## 4. Results vs the floor

All numbers from the session-2 evaluator, verified to reproduce the floor to within 0.02 mAP. Full
valid split, query 2,845 / gallery 11,376.

| model | identity mAP | R1 | team mAP | team R1 |
|---|---|---|---|---|
| **floor — shipped PRTreID** | **56.93** | 73.95 | **78.17** | 97.14 |
| session 2 best (GSR only) | 53.75 | 70.26 | 76.40 | 95.38 |
| unified, epoch 2 | 43.26 | 57.54 | **81.81** | 93.94 |
| unified, epoch 4 | 56.17 | 71.60 | 78.72 | 95.99 |
| unified, epoch 6 | 58.00 | 74.76 | 77.32 | 95.83 |
| **unified, epoch 8** | **58.75** | **76.70** | 77.45 | 96.28 |
| unified, epoch 10 | 58.53 | 76.84 | 77.53 | 96.07 |
| unified, epoch 12 (final) | 58.41 | 76.66 | 77.96 | 96.19 |

**Identity: the bar is met.** +1.82 over the floor at peak, and — more important than the peak —
epochs 6/8/10/12 read 58.00 / 58.75 / 58.53 / 58.41. That is a **plateau**, not a spike, and the
*final* epoch clears the floor by +1.48 without any checkpoint selection. R1 gains more than mAP:
+2.75. Against session 2's GSR-only run the gain is **+5.00 mAP / +6.44 R1**, from the same code
with a different manifest.

The overfitting signature that defined session 2 is **gone**: GSR-only peaked at epoch 10 of 40 and
decayed to 49.32; unified rises to a plateau and stays. That is the direct confirmation that data
volume, not schedule or architecture, was the binding constraint.

**Team: marginally sacrificed.** 77.3-78.0 at the identity plateau, against the 78.17 floor — a
0.2-0.9 deficit, inside the noise band but consistently on the low side. Epoch 2 reads **81.81
(+3.64)** and epoch 4 reads 78.72, so the team signal is strongest before identity training
dominates the shared embedding — the same tension session 2 recorded, now with a better peak. A
brief with a hard "team not sacrificed" requirement is **not** fully satisfied at the identity
optimum; epoch 4 (56.17 / 78.72) is the balanced compromise if both must clear.

## 5. What this settles

`clip-s3-002` claimed GSR train (1,343 ids) was too small and that more identities was the lever.
**Tested and upheld.** 44.7x identities moved the same architecture from 3.18 below the floor to
1.82 above it, and removed the overfitting decay. The shipped PRTreID checkpoint's advantage was
indeed its larger re-ID pretraining, and it is now matched and beaten by an encoder we control.

Caveats that keep this honest:

- The peak is +1.82 with a within-run eval series spanning 58.00-58.75; the honest interval for the
  gain over the floor is roughly **+1.1 to +1.8**, not "+1.82" as a point estimate.
- Best-epoch numbers are selected on the same valid split they are reported on. The unselected
  final epoch (58.41) is the number to quote if only one is allowed.
- This is re-ID mAP on a crop-retrieval protocol. It is **not** GS-HOTA, and it does not entitle
  anyone to compare against Broadcast2Pitch Table 5's 60.13.
- No test-split run, no submission. Everything here is valid-split.
- One seed, one recipe.

## 6. Arm (b) — WALL: prtreid's training stack cannot ingest cross-view-pair data

The second arm (gentle fine-tune of the shipped PRTreID checkpoint on the unified corpus, per
probe-1's LR lesson) was scripted (`~/work/prtreid_unified.py`) and **never reached a training
step**. Five distinct structural incompatibilities, each fixed in turn until one could not be:

| # | failure | cause | resolution |
|---|---|---|---|
| 1 | `AttributeError: 'UnifiedReid' has no attribute 'get_masks_config'` | dataset-class API expected by `build_config` | added the hook, returns None -> IdentityMask |
| 2 | `AttributeError: ... no attribute 'column_mapping'` | `PrtreidSampler` indexes int->str inverse maps | supplied them; unknown team -> sampler's `other` bucket |
| 3 | `ValueError: Sample larger than population` in `PrtreidSampler` | it is **team-aware** and needs >= 2 identities per `(video, team)` bucket; sn-reid actions often have exactly 1 | switched to `RandomIdentitySampler` |
| 4 | same error in `RandomIdentitySampler` | **this fork's** version is also camid-scoped and needs >= `batch/num_instances` (16) ids per camid | pooled train camids into one bucket (a sampling-only field; query/gallery keep real camids) |
| 5 | `TypeError: cannot unpack non-iterable NoneType` in `GiLt_loss.compute_triplet_loss` | `part_averaged_triplet_loss` returns None for this batch composition | **not resolved** — reproduced at `num_instances` 2 and 4 |

**The finding worth keeping:** prtreid/sn-gamestate's training stack is coupled end-to-end to GSR's
data shape — long tracklets, many identities per video, team-balanced batches drawn from a single
match. SoccerNet re-ID's shape (cross-view pairs, a handful of identities per action) violates that
assumption at the sampler *and* at the loss. Anyone planning this arm should budget for replacing
their sampler and loss, not for a config change. Our own trainer, which assumes nothing beyond
`(crop, labels)`, ingested the same corpus without a single change.

Timeboxed and stopped after the fifth failure rather than continuing to patch their internals.

## 7. State and artifacts

Quota **88,403 MB / 500 GB**; volume 845 GB free (92%). GPU 0 untouched all session; GPU 1 returned
to the co-tenant's 2.1 GB / 31%. No jobs of ours running. Verified zip deleted; unevaluated
checkpoints and the failed arm-(b) run directory pruned.

Server-side (`siddhanth23519@a100server1`), nothing vendored into this repo:

- `~/work/build_unified.py` — the merge (filename-only, 4 s) -> `~/work/crops_unified.json`
- `~/work/prtreid_unified.py` — arm (b), blocked at failure 5
- `~/work/train_clip.py`, `~/work/clip_model.py`, `~/work/eval_reid.py` — session 2, unchanged except
  a `--manifest` flag and ignore-index handling for missing labels
- `~/data/snreid/reid/train/` — 248,234 crops, 13 GB
- `~/runs/clip_uni/epoch{2,4,6,8,10,12}.pt` (best: **epoch8.pt**), `~/logs/clip_uni.log`,
  `~/logs/reid_dl.log`, `~/logs/prtreid_uni.log`

`build_unified.py` asserts the pid space is dense and that the two sources' identity spaces are
disjoint — the two ways a silent merge bug would corrupt the ArcFace target.
