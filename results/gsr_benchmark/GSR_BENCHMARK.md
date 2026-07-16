# SoccerNet-GSR benchmark: our pipeline scored with the official GS-HOTA

External accuracy anchor for the CV pipeline. We run our existing generator (football-YOLO detect -> ByteTrack -> PnLCalib calibrate -> project; CIELAB-KMeans team) on the SoccerNet Game State Reconstruction **valid** split (58 sequences x 30 s) and score it with the **official** metric -- we never reimplement it.

## Provenance

- Metric: **GS-HOTA** (arXiv:2404.11335), computed by `sn-trackeval 0.4.0 (commit 9c25232), SoccerNet/sn-trackeval` via `trackeval.datasets.SoccerNetGS` + stock `HOTA`/`Identity`. GS-HOTA appears under the `HOTA` name in trackeval output; the GSR specificity is the similarity (Gaussian on the projected pitch bottom-middle point, sigma from a 5 m tolerance) and the attribute gate.
- Sanity check: a ground-truth-copy submission scores GS-HOTA 100.0 across all configs, confirming the adapter/plumbing.
- Calibration: period-25 temporal reuse on 55 sequences; per-frame on the 3 pilot sequences (SNGS-021/022/023). Measured equivalence on SNGS-021: per-frame vs period-25 GS-LocA 93.2 vs 92.3 (-0.9), full GS-HOTA 31.5 vs 31.0 -- the mix is immaterial (<1 point).
- Team labels 0/1 (unsupervised KMeans) carry no left/right meaning, so per sequence the two-way permutation is resolved against GT by nearest-match agreement (<= 5 m) -- the standard resolution of an arbitrary cluster labelling, disclosed here.

## Headline (combined over the valid split)

| Config | attributes gated | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|---|
| `gs_hota_full` | role + team + **jersey** (official) | **14.8** | 6.0 | 36.4 | 91.3 | 8.1 |
| `no_jersey` | role + team | **43.1** | 50.4 | 36.8 | 92.5 | 45.7 |
| `role_only` | role only | **45.0** | 53.0 | 38.1 | 92.4 | 47.9 |
| `loc_assoc` | none (localization + association) | **48.9** | 60.2 | 39.8 | 92.5 | 51.9 |

**Reading the decomposition.** GS-LocA ~92 says that *when* our pipeline reports a player, its pitch position is essentially where the ground truth puts it -- the calibration + projection geometry is sound. The score is then eroded by attribute gating: adding the **jersey** requirement collapses GS-DetA (official `gs_hota_full`) because our pipeline has **no jersey-number model yet** and emits `jersey = null`, so every GT player with a labelled number is unmatchable. The `no_jersey` -> `gs_hota_full` drop is therefore a direct, external measurement of the pre-Layer-2 identity gap, not a geometry failure.

## Stage 2a: post-hoc track re-linking (appearance ReID) -- the association half

ByteTrack fragments heavily on this footage (mean **79 track ids/sequence** for ~14 people/frame), and
every id break costs GS-AssA and blocks identity propagation -- the weakest link in the decomposition
above (AssA ~37-40 vs LocA ~92). Stage 2a is a **post-hoc merge pass** over the persisted positions
(no re-tracking, geometry untouched): fragments are merge candidates iff temporally disjoint, same
team, same role, and gap-consistent motion (end->start feasible at <= 9 m/s + 2 m calibration slack),
then merged greedily by appearance cosine similarity above a frozen threshold. Appearance = median of
<=10 crop embeddings per fragment from a pretrained torchreid **OSNet** (`osnet_x0_25`, ImageNet), crops
recovered by re-detecting each frame and matching the persisted foot point (the parquet stores only the
projected foot point; `image_x/image_y` are the raw box bottom-middle, so the same detector reproduces
the box). Code: `generator/track_relink.py` (pure merge seams + embedder) + `eval/gsr_score.py --relink`.

**Threshold provenance.** Cosine threshold swept 0.50-0.90 on the 3 pilot sequences (SNGS-021/022/023)
**only**, then **frozen at 0.80** (the pilot-combined GS-AssA/GS-HOTA maximiser) before scoring the
other 55. Re-scored with the same official evaluator into `gsr_scores_relink.json` (baseline
`gsr_scores.json` untouched).

### Before -> after (combined over the 58-sequence valid split, threshold 0.80)

| Config | GS-AssA | GS-HOTA | GS-IDF1 | GS-DetA | GS-LocA |
|---|---|---|---|---|---|
| `gs_hota_full` | 36.4 -> **41.8** (+5.4) | 14.8 -> 15.8 | 8.1 -> 9.3 | 6.0 -> 6.0 | 91.3 -> 91.4 |
| `no_jersey` | 36.8 -> **41.0** (+4.2) | 43.1 -> 45.4 | 45.7 -> 51.4 | 50.4 -> 50.2 | 92.5 -> 92.5 |
| `role_only` | 38.1 -> **42.5** (+4.4) | 45.0 -> 47.4 | 47.9 -> 53.9 | 53.0 -> 52.8 | 92.4 -> 92.3 |
| `loc_assoc` | 39.8 -> **44.7** (+4.9) | 48.9 -> 51.7 | 51.9 -> 58.5 | 60.2 -> 59.9 | 92.5 -> 92.5 |

**Reading.** GS-AssA lifts **+4.2 to +5.4** across every attribute config and IDF1 lifts up to **+6.7**
(loc_assoc 51.9 -> 58.5) -- the identity-linking gain the December story needs. GS-DetA and GS-LocA are
flat by construction (relinking only relabels ids; it adds/removes no detection and moves no position),
so the GS-HOTA gain (+1 to +2.8) is entirely the association half. The practical stat: **mean fragments
per sequence 79 -> 36** (~55% fewer ids; per-seq 51-137 -> 27-51).

### Merge precision (honest, and low)

On the pilot (GT carries per-frame track ids, so true precision is computable) auditable pair-level
merge precision is **70/200 = 35%** at threshold 0.80 (per-seq 28/50, 20/59, 22/91). That is well above
the ~9% random baseline of merging two same-team disjoint fragments, but it means roughly two in three
merges are wrong. **Why AssA still rises:** the 35% correct merges each stitch long same-identity chains
(large AssA/IDF1 reward), while the motion+temporal constraints keep the wrong merges spatially
plausible (small penalty), so the net is strongly positive.

**Diagnosed ceiling.** The cosine similarity of constraint-valid pairs is kit-dominated (median 0.81;
85% of pairs > 0.70), i.e. ImageNet OSNet cannot separate two players **in the same kit** -- exactly the
pairs the constraints leave ambiguous. This is why AssA is near-flat over thresholds 0.50-0.80: the lift
is **constraint-driven**, appearance only breaks near-ties. Upgrade path (Stage 2b): a football/person
ReID model (OSNet Market-1501 or a pitch-tuned embedding) and close-up jersey anchors to disambiguate
same-kit fragments; that is where precision, not just recall, improves.

## Stage 2c: football-domain ReID upgrade -- measured NEGATIVE on the 80% precision gate

Stage 2a shipped the weakest possible embedder: `osnet_x0_25` with **ImageNet-classification** weights
(no re-ID objective at all). Stage 2c swaps in re-ID-objective weights, wired as a parameter
(`OsnetEmbedder(model_name, weights=...)`, `RelinkParams.weights`; `generator/track_relink.py`
`resolve_reid_weights` downloads torchreid model-zoo keys via `gdown`, back-compatible with
`weights=None`). Re-measured on the same 3 pilot sequences at the same frozen threshold 0.80, GT-audited
pair-level merge precision + the same-kit cosine-separation diagnostic (same-player vs different-player
same-kit constraint-valid pairs, n=213 / 1,633):

| Embedder (weights) | merge precision | same-player median cos | diff-player (same-kit) median cos | separation |
|---|---|---|---|---|
| `osnet_x0_25` ImageNet (Stage-2a baseline) | 70/200 = **35.0%** | 0.872 | 0.832 | +0.040 |
| `osnet_x1_0` Market-1501 | 74/214 = **34.6%** | 0.894 | 0.849 | +0.045 |
| `osnet_x1_0` MSMT17 | 59/200 = **29.5%** | 0.879 | 0.838 | +0.041 |
| `osnet_ain_x1_0` MSMT17 (best) | 79/192 = **41.1%** | 0.861 | 0.817 | +0.043 |

**Verdict: person-domain re-ID does NOT clear the pre-committed >=80% bar** (best 41.1%, +6.1 pp over
baseline). The reason is the diagnostic: the same-kit different-player median cosine stays **0.82-0.85**
and the same/different **separation stays ~0.04 across all four embedders**. Market-1501 / MSMT17 / DukeMTMC
re-ID learns to key on *clothing*, and two teammates wear identical clothing -- so a person-re-ID model hits
the exact same kit wall as ImageNet. Four embedders now agree the wall is domain, not capacity. Per the
frozen production gate, **the 58-sequence v2 re-score was NOT run** (merge precision 41% << 80%); relink
stays benchmark-side only and does **not** graduate to per-player facts.

**Downloadable weights inventory (rung 1 vs middle rung).**
- *Middle rung (used here, drop-in):* torchreid `reid_model_factory` model zoo -- `osnet_x1_0_market1501`,
  `osnet_x1_0_msmt17`, `osnet_ain_x1_0_msmt17` (Google-Drive, ~10-17 MB each), same 512-d global feature,
  same forward path as the current embedder. These are the only re-ID weights that drop straight into
  `OsnetEmbedder`.
- *True football rung (documented, NOT integrated):* the sn-gamestate/TrackLab baseline ReID is **PRTreID**
  = `bpbreid`, a **part-based** model (HRNet-32 backbone, gaussian body-part attention) trained on SoccerNet,
  weights `prtreid-soccernet-baseline.pth.tar` at `https://zenodo.org/records/10653453` (+ HRNet backbone at
  record 10604211). It is the one model designed to attend to body regions instead of global kit colour --
  the only untested lever that could plausibly break the same-kit wall. It is **not a drop-in**: it needs the
  uninstalled `prtreid`/`bpbreid` package and its part-attention head, and it is heavy for a 4 GB GPU. Its
  test-time config uses the global embedding only (no pose masks at inference), so a future integration is
  feasible but is Sem-2-scale work, not this rung. sn-reid ships no standalone released weights (checked the
  repo + releases: it is the same torchreid fork, no `.pth` artifacts).

## Identity attributes we emit

| Attribute | Emitted? | Source | Note |
|---|---|---|---|
| role | yes | football-YOLO classes (player/GK/referee) | drives the person/official split |
| team | yes | CIELAB-KMeans kit clustering, per-seq left/right resolved vs GT | |
| jersey number | **no** | -- | Layer 2 (not built); the measured gap above quantifies its future lift |

## Split caveat (do not compare as identical)

We score the **valid** split. The published references -- baseline **GS-HOTA 29.01** and 2024 challenge SOTA **63.90** -- are **test/challenge-split** numbers (arXiv:2404.11335, arXiv:2508.19182). Split, sequence set, and (for SOTA) a full jersey+ReID identity stack all differ, so these are context, not a like-for-like ranking. The honest statement: with **no jersey model**, our official `gs_hota_full` is **14.8**; strip the jersey requirement and localization + association + team recover **43.1**.

## Per-sequence table

| Seq | GT p/frame | our p/frame | jersey-null% | full GS-HOTA | full GS-DetA | loc_assoc GS-HOTA | GS-LocA |
|---|---|---|---|---|---|---|---|
| SNGS-021 | 14.3 | 14.4 | 50% | 31.5 | 22.8 | 57.0 | 93.2 |
| SNGS-022 | 12.1 | 14.9 | 20% | 19.1 | 8.9 | 46.2 | 93.5 |
| SNGS-023 | 19.2 | 18.5 | 5% | 5.5 | 2.5 | 32.8 | 81.4 |
| SNGS-024 | 9.9 | 12.7 | 27% | 11.3 | 8.0 | 23.8 | 90.0 |
| SNGS-025 | 15.5 | 13.7 | 5% | 6.1 | 2.7 | 30.6 | 86.5 |
| SNGS-026 | 10.9 | 10.4 | 10% | 11.2 | 3.2 | 47.6 | 93.7 |
| SNGS-027 | 15.6 | 12.6 | 0% | 3.5 | 0.9 | 19.9 | 90.3 |
| SNGS-028 | 10.1 | 10.3 | 7% | 7.8 | 2.2 | 36.1 | 82.4 |
| SNGS-029 | 11.8 | 12.0 | 16% | 7.6 | 3.9 | 42.6 | 85.1 |
| SNGS-030 | 13.7 | 13.4 | 12% | 11.6 | 4.8 | 37.2 | 90.2 |
| SNGS-031 | 13.3 | 12.2 | 1% | 3.8 | 1.2 | 40.4 | 80.0 |
| SNGS-032 | 12.2 | 11.4 | 32% | 18.6 | 12.0 | 40.1 | 91.9 |
| SNGS-033 | 9.0 | 13.4 | 44% | 24.8 | 15.3 | 37.6 | 93.0 |
| SNGS-034 | 18.2 | 16.8 | 0% | 0.6 | 0.2 | 10.0 | 89.9 |
| SNGS-035 | 14.7 | 15.0 | 35% | 40.1 | 21.3 | 78.5 | 91.2 |
| SNGS-036 | 16.5 | 17.7 | 30% | 27.1 | 11.6 | 67.5 | 89.1 |
| SNGS-037 | 11.9 | 12.5 | 2% | 5.2 | 1.0 | 20.9 | 89.4 |
| SNGS-038 | 13.3 | 11.5 | 0% | 3.2 | 1.8 | 40.4 | 91.9 |
| SNGS-039 | 18.0 | 17.7 | 0% | 6.7 | 2.4 | 45.5 | 92.7 |
| SNGS-040 | 17.2 | 15.5 | 0% | 2.4 | 0.7 | 39.9 | 88.5 |
| SNGS-041 | 13.7 | 13.6 | 0% | 6.8 | 2.5 | 68.6 | 94.2 |
| SNGS-042 | 18.0 | 16.8 | 15% | 9.0 | 3.5 | 50.1 | 91.8 |
| SNGS-043 | 15.6 | 14.8 | 14% | 16.4 | 6.8 | 58.8 | 94.2 |
| SNGS-044 | 17.5 | 15.0 | 29% | 19.7 | 12.9 | 47.5 | 91.2 |
| SNGS-045 | 14.4 | 15.0 | 14% | 13.6 | 6.5 | 47.6 | 94.4 |
| SNGS-046 | 15.9 | 15.1 | 17% | 14.9 | 8.6 | 51.8 | 92.0 |
| SNGS-047 | 15.4 | 15.8 | 4% | 6.6 | 3.5 | 57.5 | 91.6 |
| SNGS-048 | 18.0 | 17.6 | 47% | 32.2 | 22.2 | 62.9 | 91.1 |
| SNGS-049 | 18.6 | 17.7 | 39% | 24.8 | 14.9 | 60.1 | 88.3 |
| SNGS-050 | 19.3 | 16.5 | 0% | 7.1 | 2.3 | 35.0 | 89.4 |
| SNGS-051 | 14.4 | 17.8 | 0% | 2.8 | 0.9 | 59.6 | 88.9 |
| SNGS-052 | 17.2 | 16.9 | 0% | 3.7 | 1.5 | 48.5 | 93.8 |
| SNGS-053 | 16.7 | 16.1 | 14% | 20.6 | 8.5 | 69.2 | 91.6 |
| SNGS-054 | 13.6 | 14.9 | 0% | 3.4 | 1.2 | 46.2 | 94.1 |
| SNGS-055 | 13.5 | 13.1 | 17% | 6.4 | 3.6 | 25.4 | 87.8 |
| SNGS-056 | 12.5 | 15.9 | 7% | 6.5 | 1.9 | 32.5 | 92.9 |
| SNGS-057 | 16.8 | 13.9 | 0% | 8.3 | 2.9 | 42.0 | 86.4 |
| SNGS-058 | 13.9 | 12.4 | 12% | 10.9 | 4.8 | 44.9 | 91.1 |
| SNGS-059 | 15.9 | 15.2 | 35% | 25.9 | 16.4 | 62.3 | 95.3 |
| SNGS-078 | 11.9 | 12.4 | 0% | 3.8 | 1.2 | 49.6 | 94.6 |
| SNGS-079 | 15.7 | 12.1 | 20% | 11.4 | 7.7 | 44.0 | 90.8 |
| SNGS-080 | 13.2 | 13.7 | 2% | 7.6 | 2.2 | 40.0 | 90.7 |
| SNGS-081 | 14.5 | 14.0 | 0% | 5.6 | 2.1 | 47.5 | 91.6 |
| SNGS-082 | 17.0 | 16.7 | 0% | 5.2 | 1.9 | 50.0 | 88.5 |
| SNGS-083 | 14.1 | 13.6 | 12% | 16.8 | 5.6 | 56.6 | 96.6 |
| SNGS-084 | 14.0 | 13.3 | 4% | 5.2 | 2.9 | 50.9 | 92.4 |
| SNGS-085 | 15.6 | 15.4 | 0% | 6.9 | 2.4 | 61.1 | 89.9 |
| SNGS-086 | 17.8 | 16.9 | 6% | 7.8 | 3.5 | 32.9 | 89.9 |
| SNGS-087 | 12.7 | 15.1 | 0% | 6.3 | 1.8 | 50.0 | 89.9 |
| SNGS-088 | 16.1 | 16.1 | 21% | 20.0 | 9.6 | 58.9 | 93.3 |
| SNGS-089 | 15.5 | 14.9 | 9% | 10.3 | 3.8 | 43.5 | 90.7 |
| SNGS-090 | 13.1 | 12.9 | 0% | 4.0 | 0.8 | 40.9 | 84.4 |
| SNGS-091 | 15.4 | 14.8 | 22% | 12.8 | 7.5 | 47.6 | 88.8 |
| SNGS-092 | 15.3 | 15.2 | 5% | 7.7 | 2.5 | 52.2 | 91.4 |
| SNGS-093 | 14.1 | 13.4 | 3% | 7.0 | 1.7 | 38.8 | 95.9 |
| SNGS-094 | 15.6 | 15.6 | 10% | 13.1 | 5.4 | 55.5 | 91.9 |
| SNGS-095 | 11.6 | 14.0 | 24% | 23.0 | 11.0 | 61.2 | 92.8 |
| SNGS-096 | 19.0 | 19.4 | 45% | 29.1 | 20.5 | 52.9 | 91.5 |

## SNGS-023 autopsy (low full GS-HOTA is an annotation-density effect)

SNGS-023 posts a low official GS-HOTA despite healthy geometry. It is **not** a CV failure: our detection yield is ~18.5 players/frame against a GT of ~19.2, and GS-LocA is ~81 (vs ~93 on the pilot). The collapse is the jersey gate meeting a densely-numbered clip: **~95% of its GT players carry a labelled jersey number** (vs 50% on SNGS-021, 20% on SNGS-022), which our null-jersey pipeline cannot match. Full GS-HOTA tracks jersey-null fraction cleanly across the pilot (50% null -> 31.5, 20% -> 19.1, 5% -> 5.5). Its slightly lower GS-LocA reflects a busier 'Offside / not shown' framing (19 players/frame, calib error ~0.46 m vs ~0.28 m). The lesson: the official headline is dominated by identity annotation density -- exactly what Layer 2 targets.

## Limitations

- No jersey model: the official `gs_hota_full` is a pre-Layer-2 floor by construction.
- Team left/right resolved against GT per sequence (disclosed); it affects only `no_jersey`.
- Mixed calibration setting (3 per-frame, 55 period-25), measured equivalent to <1 GS-LocA point.
- Valid split only; not a challenge-leaderboard entry.
