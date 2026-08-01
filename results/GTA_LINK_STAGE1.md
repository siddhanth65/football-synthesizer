# GTA-Link Stage 1 — tracklet repair (Splitter + Connector), measured

Date: 2026-07-27. Plan: `docs/ATTRIBUTION_RESEARCH_PLAN.md` Stage 1.
Method source: Sun et al., *GTA: Global Tracklet Association for Multi-Object Tracking in Sports*
(ACCV 2024 W, arXiv:2411.08216). Implemented from the spec; no network, no training, CPU-side.

Code: `generator/gta_link.py` (algorithm, `GTA_VERSION = "gta-link-1.0"`),
`eval/gsr_gta.py` (GSR benchmark driver), `tools/gta_match.py` (real match),
`tests/test_gta_link.py` + `python -m generator.gta_link` (self-check).

**Headline: GS-HOTA 22.85 -> 24.18 (+1.33) on the SoccerNet-GSR valid split, 51 of 58 sequences
helped. The whole gain comes from the Connector. The Splitter is a net negative and is shipped
OFF.**

---

## 1. Inventory (the blocker, and why it was not a stopper)

The shipped PRTreID cache `outputs/gsr/relink_cache_prtreid/*.npz` holds **one median vector per
fragment** (`embeddings` shape `(n_tracks, 256)`, built from <= 10 crops per fragment by
`tools/prtreid_probe.py`). The GTA Splitter needs **per-detection** embeddings, which existed
nowhere. `results/carrier_attr/*_gallery_emb.npy` are per-detection but only over the OCR-named
gallery subset (fulham: 902 crops / 147 tracks of 8,905 tracklets), so they cannot drive a splitter.

Recomputed as a GPU job (one at a time, PRTreID SoccerNet-baseline, frame stride 2):

| target | scope | detections embedded | cache | wall time |
|---|---|---|---|---|
| SoccerNet-GSR valid | 58 sequences | 312,830 | `outputs/gsr/detembed_cache_prtreid/` 210 MB | ~62 s/seq, ~60 min |
| fulham_manutd | 11 chunks (both halves) | 83,210 | `outputs/fulham_manutd/final/gta/` 57 MB | ~4.7 min/chunk, ~52 min |

Headroom measured **before** building anything (CPU, GT-audited, all 58 sequences): 10.96% of
auditable ByteTrack fragments are contaminated (a second GT identity holding >= 20% and >= 10 rows),
row-level impurity 9.23%, mean fragment purity 0.912, 3.11 fragments per GT player. Both stages
therefore had real headroom on paper.

## 2. What was built

**Splitter.** DBSCAN (cosine) over a tracklet's per-detection embeddings; >= 2 clusters means the
tracklet is cut into temporally contiguous sub-tracklets at cluster changes. Noise points inherit
their nearest labelled neighbour *in time*; runs shorter than `min_run` are absorbed, so a blip
cannot manufacture a fragment.

**Connector.** True agglomerative merging: distance = cosine distance between tracklet **mean**
embeddings, mean **recomputed after every merge** (this is what differs from the shipped
`generator.track_relink.greedy_merge`, which is single-linkage over frozen per-fragment medians).
Hard gates: no temporal overlap, same team, same role, end-of-A -> start-of-B reachable at
<= 9 m/s (+2 m slack), distance < tau.

**One deviation from the paper, forced by the evaluator.** GTA tolerates <= 1 frame of temporal
overlap. Our default is `overlap_slack = 0`, because the SoccerNet evaluator rejects a submission
outright — *"Tracker predicts the same ID more than once in a single timestep"* — which a 1-frame
overlap produces. This was found by the run failing, not by reading. The parameter is kept and the
1-frame behaviour is pinned in `tests/test_gta_link.py`.

**Scale calibration (the reason the first attempt did nothing).** The paper's eps values are for
general-purpose ReID. PRTreID on same-kit football sits on a much tighter scale: measured
within-identity per-detection cosine **distance** median 0.044 (p90 0.093), cross-identity median
0.11-0.15 (p10 0.040-0.083). At the initial eps = 0.30 every tracklet is one DBSCAN cluster and
**zero** tracklets split. Usable eps is 0.03-0.08. The two distributions overlap heavily, which
foreshadows the negative result below.

## 3. Threshold sweep (3 frozen pilot sequences, `eval.gsr_score.PILOT_SEQS`)

Tuning was done on the pilot only, then applied to all 58 — the discipline the earlier re-link arms
used. Pilot fragment purity is 0.7995 (the *contaminated* end of the split; full-split purity is
0.9077), which turns out to matter a great deal.

| eps | tau | split | tracklets split | purity | frags/seq | merge prec | GS-HOTA | loc HOTA | loc AssA |
|---|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.020 | yes | 45 | 0.7995->0.8799 | 104->78 | 85/101 | 26.97 | 50.16 | 44.05 |
| 0.03 | 0.030 | yes | 45 | 0.7995->0.8799 | 104->64 | 138/189 | 28.89 | 53.49 | 50.22 |
| 0.03 | 0.040 | yes | 45 | 0.7995->0.8799 | 104->55 | 171/255 | 29.13 | 54.43 | 52.01 |
| 0.03 | 0.050 | yes | 45 | 0.7995->0.8799 | 104->50 | 183/276 | 29.18 | 54.93 | 52.97 |
| 0.03 | 0.060 | yes | 45 | 0.7995->0.8799 | 104->47 | 190/306 | 29.18 | 54.97 | **53.10** |
| 0.04 | 0.040 | yes | 41 | 0.7995->0.8752 | 95->57 | 114/167 | 29.18 | 54.01 | 51.19 |
| 0.04 | 0.050 | yes | 41 | 0.7995->0.8752 | 95->52 | 128/197 | **29.27** | 54.55 | 52.23 |
| 0.04 | 0.060 | yes | 41 | 0.7995->0.8752 | 95->47 | 135/225 | **29.27** | 54.53 | 52.17 |
| 0.05 | 0.050 | yes | 38 | 0.7995->0.8686 | 95->49 | 151/222 | 29.00 | 54.01 | 51.15 |
| 0.06 | 0.050 | yes | 28 | 0.7995->0.8399 | 89->48 | 136/198 | 28.70 | 52.78 | 48.97 |
| — | 0.040 | no | 0 | 0.7995 | 75->51 | 63/88 | 28.17 | 50.95 | 45.74 |
| — | 0.050 | no | 0 | 0.7995 | 75->46 | 77/109 | 28.26 | 51.45 | 46.64 |
| — | 0.060 | no | 0 | 0.7995 | 75->43 | 84/128 | 28.32 | 51.47 | 46.69 |

(Full 25-row grid: `results/gsr_benchmark/gsr_gta_sweep.json`.)

On the pilot the Splitter looks like a clear win: +0.95 to +1.0 GS-HOTA and +5 to +6 loc AssA over
its own no-split twin at matched tau. **This does not survive the full split.** Frozen picks:
eps = 0.04 / tau = 0.05 with splitter, tau = 0.06 without.

## 4. GS-HOTA on the full SoccerNet-GSR valid split (58 sequences)

Baseline on record = `eval.gsr_prtreid_relink` at cosine 0.960 + unanimous jersey propagation
(`results/gsr_benchmark/gsr_scores_prtreid_propagate.json`). Every GTA arm rewrites **only**
`track_id`/`id` (+ propagated jersey) on the same `eval_koshkina` submissions, so geometry, role,
team and confidence are byte-identical and the partition is the only variable.

| arm | split | tau | frags/seq | merges | merge prec | **GS-HOTA** | GS-DetA | GS-AssA | loc HOTA | loc AssA |
|---|---|---|---|---|---|---|---|---|---|---|
| on record (single-linkage, median emb) | no | 0.040 | 79.3->64.4 | 868 | 81.0% | 22.85 | 11.12 | 46.96 | 52.48 | 45.86 |
| GTA connector | no | 0.040 | 79.3->60.3 | 1101 | 80.1% | 23.53 | 11.34 | 48.82 | 53.60 | 47.85 |
| GTA connector | no | 0.050 | 79.3->56.3 | 1335 | 75.0% | 24.04 | 11.61 | 49.79 | 54.15 | 48.85 |
| **GTA connector (frozen pick)** | no | 0.060 | 79.3->52.5 | 1557 | 69.7% | **24.18** | 11.68 | 50.08 | **54.41** | **49.34** |
| GTA split+connect (frozen pick, eps 0.04) | yes | 0.050 | 111.4->66.7 | 2594 | 78.4% | 23.85 | 11.93 | 47.66 | 52.55 | 45.79 |
| GTA split+connect (post-hoc, eps 0.08/mr 15) | yes | 0.050 | 87.6->60.2 | 1594 | 75.0% | 24.24 | 11.77 | 49.94 | 53.84 | 48.11 |

Paired per-sequence deltas against the on-record arm (n = 58):

| arm | mean | median | helped | hurt | worst seq | best seq |
|---|---|---|---|---|---|---|
| connector only, tau 0.060 | **+1.160** | +0.430 | **51** | 7 | -0.17 | +8.17 |
| connector only, tau 0.050 | +1.045 | +0.395 | 51 | 7 | -0.17 | +8.31 |
| split+connect, eps 0.04 (frozen) | +1.040 | +0.810 | 38 | **20** | **-5.80** | +8.07 |
| split+connect, eps 0.08 (post-hoc) | +1.329 | +0.712 | 45 | 13 | -2.63 | +7.93 |

**Isolated connector result.** At matched threshold (tau 0.040 = the on-record cosine 0.960) and
essentially matched merge precision (80.1% vs 81.0%), the GTA connector produces 27% more correct
merges (1093 vs 865) and **+0.68 GS-HOTA / +1.99 GS-AssA**. That difference is attributable to two
changes, both part of the GTA spec: tracklet embeddings are means over *all* sampled detections
(~54/tracklet) rather than medians over <= 10 crops, and merging is agglomerative with mean
recomputation rather than single-linkage over frozen pair similarities.

Fragmentation on GSR: fragments per GT player **5.87 -> 4.72** (connector, tau 0.060).

## 5. Negative result: the Splitter does not pay on this data

1. **It reverses sign between pilot and full split.** Pilot (purity 0.7995): +0.95 GS-HOTA over its
   no-split twin. Full split (purity 0.9077): **-0.19 GS-HOTA (23.85 vs 24.04) and -3.06 loc AssA**
   at the pilot-frozen setting. The pilot is the contaminated tail; on 91%-pure data the Splitter
   spends most of its cuts on clean tracklets.
2. **It roughly doubles fragmentation before the connector can undo it**: 87.2 -> 111.4 tracklets
   per sequence, and fragments per GT player end *worse* than no-split (5.37 vs 4.72).
3. **It barely removes contamination at scale**: contaminated tracklets 411 -> 381 (-7%) on the full
   split, against 11 -> 3 on the 2-sequence probe. Purity 0.9077 -> 0.9423.
4. **It costs consistency**: 20 of 58 sequences hurt, worst -5.80 GS-HOTA, versus 7 hurt / worst
   -0.17 for the connector alone.
5. A post-hoc conservative setting (eps 0.08, `min_samples` 8, `min_run` 15) does reach the best
   single number, 24.24. **That setting was chosen after seeing full-split scores and is therefore
   not a clean result** — it is reported, not claimed. Even it loses to no-split on loc AssA
   (48.11 vs 49.34).

**Control against arithmetic.** Splitting a tracklet raises purity mechanically. At an equal number
of cuts on the same tracklets but with *random* cut points (5 seeds, SNGS-021/022): purity
0.940-0.945, contaminated 8-12. The appearance splitter at eps 0.04: purity 0.9672, contaminated 3.
So the splitter's purity gain is genuinely appearance-driven — it simply does not convert into
GS-HOTA once the connector is also running.

The honest reading: GTA's Splitter targets ID switches inside long tracklets. ByteTrack on this
data *breaks* rather than *switches* — it emits 79 short fragments per 750-frame sequence — so the
contamination it does carry is a smaller prize than the fragmentation, and the appearance signal
(within-identity p90 0.093 vs cross-identity p10 0.040-0.083 in cosine distance) is not clean enough
to cut only where it should.

## 6. Real broadcast: fulham_manutd

Positions via `core.registry`; 166,519 person rows, 8,905 tracklets, 11 chunks. Ground truth for the
audit is the OCR-named track set (`outputs/identity/fulham_manutd_named_tracks_both2_prtreid.parquet`,
155 named tracklets over 23 player-halves, anchor reads 98.6% verified).

| arm | tracklets split | fragments | tracklets per named-player-half | median | max | named-merge precision |
|---|---|---|---|---|---|---|
| baseline | — | 7,381 | **6.739** | 3.0 | 25 | — |
| connector, within chunk, tau 0.06 | 0 | 1,946 | **5.043** (-25.2%) | 3.0 | 21 | 82.8% (5/29 groups mix 2 names) |
| connector, **cross-chunk within half** | 0 | 1,131 | **3.783** (-43.9%) | 2.0 | 17 | 71.4% (8/28 mix) |
| split+connect, eps 0.04 (mr 5) | 86 | 1,950 | 5.130 (worse) | 3.0 | 20 | 79.3% (6/29) |
| split+connect, eps 0.04 (mr 15) | 12 | 1,945 | 5.087 (worse) | 3.0 | 21 | 82.8% (5/29) |

Same verdict as GSR: the connector helps, the splitter makes fragmentation and precision slightly
worse. On real footage the Splitter is additionally **starved by construction** — the aligned
positions are sampled every 5 frames and we embed every 2nd sample, so the median tracklet carries
**6** embeddings (only 32.8% have >= 10). DBSCAN at `min_samples = 5` has nothing to cluster. Any
future splitter attempt on real matches must first raise sampling density, which is a GPU-cost
decision, not an algorithm decision.

Cross-chunk connection is the largest fragmentation win available (6.739 -> 3.783) but it buys that
by merging across chunk boundaries where the 9 m/s motion gate is effectively inert, so those merges
rest on appearance alone — and the named-merge precision drop (82.8% -> 71.4%) is the price. It is
reported, **not** recommended for the identity chain until Stage 2's mutual-exclusion constraints can
arbitrate.

Artifacts: `results/gta_match_fulham.json`, `results/gsr_benchmark/gsr_scores_gta.json`,
`results/gsr_benchmark/gsr_gta_sweep.json`. No prior results file was overwritten.

## 7. Carrier-attribution factorisation: not run, and why

`results/CARRIER_CONSTRAINED_v2.md`'s 0.846 x 0.719 x **0.609** factorisation is tuned on
`manutd_liverpool` and scored on the 60 held-out labels of `manutd_tottenham` + `manutd_brighton`
(`tools/carrier_constrained.py`). **fulham_manutd carries none of those 90 hand labels**, so
re-running the factorisation on it would produce no comparable naming number. Moving the 0.609
factor requires the per-detection embedding pass on the three *labelled* matches — a further ~2.5 h
of GPU at the measured 4.7 min/chunk. Not started; flagged for the orchestrator to schedule.

## 8. Verdict for Stage 1

- **Ship**: the Connector, splitter OFF, tau 0.06. GS-HOTA 22.85 -> 24.18, 51/58 sequences helped,
  fragments per GT player 5.87 -> 4.72, real-match tracklets per named-player-half 6.739 -> 5.043.
- **Do not ship**: the Splitter. Negative on the full split at its frozen setting, high variance,
  and structurally starved on real-match sampling density.
- Caveat on merge precision: the frozen tau 0.06 runs at 69.7% GT-audited merge precision, *below*
  the 80% bar `tools/prtreid_probe.py` pre-committed for graduating relink to per-player facts.
  GS-HOTA improves anyway because the association gain outweighs the wrong merges, but any
  per-player fact built on these merges must use tau 0.04 (80.1%) instead, at 23.53 GS-HOTA.

---

# Addendum (2026-07-27, same day): carrier-attribution factorisation with the connector applied

Scheduled by the orchestrator after section 7 flagged it. GPU: per-detection PRTreID embeddings for
the three **labelled** matches (`manutd_liverpool`, `manutd_tottenham`, `manutd_brighton`), one match
at a time, 33 chunks, ~4.7 min/chunk. Code: `tools/gta_carrier.py`. Artifacts:
`results/gta_carrier.json`, `results/gta_carrier_loto.json`.

**Result: the naming factor does not move. The 20-23-moment label set shows +0.14 (p = 0.35), which
is one extra correct answer on a smaller denominator; the 2,232-sample paired proxy shows the
appearance matcher getting *worse*, monotonically with merge aggressiveness.**

## A1. What was done

The connector (splitter OFF, within-chunk, frozen from the GSR split) merges tracklets; each merge
group inherits its members' OCR name by the unanimous rule of
`eval.gsr_prtreid_relink.propagation_fill` (disagreeing groups vetoed). The enlarged named set
rebuilds the per-player gallery. Nothing was tuned: the operating point is the v1 frozen threshold
(`min_sim` 0.88, `min_margin` 0.01) and the config is the all-constraints-OFF v1 reproduction, which
is the configuration the on-record 0.846 / 0.719 / 0.609 was measured at — verified by exact
reproduction before any arm was run.

Three controls, because each is a way this could have produced a fake win:

1. **Merged leave-one-track-out.** The gallery and events tables carry **merged** track ids, so a
   query's whole merge group is excluded. Without this a sibling fragment of the query's own player
   re-enters its gallery and naming rises for free.
2. **Gallery budget scaled by group size** (`6 x number of original tracklets in the group`). At a
   flat 6-per-track the gallery *shrinks* when tracks merge (fulham: 902 -> 626 crops), which would
   confound the comparison with a sampling artifact.
3. **A resample control arm, `tau = 0.000`** — zero merges, gallery redrawn from the same stride-2
   cached frame grid the tau arms use. This separates "the connector changed something" from "the
   crops came from a different frame grid".

## A2. Factorisation (90 hand labels, 78 judged moments, all 3 matches)

| arm | gate-hit | team/candidate | naming | ident | end-to-end | assigned |
|---|---|---|---|---|---|---|
| baseline (on record) | 66/78 = **0.846** | 23/32 = **0.719** | 14/23 = **0.609** | 0.438 | **0.350** | 40 |
| tau 0.000 (resample control) | 66/78 = 0.846 | 20/30 = 0.667 | 12/20 = 0.600 | 0.400 | 0.324 | 37 |
| **tau 0.040** (>= 80% merge-precision bar) | 66/78 = 0.846 | 20/26 = 0.769 | 15/20 = **0.750** | 0.577 | 0.484 | **31** |
| tau 0.050 (below bar, info only) | 66/78 = 0.846 | 20/29 = 0.690 | 14/20 = 0.700 | 0.483 | 0.400 | 35 |
| tau 0.060 (below bar, info only) | 66/78 = 0.846 | 23/31 = 0.742 | 15/23 = 0.652 | 0.484 | 0.441 | 34 |

Gate-hit is **identical (0.846) in every arm** — as it must be: the connector re-partitions identity,
it does not change which box the carrier gate picks. Stage 1 therefore does *not* touch the
gate-hit factor, contrary to the expectation in `docs/ATTRIBUTION_RESEARCH_PLAN.md`.

Per match (naming, n): baseline liverpool 5/9, tottenham 3/4, brighton 6/10; tau 0.040 liverpool 5/7,
tottenham 4/5, brighton 6/8; tau 0.050 liverpool 6/8, tottenham **2/5**, brighton 6/7.

### Why 0.750 is not a result

- **It is one answer.** 14/23 -> 15/20. Wilson 95% CI 0.609 [0.408-0.778] vs 0.750 [0.531-0.888]:
  near-total overlap. Fisher exact **p = 0.35 vs baseline, p = 0.50 vs the resample control**.
- **Coverage falls 22.5%**: 40 -> 31 assigned moments. Part of the precision rise is simply
  answering less often.
- **No dose-response**: 0.750 (tau .04) > 0.700 (.05) > 0.652 (.06), and per match tottenham goes
  3/4 -> 2/5 at tau 0.050. That is the shape of noise, not of an effect.
- The label set has 23 usable moments for this factor. It cannot resolve anything smaller than
  ~0.25, which `results/CARRIER_CONSTRAINED_v2.md` already said in its own words.

## A3. The high-power proxy: gallery leave-one-group-out top-1

Same appearance decision, ~3,700 samples instead of 23: each gallery crop is scored against the rest
of its team's gallery with its own merge group removed, and counted correct when the nearest player
is its own. Queries are restricted to crops whose **source** tracklet carried a direct OCR read, so
pseudo-truth is never a label the merge itself invented.

| arm | LOTO top-1 | 95% Wilson | vs baseline | vs resample control |
|---|---|---|---|---|
| baseline | 2421/3877 = 0.6245 | 0.609-0.640 | — | — |
| tau 0.000 (resample control) | 2226/3528 = 0.6310 | 0.615-0.647 | p = 0.55 | — |
| tau 0.040 | 2273/3671 = 0.6192 | 0.603-0.635 | p = 0.65 | p = 0.31 |
| tau 0.050 | 2273/3752 = 0.6058 | 0.590-0.621 | p = 0.095 | p = 0.028 |
| tau 0.060 | 2270/3726 = 0.6092 | 0.594-0.625 | p = 0.18 | p = 0.059 |

Paired on the **identical** query crops (McNemar, exact binomial on discordant pairs), against the
resample control so the frame grid is held constant:

| arm | paired n | control | arm | control-only-right | arm-only-right | McNemar p |
|---|---|---|---|---|---|---|
| tau 0.040 | 2232 | 0.6366 | 0.6232 | 207 | 177 | 0.139 |
| tau 0.050 | 2074 | 0.6345 | 0.6085 | 232 | 178 | **0.0088** |
| tau 0.060 | 1956 | 0.6503 | 0.6201 | 219 | 160 | **0.0028** |

A clean monotone dose-response, **downward**: the more the connector merges, the worse the gallery
identifies a known player. At tau 0.040 (the only arm meeting the 80% merge-precision bar) the
decline is not significant (p = 0.14); at the below-bar taus it is.

**Mechanism.** The gallery scores a candidate by the **maximum** similarity over that player's crops.
At 70-80% merge precision a merged group pools crops of more than one player, so every wrong crop
becomes a permanent high-similarity distractor for whichever query it happens to resemble. Max-over-
crops is the scoring rule most exposed to exactly this contamination, and merge precision falls
across the arms in the same order the proxy does (80.1% -> 75.0% -> 69.7%).

## A4. Verdict

- **The 0.609 naming factor does not move.** Best point estimate 0.750 at tau 0.040, statistically
  indistinguishable from 0.609 (p = 0.35) and accompanied by a 22.5% coverage loss; the well-powered
  proxy says the matcher is flat-to-worse.
- **Gate-hit (0.846) is untouched and cannot be touched by this stage** — identical in all five arms.
- The composite therefore stays where `results/CARRIER_CONSTRAINED_v2.md` left it. Stage 1 buys
  tracking association (GS-HOTA 22.85 -> 24.18, fragments per GT player 5.87 -> 4.72, real-match
  tracklets per named-player-half 6.739 -> 5.043) and buys **nothing measurable** for naming.
- Consequence for Stage 2: more gallery crops per player is not the lever. The ceiling arithmetic in
  `docs/ATTRIBUTION_RESEARCH_PLAN.md` assumed naming could reach 0.85 from gallery/appearance work;
  this says appearance is saturated at ~0.62 LOTO top-1 and that the joint-inference constraints
  (mutex, roster, temporal) must do the work instead — and that any Stage-2 solver consuming merged
  tracklets should use tau 0.040, not the GS-HOTA-optimal 0.060.
- **Power caveat, stated plainly:** with 23 usable moments no intervention of realistic size can be
  validated on this label set. Either the label set grows (the cheapest fix) or every future
  attribution claim is reported on the LOTO proxy with the label set as a sanity check only.
