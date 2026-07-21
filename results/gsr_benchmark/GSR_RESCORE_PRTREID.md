# GS-HOTA re-score: jersey layer + PRTreID track re-link (SoccerNet-GSR valid, 58 sequences)

Artifact: `results/gsr_benchmark/gsr_scores_prtreid_relink.json`
Code: `eval/gsr_prtreid_relink.py` (scoring), `generator/track_relink.py` (merge logic),
`tools/prtreid_probe.py` (embeddings + held-out precision)
Shipped arms untouched: `gsr_scores.json`, `gsr_scores_koshkina.json`, `GSR_RESCORE_KOSHKINA.md`.

## What changed vs the 19.83 run

Exactly one thing: the identity partition. The relinked submissions are the **same
`eval_koshkina` submission files** with only `track_id` (and the derived `id`) rewritten by the
PRTreID merge map. Pitch position, role, team, confidence and the attached jersey string are copied
byte-for-byte, so detection and jersey reading are held fixed and the only scored variable is
association.

Thresholds **0.965 / 0.960** were pre-committed from the held-out merge-precision measurement
(`prtreid_heldout_soccernet.json`: 84.7% @0.965, 81.7% @0.960 over 55 non-pilot sequences) before
any GS-HOTA number was computed. They were not re-tuned against the composite.

Embedder: `prtreid-soccernet-baseline.pth.tar`, `globl` embedding, 10 crops/fragment, median
pooling. Motion gate unchanged (9.0 m/s, 2.0 m slack, temporally disjoint, same role, same team).

## Three-arm table (GS-HOTA per attribute config, 58 sequences)

| arm | gs_hota_full | no_jersey | role_only | loc_assoc | LocA (full) | IDF1 (full) | AssA (loc_assoc) |
|---|---|---|---|---|---|---|---|
| pre-jersey baseline | 14.76 | 43.06 | 44.95 | 48.91 | 91.35 | 8.07 | 39.79 |
| jersey-only (shipped, 19.83) | **19.83** | 43.06 | 44.95 | 48.91 | 91.85 | 14.73 | 39.79 |
| jersey + PRTreID relink @0.965 | **20.56** | 46.06 | 47.99 | 51.96 | 91.85 | 15.57 | 44.94 |
| jersey + PRTreID relink @0.960 | **20.65** | 46.54 | 48.52 | 52.48 | 91.85 | 15.68 | 45.86 |

Full decomposition of the headline `gs_hota_full` config:

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| jersey-only (19.83) | 19.83 | 9.89 | 39.77 | 91.85 | 14.73 |
| relink @0.965 | 20.56 | 9.90 | 42.71 | 91.85 | 15.57 |
| relink @0.960 | 20.65 | 9.89 | 43.09 | 91.85 | 15.68 |

`loc_assoc` (localization + association only, no attribute gating):

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| jersey-only | 48.91 | 60.18 | 39.79 | 92.53 | 51.89 |
| relink @0.965 | 51.96 | 60.12 | 44.94 | 92.52 | 57.06 |
| relink @0.960 | 52.48 | 60.10 | 45.86 | 92.52 | 57.99 |

DetA and LocA are flat to two decimals in every arm, as they must be: relink touches only ids.

## Merge behaviour and precision (GT-audited on all 58 sequences)

| threshold | fragments/seq | merges | merge precision (pairs) |
|---|---|---|---|
| 0.965 | 79.3 -> 67.1 | 709 | 713/845 = **84.4%** |
| 0.960 | 79.3 -> 64.4 | 868 | 865/1068 = **81.0%** |

The full-split precision reproduces the held-out probe numbers (84.7% / 81.7%) to within a point,
i.e. the pilot-free precision estimate held.

## n_seqs_hurt -- the number that must be read loudly

The 19.83 jersey run hurt **0/58** sequences. Relink is **not** free in that sense:

| threshold | helped | hurt | unchanged | mean per-seq delta | total gain | total loss |
|---|---|---|---|---|---|---|
| 0.965 | 35 | **17** | 6 | +0.60 | +39.34 | -4.33 |
| 0.960 | 38 | **15** | 5 | +0.68 | +44.12 | -4.42 |

Worst sequences @0.960 (GS-HOTA delta, merge precision, merges):
`SNGS-082 -1.76 (22/29, 22)`, `SNGS-085 -0.81 (22/23, 17)`, `SNGS-080 -0.47 (16/23, 18)`,
`SNGS-078 -0.27 (13/16, 14)`, `SNGS-045 -0.21 (8/11, 8)`.
Best: `SNGS-021 +6.57`, `SNGS-096 +5.52`, `SNGS-083 +3.41`, `SNGS-081 +2.90`, `SNGS-032 +2.67`.

Damage is bounded and asymmetric: the loss mass is ~10x smaller than the gain mass, and no sequence
loses more than 1.8 GS-HOTA points while five gain 2.5-6.6. But hurt is **not** explained by bad
merges alone -- merge precision on hurt sequences (79.2% @0.960) is barely below helped sequences
(81.8%), and `SNGS-085` lost 0.81 with 22/23 = 96% correct merges. A *correct* merge can still cost
GS-HOTA when it stitches a fragment whose jersey read disagrees with the receiving track's, or when
it consolidates ids in a way that shifts the optimal GT-to-prediction assignment. This is a real
cost, not measurement noise, and it is the honest counterweight to the headline.

## Honest verdict: was association the binding constraint?

**Partly -- and much less than the arm-level lift suggests.**

- Relink does what its precision promised. Association improves substantially where it is measured
  cleanly: `loc_assoc` GS-AssA **39.79 -> 45.86 (+6.07, +15.3% relative)**, `loc_assoc` GS-HOTA
  **48.91 -> 52.48 (+3.57)**, `no_jersey` GS-HOTA **43.06 -> 46.54 (+3.48)**. This is the first time
  relink has been allowed on the scored path and it is a clear, validated association win. The
  earlier decision to exclude OSNet relink was correct and is now correctly reversed for PRTreID.
- The headline composite barely moves: **19.83 -> 20.65 (+0.82, +4.1% relative)** at 0.960,
  **+0.73 (+3.7%)** at 0.965. Compare the jersey layer itself: +5.08 (+34%).
- Reason: under `gs_hota_full`, GS-DetA is 9.89 -- jersey gating destroys ~84% of the detection
  score (60.18 -> 9.89) because 91% of tracks abstain and abstention cannot match a GT player who
  carries a labelled number. GS-HOTA is `sqrt(DetA * AssA)`, so with DetA pinned near 10 a +3.3-point
  AssA gain buys under a point of composite. **Jersey coverage, not association, is the binding
  constraint on the headline metric.** Fixing association on top of a 9.9 DetA is pushing on the
  wrong term.

So: association was a genuine, previously untapped +15% on its own axis, and it was *not* the thing
holding the headline number down. The next real headline movement has to come from jersey recall
(91.3% track abstention) or from detection under the jersey gate, not from better ReID.

Recommended arm to quote: **0.960** -- it beats 0.965 on every config, hurts two fewer sequences, and
was pre-committed as the higher-yield secondary. Neither threshold was selected using these numbers.

## Stage 2b: jersey propagation across merge groups (attacking the DetA term)

Artifact: `results/gsr_benchmark/gsr_scores_prtreid_propagate.json` (new file; the relink and
koshkina artifacts are untouched). Code: `eval/gsr_prtreid_relink.py --propagate-jersey`.

The verdict above said the binding constraint is jersey RECALL, not association. This stage attacks
it with **no new reads**: a validated merge group is one player at 81% pair precision, so a number
read on one fragment can be carried to fragments of the same group that abstained.

**Rule (pre-committed before any GS-HOTA was computed, zero free parameters):**

1. Propagate only inside a merge group produced by the 0.960 relink; singletons are untouched.
2. Fill only members that ABSTAINED. A member that already read a number keeps it verbatim -- the
   stage is strictly additive and never overwrites or deletes a read.
3. On ANY disagreement between the group's numbered members, the whole group is left untouched.
   Chosen over a confidence-margin winner because it has no threshold to tune against the composite,
   and because an internally contradictory group is exactly the backfire case. Where the group
   agrees, the confidence-weighted vote is trivially that same number.

Everything else -- threshold 0.960, embedder, motion gate, crops, the underlying Koshkina votes -- is
byte-identical to the 20.65 arm, so propagation is the only new variable.

### Three-arm table (58 sequences)

| arm | gs_hota_full | no_jersey | role_only | loc_assoc | LocA (full) | IDF1 (full) | AssA (loc_assoc) |
|---|---|---|---|---|---|---|---|
| jersey-only (shipped, 19.83) | 19.83 | 43.06 | 44.95 | 48.91 | 91.85 | 14.73 | 39.79 |
| jersey + relink @0.960 | 20.65 | 46.54 | 48.52 | 52.48 | 91.85 | 15.68 | 45.86 |
| relink + propagation @0.960 | **22.85** | 46.54 | 48.52 | 52.48 | 91.94 | 17.74 | 45.86 |

The three jersey-off configs are **identical to the last decimal** to the relink arm -- the required
invariant, since propagation writes only `attributes.jersey`.

Full decomposition of the headline config -- **DetA is the mechanism under test**:

| arm | GS-HOTA | GS-DetA | GS-AssA | GS-LocA | IDF1 |
|---|---|---|---|---|---|
| jersey-only (19.83) | 19.83 | 9.89 | 39.77 | 91.85 | 14.73 |
| relink @0.960 | 20.65 | 9.89 | 43.09 | 91.85 | 15.68 |
| relink + propagation | **22.85** | **11.12** | 46.96 | 91.94 | 17.74 |

**GS-DetA 9.89 -> 11.12 (+1.23, +12.4% relative)** -- the first time any change has moved DetA under
the jersey gate. AssA also rises 43.09 -> 46.96 (+3.87), because numbered rows change the optimal
GT-to-prediction assignment as well as the match count. Composite **+2.20 (+10.7%)**, versus +0.82
for the relink itself.

### Coverage and propagation precision (GT-audited, all 58 sequences)

| quantity | before | after |
|---|---|---|
| tracks carrying a number | 425 / 4870 | 596 / 4870 |
| track abstention rate | **91.27%** | **87.76%** |
| player rows with a number | 49067 / 481038 (10.2%) | 66162 / 481038 (13.8%) |

Merge groups: 649 multi-fragment groups, 144 contain at least one numbered member, of those **140
agreed and 4 disagreed** (the disagreeing 4 were dropped whole by rule 3). 171 fragments filled.

Propagation precision against GT (dominant GT track of each filled fragment):

| correct | wrong | onto a GT-unnumbered player | unauditable |
|---|---|---|---|
| **133 (80.1%)** | 33 | **0** | 5 |

80.1% matches the 81.0% pair-level merge precision at this threshold, i.e. propagation inherits the
merge's error rate and adds essentially none of its own -- as expected, since the rule adds no
decision beyond "same group".

The zero in the third column is the load-bearing number. 21% of GT player/GK tracks (256/1222) carry
no jersey label, and the earlier jersey work showed that numbering such a player converts a
`null == null` match into a miss. Not one propagated number landed on an unnumbered GT track, so
every one of the 33 wrong fills went onto a player who already carried a *different* number -- a row
that was already a non-match when it was null. That is why the wrong fills cost almost nothing while
the 133 correct ones pay.

### n_seqs_hurt -- against both prior arms

| comparison | helped | hurt | unchanged | mean per-seq delta |
|---|---|---|---|---|
| propagation vs relink-only @0.960 | 43 | **0** | 15 | +2.46 |
| propagation vs jersey-only (19.83) | 54 | **1** | 3 | +3.14 |

Zero sequences lose ground to the relink arm; the 15 unchanged are the sequences where no group had
both a read and an abstainer (13 of them have zero fills). Against the jersey-only arm the hurt count
drops from **15 to 1** (`SNGS-042`, -0.017), because the sequences the relink damaged are exactly the
ones propagation repairs: `SNGS-082 +13.88`, `SNGS-085 +9.68`, `SNGS-080 +7.96` vs the relink arm --
the same three sequences that headed the relink's worst-hurt list. A merge that stitched a numbered
fragment onto an abstaining one was previously a pure identity change with a jersey inconsistency
inside it; propagation resolves that inconsistency.

### Verdict

**Yes -- attacking recall moves the composite, roughly 3x harder than attacking association.**
Same relink, same reads, one extra rule: +2.20 GS-HOTA against +0.82 for the relink alone, and it is
the first intervention to lift GS-DetA (9.89 -> 11.12) at all. The diagnosis was right and the
predicted mechanism is exactly what the decomposition shows.

The honest ceiling: DetA is still 11.12 against 60.10 unattributed, and abstention is still 87.8%.
Propagation can only reach abstainers that share a group with a read -- 144 of 649 multi-fragment
groups -- so the remaining ~85% of the DetA gap needs actual reads (better legibility/pose recall or
more crops per track), not further redistribution of the 425 numbers we already have.

## Reproduce

```
python -m eval.gsr_prtreid_relink --threshold 0.965
python -m eval.gsr_prtreid_relink --threshold 0.960
python -m eval.gsr_prtreid_relink --threshold 0.960 --propagate-jersey
python -m eval.gsr_prtreid_relink --demo        # rule self-check
```

CPU-only (PRTreID embeddings cached under `outputs/gsr/relink_cache_prtreid/`), ~4 min per arm,
resumable by disk state (per-sequence submissions + `relink_stats.json`).
