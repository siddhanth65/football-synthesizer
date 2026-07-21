# Adversarial review: PRTreID -> GS-HOTA arc (commit 96a59a2)

Independent recomputation from stored artifacts. CPU-only, read-only except this file. No commits.
Reviewer goal was to BREAK the claims; where a claim survived recomputation it is marked CONFIRMED
with the recomputed number and the exact evidence path.

## Verdict table

| # | Claim | Verdict | Recomputed vs claimed |
|---|---|---|---|
| 1 | OSNet-ImageNet arm reproduces the published Stage-2c ladder (0.872/0.832, sep +0.0401, 35.0%) | CONFIRMED | same 0.8720 / diff 0.8319, sep 0.04012, mp@0.80 = 70/200 = 35.0% -- matches 2026-07-16 ladder |
| 2 | Held-out 84.7%@0.965 (658/777), 81.7%@0.960 (988); thresholds pre-committed on pilot | CONFIRMED (with nuance) | 658/777 = 84.68%, 807/988 = 81.68% exact. 0.965 IS the smallest pilot threshold >=80% (80.9%). 0.960 is 72.5% on the pilot -- pilot-borderline, clears 80% only on held-out/full split |
| 3 | 19.83 -> 20.65 -> 22.85; jersey-off invariance; DetA 9.89/9.89/11.12 | CONFIRMED | independently re-scored the stored submissions: 19.83 / 20.65 / 22.85; DetA 9.89 / 9.89 / 11.12; no_jersey/role_only/loc_assoc identical (46.54/48.52/52.48) between relink and propagate |
| 4 | Propagation rule = fill-abstainers-only, never overwrite, drop group on any disagreement, no confidence knob, no GT leakage | CONFIRMED | code review of `propagation_fill`/`remap_submission`/`build_remap`: exactly that rule, zero tunables beyond inherited threshold, GT used only in reporting audits |
| 5 | Propagation precision 133/33/0/5 = 80.1%, zero fills onto GT-unnumbered | CONFIRMED | stored audit 133/33/0/5, 133/(133+33) = 80.12%; self-consistent and matches the parameter-free rule verified in #4 |
| 6 | n_seqs_hurt: relink 15/58, propagation 1/58 vs jersey-only, 0/58 vs relink-only | CONFIRMED | relink@0.960 = 15 hurt (worst SNGS-082 -1.76...); propagate = 1 vs jersey-only (SNGS-042 -0.017), 0 vs relink-only |
| 7 | Side-effects: default embedder still osnet, track_relink additive, shipped artifacts untouched | CONFIRMED (cleaner than claimed) | `wire_anchors` default `embedder_name="osnet"`; changes additive. `generator/track_relink.py` NOT in the commit at all -- arc reuses pre-existing merge fns. gsr_scores.json / gsr_scores_koshkina.json / eval_koshkina untouched |
| B | Brighton end-to-end: 109->153, 165->121, 53->69 | CONFIRMED | report tables + parquet: 153 attached (+40.4%), 121 ambiguous (-26.7%), 69 named frags = 69 parquet rows / 7 players |

**OVERALL: SAFE TO PRESENT.** Every headline composite in the 14.76 -> 19.83 -> 20.65 -> 22.85 chain
independently reproduced from the stored submissions with the official evaluator; the merge and
propagation rules are parameter-free and leak no ground truth into decisions; side-effects are clean
and in fact more conservative than the commit message implies (track_relink.py was not modified). One
honesty caveat below (Claim 2) should be scoped in any writeup; it does not move the numbers.

## Claim 1 -- protocol validity (CONFIRMED)

`results/gsr_benchmark/prtreid_probe_osnet_imagenet.json` pooled: same_median 0.8720, diff_median
0.8319, separation 0.04012; merge precision @0.80 = 70/200 = 35.0%. This reproduces the 2026-07-16
Stage-2a/2c ladder recorded in `git show f2f29ac:STATUS.md` ("true merge precision on pilot GT = 35%",
"ImageNet OSNet ... median cosine 0.81 ... separation ~0.04"). The load-bearing figures (sep ~0.04,
35% merge precision) reproduce exactly under the new harness, so the PRTreID vs OSNet comparison is
apples-to-apples. (The exact medians 0.872/0.832 were not individually tabulated in the old STATUS,
but the two decision figures that defined the wall were, and both reproduce.)

## Claim 2 -- headline precision + pre-commitment (CONFIRMED, with a scoping nuance)

`results/gsr_benchmark/prtreid_heldout_soccernet.json` (55 non-pilot seqs): 0.965 = 658/777 = 84.68%,
0.960 = 807/988 = 81.68%. Both exact to the claim, including the 988 merge count.

Pre-commitment test against the pilot sweep `prtreid_probe_soccernet_sweep.json`
(pilot = SNGS-021/022/023):

```
0.950 68/106 = 64.2%
0.960 58/80  = 72.5%
0.965 55/68  = 80.9%   <- smallest sweep threshold reaching >=80%
0.970 49/56  = 87.5%
```

So **0.965 is genuinely derivable from the pilot alone** as "the smallest threshold reaching >=80%"
-- the primary threshold is pre-committed as claimed, and the sweep is a clean monotonic precision
ladder with no sign of composite-driven selection.

The nuance to state honestly: the **recommended/headline threshold is 0.960**, and 0.960 is only
72.5% on the 3-sequence pilot -- below the 80% bar. It clears 80% only on the held-out split (81.7%)
and the full split (81.0%). So "pre-committed on the pilot" is strictly true for 0.965 but not for
0.960; 0.960 is precision-validated on held-out data (which is the correct validation set), just not
pilot-derivable. Because the merges actually used at 0.960 are audited at ~81% precision on both the
held-out and full splits, this does not undermine the composite -- but any writeup should scope the
"pre-committed from the pilot" phrasing to 0.965.

## Claim 3 -- GS-HOTA arms (CONFIRMED by independent re-score)

Re-scored the stored submissions under `outputs/gsr/{eval_koshkina, eval_prtreid_relink_0.960,
eval_prtreid_relink_0.960_propagate}` with `eval.gsr_score.gs_hota` + `EVAL_CONFIGS` (official
SoccerNet-GSR evaluator), independent of the stored `gsr_scores_*.json`:

| arm | GS-HOTA | GS-DetA | GS-AssA | no_jersey | role_only | loc_assoc |
|---|---|---|---|---|---|---|
| jersey-only (control) | 19.83 | 9.89 | 39.77 | 43.06 | 44.95 | 48.91 |
| relink @0.960 | 20.65 | 9.89 | 43.09 | 46.54 | 48.52 | 52.48 |
| relink + propagation | 22.85 | 11.12 | 46.96 | 46.54 | 48.52 | 52.48 |

All three composites and DetA (9.89/9.89/11.12) reproduce the commit exactly. The jersey-off configs
(no_jersey/role_only/loc_assoc) are **bit-identical between the relink and propagation arms**, the
required invariance (propagation writes only `attributes.jersey`, which those configs ignore). The
20.56@0.965 sub-arm was verified from `gsr_scores_prtreid_relink.json` only (not independently
re-scored) but is not on the headline path (0.960 is).

## Claim 4 -- propagation rule + no leakage (CONFIRMED)

`eval/gsr_prtreid_relink.py`:
- `build_remap`: `greedy_merge(fragments, sims, threshold, ...)` decides merges from PRTreID
  embeddings + the motion gate only. `gt_ids`/`merge_precision` are computed AFTER the remap and used
  purely for reporting -- **no GT enters a merge decision.**
- `propagation_fill` (lines 103-157): groups by `new_track_id`; skips singletons (`len < 2`); fills
  only members with `read[t][0] < 1` (abstainers); if `len(nums) > 1` the whole group is skipped
  (drop-on-any-disagreement). Confidence `v[1]` is loaded but **never gates anything** -- there is no
  confidence-margin knob. The only parameter is the inherited relink threshold.
- `remap_submission` guards the fill again: applies only where `role == "player"` and `jersey is
  None`, so a real read is never overwritten.
- `audit_propagation`/`gt_jersey_by_track` touch GT only to score fills for the report.

The `_demo()` self-check (`python -m eval.gsr_prtreid_relink --demo`) asserts exactly this behaviour
(agreeing group fills the abstainer; disagreeing group fills nobody; existing read untouched).

## Claim 5 -- propagation precision (CONFIRMED)

`gsr_scores_prtreid_propagate.json` -> `arms.0.960.propagation.audit` = correct 133, wrong 33,
gt_unnumbered 0, unauditable 5; precision 133/(133+33) = 133/166 = 80.12%. The zero in the
GT-unnumbered column is present and load-bearing as claimed (no fill converted a null==null GT match
into a miss). Not re-derived from embeddings (that needs the warm PRTreID cache), but the rule that
produces it was verified line-by-line in Claim 4 and the downstream composite (22.85) reproduced
independently, so the audit is trustworthy.

## Claim 6 -- n_seqs_hurt (CONFIRMED)

From `gsr_scores_prtreid_relink.json` / `..._propagate.json`:
- relink @0.960 vs jersey-only: **15 hurt**, 38 helped; worst SNGS-082 -1.76, SNGS-085 -0.81,
  SNGS-080 -0.47, SNGS-078 -0.27, SNGS-045 -0.21 (matches the report's worst list exactly).
- propagation vs jersey-only: **1 hurt** (SNGS-042 -0.017), 54 helped.
- propagation vs relink-only: **0 hurt**, 43 helped; biggest repairs SNGS-082 +13.88, SNGS-085 +9.68,
  SNGS-080 +7.96 -- i.e. propagation repairs precisely the sequences relink damaged.

## Claim 7 -- side-effects (CONFIRMED, cleaner than stated)

`git show --stat 96a59a2` touches only: STATUS.md, docs/REVIEW_CRIB.md, eval/gsr_prtreid_relink.py
(new), tools/prtreid_probe.py (new), tools/wire_anchors.py (modified), and results artifacts (all new
files). Findings:
- `tools/wire_anchors.py`: default `embedder_name="osnet"`; the diff adds a `build_embedder` helper
  and optional `min_sim`/`min_margin` params defaulting to the existing `aw.REID_MIN_SIM/MARGIN`, so
  the default code path and every shipped OSNet artifact are unchanged. Additive.
- `generator/track_relink.py`: **not in the commit** -- the arc imports the pre-existing merge
  functions unchanged. This is more conservative than the task's premise ("track_relink changes are
  additive"); there are no changes to review.
- Shipped scoring artifacts `gsr_scores.json`, `gsr_scores_koshkina.json`,
  `GSR_RESCORE_KOSHKINA.md`, and the `eval_koshkina` submissions are untouched; new arms write to
  separate dirs and separate results files.

## Brighton end-to-end (CONFIRMED)

Baseline is `results/identity/NAMED_TRACKS_both2_ain.md` (OSNet-AIN): 109 attached, 165 ambiguous,
53 named. `results/identity/NAMED_TRACKS_both2_prtreid.md` (PRTreID, embedder swapped only): 153
attached (+40.4%), 121 ambiguous (-26.7%), 69 named. `outputs/identity/
brighton_manutd_named_tracks_both2_prtreid.parquet` has 69 rows / 7 distinct `player_name` (the 6->7
weak secondary). The report itself flags the Spearman minutes proxy at -0.213 (did not improve) and
the hero-shot concentration (#8 = 82% of anchors) -- not buried.

## Residual honesty notes (already stated by the arc, restated for Sid)

1. The headline 22.85 still sits on GS-DetA 11.12 vs 60.10 unattributed; abstention is still 87.8%.
   The lift is real and validated but small in absolute terms -- +2.20 GS-HOTA from propagation, on
   an external tracking benchmark, not a ManU-facing product number.
2. Scope 0.960's "pre-committed" language to the held-out precision (it is), not the pilot (it is
   not >=80% there). 0.965 is the strictly pilot-derivable threshold.

## Evidence paths

- Scoring code: `eval/gsr_prtreid_relink.py`, `eval/gsr_score.py`
- Merge/propagation import source: `generator/track_relink.py` (unchanged this arc)
- Composites: `results/gsr_benchmark/gsr_scores_prtreid_relink.json`,
  `results/gsr_benchmark/gsr_scores_prtreid_propagate.json`,
  `results/gsr_benchmark/gsr_scores_koshkina.json`
- Precision: `results/gsr_benchmark/prtreid_heldout_soccernet.json`,
  `results/gsr_benchmark/prtreid_probe_soccernet_sweep.json`,
  `results/gsr_benchmark/prtreid_probe_osnet_imagenet.json`
- Submissions re-scored: `outputs/gsr/eval_koshkina`, `outputs/gsr/eval_prtreid_relink_0.960`,
  `outputs/gsr/eval_prtreid_relink_0.960_propagate`
- Brighton: `results/identity/NAMED_TRACKS_both2_ain.md`,
  `results/identity/NAMED_TRACKS_both2_prtreid.md`,
  `outputs/identity/brighton_manutd_named_tracks_both2_prtreid.parquet`
- Prior ladder: `git show f2f29ac:STATUS.md`
