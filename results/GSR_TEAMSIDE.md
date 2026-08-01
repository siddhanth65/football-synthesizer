# N1b team-side resolver — the second family, and the ceiling that closes it

Campaign act N1b. `results/GSR_DELEAK.md` §3 left the cluster -> `left`/`right` permutation as the
binding constraint on a legitimate SoccerNet-GSR number: the positional resolver is 45/49 on the
official test split and every miss annihilates its clip (-32.94 GS-HOTA). The brief named three
unexplored families and a target of >=0.98. All three were graded. None beat the incumbent, and the
reason is not that we searched badly.

**Headline: the rule is already at its own ceiling. Run "the deeper cluster is left" on the
GROUND-TRUTH positions — perfect detection, perfect clustering, perfect calibration — and it scores
112/115 on the development split and 45/49 on test, the same 45/49 our estimator gets. The residual
failures are not estimation errors; they are clips whose ground-truth geometry inverts. The target
of 0.98 is above the oracle ceiling of the entire positional family, so the +2.4 GS-HOTA prize the
brief costed does not exist inside it. The incumbent is kept unchanged; the test split confirms
45/49 = 0.9184 and no package was rebuilt.**

The one rule that does clear 0.98 is measured too: **`gk_self` — "a keeper's own team defends the
goal he stands in" — is 113/113 on ground truth**, including all three inversions. It is not
shippable, because it needs the keeper's *team*, and the two-way kit clustering puts a detected
keeper in his own team's cluster in **24.4% of matched rows (13 of 57 sequences)**. That is the
named upgrade path, and it is an appearance problem, not a geometry one.

---

## 1. Protocol

| | |
|---|---|
| dev pool | valid-58 + the non-degenerate train sequences extracted for this act (**67** at the graded run, 13 landed by write-up; extraction of the remaining 26 was still running) |
| dev stress pool | the same clips cut into 3 windows of ~250 frames — **204 rows, 201 gradable, 34 failures**, where the whole-clip pool has only 4 |
| oracle pool | **all 115** dev sequences on GT positions (needs no extraction, so the 39 train clips are covered in full) |
| frozen before test | `results/gsr_teamside_frozen.json`, sha256 `93776eb32b7d8e79…` |
| test reads | **one**, `--verify`, side-correctness of the frozen rule only |

The 18 degenerate train sequences (SNGS-060…077, `GSR_DELEAK.md` §3) are excluded from the *our
positions* pools; they are kept in the GT oracle pool, where kit clustering plays no part.

**Disclosure.** An early GT-oracle scan was run over all 164 sequences at once, so the fact that
three of the four test failures are ground-truth geometry inversions was seen before the freeze. It
set no threshold and selected no candidate — every selection below is computed on train+valid only —
but it was seen, and a reviewer should know it.

## 2. Family 1 — solve both permutations and vote: zero bits, provably

The brief's primary candidate was to run the frozen identity solve under both cluster -> side maps
and keep the permutation with the better internal evidence fit (total objective, summed posteriors,
OCR/roster consistency, coverage-at-abstention).

The solver never reads the side. `Identity.key` carries `(side, number)` as a *handle*; the
objective is built from `Identity.team` (a cluster index), `Identity.number` and `Identity.role`,
and the leave-one-out gallery tensor is anchored the same way. The self-roster is built from
`(cluster of the reading tracklet, number)` and then *named* through the side map, so the slot
structure is identical under both permutations as well.

Measured on six valid sequences under both maps:

| | result |
|---|---|
| identical assignment | 6/6 |
| max abs posterior delta | **0.0** (exactly) |
| identical summed posterior | 6/6 (e.g. SNGS-021 2.9516157708280364 both ways) |
| identical roster slots | 6/6 |

Every criterion the brief lists is a function of that objective, so all of them are invariant. The
only way to break the invariance is a roster whose sides come from outside the clip — a lineup sheet
— which is exactly the label read N1 removed. **Family 1 is a coin flip, 0.5000.**

## 3. Family 2 — the rules, on our positions and on ground truth

Signed-score convention: negative means "cluster 0 defends the left goal". A rule is graded only
where its evidence exists, so `n` differs by rule.

| rule | what it reads | dev (67 clips) | dev stress (68 clips, 204 windows) | **GT oracle (115)** |
|---|---|---|---|---|
| **`meanx`** (incumbent) | mean pitch-x per cluster | **63/67 = 0.9403** | 167/201 = 0.8308 | **112/115 = 0.9739** |
| `medx` | median pitch-x | 61/67 = 0.9104 | 164/201 = 0.8159 | 111/115 = 0.9652 |
| `frame_mean` | per-frame centroid gap, averaged | 62/67 = 0.9254 | 165/200 = 0.8250 | 110/115 = 0.9565 |
| `frame_t` | the same gap as a paired t | 62/67 = 0.9254 | 165/200 = 0.8250 | 110/115 = 0.9565 |
| `cent_t` | unpaired Welch t on centroids | 62/67 = 0.9254 | 165/200 = 0.8250 | 110/115 = 0.9565 |
| `frame_vote` | sign of the per-frame gap, majority | 63/67 = 0.9403 | 163/200 = 0.8150 | 109/115 = 0.9478 |
| `poss_rel` | defending block's offset from the ball | 50/65 = 0.7692 | 99/143 = 0.6923 | 108/115 = 0.9391 |
| `deep3` | mean of each cluster's 3 deepest | 52/67 = 0.7761 | 142/201 = 0.7065 | 106/115 = 0.9217 |
| `high3` | mean of each cluster's 3 highest | 50/67 = 0.7463 | 131/201 = 0.6517 | 102/115 = 0.8870 |
| `extreme_owner` | who owns the deepest / highest player | 55/67 = 0.8209 | 143/201 = 0.7114 | 101/115 = 0.8783 |
| `deep1` | the single deepest player | 48/67 = 0.7164 | 127/201 = 0.6318 | 92/115 = 0.8000 |
| `high1` | the single highest player | 40/67 = 0.5970 | 112/201 = 0.5572 | 88/115 = 0.7652 |
| `ball_rel` | mean offset from the ball | 56/67 = 0.8358 | 137/175 = 0.7829 | 82/115 = 0.7130 |
| `gk_anchor` | proximity to a detected keeper | 37/43 = 0.8605 | 55/65 = 0.8462 | 78/113 = 0.6903 |
| `poss_drift` | ball travel while a cluster holds it | 33/65 = 0.5077 | 76/143 = 0.5315 | **60/115 = 0.5217** |
| `poss_share` | sign-vote version of the same | 31/65 = 0.4769 | 74/143 = 0.5175 | **60/115 = 0.5217** |
| `gk_self` | the keeper's own team defends his goal | 15/59 = 0.2542 | 25/105 = 0.2381 | **113/113 = 1.0000** |

Read the last column first. It is the accuracy each rule reaches when its inputs are exact, and it
is the column that decides which rules are worth estimating better.

* **`meanx` is the best rule our pipeline can compute, on ground truth as well as on our own
  positions.** Every depth variant that looked like it might be more robust — medians, per-frame
  pairing, t-statistics, defensive-line and front-line order statistics — is *worse* on GT. The
  plateau the brief describes is not an estimation plateau. It is the rule's own accuracy.
* **The flow family is at chance even on ground truth.** `poss_drift` and `poss_share` score
  60/115 = 0.5217 with perfect ball, perfect players and perfect teams. A 30-second clip does not
  contain a reliably directed attack, so "which way is this team going" is not recoverable from it
  at all. This is a stronger negative than a bad estimator: there is nothing to estimate.
* **The keeper rules split cleanly.** The *proximity* keeper rule (`gk_anchor`, which is what
  "GKs sit deepest, use role not kit cluster" reduces to) is 0.6903 on GT — it is a depth rule in
  disguise and inherits every inversion. The *definitional* keeper rule (`gk_self`) is exact.

### The three ground-truth inversions

Scores below are the signed `meanx` (metres), sign convention as in §3: **negative is correct**,
so a positive GT score is a clip where the rule is wrong with exact inputs. `gk_self` is shown
beside it because it is correct on all three.

| clip | split | GT `meanx` | GT `gk_self` | our `meanx` | our verdict |
|---|---|---|---|---|---|
| SNGS-038 | valid | **+0.95** | -49.4 | -1.51 | right, **by luck** (the rule is wrong here on GT) |
| SNGS-092 | valid | **+3.76** | -85.4 | -1.13 | wrong |
| SNGS-111 | train | **+8.81** | -95.3 | not yet extracted | — |

On test the GT inversions are SNGS-126 (+3.99), SNGS-131 (+3.78), SNGS-195 (+0.69) and SNGS-197
(+5.37): our resolver is wrong on three of them and right on SNGS-195 by luck, and its fourth miss
(SNGS-129) is an estimation error. **The oracle and our estimator score the same 45/49 on test and
the same 56/58 on valid while disagreeing about which clips.** The estimation noise is a wash.

These clips are not broken clips. In SNGS-092 the ground truth has 5,544 `left` player rows and
5,574 `right` ones, and the `right` team's mean sits 3.76 m *deeper into the left half* than the
team that defends it — a spell where the camera frames one attacking phase and the defending team's
own deep line is off-screen. GT annotations only cover visible players, so ground truth inherits the
broadcast's framing.

## 4. Family 3 — ensembles: gains that do not survive a second pool

Vote ensembles of `meanx` plus 2 or 4 other rules (1,470 combinations) and 90 "fall back to rule X
when `|meanx|` < T" rules were searched on both dev pools.

| ensemble | dev whole clips | dev stress windows | GT ceiling |
|---|---|---|---|
| `meanx` alone | 0.9403 | 0.8284 | **0.9739** |
| best on whole clips (`meanx medx deep1 deep3 high1`) | **0.9701** | 0.7941 | 0.9652 |
| best on windows (`meanx frame_vote gk_anchor`) | 0.9552 | **0.8382** | 0.9652 |
| best on valid-58 alone (`meanx medx deep3 ball_rel gk_anchor`) | 0.9552 | 0.8284 | 0.9391 |
| best of all 1,470 measured **on ground truth** | — | — | 0.9739 (a tie, `meanx medx high1`) |

(The ensemble rows score all 204 window rows, treating the 3 with no computable `meanx` as a wrong
guess, so the incumbent reads 0.8284 here against 167/201 = 0.8308 in §3; every row of the table
uses the same convention, so the comparison is like-for-like.)

112 of 1,470 ensembles beat the baseline on whole clips and 15 of 1,470 on windows; the two winners
are different ensembles, and each loses on the pool it was not chosen on. With 67 clips at p=0.94
the binomial sd is 2.9 pp — a best-of-1,470 gain of two sequences is exactly noise scale. The
decisive check is the last row: **no ensemble beats `meanx` on ground truth.** Mixing a 0.80-ceiling
rule into a 0.9739-ceiling rule cannot help except by luck. The best fallback rule ("use `deep1`
when `|meanx|` < 2 m", 0.9701 on whole clips) is the same artefact.

## 5. The margin — a real ranking signal, not a usable gate

The brief asked for a confidence that separates right from wrong, noting the N1 finding that the
metre gap does not. Refined:

| pool | AUC of `|meanx|` as a confidence | wrong-clip margins (m) | correct clips below the largest wrong margin |
|---|---|---|---|
| dev, whole clips (67, 4 wrong) | **0.948** | 0.01, 0.87, 1.13, 1.84 | 8 of 63 |
| dev, 3-window stress (201, 34 wrong) | 0.773 | — | — |
| test (49, 4 wrong) | **0.867** | 0.75, 2.48, 2.80, 3.41 | 9 of 45 |

Other candidates on the stress pool: `|frame_t|` 0.722, `|cent_t|` 0.647, calibration coverage
0.665, calibrated frame count 0.667 — all weaker than the raw gap. On whole clips calibration
coverage is *anti*-correlated with correctness (AUC 0.409).

So `|meanx|` **is** a ranking signal (AUC 0.87 on test), which softens N1's "no confidence signal
exists" — but it is not a gate, and a gate is what would be needed. Catching all four test failures
costs abstaining on 13 of 49 clips, and GSR has no abstain: a submission must emit a side. A margin
is only worth having if a better rule exists for the low-margin clips, and §3-4 say none does.

The two pools also explain N1's negative. Dev's failures are 3 estimation errors (small margins) and
1 inversion; test's are 3 inversions (healthy margins, up to 3.41 m) and 1 estimation error. **A
margin flags weak evidence, and an inversion is not weak evidence — it is strong evidence pointing
the wrong way.** Any confidence fitted on one split's failure mix mis-describes the other's.

## 6. The named upgrade path: link a keeper to a team

`gk_self` is 113/113 on GT, available in 113 of 115 dev sequences, and right on all three
inversions — because it is what the GSR `left`/`right` attribute *means*. On SNGS-092, GT keepers
sit at x = -48.0 (`left`) and +37.4 (`right`) in the label file's own centred frame, with no
ambiguity whatever, while the outfield means invert.

What blocks it, measured over 5,384 keeper detections matched to GT keepers within the 5 m
tolerance on 57 dev sequences:

| | value |
|---|---|
| keeper's kit cluster == his own team's cluster, by row | **1,313/5,384 = 0.2439** |
| by sequence majority | **13/57 = 0.228** |
| implied "the keeper is in the OPPONENT's cluster" | 0.756 rows, 44/57 = 0.772 sequences |

The two-way kit KMeans does not merely scatter keepers at random — it puts them in the *wrong*
cluster about three times in four, which is what a kit designed to contrast with one's own team
produces. Running `gk_self` with the cluster inverted is therefore a real rule at roughly 0.75, far
below the incumbent's 0.94, and it cannot be used as an override: at 75% reliability it would break
more clips than it fixes, and §5 says we cannot tell which clips to apply it to.

Closing this needs a keeper -> team link that is not kit colour. That is a separate act with a
quantified prize: all four test misses, roughly +2.4 GS-HOTA, and it is the only route to it that
this study found.

## 7. Verification and package status

`results/gsr_teamside_frozen.json` was written before the test split was read, naming `meanx`
(`eval.gsr_score.resolve_team_map_free`, unchanged), no margin, and the rule that a package would be
rebuilt only on an improvement over 45/49.

| | |
|---|---|
| **test-49, frozen rule, side-correctness** | **45/49 = 0.9184** |
| wrong | SNGS-126, SNGS-129, SNGS-131, SNGS-197 |
| change vs the incumbent | **none** — same count, same sequences |
| GT-oracle `meanx` on the same 49 | 45/49 (wrong on SNGS-126, -131, -195, -197) |

Identical to `GSR_DELEAK.md` §3, and the tool's `meanx` was checked against the shipped
`resolve_team_map_free` on all 58 valid sequences first: 58/58 agreement.

**No package was rebuilt.** `results/gsr_submission/gsr_testphase_gtfree_7dd2a50a.zip` and its
manifest stand exactly as N1 left them; GS-HOTA 31.88 is unchanged, no new hash, no manifest edit.
Nothing in the prediction chain was touched — this act changed no shipped code path.

## 8. Negatives, in one place

1. **The brief's prize does not exist in this family.** 0.98 is above the oracle ceiling of every
   positional rule measured (`meanx` 0.9739 on dev, 157/164 = 0.9573 pooled over all splits). Our
   estimator already matches the oracle's count on both graded splits.
2. **Solve-both-and-vote carries zero bits**, exactly, by construction and by measurement.
3. **Attack direction is not recoverable from a 30 s clip**: the possession-drift rules score
   0.5217 on *ground truth*.
4. **The proximity keeper rule is a depth rule in disguise** (GT ceiling 0.6903) and inherits every
   inversion; the keeper's value is definitional, not geometric.
5. **Ensemble gains are pool-specific noise**: each pool's winner loses on the other, and none beats
   `meanx` on ground truth.
6. **A shape rule can look good on our positions and have no ceiling at all.** Correlating a
   cluster's centroid-x with its spread scores 0.828 on valid-58 predictions and **0.603 on ground
   truth** — it was riding the depth signal through our noise. This is why the GT-oracle scan, not
   the prediction-side accuracy, is the selection instrument here.
7. **The margin is a ranking signal (AUC 0.867 test) but not a gate**, and GSR offers no abstention
   to spend it on.
8. **The dev pool is thin where it matters.** 4 whole-clip failures at the graded run; the window
   stress pool has 34, but its failures are the *estimation* mode, not the inversion mode, so it
   cannot be used to build a whole-clip confidence. The 39-sequence train extraction (~5.5 GPU
   hours, the cost `GSR_DELEAK.md` §3 declined) was launched for this act and 13 clips had landed at
   write-up; the oracle column, which is the one that decides, covers all 39 already.

## 9. Files

- `tools/gsr_teamside.py` — the whole study (`--invariance --ceiling --dev --windows N --gklink
  --freeze --verify --demo`).
- `results/gsr_teamside_frozen.json` — the pre-declaration.
- `results/gsr_benchmark/teamside/teamside_ceiling.json` — every rule on GT, dev-115.
- `results/gsr_benchmark/teamside/teamside_dev_k1.json` — every rule on our positions, whole clips.
- `results/gsr_benchmark/teamside/teamside_dev_k3.json` — the 3-window stress pool + margin AUCs.
- `results/gsr_benchmark/teamside/teamside_gklink.json` — the keeper -> cluster linkage rate.
- `results/gsr_benchmark/teamside/teamside_invariance.json` — family 1, both permutations.
- `results/gsr_benchmark/teamside/teamside_verify.json` — the single test read.
- `outputs/gsr_train_probe/positions/*.parquet` — the train-split probe (extraction in flight).
