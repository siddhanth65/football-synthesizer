# Possession trackability-bias correction probe

Question: can the CV possession estimator's trackability bias be corrected in a validated way?

Scope: 4 matches. `mun_mci` is excluded -- the registry gives it neither a PMSR oracle nor a `ball_dir`, so there is no ground-truth possession and no CV possession to correct.

Estimand: team0 possession share (%). CV = share of `assign_possession` (smooth=True) samples on calib<=1m trackable frames -- the exact `report.facts` path. Oracle = FIFA PMSR possession re-normalised over the two named teams (the contested/dead-ball bucket dropped) for the France matches, cached Sofascore for Brighton.

## Oracle + raw estimator

| match | oracle team0 | raw CV team0 | raw error (pp) |
|-------|-------------:|-------------:|---------------:|
| france_iraq | 52.5% | 58.5% | +5.9 |
| france_senegal | 52.6% | 53.4% | +0.7 |
| france_norway | 44.5% | 47.6% | +3.1 |
| brighton_manutd | 52.0% | 44.2% | -7.8 |

Baseline median |error| = **4.53 pp**. Note the bias is consistent in *mechanism* (the positional side is over-counted) but flips relative to team0: France (the positional side) is inflated in the WC matches (+ error), Man Utd (the direct side) is deflated at Brighton (- error).

## Why post-stratification has no purchase (zone diagnostic)

Longitudinal thirds (0=defensive, 1=middle, 2=attacking by absolute pitch-x of the outfield centroid). `true_time` = share of all calib-gated frames; `tracked_poss` = share of possession samples; `team0_share` = team0's tracked possession within the zone.

### france_iraq (raw 58.5%, oracle 52.5%)

| zone | true_time | tracked_poss | team0_share |
|------|----------:|-------------:|------------:|
| 0 | 0.250 | 0.248 | 0.622 |
| 1 | 0.455 | 0.463 | 0.567 |
| 2 | 0.294 | 0.290 | 0.581 |

### france_senegal (raw 53.4%, oracle 52.6%)

| zone | true_time | tracked_poss | team0_share |
|------|----------:|-------------:|------------:|
| 0 | 0.234 | 0.191 | 0.508 |
| 1 | 0.566 | 0.628 | 0.545 |
| 2 | 0.200 | 0.182 | 0.520 |

### france_norway (raw 47.6%, oracle 44.5%)

| zone | true_time | tracked_poss | team0_share |
|------|----------:|-------------:|------------:|
| 0 | 0.330 | 0.293 | 0.385 |
| 1 | 0.476 | 0.522 | 0.541 |
| 2 | 0.195 | 0.185 | 0.435 |

### brighton_manutd (raw 44.2%, oracle 52.0%)

| zone | true_time | tracked_poss | team0_share |
|------|----------:|-------------:|------------:|
| 0 | 0.345 | 0.337 | 0.407 |
| 1 | 0.439 | 0.449 | 0.448 |
| 2 | 0.216 | 0.214 | 0.483 |

Read: `true_time` and `tracked_poss` nearly coincide in every zone (the position-only stratum barely separates trackable from untrackable frames), and `team0_share` is roughly flat across zones. In Brighton team0 (Man Utd) sits *below* its 52% oracle in **every** third -- the deficit is uniform across field position, so no reweighting of zone time can recover it. The missingness is informative *within* strata, which post-stratification cannot fix.

## Candidate corrections (all parameter-free -> LOO == direct)

Each correction is justified by the bias mechanism and uses **zero** oracle information, so its leave-one-out estimate equals its direct estimate; the LOO error below is `|corrected - oracle|`.

### poststratify_zone

| match | corrected team0 | LOO error (pp) | delta vs raw |err| (pp) |
|-------|----------------:|---------------:|----------------------:|
| france_iraq | 58.5% | +6.0 | +0.02 |
| france_senegal | 53.2% | +0.5 | -0.21 |
| france_norway | 46.9% | +2.5 | -0.67 |
| brighton_manutd | 44.1% | -7.9 | +0.03 |

median |error| = **4.20 pp** (baseline 4.53); improved=True; worsened>2pp=none; moved-toward-oracle-every-match=False. **REJECT**

### chunk_time_weight

| match | corrected team0 | LOO error (pp) | delta vs raw |err| (pp) |
|-------|----------------:|---------------:|----------------------:|
| france_iraq | 58.1% | +5.5 | -0.40 |
| france_senegal | 52.9% | +0.3 | -0.41 |
| france_norway | 48.1% | +3.6 | +0.51 |
| brighton_manutd | 44.1% | -7.9 | +0.03 |

median |error| = **4.58 pp** (baseline 4.53); improved=False; worsened>2pp=none; moved-toward-oracle-every-match=False. **REJECT**

### spell_extrapolate

| match | corrected team0 | LOO error (pp) | delta vs raw |err| (pp) |
|-------|----------------:|---------------:|----------------------:|
| france_iraq | 51.0% | -1.5 | -4.42 |
| france_senegal | 52.1% | -0.5 | -0.25 |
| france_norway | 50.4% | +6.0 | +2.84 |
| brighton_manutd | 44.3% | -7.7 | -0.18 |

median |error| = **3.74 pp** (baseline 4.53); improved=True; worsened>2pp=['france_norway']; moved-toward-oracle-every-match=False. **REJECT**

## Verdict

**No candidate meets the pre-committed acceptance** (median |error| improves AND no match worsens by >2pp AND every match moves toward its oracle).

- `poststratify_zone`: near-null (<0.7 pp any match) -- the zone diagnostic shows trackability and team0 share are flat across the only position-only stratum we can build, so there is nothing to reweight.

- `chunk_time_weight`: null-to-negative -- confirms the diagnosis that the bias is within-phase, not chunk-level coverage.

- `spell_extrapolate`: the only large lever (fixes france_iraq +5.9->-1.5, helps senegal) but it injects a center-pull from midpoint gap-splitting that *worsens* the genuinely lopsided france_norway by 2.8 pp, and it cannot recover Brighton's wholesale-missing Man Utd possessions (44.2->44.3).

**Recommendation: declare the possession estimator uncorrectable without event data.** The deficit is informative missingness *inside* every position stratum (entire untrackable transition/direct spells), not a reweightable stratum imbalance. Correcting it needs touch/possession events, not more position or timing geometry. Keep the metric as the caveated 'trackable-frame possession share' it already is; do not ship any of these corrections into `report.facts`.
