# Does the broadcast calibration failure replicate? Four matches, 4 900 re-appearances (2026-07-27)

**Yes. It replicates on every match, at every testable horizon, with almost no spread.** The frozen
B4 v1 predictive regions (`results/B4_MODEL_V1.md`, P2 emit policy) cover **20.5–21.9%** where they
promise 50% and **51.4–57.8%** where they promise 90%, on four independent matches. **14 of 14
testable buckets under-cover at both levels; 0 of 14 pass.** The single-match finding of
`results/FULL_MATCH_RECONSTRUCTION.md` was not a fixture artefact.

Nothing was refitted, no threshold moved, no re-conformalisation was performed (§6). This document
is measurement plus a verdict.

Reproduce (CPU only, ~8–12 min per match, one at a time, then a 1-min aggregation):

```
python -m tools.full_match_reconstruction --match manutd_liverpool \
       --runlog results/reconstruction/manutd_liverpool_replication_runlog.md
python -m tools.full_match_reconstruction --match manutd_brighton  --runlog ...
python -m tools.full_match_reconstruction --match liverpool_manutd --runlog ...
python -m tools.calibration_replication
```

Code: `tools/calibration_replication.py` (aggregator, `--self-check`), harness unchanged at
`tools/full_match_reconstruction.py`. Verbatim console trails:
`results/reconstruction/{match}_replication_runlog.md` and (for `tottenham_manutd`)
`results/FULL_MATCH_RECONSTRUCTION_runlog.md`. Event tables:
`results/reconstruction/{match}_reappearance_events.parquet`.

---

## 1. Per-match summary

Primary link rule only (within-track, liveness-guarded: the tracker itself carried the id across
the gap and >= 4 gated players stayed alive on every gap frame). `sigma` is the per-axis
observation-noise floor of our own projected positions, measured from midpoint triples.

| match | events | PICP50 | PICP90 | RMSE emitted | RMSE hold | sigma (m/axis) | triples | trusted grid |
|---|---|---|---|---|---|---|---|---|
| `tottenham_manutd` | 1 259 | **20.5** | **55.3** | 3.47 | 3.53 | 0.586 | 65 714 | 30.0% of 81.1 min |
| `manutd_liverpool` | 791 | **21.9** | **57.8** | 2.84 | 3.05 | **0.278** | 45 025 | 23.0% of 75.0 min |
| `manutd_brighton` | 2 033 | **21.6** | **56.3** | 3.35 | 3.50 | 0.614 | 88 133 | 37.1% of 82.1 min |
| `liverpool_manutd` | 817 | **20.7** | **51.4** | 2.98 | 2.89 | 0.516 | 41 739 | 23.5% of 68.9 min |
| **POOLED** | **4 900** | **21.2** | **55.4** | 3.24 | 3.34 | **0.541** | 240 611 | — |
| Metrica holdout (frozen, same policy) | 644 792 | 45.5–52.4 | 89.2–90.6 | — | — | 0 (exact truth) | — | — |

Spread across matches: **1.4 points** at the 50% level and **6.4 points** at the 90% level. That is
tighter than the sampling noise on some of the individual buckets. There is nothing to attribute to
"a bad fixture".

## 2. Census by duration bucket — read this before the coverage tables

| match | 0-1s | 1-3s | 3-5s | 5-10s | 10-30s | 30s+ | total |
|---|---|---|---|---|---|---|---|
| `tottenham_manutd` | 782 | 351 | 91 | 33 | 2 | 0 | 1 259 |
| `manutd_liverpool` | 532 | 209 | 36 | 14 | 0 | 0 | 791 |
| `manutd_brighton` | 1 269 | 520 | 157 | 84 | 3 | 0 | 2 033 |
| `liverpool_manutd` | 513 | 238 | 52 | 14 | 0 | 0 | 817 |
| **pooled** | **3 096** | **1 318** | **336** | **145** | **5** | **0** | **4 900** |

The structural ceiling replicates exactly as predicted: ByteTrack's association buffer re-links
short losses only, so **10-30s carries 5 events across four matches and 30s+ carries none**. Four
extra matches multiplied the 0-3s evidence by four and added nothing at the horizons where v1's
published Metrica advantage lives (−5.17 m at 10-30s, −3.92 m at 30s+). Quadrupling the corpus
again would not change that; only an identity layer would.

## 3. Coverage and error by bucket (pooled, 4 matches)

| horizon | n | RMSE emitted | RMSE hold | PICP50 (nom. 50) | PICP90 (nom. 90) | mean r50 | mean r90 | pass |
|---|---|---|---|---|---|---|---|---|
| 0-1s | 3 096 | 2.28 | 2.31 | **18.1** | **51.9** | 0.2 | 0.6 | no |
| 1-3s | 1 318 | 3.80 | 3.88 | **22.6** | **55.8** | 1.2 | 2.5 | no |
| 3-5s | 336 | 5.38 | 5.74 | **36.3** | **73.8** | 2.7 | 5.6 | no |
| 5-10s | 145 | 6.64 | 7.00 | **39.3** | **84.8** | 4.3 | 8.8 | no |
| 10-30s | 5 | 8.23 | 4.99 | 60.0 | 100.0 | 6.1 | 13.1 | (n=5) |
| 30s+ | 0 | — | — | — | — | — | — | untestable |
| **ALL** | **4 900** | **3.24** | **3.34** | **21.2** | **55.4** | 0.8 | 1.7 | **no** |

Per-match, per-bucket PICP (the replication in full; every cell below nominal):

| horizon | tottenham_manutd | manutd_liverpool | manutd_brighton | liverpool_manutd |
|---|---|---|---|---|
| 0-1s | 17.3 / 51.3 | 20.9 / 56.2 | 17.9 / 51.5 | 17.0 / 49.1 |
| 1-3s | 20.5 / 56.7 | 22.5 / 58.4 | 24.2 / 56.9 | 22.3 / 49.6 |
| 3-5s | 40.7 / 73.6 | 27.8 / 69.4 | 34.4 / 75.2 | 40.4 / 73.1 |
| 5-10s | 39.4 / 81.8 (n=33) | 35.7 / 78.6 (n=14) | 36.9 / 86.9 (n=84) | 57.1 / 85.7 (n=14) |

The shape of the failure replicates too: **coverage improves monotonically with horizon** in every
match. The regions fail worst where they are tightest and approach nominal only where they are
wide.

## 4. The observation-noise floor — and why it is NOT the explanation

| match | sigma (m per axis) | radial RMS | ALL PICP50 | ALL PICP90 | same regions widened by that match's sigma |
|---|---|---|---|---|---|
| `tottenham_manutd` | 0.586 | 0.83 | 20.5 | 55.3 | 55.4 / 71.4 |
| `manutd_liverpool` | **0.278** | 0.39 | 21.9 | 57.8 | 40.6 / 66.2 |
| `manutd_brighton` | 0.614 | 0.87 | 21.6 | 56.3 | 58.4 / 73.0 |
| `liverpool_manutd` | 0.516 | 0.73 | 20.7 | 51.4 | 51.0 / 64.9 |
| **pooled** | **0.541** | **0.77** | 21.2 | 55.4 | 55.0 / 70.4 |

**Pooled sigma = 0.541 m per axis (0.765 m RMS radially), from 240 611 midpoint triples.** That is
the honest magnitude of the unmodellable measurement floor on our footage, and it is a real result
in its own right: the frozen 90% region at 0-1s is **0.6–0.7 m** in radius, i.e. **smaller than the
noise on the truth it is being scored against**.

But sigma is **not stable across matches** — 0.278 to 0.614 m, a factor of 2.2 — and the decisive
observation is that **coverage does not follow it**. `manutd_liverpool` has half the noise of
`manutd_brighton` and *identical* coverage (21.9 / 57.8 vs 21.6 / 56.3). `liverpool_manutd` sits at
mid noise with the *worst* 90% coverage. With four matches there is no usable correlation between
sigma and PICP at all.

Conclusion, stated against my own preferred explanation: the measured observation noise explains
part of the 0-1s under-coverage (widening by the pooled sigma lifts 0-1s from 18.1/51.9 to
67.0/73.9 — and over-corrects the 50% level while still missing the 90% level) and **explains
essentially none of the 1-3s and 3-5s failure** (22.6 -> 32.4 and 36.3 -> 38.4 at the 50% level).
Widening the frozen regions in quadrature by the pooled 0.541 m would take the ALL row to
55.0 / 70.4 — the 50% level would then be roughly right and **the 90% level would still be 20
points short**. So sigma is a necessary correction and a real number, but the regions are also
intrinsically too narrow on broadcast, and the gap is not a measurement artefact.

## 5. Selection-bias check — the sample is easier than a real occlusion, on all four matches

Displacement `|truth(t1) - last seen(t0)|` (the hold-last error), against the Metrica holdout at the
same elapsed time:

| horizon | our p50 (pooled) | our mean (pooled) | Metrica p50 | per-match p50 spread |
|---|---|---|---|---|
| 0-1s | 0.52 | 1.05 | 0.73 | 0.51 – 0.53 |
| 1-3s | 1.23 | 2.28 | 3.08 | 1.05 – 1.33 |
| 3-5s | 2.43 | 3.76 | 6.10 | 1.88 – 3.12 |
| 5-10s | 3.22 | 4.74 | 10.10 | 1.98 – 4.02 |

Players in our events move **2–3x less** than Metrica players over the same elapsed time, in every
match. The tracker only re-associates a player who came back near where it left, so hold-last is
near-optimal on this sample by construction — which is why the point model shows no skill here
(ALL row, block-bootstrap CI: **vs hold** +0.09 to −0.21 m with the CI straddling zero on all four
matches; **vs anchor** +0.07 to +0.13 m, i.e. marginally *worse* than its own anchor, with the CI
excluding zero on three of four) and why that tie is **a property of the test, not a measured
property of the model**.

The rest of the bias profile replicates as well: 61.6–64.9% of track-losses ever re-appear under the
same id, only 9.6–15.1% become usable events, and a usable event's last sighting is near the image
edge on 6.4–8.5% of cases against 11.3–14.3% for terminal losses. Our truth set is dominated by
**detector dropouts on players the camera never left**, not by off-camera exits.

Crucially, the calibration failure survives this bias: on the displacement-matched Metrica subset
(the same short movements, 30 000–66 000 samples per cell) the frozen regions still cover
**52–59% / 92–95%** in all four matches' comparisons. On our comparably easy broadcast events they
cover **17–41% / 49–87%** over the testable buckets.

Physically impossible links (implied speed > 12 m/s = id switch or gross projection failure) are
1.0–2.1% per match; excluding them (a diagnostic that flatters the model) moves PICP by at most 1.2
points in any match. The under-coverage is broad, not outlier-driven.

---

## 6. What must NOT be concluded from this

* **Do not re-conformalise on re-appearances.** 4 900 events with measured truth is tempting, and
  it is the wrong sample: §5 shows it is a biased, near-stationary subset of occlusions, so
  multipliers fitted on it would be tight exactly where the model is used (>= 5 s, off-camera). No
  multiplier in this document was changed and none should be, until there is either annotated
  off-screen truth or a real identity layer.
* **This says nothing about 10-30s or 30s+.** 5 events across four matches. Every long-horizon
  claim, in either direction, remains unsupported on broadcast.
* **This is not a verdict on the point model.** §5 explains why the RMSE tie is uninformative.

## 7. The corrected claim — exact wording to use project-wide

Replace every occurrence of "calibrated 50% / 90% predictive region" (applied to broadcast) with:

> The imputed positions carry the frozen B4 v1 model's **nominal** 50% / 90% predictive regions,
> conformally calibrated on Metrica simulated censoring (6/6 buckets pass there). **On real
> broadcast footage they are not calibrated:** measured against 4 900 liveness-guarded
> re-appearances across four Premier League matches, they cover **21.2%** where they promise 50%
> and **55.4%** where they promise 90%, failing 14/14 testable buckets on every match tested. The
> regions are roughly **half the area they should be** at short horizons. Part of the gap is a
> measured observation-noise floor on our own projected truth (pooled **sigma = 0.54 m per axis**,
> 0.77 m radial, n = 240 611); widening the same regions in quadrature by it recovers the 50% level
> (55.0%) but still leaves the 90% level 20 points short (70.4%). Coverage beyond 10 s of occlusion
> is untestable with this footage and is not claimed.

Short form for captions and legends:

> model region, nominal 50/90% — MEASURED coverage on our broadcast is 21% / 55% (4 900
> re-appearances, 4 matches); not calibrated here.

`results/FULL_MATCH_RECONSTRUCTION.md` §2.4 and §5 stand as written and are now upgraded from a
single-match finding to a replicated one.

## 8. Demo honesty fix (done in the same pass)

`tools/tactical_clip.py` asserted "frozen B4 v1: 50% / 90% region" on every rendered frame. Two
changes, both shipped and re-rendered:

1. **Relabelled.** The legend now reads `IMPUTED — model region: NOMINAL 50% / 90%` with an amber
   line under it: `regions NOT calibrated here: measured 20.5% / 55.3% coverage`, and the footer
   carries the measured numbers, the sample size, the match and the pointer to this report. The
   constants live in `tactical_clip.PICP50_MEASURED` / `PICP90_MEASURED`, and its `--self-check`
   asserts nothing in the on-screen key can say "calibrated" without "NOT".
2. **The underlying defect is fixed.** `imputation_b4_external.load_ours` ghosts a track only after
   its FINAL sighting, so the clip renderer could only ever draw dead re-identification fragments
   and never a live player who dropped out for two seconds and came back. `_impute` now applies
   `full_match_reconstruction.fill_internal_gaps` (a data-assembly step — the placeholder is read by
   nothing but the choice of which slot-frames get a prediction). On the flagship passage the ghost
   population went from **0% to 56% live-track occlusions**; the panel now shows 2s/3s/9s occlusion
   ghosts that it previously omitted entirely.

Re-rendered: `results/tactical_clips/tottenham_manutd_h1_0343s.mp4` (+ `_sheet.png`),
`tottenham_manutd` h1 05:43–05:57.
