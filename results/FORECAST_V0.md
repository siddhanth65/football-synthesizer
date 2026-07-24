# FORECAST_V0 -- C5 pre-match forecast v0 (BTP B1.3 / REVIEW_CRIB Q6)

Six same-opponent home/away pairs across the ten Hag -> Amorim change. Leg-1 = the Aug-Sep 2024 (ten Hag) meeting; rematch = the Jan-Feb 2025 (Amorim) meeting. We forecast each rematch from pre-match information only and score vs three baselines. **n = 6 = proof-of-signal, not significance; every pair changes both venue and manager between legs.**

## Protocol (PRE-REGISTERED 2026-07-24, before any number was computed)

**Task.** For each of the 6 rematches (Amorim, Jan-Feb 2025), forecast from ONLY pre-match
information -- the leg-1 (ten Hag, Aug-Sep 2024) broadcast metrics, season-public data, and clubelo
Elo at the rematch date -- three targets: (a) Man Utd possession share (point + interval), (b) the
W/D/L result distribution, (c) whether the attack mix is MORE or LESS direct than leg 1. No rematch
broadcast, score, or possession value enters any forecast; those are used only to score it.

**Signal being tested.** ``PAIR_ANALYSIS_v1`` established cross-leg possession identity r = +0.83
(the one repeatable fingerprint) and block height r = -0.40 (not stable). So possession is
forecast by PERSISTENCE (leg-1 value); block height is deliberately NOT forecast.

**(a) Possession.** Primary forecast = persistence: the rematch poss-link share = the leg-1
poss-link share, verbatim. Interval = leg-1 value +/- k * sigma, with sigma = the LOPO cross-pair
residual std (each pair's band uses the residual spread of the OTHER five pairs). Report the 68%
(k=1) and 90% (k=1.64) bands and their empirical coverage. A regime-shift variant (persistence +
LOPO mean residual, i.e. the ten Hag->Amorim possession drop) is explored but only adopted if it
robustly beats persistence. Scored by MAE (pp) + interval coverage.

**(b) Result.** Forecast = 0.5 * Elo-only kickoff W/D/L (the base-subset WP model at score 0-0, 90
min left, rematch venue + Elo) + 0.5 * a pre-registered ordinal smoothing of the leg-1 result
(L -> .60/.25/.15, D -> .20/.60/.20, W -> .15/.25/.60 over L/D/W). The 0.5 weight is pre-registered
(equal trust in the strength prior and the last meeting), NOT fitted. Scored by the ranked
probability score (RPS, lower better) averaged over the 6 rematches.

**(c) Attack direction.** Forecast the SIGN of (rematch direct-share - leg-1 direct-share) from the
LOPO mean of the other five pairs' shifts (i.e. the regime tendency, honestly held out). Scored by
hit rate over 6, with n and the low 3-way coverage stated.

**Baselines every forecast must beat or tie.** (i) uninformative: possession = even 50%, result =
flat 1/3-1/3-1/3, direction = coin flip (3/6 expected); (ii) Elo-only: result = the WP kickoff
probs; possession = a LOPO linear fit of poss-link on the rematch Elo gap; (iii) persistence: the
leg-1 value verbatim (possession) / the leg-1 result smoothed prior alone (result).

**Honesty.** n=6 = proof-of-signal, never significance. Every pair changes BOTH venue AND manager
between legs, so leg-1->rematch shifts confound venue with the managerial change -- not separable
here. Any fitted parameter is LOPO. The question answered: does leg-1 broadcast analysis carry ANY
predictive information beyond Elo?


---

### (a) Possession -- per-pair forecast vs actual (poss-link share, %)

| pair | leg-1 (=forecast) | 68% band | 90% band | actual | abs err | in68 | in90 |
|------|------------------:|----------|----------|-------:|--------:|:----:|:----:|
| Brighton | 47.5 | [39.3, 55.7] | [34.1, 60.9] | 47.8 | 0.3 | Y | Y |
| Liverpool | 52.8 | [46.6, 59.0] | [42.6, 63.0] | 34.8 | 18.0 | n | n |
| Fulham | 58.0 | [49.1, 66.9] | [43.4, 72.6] | 51.7 | 6.3 | Y | Y |
| Crystal Palace | 84.4 | [75.8, 93.0] | [70.2, 98.6] | 74.0 | 10.4 | n | Y |
| Southampton | 63.9 | [55.0, 72.8] | [49.3, 78.5] | 56.0 | 7.9 | Y | Y |
| Tottenham | 30.7 | [24.1, 37.3] | [19.8, 41.6] | 35.3 | 4.6 | Y | Y |

**Possession MAE (pp), 6 rematches:**

| forecaster | MAE |
|------------|----:|
| persistence (leg-1 verbatim) = **primary** | **7.92** |
| baseline (i) uninformative (even 50%) | 10.63 |
| baseline (ii) Elo-only (LOPO poss ~ Elo gap) | 12.18 |
| regime-shift variant (persistence + LOPO drop) | 6.99 |

Interval coverage: 68% band 4/6 = 67% (nominal 68%); 90% band 5/6 = 83% (nominal 90%). Mean LOPO sigma ~ 7.9 pp.

Mean rematch-minus-leg1 residual = -6.3 pp (all rematches are Amorim, all leg-1 ten Hag, so this ten Hag->Amorim / regression-to-mean drop is NOT split from venue). The regime-shift variant applies it LOPO: it helps high-possession pairs but hurts the stable ones (Brighton, Tottenham), so it is **not adopted** -- persistence is the more defensible point forecast at n=6.

---

### (b) Result -- per-pair Elo-only vs forecast vs actual

| pair | venue | Elo gap | Elo-only L/D/W | forecast L/D/W | leg-1 | actual |
|------|:-----:|--------:|----------------|----------------|:-----:|:------:|
| Brighton | H | +1 | 0.33/0.26/0.42 | 0.46/0.25/0.28 | L | L |
| Liverpool | A | -253 | 0.65/0.21/0.14 | 0.62/0.23/0.15 | L | D |
| Fulham | A | -10 | 0.43/0.26/0.32 | 0.29/0.25/0.46 | W | W |
| Crystal Palace | H | +4 | 0.32/0.26/0.42 | 0.26/0.43/0.31 | D | L |
| Southampton | H | +206 | 0.17/0.22/0.61 | 0.16/0.24/0.60 | W | W |
| Tottenham | A | -28 | 0.44/0.25/0.30 | 0.52/0.25/0.23 | L | L |

**Mean RPS (lower better), 6 rematches:**

| forecaster | mean RPS |
|------------|---------:|
| forecast (0.5 Elo + 0.5 leg-1) = **primary** | **0.1882** |
| baseline (i) uninformative (flat 1/3) | 0.2500 |
| baseline (ii) Elo-only | 0.2445 |
| baseline (iii) persistence (leg-1 prior alone) | 0.1494 |

---

### (c) Attack direction -- more/less direct than leg 1

| pair | leg-1 direct | rematch direct | actual | LOPO pred | hit | n3 leg1/rem |
|------|-------------:|---------------:|:------:|:---------:|:---:|:-----------:|
| Brighton | 0.20 | 0.22 | more | more | Y | 10/9 |
| Liverpool | 0.27 | 0.42 | more | more | Y | 11/12 |
| Fulham | 0.25 | 0.35 | more | more | Y | 16/20 |
| Crystal Palace | 0.22 | 0.70 | more | less | n | 18/10 |
| Southampton | 0.46 | 0.30 | less | more | n | 13/10 |
| Tottenham | 1.00 | 0.56 | less | more | n | 3/16 |

LOPO regime-sign hit rate = **3/6 = 50%** (coin flip = 3/6). In-sample 'Amorim more direct' would score 4/6, but that reuses the pooled tendency these pairs define (circular). LOPO is the honest number. Leg-1 3-way coverage is low (Tottenham leg-1 n3 = 3, degenerate 100% direct); read as direction-only over tiny n.

---

## Verdict

**Does leg-1 broadcast analysis carry predictive information beyond Elo? For possession, YES (weakly): the broadcast poss-link share persists and beats both the uninformative and the Elo baselines. Result is best forecast by the last-meeting outcome (a public H2H prior, beating Elo), not by broadcast shape. Attack direction shows no signal that survives honest holdout. n=6 = proof-of-signal only.**

- **Possession -- SIGNAL (weak but real).** Persisting the leg-1 poss-link share predicts the rematch with MAE ~7.9 pp, beating the uninformative 50% baseline (~10.6 pp) and the Elo-only fit; the 68%/90% intervals are roughly calibrated at n=6. This is the r=+0.83 fingerprint carrying into a genuine forecast. Elo alone does not reproduce it, so leg-1 broadcast possession carries information beyond Elo.
- **Result -- the last meeting beats Elo, and the pre-registered blend was suboptimal.** Persistence of the leg-1 RESULT alone scores RPS 0.149, clearly beating Elo-only (0.245) and flat (0.250); the pre-registered 0.5-Elo/0.5-leg-1 blend (0.188) lands in between because its Elo half is the WEAKER signal here and drags pure persistence up. So the last-meeting outcome carries more forecast information than the strength prior for these 6 (results are opponent-stable -- 4/6 keep the exact result). Caveat: this is a public head-to-head prior, not a broadcast-derived metric; the pre-registered equal weight was a wrong call in hindsight, honestly reported.
- **Attack direction -- NO signal survives LOPO.** The honest leave-one-pair-out regime-sign forecast scores at chance (3/6); the 4/6 'Amorim more direct' hit rate is an in-sample artifact of the pooled tendency these same pairs define, and leg-1 3-way coverage is too low (n3 as small as 3) to forecast direction reliably.

## What v1 needs

- **More matches.** n=6 caps everything at proof-of-signal; the demo opponent's OTHER league games (opponent priors from THEIR matches, not just the one Utd meeting) would de-confound venue/manager and give real intervals.
- **Opponent-conditioned possession prior** (their control profile vs the field), so the forecast is not pure Utd persistence.
- **De-confound venue and manager** -- impossible in this corpus (every pair flips both); needs same-manager home/away repeats.
- **Higher ball coverage for attack typing** -- 3-way coverage ~10-20% makes the direct share too noisy to forecast; this is upstream (ball track) work.
