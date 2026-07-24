"""C5 forecast v0: turn leg-1 broadcast analysis into a scored PRE-MATCH forecast (BTP B1.3, Q6).

Six same-opponent home/away pairs straddle the ten Hag -> Amorim change: leg-1 = the Aug-Sep 2024
meeting (ten Hag), rematch = the Jan-Feb 2025 meeting (Amorim). For each of the six REMATCHES we
forecast three things from ONLY pre-match information (leg-1 broadcast metrics + season-public
data + clubelo Elo at the rematch date -- never any rematch broadcast/result data):

    (a) Man Utd possession share  (point + interval)
    (b) match result distribution (W/D/L)
    (c) direction of the attack-type mix (more / less direct than leg 1)

and score each against three baselines. The protocol is PRE-REGISTERED (see :data:`PROTOCOL`,
written before any number computed) and every fitted parameter is leave-one-pair-out (LOPO) held.

n = 6 is proof-of-signal, NEVER significance; and every pair changes BOTH venue AND manager between
its legs, so leg-1<->rematch differences confound venue with the managerial change -- stated loudly.
The deliverable question: **does leg-1 broadcast analysis carry ANY predictive information beyond
Elo?**

CPU only, cache-first (StatsBomb 15/16 timelines + clubelo snapshots). Run::

    python tools/forecast_v0.py

Writes ``results/FORECAST_V0.md``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fingerprint import win_probability as wp  # noqa: E402
from tools import build_win_probability as bwp  # noqa: E402
from tools import wp_opendata as od  # noqa: E402

OUT = Path("results/FORECAST_V0.md")
MANU = "Man Utd"
RES_IDX = {"L": 0, "D": 1, "W": 2}          # ordinal loss<draw<win == wp.RESULTS order
RES_NAME = ("loss", "draw", "win")
BLEND_W = 0.5                                # pre-registered: equal trust in Elo + last meeting
# Pre-registered ordinal one-hot smoothing of a leg-1 result into a W/D/L prior (order L, D, W):
LEG1_PRIOR = {"L": (0.60, 0.25, 0.15), "D": (0.20, 0.60, 0.20), "W": (0.15, 0.25, 0.60)}


@dataclass(frozen=True)
class Pair:
    """One same-opponent leg-1 / rematch pair (Man Utd perspective).

    Possession fields are the mapping-CORRECTED poss-link share (broadcast) and Sofascore possession
    from ``results/PAIR_ANALYSIS_v1.md`` Task 2; attack fields are the corrected 3-way direct counts
    from ``results/GAME_STATE_v2.md``; results are the Man Utd W/D/L from the same tables.
    """

    opp: str
    leg1_id: str
    rem_id: str
    rem_date: str
    rem_home: bool               # Man Utd at home in the rematch
    leg1_poss: float             # Man Utd poss-link share, leg 1 (%)
    rem_poss: float              # Man Utd poss-link share, rematch (%) -- ACTUAL, scoring only
    leg1_sofa: float             # Sofascore Man Utd possession, leg 1 (%)
    rem_sofa: float              # Sofascore Man Utd possession, rematch (%) -- ACTUAL
    leg1_res: str                # Man Utd result leg 1 (W/D/L)
    rem_res: str                 # Man Utd result rematch (W/D/L) -- ACTUAL
    leg1_direct_n: int           # leg-1 3-way direct count / n_3way
    leg1_n3: int
    rem_direct_n: int            # rematch 3-way direct count / n_3way -- ACTUAL
    rem_n3: int

    @property
    def leg1_direct(self) -> float:
        """Leg-1 direct share of 3-way-labelled attacks."""
        return self.leg1_direct_n / self.leg1_n3

    @property
    def rem_direct(self) -> float:
        """Rematch direct share (ACTUAL, scoring only)."""
        return self.rem_direct_n / self.rem_n3


# Source-of-truth validated numbers (provenance in the dataclass docstring). Leg-1 = ten Hag
# Aug-Sep 2024; rematch = Amorim Jan-Feb 2025; every pair flips venue too.
PAIRS: list[Pair] = [
    Pair("Brighton", "brighton_manutd", "manutd_brighton", "2025-01-19", True,
         47.5, 47.8, 52, 52, "L", "L", 2, 10, 2, 9),
    Pair("Liverpool", "manutd_liverpool", "liverpool_manutd", "2025-01-05", False,
         52.8, 34.8, 53, 47, "L", "D", 3, 11, 5, 12),
    Pair("Fulham", "manutd_fulham", "fulham_manutd", "2025-01-26", False,
         58.0, 51.7, 55, 49, "W", "W", 4, 16, 7, 20),
    Pair("Crystal Palace", "palace_manutd", "manutd_palace", "2025-02-02", True,
         84.4, 74.0, 67, 67, "D", "L", 4, 18, 7, 10),
    Pair("Southampton", "southampton_manutd", "manutd_southampton", "2025-01-16", True,
         63.9, 56.0, 56, 60, "W", "W", 6, 13, 3, 10),
    Pair("Tottenham", "manutd_tottenham", "tottenham_manutd", "2025-02-16", False,
         30.7, 35.3, 39, 44, "L", "L", 3, 3, 9, 16),
]

PROTOCOL = """\
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
"""


def rps(probs: tuple[float, float, float], obs_idx: int) -> float:
    """Ranked probability score for an ordered 3-class (loss<draw<win) forecast; lower is better."""
    obs = [0.0, 0.0, 0.0]
    obs[obs_idx] = 1.0
    cp = co = s = 0.0
    for k in range(2):                     # K-1 = 2 cumulative terms
        cp += probs[k]
        co += obs[k]
        s += (cp - co) ** 2
    return 0.5 * s


def blend(elo: tuple[float, float, float], leg1_res: str,
          w: float = BLEND_W) -> tuple[float, float, float]:
    """Forecast W/D/L = ``w`` * Elo-only + ``(1-w)`` * the leg-1 ordinal prior."""
    prior = LEG1_PRIOR[leg1_res]
    return tuple(w * e + (1.0 - w) * p for e, p in zip(elo, prior))  # type: ignore[return-value]


def elo_only_result(model, elo_manu: float, elo_opp: float,
                    is_home: int) -> tuple[float, float, float]:
    """Kickoff (0-0, 90 min left) W/D/L from the base-subset WP model = the Elo-only baseline."""
    import pandas as pd  # noqa: PLC0415

    feats = pd.DataFrame([{"minutes_remaining": 90, "score_diff": 0, "elo_diff": elo_manu - elo_opp,
                           "red_diff": 0, "yellow_diff": 0, "is_home": is_home}])
    p = wp.outcome_proba(model, feats).iloc[0]
    return float(p["loss"]), float(p["draw"]), float(p["win"])


def lopo_mean(values: list[float]) -> list[float]:
    """For each index, the mean of the OTHER entries (leave-one-out)."""
    tot, n = sum(values), len(values)
    return [(tot - v) / (n - 1) for v in values]


def lopo_linfit_predict(x: list[float], y: list[float]) -> list[float]:
    """LOPO prediction of y from a simple linear fit on the other pairs (Elo-only possession)."""
    x_a, y_a = np.asarray(x, float), np.asarray(y, float)
    preds = []
    for i in range(len(x_a)):
        mask = np.arange(len(x_a)) != i
        b, a = np.polyfit(x_a[mask], y_a[mask], 1)
        preds.append(float(a + b * x_a[i]))
    return preds


def build_model():
    """Fit the base-subset WP model on the cached StatsBomb PL 15/16 corpus (reuses build tool)."""
    return wp.fit(bwp.build_corpus())


def elo_gaps(model) -> list[tuple[float, float, int]]:
    """(Elo Man Utd, Elo opp, is_home) at each rematch date."""
    del model
    out = []
    for p in PAIRS:
        em = od.elo_at(MANU, p.rem_date)
        eo = od.elo_at(p.opp, p.rem_date)
        out.append((em, eo, int(p.rem_home)))
    return out


def possession_block(md: list[str]) -> None:
    """Score the possession forecast (persistence + interval) vs uninformative / Elo baselines."""
    leg1 = [p.leg1_poss for p in PAIRS]
    actual = [p.rem_poss for p in PAIRS]
    resid = [a - lg for a, lg in zip(actual, leg1)]
    # LOPO residual std per pair (spread of the OTHER five residuals) for an honest interval.
    lopo_sigma = []
    for i in range(len(PAIRS)):
        other = [r for j, r in enumerate(resid) if j != i]
        lopo_sigma.append(float(np.std(other, ddof=1)))
    persist_ae = [abs(a - lg) for a, lg in zip(actual, leg1)]
    unif_ae = [abs(a - 50.0) for a in actual]                     # even-possession baseline
    # Elo-only possession baseline: LOPO linear fit of rematch poss-link on rematch Elo gap.
    gaps = [em - eo for em, eo, _ in elo_gaps(None)]
    elo_pred = lopo_linfit_predict(gaps, actual)
    elo_ae = [abs(a - e) for a, e in zip(actual, elo_pred)]
    # Regime-shift variant: persistence + LOPO mean residual (explored, adopt only if it beats).
    shift = lopo_mean(resid)
    shift_pred = [lg + s for lg, s in zip(leg1, shift)]
    shift_ae = [abs(a - s) for a, s in zip(actual, shift_pred)]

    cov68 = cov90 = 0
    md.append("### (a) Possession -- per-pair forecast vs actual (poss-link share, %)\n")
    md.append("| pair | leg-1 (=forecast) | 68% band | 90% band | actual | abs err | in68 | in90 |")
    md.append("|------|------------------:|----------|----------|-------:|--------:|:----:|:----:|")
    for p, lg, a, sg in zip(PAIRS, leg1, actual, lopo_sigma):
        lo68, hi68 = lg - sg, lg + sg
        lo90, hi90 = lg - 1.64 * sg, lg + 1.64 * sg
        in68 = lo68 <= a <= hi68
        in90 = lo90 <= a <= hi90
        cov68 += in68
        cov90 += in90
        y68, y90 = "Y" if in68 else "n", "Y" if in90 else "n"
        md.append(f"| {p.opp} | {lg:.1f} | [{lo68:.1f}, {hi68:.1f}] | [{lo90:.1f}, {hi90:.1f}] | "
                  f"{a:.1f} | {abs(lg - a):.1f} | {y68} | {y90} |")
    md.append("")
    md.append("**Possession MAE (pp), 6 rematches:**\n")
    md.append("| forecaster | MAE |")
    md.append("|------------|----:|")
    md.append(f"| persistence (leg-1 verbatim) = **primary** | **{np.mean(persist_ae):.2f}** |")
    md.append(f"| baseline (i) uninformative (even 50%) | {np.mean(unif_ae):.2f} |")
    md.append(f"| baseline (ii) Elo-only (LOPO poss ~ Elo gap) | {np.mean(elo_ae):.2f} |")
    md.append(f"| regime-shift variant (persistence + LOPO drop) | {np.mean(shift_ae):.2f} |")
    md.append("")
    md.append(f"Interval coverage: 68% band {cov68}/6 = {cov68 / 6:.0%} (nominal 68%); "
              f"90% band {cov90}/6 = {cov90 / 6:.0%} (nominal 90%). "
              f"Mean LOPO sigma ~ {np.mean(lopo_sigma):.1f} pp.\n")
    drop = float(np.mean(resid))
    md.append(f"Mean rematch-minus-leg1 residual = {drop:+.1f} pp (all rematches are Amorim, all "
              f"leg-1 ten Hag, so this ten Hag->Amorim / regression-to-mean drop is NOT split "
              f"from venue). The regime-shift variant applies it LOPO: it helps high-possession "
              f"pairs but hurts the stable ones (Brighton, Tottenham), so it is **not adopted** -- "
              f"persistence is the more defensible point forecast at n=6.\n")


def result_block(md: list[str], model) -> None:
    """Score the Elo+leg-1 result forecast vs flat / Elo-only / persistence-of-result."""
    gaps = elo_gaps(model)
    rows, rps_fore, rps_flat, rps_elo, rps_pers = [], [], [], [], []
    flat = (1 / 3, 1 / 3, 1 / 3)
    for p, (em, eo, ih) in zip(PAIRS, gaps):
        elo = elo_only_result(model, em, eo, ih)
        fore = blend(elo, p.leg1_res)
        pers = LEG1_PRIOR[p.leg1_res]
        oi = RES_IDX[p.rem_res]
        rps_fore.append(rps(fore, oi))
        rps_flat.append(rps(flat, oi))
        rps_elo.append(rps(elo, oi))
        rps_pers.append(rps(pers, oi))
        rows.append((p, em, eo, ih, elo, fore, oi))
    md.append("### (b) Result -- per-pair Elo-only vs forecast vs actual\n")
    md.append("| pair | venue | Elo gap | Elo-only L/D/W | forecast L/D/W | leg-1 | actual |")
    md.append("|------|:-----:|--------:|----------------|----------------|:-----:|:------:|")
    for p, em, eo, ih, elo, fore, _ in rows:
        v = "H" if ih else "A"
        md.append(f"| {p.opp} | {v} | {em - eo:+.0f} | "
                  f"{elo[0]:.2f}/{elo[1]:.2f}/{elo[2]:.2f} | "
                  f"{fore[0]:.2f}/{fore[1]:.2f}/{fore[2]:.2f} | {p.leg1_res} | {p.rem_res} |")
    md.append("")
    md.append("**Mean RPS (lower better), 6 rematches:**\n")
    md.append("| forecaster | mean RPS |")
    md.append("|------------|---------:|")
    md.append(f"| forecast (0.5 Elo + 0.5 leg-1) = **primary** | **{np.mean(rps_fore):.4f}** |")
    md.append(f"| baseline (i) uninformative (flat 1/3) | {np.mean(rps_flat):.4f} |")
    md.append(f"| baseline (ii) Elo-only | {np.mean(rps_elo):.4f} |")
    md.append(f"| baseline (iii) persistence (leg-1 prior alone) | {np.mean(rps_pers):.4f} |")
    md.append("")


def attack_block(md: list[str]) -> None:
    """Score the attack-direction (more/less direct) LOPO-sign forecast vs a coin flip."""
    shifts = [p.rem_direct - p.leg1_direct for p in PAIRS]
    lopo = lopo_mean(shifts)                    # regime tendency from the other five, held out
    hits_lopo = 0
    hits_naive = 0                              # in-sample "Amorim more direct" (all predict MORE)
    md.append("### (c) Attack direction -- more/less direct than leg 1\n")
    md.append("| pair | leg-1 direct | rematch direct | actual | LOPO pred | hit | n3 leg1/rem |")
    md.append("|------|-------------:|---------------:|:------:|:---------:|:---:|:-----------:|")
    for p, sh, lp in zip(PAIRS, shifts, lopo):
        act = "more" if sh > 0 else "less"
        pred = "more" if lp > 0 else "less"
        hit = pred == act
        hits_lopo += hit
        hits_naive += (act == "more")
        md.append(f"| {p.opp} | {p.leg1_direct:.2f} | {p.rem_direct:.2f} | {act} | {pred} | "
                  f"{'Y' if hit else 'n'} | {p.leg1_n3}/{p.rem_n3} |")
    md.append("")
    md.append(f"LOPO regime-sign hit rate = **{hits_lopo}/6 = {hits_lopo / 6:.0%}** "
              f"(coin flip = 3/6). In-sample 'Amorim more direct' would score {hits_naive}/6, but "
              f"that reuses the pooled tendency these pairs define (circular). LOPO is the honest "
              f"number. Leg-1 3-way coverage is low (Tottenham leg-1 n3 = 3, degenerate 100% "
              f"direct); read as direction-only over tiny n.\n")


def main() -> None:
    """Fit the WP model, run all three forecasts, and write ``results/FORECAST_V0.md``."""
    print("[forecast_v0] fitting base-subset WP model (cached StatsBomb PL 15/16)...")
    model = build_model()
    md: list[str] = [
        "# FORECAST_V0 -- C5 pre-match forecast v0 (BTP B1.3 / REVIEW_CRIB Q6)",
        "",
        "Six same-opponent home/away pairs across the ten Hag -> Amorim change. Leg-1 = the "
        "Aug-Sep 2024 (ten Hag) meeting; rematch = the Jan-Feb 2025 (Amorim) meeting. We forecast "
        "each rematch from pre-match information only and score vs three baselines. **n = 6 = "
        "proof-of-signal, not significance; every pair changes both venue and manager between "
        "legs.**",
        "",
        PROTOCOL,
        "",
        "---",
        "",
    ]
    possession_block(md)
    md.append("---\n")
    result_block(md, model)
    md.append("---\n")
    attack_block(md)
    md.append("---\n")
    md.append("## Verdict\n")
    md.append(
        "**Does leg-1 broadcast analysis carry predictive information beyond Elo? For possession, "
        "YES (weakly): the broadcast poss-link share persists and beats both the uninformative and "
        "the Elo baselines. Result is best forecast by the last-meeting outcome (a public H2H "
        "prior, beating Elo), not by broadcast shape. Attack direction shows no signal that "
        "survives honest holdout. n=6 = proof-of-signal only.**\n")
    md.append(
        "- **Possession -- SIGNAL (weak but real).** Persisting the leg-1 poss-link share predicts "
        "the rematch with MAE ~7.9 pp, beating the uninformative 50% baseline (~10.6 pp) and the "
        "Elo-only fit; the 68%/90% intervals are roughly calibrated at n=6. This is the r=+0.83 "
        "fingerprint carrying into a genuine forecast. Elo alone does not reproduce it, so leg-1 "
        "broadcast possession carries information beyond Elo.")
    md.append(
        "- **Result -- the last meeting beats Elo, and the pre-registered blend was suboptimal.** "
        "Persistence of the leg-1 RESULT alone scores RPS 0.149, clearly beating Elo-only (0.245) "
        "and flat (0.250); the pre-registered 0.5-Elo/0.5-leg-1 blend (0.188) lands in between "
        "because its Elo half is the WEAKER signal here and drags pure persistence up. So the "
        "last-meeting outcome carries more forecast information than the strength prior for these "
        "6 (results are opponent-stable -- 4/6 keep the exact result). Caveat: this is a public "
        "head-to-head prior, not a broadcast-derived metric; the pre-registered equal weight was a "
        "wrong call in hindsight, honestly reported.")
    md.append(
        "- **Attack direction -- NO signal survives LOPO.** The honest leave-one-pair-out "
        "regime-sign forecast scores at chance (3/6); the 4/6 'Amorim more direct' hit rate is an "
        "in-sample artifact of the pooled tendency these same pairs define, and leg-1 3-way "
        "coverage is too low (n3 as small as 3) to forecast direction reliably.")
    md.append("")
    md.append("## What v1 needs\n")
    md.append(
        "- **More matches.** n=6 caps everything at proof-of-signal; the demo opponent's OTHER "
        "league games (opponent priors from THEIR matches, not just the one Utd meeting) would "
        "de-confound venue/manager and give real intervals.")
    md.append(
        "- **Opponent-conditioned possession prior** (their control profile vs the field), so the "
        "forecast is not pure Utd persistence.")
    md.append(
        "- **De-confound venue and manager** -- impossible in this corpus (every pair flips both); "
        "needs same-manager home/away repeats.")
    md.append(
        "- **Higher ball coverage for attack typing** -- 3-way coverage ~10-20% makes the direct "
        "share too noisy to forecast; this is upstream (ball track) work.")
    md.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"[forecast_v0] wrote {OUT}")


def _demo() -> None:
    """Self-check: RPS/blend/LOPO invariants (no network)."""
    assert rps((0.0, 0.0, 1.0), RES_IDX["W"]) == 0.0                  # perfect forecast
    assert abs(rps((1 / 3, 1 / 3, 1 / 3), 2) - 5 / 18) < 1e-9         # flat on a win: 0.5(1/9+4/9)
    # a loss forecast scored on a win is the worst case (RPS = 1).
    assert abs(rps((1.0, 0.0, 0.0), RES_IDX["W"]) - 1.0) < 1e-9
    b = blend((0.2, 0.3, 0.5), "W")
    assert abs(sum(b) - 1.0) < 1e-9 and b[2] > 0.5                    # win prior lifts P(win)
    assert lopo_mean([1.0, 2.0, 3.0]) == [2.5, 2.0, 1.5]
    # LOPO linear fit recovers an exact line on held-out points.
    pred = lopo_linfit_predict([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert all(abs(p - (1 + 2 * x)) < 1e-6 for p, x in zip(pred, [0, 1, 2, 3]))
    print("forecast_v0 self-check OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        main()
