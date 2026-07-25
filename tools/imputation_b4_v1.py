"""B4 model v1: quantile-GBM residual correction on the frozen B7 anchor -- train, freeze, gate.

Protocol (``docs/B4_MODEL_PLAN.md`` 4.1, frozen 2026-07-24):

    TRAIN   = Metrica Game 1 (full)        -> slot model, tau, B5 + B7 blend weights, GBM heads
    CALIB   = Metrica Game 2, first half   -> hyper-parameters, conformal k_b, skill horizon b*,
                                              SGR threshold R_max
    HOLDOUT = Metrica Game 2, second half  -> gates 1 and 2, run ONCE after the freeze print

The anchor and the gate-1 bar are BOTH ``B7 = veldecay (+) role-anchored-vote``; v1 predicts the
residual ``truth - B7_prediction`` in the attack-aligned frame, so a learner that finds nothing
ties the bar instead of losing to it.

Usage:
    python -m tools.imputation_b4_v1 [--sweep] [--no-holdout] [--out PATH]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from synthesizer.imputation import (
    BIN_LABELS,
    REPO,
    collect_samples,
    fit_blend,
    fit_decay_tau,
    fit_slot_model,
    _prep_game,
)
from synthesizer.imputation_features import (
    block_bootstrap,
    bucket_of,
    build_features,
    conformal_k,
    coverage_table,
    derive_fields,
)
from synthesizer.imputation_v1 import (
    QUANTILES,
    b7_position,
    b7_weights,
    boot_rmse_upper,
    emit,
    fit_heads,
    paired_rmse_ci,
    risk_coverage,
    sample_signs,
    sgr_threshold,
    skill_horizon,
    v1_prediction,
)

BLOCK_S = 60.0  # 1-minute block bootstrap (plan 4.2)
DELTA = 0.05  # SGR failure probability
ALPHAS = (0.5, 0.1)
# Pre-registered holdout bar (B7, Game 2 second half) copied from B4_IMPUTATION_PLAN.md.
B7_BAR_DOC = (0.66, 2.09, 4.37, 7.55, 14.91, 15.16)
B7_BAR_DOC_ALL = 11.46
# Provenance of that bar, recovered 2026-07-25 by sweeping the one free parameter: it is B7 with
# a vote EMA half-life of 0.04 s (one frame). ``VOTE_HALFLIFE_S`` read 10.0 s at the time (which
# reproduces 12.54 m ALL) and was corrected to 0.04 s on 2026-07-25; the two now agree. We anchor
# v1 on the bar's own configuration so anchor == bar, as the plan requires. See
# results/B4_MODEL_V1.md "Bar provenance" for the full sweep.
BAR_HALFLIFE_S = 0.04
SWEEP = (
    {"max_iter": 200, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 100,
     "l2_regularization": 1.0},
    {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 100,
     "l2_regularization": 1.0},
    {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 200,
     "l2_regularization": 1.0},
    {"max_iter": 200, "learning_rate": 0.10, "max_leaf_nodes": 15, "min_samples_leaf": 50,
     "l2_regularization": 5.0},
)
DEFAULT_PARAMS = SWEEP[1]


class Tee:
    """Print an ASCII line and keep it for the markdown report."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        """Print ``text`` and append it to the buffer."""
        print(text)
        self.lines.append(text)


def _rmse(err: np.ndarray) -> float:
    """Root-mean-square of an error array."""
    return float(np.sqrt(np.mean(err**2)))


def _err(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Euclidean per-sample error in metres."""
    return np.linalg.norm(pred - target, axis=1)


def _subset(d: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    """Row-select every array in a sample dict."""
    return {k: v[m] for k, v in d.items()}


def _blocks(samples: dict[str, np.ndarray], fps: float) -> np.ndarray:
    """1-minute block ids (period-aware) for the block bootstrap."""
    return samples["period"] * 100000 + (samples["frame"] / (BLOCK_S * fps)).astype(int)


def _episodes(samples: dict[str, np.ndarray]) -> np.ndarray:
    """Occlusion-episode id per sample (one episode = one continuous off-camera spell)."""
    return samples["slot_id"].astype(np.int64) * 10**7 + samples["last_frame"].astype(np.int64)


def _prep(name: str, half_w: float | None, coeffs=None, halflife: float = BAR_HALFLIFE_S):
    """Load + censor a game, derive all fields, and collect hidden samples with the B6 field."""
    g, half_w, summary = _prep_game(name, half_w=half_w)
    print(summary)
    if coeffs is None:
        coeffs = fit_slot_model(g["truth"], g["cent"], g["cam"], g["ranges"])
    fields = derive_fields(
        g["truth"], g["visible"], g["ball"], g["cam"], g["period"], g["fps"], g["ranges"], coeffs,
        halflife,
    )
    samples = collect_samples(
        g["truth"], g["visible"], g["period"], g["fps"], g["vel"],
        np.asarray(fields["slot"]), fields={"b6": np.asarray(fields["b6"])},
    )
    return g, half_w, coeffs, fields, samples


def _sweep(x_tr, r_tr, x_ca, anchor_ca, sgn_ca, target_ca, out: Tee) -> dict[str, object]:
    """Pick GBM hyper-parameters on TRAIN-fit / CALIB-score only (never on the holdout)."""
    out("\nHYPER-PARAMETER SWEEP (fit TRAIN, score CALIB; median heads only)")
    out("cfg | max_iter  lr    leaves  min_leaf  l2 | CALIB RMSE | fit s")
    best, best_rmse = DEFAULT_PARAMS, float("inf")
    for i, params in enumerate(SWEEP):
        t0 = time.time()
        heads = fit_heads(x_tr, r_tr, params, quantiles=(0.5,))
        mu, _ = v1_prediction(heads, x_ca, anchor_ca, sgn_ca)
        rm = _rmse(_err(mu, target_ca))
        out(
            f"{i:3d} | {params['max_iter']:8d} {params['learning_rate']:5.2f} "
            f"{params['max_leaf_nodes']:6d} {params['min_samples_leaf']:9d} "
            f"{params['l2_regularization']:4.1f} | {rm:10.3f} | {time.time() - t0:5.1f}"
        )
        if rm < best_rmse:
            best, best_rmse = params, rm
    out(f"chosen: {best} (CALIB RMSE {best_rmse:.3f} m)")
    return best


def _bucket_rows(err: np.ndarray, samples: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Per-bucket n / n_eps / RMSE / median error for an error array."""
    bkt = bucket_of(samples["tsls"])
    eps = _episodes(samples)
    rows = []
    for b in range(len(BIN_LABELS)):
        m = bkt == b
        rows.append(
            {
                "n": int(m.sum()),
                "n_eps": int(np.unique(eps[m]).size),
                "rmse": _rmse(err[m]) if m.any() else float("nan"),
                "p50": float(np.median(err[m])) if m.any() else float("nan"),
            }
        )
    return rows


def _gate1(
    v1_err: np.ndarray,
    b7_err: np.ndarray,
    samples: dict[str, np.ndarray],
    blocks: np.ndarray,
    out: Tee,
) -> list[str]:
    """Print the gate-1 table: v1 vs the B7 bar per bucket with paired block-bootstrap CIs."""
    rv, rb = _bucket_rows(v1_err, samples), _bucket_rows(b7_err, samples)
    orc = _bucket_rows(_err(samples["linear"], samples["target"]), samples)
    bkt = bucket_of(samples["tsls"])
    out("\nGATE 1 -- HOLDOUT (Game 2, second half). RMSE metres. Bar = B7 (frozen, training-free).")
    out(
        "horizon   |      n | n_eps |  B7 bar | doc bar |   v1   | diff (v1-B7) 95% block CI"
        " | v1 p50 | B7 p50 | oracle | verdict"
    )
    out("-" * 131)
    verdicts: list[str] = []
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            verdicts.append("n/a")
            continue
        lo, hi = paired_rmse_ci(b7_err[m], v1_err[m], blocks[m])
        verdict = "BEAT" if hi < 0 else ("LOSE" if lo > 0 else "tie")
        verdicts.append(verdict)
        out(
            f"{lab:9s} | {rv[b]['n']:6d} | {rv[b]['n_eps']:5d} | {rb[b]['rmse']:7.2f} | "
            f"{B7_BAR_DOC[b]:7.2f} | {rv[b]['rmse']:6.2f} | {rv[b]['rmse'] - rb[b]['rmse']:+7.2f} "
            f"[{lo:+6.2f}, {hi:+6.2f}] | {rv[b]['p50']:6.2f} | {rb[b]['p50']:6.2f} | "
            f"{orc[b]['rmse']:6.2f} | {verdict}"
        )
    lo, hi = paired_rmse_ci(b7_err, v1_err, blocks)
    verdict = "BEAT" if hi < 0 else ("LOSE" if lo > 0 else "tie")
    verdicts.append(verdict)
    oracle_all = _rmse(_err(samples["linear"], samples["target"]))
    out(
        f"{'ALL':9s} | {v1_err.size:6d} | {np.unique(_episodes(samples)).size:5d} | "
        f"{_rmse(b7_err):7.2f} | {B7_BAR_DOC_ALL:7.2f} | {_rmse(v1_err):6.2f} | "
        f"{_rmse(v1_err) - _rmse(b7_err):+7.2f} [{lo:+6.2f}, {hi:+6.2f}] | "
        f"{np.median(v1_err):6.2f} | {np.median(b7_err):6.2f} | {oracle_all:6.2f} | {verdict}"
    )
    out("diff < 0 means v1 beats the anchor; CI excluding 0 = significant at 95%.")
    out("n_eps = distinct occlusion episodes = the honest effective sample size (n is ~25 fps).")
    out("oracle = B2_offline linear interpolation, which CONSUMES the future re-sighting. It is a")
    out("ceiling reference, never the bar; v1 beating it in a bucket is a claim needing scrutiny.")
    return verdicts


def _window_diag(
    samples: dict[str, np.ndarray],
    mu: np.ndarray,
    anchor: np.ndarray,
    cam: np.ndarray,
    half_w: float,
    out: Tee,
) -> None:
    """Quantify how much of v1's edge is 'the hidden player must be outside the camera window'.

    Being hidden is itself information: the simulator censors exactly ``|x - cam_x| > half_w``,
    and ``ball_x`` (a feature) tracks ``cam_x``. A learner can exploit that; a hand-written
    extrapolator cannot. It is causally available in a real pipeline (an undetected player is
    presumed off-screen) -- but the HARD rectangle is an artefact of our simulator, so this is
    also the biggest transfer risk to real broadcast footage.

    Args:
        samples: Holdout samples.
        mu: v1 point predictions, shape ``(M, 2)``.
        anchor: B7 point predictions, shape ``(M, 2)``.
        cam: Per-frame camera centre track in metres.
        half_w: Censoring window half-width in metres.
        out: Report sink.
    """
    cx = cam[samples["frame"].astype(int), 0]

    def inside(pos: np.ndarray) -> float:
        """Fraction of positions falling inside the (impossible) visible window."""
        return float(np.mean(np.abs(pos[:, 0] - cx) <= half_w))

    out("\nCENSORING-GEOMETRY DIAGNOSTIC (disclosure, not a gate)")
    out(f"window half-width {half_w:.1f} m around the smoothed ball track.")
    tin = 100 * inside(samples["target"])
    out(f"  truth inside the window : {tin:5.1f}%  (0% by construction)")
    out(f"  B7 anchor inside        : {100 * inside(anchor):5.1f}%  <- impossible placements")
    out(f"  v1 inside               : {100 * inside(mu):5.1f}%")
    out(f"  last-seen inside        : {100 * inside(samples['hold']):5.1f}%")
    out("If v1 places far fewer predictions inside the (impossible) visible window than B7, part")
    out("of its edge is learnt censoring geometry, which may not transfer to real broadcast.")


def _gate2(
    resid: np.ndarray,
    w: np.ndarray,
    k: np.ndarray,
    samples: dict[str, np.ndarray],
    blocks: np.ndarray,
    out: Tee,
) -> list[dict[str, float]]:
    """Print the gate-2 conformal coverage table (PICP + MPIW) with block-bootstrap CIs."""
    bkt = bucket_of(samples["tsls"])
    rows = coverage_table(resid, bkt, w, k, ALPHAS)
    s = np.sqrt(((resid / w) ** 2).sum(axis=1))
    out("\nGATE 2 -- conformal coverage on HOLDOUT. Target: PICP within +/-5 pts of nominal.")
    out(
        "horizon   |      n | PICP50  (95% CI)      | PICP90  (95% CI)      |   r50    r90 "
        "| pass50 pass90"
    )
    out("-" * 106)
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            continue
        r = rows[b]
        ci50 = block_bootstrap((s[m] <= k[b, 0]).astype(float), blocks[m], stat="mean")
        ci90 = block_bootstrap((s[m] <= k[b, 1]).astype(float), blocks[m], stat="mean")
        p50, p90 = 100 * r["picp_50"], 100 * r["picp_90"]
        out(
            f"{lab:9s} | {r['n']:6d} | {p50:6.1f}  [{100 * ci50[0]:5.1f},{100 * ci50[1]:5.1f}] | "
            f"{p90:6.1f}  [{100 * ci90[0]:5.1f},{100 * ci90[1]:5.1f}] | {r['r_50']:5.1f}  "
            f"{r['r_90']:5.1f} | {'yes' if 45 <= p50 <= 55 else 'NO ':6s} "
            f"{'yes' if 85 <= p90 <= 95 else 'NO'}"
        )
    ok50 = sum(1 for r in rows if r["n"] and 45 <= 100 * r["picp_50"] <= 55)
    ok90 = sum(1 for r in rows if r["n"] and 85 <= 100 * r["picp_90"] <= 95)
    nb = sum(1 for r in rows if r["n"])
    out(f"gate 2: 50% -> {ok50}/{nb} buckets pass, 90% -> {ok90}/{nb} buckets pass")
    out("r50/r90 = mean equivalent region radius in metres (sqrt(area/pi)) = MPIW, per sample now.")
    return rows


def _risk_coverage(
    v1_err: np.ndarray, r90: np.ndarray, samples: dict[str, np.ndarray], out: Tee
) -> dict[str, float]:
    """Print per-bucket risk-coverage curves plus AURC and AUGRC."""
    bkt = bucket_of(samples["tsls"])
    grid = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    out("\nRISK-COVERAGE (accept smallest r90 first). Selective RMSE at each coverage level.")
    head = " ".join(f"cov{int(100 * c):>3d}" for c in grid)
    out(f"horizon   |      n | {head} |  AURC  AUGRC")
    out("-" * 92)

    def cells(rc: dict[str, np.ndarray | float]) -> str:
        """Selective RMSE at each coverage level of the grid."""
        n = np.asarray(rc["coverage"]).size
        return " ".join(
            f"{np.asarray(rc['sel_rmse'])[max(0, int(round(c * n)) - 1)]:6.2f}" for c in grid
        )

    areas: dict[str, float] = {}
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            continue
        rc = risk_coverage(v1_err[m], r90[m])
        out(f"{lab:9s} | {int(m.sum()):6d} | {cells(rc)} | {rc['aurc']:6.2f} {rc['augrc']:6.2f}")
        areas[f"augrc_{lab}"] = float(rc["augrc"])
    rc = risk_coverage(v1_err, r90)
    out(f"{'ALL':9s} | {v1_err.size:6d} | {cells(rc)} | {rc['aurc']:6.2f} {rc['augrc']:6.2f}")
    out("AURC = area under mean-error-vs-coverage (selective risk); AUGRC = area under the")
    out("GENERALIZED risk (loss summed over accepted / N total) -- penalises confident failures.")
    areas["aurc_all"], areas["augrc_all"] = rc["aurc"], rc["augrc"]
    return areas


def _worst(
    v1_err: np.ndarray, r90: np.ndarray, samples: dict[str, np.ndarray], out: Tee, top: int = 5
) -> None:
    """List the worst high-confidence failures per bucket (small r90, large error).

    De-duplicated to ONE row per occlusion episode: at 25 fps the five largest errors in a bucket
    are otherwise five consecutive frames of the same incident, which hides how many distinct
    silent failures there are.

    Args:
        v1_err: Per-sample v1 error (metres).
        r90: Calibrated 90% region radius per sample (metres).
        samples: Holdout samples.
        out: Report sink.
        top: Rows per bucket.
    """
    bkt = bucket_of(samples["tsls"])
    eps = _episodes(samples)
    out("\nWORST HIGH-CONFIDENCE FAILURES (r90 below the bucket median, largest error).")
    out("One row per occlusion EPISODE (worst frame of that episode), not per frame.")
    out("horizon   | rank | tsls s |  r90 m | error m | slot | frame | error/r90")
    out("-" * 82)
    for b, lab in enumerate(BIN_LABELS):
        m = np.flatnonzero(bkt == b)
        if m.size < 10:
            continue
        tight = m[r90[m] <= np.median(r90[m])]
        ranked = tight[np.argsort(-v1_err[tight])]
        _, first = np.unique(eps[ranked], return_index=True)
        order = ranked[np.sort(first)][:top]
        for rank, i in enumerate(order, 1):
            out(
                f"{lab:9s} | {rank:4d} | {samples['tsls'][i]:6.1f} | {r90[i]:6.1f} | "
                f"{v1_err[i]:7.1f} | {int(samples['slot_id'][i]):4d} | "
                f"{int(samples['frame'][i]):5d} | {v1_err[i] / max(r90[i], 1e-6):8.2f}"
            )


def main() -> None:
    """Run the frozen v1 protocol: fit, calibrate, freeze, then score the holdout once."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="run the CALIB hyper-parameter sweep")
    ap.add_argument("--no-holdout", action="store_true", help="stop after the freeze print")
    ap.add_argument(
        "--out",
        default="results/B4_MODEL_V1_runlog.md",
        help="machine-written run log; the hand-written analysis lives in results/B4_MODEL_V1.md",
    )
    ap.add_argument(
        "--vote-halflife", type=float, default=BAR_HALFLIFE_S,
        help="vote EMA half-life (s); the default is the value that reproduces the frozen bar",
    )
    args = ap.parse_args()
    out = Tee()
    hl = float(args.vote_halflife)

    out("=" * 100)
    out("B4 MODEL v1 -- quantile-GBM residual correction on the frozen B7 anchor")
    out("TRAIN = Game 1 (full) | CALIB = Game 2 H1 | HOLDOUT = Game 2 H2  [splits frozen 07-24]")
    out("=" * 100)
    out(f"vote EMA half-life = {hl:.2f}s (bar-provenance value; VOTE_HALFLIFE_S read 10.0s until")
    out("2026-07-25, which reproduces 12.54 m ALL instead of the frozen 11.46 m -- see report")
    out("section 'Bar provenance'. The constant now matches; anchor and bar are the same object.)")
    g1, half_w, coeffs, f1, s1 = _prep("Sample_Game_1", None, halflife=hl)
    tau, tau_rmse = fit_decay_tau(s1)
    w5 = fit_blend(s1, tau)
    w7 = b7_weights(s1, tau)
    out(f"window {2 * half_w:.1f} m x 68 m | tau={tau:.2f}s (TRAIN) | B5 weights "
        f"{np.round(w5, 2).tolist()} | B7 weights {np.round(w7, 2).tolist()}")

    g2, _, _, f2, s2 = _prep("Sample_Game_2", half_w, coeffs, halflife=hl)
    calib = _subset(s2, s2["period"] == 1)
    hold = _subset(s2, s2["period"] == 2)
    fps = g1["fps"]

    # ---- anchor, features, residual target -------------------------------------------------
    anc_tr = b7_position(s1, tau, w7)
    anc_ca = b7_position(calib, tau, w7)
    anc_ho = b7_position(hold, tau, w7)
    sgn_tr, sgn_ca, sgn_ho = sample_signs(f1, s1), sample_signs(f2, calib), sample_signs(f2, hold)
    x_tr = build_features(f1, s1, anc_tr)
    x_ca = build_features(f2, calib, anc_ca)
    x_ho = build_features(f2, hold, anc_ho)
    r_tr = (s1["target"] - anc_tr) * sgn_tr[:, None]
    out(f"\nTRAIN n={x_tr.shape[0]} | CALIB n={x_ca.shape[0]} | HOLDOUT n={x_ho.shape[0]} | "
        f"features={x_tr.shape[1]}")
    out(f"B7 anchor RMSE: TRAIN {_rmse(_err(anc_tr, s1['target'])):.2f} m | "
        f"CALIB {_rmse(_err(anc_ca, calib['target'])):.2f} m  (holdout not touched yet)")

    params = DEFAULT_PARAMS
    if args.sweep:
        params = _sweep(x_tr, r_tr, x_ca, anc_ca, sgn_ca, calib["target"], out)

    # ---- fit the 10 heads on TRAIN ---------------------------------------------------------
    t0 = time.time()
    heads = fit_heads(x_tr, r_tr, params)
    out(f"\nfitted {2 * len(QUANTILES)} quantile heads on TRAIN in {time.time() - t0:.1f} s")

    mu_ca, w_ca = v1_prediction(heads, x_ca, anc_ca, sgn_ca)
    err_v1_ca = _err(mu_ca, calib["target"])
    err_b7_ca = _err(anc_ca, calib["target"])
    blk_ca = _blocks(calib, fps)
    out(f"CALIB: v1 RMSE {_rmse(err_v1_ca):.2f} m vs B7 {_rmse(err_b7_ca):.2f} m "
        f"| mean |correction| = {np.linalg.norm(mu_ca - anc_ca, axis=1).mean():.2f} m")

    # ---- conformal k_b on CALIB ------------------------------------------------------------
    bkt_ca = bucket_of(calib["tsls"])
    resid_ca = mu_ca - calib["target"]
    k = conformal_k(resid_ca, bkt_ca, w_ca, ALPHAS)
    geo_ca = np.sqrt(w_ca[:, 0] * w_ca[:, 1])
    r90_ca = k[bkt_ca, 1] * geo_ca

    # ---- abstention layer A: skill horizon -------------------------------------------------
    b_star, rows_a = skill_horizon(err_v1_ca, err_b7_ca, bkt_ca, blk_ca)
    out("\nABSTENTION LAYER A -- skill horizon on CALIB (v1 vs B7, paired 95% block-boot CI)")
    out("horizon   |      n |   v1   |   B7   | diff 95% CI            | significantly better?")
    for b, row in enumerate(rows_a):
        if not row.get("n"):
            continue
        out(
            f"{BIN_LABELS[b]:9s} | {int(row['n']):6d} | {row['rmse_v1']:6.2f} | "
            f"{row['rmse_bar']:6.2f} | [{row['ci_lo']:+6.2f}, {row['ci_hi']:+6.2f}]        | "
            f"{'yes' if row['significant'] else 'NO'}"
        )
    star = "none (better in every bucket)" if b_star is None else BIN_LABELS[b_star]
    out(f"b* = {star}  -> buckets at and beyond b* are declared no-assert")
    sig = [b for b, r in enumerate(rows_a) if r.get("significant")]
    if b_star is not None and sig:
        out("PROTOCOL DEFECT, disclosed on CALIB before the holdout run: plan 3.3 assumes skill")
        out("decays MONOTONICALLY with horizon, so 'abstain at and beyond b*' is safe. Here the")
        out(f"only losing bucket is {BIN_LABELS[b_star]} and v1 wins "
            f"{[BIN_LABELS[b] for b in sig]}, so the literal")
        out("rule abstains everywhere. We apply it literally (pre-registration wins) and ALSO")
        out("report a clearly-labelled per-bucket variant that abstains only in losing buckets.")

    # ---- abstention layer B: SGR threshold -------------------------------------------------
    risk_bar = _rmse(err_b7_ca)
    r_max = sgr_threshold(err_v1_ca, r90_ca, blk_ca, risk_bar, delta=DELTA)
    acc = r90_ca <= r_max
    out("\nABSTENTION LAYER B -- SGR reject option, binary-searched on CALIB ONLY (delta=0.05)")
    out(f"risk target = B7 CALIB RMSE = {risk_bar:.2f} m (the anchor's own risk on this split)")
    out(f"R_max = {r_max:.2f} m  -> accepts {100 * acc.mean():.1f}% of CALIB samples, "
        f"selective RMSE {_rmse(err_v1_ca[acc]) if acc.any() else float('nan'):.2f} m "
        f"(95% upper bound {boot_rmse_upper(err_v1_ca, blk_ca, acc, DELTA):.2f} m)")
    out("NOTE: plan 3.3 says 'the smallest R_max such that ...'; taken literally that is R_max=0")
    out("(abstain everywhere), which is vacuous. We use the SGR binary search as published --")
    out("the LARGEST threshold whose 95% risk bound still meets the bar, i.e. maximal coverage.")

    out("\n" + "=" * 100)
    out("FROZEN CONFIG (nothing below this line was tuned on the holdout)")
    out("=" * 100)
    out(f"anchor          : B7 = veldecay(tau={tau:.2f}s) blended with B6_vote(halflife={hl:.2f}s),"
        f" weights {np.round(w7, 2).tolist()} (TRAIN)")
    out(f"target          : truth - B7, attack-aligned frame; TRAIN in-sample B3 RMSE "
        f"{tau_rmse:.2f} m")
    out(f"model           : HistGradientBoostingRegressor(loss='quantile') x 2 axes x "
        f"{len(QUANTILES)} quantiles {list(QUANTILES)}")
    out(f"hyper-params    : {params}")
    out(f"features        : {x_tr.shape[1]} columns, allowlist FEATURE_NAMES (blend_x/y = B7)")
    out(f"conformal k(50%): {np.round(k[:, 0], 3).tolist()}")
    out(f"conformal k(90%): {np.round(k[:, 1], 3).tolist()}")
    out(f"skill horizon b*: {'none' if b_star is None else BIN_LABELS[b_star]}")
    out(f"SGR R_max       : {r_max:.4f} m (delta={DELTA}, CALIB only)")
    out("=" * 100)

    if args.no_holdout:
        _write(out, args.out, params, tau, w7, k, b_star, r_max, [], {}, risk_bar)
        return

    # ---- HOLDOUT: run once -----------------------------------------------------------------
    mu_ho, w_ho = v1_prediction(heads, x_ho, anc_ho, sgn_ho)
    err_v1 = _err(mu_ho, hold["target"])
    err_b7 = _err(anc_ho, hold["target"])
    blk_ho = _blocks(hold, fps)
    verdicts = _gate1(err_v1, err_b7, hold, blk_ho, out)
    bkt_ho = bucket_of(hold["tsls"])
    _gate2(mu_ho - hold["target"], w_ho, k, hold, blk_ho, out)
    r90_ho = k[bkt_ho, 1] * np.sqrt(w_ho[:, 0] * w_ho[:, 1])
    areas = _risk_coverage(err_v1, r90_ho, hold, out)
    _worst(err_v1, r90_ho, hold, out)
    _window_diag(hold, mu_ho, anc_ho, g2["cam"], half_w, out)

    ems = emit(hold, mu_ho, r90_ho, bkt_ho, b_star, r_max)
    a = ems["asserted"]
    out(f"\nEMITTED (pre-registered rule): {int(a.sum())} of {a.size} holdout positions asserted "
        f"({100 * a.mean():.1f}%); {int((~a).sum())} abstained -> emitted as last-seen, "
        f"asserted=False.")
    if a.any():
        out(f"asserted-subset v1 RMSE {_rmse(err_v1[a]):.2f} m vs B7 on the same subset "
            f"{_rmse(err_b7[a]):.2f} m")
    sig = [b for b, r in enumerate(rows_a) if r.get("significant")]
    keep = np.isin(bkt_ho, sig) & (r90_ho <= r_max)
    out("\nVARIANT (NOT pre-registered, reported for transparency): abstain only in buckets where")
    out(f"v1 was not significantly better on CALIB. Asserts {100 * keep.mean():.1f}% of holdout; "
        f"asserted-subset v1 RMSE {_rmse(err_v1[keep]) if keep.any() else float('nan'):.2f} m vs "
        f"B7 {_rmse(err_b7[keep]) if keep.any() else float('nan'):.2f} m.")
    out("horizon   | asserted (pre-registered) | asserted (variant)")
    for b, lab in enumerate(BIN_LABELS):
        m = bkt_ho == b
        if m.any():
            out(f"{lab:9s} | {100 * a[m].mean():24.1f}% | {100 * keep[m].mean():17.1f}%")

    _write(out, args.out, params, tau, w7, k, b_star, r_max, verdicts, areas, risk_bar)


def _write(
    out: Tee,
    path: str,
    params: dict[str, object],
    tau: float,
    w7: np.ndarray,
    k: np.ndarray,
    b_star: int | None,
    r_max: float,
    verdicts: list[str],
    areas: dict[str, float],
    risk_bar: float,
) -> None:
    """Write the run log to ``results/B4_MODEL_V1.md`` with a header and a verdict stub."""
    dst = REPO / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    head = [
        "# B4 model v1 -- frozen holdout run",
        "",
        "Generated by `python -m tools.imputation_b4_v1 --sweep`. Protocol:",
        "`docs/B4_MODEL_PLAN.md` sections 2.1 / 3.1-3.3 / 4.1-4.3, with the 2026-07-24 ANCHOR",
        "UPDATE applied -- anchor AND gate-1 bar are both `B7 = veldecay (+) role-anchored-vote`.",
        "",
        "Frozen summary:",
        "",
        f"- anchor: B7, tau={tau:.2f} s, blend weights {np.round(w7, 2).tolist()} (TRAIN only)",
        f"- model: 10 quantile HistGBM heads, hyper-params `{params}`",
        f"- conformal k(90%) per bucket: {np.round(k[:, 1], 3).tolist()}",
        f"- skill horizon b* = {'none' if b_star is None else BIN_LABELS[b_star]}",
        f"- SGR R_max = {r_max:.4f} m (delta=0.05, CALIB risk target {risk_bar:.2f} m)",
        f"- gate-1 per-bucket verdict: {verdicts}",
        f"- AUGRC (all buckets) = {areas.get('augrc_all', float('nan')):.2f} m,"
        f" AURC = {areas.get('aurc_all', float('nan')):.2f} m",
        "",
        "Full run log below (verbatim console output).",
        "",
        "```",
    ]
    body = head + out.lines + ["```", ""]
    dst.write_text("\n".join(body), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
