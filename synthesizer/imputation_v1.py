"""B4 model v1: quantile-GBM residual correction on the frozen B7 anchor.

Implements ``docs/B4_MODEL_PLAN.md`` section 2.1 with the 2026-07-24 ANCHOR UPDATE applied:
the anchor is **B7 = veldecay (+) role-anchored-vote** (not ``B5_blend``), so v1's regression
target is ``truth - B7_prediction`` expressed in the attack-aligned frame. Ten small CPU models
(one per axis per quantile in ``{0.05, 0.25, 0.50, 0.75, 0.95}``) are fitted with sklearn's
pinball-loss ``HistGradientBoostingRegressor``; the quantile spread becomes the per-sample region
half-width, which a per-bucket split-conformal multiplier then calibrates.

Everything here is training/inference machinery only -- the split protocol, the frozen
hyper-parameters and the single holdout run live in ``tools/imputation_b4_v1.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from synthesizer.imputation import BIN_LABELS, blend_position, fit_blend
from synthesizer.imputation_features import WIDTH_FLOOR_M, align_vec

QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
Heads = list[list[HistGradientBoostingRegressor]]


# --------------------------------------------------------------------------------------
# the B7 anchor (frozen, training-free, causal)
# --------------------------------------------------------------------------------------


def b7_weights(train: dict[str, np.ndarray], tau: float) -> np.ndarray:
    """Fit B7's per-bucket blend weights (veldecay vs vote) on TRAIN only.

    Args:
        train: TRAIN samples from ``collect_samples`` (must carry the ``b6`` vote field).
        tau: The frozen velocity-decay constant, fitted on TRAIN.

    Returns:
        Per-bucket weight on the veldecay term, shape ``(len(BIN_LABELS),)``.
    """
    alt = dict(train)
    alt["slot"] = train["b6"]
    return fit_blend(alt, tau)


def b7_position(samples: dict[str, np.ndarray], tau: float, w7: np.ndarray) -> np.ndarray:
    """Return the frozen B7 point prediction for each sample, shape ``(M, 2)``.

    Args:
        samples: Samples carrying ``hold``, ``v0``, ``tsls`` and the ``b6`` vote field.
        tau: Frozen velocity-decay constant.
        w7: Frozen per-bucket blend weights from :func:`b7_weights`.

    Returns:
        Predicted positions in metres, shape ``(M, 2)``.
    """
    alt = dict(samples)
    alt["slot"] = samples["b6"]
    return blend_position(alt, tau, w7)


def sample_signs(fields: dict[str, object], samples: dict[str, np.ndarray]) -> np.ndarray:
    """Attack-direction sign (+1/-1) of each sample's own team at the query frame.

    Args:
        fields: Output of ``imputation_features.derive_fields``.
        samples: Output of ``imputation.collect_samples``.

    Returns:
        Sign array, shape ``(M,)``.
    """
    team_of = np.asarray(fields["team_of"])
    sign_frame = np.asarray(fields["sign_frame"])
    return sign_frame[samples["frame"].astype(int), team_of[samples["slot_id"].astype(int)]]


# --------------------------------------------------------------------------------------
# quantile heads
# --------------------------------------------------------------------------------------


def make_head(quantile: float, params: dict[str, object]) -> HistGradientBoostingRegressor:
    """Build one pinball-loss HistGBM head.

    Args:
        quantile: Target quantile in (0, 1).
        params: Hyper-parameters (``max_iter``, ``learning_rate``, ``max_leaf_nodes``,
            ``min_samples_leaf``, ``l2_regularization``).

    Returns:
        An unfitted regressor.
    """
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=0,
    )


def fit_heads(
    x: np.ndarray,
    resid: np.ndarray,
    params: dict[str, object],
    quantiles: tuple[float, ...] = QUANTILES,
) -> Heads:
    """Fit one head per axis per quantile on the attack-aligned residual.

    Args:
        x: Feature matrix, shape ``(M, n_features)``.
        resid: Attack-aligned residual ``truth - anchor``, shape ``(M, 2)``.
        params: Hyper-parameters passed to :func:`make_head`.
        quantiles: Quantile levels, ascending.

    Returns:
        Nested list ``heads[axis][quantile_index]``.
    """
    return [[make_head(q, params).fit(x, resid[:, ax]) for q in quantiles] for ax in range(2)]


def predict_heads(heads: Heads, x: np.ndarray) -> np.ndarray:
    """Predict every quantile of the residual, sorted to remove quantile crossing.

    Args:
        heads: Fitted heads from :func:`fit_heads`.
        x: Feature matrix, shape ``(M, n_features)``.

    Returns:
        Array of shape ``(M, 2, n_quantiles)``, ascending along the last axis.
    """
    out = np.stack([np.column_stack([m.predict(x) for m in ax_heads]) for ax_heads in heads], 1)
    return np.sort(out, axis=2)


def point_and_width(
    q: np.ndarray, floor: float = WIDTH_FLOOR_M
) -> tuple[np.ndarray, np.ndarray]:
    """Split the quantile cube into a median residual and a per-axis half-width.

    Args:
        q: Quantile predictions from :func:`predict_heads`, shape ``(M, 2, n_quantiles)``.
        floor: Minimum half-width in metres (plan 3.1 step 2).

    Returns:
        Tuple ``(mu_resid, w)`` both shape ``(M, 2)``: the median residual correction and the
        half-width ``(q_hi - q_lo)/2``.
    """
    mu = q[:, :, q.shape[2] // 2]  # middle head == 0.50 (also correct for a median-only sweep)
    w = np.maximum((q[:, :, -1] - q[:, :, 0]) / 2.0, floor)
    return mu, w


def v1_prediction(
    heads: Heads, x: np.ndarray, anchor: np.ndarray, sgn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Correct the anchor with the median residual head and return the region half-widths.

    The residual is learnt in the attack-aligned frame; ``align_vec`` is its own inverse (a
    180-degree rotation), so the same call maps the correction back to world coordinates.

    Args:
        heads: Fitted heads.
        x: Feature matrix.
        anchor: B7 point prediction in world metres, shape ``(M, 2)``.
        sgn: Attack signs from :func:`sample_signs`, shape ``(M,)``.

    Returns:
        Tuple ``(mu_world, w)``: corrected position in world metres and per-axis half-widths.
    """
    mu_resid, w = point_and_width(predict_heads(heads, x))
    return anchor + align_vec(mu_resid, sgn), w


# --------------------------------------------------------------------------------------
# block-bootstrap helpers (fast, exact: RMSE over resampled blocks = sqrt(sum sq / count))
# --------------------------------------------------------------------------------------


def _block_index(block: np.ndarray) -> tuple[np.ndarray, int]:
    """Map block ids to dense indices, returning ``(inverse, n_blocks)``."""
    uniq, inv = np.unique(block, return_inverse=True)
    return inv, uniq.size


def paired_rmse_ci(
    a: np.ndarray,
    c: np.ndarray,
    block: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
    pcts: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Block-bootstrap CI for ``RMSE(c) - RMSE(a)``, blocks paired across both methods.

    Args:
        a: Per-sample error of the reference method (metres).
        c: Per-sample error of the challenger, aligned with ``a``.
        block: 1-minute block id per sample.
        n_boot: Bootstrap replicates.
        seed: RNG seed.
        pcts: Percentiles of the replicate distribution to return.

    Returns:
        Tuple ``(lo, hi)``. ``hi < 0`` means the challenger is significantly better.
    """
    inv, nb = _block_index(block)
    sa = np.bincount(inv, weights=a**2, minlength=nb)
    sc = np.bincount(inv, weights=c**2, minlength=nb)
    cnt = np.bincount(inv, minlength=nb).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nb, (n_boot, nb))
    den = np.maximum(cnt[pick].sum(axis=1), 1.0)
    reps = np.sqrt(sc[pick].sum(axis=1) / den) - np.sqrt(sa[pick].sum(axis=1) / den)
    return float(np.percentile(reps, pcts[0])), float(np.percentile(reps, pcts[1]))


def boot_rmse_upper(
    err: np.ndarray,
    block: np.ndarray,
    mask: np.ndarray,
    delta: float = 0.05,
    n_boot: int = 400,
    seed: int = 0,
) -> float:
    """One-sided ``1-delta`` block-bootstrap upper bound on the selective RMSE.

    Args:
        err: Per-sample error (metres) over the whole split.
        block: 1-minute block id per sample.
        mask: Acceptance mask; RMSE is taken over accepted samples only.
        delta: Failure probability (0.05 -> 95% confidence).
        n_boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        The ``100*(1-delta)`` percentile of the resampled selective RMSE, or ``inf`` if nothing
        is accepted.
    """
    if not mask.any():
        return float("inf")
    inv, nb = _block_index(block)
    sq = np.bincount(inv, weights=np.where(mask, err**2, 0.0), minlength=nb)
    cnt = np.bincount(inv, weights=mask.astype(float), minlength=nb)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nb, (n_boot, nb))
    den = cnt[pick].sum(axis=1)
    reps = np.sqrt(sq[pick].sum(axis=1) / np.where(den <= 0, np.nan, den))
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return float("inf")
    return float(np.percentile(reps, 100.0 * (1.0 - delta)))


# --------------------------------------------------------------------------------------
# abstention: layer A (skill horizon) and layer B (SGR reject option)
# --------------------------------------------------------------------------------------


def skill_horizon(
    v1_err: np.ndarray,
    bar_err: np.ndarray,
    bucket: np.ndarray,
    block: np.ndarray,
    n_boot: int = 400,
) -> tuple[int | None, list[dict[str, float]]]:
    """Layer A: first bucket where v1 is NOT significantly below the anchor (plan 3.3).

    Args:
        v1_err: Per-sample v1 error on the CALIBRATION split (metres).
        bar_err: Per-sample anchor error on the same split.
        bucket: Horizon-bucket index per sample.
        block: 1-minute block id per sample.
        n_boot: Bootstrap replicates.

    Returns:
        Tuple ``(b_star, rows)`` where ``b_star`` is the first failing bucket index (``None`` if
        v1 wins everywhere) and ``rows`` holds the per-bucket diagnostic.
    """
    rows: list[dict[str, float]] = []
    b_star: int | None = None
    for b in range(len(BIN_LABELS)):
        m = bucket == b
        if not m.any():
            rows.append({"bucket": b, "n": 0})
            continue
        lo, hi = paired_rmse_ci(bar_err[m], v1_err[m], block[m], n_boot=n_boot)
        wins = hi < 0.0
        rows.append(
            {
                "bucket": b,
                "n": int(m.sum()),
                "rmse_v1": float(np.sqrt(np.mean(v1_err[m] ** 2))),
                "rmse_bar": float(np.sqrt(np.mean(bar_err[m] ** 2))),
                "ci_lo": lo,
                "ci_hi": hi,
                "significant": bool(wins),
            }
        )
        if not wins and b_star is None:
            b_star = b
    return b_star, rows


def sgr_threshold(
    err: np.ndarray,
    conf: np.ndarray,
    block: np.ndarray,
    risk_bar: float,
    delta: float = 0.05,
    n_boot: int = 400,
    steps: int = 30,
    seed: int = 0,
) -> float:
    """Layer B: binary-search the largest confidence threshold meeting a risk guarantee.

    Selection-with-Guaranteed-Risk (arXiv 1705.08500): accept ``conf <= R``, keep the largest
    ``R`` whose ``1-delta`` upper bound on selective RMSE still sits at or below ``risk_bar``.
    Coverage is monotone in ``R``, so the binary search is well posed.

    Args:
        err: Per-sample error on the CALIBRATION split (metres).
        conf: Per-sample confidence score (region radius ``r90``; smaller = more confident).
        block: 1-minute block id per sample.
        risk_bar: Target selective RMSE (metres).
        delta: Failure probability.
        n_boot: Bootstrap replicates per probe.
        steps: Binary-search iterations.
        seed: RNG seed.

    Returns:
        The frozen threshold ``R_max`` in metres; ``0.0`` if even the tightest sample fails.
    """
    lo, hi = 0.0, float(np.max(conf)) * 1.001
    if boot_rmse_upper(err, block, conf <= hi, delta, n_boot, seed) <= risk_bar:
        return hi
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if boot_rmse_upper(err, block, conf <= mid, delta, n_boot, seed) <= risk_bar:
            lo = mid
        else:
            hi = mid
    return lo


def emit(
    samples: dict[str, np.ndarray],
    mu_world: np.ndarray,
    r90: np.ndarray,
    bucket: np.ndarray,
    b_star: int | None,
    r_max: float,
) -> dict[str, np.ndarray]:
    """Apply both abstention layers and emit positions with an explicit ``asserted`` flag.

    On abstain the emitted coordinate is the LAST-SEEN position, flagged ``asserted=False`` --
    never a silent fallback to hold-last (plan 3.3).

    Args:
        samples: Samples carrying ``hold`` (last-seen position).
        mu_world: v1 point prediction in world metres, shape ``(M, 2)``.
        r90: Calibrated 90% region radius per sample (metres).
        bucket: Horizon-bucket index per sample.
        b_star: Skill horizon from :func:`skill_horizon` (``None`` = no bucket-level abstention).
        r_max: Frozen SGR threshold.

    Returns:
        Dict with ``pos`` ``(M, 2)``, ``asserted`` ``(M,)`` bool and ``r90`` ``(M,)``.
    """
    ok = r90 <= r_max
    if b_star is not None:
        ok = ok & (bucket < b_star)
    return {"pos": np.where(ok[:, None], mu_world, samples["hold"]), "asserted": ok, "r90": r90}


# --------------------------------------------------------------------------------------
# risk-coverage reporting
# --------------------------------------------------------------------------------------


def risk_coverage(err: np.ndarray, conf: np.ndarray) -> dict[str, np.ndarray | float]:
    """Risk-coverage curve plus AURC and AUGRC (arXiv 2407.01032).

    The generalized risk ``sum(loss over accepted) / n_total`` (AUGRC) is reported alongside the
    classical selective risk ``sum(loss over accepted) / n_accepted`` (AURC) because AURC hides
    how errors distribute and does not penalise confident silent failures.

    Args:
        err: Per-sample loss (error in metres).
        conf: Per-sample confidence score; smaller = more confident (accepted first).

    Returns:
        Dict with ``coverage``, ``sel_risk``, ``sel_rmse``, ``gen_risk`` arrays plus scalar
        ``aurc``/``augrc``. ``sel_rmse`` is the gate metric (RMSE on the accepted subset); the
        two areas use mean absolute error, the convention of the AUGRC paper.
    """
    order = np.argsort(conf, kind="stable")
    e = err[order]
    n = e.size
    k = np.arange(1, n + 1, dtype=float)
    cum = np.cumsum(e)
    sel = cum / k
    gen = cum / n
    return {
        "coverage": k / n,
        "sel_risk": sel,
        "sel_rmse": np.sqrt(np.cumsum(e**2) / k),
        "gen_risk": gen,
        "aurc": float(sel.mean()),
        "augrc": float(gen.mean()),
    }


def _self_check() -> None:
    """Assert the residual round-trip, the SGR search and the risk-coverage maths on toy data."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 3))
    aligned = np.column_stack([x[:, 0] * 3.0, x[:, 1] * 2.0]) + rng.normal(scale=0.3, size=(n, 2))
    anchor = np.zeros((n, 2))
    sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    truth = anchor + align_vec(aligned, sgn)  # world frame; the residual is learnable only
    resid = align_vec(truth - anchor, sgn)  # after de-rotation -> must equal ``aligned``
    assert np.allclose(resid, aligned)
    params = {
        "max_iter": 60, "learning_rate": 0.2, "max_leaf_nodes": 15,
        "min_samples_leaf": 20, "l2_regularization": 1.0,
    }
    heads = fit_heads(x, resid, params)
    mu, w = v1_prediction(heads, x, anchor, sgn)
    err = np.linalg.norm(mu - truth, axis=1)
    hold_err = np.linalg.norm(anchor - truth, axis=1)
    assert err.mean() < 0.3 * hold_err.mean(), (err.mean(), hold_err.mean())
    assert (w > 0).all()
    # SGR: accepting only tight samples must satisfy a bar the full set violates.
    block = np.arange(n) // 100
    conf = err + rng.normal(scale=0.05, size=n)
    bar = 0.8 * float(np.sqrt(np.mean(err**2)))
    r_max = sgr_threshold(err, conf, block, bar, n_boot=100, steps=20)
    assert 0.0 < r_max < conf.max(), r_max
    assert boot_rmse_upper(err, block, conf <= r_max, n_boot=100) <= bar * 1.001
    # Risk-coverage: a perfect confidence score must beat a random one on both areas.
    rc_good = risk_coverage(err, err)
    rc_rand = risk_coverage(err, rng.random(n))
    assert rc_good["augrc"] < rc_rand["augrc"] < rc_good["augrc"] * 10
    assert rc_good["aurc"] < rc_rand["aurc"]
    assert abs(rc_rand["gen_risk"][-1] - err.mean()) < 1e-9
    print("imputation_v1 self-check OK (residual round-trip, SGR guarantee, AUGRC ordering)")


if __name__ == "__main__":
    _self_check()
