"""B4 abstention policy: score P0 / P1 / P2 on the FROZEN v1 holdout predictions.

No model is refitted for the policy comparison: the frozen protocol of
``tools/imputation_b4_v1.py`` (same splits, same ``FROZEN_PARAMS``, ``random_state=0``) is replayed
once to regenerate the deterministic predictions, cached to ``data/imputation/cache/`` (gitignored),
and every policy is then a mask over that single frozen array. Gate-1 and gate-2 numbers are
re-printed as a reproduction check and must match ``results/B4_MODEL_V1.md`` exactly.

Policies compared (see ``results/B4_ABSTENTION_POLICY.md``):

    P0  literal pre-registered  -- abstain at and beyond b*  (b* = 0-1s -> abstains on everything)
    P1  per-bucket abstain      -- abstain in buckets where v1 does not significantly beat B7
    P2  defer-to-anchor         -- emit B7's own prediction in those buckets, labelled anchor

Usage:
    python -m tools.imputation_b4_abstention [--force] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synthesizer.imputation import BIN_LABELS, REPO, fit_decay_tau
from synthesizer.imputation_features import bucket_of, build_features, conformal_k
from synthesizer.imputation_v1 import (
    b7_position,
    b7_weights,
    boot_rmse_upper,
    emit,
    fit_heads,
    paired_rmse_ci,
    sample_signs,
    sgr_threshold,
    skill_horizon,
    v1_prediction,
)
from tools.imputation_b4_v1 import (
    ALPHAS,
    DELTA,
    SWEEP,
    Tee,
    _blocks,
    _err,
    _prep,
    _rmse,
    _subset,
)

FROZEN_PARAMS = SWEEP[2]  # the config the --sweep run chose on CALIB (7.011 m); frozen 2026-07-25
CACHE = REPO / "data" / "imputation" / "cache" / "b4_v1_frozen_preds.npz"
R_GRID = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.4339)
RISK_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.92)


# --------------------------------------------------------------------------------------
# frozen-prediction cache
# --------------------------------------------------------------------------------------


def build_cache(path: Path) -> dict[str, np.ndarray]:
    """Replay the frozen v1 protocol and cache CALIB + HOLDOUT predictions.

    Args:
        path: Destination ``.npz``.

    Returns:
        The cached arrays.
    """
    g1, half_w, coeffs, f1, s1 = _prep("Sample_Game_1", None)
    tau, _ = fit_decay_tau(s1)
    w7 = b7_weights(s1, tau)
    _g2, _, _, f2, s2 = _prep("Sample_Game_2", half_w, coeffs)
    calib = _subset(s2, s2["period"] == 1)
    hold = _subset(s2, s2["period"] == 2)

    anc_tr, anc_ca, anc_ho = (b7_position(s, tau, w7) for s in (s1, calib, hold))
    sgn_tr, sgn_ca, sgn_ho = sample_signs(f1, s1), sample_signs(f2, calib), sample_signs(f2, hold)
    x_tr = build_features(f1, s1, anc_tr)
    heads = fit_heads(x_tr, (s1["target"] - anc_tr) * sgn_tr[:, None], FROZEN_PARAMS)

    mu_ca, w_ca = v1_prediction(heads, build_features(f2, calib, anc_ca), anc_ca, sgn_ca)
    mu_ho, w_ho = v1_prediction(heads, build_features(f2, hold, anc_ho), anc_ho, sgn_ho)
    bkt_ca, bkt_ho = bucket_of(calib["tsls"]), bucket_of(hold["tsls"])
    k = conformal_k(mu_ca - calib["target"], bkt_ca, w_ca, ALPHAS)
    fps = g1["fps"]

    d = {
        "err_v1_ca": _err(mu_ca, calib["target"]),
        "err_b7_ca": _err(anc_ca, calib["target"]),
        "resid_v1_ca": mu_ca - calib["target"],
        "resid_b7_ca": anc_ca - calib["target"],
        "w_ca": w_ca,
        "bkt_ca": bkt_ca,
        "blk_ca": _blocks(calib, fps),
        "r90_ca": k[bkt_ca, 1] * np.sqrt(w_ca[:, 0] * w_ca[:, 1]),
        "target": hold["target"],
        "lastseen": hold["hold"],
        "mu": mu_ho,
        "anchor": anc_ho,
        "w": w_ho,
        "tsls": hold["tsls"],
        "bucket": bkt_ho,
        "block": _blocks(hold, fps),
        "slot_id": hold["slot_id"],
        "frame": hold["frame"],
        "k": k,
        "tau": np.array([tau]),
    }
    d["err_v1"] = _err(mu_ho, hold["target"])
    d["err_b7"] = _err(anc_ho, hold["target"])
    d["r90"] = k[bkt_ho, 1] * np.sqrt(w_ho[:, 0] * w_ho[:, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


def load_cache(force: bool = False) -> dict[str, np.ndarray]:
    """Load the frozen predictions, rebuilding them if absent or ``force``.

    Args:
        force: Rebuild even if the cache exists.

    Returns:
        Cached arrays.
    """
    if force or not CACHE.exists():
        print(f"building frozen-prediction cache (~15 min CPU) -> {CACHE}")
        return build_cache(CACHE)
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------
# policy machinery
# --------------------------------------------------------------------------------------


def _picp(resid: np.ndarray, w: np.ndarray, kk: np.ndarray) -> tuple[float, float]:
    """Empirical 50% / 90% coverage of the conformal regions on a subset.

    Args:
        resid: Signed residuals ``emitted - truth``, shape ``(m, 2)``.
        w: Per-sample half-widths, shape ``(m, 2)``.
        kk: Per-sample conformal multipliers, shape ``(m, 2)`` (columns = 50% / 90%).

    Returns:
        Tuple ``(picp_50, picp_90)``.
    """
    if resid.shape[0] == 0:
        return float("nan"), float("nan")
    s = np.sqrt(((resid / w) ** 2).sum(axis=1))
    return float(np.mean(s <= kk[:, 0])), float(np.mean(s <= kk[:, 1]))


def policy_points(
    d: dict[str, np.ndarray], sig: np.ndarray, defer: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Emitted point per sample and its source code (0 = v1, 1 = anchor).

    Args:
        d: Cached frozen predictions.
        sig: Bucket indices where v1 significantly beat the anchor on CALIB.
        defer: If ``True`` emit the anchor in non-``sig`` buckets (P2); else always v1.

    Returns:
        Tuple ``(point, src)`` with ``point`` shape ``(M, 2)``.
    """
    if not defer:
        return d["mu"], np.zeros(d["mu"].shape[0], dtype=int)
    use_anchor = ~np.isin(d["bucket"], sig)
    return np.where(use_anchor[:, None], d["anchor"], d["mu"]), use_anchor.astype(int)


def report_policy(
    name: str,
    mask: np.ndarray,
    point: np.ndarray,
    d: dict[str, np.ndarray],
    out: Tee,
    kk: np.ndarray | None = None,
) -> dict[str, float]:
    """Print the per-bucket + overall scorecard of one policy on the asserted subset.

    Args:
        name: Policy label.
        mask: Asserted mask.
        point: Emitted point per sample, shape ``(M, 2)``.
        d: Cached frozen predictions.
        out: Report sink.
        kk: Per-sample conformal multipliers, shape ``(M, 2)``; defaults to the frozen v1 table.

    Returns:
        Overall summary dict.
    """
    err = np.linalg.norm(point - d["target"], axis=1)
    resid = point - d["target"]
    bucket, w, k = d["bucket"], d["w"], d["k"]
    kk = k[bucket] if kk is None else kk
    out(f"\n{name}")
    out(
        "horizon   |      n | asserted | emitted RMSE |  p50 | anchor RMSE | anchor p50 |"
        " PICP50 PICP90"
    )
    out("-" * 100)
    for b, lab in enumerate(BIN_LABELS):
        inb = bucket == b
        m = inb & mask
        cov = 100.0 * (m.sum() / max(inb.sum(), 1))
        if not m.any():
            out(f"{lab:9s} | {int(inb.sum()):6d} | {cov:7.1f}% | {'--':>12s} | {'--':>4s} |"
                f" {'--':>11s} | {'--':>10s} |    --     --")
            continue
        p50, p90 = _picp(resid[m], w[m], kk[m])
        out(
            f"{lab:9s} | {int(inb.sum()):6d} | {cov:7.1f}% | {_rmse(err[m]):12.2f} |"
            f" {np.median(err[m]):4.2f} | {_rmse(d['err_b7'][m]):11.2f} |"
            f" {np.median(d['err_b7'][m]):10.2f} | {100 * p50:6.1f} {100 * p90:6.1f}"
        )
    if not mask.any():
        out(f"{'ALL':9s} | {mask.size:6d} | {0.0:7.1f}% | nothing asserted -- degenerate policy")
        return {"coverage": 0.0, "rmse": float("nan"), "p50": float("nan")}
    p50, p90 = _picp(resid[mask], w[mask], kk[mask])
    out(
        f"{'ALL':9s} | {mask.size:6d} | {100 * mask.mean():7.1f}% | {_rmse(err[mask]):12.2f} |"
        f" {np.median(err[mask]):4.2f} | {_rmse(d['err_b7'][mask]):11.2f} |"
        f" {np.median(d['err_b7'][mask]):10.2f} | {100 * p50:6.1f} {100 * p90:6.1f}"
    )
    return {
        "coverage": float(mask.mean()),
        "rmse": _rmse(err[mask]),
        "p50": float(np.median(err[mask])),
        "picp50": p50,
        "picp90": p90,
    }


def layer_b_analysis(d: dict[str, np.ndarray], r_max: float, out: Tee) -> None:
    """Investigate whether layer B can ever fire, and what would make it active.

    Args:
        d: Cached frozen predictions.
        r_max: The frozen SGR threshold.
        out: Report sink.
    """
    r90, bucket = d["r90"], d["bucket"]
    out("\n" + "=" * 100)
    out("LAYER B DIAGNOSIS -- is there any working 'I don't know'?")
    out("=" * 100)
    out(f"frozen R_max = {r_max:.4f} m; max r90 on CALIB = {d['r90_ca'].max():.2f} m "
        f"(R_max is that max x 1.001 -> accepts 100.0% of CALIB by construction)")
    out(f"HOLDOUT max r90 = {r90.max():.2f} m; rejected by the frozen R_max: "
        f"{100 * np.mean(r90 > r_max):.2f}% of holdout ({int((r90 > r_max).sum())} rows)")
    out("\nr90 distribution per bucket (metres), HOLDOUT")
    out("horizon   |      n |   p50 |   p90 |   p99 |   max | %>10m | %>15m | %>20m | %>R_max")
    out("-" * 100)
    for b, lab in enumerate(BIN_LABELS):
        m = bucket == b
        if not m.any():
            continue
        r = r90[m]
        out(
            f"{lab:9s} | {int(m.sum()):6d} | {np.percentile(r, 50):5.1f} |"
            f" {np.percentile(r, 90):5.1f} | {np.percentile(r, 99):5.1f} | {r.max():5.1f} |"
            f" {100 * np.mean(r > 10):5.1f} | {100 * np.mean(r > 15):5.1f} |"
            f" {100 * np.mean(r > 20):5.1f} | {100 * np.mean(r > r_max):7.2f}"
        )
    out("\nWHERE ABSTENTION WOULD BITE: accept r90 <= R, HOLDOUT (diagnostic, not a new threshold)")
    out("   R m | coverage | sel RMSE | cov 10-30s | cov 30s+ | rejected n")
    out("-" * 70)
    for r in R_GRID:
        acc = r90 <= r
        c1030 = 100 * np.mean(acc[bucket == 4]) if (bucket == 4).any() else float("nan")
        c30 = 100 * np.mean(acc[bucket == 5]) if (bucket == 5).any() else float("nan")
        sel = _rmse(d["err_v1"][acc]) if acc.any() else float("nan")
        out(f"{r:6.1f} | {100 * acc.mean():7.1f}% | {sel:8.2f} | {c1030:9.1f}% | {c30:8.1f}% |"
            f" {int((~acc).sum()):10d}")
    out("\nRISK-TARGET SWEEP -- R_max re-derived on CALIB ONLY for a target stated in metres.")
    out("(SGR binary search, delta=0.05, exactly as the frozen protocol; HOLDOUT columns are a")
    out("disclosed diagnostic -- no threshold may be chosen from them.)")
    out("target m |   R_max m | CALIB cov | CALIB selRMSE (95% UB) | HOLD cov | HOLD selRMSE")
    out("-" * 100)
    for bar in RISK_GRID:
        rm = sgr_threshold(d["err_v1_ca"], d["r90_ca"], d["blk_ca"], bar, delta=DELTA)
        acc_ca = d["r90_ca"] <= rm
        acc_ho = r90 <= rm
        ub = boot_rmse_upper(d["err_v1_ca"], d["blk_ca"], acc_ca, DELTA)
        sel_ca = _rmse(d["err_v1_ca"][acc_ca]) if acc_ca.any() else float("nan")
        sel_ho = _rmse(d["err_v1"][acc_ho]) if acc_ho.any() else float("nan")
        out(
            f"{bar:8.2f} | {rm:9.3f} | {100 * acc_ca.mean():8.1f}% |"
            f" {sel_ca:9.2f} ({ub:5.2f})       | {100 * acc_ho.mean():7.1f}% | {sel_ho:12.2f}"
        )


def main() -> None:
    """Score the three abstention policies on the frozen holdout and diagnose layer B."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild the frozen-prediction cache")
    ap.add_argument("--out", default="results/B4_ABSTENTION_runlog.md", help="run log path")
    args = ap.parse_args()
    out = Tee()
    d = load_cache(args.force)

    b_star, rows = skill_horizon(d["err_v1_ca"], d["err_b7_ca"], d["bkt_ca"], d["blk_ca"])
    sig = np.array([b for b, r in enumerate(rows) if r.get("significant")], dtype=int)
    risk_bar = _rmse(d["err_b7_ca"])
    r_max = sgr_threshold(d["err_v1_ca"], d["r90_ca"], d["blk_ca"], risk_bar, delta=DELTA)

    out("=" * 100)
    out("B4 ABSTENTION POLICY -- P0 / P1 / P2 on the FROZEN v1 holdout predictions (no refit)")
    out("=" * 100)
    out(f"reproduction check: holdout n={d['err_v1'].size}, v1 ALL RMSE {_rmse(d['err_v1']):.2f} m,"
        f" B7 bar {_rmse(d['err_b7']):.2f} m  (frozen record: 8.10 vs 11.46)")
    out(f"CALIB: v1 {_rmse(d['err_v1_ca']):.2f} m vs B7 {risk_bar:.2f} m | "
        f"b* = {BIN_LABELS[b_star] if b_star is not None else 'none'} | "
        f"skill buckets {[BIN_LABELS[b] for b in sig]} | R_max = {r_max:.4f} m")

    layer_b = d["r90"] <= r_max
    bucket = d["bucket"]
    masks = {
        "P0 literal (pre-registered): abstain at and beyond b*": (
            layer_b & (bucket < (b_star if b_star is not None else len(BIN_LABELS))),
            False,
        ),
        "P1 per-bucket abstain: assert v1 only where it beats the anchor": (
            layer_b & np.isin(bucket, sig),
            False,
        ),
        "P2 defer-to-anchor: emit the anchor where v1 has no edge (layer B only)": (
            layer_b,
            True,
        ),
    }
    summ: dict[str, dict[str, float]] = {}
    for name, (mask, defer) in masks.items():
        point, _src = policy_points(d, sig, defer)
        summ[name] = report_policy(name, mask, point, d, out)
    point_all = d["mu"]
    summ["unrestricted v1 (no policy at all)"] = report_policy(
        "unrestricted v1 (no policy at all)", np.ones(bucket.size, dtype=bool), point_all, d, out
    )

    # ---- P2 emits the anchor's point, so the region must be the ANCHOR's (CALIB-derived) -----
    k_anc = conformal_k(d["resid_b7_ca"], d["bkt_ca"], d["w_ca"], ALPHAS)
    defer = ~np.isin(bucket, sig)
    kk = np.where(defer[:, None], k_anc[bucket], d["k"][bucket])
    out("\nCONFORMAL MULTIPLIERS (CALIB only): v1-calibrated vs anchor-calibrated, per bucket")
    out("horizon   | k50 v1 | k50 anc | k90 v1 | k90 anc | emitted by P2")
    for b, lab in enumerate(BIN_LABELS):
        out(f"{lab:9s} | {d['k'][b, 0]:6.3f} | {k_anc[b, 0]:7.3f} | {d['k'][b, 1]:6.3f} | "
            f"{k_anc[b, 1]:7.3f} | {'anchor' if b not in sig else 'v1'}")
    summ["P2b"] = report_policy(
        "P2b defer-to-anchor WITH the anchor's own CALIB-calibrated region (adopted)",
        layer_b, policy_points(d, sig, True)[0], d, out, kk=kk,
    )

    # ---- the decision question: does deferring cost accuracy? -------------------------------
    out("\n" + "=" * 100)
    out("DOES P2 COST ACCURACY? (paired 1-minute block bootstrap, 400 reps)")
    out("=" * 100)
    p1_mask = layer_b & np.isin(bucket, sig)
    p2_point, _ = policy_points(d, sig, True)
    err_p2 = np.linalg.norm(p2_point - d["target"], axis=1)
    same = np.array_equal(p2_point[p1_mask], d["mu"][p1_mask])
    out(f"P1 and P2 emit IDENTICAL numbers on P1's asserted subset: {same} "
        f"(n={int(p1_mask.sum())}) -> zero accuracy difference where both speak.")
    lo, hi = paired_rmse_ci(d["err_v1"], err_p2, d["block"])
    out(f"P2 vs unrestricted v1, ALL {int(bucket.size)} rows: {_rmse(err_p2):.4f} m vs "
        f"{_rmse(d['err_v1']):.4f} m, diff {_rmse(err_p2) - _rmse(d['err_v1']):+.4f} m "
        f"[{lo:+.4f}, {hi:+.4f}]")
    for b in range(len(BIN_LABELS)):
        if b in sig:
            continue
        m = bucket == b
        if not m.any():
            continue
        lo, hi = paired_rmse_ci(d["err_v1"][m], d["err_b7"][m], d["block"][m])
        out(f"defer bucket {BIN_LABELS[b]}: anchor {_rmse(d['err_b7'][m]):.4f} m vs v1 "
            f"{_rmse(d['err_v1'][m]):.4f} m, diff {_rmse(d['err_b7'][m]) - _rmse(d['err_v1'][m]):+.4f}"
            f" m [{lo:+.4f}, {hi:+.4f}] | mean |v1 - anchor| = "
            f"{np.linalg.norm(d['mu'][m] - d['anchor'][m], axis=1).mean():.3f} m")

    layer_b_analysis(d, r_max, out)

    # ---- what the adopted emit() actually produces -------------------------------------------
    r90_anc = k_anc[bucket, 1] * (d["r90"] / d["k"][bucket, 1])  # same widths, anchor's own k
    ems = emit(
        {"hold": d["lastseen"]}, d["mu"], d["anchor"], d["r90"], bucket, sig, r_max, r90_anc
    )
    src = ems["source"]
    out("\nADOPTED emit() OUTPUT (synthesizer/imputation_v1.py)")
    for tag in ("v1", "anchor", "abstained"):
        n = int((src == tag).sum())
        err_t = np.linalg.norm(ems["pos"][src == tag] - d["target"][src == tag], axis=1)
        out(f"  source={tag:10s}: {n:7d} rows ({100 * n / src.size:5.2f}%), emitted-point RMSE "
            f"{_rmse(err_t) if n else float('nan'):6.2f} m")
    off = src == "abstained"
    out(f"  abstained rows: v1 would have scored {_rmse(d['err_v1'][off]):.2f} m there and the "
        f"anchor {_rmse(d['err_b7'][off]):.2f} m (vs {_rmse(d['err_v1']):.2f} m overall) -- "
        "layer B rejects genuinely hard samples.")
    assert int(ems["asserted"].sum()) == int((src != "abstained").sum())
    assert np.array_equal(ems["pos"][src == "anchor"], d["anchor"][src == "anchor"])
    assert np.array_equal(ems["pos"][off], d["lastseen"][off])

    dst = REPO / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("```\n" + "\n".join(out.lines) + "\n```\n", encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
