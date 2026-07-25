"""B4 week 1: B6_vote baseline, conformal wrapper on the frozen B5_blend, feature collector.

Runs the frozen protocol of ``docs/B4_MODEL_PLAN.md`` section 4.1:

    TRAIN   = Metrica Game 1 (full)          -> slot model, tau, blend weights, vote half-life,
                                                region half-widths ``w``
    CALIB   = Metrica Game 2, first half     -> conformal multipliers ``k_b``
    HOLDOUT = Metrica Game 2, second half    -> every reported number

Nothing is fitted on the holdout, and the B5_blend anchor is re-derived exactly as committed
(tau grid-searched on Game 1, per-bucket blend weights on Game 1).

Usage:
    python -m tools.imputation_b4_week1 [--halflife-grid] [--features] [--explore]
                                        [--vote-halflife SECONDS]
"""

from __future__ import annotations

import argparse

import numpy as np

from synthesizer.imputation import (
    BIN_LABELS,
    baseline_errors,
    blend_position,
    collect_samples,
    fit_blend,
    fit_decay_tau,
    fit_slot_model,
    print_full_table,
    _prep_game,
)
from synthesizer.imputation_features import (
    FEATURE_NAMES,
    VOTE_HALFLIFE_S,
    block_bootstrap,
    bucket_of,
    build_features,
    conformal_k,
    coverage_table,
    derive_fields,
    region_scale,
    role_offsets_vote,
    vote_field,
)

BLOCK_S = 60.0  # 1-minute block bootstrap (plan 4.2)


def _rmse(err: np.ndarray) -> float:
    """Root-mean-square of an error array."""
    return float(np.sqrt(np.mean(err**2)))


def _subset(d: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    """Row-select every array in a sample dict."""
    return {k: v[m] for k, v in d.items()}


def _blocks(samples: dict[str, np.ndarray], fps: float) -> np.ndarray:
    """1-minute block ids (period-aware) for the block bootstrap."""
    return samples["period"] * 100000 + (samples["frame"] / (BLOCK_S * fps)).astype(int)


def _prep(name: str, half_w: float | None, coeffs=None, halflife: float = VOTE_HALFLIFE_S):
    """Load + censor a game, derive all fields, and collect hidden samples with the B6 field."""
    g, half_w, summary = _prep_game(name, half_w=half_w)
    print(summary)
    if coeffs is None:
        coeffs = fit_slot_model(g["truth"], g["cent"], g["cam"], g["ranges"])
    fields = derive_fields(
        g["truth"], g["visible"], g["ball"], g["cam"], g["period"], g["fps"], g["ranges"],
        coeffs, halflife,
    )
    samples = collect_samples(
        g["truth"], g["visible"], g["period"], g["fps"], g["vel"],
        np.asarray(fields["slot"]), fields={"b6": np.asarray(fields["b6"])},
    )
    return g, half_w, coeffs, fields, samples


def _halflife_grid(g: dict, samples: dict[str, np.ndarray]) -> None:
    """Report B6_vote RMSE on TRAIN across EMA half-lives (selection happens here, not later)."""
    print("\nvote EMA half-life selection (TRAIN = Game 1 only):")
    tgt = samples["target"]
    f = samples["frame"].astype(int)
    p = samples["slot_id"].astype(int)
    for hl in (0.04, 0.25, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        off, vote, _ = role_offsets_vote(
            g["truth"], g["visible"], g["ranges"], g["fps"], halflife_s=hl
        )
        field = vote_field(off, vote, g["ranges"])
        pos = field[f, p]
        pos = np.where(np.isfinite(pos), pos, samples["hold"])
        err = np.linalg.norm(pos - tgt, axis=1)
        print(f"  halflife={hl:5.1f}s  B6_vote RMSE = {_rmse(err):6.2f} m  p50 = "
              f"{np.percentile(err, 50):5.2f} m")


def _compare_anchors(err: dict[str, np.ndarray], blocks: np.ndarray, title: str) -> None:
    """Per-bucket B5_blend vs B6_vote with a block-bootstrap CI on the RMSE difference."""
    print(f"\n{title}")
    print("horizon   |     n    | B5_blend | B6_vote  | diff (B6-B5) 95% block-boot CI | better")
    print("-" * 88)
    bkt = bucket_of(err["tsls"])
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            continue
        a, c = err["B5_blend"][m], err["B6_vote"][m]
        ra, rc = _rmse(a), _rmse(c)
        lo, hi = _paired_ci(a, c, blocks[m])
        verdict = "B6" if hi < 0 else ("B5" if lo > 0 else "tie")
        print(
            f"{lab:9s} | {int(m.sum()):8d} | {ra:8.2f} | {rc:8.2f} | "
            f"{rc - ra:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]     | {verdict}"
        )
    a, c = err["B5_blend"], err["B6_vote"]
    lo, hi = _paired_ci(a, c, blocks)
    verdict = "B6" if hi < 0 else ("B5" if lo > 0 else "tie")
    print(
        f"{'ALL':9s} | {a.size:8d} | {_rmse(a):8.2f} | {_rmse(c):8.2f} | "
        f"{_rmse(c) - _rmse(a):+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]     | {verdict}"
    )
    print("(diff < 0 means B6_vote is better; CI excluding 0 = significant at 95%)")


def _paired_ci(a: np.ndarray, c: np.ndarray, block: np.ndarray, n_boot: int = 400) -> tuple:
    """Block-bootstrap CI for ``RMSE(c) - RMSE(a)`` with blocks paired across both methods."""
    rng = np.random.default_rng(0)
    uniq, inv = np.unique(block, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=uniq.size)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    aa, cc = a[order] ** 2, c[order] ** 2
    reps = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([np.arange(bounds[j], bounds[j + 1]) for j in pick])
        reps[i] = np.sqrt(cc[idx].mean()) - np.sqrt(aa[idx].mean())
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def _wrapper_report(
    train: dict[str, np.ndarray],
    calib: dict[str, np.ndarray],
    hold: dict[str, np.ndarray],
    tau: float,
    blend_w: np.ndarray,
    fps: float,
) -> None:
    """Fit and score the uncertainty-only wrapper on the frozen B5_blend point prediction."""
    def resid(s: dict[str, np.ndarray]) -> np.ndarray:
        return blend_position(s, tau, blend_w) - s["target"]

    w = region_scale(resid(train), bucket_of(train["tsls"]))
    k = conformal_k(resid(calib), bucket_of(calib["tsls"]), w)
    rows = coverage_table(resid(hold), bucket_of(hold["tsls"]), w, k)
    rc = coverage_table(resid(calib), bucket_of(calib["tsls"]), w, k)

    print("\nFROZEN region parameters (w from TRAIN residuals, k from CALIB):")
    print("horizon   |  w_x    w_y  |  k(50%)  k(90%)")
    for b, lab in enumerate(BIN_LABELS):
        print(f"{lab:9s} | {w[b, 0]:5.2f} {w[b, 1]:5.2f} | {k[b, 0]:7.3f} {k[b, 1]:7.3f}")

    print("\nCOVERAGE -- HOLDOUT (Game 2, 2nd half). CALIB shown only as a sanity echo.")
    print(
        "horizon   |     n    | n_eps | PICP50  (95% CI)      | PICP90  (95% CI)      "
        "|  r50    r90   | calib PICP50/90"
    )
    print("-" * 124)
    episode = hold["slot_id"] * 10**7 + hold["last_frame"]
    s_hold = np.sqrt(
        (((blend_position(hold, tau, blend_w) - hold["target"]) / w[bucket_of(hold["tsls"])]) ** 2)
        .sum(axis=1)
    )
    blk = _blocks(hold, fps)
    bkt = bucket_of(hold["tsls"])
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            continue
        r = rows[b]
        ci50 = block_bootstrap((s_hold[m] <= k[b, 0]).astype(float), blk[m], stat="mean")
        ci90 = block_bootstrap((s_hold[m] <= k[b, 1]).astype(float), blk[m], stat="mean")
        print(
            f"{lab:9s} | {r['n']:8d} | {np.unique(episode[m]).size:5d} | {100 * r['picp_50']:6.1f}  "
            f"[{100 * ci50[0]:5.1f},{100 * ci50[1]:5.1f}] | {100 * r['picp_90']:6.1f}  "
            f"[{100 * ci90[0]:5.1f},{100 * ci90[1]:5.1f}] | {r['r_50']:5.1f}  {r['r_90']:5.1f} | "
            f"{100 * rc[b]['picp_50']:5.1f} / {100 * rc[b]['picp_90']:5.1f}"
        )
    ok50 = [45 <= 100 * r["picp_50"] <= 55 for r in rows if r["n"]]
    ok90 = [85 <= 100 * r["picp_90"] <= 95 for r in rows if r["n"]]
    print(
        f"gate 2 (PICP within +/-5 pts of nominal in every bucket): "
        f"50% -> {sum(ok50)}/{len(ok50)} buckets, 90% -> {sum(ok90)}/{len(ok90)} buckets"
    )
    print("r50/r90 = equivalent region radius in metres (sqrt(area/pi)); constant within a bucket")
    print("by construction -- the fallback has no per-sample width (v1's quantile heads will).")
    print("n_eps = distinct occlusion episodes contributing to the bucket = the honest effective")
    print("sample size; the raw n is ~25 near-duplicate frames per second of a single episode.")


def _feature_report(
    fields: dict, samples: dict[str, np.ndarray], tau: float, blend_w: np.ndarray
) -> None:
    """Build the feature matrix on the holdout samples and print a column summary."""
    blend = blend_position(samples, tau, blend_w)
    x = build_features(fields, samples, blend)
    print(f"\nFEATURE MATRIX: {x.shape[0]} samples x {x.shape[1]} columns ({x.dtype})")
    finite = np.isfinite(x).all(axis=0)
    print("col                  |     mean |      std |      min |      max | finite")
    print("-" * 78)
    for j, name in enumerate(FEATURE_NAMES):
        c = x[:, j]
        good = np.isfinite(c)
        print(
            f"{name:20s} | {c[good].mean():8.2f} | {c[good].std():8.2f} | {c[good].min():8.2f} | "
            f"{c[good].max():8.2f} | {'yes' if finite[j] else f'{100 * good.mean():.1f}%'}"
        )


def _explore(
    train: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    tau: float,
    blend_w: np.ndarray,
    fps: float,
    label: str,
) -> None:
    """Exploratory (NOT pre-registered) anchor variant: B5's blend with B6_vote as structure.

    Args:
        train: TRAIN samples (Game 1) -- the only split the blend weights are fitted on.
        split: The split being scored.
        tau: Frozen decay constant.
        blend_w: Frozen B5 per-bucket weights.
        fps: Frame rate.
        label: Split name for the header.
    """
    alt_train = dict(train)
    alt_train["slot"] = train["b6"]
    w_alt = fit_blend(alt_train, tau)
    alt = dict(split)
    alt["slot"] = split["b6"]
    b7 = np.linalg.norm(blend_position(alt, tau, w_alt) - split["target"], axis=1)
    b5 = np.linalg.norm(blend_position(split, tau, blend_w) - split["target"], axis=1)
    blk = _blocks(split, fps)
    bkt = bucket_of(split["tsls"])
    print(f"\nEXPLORATORY (not pre-registered) on {label}: B7 = veldecay blended with B6_vote as")
    print(f"the structural term, weights refit on TRAIN only = {np.round(w_alt, 2).tolist()}")
    print("horizon   |     n    | B5_blend | B7_vote_blend | diff 95% block-boot CI  | better")
    print("-" * 88)
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            continue
        lo, hi = _paired_ci(b5[m], b7[m], blk[m])
        verdict = "B7" if hi < 0 else ("B5" if lo > 0 else "tie")
        print(
            f"{lab:9s} | {int(m.sum()):8d} | {_rmse(b5[m]):8.2f} | {_rmse(b7[m]):13.2f} | "
            f"{_rmse(b7[m]) - _rmse(b5[m]):+6.2f} [{lo:+6.2f}, {hi:+6.2f}] | {verdict}"
        )
    lo, hi = _paired_ci(b5, b7, blk)
    print(
        f"{'ALL':9s} | {b5.size:8d} | {_rmse(b5):8.2f} | {_rmse(b7):13.2f} | "
        f"{_rmse(b7) - _rmse(b5):+6.2f} [{lo:+6.2f}, {hi:+6.2f}] |"
    )


def _v0_diag(
    hold: dict[str, np.ndarray], tau: float, blend_w: np.ndarray, g2: dict, fields: dict
) -> None:
    """Quantify how much of the frozen anchor's ``v0`` was actually observable.

    ``estimate_velocity`` runs on the full truth, so a player whose visibility flickered can
    have a last-seen velocity computed across frames where they were off camera. This is a
    diagnostic on the frozen anchor; nothing is changed.

    Args:
        hold: Holdout samples.
        tau: Frozen decay constant.
        blend_w: Frozen blend weights.
        g2: Game-2 bundle (holds the anchor's full-truth velocity field).
        fields: Derived fields (hold the visibility-masked velocity field).
    """
    ell = hold["last_frame"].astype(int)
    p = hold["slot_id"].astype(int)
    v_full = g2["vel"][ell, p]
    v_obs = np.asarray(fields["vel"])[ell, p]
    v_full = np.where(np.isfinite(v_full), v_full, 0.0)
    v_obs = np.where(np.isfinite(v_obs), v_obs, 0.0)
    d = np.linalg.norm(v_full - v_obs, axis=1)
    print(
        f"\nfrozen-anchor v0 observability: {100 * (d > 0.1).mean():.1f}% of holdout samples have "
        f"a last-seen velocity that differs (>0.1 m/s) when the 0.5 s look-back window is "
        f"restricted to\nframes where the player was actually on camera; mean |diff| = "
        f"{d.mean():.3f} m/s, p99 = {np.percentile(d, 99):.2f} m/s."
    )
    b5 = np.linalg.norm(blend_position(hold, tau, blend_w) - hold["target"], axis=1)
    alt = dict(hold)
    alt["v0"] = v_obs
    b5_obs = np.linalg.norm(blend_position(alt, tau, blend_w) - hold["target"], axis=1)
    print(
        f"B5_blend recomputed with the observable-only v0: RMSE {_rmse(b5_obs):.2f} m vs "
        f"{_rmse(b5):.2f} m as frozen (diagnostic only -- the frozen anchor is unchanged)."
    )


def main() -> None:
    """Run the week-1 protocol end to end and print every table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--halflife-grid", action="store_true", help="score the vote EMA grid on TRAIN")
    ap.add_argument("--features", action="store_true", help="build + summarise the feature matrix")
    ap.add_argument("--explore", action="store_true", help="B7 vote-blend + v0 observability")
    ap.add_argument(
        "--vote-halflife", type=float, default=VOTE_HALFLIFE_S,
        help="EMA half-life (s) for the B6_vote role offsets; selected on TRAIN only",
    )
    args = ap.parse_args()

    print("=" * 92)
    print("TRAIN = Game 1 (full) | CALIB = Game 2 H1 | HOLDOUT = Game 2 H2  [frozen 2026-07-24]")
    print("=" * 92)
    g1, half_w, coeffs, f1, s1 = _prep("Sample_Game_1", None, halflife=args.vote_halflife)
    print(f"window: {2 * half_w:.1f} m wide x 68 m tall")
    if args.halflife_grid:
        _halflife_grid(g1, s1)
    tau, tau_rmse = fit_decay_tau(s1)
    blend_w = fit_blend(s1, tau)
    print(f"\nfrozen anchor re-derived on TRAIN: tau={tau:.2f}s (B3 in-sample {tau_rmse:.2f} m), "
          f"blend weights={np.round(blend_w, 2).tolist()}")
    print(f"vote EMA half-life = {args.vote_halflife:.2f}s (selected on TRAIN)")
    err1 = baseline_errors(s1, tau, blend_w)
    print_full_table(err1, "GAME 1 = TRAIN (in-sample, not reportable)")

    del f1
    g2, _, _, f2, s2 = _prep("Sample_Game_2", half_w, coeffs, args.vote_halflife)
    calib = _subset(s2, s2["period"] == 1)
    hold = _subset(s2, s2["period"] == 2)
    print(f"\nCALIB n={calib['tsls'].size}  HOLDOUT n={hold['tsls'].size}")

    err_h = baseline_errors(hold, tau, blend_w)
    print_full_table(err_h, "HOLDOUT = GAME 2 SECOND HALF (the reportable table)")
    _compare_anchors(err_h, _blocks(hold, g1["fps"]), "ANCHOR SHOOT-OUT on HOLDOUT")
    err_c = baseline_errors(calib, tau, blend_w)
    _compare_anchors(err_c, _blocks(calib, g1["fps"]), "ANCHOR SHOOT-OUT on CALIB (echo only)")

    _wrapper_report(s1, calib, hold, tau, blend_w, g1["fps"])
    if args.explore:
        _explore(s1, calib, tau, blend_w, g1["fps"], "CALIB (the split anchor choices belong on)")
        _explore(s1, hold, tau, blend_w, g1["fps"], "HOLDOUT (disclosed look, see report)")
        _v0_diag(hold, tau, blend_w, g2, f2)
    if args.features:
        _feature_report(f2, hold, tau, blend_w)


if __name__ == "__main__":
    main()
