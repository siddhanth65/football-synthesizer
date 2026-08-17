"""v10-W5: capture part of the +6.25 GS-HOTA per-frame-accuracy oracle, in mapping space.

v10-W4 measured the prize: freezing association and replacing every solved frame's homography by a
GT-fitted one takes DEV-20 flags-ON from 55.07 to 61.32 (+6.25). Our displacement from that oracle
is 0.762 m mean / 0.479 m median, and the loss is quadratic in it. This session tries to close part
of that gap **without ground truth**, using only what the shipped decode already computed.

The lever nobody used yet: PnLCalib emits **18 camera hypotheses per frame** and our gate keeps the
first admissible one, per frame, memorylessly. All 18 are already cached
(``outputs/gsr/v10_w2_full/ctrl``). This tool asks whether a temporally-aware choice -- or a
continuous blend -- of those hypotheses beats the greedy per-frame pick.

``--oracle``    D1, measure-only. The candidate-pool oracle: per frame, the best of its own cached
                hypotheses under the GT-anchored accuracy instrument. Bounds every arm below that
                cannot invent new correspondences.
``--accuracy``  Gate 1. GT-anchored accuracy instrument (median / p90 / mean **and the mean squared
                displacement**, which is what the W4 curve says the score responds to) per arm.
``--positions`` CPU. One positions parquet set per arm (gate -> reproject -> fill), lineage frozen.
``--score``     CPU. The frozen v6 chain per arm + the v9-W7 paired scorer, flags ON and OFF.

Ground truth is read only to score and to fit the oracle homographies; no arm consumes it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_v10_w5")

#: The declared DEV-20, in the v10-W2 order.
DEV20 = ["SNGS-021", "SNGS-024", "SNGS-027", "SNGS-030", "SNGS-033", "SNGS-036", "SNGS-039",
         "SNGS-042", "SNGS-045", "SNGS-048", "SNGS-051", "SNGS-054", "SNGS-057", "SNGS-078",
         "SNGS-081", "SNGS-084", "SNGS-087", "SNGS-090", "SNGS-093", "SNGS-096"]
#: Lineage: the v10-W2 ctrl arm (the shipped decode; the half-cell rider s4 stays OFF), as in W4.
CACHE_DIR = Path("outputs/gsr/v10_w2_full/ctrl")
POSITIONS_SUBDIR = "positions_v10w2ctrl_v6det"
#: Frozen temporal fill gap (``results/gsr_calibgate_frozen.json``).
FILL_GAP = 10
#: GS-HOTA per m^2 of mean squared displacement from the oracle homography, small-error regime.
#: Fitted from the W4 F1/F2 Gaussian ladder (sigma 0.1/0.25/0.5 -> loss 0.133/0.849/3.485, with
#: E[d^2] = 2 sigma^2): 6.65 / 6.79 / 6.97. Fixed here BEFORE any W5 arm exists.
HOTA_PER_MSD = 6.9
#: The W4 measured value of removing ALL displacement (arm ``O_acc``), DEV-20 flags ON.
ORACLE_GS_HOTA = 6.254


def predicted_gain(loss_ctrl: float, loss_arm: float) -> float:
    """GS-HOTA an arm is predicted to gain, from its kernel-loss reduction (pure).

    The W4 curve is quadratic in displacement (:data:`HOTA_PER_MSD`) and saturates past the 5 m
    kernel, which is exactly :func:`kernel_loss`'s shape. The instrument's population (every GT
    person on every solved frame) is not the submission's row population, so the ABSOLUTE loss does
    not transfer -- the FRACTION does. Prediction is the measured oracle times the fractional
    reduction, which is exact at both ends (0 -> 0, full -> +6.254).
    """
    if loss_ctrl <= 0:
        return 0.0
    return ORACLE_GS_HOTA * (1.0 - loss_arm / loss_ctrl)


#: GS-HOTA's localisation kernel width, ``trackeval/datasets/soccernet_gs.calculate_sigma(5.0)``.
GS_SIGMA_M = 2.042694913268175


def kernel_loss(err: np.ndarray) -> float:
    """Mean ``1 - exp(-d^2 / 2 sigma^2)`` -- the similarity the evaluator actually loses (pure).

    The raw mean square is useless as a summary here: a handful of frames whose homography is
    hundreds of metres out dominate it, while GS-HOTA stopped charging for them at ~5 m. This is the
    evaluator's own kernel, so it is quadratic where the score is quadratic and saturates where the
    score saturates.
    """
    if not len(err):
        return 0.0
    return float(np.mean(1.0 - np.exp(-0.5 * (err / GS_SIGMA_M) ** 2)))


def _stats(err: np.ndarray) -> dict:
    """Median / p90 / p99 / mean / mean-square / kernel loss / share within the 5 m gate."""
    if not len(err):
        return {"n": 0}
    return {"n": int(len(err)), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)), "p99": float(np.percentile(err, 99)),
            "mean": float(err.mean()), "msd": float((err ** 2).mean()),
            "kloss": kernel_loss(err), "within_5m": float((err <= 5.0).mean())}


def load_seq(seq: str, out_dir: Path, cache_dir: Path
             ) -> tuple[pd.DataFrame, dict[int, list], dict[int, np.ndarray], dict[int, float], int]:
    """``(positions, {frame: candidates}, {frame: gated H}, {frame: solver err_m}, n_frames)``."""
    from generator.calibrate import select_calibration  # noqa: PLC0415
    from generator.postprocess import PLAYER_ROLES  # noqa: PLC0415
    from tools.gsr_v10_w2_calib import gate_homographies, load_cache  # noqa: PLC0415

    df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
    by_frame = load_cache(cache_dir / f"{seq}.npz")
    homs, _rep = gate_homographies(by_frame, df)
    people = df[df["role"].isin(PLAYER_ROLES)]
    feet = {int(f): g[["image_x", "image_y"]].to_numpy(dtype=float)
            for f, g in people.groupby("frame")}
    err = {}
    for fr in homs:
        foot = feet.get(fr)
        foot = foot[np.isfinite(foot).all(axis=1)] if foot is not None else None
        err[fr] = float(select_calibration(by_frame[fr], foot_points=foot).error_m)
    return df, by_frame, homs, err, int(df["frame"].max()) + 1


def foot_points_by_frame(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """``{frame: (N, 2)}`` detected player foot points -- the gate's plausibility input."""
    from generator.postprocess import PLAYER_ROLES  # noqa: PLC0415

    people = df[df["role"].isin(PLAYER_ROLES)]
    out = {}
    for f, g in people.groupby("frame"):
        p = g[["image_x", "image_y"]].to_numpy(dtype=float)
        out[int(f)] = p[np.isfinite(p).all(axis=1)]
    return out


def admissible(cands: list, foot: np.ndarray | None, max_error_m: float = 2.0) -> list[int]:
    """Indices of the hypotheses that pass BOTH shipped gates (metre error + on-pitch plausibility)."""
    from generator.postprocess import onpitch_plausible  # noqa: PLC0415

    out = []
    for i, c in enumerate(cands):
        if c.error_m > max_error_m:
            continue
        if foot is None or len(foot) == 0 or onpitch_plausible(c.homography, foot):
            out.append(i)
    return out


# === D1: the candidate-pool oracle ================================================================
def run_oracle(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str]) -> dict:
    """Per frame, how much better than the gate's pick is the best of its own cached hypotheses?

    Two accuracy references are reported for every population:

    * ``gt`` -- the v10-W2 instrument: GT foot points pushed through the homography against that
      annotation's own ``bbox_pitch`` (calibration error including whatever the annotation's own
      geometry does not model).
    * ``disp`` -- displacement from that frame's **GT-fitted** homography, evaluated on the same GT
      foot points. This is the quantity the W4 sensitivity curve prices, so its mean square times
      :data:`HOTA_PER_MSD` is a GS-HOTA prediction.
    """
    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    keys = ("sel", "best_admissible", "best_any", "gtfit")
    pooled: dict[str, dict[str, list]] = {r: {k: [] for k in keys} for r in ("gt", "disp")}
    per_seq: dict[str, dict] = {}
    pool_sizes, n_distinct, picked_rank = [], [], []
    for seq in seqs:
        df, by_frame, homs, _err, _n = load_seq(seq, out_dir, cache_dir)
        gt = gt_people(data_dir / seq)
        feet = foot_points_by_frame(df)
        vals: dict[str, dict[str, list]] = {r: {k: [] for k in keys} for r in ("gt", "disp")}
        for fr, h_sel in sorted(homs.items()):
            g = gt.get(fr)
            if g is None or len(g) < 6:  # 6 keeps the oracle fit honest, as in W4
                continue
            h_gt = estimate_homography(g[:, :2], g[:, 2:4])
            if h_gt is None or not np.isfinite(h_gt).all():
                continue
            ref_gt, ref_fit = g[:, 2:4], apply_homography(h_gt, g[:, :2])
            cands = by_frame.get(fr) or []
            adm = set(admissible(cands, feet.get(fr)))
            errs = []
            for i, c in enumerate(cands):
                p = apply_homography(c.homography, g[:, :2])
                errs.append((np.linalg.norm(p - ref_gt, axis=1),
                             np.linalg.norm(p - ref_fit, axis=1), i in adm))
            if not errs:
                continue
            e_sel = (np.linalg.norm(apply_homography(h_sel, g[:, :2]) - ref_gt, axis=1),
                     np.linalg.norm(apply_homography(h_sel, g[:, :2]) - ref_fit, axis=1))
            order = sorted(range(len(errs)), key=lambda i: float(np.median(errs[i][1])))
            best_any = order[0]
            best_adm = next((i for i in order if errs[i][2]), best_any)
            sel_i = next((i for i, c in enumerate(cands) if c.homography is h_sel), None)
            picked_rank.append(order.index(sel_i) if sel_i is not None else -1)
            pool_sizes.append(len(cands))
            grids = np.array([np.median(errs[i][1]) for i in range(len(errs))])
            n_distinct.append(int(len(np.unique(np.round(grids, 3)))))
            for ref, j in (("gt", 0), ("disp", 1)):
                vals[ref]["sel"].append(e_sel[j])
                vals[ref]["best_admissible"].append(errs[best_adm][j])
                vals[ref]["best_any"].append(errs[best_any][j])
            vals["gt"]["gtfit"].append(np.linalg.norm(ref_fit - ref_gt, axis=1))
            vals["disp"]["gtfit"].append(np.zeros(len(g)))
        rec = {}
        for ref in ("gt", "disp"):
            for k in keys:
                v = np.concatenate(vals[ref][k]) if vals[ref][k] else np.zeros(0)
                pooled[ref][k].append(v)
                rec[f"{ref}_{k}"] = _stats(v)
        per_seq[seq] = rec
        logger.info("%s: sel %.3f -> best-admissible %.3f / best-any %.3f m (disp median), "
                    "frames %d", seq, rec["disp_sel"].get("median", float("nan")),
                    rec["disp_best_admissible"].get("median", float("nan")),
                    rec["disp_best_any"].get("median", float("nan")), len(vals["disp"]["sel"]))
    rank = np.array(picked_rank)
    out: dict = {"pooled": {}, "per_seq": per_seq,
                 "pool": {"mean_candidates": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
                          "mean_distinct_by_accuracy": float(np.mean(n_distinct))
                          if n_distinct else 0.0,
                          "gate_pick_is_best_frac": float((rank == 0).mean()) if len(rank) else 0.0,
                          "gate_pick_rank_median": float(np.median(rank)) if len(rank) else -1.0,
                          "frames": len(pool_sizes)}}
    for ref in ("gt", "disp"):
        out["pooled"][ref] = {k: _stats(np.concatenate([x for x in pooled[ref][k] if len(x)]))
                              for k in keys}
    out["predicted_gs_hota"] = {k: round(predicted_gain(out["pooled"]["disp"]["sel"]["kloss"],
                                                        out["pooled"]["disp"][k]["kloss"]), 3)
                                for k in keys}
    return out


# === the arms =====================================================================================
#: Declared arms (``results/gsr_v10_w5_registered.json``). ``ctrl`` is the shipped gate's pick.
#: ``C`` / ``D`` / ``K`` are the compositions of amendment 1 -- one of them takes the third slot.
ARMS = ("ctrl", "R", "S", "B", "C", "D", "K")


def smoothed(homs: dict[int, np.ndarray], n_frames: int, err: dict[int, float]
             ) -> dict[int, np.ndarray]:
    """Arm R: mapping-space RTS smoothing, kept only on the frames the solver answered."""
    from generator.camera_track import smooth_homographies  # noqa: PLC0415

    sm, _st = smooth_homographies(homs, n_frames, err_m=err, mode="mapping")
    return {f: h for f, h in sm.items() if f in homs}


def reselect_once(cur: dict[int, np.ndarray], cur_err: dict[int, float],
                  by_frame: dict[int, list], feet: dict[int, np.ndarray],
                  ref: dict[int, np.ndarray]) -> tuple[dict, dict, int]:
    """One re-selection pass against a reference mapping: ``(homs, err_m, n_changed)``."""
    from generator.camera_track import grid_shift_m  # noqa: PLC0415

    nxt, nxt_err, changed = {}, {}, 0
    for fr in cur:
        cands = by_frame.get(fr) or []
        adm = admissible(cands, feet.get(fr))
        r = ref.get(fr)
        if r is None or not adm:
            nxt[fr], nxt_err[fr] = cur[fr], cur_err.get(fr, 1.0)
            continue
        best = adm[int(np.argmin([grid_shift_m(cands[i].homography, r) for i in adm]))]
        nxt[fr], nxt_err[fr] = cands[best].homography, float(cands[best].error_m)
        changed += int(cands[best].homography is not cur[fr])
    return nxt, nxt_err, changed


def reselect(homs: dict[int, np.ndarray], by_frame: dict[int, list], feet: dict[int, np.ndarray],
             n_frames: int, err: dict[int, float], *, passes: int = 2
             ) -> tuple[dict[int, np.ndarray], dict]:
    """Arm S: per frame, take the gate-admissible hypothesis closest to the smoothed trajectory.

    The output is always a genuine PnLCalib solve -- no blending, no re-fitting -- so a frame either
    keeps the gate's pick or swaps it for a sibling hypothesis of the same forward pass.
    """
    cur, cur_err = dict(homs), dict(err)
    st: dict = {"passes": passes, "changed": 0}
    for _ in range(passes):
        cur, cur_err, st["changed"] = reselect_once(cur, cur_err, by_frame, feet,
                                                    smoothed(cur, n_frames, cur_err))
    return cur, st


def batch_refine(homs: dict[int, np.ndarray], by_frame: dict[int, list],
                 feet: dict[int, np.ndarray], n_frames: int, *, iters: int = 8,
                 clamp_m: float = float("inf"),
                 clamp_ref: dict[int, np.ndarray] | None = None,
                 stiffness: np.ndarray | None = None
                 ) -> tuple[dict[int, np.ndarray], dict]:
    """Arm B: one robust batch solve per sequence over every frame's mapping at once.

    Unknowns are the pitch coordinates of the fixed ``CHECK_GRID`` per frame (the four-point /
    mapping parameterisation -- never decomposed camera parameters). Every gate-admissible cached
    hypothesis of a frame pulls on that frame with a Cauchy weight; a second-difference penalty
    along time pulls the whole clip straight. Solved by IRLS, each iteration an exact banded solve
    per grid coordinate. Smoothness weights come from the series' own noise estimate -- no dial.

    Args:
        stiffness: Optional ``(n_frames,)`` per-frame multiplier on the second-difference penalty
            (v10-W8 arm A). ``None`` -- the default -- is the on-record uniform prior, byte for byte.
    """
    from scipy.sparse import diags, eye  # noqa: PLC0415
    from scipy.sparse.linalg import spsolve  # noqa: PLC0415

    from generator.camera_track import CHECK_GRID, _noise_estimates, grid_compose  # noqa: PLC0415
    from generator.calibrate import apply_homography  # noqa: PLC0415

    ng = len(CHECK_GRID)
    z = np.full((n_frames, 2 * ng), np.nan)
    cand: dict[int, np.ndarray] = {}
    spreads = []
    for fr in homs:
        z[fr] = apply_homography(homs[fr], CHECK_GRID).reshape(-1)
        adm = admissible(by_frame.get(fr) or [], feet.get(fr))
        rows = [apply_homography((by_frame[fr])[i].homography, CHECK_GRID).reshape(-1)
                for i in adm]
        rows = [r for r in rows if np.isfinite(r).all()]
        if rows:
            cand[fr] = np.asarray(rows)
            d = np.median(np.linalg.norm((cand[fr] - z[fr]).reshape(len(rows), ng, 2), axis=2),
                          axis=1)
            spreads.append(float(np.median(d)))
    valid = np.isfinite(z).all(axis=1)
    idx = np.flatnonzero(valid)
    st: dict = {"frames": n_frames, "solved": len(homs), "with_candidates": len(cand),
                "iters": iters}
    if len(idx) < 8 or not cand:
        st["skipped"] = True
        return dict(homs), st
    scale = max(float(np.median(spreads)) if spreads else 0.1, 0.05)
    st["cauchy_scale_m"] = scale
    g = np.column_stack([np.interp(np.arange(n_frames), idx, z[idx, j])
                         for j in range(2 * ng)])
    lam = np.array([(_noise_estimates(z[idx, j])[0] / max(_noise_estimates(z[idx, j])[1], 1e-12))
                    ** 2 for j in range(2 * ng)])
    st["lambda_median"] = float(np.median(lam))
    d2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_frames - 2, n_frames), format="csr")
    if stiffness is None:
        reg = (d2.T @ d2).tocsr()
    else:
        # row r of d2 is the second difference centred on frame r+1, so it carries that frame's
        # multiplier; the quadratic form stays symmetric positive semi-definite.
        w_row = np.clip(np.asarray(stiffness, dtype=float)[1:n_frames - 1], 1e-6, None)
        reg = (d2.T @ diags(w_row, 0, format="csr") @ d2).tocsr()
        st["stiffness_median"] = float(np.median(w_row))
        st["stiffness_min"] = float(w_row.min())
    frames = np.array(sorted(cand))
    for _ in range(iters):
        wsum = np.zeros(n_frames)
        rhs = np.zeros((n_frames, 2 * ng))
        for fr in frames:
            c = cand[fr]
            d = np.median(np.linalg.norm((c - g[fr]).reshape(len(c), ng, 2), axis=2), axis=1)
            w = 1.0 / (1.0 + (d / scale) ** 2)
            wsum[fr] = w.sum()
            rhs[fr] = w @ c
        wmat = diags(wsum, 0, format="csr")
        for j in range(2 * ng):
            a = (wmat + lam[j] * reg + 1e-9 * eye(n_frames, format="csr")).tocsc()
            g[:, j] = spsolve(a, rhs[:, j] + 1e-9 * g[:, j])
    from generator.camera_track import grid_shift_m  # noqa: PLC0415

    ref = clamp_ref if clamp_ref is not None else homs
    out, st["compose_failed"], st["clamped"] = {}, 0, 0
    for fr in homs:
        h = grid_compose(g[fr])
        if h is None:
            st["compose_failed"] += 1
            out[fr] = homs[fr]
        elif np.isfinite(clamp_m) and grid_shift_m(ref.get(fr, homs[fr]), h) > clamp_m:
            st["clamped"] += 1
            out[fr] = ref.get(fr, homs[fr])
        else:
            out[fr] = h
    return out, st


def arm_homographies(arm: str, seq: str, out_dir: Path, cache_dir: Path
                     ) -> tuple[dict[int, np.ndarray], dict]:
    """``({frame: H}, stats)`` for one arm on one sequence (the ctrl gate is always the base)."""
    df, by_frame, homs, err, n = load_seq(seq, out_dir, cache_dir)
    feet = foot_points_by_frame(df)
    if arm == "ctrl":
        return homs, {"solved": len(homs)}
    if arm == "R":
        return smoothed(homs, n, err), {"solved": len(homs)}
    if arm == "S":
        return reselect(homs, by_frame, feet, n, err)
    if arm == "B":
        return batch_refine(homs, by_frame, feet, n)
    if arm == "C":  # B then S: the batch solve used purely as a selector
        ref, st = batch_refine(homs, by_frame, feet, n)
        out, _e, st["reselected"] = reselect_once(homs, err, by_frame, feet, ref)
        return out, st
    if arm == "D":  # S then B: re-select first, batch-refine from those picks
        picks, st = reselect(homs, by_frame, feet, n, err)
        out, st2 = batch_refine(picks, by_frame, feet, n)
        return out, st | st2
    if arm == "K":  # B under the W4 clamp against the frame's own gate solve
        return batch_refine(homs, by_frame, feet, n, clamp_m=2.0)
    raise ValueError(f"unknown arm {arm!r}")


# === gate 1: the accuracy instrument ==============================================================
def run_accuracy(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str],
                 arms: list[str]) -> dict:
    """GT-anchored accuracy per arm, paired on the frames every arm answers."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    pooled: dict[str, dict[str, list]] = {r: {a: [] for a in arms} for r in ("gt", "disp")}
    per_seq: dict[str, dict] = {}
    arm_stats: dict[str, dict] = {a: {} for a in arms}
    for seq in seqs:
        gt = gt_people(data_dir / seq)
        hs: dict[str, dict[int, np.ndarray]] = {}
        for a in arms:
            hs[a], arm_stats[a][seq] = arm_homographies(a, seq, out_dir, cache_dir)
        shared = sorted(set.intersection(*[set(hs[a]) for a in arms]) & set(gt))
        vals: dict[str, dict[str, list]] = {r: {a: [] for a in arms} for r in ("gt", "disp")}
        for fr in shared:
            g = gt[fr]
            if len(g) < 6:
                continue
            h_gt = estimate_homography(g[:, :2], g[:, 2:4])
            if h_gt is None or not np.isfinite(h_gt).all():
                continue
            ref_fit = apply_homography(h_gt, g[:, :2])
            for a in arms:
                p = apply_homography(hs[a][fr], g[:, :2])
                vals["gt"][a].append(np.linalg.norm(p - g[:, 2:4], axis=1))
                vals["disp"][a].append(np.linalg.norm(p - ref_fit, axis=1))
        rec = {"frames_paired": len(shared)}
        for r in ("gt", "disp"):
            for a in arms:
                v = np.concatenate(vals[r][a]) if vals[r][a] else np.zeros(0)
                pooled[r][a].append(v)
                rec[f"{r}_{a}"] = _stats(v)
        per_seq[seq] = rec
        logger.info("%s: %s", seq, " | ".join(
            f"{a} {rec['disp_' + a].get('median', float('nan')):.3f}m "
            f"k{rec['disp_' + a].get('kloss', float('nan')):.4f}" for a in arms))
    out: dict = {"per_seq": per_seq, "pooled": {}, "verdict": {}, "arm_stats": arm_stats}
    for r in ("gt", "disp"):
        out["pooled"][r] = {a: _stats(np.concatenate([x for x in pooled[r][a] if len(x)]))
                            for a in arms}
    base = np.concatenate([x for x in pooled["disp"][arms[0]] if len(x)])
    ctrl = out["pooled"]["disp"][arms[0]]
    for a in arms[1:]:
        arm = np.concatenate([x for x in pooled["disp"][a] if len(x)])
        cur = out["pooled"]["disp"][a]
        d = arm - base if len(arm) == len(base) else np.zeros(0)
        med_gain = 1.0 - cur["median"] / ctrl["median"]
        k_gain = 1.0 - cur["kloss"] / ctrl["kloss"]
        out["verdict"][a] = {
            "median_improvement": round(med_gain, 4), "kloss_improvement": round(k_gain, 4),
            "predicted_gs_hota": round(predicted_gain(ctrl["kloss"], cur["kloss"]), 3),
            "gate_1_pass": bool(med_gain >= 0.10 or k_gain >= 0.15),
            "rows_better": int((d < 0).sum()), "rows_worse": int((d > 0).sum()),
            "wilcoxon_p": (float(wilcoxon(d).pvalue) if len(d) and np.any(d != 0) else 1.0)}
    return out


# === positions and scoring ========================================================================
def build_positions(out_dir: Path, cache_dir: Path, seqs: list[str], arms: list[str]) -> dict:
    """One positions parquet set per arm: arm homographies -> reproject -> ``fill_calibration_gaps``."""
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415
    from tools.gsr_w2_calibswap import swap_positions  # noqa: PLC0415

    stats: dict[str, dict] = {a: {} for a in arms}
    for seq in seqs:
        df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
        for a in arms:
            homs, st = arm_homographies(a, seq, out_dir, cache_dir)
            tab, sw = swap_positions(df, homs, origin=(0.0, 0.0), mode="t1")
            tab = fill_calibration_gaps(tab, max_gap=FILL_GAP)
            sub = out_dir / f"positions_v10w5{a}_v6det"
            sub.mkdir(parents=True, exist_ok=True)
            tab.to_parquet(sub / f"{seq}.parquet", index=False)
            st.update({k: sw[k] for k in ("pitch_rows_control", "pitch_rows_final")})
            st["pitch_rows_after_fill"] = int(np.isfinite(tab["pitch_x"]).sum())
            stats[a][seq] = st
        logger.info("%s: %s", seq, {a: stats[a][seq]["pitch_rows_after_fill"] for a in arms})
    return stats


def score_arms(data_dir: Path, out_dir: Path, seqs: list[str], arms: list[str]) -> dict:
    """The frozen v6 chain per arm, then the v9-W7 paired scorer against the same-lineage control."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v6det import TAU  # noqa: PLC0415
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache_v6det"
    raw: dict[str, dict] = {}
    for a in arms:
        sub = POSITIONS_SUBDIR if a == "ctrl" else f"positions_v10w5{a}_v6det"
        res = eiou.run_point(data_dir, out_dir, seqs,
                             eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                             embedder="clip_v6det", tau=TAU, tag=f"v10w5_{a}",
                             percrop_variant="_v6_v6det", positions_subdir=sub)
        h = res["gs_hota"]
        raw[a] = {"positions_subdir": sub, "gs_hota": h, "gs_hota_per_seq": res["gs_hota_per_seq"]}
        print(f"{a:<6} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
              f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f}", flush=True)
    ctrl = out_dir / "deleak_v10w5_ctrl" / "predictions" / "data"
    paired = {a: score_pair(ctrl, out_dir / f"deleak_v10w5_{a}" / "predictions" / "data",
                            data_dir, out_dir / "v10_w5" / "pair" / a, seqs)
              for a in arms if a != "ctrl"}
    return {"arms": raw, "paired_vs_ctrl": paired}


# === self-check ===================================================================================
def _demo() -> None:
    """Assert the pure seams: the statistics block and the admissibility replay."""
    from generator.calibrate import CalibCandidate  # noqa: PLC0415

    s = _stats(np.array([0.0, 1.0, 2.0, 9.0]))
    assert s["msd"] == (0 + 1 + 4 + 81) / 4 and s["within_5m"] == 0.75, s
    good = np.array([[0.06, 0.004, -8.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])
    cands = [CalibCandidate(good, 9.0, 8, "full", 0.0, 1.0),
             CalibCandidate(good, 0.4, 8, "full", 5.0, 2.0)]
    feet = np.column_stack([np.linspace(300, 1500, 5), np.full(5, 500.0)])
    assert admissible(cands, feet) == [1], admissible(cands, feet)
    assert admissible(cands, None, max_error_m=10.0) == [0, 1]

    # a panning camera, 6 hypotheses per frame: one near-clean, five badly off. The gate is
    # simulated as "always picks a bad one"; S must find the clean sibling and B must beat the pick.
    from generator.camera_track import Pose, compose, grid_shift_m  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 120
    truth, by_frame, picks = {}, {}, {}
    for i in range(n):
        p = Pose(0.30 + 0.0010 * i, 1.85, 0.02, float(np.log(3800.0)), float(np.log(0.95)),
                 2.0, -55.0, 15.0)
        truth[i] = compose(p)
        pool = [compose(Pose(p.pan + rng.normal(0, 2e-4), p.tilt + rng.normal(0, 2e-4), p.roll,
                             p.log_f, p.log_aspect, p.cx, p.cy, p.cz))]
        pool += [compose(Pose(p.pan + rng.normal(0, 6e-3), p.tilt + rng.normal(0, 6e-3), p.roll,
                              p.log_f, p.log_aspect, p.cx, p.cy, p.cz)) for _ in range(5)]
        by_frame[i] = [CalibCandidate(h, 0.5, 20, "full", 0.0, 3.0) for h in pool]
        picks[i] = pool[1 + int(rng.integers(0, 5))]
    err = dict.fromkeys(range(n), 0.5)
    base = float(np.median([grid_shift_m(picks[i], truth[i]) for i in range(n)]))
    sel, _st = reselect(picks, by_frame, {}, n, err)
    got = float(np.median([grid_shift_m(sel[i], truth[i]) for i in range(n)]))
    assert got < 0.4 * base, (base, got)
    bat, stb = batch_refine(picks, by_frame, {}, n)
    gotb = float(np.median([grid_shift_m(bat[i], truth[i]) for i in range(n)]))
    assert gotb < 0.5 * base, (base, gotb, stb)
    print(f"gsr_v10_w5 demo OK (pick {base:.3f} m -> reselect {got:.3f} / batch {gotb:.3f})")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--results", type=Path, default=Path("results/gsr_benchmark/gsr_v10_w5.json"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else DEV20
    arms = [a for a in args.arms.split(",") if a in ARMS]
    payload: dict = {}
    t0 = time.time()
    if args.accuracy:
        payload["accuracy"] = run_accuracy(args.data_dir, args.out_dir, args.cache_dir, seqs, arms)
        for a in arms:
            v = payload["accuracy"]["pooled"]["disp"][a]
            print(f"{a:<6} n={v['n']:>7} median {v['median']:.4f} p90 {v['p90']:.4f} "
                  f"mean {v['mean']:.4f} kloss {v['kloss']:.5f}")
        print(json.dumps(payload["accuracy"]["verdict"], indent=1))
    if args.positions:
        payload["positions"] = build_positions(args.out_dir, args.cache_dir, seqs, arms)
    if args.score:
        payload["score"] = score_arms(args.data_dir, args.out_dir, seqs, arms)
    if args.oracle:
        payload["oracle"] = run_oracle(args.data_dir, args.out_dir, args.cache_dir, seqs)
        p = payload["oracle"]["pooled"]["disp"]
        for k, v in p.items():
            print(f"{k:<18} n={v['n']:>7} median {v['median']:.4f} p90 {v['p90']:.4f} "
                  f"mean {v['mean']:.4f} kloss {v['kloss']:.5f}")
        print("predicted GS-HOTA:", payload["oracle"]["predicted_gs_hota"])
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
