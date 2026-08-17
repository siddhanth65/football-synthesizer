"""v10-W8 geometry stack: a learned hypothesis ranker, and the price of a lens-distortion term.

Two registered arms, both riding the v10-W5 machinery (arm D = temporal re-selection -> batch
refinement, DEV-20 flags-ON 57.17):

``--k1``      ARM K1, measure-only forecast. v10-W2 measured a ``+1.88 px`` outer-ring radial
              residual under our *gated* homography and called a per-clip distortion term "real and
              unmodelled". This asks the discriminating question first, on CPU: does that radial
              signature survive under the **GT-fitted** homography (then it is lens distortion) or
              collapse (then it was our homography's own error wearing a radial mask)? The residual
              a GT-fitted homography still leaves bounds *every* effect outside the homography model
              class, distortion included, and the v10-W4 sensitivity curve converts that bound into
              GS-HOTA -- the curve used as a **forecasting instrument** before any GPU is booked.
``--rank``    ARM RANK. PnLCalib emits ~16 hypotheses per frame and the shipped gate picks the most
              accurate one on 25.4% of frames (W5's D1). This trains a gradient-boosted ranker
              (sklearn ``HistGradientBoostingRegressor``, no new dependency) on features that are
              computable at inference from the candidate cache alone, and replaces arm D's temporal
              re-selection rule with it. Evaluated leave-one-clip-out, so no clip's selection is ever
              made by a model that saw it.

Sub-commands ``--accuracy`` / ``--positions`` / ``--score`` mirror :mod:`tools.gsr_v10_w5`: the
GT-anchored accuracy instrument, one positions parquet set per arm, and the frozen v6 chain + the
v9-W7 paired scorer. Ground truth is read only to score, to fit the oracle homographies and to
label the ranker's TRAINING clips; no arm consumes it at inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_v10_w5 import (
    CACHE_DIR,
    DEV20,
    admissible,
    batch_refine,
    foot_points_by_frame,
    kernel_loss,
    load_seq,
    predicted_gain,
    _stats,
)

logger = logging.getLogger("gsr_v10_w8")

#: Image centre and normalising radius of the 1920x1080 GSR frame (rho ~ 1 at the corners).
IMAGE_CENTRE = np.array([960.0, 540.0])
RADIUS_NORM = 1100.0
#: The W5 control's GT-anchored kernel loss on the same population -- the forecast's denominator.
CTRL_KLOSS = 0.0831
#: The W4 measured value of removing ALL displacement from the GT-fitted homography (arm ``O_acc``).
ORACLE_GS_HOTA = 6.254
#: Half-window (frames) of the temporal reference feature.
NEIGHBOUR_HALFWIN = 5
#: Arms of this session. ``ctrl``/``D`` come from W5; ``P`` is the ranker's pick, ``PB`` is
#: ``P`` -> W5's batch refinement (the direct analogue of D, with the selection rule swapped);
#: ``A`` is D under the adaptive smoothness prior (amendment 1), ``AB`` is ``P`` under it.
ARMS = ("ctrl", "D", "P", "PB", "A", "AB")
#: Feature block handed to the ranker, in order (NaN is fed through -- HistGBT handles it natively).
FEATURES = (
    "rank", "log_err_m", "rep_px", "n_points", "mode_idx", "ransac",
    "pool_size", "n_admissible", "err_rank_frac", "rep_rank_frac",
    "d_pool_median_m", "d_gate_pick_m", "d_neighbour_m", "neighbour_n",
    "fy_over_fx", "tilt_deg", "log_focal", "cam_height_m", "cam_dist_m",
    "onpitch_frac", "n_feet", "grid_span_m",
)


# === ARM K1: is the radial residual real, and what is it worth? ==================================
def undistort(pts: np.ndarray, k1: float) -> np.ndarray:
    """One-parameter radial point warp about the image centre (pure).

    Args:
        pts: ``(N, 2)`` image points.
        k1: Radial coefficient on ``rho = r / RADIUS_NORM`` (so ``k1`` is the fractional radial
            displacement at the frame corner).

    Returns:
        ``(N, 2)`` warped points; ``k1 = 0`` is the identity.
    """
    d = (np.asarray(pts, dtype=float).reshape(-1, 2) - IMAGE_CENTRE) / RADIUS_NORM
    r2 = (d ** 2).sum(axis=1, keepdims=True)
    return IMAGE_CENTRE + RADIUS_NORM * d * (1.0 + k1 * r2)


def radial_profile(res: np.ndarray, n_bins: int = 6) -> list[dict]:
    """Median radial image residual (px) by image-radius sextile (pure).

    Args:
        res: :func:`tools.gsr_v10_w2_calib.image_residuals` output ``(N, 4)``.
        n_bins: Number of equal-count radius bins.

    Returns:
        One record per bin, outermost last -- the ``+1.88 px`` W2 number is the last bin's.
    """
    if not len(res):
        return []
    edges = np.quantile(res[:, 2], np.linspace(0, 1, n_bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (res[:, 2] >= lo) & (res[:, 2] <= hi)
        out.append({"r_lo": float(lo), "r_hi": float(hi), "n": int(m.sum()),
                    "median_radial_px": float(np.median(res[m, 3]))})
    return out


def run_k1(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str],
           k_grid: np.ndarray | None = None) -> dict:
    """Price the distortion term against its own ceiling, with the W4 curve as the forecaster.

    For every DEV frame with enough GT people this fits the **best homography** through the GT
    correspondences (least-squares DLT, no RANSAC) for each ``k1`` on the grid, and reports the
    residual it cannot remove. Three references are produced:

    * the radial residual profile under our gated homography (reproduces W2's ``+1.88 px``),
    * the same profile under the GT-fitted homography (the discriminating measurement),
    * the same profile under arm D (what the W5 accuracy arm already did to that residual).

    Returns:
        Per-clip optimal ``k1``, pooled kernel losses over the grid, the radial profiles, and the
        curve-forecast GS-HOTA of each variant.
    """
    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from tools.gsr_v10_w2_calib import image_residuals  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    ks = ([round(float(v), 4) for v in k_grid] if k_grid is not None
          else [round(float(v), 4) for v in np.concatenate(
              [np.arange(-0.06, -0.012, 0.004), np.arange(-0.012, 0.0121, 0.001),
               np.arange(0.016, 0.0601, 0.004)])])
    pooled: dict[float, list[np.ndarray]] = {k: [] for k in ks}
    best_k: dict[str, float] = {}
    res_ctrl, res_gtfit, res_d = [], [], []
    for seq in seqs:
        gt = gt_people(data_dir / seq)
        homs, _st = arm_homographies("ctrl", seq, out_dir, cache_dir)
        res_ctrl.append(image_residuals(homs, gt))
        homs_d, _st = arm_homographies("D", seq, out_dir, cache_dir)
        res_d.append(image_residuals(homs_d, gt))
        clip: dict[float, list[np.ndarray]] = {k: [] for k in ks}
        gtfit: dict[int, np.ndarray] = {}
        for fr, g in sorted(gt.items()):
            if len(g) < 6:
                continue
            for k in ks:
                q = undistort(g[:, :2], k)
                h = estimate_homography(q, g[:, 2:4], use_cv2=False)
                if h is None or not np.isfinite(h).all():
                    continue
                clip[k].append(np.linalg.norm(apply_homography(h, q) - g[:, 2:4], axis=1))
                if k == 0.0:
                    gtfit[fr] = h
        cat = {k: np.concatenate(v) for k, v in clip.items() if v}
        best_k[seq] = float(min(cat, key=lambda k: kernel_loss(cat[k])))
        res_gtfit.append(image_residuals(gtfit, gt))
        for k in ks:
            pooled[k].append(cat[k])
        logger.info("%s: k1* %+.4f, kloss %.5f -> %.5f", seq, best_k[seq],
                    kernel_loss(cat[0.0]), kernel_loss(cat[best_k[seq]]))
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    per_clip = np.concatenate([pooled[best_k[s]][i] for i, s in enumerate(seqs)])
    k0 = kernel_loss(cat[0.0])
    kg = min(kernel_loss(cat[k]) for k in ks)
    kp = kernel_loss(per_clip)
    prof = {}
    for name, chunks in (("ctrl_gated", res_ctrl), ("gtfit", res_gtfit), ("arm_D", res_d)):
        a = np.concatenate([c for c in chunks if len(c)])
        a = a[np.linalg.norm(a[:, :2], axis=1) < 200.0]  # W2's rule, same for every reference
        prof[name] = {"n": int(len(a)), "median_radial_px": float(np.median(a[:, 3])),
                      "bins": radial_profile(a)}
    out = {
        "k_grid": ks, "best_k1_per_clip": best_k,
        "pooled_kloss": {str(k): kernel_loss(cat[k]) for k in ks},
        "pooled_median_m": {str(k): float(np.median(cat[k])) for k in ks},
        "gtfit": {"kloss": k0, "median_m": float(np.median(cat[0.0])), "n": int(len(cat[0.0]))},
        "global_k1": {"k1": float(min(ks, key=lambda k: kernel_loss(cat[k]))), "kloss": kg},
        "per_clip_k1": {"kloss": kp, "median_m": float(np.median(per_clip))},
        "radial_profiles": prof,
        "forecast_gs_hota": {
            "whole_gtfit_residual": ORACLE_GS_HOTA * k0 / CTRL_KLOSS,
            "global_radial_term": ORACLE_GS_HOTA * (k0 - kg) / CTRL_KLOSS,
            "per_clip_radial_term_gt_fitted": ORACLE_GS_HOTA * (k0 - kp) / CTRL_KLOSS},
    }
    out["bar_pass"] = bool(out["forecast_gs_hota"]["per_clip_radial_term_gt_fitted"] >= 0.5)
    return out


# === ARM RANK: features, labels, model ===========================================================
def _pose_features(h: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(fy/fx, tilt_deg, log_f, camera height, camera distance)`` of a mapping, NaN if degenerate."""
    from generator.camera_track import decompose  # noqa: PLC0415

    p = decompose(h)
    if p is None:
        return (np.nan,) * 5
    fx, fy = p.focals()
    return (fy / fx, float(np.degrees(p.tilt)), p.log_f, p.cz,
            float(np.hypot(p.cx - 52.5, p.cy - 34.0)))


def _grids(homs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Each mapping's image of ``CHECK_GRID`` -- the space every distance below is measured in."""
    from generator.calibrate import apply_homography  # noqa: PLC0415
    from generator.camera_track import CHECK_GRID  # noqa: PLC0415

    return {f: apply_homography(h, CHECK_GRID) for f, h in homs.items()}


def _neighbour_reference(grids: dict[int, np.ndarray], half: int = NEIGHBOUR_HALFWIN
                         ) -> dict[int, tuple[np.ndarray, int]]:
    """``{frame: (median grid of the +-half neighbours' picks, count)}`` -- frame itself excluded."""
    frames = sorted(grids)
    out: dict[int, tuple[np.ndarray, int]] = {}
    for f in frames:
        near = [grids[g] for g in range(f - half, f + half + 1) if g != f and g in grids]
        out[f] = (np.median(np.stack(near), axis=0), len(near)) if near else (None, 0)
    return out


def frame_features(cands: list, adm: list[int], gate_grid: np.ndarray | None,
                   ref: tuple[np.ndarray, int], foot: np.ndarray | None) -> np.ndarray:
    """``(len(adm), len(FEATURES))`` feature block for one frame's admissible hypotheses (pure).

    Every column is computable at inference: the cached hypothesis metadata, the geometry of the
    hypothesis itself, the frame's own detections, and the neighbouring frames' *gate* picks.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415
    from generator.camera_track import CHECK_GRID  # noqa: PLC0415

    _MODES = ("full", "ground_plane", "main")
    grids = {i: apply_homography(cands[i].homography, CHECK_GRID) for i in adm}
    pool_med = np.median(np.stack([grids[i] for i in adm]), axis=0)
    errs = np.array([cands[i].error_m for i in adm])
    reps = np.array([cands[i].rep_err_px for i in adm])
    rows = []
    for i in adm:
        c, g = cands[i], grids[i]
        pose = _pose_features(c.homography)
        if foot is not None and len(foot):
            p = apply_homography(c.homography, foot)
            on = float(((p[:, 0] >= 0) & (p[:, 0] <= 105) & (p[:, 1] >= 0) & (p[:, 1] <= 68)).mean())
        else:
            on = np.nan
        rows.append([
            float(i),  # the cache is stored in PnLCalib's own preference order
            float(np.log1p(max(c.error_m, 0.0))), float(c.rep_err_px), float(c.n_points),
            float(_MODES.index(c.mode)), float(c.use_ransac),
            float(len(cands)), float(len(adm)),
            float((errs <= c.error_m).mean()), float((reps <= c.rep_err_px).mean()),
            float(np.median(np.linalg.norm(g - pool_med, axis=1))),
            (float(np.median(np.linalg.norm(g - gate_grid, axis=1)))
             if gate_grid is not None else np.nan),
            (float(np.median(np.linalg.norm(g - ref[0], axis=1))) if ref[1] else np.nan),
            float(ref[1]), *pose, on,
            float(len(foot)) if foot is not None else 0.0,
            float(np.linalg.norm(g.max(axis=0) - g.min(axis=0))),
        ])
    return np.asarray(rows, dtype=float)


#: Pixel-evidence columns (built separately -- they are the only features that read the image).
LINE_FEATURES = ("line_support", "line_dist_med_px", "line_dist_p75_px", "line_visible",
                 "line_support_rank_frac", "line_dist_rank_frac")


def line_evidence(h: np.ndarray, dist: np.ndarray, samples: np.ndarray,
                  radius_px: float = 4.0) -> tuple[float, float, float, int]:
    """``(support, median, p75 ridge distance px, n visible)`` of a mapping's projected pitch model.

    :func:`generator.line_calib.line_support` thresholds at ``radius_px`` and saturates near 0.95 for
    every plausible hypothesis, which is too coarse to rank sub-metre differences; the raw distance
    quantiles off the same ridge field are not.
    """
    from generator.line_calib import _project_to_image  # noqa: PLC0415

    img = _project_to_image(h, samples)
    if img is None:
        return 0.0, np.nan, np.nan, 0
    hgt, wid = dist.shape
    x, y = np.round(img[:, 0]).astype(int), np.round(img[:, 1]).astype(int)
    inside = (x >= 0) & (x < wid) & (y >= 0) & (y < hgt)
    if not inside.sum():
        return 0.0, np.nan, np.nan, 0
    d = dist[y[inside], x[inside]]
    return (float((d <= radius_px).mean()), float(np.median(d)),
            float(np.percentile(d, 75)), int(inside.sum()))


def build_line_features(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str],
                        frames: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (seq, frame, candidate) pixel evidence: the ridge field is computed once per frame."""
    import cv2  # noqa: PLC0415

    from generator.line_calib import line_mask, pitch_model_lines, ridge_field  # noqa: PLC0415
    from generator.line_calib import sample_model_points  # noqa: PLC0415

    samples, _ = sample_model_points(pitch_model_lines())
    rows = []
    for seq in seqs:
        df, by_frame, homs, _err, _n = load_seq(seq, out_dir, cache_dir)
        feet = foot_points_by_frame(df)
        want = (set(frames.loc[frames["seq"] == seq, "frame"].astype(int)) if frames is not None
                else set(homs))
        t0 = time.time()
        for fr in sorted(want & set(homs)):
            img = cv2.imread(str(data_dir / seq / "img1" / f"{fr + 1:06d}.jpg"))
            if img is None:
                continue
            dist, _n2 = ridge_field(line_mask(img))
            cands = by_frame.get(fr) or []
            for i in admissible(cands, feet.get(fr)):
                s, med, p75, vis = line_evidence(cands[i].homography, dist, samples)
                rows.append((seq, fr, i, s, med, p75, vis))
        logger.info("%s: %d rows, %.0f s", seq, len(rows), time.time() - t0)
    tab = pd.DataFrame(rows, columns=["seq", "frame", "cand", "line_support",
                                      "line_dist_med_px", "line_dist_p75_px", "line_visible"])
    g = tab.groupby(["seq", "frame"])
    tab["line_support_rank_frac"] = g["line_support"].rank(pct=True, ascending=False)
    tab["line_dist_rank_frac"] = g["line_dist_med_px"].rank(pct=True)
    return tab


def build_dataset(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str]) -> pd.DataFrame:
    """One row per (frame, admissible hypothesis): features + the GT-anchored label.

    The label is the median displacement (m) of that hypothesis from the frame's **GT-fitted**
    homography on the GT people -- the quantity the W4 curve prices, and exactly W5's oracle
    ranking key. It is used for training and evaluation only.
    """
    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    frames_out = []
    for seq in seqs:
        df, by_frame, homs, _err, _n = load_seq(seq, out_dir, cache_dir)
        feet = foot_points_by_frame(df)
        gt = gt_people(data_dir / seq)
        gate_grids = _grids(homs)
        ref = _neighbour_reference(gate_grids)
        rows, meta = [], []
        for fr in sorted(homs):
            g = gt.get(fr)
            if g is None or len(g) < 6:
                continue
            h_gt = estimate_homography(g[:, :2], g[:, 2:4])
            if h_gt is None or not np.isfinite(h_gt).all():
                continue
            cands = by_frame.get(fr) or []
            adm = admissible(cands, feet.get(fr))
            if not cands:
                continue
            ref_fit = apply_homography(h_gt, g[:, :2])
            lab = []
            for c in cands:
                p = apply_homography(c.homography, g[:, :2])
                lab.append(float(np.median(np.linalg.norm(p - ref_fit, axis=1))))
            lab = np.asarray(lab)
            pool_best = float(lab.min())
            gate_i = next((i for i, c in enumerate(cands) if c.homography is homs[fr]), -1)
            if not adm:
                continue
            feat = frame_features(cands, adm, gate_grids.get(fr), ref[fr], feet.get(fr))
            for j, i in enumerate(adm):
                rows.append(feat[j])
                meta.append((seq, fr, i, lab[i], pool_best, int(i == gate_i), len(cands)))
        if not rows:
            continue
        tab = pd.DataFrame(np.asarray(rows), columns=list(FEATURES))
        m = pd.DataFrame(meta, columns=["seq", "frame", "cand", "err_m_label", "pool_best_m",
                                        "is_gate_pick", "pool_size_all"])
        frames_out.append(pd.concat([m, tab], axis=1))
        logger.info("%s: %d rows over %d frames", seq, len(m), m["frame"].nunique())
    return pd.concat(frames_out, ignore_index=True)


def _cols(tab: pd.DataFrame) -> list[str]:
    """Feature columns present in a table: the cache block, plus pixel evidence when it was built."""
    return [c for c in (*FEATURES, *LINE_FEATURES) if c in tab.columns]


def _model():
    """The ranker: a gradient-boosted regressor on ``log1p(displacement)`` (no new dependency)."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=20260817)


def _selection_stats(tab: pd.DataFrame, score_col: str, tol: float = 1e-6) -> dict:
    """Selection accuracy / mean picked-accuracy of a scoring column, per frame (pure)."""
    picks = tab.loc[tab.groupby(["seq", "frame"])[score_col].idxmin()]
    is_best = (picks["err_m_label"] <= picks["pool_best_m"] + tol).to_numpy()
    rank = tab.groupby(["seq", "frame"])["err_m_label"].rank(method="min")
    rank_of_pick = rank.loc[picks.index].to_numpy()
    return {"frames": int(len(picks)), "accuracy": float(is_best.mean()),
            "median_rank": float(np.median(rank_of_pick)),
            "median_err_m": float(picks["err_m_label"].median()),
            "mean_err_m": float(picks["err_m_label"].mean()),
            "kloss_proxy": kernel_loss(picks["err_m_label"].to_numpy())}


def _target(tab: pd.DataFrame, centred: bool) -> np.ndarray:
    """Regression target: ``log1p(displacement)``, optionally centred within its own frame.

    Uncentred, the target is dominated by how hard the FRAME is (every hypothesis of a bad frame is
    bad), which is not the quantity the selector needs. Centring per frame turns the regressor into
    a within-frame ranker without leaving sklearn.
    """
    y = np.log1p(tab["err_m_label"].to_numpy())
    if not centred:
        return y
    return y - tab.assign(_y=y).groupby(["seq", "frame"])["_y"].transform("mean").to_numpy()


def loco_fit(tab: pd.DataFrame, seqs: list[str], *, centred: bool = True
             ) -> tuple[pd.DataFrame, dict]:
    """Leave-one-clip-out out-of-fold ranker scores + the selection read against the gate baseline.

    No clip's hypotheses are ever scored by a model that trained on that clip, so the selection
    accuracy below is a held-out number even though every clip is in DEV.
    """
    tab = tab.copy()
    tab["pred"] = np.nan
    cols = _cols(tab)
    y = _target(tab, centred)
    for seq in seqs:
        te = (tab["seq"] == seq).to_numpy()
        if not te.any():
            continue
        m = _model()
        m.fit(tab.loc[~te, cols].to_numpy(), y[~te])
        tab.loc[te, "pred"] = m.predict(tab.loc[te, cols].to_numpy())
    gate = tab[tab["is_gate_pick"] == 1]
    baseline = {"frames": int(len(gate)),
                "accuracy": float((gate["err_m_label"] <= gate["pool_best_m"] + 1e-6).mean()),
                "median_err_m": float(gate["err_m_label"].median()),
                "mean_err_m": float(gate["err_m_label"].mean()),
                "kloss_proxy": kernel_loss(gate["err_m_label"].to_numpy())}
    rep = {"gate_baseline": baseline, "ranker_loco": _selection_stats(tab, "pred"),
           "pool_oracle": {"median_err_m": float(tab.groupby(["seq", "frame"])["pool_best_m"]
                                                 .first().median())},
           "per_seq": {}}
    for seq in seqs:
        s = tab[tab["seq"] == seq]
        if not len(s):
            continue
        g = s[s["is_gate_pick"] == 1]
        rep["per_seq"][seq] = {
            "gate_accuracy": float((g["err_m_label"] <= g["pool_best_m"] + 1e-6).mean()),
            "ranker_accuracy": _selection_stats(s, "pred")["accuracy"],
            "gate_median_m": float(g["err_m_label"].median()),
            "ranker_median_m": _selection_stats(s, "pred")["median_err_m"]}
    rep["bar_pass_accuracy"] = bool(rep["ranker_loco"]["accuracy"] >= 0.50)
    return tab, rep


def fit_full(tab: pd.DataFrame, dest: Path) -> Path:
    """Fit one ranker on every row and pickle it (used only for clips outside the LOCO table)."""
    import pickle  # noqa: PLC0415

    m = _model()
    cols = _cols(tab)
    m.fit(tab[cols].to_numpy(), _target(tab, True))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        pickle.dump({"model": m, "features": cols}, fh)
    return dest


# === the arms ====================================================================================
def rank_select(seq: str, out_dir: Path, cache_dir: Path, scores: pd.DataFrame
                ) -> tuple[dict[int, np.ndarray], dict]:
    """Replace the gate's pick by the ranker's, per frame (admissible hypotheses only)."""
    df, by_frame, homs, _err, _n = load_seq(seq, out_dir, cache_dir)
    s = scores[scores["seq"] == seq]
    best = s.loc[s.groupby("frame")["pred"].idxmin()].set_index("frame")["cand"].to_dict()
    out, changed = {}, 0
    for fr, h in homs.items():
        i = best.get(fr)
        cands = by_frame.get(fr) or []
        if i is None or int(i) >= len(cands):
            out[fr] = h
            continue
        out[fr] = cands[int(i)].homography
        changed += int(out[fr] is not h)
    return out, {"solved": len(homs), "reranked": changed, "scored_frames": len(best)}


def stiffness_profile(homs: dict[int, np.ndarray], by_frame: dict[int, list],
                      feet: dict[int, np.ndarray], n_frames: int) -> np.ndarray:
    """Arm A's per-frame multiplier on the batch solve's second-difference penalty (GT-free).

    v10-W7 measured the batch arm's one loss (SNGS-021) to be 83% a single 50-frame window at the
    clip's fastest pan: with lambda ~4283 against a data weight ~12 the prior cannot follow camera
    acceleration, while the hypothesis cloud there was tight and correct. This relaxes the prior
    exactly where those two conditions hold, by the rule registered before it was written:

    ``w_f = clip((cloud_f / median cloud) * (median accel / accel_f), 0.02, 1.0)``

    with ``cloud_f`` the median distance (m) of the frame's admissible hypotheses from their own
    pool median, and ``accel_f`` the second difference of the pick series. The cap at 1.0 makes this
    a pure relaxation: never stiffer than the on-record arm anywhere.

    Edge case, left as measured: a frame whose admissible hypotheses are *identical* has
    ``cloud_f = 0`` and lands on the 0.02 floor -- which is the rule's own semantics (a unanimous
    pool is evidence to follow, not to smooth). On DEV-20 that is 21.6% of SNGS-021's frames and
    55.4% of SNGS-081's, where the clip median cloud is itself 0 and the profile becomes binary.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415
    from generator.camera_track import CHECK_GRID  # noqa: PLC0415

    ng = len(CHECK_GRID)
    grid = np.full((n_frames, ng, 2), np.nan)
    cloud = np.full(n_frames, np.nan)
    for fr, h in homs.items():
        grid[fr] = apply_homography(h, CHECK_GRID)
        adm = admissible(by_frame.get(fr) or [], feet.get(fr))
        if not adm:
            continue
        rows = np.stack([apply_homography(by_frame[fr][i].homography, CHECK_GRID) for i in adm])
        med = np.median(rows, axis=0)
        cloud[fr] = float(np.median(np.linalg.norm(rows - med, axis=2)))
    accel = np.full(n_frames, np.nan)
    ok = np.isfinite(grid.reshape(n_frames, -1)).all(axis=1)
    for f in range(1, n_frames - 1):
        if ok[f - 1] and ok[f] and ok[f + 1]:
            accel[f] = float(np.median(np.linalg.norm(
                grid[f - 1] - 2.0 * grid[f] + grid[f + 1], axis=1)))
    c_med = np.nanmedian(cloud) if np.isfinite(cloud).any() else 1.0
    a_med = np.nanmedian(accel) if np.isfinite(accel).any() else 1.0
    c = np.where(np.isfinite(cloud), cloud, c_med) / max(c_med, 1e-9)
    a = max(a_med, 1e-9) / np.maximum(np.where(np.isfinite(accel), accel, a_med), 1e-9)
    return np.clip(c * a, 0.02, 1.0)


def arm_homographies(arm: str, seq: str, out_dir: Path, cache_dir: Path,
                     scores: pd.DataFrame | None = None) -> tuple[dict[int, np.ndarray], dict]:
    """``({frame: H}, stats)`` for one W8 arm (``ctrl``/``D`` delegate to v10-W5 unchanged)."""
    import tools.gsr_v10_w5 as w5  # noqa: PLC0415

    if arm in ("ctrl", "D"):
        return w5.arm_homographies(arm, seq, out_dir, cache_dir)
    df, by_frame, homs, err, n = load_seq(seq, out_dir, cache_dir)
    feet = foot_points_by_frame(df)
    if arm == "A":  # D's temporal re-selection, batch-refined under the adaptive prior
        picks, st = w5.reselect(homs, by_frame, feet, n, err)
        out, st2 = batch_refine(picks, by_frame, feet, n,
                                stiffness=stiffness_profile(picks, by_frame, feet, n))
        return out, st | st2
    if scores is None:
        raise ValueError(f"arm {arm!r} needs the ranker scores table")
    picks, st = rank_select(seq, out_dir, cache_dir, scores)
    if arm == "P":
        return picks, st
    if arm in ("PB", "AB"):
        stiff = stiffness_profile(picks, by_frame, feet, n) if arm == "AB" else None
        out, st2 = batch_refine(picks, by_frame, feet, n, stiffness=stiff)
        return out, st | st2
    raise ValueError(f"unknown arm {arm!r}")


def run_accuracy(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str], arms: list[str],
                 scores: pd.DataFrame | None) -> dict:
    """GT-anchored accuracy per arm, paired on the frames every arm answers (W5's instrument)."""
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
            hs[a], arm_stats[a][seq] = arm_homographies(a, seq, out_dir, cache_dir, scores)
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
        out["verdict"][a] = {
            "median_improvement": round(1.0 - cur["median"] / ctrl["median"], 4),
            "kloss_improvement": round(1.0 - cur["kloss"] / ctrl["kloss"], 4),
            "predicted_gs_hota": round(predicted_gain(ctrl["kloss"], cur["kloss"]), 3),
            "rows_better": int((d < 0).sum()), "rows_worse": int((d > 0).sum()),
            "wilcoxon_p": (float(wilcoxon(d).pvalue) if len(d) and np.any(d != 0) else 1.0)}
    return out


# === positions and scoring =======================================================================
def build_positions(out_dir: Path, cache_dir: Path, seqs: list[str], arms: list[str],
                    scores: pd.DataFrame | None) -> dict:
    """One positions parquet set per arm: arm homographies -> reproject -> ``fill_calibration_gaps``."""
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415
    from tools.gsr_v10_w5 import FILL_GAP, POSITIONS_SUBDIR  # noqa: PLC0415
    from tools.gsr_w2_calibswap import swap_positions  # noqa: PLC0415

    stats: dict[str, dict] = {a: {} for a in arms}
    for seq in seqs:
        df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
        for a in arms:
            homs, st = arm_homographies(a, seq, out_dir, cache_dir, scores)
            tab, sw = swap_positions(df, homs, origin=(0.0, 0.0), mode="t1")
            tab = fill_calibration_gaps(tab, max_gap=FILL_GAP)
            sub = out_dir / f"positions_v10w8{a}_v6det"
            sub.mkdir(parents=True, exist_ok=True)
            tab.to_parquet(sub / f"{seq}.parquet", index=False)
            st.update({k: sw[k] for k in ("pitch_rows_control", "pitch_rows_final")})
            st["pitch_rows_after_fill"] = int(np.isfinite(tab["pitch_x"]).sum())
            stats[a][seq] = st
        logger.info("%s: %s", seq, {a: stats[a][seq]["pitch_rows_after_fill"] for a in arms})
    return stats


def score_arms(data_dir: Path, out_dir: Path, seqs: list[str], arms: list[str]) -> dict:
    """The frozen v6 chain per arm, then the v9-W7 paired scorer against **arm D** (the incumbent).

    ``D``'s own submission is the on-record ``deleak_v10w5_D``; it is re-run here only if absent.
    """
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v6det import TAU  # noqa: PLC0415
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache_v6det"
    raw: dict[str, dict] = {}
    for a in arms:
        res = eiou.run_point(data_dir, out_dir, seqs,
                             eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                             embedder="clip_v6det", tau=TAU, tag=f"v10w8_{a}",
                             percrop_variant="_v6_v6det",
                             positions_subdir=f"positions_v10w8{a}_v6det")
        h = res["gs_hota"]
        raw[a] = {"gs_hota": h, "gs_hota_per_seq": res["gs_hota_per_seq"]}
        print(f"{a:<6} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
              f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f}", flush=True)
    base = out_dir / "deleak_v10w5_D" / "predictions" / "data"
    paired = {a: score_pair(base, out_dir / f"deleak_v10w8_{a}" / "predictions" / "data",
                            data_dir, out_dir / "v10_w8" / "pair" / a, seqs)
              for a in arms}
    return {"arms": raw, "paired_vs_D": paired}


# === self-check ==================================================================================
def _demo() -> None:
    """Assert the pure seams: the radial warp, the feature block and the selection statistic."""
    from generator.calibrate import CalibCandidate, estimate_homography  # noqa: PLC0415

    p = np.array([[960.0, 540.0], [1920.0, 1080.0]])
    assert np.allclose(undistort(p, 0.05)[0], [960.0, 540.0]), "centre is a fixed point"
    d = np.linalg.norm(undistort(p, 0.05)[1] - IMAGE_CENTRE) / np.linalg.norm(p[1] - IMAGE_CENTRE)
    assert abs(d - (1.0 + 0.05 * (np.linalg.norm(p[1] - IMAGE_CENTRE) / RADIUS_NORM) ** 2)) < 1e-9
    assert np.allclose(undistort(p, 0.0), p)

    # a real radial warp must be recoverable by the scan the K1 arm runs
    img = np.column_stack([np.random.default_rng(0).uniform(100, 1800, 60),
                           np.random.default_rng(1).uniform(100, 1000, 60)])
    h = np.array([[0.05, 0.002, -5.0], [0.001, 0.055, -20.0], [1e-6, 8e-5, 1.0]])
    from generator.calibrate import apply_homography  # noqa: PLC0415

    pitch = apply_homography(h, img)
    warped = undistort(img, -0.03)  # the "camera" that produced the pixels
    losses = {}
    for k in (-0.06, -0.03, 0.0, 0.03):
        q = undistort(warped, k)
        hh = estimate_homography(q, pitch, use_cv2=False)
        losses[k] = float(np.median(np.linalg.norm(apply_homography(hh, q) - pitch, axis=1)))
    assert min(losses, key=losses.get) == 0.03, losses  # inverse warp wins, and by a lot
    assert losses[0.03] < 0.10 * losses[0.0], losses

    # the feature block: one row per admissible candidate, no NaN in the cache-only columns
    good = np.array([[0.06, 0.004, -8.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])
    cands = [CalibCandidate(good, 0.4, 8, "full", 0.0, 1.0),
             CalibCandidate(good * 1.001, 0.9, 8, "main", 5.0, 3.0)]
    feet = np.column_stack([np.linspace(300, 1500, 5), np.full(5, 500.0)])
    f = frame_features(cands, [0, 1], None, (None, 0), feet)
    assert f.shape == (2, len(FEATURES)), f.shape
    assert np.isfinite(f[:, :FEATURES.index("d_gate_pick_m")]).all(), f

    tab = pd.DataFrame({"seq": ["a"] * 4, "frame": [0, 0, 1, 1], "cand": [0, 1, 0, 1],
                        "err_m_label": [0.5, 0.2, 0.3, 0.9], "pool_best_m": [0.2, 0.2, 0.3, 0.3],
                        "is_gate_pick": [1, 0, 1, 0], "pred": [1.0, 0.0, 0.0, 1.0]})
    st = _selection_stats(tab, "pred")
    assert st["frames"] == 2 and st["accuracy"] == 1.0, st

    # arm A: a constant-velocity pan must keep the on-record stiffness (cap 1.0) everywhere, and an
    # accelerating window must be relaxed -- and stiffness=1 must reproduce the uniform prior.
    from generator.camera_track import Pose, compose  # noqa: PLC0415

    n = 90
    rng2 = np.random.default_rng(3)
    picks, by_frame = {}, {}
    for i in range(n):
        pan = 0.30 + 0.0010 * i + (0.00025 * (i - 45) ** 2 if 40 <= i <= 50 else 0.0)
        p = Pose(pan, 1.85, 0.02, float(np.log(3800.0)), float(np.log(0.95)), 2.0, -55.0, 15.0)
        picks[i] = compose(p)
        # a pool with real spread: a unanimous pool is a legitimate input but leaves cloud == 0,
        # which the rule reads (correctly) as "trust the data here" -- see the docstring.
        by_frame[i] = [CalibCandidate(compose(Pose(p.pan + rng2.normal(0, 3e-4), p.tilt, p.roll,
                                                   p.log_f, p.log_aspect, p.cx, p.cy, p.cz)),
                                      0.4, 20, "full", 0.0, 2.0) for _ in range(4)]
    w = stiffness_profile(picks, by_frame, {}, n)
    assert w.max() <= 1.0 and w[5] == 1.0, w[:10]
    assert w[45] < 0.5, (w[43:48], w[5])
    a1, _s1 = batch_refine(picks, by_frame, {}, n)
    a2, _s2 = batch_refine(picks, by_frame, {}, n, stiffness=np.ones(n))
    assert max(float(np.abs(a1[f] - a2[f]).max()) for f in a1) < 1e-9, "unit stiffness must be a no-op"
    print("gsr_v10_w8 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--results", type=Path, default=Path("results/gsr_benchmark/gsr_v10_w8.json"))
    ap.add_argument("--work", type=Path, default=Path("outputs/gsr/v10_w8"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--arms", default="ctrl,D,P,PB,A,AB")
    ap.add_argument("--k1", action="store_true")
    ap.add_argument("--dataset", action="store_true", help="build the ranker feature table")
    ap.add_argument("--line", action="store_true", help="build the pixel-evidence features (CPU)")
    ap.add_argument("--rank", action="store_true", help="LOCO fit + selection read")
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
    args.work.mkdir(parents=True, exist_ok=True)
    ds_path, sc_path = args.work / "rank_dataset.parquet", args.work / "rank_scores.parquet"
    ln_path = args.work / "line_features.parquet"
    payload: dict = {}
    t0 = time.time()

    if args.k1:
        payload["k1"] = run_k1(args.data_dir, args.out_dir, args.cache_dir, seqs)
        print(json.dumps(payload["k1"]["forecast_gs_hota"], indent=1))
        for name, p in payload["k1"]["radial_profiles"].items():
            print(f"{name:<11} median {p['median_radial_px']:+.3f} px  bins "
                  f"{[round(b['median_radial_px'], 2) for b in p['bins']]}")
    if args.dataset:
        tab = build_dataset(args.data_dir, args.out_dir, args.cache_dir, seqs)
        tab.to_parquet(ds_path, index=False)
        print(f"dataset {len(tab)} rows -> {ds_path}")
    if args.line:
        tab = build_line_features(args.data_dir, args.out_dir, args.cache_dir, seqs,
                                  pd.read_parquet(ds_path)[["seq", "frame"]])
        tab.to_parquet(ln_path, index=False)
        print(f"line features {len(tab)} rows -> {ln_path}")
    if args.rank:
        tab = pd.read_parquet(ds_path)
        if ln_path.exists():
            tab = tab.merge(pd.read_parquet(ln_path), on=["seq", "frame", "cand"], how="left")
            print(f"merged pixel evidence ({tab['line_dist_med_px'].isna().mean():.3f} missing)")
        scored, rep = loco_fit(tab, seqs)
        scored[["seq", "frame", "cand", "pred", "err_m_label", "pool_best_m",
                "is_gate_pick"]].to_parquet(sc_path, index=False)
        fit_full(tab, args.work / "ranker_full.pkl")
        payload["rank"] = rep
        print(json.dumps({k: v for k, v in rep.items() if k != "per_seq"}, indent=1))
    scores = pd.read_parquet(sc_path) if sc_path.exists() else None
    if args.accuracy:
        payload["accuracy"] = run_accuracy(args.data_dir, args.out_dir, args.cache_dir, seqs,
                                           arms, scores)
        for a in arms:
            v = payload["accuracy"]["pooled"]["disp"][a]
            print(f"{a:<6} n={v['n']:>7} median {v['median']:.4f} p90 {v['p90']:.4f} "
                  f"mean {v['mean']:.4f} kloss {v['kloss']:.5f}")
        print(json.dumps(payload["accuracy"]["verdict"], indent=1))
    if args.positions:
        payload["positions"] = build_positions(args.out_dir, args.cache_dir, seqs, arms, scores)
    if args.score:
        payload["score"] = score_arms(args.data_dir, args.out_dir, seqs,
                                      [a for a in arms if a not in ("ctrl", "D")])
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
