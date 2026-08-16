"""v10-W2: does PnLCalib's integer heatmap decode cost us pitch accuracy, and what is the fix worth?

The audit behind this tool (``results/gsr_v10_w2_registered.json``) found exactly two live defects in
our calibration stack, both quantisation, both in the decode of the two HRNet heatmaps:

* **peaks**: ``utils_heatmap.get_keypoints_from_heatmap_batch_maxpool`` returns the integer arg-max
  cell times 2, so every keypoint and every line endpoint sits on a 2 px grid of the 960x540 network
  input -- **4 px of a 1920x1080 broadcast frame**.
* **derived keypoints**: ``utils_heatmap.complete_keypoints`` rounds every line-intersection keypoint
  to a whole pixel of that same canvas (``round(new_kp[0], 0)``) -- 2 px at native resolution.

Both are fixed behind default-off flags on :class:`generator.calibrate.PnLCalibCalibrator`
(``subpix_decode``, ``derived_kp_scale``). This tool measures what they are worth:

``--cache``      GPU. One forward pass per frame, decoded under every arm, all 18 PnLCalib camera
                 hypotheses cached per arm (so the control is the SAME window, same weights, same
                 heatmaps, same RANSAC seed -- only the decode differs).
``--accuracy``   CPU. Homography-only GT-anchored error: GT foot points pushed through each arm's
                 gate-selected homography against that annotation's own ``bbox_pitch``. This
                 isolates calibration from detection, which the v8-W2 detection-anchored 0.475 m
                 could not.
``--positions``  CPU. Gate + fill per arm -> positions parquets (detections, tracks, embeddings and
                 OCR evidence are byte-identical across arms; only the pitch coordinate moves).
``--score``      CPU. The frozen v6 chain per arm, then the v9-W7 paired scorer flags ON and OFF.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_v10_w2_calib")

#: ``{arm: (subpix_decode, derived_kp_scale, half_cell_offset)}``. ``ctrl`` is the stock decode.
ARMS: dict[str, tuple[bool, int, bool]] = {
    "ctrl": (False, 1, False),
    "s1": (True, 1, False),    # sub-cell peaks only
    "s2": (False, 8, False),   # un-rounded line-intersection keypoints only
    "s3": (True, 8, False),    # s1 + s2 composed
    "s4": (False, 1, True),    # half-cell de-bias of the floored labels
    "s5": (False, 8, True),    # s4 + s2 composed
}
#: PnLCalib's voting order is deterministic apart from ``cv2.findHomography(..., RANSAC)`` inside
#: ``get_per_plane_correspondences``; seeding per frame keeps the arms' RANSAC draws comparable.
RNG_SEED = 20260815
#: Artifact suffix of the v6 (S4b-detector) lineage every arm rides, as in ``tools.gsr_w2_calibswap``.
VARIANT = "_v6det"
#: Our shipped positions for that lineage -- the source of foot points and of every non-pitch column.
CONTROL_SUBDIR = "positions_gate" + VARIANT
#: Frozen temporal fill gap (``results/gsr_calibgate_frozen.json``).
FILL_GAP = 10
_MODES = ("full", "ground_plane", "main")


# === the GPU pass ================================================================================
def cache_sequence(seq_dir: Path, dest_dir: Path, calibrator,
                   arms: dict[str, tuple[bool, int, bool]], *, stride: int = 1) -> dict:
    """Cache every PnLCalib hypothesis of every frame, once per decode arm, from one forward pass.

    Args:
        seq_dir: GSR sequence folder (``img1/%06d.jpg``).
        dest_dir: Output root; writes ``<arm>/<SEQ>.npz``.
        calibrator: A :class:`~generator.calibrate.PnLCalibCalibrator` (flags are mutated per arm).
        arms: ``{arm: (subpix_decode, derived_kp_scale, half_cell_offset)}``.
        stride: Take every Nth frame (1 = all).

    Returns:
        Counts + wall time for the run log.
    """
    import cv2  # noqa: PLC0415

    frames = sorted(int(p.stem) - 1 for p in (seq_dir / "img1").glob("*.jpg"))[::stride]
    cols: dict[str, dict[str, list]] = {
        a: {k: [] for k in ("frame", "rank", "err_m", "n_pts", "rep_px", "mode", "ransac")}
        for a in arms}
    hs: dict[str, list[np.ndarray]] = {a: [] for a in arms}
    solver_errors: dict[str, int] = {a: 0 for a in arms}
    t0 = time.time()
    for fr in frames:
        img = cv2.imread(str(seq_dir / "img1" / f"{fr + 1:06d}.jpg"))
        if img is None:
            logger.warning("%s: frame %d image missing", seq_dir.name, fr)
            continue
        for arm, (subpix, dks, half) in arms.items():
            calibrator.subpix_decode = subpix
            calibrator.derived_kp_scale = dks
            calibrator.half_cell_offset = half
            cv2.setRNGSeed(RNG_SEED)
            before = calibrator.n_solver_errors
            for rank, c in enumerate(calibrator.candidates(img)):
                hs[arm].append(c.homography)
                col = cols[arm]
                col["frame"].append(fr)
                col["rank"].append(rank)
                col["err_m"].append(c.error_m)
                col["n_pts"].append(c.n_points)
                col["rep_px"].append(c.rep_err_px)
                col["mode"].append(_MODES.index(c.mode))
                col["ransac"].append(c.use_ransac)
            solver_errors[arm] += calibrator.n_solver_errors - before
    out = {"seq": seq_dir.name, "n_frames": len(frames), "seconds": round(time.time() - t0, 1),
           "solver_errors": dict(solver_errors)}
    for arm in arms:
        d = dest_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{seq_dir.name}.npz",
                            homography=np.asarray(hs[arm], dtype=float).reshape(-1, 3, 3),
                            **{k: np.asarray(v) for k, v in cols[arm].items()})
        out[f"n_cand_{arm}"] = len(hs[arm])
    logger.info("%s: %d frames, %.1f s (%.2f s/frame), candidates %s, solver errors %s",
                seq_dir.name, len(frames), out["seconds"], out["seconds"] / max(1, len(frames)),
                {a: out[f"n_cand_{a}"] for a in arms}, out["solver_errors"])
    return out


def load_cache(path: Path) -> dict[int, list]:
    """Read one arm's candidate ``.npz`` back into ``{frame: [CalibCandidate, ...]}`` in rank order."""
    from generator.calibrate import CalibCandidate  # noqa: PLC0415

    with np.load(path) as z:  # NpzFile re-decompresses on EVERY key access -- read each once
        hom, fr, rk = z["homography"], z["frame"], z["rank"]
        err, npts, rep = z["err_m"], z["n_pts"], z["rep_px"]
        mode, ransac = z["mode"], z["ransac"]
    by_frame: dict[int, list] = {}
    for i in np.argsort(fr.astype(np.int64) * 100 + rk, kind="stable"):
        by_frame.setdefault(int(fr[i]), []).append(CalibCandidate(
            homography=hom[i], error_m=float(err[i]), n_points=int(npts[i]),
            mode=_MODES[int(mode[i])], use_ransac=float(ransac[i]), rep_err_px=float(rep[i])))
    return by_frame


# === the shipped gate, replayed on a cache =======================================================
def gate_homographies(by_frame: dict[int, list], df: pd.DataFrame) -> tuple[dict, dict]:
    """Apply the shipped gate (2 m keypoint error + ``onpitch_plausible``) to a cached arm.

    Args:
        by_frame: :func:`load_cache` output.
        df: The sequence's positions table (supplies each frame's own detected foot points).

    Returns:
        ``({frame: H}, {frame: rep_err_px of the accepted hypothesis})`` for accepted frames only.
    """
    from generator.calibrate import select_calibration  # noqa: PLC0415
    from generator.postprocess import PLAYER_ROLES  # noqa: PLC0415

    people = df[df["role"].isin(PLAYER_ROLES)]
    feet = {int(f): g[["image_x", "image_y"]].to_numpy(dtype=float)
            for f, g in people.groupby("frame")}
    homs: dict[int, np.ndarray] = {}
    rep: dict[int, float] = {}
    for fr, cands in by_frame.items():
        foot = feet.get(fr)
        foot = foot[np.isfinite(foot).all(axis=1)] if foot is not None else None
        res = select_calibration(cands, foot_points=foot)
        if not res.ok or res.homography is None:
            continue
        homs[fr] = res.homography
        rep[fr] = next((c.rep_err_px for c in cands if c.homography is res.homography
                        and c.error_m == res.error_m), float("nan"))
    return homs, rep


# === the accuracy instrument (homography only, GT-anchored) ======================================
def homography_error(homs: dict[int, np.ndarray], gt: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-frame errors (m) of GT foot points pushed through ``homs`` against their own GT pitch row.

    Args:
        homs: ``{frame: (3, 3) image -> uncentred pitch metres}``.
        gt: ``{frame: (N, 4)}`` from :func:`tools.gsr_w2_calibswap.gt_people`.

    Returns:
        ``{frame: (N,) errors}``. Uses the annotation's own image point, so no detection, no
        matcher and no box convention enters: this is calibration error and nothing else.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    out: dict[int, np.ndarray] = {}
    for fr, h in homs.items():
        g = gt.get(fr)
        if g is None or not len(g):
            continue
        proj = apply_homography(h, g[:, :2])
        out[fr] = np.linalg.norm(proj - g[:, 2:4], axis=1)
    return out


def _stats(err: np.ndarray) -> dict:
    """Median / p90 / p99 / mean / share within the evaluator's 5 m gate."""
    if not len(err):
        return {"n": 0}
    return {"n": int(len(err)), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)), "p99": float(np.percentile(err, 99)),
            "mean": float(err.mean()), "within_5m": float((err <= 5.0).mean())}


def accuracy_report(data_dir: Path, cache_dir: Path, out_dir: Path, seqs: list[str],
                    arms: list[str]) -> dict:
    """GT-anchored homography accuracy per arm, paired on the frames every arm calibrates."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    per_seq: dict[str, dict] = {}
    pooled: dict[str, list[np.ndarray]] = {a: [] for a in arms}
    pooled_rep: dict[str, list[float]] = {a: [] for a in arms}
    for name in seqs:
        gt = gt_people(data_dir / name)
        df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        errs, reps, cover = {}, {}, {}
        for arm in arms:
            homs, rep = gate_homographies(load_cache(cache_dir / arm / f"{name}.npz"), df)
            errs[arm] = homography_error(homs, gt)
            reps[arm] = rep
            cover[arm] = len(homs)
        shared = sorted(set.intersection(*[set(errs[a]) for a in arms]))
        rec = {"frames_calibrated": cover, "frames_paired": len(shared)}
        for arm in arms:
            e = np.concatenate([errs[arm][f] for f in shared]) if shared else np.zeros(0)
            pooled[arm].append(e)
            pooled_rep[arm].extend(reps[arm][f] for f in shared)
            rec[arm] = _stats(e)
        per_seq[name] = rec
        logger.info("%s: %s", name, " | ".join(
            f"{a} {rec[a].get('median', float('nan')):.3f}m p90 {rec[a].get('p90', 0):.3f}"
            for a in arms))
    rep_out: dict = {"seqs": seqs, "arms": arms, "per_seq": per_seq, "pooled": {}}
    base = np.concatenate(pooled[arms[0]])
    for arm in arms:
        e = np.concatenate(pooled[arm])
        st = _stats(e)
        st["rep_err_px_median"] = float(np.nanmedian(pooled_rep[arm])) if pooled_rep[arm] else None
        if arm != arms[0] and len(e) == len(base) and len(e):
            d = e - base
            st["vs_ctrl_mean_delta_m"] = float(d.mean())
            st["vs_ctrl_rows_better"] = int((d < 0).sum())
            st["vs_ctrl_rows_worse"] = int((d > 0).sum())
            st["vs_ctrl_wilcoxon_p"] = (float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        rep_out["pooled"][arm] = st
    return rep_out


# === the image-space residual field (defect 3, and the mechanism cross-check) ====================
def image_residuals(homs: dict[int, np.ndarray], gt: dict[int, np.ndarray],
                    size: tuple[float, float] = (1920.0, 1080.0)) -> np.ndarray:
    """GT pitch points pushed back into the image: ``(N, 4)`` = residual dx, dy, radius, angle-free.

    Args:
        homs: ``{frame: (3, 3) image -> pitch}``.
        gt: ``{frame: (N, 4)}`` from :func:`tools.gsr_w2_calibswap.gt_people`.
        size: Image size; the radius is measured from its centre.

    Returns:
        ``(N, 4)`` of ``[dx, dy, r, radial component of (dx, dy)]`` in pixels. A *constant* mean
        ``(dx, dy)`` is a decode offset; a mean radial component that grows with ``r`` is lens
        distortion the pinhole model is not carrying.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    cx, cy = size[0] / 2.0, size[1] / 2.0
    rows = []
    for fr, h in homs.items():
        g = gt.get(fr)
        if g is None or not len(g):
            continue
        try:
            inv = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            continue
        pred = apply_homography(inv, g[:, 2:4])
        d = g[:, :2] - pred
        rad = np.stack([g[:, 0] - cx, g[:, 1] - cy], axis=1)
        r = np.linalg.norm(rad, axis=1)
        unit = rad / np.where(r[:, None] < 1e-9, 1.0, r[:, None])
        rows.append(np.column_stack([d, r, (d * unit).sum(axis=1)]))
    return np.concatenate(rows) if rows else np.zeros((0, 4))


def radial_report(data_dir: Path, cache_dir: Path, out_dir: Path, seqs: list[str],
                  arms: list[str], n_bins: int = 6) -> dict:
    """Per-arm mean image residual overall and binned by image radius (robust to outliers)."""
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    rep: dict[str, dict] = {}
    for arm in arms:
        chunks = []
        for name in seqs:
            df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
            homs, _ = gate_homographies(load_cache(cache_dir / arm / f"{name}.npz"), df)
            chunks.append(image_residuals(homs, gt_people(data_dir / name)))
        a = np.concatenate([c for c in chunks if len(c)])
        keep = np.linalg.norm(a[:, :2], axis=1) < 200.0  # drop hopeless frames, same rule per arm
        a = a[keep]
        edges = np.quantile(a[:, 2], np.linspace(0, 1, n_bins + 1))
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (a[:, 2] >= lo) & (a[:, 2] <= hi)
            bins.append({"r_lo": float(lo), "r_hi": float(hi), "n": int(m.sum()),
                         "median_radial_px": float(np.median(a[m, 3])),
                         "median_dx_px": float(np.median(a[m, 0])),
                         "median_dy_px": float(np.median(a[m, 1]))})
        rep[arm] = {"n": int(len(a)), "median_dx_px": float(np.median(a[:, 0])),
                    "median_dy_px": float(np.median(a[:, 1])),
                    "median_radial_px": float(np.median(a[:, 3])), "bins": bins}
        logger.info("%s: median residual dx %+.3f dy %+.3f px, radial %+.3f, bins %s", arm,
                    rep[arm]["median_dx_px"], rep[arm]["median_dy_px"], rep[arm]["median_radial_px"],
                    [round(b["median_radial_px"], 3) for b in bins])
    return rep


# === positions ===================================================================================
def build_positions(cache_dir: Path, out_dir: Path, seqs: list[str], arms: list[str],
                    suffix: str) -> dict:
    """Write one positions parquet set per arm: gate -> reproject -> ``fill_calibration_gaps``."""
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415

    from tools.gsr_w2_calibswap import swap_positions  # noqa: PLC0415

    stats: dict[str, dict] = {a: {} for a in arms}
    for name in seqs:
        df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        for arm in arms:
            homs, _ = gate_homographies(load_cache(cache_dir / arm / f"{name}.npz"), df)
            tab, st = swap_positions(df, homs, origin=(0.0, 0.0), mode="t1")
            tab = fill_calibration_gaps(tab, max_gap=FILL_GAP)
            st["pitch_rows_after_fill"] = int(np.isfinite(tab["pitch_x"]).sum())
            sub = out_dir / (f"positions_{suffix}{arm}" + VARIANT)
            sub.mkdir(parents=True, exist_ok=True)
            tab.to_parquet(sub / f"{name}.parquet", index=False)
            stats[arm][name] = st
        logger.info("%s: %s", name, " | ".join(
            f"{a} {stats[a][name]['pitch_rows_after_fill']}" for a in arms))
    return stats


# === scoring =====================================================================================
def score_arms(data_dir: Path, out_dir: Path, seqs: list[str], arms: list[str], suffix: str,
               work: Path) -> dict:
    """Run the frozen v6 chain per arm, then the v9-W7 paired scorer (flags ON and OFF)."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415
    from tools.gsr_v6det import TAU  # noqa: PLC0415
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache" + VARIANT
    raw: dict[str, dict] = {}
    for arm in arms:
        res = eiou.run_point(data_dir, out_dir, seqs,
                             eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                             embedder="clip" + VARIANT, tau=TAU, tag=f"v10w2_{arm}",
                             percrop_variant="_v6" + VARIANT,
                             positions_subdir=f"positions_{suffix}{arm}" + VARIANT)
        h = res["gs_hota"]
        raw[arm] = {"gs_hota": h, "gs_hota_per_seq": res["gs_hota_per_seq"],
                    "eiou_tracks": (res["eiou"]["n_tracks_before"], res["eiou"]["n_tracks_after"])}
        print(f"{arm:<6} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
              f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f} IDF1 {h['IDF1']:7.4f}", flush=True)
    ctrl = out_dir / f"deleak_v10w2_{arms[0]}" / "predictions" / "data"
    paired = {}
    for arm in arms[1:]:
        paired[arm] = score_pair(ctrl, out_dir / f"deleak_v10w2_{arm}" / "predictions" / "data",
                                 data_dir, work / arm, seqs)
    return {"seqs": seqs, "arms": raw, "paired_vs_ctrl": paired}


# === self-check ==================================================================================
def _demo() -> None:
    """Assert the pure seams: the sub-cell decode, the gate replay and the accuracy instrument."""
    import torch  # noqa: PLC0415

    from generator.calibrate import CalibCandidate, refine_peaks  # noqa: PLC0415

    # 1. sub-cell decode recovers a known off-cell Gaussian to better than a hundredth of a cell.
    yy, xx = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    hm = torch.tensor(np.exp(-((yy - 12.4) ** 2 + (xx - 20.8) ** 2) / 8.0),
                      dtype=torch.float32).view(1, 1, 30, 30)
    stock = torch.tensor([[[[42.0, 24.0, 1.0]]]])  # arg-max cell (21, 12) times scale 2
    got = refine_peaks(stock, hm).numpy()[0, 0, 0]
    assert abs(got[0] / 2 - 20.8) < 0.01 and abs(got[1] / 2 - 12.4) < 0.01, got

    # 2. the gate replay: a hypothesis that fails the metre gate is skipped for the next one.
    good = np.array([[0.06, 0.004, -8.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])
    cands = [CalibCandidate(good, 9.0, 8, "full", 0.0, 1.0),
             CalibCandidate(good, 0.4, 8, "full", 5.0, 2.0)]
    df = pd.DataFrame({"frame": [0] * 5, "role": ["player"] * 5,
                       "image_x": np.linspace(300, 1500, 5), "image_y": [500.0] * 5})
    homs, rep = gate_homographies({0: cands}, df)
    assert set(homs) == {0} and rep[0] == 2.0, (homs, rep)
    assert not gate_homographies({0: cands[:1]}, df)[0], "both hypotheses over the gate -> dead"

    # 3. the accuracy instrument is a pure reprojection of the annotation's own point.
    h = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 1.0]])  # 20 px = 1 m
    gt = {0: np.array([[200.0, 400.0, 10.0, 21.0]])}  # projects to (10, 20) -> 1 m off
    assert np.allclose(homography_error({0: h}, gt)[0], [1.0])
    assert _stats(np.array([1.0, 2.0, 9.0]))["within_5m"] == 2 / 3

    # 4. the residual field: a pure image translation must read as a constant dx/dy, not radial.
    gt2 = {0: np.array([[200.0, 400.0, 10.0, 20.0], [600.0, 800.0, 30.0, 40.0]])}
    res = image_residuals({0: h}, gt2, size=(1920.0, 1080.0))
    assert np.allclose(res[:, :2], 0.0), res
    # a homography that reads 0.25 m short on both axes = a uniform 5 px image-space residual
    off = np.array([[0.05, 0.0, -0.25], [0.0, 0.05, -0.25], [0.0, 0.0, 1.0]])
    res = image_residuals({0: off}, gt2, size=(1920.0, 1080.0))
    assert np.allclose(res[:, 0], -5.0) and np.allclose(res[:, 1], -5.0), res
    print("gsr_v10_w2_calib demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/gsr/v10_w2_cand"))
    ap.add_argument("--results", type=Path,
                    default=Path("results/gsr_benchmark/gsr_v10_w2.json"))
    ap.add_argument("--seqs", default=None, help="comma list (default: the declared DEV-20)")
    ap.add_argument("--arms", default=",".join(ARMS), help="subset of " + ",".join(ARMS))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--suffix", default="v10w2", help="positions subdir prefix")
    ap.add_argument("--shard", default=None, help="i/n -- cache only this shard of the sequences")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--radial", action="store_true")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return

    if args.seqs:
        seqs = args.seqs.split(",")
    else:
        from eval.gsr_identity import split_sequences  # noqa: PLC0415
        seqs = split_sequences(args.data_dir, args.out_dir)[0]
    arms = [a for a in args.arms.split(",") if a in ARMS]
    payload: dict = {}

    if args.cache:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415
        todo = seqs
        if args.shard:
            i, n = (int(v) for v in args.shard.split("/"))
            todo = seqs[i::n]
        calib = PnLCalibCalibrator()
        runs = [cache_sequence(args.data_dir / s, args.cache_dir,
                               calib, {a: ARMS[a] for a in arms}, stride=args.stride)
                for s in todo]
        payload["cache"] = runs
        print(f"cached {len(runs)} sequences, {sum(r['seconds'] for r in runs) / 60:.1f} min")
    if args.accuracy:
        payload["accuracy"] = accuracy_report(args.data_dir, args.cache_dir, args.out_dir,
                                              seqs, arms)
        for a in arms:
            p = payload["accuracy"]["pooled"][a]
            print(f"{a:<6} n={p['n']:>7} median {p['median']:.4f} m  p90 {p['p90']:.4f}  "
                  f"mean {p['mean']:.4f}  <=5m {p['within_5m']:.5f}  "
                  f"rep_px {p['rep_err_px_median']}")
    if args.radial:
        payload["radial"] = radial_report(args.data_dir, args.cache_dir, args.out_dir, seqs, arms)
    if args.positions:
        payload["positions"] = build_positions(args.cache_dir, args.out_dir, seqs, arms,
                                               args.suffix)
    if args.score:
        payload["score"] = score_arms(args.data_dir, args.out_dir, seqs, arms, args.suffix,
                                      args.out_dir / "v10_w2_pair")
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
