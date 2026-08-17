"""v10-W11: mint NEW camera hypotheses from conic-derived correspondences.

v10-W8 closed the selection story: the cached hypothesis pool is 82% exhausted by arm A and the
residue is defended by an information ceiling (kb v10-w8-006). The remaining ~4.36 predicted GS-HOTA
therefore has to come from hypotheses PnLCalib never generated -- which means new correspondences.

The registered premise correction (``results/gsr_v10_w11_registered.json``) is that Falaleev &
Chen's 57-keypoint construction (arXiv:2410.07401) IS PnLCalib's keypoint vocabulary, so "add their
points" is not available. What is available, and is what this tool builds, is the conic layer
PnLCalib drops at inference: it predicts 21 keypoints that lie on three circles of known radius and
three circle centres, and then treats all 24 as independent points.
:mod:`generator.conic_calib` fits the conics, denoises the detections onto them, and derives the
missing on-circle keypoints analytically (Falaleev sec. 3.1 + Magera et al. arXiv:2504.20052 sec. 3).

``--scope``    Gate 0, CPU, cached artifacts only. Does per-frame accuracy respond to correspondence
               COUNT at all? Sizes the GPU spend before it is booked.
``--harvest``  GPU. One shipped-decode forward pass per DEV-20 frame, caching the raw keypoint and
               line dicts. Nothing in this project has ever persisted a correspondence.
``--enrich``   CPU. Conic constructions -> enlarged keypoint set -> PnLCalib's own solver -> a pool
               cache that is a strict SUPERSET of the on-record one, drop-in for
               ``tools.gsr_v10_w5 --oracle`` and ``tools.gsr_v10_w8 --accuracy`` via ``--cache-dir``.

Ground truth is read only by ``--scope`` (to measure) and never by the harvest or the enrichment.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_v10_w5 import CACHE_DIR, DEV20, POSITIONS_SUBDIR

logger = logging.getLogger("gsr_v10_w11")

#: Where this session's artifacts live. Nothing outside it is written.
WORK = Path("outputs/gsr/v10_w11")
#: The network-input frame the raw decoder coordinates live in (PnLCalib resizes 1920x1080 -> this).
NET_SIZE = (960.0, 540.0)
#: Fixed RANSAC seed, as in the v10-W2 decode, so a re-mint is reproducible.
RNG_SEED = 20260815
#: PnLCalib's own preference order, for the npz schema shared with :mod:`tools.gsr_v10_w2_calib`.
_MODES = ("full", "ground_plane", "main")


def pnlcalib_on_path() -> None:
    """Put the cloned PnLCalib repo on ``sys.path`` (its ``utils`` package is not installable)."""
    import sys  # noqa: PLC0415

    from generator.calibrate import pnlcalib_root  # noqa: PLC0415

    root = str(pnlcalib_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def world_uncentred() -> dict[int, np.ndarray]:
    """``{keypoint id: (x, y)}`` on the uncentred 105x68 pitch, from PnLCalib's own table."""
    from generator.calibrate import PITCH_LENGTH_M, PITCH_WIDTH_M  # noqa: PLC0415

    pnlcalib_on_path()
    from utils.utils_calib import keypoint_world_coords_2D  # noqa: PLC0415

    shift = np.array([PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2])
    return {i + 1: np.asarray(p, dtype=float) + shift for i, p in
            enumerate(keypoint_world_coords_2D)}


# === gate 0: does accuracy respond to correspondence count at all? ================================
def run_scope(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str]) -> dict:
    """Per-frame correspondence count against per-frame GT-anchored displacement, on DEV-20.

    ``n_points`` is PnLCalib's own ground-plane correspondence count for the frame (identical across
    that frame's hypotheses, since the metre error is scored on one fixed set). If accuracy is flat
    in it, denser correspondences cannot be the lever and the GPU harvest should be a pilot.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from tools.gsr_v10_w5 import load_seq  # noqa: PLC0415
    from tools.gsr_v10_w8 import arm_homographies  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    rows = []
    for seq in seqs:
        _df, by_frame, homs, _err, _n = load_seq(seq, out_dir, cache_dir)
        homs_a, _st = arm_homographies("A", seq, out_dir, cache_dir)
        gt = gt_people(data_dir / seq)
        for fr, h in sorted(homs.items()):
            g = gt.get(fr)
            cands = by_frame.get(fr) or []
            if g is None or len(g) < 6 or not cands:
                continue
            h_gt = estimate_homography(g[:, :2], g[:, 2:4])
            if h_gt is None or not np.isfinite(h_gt).all():
                continue
            ref = apply_homography(h_gt, g[:, :2])
            d_ctrl = float(np.median(np.linalg.norm(apply_homography(h, g[:, :2]) - ref, axis=1)))
            ha = homs_a.get(fr)
            d_a = (float(np.median(np.linalg.norm(apply_homography(ha, g[:, :2]) - ref, axis=1)))
                   if ha is not None else np.nan)
            rows.append((seq, fr, int(cands[0].n_points), d_ctrl, d_a))
    tab = pd.DataFrame(rows, columns=["seq", "frame", "n_points", "disp_ctrl_m", "disp_A_m"])
    out: dict = {"frames": int(len(tab)),
                 "n_points": {"median": float(tab["n_points"].median()),
                              "p10": float(tab["n_points"].quantile(0.10)),
                              "p90": float(tab["n_points"].quantile(0.90)),
                              "max": int(tab["n_points"].max())}}
    for col in ("disp_ctrl_m", "disp_A_m"):
        ok = tab[np.isfinite(tab[col])]
        r = spearmanr(ok["n_points"], ok[col])
        out[col] = {"spearman_rho": float(r.statistic), "p": float(r.pvalue)}
    tab["bin"] = pd.qcut(tab["n_points"], 5, duplicates="drop")
    out["quintiles"] = [
        {"n_points_lo": float(b.left), "n_points_hi": float(b.right), "frames": int(len(g)),
         "median_disp_ctrl_m": float(g["disp_ctrl_m"].median()),
         "median_disp_A_m": float(g["disp_A_m"].median())}
        for b, g in tab.groupby("bin", observed=True)]
    WORK.mkdir(parents=True, exist_ok=True)
    tab.drop(columns=["bin"]).to_parquet(WORK / "scope.parquet", index=False)
    return out


# === section 1: the GPU harvest ==================================================================
def harvest(seq_dir: Path, dest: Path, calibrator) -> dict:
    """Cache the raw keypoint and line dicts of every frame, from the SHIPPED decode.

    ``subpix_decode`` / ``derived_kp_scale`` / ``half_cell_offset`` are the ctrl arm's, byte for
    byte: kb v10-w2-002 refuted sub-cell decoding (PnLCalib trains on integer cells) and the
    half-cell rider failed its ship bar, so neither rides here.

    Args:
        seq_dir: GSR sequence folder (``img1/%06d.jpg``).
        dest: Output directory; writes ``<SEQ>.npz``.
        calibrator: A loaded :class:`~generator.calibrate.PnLCalibCalibrator`.

    Returns:
        Counts + wall time for the run log.
    """
    import cv2  # noqa: PLC0415

    calibrator._load()  # noqa: SLF001 - builds both HRNets AND puts PnLCalib on sys.path
    calibrator.subpix_decode = False
    calibrator.derived_kp_scale = 1
    calibrator.half_cell_offset = False
    kf, kid, kx, ky, kp = [], [], [], [], []
    lf, lid, lxy = [], [], []
    frames = sorted(int(p.stem) - 1 for p in (seq_dir / "img1").glob("*.jpg"))
    t0 = time.time()
    for fr in frames:
        img = cv2.imread(str(seq_dir / "img1" / f"{fr + 1:06d}.jpg"))
        if img is None:
            logger.warning("%s: frame %d image missing", seq_dir.name, fr)
            continue
        calibrator._detect(img)  # noqa: SLF001 - the seam this session exists to read
        kp_d, ln_d, _cw, _ch = calibrator.last_raw
        for k, v in kp_d.items():
            kf.append(fr), kid.append(int(k)), kx.append(float(v["x"]))
            ky.append(float(v["y"])), kp.append(float(v.get("p", 1.0)))
        for k, v in ln_d.items():
            lf.append(fr), lid.append(int(k))
            lxy.append([float(v["x_1"]), float(v["y_1"]), float(v["x_2"]), float(v["y_2"])])
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / f"{seq_dir.name}.npz",
        kp_frame=np.asarray(kf, dtype=np.int32), kp_id=np.asarray(kid, dtype=np.int16),
        kp_x=np.asarray(kx), kp_y=np.asarray(ky), kp_p=np.asarray(kp),
        ln_frame=np.asarray(lf, dtype=np.int32), ln_id=np.asarray(lid, dtype=np.int16),
        ln_xy=np.asarray(lxy, dtype=float).reshape(-1, 4))
    st = {"seq": seq_dir.name, "frames": len(frames), "keypoints": len(kf), "lines": len(lf),
          "seconds": round(time.time() - t0, 1)}
    logger.info("%s: %d frames, %d kp (%.1f/frame), %d lines (%.1f/frame), %.0f s",
                seq_dir.name, len(frames), len(kf), len(kf) / max(1, len(frames)),
                len(lf), len(lf) / max(1, len(frames)), st["seconds"])
    return st


def load_harvest(path: Path) -> dict[int, tuple[dict, dict]]:
    """Read one sequence's harvest back into ``{frame: (kp_dict, lines_dict)}`` (raw decoder units)."""
    with np.load(path) as z:
        kf, kid, kx, ky, kp = z["kp_frame"], z["kp_id"], z["kp_x"], z["kp_y"], z["kp_p"]
        lf, lid, lxy = z["ln_frame"], z["ln_id"], z["ln_xy"]
    out: dict[int, tuple[dict, dict]] = {}
    for i in range(len(kf)):
        out.setdefault(int(kf[i]), ({}, {}))[0][int(kid[i])] = {
            "x": float(kx[i]), "y": float(ky[i]), "p": float(kp[i])}
    for i in range(len(lf)):
        out.setdefault(int(lf[i]), ({}, {}))[1][int(lid[i])] = {
            "x_1": float(lxy[i, 0]), "y_1": float(lxy[i, 1]),
            "x_2": float(lxy[i, 2]), "y_2": float(lxy[i, 3])}
    return out


# === section 2/3: the conic constructions and the new hypotheses ==================================
def _line_of(v: dict) -> np.ndarray:
    """Homogeneous image line through a detected line's two endpoints (pure)."""
    return np.cross(np.array([v["x_1"], v["y_1"], 1.0]), np.array([v["x_2"], v["y_2"], 1.0]))


def enrich_frame(kp_d: dict, ln_d: dict, h_gate: np.ndarray, world: dict[int, np.ndarray],
                 scale: float) -> tuple[dict, dict]:
    """Apply the conic layer to one frame's raw decode: ``(enriched kp_dict, stats)``.

    Args:
        kp_d: Raw keypoint dict in network-input pixels.
        ln_d: Raw line dict in the same units.
        h_gate: The frame's on-record image->pitch homography (NATIVE 1920x1080 pixels), used only
            to place each model keypoint for the discrete correspondence assignment.
        world: ``{keypoint id: (x, y)}`` uncentred pitch coordinates.
        scale: Native pixels per network-input pixel (2.0 for 1920x1080 -> 960x540).

    Returns:
        The enriched keypoint dict (same schema as ``kp_d``) and the construction counts.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415
    from generator.conic_calib import derive_points  # noqa: PLC0415

    kp_xy = {k: np.array([v["x"], v["y"]]) for k, v in kp_d.items()}
    lines = {k: _line_of(v) for k, v in ln_d.items()}
    try:
        h_inv = np.linalg.inv(h_gate)
    except np.linalg.LinAlgError:
        return dict(kp_d), {"skipped_singular": 1}
    ids = sorted(world)
    proj = apply_homography(h_inv, np.array([world[k] for k in ids])) / scale
    predicted = {k: p for k, p in zip(ids, proj, strict=True) if np.isfinite(p).all()}
    den, der, st = derive_points(kp_xy, lines, predicted, NET_SIZE)
    out = {k: dict(v) for k, v in kp_d.items()}
    for k, p in den.items():
        out[k] = {**out[k], "x": float(p[0]), "y": float(p[1])}
    for k, p in der.items():
        out[k] = {"x": float(p[0]), "y": float(p[1]), "p": 1.0}
    return out, st


def enrich_sequence(seq: str, data_dir: Path, out_dir: Path, cache_dir: Path,
                    harvest_dir: Path, dest: Path) -> dict:
    """Mint the enlarged hypothesis pool for one sequence and write it in the v10-W2 npz schema.

    The on-record hypotheses come first, in their cached order, so ``select_calibration``'s pick is
    byte-identical and the ctrl arm is an exact control; the conic-derived hypotheses are appended.
    """
    import cv2  # noqa: PLC0415

    from generator.calibrate import candidates_from_dicts  # noqa: PLC0415
    from tools.gsr_v10_w2_calib import gate_homographies, load_cache  # noqa: PLC0415

    world = world_uncentred()  # also puts PnLCalib on sys.path for the import below
    from utils.utils_heatmap import complete_keypoints  # noqa: PLC0415

    by_frame = load_cache(cache_dir / f"{seq}.npz")
    df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
    homs, _rep = gate_homographies(by_frame, df)
    raw = load_harvest(harvest_dir / f"{seq}.npz")
    wid, hgt = NET_SIZE
    cols: dict[str, list] = {k: [] for k in ("frame", "rank", "err_m", "n_pts", "rep_px", "mode",
                                             "ransac")}
    hs: list[np.ndarray] = []
    tot: dict[str, int] = {}
    n_new, n_frames_new, n_frames = 0, 0, 0
    t0 = time.time()
    for fr, cands in sorted(by_frame.items()):
        extra: list = []
        if fr in homs and fr in raw:
            n_frames += 1
            kp_d, ln_d = raw[fr]
            kp_new, st = enrich_frame(kp_d, ln_d, homs[fr], world, scale=1920.0 / wid)
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + int(v)
            if len(kp_new) > len(kp_d) or st.get("denoised"):
                kpn, lnn = complete_keypoints({k: dict(v) for k, v in kp_new.items()},
                                              {k: dict(v) for k, v in ln_d.items()},
                                              w=wid, h=hgt, normalize=True)
                cv2.setRNGSeed(RNG_SEED)
                extra = candidates_from_dicts(kpn, lnn, 1920, 1080)
                if extra:
                    n_new += len(extra)
                    n_frames_new += 1
        for rank, c in enumerate([*cands, *extra]):
            hs.append(c.homography)
            cols["frame"].append(fr), cols["rank"].append(rank)
            cols["err_m"].append(c.error_m), cols["n_pts"].append(c.n_points)
            cols["rep_px"].append(c.rep_err_px), cols["mode"].append(_MODES.index(c.mode))
            cols["ransac"].append(c.use_ransac)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"{seq}.npz",
                        homography=np.asarray(hs, dtype=float).reshape(-1, 3, 3),
                        **{k: np.asarray(v) for k, v in cols.items()})
    st = {"seq": seq, "frames": n_frames, "frames_with_new": n_frames_new,
          "new_hypotheses": n_new, "constructions": tot,
          "new_per_frame": round(n_new / max(1, n_frames), 3), "seconds": round(time.time() - t0, 1)}
    logger.info("%s: %d frames, conics %d, denoised %d, derived %d "
                "(line-conic %d / tangent %d / magera %d) -> %d new hypotheses on %d frames, %.0f s",
                seq, n_frames, tot.get("conics", 0), tot.get("denoised", 0),
                tot.get("line_conic", 0) + tot.get("tangent", 0) + tot.get("magera", 0),
                tot.get("line_conic", 0), tot.get("tangent", 0), tot.get("magera", 0),
                n_new, n_frames_new, st["seconds"])
    return st


# === self-check ==================================================================================
def _demo() -> None:
    """Assert the seams that do not need a GPU: the world table and the frame enrichment."""
    world = world_uncentred()
    assert len(world) == 57
    assert np.allclose(world[51], [52.5, 34.0]) and np.allclose(world[1], [0.0, 0.0]), world[51]

    # A synthetic camera: build a raw decode from the model points, drop the derivable ones, and
    # check the enrichment puts them back within a pixel.
    h_world_to_img = np.array([[14.0, 1.1, 180.0], [1.6, 6.4, 60.0], [0.0009, 0.0075, 1.0]])
    h_gate = np.linalg.inv(h_world_to_img)  # image -> pitch, in the SAME (net) pixel units

    def to_img(ids):
        q = np.column_stack([np.array([world[i] for i in ids]), np.ones(len(ids))])
        r = q @ h_world_to_img.T
        return r[:, :2] / r[:, 2:3]

    seed = [38, 39, 42, 43, 48, 53, 2, 29, 51]
    kp_d = {i: {"x": float(p[0]), "y": float(p[1]), "p": 0.9}
            for i, p in zip(seed, to_img(seed), strict=True)}
    ends = to_img([2, 29])
    ln_d = {13: {"x_1": ends[0, 0], "y_1": ends[0, 1], "x_2": ends[1, 0], "y_2": ends[1, 1]}}
    out, st = enrich_frame(kp_d, ln_d, h_gate, world, scale=1.0)
    assert st["conics"] == 1 and st["line_conic"] == 2, st
    for kid in (32, 35, 50):
        assert kid in out, (kid, sorted(out), st)
        truth = to_img([kid])[0]
        got = np.array([out[kid]["x"], out[kid]["y"]])
        assert np.linalg.norm(got - truth) < 1.0, (kid, got, truth)
    assert set(kp_d) <= set(out), "enrichment must never drop a detected keypoint"

    # a singular gate homography must be refused, not crash
    _o2, st2 = enrich_frame(kp_d, ln_d, np.zeros((3, 3)), world, scale=1.0)
    assert st2 == {"skipped_singular": 1}, st2
    print(f"gsr_v10_w11 demo OK (derived {sorted(set(out) - set(kp_d))})")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--harvest-dir", type=Path, default=WORK / "harvest")
    ap.add_argument("--pool-dir", type=Path, default=WORK / "pool")
    ap.add_argument("--results", type=Path, default=Path("results/gsr_benchmark/gsr_v10_w11.json"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--scope", action="store_true")
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--enrich", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else DEV20
    payload: dict = {}
    t0 = time.time()

    if args.scope:
        payload["scope"] = run_scope(args.data_dir, args.out_dir, args.cache_dir, seqs)
        print(json.dumps(payload["scope"], indent=1))
    if args.harvest:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        cal = PnLCalibCalibrator()
        payload["harvest"] = [harvest(args.data_dir / s, args.harvest_dir, cal) for s in seqs]
    if args.enrich:
        payload["enrich"] = [enrich_sequence(s, args.data_dir, args.out_dir, args.cache_dir,
                                             args.harvest_dir, args.pool_dir) for s in seqs]
        tot: dict[str, int] = {}
        for r in payload["enrich"]:
            for k, v in r["constructions"].items():
                tot[k] = tot.get(k, 0) + v
        n_fr = sum(r["frames"] for r in payload["enrich"])
        payload["enrich_pooled"] = {
            "frames": n_fr, "constructions": tot,
            "new_hypotheses": sum(r["new_hypotheses"] for r in payload["enrich"]),
            "frames_with_new": sum(r["frames_with_new"] for r in payload["enrich"]),
            "derived_per_frame": round(sum(tot.get(k, 0) for k in
                                           ("line_conic", "tangent", "magera")) / max(1, n_fr), 3),
            "denoised_per_frame": round(tot.get("denoised", 0) / max(1, n_fr), 3)}
        print(json.dumps(payload["enrich_pooled"], indent=1))
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
