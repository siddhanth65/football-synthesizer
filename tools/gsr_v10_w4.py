"""v10-W4: how much GS-HOTA does the geometry axis actually hold?

Three registered questions (`results/gsr_v10_w4_registered.json`):

``--curve``    Rung 0a. The calibration-error -> GS-HOTA **sensitivity curve** nobody has published,
               plus the oracles that bound the axis: perfect camera (``O_acc``), perfect
               completeness (``O_comp``), both (``O_all``), and the price of our linear fill alone
               (``O_fill``). Every arm rewrites only ``bbox_pitch`` of the shipped submission, so
               detections, track ids, association and attributes are byte-identical; the v9 flags
               run **after** the damage, on the damaged geometry.
``--anatomy``  Rung 0b. The dead-frame census on this lineage: how many frames the solver misses,
               how many survive the fill, how many carry a camera hypothesis at all, and -- from
               each dead frame's GT-fitted camera -- which pitch landmarks were in view. That last
               column prices the central-view (circle->line) rescue without building it.
``--smooth`` / ``--score``
               Rung 1. Decompose our per-frame homographies into ``[pan, tilt, roll, log f, C]``
               (:mod:`generator.camera_track`), RTS-smooth them, and rebuild positions two ways:
               ``A`` smoothed everywhere, ``B`` smoothed only where the solver said nothing (the
               replacement for :func:`generator.postprocess.fill_calibration_gaps`).

CPU only. Ground truth is read to score, and to fit the oracle homographies; no shippable arm
consumes it.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_v10_w4")

#: The declared DEV-20, in the v10-W2 order.
DEV20 = ["SNGS-021", "SNGS-024", "SNGS-027", "SNGS-030", "SNGS-033", "SNGS-036", "SNGS-039",
         "SNGS-042", "SNGS-045", "SNGS-048", "SNGS-051", "SNGS-054", "SNGS-057", "SNGS-078",
         "SNGS-081", "SNGS-084", "SNGS-087", "SNGS-090", "SNGS-093", "SNGS-096"]
#: Lineage: the v10-W2 ctrl arm (the shipped decode; the half-cell rider stays OFF).
CACHE_DIR = Path("outputs/gsr/v10_w2_full/ctrl")
POSITIONS_SUBDIR = "positions_v10w2ctrl_v6det"
PRED_DIR = Path("outputs/gsr/deleak_v10w2_ctrl/predictions/data")
#: Uncentred (our convention) -> centred (GSR submission convention).
CENTRE = np.array([52.5, 34.0])
SEED = 20260816
FILL_GAP = 10
#: The submission writer's fixed box; a row's foot point is ``(x + w/2, y + h)``.
BOX_W, BOX_H = 16, 40
#: Rows whose re-projection through their own frame's homography reproduces the shipped coordinate
#: to better than this are "solved rows"; the rest came from the fill.
OWN_H_TOL_M = 1e-6


# === lineage loaders =============================================================================
def frame_of(image_id: str) -> int:
    """0-based frame index of a GSR ``image_id`` (pure)."""
    return int(str(image_id)[-6:]) - 1


def load_preds(seq: str, pred_dir: Path = PRED_DIR) -> list[dict]:
    """One sequence's submission rows."""
    return json.loads((pred_dir / f"{seq}.json").read_text(encoding="utf-8"))["predictions"]


def write_preds(preds: list[dict], dest: Path, seq: str) -> None:
    """Write submission rows in the shipped format."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{seq}.json").write_text(json.dumps({"predictions": preds}), encoding="utf-8")


def foot_points(preds: list[dict]) -> np.ndarray:
    """``(N, 2)`` image-space bottom-middle points of the rows' boxes (pure)."""
    return np.array([[p["bbox_image"]["x"] + p["bbox_image"]["w"] / 2.0,
                      p["bbox_image"]["y"] + p["bbox_image"]["h"]] for p in preds], dtype=float)


def pitch_points(preds: list[dict]) -> np.ndarray:
    """``(N, 2)`` submission pitch points, converted to our uncentred convention (pure)."""
    return np.array([[p["bbox_pitch"]["x_bottom_middle"], p["bbox_pitch"]["y_bottom_middle"]]
                     for p in preds], dtype=float) + CENTRE


def set_pitch(preds: list[dict], xy_uncentred: np.ndarray, mask: np.ndarray) -> None:
    """Write uncentred pitch points back into the rows selected by ``mask``, in place."""
    for i in np.flatnonzero(mask):
        x, y = float(xy_uncentred[i, 0] - CENTRE[0]), float(xy_uncentred[i, 1] - CENTRE[1])
        preds[i]["bbox_pitch"] = {"x_bottom_left": x, "y_bottom_left": y, "x_bottom_middle": x,
                                  "y_bottom_middle": y, "x_bottom_right": x, "y_bottom_right": y}


def our_homographies(seq: str, out_dir: Path, cache_dir: Path
                     ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Replay the shipped gate on the cached PnLCalib hypotheses -> ``({frame: H}, {frame: err_m})``."""
    from tools.gsr_v10_w2_calib import gate_homographies, load_cache  # noqa: PLC0415

    df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
    by_frame = load_cache(cache_dir / f"{seq}.npz")
    homs, _rep = gate_homographies(by_frame, df)
    from generator.calibrate import select_calibration  # noqa: PLC0415
    from generator.postprocess import PLAYER_ROLES  # noqa: PLC0415

    people = df[df["role"].isin(PLAYER_ROLES)]
    feet = {int(f): g[["image_x", "image_y"]].to_numpy(dtype=float)
            for f, g in people.groupby("frame")}
    err: dict[int, float] = {}
    for fr in homs:
        foot = feet.get(fr)
        foot = foot[np.isfinite(foot).all(axis=1)] if foot is not None else None
        err[fr] = float(select_calibration(by_frame[fr], foot_points=foot).error_m)
    return homs, err


def gt_homographies(seq_dir: Path) -> tuple[dict[int, np.ndarray], dict]:
    """Fit one homography per frame from that frame's OWN ground truth (the oracle camera).

    Returns:
        ``({frame: H}, residual stats)``. The residual is the instrument's noise floor: GT
        ``bbox_pitch`` is not an exact homography of GT ``bbox_image``, and this says by how much.
    """
    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    gt = gt_people(seq_dir)
    homs: dict[int, np.ndarray] = {}
    res: list[float] = []
    for fr, g in gt.items():
        if len(g) < 6:  # 4 is the minimum; 6 keeps the RANSAC fit honest
            continue
        h = estimate_homography(g[:, :2], g[:, 2:4])
        if h is None or not np.isfinite(h).all():
            continue
        e = np.linalg.norm(apply_homography(h, g[:, :2]) - g[:, 2:4], axis=1)
        homs[int(fr)] = h
        res.append(float(np.median(e)))
    return homs, {"frames": len(homs), "median_residual_m": float(np.median(res)) if res else None,
                  "p90_residual_m": float(np.percentile(res, 90)) if res else None}


def own_h_mask(preds: list[dict], homs: dict[int, np.ndarray]) -> np.ndarray:
    """True where a row's shipped coordinate IS its own frame's solved homography (pure-ish).

    The complement is the fill population: rows whose coordinate came from a neighbouring frame's
    interpolated geometry. No bookkeeping is trusted -- the classification is a re-projection test.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    feet, have = foot_points(preds), pitch_points(preds)
    out = np.zeros(len(preds), bool)
    frames = np.array([frame_of(p["image_id"]) for p in preds])
    for fr in np.unique(frames):
        h = homs.get(int(fr))
        if h is None:
            continue
        sel = frames == fr
        d = np.linalg.norm(apply_homography(h, feet[sel]) - have[sel], axis=1)
        out[np.flatnonzero(sel)[d <= OWN_H_TOL_M]] = True
    return out


# === perturbations ================================================================================
def _reproject(preds: list[dict], homs: dict[int, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """New uncentred pitch points for the masked rows through ``homs`` (rows with no H keep theirs)."""
    from generator.calibrate import apply_homography  # noqa: PLC0415

    feet, out = foot_points(preds), pitch_points(preds)
    frames = np.array([frame_of(p["image_id"]) for p in preds])
    for fr in np.unique(frames[mask]) if mask.any() else []:
        h = homs.get(int(fr))
        if h is None:
            continue
        sel = mask & (frames == fr)
        out[sel] = apply_homography(h, feet[sel])
    return out


def jitter_cameras(homs: dict[int, np.ndarray], rng: np.random.Generator, *,
                   sd_deg: float = 0.0, sd_logf: float = 0.0) -> dict[int, np.ndarray]:
    """Independent per-frame Gaussian jitter of pan/tilt and of log focal length."""
    from generator.camera_track import Pose, compose, decompose  # noqa: PLC0415

    out: dict[int, np.ndarray] = {}
    rad = np.deg2rad(sd_deg)
    for fr, h in sorted(homs.items()):
        pose = decompose(h)
        if pose is None:
            out[fr] = h
            continue
        new = Pose(pose.pan + rng.normal(0.0, rad), pose.tilt + rng.normal(0.0, rad), pose.roll,
                   pose.log_f + rng.normal(0.0, sd_logf), pose.log_aspect,
                   pose.cx, pose.cy, pose.cz)
        h2 = compose(new)
        out[fr] = h if h2 is None else h2
    return out


def dead_runs(homs: dict[int, np.ndarray], n_frames: int) -> list[int]:
    """Lengths of the maximal runs of frames the solver did not answer (pure)."""
    live = np.zeros(n_frames, bool)
    for fr in homs:
        if 0 <= fr < n_frames:
            live[fr] = True
    runs, cur = [], 0
    for ok in live:
        if ok:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    return runs


def simulate_dead(homs: dict[int, np.ndarray], n_frames: int, frac: float,
                  run_lengths: list[int], rng: np.random.Generator
                  ) -> tuple[dict[int, np.ndarray], set[int]]:
    """Replace ``frac`` of the solved frames by the fill's linear interpolation, in realistic runs.

    Run lengths are drawn from ``run_lengths`` (our own measured dead-run distribution). The
    replacement is exactly :func:`generator.postprocess._donor_homography`'s arithmetic on the
    nearest surviving solved frames, with no gap limit -- this family prices the ACCURACY damage of
    interpolated geometry; the completeness damage is priced separately by the ``O_comp`` oracle.
    """
    from generator.postprocess import _donor_homography  # noqa: PLC0415

    solved = sorted(homs)
    target = int(round(frac * len(solved)))
    lengths = run_lengths or [10]
    killed: set[int] = set()
    guard = 0
    while len(killed) < target and guard < 10_000:
        guard += 1
        ln = int(rng.choice(lengths))
        start = int(rng.integers(0, max(1, n_frames - ln)))
        killed.update(range(start, min(start + ln, n_frames)))
    killed &= set(solved)
    donors = sorted(set(solved) - killed)
    if not donors:
        return dict(homs), set()
    out = dict(homs)
    for fr in sorted(killed):
        h = _donor_homography(fr, donors, homs, None)
        if h is not None:
            out[fr] = h
    return out, killed


# === the oracle that ADDS rows ====================================================================
def completeness_rows(seq: str, preds: list[dict], gt_h: dict[int, np.ndarray],
                      out_dir: Path, data_dir: Path) -> tuple[list[dict], dict]:
    """Submission rows for the people our chain detected but could not project (pure-ish).

    A row qualifies only when the positions table holds a finite image point and **no** pitch
    coordinate -- i.e. it was lost to calibration, not to the detector, the tracker or the solver.
    Its identity is inherited from the same ByteTrack track's surviving submission rows; a track
    that never survives anywhere cannot be named and is counted, not invented.
    """
    from eval.gsr_score import load_image_id_map  # noqa: PLC0415
    from generator.calibrate import apply_homography  # noqa: PLC0415

    df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
    df = df[df["role"].isin(["player", "goalkeeper", "referee"])]
    key = {(int(p["image_id"][-6:]) - 1, round(p["bbox_image"]["x"] + p["bbox_image"]["w"] / 2, 4),
            round(p["bbox_image"]["y"] + p["bbox_image"]["h"], 4)): p for p in preds}
    track_map: dict[int, dict[int, int]] = {}
    for r in df.itertuples():
        p = key.get((int(r.frame), round(float(r.image_x), 4), round(float(r.image_y), 4)))
        if p is not None:
            track_map.setdefault(int(r.track_id), {}).setdefault(int(p["track_id"]), 0)
            track_map[int(r.track_id)][int(p["track_id"])] += 1
    template = {int(p["track_id"]): p for p in preds}
    img_ids = load_image_id_map(data_dir / seq)
    lost = df[~np.isfinite(df["pitch_x"]) & np.isfinite(df["image_x"])]
    new: list[dict] = []
    st = {"lost_rows": int(len(lost)), "no_track_identity": 0, "no_gt_camera": 0, "added": 0}
    for r in lost.itertuples():
        votes = track_map.get(int(r.track_id))
        if not votes:
            st["no_track_identity"] += 1
            continue
        h = gt_h.get(int(r.frame))
        img_id = img_ids.get(int(r.frame))
        if h is None or img_id is None:
            st["no_gt_camera"] += 1
            continue
        tid = max(votes, key=votes.get)
        xy = apply_homography(h, np.array([[float(r.image_x), float(r.image_y)]]))[0] - CENTRE
        base = template[tid]
        new.append({
            "id": f"{img_id}9{len(new):03d}", "image_id": img_id, "track_id": tid,
            "supercategory": "object", "category_id": base["category_id"],
            "attributes": dict(base["attributes"]),
            "bbox_pitch": {"x_bottom_left": float(xy[0]), "y_bottom_left": float(xy[1]),
                           "x_bottom_middle": float(xy[0]), "y_bottom_middle": float(xy[1]),
                           "x_bottom_right": float(xy[0]), "y_bottom_right": float(xy[1])},
            "bbox_image": {"x": float(r.image_x) - BOX_W / 2, "y": float(r.image_y) - BOX_H,
                           "x_center": float(r.image_x), "y_center": float(r.image_y) - BOX_H / 2,
                           "w": BOX_W, "h": BOX_H},
            "confidence": float(r.conf)})
        st["added"] += 1
    # one prediction per (track, timestep) is a hard evaluator constraint (kb v9-w8)
    seen = {(p["image_id"], p["track_id"]) for p in preds}
    keep = []
    for p in new:
        if (p["image_id"], p["track_id"]) in seen:
            continue
        seen.add((p["image_id"], p["track_id"]))
        keep.append(p)
    st["dropped_duplicate_track_timestep"] = len(new) - len(keep)
    st["added"] = len(keep)
    return keep, st


# === scoring ======================================================================================
def score_dir(src: Path, seqs: list[str], data_dir: Path, work: Path) -> dict:
    """Apply the shipped v9 flags to a submission directory and score GS-HOTA (flags ON)."""
    import shutil  # noqa: PLC0415

    from eval.gsr_score import EVAL_CONFIGS, gs_hota  # noqa: PLC0415
    from tools.gsr_v9_w7 import FLAGS_ON, _prepare  # noqa: PLC0415

    shutil.rmtree(work, ignore_errors=True)
    _prepare(src, work, seqs, FLAGS_ON)
    res = gs_hota(work, data_dir, seq_info={s: 0 for s in seqs}, **EVAL_CONFIGS["gs_hota_full"])
    return {"combined": res["combined"], "per_seq": res["per_seq"]}


#: The registered perturbation ladder (`results/gsr_v10_w4_registered.json` rung_0a).
SPECS: list[dict] = (
    [{"name": "baseline", "family": "none"}, {"name": "N_self", "family": "self"}]
    + [{"name": f"F1_gauss_row_{s}", "family": "gauss_row", "sigma": s}
       for s in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)]
    + [{"name": f"F2_gauss_frame_{s}", "family": "gauss_frame", "sigma": s}
       for s in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)]
    + [{"name": f"F3_pantilt_{s}", "family": "cam_pantilt", "sd_deg": s}
       for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)]
    + [{"name": f"F4_focal_{s}", "family": "cam_focal", "sd_logf": s}
       for s in (0.001, 0.0025, 0.005, 0.01, 0.02)]
    + [{"name": f"F5_deadsim_{int(s * 100)}pct", "family": "deadsim", "frac": s}
       for s in (0.02, 0.05, 0.10, 0.20, 0.40)]
    + [{"name": n, "family": n} for n in ("O_acc", "O_fill", "O_comp", "O_all")]
)


def build_arm(spec: dict, seqs: list[str], data_dir: Path, out_dir: Path, cache_dir: Path,
              dest: Path, ctx: dict) -> dict:
    """Write one perturbed/oracle submission and return what it changed."""
    rng = np.random.default_rng(SEED)
    moved: list[np.ndarray] = []
    stats: dict = {"rows": 0, "rows_changed": 0, "rows_added": 0}
    for seq in seqs:
        preds = copy.deepcopy(load_preds(seq))
        homs, _err = ctx["ours"][seq]
        gt_h = ctx["gt"][seq]
        before = pitch_points(preds)
        fam = spec["family"]
        mask = np.ones(len(preds), bool)
        added: list[dict] = []
        if fam == "none":
            pass
        elif fam == "gauss_row":
            set_pitch(preds, before + rng.normal(0.0, spec["sigma"], before.shape), mask)
        elif fam == "gauss_frame":
            frames = np.array([frame_of(p["image_id"]) for p in preds])
            uniq = np.unique(frames)
            draw = {int(f): rng.normal(0.0, spec["sigma"], 2) for f in uniq}
            set_pitch(preds, before + np.array([draw[int(f)] for f in frames]), mask)
        elif fam in {"self", "cam_pantilt", "cam_focal", "deadsim"}:
            mask = ctx["own"][seq]
            use = homs
            if fam == "cam_pantilt":
                use = jitter_cameras(homs, rng, sd_deg=spec["sd_deg"])
            elif fam == "cam_focal":
                use = jitter_cameras(homs, rng, sd_logf=spec["sd_logf"])
            elif fam == "deadsim":
                use, _killed = simulate_dead(homs, ctx["n_frames"][seq], spec["frac"],
                                             ctx["runs"][seq], rng)
            set_pitch(preds, _reproject(preds, use, mask), mask)
        elif fam in {"O_acc", "O_all"}:
            set_pitch(preds, _reproject(preds, gt_h, mask), mask)
        elif fam == "O_fill":
            mask = ~ctx["own"][seq]
            set_pitch(preds, _reproject(preds, gt_h, mask), mask)
        elif fam == "O_comp":
            mask = np.zeros(len(preds), bool)
        else:
            raise ValueError(spec)
        if fam in {"O_comp", "O_all"}:
            added, st = completeness_rows(seq, preds, gt_h, out_dir, data_dir)
            stats.setdefault("completeness", {})[seq] = st
        after = pitch_points(preds)
        d = np.linalg.norm(after - before, axis=1)
        moved.append(d[d > 0])
        stats["rows"] += len(preds)
        stats["rows_changed"] += int((d > 0).sum())
        stats["rows_added"] += len(added)
        write_preds(preds + added, dest, seq)
    d = np.concatenate(moved) if moved else np.zeros(0)
    stats["displacement_m"] = {"n": int(len(d)),
                               "median": float(np.median(d)) if len(d) else 0.0,
                               "p90": float(np.percentile(d, 90)) if len(d) else 0.0,
                               "mean": float(d.mean()) if len(d) else 0.0}
    return stats


def run_curve(data_dir: Path, out_dir: Path, cache_dir: Path, work: Path, seqs: list[str],
              only: list[str] | None) -> dict:
    """Build, score and tabulate every arm of the sensitivity ladder."""
    ctx = _context(data_dir, out_dir, cache_dir, seqs)
    rows: dict[str, dict] = {}
    base = None
    for spec in SPECS:
        if only and spec["name"] not in only and spec["name"] != "baseline":
            continue
        t0 = time.time()
        dest = work / "arms" / spec["name"]
        st = build_arm(spec, seqs, data_dir, out_dir, cache_dir, dest, ctx)
        sc = score_dir(dest, seqs, data_dir, work / "score")
        if spec["name"] == "baseline":
            base = sc["combined"]
        c = sc["combined"]
        rows[spec["name"]] = {
            "spec": spec, "stats": st, "combined": c, "per_seq": sc["per_seq"],
            "delta_hota": None if base is None else c["GS-HOTA"] - base["GS-HOTA"],
            "delta_deta": None if base is None else c["GS-DetA"] - base["GS-DetA"],
            "seconds": round(time.time() - t0, 1)}
        logger.info("%-22s move %.3f m (n=%d, +%d rows)  HOTA %.4f (%+0.4f)  DetA %.4f  LocA %.4f",
                    spec["name"], st["displacement_m"]["median"], st["displacement_m"]["n"],
                    st["rows_added"], c["GS-HOTA"], rows[spec["name"]]["delta_hota"] or 0.0,
                    c["GS-DetA"], c["GS-LocA"])
    return {"lineage": {"predictions": str(PRED_DIR), "positions": POSITIONS_SUBDIR,
                        "cache": str(cache_dir)},
            "gt_fit": ctx["gt_fit"], "arms": rows}


def _context(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str]) -> dict:
    """Per-sequence homographies, row classification and dead-run statistics (loaded once)."""
    ctx: dict = {"ours": {}, "gt": {}, "own": {}, "runs": {}, "n_frames": {}, "gt_fit": {}}
    for seq in seqs:
        homs, err = our_homographies(seq, out_dir, cache_dir)
        gt_h, fit = gt_homographies(data_dir / seq)
        preds = load_preds(seq)
        df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
        n = int(df["frame"].max()) + 1
        ctx["ours"][seq] = (homs, err)
        ctx["gt"][seq] = gt_h
        ctx["own"][seq] = own_h_mask(preds, homs)
        ctx["n_frames"][seq] = n
        ctx["runs"][seq] = dead_runs(homs, n)
        ctx["gt_fit"][seq] = fit
        logger.info("%s: %d frames, %d solved, %d GT cameras (resid %.3f m), %d/%d rows own-H",
                    seq, n, len(homs), len(gt_h), fit["median_residual_m"] or float("nan"),
                    int(ctx["own"][seq].sum()), len(preds))
    return ctx


# === Rung 0b: the dead-frame census ===============================================================
#: Pitch landmarks, sampled in uncentred metres, for the view classification.
def _landmarks() -> dict[str, np.ndarray]:
    """Sample points of the landmarks a calibrator can lock onto (pure)."""
    t = np.linspace(0, 2 * np.pi, 48)
    box_l = np.array([[x, y] for x in (0.0, 16.5) for y in np.linspace(13.84, 54.16, 12)]
                     + [[x, y] for x in np.linspace(0, 16.5, 12) for y in (13.84, 54.16)])
    return {
        "centre_circle": np.column_stack([52.5 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)]),
        "halfway": np.column_stack([np.full(12, 52.5), np.linspace(0, 68, 12)]),
        "box_left": box_l,
        "box_right": box_l * np.array([-1.0, 1.0]) + np.array([105.0, 0.0]),
        "goal_left": np.column_stack([np.zeros(12), np.linspace(0, 68, 12)]),
        "goal_right": np.column_stack([np.full(12, 105.0), np.linspace(0, 68, 12)]),
        "touchlines": np.array([[x, y] for x in np.linspace(0, 105, 24) for y in (0.0, 68.0)]),
    }


def visible_landmarks(h: np.ndarray, size: tuple[int, int] = (1920, 1080)) -> dict[str, int]:
    """How many sample points of each landmark fall inside the image under ``h`` (pure)."""
    from generator.calibrate import apply_homography  # noqa: PLC0415

    try:
        inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return {}
    out = {}
    for name, pts in _landmarks().items():
        img = apply_homography(inv, pts)
        ok = (np.isfinite(img).all(axis=1) & (img[:, 0] >= 0) & (img[:, 0] < size[0])
              & (img[:, 1] >= 0) & (img[:, 1] < size[1]))
        out[name] = int(ok.sum())
    return out


def run_anatomy(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str]) -> dict:
    """Dead-frame census + per-dead-frame evidence and view classification."""
    from tools.gsr_v10_w2_calib import load_cache  # noqa: PLC0415

    per_seq: dict[str, dict] = {}
    total = {"frames": 0, "dead_after_gate": 0, "dead_after_fill": 0, "rows_lost_dead": 0,
             "rows_lost_live": 0, "dead_with_hypotheses": 0, "dead_central_view": 0,
             "dead_no_gt_camera": 0, "dead_two_plus_pts": 0, "dead_four_plus_pts": 0}
    for seq in seqs:
        df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
        homs, _err = our_homographies(seq, out_dir, cache_dir)
        gt_h, _fit = gt_homographies(data_dir / seq)
        by_frame = load_cache(cache_dir / f"{seq}.npz")
        people = df[df["role"].isin(["player", "goalkeeper", "referee"])]
        live_after_fill = set(people.loc[np.isfinite(people["pitch_x"]), "frame"].astype(int))
        all_frames = set(people["frame"].astype(int))
        dead = sorted(all_frames - live_after_fill)
        dead_gate = sorted(all_frames - set(homs))
        lost = people[~np.isfinite(people["pitch_x"]) & np.isfinite(people["image_x"])]
        rows_dead = int(lost["frame"].astype(int).isin(dead).sum())
        views: dict[str, int] = {}
        n_hyp = n2 = n4 = n_nogt = 0
        for fr in dead:
            cands = by_frame.get(int(fr)) or []
            n_hyp += bool(cands)
            pts = max((c.n_points for c in cands), default=0)
            n2 += pts >= 2
            n4 += pts >= 4
            h = gt_h.get(int(fr))
            if h is None:
                n_nogt += 1
                continue
            vis = visible_landmarks(h)
            circle = vis.get("centre_circle", 0) > 0
            boxes = vis.get("box_left", 0) + vis.get("box_right", 0) > 0
            tag = ("central_view" if circle and not boxes else
                   "box_in_view" if boxes else "circle_and_box" if circle else "featureless")
            views[tag] = views.get(tag, 0) + 1
        rec = {"frames": len(all_frames), "dead_after_gate": len(dead_gate),
               "dead_after_fill": len(dead), "rows_lost_on_dead_frames": rows_dead,
               "rows_lost_on_live_frames": int(len(lost)) - rows_dead,
               "dead_with_hypotheses": n_hyp, "dead_hyp_ge2_pts": n2, "dead_hyp_ge4_pts": n4,
               "dead_no_gt_camera": n_nogt, "views": views}
        per_seq[seq] = rec
        total["frames"] += rec["frames"]
        total["dead_after_gate"] += rec["dead_after_gate"]
        total["dead_after_fill"] += rec["dead_after_fill"]
        total["rows_lost_dead"] += rows_dead
        total["rows_lost_live"] += rec["rows_lost_on_live_frames"]
        total["dead_with_hypotheses"] += n_hyp
        total["dead_two_plus_pts"] += n2
        total["dead_four_plus_pts"] += n4
        total["dead_central_view"] += views.get("central_view", 0)
        total["dead_no_gt_camera"] += n_nogt
        logger.info("%s: %d frames, dead %d -> %d after fill, %d rows; views %s", seq,
                    rec["frames"], rec["dead_after_gate"], rec["dead_after_fill"], rows_dead, views)
    return {"per_seq": per_seq, "total": total}


# === Rung 1: the RTS smoother =====================================================================
#: ``{arm: (smoother parameterisation, which frames the smoothed camera is used on)}``.
#: ``A``/``B`` are the registered arms; ``C``/``D``/``E`` are post-registration diagnostics added
#: after the camera-space series turned out to be degenerate in [Cx, Cy, Cz] (see the report).
SMOOTH_ARMS: dict[str, tuple[str, str]] = {
    "A": ("camera", "all"), "B": ("camera", "dead"), "C": ("camera", "solved"),
    "D": ("mapping", "all"), "E": ("mapping", "solved"),
}


def build_smoothed_positions(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str],
                             arms: tuple[str, ...]) -> dict:
    """Write one positions parquet set per smoother arm (see :data:`SMOOTH_ARMS`)."""
    from generator.camera_track import smooth_homographies  # noqa: PLC0415
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415
    from tools.gsr_w2_calibswap import swap_positions  # noqa: PLC0415

    stats: dict[str, dict] = {a: {} for a in arms}
    for seq in seqs:
        df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
        homs, err = our_homographies(seq, out_dir, cache_dir)
        n = int(df["frame"].max()) + 1
        for arm in arms:
            mode, where = SMOOTH_ARMS[arm]
            sm, st = smooth_homographies(homs, n, err_m=err, dead_only=(where == "dead"),
                                         mode=mode)
            if where == "dead":  # solved frames untouched; the smoother only replaces the fill
                use = dict(homs) | sm
            elif where == "solved":  # smooth the solves, leave the dead frames to the fill
                use = {f: h for f, h in sm.items() if f in homs}
            else:
                use = sm
            tab, sw = swap_positions(df, use, origin=(0.0, 0.0), mode="t1")
            tab = fill_calibration_gaps(tab, max_gap=FILL_GAP)
            sub = out_dir / f"positions_v10w4{arm}_v6det"
            sub.mkdir(parents=True, exist_ok=True)
            tab.to_parquet(sub / f"{seq}.parquet", index=False)
            st.update({k: sw[k] for k in ("pitch_rows_control", "pitch_rows_final")})
            st["pitch_rows_after_fill"] = int(np.isfinite(tab["pitch_x"]).sum())
            stats[arm][seq] = st
        logger.info("%s: %s", seq, {a: (stats[a][seq]["parameterised"], stats[a][seq]["predicted"],
                                        stats[a][seq]["clamped"],
                                        stats[a][seq]["pitch_rows_after_fill"]) for a in arms})
    return stats


def smoother_accuracy(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str]) -> dict:
    """GT-anchored homography accuracy of the raw solve vs the smoothed camera (v10-W2 instrument)."""
    from generator.camera_track import smooth_homographies  # noqa: PLC0415
    from tools.gsr_v10_w2_calib import _stats, homography_error  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    from generator.postprocess import _donor_homography  # noqa: PLC0415

    keys = ("raw", "camera", "mapping", "dead_camera", "dead_mapping", "dead_fill")
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    per_seq: dict[str, dict] = {}
    for seq in seqs:
        df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{seq}.parquet")
        homs, err = our_homographies(seq, out_dir, cache_dir)
        gt = gt_people(data_dir / seq)
        n = int(df["frame"].max()) + 1
        sm = {m: smooth_homographies(homs, n, err_m=err, mode=m)[0] for m in ("camera", "mapping")}
        e = {"raw": homography_error(homs, gt)}
        for m in ("camera", "mapping"):
            e[m] = homography_error({f: sm[m][f] for f in homs if f in sm[m]}, gt)
            e["dead_" + m] = homography_error({f: h for f, h in sm[m].items() if f not in homs}, gt)
        # what the shipped fill puts on those same frames, so "better than fill" is measurable
        donors = sorted(homs)
        dead = set(e["dead_camera"]) | set(e["dead_mapping"])
        e["dead_fill"] = homography_error(
            {f: h for f in dead if (h := _donor_homography(f, donors, homs, FILL_GAP)) is not None},
            gt)
        shared = sorted(set(e["raw"]) & set(e["camera"]) & set(e["mapping"]))
        vals = {k: (np.concatenate([e[k][f] for f in shared]) if shared else np.zeros(0))
                for k in ("raw", "camera", "mapping")}
        for k in ("dead_camera", "dead_mapping", "dead_fill"):
            vals[k] = (np.concatenate([e[k][f] for f in sorted(e[k])]) if e[k] else np.zeros(0))
        for k in keys:
            pooled[k].append(vals[k])
        per_seq[seq] = {k: _stats(vals[k]) for k in keys}
        logger.info("%s: raw %.4f -> camera %.4f / mapping %.4f m (n=%d) | dead: camera %.3f "
                    "mapping %.3f fill %.3f", seq, per_seq[seq]["raw"].get("median", float("nan")),
                    per_seq[seq]["camera"].get("median", float("nan")),
                    per_seq[seq]["mapping"].get("median", float("nan")), len(vals["raw"]),
                    per_seq[seq]["dead_camera"].get("median", float("nan")),
                    per_seq[seq]["dead_mapping"].get("median", float("nan")),
                    per_seq[seq]["dead_fill"].get("median", float("nan")))
    out: dict = {"per_seq": per_seq, "pooled": {}, "paired": {}}
    for k in keys:
        v = [x for x in pooled[k] if len(x)]
        out["pooled"][k] = _stats(np.concatenate(v) if v else np.zeros(0))
    from scipy.stats import wilcoxon  # noqa: PLC0415

    base = np.concatenate([x for x in pooled["raw"] if len(x)])
    for k in ("camera", "mapping"):
        arm = np.concatenate([x for x in pooled[k] if len(x)])
        if len(arm) != len(base) or not len(arm):
            continue
        d = arm - base
        out["paired"][k] = {"mean_delta_m": float(d.mean()), "rows_better": int((d < 0).sum()),
                            "rows_worse": int((d > 0).sum()),
                            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
    return out


def score_smooth_arms(data_dir: Path, out_dir: Path, seqs: list[str],
                      arms: tuple[str, ...]) -> dict:
    """Run the frozen v6 chain per smoother arm and pair it against the same-lineage control."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v6det import TAU  # noqa: PLC0415
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache_v6det"
    raw: dict[str, dict] = {}
    for arm in arms:
        sub = POSITIONS_SUBDIR if arm == "ctrl" else f"positions_v10w4{arm}_v6det"
        res = eiou.run_point(data_dir, out_dir, seqs,
                             eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                             embedder="clip_v6det", tau=TAU, tag=f"v10w4_{arm}",
                             percrop_variant="_v6_v6det", positions_subdir=sub)
        h = res["gs_hota"]
        raw[arm] = {"positions_subdir": sub, "gs_hota": h,
                    "gs_hota_per_seq": res["gs_hota_per_seq"],
                    "eiou_tracks": (res["eiou"]["n_tracks_before"], res["eiou"]["n_tracks_after"])}
        print(f"{arm:<6} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
              f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f}", flush=True)
    ctrl = out_dir / "deleak_v10w4_ctrl" / "predictions" / "data"
    paired = {a: score_pair(ctrl, out_dir / f"deleak_v10w4_{a}" / "predictions" / "data",
                            data_dir, out_dir / "v10_w4" / "pair" / a, seqs)
              for a in arms if a != "ctrl"}
    return {"arms": raw, "paired_vs_ctrl": paired}


# === self-check ===================================================================================
def _demo() -> None:
    """Assert the pure seams: the coordinate convention, the row classifier and the dead simulator."""
    preds = [{"image_id": "2021000004", "track_id": 1, "category_id": 1,
              "attributes": {"role": "player", "team": "left", "jersey": None},
              "bbox_image": {"x": 92.0, "y": 460.0, "x_center": 100.0, "y_center": 480.0,
                             "w": 16, "h": 40},
              "bbox_pitch": {k: v for k, v in
                             (("x_bottom_left", -47.5), ("y_bottom_left", -9.0),
                              ("x_bottom_middle", -47.5), ("y_bottom_middle", -9.0),
                              ("x_bottom_right", -47.5), ("y_bottom_right", -9.0))},
              "confidence": 0.9}]
    assert frame_of("2021000004") == 3
    assert np.allclose(foot_points(preds), [[100.0, 500.0]])
    assert np.allclose(pitch_points(preds), [[5.0, 25.0]])  # centred -> uncentred
    h = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 1.0]])  # 20 px = 1 m
    assert own_h_mask(preds, {3: h}).tolist() == [True], "5 m, 25 m is exactly this H's image"
    assert own_h_mask(preds, {3: h * np.array([[1.1], [1.0], [1.0]])}).tolist() == [False]
    set_pitch(preds, np.array([[10.0, 30.0]]), np.ones(1, bool))
    assert preds[0]["bbox_pitch"]["x_bottom_middle"] == -42.5
    assert np.allclose(pitch_points(preds), [[10.0, 30.0]])

    homs = {i: h for i in range(20) if not 8 <= i < 12}
    assert dead_runs(homs, 20) == [4], dead_runs(homs, 20)
    sim, killed = simulate_dead(homs, 20, 0.5, [3], np.random.default_rng(0))
    assert killed and killed.issubset(set(homs)), killed
    assert all(np.isfinite(v).all() for v in sim.values())
    vis = visible_landmarks(np.array([[0.06, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, 0.0, 1.0]]))
    assert vis["centre_circle"] > 0 and vis["box_left"] > 0, vis  # a wide view sees both
    print("gsr_v10_w4 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--work", type=Path, default=Path("outputs/gsr/v10_w4"))
    ap.add_argument("--results", type=Path,
                    default=Path("results/gsr_benchmark/gsr_v10_w4.json"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--only", default=None, help="--curve: comma list of arm names")
    ap.add_argument("--arms", default="ctrl,A,B,C,D,E")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--anatomy", action="store_true")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else DEV20
    payload: dict = {}
    if args.curve:
        payload["curve"] = run_curve(args.data_dir, args.out_dir, args.cache_dir, args.work, seqs,
                                     args.only.split(",") if args.only else None)
    if args.anatomy:
        payload["anatomy"] = run_anatomy(args.data_dir, args.out_dir, args.cache_dir, seqs)
    if args.smooth:
        payload["smooth"] = build_smoothed_positions(
            args.data_dir, args.out_dir, args.cache_dir, seqs,
            tuple(a for a in args.arms.split(",") if a != "ctrl"))
    if args.accuracy:
        payload["smoother_accuracy"] = smoother_accuracy(args.data_dir, args.out_dir,
                                                         args.cache_dir, seqs)
    if args.score:
        payload["score"] = score_smooth_arms(args.data_dir, args.out_dir, seqs,
                                             tuple(args.arms.split(",")))
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
