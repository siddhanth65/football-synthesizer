"""v10-W3: price a temporally-consistent broadcast camera TRACKER against our per-frame chain.

The instrument under measurement (BroadTrack, WACV 2025) is proprietary and lives outside this
repo; nothing here imports it. All this module consumes is a JSON file of per-frame **camera
parameters** in the SoccerNet broadcast convention (pan/tilt/roll degrees, tripod position in
metres, horizontal field of view, one normalised radial distortion coefficient) -- the same shape
any pan-tilt-zoom camera model emits. Our own chain enters through the v10-W2 candidate cache, so
the comparison holds the detector, the tracker, the embeddings and the OCR fixed and moves only the
camera.

Four measurements, registered in ``results/gsr_v10_w3_registered.json`` before any of them ran:

``--convention``  Resolve the rotation/distortion convention of the external file once, on ONE
                  sequence, against GT, and freeze it (a discrete 16 x 3 search, no free parameter).
``--accuracy``    GT-anchored position error, external vs ``ctrl``/``s4``, paired on shared frames.
``--coverage``    Frames each side answers, and the external error on the frames our gate rejects.
``--jitter``      Frame-to-frame (and second-difference) motion of a fixed image grid's pitch
                  image -- temporal smoothness, which association consumes.
``--export``      Per-frame ``3x3`` homographies for :mod:`tools.gsr_w2_calibswap` (stage 2), with
                  the homography-fit residual of the distortion model reported alongside.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("gsr_v10_w3_camtrack")

#: The 10 DEV-20 sequences of the v8-W2 probe, unchanged.
PROBE = ["SNGS-024", "SNGS-027", "SNGS-039", "SNGS-042", "SNGS-045",
         "SNGS-048", "SNGS-051", "SNGS-054", "SNGS-057", "SNGS-078"]
#: Artifact suffix of the v6 (S4b-detector) lineage.
VARIANT = "_v6det"
#: Our shipped positions for that lineage.
CONTROL_SUBDIR = "positions_gate" + VARIANT
#: Image grid used by the jitter instrument (fraction of width/height).
GRID = np.stack(np.meshgrid(np.linspace(0.15, 0.85, 5), np.linspace(0.35, 0.9, 5)),
                axis=-1).reshape(-1, 2)


@dataclass(frozen=True)
class Convention:
    """One candidate reading of an external camera-parameter file.

    Attributes:
        mirror: Apply ``diag(-1, -1, 1)`` between roll and tilt (the SoccerNet convention does).
        s_pan: Sign of the pan angle.
        s_tilt: Sign of the tilt angle.
        s_roll: Sign of the roll angle.
        distortion: ``'none'``, ``'inverse'`` (the coefficient undistorts) or ``'forward'``
            (the coefficient distorts, so it must be inverted numerically).
    """

    mirror: bool
    s_pan: float
    s_tilt: float
    s_roll: float
    distortion: str = "none"

    @property
    def tag(self) -> str:
        """Short readable id, e.g. ``m1p+t+r-_forward``."""
        return (f"m{int(self.mirror)}p{self.s_pan:+.0f}t{self.s_tilt:+.0f}"
                f"r{self.s_roll:+.0f}_{self.distortion}")


def _rz(a: float) -> np.ndarray:
    """Rotation about the z axis by ``a`` radians."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    """Rotation about the x axis by ``a`` radians."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def camera_frame(cp: dict, conv: Convention) -> tuple[np.ndarray, np.ndarray, float, float,
                                                      float, float]:
    """Unpack one camera-parameter dict into ``(R, C, f, cx, cy, k1)``.

    Args:
        cp: Camera parameters (SoccerNet broadcast naming).
        conv: The reading convention under test.

    Returns:
        Rotation world->camera, camera centre (m), focal length (px), principal point (px) and
        the single radial distortion coefficient.
    """
    w = float(cp["sensorResolutionWidthPixels"])
    h = float(cp["sensorResolutionHeightPixels"])
    fov = np.radians(float(cp["horizontalFieldOfViewDegrees"]))
    f = (w / 2.0) / np.tan(fov / 2.0)
    rot = (_rz(conv.s_roll * np.radians(float(cp["rollDegrees"])))
           @ (np.diag([-1.0, -1.0, 1.0]) if conv.mirror else np.eye(3))
           @ _rx(conv.s_tilt * np.radians(float(cp["tiltDegrees"])))
           @ _rz(conv.s_pan * np.radians(float(cp["panDegrees"]))))
    centre = np.array([float(cp["positionXMeters"]), float(cp["positionYMeters"]),
                       float(cp["positionZMeters"])])
    k = cp.get("normalizedRadialDistortionCoefficients") or [0.0]
    return rot, centre, f, w / 2.0, h / 2.0, float(k[0])


def image_to_pitch(cp: dict, pts: np.ndarray, conv: Convention) -> np.ndarray:
    """Back-project image points onto the ground plane, in our corner-origin metres.

    Args:
        cp: Camera parameters.
        pts: ``(N, 2)`` image points (pixels).
        conv: Reading convention.

    Returns:
        ``(N, 2)`` pitch coordinates; ``NaN`` where the ray does not meet the ground in front of
        the camera.
    """
    from core.pitch import PITCH_LEN, PITCH_WID  # noqa: PLC0415

    rot, centre, f, cx, cy, k1 = camera_frame(cp, conv)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    xn = (pts[:, 0] - cx) / f
    yn = (pts[:, 1] - cy) / f
    if conv.distortion == "inverse":  # the coefficient maps observed -> ideal directly
        r2 = xn * xn + yn * yn
        s = 1.0 + k1 * r2
        xn, yn = xn * s, yn * s
    elif conv.distortion == "forward":  # observed = ideal * (1 + k1 r_ideal^2): invert it
        ux, uy = xn.copy(), yn.copy()
        for _ in range(8):
            s = 1.0 + k1 * (ux * ux + uy * uy)
            ux, uy = xn / s, yn / s
        xn, yn = ux, uy
    rays = np.column_stack([xn, yn, np.ones(len(xn))]) @ rot  # rot.T @ d, row-wise
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -centre[2] / rays[:, 2]
    world = centre[None, :] + t[:, None] * rays
    bad = ~np.isfinite(t) | (t <= 0)
    out = world[:, :2] + np.array([PITCH_LEN / 2.0, PITCH_WID / 2.0])
    out[bad] = np.nan
    return out


def pitch_to_image(cp: dict, pts: np.ndarray, conv: Convention) -> np.ndarray:
    """Forward-project corner-origin pitch points to pixels (the self-check's inverse map)."""
    from core.pitch import PITCH_LEN, PITCH_WID  # noqa: PLC0415

    rot, centre, f, cx, cy, k1 = camera_frame(cp, conv)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2) - np.array([PITCH_LEN / 2.0, PITCH_WID / 2.0])
    world = np.column_stack([pts, np.zeros(len(pts))])
    cam = (world - centre[None, :]) @ rot.T
    xn, yn = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    if conv.distortion == "forward":
        s = 1.0 + k1 * (xn * xn + yn * yn)
        xn, yn = xn * s, yn * s
    elif conv.distortion == "inverse":
        ux, uy = xn.copy(), yn.copy()
        for _ in range(8):
            s = 1.0 + k1 * (ux * ux + uy * uy)
            ux, uy = xn / s, yn / s
        xn, yn = ux, uy
    return np.column_stack([xn * f + cx, yn * f + cy])


def load_cameras(path: Path) -> dict[int, dict]:
    """Read an external camera-parameter JSON into ``{frame index (0-based): entry}``.

    Args:
        path: JSON keyed by frame image path (``.../000123.jpg``).

    Returns:
        ``{frame: {'cp': ..., 'score': ..., 'reinit': ...}}``; entries without a ``cp`` are dropped.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for key, val in raw.items():
        if not isinstance(val, dict) or "cp" not in val:
            continue
        out[int(Path(key).stem) - 1] = val
    return out


# === convention resolution =======================================================================
def candidate_conventions() -> list[Convention]:
    """The 16 rotation readings crossed with the three distortion readings (48 candidates)."""
    return [Convention(m, sp, st, sr, d)
            for m in (True, False)
            for sp in (1.0, -1.0)
            for st in (1.0, -1.0)
            for sr in (1.0, -1.0)
            for d in ("none", "forward", "inverse")]


def convention_error(cams: dict[int, dict], gt: dict[int, np.ndarray], conv: Convention,
                     frames: list[int]) -> float:
    """Median GT-anchored error (m) of one convention over ``frames`` (``inf`` if unusable)."""
    errs = []
    for fr in frames:
        g = gt.get(fr)
        cam = cams.get(fr)
        if g is None or not len(g) or cam is None:
            continue
        proj = image_to_pitch(cam["cp"], g[:, :2], conv)
        d = np.linalg.norm(proj - g[:, 2:4], axis=1)
        errs.append(d[np.isfinite(d)])
    if not errs:
        return float("inf")
    pooled = np.concatenate(errs)
    return float(np.median(pooled)) if len(pooled) else float("inf")


def resolve_convention(data_dir: Path, cam_dir: Path, seq: str, n_frames: int = 40) -> dict:
    """Pick the reading convention that reproduces GT on ONE sequence; report the whole ranking."""
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    gt = gt_people(data_dir / seq)
    cams = load_cameras(cam_dir / f"{seq}.json")
    frames = sorted(set(gt) & set(cams))[:: max(1, len(set(gt) & set(cams)) // n_frames)]
    ranked = sorted(((convention_error(cams, gt, c, frames), c) for c in candidate_conventions()),
                    key=lambda t: t[0])
    best_err, best = ranked[0]
    logger.info("convention on %s (%d frames): best %s median %.4f m; runner-up %s %.4f m",
                seq, len(frames), best.tag, best_err, ranked[1][1].tag, ranked[1][0])
    return {"seq": seq, "n_frames": len(frames), "best": best.tag, "best_median_m": best_err,
            "ranking": [{"tag": c.tag, "median_m": (None if not np.isfinite(e) else e)}
                        for e, c in ranked[:8]]}


def convention_from_tag(tag: str) -> Convention:
    """Inverse of :attr:`Convention.tag`."""
    body, dist = tag.rsplit("_", 1)
    return Convention(mirror=body[1] == "1", s_pan=float(body[3] + "1"),
                      s_tilt=float(body[6] + "1"), s_roll=float(body[9] + "1"), distortion=dist)


# === our side ====================================================================================
def our_homographies(out_dir: Path, cache_dir: Path, arm: str, seq: str) -> dict[int, np.ndarray]:
    """Replay our shipped gate on the v10-W2 candidate cache -> ``{frame: H}`` (no GPU)."""
    import pandas as pd  # noqa: PLC0415

    from tools.gsr_v10_w2_calib import gate_homographies, load_cache  # noqa: PLC0415

    df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{seq}.parquet")
    homs, _ = gate_homographies(load_cache(cache_dir / arm / f"{seq}.npz"), df)
    return homs


def _stats(err: np.ndarray) -> dict:
    """Median / p90 / mean / within-5 m of an error array."""
    err = err[np.isfinite(err)]
    if not len(err):
        return {"n": 0}
    return {"n": int(len(err)), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)), "mean": float(err.mean()),
            "within_5m": float((err <= 5.0).mean())}


def _our_errors(homs: dict[int, np.ndarray], gt: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """GT-anchored per-frame errors for our homographies."""
    from tools.gsr_v10_w2_calib import homography_error  # noqa: PLC0415

    return homography_error(homs, gt)


def _their_errors(cams: dict[int, dict], gt: dict[int, np.ndarray],
                  conv: Convention) -> dict[int, np.ndarray]:
    """GT-anchored per-frame errors for the external camera track."""
    out: dict[int, np.ndarray] = {}
    for fr, cam in cams.items():
        g = gt.get(fr)
        if g is None or not len(g):
            continue
        proj = image_to_pitch(cam["cp"], g[:, :2], conv)
        out[fr] = np.linalg.norm(proj - g[:, 2:4], axis=1)
    return out


def accuracy_report(data_dir: Path, out_dir: Path, cache_dir: Path, cam_dir: Path,
                    seqs: list[str], conv: Convention, arms: tuple[str, ...] = ("ctrl", "s4")
                    ) -> dict:
    """Paired GT-anchored accuracy: external camera track vs each of our decode arms."""
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    per_seq: dict[str, dict] = {}
    pooled: dict[str, list[np.ndarray]] = {a: [] for a in (*arms, "ext")}
    for seq in seqs:
        gt = gt_people(data_dir / seq)
        cams = load_cameras(cam_dir / f"{seq}.json")
        errs = {a: _our_errors(our_homographies(out_dir, cache_dir, a, seq), gt) for a in arms}
        errs["ext"] = _their_errors(cams, gt, conv)
        shared = sorted(set.intersection(*[set(e) for e in errs.values()]))
        rec: dict = {"frames": {k: len(v) for k, v in errs.items()}, "frames_paired": len(shared)}
        for key, err in errs.items():
            vals = np.concatenate([err[f] for f in shared]) if shared else np.zeros(0)
            pooled[key].append(vals)
            rec[key] = _stats(vals)
        per_seq[seq] = rec
        logger.info("%s: paired %d frames | %s", seq, len(shared),
                    " ".join(f"{k} {rec[k].get('median', float('nan')):.4f}" for k in errs))
    rep: dict = {"seqs": seqs, "convention": conv.tag, "per_seq": per_seq, "pooled": {}}
    for key, chunks in pooled.items():
        rep["pooled"][key] = _stats(np.concatenate(chunks))
    base = np.concatenate(pooled[arms[-1]])
    ext = np.concatenate(pooled["ext"])
    if len(base) == len(ext) and len(base):
        from scipy.stats import wilcoxon  # noqa: PLC0415

        d = ext - base
        rep["ext_vs_" + arms[-1]] = {
            "mean_delta_m": float(d.mean()), "rows_better": int((d < 0).sum()),
            "rows_worse": int((d > 0).sum()),
            "median_ratio": float(rep["pooled"]["ext"]["median"]
                                  / rep["pooled"][arms[-1]]["median"]),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
    return rep


def coverage_report(data_dir: Path, out_dir: Path, cache_dir: Path, cam_dir: Path,
                    seqs: list[str], conv: Convention, arm: str = "s4") -> dict:
    """Who answers where: our gate, our post-fill rows, and the external track on our dead rows."""
    import pandas as pd  # noqa: PLC0415

    from generator.postprocess import PLAYER_ROLES  # noqa: PLC0415
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    per_seq: dict[str, dict] = {}
    dead_err: list[np.ndarray] = []
    live_err: list[np.ndarray] = []
    for seq in seqs:
        gt = gt_people(data_dir / seq)
        cams = load_cameras(cam_dir / f"{seq}.json")
        homs = our_homographies(out_dir, cache_dir, arm, seq)
        df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{seq}.parquet")
        people = df[df["role"].isin(PLAYER_ROLES)]
        n_frames = int(df["frame"].max()) + 1
        gate_live = set(homs)
        filled = set(people.loc[np.isfinite(people["pitch_x"].to_numpy(dtype=float)), "frame"]
                     .astype(int))
        dead_after_fill = sorted(set(range(n_frames)) - filled)
        theirs_on_dead = [f for f in dead_after_fill if f in cams]
        terr = _their_errors(cams, gt, conv)
        dead_err.append(np.concatenate([terr[f] for f in theirs_on_dead if f in terr])
                        if any(f in terr for f in theirs_on_dead) else np.zeros(0))
        live_err.append(np.concatenate([terr[f] for f in sorted(gate_live) if f in terr])
                        if any(f in terr for f in gate_live) else np.zeros(0))
        per_seq[seq] = {
            "n_frames": n_frames, "our_gate_frames": len(gate_live),
            "our_frames_after_fill": len(filled), "our_dead_after_fill": len(dead_after_fill),
            "ext_frames": len(cams),
            "ext_on_our_dead": len(theirs_on_dead),
            "ext_recovers_share": (len(theirs_on_dead) / len(dead_after_fill)
                                   if dead_after_fill else None),
            "rows_people": int(len(people)),
            "rows_people_dead": int(people["frame"].astype(int).isin(dead_after_fill).sum()),
        }
        logger.info("%s: gate %d/%d, after fill %d, dead %d, ext answers %d of them",
                    seq, len(gate_live), n_frames, len(filled), len(dead_after_fill),
                    len(theirs_on_dead))
    tot_dead = sum(v["our_dead_after_fill"] for v in per_seq.values())
    tot_rec = sum(v["ext_on_our_dead"] for v in per_seq.values())
    return {"per_seq": per_seq, "arm": arm,
            "total_dead_frames": tot_dead, "total_ext_on_dead": tot_rec,
            "recovered_share": (tot_rec / tot_dead) if tot_dead else None,
            "ext_error_on_our_dead_frames": _stats(np.concatenate(dead_err)),
            "ext_error_on_our_live_frames": _stats(np.concatenate(live_err))}


# === jitter ======================================================================================
def _grid_tracks(mapper, frames: list[int], size: tuple[float, float]) -> np.ndarray:
    """``(T, G, 2)`` pitch positions of the fixed image grid, ``NaN`` where the frame has no camera.

    Args:
        mapper: ``frame -> (G, 2)`` pitch coordinates, or ``None``.
        frames: Consecutive frame indices to evaluate.
        size: Image size (px).

    Returns:
        Array of shape ``(len(frames), len(GRID), 2)``.
    """
    del size
    out = np.full((len(frames), len(GRID), 2), np.nan)
    for i, fr in enumerate(frames):
        got = mapper(fr)
        if got is not None:
            out[i] = got
    return out


def _jitter_stats(tracks: np.ndarray) -> dict:
    """First- and second-difference statistics (metres) of a ``(T, G, 2)`` grid track."""
    d1 = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
    d2 = np.linalg.norm(tracks[2:] - 2 * tracks[1:-1] + tracks[:-2], axis=2)
    d1, d2 = d1[np.isfinite(d1)], d2[np.isfinite(d2)]
    return {"n_pairs": int(d1.size), "step_median_m": float(np.median(d1)) if d1.size else None,
            "step_p90_m": float(np.percentile(d1, 90)) if d1.size else None,
            "n_triples": int(d2.size),
            "accel_median_m": float(np.median(d2)) if d2.size else None,
            "accel_p90_m": float(np.percentile(d2, 90)) if d2.size else None,
            "accel_p99_m": float(np.percentile(d2, 99)) if d2.size else None}


def jitter_report(out_dir: Path, cache_dir: Path, cam_dir: Path, seqs: list[str],
                  conv: Convention, arms: tuple[str, ...] = ("ctrl", "s4"),
                  size: tuple[float, float] = (1920.0, 1080.0)) -> dict:
    """Temporal smoothness of each camera source, on a fixed image grid.

    Both sides are measured on the SAME frames (those every source answers), so a difference is
    smoothness and not coverage.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    grid = GRID * np.array(size)
    per_seq: dict[str, dict] = {}
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in (*arms, "ext")}
    for seq in seqs:
        cams = load_cameras(cam_dir / f"{seq}.json")
        homs = {a: our_homographies(out_dir, cache_dir, a, seq) for a in arms}
        shared = sorted(set(cams).intersection(*[set(h) for h in homs.values()]))
        # keep only maximal runs of consecutive frames, so differences are true time differences
        runs: list[list[int]] = []
        for fr in shared:
            if runs and fr == runs[-1][-1] + 1:
                runs[-1].append(fr)
            else:
                runs.append([fr])
        runs = [r for r in runs if len(r) >= 3]
        rec: dict = {"n_runs": len(runs), "n_frames_in_runs": sum(len(r) for r in runs)}
        for key in (*arms, "ext"):
            chunks = []
            for run in runs:
                if key == "ext":
                    tr = _grid_tracks(lambda fr: image_to_pitch(cams[fr]["cp"], grid, conv),
                                      run, size)
                else:
                    h = homs[key]
                    tr = _grid_tracks(lambda fr, h=h: apply_homography(h[fr], grid), run, size)
                chunks.append(tr)
            rec[key] = _jitter_stats(np.concatenate(chunks, axis=0)) if chunks else {}
            pooled[key].extend(chunks)
        per_seq[seq] = rec
        logger.info("%s: runs %d | %s", seq, len(runs),
                    " ".join(f"{k} accel {rec[k].get('accel_median_m', float('nan')):.4f}"
                             for k in (*arms, "ext")))
    rep: dict = {"per_seq": per_seq, "convention": conv.tag, "pooled": {}}
    for key, chunks in pooled.items():
        d1 = np.concatenate([np.linalg.norm(np.diff(c, axis=0), axis=2).ravel() for c in chunks])
        d2 = np.concatenate([np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=2).ravel()
                             for c in chunks])
        rep["pooled"][key] = {"step_median_m": float(np.nanmedian(d1)),
                              "step_p90_m": float(np.nanpercentile(d1, 90)),
                              "accel_median_m": float(np.nanmedian(d2)),
                              "accel_p90_m": float(np.nanpercentile(d2, 90)),
                              "accel_p99_m": float(np.nanpercentile(d2, 99))}
    return rep


# === export for the end-to-end arm ===============================================================
def export_homographies(cam_dir: Path, dest: Path, seqs: list[str], conv: Convention,
                        size: tuple[float, float] = (1920.0, 1080.0)) -> dict:
    """Write ``<SEQ>/<frame>.npy`` image->corner-origin-pitch homographies for stage 2.

    The external model carries radial distortion, which a ``3x3`` cannot express; the homography is
    least-squares fitted to the exact map over an image grid and the fit residual is reported, so
    the approximation is measured rather than assumed.
    """
    import cv2  # noqa: PLC0415

    fit = np.stack(np.meshgrid(np.linspace(0.05, 0.95, 12), np.linspace(0.25, 0.98, 10)),
                   axis=-1).reshape(-1, 2) * np.array(size)
    stats: dict[str, dict] = {}
    for seq in seqs:
        cams = load_cameras(cam_dir / f"{seq}.json")
        sub = dest / seq
        sub.mkdir(parents=True, exist_ok=True)
        res: list[float] = []
        written = 0
        for fr, cam in cams.items():
            target = image_to_pitch(cam["cp"], fit, conv)
            ok = np.isfinite(target).all(axis=1)
            if ok.sum() < 12:
                continue
            h, _ = cv2.findHomography(fit[ok], target[ok], 0)
            if h is None:
                continue
            from generator.calibrate import apply_homography  # noqa: PLC0415

            res.extend(np.linalg.norm(apply_homography(h, fit[ok]) - target[ok], axis=1).tolist())
            np.save(sub / f"{fr + 1:06d}.npy", h)
            written += 1
        arr = np.asarray(res)
        stats[seq] = {"frames_written": written, "fit_residual_median_m": float(np.median(arr)),
                      "fit_residual_p90_m": float(np.percentile(arr, 90)),
                      "fit_residual_max_m": float(arr.max())}
        logger.info("%s: %d homographies, fit residual median %.4f m p90 %.4f m", seq, written,
                    stats[seq]["fit_residual_median_m"], stats[seq]["fit_residual_p90_m"])
    return stats


# === the licence-free follow-up: smooth OUR OWN cameras ==========================================
def smooth_homographies(homs: dict[int, np.ndarray], window: int,
                        size: tuple[float, float] = (1920.0, 1080.0)) -> dict[int, np.ndarray]:
    """Temporally smooth a per-frame homography track (centred moving average, exact on a linear pan).

    Four fixed image points are pushed to the pitch, the four resulting tracks are smoothed over
    time inside each run of consecutive calibrated frames, and the homography is re-fitted from the
    four correspondences. Homographies are not a vector space; their images of fixed points are.

    Args:
        homs: ``{frame: H}`` image -> pitch.
        window: Odd number of frames in the centred average (1 = no smoothing).
        size: Image size (px).

    Returns:
        ``{frame: H}`` with the same keys.
    """
    import cv2  # noqa: PLC0415

    from generator.calibrate import apply_homography  # noqa: PLC0415

    if window <= 1:
        return dict(homs)
    ref = np.array([[0.15, 0.35], [0.85, 0.35], [0.85, 0.90], [0.15, 0.90]]) * np.array(size)
    frames = sorted(homs)
    runs: list[list[int]] = []
    for fr in frames:
        if runs and fr == runs[-1][-1] + 1:
            runs[-1].append(fr)
        else:
            runs.append([fr])
    out: dict[int, np.ndarray] = {}
    half = window // 2
    for run in runs:
        pts = np.stack([apply_homography(homs[f], ref) for f in run])  # (T, 4, 2)
        for i, fr in enumerate(run):
            lo, hi = max(0, i - half), min(len(run), i + half + 1)
            target = pts[lo:hi].mean(axis=0)
            h, _ = cv2.findHomography(ref, target, 0)
            out[fr] = h if h is not None else homs[fr]
    return out


def smoothing_report(data_dir: Path, out_dir: Path, cache_dir: Path, seqs: list[str],
                     windows: tuple[int, ...], arm: str = "s4",
                     export_dir: Path | None = None) -> dict:
    """Exploratory: what does temporal smoothing of OUR OWN homographies do to error and jitter?"""
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    grid = GRID * np.array([1920.0, 1080.0])
    rep: dict = {"arm": arm, "windows": list(windows), "per_seq": {}, "pooled": {}}
    pooled: dict[str, list[np.ndarray]] = {str(w): [] for w in windows}
    jit: dict[str, list[np.ndarray]] = {str(w): [] for w in windows}
    for seq in seqs:
        gt = gt_people(data_dir / seq)
        raw = our_homographies(out_dir, cache_dir, arm, seq)
        rec: dict = {}
        for w in windows:
            homs = smooth_homographies(raw, w)
            err = _our_errors(homs, gt)
            vals = np.concatenate([err[f] for f in sorted(err)]) if err else np.zeros(0)
            pooled[str(w)].append(vals)
            rec[str(w)] = _stats(vals)
            frames = sorted(homs)
            runs: list[list[int]] = []
            for fr in frames:
                if runs and fr == runs[-1][-1] + 1:
                    runs[-1].append(fr)
                else:
                    runs.append([fr])
            from generator.calibrate import apply_homography  # noqa: PLC0415

            tr = [np.stack([apply_homography(homs[f], grid) for f in r])
                  for r in runs if len(r) >= 3]
            if tr:
                jit[str(w)].append(np.concatenate(
                    [np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=2).ravel() for c in tr]))
            if export_dir is not None and w == windows[-1]:
                sub = export_dir / seq
                sub.mkdir(parents=True, exist_ok=True)
                for fr, h in homs.items():
                    np.save(sub / f"{fr + 1:06d}.npy", h)
        rep["per_seq"][seq] = rec
        logger.info("%s: %s", seq, " ".join(
            f"w{w} {rec[str(w)]['median']:.4f}" for w in windows))
    for w in windows:
        rep["pooled"][str(w)] = _stats(np.concatenate(pooled[str(w)]))
        rep["pooled"][str(w)]["accel_median_m"] = float(np.median(np.concatenate(jit[str(w)])))
    return rep


def failure_report(cam_dir: Path, seqs: list[str]) -> dict:
    """The external tracker's own confidence/reinit trace, plus its worst runs."""
    per_seq: dict[str, dict] = {}
    for seq in seqs:
        cams = load_cameras(cam_dir / f"{seq}.json")
        frames = sorted(cams)
        score = np.array([float(cams[f].get("score", np.nan)) for f in frames])
        reinit = [f for f in frames if cams[f].get("reinit")]
        fov = np.array([float(cams[f]["cp"]["horizontalFieldOfViewDegrees"]) for f in frames])
        pos = np.array([[float(cams[f]["cp"][k]) for k in
                         ("positionXMeters", "positionYMeters", "positionZMeters")]
                        for f in frames])
        per_seq[seq] = {
            "n_frames": len(frames),
            "score_median": float(np.nanmedian(score)), "score_p10": float(np.nanpercentile(score, 10)),
            "score_min": float(np.nanmin(score)), "reinit_frames": reinit,
            "fov_min": float(fov.min()), "fov_max": float(fov.max()),
            "position_std_m": [float(v) for v in pos.std(axis=0)],
            "position_median_m": [float(v) for v in np.median(pos, axis=0)],
            "ms_per_frame_median": float(np.median([float(cams[f].get("time", np.nan))
                                                    for f in frames])),
        }
    return {"per_seq": per_seq}


# === self-check ==================================================================================
def _demo() -> None:
    """Assert the camera algebra, the convention search and the jitter statistics."""
    cp = {"panDegrees": -30.0, "tiltDegrees": 82.0, "rollDegrees": 0.7,
          "positionXMeters": 13.0, "positionYMeters": 72.0, "positionZMeters": -12.8,
          "horizontalFieldOfViewDegrees": 21.3, "sensorResolutionWidthPixels": 1920.0,
          "sensorResolutionHeightPixels": 1080.0,
          "normalizedRadialDistortionCoefficients": [0.09]}
    # 1. round trip: wherever the back-projection answers at all, it must invert the forward map
    #    exactly -- for every one of the 48 candidate readings, distortion included.
    pitch = np.array([[52.5, 34.0], [30.0, 20.0], [70.0, 50.0]])
    full: set[str] = set()
    for conv in candidate_conventions():
        back = image_to_pitch(cp, pitch_to_image(cp, pitch, conv), conv)
        fin = np.isfinite(back).all(axis=1)
        assert np.allclose(back[fin], pitch[fin], atol=1e-3), (conv.tag, back)
        if fin.all():
            full.add(conv.distortion)
    assert full == {"none", "forward", "inverse"}, full  # each reading is realisable
    # 2. distortion actually does something (so the search is not comparing identical maps)
    good = next(c for c in candidate_conventions()
                if np.isfinite(image_to_pitch(cp, np.array([[100.0, 1000.0]]), c)).all())
    a = image_to_pitch(cp, np.array([[100.0, 1000.0]]), Convention(*vars(good).values()))
    b = image_to_pitch(cp, np.array([[100.0, 1000.0]]),
                       Convention(good.mirror, good.s_pan, good.s_tilt, good.s_roll,
                                  "forward" if good.distortion != "forward" else "inverse"))
    assert np.linalg.norm(a - b) > 0.05, (a, b)
    # 3. tag round trip
    for c in candidate_conventions()[:6]:
        assert convention_from_tag(c.tag) == c, c.tag
    # 4. jitter: a perfectly linear pan has zero acceleration, a jittered one does not
    lin = np.cumsum(np.ones((10, 4, 2)) * 0.3, axis=0)
    assert _jitter_stats(lin)["accel_median_m"] < 1e-9
    noisy = lin + (np.arange(10) % 2)[:, None, None] * np.array([0.1, 0.0])
    assert _jitter_stats(noisy)["accel_median_m"] > 0.05, _jitter_stats(noisy)
    assert abs(_jitter_stats(lin)["step_median_m"] - np.hypot(0.3, 0.3)) < 1e-9
    # 5. smoothing: window 1 is the identity; a jittered translation track loses its jitter
    from generator.calibrate import apply_homography  # noqa: PLC0415

    base = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 1.0]])
    track = {}
    for t in range(12):
        h = base.copy()
        h[0, 2] = 0.2 * t + (0.3 if t % 2 else 0.0)  # linear pan + a 0.3 m square-wave jitter
        track[t] = h
    assert all(np.allclose(smooth_homographies(track, 1)[t], track[t]) for t in track)
    sm = smooth_homographies(track, 5)
    grid = np.array([[400.0, 600.0], [1400.0, 900.0]])
    raw_a = _jitter_stats(np.stack([apply_homography(track[t], grid) for t in sorted(track)]))
    sm_a = _jitter_stats(np.stack([apply_homography(sm[t], grid) for t in sorted(sm)]))
    assert sm_a["accel_median_m"] < 0.25 * raw_a["accel_median_m"], (raw_a, sm_a)
    print("gsr_v10_w3_camtrack demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/gsr/v10_w2_full"),
                    help="v10-W2 candidate cache (the FULL-stride one: v10_w2_s5 is stride 5)")
    ap.add_argument("--cam-dir", type=Path, required=False,
                    help="directory of <SEQ>.json external camera-parameter files (outside the repo)")
    ap.add_argument("--results", type=Path, default=Path("results/gsr_benchmark/gsr_v10_w3.json"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--arms", default="ctrl,s4")
    ap.add_argument("--conv", default=None, help="frozen convention tag (else --convention first)")
    ap.add_argument("--export-dir", type=Path, default=None)
    ap.add_argument("--convention", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--smooth", default=None, help="comma list of odd windows, e.g. 1,5,9")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the result keys (e.g. pass1/pass2)")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return

    seqs = args.seqs.split(",") if args.seqs else PROBE
    arms = tuple(args.arms.split(","))
    payload: dict = {}
    conv = convention_from_tag(args.conv) if args.conv else None

    if args.convention:
        rep = resolve_convention(args.data_dir, args.cam_dir, seqs[0])
        payload["convention"] = rep
        conv = conv or convention_from_tag(rep["best"])
    if (args.accuracy or args.coverage or args.jitter or args.export) and conv is None:
        raise SystemExit("--conv TAG (or --convention) is required")
    if args.accuracy:
        payload["accuracy"] = accuracy_report(args.data_dir, args.out_dir, args.cache_dir,
                                              args.cam_dir, seqs, conv, arms)
        for k, v in payload["accuracy"]["pooled"].items():
            print(f"{k:<5} n={v['n']:>7} median {v['median']:.4f} m  p90 {v['p90']:.4f}  "
                  f"mean {v['mean']:.4f}  <=5m {v['within_5m']:.5f}")
    if args.coverage:
        payload["coverage"] = coverage_report(args.data_dir, args.out_dir, args.cache_dir,
                                              args.cam_dir, seqs, conv, arms[-1])
        c = payload["coverage"]
        print(f"dead frames {c['total_dead_frames']}, external answers {c['total_ext_on_dead']} "
              f"({c['recovered_share']}), error there {c['ext_error_on_our_dead_frames']}")
    if args.jitter:
        payload["jitter"] = jitter_report(args.out_dir, args.cache_dir, args.cam_dir, seqs,
                                          conv, arms)
        for k, v in payload["jitter"]["pooled"].items():
            print(f"{k:<5} step median {v['step_median_m']:.4f} m  accel median "
                  f"{v['accel_median_m']:.4f} m  p90 {v['accel_p90_m']:.4f}")
    if args.failures:
        payload["failures"] = failure_report(args.cam_dir, seqs)
    if args.export:
        payload["export"] = export_homographies(args.cam_dir, args.export_dir, seqs, conv)
    if args.smooth:
        windows = tuple(int(v) for v in args.smooth.split(","))
        payload["smoothing"] = smoothing_report(args.data_dir, args.out_dir, args.cache_dir,
                                                seqs, windows, arms[-1], args.export_dir)
        for w, v in payload["smoothing"]["pooled"].items():
            print(f"window {w:>3} n={v['n']:>7} median {v['median']:.4f} m  p90 {v['p90']:.4f}  "
                  f"accel {v['accel_median_m']:.4f}")

    if payload:
        if args.tag:
            payload = {f"{k}_{args.tag}": v for k, v in payload.items()}
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
