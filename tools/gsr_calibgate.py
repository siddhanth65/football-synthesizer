"""Recover the calibration dropout at its real source, and measure it on SoccerNet-GSR.

`results/GSR_ASSOCIATION.md` section 3.2 blamed the dropout on homographies that pass the 2.0 m
keypoint-reprojection gate while projecting every player off the pitch. **That diagnosis is wrong**,
and this module is what measured it (`results/GSR_CALIBGATE.md` section 1): on 96.4% of the dead
frames every detected player lands *on* the pitch, and those projections are 82-98% within the
evaluator's 5 m tolerance of a real ground-truth player. What discards them is
:func:`generator.postprocess.reject_implausible_frames`, whose ">= 8 players spanning >= 25 m" rule
encodes a wide-shot assumption a zoomed broadcast frame cannot meet.

So the fix is the trust rule (:data:`generator.postprocess.TRUST_MIN_PLAYERS` /
:data:`~generator.postprocess.TRUST_MIN_SPAN_M`, weakened to 3 players / 5 m on DEV-20), applied
inside the calibration gate via :func:`generator.calibrate.select_calibration` so a frame is judged
*before* its coordinates are thrown away. The gate also walks PnLCalib's other 17 hypotheses when
its own pick fails -- free, since they are already computed, but worth only 0.8% of the dead frames.

The discarded coordinates are gone from the cached parquets, so restoring them needs the frame's
homography back. Re-extracting would invalidate the track ids every cached jersey vote and PRTreID
embedding is keyed by, so the measurement is split in two:

* ``--cache`` (GPU): re-run PnLCalib on the frames that currently emit **no** pitch position and
  store every candidate homography. Frames that already emit positions are untouched (measured:
  236 of 240 controls keep an identical pick; the rest is ``cv2.RANSAC`` noise).
* everything else (CPU): re-select per frame, re-project the cached detections, rescore.

CLI::

    python -m tools.gsr_calibgate --cache --worst 8      # GPU: the calibration-sick sequences
    python -m tools.gsr_calibgate --diag --worst 8       # frame-level diagnosis of the cache
    python -m tools.gsr_calibgate --sweep --seqs <DEV>   # choose the trust rule on DEV-20
    python -m tools.gsr_calibgate --compare --seqs ...   # no-fill vs fill vs gate, solver arms
    python -m tools.gsr_calibgate --freeze
    python -m tools.gsr_calibgate --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR
from generator.calibrate import CalibCandidate, select_calibration
from generator.postprocess import PLAYER_ROLES, clamp_to_pitch, onpitch_plausible

logger = logging.getLogger("gsr_calibgate")

#: Per-sequence candidate caches (one npz per sequence, flat arrays keyed by frame).
CACHE_SUBDIR = "calib_candidates"
#: Where the re-gated positions land (the originals are never overwritten).
GATED_SUBDIR = "positions_gate"
_MODES = ("full", "ground_plane", "main")


def player_foot_points(df: pd.DataFrame, frame: int) -> np.ndarray:
    """Image-space foot points of one frame's detected players (the gate's plausibility evidence)."""
    grp = df[(df["frame"] == frame) & df["role"].isin(PLAYER_ROLES)]
    pts = grp[["image_x", "image_y"]].to_numpy(dtype=float)
    return pts[np.isfinite(pts).all(axis=1)]


def dead_frames(df: pd.DataFrame) -> list[int]:
    """Frames whose players carry no pitch coordinate at all -- the ones the dropout ate."""
    players = df[df["role"].isin(PLAYER_ROLES)]
    live = players.groupby("frame")["pitch_x"].apply(lambda s: bool(np.isfinite(s).any()))
    return [int(f) for f, ok in live.items() if not ok]


# === GPU: the candidate cache ====================================================================
def cache_sequence(seq_dir: Path, df: pd.DataFrame, calibrator, dest: Path,
                   *, live_sample: int = 30) -> dict:
    """Re-run PnLCalib on a sequence's dead frames (+ a live control sample) -> candidate npz.

    Args:
        seq_dir: GSR sequence folder (``img1/%06d.jpg``).
        df: The cached positions table for the sequence.
        calibrator: A :class:`~generator.calibrate.PnLCalibCalibrator`.
        dest: Output ``.npz``.
        live_sample: How many already-live frames to also cache, as the control that the new gate
            leaves healthy frames alone.

    Returns:
        Counts + wall time for the run log.
    """
    import cv2  # noqa: PLC0415

    dead = dead_frames(df)
    live = sorted(set(df["frame"].astype(int)) - set(dead))
    ctrl = live[:: max(1, len(live) // live_sample)][:live_sample] if live else []
    todo = sorted(set(dead) | set(ctrl))
    cols: dict[str, list] = {k: [] for k in
                             ("frame", "rank", "err_m", "n_pts", "rep_px", "mode", "ransac")}
    hs: list[np.ndarray] = []
    t0 = time.time()
    for fr in todo:
        img = cv2.imread(str(seq_dir / "img1" / f"{fr + 1:06d}.jpg"))
        if img is None:
            logger.warning("%s: frame %d image missing", seq_dir.name, fr)
            continue
        for rank, c in enumerate(calibrator.candidates(img)):
            hs.append(c.homography)
            cols["frame"].append(fr)
            cols["rank"].append(rank)
            cols["err_m"].append(c.error_m)
            cols["n_pts"].append(c.n_points)
            cols["rep_px"].append(c.rep_err_px)
            cols["mode"].append(_MODES.index(c.mode))
            cols["ransac"].append(c.use_ransac)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, homography=np.asarray(hs, dtype=float).reshape(-1, 3, 3),
                        **{k: np.asarray(v) for k, v in cols.items()},
                        dead=np.asarray(dead, dtype=int), control=np.asarray(ctrl, dtype=int))
    return {"n_dead": len(dead), "n_control": len(ctrl), "n_frames": len(todo),
            "n_candidates": len(hs), "seconds": round(time.time() - t0, 1)}


def load_cache(path: Path) -> tuple[dict[int, list[CalibCandidate]], list[int], list[int]]:
    """Read a candidate npz back into ``{frame: [CalibCandidate, ...]}`` (already in rank order)."""
    z = np.load(path)
    by_frame: dict[int, list[CalibCandidate]] = {}
    for i in np.argsort(z["frame"] * 100 + z["rank"], kind="stable"):
        by_frame.setdefault(int(z["frame"][i]), []).append(CalibCandidate(
            homography=z["homography"][i], error_m=float(z["err_m"][i]),
            n_points=int(z["n_pts"][i]), mode=_MODES[int(z["mode"][i])],
            use_ransac=float(z["ransac"][i]), rep_err_px=float(z["rep_px"][i])))
    return by_frame, [int(f) for f in z["dead"]], [int(f) for f in z["control"]]


# === CPU: diagnosis ==============================================================================
def diagnose(df: pd.DataFrame, cache: Path, *, max_error_m: float = 2.0) -> dict:
    """Frame-level audit of one sequence: what the old gate accepted, what the new one does.

    Returns per-frame rows plus the sequence summary: how many dead frames had a *plausible*
    hypothesis available all along, and whether any control (already-live) frame changes its pick.
    """
    by_frame, dead, ctrl = load_cache(cache)
    rows = []
    for fr, cands in by_frame.items():
        foot = player_foot_points(df, fr)
        old = select_calibration(cands, foot_points=None, max_error_m=max_error_m)
        new = select_calibration(cands, foot_points=foot, max_error_m=max_error_m)
        top = cands[0] if cands else None
        picked = next((i for i, c in enumerate(cands)
                       if c.error_m <= max_error_m and onpitch_plausible(c.homography, foot)), -1)
        rows.append({
            "frame": fr, "dead": fr in set(dead), "n_players": len(foot),
            "n_cands": len(cands),
            "old_err_m": float(old.error_m), "old_ok": bool(old.ok),
            "old_mode": top.mode if top else None,
            "old_onpitch": bool(top is not None and onpitch_plausible(top.homography, foot)),
            "old_frac_onpitch": _frac_onpitch(top.homography, foot) if top is not None else 0.0,
            "old_n_onpitch": _n_onpitch(top.homography, foot) if top is not None else 0,
            "old_span_m": _span_m(top.homography, foot) if top is not None else 0.0,
            "best_span_m": max((_span_m(c.homography, foot) for c in cands), default=0.0),
            "new_rank": picked, "new_ok": bool(new.ok), "new_err_m": float(new.error_m),
            "new_mode": cands[picked].mode if picked >= 0 else None,
            "new_ransac": cands[picked].use_ransac if picked >= 0 else None,
        })
    d = [r for r in rows if r["dead"]]
    c = [r for r in rows if not r["dead"]]
    return {"rows": rows, "summary": {
        "n_dead_cached": len(d),
        "dead_old_gate_passed": sum(r["old_ok"] for r in d),
        "dead_old_onpitch": sum(r["old_onpitch"] for r in d),
        # Why the frame is dead, decomposed: too few players *on* the pitch (the off-pitch
        # projection failure) vs enough players but too little spread (the span rule).
        "dead_too_few_onpitch": sum(r["old_n_onpitch"] < 8 for r in d),
        "dead_span_only": sum(r["old_n_onpitch"] >= 8 and not r["old_onpitch"] for r in d),
        "dead_median_span_m": float(np.median([r["old_span_m"] for r in d])) if d else float("nan"),
        "dead_median_frac_onpitch": float(np.median([r["old_frac_onpitch"] for r in d]))
        if d else float("nan"),
        "dead_recovered": sum(r["new_ok"] for r in d),
        "dead_recovered_rank0": sum(r["new_ok"] and r["new_rank"] == 0 for r in d),
        "dead_median_old_err_m": float(np.median([r["old_err_m"] for r in d])) if d else float("nan"),
        "dead_median_new_err_m": float(np.median([r["new_err_m"] for r in d if r["new_ok"]]))
        if any(r["new_ok"] for r in d) else float("nan"),
        "n_control": len(c),
        "control_pick_changed": sum(r["new_rank"] > 0 for r in c),
        "control_new_dead": sum(not r["new_ok"] for r in c),
    }}


def _onpitch(h: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """``(M, 2)`` pitch coordinates of the ``foot`` points that land on the pitch under ``h``."""
    from generator.calibrate import apply_homography  # noqa: PLC0415

    if not len(foot):
        return np.zeros((0, 2))
    proj = np.asarray([clamp_to_pitch(x, y) for x, y in apply_homography(h, foot)])
    return proj[np.isfinite(proj).all(axis=1)]


def _frac_onpitch(h: np.ndarray, foot: np.ndarray) -> float:
    """Share of ``foot`` points a homography puts on the pitch (after the edge tolerance)."""
    return float(len(_onpitch(h, foot)) / len(foot)) if len(foot) else 0.0


def _n_onpitch(h: np.ndarray, foot: np.ndarray) -> int:
    """How many ``foot`` points land on the pitch (the gate's ``min_onpitch`` quantity)."""
    return int(len(_onpitch(h, foot)))


def _span_m(h: np.ndarray, foot: np.ndarray) -> float:
    """Metres the on-pitch projection spans along its wider axis (the gate's ``min_span_m``)."""
    good = _onpitch(h, foot)
    return float(max(np.ptp(good[:, 0]), np.ptp(good[:, 1]))) if len(good) else 0.0


def audit_dead_frames(df: pd.DataFrame, cache: Path, seq_dir: Path) -> dict:
    """GT-audit what the *stock* homography would have produced on the frames it was denied.

    Answers the question the diagnosis turns on: were those homographies wrong (the premise), or
    right and thrown away? Ground truth is read **only to score**; nothing here feeds the repair.
    """
    from eval.gsr_score import CENTRE_SHIFT_X, CENTRE_SHIFT_Y, GSR_DIST_TOL_M  # noqa: PLC0415
    from generator.track_relink import load_gt_ids_by_frame  # noqa: PLC0415

    by_frame, dead, _ctrl = load_cache(cache)
    gt = load_gt_ids_by_frame(seq_dir)
    tp = fp = hit = tot = 0
    for fr in dead:
        cands = by_frame.get(fr)
        if not cands:
            continue
        gts = gt.get(int(fr), [])
        tot += len(gts)
        proj = _onpitch(cands[0].homography, player_foot_points(df, fr))
        if not gts or not len(proj):
            continue
        gxy = np.array([[a[0], a[1]] for a in gts])
        pxy = np.column_stack([proj[:, 0] - CENTRE_SHIFT_X, proj[:, 1] - CENTRE_SHIFT_Y])
        d = np.hypot(gxy[:, None, 0] - pxy[None, :, 0], gxy[:, None, 1] - pxy[None, :, 1])
        ok = d.min(axis=0) <= GSR_DIST_TOL_M
        tp += int(ok.sum())
        fp += int((~ok).sum())
        hit += int((d.min(axis=1) <= GSR_DIST_TOL_M).sum())
    return {"n_dead": len(dead), "rows": tp + fp, "precision": tp / max(tp + fp, 1),
            "gt_rows_on_dead_frames": tot, "gt_recall_if_kept": hit / max(tot, 1)}


# === CPU: apply ==================================================================================
def apply_sequence(df: pd.DataFrame, cache: Path, *, max_error_m: float = 2.0,
                   **plausibility) -> tuple[pd.DataFrame, int]:
    """Re-project the frames the new gate rescues; returns ``(table, rows_recovered)``.

    Only frames that currently carry no pitch position are touched, and only with a homography the
    new gate accepts -- so an existing coordinate is never moved and a rescued frame is never worse
    than the plausibility test allows.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    by_frame, dead, _ctrl = load_cache(cache)
    out = df.copy()
    idx_by_frame = out.groupby("frame").groups
    recovered = 0
    for fr in dead:
        cands = by_frame.get(fr)
        if not cands or fr not in idx_by_frame:
            continue
        res = select_calibration(cands, foot_points=player_foot_points(df, fr),
                                 max_error_m=max_error_m, **plausibility)
        if not res.ok:
            continue
        idx = idx_by_frame[fr]
        pts = out.loc[idx, ["image_x", "image_y"]].to_numpy(dtype=float)
        ok = np.isfinite(pts).all(axis=1)
        proj = apply_homography(res.homography, np.where(ok[:, None], pts, 0.0))
        vals = [clamp_to_pitch(x, y) if k else (float("nan"), float("nan"))
                for (x, y), k in zip(proj, ok)]
        out.loc[idx, ["pitch_x", "pitch_y"]] = vals
        out.loc[idx, "calib_error_m"] = res.error_m
        recovered += int(np.isfinite([v[0] for v in vals]).sum())
    return out, recovered


def apply_split(out_dir: Path, seqs: list[str], *, cache_dir: Path | None = None,
                dest_dir: Path | None = None, fill_gap: int | None = None,
                **plausibility) -> dict[str, dict]:
    """Write re-gated parquets for ``seqs``; optionally stack the frozen post-hoc fill on top."""
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415

    cache_dir = cache_dir or out_dir / CACHE_SUBDIR
    dest = dest_dir or out_dir / GATED_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in seqs:
        df = pd.read_parquet(out_dir / "positions" / f"{name}.parquet")
        cache = cache_dir / f"{name}.npz"
        gated, rec = (apply_sequence(df, cache, **plausibility) if cache.exists()
                      else (df.copy(), 0))
        after_gate = int(np.isfinite(gated["pitch_x"]).sum())
        if fill_gap is not None:
            gated = fill_calibration_gaps(gated, max_gap=fill_gap)
        gated.to_parquet(dest / f"{name}.parquet", index=False)
        stats[name] = {
            "rows": len(df),
            "pitch_rows_before": int(np.isfinite(df["pitch_x"]).sum()),
            "pitch_rows_gate": after_gate,
            "pitch_rows_final": int(np.isfinite(gated["pitch_x"]).sum()),
            "recovered_by_gate": rec,
        }
    return stats


def sweep_trust_rule(data_dir: Path, out_dir: Path, seqs: list[str],
                     grid: list[tuple[int, float]], *, fill_gap: int | None = 10) -> dict:
    """Score the base arm for each trust rule (+ both controls) -- the DEV-20 choice of thresholds.

    The base arm (positions -> submission, no jersey/connector/solver) is the same fast instrument
    `results/GSR_ASSOCIATION.md` section 4.1 chose ``max_gap`` with.
    """
    from tools.gsr_calibfill import build_arm, fill_split, score_arm  # noqa: PLC0415

    out: dict[str, dict] = {}

    def _record(tag: str, res: dict, extra: dict) -> None:
        out[tag] = {**extra, "combined": {k: v["combined"] for k, v in res.items()},
                    "per_seq": {k: v["per_seq"] for k, v in res.items()}}
        c = res["loc_assoc"]["combined"]
        g = res["gs_hota_full"]["combined"]
        print(f"{tag:<28} loc_assoc H {c['GS-HOTA']:6.2f} DetA {c['GS-DetA']:6.2f} "
              f"AssA {c['GS-AssA']:6.2f} LocA {c['GS-LocA']:6.2f} | full H {g['GS-HOTA']:6.2f}")

    _record("control_nofill", score_arm(out_dir / "eval", data_dir, seqs), {})
    st = fill_split(out_dir, seqs, max_gap=10)
    build_arm(data_dir, out_dir, out_dir / "positions_filled", out_dir / "eval_calibfill_base", seqs)
    _record("control_fill10", score_arm(out_dir / "eval_calibfill_base", data_dir, seqs),
            {"recovered": sum(v["recovered"] for v in st.values())})
    for min_onpitch, min_span in grid:
        for gap in ([None, fill_gap] if fill_gap is not None else [None]):
            st = apply_split(out_dir, seqs, fill_gap=gap, min_onpitch=min_onpitch,
                             min_span_m=min_span)
            build_arm(data_dir, out_dir, out_dir / GATED_SUBDIR, out_dir / BASE_ARM_DIR, seqs)
            tag = f"gate_p{min_onpitch}_s{min_span:g}" + (f"_fill{gap}" if gap else "")
            _record(tag, score_arm(out_dir / BASE_ARM_DIR, data_dir, seqs),
                    {"min_onpitch": min_onpitch, "min_span_m": min_span, "fill_gap": gap,
                     "recovered_by_gate": sum(v["recovered_by_gate"] for v in st.values()),
                     "pitch_rows_final": sum(v["pitch_rows_final"] for v in st.values()),
                     "pitch_rows_before": sum(v["pitch_rows_before"] for v in st.values())})
    return out


# === CPU: the GT-free solver arm on re-gated positions ===========================================
#: Arm directories for this experiment (never shared with the on-record calibfill ones).
BASE_ARM_DIR = "eval_calibgate_base"
KOSHKINA_ARM_DIR = "eval_calibgate_koshkina"
GTA_ARM_DIR = "eval_calibgate_gta"
BUNDLES_DIR = "identity_bundles_percrop_gate"


def solve_gate_arm(data_dir: Path, out_dir: Path, names: list[str], *, tag: str,
                   tau: float = 0.04) -> dict:
    """Build submission -> jersey -> GTA -> GT-free solver on ``positions_gate`` and score GS-HOTA.

    Byte-for-byte the frozen recipe of `results/GSR_CALIBFILL_TEST.md`; only the positions source
    differs, so the gate repair is the single variable.
    """
    from generator.identity_solve import SolverConfig  # noqa: PLC0415

    from eval.gsr_identity import load_bundles  # noqa: PLC0415
    from tools.gsr_calibfill import build_arm, gta_arm, jersey_arm  # noqa: PLC0415
    from tools.gsr_deleak import SOLVER_CONFIG, run_arm  # noqa: PLC0415

    pos = out_dir / GATED_SUBDIR
    build_arm(data_dir, out_dir, pos, out_dir / BASE_ARM_DIR, names)
    jersey_arm(out_dir, out_dir / BASE_ARM_DIR, out_dir / KOSHKINA_ARM_DIR, names)
    gta_arm(data_dir, out_dir, pos, out_dir / GTA_ARM_DIR, names, tau=tau,
            jersey_dir=out_dir / KOSHKINA_ARM_DIR)
    bundles = load_bundles(data_dir, out_dir, names, votes_subdir="koshkina_percrop_votes",
                           cache_subdir=BUNDLES_DIR, positions_subdir=GATED_SUBDIR)
    return run_arm(bundles, SolverConfig.load(SOLVER_CONFIG), data_dir, out_dir, names,
                   team_src="free", roster="self", tag=tag, score_hota=True,
                   positions_subdir=GATED_SUBDIR, base_arm=GTA_ARM_DIR)


#: Where the combined (GT-free + gate + fill) recipe is pre-declared.
FROZEN_PATH = Path("results/gsr_calibgate_frozen.json")


def freeze_recipe(path: Path = FROZEN_PATH) -> dict:
    """Pre-declare the combined recipe: the frozen GT-free chain + the repaired trust rule."""
    from datetime import datetime, timezone  # noqa: PLC0415

    from generator.postprocess import TRUST_MIN_PLAYERS, TRUST_MIN_SPAN_M  # noqa: PLC0415
    from tools.gsr_calibfill import COMBINED_FROZEN  # noqa: PLC0415

    base = json.loads(COMBINED_FROZEN.read_text(encoding="utf-8"))
    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent": {"file": str(COMBINED_FROZEN), "declared_at": base["declared_at"]},
        "team_map": base["team_map"], "roster": base["roster"],
        "solver_config": base["solver_config"],
        "solver_config_sha256": base["solver_config_sha256"],
        "calibration_fill": base["calibration_fill"],
        "calibration_gate": {
            "functions": ["generator.calibrate.PnLCalibCalibrator.candidates",
                          "generator.calibrate.select_calibration",
                          "generator.postprocess.onpitch_plausible"],
            "trust_min_players": TRUST_MIN_PLAYERS, "trust_min_span_m": TRUST_MIN_SPAN_M,
            "max_reproj_error_m": 2.0,
            "applied_to": "the frames the cached positions emit no pitch coordinate for, "
                          "re-projected from that frame's own recomputed homography",
            "chosen_on": "valid DEV-20 (results/GSR_CALIBGATE.md section 2)",
            "order": "gate first, then fill_calibration_gaps(max_gap=10) for what is still dead",
        },
        "note": "trust rule chosen on DEV-20; valid TEST-38 and the official test split are "
                "verification only",
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("frozen combined recipe -> %s", path)
    return record


def compare_arms(data_dir: Path, out_dir: Path, seqs: list[str], *, tag: str,
                 fill_gap: int | None = 10, **plausibility) -> dict:
    """The three GT-free solver arms on one partition: no-fill, fill-only, gate(+fill).

    Both controls are re-derived through this harness rather than quoted, so the deltas are
    like-for-like (`GSR_CALIBFILL_TEST.md` §4's discipline).
    """
    from eval.gsr_identity import paired_stats  # noqa: PLC0415
    from tools.gsr_calibfill import fill_split, solve_arm  # noqa: PLC0415

    arms = {"nofill": solve_arm(data_dir, out_dir, seqs, tag=f"{tag}_nofill_free_self",
                                filled=False)}
    fill_split(out_dir, seqs, max_gap=10)
    arms["fill10"] = solve_arm(data_dir, out_dir, seqs, tag=f"{tag}_fill_free_self", filled=True)
    st = apply_split(out_dir, seqs, fill_gap=fill_gap, **plausibility)
    arms["gate"] = solve_gate_arm(data_dir, out_dir, seqs, tag=tag)
    arms["gate"]["gate_stats"] = {
        "fill_gap": fill_gap, **plausibility,
        "recovered_by_gate": sum(v["recovered_by_gate"] for v in st.values()),
        "pitch_rows_before": sum(v["pitch_rows_before"] for v in st.values()),
        "pitch_rows_final": sum(v["pitch_rows_final"] for v in st.values()),
        "total_rows": sum(v["rows"] for v in st.values()), "per_seq": st}
    paired = {f"gate_vs_{k}": paired_stats(arms[k]["gs_hota_per_seq"],
                                           arms["gate"]["gs_hota_per_seq"], seqs)
              for k in ("nofill", "fill10")}
    for name, a in arms.items():
        c = a["gs_hota"]
        print(f"{name:<10} GS-HOTA {c['GS-HOTA']:7.4f}  DetA {c['GS-DetA']:7.4f}  "
              f"AssA {c['GS-AssA']:7.4f}  LocA {c['GS-LocA']:7.4f}  IDF1 {c['IDF1']:7.4f}")
    for k, p in paired.items():
        print(f"{k}: mean {p['mean']:+.2f} median {p['median']:+.2f} helped {p['helped']} "
              f"hurt {p['hurt']} worst {p['worst']:+.2f} best {p['best']:+.2f} "
              f"p={p['wilcoxon_p']:.3g}")
    return {"seqs": seqs, "arms": arms, "paired": paired}


def zip_selfscore(zip_path: Path, data_dir: Path, names: list[str], dest: Path) -> dict:
    """Extract the shipped zip and score *it* -- proof the archive holds what the arm scored."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    from eval.gsr_score import gs_hota  # noqa: PLC0415
    from tools.gsr_deleak import ZIP_TRACKER_DIR  # noqa: PLC0415

    root = Path(tempfile.mkdtemp(prefix="gsr_zipscore_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        arm = root / "arm"
        (arm / "predictions" / "data").mkdir(parents=True)
        for n in names:
            shutil.copyfile(root / ZIP_TRACKER_DIR / f"{n}.json",
                            arm / "predictions" / "data" / f"{n}.json")
        from eval.gsr_score import EVAL_CONFIGS  # noqa: PLC0415

        res = gs_hota(arm, data_dir, seq_info={n: 0 for n in names},
                      **EVAL_CONFIGS["gs_hota_full"])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def run_testsplit(data_dir: Path, out_dir: Path, results_dir: Path, *,
                  fill_gap: int | None = 10, tau: float = 0.04) -> dict:
    """The ONE test run: control, re-gate, reconnect, solve, score, audit, package, re-score the zip.

    The unrepaired and fill-repaired arms (on record at 31.88 / 33.37 GS-HOTA) are re-derived through
    this same harness first, so both deltas are like-for-like rather than quoted.
    """
    from eval.gsr_identity import paired_stats  # noqa: PLC0415
    from tools.gsr_calibfill import verify_gtfree  # noqa: PLC0415
    from tools.gsr_deleak import package_free, split_names  # noqa: PLC0415

    if not FROZEN_PATH.exists():
        raise SystemExit(f"{FROZEN_PATH} missing -- freeze before touching the test split")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    names = split_names(data_dir, "test")
    res = compare_arms(data_dir, out_dir, names, tag="t49", fill_gap=fill_gap)
    arm_dir = out_dir / "deleak_t49"
    gate = res["arms"]["gate"]
    gate["legitimacy"] = verify_gtfree(arm_dir, out_dir / GTA_ARM_DIR, data_dir,
                                       out_dir / GATED_SUBDIR, names)
    gate["manifest"] = package_free(arm_dir, names, frozen, stem="gtfree_calibgate")
    gate["zip_selfscore"] = zip_selfscore(
        Path(gate["manifest"]["zip"]), data_dir, names,
        Path(gate["manifest"]["zip"]).parent / "zip_selfscore_gtfree_calibgate.json")["combined"]
    res["paired"]["gate_vs_fill10_full"] = paired_stats(
        res["arms"]["fill10"]["gs_hota_per_seq"], gate["gs_hota_per_seq"], names)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "gsr_calibgate_testsplit.json"
    dest.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    logger.info("legitimacy: %s", gate["legitimacy"])
    logger.info("zip self-score: %s", gate["zip_selfscore"])
    logger.info("wrote %s", dest)
    return res


# === CLI =========================================================================================
def worst_sequences(out_dir: Path, n: int) -> list[str]:
    """The ``n`` valid sequences with the lowest share of frames carrying any pitch position."""
    health = {}
    for p in sorted((out_dir / "positions").glob("*.parquet")):
        df = pd.read_parquet(p)
        players = df[df["role"].isin(PLAYER_ROLES)]
        live = players.groupby("frame")["pitch_x"].apply(lambda s: bool(np.isfinite(s).any()))
        health[p.stem] = float(live.mean())
    return sorted(health, key=health.get)[:n]


def _demo() -> None:
    """Self-check: the plausibility term rejects an off-pitch homography and picks the runner-up."""
    good = np.array([[0.06, 0.004, -8.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])
    foot = np.column_stack([np.linspace(300, 1500, 12), 400 + 20 * (np.arange(12) % 5)])
    assert onpitch_plausible(good, foot)
    bad = np.array([[0.06, 0.004, 400.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])  # x >> 105
    assert not onpitch_plausible(bad, foot)
    cands = [CalibCandidate(bad, 0.36, 12, "full", 0.0, 1.0),
             CalibCandidate(good, 1.10, 12, "main", 5.0, 3.0)]
    old = select_calibration(cands, foot_points=None)
    new = select_calibration(cands, foot_points=foot)
    assert old.ok and np.allclose(old.homography, bad), "stock behaviour must be unchanged"
    assert new.ok and np.allclose(new.homography, good), "the gate must fall through to the runner-up"
    # No plausible candidate -> rejected, so the caller falls back to the temporal fill.
    assert not select_calibration([cands[0]], foot_points=foot).ok
    # Over the gate on error is still a reject even when plausible.
    assert not select_calibration([CalibCandidate(good, 9.9, 12, "full", 0.0, 1.0)],
                                  foot_points=foot).ok
    print("gsr_calibgate self-check OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names")
    ap.add_argument("--worst", type=int, default=0, help="use the N worst-calibrated sequences")
    ap.add_argument("--cache", action="store_true", help="GPU: cache PnLCalib candidates")
    ap.add_argument("--diag", action="store_true", help="frame-level diagnosis of the cache")
    ap.add_argument("--apply", action="store_true", help="write re-gated positions parquets")
    ap.add_argument("--solve", action="store_true", help="apply, then run the frozen GT-free solver")
    ap.add_argument("--sweep", action="store_true", help="DEV: score the base arm per trust rule")
    ap.add_argument("--compare", action="store_true",
                    help="the three GT-free solver arms: no-fill, fill, gate")
    ap.add_argument("--freeze", action="store_true", help="pre-declare the combined recipe")
    ap.add_argument("--solve-test", action="store_true",
                    help="the ONE test run: re-gate -> connector -> solve -> score -> package")
    ap.add_argument("--fill-gap", type=int, default=None, help="stack fill_calibration_gaps on top")
    ap.add_argument("--tag", default="worst8")
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.freeze:
        print(json.dumps(freeze_recipe(), indent=2))
        return
    seqs = (args.seqs.split(",") if args.seqs else
            worst_sequences(args.out_dir, args.worst) if args.worst else
            sorted(p.stem for p in (args.out_dir / "positions").glob("*.parquet")))
    logger.info("%d sequences: %s", len(seqs), ",".join(seqs))

    if args.cache:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        calibrator = PnLCalibCalibrator()
        stats = {}
        for i, name in enumerate(seqs):
            dest = args.out_dir / CACHE_SUBDIR / f"{name}.npz"
            if dest.exists():
                logger.info("[%d/%d] %s: cached", i + 1, len(seqs), name)
                continue
            df = pd.read_parquet(args.out_dir / "positions" / f"{name}.parquet")
            stats[name] = cache_sequence(args.data_dir / name, df, calibrator, dest)
            logger.info("[%d/%d] %s: %s", i + 1, len(seqs), name, stats[name])
        return

    if args.diag:
        payload, rows = {}, []
        for name in seqs:
            cache = args.out_dir / CACHE_SUBDIR / f"{name}.npz"
            if not cache.exists():
                continue
            df = pd.read_parquet(args.out_dir / "positions" / f"{name}.parquet")
            d = diagnose(df, cache)
            payload[name] = {**d["summary"],
                             "stock_audit": audit_dead_frames(df, cache, args.data_dir / name)}
            for r in d["rows"]:
                rows.append({"seq": name, **r})
            s, a = d["summary"], payload[name]["stock_audit"]
            print(f"{name}: dead {s['n_dead_cached']:4d} | gate passed {s['dead_old_gate_passed']:4d}"
                  f" | too few on-pitch {s['dead_too_few_onpitch']:4d} | span-only "
                  f"{s['dead_span_only']:4d} (median {s['dead_median_span_m']:4.1f} m) | stock-H "
                  f"precision {a['precision']:.3f} recall {a['gt_recall_if_kept']:.3f} | control "
                  f"changed {s['control_pick_changed']}/{s['n_control']}")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dest = args.results_dir / f"gsr_calibgate_diag_{args.tag}.json"
        dest.write_text(json.dumps({"summary": payload}, indent=1), encoding="utf-8")
        pd.DataFrame(rows).to_parquet(args.results_dir / f"gsr_calibgate_frames_{args.tag}.parquet",
                                      index=False)
        print(f"wrote {dest}")
        return

    if args.sweep:
        grid = [(8, 25.0), (3, 5.0), (1, 0.0)]
        res = sweep_trust_rule(args.data_dir, args.out_dir, seqs, grid, fill_gap=args.fill_gap)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dest = args.results_dir / f"gsr_calibgate_sweep_{args.tag}.json"
        dest.write_text(json.dumps({"seqs": seqs, "grid": grid, "arms": res}, indent=1),
                        encoding="utf-8")
        print(f"wrote {dest}")
        return

    if args.solve_test:
        run_testsplit(args.data_dir, args.out_dir, args.results_dir, fill_gap=args.fill_gap,
                      tau=args.tau)
        return

    if args.compare:
        res = compare_arms(args.data_dir, args.out_dir, seqs, tag=args.tag,
                           fill_gap=args.fill_gap)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dest = args.results_dir / f"gsr_calibgate_compare_{args.tag}.json"
        dest.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"wrote {dest}")
        return

    if args.apply or args.solve:
        st = apply_split(args.out_dir, seqs, fill_gap=args.fill_gap)
        rec = sum(v["recovered_by_gate"] for v in st.values())
        tot = sum(v["rows"] for v in st.values())
        print(f"gate recovered {rec} of {tot} rows ({100 * rec / max(tot, 1):.1f}%)")
        if not args.solve:
            return
        arm = solve_gate_arm(args.data_dir, args.out_dir, seqs, tag=f"{args.tag}_gate_free_self")
        arm["gate_fill"] = {"fill_gap": args.fill_gap, "recovered_by_gate": rec,
                            "total_rows": tot, "per_seq": st}
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dest = args.results_dir / f"gsr_calibgate_{args.tag}.json"
        dest.write_text(json.dumps(arm, indent=2, default=str), encoding="utf-8")
        c = arm["gs_hota"]
        print(f"{args.tag}: GS-HOTA {c['GS-HOTA']:.4f} DetA {c['GS-DetA']:.4f} "
              f"AssA {c['GS-AssA']:.4f} LocA {c['GS-LocA']:.4f} IDF1 {c['IDF1']:.4f}")
        print(f"wrote {dest}")
        return

    ap.error("choose --cache / --diag / --apply / --solve / --demo")


if __name__ == "__main__":
    main()
