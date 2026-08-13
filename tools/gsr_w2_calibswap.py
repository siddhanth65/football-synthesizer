"""Campaign v8 W2: price an EXTERNAL per-frame calibration inside our v6 chain.

The leader's own ablation puts **+10.28 GS-HOTA** in its homography module, measured off a weak
keypoint-only baseline (`docs/WINNER_REPO_RECON.md` §7). Ours is not weak -- `results/GSR_CALIBGATE.md`
already repaired the calibration dropout at its source. This module answers the transfer question by
measurement: given a directory of per-frame ``3x3`` homographies from *any* external calibrator, it
rebuilds our positions tables from them, leaves every other stage of the chain alone, and scores
GS-HOTA paired per sequence against our own re-derived control.

The external homographies are assumed to map **image pixels -> a pitch-metre template** whose origin
offset is given by ``--origin`` (default ``10,5``: the SoccerNet radar template's
``x in [10, 115], y in [5, 73]`` frame). Nothing here knows or cares which calibrator produced them.

Three swap arms, all pre-declared in `results/GSR_V8_W2.md` §1.4:

``t1``
    external homography wherever it is usable, ``NaN`` elsewhere -- faithful, no fill.
``t2``
    ``t1`` then :func:`generator.postprocess.fill_calibration_gaps` at the frozen ``max_gap=10``.
``t3``
    our on-record coordinate wherever finite, the external one on the rows we left ``NaN`` -- the
    "does it reach frames our gate cannot" arm.

CLI::

    python -m tools.gsr_w2_calibswap --positions --homog <dir>   # build the three arms' parquets
    python -m tools.gsr_w2_calibswap --deltas    --homog <dir>   # position deltas + GT-anchored
    python -m tools.gsr_w2_calibswap --arms                      # control + t1..t3, scored, paired
    python -m tools.gsr_w2_calibswap --demo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_w2_calibswap")

#: The 10 DEV-20 sequences of the pre-registered probe (`results/GSR_V8_W2.md` §1.1).
PROBE = ["SNGS-024", "SNGS-027", "SNGS-039", "SNGS-042", "SNGS-045",
         "SNGS-048", "SNGS-051", "SNGS-054", "SNGS-057", "SNGS-078"]
#: Artifact suffix of the v6 (S4b-detector) lineage every arm here rides.
VARIANT = "_v6det"
#: Our shipped positions for that lineage (calibgate trust rule + fill).
CONTROL_SUBDIR = "positions_gate" + VARIANT
#: Pre-gate positions of the same extraction (the un-repaired coordinates).
RAW_SUBDIR = "positions" + VARIANT
#: One positions subdirectory per swap arm.
ARM_SUBDIR = {"t1": "positions_w2t1" + VARIANT, "t2": "positions_w2t2" + VARIANT,
              "t3": "positions_w2t3" + VARIANT}
#: Frozen fill gap (results/gsr_calibgate_frozen.json).
FILL_GAP = 10
#: Image-space radius (px) inside which our detection is called the same person as a GT annotation.
GT_MATCH_PX = 60.0


# === the external calibration ====================================================================
def load_homographies(homog_dir: Path) -> dict[int, np.ndarray]:
    """Read ``<frame>.npy`` 3x3 matrices into ``{frame index: H}`` (``000001.npy`` -> frame 0).

    Args:
        homog_dir: Directory of per-frame ``.npy`` matrices.

    Returns:
        Frame index -> homography. Unusable matrices (exact identity = the external calibrator's
        hard-failure sentinel, singular, or non-finite) are dropped.
    """
    out: dict[int, np.ndarray] = {}
    for p in sorted(homog_dir.glob("*.npy")):
        h = np.load(p).astype(float)
        if h.shape != (3, 3) or not np.isfinite(h).all():
            continue
        if np.allclose(h, np.eye(3)) or abs(float(np.linalg.det(h))) < 1e-12:
            continue
        out[int(p.stem) - 1] = h
    return out


def project(h: np.ndarray, pts: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    """Push image points through ``h`` into our corner-origin pitch frame (metres).

    Args:
        h: ``(3, 3)`` image -> template homography.
        pts: ``(N, 2)`` image points.
        origin: Template coordinate of the pitch's ``(0, 0)`` corner.

    Returns:
        ``(N, 2)`` pitch coordinates, un-clamped.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    return apply_homography(h, pts) - np.asarray(origin, dtype=float)


def swap_positions(df: pd.DataFrame, homs: dict[int, np.ndarray], *,
                   origin: tuple[float, float], mode: str) -> tuple[pd.DataFrame, dict]:
    """Rebuild a positions table's pitch coordinates from an external calibration.

    Args:
        df: One sequence's on-record positions table (the control lineage).
        homs: ``{frame: H}`` from :func:`load_homographies`.
        origin: See :func:`project`.
        mode: ``'t1'`` (external only) or ``'t3'`` (ours, external only where ours is ``NaN``).

    Returns:
        ``(table, stats)``. ``table`` is a copy with ``pitch_x``/``pitch_y`` replaced;
        ``calib_error_m`` is blanked on every row the external calibration wrote, because it is our
        estimator's number and no longer describes the coordinate.
    """
    from generator.postprocess import clamp_to_pitch  # noqa: PLC0415

    if mode not in {"t1", "t3"}:
        raise ValueError(f"mode must be 't1' or 't3', got {mode!r}")
    out = df.copy()
    had = np.isfinite(df["pitch_x"].to_numpy(dtype=float))
    px = np.full(len(df), np.nan)
    py = np.full(len(df), np.nan)
    frames = df["frame"].to_numpy(dtype=int)
    img = df[["image_x", "image_y"]].to_numpy(dtype=float)
    n_proj = 0
    for fr, h in homs.items():
        sel = frames == fr
        if not sel.any():
            continue
        ok = sel & np.isfinite(img).all(axis=1)
        if not ok.any():
            continue
        proj = project(h, img[ok], origin)
        vals = np.array([clamp_to_pitch(x, y) for x, y in proj], dtype=float)
        px[ok], py[ok] = vals[:, 0], vals[:, 1]
        n_proj += int(ok.sum())
    ext = np.isfinite(px)
    if mode == "t3":
        keep = had
        px = np.where(keep, df["pitch_x"].to_numpy(dtype=float), px)
        py = np.where(keep, df["pitch_y"].to_numpy(dtype=float), py)
        wrote = ext & ~had
    else:
        wrote = ext
    out["pitch_x"], out["pitch_y"] = px, py
    out.loc[wrote, "calib_error_m"] = np.nan
    return out, {
        "rows": int(len(df)),
        "frames_with_external_h": len(homs),
        "pitch_rows_control": int(had.sum()),
        "rows_projected_by_external": n_proj,
        "rows_external_rejected_offpitch": n_proj - int(ext.sum()),
        "pitch_rows_external_usable": int(ext.sum()),
        "pitch_rows_final": int(np.isfinite(px).sum()),
        "rows_written_by_external": int(wrote.sum()),
        "rows_lost_vs_control": int((had & ~np.isfinite(px)).sum()),
    }


def build_arms(out_dir: Path, homog_root: Path, seqs: list[str], *,
               origin: tuple[float, float]) -> dict:
    """Write the ``t1``/``t2``/``t3`` positions parquets for ``seqs``. Returns per-arm stats."""
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415

    stats: dict[str, dict] = {"t1": {}, "t2": {}, "t3": {}}
    for sub in ARM_SUBDIR.values():
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name in seqs:
        df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        homs = load_homographies(homog_root / name)
        t1, st1 = swap_positions(df, homs, origin=origin, mode="t1")
        t3, st3 = swap_positions(df, homs, origin=origin, mode="t3")
        t2 = fill_calibration_gaps(t1, max_gap=FILL_GAP)
        st2 = dict(st1, pitch_rows_final=int(np.isfinite(t2["pitch_x"]).sum()))
        for arm, tab, st in (("t1", t1, st1), ("t2", t2, st2), ("t3", t3, st3)):
            tab.to_parquet(out_dir / ARM_SUBDIR[arm] / f"{name}.parquet", index=False)
            stats[arm][name] = st
        logger.info("%s: control %d -> t1 %d / t2 %d / t3 %d pitch rows (%d frames with an "
                    "external H)", name, st1["pitch_rows_control"], st1["pitch_rows_final"],
                    st2["pitch_rows_final"], st3["pitch_rows_final"], len(homs))
    return stats


# === ground truth ================================================================================
def gt_people(seq_dir: Path) -> dict[int, np.ndarray]:
    """Per-frame GT player/GK rows as ``(N, 4)`` = image x, image y (foot), pitch x, pitch y.

    Pitch coordinates are converted from GSR's centred frame to our corner-origin one.
    """
    from core.pitch import PITCH_LEN, PITCH_WID  # noqa: PLC0415

    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    idx = {im["image_id"]: int(Path(im["file_name"]).stem) - 1 for im in gt["images"]}
    rows: dict[int, list[list[float]]] = {}
    for ann in gt["annotations"]:
        attrs = ann.get("attributes") or {}
        if attrs.get("role") not in {"player", "goalkeeper"}:
            continue
        bp, bi = ann.get("bbox_pitch"), ann.get("bbox_image")
        if not bp or not bi or idx.get(ann["image_id"]) is None:
            continue
        rows.setdefault(idx[ann["image_id"]], []).append([
            float(bi["x_center"]), float(bi["y"]) + float(bi["h"]),
            float(bp["x_bottom_middle"]) + PITCH_LEN / 2.0,
            float(bp["y_bottom_middle"]) + PITCH_WID / 2.0])
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def gt_matched_error(df: pd.DataFrame, gt: dict[int, np.ndarray],
                     cols: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Per-row pitch error against the image-space-nearest GT person, for several coordinate sets.

    Args:
        df: A positions table (supplies ``frame``, ``role``, ``image_x``, ``image_y``).
        gt: :func:`gt_people` output for the same sequence.
        cols: ``{label: (pitch_x array, pitch_y array)}`` -- each aligned to ``df``'s rows.

    Returns:
        ``{label: [errors]}`` plus ``matched`` (rows with a GT partner) under key ``_matched``.
        Errors are only recorded where that label's coordinate is finite, so coverage differences
        are visible; ``_both`` holds the row indices where every label is finite (the paired set).
    """
    from generator.postprocess import PLAYER_ROLES  # noqa: PLC0415

    people = df["role"].isin(PLAYER_ROLES).to_numpy()
    frames = df["frame"].to_numpy(dtype=int)
    img = df[["image_x", "image_y"]].to_numpy(dtype=float)
    match = np.full(len(df), -1)
    gtxy = np.full((len(df), 2), np.nan)
    for fr in np.unique(frames[people]):
        g = gt.get(int(fr))
        if g is None or not len(g):
            continue
        sel = np.flatnonzero(people & (frames == fr) & np.isfinite(img).all(axis=1))
        if not len(sel):
            continue
        d = np.linalg.norm(img[sel][:, None, :] - g[None, :, :2], axis=2)
        best = d.argmin(axis=1)
        ok = d[np.arange(len(sel)), best] <= GT_MATCH_PX
        match[sel[ok]] = best[ok]
        gtxy[sel[ok]] = g[best[ok], 2:]
    matched = match >= 0
    finite = {k: matched & np.isfinite(x) & np.isfinite(y) for k, (x, y) in cols.items()}
    both = np.logical_and.reduce(list(finite.values())) if finite else np.zeros(len(df), bool)
    out: dict = {"_matched": int(matched.sum()), "_paired": int(both.sum()),
                 "_people": int(people.sum())}
    for k, (x, y) in cols.items():
        err_all = np.hypot(x[finite[k]] - gtxy[finite[k], 0], y[finite[k]] - gtxy[finite[k], 1])
        err_p = np.hypot(x[both] - gtxy[both, 0], y[both] - gtxy[both, 1])
        out[k] = {"n": int(finite[k].sum()), "median": _q(err_all, 50), "p90": _q(err_all, 90),
                  "within_5m": float(np.mean(err_all <= 5.0)) if len(err_all) else float("nan"),
                  "paired_median": _q(err_p, 50), "paired_p90": _q(err_p, 90),
                  "paired_within_5m": float(np.mean(err_p <= 5.0)) if len(err_p) else float("nan")}
    return out


def _q(a: np.ndarray, q: float) -> float:
    """Percentile of a possibly-empty array (NaN when empty)."""
    return float(np.percentile(a, q)) if len(a) else float("nan")


def delta_report(data_dir: Path, out_dir: Path, homog_root: Path, seqs: list[str], *,
                 origin: tuple[float, float]) -> dict:
    """Position deltas + GT-anchored paired accuracy + the external calibrator's branch split."""
    rep: dict[str, dict] = {}
    for name in seqs:
        df = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        raw = pd.read_parquet(out_dir / RAW_SUBDIR / f"{name}.parquet")
        homs = load_homographies(homog_root / name)
        t1, st = swap_positions(df, homs, origin=origin, mode="t1")
        ours = df[["pitch_x", "pitch_y"]].to_numpy(dtype=float)
        theirs = t1[["pitch_x", "pitch_y"]].to_numpy(dtype=float)
        both = np.isfinite(ours).all(axis=1) & np.isfinite(theirs).all(axis=1)
        d = np.linalg.norm(ours[both] - theirs[both], axis=1)
        # frames our control only has because fill_calibration_gaps bridged them
        pre = np.isfinite(raw["pitch_x"].to_numpy(dtype=float))
        gate_live = set(raw.loc[pre, "frame"].astype(int))
        bridged = np.array([f not in gate_live for f in df["frame"].astype(int)])
        gt = gt_people(data_dir / name)
        acc = gt_matched_error(df, gt, {"ours": (ours[:, 0], ours[:, 1]),
                                        "theirs": (theirs[:, 0], theirs[:, 1])})
        acc_bridged = gt_matched_error(df[bridged].reset_index(drop=True), gt, {
            "ours": (ours[bridged, 0], ours[bridged, 1]),
            "theirs": (theirs[bridged, 0], theirs[bridged, 1])})
        branches = json.loads((homog_root / name / "_branches.json").read_text(encoding="utf-8"))
        per = branches["per_frame"]
        rep[name] = {
            "rows": st,
            "delta": {"n": int(both.sum()), "median_m": _q(d, 50), "p90_m": _q(d, 90),
                      "p99_m": _q(d, 99), "over_5m": float(np.mean(d > 5.0)) if len(d) else 0.0,
                      "over_1m": float(np.mean(d > 1.0)) if len(d) else 0.0},
            "gt_error": acc,
            "gt_error_on_bridged_frames": acc_bridged,
            "their_branches": {b: sum(1 for v in per.values() if v == b)
                               for b in ("line", "point", "identity")},
            "their_seconds": branches["seconds"], "their_frames": branches["n_frames"],
        }
        logger.info("%s: delta median %.2f m p90 %.2f | GT err ours %.2f theirs %.2f (paired n=%d)",
                    name, rep[name]["delta"]["median_m"], rep[name]["delta"]["p90_m"],
                    acc["ours"]["paired_median"], acc["theirs"]["paired_median"], acc["_paired"])
    return rep


# === the decisive arms ===========================================================================
def run_arms(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str],
             arms: tuple[str, ...]) -> dict:
    """Score the control and each swap arm through the frozen v6 chain, paired per sequence."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415
    from tools.gsr_v6det import TAU  # noqa: PLC0415

    from eval.gsr_identity import paired_stats  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache" + VARIANT
    out: dict[str, dict] = {}
    for arm in arms:
        sub = CONTROL_SUBDIR if arm == "control" else ARM_SUBDIR[arm]
        res = eiou.run_point(data_dir, out_dir, seqs,
                             eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                             embedder="clip" + VARIANT, tau=TAU, tag=f"w2_{arm}",
                             percrop_variant="_v6" + VARIANT, positions_subdir=sub)
        out[arm] = {"positions_subdir": sub, "gs_hota": res["gs_hota"],
                    "gs_hota_per_seq": res["gs_hota_per_seq"], "pooled": res.get("pooled"),
                    "eiou_tracks": (res["eiou"]["n_tracks_before"], res["eiou"]["n_tracks_after"])}
        h = res["gs_hota"]
        print(f"{arm:<8} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
              f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f} IDF1 {h['IDF1']:7.4f}", flush=True)
    base = out["control"]["gs_hota_per_seq"]
    for arm in arms:
        if arm == "control":
            continue
        out[arm]["paired_vs_control"] = paired_stats(base, out[arm]["gs_hota_per_seq"], seqs)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "gsr_v8_w2_arms.json"
    dest.write_text(json.dumps({"probe": seqs, "arms": out}, indent=1, default=str),
                    encoding="utf-8")
    print(f"wrote {dest}")
    return out


def control_check(out: dict, seqs: list[str],
                  ref: Path = Path("results/gsr_benchmark/gsr_v7_control_dev.json")) -> dict:
    """Verify the re-derived control against the on-record v7 DEV control, per sequence."""
    on_record = json.loads(ref.read_text(encoding="utf-8"))["arm"]["gs_hota_per_seq"]
    got = out["control"]["gs_hota_per_seq"]
    d = {s: round(float(got[s]) - float(on_record[s]), 6) for s in seqs}
    return {"per_seq_delta": d, "max_abs": max(abs(v) for v in d.values())}


def _demo() -> None:
    """Self-check: the swap arithmetic, the origin shift, and the GT matcher."""
    h = np.array([[0.05, 0.0, 10.0], [0.0, 0.05, 5.0], [0.0, 0.0, 1.0]])  # 20 px = 1 m + origin
    df = pd.DataFrame({"frame": [0, 0, 1], "track_id": [1, 2, 1],
                       "role": ["player", "player", "player"], "team": [0, 1, 0],
                       "pitch_x": [1.0, np.nan, 3.0], "pitch_y": [2.0, np.nan, 4.0],
                       "image_x": [200.0, 400.0, 600.0], "image_y": [100.0, 200.0, 300.0],
                       "conf": [1.0] * 3, "calib_error_m": [0.5] * 3,
                       "is_actor": [False] * 3, "is_keeper": [False] * 3})
    t1, st = swap_positions(df, {0: h}, origin=(10.0, 5.0), mode="t1")
    assert np.allclose(t1["pitch_x"].to_numpy()[:2], [10.0, 20.0]), t1["pitch_x"].tolist()
    assert np.isnan(t1["pitch_x"].to_numpy()[2]), "frame 1 has no external H -> NaN"
    assert st["rows_lost_vs_control"] == 1, st
    t3, st3 = swap_positions(df, {0: h}, origin=(10.0, 5.0), mode="t3")
    assert t3["pitch_x"].to_numpy()[0] == 1.0, "t3 keeps ours where ours exists"
    assert np.isclose(t3["pitch_x"].to_numpy()[1], 20.0), "t3 fills ours-NaN from theirs"
    assert t3["pitch_x"].to_numpy()[2] == 3.0
    assert st3["rows_written_by_external"] == 1, st3
    assert not load_homographies_from({0: np.eye(3)}), "identity is the failure sentinel"
    gt = {0: np.array([[205.0, 105.0, 9.0, 21.0]])}
    acc = gt_matched_error(df, gt, {"a": (df["pitch_x"].to_numpy(), df["pitch_y"].to_numpy())})
    assert acc["_matched"] == 1 and acc["a"]["n"] == 1, acc
    assert np.isclose(acc["a"]["median"], np.hypot(1.0 - 9.0, 2.0 - 21.0)), acc
    print("gsr_w2_calibswap demo OK")


def load_homographies_from(d: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """In-memory twin of :func:`load_homographies`' usability filter (testable without files)."""
    return {k: h for k, h in d.items()
            if np.isfinite(h).all() and not np.allclose(h, np.eye(3))
            and abs(float(np.linalg.det(h))) >= 1e-12}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--homog", type=Path, default=None,
                    help="root of <SEQ>/<frame>.npy external homographies (outside the repo tree)")
    ap.add_argument("--origin", default="10,5", help="template coord of the pitch (0,0) corner")
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--arms-list", default="control,t1,t2,t3")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--deltas", action="store_true")
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    seqs = args.seqs.split(",") if args.seqs else PROBE
    origin = tuple(float(v) for v in args.origin.split(","))

    if args.demo:
        _demo()
        return
    if args.positions or args.deltas:
        if args.homog is None:
            raise SystemExit("--homog is required for --positions/--deltas")
    if args.positions:
        st = build_arms(args.out_dir, args.homog, seqs, origin=origin)
        dest = args.results_dir / "gsr_v8_w2_positions.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print(f"wrote {dest}")
    if args.deltas:
        rep = delta_report(args.data_dir, args.out_dir, args.homog, seqs, origin=origin)
        dest = args.results_dir / "gsr_v8_w2_deltas.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {dest}")
    if args.arms:
        out = run_arms(args.data_dir, args.out_dir, args.results_dir, seqs,
                       tuple(args.arms_list.split(",")))
        if "control" in out:
            print("control vs on-record:", control_check(out, seqs))


if __name__ == "__main__":
    main()
