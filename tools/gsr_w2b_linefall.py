"""Campaign v8 W2b: recover the dead frames with line evidence, and price it paired on GS-HOTA.

`results/GSR_V8_W2.md` left one constructive residue: every transferable point of the leader's
calibration came from **coverage** -- a real homography on the rows our chain leaves empty. This
module builds the three registered arms of `results/GSR_V8_W2B.md` §1.3 on top of
:mod:`generator.line_calib`, all dead-frame-only, all behind the PnLCalib gate and ahead of
``fill_calibration_gaps``:

``a``
    accept a dead frame's existing keypoint hypothesis when the frame has too few players for
    ``onpitch_plausible`` to be evaluable -- the free null hypothesis, no lines anywhere;
``b``
    ``a`` + :func:`generator.line_calib.line_support` as the acceptance evidence, and the missing
    hypotheses computed for the frames the on-record cache never covered;
``c``
    ``b`` + the donor-seeded **line solve** on whatever ``b`` still cannot accept.

CLI::

    python -m tools.gsr_w2b_linefall --census                 # the dead-frame population
    python -m tools.gsr_w2b_linefall --cache                  # GPU: hypotheses for uncached deads
    python -m tools.gsr_w2b_linefall --support                # choose min_support on live frames
    python -m tools.gsr_w2b_linefall --positions              # build the arms' parquets
    python -m tools.gsr_w2b_linefall --gt                     # GT-anchored accuracy of what we wrote
    python -m tools.gsr_w2b_linefall --arms                   # score them, paired vs the control
    python -m tools.gsr_w2b_linefall --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generator.postprocess import PLAYER_ROLES

logger = logging.getLogger("gsr_w2b_linefall")

#: The W2 probe, unchanged (`results/GSR_V8_W2.md` §1.1).
PROBE = ["SNGS-024", "SNGS-027", "SNGS-039", "SNGS-042", "SNGS-045",
         "SNGS-048", "SNGS-051", "SNGS-054", "SNGS-057", "SNGS-078"]
VARIANT = "_v6det"
RAW_SUBDIR = "positions" + VARIANT
CONTROL_SUBDIR = "positions_gate" + VARIANT
CACHE_SUBDIR = "calib_candidates"
#: Hypotheses for the dead frames the on-record cache (built for the pre-v6det lineage) never saw.
EXTRA_CACHE_SUBDIR = "calib_candidates_w2b"
ARM_SUBDIR = {a: f"positions_w2b{a}" + VARIANT for a in ("a", "b", "c", "d")}
FILL_GAP = 10


# === populations =================================================================================
def dead_frames(df: pd.DataFrame) -> list[int]:
    """Frames whose player rows carry no pitch coordinate at all."""
    people = df["role"].isin(PLAYER_ROLES)
    live = set(df.loc[people & np.isfinite(df["pitch_x"]) & np.isfinite(df["pitch_y"]),
                      "frame"].astype(int))
    return sorted(set(df["frame"].astype(int)) - live)


def load_candidates(out_dir: Path, name: str) -> dict[int, list]:
    """Merge the on-record candidate cache with this session's top-up (top-up wins on a clash)."""
    from tools.gsr_calibgate import load_cache  # noqa: PLC0415

    merged: dict[int, list] = {}
    for sub in (CACHE_SUBDIR, EXTRA_CACHE_SUBDIR):
        path = out_dir / sub / f"{name}.npz"
        if path.exists():
            by_frame, _dead, _ctrl = load_cache(path)
            merged.update({f: c for f, c in by_frame.items() if c})
    return merged


def gated_table(out_dir: Path, name: str) -> pd.DataFrame:
    """The shipped chain **up to but excluding** the fill: raw extraction -> calibgate re-gate."""
    from tools.gsr_calibgate import apply_sequence  # noqa: PLC0415

    raw = pd.read_parquet(out_dir / RAW_SUBDIR / f"{name}.parquet")
    cache = out_dir / CACHE_SUBDIR / f"{name}.npz"
    gated, _rec = apply_sequence(raw, cache) if cache.exists() else (raw.copy(), 0)
    return gated


def census(out_dir: Path, seqs: list[str]) -> dict:
    """Per-sequence dead-frame anatomy: how many, how many rows, and what evidence exists."""
    rep: dict[str, dict] = {}
    for name in seqs:
        gated = gated_table(out_dir, name)
        ctrl = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        cands = load_candidates(out_dir, name)
        dead = dead_frames(ctrl)
        rows = int(((~np.isfinite(ctrl["pitch_x"])) & np.isfinite(ctrl["image_x"])
                    & np.isin(ctrl["frame"].astype(int), dead)).sum())
        players = {f: int((gated["frame"] == f).sum()) for f in dead}
        rep[name] = {
            "frames": int(ctrl["frame"].nunique()),
            "dead_after_gate": len(dead_frames(gated)),
            "dead_after_fill": len(dead),
            "rows_on_dead_frames": rows,
            "rows_empty_on_live_frames": int(((~np.isfinite(ctrl["pitch_x"]))
                                             & np.isfinite(ctrl["image_x"])).sum()) - rows,
            "dead_with_candidates": sum(1 for f in dead if cands.get(f)),
            "dead_rows_total": sum(players.values()),
        }
        logger.info("%s: %d dead frames (%d rows), %d with hypotheses", name, len(dead), rows,
                    rep[name]["dead_with_candidates"])
    return rep


# === GPU: top up the candidate cache =============================================================
def cache_missing(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """Run PnLCalib on the dead frames that have no cached hypotheses, into a separate npz.

    The on-record cache was built for the pre-v6det extraction's dead set; the v6det lineage's dead
    frames are not the same set. This tops it up without touching the on-record artifact.
    """
    import cv2  # noqa: PLC0415

    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415
    from tools.gsr_calibgate import _MODES  # noqa: PLC0415

    calib = PnLCalibCalibrator()
    dest_dir = out_dir / EXTRA_CACHE_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for name in seqs:
        ctrl = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        have = load_candidates(out_dir, name)
        todo = [f for f in dead_frames(ctrl) if not have.get(f)]
        if not todo:
            stats[name] = {"n_frames": 0}
            continue
        cols: dict[str, list] = {k: [] for k in
                                 ("frame", "rank", "err_m", "n_pts", "rep_px", "mode", "ransac")}
        hs: list[np.ndarray] = []
        t0 = time.time()
        for fr in todo:
            img = cv2.imread(str(data_dir / name / "img1" / f"{fr + 1:06d}.jpg"))
            if img is None:
                logger.warning("%s frame %d: image missing", name, fr)
                continue
            for rank, c in enumerate(calib.candidates(img)):
                hs.append(c.homography)
                for k, v in (("frame", fr), ("rank", rank), ("err_m", c.error_m),
                             ("n_pts", c.n_points), ("rep_px", c.rep_err_px),
                             ("mode", _MODES.index(c.mode)), ("ransac", c.use_ransac)):
                    cols[k].append(v)
        np.savez_compressed(dest_dir / f"{name}.npz",
                            homography=np.asarray(hs, dtype=float).reshape(-1, 3, 3),
                            **{k: np.asarray(v) for k, v in cols.items()},
                            dead=np.asarray(todo, dtype=int), control=np.zeros(0, dtype=int))
        stats[name] = {"n_frames": len(todo), "n_candidates": len(hs),
                       "frames_with_none": len(todo) - len(set(cols["frame"])),
                       "seconds": round(time.time() - t0, 1)}
        logger.info("%s: cached %d dead frames -> %d hypotheses in %.0f s", name, len(todo),
                    len(hs), stats[name]["seconds"])
    return stats


# === choosing min_support, GT-free and score-free ================================================
def support_on_live(data_dir: Path, out_dir: Path, seqs: list[str], *, per_seq: int = 40) -> dict:
    """Line support of homographies the shipped chain **already trusts** -- the threshold's basis.

    Each sampled live frame's own homography is re-derived from its projected rows (the same DLT
    ``fill_calibration_gaps`` uses to pick donors), so this measures the support distribution of
    accepted calibrations without reading ground truth or any score.
    """
    import cv2  # noqa: PLC0415

    from generator.line_calib import line_support, pitch_model_lines  # noqa: PLC0415
    from generator.postprocess import _frame_homographies  # noqa: PLC0415

    segs = pitch_model_lines()
    rep: dict[str, dict] = {}
    pooled: list[float] = []
    for name in seqs:
        gated = gated_table(out_dir, name)
        hs = _frame_homographies(gated, 8)
        live = sorted(hs)
        pick = live[:: max(1, len(live) // per_seq)][:per_seq]
        vals = []
        for fr in pick:
            img = cv2.imread(str(data_dir / name / "img1" / f"{fr + 1:06d}.jpg"))
            if img is None:
                continue
            dist, _n = _ridge(img)
            vals.append(line_support(hs[fr], dist, segs=segs)[0])
        rep[name] = {"n": len(vals), "median": _q(vals, 50), "p10": _q(vals, 10),
                     "p05": _q(vals, 5), "min": float(min(vals)) if vals else float("nan")}
        pooled += vals
        logger.info("%s: live-frame support median %.3f p05 %.3f", name, rep[name]["median"],
                    rep[name]["p05"])
    rep["_pooled"] = {"n": len(pooled), "median": _q(pooled, 50), "p10": _q(pooled, 10),
                      "p05": _q(pooled, 5), "p01": _q(pooled, 1)}
    return rep


def _ridge(img):
    """``(dist, nearest)`` ridge field of one BGR frame."""
    from generator.line_calib import line_mask, ridge_field  # noqa: PLC0415

    return ridge_field(line_mask(img))


def _q(a, q: float) -> float:
    """Percentile of a possibly-empty sequence."""
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


# === the arms ====================================================================================
class FrameReader:
    """Lazy ``frame -> BGR image`` reader for one sequence (only dead frames are ever read)."""

    def __init__(self, seq_dir: Path):
        self.seq_dir = seq_dir

    def __call__(self, frame: int):
        """Read ``img1/%06d.jpg`` for a 0-based frame index; ``None`` when it is missing."""
        import cv2  # noqa: PLC0415

        return cv2.imread(str(self.seq_dir / "img1" / f"{frame + 1:06d}.jpg"))


def build_arms(data_dir: Path, out_dir: Path, seqs: list[str], arms: tuple[str, ...],
               *, min_support: float) -> dict:
    """Write each arm's positions parquets (stage -> fill), plus a no-stage control reproduction."""
    from generator.line_calib import recover_dead_frames  # noqa: PLC0415
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415

    # ``c`` is the arm the registered read was scored on; ``d`` is the same arm with the seed-chain
    # bug fixed (a refused-but-self-consistent solve still initialises the next frame). ``d`` is a
    # post-registration measurement -- see `results/GSR_V8_W2B.md` §3.4.
    cfg = {"a": {"min_support": 0.0, "refit": False, "extra": False, "chain_seed": False},
           "b": {"min_support": min_support, "refit": False, "extra": True, "chain_seed": False},
           "c": {"min_support": min_support, "refit": True, "extra": True, "chain_seed": False},
           "d": {"min_support": min_support, "refit": True, "extra": True, "chain_seed": True}}
    stats: dict[str, dict] = {a: {} for a in arms}
    for a in arms:
        (out_dir / ARM_SUBDIR[a]).mkdir(parents=True, exist_ok=True)
    for name in seqs:
        gated = gated_table(out_dir, name)
        onrecord = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        repro = fill_calibration_gaps(gated, max_gap=FILL_GAP)
        drift = _coord_drift(repro, onrecord)
        reader = FrameReader(data_dir / name)
        on_record_cands = {}
        for a in arms:
            c = cfg[a]
            cands = (load_candidates(out_dir, name) if c["extra"]
                     else (on_record_cands or _on_record_only(out_dir, name)))
            if not c["extra"]:
                on_record_cands = cands
            log: list[dict] = []
            staged, rec = recover_dead_frames(
                gated, reader, candidates=cands, min_support=c["min_support"],
                refit=c["refit"], chain_seed=c["chain_seed"], log=log)
            filled = fill_calibration_gaps(staged, max_gap=FILL_GAP)
            filled.to_parquet(out_dir / ARM_SUBDIR[a] / f"{name}.parquet", index=False)
            stats[a][name] = {
                "dead_frames": len(log), "recovered_frames": sum(1 for r in log if r["source"]),
                "by_source": {s: sum(1 for r in log if r["source"] == s)
                              for s in ("keypoint", "line")},
                "rows_recovered_by_stage": rec,
                "pitch_rows_control": int(np.isfinite(onrecord["pitch_x"]).sum()),
                "pitch_rows_arm": int(np.isfinite(filled["pitch_x"]).sum()),
                "control_reproduction": drift,
                "log": log,
            }
            logger.info("%s %s: %d/%d dead frames recovered (%s), +%d rows", name, a,
                        stats[a][name]["recovered_frames"], len(log),
                        stats[a][name]["by_source"], rec)
    return stats


def _on_record_only(out_dir: Path, name: str) -> dict[int, list]:
    """Candidates from the on-record cache alone (arm ``a`` may not use this session's top-up)."""
    from tools.gsr_calibgate import load_cache  # noqa: PLC0415

    path = out_dir / CACHE_SUBDIR / f"{name}.npz"
    if not path.exists():
        return {}
    by_frame, _d, _c = load_cache(path)
    return {f: c for f, c in by_frame.items() if c}


def _coord_drift(repro: pd.DataFrame, onrecord: pd.DataFrame) -> dict:
    """Max/median coordinate difference between a rebuilt control and the on-record parquet."""
    a = repro[["pitch_x", "pitch_y"]].to_numpy(dtype=float)
    b = onrecord[["pitch_x", "pitch_y"]].to_numpy(dtype=float)
    if a.shape != b.shape:
        return {"shape_mismatch": True}
    both = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    d = np.linalg.norm(a[both] - b[both], axis=1)
    return {"n": int(both.sum()), "median_m": _q(d, 50), "max_m": float(d.max()) if len(d) else 0.0,
            "rows_only_repro": int((np.isfinite(a).all(axis=1) & ~np.isfinite(b).all(axis=1)).sum()),
            "rows_only_onrecord": int((~np.isfinite(a).all(axis=1)
                                       & np.isfinite(b).all(axis=1)).sum())}


# === GT-anchored accuracy of what the stage wrote =================================================
def gt_accuracy(data_dir: Path, out_dir: Path, seqs: list[str], arms: tuple[str, ...]) -> dict:
    """Error against ground truth on exactly the rows each arm added, and on the bridged rows.

    Ground truth is read **only to score**; no arm consumes it.
    """
    from tools.gsr_w2_calibswap import gt_matched_error, gt_people  # noqa: PLC0415

    rep: dict[str, dict] = {}
    for name in seqs:
        ctrl = pd.read_parquet(out_dir / CONTROL_SUBDIR / f"{name}.parquet")
        gt = gt_people(data_dir / name)
        had = np.isfinite(ctrl["pitch_x"].to_numpy(dtype=float))
        # rows the control only has because fill_calibration_gaps interpolated a neighbour's
        # geometry: the stage replaces those with a real per-frame homography (W2 section 2.7).
        pre = set(gated_table(out_dir, name).pipe(lambda g: g.loc[
            g["role"].isin(PLAYER_ROLES) & np.isfinite(g["pitch_x"]), "frame"]).astype(int))
        bridged = had & ~np.isin(ctrl["frame"].astype(int), list(pre))
        per_arm = {}
        for a in arms:
            arm = pd.read_parquet(out_dir / ARM_SUBDIR[a] / f"{name}.parquet")
            fin = np.isfinite(arm["pitch_x"].to_numpy(dtype=float))
            new = fin & ~had
            entry: dict = {"added_rows": int(new.sum()),
                           "rows_lost_vs_control": int((had & ~fin).sum())}
            if new.sum():
                sub = arm[new].reset_index(drop=True)
                acc = gt_matched_error(sub, gt, {"arm": (sub["pitch_x"].to_numpy(dtype=float),
                                                        sub["pitch_y"].to_numpy(dtype=float))})
                entry.update(gt_matched=acc["_matched"],
                             **{k: v for k, v in acc["arm"].items()
                                if k in ("n", "median", "p90", "within_5m")})
            if bridged.any():
                sub = ctrl[bridged].reset_index(drop=True)
                acc = gt_matched_error(sub, gt, {
                    "fill": (ctrl.loc[bridged, "pitch_x"].to_numpy(dtype=float),
                             ctrl.loc[bridged, "pitch_y"].to_numpy(dtype=float)),
                    "arm": (arm.loc[bridged, "pitch_x"].to_numpy(dtype=float),
                            arm.loc[bridged, "pitch_y"].to_numpy(dtype=float))})
                entry["bridged"] = {"rows": int(bridged.sum()), "paired": acc["_paired"],
                                    "fill": acc["fill"], "arm": acc["arm"]}
            per_arm[a] = entry
        rep[name] = per_arm
        logger.info("%s: %s", name, {a: (v.get("added_rows"), round(v.get("median", float("nan")), 2))
                                     for a, v in per_arm.items()})
    return rep


# === scoring =====================================================================================
def run_arms(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str],
             arms: tuple[str, ...]) -> dict:
    """Score the control and each arm through the frozen v6 chain, paired per sequence."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415
    from eval.gsr_identity import paired_stats  # noqa: PLC0415
    from tools.gsr_v6det import TAU  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache" + VARIANT
    out: dict[str, dict] = {}
    for arm in arms:
        sub = CONTROL_SUBDIR if arm == "control" else ARM_SUBDIR[arm]
        res = eiou.run_point(data_dir, out_dir, seqs,
                             eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                             embedder="clip" + VARIANT, tau=TAU, tag=f"w2b_{arm}",
                             percrop_variant="_v6" + VARIANT, positions_subdir=sub)
        out[arm] = {"positions_subdir": sub, "gs_hota": res["gs_hota"],
                    "gs_hota_per_seq": res["gs_hota_per_seq"],
                    "eiou_tracks": (res["eiou"]["n_tracks_before"], res["eiou"]["n_tracks_after"])}
        h = res["gs_hota"]
        print(f"{arm:<8} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
              f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f} IDF1 {h['IDF1']:7.4f}", flush=True)
    base = out["control"]["gs_hota_per_seq"]
    for arm in arms:
        if arm != "control":
            out[arm]["paired_vs_control"] = paired_stats(base, out[arm]["gs_hota_per_seq"], seqs)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "gsr_v8_w2b_arms.json"
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
    """Self-check: the stage only touches dead frames, and the acceptance stack actually gates."""
    from generator.calibrate import CalibCandidate  # noqa: PLC0415
    from generator.line_calib import pitch_model_lines, recover_dead_frames  # noqa: PLC0415

    h = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])  # 10 px = 1 m
    rows = []
    for fr in (0, 1):
        for i in range(4):  # frame 0 live, frame 1 dead
            rows.append({"frame": fr, "track_id": i, "role": "player", "team": i % 2,
                         "pitch_x": 10.0 + 5 * i if fr == 0 else np.nan,
                         "pitch_y": 20.0 + 3 * i if fr == 0 else np.nan,
                         "image_x": 100.0 + 50 * i, "image_y": 200.0 + 30 * i,
                         "conf": 1.0, "calib_error_m": 0.1, "is_actor": False, "is_keeper": False})
    df = pd.DataFrame(rows)
    cand = [CalibCandidate(homography=h, error_m=0.2, n_points=9, mode="full", use_ransac=0.0,
                           rep_err_px=1.0)]
    blank = np.zeros((400, 800, 3), np.uint8)  # no grass, no lines -> zero support

    out, rec = recover_dead_frames(df, {1: blank}, candidates={1: cand}, min_support=0.0,
                                  refit=False)
    assert rec == 4, rec
    assert np.allclose(out.loc[out.frame == 1, "pitch_x"], [10.0, 15.0, 20.0, 25.0])
    assert np.allclose(out.loc[out.frame == 0, "pitch_x"], [10.0, 15.0, 20.0, 25.0]), "live frozen"

    out2, rec2 = recover_dead_frames(df, {1: blank}, candidates={1: cand}, min_support=0.5,
                                     refit=False)
    assert rec2 == 0, "an unsupported homography must be refused by the line gate"
    assert not np.isfinite(out2.loc[out2.frame == 1, "pitch_x"]).any()

    bad = [CalibCandidate(homography=h, error_m=9.9, n_points=9, mode="full", use_ransac=0.0,
                          rep_err_px=1.0)]
    _o3, rec3 = recover_dead_frames(df, {1: blank}, candidates={1: bad}, min_support=0.0,
                                    refit=False)
    assert rec3 == 0, "the metre gate must still refuse a 9.9 m hypothesis"

    _o4, rec4 = recover_dead_frames(df, {1: None}, candidates={1: cand}, min_support=0.0,
                                    refit=False)
    assert rec4 == 0, "no image -> no recovery"
    assert len(pitch_model_lines()) == 41
    print("gsr_w2b_linefall demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--arms-list", default="a,b,c")
    ap.add_argument("--min-support", type=float, default=None,
                    help="line-support acceptance threshold (default: read the --support report)")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--support", action="store_true")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    seqs = args.seqs.split(",") if args.seqs else PROBE
    arms = tuple(a for a in args.arms_list.split(",") if a)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, obj) -> None:
        dest = args.results_dir / f"gsr_v8_w2b_{name}_{args.tag}.json"
        dest.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
        print(f"wrote {dest}")

    if args.demo:
        _demo()
        return
    if args.census:
        _write("census", census(args.out_dir, seqs))
    if args.cache:
        _write("cache", cache_missing(args.data_dir, args.out_dir, seqs))
    if args.support:
        _write("support", support_on_live(args.data_dir, args.out_dir, seqs))
    min_support = args.min_support
    if (args.positions or args.gt) and min_support is None:
        path = args.results_dir / f"gsr_v8_w2b_support_{args.tag}.json"
        min_support = round(float(json.loads(path.read_text(encoding="utf-8"))["_pooled"]["p05"]), 3)
        print(f"min_support from {path}: {min_support}")
    if args.positions:
        _write("positions", build_arms(args.data_dir, args.out_dir, seqs, arms,
                                       min_support=min_support))
    if args.gt:
        _write("gt", gt_accuracy(args.data_dir, args.out_dir, seqs, arms))
    if args.arms:
        out = run_arms(args.data_dir, args.out_dir, args.results_dir, seqs, ("control", *arms))
        print("control vs on-record:", control_check(out, seqs))


if __name__ == "__main__":
    main()
