"""v10-W10: what the W7 merge+dedup oracle's 9,202 row deletions are actually worth (measure-only).

Registered in ``results/gsr_v10_w10_registered.json`` before any number below existed.

kb v10-w7-002 prices the merge+dedup oracle at +4.4999 GS-HOTA (DetA +3.5048) on DEV-20, and its
only destructive act is deleting 9,202 rows. kb v10-w7-005 and v10-w9-002 then showed the labels
that oracle consumes are noisy: 77% of ownership disagreements last <= 5 rows, and 63% of the
"duplicate" pairs have NEITHER track on the person they both claim. This module decomposes the
9,202 deletions by WHY the row was deleted, prices each bucket with the official evaluator, and
re-reads the buckets under a labeller-noise-tolerant matcher (instrument side only).

CLI::

    python -m tools.gsr_v10_w10 --decompose   # CPU: bucket the 9,202 rows, strict and tolerant
    python -m tools.gsr_v10_w10 --price       # CPU: per-bucket counterfactuals, official evaluator
    python -m tools.gsr_v10_w10 --signals     # CPU: GT-free separability (rank AUC per feature)
    python -m tools.gsr_v10_w10 --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_v10_w5 import DEV20
from tools.gsr_v10_w7 import ARM_D_PRED, merge_map, ownership, submission_rows

logger = logging.getLogger("gsr_v10_w10")

#: Session roots (never the freeze or the W8 directories).
WORK = Path("outputs/gsr/v10_w10")
RESULTS = Path("results/gsr_benchmark/gsr_v10_w10.json")
#: Per-frame calibration-hypothesis cache of the arm-D lineage (v10-W2 control run).
CALIB_CACHE = Path("outputs/gsr/v10_w2_full/ctrl")

#: A disagreement run this short is labeller flicker (the W7 convention, kb v10-w7-005).
FLICKER_MAX = 5
#: A disagreement run this long is a stable ownership difference (1 s at 25 fps).
STABLE_MIN = 25
#: Two GT players closer than this in a frame are interchangeable to the labeller (section 2).
TOL_M = 1.5
#: Signal-audit neighbourhood radii.
DENSITY_R_M = 2.0
PATH_WINDOW = 12

#: Buckets whose defining signal is GT-free-computable in principle (registered section 4).
QUALIFYING = ("junk", "dup_owner", "wrong_stable")


# === section 1: decomposition =====================================================================
def _runs(flags: list[bool]) -> list[int]:
    """Per-element length of the maximal consecutive True run containing it (pure).

    False elements get 0. ``[F, T, T, F, T] -> [0, 2, 2, 0, 1]``.
    """
    out = [0] * len(flags)
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        for k in range(i, j):
            out[k] = j - i
        i = j
    return out


def bucket_of(gt_id: int, owner: int, run: int) -> str:
    """Registered bucket name for one deleted row (pure).

    Args:
        gt_id: Nearest GT identity within 5 m, or -1.
        owner: The track's trusted dominant GT identity, or -1.
        run: Length of the disagreement run holding this row (0 when it agrees).
    """
    if owner < 0:
        return "no_owner"
    if gt_id < 0:
        return "junk"
    if gt_id == owner or run <= 0:
        return "dup_owner"
    if run <= FLICKER_MAX:
        return "wrong_flicker"
    if run >= STABLE_MIN:
        return "wrong_stable"
    return "wrong_mid"


def drop_table(seq: str, data_dir: Path, src: Path = ARM_D_PRED) -> pd.DataFrame:
    """Every arm-D row of one clip with the oracle's drop decision and its bucket.

    The drop set is re-derived exactly as :func:`tools.gsr_v10_w7.apply_merges` builds it in
    ``merge_dedup`` mode: rows are grouped by (trusted owner identity, frame) -- untrusted tracks
    each form their own group -- and every row but the highest-confidence one in a group is deleted.

    Args:
        seq: Sequence name.
        data_dir: GSR dataset root.
        src: ``predictions/data`` of the arm being decomposed.

    Returns:
        Frame with ``idx, track_id, frame, conf, pitch_x, pitch_y, gt_id, owner, nid, dropped,
        run, run_tol, bucket, bucket_tol, owner_absent``.
    """
    from generator.track_relink import load_gt_ids_by_frame  # noqa: PLC0415
    from tools.gsr_v9_factory import label_rows  # noqa: PLC0415

    preds = json.loads((src / f"{seq}.json").read_text(encoding="utf-8"))["predictions"]
    rows = submission_rows(src / f"{seq}.json", data_dir / seq)
    labelled = label_rows(rows, data_dir / seq)
    own = ownership(labelled)
    owner = dict(zip(own["track_id"].astype(int), own["gt_id"].astype(int)))
    row_gt = dict(zip(labelled["idx"], labelled["gt_id"]))
    new, _st = merge_map(own, None)
    t = rows.assign(
        conf=[float(preds[i].get("confidence") or 0.0) for i in rows["idx"]],
        owner=[owner.get(int(x), -1) for x in rows["track_id"]],
        gt_id=[int(row_gt.get(i, -1)) for i in rows["idx"]],
        nid=[new.get(int(x), int(x)) for x in rows["track_id"]],
    )
    t["gid"] = [o if o >= 0 else -int(x) - 1 for o, x in zip(t["owner"], t["track_id"])]
    size = t.groupby(["gid", "frame"])["idx"].transform("size")
    keep = set(t.sort_values("conf", ascending=False).groupby(["gid", "frame"])["idx"].first())
    t["dropped"] = [i not in keep and n > 1 for i, n in zip(t["idx"], size)]

    gt = load_gt_ids_by_frame(data_dir / seq)
    runs = np.zeros(len(t), dtype=int)
    runs_tol = np.zeros(len(t), dtype=int)
    absent = np.zeros(len(t), dtype=bool)
    pos = {i: k for k, i in enumerate(t["idx"])}
    for tid, g in t[(t["gt_id"] >= 0) & (t["owner"] >= 0)].groupby("track_id"):
        g = g.sort_values("frame")
        strict, tol, miss = [], [], []
        for f, gid, o in zip(g["frame"], g["gt_id"], g["owner"]):
            here = {int(i): (x, y) for x, y, i in gt.get(int(f), [])}
            far = True
            if int(gid) != int(o):
                a, b = here.get(int(gid)), here.get(int(o))
                far = a is None or b is None or float(np.hypot(a[0] - b[0], a[1] - b[1])) > TOL_M
                miss.append(b is None)
            else:
                miss.append(False)
            strict.append(int(gid) != int(o))
            tol.append(int(gid) != int(o) and far)
        for i, rs, rt, m in zip(g["idx"], _runs(strict), _runs(tol), miss):
            runs[pos[i]], runs_tol[pos[i]], absent[pos[i]] = rs, rt, m
    t["run"], t["run_tol"], t["owner_absent"] = runs, runs_tol, absent
    # A tolerated row keeps its foreign gt_id but carries run_tol == 0, which bucket_of reads as a
    # duplicate-match competition -- exactly the registered tolerant re-reading.
    t["bucket"] = [bucket_of(g, o, r) if d else "kept"
                   for g, o, r, d in zip(t["gt_id"], t["owner"], t["run"], t["dropped"])]
    t["bucket_tol"] = [bucket_of(g, o, rt) if d else "kept"
                       for g, o, rt, d in zip(t["gt_id"], t["owner"], t["run_tol"], t["dropped"])]
    t["seq"] = seq
    return t


def decompose(seqs: list[str], data_dir: Path, src: Path = ARM_D_PRED) -> tuple[pd.DataFrame, dict]:
    """Bucket the whole DEV drop set, strict and tolerant (returns the table and a summary)."""
    tabs = []
    for seq in seqs:
        t = drop_table(seq, data_dir, src)
        tabs.append(t)
        logger.info("%s: %d rows, %d dropped", seq, len(t), int(t["dropped"].sum()))
    t = pd.concat(tabs, ignore_index=True)
    d = t[t["dropped"]]
    summary = {
        "rows": int(len(t)), "dropped": int(len(d)),
        "strict": {k: int(v) for k, v in d["bucket"].value_counts().items()},
        "tolerant": {k: int(v) for k, v in d["bucket_tol"].value_counts().items()},
        "migration": {f"{a}->{b}": int(n) for (a, b), n
                      in d.groupby(["bucket", "bucket_tol"]).size().items() if a != b},
        "owner_absent_in_frame": int(d["owner_absent"].sum()),
        "run_len_median_wrong": float(d[d["run"] > 0]["run"].median()) if (d["run"] > 0).any()
        else float("nan"),
        "per_seq_dropped": {s: int(g["dropped"].sum()) for s, g in t.groupby("seq")},
    }
    return t, summary


# === section 1/2: counterfactual pricing ==========================================================
def write_variant(src: Path, dest: Path, seqs: list[str], drop: dict[str, set[int]],
                  relabel: dict[str, dict[int, int]] | None = None) -> dict:
    """Write a submission with ``drop`` removed and (axis B) ``relabel`` applied (per sequence)."""
    dest.mkdir(parents=True, exist_ok=True)
    stats = {}
    for seq in seqs:
        preds = json.loads((src / f"{seq}.json").read_text(encoding="utf-8"))["predictions"]
        nid = (relabel or {}).get(seq, {})
        for p in preds:
            p["track_id"] = int(nid.get(int(p["track_id"]), int(p["track_id"])))
        gone = drop.get(seq, set())
        out = [p for i, p in enumerate(preds) if i not in gone]
        (dest / f"{seq}.json").write_text(json.dumps({"predictions": out}), encoding="utf-8")
        stats[seq] = {"rows_in": len(preds), "rows_out": len(out), "dropped": len(gone)}
    return stats


def _merge_all_map(t: pd.DataFrame) -> dict[str, dict[int, int]]:
    """``{seq: {old track id: merged id}}`` for the oracle's merge-all step (pure)."""
    return {s: dict(zip(g["track_id"].astype(int), g["nid"].astype(int)))
            for s, g in t.groupby("seq")}


def legality(t: pd.DataFrame) -> dict:
    """Is any deletion bucket OPTIONAL inside the merge+dedup oracle? (pure)

    The registered axis-B decomposition wanted a 'merge-all, no deletion' baseline. The official
    evaluator refuses it: ``trackeval`` raises ``Tracker predicts the same ID more than once in a
    single timestep`` before any metric is computed. This quantifies why -- for each bucket, how
    many (merged id, frame) timesteps would carry two rows if only that bucket were restored. A
    non-zero count means the bucket is a LEGALITY PRECONDITION of the merge, not a value bucket
    that can be priced on its own.
    """
    out: dict = {"buckets": {}}
    for b in sorted(x for x in t["bucket"].unique() if x != "kept"):
        restored = t[~t["dropped"] | (t["bucket"] == b)]
        n = restored.groupby(["seq", "nid", "frame"]).size()
        out["buckets"][b] = {"rows_restored": int((t["bucket"] == b).sum()),
                             "illegal_timesteps": int((n > 1).sum()),
                             "optional": bool((n > 1).sum() == 0)}
    n_all = t.groupby(["seq", "nid", "frame"]).size()
    out["merge_all_no_deletion"] = {"illegal_timesteps": int((n_all > 1).sum()),
                                    "scoreable": bool((n_all > 1).sum() == 0)}
    kept = t[~t["dropped"]].groupby(["seq", "nid", "frame"]).size()
    out["full_oracle"] = {"illegal_timesteps": int((kept > 1).sum()),
                          "scoreable": bool((kept > 1).sum() == 0)}
    out["any_bucket_optional"] = any(v["optional"] for v in out["buckets"].values())
    return out


def price(name: str, t: pd.DataFrame, buckets: tuple[str, ...], col: str, data_dir: Path,
          seqs: list[str], *, merged: bool, base: str = "D") -> dict:
    """Score one counterfactual with the official evaluator, flags ON and OFF.

    Args:
        name: Variant name (also the working directory).
        t: Output of :func:`decompose`.
        buckets: Bucket names whose rows the arm deletes.
        col: ``bucket`` (strict) or ``bucket_tol`` (tolerant).
        data_dir: GSR dataset root.
        seqs: Sequences.
        merged: Apply the oracle's merge-all id map to the ARM (axis B).
        base: Control -- ``D`` (untouched arm D) or ``merge_all`` (merge-all, no deletion), so an
            axis-B delta is the deletion's MARGINAL value inside the oracle rather than the
            merge's value again.
    """
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    sel = t[t[col].isin(buckets)]
    drop = {s: set(g["idx"].astype(int)) for s, g in sel.groupby("seq")}
    relabel = _merge_all_map(t) if merged else None
    root = WORK / ("axisB" if merged else "axisA") / name
    arm = write_variant(ARM_D_PRED, root / "predictions" / "data", seqs, drop, relabel)
    ctrl = ARM_D_PRED
    if base == "merge_all":
        ctrl = WORK / "axisB" / "_merge_all" / "predictions" / "data"
        if not (ctrl / f"{seqs[-1]}.json").exists():
            write_variant(ARM_D_PRED, ctrl, seqs, {}, _merge_all_map(t))
    scored = score_pair(ctrl, root / "predictions" / "data", data_dir, root / "pair", seqs)
    return {"buckets": list(buckets), "column": col, "merged": merged, "control": base,
            "rows_deleted": int(len(sel)),
            "per_seq": arm,
            "delta": {f: scored[f]["delta"] for f in ("on", "off")},
            "paired": {f: scored[f]["paired"] for f in ("on", "off")},
            "arm_gs_hota": {f: scored[f]["arm"]["combined"]["GS-HOTA"] for f in ("on", "off")},
            "control_gs_hota": {f: scored[f]["control"]["combined"]["GS-HOTA"]
                                for f in ("on", "off")}}


# === section 3: GT-free signal audit ==============================================================
def features(seq: str, data_dir: Path, src: Path = ARM_D_PRED) -> pd.DataFrame:
    """Per-row GT-FREE observables for one clip (``idx`` joins :func:`drop_table`).

    Every column is computable without ground truth: confidence, box height, own-track length,
    deviation from the track's own local path, crowding (nearest neighbour, density, box overlap)
    and the frame's calibration residual.
    """
    preds = json.loads((src / f"{seq}.json").read_text(encoding="utf-8"))["predictions"]
    rows = submission_rows(src / f"{seq}.json", data_dir / seq)
    box = np.array([[(p.get("bbox_image") or {}).get(k, np.nan) for k in ("x", "y", "w", "h")]
                    for p in preds], dtype=float)
    t = rows.assign(conf=[float(preds[i].get("confidence") or 0.0) for i in rows["idx"]],
                    box_h_px=box[rows["idx"].to_numpy(), 3])
    t["track_rows"] = t.groupby("track_id")["idx"].transform("size")

    # deviation from the track's own local path (rolling median over +/- PATH_WINDOW frames)
    dev = np.full(len(t), np.nan)
    pos = {i: k for k, i in enumerate(t["idx"])}
    for _tid, g in t.groupby("track_id"):
        g = g.sort_values("frame")
        x, y = g["pitch_x"].to_numpy(float), g["pitch_y"].to_numpy(float)
        w = 2 * PATH_WINDOW + 1
        mx = pd.Series(x).rolling(w, center=True, min_periods=3).median().to_numpy()
        my = pd.Series(y).rolling(w, center=True, min_periods=3).median().to_numpy()
        for i, d in zip(g["idx"], np.hypot(x - mx, y - my)):
            dev[pos[i]] = d
    t["path_dev_m"] = dev

    nn = np.full(len(t), np.nan)
    dens = np.zeros(len(t))
    iou = np.zeros(len(t))
    for _f, g in t.groupby("frame"):
        ix = g["idx"].to_numpy()
        p = g[["pitch_x", "pitch_y"]].to_numpy(float)
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        np.fill_diagonal(d, np.inf)
        b = box[ix]
        x1 = np.maximum(b[:, None, 0], b[None, :, 0])
        y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 0] + b[:, None, 2], b[None, :, 0] + b[None, :, 2])
        y2 = np.minimum(b[:, None, 1] + b[:, None, 3], b[None, :, 1] + b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = b[:, 2] * b[:, 3]
        union = area[:, None] + area[None, :] - inter
        ov = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        np.fill_diagonal(ov, 0.0)
        for k, i in enumerate(ix):
            nn[pos[i]] = float(np.nanmin(d[k])) if len(ix) > 1 else np.nan
            dens[pos[i]] = float(np.sum(d[k] <= DENSITY_R_M))
            iou[pos[i]] = float(np.nanmax(ov[k])) if len(ix) > 1 else 0.0
    t["nn_dist_m"], t["density_2m"], t["max_box_iou"] = nn, dens, iou
    t["calib_error_m"] = [_calib(seq).get(int(f), np.nan) for f in t["frame"]]
    t["seq"] = seq
    return t[["seq", "idx", *SIGNALS]]


#: The registered GT-free observables (section 3).
SIGNALS = ("conf", "box_h_px", "track_rows", "path_dev_m", "nn_dist_m", "density_2m",
           "max_box_iou", "calib_error_m")

_CALIB: dict[str, dict[int, float]] = {}


def _calib(seq: str) -> dict[int, float]:
    """``{frame: solver keypoint error in metres}`` from the v10-W2 gate cache (memoised)."""
    if seq in _CALIB:
        return _CALIB[seq]
    try:
        from tools.gsr_v10_w5 import load_seq  # noqa: PLC0415
        _df, _bf, _h, err, _n = load_seq(seq, Path("outputs/gsr"), CALIB_CACHE)
        _CALIB[seq] = {int(k): float(v) for k, v in err.items()}
    except Exception as exc:  # noqa: BLE001 -- an absent cache must not kill the audit
        logger.warning("%s: calibration cache unavailable (%s)", seq, exc)
        _CALIB[seq] = {}
    return _CALIB[seq]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC of ``pos`` over ``neg``, NaNs dropped (pure). 0.5 = no separation."""
    from scipy.stats import rankdata  # noqa: PLC0415

    p = pos[np.isfinite(pos)]
    n = neg[np.isfinite(neg)]
    if not len(p) or not len(n):
        return float("nan")
    r = rankdata(np.concatenate([p, n]))
    u = float(r[:len(p)].sum()) - len(p) * (len(p) + 1) / 2.0
    return u / (len(p) * len(n))


def signal_audit(t: pd.DataFrame, feat: pd.DataFrame, buckets: tuple[str, ...],
                 col: str = "bucket_tol") -> dict:
    """Rank AUC of every registered signal, deleted-in-``buckets`` rows against kept rows."""
    m = t.drop(columns=[c for c in SIGNALS if c in t.columns]).merge(
        feat, on=["seq", "idx"], how="left")
    neg = m[~m["dropped"]]
    out: dict = {"kept_rows": int(len(neg)), "per_bucket": {}}
    groups = [("_union", tuple(buckets)), *[(b, (b,)) for b in buckets]]
    for name, bs in groups:
        sel = m[m[col].isin(bs)]
        out["per_bucket"][name] = {
            "n": int(len(sel)),
            "auc": {s: round(auc(sel[s].to_numpy(float), neg[s].to_numpy(float)), 4)
                    for s in SIGNALS},
            "median_deleted": {s: (round(float(np.nanmedian(sel[s])), 4) if len(sel) else None)
                               for s in SIGNALS},
        }
    out["median_kept"] = {s: round(float(np.nanmedian(neg[s])), 4) for s in SIGNALS}
    return out


# === self-check ===================================================================================
def _demo() -> None:
    """Assert the pure seams: run lengths, bucket naming and the rank AUC."""
    assert _runs([False, True, True, False, True]) == [0, 2, 2, 0, 1]
    assert _runs([]) == [] and _runs([True] * 3) == [3, 3, 3]
    assert bucket_of(-1, 7, 0) == "junk"
    assert bucket_of(7, 7, 0) == "dup_owner"
    assert bucket_of(9, 7, 0) == "dup_owner", "a tolerated row reads as a duplicate"
    assert bucket_of(9, 7, 3) == "wrong_flicker"
    assert bucket_of(9, 7, FLICKER_MAX + 1) == "wrong_mid"
    assert bucket_of(9, 7, STABLE_MIN) == "wrong_stable"
    assert bucket_of(9, -1, 30) == "no_owner"
    assert abs(auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9
    assert abs(auc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) - 0.5) < 1e-9
    assert abs(auc(np.array([np.nan, 3.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9
    assert np.isnan(auc(np.array([np.nan]), np.array([1.0])))
    # the drop rule keeps exactly one row per (owner, frame) group and never touches a singleton
    t = pd.DataFrame({"idx": [0, 1, 2], "frame": [5, 5, 6], "gid": [3, 3, 3],
                      "conf": [0.4, 0.9, 0.1]})
    size = t.groupby(["gid", "frame"])["idx"].transform("size")
    keep = set(t.sort_values("conf", ascending=False).groupby(["gid", "frame"])["idx"].first())
    assert [i not in keep and n > 1 for i, n in zip(t["idx"], size)] == [True, False, False]
    print("gsr_v10_w10 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--decompose", action="store_true", help="bucket the drop set (no metric read)")
    ap.add_argument("--price", action="store_true", help="per-bucket counterfactuals, evaluator")
    ap.add_argument("--axis", default="A,B", help="A = prune-only, B = inside the oracle")
    ap.add_argument("--signals", action="store_true", help="GT-free separability (rank AUC)")
    ap.add_argument("--legality", action="store_true",
                    help="is any deletion bucket optional inside the oracle? (no metric read)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else DEV20
    payload: dict = {}
    t0 = time.time()
    table = WORK / "drop_table.parquet"
    if args.decompose or not table.exists():
        t, summary = decompose(seqs, args.data_dir)
        table.parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(table)
        payload["decomposition"] = summary
        print(json.dumps(summary, indent=1))
    t = pd.read_parquet(table)

    if args.price:
        strict = [b for b in t["bucket"].unique() if b != "kept"]
        tol = [b for b in t["bucket_tol"].unique() if b != "kept"]
        payload["pricing"] = {}
        jobs: list[tuple[str, tuple[str, ...], str, bool, str]] = []
        if "A" in args.axis:
            jobs += [("A_all", tuple(strict), "bucket", False, "D")]
            jobs += [(f"A_strict_{b}", (b,), "bucket", False, "D") for b in sorted(strict)]
            jobs += [(f"A_tol_{b}", (b,), "bucket_tol", False, "D") for b in sorted(tol)]
            jobs += [("A_strict_reachable", tuple(b for b in QUALIFYING if b in strict),
                      "bucket", False, "D"),
                     ("A_tol_reachable", tuple(b for b in QUALIFYING if b in tol),
                      "bucket_tol", False, "D")]
        if "B" in args.axis:
            # Registered axis B is UNDEFINED: see legality() and --legality. Its baseline
            # (merge-all, no deletion) is refused by the official evaluator, so no per-bucket
            # decomposition of the +4.4999 exists. Only the full oracle is scoreable.
            jobs += [("B_all", tuple(strict), "bucket", True, "D")]
        for name, bs, col, merged, base in jobs:
            r = price(name, t, bs, col, args.data_dir, seqs, merged=merged, base=base)
            payload["pricing"][name] = r
            print(f"{name}: {r['rows_deleted']} rows | ON {r['delta']['on']['GS-HOTA']:+.4f} "
                  f"(DetA {r['delta']['on']['GS-DetA']:+.4f} "
                  f"AssA {r['delta']['on']['GS-AssA']:+.4f}) | "
                  f"OFF {r['delta']['off']['GS-HOTA']:+.4f} | "
                  f"{r['paired']['on']['GS-HOTA']['helped']}/"
                  f"{r['paired']['on']['GS-HOTA']['hurt']} "
                  f"p={r['paired']['on']['GS-HOTA']['wilcoxon_p']:.3g}")

    if args.legality:
        payload["legality"] = legality(t)
        payload["legality"]["evaluator_refusal_verbatim"] = (
            "trackeval.utils.TrackEvalException: Tracker predicts the same ID more than once in "
            "a single timestep (seq: SNGS-021, frame: 67, ids: 25)")
        print(json.dumps(payload["legality"], indent=1))

    if args.signals:
        cache = WORK / "features.parquet"
        if cache.exists():
            feat = pd.read_parquet(cache)
        else:
            feat = pd.concat([features(s, args.data_dir) for s in seqs], ignore_index=True)
            feat.to_parquet(cache)
        payload["signals"] = {
            "tolerant_qualifying": signal_audit(t, feat, QUALIFYING, "bucket_tol"),
            "strict_qualifying": signal_audit(t, feat, QUALIFYING, "bucket"),
            "all_dropped_strict": signal_audit(
                t, feat, tuple(b for b in t["bucket"].unique() if b != "kept"), "bucket"),
        }
        for k, v in payload["signals"]["tolerant_qualifying"]["per_bucket"].items():
            print(k, v["n"], v["auc"])

    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        for key, val in payload.items():  # one level deep: a second --axis must not clobber the
            if isinstance(val, dict) and isinstance(prev.get(key), dict):  # first one's variants
                prev[key].update(val)
            else:
                prev[key] = val
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
