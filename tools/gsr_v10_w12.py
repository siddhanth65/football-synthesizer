"""v10 W12: does a modern detector actually change our distribution? -- the registered kill test.

Registration: ``results/gsr_v10_w12_registered.json`` (written before any measurement).

The +0.57 GS-DetA "perfect detector" ceiling of kb v9-w7-001 prices **misses only**. A modern
detector also changes box geometry, confidence calibration and role-class quality, and those feed
the 66%-of-misses bucket that dies *after* the detector (ByteTrack's Kalman re-match). This module
compares two or more box dumps produced **in one GPU window** (kb v10-frz-003) on three axes:

* **A -- recall** vs GT rows at every confidence, and at the operating point where the arm emits the
  same number of person boxes as the control does at ``DETECT_CONF`` 0.20;
* **B -- box quality** on matched pairs: IoU with the GT box and foot-point error in GT-box-height
  units (the quantity the homography consumes);
* **C -- chain-discard survival**: a CPU ByteTrack replay (``tools.gsr_v9_w7.replay_tracker``) at the
  shipped settings, i.e. how much of each detector's recall survives the stage that discards 8% of
  the confident boxes today.

All arms are read from parquet dumps in the schema of ``tools/gsr_v9_w7_infer.py``; no GPU here.

CLI::

    python -m tools.gsr_v10_w12 --arms s4b_640=~/work/v10w12/s4b_640,rfdetr=... --out results/...
    python -m tools.gsr_v10_w12 --demo
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_v9_deta import box_iou, load_gt
from tools.gsr_v9_w7 import gt_foot, match_feet, replay_tracker

logger = logging.getLogger("gsr_v10_w12")

#: Confidence grid the sweep is measured on (the shipped floor 0.20 is a member).
GRID: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                           0.60, 0.70, 0.80, 0.90)
#: Roles that count as people (the ball never enters the person chain).
PEOPLE = ("player", "goalkeeper", "referee")
#: The shipped detector operating point.
SHIPPED_CONF = 0.20


def det_boxes(probe: pd.DataFrame, conf: float) -> dict[int, np.ndarray]:
    """Group a dump's person boxes at ``conf`` by frame as ``(N, 5)`` ``[x1,y1,x2,y2,conf]``."""
    keep = probe[probe["role"].isin(PEOPLE) & (probe["conf"] >= conf)]
    return {int(f): g[["x1", "y1", "x2", "y2", "conf"]].to_numpy(float)
            for f, g in keep.groupby("frame")}


def sweep_sequence(seq_dir: Path, probe: pd.DataFrame,
                   grid: tuple[float, ...] = GRID) -> dict[float, dict]:
    """Recall / precision / box quality of one dump against one sequence's GT, per threshold.

    Args:
        seq_dir: GSR sequence folder.
        probe: The detector dump for that sequence.
        grid: Confidence thresholds to measure.

    Returns:
        ``{threshold: {"gt", "matched", "dets", "iou", "foot"}}`` where ``iou``/``foot`` are lists of
        per-pair values (IoU with the GT box; foot-point error in GT box heights).
    """
    ts, gt_rows = load_gt(seq_dir)
    frame_of_ts = {k: int(img_id[-6:]) - 1 for img_id, k in ts.items()}
    out: dict[float, dict] = {c: {"gt": 0, "matched": 0, "dets": 0, "iou": [], "foot": []}
                              for c in grid}
    by_conf = {c: det_boxes(probe, c) for c in grid}
    for t, gts in enumerate(gt_rows):
        frame = frame_of_ts[t]
        boxes = [g["bbox_image"] for g in gts]
        for c in grid:
            b = by_conf[c].get(frame)
            rec = out[c]
            rec["gt"] += len(gts)
            if b is None or not len(b):
                continue
            rec["dets"] += len(b)
            foot = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
            pairs = match_feet(boxes, foot)
            rec["matched"] += len(pairs)
            for gi, di in pairs.items():
                d = b[di]
                rec["iou"].append(box_iou(boxes[gi], {"x": d[0], "y": d[1],
                                                      "w": d[2] - d[0], "h": d[3] - d[1]}))
                gx, gy = gt_foot(boxes[gi])
                rec["foot"].append(float(np.hypot(foot[di, 0] - gx, foot[di, 1] - gy)
                                         / max(boxes[gi]["h"], 1.0)))
    return out


def role_confusion(seq_dir: Path, probe: pd.DataFrame, conf: float = SHIPPED_CONF) -> dict:
    """Confusion of a dump's per-box role against GT role on matched pairs (axis D).

    Role errors are the census's second-largest attribute bucket (12,547 rows, +1.42 GS-DetA
    isolated / +8.13 marginal), and ``player -> referee`` alone is 7,032 of them. A detector with
    better role classes attacks that bucket directly -- a channel the miss oracle never priced.
    """
    ts, gt_rows = load_gt(seq_dir)
    frame_of_ts = {k: int(img_id[-6:]) - 1 for img_id, k in ts.items()}
    keep = probe[probe["role"].isin(PEOPLE) & (probe["conf"] >= conf)]
    by_frame = {int(f): g for f, g in keep.groupby("frame")}
    out: dict[str, int] = {}
    for t, gts in enumerate(gt_rows):
        g = by_frame.get(frame_of_ts[t])
        if g is None or not len(g):
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        roles = g["role"].tolist()
        foot = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        for gi, di in match_feet([x["bbox_image"] for x in gts], foot).items():
            key = f"{gts[gi]['attributes'].get('role')}->{roles[di]}"
            out[key] = out.get(key, 0) + 1
    return out


def _quant(values: list[float]) -> dict:
    """Median / mean / p90 of a value list (empty -> NaNs)."""
    if not values:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "p90": float("nan")}
    a = np.asarray(values, float)
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90))}


def run_roles(arm_dir: Path, data_dir: Path, seqs: list[str]) -> dict:
    """Pool :func:`role_confusion` over a sequence list and report per-GT-role accuracy."""
    total: dict[str, int] = {}
    for s in seqs:
        for k, v in role_confusion(data_dir / s, pd.read_parquet(arm_dir / f"{s}.parquet")).items():
            total[k] = total.get(k, 0) + v
    acc = {}
    for gt_role in ("player", "goalkeeper", "referee"):
        n = sum(v for k, v in total.items() if k.split("->")[0] == gt_role)
        right = total.get(f"{gt_role}->{gt_role}", 0)
        acc[gt_role] = {"n": n, "correct": right, "accuracy": right / n if n else float("nan")}
    n_all = sum(total.values())
    acc["all"] = {"n": n_all,
                  "correct": sum(v for k, v in total.items() if k.split("->")[0] == k.split("->")[1]),
                  "accuracy": (sum(v for k, v in total.items() if k.split("->")[0] == k.split("->")[1])
                               / n_all) if n_all else float("nan")}
    return {"confusion": total, "accuracy": acc}


def run_arm(arm_dir: Path, data_dir: Path, seqs: list[str], *, replay: bool = True) -> dict:
    """Sweep one detector dump over a sequence list and replay ByteTrack at the shipped point."""
    totals: dict[float, dict] = {c: {"gt": 0, "matched": 0, "dets": 0, "iou": [], "foot": []}
                                 for c in GRID}
    per_seq: dict[str, dict] = {}
    rep: dict[str, int] = defaultdict(int)
    for i, s in enumerate(seqs):
        probe = pd.read_parquet(arm_dir / f"{s}.parquet")
        got = sweep_sequence(data_dir / s, probe)
        for c, rec in got.items():
            for k in ("gt", "matched", "dets"):
                totals[c][k] += rec[k]
            totals[c]["iou"].extend(rec["iou"])
            totals[c]["foot"].extend(rec["foot"])
        per_seq[s] = {f"{c:g}": {"gt": rec["gt"], "matched": rec["matched"], "dets": rec["dets"]}
                      for c, rec in got.items()}
        if replay:
            ts, gt_rows = load_gt(data_dir / s)
            frame_of_ts = {k: int(img_id[-6:]) - 1 for img_id, k in ts.items()}
            r = replay_tracker(probe, gt_rows, frame_of_ts, detect_conf=SHIPPED_CONF, min_hits=3)
            for k, v in r.items():
                rep[k] += v
            per_seq[s]["replay"] = r
        logger.info("[%d/%d] %s: recall@0.20 %.4f", i + 1, len(seqs), s,
                    got[SHIPPED_CONF]["matched"] / max(got[SHIPPED_CONF]["gt"], 1))
    sweep = {f"{c:g}": {"gt": rec["gt"], "matched": rec["matched"], "dets": rec["dets"],
                        "recall": rec["matched"] / max(rec["gt"], 1),
                        "precision": rec["matched"] / max(rec["dets"], 1),
                        "iou": _quant(rec["iou"]), "foot": _quant(rec["foot"])}
             for c, rec in totals.items()}
    out = {"dir": str(arm_dir), "sweep": sweep, "per_seq": per_seq}
    if replay:
        out["replay"] = {**dict(rep), "recall": rep["recovered"] / max(rep["gt"], 1)}
    return out


def matched_operating_point(arm_sweep: dict, target_dets: int) -> tuple[str, dict]:
    """Pick the arm threshold whose detection count is closest to ``target_dets`` (pure)."""
    best = min(arm_sweep, key=lambda c: abs(arm_sweep[c]["dets"] - target_dets))
    return best, arm_sweep[best]


def compare(results: dict[str, dict], control: str) -> dict:
    """Build the registered A/B/C comparison table against the control arm (pure)."""
    ctl = results[control]["sweep"]
    target = ctl[f"{SHIPPED_CONF:g}"]["dets"]
    table = {}
    for name, res in results.items():
        c_key, at_match = matched_operating_point(res["sweep"], target)
        at_ship = res["sweep"][f"{SHIPPED_CONF:g}"]
        row = {
            "matched_op_conf": c_key,
            "A_recall_at_matched": at_match["recall"],
            "A_recall_at_0.20": at_ship["recall"],
            "A_precision_at_matched": at_match["precision"],
            "A_dets_at_matched": at_match["dets"],
            "A_recall_ceiling_at_0.01": res["sweep"]["0.01"]["recall"],
            "B_iou_median_at_matched": at_match["iou"]["median"],
            "B_foot_median_at_matched": at_match["foot"]["median"],
            "B_foot_p90_at_matched": at_match["foot"]["p90"],
        }
        if "replay" in res:
            row["C_replay_recall"] = res["replay"]["recall"]
            row["C_replay_rows"] = res["replay"]["rows"]
            row["C_replay_tracks"] = res["replay"]["tracks"]
            row["C_discard_frac"] = 1.0 - res["replay"]["rows"] / max(at_ship["dets"], 1)
        table[name] = row
    base = table[control]
    for name, row in table.items():
        row["dA_recall_matched"] = row["A_recall_at_matched"] - base["A_recall_at_matched"]
        row["dB_iou"] = row["B_iou_median_at_matched"] - base["B_iou_median_at_matched"]
        row["dB_foot_rel"] = (row["B_foot_median_at_matched"] / base["B_foot_median_at_matched"]
                              - 1.0) if base["B_foot_median_at_matched"] else float("nan")
        if "C_replay_recall" in row:
            row["dC_replay_recall"] = row["C_replay_recall"] - base.get("C_replay_recall", 0.0)
    return {"control": control, "target_dets": target, "table": table}


def verdict(table: dict, control: str) -> dict:
    """Apply the registered kill rule to one arm row set (pure)."""
    out = {}
    for name, row in table.items():
        if name == control:
            continue
        a = row["dA_recall_matched"] > 0.005
        b = (row["dB_iou"] > 0.02) or (row["dB_foot_rel"] < -0.10)
        c = row.get("dC_replay_recall", 0.0) > 0.005
        out[name] = {"A_pass": bool(a), "B_pass": bool(b), "C_pass": bool(c),
                     "axis_open": bool(a or b or c)}
    return out


def demo() -> None:
    """Assert the pure seams: grouping, operating-point pick, verdict thresholds."""
    df = pd.DataFrame({"frame": [0, 0, 1], "x1": [0.0, 5.0, 0.0], "y1": [0.0, 0.0, 0.0],
                       "x2": [10.0, 15.0, 10.0], "y2": [20.0, 20.0, 20.0],
                       "conf": [0.9, 0.1, 0.5], "role": ["player", "player", "ball"]})
    by = det_boxes(df, 0.2)
    assert list(by) == [0] and by[0].shape == (1, 5), by
    sw = {"0.1": {"dets": 100}, "0.2": {"dets": 60}, "0.5": {"dets": 20}}
    assert matched_operating_point(sw, 55)[0] == "0.2"
    t = {"ctl": {"dA_recall_matched": 0.0, "dB_iou": 0.0, "dB_foot_rel": 0.0},
         "arm": {"dA_recall_matched": 0.001, "dB_iou": 0.001, "dB_foot_rel": -0.01,
                 "dC_replay_recall": 0.0}}
    assert verdict(t, "ctl")["arm"]["axis_open"] is False
    t["arm"]["dB_foot_rel"] = -0.2
    assert verdict(t, "ctl")["arm"]["axis_open"] is True
    print("gsr_v10_w12 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="", help="name=dir,name=dir (first is the control)")
    ap.add_argument("--control", default="")
    ap.add_argument("--seqs", default="", help="comma list; default = the arm dir's parquets")
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--roles-only", action="store_true", help="axis D only (role vs GT role)")
    ap.add_argument("--out", type=Path,
                    default=Path("results/gsr_benchmark/gsr_v10_w12_stage1.json"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    arms = [a.split("=", 1) for a in args.arms.split(",") if a]
    if not arms:
        raise SystemExit("--arms is required")
    control = args.control or arms[0][0]
    if args.roles_only:
        payload = {}
        for name, d in arms:
            arm_dir = Path(d).expanduser()
            seqs = ([s for s in args.seqs.split(",") if s]
                    or sorted(p.stem for p in arm_dir.glob("*.parquet")))
            payload[name] = run_roles(arm_dir, args.data_dir, seqs)
            logger.info("%-14s role accuracy %s", name,
                        {k: round(v["accuracy"], 4) for k, v in payload[name]["accuracy"].items()})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logger.info("wrote %s", args.out)
        return
    results = {}
    for name, d in arms:
        arm_dir = Path(d).expanduser()
        seqs = ([s for s in args.seqs.split(",") if s]
                or sorted(p.stem for p in arm_dir.glob("*.parquet")))
        logger.info("arm %s: %d sequences from %s", name, len(seqs), arm_dir)
        results[name] = run_arm(arm_dir, args.data_dir, seqs, replay=not args.no_replay)
        results[name]["seqs"] = seqs
    cmp_ = compare(results, control)
    payload = {"arms": results, "comparison": cmp_,
               "verdict": verdict(cmp_["table"], control)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for name, row in cmp_["table"].items():
        logger.info("%-14s op=%s recall %.4f (d %+0.4f) iou %.3f foot %.4f replay %.4f", name,
                    row["matched_op_conf"], row["A_recall_at_matched"], row["dA_recall_matched"],
                    row["B_iou_median_at_matched"], row["B_foot_median_at_matched"],
                    row.get("C_replay_recall", float("nan")))
    logger.info("verdict %s", payload["verdict"])
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
