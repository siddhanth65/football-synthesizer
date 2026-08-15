"""v9 W7: the detection-miss bucket -- anatomy first, detector arms only if the anatomy asks for it.

W3 section 2 measured 20,075 live-frame detection misses on DEV-20 (8.54% of GT rows) and priced a
perfect detector at +6.99 GS-DetA (oracle) / +2..4.4 realistic. It did **not** ask where in the
chain the miss happens. A census "miss" is "no submission row within 5 m of this GT row", and the
submission is the end of a long chain:

    detector (DETECT_CONF 0.20) -> ByteTrack -> calibration re-gate -> EIoU re-association
    -> connector/solver -> writer

so a census miss can be any of:

* **detector-level** -- the detector emitted no box for that person at any confidence;
* **threshold-level** -- a box exists below ``DETECT_CONF``; a threshold change is nearly free;
* **downstream** -- a box exists at >= 0.20 in ``positions_*`` and is lost after it (calibration
  NaN, tracker, association, writer).

Only the first two are a detector-retrain lever. Stage 0 splits them before any GPU is booked.

Stages::

    python -m tools.gsr_v9_w7 --stage anatomy      # CPU, cached artifacts only
    python -m tools.gsr_v9_w7 --stage lowconf --lowconf-dir <dir>   # fold in a low-conf probe
    python -m tools.gsr_v9_w7 --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tools.gsr_v9_deta import (
    DEFAULT_ARM,
    box_iou,
    effective_attrs,
    load_gt,
    load_pred,
    match_positions,
    similarity,
)

logger = logging.getLogger("gsr_v9_w7")

#: Raw (pre-gate) extraction of the shipped S4b detector on DEV-20.
RAW_POSITIONS = Path("outputs/gsr/positions_v6det")
#: Post calibration re-gate + gap fill (the association stage's input).
GATE_POSITIONS = Path("outputs/gsr/positions_gate_v6det")
#: Image-space match gate: foot-point distance in units of the GT box height.
FOOT_GATE = 0.5
#: Frame-edge margin (px) used for the truncation split.
EDGE_PX = 12.0


# === pure helpers =================================================================================
def gt_foot(box: dict) -> tuple[float, float]:
    """Bottom-middle image point of a GSR ``bbox_image`` (pure)."""
    return box["x"] + box["w"] / 2.0, box["y"] + box["h"]


def match_feet(gt_boxes: list[dict], det_xy: np.ndarray, gate: float = FOOT_GATE) -> dict[int, int]:
    """One-to-one image-space match of GT boxes to detector foot points (pure).

    Cost is the foot-point distance divided by the GT box height, so the gate is scale free: a pair
    is kept only when the detector's foot point is within ``gate`` box-heights of the GT foot point.

    Args:
        gt_boxes: GT ``bbox_image`` dicts for one frame.
        det_xy: ``(M, 2)`` detector foot points for the same frame.
        gate: Acceptance threshold in box-height units.

    Returns:
        ``{gt index: detection index}`` for the accepted pairs.
    """
    if not gt_boxes or len(det_xy) == 0:
        return {}
    g = np.array([gt_foot(b) for b in gt_boxes], float)
    h = np.array([max(float(b["h"]), 1.0) for b in gt_boxes], float)
    cost = np.linalg.norm(g[:, None, :] - det_xy[None, :, :], axis=-1) / h[:, None]
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] <= gate}


def edge_flags(box: dict, w: float = 1920.0, h: float = 1080.0,
               margin: float = EDGE_PX) -> tuple[bool, bool]:
    """``(touches a side edge, touches the bottom edge)`` for a GT box (pure)."""
    side = box["x"] <= margin or box["x"] + box["w"] >= w - margin
    bottom = box["y"] + box["h"] >= h - margin
    return bool(side), bool(box["y"] <= margin or bottom)


def _edge_tag(side: bool, vert: bool) -> str:
    """Name a GT box's frame-edge contact (pure)."""
    return ("side" if side else "") + ("vert" if vert else "") or "interior"


def _bucket(value: float, edges: tuple[float, ...]) -> str:
    """Name the half-open bin ``value`` falls in (pure)."""
    for lo, hi in zip((0.0, *edges), (*edges, float("inf"))):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}" if hi != float("inf") else f">={lo:g}"
    return "?"


# === per-sequence anatomy =========================================================================
def _det_by_frame(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Group a positions parquet's foot points by frame (people only)."""
    ppl = df[df["role"].isin(["player", "goalkeeper", "referee"])]
    return {int(f): g[["image_x", "image_y"]].to_numpy(float)
            for f, g in ppl.groupby("frame")}


def _gt_velocity(gt_rows: list[list[dict]]) -> dict[tuple[int, int], float]:
    """Per-(timestep, track) GT foot-point speed in px/frame, a motion-blur proxy (pure)."""
    seen: dict[int, tuple[int, tuple[float, float]]] = {}
    out: dict[tuple[int, int], float] = {}
    for t, rows in enumerate(gt_rows):
        for r in rows:
            tid = int(r["track_id"])
            xy = gt_foot(r["bbox_image"])
            if tid in seen:
                t0, xy0 = seen[tid]
                dt = max(t - t0, 1)
                out[(t, tid)] = float(np.hypot(xy[0] - xy0[0], xy[1] - xy0[1]) / dt)
            seen[tid] = (t, xy)
    return out


def anatomy_sequence(seq_dir: Path, pred_path: Path, raw: Path, gate: Path,
                     lowconf: Path | None = None) -> dict:
    """Split one sequence's detection misses by where in the chain they happen, and by GT shape.

    Args:
        seq_dir: GSR sequence folder (``Labels-GameState.json`` + ``img1``).
        pred_path: The submission JSON for this sequence.
        raw: Directory of pre-gate positions parquets.
        gate: Directory of post-gate positions parquets.
        lowconf: Optional directory of low-confidence probe parquets (``frame, image_x, image_y,
            conf``); when given, misses with no >= 0.20 box are further split by whether a
            sub-threshold box exists.

    Returns:
        Counters and per-split tables for this sequence.
    """
    ts, gt_rows = load_gt(seq_dir)
    pr_rows = load_pred(pred_path, ts, len(gt_rows))
    name = seq_dir.name
    raw_df = pd.read_parquet(raw / f"{name}.parquet")
    gate_df = pd.read_parquet(gate / f"{name}.parquet")
    raw_by_frame = _det_by_frame(raw_df)
    gate_by_frame = _det_by_frame(gate_df[gate_df["pitch_x"].notna()])
    low_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if lowconf is not None:
        low_df = pd.read_parquet(lowconf / f"{name}.parquet")
        low_df = low_df[low_df["role"].isin(["player", "goalkeeper", "referee"])]
        low_by_frame = {int(f): (g[["image_x", "image_y"]].to_numpy(float),
                                 g["conf"].to_numpy(float))
                        for f, g in low_df.groupby("frame")}
    vel = _gt_velocity(gt_rows)
    # positions parquets index frames 0-based over img1/%06d.jpg, so frame = image number - 1; GT
    # timesteps skip unlabelled images, so the mapping is read off the image ids, never assumed.
    frame_of_ts = {k: int(img_id[-6:]) - 1 for img_id, k in ts.items()}

    c: Counter = Counter()
    splits: dict[str, Counter] = {k: Counter() for k in
                                  ("height", "crowd", "edge", "speed", "profile")}
    for t, (gts, prs) in enumerate(zip(gt_rows, pr_rows)):
        sim = similarity(_pitch_xy(gts), _pitch_xy(prs))
        hit = {i for i, _j in match_positions(sim)}
        frame = frame_of_ts[t]
        boxes = [g["bbox_image"] for g in gts]
        raw_hit = match_feet(boxes, raw_by_frame.get(frame, np.zeros((0, 2))))
        gate_hit = match_feet(boxes, gate_by_frame.get(frame, np.zeros((0, 2))))
        pxy, pconf = low_by_frame.get(frame, (np.zeros((0, 2)), np.zeros(0)))
        probe_hit = match_feet(boxes, pxy) if len(pxy) else {}
        dead = len(prs) == 0
        for i, g in enumerate(gts):
            c["gt"] += 1
            c["raw_detected"] += i in raw_hit
            c["probe_detected"] += i in probe_hit
            pc = float(pconf[probe_hit[i]]) if i in probe_hit else 0.0
            if i in hit:
                tag = "hit"
                c["hit"] += 1
            elif i in raw_hit:
                tag = "downstream"
                c["lost_downstream"] += 1
                c["lost_after_gate" if i in gate_hit else "lost_at_gate"] += 1
                c["lost_downstream_deadframe"] += dead
            elif pc >= 0.20:
                # a confident box the shipped extraction never wrote: tracker / extract-loop loss
                tag = "tracker"
                c["miss_tracker_level"] += 1
            elif pc >= 0.10:
                tag = "conf_free"       # inside the model's existing conf=0.10 call: free to recover
                c["miss_conf_free"] += 1
            elif pc > 0.0:
                tag = "conf_deep"       # below the model call's own floor
                c["miss_conf_deep"] += 1
            else:
                tag = "detector"
                c["miss_detector_level"] += 1
            if tag != "hit":
                c["miss_deadframe" if dead else "miss_live"] += 1
                role, _team, _j = effective_attrs(g["attributes"].get("role"),
                                                  g["attributes"].get("team"),
                                                  g["attributes"].get("jersey"))
                c[f"missrole::{tag}::{role}"] += 1
            box = g["bbox_image"]
            other = max((box_iou(box, o["bbox_image"]) for k, o in enumerate(gts) if k != i),
                        default=0.0)
            side, vert = edge_flags(box)
            speed = vel.get((t, int(g["track_id"])))
            splits["height"][(tag, _bucket(float(box["h"]), (40, 50, 60, 80, 110, 141)))] += 1
            splits["crowd"][(tag, _bucket(other, (0.05, 0.3, 0.5)))] += 1
            splits["edge"][(tag, _edge_tag(side, vert))] += 1
            splits["speed"][(tag, "new" if speed is None else _bucket(speed, (2, 5, 10, 20)))] += 1
            # joint profile: which of the four measured excuses this row carries, if any
            prof = "".join(k for k, on in (("S", float(box["h"]) < 50.0), ("E", side or vert),
                                           ("C", other >= 0.3),
                                           ("M", (speed or 0.0) >= 10.0)) if on) or "none"
            splits["profile"][(tag, prof)] += 1
    c["n_frames"] = len(gt_rows)
    c["raw_rows"] = int(len(raw_df))
    c["gate_rows_valid"] = int(gate_df["pitch_x"].notna().sum())
    c["pred_rows"] = int(sum(len(p) for p in pr_rows))
    return {"counts": dict(c),
            "splits": {k: {f"{a}|{b}": n for (a, b), n in v.items()} for k, v in splits.items()}}


# === zero-GPU tracker replay ======================================================================
def replay_tracker(probe: pd.DataFrame, gt_rows: list[list[dict]], frame_of_ts: dict[int, int], *,
                   detect_conf: float, min_hits: int, lost_buffer: int = 60,
                   activation: float | None = None) -> dict:
    """Re-run ByteTrack over cached detector boxes and count how many GT rows survive it.

    The shipped extraction filters the detector at ``DETECT_CONF`` and hands the survivors to
    ``supervision.ByteTrack``; whatever the tracker does not return is never written to
    ``positions``. Because the probe parquet holds every box the detector produced, that whole stage
    can be replayed on CPU for any ``(detect_conf, min_hits)`` before a GPU-hour is booked.

    Mirrors :func:`generator.extract.extract_positions` exactly on the two points that matter: the
    tracker is only updated on frames that carry at least one detection, and the class ids handed in
    are the role ids.

    Args:
        probe: All detector boxes for one sequence (``frame, image_x, image_y, x1..y2, conf, role``).
        gt_rows: Per-timestep GT annotations.
        frame_of_ts: Timestep -> 0-based frame index.
        detect_conf: Confidence floor of the feed (shipped: 0.20).
        min_hits: ``minimum_consecutive_frames`` (shipped: 3).
        lost_buffer: ``lost_track_buffer`` (shipped: 60).
        activation: ``track_activation_threshold``; ``None`` keeps supervision's 0.25 default.

    Returns:
        ``{"gt": n, "recovered": n, "rows": n, "tracks": n}`` -- ``recovered`` counts GT rows that
        have a tracked box within :data:`FOOT_GATE` box-heights.
    """
    import supervision as sv  # noqa: PLC0415

    kw = {} if activation is None else {"track_activation_threshold": activation}
    bt = sv.ByteTrack(minimum_consecutive_frames=min_hits, lost_track_buffer=lost_buffer, **kw)
    keep = probe[probe["role"].isin(["player", "goalkeeper", "referee"])
                 & (probe["conf"] >= detect_conf)]
    by_frame = {int(f): g for f, g in keep.groupby("frame")}
    tracked: dict[int, np.ndarray] = {}
    n_rows, ids = 0, set()
    for frame in range(int(probe["frame"].max()) + 1):
        g = by_frame.get(frame)
        if g is None or not len(g):
            continue  # extract.py skips the tracker update on empty frames
        det = sv.Detections(xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(float),
                            confidence=g["conf"].to_numpy(float),
                            class_id=np.zeros(len(g), int))
        det = bt.update_with_detections(det)
        if not len(det):
            continue
        n_rows += len(det)
        if det.tracker_id is not None:
            ids.update(int(t) for t in det.tracker_id)
        tracked[frame] = np.column_stack([(det.xyxy[:, 0] + det.xyxy[:, 2]) / 2, det.xyxy[:, 3]])
    gt_n = recovered = 0
    for t, gts in enumerate(gt_rows):
        gt_n += len(gts)
        xy = tracked.get(frame_of_ts[t])
        if xy is None:
            continue
        recovered += len(match_feet([g["bbox_image"] for g in gts], xy))
    return {"gt": gt_n, "recovered": recovered, "rows": n_rows, "tracks": len(ids)}


#: Replay grid, registered in ``results/GSR_V9_W7.md`` section 2 before it was run.
REPLAY_GRID: tuple[tuple[str, dict], ...] = (
    ("shipped c0.20 m3", {"detect_conf": 0.20, "min_hits": 3}),
    ("c0.20 m1", {"detect_conf": 0.20, "min_hits": 1}),
    ("c0.10 m3", {"detect_conf": 0.10, "min_hits": 3}),
    ("c0.10 m1", {"detect_conf": 0.10, "min_hits": 1}),
    ("c0.05 m1", {"detect_conf": 0.05, "min_hits": 1}),
    ("c0.10 m1 a0.10", {"detect_conf": 0.10, "min_hits": 1, "activation": 0.10}),
    ("c0.05 m1 a0.10", {"detect_conf": 0.05, "min_hits": 1, "activation": 0.10}),
    ("c0.20 m1 a0.10", {"detect_conf": 0.20, "min_hits": 1, "activation": 0.10}),
)


def run_replay(arm_dir: Path, data_dir: Path, lowconf: Path, out: Path) -> dict:
    """Price every :data:`REPLAY_GRID` point on DEV-20 recall, on CPU."""
    seqs = sorted(p.stem for p in (arm_dir / "predictions" / "data").glob("*.json"))
    totals: dict[str, Counter] = {name: Counter() for name, _ in REPLAY_GRID}
    per_seq: dict[str, dict] = {}
    for i, s in enumerate(seqs):
        ts, gt_rows = load_gt(data_dir / s)
        frame_of_ts = {k: int(img_id[-6:]) - 1 for img_id, k in ts.items()}
        probe = pd.read_parquet(lowconf / f"{s}.parquet")
        per_seq[s] = {}
        for name, kw in REPLAY_GRID:
            r = replay_tracker(probe, gt_rows, frame_of_ts, **kw)
            per_seq[s][name] = r
            totals[name].update(r)
        logger.info("[%d/%d] %s %s", i + 1, len(seqs), s,
                    {n: per_seq[s][n]["recovered"] for n, _ in REPLAY_GRID})
    summary = {n: {**dict(c), "recall": c["recovered"] / c["gt"]} for n, c in totals.items()}
    for n, v in summary.items():
        logger.info("%-18s recall %.4f  rows %d  tracks %d", n, v["recall"], v["rows"], v["tracks"])
    payload = {"grid": {n: kw for n, kw in REPLAY_GRID}, "summary": summary, "per_seq": per_seq}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info("wrote %s", out)
    return payload


def _pitch_xy(rows: list[dict]) -> np.ndarray:
    """Bottom-middle pitch points of GSR rows as ``(N, 2)`` (pure)."""
    if not rows:
        return np.zeros((0, 2))
    return np.array([[r["bbox_pitch"]["x_bottom_middle"], r["bbox_pitch"]["y_bottom_middle"]]
                     for r in rows], float)


def run_anatomy(arm_dir: Path, data_dir: Path, out: Path, raw: Path, gate: Path,
                lowconf: Path | None = None) -> dict:
    """Run :func:`anatomy_sequence` over every sequence the arm holds and write the payload."""
    seqs = sorted(p.stem for p in (arm_dir / "predictions" / "data").glob("*.json"))
    per_seq = {}
    for i, s in enumerate(seqs):
        per_seq[s] = anatomy_sequence(data_dir / s,
                                      arm_dir / "predictions" / "data" / f"{s}.json",
                                      raw, gate, lowconf)
        cts = per_seq[s]["counts"]
        logger.info("[%d/%d] %s gt=%d miss=%d (down=%d trk=%d free=%d deep=%d det=%d)",
                    i + 1, len(seqs), s, cts["gt"],
                    cts.get("miss_live", 0) + cts.get("miss_deadframe", 0),
                    cts.get("lost_downstream", 0), cts.get("miss_tracker_level", 0),
                    cts.get("miss_conf_free", 0), cts.get("miss_conf_deep", 0),
                    cts.get("miss_detector_level", 0))
    total: Counter = Counter()
    for v in per_seq.values():
        total.update(v["counts"])
    payload = {"arm": str(arm_dir), "raw": str(raw), "lowconf": str(lowconf) if lowconf else None,
               "seqs": seqs, "total": dict(total), "per_seq": per_seq}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info("wrote %s", out)
    return payload


# === paired scoring of two extraction lineages ====================================================
#: The shipped post-processing stack (v9 W3 vote + v9 W4 keeper-side repair), applied to BOTH sides.
FLAGS_ON = {"vote": ("role", "team", "jersey"), "gk": ("team",)}
FLAGS_OFF: dict[str, tuple[str, ...]] = {"vote": (), "gk": ()}


def _prepare(src: Path, dest: Path, seqs: list[str], flags: dict) -> Counter:
    """Copy a submission into ``dest``, applying the post-processing stack; returns rows changed."""
    from eval.gsr_score import gk_side_repair, vote_track_attributes  # noqa: PLC0415

    (dest / "predictions" / "data").mkdir(parents=True, exist_ok=True)
    changed: Counter = Counter()
    for s in seqs:
        payload = json.loads((src / f"{s}.json").read_text(encoding="utf-8"))
        preds = payload["predictions"]
        changed.update(vote_track_attributes(preds, flags["vote"]))
        changed.update(gk_side_repair(preds, flags["gk"]))
        (dest / "predictions" / "data" / f"{s}.json").write_text(
            json.dumps({"predictions": preds}), encoding="utf-8")
    return changed


def score_pair(ctrl_src: Path, arm_src: Path, data_dir: Path, work: Path,
               seqs: list[str]) -> dict:
    """Score a control and an arm submission against each other, flags ON and OFF.

    Args:
        ctrl_src: ``predictions/data`` directory of the same-stack control.
        arm_src: ``predictions/data`` directory of the arm.
        data_dir: GSR ground truth root.
        work: Scratch root for the post-processed copies.
        seqs: Sequences to score (both sides must hold all of them).

    Returns:
        ``{flag state: {"control": ..., "arm": ..., "paired": ...}}``.
    """
    from scipy.stats import wilcoxon  # noqa: PLC0415

    from eval.gsr_score import EVAL_CONFIGS, gs_hota  # noqa: PLC0415

    out: dict[str, dict] = {}
    for state, flags in (("off", FLAGS_OFF), ("on", FLAGS_ON)):
        res = {}
        for side, src in (("control", ctrl_src), ("arm", arm_src)):
            dest = work / f"{state}_{side}"
            shutil.rmtree(dest, ignore_errors=True)
            changed = _prepare(src, dest, seqs, flags)
            res[side] = gs_hota(dest, data_dir, seq_info={s: 0 for s in seqs},
                                **EVAL_CONFIGS["gs_hota_full"])
            res[side]["changed"] = dict(changed)
        paired = {}
        for key in ("GS-HOTA", "GS-DetA", "GS-AssA"):
            d = np.array([res["arm"]["per_seq"][s][key] - res["control"]["per_seq"][s][key]
                          for s in seqs])
            paired[key] = {"mean": float(d.mean()), "helped": int((d > 0).sum()),
                           "hurt": int((d < 0).sum()),
                           "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        out[state] = {
            "control": {"combined": res["control"]["combined"],
                        "per_seq": res["control"]["per_seq"],
                        "changed": res["control"]["changed"]},
            "arm": {"combined": res["arm"]["combined"], "per_seq": res["arm"]["per_seq"],
                    "changed": res["arm"]["changed"]},
            "delta": {k: res["arm"]["combined"][k] - res["control"]["combined"][k]
                      for k in res["arm"]["combined"] if isinstance(res["arm"]["combined"][k],
                                                                    (int, float))},
            "paired": paired,
        }
        logger.info("flags %s: control DetA %.4f HOTA %.4f | arm DetA %.4f HOTA %.4f | "
                    "delta DetA %+.4f HOTA %+.4f | paired HOTA %+.4f %d/%d p=%.3g", state,
                    res["control"]["combined"]["GS-DetA"], res["control"]["combined"]["GS-HOTA"],
                    res["arm"]["combined"]["GS-DetA"], res["arm"]["combined"]["GS-HOTA"],
                    out[state]["delta"]["GS-DetA"], out[state]["delta"]["GS-HOTA"],
                    paired["GS-HOTA"]["mean"], paired["GS-HOTA"]["helped"],
                    paired["GS-HOTA"]["hurt"], paired["GS-HOTA"]["wilcoxon_p"])
    return out


# === self-check ===================================================================================
def demo() -> None:
    """Assert the pure seams: foot points, the scale-free match gate, edge and bucket helpers."""
    b = {"x": 100.0, "y": 200.0, "w": 40.0, "h": 100.0}
    assert gt_foot(b) == (120.0, 300.0)
    # within half a box height -> matched; beyond -> not
    assert match_feet([b], np.array([[120.0, 340.0]])) == {0: 0}
    assert match_feet([b], np.array([[120.0, 360.0]])) == {}
    assert match_feet([], np.array([[0.0, 0.0]])) == {}
    # one-to-one: two GT, one detection -> only the nearer one is claimed
    b2 = {"x": 500.0, "y": 200.0, "w": 40.0, "h": 100.0}
    assert match_feet([b, b2], np.array([[121.0, 301.0]])) == {0: 0}
    assert edge_flags({"x": 0.0, "y": 500.0, "w": 30.0, "h": 80.0}) == (True, False)
    assert edge_flags({"x": 800.0, "y": 1000.0, "w": 30.0, "h": 80.0}) == (False, True)
    assert edge_flags({"x": 800.0, "y": 500.0, "w": 30.0, "h": 80.0}) == (False, False)
    assert _bucket(35.0, (40, 50)) == "0-40"
    assert _bucket(45.0, (40, 50)) == "40-50"
    assert _bucket(90.0, (40, 50)) == ">=50"
    rows = [[{"track_id": 1, "bbox_image": {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}}],
            [{"track_id": 1, "bbox_image": {"x": 3.0, "y": 4.0, "w": 10.0, "h": 10.0}}]]
    assert abs(_gt_velocity(rows)[(1, 1)] - 5.0) < 1e-9
    print("gsr_v9_w7 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("anatomy", "replay", "pair"),
                    default="anatomy")
    ap.add_argument("--ctrl-dir", type=Path, default=None,
                    help="--stage pair: the same-stack control arm directory")
    ap.add_argument("--work", type=Path,
                    default=Path("outputs/gsr/v9_w7/pair"))
    ap.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--raw", type=Path, default=RAW_POSITIONS)
    ap.add_argument("--gate", type=Path, default=GATE_POSITIONS)
    ap.add_argument("--lowconf", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("results/gsr_benchmark/gsr_v9_w7_anatomy.json"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if args.stage == "pair":
        if args.ctrl_dir is None:
            raise SystemExit("--stage pair needs --ctrl-dir")
        ctrl = args.ctrl_dir / "predictions" / "data"
        arm = args.arm_dir / "predictions" / "data"
        seqs = sorted(p.stem for p in ctrl.glob("*.json"))
        payload = score_pair(ctrl, arm, args.data_dir, args.work, seqs)
        payload["control_dir"] = str(args.ctrl_dir)
        payload["arm_dir"] = str(args.arm_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logger.info("wrote %s", args.out)
        return
    if args.stage == "replay":
        if args.lowconf is None:
            raise SystemExit("--stage replay needs --lowconf")
        run_replay(args.arm_dir, args.data_dir, args.lowconf, args.out)
        return
    run_anatomy(args.arm_dir, args.data_dir, args.out, args.raw, args.gate, args.lowconf)


if __name__ == "__main__":
    main()
