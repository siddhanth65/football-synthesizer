"""Honest accuracy of the fine-tuned ball detector against hand-clicked ground truth.

The possession pipeline reports "ball detected in X% of frames", but *detected* is not *correct* -- a
confident peak on the wrong blob still counts. This harness closes that gap: against the click
annotations (``outputs/ball_annotations/chunk_*.csv`` -- ``frame, x, y, visible``) it reports, per chunk,

* **precision / recall / F1** -- treating a detection within ``tol_px`` of the clicked ball as a hit
  (visible frames only for recall; invisible frames catch false positives),
* **localization error** -- median / 90th-percentile pixel distance on hits, and the same in **pitch
  metres** (project both the detection and the GT click through the frame homography) -- the number that
  actually matters for possession (a 2 m radius decides the carrier).

Run a train/holdout split (e.g. fine-tuned on 000+001, evaluate on 006) to see real generalisation, not
just training-set recall. Pure scoring (:func:`score_detections`) is unit-tested; the runner needs the
video + weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TOL_PX = 12.0  # a detection within this pixel distance of the clicked ball is a hit


def score_detections(gt: pd.DataFrame, det: dict[int, tuple[float, float]], *,
                     tol_px: float = DEFAULT_TOL_PX) -> dict:
    """Score per-frame ball detections against click ground truth.

    Args:
        gt: ``frame, x, y, visible`` -- one row per annotated frame (``visible`` in {0,1}).
        det: ``{frame: (x, y)}`` image-px detections (<=1 per frame, as the peak detector emits).
        tol_px: hit threshold in pixels.

    Returns:
        ``{n_visible, n_invisible, tp, fp, fn, tn, precision, recall, f1, med_err_px, p90_err_px,
        errors_px}`` where ``errors_px`` is the list of hit distances (for downstream metre conversion).
    """
    vis = {int(r.frame): (float(r.x), float(r.y)) for r in gt.itertuples(index=False) if int(r.visible) == 1}
    invis = {int(r.frame) for r in gt.itertuples(index=False) if int(r.visible) == 0}
    tp = fp = fn = tn = 0
    errs: list[float] = []
    for fr, (gx, gy) in vis.items():
        if fr in det:
            dx, dy = det[fr]
            d = float(np.hypot(dx - gx, dy - gy))
            if d <= tol_px:
                tp += 1
                errs.append(d)
            else:
                fp += 1  # fired in the wrong place
                fn += 1  # and missed the real ball
        else:
            fn += 1
    for fr in invis:
        if fr in det:
            fp += 1  # ball not visible but detector fired
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision) and not np.isnan(recall) else float("nan"))
    return {"n_visible": len(vis), "n_invisible": len(invis), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "med_err_px": float(np.median(errs)) if errs else float("nan"),
            "p90_err_px": float(np.percentile(errs, 90)) if errs else float("nan"),
            "errors_px": errs}


def metre_errors(gt: pd.DataFrame, det: dict[int, tuple[float, float]], players: pd.DataFrame, *,
                 tol_px: float = DEFAULT_TOL_PX) -> dict:
    """Localization error in **pitch metres** on hits, via the per-frame player-correspondence homography.

    Projects both the clicked ball and the detection through the same homography and measures their
    distance; this is the error that actually moves possession (carrier radius is ~2 m).
    """
    import cv2  # noqa: PLC0415

    vis = {int(r.frame): (float(r.x), float(r.y)) for r in gt.itertuples(index=False) if int(r.visible) == 1}
    errs_m = []
    for fr, (gx, gy) in vis.items():
        if fr not in det:
            continue
        dx, dy = det[fr]
        if float(np.hypot(dx - gx, dy - gy)) > tol_px:
            continue
        g = players[players["frame"] == fr].dropna(subset=["image_x", "image_y", "pitch_x", "pitch_y"])
        if len(g) < 6:
            continue
        h, _ = cv2.findHomography(g[["image_x", "image_y"]].to_numpy(np.float32),
                                  g[["pitch_x", "pitch_y"]].to_numpy(np.float32), cv2.RANSAC, 12.0)
        if h is None:
            continue
        pts = cv2.perspectiveTransform(np.array([[[gx, gy], [dx, dy]]], np.float32), h)[0]
        errs_m.append(float(np.hypot(pts[0, 0] - pts[1, 0], pts[0, 1] - pts[1, 1])))
    return {"n_metre_hits": len(errs_m),
            "med_err_m": float(np.median(errs_m)) if errs_m else float("nan"),
            "p90_err_m": float(np.percentile(errs_m, 90)) if errs_m else float("nan")}


def evaluate_chunk(video: str, annotations: str, positions: str, weights: str, *,
                   base: str = "tracknetv2", thr: float = 0.5, tol_px: float = DEFAULT_TOL_PX) -> dict:
    """Detect on the annotated frames of one chunk and score vs the clicks (px + metres)."""
    from tools.ball_possession import _load_model, detect_ball_imagexy  # noqa: PLC0415

    gt = pd.read_csv(annotations)
    players = pd.read_parquet(positions)
    model, dev = _load_model(weights, base=base)
    frames = sorted(int(f) for f in gt["frame"].unique())
    det = detect_ball_imagexy(video, frames, model, dev, thr=thr)
    res = score_detections(gt, det, tol_px=tol_px)
    res.update(metre_errors(gt, det, players, tol_px=tol_px))
    res.pop("errors_px", None)
    return res


def _fmt(name: str, r: dict) -> str:
    return (f"  {name:<12} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}  "
            f"| loc {r['med_err_px']:.1f}px (p90 {r['p90_err_px']:.1f})  "
            f"{r['med_err_m']:.2f}m (p90 {r['p90_err_m']:.2f})  "
            f"| vis={r['n_visible']} tp={r['tp']} fp={r['fp']} fn={r['fn']}")


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="outputs/ball_finetuned/tracknetv2_v2.pth")
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tol-px", type=float, default=DEFAULT_TOL_PX)
    ap.add_argument("--chunks-dir",
                    default="C:/Users/siddh_ygv5bws/OneDrive/Desktop/cv-football/chunks")
    ap.add_argument("--chunks", nargs="+", default=["chunk_000", "chunk_001", "chunk_006"])
    ap.add_argument("--holdout", default=None, help="chunk name to label as the holdout in the report")
    args = ap.parse_args()
    print(f"Ball detector eval — weights={args.weights}  tol={args.tol_px:.0f}px  thr={args.thr}\n")
    for ck in args.chunks:
        video = f"{args.chunks_dir}/{ck}.mp4"
        ann = f"outputs/ball_annotations/{ck}.csv"
        pos = f"outputs/{ck.replace('chunk_', 'chunk')}_dense.parquet"
        if not (Path(ann).exists() and Path(pos).exists()):
            print(f"  {ck}: missing annotations or positions — skipped")
            continue
        r = evaluate_chunk(video, ann, pos, args.weights, base=args.base, thr=args.thr, tol_px=args.tol_px)
        tag = ck + (" [HOLDOUT]" if ck == args.holdout else "")
        print(_fmt(tag, r))


if __name__ == "__main__":
    main()
