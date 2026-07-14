"""Validate a fine-tuned ball detector: peak-detection accuracy vs annotations + temporal detection rate.

Mirrors ``tools/finetune_ball.py``'s model build (TrackNetV2 via WASB's factory) and runs inference on
(a) the annotated frames — measuring how often the heatmap peak lands on the labelled ball (recall) and
how often it fires when no ball is visible (false-positive rate); and (b) a contiguous live window of
unlabelled frames — measuring the raw per-frame detection rate as a generalisation signal.

NOTE: by default the annotated frames are the *training* frames, so (a) is a fit check, not held-out
accuracy. Combined with (b) on unseen frames it tells us whether the domain adaptation took. The
pretrained baseline managed ~15% even on in-distribution frames, so a large jump here is real evidence.

Run:
    python tools/validate_ball.py --annotations outputs/ball_annotations/chunk_000.csv \
        --video "<...>/chunk_000.mp4" --weights outputs/ball_finetuned/tracknetv2_ours.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_WASB_SRC = os.environ.get("FOOTBALL_WASB_PATH", str(Path.home() / "WASB-SBDT" / "src"))
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
INP_W, INP_H, FRAMES_IN = 512, 288, 3


def _window(cap, fr):
    """Stack frames ``fr-2,fr-1,fr`` as a 9-ch ImageNet-normalised input at INP_W x INP_H."""
    import cv2
    frames = []
    for f in (fr - 2, fr - 1, fr):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0))
        ok, bgr = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(cv2.resize(bgr, (INP_W, INP_H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
    return np.concatenate(frames, 0)


def _peak(hm: np.ndarray, thr: float):
    """Argmax peak of a heatmap; returns ((x,y), score) or (None, score) if below threshold."""
    y, x = np.unravel_index(int(hm.argmax()), hm.shape)
    s = float(hm[y, x])
    return ((float(x), float(y)), s) if s > thr else (None, s)


def _load_model(weights: str, base: str = "tracknetv2"):
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, _WASB_SRC)
    from models import build_model  # noqa: E402
    cfg = {"model": OmegaConf.load(os.path.join(_WASB_SRC, "configs", "model", f"{base}.yaml"))}
    model = build_model(cfg)
    ck = torch.load(weights, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), dev


def evaluate_annotated(model, dev, annotations: str, video: str, *, thr: float, tol_px: float,
                       sample: int | None, include_frames: set[int] | None = None):
    """Recall on visible-ball frames (peak within ``tol_px`` of label) + FP rate on not-visible frames.

    With ``include_frames`` only those frames are scored (used for held-out validation, so training
    frames never leak into the metric). ``hits`` and the raw ``dists`` list are returned alongside the
    per-chunk summary so a caller can aggregate recall/median-error across multiple chunks.
    """
    import cv2
    import torch
    df = pd.read_csv(annotations)
    if include_frames is not None:
        df = df[df["frame"].isin(include_frames)]
    if sample:
        df = df.iloc[:: max(1, len(df) // sample)]
    cap = cv2.VideoCapture(video)
    wo, ho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = INP_W / wo, INP_H / ho
    hits = vis = 0
    dists = []
    fp = neg = 0
    for r in df.itertuples(index=False):
        win = _window(cap, int(r.frame))
        if win is None:
            continue
        with torch.no_grad():
            hm = model(torch.from_numpy(win[None]).to(dev))[0].sigmoid()[0, -1].cpu().numpy()
        xy, score = _peak(hm, thr)
        if int(r.visible) == 1:
            vis += 1
            if xy is not None:
                d = float(np.hypot(xy[0] - r.x * sx, xy[1] - r.y * sy))
                dists.append(d)
                if d <= tol_px:
                    hits += 1
        else:
            neg += 1
            if xy is not None:
                fp += 1
    cap.release()
    return {"visible_frames": vis, "hits": hits, "dists": dists,
            "recall": hits / vis if vis else 0.0,
            "median_err_px": float(np.median(dists)) if dists else None,
            "neg_frames": neg, "fp_rate": fp / neg if neg else 0.0}


def evaluate_window(model, dev, video: str, start: int, n: int, *, thr: float):
    """Raw per-frame detection rate over a contiguous unlabelled live window (generalisation signal)."""
    from collections import deque

    import cv2
    import torch
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(start - (FRAMES_IN - 1), 0))
    buf: deque = deque(maxlen=FRAMES_IN)
    fired = total = 0
    fr = max(start - (FRAMES_IN - 1), 0)
    while fr < start + n:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(bgr, (INP_W, INP_H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        buf.append(((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
        if len(buf) == FRAMES_IN and fr >= start:
            with torch.no_grad():
                win = np.concatenate(list(buf), 0)
                hm = model(torch.from_numpy(win[None]).to(dev))[0].sigmoid()[0, -1].cpu().numpy()
            total += 1
            if float(hm.max()) > thr:
                fired += 1
        fr += 1
    cap.release()
    return {"window_frames": total, "detection_rate": fired / total if total else 0.0}


def evaluate_manifest_holdout(model, dev, manifest: str, *, thr: float, tol_px: float, stride: int):
    """Per-match held-out recall + median localisation error over every manifest chunk.

    For each chunk only the held-out frames (:func:`holdout_frames`, the exact rows the trainer
    excluded) are scored, then hits and errors are pooled by match so the reported recall is honest
    out-of-sample accuracy. Returns ``{match: {visible, hits, recall, median_err_px}}``.
    """
    from tools.finetune_ball import holdout_frames, read_manifest  # noqa: PLC0415

    agg: dict[str, dict] = {}
    for r in read_manifest(manifest):
        hf = holdout_frames(r["annotations"], stride)
        if not hf:
            continue
        res = evaluate_annotated(model, dev, r["annotations"], r["video"], thr=thr, tol_px=tol_px,
                                 sample=None, include_frames=hf)
        m = agg.setdefault(r["match"], {"visible": 0, "hits": 0, "dists": []})
        m["visible"] += res["visible_frames"]
        m["hits"] += res["hits"]
        m["dists"] += res["dists"]
        print(f"  {r['match']:14s} {r['chunk']:14s} held-out visible={res['visible_frames']:3d}"
              f"  recall={res['recall']:.1%}  median_err={res['median_err_px']}px")
    out = {}
    for match, m in agg.items():
        out[match] = {"visible": m["visible"], "hits": m["hits"],
                      "recall": m["hits"] / m["visible"] if m["visible"] else 0.0,
                      "median_err_px": float(np.median(m["dists"])) if m["dists"] else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=None,
                    help="corpus manifest -> per-match HELD-OUT recall/error (overrides single-chunk mode)")
    ap.add_argument("--holdout-stride", type=int, default=5, dest="holdout_stride",
                    help="must match the finetune stride so validation uses the same reserved rows")
    ap.add_argument("--annotations")
    ap.add_argument("--video")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tol-px", type=float, default=8.0, help="hit tolerance in the 512x288 space")
    ap.add_argument("--sample", type=int, default=120, help="subsample annotations for speed (0=all)")
    ap.add_argument("--win-start", type=int, default=20620)
    ap.add_argument("--win-n", type=int, default=150)
    args = ap.parse_args()

    model, dev = _load_model(args.weights, args.base)
    print(f"loaded {args.base} on {dev}; thr={args.thr} tol={args.tol_px}px")

    if args.manifest:
        print(f"[held-out per chunk] stride={args.holdout_stride}")
        per_match = evaluate_manifest_holdout(model, dev, args.manifest, thr=args.thr,
                                              tol_px=args.tol_px, stride=args.holdout_stride)
        print("\n[per-match held-out]")
        for match, m in sorted(per_match.items()):
            print(f"  {match:14s} visible={m['visible']:3d}  recall={m['recall']:.1%}"
                  f"  median_err={m['median_err_px']}px")
        return

    if not (args.annotations and args.video):
        ap.error("single-chunk mode needs --annotations and --video (or pass --manifest)")
    a = evaluate_annotated(model, dev, args.annotations, args.video, thr=args.thr, tol_px=args.tol_px,
                           sample=args.sample or None)
    print(f"\n[annotated frames]  visible={a['visible_frames']}  recall={a['recall']:.1%}"
          f"  median_err={a['median_err_px']}px  |  neg={a['neg_frames']}  fp_rate={a['fp_rate']:.1%}")
    w = evaluate_window(model, dev, args.video, args.win_start, args.win_n, thr=args.thr)
    print(f"[live window @ {args.win_start}]  frames={w['window_frames']}  "
          f"detection_rate={w['detection_rate']:.1%}")


if __name__ == "__main__":
    main()
