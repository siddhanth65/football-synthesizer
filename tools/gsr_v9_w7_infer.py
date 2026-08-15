"""Standalone low-confidence detector dump for the v9-W7 miss anatomy (server-side, no repo imports).

The shipped chain runs the football YOLO at ``conf=BALL_CONF`` (0.10) and then keeps only person
boxes at ``conf >= DETECT_CONF`` (0.20) before handing them to ByteTrack. Two bands are therefore
invisible to the cached artifacts:

* ``[0.10, 0.20)`` -- already produced by the shipped inference call and thrown away in Python, so
  raising the yield here costs **zero** extra GPU;
* ``[conf_floor, 0.10)`` -- needs the model call itself to run lower.

This script dumps every person box above ``--conf`` for a list of sequences so the anatomy can ask
whether a "detection miss" is a missing box or only a missing *confident* box.

Deliberately dependency-light (ultralytics + pandas + numpy) and import-free of this repo: it is
copied to the cluster and run there, where the repo checkout may be stale.

Usage (cluster)::

    python gsr_v9_w7_infer.py --weights ~/runs/det/gsr_v3_ft_b/weights/last.pt \\
        --data ~/data/gamestate-2024 --seqs SNGS-021,... --out ~/work/v9w7/lowconf --conf 0.01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

#: Canonical role names of the S4b detector's four classes, by class id.
ROLE_BY_ID = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def dump_sequence(model, seq_dir: Path, out_path: Path, *, conf: float, imgsz: int, iou: float,
                  max_det: int, batch: int, channels: str = "bgr") -> dict:
    """Run the detector over one sequence's frames and write every box above ``conf``.

    Args:
        model: A constructed ultralytics ``YOLO``.
        seq_dir: Sequence folder holding ``img1/%06d.jpg``.
        out_path: Destination parquet.
        conf: Confidence floor of the dump.
        imgsz: Inference long side (the pipeline uses 640).
        iou: NMS IoU (the pipeline uses ultralytics' 0.7 default).
        max_det: Max detections per image.
        batch: Frames per forward pass.
        channels: ``'bgr'`` feeds ultralytics the file paths (it decodes BGR, the convention its
            numpy input path documents); ``'rgb'`` decodes with OpenCV and swaps to RGB first,
            which is literally what ``generator.extract.extract_positions`` hands the detector.

    Returns:
        ``{"frames": n, "rows": n, "seconds": s}``.
    """
    import cv2

    frames = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames under {seq_dir / 'img1'}")
    t0 = time.time()
    recs: list[dict] = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        source = ([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in chunk]
                  if channels == "rgb" else [str(p) for p in chunk])
        results = model.predict(source, verbose=False, conf=conf, imgsz=imgsz,
                                iou=iou, max_det=max_det)
        for path, r in zip(chunk, results):
            # frames are img1/%06d.jpg, 1-based; the pipeline's parquet frame index is 0-based
            fidx = int(path.stem) - 1
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.int().cpu().numpy()
            for b, c, k in zip(xyxy, cf, cls):
                recs.append({"frame": fidx, "image_x": float((b[0] + b[2]) / 2),
                             "image_y": float(b[3]), "x1": float(b[0]), "y1": float(b[1]),
                             "x2": float(b[2]), "y2": float(b[3]), "conf": float(c),
                             "cls": int(k), "role": ROLE_BY_ID.get(int(k), "?")})
    df = pd.DataFrame.from_records(recs) if recs else pd.DataFrame(
        columns=["frame", "image_x", "image_y", "x1", "y1", "x2", "y2", "conf", "cls", "role"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {"frames": len(frames), "rows": int(len(df)), "seconds": round(time.time() - t0, 1)}


def demo() -> None:
    """Assert the frame-index convention and the role map (pure, no GPU)."""
    assert int(Path("img1/000001.jpg").stem) - 1 == 0
    assert int(Path("img1/000750.jpg").stem) - 1 == 749
    assert ROLE_BY_ID[2] == "player"
    b = np.array([10.0, 20.0, 30.0, 60.0])
    assert ((b[0] + b[2]) / 2, b[3]) == (20.0, 60.0)
    print("gsr_v9_w7_infer demo: OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="1")
    ap.add_argument("--channels", choices=("bgr", "rgb"), default="bgr",
                    help="'rgb' reproduces generator.extract's numpy input exactly")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    import hashlib

    from ultralytics import YOLO

    md5 = hashlib.md5(args.weights.read_bytes()).hexdigest()  # noqa: S324 - provenance
    print(f"weights {args.weights} md5 {md5}", flush=True)
    model = YOLO(str(args.weights))
    model.to(f"cuda:{args.device}")
    seqs = [s for s in args.seqs.split(",") if s]
    for i, s in enumerate(seqs):
        dest = args.out / f"{s}.parquet"
        if dest.exists():
            print(f"[{i + 1}/{len(seqs)}] {s}: cached", flush=True)
            continue
        st = dump_sequence(model, args.data / s, dest, conf=args.conf, imgsz=args.imgsz,
                           iou=args.iou, max_det=args.max_det, batch=args.batch,
                           channels=args.channels)
        print(f"[{i + 1}/{len(seqs)}] {s}: {st}", flush=True)


if __name__ == "__main__":
    main()
