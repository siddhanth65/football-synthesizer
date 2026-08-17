"""Standalone RF-DETR box dump for the v10-W12 detector kill experiment (server-side, no repo imports).

Writes the **same parquet schema** as :mod:`tools.gsr_v9_w7_infer` (the S4b/YOLO dump), so both
detectors can be compared with one analysis tool and replayed through the same CPU ByteTrack
instrument:

    frame, image_x, image_y, x1, y1, x2, y2, conf, cls, role

``frame`` is 0-based (``img1/%06d.jpg`` minus one), ``image_x/image_y`` is the bottom-middle foot
point -- the quantity the homography consumes.

Deliberately dependency-light (rfdetr + pandas + numpy + opencv) and import-free of this repo: it is
copied to the cluster and run there.

Usage (cluster)::

    python gsr_v10_w12_infer.py --probe --weights ~/models/rfdetr_soccernet/checkpoint_best_regular.pth
    python gsr_v10_w12_infer.py --weights <ckpt> --data ~/data/gamestate-2024 \\
        --seqs SNGS-021,... --out ~/work/v10w12/rfdetr_soccer --conf 0.01 --resolution 1288
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

#: Class ids of the SoccerNet RF-DETR fine-tune (config.json of julianzu9612/RFDETR-Soccernet).
#: rfdetr emits 1-based class ids for COCO-style checkpoints, so the map is verified by --probe
#: before any dump is trusted (a player class that is not ~20 boxes/frame fails the check).
ROLE_BY_ID = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}


def build_model(weights: Path, resolution: int | None, size: str = "large_deprecated"):
    """Construct an RF-DETR of the checkpoint's own generation (lazy import).

    The 2026 ``rfdetr`` package's ``RFDETRLarge`` is a DIFFERENT network from the 2025 one the
    SoccerNet fine-tune was trained with (patch 16 / dinov2_windowed_small / res 704 vs patch 14 /
    dinov2_windowed_base / hidden 384 / res 560) and the loader refuses to cross that gap. The 2025
    architecture ships as ``RFDETRLargeDeprecated``, whose defaults match the checkpoint's own
    training args exactly, so only ``pretrain_weights`` (and an optional ``resolution`` override,
    to price the resolution axis) has to be passed.
    """
    import rfdetr  # noqa: PLC0415

    cls = {"large_deprecated": rfdetr.RFDETRLargeDeprecated, "large": rfdetr.RFDETRLarge,
           "base": rfdetr.RFDETRBase}[size]
    kw: dict = {"pretrain_weights": str(weights)}
    if resolution:
        kw["resolution"] = int(resolution)
    return cls(**kw)


def probe_checkpoint(weights: Path) -> dict:
    """Report what a checkpoint actually contains, so the model class is read and not guessed."""
    import torch  # noqa: PLC0415

    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    keys = list(ck.keys()) if isinstance(ck, dict) else []
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    n_par = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    head = [k for k in sd if "class_embed" in k][:4]
    shapes = {k: tuple(sd[k].shape) for k in head}
    pos = {k: tuple(sd[k].shape) for k in sd if "pos_embed" in k or "position" in k}
    out = {"top_keys": keys[:8], "params": int(n_par), "class_head": shapes,
           "pos_embed": dict(list(pos.items())[:4]), "args": str(ck.get("args"))[:600]
           if isinstance(ck, dict) else ""}
    print(out)
    return out


def dump_sequence(model, seq_dir: Path, out_path: Path, *, conf: float, batch: int) -> dict:
    """Run RF-DETR over one sequence's frames and write every box above ``conf``."""
    import cv2  # noqa: PLC0415

    frames = sorted((seq_dir / "img1").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames under {seq_dir / 'img1'}")
    t0 = time.time()
    recs: list[dict] = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in chunk]
        dets = model.predict(imgs, threshold=conf)
        if not isinstance(dets, list):
            dets = [dets]
        for path, d in zip(chunk, dets):
            fidx = int(path.stem) - 1
            if d.xyxy is None or len(d) == 0:
                continue
            cid = np.asarray(d.class_id)
            cf = (np.asarray(d.confidence) if d.confidence is not None
                  else np.ones(len(d), float))
            for b, c, k in zip(np.asarray(d.xyxy, float), cf, cid):
                recs.append({"frame": fidx, "image_x": float((b[0] + b[2]) / 2),
                             "image_y": float(b[3]), "x1": float(b[0]), "y1": float(b[1]),
                             "x2": float(b[2]), "y2": float(b[3]), "conf": float(c),
                             "cls": int(k), "role": ROLE_BY_ID.get(int(k), "?")})
    cols = ["frame", "image_x", "image_y", "x1", "y1", "x2", "y2", "conf", "cls", "role"]
    df = pd.DataFrame.from_records(recs) if recs else pd.DataFrame(columns=cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    per_role = df["role"].value_counts().to_dict() if len(df) else {}
    return {"frames": len(frames), "rows": int(len(df)), "seconds": round(time.time() - t0, 1),
            "per_role": per_role}


def demo() -> None:
    """Assert the schema conventions shared with the YOLO dump (pure, no GPU)."""
    assert int(Path("img1/000001.jpg").stem) - 1 == 0
    assert ROLE_BY_ID[1] == "player"
    b = np.array([10.0, 20.0, 30.0, 60.0])
    assert ((b[0] + b[2]) / 2, b[3]) == (20.0, 60.0)
    print("gsr_v10_w12_infer demo: OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--resolution", type=int, default=0)
    ap.add_argument("--size", choices=("base", "large", "large_deprecated"),
                    default="large_deprecated")
    ap.add_argument("--rolemap", default="", help="cls:role,... overriding ROLE_BY_ID")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="1")
    ap.add_argument("--probe", action="store_true", help="report checkpoint structure and exit")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if args.probe:
        probe_checkpoint(args.weights)
        return
    import hashlib
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    md5 = hashlib.md5(args.weights.read_bytes()).hexdigest()  # noqa: S324 - provenance
    print(f"weights {args.weights} md5 {md5}", flush=True)
    if args.rolemap:
        ROLE_BY_ID.clear()
        ROLE_BY_ID.update({int(k): v for k, v in (p.split(":") for p in args.rolemap.split(","))})
    print(f"rolemap {ROLE_BY_ID}", flush=True)
    model = build_model(args.weights, args.resolution or None, size=args.size)
    seqs = [s for s in args.seqs.split(",") if s]
    for i, s in enumerate(seqs):
        dest = args.out / f"{s}.parquet"
        if dest.exists():
            print(f"[{i + 1}/{len(seqs)}] {s}: cached", flush=True)
            continue
        st = dump_sequence(model, args.data / s, dest, conf=args.conf, batch=args.batch)
        print(f"[{i + 1}/{len(seqs)}] {s}: {st}", flush=True)


if __name__ == "__main__":
    main()
