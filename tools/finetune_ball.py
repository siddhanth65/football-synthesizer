"""Fine-tune a ball detector (TrackNetV2 by default) on annotations from THIS footage.

The pretrained soccer weights don't transfer (domain gap); this adapts them to our 1024x576 broadcast
using the frames labelled by ``tools/annotate_ball.py``. Builds 3-frame inputs + Gaussian-at-ball
heatmap targets, fine-tunes from the zoo checkpoint, and saves the adapted weights. Reuses WASB's model
factory (clone at ``~/WASB-SBDT``).

Run (after annotating):
    python tools/finetune_ball.py --annotations outputs/ball_annotations/chunk_000.csv \
        --video "<...>/chunk_000.mp4" --base tracknetv2 --epochs 30 \
        --out outputs/ball_finetuned/tracknetv2_ours.pth
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


def _gaussian_heatmap(cx, cy, h, w, sigma=4.0):
    ys, xs = np.mgrid[0:h, 0:w]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def _window(cap, fr):
    import cv2
    frames = []
    for f in (fr - 2, fr - 1, fr):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0))
        ok, bgr = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(cv2.resize(bgr, (INP_W, INP_H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
    return np.concatenate(frames, 0)  # [9, H, W]


def holdout_frames(annotations: str, stride: int = 5) -> set[int]:
    """Frames reserved for held-out validation: every ``stride``-th visible row per CSV.

    Deterministic by file row order -- the visible-ball rows are enumerated in order and every
    ``stride``-th one (1-indexed positions ``stride, 2*stride, ...``) is reserved. Non-visible rows
    are never held out (no localisation target). ``stride <= 0`` disables the split (empty set). The
    same function must gate both training-exclusion and validation-inclusion so the split can never
    leak.

    Args:
        annotations: Path to a ``frame,x,y,visible`` annotation CSV.
        stride: Reserve one in every ``stride`` visible rows.

    Returns:
        The set of held-out frame indices for this CSV.
    """
    if stride <= 0:
        return set()
    df = pd.read_csv(annotations)
    vis = df[df["visible"].astype(int) == 1]
    return {int(r.frame) for k, r in enumerate(vis.itertuples(index=False))
            if (k % stride) == (stride - 1)}


def build_dataset(annotations: str, video: str, exclude_frames: set[int] | None = None,
                  store_dtype: str = "float32"):
    """Return ``(X[n,9,H,W], Y[n,H,W])``: stacked frame windows + ball heatmap targets.

    Args:
        annotations: Path to a ``frame,x,y,visible`` annotation CSV.
        video: Matching source video (annotation frames index into it).
        exclude_frames: Held-out frames to drop from the built set (see :func:`holdout_frames`).
        store_dtype: In-RAM dtype for the stacked arrays. ``float32`` (default) preserves the exact
            v5 recipe; ``float16`` halves the resident footprint (the ImageNet-normalised windows keep
            ample precision) so the full corpus fits a 16 GB laptop without an OOM kill -- batches are
            cast back to float32 at train time so the optimisation is numerically unchanged.
    """
    import cv2
    exclude = exclude_frames or set()
    df = pd.read_csv(annotations)
    cap = cv2.VideoCapture(video)
    wo, ho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = INP_W / wo, INP_H / ho
    xs, ys = [], []
    for r in df.itertuples(index=False):
        if int(r.frame) in exclude:
            continue
        win = _window(cap, int(r.frame))
        if win is None:
            continue
        xs.append(win)
        ys.append(_gaussian_heatmap(r.x * sx, r.y * sy, INP_H, INP_W) if int(r.visible) == 1
                  else np.zeros((INP_H, INP_W), np.float32))
    cap.release()
    return np.stack(xs).astype(store_dtype), np.stack(ys).astype(store_dtype)


def read_manifest(path: str) -> list[dict]:
    """Read the corpus manifest -> ``[{match, chunk, annotations, video, positions}, ...]``.

    Only rows whose annotation CSV exists on disk are returned (rows not yet labelled are skipped
    with a note). The manifest (``data/ball_annotations/manifest.csv``) is the *full* plan across
    every match, so the training set grows simply by annotating more chunks -- no command change.

    Args:
        path: Path to the manifest CSV.

    Returns:
        One dict per annotated chunk, preserving the manifest column names.
    """
    import csv  # noqa: PLC0415

    rows, skipped = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if os.path.exists(row["annotations"]):
                rows.append(row)
            else:
                skipped.append(f"{row['match']}/{row['chunk']}")
    print(f"manifest {path}: {len(rows)} annotated chunk(s)"
          + (f"; skipped (not yet labelled): {', '.join(skipped)}" if skipped else ""))
    return rows


def load_manifest(path: str) -> list[tuple[str, str]]:
    """Back-compat shim -> ``[(annotations_csv, video), ...]`` (see :func:`read_manifest`)."""
    return [(r["annotations"], r["video"]) for r in read_manifest(path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=None,
                    help="corpus manifest CSV -> train on every annotated match/chunk in it (permanent set)")
    ap.add_argument("--annotations", action="append", default=[],
                    help="annotations CSV (repeat with a matching --video to add chunks beyond the manifest)")
    ap.add_argument("--video", action="append", default=[],
                    help="video for the same-position --annotations (repeatable; pairs by order)")
    ap.add_argument("--base", default="tracknetv2", help="zoo base model (tracknetv2 transferred best)")
    ap.add_argument("--weights", default=None, help="base checkpoint (defaults to ~/WASB-SBDT zoo)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    ap.add_argument("--holdout-stride", type=int, default=5, dest="holdout_stride",
                    help="reserve one in N visible rows per CSV for validation (0=train on all)")
    ap.add_argument("--holdout-out", default="data/ball_annotations/holdout_split.csv",
                    dest="holdout_out", help="sidecar CSV recording the held-out rows (reproducible)")
    ap.add_argument("--out", default="outputs/ball_finetuned/ball_ours.pth")
    ap.add_argument("--store-dtype", default="float32", choices=("float32", "float16"),
                    dest="store_dtype", help="in-RAM dataset dtype; float16 halves RAM (see build_dataset)")
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, _WASB_SRC)
    from models import build_model  # noqa: E402

    if len(args.annotations) != len(args.video):
        ap.error(f"got {len(args.annotations)} --annotations but {len(args.video)} --video (must pair 1:1)")
    rows = read_manifest(args.manifest) if args.manifest else []
    for a, v in zip(args.annotations, args.video):
        rows.append({"match": "extra", "chunk": Path(a).stem, "annotations": a, "video": v})
    if not rows:
        ap.error("no training data: pass --manifest and/or --annotations/--video")

    excl = {r["annotations"]: holdout_frames(r["annotations"], args.holdout_stride) for r in rows}
    held = []
    for r in rows:
        hf = excl[r["annotations"]]
        df = pd.read_csv(r["annotations"])
        for rec in df[df["frame"].isin(hf)].itertuples(index=False):
            held.append({"match": r["match"], "chunk": r["chunk"], "annotations": r["annotations"],
                         "video": r["video"], "frame": int(rec.frame), "x": float(rec.x),
                         "y": float(rec.y)})
    if args.holdout_stride > 0:
        hp = Path(args.holdout_out)
        hp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(held, columns=["match", "chunk", "annotations", "video", "frame", "x", "y"]
                     ).to_csv(hp, index=False)
        print(f"held out {len(held)} visible rows for validation -> {hp}")

    print(f"building dataset from {len(rows)} chunk(s) (train rows only) ...")
    parts = [build_dataset(r["annotations"], r["video"], excl[r["annotations"]],
                           store_dtype=args.store_dtype) for r in rows]
    X = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    del parts  # free the per-chunk copies; keep only the concatenated arrays resident (16 GB laptop)
    print(f"  {len(X)} windows ({int((Y.reshape(len(Y), -1).max(1) > 0).sum())} with a visible ball)")

    cfg = {"model": OmegaConf.load(os.path.join(_WASB_SRC, "configs", "model", f"{args.base}.yaml"))}
    model = build_model(cfg)
    base = args.weights or str(Path.home() / "WASB-SBDT" / "pretrained_weights" / f"{args.base}_soccer_best.pth.tar")
    ck = torch.load(base, map_location="cpu")
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                           for k, v in ck["model_state_dict"].items()})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
                        batch_size=args.batch_size, shuffle=True, pin_memory=(dev == "cuda"))
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev).float(), yb.to(dev).float()
            opt.zero_grad()
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                pred = model(xb)[0][:, -1]      # last-frame heatmap logits [B,H,W]
                loss = lossf(pred, yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += loss.detach().item() * len(xb)
        print(f"epoch {ep+1}/{args.epochs}  loss {tot/len(X):.4f}", flush=True)
        if (ep + 1) % 5 == 0:
            ck_path = out_path.with_stem(f"{out_path.stem}_ep{ep+1}")
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep + 1}, ck_path)
            print(f"  checkpoint -> {ck_path}", flush=True)
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
