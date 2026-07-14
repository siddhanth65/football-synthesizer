"""Eyeball a fine-tuned ball detector on UNSEEN frames: overlay the predicted ball on real frames.

Validation recall on the training frames can read 100% purely from overfitting, so the honest check is
visual: run the detector over a contiguous live window the model never saw labelled, mark its predicted
ball position on each frame, and tile a grid so a human can confirm it's tracking the *ball* (not a
logo/line). Saves a PNG montage.

Run:
    python tools/preview_ball.py --video "<...>/chunk_000.mp4" \
        --weights outputs/ball_finetuned/tracknetv2_ours.pth --start 20620 --n 16 \
        --out results/ball_preview.png
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

_WASB_SRC = os.environ.get("FOOTBALL_WASB_PATH", str(Path.home() / "WASB-SBDT" / "src"))
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
INP_W, INP_H, FRAMES_IN = 512, 288, 3


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--start", type=int, default=20620)
    ap.add_argument("--n", type=int, default=16, help="frames to tile")
    ap.add_argument("--stride", type=int, default=6, help="frames between tiles (spread the window)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="results/ball_preview.png")
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    model, dev = _load_model(args.weights, args.base)
    cap = cv2.VideoCapture(args.video)
    wo, ho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = wo / INP_W, ho / INP_H

    want = [args.start + i * args.stride for i in range(args.n)]
    lo = want[0] - (FRAMES_IN - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(lo, 0))
    buf: deque = deque(maxlen=FRAMES_IN)
    shown: list[tuple[int, np.ndarray, tuple | None, float]] = []
    fr = max(lo, 0)
    wantset = set(want)
    while fr <= want[-1] and len(shown) < args.n:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb_full, (INP_W, INP_H)).astype(np.float32) / 255.0
        buf.append(((small - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
        if len(buf) == FRAMES_IN and fr in wantset:
            with torch.no_grad():
                win = np.concatenate(list(buf), 0)
                hm = model(torch.from_numpy(win[None]).to(dev))[0].sigmoid()[0, -1].cpu().numpy()
            y, x = np.unravel_index(int(hm.argmax()), hm.shape)
            s = float(hm[y, x])
            xy = (x * sx, y * sy) if s > args.thr else None
            shown.append((fr, rgb_full, xy, s))
        fr += 1
    cap.release()

    cols = 4
    rows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6))
    for ax, (f, img, xy, s) in zip(np.atleast_1d(axes).ravel(), shown):
        ax.imshow(img)
        if xy is not None:
            ax.add_patch(plt.Circle(xy, 14, fill=False, color="lime", lw=2))
            ax.plot(*xy, "+", color="lime", ms=8)
        ax.set_title(f"f{f}  conf={s:.2f}" + ("" if xy else "  (no det)"), fontsize=8)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(shown):]:
        ax.axis("off")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    n_det = sum(1 for _, _, xy, _ in shown if xy is not None)
    print(f"tiled {len(shown)} frames ({n_det} with a detection) -> {args.out}")


if __name__ == "__main__":
    main()
