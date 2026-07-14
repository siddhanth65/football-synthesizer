"""Zoom-crop verification: show a tight box around each predicted ball so a human can confirm it's a ball.

The grid preview proves the detector *fires* on unseen frames; this proves *what* it fires on. For each
sampled frame it crops a small box centred on the predicted peak and enlarges it — if there's a white
ball in the centre of most crops, the detector generalised; if it's grass/lines/logos, it overfit.

Run:
    python tools/zoom_ball.py --video "<...>/chunk_000.mp4" \
        --weights outputs/ball_finetuned/tracknetv2_ours.pth --start 20620 --n 16 --out results/ball_zoom.png
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
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--box", type=int, default=48, help="half-size of crop box in original px")
    ap.add_argument("--out", default="results/ball_zoom.png")
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(want[0] - (FRAMES_IN - 1), 0))
    buf: deque = deque(maxlen=FRAMES_IN)
    crops: list[tuple[int, np.ndarray, float, bool]] = []
    fr = max(want[0] - (FRAMES_IN - 1), 0)
    wantset = set(want)
    b = args.box
    while fr <= want[-1] and len(crops) < args.n:
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
            cx, cy = int(x * sx), int(y * sy)
            x0, y0 = max(cx - b, 0), max(cy - b, 0)
            crop = rgb_full[y0:y0 + 2 * b, x0:x0 + 2 * b]
            crops.append((fr, crop, s, s > args.thr))
        fr += 1
    cap.release()

    cols = 4
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, (f, crop, s, det) in zip(np.atleast_1d(axes).ravel(), crops):
        ax.imshow(crop)
        h, w = crop.shape[:2]
        ax.plot(w / 2, h / 2, "+", color="lime", ms=14, mew=2)  # predicted ball at crop centre
        ax.set_title(f"f{f}  conf={s:.2f}" + ("" if det else " (no det)"), fontsize=9)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(crops):]:
        ax.axis("off")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"tiled {len(crops)} zoom crops -> {args.out}")


if __name__ == "__main__":
    main()
