"""Benchmark candidate football detectors on held-out broadcast frames (Pass C, review section 5).

Runs each candidate on the same frames, tabulates per-role detection counts, and writes side-by-side
overlays so role accuracy can be judged by eye. Without manual ground truth this reports *proxy*
recall (counts per role; a wide tactical shot shows ~18-22 outfield players + GK(s) + officials + the
ball), so treat it as a screen, then confirm the winner on annotated frames.

Run: ``python tools/benchmark_detectors.py``  (downloads HF weights on first use).
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEO = r"C:\Users\siddh_ygv5bws\OneDrive\Desktop\cv-football\chunks\chunk_000.mp4"
FRAMES = [13031, 13050, 13065, 21726, 21740, 21765]  # wide tactical frames from both segments
CONF = 0.25
OUT = Path("outputs/benchmark")
_ROLE_COLOR = {"player": (0, 200, 0), "goalkeeper": (0, 165, 255), "referee": (255, 0, 255),
               "ball": (0, 0, 255)}


def _norm_role(name: str) -> str | None:
    """Map a model's class name to a canonical role (or None to ignore non-football classes)."""
    n = name.lower()
    if "ball" in n:
        return "ball"
    if "goal" in n or n == "gk":
        return "goalkeeper"
    if "ref" in n:
        return "referee"
    if "player" in n or n == "person":
        return "player"
    return None


def _hf_pt(repo: str) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415

    pts = [f for f in list_repo_files(repo) if f.endswith(".pt")]
    pref = [f for f in pts if "best" in f.lower()] or pts
    return hf_hub_download(repo, pref[0])


def _ultra(weights: str):
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(weights)
    names = model.names

    def run(frame_bgr):
        r = model(frame_bgr, verbose=False, conf=CONF)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        out = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().cpu().tolist()):
            role = _norm_role(names[c])
            if role:
                out.append((role, b))
        return out

    return run


def _rfdetr():
    from rfdetr import RFDETRNano  # noqa: PLC0415

    model = RFDETRNano()
    coco = {1: "player", 37: "ball"}  # rfdetr COCO-91 person/sports-ball

    def run(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(rgb, threshold=CONF)
        out = []
        for cid, b in zip(np.asarray(det.class_id), np.asarray(det.xyxy)):
            if int(cid) in coco:
                out.append((coco[int(cid)], b))
        return out

    return run


def _candidates():
    """Build the runnable candidates, skipping any that fail to load (missing pkg/weights)."""
    specs = [
        ("coco_yolov8s", lambda: _ultra("yolov8s.pt")),
        ("hf_uisikdag_v8", lambda: _ultra(_hf_pt("uisikdag/yolo-v8-football-players-detection"))),
        ("hf_soccana_v11", lambda: _ultra(_hf_pt("Adit-jain/soccana"))),
        ("rfdetr_coco", _rfdetr),
    ]
    built = {}
    for name, factory in specs:
        try:
            built[name] = factory()
            print(f"loaded: {name}")
        except Exception as exc:  # noqa: BLE001 - a candidate that won't load is just skipped
            print(f"SKIP {name}: {type(exc).__name__}: {exc}")
    return built


def main() -> None:
    cands = _candidates()
    frames = {}
    cap = cv2.VideoCapture(VIDEO)
    for fr in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if ok:
            frames[fr] = img
    cap.release()

    # role counts: agg[name][role] summed across frames, and per-frame for the table.
    agg = {name: collections.Counter() for name in cands}
    print("\nper-frame role counts (player / GK / referee / ball):")
    header = f"{'frame':>7} | " + " | ".join(f"{n:^26}" for n in cands)
    print(header)
    for fr, img in frames.items():
        cells = []
        for name, run in cands.items():
            dets = run(img)
            c = collections.Counter(r for r, _ in dets)
            agg[name].update(c)
            cells.append(f"{c['player']:2d}/{c['goalkeeper']:1d}/{c['referee']:1d}/{c['ball']:1d}")
            # overlay
            ov = img.copy()
            for role, b in dets:
                x1, y1, x2, y2 = b.astype(int)
                cv2.rectangle(ov, (x1, y1), (x2, y2), _ROLE_COLOR[role], 2)
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"frame_{fr:05d}.png"), ov)
        print(f"{fr:>7} | " + " | ".join(f"{c:^26}" for c in cells))

    nfr = len(frames)
    print(f"\naverages over {nfr} frames (player / GK-rate / ref-rate / ball-rate):")
    for name, c in agg.items():
        gk_rate = c["goalkeeper"] / nfr
        ref_rate = c["referee"] / nfr
        ball_rate = c["ball"] / nfr
        print(f"  {name:18s}: players/frame={c['player']/nfr:5.1f} | GK/fr={gk_rate:.2f} "
              f"| ref/fr={ref_rate:.2f} | ball/fr={ball_rate:.2f}")
    print(f"\noverlays written under {OUT}/<candidate>/ for visual role-accuracy inspection.")


if __name__ == "__main__":
    main()
