"""Render an annotated MP4 of the full CV stack: player boxes + teams + ball + calibrated pitch lines.

A visual acceptance test of the whole generator on real footage. Over a contiguous window it draws, per
frame: the football-YOLO detection **boxes** with ByteTrack ids (coloured by jersey team), the
**fine-tuned ball** with a fading trail, and the **pitch lines** projected from the homography (fit from
the cached player ``pitch<->image`` correspondences, so no PnLCalib re-run). Writes an MP4 you can scrub
to judge whether detection / tracking / team-split / ball / calibration all line up at once.

Run:
    python tools/render_clip.py --video "<...>/chunk_000.mp4" --positions outputs/chunk000_dense.parquet \
        --weights outputs/ball_finetuned/tracknetv2_ours.pth --start 20400 --end 21000 \
        --team0 "Man United" --team1 "Man City" --out results/cv_overlay_clip.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WASB_SRC = os.environ.get("FOOTBALL_WASB_PATH", str(Path.home() / "WASB-SBDT" / "src"))
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
INP_W, INP_H, FRAMES_IN = 512, 288, 3
from core.pitch import PITCH_LEN, PITCH_WID  # noqa: E402


def _pitch_polylines() -> list[np.ndarray]:
    """Standard pitch markings as point-lists in 105x68 corner-origin metres (for projection)."""
    cx, cy = PITCH_LEN / 2, PITCH_WID / 2
    lines = [
        np.array([(0, 0), (PITCH_LEN, 0), (PITCH_LEN, PITCH_WID), (0, PITCH_WID), (0, 0)], float),
        np.array([(cx, 0), (cx, PITCH_WID)], float),
        np.array([(0, 13.84), (16.5, 13.84), (16.5, 54.16), (0, 54.16)], float),          # L pen box
        np.array([(PITCH_LEN, 13.84), (88.5, 13.84), (88.5, 54.16), (PITCH_LEN, 54.16)], float),  # R
        np.array([(0, 24.84), (5.5, 24.84), (5.5, 43.16), (0, 43.16)], float),            # L goal area
        np.array([(PITCH_LEN, 24.84), (99.5, 24.84), (99.5, 43.16), (PITCH_LEN, 43.16)], float),  # R
    ]
    circ = np.array([(cx + 9.15 * np.cos(t), cy + 9.15 * np.sin(t))
                     for t in np.linspace(0, 2 * np.pi, 48)], float)
    lines.append(circ)
    return lines


def _load_ball_model(weights: str, base: str = "tracknetv2"):
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


def _cached_homographies(players: pd.DataFrame, *, min_pts: int = 6):
    """Map each cached frame -> (pitch_pts, image_pts) arrays for per-frame homography fitting."""
    corr = {}
    g = players.dropna(subset=["image_x", "image_y", "pitch_x", "pitch_y"])
    for fr, sub in g.groupby("frame"):
        if len(sub) >= min_pts:
            corr[int(fr)] = (sub[["pitch_x", "pitch_y"]].to_numpy(np.float32),
                             sub[["image_x", "image_y"]].to_numpy(np.float32))
    return corr, np.array(sorted(corr), int)


def _nearest_h(fr, corr, cached_frames):
    import cv2
    if len(cached_frames) == 0:
        return None
    cf = int(cached_frames[np.argmin(np.abs(cached_frames - fr))])
    pitch_pts, image_pts = corr[cf]
    h, _ = cv2.findHomography(pitch_pts, image_pts, cv2.RANSAC, 12.0)
    return h


def _team_colors(team_clf, crops_bgr, labels):
    """Decide which KMeans team is the 'red' side -> map team id to a BGR draw colour (red vs sky-blue)."""
    reds = {}
    for lab in set(int(x) for x in labels):
        sel = [c for c, m in zip(crops_bgr, labels) if int(m) == lab and c.size]
        reds[lab] = float(np.mean([c[..., 2].mean() - c[..., 0].mean() for c in sel])) if sel else 0.0
    red_team = max(reds, key=reds.get) if reds else 0
    return {red_team: (60, 60, 235), (1 - red_team): (235, 180, 70)}  # BGR: red, sky-blue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--weights", default=None, help="ball detector weights; omit to skip the ball overlay")
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--start", type=int, default=20400)
    ap.add_argument("--end", type=int, default=21000)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--fps", type=float, default=50.0, help="output fps (50/stride keeps real-time)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--team0", default="team 0")
    ap.add_argument("--team1", default="team 1")
    ap.add_argument("--out", default="results/cv_overlay_clip.mp4")
    args = ap.parse_args()

    import cv2
    import torch

    from generator.extract import (ROLE_NAME, _build_detector, _collect_team_crops, _safe_crop)
    from generator.teams import JerseyColorTeamClassifier
    from generator.tracking import build_tracker

    pos = pd.read_parquet(args.positions)
    players = pos[pos["role"].isin(["player", "goalkeeper", "referee"])]
    corr, cached_frames = _cached_homographies(players)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = _build_detector(device, "football")
    tracker = build_tracker("bytetrack")
    ball_model, bdev = _load_ball_model(args.weights, args.base) if args.weights else (None, None)

    print("fitting team classifier ...")
    crops = _collect_team_crops(cv2, args.video, detector)
    team_clf = JerseyColorTeamClassifier().fit(crops)
    labels0 = team_clf.predict(crops)
    colors = _team_colors(team_clf, crops, labels0)
    names = {0: args.team0, 1: args.team1}
    polylines = _pitch_polylines()

    cap = cv2.VideoCapture(args.video)
    wo, ho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = wo / INP_W, ho / INP_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (wo, ho))

    lo = args.start - (FRAMES_IN - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(lo, 0))
    buf: deque = deque(maxlen=FRAMES_IN)
    trail: deque = deque(maxlen=12)
    fr = max(lo, 0)
    n_written = 0
    while fr <= args.end:
        ok, bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(bgr, (INP_W, INP_H))
        rgbn = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        buf.append(((rgbn - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
        render = fr >= args.start and (fr - args.start) % args.stride == 0
        if render and (ball_model is None or len(buf) == FRAMES_IN):
            canvas = bgr.copy()
            # 1) pitch lines from the nearest cached homography
            h = _nearest_h(fr, corr, cached_frames)
            if h is not None:
                for pl in polylines:
                    ip = cv2.perspectiveTransform(pl.reshape(-1, 1, 2), h).reshape(-1, 2)
                    cv2.polylines(canvas, [ip.astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
            # 2) player boxes + team + track id
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                xyxy, _, role_ids, tids, _ = tracker.update(detector, rgb)
            except Exception:  # noqa: BLE001
                xyxy, role_ids, tids = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, int)
            if len(xyxy):
                pcrops = [_safe_crop(bgr, b) for b in xyxy]
                teams = team_clf.predict(pcrops)
                for b, rid, tid, tm in zip(xyxy, role_ids, tids, teams):
                    x1, y1, x2, y2 = (int(v) for v in b)
                    role = ROLE_NAME.get(int(rid), "player")
                    col = (190, 190, 190) if role == "referee" else colors.get(int(tm), (200, 200, 200))
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(canvas, f"{int(tid)}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                col, 1, cv2.LINE_AA)
            # 3) fine-tuned ball + trail (only if a ball model was supplied)
            if ball_model is not None:
                with torch.no_grad():
                    hm = ball_model(torch.from_numpy(np.concatenate(list(buf), 0)[None]).to(bdev))
                    hm = hm[0].sigmoid()[0, -1].cpu().numpy()
                yb, xb = np.unravel_index(int(hm.argmax()), hm.shape)
                if float(hm[yb, xb]) > args.thr:
                    bx, by = int(xb * sx), int(yb * sy)
                    trail.append((bx, by))
                for i in range(1, len(trail)):
                    cv2.line(canvas, trail[i - 1], trail[i], (255, 255, 255), 2, cv2.LINE_AA)
                if trail:
                    cv2.circle(canvas, trail[-1], 7, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, trail[-1], 5, (255, 255, 255), -1, cv2.LINE_AA)
            # banner
            ball_tag = "ball + " if ball_model is not None else ""
            cv2.rectangle(canvas, (0, 0), (wo, 34), (20, 20, 20), -1)
            cv2.putText(canvas, f"f{fr}  {names[0]} (red)  vs  {names[1]} (blue)  | {ball_tag}pitch calib",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            n_written += 1
            if n_written % 50 == 0:
                print(f"  {n_written} frames written (at {fr}) ...")
        fr += 1
    cap.release()
    writer.release()
    print(f"wrote {n_written} frames -> {args.out}")


if __name__ == "__main__":
    main()
