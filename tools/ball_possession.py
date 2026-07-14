"""Demonstrate the ball unlock: fine-tuned detections -> projected ball -> linked track -> possession.

End-to-end on one chunk, reusing cached player tracks (so no PnLCalib re-run): for each sampled frame in
a window we (1) run the fine-tuned detector for the ball in image px, (2) fit that frame's homography
from the player ``image_xy -> pitch_xy`` correspondences already in the parquet and project the ball to
pitch metres, then (3) :func:`generator.ball.link_ball` cleans the trajectory and
:func:`generator.ball.assign_possession` gives nearest-carrier possession. Reports possession share,
switches, and the longest spell — the signals that gate PPDA / passing networks / line-breaks.

Run:
    python tools/ball_possession.py --video "<...>/chunk_000.mp4" \
        --positions outputs/chunk000_dense.parquet --weights outputs/ball_finetuned/tracknetv2_ours.pth \
        --start 18000 --end 24000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

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


def detect_ball_imagexy(video: str, frames: list[int], model, dev, *, thr: float = 0.5) -> dict:
    """Return ``{frame: (x, y)}`` ball image-px peaks for the requested frames (contiguous read)."""
    import cv2
    import torch
    want = set(frames)
    lo, hi = min(frames), max(frames)
    cap = cv2.VideoCapture(video)
    wo, ho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = wo / INP_W, ho / INP_H
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(lo - (FRAMES_IN - 1), 0))
    buf: deque = deque(maxlen=FRAMES_IN)
    out: dict[int, tuple[float, float]] = {}
    fr = max(lo - (FRAMES_IN - 1), 0)
    while fr <= hi:
        ok, bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(bgr, (INP_W, INP_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        buf.append(((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
        if len(buf) == FRAMES_IN and fr in want:
            with torch.no_grad():
                win = np.concatenate(list(buf), 0)
                hm = model(torch.from_numpy(win[None]).to(dev))[0].sigmoid()[0, -1].cpu().numpy()
            y, x = np.unravel_index(int(hm.argmax()), hm.shape)
            if float(hm[y, x]) > thr:
                out[fr] = (x * sx, y * sy)
        fr += 1
    cap.release()
    return out


def project_ball(ball_imgxy: dict, players: pd.DataFrame, *, min_pts: int = 6,
                 max_reproj_px: float = 12.0, return_stats: bool = False):
    """Fit a per-frame homography from player image<->pitch and project each frame's ball to pitch m.

    The projection is a real yield bottleneck distinct from detection: a frame needs >= ``min_pts`` player
    correspondences in view (close-ups fail) and a non-degenerate homography. With ``return_stats`` the
    second element reports why frames were lost -- ``detected`` / ``too_few_pts`` / ``homography_failed``
    / ``projected`` -- so the projection yield can be read separately from the detection rate.
    """
    import cv2
    rows = []
    too_few = homog_fail = 0
    for fr, (bx, by) in ball_imgxy.items():
        g = players[players["frame"] == fr].dropna(subset=["image_x", "image_y", "pitch_x", "pitch_y"])
        if len(g) < min_pts:
            too_few += 1
            continue
        src = g[["image_x", "image_y"]].to_numpy(np.float32)
        dst = g[["pitch_x", "pitch_y"]].to_numpy(np.float32)
        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, max_reproj_px)
        if h is None:
            homog_fail += 1
            continue
        p = cv2.perspectiveTransform(np.array([[[bx, by]]], np.float32), h)[0, 0]
        rows.append({"frame": int(fr), "x": float(p[0]), "y": float(p[1])})
    out = pd.DataFrame(rows, columns=["frame", "x", "y"]).sort_values("frame")
    if return_stats:
        stats = {"detected": len(ball_imgxy), "too_few_pts": too_few,
                 "homography_failed": homog_fail, "projected": len(out)}
        return out, stats
    return out


def summarise(poss: pd.DataFrame, team_names: dict | None = None) -> str:
    """Possession share, switch count, longest spell from an ``assign_possession`` frame table."""
    if poss.empty:
        return "no possession assigned (no ball within radius of any player)"
    team = poss["team"].to_numpy()
    share = poss["team"].value_counts(normalize=True).sort_index()
    switches = int((np.diff(team) != 0).sum())
    # longest consecutive spell (in possession samples)
    best = cur = 1
    for i in range(1, len(team)):
        cur = cur + 1 if team[i] == team[i - 1] else 1
        best = max(best, cur)
    nm = team_names or {}
    lines = [f"possession frames: {len(poss)}   switches: {switches}   longest spell: {best} samples"]
    for t, s in share.items():
        lines.append(f"  {nm.get(int(t), f'team {int(t)}')}: {s:.1%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--start", type=int, default=18000)
    ap.add_argument("--end", type=int, default=24000)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/ball_track_chunk000.parquet")
    ap.add_argument("--team0", default="team 0", help="display name for team label 0")
    ap.add_argument("--team1", default="team 1", help="display name for team label 1")
    args = ap.parse_args()

    from generator.ball import assign_possession, link_ball

    pos = pd.read_parquet(args.positions)
    players = pos[pos["role"].isin(["player", "goalkeeper"])]
    win = players[(players["frame"] >= args.start) & (players["frame"] <= args.end)]
    frames = sorted(win["frame"].unique().astype(int))
    print(f"window {args.start}-{args.end}: {len(frames)} sampled frames with players")

    model, dev = _load_model(args.weights, args.base)
    print("detecting ball ...")
    ball_img = detect_ball_imagexy(args.video, frames, model, dev, thr=args.thr)
    print(f"  detection: ball found in {len(ball_img)}/{len(frames)} sampled frames "
          f"({len(ball_img)/max(len(frames),1):.0%})")

    ball_pitch, yld = project_ball(ball_img, players, return_stats=True)
    det = max(yld["detected"], 1)
    print(f"  projection yield: {yld['projected']}/{yld['detected']} detections -> pitch "
          f"({yld['projected']/det:.0%})  | lost: {yld['too_few_pts']} too-few-corr, "
          f"{yld['homography_failed']} homography-fail")
    print(f"  end-to-end ball recall: {yld['projected']}/{len(frames)} frames "
          f"({yld['projected']/max(len(frames),1):.0%})  linking ...")
    track = link_ball(ball_pitch, fps=args.fps)
    obs = int(track["observed"].sum()) if len(track) else 0
    print(f"  linked track: {len(track)} samples ({obs} observed, {len(track)-obs} interpolated)")

    poss = assign_possession(track, win)
    names = {0: args.team0, 1: args.team1}
    print("\n" + summarise(poss, team_names=names))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    track.to_parquet(args.out, index=False)
    print(f"\nsaved ball track -> {args.out}")


if __name__ == "__main__":
    main()
