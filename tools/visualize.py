"""Visual verification: positions parquet + video -> side-by-side overlays + track trajectories.

For manual confirmation of the generator. For sampled frames it renders two panels side by side:
  * LEFT  -- the real video frame with each detected player marked at its foot point, coloured by
             team, labelled with its persistent track id (ball = white star, ball-carrier ringed).
  * RIGHT -- the top-down freeze frame the model consumes (same players on a 105x68 pitch).
Plus a trajectory plot showing every persistent track id's path over the segment (ID persistence).

Run: ``python tools/visualize.py --positions outputs/chunk000_gpu.parquet \
        --video "<...>/chunk_000.mp4" --out outputs/viz --n-frames 4``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import generator.*

PITCH_LEN, PITCH_WID = 105.0, 68.0
_TEAM_COLORS = {0: "#e6194B", 1: "#4363d8", -1: "#aaaaaa"}  # team0 red, team1 blue, ball/other grey


def _draw_pitch(ax) -> None:
    """Draw a simple 105x68 pitch (lines, boxes, centre circle) on a matplotlib axis."""
    import matplotlib.patches as patches

    ax.add_patch(patches.Rectangle((0, 0), PITCH_LEN, PITCH_WID, ec="white", fc="#2d8a4e", lw=2))
    ax.plot([PITCH_LEN / 2, PITCH_LEN / 2], [0, PITCH_WID], color="white", lw=1)
    ax.add_patch(patches.Circle((PITCH_LEN / 2, PITCH_WID / 2), 9.15, ec="white", fc="none", lw=1))
    for x0 in (0, PITCH_LEN - 16.5):  # penalty boxes
        ax.add_patch(patches.Rectangle((x0, 13.84), 16.5, 40.32, ec="white", fc="none", lw=1))
    ax.set_xlim(-3, PITCH_LEN + 3)
    ax.set_ylim(-3, PITCH_WID + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d8a4e")
    ax.invert_yaxis()  # image-style y so the pitch panel matches the video orientation


def _plot_pitch_lines_on_video(axv, homography) -> None:
    """Overlay the reconstructed pitch model on the video panel (proves the calibration visually)."""
    from tools.diag_calib import _CIRCLE, _PITCH_LINES, _homography_projector

    proj = _homography_projector(homography)
    for a, b in _PITCH_LINES:
        ia, ib = proj(np.array(a, float)), proj(np.array(b, float))
        if ia is not None and ib is not None:
            axv.plot([ia[0], ib[0]], [ia[1], ib[1]], color="yellow", lw=1.5, zorder=2)
    cpts = [proj(np.array(p, float)) for p in _CIRCLE]
    cpts = [p for p in cpts if p is not None]
    if len(cpts) > 2:
        cpts.append(cpts[0])
        axv.plot([p[0] for p in cpts], [p[1] for p in cpts], color="yellow", lw=1.5, zorder=2)


def render_overlays(
    positions: pd.DataFrame, video: str, out_dir: Path, n_frames: int = 4, calibrator=None
) -> list[Path]:
    """Render side-by-side (video | pitch) overlays for ``n_frames`` evenly spaced frames.

    If ``calibrator`` is given, the reconstructed pitch lines are drawn on the video panel (yellow) so
    each frame self-verifies: the lines should sit on the real painted markings.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(positions["frame"].unique())
    picks = frames[:: max(len(frames) // n_frames, 1)][:n_frames]
    cap = cv2.VideoCapture(video)
    written: list[Path] = []
    for fr in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        grp = positions[positions["frame"] == fr]
        players = grp[grp["role"] != "ball"]
        ball = grp[grp["role"] == "ball"]

        fig, (axv, axp) = plt.subplots(1, 2, figsize=(20, 6))
        axv.imshow(rgb)
        axv.set_title(f"video frame {int(fr)} (detections + track id; yellow = reconstructed pitch)")
        axv.axis("off")
        if calibrator is not None:
            res = calibrator.calibrate_frame(frame_bgr)
            if res.homography is not None:
                _plot_pitch_lines_on_video(axv, res.homography)
                axv.set_xlim(0, rgb.shape[1])
                axv.set_ylim(rgb.shape[0], 0)
        for _, p in players.iterrows():
            c = _TEAM_COLORS.get(int(p["team"]), "#aaaaaa")
            axv.scatter(p["image_x"], p["image_y"], s=90, c=c, edgecolors="white", lw=1.2, zorder=3)
            if bool(p.get("is_actor", False)):
                axv.scatter(p["image_x"], p["image_y"], s=320, facecolors="none",
                            edgecolors="yellow", lw=2.2, zorder=4)
            axv.text(p["image_x"] + 6, p["image_y"], str(int(p["track_id"])), color="white",
                     fontsize=8, zorder=5)
        for _, b in ball.iterrows():
            axv.scatter(b["image_x"], b["image_y"], s=160, marker="*", c="white",
                        edgecolors="black", lw=1, zorder=6)

        _draw_pitch(axp)
        axp.set_title(f"top-down freeze frame (calib err {grp['calib_error_m'].mean():.2f} m)")
        valid = players.dropna(subset=["pitch_x", "pitch_y"])
        for _, p in valid.iterrows():
            c = _TEAM_COLORS.get(int(p["team"]), "#aaaaaa")
            axp.scatter(p["pitch_x"], p["pitch_y"], s=140, c=c, edgecolors="white", lw=1, zorder=3)
            if bool(p.get("is_keeper", False)):
                axp.scatter(p["pitch_x"], p["pitch_y"], s=300, facecolors="none",
                            edgecolors="black", lw=1.6, zorder=4)
            axp.text(p["pitch_x"] + 0.8, p["pitch_y"], str(int(p["track_id"])), color="white",
                     fontsize=8, zorder=5)
        for _, b in ball.dropna(subset=["pitch_x", "pitch_y"]).iterrows():
            axp.scatter(b["pitch_x"], b["pitch_y"], s=160, marker="*", c="white",
                        edgecolors="black", lw=1, zorder=6)
        fig.patch.set_facecolor("#1a1a2e")
        out = out_dir / f"overlay_{int(fr):05d}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(out)
    cap.release()
    return written


def render_tracks(positions: pd.DataFrame, out_dir: Path) -> Path:
    """Plot every persistent track id's pitch trajectory over the segment (ID persistence)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    players = positions[positions["role"] != "ball"].dropna(subset=["pitch_x", "pitch_y"])
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_pitch(ax)
    for tid, tr in players.sort_values("frame").groupby("track_id"):
        c = _TEAM_COLORS.get(int(tr["team"].mode().iloc[0]), "#aaaaaa")
        ax.plot(tr["pitch_x"], tr["pitch_y"], "-", color=c, lw=1.5, alpha=0.8, zorder=3)
        ax.scatter(tr["pitch_x"].iloc[0], tr["pitch_y"].iloc[0], s=40, c=c,
                   edgecolors="white", lw=1, zorder=4)  # start
        ax.text(tr["pitch_x"].iloc[-1], tr["pitch_y"].iloc[-1], str(int(tid)), color="white",
                fontsize=9, zorder=5)  # end-of-path id label
    n_ids = players["track_id"].nunique()
    n_fr = positions["frame"].nunique()
    ax.set_title(f"{n_ids} persistent track ids over {n_fr} frames (each line = one player's path)")
    fig.patch.set_facecolor("#1a1a2e")
    out = out_dir / "tracks.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/viz")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--pitch-lines", action="store_true",
                    help="overlay reconstructed pitch lines on the video (recomputes calibration)")
    args = ap.parse_args()
    pos = pd.read_parquet(args.positions)
    out = Path(args.out)
    calibrator = None
    if args.pitch_lines:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        calibrator = PnLCalibCalibrator()
    overlays = render_overlays(pos, args.video, out, n_frames=args.n_frames, calibrator=calibrator)
    tracks = render_tracks(pos, out)
    print(f"wrote {len(overlays)} overlays + tracks plot to {out}/")
    for p in [*overlays, tracks]:
        print("  ", p)


if __name__ == "__main__":
    main()
