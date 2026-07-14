"""Eyeball the C3 attacker head + the metric engine on real frames (top-down galleries).

For a spread of busy frames it renders, per frame, two top-down panels (a third broadcast panel when
``--video`` is given):

  * RUN HEAD -- each player's dot with a GREEN arrow to where they ACTUALLY are 1.5 s later and an
    ORANGE arrow to where the learned run head PREDICTS they go. Arrows that roughly agree = the head
    is sensible. (Predictions are in-sample here -- this is an illustration, not the held-out metric.)
  * TEAM SHAPE -- each team's convex hull (filled), centroid (ringed), the five width-lanes (dotted),
    an arrow showing the keeper-derived ATTACK DIRECTION, and a vertical BUILD-UP HEIGHT line. This is
    the geometry behind the structural fingerprint, drawn so it can be sanity-checked by eye.

Run: ``python tools/inspect_attacker.py --positions outputs/chunk000_dense.parquet \
        --video "<...>/chunk_000.mp4" --out outputs/inspect_attacker``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.heads import RUN_FEATURES, run_examples, speed_heading_predictor  # noqa: E402
from attacker.labels import RUN_HORIZON_S, run_targets  # noqa: E402
from attacker.tracks import DEFAULT_FPS, build_tracks  # noqa: E402
from fingerprint.structural_metrics import (  # noqa: E402
    DEF_LINE_QUANTILE,
    LANE_EDGES,
    attacking_coord,
    direction_metrics,
    resolve_attack_directions,
)
from tools.visualize import PITCH_LEN, PITCH_WID, _TEAM_COLORS, _draw_pitch  # noqa: E402


def _fit_run_model(ex: pd.DataFrame):
    """The magnitude-faithful speed x heading predictor (so arrows show realistic run length)."""
    return speed_heading_predictor(ex[list(RUN_FEATURES)].to_numpy(), ex[["dx", "dy"]].to_numpy())


def _pick_frames(rt: pd.DataFrame, n: int, min_players: int = 8) -> list[int]:
    """The ``n`` widest busy frames (most players spanning the most pitch) -- clearest to eyeball."""
    stats = rt.groupby("frame")["x_s"].agg(count="count", span=np.ptp)
    good = stats[stats["count"] >= min_players]
    good = (good if not good.empty else stats).sort_values("span", ascending=False)
    return sorted(int(f) for f in good.head(n).index)


def _panel_run(ax, frame_pred: pd.DataFrame) -> None:
    _draw_pitch(ax)
    for _, p in frame_pred.iterrows():
        c = _TEAM_COLORS.get(int(p["team"]), "#aaaaaa")
        ax.scatter(p["x_s"], p["y_s"], s=70, c=c, edgecolors="white", lw=1, zorder=5)
        ax.annotate("", xy=(p["x_fut"], p["y_fut"]), xytext=(p["x_s"], p["y_s"]),
                    arrowprops=dict(arrowstyle="-|>", color="#39FF14", lw=2), zorder=4)  # actual
        if pd.notna(p.get("vx")) and pd.notna(p.get("vy")):  # constant-velocity baseline (cyan dotted)
            ax.annotate("", xy=(p["x_s"] + p["vx"] * RUN_HORIZON_S, p["y_s"] + p["vy"] * RUN_HORIZON_S),
                        xytext=(p["x_s"], p["y_s"]),
                        arrowprops=dict(arrowstyle="-|>", color="#00E5EE", lw=1.4, ls=":"), zorder=3)
        ax.annotate("", xy=(p["x_s"] + p["pred_dx"], p["y_s"] + p["pred_dy"]),
                    xytext=(p["x_s"], p["y_s"]),
                    arrowprops=dict(arrowstyle="-|>", color="#FF8C00", lw=2, ls="--"), zorder=4)  # learned
    ax.set_title("RUN HEAD: green = actual +1.5s | orange = predicted (speed x heading) | cyan = const-vel",
                 fontsize=11)


def _panel_shape(ax, players: pd.DataFrame, attack_dirs: dict[int, int]) -> None:
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    _draw_pitch(ax)
    for edge in LANE_EDGES[1:-1]:  # the five width-lanes (dotted)
        ax.axhline(edge, color="white", ls=":", lw=0.6, alpha=0.5, zorder=1)
    for team, g in players.groupby("team"):
        if int(team) < 0 or len(g) < 4:
            continue
        c = _TEAM_COLORS.get(int(team), "#aaaaaa")
        px, py = g["pitch_x"].to_numpy(), g["pitch_y"].to_numpy()
        pts = np.column_stack([px, py])
        if len(pts) >= 3:
            try:
                v = ConvexHull(pts).vertices
                ax.fill(px[v], py[v], color=c, alpha=0.15, zorder=2)
                ax.plot(np.r_[px[v], px[v][0]], np.r_[py[v], py[v][0]], color=c, lw=1.2, zorder=2)
            except Exception:  # noqa: BLE001 - collinear cloud, skip the hull
                pass
        cx, cy = px.mean(), py.mean()
        ax.scatter(cx, cy, s=260, facecolors="none", edgecolors=c, lw=2.4, zorder=6)
        adir = attack_dirs.get(int(team))
        width, length = float(np.ptp(py)), float(np.ptp(px))
        label = f"T{int(team)}  w{width:.0f} l{length:.0f}"
        if adir is not None:
            ax.annotate("", xy=(cx + adir * 14, cy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5), zorder=6)
            dm = direction_metrics(px, g.get("is_keeper", pd.Series(False, index=g.index)).to_numpy(),
                                   adir)
            real_buildup = dm["buildup_height"] if adir > 0 else PITCH_LEN - dm["buildup_height"]
            ax.plot([real_buildup, real_buildup], [0, PITCH_WID], color=c, lw=1.4, alpha=0.7, zorder=3)
            label += f" h{dm['buildup_height']:.0f}"
        ax.text(cx, cy - 4, label, color=c, fontsize=9, fontweight="bold", ha="center", zorder=7)
    ax.set_title("TEAM SHAPE: hull + centroid (ring), arrow = attack dir, line = build-up height",
                 fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--video", default=None, help="optional broadcast video for a third panel")
    ap.add_argument("--out", default="outputs/inspect_attacker")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = pd.read_parquet(args.positions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    attack_dirs = resolve_attack_directions(pos)
    tracks = build_tracks(pos, fps=args.fps)
    rt = run_targets(tracks, horizon_s=RUN_HORIZON_S, fps=args.fps)
    ex = run_examples(rt, attack_dirs)
    predict = _fit_run_model(ex)
    pred_norm = predict(ex[list(RUN_FEATURES)].to_numpy())
    ex["pred_ndx"], ex["pred_ndy"] = pred_norm[:, 0], pred_norm[:, 1]
    ex["sign"] = ex["team"].map(lambda t: attack_dirs.get(int(t), 0))
    ex["pred_dx"], ex["pred_dy"] = ex["sign"] * ex["pred_ndx"], ex["pred_ndy"]
    pred = rt.merge(ex[["frame", "track_id", "pred_dx", "pred_dy"]], on=["frame", "track_id"])

    cap = None
    if args.video:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(args.video)
    frames = _pick_frames(pred, args.n)
    for fr in frames:
        fp = pred[pred["frame"] == fr]
        players = pos[(pos["frame"] == fr) & (pos["role"].isin(["player", "goalkeeper"]))].dropna(
            subset=["pitch_x", "pitch_y"])
        ncol = 3 if cap is not None else 2
        fig, axes = plt.subplots(1, ncol, figsize=(10 * ncol, 8))
        axv = None
        if cap is not None:
            axv, ax_run, ax_shape = axes
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, bgr = cap.read()
            if ok:
                axv.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                for _, p in players.iterrows():
                    axv.scatter(p["image_x"], p["image_y"], s=60,
                                c=_TEAM_COLORS.get(int(p["team"]), "#aaaaaa"), edgecolors="white", lw=1)
            axv.set_title(f"broadcast frame {fr}", fontsize=11)
            axv.axis("off")
        else:
            ax_run, ax_shape = axes
        _panel_run(ax_run, fp)
        _panel_shape(ax_shape, players, attack_dirs)
        fig.suptitle(f"frame {fr} ({fr / args.fps / 60:.1f} min) -- {len(fp)} run arrows, "
                     f"{len(players)} players", fontsize=13)
        fig.patch.set_facecolor("#1a1a2e")
        fig.savefig(out / f"inspect_{fr:05d}.png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    if cap is not None:
        cap.release()

    (out / "README.txt").write_text(
        "ATTACKER + METRIC-ENGINE EYEBALL\n================================\n"
        "inspect_*.png (one per busy frame across the match):\n"
        "  RUN HEAD panel  : each dot = a player. GREEN arrow = where they ACTUALLY moved over the\n"
        "    next 1.5 s. ORANGE dashed = predicted run (speed x heading head -- realistic length).\n"
        "    CYAN dotted = constant-velocity baseline. Orange/cyan should track green in direction and\n"
        "    roughly in length. (in-sample, illustrative; held-out: the RMSE-optimal mean head scores\n"
        "    5.11 m / hit@3m 0.566 vs old-360 6.33/0.244; speed x heading restores run magnitude.)\n"
        "  TEAM SHAPE panel: per team -> filled convex hull (territory), ringed centroid, big arrow =\n"
        "    keeper-derived ATTACK DIRECTION (should point at the goal that team attacks), vertical\n"
        "    line = build-up height (avg position up the pitch). Dotted horizontals = the 5 lanes.\n"
        "  Check: attack arrows of the two teams point at OPPOSITE goals; hulls/centroids sit on the\n"
        "    players; predicted runs broadly track actual runs.\n")
    print(f"wrote {len(frames)} inspection frames to {out}/ (see README.txt)")
    print("open:", out / "README.txt")


if __name__ == "__main__":
    main()
