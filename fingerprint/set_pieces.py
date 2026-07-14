"""Heuristic set-piece / restart detection from the linked ball track (corners, throw-ins, goal kicks, FKs).

Set pieces are **dead-ball restarts** -- the ball leaves play, then re-enters where the restart is taken.
On a linked ball track that shows up as a **long gap** (the ball is untracked / out of play) followed by a
re-entry at a pitch landmark; we classify the re-entry by **where** it lands:

* **corner** -- within ``corner_m`` of a pitch corner,
* **throw_in** -- on a touchline (``|y|`` near 0 or 68) away from the corners,
* **goal_kick** -- inside either six-yard box, central,
* **free_kick** -- re-entry elsewhere (the fuzzy residual).

Using the *gap* (not raw stillness) is deliberate: in our footage the ball is slow even in open play
(dribbling + interpolation jitter), so a speed threshold floods with false positives -- the out-of-play
gap is the reliable restart signal. Boundary classes (corner/throw-in/goal kick) are trustworthy; the
free-kick residual is weaker. A positional heuristic, not event truth -- validate counts against a known
match. Pure numpy/pandas; works on ``generator.ball.link_ball`` output (pitch metres).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.structural_metrics import PITCH_LEN, PITCH_WID

MIN_GAP_FRAMES = 50    # ball untracked for at least this many frames = a likely out-of-play stoppage
RELIABLE_TYPES = ("corner", "throw_in", "goal_kick")  # boundary-anchored = trustworthy; free_kick is a guess
SETTLE_MAX_MS = 6.0    # the re-entry should not be a fast fly-through (m/s) to count as a restart
CORNER_M = 4.0         # within this of a pitch corner -> corner
TOUCHLINE_M = 1.5      # within this of a touchline -> throw-in
SIX_YARD = 5.5         # six-yard box depth (m)
GOAL_HALF_WID = 9.16   # half-width of the six-yard box region (m)


def _classify(x: float, y: float) -> str:
    """Label a dead-ball location as corner / throw_in / goal_kick / free_kick."""
    near_x_end = min(x, PITCH_LEN - x) <= CORNER_M
    near_y_end = min(y, PITCH_WID - y) <= CORNER_M
    if near_x_end and near_y_end:
        return "corner"
    if min(y, PITCH_WID - y) <= TOUCHLINE_M:
        return "throw_in"
    if min(x, PITCH_LEN - x) <= SIX_YARD and abs(y - PITCH_WID / 2) <= GOAL_HALF_WID:
        return "goal_kick"
    return "free_kick"


def ball_speed(ball: pd.DataFrame, *, fps: float = 50.0) -> pd.DataFrame:
    """Add a ``speed_ms`` column (pitch m/s between consecutive ball samples)."""
    b = ball.sort_values("frame").reset_index(drop=True)
    if len(b) < 2:
        return b.assign(speed_ms=np.nan)
    df = np.diff(b["frame"].to_numpy()) / fps
    dx, dy = np.diff(b["x"].to_numpy()), np.diff(b["y"].to_numpy())
    speed = np.concatenate([[np.nan], np.hypot(dx, dy) / np.where(df > 0, df, np.nan)])
    return b.assign(speed_ms=speed)


def detect_set_pieces(ball: pd.DataFrame, *, fps: float = 50.0, min_gap_frames: int = MIN_GAP_FRAMES,
                      settle_max_ms: float = SETTLE_MAX_MS) -> pd.DataFrame:
    """Detect dead-ball restarts as ball re-entries after an out-of-play gap.

    Finds frame gaps ``> min_gap_frames`` in the ball track (ball out of play / untracked); the first
    sample after each gap is a restart candidate, classified by :func:`_classify`. Fast fly-through
    re-entries (speed ``> settle_max_ms``) are dropped -- a restart is taken from a settled ball.

    Returns:
        ``frame, type, x, y, gap_frames`` -- one row per detected restart (``frame`` = the re-entry).
    """
    cols = ["frame", "type", "x", "y", "gap_frames"]
    bs = ball_speed(ball, fps=fps)
    if len(bs) < 2:
        return pd.DataFrame(columns=cols)
    fr = bs["frame"].to_numpy()
    xs, ys, sp = bs["x"].to_numpy(), bs["y"].to_numpy(), bs["speed_ms"].to_numpy()
    gaps = np.diff(fr)
    rows = []
    for k in np.where(gaps > min_gap_frames)[0]:
        i = k + 1  # the re-entry sample
        if not np.isnan(sp[i]) and sp[i] > settle_max_ms:
            continue
        rows.append({"frame": int(fr[i]), "type": _classify(float(xs[i]), float(ys[i])),
                     "x": float(xs[i]), "y": float(ys[i]), "gap_frames": int(gaps[k])})
    return pd.DataFrame(rows, columns=cols)


def set_piece_summary(events: pd.DataFrame) -> dict[str, int]:
    """Counts per set-piece type, plus ``reliable`` (boundary-anchored) / ``candidate`` (free_kick) totals.

    ``reliable`` sums the boundary-defined classes (corner/throw_in/goal_kick), which the heuristic gets
    right; ``candidate`` is the free-kick residual (open-play re-entries, dominated by tracking dropouts)
    -- report it as low-confidence, not a hard count.
    """
    if events.empty:
        return {"total": 0, "reliable": 0, "candidate": 0}
    out = {k: int(v) for k, v in events["type"].value_counts().to_dict().items()}
    out["reliable"] = int(sum(out.get(t, 0) for t in RELIABLE_TYPES))
    out["candidate"] = int(out.get("free_kick", 0))
    out["total"] = int(len(events))
    return out


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ball", required=True, help="linked ball track parquet (frame,x,y)")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()
    ev = detect_set_pieces(pd.read_parquet(args.ball), fps=args.fps)
    print("Set-piece restarts detected:\n")
    print(ev.round(1).to_string(index=False) if not ev.empty else "  (none)")
    print("\nsummary:", set_piece_summary(ev))


if __name__ == "__main__":
    main()
