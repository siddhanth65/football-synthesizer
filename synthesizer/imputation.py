"""B4 censoring simulator + off-screen imputation baselines, scored on Metrica truth.

Metrica open data is genuine full-pitch, 22-player, 25 fps tracking (audited in
``tools/imputation_audit.py``). Here we (1) simulate a broadcast camera as a moving
visibility window over that truth -- tuned so visible-players-per-frame matches our
measured broadcast stat (~11.8/22 on trusted frames) -- and (2) score the two
pre-registered baselines on the hidden (off-screen) players:

    B1_hold   : last-seen position held.
    B2_linear : linear interpolation between the last and next sighting (OFFLINE, the
                published-SOTA bar). Its ONLINE/causal variant has no future sighting, so
                it degenerates to hold == B1 -- reported explicitly.

Coordinates are converted to metres (pitch 105 x 68) so RMSE is in metres. Errors are
binned by time-since-last-seen. CPU-only, no torch.

Usage:
    python -m synthesizer.imputation
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
METRICA = REPO / "data" / "imputation" / "metrica"
PITCH_L = 105.0
PITCH_W = 68.0
TARGET_VISIBLE = 11.8  # mean visible players / 22 on trusted broadcast frames
BIN_EDGES_S = (0.0, 1.0, 3.0, 5.0, 10.0, 30.0, np.inf)
BIN_LABELS = ("0-1s", "1-3s", "3-5s", "5-10s", "10-30s", "30s+")


def load_metrica_match(
    home_csv: Path, away_csv: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a Metrica match into metre-scaled truth arrays plus ball and period.

    Args:
        home_csv: Home-team ``*_RawTrackingData_Home_Team.csv``.
        away_csv: Away-team ``*_RawTrackingData_Away_Team.csv``.

    Returns:
        Tuple ``(truth_m, ball_m, period, fps)``: ``truth_m`` shape
        ``(n_frames, n_slots, 2)`` in metres (NaN where a slot is off the pitch),
        ``ball_m`` shape ``(n_frames, 2)`` in metres (NaN when untracked), ``period`` the
        per-frame period id, and ``fps`` the measured frame rate.
    """
    home_xy, ball_h, period, time_s = _load_team(home_csv)
    away_xy, _, _, _ = _load_team(away_csv)
    n = min(home_xy.shape[0], away_xy.shape[0])
    truth = np.concatenate([home_xy[:n], away_xy[:n]], axis=1)
    scale = np.array([PITCH_L, PITCH_W])
    truth_m = truth * scale
    ball_m = ball_h[:n] * scale
    dt = np.diff(time_s[:n])
    fps = 1.0 / float(np.median(dt[np.isfinite(dt)]))
    return truth_m, ball_m, period[:n], fps


def _load_team(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one Metrica team CSV into ``(xy, ball_xy, period, time_s)`` (normalised)."""
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    name_row = rows[2]
    x_idx: list[int] = []
    ball_x = -1
    for idx in range(3, len(name_row), 2):
        label = name_row[idx].strip().lower()
        if label == "ball":
            ball_x = idx
        elif label:
            x_idx.append(idx)
    data = rows[3:]
    n = len(data)
    xy = np.full((n, len(x_idx), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    period = np.zeros(n, dtype=int)
    time_s = np.full(n, np.nan)
    for r, row in enumerate(data):
        period[r] = int(row[0]) if row[0].strip() else 0
        time_s[r] = _to_float(row[2])
        for p, col in enumerate(x_idx):
            xy[r, p, 0] = _to_float(row[col])
            xy[r, p, 1] = _to_float(row[col + 1])
        if ball_x >= 0:
            ball[r, 0] = _to_float(row[ball_x])
            ball[r, 1] = _to_float(row[ball_x + 1])
    return xy, ball, period, time_s


def _to_float(token: str) -> float:
    """Parse a CSV token to float, mapping blanks/``NaN`` to ``np.nan``."""
    token = token.strip()
    if not token or token.lower() == "nan":
        return float("nan")
    return float(token)


def smooth_camera(ball_m: np.ndarray, fps: float, win_s: float = 1.0) -> np.ndarray:
    """Build a smooth camera-centre track from the (gappy) ball position.

    Gaps are linearly interpolated across frame index, then a centred moving average of
    ``win_s`` seconds smooths broadcast-like panning.

    Args:
        ball_m: Ball positions in metres, shape ``(n, 2)``, NaN where untracked.
        fps: Frame rate.
        win_s: Moving-average window in seconds.

    Returns:
        Camera-centre positions in metres, shape ``(n, 2)``, no NaN.
    """
    n = ball_m.shape[0]
    frames = np.arange(n)
    cam = np.empty_like(ball_m)
    for ax in range(2):
        col = ball_m[:, ax]
        ok = np.isfinite(col)
        cam[:, ax] = np.interp(frames, frames[ok], col[ok])
    w = max(1, int(round(win_s * fps)))
    kernel = np.ones(w) / w
    for ax in range(2):
        cam[:, ax] = np.convolve(cam[:, ax], kernel, mode="same")
    return cam


def visibility_mask(
    truth_m: np.ndarray, cam: np.ndarray, half_w: float, half_h: float
) -> np.ndarray:
    """Mark which players fall inside the axis-aligned broadcast window each frame.

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)``.
        cam: Camera centres in metres, shape ``(n, 2)``.
        half_w: Half window width along the pitch length (x), metres.
        half_h: Half window height along the pitch width (y), metres.

    Returns:
        Boolean visibility mask, shape ``(n, slots)`` (False where a slot is inactive).
    """
    dx = np.abs(truth_m[:, :, 0] - cam[:, None, 0])
    dy = np.abs(truth_m[:, :, 1] - cam[:, None, 1])
    inside = (dx <= half_w) & (dy <= half_h)
    return inside & np.isfinite(truth_m[:, :, 0])


def tune_window(
    truth_m: np.ndarray,
    cam: np.ndarray,
    target: float = TARGET_VISIBLE,
    half_h: float = PITCH_W / 2,
) -> tuple[float, float]:
    """Bisect the window half-width so mean visible players/frame hits ``target``.

    Height is fixed to full pitch (broadcast pans horizontally; vertical is a knob left
    for later calibration). Returns the tuned half-width and the achieved mean.

    Args:
        truth_m: Truth positions in metres.
        cam: Camera centres in metres.
        target: Desired mean visible players per frame (out of 22).
        half_h: Fixed window half-height in metres.

    Returns:
        Tuple ``(half_w, achieved_mean)``.
    """
    lo, hi = 1.0, PITCH_L
    achieved = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        achieved = float(visibility_mask(truth_m, cam, mid, half_h).sum(axis=1).mean())
        if achieved < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), achieved


def _last_visible_idx(visible: np.ndarray) -> np.ndarray:
    """Index of the most recent visible frame at each position (-1 before first)."""
    n = visible.size
    idx = np.where(visible, np.arange(n), -1)
    return np.maximum.accumulate(idx)


def _next_visible_idx(visible: np.ndarray) -> np.ndarray:
    """Index of the next visible frame at each position (``n`` after last)."""
    n = visible.size
    idx = np.where(visible, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def score_baselines(
    truth_m: np.ndarray, visible: np.ndarray, period: np.ndarray, fps: float
) -> dict[str, np.ndarray]:
    """Estimate hidden players with B1/B2 and collect per-sample error + horizon.

    Scoring is restricted to the extrapolation regime: frames where a slot is active
    (truth known), currently hidden, and has a prior sighting this period. Interpolation
    never crosses the halftime break (periods are processed independently).

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)``.
        visible: Visibility mask, shape ``(n, slots)``.
        period: Per-frame period id, shape ``(n,)``.
        fps: Frame rate.

    Returns:
        Dict with ``tsls`` (seconds since last seen), ``hold`` and ``linear`` error
        arrays (metres), all 1-D and aligned.
    """
    tsls_all: list[np.ndarray] = []
    hold_all: list[np.ndarray] = []
    lin_all: list[np.ndarray] = []
    slots = truth_m.shape[1]
    for per in np.unique(period):
        sel = period == per
        tr_p = truth_m[sel]
        vis_p = visible[sel]
        m = tr_p.shape[0]
        for p in range(slots):
            vis = vis_p[:, p]
            if not vis.any():
                continue
            tr = tr_p[:, p]
            finite = np.isfinite(tr[:, 0])
            last = _last_visible_idx(vis)
            nxt = _next_visible_idx(vis)
            hidden = (~vis) & finite & (last >= 0)
            hi = np.where(hidden)[0]
            if hi.size == 0:
                continue
            li = last[hi]
            ni = nxt[hi]
            target = tr[hi]
            hold_pos = tr[li]
            lin_pos = hold_pos.copy()
            has_next = ni < m
            if has_next.any():
                w = (hi[has_next] - li[has_next]) / (ni[has_next] - li[has_next])
                lin_pos[has_next] = (
                    tr[li[has_next]] * (1 - w[:, None]) + tr[ni[has_next]] * w[:, None]
                )
            tsls_all.append((hi - li) / fps)
            hold_all.append(np.linalg.norm(hold_pos - target, axis=1))
            lin_all.append(np.linalg.norm(lin_pos - target, axis=1))
    return {
        "tsls": np.concatenate(tsls_all),
        "hold": np.concatenate(hold_all),
        "linear": np.concatenate(lin_all),
    }


def _bin_stats(err: np.ndarray, tsls: np.ndarray) -> list[dict[str, float]]:
    """Compute per-horizon-bin n / RMSE / p50 / p90 of an error array (metres)."""
    out: list[dict[str, float]] = []
    for i in range(len(BIN_LABELS)):
        m = (tsls >= BIN_EDGES_S[i]) & (tsls < BIN_EDGES_S[i + 1])
        e = err[m]
        if e.size:
            out.append(
                {
                    "n": int(e.size),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "p50": float(np.percentile(e, 50)),
                    "p90": float(np.percentile(e, 90)),
                }
            )
        else:
            out.append({"n": 0, "rmse": float("nan"), "p50": float("nan"), "p90": float("nan")})
    return out


def print_table(scores: dict[str, np.ndarray]) -> None:
    """Print the ASCII RMSE-by-horizon table for B1 and B2 (offline)."""
    hold = _bin_stats(scores["hold"], scores["tsls"])
    lin = _bin_stats(scores["linear"], scores["tsls"])
    print(
        "horizon   |     n    | B1_hold  (=B2_online)     | B2_linear (offline)       "
    )
    print(
        "          |          | rmse    p50    p90        | rmse    p50    p90        "
    )
    print("-" * 74)
    for i, lab in enumerate(BIN_LABELS):
        h, ln = hold[i], lin[i]
        print(
            f"{lab:9s} | {h['n']:8d} | {h['rmse']:6.2f} {h['p50']:6.2f} {h['p90']:6.2f}    "
            f"   | {ln['rmse']:6.2f} {ln['p50']:6.2f} {ln['p90']:6.2f}"
        )
    print("-" * 74)
    tot_h = np.sqrt(np.mean(scores["hold"] ** 2))
    tot_l = np.sqrt(np.mean(scores["linear"] ** 2))
    print(
        f"{'ALL':9s} | {scores['hold'].size:8d} | {tot_h:6.2f} (rmse)"
        f"              |   {tot_l:6.2f} (rmse)"
    )
    print("units = metres. B2_online (causal, no future sighting) degenerates to B1_hold.")


def _self_check() -> None:
    """Assert linear interpolation and index fills behave on a tiny hand-checked case."""
    vis = np.array([True, False, False, False, True])
    assert list(_last_visible_idx(vis)) == [0, 0, 0, 0, 4]
    assert list(_next_visible_idx(vis)) == [0, 4, 4, 4, 4]
    # slot moves 0->4 m in x over 4 frames; frame 2 truth = 2 m; linear should nail it.
    truth = np.zeros((5, 1, 2))
    truth[:, 0, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    period = np.ones(5, dtype=int)
    sc = score_baselines(truth, vis[:, None], period, fps=1.0)
    # hidden frames 1,2,3: hold err = distance to frame0 = 1,2,3; linear err ~ 0.
    assert np.allclose(np.sort(sc["hold"]), [1.0, 2.0, 3.0])
    assert np.max(sc["linear"]) < 1e-9
    print("self-check OK (index fills + linear interp exact)")


def main() -> None:
    """Load Metrica, tune the window to ~11.8/22, run baselines, print the table."""
    _self_check()
    truth_m, ball_m, period, fps = load_metrica_match(
        METRICA / "Sample_Game_1_RawTrackingData_Home_Team.csv",
        METRICA / "Sample_Game_1_RawTrackingData_Away_Team.csv",
    )
    print(f"loaded {truth_m.shape[0]} frames, {truth_m.shape[1]} slots, fps={fps:.1f}")
    cam = smooth_camera(ball_m, fps)
    half_w, achieved = tune_window(truth_m, cam)
    visible = visibility_mask(truth_m, cam, half_w, PITCH_W / 2)
    counts = visible.sum(axis=1)
    print(
        f"window: {2 * half_w:.1f} m wide x {PITCH_W:.0f} m tall (full height) -> "
        f"visible/frame mean={achieved:.2f} (target {TARGET_VISIBLE})"
    )
    print(
        f"visible/frame dist: min={counts.min()} p10={np.percentile(counts, 10):.0f} "
        f"p50={int(np.median(counts))} p90={np.percentile(counts, 90):.0f} max={counts.max()}"
    )
    hidden_active = ((~visible) & np.isfinite(truth_m[:, :, 0])).sum()
    print(f"hidden active player-frames (scoring pool)={hidden_active}")
    scores = score_baselines(truth_m, visible, period, fps)
    print()
    print_table(scores)


if __name__ == "__main__":
    main()
