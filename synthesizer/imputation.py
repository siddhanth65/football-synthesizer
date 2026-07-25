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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Load a Metrica match into metre-scaled truth arrays plus ball and period.

    Args:
        home_csv: Home-team ``*_RawTrackingData_Home_Team.csv``.
        away_csv: Away-team ``*_RawTrackingData_Away_Team.csv``.

    Returns:
        Tuple ``(truth_m, ball_m, period, fps, n_home)``: ``truth_m`` shape
        ``(n_frames, n_slots, 2)`` in metres (NaN where a slot is off the pitch),
        ``ball_m`` shape ``(n_frames, 2)`` in metres (NaN when untracked), ``period`` the
        per-frame period id, ``fps`` the measured frame rate, and ``n_home`` the number of
        home slots (slots ``0:n_home`` are home, the rest away).
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
    return truth_m, ball_m, period[:n], fps, home_xy.shape[1]


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


SPEED_CAP = 12.0  # m/s clamp on the extrapolation velocity (well above sprint speed)


def team_ranges(n_home: int, n_slots: int) -> tuple[tuple[int, int], ...]:
    """Return ``((0, n_home), (n_home, n_slots))`` slot ranges for home/away."""
    return (0, n_home), (n_home, n_slots)


def estimate_velocity(
    truth_m: np.ndarray, period: np.ndarray, fps: float, win_s: float = 0.5
) -> np.ndarray:
    """Per-frame velocity from a trailing ``win_s`` window, speed-capped and period-safe.

    Velocity is a finite difference over the last ``win_s`` seconds of observed motion.
    Just before a player leaves frame they were visible, so ``truth`` around the last-seen
    index is genuine observed trajectory, not off-screen truth peeking.

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)``.
        period: Per-frame period id, shape ``(n,)``.
        fps: Frame rate.
        win_s: Look-back window in seconds.

    Returns:
        Velocity array in m/s, shape ``(n, slots, 2)`` (NaN before the window is available
        or across a period boundary), clamped to ``SPEED_CAP``.
    """
    n = truth_m.shape[0]
    k = max(1, int(round(win_s * fps)))
    vel = np.full_like(truth_m, np.nan)
    if n <= k:
        return vel
    raw = (truth_m[k:] - truth_m[:-k]) / (k / fps)
    same = period[k:] == period[:-k]
    raw[~same] = np.nan
    vel[k:] = raw
    speed = np.linalg.norm(vel, axis=2)
    over = speed > SPEED_CAP
    with np.errstate(invalid="ignore"):
        scale = np.where(over, SPEED_CAP / np.where(speed == 0, 1.0, speed), 1.0)
    vel *= scale[:, :, None]
    return vel


def visible_centroid(
    truth_m: np.ndarray, visible: np.ndarray, ranges: tuple[tuple[int, int], ...]
) -> np.ndarray:
    """Per-frame centroid of each slot's *visible* teammates, excluding the slot itself.

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)``.
        visible: Visibility mask, shape ``(n, slots)``.
        ranges: Home/away slot ranges from :func:`team_ranges`.

    Returns:
        Centroid array, shape ``(n, slots, 2)`` (NaN where a slot has no visible teammate).
    """
    n, slots, _ = truth_m.shape
    cent = np.full((n, slots, 2), np.nan)
    for lo, hi in ranges:
        pos = truth_m[:, lo:hi, :]
        vis = visible[:, lo:hi]
        vpos = np.where(vis[:, :, None], pos, 0.0)
        sum_all = vpos.sum(axis=1)
        cnt_all = vis.sum(axis=1)
        for j in range(lo, hi):
            self_pos = np.where(visible[:, j, None], truth_m[:, j, :], 0.0)
            denom = cnt_all - visible[:, j].astype(float)
            num = sum_all - self_pos
            ok = denom > 0
            c = np.full((n, 2), np.nan)
            c[ok] = num[ok] / denom[ok, None]
            cent[:, j, :] = c
    return cent


def fit_slot_model(
    truth_m: np.ndarray,
    cent: np.ndarray,
    cam: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
) -> dict[tuple[str, int], np.ndarray]:
    """Fit per-(side, local-slot) OLS mapping visible structure + ball -> player position.

    Features per frame are ``[centroid_x, centroid_y, ball_x, ball_y, 1]`` (``cam`` is the
    smoothed/gap-filled ball). Target is the player's true position. Fit on all frames where
    the target is active and has >=1 visible teammate. Frozen for held-out inference.

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)``.
        cent: Visible-teammate centroid, shape ``(n, slots, 2)``.
        cam: Smoothed ball track in metres, shape ``(n, 2)``.
        ranges: Home/away slot ranges.

    Returns:
        Dict ``(side, local_idx) -> coef`` where ``coef`` has shape ``(5, 2)``.
    """
    coeffs: dict[tuple[str, int], np.ndarray] = {}
    for si, (lo, hi) in enumerate(ranges):
        side = "home" if si == 0 else "away"
        for j in range(lo, hi):
            ok = np.isfinite(cent[:, j, 0]) & np.isfinite(truth_m[:, j, 0])
            if ok.sum() < 50:
                continue
            feat = np.column_stack(
                [cent[ok, j, 0], cent[ok, j, 1], cam[ok, 0], cam[ok, 1], np.ones(ok.sum())]
            )
            coef, *_ = np.linalg.lstsq(feat, truth_m[ok, j, :], rcond=None)
            coeffs[(side, j - lo)] = coef
    return coeffs


def slot_prediction(
    cent: np.ndarray,
    cam: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
    coeffs: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    """Apply a fitted slot model to produce a per-frame position field.

    Args:
        cent: Visible-teammate centroid, shape ``(n, slots, 2)``.
        cam: Smoothed ball track, shape ``(n, 2)``.
        ranges: Home/away slot ranges.
        coeffs: Fitted coefficients from :func:`fit_slot_model`.

    Returns:
        Predicted positions, shape ``(n, slots, 2)`` (NaN where no visible teammate or no
        fitted slot).
    """
    n, slots, _ = cent.shape
    pred = np.full((n, slots, 2), np.nan)
    feat_ball = np.column_stack([cam[:, 0], cam[:, 1], np.ones(n)])
    for si, (lo, hi) in enumerate(ranges):
        side = "home" if si == 0 else "away"
        for j in range(lo, hi):
            coef = coeffs.get((side, j - lo))
            if coef is None:
                continue
            ok = np.isfinite(cent[:, j, 0])
            feat = np.column_stack([cent[ok, j, 0], cent[ok, j, 1], feat_ball[ok]])
            pred[ok, j, :] = feat @ coef
    return pred


def collect_samples(
    truth_m: np.ndarray,
    visible: np.ndarray,
    period: np.ndarray,
    fps: float,
    vel: np.ndarray,
    slot_pred: np.ndarray,
    fields: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Gather per-hidden-sample states shared by every baseline.

    For each active, currently hidden slot-frame with a prior sighting this period, record
    time-since-last-seen and the pieces each baseline needs: last-seen position (hold),
    offline-linear position, last-seen velocity, slot-model position (fallback to hold), and
    the truth target.

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)``.
        visible: Visibility mask, shape ``(n, slots)``.
        period: Per-frame period id.
        fps: Frame rate.
        vel: Per-frame velocity from :func:`estimate_velocity`.
        slot_pred: Per-frame slot-model position field from :func:`slot_prediction`.
        fields: Optional extra per-frame position fields, shape ``(n, slots, 2)`` each,
            sampled at the query frame (e.g. the ``B6_vote`` field).

    Returns:
        Dict of aligned 1-D/2-D arrays: ``tsls`` (s), ``target``, ``hold``, ``linear``,
        ``v0``, ``slot`` (all metre positions, shape ``(M, 2)`` except ``tsls``), plus the
        bookkeeping columns ``frame`` (global frame index), ``slot_id``, ``last_frame``
        (global index of the last sighting) and ``period``. Any per-frame field passed in
        ``fields`` is sampled at the query frame and added under its own key (NaN rows fall
        back to the hold position).
    """
    keys = ("tsls", "target", "hold", "linear", "v0", "slot")
    idx_keys = ("frame", "slot_id", "last_frame", "period")
    fields = fields or {}
    acc: dict[str, list[np.ndarray]] = {k: [] for k in (*keys, *idx_keys, *fields)}
    slots = truth_m.shape[1]
    for per in np.unique(period):
        sel = period == per
        gidx = np.where(sel)[0]
        tr_p = truth_m[sel]
        vis_p = visible[sel]
        vel_p = vel[sel]
        slot_p = slot_pred[sel]
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
            v0 = vel_p[li, p]
            v0 = np.where(np.isfinite(v0), v0, 0.0)
            slot_pos = slot_p[hi, p]
            slot_pos = np.where(np.isfinite(slot_pos), slot_pos, hold_pos)
            acc["tsls"].append((hi - li) / fps)
            acc["target"].append(target)
            acc["hold"].append(hold_pos)
            acc["linear"].append(lin_pos)
            acc["v0"].append(v0)
            acc["slot"].append(slot_pos)
            acc["frame"].append(gidx[hi])
            acc["slot_id"].append(np.full(hi.size, p))
            acc["last_frame"].append(gidx[li])
            acc["period"].append(np.full(hi.size, per))
            for name, fld in fields.items():
                pos = fld[gidx[hi], p]
                acc[name].append(np.where(np.isfinite(pos), pos, hold_pos))
    return {k: np.concatenate(v) for k, v in acc.items()}


def veldecay_position(base: np.ndarray, v0: np.ndarray, tsls: np.ndarray, tau: float) -> np.ndarray:
    """Extrapolate with velocity whose speed decays exponentially toward zero.

    Displacement is the integral of ``v0 * exp(-t / tau)``: ``v0 * tau * (1 - exp(-t/tau))``,
    a bounded drift of ``v0 * tau`` as ``t -> inf``.

    Args:
        base: Last-seen positions, shape ``(M, 2)``.
        v0: Last-seen velocities m/s, shape ``(M, 2)``.
        tsls: Time-since-last-seen in seconds, shape ``(M,)``.
        tau: Decay time-constant in seconds.

    Returns:
        Predicted positions, shape ``(M, 2)``.
    """
    factor = tau * (1.0 - np.exp(-tsls / tau))
    return base + v0 * factor[:, None]


def _bucket_index(tsls: np.ndarray) -> np.ndarray:
    """Map each time-since-seen to its horizon-bucket index (0..len(BIN_LABELS)-1)."""
    idx = np.zeros(tsls.shape, dtype=int)
    for i in range(1, len(BIN_LABELS)):
        idx[tsls >= BIN_EDGES_S[i]] = i
    return idx


def fit_decay_tau(
    samples: dict[str, np.ndarray], grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Grid-search the decay tau minimising overall B3 RMSE (in-sample game only).

    Args:
        samples: Output of :func:`collect_samples`.
        grid: Candidate tau values (s); default 0.25..10.0 step 0.25.

    Returns:
        Tuple ``(best_tau, best_rmse)`` in seconds and metres.
    """
    if grid is None:
        grid = np.arange(0.25, 10.01, 0.25)
    base, v0, tsls, tgt = samples["hold"], samples["v0"], samples["tsls"], samples["target"]
    best_tau, best_rmse = float(grid[0]), float("inf")
    for tau in grid:
        pred = veldecay_position(base, v0, tsls, float(tau))
        rmse = float(np.sqrt(np.mean(np.sum((pred - tgt) ** 2, axis=1))))
        if rmse < best_rmse:
            best_tau, best_rmse = float(tau), rmse
    return best_tau, best_rmse


def fit_blend(
    samples: dict[str, np.ndarray], tau: float, grid: np.ndarray | None = None
) -> np.ndarray:
    """Fit a per-bucket linear blend weight ``w`` for ``w*B3 + (1-w)*B4`` (in-sample).

    Args:
        samples: Output of :func:`collect_samples`.
        tau: Frozen decay constant for B3.
        grid: Candidate weights in [0, 1]; default 0..1 step 0.05.

    Returns:
        Per-bucket weight array, shape ``(len(BIN_LABELS),)`` (weight on B3).
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    b3 = veldecay_position(samples["hold"], samples["v0"], samples["tsls"], tau)
    b4 = samples["slot"]
    tgt = samples["target"]
    bkt = _bucket_index(samples["tsls"])
    weights = np.zeros(len(BIN_LABELS))
    for b in range(len(BIN_LABELS)):
        m = bkt == b
        if not m.any():
            weights[b] = 1.0
            continue
        best_w, best_rmse = 1.0, float("inf")
        for w in grid:
            pred = w * b3[m] + (1.0 - w) * b4[m]
            rmse = float(np.sqrt(np.mean(np.sum((pred - tgt[m]) ** 2, axis=1))))
            if rmse < best_rmse:
                best_w, best_rmse = float(w), rmse
        weights[b] = best_w
    return weights


def baseline_errors(
    samples: dict[str, np.ndarray], tau: float, blend_w: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute per-sample error (m) for all five baselines plus the ``tsls`` axis.

    Args:
        samples: Output of :func:`collect_samples`.
        tau: Frozen decay constant.
        blend_w: Frozen per-bucket B3 weight from :func:`fit_blend`.

    Returns:
        Dict with ``tsls`` and error arrays ``B1_hold``, ``B2_offline``, ``B3_veldecay``,
        ``B4_slot``, ``B5_blend`` (metres).
    """
    tgt = samples["target"]

    def err(pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pos - tgt, axis=1)

    b3 = veldecay_position(samples["hold"], samples["v0"], samples["tsls"], tau)
    b4 = samples["slot"]
    w = blend_w[_bucket_index(samples["tsls"])][:, None]
    b5 = w * b3 + (1.0 - w) * b4
    out = {
        "tsls": samples["tsls"],
        "B1_hold": err(samples["hold"]),
        "B2_offline": err(samples["linear"]),
        "B3_veldecay": err(b3),
        "B4_slot": err(b4),
        "B5_blend": err(b5),
    }
    if "b6" in samples:
        out["B6_vote"] = err(samples["b6"])
    return out


def blend_position(samples: dict[str, np.ndarray], tau: float, blend_w: np.ndarray) -> np.ndarray:
    """Return the frozen ``B5_blend`` point prediction for each sample, shape ``(M, 2)``."""
    b3 = veldecay_position(samples["hold"], samples["v0"], samples["tsls"], tau)
    w = blend_w[_bucket_index(samples["tsls"])][:, None]
    return w * b3 + (1.0 - w) * samples["slot"]


BASELINE_ORDER = ("B1_hold", "B2_offline", "B3_veldecay", "B4_slot", "B5_blend")


def print_full_table(errors: dict[str, np.ndarray], title: str) -> None:
    """Print RMSE / p50 / p90 by horizon for every baseline present, and per-bucket winners."""
    tsls = errors["tsls"]
    order = tuple(b for b in (*BASELINE_ORDER, "B6_vote") if b in errors)
    stats = {b: _bin_stats(errors[b], tsls) for b in order}
    print("=" * 92)
    print(title)
    print("=" * 92)
    for metric in ("rmse", "p50", "p90"):
        print(f"[{metric.upper()} metres]")
        header = "horizon   |     n    | " + " ".join(f"{b:>11s}" for b in order)
        print(header)
        print("-" * len(header))
        for i, lab in enumerate(BIN_LABELS):
            n = stats["B1_hold"][i]["n"]
            cells = " ".join(f"{stats[b][i][metric]:11.2f}" for b in order)
            print(f"{lab:9s} | {n:8d} | {cells}")
        overall = " ".join(
            f"{np.sqrt(np.mean(errors[b] ** 2)) if metric == 'rmse' else np.percentile(errors[b], int(metric[1:])):11.2f}"
            for b in order
        )
        print(f"{'ALL':9s} | {tsls.size:8d} | {overall}")
        print()
    print("per-bucket winner (lowest RMSE):")
    for i, lab in enumerate(BIN_LABELS):
        if stats["B1_hold"][i]["n"] == 0:
            continue
        best = min(order, key=lambda b: stats[b][i]["rmse"])
        row = "  ".join(f"{b}={stats[b][i]['rmse']:5.2f}" for b in order)
        print(f"  {lab:6s} winner={best:11s} | {row}")
    print()


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
    # veldecay: constant 2 m/s, tau huge -> ~linear drift; tau tiny -> frozen near base.
    base = np.zeros((1, 2))
    v0 = np.array([[2.0, 0.0]])
    t = np.array([2.0])
    assert abs(veldecay_position(base, v0, t, tau=1e6)[0, 0] - 4.0) < 1e-3  # ~v0*t
    assert veldecay_position(base, v0, t, tau=1e-6)[0, 0] < 1e-3  # frozen
    print("self-check OK (index fills + linear interp exact + veldecay limits)")


def _prep_game(name: str, half_w: float | None) -> tuple[dict, np.ndarray, str]:
    """Load a game, censor it, and return the ingredients needed to fit or score.

    Args:
        name: Metrica game stem, e.g. ``"Sample_Game_1"``.
        half_w: Fixed window half-width (m); ``None`` tunes it on this game (train only).

    Returns:
        Tuple ``(bundle, half_w, summary)`` where ``bundle`` holds ``truth/visible/period/
        fps/cam/vel/cent/ranges`` and ``summary`` is a one-line stats string.
    """
    truth_m, ball_m, period, fps, n_home = load_metrica_match(
        METRICA / f"{name}_RawTrackingData_Home_Team.csv",
        METRICA / f"{name}_RawTrackingData_Away_Team.csv",
    )
    cam = smooth_camera(ball_m, fps)
    if half_w is None:
        half_w, _ = tune_window(truth_m, cam)
    visible = visibility_mask(truth_m, cam, half_w, PITCH_W / 2)
    ranges = team_ranges(n_home, truth_m.shape[1])
    vel = estimate_velocity(truth_m, period, fps)
    cent = visible_centroid(truth_m, visible, ranges)
    counts = visible.sum(axis=1)
    hidden = int(((~visible) & np.isfinite(truth_m[:, :, 0])).sum())
    summary = (
        f"{name}: {truth_m.shape[0]} fr, {truth_m.shape[1]} slots, fps={fps:.1f}, "
        f"visible/frame mean={counts.mean():.2f} (min={counts.min()} p50={int(np.median(counts))} "
        f"max={counts.max()}), hidden scoring pool={hidden}"
    )
    bundle = {
        "truth": truth_m, "visible": visible, "period": period, "fps": fps,
        "cam": cam, "vel": vel, "cent": cent, "ranges": ranges, "ball": ball_m,
    }
    return bundle, half_w, summary


def main() -> None:
    """Fit B3/B4/B5 on Game 1, freeze, and score all baselines on both games."""
    _self_check()
    g1, half_w, s1 = _prep_game("Sample_Game_1", half_w=None)
    print(f"window tuned on Game 1: {2 * half_w:.1f} m wide x {PITCH_W:.0f} m tall")
    print(s1)

    # Fit the slot model, decay tau, and blend weights on Game 1 ONLY.
    slot_coef = fit_slot_model(g1["truth"], g1["cent"], g1["cam"], g1["ranges"])
    slot_pred1 = slot_prediction(g1["cent"], g1["cam"], g1["ranges"], slot_coef)
    samp1 = collect_samples(
        g1["truth"], g1["visible"], g1["period"], g1["fps"], g1["vel"], slot_pred1
    )
    tau, tau_rmse = fit_decay_tau(samp1)
    blend_w = fit_blend(samp1, tau)
    print(f"\nFITTED (Game 1 only): decay tau={tau:.2f} s (B3 in-sample RMSE {tau_rmse:.2f} m)")
    print(f"blend weights on B3 per bucket {BIN_LABELS} = " f"{np.round(blend_w, 2).tolist()}")
    _report_slot_coef(slot_coef)

    err1 = baseline_errors(samp1, tau, blend_w)
    print()
    print_full_table(err1, "GAME 1  (IN-SAMPLE -- fit here)")

    # Score Game 2 held-out under the SAME window and frozen models.
    g2, _, s2 = _prep_game("Sample_Game_2", half_w=half_w)
    print(s2)
    slot_pred2 = slot_prediction(g2["cent"], g2["cam"], g2["ranges"], slot_coef)
    samp2 = collect_samples(
        g2["truth"], g2["visible"], g2["period"], g2["fps"], g2["vel"], slot_pred2
    )
    err2 = baseline_errors(samp2, tau, blend_w)
    print()
    print_full_table(err2, "GAME 2  (HELD-OUT -- this is the table that matters)")


def _report_slot_coef(coeffs: dict[tuple[str, int], np.ndarray]) -> None:
    """Print a compact summary of the fitted slot coefficients (magnitude of each term)."""
    print(f"slot model: {len(coeffs)} per-(side,slot) OLS fits, features "
          "[centroid_x, centroid_y, ball_x, ball_y, 1] -> (x, y)")
    cx = np.array([c[0, 0] for c in coeffs.values()])
    cbx = np.array([c[2, 0] for c in coeffs.values()])
    print(f"  x-on-centroid_x coef: mean={cx.mean():.2f} range[{cx.min():.2f},{cx.max():.2f}]")
    print(f"  x-on-ball_x    coef: mean={cbx.mean():.2f} range[{cbx.min():.2f},{cbx.max():.2f}]")


if __name__ == "__main__":
    main()
