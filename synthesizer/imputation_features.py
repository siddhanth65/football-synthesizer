"""B4 week-1 scaffolding: role-anchored vote baseline, v1 features, conformal regions.

Three pieces, all CPU-only and dependency-free beyond numpy:

* :func:`role_offsets_vote` / :func:`vote_field` -- the training-free ``B6_vote`` estimator of
  arXiv 2607.11548 (each visible player votes for the full-team centroid by subtracting its own
  EMA role offset; a hidden player is placed at ``voted_centroid + own EMA offset``). Zero learned
  parameters, current-match-only, strictly causal.
* :func:`derive_fields` + :func:`build_features` -- the ~36-column attack-aligned feature matrix of
  ``docs/B4_MODEL_PLAN.md`` section 2.1. Every structural column is computed from VISIBLE players
  only, which :func:`tests.test_imputation_features` verifies by recomputing everything from a
  truth array whose hidden entries are NaN and demanding bit-identical output.
* :func:`region_scale` / :func:`conformal_k` / :func:`coverage_table` -- the uncertainty-only
  wrapper (plan section 2.2/3.1): per-bucket elliptical predictive regions around a FROZEN point
  prediction, with a split-conformal correction, plus PICP/MPIW scoring.

Nothing here fits a point predictor. The B5_blend anchor is used exactly as committed.
"""

from __future__ import annotations

import warnings

import numpy as np

from synthesizer.imputation import (
    BIN_LABELS,
    PITCH_L,
    PITCH_W,
    _bucket_index,
    estimate_velocity,
    slot_prediction,
    visible_centroid,
)

CENTRE = np.array([PITCH_L / 2.0, PITCH_W / 2.0])
VOTE_HALFLIFE_S = 10.0  # EMA half-life for role offsets; selected on TRAIN (Game 1) only.
WIDTH_FLOOR_M = 0.25  # plan 3.1 step 2: floor the per-axis half-width so the score is finite.


# --------------------------------------------------------------------------------------
# B6_vote: role-anchored centroid vote (training-free)
# --------------------------------------------------------------------------------------


def role_offsets_vote(
    truth_m: np.ndarray,
    visible: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
    fps: float,
    halflife_s: float = VOTE_HALFLIFE_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the causal EMA role-offset / centroid-vote recursion over a match.

    At every frame each visible player ``i`` votes for the full-team centroid with
    ``pos_i - offset_i``; the vote is the mean of those votes. Each visible player's offset is
    then nudged toward ``pos_i - voted_centroid`` by an EMA step. Offsets of hidden players are
    frozen (never updated from unseen truth), so a hidden player's estimate uses only the offset
    it carried when it was last on camera. The prediction is invariant to a global shift of all
    offsets, so no re-centering is needed.

    Args:
        truth_m: Truth positions in metres, shape ``(n, slots, 2)`` (NaN where inactive).
        visible: Visibility mask, shape ``(n, slots)``.
        ranges: Home/away slot ranges (contiguous), from ``imputation.team_ranges``.
        fps: Frame rate.
        halflife_s: EMA half-life in seconds.

    Returns:
        Tuple ``(off, vote, count)``: ``off`` shape ``(n, slots, 2)`` role offsets, ``vote``
        shape ``(n, n_teams, 2)`` voted full-team centroids (NaN until a team is first seen),
        ``count`` shape ``(n, n_teams)`` number of voters.
    """
    n, slots, _ = truth_m.shape
    starts = np.array([lo for lo, _ in ranges])
    n_team = len(ranges)
    alpha = 1.0 - 0.5 ** (1.0 / max(1.0, halflife_s * fps))
    vis_f = visible.astype(np.float64)
    pos0 = np.where(visible[:, :, None], truth_m, 0.0)
    sum_pos = np.add.reduceat(pos0, starts, axis=1)
    count = np.add.reduceat(vis_f, starts, axis=1)
    team_of = np.zeros(slots, dtype=int)
    for si, (lo, hi) in enumerate(ranges):
        team_of[lo:hi] = si

    off = np.zeros((n, slots, 2), dtype=np.float32)
    vote = np.full((n, n_team, 2), np.nan)
    cur = np.zeros((slots, 2))
    last = np.full((n_team, 2), np.nan)
    for t in range(n):
        vt = vis_f[t][:, None]
        off_sum = np.add.reduceat(cur * vt, starts, axis=0)
        cnt = count[t]
        ok = cnt > 0
        voted = last.copy()
        voted[ok] = (sum_pos[t][ok] - off_sum[ok]) / cnt[ok, None]
        last = voted
        vote[t] = voted
        step = np.where(visible[t][:, None], truth_m[t] - voted[team_of] - cur, 0.0)
        cur = cur + alpha * np.nan_to_num(step)
        off[t] = cur
    return off, vote, count


def vote_field(
    off: np.ndarray, vote: np.ndarray, ranges: tuple[tuple[int, int], ...]
) -> np.ndarray:
    """Expand vote + role offsets into the per-frame ``B6_vote`` position field.

    Args:
        off: Role offsets from :func:`role_offsets_vote`, shape ``(n, slots, 2)``.
        vote: Voted centroids, shape ``(n, n_teams, 2)``.
        ranges: Home/away slot ranges.

    Returns:
        Position field, shape ``(n, slots, 2)`` (NaN before a team is first seen).
    """
    team_of = np.zeros(off.shape[1], dtype=int)
    for si, (lo, hi) in enumerate(ranges):
        team_of[lo:hi] = si
    return vote[:, team_of, :] + off


# --------------------------------------------------------------------------------------
# attack-aligned frame + causal observables
# --------------------------------------------------------------------------------------


def attack_signs(
    truth_m: np.ndarray,
    visible: np.ndarray,
    period: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], int]:
    """Infer each team's attacking direction per period from its deepest visible player.

    The slot whose mean visible x is furthest from the halfway line is the keeper; the team
    attacks away from that end. Computed from visible frames only, so it is unchanged if hidden
    entries are blanked (leakage guard).

    Args:
        truth_m: Truth positions in metres.
        visible: Visibility mask.
        period: Per-frame period id.
        ranges: Home/away slot ranges.

    Returns:
        Dict ``(period, team_index) -> +1`` (attacks toward increasing x) or ``-1``.
    """
    signs: dict[tuple[int, int], int] = {}
    pos = np.where(visible[:, :, None], truth_m, np.nan)
    for per in np.unique(period):
        sel = period == per
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_x = np.nanmean(pos[sel, :, 0], axis=0)
        for si, (lo, hi) in enumerate(ranges):
            block = mean_x[lo:hi]
            dev = np.abs(block - CENTRE[0])
            dev = np.where(np.isfinite(dev), dev, -1.0)
            keeper_x = block[int(np.argmax(dev))]
            signs[(int(per), si)] = 1 if keeper_x < CENTRE[0] else -1
    return signs


def align_pos(pos: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Rotate positions 180 degrees about the pitch centre where ``sign`` is -1."""
    return CENTRE + sign[:, None] * (pos - CENTRE)


def align_vec(vec: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Rotate displacement vectors 180 degrees where ``sign`` is -1."""
    return sign[:, None] * vec


def ffill_ball(ball_m: np.ndarray) -> np.ndarray:
    """Causally forward-fill ball gaps (leading gaps take the first observed position).

    Args:
        ball_m: Ball positions in metres, shape ``(n, 2)`` with NaN gaps.

    Returns:
        Gap-free ball track, shape ``(n, 2)``. Unlike ``imputation.smooth_camera`` this uses no
        future frames, so it is safe as a model input.
    """
    n = ball_m.shape[0]
    ok = np.isfinite(ball_m[:, 0])
    idx = np.maximum.accumulate(np.where(ok, np.arange(n), -1))
    first = int(np.argmax(ok)) if ok.any() else 0
    idx = np.where(idx < 0, first, idx)
    out = ball_m[idx].copy()
    if not ok.any():
        out[:] = CENTRE
    return out


def _team_stats(
    truth_m: np.ndarray,
    visible: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
    sign_frame: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-frame per-team visible-only structure in the attack-aligned frame.

    Args:
        truth_m: Truth positions in metres.
        visible: Visibility mask.
        ranges: Home/away slot ranges.
        sign_frame: Attack sign per frame per team, shape ``(n, n_teams)``.

    Returns:
        Dict with ``cent`` ``(n, n_teams, 2)``, ``spread`` ``(n, n_teams, 2)``, ``defline``
        ``(n, n_teams)`` and ``count`` ``(n, n_teams)``.
    """
    n = truth_m.shape[0]
    n_team = len(ranges)
    pos = np.where(visible[:, :, None], truth_m, np.nan)
    cent = np.full((n, n_team, 2), np.nan)
    spread = np.full((n, n_team, 2), np.nan)
    defline = np.full((n, n_team), np.nan)
    count = np.zeros((n, n_team))
    for si, (lo, hi) in enumerate(ranges):
        block = pos[:, lo:hi, :]
        sgn = sign_frame[:, si]
        aligned = CENTRE + sgn[:, None, None] * (block - CENTRE)
        cnt = np.isfinite(block[:, :, 0]).sum(axis=1)
        count[:, si] = cnt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cent[:, si, :] = np.nanmean(aligned, axis=1)
            spread[:, si, :] = np.nanstd(aligned, axis=1)
            xs = np.sort(aligned[:, :, 0], axis=1)  # NaN sorts last
        defline[:, si] = np.where(cnt >= 3, xs[:, 1], xs[:, 0])
    return {"cent": cent, "spread": spread, "defline": defline, "count": count}


# --------------------------------------------------------------------------------------
# feature builder
# --------------------------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    # last-seen memory (8)
    "last_x", "last_y", "v0_x", "v0_y", "speed0", "tsls", "log1p_tsls", "dist_last_boundary",
    # ball (7)
    "ball_x", "ball_y", "ball_last_x", "ball_last_y", "ball_disp_x", "ball_disp_y",
    "dist_last_ball_now",
    # visible-team structure (11)
    "own_cent_x", "own_cent_y", "own_cent_disp_x", "own_cent_disp_y", "opp_cent_x", "opp_cent_y",
    "own_spread_x", "own_spread_y", "own_defline_x", "n_vis_teammates", "n_vis_opponents",
    # role (5)
    "role_ord", "slot_x", "slot_y", "off_role_x", "off_role_y",
    # role-anchored centroid vote (3)
    "vote_x", "vote_y", "vote_n",
    # anchors (2)
    "blend_x", "blend_y",
)


def derive_fields(
    truth_m: np.ndarray,
    visible: np.ndarray,
    ball_m: np.ndarray,
    cam: np.ndarray,
    period: np.ndarray,
    fps: float,
    ranges: tuple[tuple[int, int], ...],
    slot_coeffs: dict[tuple[str, int], np.ndarray],
    halflife_s: float = VOTE_HALFLIFE_S,
) -> dict[str, object]:
    """Derive every per-frame field the feature builder reads, from visible players only.

    Velocity is recomputed from a visibility-masked copy of the truth (a hidden player's motion
    is not observable), which is what makes the leakage guard bit-exact.

    Args:
        truth_m: Truth positions in metres.
        visible: Visibility mask.
        ball_m: Raw ball track in metres (NaN gaps) -- an external observable.
        cam: Smoothed ball/camera track used by the frozen slot model.
        period: Per-frame period id.
        fps: Frame rate.
        ranges: Home/away slot ranges.
        slot_coeffs: Frozen slot-model coefficients fitted on TRAIN.
        halflife_s: EMA half-life for the role offsets.

    Returns:
        Dict of per-frame fields consumed by :func:`build_features` (plus ``b6`` and ``vote_n``
        fields usable as standalone baseline inputs).
    """
    masked = np.where(visible[:, :, None], truth_m, np.nan)
    signs = attack_signs(truth_m, visible, period, ranges)
    n_team = len(ranges)
    sign_frame = np.ones((truth_m.shape[0], n_team))
    for (per, si), sgn in signs.items():
        sign_frame[period == per, si] = sgn
    team_of = np.zeros(truth_m.shape[1], dtype=int)
    for si, (lo, hi) in enumerate(ranges):
        team_of[lo:hi] = si

    cent = visible_centroid(truth_m, visible, ranges)
    off, vote, vcount = role_offsets_vote(truth_m, visible, ranges, fps, halflife_s)
    return {
        "masked": masked,
        "vel": estimate_velocity(masked, period, fps),
        "cent": cent,
        "slot": slot_prediction(cent, cam, ranges, slot_coeffs),
        "off": off,
        "vote": vote,
        "vote_n": vcount,
        "b6": vote_field(off, vote, ranges),
        "ball": ffill_ball(ball_m),
        "team": _team_stats(truth_m, visible, ranges, sign_frame),
        "sign_frame": sign_frame,
        "team_of": team_of,
        "ranges": ranges,
    }


def build_features(
    fields: dict[str, object], samples: dict[str, np.ndarray], blend: np.ndarray
) -> np.ndarray:
    """Build the attack-aligned feature matrix for a set of hidden samples.

    Args:
        fields: Output of :func:`derive_fields`.
        samples: Output of ``imputation.collect_samples`` (needs ``frame``, ``slot_id``,
            ``last_frame``, ``tsls``).
        blend: The frozen anchor's point prediction per sample, shape ``(M, 2)``.

    Returns:
        Feature matrix, shape ``(M, len(FEATURE_NAMES))``, float32, columns in
        :data:`FEATURE_NAMES` order.
    """
    f = samples["frame"].astype(int)
    p = samples["slot_id"].astype(int)
    ell = samples["last_frame"].astype(int)
    tsls = samples["tsls"]
    team_of = np.asarray(fields["team_of"])
    ranges = fields["ranges"]
    sign_frame = np.asarray(fields["sign_frame"])
    masked = np.asarray(fields["masked"])
    team = fields["team"]
    side = team_of[p]
    opp = 1 - side
    sgn = sign_frame[f, side]

    last = align_pos(masked[ell, p], sgn)
    v0 = np.nan_to_num(np.asarray(fields["vel"])[ell, p])
    v0a = align_vec(v0, sgn)
    raw_last = masked[ell, p]
    boundary = np.minimum(
        np.minimum(raw_last[:, 0], PITCH_L - raw_last[:, 0]),
        np.minimum(raw_last[:, 1], PITCH_W - raw_last[:, 1]),
    )

    ball = np.asarray(fields["ball"])
    ball_now = align_pos(ball[f], sgn)
    ball_last = align_pos(ball[ell], sgn)
    ball_disp = align_vec(ball[f] - ball[ell], sgn)
    d_ball = np.linalg.norm(raw_last - ball[f], axis=1)

    cent = np.asarray(fields["cent"])
    own_cent = align_pos(cent[f, p], sgn)
    own_disp = align_vec(cent[f, p] - cent[ell, p], sgn)
    opp_cent = np.asarray(team["cent"])[f, opp]
    spread = np.asarray(team["spread"])[f, side]
    defline = np.asarray(team["defline"])[f, side]
    n_own = np.asarray(team["count"])[f, side]
    n_opp = np.asarray(team["count"])[f, opp]

    off = np.asarray(fields["off"])
    role_ord = np.zeros(f.size)
    off_x = off[ell, p, 0] * sgn
    active = np.isfinite(masked[:, :, 0])
    for si, (lo, hi) in enumerate(ranges):
        m = side == si
        if not m.any():
            continue
        block = off[ell[m], lo:hi, 0] * sgn[m, None]
        alive = active[ell[m], lo:hi]
        role_ord[m] = (alive & (block < off_x[m, None])).sum(axis=1)

    slot_f = np.asarray(fields["slot"])
    slot_now = np.where(np.isfinite(slot_f[f, p]), slot_f[f, p], masked[ell, p])
    slot_last = np.where(np.isfinite(slot_f[ell, p]), slot_f[ell, p], masked[ell, p])
    off_role = align_vec(masked[ell, p] - slot_last, sgn)

    vote_a = align_pos(np.asarray(fields["vote"])[f, side], sgn)
    slot_a = align_pos(slot_now, sgn)
    blend_a = align_pos(blend, sgn)
    cols = [
        last[:, 0], last[:, 1], v0a[:, 0], v0a[:, 1], np.linalg.norm(v0, axis=1),
        tsls, np.log1p(tsls), boundary,
        ball_now[:, 0], ball_now[:, 1], ball_last[:, 0], ball_last[:, 1],
        ball_disp[:, 0], ball_disp[:, 1], d_ball,
        own_cent[:, 0], own_cent[:, 1], own_disp[:, 0], own_disp[:, 1],
        opp_cent[:, 0], opp_cent[:, 1], spread[:, 0], spread[:, 1], defline, n_own, n_opp,
        role_ord, slot_a[:, 0], slot_a[:, 1], off_role[:, 0], off_role[:, 1],
        vote_a[:, 0], vote_a[:, 1], np.asarray(fields["vote_n"])[f, side],
        blend_a[:, 0], blend_a[:, 1],
    ]
    x = np.column_stack(cols).astype(np.float32)
    assert x.shape[1] == len(FEATURE_NAMES), (x.shape, len(FEATURE_NAMES))
    return x


# --------------------------------------------------------------------------------------
# uncertainty-only wrapper: per-bucket conformal ellipses on a frozen point prediction
# --------------------------------------------------------------------------------------


def region_scale(
    resid: np.ndarray, bucket: np.ndarray, floor: float = WIDTH_FLOOR_M
) -> np.ndarray:
    """Per-bucket per-axis half-width from residual quantiles (plan 3.1 step 2).

    Args:
        resid: Signed residuals ``pred - truth``, shape ``(M, 2)``, metres.
        bucket: Horizon-bucket index per sample, shape ``(M,)``.
        floor: Minimum half-width in metres.

    Returns:
        Half-widths, shape ``(n_buckets, 2)``.
    """
    w = np.full((len(BIN_LABELS), 2), floor)
    for b in range(len(BIN_LABELS)):
        m = bucket == b
        if m.sum() < 20:
            continue
        q = np.percentile(resid[m], [5, 95], axis=0)
        w[b] = np.maximum((q[1] - q[0]) / 2.0, floor)
    return w


def _radial_score(resid: np.ndarray, bucket: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalised radial nonconformity score ``sqrt((rx/wx)^2 + (ry/wy)^2)``."""
    ww = w[bucket]
    return np.sqrt(((resid / ww) ** 2).sum(axis=1))


def conformal_k(
    resid: np.ndarray, bucket: np.ndarray, w: np.ndarray, alphas: tuple[float, ...] = (0.5, 0.1)
) -> np.ndarray:
    """Per-bucket split-conformal multipliers at each miscoverage level (plan 3.1 step 4).

    Args:
        resid: Signed calibration residuals ``pred - truth``, shape ``(M, 2)``.
        bucket: Horizon-bucket index per calibration sample.
        w: Half-widths from :func:`region_scale`.
        alphas: Miscoverage levels; ``0.5`` -> 50% region, ``0.1`` -> 90% region.

    Returns:
        Multipliers, shape ``(n_buckets, len(alphas))``; ``inf`` where a bucket has too few
        calibration points to certify the level.
    """
    s = _radial_score(resid, bucket, w)
    k = np.full((len(BIN_LABELS), len(alphas)), np.inf)
    for b in range(len(BIN_LABELS)):
        sb = np.sort(s[bucket == b])
        nb = sb.size
        if nb == 0:
            continue
        for j, a in enumerate(alphas):
            rank = int(np.ceil((nb + 1) * (1.0 - a)))
            if rank <= nb:
                k[b, j] = float(sb[rank - 1])
    return k


def coverage_table(
    resid: np.ndarray,
    bucket: np.ndarray,
    w: np.ndarray,
    k: np.ndarray,
    alphas: tuple[float, ...] = (0.5, 0.1),
) -> list[dict[str, float]]:
    """Score empirical coverage (PICP) and sharpness (MPIW) of the conformal regions.

    Args:
        resid: Signed held-out residuals ``pred - truth``, shape ``(M, 2)``.
        bucket: Horizon-bucket index per held-out sample.
        w: Frozen half-widths.
        k: Frozen conformal multipliers.
        alphas: Miscoverage levels matching ``k``'s columns.

    Returns:
        One dict per bucket with ``n`` plus ``picp_<pct>`` and ``r_<pct>`` (equivalent region
        radius in metres, ``sqrt(area/pi)``) for each level.
    """
    s = _radial_score(resid, bucket, w)
    rows: list[dict[str, float]] = []
    for b in range(len(BIN_LABELS)):
        m = bucket == b
        row: dict[str, float] = {"n": int(m.sum())}
        for j, a in enumerate(alphas):
            pct = int(round((1.0 - a) * 100))
            inside = s[m] <= k[b, j] if m.any() else np.zeros(0, dtype=bool)
            row[f"picp_{pct}"] = float(inside.mean()) if m.any() else float("nan")
            row[f"r_{pct}"] = float(k[b, j] * np.sqrt(w[b, 0] * w[b, 1]))
        rows.append(row)
    return rows


def block_bootstrap(
    values: np.ndarray,
    block: np.ndarray,
    stat: str = "rmse",
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """Block-bootstrap a 95% CI for a statistic over correlated 25 fps samples.

    Blocks are resampled with replacement (plan 4.2: never i.i.d. resampling).

    Args:
        values: Per-sample values (errors in metres, or an inside-region indicator).
        block: Block id per sample (1-minute blocks).
        stat: ``"rmse"`` or ``"mean"``.
        n_boot: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        Tuple ``(lo, hi)``, the 2.5th and 97.5th percentiles of the replicate statistic.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(block, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=uniq.size)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    vals = values[order]
    reps = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        chunks = [vals[bounds[j] : bounds[j + 1]] for j in pick]
        v = np.concatenate(chunks) if chunks else vals
        reps[i] = np.sqrt(np.mean(v**2)) if stat == "rmse" else float(np.mean(v))
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def bucket_of(tsls: np.ndarray) -> np.ndarray:
    """Public alias for the frozen horizon-bucket index."""
    return _bucket_index(tsls)


def _self_check() -> None:
    """Assert the vote recursion, the causal ball fill and the conformal maths on toy cases."""
    # Two players 10 m apart on a 1-slot-per-team pitch; player 1 hidden after frame 5.
    n, fps = 80, 25.0
    truth = np.full((n, 4, 2), np.nan)
    truth[:, 0] = [30.0, 34.0]
    truth[:, 1] = [40.0, 34.0]
    truth[:, 2] = [70.0, 34.0]
    truth[:, 3] = [80.0, 34.0]
    vis = np.ones((n, 4), dtype=bool)
    vis[60:, 1] = False
    ranges = ((0, 2), (2, 4))
    off, vote, cnt = role_offsets_vote(truth, vis, ranges, fps, halflife_s=0.1)
    pred = vote_field(off, vote, ranges)
    # Static formation -> the vote must recover the hidden player's true position.
    assert abs(pred[-1, 1, 0] - 40.0) < 0.1, pred[-1, 1]
    assert cnt[-1, 0] == 1 and cnt[0, 0] == 2
    # Hidden player's offset is frozen from its last sighting.
    assert np.allclose(off[-1, 1], off[59, 1])
    # Causal ball fill: never looks forward.
    ball = np.array([[np.nan, np.nan], [1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
    ff = ffill_ball(ball)
    assert np.allclose(ff, [[1, 2], [1, 2], [1, 2], [3, 4]])
    # Conformal: unit-scale isotropic residuals, k(alpha=0.1) ~= the 90th percentile radius.
    rng = np.random.default_rng(0)
    r = rng.normal(size=(4000, 2))
    bkt = np.zeros(4000, dtype=int)
    w = np.ones((len(BIN_LABELS), 2))
    k = conformal_k(r, bkt, w)
    rows = coverage_table(r, bkt, w, k)
    assert 0.45 <= rows[0]["picp_50"] <= 0.55, rows[0]
    assert 0.85 <= rows[0]["picp_90"] <= 0.95, rows[0]
    print("imputation_features self-check OK (vote recovery, causal ball fill, conformal cover)")


if __name__ == "__main__":
    _self_check()
