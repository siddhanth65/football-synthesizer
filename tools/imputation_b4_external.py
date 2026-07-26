"""B4 M3: run the FROZEN v1 on external tracking -- SkillCorner opendata and our own footage.

Two checks, neither of which is a validation against ground truth. Both are labelled as what
they are.

**SkillCorner (``--source skillcorner``).** Their opendata carries a per-player
``is_detected`` flag: detected points come from the broadcast image, everything else is
SkillCorner's own extrapolation. We treat DETECTED as observed and their extrapolated segments
as the hidden set, then score our frozen v1 and the frozen B7 anchor against those extrapolated
positions. What this CAN establish: whether v1 still beats the anchor when the camera, the
censoring pattern and the sport are real rather than simulated, both being measured against the
same third-party reference. What it CANNOT establish: an absolute accuracy number -- SkillCorner's
extrapolation is another estimator, not truth, and it is very likely bidirectional (a delivered
data product can use the re-sighting), which makes it oracle-class. Agreement with it is
therefore not accuracy, and a method that predicts the way SkillCorner predicts scores well for
free. The comparison is reported per horizon so the reader can see where that bias bites.

**Our own footage (``--source ours``).** No truth of any kind exists off-camera, so this is a
SMELL TEST, not a validation: are v1's outputs physically sane on real broadcast tracks
(off-pitch rate, implied speed, placement inside the visible region, team shape)? Our tracks also
fragment on re-identification, so a "hidden" player is sometimes a live player under a new track
id; the numbers here are therefore upper bounds on genuine off-screen error behaviour.

Usage:
    python -m tools.imputation_b4_external --source skillcorner [--cache PATH]
    python -m tools.imputation_b4_external --source ours --match brighton_manutd --chunks 3
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Sequence

import numpy as np

from synthesizer.imputation import (
    BIN_LABELS,
    PITCH_L,
    PITCH_W,
    REPO,
    collect_samples,
    estimate_velocity,
    fit_decay_tau,
    smooth_camera,
    slot_prediction,
    visible_centroid,
)
from synthesizer.imputation_features import (
    attack_signs,
    build_features,
    bucket_of,
    derive_fields,
)
from synthesizer.imputation_v1 import (
    b7_position,
    b7_weights,
    fit_heads,
    paired_rmse_ci,
    sample_signs,
    v1_prediction,
)
from tools.imputation_b4_transfer import (
    BAR_HALFLIFE_S,
    GEOM_BY_NAME,
    PARAMS,
    Tee,
    _blocks,
    _err,
    _rmse,
    build_split,
    cached,
    censor,
    load_game,
    tune_scale,
)

SC_DIR = REPO / "data" / "imputation" / "skillcorner"
SC_MATCH = "1886347"
CORNER_KEYS = (
    "x_top_left", "y_top_left", "x_bottom_left", "y_bottom_left",
    "x_bottom_right", "y_bottom_right", "x_top_right", "y_top_right",
)
MIN_TRACK_S = 2.0  # our footage: ignore tracks shorter than this (pure re-id churn)
MAX_GHOST_S = 60.0  # our footage: follow a dead track this long before dropping it
EDGE_FRAC = 0.10  # our footage: "left the frame" = last seen within this fraction of the edge


# --------------------------------------------------------------------------------------
# frozen v1 (fitted exactly as in the gate run, then cached)
# --------------------------------------------------------------------------------------


def fit_frozen_v1(cache_dir: str, out: Tee) -> tuple[object, dict, float, np.ndarray, dict]:
    """Fit (or load) the frozen v1 heads plus the frozen anchor, on Metrica TRAIN.

    Shares the joblib cache written by ``tools.imputation_b4_transfer`` (same keys), so the two
    M3 tools fit the model once between them. The fit is deterministic, so this changes nothing.

    Args:
        cache_dir: Cache directory; empty refits from scratch (~10 CPU minutes).
        out: Report sink.

    Returns:
        Tuple ``(heads, coeffs, tau, w7, train_game)``.
    """
    g1 = load_game("Sample_Game_1")
    base = GEOM_BY_NAME["rect_base"]
    scale, _ = tune_scale(g1, base)
    g1["visible"] = censor(g1, base, scale)
    train: dict[str, object] = {}

    def bits() -> dict[str, object]:
        """Censor TRAIN and collect its samples (only when a cache misses)."""
        if not train:
            f1, s1, coeffs, _ = build_split(g1, base, scale, None)
            tau, _ = fit_decay_tau(s1)
            train.update({"f1": f1, "s1": s1, "coeffs": coeffs, "tau": tau,
                          "w7": b7_weights(s1, tau)})
        return train

    meta = cached(cache_dir, "train_meta", lambda: {k: bits()[k] for k in ("coeffs", "tau", "w7")},
                  out)

    def build_v1() -> object:
        """Fit v1 exactly as the gate run did."""
        f1, s1 = bits()["f1"], bits()["s1"]
        anc = b7_position(s1, float(meta["tau"]), np.asarray(meta["w7"]))
        x = build_features(f1, s1, anc)
        r = (s1["target"] - anc) * sample_signs(f1, s1)[:, None]
        return fit_heads(x, r, PARAMS)

    t0 = time.time()
    heads = cached(cache_dir, "v1_heads", build_v1, out)
    out(f"frozen v1 ready in {time.time() - t0:.0f} s")
    return heads, meta["coeffs"], float(meta["tau"]), np.asarray(meta["w7"]), g1


def depth_rank(
    truth: np.ndarray,
    visible: np.ndarray,
    period: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
) -> list[np.ndarray]:
    """Rank each team's slots by mean attack-aligned x (0 = deepest), from visible frames only.

    Args:
        truth: Positions in metres, shape ``(n, slots, 2)``.
        visible: Visibility mask.
        period: Per-frame period id.
        ranges: Team slot ranges.

    Returns:
        One array of local-slot indices per team, ordered deepest first.
    """
    signs = attack_signs(truth, visible, period, ranges)
    n_team = len(ranges)
    sign_frame = np.ones((truth.shape[0], n_team))
    for (per, si), sgn in signs.items():
        sign_frame[period == per, si] = sgn
    order: list[np.ndarray] = []
    pos = np.where(visible[:, :, None], truth, np.nan)
    for si, (lo, hi) in enumerate(ranges):
        aligned = sign_frame[:, si, None] * (pos[:, lo:hi, 0] - PITCH_L / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # slots that are never visible
            mean_x = np.nanmean(np.where(np.isfinite(aligned), aligned, np.nan), axis=0)
        mean_x = np.where(np.isfinite(mean_x), mean_x, 1e6)
        order.append(np.argsort(mean_x, kind="stable"))
    return order


def remap_slot_coeffs(
    coeffs: dict[tuple[str, int], np.ndarray],
    train_order: list[np.ndarray],
    ext_order: list[np.ndarray],
) -> dict[tuple[str, int], np.ndarray]:
    """Give every external slot the TRAIN slot coefficient of the same relative depth rank.

    The frozen slot model is keyed by (side, local column index), and those indices mean nothing
    across datasets. Matching on depth rank -- deepest external player gets the deepest TRAIN
    slot's coefficients -- is the cheapest defensible correspondence, and it is the same ordering
    the ``role_ord`` feature already uses.

    Args:
        coeffs: Frozen ``fit_slot_model`` output.
        train_order: Depth-ranked TRAIN local slot indices per team.
        ext_order: Depth-ranked external local slot indices per team.

    Returns:
        A coefficient dict keyed by the external slots.
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for si, side in enumerate(("home", "away")):
        tr = [j for j in train_order[si] if (side, int(j)) in coeffs]
        ext = ext_order[si]
        if not tr or ext.size == 0:
            continue
        for rank, j in enumerate(ext):
            q = rank / max(ext.size - 1, 1)
            src = int(tr[min(int(round(q * (len(tr) - 1))), len(tr) - 1)])
            out[(side, int(j))] = coeffs[(side, src)]
    return out


# --------------------------------------------------------------------------------------
# SkillCorner loader
# --------------------------------------------------------------------------------------


def load_skillcorner() -> dict:
    """Load SkillCorner opendata into the same bundle shape the Metrica pipeline uses.

    Their coordinate frame is centred on the pitch (x in +/-52, y in +/-34); we shift and scale
    to the 105 x 68 frame used everywhere else. ``is_detected`` becomes the visibility mask, and
    their reported positions become the (pseudo-)truth -- see the module docstring for what that
    does and does not license. Frames with no player data (stoppages, replays) keep the previous
    period id so a long unseen spell is measured in wall-clock seconds.

    Returns:
        Bundle dict with ``truth/ball/period/fps/ranges/visible/cam/vel/quad``.
    """
    meta = json.loads((SC_DIR / f"{SC_MATCH}_match.json").read_text(encoding="utf-8"))
    teams = [meta["home_team"]["id"], meta["away_team"]["id"]]
    by_team: dict[int, list[int]] = {t: [] for t in teams}
    for p in meta["players"]:
        by_team[p["team_id"]].append(int(p["id"]))
    slot_of: dict[int, int] = {}
    ranges: list[tuple[int, int]] = []
    k = 0
    for t in teams:
        lo = k
        for pid in sorted(by_team[t]):
            slot_of[pid] = k
            k += 1
        ranges.append((lo, k))
    n = max(p["end_frame"] for p in meta["match_periods"]) + 1
    truth = np.full((n, k, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    visible = np.zeros((n, k), dtype=bool)
    period = np.zeros(n, dtype=int)
    quad = np.full((n, 4, 2), np.nan)
    sx, sy = PITCH_L / float(meta["pitch_length"]), 1.0
    with (SC_DIR / f"{SC_MATCH}_tracking_extrapolated.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            f = int(rec["frame"])
            if f >= n:
                continue
            if rec["period"] is not None:
                period[f] = int(rec["period"])
            for p in rec["player_data"]:
                s = slot_of.get(int(p["player_id"]))
                if s is None or p["x"] is None:
                    continue
                truth[f, s] = [(float(p["x"]) + meta["pitch_length"] / 2) * sx,
                               (float(p["y"]) + PITCH_W / 2) * sy]
                visible[f, s] = bool(p["is_detected"])
            bd = rec["ball_data"]
            if bd["x"] is not None:
                ball[f] = [(float(bd["x"]) + meta["pitch_length"] / 2) * sx,
                           float(bd["y"]) + PITCH_W / 2]
            ic = rec["image_corners_projection"]
            vals = [ic[key] for key in CORNER_KEYS]
            if all(v is not None for v in vals):
                q = np.array(vals, dtype=float).reshape(4, 2)
                quad[f, :, 0] = (q[:, 0] + meta["pitch_length"] / 2) * sx
                quad[f, :, 1] = q[:, 1] + PITCH_W / 2
    period = _ffill_period(period)
    fps = 10.0
    return {
        "truth": truth, "ball": ball, "period": period, "fps": fps,
        "ranges": tuple(ranges), "visible": visible, "cam": smooth_camera(ball, fps),
        "vel": estimate_velocity(np.where(visible[:, :, None], truth, np.nan), period, fps),
        "quad": quad, "label": f"SkillCorner {SC_MATCH} (A-League, 10 fps)",
    }


def _ffill_period(period: np.ndarray) -> np.ndarray:
    """Carry the last live period id across dead frames so gaps count wall-clock time."""
    idx = np.maximum.accumulate(np.where(period > 0, np.arange(period.size), -1))
    return np.where(idx < 0, period[0], period[np.maximum(idx, 0)])


def inside_quad(pos: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Point-in-convex-quad test; ``quad`` is ``(M, 4, 2)`` ordered around the ring."""
    ok = np.ones(pos.shape[0], dtype=bool)
    area = np.zeros(pos.shape[0])
    for i in range(4):
        a, b = quad[:, i], quad[:, (i + 1) % 4]
        area += a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    sgn = np.where(area > 0, 1.0, -1.0)
    for i in range(4):
        a, b = quad[:, i], quad[:, (i + 1) % 4]
        cross = (b[:, 0] - a[:, 0]) * (pos[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            pos[:, 0] - a[:, 0]
        )
        ok &= (cross * sgn) >= -1e-9
    return ok


# --------------------------------------------------------------------------------------
# our-footage loader
# --------------------------------------------------------------------------------------


def load_ours(match_id: str, n_chunks: int, chunks: Sequence[str] | None = None) -> dict:
    """Build a bundle from our own tracking parquets: tracks become slots, gaps become hidden.

    There is no truth off camera, so a dead track is followed for :data:`MAX_GHOST_S` with a
    placeholder position (never scored) purely so the machinery emits predictions we can inspect.
    Track fragmentation means a "hidden" slot is sometimes a live player under a new id.

    Args:
        match_id: Registry match id.
        n_chunks: How many chunks to use (each ~10 minutes of broadcast).
        chunks: Explicit chunk keys to load; overrides ``n_chunks`` when given (used by
            ``tools.tactical_clip`` to build a bundle for one passage's chunk only).

    Returns:
        Bundle dict with the usual fields plus an ``edge`` field marking the samples whose last
        sighting sat near the image border (a genuine frame exit, not a tracker drop), and the
        bookkeeping that maps slots back to source tracks: ``slot_track``
        (``{period: {slot: track_id}}``) and ``frame_map`` (``{period: (offset, f0, step)}``).
    """
    import pandas as pd  # noqa: PLC0415

    from core import registry  # noqa: PLC0415

    m = registry.get(match_id)
    df = m.load_aligned()
    df = df[
        (df["calib_error_m"] <= 1.0) & df["pitch_x"].notna() & df["pitch_y"].notna()
        & df["team"].isin([0, 1]) & df["role"].isin(["player", "goalkeeper"])
    ]
    chunks = list(chunks) if chunks is not None else sorted(df["chunk"].unique())[:n_chunks]
    img_w = float(df["image_x"].max())
    balls = dict(m.ball_chunks())
    parts = []
    for chunk in chunks:
        d = df[df["chunk"] == chunk]
        step = int(np.median(np.diff(np.sort(d["frame"].unique()))))
        fps = m.chunk_fps(chunk) / step
        life = d.groupby("track_id")["frame"].agg(lambda s: (s.max() - s.min()) / (fps * step))
        d = d[d["track_id"].isin(life[life >= MIN_TRACK_S].index)]
        team_of = d.groupby("track_id")["team"].first()
        parts.append((chunk, d, step, fps, team_of))
    # One global slot layout so team membership means the same thing in every chunk; a slot is
    # still NOT the same player across chunks (track ids restart), which is why chunks are scored
    # as separate periods.
    cap = [max(int((t == side).sum()) for _, _, _, _, t in parts) for side in (0, 1)]
    n_slots = cap[0] + cap[1]
    n_tot = sum(int((d["frame"].max() - d["frame"].min()) // step) + 1 for _, d, step, _, _ in parts)
    truth = np.full((n_tot, n_slots, 2), np.nan)
    visible = np.zeros((n_tot, n_slots), dtype=bool)
    edge = np.zeros((n_tot, n_slots), dtype=bool)
    ball = np.full((n_tot, 2), np.nan)
    period = np.zeros(n_tot, dtype=int)
    n_tracks = 0
    off = 0
    slot_track: dict[int, dict[int, int]] = {}
    frame_map: dict[int, tuple[int, int, int]] = {}
    for ci, (chunk, d, step, fps, team_of) in enumerate(parts):
        f0 = int(d["frame"].min())
        n = int((d["frame"].max() - f0) // step) + 1
        slot: dict[int, int] = {}
        for side, base in ((0, 0), (1, cap[0])):
            for k, t in enumerate(sorted(team_of[team_of == side].index)):
                slot[int(t)] = base + k
        n_tracks += len(slot)
        slot_track[ci + 1] = {int(s): int(t) for t, s in slot.items()}
        frame_map[ci + 1] = (off, int(d["frame"].min()), int(step))
        rows = d[["frame", "track_id", "pitch_x", "pitch_y", "image_x"]].to_numpy()
        fi = off + ((rows[:, 0] - f0) // step).astype(int)
        si = np.array([slot[int(t)] for t in rows[:, 1]])
        truth[fi, si] = rows[:, 2:4]
        visible[fi, si] = True
        period[off : off + n] = ci + 1
        for s in np.unique(si):
            v = np.flatnonzero(visible[off : off + n, s]) + off
            last = int(v[-1])
            ghost = min(off + n, last + 1 + int(MAX_GHOST_S * fps))
            truth[last + 1 : ghost, s] = truth[last, s]  # placeholder target, never scored
            ix = rows[(si == s) & (fi == last), 4]
            if ix.size:
                edge[last:ghost, s] = min(ix.min(), img_w - ix.max()) <= EDGE_FRAC * img_w
        if chunk in balls:
            b = pd.read_parquet(balls[chunk])
            bi = off + ((b["frame"].to_numpy() - f0) // step).astype(int)
            ok = (bi >= off) & (bi < off + n)
            ball[bi[ok]] = b[["x", "y"]].to_numpy()[ok]
        off += n
    fps = parts[0][3]
    return {
        "truth": truth, "ball": ball, "period": period, "fps": fps,
        "ranges": ((0, cap[0]), (cap[0], n_slots)), "visible": visible,
        "cam": smooth_camera(ball, fps),
        "vel": estimate_velocity(np.where(visible[:, :, None], truth, np.nan), period, fps),
        "edge": edge,
        "slot_track": slot_track,
        "frame_map": frame_map,
        "label": f"{match_id} ({len(chunks)} chunks, {n_tracks} tracks >= {MIN_TRACK_S:.0f}s)",
    }


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def run_bundle(
    bundle: dict,
    heads: object,
    coeffs: dict,
    tau: float,
    w7: np.ndarray,
    train_order: list[np.ndarray],
    out: Tee,
    have_truth: bool,
    max_samples: int = 400_000,
) -> dict[str, np.ndarray]:
    """Derive fields, predict with the frozen v1, and print the diagnostics for one bundle.

    Args:
        bundle: Loaded external bundle.
        heads: Frozen v1 heads.
        coeffs: Frozen slot-model coefficients (TRAIN).
        tau: Frozen decay constant.
        w7: Frozen B7 blend weights.
        train_order: TRAIN depth ranking, for the slot-coefficient correspondence.
        out: Report sink.
        have_truth: Whether ``target`` carries a (pseudo-)reference worth scoring against.
        max_samples: Stride-subsample above this many hidden samples. Our own footage has
            hundreds of track slots, and one feature (``role_ord``) is O(samples x slots).

    Returns:
        Dict with the samples, the v1 prediction and the anchor prediction.
    """
    truth, visible = bundle["truth"], bundle["visible"]
    ranges, fps = bundle["ranges"], bundle["fps"]
    ext_order = depth_rank(truth, visible, bundle["period"], ranges)
    coeffs_ext = remap_slot_coeffs(coeffs, train_order, ext_order)
    fields = derive_fields(
        truth, visible, bundle["ball"], bundle["cam"], bundle["period"], fps, ranges,
        coeffs_ext, BAR_HALFLIFE_S,
    )
    samples = collect_samples(
        truth, visible, bundle["period"], fps, bundle["vel"],
        slot_prediction(visible_centroid(truth, visible, ranges), bundle["cam"], ranges,
                        coeffs_ext),
        fields={"b6": np.asarray(fields["b6"])},
    )
    n_all = samples["tsls"].size
    if n_all > max_samples:
        step = int(np.ceil(n_all / max_samples))
        samples = _sub(samples, np.arange(n_all) % step == 0)
        out(f"  subsampled every {step}th hidden sample ({n_all} -> {samples['tsls'].size}) to "
            "bound the O(samples x slots) role feature")
    anchor = b7_position(samples, tau, w7)
    x = build_features(fields, samples, anchor)
    mu, _ = v1_prediction(heads, x, anchor, sample_signs(fields, samples))
    counts = visible.sum(axis=1)
    out(f"\n{bundle['label']}: {truth.shape[0]} frames x {truth.shape[1]} slots, fps={fps:.1f}")
    live = counts > 0
    out(f"  visible/frame over frames with any sighting: mean {counts[live].mean():.2f} "
        f"(p50 {np.median(counts[live]):.0f}, p90 {np.percentile(counts[live], 90):.0f}) on "
        f"{int(live.sum())} of {counts.size} frames; hidden samples scored {samples['tsls'].size}")
    bkt = bucket_of(samples["tsls"])
    out("  horizon mix: " + " ".join(
        f"{lab}={100 * np.mean(bkt == b):.0f}%" for b, lab in enumerate(BIN_LABELS)))
    if have_truth:
        _score(samples, mu, anchor, fps, out)
    _plausibility(bundle, samples, mu, anchor, out)
    return {"samples": samples, "mu": mu, "anchor": anchor}


def _score(
    samples: dict[str, np.ndarray], mu: np.ndarray, anchor: np.ndarray, fps: float, out: Tee
) -> None:
    """Per-bucket v1-vs-anchor table against a pseudo-truth reference."""
    v1e, b7e = _err(mu, samples["target"]), _err(anchor, samples["target"])
    bkt = bucket_of(samples["tsls"])
    blocks = _blocks(samples, fps)
    out("  vs the reference positions (NOT ground truth -- see module docstring):")
    out("  horizon   |      n |   B7   |   v1   |  margin  [95% block CI]")
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if m.sum() < 50:
            continue
        lo, hi = paired_rmse_ci(b7e[m], v1e[m], blocks[m])
        out(f"  {lab:9s} | {int(m.sum()):6d} | {_rmse(b7e[m]):6.2f} | {_rmse(v1e[m]):6.2f} | "
            f"{_rmse(v1e[m]) - _rmse(b7e[m]):+7.2f} [{lo:+6.2f},{hi:+6.2f}]")
    lo, hi = paired_rmse_ci(b7e, v1e, blocks)
    out(f"  {'ALL':9s} | {v1e.size:6d} | {_rmse(b7e):6.2f} | {_rmse(v1e):6.2f} | "
        f"{_rmse(v1e) - _rmse(b7e):+7.2f} [{lo:+6.2f},{hi:+6.2f}]")


def _plausibility(
    bundle: dict, samples: dict[str, np.ndarray], mu: np.ndarray, anchor: np.ndarray, out: Tee
) -> None:
    """Distributional smell test: off-pitch, inside-the-visible-region, implied speed, shape."""
    f = samples["frame"].astype(int)
    tsls = np.maximum(samples["tsls"], 1e-3)
    named = [("last-seen", samples["hold"]), ("B7", anchor), ("v1", mu)]
    if "quad" in bundle:
        named.insert(0, ("reference", samples["target"]))
    out("  method    | off-pitch | inside visible region | implied speed m/s p50 / p90 / max")
    for name, pos in named:
        if "quad" in bundle:
            q = bundle["quad"][f]
            ok = np.isfinite(q[:, 0, 0])
            ins = float(inside_quad(pos[ok], q[ok]).mean()) if ok.any() else float("nan")
        else:
            ins = _inside_hull_proxy(bundle, samples, pos)
        spd = np.linalg.norm(pos - samples["hold"], axis=1) / tsls
        off = float((
            (pos[:, 0] < -1) | (pos[:, 0] > PITCH_L + 1)
            | (pos[:, 1] < -1) | (pos[:, 1] > PITCH_W + 1)
        ).mean())
        out(f"  {name:9s} | {100 * off:8.1f}% | {100 * ins:20.1f}% | "
            f"{np.percentile(spd, 50):6.2f} / {np.percentile(spd, 90):6.2f} / {spd.max():7.2f}")
    if "edge" in bundle:
        e = bundle["edge"][f, samples["slot_id"].astype(int)]
        out(f"  last sighting within {100 * EDGE_FRAC:.0f}% of the image edge (a genuine frame "
            f"exit rather than a tracker drop): {100 * e.mean():.1f}% of samples")
        if e.any():
            spd_e = np.linalg.norm(mu[e] - samples["hold"][e], axis=1) / tsls[e]
            ins_e = _inside_hull_proxy(bundle, _sub(samples, e), mu[e])
            out(f"  edge-exit subset only: v1 inside the visible box {100 * ins_e:.1f}%, "
                f"implied speed p50 {np.percentile(spd_e, 50):.2f} m/s")
    _shape(bundle, samples, mu, out)


def _sub(samples: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    """Row-select a sample dict with a boolean mask."""
    return {k: v[m] for k, v in samples.items()}


def _inside_hull_proxy(bundle: dict, samples: dict[str, np.ndarray], pos: np.ndarray) -> float:
    """Fraction of predictions inside the bounding box of the visible players at that frame.

    A proxy for the camera footprint when no projection is stored: the visible players' bounding
    box under-states the true field of view (it stops at the outermost detected player), so this
    is a LOWER bound on 'placed where the camera can see'.
    """
    vis = bundle["visible"]
    truth = bundle["truth"]
    p = np.where(vis[:, :, None], truth, np.nan)
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(p, axis=1)
        hi = np.nanmax(p, axis=1)
    f = samples["frame"].astype(int)
    ok = np.isfinite(lo[f, 0])
    inside = (
        (pos[:, 0] >= lo[f, 0]) & (pos[:, 0] <= hi[f, 0])
        & (pos[:, 1] >= lo[f, 1]) & (pos[:, 1] <= hi[f, 1])
    )
    return float(inside[ok].mean()) if ok.any() else float("nan")


def _shape(bundle: dict, samples: dict[str, np.ndarray], mu: np.ndarray, out: Tee) -> None:
    """Team-shape sanity: does completing the team with imputed players keep a sane x-span?

    A real team's longitudinal span sits around 40-60 m. If the imputed players blow the span out
    to most of the pitch, the completed formation is not physically sane whatever the RMSE says.
    """
    f = samples["frame"].astype(int)
    p = samples["slot_id"].astype(int)
    n_frames, n_slots = bundle["visible"].shape
    team_of = np.zeros(n_slots, dtype=int)
    for si, (lo, hi) in enumerate(bundle["ranges"]):
        team_of[lo:hi] = si
    side = team_of[p]
    pos = np.where(bundle["visible"][:, :, None], bundle["truth"], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody visible
        lo_v = np.stack([np.nanmin(pos[:, lo:hi, 0], axis=1) for lo, hi in bundle["ranges"]], 1)
        hi_v = np.stack([np.nanmax(pos[:, lo:hi, 0], axis=1) for lo, hi in bundle["ranges"]], 1)
    key = f * 2 + side
    lo_i = np.full(n_frames * 2, np.inf)
    hi_i = np.full(n_frames * 2, -np.inf)
    np.minimum.at(lo_i, key, mu[:, 0])
    np.maximum.at(hi_i, key, mu[:, 0])
    flat_lo, flat_hi = lo_v.reshape(-1), hi_v.reshape(-1)
    span_vis = flat_hi - flat_lo
    span_all = np.fmax(flat_hi, hi_i) - np.fmin(flat_lo, lo_i)
    m = np.isfinite(span_vis) & np.isfinite(span_all)
    out(f"  team x-span at the query frames: visible-only {np.nanmean(span_vis[m]):.1f} m -> "
        f"completed with v1 {np.nanmean(span_all[m]):.1f} m "
        f"(> 90 m in {100 * np.mean(span_all[m] > 90):.1f}% of team-frames)")


def main() -> None:
    """Run the external transfer checks."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("skillcorner", "ours"), required=True)
    ap.add_argument("--match", default="brighton_manutd", help="registry id for --source ours")
    ap.add_argument("--chunks", type=int, default=3, help="chunks to use for --source ours")
    ap.add_argument("--cache", default="",
                    help="joblib cache DIRECTORY shared with tools.imputation_b4_transfer")
    ap.add_argument("--out", default="", help="optional run-log path")
    args = ap.parse_args()
    out = Tee()
    out("=" * 100)
    out(f"B4 M3 -- FROZEN v1 on external tracking: {args.source}")
    out("=" * 100)
    heads, coeffs, tau, w7, g1 = fit_frozen_v1(args.cache, out)
    train_order = depth_rank(g1["truth"], g1["visible"], g1["period"], g1["ranges"])
    if args.source == "skillcorner":
        bundle = load_skillcorner()
        run_bundle(bundle, heads, coeffs, tau, w7, train_order, out, have_truth=True)
    else:
        bundle = load_ours(args.match, args.chunks)
        run_bundle(bundle, heads, coeffs, tau, w7, train_order, out, have_truth=False)
    if args.out:
        dst = REPO / args.out
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(["```", *out.lines, "```", ""]), encoding="utf-8")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
