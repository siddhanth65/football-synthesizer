"""B4 M3: does the FROZEN v1 survive a camera it never trained on?

v1 passed its gates against a single censoring geometry -- a hard rectangle, 33.8 m x 68 m,
centred on the smoothed ball track. The gate run's own diagnostic
(``results/B4_MODEL_V1.md`` section 6.1) showed v1 places 14.9% of its predictions inside the
band where a hidden player cannot be, against 18.7% for the anchor and 11.5% for the truth, i.e.
it has partly learnt the simulator's window. This tool asks the only question that matters: with
**nothing refitted**, how much of v1's margin over the B7 anchor survives when the censoring
geometry changes?

Protocol (frozen v1, no refit anywhere in the geometry sweep):

* TRAIN = Metrica Game 1 under the ORIGINAL rectangle -> slot model, tau, B7 weights, GBM heads.
* Every geometry is scale-tuned on Game 1 (TRAIN) to the same 11.8 visible players/frame, so the
  comparison is between *shapes*, not between amounts of information.
* HOLDOUT = Game 2 second half, re-censored by each geometry. Anchor and v1 are both re-run on
  that geometry's samples; the reported quantity is the MARGIN ``RMSE(v1) - RMSE(B7)``, because
  absolute RMSE is not comparable across geometries (different hidden sets, different horizons).

The geometry family is grounded in real broadcast tracking: SkillCorner opendata match 1886347
reports a per-frame camera footprint (``image_corners_projection``) alongside a detected /
extrapolated flag per player. Measured there: detected players are 98.4% inside that footprint,
extrapolated ones 5.3%, and the detection rate crosses 0% -> 94% within about 2 m of the
footprint edge. Real censoring is therefore also a hard geometric window -- but a **trapezoid**
that widens away from the camera, not a rectangle. ``trapezoid_sc`` and ``sc_realistic`` below
use that measured shape.

Usage:
    python -m tools.imputation_b4_transfer [--geoms a,b,c] [--mitigate] [--out PATH]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from synthesizer.imputation import (
    BIN_LABELS,
    METRICA,
    PITCH_L,
    PITCH_W,
    REPO,
    TARGET_VISIBLE,
    collect_samples,
    estimate_velocity,
    fit_decay_tau,
    fit_slot_model,
    load_metrica_match,
    smooth_camera,
    team_ranges,
    tune_window,
    visible_centroid,
)
from synthesizer.imputation_features import build_features, bucket_of, derive_fields
from synthesizer.imputation_v1 import (
    QUANTILES,
    Heads,
    b7_position,
    b7_weights,
    fit_heads,
    paired_rmse_ci,
    sample_signs,
    v1_prediction,
)

# Frozen v1 hyper-parameters (results/B4_MODEL_V1.md section 1) -- NOT re-tuned here.
PARAMS: dict[str, object] = {
    "max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 63,
    "min_samples_leaf": 200, "l2_regularization": 1.0,
}
BAR_HALFLIFE_S = 0.04  # the vote EMA half-life that produced the frozen bar
BLOCK_S = 60.0

# SkillCorner footprint, measured on match 1886347 (37 530 frames with a reported projection),
# medians relative to the footprint's own centre x, converted from their centred pitch frame to
# Metrica's 0..105 x 0..68 frame with the camera on the y = 0 touchline:
#   near edge  y = -30.1 (-> 3.9), half-width 11.0 m
#   far  edge  y = +39.0 (-> 73.0, past the far touchline), half-width 24.1 m
TRAP_Y_NEAR, TRAP_Y_FAR = 3.9, 73.0
TRAP_HW_NEAR, TRAP_HW_FAR = 11.0, 24.1
SOFT_BLOCK_S = 10.0  # a player's edge offset is resampled this often (keeps occlusions coherent)


# --------------------------------------------------------------------------------------
# censoring geometries -- each mask is monotone increasing in ``scale``
# --------------------------------------------------------------------------------------

MaskFn = Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Geometry:
    """One censoring geometry: a visibility predicate plus how the camera tracks the ball.

    Attributes:
        name: Short key used on the command line and in tables.
        desc: One-line human description for the report.
        mask: ``(pos, cam_rows, scale, frames) -> bool`` where ``pos`` is ``(rows, slots, 2)``,
            ``cam_rows`` is the camera centre for each row and ``frames`` the global frame index
            of each row (needed by the stochastic edges). ``scale`` may be a scalar or a
            ``(rows, slots)`` array; the mask is monotone increasing in it.
        lag_s: Camera reaction delay in seconds (the operator follows the ball late).
        lo: Lower bracket for the scale bisection.
        hi: Upper bracket for the scale bisection.
        in_train_family: True if a domain-randomised v1.1 saw this SHAPE family in training.
        diag: Mask used by the inside-the-window diagnostic; for a feathered geometry this is the
            un-jittered base, so the diagnostic reports membership of the NOMINAL window (a
            per-sample jitter is not recoverable from a point query). Defaults to ``mask``.
    """

    name: str
    desc: str
    mask: MaskFn
    lag_s: float = 0.0
    lo: float = 1.0
    hi: float = 105.0
    in_train_family: bool = False
    diag: MaskFn | None = None


def _rect(half_h: float) -> MaskFn:
    """Axis-aligned rectangle of half-height ``half_h``; ``scale`` is the half-width."""

    def fn(pos: np.ndarray, cam: np.ndarray, scale: float, frames: np.ndarray) -> np.ndarray:
        dx = np.abs(pos[:, :, 0] - cam[:, None, 0])
        dy = np.abs(pos[:, :, 1] - cam[:, None, 1])
        return (dx <= scale) & (dy <= half_h)

    return fn


def _soft(sd: float, base: MaskFn, seed: int = 0) -> MaskFn:
    """Feather a mask's edge by jittering the scale it sees, per slot, by ``N(0, sd)``.

    Every mask here accepts an array-valued ``scale``, so feathering is just a noisy scale. The
    jitter is resampled every :data:`SOFT_BLOCK_S` seconds per slot rather than every frame, so
    occlusion episodes stay coherent (a per-frame coin flip would shred them into 25 fps flicker
    and destroy the horizon buckets). Metrica is 25 fps throughout, which the block length
    assumes.

    Args:
        sd: Standard deviation of the edge offset, in the base mask's scale units (metres for a
            rectangle, a multiplier for the trapezoid).
        base: Mask to feather.
        seed: RNG seed -- fixed, so every run censors identically.

    Returns:
        A mask function with a soft population-level edge.
    """

    def fn(pos: np.ndarray, cam: np.ndarray, scale: float, frames: np.ndarray) -> np.ndarray:
        blk = (frames // int(round(SOFT_BLOCK_S * 25.0))).astype(int)
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, sd, size=(int(blk.max()) + 1, pos.shape[1]))
        return base(pos, cam, scale + eps[blk], frames)

    return fn


def _ellipse(pos: np.ndarray, cam: np.ndarray, scale: float, frames: np.ndarray) -> np.ndarray:
    """Elliptical (vignette-like) window: ``(dx/scale)^2 + (dy/34)^2 <= 1``."""
    dx = (pos[:, :, 0] - cam[:, None, 0]) / scale
    dy = (pos[:, :, 1] - cam[:, None, 1]) / (PITCH_W / 2)
    return dx**2 + dy**2 <= 1.0


def _trapezoid(pos: np.ndarray, cam: np.ndarray, scale: float, frames: np.ndarray) -> np.ndarray:
    """Real broadcast footprint (SkillCorner-measured): widens away from the camera side."""
    y = pos[:, :, 1]
    t = (y - TRAP_Y_NEAR) / (TRAP_Y_FAR - TRAP_Y_NEAR)
    hw = scale * (TRAP_HW_NEAR + t * (TRAP_HW_FAR - TRAP_HW_NEAR))
    return (y >= TRAP_Y_NEAR) & (np.abs(pos[:, :, 0] - cam[:, None, 0]) <= hw)


GEOMS: tuple[Geometry, ...] = (
    Geometry("rect_base", "TRAINING geometry: hard rectangle, full pitch height (control)",
             _rect(PITCH_W / 2), in_train_family=True),
    Geometry("soft_edge", "rectangle with a feathered edge (per-slot N(0, 4 m), 10 s blocks)",
             _soft(4.0, _rect(PITCH_W / 2)), diag=_rect(PITCH_W / 2)),
    Geometry("aspect_25", "rectangle, half-height 25 m (squarer window, wider in x)",
             _rect(25.0), in_train_family=True),
    Geometry("aspect_15", "rectangle, half-height 15 m (tight zoom, much wider in x)",
             _rect(15.0), in_train_family=True),
    Geometry("lag_1s", "rectangle, camera reacts 1.0 s late", _rect(PITCH_W / 2), lag_s=1.0,
             in_train_family=True),
    Geometry("lag_2s", "rectangle, camera reacts 2.0 s late", _rect(PITCH_W / 2), lag_s=2.0,
             in_train_family=True),
    Geometry("ellipse", "elliptical / vignette window", _ellipse),
    Geometry("trapezoid_sc", "SkillCorner-measured broadcast footprint (widens with distance)",
             _trapezoid, lo=0.2, hi=6.0),
    Geometry("sc_realistic", "SkillCorner footprint + 0.5 s camera lag + ~2 m feathered edge",
             _soft(0.12, _trapezoid), lag_s=0.5, lo=0.2, hi=6.0, diag=_trapezoid),
)
GEOM_BY_NAME = {g.name: g for g in GEOMS}
# v1.1 domain randomisation trains on these TRAIN censorings (rectangle family only, so the
# trapezoid / ellipse / soft-edge evaluations stay out-of-family).
MITIGATION_TRAIN = ("rect_base", "aspect_25", "aspect_15", "lag_1s")


# --------------------------------------------------------------------------------------
# censoring + sample construction
# --------------------------------------------------------------------------------------


def lagged_cam(cam: np.ndarray, fps: float, lag_s: float) -> np.ndarray:
    """Delay the camera track by ``lag_s`` seconds (the first frames repeat the start)."""
    if lag_s <= 0:
        return cam
    k = int(round(lag_s * fps))
    idx = np.maximum(np.arange(cam.shape[0]) - k, 0)
    return cam[idx]


def censor(game: dict, geom: Geometry, scale: float) -> np.ndarray:
    """Apply a geometry to a loaded game and return the visibility mask."""
    truth = game["truth"]
    cam = lagged_cam(game["cam"], game["fps"], geom.lag_s)
    frames = np.arange(truth.shape[0])
    return geom.mask(truth, cam, scale, frames) & np.isfinite(truth[:, :, 0])


def tune_scale(game: dict, geom: Geometry, target: float = TARGET_VISIBLE) -> tuple[float, float]:
    """Bisect a geometry's scale on TRAIN so the mean visible count hits ``target``.

    Args:
        game: Loaded game bundle (TRAIN).
        geom: Geometry to tune.
        target: Mean visible players per frame.

    Returns:
        Tuple ``(scale, achieved_mean)``.
    """
    lo, hi, achieved = geom.lo, geom.hi, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        achieved = float(censor(game, geom, mid).sum(axis=1).mean())
        if achieved < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), achieved


def load_game(name: str) -> dict:
    """Load one Metrica game and everything that does not depend on the censoring geometry."""
    truth, ball, period, fps, n_home = load_metrica_match(
        METRICA / f"{name}_RawTrackingData_Home_Team.csv",
        METRICA / f"{name}_RawTrackingData_Away_Team.csv",
    )
    return {
        "truth": truth, "ball": ball, "period": period, "fps": fps,
        "cam": smooth_camera(ball, fps), "ranges": team_ranges(n_home, truth.shape[1]),
        "vel": estimate_velocity(truth, period, fps),
    }


def build_split(
    game: dict, geom: Geometry, scale: float, coeffs: dict | None
) -> tuple[dict, dict, dict, np.ndarray]:
    """Censor a game, derive fields, and collect the hidden samples.

    Args:
        game: Loaded game bundle.
        geom: Censoring geometry.
        scale: Tuned geometry scale.
        coeffs: Frozen slot-model coefficients, or ``None`` to fit them (TRAIN only).

    Returns:
        Tuple ``(fields, samples, coeffs, visible)``.
    """
    visible = censor(game, geom, scale)
    cent = visible_centroid(game["truth"], visible, game["ranges"])
    if coeffs is None:
        coeffs = fit_slot_model(game["truth"], cent, game["cam"], game["ranges"])
    fields = derive_fields(
        game["truth"], visible, game["ball"], game["cam"], game["period"], game["fps"],
        game["ranges"], coeffs, BAR_HALFLIFE_S,
    )
    samples = collect_samples(
        game["truth"], visible, game["period"], game["fps"], game["vel"],
        np.asarray(fields["slot"]), fields={"b6": np.asarray(fields["b6"])},
    )
    return fields, samples, coeffs, visible


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


class Tee:
    """Print an ASCII line and keep it for the markdown run log."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        """Print ``text`` (flushed -- this tool runs for an hour) and buffer it for the log."""
        print(text, flush=True)
        self.lines.append(text)


def _rmse(err: np.ndarray) -> float:
    """Root-mean-square of an error array."""
    return float(np.sqrt(np.mean(err**2)))


def _err(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Euclidean per-sample error in metres."""
    return np.linalg.norm(pred - target, axis=1)


def _subset(d: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    """Row-select every array in a sample dict."""
    return {k: v[m] for k, v in d.items()}


def _blocks(samples: dict[str, np.ndarray], fps: float) -> np.ndarray:
    """1-minute block ids (period-aware) for the block bootstrap."""
    return samples["period"] * 100000 + (samples["frame"] / (BLOCK_S * fps)).astype(int)


def inside_rate(
    pos: np.ndarray, samples: dict[str, np.ndarray], game: dict, geom: Geometry, scale: float
) -> float:
    """Fraction of predicted positions falling inside the (impossible) visible region.

    Every sample is a player who is hidden at that frame, so a correct prediction is outside the
    visible region by construction. This is the 2-D version of the gate run's x-band diagnostic,
    and it is exact: the truth scores 0.0% for every geometry.

    Args:
        pos: Predicted positions, shape ``(M, 2)``.
        samples: The samples those predictions belong to.
        game: Loaded game bundle.
        geom: Censoring geometry.
        scale: Tuned geometry scale.

    Returns:
        Fraction in [0, 1].
    """
    f = samples["frame"].astype(int)
    cam = lagged_cam(game["cam"], game["fps"], geom.lag_s)[f]
    mask = geom.diag or geom.mask
    return float(mask(pos[:, None, :], cam, scale, f)[:, 0].mean())


def off_pitch_rate(pos: np.ndarray) -> float:
    """Fraction of positions outside the pitch rectangle by more than 1 m."""
    out = (
        (pos[:, 0] < -1.0) | (pos[:, 0] > PITCH_L + 1.0)
        | (pos[:, 1] < -1.0) | (pos[:, 1] > PITCH_W + 1.0)
    )
    return float(out.mean())


def geometry_row(
    out: Tee,
    geom: Geometry,
    scale: float,
    achieved: float,
    hold: dict[str, np.ndarray],
    v1_err: np.ndarray,
    b7_err: np.ndarray,
    blocks: np.ndarray,
) -> dict[str, object]:
    """Print the per-bucket v1-vs-anchor table for one geometry and return its summary."""
    bkt = bucket_of(hold["tsls"])
    out(f"\n--- {geom.name}: {geom.desc}")
    out(f"    scale={scale:.3f} tuned on TRAIN -> {achieved:.2f} visible/frame; "
        f"holdout n={v1_err.size}")
    out("horizon   |      n |   B7   |   v1   |  margin  [95% block CI]  | % of B7")
    out("-" * 76)
    per_bucket: list[float] = []
    for b, lab in enumerate(BIN_LABELS):
        m = bkt == b
        if not m.any():
            per_bucket.append(float("nan"))
            continue
        rb, rv = _rmse(b7_err[m]), _rmse(v1_err[m])
        lo, hi = paired_rmse_ci(b7_err[m], v1_err[m], blocks[m])
        per_bucket.append(rv - rb)
        out(f"{lab:9s} | {int(m.sum()):6d} | {rb:6.2f} | {rv:6.2f} | {rv - rb:+7.2f} "
            f"[{lo:+6.2f},{hi:+6.2f}] | {100 * (rv - rb) / rb:+6.1f}%")
    rb, rv = _rmse(b7_err), _rmse(v1_err)
    lo, hi = paired_rmse_ci(b7_err, v1_err, blocks)
    out(f"{'ALL':9s} | {v1_err.size:6d} | {rb:6.2f} | {rv:6.2f} | {rv - rb:+7.2f} "
        f"[{lo:+6.2f},{hi:+6.2f}] | {100 * (rv - rb) / rb:+6.1f}%")
    return {
        "name": geom.name, "scale": scale, "visible": achieved, "n": int(v1_err.size),
        "b7": rb, "v1": rv, "margin": rv - rb, "ci": (lo, hi), "buckets": per_bucket,
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def main() -> None:
    """Fit the frozen v1 on the training geometry, then score it under every other geometry."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geoms", default="", help="comma-separated geometry subset (default: all)")
    ap.add_argument("--mitigate", action="store_true",
                    help="also fit and score v1.1 with train-time camera-shape randomisation")
    ap.add_argument("--out", default="results/B4_TRANSFER_M3_runlog.md", help="run log path")
    ap.add_argument("--cache-dir", default="",
                    help="joblib cache for the fitted heads; the fits are deterministic and cost "
                         "~10 min (v1) and ~20 min (v1.1), so a resumed run should reuse them")
    args = ap.parse_args()
    names = [n for n in args.geoms.split(",") if n] or [g.name for g in GEOMS]
    out = Tee()

    out("=" * 100)
    out("B4 M3 -- TRANSFER: the FROZEN v1 under censoring geometries it never trained on")
    out("=" * 100)
    t0 = time.time()
    g1, g2 = load_game("Sample_Game_1"), load_game("Sample_Game_2")
    out(f"loaded both Metrica games in {time.time() - t0:.0f} s; fps={g1['fps']:.1f}")

    base = GEOM_BY_NAME["rect_base"]
    scale_base, ach_base = tune_scale(g1, base)
    ref_half_w, _ = tune_window(g1["truth"], g1["cam"])
    assert abs(scale_base - ref_half_w) < 1e-6, (scale_base, ref_half_w)
    out(f"training geometry: half-width {scale_base:.3f} m ({ach_base:.2f} visible/frame) -- "
        "identical to the frozen protocol's tune_window")

    train: dict[str, object] = {}

    def train_bits() -> dict[str, object]:
        """Censor TRAIN and collect its samples (only needed when a cache misses)."""
        if not train:
            f1, s1, coeffs, _ = build_split(g1, base, scale_base, None)
            tau, _ = fit_decay_tau(s1)
            train.update({"f1": f1, "s1": s1, "coeffs": coeffs, "tau": tau,
                          "w7": b7_weights(s1, tau)})
        return train

    meta = cached(
        args.cache_dir, "train_meta",
        lambda: {k: train_bits()[k] for k in ("coeffs", "tau", "w7")}, out,
    )
    coeffs, tau, w7 = meta["coeffs"], float(meta["tau"]), np.asarray(meta["w7"])
    out(f"frozen anchor: tau={tau:.2f}s, B7 weights {np.round(w7, 2).tolist()} (TRAIN only)")

    def build_v1() -> Heads:
        """Fit v1 exactly as the gate run did: residual on B7, attack-aligned, on TRAIN."""
        f1, s1 = train_bits()["f1"], train_bits()["s1"]
        anc = b7_position(s1, tau, w7)
        x = build_features(f1, s1, anc)
        r = (s1["target"] - anc) * sample_signs(f1, s1)[:, None]
        return fit_heads(x, r, PARAMS)

    t0 = time.time()
    heads = cached(args.cache_dir, "v1_heads", build_v1, out)
    train.clear()
    out(f"v1: all {2 * len(QUANTILES)} heads ready in {time.time() - t0:.0f} s -- all five "
        "quantiles are needed because predict_heads SORTS across them to remove crossing, so the "
        "point estimate is not the raw 0.50 head")

    heads11 = None
    if args.mitigate:
        heads11 = cached(
            args.cache_dir, "v11_heads", lambda: _fit_mitigated(g1, coeffs, tau, w7, out), out
        )

    summaries: list[dict[str, object]] = []
    sums11: list[dict[str, object]] = []
    for name in names:
        geom = GEOM_BY_NAME[name]
        scale, ach = tune_scale(g1, geom)
        f2, s2, _, _ = build_split(g2, geom, scale, coeffs)
        hold = _subset(s2, s2["period"] == 2)
        anchor = b7_position(hold, tau, w7)
        sgn = sample_signs(f2, hold)
        x = build_features(f2, hold, anchor)
        mu, _ = v1_prediction(heads, x, anchor, sgn)
        b7_err = _err(anchor, hold["target"])
        v1_err = _err(mu, hold["target"])
        blocks = _blocks(hold, g2["fps"])
        row = geometry_row(out, geom, scale, ach, hold, v1_err, b7_err, blocks)
        row["inside_truth"] = inside_rate(hold["target"], hold, g2, geom, scale)
        row["inside_b7"] = inside_rate(anchor, hold, g2, geom, scale)
        row["inside_v1"] = inside_rate(mu, hold, g2, geom, scale)
        row["inside_hold"] = inside_rate(hold["hold"], hold, g2, geom, scale)
        out(f"    inside the impossible visible region: truth {100 * row['inside_truth']:.1f}% "
            f"| last-seen {100 * row['inside_hold']:.1f}% | B7 {100 * row['inside_b7']:.1f}% "
            f"| v1 {100 * row['inside_v1']:.1f}%")
        summaries.append(row)
        if heads11 is not None:
            mu11, _ = v1_prediction(heads11, x, anchor, sgn)
            e11 = _err(mu11, hold["target"])
            lo, hi = paired_rmse_ci(b7_err, e11, blocks)
            r11 = {
                "name": geom.name, "b7": _rmse(b7_err), "v1": _rmse(v1_err), "v11": _rmse(e11),
                "margin": _rmse(e11) - _rmse(b7_err), "ci": (lo, hi),
                "inside_v11": inside_rate(mu11, hold, g2, geom, scale),
                "in_family": geom.in_train_family,
            }
            out(f"    v1.1 (camera-randomised training): {r11['v11']:.2f} m, margin "
                f"{r11['margin']:+.2f} [{lo:+.2f},{hi:+.2f}], inside "
                f"{100 * r11['inside_v11']:.1f}%"
                f"  [{'in' if geom.in_train_family else 'OUT of'}-family]")
            sums11.append(r11)
        del f2, s2, x

    _summary(out, summaries, sums11)
    _write(out, args.out)


def cached(cache_dir: str, name: str, build: Callable[[], object], out: Tee) -> object:
    """Load a fitted object from a joblib cache, or build and store it.

    The fits are deterministic (``random_state=0``, fixed data), so caching changes no number --
    it only makes a resumed run cheap after an interrupted one. This matters because the sweep
    takes over an hour and the whole point of the exercise is that nothing is refitted.

    Args:
        cache_dir: Directory for the cache; empty disables caching.
        name: Cache key.
        build: Callable that produces the object.
        out: Report sink.

    Returns:
        The cached or freshly built object.
    """
    import joblib  # noqa: PLC0415  (ships with scikit-learn; not a new dependency)

    if not cache_dir:
        return build()
    path = Path(cache_dir) / f"{name}.joblib"
    if path.exists():
        out(f"loaded {name} from cache {path}")
        return joblib.load(path)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return obj


def _fit_mitigated(g1: dict, coeffs: dict, tau: float, w7: np.ndarray, out: Tee) -> Heads:
    """Fit v1.1: same architecture, TRAIN pooled over several rectangle-family censorings.

    Each censoring contributes every second sample (25 fps rows are near-duplicates anyway), so
    v1.1 sees ~2x v1's rows spread over 4 window shapes instead of 1x on one shape.

    Args:
        g1: TRAIN game bundle.
        coeffs: Frozen slot-model coefficients (fitted under the base geometry).
        tau: Frozen decay constant.
        w7: Frozen B7 blend weights.
        out: Report sink.

    Returns:
        Fitted median heads for v1.1.
    """
    out("\nMITIGATION v1.1 -- train-time camera-shape randomisation (a NEW model; v1's gate")
    out(f"record stands untouched). TRAIN censorings pooled: {list(MITIGATION_TRAIN)}")
    xs, rs = [], []
    for name in MITIGATION_TRAIN:
        geom = GEOM_BY_NAME[name]
        scale, ach = tune_scale(g1, geom)
        f, s, _, _ = build_split(g1, geom, scale, coeffs)
        anc = b7_position(s, tau, w7)
        sgn = sample_signs(f, s)
        xs.append(build_features(f, s, anc)[::2])
        rs.append(((s["target"] - anc) * sgn[:, None])[::2])
        out(f"  {name:12s} scale={scale:.2f} visible={ach:.2f} n={xs[-1].shape[0]} (stride 2)")
        del f, s
    x = np.concatenate(xs)
    r = np.concatenate(rs)
    del xs, rs
    t0 = time.time()
    heads = fit_heads(x, r, PARAMS)
    out(f"  fitted v1.1 ({2 * len(QUANTILES)} heads) on n={x.shape[0]} in {time.time() - t0:.0f} s")
    return heads


def _summary(out: Tee, rows: list[dict[str, object]], rows11: list[dict[str, object]]) -> None:
    """Print the headline per-geometry margin table."""
    base = next((r for r in rows if r["name"] == "rect_base"), None)
    out("\n" + "=" * 100)
    out("SUMMARY -- margin = RMSE(v1) - RMSE(B7 anchor), metres. Negative = v1 better.")
    out("Absolute RMSE is NOT comparable across geometries (different hidden sets); the MARGIN is.")
    out("=" * 100)
    out("geometry     | vis/fr |      n |    B7 |    v1 | margin [95% CI]       | kept | v1 inside")
    out("-" * 100)
    for r in rows:
        kept = ""
        if base is not None and float(base["margin"]) != 0.0:
            kept = f"{100 * float(r['margin']) / float(base['margin']):4.0f}%"
        ci = r["ci"]
        out(f"{r['name']:12s} | {float(r['visible']):6.2f} | {int(r['n']):6d} | "
            f"{float(r['b7']):5.2f} | {float(r['v1']):5.2f} | {float(r['margin']):+6.2f} "
            f"[{ci[0]:+5.2f},{ci[1]:+5.2f}] | {kept:>4s} | "
            f"{100 * float(r['inside_v1']):5.1f}%")
    out("'kept' = this geometry's margin as a percentage of the training geometry's margin.")
    if rows11:
        out("\nv1.1 (camera-shape randomised training) vs v1, same geometries:")
        out("geometry     | family |    v1 |  v1.1 | v1 margin | v1.1 margin | v1.1 inside")
        out("-" * 84)
        by_name = {r["name"]: r for r in rows}
        for r in rows11:
            v1m = float(by_name[r["name"]]["margin"])
            out(f"{r['name']:12s} | {'in ' if r['in_family'] else 'OUT':6s} | "
                f"{float(r['v1']):5.2f} | {float(r['v11']):5.2f} | {v1m:+9.2f} | "
                f"{float(r['margin']):+11.2f} | {100 * float(r['inside_v11']):10.1f}%")


def _write(out: Tee, path: str) -> None:
    """Write the verbatim run log next to the hand-written analysis."""
    dst = REPO / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    head = [
        "# B4 M3 transfer -- run log",
        "",
        "Generated by `python -m tools.imputation_b4_transfer --mitigate`. The hand-written",
        "analysis and verdict live in `results/B4_TRANSFER_M3.md` and are not overwritten by a",
        "re-run.",
        "",
        "```",
    ]
    dst.write_text("\n".join(head + out.lines + ["```", ""]), encoding="utf-8")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
