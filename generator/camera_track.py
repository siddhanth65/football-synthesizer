"""Temporal camera modelling: decompose per-frame ground homographies, smooth, recompose.

Our calibration is per-frame and memoryless: every frame's homography is solved from that frame's
keypoints alone (``generator.calibrate.PnLCalibCalibrator``), and a frame the solver cannot answer
is patched afterwards by linear interpolation of *neighbouring homographies*
(:func:`generator.postprocess.fill_calibration_gaps`). Both steps ignore the fact that a broadcast
camera is a physical object with continuous pan / tilt / zoom.

This module supplies the missing temporal layer, in the parameterisation the literature uses
(BroadTrack, WACV'25; BHITK's two-stage Kalman over homographies):

1. :func:`decompose` turns an image -> pitch ground homography into
   ``[pan, tilt, roll, log f, log aspect, Cx, Cy, Cz]`` under a pinhole camera whose principal point
   sits at the image centre. The two focal lengths are kept **separate** (``log f`` is their
   geometric mean, ``log aspect`` is ``log fy/fx``) because PnLCalib's ``cv2.calibrateCamera`` call
   does not fix the aspect ratio and its solves come back at ``fy/fx ~ 0.94-0.97``: a square-pixel
   model reproduces only ~40-97% of the frames depending on the sequence, while this one reproduces
   essentially all of them. The round trip is checked per frame, never assumed.
2. :func:`rts_smooth` runs a forward Kalman filter + Rauch-Tung-Striebel backward pass over the
   parameter time series with a constant-velocity process model. Measurement variance is set from
   the per-frame solver residual; process noise is estimated from the series itself, so there is no
   tuned dial. Frames with no solve are simply missing measurements -- the smoother predicts them.
3. :func:`compose` maps parameters back to a homography.

Nothing here reads ground truth, and nothing here is enabled anywhere by default: it is a component
for a registered experiment (v10-W4), not a change to the shipped chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Principal point assumed by :func:`decompose` / :func:`compose` (SoccerNet-GSR is 1920x1080).
PRINCIPAL_POINT = (960.0, 540.0)
#: Uncentred (corner-origin) -> centred (centre-spot origin) pitch metres, PnLCalib's world frame.
_TO_CENTRED = np.array([[1.0, 0.0, -52.5], [0.0, 1.0, -34.0], [0.0, 0.0, 1.0]])
#: Image points the round-trip / clamp checks are evaluated on (a 3x3 grid inside the frame).
CHECK_GRID = np.array([[x, y] for x in (200.0, 960.0, 1720.0) for y in (300.0, 700.0, 1000.0)])
#: Parameter order of the state vector.
PARAMS = ("pan", "tilt", "roll", "log_f", "log_aspect", "cx", "cy", "cz")


@dataclass(frozen=True)
class Pose:
    """One frame's camera pose. Angles in radians, ``log_f`` in log pixels, centre in metres."""

    pan: float
    tilt: float
    roll: float
    log_f: float
    log_aspect: float
    cx: float
    cy: float
    cz: float

    def to_array(self) -> np.ndarray:
        """The pose as an ``(8,)`` vector in :data:`PARAMS` order."""
        return np.array([self.pan, self.tilt, self.roll, self.log_f, self.log_aspect,
                         self.cx, self.cy, self.cz])

    def focals(self) -> tuple[float, float]:
        """``(fx, fy)`` in pixels."""
        return (float(np.exp(self.log_f - self.log_aspect / 2.0)),
                float(np.exp(self.log_f + self.log_aspect / 2.0)))


def _rz(a: float) -> np.ndarray:
    """Rotation about z by ``a`` radians."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    """Rotation about x by ``a`` radians."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _euler_zxz(r: np.ndarray) -> tuple[float, float, float]:
    """``R = Rz(roll) Rx(tilt) Rz(pan)`` -> ``(pan, tilt, roll)`` radians (pure)."""
    tilt = float(np.arccos(np.clip(r[2, 2], -1.0, 1.0)))
    if abs(np.sin(tilt)) < 1e-9:  # gimbal lock: a broadcast camera never sits here
        return float(np.arctan2(r[1, 0], r[0, 0])), tilt, 0.0
    pan = float(np.arctan2(r[2, 0], r[2, 1]))
    roll = float(np.arctan2(r[0, 2], -r[1, 2]))
    return pan, tilt, roll


def decompose(h: np.ndarray, *, pp: tuple[float, float] = PRINCIPAL_POINT) -> Pose | None:
    """Ground homography (image -> uncentred pitch metres) -> camera pose.

    Args:
        h: ``(3, 3)`` image -> pitch homography, the project convention (``[0,105] x [0,68]`` m).
        pp: Principal point in pixels.

    Returns:
        The pose, or ``None`` when the homography is degenerate or implies a non-physical camera
        (negative focal length squared, camera below the pitch).
    """
    if not np.isfinite(h).all():
        return None
    try:
        g = np.linalg.inv(_TO_CENTRED @ h)  # centred pitch -> image
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(g).all() or abs(g[2, 2]) < 1e-15:
        return None
    g = np.array([[1.0, 0.0, -pp[0]], [0.0, 1.0, -pp[1]], [0.0, 0.0, 1.0]]) @ (g / g[2, 2])
    a, b = g[:, 0], g[:, 1]
    # r1 . r2 = 0 and |r1| = |r2| are two linear equations in (1/fx^2, 1/fy^2) under
    # K = diag(fx, fy, 1); solving both is what lets a non-square-pixel solve round-trip exactly.
    mat = np.array([[a[0] * b[0], a[1] * b[1]], [a[0] ** 2 - b[0] ** 2, a[1] ** 2 - b[1] ** 2]])
    rhs = np.array([-a[2] * b[2], -(a[2] ** 2 - b[2] ** 2)])
    try:
        uv = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(uv).all() or uv[0] <= 0.0 or uv[1] <= 0.0:
        return None
    fx, fy = float(1.0 / np.sqrt(uv[0])), float(1.0 / np.sqrt(uv[1]))
    m = np.diag([1.0 / fx, 1.0 / fy, 1.0]) @ g
    scale = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))  # |r1| = |r2| = 1
    m = m * scale
    r1, r2, t = m[:, 0], m[:, 1], m[:, 2]
    rot = np.column_stack([r1, r2, np.cross(r1, r2)])
    u, _s, vt = np.linalg.svd(rot)
    rot = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    centre = -rot.T @ t
    if centre[2] <= 0.0:
        # The overall scale of [r1 r2 t] is signed: negating it flips r1, r2 and t but leaves
        # r3 = r1 x r2 alone, which is the same camera seen from the other side of the pitch plane.
        # Take the branch with the camera above the pitch so the angle series has one convention.
        rot = rot @ np.diag([-1.0, -1.0, 1.0])
        t = -t
        centre = -rot.T @ t
        if centre[2] <= 0.0:
            return None
    pan, tilt, roll = _euler_zxz(rot)
    return Pose(pan, tilt, roll, float(np.log(np.sqrt(fx * fy))), float(np.log(fy / fx)),
                *(float(v) for v in centre))


def compose(pose: Pose, *, pp: tuple[float, float] = PRINCIPAL_POINT) -> np.ndarray | None:
    """Camera pose -> image -> uncentred-pitch ground homography (the inverse of :func:`decompose`)."""
    rot = _rz(pose.roll) @ _rx(pose.tilt) @ _rz(pose.pan)
    fx, fy = pose.focals()
    k = np.array([[fx, 0.0, pp[0]], [0.0, fy, pp[1]], [0.0, 0.0, 1.0]])
    centre = np.array([pose.cx, pose.cy, pose.cz])
    g = k @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ centre])
    if abs(float(np.linalg.det(g))) < 1e-18:
        return None
    h = np.linalg.inv(_TO_CENTRED) @ np.linalg.inv(g)
    if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-15:
        return None
    return h / h[2, 2]


def grid_shift_m(h_a: np.ndarray, h_b: np.ndarray, grid: np.ndarray = CHECK_GRID) -> float:
    """Median pitch-metre distance between the two homographies' images of ``grid`` (pure)."""
    from generator.calibrate import apply_homography  # noqa: PLC0415 - avoid an import cycle

    pa, pb = apply_homography(h_a, grid), apply_homography(h_b, grid)
    if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
        return float("inf")
    return float(np.median(np.linalg.norm(pa - pb, axis=1)))


def pose_series(homs: dict[int, np.ndarray], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a ``{frame: H}`` map into an ``(n_frames, 7)`` series + a validity mask.

    Angles are unwrapped along the valid frames so the series is continuous for a linear smoother.
    A frame whose decomposition fails, or whose round trip does not reproduce the homography, is
    marked invalid: the smoother then treats it as a missing measurement rather than trusting a
    parameterisation that does not describe it.
    """
    z = np.full((n_frames, len(PARAMS)), np.nan)
    for fr, h in homs.items():
        if not 0 <= fr < n_frames:
            continue
        pose = decompose(h)
        if pose is None:
            continue
        back = compose(pose)
        if back is None or grid_shift_m(h, back) > 0.05:  # 5 cm: the model must be exact, not close
            continue
        z[fr] = pose.to_array()
    valid = np.isfinite(z).all(axis=1)
    idx = np.flatnonzero(valid)
    for j in range(3):  # pan / tilt / roll live on a circle
        if len(idx):
            z[idx, j] = np.unwrap(z[idx, j])
    return z, valid


def _noise_estimates(x: np.ndarray) -> tuple[float, float]:
    """``(measurement sd, acceleration sd)`` of one series from its second differences (pure).

    For ``x = s + n`` with smooth ``s`` and white ``n``, the second difference has variance
    ``6 var(n) + var(acceleration)``. A robust (MAD) variance of the second difference therefore
    splits into the two, and both the Kalman ``R`` and ``Q`` follow from the data with no dial.
    """
    d2 = np.diff(x, n=2)
    if len(d2) < 8:
        return 1e-6, 1e-6
    mad = float(np.median(np.abs(d2 - np.median(d2)))) * 1.4826
    var2 = max(mad, 1e-12) ** 2
    # split: the high-frequency part is noise, the rest is real acceleration. A pure random walk in
    # acceleration would show var(d2) = var(a); we attribute the smaller of the two to acceleration.
    var_n = var2 / 6.0
    d1 = np.diff(x)
    mad1 = float(np.median(np.abs(d1 - np.median(d1)))) * 1.4826
    var_a = max(var2 - 6.0 * var_n, (mad1 ** 2) * 1e-4, 1e-14)
    return float(np.sqrt(var_n)), float(np.sqrt(var_a))


def rts_smooth(x: np.ndarray, valid: np.ndarray, *, meas_sd: float, accel_sd: float,
               weight: np.ndarray | None = None) -> np.ndarray:
    """Constant-velocity Kalman filter + RTS backward pass over one scalar series (pure).

    Args:
        x: ``(N,)`` measurements; entries where ``valid`` is False are ignored.
        valid: ``(N,)`` measurement mask.
        meas_sd: Measurement standard deviation.
        accel_sd: Process (acceleration) standard deviation, per frame.
        weight: Optional ``(N,)`` per-frame multiplier on the measurement standard deviation
            (``1.0`` = as given); this is where the solver's own residual enters.

    Returns:
        ``(N,)`` smoothed values -- including at frames with no measurement, where the value is the
        model's own prediction.
    """
    n = len(x)
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    q = (accel_sd ** 2) * np.array([[0.25, 0.5], [0.5, 1.0]])  # CV discretisation, dt = 1 frame
    h = np.array([[1.0, 0.0]])
    xf = np.zeros((n, 2))
    pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))
    pp = np.zeros((n, 2, 2))
    first = int(np.argmax(valid)) if valid.any() else 0
    state = np.array([x[first] if valid.any() else 0.0, 0.0])
    cov = np.diag([max(meas_sd, 1e-9) ** 2 * 1e2, max(accel_sd, 1e-9) ** 2 * 1e4])
    for i in range(n):
        state = f @ state
        cov = f @ cov @ f.T + q
        xp[i], pp[i] = state, cov
        if valid[i]:
            r = (meas_sd * (1.0 if weight is None else float(weight[i]))) ** 2
            s = float((h @ cov @ h.T)[0, 0]) + max(r, 1e-18)
            k = (cov @ h.T / s).ravel()
            state = state + k * (x[i] - float(state[0]))
            cov = (np.eye(2) - np.outer(k, h)) @ cov
        xf[i], pf[i] = state, cov
    xs = xf.copy()
    for i in range(n - 2, -1, -1):
        try:
            gain = pf[i] @ f.T @ np.linalg.inv(pp[i + 1])
        except np.linalg.LinAlgError:  # pragma: no cover - a singular predicted covariance
            continue
        xs[i] = xf[i] + gain @ (xs[i + 1] - xp[i + 1])
    return xs[:, 0]


def grid_series(homs: dict[int, np.ndarray], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode each homography as the pitch coordinates of :data:`CHECK_GRID` (the mapping space).

    The physical decomposition is exact but **degenerate**: on our per-frame solves the camera
    position and the focal length trade off against each other at metre scale, so smoothing the
    parameters independently can move the image->pitch map even where the map itself was fine. This
    encoding has no such freedom -- it smooths what the homography actually does.
    """
    from generator.calibrate import apply_homography  # noqa: PLC0415

    z = np.full((n_frames, 2 * len(CHECK_GRID)), np.nan)
    for fr, h in homs.items():
        if not 0 <= fr < n_frames:
            continue
        p = apply_homography(h, CHECK_GRID)
        if np.isfinite(p).all():
            z[fr] = p.reshape(-1)
    return z, np.isfinite(z).all(axis=1)


def grid_compose(vec: np.ndarray) -> np.ndarray | None:
    """Re-fit a homography from :data:`CHECK_GRID` and its smoothed pitch coordinates (pure)."""
    from generator.calibrate import estimate_homography  # noqa: PLC0415

    h = estimate_homography(CHECK_GRID, vec.reshape(-1, 2), use_cv2=False)
    return h if h is not None and np.isfinite(h).all() else None


def smooth_homographies(homs: dict[int, np.ndarray], n_frames: int, *,
                        err_m: dict[int, float] | None = None,
                        clamp_m: float = 2.0,
                        dead_only: bool = False,
                        mode: str = "camera") -> tuple[dict[int, np.ndarray], dict]:
    """RTS-smooth a sequence's per-frame homographies; predict the frames that have none.

    Args:
        homs: ``{frame: H}`` from the per-frame solver (image -> uncentred pitch metres).
        n_frames: Length of the sequence.
        err_m: Optional ``{frame: solver reprojection error in metres}``; a frame whose solve fits
            worse than the sequence median is trusted proportionally less.
        clamp_m: A smoothed homography that moves :data:`CHECK_GRID` further than this from the
            frame's own solve is refused and the solve is kept (the 2024 GSR winner's guard,
            expressed in metres). ``inf`` disables the clamp.
        dead_only: Emit the smoothed homography **only** for frames the solver did not answer, and
            pass every solved frame through untouched.
        mode: ``"camera"`` smooths ``[pan, tilt, roll, log f, log aspect, C]``; ``"mapping"`` smooths
            the pitch coordinates of a fixed image grid and re-fits the homography from them.

    Returns:
        ``({frame: H}, stats)``.
    """
    if mode == "camera":
        z, valid = pose_series(homs, n_frames)
        names: tuple[str, ...] = PARAMS
    elif mode == "mapping":
        z, valid = grid_series(homs, n_frames)
        names = tuple(f"g{i // 2}{'xy'[i % 2]}" for i in range(z.shape[1]))
    else:
        raise ValueError(f"mode must be 'camera' or 'mapping', got {mode!r}")
    stats: dict = {"frames": n_frames, "solved": int(len(homs)), "parameterised": int(valid.sum()),
                   "mode": mode, "clamped": 0, "predicted": 0, "compose_failed": 0}
    if valid.sum() < 8:  # nothing to smooth
        stats["skipped"] = True
        return ({} if dead_only else dict(homs)), stats
    idx = np.flatnonzero(valid)
    weight = np.ones(n_frames)
    if err_m:
        med = float(np.median([err_m[f] for f in idx if f in err_m] or [1.0]))
        for f in idx:
            weight[f] = float(np.clip(err_m.get(int(f), med) / max(med, 1e-6), 0.5, 5.0))
    filled = np.zeros_like(z)
    noise = {}
    for j, name in enumerate(names):
        series = np.interp(np.arange(n_frames), idx, z[idx, j])  # only feeds the noise estimate
        meas_sd, accel_sd = _noise_estimates(z[idx, j])
        noise[name] = {"meas_sd": meas_sd, "accel_sd": accel_sd}
        filled[:, j] = rts_smooth(series, valid, meas_sd=meas_sd, accel_sd=accel_sd, weight=weight)
    stats["noise"] = noise
    out: dict[int, np.ndarray] = {}
    for fr in range(n_frames):
        h_new = compose(Pose(*filled[fr])) if mode == "camera" else grid_compose(filled[fr])
        raw = homs.get(fr)
        if h_new is None:
            stats["compose_failed"] += 1
            if raw is not None and not dead_only:
                out[fr] = raw
            continue
        if raw is None:
            stats["predicted"] += 1
            out[fr] = h_new
            continue
        if dead_only:
            continue
        if grid_shift_m(raw, h_new) > clamp_m:
            stats["clamped"] += 1
            out[fr] = raw
        else:
            out[fr] = h_new
    return out, stats


def _demo() -> None:
    """Self-check: the decomposition round-trips, and the smoother removes noise it is given."""
    from generator.calibrate import apply_homography  # noqa: PLC0415

    truth = Pose(pan=0.35, tilt=1.85, roll=0.02, log_f=float(np.log(3800.0)),
                 log_aspect=float(np.log(0.95)), cx=2.0, cy=-55.0, cz=15.0)
    h = compose(truth)
    assert h is not None
    got = decompose(h)
    assert got is not None
    for a, b in zip(truth.to_array(), got.to_array(), strict=True):
        assert abs(a - b) < 1e-6, (truth, got)
    assert grid_shift_m(h, compose(got)) < 1e-9

    # a panning camera measured with noise: the smoother must land closer to the truth than the
    # measurements do, and must predict the gap it is never shown.
    rng = np.random.default_rng(0)
    n = 200
    pans = 0.35 + 0.0008 * np.arange(n)
    homs, clean = {}, {}
    for i, p in enumerate(pans):
        clean[i] = compose(Pose(p, 1.85, 0.02, truth.log_f, truth.log_aspect, 2.0, -55.0, 15.0))
        if 100 <= i < 120:
            continue  # a 20-frame dead run
        noisy = Pose(p + rng.normal(0, 3e-4), 1.85 + rng.normal(0, 3e-4), 0.02, truth.log_f,
                     truth.log_aspect, 2.0, -55.0, 15.0)
        homs[i] = compose(noisy)
    sm, stats = smooth_homographies(homs, n)
    assert stats["predicted"] == 20, stats
    grid = CHECK_GRID
    raw_err = np.median([np.median(np.linalg.norm(apply_homography(homs[i], grid)
                                                  - apply_homography(clean[i], grid), axis=1))
                         for i in homs])
    sm_err = np.median([np.median(np.linalg.norm(apply_homography(sm[i], grid)
                                                 - apply_homography(clean[i], grid), axis=1))
                        for i in homs])
    assert sm_err < 0.6 * raw_err, (raw_err, sm_err)
    gap_err = np.median([grid_shift_m(sm[i], clean[i]) for i in range(100, 120)])
    assert gap_err < raw_err, (gap_err, raw_err)

    sm2, stats2 = smooth_homographies(homs, n, mode="mapping")
    assert stats2["parameterised"] == len(homs), stats2
    map_err = np.median([np.median(np.linalg.norm(apply_homography(sm2[i], grid)
                                                  - apply_homography(clean[i], grid), axis=1))
                         for i in homs])
    assert map_err < 0.6 * raw_err, (raw_err, map_err)
    print(f"camera_track demo OK (raw {raw_err:.3f} m -> camera-space {sm_err:.3f} m / "
          f"mapping-space {map_err:.3f} m, gap prediction {gap_err:.3f} m)")


if __name__ == "__main__":
    _demo()
