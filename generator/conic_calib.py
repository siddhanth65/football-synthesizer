"""Conic constructions on the three pitch circles, applied at INFERENCE.

PnLCalib's keypoint vocabulary is Falaleev & Chen's (arXiv:2410.07401 sec. 3.1) item for item:
``utils/utils_calib.py::keypoint_world_coords_2D`` holds 30 line-line intersections (ids 1-30),
6 line-conic intersections (31-36), 8 conic tangent points (37-44) and 13 further structural points
(45-57). Their label generator (``utils/utils_keypoints.py``) runs the conic constructions at
TRAINING time and trains one HRNet channel per derived point.

What PnLCalib does **not** do is use any conic relation at INFERENCE: all 57 keypoints come out of
independent heatmap channels, and nothing enforces that 21 of them lie on three circles of known
radius whose centres it separately predicts. This module supplies exactly that missing layer:

* :func:`fit_conic` -- direct least-squares ellipse fit (Halir-Flusser form of Fitzgibbon).
* :func:`project_to_conic` -- pull a detected keypoint onto the fitted conic (denoising; Falaleev's
  "use computed keypoints rather than the raw annotation", moved from training to inference).
* :func:`conic_line_points` / :func:`polar` -- the two primitives every construction below is built
  from (a line meets a conic in two points; the polar of a pole is a line).
* :func:`derive_points` -- line-conic intersections, tangent points from a detected external
  keypoint, and the Magera et al. (arXiv:2504.20052 sec. 3.1) vanishing-line construction that turns
  one conic + its centre + one line into the circle's axis extremes and quarter turns.

Everything here is pure geometry on our own detections: no ground truth, no learned parameter, no
tuning dial. Coordinates are plain image pixels in whatever space the caller works in (conics are
projective, so any affine rescale of the input rescales the output consistently).
"""

from __future__ import annotations

import numpy as np

#: Radius of every circular marking on a full-size pitch (centre circle and both penalty arcs).
CIRCLE_RADIUS_M = 9.15
#: ``name -> (centre_xy_m, PnLCalib keypoint ids lying on the circle, the centre's own keypoint id)``
#: on the uncentred 105x68 pitch. Ids are 1-based, matching ``keypoint_world_coords_2D``.
CIRCLES: dict[str, tuple[tuple[float, float], tuple[int, ...], int]] = {
    "left": ((11.0, 34.0), (31, 34, 37, 41, 47), 45),
    "centre": ((52.5, 34.0), (32, 35, 38, 39, 42, 43, 48, 49, 50, 53, 54), 51),
    "right": ((94.0, 34.0), (33, 36, 40, 44, 55), 57),
}
#: ``circle -> (PnLCalib line id, the two keypoint ids that line cuts the circle at)``. The line ids
#: index ``utils_heatmap.complete_keypoints``'s ``lines_list`` (1-based): 13 = "Middle line",
#: 2 = "Big rect. left main", 5 = "Big rect. right main".
LINE_CUTS: dict[str, tuple[int, tuple[int, int]]] = {
    "left": (2, (31, 34)), "centre": (13, (32, 35)), "right": (5, (33, 36)),
}
#: ``external keypoint id -> (circle, the tangent-point ids reachable from it)``. Derived exactly
#: from ``keypoint_world_coords_2D`` (each listed tangency is within 0.03 m of the analytic tangent
#: point of that external point); the second tangent point of ids 5/25/6/26 is not in the vocabulary.
TANGENT_FROM: dict[int, tuple[str, tuple[int, ...]]] = {
    5: ("left", (37,)), 25: ("left", (41,)),
    2: ("centre", (38, 39)), 29: ("centre", (42, 43)),
    6: ("right", (40,)), 26: ("right", (44,)),
}
#: Circle keypoints reachable by the Magera construction, per circle and per construction group:
#: ``par`` = the diameter parallel to the reference line, ``perp`` = the perpendicular diameter,
#: ``quarter`` = the four 45-degree points off the tangent trapezoid's diagonals. Groups are matched
#: to ids separately so one group can never consume another's point. Ends of a diameter that have no
#: keypoint id are simply absent -- the assignment picks the nearer candidate for the end that does.
#: Keypoint 52 is deliberately excluded: PnLCalib's table puts it at ``[61.5, 34]``, 0.15 m inside
#: the true circle point ``[61.65, 34]``, so a geometrically derived point filed under that id would
#: inject a known systematic error.
MAGERA_AXES: dict[str, dict[str, tuple[int, ...]]] = {
    "left": {"par": (), "perp": (47,), "quarter": ()},
    "centre": {"par": (32, 35), "perp": (50,), "quarter": (48, 49, 53, 54)},
    "right": {"par": (), "perp": (55,), "quarter": ()},
}
#: Pitch lines that run along the world **y** axis (constant x), so the diameter parallel to them is
#: always the circle's y-diameter. 2/5 = big rect. left/right main, 13 = middle line,
#: 15/16 = side line left/right. Restricting the reference line to this family is what makes the
#: ``par``/``perp`` group labels above unambiguous.
Y_AXIS_LINES: tuple[int, ...] = (2, 5, 13, 15, 16)
#: A derived point this far outside the image (in the caller's pixel units) is discarded.
FRAME_MARGIN_PX = 200.0
#: Fewest detected on-circle keypoints that may be fitted with a conic (5 = a conic's 5 d.o.f.).
MIN_CONIC_POINTS = 5


def fit_conic(pts: np.ndarray) -> np.ndarray | None:
    """Direct least-squares ellipse fit, returned as a symmetric ``(3, 3)`` conic matrix.

    Halir & Flusser's numerically stable rearrangement of Fitzgibbon's method: minimise the algebraic
    distance ``|D a|^2`` subject to the ellipse constraint ``4ac - b^2 = 1``, split into quadratic and
    linear blocks so the generalised eigenproblem stays well conditioned.

    Args:
        pts: ``(N, 2)`` image points, ``N >= 5``.

    Returns:
        The conic ``C`` with ``x^T C x = 0`` for ``x = (x, y, 1)``, or ``None`` if the points are too
        few, degenerate, or fit a hyperbola/parabola rather than a real ellipse.
    """
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < MIN_CONIC_POINTS:
        return None
    # Work centred and unit-scaled: the fit is exactly equivariant under this similarity and the
    # eigenproblem is otherwise ill conditioned at 1e3-pixel coordinates.
    mu = p.mean(axis=0)
    scale = float(np.sqrt((((p - mu) ** 2).sum(axis=1)).mean()))
    if not np.isfinite(scale) or scale < 1e-9:
        return None
    q = (p - mu) / scale
    x, y = q[:, 0], q[:, 1]
    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones(len(q))])
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t
    c_inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    try:
        evals, evecs = np.linalg.eig(c_inv @ m)
    except np.linalg.LinAlgError:
        return None
    cond = 4.0 * evecs[0] * evecs[2] - evecs[1] ** 2
    ok = np.flatnonzero((cond > 0) & np.isfinite(evals))
    if not len(ok):
        return None
    a1 = np.real(evecs[:, ok[0]])
    a, b, c = a1
    d, e, f = np.real(t @ a1)
    # Undo the similarity: x_norm = (x - mu) / scale.
    sx, sy = mu
    conic_n = np.array([[a, b / 2.0, d / 2.0], [b / 2.0, c, e / 2.0], [d / 2.0, e / 2.0, f]])
    t_inv = np.array([[1.0 / scale, 0.0, -sx / scale], [0.0, 1.0 / scale, -sy / scale],
                      [0.0, 0.0, 1.0]])
    conic = t_inv.T @ conic_n @ t_inv
    if not np.isfinite(conic).all():
        return None
    # Real ellipse test on the upper-left 2x2 block plus a positive-area check.
    a2 = conic[:2, :2]
    if np.linalg.det(a2) <= 0 or abs(np.linalg.det(conic)) < 1e-30:
        return None
    return conic / np.abs(conic).max()


def _unit(v: np.ndarray) -> np.ndarray:
    """Scale a homogeneous 3-vector to unit norm (pure).

    Every construction here is a chain of cross products and conic products, each of which multiplies
    the coefficient magnitude. Left alone, a diagonal of the tangent trapezoid arrives with
    coefficients around 1e20, and the discriminant in :func:`conic_line_points` is then the
    difference of two 1e80 quantities -- catastrophic cancellation that silently reports a genuine
    secant as missing the conic. Normalising after every step costs nothing and removes it.
    """
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=float) / n if n > 1e-300 else np.asarray(v, dtype=float)


def polar(conic: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """The polar line ``l = C p`` of a pole ``p`` with respect to ``conic`` (homogeneous, pure).

    For a point ON the conic this is the tangent there; for a point outside it is the chord joining
    the two points of tangency, which is how :func:`derive_points` gets tangent points without ever
    parameterising the ellipse.
    """
    p = np.asarray(pole, dtype=float).reshape(-1)
    if p.size == 2:
        p = np.array([p[0], p[1], 1.0])
    return _unit(np.asarray(conic, dtype=float) @ _unit(p))


def conic_line_points(conic: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two real intersections of a homogeneous ``line`` with ``conic``, or ``None``.

    Two points ``p``, ``q`` spanning the line are taken from cross products with the two most
    orthogonal axis lines; substituting ``p + t q`` into ``x^T C x = 0`` gives a scalar quadratic.

    Returns:
        ``(p1, p2)`` as inhomogeneous ``(2,)`` arrays, or ``None`` when the line misses the conic,
        is tangent to it, or the intersections lie at infinity.
    """
    line = _unit(np.asarray(line, dtype=float).reshape(3))
    if not np.isfinite(line).all() or np.abs(line[:2]).max() < 1e-12:
        return None
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    span = [np.cross(line, ax) for ax in axes]
    span.sort(key=lambda v: -float(np.abs(v).max()))
    p, q = span[0], span[1]
    if float(np.abs(np.cross(p, q)).max()) < 1e-12:
        return None
    a = float(q @ conic @ q)
    b = 2.0 * float(p @ conic @ q)
    c = float(p @ conic @ p)
    if abs(a) < 1e-18:
        return None
    disc = b * b - 4.0 * a * c
    if disc <= 0:
        return None
    roots = ((-b + np.sqrt(disc)) / (2.0 * a), (-b - np.sqrt(disc)) / (2.0 * a))
    out = []
    for t in roots:
        h = p + t * q
        if abs(h[2]) < 1e-12:
            return None
        out.append(h[:2] / h[2])
    return out[0], out[1]


def project_to_conic(conic: np.ndarray, pts: np.ndarray, iters: int = 8) -> np.ndarray:
    """Pull points onto the conic along the local gradient (Newton on ``x^T C x = 0``).

    This is the denoising step: detection noise perpendicular to the conic is removed, noise along it
    is left alone (nothing here can know where along the circle the true point was).

    Args:
        conic: ``(3, 3)`` conic matrix.
        pts: ``(N, 2)`` points, expected already near the conic.
        iters: Newton steps.

    Returns:
        ``(N, 2)``; a point whose gradient vanishes (the conic's centre) is returned unchanged.
    """
    # ponytail: Newton onto the zero level set, not the exact quartic nearest-point solution. For a
    # point already within ~30 px of the conic the two agree to well under 0.01 px, which covers the
    # whole detection-noise regime; swap in the quartic only if a caller starts far from the curve.
    p = np.asarray(pts, dtype=float).reshape(-1, 2).copy()
    for _ in range(iters):
        h = np.column_stack([p, np.ones(len(p))])
        val = np.einsum("ni,ij,nj->n", h, conic, h)
        grad = 2.0 * (h @ conic.T)[:, :2]
        g2 = (grad ** 2).sum(axis=1)
        step = np.where(g2 > 1e-18, val / np.where(g2 > 1e-18, g2, 1.0), 0.0)
        p -= step[:, None] * grad
    return p


def _pick(cands: tuple[np.ndarray, ...] | list[np.ndarray],
          targets: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Assign candidate image points to keypoint ids by minimum total distance (pure).

    ``targets`` holds where the frame's EXISTING homography puts each id, so the discrete left/right
    and near/far ambiguity that every conic construction carries (Magera et al. resolve it "based on
    priors on the camera position") is settled by our own on-record estimate. Only the ASSIGNMENT
    uses it; the point's position is pure conic geometry.

    A global (rectangular Hungarian) assignment rather than greedy nearest, because greedy lets an
    early id steal the point a later id needed. Callers must pass one construction group at a time --
    ids from different groups are not competing for the same points.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    ids = sorted(targets)
    pts = [np.asarray(c, dtype=float) for c in cands]
    if not ids or not pts:
        return {}
    cost = np.array([[float(np.linalg.norm(p - targets[k])) for p in pts] for k in ids])
    if not np.isfinite(cost).all():
        return {}
    rows, cols = linear_sum_assignment(cost)
    return {ids[r]: pts[c] for r, c in zip(rows, cols, strict=True)}


def derive_points(kp_xy: dict[int, np.ndarray], lines: dict[int, np.ndarray],
                  predicted: dict[int, np.ndarray], size: tuple[float, float],
                  ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Denoise the on-circle keypoints and derive new ones from the fitted conics.

    Args:
        kp_xy: ``{keypoint id: (x, y)}`` detected keypoints, image pixels.
        lines: ``{line id: (a, b, c)}`` detected pitch lines as homogeneous image lines.
        predicted: ``{keypoint id: (x, y)}`` where the frame's existing homography puts each model
            keypoint -- used ONLY to resolve discrete correspondence ambiguity (see :func:`_pick`).
        size: ``(width, height)`` of the image the coordinates live in.

    Returns:
        ``(denoised, derived, stats)``. ``denoised`` holds replacements for keypoints already in
        ``kp_xy``; ``derived`` holds keypoints that were absent. Both are ``{id: (x, y)}``.
    """
    wid, hgt = size
    denoised: dict[int, np.ndarray] = {}
    derived: dict[int, np.ndarray] = {}
    stats: dict = {"conics": 0, "denoised": 0, "line_conic": 0, "tangent": 0, "magera": 0}

    def keep(p: np.ndarray) -> bool:
        return bool(np.isfinite(p).all()
                    and -FRAME_MARGIN_PX <= p[0] <= wid + FRAME_MARGIN_PX
                    and -FRAME_MARGIN_PX <= p[1] <= hgt + FRAME_MARGIN_PX)

    def emit(found: dict[int, np.ndarray], bucket: str) -> None:
        for kid, p in found.items():
            if kid in kp_xy or kid in derived or not keep(p):
                continue
            derived[kid] = p
            stats[bucket] += 1

    for name, (_centre_m, ids, centre_id) in CIRCLES.items():
        have = [k for k in ids if k in kp_xy]
        if len(have) < MIN_CONIC_POINTS:
            continue
        conic = fit_conic(np.array([kp_xy[k] for k in have]))
        if conic is None:
            continue
        stats["conics"] += 1
        for kid, p in zip(have, project_to_conic(conic, np.array([kp_xy[k] for k in have])),
                          strict=True):
            if keep(p):
                denoised[kid] = p
                stats["denoised"] += 1

        # C3: the conic cut by its own detected pitch line.
        line_id, cut_ids = LINE_CUTS[name]
        if line_id in lines:
            hit = conic_line_points(conic, lines[line_id])
            if hit is not None:
                emit(_pick(hit, {k: predicted[k] for k in cut_ids if k in predicted}), "line_conic")

        # C4: tangency from a detected external keypoint (projectively invariant).
        for ext_id, (circ, tang_ids) in TANGENT_FROM.items():
            if circ != name or ext_id not in kp_xy:
                continue
            hit = conic_line_points(conic, polar(conic, kp_xy[ext_id]))
            if hit is not None:
                emit(_pick(hit, {k: predicted[k] for k in tang_ids if k in predicted}), "tangent")

        # C5: Magera's vanishing-line construction -- conic + its own centre + any one pitch line
        # of known world direction. Groups are matched to ids independently (see :func:`_pick`).
        axes = MAGERA_AXES[name]
        line1 = next((lines[i] for i in sorted(lines) if i in Y_AXIS_LINES), None)
        if centre_id not in kp_xy or line1 is None:
            continue
        c = _unit(np.array([*kp_xy[centre_id], 1.0]))
        v = _unit(np.cross(_unit(line1), _unit(conic @ c)))   # vanishing point of line1's direction
        par = conic_line_points(conic, np.cross(v, c))        # the diameter parallel to line1
        perp = conic_line_points(conic, conic @ v)            # its perpendicular diameter
        if par is None or perp is None:
            continue
        for group, hit in (("par", par), ("perp", perp)):
            emit(_pick(hit, {k: predicted[k] for k in axes[group] if k in predicted}), "magera")
        want_q = {k: predicted[k] for k in axes["quarter"] if k in predicted}
        if not want_q:
            continue
        tangents = [polar(conic, p) for p in (*par, *perp)]
        corners = [_unit(np.cross(tangents[i], tangents[j])) for i in (0, 1) for j in (2, 3)]
        quarters: list[np.ndarray] = []
        for i, j in ((0, 3), (1, 2)):  # opposite corners of the tangent trapezoid
            hit = conic_line_points(conic, np.cross(corners[i], corners[j]))
            if hit is not None:
                quarters.extend(hit)
        emit(_pick(quarters, want_q), "magera")
    return denoised, derived, stats


def _demo() -> None:
    """Assert every construction against a synthetic camera whose answer is known exactly."""
    # A plausible broadcast mapping: pitch metres -> image pixels.
    h_world_to_img = np.array([[14.0, 1.1, 180.0], [1.6, 6.4, 60.0], [0.0009, 0.0075, 1.0]])

    def to_img(pts: np.ndarray) -> np.ndarray:
        q = np.column_stack([np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))])
        r = q @ h_world_to_img.T
        return r[:, :2] / r[:, 2:3]

    world = {
        31: (16.5, 26.68), 34: (16.5, 41.31), 37: (19.99, 32.29), 41: (19.99, 35.7),
        47: (20.15, 34.0), 45: (11.0, 34.0),
        32: (52.5, 24.85), 35: (52.5, 43.15), 38: (43.68, 31.53), 39: (61.31, 31.53),
        42: (43.68, 36.46), 43: (61.31, 36.46), 48: (46.03, 27.53), 49: (58.97, 27.53),
        50: (43.35, 34.0), 53: (46.03, 40.47), 54: (58.97, 40.47), 51: (52.5, 34.0),
        2: (52.5, 0.0), 29: (52.5, 68.0), 5: (16.5, 13.84), 25: (16.5, 54.16),
    }
    img = {k: to_img(np.array([v]))[0] for k, v in world.items()}

    # 1. the fit recovers the true conic. Measured on EXACT circle points, because PnLCalib's own
    # table is rounded to 0.01 m (id 38 sits at radius 9.16, not 9.15) and would floor the test at
    # ~0.15 px. Residual is geometric: the distance the Newton projection has to move a point.
    exact = to_img(np.column_stack([52.5 + CIRCLE_RADIUS_M * np.cos(np.linspace(0, 6.0, 11)),
                                    34.0 + CIRCLE_RADIUS_M * np.sin(np.linspace(0, 6.0, 11))]))
    conic_exact = fit_conic(exact)
    assert conic_exact is not None
    assert float(np.abs(project_to_conic(conic_exact, exact) - exact).max()) < 1e-6

    on_centre = [32, 35, 38, 39, 42, 43, 48, 49, 50, 53, 54]
    conic = fit_conic(np.array([img[k] for k in on_centre]))
    assert conic is not None
    pts = np.array([img[k] for k in on_centre])
    assert float(np.linalg.norm(project_to_conic(conic, pts) - pts, axis=1).max()) < 0.3

    # 2. denoising: perturb the detections by 3 px and the fit must pull them back most of the way.
    rng = np.random.default_rng(20260817)
    noisy = {k: img[k] + rng.normal(0, 3.0, 2) for k in on_centre}
    cn = fit_conic(np.array([noisy[k] for k in on_centre]))
    assert cn is not None
    fixed = project_to_conic(cn, np.array([noisy[k] for k in on_centre]))
    before = float(np.mean([np.linalg.norm(noisy[k] - img[k]) for k in on_centre]))
    after = float(np.mean([np.linalg.norm(f - img[k]) for f, k in zip(fixed, on_centre,
                                                                      strict=True)]))
    assert after < before, (before, after)

    # 3. the three derivations, from a keypoint set with every derivable id REMOVED.
    mid = np.cross(np.array([*to_img(np.array([[52.5, 0.0]]))[0], 1.0]),
                   np.array([*to_img(np.array([[52.5, 68.0]]))[0], 1.0]))
    seed_ids = [38, 39, 42, 43, 48, 53, 2, 29, 51]  # 5 on-circle points + externals + centre
    kp = {k: img[k] for k in seed_ids}
    pred = {k: img[k] for k in world}  # a perfect "existing homography" for the id assignment only
    den, der, st = derive_points(kp, {13: mid}, pred, (960.0, 540.0))
    for kid in (32, 35, 50, 49, 54):
        assert kid in der, (kid, sorted(der), st)
        assert np.linalg.norm(der[kid] - img[kid]) < 0.5, (kid, der[kid], img[kid])
    assert st["conics"] == 1 and st["line_conic"] == 2 and len(den) == 6, st

    # 4. tangency: drop the tangent points and rebuild them from the external keypoints.
    kp2 = {k: img[k] for k in (32, 35, 48, 49, 53, 54, 2, 29, 51)}
    _d2, der2, st2 = derive_points(kp2, {13: mid}, pred, (960.0, 540.0))
    for kid in (38, 39, 42, 43):
        assert kid in der2 and np.linalg.norm(der2[kid] - img[kid]) < 0.5, (kid, st2)

    # 5. a degenerate input must be refused, not guessed at.
    assert fit_conic(np.column_stack([np.arange(8.0), np.arange(8.0)])) is None
    assert fit_conic(np.zeros((3, 2))) is None
    assert conic_line_points(conic, np.array([0.0, 0.0, 1.0])) is None
    print(f"conic_calib demo OK (denoise {before:.3f} -> {after:.3f} px, derived {sorted(der)})")


if __name__ == "__main__":
    _demo()
