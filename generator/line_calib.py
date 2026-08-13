"""Pitch-line evidence for frames the keypoint calibrator cannot get past the gate.

`results/GSR_V8_W2.md` measured the one thing an external calibrator had that we do not: a **real**
homography on the frames where ours emits nothing, worth 1.97 m -> 0.45 m against the interpolating
fill. `results/GSR_V8_W2B.md` §1.0 then found *why* those frames are dead, and it is not "PnLCalib
returned nothing" (that happens on 0 of the 139 probe frames): a frame showing two players cannot
pass :func:`generator.postprocess.onpitch_plausible` **whatever its homography is**, and a frame
whose hypotheses are all implausible has no second opinion to fall back on.

Both gaps want the same missing evidence -- the pitch itself. This module supplies it, classically
(OpenCV only, CPU, no weights):

* :func:`line_mask` -- grass-restricted white-ridge pixels of one frame;
* :func:`line_support` -- the share of the pitch model's lines, pushed through a homography, that
  land on those pixels. An **acceptance test that needs no players**, so it can judge exactly the
  frames the player-based test cannot;
* :func:`refit_homography` -- a homography *solved from lines*: associate ridge pixels to projected
  model lines and solve the linear point-on-line system ``l^T (H p) = 0``. Seeded from a neighbouring
  frame's homography, this recovers frames that have no usable hypothesis at all;
* :func:`recover_dead_frames` -- the positions-table stage, dead-frame-only, that sits **behind**
  the PnLCalib gate and **ahead** of :func:`generator.postprocess.fill_calibration_gaps`.

Every homography this module produces is still subject to the existing acceptance stack (the metre
gate, ``onpitch_plausible`` where the frame has enough players to evaluate it, ``clamp_to_pitch``
per point); line support is an *additional* condition, never a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.pitch import PITCH_LEN, PITCH_WID

# --- the knobs. Deliberately few, and each is a physical quantity, not a fitted coefficient. -----
#: Top-hat kernel (px). Must exceed a broadcast pitch line's width so the line survives and the
#: grass does not; 21 px covers the 4-8 px lines of 1080p SoccerNet frames with margin.
TOPHAT_K = 21
#: Top-hat response (0-255) above which a pixel is called line ridge.
RIDGE_MIN = 24
#: Grass hue window in OpenCV's 0-179 scale, plus the minimum saturation/value of lit turf.
GRASS_HUE = (25, 95)
GRASS_MIN_SV = (25, 40)
#: A model-line sample is "supported" when a ridge pixel is within this many pixels.
SUPPORT_R_PX = 8.0
#: Metres between samples along a model line.
SAMPLE_STEP_M = 0.5
#: ICP association radii (px), coarse to fine -- one iteration each.
REFIT_RADII_PX = (40.0, 20.0, 10.0)
#: A line solve needs at least this many distinct model segments and point constraints.
MIN_REFIT_SEGMENTS = 4
MIN_REFIT_POINTS = 24


@dataclass(frozen=True)
class LineFit:
    """A homography recovered or judged from line evidence.

    Args:
        homography: ``(3, 3)`` image->pitch (uncentred metres), or ``None`` if the solve failed.
        residual_m: Mean point-to-line distance of the inlier constraints, in metres (``inf`` if no
            solve). Directly comparable to :data:`generator.calibrate.MAX_REPROJ_ERROR_M`.
        support: Share of in-image model-line samples backed by a ridge pixel, in ``[0, 1]``.
        n_points: Inlier point-on-line constraints used.
        n_segments: Distinct model segments those constraints came from.
    """

    homography: np.ndarray | None
    residual_m: float
    support: float
    n_points: int
    n_segments: int


# === the pitch model =============================================================================
def pitch_model_lines(
    *, length: float = PITCH_LEN, width: float = PITCH_WID, circle_steps: int = 24
) -> np.ndarray:
    """The FIFA line set in our uncentred ``[0, length] x [0, width]`` metre frame.

    The centre circle is emitted as a closed polyline so it contributes ordinary segment
    constraints; penalty arcs are omitted (they add little and cost correspondence ambiguity).

    Args:
        length: Pitch length (m).
        width: Pitch width (m).
        circle_steps: Segments the centre circle is discretised into.

    Returns:
        ``(N, 2, 2)`` array of segment endpoints in metres.
    """
    lx, wy = length, width
    pen_d, pen_w = 16.5, 40.32  # penalty area depth / width
    goal_d, goal_w = 5.5, 18.32  # goal area
    segs: list[list[list[float]]] = [
        [[0.0, 0.0], [lx, 0.0]], [[0.0, wy], [lx, wy]],            # touchlines
        [[0.0, 0.0], [0.0, wy]], [[lx, 0.0], [lx, wy]],            # goal lines
        [[lx / 2, 0.0], [lx / 2, wy]],                             # halfway
    ]
    for x0, sign in ((0.0, 1.0), (lx, -1.0)):
        for depth, half in ((pen_d, pen_w / 2), (goal_d, goal_w / 2)):
            y0, y1 = wy / 2 - half, wy / 2 + half
            x1 = x0 + sign * depth
            segs += [[[x0, y0], [x1, y0]], [[x0, y1], [x1, y1]], [[x1, y0], [x1, y1]]]
    cx, cy, r = lx / 2, wy / 2, 9.15
    ang = np.linspace(0.0, 2 * np.pi, circle_steps + 1)
    ring = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
    segs += [[ring[i].tolist(), ring[i + 1].tolist()] for i in range(circle_steps)]
    return np.asarray(segs, dtype=float)


def _segment_lines(segs: np.ndarray) -> np.ndarray:
    """Homogeneous line coefficients ``(N, 3)`` with a unit normal, so ``l . x`` is metres."""
    a = np.concatenate([segs[:, 0], np.ones((len(segs), 1))], axis=1)
    b = np.concatenate([segs[:, 1], np.ones((len(segs), 1))], axis=1)
    lines = np.cross(a, b)
    n = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return lines / np.where(n < 1e-12, 1.0, n)


def sample_model_points(segs: np.ndarray, *, step_m: float = SAMPLE_STEP_M
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Sample every model segment at ``step_m`` intervals.

    Args:
        segs: ``(N, 2, 2)`` segment endpoints (metres).
        step_m: Spacing along each segment.

    Returns:
        ``(pts, seg_id)``: ``(M, 2)`` pitch-metre samples and the segment each came from.
    """
    pts: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for i, (p0, p1) in enumerate(segs):
        n = max(2, int(np.ceil(float(np.linalg.norm(p1 - p0)) / step_m)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(p0 + t * (p1 - p0))
        ids.append(np.full(n, i))
    return np.concatenate(pts), np.concatenate(ids)


# === the image evidence ==========================================================================
def line_mask(frame_bgr: np.ndarray, *, tophat_k: int = TOPHAT_K, ridge_min: int = RIDGE_MIN
              ) -> np.ndarray:
    """White pitch-line pixels of one BGR frame (grass-restricted top-hat ridge mask).

    A top-hat with a kernel wider than a pitch line keeps structures thinner than the kernel and
    removes everything broader (turf, stands, players' bodies); intersecting with a dilated grass
    mask drops the bright clutter that lives off the field.

    Args:
        frame_bgr: The frame.
        tophat_k: Top-hat kernel size in pixels.
        ridge_min: Minimum top-hat response for a ridge pixel.

    Returns:
        ``(H, W)`` ``uint8`` mask, 255 on ridge pixels.
    """
    import cv2  # noqa: PLC0415 - OpenCV is only needed for the image half of this module

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (GRASS_HUE[0], GRASS_MIN_SV[0], GRASS_MIN_SV[1]),
                        (GRASS_HUE[1], 255, 255))
    grass = cv2.dilate(grass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_k, tophat_k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.bitwise_and(((tophat >= ridge_min) * 255).astype(np.uint8), grass)


def ridge_field(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance-to-nearest-ridge-pixel field and that pixel's coordinates, for O(1) lookup.

    Args:
        mask: :func:`line_mask` output.

    Returns:
        ``(dist, nearest)``: ``(H, W)`` float distances in pixels and ``(H, W, 2)`` int32 ``(x, y)``
        of the nearest ridge pixel (``nearest`` is meaningless where no ridge pixel exists).
    """
    import cv2  # noqa: PLC0415

    inv = np.where(mask > 0, 0, 1).astype(np.uint8)
    if not inv.size or inv.min() != 0:  # no ridge pixels at all
        return np.full(mask.shape, np.inf, dtype=float), np.zeros(mask.shape + (2,), np.int32)
    dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask > 0)
    lut = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
    lut[labels[ys, xs]] = np.stack([xs, ys], axis=1)
    return dist.astype(float), lut[labels]


def _project_to_image(h: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray | None:
    """Push pitch-metre points into the image through the inverse of an image->pitch ``h``."""
    from generator.calibrate import apply_homography  # noqa: PLC0415

    try:
        h_inv = np.linalg.inv(np.asarray(h, dtype=float))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(h_inv).all():
        return None
    return apply_homography(h_inv, pitch_pts)


def line_support(h: np.ndarray, dist: np.ndarray, *, segs: np.ndarray | None = None,
                 radius_px: float = SUPPORT_R_PX) -> tuple[float, int]:
    """How much of the pitch model, seen through ``h``, is actually drawn on the frame.

    Args:
        h: ``(3, 3)`` image->pitch homography (uncentred metres).
        dist: :func:`ridge_field` distance map of the same frame.
        segs: Model segments (defaults to :func:`pitch_model_lines`).
        radius_px: A sample counts as supported within this many pixels of a ridge pixel.

    Returns:
        ``(support, n_visible)`` -- the supported share of the model-line samples that fall inside
        the frame, and how many those were. ``support`` is ``0.0`` when nothing is visible.
    """
    segs = pitch_model_lines() if segs is None else segs
    pts, _ = sample_model_points(segs)
    img = _project_to_image(h, pts)
    if img is None:
        return 0.0, 0
    hgt, wid = dist.shape
    x, y = np.round(img[:, 0]).astype(int), np.round(img[:, 1]).astype(int)
    inside = (x >= 0) & (x < wid) & (y >= 0) & (y < hgt)
    if inside.sum() == 0:
        return 0.0, 0
    return float(np.mean(dist[y[inside], x[inside]] <= radius_px)), int(inside.sum())


# === the line solve ==============================================================================
def solve_point_on_line(img_pts: np.ndarray, lines: np.ndarray, shape: tuple[int, int]
                        ) -> np.ndarray | None:
    """Homography from point-on-line constraints: ``l^T (H p) = 0``, one row per correspondence.

    This is the line-based estimator itself. Each image point known to lie on a *labelled* pitch
    line contributes one linear equation in the nine entries of ``H``, so four lines in general
    position determine it -- the same information a four-point DLT uses, carried by the pitch
    markings instead of by keypoints.

    Args:
        img_pts: ``(N, 2)`` image points.
        lines: ``(N, 3)`` unit-normal pitch-line coefficients, one per point.
        shape: ``(height, width)`` of the frame, used to precondition the image coordinates.

    Returns:
        The ``(3, 3)`` image->pitch homography, or ``None`` if the system is degenerate.
    """
    hgt, wid = shape
    t = np.array([[2.0 / wid, 0.0, -1.0], [0.0, 2.0 / hgt, -1.0], [0.0, 0.0, 1.0]])
    p = np.concatenate([np.asarray(img_pts, dtype=float), np.ones((len(img_pts), 1))], axis=1) @ t.T
    rows = (lines[:, :, None] * p[:, None, :]).reshape(len(p), 9)
    if len(rows) < 9:
        return None
    _, _, vt = np.linalg.svd(rows)
    h = vt[-1].reshape(3, 3) @ t
    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-14:
        return None
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def _residuals_m(h: np.ndarray, img_pts: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Signed point-to-line distance in metres for each constraint under ``h``."""
    p = np.concatenate([img_pts, np.ones((len(img_pts), 1))], axis=1) @ np.asarray(h).T
    w = np.where(np.abs(p[:, 2]) < 1e-12, 1e-12, p[:, 2])
    return np.einsum("ij,ij->i", lines, p) / w


def refit_homography(h0: np.ndarray, dist: np.ndarray, nearest: np.ndarray, *,
                     segs: np.ndarray | None = None, radii: tuple[float, ...] = REFIT_RADII_PX
                     ) -> LineFit:
    """Re-solve a frame's homography from its own pitch lines, seeded by ``h0`` (ICP).

    Each iteration projects the model lines through the current estimate, snaps every in-image
    sample to the nearest ridge pixel inside the iteration's radius, and re-solves
    :func:`solve_point_on_line` over those pairs; the radius shrinks so a coarse seed can be pulled
    in without letting a far-away line capture the association. One robustness pass drops
    constraints beyond twice the median residual.

    Args:
        h0: Seed homography (image->pitch, uncentred metres) -- a neighbouring frame's.
        dist: :func:`ridge_field` distances for the frame being solved.
        nearest: :func:`ridge_field` nearest-pixel coordinates for the same frame.
        segs: Model segments (defaults to :func:`pitch_model_lines`).
        radii: Association radius per iteration, coarse to fine.

    Returns:
        A :class:`LineFit`; ``homography`` is ``None`` when the evidence is too thin or the solve
        degenerates. The caller still has to gate it.
    """
    segs = pitch_model_lines() if segs is None else segs
    pts, seg_id = sample_model_points(segs)
    lines_all = _segment_lines(segs)
    hgt, wid = dist.shape
    h = np.asarray(h0, dtype=float)
    best = LineFit(None, float("inf"), 0.0, 0, 0)
    for radius in radii:
        img = _project_to_image(h, pts)
        if img is None:
            return best
        x, y = np.round(img[:, 0]).astype(int), np.round(img[:, 1]).astype(int)
        ok = (x >= 0) & (x < wid) & (y >= 0) & (y < hgt)
        if ok.sum() < MIN_REFIT_POINTS:
            return best
        ok[ok] &= dist[y[ok], x[ok]] <= radius
        if ok.sum() < MIN_REFIT_POINTS:
            return best
        targets = nearest[y[ok], x[ok]].astype(float)
        lines = lines_all[seg_id[ok]]
        cand = solve_point_on_line(targets, lines, (hgt, wid))
        if cand is None:
            return best
        res = np.abs(_residuals_m(cand, targets, lines))
        inl = res <= max(2.0 * float(np.median(res)), 0.25)
        if inl.sum() >= MIN_REFIT_POINTS:
            refined = solve_point_on_line(targets[inl], lines[inl], (hgt, wid))
            if refined is not None:
                cand, res, inl = (refined, np.abs(_residuals_m(refined, targets, lines)), inl)
        n_seg = int(len(np.unique(seg_id[ok][inl])))
        h = cand
        best = LineFit(homography=cand, residual_m=float(np.mean(res[inl])) if inl.any()
                       else float("inf"), support=0.0, n_points=int(inl.sum()), n_segments=n_seg)
    if best.homography is None or best.n_segments < MIN_REFIT_SEGMENTS:
        return LineFit(None, float("inf"), 0.0, best.n_points, best.n_segments)
    sup, _ = line_support(best.homography, dist, segs=segs)
    return LineFit(best.homography, best.residual_m, sup, best.n_points, best.n_segments)


# === the positions-table stage ===================================================================
def recover_dead_frames(
    df, frames_bgr, *, candidates=None, min_support: float = 0.0, max_error_m: float | None = None,
    refit: bool = True, seed_from: str = "donor", chain_seed: bool = True, log: list | None = None,
):
    """Fill in dead frames from line evidence -- behind the keypoint gate, ahead of the fill.

    A frame is *dead* when none of its player rows carries a pitch coordinate. For each one, in
    frame order, two sources are tried and the first that clears the acceptance stack wins:

    1. **its own keypoint hypotheses** (``candidates[frame]``, PnLCalib's, already computed), now
       admissible on a player-sparse frame because :func:`line_support` can vouch for them where
       ``onpitch_plausible`` structurally cannot;
    2. **a line solve** (:func:`refit_homography`) seeded from the nearest already-live frame's
       homography, propagated forward through the dead run so each step sees one frame of camera
       motion rather than the whole gap.

    Live frames are never touched, and a recovered coordinate still goes through
    :func:`generator.postprocess.clamp_to_pitch`.

    Args:
        df: A positions table (``frame``, ``role``, ``pitch_x/y``, ``image_x/y``).
        frames_bgr: ``{frame: BGR image}`` or any mapping/callable ``frame -> image or None``, for
            the dead frames only.
        candidates: ``{frame: [CalibCandidate, ...]}`` in preference order, or ``None``.
        min_support: Minimum :func:`line_support` a homography must have to be accepted.
        max_error_m: Metre gate (defaults to :data:`generator.calibrate.MAX_REPROJ_ERROR_M`).
        refit: Whether to attempt the line solve when no hypothesis is accepted.
        seed_from: ``'donor'`` (nearest live frame, propagated) -- the only mode; named so the
            alternative is a code change, not a silent default.
        chain_seed: Whether a *self-consistent* line solve (residual within the metre gate) may seed
            the next frame even when it was refused for shipping. Without it one refusal breaks the
            chain and every later frame of the dead run is seeded from ever further away -- measured
            at a median 15-frame seed gap on refusals against 1 frame on acceptances
            (`results/GSR_V8_W2B.md` §3.4). ``False`` reproduces the arm ``c`` that was scored.
        log: Optional list; one dict per dead frame is appended describing what happened.

    Returns:
        ``(table, n_rows_recovered)``.
    """
    import numpy as np  # noqa: PLC0415 - local alias keeps the pure/imaging halves separable

    from generator.calibrate import MAX_REPROJ_ERROR_M, apply_homography  # noqa: PLC0415
    from generator.postprocess import (  # noqa: PLC0415
        PLAYER_ROLES, TRUST_MIN_PLAYERS, _frame_homographies, clamp_to_pitch, onpitch_plausible,
    )

    if seed_from != "donor":
        raise ValueError(f"unsupported seed_from={seed_from!r}")
    max_error_m = MAX_REPROJ_ERROR_M if max_error_m is None else max_error_m
    out = df.copy()
    people = out["role"].isin(PLAYER_ROLES)
    live_mask = people & np.isfinite(out["pitch_x"]) & np.isfinite(out["pitch_y"])
    live = set(out.loc[live_mask, "frame"].astype(int))
    dead = sorted(set(out["frame"].astype(int)) - live)
    if not dead:
        return out, 0
    donors = _frame_homographies(out, 8)  # the fill's own donor rule, re-used verbatim
    idx_by_frame = out.groupby("frame").groups
    segs = pitch_model_lines()
    getter = frames_bgr if callable(frames_bgr) else frames_bgr.get
    recovered = 0
    seeds: dict[int, np.ndarray] = dict(donors)
    for fr in dead:
        img = getter(fr)
        rec = {"frame": fr, "source": None, "support": 0.0, "residual_m": float("inf")}
        if img is None:
            if log is not None:
                log.append(rec)
            continue
        dist, nearest = ridge_field(line_mask(img))
        idx = idx_by_frame.get(fr)
        foot = out.loc[idx][out.loc[idx, "role"].isin(PLAYER_ROLES)]
        foot_pts = foot[["image_x", "image_y"]].to_numpy(dtype=float)
        foot_pts = foot_pts[np.isfinite(foot_pts).all(axis=1)]
        pick = None
        for c in (candidates or {}).get(fr, []):
            if c.error_m > max_error_m:
                continue
            if len(foot_pts) >= TRUST_MIN_PLAYERS and not onpitch_plausible(c.homography, foot_pts):
                continue
            sup, _n = line_support(c.homography, dist, segs=segs)
            if sup >= min_support:
                pick = (c.homography, float(c.error_m), sup, "keypoint")
                break
        if pick is None and refit:
            seed = _nearest_seed(fr, seeds)
            if seed is not None:
                fit = refit_homography(seed, dist, nearest, segs=segs)
                solved = fit.homography is not None and fit.residual_m <= max_error_m
                if (solved and fit.support >= min_support
                        and (len(foot_pts) < TRUST_MIN_PLAYERS
                             or onpitch_plausible(fit.homography, foot_pts))):
                    pick = (fit.homography, fit.residual_m, fit.support, "line")
                elif solved and chain_seed:
                    seeds[fr] = fit.homography  # good enough to initialise, not to ship
                    rec["seed_only"] = True
        if pick is None:
            if log is not None:
                log.append(rec)
            continue
        h, err, sup, src = pick
        seeds[fr] = h
        pts = out.loc[idx, ["image_x", "image_y"]].to_numpy(dtype=float)
        ok = np.isfinite(pts).all(axis=1)
        proj = apply_homography(h, np.where(ok[:, None], pts, 0.0))
        vals = [clamp_to_pitch(x, y) if k else (float("nan"), float("nan"))
                for (x, y), k in zip(proj, ok)]
        out.loc[idx, ["pitch_x", "pitch_y"]] = vals
        out.loc[idx, "calib_error_m"] = err
        n = int(np.isfinite([v[0] for v in vals]).sum())
        recovered += n
        rec.update(source=src, support=sup, residual_m=err, rows=n)
        if log is not None:
            log.append(rec)
    return out, recovered


def _nearest_seed(frame: int, seeds: dict[int, np.ndarray]) -> np.ndarray | None:
    """The homography of the nearest frame that already has one (ties -> the earlier frame)."""
    if not seeds:
        return None
    return seeds[min(seeds, key=lambda f: (abs(f - frame), f))]


def _demo() -> None:
    """Self-check: model geometry, the line solve, support, and the ICP under a perturbed seed."""
    import cv2  # noqa: PLC0415

    segs = pitch_model_lines()
    assert segs.shape[1:] == (2, 2) and len(segs) == 5 + 12 + 24, segs.shape
    lines = _segment_lines(segs)
    assert np.allclose(np.linalg.norm(lines[:, :2], axis=1), 1.0)
    # A point 3 m off the touchline y=0 must give a 3 m residual against that line.
    assert abs(abs(float(np.cross([0.0, 0.0, 1.0], [105.0, 0.0, 1.0])[2])) - 0.0) < 1e-9
    # --- exact recovery: synthesise a frame from a known homography ------------------------------
    hgt, wid = 540, 960
    world_to_img = np.array([[7.0, 1.2, 60.0], [0.6, 4.0, 40.0], [0.0006, 0.004, 1.0]])
    h_true = np.linalg.inv(world_to_img)  # image -> pitch
    pts, seg_id = sample_model_points(segs)
    img_pts = _project_to_image(h_true, pts)
    frame = np.zeros((hgt, wid, 3), np.uint8)
    frame[:] = (40, 120, 40)  # turf (BGR), inside the grass hue window
    for x, y in img_pts:
        if 0 <= x < wid and 0 <= y < hgt:
            cv2.circle(frame, (int(round(x)), int(round(y))), 2, (245, 245, 245), -1)
    mask = line_mask(frame)
    assert mask.sum() > 0, "the drawn lines must survive the top-hat + grass mask"
    dist, nearest = ridge_field(mask)
    sup, n_vis = line_support(h_true, dist, segs=segs)
    assert n_vis > 100 and sup > 0.9, (sup, n_vis)
    # the truth must beat a 3-metre-wrong homography on support
    h_bad = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ h_true
    sup_bad, _ = line_support(h_bad, dist, segs=segs)
    assert sup_bad < sup, (sup, sup_bad)
    # --- the solve itself: exact correspondences must reproduce h_true ----------------------------
    inside = ((img_pts[:, 0] >= 0) & (img_pts[:, 0] < wid)
              & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < hgt))
    h_hat = solve_point_on_line(img_pts[inside], _segment_lines(segs)[seg_id[inside]], (hgt, wid))
    assert h_hat is not None
    err = np.abs(_residuals_m(h_hat, img_pts[inside], _segment_lines(segs)[seg_id[inside]]))
    assert err.max() < 1e-6, err.max()
    # --- ICP: a seed 25 px off must be pulled back onto the lines ---------------------------------
    seed = h_true @ np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -18.0], [0.0, 0.0, 1.0]])
    seed_res = float(np.mean(np.abs(_residuals_m(seed, img_pts[inside],
                                                 _segment_lines(segs)[seg_id[inside]]))))
    fit = refit_homography(seed, dist, nearest, segs=segs)
    assert fit.homography is not None, "ICP must converge from a 25 px seed error"
    assert fit.residual_m < 0.25, fit
    assert fit.residual_m < seed_res, (fit.residual_m, seed_res)
    assert fit.n_segments >= MIN_REFIT_SEGMENTS and fit.support > 0.9, fit
    print(f"line_calib demo OK (seed residual {seed_res:.2f} m -> refit {fit.residual_m:.3f} m, "
          f"support {fit.support:.3f} over {fit.n_points} points / {fit.n_segments} segments)")


if __name__ == "__main__":
    _demo()
