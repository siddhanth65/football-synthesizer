"""Unit checks for the B4 M3 transfer harness (censoring geometries + external loaders).

Synthetic data only -- no Metrica, no SkillCorner, no GPU, so this stays fast enough to run
while a heavy job is in flight. The load-bearing checks are that every geometry is monotone in
its scale (otherwise the bisection that equalises visibility is meaningless) and that the
trapezoid really does censor by y (the whole point of the SkillCorner-shaped test).
"""

from __future__ import annotations

import numpy as np

from tools.imputation_b4_external import inside_quad
from tools.imputation_b4_transfer import GEOMS, censor, lagged_cam, tune_scale


def _fake_game(n: int = 400, slots: int = 22, seed: int = 0) -> dict:
    """A toy 25 fps game: players scattered over the pitch, camera drifting along x."""
    rng = np.random.default_rng(seed)
    truth = np.empty((n, slots, 2))
    truth[:, :, 0] = rng.uniform(0, 105, (n, slots))
    truth[:, :, 1] = rng.uniform(0, 68, (n, slots))
    cam = np.column_stack([np.linspace(20, 85, n), np.full(n, 34.0)])
    return {"truth": truth, "cam": cam, "fps": 25.0}


def test_every_geometry_is_monotone_in_scale() -> None:
    """The visible count must increase with scale, or tune_scale's bisection is invalid."""
    g = _fake_game()
    for geom in GEOMS:
        counts = [
            censor(g, geom, s).sum()
            for s in np.linspace(geom.lo, geom.hi, 8)
        ]
        assert counts == sorted(counts), (geom.name, counts)


def test_tune_scale_hits_the_target_visibility() -> None:
    """Each geometry is scale-tuned to the same mean visible count (that is the whole design)."""
    g = _fake_game()
    for geom in GEOMS:
        scale, achieved = tune_scale(g, geom, target=8.0)
        assert abs(achieved - 8.0) < 0.35, (geom.name, scale, achieved)


def test_trapezoid_censors_by_y_not_just_x() -> None:
    """The SkillCorner footprint must exclude the near touchline strip and widen with y."""
    geom = next(g for g in GEOMS if g.name == "trapezoid_sc")
    truth = np.array([[[52.5, 1.0], [52.5, 10.0], [52.5, 60.0], [70.0, 60.0], [70.0, 10.0]]])
    cam = np.array([[52.5, 34.0]])
    vis = geom.mask(truth, cam, 1.0, np.zeros(1, dtype=int))[0]
    assert not vis[0], "a player on the near touchline is outside the broadcast footprint"
    assert vis[1] and vis[2], "players at the camera centre are visible"
    # 17.5 m off-centre: outside the narrow near edge, inside the wide far edge.
    assert not vis[4] and vis[3]


def test_lagged_camera_delays_and_clamps() -> None:
    """A lagged camera repeats the opening frame instead of running off the front of the array."""
    cam = np.column_stack([np.arange(10.0), np.zeros(10)])
    out = lagged_cam(cam, fps=25.0, lag_s=0.08)  # 2 frames
    assert out[0, 0] == 0.0 and out[5, 0] == 3.0
    assert lagged_cam(cam, 25.0, 0.0) is cam


def test_inside_quad_matches_a_hand_checked_case() -> None:
    """Point-in-quad works for both ring orientations."""
    q = np.array([[[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]]])
    pts = np.array([[5.0, 5.0], [11.0, 5.0], [-0.5, 5.0]])
    got = inside_quad(pts, np.repeat(q, 3, axis=0))
    assert list(got) == [True, False, False]
    rev = np.repeat(q[:, ::-1], 3, axis=0)
    assert list(inside_quad(pts, rev)) == [True, False, False]
