"""Pure-seam checks for the Man Utd identity profile engine (no I/O, no GPU)."""
from __future__ import annotations

import numpy as np

from tools import manutd_identity as mi


def test_channel_and_zone_binning() -> None:
    """Lateral and vertical thirds bin at the right edges, endpoints clamped."""
    assert list(mi._bin(np.array([0.0, 34.0, 68.0]), mi.CHAN_EDGES, mi.CHANNELS)) == \
        ["left", "central", "right"]
    assert list(mi._bin(np.array([0.0, 52.5, 105.0]), mi.ZONE_EDGES, mi.ZONES)) == list(mi.ZONES)


def test_sign_test_matches_exact_binomial() -> None:
    """The two-sided exact sign test agrees with hand-computed tails."""
    assert mi._sign_p(6, 12) == 1.0
    assert mi._sign_p(12, 12) == 0.0005
    assert mi._sign_p(11, 12) == 0.0063
    assert mi._sign_p(1, 12) == mi._sign_p(11, 12)


def test_variance_test_is_two_sided_and_symmetric() -> None:
    """The F-test gives the same p whichever way the two SDs are passed."""
    a, b = {"sd": 4.25, "n": 12}, {"sd": 3.09, "n": 12}
    assert mi._var_test(a, b)["p"] == mi._var_test(b, a)["p"]
    assert mi._var_test(a, a)["F"] == 1.0


def test_pooling_sums_counts_not_shares() -> None:
    """Pooling two matches weights by sample count rather than averaging per-match shares."""
    big = {"occupancy": {"left": 90, "central": 5, "right": 5}, "n_samples": 100,
           "forward_m": dict.fromkeys(mi.CHANNELS, 1.0), "entries": dict.fromkeys(mi.CHANNELS, 1),
           "grid": {z: dict.fromkeys(mi.CHANNELS, 1) for z in mi.ZONES},
           "zone_share": {z: 1 / 3 for z in mi.ZONES}}
    small = {**big, "occupancy": {"left": 0, "central": 5, "right": 5}, "n_samples": 10}
    pooled = mi._pool_ball([big, small])
    assert pooled["occupancy_share"]["left"] == 0.818     # 90/110, not the 0.45 share-average
