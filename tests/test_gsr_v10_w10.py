"""Pure seams of the v10-W10 deletion-bucket instrument."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.gsr_v10_w10 import FLICKER_MAX, STABLE_MIN, _demo, _runs, auc, bucket_of, legality


def test_demo() -> None:
    """The module self-check must pass."""
    _demo()


def test_runs_lengths() -> None:
    """Every element of a True run carries the run's full length."""
    assert _runs([True, False, True, True, True]) == [1, 0, 3, 3, 3]
    assert _runs([False] * 4) == [0, 0, 0, 0]


def test_bucket_boundaries() -> None:
    """The flicker/mid/stable cut points are exactly the registered ones."""
    assert bucket_of(9, 7, FLICKER_MAX) == "wrong_flicker"
    assert bucket_of(9, 7, FLICKER_MAX + 1) == "wrong_mid"
    assert bucket_of(9, 7, STABLE_MIN - 1) == "wrong_mid"
    assert bucket_of(9, 7, STABLE_MIN) == "wrong_stable"
    # a tolerated row (run collapsed to 0) reads as a duplicate, junk still reads as junk
    assert bucket_of(9, 7, 0) == "dup_owner"
    assert bucket_of(-1, 7, 0) == "junk"


def test_auc_direction() -> None:
    """AUC > 0.5 means the deleted population scores higher on that signal."""
    assert auc(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == 0.0


def test_legality_flags_mandatory_deletions() -> None:
    """A restored deletion re-creates a duplicate id in a timestep, so no bucket is optional."""
    t = pd.DataFrame({
        "seq": ["S"] * 3, "idx": [0, 1, 2], "frame": [4, 4, 5], "nid": [100007, 100007, 100007],
        "dropped": [True, False, False], "bucket": ["junk", "kept", "kept"],
    })
    out = legality(t)
    assert out["full_oracle"] == {"illegal_timesteps": 0, "scoreable": True}
    assert out["merge_all_no_deletion"]["illegal_timesteps"] == 1
    assert out["buckets"]["junk"]["illegal_timesteps"] == 1
    assert out["any_bucket_optional"] is False
