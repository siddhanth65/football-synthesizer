"""Tests for the v9 associator training-data factory.

Thin on purpose: :func:`tools.gsr_v9_factory._demo` asserts the pairing rule, the physical filter,
the labels and the jersey evidence on a synthetic sequence.
"""

from __future__ import annotations

from tools.gsr_v9_factory import FactoryParams, _demo, build_pairs, summarise_tracklets


def test_demo_self_check() -> None:
    """The synthetic pairing/labelling scenario holds."""
    _demo()


def test_empty_inputs_are_safe() -> None:
    """A sequence with no usable tracklet yields no pairs rather than raising."""
    import pandas as pd

    empty = pd.DataFrame(columns=["track_id", "frame", "pitch_x", "pitch_y", "role", "team",
                                  "conf", "calib_error_m", "gt_id"])
    assert summarise_tracklets(empty, {}, FactoryParams()).empty
    assert build_pairs(summarise_tracklets(empty, {}, FactoryParams()), FactoryParams()).empty
