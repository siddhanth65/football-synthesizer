"""Targeted tests for the v9 W2 associator: the model's masking contract and the driver's plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.assoc_twix import SIDE_DIM, _demo as assoc_demo
from tools.gsr_v9_train import _demo as train_demo, greedy_merges, side_features
from tools.gsr_v9_factory import FactoryParams, summarise_tracklets


def test_assoc_twix_demo() -> None:
    """Shapes, padding invariance, inter-pair masking and the constant-scale property hold."""
    assoc_demo()


def test_train_demo() -> None:
    """Window builder, matched-count merger and the GT oracle splitter behave as specified."""
    train_demo()


def test_side_features_layout() -> None:
    """A jersey conflict, an unknown team and a missing CLIP distance all land in the right slots."""
    pairs = pd.DataFrame([{
        "team_a": -1, "team_b": 0, "role_same": 0, "num_agree": 0, "num_mass_a": 2.0,
        "num_mass_b": 4.0, "clip_cos_dist": np.nan, "gap_s": 100.0,
    }])
    f = side_features(pairs)[0]
    assert f.shape == (SIDE_DIM,)
    assert f[0] == 0.0 and f[1] == 1.0 and f[2] == 0.0
    assert f[3] == 0.0 and f[4] == 1.0
    assert f[6] == 0.6 and f[7] == 1.0  # missing appearance -> the impostor prior, flagged
    assert f[8] == 4.0  # gap clipped


def test_greedy_merges_respects_temporal_chain() -> None:
    """A top-scored pair that would put one identity in two places at once is refused."""
    rows = pd.DataFrame({
        "frame": list(range(0, 20)) + list(range(10, 30)),
        "track_id": [1] * 20 + [2] * 20,
        "role": "player", "team": 0, "conf": 0.9, "calib_error_m": 0.1,
        "pitch_x": 10.0, "pitch_y": 34.0, "gt_id": 5,
    })
    tr = summarise_tracklets(rows, {}, FactoryParams())
    pairs = pd.DataFrame([{"tid_a": 1, "tid_b": 2}])
    remap, accepted = greedy_merges(tr, pairs, np.array([9.0]), 1)
    assert remap == {1: 1, 2: 2} and accepted == []
