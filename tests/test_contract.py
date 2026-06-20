"""Tests for the unified freeze-frame contract (the foundation built first)."""

from __future__ import annotations

import math

import numpy as np

from generator.contract import (
    FEATURE_NAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    FreezeFrame,
    PlayerNode,
    Substrate,
    from_statsbomb,
    node_feature_matrix,
    orient_left_to_right,
    rescale_xy,
    to_model_frame,
    to_statsbomb_dataframe,
)


def test_rescale_corner_and_centre():
    assert rescale_xy(105.0, 68.0, 105.0, 68.0) == (PITCH_LENGTH, PITCH_WIDTH)
    assert rescale_xy(52.5, 34.0, 105.0, 68.0) == (PITCH_LENGTH / 2, PITCH_WIDTH / 2)


def test_from_statsbomb_masks_velocity_and_orientation():
    ff = from_statsbomb([{"location": [60, 40], "teammate": True, "actor": True}])
    assert ff.substrate is Substrate.SB360
    p = ff.players[0]
    assert not p.has_velocity and not p.has_orientation
    _, mask = node_feature_matrix(ff)
    vi = FEATURE_NAMES.index("vx_norm")
    oi = FEATURE_NAMES.index("sin_orientation")
    assert mask[0, vi] == 0.0  # velocity masked for 360
    assert mask[0, oi] == 0.0  # orientation masked for 360


def test_orient_flips_left_to_right_and_negates_velocity():
    # Teammates sitting in the LEFT half should be flipped to attack towards x = 120.
    frame = FreezeFrame(
        players=[
            PlayerNode(x=30, y=20, is_teammate=True, vx=2.0, vy=-1.0, orientation=0.0),
            PlayerNode(x=40, y=50, is_teammate=True),
            PlayerNode(x=90, y=40, is_teammate=False),
        ],
        substrate=Substrate.BROADCAST_CV,
        ball=(35.0, 30.0),
    )
    out = orient_left_to_right(frame)
    mean_tm_x = np.mean([p.x for p in out.players if p.is_teammate])
    assert mean_tm_x >= PITCH_LENGTH / 2  # now attacking right
    p0 = out.players[0]
    assert math.isclose(p0.x, PITCH_LENGTH - 30) and math.isclose(p0.y, PITCH_WIDTH - 20)
    assert math.isclose(p0.vx, -2.0) and math.isclose(p0.vy, 1.0)  # velocity negated
    assert math.isclose(p0.orientation, math.pi)  # rotated 180 degrees
    assert out.ball == (PITCH_LENGTH - 35.0, PITCH_WIDTH - 30.0)


def test_orient_noop_when_already_left_to_right():
    frame = FreezeFrame(
        players=[PlayerNode(x=90, y=40, is_teammate=True), PlayerNode(x=30, y=40, is_teammate=False)],
        substrate=Substrate.TRACKING,
    )
    assert orient_left_to_right(frame) is frame


def test_model_bridges_shape():
    ff = from_statsbomb(
        [
            {"location": [100, 40], "teammate": True, "actor": True},
            {"location": [110, 30], "teammate": False, "keeper": True},
        ]
    )
    model_frame = to_model_frame(ff)
    assert {"x", "y", "teammate", "actor", "keeper"} == set(model_frame[0])
    df = to_statsbomb_dataframe(ff)
    assert list(df.columns) == ["location", "teammate", "actor", "keeper"]
    assert len(df) == 2


def test_feature_matrix_velocity_observed_for_cv():
    frame = FreezeFrame(
        players=[PlayerNode(x=60, y=40, is_teammate=True, vx=1.0, vy=2.0)],
        substrate=Substrate.BROADCAST_CV,
        ball=(60.0, 40.0),
    )
    x, mask = node_feature_matrix(frame)
    assert x.shape == (1, len(FEATURE_NAMES)) and mask.shape == x.shape
    vi = FEATURE_NAMES.index("vx_norm")
    assert mask[0, vi] == 1.0  # velocity observed for CV
    bi = FEATURE_NAMES.index("dist_to_ball_norm")
    assert mask[0, bi] == 1.0  # ball present
