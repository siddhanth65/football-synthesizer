"""Tests for the pure seams of generator.extract (transform+gate, row building, schema).

These run without the CV stack -- importing generator.extract must not require opencv/ultralytics
because the heavy imports are inside extract_positions().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.calibrate import CalibrationResult, apply_homography
from generator.extract import (
    POSITIONS_COLUMNS,
    build_ball_row,
    build_player_rows,
    empty_positions,
    transform_and_gate,
)

_H = np.array([[0.05, 0.002, -3.0], [0.001, 0.06, -2.0], [0.0005, 0.0008, 1.0]])


def _ok_calib(img, pitch):
    return CalibrationResult(homography=_H, error_m=0.0, n_points=len(img), ok=True)


def test_transform_and_gate_projects_when_ok():
    img = np.array([[640.0, 360.0], [300.0, 500.0], [900.0, 200.0], [120.0, 690.0]])
    expected = apply_homography(_H, img)  # all land on the pitch for this H
    out = transform_and_gate(img, _ok_calib(img, expected))
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_transform_and_gate_returns_nan_when_gate_failed():
    img = np.array([[640.0, 360.0], [300.0, 500.0]])
    failed = CalibrationResult(homography=_H, error_m=9.9, n_points=2, ok=False)
    out = transform_and_gate(img, failed)
    assert np.isnan(out).all()


def test_transform_and_gate_returns_nan_without_calibrator():
    out = transform_and_gate(np.array([[1.0, 2.0]]), None)
    assert np.isnan(out).all()


def test_transform_and_gate_drops_off_pitch_points():
    # Identity homography (image coords already in 'metres'): an off-pitch point must drop to NaN.
    identity = np.eye(3)
    calib = CalibrationResult(homography=identity, error_m=0.0, n_points=2, ok=True)
    img = np.array([[52.0, 34.0], [200.0, 30.0]])  # second point is well off the 105x68 pitch
    out = transform_and_gate(img, calib)
    assert not np.isnan(out[0]).any()  # centre stays
    assert np.isnan(out[1]).any()  # the off-pitch one is dropped


def test_build_player_rows_schema_and_values():
    rows = build_player_rows(
        frame_idx=7,
        track_ids=np.array([3, 4]),
        teams=np.array([0, 1]),
        foot_xy=np.array([[10.0, 20.0], [30.0, 40.0]]),
        pitch_xy=np.array([[50.0, 34.0], [np.nan, np.nan]]),
        confs=np.array([0.9, 0.8]),
        calib_error_m=1.2,
    )
    assert len(rows) == 2
    assert rows[0]["frame"] == 7 and rows[0]["role"] == "player"
    assert rows[0]["track_id"] == 3 and rows[0]["team"] == 0
    assert rows[0]["calib_error_m"] == 1.2
    assert np.isnan(rows[1]["pitch_x"])


def test_build_player_rows_carries_roles():
    rows = build_player_rows(
        frame_idx=0,
        track_ids=np.array([1, 2, 3]),
        teams=np.array([0, 1, -1]),
        foot_xy=np.zeros((3, 2)),
        pitch_xy=np.zeros((3, 2)),
        confs=np.full(3, 0.9),
        calib_error_m=0.1,
        roles=np.array(["player", "goalkeeper", "referee"]),
    )
    assert [r["role"] for r in rows] == ["player", "goalkeeper", "referee"]
    assert rows[2]["team"] == -1  # referee carried with no team


def test_build_ball_row():
    row = build_ball_row(2, np.array([100.0, 200.0]), np.array([52.5, 34.0]), 0.5)
    assert row["role"] == "ball" and row["track_id"] == -1 and row["team"] == -1
    assert row["pitch_x"] == 52.5 and row["conf"] == 1.0


def test_empty_positions_has_full_schema():
    df = empty_positions()
    assert tuple(df.columns) == POSITIONS_COLUMNS
    assert len(df) == 0


def test_built_rows_are_postprocessable():
    """Built rows feed postprocess (with a plausible, well-spread frame) and yield actor/keeper flags."""
    from generator.postprocess import postprocess

    n = 12  # enough, well-spread players to pass the plausibility gate
    track_ids = np.arange(n)
    teams = np.array([0] * 6 + [1] * 6)
    pitch = np.column_stack([
        np.concatenate([np.linspace(5, 45, 6), np.linspace(60, 100, 6)]),
        np.linspace(10, 58, n),
    ])
    foot = np.zeros((n, 2))
    rows = build_player_rows(0, track_ids, teams, foot, pitch, np.full(n, 0.9), 0.4)
    rows.append(build_ball_row(0, np.array([0.0, 0.0]), pitch[0] + np.array([0.3, 0.0]), 0.4))
    out = postprocess(pd.DataFrame(rows))
    assert out["pitch_x"].notna().any()  # plausible frame survived the gate
    assert out["is_actor"].sum() == 1 and out["is_keeper"].sum() == 2
