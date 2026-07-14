"""Tests for the pitch-control surface (no ball)."""

from __future__ import annotations

import numpy as np

from fingerprint.pitch_control import control_field


def test_control_field_split_and_midpoint_balanced():
    # team 0 left (x=40), team 1 right (x=65), symmetric about the halfway line (52.5)
    ctrl, gx = control_field([40.0, 65.0], [34.0, 34.0], [0, 1])
    mid_row = ctrl.shape[0] // 2
    i_mid = int(np.argmin(np.abs(gx - 52.5)))
    i_left = int(np.argmin(np.abs(gx - 40.0)))
    i_right = int(np.argmin(np.abs(gx - 65.0)))
    assert 0.35 < ctrl[mid_row, i_mid] < 0.65               # balanced between the two
    assert ctrl[mid_row, i_left] > ctrl[mid_row, i_right]    # team 0 controls its own side


def test_control_field_velocity_shifts_influence_forward():
    still, gx = control_field([40.0, 65.0], [34.0, 34.0], [0, 1])
    moving, _ = control_field([40.0, 65.0], [34.0, 34.0], [0, 1], vx=[12.0, 0.0], vy=[0.0, 0.0])
    i_ahead = int(np.argmin(np.abs(gx - 50.0)))             # a cell ahead of the moving team-0 player
    mid_row = still.shape[0] // 2
    assert moving[mid_row, i_ahead] > still[mid_row, i_ahead]  # forward velocity extends control
