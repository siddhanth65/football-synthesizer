"""Tests for the GSR adapter seam (pure; no CV model, no metric stack).

These pin the conversion from our positions table to the SoccerNet-GSR prediction JSON: the centred
coordinate shift, the null-jersey / team-side attribute mapping, ball and NaN dropping, and the
per-sequence team-label permutation resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.gsr_score import (
    CENTRE_SHIFT_X,
    CENTRE_SHIFT_Y,
    build_submission,
    resolve_team_map,
    row_to_prediction,
    to_centred,
)


def test_to_centred_shifts_origin_to_pitch_middle():
    # A corner-origin foot point at the pitch centre becomes (0, 0) in the GSR centred frame.
    cx, cy = to_centred(CENTRE_SHIFT_X, CENTRE_SHIFT_Y)
    assert abs(cx) < 1e-9 and abs(cy) < 1e-9
    # The corner maps to the negative extreme.
    assert to_centred(0.0, 0.0) == (-CENTRE_SHIFT_X, -CENTRE_SHIFT_Y)


def test_row_to_prediction_player_carries_side_and_null_jersey():
    row = {"role": "player", "team": 1, "track_id": 7, "pitch_x": CENTRE_SHIFT_X,
           "pitch_y": CENTRE_SHIFT_Y, "image_x": 960.0, "image_y": 540.0, "conf": 0.9}
    pred = row_to_prediction(row, "2021000005", {0: "left", 1: "right"})
    assert pred is not None
    assert pred["attributes"] == {"role": "player", "jersey": None, "team": "right"}
    assert pred["supercategory"] == "object" and pred["track_id"] == 7
    assert abs(pred["bbox_pitch"]["x_bottom_middle"]) < 1e-9
    # left/middle/right all collapse to the projected foot point (no pitch-space width).
    bp = pred["bbox_pitch"]
    assert bp["x_bottom_left"] == bp["x_bottom_middle"] == bp["x_bottom_right"]


def test_referee_has_no_team_side():
    row = {"role": "referee", "team": -1, "track_id": 3, "pitch_x": 50.0, "pitch_y": 30.0,
           "image_x": 10.0, "image_y": 20.0, "conf": 0.5}
    pred = row_to_prediction(row, "2021000005", {0: "left", 1: "right"})
    assert pred["attributes"]["team"] is None and pred["attributes"]["role"] == "referee"


def test_ball_and_nan_rows_are_dropped():
    ball = {"role": "ball", "team": -1, "track_id": -1, "pitch_x": 50.0, "pitch_y": 30.0,
            "image_x": 1.0, "image_y": 2.0, "conf": 1.0}
    nan_row = {"role": "player", "team": 0, "track_id": 5, "pitch_x": np.nan, "pitch_y": np.nan,
               "image_x": 1.0, "image_y": 2.0, "conf": 1.0}
    assert row_to_prediction(ball, "2021000001", {0: "left", 1: "right"}) is None
    assert row_to_prediction(nan_row, "2021000001", {0: "left", 1: "right"}) is None


def test_build_submission_maps_frames_to_image_ids_and_skips_unknown():
    df = pd.DataFrame([
        {"frame": 0, "role": "player", "team": 0, "track_id": 1, "pitch_x": 10.0, "pitch_y": 10.0,
         "image_x": 5.0, "image_y": 6.0, "conf": 0.8},
        {"frame": 9, "role": "player", "team": 0, "track_id": 1, "pitch_x": 11.0, "pitch_y": 10.0,
         "image_x": 5.0, "image_y": 6.0, "conf": 0.8},  # frame not in the map -> skipped
    ])
    sub = build_submission(df, {0: "2021000001"}, {0: "left", 1: "right"})
    assert len(sub["predictions"]) == 1
    assert sub["predictions"][0]["image_id"] == "2021000001"


def test_resolve_team_map_prefers_the_higher_agreement_permutation():
    # GT: a left player near (-20, 0) and a right player near (20, 0) (centred metres) each frame.
    gt_people = {f: [(-20.0, 0.0, "player", "left"), (20.0, 0.0, "player", "right")]
                 for f in range(5)}
    # Our team 1 sits by the left GT player, team 0 by the right one -> swapped mapping wins.
    rows = []
    for f in range(5):
        rows.append({"frame": f, "role": "player", "team": 1, "track_id": 1,
                     "pitch_x": to_gt(-20.0), "pitch_y": to_gt_y(0.0)})
        rows.append({"frame": f, "role": "player", "team": 0, "track_id": 2,
                     "pitch_x": to_gt(20.0), "pitch_y": to_gt_y(0.0)})
    df = pd.DataFrame(rows)
    assert resolve_team_map(df, gt_people) == {0: "right", 1: "left"}


def to_gt(centred_x: float) -> float:
    """Inverse of the centred shift on x (helper for building uncentred test rows)."""
    return centred_x + CENTRE_SHIFT_X


def to_gt_y(centred_y: float) -> float:
    """Inverse of the centred shift on y (helper for building uncentred test rows)."""
    return centred_y + CENTRE_SHIFT_Y
