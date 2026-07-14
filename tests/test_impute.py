"""Tests for P2 off-screen handling: de-biased defensive line + direction completion + role imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.impute import (
    BACK_REF,
    LINE_DEBIAS_SLOPE,
    complete_directions,
    debias_slope,
    line_estimates,
    line_from_deepest,
)


def test_line_from_deepest_averages_the_deepest_n():
    ax = np.array([10.0, 12.0, 14.0, 60.0, 70.0])   # deepest 4 = 10,12,14,60
    assert line_from_deepest(ax, n=4) == (10 + 12 + 14 + 60) / 4
    assert line_from_deepest(ax, n=2) == 11.0


def test_complete_directions_fills_the_opposite_goal():
    assert complete_directions({0: 1}) == {0: 1, 1: -1}          # single goalmouth -> infer the other
    assert complete_directions({1: -1}) == {1: -1, 0: 1}
    assert complete_directions({0: 1, 1: -1}) == {0: 1, 1: -1}   # already complete -> unchanged
    assert complete_directions({}) == {}


def _one_frame(team_line_x, back_role_tracks, n_back_roles):
    """A single-frame team: a keeper at x=5 (attacks +1) + 4 outfield around ``team_line_x``.

    ``n_back_roles`` of the outfield tracks are labelled a back-line role, the rest midfield, so the
    returned ``roles`` table produces exactly that visible back-line count.
    """
    rows = [{"chunk": "c0", "frame": 0, "team": 0, "role": "goalkeeper", "is_keeper": True,
             "track_id": 99, "pitch_x": 5.0, "pitch_y": 34.0}]
    roles = [{"chunk": "c0", "team": 0, "track_id": 99, "role": "GK", "formation": "4-4-2",
              "fit_cost_m": 1.0, "frames": 1}]
    xs = [team_line_x, team_line_x + 2, team_line_x + 4, team_line_x + 30]
    back = ["LCB", "RCB", "LB", "RB"]
    for i, x in enumerate(xs):
        rows.append({"chunk": "c0", "frame": 0, "team": 0, "role": "player", "is_keeper": False,
                     "track_id": i, "pitch_x": x, "pitch_y": 20.0 + i})
        roles.append({"chunk": "c0", "team": 0, "track_id": i,
                      "role": back[i] if i < n_back_roles else "CM", "formation": "4-4-2",
                      "fit_cost_m": 1.0, "frames": 1})
    return pd.DataFrame(rows), pd.DataFrame(roles)


def test_line_estimates_debiases_down_when_back_line_censored():
    pos, roles = _one_frame(20.0, back_role_tracks=None, n_back_roles=0)  # no back-line seen
    le = line_estimates(pos, roles)
    row = le[le["team"] == 0].iloc[0]
    raw = (20 + 22 + 24 + 50) / 4                      # deepest-4 mean
    assert row["line_raw"] == raw
    assert row["n_back"] == 0
    # fully censored -> full correction of slope*BACK_REF (downward, slope<0)
    assert row["line_debiased"] == raw + LINE_DEBIAS_SLOPE * (BACK_REF - 0)
    assert row["line_debiased"] < row["line_raw"]


def test_line_estimates_no_correction_when_back_line_fully_visible():
    pos, roles = _one_frame(20.0, back_role_tracks=None, n_back_roles=4)  # all 4 back defenders seen
    row = line_estimates(pos, roles).iloc[0]
    assert row["n_back"] == BACK_REF
    assert row["line_debiased"] == row["line_raw"]     # nothing to correct


def test_debias_slope_recovers_negative_slope_on_censored_synthetic():
    # synthetic per-frame line estimates: line reads higher as fewer back defenders are visible
    lines = pd.DataFrame({
        "n_back": [0, 1, 2, 4] * 8,
        "line_raw": [40.0, 33.0, 27.0, 20.0] * 8,   # falls as n_back rises -> negative slope
    })
    assert debias_slope(lines, predictor="n_back") < 0
