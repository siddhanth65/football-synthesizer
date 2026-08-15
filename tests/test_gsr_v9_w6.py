"""Tests for the v9-W6 jersey name-borrow: candidate filter, one-borrow rule, writer, audit.

Pure functions only -- no model, no metric stack, no data on disk. These pin the two rules the
cost asymmetry depends on: a goalkeeper or referee track can never be a borrow endpoint, and a
borrowing track takes exactly one number regardless of how many lenders clear the bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.gsr_v9_w6 import apply_borrow, audit_rows, borrow_plan, candidate_mask, submission_rows

_COLS = {"num_mass_a": 0.0, "num_mass_b": 0.0, "gap_frames": 10, "dist_m": 1.0}


def _pair(a: int, b: int, na: int, nb: int, ra: str = "player", rb: str = "player",
          ta: int = 0, tb: int = 0) -> dict:
    return {"tid_a": a, "tid_b": b, "num_a": na, "num_b": nb, "role_a": ra, "role_b": rb,
            "team_a": ta, "team_b": tb, **_COLS}


def test_candidate_mask_needs_exactly_one_named_side():
    pairs = pd.DataFrame([_pair(1, 2, 7, -1), _pair(3, 4, -1, 7),
                          _pair(5, 6, 7, 9), _pair(7, 8, -1, -1)])
    assert list(candidate_mask(pairs)) == [True, True, False, False]


def test_candidate_mask_hard_excludes_keepers_referees_and_cross_team_pairs():
    pairs = pd.DataFrame([
        _pair(1, 2, 7, -1, rb="goalkeeper"),
        _pair(3, 4, 7, -1, rb="referee"),
        _pair(5, 6, 7, -1, ra="goalkeeper"),
        _pair(7, 8, 7, -1, ra="referee"),
        _pair(9, 10, 7, -1, tb=1),        # different team
        _pair(11, 12, 7, -1, ta=-1, tb=-1),  # team unknown on both sides
        _pair(13, 14, 7, -1),             # the only legal one
    ])
    assert list(candidate_mask(pairs)) == [False] * 6 + [True]
    assert list(candidate_mask(pd.DataFrame())) == []


def test_one_borrow_per_track_and_order_independence():
    pairs = pd.DataFrame([_pair(1, 9, 7, -1), _pair(2, 9, 11, -1), _pair(3, 9, 13, -1)])
    mask = candidate_mask(pairs)
    plan = borrow_plan(pairs, np.array([0.1, 5.0, 2.0]), mask, 0.0)
    assert [p["track_id"] for p in plan] == [9]
    assert plan[0]["number"] == 11 and plan[0]["lender"] == 2
    # ties go to the smaller lender id, so the plan cannot depend on row order
    tie = borrow_plan(pairs, np.array([5.0, 5.0, 5.0]), mask, 0.0)
    assert tie[0]["lender"] == 1 and tie[0]["number"] == 7
    rev = pairs.iloc[::-1].reset_index(drop=True)
    assert borrow_plan(rev, np.array([5.0, 5.0, 5.0]), candidate_mask(rev), 0.0) == tie
    # a lender may serve several borrowers
    many = pd.DataFrame([_pair(1, 9, 7, -1), _pair(1, 10, 7, -1)])
    assert len(borrow_plan(many, np.array([1.0, 1.0]), candidate_mask(many), 0.0)) == 2
    assert borrow_plan(pairs, np.array([0.1, 5.0, 2.0]), mask, 9.0) == []


def test_apply_borrow_writes_every_row_of_the_track_but_never_a_non_player_row():
    plan = [{"track_id": 9, "lender": 2, "number": 11, "score": 5.0, "gap_frames": 1,
             "dist_m": 0.1}]
    preds = [{"image_id": "a", "track_id": 9,
              "attributes": {"role": "player", "team": "left", "jersey": None}},
             {"image_id": "b", "track_id": 9,
              "attributes": {"role": "player", "team": "left", "jersey": None}},
             {"image_id": "a", "track_id": 9,
              "attributes": {"role": "goalkeeper", "team": "left", "jersey": None}},
             {"image_id": "a", "track_id": 4,
              "attributes": {"role": "player", "team": "left", "jersey": None}}]
    assert apply_borrow(preds, plan) == {"tracks": 1, "rows": 2}
    assert [p["attributes"]["jersey"] for p in preds] == ["11", "11", None, None]


def test_audit_splits_newly_correct_from_over_naming():
    ctrl = [{"image_id": i, "track_id": 9, "attributes": {"role": "player", "jersey": None}}
            for i in "abcd"]
    arm = [{"image_id": i, "track_id": 9, "attributes": {"role": "player", "jersey": "11"}}
           for i in "abcd"]
    gtj = {("a", 9): "11", ("b", 9): None, ("c", 9): "4"}
    assert audit_rows(ctrl, arm, gtj) == {"touched": 4, "newly_correct": 1, "overnamed": 1,
                                          "newly_wrong_named": 1, "unmatched": 1}
    assert audit_rows(ctrl, ctrl, gtj) == {}


def test_submission_rows_keeps_frames_teams_and_the_solver_numbers():
    def row(img, tid, role, team, jersey, x):
        return {"image_id": img, "track_id": tid, "confidence": 0.5,
                "attributes": {"role": role, "team": team, "jersey": jersey},
                "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": 0.0}}

    rows, jersey = submission_rows([row("2021000004", 1, "player", "left", "19", 1.0),
                                    row("2021000005", 1, "player", "left", "19", 1.5),
                                    row("2021000005", 2, "player", "right", None, 3.0),
                                    row("2021000005", 3, "ball", None, None, 0.0)])
    assert list(rows["frame"]) == [3, 4, 4]
    assert list(rows["team"]) == [0, 0, 1]
    assert set(rows["track_id"]) == {1, 2}, "the ball row must be dropped"
    assert jersey == {1: {"number": 19, "mass": 1.0, "share": 1.0, "n_reads": 1}}
