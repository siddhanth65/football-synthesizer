"""Tests for the v9-W4 keeper repair (pure): the side rule, the role rule and their guards.

No CV stack, no metric stack, no data on disk -- these pin
:func:`eval.gsr_score.gk_side_repair`, the registered stage-2 component, and the track bucketing
of :mod:`tools.gsr_v9_w4`.
"""

from __future__ import annotations

import numpy as np

from eval.gsr_score import GK_MIN_ABSX_M, gk_side_repair, penalty_frac
from tools.gsr_v9_w4 import classify_track


def _rows(track: int, role: str, team: str | None, x: float, y: float = 0.0,
          n: int = 3) -> list[dict]:
    return [{"track_id": track, "attributes": {"role": role, "team": team, "jersey": None},
             "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y}} for _ in range(n)]


def test_penalty_frac_uses_the_fifa_box_in_the_centred_frame():
    # 52.5 - 16.5 = 36 m: inside the box only when |x| >= 36 AND |y| <= 20.16.
    assert penalty_frac(np.array([40.0]), np.array([0.0])) == 1.0
    assert penalty_frac(np.array([-40.0]), np.array([0.0])) == 1.0
    assert penalty_frac(np.array([35.0]), np.array([0.0])) == 0.0
    assert penalty_frac(np.array([40.0]), np.array([25.0])) == 0.0
    assert penalty_frac(np.array([]), np.array([])) == 0.0


def test_side_rule_puts_a_keeper_in_the_half_he_stands_in():
    # The kit cluster says 'right' for a keeper standing at x = -47: geometry overrides it.
    preds = _rows(1, "goalkeeper", "right", -47.0) + _rows(2, "goalkeeper", "left", 48.0)
    changed = gk_side_repair(preds, ("team",))
    assert changed == {"team": 6}
    assert preds[0]["attributes"]["team"] == "left"
    assert preds[3]["attributes"]["team"] == "right"


def test_side_rule_leaves_outfield_rows_and_an_empty_flag_alone():
    preds = _rows(1, "player", "right", -47.0) + _rows(2, "referee", None, 10.0)
    assert gk_side_repair(preds, ("team",)) == {"team": 0}
    assert preds[0]["attributes"]["team"] == "right"
    assert gk_side_repair(_rows(1, "goalkeeper", "right", -47.0), ()) == {}


def test_role_rule_takes_the_most_extreme_candidate_on_a_keeperless_half():
    preds = (_rows(1, "player", "right", -47.0)      # the missed keeper, deepest
             + _rows(2, "player", "right", -38.0)    # a deep defender, in the box but shallower
             + _rows(3, "player", "left", -10.0))    # midfield: not a candidate at all
    changed = gk_side_repair(preds, ("role", "team"))
    assert changed["role"] == 3
    assert [p["attributes"]["role"] for p in preds[:3]] == ["goalkeeper"] * 3
    assert [p["attributes"]["role"] for p in preds[3:]] == ["player"] * 6
    assert preds[0]["attributes"]["team"] == "left"   # the side rule then fixes his team
    assert preds[3]["attributes"]["team"] == "right"  # the defender keeps his kit team


def test_role_rule_never_creates_a_second_keeper_on_a_half():
    preds = _rows(1, "goalkeeper", "left", -49.0) + _rows(2, "player", "right", -47.0)
    assert gk_side_repair(preds, ("role", "team"))["role"] == 0
    assert preds[3]["attributes"]["role"] == "player"


def test_role_rule_respects_its_train_fitted_thresholds():
    # Just inside the box in x but outside in y -> penalty share 0 -> not a candidate.
    preds = _rows(1, "player", "right", -(GK_MIN_ABSX_M + 5.0), y=30.0)
    assert gk_side_repair(preds, ("role", "team"))["role"] == 0
    # Deep enough and inside the box -> a candidate.
    preds = _rows(1, "player", "right", -(GK_MIN_ABSX_M + 5.0), y=0.0)
    assert gk_side_repair(preds, ("role", "team"))["role"] == 3


def test_classify_track_names_the_stage_one_buckets():
    base = {"n_matched": 10, "purity": 1.0, "gt_role": "goalkeeper", "our_role": "goalkeeper",
            "gt_team": "left", "our_team": "left"}
    assert classify_track(base) == "ok"
    assert classify_track({**base, "our_team": "right"}) == "team:goalkeeper"
    assert classify_track({**base, "our_role": "player"}) == "role:goalkeeper->player"
    assert classify_track({**base, "purity": 0.5}) == "contaminated"
    assert classify_track({**base, "n_matched": 0}) == "unmatched"
