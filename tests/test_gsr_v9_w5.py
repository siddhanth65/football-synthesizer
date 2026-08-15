"""Tests for the v9-W5 second-keeper repair and the row-level sub-population audit.

Pure functions only -- no CV stack, no metric stack, no data on disk. These pin the temporal guard
that W4 lacked (``role2`` / ``role2dom``) and the audit that reports right->wrong rows.
"""

from __future__ import annotations

from eval.gsr_score import GK_MIN_ABSX_M, GK_MIN_PEN_FRAC, gk_side_repair
from tools.gsr_v9_w5 import audit_rows

DEEP = -(GK_MIN_ABSX_M + 10.0)
SHALLOW = -(GK_MIN_ABSX_M + 2.0)


def _rows(track: int, role: str, team: str | None, x: float, frames: list[str],
          y: float = 0.0) -> list[dict]:
    return [{"image_id": f, "track_id": track,
             "attributes": {"role": role, "team": team, "jersey": None},
             "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y}} for f in frames]


def test_role2_recovers_a_keeper_fragment_that_never_coexists_with_the_labelled_one():
    preds = (_rows(1, "goalkeeper", "right", SHALLOW, ["a", "b"])
             + _rows(2, "player", "right", DEEP, ["c", "d"]))
    changed = gk_side_repair(preds, ("role2", "team"))
    assert changed == {"role2": 2, "team": 4}
    assert [p["attributes"]["role"] for p in preds] == ["goalkeeper"] * 4
    assert {p["attributes"]["team"] for p in preds} == {"left"}


def test_role2_still_refuses_a_concurrent_candidate_but_role2dom_takes_the_deeper_one():
    preds = (_rows(1, "goalkeeper", "right", SHALLOW, ["a", "b"])
             + _rows(2, "player", "right", DEEP, ["a"]))
    assert gk_side_repair([dict(p) for p in preds], ("role2",))["role2"] == 0
    assert gk_side_repair([dict(p) for p in preds], ("role2dom",))["role2dom"] == 1
    # the concurrent candidate is SHALLOWER than the labelled keeper -> dominance refuses too
    other = (_rows(1, "goalkeeper", "right", DEEP, ["a", "b"])
             + _rows(2, "player", "right", SHALLOW, ["a"]))
    assert gk_side_repair(other, ("role2dom",))["role2dom"] == 0


def test_role2_keeps_the_w4_thresholds_and_the_half_split():
    # outside the penalty box in y -> penalty share 0 -> not a candidate at all
    assert gk_side_repair(_rows(1, "player", "right", DEEP, ["a"], y=30.0),
                          ("role2",))["role2"] == 0
    # not deep enough in x
    assert gk_side_repair(_rows(1, "player", "right", -(GK_MIN_ABSX_M - 1.0), ["a"]),
                          ("role2",))["role2"] == 0
    # a keeper on the OTHER half never blocks this one
    preds = (_rows(1, "goalkeeper", "left", -DEEP, ["a"])
             + _rows(2, "player", "right", DEEP, ["a"]))
    assert gk_side_repair(preds, ("role2",))["role2"] == 1
    assert GK_MIN_PEN_FRAC == 0.90


def test_audit_rows_counts_every_sub_population_and_ignores_untouched_rows():
    ctrl = (_rows(7, "player", "left", DEEP, ["a"]) + _rows(8, "player", "left", DEEP, ["b"])
            + _rows(9, "player", "left", DEEP, ["c"]))
    arm = (_rows(7, "goalkeeper", "right", DEEP, ["a"])   # wrong -> right
           + _rows(8, "goalkeeper", "left", DEEP, ["b"])  # right -> wrong
           + _rows(9, "player", "left", DEEP, ["c"]))     # untouched
    gtm = {("a", 7): {"role": "goalkeeper", "team": "right"},
           ("b", 8): {"role": "player", "team": "left"},
           ("c", 9): {"role": "player", "team": "left"}}
    assert audit_rows(ctrl, arm, gtm) == {"wrong2right": 1, "right2wrong": 1}
    assert audit_rows(ctrl, ctrl, gtm) == {}
    assert audit_rows(ctrl, arm, {}) == {"changed_unmatched": 2}
