"""Tests for the v9-W3 GS-DetA decomposition seams and the per-track attribute vote (pure).

No CV stack, no metric stack, no data on disk: these pin the evaluator-mirroring arithmetic
(:mod:`tools.gsr_v9_deta`) and the registered stage-2 component
(:func:`eval.gsr_score.vote_track_attributes`).
"""

from __future__ import annotations

import numpy as np

from eval.gsr_score import vote_track_attributes
from tools.gsr_v9_deta import (
    ALPHAS,
    SIGMA,
    alphas_passed,
    effective_attrs,
    match_positions,
    similarity,
)


def test_sigma_and_alphas_mirror_the_official_evaluator():
    # soccernet_gs.calculate_sigma: 5 m must score exactly the smallest HOTA alpha.
    assert abs(float(np.exp(-0.5 * (5.0 / SIGMA) ** 2)) - 0.05) < 1e-12
    assert len(ALPHAS) == 19 and abs(ALPHAS[0] - 0.05) < 1e-12 and abs(ALPHAS[-1] - 0.95) < 1e-12
    # A pair 5 m apart clears one threshold; a perfect pair clears all of them.
    assert alphas_passed(0.05) == 1
    assert alphas_passed(1.0) == 19
    assert alphas_passed(0.049) == 0


def test_match_positions_is_one_to_one_and_cut_at_five_metres():
    gt = np.array([[0.0, 0.0], [3.0, 0.0]])
    pr = np.array([[0.2, 0.0], [3.1, 0.0], [40.0, 0.0]])
    pairs = match_positions(similarity(gt, pr))
    assert pairs == [(0, 0), (1, 1)]
    # Beyond the 5 m tolerance nothing matches, however alone the two rows are.
    assert match_positions(similarity(np.array([[0.0, 0.0]]), np.array([[5.5, 0.0]]))) == []


def test_effective_attrs_mirrors_the_evaluator_preprocessing():
    # Team only for players/GKs, jersey only for players; both sides get the same treatment, so
    # None == None matches (an unnumbered GT row is free when we also emit null).
    assert effective_attrs("referee", "left", "7") == ("referee", None, None)
    assert effective_attrs("goalkeeper", "left", "7") == ("goalkeeper", "left", None)
    assert effective_attrs("player", "left", 7) == ("player", "left", "7")
    assert effective_attrs("player", None, "") == ("player", None, None)


def _rows(track: int, values: list[tuple[str, str | None, str | None]]) -> list[dict]:
    return [{"track_id": track, "attributes": {"role": r, "team": t, "jersey": j}}
            for r, t, j in values]


def test_vote_collapses_a_track_onto_its_majority_and_leaves_the_ball_alone():
    preds = _rows(1, [("player", "left", "7"), ("player", "left", "7"),
                      ("referee", None, None)]) + _rows(9, [("ball", None, None)])
    changed = vote_track_attributes(preds, ("role", "team", "jersey"))
    assert [p["attributes"]["role"] for p in preds] == ["player", "player", "player", "ball"]
    assert preds[2]["attributes"] == {"role": "player", "team": "left", "jersey": "7"}
    assert changed == {"role": 1, "team": 1, "jersey": 1}


def test_vote_never_invents_a_value_and_is_a_noop_with_no_keys():
    preds = _rows(1, [("referee", None, None), ("referee", None, None)])
    assert vote_track_attributes(preds, ("role", "team")) == {"role": 0, "team": 0}
    assert preds[0]["attributes"]["team"] is None  # no non-null candidate -> stays null
    preds2 = _rows(2, [("player", "left", None), ("player", "right", None)])
    assert vote_track_attributes(preds2, ()) == {}
    assert preds2[1]["attributes"]["team"] == "right"  # untouched when the flag is off
