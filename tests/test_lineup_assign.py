"""Targeted tests for the pure seams of :mod:`generator.lineup_assign` (B-1).

No GPU, no artifacts -- cost matrix, abstention, sub re-keying, GK veto, window derivation.
"""
from __future__ import annotations

import pandas as pd

from generator import lineup_assign as la


def _p(team, name, shirt, pos, is_sub, minutes):
    on_h1, on_h2 = la._window_flags(is_sub, minutes)
    return la.PlayerCand(team, name, shirt, pos, is_sub, minutes, on_h1, on_h2)


def test_window_flags():
    assert la._window_flags(False, 90.0) == (True, True)     # ever-present starter
    assert la._window_flags(False, 45.0) == (True, False)    # subbed off at HT
    assert la._window_flags(False, 65.0) == (True, True)     # subbed at 65'
    assert la._window_flags(True, 11.0) == (False, True)     # late sub
    assert la._window_flags(True, 46.0) == (True, True)      # first-half sub still crosses HT
    assert la._window_flags(True, float("nan")) == (False, False)


def test_position_cost_bands():
    assert la.position_cost("F", 0.82) == 0.0
    assert la.position_cost("G", 0.05) == 0.0
    assert la.position_cost("D", 0.82) > la.position_cost("D", 0.30)
    assert la.position_cost("M", None) == la.NEUTRAL_U_COST


def test_cost_matrix_jersey_is_infeasible_on_mismatch():
    mu = _p(0, "Bruno", 8, "M", False, 79.0)
    df = _p(0, "Dalot", 20, "D", False, 90.0)
    g8 = la.TrackGroup(0, "h1", 8, 1, 3, 0.55, False, (1,))
    m = la.build_cost_matrix([mu, df], [g8])
    assert m[0, 0] < la.ABSTAIN_FLOOR       # #8 midfielder -> group #8 assignable
    assert m[1, 0] >= la.INFEASIBLE         # #20 cannot be group #8


def test_hungarian_two_numbers_no_conflict():
    mu = _p(0, "Bruno", 8, "M", False, 79.0)
    df = _p(0, "Dalot", 20, "D", False, 90.0)
    g8 = la.TrackGroup(0, "h1", 8, 2, 5, 0.55, False, (1,))
    g20 = la.TrackGroup(0, "h1", 20, 1, 3, 0.30, False, (2,))
    res = la.solve_assignment([mu, df], [g8, g20])
    assert {a.number: a.player for a in res} == {8: "Bruno", 20: "Dalot"}
    assert all(a.confidence > 0.8 for a in res)


def test_abstain_when_no_candidate():
    df = _p(0, "Dalot", 20, "D", False, 90.0)
    ghost = la.TrackGroup(0, "h1", 99, 1, 1, None, False, (7,))  # no #99 rostered
    res = la.solve_assignment([df], [ghost])
    assert res[0].player is None and res[0].method == "abstain:no_candidate"


def test_sub_rekeying_by_half():
    mu = _p(0, "Bruno", 8, "M", False, 79.0)
    antony = _p(0, "Antony", 21, "M", True, 8.0)  # h2 only
    g21_h1 = la.TrackGroup(0, "h1", 21, 1, 1, None, False, (9,))
    g21_h2 = la.TrackGroup(0, "h2", 21, 1, 1, None, False, (9,))
    # h1: Antony not a candidate -> abstain.
    res_h1 = la.solve_assignment([mu], [g21_h1])
    assert res_h1[0].player is None
    # h2: Antony present -> assigned.
    res_h2 = la.solve_assignment([mu, antony], [g21_h2])
    assert res_h2[0].player == "Antony"


def test_gk_veto():
    mu = _p(0, "Bruno", 8, "M", False, 79.0)
    keeper_grp = la.TrackGroup(0, "h1", 8, 1, 1, 0.05, True, (11,))  # keeper track, outfield #8
    res = la.solve_assignment([mu], [keeper_grp])
    assert res[0].player is None  # veto -> abstain, never forced


def test_low_vote_mass_still_assigns_but_lower_conf():
    a1 = la.solve_assignment([_p(0, "X", 5, "D", False, 90.0)],
                             [la.TrackGroup(0, "h1", 5, 1, 1, None, False, (1,))])[0]
    a5 = la.solve_assignment([_p(0, "X", 5, "D", False, 90.0)],
                             [la.TrackGroup(0, "h1", 5, 1, 5, None, False, (1,))])[0]
    assert a1.player == "X" and a5.player == "X"
    assert a5.confidence > a1.confidence


def test_roster_candidates_parses_oracle_shape():
    orc = pd.DataFrame({
        "name": ["Keeper A", "Sub B"],
        "teamName": ["Man Utd FC", "Man Utd FC"],
        "shirtNumber": ["1", "21"],
        "position": ["G", "M"],
        "substitute": [False, True],
        "minutesPlayed": [90.0, 8.0],
    })
    cands = la.roster_candidates(orc, lambda n: 0 if "man" in str(n).lower() else None)
    by_shirt = {c.shirt: c for c in cands}
    assert by_shirt[1].position == "G" and by_shirt[1].on_h1 and by_shirt[1].on_h2
    assert by_shirt[21].is_sub and not by_shirt[21].on_h1 and by_shirt[21].on_h2
