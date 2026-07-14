"""Tests for the C4 instinct engine (counterfactual run optimisation; pure, no GAT)."""

from __future__ import annotations

import numpy as np

from attacker.instinct import (
    candidate_targets,
    matched_actual_dir,
    move_player,
    offball_attacker_indices,
    optimal_run,
)
from generator.contract import FreezeFrame, PlayerNode, Substrate


def _frame():
    return FreezeFrame(players=[
        PlayerNode(60.0, 40.0, is_teammate=True, is_actor=True),    # 0: ball-carrier
        PlayerNode(50.0, 30.0, is_teammate=True),                   # 1: off-ball attacker
        PlayerNode(20.0, 40.0, is_teammate=True, is_keeper=True),   # 2: keeper (excluded)
        PlayerNode(70.0, 40.0, is_teammate=False),                  # 3: opponent (excluded)
    ], substrate=Substrate.BROADCAST_CV)


def test_candidate_targets_count_and_bounds():
    c = candidate_targets(60.0, 40.0, n_dirs=8, rings=(0.5, 1.0))
    assert len(c) == 16
    assert all(0.0 <= x <= 120.0 and 0.0 <= y <= 80.0 for x, y in c)


def test_offball_attacker_indices_excludes_actor_keeper_opponent():
    assert offball_attacker_indices(_frame()) == [1]


def test_move_player_relocates_only_that_node():
    f2 = move_player(_frame(), 1, 90.0, 20.0)
    assert (f2.players[1].x, f2.players[1].y) == (90.0, 20.0)
    assert f2.players[1].is_teammate and not f2.players[1].is_actor
    assert (f2.players[0].x, f2.players[0].y) == (60.0, 40.0)        # others unchanged


def test_optimal_run_points_toward_value_gradient():
    # player value rewards being further up-pitch (+x) -> optimal run is toward +x
    def pvalue(fr, idx):
        return fr.players[idx].x / 120.0

    o = optimal_run(pvalue, _frame(), 1)
    assert o["value_gain"] > 0
    assert o["opt_dir"][0] > 0.9                                     # mostly toward +x (goal)


def test_matched_actual_dir_follows_nearest_teammate():
    now = _frame()                                                  # attacker idx 1 at (50, 30)
    fut = move_player(now, 1, 60.0, 30.0)                            # actually ran +10 in x
    d = matched_actual_dir(now, fut, 1)
    assert d[0] > 0.9 and abs(d[1]) < 1e-6
