"""Tests for the pure Stage-2 identity solve (no CV model, no metric stack).

Thin on purpose: :func:`generator.identity_solve._demo` already asserts the full synthetic scenario,
so this pins it in CI plus the three behaviours the gates actually depend on -- mutual exclusion
beating a stronger unary score, the abstain class winning when there is no evidence, and the
exclusion groups being exact (built from shared frames, not from interval envelopes).
"""

from __future__ import annotations

import numpy as np

from generator.identity_solve import (
    Identity,
    SolverConfig,
    Tracklet,
    _demo,
    digit_confusion_prior,
    exclusion_groups,
    solve_assignment,
    solve_sequence,
)


def _trk(tid: int, lo: int, hi: int, emb: np.ndarray, reads=()) -> Tracklet:
    return Tracklet(tid, hi - lo + 1, 0, 1.0, {"player": 1.0}, tuple(reads), emb, (lo, hi))


def test_demo_selfcheck_passes():
    _demo()


def test_mutex_forces_the_second_best_tracklet_off_a_contested_identity():
    """Three tracklets, two identities: two overlapping tracklets both prefer #7 on their own."""
    a, b = np.eye(4)[0], np.eye(4)[1]
    cfg = SolverConfig(confusion=digit_confusion_prior([(7, 7)]), app_gain=30.0, sim_none=0.80,
                       topk=1, pi_none=0.5)
    ids = [Identity(("left", 7), 0, 7), Identity(("left", 9), 0, 9)]
    gal = [a[None, :], b[None, :]]
    tracklets = [_trk(0, 0, 10, a[None, :]),
                 _trk(1, 5, 15, (0.99 * a + 0.14 * b)[None, :]),
                 _trk(2, 20, 30, b[None, :])]
    alive = [[0]] * 5 + [[0, 1]] * 6 + [[1]] * 5 + [[2]] * 11
    assign, probs = solve_sequence(tracklets, ids, gal, alive, cfg)
    assert probs[0].argmax() == 0 and probs[1].argmax() == 0  # unconstrained: both want #7
    assert assign[0] is ids[0]
    assert assign[1] is not ids[0]
    assert assign[2] is ids[1]


def test_exclusion_groups_use_shared_frames_not_interval_envelopes():
    """Tracklets 0 and 1 never share a frame even though 2's envelope spans both."""
    alive = [[0, 2], [0, 2], [2], [1, 2], [1, 2]]
    assert exclusion_groups(alive) == [(0, 2), (1, 2)]
    assert exclusion_groups([[0], [1]]) == []          # no co-occurrence -> no constraint


def test_solver_abstains_when_the_posterior_is_flat():
    probs = np.array([[0.02, 0.02, 0.96], [0.02, 0.02, 0.96]])
    pick = solve_assignment(probs, np.array([100.0, 100.0]), [(0, 1)], [0, 0])
    assert list(pick) == [-1, -1]


def test_squad_size_constraint_caps_concurrent_names():
    """Four co-occurring tracklets, four identities, but only two may be named at once."""
    probs = np.array([[0.9, 0.0, 0.0, 0.0, 0.1],
                      [0.0, 0.9, 0.0, 0.0, 0.1],
                      [0.0, 0.0, 0.9, 0.0, 0.1],
                      [0.0, 0.0, 0.0, 0.9, 0.1]])
    pick = solve_assignment(probs, np.ones(4), [(0, 1, 2, 3)], [0, 0, 0, 0], max_concurrent=2)
    assert sum(p >= 0 for p in pick) == 2
