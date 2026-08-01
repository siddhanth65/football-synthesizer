"""Tests for the FOOTPASS VAL attribution scorer and the prep seams (CPU, no data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval import footpass_score as fs
from tools import footpass_prep as fp


def test_selftest_fixture() -> None:
    """The module's own deterministic mini-fixture (arithmetic of the whole metric)."""
    fs._selftest()


def test_unanswered_events_still_cost_coverage() -> None:
    """An event with no prediction inside the tolerance stays in the denominator."""
    ev = pd.DataFrame({"game": ["g"] * 4, "half": ["h1"] * 4, "frame": [0, 100, 200, 300],
                       "team": [1, 1, 2, 2], "shirt": [7, 7, 9, 9], "role_id": [10] * 4,
                       "action": [2] * 4, "on_screen": [True] * 4})
    pr = pd.DataFrame({"game": ["g"], "frame": [0], "pred_team": [0], "pred_shirt": [7],
                       "conf": [0.99]})
    res = fs.score(ev, pr, tol=2)
    assert res["overall"]["n_events"] == 4
    assert res["overall"]["n_answered"] == 1
    assert res["overall"]["coverage_at_0.85"] == pytest.approx(0.25)


def test_no_predictions_scores_zero_not_crash() -> None:
    """An empty prediction table is a legal (0-coverage) result, not an exception."""
    ev = pd.DataFrame({"game": ["g"], "half": ["h1"], "frame": [10], "team": [1], "shirt": [7],
                       "role_id": [10], "action": [2], "on_screen": [False]})
    res = fs.score(ev, pd.DataFrame(columns=["game", "frame", "pred_team", "pred_shirt", "conf"]))
    assert res["overall"]["n_answered"] == 0
    assert res["overall"]["coverage_at_0.85"] == 0.0
    assert res["on_screen"]["n_events"] == 0 and res["off_screen"]["n_events"] == 1


def test_on_screen_from_visibility_runs() -> None:
    """``_on_screen`` reads the digest's inclusive run intervals correctly."""
    runs = np.array([[0, 10, 20], [0, 40, 50], [1, 0, 5]], np.int64)
    got = fs._on_screen(np.array([9, 10, 20, 21, 45, 3]), np.array([0, 0, 0, 0, 0, 1]), runs)
    assert got.tolist() == [False, True, True, False, True, True]


def test_align_offset_finds_a_one_frame_peak() -> None:
    """The alignment proof recovers a known shift exactly, with a clean runner-up gap."""
    rng = np.random.default_rng(0)
    truth = np.unique(rng.integers(0, 50_000, 200))
    res = fp.align_offset(truth - 7, truth, search=30)
    assert res["best_offset"] == 7
    assert res["n_matched"] == truth.size
    assert res["runner_up"] < 5


def test_align_offset_reports_no_peak_when_unrelated() -> None:
    """Unrelated cut trains must not produce a confident offset."""
    rng = np.random.default_rng(1)
    a = np.unique(rng.integers(0, 50_000, 200))
    b = np.unique(rng.integers(0, 50_000, 200))
    res = fp.align_offset(a, b, search=30)
    assert res["n_matched"] <= 5
