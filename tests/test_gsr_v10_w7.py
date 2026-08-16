"""Targeted checks for the v10-W7 association kill-experiments (ownership, merges, cut scoring)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.gsr_v10_w7 import (
    extract_cuts,
    merge_map,
    ownership,
    score_cuts,
    true_cuts,
)


def _rows() -> pd.DataFrame:
    """Two fragments of GT identity 7 (the second one 5/6 pure) plus an unauditable track."""
    return pd.DataFrame({"track_id": [1] * 6 + [2] * 6 + [3] * 3,
                         "frame": list(range(6)) + list(range(10, 16)) + [0, 1, 2],
                         "gt_id": [7] * 6 + [7, 7, 7, 7, 7, 9] + [-1, -1, -1]})


def test_ownership_trusts_only_audited_tracks() -> None:
    """A track with fewer than five audited rows gets no identity; purity is row-weighted."""
    own = ownership(_rows())
    assert list(own["gt_id"]) == [7, 7, -1]
    assert own.loc[1, "purity"] == 5 / 6
    assert own.loc[2, "n_audited"] == 0


def test_merge_map_grants_disjoint_and_blocks_overlap() -> None:
    """Same-identity fragments merge only when they share no frame (evaluator forbids stacking)."""
    own = ownership(_rows())
    new, st = merge_map(own)
    assert new == {1: 100007, 2: 100007}
    assert (st["merges_granted"], st["groups_merged"]) == (1, 1)
    blocked, bst = merge_map(own, {1: set(range(6)), 2: set(range(3, 9))})
    assert blocked == {} and bst["merges_blocked_overlap"] == 1


def test_true_cuts_ignores_flicker_at_min_run() -> None:
    """A one-row ownership blip is a change point only when no minimum run is required."""
    g = pd.DataFrame({"frame": range(11), "gt_id": [4] * 5 + [8] + [4] * 5})
    assert true_cuts(g) == [5, 6]
    assert true_cuts(g, min_run=5) == []
    real = pd.DataFrame({"frame": range(10), "gt_id": [4] * 5 + [8] * 5})
    assert true_cuts(real, min_run=5) == [5]


def test_cut_extraction_and_scoring_are_sequence_unique() -> None:
    """Track ids repeat across clips, so cuts must be keyed on the sequence-unique uid."""
    feat = pd.DataFrame({"uid": ["A:1"] * 4 + ["B:1"] * 2, "frame": [0, 3, 40, 60, 0, 3],
                         "purity": [0.5] * 4 + [1.0] * 2,
                         "cut_frames": [(3,)] * 4 + [()] * 2})
    cuts = extract_cuts(feat, np.array([0.1, 0.9, 0.8, 0.2, 0.1, 0.9]), 0.5)
    assert cuts == {"A:1": [3, 40], "B:1": [3]}
    s = score_cuts(feat, cuts)
    assert (s["tp"], s["fp"], s["true_cuts"]) == (1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 1.0
    assert s["pure_tracklets"] == 1 and s["false_cut_rate"] == 1.0
    assert (s["cuts_all"], s["cuts_all_on_a_true_change"]) == (3, 1)
