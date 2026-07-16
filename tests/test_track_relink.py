"""Tests for the pure track re-link merge logic (no CV model, no metric stack).

These pin the fragment-merge constraints (temporal disjointness, team/role match, gap-consistent
motion) and the greedy agglomerative merge + GT-based merge-precision accounting on synthetic
fragments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.track_relink import (
    Fragment,
    frags_mergeable,
    greedy_merge,
    merge_precision,
    pair_similarities,
    summarize_fragments,
)


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, np.float32)
    return a / np.linalg.norm(a)


def test_frags_mergeable_gates_overlap_team_role_and_motion():
    a = Fragment(1, 0, 10, (10.0, 10.0), (12.0, 10.0), "player", 0, 11)
    reachable = Fragment(2, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 0, 11)  # 0.5 m / 5 frames
    overlap = Fragment(3, 5, 20, (12.5, 10.0), (14.0, 10.0), "player", 0, 11)
    other_team = Fragment(4, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 1, 11)
    other_role = Fragment(5, 15, 25, (12.5, 10.0), (14.0, 10.0), "goalkeeper", 0, 11)
    too_far = Fragment(6, 15, 25, (40.0, 10.0), (41.0, 10.0), "player", 0, 11)  # 28 m / 5 frames
    assert frags_mergeable(a, reachable)
    assert frags_mergeable(reachable, a)  # order-independent
    assert not frags_mergeable(a, overlap)
    assert not frags_mergeable(a, other_team)
    assert not frags_mergeable(a, other_role)
    assert not frags_mergeable(a, too_far)


def test_pair_similarities_only_scores_constraint_valid_pairs():
    a = Fragment(1, 0, 10, (10.0, 10.0), (12.0, 10.0), "player", 0, 11)
    b = Fragment(2, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 0, 11)
    c = Fragment(3, 5, 12, (50.0, 30.0), (52.0, 30.0), "player", 1, 8)  # overlaps a, other team
    emb = {1: _unit([1.0, 0.0]), 2: _unit([0.99, 0.14]), 3: _unit([0.0, 1.0])}
    sims = pair_similarities([a, b, c], emb)
    assert set(sims) == {(0, 1)}
    assert sims[(0, 1)] > 0.98


def test_greedy_merge_chains_disjoint_fragments_and_respects_threshold():
    # Three consecutive fragments of one player; a->b->c each reachable, all near-identical.
    a = Fragment(1, 0, 10, (10.0, 10.0), (11.0, 10.0), "player", 0, 11)
    b = Fragment(2, 15, 25, (11.5, 10.0), (12.5, 10.0), "player", 0, 11)
    c = Fragment(3, 30, 40, (13.0, 10.0), (14.0, 10.0), "player", 0, 11)
    frags = [a, b, c]
    emb = {1: _unit([1.0, 0.02]), 2: _unit([1.0, 0.0]), 3: _unit([0.99, 0.03])}
    sims = pair_similarities(frags, emb)
    remap = greedy_merge(frags, sims, threshold=0.9)
    assert remap[1] == remap[2] == remap[3] == 1  # all chain into the smallest id
    # A high threshold blocks every merge -> identity map.
    remap_hi = greedy_merge(frags, sims, threshold=0.999999)
    assert remap_hi == {1: 1, 2: 2, 3: 3}


def test_greedy_merge_blocks_when_intermediate_makes_span_overlap():
    # a (0-10) and c (5-15) overlap; even if appearance says merge, the temporal chain forbids it.
    a = Fragment(1, 0, 10, (10.0, 10.0), (10.0, 10.0), "player", 0, 11)
    c = Fragment(3, 5, 15, (10.0, 10.0), (10.0, 10.0), "player", 0, 11)
    emb = {1: _unit([1.0, 0.0]), 3: _unit([1.0, 0.0])}
    sims = pair_similarities([a, c], emb)
    assert sims == {}  # overlap -> never even a candidate
    assert greedy_merge([a, c], sims, threshold=0.5) == {1: 1, 3: 3}


def test_summarize_fragments_drops_nan_only_and_uses_finite_boundaries():
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "role": "player", "team": 0, "pitch_x": np.nan,
         "pitch_y": np.nan, "image_x": 1.0, "image_y": 2.0},
        {"frame": 1, "track_id": 1, "role": "player", "team": 0, "pitch_x": 20.0,
         "pitch_y": 30.0, "image_x": 1.0, "image_y": 2.0},
        {"frame": 2, "track_id": 1, "role": "player", "team": 0, "pitch_x": 22.0,
         "pitch_y": 30.0, "image_x": 1.0, "image_y": 2.0},
        {"frame": 0, "track_id": 9, "role": "player", "team": 1, "pitch_x": np.nan,
         "pitch_y": np.nan, "image_x": 1.0, "image_y": 2.0},  # NaN-only -> dropped
        {"frame": 0, "track_id": -1, "role": "ball", "team": -1, "pitch_x": 5.0,
         "pitch_y": 5.0, "image_x": 1.0, "image_y": 2.0},  # ball -> dropped
    ])
    frags = summarize_fragments(df)
    assert len(frags) == 1
    f = frags[0]
    assert f.track_id == 1 and f.start_frame == 0 and f.end_frame == 2
    assert f.start_xy == (20.0, 30.0) and f.end_xy == (22.0, 30.0)  # finite-pitch boundaries


def test_merge_precision_counts_only_auditable_pairs():
    # remap groups {1,2,3} and {4,5}; GT ids: 1,2 same (100); 3 different (200); 4,5 same (300).
    remap = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4}
    gt_ids = {1: 100, 2: 100, 3: 200, 4: 300, 5: 300}
    correct, total = merge_precision(remap, gt_ids)
    # group {1,2,3}: pairs (1,2)=ok (1,3)=bad (2,3)=bad; group {4,5}: (4,5)=ok -> 2/4
    assert (correct, total) == (2, 4)
    # A fragment with no GT id makes its pairs unauditable (excluded from totals).
    correct2, total2 = merge_precision(remap, {1: 100, 2: 100, 4: 300, 5: 300})
    assert (correct2, total2) == (2, 2)  # the two (1/2,3) pairs drop out
