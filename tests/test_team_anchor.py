"""Tests for cross-chunk jersey-colour team anchoring (pure parts; no video)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.team_anchor import (
    apply_chunkwise_team_labels,
    apply_global_team_labels,
    estimate_player_box,
    global_team_map,
)


def test_estimate_player_box_scales_with_depth_and_clips():
    near = estimate_player_box(640, 700, frame_h=720, frame_w=1280)   # bottom of frame -> tall
    far = estimate_player_box(640, 120, frame_h=720, frame_w=1280)    # top of frame -> short
    near_h, far_h = near[3] - near[1], far[3] - far[1]
    assert near_h > far_h > 0
    assert near[0] >= 0 and near[2] <= 1280 and near[3] <= 720      # clipped to frame


def test_apply_global_team_labels_is_consistent_across_chunks():
    # two chunks with the per-chunk team labels SWAPPED; colours: red (high a*) vs blue (low a*,b*)
    red, blue = np.array([50.0, 60.0, 40.0]), np.array([50.0, -20.0, -40.0])
    rows = [
        {"chunk": "A", "role": "player", "team": 0, "track_id": 1},  # red
        {"chunk": "A", "role": "player", "team": 1, "track_id": 2},  # blue
        {"chunk": "B", "role": "player", "team": 1, "track_id": 1},  # red (label swapped vs A)
        {"chunk": "B", "role": "player", "team": 0, "track_id": 2},  # blue
        {"chunk": "A", "role": "referee", "team": 0, "track_id": 9},  # -> -1
    ]
    colors = {("A", 1): red, ("A", 2): blue, ("B", 1): red, ("B", 2): blue}
    out = apply_global_team_labels(pd.DataFrame(rows), colors)
    team = {(r.chunk, r.track_id): r.team for r in out.itertuples(index=False) if r.role == "player"}
    assert team[("A", 1)] == team[("B", 1)]          # both red -> same global team
    assert team[("A", 2)] == team[("B", 2)]          # both blue -> same global team
    assert team[("A", 1)] != team[("A", 2)]          # red != blue
    assert int(out[out["role"] == "referee"]["team"].iloc[0]) == -1


def test_chunkwise_anchoring_survives_a_noisy_crossover_track():
    # Within each chunk the two kits separate cleanly; one chunk has a single noisy track whose colour
    # leans toward the other team. Pooling could mislabel the chunk; chunkwise matching should not.
    red, blue = np.array([50.0, 60.0, 40.0]), np.array([50.0, -20.0, -40.0])
    noisy_red = np.array([50.0, 35.0, 20.0])   # a red track contaminated toward neutral
    rows = []
    colors = {}
    for ck in ("A", "B", "C"):
        for tid, base in [(1, red), (2, red), (3, blue), (4, blue)]:
            rows.append({"chunk": ck, "role": "player", "team": 0, "track_id": tid})
            colors[(ck, tid)] = base
    colors[("B", 2)] = noisy_red    # one noisy red in chunk B
    out = apply_chunkwise_team_labels(pd.DataFrame(rows), colors)
    team = {(r.chunk, r.track_id): r.team for r in out.itertuples(index=False)}
    for ck in ("A", "B", "C"):
        assert team[(ck, 1)] == team[(ck, 2)]        # both red -> same team within chunk
        assert team[(ck, 3)] == team[(ck, 4)]        # both blue -> same team
        assert team[(ck, 1)] != team[(ck, 3)]        # red != blue
    # global consistency: red is the same global id in every chunk
    assert team[("A", 1)] == team[("B", 1)] == team[("C", 1)]


def test_global_team_map_anchors_dark_kit_to_team0_across_chunks():
    navy, white = np.array([25.0, 4.0, -14.0]), np.array([68.0, 4.0, -1.0])
    # per-(chunk, per-chunk-team) centroids, with the per-chunk numbering SWAPPED between chunks
    cent = {
        ("A", 0): navy, ("A", 1): white,    # chunk A: 0=navy
        ("B", 1): navy, ("B", 0): white,    # chunk B: 1=navy (swapped)
    }
    m = global_team_map(cent, dark_is_team0=True)
    assert m[("A", 0)] == 0 and m[("B", 1)] == 0      # navy -> global 0 (France) in both
    assert m[("A", 1)] == 1 and m[("B", 0)] == 1      # white -> global 1 (Senegal) in both
