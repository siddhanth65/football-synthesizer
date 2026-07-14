"""Tests for possession-derived metrics: passes, passing networks, PPDA proxy."""

from __future__ import annotations

import pandas as pd

from fingerprint.possession_metrics import (
    extract_passes,
    network_metrics,
    passing_network,
    ppda,
)


def _players(rows: list[tuple]) -> pd.DataFrame:
    """rows: (frame, track_id, team, x, y)."""
    return pd.DataFrame(rows, columns=["frame", "track_id", "team", "pitch_x", "pitch_y"]).assign(
        role="player")


def _possession(rows: list[tuple]) -> pd.DataFrame:
    """rows: (frame, carrier, team)."""
    return pd.DataFrame(rows, columns=["frame", "carrier", "team"]).assign(dist_m=1.0)


def test_extract_passes_handoff_same_team_only():
    players = _players([
        (0, 1, 0, 20, 30), (5, 2, 0, 40, 30), (10, 5, 1, 60, 30), (200, 1, 0, 20, 30),
    ])
    poss = _possession([(0, 1, 0), (5, 2, 0), (10, 5, 1), (200, 1, 0)])
    passes = extract_passes(poss, players)
    # 1->2 (same team, gap 5) is a pass; 2->5 is a turnover; 5->1 gap 190 > max → dropped.
    assert len(passes) == 1
    r = passes.iloc[0]
    assert (int(r.from_id), int(r.to_id), int(r.team)) == (1, 2, 0)
    assert r.length_m == 20.0


def test_extract_passes_respects_max_gap():
    players = _players([(0, 1, 0, 0, 0), (80, 2, 0, 3, 4)])
    poss = _possession([(0, 1, 0), (80, 2, 0)])
    assert extract_passes(poss, players, max_gap_frames=50).empty
    assert len(extract_passes(poss, players, max_gap_frames=100)) == 1


def test_extract_passes_seconds_window_is_fps_consistent():
    # A 30-native-frame carrier gap: 1.2 s at 25 fps but 0.5 s at 59.94 fps. A 0.9 s window must ADMIT
    # it at 25 fps (30 <= 0.9*25=22.5? no -> excluded) ... verify the frame-count derivation directly.
    players = _players([(0, 1, 0, 0, 0), (30, 2, 0, 3, 4)])
    poss = _possession([(0, 1, 0), (30, 2, 0)])
    # 0.9 s @ 25 fps -> round(22.5) = 22 frames < 30 -> excluded.
    assert extract_passes(poss, players, max_gap_s=0.9, fps=25.0).empty
    # 0.9 s @ 59.94 fps -> round(53.9) = 54 frames >= 30 -> a pass.
    assert len(extract_passes(poss, players, max_gap_s=0.9, fps=59.94)) == 1
    # Same physical window (1.4 s) admits the 30-frame gap at both fps: 1.4*25=35, 1.4*59.94=84.
    assert len(extract_passes(poss, players, max_gap_s=1.4, fps=25.0)) == 1
    assert len(extract_passes(poss, players, max_gap_s=1.4, fps=59.94)) == 1


def test_extract_passes_seconds_requires_fps():
    players = _players([(0, 1, 0, 0, 0), (5, 2, 0, 3, 4)])
    poss = _possession([(0, 1, 0), (5, 2, 0)])
    for bad in (None, 0.0, -5.0):
        try:
            extract_passes(poss, players, max_gap_s=0.9, fps=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for fps={bad}")


def test_passing_network_nodes_and_edges():
    players = _players([
        (0, 1, 0, 20, 30), (5, 2, 0, 40, 30), (8, 1, 0, 25, 30), (12, 3, 0, 30, 50),
    ])
    poss = _possession([(0, 1, 0), (5, 2, 0), (8, 1, 0), (12, 3, 0)])
    passes = extract_passes(poss, players)
    nodes, edges = passing_network(passes, players, team=0)
    assert set(nodes["track_id"]) == {1, 2, 3}
    assert int(edges["count"].sum()) == 3
    m = network_metrics(nodes, edges)
    assert m["n_players"] == 3 and m["n_passes"] == 3
    assert m["top_connector"] == 1  # player 1 is in every pass


def test_ppda_counts_zone_passes_and_pressures():
    # team 0 attacks toward x=105; its build-up zone is x <= 63. Two passes there, one pressured frame.
    players = _players([
        (0, 1, 0, 20, 30), (0, 9, 1, 21, 30),     # defender 9 within 3m of carrier 1 → a pressure
        (5, 2, 0, 40, 30), (5, 9, 1, 80, 30),      # defender far → no pressure
        (10, 1, 0, 90, 30),                        # carrier in attacking third (x=90>63) → out of zone
    ])
    poss = _possession([(0, 1, 0), (5, 2, 0), (10, 1, 0)])
    out = ppda(poss, players, directions={0: 1, 1: -1})
    row = out[out["team"] == 1].iloc[0]  # team 1 presses team 0
    # two zone hand-offs originate at x<=63: 1->2 (x0=20) and 2->1 (x0=40).
    assert row.passes_allowed == 2
    assert row.pressures == 1        # frame 0 carrier pressured; frame 5 not; frame 10 out of zone
    assert row.ppda == 2.0
