"""Checks for the Lucey entropy-map probe: gap policy, direction normalisation, entropy, windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID
from tools.entropy_map_probe import cell_ids, entropy_map, possession_strings, segments


def _seq(ts, teams, *, dirs=1, xs=None) -> pd.DataFrame:
    """One-chunk 1 Hz sequence with the columns ``possession_strings`` consumes."""
    return pd.DataFrame({
        "match": "m", "chunk": "c", "t": list(ts), "x": xs if xs is not None else [10.0] * len(ts),
        "y": [34.0] * len(ts), "team": list(teams),
        "dir": [dirs] * len(ts), "observed": [True] * len(ts),
    })


def test_gap_policy_bridges_only_up_to_G():
    seq = _seq([0, 1, 3, 4], [0, 0, 0, 0])            # one missing second at t=2
    assert [s.x.size for s in possession_strings(seq, gap_s=0)] == [2, 2]
    (one,) = possession_strings(seq, gap_s=1)
    assert one.x.size == 5                            # t=0..4 inclusive, the hole now carries a value
    assert one.probe_filled.tolist() == [False, False, True, False, False]  # t=2 was invented


def test_other_team_label_breaks_a_string_even_inside_the_gap():
    seq = _seq([0, 1, 2, 3], [0, 0, 1, 0])
    lens = sorted(s.x.size for s in possession_strings(seq, gap_s=5) if s.team_id == 0)
    assert lens == [2]                                 # the t=2 turnover splits it, no bridge


def test_direction_normalisation_is_a_180_degree_rotation():
    seq = _seq([0, 1], [0, 0], dirs=-1, xs=[20.0, 30.0])
    (s,) = possession_strings(seq, gap_s=0)
    assert np.allclose(s.x, [PITCH_LEN - 20.0, PITCH_LEN - 30.0])
    assert np.allclose(s.y, PITCH_WID - 34.0)


def test_segment_count_is_T1_minus_T_plus_1_and_shorter_strings_are_dropped():
    seq = _seq(range(7), [0] * 7)
    st = possession_strings(seq, gap_s=0)
    start, end = segments(st, T=3, grid=(4, 3))
    assert start.size == 7 - 3 + 1
    assert segments(st, T=8, grid=(4, 3))[0].size == 0


def test_entropy_is_log2_of_a_uniform_destination_set_and_NaN_below_min_n():
    start = np.zeros(8, int)
    end = np.array([0, 1, 2, 3] * 2)
    ent, counts = entropy_map(start, end, grid=(4, 3), min_n=5)
    assert counts[0] == 8
    assert abs(ent[0] - 2.0) < 1e-9                    # 4 equiprobable destinations = 2 bits
    assert np.isnan(entropy_map(start[:4], end[:4], grid=(4, 3), min_n=5)[0][0])


def test_cell_ids_clip_to_the_pitch():
    c = cell_ids(np.array([-5.0, 200.0]), np.array([-1.0, 99.0]), (4, 3))
    assert c.tolist() == [0, 4 * 3 - 1]
