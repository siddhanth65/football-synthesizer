"""Unit tests for the live-play / broadcast-content filter (:mod:`generator.live_play`).

Covers the pure classifier on hand-built synthetic rows (every branch + boundary) and the pandas
helpers on tiny synthetic dense tables, including the zero-detection -> ``graphic`` recovery via the
sampling grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator.live_play import (
    CLOSEUP_MAX_DET,
    LIVE_MIN_DET,
    LIVE_MIN_XSPREAD,
    SHOT_CLOSE_UP,
    SHOT_GRAPHIC,
    SHOT_LIVE_WIDE,
    SHOT_REPLAY_OTHER,
    SHOT_TYPES,
    class_fractions,
    classify_dense,
    classify_frame,
    per_frame_signals,
)


# === pure classifier =============================================================================
def test_live_wide_needs_both_count_and_spread() -> None:
    assert classify_frame(11, 1500.0) == SHOT_LIVE_WIDE
    # enough players but clustered (narrow span) is not a wide tactical view
    assert classify_frame(11, 200.0) == SHOT_REPLAY_OTHER


def test_close_up_is_few_bodies() -> None:
    assert classify_frame(1, 50.0) == SHOT_CLOSE_UP
    assert classify_frame(CLOSEUP_MAX_DET, 900.0) == SHOT_CLOSE_UP  # count dominates spread


def test_replay_other_is_the_medium_band() -> None:
    # 5-7 bodies: too many for a close-up, too few / too narrow for wide play
    assert classify_frame(6, 1500.0) == SHOT_REPLAY_OTHER
    assert classify_frame(6, 100.0) == SHOT_REPLAY_OTHER


def test_graphic_on_absent_or_empty() -> None:
    assert classify_frame(0, 0.0) == SHOT_GRAPHIC
    assert classify_frame(11, 1500.0, present=False) == SHOT_GRAPHIC


def test_boundary_thresholds_are_inclusive() -> None:
    # exactly at the live-wide gate qualifies
    assert classify_frame(LIVE_MIN_DET, LIVE_MIN_XSPREAD) == SHOT_LIVE_WIDE
    # one below either sub-threshold falls out of live_wide
    assert classify_frame(LIVE_MIN_DET - 1, LIVE_MIN_XSPREAD) == SHOT_REPLAY_OTHER
    assert classify_frame(LIVE_MIN_DET, LIVE_MIN_XSPREAD - 1) == SHOT_REPLAY_OTHER


# === pandas helpers ==============================================================================
def _dense_frame(frame: int, n: int, xs: list[float], calib: float = 0.2) -> pd.DataFrame:
    """A synthetic dense block: ``n`` players at frame ``frame`` spanning image_x values ``xs``."""
    return pd.DataFrame({
        "frame": frame,
        "track_id": list(range(n)),
        "role": "player",
        "image_x": xs,
        "image_y": np.linspace(700, 900, n),
        "pitch_x": np.linspace(10, 90, n),
        "pitch_y": np.linspace(10, 60, n),
        "calib_error_m": calib,
    })


def test_per_frame_signals_counts_and_spread() -> None:
    dense = pd.concat([
        _dense_frame(100, 10, list(np.linspace(200, 1700, 10))),   # wide
        _dense_frame(105, 2, [900.0, 950.0]),                      # close-up
    ], ignore_index=True)
    sig = per_frame_signals(dense)
    assert sig.loc[100, "n_det"] == 10
    assert sig.loc[100, "x_spread"] == 1500.0
    assert bool(sig.loc[100, "ge6"]) is True
    assert sig.loc[105, "n_det"] == 2
    assert bool(sig.loc[105, "ge6"]) is False


def test_classify_dense_recovers_graphic_from_grid() -> None:
    dense = pd.concat([
        _dense_frame(100, 11, list(np.linspace(200, 1700, 11))),   # live_wide
        _dense_frame(105, 3, [880.0, 900.0, 920.0]),               # close_up
    ], ignore_index=True)
    grid = pd.Index([100, 105, 110, 115], name="frame")  # 110,115 absent -> graphic
    tab = classify_dense(dense, grid)
    assert tab.loc[100, "shot_type"] == SHOT_LIVE_WIDE
    assert tab.loc[105, "shot_type"] == SHOT_CLOSE_UP
    assert tab.loc[110, "shot_type"] == SHOT_GRAPHIC
    assert tab.loc[115, "shot_type"] == SHOT_GRAPHIC
    assert tab.loc[110, "n_det"] == 0


def test_class_fractions_sum_to_one_and_cover_all_types() -> None:
    labels = pd.Series([SHOT_LIVE_WIDE, SHOT_LIVE_WIDE, SHOT_CLOSE_UP, SHOT_GRAPHIC])
    fr = class_fractions(labels)
    assert set(fr) == set(SHOT_TYPES)
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert fr[SHOT_LIVE_WIDE] == 0.5
    assert fr[SHOT_REPLAY_OTHER] == 0.0
