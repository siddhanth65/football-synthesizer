"""Tests for the defensive block-geometry classifier + the FBref style-factor loader (no video/GPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.block_height import (
    HIGH_BLOCK_MIN_M,
    LOW_BLOCK_MAX_M,
    MIN_SPELL_FRAMES,
    _classify,
    block_spells,
)


def test_classify_thresholds():
    assert _classify(LOW_BLOCK_MAX_M - 1) == "low"
    assert _classify(LOW_BLOCK_MAX_M) == "mid"          # boundary belongs to mid
    assert _classify((LOW_BLOCK_MAX_M + HIGH_BLOCK_MIN_M) / 2) == "mid"
    assert _classify(HIGH_BLOCK_MIN_M) == "high"        # boundary belongs to high
    assert _classify(HIGH_BLOCK_MIN_M + 10) == "high"


def test_block_spells_segments_on_team_change_and_ignores_short():
    # team 0 defends a low spell, flips to team 1 (high), then a too-short team-0 spell (dropped).
    n = MIN_SPELL_FRAMES + 2
    rows = []
    for i in range(n):                       # long low spell, team 0
        rows.append({"chunk": "c", "frame": i, "def_team": 0, "line_m": 20.0,
                     "vspread_m": 5.0, "ball_adv_m": 40.0, "ball_to_block_m": 20.0, "usable": True})
    for i in range(n):                       # long high spell, team 1
        rows.append({"chunk": "c", "frame": 100 + i, "def_team": 1, "line_m": 60.0,
                     "vspread_m": 5.0, "ball_adv_m": 40.0, "ball_to_block_m": -20.0, "usable": True})
    for i in range(2):                       # short team-0 spell -> ignored
        rows.append({"chunk": "c", "frame": 200 + i, "def_team": 0, "line_m": 20.0,
                     "vspread_m": 5.0, "ball_adv_m": 40.0, "ball_to_block_m": 20.0, "usable": True})
    sp = block_spells(pd.DataFrame(rows))
    assert len(sp) == 2, sp
    assert sp.iloc[0]["block_class"] == "low" and sp.iloc[0]["def_team"] == 0
    assert sp.iloc[1]["block_class"] == "high" and sp.iloc[1]["def_team"] == 1


def test_block_spells_needs_usable_frames():
    # a full spell but no usable geometry -> no classified spell.
    rows = [{"chunk": "c", "frame": i, "def_team": 0, "line_m": np.nan, "vspread_m": np.nan,
             "ball_adv_m": 40.0, "ball_to_block_m": np.nan, "usable": False}
            for i in range(MIN_SPELL_FRAMES + 3)]
    assert block_spells(pd.DataFrame(rows)).empty
