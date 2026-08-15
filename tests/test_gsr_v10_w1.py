"""Targeted checks for the v10-W1 VLM jersey-reading harness (parser + scorers)."""

from __future__ import annotations

import pandas as pd

from tools.gsr_v10_w1 import (
    parse_read,
    score_tier_a,
    score_tier_b_percrop,
    score_tier_b_tracklet,
)


def test_parse_read_formats() -> None:
    """Numbers are taken only from 1-99; refusals and out-of-range replies are abstains."""
    assert parse_read('{"number": 23}') == 23
    assert parse_read('{"number": "7"}') == 7
    assert parse_read('{"number": null}') is None
    assert parse_read("The number is not legible in any view.") is None
    assert parse_read('{"number": 0}') is None
    assert parse_read('{"number": 123}') is None
    assert parse_read("36") == 36
    assert parse_read("") is None


def _queue() -> pd.DataFrame:
    return pd.DataFrame([
        {"queue_id": "A1", "tier": "A", "label": "29", "note": None, "gt_number": None,
         "cells": ["1", "2"]},
        {"queue_id": "A2", "tier": "A", "label": "30", "note": "u", "gt_number": None,
         "cells": ["1", "2"]},
        {"queue_id": "A3", "tier": "A", "label": "none", "note": None, "gt_number": None,
         "cells": ["1", "2"]},
        {"queue_id": "A4", "tier": "A", "label": "none", "note": None, "gt_number": None,
         "cells": ["1", "2"]},
        {"queue_id": "B1", "tier": "B", "label": "1", "note": None, "gt_number": 8,
         "cells": ["1", "2"]},
    ])


def test_score_tier_a_note_exclusion_and_rates() -> None:
    """The stray-``u`` row leaves the 13-number pool but returns for the 15-number read."""
    reads = pd.DataFrame([{"key": "A1", "read": 29}, {"key": "A2", "read": 30},
                          {"key": "A3", "read": 7}, {"key": "A4", "read": None}])
    reads["read"] = reads["read"].astype("Int64")
    strict = score_tier_a(_queue(), reads, drop_noted=True)
    assert strict["n_numbered"] == 1 and strict["correct"] == 1 and strict["precision"] == 1.0
    assert strict["n_none"] == 2 and strict["fp_on_none"] == 1 and strict["fp_rate"] == 0.5
    loose = score_tier_a(_queue(), reads, drop_noted=False)
    assert loose["n_numbered"] == 2 and loose["correct"] == 2


def test_score_tier_b_tracklet_and_percrop() -> None:
    """Tier B scores against SoccerNet's own number; per-crop splits on Sid's visibility cells."""
    trk = pd.DataFrame([{"key": "B1", "read": 8}])
    trk["read"] = trk["read"].astype("Int64")
    assert score_tier_b_tracklet(_queue(), trk)["correct"] == 1

    percrop = pd.DataFrame([{"key": "B1", "cell": 1, "read": 8}, {"key": "B1", "cell": 2, "read": 5}])
    percrop["read"] = percrop["read"].astype("Int64")
    got = score_tier_b_percrop(_queue(), percrop)
    assert got["visible"] == 1 and got["vis_correct"] == 1  # cell 1 is the human-visible one
    assert got["hidden"] == 1 and got["hid_emitted"] == 1 and got["hid_fp_rate"] == 1.0
