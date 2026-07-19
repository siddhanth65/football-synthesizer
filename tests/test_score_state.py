"""Tests for the score-state segmentation seam (pure; no video / no match data)."""

from __future__ import annotations

import pandas as pd

from fingerprint.score_state import STOPPAGE_S, annotate, score_state_at


def test_liverpool_state_transitions():
    # Level until the 34:26 (2066 s) opener, chasing after; H1 goals carry into H2.
    assert score_state_at("manutd_liverpool", "h1", 2000.0)["state"] == "level"
    s = score_state_at("manutd_liverpool", "h1", 2100.0)
    assert (s["state"], s["scoreline"]) == ("chasing", "0-1")
    assert score_state_at("manutd_liverpool", "h1", 2600.0)["scoreline"] == "0-2"
    assert score_state_at("manutd_liverpool", "h2", 100.0)["scoreline"] == "0-2"
    assert score_state_at("manutd_liverpool", "h2", 800.0)["scoreline"] == "0-3"


def test_fulham_level_then_leading():
    assert score_state_at("manutd_fulham", "h2", 2000.0)["state"] == "level"
    lead = score_state_at("manutd_fulham", "h2", 2600.0)
    assert (lead["state"], lead["scoreline"]) == ("leading", "1-0")
    # No H1 goals: whole first half is level.
    assert score_state_at("manutd_fulham", "h1", 1500.0)["state"] == "level"


def test_brighton_equaliser_then_late_loss():
    assert score_state_at("brighton_manutd", "h1", 100.0)["scoreline"] == "0-0"
    assert score_state_at("brighton_manutd", "h2", 500.0)["state"] == "chasing"   # 0-1
    assert score_state_at("brighton_manutd", "h2", 1000.0)["state"] == "level"    # 1-1 equaliser
    end = score_state_at("brighton_manutd", "h2", 2900.0)
    assert (end["state"], end["scoreline"], end["stoppage"]) == ("chasing", "1-2", True)


def test_stoppage_flag_boundary():
    assert score_state_at("manutd_fulham", "h1", STOPPAGE_S - 1)["stoppage"] is False
    assert score_state_at("manutd_fulham", "h1", STOPPAGE_S + 1)["stoppage"] is True


def test_annotate_uses_chunk_offsets_and_grid_fps():
    # frame 2500 at 25 fps = 100 s into a chunk that starts at 2000 s -> t_s 2100 -> chasing 0-1.
    df = pd.DataFrame({"chunk": ["h1_chunk_003", "h1_chunk_003"], "frame": [1250, 2500]})
    out = annotate("manutd_liverpool", df, offsets={"h1_chunk_003": 2000.0})
    assert list(out["t_s"]) == [2050.0, 2100.0]
    assert list(out["state"]) == ["level", "chasing"]        # 2050 s < opener, 2100 s > opener
    assert list(out["scoreline"]) == ["0-0", "0-1"]
