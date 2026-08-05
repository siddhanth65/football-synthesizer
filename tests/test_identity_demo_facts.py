"""Render-time fact extraction for the identity demo must match the persisted artifacts.

Guards the regex/JSON parsing in :func:`tools.make_identity_demo.identity_facts` -- if an artifact
schema drifts (a renamed column, a reworded MD table), these assertions fail loudly rather than
letting the demo render a wrong or missing on-screen number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import make_identity_demo as demo  # noqa: E402

_ARTIFACTS = demo.NAMED_TRACKS.exists() and demo.ORACLE.exists()
_SURVIVORS = demo.ANCHOR_ROOT / "spotcheck_step3" / "_survivors"


@pytest.mark.skipif(not _ARTIFACTS, reason="identity artifacts not present")
def test_identity_facts_match_artifacts() -> None:
    """Every headline figure is read straight from disk, not hardcoded."""
    if not any(_SURVIVORS.glob("*.jpg")):
        pytest.skip(
            "_survivors crops purged by results/STORAGE_RECLAIM_LOG_2026-08-01.md row 2 "
            "(Tier-A cleanup) -- hero_number/hero_share cannot be recomputed without "
            "re-running the closeup anchor probe"
        )

    f = demo.identity_facts()

    assert f["n_named_players"] == 3
    assert f["n_named_frags"] == 43
    assert f["anchor_final"] == 226
    assert f["anchor_shots"] == 95

    # verified precision is a fraction of the audited sample, not > 1
    assert 0.9 <= f["anchor_precision"] <= 1.0
    assert f["anchor_verified_correct"] <= f["anchor_verified_n"]

    # hero-shot concentration: #8 dominates
    assert f["hero_number"] == 8
    assert f["hero_share"] > 0.5

    # Bruno's Sofascore row is the validation anchor
    bruno = next(r for r in f["validation"] if r["name"] == "Bruno Fernandes")
    assert bruno["oracle_min"] == 79.0
    assert bruno["touches"] == 49.0
    assert bruno["our_min"] < bruno["oracle_min"]  # visible-minutes << played-minutes, by design

    # external benchmark: official floor and the relink association lift
    assert round(f["gsr"]["gs_hota_full"]["GS-HOTA"], 1) == 14.8
    before = f["relink"]["before"]["loc_assoc"]["GS-AssA"]
    after = f["relink"]["after"]["loc_assoc"]["GS-AssA"]
    assert after > before
