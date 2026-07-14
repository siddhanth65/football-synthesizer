"""Tests for the grounded pundit report (uses the real roster + parsed PMSR; FIFA-only path = no CV load)."""

from __future__ import annotations

from pathlib import Path

import pytest

from report.pundit import generate, load_bundle

_HAVE_DATA = Path("data/france_roster.json").exists() and Path("outputs/pmsr/france_senegal.json").exists()
pytestmark = pytest.mark.skipif(not _HAVE_DATA, reason="roster/PMSR ground truth not present")


def test_bundle_resolves_france_index_for_away_match():
    # France is the AWAY side vs Norway -> index 1 in FIFA arrays
    b = load_bundle("france_norway")
    assert b["france_idx"] == 1 and b["opp_idx"] == 0
    b2 = load_bundle("france_senegal")
    assert b2["france_idx"] == 0


def test_report_is_grounded_and_names_players():
    md = generate("france_senegal")  # CV not required; FIFA + roster path
    assert "Senegal" in md
    assert "Mbappe" in md and "Olise" in md          # named attackers
    assert "xG" in md and "line breaks" in md        # grounded in numbers
    assert "low block" in md.lower()                 # links threat to opponent block
    assert "nothing invented" in md                  # the grounding disclaimer
