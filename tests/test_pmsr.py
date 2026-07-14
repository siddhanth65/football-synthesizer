"""Tests for FIFA PMSR parsing (pure regex on sample text; no PDF needed)."""

from __future__ import annotations

from tools.parse_pmsr import parse_header, parse_phases, parse_stats

_PHASES = """France Phases of Play Senegal
IN POSSESSION
41% Build Up Unopposed 38%
19% Progression 17%
OUT OF POSSESSION
30% Mid Block 31%
13% Low Block 17%
"""

_STATS = """France Senegal
Total 49.4% 6.2% 44.5% Total
1.62 xG (Expected Goals) 0.39
118 Completed Line Breaks 101
99 Receptions in the Final Third 78
"""


def test_parse_phases_home_away():
    p = parse_phases(_PHASES)
    assert p["build_up_unopposed"] == [41, 38]
    assert p["mid_block"] == [30, 31]
    assert p["low_block"] == [13, 17]


def test_parse_stats_pulls_home_away_numbers():
    s = parse_stats(_STATS)
    assert s["xg"] == [1.62, 0.39]
    assert s["completed_line_breaks"] == [118.0, 101.0]
    assert s["possession_pct"][0] == 49.4
    assert s["receptions_final_third"] == [99.0, 78.0]


def test_parse_header_teams_and_score():
    h = parse_header("\n3\n1\nFrance Senegal\n")
    assert h.get("score") == [3, 1]
