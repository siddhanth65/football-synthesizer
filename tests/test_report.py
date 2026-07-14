"""Test the descriptive EFI-style report renderer (A3)."""

from __future__ import annotations

import pandas as pd

from report.build_report import render_match_report


def test_render_match_report_writes_html(tmp_path):
    facets = pd.DataFrame([
        {"team": 0, "frames": 100, "att_buildup_height": 47.0, "att_threat_xt": 0.04,
         "def_line_height": 37.0, "pass_nearest_mate_m": 8.5, "gk_sweeper_height_m": 8.0,
         "phys_top_speed_kmh": 34.0},
        {"team": 1, "frames": 90, "att_buildup_height": 42.0, "att_threat_xt": 0.03,
         "def_line_height": 41.0, "pass_nearest_mate_m": 9.5, "gk_sweeper_height_m": 13.0,
         "phys_top_speed_kmh": 33.0},
    ])
    phases = pd.DataFrame([
        {"team": 0, "phase": "in_poss", "frames": 50, "buildup_height": 46, "def_line_height": 40,
         "width": 31, "compactness": 11},
        {"team": 0, "phase": "out_poss", "frames": 50, "buildup_height": 41, "def_line_height": 35,
         "width": 30, "compactness": 11},
    ])
    out = render_match_report(facets, phases, team_names=("Red", "Blue"), out_path=tmp_path / "r.html")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Red" in html and "Blue" in html
    for token in ("Attacking", "Build-up height", "Defending", "Goalkeeping", "Sweeper height",
                  "Line height by phase"):
        assert token in html
