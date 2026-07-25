"""Pundit-voice invariants for the narrative scouting pack v2 (no heavy data needed).

Guards the three things the pundit pass must keep true in the generator: every match with a
story has a seam read and a result label, multi-paragraph prose actually renders as separate
paragraphs, the validation appendix stays present but collapsed, and the caveated space-control
proxy (which inverts the event oracle) is never narrated in the prose.
"""
from __future__ import annotations

import re

from tools import render_scouting_v2 as rs

# The space-control / trackable-possession proxy values the reports flag as inverting the oracle.
_CAVEATED_PROXY = re.compile(r"space[- ]control|possession (?:share|proxy) of", re.IGNORECASE)


def test_prose_tables_cover_the_same_matches() -> None:
    """Every match with story prose also has a seam read, a result and an event-layer line."""
    assert set(rs.STORY) == set(rs.SEAMS) == set(rs.EVENTS_VALIDATED) == set(rs.RESULT)
    assert set(rs.STORY) <= set(rs.SHORT)


def test_story_paragraphs_render_separately() -> None:
    """Blank-line-separated story prose becomes one ``<p>`` per paragraph."""
    for match_id, prose in rs.STORY.items():
        n_par = len(prose.split("\n\n"))
        assert n_par >= 2, f"{match_id}: pundit story should open with more than one paragraph"
        html = "".join(rs._p(par) for par in prose.split("\n\n"))
        assert html.count("<p>") == n_par


def test_prose_never_narrates_the_caveated_territory_proxy() -> None:
    """Prose must not quote the space-control proxy as fact; abstention wording is allowed."""
    for match_id, prose in {**rs.STORY, **rs.SEAMS}.items():
        hit = _CAVEATED_PROXY.search(prose)
        assert hit is None, f"{match_id}: narrates the caveated proxy ({hit.group(0)!r})"


def test_appendix_is_collapsed_but_present() -> None:
    """The validation appendix renders inside a ``<details>`` with its tier chip intact."""
    html = rs._collapsed_panel("Validation appendix", rs.ORACLE, rs.APPX_HINT, "<p>body</p>")
    assert "<details class=\"appx\">" in html
    assert "<summary>" in html and "Validation appendix" in html
    assert "tag-oracle" in html and "<p>body</p>" in html


def test_code_spans_survive_escaping() -> None:
    """``<code>`` file references in prose render as markup, everything else stays escaped."""
    out = rs._inline("see <code>results/x.md</code> & <b>not this</b>")
    assert "<code>results/x.md</code>" in out
    assert "&lt;b&gt;" in out and "&amp;" in out
