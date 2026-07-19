"""Targeted checks on the pure seams of the HTML scouting-pack renderer.

Covers the tier-label mapping and the pitch->SVG scaling math (corner + centre invariants). No
network, no fact store: only the pure helpers.
"""
from __future__ import annotations

from core.pitch import PITCH_LEN, PITCH_WID
from tools.render_html_report import pitch_to_svg, tier_badge


def test_tier_badge_mapping() -> None:
    """Each gate tier maps to a distinct label + css class; unknown falls back to absolute."""
    assert tier_badge("absolute")[1] == "t-abs"
    assert "relative" in tier_badge("comparative")[0].lower()
    assert tier_badge("comparative")[1] == "t-cmp"
    assert tier_badge("abstain")[0] == "ABSTAINED"
    assert tier_badge("abstain")[1] == "t-abst"
    assert tier_badge("nonsense") == tier_badge("absolute")  # safe fallback


def test_pitch_to_svg_corners_and_centre() -> None:
    """(0,0) and (105,68) map to opposite inner corners; the centre maps to the box centre."""
    w, h, m = 420.0, 280.0, 14.0
    # pitch (0,0) = bottom-left touchline corner -> (margin, h-margin) after the y-flip
    x, y = pitch_to_svg(0, 0, w, h, m)
    assert abs(x - m) < 1e-6 and abs(y - (h - m)) < 1e-6
    # pitch (105,68) = top-right corner -> (w-margin, margin)
    x, y = pitch_to_svg(PITCH_LEN, PITCH_WID, w, h, m)
    assert abs(x - (w - m)) < 1e-6 and abs(y - m) < 1e-6
    # centre maps to the exact centre of the inner box
    x, y = pitch_to_svg(PITCH_LEN / 2, PITCH_WID / 2, w, h, m)
    assert abs(x - w / 2) < 1e-6 and abs(y - h / 2) < 1e-6


def test_pitch_to_svg_monotonic_and_bounded() -> None:
    """px increases with pitch x; all mapped points stay inside the inner drawable rectangle."""
    w, h, m = 420.0, 280.0, 14.0
    x_lo, _ = pitch_to_svg(20, 34, w, h, m)
    x_hi, _ = pitch_to_svg(80, 34, w, h, m)
    assert x_hi > x_lo
    for xm in (0, 30, 105):
        for ym in (0, 34, 68):
            px, py = pitch_to_svg(xm, ym, w, h, m)
            assert m - 1e-6 <= px <= w - m + 1e-6
            assert m - 1e-6 <= py <= h - m + 1e-6
