"""Generate the FIFA-EFI-style team report for any team (the end deliverable).

Mirrors the FIFA EFI post-match layout (phases of play, line breaks, defensive line height,
receptions between lines, pressure, attack channels) computed from our pipeline, AND adds the
synthesizer's **forecast**: how the team is likely to play -- optionally vs a chosen opponent --
with calibrated uncertainty. Rendered to HTML/PDF via Jinja2.

This is what the supervisor sees: a familiar-looking report that both *describes* and *predicts*.
"""

from __future__ import annotations

from pathlib import Path

REPORT_DIR = Path("results/reports")


def build_team_report(team: str, opponent: str | None = None, *, out_dir: Path = REPORT_DIR) -> Path:
    """Render a FIFA-style report for ``team`` (optionally a matchup vs ``opponent``). TODO.

    Sections: identity fingerprint, phases of play, line breaks, defensive line height, attack
    channels (descriptive) + predicted tendencies with uncertainty (from the synthesizer).
    """
    raise NotImplementedError("TODO: Jinja2 template + figures (mplsoccer) -> HTML/PDF")
