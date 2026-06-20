"""Parse FIFA Enhanced Football Intelligence (EFI) post-match PDFs into tidy tables.

EFI reports (e.g. the France-Senegal sample) carry BOTH team-level metrics (phases of play, line
breaks, defensive line height, pressure, forced turnovers) and PER-PLAYER attacking detail
(movement-to-receive types, offers-to-receive, individual line breaks). The per-player tables are
the off-ball-intent labels for the attacker rebuild (C3); the team tables are the validation oracle
(C6) and synthesizer features.

Implementation note: use ``pdfplumber`` to extract the tabular pages. EFI layouts are stable across
matches, so per-section extractors keyed by page title are the pragmatic approach.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_team_metrics(pdf_path: str | Path) -> pd.DataFrame:
    """Parse the team-level EFI metrics (one row per team).

    Returns columns like ``team, possession, line_breaks, def_line_height, phases_*, pressure,
    forced_turnovers, ...``. TODO: implement with pdfplumber per-section extraction.
    """
    raise NotImplementedError("TODO: pdfplumber extraction of EFI team metric pages")


def parse_player_movement(pdf_path: str | Path) -> pd.DataFrame:
    """Parse the per-player 'Movement to Receive' / 'Offering to Receive' tables.

    Returns one row per player with ``in_front, in_between, out_to_in, in_to_out, in_behind,
    offers_made, offers_received, ...`` -- the off-ball-intent labels for C3. TODO.
    """
    raise NotImplementedError("TODO: pdfplumber extraction of per-player movement pages")
