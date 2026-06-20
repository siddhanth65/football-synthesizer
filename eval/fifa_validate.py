"""Validate pipeline-derived metrics against FIFA EFI (the public oracle, C6).

For matches with an EFI report, compute our metrics (team line height, line breaks, phases of play,
pressure; per-player movement/offers) and correlate against FIFA's published numbers. Reports
per-metric correlation + bias -- the "does it measure what it claims" gate that backs the accuracy
requirement.
"""

from __future__ import annotations

import pandas as pd


def validate(ours: pd.DataFrame, efi: pd.DataFrame) -> pd.DataFrame:
    """Per-metric correlation + bias between our metrics and FIFA EFI. TODO."""
    raise NotImplementedError("TODO: align metrics + correlation/bias table")
