"""Assemble synthesizer inputs: team fingerprints + opponent fingerprint + context + EFI features."""

from __future__ import annotations

import pandas as pd


def build_features(team: str, opponent: str | None, history: pd.DataFrame, *, context: dict | None = None):
    """Build the feature vector for ``g(z_T, z_O, ctx)``. TODO: encode history -> z_T, z_O + ctx."""
    raise NotImplementedError("TODO: team-history encoder + opponent + context features")
