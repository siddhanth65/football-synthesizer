"""Per-player run + receiver labels from dense tracks and FIFA EFI.

- **Run target**: a player's *displacement* at t+1.5 s read directly off the persistent track (no
  nearest-neighbour ID guessing -- the fix for the old 360 label noise).
- **Receiver**: the actual next on-ball receiver, enriched with EFI movement-to-receive / offers
  labels (:mod:`ingest.fifa_efi`).
"""

from __future__ import annotations

import pandas as pd

RUN_HORIZON_S = 1.5


def run_targets(tracks: pd.DataFrame, *, horizon_s: float = RUN_HORIZON_S) -> pd.DataFrame:
    """Per-player displacement label at ``t + horizon_s`` from persistent tracks. TODO."""
    raise NotImplementedError("TODO: displacement labels from dense tracks")


def receiver_labels(tracks: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Next-receiver labels (+ optional EFI movement/offers enrichment). TODO."""
    raise NotImplementedError("TODO: receiver labels (+ EFI enrichment)")
