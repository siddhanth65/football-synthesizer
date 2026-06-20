"""Backtest the synthesizer: did teams actually play how we predicted?

Leave-one-match-out (temporal: predict future from past) and leave-one-team-out splits. Compare
predicted tendency distributions to actual match outcomes AND to FIFA EFI of those matches. Report
CRPS / Brier + reliability, and the key result: **skill of Tier B over the Tier A team-average**.
"""

from __future__ import annotations

import pandas as pd


def backtest(history: pd.DataFrame, *, scheme: str = "leave_one_match_out") -> pd.DataFrame:
    """Run the backtest; return per-tendency CRPS/Brier + Tier-B-minus-Tier-A skill. TODO."""
    raise NotImplementedError("TODO: temporal LOMO / LOTO backtest vs actual + EFI")
