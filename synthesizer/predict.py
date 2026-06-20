"""Tendency predictor: Tier A marginal P(y|z_T) -> Tier B matchup P(y|z_T, z_O).

Outputs **distributions** (not point predictions) over tendencies: block height / press intensity,
attack-channel split (L/C/R), directness, likely formation. Wrap with split-conformal prediction
sets for calibrated uncertainty.
"""

from __future__ import annotations


def predict_tendencies(team: str, opponent: str | None = None, *, tier: str = "B") -> dict:
    """Predict tendency distributions for ``team`` (optionally vs ``opponent``). TODO."""
    raise NotImplementedError("TODO: marginal (A) and opponent-conditioned (B) tendency models")
