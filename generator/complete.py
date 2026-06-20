"""Off-screen player completion (novelty C2).

Broadcast shows only ~6-14 of 20 outfielders. This module imputes the unseen players conditioned on
the visible players' **position + velocity + body orientation** and a learned **team formation
prior** -- a twist over position-only imputation baselines (AgentImputer ~6.9 m is the bar to beat).
Outputs imputed :class:`PlayerNode` objects (``observed=False``) appended to a partial frame.

Trained on full StatsBomb 360 / tracking frames with simulated broadcast masking (keep players
within ~30 m of the ball, per Royal Soc 2025). Loss = per-node Gaussian-mixture NLL + a Sinkhorn
set term for permutation-invariance. Evaluated on mean position error AND downstream tactical
fidelity (pitch-control / EFI shape).
"""

from __future__ import annotations

from generator.contract import FreezeFrame


def complete_frame(frame: FreezeFrame, *, n_total_outfield: int = 20) -> FreezeFrame:
    """Return ``frame`` with imputed off-screen players (``observed=False``) added. TODO: train model."""
    raise NotImplementedError("TODO: orientation/velocity-conditioned set imputation + formation prior")
