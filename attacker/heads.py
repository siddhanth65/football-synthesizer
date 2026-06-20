"""Per-player node-level run + receiver heads (retrained on video tracks).

Reuses the relational backbone from ``football-state-of-play`` (``models/gnn.py``, ``models/heads``)
but trains node-level heads on the richer contract (velocity + orientation features) with real
per-player labels (:mod:`attacker.labels`):

- **run head**: predicts a 2D displacement *distribution* (Gaussian) at t+1.5 s.
- **receiver head**: node softmax over teammates.

Evaluated against the old 360 baselines in :mod:`eval.attacker_eval` (run RMSE / hit@3m; receiver
top-1/3 vs the nearest-teammate heuristic).
"""

from __future__ import annotations


def train_run_head(*args, **kwargs):
    """Train the per-player displacement-distribution run head. TODO."""
    raise NotImplementedError("TODO: node-level run head on contract features")


def train_receiver_head(*args, **kwargs):
    """Train the node-level receiver softmax head. TODO."""
    raise NotImplementedError("TODO: node-level receiver head on contract features")
