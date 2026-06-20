"""Evaluate the rebuilt attacker heads against the OLD StatsBomb-360 baselines (the C3 headline).

Old 360 numbers to beat (from football-state-of-play): run RMSE 6.33 m / hit@3m 0.244 (no-move 6.46
m / 0.221); receiver top-1 0.41 / top-3 0.78 (nearest-teammate heuristic ~0.42 top-1). Report the
video-trained heads on the same metrics so the improvement is unambiguous.
"""

from __future__ import annotations

OLD_360 = {
    "run_rmse_m": 6.33,
    "run_hit3m": 0.244,
    "run_nomove_rmse_m": 6.46,
    "receiver_top1": 0.41,
    "receiver_top3": 0.78,
    "receiver_heuristic_top1": 0.42,
}


def evaluate(predictions, labels) -> dict:
    """Return run RMSE / hit@3m and receiver top-1/3, alongside :data:`OLD_360`. TODO."""
    raise NotImplementedError("TODO: compute attacker metrics vs OLD_360")
