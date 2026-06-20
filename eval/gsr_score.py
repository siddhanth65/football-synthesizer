"""Score the freeze-frame generator with GS-HOTA on SoccerNet-GSR (honest accuracy anchor).

GS-HOTA jointly measures detection, association and identity on the pitch minimap. Reuse the
SoccerNet ``sn-gamestate`` / TrackLab evaluator; we wrap it so the generator's output is scored on
the GSR test split and reported alongside the literature (SOTA ~63.81).
"""

from __future__ import annotations


def gs_hota(predictions, ground_truth) -> float:
    """Return GS-HOTA for the generator's minimap predictions. TODO: wrap TrackLab evaluator."""
    raise NotImplementedError("TODO: integrate SoccerNet GS-HOTA evaluation")
