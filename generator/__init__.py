"""Freeze-frame generator: broadcast video -> unified, substrate-aware freeze-frames.

The contract (:mod:`generator.contract`) is the foundation everything plugs into. The heavy CV
steps (:mod:`generator.extract`, :mod:`generator.calibrate`, :mod:`generator.complete`) require the
optional ``[cv]`` dependency stack and a GPU.
"""

from generator.contract import (
    FEATURE_NAMES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    FreezeFrame,
    PlayerNode,
    Substrate,
    from_statsbomb,
    node_feature_matrix,
    orient_left_to_right,
    rescale_xy,
    to_model_frame,
    to_statsbomb_dataframe,
)

__all__ = [
    "FEATURE_NAMES",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "FreezeFrame",
    "PlayerNode",
    "Substrate",
    "from_statsbomb",
    "node_feature_matrix",
    "orient_left_to_right",
    "rescale_xy",
    "to_model_frame",
    "to_statsbomb_dataframe",
]
