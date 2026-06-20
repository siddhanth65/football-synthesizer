"""Aggregate the trained GAT's per-frame reads into a per-team identity vector.

Mirrors ``football-state-of-play`` ``eval/team_metrics.py`` (attack_success, attack_dxt,
option_richness, solidity, press_decisiveness, line_height, lane_suppression) and adds classical
style axes (formation/role occupancy, line height, team width, directness, L/C/R attack share,
PPDA). One vector per team-match; a team-history encoder later produces the embedding ``z_T`` the
synthesizer conditions on.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from generator.contract import FreezeFrame

ATTACK_AXES = ("attack_success", "attack_dxt", "option_richness", "directness", "channel_left",
               "channel_central", "channel_right")
DEFEND_AXES = ("solidity", "press_decisiveness", "lane_suppression", "line_height", "team_width")


def fingerprint_from_frames(frames: Iterable[FreezeFrame], *, model=None) -> pd.Series:
    """Run the GAT over frames and aggregate to one team-identity vector. TODO: wire the model."""
    raise NotImplementedError("TODO: GAT reads -> team_metrics schema + style axes aggregation")
