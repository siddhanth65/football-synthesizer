"""Pure-seam checks for the Phase-A player profiles (no GPU, no data files)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint import player_profiles as pp


def test_kmeans_1d_recovers_three_bands() -> None:
    """Three well-separated groups cluster into ascending centres near their true means."""
    x = np.array([20, 21, 22, 45, 46, 47, 90, 91, 92], float)
    lab, c = pp.kmeans_1d(x, 3)
    assert list(c) == sorted(c)                      # centres ascending -> def/mid/att order
    assert abs(c[0] - 21) < 2 and abs(c[2] - 91) < 2
    assert lab[0] == 0 and lab[-1] == 2              # rearmost group -> defensive band


def test_involvement_poss_link_fraction() -> None:
    """A player on one of two Man Utd edges appears in exactly half the possession-links."""
    ledger = pd.DataFrame({
        "class": ["PASS", "PASS", "PASS"],
        "chunk": ["h1_chunk_000"] * 3,
        "frame_index": [0, 25, 50],
        "team": [0, 0, 0],
        "team_name": ["Man Utd"] * 3,
        "player": ["Bruno", None, "Casemiro"],
    })
    inv = pp.involvement(ledger, manu_int=0, roster={"Bruno": "Man Utd", "Casemiro": "Man Utd"})
    assert inv.attrs["n_manu_links"] == 2            # 3 consecutive passes -> 2 edges
    bruno = inv[inv["player"] == "Bruno"].iloc[0]
    assert bruno["attr_pass"] == 1
    assert bruno["poss_link_frac"] == 0.5            # endpoint of edge 1 of 2


def test_line_composition_names_rearmost_defender() -> None:
    """The rearmost named player lands in the defensive band (needs >=25 trusted frames each)."""
    n = 30  # above MIN_TRUST_FR so the named positions are trusted
    players = ["CB"] * n + ["CM"] * n + ["ST"] * n
    xs = [20.0] * n + [45.0] * n + [88.0] * n
    pos = pd.DataFrame({
        "player": players,
        "track_id": [1] * n + [2] * n + [3] * n,
        "team_int": [0] * (3 * n),
        "role": ["player"] * (3 * n),
        "chunk": ["h1_chunk_000"] * (3 * n),
        "frame": list(range(3 * n)),
        "x": xs,
        "y": [34.0] * (3 * n),
    })
    geom = pp.player_geometry(pos, {"CB": "Man Utd", "CM": "Man Utd", "ST": "Man Utd"})
    assert geom["trusted"].all()
    lc = pp.line_composition(pos, manu_int=0, geom=geom)
    assert lc["centres_m"][0] < lc["centres_m"][2]
    assert any("CB" in a for a in lc["bands"]["defensive"])
    assert any("ST" in a for a in lc["bands"]["attacking"])
