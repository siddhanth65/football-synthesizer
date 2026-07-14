"""Tests for positional-role + formation inference (no ball, synthetic formations)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.roles import (
    FORMATIONS,
    assign_roles,
    infer_formation,
    relabel_to_roles,
    stable_role_id,
    track_means,
)


def _team_from_template(team, formation, attack_dir, *, n_frames=40, jitter=0.5, seed=0):
    """Lay players on a formation template (in pitch coords for the given attack direction) over frames."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid, (_, ax, ay) in enumerate(FORMATIONS[formation]):
        px = ax if attack_dir == 1 else (105.0 - ax)      # invert attacking-x to pitch-x
        py = ay if attack_dir == 1 else (68.0 - ay)
        is_kp = FORMATIONS[formation][tid][0] == "GK"
        for f in range(n_frames):
            rows.append({"frame": f, "track_id": team * 100 + tid, "team": team,
                         "role": "goalkeeper" if is_kp else "player", "is_keeper": is_kp,
                         "pitch_x": px + rng.normal(0, jitter), "pitch_y": py + rng.normal(0, jitter)})
    return rows


def test_track_means_normalises_to_attacking_coords():
    rows = _team_from_template(0, "4-3-3", attack_dir=-1)   # attacks -x
    tm = track_means(pd.DataFrame(rows), team=0, attack_dir=-1)
    assert len(tm) == 11
    gk = tm[tm["is_keeper"]].iloc[0]
    assert gk["ax"] < 15            # keeper is deepest in attacking-x regardless of pitch side


def test_infer_formation_recovers_the_planted_shape():
    rows = _team_from_template(0, "4-2-3-1", attack_dir=1)
    tm = track_means(pd.DataFrame(rows), team=0, attack_dir=1)
    name, cost, assign = infer_formation(tm)
    assert name == "4-2-3-1"        # the planted formation is the best fit
    assert cost < 3.0               # tight fit on clean synthetic data
    assert assign[0 * 100 + 0] == "GK"   # the keeper track maps to GK


def test_assign_roles_gives_eleven_slots_per_team():
    rows = _team_from_template(0, "4-3-3", attack_dir=1) + _team_from_template(1, "4-4-2", attack_dir=-1)
    df = pd.DataFrame(rows)
    df["chunk"] = "A"
    roles = assign_roles(df)
    assert set(roles.groupby("team").size()) == {11}
    assert (roles[roles["team"] == 0]["formation"] == "4-3-3").all()
    assert (roles[roles["team"] == 1]["formation"] == "4-4-2").all()


def test_low_presence_tracks_are_dropped():
    rows = _team_from_template(0, "4-3-3", attack_dir=1)
    # add a transient false track present in only a few frames
    rows += [{"frame": f, "track_id": 999, "team": 0, "role": "player", "is_keeper": False,
              "pitch_x": 52.0, "pitch_y": 34.0} for f in range(5)]
    tm = track_means(pd.DataFrame(rows), team=0, attack_dir=1, min_frames=30)
    assert 999 not in set(tm["track_id"])


def test_relabel_collapses_fragments_to_roles_and_drops_unlabelled():
    df = pd.DataFrame(_team_from_template(0, "4-3-3", attack_dir=1))
    df["chunk"] = "A"
    roles = assign_roles(df)
    # possession over labelled carriers + one unlabelled fragment id (12345)
    poss = pd.DataFrame({"frame": [0, 1, 2], "carrier": [0, 12345, 3], "team": [0, 0, 0],
                         "dist_m": [0.5, 0.5, 0.5]})
    poss2, pl2, labels = relabel_to_roles(poss, df, roles, chunk="A")
    assert 12345 not in set(poss2["carrier"])              # unlabelled fragment dropped
    assert set(poss2["carrier"]).issubset(set(pl2["track_id"]))
    assert pl2["track_id"].nunique() == 11                  # exactly the 11 role slots
    gk_id = stable_role_id(0, "GK")
    assert labels[gk_id] == "0:GK"
