"""Tests for the positions-table -> contract-frame seam and the trained-GAT bridge.

The positions->frames path is pure (numpy/pandas) and always runs. The GAT bridge needs the sibling
``football-state-of-play`` repo, its torch-geometric stack and the trained checkpoint, so its test is
skipped cleanly when any of those is absent (keeps ``pytest`` green on a clean clone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from generator.contract import PITCH_LENGTH, FreezeFrame, Substrate, to_statsbomb_dataframe
from generator.to_frames import MIN_PLAYERS, frame_from_positions, frames_from_positions


def _synthetic_positions(frame_id: int = 0, *, attacking_left: bool = True) -> pd.DataFrame:
    """Two teams of 11 on a 105x68 pitch. Team 1 sits in one half; team 2 mirrors it.

    When ``attacking_left`` the ball-carrier's team (team 1) is in the LEFT half, so the contract
    must flip it to attack towards x = 120.
    """
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    t1_x = rng.uniform(5, 50, 11) if attacking_left else rng.uniform(55, 100, 11)
    t2_x = rng.uniform(55, 100, 11) if attacking_left else rng.uniform(5, 50, 11)
    for x in t1_x:
        rows.append({"frame": frame_id, "role": "player", "team": 1,
                     "pitch_x": float(x), "pitch_y": float(rng.uniform(5, 63)), "is_actor": False})
    for x in t2_x:
        rows.append({"frame": frame_id, "role": "player", "team": 2,
                     "pitch_x": float(x), "pitch_y": float(rng.uniform(5, 63)), "is_actor": False})
    rows[0]["is_actor"] = True  # a team-1 player carries the ball
    return pd.DataFrame(rows)


def test_frame_from_positions_orients_attack_left_to_right():
    frame = frame_from_positions(_synthetic_positions(attacking_left=True))
    assert frame is not None
    assert frame.substrate is Substrate.BROADCAST_CV
    assert frame.n_players == 22
    teammates = [p for p in frame.players if p.is_teammate]
    assert len(teammates) == 11
    # The attacking team must end up advanced (mean x in the attacking half).
    assert float(np.mean([p.x for p in teammates])) >= PITCH_LENGTH / 2
    # Exactly one actor and one keeper per side.
    assert sum(p.is_actor for p in frame.players) == 1
    assert sum(p.is_keeper for p in frame.players) == 2


def test_sparse_frame_is_rejected():
    sparse = _synthetic_positions().iloc[: MIN_PLAYERS - 1]
    assert frame_from_positions(sparse) is None


def test_velocity_masked_when_absent_and_passed_through_when_present():
    no_v = frame_from_positions(_synthetic_positions())
    assert no_v is not None and not any(p.has_velocity for p in no_v.players)

    df = _synthetic_positions()
    df["vx"] = 1.0
    df["vy"] = -2.0
    with_v = frame_from_positions(df)
    assert with_v is not None and all(p.has_velocity for p in with_v.players)


def test_frames_from_positions_yields_downprojectable_frames():
    positions = pd.concat([_synthetic_positions(0), _synthetic_positions(1)], ignore_index=True)
    out = list(frames_from_positions(positions))
    assert [fr for fr, _ in out] == [0, 1]
    df = to_statsbomb_dataframe(out[0][1])
    assert list(df.columns) == ["location", "teammate", "actor", "keeper"]
    assert len(df) == 22


def test_gat_bridge_runs_on_synthetic_frame():
    """End-to-end smoke: contract frame -> trained GAT. Skipped without the sibling stack."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from generator import sop_bridge

    try:
        ckpt = sop_bridge.checkpoint_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if not ckpt.exists():
        pytest.skip(f"no trained checkpoint at {ckpt}")

    frame = frame_from_positions(_synthetic_positions())
    assert frame is not None
    model = sop_bridge.load_model()
    pred = sop_bridge.predict_frame(model, frame)
    assert set(pred) == {"success", "dynamic_xt", "p_defstop", "top_receiver"}
    assert 0.0 <= pred["success"] <= 1.0  # a sigmoid probability
