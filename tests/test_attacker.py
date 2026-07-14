"""Tests for the C3 attacker rebuild: dense tracks, run/receiver labels, and the heads."""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.heads import receiver_candidates, run_examples, train_run_head
from attacker.labels import receiver_labels, run_targets
from attacker.tracks import build_tracks


def _linear_track(track_id=1, team=0, vx=2.0, n=31, step=5, fps=50.0):
    """A player moving at constant ``vx`` m/s in +x (frames spaced by ``step``)."""
    fr = np.arange(n) * step
    return pd.DataFrame({
        "frame": fr, "track_id": track_id, "role": "player", "team": team,
        "pitch_x": (fr / fps) * vx, "pitch_y": 34.0, "is_actor": False,
    })


def test_build_tracks_recovers_constant_velocity():
    tr = build_tracks(_linear_track(vx=2.0), fps=50.0)
    interior = tr["vx"].to_numpy()[2:-2]  # ignore the NaN first point / edge effects
    assert np.allclose(interior, 2.0, atol=1e-6)
    assert np.allclose(tr["vy"].dropna(), 0.0, atol=1e-6)


def test_run_targets_reads_future_displacement():
    tr = build_tracks(_linear_track(vx=2.0, n=41), fps=50.0)
    rt = run_targets(tr, horizon_s=1.5, fps=50.0)  # +1.5 s at 2 m/s -> +3 m
    assert (rt["frame"] == 0).any()
    assert abs(float(rt.loc[rt["frame"] == 0, "dx"].iloc[0]) - 3.0) < 1e-6


def test_receiver_labels_from_carrier_transitions():
    rows = [  # carrier 1 -> 2 (pass), then 2 -> 3 (pass), a team switch must not count
        {"frame": 0, "track_id": 1, "team": 0, "is_actor": True},
        {"frame": 10, "track_id": 2, "team": 0, "is_actor": True},
        {"frame": 60, "track_id": 3, "team": 0, "is_actor": True},
        {"frame": 70, "track_id": 9, "team": 1, "is_actor": True},  # other team gains it
    ]
    ev = receiver_labels(pd.DataFrame(rows), fps=50.0)
    assert list(zip(ev["carrier"], ev["receiver"])) == [(1, 2), (2, 3)]


def test_chunk_local_ids_do_not_merge_across_chunks():
    # ByteTrack resets ids per chunk, so track_id 1 recurs in both chunks at the SAME frames but is a
    # DIFFERENT player. Grouping by track_id alone would interleave the two y-positions (spurious
    # teleport velocity) and let carrier transitions cross the chunk boundary. The fix keys a track on
    # (chunk, track_id) and compares carriers within a chunk.
    fr = np.arange(7) * 5
    a = pd.DataFrame({"chunk": "h1_chunk_000", "frame": fr, "track_id": 1, "role": "player",
                      "team": 0, "pitch_x": (fr / 50.0) * 2.0, "pitch_y": 10.0, "is_actor": False})
    b = pd.DataFrame({"chunk": "h1_chunk_001", "frame": fr, "track_id": 1, "role": "player",
                      "team": 0, "pitch_x": 40.0, "pitch_y": 60.0, "is_actor": False})
    tr = build_tracks(pd.concat([a, b], ignore_index=True), fps=50.0)

    # two distinct physical tracks, not one merged blob
    assert tr.groupby(["chunk", "track_id"]).ngroups == 2
    # chunk A: clean vx=+2, vy=0 -- interleaving with chunk B (y jumps 10<->60) would blow vy up
    ga = tr[tr["chunk"] == "h1_chunk_000"].sort_values("frame")
    assert np.allclose(ga["vx"].to_numpy()[2:-2], 2.0, atol=1e-6)
    assert np.allclose(ga["vy"].dropna().to_numpy(), 0.0, atol=1e-6)
    gb = tr[tr["chunk"] == "h1_chunk_001"].sort_values("frame")
    assert np.allclose(gb["vy"].dropna().to_numpy(), 0.0, atol=1e-6)

    # receiver labels: each chunk has its own carrier->receiver pass; no cross-chunk contamination
    acts = pd.DataFrame([
        {"chunk": "h1_chunk_000", "frame": 0, "track_id": 1, "team": 0, "is_actor": True},
        {"chunk": "h1_chunk_000", "frame": 10, "track_id": 7, "team": 0, "is_actor": True},
        {"chunk": "h1_chunk_001", "frame": 0, "track_id": 1, "team": 0, "is_actor": True},
        {"chunk": "h1_chunk_001", "frame": 10, "track_id": 9, "team": 0, "is_actor": True},
    ])
    ev = receiver_labels(acts, fps=50.0)
    got = set(zip(ev["carrier"], ev["receiver"], ev["chunk"]))
    assert got == {(1, 7, "h1_chunk_000"), (1, 9, "h1_chunk_001")}


def test_receiver_candidates_resolve_pass_frame_within_chunk():
    # Same (frame, carrier track_id) exists in two chunks with DIFFERENT team-mate geometry. The
    # candidate builder must read positions from the event's own chunk, not merge both frames.
    def _chunk(name, mate_x):
        return pd.DataFrame({
            "chunk": name, "frame": [0, 0], "track_id": [1, 2], "team": [0, 0],
            "x_s": [50.0, mate_x], "y_s": [34.0, 34.0], "is_actor": [True, False],
        })
    tracks = pd.concat([_chunk("h1_chunk_000", 55.0), _chunk("h1_chunk_001", 90.0)],
                       ignore_index=True)
    recv = pd.DataFrame([
        {"chunk": "h1_chunk_000", "frame": 0, "carrier": 1, "receiver": 2, "team": 0, "dt_s": 0.2},
        {"chunk": "h1_chunk_001", "frame": 0, "carrier": 1, "receiver": 2, "team": 0, "dt_s": 0.2},
    ])
    cand = receiver_candidates(tracks, recv, {0: 1})
    # one candidate per event, each with exactly one on-pitch team-mate from its own chunk (dist 5, 40)
    assert cand["event"].nunique() == 2 and len(cand) == 2
    assert sorted(round(d, 3) for d in cand["dist_carrier"]) == [5.0, 40.0]


def test_run_examples_normalises_attack_direction():
    # team 1 attacks -x; normalisation must flip x, vx and dx so it reads as attacking +x
    run_df = pd.DataFrame({
        "team": [1], "x_s": [80.0], "y_s": [34.0], "vx": [-3.0], "vy": [1.0],
        "dx": [-4.5], "dy": [1.5], "track_id": [7],
    })
    ex = run_examples(run_df, {1: -1})
    assert abs(ex["x"].iloc[0] - 25.0) < 1e-9       # 105 - 80
    assert abs(ex["vx"].iloc[0] - 3.0) < 1e-9        # -(-3)
    assert abs(ex["dx"].iloc[0] - 4.5) < 1e-9        # -(-4.5)
    assert abs(ex["dy"].iloc[0] - 1.5) < 1e-9        # width axis unchanged


def test_train_run_head_beats_no_move_on_ballistic_motion():
    rng = np.random.default_rng(0)
    rows = []
    for tid in range(60):  # constant-velocity players: displacement == v * 1.5 exactly
        vx, vy = rng.uniform(-4, 4), rng.uniform(-3, 3)
        for k in range(4):
            rows.append({"team": 0, "track_id": tid, "x_s": 20 + k, "y_s": 30 + k,
                         "vx": vx, "vy": vy, "dx": vx * 1.5, "dy": vy * 1.5})
    res = train_run_head(run_df=pd.DataFrame(rows), attack_dirs={0: 1}, seed=0)
    m = res["metrics"]
    assert m["const_vel"]["rmse_m"] < 1e-6           # exact for ballistic motion
    assert m["learned"]["rmse_m"] < m["no_move"]["rmse_m"]  # motion features carry the signal
    assert m["learned"]["hit@3m"] >= m["no_move"]["hit@3m"]


def test_speed_heading_recovers_magnitude_when_mean_shrinks():
    # constant features but random run direction at a fixed speed: the mean predictor collapses to ~0
    # (directions cancel), while speed x heading keeps realistic length.
    rng = np.random.default_rng(1)
    disp = 6.0 * 1.5  # 9 m over the horizon
    rows = []
    for tid in range(80):
        ang = rng.uniform(-np.pi, np.pi)
        for _ in range(3):
            rows.append({"team": 0, "track_id": tid, "x_s": 30.0, "y_s": 34.0, "vx": 0.0, "vy": 0.0,
                         "dx": disp * np.cos(ang), "dy": disp * np.sin(ang)})
    m = train_run_head(run_df=pd.DataFrame(rows), attack_dirs={0: 1}, seed=0)["metrics"]
    assert m["learned"]["med_disp_m"] < 2.0          # mean-regression shrinks toward zero
    assert m["speed_heading"]["med_disp_m"] > 6.0     # decomposition recovers ~9 m


def test_receiver_candidates_rank_nearest_first():
    tracks = pd.DataFrame({
        "frame": [0, 0, 0, 0], "track_id": [1, 2, 3, 4], "team": [0, 0, 0, 0],
        "x_s": [50.0, 55.0, 70.0, 50.0], "y_s": [34.0, 34.0, 34.0, 50.0], "is_actor": [True, False, False, False],
    })
    recv = pd.DataFrame([{"frame": 0, "carrier": 1, "receiver": 2, "team": 0, "dt_s": 0.2}])
    cand = receiver_candidates(tracks, recv, {0: 1})
    assert len(cand) == 3 and int(cand["is_receiver"].sum()) == 1
    nearest = cand.sort_values("dist_carrier").iloc[0]
    assert int(nearest["candidate"]) == 2 and int(nearest["is_receiver"]) == 1
