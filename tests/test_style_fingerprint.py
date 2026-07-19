"""Tests for the style fingerprint v1 pure seams (no video / no match data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID
from fingerprint.style_fingerprint import (
    DEF_THIRD_X,
    PRESS_RADIUS_M,
    _orient,
    phase_by_frame,
    sw_directions,
    sw_embed,
)


def test_sw_embed_permutation_invariant():
    rng = np.random.default_rng(1)
    dirs = sw_directions(8)
    pts = rng.uniform(0, 60, size=(10, 2))
    e0 = sw_embed(pts, dirs)
    e1 = sw_embed(pts[rng.permutation(10)], dirs)
    assert np.allclose(e0, e1)


def test_sw_embed_translation_invariant_after_centering():
    rng = np.random.default_rng(2)
    dirs = sw_directions(8)
    pts = rng.uniform(0, 60, size=(9, 2))
    e0 = sw_embed(pts, dirs, center=True)
    e1 = sw_embed(pts + np.array([13.0, -8.0]), dirs, center=True)
    assert np.allclose(e0, e1)
    # ...and the uncentred embedding is NOT translation-invariant (it encodes field position).
    assert not np.allclose(sw_embed(pts, dirs), sw_embed(pts + 4.0, dirs))


def test_sw_embed_distance_matches_sliced_w2_of_a_translation():
    # For a pure translation of magnitude s in 2D the sliced-W2 distance is s/sqrt(2)
    # (SW2^2 = E_theta[(s cos theta)^2] = s^2/2); the embedding L2 must reproduce it.
    dirs = sw_directions(64)
    pts = np.random.default_rng(3).uniform(0, 50, size=(20, 2))
    d = float(np.linalg.norm(sw_embed(pts, dirs) - sw_embed(pts + np.array([6.0, 0.0]), dirs)))
    assert abs(d - 6.0 / np.sqrt(2)) < 0.2


def test_orient_flips_180_when_attacking_negative_x():
    px, py = _orient(np.array([10.0, 90.0]), np.array([20.0, 5.0]), -1)
    assert np.allclose(px, PITCH_LEN - np.array([10.0, 90.0]))
    assert np.allclose(py, PITCH_WID - np.array([20.0, 5.0]))
    # +1 is a no-op.
    px2, py2 = _orient(np.array([10.0]), np.array([20.0]), 1)
    assert px2[0] == 10.0 and py2[0] == 20.0


def test_phase_by_frame_transition_windows():
    # team 0 holds, flips to team 1 at frame 20; window = 5s * 25fps = 125 frames.
    poss = pd.DataFrame({"frame": [0, 10, 20, 100, 300], "carrier": [1, 1, 2, 2, 2],
                         "team": [0, 0, 1, 1, 1]})
    ph = phase_by_frame(poss, teams=[0, 1], fps=25.0, transition_s=5.0)
    at20 = ph[ph["frame"] == 20].set_index("team")["phase"].to_dict()
    assert at20 == {1: "trans_pos", 0: "trans_neg"}
    at100 = ph[ph["frame"] == 100].set_index("team")["phase"].to_dict()   # still inside 125-frame win
    assert at100 == {1: "trans_pos", 0: "trans_neg"}
    at300 = ph[ph["frame"] == 300].set_index("team")["phase"].to_dict()   # outside window
    assert at300 == {1: "in_poss", 0: "out_poss"}
    # before any flip -> plain possession phases
    at0 = ph[ph["frame"] == 0].set_index("team")["phase"].to_dict()
    assert at0 == {0: "in_poss", 1: "out_poss"}


class _FakeMatch:
    """Minimal Match stand-in: one chunk, in-memory aligned + ball, fixed fps."""

    id = "fake"
    teams = ("A", "B")

    def __init__(self, aligned: pd.DataFrame, ball_path):
        self._aligned = aligned
        self._ball_path = ball_path

    def load_aligned(self):
        return self._aligned

    def ball_chunks(self):
        return [("h1_chunk_000", self._ball_path)]

    def chunk_fps(self, _ck, default=25.0):
        return 25.0


def test_counterpress_pressure_radius_and_third_filter(tmp_path):
    import fingerprint.style_fingerprint as sfmod

    # Build a chunk where team 0 (attacks +x, keeper at x~2) loses the ball at midfield (outside its
    # own third) with a team-0 chaser 3 m away (< 4.57 m -> a counter-press), then regains it.
    frames = [0, 5, 10, 15]
    rows = []
    # keepers to fix directions: team 0 keeper near x=2 (defends left -> attacks +x);
    # team 1 keeper near x=103 (defends right -> attacks -x).
    for fr in frames:
        rows.append({"frame": fr, "track_id": 90, "role": "goalkeeper", "team": 0,
                     "pitch_x": 2.0, "pitch_y": 34.0, "is_keeper": True, "is_actor": False})
        rows.append({"frame": fr, "track_id": 91, "role": "goalkeeper", "team": 1,
                     "pitch_x": 103.0, "pitch_y": 34.0, "is_keeper": True, "is_actor": False})
    # carriers near the ball at x=52 (midfield): frame 0 team-0 player, then team-1 wins it.
    ball = pd.DataFrame({"frame": frames, "x": [52.0, 52.0, 52.0, 52.0], "y": [34.0] * 4})
    ball_path = tmp_path / "ball.parquet"
    ball.to_parquet(ball_path)
    # team-0 carrier at f0; team-1 carrier from f5 (the win); team-0 chaser 3 m away in the window;
    # team-0 regains at f15.
    ppl = [
        (0, 1, 0, 52.0, 34.0), (0, 2, 1, 40.0, 34.0),
        (5, 1, 0, 55.0, 34.0), (5, 2, 1, 52.0, 34.0),    # team1 nearest -> wins
        (10, 1, 0, 53.5, 34.0), (10, 2, 1, 52.0, 34.0),  # team0 player 1.5m from ball -> pressure
        (15, 1, 0, 52.0, 34.0), (15, 2, 1, 48.0, 34.0),  # team0 nearest -> regain
    ]
    for fr, tid, team, x, y in ppl:
        rows.append({"frame": fr, "track_id": tid, "role": "player", "team": team,
                     "pitch_x": x, "pitch_y": y, "is_keeper": False, "is_actor": False})
    aligned = pd.DataFrame(rows)
    aligned["chunk"] = "h1_chunk_000"
    m = _FakeMatch(aligned, ball_path)
    cp = sfmod.counterpress(m, transition_s=5.0)
    row0 = cp[cp["team"] == 0].iloc[0]
    assert row0["losses_outside_third"] == 1          # loss at x=52 is outside team-0's own third
    assert row0["counterpress_frac"] == 1.0           # chaser within 4.57 m in the window
    assert row0["regain_5s_frac"] == 1.0              # team 0 back on the ball by f15
    # sanity on the constants the definition rides on
    assert abs(PRESS_RADIUS_M - 4.57) < 1e-9
    assert abs(DEF_THIRD_X - PITCH_LEN / 3) < 1e-9
