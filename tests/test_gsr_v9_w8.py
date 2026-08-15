"""Checks for the v9 W8 attach oracle's non-trivial logic.

The load-bearing invariant is the one the official evaluator enforces: a submission may not emit the
same track id twice in one timestep ("Tracker predicts the same ID more than once in a single
timestep"), so two discarded boxes competing for one host must not both be written.
"""

from __future__ import annotations

import numpy as np

from tools.gsr_v9_w8 import SeqState, build_attach_arm, candidate_host, pan_shift, separation


def _state(n_ts: int = 3) -> SeqState:
    """A hand-built sequence state: one host track present at t=0 and t=2, absent at t=1."""
    st = SeqState.__new__(SeqState)
    st.name = "FAKE"
    st.gt_rows = [[] for _ in range(n_ts)]
    st.pr_rows = [[] for _ in range(n_ts)]
    st.frame_of_ts = {t: t for t in range(n_ts)}
    st.id_of_ts = {t: f"img{t}" for t in range(n_ts)}
    row = {"track_id": 7, "attributes": {"role": "player", "team": "left", "jersey": "9"},
           "bbox_image": {"x": 90.0, "y": 160.0, "x_center": 100.0, "y_center": 180.0,
                          "w": 20.0, "h": 40.0},
           "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}}
    st.track_ts = {7: {0: row, 2: row}}
    st.track_span = {7: (0, 2)}
    st.track_gt = {7: 7}
    st.gt_track_tracks = {7: [7]}
    st.gt_covered = {t: set() for t in range(n_ts)}
    st.gt_of_discard = {}
    st.cum = np.zeros((n_ts, 2))
    st.pan = np.zeros(n_ts)
    st.kept = {t: np.zeros((0, 2)) for t in range(n_ts)}
    st.homog = {t: np.eye(3) for t in range(n_ts)}
    st.discards = {t: [] for t in range(n_ts)}
    return st


def _box(cx: float, cy: float, h: float = 40.0) -> tuple:
    """A discard tuple centred on ``(cx, cy)`` with foot point at the box bottom."""
    return (np.array([cx, cy]), np.array([cx - 10.0, cy - h, cx + 10.0, cy]), 0.9, "player")


def test_one_attach_per_host_per_timestep() -> None:
    """Two discards contending for the same absent host produce exactly one row, the nearer."""
    st = _state()
    st.discards[1] = [_box(105.0, 200.0), _box(101.0, 200.0)]
    rows, audit = build_attach_arm(st, "rule", gap=3, tau=1.0)
    assert len(rows) == 1, rows
    assert audit["dropped_host_conflict"] == 1
    assert rows[0]["track_id"] == 7
    # the nearer box (101) wins
    assert rows[0]["bbox_image"]["x_center"] == 101.0
    # and it inherits the host's own attributes, not GT's
    assert rows[0]["attributes"] == {"role": "player", "team": "left", "jersey": "9"}


def test_no_attach_onto_a_host_that_is_present() -> None:
    """A host with a row at this timestep is never a candidate (that would be a duplicate id)."""
    st = _state()
    st.discards[0] = [_box(101.0, 200.0)]
    assert candidate_host(st, 0, np.array([101.0, 200.0]), 40.0, gap=3, tau=1.0) is None
    assert build_attach_arm(st, "rule", gap=3, tau=1.0)[0] == []


def test_candidate_rule_gates() -> None:
    """Distance, size consistency and the separation gate each veto a pick."""
    st = _state()
    foot = np.array([101.0, 200.0])
    assert candidate_host(st, 1, foot, 40.0, gap=3, tau=1.0)[0] == 7
    assert candidate_host(st, 1, foot, 40.0, gap=3, tau=0.001) is None   # too far
    assert candidate_host(st, 1, foot, 200.0, gap=3, tau=1.0) is None    # 5x the host's height
    assert candidate_host(st, 1, foot, 40.0, gap=0, tau=1.0) is None     # no row inside the gap
    st.kept[1] = np.array([[103.0, 200.0]])
    assert separation(st, 1, foot, 40.0) == 2.0 / 40.0
    assert build_attach_arm(st, "rule", gap=3, tau=1.0, sep_min=0.5)[0] == []


def test_pan_gate_and_pan_estimate() -> None:
    """The pan gate skips low-motion frames; the estimate is the median, not the mean."""
    st = _state()
    st.discards[1] = [_box(101.0, 200.0)]
    assert build_attach_arm(st, "rule", gap=3, tau=1.0, pan_min=8.0)[0] == []
    st.pan[1] = 12.0
    assert len(build_attach_arm(st, "rule", gap=3, tau=1.0, pan_min=8.0)[0]) == 1
    assert pan_shift({1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)},
                     {1: (2.0, 0.0), 2: (2.0, 0.0), 3: (99.0, 0.0)}) == (2.0, 0.0)
