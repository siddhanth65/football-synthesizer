"""Tests for the demo-video composer (tools/make_demo.py).

The video itself is not rendered here (too slow); what is tested is everything that could put a
*wrong* or *invented* number on screen: fact sourcing, the interpolation rules (never manufacture a
homography the pipeline did not solve), the structure overlays' abstain behaviour, and the segment
plan's integrity against the registry's artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.pitch import PITCH_LEN, PITCH_WID
from tools.make_demo import (
    MATCH_ID,
    MIN_TEAM_PLAYERS,
    SECTION_CARDS,
    SECTION_TITLES,
    SEGMENTS,
    VIDEO_ROOT,
    FrameState,
    TopDown,
    _lerp_state,
    def_line_x,
    demo_facts,
    lane_shares,
)

KNOWN_OVERLAYS = {"markers", "ids", "lines", "ball", "topdown", "carrier", "calib",
                  "struct_line", "struct_lanes"}


def _player(tid: int, team: int, img: tuple[float, float],
            pitch: tuple[float, float] | None) -> dict:
    """Minimal player record as ChunkClip builds them."""
    return {"tid": tid, "team": team, "role": "player", "img": img, "pitch": pitch,
            "actor": False, "keeper": False}


# --- fact sourcing --------------------------------------------------------------------------------
def test_demo_facts_are_sourced_and_plausible() -> None:
    """Every displayed figure comes from an artifact and lands in a sane range."""
    f = demo_facts(MATCH_ID)
    assert f["ball_coverage"] == pytest.approx(0.519, abs=1e-3)  # persisted eval artifact
    assert f["pass_recall"] == pytest.approx(0.482, abs=1e-3)
    assert f["fixture"].startswith("Brighton 2-1")
    assert 0.0 < f["calib_all_frames"] < f["detect_all_frames"] <= 1.0
    assert f["n_chunks"] == 11
    for key in ("ball_coverage/pass_recall", "fixture", "detect/calib_all_frames"):
        assert f["provenance"][key]


# --- interpolation: never invent geometry ---------------------------------------------------------
def test_lerp_keeps_players_but_drops_geometry_across_a_calibration_failure() -> None:
    """Blending a calibrated sample with an uncalibrated one must not yield a calibrated frame."""
    good = FrameState(players=[_player(1, 0, (10.0, 10.0), (10.0, 10.0))], calibrated=True,
                      calib_err=0.2, lines_img=[np.zeros((2, 2))])
    bad = FrameState(players=[_player(1, 0, (30.0, 10.0), None)], calibrated=False,
                     calib_err=float("inf"))
    out = _lerp_state(good, bad, 0.25)
    assert out.calibrated is False
    assert out.lines_img == []                      # no homography is manufactured
    assert out.players[0]["img"] == (15.0, 10.0)    # detections still interpolate


def test_lerp_interpolates_between_two_calibrated_samples() -> None:
    """Both ends calibrated -> positions, geometry and the ball all blend."""
    a = FrameState(players=[_player(7, 1, (0.0, 0.0), (0.0, 0.0))], calibrated=True, calib_err=0.2,
                   lines_img=[np.zeros((2, 2))], ball_pitch=(0.0, 0.0), ball_img=(0.0, 0.0),
                   ball_observed=True)
    b = FrameState(players=[_player(7, 1, (10.0, 20.0), (10.0, 20.0))], calibrated=True,
                   calib_err=0.4, lines_img=[np.ones((2, 2))], ball_pitch=(10.0, 0.0),
                   ball_img=(10.0, 0.0), ball_observed=True)
    out = _lerp_state(a, b, 0.5)
    assert out.calibrated is True
    assert out.calib_err == pytest.approx(0.3)
    assert out.players[0]["pitch"] == pytest.approx((5.0, 10.0))
    assert out.ball_pitch == pytest.approx((5.0, 0.0))


def test_lerp_marks_a_held_ball_as_inferred() -> None:
    """A ball known on only one side of the bracket is held and flagged not-observed."""
    a = FrameState(calibrated=True, ball_pitch=(50.0, 30.0), ball_img=(1.0, 1.0), ball_observed=True)
    b = FrameState(calibrated=True)
    out = _lerp_state(a, b, 0.2)
    assert out.ball_pitch == (50.0, 30.0)
    assert out.ball_observed is False


# --- structure overlays ---------------------------------------------------------------------------
def test_def_line_abstains_below_the_minimum_visible_players() -> None:
    """Below MIN_TEAM_PLAYERS projected outfielders the line reports nothing (no guessing)."""
    st = FrameState(players=[_player(i, 0, (0.0, 0.0), (float(i), 30.0))
                             for i in range(MIN_TEAM_PLAYERS - 1)], calibrated=True)
    assert def_line_x(st, 0, 1) is None
    assert lane_shares(st, 0) is None


def test_def_line_follows_the_attacking_direction() -> None:
    """The deepest-4 line is measured toward the attacked goal and drawn at the mirrored x."""
    xs = [20.0, 25.0, 30.0, 60.0, 80.0]
    st = FrameState(players=[_player(i, 0, (0.0, 0.0), (x, 34.0)) for i, x in enumerate(xs)],
                    calibrated=True)
    x_pos, height = def_line_x(st, 0, 1)
    assert height == pytest.approx(np.mean([20.0, 25.0, 30.0, 60.0]))
    assert x_pos == pytest.approx(height)
    x_neg, height_neg = def_line_x(st, 0, -1)
    assert height_neg == pytest.approx(np.mean([PITCH_LEN - x for x in (80.0, 60.0, 30.0, 25.0)]))
    assert x_neg == pytest.approx(PITCH_LEN - height_neg)


def test_lane_shares_sum_to_one_and_ignore_unprojected_tracks() -> None:
    """Lane occupation uses only projected players and is a proper distribution."""
    ys = [3.0, 20.0, 34.0, 50.0, 65.0]
    players = [_player(i, 0, (0.0, 0.0), (50.0, y)) for i, y in enumerate(ys)]
    players.append(_player(99, 0, (5.0, 5.0), None))  # detected, never projected
    st = FrameState(players=players, calibrated=True)
    shares = lane_shares(st, 0)
    assert shares.sum() == pytest.approx(1.0)
    assert shares == pytest.approx(np.full(5, 0.2))


# --- panel geometry -------------------------------------------------------------------------------
def test_topdown_maps_the_pitch_inside_the_panel() -> None:
    """Metre coordinates land inside the drawn panel, origin top-left."""
    td = TopDown(width=690)
    x0, y0 = td.to_px(0.0, 0.0)
    x1, y1 = td.to_px(PITCH_LEN, PITCH_WID)
    assert 0 <= x0 < x1 <= td.w
    assert 0 <= y0 < y1 <= td.h


# --- segment plan integrity -----------------------------------------------------------------------
def test_segment_plan_is_coherent_and_backed_by_real_footage() -> None:
    """Every planned segment points at an existing chunk video and a known section/overlay set."""
    for seg in SEGMENTS:
        assert seg.section in SECTION_TITLES
        assert seg.section in SECTION_CARDS
        assert seg.layout in ("video", "split")
        assert set(seg.overlays) <= KNOWN_OVERLAYS
        assert seg.end > seg.start
        video = VIDEO_ROOT / MATCH_ID / seg.half / f"chunk_{seg.chunk:03d}.mp4"
        assert video.exists(), f"missing footage for {seg}"


def test_demo_runs_about_five_minutes() -> None:
    """The stitched plan plus cards stays in the 4.5-6 minute showcase window."""
    live = sum(seg.end - seg.start + 1 for seg in SEGMENTS) / 25.0
    cards = 6.0 + 3.0 * len(SECTION_CARDS) + 13.0
    assert 270 <= live + cards <= 360
