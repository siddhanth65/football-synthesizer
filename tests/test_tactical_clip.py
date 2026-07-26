"""Targeted checks for the tactical-clip seams that are easy to get silently wrong.

Three of them, all pure functions with no artifacts on disk:

* **the match clock** -- a passage is addressed in seconds into a half, but the pixels live in
  per-chunk videos of unequal length. An off-by-one chunk boundary renders the wrong football.
* **role bands** -- the band is measured toward each team's *own* attacking goal, so the same pitch
  x must yield opposite bands for the two teams. Dropping the direction silently mirrors one team.
* **the refusal gate and the ghost filter** -- the two places where the tool is allowed to say no.
  A ghost drawn on top of a player we can already see is a re-identification duplicate, not a
  discovery (``results/B4_TRANSFER_M3.md``), and a passage with holes must be refused, not faked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.tactical_clip import (  # noqa: E402
    DUP_M,
    K50_ANC,
    K50_V1,
    K90_ANC,
    K90_V1,
    SQUAD,
    ChunkIndex,
    Ghost,
    Quality,
    _spread,
    block_class,
    phase_of,
    role_bands,
)


def _idx() -> ChunkIndex:
    """Two chunks of unequal length, the shape every real half has."""
    return ChunkIndex("h1", ("h1_chunk_000", "h1_chunk_001"), (0.0, 600.0), (600.0, 540.0),
                      (25.0, 25.0))


@pytest.mark.parametrize(("t_s", "expect"), [
    (0.0, (0, 0)),
    (599.9, (0, 14998)),
    (600.0, (1, 0)),          # the boundary belongs to the SECOND chunk, frame 0
    (610.0, (1, 250)),
    (1e9, (1, 13499)),        # clamped inside the half, never past the last frame
])
def test_chunk_clock_locates_the_right_frame(t_s: float, expect: tuple[int, int]) -> None:
    """Half-seconds map to the right (chunk, frame) across an uneven chunk boundary."""
    assert _idx().locate(t_s) == expect


def test_chunk_clock_round_trips() -> None:
    """``locate`` and ``to_half_s`` are inverses to within one source frame."""
    idx = _idx()
    for t in (0.0, 12.5, 599.0, 600.4, 1100.0):
        i, fr = idx.locate(t)
        assert abs(idx.to_half_s(i, fr) - t) <= 1.0 / 25.0


def _win() -> pd.DataFrame:
    """One keeper and three outfielders, two teams, spread along the pitch."""
    return pd.DataFrame({
        "track_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "team": [0, 0, 0, 0, 1, 1, 1, 1],
        "role": ["goalkeeper", "goalkeeper", *["player"] * 6],
        "is_keeper": [True, True, *[False] * 6],
        "pitch_x": [5.0, 6.0, 20.0, 22.0, 20.0, 22.0, 95.0, 96.0],
    })


def test_role_bands_follow_each_team_attack_direction() -> None:
    """The same pitch x is DEF for one team and ATT for the other -- direction is load-bearing."""
    assert role_bands(_win(), {0: 1, 1: -1}) == {1: "GK", 2: "DEF", 3: "ATT", 4: "DEF"}
    assert role_bands(_win(), {0: 1, 1: 1}) == {1: "GK", 2: "DEF", 3: "DEF", 4: "ATT"}


def test_role_bands_skip_teams_without_a_direction() -> None:
    """A team with no resolved attack direction gets no band rather than a guessed one."""
    assert set(role_bands(_win(), {0: 1})) == {1, 2}


@pytest.mark.parametrize(("kw", "head"), [
    ({"frame_cov": 0.5}, "broken passage"),
    ({"players": 5.0}, "tracking too sparse"),
    ({"ball_cov": 0.1}, "ball not tracked"),
    ({"both_teams": 0.1}, "only one team"),
])
def test_refusal_gate_names_the_failure(kw: dict[str, float], head: str) -> None:
    """Each gate refuses with its own reason; a clean passage passes."""
    good = {"players": 20.0, "ball_cov": 1.0, "both_teams": 1.0, "frame_cov": 1.0}
    assert Quality(**good).refusal() is None
    assert Quality(**{**good, **kw}).refusal().startswith(head)


def test_conformal_regions_are_ordered_and_frozen() -> None:
    """The 90% region is wider than the 50% in every bucket, for both emitted sources."""
    assert (K90_V1 > K50_V1).all()
    assert (K90_ANC > K50_ANC).all()
    assert len(K50_V1) == len(K90_V1) == len(K50_ANC) == len(K90_ANC) == 6


class _FakePassage:
    """Minimal stand-in exposing exactly what ``Passage.ghosts_at`` reads."""

    def __init__(self, ghosts: list[Ghost], live: list[dict]) -> None:
        self.ghosts = {100: ghosts}
        self._st = type("S", (), {"players": live})()

    def at(self, _frame: int):  # noqa: ANN202 - test double
        """Return the single fixed frame state."""
        return self._st


def _ghost(x: float, tsls: float, team: int = 0) -> Ghost:
    """A ghost at ``(x, 34)`` with trivially small regions."""
    return Ghost(tid=int(x), team=team, band="MID", pos=(x, 34.0), semi50=(1.0, 1.0),
                 semi90=(2.0, 2.0), source="v1", tsls=tsls)


def test_ghosts_drop_reid_duplicates_and_respect_the_squad_cap() -> None:
    """A ghost on top of a live team-mate is dropped; the rest are capped at the missing XI."""
    from tools.tactical_clip import Passage

    live = [{"team": 0, "pitch": (50.0, 34.0)}] * 9
    near = _ghost(50.0 + DUP_M / 2, 3.0)          # inside DUP_M of a live player -> a re-id
    far = [_ghost(10.0 + i, 1.0 + i) for i in range(5)]
    pas = _FakePassage([near, *far], live)
    out = Passage.ghosts_at(pas, 100)
    assert near not in out, "a ghost inside DUP_M of a live team-mate must not be drawn"
    assert len(out) == SQUAD - len(live), "at most (11 - observed) ghosts per team"
    assert [g.tsls for g in out] == sorted(g.tsls for g in out), "most-recently-seen first"


def test_ghosts_are_empty_when_the_nearest_sample_is_far_away() -> None:
    """No ghosts are invented across a broadcast cut."""
    from tools.tactical_clip import Passage

    pas = _FakePassage([_ghost(10.0, 1.0)], [])
    assert Passage.ghosts_at(pas, 100_000) == []


def test_spread_picks_distinct_situation_types() -> None:
    """``--auto N`` shows N different situations before it repeats one."""
    sl = pd.DataFrame([
        {"match": "a", "phase": "build-up", "block": "high", "score": 0.9},
        {"match": "a", "phase": "build-up", "block": "high", "score": 0.8},
        {"match": "a", "phase": "attack", "block": "low", "score": 0.7},
    ])
    assert [r["score"] for r in _spread(sl, 2)] == [0.9, 0.7]
    assert len(_spread(sl, 3)) == 3
    assert _spread(sl, 0) == []
    assert _spread(pd.DataFrame(), 3) == []


@pytest.mark.parametrize(("line_m", "expect"), [
    (20.0, "low"), (40.0, "mid"), (60.0, "high"),
    (-7.0, None),        # the real fulham_manutd read: a de-bias that lands behind the goal line
    (140.0, None), (float("nan"), None),
])
def test_block_class_refuses_a_line_off_the_pitch(line_m: float, expect: str | None) -> None:
    """An out-of-range de-biased line is a failed measurement, not a low block."""
    assert block_class(line_m) == expect


def test_phase_needs_stable_possession() -> None:
    """A corner-style scramble is labelled contested, not 'build-up' for whoever led the flips."""
    assert phase_of(10.0, 0.95, 1) == ("build-up", 1)
    assert phase_of(50.0, 0.95, 0) == ("progression", 0)
    assert phase_of(90.0, 0.95, 0) == ("attack", 0)
    label, team = phase_of(10.0, 0.51, 1)
    assert team is None and label.startswith("contested")


def test_ellipse_semi_axes_grow_with_the_horizon() -> None:
    """A longer occlusion must draw a bigger region -- the visual claim of the whole clip."""
    w = np.array([2.0, 1.0])
    short = K90_V1[1] * w
    long = K90_V1[5] * w
    assert (long > short).all()
