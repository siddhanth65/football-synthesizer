"""Targeted checks for the v10-W9 duplicate-concurrent-track absorber."""

from __future__ import annotations

import numpy as np

from eval.gsr_score import accepts_pair, dedup_absorb, pair_features, vote_track_attributes

PARAMS = {"n_ov": 5, "d_med": 1.0, "cos": None, "same_team": False, "min_track_rows": 25}


def _row(tid: int, f: int, x: float, conf: float = 0.9, team: str = "left") -> dict:
    return {"image_id": f"{f:06d}", "track_id": tid, "confidence": conf,
            "attributes": {"role": "player", "team": team, "jersey": None},
            "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": 0.0}}


def test_absorb_drops_overlap_and_relinks_remainder() -> None:
    """The loser's overlapping rows go; its disjoint rows join the winner's id."""
    preds = [_row(1, f, 0.0) for f in range(40)] + [_row(2, f, 0.2, 0.8) for f in range(20, 60)]
    st = dedup_absorb(preds, PARAMS)
    assert (st["pairs"], st["components"], st["tracks_absorbed"]) == (1, 1, 1)
    assert (st["rows_dropped"], st["rows_relabelled"]) == (20, 20)
    assert len(preds) == 60
    assert {p["track_id"] for p in preds} == {1}
    assert len({p["image_id"] for p in preds}) == 60, "the evaluator forbids two rows of one id"


def test_off_is_a_no_op() -> None:
    """A None config must not touch a single row (the shipped default)."""
    preds = [_row(1, f, 0.0) for f in range(40)] + [_row(2, f, 0.1) for f in range(40)]
    assert dedup_absorb(preds, None) == {}
    assert len(preds) == 80 and {p["track_id"] for p in preds} == {1, 2}


def test_geometry_and_length_gates_bind() -> None:
    """Distant tracks and short tracks are never duplicates."""
    far = [_row(1, f, 0.0) for f in range(40)] + [_row(2, f, 3.0) for f in range(40)]
    assert dedup_absorb(far, PARAMS)["pairs"] == 0
    short = [_row(1, f, 0.0) for f in range(40)] + [_row(2, f, 0.1) for f in range(10)]
    assert dedup_absorb(short, PARAMS)["pairs"] == 0


def test_appearance_degrades_gracefully() -> None:
    """A missing embedding falls back to geometry; a present one can veto or confirm."""
    strict = {**PARAMS, "cos": 0.8}
    base = {"a": 1, "b": 2, "n_ov": 30, "d_med": 0.3, "cos": float("nan"), "same_team": 1}
    assert accepts_pair(base, strict)
    assert not accepts_pair({**base, "cos": 0.5}, strict)
    assert accepts_pair({**base, "cos": 0.9}, strict)


def test_pair_features_ignore_non_finite_positions() -> None:
    """Dead-frame rows carry NaN pitch coordinates and must not make a pair look coincident."""
    info = {1: {"pos": {f: (np.nan, np.nan) for f in range(30)}, "team": "left"},
            2: {"pos": {f: (0.0, 0.0) for f in range(30)}, "team": "left"}}
    assert pair_features(info, {}, 25) == []
    info[1]["pos"][0] = (0.1, 0.0)
    feats = pair_features(info, {}, 25)
    assert len(feats) == 1 and feats[0]["n_ov"] == 1 and feats[0]["d_med"] == 0.1


def test_absorb_runs_before_the_vote_and_changes_what_it_votes_on() -> None:
    """The placement is load-bearing: a merged track votes one jersey, two tracks vote two."""
    preds = ([_row(1, f, 0.0) for f in range(40)] + [_row(2, f, 0.2, 0.8) for f in range(20, 60)])
    for p in preds:
        p["attributes"]["jersey"] = "7" if p["track_id"] == 1 else "9"
    dedup_absorb(preds, PARAMS)
    vote_track_attributes(preds, ("jersey",))
    assert {p["attributes"]["jersey"] for p in preds} == {"7"}
