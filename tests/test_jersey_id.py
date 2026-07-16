"""Pure-seam tests for jersey ID: label mapping and tracklet voting (no GPU, no dataset)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from generator import jersey_id as J


def _onehot(cls: int, peak: float = 0.9) -> np.ndarray:
    """A single crop's softmax row peaking on ``cls`` with the rest spread uniformly."""
    row = np.full(J.NUM_CLASSES, (1 - peak) / (J.NUM_CLASSES - 1), dtype=np.float32)
    row[cls] = peak
    return row


def test_label_class_roundtrip() -> None:
    assert J.to_class(-1) == 0
    assert J.from_class(0) == -1
    for n in (1, 7, 10, 99):
        assert J.to_class(n) == n
        assert J.from_class(n) == n


def test_to_class_out_of_range() -> None:
    for bad in (0, 100, -2):
        with pytest.raises(ValueError):
            J.to_class(bad)


def test_load_gt(tmp_path) -> None:
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"0": 10, "1": -1, "2": 99}), encoding="utf-8")
    assert J.load_gt(p) == {"0": 10, "1": -1, "2": 99}


def test_vote_consistent_number_wins() -> None:
    probs = np.stack([_onehot(10) for _ in range(5)])
    label, conf = J.aggregate_votes(probs, min_conf=0.3)
    assert label == 10
    assert conf > 0.3


def test_vote_illegible_class_reads_minus_one() -> None:
    probs = np.stack([_onehot(J.ILLEGIBLE) for _ in range(5)])
    label, _ = J.aggregate_votes(probs, min_conf=0.3)
    assert label == -1


def test_vote_low_confidence_falls_to_minus_one() -> None:
    # A number peak that is real but below threshold -> abstain to -1.
    probs = np.stack([_onehot(7, peak=0.2) for _ in range(5)])
    label, conf = J.aggregate_votes(probs, min_conf=0.5)
    assert label == -1
    assert conf < 0.5


def test_vote_noise_cancels_true_number_survives() -> None:
    # Two crops show '8' confidently; three back-view crops scatter over different wrong classes.
    rng = np.random.default_rng(0)
    noise = [_onehot(int(rng.integers(20, 99)), peak=0.35) for _ in range(3)]
    probs = np.stack([_onehot(8), _onehot(8), *noise])
    label, _ = J.aggregate_votes(probs, min_conf=0.3)
    assert label == 8


def test_empty_tracklet() -> None:
    assert J.aggregate_votes(np.empty((0, J.NUM_CLASSES))) == (-1, 0.0)


def test_weighted_downweights_diffuse_crops() -> None:
    # One sharp '5' crop plus many weakly-'6' crops: weighting favors the confident crop.
    sharp = _onehot(5, peak=0.95)
    weak = [_onehot(6, peak=0.12) for _ in range(6)]
    probs = np.stack([sharp, *weak])
    assert J.aggregate_votes(probs, min_conf=0.2, weighted=True)[0] == 5


def test_decide_matches_aggregate() -> None:
    probs = np.stack([_onehot(23) for _ in range(4)])
    mean_p = J.tracklet_mean(probs)
    assert J.decide(mean_p, min_conf=0.3) == J.aggregate_votes(probs, min_conf=0.3)
