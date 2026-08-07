"""Tests for the Dirichlet evidential jersey head and its path through the per-crop evidence."""

from __future__ import annotations

import numpy as np
import torch

from eval.gsr_jersey import percrop_frame
from generator.evidential_jersey import (
    NUM_CLASSES,
    EvidentialGateHead,
    dirichlet_uncertainty,
    edl_loss,
    fuse_tracklet,
    kl_dirichlet_uniform,
    player_disjoint_split,
    roc_auc,
    trunk_dist,
)
from generator.jersey_id import parseq_positions_to_probs
from tools.ocr_density import crop_alpha, crop_probs_from_percrop, reads_for_sequence


def test_dirichlet_primitives() -> None:
    """Uniform evidence is zero-KL and maximally uncertain; concentrated evidence is neither."""
    flat = torch.ones(1, 4)
    assert abs(float(kl_dirichlet_uniform(flat))) < 1e-6
    assert abs(float(dirichlet_uncertainty(flat)) - 1.0) < 1e-6
    sharp = torch.tensor([[9.0, 1.0, 1.0, 1.0]])
    assert float(kl_dirichlet_uniform(sharp)) > 0.5
    assert float(dirichlet_uncertainty(sharp)) < float(dirichlet_uncertainty(flat))


def test_edl_loss_prefers_the_truth_and_penalises_wrong_evidence() -> None:
    """Evidence on the right class lowers the loss; the KL term only punishes the wrong ones."""
    y = torch.tensor([0])
    right = torch.tensor([[9.0, 1.0, 1.0, 1.0]])
    wrong = torch.tensor([[1.0, 9.0, 1.0, 1.0]])
    assert float(edl_loss(right, y)) < float(edl_loss(torch.ones(1, 4), y))
    assert float(edl_loss(wrong, y)) > float(edl_loss(torch.ones(1, 4), y))
    # a correct, confident row is not punished by the regulariser
    assert abs(float(edl_loss(right, y, 1.0)) - float(edl_loss(right, y, 0.0))) < 1e-6


def test_trunk_dist_matches_the_shipped_numpy_fold() -> None:
    """The torch fold of PARSeq's positional softmaxes equals generator.jersey_id's."""
    rng = np.random.default_rng(3)
    raw = rng.random((6, 3, 11)).astype(np.float32)
    pos = (raw / raw.sum(-1, keepdims=True)).astype(np.float32)
    feats = np.concatenate([rng.random((6, 20)).astype(np.float32), pos.reshape(6, 33)], 1)
    got = trunk_dist(torch.as_tensor(feats)).numpy()
    want = np.stack([parseq_positions_to_probs(r[0], r[1]) for r in pos])
    assert np.abs(got - want).max() < 1e-6


def test_gate_head_cannot_regress_the_trunk_ranking() -> None:
    """The gate head's number argmax IS the frozen trunk's, whatever its weights say."""
    rng = np.random.default_rng(11)
    raw = rng.random((8, 3, 11)).astype(np.float32)
    pos = (raw / raw.sum(-1, keepdims=True)).astype(np.float32)
    feats = np.concatenate([rng.random((8, 20)).astype(np.float32), pos.reshape(8, 33)], 1)
    want = np.stack([parseq_positions_to_probs(r[0], r[1]) for r in pos])
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        alpha = EvidentialGateHead(feats.shape[1], hidden=8)(torch.as_tensor(feats)).detach().numpy()
        assert np.array_equal(alpha[:, 1:].argmax(1), want[:, 1:].argmax(1))
        assert (alpha >= 1.0).all()


def test_fuse_tracklet_accumulates_and_filters() -> None:
    """Evidence adds across crops, the uncertainty filter bites, and no evidence means abstain."""
    one = np.ones((1, NUM_CLASSES), np.float32)
    one[:, 7] += 20.0
    three = np.repeat(one, 3, axis=0)
    assert fuse_tracklet(one)[0][0] == 7
    assert fuse_tracklet(three)[0][1] > fuse_tracklet(one)[0][1]
    assert fuse_tracklet(three, max_u=NUM_CLASSES / three.sum(1)[0] * 0.99) == []
    assert fuse_tracklet(three, min_crops=4) == []
    assert fuse_tracklet(np.ones((3, NUM_CLASSES), np.float32)) == []
    assert fuse_tracklet(np.empty((0, NUM_CLASSES), np.float32)) == []


def test_percrop_frame_persists_alpha_and_the_chain_reads_it_back() -> None:
    """An evidential per-crop frame round-trips to the Dirichlet MEAN through the shipped path."""
    from pathlib import Path  # noqa: PLC0415

    n = 4
    probs = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES, np.float32)
    alpha = np.ones((n, NUM_CLASSES), np.float32)
    alpha[:, 9] += 49.0
    detail = {"leg": np.full(n, 0.9, np.float32), "torso": np.ones(n, bool),
              "p0": np.zeros((n, 11), np.float32), "p1": np.zeros((n, 11), np.float32),
              "alpha": alpha}
    flat = [Path(f"t1_{i:06d}.jpg") for i in range(n)]
    frame = percrop_frame({}, probs, detail, [1] * n, flat)
    assert "alpha" in frame.columns
    assert np.allclose(frame["u"].to_numpy(), NUM_CLASSES / alpha.sum(1))
    assert np.allclose(crop_alpha(frame), alpha)
    back = crop_probs_from_percrop(frame)
    assert np.allclose(back, alpha / alpha.sum(1, keepdims=True), atol=1e-6)
    rule = {"min_crop_conf": 0.1, "min_votes": 1, "min_legibility": 0.0, "emit_all": False,
            "fuse": True, "max_u": 1.01, "max_p_none": 1.01}
    assert reads_for_sequence(frame, rule)[1][0][0] == 9


def test_percrop_frame_without_a_head_is_unchanged() -> None:
    """No head -> no new columns, and the folded-softmax path is the one that runs."""
    from pathlib import Path  # noqa: PLC0415

    n = 3
    probs = np.zeros((n, NUM_CLASSES), np.float32)
    probs[:, 0] = 1.0
    detail = {"leg": np.full(n, 0.1, np.float32), "torso": np.zeros(n, bool),
              "p0": np.full((n, 11), np.nan, np.float32),
              "p1": np.full((n, 11), np.nan, np.float32)}
    frame = percrop_frame({}, probs, detail, [1] * n, [Path(f"t1_{i:06d}.jpg") for i in range(n)])
    assert "alpha" not in frame.columns and "u" not in frame.columns
    assert crop_alpha(frame) is None
    assert np.allclose(crop_probs_from_percrop(frame)[:, 0], 1.0)


def test_player_disjoint_split_and_auc() -> None:
    """The split never leaks a player, and the AUC is 0.5 on a constant score."""
    keys = np.array([f"p{i % 40}" for i in range(400)])
    hold = player_disjoint_split(keys, 0.10, seed=0)
    assert set(keys[hold]).isdisjoint(set(keys[~hold]))
    assert 0.0 < hold.mean() < 0.3
    assert abs(roc_auc(np.zeros(20), np.arange(20) < 10) - 0.5) < 1e-9
    assert roc_auc(np.arange(20.0), np.arange(20) >= 10) == 1.0
