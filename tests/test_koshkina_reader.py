"""Unit tests for the Koshkina reader's pure seams (positional-softmax -> jersey distribution).

Covers only CPU-pure logic: the PARSeq JSON handoff schema (:func:`parseq_positions_to_probs`) and
the torso-crop keypoint gate. GPU inference (legibility / pose / PARSeq) is not unit-tested.
"""

from __future__ import annotations

import numpy as np

from generator.jersey_id import (
    ILLEGIBLE,
    NUM_CLASSES,
    decide,
    parseq_positions_to_probs,
    roster_mask,
    torso_from_keypoints,
)

# Token layout the sidecar emits: index 0 = [E]/end, 1..10 = digit 0..9.
_E = 0


def _onehot(idx: int) -> list[float]:
    v = [0.0] * 11
    v[idx] = 1.0
    return v


def test_probs_shape_and_normalization() -> None:
    """Any positional pair yields a NUM_CLASSES vector summing to ~1."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        p0 = rng.dirichlet(np.ones(11))
        p1 = rng.dirichlet(np.ones(11))
        out = parseq_positions_to_probs(p0, p1)
        assert out.shape == (NUM_CLASSES,)
        assert abs(float(out.sum()) - 1.0) < 1e-5


def test_single_digit_read() -> None:
    """pos0='8' (token 9), pos1=E -> number 8 wins."""
    out = parseq_positions_to_probs(_onehot(9), _onehot(_E))
    num, conf = decide(out, min_conf=0.0)
    assert num == 8 and conf > 0.99


def test_two_digit_read() -> None:
    """pos0='2' (token 3), pos1='0' (token 1) -> number 20 wins (the Dalot back-read)."""
    out = parseq_positions_to_probs(_onehot(3), _onehot(1))
    num, conf = decide(out, min_conf=0.0)
    assert num == 20 and conf > 0.99


def test_empty_and_leading_zero_read_illegible() -> None:
    """pos0=E -> illegible; pos0='0' (leading zero, token 1) -> illegible (not a 1..99 number)."""
    for tok in (_E, 1):
        out = parseq_positions_to_probs(_onehot(tok), _onehot(5))
        assert int(out.argmax()) == ILLEGIBLE
        assert decide(out, min_conf=0.0)[0] == -1


def test_roster_mask_redistributes_offroster_mass() -> None:
    """An off-roster peak is masked; the second-choice on-roster number then clears the floor.

    pos0 mass split 55% on '7'(token 8) / 45% on '2'(token 3); pos1='0'(token 1). Raw peak is 70
    (off a roster of {8, 20}); masking to the roster leaves 20 as the winner with lifted confidence.
    """
    p0 = [0.0] * 11
    p0[8] = 0.55  # digit 7
    p0[3] = 0.45  # digit 2
    out = parseq_positions_to_probs(p0, _onehot(1))  # pos1 = digit 0
    assert decide(out, min_conf=0.0)[0] == 70  # raw: 70 wins
    mask = roster_mask([8, 20])
    num, conf = decide(out, min_conf=0.0, mask=mask)
    assert num == 20 and conf > 0.99  # 70 removed, all mass on the sole valid two-digit read


def test_torso_gate_rejects_missing_and_crops_valid() -> None:
    """No keypoints -> None; a full COCO-17 set crops the shoulder-hip band from the image."""
    img = np.zeros((100, 60, 3), dtype=np.uint8)
    assert torso_from_keypoints(img, None) is None
    assert torso_from_keypoints(img, [[0, 0, 1.0]] * 5) is None  # < 12 keypoints
    kp = [[0.0, 0.0, 1.0]] * 17
    kp[6] = [40.0, 25.0, 1.0]  # right shoulder
    kp[5] = [15.0, 25.0, 1.0]  # left shoulder
    kp[11] = [18.0, 70.0, 1.0]  # left hip
    kp[12] = [38.0, 70.0, 1.0]  # right hip
    crop = torso_from_keypoints(img, kp)
    assert crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("koshkina reader seam tests OK")
