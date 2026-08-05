"""Clean-room ExpansionIoU association: the self-check plus the two contract invariants."""

from __future__ import annotations

import numpy as np

from tools.gsr_eiou import EiouParams, _demo, associate, eiou_matrix


def test_demo_self_check() -> None:
    """The paper's motivating case: plain IoU fails across the gap, EIoU at e=0.7 recovers it."""
    _demo()


def test_eiou_is_monotone_in_expansion() -> None:
    """Expanding both boxes can only raise their overlap, never lower it."""
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.uniform(0, 500, 20), rng.uniform(0, 500, 20)])
    boxes = np.column_stack([a, a + rng.uniform(10, 60, (20, 2))])
    prev = eiou_matrix(boxes[:10], boxes[10:], 0.0)
    for e in (0.2, 0.5, 1.0, 2.0):
        cur = eiou_matrix(boxes[:10], boxes[10:], e)
        assert (cur >= prev - 1e-9).all()
        prev = cur


def test_every_detection_gets_an_id() -> None:
    """No row may be dropped: GS-DetA must stay comparable to the control's."""
    rng = np.random.default_rng(1)
    frames = np.repeat(np.arange(40), 6)
    x = 30.0 * np.tile(np.arange(6), 40) + 2.0 * np.repeat(np.arange(40), 6)
    boxes = np.column_stack([x, np.zeros_like(x), x + 20.0, np.full_like(x, 40.0)])
    ids = associate(frames, boxes, None, None, EiouParams())
    assert (ids >= 1).all() and len(ids) == len(frames)
    # Six well-separated players moving together are six tracks, not sixty.
    assert len(set(ids.tolist())) == 6, sorted(set(ids.tolist()))
    again = associate(frames, boxes, None, None, EiouParams())
    assert (again == ids).all(), "association must be deterministic"
    del rng
