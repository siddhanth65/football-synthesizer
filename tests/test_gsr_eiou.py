"""Clean-room ExpansionIoU association: the self-check plus the two contract invariants."""

from __future__ import annotations

import numpy as np

from tools.gsr_eiou import FLOOR, VARIANT, EiouParams, _demo, associate, eiou_matrix, set_embedder


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


def test_percrop_variant_never_collides_with_the_shipped_arm(tmp_path) -> None:  # noqa: ANN001
    """A reader-swap arm must key its own bundle/votes caches, or it silently reuses the control's."""
    import eval.gsr_gta as gta  # noqa: PLC0415

    from tools.gsr_v4 import config_key, votes_dir  # noqa: PLC0415

    before = (gta.EMBEDDER, gta.CACHE_SUBDIR)
    try:
        set_embedder("clip" + VARIANT)
        shipped, swapped = VARIANT, "_v6" + VARIANT
        assert config_key(FLOOR, 0.45, True, shipped, False) != config_key(
            FLOOR, 0.45, True, swapped, False)
        # The vote cache is sourced from the matching per-crop directory and named for it.
        assert votes_dir(tmp_path, FLOOR, swapped).name == "koshkina_percrop_votes_f080_v6_eiou"
        assert votes_dir(tmp_path, FLOOR, shipped).name == "koshkina_percrop_votes_f080_eiou"
    finally:
        gta.EMBEDDER, gta.CACHE_SUBDIR = before


def test_votes_cache_is_invalidated_when_the_rule_changes(tmp_path) -> None:  # noqa: ANN001
    """S3 arms A and B shared a vote-cache path: a stale rule must wipe the cache, not be reused."""
    import json  # noqa: PLC0415

    from tools.gsr_v4 import rule_for, votes_dir  # noqa: PLC0415

    dest = votes_dir(tmp_path, FLOOR, "")            # cold: writes the rule stamp, no evidence
    stale = dest / "SNGS-000.json"
    stale.write_text("{}", encoding="utf-8")
    assert votes_dir(tmp_path, FLOOR, "") == dest and stale.exists(), "same rule must reuse"
    (dest / "_rule.json").write_text(json.dumps({**rule_for(FLOOR), "min_votes": 99}),
                                     encoding="utf-8")
    votes_dir(tmp_path, FLOOR, "")
    assert not stale.exists(), "a rule change must invalidate the cached votes"
