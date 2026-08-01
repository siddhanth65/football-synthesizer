"""Tests for the pure GTA-Link splitter/connector seams (no CV model, no metric stack).

Thin on purpose: :func:`generator.gta_link._demo` already asserts the full synthetic scenario, so
this pins it in CI plus the one behaviour that actually broke in practice -- a 1-frame temporal
overlap must NOT survive, because the SoccerNet evaluator rejects a submission that repeats a track
id inside one timestep.
"""

from __future__ import annotations

import numpy as np

from generator.gta_link import GtaParams, Fragment, _demo, connect, split_track


def test_demo_selfcheck_passes():
    _demo()


def test_connector_rejects_single_frame_overlap_by_default():
    e = {1: np.array([1.0, 0.0], np.float32), 2: np.array([1.0, 0.0], np.float32)}
    a = Fragment(1, 0, 10, (0.0, 0.0), (1.0, 0.0), "player", 0, 11)
    b = Fragment(2, 10, 20, (1.0, 0.0), (2.0, 0.0), "player", 0, 11)  # shares frame 10
    assert connect([a, b], e, {1: 5, 2: 5}, GtaParams())[2] == 2
    assert connect([a, b], e, {1: 5, 2: 5}, GtaParams(overlap_slack=1))[2] == 1


def test_splitter_is_a_noop_on_a_single_identity():
    rng = np.random.default_rng(1)
    frames = np.arange(80)
    v = np.zeros(8)
    v[0] = 1.0
    embs = v + 0.01 * rng.normal(size=(80, 8))
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    assert split_track(frames, embs, eps=0.04, min_samples=5, min_run=5).max() == 0


def test_loto_grades_against_the_anchor_truth_not_the_arms_own_label(tmp_path):
    """A relabelled gallery must NOT score itself correct (the self-consistency trap)."""
    import pandas as pd

    from tools.gta_carrier import loto_query_hits

    # Two crops of team 0 in one chunk: track 1 is the query (anchor truth "A"), track 2 is the
    # only other gallery entry. Their embeddings are identical, so track 2 is always the top-1.
    gal = pd.DataFrame([
        {"chunk": "c0", "frame": 0, "track_id": 1, "src_track": 1, "team": 0, "player": "A"},
        {"chunk": "c0", "frame": 9, "track_id": 2, "src_track": 2, "team": 0, "player": "A"},
    ])
    emb = np.tile(np.array([[1.0, 0.0]], np.float32), (2, 1))
    gal.to_parquet(tmp_path / "m_gallery.parquet", index=False)
    np.save(tmp_path / "m_gallery_emb.npy", emb)
    assert loto_query_hits(tmp_path, "m", {("c0", 1): "A"}) == {("c0", 1, 0): 1}

    # Same geometry, but the arm has relabelled every crop "Z". Self-consistency would score 1;
    # grading against the anchor truth "A" must score 0.
    gal["player"] = "Z"
    gal.to_parquet(tmp_path / "m_gallery.parquet", index=False)
    assert loto_query_hits(tmp_path, "m", {("c0", 1): "A"}) == {("c0", 1, 0): 0}
