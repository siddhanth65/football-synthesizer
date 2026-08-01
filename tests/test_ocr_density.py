"""Tests for the per-crop OCR evidence path (persistence + the densifying aggregation)."""

from __future__ import annotations

import numpy as np

from eval.gsr_jersey import percrop_frame
from generator.jersey_id import (
    ILLEGIBLE,
    NUM_CLASSES,
    aggregate_votes,
    parseq_positions_to_probs,
    percrop_votes,
)
from tools.ocr_density import crop_probs_from_percrop, measure


def _crops(n_illegible: int, reads: list[tuple[int, float]]) -> np.ndarray:
    """Build ``[n, NUM_CLASSES]`` rows: illegible one-hots plus confident single-number reads."""
    rows = []
    for _ in range(n_illegible):
        r = np.zeros(NUM_CLASSES, np.float32)
        r[ILLEGIBLE] = 1.0
        rows.append(r)
    for num, conf in reads:
        r = np.full(NUM_CLASSES, (1.0 - conf) / (NUM_CLASSES - 1), np.float32)
        r[num] = conf
        rows.append(r)
    return np.stack(rows)


def test_illegible_crops_veto_the_shipped_aggregation() -> None:
    """The measured 0.087 density defect: 3 confident reads among 16 crops abstain under the mean."""
    probs = _crops(13, [(7, 0.99)] * 3)
    assert aggregate_votes(probs) == (-1, aggregate_votes(probs)[1])
    assert aggregate_votes(probs)[0] == -1
    # ... and are recovered when crops that read nothing simply do not vote.
    votes = percrop_votes(probs, min_crop_conf=0.5, min_votes=2)
    assert [n for n, _c in votes] == [7]


def test_min_votes_and_emit_all() -> None:
    """``min_votes`` suppresses singletons; ``emit_all`` preserves disagreement for the solver."""
    probs = _crops(4, [(7, 0.95)] * 3 + [(1, 0.95)] * 2 + [(23, 0.95)])
    assert [n for n, _c in percrop_votes(probs, min_votes=2)] == [7]
    assert sorted(n for n, _c in percrop_votes(probs, min_votes=2, emit_all=True)) == [1, 7]
    assert sorted(n for n, _c in percrop_votes(probs, min_votes=3, emit_all=True)) == [7]
    assert percrop_votes(probs, min_crop_conf=0.99) == []
    assert percrop_votes(np.empty((0, NUM_CLASSES), np.float32)) == []


def test_percrop_frame_round_trips_the_distributions() -> None:
    """Persisted ``p0``/``p1`` rebuild the folded distribution exactly; no-read crops stay illegible."""
    p0 = np.zeros((2, 11), np.float32)
    p1 = np.zeros((2, 11), np.float32)
    p0[0, 8] = 1.0  # digit 7 at position 0
    p1[0, 0] = 1.0  # end token at position 1 -> single-digit "7"
    p0[1] = np.nan
    p1[1] = np.nan
    probs = np.zeros((2, NUM_CLASSES), np.float32)
    probs[:, ILLEGIBLE] = 1.0
    probs[0] = parseq_positions_to_probs(p0[0], p1[0])
    detail = {"leg": np.array([0.9, 0.1], np.float32), "torso": np.array([True, False]),
              "p0": p0, "p1": p1}
    from pathlib import Path  # noqa: PLC0415

    flat = [Path("t3_000012.jpg"), Path("t3_000030.jpg")]
    df = percrop_frame({}, probs, detail, [3, 3], flat)
    assert df["frame"].tolist() == [12, 30]
    assert df["number"].tolist() == [7, -1]
    assert df["track_id"].tolist() == [3, 3]
    rebuilt = crop_probs_from_percrop(df)
    assert np.allclose(rebuilt, probs, atol=1e-6)


def test_vote_cache_accepts_one_or_many_votes_per_track(tmp_path) -> None:  # noqa: ANN001
    """The solver's vote loader reads the on-record single-vote and the densified many-vote form."""
    import json  # noqa: PLC0415

    from eval.gsr_identity import _load_votes  # noqa: PLC0415

    old = tmp_path / "old.json"
    old.write_text(json.dumps({"votes": {"3": [7, 0.9], "4": [-1, 0.1]}}), encoding="utf-8")
    assert _load_votes(old) == {3: [(7, 0.9)], 4: [(-1, 0.1)]}
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"votes": {"3": [[7, 0.9], [1, 0.6]]}}), encoding="utf-8")
    assert _load_votes(new) == {3: [(7, 0.9), (1, 0.6)]}


def test_measure_counts_density_and_precision() -> None:
    """``d`` is per track over all tracks; precision is graded only on numbered GT players."""
    import pandas as pd  # noqa: PLC0415

    probs = np.concatenate([_crops(0, [(7, 0.99)] * 2), _crops(0, [(9, 0.99)] * 2),
                            _crops(3, [])])
    detail = {"leg": np.ones(7, np.float32), "torso": np.ones(7, bool),
              "p0": np.full((7, 11), np.nan, np.float32),
              "p1": np.full((7, 11), np.nan, np.float32)}
    from pathlib import Path  # noqa: PLC0415

    flat = [Path(f"t1_{i:06d}.jpg") for i in range(2)] + \
           [Path(f"t2_{i:06d}.jpg") for i in range(2)] + \
           [Path(f"t3_{i:06d}.jpg") for i in range(3)]
    df = percrop_frame({}, probs, detail, [1, 1, 2, 2, 3, 3, 3], flat)
    # p0/p1 are NaN, so rebuild-from-parquet would lose the reads: grade the frame's own numbers.
    df = df.assign(p0=list(np.eye(11, dtype=np.float32)[[8, 8, 10, 10, 0, 0, 0]]),
                   p1=list(np.eye(11, dtype=np.float32)[[0] * 7]))
    gt = {"S": {1: "7", 2: "8", 3: "5"}}  # track 1 right, track 2 wrong, track 3 never reads
    res = measure({"S": pd.DataFrame(df)}, gt, {"min_crop_conf": 0.5, "min_votes": 1,
                                                "emit_all": False})
    assert res["n_tracks"] == 3
    assert res["n_tracks_read"] == 2
    assert res["d"] == 2 / 3
    assert res["read_precision"] == 0.5
