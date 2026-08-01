"""Deterministic tests for the evidence-density simulator's two invented pieces.

The solver is Stage 2's and already has ``tests/test_identity_solve.py``; what is new here is the
fragmenter (does it tile the truth, and does it hit its measured target?) and the mention binder
(does a lagged commentary name land on the wrong tracklet, as it must?).
"""

from __future__ import annotations

import numpy as np

from tools.evidence_sim import (
    FPS,
    SimConfig,
    bind_mentions,
    calibrate_fragmentation,
    draw_spans,
    fragment,
    ocr_reads,
)
from tools.footpass_digest import visibility_runs


def _half() -> dict:
    """Two players, one on-screen run each, one action event per second, straight-line motion."""
    n_steps, stride, f0 = 400, 10, 0
    pos = np.zeros((n_steps, 2, 2), np.float32)
    pos[:, 0, 0] = np.linspace(0.0, 1.0, n_steps)          # player 0 crosses the pitch
    pos[:, 1, 0] = np.linspace(1.0, 0.0, n_steps)          # player 1 crosses the other way
    pos[:, :, 1] = 0.5
    frames = np.arange(100, 3900, 250)
    events = np.column_stack([frames, np.arange(frames.size) % 2, np.ones(frames.size, int)])
    return {"key": "test_H1", "shirt": np.array([7, 9]), "role": np.array([2, 3]),
            "team": np.array([1, 2]), "window": np.array([[0, 3999], [0, 3999]]),
            "runs": np.array([[0, 0, 3999], [1, 0, 3999]]),
            "events": events, "pos": pos, "grid": np.array([f0, stride])}


def test_visibility_runs_split_on_gaps() -> None:
    frames = np.array([0, 1, 2, 3, 4, 7, 8, 9])
    vis = np.array([True, True, False, True, True, True, False, True])
    assert visibility_runs(frames, vis).tolist() == [[0, 1], [3, 4], [7, 7], [9, 9]]
    assert visibility_runs(frames, np.zeros(8, bool)).shape == (0, 2)


def test_fragmenter_tiles_the_runs_without_gaps_or_overlaps() -> None:
    rng = np.random.default_rng(0)
    runs = np.array([[0, 100, 999], [0, 2000, 2400], [1, 0, 3999]])
    trk = fragment(runs, (0, 3999), rate=4.72, scale=1.0, rng=rng)
    for p, a, b in runs:
        mine = trk[trk[:, 0] == p]
        mine = mine[(mine[:, 1] >= a) & (mine[:, 2] <= b)]
        cover = sorted(mine[:, 1:3].tolist())
        assert cover[0][0] == a and cover[-1][1] == b
        assert all(y[0] == x[1] + 1 for x, y in zip(cover, cover[1:]))
    assert (trk[:, 3] == trk[:, 2] - trk[:, 1] + 1).all()


def test_fragmenter_clips_to_the_chunk_and_merges_when_rate_is_zero() -> None:
    rng = np.random.default_rng(1)
    runs = np.array([[0, 0, 3999]])
    trk = fragment(runs, (1000, 1999), rate=4.72, scale=1.0, rng=rng)
    assert trk[:, 1].min() == 1000 and trk[:, 2].max() == 1999
    flat = fragment(np.array([[0, 0, 99], [0, 500, 599]]), (0, 3999), 0.0, 1.0, rng)
    assert flat.tolist() == [[0, 0, 599, 200]]  # one tracklet, but only 200 on-screen frames


def test_fragmentation_hits_its_calibrated_target() -> None:
    half = _half()
    for rate in (4.72, 6.74):
        scale = calibrate_fragmentation([half], rate, 3000)
        rng = np.random.default_rng(3)
        n = vis = 0
        for lo in (0, 3000):
            t = fragment(half["runs"], (lo, min(lo + 2999, 3999)), rate, scale, rng)
            n += t.shape[0]
            vis += int(t[:, 3].sum())
        assert abs(n / (vis / FPS / 30.0) - rate) < 0.25 * rate


def test_drawn_spans_match_the_measured_gsr_shape() -> None:
    sp = draw_spans(np.random.default_rng(4), 100_000, 1.0)
    for q, want in ((25, 20.0), (50, 105.0), (75, 343.0), (90, 635.1)):
        assert abs(np.percentile(sp, q) - want) / want < 0.15


def test_reads_hit_the_requested_density_and_prefer_long_tracklets() -> None:
    rng = np.random.default_rng(5)
    lens = np.concatenate([np.full(500, 20), np.full(500, 400)])
    trk = np.column_stack([np.zeros(1000, int), np.zeros(1000, int), lens - 1, lens])
    cfg = SimConfig(ocr_density=0.20, ocr_precision=1.0)
    got = [ocr_reads(trk, cfg, rng, 2) for _ in range(20)]
    dens = np.mean([len(r) / 1000 for r in got])
    long_share = np.mean([np.mean([t >= 500 for t, _, _ in r]) for r in got])
    assert abs(dens - 0.20) < 0.02, dens
    assert 0.5 < long_share < 0.95, long_share      # biased to long tracklets, not degenerate


def test_mention_binds_to_the_actor_at_zero_lag_and_drifts_with_lag() -> None:
    half = _half()
    rng = np.random.default_rng(6)
    trk = fragment(half["runs"], (0, 3999), rate=0.0, scale=1.0, rng=rng)

    def hit_rate(lag: float, prec: float) -> float:
        r = np.random.default_rng(11)
        cfg = SimConfig(ocr_density=0.0, mention_rate=60.0, mention_precision=prec,
                        mention_lag_iqr=lag)
        got = bind_mentions(half, trk, (0, 3999), cfg, r)
        assert got, "no mentions generated"
        # the evidence is (tracklet, named player); it is *right* only if both agree with the actor
        return float(np.mean([trk[t, 0] == named for t, named, _ in got]))

    assert hit_rate(1e-6, 1.0) > 0.95
    assert hit_rate(8.0, 1.0) < 0.75          # a lagged name lands on whoever is near the ball now
    assert hit_rate(1e-6, 0.0) < 0.05         # a wrong name is never the actor
