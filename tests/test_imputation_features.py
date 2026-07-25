"""Leakage guard + unit checks for the B4 v1 feature collector (plan 4.2).

The load-bearing test is :func:`test_no_hidden_truth_leakage`: every field and every feature
column is recomputed from a truth array whose hidden entries are NaN, and must come out
bit-identical. If that fails, no B4 number is trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthesizer.imputation import (
    METRICA,
    collect_samples,
    fit_slot_model,
    smooth_camera,
    team_ranges,
    visibility_mask,
    visible_centroid,
    _prep_game,
)
from synthesizer.imputation_features import (
    FEATURE_NAMES,
    build_features,
    conformal_k,
    coverage_table,
    derive_fields,
    ffill_ball,
    region_scale,
    role_offsets_vote,
    vote_field,
)

N_FRAMES = 4000
FPS = 25.0


def _synthetic_match(seed: int = 0) -> dict:
    """Build a small synthetic 2x11 match with a ball-following broadcast window."""
    rng = np.random.default_rng(seed)
    n, slots = N_FRAMES, 22
    base_x = np.concatenate([np.linspace(10, 80, 11), np.linspace(25, 95, 11)])
    base_y = np.tile(np.linspace(8, 60, 11), 2)
    drift = np.cumsum(rng.normal(0, 0.08, size=(n, slots, 2)), axis=0)
    truth = np.stack([base_x, base_y], axis=1)[None] + drift
    truth[:, :, 0] = np.clip(truth[:, :, 0], 0.5, 104.5)
    truth[:, :, 1] = np.clip(truth[:, :, 1], 0.5, 67.5)
    ball = np.column_stack(
        [52.5 + 35 * np.sin(np.arange(n) / 400.0), 34 + 12 * np.cos(np.arange(n) / 310.0)]
    )
    ball[rng.random(n) < 0.05] = np.nan
    period = np.where(np.arange(n) < n // 2, 1, 2)
    cam = smooth_camera(ball, FPS)
    visible = visibility_mask(truth, cam, half_w=17.0, half_h=34.0)
    return {
        "truth": truth, "visible": visible, "ball": ball, "cam": cam, "period": period,
        "fps": FPS, "ranges": team_ranges(11, slots),
    }


def _pipeline(g: dict, truth_for_fields: np.ndarray, coeffs) -> tuple[np.ndarray, dict]:
    """Derive fields + features from ``truth_for_fields`` while sampling from real truth."""
    fields = derive_fields(
        truth_for_fields, g["visible"], g["ball"], g["cam"], g["period"], g["fps"],
        g["ranges"], coeffs,
    )
    samples = collect_samples(
        g["truth"], g["visible"], g["period"], g["fps"],
        np.zeros_like(g["truth"]), np.asarray(fields["slot"]),
    )
    blend = samples["hold"]  # stand-in anchor: supplied externally, identical in both runs
    return build_features(fields, samples, blend), samples


def test_no_hidden_truth_leakage() -> None:
    """Every feature column must be bit-identical when hidden truth is blanked to NaN."""
    g = _synthetic_match()
    masked = np.where(g["visible"][:, :, None], g["truth"], np.nan)
    assert not np.array_equal(g["truth"], masked, equal_nan=True), "guard would be vacuous"
    coeffs = fit_slot_model(
        g["truth"], visible_centroid(g["truth"], g["visible"], g["ranges"]), g["cam"], g["ranges"]
    )
    x_full, samples = _pipeline(g, g["truth"], coeffs)
    x_masked, _ = _pipeline(g, masked, coeffs)
    assert x_full.shape[1] == len(FEATURE_NAMES) == 36
    assert x_full.shape[0] > 1000, x_full.shape
    bad = [
        FEATURE_NAMES[j]
        for j in range(x_full.shape[1])
        if not np.array_equal(x_full[:, j], x_masked[:, j], equal_nan=True)
    ]
    assert not bad, f"columns that read hidden truth: {bad}"
    # Non-vacuity: a column built from the hidden player's CURRENT truth would differ.
    f = samples["frame"].astype(int)
    p = samples["slot_id"].astype(int)
    leaky_full = g["truth"][f, p, 0]
    leaky_masked = masked[f, p, 0]
    assert not np.array_equal(leaky_full, leaky_masked, equal_nan=True)


def test_vote_is_causal_and_frozen_while_hidden() -> None:
    """A hidden player's role offset must never update, and the vote must ignore it."""
    g = _synthetic_match(seed=1)
    off, vote, count = role_offsets_vote(g["truth"], g["visible"], g["ranges"], g["fps"])
    field = vote_field(off, vote, g["ranges"])
    assert field.shape == g["truth"].shape
    p = 3
    hidden = ~g["visible"][:, p]
    runs = np.flatnonzero(hidden[1:] & hidden[:-1]) + 1
    assert runs.size > 10
    assert np.allclose(off[runs, p], off[runs - 1, p]), "offset moved while player was hidden"
    assert np.all(count[:, 0] <= 11)


def test_ffill_ball_is_causal() -> None:
    """The ball fill must never use a future frame."""
    ball = np.array([[1.0, 1.0], [np.nan, np.nan], [5.0, 5.0], [np.nan, np.nan]])
    assert np.allclose(ffill_ball(ball), [[1, 1], [1, 1], [5, 5], [5, 5]])


def test_conformal_coverage_is_nominal_on_gaussian_residuals() -> None:
    """Split conformal on i.i.d. residuals must hit 50/90 within sampling noise."""
    rng = np.random.default_rng(2)
    cal = rng.normal(scale=[3.0, 1.5], size=(5000, 2))
    test = rng.normal(scale=[3.0, 1.5], size=(5000, 2))
    bkt_c = np.zeros(5000, dtype=int)
    w = region_scale(cal, bkt_c)
    k = conformal_k(cal, bkt_c, w)
    rows = coverage_table(test, bkt_c, w, k)
    assert 0.47 <= rows[0]["picp_50"] <= 0.53, rows[0]
    assert 0.88 <= rows[0]["picp_90"] <= 0.92, rows[0]
    assert rows[0]["r_90"] > rows[0]["r_50"] > 0


@pytest.mark.skipif(
    not (METRICA / "Sample_Game_2_RawTrackingData_Home_Team.csv").exists(),
    reason="Metrica open data not present (data/imputation is gitignored)",
)
def test_no_leakage_on_real_metrica_slice() -> None:
    """Same guard, on a real censored Metrica slice rather than synthetic motion."""
    g, _, _ = _prep_game("Sample_Game_2", half_w=16.9)
    sl = slice(0, 15000)
    sub = {
        "truth": g["truth"][sl], "visible": g["visible"][sl], "ball": g["ball"][sl],
        "cam": g["cam"][sl], "period": g["period"][sl], "fps": g["fps"], "ranges": g["ranges"],
    }
    coeffs = fit_slot_model(
        sub["truth"], visible_centroid(sub["truth"], sub["visible"], sub["ranges"]),
        sub["cam"], sub["ranges"],
    )
    masked = np.where(sub["visible"][:, :, None], sub["truth"], np.nan)
    x_full, _ = _pipeline(sub, sub["truth"], coeffs)
    x_masked, _ = _pipeline(sub, masked, coeffs)
    bad = [
        FEATURE_NAMES[j]
        for j in range(x_full.shape[1])
        if not np.array_equal(x_full[:, j], x_masked[:, j], equal_nan=True)
    ]
    assert not bad, f"columns that read hidden truth: {bad}"
