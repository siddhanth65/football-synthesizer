"""Tests for the Tier-A marginal tendency predictor (C5, the buildable tier)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthesizer.features import TENDENCIES, team_tendency_samples
from synthesizer.predict import marginal_distribution, predict_tendencies


def _match(team_line_x, n_chunks=4, n_frames=30, seed=0):
    """A team holding a defensive line near ``team_line_x`` across several chunks (keeper anchored left)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_chunks):
        for f in range(n_frames):
            rows.append({"chunk": f"chunk_{c}", "frame": f, "team": 0, "role": "goalkeeper",
                         "is_keeper": True, "track_id": 99, "pitch_x": 3.0, "pitch_y": 34.0,
                         "calib_error_m": 0.2})
            for i in range(6):  # outfield around the line
                rows.append({"chunk": f"chunk_{c}", "frame": f, "team": 0, "role": "player",
                             "is_keeper": False, "track_id": i, "calib_error_m": 0.2,
                             "pitch_x": team_line_x + rng.normal(0, 2), "pitch_y": 10 + i * 8.0})
    return pd.DataFrame(rows)


def test_tendency_samples_one_row_per_chunk():
    s = team_tendency_samples(_match(40.0, n_chunks=5), team=0)
    assert len(s) == 5
    assert set(TENDENCIES).issubset(s.columns)


def test_marginal_distribution_has_mean_and_interval():
    s = team_tendency_samples(_match(40.0), team=0)
    d = marginal_distribution(s, interval=0.8)
    bh = d["buildup_height"]
    assert bh["lo"] <= bh["mean"] <= bh["hi"]      # interval brackets the mean
    assert bh["n"] == 4


def test_tier_a_reflects_a_higher_line():
    low = predict_tendencies(_match(25.0), team=0, tier="A")["buildup_height"]["mean"]
    high = predict_tendencies(_match(60.0), team=0, tier="A")["buildup_height"]["mean"]
    assert high > low + 20                          # a higher planted line -> higher build-up height


def test_tier_b_is_explicitly_data_gated():
    with pytest.raises(NotImplementedError):
        predict_tendencies(_match(40.0), team=0, tier="B")


def test_opponent_model_fit_backtest_forecast():
    from synthesizer.opponent_model import backtest, fit, forecast  # noqa: PLC0415

    # 6 team-matches with a clean negative opp-depth -> attacking-commitment signal
    obs = pd.DataFrame({
        "match": ["m1", "m1", "m2", "m2", "m3", "m3"],
        "team": list("ABCDEF"), "opponent": list("BADCFE"),
        "attacking_third_share": [0.45, 0.15, 0.40, 0.20, 0.48, 0.12],
        "def_line_height": [55, 30, 52, 34, 58, 28],
        "width": [40, 34, 39, 35, 41, 33], "wing_share": [0.25, 0.2, 0.24, 0.2, 0.26, 0.19],
        "opp_def_depth": [30, 55, 34, 50, 28, 58],
    })
    m = fit(obs)
    assert m["attacking_third_share"][0] < 0               # slope: deeper opponent -> more forward
    bt = backtest(obs).set_index("tendency")
    assert bt.loc["attacking_third_share", "tier_b_mae"] < bt.loc["attacking_third_share", "tier_a_mae"]
    deep = forecast(28, m)["attacking_third_share"]["mean"]
    high = forecast(58, m)["attacking_third_share"]["mean"]
    assert deep > high                                      # forecast monotone in opponent depth


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # backtest.py is retained only as history
def test_deprecated_backtest_still_scores_skill():
    from synthesizer.backtest import backtest  # noqa: PLC0415

    # a clean linear opponent signal -> Tier B should help (positive skill on this synthetic)
    prof = pd.DataFrame({
        "opp_low_block": [10, 20, 30, 40],
        "france_att3rd": [0.20, 0.30, 0.40, 0.50],   # perfectly linear in the feature
        "france_line": [45, 48, 51, 54], "france_wing": [0.2, 0.2, 0.2, 0.2],
    })
    res = backtest(prof).set_index("tendency")
    assert "skill" in res.columns
    assert res.loc["france_att3rd", "tier_b_mae"] < res.loc["france_att3rd", "tier_a_mae"]  # signal -> B wins
