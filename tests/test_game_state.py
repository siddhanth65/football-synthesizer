"""Pure-seam tests for the game-state layer: manager metadata, possession norm, attack typing, WP."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import registry
from fingerprint import attack_typing as at
from fingerprint import win_probability as wp
from fingerprint.possession_metrics import normalize_kpi, possession_share_proxy


def test_manager_metadata_split():
    th = registry.by_manager("ten_hag")
    am = registry.by_manager("amorim")
    assert len(th) == 6 and len(am) == 6
    assert {m.id for m in th} == {"brighton_manutd", "manutd_liverpool", "manutd_fulham",
                                  "palace_manutd", "manutd_tottenham", "southampton_manutd"}
    # every dated ManU fixture carries a regime + date; the split respects the sacking date.
    for m in th:
        assert m.date < "2024-10-28", m.id          # ten Hag sacked
    for m in am:
        assert m.date >= "2024-11-24", m.id         # Amorim's first PL match


def test_possession_normalization():
    # KPI/(1 - poss_share): a team seeing 30% of the ball defends 70% -> conceded rate scaled up.
    assert normalize_kpi(10.0, 0.3) == 10.0 / 0.7
    assert np.isnan(normalize_kpi(1.0, 1.0))        # guarded domain
    assert np.isnan(normalize_kpi(1.0, -0.1))
    assert possession_share_proxy(300, 500) == 0.375
    assert np.isnan(possession_share_proxy(0, 0))


def test_attack_typing_rules_and_coverage():
    # spell splitting: a team change and an intra-team sampling gap both break a run.
    poss = pd.DataFrame({"frame": [0, 10, 200, 210, 400], "carrier": [1, 2, 1, 3, 4],
                         "team": [0, 0, 0, 1, 1]})
    sp = at.possession_spells(poss, max_gap_frames=25)
    # team-0 run [0,10,200] splits at the 10->200 gap; team 1 [210,400] splits at 210->400.
    assert [s["team"] for s in sp] == [0, 0, 1, 1]
    assert sp[0]["f1"] == 10 and sp[1]["f0"] == 200
    # 3-way rules key off deep start + entry time + pass count.
    assert at.label_sequence(1, 30.0, 5.0, 6.0) == ("direct", "3way")
    assert at.label_sequence(4, 30.0, 5.0, 6.0) == ("fast_transition", "3way")
    assert at.label_sequence(9, 30.0, 4.0, 9.0) == ("sustained_build_up", "3way")
    assert at.label_sequence(2, 90.0, 0.0, 1.0)[1] == "high"   # regain already in the third
    # type_mix reports coverage and per-label shares over the 3-way tier.
    seqs = pd.DataFrame({
        "team": [0, 0, 0, 0],
        "label": ["direct", "fast_transition", "sustained_build_up", "fast"],
        "mode": ["3way", "3way", "3way", "2way"]})
    mix = at.type_mix(seqs, 0)
    assert mix["n_3way"] == 3 and mix["n_2way"] == 1
    assert abs(mix["cov_3way"] - 0.75) < 1e-9
    assert abs(mix["direct"] - 1 / 3) < 1e-9


def test_wp_snapshots_symmetry_and_monotone_calibration():
    # symmetry: away perspective mirrors home; state reflects only past events.
    snap = wp.match_snapshots([(20, 1)], [], [(30, -1)], elo_home=1800, elo_away=1700,
                              home_result="win")
    h = snap[snap["is_home"] == 1]
    a = snap[snap["is_home"] == 0]
    assert (h["score_diff"].to_numpy() == -a["score_diff"].to_numpy()).all()
    assert (h["result"] == "win").all() and (a["result"] == "loss").all()
    assert int(h[h["minute"] == 25]["score_diff"].iloc[0]) == 1
    # single-perspective inference features take is_home explicitly.
    pf = wp.perspective_features([(10, 1)], [], [], elo_for=1800, elo_against=1750, is_home=0)
    assert (pf["is_home"] == 0).all() and set(wp.FEATURES) <= set(pf.columns)


def test_wp_module_selfcheck():
    wp._demo()      # synthetic held-out ECE gate + elo monotonicity
    at._demo()
