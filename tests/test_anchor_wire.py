"""CPU tests for the Stage-2c anchor-wiring seams: margin gate + propagation guard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.anchor_wire import (
    Attachment,
    build_roster_maps,
    choose_track,
    merge_intervals,
    nearest_wide_frame,
    parse_survivor_name,
    resolve_identities,
)


def test_parse_survivor_name() -> None:
    assert parse_survivor_name("h1_chunk_000_f10645_n08_c0.799.jpg") == (
        "h1_chunk_000", 10645, 8, 0.799)
    assert parse_survivor_name("h2_chunk_005_f42_n20_c0.928.jpg") == ("h2_chunk_005", 42, 20, 0.928)
    assert parse_survivor_name("not_a_crop.png") is None


def test_nearest_wide_frame() -> None:
    wide = np.array([10, 50, 90, 200])
    assert nearest_wide_frame(48, wide, win=25) == 50   # 50 is 2 away, inside window
    assert nearest_wide_frame(130, wide, win=25) is None  # nearest (90/200) both > 25 away
    assert nearest_wide_frame(0, np.array([], int), win=25) is None


def test_choose_track_margin_rule() -> None:
    assert choose_track({}) == (None, "no_candidate")
    assert choose_track({7: 0.90, 3: 0.60}) == (7, "reid_margin")   # 0.30 gap >> 0.05
    assert choose_track({7: 0.82, 3: 0.80})[0] is None              # 0.02 < 0.05 -> ambiguous
    assert choose_track({7: 0.82, 3: 0.80})[1] == "ambiguous"
    assert choose_track({7: 0.40}) == (None, "low_sim")             # below MIN_SIM 0.50
    assert choose_track({7: 0.70}) == (7, "sole_candidate")         # one candidate, above sim


def test_guard_within_fragment_agreement_and_disagreement() -> None:
    atts = [
        Attachment("h1", 5, 8, 0, 0.9, "reid_margin"),
        Attachment("h1", 5, 8, 0, 0.8, "reid_margin"),   # agrees -> n_anchors 2
        Attachment("h1", 9, 8, 0, 0.9, "reid_margin"),
        Attachment("h1", 9, 10, 0, 0.9, "reid_margin"),  # disagrees on same fragment -> flag
    ]
    resolved, flags = resolve_identities(atts)
    by_tid = {r.track_id: r for r in resolved}
    assert by_tid[5].number == 8 and by_tid[5].n_anchors == 2 and by_tid[5].basis == "fragment"
    assert 9 not in by_tid
    assert any(f["type"] == "fragment_disagreement" and f["track_id"] == 9 for f in flags)


def test_guard_merge_requires_two_agreeing_anchors() -> None:
    remap = {("h1", 1): "g", ("h1", 2): "g", ("h1", 3): "g"}  # relink group of 3 fragments

    # two agreeing anchors -> name crosses onto the un-anchored sibling (3), basis merge_2anchor
    resolved, _ = resolve_identities(
        [Attachment("h1", 1, 8, 0, 0.9, "reid_margin"),
         Attachment("h1", 2, 8, 0, 0.9, "reid_margin")], remap)
    got = {r.track_id: r for r in resolved}
    assert got[1].number == got[2].number == got[3].number == 8
    assert all(r.basis == "merge_2anchor" for r in resolved)

    # single anchor -> does NOT cross the merge; only the anchored fragment is named
    resolved, _ = resolve_identities([Attachment("h1", 1, 8, 0, 0.9, "reid_margin")], remap)
    got = {r.track_id: r for r in resolved}
    assert set(got) == {1} and got[1].basis == "fragment"

    # two anchors that disagree -> no crossing; each keeps its own name
    resolved, _ = resolve_identities(
        [Attachment("h1", 1, 8, 0, 0.9, "reid_margin"),
         Attachment("h1", 2, 10, 0, 0.9, "reid_margin")], remap)
    got = {r.track_id: (r.number, r.basis) for r in resolved}
    assert got == {1: (8, "fragment"), 2: (10, "fragment")}


def test_build_roster_maps_uses_shirt_number_and_team_split() -> None:
    oracle = pd.DataFrame({
        "name": ["Bruno Fernandes", "Diogo Dalot", "Julio Enciso", "No Number"],
        "shirtNumber": ["8", "20", "10", None],
        "teamName": ["Manchester United", "Manchester United", "Brighton", "Brighton"],
    })

    def team_id(name: str) -> int:
        return 0 if "Man" in name or "United" in name else 1

    name_by, teams_by = build_roster_maps(oracle, team_id)
    assert name_by[(0, 8)] == "Bruno Fernandes"
    assert name_by[(0, 20)] == "Diogo Dalot"     # shirt 20 = Dalot (jerseyNumber would say 2)
    assert name_by[(1, 10)] == "Julio Enciso"
    assert teams_by[10] == {1}
    assert (1, None) not in name_by             # NaN shirt number dropped


def test_merge_intervals() -> None:
    assert merge_intervals([]) == 0
    assert merge_intervals([(0, 10), (5, 20), (30, 35)]) == 25  # union: 20 + 5
