"""Targeted tests for :mod:`tools.score_carrier_labels` (pure scoring, no artefacts needed)."""
from __future__ import annotations

import pandas as pd

from tools.score_carrier_labels import load_labels, norm_name, score_labels

MATCH = "manutd_liverpool"


def _labels() -> pd.DataFrame:
    """Ten hand-labelled moments covering every answer type and both distance bands."""
    rows = [
        # id, frame, dist, answer, confidence
        ("000", 10, 0.5, "Bruno Fernandes", "3"),    # assigned + correct, near, in gallery
        ("001", 20, 0.5, "Kobbie Mainoo", "3"),      # assigned + wrong, near, in gallery
        ("002", 30, 2.4, "Diogo Dalot", "2"),        # assigned + correct, far band
        ("003", 40, 2.4, "Tom Heaton", "2"),         # assigned + wrong, NOT gallery-covered
        ("004", 50, 1.5, "Marcus Rashford", "3"),    # abstained below similarity
        ("005", 60, 1.5, "Amad Diallo", "1"),        # abstained below margin
        ("006", 70, 0.9, "WRONG_PLAYER_BOXED", "3"),  # carrier-selection error, model assigned
        ("007", 80, 0.9, "UNKNOWN", "1"),            # excluded
        ("008", 90, 0.9, "NOT_A_PASS", "3"),         # excluded
        ("009", 100, 0.9, "", ""),                   # excluded (blank)
    ]
    return pd.DataFrame([{"id": i, "match": MATCH, "chunk": "h1_chunk_000", "frame": f,
                          "team_tracked": 0, "carrier_dist_m": d, "player_name": p,
                          "confidence_1to3": c, "notes": ""} for i, f, d, p, c in rows])


def _preds() -> pd.DataFrame:
    """Model answers at a frozen point of ``min_sim=0.88``, ``min_margin=0.01``."""
    rows = [
        (10, "Bruno Fernandes", 0.95, 0.05),
        (20, "Casemiro", 0.93, 0.04),
        (30, "Diogo Dalot", 0.91, 0.03),
        (40, "Casemiro", 0.90, 0.02),
        (50, None, 0.70, 0.10),   # below similarity
        (60, None, 0.95, 0.001),  # below margin
        (70, "Casemiro", 0.94, 0.05),
        (80, None, 0.50, 0.30),
        (90, "Casemiro", 0.99, 0.20),
        (100, "Casemiro", 0.99, 0.20),
    ]
    return pd.DataFrame([{"match": MATCH, "chunk": "h1_chunk_000", "frame": f, "pred_player": p,
                          "best_sim": s, "margin": m, "min_sim": 0.88, "min_margin": 0.01}
                         for f, p, s, m in rows])


GALLERY = {MATCH: {norm_name(n) for n in
                   ["Bruno Fernandes", "Kobbie Mainoo", "Diogo Dalot", "Casemiro",
                    "Marcus Rashford", "Amad Diallo"]}}


def test_exclusions_are_counted_and_removed() -> None:
    res = score_labels(_labels(), _preds(), GALLERY)
    ex = res["exclusions"]
    assert ex == {"total": 10, "blank": 1, "UNKNOWN": 1, "NOT_A_PASS": 1, "judged": 7}


def test_headline_rates() -> None:
    res = score_labels(_labels(), _preds(), GALLERY)
    hl = res["headline"]
    assert hl["n_judged"] == 7
    assert hl["n_wrong_box"] == 1
    assert hl["carrier_selection_error_rate"] == round(1 / 7, 3)
    # correct box + assigned: ids 000-003 -> 2 of 4 correct
    assert hl["n_assigned_good_box"] == 4
    assert hl["identification_precision"] == 0.5
    # end to end adds the WRONG_PLAYER_BOXED moment the model still answered: 2 of 5
    assert hl["n_assigned_all"] == 5
    assert hl["end_to_end_precision"] == 0.4


def test_abstention_breakdown() -> None:
    ab = score_labels(_labels(), _preds(), GALLERY)["abstention"]
    assert ab["assigned"] == 5
    assert ab["abstain_below_similarity"] == 1
    assert ab["abstain_below_margin"] == 1
    assert ab["abstain_no_embedding"] == 0
    assert ab["no_prediction_row"] == 0
    assert ab["abstain_rate"] == round(2 / 7, 3)


def test_missing_prediction_row_is_flagged_not_counted_as_assigned() -> None:
    preds = _preds()
    res = score_labels(_labels(), preds[preds["frame"] != 10], GALLERY)
    assert res["abstention"]["no_prediction_row"] == 1
    assert res["headline"]["n_assigned_good_box"] == 3


def test_splits_by_distance_and_gallery_coverage() -> None:
    res = score_labels(_labels(), _preds(), GALLERY)
    by_dist = {r["dist_band"]: r for r in res["by_dist"]}
    assert by_dist["0-1m"]["n_judged"] == 2          # ids 000, 001 (wrong box is excluded here)
    assert by_dist["0-1m"]["precision"] == 0.5
    assert by_dist["2-3m"]["n_assigned"] == 2        # ids 002, 003
    assert by_dist["2-3m"]["precision"] == 0.5
    by_gal = {r["gallery_covered"]: r for r in res["by_gallery"]}
    assert by_gal[False]["n_judged"] == 1            # Tom Heaton is not in the gallery
    assert by_gal[False]["precision"] == 0.0         # uncoverable -> guaranteed wrong
    assert by_gal[True]["precision"] == round(2 / 3, 3)


def test_name_matching_is_accent_and_case_insensitive(tmp_path) -> None:
    labels = _labels()
    labels.loc[0, "player_name"] = "  bruno  FERNANDES "
    assert score_labels(labels, _preds(), GALLERY)["headline"]["identification_precision"] == 0.5
    path = tmp_path / "labels_filled.csv"
    labels.to_csv(path, index=False, encoding="utf-8-sig")
    round_trip = score_labels(load_labels(path), _preds(), GALLERY)
    assert round_trip["exclusions"]["judged"] == 7
    # the (match, chunk, frame) join must survive the CSV round trip (Int64 vs int64 frames)
    assert round_trip["abstention"]["no_prediction_row"] == 0
    assert round_trip["headline"]["identification_precision"] == 0.5
