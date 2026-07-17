"""CPU tests for the B2 Stage-2b-step1 pure seams: negative harvest label rule, shot segmentation,
per-shot consensus pooling, and negative-crop pseudo-tracklet grouping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from generator import jersey_id as J
from tools import closeup_anchor_probe as P
from tools import train_jersey as T


def _prob_row(cls: int, mass: float = 0.9) -> np.ndarray:
    """A softmax-like row with ``mass`` on class ``cls`` and the remainder spread over the rest."""
    row = np.full(J.NUM_CLASSES, (1.0 - mass) / (J.NUM_CLASSES - 1), dtype=np.float32)
    row[cls] = mass
    return row


def test_has_player_inside_containment() -> None:
    """A COCO box is a player iff a football-YOLO player point falls in it (with the margin)."""
    box = np.array([100.0, 200.0, 160.0, 340.0])  # x1,y1,x2,y2
    inside = np.array([[130.0, 300.0]], dtype=np.float32)  # point within the box -> player
    outside = np.array([[400.0, 300.0]], dtype=np.float32)  # far away -> non-player
    assert P._has_player_inside(box, inside) is True
    assert P._has_player_inside(box, outside) is False
    assert P._has_player_inside(box, np.empty((0, 2), dtype=np.float32)) is False
    # margin admits a point just outside the raw box edge (crowd/player boundary slack).
    assert P._has_player_inside(box, np.array([[168.0, 300.0]], dtype=np.float32)) is True


def test_shots_segment_by_gap() -> None:
    """Contiguous sampled frames within SHOT_GAP_FRAMES form one shot; a larger gap splits."""
    frames = [10, 15, 20, 100, 105]  # gap 80 > SHOT_GAP_FRAMES splits into two runs
    assert P._shots(frames) == [(10, 20), (100, 105)]
    assert P._shots([]) == []


def test_consensus_pools_agreement_and_rejects_noise() -> None:
    """A shot whose crops agree on a number votes it; an illegible-dominated shot votes -1."""
    shots = [(0, 4), (10, 14)]
    crops = [
        {"frame": 0, "_prob": _prob_row(8), "crop_path": "a"},
        {"frame": 2, "_prob": _prob_row(8), "crop_path": "b"},
        {"frame": 4, "_prob": _prob_row(8, 0.95), "crop_path": "c"},  # highest mass -> rep
        {"frame": 10, "_prob": _prob_row(J.ILLEGIBLE), "crop_path": "d"},
        {"frame": 12, "_prob": _prob_row(J.ILLEGIBLE), "crop_path": "e"},
    ]
    out = P._consensus_anchors(crops, shots, min_conf=0.30)
    assert len(out) == 1  # only the agreeing shot yields an anchor
    (a,) = out
    assert a["pred"] == 8 and a["shot"] == 0 and a["n_crops"] == 3
    assert a["rep_crop"] == "c"  # representative = crop with most mass on the winning number


def test_negative_tracklets_group_and_label(tmp_path: Path) -> None:
    """Harvested crops chunk into ILLEGIBLE pseudo-tracklets of the requested size."""
    for i in range(5):
        (tmp_path / f"crop_{i}.jpg").write_bytes(b"")
    items = T._negative_tracklets(tmp_path, per_tracklet=2)
    assert [len(c) for _, _, c in items] == [2, 2, 1]  # 5 crops -> 2,2,1
    assert all(cls == J.ILLEGIBLE for _, cls, _ in items)
    assert T._negative_tracklets(tmp_path / "empty", per_tracklet=2) == []


def test_kit_dist_ok_keeps_players_drops_far_colours() -> None:
    """LEVER A: within KIT_DIST_MAX of a centroid passes; a far (black-kit) colour fails."""
    cent = np.array([[69.1, 4.5, -10.7], [42.8, 29.9, 7.3]], dtype=np.float32)  # Brighton, Man Utd
    on_red = np.array([41.6, 48.0, 12.0], dtype=np.float32)   # confirmed "8" back, dmin ~18.7
    referee = np.array([14.9, 2.0, -2.0], dtype=np.float32)   # black kit, dmin ~40.5
    assert P._kit_dist_ok(on_red, cent) is True
    assert P._kit_dist_ok(referee, cent) is False
    # threshold is inclusive and the knob bites: a colour just past 22 fails, just under passes.
    assert P._kit_dist_ok(cent[1] + np.array([21.0, 0, 0], np.float32), cent) is True
    assert P._kit_dist_ok(cent[1] + np.array([23.0, 0, 0], np.float32), cent) is False


def test_digit_agreement_requires_matching_confident_token() -> None:
    """LEVER B: agreement iff a confident OCR digit token equals the classifier's number."""
    assert P._digit_agreement([("20", 1.0)], 20) is True
    assert P._digit_agreement([("20", 0.99)], 24) is False   # 20->24 misread: OCR disagrees
    assert P._digit_agreement([], 8) is False                 # no digit region (front/side crop)
    assert P._digit_agreement([("8", 0.3)], 8) is False       # low-conf token below the floor
    assert P._digit_agreement([("x", 0.9)], 8) is False       # non-digit token ignored
    assert P._digit_agreement([("11", 0.9), ("8", 0.9)], 8) is True  # any matching token suffices


def _row(pairs: dict[int, float]) -> np.ndarray:
    """A prob row with the given class->mass entries (rest zero)."""
    row = np.zeros(J.NUM_CLASSES, dtype=np.float32)
    for cls, mass in pairs.items():
        row[cls] = mass
    return row


def test_roster_mask_blocks_invalid_and_lifts_valid() -> None:
    """LEVER 1: an off-roster peak can never win, and its mass lifts a sub-threshold valid read."""
    mask = J.roster_mask([8, 10])
    assert mask[J.ILLEGIBLE] and mask[8] and mask[10]
    assert not mask[33] and not mask[99]
    # off-roster 33 outscores valid 8 raw -> raw decides 33; masking removes 33 -> 8 wins.
    row = _row({33: 0.6, 8: 0.35, J.ILLEGIBLE: 0.05})
    raw_num, raw_conf = J.decide(row, min_conf=0.0)
    assert raw_num == 33 and raw_conf == pytest.approx(0.6, abs=1e-5)
    num, conf = J.decide(row, min_conf=0.0, mask=mask)
    assert num == 8 and conf > 0.8
    # a valid read just below 0.70 clears it once off-roster mass is redistributed.
    row2 = _row({8: 0.65, 33: 0.30, J.ILLEGIBLE: 0.05})
    assert J.decide(row2, min_conf=0.70)[0] == -1                    # raw: below the bar
    assert J.decide(row2, min_conf=0.70, mask=mask)[0] == 8          # masked: clears it


def test_iou_overlap_and_disjoint() -> None:
    """IoU is 1.0 for identical boxes, 0.0 for disjoint, in-between for partial overlap."""
    assert P._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert P._iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0
    assert P._iou((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5 / 1.5  # inter 50, union 150


def _crop(frame: int, box: tuple[int, int, int, int], pred: int, conf: float,
          *, kit_ok: bool = True, toks: list | None = None) -> dict:
    """A synthetic candidate crop record for the tracker/agreement seams."""
    return {"frame": frame, "box": box, "pred": pred, "conf": conf, "pred_m": pred,
            "conf_m": conf, "kit_ok": kit_ok, "toks": toks or []}


def test_link_tracklets_separates_two_people() -> None:
    """Overlapping boxes across sampled frames chain into one track; a far box is a second track."""
    a = (0, 0, 10, 20)
    b = (100, 0, 110, 20)
    crops = [_crop(0, a, 8, 0.6), _crop(0, b, 8, 0.6),
             _crop(5, a, 8, 0.6), _crop(5, b, 8, 0.6), _crop(10, a, 8, 0.6)]
    tracks = sorted(P._link_tracklets(crops), key=len, reverse=True)
    assert [len(t) for t in tracks] == [3, 2]
    assert [c["frame"] for c in tracks[0]] == [0, 5, 10]


def test_agreement_admits_consistent_run_and_guards() -> None:
    """LEVER 2: an N-consecutive same-number run passes only with kit + one OCR agreement."""
    a = (0, 0, 10, 20)
    run3 = [_crop(0, a, 8, 0.6), _crop(5, a, 8, 0.62, toks=[("8", 0.9)]), _crop(10, a, 8, 0.6)]
    assert len(P._agreement_admit(run3, n=2)) == 3
    assert len(P._agreement_admit(run3, n=3)) == 3
    # a 2-frame run fails n=3.
    assert P._agreement_admit(run3[:2], n=3) == []
    # no OCR agreement anywhere -> rejected even though the number is consistent.
    no_ocr = [_crop(0, a, 8, 0.6), _crop(5, a, 8, 0.6), _crop(10, a, 8, 0.6)]
    assert P._agreement_admit(no_ocr, n=2) == []
    # a kit failure inside the run breaks it (all-crops kit gate).
    bad_kit = [_crop(0, a, 8, 0.6, toks=[("8", 0.9)]),
               _crop(5, a, 8, 0.6, kit_ok=False), _crop(10, a, 8, 0.6)]
    assert P._agreement_admit(bad_kit, n=3) == []
    # a sub-floor confidence breaks the run.
    low = [_crop(0, a, 8, 0.6, toks=[("8", 0.9)]), _crop(5, a, 8, 0.3), _crop(10, a, 8, 0.6)]
    assert P._agreement_admit(low, n=3) == []


if __name__ == "__main__":
    test_has_player_inside_containment()
    test_shots_segment_by_gap()
    test_consensus_pools_agreement_and_rejects_noise()
    test_negative_tracklets_group_and_label(Path("_tmp_negtest"))
    test_kit_dist_ok_keeps_players_drops_far_colours()
    test_digit_agreement_requires_matching_confident_token()
    test_roster_mask_blocks_invalid_and_lifts_valid()
    test_iou_overlap_and_disjoint()
    test_link_tracklets_separates_two_people()
    test_agreement_admits_consistent_run_and_guards()
    print("ok")
