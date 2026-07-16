"""CPU tests for the B2 Stage-2b-step1 pure seams: negative harvest label rule, shot segmentation,
per-shot consensus pooling, and negative-crop pseudo-tracklet grouping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


if __name__ == "__main__":
    test_has_player_inside_containment()
    test_shots_segment_by_gap()
    test_consensus_pools_agreement_and_rejects_noise()
    test_negative_tracklets_group_and_label(Path("_tmp_negtest"))
    test_kit_dist_ok_keeps_players_drops_far_colours()
    test_digit_agreement_requires_matching_confident_token()
    print("ok")
