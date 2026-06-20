"""Tests for the jersey-colour team classifier (synthetic crops; needs only cv2/numpy/sklearn)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from generator.teams import JerseyColorTeamClassifier, jersey_color

_GRASS_BGR = (55, 150, 55)  # green pitch background


def _player_crop(team_bgr, *, size=64, noise=8, seed=0):
    """A crop: green grass everywhere, a solid team-colour torso rectangle in the middle."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = _GRASS_BGR
    img[int(0.18 * size):int(0.55 * size), int(0.20 * size):int(0.80 * size)] = team_bgr
    return np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)


def test_jersey_color_ignores_grass_and_picks_torso():
    # A red torso on grass should yield a LAB colour with strongly positive a* (red), not green.
    red = jersey_color(_player_crop((0, 0, 200)))
    assert red[1] > 20.0  # a* clearly on the red side


def test_two_teams_split_balanced_and_consistent():
    reds = [_player_crop((0, 0, 200), seed=i) for i in range(8)]
    blues = [_player_crop((200, 0, 0), seed=100 + i) for i in range(8)]
    clf = JerseyColorTeamClassifier().fit(reds + blues)
    labels = clf.predict(reds + blues)
    red_labels, blue_labels = labels[:8], labels[8:]
    # Each team maps to ONE consistent cluster, and the two teams differ (no collapse).
    assert len(set(red_labels)) == 1 and len(set(blue_labels)) == 1
    assert red_labels[0] != blue_labels[0]
    # Balanced: 8 and 8.
    assert sorted(np.bincount(labels).tolist()) == [8, 8]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        JerseyColorTeamClassifier().predict([_player_crop((0, 0, 200))])


def test_fit_needs_enough_crops():
    with pytest.raises(ValueError, match="need >="):
        JerseyColorTeamClassifier().fit([_player_crop((0, 0, 200))])


def test_empty_crop_is_safe():
    out = jersey_color(np.zeros((0, 0, 3), np.uint8))
    assert out.shape == (3,)
