"""Tests for the Stage-1b factorized-digit seams (pure; no model, no GPU).

Pin the digit encode/decode round-trip and the head-softmax -> Stage-1 ``[NUM_CLASSES]`` composition
that lets the factorized model reuse the single-head pooling/threshold/eval path unchanged.
"""

from __future__ import annotations

import numpy as np

from generator.jersey_id import (
    ILLEGIBLE,
    NUM_CLASSES,
    TENS_CLASSES,
    TENS_NONE,
    UNITS_CLASSES,
    decide,
    from_digits,
    heads_to_number_probs,
    to_digits,
)


def test_digit_roundtrip_all_numbers():
    # Every jersey number survives encode -> decode; single digits use the TENS_NONE marker.
    for n in range(1, 100):
        tens, units, legible = to_digits(n)
        assert legible == 1
        assert from_digits(tens, units) == n
        if n < 10:
            assert tens == TENS_NONE


def test_illegible_maps_to_zero_legibility():
    tens, units, legible = to_digits(-1)
    assert legible == 0 and 0 <= tens < TENS_CLASSES and 0 <= units < UNITS_CLASSES


def test_heads_compose_to_confident_number():
    # One crop: legible, tens digit 6, units digit 2 -> number 62 should win the composed vector.
    tens_p = np.zeros((1, TENS_CLASSES), dtype=np.float32)
    tens_p[0, 6] = 1.0
    units_p = np.zeros((1, UNITS_CLASSES), dtype=np.float32)
    units_p[0, 2] = 1.0
    legible_p = np.array([[0.1, 0.9]], dtype=np.float32)
    probs = heads_to_number_probs(tens_p, units_p, legible_p)
    assert probs.shape == (1, NUM_CLASSES)
    assert int(probs[0].argmax()) == 62
    assert decide(probs[0], min_conf=0.2) == (62, probs[0, 62])


def test_heads_illegible_routes_to_minus_one():
    # High illegible mass, diffuse digits -> the composed vote reads -1.
    tens_p = np.full((1, TENS_CLASSES), 1.0 / TENS_CLASSES, dtype=np.float32)
    units_p = np.full((1, UNITS_CLASSES), 1.0 / UNITS_CLASSES, dtype=np.float32)
    legible_p = np.array([[0.95, 0.05]], dtype=np.float32)
    probs = heads_to_number_probs(tens_p, units_p, legible_p)
    assert int(probs[0].argmax()) == ILLEGIBLE
    assert decide(probs[0], min_conf=0.2)[0] == -1


def test_single_digit_number_composes():
    # Number 7: tens=NONE, units=7 -> index 7 wins.
    tens_p = np.zeros((1, TENS_CLASSES), dtype=np.float32)
    tens_p[0, TENS_NONE] = 1.0
    units_p = np.zeros((1, UNITS_CLASSES), dtype=np.float32)
    units_p[0, 7] = 1.0
    legible_p = np.array([[0.2, 0.8]], dtype=np.float32)
    probs = heads_to_number_probs(tens_p, units_p, legible_p)
    assert int(probs[0].argmax()) == 7
