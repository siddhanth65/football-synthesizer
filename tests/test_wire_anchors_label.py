"""Regression test for the wire_anchors embedder-label wart.

The NAMED_TRACKS report must name the ACTUAL ReID embedder used: an ImageNet-classification baseline
when ``weights is None``, and a re-ID-objective run (AIN/MSMT checkpoint) otherwise -- never the
default label when non-default weights were passed.
"""
from __future__ import annotations

from tools.wire_anchors import _embedder_label


def test_embedder_label_baseline_is_imagenet() -> None:
    label = _embedder_label("osnet_x0_25", None)
    assert "ImageNet" in label
    assert "`osnet_x0_25`" in label
    assert "weights=" not in label  # baseline never advertises a weights key


def test_embedder_label_reports_actual_reid_weights() -> None:
    label = _embedder_label("osnet_ain_x1_0", "osnet_ain_x1_0_msmt17")
    assert "ImageNet" not in label  # the exact wart: AIN run must NOT read as the baseline
    assert "re-ID" in label
    assert "`osnet_ain_x1_0`" in label
    assert "weights=`osnet_ain_x1_0_msmt17`" in label


if __name__ == "__main__":
    test_embedder_label_baseline_is_imagenet()
    test_embedder_label_reports_actual_reid_weights()
    print("ok")
