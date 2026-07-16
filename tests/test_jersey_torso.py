"""Stage-1c torso-crop seam: the fixed band maps to the right pixel rows and resizes to input."""

from __future__ import annotations

from PIL import Image

from generator import jersey_id as J


def test_torso_band_maps_to_committed_rows() -> None:
    """torso_crop takes full width and the [0.15, 0.55] height band (pre-committed constants)."""
    assert J.TORSO_BAND == (0.15, 0.55)
    img = Image.new("RGB", (40, 100))
    out = J.torso_crop(img)
    # full width preserved; height band = round(0.15*100)..round(0.55*100) = rows 15..55 -> 40 px.
    assert out.size == (40, 40)


def test_torso_transform_yields_input_tensor() -> None:
    """The torso transform still emits a [3, INPUT_H, INPUT_W] normalized tensor."""
    tf = J.build_transform(train=False, torso=True)
    t = tf(Image.new("RGB", (48, 128)))
    assert tuple(t.shape) == (3, J.INPUT_H, J.INPUT_W)


if __name__ == "__main__":
    test_torso_band_maps_to_committed_rows()
    test_torso_transform_yields_input_tensor()
    print("ok")
