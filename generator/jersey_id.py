"""Jersey-number recognition for player tracklets (SoccerNet jersey-2023 recipe class).

Two challenge stages -- legibility ("is a number visible?") and number recognition -- are folded
into a single 100-way crop classifier: class ``0`` is *illegible / no legible number* (ground-truth
``-1``) and classes ``1..99`` are the jersey number. Per-tracklet decisions come from
confidence-weighted voting over the tracklet's crops (:func:`aggregate_votes`): crops that actually
show the number accumulate consistent mass on the true class, back-view / blurred crops spread thin
and cancel, and a tracklet with no consistent number falls through to ``-1``.

Stage-2 wiring contract (see :class:`JerseyRecognizer`): a track's crop paths in ->
``(jersey_number:int, confidence:float)`` out, where ``jersey_number == -1`` means "no number read".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 100
"""Class ``0`` = illegible/no-number (gt ``-1``); classes ``1..99`` = jersey number."""

ILLEGIBLE = 0
INPUT_H = 224
INPUT_W = 112
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def to_class(label: int) -> int:
    """Map a ground-truth jersey label to a class index.

    Args:
        label: Ground-truth value: ``-1`` (illegible) or a jersey number ``1..99``.

    Returns:
        Class index: ``0`` for illegible, else the number itself.

    Raises:
        ValueError: If ``label`` is not ``-1`` or in ``1..99``.
    """
    if label == -1:
        return ILLEGIBLE
    if 1 <= label <= 99:
        return label
    raise ValueError(f"jersey label out of range: {label}")


def from_class(cls: int) -> int:
    """Inverse of :func:`to_class`: class index -> jersey label (``0`` -> ``-1``)."""
    return -1 if cls == ILLEGIBLE else cls


def load_gt(path: str | Path) -> dict[str, int]:
    """Load a ``*_gt.json`` mapping ``{tracklet_id: jersey_label}`` (labels ``-1`` or ``1..99``)."""
    return {str(k): int(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def build_transform(*, train: bool) -> transforms.Compose:
    """Build the crop preprocessing pipeline.

    Args:
        train: If true, add light photometric/affine augmentation; eval is deterministic.

    Returns:
        A torchvision transform mapping a PIL crop to a normalized ``[3, H, W]`` tensor.
    """
    steps: list[object] = [transforms.Resize((INPUT_H, INPUT_W))]
    if train:
        steps += [
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.RandomAffine(degrees=5, translate=(0.06, 0.06), scale=(0.9, 1.1)),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(steps)


def build_model(*, pretrained: bool = True) -> nn.Module:
    """ResNet18 backbone with a :data:`NUM_CLASSES`-way head.

    Args:
        pretrained: Load ImageNet-1k weights for the backbone (large win on this small dataset).

    Returns:
        The classifier module (final ``fc`` re-initialized to :data:`NUM_CLASSES` outputs).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    return net


def tracklet_mean(probs: np.ndarray, *, weighted: bool = True) -> np.ndarray:
    """Pool per-crop softmax rows into one ``[NUM_CLASSES]`` tracklet vector.

    Args:
        probs: ``[n_crops, NUM_CLASSES]`` softmax rows. Empty -> zeros.
        weighted: Weight each crop by its peak probability (down-weights diffuse back-view crops).

    Returns:
        The (weighted) mean softmax vector for the tracklet.
    """
    if probs.size == 0:
        return np.zeros(NUM_CLASSES, dtype=np.float32)
    if weighted:
        w = probs.max(axis=1, keepdims=True)
        return (probs * w).sum(axis=0) / max(float(w.sum()), 1e-8)
    return probs.mean(axis=0)


def decide(mean_p: np.ndarray, *, min_conf: float = 0.30) -> tuple[int, float]:
    """Turn a pooled tracklet vector into a label + confidence.

    Args:
        mean_p: A :func:`tracklet_mean` vector. All-zero -> ``(-1, 0.0)``.
        min_conf: Floor on the winning number's probability; below it (or if illegible wins),
            the tracklet reads ``-1``.

    Returns:
        ``(jersey_label, confidence)`` with ``jersey_label`` in ``{-1} u {1..99}``.
    """
    if not mean_p.any():
        return -1, 0.0
    if int(mean_p.argmax()) == ILLEGIBLE:
        return -1, float(mean_p[ILLEGIBLE])
    best_num = int(mean_p[1:].argmax()) + 1
    conf = float(mean_p[best_num])
    return (best_num, conf) if conf >= min_conf else (-1, conf)


def aggregate_votes(
    probs: np.ndarray, *, min_conf: float = 0.30, weighted: bool = True
) -> tuple[int, float]:
    """Aggregate per-crop class probabilities into one tracklet label.

    Confidence-weighted mean softmax over the tracklet's crops (:func:`tracklet_mean`), then
    thresholded (:func:`decide`): crops that show the number accumulate consistent mass, noise
    cancels, and a tracklet with no confident number reads ``-1``.

    Args:
        probs: ``[n_crops, NUM_CLASSES]`` softmax rows. Empty -> ``(-1, 0.0)``.
        min_conf: Floor on the winning number's aggregated probability; below it, read ``-1``.
        weighted: Down-weight diffuse crops by their peak probability.

    Returns:
        ``(jersey_label, confidence)`` where ``jersey_label`` is ``-1`` or ``1..99``.
    """
    return decide(tracklet_mean(probs, weighted=weighted), min_conf=min_conf)


class JerseyRecognizer:
    """Loaded jersey model + Stage-2 inference contract (crop paths -> number + confidence)."""

    def __init__(self, model: nn.Module, device: str, *, min_conf: float = 0.30) -> None:
        """Wrap a model for inference.

        Args:
            model: A :func:`build_model` network with trained weights, already on ``device``.
            device: ``"cuda"`` or ``"cpu"``.
            min_conf: Tracklet-level ``-1`` threshold passed to :func:`aggregate_votes`.
        """
        self.model = model.eval()
        self.device = device
        self.min_conf = min_conf
        self.tf = build_transform(train=False)

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, device: str | None = None, *, min_conf: float = 0.30
    ) -> JerseyRecognizer:
        """Load a recognizer from a ``train_jersey.py`` checkpoint (``model_state`` key)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(pretrained=False).to(device)
        state = torch.load(Path(ckpt_path), map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        return cls(model, device, min_conf=min_conf)

    @torch.no_grad()
    def crop_probs(self, paths: Sequence[str | Path], batch_size: int = 256) -> np.ndarray:
        """Softmax probabilities for each readable crop.

        Args:
            paths: Crop image paths for one tracklet. Unreadable files are skipped.
            batch_size: Forward-pass batch size (fp16 on CUDA).

        Returns:
            ``[n_ok, NUM_CLASSES]`` softmax rows (``n_ok`` may be ``< len(paths)``).
        """
        tensors = []
        for p in paths:
            try:
                tensors.append(self.tf(Image.open(p).convert("RGB")))
            except (OSError, ValueError):
                continue
        if not tensors:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        out = []
        use_amp = self.device == "cuda"
        for i in range(0, len(tensors), batch_size):
            xb = torch.stack(tensors[i : i + batch_size]).to(self.device)
            with torch.amp.autocast(self.device, enabled=use_amp):
                logits = self.model(xb)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out)

    def predict_tracklet(
        self, crops: str | Path | Sequence[str | Path], batch_size: int = 256
    ) -> tuple[int, float]:
        """Predict one tracklet's jersey number (Stage-2 contract).

        Args:
            crops: A directory of ``*.jpg`` crops, or an explicit sequence of crop paths.
            batch_size: Forward-pass batch size.

        Returns:
            ``(jersey_number, confidence)``; ``jersey_number == -1`` means no number read.
        """
        if isinstance(crops, (str, Path)):
            paths: Sequence[str | Path] = sorted(Path(crops).glob("*.jpg"))
        else:
            paths = crops
        return aggregate_votes(self.crop_probs(paths, batch_size), min_conf=self.min_conf)
