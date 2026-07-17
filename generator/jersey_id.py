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
from collections.abc import Iterable, Sequence
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

# Stage-1c torso-guided crop: the jersey number lives in the upper-back region, so feeding the
# whole body wastes input resolution on legs/grass. This fixed vertical band (fraction of crop
# height, full width) is a pre-committed lazy proxy for a pose/torso model -- no tuning, no pose
# net. Cropping to the band then resizing to INPUT_H x INPUT_W gives the number far more pixels.
TORSO_BAND = (0.15, 0.55)
"""Pre-committed (top, bottom) fractions of crop height for the torso/number band."""

# Stage-1b factorized digit heads: a number is decoded from a tens digit (0-9 or "none" =
# single-digit) and a units digit (0-9); a separate legibility head produces the -1 gate. Sharing
# digit statistics across all numbers (units "2" is trained by 2/12/22/... not by jersey 62 alone)
# directly attacks the rare high-number collapse of the single 100-way head.
TENS_CLASSES = 11
UNITS_CLASSES = 10
TENS_NONE = 10
"""Tens-digit class marking a single-digit number (``label < 10``)."""


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


def to_digits(label: int) -> tuple[int, int, int]:
    """Map a jersey label to ``(tens, units, legible)`` targets for the factorized heads.

    Args:
        label: Ground-truth value: ``-1`` (illegible) or a jersey number ``1..99``.

    Returns:
        ``(tens, units, legible)`` where ``legible`` is ``0`` for ``-1`` else ``1``. For an
        illegible label the digit slots are placeholders (``TENS_NONE, 0``) meant to be masked
        out of the digit loss; single-digit numbers use ``tens == TENS_NONE``.

    Raises:
        ValueError: If ``label`` is not ``-1`` or in ``1..99``.
    """
    if label == -1:
        return TENS_NONE, 0, 0
    if 1 <= label <= 9:
        return TENS_NONE, label, 1
    if 10 <= label <= 99:
        return label // 10, label % 10, 1
    raise ValueError(f"jersey label out of range: {label}")


def from_digits(tens: int, units: int) -> int:
    """Inverse of :func:`to_digits` digits: ``(tens, units)`` -> jersey number ``0..99``."""
    return units if tens == TENS_NONE else tens * 10 + units


# Per-number digit indices (index by jersey number 1..99) for the vectorized head composition.
_NUM_TENS = np.array([to_digits(n)[0] if n >= 1 else TENS_NONE for n in range(NUM_CLASSES)])
_NUM_UNITS = np.array([to_digits(n)[1] if n >= 1 else 0 for n in range(NUM_CLASSES)])


def roster_mask(valid_numbers: Iterable[int]) -> np.ndarray:
    """Boolean class mask (length :data:`NUM_CLASSES`) for roster-constrained decoding.

    Keeps :data:`ILLEGIBLE` and every jersey number a squad can actually wear; masks the rest.
    Multiplying a softmax row by this mask and renormalising (see :func:`decide`) removes all
    probability mass from numbers no player on either team wears, so an invalid number can never win
    the argmax and its mass is redistributed to the valid set -- some true reads whose peak sat just
    below a confidence floor clear it once the off-roster mass is removed.

    Args:
        valid_numbers: The union of both squads' back-of-shirt numbers (values outside ``1..99`` are
            ignored).

    Returns:
        A boolean ``[NUM_CLASSES]`` array: ``True`` at :data:`ILLEGIBLE` and each valid number.
    """
    m = np.zeros(NUM_CLASSES, dtype=bool)
    m[ILLEGIBLE] = True
    for n in valid_numbers:
        if 1 <= int(n) <= 99:
            m[int(n)] = True
    return m


def load_gt(path: str | Path) -> dict[str, int]:
    """Load a ``*_gt.json`` mapping ``{tracklet_id: jersey_label}`` (labels ``-1`` or ``1..99``)."""
    return {str(k): int(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def torso_crop(img: Image.Image) -> Image.Image:
    """Crop the pre-committed :data:`TORSO_BAND` vertical band (number region) at full width.

    Args:
        img: A full-body player crop.

    Returns:
        The ``[TORSO_BAND[0], TORSO_BAND[1]]`` height band, full width -- a fixed proxy for a
        pose/torso model so the number occupies far more pixels after resize.
    """
    w, h = img.size
    return img.crop((0, round(TORSO_BAND[0] * h), w, round(TORSO_BAND[1] * h)))


def build_transform(*, train: bool, torso: bool = False) -> transforms.Compose:
    """Build the crop preprocessing pipeline.

    Args:
        train: If true, add light photometric/affine augmentation; eval is deterministic.
        torso: If true, first crop to the :data:`TORSO_BAND` number region (Stage 1c).

    Returns:
        A torchvision transform mapping a PIL crop to a normalized ``[3, H, W]`` tensor.
    """
    steps: list[object] = [transforms.Lambda(torso_crop)] if torso else []
    steps.append(transforms.Resize((INPUT_H, INPUT_W)))
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


class MultiHeadJersey(nn.Module):
    """ResNet18 trunk with factorized tens/units digit heads plus a legibility head (Stage 1b).

    The single 100-way Stage-1 head starves rare high numbers (jersey 62 gets ~1 tracklet).
    Factorizing into per-digit heads shares statistics across numbers, and a dedicated legibility
    head carries the ``-1`` decision instead of folding it into a number class.
    """

    def __init__(self, *, pretrained: bool = True) -> None:
        """Build the trunk and three heads.

        Args:
            pretrained: Load ImageNet-1k trunk weights (large win on this small dataset).
        """
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        trunk = resnet18(weights=weights)
        feat_dim = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.tens = nn.Linear(feat_dim, TENS_CLASSES)
        self.units = nn.Linear(feat_dim, UNITS_CLASSES)
        self.legible = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(tens_logits, units_logits, legible_logits)`` for a crop batch."""
        f = self.trunk(x)
        return self.tens(f), self.units(f), self.legible(f)


def heads_to_number_probs(
    tens_p: np.ndarray, units_p: np.ndarray, legible_p: np.ndarray
) -> np.ndarray:
    """Compose factorized head softmaxes into the Stage-1 ``[.., NUM_CLASSES]`` layout.

    Index ``0`` carries the illegible mass ``P(illegible)``; index ``n`` (``1..99``) carries
    ``P(legible) * P(tens(n)) * P(units(n))``. Reusing this layout lets the factorized model share
    the Stage-1 pooling / threshold / eval path unchanged -- the rare-number win lives in how the
    marginals are *estimated* (shared digit features), not in the downstream vote.

    Args:
        tens_p: ``[n, TENS_CLASSES]`` tens-digit softmax rows.
        units_p: ``[n, UNITS_CLASSES]`` units-digit softmax rows.
        legible_p: ``[n, 2]`` legibility softmax rows (column ``1`` = legible).

    Returns:
        ``[n, NUM_CLASSES]`` rows compatible with :func:`tracklet_mean` / :func:`decide`.
    """
    out = np.zeros((tens_p.shape[0], NUM_CLASSES), dtype=np.float32)
    out[:, ILLEGIBLE] = legible_p[:, 0]
    out[:, 1:] = legible_p[:, 1:2] * tens_p[:, _NUM_TENS[1:]] * units_p[:, _NUM_UNITS[1:]]
    return out


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


def decide(
    mean_p: np.ndarray, *, min_conf: float = 0.30, mask: np.ndarray | None = None
) -> tuple[int, float]:
    """Turn a pooled tracklet vector into a label + confidence.

    Args:
        mean_p: A :func:`tracklet_mean` vector. All-zero -> ``(-1, 0.0)``.
        min_conf: Floor on the winning number's probability; below it (or if illegible wins),
            the tracklet reads ``-1``.
        mask: Optional :func:`roster_mask` boolean vector. When given, ``mean_p`` is restricted to
            the kept classes and renormalised over them before the decision (roster-constrained
            decoding), so off-roster numbers cannot win and their mass lifts the valid reads.

    Returns:
        ``(jersey_label, confidence)`` with ``jersey_label`` in ``{-1} u {1..99}``.
    """
    if mask is not None:
        mean_p = mean_p * mask
        total = float(mean_p.sum())
        if total > 0.0:
            mean_p = mean_p / total
    if not mean_p.any():
        return -1, 0.0
    if int(mean_p.argmax()) == ILLEGIBLE:
        return -1, float(mean_p[ILLEGIBLE])
    best_num = int(mean_p[1:].argmax()) + 1
    conf = float(mean_p[best_num])
    return (best_num, conf) if conf >= min_conf else (-1, conf)


def aggregate_votes(
    probs: np.ndarray, *, min_conf: float = 0.30, weighted: bool = True,
    mask: np.ndarray | None = None,
) -> tuple[int, float]:
    """Aggregate per-crop class probabilities into one tracklet label.

    Confidence-weighted mean softmax over the tracklet's crops (:func:`tracklet_mean`), then
    thresholded (:func:`decide`): crops that show the number accumulate consistent mass, noise
    cancels, and a tracklet with no confident number reads ``-1``.

    Args:
        probs: ``[n_crops, NUM_CLASSES]`` softmax rows. Empty -> ``(-1, 0.0)``.
        min_conf: Floor on the winning number's aggregated probability; below it, read ``-1``.
        weighted: Down-weight diffuse crops by their peak probability.
        mask: Optional :func:`roster_mask` for roster-constrained decoding (see :func:`decide`).

    Returns:
        ``(jersey_label, confidence)`` where ``jersey_label`` is ``-1`` or ``1..99``.
    """
    return decide(tracklet_mean(probs, weighted=weighted), min_conf=min_conf, mask=mask)


class JerseyRecognizer:
    """Loaded jersey model + Stage-2 inference contract (crop paths -> number + confidence)."""

    def __init__(
        self, model: nn.Module, device: str, *, min_conf: float = 0.30, torso: bool = False
    ) -> None:
        """Wrap a model for inference.

        Args:
            model: A :func:`build_model` network with trained weights, already on ``device``.
            device: ``"cuda"`` or ``"cpu"``.
            min_conf: Tracklet-level ``-1`` threshold passed to :func:`aggregate_votes`.
            torso: Preprocess crops with the Stage-1c torso band (must match training).
        """
        self.model = model.eval()
        self.device = device
        self.min_conf = min_conf
        self.multihead = isinstance(model, MultiHeadJersey)
        self.tf = build_transform(train=False, torso=torso)

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, device: str | None = None, *, min_conf: float = 0.30
    ) -> JerseyRecognizer:
        """Load a recognizer from a ``train_jersey.py`` checkpoint (``model_state`` key).

        The architecture (single 100-way head vs Stage-1b factorized heads) is auto-detected from
        the checkpoint's ``arch`` tag or its state-dict keys, and the Stage-1c ``torso`` crop flag
        is read from the checkpoint, so old and new checkpoints both load with matching preprocessing.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(Path(ckpt_path), map_location=device)
        sd = state["model_state"] if "model_state" in state else state
        multihead = state.get("arch") == "multihead" or "tens.weight" in sd
        model = (MultiHeadJersey if multihead else build_model)(pretrained=False).to(device)
        model.load_state_dict(sd)
        return cls(model, device, min_conf=min_conf, torso=bool(state.get("torso", False)))

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
                raw = self.model(xb)
            if self.multihead:
                tens_l, units_l, leg_l = raw
                out.append(heads_to_number_probs(
                    torch.softmax(tens_l.float(), 1).cpu().numpy(),
                    torch.softmax(units_l.float(), 1).cpu().numpy(),
                    torch.softmax(leg_l.float(), 1).cpu().numpy(),
                ))
            else:
                out.append(torch.softmax(raw.float(), dim=1).cpu().numpy())
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
