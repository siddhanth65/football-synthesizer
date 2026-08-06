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

OCR_PERCROP_VERSION = "ocr-percrop-1.2"
"""Stamp for the persisted per-crop OCR evidence (see :meth:`KoshkinaRecognizer.crop_reads`).

``1.1`` added the crop geometry to the run: ``tools.ocr_match`` can widen the foot-point box before
cutting (``--crop-scale``, ``results/OCR_DOMAIN_SHIFT.md`` §7). ``1.2`` adds the *reader weights*:
``eval.gsr_jersey --parseq-ckpt`` selects which PARSeq fine-tune produced the positional softmaxes
(campaign v6 S3). The schema is unchanged in both cases; the stamp moves because the *evidence* a
parquet holds depends on the geometry it was cut at and on the model that read it. The checkpoint
stem is additionally persisted per row (``reader``), so a parquet names its own weights.
"""

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


def parseq_positions_to_probs(p0: Sequence[float], p1: Sequence[float]) -> np.ndarray:
    """Fold PARSeq's two positional softmaxes into a :data:`NUM_CLASSES` jersey distribution (pure).

    The Koshkina STR head decodes a jersey number left-to-right; :mod:`tools.koshkina_str_sidecar`
    emits, per torso crop, the softmax over tokens ``[E, 0, 1, ..., 9]`` at string positions 0 and 1
    (token index ``0`` = end/empty, ``1..10`` = digit ``0..9``). This maps those two 11-vectors onto
    the same ``[NUM_CLASSES]`` layout the ResNet reader uses (index ``0`` = illegible, ``n`` = number
    ``n``), so the identical downstream :func:`decide` / :func:`roster_mask` / vote path applies to
    both readers:

    * empty read (``pos0 = E``) or a leading zero (``pos0 = '0'``) -> illegible mass at index ``0``;
    * ``pos0 = d0`` (``1..9``), ``pos1 = E`` -> single-digit number ``d0``;
    * ``pos0 = d0`` (``1..9``), ``pos1 = d1`` (``0..9``) -> two-digit number ``d0 * 10 + d1``.

    Args:
        p0: Length-11 softmax over ``[E, 0..9]`` at string position 0.
        p1: Length-11 softmax over ``[E, 0..9]`` at string position 1.

    Returns:
        A ``[NUM_CLASSES]`` float32 distribution that sums to 1 (up to float error).
    """
    a0 = np.asarray(p0, dtype=np.float64)
    a1 = np.asarray(p1, dtype=np.float64)
    out = np.zeros(NUM_CLASSES, dtype=np.float64)
    out[ILLEGIBLE] = a0[0] + a0[1]  # empty read, or leading-zero -> not a 1..99 number
    d0 = a0[2:11]  # P(pos0 = digit 1..9); token index d+1 holds digit d
    out[1:10] += d0 * a1[0]  # single-digit numbers 1..9 (pos1 = E)
    out[10:100] += np.outer(d0, a1[1:11]).reshape(-1)  # two-digit numbers 10..99 in order
    return out.astype(np.float32)


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


def percrop_votes(
    probs: np.ndarray, *, min_crop_conf: float = 0.50, min_votes: int = 1, emit_all: bool = False,
) -> list[tuple[int, float]]:
    """Tracklet reads from per-crop distributions, counting only crops that actually read a number.

    This is the densifying alternative to :func:`aggregate_votes`. ``aggregate_votes`` pools *every*
    crop, and :meth:`KoshkinaRecognizer.crop_probs` hands an illegible crop a one-hot on
    :data:`ILLEGIBLE` whose peak is ``1.0`` -- the maximum weight :func:`tracklet_mean` can give a
    row. A tracklet therefore has to be *majority* legible-and-agreeing before the mean's argmax
    stops being ``ILLEGIBLE``, so three confident reads among sixteen crops abstain. Here crops that
    read nothing simply do not vote, which is the aggregation the Koshkina pipeline's own legibility
    filter implies.

    Args:
        probs: ``[n_crops, NUM_CLASSES]`` per-crop distributions for one tracklet.
        min_crop_conf: Floor on a single crop's winning-number mass for it to vote.
        min_votes: Minimum number of agreeing crops before a number is emitted.
        emit_all: Emit every number clearing ``min_votes`` (disagreement is preserved for a solver
            with a confusion prior); otherwise only the highest-mass number.

    Returns:
        ``[(number, mean_crop_confidence), ...]`` ordered by total mass, empty when nothing clears.
    """
    if probs.size == 0:
        return []
    best = probs[:, 1:].argmax(axis=1) + 1
    conf = probs[np.arange(probs.shape[0]), best]
    keep = (conf >= min_crop_conf) & (probs.argmax(axis=1) != ILLEGIBLE)
    tally: dict[int, list[float]] = {}
    for num, c in zip(best[keep], conf[keep]):
        tally.setdefault(int(num), []).append(float(c))
    out = [(n, float(np.mean(cs)), float(np.sum(cs)))
           for n, cs in tally.items() if len(cs) >= min_votes]
    out.sort(key=lambda r: -r[2])
    return [(n, c) for n, c, _m in (out if emit_all else out[:1])]


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


# Torso-crop pose gate (mirrors jersey-number-pipeline helpers.get_points / generate_crops): the
# jersey number sits between the shoulders (COCO idx 5,6) and hips (idx 11,12). torchvision
# KeypointRCNN sets keypoints[..., 2] = 1.0 for every predicted joint, so this 0.4 confidence gate is
# a no-op in practice (matching the reproduced 86.13% chain) -- it only rejects a crop with < 12
# keypoints (i.e. no person detected).
_POSE_CONF = 0.4
_TORSO_JOINTS = (6, 5, 11, 12)  # right/left shoulder, left/right hip
_TORSO_PAD = 5


def torso_from_keypoints(img_bgr: np.ndarray, kp: list | None) -> np.ndarray | None:
    """Crop the shoulder-to-hip torso band from ``img_bgr`` using COCO-17 keypoints (pure).

    Args:
        img_bgr: The full person crop (BGR) the keypoints were estimated on.
        kp: ``[17, 3]`` ``(x, y, score)`` keypoints, or ``None`` when no person was detected.

    Returns:
        The torso sub-image (BGR), or ``None`` if the keypoints are missing/unreliable or the box is
        degenerate.
    """
    if kp is None or len(kp) < 12:
        return None
    pts = []
    for j in _TORSO_JOINTS:
        if kp[j][2] < _POSE_CONF:
            return None
        pts.append(kp[j][:2])
    h, w = img_bgr.shape[:2]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = int(max(0.0, min(xs) - _TORSO_PAD))
    y1 = int(max(0.0, min(ys) - _TORSO_PAD))
    x2 = int(min(float(w - 1), max(xs) + _TORSO_PAD))
    y2 = int(min(float(h - 1), max(ys)))  # helpers.generate_crops does not pad the bottom
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


class _Legibility34(nn.Module):
    """ResNet34 binary legibility classifier, weight-compatible with the pipeline's checkpoint.

    Mirrors ``jersey-number-pipeline networks.LegibilityClassifier34`` (submodule named ``model_ft``
    so the published state dict loads unchanged); ``forward`` returns the sigmoid legibility score.
    """

    def __init__(self) -> None:
        """Build a ResNet34 trunk with a single-logit head (no pretrained download)."""
        super().__init__()
        from torchvision.models import resnet34  # noqa: PLC0415

        self.model_ft = resnet34(weights=None)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sigmoid legibility score for a crop batch."""
        return torch.sigmoid(self.model_ft(x))


class KoshkinaRecognizer:
    """Koshkina & Elder recognizer chain wrapped in the :class:`JerseyRecognizer` reader contract.

    Drop-in for :class:`JerseyRecognizer`: :meth:`crop_probs` maps a list of person-crop paths to
    ``[n, NUM_CLASSES]`` softmax rows aligned 1:1 with the input, so the close-up anchor funnel
    (kit gate, roster mask, N-agreement) swaps readers by a single flag. Internally, per crop:

    1. ResNet34 legibility gate (main env, GPU) -- crops scoring ``<= leg_thresh`` read illegible.
    2. torchvision KeypointRCNN pose (main env, GPU) + :func:`torso_from_keypoints` -> torso RoI.
    3. SoccerNet-fine-tuned PARSeq in the py3.11 sidecar (subprocess JSON handoff, resumable) ->
       positional softmaxes, folded to the ``[NUM_CLASSES]`` layout by
       :func:`parseq_positions_to_probs`.

    The main env never imports strhub; PARSeq runs only in the sidecar interpreter. The work dir is
    wiped on construction (the caller's per-chunk checkpoint provides resume; a redone chunk gets
    fresh torso crops), matching the pipeline's resumable-by-disk-state pattern.
    """

    def __init__(
        self,
        *,
        legibility_weights: str | Path,
        sidecar_python: str | Path,
        sidecar_script: str | Path,
        parseq_ckpt: str | Path,
        parseq_repo: str | Path,
        work_dir: str | Path,
        device: str | None = None,
        min_conf: float = 0.30,
        leg_thresh: float = 0.5,
        leg_batch: int = 128,
        pose_batch: int = 8,
    ) -> None:
        """Load the legibility + pose models and prepare the (wiped) sidecar work dir.

        Args:
            legibility_weights: Path to ``legibility_resnet34_soccer_*.pth``.
            sidecar_python: Path to the py3.11 sidecar interpreter (``jersey-str-env``).
            sidecar_script: Path to :mod:`tools.koshkina_str_sidecar`.
            parseq_ckpt: Path to the SoccerNet-fine-tuned PARSeq checkpoint.
            parseq_repo: ``jersey-number-pipeline`` root (holds ``str/parseq`` for strhub).
            work_dir: Scratch dir for torso crops + the sidecar's positional-softmax JSON (wiped).
            device: ``"cuda"``/``"cpu"`` (auto-detected when ``None``).
            min_conf: Interface-parity threshold for :meth:`predict_tracklet` (unused by the funnel).
            leg_thresh: Legibility sigmoid floor; ``<=`` reads illegible.
            leg_batch: Legibility forward-pass batch size.
            pose_batch: KeypointRCNN batch size (small for the 4 GB GPU alongside YOLO).
        """
        import shutil  # noqa: PLC0415
        import time  # noqa: PLC0415

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_conf = min_conf
        self.leg_thresh = leg_thresh
        self.leg_batch = leg_batch
        self.pose_batch = pose_batch
        # The sidecar runs with cwd=parseq_repo, so every path handed to it must be absolute.
        self.sidecar_python = Path(sidecar_python).resolve()
        self.sidecar_script = Path(sidecar_script).resolve()
        self.parseq_ckpt = Path(parseq_ckpt).resolve()
        self.parseq_repo = Path(parseq_repo).resolve()
        self.work_dir = Path(work_dir).resolve()
        # Fresh torso crops per run: torso names (c{call}_{i}) restart at call 0, so a stale crop of
        # the same name would be re-used by the resumable sidecar -> the wipe must fully succeed.
        # Retry past Windows' transient WinError 145 before giving up.
        for _ in range(5):
            if not self.work_dir.exists():
                break
            try:
                shutil.rmtree(self.work_dir)
            except OSError:
                time.sleep(0.4)
        self.torso_dir = self.work_dir / "torso"
        self.torso_dir.mkdir(parents=True, exist_ok=True)
        self.parseq_json = self.work_dir / "parseq_positions.json"
        self._call = 0

        self._leg_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        self._leg = _Legibility34().to(self.device).eval()
        sd = torch.load(Path(legibility_weights), map_location=self.device)
        if hasattr(sd, "_metadata"):
            del sd._metadata
        self._leg.load_state_dict(sd)
        self._pose, self._pose_tf = self._build_pose()

    def _build_pose(self):  # noqa: ANN202
        """KeypointRCNN pose model + its input transform (top-down COCO-17, shrunk input pyramid)."""
        from torchvision.models.detection import (  # noqa: PLC0415
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.5)
        model.transform.min_size = (256,)  # crops are tiny; do not up-sample to 800 px
        model.transform.max_size = 480
        return model.to(self.device).eval(), weights.transforms()

    @torch.no_grad()
    def _legibility_scores(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Per-crop ResNet34 legibility sigmoid score (``NaN`` where the file is unreadable)."""
        out = np.full(len(paths), np.nan, dtype=np.float32)
        tensors: list[torch.Tensor] = []
        idxs: list[int] = []
        for i, p in enumerate(paths):
            try:
                tensors.append(self._leg_tf(Image.open(p).convert("RGB")))
                idxs.append(i)
            except (OSError, ValueError):
                continue
        use_amp = self.device == "cuda"
        for s in range(0, len(tensors), self.leg_batch):
            xb = torch.stack(tensors[s : s + self.leg_batch]).to(self.device)
            with torch.amp.autocast(self.device, enabled=use_amp):
                scores = self._leg(xb).float().squeeze(1).cpu().numpy()
            for j, sc in enumerate(scores):
                out[idxs[s + j]] = float(sc)
        return out

    def _legible_indices(self, paths: Sequence[str | Path]) -> list[int]:
        """Return the input indices whose crop passes the ResNet34 legibility gate."""
        scores = self._legibility_scores(paths)
        return [i for i, sc in enumerate(scores) if np.isfinite(sc) and sc > self.leg_thresh]

    @torch.no_grad()
    def _write_torso_crops(
        self, legible_idx: list[int], paths: Sequence[str | Path]
    ) -> dict[str, int]:
        """Pose + torso-crop the legible crops; return ``{torso_filename: input_index}``."""
        import cv2  # noqa: PLC0415

        torso_map: dict[str, int] = {}
        batch: list[torch.Tensor] = []
        bmeta: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            if not batch:
                return
            outs = self._pose([self._pose_tf(b.to(self.device)) for b in batch])
            for (i, bgr), out in zip(bmeta, outs):
                kp = out["keypoints"][0].cpu().numpy().tolist() if len(out["boxes"]) else None
                torso = torso_from_keypoints(bgr, kp)
                if torso is not None and torso.size:
                    name = f"c{self._call:04d}_{i:06d}.jpg"
                    cv2.imwrite(str(self.torso_dir / name), torso)
                    torso_map[name] = i
            batch.clear()
            bmeta.clear()

        for i in legible_idx:
            bgr = cv2.imread(str(paths[i]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(rgb).permute(2, 0, 1))  # uint8 CHW
            bmeta.append((i, bgr))
            if len(batch) >= self.pose_batch:
                flush()
        flush()
        return torso_map

    def _run_sidecar(self) -> None:
        """Invoke the py3.11 PARSeq sidecar over the torso dir (resumable JSON, raises on failure)."""
        import subprocess  # noqa: PLC0415

        subprocess.run(
            [
                str(self.sidecar_python), str(self.sidecar_script),
                "--crops-dir", str(self.torso_dir),
                "--out-json", str(self.parseq_json),
                "--ckpt", str(self.parseq_ckpt),
                "--parseq-repo", str(self.parseq_repo),
            ],
            check=True,
            cwd=str(self.parseq_repo),
        )

    def crop_reads(
        self, paths: Sequence[str | Path]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run the chain and return both the folded softmax rows and the raw per-crop evidence.

        :meth:`crop_probs` throws the per-crop evidence away (it folds straight into a tracklet
        vote), which is what made the digit-confusion prior unfittable in Stage 2. This variant
        keeps it, so callers can persist ``(legibility, torso, PARSeq positional softmax)`` per crop
        and re-aggregate offline without a second GPU pass.

        Args:
            paths: Person-crop image paths, aligned 1:1 with every returned array's first axis.

        Returns:
            ``(probs, detail)`` where ``probs`` is ``[n, NUM_CLASSES]`` (see :meth:`crop_probs`) and
            ``detail`` holds ``leg`` ``[n]`` legibility sigmoid (``NaN`` = unreadable file),
            ``torso`` ``[n]`` bool (a pose torso RoI was produced), and ``p0`` / ``p1``
            ``[n, 11]`` PARSeq positional softmaxes (all-``NaN`` rows where no read happened).
        """
        n = len(paths)
        out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        out[:, ILLEGIBLE] = 1.0
        detail = {"leg": np.full(n, np.nan, np.float32), "torso": np.zeros(n, bool),
                  "p0": np.full((n, 11), np.nan, np.float32),
                  "p1": np.full((n, 11), np.nan, np.float32)}
        if n == 0:
            return out, detail
        detail["leg"] = self._legibility_scores(paths)
        legible_idx = [i for i, sc in enumerate(detail["leg"])
                       if np.isfinite(sc) and sc > self.leg_thresh]
        torso_map = self._write_torso_crops(legible_idx, paths)
        self._call += 1
        if self.device == "cuda":
            torch.cuda.empty_cache()
        if torso_map:
            self._run_sidecar()
            positions = json.loads(self.parseq_json.read_text(encoding="utf-8"))
            for name, i in torso_map.items():
                detail["torso"][i] = True
                entry = positions.get(name)
                if entry is not None:
                    detail["p0"][i] = entry["p0"]
                    detail["p1"][i] = entry["p1"]
                    out[i] = parseq_positions_to_probs(entry["p0"], entry["p1"])
        return out, detail

    def crop_probs(self, paths: Sequence[str | Path], batch_size: int = 256) -> np.ndarray:
        """Softmax rows for each crop via the Koshkina chain (illegible one-hot when no read).

        Args:
            paths: Person-crop image paths (one close-up shot's crops). Aligned 1:1 with the output.
            batch_size: Unused (batches are set at construction); kept for reader-interface parity.

        Returns:
            ``[len(paths), NUM_CLASSES]`` softmax rows. A crop that fails legibility, has no pose, or
            gets no PARSeq read reads illegible (mass on :data:`ILLEGIBLE`).
        """
        return self.crop_reads(paths)[0]
