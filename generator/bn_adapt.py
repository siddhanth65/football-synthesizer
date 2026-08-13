"""Test-time BatchNorm statistics adaptation (no labels, no gradients, no loss).

A network trained on one corpus carries that corpus's per-channel activation statistics frozen in
its BatchNorm ``running_mean`` / ``running_var`` buffers. On a test sequence shot in another stadium,
under other lighting, with other kits, those buffers are wrong -- every convolution downstream is
normalised by numbers that describe a different image distribution. Recomputing them from the test
sequence's OWN unlabeled frames costs one forward pass and touches no weight (Schneider et al. 2020,
"Improving robustness against common corruptions by covariate shift adaptation"; the same lever was
measured at +0.18 mAP on the prtreid re-ID backbone in ``results/CLUSTER_SESSION1.md`` section 8).

Nothing here reads a label, a score or a ground-truth file: the only input is raw frames.

Two entry points:

* :func:`recomputing` -- generic context manager, for any module with BatchNorm (used on the CLIP
  embedder's BNNeck, which is a ``BatchNorm1d`` despite the transformer trunk being LayerNorm-only).
* :func:`adapt_yolo` -- driver for an ultralytics detector, which needs the fuse dance documented
  there.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

#: Frames sampled per sequence for the adaptation pass, fixed a priori (results/GSR_V8_W4.md 1).
ADAPT_STRIDE = 5


def bn_layers(model: Any) -> list[Any]:
    """Every BatchNorm module in ``model`` that tracks running statistics.

    Args:
        model: Any ``torch.nn.Module``.

    Returns:
        The BatchNorm submodules, in ``named_modules`` order.
    """
    from torch.nn.modules.batchnorm import _BatchNorm  # noqa: PLC0415

    return [m for m in model.modules() if isinstance(m, _BatchNorm) and m.track_running_stats]


@contextmanager
def recomputing(model: Any, *, momentum: float | None = None, reset: bool = True) -> Iterator[list]:
    """Put only the BatchNorm layers of ``model`` in statistics-collecting mode.

    Everything else stays exactly as the caller left it (eval mode, frozen weights). ``momentum=None``
    selects PyTorch's cumulative moving average, i.e. the running statistics converge to the plain
    mean/variance over every batch seen inside the block, independent of batch order.

    Args:
        model: Module whose BatchNorm layers should collect statistics.
        momentum: BatchNorm momentum during the block; ``None`` = cumulative average over the block.
        reset: Discard the training-corpus statistics first (full recomputation). ``False`` keeps
            them and lets ``momentum`` blend the test sequence in.

    Yields:
        The BatchNorm modules being adapted.
    """
    bns = bn_layers(model)
    saved = [(m.training, m.momentum, m.num_batches_tracked.clone()) for m in bns]
    for m in bns:
        if reset:
            m.reset_running_stats()
        m.momentum = momentum
        m.train()
    try:
        yield bns
    finally:
        for m, (was_training, mom, nbt) in zip(bns, saved, strict=True):
            m.momentum = mom
            m.train(was_training)
            if not reset:
                m.num_batches_tracked.copy_(nbt)


def stats_of(model: Any) -> dict[str, Any]:
    """Snapshot every BatchNorm running buffer as ``{buffer name: cpu tensor}``."""
    from torch.nn.modules.batchnorm import _BatchNorm  # noqa: PLC0415

    out = {}
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm) and m.track_running_stats:
            out[f"{name}.running_mean"] = m.running_mean.detach().cpu().clone()
            out[f"{name}.running_var"] = m.running_var.detach().cpu().clone()
    return out


def apply_stats(model: Any, stats: dict[str, Any] | str | Path) -> int:
    """Load a :func:`stats_of` snapshot into ``model`` in place.

    Args:
        model: Module to write into.
        stats: A snapshot dict, or a path to one saved with ``torch.save``.

    Returns:
        Number of buffers written.

    Raises:
        KeyError: A BatchNorm buffer of ``model`` is missing from the snapshot (wrong architecture).
    """
    import torch  # noqa: PLC0415

    if isinstance(stats, (str, Path)):
        stats = torch.load(stats, map_location="cpu", weights_only=True)
    n = 0
    for key, buf in model.named_buffers():
        if key.endswith(("running_mean", "running_var")):
            if key not in stats:
                raise KeyError(f"BN snapshot has no {key}")
            buf.copy_(stats[key].to(buf.device))
            n += 1
    return n


def adapt_yolo(yolo: Any, frames: Iterable[Any], *, momentum: float | None = None) -> dict[str, Any]:
    """Recompute an ultralytics detector's BatchNorm statistics on ``frames``.

    The fuse dance: ultralytics folds Conv+BN into a single convolution the first time a model
    predicts (``AutoBackend`` calls ``model.fuse()`` in place), after which the model holds **zero**
    BatchNorm modules and no adaptation is possible. So fusion is suppressed for the duration of the
    pass, and the predictor is dropped afterwards -- the next real prediction rebuilds it and fuses
    the model *with the adapted statistics folded in*, so inference runs the ordinary fused path.

    Frames go through the detector's normal ``__call__``, which means the adaptation pass sees
    byte-identical preprocessing (letterbox, scaling, channel order) to inference. Predictions made
    during the pass are discarded; only the BatchNorm buffers change.

    Args:
        yolo: An ``ultralytics.YOLO`` whose model has not predicted yet (still unfused).
        frames: Raw frames of ONE sequence, in the same colour convention the pipeline feeds the
            detector at inference time.
        momentum: See :func:`recomputing`; ``None`` = full cumulative recomputation.

    Returns:
        The adapted snapshot plus ``{"n_bn": .., "n_frames": ..}`` bookkeeping.

    Raises:
        RuntimeError: The model is already fused (no BatchNorm left to adapt).
    """
    from itertools import chain  # noqa: PLC0415

    import torch  # noqa: PLC0415

    model = yolo.model
    if not bn_layers(model):
        raise RuntimeError("model has no BatchNorm layers (already fused?)")
    stream = iter(frames)  # streamed: a 1080p sequence does not fit in RAM as a list
    first = next(stream)
    orig_fuse = model.fuse
    model.fuse = lambda verbose=True: model  # noqa: ARG005 - keep BN alive through predictor setup
    try:
        yolo(first, verbose=False)  # build the predictor first: its setup calls model.eval()
        with recomputing(model, momentum=momentum) as bns, torch.no_grad():
            for frame in chain([first], stream):
                yolo(frame, verbose=False)
        n_bn, n_seen = len(bns), int(bns[0].num_batches_tracked)
    finally:
        model.fuse = orig_fuse
        yolo.predictor = None  # next predict rebuilds and fuses, folding the adapted statistics in
    out: dict[str, Any] = stats_of(model)
    out.update(n_bn=n_bn, n_frames=n_seen, n_batches=n_seen, momentum=momentum)
    return out


def _demo() -> None:
    """Self-check on a toy CNN: cumulative recomputation equals the plain dataset statistics."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()).eval()
    bn = net[1]
    bn.running_mean.fill_(9.0)  # a plainly wrong "training corpus" statistic
    data = [torch.randn(2, 3, 8, 8) * 3 + 1 for _ in range(5)]
    conv = net[0]
    with torch.no_grad():
        acts = torch.cat([conv(x) for x in data])
    want_mean = acts.mean(dim=(0, 2, 3))
    with recomputing(net) as bns, torch.no_grad():
        assert bns == [bn]
        for x in data:
            net(x)
    assert not bn.training, "BN left in train mode"
    assert torch.allclose(bn.running_mean, want_mean, atol=1e-5), (bn.running_mean, want_mean)
    assert int(bn.num_batches_tracked) == len(data)

    snap = stats_of(net)
    fresh = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()).eval()
    assert apply_stats(fresh, snap) == 2
    assert torch.allclose(fresh[1].running_mean, want_mean, atol=1e-6)

    before = bn.running_mean.clone()
    with recomputing(net, momentum=0.1, reset=False), torch.no_grad():
        net(data[0])
    assert not torch.allclose(bn.running_mean, before), "momentum arm did not move the statistics"
    print("bn_adapt demo OK:", f"{len(snap)} buffers, n_batches={int(bn.num_batches_tracked)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
