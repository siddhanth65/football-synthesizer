"""Tests for test-time BatchNorm statistics adaptation (`generator.bn_adapt`). CPU only."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from generator.bn_adapt import (  # noqa: E402
    apply_stats,
    bn_layers,
    recomputing,
    stats_of,
)


def _net(seed: int = 0):
    """A tiny conv+BN net in eval mode with deliberately wrong running statistics."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()).eval()
    net[1].running_mean.fill_(9.0)
    net[1].running_var.fill_(4.0)
    return net


def _data(n: int = 5):
    torch.manual_seed(1)
    return [torch.randn(2, 3, 8, 8) * 3 + 1 for _ in range(n)]


def test_cumulative_recomputation_equals_dataset_statistics():
    """momentum=None must converge to the plain mean/variance of everything seen."""
    net, data = _net(), _data()
    with torch.no_grad():
        acts = torch.cat([net[0](x) for x in data])
    with recomputing(net), torch.no_grad():
        for x in data:
            net(x)
    assert torch.allclose(net[1].running_mean, acts.mean(dim=(0, 2, 3)), atol=1e-5)
    # BatchNorm accumulates the UNBIASED per-batch variance, averaged over batches.
    want_var = torch.stack([net[0](x).var(dim=(0, 2, 3), unbiased=True) for x in data]).mean(0)
    assert torch.allclose(net[1].running_var, want_var, atol=1e-4)
    assert int(net[1].num_batches_tracked) == len(data)


def test_block_restores_eval_mode_and_touches_no_weight():
    """The adaptation must leave the module exactly as it found it except for the BN buffers."""
    net = _net()
    weights = {k: v.clone() for k, v in net.state_dict().items()
               if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    assert not any(m.training for m in net.modules())
    with recomputing(net) as bns, torch.no_grad():
        assert bns == bn_layers(net) and all(m.training for m in bns)
        assert not net[0].training, "only BatchNorm may enter train mode"
        for x in _data(2):
            net(x)
    assert not any(m.training for m in net.modules()), "left in train mode"
    for k, v in weights.items():
        assert torch.equal(net.state_dict()[k], v), f"{k} changed"


def test_no_gradient_is_produced():
    """No parameter may acquire a gradient: this is a statistics update, not a training step."""
    net = _net()
    with recomputing(net), torch.no_grad():
        for x in _data(2):
            net(x)
    assert all(p.grad is None for p in net.parameters())


def test_reset_discards_the_training_corpus_statistics():
    """reset=True forgets the checkpoint's numbers; reset=False blends into them."""
    net = _net()
    with recomputing(net, reset=True), torch.no_grad():
        net(_data(1)[0])
    assert abs(float(net[1].running_mean.mean()) - 9.0) > 1.0

    net2 = _net()
    before = net2[1].running_mean.clone()
    with recomputing(net2, momentum=0.1, reset=False), torch.no_grad():
        net2(_data(1)[0])
    moved = net2[1].running_mean - before
    assert torch.all(moved.abs() > 0), "momentum arm did not move the statistics"
    assert float(moved.abs().max()) < abs(9.0 - float(net[1].running_mean.mean()))


def test_snapshot_roundtrip_and_isolation(tmp_path):
    """A snapshot transplants BN buffers between identical architectures, and only those."""
    net = _net()
    with recomputing(net), torch.no_grad():
        for x in _data():
            net(x)
    snap = stats_of(net)
    assert set(snap) == {"1.running_mean", "1.running_var"}

    path = tmp_path / "bn.pt"
    torch.save(snap, path)
    fresh = _net(seed=3)
    conv_before = fresh[0].weight.clone()
    assert apply_stats(fresh, path) == 2
    assert torch.allclose(fresh[1].running_mean, net[1].running_mean)
    assert torch.equal(fresh[0].weight, conv_before), "apply_stats must not touch weights"


def test_apply_stats_rejects_a_foreign_snapshot():
    """A snapshot from another architecture must fail loudly, not silently half-load."""
    with pytest.raises(KeyError):
        apply_stats(_net(), {"nope.running_mean": torch.zeros(4)})


def test_per_sequence_isolation():
    """Two 'sequences' adapted from the same checkpoint must not see each other's data."""
    torch.manual_seed(7)
    seq_a = [torch.randn(2, 3, 8, 8) * 1.0 - 5.0 for _ in range(3)]
    seq_b = [torch.randn(2, 3, 8, 8) * 1.0 + 5.0 for _ in range(3)]
    snaps = []
    for seq in (seq_a, seq_b):
        net = _net()  # a fresh model per sequence, exactly as the runner rebuilds one
        with recomputing(net), torch.no_grad():
            for x in seq:
                net(x)
        snaps.append(stats_of(net))
    assert not torch.allclose(snaps[0]["1.running_mean"], snaps[1]["1.running_mean"])

    net = _net()  # sequential adaptation on ONE model must still reset, not accumulate
    with recomputing(net), torch.no_grad():
        for x in seq_a:
            net(x)
    with recomputing(net), torch.no_grad():
        for x in seq_b:
            net(x)
    assert torch.allclose(net[1].running_mean, snaps[1]["1.running_mean"], atol=1e-6)


def test_stage_is_off_by_default():
    """The shipped chain must be untouched: the module flag defaults to None."""
    import generator.extract as ex

    assert ex.BN_STATS is None
