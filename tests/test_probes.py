"""
Smoke tests for trainprobe v0.1.

All tests run on CPU, require no WandB, and complete in < 5 seconds total.
"""
import json
import os
import tempfile
import warnings

import pytest
import torch
import torch.nn as nn

import trainprobe as tp
from trainprobe.aggregator import SignalAggregator
from trainprobe.hooks import HookManager, _move_to_cpu, _move_to_device
from trainprobe.probes.geometry import EffectiveRankProbe, CollapseProbe, DeadNeuronProbe, _effective_rank
from trainprobe.probes.weight_grad import GradNormProbe, UpdateRatioProbe
from trainprobe.probes.task import LinearProbeProbe, SpuriousFeatureProbe, _lstsq_accuracy
from trainprobe.registry import auto_filter, LayerFilter
from trainprobe.schedule import EveryNSteps, OnLossSpike, Exponential, Composite
from trainprobe.backends import JsonlBackend, StdoutBackend, CompositeBackend, resolve_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model():
    return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))


def _run_one_step(model, scope, step=0, loss=1.0):
    x = torch.randn(4, 32)
    out = model(x).sum()
    out.backward()
    scope.step(step, loss=loss)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def test_effective_rank_rank1():
    # Truly rank-1: all rows identical
    x = torch.randn(1, 64).expand(32, -1).clone()
    rank = _effective_rank(x)
    assert rank < 1.5, f"Expected rank ≈ 1 for rank-1 matrix, got {rank:.4f}"


def test_effective_rank_full_rank():
    torch.manual_seed(42)
    x = torch.randn(32, 8)
    rank = _effective_rank(x)
    # Full-rank random matrix should have effective rank near min(32, 8) = 8
    assert rank > 4.0, f"Expected high effective rank for random matrix, got {rank:.4f}"


def test_effective_rank_single_row():
    x = torch.randn(1, 16)
    rank = _effective_rank(x)
    assert rank >= 1.0


def test_effective_rank_zero_matrix():
    x = torch.zeros(8, 16)
    rank = _effective_rank(x)
    assert rank == 1.0


# ---------------------------------------------------------------------------
# Geometry probes
# ---------------------------------------------------------------------------

def test_effective_rank_probe():
    probe = EffectiveRankProbe()
    acts = {"layer1": torch.randn(16, 32)}
    result = probe.run(acts, None, 0)
    assert "layer1/rank" in result
    assert isinstance(result["layer1/rank"], float)


def test_collapse_probe_collapsed():
    probe = CollapseProbe()
    # All rows identical → near-zero sv_ratio
    x = torch.randn(1, 32).expand(16, -1).clone()
    result = probe.run({"layer1": x}, None, 0)
    assert "layer1/norm_cv" in result
    assert "layer1/sv_ratio" in result
    assert result["layer1/sv_ratio"] < 0.01


def test_collapse_probe_healthy():
    probe = CollapseProbe()
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    result = probe.run({"layer1": x}, None, 0)
    assert result["layer1/sv_ratio"] > 0.05  # not fully collapsed


def test_dead_neuron_probe_all_dead():
    probe = DeadNeuronProbe()
    result = probe.run({"l": torch.zeros(8, 16)}, None, 0)
    assert result["l/dead_fraction"] == pytest.approx(1.0)


def test_dead_neuron_probe_none_dead():
    probe = DeadNeuronProbe()
    torch.manual_seed(1)
    result = probe.run({"l": torch.randn(8, 16).abs()}, None, 0)
    assert result["l/dead_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Weight / gradient probes
# ---------------------------------------------------------------------------

def test_grad_norm_probe():
    model = _small_model()
    x = torch.randn(4, 32)
    loss = model(x).sum()
    loss.backward()

    probe = GradNormProbe()
    result = probe.run(None, model, 0)
    assert "total_grad_norm" in result
    assert result["total_grad_norm"] > 0.0
    # Should have per-parameter entries
    assert any("/grad_norm" in k for k in result)


def test_grad_norm_probe_no_grad():
    model = _small_model()
    probe = GradNormProbe()
    result = probe.run(None, model, 0)
    assert result["total_grad_norm"] == pytest.approx(0.0)


def test_update_ratio_probe():
    model = _small_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    probe = UpdateRatioProbe()

    # First step: captures prev_params, no ratios yet
    x = torch.randn(4, 32)
    model(x).sum().backward()
    result0 = probe.run(None, model, 0)
    opt.step(); opt.zero_grad()
    assert len(result0) == 0  # no prev snapshot yet

    # Second step: ratios computed
    model(x).sum().backward()
    result1 = probe.run(None, model, 1)
    opt.step(); opt.zero_grad()
    assert len(result1) > 0
    assert all("update_ratio" in k for k in result1)
    assert all(v >= 0.0 for v in result1.values())


# ---------------------------------------------------------------------------
# Task probes
# ---------------------------------------------------------------------------

def test_lstsq_accuracy_binary():
    torch.manual_seed(0)
    N, D = 64, 16
    X = torch.randn(N, D)
    w_true = torch.randn(D)
    labels = (X @ w_true > 0).long()

    X_centered = X - X.mean(0)
    acc = _lstsq_accuracy(X_centered, labels)
    # With a linear-separable task accuracy should be high
    assert acc > 0.8, f"Expected >0.8 accuracy, got {acc:.3f}"


def test_lstsq_accuracy_multiclass():
    torch.manual_seed(1)
    N, D, K = 128, 32, 4
    X = torch.randn(N, D)
    W_true = torch.randn(D, K)
    labels = (X @ W_true).argmax(dim=1)

    X_centered = X - X.mean(0)
    acc = _lstsq_accuracy(X_centered, labels)
    assert acc > 0.7


def test_linear_probe_probe():
    torch.manual_seed(2)
    N, D = 32, 16
    acts = torch.randn(N, D)
    labels = torch.randint(0, 3, (N,))

    probe = LinearProbeProbe()
    result = probe.run({"layer1": acts}, None, 0, labels=labels)
    assert "layer1/accuracy" in result
    assert 0.0 <= result["layer1/accuracy"] <= 1.0


def test_linear_probe_probe_no_labels():
    probe = LinearProbeProbe()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = probe.run({"layer1": torch.randn(8, 16)}, None, 0)
        # Warns exactly once
        result2 = probe.run({"layer1": torch.randn(8, 16)}, None, 1)
    assert result == {}
    assert result2 == {}
    assert sum(1 for warning in w if "LinearProbeProbe" in str(warning.message)) == 1


def test_spurious_probe():
    torch.manual_seed(3)
    N, D = 32, 16
    acts = torch.randn(N, D)
    genuine = torch.randint(0, 2, (N,))
    spurious = torch.randint(0, 2, (N,))

    probe = SpuriousFeatureProbe()
    result = probe.run({"l": acts}, None, 0, genuine_labels=genuine, spurious_labels=spurious)
    for k in ("l/genuine_accuracy", "l/spurious_accuracy", "l/gap"):
        assert k in result
    assert result["l/gap"] == pytest.approx(
        result["l/genuine_accuracy"] - result["l/spurious_accuracy"]
    )


def test_spurious_probe_missing_labels():
    probe = SpuriousFeatureProbe()
    assert probe.run({"l": torch.randn(8, 16)}, None, 0) == {}


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def test_every_n_steps():
    s = EveryNSteps(10)
    fired = [i for i in range(50) if s.should_run(i)]
    assert fired == list(range(0, 50, 10))


def test_on_loss_spike_fires():
    s = OnLossSpike(threshold=2.0, cooldown=5, fallback_every=1000)
    # Normal losses
    for i in range(10):
        s.should_run(i, loss=1.0)
    # Big spike
    assert s.should_run(10, loss=3.0)
    # Cooldown: same spike right after should NOT fire
    assert not s.should_run(11, loss=3.0)


def test_on_loss_spike_fallback_warns_once():
    s = OnLossSpike(fallback_every=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for i in range(20):
            s.should_run(i, loss=None)
    spike_warns = [x for x in w if "OnLossSpike" in str(x.message)]
    assert len(spike_warns) == 1


def test_exponential_schedule():
    s = Exponential(base=2, start=4)
    fired = [i for i in range(64) if s.should_run(i)]
    # Should fire at 4, 8, 16, 32
    assert fired[:4] == [4, 8, 16, 32]


def test_composite_schedule():
    s = Composite([EveryNSteps(7), EveryNSteps(11)])
    fired = [i for i in range(100) if s.should_run(i)]
    expected = sorted(set(range(0, 100, 7)) | set(range(0, 100, 11)))
    assert fired == expected


# ---------------------------------------------------------------------------
# Registry / layer selection
# ---------------------------------------------------------------------------

def test_auto_filter_small_model():
    model = _small_model()
    reg = auto_filter(model)
    # 2 Linear layers → stride=1, both selected (ReLU has no params)
    assert len(reg) == 2
    assert all(isinstance(n, str) for n in reg.names)


def test_auto_filter_excludes_norm():
    model = nn.Sequential(
        nn.Linear(16, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 8),
    )
    reg = auto_filter(model)
    assert all("norm" not in n.lower() for n in reg.names)


def test_layers_namespace_by_name():
    reg = tp.layers.by_name("0", "2")
    assert reg.names == ["0", "2"]


def test_layers_namespace_linear():
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    reg = tp.layers.linear(model)
    assert len(reg) == 2


# ---------------------------------------------------------------------------
# HookManager
# ---------------------------------------------------------------------------

def test_hook_manager_captures_activations():
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
    probe_batch = torch.randn(8, 32)
    device = torch.device("cpu")

    reg = auto_filter(model)
    hm = HookManager()
    cache = hm.run_forward(model, reg.names, probe_batch, device)

    assert len(cache) == 2  # two Linear layers
    for _, acts in cache.items():
        assert acts.dim() == 2  # (B, D)


def test_hook_manager_no_permanent_hooks():
    model = nn.Linear(8, 4)
    probe_batch = torch.randn(4, 8)
    hm = HookManager()
    hm.run_forward(model, [""], probe_batch, torch.device("cpu"))
    # No hooks should remain after run
    assert hm._handles == []


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def test_jsonl_backend():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        backend = JsonlBackend(path)
        backend.log({"trainprobe/loss": 1.5, "trainprobe/effective_rank/l/rank": 3.2}, step=10)
        with open(path) as f:
            record = json.loads(f.readline())
        assert record["step"] == 10
        assert record["trainprobe/loss"] == pytest.approx(1.5)
    finally:
        os.unlink(path)


def test_jsonl_backend_empty_metrics():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        backend = JsonlBackend(path)
        backend.log({}, step=0)  # should not write anything
        assert os.path.getsize(path) == 0
    finally:
        os.unlink(path)


def test_resolve_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown logger"):
        resolve_backend("notabackend")


def test_resolve_backend_jsonl_by_extension():
    b = resolve_backend("/tmp/foo.jsonl")
    assert isinstance(b, JsonlBackend)


# ---------------------------------------------------------------------------
# End-to-end: attach + step
# ---------------------------------------------------------------------------

def test_attach_and_step_basic():
    model = _small_model()
    probe_batch = torch.randn(16, 32)
    scope = tp.attach(model, probe_batch=probe_batch, schedule=tp.every(2), logger="stdout")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(6):
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        scope.step(step, loss=loss.item())


def test_attach_jsonl_end_to_end():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        model = _small_model()
        probe_batch = torch.randn(16, 32)
        scope = tp.attach(
            model,
            probe_batch=probe_batch,
            schedule=tp.every(1),
            logger=path,
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(3):
            model(torch.randn(4, 32)).sum().backward()
            opt.step(); opt.zero_grad()
            scope.step(step, loss=1.0)

        with open(path) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 3
        # Activation probe ran at step 0, 1, 2 — should have effective rank
        assert any("effective_rank" in str(r) for r in records)
    finally:
        os.unlink(path)


def test_capture_batch_auto():
    """scope.capture_batch is called before first scheduled probe step."""
    model = _small_model()
    scope = tp.attach(model, probe_batch=None, schedule=tp.every(5), logger="stdout")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(10):
        batch = torch.randn(8, 32)
        if step == 0:
            scope.capture_batch(batch)
        model(batch).sum().backward()
        opt.step(); opt.zero_grad()
        scope.step(step, loss=1.0)


def test_no_probe_batch_warns():
    model = _small_model()
    scope = tp.attach(model, probe_batch=None, schedule=tp.every(1), logger="stdout")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scope.step(0, loss=1.0)
    assert any("probe_batch" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Aggregator / CUSUM
# ---------------------------------------------------------------------------

def test_cusum_detects_step_change():
    agg = SignalAggregator(warmup=5, k=0.5, h=3.0, cooldown=2)
    key = "trainprobe/effective_rank/layer1/rank"

    # Baseline: stable at 5.0
    for i in range(8):
        agg.update({key: 5.0}, step=i)
        assert not agg.phase_transition_detected()

    # Sudden shift up to 10.0
    for i in range(8, 12):
        agg.update({key: 10.0}, step=i)

    assert agg.phase_transition_detected()


def test_cusum_cooldown():
    agg = SignalAggregator(warmup=5, k=0.5, h=3.0, cooldown=10)
    key = "trainprobe/effective_rank/layer1/rank"

    for i in range(8):
        agg.update({key: 5.0}, step=i)
    for i in range(8, 15):
        agg.update({key: 20.0}, step=i)

    detections = sum(1 for i in range(8, 15) if agg._transition_detected)
    # At most one detection within cooldown window
    assert detections <= 1
