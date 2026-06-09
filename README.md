# trainprobe

> WandB logs your loss.  trainprobe tells you what your model is learning.

Your training loss is healthy.  Your representations might still be collapsing.  trainprobe attaches to any PyTorch training loop and emits interpretability-grade signals — effective rank, feature collapse, dead neurons, gradient norms — directly into your WandB run alongside standard metrics.

```
pip install trainprobe          # core (torch + numpy)
pip install trainprobe[wandb]   # + WandB logging
```

## Quickstart

```python
import trainprobe as tp

scope = tp.attach(model, probe_batch=val_batch)   # one line
for step, batch in enumerate(loader):
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scope.step(step, loss=loss.item())            # one line
```

All metrics appear in WandB under the `trainprobe/*` namespace.  No changes to your model, no custom training loop required.

---

## Framework integrations (zero-touch)

**HuggingFace Trainer**
```python
from trainprobe.integrations import TrainProbeCallback

trainer = Trainer(
    model=model,
    callbacks=[TrainProbeCallback(probe_batch=val_batch)],
)
```

**PyTorch Lightning**
```python
from trainprobe.integrations import LightningTrainProbeCallback

trainer = pl.Trainer(callbacks=[LightningTrainProbeCallback()])
# Probe batch is captured automatically from the first training batch.
```

---

## What each probe measures

All five default probes run out-of-the-box with no configuration.

### EffectiveRankProbe  `trainprobe/effective_rank/`

`exp(H(σ / ‖σ‖₁))` — the exponential Shannon entropy of the normalized singular value distribution of each layer's activation matrix.

- **Range**: `[1, min(batch, dim)]`
- **Healthy training**: rises during early training as the encoder spreads features across more directions, then plateaus.
- **Collapse signal**: drops sharply toward 1 when representations become nearly rank-1.  A loss curve that looks fine while effective rank is falling is the canonical SSL collapse failure mode.

### CollapseProbe  `trainprobe/collapse/`

Two complementary collapse diagnostics:

| Metric | Definition | Collapse = |
|---|---|---|
| `norm_cv` | std / mean of per-sample L2 norms | near **0** (hyperspherical collapse) |
| `sv_ratio` | min / max singular value | near **0** (dimensional collapse) |

Use `sv_ratio` to catch dimensional collapse (one direction dominates) even when norms look healthy.

### DeadNeuronProbe  `trainprobe/dead_neurons/`

Fraction of neurons whose max absolute activation across the probe batch is exactly zero.  Measures network utilization.  Rising dead-neuron fraction often precedes representational collapse and indicates poor initialization or overly aggressive weight decay.

### GradNormProbe  `trainprobe/grad_norm/`  *(free probe — runs every step)*

Per-parameter and total L2 gradient norm.  Useful for diagnosing gradient vanishing/explosion and for checking that all parts of the network are receiving signal.  Run every step at near-zero overhead (no forward pass required).

### UpdateRatioProbe  `trainprobe/update_ratio/`  *(free probe — runs every step)*

`‖Δw‖ / ‖w‖` per parameter across consecutive optimizer steps — the standard "lr-free" learning rate diagnostic.  Healthy range: `[1e-3, 1e-2]`.  Values below `1e-4` indicate a parameter is barely moving; values above `0.1` suggest updates may be too large.

---

## WandB metric reference

| Key | Type | When logged |
|---|---|---|
| `trainprobe/loss` | scalar | every step (if `loss=` passed) |
| `trainprobe/effective_rank/{layer}/rank` | scalar | every probe step |
| `trainprobe/collapse/{layer}/norm_cv` | scalar | every probe step |
| `trainprobe/collapse/{layer}/sv_ratio` | scalar | every probe step |
| `trainprobe/dead_neurons/{layer}/dead_fraction` | scalar | every probe step |
| `trainprobe/grad_norm/{param}/grad_norm` | scalar | every step |
| `trainprobe/grad_norm/total_grad_norm` | scalar | every step |
| `trainprobe/update_ratio/{param}/update_ratio` | scalar | every step |
| `trainprobe/events/phase_transition` | 0/1 marker | on CUSUM detection |

Layer names use `/` instead of `.` so WandB renders them as nested groups.

**v0.2 probes** (in `trainprobe.probes.task`, not in the default suite):

| Key | When logged |
|---|---|
| `trainprobe/linear_probe/{layer}/accuracy` | every probe step (requires `labels=`) |
| `trainprobe/spurious/{layer}/genuine_accuracy` | every probe step |
| `trainprobe/spurious/{layer}/spurious_accuracy` | every probe step |
| `trainprobe/spurious/{layer}/gap` | every probe step |

---

## Scheduling

Activation probes require a forward pass on the probe batch.  By default they run every 250 steps.

```python
tp.attach(model, schedule=tp.every(250))                    # default
tp.attach(model, schedule=tp.on_loss_spike(threshold=2.0))  # fire on spikes
tp.attach(model, schedule=tp.exponential(base=2, start=100))
tp.attach(model, schedule=tp.composite([
    tp.every(500),
    tp.on_loss_spike(threshold=1.5, cooldown=50),
]))
```

Phase transition detection fires an extra probe pass automatically when the CUSUM test on effective rank detects a structural change in representation geometry.

---

## Logging backends

```python
tp.attach(model, logger="wandb")          # default; requires wandb
tp.attach(model, logger="jsonl")          # writes trainprobe_log.jsonl
tp.attach(model, logger="run.jsonl")      # explicit path
tp.attach(model, logger="stdout")         # debugging
```

---

## Layer selection

```python
# Default: auto_filter picks based on model depth
tp.attach(model)

# Manual
tp.attach(model, layers=tp.layers.by_name("encoder.layer.0", "encoder.layer.11"))
tp.attach(model, layers=tp.layers.linear(model))
```

`auto_filter` excludes norm layers (LayerNorm, BatchNorm, etc.) and applies stride-based subsampling for large models (>96 parameterized layers → every 4th).

---

## v0.2 probes (available now, not in default suite)

```python
from trainprobe.probes.task import LinearProbeProbe, SpuriousFeatureProbe

scope = tp.attach(
    model,
    probes=tp.default_suite() + [LinearProbeProbe()],
)
# Then pass labels each step:
scope.step(step, loss=loss.item(), labels=y)
```

**SpuriousFeatureProbe** is unique to this library.  It fits two linear probes — one for genuine task labels, one for known spurious/shortcut labels — and reports the accuracy gap per layer.  A negative gap (`spurious_accuracy > genuine_accuracy`) is the signature of shortcut learning.

```python
scope.step(step, loss=loss.item(),
           genuine_labels=y_true,
           spurious_labels=y_spurious)
```

---

## How it works

trainprobe separates two tiers of probes:

**Free probes** (`GradNormProbe`, `UpdateRatioProbe`) inspect `model.parameters()` only and run every step at ~zero overhead.

**Activation probes** (`EffectiveRankProbe`, `CollapseProbe`, `DeadNeuronProbe`) arm forward hooks on selected layers, run a single `torch.no_grad()` forward pass on a fixed probe batch, disarm the hooks, and analyse the cached activations.  Hooks are never permanently registered — the arm/disarm cycle wraps exactly one probe pass.

The probe batch is stored on CPU and moved to the model device at probe time.

---

## Comparison with TRACE (EMNLP 2025)

[TRACE](https://arxiv.org/abs/2501.02308) is the closest prior work.  Key differences:

| | trainprobe | TRACE |
|---|---|---|
| **Modality** | Any PyTorch model (SSL, vision, LLMs) | Language models only |
| **Corpus** | Bring your own data | Coupled to ABSynth synthetic corpus |
| **Framework** | Framework-agnostic | Custom training loop |
| **Logging** | WandB / JSONL / stdout | Custom format |
| **Spurious feature detection** | Built-in | Not supported |
| **Phase transition detection** | CUSUM on effective rank | Not supported |

trainprobe is not a replacement for TRACE's fine-grained linguistic analysis.  It is a general-purpose representation observability layer that works with the data and training loop you already have.

---

## Installation

```bash
# Core (torch + numpy only)
pip install trainprobe

# With WandB integration
pip install "trainprobe[wandb]"

# Development
pip install "trainprobe[dev]"
```

Requires Python ≥ 3.9 and PyTorch ≥ 1.13.

---

## Contributing

Issues and PRs welcome.  To run the test suite:

```bash
pytest tests/ -v
```

All 39 tests run on CPU in under 5 seconds.
