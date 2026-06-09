# AGENTS.md — trainprobe codebase guide

This file describes the trainprobe codebase for AI coding agents.  Read it before making any changes.

## What this project does

trainprobe is a lightweight PyTorch library that attaches to any training loop and emits representation-geometry signals (effective rank, feature collapse, dead neurons, gradient norms) to WandB or JSONL.  The central value proposition: your loss curve doesn't tell you whether your model is learning good representations or collapsing to a shortcut.  trainprobe does.

---

## Module map

```
trainprobe/
├── __init__.py          Public API: attach(), every(), on_loss_spike(), layers namespace
├── scope.py             TrainProbeScope — main lifecycle object (attach returns one of these)
├── schedule.py          When to run activation probes: EveryNSteps, OnLossSpike, Exponential, Composite
├── hooks.py             HookManager (arm/disarm per pass), ActivationCache, tensor normalization
├── registry.py          LayerRegistry, LayerFilter, auto_filter — which layers to probe
├── aggregator.py        SignalAggregator — CUSUM-based phase transition detection on effective_rank
├── runner.py            ProbeRunner — dispatches probes with per-probe error isolation
├── probes/
│   ├── __init__.py      Probe base class + default_suite()
│   ├── geometry.py      EffectiveRankProbe, CollapseProbe, DeadNeuronProbe, _effective_rank()
│   ├── weight_grad.py   GradNormProbe, UpdateRatioProbe (free probes, run every step)
│   └── task.py          LinearProbeProbe, SpuriousFeatureProbe (v0.2, not in default suite)
├── backends/
│   └── __init__.py      WandbBackend, JsonlBackend, StdoutBackend, CompositeBackend, resolve_backend()
└── integrations/
    └── __init__.py      TrainProbeCallback (HF), LightningTrainProbeCallback, StablePretrainingPlugin
```

---

## Key design constraints — do not violate

1. **No sklearn**.  All linear algebra uses `torch.linalg`.  `torch.linalg.lstsq(driver="gelsd")` is the solver for linear probes.

2. **WandB is optional**.  Core install is `torch + numpy` only.  `WandbBackend` does a lazy `import wandb` at log time and warns once if missing.  Never add wandb to base dependencies.

3. **Hooks must never be permanently registered**.  `HookManager.arm()` registers hooks, `HookManager.disarm()` removes them.  The `run_forward()` method wraps these in a `try/finally` so hooks are always removed even if the forward pass raises.  Permanent hooks break multi-GPU training and checkpoint loading.

4. **Metric names are load-bearing**.  The schema `trainprobe/{probe_name}/{layer_name}/{metric}` must not change between patch versions — doing so silently breaks existing WandB runs.  Layer name dots → slashes (WandB renders slashes as nested groups).

5. **Free probes run every step**.  Probes with `needs_activations = False` must not trigger a forward pass.  They inspect `model.parameters()` and `.grad` only.

6. **Probe errors must not crash training**.  `ProbeRunner` wraps each probe call in try/except and warns once per probe class.

---

## Data flow for a single scope.step() call

```
scope.step(step, loss=loss.item(), **ctx)
  │
  ├─ ProbeRunner.run_free(free_probes, model, step, **ctx)
  │    └─ for each GradNormProbe, UpdateRatioProbe:
  │         probe.run(None, model, step) → dict
  │         prefix with trainprobe/{probe.name}/
  │
  ├─ schedule.should_run(step, loss) OR _pending_immediate?
  │    └─ yes:
  │         HookManager.run_forward(model, layer_names, probe_batch, device)
  │           └─ arm → torch.no_grad() forward → disarm → ActivationCache
  │         ProbeRunner.run_activation(activation_probes, cache, step, **ctx)
  │           └─ for each EffectiveRankProbe, CollapseProbe, DeadNeuronProbe:
  │                probe.run(cache, None, step) → dict
  │         SignalAggregator.update(act_metrics, step)  ← CUSUM check
  │           └─ if phase_transition_detected():
  │                _pending_immediate = True
  │                log trainprobe/events/phase_transition = 1
  │                call on_phase_transition callback if set
  │
  └─ Backend.log(all_metrics, step)
```

---

## Activation tensor normalization (hooks.py)

Forward hook outputs are reduced to `(B, D)` before storage:

| Input shape | Reduction |
|---|---|
| `(B, C, H, W)` | spatial mean → `(B, C)` |
| `(B, S, D)` | CLS token `[:, 0, :]` → `(B, D)` |
| `(B, D)` | pass through |
| `tuple` | take first element, then apply above |

---

## Probe batch

The probe batch is stored on CPU (`_move_to_cpu` in hooks.py) and moved to the model device at probe time via `_move_to_device`.  It can be a plain tensor, a dict (for HF-style `model(**batch)` calls), or a list/tuple (for positional `model(*batch)` calls).

If `probe_batch=None` is passed to `tp.attach()`, activation probes will warn and skip until `scope.capture_batch(batch)` is called.  Framework integrations (Lightning) wire this automatically; HuggingFace requires explicit `probe_batch=`.

---

## CUSUM phase transition detection (aggregator.py)

Tracked signals: any key in `act_metrics` matching `trainprobe/effective_rank/*/rank`.

- 10-step warmup establishes baseline mean/std.
- `S_t = max(0, S_{t-1} + (x_t - μ) / σ - k)` with `k=0.5`, fires when `S_t > h=5.0`.
- Cooldown: minimum 5 probe steps between triggers.
- After firing: CUSUM accumulator resets, baseline recomputed from recent history.
- Detection sets `_pending_immediate = True` on the scope → next `step()` runs activation probes regardless of schedule.

---

## auto_filter (registry.py)

Selects layers based on depth:
- ≤24 parameterized non-norm layers → all of them (stride 1)
- 25–96 → every 2nd (stride 2)
- >96 → every 4th (stride 4)

Excluded: `LayerNorm`, `BatchNorm*`, `GroupNorm`, `InstanceNorm*`.  These are excluded because their activation geometry reflects learnable rescaling rather than representational structure.

---

## Adding a new probe

1. Subclass `Probe` in an appropriate file under `trainprobe/probes/`.
2. Set `needs_activations: bool` and `name: str` as class attributes.
3. Implement `run(self, activations, model, step, **ctx) -> dict`.
   - Return `{relative_key: scalar_value}` — the runner prepends `trainprobe/{self.name}/`.
   - Layer-scoped metrics use `{layer_name}/{metric}` as the key.
4. If the probe needs labels/aux tensors, read them from `**ctx` and return `{}` gracefully when absent.
5. Add a smoke test in `tests/test_probes.py`.

Do **not** add probes to `default_suite()` unless they work without any `**ctx` arguments (i.e., are safe for label-free SSL loops).

---

## Adding a new backend

Subclass `Backend` in `trainprobe/backends/__init__.py` and handle the `log(metrics, step)` method.  Add a branch to `resolve_backend()`.  WandB is the only optional dependency — any new backend should work with core install.

---

## Known limitations (v0.1)

- `UpdateRatioProbe._prev_params` is not serialized.  After a crash/resume, update ratios are incorrect for the first step.  Will be addressed in v0.2.
- `TrainProbeCallback` (HuggingFace) cannot auto-capture the probe batch because the HF Trainer callback API does not expose the training batch.  Pass `probe_batch=` explicitly.
- `GradNormProbe` and `UpdateRatioProbe` log one metric per named parameter.  For very large models (100B+), this may generate excessive WandB metrics.  Add per-layer aggregation in v0.2 if needed.
- Probe batch recommended size: 128–256.  Too small (< 16) gives noisy rank estimates; too large slows the probe forward pass.

---

## Running the test suite

```bash
pytest tests/ -v          # 39 tests, all CPU, ~2 seconds
```

All tests pass on PyTorch ≥ 1.13, Python 3.9–3.12, no GPU required.
