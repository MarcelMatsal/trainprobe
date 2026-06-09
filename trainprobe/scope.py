"""
TrainProbeScope — the central lifecycle object returned by tp.attach().

Each call to scope.step() does two things:
  1. Run free probes (every step, ~zero overhead).
  2. Conditionally run activation probes (on schedule or when phase transition
     was detected on the previous probe pass).

Activation probes require a probe_batch.  If one was not passed to tp.attach(),
call scope.capture_batch(batch) before the first scheduled probe step.  After
capture, the batch is stored on CPU and moved to the model device at probe time.

Phase-transition detection: after each activation probe pass, CUSUM is run on
effective_rank signals.  A detected transition sets _pending_immediate = True so
the next scope.step() runs another full activation pass (regardless of schedule)
and fires the on_phase_transition callback.
"""
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from .aggregator import SignalAggregator
from .backends import Backend, resolve_backend
from .hooks import HookManager, _move_to_cpu
from .registry import LayerRegistry, auto_filter
from .runner import ProbeRunner
from .schedule import Schedule, EveryNSteps
from .probes import Probe, default_suite


class TrainProbeScope:
    def __init__(
        self,
        model: nn.Module,
        probe_batch,
        probes: List[Probe],
        schedule: Schedule,
        logger: Union[str, Backend],
        layer_registry: Optional[LayerRegistry] = None,
    ):
        self._model = model
        self._probe_batch = None if probe_batch is None else _move_to_cpu(probe_batch)
        self._probes = probes
        self._schedule = schedule
        self._backend = resolve_backend(logger)
        self._registry = layer_registry if layer_registry is not None else auto_filter(model)

        self._free_probes = [p for p in probes if not p.needs_activations]
        self._activation_probes = [p for p in probes if p.needs_activations]

        self._hook_manager = HookManager()
        self._runner = ProbeRunner()
        self._aggregator = SignalAggregator()

        self._pending_immediate: bool = False
        self._on_phase_transition: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_batch(self, batch) -> None:
        """Store the first batch as the probe batch (no-op after first call)."""
        if self._probe_batch is None:
            self._probe_batch = _move_to_cpu(batch)

    def step(self, step: int, loss: Optional[float] = None, **ctx) -> None:
        """Called once per training step.

        Parameters
        ----------
        step : int
            Global training step index (used for WandB x-axis and scheduling).
        loss : float, optional
            Current training loss.  Required for OnLossSpike scheduling and
            logged as ``trainprobe/loss`` for convenience.
        **ctx :
            Passed through to probes.  LinearProbeProbe reads ``labels``;
            SpuriousFeatureProbe reads ``genuine_labels`` and ``spurious_labels``.
        """
        metrics: dict = {}

        if loss is not None:
            metrics["trainprobe/loss"] = loss

        # --- Free probes: every step ---
        free_metrics = self._runner.run_free(self._free_probes, self._model, step, **ctx)
        metrics.update(free_metrics)

        # --- Activation probes: on schedule or pending ---
        should_run = self._pending_immediate or self._schedule.should_run(step, loss=loss)

        if should_run and self._activation_probes:
            if self._probe_batch is None:
                warnings.warn(
                    "trainprobe: activation probe scheduled at step "
                    f"{step} but no probe_batch is available. "
                    "Pass probe_batch= to tp.attach() or call scope.capture_batch(batch) "
                    "before the first probe step.",
                    stacklevel=2,
                )
            else:
                self._pending_immediate = False
                device = _model_device(self._model)
                cache = self._hook_manager.run_forward(
                    self._model,
                    self._registry.names,
                    self._probe_batch,
                    device,
                )
                act_metrics = self._runner.run_activation(
                    self._activation_probes, cache, step, **ctx
                )
                metrics.update(act_metrics)

                self._aggregator.update(act_metrics, step)
                if self._aggregator.phase_transition_detected():
                    self._pending_immediate = True
                    metrics["trainprobe/events/phase_transition"] = 1
                    if self._on_phase_transition is not None:
                        try:
                            self._on_phase_transition(step, metrics)
                        except Exception as exc:
                            warnings.warn(
                                f"trainprobe: on_phase_transition callback raised: {exc}"
                            )

        self._backend.log(metrics, step)

    @property
    def on_phase_transition(self) -> Optional[Callable]:
        return self._on_phase_transition

    @on_phase_transition.setter
    def on_phase_transition(self, fn: Optional[Callable]) -> None:
        self._on_phase_transition = fn

    # ------------------------------------------------------------------
    # Context manager support (optional convenience)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
