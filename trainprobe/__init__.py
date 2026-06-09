"""
trainprobe — training dynamics observability for PyTorch models.

WandB logs your loss.  trainprobe tells you what your model is learning.

Quick start::

    import trainprobe as tp

    scope = tp.attach(model, probe_batch=val_batch)
    for step, batch in enumerate(loader):
        loss = compute(batch); loss.backward(); opt.step()
        scope.step(step, loss=loss.item())
"""
from typing import List, Optional, Union

from .scope import TrainProbeScope
from .schedule import Schedule, EveryNSteps, OnLossSpike, Exponential, Composite
from .registry import LayerFilter, LayerRegistry, auto_filter
from .probes import Probe, default_suite
from .backends import Backend


def attach(
    model,
    probe_batch=None,
    probes: Optional[List[Probe]] = None,
    schedule: Optional[Schedule] = None,
    logger: Union[str, Backend] = "wandb",
    layers: Optional[LayerRegistry] = None,
    **kwargs,
) -> TrainProbeScope:
    """Attach trainprobe to a model and return a TrainProbeScope.

    Parameters
    ----------
    model :
        Any ``nn.Module``.
    probe_batch :
        Fixed batch used for activation probes.  Stored on CPU, moved to model
        device at probe time.  If omitted, call ``scope.capture_batch(batch)``
        before the first scheduled probe step.
    probes :
        List of Probe instances.  Defaults to ``default_suite()`` (5 probes).
    schedule :
        When to run activation probes.  Defaults to ``every(250)``.
    logger :
        ``'wandb'`` (default), ``'jsonl'``, ``'stdout'``, a ``.jsonl`` path,
        or a Backend instance.  WandB must be installed separately
        (``pip install trainprobe[wandb]``).
    layers :
        Custom LayerRegistry from ``tp.layers.*``.  Defaults to auto_filter.
    """
    if probes is None:
        probes = default_suite()
    if schedule is None:
        schedule = EveryNSteps(250)
    return TrainProbeScope(model, probe_batch, probes, schedule, logger, layers)


def every(n: int) -> EveryNSteps:
    """Fire activation probes every n steps."""
    return EveryNSteps(n)


def on_loss_spike(threshold: float = 2.0, cooldown: int = 100) -> OnLossSpike:
    """Fire when loss / recent_baseline > threshold, with cooldown steps between triggers."""
    return OnLossSpike(threshold=threshold, cooldown=cooldown)


def exponential(base: float = 2.0, start: int = 100) -> Exponential:
    """Fire at start, start*base, start*base^2, ..."""
    return Exponential(base=base, start=start)


def composite(schedules: List[Schedule]) -> Composite:
    """Fire when any sub-schedule fires."""
    return Composite(schedules)


class _LayersNamespace:
    """Namespace for layer-selection strategies.  Access via ``tp.layers.*``."""

    @staticmethod
    def by_name(*names: str) -> LayerRegistry:
        """Select layers by exact dot-notation module name."""
        return LayerRegistry(list(names))

    @staticmethod
    def linear(model) -> LayerRegistry:
        """Select all nn.Linear layers."""
        return LayerFilter.linear(model)

    @staticmethod
    def conv(model) -> LayerRegistry:
        """Select all Conv layers."""
        return LayerFilter.conv(model)

    @staticmethod
    def auto(model) -> LayerRegistry:
        """Automatic selection based on model depth (default)."""
        return auto_filter(model)

    @staticmethod
    def by_type(model, *types) -> LayerRegistry:
        """Select layers matching any of the given nn.Module types."""
        return LayerFilter.by_type_filter(model, *types)


layers = _LayersNamespace()

__all__ = [
    "attach",
    "every",
    "on_loss_spike",
    "exponential",
    "composite",
    "layers",
    "Probe",
    "default_suite",
    "TrainProbeScope",
    "Backend",
]

__version__ = "0.1.0"
