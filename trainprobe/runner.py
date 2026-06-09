"""
ProbeRunner: dispatches probes with error isolation.

Each probe is called inside a try/except so a buggy probe cannot crash training.
Failures are warned once per probe class, not on every step, to avoid log spam.
"""
import warnings
from typing import Dict, List, Optional

import torch.nn as nn

from .probes import Probe
from .hooks import ActivationCache


class ProbeRunner:
    def __init__(self):
        self._failed: set = set()  # probe names that have already warned

    def run_free(
        self,
        probes: List[Probe],
        model: nn.Module,
        step: int,
        **ctx,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for probe in probes:
            try:
                result = probe.run(None, model, step, **ctx)
                for k, v in result.items():
                    metrics[f"trainprobe/{probe.name}/{k}"] = v
            except Exception as exc:
                self._warn(probe.name, exc)
        return metrics

    def run_activation(
        self,
        probes: List[Probe],
        cache: ActivationCache,
        step: int,
        **ctx,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for probe in probes:
            try:
                result = probe.run(cache, None, step, **ctx)
                for k, v in result.items():
                    metrics[f"trainprobe/{probe.name}/{k}"] = v
            except Exception as exc:
                self._warn(probe.name, exc)
        return metrics

    def _warn(self, name: str, exc: Exception) -> None:
        if name not in self._failed:
            warnings.warn(
                f"trainprobe: probe '{name}' raised an exception and will be skipped: {exc}",
                stacklevel=2,
            )
            self._failed.add(name)
