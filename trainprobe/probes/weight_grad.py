"""
Free probes (needs_activations=False) that inspect model.parameters() directly.
These run every step at near-zero overhead because they require no forward pass.

Limitation (v0.1): UpdateRatioProbe._prev_params is not serialized, so ratios
are lost on crash/resume.  The first step after attach() always returns no ratios.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from . import Probe


class GradNormProbe(Probe):
    """Per-parameter and total L2 gradient norm.

    Useful for detecting gradient vanishing/explosion and identifying which parts
    of the network are receiving signal vs. getting no gradient flow.
    """

    needs_activations = False
    name = "grad_norm"

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        metrics: Dict[str, float] = {}
        total_sq = 0.0
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.detach().norm(2).item()
                safe_name = param_name.replace(".", "/")
                metrics[f"{safe_name}/grad_norm"] = norm
                total_sq += norm * norm
        metrics["total_grad_norm"] = total_sq ** 0.5
        return metrics


class UpdateRatioProbe(Probe):
    """||Δw|| / ||w|| per parameter across consecutive optimizer steps.

    The update ratio is the standard "lr-free" signal for diagnosing learning
    dynamics.  Values in [1e-3, 1e-2] are typical for healthy training; ratios
    below 1e-4 suggest the parameter is barely moving, and above 1e-1 suggests
    potentially too-large updates.
    """

    needs_activations = False
    name = "update_ratio"

    def __init__(self):
        self._prev: Dict[str, torch.Tensor] = {}

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        metrics: Dict[str, float] = {}
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            current = param.detach().cpu()
            safe_name = param_name.replace(".", "/")

            if param_name in self._prev:
                delta_norm = (current - self._prev[param_name]).norm(2).item()
                weight_norm = current.norm(2).item()
                if weight_norm > 1e-10:
                    metrics[f"{safe_name}/update_ratio"] = delta_norm / weight_norm

            self._prev[param_name] = current.clone()
        return metrics
