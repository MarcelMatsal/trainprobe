from typing import Optional

import torch
import torch.nn as nn

from . import Probe


def _effective_rank(tensor: torch.Tensor) -> float:
    """exp(H) of the normalized singular value distribution.

    Range is [1, min(B, D)].  A rank-1 matrix returns 1.0; a matrix with all
    equal singular values returns min(B, D).

    Uses relative thresholding (sv < sv_max * 1e-6) to filter numerical noise
    without discarding genuine small-but-nonzero singular values.
    """
    if tensor.shape[0] < 2:
        return 1.0
    _, s, _ = torch.linalg.svd(tensor.float(), full_matrices=False)
    if s.numel() == 0 or s[0].item() < 1e-12:
        return 1.0
    threshold = max(s[0].item() * 1e-6, 1e-10)
    s = s[s > threshold]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    h = -(p * torch.log(p + 1e-30)).sum()
    return h.exp().item()


class EffectiveRankProbe(Probe):
    """exp(Shannon entropy of singular value distribution).

    Tracks the intrinsic dimensionality of each layer's representation.  A
    healthy SSL encoder should show rising effective rank over early training as
    features spread across more dimensions, followed by a plateau.  Collapse
    appears as a sharp drop to near 1.
    """

    needs_activations = True
    name = "effective_rank"

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        metrics = {}
        for layer_name, acts in activations.items():
            metrics[f"{layer_name}/rank"] = _effective_rank(acts)
        return metrics


class CollapseProbe(Probe):
    """Two complementary collapse diagnostics per layer.

    ``norm_cv``: coefficient of variation (std/mean) of per-sample L2 norms.
      Near 0 → representations cluster on a hypersphere (hyperspherical collapse).
      High   → norms vary across samples (healthy diversity).

    ``sv_ratio``: min/max singular value of the activation matrix.
      Near 0 → one direction dominates (dimensional collapse).
      Near 1 → dimensions are roughly equally used.
    """

    needs_activations = True
    name = "collapse"

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        metrics = {}
        for layer_name, acts in activations.items():
            acts_f = acts.float()
            norms = acts_f.norm(dim=-1)
            mean_norm = norms.mean().item()
            norm_cv = (norms.std() / mean_norm).item() if mean_norm > 1e-10 else 0.0

            _, s_all, _ = torch.linalg.svd(acts_f, full_matrices=False)
            # Use unfiltered SVD so a rank-1 matrix gives sv_ratio ≈ 0, not 1.0.
            # s_all is sorted descending; s_all[-1] is the smallest singular value.
            sv_ratio = (s_all[-1] / s_all[0]).item() if s_all[0].item() > 1e-12 else 0.0

            metrics[f"{layer_name}/norm_cv"] = norm_cv
            metrics[f"{layer_name}/sv_ratio"] = sv_ratio
        return metrics


class DeadNeuronProbe(Probe):
    """Fraction of neurons whose max absolute activation is zero across the probe batch.

    A neuron that never activates contributes nothing to representation learning.
    Rising dead-neuron fraction can precede collapse or indicate poor initialization.
    """

    needs_activations = True
    name = "dead_neurons"

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        metrics = {}
        for layer_name, acts in activations.items():
            max_abs = acts.abs().max(dim=0).values
            dead_fraction = (max_abs == 0).float().mean().item()
            metrics[f"{layer_name}/dead_fraction"] = dead_fraction
        return metrics
