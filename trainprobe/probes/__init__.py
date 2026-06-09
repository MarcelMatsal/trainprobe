from typing import List, Optional

import torch.nn as nn


class Probe:
    """Base class for all trainprobe probes.

    Subclasses set ``needs_activations`` to True if they require a forward pass
    on the probe batch.  Free probes (needs_activations=False) run every step at
    near-zero overhead.
    """

    needs_activations: bool = False
    name: str = "probe"

    def run(self, activations, model: Optional[nn.Module], step: int, **ctx) -> dict:
        """Return a flat dict of {relative_key: scalar_value}.

        Relative keys are prefixed with ``trainprobe/{self.name}/`` by ProbeRunner.
        Layer-scoped metrics should use ``{layer_name}/{metric}`` as the key.
        Model-level metrics use a plain ``{metric}`` key.
        """
        raise NotImplementedError


def default_suite() -> List[Probe]:
    """The five default probes shipped in v0.1."""
    from .geometry import EffectiveRankProbe, CollapseProbe, DeadNeuronProbe
    from .weight_grad import GradNormProbe, UpdateRatioProbe

    return [
        EffectiveRankProbe(),
        CollapseProbe(),
        DeadNeuronProbe(),
        GradNormProbe(),
        UpdateRatioProbe(),
    ]
