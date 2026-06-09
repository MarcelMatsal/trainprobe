from typing import List, Optional, Tuple, Type

import torch.nn as nn

_NORM_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


class LayerRegistry:
    """Holds the ordered list of layer names (dot-notation) to probe."""

    def __init__(self, names: List[str]):
        self._names = list(names)

    @property
    def names(self) -> List[str]:
        return self._names

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"LayerRegistry({self._names})"


class LayerFilter:
    """Namespace for composable layer-selection strategies."""

    @staticmethod
    def by_name(*names: str) -> LayerRegistry:
        return LayerRegistry(list(names))

    @staticmethod
    def by_type_filter(model: nn.Module, *types: Type[nn.Module]) -> LayerRegistry:
        selected = [
            name for name, mod in model.named_modules() if isinstance(mod, types)
        ]
        return LayerRegistry(selected)

    @staticmethod
    def linear(model: nn.Module) -> LayerRegistry:
        return LayerFilter.by_type_filter(model, nn.Linear)

    @staticmethod
    def conv(model: nn.Module) -> LayerRegistry:
        return LayerFilter.by_type_filter(model, nn.Conv1d, nn.Conv2d, nn.Conv3d)

    @staticmethod
    def auto(model: nn.Module) -> LayerRegistry:
        return auto_filter(model)


def auto_filter(model: nn.Module) -> LayerRegistry:
    """
    Select layers based on model depth:
      ≤24 modules  → all parameterized non-norm layers
      25–96         → every 2nd
      >96           → every 4th

    Norm layers are excluded because their activation geometry is less interpretable
    and they undergo learnable re-scaling that conflates with structural changes.
    """
    candidates: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue  # skip root module
        if isinstance(module, _NORM_TYPES):
            continue
        # Must have own parameters (not just inherited from children)
        own_params = list(module.parameters(recurse=False))
        if own_params:
            candidates.append(name)

    n = len(candidates)
    if n <= 24:
        stride = 1
    elif n <= 96:
        stride = 2
    else:
        stride = 4

    return LayerRegistry(candidates[::stride])
