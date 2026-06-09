import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _normalize_activation(output) -> Optional[torch.Tensor]:
    """Reduce activation to (B, D) regardless of source layer type."""
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        return None
    if output.dim() == 4:
        # (B, C, H, W) → spatial mean → (B, C)
        return output.mean(dim=(2, 3))
    if output.dim() == 3:
        # (B, S, D) → CLS token → (B, D)
        return output[:, 0, :]
    if output.dim() == 2:
        return output
    if output.dim() == 1:
        return output.unsqueeze(0)
    return None


class ActivationCache:
    def __init__(self):
        self._data: Dict[str, torch.Tensor] = {}

    def store(self, name: str, tensor: torch.Tensor):
        normalized = _normalize_activation(tensor)
        if normalized is not None:
            self._data[name] = normalized.detach().cpu()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def items(self):
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)


class HookManager:
    def __init__(self):
        self._handles: List = []
        self._cache: Optional[ActivationCache] = None

    def _make_hook(self, safe_name: str, cache: ActivationCache):
        def hook(module, input, output):
            cache.store(safe_name, output)
        return hook

    def arm(self, model: nn.Module, layer_names: List[str]) -> ActivationCache:
        """Register hooks on the given layers (dot-notation names)."""
        self._cache = ActivationCache()
        self._handles = []
        name_set = set(layer_names)
        for name, module in model.named_modules():
            if name in name_set:
                safe_name = name.replace(".", "/")
                handle = module.register_forward_hook(self._make_hook(safe_name, self._cache))
                self._handles.append(handle)
        return self._cache

    def disarm(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def run_forward(
        self,
        model: nn.Module,
        layer_names: List[str],
        probe_batch,
        device: torch.device,
    ) -> ActivationCache:
        """Arm hooks, run a no-grad forward pass on probe_batch, disarm, return cache."""
        cache = self.arm(model, layer_names)
        try:
            with torch.no_grad():
                _forward_model(model, probe_batch, device)
        except Exception as e:
            warnings.warn(f"trainprobe: probe forward pass failed: {e}")
        finally:
            self.disarm()
        return cache


def _move_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
        return type(batch)(moved)
    return batch


def _move_to_cpu(batch):
    return _move_to_device(batch, torch.device("cpu"))


def _forward_model(model: nn.Module, batch, device: torch.device):
    batch = _move_to_device(batch, device)
    if isinstance(batch, dict):
        model(**batch)
    elif isinstance(batch, (list, tuple)):
        model(*batch)
    else:
        model(batch)
