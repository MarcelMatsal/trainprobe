"""
Logging backends.  WandB is an optional dependency (pip install trainprobe[wandb]).
All other backends (jsonl, stdout) work with core torch+numpy install.
"""
import json
import sys
from typing import List, Union


def _to_python(v):
    """Convert a value to a JSON-serialisable Python scalar."""
    if isinstance(v, (int, float, bool)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Backend:
    def log(self, metrics: dict, step: int) -> None:
        raise NotImplementedError


class WandbBackend(Backend):
    """Logs to the active wandb run under the trainprobe/* namespace.

    Safe to call when no run is active — silently skips in that case.
    """

    def log(self, metrics: dict, step: int) -> None:
        try:
            import wandb
        except ImportError:
            import warnings
            warnings.warn(
                "trainprobe: wandb not installed; install with "
                "`pip install trainprobe[wandb]` or choose logger='jsonl'.",
                stacklevel=4,
            )
            return
        if wandb.run is not None and metrics:
            wandb.log(metrics, step=step)


class JsonlBackend(Backend):
    """Appends one JSON object per probe step to a .jsonl file."""

    def __init__(self, path: str = "trainprobe_log.jsonl"):
        self._path = path

    def log(self, metrics: dict, step: int) -> None:
        if not metrics:
            return
        record = {"step": step}
        for k, v in metrics.items():
            converted = _to_python(v)
            if converted is not None:
                record[k] = converted
        with open(self._path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


class StdoutBackend(Backend):
    """Prints metrics to stdout — useful for debugging without WandB."""

    def log(self, metrics: dict, step: int) -> None:
        if not metrics:
            return
        parts = " ".join(
            f"{k}={v:.4f}" for k, v in sorted(metrics.items())
            if isinstance(v, (int, float))
        )
        print(f"[trainprobe] step={step} {parts}", file=sys.stderr)


class CompositeBackend(Backend):
    def __init__(self, backends: List[Backend]):
        self._backends = backends

    def log(self, metrics: dict, step: int) -> None:
        for b in self._backends:
            b.log(metrics, step)


def resolve_backend(logger: Union[str, Backend]) -> Backend:
    if isinstance(logger, Backend):
        return logger
    if logger == "wandb":
        return WandbBackend()
    if logger == "jsonl":
        return JsonlBackend("trainprobe_log.jsonl")
    if logger == "stdout":
        return StdoutBackend()
    if isinstance(logger, str) and logger.endswith(".jsonl"):
        return JsonlBackend(logger)
    raise ValueError(
        f"trainprobe: unknown logger {logger!r}. "
        "Valid options: 'wandb', 'jsonl', 'stdout', a .jsonl path, or a Backend instance."
    )
