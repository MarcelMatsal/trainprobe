import warnings
from collections import deque
from typing import List, Optional


class Schedule:
    def should_run(self, step: int, loss: Optional[float] = None) -> bool:
        raise NotImplementedError


class EveryNSteps(Schedule):
    def __init__(self, n: int = 250):
        self._n = n

    def should_run(self, step: int, loss: Optional[float] = None) -> bool:
        return step % self._n == 0


class OnLossSpike(Schedule):
    def __init__(self, threshold: float = 2.0, cooldown: int = 100, fallback_every: int = 250):
        self._threshold = threshold
        self._cooldown = cooldown
        self._fallback_every = fallback_every
        self._recent = deque(maxlen=20)
        self._last_fired = -(cooldown + 1)
        self._warned = False

    def should_run(self, step: int, loss: Optional[float] = None) -> bool:
        if loss is None:
            if not self._warned:
                warnings.warn(
                    "trainprobe: OnLossSpike schedule requires loss= in scope.step(); "
                    f"falling back to every {self._fallback_every} steps. "
                    "(This warning fires once.)",
                    stacklevel=4,
                )
                self._warned = True
            return step % self._fallback_every == 0

        self._recent.append(loss)
        if len(self._recent) < 5:
            return False

        baseline = sum(list(self._recent)[:-1]) / (len(self._recent) - 1)
        is_spike = loss / (baseline + 1e-10) > self._threshold

        if is_spike and (step - self._last_fired) > self._cooldown:
            self._last_fired = step
            return True

        return False


class Exponential(Schedule):
    """Fire at steps start, start*base, start*base^2, ..."""

    def __init__(self, base: float = 2.0, start: int = 100):
        self._base = base
        self._next = start

    def should_run(self, step: int, loss: Optional[float] = None) -> bool:
        if step >= self._next:
            self._next = max(step + 1, int(self._next * self._base))
            return True
        return False


class Composite(Schedule):
    def __init__(self, schedules: List[Schedule]):
        self._schedules = schedules

    def should_run(self, step: int, loss: Optional[float] = None) -> bool:
        return any(s.should_run(step, loss=loss) for s in self._schedules)
