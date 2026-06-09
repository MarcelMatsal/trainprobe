"""
CUSUM-based phase transition detector.

Applied only to effective_rank signals (keys matching trainprobe/effective_rank/*/rank).
Warmup of 10 probe steps establishes the baseline mean/std. After warmup, the
one-sided CUSUM statistic S_t = max(0, S_{t-1} + z_t - k) fires when S_t > h.
Cooldown of 5 probe steps prevents repeated triggers during noisy transitions.
Baseline is updated from recent history after each firing.
"""

from typing import Dict


class SignalAggregator:
    def __init__(
        self,
        warmup: int = 10,
        k: float = 0.5,
        h: float = 5.0,
        cooldown: int = 5,
    ):
        self._warmup = warmup
        self._k = k
        self._h = h
        self._cooldown = cooldown

        self._probe_count = 0
        self._last_transition_probe = -(cooldown + 1)
        self._transition_detected = False

        # Per-signal state (keyed by full metric name)
        self._history: Dict[str, list] = {}
        self._cusum_s: Dict[str, float] = {}
        self._baseline: Dict[str, tuple] = {}  # (mean, std)

    def update(self, metrics: dict, step: int) -> None:
        self._probe_count += 1
        self._transition_detected = False

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if "effective_rank" in key and key.endswith("/rank"):
                self._update_cusum(key, float(value))

    def _update_cusum(self, key: str, value: float) -> None:
        if key not in self._history:
            self._history[key] = []
            self._cusum_s[key] = 0.0

        hist = self._history[key]
        hist.append(value)

        if len(hist) < self._warmup:
            return

        if key not in self._baseline:
            mu = sum(hist) / len(hist)
            variance = sum((x - mu) ** 2 for x in hist) / len(hist)
            sigma = max(variance ** 0.5, 1e-6)
            self._baseline[key] = (mu, sigma)

        mu, sigma = self._baseline[key]
        z = (value - mu) / sigma
        self._cusum_s[key] = max(0.0, self._cusum_s[key] + z - self._k)

        if self._cusum_s[key] > self._h:
            steps_since = self._probe_count - self._last_transition_probe
            if steps_since > self._cooldown:
                self._transition_detected = True
                self._last_transition_probe = self._probe_count
                self._cusum_s[key] = 0.0
                # Rebaseline from recent history
                recent = hist[-self._warmup:]
                mu2 = sum(recent) / len(recent)
                var2 = sum((x - mu2) ** 2 for x in recent) / len(recent)
                self._baseline[key] = (mu2, max(var2 ** 0.5, 1e-6))

    def phase_transition_detected(self) -> bool:
        return self._transition_detected
