from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PersonalizationConfig


class RunningStats:
    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean if self._count else 0.0

    @property
    def variance(self) -> float:
        return (self._m2 / (self._count - 1)) if self._count > 1 else 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def count(self) -> int:
        return self._count

    def summary(self) -> dict[str, float]:
        return {"count": float(self._count), "mean": self.mean, "std": self.std}


@dataclass
class DirectiveContext:
    normalized_load: float
    load_level: str
    confidence: float
    trend: str
    suppress_adaptation: bool
    timestamp: float


class PersonalizationManager:
    def __init__(self, config: PersonalizationConfig, step_size_seconds: float) -> None:
        self._config = config
        self._step_size = step_size_seconds
        self._baseline_stats = RunningStats()
        self._last_directive: Optional[DirectiveContext] = None
        self._last_raw_load: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    @property
    def baseline_ready(self) -> bool:
        required_samples = max(1, int(self._config.baseline_duration_seconds / self._step_size))
        return self._baseline_stats.count >= required_samples

    def ingest(self, raw_load: float, confidence: float, timestamp: Optional[float] = None) -> DirectiveContext:
        timestamp = timestamp or time.time()
        if not self.baseline_ready:
            self._baseline_stats.update(raw_load)
            norm_load = 0.5
        else:
            baseline_mean = self._baseline_stats.mean
            baseline_std = max(self._baseline_stats.std, 1e-3)
            adaptive_baseline = (1 - self._config.adaptation_rate) * baseline_mean + self._config.adaptation_rate * raw_load
            self._baseline_stats.update(adaptive_baseline)
            z = (raw_load - baseline_mean) / baseline_std
            norm_load = float(1 / (1 + np.exp(-z)))
        load_level = self._categorize(norm_load)
        trend = self._infer_trend(raw_load)
        suppress = self._should_suppress(timestamp, confidence)
        directive = DirectiveContext(
            normalized_load=norm_load,
            load_level=load_level,
            confidence=confidence,
            trend=trend,
            suppress_adaptation=suppress,
            timestamp=timestamp,
        )
        self._last_directive = directive
        self._last_raw_load = raw_load
        self._last_timestamp = timestamp
        return directive

    def _categorize(self, normalized_load: float) -> str:
        if normalized_load >= self._config.high_load_threshold:
            return "high"
        if normalized_load <= self._config.low_load_threshold:
            return "low"
        return "medium"

    def _infer_trend(self, raw_load: float) -> str:
        if self._last_raw_load is None:
            return "stable"
        delta = raw_load - self._last_raw_load
        if abs(delta) < 0.02:
            return "stable"
        return "rising" if delta > 0 else "falling"

    def _should_suppress(self, timestamp: float, confidence: float) -> bool:
        if confidence < 0.3:
            return True
        if self._last_directive and (timestamp - self._last_directive.timestamp) < self._config.min_directive_interval_seconds:
            return True
        return False

    def baseline_summary(self) -> dict[str, float]:
        return self._baseline_stats.summary()

    def reset(self) -> None:
        self.__init__(self._config, self._step_size)
