"""CPU core demand prediction via Exponential Moving Average (EMA).

Maintains a smoothed predicted_cores estimate and detects stability
for scheduling decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---- Configurable parameters ----

# EMA alpha: weight given to the most recent observation
DEFAULT_ALPHA = 0.3

# Stability window: number of consecutive windows to check
STABILITY_WINDOW = 3

# Stability threshold: max-min difference relative to mean
STABILITY_THRESHOLD = 0.30

# Small epsilon to avoid division by zero
_EPSILON = 1e-12


@dataclass
class PredictionState:
    """Current state of the CPU core predictor."""

    predicted_cores: float = 0.0
    requested_cores: int = 1
    stable: bool = False
    observation_count: int = 0
    recent_predictions: list[float] = field(default_factory=list)


class Predictor:
    """EMA-based predictor for tool CPU core demand.

    Updates on each monitoring window and tracks stability via
    consecutive-window consistency check.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        stability_window: int = STABILITY_WINDOW,
        stability_threshold: float = STABILITY_THRESHOLD,
    ) -> None:
        self._alpha = alpha
        self._stability_window = stability_window
        self._stability_threshold = stability_threshold

        self._predicted_cores: float = 0.0
        self._observation_count: int = 0
        self._recent_predictions: list[float] = []

    @property
    def predicted_cores(self) -> float:
        return self._predicted_cores

    @property
    def requested_cores(self) -> int:
        """Predicted cores rounded up to the nearest integer (min 1)."""
        return max(1, math.ceil(self._predicted_cores))

    @property
    def stable(self) -> bool:
        """True if the last N predictions are stable."""
        if len(self._recent_predictions) < self._stability_window:
            return False
        recent = self._recent_predictions[-self._stability_window :]
        max_val = max(recent)
        min_val = min(recent)
        mean_val = sum(recent) / len(recent)
        # Avoid division by near-zero mean
        if mean_val < _EPSILON:
            return max_val - min_val < _EPSILON
        return (max_val - min_val) <= self._stability_threshold * max(mean_val, _EPSILON)

    @property
    def observation_count(self) -> int:
        return self._observation_count

    def update(self, observed_effective_cores: float) -> PredictionState:
        """Update the predictor with a new observation.

        Args:
            observed_effective_cores: The effective_cores observed in
                the most recent monitoring window.

        Returns:
            Current PredictionState after the update.
        """
        self._observation_count += 1

        if self._observation_count == 1:
            # First valid observation: bootstrap
            self._predicted_cores = observed_effective_cores
        else:
            # EMA update
            self._predicted_cores = (
                self._alpha * observed_effective_cores
                + (1 - self._alpha) * self._predicted_cores
            )

        # Track recent predictions for stability check
        self._recent_predictions.append(self._predicted_cores)
        # Keep only the last N+1 to bound memory
        if len(self._recent_predictions) > self._stability_window + 5:
            self._recent_predictions = self._recent_predictions[-(self._stability_window + 5):]

        return PredictionState(
            predicted_cores=self._predicted_cores,
            requested_cores=self.requested_cores,
            stable=self.stable,
            observation_count=self._observation_count,
            recent_predictions=list(self._recent_predictions[-self._stability_window:]),
        )

    def check_divergence(self, current_value: float) -> bool:
        """Check if recent observations have diverged from the stable baseline.

        Returns True if the last 3 predicted values differ significantly
        from the previous stable values, indicating a phase change.
        """
        if not self.stable:
            return False
        if len(self._recent_predictions) < self._stability_window + 3:
            return False
        # Compare the last 3 predictions to the 3 before them
        old = self._recent_predictions[-(self._stability_window + 3):-self._stability_window]
        new = self._recent_predictions[-self._stability_window:]
        old_mean = sum(old) / len(old)
        new_mean = sum(new) / len(new)
        if old_mean < _EPSILON and new_mean < _EPSILON:
            return False
        max_mean = max(abs(old_mean), abs(new_mean), _EPSILON)
        return abs(new_mean - old_mean) / max_mean > self._stability_threshold

    def reset_stability(self) -> None:
        """Reset stability tracking — used after phase change detection."""
        # Keep the current predicted_cores but clear recent history
        # so we need fresh consecutive stable windows
        self._recent_predictions = self._recent_predictions[-1:]
