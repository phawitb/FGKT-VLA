"""Counterfactual recovery utility computed from paired rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from fcut_vla.evaluation.statistics import paired_bootstrap_ci


class CounterfactualInputError(ValueError):
    """Raised when utility inputs violate paired-evaluation requirements."""


@dataclass(frozen=True)
class CostNormalization:
    bytes_scale: float
    latency_scale: float

    def __post_init__(self) -> None:
        if self.bytes_scale <= 0 or self.latency_scale <= 0:
            raise CounterfactualInputError("cost normalization scales must be positive")


@dataclass(frozen=True)
class UtilityObservation:
    failure_id: str
    adapter_id: str
    recovery_gain: float
    retention_risk: float
    normalized_cost: float
    utility: float
    utility_lower: float
    utility_upper: float

    @property
    def has_positive_utility(self) -> bool:
        return self.utility_lower > 0

    def is_confidently_above(self, other: "UtilityObservation") -> bool:
        return self.utility_lower > other.utility_upper


def _paired_means(before: Sequence[float], after: Sequence[float], label: str) -> tuple[float, float]:
    if not before or len(before) != len(after):
        raise CounterfactualInputError(f"{label} requires non-empty paired inputs")
    values = tuple(float(item) for item in before) + tuple(float(item) for item in after)
    if not all(math.isfinite(item) for item in values):
        raise CounterfactualInputError(f"{label} values must be finite")
    return sum(before) / len(before), sum(after) / len(after)


def compute_utility(
    *,
    failure_id: str,
    adapter_id: str,
    recovery_before: Sequence[float],
    recovery_after: Sequence[float],
    retention_before: Mapping[str, Sequence[float]],
    retention_after: Mapping[str, Sequence[float]],
    communication_bytes: int,
    latency_seconds: float,
    cost_normalization: CostNormalization,
    lambda_retention: float = 1.0,
    lambda_cost: float = 0.0,
) -> UtilityObservation:
    before_mean, after_mean = _paired_means(
        recovery_before, recovery_after, "paired recovery"
    )
    recovery_gain = after_mean - before_mean

    if set(retention_before) != set(retention_after):
        raise CounterfactualInputError("retention task sets must match")
    task_risks: list[float] = []
    for task in sorted(retention_before):
        old_mean, new_mean = _paired_means(
            retention_before[task], retention_after[task], "paired retention"
        )
        task_risks.append(max(0.0, old_mean - new_mean))
    retention_risk = sum(task_risks) / len(task_risks) if task_risks else 0.0

    if communication_bytes < 0 or latency_seconds < 0:
        raise CounterfactualInputError("communication and latency costs must be non-negative")
    normalized_cost = 0.5 * (
        communication_bytes / cost_normalization.bytes_scale
        + latency_seconds / cost_normalization.latency_scale
    )
    utility = (
        recovery_gain
        - lambda_retention * retention_risk
        - lambda_cost * normalized_cost
    )

    interval = paired_bootstrap_ci(
        recovery_before,
        recovery_after,
        confidence=0.95,
        samples=1000,
        seed=0,
    )
    fixed_penalty = lambda_retention * retention_risk + lambda_cost * normalized_cost
    return UtilityObservation(
        failure_id=failure_id,
        adapter_id=adapter_id,
        recovery_gain=recovery_gain,
        retention_risk=retention_risk,
        normalized_cost=normalized_cost,
        utility=utility,
        utility_lower=interval.lower - fixed_penalty,
        utility_upper=interval.upper - fixed_penalty,
    )

