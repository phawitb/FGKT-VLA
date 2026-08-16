"""Retention-gated global promotion with rollback-safe state handling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from torch import Tensor


@dataclass(frozen=True)
class ConsolidationDecision:
    promote: bool
    abstained: bool
    gain_lower: float
    average_risk_upper: float
    worst_task_risk_upper: float
    reasons: tuple[str, ...]


def evaluate_gate(
    *,
    gain_lower: float,
    retention_risk_upper: Mapping[str, float],
    min_gain: float,
    max_average_risk: float,
    max_worst_task_risk: float,
    matched: bool = True,
) -> ConsolidationDecision:
    values = tuple(float(value) for value in retention_risk_upper.values())
    scalars = (gain_lower, min_gain, max_average_risk, max_worst_task_risk, *values)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("gate values must be finite")
    if any(value < 0 for value in values):
        raise ValueError("retention risk bounds must be non-negative")
    average = sum(values) / len(values) if values else 0.0
    worst = max(values, default=0.0)
    reasons: list[str] = []
    if not matched:
        reasons.append("no_match")
    if gain_lower < min_gain:
        reasons.append("gain")
    if average > max_average_risk:
        reasons.append("average_retention")
    if worst > max_worst_task_risk:
        reasons.append("worst_task")
    return ConsolidationDecision(
        promote=not reasons,
        abstained=not matched,
        gain_lower=float(gain_lower),
        average_risk_upper=average,
        worst_task_risk_upper=worst,
        reasons=tuple(reasons),
    )


def promote_with_rollback(
    current: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    decision: ConsolidationDecision,
) -> dict[str, Tensor]:
    selected = candidate if decision.promote else current
    return {name: tensor.clone() for name, tensor in selected.items()}
