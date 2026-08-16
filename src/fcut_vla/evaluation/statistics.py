"""Small deterministic statistical utilities used before heavyweight analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from .metrics import MetricInputError


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap_ci(
    before: Sequence[float],
    after: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    if not before or len(before) != len(after):
        raise MetricInputError("paired bootstrap requires equal non-empty inputs")
    if not 0 < confidence < 1 or samples < 1:
        raise MetricInputError("confidence and bootstrap sample count are invalid")
    differences = [float(new) - float(old) for old, new in zip(before, after, strict=True)]
    if not all(math.isfinite(value) for value in differences):
        raise MetricInputError("paired bootstrap values must be finite")
    estimate = sum(differences) / len(differences)
    generator = random.Random(seed)
    bootstrapped = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    alpha = (1 - confidence) / 2
    return ConfidenceInterval(
        estimate=estimate,
        lower=_quantile(bootstrapped, alpha),
        upper=_quantile(bootstrapped, 1 - alpha),
        confidence=confidence,
    )
