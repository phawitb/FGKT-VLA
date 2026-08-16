"""Retrieval and continual-learning metrics with strict input validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class MetricInputError(ValueError):
    """Raised when metric inputs cannot support the requested comparison."""


def ndcg_at_k(true_utility: Sequence[float], ranking: Sequence[int], k: int) -> float:
    if len(true_utility) == 0 or k < 1:
        raise MetricInputError("NDCG requires utilities and positive k")
    if len(set(ranking)) != len(ranking) or any(index < 0 or index >= len(true_utility) for index in ranking):
        raise MetricInputError("ranking must contain unique valid candidate indices")
    depth = min(k, len(true_utility), len(ranking))

    def dcg(indices: Sequence[int]) -> float:
        return sum(
            (2 ** max(0.0, float(true_utility[index])) - 1) / math.log2(position + 2)
            for position, index in enumerate(indices[:depth])
        )

    ideal = sorted(range(len(true_utility)), key=lambda index: true_utility[index], reverse=True)
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return 1.0
    return dcg(ranking) / ideal_score


def selection_regret(true_utility: Sequence[float], selected: Sequence[int]) -> float:
    if not true_utility:
        raise MetricInputError("selection regret requires candidate utilities")
    if any(index < 0 or index >= len(true_utility) for index in selected):
        raise MetricInputError("selected candidate index is invalid")
    oracle = max(0.0, max(float(item) for item in true_utility))
    realized = max((float(true_utility[index]) for index in selected), default=0.0)
    return max(0.0, oracle - max(0.0, realized))


@dataclass(frozen=True)
class ContinualMetrics:
    final_average: float
    backward_transfer: float
    forgetting: float


def continual_metrics(transfer_matrix: Sequence[Sequence[float]]) -> ContinualMetrics:
    size = len(transfer_matrix)
    if size == 0 or any(len(row) != size for row in transfer_matrix):
        raise MetricInputError("transfer matrix must be non-empty and square")
    diagonal = [float(transfer_matrix[index][index]) for index in range(size)]
    if any(math.isnan(value) for value in diagonal):
        raise MetricInputError("transfer matrix diagonal must be observed")
    final = [float(value) for value in transfer_matrix[-1]]
    if any(math.isnan(value) for value in final):
        raise MetricInputError("final transfer-matrix row must be observed")

    final_average = sum(final) / size
    if size == 1:
        return ContinualMetrics(final_average, 0.0, 0.0)
    backward_transfer = sum(final[index] - diagonal[index] for index in range(size - 1)) / (size - 1)
    forgetting_values = []
    for task in range(size - 1):
        history = [
            float(transfer_matrix[stage][task])
            for stage in range(task, size)
            if not math.isnan(float(transfer_matrix[stage][task]))
        ]
        forgetting_values.append(max(history) - final[task])
    forgetting = sum(forgetting_values) / len(forgetting_values)
    return ContinualMetrics(final_average, backward_transfer, forgetting)

