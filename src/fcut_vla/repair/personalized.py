"""Build sparse, failure-conditioned repairs from private adapter deltas."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from torch import Tensor

from fcut_vla.adapters.composition import compose_lora
from fcut_vla.utility.ranker import UtilityPrediction, select_with_abstention


@dataclass(frozen=True)
class PersonalizedRepair:
    state_dict: dict[str, Tensor]
    selected_adapter_ids: tuple[str, ...]
    coefficients: tuple[float, ...]
    abstained: bool


def build_personalized_repair(
    target_adapter: Mapping[str, Tensor],
    candidate_adapters: Mapping[str, Mapping[str, Tensor]],
    prediction: UtilityPrediction,
    *,
    top_k: int,
    coefficient_budget: float = 1.0,
) -> PersonalizedRepair:
    if prediction.lower_bound.shape[0] != 1:
        raise ValueError("personalized repair expects one target failure")
    ordered_ids = tuple(sorted(candidate_adapters))
    if prediction.lower_bound.shape[1] != len(ordered_ids):
        raise ValueError("prediction count must match candidate adapter count")

    selected_indices = select_with_abstention(prediction, top_k=top_k)[0]
    if not selected_indices:
        return PersonalizedRepair(
            {name: tensor.clone() for name, tensor in target_adapter.items()}, (), (), True
        )

    selected_ids = tuple(ordered_ids[index] for index in selected_indices)
    positive_bounds = [float(prediction.lower_bound[0, index]) for index in selected_indices]
    total = sum(positive_bounds)
    coefficients = tuple(coefficient_budget * value / total for value in positive_bounds)
    composed = compose_lora(
        target_adapter,
        [candidate_adapters[adapter_id] for adapter_id in selected_ids],
        coefficients,
        coefficient_budget=coefficient_budget,
    )
    return PersonalizedRepair(composed, selected_ids, coefficients, False)
