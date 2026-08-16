"""Failure-conditioned, identity-free candidate utility ranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UtilityPrediction:
    gain: Tensor
    risk: Tensor
    uncertainty: Tensor
    utility: Tensor
    lower_bound: Tensor
    valid_mask: Tensor


class UtilityRanker(nn.Module):
    """Score each adapter descriptor against failure tokens without identity features."""

    def __init__(
        self,
        *,
        failure_dim: int,
        descriptor_dim: int,
        hidden_dim: int,
        heads: int,
        lambda_retention: float = 1.0,
        confidence_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.failure_projection = nn.Linear(failure_dim, hidden_dim)
        self.descriptor_projection = nn.Linear(descriptor_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.lambda_retention = lambda_retention
        self.confidence_multiplier = confidence_multiplier

    def forward(
        self,
        failure: Tensor,
        descriptors: Tensor,
        candidate_mask: Tensor | None = None,
        failure_mask: Tensor | None = None,
        normalized_cost: Tensor | None = None,
        lambda_cost: float = 0.0,
    ) -> UtilityPrediction:
        if failure.ndim != 3 or descriptors.ndim != 3:
            raise ValueError("failure and descriptors must be batched token tensors")
        batch, candidates, _ = descriptors.shape
        if failure.shape[0] != batch:
            raise ValueError("failure and descriptor batch sizes must match")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                (batch, candidates), dtype=torch.bool, device=descriptors.device
            )
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask has incompatible shape")

        queries = self.descriptor_projection(descriptors)
        memory = self.failure_projection(failure)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=None if failure_mask is None else ~failure_mask,
            need_weights=False,
        )
        features = queries + attended
        gain = self.gain_head(features).squeeze(-1)
        risk = F.softplus(self.risk_head(features).squeeze(-1))
        uncertainty = F.softplus(self.uncertainty_head(features).squeeze(-1)) + 1e-6
        utility = gain - self.lambda_retention * risk
        if normalized_cost is not None:
            if normalized_cost.shape != utility.shape:
                raise ValueError("normalized_cost has incompatible shape")
            utility = utility - lambda_cost * normalized_cost
        lower_bound = utility - self.confidence_multiplier * uncertainty
        return UtilityPrediction(gain, risk, uncertainty, utility, lower_bound, candidate_mask)


def select_with_abstention(
    prediction: UtilityPrediction, *, top_k: int
) -> tuple[tuple[int, ...], ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selections: list[tuple[int, ...]] = []
    for lower, valid in zip(prediction.lower_bound, prediction.valid_mask, strict=True):
        eligible = [index for index in range(lower.numel()) if valid[index] and lower[index] > 0]
        eligible.sort(key=lambda index: (-float(lower[index]), index))
        selections.append(tuple(eligible[:top_k]))
    return tuple(selections)
