"""Validated sparse composition of LoRA adapter deltas."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


class CompositionError(ValueError):
    pass


def compose_lora(
    base: Mapping[str, Tensor],
    candidates: Sequence[Mapping[str, Tensor]],
    coefficients: Sequence[float],
    *,
    coefficient_budget: float = 1.0,
) -> dict[str, Tensor]:
    values = tuple(float(value) for value in coefficients)
    if len(candidates) != len(values):
        raise CompositionError("candidate and coefficient counts must match")
    if coefficient_budget < 0 or any(not math.isfinite(value) or value < 0 for value in values):
        raise CompositionError("coefficients and budget must be finite and non-negative")
    if sum(values) > coefficient_budget + 1e-8:
        raise CompositionError("coefficient sum exceeds budget")

    result = {name: tensor.clone() for name, tensor in base.items()}
    for candidate, coefficient in zip(candidates, values, strict=True):
        if candidate.keys() != base.keys():
            raise CompositionError("adapter parameter keys must match")
        for name, delta in candidate.items():
            if delta.shape != base[name].shape:
                raise CompositionError(f"adapter shape mismatch for {name}")
            if not torch.isfinite(delta).all():
                raise CompositionError(f"adapter contains non-finite values for {name}")
            result[name] = result[name] + coefficient * delta
    return result
