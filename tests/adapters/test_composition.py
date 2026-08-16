import pytest
import torch

from fcut_vla.adapters.composition import CompositionError, compose_lora


def test_composition_uses_nonnegative_sparse_coefficients_with_sum_bound():
    base = {"layer.A": torch.tensor([1.0, 2.0])}
    candidates = [
        {"layer.A": torch.tensor([2.0, 0.0])},
        {"layer.A": torch.tensor([0.0, 4.0])},
    ]

    result = compose_lora(base, candidates, coefficients=(0.25, 0.5), coefficient_budget=1.0)

    assert torch.allclose(result["layer.A"], torch.tensor([1.5, 4.0]))


@pytest.mark.parametrize("coefficients", [(-0.1, 0.2), (0.8, 0.3)])
def test_composition_rejects_negative_or_over_budget_coefficients(coefficients):
    state = {"A": torch.ones(1)}

    with pytest.raises(CompositionError):
        compose_lora(state, [state, state], coefficients, coefficient_budget=1.0)

