import torch

from fcut_vla.repair.personalized import build_personalized_repair
from fcut_vla.utility.ranker import UtilityPrediction


def _prediction(lower_bound, valid_mask=None):
    lower = torch.tensor([lower_bound], dtype=torch.float32)
    mask = torch.ones_like(lower, dtype=torch.bool) if valid_mask is None else torch.tensor([valid_mask])
    return UtilityPrediction(
        gain=lower,
        risk=torch.zeros_like(lower),
        uncertainty=torch.ones_like(lower) * 0.1,
        utility=lower,
        lower_bound=lower,
        valid_mask=mask,
    )


def test_personalized_repair_has_deterministic_top1_tie_break():
    target = {"A": torch.tensor([1.0])}
    candidates = {
        "adapter-b": {"A": torch.tensor([20.0])},
        "adapter-a": {"A": torch.tensor([10.0])},
    }

    repair = build_personalized_repair(
        target, candidates, _prediction([0.5, 0.5]), top_k=1, coefficient_budget=0.5
    )

    assert repair.selected_adapter_ids == ("adapter-a",)
    assert repair.coefficients == (0.5,)
    assert torch.allclose(repair.state_dict["A"], torch.tensor([6.0]))


def test_personalized_repair_abstains_when_no_candidate_has_positive_bound():
    target = {"A": torch.tensor([1.0])}
    repair = build_personalized_repair(
        target,
        {"adapter-a": {"A": torch.tensor([10.0])}},
        _prediction([-0.01]),
        top_k=1,
    )

    assert repair.abstained
    assert repair.selected_adapter_ids == ()
    assert torch.equal(repair.state_dict["A"], target["A"])
    assert repair.state_dict["A"].data_ptr() != target["A"].data_ptr()


def test_personalized_repair_respects_candidate_mask_and_budget():
    target = {"A": torch.tensor([0.0])}
    candidates = {
        "a": {"A": torch.tensor([1.0])},
        "b": {"A": torch.tensor([2.0])},
        "c": {"A": torch.tensor([3.0])},
    }
    repair = build_personalized_repair(
        target,
        candidates,
        _prediction([0.4, 0.9, 0.2], [True, False, True]),
        top_k=2,
        coefficient_budget=0.6,
    )

    assert repair.selected_adapter_ids == ("a", "c")
    assert all(value >= 0 for value in repair.coefficients)
    assert sum(repair.coefficients) <= 0.600001
