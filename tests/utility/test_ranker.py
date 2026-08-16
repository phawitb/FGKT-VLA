import torch

from fcut_vla.utility.ranker import UtilityPrediction, UtilityRanker, select_with_abstention


def test_candidate_scores_are_permutation_equivariant_with_variable_masks():
    torch.manual_seed(7)
    ranker = UtilityRanker(failure_dim=3, descriptor_dim=4, hidden_dim=8, heads=2)
    ranker.eval()
    failure = torch.randn(2, 5, 3)
    descriptors = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    permutation = torch.tensor([2, 0, 3, 1])

    original = ranker(failure, descriptors, candidate_mask=mask)
    permuted = ranker(
        failure,
        descriptors[:, permutation],
        candidate_mask=mask[:, permutation],
    )

    assert torch.allclose(permuted.gain, original.gain[:, permutation], atol=1e-6)
    assert torch.equal(permuted.valid_mask, original.valid_mask[:, permutation])


def test_ranker_exposes_separate_gain_risk_and_uncertainty_heads():
    ranker = UtilityRanker(failure_dim=3, descriptor_dim=4, hidden_dim=8, heads=2)
    prediction = ranker(torch.zeros(1, 2, 3), torch.zeros(1, 3, 4))

    assert prediction.gain.shape == (1, 3)
    assert prediction.risk.shape == (1, 3)
    assert prediction.uncertainty.shape == (1, 3)
    assert prediction.gain.data_ptr() != prediction.risk.data_ptr()
    assert torch.all(prediction.risk >= 0)
    assert torch.all(prediction.uncertainty > 0)


def test_selection_abstains_unless_lower_confidence_bound_is_positive():
    prediction = UtilityPrediction(
        gain=torch.tensor([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]),
        risk=torch.zeros(2, 3),
        uncertainty=torch.tensor([[0.1, 0.9, 0.1], [0.2, 0.2, 0.2]]),
        utility=torch.tensor([[0.8, 0.7, 0.6], [-0.1, 0.0, 0.1]]),
        lower_bound=torch.tensor([[0.7, -0.2, 0.5], [-0.3, -0.2, -0.1]]),
        valid_mask=torch.tensor([[True, True, False], [True, True, True]]),
    )

    assert select_with_abstention(prediction, top_k=2) == ((0,), ())


def test_ranker_public_inputs_contain_no_client_identity_channel():
    ranker = UtilityRanker(failure_dim=3, descriptor_dim=4, hidden_dim=8, heads=2)

    assert "client" not in str(ranker.forward.__annotations__).lower()

