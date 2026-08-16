import pytest

from fcut_vla.utility.counterfactual import (
    CostNormalization,
    CounterfactualInputError,
    UtilityObservation,
    compute_utility,
)


def test_compute_utility_combines_paired_gain_retention_and_normalized_cost():
    observation = compute_utility(
        failure_id="f1",
        adapter_id="a1",
        recovery_before=[0, 0, 1, 0],
        recovery_after=[1, 1, 1, 0],
        retention_before={"old-a": [1, 1], "old-b": [1, 1]},
        retention_after={"old-a": [1, 0], "old-b": [1, 1]},
        communication_bytes=500,
        latency_seconds=2,
        cost_normalization=CostNormalization(bytes_scale=1000, latency_scale=10),
        lambda_retention=0.5,
        lambda_cost=0.2,
    )
    assert observation.recovery_gain == pytest.approx(0.5)
    assert observation.retention_risk == pytest.approx(0.25)
    assert observation.normalized_cost == pytest.approx(0.35)
    assert observation.utility == pytest.approx(0.305)


def test_utility_confidence_controls_positive_and_tied_decisions():
    positive = UtilityObservation("f", "positive", 0.4, 0.0, 0.0, 0.4, 0.1, 0.5)
    uncertain = UtilityObservation("f", "uncertain", 0.2, 0.0, 0.0, 0.2, -0.1, 0.5)
    tied = UtilityObservation("f", "tied", 0.3, 0.0, 0.0, 0.3, 0.2, 0.6)
    assert positive.has_positive_utility
    assert not uncertain.has_positive_utility
    assert not positive.is_confidently_above(tied)


def test_no_match_when_every_lower_bound_is_non_positive():
    observations = [
        UtilityObservation("f", "a", 0.0, 0.0, 0.0, 0.0, -0.1, 0.1),
        UtilityObservation("f", "b", -0.2, 0.0, 0.0, -0.2, -0.4, 0.0),
    ]
    assert not any(item.has_positive_utility for item in observations)


def test_rejects_unpaired_rollouts_and_invalid_cost_scales():
    kwargs = dict(
        failure_id="f1",
        adapter_id="a1",
        recovery_before=[0, 1],
        recovery_after=[1],
        retention_before={},
        retention_after={},
        communication_bytes=1,
        latency_seconds=1,
        cost_normalization=CostNormalization(bytes_scale=1, latency_scale=1),
    )
    with pytest.raises(CounterfactualInputError, match="paired recovery"):
        compute_utility(**kwargs)
    with pytest.raises(CounterfactualInputError, match="positive"):
        CostNormalization(bytes_scale=0, latency_scale=1)


def test_rejects_retention_task_or_seed_mismatch():
    common = dict(
        failure_id="f1",
        adapter_id="a1",
        recovery_before=[0],
        recovery_after=[1],
        communication_bytes=1,
        latency_seconds=1,
        cost_normalization=CostNormalization(bytes_scale=1, latency_scale=1),
    )
    with pytest.raises(CounterfactualInputError, match="retention task sets"):
        compute_utility(
            **common,
            retention_before={"old": [1]},
            retention_after={"other": [1]},
        )
    with pytest.raises(CounterfactualInputError, match="paired retention"):
        compute_utility(
            **common,
            retention_before={"old": [1, 1]},
            retention_after={"old": [1]},
        )

