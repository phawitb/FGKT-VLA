import pytest

from fcut_vla.jobs.build_utility_labels import (
    PairResult,
    assign_failure_strata,
    build_utility_labels,
)
from fcut_vla.utility.counterfactual import CostNormalization


def _result(adapter_digest, after):
    return PairResult(
        failure_hash="failure-sha",
        adapter_digest=adapter_digest,
        descriptor={"skill_prototype": [0.1], "reliability": [1.0]},
        recovery_before=(0.0, 0.0, 0.0, 0.0),
        recovery_after=tuple(after),
        retention_before={"old": (1.0, 1.0)},
        retention_after={"old": (1.0, 1.0)},
        communication_bytes=100,
        latency_seconds=0.1,
        pair_plan_hash="pair-plan-sha",
        complete=True,
        seen_task=False,
    )


def test_labels_require_complete_pairs_and_assign_no_match():
    labels = build_utility_labels(
        [_result("neutral", (0, 0, 0, 0)), _result("harmful", (0, 0, 0, 0))],
        cost_normalization=CostNormalization(1000, 1),
    )

    assert assign_failure_strata(labels)["failure-sha"] == "no_match"
    assert all(label.utility.utility_lower <= 0 for label in labels)
    assert all("client" not in label.to_json().lower() for label in labels)


def test_positive_compositional_label_and_incomplete_pair_rejection():
    positive = _result("positive", (1, 1, 1, 1))
    labels = build_utility_labels(
        [positive], cost_normalization=CostNormalization(1000, 1)
    )
    assert assign_failure_strata(labels)["failure-sha"] == "compositional"

    with pytest.raises(ValueError, match="complete"):
        build_utility_labels(
            [positive.__class__(**{**positive.__dict__, "complete": False})],
            cost_normalization=CostNormalization(1000, 1),
        )
