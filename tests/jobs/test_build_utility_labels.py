import pytest

from fcut_vla.jobs.build_utility_labels import (
    PairResult,
    assign_failure_strata,
    build_utility_labels,
)
from fcut_vla.libero.pairing import (
    AdapterCandidate,
    FailureTarget,
    InitialState,
    build_pair_plan,
)
from fcut_vla.utility.counterfactual import CostNormalization


def _plan(*, label_eligible=True):
    adapter_digests = ("neutral", "harmful", "positive", "self-replay", "candidate")
    return build_pair_plan(
        failures=(FailureTarget("failure-sha", "problem"),),
        adapters=tuple(
            AdapterCandidate(digest, "candidate/repo", f"{digest}-rev")
            for digest in adapter_digests
        ),
        seeds=(101,),
        initial_states=(InitialState(0, "state-sha"),),
        episode_budget=1,
        environment={"suite": "libero_spatial"},
        benchmark_hash="benchmark-sha",
        baseline_policy=("base/repo", "base-rev"),
        purpose="counterfactual_repair" if label_eligible else "plumbing_self_replay",
        label_eligible=label_eligible,
    )


def _result(adapter_digest, after, *, plan):
    matching = [pair for pair in plan.pairs if pair.key.adapter_digest == adapter_digest]
    pair_key = matching[0].key if matching else plan.pairs[0].key
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
        pair_plan_hash=plan.content_hash,
        pair_key=pair_key,
        complete=True,
        seen_task=False,
    )


def test_labels_require_complete_pairs_and_assign_no_match():
    plan = _plan()
    labels = build_utility_labels(
        [
            _result("neutral", (0, 0, 0, 0), plan=plan),
            _result("harmful", (0, 0, 0, 0), plan=plan),
        ],
        pair_plans={plan.content_hash: plan},
        cost_normalization=CostNormalization(1000, 1),
    )

    assert assign_failure_strata(labels)["failure-sha"] == "no_match"
    assert all(label.utility.utility_lower <= 0 for label in labels)
    assert all("client" not in label.to_json().lower() for label in labels)


def test_positive_compositional_label_and_incomplete_pair_rejection():
    plan = _plan()
    positive = _result("positive", (1, 1, 1, 1), plan=plan)
    labels = build_utility_labels(
        [positive],
        pair_plans={plan.content_hash: plan},
        cost_normalization=CostNormalization(1000, 1),
    )
    assert assign_failure_strata(labels)["failure-sha"] == "compositional"

    with pytest.raises(ValueError, match="complete"):
        build_utility_labels(
            [positive.__class__(**{**positive.__dict__, "complete": False})],
            pair_plans={plan.content_hash: plan},
            cost_normalization=CostNormalization(1000, 1),
        )


def test_self_replay_pair_is_never_eligible_for_utility_labels():
    plan = _plan(label_eligible=False)
    result = _result("self-replay", (1, 1, 1, 1), plan=plan)

    with pytest.raises(ValueError, match="label-eligible"):
        build_utility_labels(
            [result],
            pair_plans={plan.content_hash: plan},
            cost_normalization=CostNormalization(1000, 1),
        )


def test_utility_labels_reject_missing_or_tampered_pair_plan():
    plan = _plan()
    result = _result("candidate", (1, 1, 1, 1), plan=plan)

    with pytest.raises(ValueError, match="pair plan is missing"):
        build_utility_labels(
            [result],
            pair_plans={},
            cost_normalization=CostNormalization(1000, 1),
        )

    object.__setattr__(plan, "label_eligible", False)
    with pytest.raises(ValueError, match="content hash"):
        build_utility_labels(
            [result],
            pair_plans={plan.content_hash: plan},
            cost_normalization=CostNormalization(1000, 1),
        )


def test_utility_labels_reject_result_not_contained_in_pair_plan():
    plan = _plan()
    unrelated = _result(
        "unrelated-adapter", (1, 1, 1, 1), plan=plan
    )

    with pytest.raises(ValueError, match="exactly one frozen pair"):
        build_utility_labels(
            [unrelated],
            pair_plans={plan.content_hash: plan},
            cost_normalization=CostNormalization(1000, 1),
        )
