from dataclasses import replace
from hashlib import sha256
import json

import pytest

from fcut_vla.libero.pairing import (
    AdapterCandidate,
    FailureTarget,
    InitialState,
    PairingError,
    build_pair_plan,
    build_repair_pair_plan,
    validate_pair_plan,
    validate_completed_pair,
)


def _plan():
    return build_pair_plan(
        failures=(FailureTarget("failure-sha", "libero_spatial_1"),),
        adapters=(AdapterCandidate("adapter-sha", "phawitbinabik/client-a", "adapter-rev"),),
        seeds=(101, 102),
        initial_states=(InitialState(0, "state-0-sha"), InitialState(1, "state-1-sha")),
        episode_budget=2,
        environment={"suite": "libero_spatial", "use_async_envs": False},
        benchmark_hash="benchmark-sha",
        baseline_policy=("lerobot/smolvla_base", "base-rev"),
    )


def test_pair_plan_preserves_every_seed_and_initial_state():
    plan = _plan()

    assert len(plan.pairs) == 4
    assert [(pair.key.seed, pair.key.initial_state_index) for pair in plan.pairs] == [
        (101, 0),
        (101, 1),
        (102, 0),
        (102, 1),
    ]
    assert plan.content_hash
    assert plan.to_mapping()["content_hash"] == plan.content_hash


def test_pair_plan_validation_rejects_hash_consistent_internal_mismatch():
    plan = _plan()
    pair = plan.pairs[0]
    malformed = replace(
        plan,
        pairs=(replace(pair, candidate=replace(pair.candidate, role="baseline")),),
    )
    payload = malformed.to_mapping()
    payload.pop("content_hash")
    object.__setattr__(
        malformed,
        "content_hash",
        sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )

    with pytest.raises(PairingError, match="roles"):
        validate_pair_plan(malformed)


def test_pair_plan_rejects_label_eligible_self_replay_semantics():
    with pytest.raises(PairingError, match="purpose and label eligibility"):
        build_pair_plan(
            failures=(FailureTarget("failure-sha", "problem"),),
            adapters=(AdapterCandidate("adapter-sha", "repo", "rev"),),
            seeds=(101,),
            initial_states=(InitialState(0, "state-sha"),),
            episode_budget=1,
            environment={"suite": "libero_spatial"},
            benchmark_hash="benchmark-sha",
            baseline_policy=("base/repo", "base-rev"),
            purpose="plumbing_self_replay",
            label_eligible=True,
        )

    plan = replace(_plan(), purpose="plumbing_self_replay", label_eligible=True)
    payload = plan.to_mapping()
    payload.pop("content_hash")
    object.__setattr__(
        plan,
        "content_hash",
        sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
    with pytest.raises(PairingError, match="purpose and label eligibility"):
        validate_pair_plan(plan)


def test_completed_pair_requires_exact_key_roles_and_completion():
    pair = _plan().pairs[0]
    baseline = replace(pair.baseline, complete=True)
    candidate = replace(pair.candidate, complete=True)

    validated = validate_completed_pair(pair, baseline, candidate)

    assert validated == (baseline, candidate)
    with pytest.raises(PairingError, match="initial state"):
        validate_completed_pair(
            pair,
            baseline,
            replace(candidate, initial_state_hash="changed"),
        )
    with pytest.raises(PairingError, match="roles"):
        validate_completed_pair(pair, baseline, replace(baseline, complete=True))
    with pytest.raises(PairingError, match="complete"):
        validate_completed_pair(pair, baseline, pair.candidate)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"seeds": (101, 101)}, "duplicate seed"),
        ({"episode_budget": 0}, "episode budget"),
        ({"benchmark_hash": ""}, "benchmark"),
    ],
)
def test_pair_plan_rejects_ambiguous_or_invalid_contract(kwargs, message):
    arguments = {
        "failures": (FailureTarget("failure-sha", "problem"),),
        "adapters": (AdapterCandidate("adapter-sha", "repo", "rev"),),
        "seeds": (101,),
        "initial_states": (InitialState(0, "state-sha"),),
        "episode_budget": 1,
        "environment": {"suite": "libero_spatial"},
        "benchmark_hash": "benchmark-sha",
        "baseline_policy": ("base/repo", "base-rev"),
    }
    arguments.update(kwargs)

    with pytest.raises(PairingError, match=message):
        build_pair_plan(**arguments)


def test_repair_plan_uses_failed_adapter_as_distinct_baseline():
    source_contract = {
        "source_run_hash": "run-sha",
        "source_episode_sha256": "episode-sha",
        "task_alias": "libero_spatial.task0",
        "problem_id": "problem",
        "instruction": "instruction",
        "dataset_repo": "dataset/repo",
        "dataset_revision": "dataset-sha",
        "base_policy_repo": "base/repo",
        "base_policy_revision": "base-sha",
        "lerobot_commit": "lerobot-sha",
        "peft_version": "0.20.0",
        "hf_libero_version": "0.1.4",
    }
    plan = build_repair_pair_plan(
        failure=FailureTarget("failure-sha", "problem"),
        source_adapter_sha256="source-sha",
        candidate_metadata={
            "recipe_sha256": "recipe-sha",
            "source_adapter_sha256": "source-sha",
            "candidate_adapter_sha256": "candidate-sha",
        },
        recipe={
            "content_hash": "recipe-sha",
            "source_adapter_sha256": "source-sha",
            "failure_hash": "failure-sha",
            "dataset_revision": "dataset-sha",
            **source_contract,
        },
        seed=101,
        initial_state=InitialState(0, "state-sha"),
        environment={"suite": "libero_spatial"},
        benchmark_hash="benchmark-sha",
        source_contract=source_contract,
    )

    pair = plan.pairs[0]
    assert pair.baseline.policy_revision == "source-sha"
    assert pair.candidate.policy_revision == "candidate-sha"
    assert plan.purpose == "counterfactual_repair"
    assert plan.label_eligible is True


def test_repair_plan_rejects_same_or_unbound_candidate():
    source_contract = {
        "source_run_hash": "run-sha",
        "source_episode_sha256": "episode-sha",
        "task_alias": "libero_spatial.task0",
        "problem_id": "problem",
        "instruction": "instruction",
        "dataset_repo": "dataset/repo",
        "dataset_revision": "dataset-sha",
        "base_policy_repo": "base/repo",
        "base_policy_revision": "base-sha",
        "lerobot_commit": "lerobot-sha",
        "peft_version": "0.20.0",
        "hf_libero_version": "0.1.4",
    }
    arguments = {
        "failure": FailureTarget("failure-sha", "problem"),
        "source_adapter_sha256": "source-sha",
        "candidate_metadata": {
            "recipe_sha256": "recipe-sha",
            "source_adapter_sha256": "source-sha",
            "candidate_adapter_sha256": "source-sha",
        },
        "recipe": {
            "content_hash": "recipe-sha",
            "source_adapter_sha256": "source-sha",
            "failure_hash": "failure-sha",
            "dataset_revision": "dataset-sha",
            **source_contract,
        },
        "seed": 101,
        "initial_state": InitialState(0, "state-sha"),
        "environment": {"suite": "libero_spatial"},
        "benchmark_hash": "benchmark-sha",
        "source_contract": source_contract,
    }
    with pytest.raises(PairingError, match="distinct verified adapters"):
        build_repair_pair_plan(**arguments)

    arguments["candidate_metadata"] = {
        **arguments["candidate_metadata"],
        "candidate_adapter_sha256": "candidate-sha",
        "recipe_sha256": "other-recipe",
    }
    with pytest.raises(PairingError, match="candidate metadata"):
        build_repair_pair_plan(**arguments)

    arguments["candidate_metadata"]["candidate_adapter_sha256"] = None
    arguments["candidate_metadata"]["recipe_sha256"] = "recipe-sha"
    with pytest.raises(PairingError, match="must be a string"):
        build_repair_pair_plan(**arguments)

    arguments["candidate_metadata"]["candidate_adapter_sha256"] = "candidate-sha"
    arguments["source_contract"] = {**source_contract, "problem_id": "other"}
    arguments["recipe"] = {**arguments["recipe"], "problem_id": "other"}
    with pytest.raises(PairingError, match="provenance"):
        build_repair_pair_plan(**arguments)
