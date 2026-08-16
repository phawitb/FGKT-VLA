from dataclasses import replace

import pytest

from fcut_vla.libero.pairing import (
    AdapterCandidate,
    FailureTarget,
    InitialState,
    PairingError,
    build_pair_plan,
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
