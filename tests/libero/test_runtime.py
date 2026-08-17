from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from fcut_vla.libero.rollout import PreparedObservation
from fcut_vla.libero.runtime import (
    LeRobotRecordPlan,
    RuntimeComponents,
    RuntimeTask,
    record_lerobot_episode,
    _build_with_env_cleanup,
    verified_hf_libero_version,
)


@dataclass
class FakeEnv:
    num_envs: int = 1
    steps: int = 0

    def reset(self, *, seed, options):
        return {"state": np.zeros((1, 8), dtype=np.float32)}, {}

    def step(self, action):
        self.steps += 1
        return (
            {"state": np.ones((1, 8), dtype=np.float32)},
            np.array([0.0]),
            np.array([False]),
            np.array([False]),
            {"is_success": np.array([False])},
        )


class FakeBindings:
    def __init__(self):
        self.env = FakeEnv()

    def resolve_task(self, plan):
        return RuntimeTask(
            problem_id="pick_up_black_bowl",
            instruction="pick up the black bowl",
            initial_states=np.arange(12, dtype=np.float32).reshape(3, 4),
            max_steps=2,
        )

    def build_components(self, plan, task):
        def prepare(observation):
            return PreparedObservation(
                policy_input=observation,
                features=(0.1, 0.2),
                proprioception=tuple(float(x) for x in observation["state"][0]),
            )

        return RuntimeComponents(self.env, prepare, lambda _: (0.0,) * 7, lambda: None)


def test_runtime_binds_resolved_task_selected_state_and_verified_policy(tmp_path):
    plan = LeRobotRecordPlan(
        checkpoint=Path("/private/adapter"),
        policy_repo="phawitbinabik/fgkt-vla-adapters",
        policy_revision="adapter-sha",
        suite="libero_spatial",
        task_id=0,
        task_alias="spatial.task0",
        seed=101,
        episode_index=0,
        initial_state_index=1,
        output_path=tmp_path / "episode.json",
    )
    bindings = FakeBindings()

    record = record_lerobot_episode(plan, bindings=bindings)

    assert record.problem_id == "pick_up_black_bowl"
    assert record.initial_state_hash != ""
    assert record.key.initial_state_index == 1
    assert record.policy_revision == "adapter-sha"
    assert not record.terminal_success
    assert bindings.env.steps == 2
    assert plan.output_path.read_text() == record.to_json() + "\n"


def test_runtime_closes_environment_after_rollout_failure(tmp_path):
    plan = LeRobotRecordPlan(
        Path("adapter"),
        "phawitbinabik/fgkt-vla-adapters",
        "adapter-sha",
        "libero_spatial",
        0,
        "spatial.task0",
        101,
        0,
        0,
        tmp_path / "episode.json",
    )
    bindings = FakeBindings()
    closed = []
    components = bindings.build_components(plan, bindings.resolve_task(plan))
    bindings.build_components = lambda plan, task: RuntimeComponents(
        components.env,
        components.prepare,
        lambda _: (_ for _ in ()).throw(RuntimeError("policy failed")),
        lambda: closed.append(True),
    )

    try:
        record_lerobot_episode(plan, bindings=bindings)
    except RuntimeError as error:
        assert str(error) == "policy failed"
    else:
        raise AssertionError("expected policy failure")

    assert closed == [True]


def test_official_binding_closes_environment_when_policy_construction_fails():
    closed = []

    class Env:
        def close(self):
            closed.append(True)

    def fail():
        raise RuntimeError("PEFT load failed")

    try:
        _build_with_env_cleanup(Env(), fail)
    except RuntimeError as error:
        assert str(error) == "PEFT load failed"
    else:
        raise AssertionError("expected construction failure")

    assert closed == [True]


def test_hf_libero_version_must_match_frozen_runtime(monkeypatch):
    monkeypatch.setattr("fcut_vla.libero.runtime.distribution_version", lambda _: "0.1.5")

    with pytest.raises(ValueError, match="hf-libero version"):
        verified_hf_libero_version("0.1.4")
