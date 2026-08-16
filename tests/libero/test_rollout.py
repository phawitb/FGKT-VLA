from dataclasses import dataclass

import numpy as np
import pytest
import torch

from fcut_vla.libero.recorder import EpisodeIdentity, SanitizedEpisodeRecorder
from fcut_vla.libero.rollout import (
    PreparedObservation,
    image_moments_v1,
    run_single_episode,
)


def _identity() -> EpisodeIdentity:
    return EpisodeIdentity(
        "spatial.task0",
        "pick_up_black_bowl",
        101,
        0,
        0,
        "state-sha",
        "phawitbinabik/fgkt-vla",
        "policy-sha",
        "pick up the black bowl",
    )


@dataclass
class FakeVectorEnv:
    success_at: int | None = None
    final_info_form: str | None = None
    num_envs: int = 1
    steps: int = 0
    reset_seed: object = None

    def reset(self, *, seed, options):
        self.reset_seed = seed
        return {"state": np.zeros((1, 8), dtype=np.float32)}, {}

    def step(self, action):
        assert action.shape == (1, 7)
        self.steps += 1
        success = self.steps == self.success_at
        observation = {"state": np.full((1, 8), self.steps, dtype=np.float32)}
        info = {"is_success": np.array([success])}
        if success and self.final_info_form == "mapping":
            info = {
                "is_success": np.array([False]),
                "final_info": {"is_success": np.array([True])},
            }
        elif success and self.final_info_form == "sequence":
            info = {
                "is_success": np.array([False]),
                "final_info": [{"is_success": True}],
            }
        return (
            observation,
            np.array([float(success)]),
            np.array([success]),
            np.array([False]),
            info,
        )


def _prepare(observation):
    state = tuple(float(value) for value in observation["state"][0])
    return PreparedObservation(
        policy_input=observation,
        features=(state[0], 0.5),
        proprioception=state,
    )


def _select_action(policy_input):
    assert "state" in policy_input
    return (0.0,) * 7


def test_rollout_forces_terminal_failure_at_horizon_and_uses_frozen_seed():
    env = FakeVectorEnv()
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=2)

    record = run_single_episode(env, recorder, prepare=_prepare, select_action=_select_action)

    assert env.reset_seed == [101]
    assert env.steps == 2
    assert not record.terminal_success
    assert record.steps[-1].done


def test_rollout_stops_at_success_before_horizon():
    env = FakeVectorEnv(success_at=2)
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=10)

    record = run_single_episode(env, recorder, prepare=_prepare, select_action=_select_action)

    assert env.steps == 2
    assert record.terminal_success
    assert record.steps[-1].reward == 1.0


def test_rollout_rejects_non_single_vector_environment():
    env = FakeVectorEnv(num_envs=2)
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=2)

    with pytest.raises(ValueError, match="batch size 1"):
        run_single_episode(env, recorder, prepare=_prepare, select_action=_select_action)


def test_image_moments_v1_is_ordered_numeric_and_discards_pixels():
    observation = {
        "observation.state": torch.arange(8, dtype=torch.float32).reshape(1, 8),
        "observation.images.wrist_image": torch.ones((1, 3, 2, 2)),
        "observation.images.image": torch.zeros((1, 3, 2, 2)),
    }

    features, proprioception = image_moments_v1(observation)

    assert features == (0.0,) * 6 + (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert proprioception == tuple(float(value) for value in range(8))
    assert all(not hasattr(value, "shape") for value in features)


def test_image_moments_v1_requires_two_rgb_cameras_and_eight_state_values():
    bad = {
        "observation.state": torch.zeros((1, 7)),
        "observation.images.image": torch.zeros((1, 3, 2, 2)),
    }

    with pytest.raises(ValueError, match="two RGB cameras"):
        image_moments_v1(bad)


@pytest.mark.parametrize("final_info_form", ["mapping", "sequence"])
def test_terminal_final_info_takes_precedence_over_top_level_status(final_info_form):
    env = FakeVectorEnv(success_at=1, final_info_form=final_info_form)
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=2)

    record = run_single_episode(env, recorder, prepare=_prepare, select_action=_select_action)

    assert record.terminal_success


@pytest.mark.parametrize("action", [(0.0,) * 6, (0.0,) * 6 + (float("nan"),)])
def test_invalid_action_never_reaches_environment(action):
    env = FakeVectorEnv()
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=2)

    with pytest.raises(ValueError, match="seven finite"):
        run_single_episode(env, recorder, prepare=_prepare, select_action=lambda _: action)

    assert env.steps == 0
