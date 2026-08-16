"""Single-environment rollout loop feeding the authoritative FGKT recorder."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
import math
from typing import Any

import numpy as np

from fcut_vla.libero.episode import EpisodeRecord
from fcut_vla.libero.recorder import SanitizedEpisodeRecorder, executed_action_summary


@dataclass(frozen=True)
class PreparedObservation:
    policy_input: Any
    features: tuple[float, ...]
    proprioception: tuple[float, ...]


def image_moments_v1(policy_observation: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Extract identity-free RGB moments and the exact policy-compatible state vector."""
    image_keys = sorted(key for key in policy_observation if key.startswith("observation.images."))
    if len(image_keys) != 2:
        raise ValueError("image_moments_v1 requires exactly two RGB cameras")

    features: list[float] = []
    for key in image_keys:
        image = policy_observation[key]
        if tuple(image.shape[:2]) != (1, 3) or len(image.shape) != 4:
            raise ValueError("image_moments_v1 requires batched channel-first RGB cameras")
        means = image.mean(dim=(2, 3)).detach().to("cpu").reshape(-1).tolist()
        stddevs = image.std(dim=(2, 3), correction=0).detach().to("cpu").reshape(-1).tolist()
        features.extend(float(value) for value in means)
        features.extend(float(value) for value in stddevs)

    state = policy_observation.get("observation.state")
    if state is None or tuple(state.shape) != (1, 8):
        raise ValueError("image_moments_v1 requires one eight-value policy state")
    proprioception = tuple(float(value) for value in state.detach().to("cpu").reshape(-1).tolist())
    if not all(math.isfinite(value) for value in (*features, *proprioception)):
        raise ValueError("image moments and proprioception must be finite")
    return tuple(features), proprioception


def _first_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("single-environment rollout received a non-scalar flag")
    return bool(array.reshape(-1)[0])


def _success_from_info(info: Mapping[str, Any]) -> bool:
    final_info = info.get("final_info")
    if isinstance(final_info, Mapping) and "is_success" in final_info:
        return _first_bool(final_info["is_success"])
    if isinstance(final_info, (list, tuple, np.ndarray)) and len(final_info) == 1:
        item = final_info[0]
        if isinstance(item, Mapping) and "is_success" in item:
            return bool(item["is_success"])
    if "is_success" in info:
        return _first_bool(info["is_success"])
    return False


def run_single_episode(
    env: Any,
    recorder: SanitizedEpisodeRecorder,
    *,
    prepare: Callable[[Any], PreparedObservation],
    select_action: Callable[[Any], tuple[float, ...]],
) -> EpisodeRecord:
    """Run one synchronous vectorized environment with an exact batch size of one."""
    if int(getattr(env, "num_envs", -1)) != 1:
        raise ValueError("authoritative recorder requires environment batch size 1")

    observation, _ = env.reset(
        seed=[recorder.identity.seed],
        options={"lerobot_new_rollout": True},
    )
    for _ in range(recorder.max_steps):
        prepared = prepare(observation)
        action = tuple(float(value) for value in select_action(prepared.policy_input))
        if len(action) != 7 or not all(math.isfinite(value) for value in action):
            raise ValueError("LIBERO policy must produce exactly seven finite action values")
        action_summary = executed_action_summary(action)
        next_observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        env_done = _first_bool(terminated) or _first_bool(truncated)
        success = _success_from_info(info)
        recorder.append(
            features=prepared.features,
            proprioception=prepared.proprioception,
            executed_action=action,
            action_statistics=action_summary,
            reward=float(np.asarray(reward).reshape(-1)[0]),
            success=success,
            env_done=env_done,
        )
        observation = next_observation
        if env_done:
            break
    return recorder.finalize()
