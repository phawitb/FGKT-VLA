"""Privacy-bounded authoritative episode recording for official LIBERO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np

from fcut_vla.libero.episode import EpisodeKey, EpisodeRecord, EpisodeStep


def hash_initial_state(state: np.ndarray) -> str:
    """Hash one selected simulator state with canonical shape, dtype, and bytes."""
    array = np.asarray(state)
    if array.dtype.hasobject:
        raise ValueError("initial state must not contain Python objects")
    little_endian_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(little_endian_dtype, copy=False))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(header + b"\0" + canonical.tobytes(order="C")).hexdigest()


def executed_action_summary(action: Iterable[float]) -> tuple[float, float, float, float]:
    """Return mean, population stddev, L2 norm, and max-absolute executed action."""
    values = tuple(float(value) for value in action)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("executed action must contain finite values")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    l2_norm = math.sqrt(math.fsum(value * value for value in values))
    return (mean, math.sqrt(variance), l2_norm, max(abs(value) for value in values))


@dataclass(frozen=True)
class EpisodeIdentity:
    task_alias: str
    problem_id: str
    seed: int
    episode_index: int
    initial_state_index: int
    initial_state_hash: str
    policy_repo: str
    policy_revision: str
    instruction: str


class SanitizedEpisodeRecorder:
    """Collect one canonical episode without retaining images or simulator state."""

    def __init__(self, identity: EpisodeIdentity, *, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.identity = identity
        self.max_steps = max_steps
        self._steps: list[EpisodeStep] = []
        self._dimensions: tuple[int, int, int, int] | None = None
        self._terminal = False

    def append(
        self,
        *,
        features: Iterable[float],
        proprioception: Iterable[float],
        executed_action: Iterable[float],
        action_statistics: Iterable[float],
        reward: float,
        success: bool,
        env_done: bool,
    ) -> None:
        if self._terminal:
            raise ValueError("cannot append after terminal episode step")
        if success and not env_done:
            raise ValueError("success must coincide with environment termination")

        reaches_horizon = len(self._steps) + 1 == self.max_steps
        step = EpisodeStep.from_mapping(
            {
                "features": tuple(features),
                "proprioception": tuple(proprioception),
                "executed_action": tuple(executed_action),
                "action_statistics": tuple(action_statistics),
                "reward": reward,
                "done": bool(env_done or reaches_horizon),
                "success": bool(success),
            }
        )
        dimensions = step.dimensions()
        if self._dimensions is None:
            self._dimensions = dimensions
        elif dimensions != self._dimensions:
            raise ValueError("episode step dimensions changed during recording")
        self._steps.append(step)
        self._terminal = step.done

    def finalize(self) -> EpisodeRecord:
        if not self._steps or not self._terminal:
            raise ValueError("episode cannot finalize before a terminal step")
        identity = self.identity
        record = EpisodeRecord(
            schema_version=1,
            key=EpisodeKey(
                identity.task_alias,
                identity.seed,
                identity.episode_index,
                identity.initial_state_index,
            ),
            problem_id=identity.problem_id,
            initial_state_hash=identity.initial_state_hash,
            policy_repo=identity.policy_repo,
            policy_revision=identity.policy_revision,
            instruction=identity.instruction.strip(),
            terminal_success=self._steps[-1].success,
            steps=tuple(self._steps),
        )
        record.validate()
        return record
