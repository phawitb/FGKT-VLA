"""Canonical, hash-addressed episode records produced by LIBERO evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any


class EpisodeValidationError(ValueError):
    """Raised when recorded evaluation data cannot be trusted."""


@dataclass(frozen=True, order=True)
class EpisodeKey:
    task_alias: str
    seed: int
    episode_index: int
    initial_state_index: int


@dataclass(frozen=True)
class EpisodeStep:
    features: tuple[float, ...]
    proprioception: tuple[float, ...]
    executed_action: tuple[float, ...]
    action_statistics: tuple[float, ...]
    reward: float
    done: bool
    success: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EpisodeStep":
        step = cls(
            features=tuple(float(value) for value in raw["features"]),
            proprioception=tuple(float(value) for value in raw["proprioception"]),
            executed_action=tuple(float(value) for value in raw["executed_action"]),
            action_statistics=tuple(float(value) for value in raw["action_statistics"]),
            reward=float(raw["reward"]),
            done=bool(raw["done"]),
            success=bool(raw["success"]),
        )
        vectors = (
            step.features,
            step.proprioception,
            step.executed_action,
            step.action_statistics,
        )
        if any(not vector for vector in vectors):
            raise EpisodeValidationError("episode step vectors must be non-empty")
        if not all(math.isfinite(value) for vector in vectors for value in vector) or not math.isfinite(
            step.reward
        ):
            raise EpisodeValidationError("episode step values must be finite")
        return step

    def dimensions(self) -> tuple[int, int, int, int]:
        return tuple(
            len(vector)
            for vector in (
                self.features,
                self.proprioception,
                self.executed_action,
                self.action_statistics,
            )
        )

    def failure_features(self) -> dict[str, tuple[float, ...]]:
        return {
            "features": self.features,
            "proprioception": self.proprioception,
            "executed_action": self.executed_action,
            "action_statistics": self.action_statistics,
        }


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: int
    key: EpisodeKey
    problem_id: str
    initial_state_hash: str
    policy_repo: str
    policy_revision: str
    instruction: str
    terminal_success: bool
    steps: tuple[EpisodeStep, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EpisodeRecord":
        try:
            record = cls(
                schema_version=int(raw["schema_version"]),
                key=EpisodeKey(
                    task_alias=str(raw["task_alias"]),
                    seed=int(raw["seed"]),
                    episode_index=int(raw["episode_index"]),
                    initial_state_index=int(raw["initial_state_index"]),
                ),
                problem_id=str(raw["problem_id"]),
                initial_state_hash=str(raw["initial_state_hash"]),
                policy_repo=str(raw["policy_repo"]),
                policy_revision=str(raw["policy_revision"]),
                instruction=str(raw["instruction"]).strip(),
                terminal_success=bool(raw["terminal_success"]),
                steps=tuple(EpisodeStep.from_mapping(step) for step in raw["steps"]),
            )
        except (KeyError, TypeError) as error:
            raise EpisodeValidationError(f"malformed episode record: {error}") from error
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != 1:
            raise EpisodeValidationError("unsupported episode schema version")
        if self.key.seed < 0 or self.key.episode_index < 0 or self.key.initial_state_index < 0:
            raise EpisodeValidationError("episode indices and seed must be non-negative")
        required = (
            self.key.task_alias,
            self.problem_id,
            self.initial_state_hash,
            self.policy_repo,
            self.policy_revision,
            self.instruction,
        )
        if any(not value for value in required):
            raise EpisodeValidationError("episode identity fields must be non-empty")
        if not self.steps:
            raise EpisodeValidationError("episode requires at least one step")
        dimensions = self.steps[0].dimensions()
        if any(step.dimensions() != dimensions for step in self.steps):
            raise EpisodeValidationError("episode step dimensions must be consistent")
        terminal = self.steps[-1]
        if not terminal.done:
            raise EpisodeValidationError("last episode step must be terminal")
        if terminal.success != self.terminal_success:
            raise EpisodeValidationError("last-step success must match terminal_success")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_alias": self.key.task_alias,
            "problem_id": self.problem_id,
            "seed": self.key.seed,
            "episode_index": self.key.episode_index,
            "initial_state_index": self.key.initial_state_index,
            "initial_state_hash": self.initial_state_hash,
            "policy_repo": self.policy_repo,
            "policy_revision": self.policy_revision,
            "instruction": self.instruction,
            "terminal_success": self.terminal_success,
            "steps": [asdict(step) for step in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return sha256(self.to_json().encode()).hexdigest()


def validate_episode_shard(path: Path) -> tuple[EpisodeRecord, ...]:
    records: list[EpisodeRecord] = []
    keys: set[EpisodeKey] = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = EpisodeRecord.from_mapping(json.loads(line))
        except (json.JSONDecodeError, EpisodeValidationError) as error:
            raise EpisodeValidationError(f"invalid episode at line {line_number}: {error}") from error
        if record.key in keys:
            raise EpisodeValidationError(f"duplicate episode key: {record.key}")
        keys.add(record.key)
        records.append(record)
    if not records:
        raise EpisodeValidationError("episode shard must be non-empty")
    return tuple(sorted(records, key=lambda record: record.key))
