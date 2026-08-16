"""Privacy-bounded temporal context extracted from a failed episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

from fcut_vla.types import FailureId


class FailureContextError(ValueError):
    """Raised when an episode cannot define a valid failure window."""


@dataclass(frozen=True)
class FailureStep:
    features: tuple[float, ...]
    proprioception: tuple[float, ...]
    executed_action: tuple[float, ...]
    action_statistics: tuple[float, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Sequence[float]]) -> "FailureStep":
        return cls(
            features=tuple(float(value) for value in raw["features"]),
            proprioception=tuple(float(value) for value in raw["proprioception"]),
            executed_action=tuple(float(value) for value in raw["executed_action"]),
            action_statistics=tuple(float(value) for value in raw["action_statistics"]),
        )

    @classmethod
    def zeros_like(cls, other: "FailureStep") -> "FailureStep":
        return cls(
            features=(0.0,) * len(other.features),
            proprioception=(0.0,) * len(other.proprioception),
            executed_action=(0.0,) * len(other.executed_action),
            action_statistics=(0.0,) * len(other.action_statistics),
        )

    def dimensions(self) -> tuple[int, int, int, int]:
        return (
            len(self.features),
            len(self.proprioception),
            len(self.executed_action),
            len(self.action_statistics),
        )

    def validate(self) -> None:
        values = self.features + self.proprioception + self.executed_action + self.action_statistics
        if not all(math.isfinite(value) for value in values):
            raise FailureContextError("failure step values must be finite")


@dataclass(frozen=True)
class FailureContext:
    failure_id: FailureId
    instruction: str
    steps: tuple[FailureStep, ...]
    mask: tuple[bool, ...]
    failure_step: int

    @classmethod
    def from_episode(
        cls,
        *,
        failure_id: FailureId,
        instruction: str,
        steps: Sequence[Mapping[str, Sequence[float]]],
        failure_step: int,
        window_size: int,
        terminal_success: bool,
    ) -> "FailureContext":
        if terminal_success:
            raise FailureContextError("context requires a failed episode")
        if not 0 <= failure_step < len(steps):
            raise FailureContextError("failure_step must index an episode step")
        if window_size < 1:
            raise FailureContextError("window_size must be positive")
        instruction = instruction.strip()
        if not instruction:
            raise FailureContextError("instruction must be non-empty")

        parsed = tuple(FailureStep.from_mapping(raw) for raw in steps)
        reference_dims = parsed[0].dimensions()
        if any(step.dimensions() != reference_dims for step in parsed):
            raise FailureContextError("episode steps must have consistent dimensions")
        for step in parsed:
            step.validate()

        start = max(0, failure_step - window_size + 1)
        selected = parsed[start : failure_step + 1]
        padding_count = window_size - len(selected)
        padding = (FailureStep.zeros_like(parsed[0]),) * padding_count
        context_steps = padding + selected
        mask = (False,) * padding_count + (True,) * len(selected)
        return cls(
            failure_id=failure_id,
            instruction=instruction,
            steps=context_steps,
            mask=mask,
            failure_step=failure_step,
        )

    def to_json(self) -> str:
        payload = {
            "failure_id": str(self.failure_id),
            "instruction": self.instruction,
            "steps": [asdict(step) for step in self.steps],
            "mask": self.mask,
            "failure_step": self.failure_step,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

