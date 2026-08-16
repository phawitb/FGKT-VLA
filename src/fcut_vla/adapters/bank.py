"""Adapter metadata bank with a strict privacy boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from fcut_vla.types import AdapterId


class AdapterBankError(ValueError):
    """Raised when adapter metadata is unsafe or structurally incompatible."""


FORBIDDEN_FIELDS = frozenset(
    {
        "raw_images",
        "raw_trajectories",
        "raw_trajectory",
        "task_id",
        "client_id",
        "source_client_id",
        "client_ownership",
        "private_path",
    }
)


@dataclass(frozen=True)
class AdapterDescriptor:
    skill_prototype: tuple[float, ...]
    update_sketch: tuple[float, ...]
    reliability: tuple[float, ...]
    compatibility: tuple[float, ...]
    adapter_norm: float
    data_count: int

    def __post_init__(self) -> None:
        vector_fields = (
            self.skill_prototype,
            self.update_sketch,
            self.reliability,
            self.compatibility,
        )
        if any(not values for values in vector_fields):
            raise AdapterBankError("descriptor vectors must be non-empty")
        all_values = tuple(value for values in vector_fields for value in values) + (
            self.adapter_norm,
        )
        if not all(math.isfinite(float(value)) for value in all_values):
            raise AdapterBankError("descriptor values must be finite")
        if self.adapter_norm < 0:
            raise AdapterBankError("adapter_norm must be non-negative")
        if self.data_count < 1:
            raise AdapterBankError("data_count must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdapterDescriptor":
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(raw))
        if forbidden:
            raise AdapterBankError(f"forbidden descriptor field: {', '.join(forbidden)}")
        return cls(
            skill_prototype=tuple(float(value) for value in raw["skill_prototype"]),
            update_sketch=tuple(float(value) for value in raw["update_sketch"]),
            reliability=tuple(float(value) for value in raw["reliability"]),
            compatibility=tuple(float(value) for value in raw["compatibility"]),
            adapter_norm=float(raw["adapter_norm"]),
            data_count=int(raw["data_count"]),
        )

    def dimensions(self) -> tuple[int, int, int, int]:
        return (
            len(self.skill_prototype),
            len(self.update_sketch),
            len(self.reliability),
            len(self.compatibility),
        )

    def to_ranker_features(self) -> tuple[float, ...]:
        return (
            self.skill_prototype
            + self.update_sketch
            + self.reliability
            + self.compatibility
            + (float(self.adapter_norm), float(self.data_count))
        )


class AdapterBank:
    def __init__(self) -> None:
        self._descriptors: dict[AdapterId, AdapterDescriptor] = {}
        self._private_paths: dict[AdapterId, Path] = {}
        self._dimensions: tuple[int, int, int, int] | None = None

    def register(
        self,
        adapter_id: AdapterId,
        descriptor: AdapterDescriptor,
        *,
        private_path: Path,
    ) -> None:
        if adapter_id in self._descriptors:
            raise AdapterBankError(f"adapter already registered: {adapter_id}")
        if self._dimensions is None:
            self._dimensions = descriptor.dimensions()
        elif descriptor.dimensions() != self._dimensions:
            raise AdapterBankError(
                f"descriptor dimensions {descriptor.dimensions()} do not match {self._dimensions}"
            )
        self._descriptors[adapter_id] = descriptor
        self._private_paths[adapter_id] = Path(private_path)

    def query_candidates(self) -> tuple[tuple[AdapterId, AdapterDescriptor], ...]:
        return tuple(self._descriptors.items())

    def private_path(self, adapter_id: AdapterId) -> Path:
        try:
            return self._private_paths[adapter_id]
        except KeyError as error:
            raise AdapterBankError(f"unknown adapter: {adapter_id}") from error
