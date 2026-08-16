"""Small immutable identifiers shared across the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class ClientStageId:
    client: str
    stage: int

    def __post_init__(self) -> None:
        if not self.client:
            raise ValueError("client must be non-empty")
        if self.stage < 1:
            raise ValueError("stage must be at least 1")

    def __str__(self) -> str:
        return f"{self.client}:stage-{self.stage}"


@dataclass(frozen=True, order=True)
class AdapterId:
    client: str
    stage: int
    digest: str

    def __post_init__(self) -> None:
        ClientStageId(self.client, self.stage)
        if not self.digest:
            raise ValueError("digest must be non-empty")

    def __str__(self) -> str:
        return f"{self.client}:stage-{self.stage}:{self.digest}"


@dataclass(frozen=True, order=True)
class FailureId:
    episode: str
    step: int

    def __post_init__(self) -> None:
        if not self.episode:
            raise ValueError("episode must be non-empty")
        if self.step < 0:
            raise ValueError("step must be non-negative")

    def __str__(self) -> str:
        return f"{self.episode}:step-{self.step}"


@dataclass(frozen=True)
class SeedSet:
    values: tuple[int, ...]

    @classmethod
    def from_iterable(cls, values: Iterable[int]) -> "SeedSet":
        normalized = tuple(sorted(set(int(value) for value in values)))
        if not normalized:
            raise ValueError("seed set requires at least one value")
        if any(value < 0 for value in normalized):
            raise ValueError("seeds must be non-negative")
        return cls(normalized)

