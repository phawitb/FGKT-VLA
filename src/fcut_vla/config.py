"""Validated, hash-stable experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .types import SeedSet


class ConfigError(ValueError):
    """Raised when a configuration violates an experimental constraint."""


FORBIDDEN_RANKER_FEATURES = frozenset(
    {"task_id", "client_id", "source_client_id", "client_ownership"}
)


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    target_modules: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LoraConfig":
        rank = int(raw["rank"])
        if rank <= 0:
            raise ConfigError("LoRA rank must be positive")
        alpha = int(raw["alpha"])
        if alpha <= 0:
            raise ConfigError("LoRA alpha must be positive")
        targets = tuple(sorted(set(str(item) for item in raw["target_modules"])))
        if not targets:
            raise ConfigError("LoRA target_modules must be non-empty")
        return cls(rank=rank, alpha=alpha, target_modules=targets)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    backbone: str
    lora: LoraConfig
    ranker_features: tuple[str, ...]
    train_seeds: SeedSet
    evaluation_seeds: SeedSet
    counterfactual_seeds: SeedSet
    personalized_repair_first: bool
    retention_gate: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        features = tuple(sorted(set(str(item) for item in raw["ranker_features"])))
        leaked = sorted(FORBIDDEN_RANKER_FEATURES.intersection(features))
        if leaked:
            raise ConfigError(f"forbidden ranker feature: {', '.join(leaked)}")

        evaluation = SeedSet.from_iterable(raw["evaluation_seeds"])
        counterfactual = SeedSet.from_iterable(raw["counterfactual_seeds"])
        if evaluation != counterfactual:
            raise ConfigError("counterfactual evaluation requires paired seed sets")

        if not bool(raw["personalized_repair_first"]):
            raise ConfigError("personalized repair must precede global consolidation")
        if not bool(raw["retention_gate"]):
            raise ConfigError("retention gate must be enabled")

        name = str(raw["experiment_name"]).strip()
        if not name:
            raise ConfigError("experiment_name must be non-empty")
        backbone = str(raw["backbone"]).strip().lower()
        if backbone != "smolvla":
            raise ConfigError("primary backbone must be smolvla")

        return cls(
            experiment_name=name,
            backbone=backbone,
            lora=LoraConfig.from_mapping(raw["lora"]),
            ranker_features=features,
            train_seeds=SeedSet.from_iterable(raw["train_seeds"]),
            evaluation_seeds=evaluation,
            counterfactual_seeds=counterfactual,
            personalized_repair_first=True,
            retention_gate=True,
        )

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return ExperimentConfig.from_mapping(raw)

