from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fcut_vla.config import ConfigError, ExperimentConfig, load_config
from fcut_vla.types import AdapterId, ClientStageId, FailureId, SeedSet


def valid_mapping() -> dict:
    return {
        "experiment_name": "smoke",
        "backbone": "smolvla",
        "lora": {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        "ranker_features": ["failure_context", "skill_prototype", "update_sketch"],
        "train_seeds": [11, 22],
        "evaluation_seeds": [101, 102, 103],
        "counterfactual_seeds": [101, 102, 103],
        "personalized_repair_first": True,
        "retention_gate": True,
    }


def test_valid_configuration_is_frozen_and_hash_is_stable():
    first = ExperimentConfig.from_mapping(valid_mapping())
    second = ExperimentConfig.from_mapping(valid_mapping())
    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64
    with pytest.raises(FrozenInstanceError):
        first.backbone = "other"


@pytest.mark.parametrize("feature", ["task_id", "client_id", "source_client_id"])
def test_rejects_identity_leaking_ranker_features(feature: str):
    raw = valid_mapping()
    raw["ranker_features"] = ["failure_context", feature]
    with pytest.raises(ConfigError, match="forbidden ranker feature"):
        ExperimentConfig.from_mapping(raw)


def test_rejects_unpaired_counterfactual_seed_sets():
    raw = valid_mapping()
    raw["counterfactual_seeds"] = [101, 999]
    with pytest.raises(ConfigError, match="paired"):
        ExperimentConfig.from_mapping(raw)


@pytest.mark.parametrize("rank", [0, -1])
def test_rejects_non_positive_lora_rank(rank: int):
    raw = valid_mapping()
    raw["lora"]["rank"] = rank
    with pytest.raises(ConfigError, match="LoRA rank"):
        ExperimentConfig.from_mapping(raw)


def test_requires_personalized_repair_and_retention_gate():
    raw = valid_mapping()
    raw["personalized_repair_first"] = False
    with pytest.raises(ConfigError, match="personalized"):
        ExperimentConfig.from_mapping(raw)

    raw = valid_mapping()
    raw["retention_gate"] = False
    with pytest.raises(ConfigError, match="retention"):
        ExperimentConfig.from_mapping(raw)


def test_load_config_reads_yaml(tmp_path: Path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
experiment_name: smoke
backbone: smolvla
lora:
  rank: 8
  alpha: 16
  target_modules: [q_proj, v_proj]
ranker_features: [failure_context, skill_prototype, update_sketch]
train_seeds: [11, 22]
evaluation_seeds: [101, 102, 103]
counterfactual_seeds: [101, 102, 103]
personalized_repair_first: true
retention_gate: true
""".strip()
    )
    config = load_config(path)
    assert config.experiment_name == "smoke"
    assert config.lora.rank == 8


def test_identifier_types_are_stable_and_seed_set_is_nonempty():
    assert str(ClientStageId("client-1", 2)) == "client-1:stage-2"
    assert str(AdapterId("client-1", 2, "abc")) == "client-1:stage-2:abc"
    assert str(FailureId("episode-7", 41)) == "episode-7:step-41"
    assert SeedSet.from_iterable([3, 1, 3]).values == (1, 3)
    with pytest.raises(ValueError, match="at least one"):
        SeedSet.from_iterable([])
