from hashlib import sha256
import json

import pytest
import torch
from safetensors.torch import save_file

from fcut_vla.adapters.artifact import (
    AdapterArtifactError,
    validate_adapter_checkpoint,
    validate_adapter_run,
)


def _write_adapter(tmp_path, **overrides):
    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "modules_to_save": [],
        "target_modules": ["action_in_proj", "q_proj"],
        **overrides,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    return tmp_path


def _validate(checkpoint, **overrides):
    arguments = {
        "expected_rank": 8,
        "expected_alpha": 16,
        "trainable_parameters": 4_000_000,
        "total_parameters": 450_000_000,
        "completed_step": 10,
        "expected_step": 10,
        "resolved_targets": ("model.action_in_proj", "model.expert.q_proj"),
    }
    arguments.update(overrides)
    return validate_adapter_checkpoint(checkpoint, **arguments)


def test_valid_lora_artifact_records_digest_and_identity_free_metadata(tmp_path):
    checkpoint = _write_adapter(tmp_path)

    artifact = _validate(checkpoint)
    payload = json.loads(artifact.to_json())

    assert artifact.trainable_ratio < 0.10
    assert artifact.adapter_sha256 == sha256(
        (checkpoint / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    assert payload["resolved_targets"] == ["model.action_in_proj", "model.expert.q_proj"]
    assert "client_id" not in artifact.to_json()
    assert "private_path" not in artifact.to_json()


def test_full_model_only_checkpoint_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"full-model")

    with pytest.raises(AdapterArtifactError, match="adapter_model.safetensors"):
        _validate(tmp_path)


@pytest.mark.parametrize(
    "config_override, message",
    [
        ({"peft_type": "IA3"}, "LORA"),
        ({"r": 4}, "rank"),
        ({"lora_alpha": 8}, "alpha"),
        ({"modules_to_save": ["head"]}, "modules_to_save"),
        ({"target_modules": []}, "target"),
    ],
)
def test_adapter_config_must_match_frozen_contract(tmp_path, config_override, message):
    checkpoint = _write_adapter(tmp_path, **config_override)

    with pytest.raises(AdapterArtifactError, match=message):
        _validate(checkpoint)


@pytest.mark.parametrize(
    "argument_override, message",
    [
        ({"trainable_parameters": 0}, "trainable"),
        ({"trainable_parameters": 45_000_000}, "10%"),
        ({"total_parameters": 0}, "total"),
        ({"completed_step": 9}, "step"),
        ({"resolved_targets": ()}, "target"),
    ],
)
def test_adapter_training_evidence_must_pass_completion_gates(
    tmp_path, argument_override, message
):
    checkpoint = _write_adapter(tmp_path)

    with pytest.raises(AdapterArtifactError, match=message):
        _validate(checkpoint, **argument_override)


def test_empty_adapter_weights_are_rejected(tmp_path):
    checkpoint = _write_adapter(tmp_path)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(AdapterArtifactError, match="empty"):
        _validate(checkpoint)


def test_run_validation_extracts_step_counts_and_resolved_lora_targets(tmp_path):
    output = tmp_path / "client-adapter"
    checkpoint = output / "checkpoints" / "000010" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    _write_adapter(checkpoint)
    save_file(
        {
            "base_model.model.model.expert.q_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.model.expert.q_proj.lora_B.weight": torch.ones(3, 2),
        },
        checkpoint / "adapter_model.safetensors",
    )
    training_state = checkpoint.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text('{"step": 10}')
    log = tmp_path / "stderr.log"
    log.write_text(
        "num_learnable_params=4000000 (4M)\nnum_total_params=450000000 (450M)\n"
    )

    artifact = validate_adapter_run(
        output,
        log_path=log,
        expected_rank=8,
        expected_alpha=16,
        expected_step=10,
    )

    assert artifact.completed_step == 10
    assert artifact.resolved_targets == ("base_model.model.model.expert.q_proj",)
