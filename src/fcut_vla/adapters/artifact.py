"""Validation and identity-free metadata for private PEFT adapter artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from safetensors import safe_open


class AdapterArtifactError(ValueError):
    """Raised when a checkpoint is not a valid private LoRA adapter."""


@dataclass(frozen=True)
class AdapterArtifact:
    schema_version: int
    peft_type: str
    rank: int
    alpha: int
    adapter_bytes: int
    adapter_sha256: str
    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float
    completed_step: int
    resolved_targets: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _load_adapter_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "adapter_config.json"
    if not path.is_file():
        raise AdapterArtifactError(f"missing adapter_config.json in {checkpoint}")
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise AdapterArtifactError("adapter_config.json is not valid JSON") from error
    if not isinstance(config, dict):
        raise AdapterArtifactError("adapter_config.json must contain an object")
    return config


def validate_adapter_checkpoint(
    checkpoint: Path,
    *,
    expected_rank: int,
    expected_alpha: int,
    trainable_parameters: int,
    total_parameters: int,
    completed_step: int,
    expected_step: int,
    resolved_targets: tuple[str, ...],
) -> AdapterArtifact:
    checkpoint = Path(checkpoint)
    weights = checkpoint / "adapter_model.safetensors"
    if not weights.is_file():
        raise AdapterArtifactError(f"missing adapter_model.safetensors in {checkpoint}")
    adapter_bytes = weights.stat().st_size
    if adapter_bytes == 0:
        raise AdapterArtifactError("adapter weights must not be empty")

    config = _load_adapter_config(checkpoint)
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise AdapterArtifactError("adapter PEFT type must be LORA")
    if int(config.get("r", 0)) != expected_rank:
        raise AdapterArtifactError("adapter rank does not match frozen configuration")
    if int(config.get("lora_alpha", 0)) != expected_alpha:
        raise AdapterArtifactError("adapter alpha does not match frozen configuration")
    modules_to_save = config.get("modules_to_save")
    if modules_to_save not in (None, []):
        raise AdapterArtifactError("modules_to_save must be empty for private adapters")
    configured_targets = config.get("target_modules")
    if not configured_targets:
        raise AdapterArtifactError("adapter target modules must be non-empty")

    if not isinstance(trainable_parameters, int) or trainable_parameters <= 0:
        raise AdapterArtifactError("trainable parameter count must be positive")
    if not isinstance(total_parameters, int) or total_parameters <= 0:
        raise AdapterArtifactError("total parameter count must be positive")
    ratio = trainable_parameters / total_parameters
    if not math.isfinite(ratio) or ratio >= 0.10:
        raise AdapterArtifactError("trainable parameters must be below 10% of total")
    if completed_step != expected_step:
        raise AdapterArtifactError("completed step does not match requested step")

    targets = tuple(sorted(set(str(target).strip() for target in resolved_targets)))
    if not targets or any(not target for target in targets):
        raise AdapterArtifactError("resolved adapter targets must be non-empty")

    return AdapterArtifact(
        schema_version=1,
        peft_type="LORA",
        rank=expected_rank,
        alpha=expected_alpha,
        adapter_bytes=adapter_bytes,
        adapter_sha256=sha256(weights.read_bytes()).hexdigest(),
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_ratio=ratio,
        completed_step=completed_step,
        resolved_targets=targets,
    )


def _latest_numeric_checkpoint(output_dir: Path) -> Path:
    candidates = [
        path
        for path in (Path(output_dir) / "checkpoints").iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not candidates:
        raise AdapterArtifactError(f"no numeric checkpoint found under {output_dir}")
    return max(candidates, key=lambda path: int(path.name))


def _parameter_counts(log_path: Path) -> tuple[int, int]:
    text = Path(log_path).read_text()
    trainable_matches = re.findall(r"num_learnable_params=(\d+)", text)
    total_matches = re.findall(r"num_total_params=(\d+)", text)
    if not trainable_matches or not total_matches:
        raise AdapterArtifactError("training log does not contain parameter counts")
    return int(trainable_matches[-1]), int(total_matches[-1])


def _resolved_lora_targets(weights_path: Path) -> tuple[str, ...]:
    targets: set[str] = set()
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                match = re.match(r"(.+)\.lora_[AB](?:\.default)?\.weight$", key)
                if match:
                    targets.add(match.group(1))
    except Exception as error:
        raise AdapterArtifactError("adapter_model.safetensors is not readable") from error
    if not targets:
        raise AdapterArtifactError("adapter weights contain no resolved LoRA targets")
    return tuple(sorted(targets))


def validate_adapter_run(
    output_dir: Path,
    *,
    log_path: Path,
    expected_rank: int,
    expected_alpha: int,
    expected_step: int,
) -> AdapterArtifact:
    checkpoint_root = _latest_numeric_checkpoint(Path(output_dir))
    checkpoint = checkpoint_root / "pretrained_model"
    step_path = checkpoint_root / "training_state" / "training_step.json"
    if not step_path.is_file():
        raise AdapterArtifactError("checkpoint is missing training_step.json")
    try:
        completed_step = int(json.loads(step_path.read_text())["step"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise AdapterArtifactError("training_step.json is malformed") from error
    trainable, total = _parameter_counts(log_path)
    targets = _resolved_lora_targets(checkpoint / "adapter_model.safetensors")
    return validate_adapter_checkpoint(
        checkpoint,
        expected_rank=expected_rank,
        expected_alpha=expected_alpha,
        trainable_parameters=trainable,
        total_parameters=total,
        completed_step=completed_step,
        expected_step=expected_step,
        resolved_targets=targets,
    )
