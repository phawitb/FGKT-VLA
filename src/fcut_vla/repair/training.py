"""Render and verify failure-aligned continuation training plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from fcut_vla.adapters.artifact import AdapterArtifactError, validate_adapter_run

from fcut_vla.repair.continuation import (
    ContinuationRecipe,
    validate_continuation_recipe,
)


class RepairTrainingError(ValueError):
    """Raised when continuation training inputs or outputs are invalid."""


def _exact_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RepairTrainingError(f"{label} must be a positive exact integer")
    return value


def validate_training_runtime(
    *,
    expected_lerobot_commit: str,
    expected_peft_version: str,
    actual_lerobot_commit: str,
    actual_peft_version: str,
    lerobot_dirty: bool,
) -> dict[str, str]:
    if lerobot_dirty:
        raise RepairTrainingError("LeRobot checkout is dirty")
    expected = {
        "lerobot_commit": expected_lerobot_commit,
        "peft_version": expected_peft_version,
    }
    actual = {
        "lerobot_commit": actual_lerobot_commit,
        "peft_version": actual_peft_version,
    }
    if actual != expected:
        raise RepairTrainingError(
            f"repair training runtime provenance mismatch: expected {expected}, got {actual}"
        )
    return actual


def validate_artifact_separation(source_run_dir: Path, candidate_run_dir: Path) -> None:
    source = Path(source_run_dir).resolve()
    candidate = Path(candidate_run_dir).resolve()
    if source == candidate or source in candidate.parents or candidate in source.parents:
        raise RepairTrainingError("source and candidate artifact trees must not overlap")


def _processor_artifacts(pretrained: Path) -> tuple[tuple[str, str], ...]:
    paths = sorted(
        path
        for path in Path(pretrained).iterdir()
        if path.is_file()
        and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
    )
    if not paths:
        raise RepairTrainingError("processor artifact set is empty")
    return tuple((path.name, sha256(path.read_bytes()).hexdigest()) for path in paths)


def _processor_manifest_sha256(artifacts: tuple[tuple[str, str], ...]) -> str:
    return sha256(json.dumps(artifacts, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ContinuationTrainingPlan:
    recipe: ContinuationRecipe
    source_checkpoint_dir: Path
    source_pretrained_dir: Path
    output_dir: Path
    save_frequency: int
    python_executable: Path

    def __post_init__(self) -> None:
        validate_continuation_recipe(self.recipe)
        checkpoint = Path(self.source_checkpoint_dir)
        pretrained = Path(self.source_pretrained_dir)
        output = Path(self.output_dir)
        executable = Path(self.python_executable)
        _exact_positive_int(self.save_frequency, "save frequency")
        if not checkpoint.is_dir():
            raise RepairTrainingError("source checkpoint directory does not exist")
        expected_pretrained = checkpoint / "pretrained_model"
        required = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        if pretrained.resolve() != expected_pretrained.resolve() or any(
            not (pretrained / name).is_file() for name in required
        ):
            raise RepairTrainingError("source pretrained adapter is missing or inconsistent")
        digest_contract = {
            "adapter_config.json": self.recipe.source_adapter_config_sha256,
            "config.json": self.recipe.source_policy_config_sha256,
            "policy_preprocessor.json": self.recipe.source_preprocessor_sha256,
            "policy_postprocessor.json": self.recipe.source_postprocessor_sha256,
        }
        for name, expected_digest in digest_contract.items():
            actual_digest = sha256((pretrained / name).read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                raise RepairTrainingError(f"source {name} digest does not match recipe")
        if _processor_artifacts(pretrained) != self.recipe.source_processor_artifacts:
            raise RepairTrainingError("source processor artifact manifest does not match recipe")
        try:
            output.resolve().relative_to(checkpoint.resolve())
        except ValueError:
            pass
        else:
            raise RepairTrainingError("candidate output must be outside source checkpoint")
        if not executable.is_file():
            raise RepairTrainingError("Python executable does not exist")

@dataclass(frozen=True)
class RepairCandidateReport:
    schema_version: int
    recipe_sha256: str
    source_adapter_sha256: str
    candidate_adapter_sha256: str
    adapter_bytes: int
    rank: int
    alpha: int
    trainable_parameters: int
    total_parameters: int
    completed_step: int
    resolved_targets: tuple[str, ...]
    adapter_config_sha256: str
    policy_config_sha256: str
    preprocessor_sha256: str
    postprocessor_sha256: str
    processor_manifest_sha256: str


def render_continuation_command(plan: ContinuationTrainingPlan) -> tuple[str, ...]:
    episodes = json.dumps(list(plan.recipe.episode_indices), separators=(",", ":"))
    hyperparameters = plan.recipe.hyperparameters
    return (
        str(plan.python_executable),
        "-m",
        "fcut_vla.repair.pinned_train",
        f"--fgkt-tokenizer-repo={plan.recipe.tokenizer_repo}",
        f"--fgkt-tokenizer-revision={plan.recipe.tokenizer_revision}",
        "--",
        f"--policy.path={plan.source_pretrained_dir}",
        f"--dataset.repo_id={plan.recipe.dataset_repo}",
        f"--dataset.revision={plan.recipe.dataset_revision}",
        f"--dataset.episodes={episodes}",
        f"--steps={hyperparameters.steps}",
        f"--batch_size={hyperparameters.batch_size}",
        f"--policy.optimizer_lr={hyperparameters.learning_rate}",
        f"--seed={hyperparameters.training_seed}",
        f"--save_freq={plan.save_frequency}",
        "--env_eval_freq=0",
        "--policy.push_to_hub=false",
        f"--policy.device={plan.recipe.device}",
        f"--output_dir={plan.output_dir}",
    )


def _source_int(evidence: Mapping[str, Any], key: str) -> int:
    value = evidence.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RepairTrainingError(f"source evidence {key} must be a positive exact integer")
    return value


def verify_repair_candidate(
    *,
    plan: ContinuationTrainingPlan,
    source_evidence: Mapping[str, Any],
    candidate_log: Path,
) -> RepairCandidateReport:
    source_digest = source_evidence.get("adapter_sha256")
    if not isinstance(source_digest, str) or source_digest != plan.recipe.source_adapter_sha256:
        raise RepairTrainingError("source adapter digest does not match continuation recipe")
    source_targets_raw = source_evidence.get("resolved_targets")
    if not isinstance(source_targets_raw, list) or any(
        not isinstance(target, str) or not target for target in source_targets_raw
    ):
        raise RepairTrainingError("source target contract is invalid")
    source_targets = tuple(source_targets_raw)
    if source_targets != plan.recipe.source_target_modules:
        raise RepairTrainingError("source target contract does not match continuation recipe")
    rank = _source_int(source_evidence, "rank")
    alpha = _source_int(source_evidence, "alpha")
    source_trainable = _source_int(source_evidence, "trainable_parameters")
    source_total = _source_int(source_evidence, "total_parameters")
    try:
        artifact = validate_adapter_run(
            plan.output_dir,
            log_path=Path(candidate_log),
            expected_rank=rank,
            expected_alpha=alpha,
            expected_step=plan.recipe.hyperparameters.steps,
        )
    except (AdapterArtifactError, FileNotFoundError) as error:
        raise RepairTrainingError(f"candidate adapter validation failed: {error}") from error
    if artifact.adapter_sha256 == source_digest:
        raise RepairTrainingError("repair candidate must have a distinct adapter digest")
    if artifact.resolved_targets != source_targets:
        raise RepairTrainingError("candidate LoRA target contract differs from source")
    if (
        artifact.trainable_parameters != source_trainable
        or artifact.total_parameters != source_total
    ):
        raise RepairTrainingError("candidate parameter contract differs from source")
    checkpoint = (
        plan.output_dir
        / "checkpoints"
        / f"{plan.recipe.hyperparameters.steps:06d}"
        / "pretrained_model"
    )
    config_path = checkpoint / "adapter_config.json"
    try:
        adapter_config = json.loads(config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RepairTrainingError("candidate adapter config is missing or invalid") from error
    if adapter_config.get("base_model_name_or_path") != plan.recipe.base_policy_repo:
        raise RepairTrainingError("candidate base policy differs from continuation recipe")
    if adapter_config.get("revision") != plan.recipe.base_policy_revision:
        raise RepairTrainingError("candidate base revision differs from continuation recipe")
    source_config_path = plan.source_pretrained_dir / "adapter_config.json"
    try:
        source_config = json.loads(source_config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RepairTrainingError("source adapter config is missing or invalid") from error
    if adapter_config.get("target_modules") != source_config.get("target_modules"):
        raise RepairTrainingError("candidate target configuration differs from source")
    policy_config_path = checkpoint / "config.json"
    preprocessor_path = checkpoint / "policy_preprocessor.json"
    postprocessor_path = checkpoint / "policy_postprocessor.json"
    try:
        policy_config = json.loads(policy_config_path.read_text())
        preprocessor_digest = sha256(preprocessor_path.read_bytes()).hexdigest()
        postprocessor_digest = sha256(postprocessor_path.read_bytes()).hexdigest()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RepairTrainingError("candidate policy/processor config is missing or invalid") from error
    if policy_config.get("optimizer_lr") != plan.recipe.hyperparameters.learning_rate:
        raise RepairTrainingError("candidate policy learning rate differs from recipe")
    if preprocessor_digest != plan.recipe.source_preprocessor_sha256:
        raise RepairTrainingError("candidate preprocessor contract differs from source")
    if postprocessor_digest != plan.recipe.source_postprocessor_sha256:
        raise RepairTrainingError("candidate postprocessor contract differs from source")
    candidate_processor_artifacts = _processor_artifacts(checkpoint)
    if candidate_processor_artifacts != plan.recipe.source_processor_artifacts:
        raise RepairTrainingError("candidate processor artifact manifest differs from source")
    return RepairCandidateReport(
        schema_version=1,
        recipe_sha256=plan.recipe.content_hash,
        source_adapter_sha256=source_digest,
        candidate_adapter_sha256=artifact.adapter_sha256,
        adapter_bytes=artifact.adapter_bytes,
        rank=artifact.rank,
        alpha=artifact.alpha,
        trainable_parameters=artifact.trainable_parameters,
        total_parameters=artifact.total_parameters,
        completed_step=artifact.completed_step,
        resolved_targets=artifact.resolved_targets,
        adapter_config_sha256=sha256(config_path.read_bytes()).hexdigest(),
        policy_config_sha256=sha256(policy_config_path.read_bytes()).hexdigest(),
        preprocessor_sha256=preprocessor_digest,
        postprocessor_sha256=postprocessor_digest,
        processor_manifest_sha256=_processor_manifest_sha256(
            candidate_processor_artifacts
        ),
    )
