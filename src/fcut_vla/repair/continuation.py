"""Immutable recipes for task-aligned continuation of a failed adapter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any

from fcut_vla.libero.task_resolution import normalize_instruction


class ContinuationError(ValueError):
    """Raised when continuation inputs or provenance are invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _required(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ContinuationError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ContinuationError(f"{label} must be non-empty")
    return normalized


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, order=True)
class DatasetEpisode:
    index: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class ContinuationHyperparameters:
    steps: int
    batch_size: int
    learning_rate: float
    training_seed: int


@dataclass(frozen=True)
class ContinuationRecipe:
    schema_version: int
    recipe_type: str
    source_run_hash: str
    source_adapter_sha256: str
    source_adapter_config_sha256: str
    source_policy_config_sha256: str
    source_preprocessor_sha256: str
    source_postprocessor_sha256: str
    source_processor_artifacts: tuple[tuple[str, str], ...]
    source_episode_sha256: str
    failure_hash: str
    task_alias: str
    problem_id: str
    instruction: str
    dataset_repo: str
    dataset_revision: str
    episode_indices: tuple[int, ...]
    episode_selection_sha256: str
    base_policy_repo: str
    base_policy_revision: str
    lerobot_commit: str
    peft_version: str
    hf_libero_version: str
    python_version: str
    torch_version: str
    device: str
    tokenizer_repo: str
    tokenizer_revision: str
    hyperparameters: ContinuationHyperparameters
    source_target_modules: tuple[str, ...]
    content_hash: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical(self.to_mapping()).decode()


def resolve_task_episode_indices(
    episodes: Iterable[DatasetEpisode], instruction: str
) -> tuple[int, ...]:
    target = normalize_instruction(instruction)
    if not target:
        raise ContinuationError("failure instruction must be non-empty")
    by_index: dict[int, tuple[str, ...]] = {}
    for episode in episodes:
        if not _is_exact_int(episode.index):
            raise ContinuationError("dataset episode indices must be exact integers")
        if episode.index < 0:
            raise ContinuationError("dataset episode indices must be non-negative")
        normalized_tasks = tuple(
            sorted({normalize_instruction(task) for task in episode.tasks})
        )
        if not normalized_tasks or any(not task for task in normalized_tasks):
            raise ContinuationError("dataset episode tasks must be non-empty")
        previous = by_index.setdefault(episode.index, normalized_tasks)
        if previous != normalized_tasks:
            raise ContinuationError("dataset episode metadata contains conflicting indices")
    matched = tuple(
        sorted(index for index, tasks in by_index.items() if target in tasks)
    )
    if not matched:
        raise ContinuationError("no dataset episodes match the failure instruction")
    return matched


def _payload(recipe: ContinuationRecipe) -> dict[str, Any]:
    payload = recipe.to_mapping()
    payload.pop("content_hash")
    return payload


def _validate_semantics(recipe: ContinuationRecipe) -> None:
    if recipe.schema_version != 1:
        raise ContinuationError("continuation recipe schema version must be 1")
    if recipe.recipe_type != "failure_aligned_continuation_v1":
        raise ContinuationError("continuation recipe type is unsupported")
    labels = {
        "source run hash": recipe.source_run_hash,
        "source adapter digest": recipe.source_adapter_sha256,
        "source adapter config digest": recipe.source_adapter_config_sha256,
        "source policy config digest": recipe.source_policy_config_sha256,
        "source preprocessor digest": recipe.source_preprocessor_sha256,
        "source postprocessor digest": recipe.source_postprocessor_sha256,
        "source episode digest": recipe.source_episode_sha256,
        "failure hash": recipe.failure_hash,
        "task alias": recipe.task_alias,
        "problem id": recipe.problem_id,
        "instruction": recipe.instruction,
        "dataset repository": recipe.dataset_repo,
        "dataset revision": recipe.dataset_revision,
        "base policy repository": recipe.base_policy_repo,
        "base policy revision": recipe.base_policy_revision,
        "LeRobot commit": recipe.lerobot_commit,
        "PEFT version": recipe.peft_version,
        "hf-libero version": recipe.hf_libero_version,
        "Python version": recipe.python_version,
        "PyTorch version": recipe.torch_version,
        "device": recipe.device,
        "tokenizer repository": recipe.tokenizer_repo,
        "tokenizer revision": recipe.tokenizer_revision,
    }
    for label, value in labels.items():
        canonical = _required(value, label)
        if label == "instruction":
            canonical = " ".join(canonical.split())
        if value != canonical:
            raise ContinuationError(f"{label} is not in canonical form")
    if recipe.device not in {"cpu", "cuda", "mps"}:
        raise ContinuationError("continuation device is unsupported")
    if (
        not recipe.source_processor_artifacts
        or recipe.source_processor_artifacts
        != tuple(sorted(set(recipe.source_processor_artifacts)))
        or any(
            not isinstance(name, str)
            or not name
            or not name.startswith(("policy_preprocessor", "policy_postprocessor"))
            or "/" in name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in recipe.source_processor_artifacts
        )
    ):
        raise ContinuationError("source processor artifact manifest is invalid")
    if (
        not recipe.episode_indices
        or any(not _is_exact_int(index) for index in recipe.episode_indices)
        or recipe.episode_indices != tuple(sorted(set(recipe.episode_indices)))
        or any(index < 0 for index in recipe.episode_indices)
    ):
        raise ContinuationError("episode indices must be non-empty, unique, and sorted")
    if (
        not recipe.source_target_modules
        or any(
            not isinstance(module, str) or not module.strip()
            for module in recipe.source_target_modules
        )
        or any(module != module.strip() for module in recipe.source_target_modules)
        or recipe.source_target_modules != tuple(sorted(set(recipe.source_target_modules)))
    ):
        raise ContinuationError(
            "source target modules must be non-empty and in canonical unique sorted form"
        )
    hyperparameters = recipe.hyperparameters
    if not isinstance(hyperparameters, ContinuationHyperparameters):
        raise ContinuationError("continuation hyperparameters have an invalid type")
    if not _is_exact_int(hyperparameters.steps):
        raise ContinuationError("steps hyperparameter must be an exact integer")
    if hyperparameters.steps < 1:
        raise ContinuationError("continuation steps hyperparameter must be positive")
    if not _is_exact_int(hyperparameters.batch_size):
        raise ContinuationError("batch-size hyperparameter must be an exact integer")
    if hyperparameters.batch_size < 1:
        raise ContinuationError("continuation batch-size hyperparameter must be positive")
    if (
        not isinstance(hyperparameters.learning_rate, float)
        or isinstance(hyperparameters.learning_rate, bool)
        or not math.isfinite(hyperparameters.learning_rate)
        or hyperparameters.learning_rate <= 0
    ):
        raise ContinuationError(
            "learning-rate hyperparameter must be a positive finite float"
        )
    if not _is_exact_int(hyperparameters.training_seed):
        raise ContinuationError("training-seed hyperparameter must be an exact integer")
    if hyperparameters.training_seed < 0:
        raise ContinuationError("continuation training-seed hyperparameter must be non-negative")
    selection_hash = sha256(_canonical(list(recipe.episode_indices))).hexdigest()
    if recipe.episode_selection_sha256 != selection_hash:
        raise ContinuationError("episode selection hash is inconsistent")


def validate_continuation_recipe(recipe: ContinuationRecipe) -> ContinuationRecipe:
    claimed_hash = recipe.content_hash
    computed_hash = sha256(_canonical(_payload(recipe))).hexdigest()
    if claimed_hash != computed_hash:
        raise ContinuationError("continuation recipe content hash is invalid")
    _validate_semantics(recipe)
    return recipe


def build_continuation_recipe(
    *,
    source_run_hash: str,
    source_adapter_sha256: str,
    source_adapter_config_sha256: str,
    source_policy_config_sha256: str,
    source_preprocessor_sha256: str,
    source_postprocessor_sha256: str,
    source_processor_artifacts: Iterable[tuple[str, str]],
    source_episode_sha256: str,
    failure_hash: str,
    task_alias: str,
    problem_id: str,
    instruction: str,
    dataset_repo: str,
    dataset_revision: str,
    episode_indices: Iterable[int],
    base_policy_repo: str,
    base_policy_revision: str,
    lerobot_commit: str,
    peft_version: str,
    hf_libero_version: str,
    python_version: str,
    torch_version: str,
    device: str,
    tokenizer_repo: str,
    tokenizer_revision: str,
    hyperparameters: ContinuationHyperparameters,
    source_target_modules: Iterable[str],
) -> ContinuationRecipe:
    raw_indices = tuple(episode_indices)
    if any(not _is_exact_int(index) for index in raw_indices):
        raise ContinuationError("episode indices must be exact integers")
    indices = tuple(sorted(set(raw_indices)))
    raw_target_modules = tuple(source_target_modules)
    if any(not isinstance(module, str) for module in raw_target_modules):
        raise ContinuationError("source target modules must contain only strings")
    target_modules = tuple(sorted(set(module.strip() for module in raw_target_modules)))
    processor_artifacts = tuple(sorted(source_processor_artifacts))
    recipe = ContinuationRecipe(
        schema_version=1,
        recipe_type="failure_aligned_continuation_v1",
        source_run_hash=_required(source_run_hash, "source run hash"),
        source_adapter_sha256=_required(source_adapter_sha256, "source adapter digest"),
        source_adapter_config_sha256=_required(
            source_adapter_config_sha256, "source adapter config digest"
        ),
        source_policy_config_sha256=_required(
            source_policy_config_sha256, "source policy config digest"
        ),
        source_preprocessor_sha256=_required(
            source_preprocessor_sha256, "source preprocessor digest"
        ),
        source_postprocessor_sha256=_required(
            source_postprocessor_sha256, "source postprocessor digest"
        ),
        source_processor_artifacts=processor_artifacts,
        source_episode_sha256=_required(source_episode_sha256, "source episode digest"),
        failure_hash=_required(failure_hash, "failure hash"),
        task_alias=_required(task_alias, "task alias"),
        problem_id=_required(problem_id, "problem id"),
        instruction=" ".join(_required(instruction, "instruction").split()),
        dataset_repo=_required(dataset_repo, "dataset repository"),
        dataset_revision=_required(dataset_revision, "dataset revision"),
        episode_indices=indices,
        episode_selection_sha256=sha256(_canonical(list(indices))).hexdigest(),
        base_policy_repo=_required(base_policy_repo, "base policy repository"),
        base_policy_revision=_required(base_policy_revision, "base policy revision"),
        lerobot_commit=_required(lerobot_commit, "LeRobot commit"),
        peft_version=_required(peft_version, "PEFT version"),
        hf_libero_version=_required(hf_libero_version, "hf-libero version"),
        python_version=_required(python_version, "Python version"),
        torch_version=_required(torch_version, "PyTorch version"),
        device=_required(device, "device"),
        tokenizer_repo=_required(tokenizer_repo, "tokenizer repository"),
        tokenizer_revision=_required(tokenizer_revision, "tokenizer revision"),
        hyperparameters=hyperparameters,
        source_target_modules=target_modules,
        content_hash="",
    )
    _validate_semantics(recipe)
    return replace(
        recipe, content_hash=sha256(_canonical(_payload(recipe))).hexdigest()
    )
