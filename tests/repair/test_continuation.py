from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from fcut_vla.repair.continuation import (
    ContinuationError,
    ContinuationHyperparameters,
    DatasetEpisode,
    build_continuation_recipe,
    resolve_task_episode_indices,
    validate_continuation_recipe,
)


def _arguments():
    return {
        "source_run_hash": "source-run-sha",
        "source_adapter_sha256": "source-adapter-sha",
        "source_adapter_config_sha256": "adapter-config-sha",
        "source_episode_sha256": "source-episode-sha",
        "failure_hash": "failure-sha",
        "task_alias": "libero_spatial.task0",
        "problem_id": "pick_up_black_bowl",
        "instruction": "Put the black bowl on the plate",
        "dataset_repo": "lerobot/libero_spatial_image",
        "dataset_revision": "dataset-sha",
        "episode_indices": (8, 3),
        "base_policy_repo": "lerobot/smolvla_base",
        "base_policy_revision": "base-sha",
        "lerobot_commit": "lerobot-sha",
        "peft_version": "0.18.1",
        "hf_libero_version": "0.1.4",
        "python_version": "3.12.13",
        "torch_version": "2.11.0+cu130",
        "device": "cuda",
        "tokenizer_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "tokenizer_revision": "tokenizer-sha",
        "hyperparameters": ContinuationHyperparameters(
            steps=20,
            batch_size=2,
            learning_rate=1e-4,
            training_seed=2026,
        ),
        "source_target_modules": ("model.layers.1.q_proj", "model.layers.0.q_proj"),
        "source_policy_config_sha256": "policy-config-sha",
        "source_preprocessor_sha256": "preprocessor-sha",
        "source_postprocessor_sha256": "postprocessor-sha",
        "source_processor_artifacts": (
            ("policy_postprocessor.json", sha256(b"post").hexdigest()),
            ("policy_preprocessor.json", sha256(b"pre").hexdigest()),
        ),
    }


def _rehash(recipe):
    payload = recipe.to_mapping()
    payload.pop("content_hash")
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return replace(recipe, content_hash=digest)


def test_task_episode_resolution_is_exact_sorted_and_nonempty():
    episodes = (
        DatasetEpisode(8, ("put the bowl on the plate",)),
        DatasetEpisode(3, ("  Put  the BOWL on the plate ",)),
        DatasetEpisode(4, ("open the drawer",)),
    )

    assert resolve_task_episode_indices(
        episodes, "put the bowl on the plate"
    ) == (3, 8)


@pytest.mark.parametrize(
    "episodes",
    [(), (DatasetEpisode(0, ("other instruction",)),)],
)
def test_task_episode_resolution_rejects_no_match(episodes):
    with pytest.raises(ContinuationError, match="no dataset episodes"):
        resolve_task_episode_indices(episodes, "target instruction")


def test_task_episode_resolution_rejects_invalid_or_conflicting_metadata():
    with pytest.raises(ContinuationError, match="non-negative"):
        resolve_task_episode_indices(
            (DatasetEpisode(-1, ("target",)),), "target"
        )
    with pytest.raises(ContinuationError, match="conflicting"):
        resolve_task_episode_indices(
            (
                DatasetEpisode(2, ("target",)),
                DatasetEpisode(2, ("different",)),
            ),
            "target",
        )


def test_recipe_hash_binds_episode_selection_and_is_canonical():
    recipe = build_continuation_recipe(**_arguments())
    reordered = build_continuation_recipe(
        **{**_arguments(), "episode_indices": (3, 8)}
    )
    changed = build_continuation_recipe(
        **{**_arguments(), "episode_indices": (3, 9)}
    )

    assert recipe.episode_indices == (3, 8)
    assert recipe.source_target_modules == (
        "model.layers.0.q_proj",
        "model.layers.1.q_proj",
    )
    assert validate_continuation_recipe(recipe) == recipe
    assert reordered.content_hash == recipe.content_hash
    assert changed.content_hash != recipe.content_hash
    assert build_continuation_recipe(
        **{**_arguments(), "device": "mps"}
    ).content_hash != recipe.content_hash
    assert build_continuation_recipe(
        **{**_arguments(), "source_policy_config_sha256": "changed-config-sha"}
    ).content_hash != recipe.content_hash


def test_recipe_rejects_mutated_payload_with_old_hash():
    recipe = build_continuation_recipe(**_arguments())

    with pytest.raises(ContinuationError, match="content hash"):
        validate_continuation_recipe(
            replace(
                recipe,
                hyperparameters=replace(recipe.hyperparameters, training_seed=999),
            )
        )


@pytest.mark.parametrize(
    "change,message",
    [
        ({"episode_indices": ()}, "episode indices"),
        (
            {"hyperparameters": ContinuationHyperparameters(0, 2, 1e-4, 1)},
            "steps",
        ),
        ({"dataset_revision": ""}, "dataset revision"),
        ({"source_target_modules": ()}, "target modules"),
    ],
)
def test_recipe_rejects_incomplete_or_invalid_contract(change, message):
    with pytest.raises(ContinuationError, match=message):
        build_continuation_recipe(**{**_arguments(), **change})


@pytest.mark.parametrize("indices", [(True,), (2.9,)])
def test_recipe_rejects_lossy_or_boolean_episode_indices(indices):
    with pytest.raises(ContinuationError, match="exact integers"):
        build_continuation_recipe(**{**_arguments(), "episode_indices": indices})


@pytest.mark.parametrize(
    "hyperparameters",
    [
        ContinuationHyperparameters(True, 2, 1e-4, 1),
        ContinuationHyperparameters(20, 1.5, 1e-4, 1),
        ContinuationHyperparameters(20, 2, True, 1),
        ContinuationHyperparameters(20, 2, float("inf"), 1),
        ContinuationHyperparameters(20, 2, 1e-4, False),
    ],
)
def test_recipe_rejects_wrong_hyperparameter_numeric_types(hyperparameters):
    with pytest.raises(ContinuationError, match="hyperparameter"):
        build_continuation_recipe(
            **{**_arguments(), "hyperparameters": hyperparameters}
        )


@pytest.mark.parametrize("targets", [("",), ("q_proj", 3)])
def test_recipe_rejects_empty_or_non_string_target_modules(targets):
    with pytest.raises(ContinuationError, match="target modules"):
        build_continuation_recipe(
            **{**_arguments(), "source_target_modules": targets}
        )


def test_episode_resolver_rejects_boolean_index():
    with pytest.raises(ContinuationError, match="exact integers"):
        resolve_task_episode_indices((DatasetEpisode(True, ("target",)),), "target")


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_run_hash", None),
        ("instruction", Path("private/failure.txt")),
        ("dataset_revision", 123),
    ],
)
def test_recipe_rejects_non_string_textual_identities(field, value):
    with pytest.raises(ContinuationError, match="must be a string"):
        build_continuation_recipe(**{**_arguments(), field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"instruction": "  Put   the black bowl on the plate "},
        {"dataset_revision": " dataset-sha "},
        {"source_target_modules": (" model.layers.0.q_proj",)},
    ],
)
def test_validator_rejects_hash_consistent_noncanonical_text(change):
    malformed = _rehash(replace(build_continuation_recipe(**_arguments()), **change))

    with pytest.raises(ContinuationError, match="canonical"):
        validate_continuation_recipe(malformed)
