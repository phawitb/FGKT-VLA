from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fcut_vla.benchmark.manifest import (
    FailureStratum,
    ManifestError,
    load_manifest,
    validate_no_leakage,
)


MANIFEST_PATH = Path("manifests/fedlibero_fail_v1.yaml")


def raw_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def test_manifest_has_five_clients_and_four_ordered_stages_each():
    manifest = load_manifest(MANIFEST_PATH)
    by_client = manifest.assignments_by_client()
    assert set(by_client) == {"c1", "c2", "c3", "c4", "c5"}
    assert all([item.stage for item in stages] == [1, 2, 3, 4] for stages in by_client.values())


def test_manifest_contains_all_failure_strata():
    manifest = load_manifest(MANIFEST_PATH)
    assert set(manifest.failure_strata) == set(FailureStratum)


def test_manifest_has_disjoint_task_skill_combinations_across_splits():
    report = validate_no_leakage(load_manifest(MANIFEST_PATH))
    assert report.is_clean
    assert report.overlapping_task_skill_combinations == ()


def test_manifest_has_held_out_client_assignments():
    manifest = load_manifest(MANIFEST_PATH)
    assert any(item.split == "test" for item in manifest.assignments)
    train_keys = {item.assignment_key for item in manifest.assignments if item.split == "train"}
    test_keys = {item.assignment_key for item in manifest.assignments if item.split == "test"}
    assert train_keys.isdisjoint(test_keys)


def test_initial_state_seeds_are_disjoint_across_splits():
    manifest = load_manifest(MANIFEST_PATH)
    seed_sets = list(manifest.initial_state_seeds.values())
    assert all(left.isdisjoint(right) for index, left in enumerate(seed_sets) for right in seed_sets[index + 1 :])


def test_ranker_features_exclude_identity_and_ownership():
    manifest = load_manifest(MANIFEST_PATH)
    assert "task_id" not in manifest.ranker_features
    assert "client_id" not in manifest.ranker_features
    assert "source_client_id" not in manifest.ranker_features
    assert "client_ownership" not in manifest.ranker_features


def test_validator_rejects_seed_leakage(tmp_path: Path):
    raw = raw_manifest()
    raw["initial_state_seeds"]["test"][0] = raw["initial_state_seeds"]["train"][0]
    path = tmp_path / "leaky.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ManifestError, match="initial-state seed leakage"):
        load_manifest(path)


def test_validator_rejects_task_skill_combination_leakage(tmp_path: Path):
    raw = raw_manifest()
    train_task = next(item for item in raw["assignments"] if item["split"] == "train")
    test_task = next(item for item in raw["assignments"] if item["split"] == "test")
    raw["tasks"][test_task["task"]]["skills"] = deepcopy(raw["tasks"][train_task["task"]]["skills"])
    path = tmp_path / "leaky.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ManifestError, match="task-skill combination leakage"):
        load_manifest(path)


def test_validator_rejects_missing_or_duplicate_stage(tmp_path: Path):
    raw = raw_manifest()
    c1 = [item for item in raw["assignments"] if item["client"] == "c1"]
    c1[-1]["stage"] = 3
    path = tmp_path / "bad-stage.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ManifestError, match="stages 1 through 4"):
        load_manifest(path)


def test_manifest_hash_is_stable():
    first = load_manifest(MANIFEST_PATH)
    second = load_manifest(MANIFEST_PATH)
    assert first.manifest_hash() == second.manifest_hash()
    assert len(first.manifest_hash()) == 64
