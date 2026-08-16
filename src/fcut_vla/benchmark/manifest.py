"""Parse and validate the immutable FedLIBERO-Fail split manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ManifestError(ValueError):
    """Raised when a benchmark manifest permits experimental leakage."""


class FailureStratum(str, Enum):
    SEEN_TASK = "seen_task"
    COMPOSITIONAL = "compositional"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    suite: str
    instruction: str
    skills: tuple[str, ...]

    @property
    def skill_combination(self) -> tuple[str, ...]:
        return tuple(sorted(self.skills))


@dataclass(frozen=True)
class TaskStage:
    client: str
    stage: int
    task: str
    split: str
    assignment_key: str


@dataclass(frozen=True)
class LeakageReport:
    overlapping_task_skill_combinations: tuple[tuple[str, ...], ...]
    overlapping_initial_state_seeds: tuple[int, ...]
    duplicate_assignment_keys: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (
            self.overlapping_task_skill_combinations
            or self.overlapping_initial_state_seeds
            or self.duplicate_assignment_keys
        )


@dataclass(frozen=True)
class FedLiberoManifest:
    version: str
    ranker_features: tuple[str, ...]
    failure_strata: tuple[FailureStratum, ...]
    tasks: Mapping[str, TaskDefinition]
    assignments: tuple[TaskStage, ...]
    initial_state_seeds: Mapping[str, frozenset[int]]

    def assignments_by_client(self) -> dict[str, list[TaskStage]]:
        grouped: dict[str, list[TaskStage]] = {}
        for item in self.assignments:
            grouped.setdefault(item.client, []).append(item)
        return {client: sorted(items, key=lambda item: item.stage) for client, items in grouped.items()}

    def manifest_hash(self) -> str:
        payload = {
            "version": self.version,
            "ranker_features": self.ranker_features,
            "failure_strata": [item.value for item in self.failure_strata],
            "tasks": {name: asdict(task) for name, task in sorted(self.tasks.items())},
            "assignments": [asdict(item) for item in self.assignments],
            "initial_state_seeds": {
                split: sorted(seeds) for split, seeds in sorted(self.initial_state_seeds.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


def _duplicates(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if values.count(value) > 1}))


def validate_no_leakage(manifest: FedLiberoManifest) -> LeakageReport:
    combos_by_split: dict[str, set[tuple[str, ...]]] = {}
    for assignment in manifest.assignments:
        combo = manifest.tasks[assignment.task].skill_combination
        combos_by_split.setdefault(assignment.split, set()).add(combo)

    overlapping_combos: set[tuple[str, ...]] = set()
    split_names = sorted(combos_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlapping_combos.update(combos_by_split[left].intersection(combos_by_split[right]))

    overlapping_seeds: set[int] = set()
    seed_splits = sorted(manifest.initial_state_seeds)
    for index, left in enumerate(seed_splits):
        for right in seed_splits[index + 1 :]:
            overlapping_seeds.update(
                manifest.initial_state_seeds[left].intersection(manifest.initial_state_seeds[right])
            )

    keys = [item.assignment_key for item in manifest.assignments]
    return LeakageReport(
        overlapping_task_skill_combinations=tuple(sorted(overlapping_combos)),
        overlapping_initial_state_seeds=tuple(sorted(overlapping_seeds)),
        duplicate_assignment_keys=_duplicates(keys),
    )


def _parse(raw: Mapping[str, Any]) -> FedLiberoManifest:
    tasks = {
        name: TaskDefinition(
            name=name,
            suite=str(values["suite"]),
            instruction=str(values["instruction"]),
            skills=tuple(sorted(str(skill) for skill in values["skills"])),
        )
        for name, values in raw["tasks"].items()
    }
    assignments = tuple(
        TaskStage(
            client=str(item["client"]),
            stage=int(item["stage"]),
            task=str(item["task"]),
            split=str(item["split"]),
            assignment_key=str(item["assignment_key"]),
        )
        for item in raw["assignments"]
    )
    manifest = FedLiberoManifest(
        version=str(raw["version"]),
        ranker_features=tuple(sorted(str(item) for item in raw["ranker_features"])),
        failure_strata=tuple(FailureStratum(item) for item in raw["failure_strata"]),
        tasks=tasks,
        assignments=assignments,
        initial_state_seeds={
            split: frozenset(int(seed) for seed in seeds)
            for split, seeds in raw["initial_state_seeds"].items()
        },
    )
    return manifest


def _validate_structure(manifest: FedLiberoManifest) -> None:
    if set(manifest.assignments_by_client()) != {"c1", "c2", "c3", "c4", "c5"}:
        raise ManifestError("manifest must define clients c1 through c5")
    for client, stages in manifest.assignments_by_client().items():
        if [item.stage for item in stages] != [1, 2, 3, 4]:
            raise ManifestError(f"{client} must contain stages 1 through 4 exactly once")
    if any(item.task not in manifest.tasks for item in manifest.assignments):
        raise ManifestError("every assignment task must have a task definition")
    if set(manifest.failure_strata) != set(FailureStratum):
        raise ManifestError("all failure strata must be present")
    forbidden = {"task_id", "client_id", "source_client_id", "client_ownership"}
    leaked = forbidden.intersection(manifest.ranker_features)
    if leaked:
        raise ManifestError(f"forbidden ranker feature: {', '.join(sorted(leaked))}")


def load_manifest(path: Path) -> FedLiberoManifest:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest root must be a mapping")
    manifest = _parse(raw)
    _validate_structure(manifest)
    report = validate_no_leakage(manifest)
    if report.overlapping_initial_state_seeds:
        raise ManifestError(
            f"initial-state seed leakage: {report.overlapping_initial_state_seeds}"
        )
    if report.overlapping_task_skill_combinations:
        raise ManifestError(
            "task-skill combination leakage: "
            f"{report.overlapping_task_skill_combinations}"
        )
    if report.duplicate_assignment_keys:
        raise ManifestError(f"duplicate assignment keys: {report.duplicate_assignment_keys}")
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    print(manifest.manifest_hash())


if __name__ == "__main__":
    main()

