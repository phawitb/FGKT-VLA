"""Resolve logical benchmark aliases to immutable installed LIBERO tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Iterable


class TaskResolutionError(ValueError):
    """Raised when an alias cannot be mapped to exactly one installed task."""


@dataclass(frozen=True)
class InstalledTask:
    suite: str
    problem_id: str
    instruction: str
    bddl_path: Path
    initial_states_path: Path


@dataclass(frozen=True)
class ResolvedTask:
    alias: str
    suite: str
    instruction: str
    problem_id: str
    bddl_path: str
    bddl_sha256: str
    initial_states_path: str
    initial_states_sha256: str


def normalize_instruction(value: str) -> str:
    return " ".join(value.split()).casefold()


def _file_hash(path: Path, label: str) -> str:
    if not path.is_file():
        raise TaskResolutionError(f"{label} file does not exist: {path}")
    return sha256(path.read_bytes()).hexdigest()


def resolve_task(
    *,
    alias: str,
    suite: str,
    instruction: str,
    installed_tasks: Iterable[InstalledTask],
) -> ResolvedTask:
    alias = alias.strip()
    suite = suite.strip()
    normalized = normalize_instruction(instruction)
    if not alias or not suite or not normalized:
        raise TaskResolutionError("alias, suite, and instruction must be non-empty")
    matches = [
        task
        for task in installed_tasks
        if task.suite == suite and normalize_instruction(task.instruction) == normalized
    ]
    if not matches:
        raise TaskResolutionError(
            f"no installed LIBERO task matches {alias!r} in suite {suite!r}"
        )
    if len(matches) > 1:
        problem_ids = ", ".join(sorted(task.problem_id for task in matches))
        raise TaskResolutionError(
            f"multiple installed LIBERO tasks match {alias!r}: {problem_ids}"
        )
    task = matches[0]
    return ResolvedTask(
        alias=alias,
        suite=suite,
        instruction=" ".join(instruction.split()),
        problem_id=task.problem_id,
        bddl_path=str(task.bddl_path.resolve()),
        bddl_sha256=_file_hash(task.bddl_path, "BDDL"),
        initial_states_path=str(task.initial_states_path.resolve()),
        initial_states_sha256=_file_hash(task.initial_states_path, "initial-state"),
    )


def freeze_task_resolution(
    tasks: Iterable[ResolvedTask], *, libero_commit: str
) -> str:
    commit = libero_commit.strip()
    if not commit:
        raise TaskResolutionError("LIBERO commit must be non-empty")
    ordered = sorted(tasks, key=lambda task: task.alias)
    aliases = [task.alias for task in ordered]
    if len(set(aliases)) != len(aliases):
        raise TaskResolutionError("resolved task aliases must be unique")
    payload = {
        "schema_version": 1,
        "libero_commit": commit,
        "tasks": [asdict(task) for task in ordered],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_sha256"] = sha256(canonical.encode()).hexdigest()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
