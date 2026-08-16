from hashlib import sha256
import json

import pytest

from fcut_vla.libero.task_resolution import (
    InstalledTask,
    TaskResolutionError,
    freeze_task_resolution,
    resolve_task,
)


def _installed(tmp_path, *, problem_id="pick_up_mug", suite="libero_spatial"):
    bddl = tmp_path / f"{problem_id}.bddl"
    states = tmp_path / f"{problem_id}.states"
    bddl.write_bytes(b"problem definition")
    states.write_bytes(b"initial states")
    return InstalledTask(
        suite=suite,
        problem_id=problem_id,
        instruction="Pick   up the MUG",
        bddl_path=bddl,
        initial_states_path=states,
    )


def test_resolve_task_requires_one_normalized_instruction_match(tmp_path):
    installed = _installed(tmp_path)

    resolved = resolve_task(
        alias="spatial.pick_mug",
        suite="libero_spatial",
        instruction="  pick up the mug ",
        installed_tasks=[installed],
    )

    assert resolved.problem_id == "pick_up_mug"
    assert resolved.bddl_sha256 == sha256(installed.bddl_path.read_bytes()).hexdigest()
    assert resolved.initial_states_sha256 == sha256(
        installed.initial_states_path.read_bytes()
    ).hexdigest()


def test_resolution_does_not_match_same_instruction_from_other_suite(tmp_path):
    installed = _installed(tmp_path, suite="libero_goal")

    with pytest.raises(TaskResolutionError, match="no installed LIBERO task"):
        resolve_task(
            alias="spatial.pick_mug",
            suite="libero_spatial",
            instruction="pick up the mug",
            installed_tasks=[installed],
        )


@pytest.mark.parametrize("count", [0, 2])
def test_resolve_task_rejects_zero_or_multiple_matches(tmp_path, count):
    installed = [
        _installed(tmp_path, problem_id=f"pick_up_mug_{index}") for index in range(count)
    ]

    with pytest.raises(TaskResolutionError):
        resolve_task(
            alias="spatial.pick_mug",
            suite="libero_spatial",
            instruction="pick up the mug",
            installed_tasks=installed,
        )


def test_freeze_task_resolution_is_order_independent_and_self_hashing(tmp_path):
    first = resolve_task(
        alias="z-task",
        suite="libero_spatial",
        instruction="pick up the mug",
        installed_tasks=[_installed(tmp_path, problem_id="z")],
    )
    second_installed = _installed(tmp_path, problem_id="a", suite="libero_goal")
    second = resolve_task(
        alias="a-task",
        suite="libero_goal",
        instruction="pick up the mug",
        installed_tasks=[second_installed],
    )

    left = freeze_task_resolution([first, second], libero_commit="abc123")
    right = freeze_task_resolution([second, first], libero_commit="abc123")
    payload = json.loads(left)
    claimed_hash = payload.pop("content_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    assert left == right
    assert [task["alias"] for task in json.loads(left)["tasks"]] == ["a-task", "z-task"]
    assert claimed_hash == sha256(canonical.encode()).hexdigest()
