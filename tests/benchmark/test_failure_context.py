import pytest

from fcut_vla.benchmark.failure_context import FailureContext, FailureContextError
from fcut_vla.types import FailureId


def episode_steps(count: int) -> list[dict]:
    return [
        {
            "features": [float(index), float(index + 1)],
            "proprioception": [float(index) / 10],
            "executed_action": [float(index) / 100, -float(index) / 100],
            "action_statistics": [0.1 + index, 0.2 + index],
        }
        for index in range(count)
    ]


def test_failure_window_ends_at_failure_step():
    context = FailureContext.from_episode(
        failure_id=FailureId("episode-1", 4),
        instruction="put the mug in the drawer",
        steps=episode_steps(7),
        failure_step=4,
        window_size=3,
        terminal_success=False,
    )
    assert context.mask == (True, True, True)
    assert [step.features[0] for step in context.steps] == [2.0, 3.0, 4.0]


def test_short_episode_is_left_padded_and_masked():
    context = FailureContext.from_episode(
        failure_id=FailureId("episode-short", 1),
        instruction="open the drawer",
        steps=episode_steps(2),
        failure_step=1,
        window_size=4,
        terminal_success=False,
    )
    assert context.mask == (False, False, True, True)
    assert context.steps[0].features == (0.0, 0.0)
    assert context.steps[-1].features == (1.0, 2.0)


def test_serialization_and_hash_are_deterministic():
    kwargs = dict(
        failure_id=FailureId("episode-stable", 2),
        instruction="close the cabinet",
        steps=episode_steps(3),
        failure_step=2,
        window_size=3,
        terminal_success=False,
    )
    first = FailureContext.from_episode(**kwargs)
    second = FailureContext.from_episode(**kwargs)
    assert first.to_json() == second.to_json()
    assert first.content_hash() == second.content_hash()
    assert len(first.content_hash()) == 64


def test_rejects_success_episode_and_invalid_failure_index():
    with pytest.raises(FailureContextError, match="failed episode"):
        FailureContext.from_episode(
            failure_id=FailureId("episode-success", 1),
            instruction="open the drawer",
            steps=episode_steps(2),
            failure_step=1,
            window_size=2,
            terminal_success=True,
        )
    with pytest.raises(FailureContextError, match="failure_step"):
        FailureContext.from_episode(
            failure_id=FailureId("episode-invalid", 8),
            instruction="open the drawer",
            steps=episode_steps(2),
            failure_step=8,
            window_size=2,
            terminal_success=False,
        )


def test_rejects_inconsistent_step_dimensions():
    steps = episode_steps(3)
    steps[2]["features"] = [1.0]
    with pytest.raises(FailureContextError, match="consistent dimensions"):
        FailureContext.from_episode(
            failure_id=FailureId("episode-dims", 2),
            instruction="open the drawer",
            steps=steps,
            failure_step=2,
            window_size=3,
            terminal_success=False,
        )

