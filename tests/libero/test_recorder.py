import pytest
import numpy as np

from fcut_vla.libero.recorder import (
    EpisodeIdentity,
    SanitizedEpisodeRecorder,
    executed_action_summary,
    hash_initial_state,
)


def _identity() -> EpisodeIdentity:
    return EpisodeIdentity(
        task_alias="spatial.task0",
        problem_id="pick_up_black_bowl",
        seed=101,
        episode_index=0,
        initial_state_index=0,
        initial_state_hash="state-sha",
        policy_repo="phawitbinabik/fgkt-vla",
        policy_revision="policy-sha",
        instruction="pick up the black bowl",
    )


def _append(recorder: SanitizedEpisodeRecorder, *, success=False, env_done=False):
    recorder.append(
        features=(0.1, 0.2),
        proprioception=(0.0,) * 8,
        executed_action=(0.0,) * 7,
        action_statistics=(0.0, 0.1),
        reward=float(success),
        success=success,
        env_done=env_done,
    )


def test_recorder_forces_failed_terminal_step_at_frozen_horizon():
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=2)

    _append(recorder)
    _append(recorder)
    record = recorder.finalize()

    assert not record.terminal_success
    assert not record.steps[0].done
    assert record.steps[1].done
    assert record.key.seed == 101


def test_recorder_stops_on_success_and_rejects_frames_after_terminal():
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=280)

    _append(recorder, success=True, env_done=True)

    assert recorder.finalize().terminal_success
    with pytest.raises(ValueError, match="terminal"):
        _append(recorder)


def test_recorder_requires_terminal_and_fixed_step_dimensions():
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=3)
    _append(recorder)

    with pytest.raises(ValueError, match="terminal"):
        recorder.finalize()
    with pytest.raises(ValueError, match="dimensions"):
        recorder.append(
            features=(0.1,),
            proprioception=(0.0,) * 8,
            executed_action=(0.0,) * 7,
            action_statistics=(0.0, 0.1),
            reward=0.0,
            success=False,
            env_done=False,
        )


def test_recorder_rejects_ambiguous_success_without_done():
    recorder = SanitizedEpisodeRecorder(_identity(), max_steps=2)

    with pytest.raises(ValueError, match="success"):
        _append(recorder, success=True, env_done=False)


def test_initial_state_hash_binds_shape_dtype_and_canonical_bytes():
    contiguous = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    same_values_noncontiguous = np.asfortranarray(contiguous)

    assert hash_initial_state(contiguous) == hash_initial_state(same_values_noncontiguous)
    assert hash_initial_state(contiguous) != hash_initial_state(contiguous.astype(np.float64))
    assert hash_initial_state(contiguous) != hash_initial_state(contiguous.reshape(4))


def test_executed_action_summary_has_frozen_non_distribution_definition():
    summary = executed_action_summary((3.0, 4.0))

    assert summary == (3.5, 0.5, 5.0, 4.0)
    with pytest.raises(ValueError, match="action"):
        executed_action_summary(())
