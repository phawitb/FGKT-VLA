import json
from pathlib import Path

import pytest

from fcut_vla.libero.evaluator import (
    EvalShardPlan,
    FgktLiberoEvaluator,
    adapter_checkpoint_from_report,
)


def test_gpu_command_pins_fgkt_libero_contract(tmp_path):
    plan = EvalShardPlan(
        policy_path=Path("/private/adapter"),
        policy_revision="adapter-sha",
        suite="libero_spatial",
        seed=101,
        episodes=1,
        output_dir=tmp_path / "eval",
        task_ids=(0,),
    )

    evaluator = FgktLiberoEvaluator(plan)
    command = evaluator.render_command()

    assert command.startswith("MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ")
    assert "python -m lerobot.scripts.lerobot_eval" in command
    assert "--eval.use_async_envs=false" in command
    assert "--eval.batch_size=1" in command
    assert "--eval.n_episodes=1" in command
    assert "--seed=101" in command
    assert "--env.task=libero_spatial" in command
    assert "--env.task_ids=[0]" in command
    assert 'observation.images.wrist_image' in command
    assert "adapter-sha" in evaluator.identity_json()


@pytest.mark.parametrize("episodes", [0, -1])
def test_eval_plan_rejects_unbounded_episode_budget(tmp_path, episodes):
    with pytest.raises(ValueError, match="episodes"):
        EvalShardPlan(Path("adapter"), "rev", "libero_spatial", 101, episodes, tmp_path)


def test_eval_info_requires_expected_episode_count_and_policy_identity(tmp_path):
    plan = EvalShardPlan(Path("adapter"), "rev", "libero_spatial", 101, 1, tmp_path)
    evaluator = FgktLiberoEvaluator(plan)
    info = tmp_path / "eval_info.json"
    info.write_text(json.dumps({"aggregated": {"pc_success": 1.0}, "num_episodes": 1}))

    summary = evaluator.validate_eval_info(info)

    assert summary["success_rate"] == 1.0
    assert summary["policy_revision"] == "rev"


def test_suite_eval_info_uses_official_overall_episode_count(tmp_path):
    plan = EvalShardPlan(
        Path("adapter"), "rev", "libero_spatial", 101, 1, tmp_path, task_ids=(0,)
    )
    evaluator = FgktLiberoEvaluator(plan)
    info = tmp_path / "eval_info.json"
    info.write_text(json.dumps({"overall": {"pc_success": 100.0, "n_episodes": 1}}))

    assert evaluator.validate_eval_info(info)["success_rate"] == 1.0


def test_suite_eval_info_counts_episodes_per_selected_task(tmp_path):
    plan = EvalShardPlan(
        Path("adapter"), "rev", "libero_spatial", 101, 1, tmp_path, task_ids=(0, 1)
    )
    evaluator = FgktLiberoEvaluator(plan)
    info = tmp_path / "eval_info.json"
    info.write_text(json.dumps({"overall": {"pc_success": 50.0, "n_episodes": 2}}))

    summary = evaluator.validate_eval_info(info)

    assert summary["total_episodes"] == 2
    assert summary["success_rate"] == 0.5


@pytest.mark.parametrize("task_ids", [(), (-1,), (0, 0)])
def test_eval_plan_rejects_invalid_task_ids(tmp_path, task_ids):
    with pytest.raises(ValueError, match="task_ids"):
        EvalShardPlan(
            Path("adapter"),
            "rev",
            "libero_spatial",
            101,
            1,
            tmp_path,
            task_ids=task_ids,
        )


def test_adapter_checkpoint_resolution_requires_verified_step_files(tmp_path):
    checkpoint = (
        tmp_path
        / "checkpoints"
        / "client-adapter"
        / "checkpoints"
        / "000010"
        / "pretrained_model"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

    assert adapter_checkpoint_from_report(tmp_path, {"valid": True, "completed_step": 10}) == checkpoint
    with pytest.raises(ValueError, match="verified"):
        adapter_checkpoint_from_report(tmp_path, {"valid": False, "completed_step": 10})
