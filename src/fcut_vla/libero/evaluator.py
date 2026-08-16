"""Thin, version-pinned adapter around the official LeRobot LIBERO evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


def adapter_checkpoint_from_report(run_dir: Path, report: dict[str, Any]) -> Path:
    if report.get("valid") is not True:
        raise ValueError("adapter run must be verified before evaluation")
    step = int(report.get("completed_step", -1))
    checkpoint = (
        Path(run_dir)
        / "checkpoints"
        / "client-adapter"
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )
    required = (checkpoint / "adapter_config.json", checkpoint / "adapter_model.safetensors")
    if step < 1 or any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError("verified adapter checkpoint files are missing")
    return checkpoint


@dataclass(frozen=True)
class EvalShardPlan:
    policy_path: Path
    policy_revision: str
    suite: str
    seed: int
    episodes: int
    output_dir: Path
    task_ids: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not str(self.policy_path) or not self.policy_revision.strip():
            raise ValueError("policy path and revision must be non-empty")
        if self.suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError("unsupported LIBERO suite")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if (
            not self.task_ids
            or any(task_id < 0 for task_id in self.task_ids)
            or len(set(self.task_ids)) != len(self.task_ids)
        ):
            raise ValueError("task_ids must be non-empty, unique, and non-negative")


class FgktLiberoEvaluator:
    RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}

    def __init__(self, plan: EvalShardPlan) -> None:
        self.plan = plan

    def argv(self) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            f"--policy.path={self.plan.policy_path}",
            "--policy.device=cuda",
            "--env.type=libero",
            f"--env.task={self.plan.suite}",
            "--env.task_ids=" + json.dumps(self.plan.task_ids, separators=(",", ":")),
            "--env.max_parallel_tasks=1",
            "--eval.batch_size=1",
            f"--eval.n_episodes={self.plan.episodes}",
            "--eval.use_async_envs=false",
            f"--seed={self.plan.seed}",
            "--rename_map=" + json.dumps(self.RENAME_MAP, sort_keys=True, separators=(",", ":")),
            f"--output_dir={self.plan.output_dir}",
        )

    def render_command(self) -> str:
        return "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl " + shlex.join(self.argv())

    def identity_json(self) -> str:
        payload = {
            **asdict(self.plan),
            "policy_path": str(self.plan.policy_path),
            "output_dir": str(self.plan.output_dir),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def run(self) -> dict[str, Any]:
        environment = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
        self.plan.output_dir.mkdir(parents=True, exist_ok=False)
        subprocess.run(self.argv(), check=True, env=environment)
        return self.validate_eval_info(self.plan.output_dir / "eval_info.json")

    def validate_eval_info(self, path: Path) -> dict[str, Any]:
        try:
            info = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("eval_info.json is missing or malformed") from error
        if not isinstance(info, dict):
            raise ValueError("eval_info.json must contain an object")
        uses_overall = isinstance(info.get("overall"), dict)
        aggregated = info.get("overall") if uses_overall else info.get("aggregated")
        if not isinstance(aggregated, dict) or "pc_success" not in aggregated:
            raise ValueError("eval_info.json is missing pc_success")
        count = info.get("num_episodes")
        if count is None and isinstance(info.get("per_episode"), list):
            count = len(info["per_episode"])
        if count is None:
            count = aggregated.get("n_episodes")
        expected_count = self.plan.episodes * len(self.plan.task_ids) if uses_overall else self.plan.episodes
        if int(count if count is not None else -1) != expected_count:
            raise ValueError("eval_info.json episode count does not match plan")
        success = float(aggregated["pc_success"])
        if success > 1:
            success /= 100.0
        if not 0 <= success <= 1:
            raise ValueError("eval success rate must be between zero and one")
        return {
            "policy_revision": self.plan.policy_revision,
            "suite": self.plan.suite,
            "seed": self.plan.seed,
            "episodes": self.plan.episodes,
            "total_episodes": expected_count,
            "success_rate": success,
        }
