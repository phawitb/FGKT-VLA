"""Runtime orchestration for one verified official-LeRobot LIBERO episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from fcut_vla.libero.episode import EpisodeRecord
from fcut_vla.libero.recorder import EpisodeIdentity, SanitizedEpisodeRecorder, hash_initial_state
from fcut_vla.libero.rollout import PreparedObservation, image_moments_v1, run_single_episode


@dataclass(frozen=True)
class LeRobotRecordPlan:
    checkpoint: Path
    policy_repo: str
    policy_revision: str
    suite: str
    task_id: int
    task_alias: str
    seed: int
    episode_index: int
    initial_state_index: int
    output_path: Path

    def __post_init__(self) -> None:
        required = (self.policy_repo, self.policy_revision, self.suite, self.task_alias)
        if any(not value.strip() for value in required):
            raise ValueError("record plan identity fields must be non-empty")
        if min(self.task_id, self.seed, self.episode_index, self.initial_state_index) < 0:
            raise ValueError("record plan indices and seed must be non-negative")


@dataclass(frozen=True)
class RuntimeTask:
    problem_id: str
    instruction: str
    initial_states: np.ndarray
    max_steps: int


@dataclass(frozen=True)
class RuntimeComponents:
    env: Any
    prepare: Callable[[Any], PreparedObservation]
    select_action: Callable[[Any], tuple[float, ...]]
    close: Callable[[], None]


class RuntimeBindings(Protocol):
    def resolve_task(self, plan: LeRobotRecordPlan) -> RuntimeTask: ...

    def build_components(
        self, plan: LeRobotRecordPlan, task: RuntimeTask
    ) -> RuntimeComponents: ...


def _build_with_env_cleanup(env: Any, builder: Callable[[], RuntimeComponents]) -> RuntimeComponents:
    try:
        return builder()
    except BaseException:
        env.close()
        raise


class OfficialLeRobotBindings:
    """Lazy Linux/CUDA bindings to the pinned official LeRobot runtime."""

    RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}

    def resolve_task(self, plan: LeRobotRecordPlan) -> RuntimeTask:
        from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite, get_task_init_states

        suite = _get_suite(plan.suite)
        if plan.task_id >= len(suite.tasks):
            raise ValueError("task id is outside the selected LIBERO suite")
        installed = suite.get_task(plan.task_id)
        states = get_task_init_states(suite, plan.task_id)
        return RuntimeTask(
            problem_id=str(installed.name),
            instruction=str(installed.language),
            initial_states=np.asarray(states),
            max_steps=int(TASK_SUITE_MAX_STEPS[plan.suite]),
        )

    def build_components(
        self, plan: LeRobotRecordPlan, task: RuntimeTask
    ) -> RuntimeComponents:
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        from lerobot.utils.random_utils import set_seed

        set_seed(plan.seed)
        env_cfg = LiberoEnv(task=plan.suite, task_ids=[plan.task_id])
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        env = envs[plan.suite][plan.task_id]
        return _build_with_env_cleanup(
            env,
            lambda: self._finish_components(plan, task, env_cfg, env),
        )

    def _finish_components(
        self, plan: LeRobotRecordPlan, task: RuntimeTask, env_cfg: Any, env: Any
    ) -> RuntimeComponents:
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        env.envs[0].unwrapped.init_state_id = plan.initial_state_index

        policy_cfg = PreTrainedConfig.from_pretrained(str(plan.checkpoint))
        policy_cfg.pretrained_path = plan.checkpoint
        policy_cfg.device = "cuda"
        policy = make_policy(
            cfg=policy_cfg,
            env_cfg=env_cfg,
            rename_map=self.RENAME_MAP,
        )
        policy.eval()
        policy.reset()

        preprocessor_overrides = {
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": self.RENAME_MAP},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(plan.checkpoint),
            pretrained_revision=policy_cfg.pretrained_revision,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )

        def prepare(raw_observation: Any) -> PreparedObservation:
            observation = preprocess_observation(raw_observation)
            observation["task"] = [task.instruction]
            observation = env_preprocessor(observation)
            features, proprioception = image_moments_v1(observation)
            return PreparedObservation(
                policy_input=preprocessor(observation),
                features=features,
                proprioception=proprioception,
            )

        def select_action(policy_input: Any) -> tuple[float, ...]:
            with torch.inference_mode():
                action = policy.select_action(policy_input)
            action = postprocessor(action)
            transition = env_postprocessor({"action": action})
            executed = transition["action"].detach().to("cpu")
            if tuple(executed.shape) != (1, 7):
                raise ValueError("official LeRobot policy produced an unexpected action shape")
            return tuple(float(value) for value in executed.reshape(-1).tolist())

        return RuntimeComponents(env, prepare, select_action, env.close)


def record_lerobot_episode(
    plan: LeRobotRecordPlan, *, bindings: RuntimeBindings
) -> EpisodeRecord:
    """Resolve frozen identity, run exactly one episode, and atomically persist it."""
    if plan.output_path.exists():
        raise ValueError("episode output already exists")
    task = bindings.resolve_task(plan)
    states = np.asarray(task.initial_states)
    if states.ndim < 1 or plan.initial_state_index >= len(states):
        raise ValueError("initial state index is outside the resolved task state array")
    if task.max_steps < 1 or not task.problem_id.strip() or not task.instruction.strip():
        raise ValueError("resolved runtime task is incomplete")

    identity = EpisodeIdentity(
        task_alias=plan.task_alias,
        problem_id=task.problem_id,
        seed=plan.seed,
        episode_index=plan.episode_index,
        initial_state_index=plan.initial_state_index,
        initial_state_hash=hash_initial_state(states[plan.initial_state_index]),
        policy_repo=plan.policy_repo,
        policy_revision=plan.policy_revision,
        instruction=task.instruction,
    )
    recorder = SanitizedEpisodeRecorder(identity, max_steps=task.max_steps)
    components = bindings.build_components(plan, task)
    try:
        record = run_single_episode(
            components.env,
            recorder,
            prepare=components.prepare,
            select_action=components.select_action,
        )
    finally:
        components.close()

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.output_path.with_name(plan.output_path.name + ".tmp")
    temporary.write_text(record.to_json() + "\n")
    temporary.replace(plan.output_path)
    return record
