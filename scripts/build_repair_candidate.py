#!/usr/bin/env python3
"""Train one immutable failure-aligned continuation adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import torch
import yaml
from huggingface_hub import HfApi

from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.episode import validate_episode_shard
from fcut_vla.libero.runtime import CANONICAL_ADAPTER_REPO, verified_hf_libero_version
from fcut_vla.repair.continuation import (
    ContinuationHyperparameters,
    DatasetEpisode,
    build_continuation_recipe,
    resolve_task_episode_indices,
)
from fcut_vla.repair.training import (
    ContinuationTrainingPlan,
    materialize_candidate_policy_config,
    render_continuation_command,
    validate_artifact_separation,
    validate_training_runtime,
    verify_repair_candidate,
)
from scripts.verify_adapter_run import verify_run


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def _load_dataset_episodes(repo: str, revision: str) -> tuple[DatasetEpisode, ...]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo, revision=revision)
    if metadata.episodes is None:
        raise ValueError("dataset metadata contains no episodes")
    episodes = []
    for row in metadata.episodes:
        index = row.get("episode_index")
        tasks = row.get("tasks")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("dataset metadata episode_index must be an exact integer")
        if not isinstance(tasks, list) or any(not isinstance(task, str) for task in tasks):
            raise ValueError("dataset metadata tasks must be a list of strings")
        episodes.append(DatasetEpisode(index=index, tasks=tuple(tasks)))
    return tuple(episodes)


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_bytes())
    if not isinstance(config, dict):
        raise ValueError("repair config must contain a mapping")
    if config.get("recipe_type") != "failure_aligned_continuation_v1":
        raise ValueError("repair config recipe_type is unsupported")
    return config


def _resolve_tokenizer_identity(
    preprocessor_path: Path, expected_repo: str, expected_revision: str
) -> tuple[str, str]:
    payload = json.loads(Path(preprocessor_path).read_text())
    names: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "tokenizer_name" and isinstance(child, str):
                    names.append(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    unique = sorted(set(names))
    if len(unique) != 1 or unique[0] != expected_repo:
        raise ValueError("source preprocessor must identify exactly one tokenizer repository")
    info = HfApi().model_info(unique[0], revision=expected_revision)
    revision = getattr(info, "sha", None)
    if revision != expected_revision:
        raise ValueError("tokenizer repository did not resolve to the frozen revision")
    return unique[0], revision


def _verified_source_runtime(source_manifest: dict) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    lerobot_root = repo_root / ".deps" / "lerobot"
    actual_commit = subprocess.run(
        ["git", "-C", str(lerobot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(lerobot_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    expected = source_manifest["runtime"]
    return validate_training_runtime(
        expected_lerobot_commit=expected["lerobot_commit"],
        expected_peft_version=expected["peft_version"],
        actual_lerobot_commit=actual_commit,
        actual_peft_version=version("peft"),
        lerobot_dirty=dirty,
    )


def _completion_bytes(recipe_hash: str, metadata_bytes: bytes, report: dict) -> bytes:
    payload = {
        "schema_version": 1,
        "recipe_sha256": recipe_hash,
        "candidate_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "candidate_adapter_sha256": report["candidate_adapter_sha256"],
        "adapter_config_sha256": report["adapter_config_sha256"],
        "policy_config_sha256": report["policy_config_sha256"],
        "preprocessor_sha256": report["preprocessor_sha256"],
        "postprocessor_sha256": report["postprocessor_sha256"],
        "processor_manifest_sha256": report["processor_manifest_sha256"],
    }
    return _canonical(
        {**payload, "completion_sha256": sha256(_canonical(payload)).hexdigest()}
    ) + b"\n"


def _prepare_training_attempt(run_dir: Path, attempt_dir: Path, *, resume: bool) -> None:
    run_dir = Path(run_dir).resolve()
    attempt_dir = Path(attempt_dir).resolve()
    if attempt_dir.parent != run_dir or attempt_dir.name != ".training-attempt":
        raise ValueError("training attempt path is outside the candidate run")
    if not attempt_dir.exists():
        return
    if not resume:
        raise ValueError("incomplete training attempt exists; pass --resume")
    failed_root = run_dir / "failed-attempts"
    failed_root.mkdir(exist_ok=True)
    index = 1
    while (failed_root / f"attempt-{index:04d}").exists():
        index += 1
    attempt_dir.replace(failed_root / f"attempt-{index:04d}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-adapter-run", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--failure-shard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    source_report = verify_run(args.source_adapter_run)
    source_manifest = json.loads((args.source_adapter_run / "run_manifest.json").read_text())
    source_evidence = json.loads((args.source_adapter_run / "adapter_metadata.json").read_text())
    episodes = validate_episode_shard(args.episode)
    if len(episodes) != 1:
        raise ValueError("repair candidate requires exactly one failed source episode")
    episode = episodes[0]
    if episode.terminal_success:
        raise ValueError("repair candidate requires a failed source episode")
    if (episode.policy_repo, episode.policy_revision) != (
        CANONICAL_ADAPTER_REPO,
        source_report["adapter_sha256"],
    ):
        raise ValueError("failed episode policy does not match source adapter")

    shard_manifest = json.loads((args.failure_shard / "SHARD.json").read_text())
    write_failure_shard(
        (episode,),
        output_dir=args.failure_shard,
        window_size=int(shard_manifest.get("window_size", -1)),
        resume=True,
    )
    failures = [
        json.loads(line)
        for line in (args.failure_shard / "failures.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(failures) != 1 or failures[0].get("source_episode_sha256") != episode.content_hash():
        raise ValueError("failure shard is not bound to the failed source episode")
    failure_hash = sha256(_canonical(failures[0]["context"])).hexdigest()

    config = _load_config(args.config)
    dataset_repo = _required_text(source_manifest["dataset_repo_id"], "dataset repository")
    dataset_revision = _required_text(source_manifest["dataset_revision"], "dataset revision")
    episode_indices = resolve_task_episode_indices(
        _load_dataset_episodes(dataset_repo, dataset_revision), episode.instruction
    )
    runtime = _verified_source_runtime(source_manifest)
    source_checkpoint = (
        args.source_adapter_run
        / "checkpoints"
        / "client-adapter"
        / "checkpoints"
        / f"{int(source_report['completed_step']):06d}"
    )
    source_pretrained = source_checkpoint / "pretrained_model"
    source_contract_paths = {
        "source_adapter_config_sha256": source_pretrained / "adapter_config.json",
        "source_policy_config_sha256": source_pretrained / "config.json",
        "source_preprocessor_sha256": source_pretrained / "policy_preprocessor.json",
        "source_postprocessor_sha256": source_pretrained / "policy_postprocessor.json",
    }
    try:
        source_contract = {
            name: sha256(path.read_bytes()).hexdigest()
            for name, path in source_contract_paths.items()
        }
    except FileNotFoundError as error:
        raise ValueError("source policy/processor contract is incomplete") from error
    source_processor_paths = sorted(
        path
        for path in source_pretrained.iterdir()
        if path.is_file()
        and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
    )
    if not source_processor_paths:
        raise ValueError("source processor artifact set is empty")
    source_processor_artifacts = tuple(
        (path.name, sha256(path.read_bytes()).hexdigest())
        for path in source_processor_paths
    )
    tokenizer_repo, tokenizer_revision = _resolve_tokenizer_identity(
        source_pretrained / "policy_preprocessor.json",
        _required_text(config["tokenizer_repo"], "tokenizer repository"),
        _required_text(config["tokenizer_revision"], "tokenizer revision"),
    )
    recipe = build_continuation_recipe(
        source_run_hash=_required_text(source_report["run_hash"], "source run hash"),
        source_adapter_sha256=_required_text(
            source_report["adapter_sha256"], "source adapter digest"
        ),
        **source_contract,
        source_processor_artifacts=source_processor_artifacts,
        tokenizer_repo=tokenizer_repo,
        tokenizer_revision=tokenizer_revision,
        source_episode_sha256=episode.content_hash(),
        failure_hash=failure_hash,
        task_alias=episode.key.task_alias,
        problem_id=episode.problem_id,
        instruction=episode.instruction,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        episode_indices=episode_indices,
        base_policy_repo=_required_text(source_manifest["base_policy"], "base policy repository"),
        base_policy_revision=_required_text(
            source_manifest["base_policy_revision"], "base policy revision"
        ),
        lerobot_commit=_required_text(runtime["lerobot_commit"], "LeRobot commit"),
        peft_version=_required_text(runtime["peft_version"], "PEFT version"),
        hf_libero_version=verified_hf_libero_version(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        device=config["device"],
        hyperparameters=ContinuationHyperparameters(
            steps=config["continuation_steps"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            training_seed=config["training_seed"],
        ),
        source_target_modules=source_evidence["resolved_targets"],
    )
    run_dir = args.output_root / "repair-candidate" / recipe.content_hash[:16]
    validate_artifact_separation(args.source_adapter_run, run_dir)
    existed = run_dir.exists()
    if existed and not args.resume:
        raise ValueError(f"repair candidate run exists; pass --resume: {run_dir}")
    attempt_dir = run_dir / ".training-attempt"
    final_training_dir = run_dir / "training"
    attempt_plan = ContinuationTrainingPlan(
        recipe=recipe,
        source_checkpoint_dir=source_checkpoint,
        source_pretrained_dir=source_pretrained,
        output_dir=attempt_dir,
        save_frequency=config["save_frequency"],
        python_executable=Path(sys.executable),
    )
    final_plan = replace(attempt_plan, output_dir=final_training_dir)
    command = render_continuation_command(attempt_plan)
    recipe_bytes = (recipe.to_json() + "\n").encode()
    _write_immutable(run_dir / "repair_recipe.json", recipe_bytes)
    _write_immutable(run_dir / "command.sh", (shlex.join(command) + "\n").encode())
    result = {
        "command": list(command),
        "episode_indices": list(recipe.episode_indices),
        "recipe_sha256": recipe.content_hash,
        "run_dir": str(run_dir),
    }
    if args.dry_run:
        print(json.dumps(result, sort_keys=True))
        return

    candidate_log = run_dir / "logs" / "stderr.log"
    if (run_dir / "COMPLETE").exists():
        if not args.resume:
            raise ValueError("completed repair candidate is immutable")
        candidate_report = verify_repair_candidate(
            plan=final_plan,
            source_evidence=source_evidence,
            candidate_log=candidate_log,
        )
    elif final_training_dir.exists():
        if not args.resume:
            raise ValueError("unfinalized candidate output exists; pass --resume")
        _prepare_training_attempt(run_dir, attempt_dir, resume=True)
        candidate_report = verify_repair_candidate(
            plan=final_plan,
            source_evidence=source_evidence,
            candidate_log=candidate_log,
        )
    else:
        _prepare_training_attempt(run_dir, attempt_dir, resume=args.resume)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "MUJOCO_GL": "egl",
        }
        with (run_dir / "logs" / "stdout.log").open("a") as stdout, candidate_log.open(
            "a"
        ) as stderr:
            completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
        if completed.returncode:
            _prepare_training_attempt(run_dir, attempt_dir, resume=True)
            raise SystemExit(completed.returncode)
        materialize_candidate_policy_config(attempt_plan)
        verify_repair_candidate(
            plan=attempt_plan,
            source_evidence=source_evidence,
            candidate_log=candidate_log,
        )
        attempt_dir.replace(final_training_dir)
        candidate_report = verify_repair_candidate(
            plan=final_plan,
            source_evidence=source_evidence,
            candidate_log=candidate_log,
        )
    report_mapping = asdict(candidate_report)
    metadata_bytes = _canonical(report_mapping) + b"\n"
    _write_immutable(run_dir / "candidate_metadata.json", metadata_bytes)
    _write_immutable(
        run_dir / "COMPLETE",
        _completion_bytes(recipe.content_hash, metadata_bytes, report_mapping),
    )
    print(json.dumps({**result, **report_mapping}, sort_keys=True))


if __name__ == "__main__":
    main()
