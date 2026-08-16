"""Shared immutable orchestration for RTX GPU entry points."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml
from huggingface_hub import snapshot_download

from fcut_vla.adapters.artifact import validate_adapter_run


STAGE_COMMANDS = {
    "train_client_adapter": (
        "python -m lerobot.scripts.lerobot_train --policy.path={base_policy} "
        "--policy.pretrained_revision={base_policy_revision} "
        "--policy.input_features=null --policy.output_features=null "
        "--dataset.repo_id={dataset} --dataset.revision={dataset_revision} "
        "--job_name={job_name} --policy.device={device} "
        "--batch_size={batch_size} --steps={steps} --seed={train_seed} "
        "--save_freq={save_freq} --env_eval_freq={env_eval_freq} "
        "{peft_arguments} "
        "--policy.push_to_hub={push_to_hub} {hub_repo_argument} "
        "--output_dir={run_dir}/checkpoints/client-adapter"
    ),
    "generate_failures": (
        "python -m fcut_vla.jobs.generate_failures --dataset {dataset} "
        "--seeds {seeds} --output {run_dir}/failures"
    ),
    "build_utility_labels": (
        "python -m fcut_vla.jobs.build_utility_labels --seeds {seeds} "
        "--output {run_dir}/utility-labels"
    ),
    "train_utility_ranker": (
        "python -m fcut_vla.jobs.train_utility_ranker "
        "--output {run_dir}/checkpoints/utility-ranker"
    ),
    "run_repair_eval": (
        "python -m fcut_vla.jobs.run_repair_eval --seeds {seeds} "
        "--output {run_dir}/metrics.json"
    ),
    "run_continual_experiment": (
        "python -m fcut_vla.jobs.run_continual_experiment --seeds {seeds} "
        "--output {run_dir}/continual"
    ),
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _runtime_provenance(config: dict) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    lerobot_commit = subprocess.run(
        ["git", "-C", str(repo_root / ".deps" / "lerobot"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root / ".deps" / "lerobot"), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("LeRobot checkout is dirty; refuse untracked runtime provenance")
    peft_version = version("peft")
    expected = {
        "lerobot_commit": str(config["lerobot_commit"]),
        "peft_version": str(config["peft_version"]),
    }
    actual = {"lerobot_commit": lerobot_commit, "peft_version": peft_version}
    if actual != expected:
        raise RuntimeError(f"runtime provenance mismatch: expected {expected}, got {actual}")
    return actual


def _resolve_base_policy(config: dict) -> Path:
    return Path(
        snapshot_download(
            repo_id=str(config["base_policy"]),
            revision=str(config["base_policy_revision"]),
        )
    )


def _normalize_adapter_provenance(output_dir: Path, base_policy: str, revision: str) -> None:
    checkpoints = sorted(
        (path for path in (Path(output_dir) / "checkpoints").iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not checkpoints:
        raise RuntimeError("cannot normalize adapter provenance without a checkpoint")
    config_path = checkpoints[-1] / "pretrained_model" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = base_policy
    config["revision"] = revision
    config_path.write_bytes(_canonical(config))


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    if not path.exists():
        path.write_bytes(content)


def _completion_record(
    run_hash: str,
    metadata_bytes: bytes,
    adapter_sha256: str,
    adapter_config_sha256: str,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_hash": run_hash,
        "adapter_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "adapter_sha256": adapter_sha256,
        "adapter_config_sha256": adapter_config_sha256,
    }
    record = {**payload, "completion_sha256": sha256(_canonical(payload)).hexdigest()}
    return _canonical(record) + b"\n"


def _normalized_peft(config: dict) -> dict:
    lora = config.get("lora")
    if not isinstance(lora, dict):
        raise ValueError("lora configuration must be a mapping")
    rank = int(lora.get("rank", 0))
    alpha = int(lora.get("alpha", 0))
    if rank < 1:
        raise ValueError("lora rank must be positive")
    if alpha < 1:
        raise ValueError("lora alpha must be positive")
    targets = lora.get("target_modules")
    if targets == "native_default":
        normalized_targets: str | list[str] = "native_default"
    elif isinstance(targets, list) and targets:
        normalized_targets = sorted({str(item).strip() for item in targets if str(item).strip()})
        if not normalized_targets:
            raise ValueError("lora target_modules must be non-empty")
    else:
        raise ValueError("lora target_modules must be native_default or a non-empty list")
    return {
        "method_type": "LORA",
        "rank": rank,
        "alpha": alpha,
        "target_modules": normalized_targets,
    }


def render_peft_arguments(config: dict) -> tuple[str, ...]:
    peft = _normalized_peft(config)
    arguments = [
        "--peft.method_type=LORA",
        f"--peft.r={peft['rank']}",
        f"--peft.lora_alpha={peft['alpha']}",
        "--peft.full_training_modules=[]",
    ]
    if peft["target_modules"] != "native_default":
        encoded_targets = json.dumps(peft["target_modules"], separators=(",", ":"))
        arguments.append("--peft.target_modules=" + shlex.quote(encoded_targets))
    return tuple(arguments)


def _validate_training_checkpoint(output_dir: Path) -> Path:
    candidates = sorted(Path(output_dir).glob("checkpoints/*/pretrained_model"))
    if not candidates:
        raise RuntimeError(f"no pretrained checkpoint found under {output_dir}")
    checkpoint = candidates[-1]
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete pretrained checkpoint {checkpoint}; missing: {', '.join(missing)}"
        )
    if any((checkpoint / name).stat().st_size == 0 for name in required):
        raise RuntimeError(f"pretrained checkpoint contains empty required files: {checkpoint}")
    return checkpoint


def run_stage(stage: str) -> None:
    parser = argparse.ArgumentParser(description=f"FCUT-VLA GPU stage: {stage}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    peft = _normalized_peft(config)
    runtime = _runtime_provenance(config)
    base_policy_snapshot = _resolve_base_policy(config)
    evaluation = [int(seed) for seed in config["evaluation_seeds"]]
    counterfactual = [int(seed) for seed in config["counterfactual_seeds"]]
    if evaluation != counterfactual:
        raise ValueError("evaluation and counterfactual seeds must be exactly paired")

    manifest_path = Path(config["benchmark_manifest"])
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    identity = {
        "schema_version": 1,
        "stage": stage,
        "config_sha256": sha256(config_bytes).hexdigest(),
        "benchmark_manifest": config["benchmark_manifest"],
        "manifest_hash": manifest_hash,
        "base_policy": config["base_policy"],
        "base_policy_revision": config["base_policy_revision"],
        "dataset_repo_id": config["dataset_repo_id"],
        "dataset_revision": config["dataset_revision"],
        "runtime": runtime,
        "evaluation_seeds": evaluation,
        "counterfactual_seeds": counterfactual,
        "peft": peft,
    }
    run_hash = sha256(_canonical(identity)).hexdigest()[:16]
    run_dir = args.output_root / config["experiment_name"] / stage / run_hash
    already_exists = run_dir.exists()
    if already_exists and not args.resume:
        raise RuntimeError(f"run already exists; pass --resume: {run_dir}")
    if (run_dir / "COMPLETE").exists():
        raise RuntimeError(f"refusing to overwrite completed run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        **identity,
        "run_hash": run_hash,
        "git_commit": _git_commit(),
        "hf_namespace": config["hf_namespace"],
        "device": config["device"],
        "peft": peft,
        "status": "initialized",
    }
    command_body = STAGE_COMMANDS[stage].format(
        dataset=shlex.quote(config["dataset_repo_id"]),
        dataset_revision=shlex.quote(config["dataset_revision"]),
        hf_namespace=shlex.quote(config["hf_namespace"]),
        base_policy=shlex.quote(str(base_policy_snapshot)),
        base_policy_revision=shlex.quote(config["base_policy_revision"]),
        model_prefix=shlex.quote(config["model_prefix"]),
        job_name=shlex.quote(f"{config['model_prefix']}-{stage}"),
        device=shlex.quote(config["device"]),
        batch_size=int(config["batch_size"]),
        steps=int(config["steps"]),
        train_seed=int(config["train_seeds"][0]),
        save_freq=int(config["save_freq"]),
        env_eval_freq=int(config["env_eval_freq"]),
        peft_arguments=" ".join(render_peft_arguments(config)),
        push_to_hub=str(bool(config["push_to_hub"])).lower(),
        hub_repo_argument=(
            "--policy.repo_id="
            + shlex.quote(f"{config['hf_namespace']}/{config['model_prefix']}-client-adapter")
            if config["push_to_hub"]
            else ""
        ),
        seeds=",".join(str(seed) for seed in evaluation),
        run_dir=shlex.quote(str(run_dir)),
    )
    command = (
        "PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl " + command_body
    )
    _write_immutable(run_dir / "frozen_config.yaml", config_bytes)
    _write_immutable(run_dir / "run_manifest.json", _canonical(manifest))
    _write_immutable(run_dir / "command.sh", (command + "\n").encode())
    (run_dir / "logs").mkdir(exist_ok=True)

    result = {
        "stage": stage,
        "run_dir": str(run_dir),
        "run_hash": run_hash,
        "dry_run": args.dry_run,
        "resumed": already_exists and args.resume,
        "device": config["device"],
        "hf_namespace": config["hf_namespace"],
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(result, sort_keys=True))
        return

    if stage != "train_client_adapter":
        raise RuntimeError(
            f"{stage} backend is not implemented yet; use --dry-run to inspect the run plan"
        )

    with (run_dir / "logs" / "stdout.log").open("a") as stdout, (
        run_dir / "logs" / "stderr.log"
    ).open("a") as stderr:
        completed = subprocess.run(command, shell=True, stdout=stdout, stderr=stderr)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if stage == "train_client_adapter":
        _normalize_adapter_provenance(
            run_dir / "checkpoints" / "client-adapter",
            str(config["base_policy"]),
            str(config["base_policy_revision"]),
        )
        artifact = validate_adapter_run(
            run_dir / "checkpoints" / "client-adapter",
            log_path=run_dir / "logs" / "stderr.log",
            expected_rank=peft["rank"],
            expected_alpha=peft["alpha"],
            expected_step=int(config["steps"]),
        )
        metadata_bytes = (artifact.to_json() + "\n").encode()
        _write_immutable(run_dir / "adapter_metadata.json", metadata_bytes)
        _write_immutable(
            run_dir / "COMPLETE",
            _completion_record(
                run_hash,
                metadata_bytes,
                artifact.adapter_sha256,
                sha256(
                    (
                        run_dir
                        / "checkpoints"
                        / "client-adapter"
                        / "checkpoints"
                        / f"{int(config['steps']):06d}"
                        / "pretrained_model"
                        / "adapter_config.json"
                    ).read_bytes()
                ).hexdigest(),
            ),
        )
    print(json.dumps(result, sort_keys=True))
