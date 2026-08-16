"""Shared immutable orchestration for RTX GPU entry points."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


STAGE_COMMANDS = {
    "train_client_adapter": (
        "lerobot-train --policy.type=smolvla --dataset.repo_id={dataset} "
        "--policy.push_to_hub=true --policy.repo_id={hf_namespace}/{model_prefix}-client-adapter "
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


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    if not path.exists():
        path.write_bytes(content)


def run_stage(stage: str) -> None:
    parser = argparse.ArgumentParser(description=f"FCUT-VLA GPU stage: {stage}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    evaluation = [int(seed) for seed in config["evaluation_seeds"]]
    counterfactual = [int(seed) for seed in config["counterfactual_seeds"]]
    if evaluation != counterfactual:
        raise ValueError("evaluation and counterfactual seeds must be exactly paired")

    identity = {
        "schema_version": 1,
        "stage": stage,
        "config_sha256": sha256(config_bytes).hexdigest(),
        "benchmark_manifest": config["benchmark_manifest"],
        "evaluation_seeds": evaluation,
        "counterfactual_seeds": counterfactual,
    }
    run_hash = sha256(_canonical(identity)).hexdigest()[:16]
    run_dir = args.output_root / config["experiment_name"] / stage / run_hash
    already_exists = run_dir.exists()
    if already_exists and not args.resume:
        raise RuntimeError(f"run already exists; pass --resume: {run_dir}")
    if (run_dir / "COMPLETE").exists():
        raise RuntimeError(f"refusing to overwrite completed run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(config["benchmark_manifest"])
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    manifest = {
        **identity,
        "run_hash": run_hash,
        "manifest_hash": manifest_hash,
        "git_commit": _git_commit(),
        "hf_namespace": config["hf_namespace"],
        "device": "cuda",
        "status": "planned" if args.dry_run else "running",
    }
    command_body = STAGE_COMMANDS[stage].format(
        dataset=shlex.quote(config["dataset_repo_id"]),
        hf_namespace=shlex.quote(config["hf_namespace"]),
        model_prefix=shlex.quote(config["model_prefix"]),
        seeds=",".join(str(seed) for seed in evaluation),
        run_dir=shlex.quote(str(run_dir)),
    )
    command = f"MUJOCO_GL=egl {command_body}"
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
        "device": "cuda",
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
    (run_dir / "COMPLETE").write_text(run_hash + "\n")
    print(json.dumps(result, sort_keys=True))
