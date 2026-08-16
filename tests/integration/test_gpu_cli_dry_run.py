import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts._gpu_job import (
    _resolve_base_policy,
    _runtime_provenance,
    _validate_training_checkpoint,
    render_peft_arguments,
)


SCRIPTS = (
    "train_client_adapter.py",
    "generate_failures.py",
    "build_utility_labels.py",
    "train_utility_ranker.py",
    "run_repair_eval.py",
    "run_continual_experiment.py",
)


def _run(script: str, output_root: Path, *extra: str, check: bool = True):
    return subprocess.run(
        [
            sys.executable,
            f"scripts/{script}",
            "--config",
            "configs/gpu/development.yaml",
            "--output-root",
            str(output_root),
            "--dry-run",
            *extra,
        ],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_each_gpu_entrypoint_renders_without_importing_cuda(script, tmp_path):
    completed = _run(script, tmp_path)
    result = json.loads(completed.stdout)

    assert result["dry_run"] is True
    assert result["device"] == "cuda"
    assert result["hf_namespace"] == "phawitbinabik"
    assert "MUJOCO_GL=egl" in result["command"]
    assert "PYTHONNOUSERSITE=1" in result["command"]
    assert "PYTHONUNBUFFERED=1" in result["command"]
    assert "causalvla" not in result["command"].lower()
    assert Path(result["run_dir"], "frozen_config.yaml").exists()
    assert Path(result["run_dir"], "run_manifest.json").exists()


def test_hash_and_paired_seeds_are_stable_across_output_roots(tmp_path):
    first = json.loads(_run("run_repair_eval.py", tmp_path / "a").stdout)
    second = json.loads(_run("run_repair_eval.py", tmp_path / "b").stdout)
    manifest = json.loads(Path(first["run_dir"], "run_manifest.json").read_text())

    assert first["run_hash"] == second["run_hash"]
    assert manifest["evaluation_seeds"] == [101, 102, 103, 104]
    assert manifest["counterfactual_seeds"] == [101, 102, 103, 104]
    assert manifest["manifest_hash"]
    assert manifest["git_commit"]


def test_resume_reuses_incomplete_run_and_completed_run_is_immutable(tmp_path):
    initial = json.loads(_run("generate_failures.py", tmp_path).stdout)
    resumed = json.loads(_run("generate_failures.py", tmp_path, "--resume").stdout)
    assert resumed["run_dir"] == initial["run_dir"]
    assert resumed["resumed"] is True

    Path(initial["run_dir"], "COMPLETE").write_text("done\n")
    rejected = _run("generate_failures.py", tmp_path, "--resume", check=False)
    assert rejected.returncode != 0
    assert "completed run" in rejected.stderr.lower()


def test_missing_required_arguments_are_rejected():
    completed = subprocess.run(
        [sys.executable, "scripts/train_client_adapter.py", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--config" in completed.stderr


def test_training_command_is_an_independent_bounded_smolvla_smoke(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_client_adapter.py",
            "--config",
            "configs/local/smolvla_smoke.yaml",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    command = result["command"]
    manifest = json.loads(Path(result["run_dir"], "run_manifest.json").read_text())

    assert "python -m lerobot.scripts.lerobot_train" in command
    policy_path_arg = next(
        argument for argument in shlex.split(command) if argument.startswith("--policy.path=")
    )
    policy_snapshot = Path(policy_path_arg.split("=", 1)[1])
    assert policy_snapshot.is_dir()
    assert policy_snapshot.name == "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    assert "--policy.pretrained_revision=c83c3163b8ca9b7e67c509fffd9121e66cb96205" in command
    assert "--dataset.revision=d86c0b94922572b3b657e1d1a3d01f0952ddeb46" in command
    assert "--policy.type=smolvla" not in command
    assert "--policy.input_features=null" in command
    assert "--policy.output_features=null" in command
    assert "--policy.device=mps" in command
    assert "--batch_size=2" in command
    assert "--steps=10" in command
    assert "--save_freq=5" in command
    assert "--env_eval_freq=0" in command
    assert "--policy.push_to_hub=false" in command
    assert "--peft.method_type=LORA" in command
    assert "--peft.r=8" in command
    assert "--peft.lora_alpha=16" in command
    assert "--peft.full_training_modules=[]" in command
    assert "--peft.target_modules" not in command
    assert "causalvla" not in command.lower()
    assert manifest["status"] == "initialized"
    assert manifest["device"] == "mps"
    assert manifest["base_policy"] == "lerobot/smolvla_base"
    assert manifest["base_policy_revision"] == "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    assert manifest["dataset_revision"] == "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"
    assert manifest["runtime"] == {
        "lerobot_commit": "7e241bd630a3719a56157a497ce5d08f244784f1",
        "peft_version": "0.20.0",
    }
    assert manifest["peft"] == {
        "alpha": 16,
        "method_type": "LORA",
        "rank": 8,
        "target_modules": "native_default",
    }


@pytest.mark.parametrize(
    "lora, message",
    [
        ({"rank": 0, "alpha": 16, "target_modules": "native_default"}, "rank"),
        ({"rank": 8, "alpha": 0, "target_modules": "native_default"}, "alpha"),
        ({"rank": 8, "alpha": 16, "target_modules": []}, "target_modules"),
    ],
)
def test_invalid_lora_configuration_is_rejected(lora, message):
    with pytest.raises(ValueError, match=message):
        render_peft_arguments({"lora": lora})


def test_training_completion_requires_loadable_checkpoint_files(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "000010" / "pretrained_model"
    checkpoint.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="model.safetensors"):
        _validate_training_checkpoint(tmp_path)

    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    assert _validate_training_checkpoint(tmp_path) == checkpoint


def test_base_policy_snapshot_resolution_uses_frozen_revision(monkeypatch, tmp_path):
    observed = {}

    def fake_snapshot_download(**kwargs):
        observed.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr("scripts._gpu_job.snapshot_download", fake_snapshot_download)
    resolved = _resolve_base_policy(
        {
            "base_policy": "lerobot/smolvla_base",
            "base_policy_revision": "base-commit-sha",
        }
    )

    assert resolved == tmp_path / "snapshot"
    assert observed == {
        "repo_id": "lerobot/smolvla_base",
        "revision": "base-commit-sha",
    }


def test_runtime_provenance_rejects_dirty_lerobot_checkout(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(stdout="lerobot-sha\n"),
            SimpleNamespace(stdout=" M src/lerobot/policies/factory.py\n"),
        ]
    )
    monkeypatch.setattr("scripts._gpu_job.subprocess.run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("scripts._gpu_job.version", lambda package: "0.20.0")

    with pytest.raises(RuntimeError, match="dirty"):
        _runtime_provenance(
            {"lerobot_commit": "lerobot-sha", "peft_version": "0.20.0"}
        )
