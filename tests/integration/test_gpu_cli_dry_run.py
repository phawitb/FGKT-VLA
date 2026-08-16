import json
import subprocess
import sys
from pathlib import Path

import pytest


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
