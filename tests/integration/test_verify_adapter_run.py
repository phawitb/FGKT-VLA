from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from safetensors.torch import save_file
import torch


def _rewrite_completion(run: Path) -> None:
    metadata_bytes = (run / "adapter_metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    manifest = json.loads((run / "run_manifest.json").read_text())
    adapter_config = next(run.rglob("adapter_config.json"))
    payload = {
        "schema_version": 1,
        "run_hash": manifest["run_hash"],
        "adapter_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "adapter_sha256": metadata["adapter_sha256"],
        "adapter_config_sha256": sha256(adapter_config.read_bytes()).hexdigest(),
    }
    completion = {
        **payload,
        "completion_sha256": sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (run / "COMPLETE").write_text(
        json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n"
    )

def _write_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "client-adapter" / "checkpoints" / "000010" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "adapter_model.safetensors"
    save_file(
        {"base_model.model.model.expert.q_proj.lora_A.default.weight": torch.ones(1, 1)},
        weights,
    )
    training_state = checkpoint.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text('{"step":10}')
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "base_model_name_or_path": "lerobot/smolvla_base",
                "revision": "base-sha",
                "target_modules": r"model\.expert\.q_proj",
            }
        )
    )
    frozen_config = b"steps: 10\n"
    (run / "frozen_config.yaml").write_bytes(frozen_config)
    benchmark_manifest = tmp_path / "benchmark.yaml"
    benchmark_manifest.write_text("schema_version: 1\n")
    identity = {
        "schema_version": 1,
        "stage": "train_client_adapter",
        "config_sha256": sha256(frozen_config).hexdigest(),
        "benchmark_manifest": str(benchmark_manifest),
        "manifest_hash": sha256(benchmark_manifest.read_bytes()).hexdigest(),
        "base_policy": "lerobot/smolvla_base",
        "base_policy_revision": "base-sha",
        "dataset_repo_id": "lerobot/libero_spatial_image",
        "dataset_revision": "dataset-sha",
        "runtime": {"lerobot_commit": "lerobot-sha", "peft_version": "0.20.0"},
        "evaluation_seeds": [101],
        "counterfactual_seeds": [101],
        "peft": {
            "method_type": "LORA",
            "rank": 8,
            "alpha": 16,
            "target_modules": "native_default",
        },
    }
    run_hash = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    metadata = {
        "schema_version": 1,
        "peft_type": "LORA",
        "rank": 8,
        "alpha": 16,
        "adapter_bytes": weights.stat().st_size,
        "adapter_sha256": sha256(weights.read_bytes()).hexdigest(),
        "trainable_parameters": 4_000_000,
        "total_parameters": 450_000_000,
        "trainable_ratio": 4_000_000 / 450_000_000,
        "completed_step": 10,
        "resolved_targets": ["base_model.model.model.expert.q_proj"],
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (run / "adapter_metadata.json").write_bytes(metadata_bytes)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                **identity,
                "run_hash": run_hash,
                "device": "mps",
            }
        )
    )
    _rewrite_completion(run)
    return run


def _verify(run: Path):
    return subprocess.run(
        [sys.executable, "scripts/verify_adapter_run.py", "--run-dir", str(run)],
        capture_output=True,
        text=True,
    )


def test_verify_run_accepts_matching_complete_adapter_metadata(tmp_path):
    result = _verify(_write_run(tmp_path))

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["adapter_sha256"]
    assert report["trainable_ratio"] < 0.10
    assert report["resolved_target_count"] == 1


def test_verify_run_detects_adapter_tampering(tmp_path):
    run = _write_run(tmp_path)
    next(run.rglob("adapter_model.safetensors")).write_bytes(b"changed")

    result = _verify(run)

    assert result.returncode != 0
    assert "sha256" in result.stderr.lower()


def test_verify_run_rejects_missing_or_mismatched_complete_marker(tmp_path):
    run = _write_run(tmp_path)
    completion = json.loads((run / "COMPLETE").read_text())
    completion["run_hash"] = "wrong-hash"
    (run / "COMPLETE").write_text(json.dumps(completion))

    result = _verify(run)

    assert result.returncode != 0
    assert "complete" in result.stderr.lower()


def test_verify_run_detects_coordinated_weight_and_metadata_tampering(tmp_path):
    run = _write_run(tmp_path)
    weights = next(run.rglob("adapter_model.safetensors"))
    weights.write_bytes(b"replacement-adapter")
    metadata_path = run / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["adapter_bytes"] = weights.stat().st_size
    metadata["adapter_sha256"] = sha256(weights.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))

    result = _verify(run)

    assert result.returncode != 0
    assert "complete" in result.stderr.lower()


def test_verify_run_recomputes_parameter_ratio(tmp_path):
    run = _write_run(tmp_path)
    metadata_path = run / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["trainable_ratio"] = 0.001
    metadata_path.write_text(json.dumps(metadata))
    _rewrite_completion(run)

    result = _verify(run)

    assert result.returncode != 0
    assert "ratio" in result.stderr.lower()


def test_verify_run_recomputes_resolved_targets(tmp_path):
    run = _write_run(tmp_path)
    metadata_path = run / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["resolved_targets"] = ["fabricated.target"]
    metadata_path.write_text(json.dumps(metadata))
    _rewrite_completion(run)

    result = _verify(run)

    assert result.returncode != 0
    assert "target" in result.stderr.lower()


def test_verify_run_requires_latest_requested_training_step(tmp_path):
    run = _write_run(tmp_path)
    training_step = next(run.rglob("training_step.json"))
    training_step.write_text('{"step":9}')

    result = _verify(run)

    assert result.returncode != 0
    assert "step" in result.stderr.lower()


def test_verify_run_rejects_adapter_target_config_tampering(tmp_path):
    run = _write_run(tmp_path)
    config_path = next(run.rglob("adapter_config.json"))
    config = json.loads(config_path.read_text())
    config["target_modules"] = "unrelated\\.module"
    config_path.write_text(json.dumps(config))
    _rewrite_completion(run)

    result = _verify(run)

    assert result.returncode != 0
    assert "target" in result.stderr.lower()


def test_verify_run_rejects_broadened_adapter_target_regex(tmp_path):
    run = _write_run(tmp_path)
    config_path = next(run.rglob("adapter_config.json"))
    config = json.loads(config_path.read_text())
    config["target_modules"] = ".*"
    config_path.write_text(json.dumps(config))

    result = _verify(run)

    assert result.returncode != 0
    assert "adapter config" in result.stderr.lower()


def test_verify_run_detects_frozen_config_tampering(tmp_path):
    run = _write_run(tmp_path)
    (run / "frozen_config.yaml").write_text("steps: 11\n")

    result = _verify(run)

    assert result.returncode != 0
    assert "config" in result.stderr.lower()


def test_verify_run_detects_benchmark_manifest_tampering(tmp_path):
    run = _write_run(tmp_path)
    manifest = json.loads((run / "run_manifest.json").read_text())
    Path(manifest["benchmark_manifest"]).write_text("schema_version: 2\n")

    result = _verify(run)

    assert result.returncode != 0
    assert "benchmark manifest" in result.stderr.lower()
