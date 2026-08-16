#!/usr/bin/env python3
"""Verify a completed immutable FGKT-VLA client adapter run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import yaml

from fcut_vla.adapters.artifact import resolved_lora_targets


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _targets_match_config(configured: object, actual: list[str]) -> bool:
    normalized = [target.removeprefix("base_model.model.") for target in actual]
    if isinstance(configured, str) and configured:
        try:
            expression = re.compile(configured)
        except re.error:
            return False
        return all(expression.fullmatch(target) is not None for target in normalized)
    if isinstance(configured, list) and configured:
        suffixes = [str(target) for target in configured if str(target)]
        return (
            bool(suffixes)
            and all(any(target.endswith(suffix) for suffix in suffixes) for target in normalized)
            and all(any(target.endswith(suffix) for target in normalized) for suffix in suffixes)
        )
    return False


def verify_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = _read_object(run_dir / "run_manifest.json", "run manifest")
    metadata = _read_object(run_dir / "adapter_metadata.json", "adapter metadata")
    frozen_config = run_dir / "frozen_config.yaml"
    if not frozen_config.is_file():
        raise ValueError(f"missing frozen config: {frozen_config}")
    config_hash = sha256(frozen_config.read_bytes()).hexdigest()
    if config_hash != manifest.get("config_sha256"):
        raise ValueError("frozen config SHA256 does not match run manifest")
    frozen = yaml.safe_load(frozen_config.read_bytes())
    if not isinstance(frozen, dict):
        raise ValueError("frozen config must contain a mapping")
    benchmark_manifest = Path(str(manifest.get("benchmark_manifest", "")))
    if not benchmark_manifest.is_file():
        raise ValueError(f"missing benchmark manifest: {benchmark_manifest}")
    benchmark_hash = sha256(benchmark_manifest.read_bytes()).hexdigest()
    if benchmark_hash != manifest.get("manifest_hash"):
        raise ValueError("benchmark manifest SHA256 does not match run manifest")

    identity_keys = (
        "schema_version",
        "stage",
        "config_sha256",
        "benchmark_manifest",
        "manifest_hash",
        "base_policy",
        "base_policy_revision",
        "dataset_repo_id",
        "dataset_revision",
        "runtime",
        "evaluation_seeds",
        "counterfactual_seeds",
        "peft",
    )
    identity = {key: manifest[key] for key in identity_keys}
    expected_run_hash = sha256(_canonical(identity)).hexdigest()[:16]
    run_hash = str(manifest.get("run_hash", ""))
    if run_hash != expected_run_hash:
        raise ValueError("run hash does not match immutable manifest identity")
    complete = _read_object(run_dir / "COMPLETE", "COMPLETE marker")
    completion_payload = {
        "schema_version": complete.get("schema_version"),
        "run_hash": complete.get("run_hash"),
        "adapter_metadata_sha256": complete.get("adapter_metadata_sha256"),
        "adapter_sha256": complete.get("adapter_sha256"),
        "adapter_config_sha256": complete.get("adapter_config_sha256"),
    }
    if complete.get("completion_sha256") != sha256(_canonical(completion_payload)).hexdigest():
        raise ValueError("COMPLETE marker digest is invalid")
    if complete.get("run_hash") != run_hash:
        raise ValueError("COMPLETE marker does not match run hash")
    metadata_hash = sha256((run_dir / "adapter_metadata.json").read_bytes()).hexdigest()
    if complete.get("adapter_metadata_sha256") != metadata_hash:
        raise ValueError("COMPLETE marker does not match adapter metadata")

    step = int(metadata["completed_step"])
    if step != int(frozen.get("steps", -1)):
        raise ValueError("completed step does not match frozen config")
    checkpoint_root = run_dir / "checkpoints" / "client-adapter" / "checkpoints"
    numeric_steps = sorted(
        int(path.name) for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if not numeric_steps or numeric_steps[-1] != step:
        raise ValueError("completed step is not the latest numeric checkpoint")
    checkpoint = (
        run_dir
        / "checkpoints"
        / "client-adapter"
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )
    weights = checkpoint / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError(f"missing adapter weights: {weights}")
    actual_hash = sha256(weights.read_bytes()).hexdigest()
    if complete.get("adapter_sha256") != actual_hash:
        raise ValueError("COMPLETE marker adapter SHA256 does not match adapter weights")
    if actual_hash != metadata.get("adapter_sha256"):
        raise ValueError("adapter SHA256 does not match adapter metadata")
    if weights.stat().st_size != int(metadata.get("adapter_bytes", -1)):
        raise ValueError("adapter byte size does not match adapter metadata")
    adapter_config = _read_object(checkpoint / "adapter_config.json", "adapter config")
    adapter_config_hash = sha256((checkpoint / "adapter_config.json").read_bytes()).hexdigest()
    if complete.get("adapter_config_sha256") != adapter_config_hash:
        raise ValueError("COMPLETE marker does not match adapter config")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter config peft_type must be LORA")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise ValueError("adapter config modules_to_save must be empty")
    if adapter_config.get("base_model_name_or_path") != manifest.get("base_policy"):
        raise ValueError("adapter base policy does not match run manifest")
    if adapter_config.get("revision") != manifest.get("base_policy_revision"):
        raise ValueError("adapter base revision does not match run manifest")
    training_step = _read_object(
        checkpoint.parent / "training_state" / "training_step.json", "training step"
    )
    if int(training_step.get("step", -1)) != step:
        raise ValueError("training step does not match completed step")
    trainable = int(metadata["trainable_parameters"])
    total = int(metadata["total_parameters"])
    if trainable <= 0 or total <= 0:
        raise ValueError("parameter counts must be positive")
    expected_ratio = trainable / total
    ratio = float(metadata["trainable_ratio"])
    if ratio != expected_ratio:
        raise ValueError("adapter trainable ratio does not match parameter counts")
    if not 0 < ratio < 0.10:
        raise ValueError("adapter trainable ratio must be between zero and 10%")
    targets = metadata.get("resolved_targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("adapter metadata requires resolved targets")
    actual_targets = list(resolved_lora_targets(weights))
    if targets != actual_targets:
        raise ValueError("resolved target list does not match adapter weights")
    if not _targets_match_config(adapter_config.get("target_modules"), actual_targets):
        raise ValueError("adapter target configuration does not match resolved targets")
    peft = manifest.get("peft", {})
    if int(peft.get("rank", -1)) != int(metadata["rank"]):
        raise ValueError("adapter rank does not match run manifest")
    if int(peft.get("alpha", -1)) != int(metadata["alpha"]):
        raise ValueError("adapter alpha does not match run manifest")
    if int(adapter_config.get("r", -1)) != int(metadata["rank"]):
        raise ValueError("adapter rank does not match adapter config")
    if int(adapter_config.get("lora_alpha", -1)) != int(metadata["alpha"]):
        raise ValueError("adapter alpha does not match adapter config")
    return {
        "valid": True,
        "run_hash": run_hash,
        "device": manifest.get("device"),
        "completed_step": step,
        "adapter_sha256": actual_hash,
        "adapter_bytes": weights.stat().st_size,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": ratio,
        "resolved_target_count": len(targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify_run(args.run_dir)
    except (KeyError, TypeError, ValueError) as error:
        print(f"adapter run verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
