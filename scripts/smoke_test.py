#!/usr/bin/env python3
"""Deterministic CPU-only end-to-end smoke test for FCUT-VLA."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fcut_vla.config import load_config
from fcut_vla.repair.consolidation import evaluate_gate, promote_with_rollback
from fcut_vla.repair.personalized import build_personalized_repair
from fcut_vla.utility.ranker import UtilityPrediction


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run(config_path: Path, output_root: Path) -> dict[str, str]:
    config = load_config(config_path)
    clients = ("positive", "neutral", "harmful")
    target = {"adapter": torch.tensor([0.0])}
    adapters = {
        "harmful": {"adapter": torch.tensor([-1.0])},
        "neutral": {"adapter": torch.tensor([0.0])},
        "positive": {"adapter": torch.tensor([1.0])},
    }
    lower = torch.tensor([[-0.30, 0.0, 0.40]])
    prediction = UtilityPrediction(
        gain=torch.tensor([[-0.20, 0.05, 0.50]]),
        risk=torch.tensor([[0.10, 0.05, 0.02]]),
        uncertainty=torch.tensor([[0.10, 0.05, 0.08]]),
        utility=torch.tensor([[-0.20, 0.05, 0.48]]),
        lower_bound=lower,
        valid_mask=torch.ones_like(lower, dtype=torch.bool),
    )
    repair = build_personalized_repair(
        target, adapters, prediction, top_k=1, coefficient_budget=0.5
    )
    gate = evaluate_gate(
        gain_lower=0.12,
        retention_risk_upper={"prior-a": 0.01, "prior-b": 0.02},
        min_gain=0.10,
        max_average_risk=0.02,
        max_worst_task_risk=0.03,
    )
    promoted = promote_with_rollback(target, repair.state_dict, gate)
    metrics = {
        "failure_requests": 1,
        "selected_adapters": list(repair.selected_adapter_ids),
        "abstained": repair.abstained,
        "repair_value": float(promoted["adapter"].item()),
        "gain_lower": gate.gain_lower,
        "average_risk_upper": gate.average_risk_upper,
        "worst_task_risk_upper": gate.worst_task_risk_upper,
        "gate_promoted": gate.promote,
    }
    metrics_bytes = _canonical_bytes(metrics)
    metrics_hash = sha256(metrics_bytes).hexdigest()
    manifest_core = {
        "schema_version": 1,
        "config_hash": config.config_hash(),
        "clients": list(clients),
        "evaluation_seeds": list(config.evaluation_seeds.values),
        "metrics_sha256": metrics_hash,
        "device": "cpu",
    }
    run_hash = sha256(_canonical_bytes(manifest_core)).hexdigest()[:16]
    run_dir = output_root / config.experiment_name / run_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metrics.json": metrics_bytes,
        "manifest.json": _canonical_bytes({**manifest_core, "run_hash": run_hash}),
        "COMPLETE": (run_hash + "\n").encode("utf-8"),
    }
    for name, content in artifacts.items():
        path = run_dir / name
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        path.write_bytes(content)
    return {"run_dir": str(run_dir), "run_hash": run_hash, "metrics_sha256": metrics_hash}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
