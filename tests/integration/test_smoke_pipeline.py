import json
import subprocess
import sys
from pathlib import Path


def _run_smoke(output_root: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_test.py",
            "--config",
            "configs/smoke.yaml",
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_smoke_pipeline_is_deterministic_and_writes_immutable_artifacts(tmp_path):
    first = _run_smoke(tmp_path / "one")
    second = _run_smoke(tmp_path / "two")

    assert first["run_hash"] == second["run_hash"]
    assert first["metrics_sha256"] == second["metrics_sha256"]
    run_dir = Path(first["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["clients"] == ["positive", "neutral", "harmful"]
    assert metrics["selected_adapters"] == ["positive"]
    assert metrics["gate_promoted"] is True
    assert (run_dir / "COMPLETE").exists()
