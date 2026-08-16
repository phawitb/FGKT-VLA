import json
from pathlib import Path
import subprocess
import sys

from tests.integration.test_verify_adapter_run import _write_run


def test_record_cli_dry_run_binds_verified_adapter_without_importing_libero(tmp_path):
    run = _write_run(tmp_path)
    output = tmp_path / "episode.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/record_libero_episode.py",
            "--adapter-run",
            str(run),
            "--output",
            str(output),
            "--suite",
            "libero_spatial",
            "--task-id",
            "0",
            "--seed",
            "101",
            "--initial-state-index",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["policy_revision"]
    assert payload["policy_repo"] == "phawitbinabik/fgkt-vla-adapters"
    assert payload["task_alias"] == "libero_spatial.task0"
    assert payload["checkpoint"].endswith("000010/pretrained_model")
    assert payload["output_path"] == str(output)
    assert not output.exists()
