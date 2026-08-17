import json
from pathlib import Path
import subprocess
import sys

from tests.libero.test_episode import episode_mapping


def test_failure_shard_cli_validates_episode_and_is_resume_safe(tmp_path):
    episode = tmp_path / "episode.json"
    episode.write_text(json.dumps(episode_mapping(success=False)) + "\n")
    output = tmp_path / "failure-shard"
    command = [
        sys.executable,
        "scripts/build_failure_shard.py",
        "--episode",
        str(episode),
        "--output-dir",
        str(output),
        "--window-size",
        "16",
    ]

    first = subprocess.run(command, capture_output=True, text=True)
    resumed = subprocess.run([*command, "--resume"], capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert resumed.returncode == 0, resumed.stderr
    report = json.loads(first.stdout)
    assert report == json.loads(resumed.stdout)
    assert report["failure_count"] == 1
    assert report["window_size"] == 16
    assert (output / "failures.jsonl").is_file()
    assert (output / "failures.sha256").is_file()
    assert (output / "SHARD.json").is_file()
