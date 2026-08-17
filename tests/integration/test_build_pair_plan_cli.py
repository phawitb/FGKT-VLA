import json
from pathlib import Path
import subprocess
import sys
import os

from scripts.verify_adapter_run import verify_run
from tests.integration.test_verify_adapter_run import _write_run
from tests.libero.test_episode import episode_mapping
from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.episode import EpisodeRecord


def test_pair_plan_cli_binds_episode_failure_adapter_and_base_policy(tmp_path):
    run = _write_run(tmp_path)
    report = verify_run(run)
    raw = episode_mapping(success=False)
    raw["task_alias"] = "libero_spatial.task0"
    raw["policy_repo"] = "phawitbinabik/fgkt-vla-adapters"
    raw["policy_revision"] = report["adapter_sha256"]
    episode_record = EpisodeRecord.from_mapping(raw)
    episode = tmp_path / "episode.json"
    episode.write_text(episode_record.to_json() + "\n")
    failure_shard = tmp_path / "failure-shard"
    write_failure_shard((episode_record,), output_dir=failure_shard, window_size=16)
    output = tmp_path / "pair-plan.json"
    command = [
        sys.executable,
        "scripts/build_pair_plan.py",
        "--episode",
        str(episode),
        "--failure-shard",
        str(failure_shard),
        "--adapter-run",
        str(run),
        "--output",
        str(output),
    ]
    fake_site = tmp_path / "fake-site"
    metadata = fake_site / "hf_libero-0.1.4.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: hf-libero\nVersion: 0.1.4\n")
    env = {**os.environ, "PYTHONPATH": f"{fake_site}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}

    first = subprocess.run(command, capture_output=True, text=True, env=env)
    resumed = subprocess.run([*command, "--resume"], capture_output=True, text=True, env=env)

    assert first.returncode == 0, first.stderr
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads(first.stdout)
    plan = json.loads(output.read_text())
    assert summary == json.loads(resumed.stdout)
    assert summary["pair_count"] == 1
    assert plan["purpose"] == "plumbing_self_replay"
    assert plan["label_eligible"] is False
    assert plan["baseline_policy_repo"] == "lerobot/smolvla_base"
    assert plan["baseline_policy_revision"] == "base-sha"
    assert plan["pairs"][0]["candidate"]["policy_revision"] == report["adapter_sha256"]
    assert plan["pairs"][0]["key"]["initial_state_hash"] == episode_record.initial_state_hash
    assert plan["environment"]["max_steps"] == 280
    assert plan["environment"]["fps"] == 20
    assert plan["environment"]["observation_shape"] == [360, 360]
    assert plan["environment"]["lerobot_commit"] == "lerobot-sha"
    assert plan["environment"]["hf_libero_version"] == "0.1.4"
