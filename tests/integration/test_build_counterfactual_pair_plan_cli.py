import json

import scripts.build_counterfactual_pair_plan as cli
from tests.integration.test_build_repair_candidate_cli import _artifacts


def test_counterfactual_pair_plan_is_eligible_and_resume_safe(tmp_path, monkeypatch, capsys):
    source_run, episode, shard, _ = _artifacts(tmp_path)
    source_metadata = json.loads((source_run / "adapter_metadata.json").read_text())
    source_manifest = json.loads((source_run / "run_manifest.json").read_text())
    episode_record = cli.validate_episode_shard(episode)[0]
    failure = json.loads((shard / "failures.jsonl").read_text())
    failure_hash = cli.sha256(cli._canonical(failure["context"])).hexdigest()
    recipe = {
        "content_hash": "recipe-sha",
        "source_adapter_sha256": source_metadata["adapter_sha256"],
        "failure_hash": failure_hash,
        "source_run_hash": source_manifest["run_hash"],
        "source_episode_sha256": episode_record.content_hash(),
        "task_alias": episode_record.key.task_alias,
        "problem_id": episode_record.problem_id,
        "instruction": episode_record.instruction,
        "dataset_repo": source_manifest["dataset_repo_id"],
        "dataset_revision": source_manifest["dataset_revision"],
        "base_policy_repo": source_manifest["base_policy"],
        "base_policy_revision": source_manifest["base_policy_revision"],
        "lerobot_commit": source_manifest["runtime"]["lerobot_commit"],
        "peft_version": source_manifest["runtime"]["peft_version"],
        "hf_libero_version": "0.1.4",
    }
    candidate = {
        "recipe_sha256": "recipe-sha",
        "source_adapter_sha256": source_metadata["adapter_sha256"],
        "candidate_adapter_sha256": "candidate-sha",
    }
    monkeypatch.setattr(
        cli, "_verify_candidate_run", lambda *args: (recipe, candidate)
    )
    monkeypatch.setattr(cli, "verified_hf_libero_version", lambda: "0.1.4")
    output = tmp_path / "counterfactual-pair.json"
    arguments = [
        "--source-adapter-run", str(source_run),
        "--candidate-run", str(tmp_path / "candidate"),
        "--episode", str(episode),
        "--failure-shard", str(shard),
        "--output", str(output),
    ]

    cli.main(arguments)
    first = output.read_bytes()
    cli.main([*arguments, "--resume"])

    plan = json.loads(first)
    assert output.read_bytes() == first
    assert plan["purpose"] == "counterfactual_repair"
    assert plan["label_eligible"] is True
    assert plan["baseline_policy_revision"] == source_metadata["adapter_sha256"]
    assert plan["pairs"][0]["candidate"]["policy_revision"] == "candidate-sha"
    capsys.readouterr()
