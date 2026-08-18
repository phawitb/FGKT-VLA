import json
from pathlib import Path

import yaml
import pytest

import scripts.build_repair_candidate as cli
from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.episode import EpisodeRecord
from fcut_vla.repair.continuation import DatasetEpisode
from tests.integration.test_verify_adapter_run import _write_run
from tests.libero.test_episode import episode_mapping


def _artifacts(tmp_path):
    run = _write_run(tmp_path)
    checkpoint = next(run.rglob("pretrained_model"))
    (checkpoint / "train_config.json").write_text("{}")
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "policy_preprocessor.json").write_text(
        '{"kind":"pre","tokenizer_name":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"}'
    )
    (checkpoint / "policy_postprocessor.json").write_text('{"kind":"post"}')
    (checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(
        b"normalizer"
    )
    (checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        b"unnormalizer"
    )
    raw = episode_mapping(success=False)
    raw["task_alias"] = "libero_spatial.task0"
    raw["policy_repo"] = "phawitbinabik/fgkt-vla-adapters"
    metadata = json.loads((run / "adapter_metadata.json").read_text())
    raw["policy_revision"] = metadata["adapter_sha256"]
    raw["instruction"] = "put the black bowl on the plate"
    episode_record = EpisodeRecord.from_mapping(raw)
    episode = tmp_path / "episode.json"
    episode.write_text(episode_record.to_json() + "\n")
    shard = tmp_path / "failure-shard"
    write_failure_shard((episode_record,), output_dir=shard, window_size=16)
    config = tmp_path / "repair.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "recipe_type": "failure_aligned_continuation_v1",
                "continuation_steps": 20,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "training_seed": 2026,
                "save_frequency": 10,
                "device": "cuda",
                "tokenizer_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "tokenizer_revision": "tokenizer-sha",
            },
            sort_keys=True,
        )
    )
    return run, episode, shard, config


def test_repair_candidate_cli_dry_run_freezes_task_episode_recipe(tmp_path, monkeypatch, capsys):
    run, episode, shard, config = _artifacts(tmp_path)
    monkeypatch.setattr(
        cli,
        "_load_dataset_episodes",
        lambda repo, revision: (
            DatasetEpisode(8, ("put the black bowl on the plate",)),
            DatasetEpisode(3, ("put the black bowl on the plate",)),
            DatasetEpisode(4, ("other task",)),
        ),
    )
    monkeypatch.setattr(cli, "verified_hf_libero_version", lambda: "0.1.4")
    monkeypatch.setattr(
        cli,
        "_resolve_tokenizer_identity",
        lambda path, repo, revision: (repo, revision),
    )
    monkeypatch.setattr(
        cli,
        "_verified_source_runtime",
        lambda manifest: manifest["runtime"],
    )
    output_root = tmp_path / "output"
    arguments = [
        "--source-adapter-run",
        str(run),
        "--episode",
        str(episode),
        "--failure-shard",
        str(shard),
        "--config",
        str(config),
        "--output-root",
        str(output_root),
        "--dry-run",
    ]

    cli.main(arguments)
    first = json.loads(capsys.readouterr().out)
    cli.main([*arguments, "--resume"])
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["episode_indices"] == [3, 8]
    assert "--dataset.episodes=[3,8]" in first["command"]
    run_dir = Path(first["run_dir"])
    assert json.loads((run_dir / "repair_recipe.json").read_text())["episode_indices"] == [3, 8]
    assert not list(run_dir.glob("**/adapter_model.safetensors"))


def test_resume_quarantines_an_incomplete_training_attempt(tmp_path):
    run_dir = tmp_path / "run"
    attempt = run_dir / ".training-attempt"
    attempt.mkdir(parents=True)
    (attempt / "partial.bin").write_bytes(b"partial")

    cli._prepare_training_attempt(run_dir, attempt, resume=True)

    assert not attempt.exists()
    quarantined = list((run_dir / "failed-attempts").glob("attempt-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "partial.bin").read_bytes() == b"partial"


@pytest.mark.parametrize("value", [None, 123, Path("private/value")])
def test_cli_provenance_text_rejects_non_strings(value):
    with pytest.raises(ValueError, match="must be a string"):
        cli._required_text(value, "dataset revision")
