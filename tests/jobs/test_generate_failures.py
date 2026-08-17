from tests.libero.test_episode import episode_mapping

import json

import pytest

from fcut_vla.jobs.generate_failures import extract_failures, write_failure_shard
from fcut_vla.libero.episode import EpisodeRecord


def test_failed_episode_extracts_context_but_success_is_excluded():
    failed = EpisodeRecord.from_mapping(episode_mapping(success=False, episode_index=0))
    successful = EpisodeRecord.from_mapping(episode_mapping(success=True, episode_index=1))

    extracted = extract_failures([failed, successful], window_size=4)

    assert len(extracted) == 1
    assert extracted[0].context.mask == (False, False, True, True)
    assert extracted[0].source_episode_sha256 == failed.content_hash()
    assert str(extracted[0].context.failure_id) == "t03_open_top_drawer-seed101-ep0:step-1"


def test_failure_extraction_is_deterministic_regardless_of_input_order():
    later = EpisodeRecord.from_mapping(episode_mapping(episode_index=2))
    earlier = EpisodeRecord.from_mapping(episode_mapping(episode_index=0))

    left = extract_failures([later, earlier], window_size=2)
    right = extract_failures([earlier, later], window_size=2)

    assert [item.to_json() for item in left] == [item.to_json() for item in right]


def test_failure_shard_is_hash_bound_and_resume_safe(tmp_path):
    records = (
        EpisodeRecord.from_mapping(episode_mapping(success=False, episode_index=0)),
        EpisodeRecord.from_mapping(episode_mapping(success=True, episode_index=1)),
    )

    first = write_failure_shard(records, output_dir=tmp_path / "failures", window_size=4)
    resumed = write_failure_shard(
        records, output_dir=tmp_path / "failures", window_size=4, resume=True
    )

    assert first == resumed
    assert first["failure_count"] == 1
    assert first["episode_count"] == 2
    assert (tmp_path / "failures" / "failures.sha256").read_text().strip() == first["failures_sha256"]
    assert json.loads((tmp_path / "failures" / "SHARD.json").read_text()) == first

    failures = tmp_path / "failures" / "failures.jsonl"
    failures.write_text(failures.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash"):
        write_failure_shard(records, output_dir=tmp_path / "failures", window_size=4, resume=True)


def test_failure_shard_refuses_existing_directory_without_resume(tmp_path):
    output = tmp_path / "failures"
    output.mkdir()

    with pytest.raises(ValueError, match="resume"):
        write_failure_shard(
            (EpisodeRecord.from_mapping(episode_mapping()),),
            output_dir=output,
            window_size=2,
        )


def test_failure_shard_rejects_duplicate_episode_keys(tmp_path):
    record = EpisodeRecord.from_mapping(episode_mapping())

    with pytest.raises(ValueError, match="duplicate episode"):
        write_failure_shard(
            (record, record), output_dir=tmp_path / "failures", window_size=2
        )
