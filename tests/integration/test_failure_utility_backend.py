from pathlib import Path

from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.evaluator import FixtureEvaluator


FIXTURE = Path(__file__).parents[1] / "fixtures" / "libero" / "episodes.jsonl"


def test_fixture_failure_backend_is_byte_stable_and_resume_safe(tmp_path):
    records = FixtureEvaluator(FIXTURE).episode_records()

    first = write_failure_shard(records, output_dir=tmp_path / "a", window_size=4)
    second = write_failure_shard(records, output_dir=tmp_path / "b", window_size=4)
    resumed = write_failure_shard(
        records, output_dir=tmp_path / "a", window_size=4, resume=True
    )

    assert first == second == resumed
    assert first["failure_count"] == 1
    assert (tmp_path / "a" / "failures.jsonl").read_bytes() == (
        tmp_path / "b" / "failures.jsonl"
    ).read_bytes()
