"""Extract privacy-bounded failure contexts from validated episode records."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fcut_vla.benchmark.failure_context import FailureContext
from fcut_vla.libero.episode import EpisodeRecord
from fcut_vla.types import FailureId


@dataclass(frozen=True)
class ExtractedFailure:
    context: FailureContext
    source_episode_sha256: str

    def to_json(self) -> str:
        payload = {
            "context": json.loads(self.context.to_json()),
            "source_episode_sha256": self.source_episode_sha256,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def extract_failures(
    records: Iterable[EpisodeRecord], *, window_size: int
) -> tuple[ExtractedFailure, ...]:
    extracted: list[ExtractedFailure] = []
    for record in sorted(records, key=lambda item: item.key):
        if record.terminal_success:
            continue
        last_step = len(record.steps) - 1
        episode_name = (
            f"{record.key.task_alias}-seed{record.key.seed}-ep{record.key.episode_index}"
        )
        context = FailureContext.from_episode(
            failure_id=FailureId(episode_name, last_step),
            instruction=record.instruction,
            steps=[step.failure_features() for step in record.steps],
            failure_step=last_step,
            window_size=window_size,
            terminal_success=False,
        )
        extracted.append(ExtractedFailure(context, record.content_hash()))
    return tuple(extracted)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, data: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def write_failure_shard(
    records: Iterable[EpisodeRecord],
    *,
    output_dir: Path,
    window_size: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Write or validate one immutable, hash-bound Stage B failure shard."""
    records = tuple(sorted(records, key=lambda record: record.key))
    if not records:
        raise ValueError("failure shard requires at least one episode")
    if len({record.key for record in records}) != len(records):
        raise ValueError("failure shard contains duplicate episode keys")
    if window_size < 1:
        raise ValueError("failure window size must be positive")

    failures = extract_failures(records, window_size=window_size)
    failure_text = "".join(item.to_json() + "\n" for item in failures)
    failure_hash = sha256(failure_text.encode()).hexdigest()
    expected = {
        "schema_version": 1,
        "episode_count": len(records),
        "failure_count": len(failures),
        "window_size": window_size,
        "source_episode_sha256": [record.content_hash() for record in records],
        "failures_sha256": failure_hash,
    }

    output_dir = Path(output_dir)
    shard_path = output_dir / "SHARD.json"
    failures_path = output_dir / "failures.jsonl"
    digest_path = output_dir / "failures.sha256"
    if output_dir.exists():
        if not resume:
            raise ValueError("failure shard directory exists; pass resume to validate it")
        try:
            actual = json.loads(shard_path.read_text())
            actual_text = failures_path.read_text()
            claimed_digest = digest_path.read_text().strip()
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing failure shard is incomplete or malformed") from error
        actual_digest = sha256(actual_text.encode()).hexdigest()
        if actual_digest != claimed_digest or actual_digest != actual.get("failures_sha256"):
            raise ValueError("existing failure shard hash does not match its contents")
        if actual != expected or actual_text != failure_text:
            raise ValueError("existing failure shard does not match requested episodes")
        return actual

    output_dir.mkdir(parents=True)
    _atomic_write(failures_path, failure_text)
    _atomic_write(digest_path, failure_hash + "\n")
    _atomic_write(shard_path, _canonical_json(expected) + "\n")
    return expected
