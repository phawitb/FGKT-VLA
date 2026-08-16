"""Extract privacy-bounded failure contexts from validated episode records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Iterable

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
