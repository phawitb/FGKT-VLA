"""Build privacy-safe utility labels from complete paired rollout results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fcut_vla.utility.counterfactual import (
    CostNormalization,
    UtilityObservation,
    compute_utility,
)
from fcut_vla.utility.privacy import assert_ranker_payload_safe


@dataclass(frozen=True)
class PairResult:
    failure_hash: str
    adapter_digest: str
    descriptor: Mapping[str, Any]
    recovery_before: Sequence[float]
    recovery_after: Sequence[float]
    retention_before: Mapping[str, Sequence[float]]
    retention_after: Mapping[str, Sequence[float]]
    communication_bytes: int
    latency_seconds: float
    pair_plan_hash: str
    complete: bool
    seen_task: bool


@dataclass(frozen=True)
class UtilityLabel:
    failure_hash: str
    adapter_digest: str
    descriptor: dict[str, Any]
    utility: UtilityObservation
    pair_plan_hash: str
    seen_task: bool

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "failure_hash": self.failure_hash,
            "adapter_digest": self.adapter_digest,
            "descriptor": self.descriptor,
            "utility": asdict(self.utility),
            "pair_plan_hash": self.pair_plan_hash,
            "seen_task": self.seen_task,
        }
        assert_ranker_payload_safe(payload)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))


def build_utility_labels(
    pair_results: Iterable[PairResult],
    *,
    cost_normalization: CostNormalization,
    lambda_retention: float = 1.0,
    lambda_cost: float = 0.0,
) -> tuple[UtilityLabel, ...]:
    labels: list[UtilityLabel] = []
    seen_keys: set[tuple[str, str]] = set()
    for result in pair_results:
        if not result.complete:
            raise ValueError("utility labels require complete paired results")
        if not result.failure_hash or not result.adapter_digest or not result.pair_plan_hash:
            raise ValueError("utility label identities must be non-empty")
        key = (result.failure_hash, result.adapter_digest)
        if key in seen_keys:
            raise ValueError("duplicate failure/adapter utility result")
        seen_keys.add(key)
        descriptor = dict(result.descriptor)
        assert_ranker_payload_safe(descriptor)
        utility = compute_utility(
            failure_id=result.failure_hash,
            adapter_id=result.adapter_digest,
            recovery_before=result.recovery_before,
            recovery_after=result.recovery_after,
            retention_before=result.retention_before,
            retention_after=result.retention_after,
            communication_bytes=result.communication_bytes,
            latency_seconds=result.latency_seconds,
            cost_normalization=cost_normalization,
            lambda_retention=lambda_retention,
            lambda_cost=lambda_cost,
        )
        labels.append(
            UtilityLabel(
                result.failure_hash,
                result.adapter_digest,
                descriptor,
                utility,
                result.pair_plan_hash,
                bool(result.seen_task),
            )
        )
    if not labels:
        raise ValueError("utility label ledger must be non-empty")
    return tuple(sorted(labels, key=lambda item: (item.failure_hash, item.adapter_digest)))


def assign_failure_strata(labels: Iterable[UtilityLabel]) -> dict[str, str]:
    grouped: dict[str, list[UtilityLabel]] = {}
    for label in labels:
        grouped.setdefault(label.failure_hash, []).append(label)
    strata: dict[str, str] = {}
    for failure_hash, candidates in sorted(grouped.items()):
        positive = [label for label in candidates if label.utility.has_positive_utility]
        if not positive:
            strata[failure_hash] = "no_match"
        elif any(label.seen_task for label in positive):
            strata[failure_hash] = "seen_task"
        else:
            strata[failure_hash] = "compositional"
    return strata
