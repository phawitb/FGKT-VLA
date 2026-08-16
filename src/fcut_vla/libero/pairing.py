"""Deterministic paired rollout plans for counterfactual adapter evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from collections.abc import Iterable, Mapping
from typing import Any


class PairingError(ValueError):
    """Raised when baseline and candidate rollouts cannot be compared exactly."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True, order=True)
class FailureTarget:
    content_hash: str
    problem_id: str


@dataclass(frozen=True, order=True)
class AdapterCandidate:
    digest: str
    policy_repo: str
    policy_revision: str


@dataclass(frozen=True, order=True)
class InitialState:
    index: int
    content_hash: str


@dataclass(frozen=True, order=True)
class PairKey:
    failure_hash: str
    adapter_digest: str
    problem_id: str
    seed: int
    initial_state_index: int
    initial_state_hash: str
    episode_budget: int
    environment_hash: str


@dataclass(frozen=True)
class PairMember:
    role: str
    key: PairKey
    policy_repo: str
    policy_revision: str
    initial_state_hash: str
    complete: bool = False


@dataclass(frozen=True)
class CounterfactualPair:
    key: PairKey
    baseline: PairMember
    candidate: PairMember


@dataclass(frozen=True)
class PairPlan:
    schema_version: int
    benchmark_hash: str
    environment: dict[str, Any]
    baseline_policy_repo: str
    baseline_policy_revision: str
    pairs: tuple[CounterfactualPair, ...]
    content_hash: str

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "benchmark_hash": self.benchmark_hash,
            "environment": self.environment,
            "baseline_policy_repo": self.baseline_policy_repo,
            "baseline_policy_revision": self.baseline_policy_revision,
            "pairs": [asdict(pair) for pair in self.pairs],
        }
        return {**payload, "content_hash": self.content_hash}

    def to_json(self) -> str:
        return _canonical(self.to_mapping()).decode()


def _required(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise PairingError(f"{label} must be non-empty")
    return normalized


def build_pair_plan(
    *,
    failures: Iterable[FailureTarget],
    adapters: Iterable[AdapterCandidate],
    seeds: Iterable[int],
    initial_states: Iterable[InitialState],
    episode_budget: int,
    environment: Mapping[str, Any],
    benchmark_hash: str,
    baseline_policy: tuple[str, str],
) -> PairPlan:
    failures = tuple(sorted(set(failures)))
    adapters = tuple(sorted(set(adapters)))
    seeds = tuple(int(seed) for seed in seeds)
    states = tuple(sorted(set(initial_states)))
    if not failures or not adapters or not states or not seeds:
        raise PairingError("pair plan inputs must be non-empty")
    if len(seeds) != len(set(seeds)):
        raise PairingError("duplicate seed in pair plan")
    if any(seed < 0 for seed in seeds):
        raise PairingError("seeds must be non-negative")
    if int(episode_budget) < 1:
        raise PairingError("episode budget must be positive")
    benchmark_hash = _required(benchmark_hash, "benchmark hash")
    baseline_repo = _required(baseline_policy[0], "baseline policy repo")
    baseline_revision = _required(baseline_policy[1], "baseline policy revision")
    environment_payload = dict(environment)
    if not environment_payload:
        raise PairingError("environment settings must be non-empty")
    environment_hash = sha256(_canonical(environment_payload)).hexdigest()

    pairs: list[CounterfactualPair] = []
    for failure in failures:
        _required(failure.content_hash, "failure hash")
        _required(failure.problem_id, "problem id")
        for adapter in adapters:
            _required(adapter.digest, "adapter digest")
            _required(adapter.policy_repo, "candidate policy repo")
            _required(adapter.policy_revision, "candidate policy revision")
            for seed in sorted(seeds):
                for state in states:
                    if state.index < 0 or not state.content_hash:
                        raise PairingError("initial state identity is invalid")
                    key = PairKey(
                        failure.content_hash,
                        adapter.digest,
                        failure.problem_id,
                        seed,
                        state.index,
                        state.content_hash,
                        int(episode_budget),
                        environment_hash,
                    )
                    baseline = PairMember(
                        "baseline", key, baseline_repo, baseline_revision, state.content_hash
                    )
                    candidate = PairMember(
                        "candidate",
                        key,
                        adapter.policy_repo,
                        adapter.policy_revision,
                        state.content_hash,
                    )
                    pairs.append(CounterfactualPair(key, baseline, candidate))

    pairs_tuple = tuple(sorted(pairs, key=lambda pair: pair.key))
    payload = {
        "schema_version": 1,
        "benchmark_hash": benchmark_hash,
        "environment": environment_payload,
        "baseline_policy_repo": baseline_repo,
        "baseline_policy_revision": baseline_revision,
        "pairs": [asdict(pair) for pair in pairs_tuple],
    }
    return PairPlan(
        1,
        benchmark_hash,
        environment_payload,
        baseline_repo,
        baseline_revision,
        pairs_tuple,
        sha256(_canonical(payload)).hexdigest(),
    )


def validate_completed_pair(
    pair: CounterfactualPair,
    baseline: PairMember,
    candidate: PairMember,
) -> tuple[PairMember, PairMember]:
    if (baseline.role, candidate.role) != ("baseline", "candidate"):
        raise PairingError("completed pair roles must be baseline and candidate")
    if baseline.key != pair.key or candidate.key != pair.key:
        raise PairingError("completed pair keys do not match frozen plan")
    if (
        baseline.initial_state_hash != pair.key.initial_state_hash
        or candidate.initial_state_hash != pair.key.initial_state_hash
    ):
        raise PairingError("completed pair initial state does not match frozen plan")
    if (baseline.policy_repo, baseline.policy_revision) != (
        pair.baseline.policy_repo,
        pair.baseline.policy_revision,
    ) or (candidate.policy_repo, candidate.policy_revision) != (
        pair.candidate.policy_repo,
        pair.candidate.policy_revision,
    ):
        raise PairingError("completed pair policy identity does not match frozen plan")
    if not baseline.complete or not candidate.complete:
        raise PairingError("both pair members must be complete")
    return baseline, candidate
