"""Deterministic paired rollout plans for counterfactual adapter evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from collections.abc import Iterable, Mapping
from typing import Any


class PairingError(ValueError):
    """Raised when baseline and candidate rollouts cannot be compared exactly."""


_PURPOSE_LABEL_ELIGIBILITY = {
    "counterfactual_repair": True,
    "plumbing_self_replay": False,
}

_REPAIR_SOURCE_CONTRACT_FIELDS = frozenset(
    {
        "source_run_hash",
        "source_episode_sha256",
        "task_alias",
        "problem_id",
        "instruction",
        "dataset_repo",
        "dataset_revision",
        "base_policy_repo",
        "base_policy_revision",
        "lerobot_commit",
        "peft_version",
        "hf_libero_version",
    }
)


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
    purpose: str
    label_eligible: bool
    pairs: tuple[CounterfactualPair, ...]
    content_hash: str

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "benchmark_hash": self.benchmark_hash,
            "environment": self.environment,
            "baseline_policy_repo": self.baseline_policy_repo,
            "baseline_policy_revision": self.baseline_policy_revision,
            "purpose": self.purpose,
            "label_eligible": self.label_eligible,
            "pairs": [asdict(pair) for pair in self.pairs],
        }
        return {**payload, "content_hash": self.content_hash}

    def to_json(self) -> str:
        return _canonical(self.to_mapping()).decode()


def validate_pair_plan(plan: PairPlan) -> PairPlan:
    payload = plan.to_mapping()
    claimed = payload.pop("content_hash")
    computed = sha256(_canonical(payload)).hexdigest()
    if claimed != computed:
        raise PairingError("pair plan content hash does not match its canonical payload")
    if plan.schema_version != 1 or not plan.pairs:
        raise PairingError("pair plan schema or pair set is invalid")
    _required(plan.benchmark_hash, "benchmark hash")
    _required(plan.purpose, "pair-plan purpose")
    _validate_purpose_eligibility(plan.purpose, plan.label_eligible)
    environment_hash = sha256(_canonical(plan.environment)).hexdigest()
    seen_keys: set[PairKey] = set()
    for pair in plan.pairs:
        if pair.key in seen_keys:
            raise PairingError("pair plan contains a duplicate pair key")
        seen_keys.add(pair.key)
        if (pair.baseline.role, pair.candidate.role) != ("baseline", "candidate"):
            raise PairingError("pair plan member roles are invalid")
        if pair.baseline.key != pair.key or pair.candidate.key != pair.key:
            raise PairingError("pair plan member keys do not match the frozen pair key")
        if pair.key.environment_hash != environment_hash:
            raise PairingError("pair plan environment hash is inconsistent")
        if (
            pair.baseline.initial_state_hash != pair.key.initial_state_hash
            or pair.candidate.initial_state_hash != pair.key.initial_state_hash
        ):
            raise PairingError("pair plan initial-state identity is inconsistent")
        if (pair.baseline.policy_repo, pair.baseline.policy_revision) != (
            plan.baseline_policy_repo,
            plan.baseline_policy_revision,
        ):
            raise PairingError("pair plan baseline policy identity is inconsistent")
    return plan


def _required(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise PairingError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized:
        raise PairingError(f"{label} must be non-empty")
    return normalized


def _validate_purpose_eligibility(purpose: str, label_eligible: bool) -> None:
    expected = _PURPOSE_LABEL_ELIGIBILITY.get(purpose)
    if expected is None or bool(label_eligible) != expected:
        raise PairingError("pair-plan purpose and label eligibility are inconsistent")


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
    purpose: str = "counterfactual_repair",
    label_eligible: bool = True,
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
    purpose = _required(purpose, "pair-plan purpose")
    _validate_purpose_eligibility(purpose, label_eligible)
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
        "purpose": purpose,
        "label_eligible": bool(label_eligible),
        "pairs": [asdict(pair) for pair in pairs_tuple],
    }
    return PairPlan(
        schema_version=1,
        benchmark_hash=benchmark_hash,
        environment=environment_payload,
        baseline_policy_repo=baseline_repo,
        baseline_policy_revision=baseline_revision,
        purpose=purpose,
        label_eligible=bool(label_eligible),
        pairs=pairs_tuple,
        content_hash=sha256(_canonical(payload)).hexdigest(),
    )


def build_repair_pair_plan(
    *,
    failure: FailureTarget,
    source_adapter_sha256: str,
    candidate_metadata: Mapping[str, Any],
    recipe: Mapping[str, Any],
    seed: int,
    initial_state: InitialState,
    environment: Mapping[str, Any],
    benchmark_hash: str,
    source_contract: Mapping[str, Any],
) -> PairPlan:
    """Build one label-eligible pair from a verified failed adapter and its repair."""
    from fcut_vla.libero.runtime import CANONICAL_ADAPTER_REPO

    source_digest = _required(source_adapter_sha256, "source adapter digest")
    candidate_digest = _required(
        candidate_metadata.get("candidate_adapter_sha256", ""),
        "candidate adapter digest",
    )
    if candidate_digest == source_digest:
        raise PairingError("repair pair requires distinct verified adapters")
    if (
        candidate_metadata.get("source_adapter_sha256") != source_digest
        or candidate_metadata.get("recipe_sha256") != recipe.get("content_hash")
    ):
        raise PairingError("candidate metadata is not bound to the repair recipe")
    if (
        recipe.get("source_adapter_sha256") != source_digest
        or recipe.get("failure_hash") != failure.content_hash
    ):
        raise PairingError("repair recipe is not bound to the source failure")
    if set(source_contract) != _REPAIR_SOURCE_CONTRACT_FIELDS:
        raise PairingError("repair recipe provenance does not match verified source artifacts")
    for field, expected in source_contract.items():
        canonical = _required(expected, f"repair source contract {field}")
        if canonical != expected or recipe.get(field) != expected:
            raise PairingError(
                "repair recipe provenance does not match verified source artifacts"
            )
    if failure.problem_id != source_contract["problem_id"]:
        raise PairingError("repair recipe provenance does not match verified source artifacts")
    return build_pair_plan(
        failures=(failure,),
        adapters=(
            AdapterCandidate(
                candidate_digest,
                CANONICAL_ADAPTER_REPO,
                candidate_digest,
            ),
        ),
        seeds=(seed,),
        initial_states=(initial_state,),
        episode_budget=1,
        environment=environment,
        benchmark_hash=benchmark_hash,
        baseline_policy=(CANONICAL_ADAPTER_REPO, source_digest),
        purpose="counterfactual_repair",
        label_eligible=True,
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
