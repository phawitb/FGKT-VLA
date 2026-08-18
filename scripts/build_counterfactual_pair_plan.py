#!/usr/bin/env python3
"""Freeze one label-eligible source-versus-repair LIBERO pair plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from dataclasses import asdict
from typing import Any

from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.episode import validate_episode_shard
from fcut_vla.libero.pairing import FailureTarget, InitialState, build_repair_pair_plan
from fcut_vla.libero.runtime import (
    CANONICAL_ADAPTER_REPO,
    canonical_environment_contract,
    verified_hf_libero_version,
)
from scripts.verify_adapter_run import verify_run
from fcut_vla.repair.continuation import continuation_recipe_from_mapping
from fcut_vla.repair.training import ContinuationTrainingPlan, verify_repair_candidate


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _verify_candidate_run(
    candidate_run: Path,
    source_run: Path,
    source_report: dict[str, Any],
    source_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_run = Path(candidate_run)
    recipe = json.loads((candidate_run / "repair_recipe.json").read_text())
    recipe_object = continuation_recipe_from_mapping(recipe)
    metadata_bytes = (candidate_run / "candidate_metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    complete = json.loads((candidate_run / "COMPLETE").read_text())
    recipe_payload = dict(recipe)
    recipe_hash = recipe_payload.pop("content_hash", None)
    if recipe_hash != sha256(_canonical(recipe_payload)).hexdigest():
        raise ValueError("repair recipe content hash is invalid")
    completion_hash = complete.pop("completion_sha256", None)
    if completion_hash != sha256(_canonical(complete)).hexdigest():
        raise ValueError("repair candidate completion hash is invalid")
    expected_completion = {
        "schema_version": 1,
        "recipe_sha256": recipe_hash,
        "candidate_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "candidate_adapter_sha256": metadata.get("candidate_adapter_sha256"),
        "adapter_config_sha256": metadata.get("adapter_config_sha256"),
        "policy_config_sha256": metadata.get("policy_config_sha256"),
        "preprocessor_sha256": metadata.get("preprocessor_sha256"),
        "postprocessor_sha256": metadata.get("postprocessor_sha256"),
        "processor_manifest_sha256": metadata.get("processor_manifest_sha256"),
    }
    if complete != expected_completion or metadata.get("recipe_sha256") != recipe_hash:
        raise ValueError("repair candidate completion envelope is inconsistent")
    source_checkpoint = (
        Path(source_run)
        / "checkpoints"
        / "client-adapter"
        / "checkpoints"
        / f"{int(source_report['completed_step']):06d}"
    )
    plan = ContinuationTrainingPlan(
        recipe=recipe_object,
        source_checkpoint_dir=source_checkpoint,
        source_pretrained_dir=source_checkpoint / "pretrained_model",
        output_dir=candidate_run / "training",
        save_frequency=1,
        python_executable=Path(sys.executable),
    )
    recomputed = verify_repair_candidate(
        plan=plan,
        source_evidence=source_evidence,
        candidate_log=candidate_run / "logs" / "stderr.log",
    )
    if _canonical(asdict(recomputed)) != _canonical(metadata):
        raise ValueError("repair candidate metadata does not match authoritative verification")
    return recipe, metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-adapter-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--failure-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    source_report = verify_run(args.source_adapter_run)
    source_manifest = json.loads((args.source_adapter_run / "run_manifest.json").read_text())
    source_evidence = json.loads((args.source_adapter_run / "adapter_metadata.json").read_text())
    recipe, candidate = _verify_candidate_run(
        args.candidate_run, args.source_adapter_run, source_report, source_evidence
    )
    episodes = validate_episode_shard(args.episode)
    if len(episodes) != 1 or episodes[0].terminal_success:
        raise ValueError("counterfactual plan requires exactly one failed episode")
    episode = episodes[0]
    source_digest = str(source_report["adapter_sha256"])
    if (episode.policy_repo, episode.policy_revision) != (CANONICAL_ADAPTER_REPO, source_digest):
        raise ValueError("failed episode policy does not match source adapter")
    shard = json.loads((args.failure_shard / "SHARD.json").read_text())
    write_failure_shard(
        (episode,),
        output_dir=args.failure_shard,
        window_size=int(shard.get("window_size", -1)),
        resume=True,
    )
    failures = [
        json.loads(line)
        for line in (args.failure_shard / "failures.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(failures) != 1 or failures[0].get("source_episode_sha256") != episode.content_hash():
        raise ValueError("failure shard is not bound to the failed episode")
    failure_hash = sha256(_canonical(failures[0]["context"])).hexdigest()
    match = re.fullmatch(
        r"(libero_spatial|libero_object|libero_goal|libero_10)\.task(0|[1-9][0-9]*)",
        episode.key.task_alias,
    )
    if match is None:
        raise ValueError("episode task alias is not canonical")
    suite, task_text = match.groups()
    environment = canonical_environment_contract(
        suite=suite,
        task_id=int(task_text),
        lerobot_commit=str(source_manifest["runtime"]["lerobot_commit"]),
        hf_libero_version=verified_hf_libero_version(),
    )
    plan = build_repair_pair_plan(
        failure=FailureTarget(failure_hash, episode.problem_id),
        source_adapter_sha256=source_digest,
        candidate_metadata=candidate,
        recipe=recipe,
        seed=episode.key.seed,
        initial_state=InitialState(
            episode.key.initial_state_index, episode.initial_state_hash
        ),
        environment=environment,
        benchmark_hash=str(source_manifest["manifest_hash"]),
        source_contract={
            "source_run_hash": source_report["run_hash"],
            "source_episode_sha256": episode.content_hash(),
            "task_alias": episode.key.task_alias,
            "problem_id": episode.problem_id,
            "instruction": episode.instruction,
            "dataset_repo": source_manifest["dataset_repo_id"],
            "dataset_revision": source_manifest["dataset_revision"],
            "base_policy_repo": source_manifest["base_policy"],
            "base_policy_revision": source_manifest["base_policy_revision"],
            "lerobot_commit": source_manifest["runtime"]["lerobot_commit"],
            "peft_version": source_manifest["runtime"]["peft_version"],
            "hf_libero_version": environment["hf_libero_version"],
        },
    )
    payload = (plan.to_json() + "\n").encode()
    if args.output.exists():
        if not args.resume or args.output.read_bytes() != payload:
            raise ValueError("existing pair plan does not match requested artifacts")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(args.output)
    print(json.dumps({"content_hash": plan.content_hash, "pair_count": 1, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
