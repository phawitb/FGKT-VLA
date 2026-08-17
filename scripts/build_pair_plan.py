#!/usr/bin/env python3
"""Freeze one exact Stage C smoke pair from verified Stage A/B artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.episode import validate_episode_shard
from fcut_vla.libero.pairing import (
    AdapterCandidate,
    FailureTarget,
    InitialState,
    build_pair_plan,
)
from fcut_vla.libero.runtime import (
    CANONICAL_ADAPTER_REPO,
    canonical_environment_contract,
    verified_hf_libero_version,
)
from scripts.verify_adapter_run import verify_run


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--failure-shard", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    episodes = validate_episode_shard(args.episode)
    if len(episodes) != 1:
        raise ValueError("smoke pair plan requires exactly one episode")
    episode = episodes[0]

    shard = json.loads((args.failure_shard / "SHARD.json").read_text())
    write_failure_shard(
        (episode,),
        output_dir=args.failure_shard,
        window_size=int(shard.get("window_size", -1)),
        resume=True,
    )
    failure_bytes = (args.failure_shard / "failures.jsonl").read_bytes()
    digest = sha256(failure_bytes).hexdigest()
    if digest != shard.get("failures_sha256"):
        raise ValueError("failure shard hash does not match failures.jsonl")
    if shard.get("source_episode_sha256") != [episode.content_hash()]:
        raise ValueError("failure shard is not bound to the requested episode")
    failures = [json.loads(line) for line in failure_bytes.decode().splitlines() if line.strip()]
    if len(failures) != 1:
        raise ValueError("smoke pair plan requires exactly one failure")
    failure = failures[0]
    if failure.get("source_episode_sha256") != episode.content_hash():
        raise ValueError("failure record source hash does not match episode")
    failure_hash = sha256(_canonical(failure["context"]).encode()).hexdigest()

    report = verify_run(args.adapter_run)
    adapter_sha = str(report["adapter_sha256"])
    if (episode.policy_repo, episode.policy_revision) != (
        CANONICAL_ADAPTER_REPO,
        adapter_sha,
    ):
        raise ValueError("episode policy identity does not match verified adapter")
    manifest = json.loads((args.adapter_run / "run_manifest.json").read_text())

    match = re.fullmatch(r"(libero_spatial|libero_object|libero_goal|libero_10)\.task(0|[1-9][0-9]*)", episode.key.task_alias)
    if match is None:
        raise ValueError("episode task alias is not a canonical LIBERO task alias")
    prefix, task_text = match.groups()
    task_id = int(task_text)
    environment = canonical_environment_contract(
        suite=prefix,
        task_id=task_id,
        lerobot_commit=str(manifest["runtime"]["lerobot_commit"]),
        hf_libero_version=verified_hf_libero_version(),
    )
    plan = build_pair_plan(
        failures=(FailureTarget(failure_hash, episode.problem_id),),
        adapters=(AdapterCandidate(adapter_sha, CANONICAL_ADAPTER_REPO, adapter_sha),),
        seeds=(episode.key.seed,),
        initial_states=(InitialState(episode.key.initial_state_index, episode.initial_state_hash),),
        episode_budget=1,
        environment=environment,
        benchmark_hash=str(manifest["manifest_hash"]),
        baseline_policy=(str(manifest["base_policy"]), str(manifest["base_policy_revision"])),
        purpose="plumbing_self_replay",
        label_eligible=False,
    )
    payload = plan.to_json() + "\n"
    if args.output.exists():
        if not args.resume:
            raise ValueError("pair plan exists; pass --resume to validate it")
        if args.output.read_text() != payload:
            raise ValueError("existing pair plan does not match requested artifacts")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.output)
    print(
        json.dumps(
            {"content_hash": plan.content_hash, "pair_count": len(plan.pairs), "output": str(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
