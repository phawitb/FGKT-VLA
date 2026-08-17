#!/usr/bin/env python3
"""Record one authoritative privacy-bounded LIBERO episode from a verified adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcut_vla.libero.evaluator import adapter_checkpoint_from_report
from fcut_vla.libero.runtime import (
    CANONICAL_ADAPTER_REPO,
    LeRobotRecordPlan,
    OfficialLeRobotBindings,
    record_lerobot_episode,
)
from scripts.verify_adapter_run import verify_run


def _plan_mapping(plan: LeRobotRecordPlan) -> dict[str, object]:
    return {
        "checkpoint": str(plan.checkpoint),
        "policy_repo": plan.policy_repo,
        "policy_revision": plan.policy_revision,
        "suite": plan.suite,
        "task_id": plan.task_id,
        "task_alias": plan.task_alias,
        "seed": plan.seed,
        "episode_index": plan.episode_index,
        "initial_state_index": plan.initial_state_index,
        "output_path": str(plan.output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = verify_run(args.adapter_run)
    checkpoint = adapter_checkpoint_from_report(args.adapter_run, report)
    plan = LeRobotRecordPlan(
        checkpoint=checkpoint,
        policy_repo=CANONICAL_ADAPTER_REPO,
        policy_revision=str(report["adapter_sha256"]),
        suite=args.suite,
        task_id=args.task_id,
        task_alias=f"{args.suite}.task{args.task_id}",
        seed=args.seed,
        episode_index=args.episode_index,
        initial_state_index=args.initial_state_index,
        output_path=args.output,
    )
    if args.dry_run:
        print(json.dumps(_plan_mapping(plan), sort_keys=True))
        return

    record = record_lerobot_episode(plan, bindings=OfficialLeRobotBindings())
    print(
        json.dumps(
            {
                "content_sha256": record.content_hash(),
                "steps": len(record.steps),
                "terminal_success": record.terminal_success,
                **_plan_mapping(plan),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
