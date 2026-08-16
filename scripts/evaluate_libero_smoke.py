#!/usr/bin/env python3
"""Run or render one bounded official LeRobot LIBERO adapter evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcut_vla.libero.evaluator import (
    EvalShardPlan,
    FgktLiberoEvaluator,
    adapter_checkpoint_from_report,
)
from scripts.verify_adapter_run import verify_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = verify_run(args.adapter_run)
    checkpoint = adapter_checkpoint_from_report(args.adapter_run, report)
    plan = EvalShardPlan(
        checkpoint,
        str(report["adapter_sha256"]),
        args.suite,
        args.seed,
        args.episodes,
        args.output_dir,
        task_ids=(args.task_id,),
    )
    evaluator = FgktLiberoEvaluator(plan)
    if args.dry_run:
        print(json.dumps({"command": evaluator.render_command(), "plan": json.loads(evaluator.identity_json())}, sort_keys=True))
        return
    print(json.dumps(evaluator.run(), sort_keys=True))


if __name__ == "__main__":
    main()
