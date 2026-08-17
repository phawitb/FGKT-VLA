#!/usr/bin/env python3
"""Build or validate an immutable Stage B failure shard from episode records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcut_vla.jobs.generate_failures import write_failure_shard
from fcut_vla.libero.episode import validate_episode_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = tuple(
        record
        for episode_path in args.episode
        for record in validate_episode_shard(episode_path)
    )
    report = write_failure_shard(
        records,
        output_dir=args.output_dir,
        window_size=args.window_size,
        resume=args.resume,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
