"""Run LeRobot training while pinning its otherwise mutable tokenizer lookup."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fgkt-tokenizer-repo", required=True)
    parser.add_argument("--fgkt-tokenizer-revision", required=True)
    parser.add_argument("lerobot_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    lerobot_args = list(args.lerobot_args)
    if not lerobot_args or lerobot_args.pop(0) != "--":
        raise ValueError("pinned training arguments must be separated by --")

    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained.__func__

    def pinned_from_pretrained(cls, name, *call_args, **call_kwargs):
        if name == args.fgkt_tokenizer_repo:
            call_kwargs["revision"] = args.fgkt_tokenizer_revision
        return original(cls, name, *call_args, **call_kwargs)

    AutoTokenizer.from_pretrained = classmethod(pinned_from_pretrained)
    sys.argv = ["lerobot_train", *lerobot_args]
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()
