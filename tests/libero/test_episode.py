import json
import math

import pytest

from fcut_vla.libero.episode import (
    EpisodeRecord,
    EpisodeValidationError,
    validate_episode_shard,
)


def episode_mapping(*, success=False, episode_index=0):
    step = {
        "features": [0.1, 0.2],
        "proprioception": [0.3],
        "executed_action": [0.4, 0.5],
        "action_statistics": [0.6],
        "reward": 0.0,
        "done": False,
        "success": False,
    }
    terminal = {**step, "reward": float(success), "done": True, "success": success}
    return {
        "schema_version": 1,
        "task_alias": "t03_open_top_drawer",
        "problem_id": "open_the_top_drawer",
        "seed": 101,
        "episode_index": episode_index,
        "initial_state_index": episode_index,
        "initial_state_hash": f"state-{episode_index}",
        "policy_repo": "phawitbinabik/fcut-vla-dev",
        "policy_revision": "abc123",
        "instruction": "open the top drawer",
        "terminal_success": success,
        "steps": [step, terminal],
    }


def test_episode_record_has_stable_content_hash():
    left = EpisodeRecord.from_mapping(episode_mapping())
    right = EpisodeRecord.from_mapping(json.loads(json.dumps(episode_mapping())))

    assert left.content_hash() == right.content_hash()
    assert left.key.episode_index == 0


def test_episode_rejects_nonfinite_or_inconsistent_steps():
    nonfinite = episode_mapping()
    nonfinite["steps"][0]["features"][0] = math.nan
    inconsistent = episode_mapping()
    inconsistent["steps"][1]["features"] = [1.0]

    with pytest.raises(EpisodeValidationError, match="finite"):
        EpisodeRecord.from_mapping(nonfinite)
    with pytest.raises(EpisodeValidationError, match="dimensions"):
        EpisodeRecord.from_mapping(inconsistent)


def test_episode_requires_terminal_step_to_match_terminal_success():
    raw = episode_mapping(success=True)
    raw["steps"][-1]["success"] = False

    with pytest.raises(EpisodeValidationError, match="terminal_success"):
        EpisodeRecord.from_mapping(raw)


def test_shard_rejects_duplicate_episode_keys(tmp_path):
    shard = tmp_path / "episodes.jsonl"
    raw = episode_mapping()
    shard.write_text("\n".join((json.dumps(raw), json.dumps(raw))) + "\n")

    with pytest.raises(EpisodeValidationError, match="duplicate episode key"):
        validate_episode_shard(shard)

