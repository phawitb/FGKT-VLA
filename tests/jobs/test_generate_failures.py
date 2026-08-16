from tests.libero.test_episode import episode_mapping

from fcut_vla.jobs.generate_failures import extract_failures
from fcut_vla.libero.episode import EpisodeRecord


def test_failed_episode_extracts_context_but_success_is_excluded():
    failed = EpisodeRecord.from_mapping(episode_mapping(success=False, episode_index=0))
    successful = EpisodeRecord.from_mapping(episode_mapping(success=True, episode_index=1))

    extracted = extract_failures([failed, successful], window_size=4)

    assert len(extracted) == 1
    assert extracted[0].context.mask == (False, False, True, True)
    assert extracted[0].source_episode_sha256 == failed.content_hash()
    assert str(extracted[0].context.failure_id) == "t03_open_top_drawer-seed101-ep0:step-1"


def test_failure_extraction_is_deterministic_regardless_of_input_order():
    later = EpisodeRecord.from_mapping(episode_mapping(episode_index=2))
    earlier = EpisodeRecord.from_mapping(episode_mapping(episode_index=0))

    left = extract_failures([later, earlier], window_size=2)
    right = extract_failures([earlier, later], window_size=2)

    assert [item.to_json() for item in left] == [item.to_json() for item in right]
