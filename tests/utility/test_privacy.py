import pytest

from fcut_vla.utility.privacy import PrivacyBoundaryError, assert_ranker_payload_safe


@pytest.mark.parametrize(
    "forbidden",
    ["client_id", "task_id", "raw_images", "private_path", "raw_simulator_state", "trajectory"],
)
def test_recursive_privacy_validator_rejects_forbidden_nested_key(forbidden):
    with pytest.raises(PrivacyBoundaryError, match=forbidden):
        assert_ranker_payload_safe({"nested": [{forbidden: "secret"}]})


def test_recursive_privacy_validator_accepts_sanitized_ranker_payload():
    payload = {
        "failure_hash": "failure-sha",
        "adapter_digest": "adapter-sha",
        "descriptor": {"skill_prototype": [0.1, 0.2]},
        "utility": {"estimate": 0.3, "mask": [True, False]},
    }

    assert_ranker_payload_safe(payload)
