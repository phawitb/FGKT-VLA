"""Recursive privacy gate for utility-ranker payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fcut_vla.adapters.bank import FORBIDDEN_FIELDS


class PrivacyBoundaryError(ValueError):
    """Raised when private rollout or ownership data crosses into ranker input."""


RANKER_FORBIDDEN_FIELDS = FORBIDDEN_FIELDS.union(
    {
        "raw_simulator_state",
        "raw_frames",
        "trajectory",
        "trajectories",
    }
)


def assert_ranker_payload_safe(payload: object) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().casefold()
            if normalized in RANKER_FORBIDDEN_FIELDS:
                raise PrivacyBoundaryError(f"forbidden ranker field: {key}")
            assert_ranker_payload_safe(value)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            assert_ranker_payload_safe(value)
