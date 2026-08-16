from pathlib import Path

import pytest

from fcut_vla.adapters.bank import (
    AdapterBank,
    AdapterBankError,
    AdapterDescriptor,
)
from fcut_vla.types import AdapterId


def descriptor(**overrides) -> AdapterDescriptor:
    values = {
        "skill_prototype": (0.1, 0.2, 0.3),
        "update_sketch": (0.4, 0.5),
        "reliability": (0.6, 0.1),
        "compatibility": (1.0, 0.0),
        "adapter_norm": 1.25,
        "data_count": 50,
    }
    values.update(overrides)
    return AdapterDescriptor(**values)


def test_descriptor_exposes_only_numeric_ranker_features():
    item = descriptor()
    assert item.to_ranker_features() == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 1.0, 0.0, 1.25, 50.0)
    assert not hasattr(item, "task_id")
    assert not hasattr(item, "client_id")
    assert not hasattr(item, "raw_trajectory")


@pytest.mark.parametrize("field", ["raw_images", "raw_trajectories", "task_id", "client_id"])
def test_descriptor_mapping_rejects_private_or_identity_fields(field: str):
    raw = {
        "skill_prototype": [0.1],
        "update_sketch": [0.2],
        "reliability": [0.3],
        "compatibility": [1.0],
        "adapter_norm": 1.0,
        "data_count": 10,
        field: "forbidden",
    }
    with pytest.raises(AdapterBankError, match="forbidden descriptor field"):
        AdapterDescriptor.from_mapping(raw)


def test_bank_keeps_private_path_out_of_candidate_descriptor(tmp_path: Path):
    bank = AdapterBank()
    adapter_id = AdapterId("c1", 1, "abc")
    private_path = tmp_path / "client-c1" / "adapter.safetensors"
    bank.register(adapter_id, descriptor(), private_path=private_path)

    candidates = bank.query_candidates()
    assert candidates == ((adapter_id, descriptor()),)
    assert str(private_path) not in repr(candidates)
    assert bank.private_path(adapter_id) == private_path


def test_bank_rejects_duplicate_adapter_and_descriptor_dimension_drift(tmp_path: Path):
    bank = AdapterBank()
    first_id = AdapterId("c1", 1, "abc")
    bank.register(first_id, descriptor(), private_path=tmp_path / "first")
    with pytest.raises(AdapterBankError, match="already registered"):
        bank.register(first_id, descriptor(), private_path=tmp_path / "duplicate")
    with pytest.raises(AdapterBankError, match="descriptor dimensions"):
        bank.register(
            AdapterId("c2", 1, "def"),
            descriptor(skill_prototype=(0.1, 0.2)),
            private_path=tmp_path / "second",
        )


def test_descriptor_rejects_invalid_counts_and_non_finite_values():
    with pytest.raises(AdapterBankError, match="data_count"):
        descriptor(data_count=0)
    with pytest.raises(AdapterBankError, match="finite"):
        descriptor(adapter_norm=float("nan"))
