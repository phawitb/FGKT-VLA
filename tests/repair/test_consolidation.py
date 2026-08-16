import torch

from fcut_vla.repair.consolidation import evaluate_gate, promote_with_rollback


def test_gate_requires_gain_average_and_worst_task_bounds():
    accepted = evaluate_gate(
        gain_lower=0.12,
        retention_risk_upper={"old-a": 0.01, "old-b": 0.03},
        min_gain=0.1,
        max_average_risk=0.025,
        max_worst_task_risk=0.04,
    )
    low_gain = evaluate_gate(
        gain_lower=0.09,
        retention_risk_upper={"old-a": 0.0},
        min_gain=0.1,
        max_average_risk=0.025,
        max_worst_task_risk=0.04,
    )
    high_average = evaluate_gate(
        gain_lower=0.2,
        retention_risk_upper={"old-a": 0.03, "old-b": 0.03},
        min_gain=0.1,
        max_average_risk=0.025,
        max_worst_task_risk=0.04,
    )
    high_worst = evaluate_gate(
        gain_lower=0.2,
        retention_risk_upper={"old-a": 0.0, "old-b": 0.05},
        min_gain=0.1,
        max_average_risk=0.03,
        max_worst_task_risk=0.04,
    )

    assert accepted.promote
    assert not low_gain.promote and "gain" in low_gain.reasons
    assert not high_average.promote and "average_retention" in high_average.reasons
    assert not high_worst.promote and "worst_task" in high_worst.reasons


def test_no_match_abstains_and_rollback_preserves_current_state():
    decision = evaluate_gate(
        gain_lower=1.0,
        retention_risk_upper={},
        min_gain=0.1,
        max_average_risk=0.02,
        max_worst_task_risk=0.04,
        matched=False,
    )
    current = {"A": torch.tensor([1.0])}
    rejected = {"A": torch.tensor([99.0])}

    promoted = promote_with_rollback(current, rejected, decision)

    assert not decision.promote and decision.abstained
    assert promoted.keys() == current.keys()
    assert torch.equal(promoted["A"], current["A"])
    assert promoted["A"].data_ptr() != current["A"].data_ptr()
