import math

import pytest

from fcut_vla.evaluation.metrics import (
    MetricInputError,
    continual_metrics,
    ndcg_at_k,
    selection_regret,
)
from fcut_vla.evaluation.statistics import paired_bootstrap_ci


def test_ndcg_is_one_for_perfect_ranking_and_lower_when_reversed():
    true_utility = [3.0, 2.0, 0.0]
    assert ndcg_at_k(true_utility, [0, 1, 2], k=3) == pytest.approx(1.0)
    assert ndcg_at_k(true_utility, [2, 1, 0], k=3) < 1.0


def test_selection_regret_handles_top_k_and_abstention():
    utilities = [0.6, 0.2, -0.1]
    assert selection_regret(utilities, [1]) == pytest.approx(0.4)
    assert selection_regret(utilities, [0, 1]) == pytest.approx(0.0)
    assert selection_regret([-0.1, -0.2], []) == pytest.approx(0.0)
    assert selection_regret([0.4, -0.2], []) == pytest.approx(0.4)


def test_continual_metrics_match_hand_computed_transfer_matrix():
    matrix = [
        [0.8, math.nan, math.nan],
        [0.6, 0.7, math.nan],
        [0.5, 0.8, 0.9],
    ]
    result = continual_metrics(matrix)
    assert result.final_average == pytest.approx((0.5 + 0.8 + 0.9) / 3)
    assert result.backward_transfer == pytest.approx((-0.3 + 0.1) / 2)
    assert result.forgetting == pytest.approx((0.3 + 0.0) / 2)


def test_continual_metrics_reject_non_square_or_missing_diagonal():
    with pytest.raises(MetricInputError, match="square"):
        continual_metrics([[0.8], [0.7]])
    with pytest.raises(MetricInputError, match="diagonal"):
        continual_metrics([[math.nan, math.nan], [0.7, 0.8]])


def test_paired_bootstrap_has_exact_interval_for_constant_difference():
    interval = paired_bootstrap_ci(
        before=[0.0, 1.0, 2.0, 3.0],
        after=[1.0, 2.0, 3.0, 4.0],
        confidence=0.95,
        samples=500,
        seed=17,
    )
    assert interval.estimate == pytest.approx(1.0)
    assert interval.lower == pytest.approx(1.0)
    assert interval.upper == pytest.approx(1.0)


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(MetricInputError, match="paired"):
        paired_bootstrap_ci([0.0, 1.0], [1.0], samples=100)
