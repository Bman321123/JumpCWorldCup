"""Pinned regression tests for v1.0 bugs B1 (ceil thresholds) and B2 (decay)."""
import pytest
from scipy.stats import poisson

from src.count_math import count_prob, decay_weight
from src.types import Condition

LAM = 2.6
CDF = lambda k: float(poisson.cdf(k, LAM)) if k >= 0 else 0.0


def test_over_half_line_is_strictly_more():          # B1 — the bug that mattered most
    # "over 2.5" means N >= 3, NOT N >= 2
    assert count_prob(CDF, 2.5, Condition.GTE) == pytest.approx(1 - poisson.cdf(2, LAM))


def test_integer_gte_unchanged():
    # "2 or more" means N >= 2
    assert count_prob(CDF, 2.0, Condition.GTE) == pytest.approx(1 - poisson.cdf(1, LAM))


def test_gte_lt_are_complements():
    for thr in (1.5, 2.0, 2.5, 3.0, 4.5):
        total = count_prob(CDF, thr, Condition.GTE) + count_prob(CDF, thr, Condition.LT)
        assert total == pytest.approx(1.0)


def test_under_half_line_includes_boundary():
    # "under 2.5" includes the 2-goal outcome
    assert count_prob(CDF, 2.5, Condition.LT) == pytest.approx(poisson.cdf(2, LAM))


def test_fewer_than_integer():
    # "fewer than 3" means N <= 2
    assert count_prob(CDF, 3.0, Condition.LT) == pytest.approx(poisson.cdf(2, LAM))


def test_exactly():
    assert count_prob(CDF, 2.0, Condition.EQ) == pytest.approx(poisson.pmf(2, LAM))


def test_gte_zero_is_certain():
    assert count_prob(CDF, 0.0, Condition.GTE) == pytest.approx(1.0)


def test_decay_half_life():                          # B2 — was a 106-YEAR half-life
    assert decay_weight(500, half_life_days=500) == pytest.approx(0.5)
    assert decay_weight(1000, half_life_days=500) == pytest.approx(0.25)
    assert decay_weight(0, half_life_days=500) == pytest.approx(1.0)
