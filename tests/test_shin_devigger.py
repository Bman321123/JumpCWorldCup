import numpy as np
import pytest

from src.shin_devigger import ShinDevigger, american_to_implied, shin_devig


def test_three_way_sums_to_one():
    d = ShinDevigger()
    p = d.devig_american([-120, +280, +450])
    assert abs(p.sum() - 1.0) < 1e-8
    assert all(0 < x < 1 for x in p)


def test_pick_em_market():
    p = ShinDevigger().devig_american([-110, -110])
    assert abs(p[0] - 0.5) < 0.01


def test_heavy_favorite():
    p = ShinDevigger().devig_american([-2000, +900])
    assert p[0] > 0.90


def test_shin_corrects_favorite_longshot_bias():
    """Longshots carry a disproportionate share of the vig, so Shin assigns the
    favorite MORE probability than naive multiplicative normalization and the
    longshot LESS."""
    q = american_to_implied([-300, +250, +700])
    p_shin = shin_devig(q)
    p_mult = q / q.sum()
    assert p_shin[0] >= p_mult[0] - 1e-9
    assert p_shin[-1] <= p_mult[-1] + 1e-9


def test_two_way_decimal():
    p_yes = ShinDevigger().devig_two_way_decimal(1.74, 2.05)
    assert 0.50 < p_yes < 0.62


def test_no_vig_market_normalizes():
    p = shin_devig(np.array([0.6, 0.35]))
    assert abs(p.sum() - 1.0) < 1e-12


def test_invalid_odds_raise():
    with pytest.raises(ValueError):
        ShinDevigger().devig_american([0, 100])
