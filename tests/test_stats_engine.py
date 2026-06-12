import math

import pytest

from src.stats_engine import ModelParameters, StatsEngine
from src.types import Condition, MatchContext, TemporalWindow


@pytest.fixture
def engine():
    params = ModelParameters(
        mu=0.18, gamma=0.25, rho=-0.12,
        attack={"ARG": 0.45, "FRA": 0.42, "BRA": 0.40, "KSA": -0.35, "RSA": -0.30},
        defense={"ARG": 0.40, "FRA": 0.38, "BRA": 0.30, "KSA": -0.25, "RSA": -0.20})
    return StatsEngine(params)


def test_result_trio_sums_to_one(engine):
    r = engine.result_probs("ARG", "KSA")
    assert sum(r.values()) == pytest.approx(1.0)


def test_strong_team_favored(engine):
    r = engine.result_probs("BRA", "RSA")
    assert r["home_win"] > 0.55


def test_btts_differs_from_independence(engine):     # B6 — must use the tau matrix
    lam_h, lam_a = engine.expected_goals("ARG", "FRA")
    independent = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
    btts = engine.goal_market("ARG", "FRA", "BTTS", "MATCH", 1.0,
                              Condition.BINARY_YES)
    assert btts != pytest.approx(independent, abs=1e-6)


def test_over_under_complement(engine):
    over = engine.goal_market("ARG", "FRA", "GOALS", "MATCH", 2.5, Condition.GTE)
    under = engine.goal_market("ARG", "FRA", "GOALS", "MATCH", 2.5, Condition.LT)
    assert over + under == pytest.approx(1.0)


def test_first_half_fewer_goals(engine):
    full = engine.goal_market("ARG", "FRA", "GOALS", "MATCH", 1.5, Condition.GTE)
    h1 = engine.goal_market("ARG", "FRA", "GOALS", "MATCH", 1.5, Condition.GTE,
                            TemporalWindow.H1)
    assert h1 < full


def test_advance_geq_win90(engine):                  # B8 — ET/pens layer
    r = engine.result_probs("ARG", "FRA")
    adv = engine.advance_prob("ARG", "FRA", "HOME")
    assert adv >= r["home_win"]
    adv_away = engine.advance_prob("ARG", "FRA", "AWAY")
    assert adv + adv_away == pytest.approx(1.0)


def test_cards_gte_zero_is_certain(engine):
    p = engine.card_market("ARG", "FRA", "MATCH", "YELLOWS", 0.0, Condition.GTE)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_cards_second_half_heavier(engine):          # B9 — cards split ~33/67
    h1 = engine.card_market("ARG", "FRA", "MATCH", "YELLOWS", 2.0, Condition.GTE,
                            TemporalWindow.H1)
    h2 = engine.card_market("ARG", "FRA", "MATCH", "YELLOWS", 2.0, Condition.GTE,
                            TemporalWindow.H2)
    assert h2 > h1


def test_corners_reasonable(engine):
    p = engine.corner_market("ARG", "FRA", "MATCH", 5.0, Condition.GTE)
    assert p > 0.70


def test_altitude_lowers_goals(engine):
    ctx = MatchContext("ARG", "FRA", "2026-06-20", goal_multiplier=0.92)
    lam_alt = engine.expected_goals("ARG", "FRA", ctx)
    lam_base = engine.expected_goals("ARG", "FRA")
    assert lam_alt[0] < lam_base[0]


def test_host_advantage_only_for_hosts(engine):
    ctx_host = MatchContext("ARG", "FRA", "2026-06-20", home_is_host=True)
    lam_host = engine.expected_goals("ARG", "FRA", ctx_host)[0]
    lam_neutral = engine.expected_goals("ARG", "FRA")[0]
    assert lam_host > lam_neutral


def test_unknown_team_uses_league_average(engine):
    r = engine.result_probs("ARG", "XYZ")
    assert sum(r.values()) == pytest.approx(1.0)
    assert r["home_win"] > r["away_win"]      # ARG above-average vs average
