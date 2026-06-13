import pytest

from src.platform_client import to_platform_probability
from src.uncertainty import ci_band, ci_halfwidth


def test_market_anchored_agreement_is_tight():
    # model and market agree, validated family -> narrow CI
    hw = ci_halfwidth(0.60, 0.60, 0.60, "MATCH_RESULT")
    assert hw < 0.08


def test_model_only_micro_market_is_wide():
    # no market, player prop -> wide CI
    hw = ci_halfwidth(0.40, 0.40, None, "PLAYER_MARKET")
    assert hw > 0.15


def test_disagreement_widens():
    agree = ci_halfwidth(0.55, 0.55, 0.55, "GOAL_MARKET")
    disagree = ci_halfwidth(0.55, 0.40, 0.65, "GOAL_MARKET")
    assert disagree > agree


def test_fallback_is_very_wide():
    hw = ci_halfwidth(0.45, 0.45, None, "FALLBACK")
    assert hw > 0.25


def test_ci_band_within_bounds():
    b = ci_band(0.95, 0.95, None, "PLAYER_MARKET")
    assert b["high"] <= 0.99 and b["low"] >= 0.01
    assert b["low"] < 0.95 < b["high"]


# ----- the 50% restriction -----

def test_never_submits_50():
    assert to_platform_probability(0.50) != 50
    assert to_platform_probability(0.499) != 50
    assert to_platform_probability(0.503) != 50


def test_50_nudges_to_nearer_side():
    assert to_platform_probability(0.503) == 51     # leans over -> 51
    assert to_platform_probability(0.497) == 49     # leans under -> 49
    assert to_platform_probability(0.50) == 51      # exact coin flip -> 51


def test_other_values_unchanged():
    assert to_platform_probability(0.62) == 62
    assert to_platform_probability(0.07) == 7
    assert to_platform_probability(0.999) == 99
    assert to_platform_probability(0.0) == 1
