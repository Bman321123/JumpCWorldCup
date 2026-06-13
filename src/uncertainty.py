"""Estimated 95% confidence interval for a submitted probability.

This is an honest uncertainty band, not a true posterior: it widens when we have
no sharp market to anchor to, when our model and the market disagree, and for
the noisier question families. It tells you how much to trust the point
estimate — a tight +-5% on a market-anchored 1X2, a wide +-25% on a model-only
player prop.
"""
from __future__ import annotations

import math
from typing import Optional

# per-family baseline std-error of our probability estimate (from how noisy that
# market type has been for us; mirrors the submission-policy tau intuition)
FAMILY_SIGMA = {
    "MATCH_RESULT": 0.05,
    "GOAL_MARKET": 0.05,
    "CORNER_MARKET": 0.08,
    "CARD_MARKET": 0.08,
    "SHOTS_MARKET": 0.09,
    "OFFSIDE_MARKET": 0.11,
    "PENALTY_MARKET": 0.06,
    "PLAYER_MARKET": 0.12,
    "FALLBACK": 0.20,
}
DEFAULT_SIGMA = 0.09
NO_MARKET_WIDEN = 1.6
Z95 = 1.96
HW_MIN, HW_MAX = 0.02, 0.40


def ci_halfwidth(p_final: float, p_model: Optional[float],
                 p_market: Optional[float], family: str) -> float:
    """95% CI half-width (a +- fraction). Tighter when market-anchored and the
    sources agree; wider model-only or on disagreement."""
    base = FAMILY_SIGMA.get(family, DEFAULT_SIGMA)
    if p_market is not None:
        gap = abs((p_model if p_model is not None else p_final) - p_market)
        sigma = math.sqrt((base * 0.5) ** 2 + (gap * 0.5) ** 2)
    else:
        sigma = base * NO_MARKET_WIDEN
    return max(HW_MIN, min(Z95 * sigma, HW_MAX))


def ci_band(p_final: float, p_model: Optional[float], p_market: Optional[float],
            family: str) -> dict:
    hw = ci_halfwidth(p_final, p_model, p_market, family)
    return {"halfwidth": round(hw, 4),
            "low": round(max(0.01, p_final - hw), 4),
            "high": round(min(0.99, p_final + hw), 4)}
