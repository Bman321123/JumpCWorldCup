"""Market/model blending in log-odds space (PRD v2.2 §4.6).

Weights are fixed priors per family — never re-optimized on small in-tournament
windows. Early (non-closing) lines get a 0.85 discount on the market weight.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

EPS = 1e-6

DEFAULT_WEIGHTS: Dict[str, float] = {
    "MATCH_RESULT": 0.80,
    "GOAL_MARKET": 0.75,
    "CORNER_MARKET": 0.40,
    "CARD_MARKET": 0.30,
    "OFFSIDE_MARKET": 0.10,
    "SHOTS_MARKET": 0.40,
    "PENALTY_MARKET": 0.30,
    "PLAYER_MARKET": 0.50,
}
EARLY_LINE_DISCOUNT = 0.85
DEFER_GAP = 0.30           # model/market gap at which we fully defer to the sharp line


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1.0 - p)))


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class EnsembleBlender:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def blend(self, market_prob: Optional[float], model_prob: float,
              family: str, is_closing_line: bool = True) -> Tuple[float, str]:
        if market_prob is None:
            return float(model_prob), "model"
        w = self.weights.get(family, 0.5)
        if not is_closing_line:
            w *= EARLY_LINE_DISCOUNT
        # Market deference: on the markets the sharp line already wins (high base
        # weight — 1X2, totals, BTTS), a large model/market disagreement means
        # the model is likely broken for this matchup (sparse international data
        # compresses favorites). Defer further toward the sharp line as the gap
        # grows. NOT applied to the thin micro-markets (corners/cards/sot/fouls)
        # where our gated ML has earned the right to deviate.
        base = self.weights.get(family, 0.5)
        if base >= 0.70:
            gap = abs(market_prob - model_prob)
            w = w + (1.0 - w) * min(gap / DEFER_GAP, 1.0)
        blended = _expit(w * _logit(market_prob) + (1.0 - w) * _logit(model_prob))
        return blended, f"blend_w{w:.2f}"

    def consensus_shrink(self, p_yours: float, p_market: float, lam: float) -> float:
        """Leaderboard-conditional shrink toward the market (PRD §3.2):
        lam=1 keeps your number; lam=0 submits the market."""
        return _expit(lam * _logit(p_yours) + (1.0 - lam) * _logit(p_market))
