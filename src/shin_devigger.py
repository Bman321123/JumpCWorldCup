"""Shin (1991/1993) de-vigging, standard Strumbelj (2014) inversion (PRD v2.2 §4.3).

Replaces v1.0's non-standard model (bug B3). Given bookmaker implied
probabilities q_i (summing to beta > 1 because of the vig):

    p_i(z) = ( sqrt(z^2 + 4(1-z) q_i^2 / beta) - z ) / (2(1-z))

z (insider fraction) is chosen so sum(p_i) = 1. Sum p(0) = sqrt(beta) > 1 and the
sum decreases in z, so a bracketed root always exists for vigged markets.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def american_to_implied(odds: Sequence[float]) -> np.ndarray:
    q = []
    for x in odds:
        if x > 0:
            q.append(100.0 / (x + 100.0))
        elif x < 0:
            q.append(abs(x) / (abs(x) + 100.0))
        else:
            raise ValueError(f"Invalid American odds: {x}")
    return np.asarray(q, dtype=float)


def decimal_to_implied(odds: Sequence[float]) -> np.ndarray:
    arr = np.asarray(odds, dtype=float)
    if np.any(arr <= 1.0):
        raise ValueError(f"Invalid decimal odds: {odds}")
    return 1.0 / arr


def shin_devig(q: np.ndarray) -> np.ndarray:
    """Devig implied probabilities -> true probabilities summing to exactly 1."""
    q = np.asarray(q, dtype=float)
    if len(q) < 2:
        raise ValueError("Need at least 2 outcomes to devig.")
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError(f"Implied probabilities out of range: {q}")
    beta = q.sum()
    if beta <= 1.0:
        logger.warning("Implied probs sum to %.4f <= 1; normalizing only.", beta)
        return q / beta

    def p_of_z(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / beta) - z) / (2.0 * (1.0 - z))

    try:
        z_star = brentq(lambda z: p_of_z(z).sum() - 1.0, 0.0, 0.9, xtol=1e-12)
    except ValueError:
        logger.warning("Shin root not bracketed (beta=%.4f); multiplicative fallback.", beta)
        return q / beta
    p = p_of_z(z_star)
    return p / p.sum()


class ShinDevigger:
    def devig_american(self, american_odds: Sequence[float]) -> np.ndarray:
        return shin_devig(american_to_implied(american_odds))

    def devig_decimal(self, decimal_odds: Sequence[float]) -> np.ndarray:
        return shin_devig(decimal_to_implied(decimal_odds))

    def devig_two_way_decimal(self, yes_odds: float, no_odds: float) -> float:
        return float(self.devig_decimal([yes_odds, no_odds])[0])

    def devig_two_way_american(self, yes_odds: float, no_odds: float) -> float:
        return float(self.devig_american([yes_odds, no_odds])[0])
