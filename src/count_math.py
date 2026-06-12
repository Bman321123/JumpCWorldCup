"""Threshold semantics and time-decay weights (PRD v2.2 §4.4, §4.2).

count_prob fixes v1.0 bug B1: `int(threshold)` silently turned "over 2.5" into
P(>=2). ceil unifies integer and half-line thresholds:
    "k or more" / "at least k" (integer k) -> N >= k
    "over k.5"                              -> N >= k+1
    "under k.5"                             -> N <= k
    "fewer than k" (integer k)              -> N <= k-1
GTE and LT at the same threshold are exact complements.
"""
from __future__ import annotations

import math
from typing import Callable

from .types import Condition


def count_prob(cdf: Callable[[int], float], threshold: float, condition: Condition) -> float:
    k = math.ceil(threshold)
    if condition == Condition.GTE:
        return float(1.0 - cdf(k - 1))
    if condition == Condition.LT:
        return float(cdf(k - 1))
    if condition == Condition.EQ:
        k = int(round(threshold))
        return float(cdf(k) - cdf(k - 1))
    raise ValueError(f"count_prob does not handle condition {condition}")


def decay_weight(days_ago: float, half_life_days: float = 500.0) -> float:
    """Exponential decay with an explicit half-life in days (fixes v1.0 bug B2,
    whose constant implied a 106-year half-life)."""
    if days_ago < 0:
        return 1.0
    return math.exp(-math.log(2.0) * days_ago / half_life_days)


def clip_prob(p: float, floor: float = 0.001, ceiling: float = 0.999) -> float:
    return min(max(float(p), floor), ceiling)
