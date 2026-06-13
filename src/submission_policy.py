"""Submission policy — what number to actually enter.

PRINCIPLE (per the user, and correct for crowd-relative RBP scoring): the crowd
is NOT truth. We do not shrink toward it. Our submission is our own calibrated,
market-blended estimate — that already shrinks toward the SHARP MARKET (the
reliable signal) via the blend + market deference, which is the right anchor.

The crowd serves two other purposes:
  1. SELECTION / OPPORTUNITY: where we diverge from the crowd is where RBP is
     won or lost. `rbp_opportunity` quantifies that, for picking and sizing bets.
  2. DELIBERATE POSITION PLAYS (knockouts only): when LEADING we may hug the
     crowd to deny chasers variance; when TRAILING we may extremize AWAY from
     the crowd to manufacture the variance we need to catch up. These are
     conscious game-theory choices, never a default.

Default (neutral) = submit our honest number. Full stop.
"""
from __future__ import annotations

from typing import Optional

SUBMIT_FLOOR, SUBMIT_CAP = 0.03, 0.97
PLAYER_PROP_CAP = 0.85

# position -> lambda on (our_number - crowd). 1.0 = our number; <1 hugs crowd;
# >1 extremizes away from it. Only applied when a position is explicitly set.
POSITION_LAMBDA = {
    "neutral": 1.0,        # submit our number; crowd ignored for the value
    "leading": 0.65,       # defensive: shrink toward crowd, deny chasers variance
    "trailing": 1.25,      # aggressive: extremize away from crowd for RBP
    "desperate": 1.5,
}
LAMBDA_CAP = 1.6


def submission(p_hat: float, crowd: Optional[float] = None,
               family: str = "", position: str = "neutral") -> float:
    """The number to enter. Default returns our honest estimate; the crowd only
    bends it for explicit leading/trailing position plays."""
    p = float(min(max(p_hat, SUBMIT_FLOOR), SUBMIT_CAP))
    if crowd is None or position == "neutral":
        return p
    lam = min(POSITION_LAMBDA.get(position, 1.0), LAMBDA_CAP)
    f = crowd + lam * (p_hat - crowd)
    return float(min(max(f, SUBMIT_FLOOR), SUBMIT_CAP))


def rbp_opportunity(p_hat: float, crowd: Optional[float]) -> Optional[float]:
    """Signed divergence from the crowd = where RBP is made. Positive: we're
    higher than the crowd; negative: lower. Magnitude ~ how much we stand to
    gain (or lose) relative to the field if we are right. For SELECTION, not for
    moving the submitted value."""
    if crowd is None:
        return None
    return round(p_hat - crowd, 4)


def opportunity_label(p_hat: float, crowd: Optional[float]) -> str:
    """Human tag for the dashboard."""
    if crowd is None:
        return "no crowd"
    d = p_hat - crowd
    if abs(d) < 0.04:
        return "consensus (low RBP)"
    side = "ABOVE" if d > 0 else "BELOW"
    strength = "strong" if abs(d) > 0.12 else "mild"
    return f"{strength} edge {side} crowd ({d:+.0%})"
