"""Submission policy: what number to actually enter, given model, crowd, and
leaderboard position.

THE QUESTION: is submitting the model's predicted probability optimal?

Setup. Scoring is crowd-relative (observed RBP, 2026-06-12): per question your
score is proportional to (q - o)^2 - (f - o)^2, where q = crowd forecast,
f = your submission, o = outcome. Write your submission as

    f = q + lam * (p_hat - q)

lam = 1 is "submit the model number", lam = 0 is "submit the crowd".
With true probability p and model estimate p_hat = p + eps, E[eps^2] = tau^2:

    Expected score  E(lam)  prop_to  (q-p)^2 * (2*lam - lam^2) - lam^2 * tau^2
    Score std       SD(lam) prop_to  2 * lam * |p - q| * sqrt(p*(1-p))

Three consequences, in order of importance:

1. SHRINK FOR ESTIMATION ERROR (improves expected score itself).
   dE/dlam = 0  =>  lam* = edge^2 / (edge^2 + tau^2),  edge = p_hat - q.
   Raw honesty (lam = 1) is only optimal if your model has zero error. With a
   noisy model, the optimal submission sits BETWEEN model and crowd, weighted
   by signal-to-noise. tau comes from the backtest, per family: validated
   families (goal markets) earn lam near 1; unvalidated ones (player props
   before data) deserve lam near 0.4-0.6. This is not a style choice — it is
   the expected-score maximizer.

2. POSITION-DEPENDENT VARIANCE (rank optimization, finite n).
   At the honest point the expectation curve is FLAT (dE/dlam = 0) but the
   variance curve is not (dSD/dlam > 0). So a small extremization buys
   first-order variance at only second-order expected cost. Maximizing
   E + kappa*SD gives
       lam* = edge^2/(edge^2 + tau^2) + kappa * sqrt(p(1-p)) / |edge|.
   kappa > 0 (trailing, late rounds): extremize past the shrunk point.
   kappa < 0 (leading): shrink toward the crowd and deny chasers variance.
   kappa = 0 (early / mid-pack with many questions left): pure shrinkage.

3. SAMPLE SIZE IS WHY THIS MATTERS ("only a couple thousand questions").
   With n -> infinity the best-calibrated forecaster wins with certainty and
   lam* = shrinkage only. At n ~ 1040 — concentrated by round weights into
   far fewer effective questions — luck carries real weight, and the variance
   term is a legitimate instrument. But note the asymmetry: kappa enters
   linearly while overconfidence costs quadratically. Small tilts (lam up to
   ~1.3) are defensible; submitting 0.95s is not a variance strategy, it is a
   donation (see Q10, 2026-06-11: -42 RBP).

Defaults here are deliberately conservative; recompute tau per family from
predictions_log as real outcomes accumulate.
"""
from __future__ import annotations

import math
from typing import Optional

# Per-family model error (tau) priors — overwrite from backtest/predictions_log.
FAMILY_TAU = {
    "MATCH_RESULT": 0.05,
    "GOAL_MARKET": 0.05,
    "CORNER_MARKET": 0.08,
    "CARD_MARKET": 0.08,
    "SHOTS_MARKET": 0.09,
    "OFFSIDE_MARKET": 0.10,
    "PENALTY_MARKET": 0.06,
    "PLAYER_MARKET": 0.12,
}
DEFAULT_TAU = 0.08

# Leaderboard position -> kappa (risk appetite for rank, not expectation)
KAPPA_BY_POSITION = {
    "leading": -0.02,        # protect rank: shrink toward crowd
    "neutral": 0.0,          # accumulate expected score
    "trailing": 0.03,        # need variance: modest extremization
    "desperate": 0.06,       # final rounds, far behind
}

LAMBDA_MAX = 1.5             # extremization never exceeds this
SUBMIT_FLOOR, SUBMIT_CAP = 0.03, 0.97


def optimal_lambda(edge: float, tau: float, kappa: float = 0.0,
                   p_hat: float = 0.5) -> float:
    """lam* per the derivation above. edge = p_hat - crowd."""
    e2 = edge * edge
    lam = e2 / (e2 + tau * tau) if (e2 + tau * tau) > 0 else 0.0
    if kappa != 0.0 and abs(edge) > 1e-9:
        sigma_o = math.sqrt(max(p_hat * (1.0 - p_hat), 1e-9))
        lam += kappa * sigma_o / abs(edge)
    return min(max(lam, 0.0), LAMBDA_MAX)


def submission(p_hat: float, crowd: Optional[float] = None,
               family: str = "", position: str = "neutral") -> float:
    """The number to enter on the platform.

    No crowd visible -> submit the model probability (there is nothing to
    shrink toward; the blend already anchored to the betting market).
    """
    if crowd is None:
        return float(min(max(p_hat, SUBMIT_FLOOR), SUBMIT_CAP))
    tau = FAMILY_TAU.get(family, DEFAULT_TAU)
    kappa = KAPPA_BY_POSITION.get(position, 0.0)
    lam = optimal_lambda(p_hat - crowd, tau, kappa, p_hat)
    f = crowd + lam * (p_hat - crowd)
    return float(min(max(f, SUBMIT_FLOOR), SUBMIT_CAP))
