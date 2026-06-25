"""Market-implied expected goals (the keystone line+model fix).

On a lopsided match the sparse-data Dixon-Coles fit compresses the favorite
(Brazil rated ~equal to Morocco), so every goal-derived MICRO-market priced off
those lambdas underrates the favorite's dominance. The sharp 1X2 (+ total) line
knows the real team strengths. This module backs out the (lambda_home,
lambda_away) whose Dixon-Coles score matrix reproduces the devigged market 1X2
and over/under, and the orchestrator then prices first-goal/H2/comparative
questions off THOSE — the model supplies the shape, the market the scale.

When no market exists, the pipeline keeps the structural lambdas unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

logger = logging.getLogger(__name__)

MAX_GOALS = 12


def _matrix(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    M = poisson.pmf(g, lam_h)[:, None] * poisson.pmf(g, lam_a)[None, :]
    M[0, 0] *= max(1.0 - lam_h * lam_a * rho, 1e-10)
    M[1, 0] *= max(1.0 + lam_a * rho, 1e-10)
    M[0, 1] *= max(1.0 + lam_h * rho, 1e-10)
    M[1, 1] *= max(1.0 - rho, 1e-10)
    return M / M.sum()


def _outcomes(lam_h: float, lam_a: float, rho: float, total_line: Optional[float]):
    M = _matrix(lam_h, lam_a, rho)
    home = float(np.tril(M, -1).sum())
    away = float(np.triu(M, 1).sum())
    over = None
    if total_line is not None:
        g = np.arange(MAX_GOALS + 1)
        tot = g[:, None] + g[None, :]
        k = int(np.ceil(total_line))
        over = float(M[tot >= k].sum())
    return home, away, over


def market_implied_lambdas(home_win: float, away_win: float,
                           total_line: Optional[float], p_over: Optional[float],
                           rho: float, guess: Tuple[float, float] = (1.3, 1.1)
                           ) -> Optional[Tuple[float, float]]:
    """Solve for (lam_home, lam_away) matching the devigged market. Returns None
    on failure (caller keeps structural lambdas)."""
    # Scrapers hand totals back as STRING keys/values ("2.5"), which crashed the
    # solve at np.ceil(total_line) and silently killed the sharp-line anchor on
    # every match. Coerce defensively; bail to structural if anything isn't numeric.
    try:
        home_win, away_win = float(home_win), float(away_win)
        total_line = None if total_line is None else float(total_line)
        p_over = None if p_over is None else float(p_over)
    except (TypeError, ValueError):
        return None
    have_total = total_line is not None and p_over is not None

    def resid(x):
        lh, la = float(np.exp(x[0])), float(np.exp(x[1]))   # positivity
        h, a, over = _outcomes(lh, la, rho, total_line if have_total else None)
        r = [h - home_win, a - away_win]
        if have_total:
            r.append((over - p_over))
        return r

    try:
        sol = least_squares(resid, np.log(np.array(guess)), method="lm",
                            max_nfev=200)
        lh, la = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
        if not (0.1 <= lh <= 6 and 0.1 <= la <= 6):
            return None
        # sanity: must reproduce 1X2 ordering within tolerance
        h, a, _ = _outcomes(lh, la, rho, None)
        if abs(h - home_win) > 0.05 or abs(a - away_win) > 0.05:
            logger.info("Market-lambda fit loose (h %.3f vs %.3f); skipping.",
                        h, home_win)
            return None
        return lh, la
    except Exception as e:                       # noqa: BLE001
        logger.warning("Market-lambda solve failed: %s", e)
        return None
