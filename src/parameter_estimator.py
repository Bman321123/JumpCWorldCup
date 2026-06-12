"""Dixon-Coles fit: weighted, penalized MLE with Elo-anchored shrinkage
(PRD v2.2 §4.2, §6.4).

- Time decay: w = exp(-ln2 * days_ago / half_life_days)  (fixes v1.0 bug B2)
- Importance weights: finals 1.0 > qualifiers 0.8 > friendlies 0.4
- Ridge toward Elo-implied strengths (or 0) for sparse teams
- Identification via mean-zero penalty on attack and defense vectors
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .count_math import decay_weight
from .stats_engine import ModelParameters

logger = logging.getLogger(__name__)

IMPORTANCE = [
    ("friendly", 0.4),
    ("qualification", 0.8),
    ("qualifier", 0.8),
    ("fifa world cup", 1.0),
    ("uefa euro", 0.95),
    ("copa am", 0.95),
    ("african cup", 0.9),
    ("africa cup", 0.9),
    ("afc asian cup", 0.9),
    ("gold cup", 0.85),
    ("nations league", 0.85),
]
DEFAULT_IMPORTANCE = 0.7


def importance_weight(tournament: str) -> float:
    t = (tournament or "").lower()
    if "friendly" in t:
        return 0.4
    if "qualification" in t or "qualifier" in t:
        return 0.8
    for key, w in IMPORTANCE:
        if key in t:
            return w
    return DEFAULT_IMPORTANCE


def fit_dixon_coles(
    df: pd.DataFrame,
    half_life_days: float = 500.0,
    ridge: float = 5.0,
    elo_attack_prior: Optional[Dict[str, float]] = None,
    elo_defense_prior: Optional[Dict[str, float]] = None,
    asof: Optional[str] = None,
    min_team_matches: int = 5,
    maxiter: int = 300,
) -> ModelParameters:
    """df columns: date, home_team, away_team, home_score, away_score, neutral,
    tournament. Rows after `asof` are excluded (as-of discipline for backtests)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    asof_dt = pd.to_datetime(asof) if asof else df["date"].max() + pd.Timedelta(days=1)
    df = df[df["date"] < asof_dt].dropna(subset=["home_score", "away_score"])

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    teams = sorted(counts[counts >= min_team_matches].index)
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]
    if df.empty:
        raise ValueError("No matches left after filtering; check inputs.")
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(dtype=float)
    ag = df["away_score"].to_numpy(dtype=float)
    home_flag = 1.0 - df["neutral"].fillna(0).astype(float).to_numpy()
    days_ago = (asof_dt - df["date"]).dt.days.to_numpy(dtype=float)
    w = np.array([decay_weight(d, half_life_days) for d in days_ago])
    w *= df["tournament"].fillna("").map(importance_weight).to_numpy()

    pa = np.array([(elo_attack_prior or {}).get(t, 0.0) for t in teams])
    pd_ = np.array([(elo_defense_prior or {}).get(t, 0.0) for t in teams])

    lg_h = gammaln(hg + 1.0)
    lg_a = gammaln(ag + 1.0)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)

    def nll(x: np.ndarray) -> float:
        mu, gamma, rho = x[0], x[1], x[2]
        A, D = x[3:3 + n], x[3 + n:]
        log_lh = mu + A[hi] - D[ai] + gamma * home_flag
        log_la = mu + A[ai] - D[hi]
        lh, la = np.exp(log_lh), np.exp(log_la)
        ll = hg * log_lh - lh - lg_h + ag * log_la - la - lg_a
        tau = np.ones_like(lh)
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m11] = 1.0 - rho
        ll += np.log(np.clip(tau, 1e-10, None))
        pen = ridge * (np.sum((A - pa) ** 2) + np.sum((D - pd_) ** 2))
        pen += 1000.0 * (A.mean() ** 2 + D.mean() ** 2)        # identification
        return -float(np.sum(w * ll)) + pen

    x0 = np.zeros(3 + 2 * n)
    x0[0], x0[1], x0[2] = 0.15, 0.25, -0.1
    bounds = [(-1.0, 1.5), (0.0, 0.8), (-0.3, 0.1)] + [(-3.0, 3.0)] * (2 * n)
    t0 = time.time()
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": 400000})
    logger.info("Dixon-Coles fit: %d teams, %d matches, %.1fs, nll=%.1f, converged=%s",
                n, len(df), time.time() - t0, res.fun, res.success)

    x = res.x
    return ModelParameters(
        mu=float(x[0]), gamma=float(x[1]), rho=float(x[2]),
        attack={t: float(x[3 + i]) for t, i in idx.items()},
        defense={t: float(x[3 + n + i]) for t, i in idx.items()},
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_cutoff=str(asof_dt.date()),
        half_life_days=half_life_days,
        n_matches_fit=int(len(df)),
    )


def elo_to_priors(elo: Dict[str, float], scale: float = 0.0009) -> Dict[str, Dict[str, float]]:
    """Map Elo ratings to attack/defense priors: ~0.9 strength points per 1000 Elo
    above the mean, split evenly between attack and defense."""
    if not elo:
        return {"attack": {}, "defense": {}}
    mean = float(np.mean(list(elo.values())))
    att = {t: (r - mean) * scale for t, r in elo.items()}
    dfn = {t: (r - mean) * scale for t, r in elo.items()}
    return {"attack": att, "defense": dfn}
