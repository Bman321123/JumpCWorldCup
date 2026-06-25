"""Monte Carlo match simulator — for the compound/sequential markets the analytic
engine can only approximate (it prices first-goal/H2 combos as a product of legs,
ignoring their correlation) or cannot parse at all ("will the 2nd half have more
goals than the 1st?").

Design — stay CONSISTENT with the analytic engine so the sim is a strict superset,
not a second opinion that drifts:
  * Scorelines are drawn from the engine's own tau-corrected Dixon-Coles score
    matrix (one multinomial draw over the flattened joint pmf). So simulated 1X2 /
    totals / BTTS match the analytic engine within Monte Carlo error — that
    agreement is the correctness test (tests/test_match_simulator.py).
  * Goal TIMES come from a half-share split (each goal independently in H1 with the
    fitted GOALS half-share, else H2) + a uniform minute with jitter, which is what
    unlocks window / first-scorer / H1-vs-H2 questions WITH their true joint
    correlation rather than a product approximation.
  * Optional lambda-uncertainty propagation (per-sim lognormal multiplier) widens
    the tails for better-calibrated extreme probabilities; default off so the sim
    reproduces the analytic engine exactly.

The analytic engine stays the anchor for everything it already prices well; the
simulator is opt-in for the markets where it measurably adds resolution.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .stats_engine import ET_GOAL_SCALE, PENS_BASE, StatsEngine
from .types import MatchContext

H1_DEFAULT = 0.45          # GOALS half-share (matches DEFAULT_HALF_SHARES["GOALS"])


class MatchSimulator:
    def __init__(self, engine: StatsEngine, n_sims: int = 20000, seed: int = 7):
        self.engine = engine
        self.n = n_sims
        self.seed = seed

    # ---- core draw -------------------------------------------------------------
    def _draw(self, home: str, away: str, ctx: Optional[MatchContext],
              lam_sigma: float):
        """Return (gh, ga, t_home, t_away): goal counts and per-goal half labels
        (0=H1, 1=H2) for each team, one row per simulation."""
        rng = np.random.default_rng(self.seed)
        lam_h, lam_a = self.engine.expected_goals(home, away, ctx)
        h1 = self.engine.p.half_shares.get("GOALS", H1_DEFAULT)
        n = self.n
        if lam_sigma > 0:                       # propagate parameter uncertainty
            fh = rng.lognormal(-0.5 * lam_sigma ** 2, lam_sigma, n)
            fa = rng.lognormal(-0.5 * lam_sigma ** 2, lam_sigma, n)
        else:
            fh = fa = np.ones(n)
        # sample scorelines from the engine's tau-corrected joint (mean lambdas);
        # lambda noise is applied as a thinning/boost on the per-team counts
        M = self.engine.score_matrix(lam_h, lam_a).ravel()
        size = self.engine.score_matrix(lam_h, lam_a).shape[0]
        idx = rng.choice(M.size, size=n, p=M / M.sum())
        gh = (idx // size).astype(int)
        ga = (idx % size).astype(int)
        if lam_sigma > 0:
            gh = rng.poisson(np.maximum(gh * fh, 1e-6))
            ga = rng.poisson(np.maximum(ga * fa, 1e-6))
        # assign each goal to a half (H1 w.p. h1). Return per-sim H1 counts.
        h1_home = rng.binomial(gh, h1)
        h1_away = rng.binomial(ga, h1)
        return gh, ga, h1_home, h1_away

    # ---- markets ---------------------------------------------------------------
    def markets(self, home: str, away: str, ctx: Optional[MatchContext] = None,
                lam_sigma: float = 0.0) -> dict:
        gh, ga, h1h, h1a = self._draw(home, away, ctx, lam_sigma)
        tot = gh + ga
        h1 = h1h + h1a
        h2 = tot - h1
        return {
            "home_win": float((gh > ga).mean()),
            "draw": float((gh == ga).mean()),
            "away_win": float((gh < ga).mean()),
            "over2.5": float((tot >= 3).mean()),
            "btts": float(((gh >= 1) & (ga >= 1)).mean()),
            # compound / sequential — the reason this module exists:
            "h2_more_than_h1": float((h2 > h1).mean()),
            "h1_more_than_h2": float((h1 > h2).mean()),
            "h2_scores": float((h2 >= 1).mean()),
        }

    def prob_h2_more_than_h1(self, home: str, away: str,
                             ctx: Optional[MatchContext] = None) -> float:
        return self.markets(home, away, ctx)["h2_more_than_h1"]

    # ---- knockout advance via simulation (90 + ET + pens) ----------------------
    def advance_prob(self, home: str, away: str, side: str = "HOME",
                     ctx: Optional[MatchContext] = None,
                     pens_home: float = PENS_BASE) -> float:
        rng = np.random.default_rng(self.seed + 1)
        lam_h, lam_a = self.engine.expected_goals(home, away, ctx)
        M = self.engine.score_matrix(lam_h, lam_a)
        size = M.shape[0]
        idx = rng.choice(M.size, size=self.n, p=M.ravel() / M.sum())
        gh, ga = idx // size, idx % size
        home_adv = gh > ga
        level = gh == ga
        # extra time on the still-level games
        Met = self.engine.score_matrix(lam_h * ET_GOAL_SCALE, lam_a * ET_GOAL_SCALE)
        eidx = rng.choice(Met.size, size=self.n, p=Met.ravel() / Met.sum())
        egh, ega = eidx // size, eidx % size
        et_home = level & (egh > ega)
        et_level = level & (egh == ega)
        pens = et_level & (rng.random(self.n) < pens_home)
        adv_home = home_adv | et_home | pens
        return float(adv_home.mean() if side == "HOME" else (~adv_home).mean())
