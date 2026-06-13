"""Statistical model layer (PRD v2.2 §4.2, §4.4, §4.5, §4.7, §4.8, §6.5).

Every goal-derived market reads off one tau-corrected Dixon-Coles score matrix
(fixes v1.0 bug B6 — BTTS/totals previously ignored the correction). Count
thresholds use ceil semantics from count_math (bug B1).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import nbinom, poisson

from .count_math import count_prob
from .types import Condition, MatchContext, TemporalWindow

logger = logging.getLogger(__name__)

MAX_GOALS = 12
ET_GOAL_SCALE = 0.30          # 30/90 minutes * ~0.9 tempo factor (PRD §4.7)
PENS_BASE = 0.50

# H1 share of full-match counts; H2 = 1 - share (PRD §4.8, fixes B9 incl. cards)
DEFAULT_HALF_SHARES = {"GOALS": 0.45, "CORNERS": 0.46, "CARDS": 0.33,
                       "OFFSIDES": 0.48, "SHOTS": 0.46, "FOULS": 0.47}

# Per-team per-match fallback rates when no fitted value exists
DEFAULTS = {
    "corner_for": 4.9, "corner_against": 4.9,
    "yellow": 1.7, "red": 0.09, "offside": 2.0,
    "yellow_var_ratio": 1.5, "red_var_ratio": 1.55,
    "sot_for": 4.3,                  # shots on target per team per match
    "fouls": 12.0,                   # fouls committed per team per match
    "fouls_var_ratio": 1.3,
    "penalty_awarded": 0.29,         # P(any penalty kick awarded), VAR era
}


@dataclass
class ModelParameters:
    mu: float = 0.18
    gamma: float = 0.25              # home advantage; applied only for hosts at the WC
    rho: float = -0.11
    attack: Dict[str, float] = field(default_factory=dict)
    defense: Dict[str, float] = field(default_factory=dict)
    corner_for: Dict[str, float] = field(default_factory=dict)
    corner_against: Dict[str, float] = field(default_factory=dict)
    yellow_rates: Dict[str, float] = field(default_factory=dict)
    red_rates: Dict[str, float] = field(default_factory=dict)
    offside_rates: Dict[str, float] = field(default_factory=dict)
    sot_rates: Dict[str, float] = field(default_factory=dict)
    fouls_rates: Dict[str, float] = field(default_factory=dict)
    half_shares: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HALF_SHARES))
    fitted_at: str = ""
    data_cutoff: str = ""
    half_life_days: float = 500.0
    n_matches_fit: int = 0

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "ModelParameters":
        with open(path) as f:
            d = json.load(f)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def nbinom_from_mean(mu: float, var_ratio: float):
    """NB parameterized by mean and variance ratio; scipy nbinom(n, p) with
    p = mu/var, n = mu^2/(var-mu) gives mean mu, variance var (PRD §4.5, B7)."""
    mu = max(mu, 1e-6)
    var = max(mu * var_ratio, mu + 1e-6)
    p = mu / var
    n = mu * mu / (var - mu)
    return nbinom(n, p)


class StatsEngine:
    def __init__(self, params: ModelParameters):
        self.p = params

    # ----- lambdas -----

    def expected_goals(self, home: str, away: str, ctx: Optional[MatchContext] = None
                       ) -> Tuple[float, float]:
        # market-implied lambdas (when a sharp line exists) replace the
        # compressed structural estimate; late-news absence multipliers still
        # apply on top in case they post-date the scraped line.
        if ctx and ctx.lambda_home_override is not None:
            return (ctx.lambda_home_override * ctx.home_absence_mult,
                    ctx.lambda_away_override * ctx.away_absence_mult)
        a_h = self._strength(self.p.attack, home)
        d_h = self._strength(self.p.defense, home)
        a_a = self._strength(self.p.attack, away)
        d_a = self._strength(self.p.defense, away)
        adv_h = self.p.gamma if (ctx and ctx.home_is_host) else 0.0
        adv_a = self.p.gamma if (ctx and ctx.away_is_host) else 0.0
        gm = ctx.goal_multiplier if ctx else 1.0
        lam_h = float(np.exp(self.p.mu + a_h - d_a + adv_h)) * gm
        lam_a = float(np.exp(self.p.mu + a_a - d_h + adv_a)) * gm
        if ctx:
            lam_h *= ctx.home_absence_mult
            lam_a *= ctx.away_absence_mult
        return lam_h, lam_a

    def _strength(self, table: Dict[str, float], team: str) -> float:
        if team not in table:
            logger.warning("No fitted strength for %s; using league average (0.0).", team)
        return table.get(team, 0.0)

    # ----- score matrix and goal-derived markets -----

    def score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        g = np.arange(MAX_GOALS + 1)
        M = poisson.pmf(g, lam_h)[:, None] * poisson.pmf(g, lam_a)[None, :]
        rho = self.p.rho
        M[0, 0] *= max(1.0 - lam_h * lam_a * rho, 1e-10)
        M[1, 0] *= max(1.0 + lam_a * rho, 1e-10)
        M[0, 1] *= max(1.0 + lam_h * rho, 1e-10)
        M[1, 1] *= max(1.0 - rho, 1e-10)
        return M / M.sum()

    def _window_matrix(self, home: str, away: str, window: TemporalWindow,
                       ctx: Optional[MatchContext]) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home, away, ctx)
        s = self._share("GOALS", window)
        return self.score_matrix(lam_h * s, lam_a * s)

    def _share(self, metric: str, window: TemporalWindow) -> float:
        if window == TemporalWindow.FULL:
            return 1.0
        h1 = self.p.half_shares.get(metric, DEFAULT_HALF_SHARES[metric])
        return h1 if window == TemporalWindow.H1 else 1.0 - h1

    def result_probs(self, home: str, away: str, ctx: Optional[MatchContext] = None,
                     window: TemporalWindow = TemporalWindow.FULL) -> Dict[str, float]:
        """1X2 probabilities; window=H1 answers 'tied/leading at halftime'."""
        M = self._window_matrix(home, away, window, ctx)
        hw, dr, aw = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
        t = hw + dr + aw
        return {"home_win": hw / t, "draw": dr / t, "away_win": aw / t}

    def goal_market(self, home: str, away: str, metric: str, target: str,
                    threshold: float, condition: Condition,
                    window: TemporalWindow = TemporalWindow.FULL,
                    ctx: Optional[MatchContext] = None) -> float:
        M = self._window_matrix(home, away, window, ctx)
        if metric == "BTTS":
            return float(M[1:, 1:].sum())
        if metric == "BTTS_AND_TOTAL":
            # compound (observed live): both teams score AND total >= threshold,
            # exact from the same tau-corrected matrix
            g = np.arange(MAX_GOALS + 1)
            tot = g[:, None] + g[None, :]
            k = int(np.ceil(threshold))
            mask = (g[:, None] >= 1) & (g[None, :] >= 1) & (tot >= k)
            return float(M[mask].sum())
        if metric == "CLEAN_SHEET":
            if target == "HOME":
                return float(M[:, 0].sum())          # away scores 0
            if target == "AWAY":
                return float(M[0, :].sum())
            return float(M[:, 0].sum() + M[0, :].sum() - M[0, 0])
        # totals / team totals
        if target == "HOME":
            pmf = M.sum(axis=1)
        elif target == "AWAY":
            pmf = M.sum(axis=0)
        else:
            g = np.arange(MAX_GOALS + 1)
            tot = g[:, None] + g[None, :]
            pmf = np.bincount(tot.ravel(), weights=M.ravel())
        cdfv = np.cumsum(pmf)
        cdf = lambda k: 0.0 if k < 0 else float(cdfv[min(k, len(cdfv) - 1)])
        return count_prob(cdf, threshold, condition)

    def first_goal_combo(self, home: str, away: str, first_side: str,
                         scorer_side: str, scorer_window: TemporalWindow,
                         ctx: Optional[MatchContext] = None) -> float:
        """P(first goal by `first_side` AND `scorer_side` scores in window).
        Leg 1: race of two Poissons — P(side first | any goal) = lam_side/total,
        times P(any goal). Leg 2: window marginal. Joint approximated as the
        product (legs are mildly positively correlated through 'goals happen';
        acceptable at these magnitudes — observed-live question type)."""
        lam_h, lam_a = self.expected_goals(home, away, ctx)
        p_any = 1.0 - float(np.exp(-(lam_h + lam_a)))
        lam_first = lam_h if first_side == "HOME" else lam_a
        p_first = p_any * lam_first / (lam_h + lam_a)
        p_scores = self.goal_market(home, away, "GOALS",
                                    scorer_side, 1.0, Condition.GTE,
                                    scorer_window, ctx)
        return p_first * p_scores

    def first_in_window(self, home: str, away: str, side: str,
                        window: TemporalWindow, ctx: Optional[MatchContext] = None) -> float:
        """P(`side` scores the FIRST goal within the window). A race of the two
        teams' goal rates over that window: P(any goal) * share of the rate.
        Distinct from 'team scores in window' (which ignores who's first)."""
        lam_h, lam_a = self.expected_goals(home, away, ctx)
        s = self._share("GOALS", window)
        lam_h, lam_a = lam_h * s, lam_a * s
        tot = lam_h + lam_a
        if tot <= 0:
            return 0.0
        p_any = 1.0 - float(np.exp(-tot))
        lam_side = lam_h if side == "HOME" else lam_a
        return p_any * lam_side / tot

    # ----- knockout: extra time & penalties (PRD §4.7, fixes B8) -----

    def advance_prob(self, home: str, away: str, side: str = "HOME",
                     ctx: Optional[MatchContext] = None, pens_home: float = PENS_BASE) -> float:
        lam_h, lam_a = self.expected_goals(home, away, ctx)
        M90 = self.score_matrix(lam_h, lam_a)
        p_hw, p_dr = float(np.tril(M90, -1).sum()), float(np.trace(M90))
        p_aw = 1.0 - p_hw - p_dr
        Met = self.score_matrix(lam_h * ET_GOAL_SCALE, lam_a * ET_GOAL_SCALE)
        et_hw, et_dr = float(np.tril(Met, -1).sum()), float(np.trace(Met))
        et_aw = 1.0 - et_hw - et_dr
        adv_home = p_hw + p_dr * (et_hw + et_dr * pens_home)
        adv_away = p_aw + p_dr * (et_aw + et_dr * (1.0 - pens_home))
        return adv_home if side == "HOME" else adv_away

    # ----- corners (PRD §4.5: opponent-adjusted, fixes v1.0 gap) -----

    def corner_lambdas(self, home: str, away: str, ctx: Optional[MatchContext] = None
                       ) -> Tuple[float, float]:
        cf_h = self.p.corner_for.get(home, DEFAULTS["corner_for"])
        cf_a = self.p.corner_for.get(away, DEFAULTS["corner_for"])
        ca_h = self.p.corner_against.get(home, DEFAULTS["corner_against"])
        ca_a = self.p.corner_against.get(away, DEFAULTS["corner_against"])
        avg = DEFAULTS["corner_against"]
        cm = ctx.corner_multiplier if ctx else 1.0
        return cf_h * (ca_a / avg) * cm, cf_a * (ca_h / avg) * cm

    def corner_market(self, home: str, away: str, target: str, threshold: float,
                      condition: Condition, window: TemporalWindow = TemporalWindow.FULL,
                      ctx: Optional[MatchContext] = None) -> float:
        lam_h, lam_a = self.corner_lambdas(home, away, ctx)
        s = self._share("CORNERS", window)
        lam = {"HOME": lam_h, "AWAY": lam_a}.get(target, lam_h + lam_a) * s
        return count_prob(lambda k: float(poisson.cdf(k, lam)), threshold, condition)

    # ----- cards (PRD §4.5: NB, referee + motivation multipliers) -----

    def card_market(self, home: str, away: str, target: str, card_type: str,
                    threshold: float, condition: Condition,
                    window: TemporalWindow = TemporalWindow.FULL,
                    ctx: Optional[MatchContext] = None, ref_mult: float = 1.0) -> float:
        if card_type == "REDS":
            base, ratio = self.p.red_rates, DEFAULTS["red_var_ratio"]
            default = DEFAULTS["red"]
        elif card_type == "FOULS":
            base, ratio = self.p.fouls_rates, DEFAULTS["fouls_var_ratio"]
            default = DEFAULTS["fouls"]
        else:
            base, ratio = self.p.yellow_rates, DEFAULTS["yellow_var_ratio"]
            default = DEFAULTS["yellow"]
        intensity = ctx.card_intensity if ctx else 1.0
        mu_h = base.get(home, default) * ref_mult * intensity
        mu_a = base.get(away, default) * ref_mult * intensity
        if card_type == "CARDS":             # "total cards" = yellows + reds
            mu_h += self.p.red_rates.get(home, DEFAULTS["red"]) * ref_mult * intensity
            mu_a += self.p.red_rates.get(away, DEFAULTS["red"]) * ref_mult * intensity
        mu = {"HOME": mu_h, "AWAY": mu_a}.get(target, mu_h + mu_a)
        mu *= self._share("FOULS" if card_type == "FOULS" else "CARDS", window)
        dist = nbinom_from_mean(mu, ratio)
        return count_prob(lambda k: float(dist.cdf(k)), threshold, condition)

    # ----- offsides (PRD §4.5: heavily shrunk Poisson) -----

    def offside_market(self, home: str, away: str, target: str, threshold: float,
                       condition: Condition, window: TemporalWindow = TemporalWindow.FULL,
                       ctx: Optional[MatchContext] = None) -> float:
        lam_h = self.p.offside_rates.get(home, DEFAULTS["offside"])
        lam_a = self.p.offside_rates.get(away, DEFAULTS["offside"])
        lam = {"HOME": lam_h, "AWAY": lam_a}.get(target, lam_h + lam_a)
        lam *= self._share("OFFSIDES", window)
        return count_prob(lambda k: float(poisson.cdf(k, lam)), threshold, condition)

    # ----- shots on target (observed question type, 2026-06-11) -----

    def shots_market(self, home: str, away: str, target: str, threshold: float,
                     condition: Condition, window: TemporalWindow = TemporalWindow.FULL,
                     ctx: Optional[MatchContext] = None) -> float:
        lam_h, lam_a = self._sot_lambdas(home, away, ctx)
        s = self._share("SHOTS", window)
        lam = {"HOME": lam_h, "AWAY": lam_a}.get(target, lam_h + lam_a) * s
        return count_prob(lambda k: float(poisson.cdf(k, lam)), threshold, condition)

    def _sot_lambdas(self, home: str, away: str,
                     ctx: Optional[MatchContext]) -> Tuple[float, float]:
        """SOT scales with attacking strength: anchor the default on the ratio of
        the team's expected goals to the league-average expected goals."""
        lam_h, lam_a = self.expected_goals(home, away, ctx)
        avg_goals = float(np.exp(self.p.mu))
        base = DEFAULTS["sot_for"]
        sot_h = self.p.sot_rates.get(home, base * lam_h / avg_goals)
        sot_a = self.p.sot_rates.get(away, base * lam_a / avg_goals)
        return sot_h, sot_a

    # ----- comparative markets: P(team X stat > team Y stat) (observed live) -----

    def comparative_prob(self, home: str, away: str, metric: str, target: str,
                         window: TemporalWindow = TemporalWindow.FULL,
                         ctx: Optional[MatchContext] = None) -> float:
        """P(target team's count is STRICTLY greater than the opponent's),
        independent Poissons. Ties count as NO — exactly the live phrasing
        'will X have more ... than Y'."""
        if metric == "CORNERS":
            lam_h, lam_a = self.corner_lambdas(home, away, ctx)
            share = self._share("CORNERS", window)
        elif metric == "SOT":
            lam_h, lam_a = self._sot_lambdas(home, away, ctx)
            share = self._share("SHOTS", window)
        elif metric in ("CARDS", "YELLOWS"):
            lam_h = self.p.yellow_rates.get(home, DEFAULTS["yellow"])
            lam_a = self.p.yellow_rates.get(away, DEFAULTS["yellow"])
            share = self._share("CARDS", window)
        elif metric == "FOULS":
            lam_h = self.p.fouls_rates.get(home, DEFAULTS["fouls"])
            lam_a = self.p.fouls_rates.get(away, DEFAULTS["fouls"])
            share = self._share("FOULS", window)
        elif metric == "OFFSIDES":
            lam_h = self.p.offside_rates.get(home, DEFAULTS["offside"])
            lam_a = self.p.offside_rates.get(away, DEFAULTS["offside"])
            share = self._share("OFFSIDES", window)
        elif metric == "GOALS":
            lam_h, lam_a = self.expected_goals(home, away, ctx)
            share = self._share("GOALS", window)
        else:
            raise ValueError(f"comparative_prob: unsupported metric {metric}")
        lam_h, lam_a = lam_h * share, lam_a * share
        if target == "AWAY":
            lam_h, lam_a = lam_a, lam_h
        return _poisson_greater(lam_h, lam_a)

    # ----- "both teams >= k <metric>" (observed live 2026-06-13) -----

    def both_teams_prob(self, home: str, away: str, metric: str, threshold: float,
                        window: TemporalWindow = TemporalWindow.FULL,
                        ctx: Optional[MatchContext] = None,
                        ref_mult: float = 1.0) -> float:
        """P(home count >= k AND away count >= k). Independence approximation
        per team. v1 of the parser priced 'both teams 1+ SOT at halftime' as
        TOTAL SOT >= 1 (97% on a ~68% event) — this is the fix."""
        from .count_math import count_prob
        k = threshold
        if metric in ("CARDS", "YELLOWS", "REDS"):
            ps = []
            for target in ("HOME", "AWAY"):
                ps.append(self.card_market(home, away, target, metric, k,
                                           Condition.GTE, window, ctx, ref_mult))
            return ps[0] * ps[1]
        if metric == "CORNERS":
            lam_h, lam_a = self.corner_lambdas(home, away, ctx)
            share = self._share("CORNERS", window)
        elif metric == "SOT":
            lam_h, lam_a = self._sot_lambdas(home, away, ctx)
            share = self._share("SHOTS", window)
        elif metric == "FOULS":
            lam_h = self.p.fouls_rates.get(home, DEFAULTS["fouls"])
            lam_a = self.p.fouls_rates.get(away, DEFAULTS["fouls"])
            share = self._share("FOULS", window)
        elif metric == "OFFSIDES":
            lam_h = self.p.offside_rates.get(home, DEFAULTS["offside"])
            lam_a = self.p.offside_rates.get(away, DEFAULTS["offside"])
            share = self._share("OFFSIDES", window)
        else:
            raise ValueError(f"both_teams_prob: unsupported metric {metric}")
        p_h = count_prob(lambda j: float(poisson.cdf(j, lam_h * share)), k,
                         Condition.GTE)
        p_a = count_prob(lambda j: float(poisson.cdf(j, lam_a * share)), k,
                         Condition.GTE)
        return p_h * p_a

    # ----- penalty awarded (observed question type) -----

    def penalty_prob(self, ctx: Optional[MatchContext] = None) -> float:
        p = DEFAULTS["penalty_awarded"]
        if ctx and ctx.card_intensity > 1.0:
            p *= 1.05                        # scrappy/high-stakes matches: slight bump
        return min(p, 0.40)


def _poisson_greater(lam_x: float, lam_y: float, n_max: int = 40) -> float:
    """P(X > Y) for independent Poissons."""
    ks = np.arange(n_max + 1)
    py = poisson.pmf(ks, lam_y)
    p_x_gt = 1.0 - poisson.cdf(ks, lam_x)    # P(X > k) per k
    return float(np.sum(py * p_x_gt))
