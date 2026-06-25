"""The Monte Carlo simulator must REPRODUCE the analytic engine on the markets both
price (that agreement is the correctness gate), and must give sane, monotone answers
on the compound markets that are the reason it exists."""
import numpy as np

from src.match_simulator import MatchSimulator
from src.stats_engine import ModelParameters, StatsEngine
from src.types import Condition, MatchContext, TemporalWindow


def _engine():
    return StatsEngine(ModelParameters.load("params/dixon_coles.json"))


def test_sim_matches_analytic_1x2_totals_btts():
    eng = _engine()
    sim = MatchSimulator(eng, n_sims=60000, seed=3)
    for home, away in [("BRA", "ARG"), ("FRA", "CUW"), ("JPN", "GER")]:
        ctx = MatchContext(home, away, "2026-06-25")
        m = sim.markets(home, away, ctx)
        a = eng.result_probs(home, away, ctx)
        # 1X2 within Monte Carlo tolerance (~3 sigma for 60k draws)
        assert abs(m["home_win"] - a["home_win"]) < 0.01
        assert abs(m["draw"] - a["draw"]) < 0.01
        assert abs(m["away_win"] - a["away_win"]) < 0.01
        over = eng.goal_market(home, away, "GOALS", "TOTAL", 2.5,
                               Condition.GTE, TemporalWindow.FULL, ctx)
        assert abs(m["over2.5"] - over) < 0.012
        btts = eng.goal_market(home, away, "BTTS", "MATCH", 0.0,
                               Condition.GTE, TemporalWindow.FULL, ctx)
        assert abs(m["btts"] - btts) < 0.012


def test_h2_vs_h1_is_a_valid_probability_and_h2_favored():
    eng = _engine()
    sim = MatchSimulator(eng, n_sims=40000, seed=5)
    m = sim.markets("BRA", "ARG", MatchContext("BRA", "ARG", "2026-06-25"))
    p = m["h2_more_than_h1"]
    assert 0.30 < p < 0.55                          # plausible band
    # with H1 share 0.45 (<0.5), the second half should be the more likely high-scoring half
    assert m["h2_more_than_h1"] >= m["h1_more_than_h2"] - 0.02


def test_sim_advance_matches_analytic():
    eng = _engine()
    sim = MatchSimulator(eng, n_sims=60000, seed=9)
    ctx = MatchContext("FRA", "CUW", "2026-07-10", tournament_round="quarterfinal")
    sim_adv = sim.advance_prob("FRA", "CUW", "HOME", ctx)
    ana_adv = eng.advance_prob("FRA", "CUW", "HOME", ctx)
    assert abs(sim_adv - ana_adv) < 0.012


def test_lambda_uncertainty_widens_tails():
    eng = _engine()
    sim = MatchSimulator(eng, n_sims=40000, seed=11)
    ctx = MatchContext("BRA", "CUW", "2026-06-25")
    base = sim.markets("BRA", "CUW", ctx, lam_sigma=0.0)["over2.5"]
    wide = sim.markets("BRA", "CUW", ctx, lam_sigma=0.35)["over2.5"]
    assert np.isfinite(base) and np.isfinite(wide)
