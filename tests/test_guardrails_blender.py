import pytest

from src.ensemble_blender import EnsembleBlender
from src.guardrail_validator import GuardrailValidator, pav_non_increasing
from src.types import (Condition, ParsedQuestion, Prediction, QuestionFamily,
                       ResultScope, TemporalWindow)


def _q(qid, family, target, threshold, condition, scope=ResultScope.NONE,
       metric="GOALS"):
    return ParsedQuestion(qid, qid, family, "ARG", "FRA", target, metric,
                          threshold, condition, TemporalWindow.FULL, scope, 1.0)


def _p(qid, p):
    return Prediction(qid, qid, "GOAL_MARKET", p, p, None, p, p)


def test_pav_projection_is_non_increasing():
    out = pav_non_increasing([0.8, 0.85, 0.5, 0.55, 0.2])
    assert all(out[i] >= out[i + 1] - 1e-12 for i in range(len(out) - 1))
    assert out[0] == pytest.approx(0.825)            # pooled violator


def test_ladder_monotonicity_enforced():
    v = GuardrailValidator()
    parsed = [_q(f"Q{i}", QuestionFamily.GOAL_MARKET, "MATCH", t, Condition.GTE)
              for i, t in enumerate([1.5, 2.5, 3.5])]
    preds = [_p("Q0", 0.70), _p("Q1", 0.75), _p("Q2", 0.30)]  # Q1 violates
    out = v.validate_match_set(preds, parsed)
    vals = [p.p_final for p in out]
    assert vals[0] >= vals[1] >= vals[2]


def test_trio_renormalized():
    v = GuardrailValidator()
    parsed = [
        _q("H", QuestionFamily.MATCH_RESULT, "HOME", 0, Condition.BINARY_YES,
           ResultScope.WIN_90, "RESULT"),
        _q("D", QuestionFamily.MATCH_RESULT, "DRAW", 0, Condition.BINARY_YES,
           ResultScope.WIN_90, "RESULT"),
        _q("A", QuestionFamily.MATCH_RESULT, "AWAY", 0, Condition.BINARY_YES,
           ResultScope.WIN_90, "RESULT"),
    ]
    preds = [_p("H", 0.5), _p("D", 0.3), _p("A", 0.3)]   # sums to 1.1
    out = v.validate_match_set(preds, parsed)
    assert sum(p.p_final for p in out) == pytest.approx(1.0, abs=0.01)


def test_clip_is_last():
    v = GuardrailValidator()
    preds = [_p("Q0", 0.99999), _p("Q1", 1e-7)]
    out = v.validate_match_set(preds, [])
    assert out[0].p_final == 0.999
    assert out[1].p_final == 0.001


def test_blend_no_market_passthrough():
    b = EnsembleBlender()
    p, src = b.blend(None, 0.62, "GOAL_MARKET")
    assert p == 0.62 and src == "model"


def test_blend_sits_between():
    b = EnsembleBlender()
    p, _ = b.blend(0.50, 0.70, "GOAL_MARKET", is_closing_line=True)
    assert 0.50 < p < 0.70
    assert p < 0.60                                  # market-heavy (w=0.75)


def test_early_line_discount():
    b = EnsembleBlender()
    p_close, _ = b.blend(0.50, 0.70, "GOAL_MARKET", is_closing_line=True)
    p_early, _ = b.blend(0.50, 0.70, "GOAL_MARKET", is_closing_line=False)
    assert p_early > p_close                         # early line trusted less


def test_market_deference_on_main_markets():
    """A compressed model must not drag a sharp 1X2 line. Large gap -> defer."""
    b = EnsembleBlender()
    # market 0.58, model 0.36 (the Brazil bug): base weight 0.80
    p, src = b.blend(0.58, 0.36, "MATCH_RESULT", is_closing_line=True)
    assert p > 0.54                       # pulled close to the sharp 0.58, not 0.51
    # small disagreement: normal blend, model still contributes
    p2, _ = b.blend(0.55, 0.50, "MATCH_RESULT", is_closing_line=True)
    assert 0.52 < p2 < 0.555


def test_no_deference_on_micro_markets():
    """Thin micro-markets keep the validated model's deviation (low base weight)."""
    b = EnsembleBlender()
    # corners base weight 0.40; model 0.30 vs market 0.55 — model must still pull
    p, _ = b.blend(0.55, 0.30, "CORNER_MARKET", is_closing_line=True)
    assert p < 0.50                       # model deviation preserved, not deferred away


def test_consensus_shrink_bounds():
    b = EnsembleBlender()
    assert b.consensus_shrink(0.7, 0.5, 1.0) == pytest.approx(0.7)
    assert b.consensus_shrink(0.7, 0.5, 0.0) == pytest.approx(0.5)
