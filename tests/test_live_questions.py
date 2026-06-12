"""Replay of the ACTUAL question set from CZE vs KOR, 2026-06-11 (Group A).

This is the ground truth for what the platform asks: comparative markets,
compound conjunctions, penalty-awarded, shots on target, and player props.
Every question must parse (no fallbacks) and produce a sane probability.
"""
from pathlib import Path

import pytest

from src.orchestrator import Orchestrator
from src.question_classifier import QuestionClassifier
from src.stats_engine import ModelParameters
from src.types import Condition, QuestionFamily, TemporalWindow

ROOT = Path(__file__).resolve().parents[1]

# home = CZE, away = KOR (order as listed on the platform's match page)
LIVE_QUESTIONS = [
    "At halftime, will Czechia have more corner kicks than South Korea?",
    "Will South Korea be caught offside 2 or more times?",
    "Will there be 2 or more total cards shown in the second half?",
    "Will both teams score AND the match have 3 or more total goals?",
    "Will a penalty kick be awarded in the match?",
    "Will South Korea have more shots on target than Czechia in the second half?",
    "Will South Korea win the match?",
    "Will South Korea score in the second half?",
    "Will Son Heung-min score a goal (excluding own goals)?",
    "Will Patrik Schick have at least 1 shot on target?",
]


@pytest.fixture
def clf():
    return QuestionClassifier(str(ROOT / "config" / "groups.json"), {"group": 1.0})


@pytest.fixture
def orch(tmp_path):
    params = ModelParameters(
        mu=0.18, gamma=0.25, rho=-0.12,
        attack={"CZE": 0.10, "KOR": 0.15}, defense={"CZE": 0.05, "KOR": 0.10})
    p = tmp_path / "params.json"
    params.save(str(p))
    return Orchestrator(config_dir=str(ROOT / "config"), params_path=str(p),
                        online=False)


def test_all_live_questions_parse_without_fallback(clf):
    for text in LIVE_QUESTIONS:
        q = clf.parse(text, "CZE", "KOR", "group")
        assert q is not None, text


def test_q1_comparative_corners_at_halftime(clf):
    q = clf.parse(LIVE_QUESTIONS[0], "CZE", "KOR", "group")
    assert q.condition == Condition.MORE_THAN_OPP
    assert q.family == QuestionFamily.CORNER_MARKET
    assert q.target == "HOME"                 # Czechia is the subject, not Korea
    assert q.window == TemporalWindow.H1


def test_q3_total_cards_includes_reds(clf):
    q = clf.parse(LIVE_QUESTIONS[2], "CZE", "KOR", "group")
    assert q.family == QuestionFamily.CARD_MARKET
    assert q.metric == "CARDS"
    assert q.window == TemporalWindow.H2
    assert q.threshold == 2.0 and q.condition == Condition.GTE


def test_q4_compound(clf):
    q = clf.parse(LIVE_QUESTIONS[3], "CZE", "KOR", "group")
    assert q.metric == "BTTS_AND_TOTAL"
    assert q.threshold == 3.0


def test_q5_penalty(clf):
    q = clf.parse(LIVE_QUESTIONS[4], "CZE", "KOR", "group")
    assert q.family == QuestionFamily.PENALTY_MARKET


def test_q6_comparative_sot_h2(clf):
    q = clf.parse(LIVE_QUESTIONS[5], "CZE", "KOR", "group")
    assert q.condition == Condition.MORE_THAN_OPP
    assert q.family == QuestionFamily.SHOTS_MARKET
    assert q.target == "AWAY"                 # South Korea is the subject
    assert q.window == TemporalWindow.H2


def test_q9_q10_player_props(clf):
    q9 = clf.parse(LIVE_QUESTIONS[8], "CZE", "KOR", "group")
    assert q9.family == QuestionFamily.PLAYER_MARKET and q9.metric == "PLAYER_GOAL"
    assert "Son" in q9.target
    q10 = clf.parse(LIVE_QUESTIONS[9], "CZE", "KOR", "group")
    assert q10.family == QuestionFamily.PLAYER_MARKET and q10.metric == "PLAYER_SOT"
    assert q10.threshold == 1.0


def test_full_live_set_end_to_end(orch):
    manifest = orch.predict_match("CZE", "KOR", "2026-06-11", LIVE_QUESTIONS,
                                  tournament_round="group")
    preds = manifest["predictions"]
    assert len(preds) == 10
    assert all(p["source"] != "fallback" for p in preds), \
        [p["question_text"] for p in preds if p["source"] == "fallback"]
    for p in preds:
        assert 0.001 <= p["final_probability"] <= 0.999


def test_no_player_prop_above_85(orch):
    """Live lesson: a 95% on 'Schick >= 1 SOT' cost -42 RBP. Cap holds."""
    manifest = orch.predict_match("CZE", "KOR", "2026-06-11", LIVE_QUESTIONS[8:],
                                  tournament_round="group")
    for p in manifest["predictions"]:
        assert p["final_probability"] <= 0.85


def test_h2_cards_not_overconfident(orch):
    """Live lesson: 83% on '2+ H2 cards' was the miscalibration; model lands ~0.6."""
    manifest = orch.predict_match("CZE", "KOR", "2026-06-11", [LIVE_QUESTIONS[2]],
                                  tournament_round="group")
    p = manifest["predictions"][0]["final_probability"]
    assert 0.40 < p < 0.75


def test_comparative_complement_sanity(orch):
    """P(A more) + P(B more) + P(tie) = 1."""
    eng = orch.engine
    p_home = eng.comparative_prob("CZE", "KOR", "CORNERS", "HOME")
    p_away = eng.comparative_prob("CZE", "KOR", "CORNERS", "AWAY")
    assert p_home + p_away < 1.0              # strict 'more' leaves room for ties
    assert p_home + p_away > 0.6


def test_compound_leq_marginals(orch):
    eng = orch.engine
    both_and_3 = eng.goal_market("CZE", "KOR", "BTTS_AND_TOTAL", "MATCH", 3.0,
                                 Condition.GTE)
    btts = eng.goal_market("CZE", "KOR", "BTTS", "MATCH", 1.0, Condition.BINARY_YES)
    over25 = eng.goal_market("CZE", "KOR", "GOALS", "MATCH", 2.5, Condition.GTE)
    assert both_and_3 <= min(btts, over25) + 1e-9


def test_penalty_in_sane_band(orch):
    p = orch.engine.penalty_prob()
    assert 0.20 <= p <= 0.40


# ---- formats observed live on the platform API, 2026-06-12 (USA vs PAR) ----

def test_finish_with_more_corners_comparative(clf):
    q = clf.parse("Will Paraguay finish with more corner kicks than United States?",
                  "USA", "PAR", "group")
    assert q.condition == Condition.MORE_THAN_OPP
    assert q.family == QuestionFamily.CORNER_MARKET
    assert q.target == "AWAY"


def test_tied_at_halftime(clf, orch):
    q = clf.parse("At halftime, will the match be tied?", "CZE", "KOR", "group")
    assert q.family == QuestionFamily.MATCH_RESULT
    assert q.target == "DRAW"
    assert q.window == TemporalWindow.H1
    # H1 draws are far likelier than full-match draws
    r_h1 = orch.engine.result_probs("CZE", "KOR", window=TemporalWindow.H1)
    r_ft = orch.engine.result_probs("CZE", "KOR")
    assert r_h1["draw"] > r_ft["draw"]


def test_cross_event_compound_flags_not_silently_prices(orch):
    """'USA scores first goal AND Paraguay scores in H2' must hit the flagged
    fallback — pricing one leg as if it were the whole question is the silent
    failure mode (would have submitted 53% on a ~22% event)."""
    manifest = orch.predict_match(
        "CZE", "KOR", "2026-06-17",
        ["Will Czechia score the first goal of the game and South Korea score "
         "in the second half?"], tournament_round="group")
    pred = manifest["predictions"][0]
    assert pred["source"] == "fallback"
    assert "review" in pred["notes"].lower()
