from pathlib import Path

import pytest

from src.question_classifier import QuestionClassifier
from src.types import Condition, QuestionFamily, QuestionParseError, ResultScope, TemporalWindow

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "groups.json")


@pytest.fixture
def clf():
    return QuestionClassifier(CONFIG, {"group": 1.0, "final": 16.0})


CASES = [
    ("Will Argentina be caught offside 2 or more times?",
     QuestionFamily.OFFSIDE_MARKET, "HOME", 2.0, Condition.GTE, TemporalWindow.FULL),
    ("Will there be over 2.5 goals in the match?",
     QuestionFamily.GOAL_MARKET, "MATCH", 2.5, Condition.GTE, TemporalWindow.FULL),
    ("Will France receive 3 or more yellow cards?",
     QuestionFamily.CARD_MARKET, "AWAY", 3.0, Condition.GTE, TemporalWindow.FULL),
    ("Will there be fewer than 9 corners in the first half?",
     QuestionFamily.CORNER_MARKET, "MATCH", 9.0, Condition.LT, TemporalWindow.H1),
    ("Will there be under 4.5 yellow cards?",
     QuestionFamily.CARD_MARKET, "MATCH", 4.5, Condition.LT, TemporalWindow.FULL),
    ("Will there be at least 10 corners?",
     QuestionFamily.CORNER_MARKET, "MATCH", 10.0, Condition.GTE, TemporalWindow.FULL),
]


@pytest.mark.parametrize("text,family,target,threshold,condition,window", CASES)
def test_parse_cases(clf, text, family, target, threshold, condition, window):
    q = clf.parse(text, "ARG", "FRA", "group")
    assert q.family == family
    assert q.target == target
    assert q.threshold == threshold
    assert q.condition == condition
    assert q.window == window


def test_btts(clf):
    q = clf.parse("Will both teams score?", "ARG", "FRA", "group")
    assert q.family == QuestionFamily.GOAL_MARKET
    assert q.metric == "BTTS"
    assert q.condition == Condition.BINARY_YES


def test_over_integer_means_strictly_more(clf):
    q = clf.parse("Will there be over 2 goals?", "ARG", "FRA", "group")
    assert q.threshold == 2.5          # "over 2" == N >= 3, encoded via ceil(2.5)


def test_result_sides(clf):
    q = clf.parse("Will Argentina win?", "ARG", "FRA", "group")
    assert q.family == QuestionFamily.MATCH_RESULT and q.target == "HOME"
    assert q.scope == ResultScope.WIN_90
    q = clf.parse("Will France win?", "ARG", "FRA", "group")
    assert q.target == "AWAY"
    q = clf.parse("Will the match end in a draw?", "ARG", "FRA", "group")
    assert q.target == "DRAW"


def test_knockout_win_defaults_to_advance(clf):      # B8 — scope disambiguation
    q = clf.parse("Will Argentina win?", "ARG", "FRA", "final")
    assert q.scope == ResultScope.ADVANCE
    q = clf.parse("Will Argentina win in 90 minutes?", "ARG", "FRA", "final")
    assert q.scope == ResultScope.WIN_90
    q = clf.parse("Will France advance?", "ARG", "FRA", "final")
    assert q.scope == ResultScope.ADVANCE and q.target == "AWAY"


def test_alias_resolution(clf):
    q = clf.parse("Will South Korea win?", "KOR", "CZE", "group")
    assert q.target == "HOME"
    q = clf.parse("Will Turkey receive 2 or more yellow cards?", "USA", "TUR", "group")
    assert q.target == "AWAY"


def test_round_weight(clf):
    q = clf.parse("Will both teams score?", "ARG", "FRA", "final")
    assert q.round_weight == 16.0


def test_unparseable_raises(clf):
    with pytest.raises(QuestionParseError):
        clf.parse("What is the airspeed velocity of an unladen swallow?",
                  "ARG", "FRA", "group")
