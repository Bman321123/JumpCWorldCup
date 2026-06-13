"""Offline tests for the scraper parsers and aggregator (synthetic payloads
shaped like the real APIs — no network in tests)."""
import pytest

from src.scrapers.aggregator import aggregate
from src.scrapers.common import BookOdds, american_to_decimal, teams_match
from src.scrapers.draftkings import parse_draftkings
from src.scrapers.fanduel import parse_fanduel_markets
from src.scrapers.pinnacle import parse_pinnacle


def test_american_to_decimal():
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal("+150") == pytest.approx(2.5)
    assert american_to_decimal("−110") == pytest.approx(1.909, abs=1e-3)  # unicode minus
    assert american_to_decimal(None) is None
    assert american_to_decimal(0) is None


def test_teams_match_order_insensitive():
    assert teams_match("Mexico", "South Africa", "Mexico", "South Africa")
    assert teams_match("South Africa", "Mexico", "Mexico", "South Africa")
    assert not teams_match("Mexico", "Canada", "Mexico", "South Africa")


PIN_MATCHUPS = [
    {"id": 1, "participants": [
        {"alignment": "home", "name": "Mexico"},
        {"alignment": "away", "name": "South Africa"}],
     "startTime": "2026-06-11T20:00:00Z"},
    {"id": 2, "type": "special",
     "special": {"description": "Both Teams To Score", "category": "Soccer"},
     "parent": {"id": 1},
     "participants": [{"id": 21, "name": "Yes"}, {"id": 22, "name": "No"}]},
    {"id": 3, "parent": {"id": 1}, "participants": [
        {"alignment": "home", "name": "Mexico (Corners)"},
        {"alignment": "away", "name": "South Africa (Corners)"}]},
]
PIN_MARKETS = [
    {"matchupId": 1, "type": "moneyline", "period": 0, "prices": [
        {"designation": "home", "price": -150},
        {"designation": "draw", "price": +280},
        {"designation": "away", "price": +450}]},
    {"matchupId": 1, "type": "total", "period": 0, "prices": [
        {"designation": "over", "price": -105, "points": 2.5},
        {"designation": "under", "price": -115, "points": 2.5}]},
    {"matchupId": 1, "type": "total", "period": 1, "prices": [
        {"designation": "over", "price": +110, "points": 1.0},
        {"designation": "under", "price": -135, "points": 1.0}]},
    {"matchupId": 2, "type": "moneyline", "period": 0, "prices": [
        {"participantId": 21, "price": +120},
        {"participantId": 22, "price": -145}]},
    {"matchupId": 3, "type": "total", "period": 0, "prices": [
        {"designation": "over", "price": -110, "points": 9.5},
        {"designation": "under", "price": -110, "points": 9.5}]},
    {"matchupId": 3, "type": "total", "period": 1, "prices": [
        {"designation": "over", "price": -105, "points": 4.5},
        {"designation": "under", "price": -115, "points": 4.5}]},
]


def test_parse_pinnacle():
    games = parse_pinnacle(PIN_MATCHUPS, PIN_MARKETS)
    assert len(games) == 1
    g = games[0]
    assert g.h2h and g.h2h["home"] == pytest.approx(1.667, abs=1e-3)
    assert 2.5 in g.totals
    assert 1.0 in g.h1_totals                    # first-half totals captured
    assert g.btts == (pytest.approx(2.2), pytest.approx(1.690, abs=1e-3))
    assert 9.5 in g.corner_totals                # sharp corner lines captured
    assert 4.5 in g.h1_corner_totals


def test_corner_market_lookup_integer_threshold():
    """'10 or more corners' must read the 9.5 book line, never a 10.0 push line."""
    from src.orchestrator import Orchestrator
    table = {9.5: 0.46, 10.0: 0.33}
    from src.types import Condition
    assert Orchestrator._totals_lookup(table, 10.0, Condition.GTE) == 0.46
    assert Orchestrator._totals_lookup(table, 9.5, Condition.GTE) == 0.46
    assert Orchestrator._totals_lookup(table, 10.0, Condition.LT) == pytest.approx(0.54)


DK_PAYLOAD = {"eventGroup": {
    "events": [{"eventId": 9, "teamName1": "Mexico", "teamName2": "South Africa",
                "startDate": "2026-06-11T20:00:00Z"}],
    "offerCategories": [{"offerSubcategoryDescriptors": [{"offerSubcategory": {
        "offers": [[
            {"eventId": 9, "label": "Moneyline", "outcomes": [
                {"label": "Mexico", "oddsAmerican": "-145"},
                {"label": "Draw", "oddsAmerican": "+270"},
                {"label": "South Africa", "oddsAmerican": "+430"}]},
            {"eventId": 9, "label": "Total Goals", "outcomes": [
                {"label": "Over", "oddsAmerican": "-110", "line": 2.5},
                {"label": "Under", "oddsAmerican": "-110", "line": 2.5}]},
            {"eventId": 9, "label": "Both Teams to Score", "outcomes": [
                {"label": "Yes", "oddsAmerican": "+115"},
                {"label": "No", "oddsAmerican": "-140"}]},
        ]]}}]}],
}}


def test_parse_draftkings():
    games = parse_draftkings(DK_PAYLOAD)
    assert len(games) == 1
    g = games[0]
    assert g.h2h and abs(g.h2h["draw"] - 3.7) < 0.01
    assert g.totals[2.5][0] == pytest.approx(1.909, abs=1e-3)
    assert g.btts is not None


def _dec(d):
    return {"winRunnerOdds": {"trueOdds": {"decimalOdds": {"decimalOdds": d}}}}


# merged {marketId: market} dict, as FanDuel's event-page tabs return
FD_MARKETS = {
    "m1": {"marketType": "WIN-DRAW-WIN", "runners": [
        {"runnerName": "Mexico", **_dec(1.7)},
        {"runnerName": "Draw", **_dec(3.8)},
        {"runnerName": "South Africa", **_dec(5.4)}]},
    "m2": {"marketType": "OVER_UNDER_25", "runners": [
        {"runnerName": "Over 2.5 Goals", "handicap": 0, **_dec(1.92)},
        {"runnerName": "Under 2.5 Goals", "handicap": 0, **_dec(1.92)}]},
    "m3": {"marketType": "BOTH_TEAMS_TO_SCORE", "runners": [
        {"runnerName": "Yes", **_dec(2.18)},
        {"runnerName": "No", **_dec(1.67)}]},
    "m4": {"marketType": "PLAYER_TO_HAVE_1_OR_MORE_SHOTS", "runners": [
        {"runnerName": "Hirving Lozano", **_dec(1.3)}]},
}


def test_parse_fanduel_markets():
    g = parse_fanduel_markets(FD_MARKETS, "Mexico", "South Africa")
    assert g is not None
    assert g.h2h and {"home", "draw", "away"} <= set(g.h2h)
    assert 2.5 in g.totals
    assert g.btts is not None
    assert "Hirving Lozano" in g.player_shots          # player markets captured


def test_aggregate_pinnacle_is_sharp_anchor():
    pin = parse_pinnacle(PIN_MATCHUPS, PIN_MARKETS)[0]
    dk = parse_draftkings(DK_PAYLOAD)[0]
    out = aggregate([dk, pin], "Mexico", "South Africa")
    assert out is not None
    # h2h must come from Pinnacle alone, not a median
    from src.shin_devigger import decimal_to_implied, shin_devig
    pin_h2h = shin_devig(decimal_to_implied(
        [pin.h2h["home"], pin.h2h["draw"], pin.h2h["away"]]))
    assert out["h2h"]["home"] == pytest.approx(float(pin_h2h[0]), abs=1e-9)
    assert abs(sum(out["h2h"].values()) - 1.0) < 1e-9
    assert 1.0 in out["h1_totals"]               # Pinnacle period-1 totals flow through


def test_aggregate_median_without_pinnacle():
    dk = parse_draftkings(DK_PAYLOAD)[0]
    fd = parse_fanduel_markets(FD_MARKETS, "Mexico", "South Africa")
    out = aggregate([dk, fd], "Mexico", "South Africa")
    assert out is not None
    assert 0.30 < out["h2h"]["draw"] + out["h2h"]["away"] < 0.50
    assert out["btts"] is not None


def test_aggregate_empty():
    assert aggregate([], "Mexico", "South Africa") is None
