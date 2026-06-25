"""Round inference from UTC kickoff times — the bug that skipped knockouts AND
the imminent one that would skip the final group games (kickoff spills past midnight
UTC). Cases use real opening_times observed on the platform 2026-06-24.

Also pins the accented-player-name classification fix (the Sangaré 0.999 landmine)."""
from src.question_classifier import QuestionClassifier
from src.schedule import infer_round, local_match_date
from src.types import QuestionFamily


def test_accented_player_name_routes_to_player_market():
    """'Will Ibrahim Sangaré have at least 1 shot on target?' must be a PLAYER prop,
    not a TEAM shots-on-target market (which prices ~0.99 and cost a 0.94 Brier)."""
    qc = QuestionClassifier("config/groups.json")
    # CIV vs CUW: Sangaré is neither team, so side is None and the player path applies
    p = qc.parse("Will Ibrahim Sangaré have at least 1 shot on target?", "CIV", "CUW")
    assert p.family == QuestionFamily.PLAYER_MARKET, p.family
    # a genuine TEAM shots question (the team IS a side) stays a team market
    t = qc.parse("Will Ivory Coast have 3 or more shots on target?", "CIV", "CUW")
    assert t.family == QuestionFamily.SHOTS_MARKET, t.family


def test_normal_group_game():
    assert infer_round("2026-06-24T22:00:00.000Z") == "group"


def test_final_group_games_spill_past_midnight_utc():
    # JOR vs ARG / ALG vs AUT kick off 02:00Z on Jun 28 but are GROUP games.
    # The old date-string cutoff "2026-06-28" <= "2026-06-27" => skipped them.
    assert local_match_date("2026-06-28T02:00:00.000Z") == "2026-06-27"
    assert infer_round("2026-06-28T02:00:00.000Z") == "group"


def test_round_of_32_is_priced_not_skipped():
    assert infer_round("2026-06-28T19:00:00.000Z") == "round_of_32"
    assert infer_round("2026-07-03T23:00:00.000Z") == "round_of_32"
    # spillover past midnight stays in R32, not bumped into R16
    assert infer_round("2026-07-04T02:00:00.000Z") == "round_of_32"


def test_later_rounds_label_correctly():
    assert infer_round("2026-07-04T19:00:00.000Z") == "round_of_16"
    assert infer_round("2026-07-10T20:00:00.000Z") == "quarterfinal"
    assert infer_round("2026-07-14T23:00:00.000Z") == "semifinal"
    assert infer_round("2026-07-15T23:00:00.000Z") == "semifinal"
    assert infer_round("2026-07-18T22:00:00.000Z") == "third_place"
    assert infer_round("2026-07-19T22:00:00.000Z") == "final"


def test_rest_day_spillover_absorbed_not_skipped():
    # a QF game spilling to 02:00Z Jul 12 (a rest day) must still be a knockout
    assert infer_round("2026-07-12T02:00:00.000Z") == "quarterfinal"


def test_unparseable_or_pretournament_returns_none():
    assert infer_round(None) is None
    assert infer_round("not-a-date") is None
    assert infer_round("2026-06-01T18:00:00.000Z") is None  # before the tournament
