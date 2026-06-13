"""Pinned regression for v1.0 bug B4 (inverted must-win) + player layer shrinkage."""
import pytest

from src.context_resolver import card_intensity, qualification_states
from src.player_layer import PlayerShares, anytime_scorer_prob, shrunk_rate
from src.types import MotivationState


def even_probs(home, away):
    return (0.40, 0.25, 0.35)


def test_must_win_detected_correctly():              # B4
    # Last matchday. C has 1 pt: alive only with a win -> MUST_WIN, not ELIMINATED.
    standings = {"A": {"pts": 6, "gd": 4}, "B": {"pts": 4, "gd": 1},
                 "C": {"pts": 1, "gd": -2}, "D": {"pts": 1, "gd": -3}}
    remaining = [("A", "B"), ("C", "D")]
    states = qualification_states(standings, remaining, ("C", "D"), even_probs,
                                  n_sims=600)
    assert states["C"] == MotivationState.MUST_WIN
    assert states["D"] == MotivationState.MUST_WIN


def test_safe_team_detected():
    standings = {"A": {"pts": 6, "gd": 5}, "B": {"pts": 3, "gd": 0},
                 "C": {"pts": 3, "gd": -1}, "D": {"pts": 0, "gd": -4}}
    remaining = [("A", "D"), ("B", "C")]
    states = qualification_states(standings, remaining, ("A", "D"), even_probs,
                                  n_sims=600)
    assert states["A"] == MotivationState.SAFE       # 6 pts: through regardless


def test_eliminated_only_when_win_does_nothing():
    # Force the third-place table to zero so 3 points never advances.
    standings = {"A": {"pts": 6, "gd": 5}, "B": {"pts": 4, "gd": 2},
                 "C": {"pts": 4, "gd": 1}, "D": {"pts": 0, "gd": -8}}
    remaining = [("A", "B"), ("C", "D")]
    states = qualification_states(
        standings, remaining, ("C", "D"), even_probs, n_sims=600,
        third_adv_by_points={k: 0.0 for k in range(10)})
    assert states["D"] == MotivationState.ELIMINATED


def test_card_intensity_direction():                 # B4 — dead rubber DOWN, must-win UP
    up = card_intensity(MotivationState.MUST_WIN, MotivationState.NORMAL, "group")
    down = card_intensity(MotivationState.SAFE, MotivationState.ELIMINATED, "group")
    assert up > 1.0 > down


def test_shrunk_rate_kills_streak_faith():
    # "Scored in 8 of 10" -> 0.50 with FW prior, never 0.8 (PRD §4.9)
    assert shrunk_rate(8, 10, prior=0.35, prior_n=20) == pytest.approx(0.50)


def test_anytime_scorer_sane():
    p = anytime_scorer_prob(lam_team=1.8, involvement_share=0.30)
    assert 0.35 < p < 0.50


def test_player_sot_opponent_adjustment(tmp_path):
    """Same player, harder opponent -> lower SOT prob (the matchup the user
    described: leaky defense vs elite defense)."""
    import json
    from src.player_layer import PlayerShares, player_prop_prob
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"players": {
        "Star FW": {"team": "QAT", "position": "FW", "share": 0.30,
                    "sot90": 1.3, "apps": 10, "expected_minutes": 90}}}))
    ps = PlayerShares(str(path))
    leaky, _ = player_prop_prob("Star FW", "PLAYER_SOT", 1.0, 0.8, 1.6,
                                "QAT", "SUI", ps, opp_sot_against=5.0)
    elite, _ = player_prop_prob("Star FW", "PLAYER_SOT", 1.0, 0.8, 1.6,
                                "QAT", "BRA", ps, opp_sot_against=2.5)
    assert leaky > elite                      # leaky defense -> more SOT
    assert elite <= 0.85 and leaky <= 0.85


def test_player_sot_uses_fd_shots_market(tmp_path):
    """The FanDuel player-shots market moves the estimate (real signal even with
    no SOT counterpart) — shots bridged to SOT."""
    import json
    from src.player_layer import PlayerShares, player_prop_prob
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"players": {
        "Star FW": {"team": "QAT", "position": "FW", "share": 0.30,
                    "sot90": 1.0, "apps": 8, "expected_minutes": 90}}}))
    ps = PlayerShares(str(path))
    base, _ = player_prop_prob("Star FW", "PLAYER_SOT", 1.0, 0.8, 1.6,
                               "QAT", "SUI", ps, opp_sot_against=4.3)
    moved, note = player_prop_prob("Star FW", "PLAYER_SOT", 1.0, 0.8, 1.6,
                                   "QAT", "SUI", ps, opp_sot_against=4.3,
                                   fd_shots={"1+_FULL": 0.55})
    assert abs(moved - base) > 0.01           # the market signal moves the estimate
    assert "FD_shots" in note
    # a very strong shots market (implies high shot volume) lifts SOT above model
    hot, _ = player_prop_prob("Star FW", "PLAYER_SOT", 1.0, 0.8, 1.6,
                              "QAT", "SUI", ps, opp_sot_against=4.3,
                              fd_shots={"1+_FULL": 0.97})
    assert hot > base


def test_sparse_player_sot_shrinks_off_the_floor(tmp_path):
    """A star with few logged matches and 0 SOT must NOT collapse to the 3%
    floor — sparse rates shrink toward the position prior (the Schick lesson)."""
    import json
    from src.player_layer import PlayerShares, player_prop_prob
    path = tmp_path / "shares.json"
    path.write_text(json.dumps({"players": {
        "Sparse Star": {"team": "QAT", "position": "FW", "share": 0.0,
                        "sot90": 0.0, "apps": 3, "expected_minutes": 90}}}))
    ps = PlayerShares(str(path))
    p, note = player_prop_prob("Sparse Star", "PLAYER_SOT", 1.0,
                               1.4, 1.2, "QAT", "SUI", ps)
    assert p > 0.30          # shrunk toward FW prior, not stuck at 0.03
    assert p <= 0.85         # cap still holds


def test_availability_multiplier_floor(tmp_path):
    import json
    path = tmp_path / "shares.json"
    path.write_text(json.dumps({"players": {
        "Star A": {"team": "FRA", "share": 0.5},
        "Star B": {"team": "FRA", "share": 0.5},
        "Star C": {"team": "FRA", "share": 0.5}}}))
    ps = PlayerShares(str(path))
    m = ps.availability_multiplier("FRA", ["Star A", "Star B", "Star C"])
    assert m == pytest.approx(0.70)                  # floored at -30%
    assert ps.availability_multiplier("FRA", ["Unknown"]) == 1.0
    assert ps.availability_multiplier("ARG", ["Star A"]) == 1.0   # wrong team ignored
