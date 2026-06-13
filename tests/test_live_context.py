"""Current-events logic — absence derivation is pure and testable offline."""
from src.live_context import derive_absences

SHARES = {
    "Star Striker": {"team": "FRA", "share": 0.30, "expected_minutes": 90},
    "Key Mid": {"team": "FRA", "share": 0.15, "expected_minutes": 80},
    "Fringe Sub": {"team": "FRA", "share": 0.04, "expected_minutes": 40},
    "Other Team Guy": {"team": "GER", "share": 0.40, "expected_minutes": 90},
}


def test_key_player_missing_is_an_absence():
    xi = ["Key Mid", "Some Defender", "Some Keeper"]      # Star Striker NOT in XI
    out = derive_absences("FRA", xi, SHARES)
    assert "Star Striker" in out
    assert "Key Mid" not in out                           # he IS in the XI


def test_fringe_player_not_flagged():
    xi = ["Star Striker", "Key Mid"]                       # Fringe Sub out, but low share
    out = derive_absences("FRA", xi, SHARES)
    assert "Fringe Sub" not in out                         # below the key threshold


def test_other_team_players_ignored():
    xi = ["Star Striker", "Key Mid"]
    out = derive_absences("FRA", xi, SHARES)
    assert "Other Team Guy" not in out                     # wrong team


def test_no_lineup_no_absences():
    assert derive_absences("FRA", [], SHARES) == []        # XI not published yet
