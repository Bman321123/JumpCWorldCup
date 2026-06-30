"""Skill-weighted shrink-only calibration — the safety properties that keep it from
overfitting: it only ever DAMPENS confidence (never amplifies), it's a no-op without
a table, well-evidenced no-edge families shrink, and a perfectly-tracking family is
left alone."""
from src.family_calibration import K_MIN, apply, family_key, fit, loo_brier


def test_apply_is_noop_without_table():
    assert apply(0.83, "Will Germany win?", "MATCH_RESULT", None) == 0.83
    assert apply(0.83, "Will Germany win?", "MATCH_RESULT", {}) == 0.83


def test_apply_only_shrinks_never_amplifies():
    table = {"team_sot": {"k": 0.35}}
    p = apply(0.90, "Will Portugal have 5 or more shots on target?", "SHOTS_MARKET", table)
    assert 0.50 < p < 0.90                       # pulled toward 50, never past it
    # a family not in the table is untouched (k=1)
    assert apply(0.90, "Will Germany win?", "MATCH_RESULT", table) == 0.90


def test_no_edge_family_shrinks_skilful_family_does_not():
    # family A: predictions uncorrelated with outcome (no edge) -> k < 1
    no_edge = [(0.8, 0.0, "A"), (0.2, 1.0, "A"), (0.75, 0.0, "A"), (0.3, 1.0, "A")] * 6
    # family B: predictions track outcomes perfectly (real edge) -> k clamps at 1
    skilled = [(0.9, 1.0, "B"), (0.1, 0.0, "B"), (0.85, 1.0, "B"), (0.15, 0.0, "B")] * 6
    f = fit(no_edge + skilled, min_n=8)
    assert f["A"]["k"] < 1.0
    assert f["A"]["k"] >= K_MIN                   # never below the floor
    assert f["B"]["k"] == 1.0                     # earned confidence is preserved


def test_loo_beats_raw_on_a_clearly_overconfident_family():
    rows = [(0.85, 0.0, "X"), (0.80, 0.0, "X"), (0.9, 0.0, "X"), (0.2, 1.0, "X")] * 8
    raw, cal = loo_brier(rows, min_n=8)
    assert cal < raw                             # shrinking overconfidence helps OOS


def test_family_key_uses_parser_family_for_player_vs_team():
    assert family_key("Will X have 5 or more shots on target?", "SHOTS_MARKET") == "team_sot"
    assert family_key("Will Kai Havertz (Germany) have 2 or more shots on target?",
                      "PLAYER_MARKET") == "player_sot"
    assert family_key("Will A have more corners than B?", "CORNER_MARKET") == "cmp_corners"
