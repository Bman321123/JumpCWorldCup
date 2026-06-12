"""Crowd capture: block parsing, storage dedupe, fuzzy join — all offline."""
import pytest

from src.crowd_capture import (fuzzy_lookup, latest_crowd, normalize_question,
                               parse_block, parse_blocks, store_capture)


def test_labeled_crowd_and_own():
    row = parse_block("Q3\nWill there be 2 or more total cards shown in the "
                      "second half?\nCrowd: 58%\nYour forecast: 83%")
    assert row["crowd_pct"] == 58.0
    assert row["own_pct"] == 83.0
    assert row["ambiguous"] == 0
    assert row["question_text"].startswith("Will there be 2 or more")


def test_single_unlabeled_percent():
    row = parse_block("Will a penalty kick be awarded in the match?\n33% YES")
    assert row["crowd_pct"] == 33.0
    assert row["ambiguous"] == 0


def test_multiple_unlabeled_is_flagged_not_dropped():
    row = parse_block("Will South Korea win the match?\n51%\n38%")
    assert row["crowd_pct"] == 51.0
    assert row["ambiguous"] == 1                 # kept, but flagged for review


def test_forecasters_hint_counts_as_crowd():
    row = parse_block("Will both teams score AND the match have 3 or more "
                      "total goals?\n38% of forecasters say YES\nYou: 67%")
    assert row["crowd_pct"] == 38.0
    assert row["own_pct"] == 67.0


def test_no_question_returns_none():
    assert parse_block("Leaderboard\nRank 42\nTop 10%") is None
    assert parse_block("Some banner with 50% off?") is None  # too short a question


def test_parse_blocks_dedupes():
    b = "Will Mexico win the match?\nCrowd: 60%"
    assert len(parse_blocks([b, b, b])) == 1


def test_store_and_latest_and_fuzzy(tmp_path):
    db = str(tmp_path / "t.db")
    rows = parse_blocks([
        "Will there be 10 or more corners?\nCrowd: 46%",
        "Will Mexico win the match?\nCrowd: 61%",
    ])
    assert store_capture(db, rows, "http://x") == 2
    # same day re-capture replaces, not duplicates
    rows2 = parse_blocks(["Will Mexico win the match?\nCrowd: 64%"])
    store_capture(db, rows2, "http://x")
    crowd = latest_crowd(db)
    assert len(crowd) == 2
    assert crowd[normalize_question("Will Mexico win the match?")]["crowd_pct"] == 64.0

    # fuzzy join survives platform prefixes/suffixes our parser didn't strip
    hit = fuzzy_lookup("Will Mexico win the match", crowd)
    assert hit and hit["crowd_pct"] == 64.0
    hit2 = fuzzy_lookup("Will there be ten or more corners?", crowd)  # close text
    assert hit2 is None or hit2["crowd_pct"] == 46.0
    assert fuzzy_lookup("Will Patrik Schick score a hat-trick?", crowd) is None
