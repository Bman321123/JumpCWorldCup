"""Regression tests for the autonomous daemon's safety helpers (post-review)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_submit", ROOT / "tools" / "live_submit.py")
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)


def test_resolve_never_fabricates_codes():
    """The Austria->AUS(Australia) collision bug: unresolved names must return
    None (skip the match), never a fabricated 3-letter code."""
    idx = ls._codes()
    assert ls._resolve("Germany", idx) == "GER"
    assert ls._resolve("New Zealand", idx) == "NZL"
    assert ls._resolve("Zorgon FC", idx) is None        # unknown -> None, not 'ZOR'
    assert ls._resolve("Austrglia", idx) is None         # typo -> None, not 'AUS'


def test_hours_to_robust():
    assert ls._hours_to("2026-06-25T19:00:00.000Z") != 999.0   # parses Z
    assert ls._hours_to("2026-06-25T19:00:00") != 999.0        # naive -> UTC
    assert ls._hours_to(None) == 999.0                          # non-string -> skip
    assert ls._hours_to(1782345600) == 999.0                    # int epoch -> skip
    assert ls._hours_to("garbage") == 999.0


def test_infer_round_skips_unmapped_knockout():
    assert ls._infer_round("2026-06-24") == "group"     # group stage
    assert ls._infer_round("2026-06-27") == "group"     # last group day
    assert ls._infer_round("2026-07-05") is None        # knockout -> skip, not misprice
    assert ls._infer_round("") is None


def test_journal_roundtrip(tmp_path, monkeypatch):
    j = {"abc": {"prediction_id": "p1", "probability": 62}}
    monkeypatch.setattr(ls, "JOURNAL", tmp_path / "j.json")
    ls._save_journal(j)
    assert ls._load_journal() == j
    # missing file -> empty, never crashes
    monkeypatch.setattr(ls, "JOURNAL", tmp_path / "nope.json")
    assert ls._load_journal() == {}
