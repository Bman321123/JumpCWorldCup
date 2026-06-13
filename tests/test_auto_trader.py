"""Auto-trader confidence + gating — all offline, no platform calls."""
import pytest

from src.auto_trader import (DEFAULT_CRITERIA, load_criteria, plan_submissions,
                             score_confidence)


def _pred(qid="Q1", family="GOAL_MARKET", source="blend_w0.75",
          final=0.55, market=0.54, model=0.58, cal=0.56):
    return {"question_id": qid, "question_text": f"q {qid}",
            "question_family": family, "source": source,
            "final_probability": final, "market_probability": market,
            "model_probability": model, "model_calibrated": cal}


CRIT = dict(DEFAULT_CRITERIA)


def test_market_anchored_validated_is_eligible():
    conf, elig, _ = score_confidence(_pred(), 1.0, CRIT)
    assert elig and conf >= 0.70


def test_fallback_is_vetoed():
    conf, elig, reasons = score_confidence(
        _pred(source="fallback"), 1.0, CRIT)
    assert not elig and conf == 0.0
    assert any("fallback" in r for r in reasons)


def test_player_prop_vetoed_until_shares():
    conf, elig, reasons = score_confidence(
        _pred(family="PLAYER_MARKET", market=None), 1.0, CRIT)
    assert not elig
    assert any("player prop" in r for r in reasons)


def test_knockout_weight_blocks_auto():
    conf, elig, reasons = score_confidence(_pred(), 4.0, CRIT)   # R16
    assert not elig
    assert any("human only" in r for r in reasons)


def test_large_market_deviation_drops_confidence():
    base, _, _ = score_confidence(_pred(final=0.54), 1.0, CRIT)
    dev, elig, reasons = score_confidence(
        _pred(final=0.85, market=0.54), 1.0, CRIT)   # 0.31 gap
    assert dev < base
    assert any("deviation" in r for r in reasons)


def test_unvalidated_no_market_vetoed():
    conf, elig, reasons = score_confidence(
        _pred(family="OFFSIDE_MARKET", market=None), 1.0, CRIT)
    assert not elig
    assert any("no market & unvalidated" in r for r in reasons)


def test_offside_with_market_can_pass():
    conf, elig, _ = score_confidence(
        _pred(family="OFFSIDE_MARKET", market=0.55, final=0.55), 1.0, CRIT)
    assert elig                          # market anchor carries it


def test_plan_respects_per_match_cap():
    crit = dict(DEFAULT_CRITERIA, max_submissions_per_match=2)
    manifest = {"round_weight": 1.0, "predictions": [
        _pred(qid=f"Q{i}") for i in range(5)]}
    submit_values = {f"Q{i}": 55 for i in range(5)}
    decisions = plan_submissions(manifest, submit_values, crit)
    assert sum(d.auto_eligible for d in decisions) == 2
    assert any("per-match cap" in r for d in decisions for r in d.reasons)


def test_default_config_is_disarmed():
    crit = load_criteria(None)
    assert crit["armed"] is False        # the safety default


def test_shipped_config_is_disarmed():
    from pathlib import Path
    crit = load_criteria(str(Path(__file__).resolve().parents[1]
                             / "config" / "auto_trade.json"))
    assert crit["armed"] is False        # never commit an armed config
