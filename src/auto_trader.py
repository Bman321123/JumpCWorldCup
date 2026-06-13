"""Autonomous submission engine — DISARMED BY DEFAULT.

The bot scores its own confidence per question and decides which it is willing
to submit without a human. It NEVER submits unless config/auto_trade.json has
"armed": true AND the caller passes --go. Arming is a human act; this module
will not flip it.

Confidence model (self-defined, 0..1) — a question is auto-eligible only when
the system has a defensible reason to trust the number:

  + market-anchored (a sharp devigged line backs it)      strongest signal
  + family passed its backtest ship-gate                   validated structure
  + calibrated inside the guard band                       evidence-backed
  - fallback / unmodeled                                   hard veto
  - player prop without real involvement shares            hard veto (flat prior)
  - |our number - market| large                            disagreement → human
  - round weight above the group stage                     knockouts → human

Everything is logged to auto_trade_log regardless of armed state, so a dry run
produces a full audit of what WOULD have been submitted.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

VALIDATED_FAMILIES = {"GOAL_MARKET", "MATCH_RESULT", "CORNER_MARKET"}
PLAYER_FAMILIES = {"PLAYER_MARKET"}

DEFAULT_CRITERIA = {
    "armed": False,                       # MASTER SWITCH — human sets this
    "min_confidence": 0.70,               # auto-submit only at/above this
    "max_round_weight": 1.0,              # group stage only until proven
    "max_deviation_from_market": 0.15,    # bigger gap -> hand to human
    "require_market_or_validated": True,  # no edge basis -> skip
    "exclude_families": ["PLAYER_MARKET"],   # flat priors until shares land
    "max_submissions_per_match": 10,
}


@dataclass
class Decision:
    question_id: str
    question: str
    family: str
    submit_value: int                     # platform integer 1-99
    confidence: float
    auto_eligible: bool
    reasons: List[str] = field(default_factory=list)


def load_criteria(path: Optional[str] = None) -> dict:
    crit = dict(DEFAULT_CRITERIA)
    if path and Path(path).exists():
        with open(path) as f:
            crit.update(json.load(f))
    return crit


def score_confidence(pred: dict, round_weight: float, crit: dict) -> tuple:
    """Return (confidence in 0..1, eligible bool, reasons)."""
    reasons: List[str] = []
    fam = pred["question_family"]

    if pred["source"] == "fallback":
        return 0.0, False, ["fallback/unmodeled — veto"]
    if fam in PLAYER_FAMILIES and fam in crit["exclude_families"]:
        return 0.0, False, ["player prop on flat prior — veto"]
    if fam in crit["exclude_families"]:
        return 0.0, False, [f"{fam} excluded by config"]

    conf = 0.0
    has_market = pred.get("market_probability") is not None
    if has_market:
        conf += 0.45
        reasons.append("market-anchored +0.45")
    if fam in VALIDATED_FAMILIES:
        conf += 0.30
        reasons.append("validated family +0.30")
    if pred.get("model_calibrated") not in (None, pred.get("model_probability")):
        conf += 0.10
        reasons.append("calibrated +0.10")
    # base credit so a market-anchored validated family clears the bar
    conf += 0.15
    reasons.append("base +0.15")

    if crit["require_market_or_validated"] and not (has_market
                                                    or fam in VALIDATED_FAMILIES):
        return min(conf, 0.4), False, reasons + ["no market & unvalidated — veto"]

    if has_market:
        dev = abs(pred["final_probability"] - pred["market_probability"])
        if dev > crit["max_deviation_from_market"]:
            conf -= 0.25
            reasons.append(f"deviation {dev:.2f} from market -0.25")

    if round_weight > crit["max_round_weight"]:
        return conf, False, reasons + [
            f"round weight {round_weight} > {crit['max_round_weight']} — human only"]

    conf = max(0.0, min(conf, 1.0))
    eligible = conf >= crit["min_confidence"]
    reasons.append(f"confidence {conf:.2f} "
                   f"{'>=' if eligible else '<'} {crit['min_confidence']}")
    return conf, eligible, reasons


def plan_submissions(manifest: dict, submit_values: dict, crit: dict) -> List[Decision]:
    """submit_values: question_id -> int (post-policy). Returns one Decision per
    question; auto_eligible flags which the bot will submit when armed."""
    rw = manifest.get("round_weight", 1.0)
    out = []
    eligible_count = 0
    for pred in manifest["predictions"]:
        conf, eligible, reasons = score_confidence(pred, rw, crit)
        if eligible and eligible_count >= crit["max_submissions_per_match"]:
            eligible, reasons = False, reasons + ["per-match cap reached"]
        if eligible:
            eligible_count += 1
        out.append(Decision(
            question_id=pred["question_id"], question=pred["question_text"],
            family=pred["question_family"],
            submit_value=submit_values.get(pred["question_id"], 50),
            confidence=conf, auto_eligible=eligible, reasons=reasons))
    return out
