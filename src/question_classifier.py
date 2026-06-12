"""Deterministic question-template parser (PRD v2.2 §6.1).

Fails loudly with QuestionParseError — never a silent wrong mapping. Team names
come from config/groups.json; the draw is data, never hardcoded.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Dict, Optional, Tuple

from .types import (Condition, ParsedQuestion, QuestionFamily,
                    QuestionParseError, ResultScope, TemporalWindow)

KNOCKOUT_ROUNDS = {"round_of_32", "round_of_16", "quarterfinal", "semifinal",
                   "third_place", "final"}

_THRESHOLD_PATTERNS: Tuple[Tuple[str, Condition], ...] = (
    (r"over\s+(\d+(?:\.\d+)?)", Condition.GTE),
    (r"under\s+(\d+(?:\.\d+)?)", Condition.LT),
    (r"more than\s+(\d+(?:\.\d+)?)", Condition.GTE),
    (r"(\d+(?:\.\d+)?)\s*(?:or more|or greater|or above|\+)", Condition.GTE),
    (r"at least\s+(\d+(?:\.\d+)?)", Condition.GTE),
    (r"fewer than\s+(\d+(?:\.\d+)?)", Condition.LT),
    (r"less than\s+(\d+(?:\.\d+)?)", Condition.LT),
    (r"exactly\s+(\d+(?:\.\d+)?)", Condition.EQ),
)
# NOTE on semantics: "more than 2" (integer) means N >= 3. count_math.count_prob
# uses ceil(), so we bump integer "more than"/"over" thresholds by 0.5 to encode
# strict inequality; half-line thresholds need no bump.


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


class QuestionClassifier:
    def __init__(self, groups_config_path: str, round_weights: Optional[Dict[str, float]] = None):
        with open(groups_config_path) as f:
            cfg = json.load(f)
        self.alias_to_code: Dict[str, str] = {}
        for code, info in cfg["teams"].items():
            self.alias_to_code[_norm(info["name"])] = code
            self.alias_to_code[_norm(code)] = code
            for a in info.get("aliases", []):
                self.alias_to_code[_norm(a)] = code
        self.round_weights = round_weights or {}

    def _find_team(self, text: str, home: str, away: str) -> Optional[str]:
        """Return 'HOME'/'AWAY' if the question names one of the two teams."""
        for alias, code in sorted(self.alias_to_code.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                if code == home:
                    return "HOME"
                if code == away:
                    return "AWAY"
        return None

    def parse(self, raw_text: str, home_team: str, away_team: str,
              tournament_round: str = "group",
              question_id: Optional[str] = None) -> ParsedQuestion:
        text = _norm(raw_text)
        qid = question_id or f"q_{uuid.uuid4().hex[:8]}"
        weight = float(self.round_weights.get(tournament_round, 1.0))
        side = self._find_team(text, home_team, away_team)

        window = TemporalWindow.FULL
        if re.search(r"\b(first half|1st half|opening half)\b", text):
            window = TemporalWindow.H1
        elif re.search(r"\b(second half|2nd half)\b", text):
            window = TemporalWindow.H2

        threshold, condition = self._parse_threshold(text)

        family, metric = self._family_metric(text)

        if family == QuestionFamily.MATCH_RESULT:
            return self._parse_result(raw_text, text, qid, home_team, away_team,
                                      side, tournament_round, weight, window)

        if metric == "BTTS":
            return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET,
                                  home_team, away_team, "MATCH", "BTTS", 1.0,
                                  Condition.BINARY_YES, window, ResultScope.NONE, weight)

        if metric == "CLEAN_SHEET":
            target = side or "MATCH"
            return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET,
                                  home_team, away_team, target, "CLEAN_SHEET", 1.0,
                                  Condition.BINARY_YES, window, ResultScope.NONE, weight)

        if threshold is None or condition is None:
            raise QuestionParseError(
                f"Could not parse threshold/condition from: {raw_text!r}")

        target = side or "MATCH"
        return ParsedQuestion(raw_text, qid, family, home_team, away_team, target,
                              metric, threshold, condition, window,
                              ResultScope.NONE, weight)

    # ----- helpers -----

    def _parse_threshold(self, text: str) -> Tuple[Optional[float], Optional[Condition]]:
        for pattern, cond in _THRESHOLD_PATTERNS:
            m = re.search(pattern, text)
            if m:
                thr = float(m.group(1))
                # strict ">" phrasings on integer lines mean N >= thr+1
                if cond == Condition.GTE and pattern.startswith((r"over", r"more than")) \
                        and thr == int(thr):
                    thr += 0.5
                return thr, cond
        return None, None

    def _family_metric(self, text: str) -> Tuple[QuestionFamily, str]:
        if "offside" in text:
            return QuestionFamily.OFFSIDE_MARKET, "OFFSIDES"
        if "corner" in text:
            return QuestionFamily.CORNER_MARKET, "CORNERS"
        if "red card" in text:
            return QuestionFamily.CARD_MARKET, "REDS"
        if re.search(r"\b(yellow card|booking|caution|card)", text):
            return QuestionFamily.CARD_MARKET, "YELLOWS"
        if "both teams" in text and "score" in text:
            return QuestionFamily.GOAL_MARKET, "BTTS"
        if "clean sheet" in text:
            return QuestionFamily.GOAL_MARKET, "CLEAN_SHEET"
        if re.search(r"\b(win|draw|advance|qualify|progress|go through)\b", text):
            return QuestionFamily.MATCH_RESULT, "RESULT"
        if re.search(r"\b(goal|goals|score)\b", text):
            return QuestionFamily.GOAL_MARKET, "GOALS"
        raise QuestionParseError(f"No question family matched: {text!r}")

    def _parse_result(self, raw: str, text: str, qid: str, home: str, away: str,
                      side: Optional[str], rnd: str, weight: float,
                      window: TemporalWindow) -> ParsedQuestion:
        is_knockout = rnd in KNOCKOUT_ROUNDS
        explicit_90 = bool(re.search(
            r"\b(90 minutes|ninety minutes|regulation|normal time|full time result)\b", text))
        if re.search(r"\b(advance|qualify|progress|go through)\b", text):
            scope = ResultScope.ADVANCE
        elif is_knockout and not explicit_90:
            # Platform convention unverified (PRD §0.4) — default knockout "win"
            # to ADVANCE; flip here if the rules say otherwise.
            scope = ResultScope.ADVANCE
        else:
            scope = ResultScope.WIN_90

        if re.search(r"\bdraw\b", text):
            target = "DRAW"
            scope = ResultScope.WIN_90
        elif side is not None:
            target = side
        else:
            raise QuestionParseError(
                f"MATCH_RESULT question names neither team nor draw: {raw!r}")
        return ParsedQuestion(raw, qid, QuestionFamily.MATCH_RESULT, home, away,
                              target, "RESULT", 0.0, Condition.BINARY_YES, window,
                              scope, weight)
