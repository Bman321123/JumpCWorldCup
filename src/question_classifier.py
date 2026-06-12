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
        if re.search(r"\b(first half|1st half|opening half|halftime|half-time|half time)\b",
                     text):
            window = TemporalWindow.H1
        elif re.search(r"\b(second half|2nd half)\b", text):
            window = TemporalWindow.H2

        threshold, condition = self._parse_threshold(text)

        # comparative: "will X have more <metric> than Y" (observed live 2026-06-11)
        comp = re.search(r"\bwill (.+?) have more ([a-z' \-]+?) than (.+?)(?:\?|$| in\b| at\b| during\b)",
                         text)
        if comp:
            # target must be the team in group(1), NOT whichever alias is longest
            subject_side = self._find_team(comp.group(1), home_team, away_team)
            if subject_side is not None:
                _, metric = self._family_metric(comp.group(2))
                family = self._comparative_family(metric)
                return ParsedQuestion(raw_text, qid, family, home_team, away_team,
                                      subject_side, metric, 0.0,
                                      Condition.MORE_THAN_OPP, window,
                                      ResultScope.NONE, weight)

        # compound: "both teams score AND N or more total goals" (observed live)
        if ("both teams" in text and "score" in text
                and re.search(r"\band\b", text) and threshold is not None):
            return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET,
                                  home_team, away_team, "MATCH", "BTTS_AND_TOTAL",
                                  threshold, Condition.GTE, window,
                                  ResultScope.NONE, weight)

        # penalty awarded (observed live)
        if "penalt" in text and "shootout" not in text:
            return ParsedQuestion(raw_text, qid, QuestionFamily.PENALTY_MARKET,
                                  home_team, away_team, "MATCH", "PENALTY", 1.0,
                                  Condition.BINARY_YES, window, ResultScope.NONE,
                                  weight)

        # player props (observed live): "will <name> score a goal" / "... shot(s) on target"
        if side is None and "both teams" not in text:
            player = re.search(r"^will ([a-z .'\-]+?) (?:score|have|get|record)\b", text)
            if player and "team" not in player.group(1) and "there" != player.group(1).strip():
                name = player.group(1).strip().title()
                if re.search(r"shots? on target", text):
                    return ParsedQuestion(raw_text, qid, QuestionFamily.PLAYER_MARKET,
                                          home_team, away_team, name, "PLAYER_SOT",
                                          threshold if threshold is not None else 1.0,
                                          condition or Condition.GTE, window,
                                          ResultScope.NONE, weight)
                if re.search(r"\bscore\b", text):
                    return ParsedQuestion(raw_text, qid, QuestionFamily.PLAYER_MARKET,
                                          home_team, away_team, name, "PLAYER_GOAL",
                                          1.0, Condition.BINARY_YES, window,
                                          ResultScope.NONE, weight)

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
            # "Will South Korea score (in the second half)?" — implicit >= 1 goal
            if (family == QuestionFamily.GOAL_MARKET and metric == "GOALS"
                    and side is not None and re.search(r"\bscore\b", text)):
                return ParsedQuestion(raw_text, qid, family, home_team, away_team,
                                      side, "GOALS", 1.0, Condition.GTE, window,
                                      ResultScope.NONE, weight)
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
        if re.search(r"shots? on target", text):
            return QuestionFamily.SHOTS_MARKET, "SOT"
        if "red card" in text:
            return QuestionFamily.CARD_MARKET, "REDS"
        if "yellow card" in text:
            return QuestionFamily.CARD_MARKET, "YELLOWS"
        if re.search(r"\b(booking|caution|cards?)\b", text):
            return QuestionFamily.CARD_MARKET, "CARDS"   # "total cards" = Y + R
        if "both teams" in text and "score" in text:
            return QuestionFamily.GOAL_MARKET, "BTTS"
        if "clean sheet" in text:
            return QuestionFamily.GOAL_MARKET, "CLEAN_SHEET"
        if re.search(r"\b(win|draw|advance|qualify|progress|go through)\b", text):
            return QuestionFamily.MATCH_RESULT, "RESULT"
        if re.search(r"\b(goal|goals|score)\b", text):
            return QuestionFamily.GOAL_MARKET, "GOALS"
        raise QuestionParseError(f"No question family matched: {text!r}")

    @staticmethod
    def _comparative_family(metric: str) -> QuestionFamily:
        return {
            "CORNERS": QuestionFamily.CORNER_MARKET,
            "SOT": QuestionFamily.SHOTS_MARKET,
            "CARDS": QuestionFamily.CARD_MARKET,
            "YELLOWS": QuestionFamily.CARD_MARKET,
            "OFFSIDES": QuestionFamily.OFFSIDE_MARKET,
            "GOALS": QuestionFamily.GOAL_MARKET,
        }.get(metric, QuestionFamily.GOAL_MARKET)

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
