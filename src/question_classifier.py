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
    (r"(\d+(?:\.\d+)?)\s*or fewer", Condition.LT),     # "2 or fewer goals" = <= 2
    (r"(\d+(?:\.\d+)?)\s*or less", Condition.LT),
    (r"at most\s+(\d+(?:\.\d+)?)", Condition.LT),
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
        # Strip parenthetical clarifiers — "(Germany)", "(excluding own goals)",
        # "(90 minutes + stoppage time)". They are NEVER the subject, but a team tag
        # like "Kai Havertz (Germany)" otherwise makes _find_team see "Germany" and
        # route the PLAYER prop to a TEAM market priced ~0.99 (a 0.94-Brier landmine).
        # The win=90 cue ("regulation") sits OUTSIDE the parens, so it is preserved.
        text = re.sub(r"\s*\([^)]*\)", "", text).strip()
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

        # comparative: "will X have/finish with more <metric> than Y" (observed live)
        comp = re.search(r"\bwill (.+?) (?:have|finish with|end with|record|commit|"
                         r"receive|get|be shown|pick up|collect|win|take) "
                         r"more ([a-z' \-]+?) than (.+?)(?:\?|$| in\b| at\b| during\b)",
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

        # first goal of a HALF (single team, no conjunction): "Will X score the
        # first goal of the second half?" — a race within that window, NOT the
        # same as "X scores in the half" (which overprices the weaker team).
        fgh = re.search(r"\bwill (.+?) score the first goal of the (first|second) half",
                        text)
        if fgh and "and" not in fgh.group(0):
            side = self._find_team(fgh.group(1), home_team, away_team)
            if side:
                w = "H1" if fgh.group(2) == "first" else "H2"
                return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET,
                                      home_team, away_team, side,
                                      f"FIRST_IN_HALF|{side}|{w}", 1.0,
                                      Condition.BINARY_YES,
                                      TemporalWindow.H1 if w == "H1" else TemporalWindow.H2,
                                      ResultScope.NONE, weight)

        # first-goal compound (live every match): "Will X score the first goal
        # of the game and Y score in the second half?"
        fg = re.search(r"\bwill (.+?) score the first goal\b.*?\band (.+?) score"
                       r"(?: a goal)?(?: in the (first|second) half)?", text)
        if fg:
            s1 = self._find_team(fg.group(1), home_team, away_team)
            s2 = self._find_team(fg.group(2), home_team, away_team)
            if s1 and s2:
                w2 = {"first": "H1", "second": "H2", None: "FULL"}[fg.group(3)]
                return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET,
                                      home_team, away_team, s1,
                                      f"FGCOMBO|{s1}|{s2}|{w2}", 1.0,
                                      Condition.BINARY_YES, window,
                                      ResultScope.NONE, weight)

        # any OTHER conjunction is an unsupported compound — fail loudly so it
        # routes to the flagged fallback instead of silently pricing one leg
        # (live 2026-06-12: this class of question would otherwise parse as a
        # simple team-scores question — 53% submitted on a ~22% event)
        if re.search(r"\band\b", text) and re.search(r"\b(scores?|goals?)\b", text):
            raise QuestionParseError(f"Unsupported compound question: {raw_text!r}")

        # penalty awarded (observed live)
        if "penalt" in text and "shootout" not in text:
            return ParsedQuestion(raw_text, qid, QuestionFamily.PENALTY_MARKET,
                                  home_team, away_team, "MATCH", "PENALTY", 1.0,
                                  Condition.BINARY_YES, window, ResultScope.NONE,
                                  weight)

        # "both teams >= k <metric>" (observed live 2026-06-13): each team must
        # reach the threshold — NOT a total. v1 priced 'both teams 1+ SOT' as
        # total SOT >= 1: 97% on a ~68% event.
        if "both teams" in text and threshold is not None:
            family, metric = self._family_metric(text)
            if metric in ("SOT", "CORNERS", "CARDS", "YELLOWS", "REDS",
                          "OFFSIDES", "FOULS"):
                return ParsedQuestion(raw_text, qid, family, home_team, away_team,
                                      "MATCH", f"BOTH|{metric}", threshold,
                                      Condition.GTE, window, ResultScope.NONE,
                                      weight)

        # hydration / cooling / water break (new R32+ market): "Will a goal be scored
        # before the first hydration break?" FIFA cooling breaks fall ~30' when it is
        # hot, so this is P(a goal before ~minute 30) — a goal-timing question.
        if re.search(r"\b(hydration|cooling|water)\s+break\b", text) and "goal" in text:
            return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET,
                                  home_team, away_team, "MATCH", "GOAL_BEFORE_BREAK",
                                  1.0, Condition.BINARY_YES, window,
                                  ResultScope.NONE, weight)

        # stoppage / added / injury-time goal (new R16+ market): a goal in added time,
        # a late high-intensity window — priced via goal timing.
        if (("stoppage time" in text or "added time" in text or "injury time" in text)
                and re.search(r"\bgoal|score", text)):
            half = "H1" if ("first half" in text or "1st half" in text) else "H2"
            return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET, home_team,
                                  away_team, "MATCH", f"GOAL_STOPPAGE|{half}", 1.0,
                                  Condition.BINARY_YES, window, ResultScope.NONE, weight)

        # total shots (on AND off target) — a different metric from shots ON target.
        if re.search(r"total shots\b", text) and "on target" not in text and threshold is not None:
            return ParsedQuestion(raw_text, qid, QuestionFamily.SHOTS_MARKET, home_team,
                                  away_team, "TOTAL", "TOTAL_SHOTS", threshold,
                                  condition or Condition.GTE, window, ResultScope.NONE, weight)

        # substitution markets — no structural driver; a calibrated base-rate PRIOR
        # (subs before halftime are uncommon, mostly injury-driven). The live-results
        # family calibration will refine it as these settle.
        if "substitut" in text:
            pr = 0.14 if ("before halftime" in text or "first half" in text
                          or "before half-time" in text) else 0.75
            return ParsedQuestion(raw_text, qid, QuestionFamily.GOAL_MARKET, home_team,
                                  away_team, "MATCH", f"PRIOR|{pr}", 1.0,
                                  Condition.BINARY_YES, window, ResultScope.NONE, weight)

        # player props (observed live): "will <name> score a goal" / "... shot(s) on target"
        # The name class must allow ACCENTED letters (é, í, ñ, ø, ü, ...) — otherwise a
        # name like "Sangaré" fails to match here, falls through to _family_metric, and
        # gets priced as a TEAM shots-on-target market (P >= 0.99), the worst overconfidence
        # landmine on the board. [^\W\d_] matches any Unicode letter (re is unicode by default).
        if side is None and "both teams" not in text:
            player = re.search(r"^will ((?:[^\W\d_]|[ .'\-])+?) (?:score|have|get|record)\b", text)
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
            # "Will a card be shown / a goal be scored / an offside be called (in the
            # first half)?" — indefinite article on a count metric = implicit >= 1.
            if re.search(r"\b(a|an|any)\b", text) and re.search(
                    r"\b(shown|scored|called|awarded|committed|recorded|be there)\b", text):
                return ParsedQuestion(raw_text, qid, family, home_team, away_team,
                                      side or "MATCH", metric, 1.0, Condition.GTE,
                                      window, ResultScope.NONE, weight)
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
                # INCLUSIVE "N or fewer / or less / at most" means <= N, encode as < N+0.5
                # (distinct from exclusive "fewer than N" = < N, which is left as-is)
                elif cond == Condition.LT and thr == int(thr) and (
                        "or fewer" in pattern or "or less" in pattern or "at most" in pattern):
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
        if "foul" in text:
            return QuestionFamily.CARD_MARKET, "FOULS"
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
        if re.search(r"\b(win|winner|draw|tied?|level|ahead|leading|advance|qualify|"
                     r"progress|go through)\b", text):
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
        elif window != TemporalWindow.FULL or "halftime" in text or "half-time" in text:
            # a WINDOWED result ("ahead/tied at halftime") is about the state at that
            # point, never advancement — always the 90-min (windowed) result.
            scope = ResultScope.WIN_90
        elif is_knockout and not explicit_90:
            # CONFIRMED 2026-06-13 (platform rule): knockout "win" = ADVANCE.
            scope = ResultScope.ADVANCE
        else:
            scope = ResultScope.WIN_90

        if re.search(r"\b(draw|tied?|level)\b", text):
            target = "DRAW"                      # incl. "At halftime, will the match be tied?"
            scope = ResultScope.WIN_90
        elif side is not None:
            target = side
        else:
            raise QuestionParseError(
                f"MATCH_RESULT question names neither team nor draw: {raw!r}")
        return ParsedQuestion(raw, qid, QuestionFamily.MATCH_RESULT, home, away,
                              target, "RESULT", 0.0, Condition.BINARY_YES, window,
                              scope, weight)
