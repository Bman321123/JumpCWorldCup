"""Shared types for the forecasting pipeline (PRD v2.2 §5.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionFamily(Enum):
    MATCH_RESULT = "MATCH_RESULT"
    GOAL_MARKET = "GOAL_MARKET"
    CORNER_MARKET = "CORNER_MARKET"
    CARD_MARKET = "CARD_MARKET"
    OFFSIDE_MARKET = "OFFSIDE_MARKET"
    SHOTS_MARKET = "SHOTS_MARKET"          # shots on target (observed live 2026-06-11)
    PENALTY_MARKET = "PENALTY_MARKET"      # "will a penalty be awarded" (observed live)
    PLAYER_MARKET = "PLAYER_MARKET"


class Condition(Enum):
    GTE = "GTE"
    LT = "LT"
    EQ = "EQ"
    BINARY_YES = "YES"
    MORE_THAN_OPP = "GT_OPP"               # comparative: team X stat > team Y stat


class TemporalWindow(Enum):
    FULL = "FULL"
    H1 = "H1"
    H2 = "H2"


class ResultScope(Enum):
    """Knockout disambiguation: result in 90 minutes vs. progressing by any means."""
    WIN_90 = "WIN_90"
    ADVANCE = "ADVANCE"
    NONE = "NONE"


class MotivationState(Enum):
    MUST_WIN = "MUST_WIN"
    SAFE = "SAFE"
    ELIMINATED = "ELIMINATED"
    NORMAL = "NORMAL"


class QuestionParseError(Exception):
    """Raised when a question cannot be mapped to a ParsedQuestion. Never guess silently."""


@dataclass
class ParsedQuestion:
    raw_text: str
    question_id: str
    family: QuestionFamily
    home_team: str                      # FIFA code, e.g. "ARG"
    away_team: str
    target: str                         # "HOME" | "AWAY" | "MATCH" | "DRAW"
    metric: str                         # GOALS|BTTS|CORNERS|YELLOWS|REDS|OFFSIDES|RESULT|CLEAN_SHEET
    threshold: float
    condition: Condition
    window: TemporalWindow
    scope: ResultScope = ResultScope.NONE
    round_weight: float = 1.0


@dataclass
class MatchContext:
    home_team: str
    away_team: str
    match_date: str
    tournament_round: str = "group"
    stadium: Optional[str] = None
    home_is_host: bool = False
    away_is_host: bool = False
    referee_id: Optional[str] = None
    goal_multiplier: float = 1.0        # altitude/weather, applied to lambdas
    corner_multiplier: float = 1.0
    card_intensity: float = 1.0         # motivation / knockout intensity
    home_state: MotivationState = MotivationState.NORMAL
    away_state: MotivationState = MotivationState.NORMAL
    home_absence_mult: float = 1.0      # player layer, applied to attack lambda
    away_absence_mult: float = 1.0
    notes: str = ""


@dataclass
class Prediction:
    question_id: str
    question_text: str
    family: str
    p_model_raw: float                  # stats engine output
    p_model_cal: float                  # after per-family calibrator (model branch only)
    p_market: Optional[float]           # Shin-devigged; NEVER calibrated (PRD B5)
    p_blend: float                      # logit-space blend
    p_final: float                      # after coherence guardrails + clip — submit this
    source: str = "model"
    round_weight: float = 1.0
    notes: str = ""
