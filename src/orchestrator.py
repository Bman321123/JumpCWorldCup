"""Match-time pipeline (PRD v2.2 §5, §6.9).

Order: parse -> context -> market (Shin devig) -> model -> calibrate MODEL ONLY
-> logit blend -> coherence guardrails over the full question set -> clip -> log.
Per-question failures fall back to base rates with an alert; never silence,
never 0.5, never an unanswered question.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .calibration_layer import CalibrationLayer
from .context_resolver import ContextResolver
from .ensemble_blender import EnsembleBlender
from .guardrail_validator import GuardrailValidator
from .odds_client import OddsClient
from .player_layer import PlayerShares
from .question_classifier import QuestionClassifier
from .stats_engine import ModelParameters, StatsEngine
from .types import (Condition, MatchContext, MotivationState, ParsedQuestion,
                    Prediction, QuestionFamily, QuestionParseError, ResultScope)

logger = logging.getLogger(__name__)

FALLBACK_RATES = {  # emergency only — review any match that uses these
    "win": 0.38, "draw": 0.25, "btts": 0.47, "goal": 0.53,
    "corner": 0.50, "card": 0.55, "offside": 0.55, "default": 0.45,
}


class Orchestrator:
    def __init__(self, config_dir: str, params_path: str,
                 calibrators_path: Optional[str] = None,
                 db_path: Optional[str] = None,
                 odds_api_key: Optional[str] = None,
                 player_shares_path: Optional[str] = None,
                 online: bool = True):
        cfg = Path(config_dir)
        with open(cfg / "round_weights.json") as f:
            self.round_weights = {k: v for k, v in json.load(f).items()
                                  if not k.startswith("_")}
        with open(cfg / "groups.json") as f:
            groups = json.load(f)
        self.team_names = {code: t["name"] for code, t in groups["teams"].items()}

        self.classifier = QuestionClassifier(str(cfg / "groups.json"), self.round_weights)
        self.engine = StatsEngine(ModelParameters.load(params_path))
        self.resolver = ContextResolver(str(cfg / "venues.json"),
                                        str(cfg / "referee_table.json"), online=online)
        self.calibration = (CalibrationLayer(path=calibrators_path)
                            if calibrators_path else CalibrationLayer())
        self.blender = EnsembleBlender()
        self.validator = GuardrailValidator()
        self.players = PlayerShares(player_shares_path)
        self.db_path = db_path
        self.odds = OddsClient(odds_api_key, db_path, self.team_names) if online else None

    # ----- main entry -----

    def predict_match(self, home: str, away: str, match_date: str, questions: List[str],
                      tournament_round: str = "group", stadium: Optional[str] = None,
                      referee_id: Optional[str] = None,
                      home_absences: Optional[List[str]] = None,
                      away_absences: Optional[List[str]] = None,
                      home_state: MotivationState = MotivationState.NORMAL,
                      away_state: MotivationState = MotivationState.NORMAL,
                      output_dir: Optional[str] = None) -> dict:
        ctx = self.resolver.resolve(home, away, match_date, tournament_round,
                                    stadium, referee_id, home_state, away_state)
        ctx.home_absence_mult = self.players.availability_multiplier(home, home_absences or [])
        ctx.away_absence_mult = self.players.availability_multiplier(away, away_absences or [])

        market = self.odds.market_probs(home, away) if self.odds else None

        parsed_list: List[ParsedQuestion] = []
        predictions: List[Prediction] = []
        for i, text in enumerate(questions):
            qid = f"Q{i + 1:03d}"
            try:
                parsed = self.classifier.parse(text, home, away, tournament_round, qid)
                parsed_list.append(parsed)
                predictions.append(self._predict_one(parsed, ctx, market))
            except (QuestionParseError, KeyError, ValueError) as e:
                logger.error("PIPELINE FALLBACK for %r: %s — REVIEW MANUALLY", text, e)
                predictions.append(self._fallback(text, qid, tournament_round))

        predictions = self.validator.validate_match_set(predictions, parsed_list)
        manifest = self._manifest(home, away, match_date, tournament_round, ctx,
                                  market, predictions)
        if output_dir:
            self._write_outputs(manifest, output_dir)
        if self.db_path:
            self._log(manifest)
        return manifest

    # ----- single question -----

    def _predict_one(self, q: ParsedQuestion, ctx: MatchContext,
                     market: Optional[dict]) -> Prediction:
        p_model = self._model_prob(q, ctx)
        p_market = self._market_prob(q, market)
        p_cal = self.calibration.calibrate(p_model, q.family.value)  # model branch only (B5)
        p_blend, source = self.blender.blend(
            p_market, p_cal, q.family.value,
            is_closing_line=bool(market and market.get("is_closing")))
        lam_h, lam_a = self.engine.expected_goals(q.home_team, q.away_team, ctx)
        return Prediction(
            question_id=q.question_id, question_text=q.raw_text,
            family=q.family.value, p_model_raw=p_model, p_model_cal=p_cal,
            p_market=p_market, p_blend=p_blend, p_final=p_blend, source=source,
            round_weight=q.round_weight,
            notes=f"lam_h={lam_h:.3f} lam_a={lam_a:.3f}"
                  + (f"; {ctx.notes}" if ctx.notes else ""))

    def _model_prob(self, q: ParsedQuestion, ctx: MatchContext) -> float:
        f = q.family
        if f == QuestionFamily.MATCH_RESULT:
            if q.scope == ResultScope.ADVANCE and q.target in ("HOME", "AWAY"):
                return self.engine.advance_prob(q.home_team, q.away_team, q.target, ctx)
            r = self.engine.result_probs(q.home_team, q.away_team, ctx)
            return {"HOME": r["home_win"], "DRAW": r["draw"],
                    "AWAY": r["away_win"]}[q.target]
        if f == QuestionFamily.GOAL_MARKET:
            return self.engine.goal_market(q.home_team, q.away_team, q.metric,
                                           q.target, q.threshold, q.condition,
                                           q.window, ctx)
        if f == QuestionFamily.CORNER_MARKET:
            return self.engine.corner_market(q.home_team, q.away_team, q.target,
                                             q.threshold, q.condition, q.window, ctx)
        if f == QuestionFamily.CARD_MARKET:
            ref_mult = self.resolver.referees.multiplier(ctx.referee_id, q.metric)
            return self.engine.card_market(q.home_team, q.away_team, q.target,
                                           q.metric, q.threshold, q.condition,
                                           q.window, ctx, ref_mult)
        if f == QuestionFamily.OFFSIDE_MARKET:
            return self.engine.offside_market(q.home_team, q.away_team, q.target,
                                              q.threshold, q.condition, q.window, ctx)
        raise ValueError(f"No model dispatch for family {f}")

    def _market_prob(self, q: ParsedQuestion, market: Optional[dict]) -> Optional[float]:
        if not market:
            return None
        if (q.family == QuestionFamily.MATCH_RESULT and market.get("h2h")
                and q.scope != ResultScope.ADVANCE):
            return {"HOME": market["h2h"]["home"], "DRAW": market["h2h"]["draw"],
                    "AWAY": market["h2h"]["away"]}.get(q.target)
        if q.family == QuestionFamily.GOAL_MARKET:
            if q.metric == "BTTS" and market.get("btts") is not None:
                return market["btts"]
            if (q.metric == "GOALS" and q.target == "MATCH"
                    and q.window.value == "FULL"):
                p_over = market.get("totals", {}).get(q.threshold)
                if p_over is None:
                    return None
                if q.condition == Condition.GTE:
                    return p_over
                if q.condition == Condition.LT:
                    return 1.0 - p_over
        return None

    def _fallback(self, text: str, qid: str, rnd: str) -> Prediction:
        t = text.lower()
        for key in ("btts", "corner", "card", "offside", "draw", "goal", "win"):
            if key in t or (key == "btts" and "both teams" in t):
                p = FALLBACK_RATES[key]
                break
        else:
            p = FALLBACK_RATES["default"]
        return Prediction(question_id=qid, question_text=text, family="FALLBACK",
                          p_model_raw=p, p_model_cal=p, p_market=None,
                          p_blend=p, p_final=p, source="fallback",
                          round_weight=float(self.round_weights.get(rnd, 1.0)),
                          notes="PIPELINE FALLBACK — review before submitting")

    # ----- output -----

    def _manifest(self, home, away, date, rnd, ctx, market, predictions) -> dict:
        lam_h, lam_a = self.engine.expected_goals(home, away, ctx)
        return {
            "match_id": f"{home}_v_{away}_{date}",
            "home_team": home, "away_team": away, "match_date": date,
            "tournament_round": rnd,
            "round_weight": float(self.round_weights.get(rnd, 1.0)),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_params": {"lambda_home": round(lam_h, 3),
                             "lambda_away": round(lam_a, 3),
                             "rho": self.engine.p.rho},
            "context": {"stadium": ctx.stadium, "referee": ctx.referee_id,
                        "goal_multiplier": round(ctx.goal_multiplier, 3),
                        "card_intensity": round(ctx.card_intensity, 3),
                        "notes": ctx.notes},
            "market_available": market is not None,
            "predictions": [{
                "question_id": p.question_id, "question_text": p.question_text,
                "question_family": p.family,
                "final_probability": round(p.p_final, 4),
                "model_probability": round(p.p_model_raw, 4),
                "model_calibrated": round(p.p_model_cal, 4),
                "market_probability": (round(p.p_market, 4)
                                       if p.p_market is not None else None),
                "source": p.source, "notes": p.notes,
            } for p in predictions],
        }

    def _write_outputs(self, manifest: dict, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / manifest["match_id"]
        with open(f"{base}.json", "w") as f:
            json.dump(manifest, f, indent=1)
        lines = [f"{manifest['match_id']}  round={manifest['tournament_round']} "
                 f"weight={manifest['round_weight']}",
                 f"lambdas: H={manifest['model_params']['lambda_home']} "
                 f"A={manifest['model_params']['lambda_away']}  "
                 f"market={'YES' if manifest['market_available'] else 'NO'}", "-" * 78]
        for p in manifest["predictions"]:
            mkt = f"{p['market_probability']:.3f}" if p["market_probability"] else "  -  "
            lines.append(f"{p['final_probability']:.3f}  (mdl {p['model_probability']:.3f} "
                         f"| mkt {mkt} | {p['source']:>12})  {p['question_text']}")
        with open(f"{base}_summary.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    def _log(self, manifest: dict) -> None:
        try:
            con = sqlite3.connect(self.db_path)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for p in manifest["predictions"]:
                con.execute(
                    """INSERT OR REPLACE INTO predictions_log
                       (prediction_id, match_id, question_id, question_text,
                        question_family, p_model_raw, p_model_cal, p_market,
                        p_blend, submitted_probability, round_weight, source,
                        submitted_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (uuid.uuid4().hex, manifest["match_id"], p["question_id"],
                     p["question_text"], p["question_family"],
                     p["model_probability"], p["model_calibrated"],
                     p["market_probability"], p["final_probability"],
                     p["final_probability"], manifest["round_weight"],
                     p["source"], now))
            con.commit()
            con.close()
        except sqlite3.Error as e:
            logger.warning("predictions_log write failed: %s", e)
