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
                    Prediction, QuestionFamily, QuestionParseError, ResultScope,
                    TemporalWindow)

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
        self.team_aliases = {code: [t["name"], code] + t.get("aliases", [])
                             for code, t in groups["teams"].items()}

        self.classifier = QuestionClassifier(str(cfg / "groups.json"), self.round_weights)
        self.engine = StatsEngine(ModelParameters.load(params_path))
        self.resolver = ContextResolver(str(cfg / "venues.json"),
                                        str(cfg / "referee_table.json"), online=online)
        if calibrators_path is None:
            default_cal = Path(params_path).parent / "calibrators.joblib"
            if default_cal.exists():
                calibrators_path = str(default_cal)
        self.calibration = (CalibrationLayer(path=calibrators_path)
                            if calibrators_path else CalibrationLayer())
        self.blender = EnsembleBlender()
        self.validator = GuardrailValidator()
        self.players = PlayerShares(player_shares_path)
        # Gated ML micro-models: a family is ACTIVE only if it passed its
        # walk-forward ship-gate at train time. Inactive families fall back to
        # the structural model. The registry auto-wires any passer.
        from .ml_models import MLRegistry
        self.ml = MLRegistry(str(Path(params_path).parent))
        self.db_path = db_path
        # Odds hierarchy: scraped sharp books first (free), The Odds API as
        # fallback only if a key is configured, else pure model.
        self.odds_sources = []
        if online:
            from .scrapers.aggregator import ScrapedOddsClient
            self.odds_sources.append(ScrapedOddsClient(
                db_path, self.team_names, str(cfg / "scrapers.json"),
                self.team_aliases))
            import os
            if odds_api_key or os.environ.get("ODDS_API_KEY"):
                self.odds_sources.append(OddsClient(odds_api_key, db_path,
                                                    self.team_names))

    # ----- main entry -----

    def predict_match(self, home: str, away: str, match_date: str, questions: List[str],
                      tournament_round: str = "group", stadium: Optional[str] = None,
                      referee_id: Optional[str] = None,
                      home_absences: Optional[List[str]] = None,
                      away_absences: Optional[List[str]] = None,
                      home_state: MotivationState = MotivationState.NORMAL,
                      away_state: MotivationState = MotivationState.NORMAL,
                      output_dir: Optional[str] = None,
                      lookup_referee: bool = True,
                      compute_motivation: bool = True) -> dict:
        # group-stage motivation: from matchday 2 the standings decide who is
        # must-win / safe / eliminated (affects card intensity). No-op on
        # matchday 1 (everyone NORMAL) and gracefully skipped offline.
        if (compute_motivation and tournament_round == "group" and self.odds_sources
                and home_state == MotivationState.NORMAL
                and away_state == MotivationState.NORMAL):
            try:
                from .standings import group_motivation
                probs_fn = lambda h, a: tuple(self.engine.result_probs(h, a).values())
                home_state, away_state = group_motivation(home, away, probs_fn)
                if home_state != MotivationState.NORMAL or away_state != MotivationState.NORMAL:
                    logger.info("Motivation %s=%s %s=%s", home, home_state.value,
                                away, away_state.value)
            except Exception as e:               # noqa: BLE001
                logger.warning("Motivation skipped: %s", e)
        # attach the assigned referee (ESPN) when online and not supplied, so
        # card markets use the real official instead of league-average
        if referee_id is None and self.odds_sources and lookup_referee:
            home_name = self.team_names.get(home, home)
            away_name = self.team_names.get(away, away)
            try:
                from .espn_live import match_referee
                referee_id = match_referee(home_name, away_name)
                if referee_id:
                    logger.info("Referee for %s v %s: %s", home, away, referee_id)
            except Exception as e:               # noqa: BLE001
                logger.warning("Referee lookup skipped: %s", e)
        # live current events: auto-derive absences from the official XI once
        # published (~60 min pre-kickoff). Hard structured fact -> auto-applied.
        if self.odds_sources and not home_absences and not away_absences:
            try:
                from .live_context import confirmed_xi, derive_absences
                xi = confirmed_xi(self.team_names.get(home, home),
                                  self.team_names.get(away, away))
                if xi:
                    home_absences = derive_absences(home, xi.get("home", []),
                                                    self.players.players)
                    away_absences = derive_absences(away, xi.get("away", []),
                                                    self.players.players)
                    if home_absences or away_absences:
                        logger.info("Lineup absences: %s=%s %s=%s", home,
                                    home_absences, away, away_absences)
            except Exception as e:               # noqa: BLE001
                logger.warning("Lineup check skipped: %s", e)

        ctx = self.resolver.resolve(home, away, match_date, tournament_round,
                                    stadium, referee_id, home_state, away_state)
        ctx.home_absence_mult = self.players.availability_multiplier(home, home_absences or [])
        ctx.away_absence_mult = self.players.availability_multiplier(away, away_absences or [])

        market = None
        for source in self.odds_sources:
            market = source.market_probs(home, away)
            if market:
                logger.info("Market data from %s", type(source).__name__)
                break

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
        if q.condition == Condition.MORE_THAN_OPP:
            return self.engine.comparative_prob(q.home_team, q.away_team, q.metric,
                                                q.target, q.window, ctx)
        if q.metric.startswith("BOTH|"):
            ref_mult = self.resolver.referees.multiplier(ctx.referee_id, "YELLOWS")
            return self.engine.both_teams_prob(q.home_team, q.away_team,
                                               q.metric.split("|")[1], q.threshold,
                                               q.window, ctx, ref_mult)
        if f == QuestionFamily.PENALTY_MARKET:
            return self.engine.penalty_prob(ctx)
        if f == QuestionFamily.SHOTS_MARKET:
            return self.engine.shots_market(q.home_team, q.away_team, q.target,
                                            q.threshold, q.condition, q.window, ctx)
        if f == QuestionFamily.PLAYER_MARKET:
            from .player_layer import player_prop_prob
            lam_h, lam_a = self.engine.expected_goals(q.home_team, q.away_team, ctx)
            p, note = player_prop_prob(q.target, q.metric, q.threshold,
                                       lam_h, lam_a, q.home_team, q.away_team,
                                       self.players)
            if "REVIEW" in note:
                logger.warning("Player prop needs review: %s", note)
            return p
        if f == QuestionFamily.MATCH_RESULT:
            if q.scope == ResultScope.ADVANCE and q.target in ("HOME", "AWAY"):
                return self.engine.advance_prob(q.home_team, q.away_team, q.target, ctx)
            r = self.engine.result_probs(q.home_team, q.away_team, ctx, q.window)
            return {"HOME": r["home_win"], "DRAW": r["draw"],
                    "AWAY": r["away_win"]}[q.target]
        if f == QuestionFamily.GOAL_MARKET:
            if q.metric.startswith("FGCOMBO|"):
                _, s1, s2, w2 = q.metric.split("|")
                return self.engine.first_goal_combo(
                    q.home_team, q.away_team, s1, s2, TemporalWindow(w2), ctx)
            return self.engine.goal_market(q.home_team, q.away_team, q.metric,
                                           q.target, q.threshold, q.condition,
                                           q.window, ctx)
        if f == QuestionFamily.CORNER_MARKET:
            ml = self._ml_total("corners", q, ctx)
            if ml is not None:
                return ml
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

    # families whose per-team rates transfer directly from club training to WC
    # deployment (raw rate tables, not Dixon-Coles): corners/cards/sot/fouls.
    _ML_OPP = {"corners": True, "sot": True, "cards": False, "fouls": False}

    def _ml_total(self, family: str, q: ParsedQuestion, ctx: MatchContext):
        """Gated ML for full-match total markets. Returns P(condition) or None
        when the family's model is inactive / question isn't a match total."""
        model = self.ml.get(family)
        if (model is None or q.target != "MATCH"
                or q.window != TemporalWindow.FULL
                or q.condition not in (Condition.GTE, Condition.LT)):
            return None
        import math
        from scipy.stats import poisson
        from .stats_engine import DEFAULTS
        p = self.engine.p
        if family == "corners":
            hf, hg = p.corner_for.get(q.home_team, DEFAULTS["corner_for"]), \
                     p.corner_against.get(q.home_team, DEFAULTS["corner_against"])
            af, ag = p.corner_for.get(q.away_team, DEFAULTS["corner_for"]), \
                     p.corner_against.get(q.away_team, DEFAULTS["corner_against"])
            avg = DEFAULTS["corner_against"]
        elif family == "sot":
            hf = p.sot_rates.get(q.home_team, DEFAULTS["sot_for"]); hg = DEFAULTS["sot_for"]
            af = p.sot_rates.get(q.away_team, DEFAULTS["sot_for"]); ag = DEFAULTS["sot_for"]
            avg = DEFAULTS["sot_for"]
        elif family == "cards":
            hf = p.yellow_rates.get(q.home_team, DEFAULTS["yellow"]) + \
                 p.red_rates.get(q.home_team, DEFAULTS["red"])
            af = p.yellow_rates.get(q.away_team, DEFAULTS["yellow"]) + \
                 p.red_rates.get(q.away_team, DEFAULTS["red"])
            hg = ag = avg = hf + af
        elif family == "fouls":
            hf = p.fouls_rates.get(q.home_team, DEFAULTS["fouls"])
            af = p.fouls_rates.get(q.away_team, DEFAULTS["fouls"])
            hg = ag = avg = DEFAULTS["fouls"]
        else:
            return None
        # current events reach the ML: a key attacker out lowers the team's
        # attacking output (corners, shots on target). Cards/fouls unaffected.
        if family in ("corners", "sot"):
            hf *= ctx.home_absence_mult
            af *= ctx.away_absence_mult
        lam = (hf * (ag / avg) + af * (hg / avg)) if self._ML_OPP[family] else hf + af
        lam_h, lam_a = self.engine.expected_goals(q.home_team, q.away_team, ctx)
        k = math.ceil(q.threshold)
        struct_over = float(1.0 - poisson.cdf(k - 1, lam))
        feats = {"home_for": hf, "home_against": hg, "away_for": af,
                 "away_against": ag, "home_goals": lam_h, "away_goals": lam_a,
                 "lam_struct": lam, "threshold": q.threshold,
                 "struct_prob": struct_over, "mkt": float("nan")}
        p_over = model.prob_over(feats)
        if p_over is None:
            return None
        return p_over if q.condition == Condition.GTE else 1.0 - p_over

    def _market_prob(self, q: ParsedQuestion, market: Optional[dict]) -> Optional[float]:
        if not market:
            return None
        if (q.family == QuestionFamily.MATCH_RESULT and market.get("h2h")
                and q.scope != ResultScope.ADVANCE
                and q.window.value == "FULL"):    # never price an H1 result off 90' odds
            return {"HOME": market["h2h"]["home"], "DRAW": market["h2h"]["draw"],
                    "AWAY": market["h2h"]["away"]}.get(q.target)
        if q.family == QuestionFamily.GOAL_MARKET:
            if q.metric == "BTTS" and market.get("btts") is not None:
                return market["btts"]
            if q.metric == "GOALS" and q.target == "MATCH":
                table = (market.get("totals", {}) if q.window.value == "FULL"
                         else market.get("h1_totals", {}) if q.window.value == "H1"
                         else {})
                return self._totals_lookup(table, q.threshold, q.condition)
        if (q.family == QuestionFamily.CORNER_MARKET and q.target == "MATCH"
                and q.condition in (Condition.GTE, Condition.LT)):
            table = (market.get("corner_totals", {}) if q.window.value == "FULL"
                     else market.get("h1_corner_totals", {}) if q.window.value == "H1"
                     else {})
            return self._totals_lookup(table, q.threshold, q.condition)
        if (q.family == QuestionFamily.CARD_MARKET and q.target == "MATCH"
                and q.metric == "CARDS"
                and q.condition in (Condition.GTE, Condition.LT)):
            table = (market.get("booking_totals", {}) if q.window.value == "FULL"
                     else market.get("h1_booking_totals", {}) if q.window.value == "H1"
                     else {})
            return self._totals_lookup(table, q.threshold, q.condition)
        return None

    @staticmethod
    def _totals_lookup(table: dict, threshold: float,
                       condition: Condition) -> Optional[float]:
        """'k or more' (integer k) == 'over k-0.5'; half-line thresholds map to
        their own book line. Whole book lines have push semantics — never use
        them for an integer GTE/LT question."""
        if not table:
            return None
        line = threshold - 0.5 if float(threshold).is_integer() else threshold
        p_over = table.get(line, table.get(str(line)))
        if p_over is None:
            return None
        return p_over if condition == Condition.GTE else 1.0 - p_over

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
