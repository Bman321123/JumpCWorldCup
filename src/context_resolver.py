"""Match context: venue/altitude, weather, referee, motivation states
(PRD v2.2 §6.6 — must-win logic fixed vs v1.0 bug B4).
"""
from __future__ import annotations

import json
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

from .types import MatchContext, MotivationState

logger = logging.getLogger(__name__)

HOSTS = {"MEX", "CAN", "USA"}
ALTITUDE_GOAL_MULT = 0.92       # >= 1500m (Azteca 2240m, Akron 1560m)
ALTITUDE_CORNER_MULT = 0.95
ALTITUDE_THRESHOLD_M = 1500.0
HEAT_GOAL_MULT = 0.95           # kickoff temp > 32C
HEAT_THRESHOLD_C = 32.0
RAIN_CORNER_MULT = 1.08         # precipitation probability > 60%
REF_SHRINK_N = 10.0             # shrink referee multiplier toward 1.0 below ~10 matches

MUST_WIN_CARD_MULT = 1.10
DEAD_RUBBER_CARD_MULT = 0.92
KNOCKOUT_CARD_MULT = 1.08

# P(a third-placed team with N points advances) — 8 best of 12 groups.
# Placeholder until simulated cross-group (PRD §6.6); injectable for tests.
THIRD_PLACE_ADV_BY_POINTS = {0: 0.0, 1: 0.02, 2: 0.10, 3: 0.35, 4: 0.80, 5: 0.97,
                             6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}


class RefereeTable:
    def __init__(self, path: Optional[str] = None):
        self.table: Dict[str, dict] = {}
        if path:
            try:
                with open(path) as f:
                    self.table = json.load(f).get("referees", {})
            except FileNotFoundError:
                logger.warning("No referee table at %s; multipliers default to 1.0", path)

    def multiplier(self, referee_id: Optional[str], card_type: str = "YELLOWS",
                   global_avg: float = 3.6) -> float:
        if not referee_id or referee_id not in self.table:
            return 1.0
        ref = self.table[referee_id]
        rate = ref.get("yellow_per_match" if card_type == "YELLOWS" else "red_per_match")
        n = float(ref.get("total_matches", 0))
        if rate is None or n <= 0:
            return 1.0
        raw = rate / (global_avg if card_type == "YELLOWS" else 0.18)
        return float((n * raw + REF_SHRINK_N * 1.0) / (n + REF_SHRINK_N))


def fetch_weather(lat: float, lon: float, iso_date: str) -> Optional[dict]:
    """Open-Meteo, free, no key. Graceful None on any failure."""
    try:
        import requests
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "daily": "temperature_2m_max,precipitation_probability_max",
                    "start_date": iso_date, "end_date": iso_date, "timezone": "auto"},
            timeout=6)
        r.raise_for_status()
        d = r.json().get("daily", {})
        return {"temp_max_c": d.get("temperature_2m_max", [None])[0],
                "precip_prob": d.get("precipitation_probability_max", [None])[0]}
    except Exception as e:          # noqa: BLE001 — context must never kill the pipeline
        logger.warning("Weather fetch failed: %s", e)
        return None


def qualification_states(
    standings: Dict[str, Dict[str, float]],
    remaining: List[Tuple[str, str]],
    focal: Tuple[str, str],
    probs_fn: Callable[[str, str], Tuple[float, float, float]],
    n_sims: int = 2000,
    third_adv_by_points: Optional[Dict[int, float]] = None,
    seed: int = 7,
) -> Dict[str, MotivationState]:
    """Monte Carlo over remaining group fixtures. For each team in the focal
    match, estimate P(advance | focal result) and classify.

    Correct logic (v1.0 had it inverted): MUST_WIN = alive with a win, dead
    without one. ELIMINATED = dead even with a win. SAFE = through regardless.
    """
    third_table = third_adv_by_points or THIRD_PLACE_ADV_BY_POINTS
    rng = random.Random(seed)
    teams = list(standings.keys())
    others = [m for m in remaining if m != focal]
    f_home, f_away = focal

    def adv_prob(team: str, forced: str) -> float:
        hits = 0.0
        for _ in range(n_sims):
            pts = {t: standings[t].get("pts", 0) for t in teams}
            gd = {t: standings[t].get("gd", 0) for t in teams}
            _apply(pts, gd, f_home, f_away, forced, rng)
            for h, a in others:
                ph, pd_, _ = probs_fn(h, a)
                u = rng.random()
                res = "H" if u < ph else ("D" if u < ph + pd_ else "A")
                _apply(pts, gd, h, a, res, rng)
            order = sorted(teams, key=lambda t: (pts[t], gd[t], rng.random()),
                           reverse=True)
            rank = order.index(team)
            if rank <= 1:
                hits += 1.0
            elif rank == 2:
                hits += third_table.get(int(pts[team]), 0.0)
        return hits / n_sims

    out: Dict[str, MotivationState] = {}
    for team, win_res, draw_res, loss_res in (
            (f_home, "H", "D", "A"), (f_away, "A", "D", "H")):
        p_win = adv_prob(team, win_res)
        p_draw = adv_prob(team, draw_res)
        p_loss = adv_prob(team, loss_res)
        if p_win < 0.01:
            out[team] = MotivationState.ELIMINATED
        elif min(p_win, p_draw, p_loss) > 0.95:
            out[team] = MotivationState.SAFE
        elif p_win >= 0.20 and p_draw <= 0.15 and p_loss <= 0.10:
            out[team] = MotivationState.MUST_WIN
        else:
            out[team] = MotivationState.NORMAL
    return out


def _apply(pts, gd, home, away, result, rng) -> None:
    margin = rng.choices([1, 2, 3], weights=[0.60, 0.25, 0.15])[0]
    if result == "H":
        pts[home] += 3
        gd[home] += margin
        gd[away] -= margin
    elif result == "A":
        pts[away] += 3
        gd[away] += margin
        gd[home] -= margin
    else:
        pts[home] += 1
        pts[away] += 1


def card_intensity(home_state: MotivationState, away_state: MotivationState,
                   tournament_round: str) -> float:
    if tournament_round != "group":
        return KNOCKOUT_CARD_MULT
    states = {home_state, away_state}
    if MotivationState.MUST_WIN in states:
        return MUST_WIN_CARD_MULT
    if states <= {MotivationState.SAFE, MotivationState.ELIMINATED}:
        return DEAD_RUBBER_CARD_MULT          # dead rubber — v1.0 had this backwards
    return 1.0


class ContextResolver:
    def __init__(self, venues_path: Optional[str] = None,
                 referee_path: Optional[str] = None, online: bool = True):
        self.venues: Dict[str, dict] = {}
        if venues_path:
            with open(venues_path) as f:
                self.venues = {v["stadium"].lower(): v for v in json.load(f)["venues"]}
        self.referees = RefereeTable(referee_path)
        self.online = online

    def resolve(self, home: str, away: str, match_date: str,
                tournament_round: str = "group", stadium: Optional[str] = None,
                referee_id: Optional[str] = None,
                home_state: MotivationState = MotivationState.NORMAL,
                away_state: MotivationState = MotivationState.NORMAL) -> MatchContext:
        notes = []
        goal_mult, corner_mult = 1.0, 1.0
        venue = self.venues.get((stadium or "").lower())
        if venue:
            if venue.get("elevation_m", 0) >= ALTITUDE_THRESHOLD_M:
                goal_mult *= ALTITUDE_GOAL_MULT
                corner_mult *= ALTITUDE_CORNER_MULT
                notes.append(f"altitude {venue['elevation_m']}m")
            if self.online:
                wx = fetch_weather(venue["lat"], venue["lon"], match_date)
                if wx:
                    if (wx.get("temp_max_c") or 0) > HEAT_THRESHOLD_C:
                        goal_mult *= HEAT_GOAL_MULT
                        notes.append(f"heat {wx['temp_max_c']}C")
                    if (wx.get("precip_prob") or 0) > 60:
                        corner_mult *= RAIN_CORNER_MULT
                        notes.append("rain")
        return MatchContext(
            home_team=home, away_team=away, match_date=match_date,
            tournament_round=tournament_round, stadium=stadium,
            home_is_host=home in HOSTS, away_is_host=away in HOSTS,
            referee_id=referee_id, goal_multiplier=goal_mult,
            corner_multiplier=corner_mult,
            card_intensity=card_intensity(home_state, away_state, tournament_round),
            home_state=home_state, away_state=away_state,
            notes="; ".join(notes))
