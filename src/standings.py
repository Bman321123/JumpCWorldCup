"""Live group standings + remaining fixtures from ESPN (ROADMAP D).

Feeds the motivation Monte Carlo (context_resolver.qualification_states) so that
from matchday 2 the model knows who is must-win / safe / eliminated and adjusts
card intensity accordingly. Degrades to empty (NORMAL states) on any failure.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
GROUP_WINDOW = "20260611-20260712"


def _groups() -> Dict[str, str]:
    """team code -> group letter, plus an alias index name->code."""
    with open(Path(__file__).resolve().parents[1] / "config" / "groups.json") as f:
        cfg = json.load(f)
    team_group = {}
    for letter, codes in cfg["groups"].items():
        for c in codes:
            team_group[c] = letter
    alias = {}
    for code, t in cfg["teams"].items():
        alias[t["name"].lower()] = code
        alias[code.lower()] = code
        for a in t.get("aliases", []):
            alias[a.lower()] = code
    return team_group, alias


def fetch_group_state(code: str, alias: dict) -> Optional[str]:
    return alias.get(code.lower())


def load_standings() -> Tuple[Dict[str, dict], List[Tuple[str, str, str]]]:
    """Returns (standings, fixtures). standings: code -> {pts, gd, group}.
    fixtures: list of (home_code, away_code, status) where status in
    {'post','pre'}."""
    team_group, alias = _groups()
    standings = {c: {"pts": 0, "gd": 0, "group": g} for c, g in team_group.items()}
    fixtures: List[Tuple[str, str, str]] = []
    try:
        import requests
        board = requests.get(f"{BASE}/scoreboard",
                             params={"dates": GROUP_WINDOW, "limit": 200},
                             timeout=15, headers={"User-Agent": "Mozilla/5.0"}).json()
    except Exception as e:                       # noqa: BLE001
        logger.warning("Standings fetch failed: %s", e)
        return standings, fixtures

    for ev in board.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        cs = comp.get("competitors", [])
        if len(cs) != 2:
            continue
        home = next((c for c in cs if c.get("homeAway") == "home"), cs[0])
        away = next((c for c in cs if c.get("homeAway") == "away"), cs[1])
        hc = alias.get((home.get("team", {}).get("displayName") or "").lower())
        ac = alias.get((away.get("team", {}).get("displayName") or "").lower())
        if not hc or not ac:
            continue
        state = ev.get("status", {}).get("type", {}).get("state", "pre")
        if state == "post":
            try:
                hs, as_ = int(home.get("score")), int(away.get("score"))
            except (TypeError, ValueError):
                continue
            fixtures.append((hc, ac, "post"))
            if hc in standings and ac in standings:
                standings[hc]["gd"] += hs - as_
                standings[ac]["gd"] += as_ - hs
                if hs > as_:
                    standings[hc]["pts"] += 3
                elif as_ > hs:
                    standings[ac]["pts"] += 3
                else:
                    standings[hc]["pts"] += 1
                    standings[ac]["pts"] += 1
        else:
            fixtures.append((hc, ac, "pre"))
    return standings, fixtures


def group_motivation(home: str, away: str,
                     probs_fn: Callable[[str, str], Tuple[float, float, float]]):
    """Return (home_state, away_state) MotivationState for a group match, or
    (NORMAL, NORMAL) if standings/remaining can't be built."""
    from .context_resolver import qualification_states
    from .types import MotivationState
    standings, fixtures = load_standings()
    if home not in standings or away not in standings:
        return MotivationState.NORMAL, MotivationState.NORMAL
    grp = standings[home]["group"]
    members = {c for c, s in standings.items() if s["group"] == grp}
    grp_standings = {c: standings[c] for c in members}
    remaining = [(h, a) for h, a, st in fixtures
                 if st == "pre" and h in members and a in members]
    if (home, away) not in remaining:
        remaining = remaining + [(home, away)]
    try:
        states = qualification_states(grp_standings, remaining, (home, away),
                                      probs_fn, n_sims=800)
        return (states.get(home, MotivationState.NORMAL),
                states.get(away, MotivationState.NORMAL))
    except Exception as e:                       # noqa: BLE001
        logger.warning("Motivation MC failed: %s", e)
        return MotivationState.NORMAL, MotivationState.NORMAL
