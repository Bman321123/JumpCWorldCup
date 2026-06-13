"""Live current-events layer (ROADMAP: news/lineups/status).

Philosophy: AUTO-APPLY only hard structured facts (match status, the official
starting XI once published, weather). SURFACE news headlines for human judgment
— never auto-parse free text into probabilities (that is the false-signal
trap). The sharp market line already aggregates all news faster than any
scraper, so the structured layer's job is the micro-markets the line doesn't
fully price and the final hour before the line settles.

All functions degrade to None / empty on any failure; the pipeline proceeds.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
HEADERS = {"User-Agent": "Mozilla/5.0"}
KEY_SHARE = 0.10            # a player above this involvement is a "key starter"
KEY_MINUTES = 70


def _teams_match(name: str, home: str, away: str) -> bool:
    n = name.lower()
    return (home.lower() in n or n in home.lower()) and \
           (away.lower() in n or n in away.lower())


def _find_event(home_name: str, away_name: str, days: int = 3) -> Optional[dict]:
    try:
        import requests
        start = date.today().strftime("%Y%m%d")
        end = (date.today() + timedelta(days=days)).strftime("%Y%m%d")
        board = requests.get(f"{ESPN}/scoreboard",
                             params={"dates": f"{start}-{end}", "limit": 100},
                             timeout=12, headers=HEADERS).json()
        for ev in board.get("events", []):
            if _teams_match(ev.get("name", ""), home_name, away_name):
                return ev
    except Exception as e:                       # noqa: BLE001
        logger.warning("Event lookup failed: %s", e)
    return None


def match_status(home_name: str, away_name: str) -> Optional[dict]:
    """{state, detail, venue, postponed} or None. Detects postponement / time /
    venue change — a hard structured fact we act on."""
    ev = _find_event(home_name, away_name)
    if not ev:
        return None
    st = ev.get("status", {}).get("type", {})
    comp = (ev.get("competitions") or [{}])[0]
    state = st.get("state")
    name = (st.get("name") or "").upper()
    return {"state": state, "detail": st.get("detail"),
            "venue": comp.get("venue", {}).get("fullName"),
            "postponed": "POSTPONED" in name or "CANCELED" in name
                         or "DELAYED" in name,
            "kickoff": ev.get("date")}


def confirmed_xi(home_name: str, away_name: str) -> Optional[Dict[str, List[str]]]:
    """Starting XI per side once ESPN publishes it (~60 min pre-kickoff), else
    None. Returns {'home': [names], 'away': [names]}."""
    ev = _find_event(home_name, away_name)
    if not ev:
        return None
    try:
        import requests
        s = requests.get(f"{ESPN}/summary", params={"event": ev["id"]},
                         timeout=12, headers=HEADERS).json()
    except Exception as e:                       # noqa: BLE001
        logger.warning("Lineup fetch failed: %s", e)
        return None
    out: Dict[str, List[str]] = {}
    for r in s.get("rosters", []):
        side = r.get("homeAway")
        starters = [p.get("athlete", {}).get("displayName")
                    for p in r.get("roster", []) if p.get("starter")]
        starters = [n for n in starters if n]
        if side in ("home", "away") and starters:
            out[side] = starters
    return out or None


def derive_absences(team_code: str, xi_names: List[str],
                    player_shares: dict) -> List[str]:
    """Key players (high involvement, regular starters) for this team who are
    NOT in the confirmed XI -> treat as absences. Pure / testable."""
    if not xi_names:
        return []
    xi_lower = {n.lower() for n in xi_names}
    absences = []
    for name, info in player_shares.items():
        if info.get("team") != team_code:
            continue
        if (info.get("share", 0) >= KEY_SHARE
                and info.get("expected_minutes", 0) >= KEY_MINUTES
                and name.lower() not in xi_lower):
            absences.append(name)
    return absences


def news_headlines(home_name: str, away_name: str, limit: int = 6) -> List[str]:
    """Recent headlines via Google News RSS (free, no key). SURFACED for human
    judgment only — never auto-applied to probabilities."""
    import urllib.parse
    import xml.etree.ElementTree as ET
    q = urllib.parse.quote(f"{home_name} {away_name} World Cup team news lineup injury")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        import requests
        r = requests.get(url, timeout=10, headers=HEADERS)
        root = ET.fromstring(r.content)
        out = []
        for item in root.iter("item"):
            title = item.findtext("title")
            if title:
                out.append(title)
            if len(out) >= limit:
                break
        return out
    except Exception as e:                       # noqa: BLE001
        logger.warning("News fetch failed: %s", e)
        return []
