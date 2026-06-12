"""DraftKings scraper — soccer game lines via the public v5 eventgroups API.

Refactored from CoreProp/scrapers/draftkings.py (same curl_cffi transport and
headers). CoreProp used the controldata subcategory API for player props; for
game-level soccer markets the older v5 eventgroups endpoint is simpler and
returns Moneyline / Total Goals / Both Teams to Score in one call.

The World Cup eventGroupId changes per competition — discover it once with
discover_event_group() and pin it in config/scrapers.json.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .common import BookOdds, CHROME_HEADERS, american_to_decimal

logger = logging.getLogger(__name__)

DK_HEADERS = {
    **CHROME_HEADERS,
    "Origin": "https://sportsbook.draftkings.com",
    "Referer": "https://sportsbook.draftkings.com/",
}
V5 = "https://sportsbook.draftkings.com/sites/US-SB/api/v5"
NAV = "https://sportsbook.draftkings.com/sites/US-SB/api/v3/nav/leagues"

GAME_LINE_LABELS = {"moneyline", "total goals", "total", "both teams to score"}


def _session():
    from curl_cffi import requests as cffi_requests
    return cffi_requests.Session(impersonate="chrome")


def discover_event_group(name_hint: str = "world cup") -> Optional[int]:
    """Best-effort: walk DK's nav tree for a soccer league whose name matches."""
    try:
        s = _session()
        r = s.get(NAV, headers=DK_HEADERS, timeout=15)
        s.close()
        if r.status_code != 200:
            return None
        for league in r.json() if isinstance(r.json(), list) else []:
            if name_hint in str(league.get("name", "")).lower():
                return league.get("eventGroupId") or league.get("id")
    except Exception as e:                       # noqa: BLE001
        logger.warning("DK discovery failed: %s", e)
    return None


def scrape_draftkings_soccer(event_group_id: Optional[int]) -> List[BookOdds]:
    if not event_group_id:
        event_group_id = discover_event_group()
        if not event_group_id:
            logger.info("DraftKings: no event group id; skipping.")
            return []
    try:
        s = _session()
        r = s.get(f"{V5}/eventgroups/{event_group_id}?format=json",
                  headers=DK_HEADERS, timeout=20)
        s.close()
        if r.status_code != 200:
            logger.warning("DraftKings HTTP %d", r.status_code)
            return []
        return parse_draftkings(r.json())
    except Exception as e:                       # noqa: BLE001
        logger.warning("DraftKings scrape failed: %s", e)
        return []


def parse_draftkings(payload: dict) -> List[BookOdds]:
    """Pure parser for the v5 eventgroup payload — unit-testable offline."""
    group = payload.get("eventGroup", {})
    events: Dict[str, BookOdds] = {}
    for ev in group.get("events", []):
        home = ev.get("teamName1") or ev.get("team1", {}).get("name")
        away = ev.get("teamName2") or ev.get("team2", {}).get("name")
        if home and away:
            events[str(ev["eventId"])] = BookOdds(
                book="draftkings", home_name=home, away_name=away,
                kickoff=ev.get("startDate", ""))

    for cat in group.get("offerCategories", []):
        for sub in cat.get("offerSubcategoryDescriptors", []):
            offers = sub.get("offerSubcategory", {}).get("offers", [])
            for offer_list in offers:
                for offer in offer_list:
                    label = str(offer.get("label", "")).lower()
                    if label not in GAME_LINE_LABELS:
                        continue
                    game = events.get(str(offer.get("eventId")))
                    if game is None:
                        continue
                    _apply_offer(game, label, offer.get("outcomes", []))
    return [g for g in events.values() if g.h2h or g.totals or g.btts]


def _apply_offer(game: BookOdds, label: str, outcomes: list) -> None:
    if label == "moneyline":
        h2h = {}
        for o in outcomes:
            dec = american_to_decimal(o.get("oddsAmerican"))
            name = str(o.get("label", "")).lower()
            if dec is None:
                continue
            if name == "draw":
                h2h["draw"] = dec
            elif name and name in game.home_name.lower() or game.home_name.lower() in name:
                h2h["home"] = dec
            elif name and name in game.away_name.lower() or game.away_name.lower() in name:
                h2h["away"] = dec
        if {"home", "draw", "away"} <= set(h2h):
            game.h2h = h2h
    elif label in ("total goals", "total"):
        by_line: Dict[float, dict] = {}
        for o in outcomes:
            line = o.get("line")
            dec = american_to_decimal(o.get("oddsAmerican"))
            if line is None or dec is None:
                continue
            side = str(o.get("label", "")).lower()
            by_line.setdefault(float(line), {})[side] = dec
        for line, sides in by_line.items():
            if {"over", "under"} <= set(sides):
                game.totals[line] = (sides["over"], sides["under"])
    elif label == "both teams to score":
        yes = next((american_to_decimal(o.get("oddsAmerican")) for o in outcomes
                    if str(o.get("label", "")).lower() == "yes"), None)
        no = next((american_to_decimal(o.get("oddsAmerican")) for o in outcomes
                   if str(o.get("label", "")).lower() == "no"), None)
        if yes and no:
            game.btts = (yes, no)
