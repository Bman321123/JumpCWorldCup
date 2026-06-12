"""Pinnacle scraper — soccer match markets via guest.api.arcadia.pinnacle.com.

Refactored from CoreProp/scrapers/pinnacle.py (same transport: curl_cffi with
Chrome impersonation, same matchups + markets/straight join), retargeted from
US player props to World Cup 1X2 / totals / BTTS. Pinnacle is the sharp anchor:
when it has a market, no consensus is taken.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .common import BookOdds, CHROME_HEADERS, american_to_decimal

logger = logging.getLogger(__name__)

BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
PINNACLE_HEADERS = {
    **CHROME_HEADERS,
    "Origin": "https://www.pinnacle.com",
    "Referer": "https://www.pinnacle.com/",
}
WORLD_CUP_NAME_HINTS = ("fifa world cup", "world cup")
EXCLUDE_HINTS = ("qualif", "u20", "u-20", "u17", "u-17", "women", "esoccer", "friendl")


def _get(session, url: str):
    r = session.get(url, headers=PINNACLE_HEADERS, timeout=20)
    if r.status_code != 200:
        logger.warning("Pinnacle HTTP %d on %s", r.status_code, url)
        return None
    return r.json()


def discover_league_ids(session) -> List[int]:
    """Find active Pinnacle league ids whose name says World Cup."""
    sports = _get(session, f"{BASE}/sports") or []
    soccer_id = next((s["id"] for s in sports
                      if s.get("name", "").lower() == "soccer"), 29)
    leagues = _get(session, f"{BASE}/sports/{soccer_id}/leagues?all=false") or []
    out = []
    for lg in leagues:
        name = lg.get("name", "").lower()
        if any(h in name for h in WORLD_CUP_NAME_HINTS) \
                and not any(x in name for x in EXCLUDE_HINTS):
            out.append(lg["id"])
    logger.info("Pinnacle: %d World Cup league id(s) discovered: %s", len(out), out)
    return out


def scrape_pinnacle_soccer(league_ids: Optional[List[int]] = None) -> List[BookOdds]:
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
    except Exception as e:                       # noqa: BLE001
        logger.warning("curl_cffi unavailable: %s", e)
        return []
    try:
        if not league_ids:
            league_ids = discover_league_ids(session)
        out: List[BookOdds] = []
        for lid in league_ids:
            matchups = _get(session, f"{BASE}/leagues/{lid}/matchups") or []
            markets = _get(session, f"{BASE}/leagues/{lid}/markets/straight") or []
            out.extend(parse_pinnacle(matchups, markets))
        return out
    except Exception as e:                       # noqa: BLE001
        logger.warning("Pinnacle scrape failed: %s", e)
        return []
    finally:
        session.close()


def parse_pinnacle(matchups: list, markets: list) -> List[BookOdds]:
    """Pure parser — unit-testable offline."""
    games: Dict[int, BookOdds] = {}
    btts_specials: Dict[int, dict] = {}          # special matchup id -> info

    for m in matchups:
        if m.get("type") == "special":
            special = m.get("special", {})
            desc = special.get("description", "").lower()
            parent = (m.get("parent") or {})
            if "both teams to score" in desc and parent.get("id"):
                yes_pid = no_pid = None
                for p in m.get("participants", []):
                    if p.get("name") == "Yes":
                        yes_pid = p.get("id")
                    elif p.get("name") == "No":
                        no_pid = p.get("id")
                btts_specials[m["id"]] = {"parent": parent["id"],
                                          "yes": yes_pid, "no": no_pid}
            continue
        parts = m.get("participants", [])
        home = next((p.get("name") for p in parts if p.get("alignment") == "home"), None)
        away = next((p.get("name") for p in parts if p.get("alignment") == "away"), None)
        if not (home and away):
            continue
        if "(" in home:                          # "Team (Corners)" etc. — handled below
            continue
        games[m["id"]] = BookOdds(book="pinnacle", home_name=home,
                                  away_name=away,
                                  kickoff=m.get("startTime", ""))

    # Pinnacle lists corners/bookings as separate "Team (Corners)" matchups —
    # sharp corner lines exist for every WC fixture (discovered live 2026-06-12).
    SUFFIXES = {"(corners)": ("corner_totals", "h1_corner_totals"),
                "(bookings)": ("booking_totals", "h1_booking_totals"),
                "(cards)": ("booking_totals", "h1_booking_totals")}
    derived_map: Dict[int, tuple] = {}           # matchup id -> (game id, fields)
    name_index = {(g.home_name.lower(), g.away_name.lower()): mid
                  for mid, g in games.items()}
    for m in matchups:
        parts = m.get("participants", [])
        home = next((p.get("name") for p in parts if p.get("alignment") == "home"), None)
        away = next((p.get("name") for p in parts if p.get("alignment") == "away"), None)
        if not (home and away):
            continue
        suffix = next((s for s in SUFFIXES if s in home.lower()), None)
        if suffix is None:
            continue
        fields = SUFFIXES[suffix]
        parent_id = (m.get("parent") or {}).get("id")
        if parent_id in games:
            derived_map[m["id"]] = (parent_id, fields)
        else:
            key = (home.lower().replace(f" {suffix}", ""),
                   away.lower().replace(f" {suffix}", ""))
            if key in name_index:
                derived_map[m["id"]] = (name_index[key], fields)

    for mkt in markets:
        mid = mkt.get("matchupId")
        mtype = mkt.get("type")
        period = mkt.get("period", 0)
        prices = mkt.get("prices", [])

        if mid in games and mtype == "moneyline" and period == 0:
            by_desig = {p.get("designation"): american_to_decimal(p.get("price"))
                        for p in prices}
            if all(by_desig.get(k) for k in ("home", "draw", "away")):
                games[mid].h2h = {k: by_desig[k] for k in ("home", "draw", "away")}

        elif mid in games and mtype == "total" and period in (0, 1):
            line = next((p.get("points") for p in prices
                         if p.get("points") is not None), None)
            over = next((american_to_decimal(p.get("price")) for p in prices
                         if p.get("designation") == "over"), None)
            under = next((american_to_decimal(p.get("price")) for p in prices
                          if p.get("designation") == "under"), None)
            if line is not None and over and under:
                target = games[mid].totals if period == 0 else games[mid].h1_totals
                target[float(line)] = (over, under)

        elif mid in derived_map and mtype == "total" and period in (0, 1):
            game_id, fields = derived_map[mid]
            game = games[game_id]
            line = next((p.get("points") for p in prices
                         if p.get("points") is not None), None)
            over = next((american_to_decimal(p.get("price")) for p in prices
                         if p.get("designation") == "over"), None)
            under = next((american_to_decimal(p.get("price")) for p in prices
                          if p.get("designation") == "under"), None)
            # guard: booking POINTS lines (yellow=10/red=25 scoring) sit at 20+;
            # only card-count lines are usable
            if line is not None and over and under and float(line) < 16:
                target = getattr(game, fields[0] if period == 0 else fields[1])
                target[float(line)] = (over, under)

        elif mid in btts_specials and mtype == "moneyline":
            info = btts_specials[mid]
            game = games.get(info["parent"])
            if game is None:
                continue
            yes = next((american_to_decimal(p.get("price")) for p in prices
                        if p.get("participantId") == info["yes"]), None)
            no = next((american_to_decimal(p.get("price")) for p in prices
                       if p.get("participantId") == info["no"]), None)
            if yes and no:
                game.btts = (yes, no)

    return [g for g in games.values()
            if g.h2h or g.totals or g.btts]
