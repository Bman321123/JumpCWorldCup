"""FanDuel scraper — soccer match markets via the two-step sbapi flow.

Refactored from CoreProp/scrapers/fanduel.py: identical strategy (Phase 1
content-managed-page discovery with the public _ak token, Phase 2 event-page
JSON per event, httpx transport — bypasses PerimeterX entirely). CoreProp
skipped MATCH_RESULT / BOTH_TEAMS_TO_SCORE / totals as "game-level noise";
here they are the whole point.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from .common import BookOdds, american_to_decimal

logger = logging.getLogger(__name__)

FD_AK_TOKEN = "FhMFpcPWXMeyZxOx"                # public web token (from CoreProp)
SBAPI = "https://sbapi.nj.sportsbook.fanduel.com/api"
FD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}
PAGE_IDS = ("world-cup", "soccer")              # tried in order for discovery

_TOTAL_RE = re.compile(r"(over|under)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def scrape_fanduel_soccer(home_name: str, away_name: str) -> List[BookOdds]:
    """Discover the event for one fixture and pull its match markets."""
    try:
        import httpx
        with httpx.Client(timeout=15) as client:
            eid, ev_home, ev_away, kickoff = _find_event(client, home_name, away_name)
            if eid is None:
                logger.info("FanDuel: no event found for %s vs %s", home_name, away_name)
                return []
            r = client.get(f"{SBAPI}/event-page",
                           params={"_ak": FD_AK_TOKEN, "eventId": eid},
                           headers=FD_HEADERS)
            if r.status_code != 200:
                logger.warning("FanDuel event-page HTTP %d", r.status_code)
                return []
            game = parse_fanduel_event(r.json(), ev_home, ev_away, kickoff)
            return [game] if game else []
    except Exception as e:                       # noqa: BLE001
        logger.warning("FanDuel scrape failed: %s", e)
        return []


def _find_event(client, home_name: str, away_name: str):
    from .common import teams_match
    for page_id in PAGE_IDS:
        try:
            r = client.get(f"{SBAPI}/content-managed-page",
                           params={"page": "CUSTOM", "customPageId": page_id,
                                   "_ak": FD_AK_TOKEN},
                           headers=FD_HEADERS)
            if r.status_code != 200:
                continue
            events = r.json().get("attachments", {}).get("events", {})
            for eid, ev in events.items():
                name = ev.get("name", "")
                for sep in (" v ", " vs ", " @ "):
                    if sep in name:
                        eh, ea = name.split(sep, 1)
                        if teams_match(eh, ea, home_name, away_name):
                            return eid, eh.strip(), ea.strip(), ev.get("openDate", "")
        except Exception as e:                   # noqa: BLE001
            logger.debug("FanDuel discovery %s: %s", page_id, e)
    return None, None, None, None


def parse_fanduel_event(payload: dict, home_name: str, away_name: str,
                        kickoff: str = "") -> Optional[BookOdds]:
    """Pure parser for an event-page payload — unit-testable offline."""
    markets = payload.get("attachments", {}).get("markets", {})
    game = BookOdds(book="fanduel", home_name=home_name, away_name=away_name,
                    kickoff=kickoff)
    for mkt in markets.values():
        mtype = str(mkt.get("marketType", "")).upper()
        runners = mkt.get("runners", [])
        if mtype == "MATCH_RESULT" or mtype == "MONEY_LINE_3_WAY":
            h2h: Dict[str, float] = {}
            for run in runners:
                dec = _runner_decimal(run)
                name = str(run.get("runnerName", "")).lower()
                if dec is None:
                    continue
                if name == "draw":
                    h2h["draw"] = dec
                elif name in home_name.lower() or home_name.lower() in name:
                    h2h["home"] = dec
                elif name in away_name.lower() or away_name.lower() in name:
                    h2h["away"] = dec
            if {"home", "draw", "away"} <= set(h2h):
                game.h2h = h2h
        elif mtype == "BOTH_TEAMS_TO_SCORE":
            yes = no = None
            for run in runners:
                dec = _runner_decimal(run)
                name = str(run.get("runnerName", "")).lower()
                if name == "yes":
                    yes = dec
                elif name == "no":
                    no = dec
            if yes and no:
                game.btts = (yes, no)
        elif "TOTAL" in mtype and "TEAM" not in mtype and "1ST" not in mtype \
                and "FIRST" not in mtype:
            by_line: Dict[float, dict] = {}
            for run in runners:
                m = _TOTAL_RE.search(str(run.get("runnerName", "")))
                dec = _runner_decimal(run)
                line = run.get("handicap")
                if m and dec is not None:
                    side = m.group(1).lower()
                    line = float(line) if line is not None else float(m.group(2))
                    by_line.setdefault(line, {})[side] = dec
            for line, sides in by_line.items():
                if {"over", "under"} <= set(sides):
                    game.totals[line] = (sides["over"], sides["under"])
    if game.h2h or game.totals or game.btts:
        return game
    return None


def _runner_decimal(runner: dict) -> Optional[float]:
    odds = runner.get("winRunnerOdds", {})
    dec = odds.get("trueOdds", {}).get("decimalOdds", {}).get("decimalOdds")
    if dec:
        return float(dec)
    american = odds.get("americanDisplayOdds", {}).get("americanOdds")
    return american_to_decimal(american)
