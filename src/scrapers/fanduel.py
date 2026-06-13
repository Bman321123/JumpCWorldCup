"""FanDuel scraper — WC soccer match markets via the two-step sbapi flow.

Confirmed live (2026-06-13): customPageId='fifa-world-cup' lists ~70 events;
per-event markets are paginated across tabs (popular/goals/corners/shots). FD
carries 1X2, full + first-half totals, BTTS, team totals, corners, AND
per-player shot markets — a deep second book for the line-comparison layer and
a market anchor for player props.

Strategy from CoreProp/scrapers/fanduel.py (httpx, public _ak token, two-step
discovery -> event-page JSON) retargeted to soccer game + player markets.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from .common import BookOdds, alias_hit, american_to_decimal

logger = logging.getLogger(__name__)

FD_AK_TOKEN = "FhMFpcPWXMeyZxOx"
SBAPI = "https://sbapi.nj.sportsbook.fanduel.com/api"
PAGE_ID = "fifa-world-cup"
TABS = ["popular", "goals", "corners", "shots"]
FD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}
_TOTAL_RE = re.compile(r"(over|under)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_PLAYER_SHOTS_RE = re.compile(r"PLAYER_TO_HAVE_(\d+)_OR_MORE_SHOTS(_IN_1ST_HALF)?")


def scrape_fanduel_soccer(home_aliases, away_aliases) -> List[BookOdds]:
    """home_aliases/away_aliases: name or list of aliases for matching."""
    try:
        import httpx
        with httpx.Client(timeout=15) as client:
            nav = client.get(f"{SBAPI}/content-managed-page",
                             params={"page": "CUSTOM", "customPageId": PAGE_ID,
                                     "_ak": FD_AK_TOKEN}, headers=FD_HEADERS)
            if nav.status_code != 200:
                logger.warning("FanDuel nav HTTP %d", nav.status_code)
                return []
            events = nav.json().get("attachments", {}).get("events", {})
            eid = ev_home = ev_away = None
            for k, ev in events.items():
                name = ev.get("name", "")
                for sep in (" v ", " vs ", " @ "):
                    if sep in name:
                        eh, ea = name.split(sep, 1)
                        if alias_hit(eh, home_aliases) and alias_hit(ea, away_aliases):
                            eid, ev_home, ev_away = k, eh.strip(), ea.strip()
                        elif alias_hit(eh, away_aliases) and alias_hit(ea, home_aliases):
                            eid, ev_home, ev_away = k, eh.strip(), ea.strip()
            if eid is None:
                logger.info("FanDuel: no event for given fixture")
                return []
            markets: Dict[str, dict] = {}
            for tab in TABS:
                r = client.get(f"{SBAPI}/event-page",
                               params={"_ak": FD_AK_TOKEN, "eventId": eid, "tab": tab},
                               headers=FD_HEADERS)
                if r.status_code == 200:
                    markets.update(r.json().get("attachments", {}).get("markets", {}))
            game = parse_fanduel_markets(markets, ev_home, ev_away)
            return [game] if game else []
    except Exception as e:                       # noqa: BLE001
        logger.warning("FanDuel scrape failed: %s", e)
        return []


def _runner_decimal(runner: dict) -> Optional[float]:
    o = runner.get("winRunnerOdds", {})
    dec = o.get("trueOdds", {}).get("decimalOdds", {}).get("decimalOdds")
    if dec:
        return float(dec)
    return american_to_decimal(o.get("americanDisplayOdds", {}).get("americanOdds"))


def parse_fanduel_markets(markets: dict, home_name: str, away_name: str
                          ) -> Optional[BookOdds]:
    """Pure parser over a merged {marketId: market} dict — unit-testable."""
    g = BookOdds(book="fanduel", home_name=home_name, away_name=away_name)
    for m in markets.values():
        mt = str(m.get("marketType", "")).upper()
        runners = m.get("runners", [])

        if mt == "WIN-DRAW-WIN":
            h2h = {}
            for ru in runners:
                dec = _runner_decimal(ru)
                nm = str(ru.get("runnerName", "")).strip().lower()
                if dec is None:
                    continue
                if nm == "draw" or nm == "tie":
                    h2h["draw"] = dec
                elif alias_hit(nm, home_name):
                    h2h["home"] = dec
                elif alias_hit(nm, away_name):
                    h2h["away"] = dec
            if {"home", "draw", "away"} <= set(h2h):
                g.h2h = h2h

        elif mt == "BOTH_TEAMS_TO_SCORE":
            yes = next((_runner_decimal(r) for r in runners
                        if str(r.get("runnerName", "")).lower() == "yes"), None)
            no = next((_runner_decimal(r) for r in runners
                       if str(r.get("runnerName", "")).lower() == "no"), None)
            if yes and no:
                g.btts = (yes, no)

        elif re.fullmatch(r"OVER_UNDER_\d+", mt):
            _collect_total(g.totals, runners)
        elif mt.startswith("1ST_HALF_OVER/UNDER") and "GOAL" in mt:
            _collect_total(g.h1_totals, runners)
        elif "CORNER" in mt and ("OVER" in mt or "TOTAL" in mt):
            _collect_total(g.corner_totals, runners)

        else:
            sm = _PLAYER_SHOTS_RE.search(mt)
            if sm:
                thr = float(sm.group(1))
                half = "H1" if sm.group(2) else "FULL"
                for ru in runners:
                    dec = _runner_decimal(ru)
                    nm = ru.get("runnerName")
                    if dec and nm:
                        # one-sided yes price; shade ~5% overround off implied
                        p = min((1.0 / dec) / 1.05, 0.97)
                        g.player_shots.setdefault(nm, {})[f"{int(thr)}+_{half}"] = p
    if g.h2h or g.totals or g.btts or g.corner_totals or g.player_shots:
        return g
    return None


def _collect_total(target: dict, runners: list) -> None:
    by_line: Dict[float, dict] = {}
    for ru in runners:
        m = _TOTAL_RE.search(str(ru.get("runnerName", "")))
        dec = _runner_decimal(ru)
        if not m or dec is None:
            continue
        line = float(ru.get("handicap") or 0) or float(m.group(2))
        by_line.setdefault(line, {})[m.group(1).lower()] = dec
    for line, sides in by_line.items():
        if {"over", "under"} <= set(sides):
            target[line] = (sides["over"], sides["under"])
