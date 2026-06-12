"""Sharp-book odds via The Odds API v4, with SQLite caching and Shin devigging
(PRD v2.2 §6.2). Prefers sharp books (Pinnacle, exchanges) and falls back to a
median-of-books consensus. Everything degrades gracefully to None -> pure model.

Quota note: one request costs markets x regions credits. Cache aggressively.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .shin_devigger import ShinDevigger, decimal_to_implied, shin_devig

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"          # verify the 2026 key (PRD §0.4)
MARKETS = "h2h,totals,btts"
REGION = "eu"
SHARP_BOOKS = ["pinnacle", "betfair_ex_eu", "smarkets", "matchbook", "marathonbet"]
CACHE_TTL_MATCHDAY_S = 300
CACHE_TTL_DEFAULT_S = 3600
CLOSING_WINDOW_MIN = 30


class OddsClient:
    def __init__(self, api_key: Optional[str] = None, db_path: Optional[str] = None,
                 team_names: Optional[Dict[str, str]] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        self.db_path = db_path
        self.team_names = team_names or {}       # FIFA code -> The Odds API name
        self.devigger = ShinDevigger()

    # ----- fetching -----

    def fetch_events(self, ttl_s: int = CACHE_TTL_DEFAULT_S) -> Optional[list]:
        if not self.api_key:
            logger.info("No ODDS_API_KEY set; market layer disabled.")
            return None
        cached = self._cache_get("events", ttl_s)
        if cached is not None:
            return cached
        try:
            import requests
            r = requests.get(
                f"{BASE_URL}/sports/{SPORT_KEY}/odds",
                params={"apiKey": self.api_key, "regions": REGION,
                        "markets": MARKETS, "oddsFormat": "decimal",
                        "dateFormat": "iso"},
                timeout=10)
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining")
            logger.info("Odds API fetched; credits remaining: %s", remaining)
            data = r.json()
            self._cache_put("events", data)
            return data
        except Exception as e:                   # noqa: BLE001
            logger.warning("Odds fetch failed: %s", e)
            return None

    def market_probs(self, home_code: str, away_code: str,
                     kickoff_iso: Optional[str] = None) -> Optional[dict]:
        """Returns {'h2h': {'home','draw','away'}, 'totals': {line: p_over},
        'btts': p_yes, 'is_closing': bool} or None."""
        events = self.fetch_events()
        if not events:
            return None
        home_name = self.team_names.get(home_code, home_code)
        away_name = self.team_names.get(away_code, away_code)
        event = self._match_event(events, home_name, away_name)
        if event is None:
            logger.info("No odds event found for %s vs %s", home_name, away_name)
            return None
        is_closing = self._is_closing(event.get("commence_time"))
        out: dict = {"is_closing": is_closing, "h2h": None, "totals": {}, "btts": None}

        h2h = self._collect(event, "h2h")
        if h2h:
            probs = self._devig_h2h(h2h, event["home_team"], event["away_team"])
            if probs is not None:
                out["h2h"] = probs
        totals = self._collect(event, "totals")
        for line, pair in self._pair_totals(totals).items():
            try:
                p = shin_devig(decimal_to_implied(pair))
                out["totals"][line] = float(p[0])     # P(over)
            except ValueError:
                continue
        btts = self._collect(event, "btts")
        pair = self._pair_yes_no(btts)
        if pair:
            try:
                out["btts"] = float(shin_devig(decimal_to_implied(pair))[0])
            except ValueError:
                pass
        return out

    # ----- internals -----

    @staticmethod
    def _is_closing(commence_iso: Optional[str]) -> bool:
        if not commence_iso:
            return False
        try:
            ko = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            mins = (ko - datetime.now(timezone.utc)).total_seconds() / 60.0
            return 0 <= mins <= CLOSING_WINDOW_MIN
        except ValueError:
            return False

    @staticmethod
    def _match_event(events: list, home_name: str, away_name: str) -> Optional[dict]:
        hn, an = home_name.lower(), away_name.lower()
        for ev in events:
            eh, ea = ev.get("home_team", "").lower(), ev.get("away_team", "").lower()
            if (hn in eh or eh in hn) and (an in ea or ea in an):
                return ev
            if (hn in ea or ea in hn) and (an in eh or eh in an):
                return ev                          # listed order flipped
        return None

    @staticmethod
    def _collect(event: dict, market_key: str) -> List[dict]:
        """[{book, outcomes:[{name, price, point}]}] sorted sharp-first."""
        rows = []
        for bm in event.get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk.get("key") == market_key:
                    rows.append({"book": bm.get("key"), "outcomes": mk.get("outcomes", [])})
        rows.sort(key=lambda r: SHARP_BOOKS.index(r["book"])
                  if r["book"] in SHARP_BOOKS else 99)
        return rows

    def _devig_h2h(self, rows: List[dict], home_name: str, away_name: str
                   ) -> Optional[Dict[str, float]]:
        per_book = []
        for row in rows:
            prices = {}
            for o in row["outcomes"]:
                name = o.get("name", "")
                if name == home_name:
                    prices["home"] = o["price"]
                elif name == away_name:
                    prices["away"] = o["price"]
                elif name.lower() == "draw":
                    prices["draw"] = o["price"]
            if {"home", "draw", "away"} <= set(prices):
                try:
                    p = shin_devig(decimal_to_implied(
                        [prices["home"], prices["draw"], prices["away"]]))
                    per_book.append((row["book"], p))
                except ValueError:
                    continue
        if not per_book:
            return None
        for book, p in per_book:
            if book in SHARP_BOOKS:
                return {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}
        med = np.median(np.array([p for _, p in per_book]), axis=0)
        med = med / med.sum()
        return {"home": float(med[0]), "draw": float(med[1]), "away": float(med[2])}

    @staticmethod
    def _pair_totals(rows: List[dict]) -> Dict[float, List[float]]:
        """{line: [over_price, under_price]} from the sharpest book carrying both."""
        out: Dict[float, List[float]] = {}
        for row in rows:
            by_line: Dict[float, Dict[str, float]] = {}
            for o in row["outcomes"]:
                pt = o.get("point")
                if pt is None:
                    continue
                by_line.setdefault(float(pt), {})[o.get("name", "").lower()] = o["price"]
            for line, sides in by_line.items():
                if line not in out and {"over", "under"} <= set(sides):
                    out[line] = [sides["over"], sides["under"]]
        return out

    @staticmethod
    def _pair_yes_no(rows: List[dict]) -> Optional[List[float]]:
        for row in rows:
            sides = {o.get("name", "").lower(): o["price"] for o in row["outcomes"]}
            if {"yes", "no"} <= set(sides):
                return [sides["yes"], sides["no"]]
        return None

    # ----- sqlite cache -----

    def _cache_get(self, key: str, ttl_s: int):
        if not self.db_path:
            return None
        try:
            con = sqlite3.connect(self.db_path)
            row = con.execute(
                "SELECT raw_json, fetched_at FROM market_cache WHERE cache_key = ?",
                (key,)).fetchone()
            con.close()
            if row and time.time() - row[1] < ttl_s:
                return json.loads(row[0])
        except sqlite3.Error:
            pass
        return None

    def _cache_put(self, key: str, data) -> None:
        if not self.db_path:
            return
        try:
            con = sqlite3.connect(self.db_path)
            con.execute(
                "INSERT OR REPLACE INTO market_cache (cache_key, raw_json, fetched_at) "
                "VALUES (?,?,?)", (key, json.dumps(data), time.time()))
            con.commit()
            con.close()
        except sqlite3.Error as e:
            logger.warning("Cache write failed: %s", e)
