"""Scraped-odds aggregator — drop-in replacement for The Odds API client.

Hierarchy per market (PRD §3.1): Pinnacle is the sharp anchor — if it quotes
the market, use it alone. Otherwise take the median devigged probability
across DraftKings / FanDuel. Everything is Shin-devigged; output shape is
identical to OddsClient.market_probs so the orchestrator can't tell the
difference. All raw book odds are cached to SQLite for CLV analysis.

Live smoke test:  python -m src.scrapers.aggregator MEX RSA
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from ..shin_devigger import decimal_to_implied, shin_devig
from .common import BookOdds, teams_match
from .draftkings import scrape_draftkings_soccer
from .fanduel import scrape_fanduel_soccer
from .pinnacle import scrape_pinnacle_soccer

logger = logging.getLogger(__name__)

CACHE_TTL_S = 300
CLOSING_WINDOW_MIN = 30


class ScrapedOddsClient:
    def __init__(self, db_path: Optional[str] = None,
                 team_names: Optional[Dict[str, str]] = None,
                 config_path: Optional[str] = None,
                 team_aliases: Optional[Dict[str, list]] = None):
        self.db_path = db_path
        self.team_names = team_names or {}
        self.team_aliases = team_aliases or {}
        self.config = {}
        if config_path:
            try:
                with open(config_path) as f:
                    self.config = json.load(f)
            except FileNotFoundError:
                pass

    def _aliases(self, code: str) -> list:
        name = self.team_names.get(code, code)
        return self.team_aliases.get(code) or [name, code]

    def market_probs(self, home_code: str, away_code: str,
                     kickoff_iso: Optional[str] = None) -> Optional[dict]:
        home_name = self.team_names.get(home_code, home_code)
        away_name = self.team_names.get(away_code, away_code)
        home_al, away_al = self._aliases(home_code), self._aliases(away_code)

        cache_key = f"scrape:{home_code}:{away_code}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        books = self._collect(home_name, away_name, home_al, away_al)
        if not books:
            return None
        out = aggregate(books, home_al, away_al)
        if out is None:
            return None
        out["is_closing"] = self._is_closing(kickoff_iso or out.get("kickoff"))
        self._cache_put(cache_key, out)
        return out

    def _collect(self, home_name: str, away_name: str,
                 home_al: list, away_al: list) -> List[BookOdds]:
        books: List[BookOdds] = []
        pin_ids = self.config.get("pinnacle_league_ids") or None
        for fetch, label in (
                (lambda: scrape_pinnacle_soccer(pin_ids), "pinnacle"),
                (lambda: scrape_draftkings_soccer(
                    self.config.get("dk_event_group_id")), "draftkings"),
                (lambda: scrape_fanduel_soccer(home_name, away_name), "fanduel")):
            try:
                books.extend(fetch())
            except Exception as e:               # noqa: BLE001
                logger.warning("%s scraper failed: %s", label, e)
        matched = [b for b in books
                   if teams_match(b.home_name, b.away_name, home_al, away_al)]
        logger.info("Scrapers: %d books matched %s vs %s (%s)",
                    len(matched), home_name, away_name,
                    [b.book for b in matched])
        return matched


    @staticmethod
    def _is_closing(kickoff_iso: Optional[str]) -> bool:
        if not kickoff_iso:
            return False
        try:
            ko = datetime.fromisoformat(str(kickoff_iso).replace("Z", "+00:00"))
            mins = (ko - datetime.now(timezone.utc)).total_seconds() / 60.0
            return 0 <= mins <= CLOSING_WINDOW_MIN
        except ValueError:
            return False

    def _cache_get(self, key: str):
        if not self.db_path:
            return None
        try:
            con = sqlite3.connect(self.db_path)
            row = con.execute("SELECT raw_json, fetched_at FROM market_cache "
                              "WHERE cache_key = ?", (key,)).fetchone()
            con.close()
            if row and time.time() - row[1] < CACHE_TTL_S:
                return json.loads(row[0])
        except sqlite3.Error:
            pass
        return None

    def _cache_put(self, key: str, data) -> None:
        if not self.db_path:
            return
        try:
            con = sqlite3.connect(self.db_path)
            con.execute("INSERT OR REPLACE INTO market_cache "
                        "(cache_key, raw_json, fetched_at) VALUES (?,?,?)",
                        (key, json.dumps(data), time.time()))
            con.commit()
            con.close()
        except sqlite3.Error as e:
            logger.warning("Cache write failed: %s", e)


def aggregate(books: List[BookOdds], home, away) -> Optional[dict]:
    """Pure aggregation — unit-testable offline. Sharp-anchor then median.
    home/away may be a name string or a list of aliases."""
    if not books:
        return None
    from .common import alias_hit
    ordered = sorted(books, key=lambda b: 0 if b.book == "pinnacle" else 1)

    def flipped(b: BookOdds) -> bool:
        return alias_hit(b.home_name, away) and alias_hit(b.away_name, home)

    h2h_probs = []
    for b in ordered:
        if not b.h2h:
            continue
        trio = [b.h2h["home"], b.h2h["draw"], b.h2h["away"]]
        if flipped(b):
            trio = [trio[2], trio[1], trio[0]]
        try:
            p = shin_devig(decimal_to_implied(trio))
            h2h_probs.append((b.book, p))
        except ValueError:
            continue

    out: dict = {"h2h": None, "totals": {}, "h1_totals": {}, "btts": None,
                 "corner_totals": {}, "h1_corner_totals": {},
                 "booking_totals": {}, "h1_booking_totals": {},
                 "books": [b.book for b in ordered],
                 "kickoff": next((b.kickoff for b in ordered if b.kickoff), None)}

    if h2h_probs:
        if h2h_probs[0][0] == "pinnacle":
            p = h2h_probs[0][1]
        else:
            p = np.median(np.array([x for _, x in h2h_probs]), axis=0)
            p = p / p.sum()
        out["h2h"] = {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}

    for field in ("totals", "h1_totals", "corner_totals", "h1_corner_totals",
                  "booking_totals", "h1_booking_totals"):
        per_line: Dict[float, List[float]] = {}
        for b in ordered:
            for line, (over, under) in getattr(b, field).items():
                try:
                    p_over = float(shin_devig(decimal_to_implied([over, under]))[0])
                except ValueError:
                    continue
                per_line.setdefault(line, [])
                if b.book == "pinnacle":
                    per_line[line] = [p_over]            # sharp anchor wins
                elif len(per_line[line]) == 0 or not _has_pinnacle(ordered, b, line, field):
                    per_line[line].append(p_over)
        out[field] = {line: float(np.median(ps)) for line, ps in per_line.items() if ps}

    btts_probs = []
    for b in ordered:
        if not b.btts:
            continue
        try:
            p_yes = float(shin_devig(decimal_to_implied(list(b.btts)))[0])
        except ValueError:
            continue
        if b.book == "pinnacle":
            btts_probs = [p_yes]
            break
        btts_probs.append(p_yes)
    if btts_probs:
        out["btts"] = float(np.median(btts_probs))

    if out["h2h"] or out["totals"] or out["btts"]:
        return out
    return None


def _has_pinnacle(books: List[BookOdds], current: BookOdds, line: float,
                  field: str) -> bool:
    return any(b.book == "pinnacle" and line in getattr(b, field)
               for b in books if b is not current)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = Path(__file__).resolve().parents[2]
    with open(root / "config" / "groups.json") as f:
        names = {c: t["name"] for c, t in json.load(f)["teams"].items()}
    home, away = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("MEX", "RSA")
    client = ScrapedOddsClient(team_names=names,
                               config_path=str(root / "config" / "scrapers.json"))
    result = client.market_probs(home, away)
    print(json.dumps(result, indent=1) if result else "No markets found.")
