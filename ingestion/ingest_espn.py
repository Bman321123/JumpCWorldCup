"""Per-team micro stats from ESPN's public soccer API (ROADMAP 1.1).

FBref sits behind Cloudflare; ESPN's site API serves the same match stats
(corners, cards, offsides, shots on target, fouls) as clean JSON with no
auth. Output lands in data/match_stats/ in the same shape the aggregator
(run_fbref_scrape.aggregate) consumes.

  python ingestion/ingest_espn.py                # recent majors + WC2026 so far
  python ingestion/ingest_espn.py --comps WC2026 # nightly in-tournament refresh
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingestion.run_fbref_scrape import PARSED_DIR, aggregate  # noqa: E402

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
DELAY_S = 0.5

COMPS = {
    # major finals
    "WC2014": ("fifa.world", "20140612-20140713"),
    "WC2018": ("fifa.world", "20180614-20180715"),
    "WC2022": ("fifa.world", "20221120-20221218"),
    "EURO2020": ("uefa.euro", "20210611-20210711"),
    "EURO2024": ("uefa.euro", "20240614-20240714"),
    "COPA2021": ("conmebol.america", "20210613-20210710"),
    "COPA2024": ("conmebol.america", "20240620-20240715"),
    # continental cups — these are the African/Asian/CONCACAF WC teams' real games
    "AFCON2023": ("caf.nations", "20240113-20240211"),
    "ASIANCUP2023": ("afc.asian.cup", "20240112-20240210"),
    "GOLDCUP2023": ("concacaf.gold", "20230624-20230716"),
    "GOLDCUP2025": ("concacaf.gold", "20250614-20250706"),
    "NATIONS2024": ("uefa.nations", "20240905-20241119"),
    "NATIONS2025": ("uefa.nations", "20250320-20250608"),
    # WC2026 qualifiers — the most recent form for the actual tournament teams
    "WCQ_UEFA": ("fifa.worldq.uefa", "20250321-20251118"),
    "WCQ_CAF": ("fifa.worldq.caf", "20250301-20251118"),
    "WCQ_AFC": ("fifa.worldq.afc", "20241101-20251118"),
    "WCQ_CONCACAF": ("fifa.worldq.concacaf", "20250601-20251118"),
    # the tournament itself, so far
    "WC2026": ("fifa.world", f"20260611-{date.today().strftime('%Y%m%d')}"),
}
# every comp except WC2026 is historical/static; this is the default broad backfill
ALL_BACKFILL = ",".join(c for c in COMPS if c != "WC2026")
STAT_MAP = {"wonCorners": "corners", "yellowCards": "yellows",
            "redCards": "reds", "offsides": "offsides",
            "shotsOnTarget": "sot", "foulsCommitted": "fouls"}
PLAYER_STATS = ("totalShots", "shotsOnTarget", "totalGoals", "goalAssists",
                "foulsCommitted", "offsides", "yellowCards", "redCards")


def _get(url: str, params: dict | None = None) -> dict:
    time.sleep(DELAY_S)
    r = requests.get(url, params=params, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def ingest(comp: str, league: str, dates: str) -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    board = _get(f"{BASE}/{league}/scoreboard", {"dates": dates, "limit": 200})
    events = board.get("events", [])
    print(f"{comp}: {len(events)} events")
    saved = 0
    for ev in events:
        out_path = PARSED_DIR / f"espn_{comp}_{ev['id']}.json"
        # re-fetch if an old file lacks the richer player/referee data
        if out_path.exists():
            try:
                if "players" in json.loads(out_path.read_text()):
                    saved += 1
                    continue
            except (ValueError, OSError):
                pass
        status = ev.get("status", {}).get("type", {}).get("state")
        if status != "post":                    # only completed matches
            continue
        try:
            summary = _get(f"{BASE}/{league}/summary", {"event": ev["id"]})
        except requests.RequestException as e:
            print(f"  {ev['id']}: {e}")
            continue

        teams = {}
        for t in summary.get("boxscore", {}).get("teams", []):
            name = t.get("team", {}).get("displayName")
            stats = {}
            for s in t.get("statistics", []):
                key = STAT_MAP.get(s.get("name"))
                if key:
                    try:
                        stats[key] = int(float(s.get("displayValue", "")))
                    except (ValueError, TypeError):
                        pass
            if name and stats:
                teams[name] = stats

        players = _parse_players(summary)
        referee = _parse_referee(summary)
        if len(teams) == 2:
            out_path.write_text(json.dumps(
                {"source": "espn", "event": ev.get("name"),
                 "date": ev.get("date"), "teams": teams,
                 "players": players, "referee": referee}, indent=1))
            saved += 1
    print(f"{comp}: {saved} match stat files in {PARSED_DIR}")
    return saved


def _parse_players(summary: dict) -> list:
    """Per-player line: team, name, position, starter, minutes proxy, stats."""
    out = []
    for r in summary.get("rosters", []):
        team = r.get("team", {}).get("displayName")
        for p in r.get("roster", []):
            ath = p.get("athlete", {}).get("displayName")
            if not ath:
                continue
            stats = {s["name"]: s.get("value") for s in p.get("stats", [])}
            if not stats:
                continue
            starter = bool(p.get("starter"))
            subbed_out = p.get("subbedOut") not in (None, False)
            subbed_in = p.get("subbedIn") not in (None, False)
            if not (starter or subbed_in):
                continue                          # unused bench player
            minutes = 90 if starter and not subbed_out else 65 if starter else 25
            row = {"team": team, "name": ath,
                   "position": p.get("position", {}).get("abbreviation", "M"),
                   "starter": starter, "minutes": minutes}
            for k in PLAYER_STATS:
                v = stats.get(k)
                row[k] = int(v) if isinstance(v, (int, float)) else 0
            out.append(row)
    return out


def _parse_referee(summary: dict) -> str | None:
    officials = summary.get("gameInfo", {}).get("officials", [])
    if not officials:
        return None
    main = min(officials, key=lambda o: o.get("order", 99))
    return main.get("fullName") or main.get("displayName")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default=ALL_BACKFILL + ",WC2026",
                    help="comma list of COMPS keys, or 'ALL' for everything")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="skip the micro-rate merge (do it once at the end of a batch)")
    args = ap.parse_args()
    comps = list(COMPS) if args.comps.strip().upper() == "ALL" else \
        [c.strip() for c in args.comps.split(",")]
    total = 0
    for comp in comps:
        if comp not in COMPS:
            print(f"unknown comp {comp!r} — skipping")
            continue
        league, dates = COMPS[comp]
        try:                                          # per-comp isolation
            total += ingest(comp, league, dates)
        except Exception as e:                        # noqa: BLE001
            print(f"{comp}: FAILED ({e}) — continuing")
    print(f"total match-stat files: {total}")
    if not args.no_aggregate:
        aggregate()
