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
    "WC2018": ("fifa.world", "20180614-20180715"),
    "WC2022": ("fifa.world", "20221120-20221218"),
    "EURO2024": ("uefa.euro", "20240614-20240714"),
    "COPA2024": ("conmebol.america", "20240620-20240715"),
    "WC2026": ("fifa.world", f"20260611-{date.today().strftime('%Y%m%d')}"),
}
STAT_MAP = {"wonCorners": "corners", "yellowCards": "yellows",
            "redCards": "reds", "offsides": "offsides",
            "shotsOnTarget": "sot", "foulsCommitted": "fouls"}


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
        if out_path.exists():
            saved += 1
            continue
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
        if len(teams) == 2:
            out_path.write_text(json.dumps(
                {"source": "espn", "event": ev.get("name"),
                 "date": ev.get("date"), "teams": teams}, indent=1))
            saved += 1
    print(f"{comp}: {saved} match stat files in {PARSED_DIR}")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="WC2018,WC2022,EURO2024,COPA2024,WC2026")
    args = ap.parse_args()
    for comp in args.comps.split(","):
        comp = comp.strip()
        if comp in COMPS:
            league, dates = COMPS[comp]
            ingest(comp, league, dates)
    aggregate()
