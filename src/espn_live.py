"""Best-effort live lookups from ESPN for an upcoming WC2026 fixture.

Pre-match data availability is not guaranteed (ESPN often lists the referee only
once it is officially assigned). Every function degrades to None so the pipeline
simply proceeds without the signal.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"


def _teams_match(name: str, home: str, away: str) -> bool:
    n = name.lower()
    return (home.lower() in n or n in home.lower()) and \
           (away.lower() in n or n in away.lower())


def match_referee(home_name: str, away_name: str,
                  window_days: int = 3) -> Optional[str]:
    """Return the assigned referee for an upcoming fixture, or None."""
    try:
        import requests
        today = date.today().strftime("%Y%m%d")
        end = today
        board = requests.get(f"{BASE}/scoreboard",
                             params={"dates": f"{today}-{end}", "limit": 100},
                             timeout=12, headers={"User-Agent": "Mozilla/5.0"}).json()
        for ev in board.get("events", []):
            if not _teams_match(ev.get("name", ""), home_name, away_name):
                continue
            summary = requests.get(f"{BASE}/summary", params={"event": ev["id"]},
                                   timeout=12,
                                   headers={"User-Agent": "Mozilla/5.0"}).json()
            officials = summary.get("gameInfo", {}).get("officials", [])
            if officials:
                main = min(officials, key=lambda o: o.get("order", 99))
                return main.get("fullName") or main.get("displayName")
        return None
    except Exception as e:                       # noqa: BLE001
        logger.warning("Referee lookup failed: %s", e)
        return None
