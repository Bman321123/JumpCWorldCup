"""Shared types for the sportsbook scrapers (refactored from friedman-max/CoreProp).

CoreProp's scrapers targeted US player props and deliberately skipped soccer
game-level markets; these are the same transport patterns (curl_cffi Chrome
impersonation, the same endpoints) pointed at exactly those markets:
1X2, totals (incl. first half where offered), and BTTS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

CHROME_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


@dataclass
class BookOdds:
    """One book's soccer match markets, decimal odds throughout."""
    book: str
    home_name: str
    away_name: str
    kickoff: str = ""                                        # ISO when known
    h2h: Optional[Dict[str, float]] = None                   # {home, draw, away}
    totals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    h1_totals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    btts: Optional[Tuple[float, float]] = None               # (yes, no)
    corner_totals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    h1_corner_totals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    booking_totals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    h1_booking_totals: Dict[float, Tuple[float, float]] = field(default_factory=dict)


def american_to_decimal(american) -> Optional[float]:
    """Accepts int or display strings like '+150' / '−110' (incl. unicode minus)."""
    if american is None:
        return None
    if isinstance(american, str):
        american = american.replace("−", "-").replace("+", "").strip()
        if not american:
            return None
        try:
            american = int(american)
        except ValueError:
            return None
    if american == 0:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def _as_aliases(x) -> list:
    return [x.lower()] if isinstance(x, str) else [a.lower() for a in x]


def alias_hit(event_name: str, aliases) -> bool:
    """True if any alias matches the event team name (books abbreviate:
    Pinnacle says 'USA', FanDuel 'United States')."""
    n = event_name.lower().strip()
    for a in _as_aliases(aliases):
        if a in n or n in a:
            return True
    return False


def teams_match(event_home: str, event_away: str, home, away) -> bool:
    """Order-insensitive match; home/away may be a name or a list of aliases."""
    return ((alias_hit(event_home, home) and alias_hit(event_away, away))
            or (alias_hit(event_home, away) and alias_hit(event_away, home)))
