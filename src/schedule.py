"""WC2026 round inference from a match's UTC kickoff time.

The platform gives each match an `opening_time` (kickoff) in UTC. Every 2026 venue
is in North America (UTC-4..-7), so a late-evening local kickoff lands *after*
midnight UTC the next day — e.g. the final group games kick off 2026-06-28 02:00Z.
A naive date-string cutoff (`date <= "2026-06-27"`) therefore MISCLASSIFIES those
spillover games as knockouts and skips them. (It also skipped real knockouts.)

Fix: shift the UTC instant back 6h before taking the date. That maps every kickoff
onto its local match day — no WC kickoff falls in the 06:00-16:00 UTC dead zone, so
the 6h shift never crosses a match from one day's window into another's. Then bucket
that date into the official FIFA round windows.

Only the GROUP-vs-KNOCKOUT split changes the submitted probability (knockout "win"
= advance, ET+pens via StatsEngine.advance_prob); the finer knockout sub-round only
sets the reporting `round_weight`. So this is robust where it matters: any post-group
date is a knockout and is priced, never skipped. `None` is returned only for an
unparseable/pre-tournament time (the daemon skips those, which is correct).

Source: FIFA 2026 schedule — group Jun 11-27, R32 Jun 28-Jul 3, R16 Jul 4-7,
QF Jul 9-11, SF Jul 14-15, 3rd Jul 18, Final Jul 19.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# (round_key, first_local_date, last_local_date). End dates are stretched across the
# rest-day gaps between rounds so the windows are CONTIGUOUS from Jun 28 on — every
# post-group match date lands in exactly one knockout window and nothing is skipped.
# (No matches are played on the absorbed rest days, so this only affects labelling.)
_WINDOWS = (
    ("group",        "2026-06-11", "2026-06-27"),
    ("round_of_32",  "2026-06-28", "2026-07-03"),
    ("round_of_16",  "2026-07-04", "2026-07-08"),   # absorbs Jul 8 rest day
    ("quarterfinal", "2026-07-09", "2026-07-13"),   # absorbs Jul 12-13 rest days
    ("semifinal",    "2026-07-14", "2026-07-17"),   # absorbs Jul 16-17 rest days
    ("third_place",  "2026-07-18", "2026-07-18"),
    ("final",        "2026-07-19", "2026-08-31"),    # open-ended tail
)
_TOURNAMENT_START = "2026-06-11"
_GROUP_END = "2026-06-27"
_SHIFT_H = 6


def local_match_date(opening_time: Optional[str]) -> Optional[str]:
    """UTC kickoff -> the match's local calendar date (ISO), via the 6h back-shift."""
    if not isinstance(opening_time, str):
        return None
    try:
        ko = datetime.fromisoformat(opening_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)
    shifted = ko.astimezone(timezone.utc) - timedelta(hours=_SHIFT_H)
    return shifted.date().isoformat()


def infer_round(opening_time: Optional[str]) -> Optional[str]:
    """Return the FIFA round key for a kickoff time, or None if it can't be placed
    in the tournament at all (unparseable / before Jun 11). Post-group dates always
    resolve to a knockout round — they are never skipped."""
    d = local_match_date(opening_time)
    if d is None or d < _TOURNAMENT_START:
        return None
    for rnd, lo, hi in _WINDOWS:
        if lo <= d <= hi:
            return rnd
    # Defensive: contiguous windows mean we only reach here for a date past the final
    # window tail — treat as a generic knockout rather than skip.
    return "group" if d <= _GROUP_END else "final"
