"""Player layer (PRD v2.2 §4.9, §6.10): availability deltas, suspension
tracking, optional player props. Anchored to the team model — the team lambda
already contains the stars; this layer only handles deviations and props.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

REPLACEMENT_DISCOUNT = 0.4      # replacement supplies ~60% of an absent starter
ABSENCE_FLOOR = 0.70            # never cut a team lambda by more than 30%
YELLOWS_FOR_BAN = 2             # accumulated yellows -> one-match ban (verify 2026 rule)


def shrunk_rate(successes: float, n: float, prior: float, prior_n: float = 20.0) -> float:
    """Beta-binomial posterior mean. 'Scored in 8 of 10' with FW prior 0.35
    -> (8 + 7) / 30 = 0.50, not 0.80. Nobody 'always scores'."""
    return (successes + prior * prior_n) / (n + prior_n)


def anytime_scorer_prob(lam_team: float, involvement_share: float,
                        expected_minutes: float = 90.0) -> float:
    lam_p = lam_team * involvement_share * (expected_minutes / 90.0)
    return 1.0 - math.exp(-lam_p)


class PlayerShares:
    """player -> {team, share, expected_minutes}; built from FBref xG+xA shares."""

    def __init__(self, path: Optional[str] = None):
        self.players: Dict[str, dict] = {}
        if path:
            try:
                with open(path) as f:
                    self.players = json.load(f).get("players", {})
            except FileNotFoundError:
                logger.warning("No player shares at %s; availability deltas inert.", path)

    def availability_multiplier(self, team: str, absent: Iterable[str]) -> float:
        mult = 1.0
        for name in absent:
            p = self.players.get(name)
            if not p or p.get("team") != team:
                logger.warning("Unknown/foreign absent player %r for %s; ignored.",
                               name, team)
                continue
            mult -= float(p.get("share", 0.0)) * REPLACEMENT_DISCOUNT
        return max(mult, ABSENCE_FLOOR)


class SuspensionTracker:
    """Derives bans from our own cards records — knowable the moment the second
    yellow is shown, often before micro-markets reprice."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def banned_players(self, team: str, as_of_round: str = "group") -> List[str]:
        try:
            con = sqlite3.connect(self.db_path)
            rows = con.execute(
                """SELECT player, SUM(yellows) AS y, SUM(reds) AS r
                   FROM player_cards WHERE team = ? AND wiped = 0
                   GROUP BY player""", (team,)).fetchall()
            con.close()
        except sqlite3.Error as e:
            logger.warning("Suspension query failed: %s", e)
            return []
        banned = [p for p, y, r in rows
                  if (y or 0) >= YELLOWS_FOR_BAN or (r or 0) >= 1]
        return banned

    def record_cards(self, player: str, team: str, match_id: str,
                     yellows: int = 0, reds: int = 0) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO player_cards (player, team, match_id, yellows, reds, wiped) "
            "VALUES (?,?,?,?,?,0)", (player, team, match_id, yellows, reds))
        con.commit()
        con.close()

    def wipe_after_quarterfinals(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("UPDATE player_cards SET wiped = 1 WHERE yellows > 0 AND reds = 0")
        con.commit()
        con.close()
