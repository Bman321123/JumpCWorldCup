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


# Overconfidence caps for player props. Live lesson (2026-06-11): a 95% on
# "Schick >= 1 SOT" missed and cost -42 RBP — even elite strikers blank ~25-30%
# of matches. No player prop leaves this module above the cap.
PLAYER_PROP_CAP = 0.85
PLAYER_PROP_FLOOR = 0.03
DEFAULT_SHARE_BY_POS = {"FW": 0.25, "MF": 0.12, "DF": 0.04, "GK": 0.005}
DEFAULT_SOT90_BY_POS = {"FW": 1.3, "MF": 0.7, "DF": 0.25, "GK": 0.01}
UNKNOWN_PLAYER_SHARE = 0.18              # assume attacking player if platform asks
UNKNOWN_PLAYER_SOT90 = 1.1


def player_prop_prob(name: str, metric: str, threshold: float,
                     lam_home: float, lam_away: float,
                     home: str, away: str,
                     shares: "PlayerShares") -> tuple[float, str]:
    """Returns (probability, audit note). Works with or without a shares entry;
    unknown players get attacking-position priors and a review flag."""
    info = shares.players.get(name)
    if info and info.get("team") in (home, away):
        lam_team = lam_home if info["team"] == home else lam_away
        share = float(info.get("share",
                               DEFAULT_SHARE_BY_POS.get(info.get("position", "MF"), 0.12)))
        sot90 = float(info.get("sot90",
                               DEFAULT_SOT90_BY_POS.get(info.get("position", "MF"), 0.7)))
        minutes = float(info.get("expected_minutes", 90.0))
        note = f"shares[{name}] team={info['team']} share={share:.2f}"
    else:
        lam_team = max(lam_home, lam_away)   # conservative: assume the stronger side
        share, sot90, minutes = UNKNOWN_PLAYER_SHARE, UNKNOWN_PLAYER_SOT90, 90.0
        note = f"UNKNOWN PLAYER {name} — priors used, REVIEW"
        logger.warning("Player prop for unknown player %r; using priors.", name)
    if metric == "PLAYER_GOAL":
        p = anytime_scorer_prob(lam_team, share, minutes)
    elif metric == "PLAYER_SOT":
        lam_sot = sot90 * (minutes / 90.0)
        k = max(int(math.ceil(threshold)), 1)
        # P(N >= k), Poisson
        cdf = 0.0
        term = math.exp(-lam_sot)
        for i in range(k):
            cdf += term
            term *= lam_sot / (i + 1)
        p = 1.0 - cdf
    else:
        raise ValueError(f"Unknown player metric {metric}")
    return min(max(p, PLAYER_PROP_FLOOR), PLAYER_PROP_CAP), note


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
