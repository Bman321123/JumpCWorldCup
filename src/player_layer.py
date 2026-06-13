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
SOT_LEAGUE_AVG = 4.3                     # team SOT/match baseline (for opp. normalization)
SOT_PER_SHOT = 0.38                      # ~38% of shots are on target (shots->SOT bridge)
MARKET_BLEND = 0.55                      # weight on the FanDuel-shots-derived estimate


def _poisson_ge(lam: float, k: int) -> float:
    cdf, term = 0.0, math.exp(-lam)
    for i in range(max(k, 0)):
        cdf += term
        term *= lam / (i + 1)
    return 1.0 - cdf


def player_prop_prob(name: str, metric: str, threshold: float,
                     lam_home: float, lam_away: float,
                     home: str, away: str,
                     shares: "PlayerShares",
                     opp_sot_against: Optional[float] = None,
                     fd_shots: Optional[dict] = None) -> tuple[float, str]:
    """Full-picture player prop: the player's own rate x the opponent's
    defensive quality x expected opportunity, blended with the FanDuel player
    SHOTS market (shots -> SOT bridge) when available. Returns (prob, note).

    This is the human reasoning made explicit: how good is the player, how many
    chances do they get, and how leaky is THIS opponent.
    """
    info = shares.players.get(name)
    on_team = info.get("team") if info else None
    if info and on_team in (home, away):
        lam_team = lam_home if on_team == home else lam_away
        pos = info.get("position", "MF")
        apps = float(info.get("apps", 0))
        K = 5.0
        prior_share = DEFAULT_SHARE_BY_POS.get(pos, 0.12)
        prior_sot90 = DEFAULT_SOT90_BY_POS.get(pos, 0.7)
        share = (float(info.get("share", prior_share)) * apps + prior_share * K) / (apps + K)
        sot90 = (float(info.get("sot90", prior_sot90)) * apps + prior_sot90 * K) / (apps + K)
        minutes = float(info.get("expected_minutes", 90.0))
        note = f"{name} share={share:.2f} sot90={sot90:.2f} apps={int(apps)}"
    else:
        lam_team = max(lam_home, lam_away)
        share, sot90, minutes, on_team = UNKNOWN_PLAYER_SHARE, UNKNOWN_PLAYER_SOT90, 90.0, None
        note = f"UNKNOWN PLAYER {name} — priors, REVIEW"
        logger.warning("Player prop for unknown player %r; priors used.", name)

    if metric == "PLAYER_GOAL":
        p = anytime_scorer_prob(lam_team, share, minutes)
        return min(max(p, PLAYER_PROP_FLOOR), PLAYER_PROP_CAP), note

    if metric != "PLAYER_SOT":
        raise ValueError(f"Unknown player metric {metric}")

    # --- model estimate: player rate x opponent defense x opportunity ---
    opp_factor = 1.0
    if opp_sot_against:
        opp_factor = max(0.6, min(opp_sot_against / SOT_LEAGUE_AVG, 1.6))
        note += f" opp_def={opp_factor:.2f}"
    lam_sot = sot90 * opp_factor * (minutes / 90.0)

    # --- market signal: FanDuel player SHOTS -> SOT (real signal even with no
    #     SOT market; bridges the "no counterpart" gap the user described) ---
    if fd_shots:
        p1 = fd_shots.get("1+_FULL")
        if p1 and 0 < p1 < 0.999:
            lam_shots = -math.log(1.0 - p1)              # implied shots rate
            lam_sot_mkt = lam_shots * SOT_PER_SHOT
            lam_sot = (1 - MARKET_BLEND) * lam_sot + MARKET_BLEND * lam_sot_mkt
            note += f" +FD_shots(p1={p1:.2f})"

    p = _poisson_ge(lam_sot, max(int(math.ceil(threshold)), 1))
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
