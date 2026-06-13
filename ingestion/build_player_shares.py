"""Build config/player_shares.json from ESPN per-player match data (ROADMAP A).

Involvement share = a player's share of his team's attacking output (shots
weighted up for goals/assists) across all matches in the corpus. Plus SOT/90
and expected minutes, for the anytime-scorer and player-SOT prop models.

Only players with >= MIN_APPS appearances get an entry; everyone else falls
back to the position priors already in player_layer.py. The 0.85 prop cap
still binds on top of all of this.

  python ingestion/build_player_shares.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATS_DIR = ROOT / "data" / "match_stats"
MIN_APPS = 2
SHARE_CAP = 0.45                 # nobody is 'always' the whole attack
# attacking-contribution weights: a goal/assist signals more threat than a shot
W_SHOT, W_GOAL, W_ASSIST = 1.0, 2.0, 1.0


def _code_index() -> dict:
    with open(ROOT / "config" / "groups.json") as f:
        teams = json.load(f)["teams"]
    idx = {}
    for code, t in teams.items():
        idx[t["name"].lower()] = code
        idx[code.lower()] = code
        for a in t.get("aliases", []):
            idx[a.lower()] = code
    return idx


def _contribution(p: dict) -> float:
    return (W_SHOT * p.get("totalShots", 0) + W_GOAL * p.get("totalGoals", 0)
            + W_ASSIST * p.get("goalAssists", 0))


def main() -> None:
    idx = _code_index()
    # per player: accumulators; per (team-code) per match: team contribution
    players: dict = defaultdict(lambda: {
        "apps": 0, "starts": 0, "minutes": 0, "sot": 0, "shots": 0,
        "goals": 0, "assists": 0, "contrib": 0.0, "pos": defaultdict(int),
        "team": None})
    team_contrib_by_match: dict = defaultdict(float)

    files = sorted(STATS_DIR.glob("*.json"))
    for f in files:
        data = json.loads(f.read_text())
        rows = data.get("players") or []
        match_id = f.stem
        # first pass: team contribution totals for this match
        for p in rows:
            code = idx.get((p.get("team") or "").lower())
            if code:
                team_contrib_by_match[(match_id, code)] += _contribution(p)
        # second pass: accumulate per player
        for p in rows:
            code = idx.get((p.get("team") or "").lower())
            if not code:
                continue
            key = (p["name"], code)
            acc = players[key]
            acc["team"] = code
            acc["apps"] += 1
            acc["starts"] += 1 if p.get("starter") else 0
            acc["minutes"] += p.get("minutes", 0)
            acc["sot"] += p.get("shotsOnTarget", 0)
            acc["shots"] += p.get("totalShots", 0)
            acc["goals"] += p.get("totalGoals", 0)
            acc["assists"] += p.get("goalAssists", 0)
            acc["contrib"] += _contribution(p)
            acc["pos"][p.get("position", "M")] += 1

    # team total contribution across matches (denominator for shares)
    team_total: dict = defaultdict(float)
    for (match_id, code), c in team_contrib_by_match.items():
        team_total[code] += c

    out = {}
    for (name, code), acc in players.items():
        if acc["apps"] < MIN_APPS:
            continue
        denom = team_total.get(code, 0.0)
        share = (acc["contrib"] / denom) if denom > 0 else 0.0
        share = min(share, SHARE_CAP)
        minutes90 = max(acc["minutes"] / 90.0, 0.5)
        sot90 = acc["sot"] / minutes90
        start_rate = acc["starts"] / acc["apps"]
        exp_minutes = 90 if start_rate >= 0.6 else 70 if start_rate >= 0.3 else 40
        position = max(acc["pos"].items(), key=lambda kv: kv[1])[0]
        pos_group = ("FW" if position in ("F", "ST", "CF", "LW", "RW", "FW")
                     else "DF" if position in ("D", "CB", "LB", "RB", "DF")
                     else "GK" if position in ("G", "GK")
                     else "MF")
        out[name] = {"team": code, "position": pos_group,
                     "share": round(share, 4), "sot90": round(sot90, 3),
                     "expected_minutes": exp_minutes, "apps": acc["apps"],
                     "goals": acc["goals"]}

    path = ROOT / "config" / "player_shares.json"
    path.write_text(json.dumps({"_source": "ESPN per-match player stats",
                                "players": out}, indent=1))
    covered = len({v["team"] for v in out.values()})
    print(f"player_shares.json: {len(out)} players across {covered} teams")
    top = sorted(out.items(), key=lambda kv: -kv[1]["share"])[:12]
    for name, v in top:
        print(f"  {name:24s} {v['team']}  share={v['share']:.3f}  "
              f"sot90={v['sot90']:.2f}  apps={v['apps']}  goals={v['goals']}")


if __name__ == "__main__":
    main()
