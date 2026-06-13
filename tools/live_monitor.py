"""Continuous current-events monitor (ROADMAP: live updates).

Each pass, for every open match kicking off within --hours, it refreshes the
hard structured facts (status, official XI -> absences, sharp line snapshot)
plus news headlines, diffs against the last snapshot, and prints what CHANGED.
Run it on a loop near kickoff:

  python tools/live_monitor.py                 # one pass
  /loop 5m python tools/live_monitor.py        # continuous (Claude Code)

What auto-moves numbers downstream: postponement/status, the official lineup
(absences -> availability), weather. News headlines are printed for your
judgment only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live_context import (confirmed_xi, derive_absences,    # noqa: E402
                              match_status, news_headlines)
from src.platform_client import PlatformClient                 # noqa: E402
from src.player_layer import PlayerShares                       # noqa: E402
from src.scrapers.aggregator import ScrapedOddsClient          # noqa: E402

LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
SNAP_DIR = ROOT / "data" / "live_snapshots"
LINE_MOVE = 0.03            # devigged-prob move worth flagging


def _codes():
    with open(ROOT / "config" / "groups.json") as f:
        teams = json.load(f)["teams"]
    idx, names, aliases = {}, {}, {}
    for code, t in teams.items():
        names[code] = t["name"]
        aliases[code] = [t["name"], code] + t.get("aliases", [])
        idx[code.lower()] = code
        idx[t["name"].lower()] = code
        for a in t.get("aliases", []):
            idx[a.lower()] = code
    return idx, names, aliases


def _hours_to(iso: str) -> float:
    try:
        ko = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (ko - datetime.now(timezone.utc)).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return 999.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0,
                    help="only monitor matches kicking off within this window")
    ap.add_argument("--news", action="store_true", help="include news headlines")
    args = ap.parse_args()
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    idx, names, aliases = _codes()
    players = PlayerShares(str(ROOT / "config" / "player_shares.json"))
    odds = ScrapedOddsClient(team_names=names,
                             config_path=str(ROOT / "config" / "scrapers.json"),
                             team_aliases=aliases)
    client = PlatformClient()

    stamp = datetime.now(timezone.utc).strftime("%H:%M")
    print(f"[{stamp}] current-events sweep (next {args.hours:.0f}h)")
    changes = 0
    for m in client.list_matches(lobby_id=LOBBY_ID):
        ko = m.get("closing_time") or m.get("opening_time") or ""
        if _hours_to(ko) > args.hours:
            continue
        parts = m["name"].replace(" vs ", "|").split("|")
        if len(parts) != 2:
            continue
        home = idx.get(parts[0].strip().lower(), parts[0].strip().upper()[:3])
        away = idx.get(parts[1].strip().lower(), parts[1].strip().upper()[:3])
        hn, an = names.get(home, home), names.get(away, away)

        snap = {"status": None, "absences": [], "h2h": None}
        st = match_status(hn, an)
        if st:
            snap["status"] = st.get("detail")
            if st.get("postponed"):
                snap["status"] = "POSTPONED/DELAYED"
        xi = confirmed_xi(hn, an)
        if xi:
            snap["absences"] = sorted(
                derive_absences(home, xi.get("home", []), players.players)
                + derive_absences(away, xi.get("away", []), players.players))
        mk = odds.market_probs(home, away)
        if mk and mk.get("h2h"):
            snap["h2h"] = [round(mk["h2h"][k], 3) for k in ("home", "draw", "away")]

        prev_path = SNAP_DIR / f"{m['name'].replace(' ', '_')}.json"
        prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
        msgs = []
        if snap["status"] and snap["status"] != prev.get("status") \
                and "POSTPONED" in (snap["status"] or ""):
            msgs.append(f"STATUS -> {snap['status']}")
        new_abs = set(snap["absences"]) - set(prev.get("absences", []))
        if new_abs:
            msgs.append(f"LINEUP: key player(s) OUT -> {sorted(new_abs)}")
        if snap["h2h"] and prev.get("h2h"):
            mv = max(abs(a - b) for a, b in zip(snap["h2h"], prev["h2h"]))
            if mv >= LINE_MOVE:
                msgs.append(f"LINE MOVE {mv:.3f}: {prev['h2h']} -> {snap['h2h']}")
        if msgs:
            changes += 1
            print(f"  ** {m['name']} (T-{_hours_to(ko):.1f}h)")
            for msg in msgs:
                print(f"       {msg}")
        prev_path.write_text(json.dumps(snap))
        if args.news and (new_abs or not prev):
            for h in news_headlines(hn, an, 3):
                print(f"       news: {h}")
    print(f"  {changes} change(s) this sweep." if changes
          else "  no changes.")


if __name__ == "__main__":
    main()
