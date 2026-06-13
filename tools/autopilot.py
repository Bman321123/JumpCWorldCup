"""Autopilot — score every open market, show what the bot WOULD auto-submit,
and (only if armed + --go) actually submit the eligible ones.

  python tools/autopilot.py                       # dry run, all matches closing soon
  python tools/autopilot.py --match "QAT vs SUI"  # one match
  python tools/autopilot.py --go                  # submit eligible — refuses if disarmed

Safety: nothing is sent unless config/auto_trade.json has "armed": true AND you
pass --go. Even then, only questions clearing the confidence bar are submitted;
the rest are listed for you to handle by hand.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auto_trader import load_criteria, plan_submissions       # noqa: E402
from src.crowd_capture import fuzzy_lookup, latest_crowd          # noqa: E402
from src.orchestrator import Orchestrator                         # noqa: E402
from src.platform_client import PlatformClient, to_platform_probability  # noqa: E402
from src.submission_policy import submission                      # noqa: E402

LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
CODE = {}    # platform name token -> FIFA code, filled from groups.json


def load_codes():
    with open(ROOT / "config" / "groups.json") as f:
        teams = json.load(f)["teams"]
    idx = {}
    for code, t in teams.items():
        idx[code.lower()] = code
        idx[t["name"].lower()] = code
        for a in t.get("aliases", []):
            idx[a.lower()] = code
    return idx


def resolve(name: str, idx: dict) -> str:
    return idx.get(name.strip().lower(), name.strip().upper()[:3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", help="platform match name; default = all open")
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--round", default="group")
    ap.add_argument("--position", default="neutral")
    ap.add_argument("--go", action="store_true", help="submit eligible (needs armed)")
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    crit = load_criteria(str(ROOT / "config" / "auto_trade.json"))
    idx = load_codes()
    client = PlatformClient()
    matches = client.list_matches(lobby_id=LOBBY_ID)
    if args.match:
        matches = [m for m in matches if m.get("name", "").lower() == args.match.lower()]
    if not matches:
        sys.exit("No matching open matches.")

    crowd = latest_crowd(args.db) if Path(args.db).exists() else {}
    orch = Orchestrator(config_dir=str(ROOT / "config"),
                        params_path=str(ROOT / "params" / "dixon_coles.json"),
                        db_path=args.db if Path(args.db).exists() else None,
                        player_shares_path=str(ROOT / "config" / "player_shares.json"),
                        online=True)

    print(f"AUTOPILOT  armed={crit['armed']}  min_confidence={crit['min_confidence']}  "
          f"{'LIVE --go' if args.go else 'DRY RUN'}\n")
    grand_eligible = []
    for m in matches:
        parts = m["name"].replace(" vs ", " | ").split(" | ")
        if len(parts) != 2:
            continue
        home, away = resolve(parts[0], idx), resolve(parts[1], idx)
        markets = client.list_markets(LOBBY_ID, m["id"])
        questions = [mk.get("question") or mk.get("title") for mk in markets]
        manifest = orch.predict_match(home, away, args.date, questions, args.round)

        submit_values = {}
        for mk, pred in zip(markets, manifest["predictions"]):
            pred["market_id"] = mk["id"]
            hit = fuzzy_lookup(pred["question_text"], crowd)
            crowd_p = hit["crowd_pct"] / 100.0 if hit else None
            f = submission(pred["final_probability"], crowd_p,
                           pred["question_family"], args.position)
            submit_values[pred["question_id"]] = to_platform_probability(f)

        decisions = plan_submissions(manifest, submit_values, crit)
        mid_by_q = {p["question_id"]: p["market_id"] for p in manifest["predictions"]}
        print(f"=== {m['name']}  closes {m.get('closing_time')} ===")
        for d in decisions:
            mark = "AUTO" if d.auto_eligible else "hand"
            print(f"  [{mark}] {d.submit_value:>3d}%  conf={d.confidence:.2f}  "
                  f"{d.question}")
            if d.auto_eligible:
                grand_eligible.append({"market_id": mid_by_q[d.question_id],
                                       "lobby_id": LOBBY_ID,
                                       "probability": d.submit_value,
                                       "question": d.question})

    print(f"\n{len(grand_eligible)} question(s) clear the auto bar.")
    if not args.go:
        print("DRY RUN — nothing submitted.")
        return
    if not crit["armed"]:
        print("REFUSING: config armed=false. Set it true yourself to enable autopilot.")
        return
    payload = [{k: e[k] for k in ("market_id", "lobby_id", "probability")}
               for e in grand_eligible]
    result = client.submit_batch(payload)
    log = ROOT / "data" / "auto_trade_log"
    log.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (log / f"{stamp}.json").write_text(json.dumps(
        {"submitted": grand_eligible, "result": result}, indent=1))
    print(f"SUBMITTED {len(payload)} prediction(s); logged to {log}/{stamp}.json")


if __name__ == "__main__":
    main()
