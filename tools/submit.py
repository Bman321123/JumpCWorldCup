"""End-to-end platform workflow: pull the REAL questions from the bot API, run
the pipeline (model + scraped sharp odds + crowd capture if present), and show
the submission sheet. Nothing is sent without --go.

  python tools/submit.py --match "USA vs PAR" --home USA --away PAR \
      --date 2026-06-12 --round group                  # dry run (default)
  python tools/submit.py ... --go                      # actually submit
  python tools/submit.py ... --go --update             # revise existing entries

Pipeline -> platform mapping: probabilities are integers 1-99 on the platform;
the policy caps [0.03, 0.97] and the 0.85 player-prop ceiling bind first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging                                              # noqa: E402

from src.crowd_capture import fuzzy_lookup, latest_crowd    # noqa: E402
from src.orchestrator import Orchestrator                   # noqa: E402
from src.platform_client import (PlatformClient,            # noqa: E402
                                 to_platform_probability)
from src.submission_policy import submission                # noqa: E402

EVENT_ID = "aa5572ec-5930-4d99-b06b-f8966333d172"   # Jump Trading Probability Cup
LOBBY_ID = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"   # SportsPredict public lobby


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help='platform name, e.g. "USA vs PAR"')
    ap.add_argument("--home", required=True, help="FIFA code for the model")
    ap.add_argument("--away", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--round", default="group")
    ap.add_argument("--position", default="neutral",
                    choices=["leading", "neutral", "trailing", "desperate"])
    ap.add_argument("--go", action="store_true", help="actually submit")
    ap.add_argument("--update", action="store_true",
                    help="with --go: revise existing predictions instead")
    ap.add_argument("--offline", action="store_true", help="skip odds scrape")
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    client = PlatformClient()
    matches = client.list_matches(lobby_id=LOBBY_ID)
    match = next((m for m in matches
                  if m.get("name", "").lower() == args.match.lower()), None)
    if match is None:
        names = [m.get("name") for m in matches]
        sys.exit(f"Match {args.match!r} not found. Open matches: {names}")
    markets = client.list_markets(LOBBY_ID, match["id"])
    questions = [m.get("question") or m.get("title") for m in markets]
    print(f"{match['name']}  closes {match.get('closing_time')}  "
          f"{len(markets)} open markets\n")

    orch = Orchestrator(
        config_dir=str(ROOT / "config"),
        params_path=str(ROOT / "params" / "dixon_coles.json"),
        db_path=args.db if Path(args.db).exists() else None,
        player_shares_path=str(ROOT / "config" / "player_shares.json"),
        online=not args.offline)
    manifest = orch.predict_match(args.home, args.away, args.date, questions,
                                  args.round)

    crowd = latest_crowd(args.db) if Path(args.db).exists() else {}
    rows = []
    for mkt, pred in zip(markets, manifest["predictions"]):
        hit = fuzzy_lookup(pred["question_text"], crowd)
        crowd_p = hit["crowd_pct"] / 100.0 if hit else None
        final = submission(pred["final_probability"], crowd_p,
                           pred["question_family"], args.position)
        rows.append({"market_id": mkt["id"], "lobby_id": LOBBY_ID,
                     "probability": to_platform_probability(final),
                     "question": pred["question_text"],
                     "pipeline": pred["final_probability"], "crowd": crowd_p,
                     "fallback": pred["source"] == "fallback"})

    print(f"{'SUBMIT':>7} {'pipe':>6} {'crowd':>6}  question")
    print("-" * 96)
    for r in rows:
        crowd_s = f"{r['crowd']:.2f}" if r["crowd"] is not None else "  -  "
        flag = "  << FALLBACK — review!" if r["fallback"] else ""
        print(f"{r['probability']:>6d}% {r['pipeline']:>6.3f} {crowd_s:>6}  "
              f"{r['question']}{flag}")

    if not args.go:
        print("\nDRY RUN — nothing submitted. Re-run with --go to submit.")
        return

    if args.update:
        existing = {p.get("market_id"): p for p in client.list_predictions(LOBBY_ID)
                    if isinstance(p, dict)}
        done = 0
        for r in rows:
            prev = existing.get(r["market_id"])
            if prev and prev.get("probability") != r["probability"]:
                client.update_prediction(prev["id"], r["probability"])
                done += 1
        print(f"\nUpdated {done} prediction(s).")
    else:
        payload = [{"market_id": r["market_id"], "lobby_id": r["lobby_id"],
                    "probability": r["probability"]} for r in rows]
        result = client.submit_batch(payload)
        print("\nSUBMITTED:", json.dumps(result, indent=1)[:800])


if __name__ == "__main__":
    main()
