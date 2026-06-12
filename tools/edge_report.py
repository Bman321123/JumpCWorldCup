"""Edge report — the per-match submission sheet (ROADMAP 3.1).

Joins three sources for one match and prints what to actually enter:
  - captured platform questions + crowd numbers (crowd_capture table, via the
    browser extension) — the question texts themselves come from the platform,
    so nothing is typed by hand
  - our pipeline probability (model + scraped sharp market blend)
  - the submission policy (crowd-relative shrink/extremize per family)

Usage:
  python tools/edge_report.py --home CZE --away KOR --date 2026-06-17
  python tools/edge_report.py ... --position trailing      # late-tournament
  python tools/edge_report.py ... --questions-file qs.txt  # no capture yet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging                                            # noqa: E402

from src.crowd_capture import fuzzy_lookup, latest_crowd  # noqa: E402
from src.orchestrator import Orchestrator                 # noqa: E402
from src.submission_policy import submission              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--round", default="group")
    ap.add_argument("--position", default="neutral",
                    choices=["leading", "neutral", "trailing", "desperate"])
    ap.add_argument("--questions-file", help="fallback if no capture exists")
    ap.add_argument("--hours", type=float, default=36.0,
                    help="how far back to trust captured crowd numbers")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    crowd = latest_crowd(args.db, hours=args.hours) if Path(args.db).exists() else {}
    if args.questions_file:
        questions = [q.strip() for q in
                     Path(args.questions_file).read_text().splitlines() if q.strip()]
    elif crowd:
        questions = [row["question_text"] for row in crowd.values()]
    else:
        ap.error("No captured questions in the last "
                 f"{args.hours:.0f}h and no --questions-file. Run the extension "
                 "+ tools/crowd_server.py on the match page first.")

    orch = Orchestrator(
        config_dir=str(ROOT / "config"),
        params_path=str(ROOT / "params" / "dixon_coles.json"),
        db_path=args.db if Path(args.db).exists() else None,
        player_shares_path=str(ROOT / "config" / "player_shares.json"),
        online=not args.offline)
    manifest = orch.predict_match(args.home, args.away, args.date, questions,
                                  args.round)

    rows = []
    for p in manifest["predictions"]:
        hit = fuzzy_lookup(p["question_text"], crowd)
        crowd_p = hit["crowd_pct"] / 100.0 if hit else None
        submit = submission(p["final_probability"], crowd_p,
                            p["question_family"], args.position)
        rows.append({
            "question": p["question_text"], "family": p["question_family"],
            "model": p["model_probability"], "market": p["market_probability"],
            "pipeline": p["final_probability"], "crowd": crowd_p,
            "submit": round(submit, 3),
            "edge": (round(abs(p["final_probability"] - crowd_p), 3)
                     if crowd_p is not None else None),
            "flags": ("AMBIGUOUS-CROWD " if hit and hit.get("ambiguous") else "")
                     + ("FALLBACK " if p["source"] == "fallback" else ""),
        })
    rows.sort(key=lambda r: -(r["edge"] or 0))

    print(f"\nEDGE REPORT  {manifest['match_id']}  round={args.round} "
          f"position={args.position}")
    print(f"{'SUBMIT':>7} {'crowd':>6} {'ours':>6} {'mkt':>6} {'edge':>6}  question")
    print("-" * 100)
    for r in rows:
        crowd_s = f"{r['crowd']:.2f}" if r["crowd"] is not None else "  -  "
        mkt_s = f"{r['market']:.2f}" if r["market"] is not None else "  -  "
        edge_s = f"{r['edge']:.2f}" if r["edge"] is not None else "  -  "
        warn = f"  << {r['flags']}" if r["flags"] else ""
        print(f"{r['submit']:>7.3f} {crowd_s:>6} {r['pipeline']:>6.3f} "
              f"{mkt_s:>6} {edge_s:>6}  {r['question']}{warn}")

    out_dir = ROOT / "output" / "edge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['match_id']}_edge.json"
    with open(out_path, "w") as f:
        json.dump({"manifest": manifest, "edge_rows": rows,
                   "position": args.position}, f, indent=1)
    print(f"\nSaved -> {out_path}")
    if any(r["flags"] for r in rows):
        print("Review flagged rows by eye before submitting.")


if __name__ == "__main__":
    main()
