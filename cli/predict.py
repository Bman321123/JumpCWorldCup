"""Match prediction CLI (PRD v2.2 §6.9).

Examples:
  python cli/predict.py --home MEX --away RSA --date 2026-06-11 --round group \
      --stadium "Estadio Azteca" \
      --questions "Will Mexico win?; Will there be over 2.5 goals?; Will both teams score?; \
Will there be 10 or more corners?; Will there be 4 or more yellow cards?"

  python cli/predict.py --home ARG --away FRA --date 2026-07-19 --round final \
      --questions-file questions.txt --offline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging                                    # noqa: E402

from src.orchestrator import Orchestrator         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="FIFA code, e.g. MEX")
    ap.add_argument("--away", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--round", default="group",
                    choices=["group", "round_of_32", "round_of_16", "quarterfinal",
                             "semifinal", "third_place", "final"])
    ap.add_argument("--questions", help="semicolon-separated question texts")
    ap.add_argument("--questions-file", help="file with one question per line")
    ap.add_argument("--stadium", default=None)
    ap.add_argument("--referee", default=None)
    ap.add_argument("--home-absences", default="", help="comma-separated player names")
    ap.add_argument("--away-absences", default="")
    ap.add_argument("--offline", action="store_true",
                    help="skip odds + weather fetches (pure model)")
    ap.add_argument("--params", default=str(ROOT / "params" / "dixon_coles.json"))
    ap.add_argument("--config", default=str(ROOT / "config"))
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--out", default=str(ROOT / "output" / "predictions"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.questions_file:
        questions = [q.strip() for q in Path(args.questions_file).read_text().splitlines()
                     if q.strip()]
    elif args.questions:
        questions = [q.strip() for q in args.questions.split(";") if q.strip()]
    else:
        ap.error("provide --questions or --questions-file")

    orch = Orchestrator(
        config_dir=args.config, params_path=args.params,
        db_path=args.db if Path(args.db).exists() else None,
        player_shares_path=str(ROOT / "config" / "player_shares.json"),
        online=not args.offline)
    manifest = orch.predict_match(
        args.home, args.away, args.date, questions, args.round,
        stadium=args.stadium, referee_id=args.referee,
        home_absences=[s for s in args.home_absences.split(",") if s],
        away_absences=[s for s in args.away_absences.split(",") if s],
        output_dir=args.out)

    print(f"\n{manifest['match_id']}  weight={manifest['round_weight']}  "
          f"lamH={manifest['model_params']['lambda_home']} "
          f"lamA={manifest['model_params']['lambda_away']}")
    print("-" * 78)
    for p in manifest["predictions"]:
        mkt = f"{p['market_probability']:.3f}" if p["market_probability"] else "  -  "
        print(f"{p['final_probability']:.3f}  (mdl {p['model_probability']:.3f} | "
              f"mkt {mkt} | {p['source']:>12})  {p['question_text']}")
    print(f"\nManifest -> {args.out}/{manifest['match_id']}.json")


if __name__ == "__main__":
    main()
