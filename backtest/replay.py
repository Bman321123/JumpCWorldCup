"""Backtest harness with strict as-of fits (PRD v2.2 §9).

For each tournament: fit Dixon-Coles only on matches BEFORE its start, predict
the standard goal-family question set per match, score Brier vs the base-rate
baseline. Families must beat their baseline to earn deviation rights.

Usage: python backtest/replay.py [--tournaments WC2022,EURO2024]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parameter_estimator import fit_dixon_coles            # noqa: E402
from src.stats_engine import StatsEngine                       # noqa: E402
from src.types import Condition, TemporalWindow                # noqa: E402

TOURNAMENTS = {
    "WC2018": {"like": "FIFA World Cup", "start": "2018-06-14", "end": "2018-07-15"},
    "WC2022": {"like": "FIFA World Cup", "start": "2022-11-20", "end": "2022-12-18"},
    "EURO2024": {"like": "UEFA Euro", "start": "2024-06-14", "end": "2024-07-14"},
    "COPA2024": {"like": "Copa Am", "start": "2024-06-20", "end": "2024-07-15"},
}
BASE_RATES = {"over25": 0.53, "btts": 0.47, "home_win": 0.41, "draw": 0.25}


def replay(db: str, name: str, spec: dict, half_life: float = 500.0,
           maxiter: int = 200) -> dict:
    con = sqlite3.connect(db)
    hist = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score, neutral, "
        "tournament FROM matches WHERE date >= ?", con,
        params=((pd.Timestamp(spec["start"]) - pd.Timedelta(days=2920)).date().isoformat(),))
    matches = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score FROM matches "
        "WHERE tournament LIKE ? AND date BETWEEN ? AND ? "
        "AND tournament NOT LIKE '%qualification%' ORDER BY date", con,
        params=(f"%{spec['like']}%", spec["start"], spec["end"]))
    con.close()
    if matches.empty:
        return {"error": f"no matches found for {name}"}

    params = fit_dixon_coles(hist, half_life_days=half_life, asof=spec["start"],
                             maxiter=maxiter)
    engine = StatsEngine(params)

    rows = []
    for m in matches.itertuples(index=False):
        if m.home_team not in params.attack or m.away_team not in params.attack:
            continue
        total = m.home_score + m.away_score
        r = engine.result_probs(m.home_team, m.away_team)
        p_over = engine.goal_market(m.home_team, m.away_team, "GOALS", "MATCH",
                                    2.5, Condition.GTE, TemporalWindow.FULL)
        p_btts = engine.goal_market(m.home_team, m.away_team, "BTTS", "MATCH",
                                    1.0, Condition.BINARY_YES, TemporalWindow.FULL)
        rows.append({
            "p_home": r["home_win"], "o_home": int(m.home_score > m.away_score),
            "p_draw": r["draw"], "o_draw": int(m.home_score == m.away_score),
            "p_over": p_over, "o_over": int(total > 2.5),
            "p_btts": p_btts, "o_btts": int(m.home_score > 0 and m.away_score > 0),
        })
    d = pd.DataFrame(rows)

    def brier(p, o):
        return float(np.mean((d[p] - d[o]) ** 2))

    def base(rate, o):
        return float(np.mean((rate - d[o]) ** 2))

    return {
        "tournament": name, "n_matches": len(d),
        "model": {"home_win": round(brier("p_home", "o_home"), 5),
                  "draw": round(brier("p_draw", "o_draw"), 5),
                  "over25": round(brier("p_over", "o_over"), 5),
                  "btts": round(brier("p_btts", "o_btts"), 5)},
        "base_rate": {"home_win": round(base(BASE_RATES["home_win"], "o_home"), 5),
                      "draw": round(base(BASE_RATES["draw"], "o_draw"), 5),
                      "over25": round(base(BASE_RATES["over25"], "o_over"), 5),
                      "btts": round(base(BASE_RATES["btts"], "o_btts"), 5)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--tournaments", default="WC2022,EURO2024")
    ap.add_argument("--out", default=str(ROOT / "output" / "backtest_report.json"))
    args = ap.parse_args()

    reports = []
    for name in args.tournaments.split(","):
        name = name.strip()
        if name not in TOURNAMENTS:
            print(f"Unknown tournament {name}; options: {list(TOURNAMENTS)}")
            continue
        print(f"=== {name}: fitting as-of {TOURNAMENTS[name]['start']} ===")
        rep = replay(args.db, name, TOURNAMENTS[name])
        reports.append(rep)
        if "error" in rep:
            print(rep["error"])
            continue
        print(f"{name} ({rep['n_matches']} matches)   model | base-rate")
        for k in ("home_win", "draw", "over25", "btts"):
            mark = "BEATS" if rep["model"][k] < rep["base_rate"][k] else "LOSES"
            print(f"  {k:9s} {rep['model'][k]:.5f} | {rep['base_rate'][k]:.5f}   {mark}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(reports, f, indent=1)
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
