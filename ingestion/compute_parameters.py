"""Fit Dixon-Coles parameters from the matches table -> params/dixon_coles.json.

Usage: python ingestion/compute_parameters.py [--since 2019-01-01] [--half-life 500]
Run nightly during the tournament (PRD v2.2 §6.4). Use --asof for backtest fits.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parameter_estimator import elo_to_priors, fit_dixon_coles  # noqa: E402


def load_matches(db: str, since: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score, neutral, "
        "tournament FROM matches WHERE date >= ?", con, params=(since,))
    con.close()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--half-life", type=float, default=500.0)
    ap.add_argument("--ridge", type=float, default=5.0)
    ap.add_argument("--asof", default=None, help="data cutoff (backtest discipline)")
    ap.add_argument("--elo", default=None, help="optional params/elo.json for priors")
    ap.add_argument("--out", default=str(ROOT / "params" / "dixon_coles.json"))
    ap.add_argument("--maxiter", type=int, default=300)
    args = ap.parse_args()

    df = load_matches(args.db, args.since)
    print(f"Fitting Dixon-Coles on {len(df)} matches since {args.since} ...")

    priors = {"attack": {}, "defense": {}}
    if args.elo and Path(args.elo).exists():
        with open(args.elo) as f:
            priors = elo_to_priors(json.load(f))
        print(f"Elo priors loaded for {len(priors['attack'])} teams")

    params = fit_dixon_coles(df, half_life_days=args.half_life, ridge=args.ridge,
                             elo_attack_prior=priors["attack"],
                             elo_defense_prior=priors["defense"],
                             asof=args.asof, maxiter=args.maxiter)

    # Engine lookups are keyed by FIFA code (MEX), the dataset by name (Mexico):
    # add code-keyed aliases for the 48 WC teams from config/groups.json.
    with open(ROOT / "config" / "groups.json") as f:
        teams_cfg = json.load(f)["teams"]
    name_lookup = {}
    for code, t in teams_cfg.items():
        for cand in [t["name"]] + t.get("aliases", []):
            name_lookup[cand.lower()] = code
    mapped = 0
    for ds_name in list(params.attack):
        code = name_lookup.get(ds_name.lower())
        if code:
            params.attack[code] = params.attack[ds_name]
            params.defense[code] = params.defense[ds_name]
            mapped += 1
    print(f"FIFA-code aliases added for {mapped}/48 WC teams")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    params.save(args.out)
    print(f"Saved {args.out}: mu={params.mu:.3f} gamma={params.gamma:.3f} "
          f"rho={params.rho:.3f}, {len(params.attack)} teams, "
          f"{params.n_matches_fit} matches")
    for t in ("Argentina", "France", "Brazil", "Curaçao", "Jordan"):
        if t in params.attack:
            print(f"  {t:12s} attack={params.attack[t]:+.3f} "
                  f"defense={params.defense[t]:+.3f}")


if __name__ == "__main__":
    main()
