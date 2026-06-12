"""As-of feature store for the ML layer (PRD v2.2 §8.3-8.4).

Builds question-level rows for the GOAL family from the matches table. Every
feature uses only matches strictly BEFORE the row's kickoff (leakage rule).
Feature batch 1 = groups A (team form), C (context), F (question descriptors);
market features (D) and player aggregates (B) join when their sources land.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parameter_estimator import importance_weight  # noqa: E402

THRESHOLDS = [1.5, 2.5, 3.5]
MIN_PRIOR_MATCHES = 10
WINDOWS = (5, 10, 20)


def build(db_path: str, out_csv: str, min_date: str = "2014-01-01") -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT date, home_team, away_team, home_score, away_score, neutral, "
        "tournament FROM matches ORDER BY date", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])

    hist: dict = defaultdict(lambda: deque(maxlen=max(WINDOWS)))
    last_played: dict = {}
    rows = []
    for r in df.itertuples(index=False):
        if r.date >= pd.Timestamp(min_date):
            h_hist, a_hist = list(hist[r.home_team]), list(hist[r.away_team])
            if len(h_hist) >= MIN_PRIOR_MATCHES and len(a_hist) >= MIN_PRIOR_MATCHES:
                feats = {
                    "kickoff": r.date, "home": r.home_team, "away": r.away_team,
                    "neutral": int(r.neutral or 0),
                    "importance": importance_weight(r.tournament),
                    "home_rest_days": _rest(last_played, r.home_team, r.date),
                    "away_rest_days": _rest(last_played, r.away_team, r.date),
                }
                for side, h in (("home", h_hist), ("away", a_hist)):
                    for w in WINDOWS:
                        recent = h[-w:]
                        feats[f"{side}_gf_{w}"] = float(np.mean([x[0] for x in recent]))
                        feats[f"{side}_ga_{w}"] = float(np.mean([x[1] for x in recent]))
                total = r.home_score + r.away_score
                for thr in THRESHOLDS:
                    rows.append({**feats, "metric_btts": 0, "threshold": thr,
                                 "label": int(total > thr)})
                rows.append({**feats, "metric_btts": 1, "threshold": 1.0,
                             "label": int(r.home_score > 0 and r.away_score > 0)})
        hist[r.home_team].append((r.home_score, r.away_score))
        hist[r.away_team].append((r.away_score, r.home_score))
        last_played[r.home_team] = r.date
        last_played[r.away_team] = r.date

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Feature store: {len(out)} question rows -> {out_csv}")
    return out


def _rest(last_played: dict, team: str, date) -> float:
    prev = last_played.get(team)
    return float(min((date - prev).days, 60)) if prev is not None else 60.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--out", default=str(ROOT / "data" / "features_goal.csv"))
    ap.add_argument("--min-date", default="2014-01-01")
    args = ap.parse_args()
    build(args.db, args.out, args.min_date)
