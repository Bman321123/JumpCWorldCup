"""Download the martj42 international results dataset and load the matches table.

Usage: python ingestion/ingest_historical.py [--db data/wc_forecasting.db]
football-data.co.uk is club-only (PRD v2.2 F3); this dataset is the backbone for
the Dixon-Coles fit. Corners/cards/offsides come from FBref (ingest_fbref.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.db import init_db  # noqa: E402

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/international_results/"
               "master/results.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "wc_forecasting.db"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "results.csv"

    print(f"Downloading {RESULTS_URL} ...")
    df = pd.read_csv(RESULTS_URL)
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} rows to {raw_path}")

    df = df.dropna(subset=["home_score", "away_score"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["match_id"] = (df["date"].astype(str) + "_" + df["home_team"] + "_"
                      + df["away_team"]).str.replace(" ", "_")
    out = df[["match_id", "date", "home_team", "away_team", "home_score",
              "away_score", "tournament", "city", "country", "neutral"]].copy()
    out = out.drop_duplicates(subset=["match_id"])

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    init_db(args.db)
    import sqlite3
    con = sqlite3.connect(args.db)
    con.execute("DELETE FROM matches")
    out.to_sql("matches", con, if_exists="append", index=False)
    n = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    con.commit()
    con.close()
    print(f"matches table: {n} rows in {args.db}")


if __name__ == "__main__":
    main()
