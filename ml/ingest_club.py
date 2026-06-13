"""Download football-data.co.uk club CSVs into one normalized corpus (ROADMAP C).

Club football is where corners/cards/shots labels exist at volume WITH closing
odds. The physics transfers to internationals; base-rate differences are handled
later by internationals-only recalibration. ~5 leagues x 8 seasons ~= 15k matches.

  python ml/ingest_club.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "club_matches.csv"

LEAGUES = ["E0", "E1", "D1", "I1", "SP1", "F1", "N1", "P1", "SC0", "T1"]
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
BASE = "https://www.football-data.co.uk/mmz4281"

KEEP = {"Date": "date", "HomeTeam": "home", "AwayTeam": "away",
        "FTHG": "fthg", "FTAG": "ftag", "HC": "hc", "AC": "ac",
        "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
        "HST": "hst", "AST": "ast", "HF": "hf", "AF": "af",
        "B365H": "oh", "B365D": "od", "B365A": "oa"}


def main() -> None:
    frames = []
    for season in SEASONS:
        for lg in LEAGUES:
            url = f"{BASE}/{season}/{lg}.csv"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code != 200 or not r.content:
                    continue
                df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1",
                                 on_bad_lines="skip")
                have = {k: v for k, v in KEEP.items() if k in df.columns}
                if "HC" not in df.columns:
                    continue                      # no corners -> useless here
                sub = df[list(have)].rename(columns=have)
                sub["league"] = lg
                sub["season"] = season
                frames.append(sub)
            except Exception as e:                # noqa: BLE001
                print(f"  {season}/{lg}: {e}")
    if not frames:
        sys.exit("No club data downloaded.")
    allm = pd.concat(frames, ignore_index=True)
    allm = allm.dropna(subset=["fthg", "ftag", "hc", "ac"])
    allm["date"] = pd.to_datetime(allm["date"], dayfirst=True, errors="coerce")
    allm = allm.dropna(subset=["date"]).sort_values("date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allm.to_csv(OUT, index=False)
    print(f"club corpus: {len(allm)} matches, "
          f"{allm['date'].dt.year.min()}-{allm['date'].dt.year.max()} -> {OUT}")


if __name__ == "__main__":
    main()
